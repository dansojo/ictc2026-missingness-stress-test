"""Outcome-free cold-start personalization and semantic daily indicators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import PRIMITIVE_DEFINITIONS


MAD_NORMALIZATION = 1.4826
PRIMARY_LAMBDA_DAYS = 7.0
PERSONALIZATION_K_VALUES = (3, 7, 14, 21)
EVALUATION_DAY_START = 22

_PRIMITIVE_BY_NAME = {definition.name: definition for definition in PRIMITIVE_DEFINITIONS}
_USAGE_SCREEN_PROXY = {
    "usage_load_24h": "screen_load_24h",
    "usage_disengagement_p90": "screen_disengagement_p90",
}

_SENSOR_METADATA_COLUMNS = (
    "sensor_day_id",
    "subject_id",
    "lifelog_date",
    "sleep_date",
    "collection_start_date",
    "elapsed_day_index",
    "any_source_record_observed",
    "any_measurement_support_observed",
    "any_sensor_observed",
)


@dataclass(frozen=True)
class LosoRepresentationTables:
    """Separate outcome-free Task 3 artifacts."""

    representations: pd.DataFrame
    baselines: pd.DataFrame
    confidence_audit: pd.DataFrame
    placebo_candidates: pd.DataFrame


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a symmetric weighted median for positive finite weights."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    half = float(sorted_weights.sum()) / 2.0
    index = int(np.searchsorted(cumulative, half, side="left"))
    if np.isclose(cumulative[index], half, rtol=0.0, atol=1e-12):
        next_index = index + 1
        if next_index < len(sorted_values):
            return float((sorted_values[index] + sorted_values[next_index]) / 2.0)
    return float(sorted_values[index])


def robust_location_scale(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> dict[str, float | int]:
    """Estimate a quality-weighted median and normalized weighted MAD."""
    value_array = np.asarray(values, dtype=float).reshape(-1)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if value_array.size != weight_array.size:
        raise ValueError("values and weights must have the same length")
    finite_weights = np.isfinite(weight_array)
    if bool(((weight_array[finite_weights] < 0) | (weight_array[finite_weights] > 1)).any()):
        raise ValueError("finite weights must be between zero and one")

    valid = np.isfinite(value_array) & finite_weights & (weight_array > 0)
    if not bool(valid.any()):
        return {
            "center": float("nan"),
            "scale": float("nan"),
            "n_eff": 0.0,
            "contributing_rows": 0,
        }

    valid_values = value_array[valid]
    valid_weights = weight_array[valid]
    center = _weighted_median(valid_values, valid_weights)
    absolute_deviations = np.abs(valid_values - center)
    scale = MAD_NORMALIZATION * _weighted_median(absolute_deviations, valid_weights)
    return {
        "center": float(center),
        "scale": float(scale),
        "n_eff": float(valid_weights.sum()),
        "contributing_rows": int(valid.sum()),
    }


def shrink_location_scale(
    *,
    personal_center: float,
    personal_scale: float,
    population_center: float,
    population_scale: float,
    n_eff: float,
    lambda_days: float = PRIMARY_LAMBDA_DAYS,
) -> dict[str, Any]:
    """Apply the frozen quality-weighted cold-start shrinkage rule."""
    if not np.isfinite(n_eff) or n_eff < 0:
        raise ValueError("n_eff must be finite and nonnegative")
    if not np.isfinite(lambda_days) or lambda_days < 0:
        raise ValueError("lambda_days must be finite and nonnegative")
    if not np.isfinite(population_center) or not np.isfinite(population_scale) or population_scale <= 0:
        return {
            "center": float("nan"),
            "scale": float("nan"),
            "center_weight": float("nan"),
            "scale_weight": float("nan"),
            "scale_floor": float("nan"),
            "status": "degenerate",
        }

    denominator = n_eff + lambda_days
    center_weight = 0.0 if denominator == 0 else float(n_eff / denominator)
    if n_eff > 0:
        if not np.isfinite(personal_center):
            return {
                "center": float("nan"),
                "scale": float("nan"),
                "center_weight": center_weight,
                "scale_weight": 0.0,
                "scale_floor": 0.1 * float(population_scale),
                "status": "insufficient",
            }
        center = (
            center_weight * float(personal_center)
            + (1.0 - center_weight) * float(population_center)
        )
    else:
        center = float(population_center)

    scale_weight = (
        center_weight
        if n_eff >= 3.0 and np.isfinite(personal_scale)
        else 0.0
    )
    candidate_scale = (
        float(population_scale)
        if scale_weight == 0.0
        else (
            scale_weight * float(personal_scale)
            + (1.0 - scale_weight) * float(population_scale)
        )
    )
    scale_floor = 0.1 * float(population_scale)
    scale = max(float(candidate_scale), scale_floor)
    return {
        "center": float(center),
        "scale": float(scale),
        "center_weight": float(center_weight),
        "scale_weight": float(scale_weight),
        "scale_floor": float(scale_floor),
        "status": "observed",
    }


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Missing required primitive metadata column: {column}")
    return pd.to_numeric(frame[column], errors="coerce")


def compute_primitive_confidence(
    primitives: pd.DataFrame,
    *,
    primitive_names: Sequence[str] | None = None,
    usage_policy: str = "screen_proxy",
) -> pd.DataFrame:
    """Build the frozen long-form primitive confidence audit.

    UsageStats event density remains descriptive.  Under the primary policy,
    the corresponding screen stream supplies only its availability metadata.
    """
    if usage_policy not in {"screen_proxy", "unweighted"}:
        raise ValueError("usage_policy must be screen_proxy or unweighted")
    if primitives.columns.duplicated().any():
        raise ValueError("Primitive input columns must be unique")
    for required in ("sensor_day_id", "subject_id"):
        if required not in primitives.columns:
            raise ValueError(f"Missing required key column: {required}")

    names = tuple(_PRIMITIVE_BY_NAME) if primitive_names is None else tuple(primitive_names)
    if len(names) != len(set(names)):
        raise ValueError("primitive_names must be unique")
    unknown = sorted(set(names).difference(_PRIMITIVE_BY_NAME))
    if unknown:
        raise ValueError(f"Unknown primitive names: {unknown}")

    key_columns = [
        column
        for column in (
            "sensor_day_id",
            "subject_id",
            "lifelog_date",
            "sleep_date",
            "collection_start_date",
            "elapsed_day_index",
            "any_source_record_observed",
            "any_measurement_support_observed",
            "any_sensor_observed",
        )
        if column in primitives.columns
    ]
    records: list[pd.DataFrame] = []
    for primitive_name in names:
        definition = _PRIMITIVE_BY_NAME[primitive_name]
        values = _numeric_series(primitives, primitive_name)
        status_column = f"{primitive_name}__status"
        if status_column not in primitives.columns:
            raise ValueError(f"Missing required primitive metadata column: {status_column}")
        primitive_status = primitives[status_column].astype("string")
        validity = (values.notna() & np.isfinite(values) & primitive_status.eq("observed")).astype(float)

        censoring_column = f"{primitive_name}__censoring"
        if censoring_column in primitives.columns:
            censoring = primitives[censoring_column].astype("string")
            censoring_component = censoring.eq("none").fillna(False).astype(float)
        else:
            censoring = pd.Series("none", index=primitives.index, dtype="string")
            censoring_component = pd.Series(1.0, index=primitives.index, dtype=float)

        support_column = f"{primitive_name}__support_density"
        support_density = (
            pd.to_numeric(primitives[support_column], errors="coerce")
            if support_column in primitives.columns
            else pd.Series(np.nan, index=primitives.index, dtype=float)
        )

        proxy_source = primitive_name
        unweighted_usage_policy = (
            primitive_name in _USAGE_SCREEN_PROXY and usage_policy == "unweighted"
        )
        if unweighted_usage_policy:
            coverage_component = pd.Series(1.0, index=primitives.index, dtype=float)
            gap_component = pd.Series(1.0, index=primitives.index, dtype=float)
            proxy_source = "unweighted"
        else:
            availability_name = _USAGE_SCREEN_PROXY.get(primitive_name, primitive_name)
            availability_definition = _PRIMITIVE_BY_NAME[availability_name]
            coverage = _numeric_series(
                primitives, f"{availability_name}__availability_coverage"
            )
            longest_gap = _numeric_series(
                primitives, f"{availability_name}__longest_gap_minutes"
            )
            expected_column = f"{availability_name}__expected_epochs"
            if expected_column in primitives.columns:
                reported_expected = _numeric_series(primitives, expected_column)
                if (
                    reported_expected.isna().any()
                    or not np.isfinite(reported_expected).all()
                    or not reported_expected.eq(
                        float(availability_definition.expected_epochs)
                    ).all()
                ):
                    raise ValueError(
                        f"{expected_column} must match the frozen expected_epochs contract"
                    )
            expected_epochs = pd.Series(
                float(availability_definition.expected_epochs),
                index=primitives.index,
                dtype=float,
            )
            denominator = expected_epochs * float(availability_definition.cadence_minutes)
            coverage_component = coverage.clip(0.0, 1.0).fillna(0.0)
            gap_component = (1.0 - longest_gap / denominator).clip(0.0, 1.0).fillna(0.0)
            proxy_source = availability_name

        if unweighted_usage_policy:
            quality = validity
            coverage_only = validity
            unweighted = validity
        else:
            quality = coverage_component * gap_component * validity * censoring_component
            coverage_only = coverage_component * validity
            unweighted = validity * censoring_component
        value_only = validity
        audit = primitives.loc[:, key_columns].copy()
        audit["primitive"] = primitive_name
        audit["value"] = values.astype(float)
        audit["coverage_component"] = coverage_component.astype(float)
        audit["gap_component"] = gap_component.astype(float)
        audit["validity_component"] = validity.astype(float)
        audit["censoring_component"] = censoring_component.astype(float)
        audit["quality_confidence"] = quality.clip(0.0, 1.0).astype(float)
        audit["coverage_only_confidence"] = coverage_only.clip(0.0, 1.0).astype(float)
        audit["value_only_confidence"] = value_only.astype(float)
        audit["unweighted_confidence"] = unweighted.astype(float)
        audit["proxy_source"] = proxy_source
        audit["support_density"] = support_density.astype(float)
        audit["primitive_status"] = primitive_status
        audit["censoring"] = censoring
        audit["usage_policy"] = usage_policy if primitive_name in _USAGE_SCREEN_PROXY else "not_applicable"
        records.append(audit)

    if not records:
        return pd.DataFrame()
    result = pd.concat(records, ignore_index=True)
    primitive_order = {name: index for index, name in enumerate(names)}
    result["_primitive_order"] = result["primitive"].map(primitive_order)
    result = result.sort_values(
        ["sensor_day_id", "_primitive_order"], kind="stable"
    ).drop(columns="_primitive_order").reset_index(drop=True)
    return result


def _valid_weighted_sources(
    row: pd.Series,
    sources: Sequence[tuple[str, str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    weights: list[float] = []
    for value_column, confidence_column, direction in sources:
        value = pd.to_numeric(pd.Series([row[value_column]]), errors="coerce").iloc[0]
        confidence = pd.to_numeric(
            pd.Series([row[confidence_column]]), errors="coerce"
        ).iloc[0]
        if np.isfinite(confidence) and not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"{confidence_column} must be between zero and one")
        if np.isfinite(value) and np.isfinite(confidence) and float(confidence) > 0:
            values.append(float(direction) * float(value))
            weights.append(float(confidence))
    return np.asarray(values, dtype=float), np.asarray(weights, dtype=float)


def _semantic_summary(
    row: pd.Series,
    sources: Sequence[tuple[str, str, float]],
    *,
    operation: str,
    expected_source_count: int,
    minimum_sources: int = 1,
) -> tuple[float, float, int, str]:
    values, weights = _valid_weighted_sources(row, sources)
    modalities = int(len(values))
    if modalities < minimum_sources:
        return float("nan"), 0.0, modalities, "insufficient"
    if operation == "mean":
        value = float(np.average(values, weights=weights))
    elif operation == "median":
        value = _weighted_median(values, weights)
    elif operation == "rms":
        value = float(np.sqrt(np.average(np.square(values), weights=weights)))
    else:
        raise ValueError(f"Unsupported semantic operation: {operation}")
    confidence = min(float(weights.sum()) / float(expected_source_count), 1.0)
    return value, confidence, modalities, "derived"


def _store_semantic(
    record: dict[str, Any],
    name: str,
    summary: tuple[float, float, int, str],
) -> None:
    value, confidence, modalities, status = summary
    record[name] = value
    record[f"{name}__confidence"] = confidence
    record[f"{name}__modalities"] = modalities
    record[f"{name}__status"] = status


def _compose_semantic_columns_reference(representations: pd.DataFrame) -> pd.DataFrame:
    """Slow independent formula reference used to test vectorized parity."""
    if representations.columns.duplicated().any():
        raise ValueError("Representation input columns must be unique")
    required_primitives = (
        "screen_load_24h",
        "usage_load_24h",
        "screen_disengagement_p90",
        "usage_disengagement_p90",
        "phone_activity_load_24h",
        "step_load_24h",
        "phone_activity_disengagement_p90",
        "step_disengagement_p90",
        "mobile_light_exposure_24h",
        "wearable_light_exposure_24h",
        "mobile_low_light_onset",
        "wearable_low_light_onset",
        "heart_rate_settling_delta",
        "heart_rate_settling_slope",
        "charging_onset",
    )
    required_columns: set[str] = set()
    for primitive in required_primitives:
        required_columns.update(
            (primitive, f"{primitive}__z", f"{primitive}__confidence")
        )
    missing = sorted(required_columns.difference(representations.columns))
    if missing:
        raise ValueError(f"Missing semantic source columns: {missing}")

    output_records: list[dict[str, Any]] = []
    for _, row in representations.iterrows():
        record = row.to_dict()
        digital_engagement_sources = (
            ("screen_load_24h__z", "screen_load_24h__confidence", 1.0),
            ("usage_load_24h__z", "usage_load_24h__confidence", 1.0),
        )
        digital_timing_sources = (
            ("screen_disengagement_p90", "screen_disengagement_p90__confidence", 1.0),
            ("usage_disengagement_p90", "usage_disengagement_p90__confidence", 1.0),
        )
        physical_activity_sources = (
            ("phone_activity_load_24h__z", "phone_activity_load_24h__confidence", 1.0),
            ("step_load_24h__z", "step_load_24h__confidence", 1.0),
        )
        physical_timing_sources = (
            (
                "phone_activity_disengagement_p90",
                "phone_activity_disengagement_p90__confidence",
                1.0,
            ),
            ("step_disengagement_p90", "step_disengagement_p90__confidence", 1.0),
        )
        light_exposure_sources = (
            (
                "mobile_light_exposure_24h__z",
                "mobile_light_exposure_24h__confidence",
                1.0,
            ),
            (
                "wearable_light_exposure_24h__z",
                "wearable_light_exposure_24h__confidence",
                1.0,
            ),
        )
        low_light_sources = (
            ("mobile_low_light_onset", "mobile_low_light_onset__confidence", 1.0),
            ("wearable_low_light_onset", "wearable_low_light_onset__confidence", 1.0),
        )
        physiology_sources = (
            (
                "heart_rate_settling_delta__z",
                "heart_rate_settling_delta__confidence",
                -1.0,
            ),
            (
                "heart_rate_settling_slope__z",
                "heart_rate_settling_slope__confidence",
                -1.0,
            ),
        )

        _store_semantic(
            record,
            "digital_engagement_score",
            _semantic_summary(
                row, digital_engagement_sources, operation="mean", expected_source_count=2
            ),
        )
        _store_semantic(
            record,
            "digital_disengagement_time",
            _semantic_summary(
                row, digital_timing_sources, operation="median", expected_source_count=2
            ),
        )
        _store_semantic(
            record,
            "physical_activity_score",
            _semantic_summary(
                row, physical_activity_sources, operation="mean", expected_source_count=2
            ),
        )
        _store_semantic(
            record,
            "physical_disengagement_time",
            _semantic_summary(
                row, physical_timing_sources, operation="median", expected_source_count=2
            ),
        )
        _store_semantic(
            record,
            "light_exposure_score",
            _semantic_summary(
                row, light_exposure_sources, operation="mean", expected_source_count=2
            ),
        )
        _store_semantic(
            record,
            "low_light_onset",
            _semantic_summary(
                row, low_light_sources, operation="median", expected_source_count=2
            ),
        )
        _store_semantic(
            record,
            "physiological_settling_score",
            _semantic_summary(
                row, physiology_sources, operation="mean", expected_source_count=2
            ),
        )

        enriched = pd.Series(record)
        winddown_sources = (
            (
                "digital_disengagement_time",
                "digital_disengagement_time__confidence",
                1.0,
            ),
            (
                "physical_disengagement_time",
                "physical_disengagement_time__confidence",
                1.0,
            ),
            ("low_light_onset", "low_light_onset__confidence", 1.0),
            ("charging_onset", "charging_onset__confidence", 1.0),
        )
        winddown_center = _semantic_summary(
            enriched, winddown_sources, operation="median", expected_source_count=4
        )
        _store_semantic(record, "winddown_center", winddown_center)

        enriched = pd.Series(record)
        winddown_values, winddown_weights = _valid_weighted_sources(
            enriched, winddown_sources
        )
        if len(winddown_values) < 2:
            spread = (float("nan"), 0.0, int(len(winddown_values)), "insufficient")
        else:
            center = float(record["winddown_center"])
            spread_value = _weighted_median(
                np.abs(winddown_values - center), winddown_weights
            )
            spread = (
                spread_value,
                min(float(winddown_weights.sum()) / 4.0, 1.0),
                int(len(winddown_values)),
                "derived",
            )
        _store_semantic(record, "winddown_spread", spread)

        # Routine deviation intentionally uses eight non-overlapping standardized
        # components, rather than the raw-clock wind-down summaries.
        digital_timing_z = _semantic_summary(
            row,
            tuple(
                (f"{name}__z", f"{name}__confidence", 1.0)
                for name in ("screen_disengagement_p90", "usage_disengagement_p90")
            ),
            operation="mean",
            expected_source_count=2,
        )
        physical_timing_z = _semantic_summary(
            row,
            tuple(
                (f"{name}__z", f"{name}__confidence", 1.0)
                for name in (
                    "phone_activity_disengagement_p90",
                    "step_disengagement_p90",
                )
            ),
            operation="mean",
            expected_source_count=2,
        )
        low_light_z = _semantic_summary(
            row,
            tuple(
                (f"{name}__z", f"{name}__confidence", 1.0)
                for name in ("mobile_low_light_onset", "wearable_low_light_onset")
            ),
            operation="mean",
            expected_source_count=2,
        )
        routine_components = (
            (record["digital_engagement_score"], record["digital_engagement_score__confidence"]),
            (digital_timing_z[0], digital_timing_z[1]),
            (record["physical_activity_score"], record["physical_activity_score__confidence"]),
            (physical_timing_z[0], physical_timing_z[1]),
            (record["light_exposure_score"], record["light_exposure_score__confidence"]),
            (low_light_z[0], low_light_z[1]),
            (
                record["physiological_settling_score"],
                record["physiological_settling_score__confidence"],
            ),
            (row["charging_onset__z"], row["charging_onset__confidence"]),
        )
        routine_values = np.asarray([item[0] for item in routine_components], dtype=float)
        routine_weights = np.asarray([item[1] for item in routine_components], dtype=float)
        valid_routine = (
            np.isfinite(routine_values)
            & np.isfinite(routine_weights)
            & (routine_weights > 0)
        )
        if not bool(valid_routine.any()):
            routine = (float("nan"), 0.0, 0, "insufficient")
        else:
            routine = (
                float(
                    np.sqrt(
                        np.average(
                            np.square(routine_values[valid_routine]),
                            weights=routine_weights[valid_routine],
                        )
                    )
                ),
                min(float(routine_weights[valid_routine].sum()) / 8.0, 1.0),
                int(valid_routine.sum()),
                "derived",
            )
        _store_semantic(record, "routine_deviation", routine)
        output_records.append(record)

    result = pd.DataFrame.from_records(output_records)
    sort_columns = [
        column
        for column in ("sensor_day_id", "held_out_subject", "k", "method", "donor_subject")
        if column in result.columns
    ]
    if sort_columns:
        result = result.sort_values(sort_columns, kind="stable", na_position="first")
    return result.reset_index(drop=True)


def _source_matrices(
    frame: pd.DataFrame,
    sources: Sequence[tuple[str, str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.column_stack(
        [
            pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
            * float(direction)
            for value_column, _, direction in sources
        ]
    )
    weights = np.column_stack(
        [
            pd.to_numeric(frame[confidence_column], errors="coerce").to_numpy(dtype=float)
            for _, confidence_column, _ in sources
        ]
    )
    invalid_finite_weight = np.isfinite(weights) & ((weights < 0) | (weights > 1))
    if bool(invalid_finite_weight.any()):
        raise ValueError("Semantic source confidence must be between zero and one")
    return values, weights


def _summary_arrays(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    operation: str,
    expected_source_count: int,
    minimum_sources: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    effective_weights = np.where(valid, weights, 0.0)
    effective_values = np.where(valid, values, 0.0)
    modalities = valid.sum(axis=1).astype(int)
    sufficient = modalities >= int(minimum_sources)
    weight_sum = effective_weights.sum(axis=1)
    output = np.full(len(values), np.nan, dtype=float)
    if operation == "mean":
        numerator = (effective_values * effective_weights).sum(axis=1)
        np.divide(numerator, weight_sum, out=output, where=sufficient & (weight_sum > 0))
    elif operation == "rms":
        numerator = (np.square(effective_values) * effective_weights).sum(axis=1)
        mean_square = np.full(len(values), np.nan, dtype=float)
        np.divide(
            numerator,
            weight_sum,
            out=mean_square,
            where=sufficient & (weight_sum > 0),
        )
        output[sufficient] = np.sqrt(mean_square[sufficient])
    elif operation == "median":
        for row_index in np.flatnonzero(sufficient):
            row_valid = valid[row_index]
            output[row_index] = _weighted_median(
                values[row_index, row_valid], weights[row_index, row_valid]
            )
    else:
        raise ValueError(f"Unsupported semantic operation: {operation}")
    confidence = np.zeros(len(values), dtype=float)
    confidence[sufficient] = np.minimum(
        weight_sum[sufficient] / float(expected_source_count), 1.0
    )
    status = np.where(sufficient, "derived", "insufficient")
    return output, confidence, modalities, status


def _append_summary_arrays(
    output: dict[str, Any],
    name: str,
    summary: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    value, confidence, modalities, status = summary
    output[name] = value
    output[f"{name}__confidence"] = confidence
    output[f"{name}__modalities"] = modalities
    output[f"{name}__status"] = status


def compose_semantic_columns(representations: pd.DataFrame) -> pd.DataFrame:
    """Append the ten frozen semantic columns with vectorized row summaries."""
    if representations.columns.duplicated().any():
        raise ValueError("Representation input columns must be unique")
    source_primitives = (
        "screen_load_24h",
        "usage_load_24h",
        "screen_disengagement_p90",
        "usage_disengagement_p90",
        "phone_activity_load_24h",
        "step_load_24h",
        "phone_activity_disengagement_p90",
        "step_disengagement_p90",
        "mobile_light_exposure_24h",
        "wearable_light_exposure_24h",
        "mobile_low_light_onset",
        "wearable_low_light_onset",
        "heart_rate_settling_delta",
        "heart_rate_settling_slope",
        "charging_onset",
    )
    required_columns = {
        column
        for primitive in source_primitives
        for column in (primitive, f"{primitive}__z", f"{primitive}__confidence")
    }
    missing = sorted(required_columns.difference(representations.columns))
    if missing:
        raise ValueError(f"Missing semantic source columns: {missing}")
    frame = representations.copy(deep=True)
    semantic: dict[str, Any] = {}

    source_groups = {
        "digital_engagement_score": (
            (
                ("screen_load_24h__z", "screen_load_24h__confidence", 1.0),
                ("usage_load_24h__z", "usage_load_24h__confidence", 1.0),
            ),
            "mean",
        ),
        "digital_disengagement_time": (
            (
                ("screen_disengagement_p90", "screen_disengagement_p90__confidence", 1.0),
                ("usage_disengagement_p90", "usage_disengagement_p90__confidence", 1.0),
            ),
            "median",
        ),
        "physical_activity_score": (
            (
                ("phone_activity_load_24h__z", "phone_activity_load_24h__confidence", 1.0),
                ("step_load_24h__z", "step_load_24h__confidence", 1.0),
            ),
            "mean",
        ),
        "physical_disengagement_time": (
            (
                (
                    "phone_activity_disengagement_p90",
                    "phone_activity_disengagement_p90__confidence",
                    1.0,
                ),
                ("step_disengagement_p90", "step_disengagement_p90__confidence", 1.0),
            ),
            "median",
        ),
        "light_exposure_score": (
            (
                (
                    "mobile_light_exposure_24h__z",
                    "mobile_light_exposure_24h__confidence",
                    1.0,
                ),
                (
                    "wearable_light_exposure_24h__z",
                    "wearable_light_exposure_24h__confidence",
                    1.0,
                ),
            ),
            "mean",
        ),
        "low_light_onset": (
            (
                ("mobile_low_light_onset", "mobile_low_light_onset__confidence", 1.0),
                ("wearable_low_light_onset", "wearable_low_light_onset__confidence", 1.0),
            ),
            "median",
        ),
        "physiological_settling_score": (
            (
                (
                    "heart_rate_settling_delta__z",
                    "heart_rate_settling_delta__confidence",
                    -1.0,
                ),
                (
                    "heart_rate_settling_slope__z",
                    "heart_rate_settling_slope__confidence",
                    -1.0,
                ),
            ),
            "mean",
        ),
    }
    for name, (sources, operation) in source_groups.items():
        values, weights = _source_matrices(frame, sources)
        summary = _summary_arrays(
            values,
            weights,
            operation=operation,
            expected_source_count=2,
        )
        _append_summary_arrays(semantic, name, summary)

    winddown_values = np.column_stack(
        [
            semantic["digital_disengagement_time"],
            semantic["physical_disengagement_time"],
            semantic["low_light_onset"],
            pd.to_numeric(frame["charging_onset"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    winddown_weights = np.column_stack(
        [
            semantic["digital_disengagement_time__confidence"],
            semantic["physical_disengagement_time__confidence"],
            semantic["low_light_onset__confidence"],
            pd.to_numeric(
                frame["charging_onset__confidence"], errors="coerce"
            ).to_numpy(dtype=float),
        ]
    )
    winddown_center = _summary_arrays(
        winddown_values,
        winddown_weights,
        operation="median",
        expected_source_count=4,
    )
    _append_summary_arrays(semantic, "winddown_center", winddown_center)
    valid_winddown = (
        np.isfinite(winddown_values)
        & np.isfinite(winddown_weights)
        & (winddown_weights > 0)
    )
    spread_values = np.abs(winddown_values - winddown_center[0][:, None])
    spread = _summary_arrays(
        spread_values,
        winddown_weights,
        operation="median",
        expected_source_count=4,
        minimum_sources=2,
    )
    # Preserve the count of available timing domains even when spread abstains.
    spread = (spread[0], spread[1], valid_winddown.sum(axis=1).astype(int), spread[3])
    _append_summary_arrays(semantic, "winddown_spread", spread)

    standardized_timing_groups = (
        (
            ("screen_disengagement_p90__z", "screen_disengagement_p90__confidence", 1.0),
            ("usage_disengagement_p90__z", "usage_disengagement_p90__confidence", 1.0),
        ),
        (
            (
                "phone_activity_disengagement_p90__z",
                "phone_activity_disengagement_p90__confidence",
                1.0,
            ),
            ("step_disengagement_p90__z", "step_disengagement_p90__confidence", 1.0),
        ),
        (
            ("mobile_low_light_onset__z", "mobile_low_light_onset__confidence", 1.0),
            ("wearable_low_light_onset__z", "wearable_low_light_onset__confidence", 1.0),
        ),
    )
    standardized_summaries = []
    for sources in standardized_timing_groups:
        values, weights = _source_matrices(frame, sources)
        standardized_summaries.append(
            _summary_arrays(values, weights, operation="mean", expected_source_count=2)
        )
    routine_values = np.column_stack(
        [
            semantic["digital_engagement_score"],
            standardized_summaries[0][0],
            semantic["physical_activity_score"],
            standardized_summaries[1][0],
            semantic["light_exposure_score"],
            standardized_summaries[2][0],
            semantic["physiological_settling_score"],
            pd.to_numeric(frame["charging_onset__z"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    routine_weights = np.column_stack(
        [
            semantic["digital_engagement_score__confidence"],
            standardized_summaries[0][1],
            semantic["physical_activity_score__confidence"],
            standardized_summaries[1][1],
            semantic["light_exposure_score__confidence"],
            standardized_summaries[2][1],
            semantic["physiological_settling_score__confidence"],
            pd.to_numeric(
                frame["charging_onset__confidence"], errors="coerce"
            ).to_numpy(dtype=float),
        ]
    )
    routine = _summary_arrays(
        routine_values,
        routine_weights,
        operation="rms",
        expected_source_count=8,
    )
    _append_summary_arrays(semantic, "routine_deviation", routine)

    result = pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(semantic)], axis=1
    )
    sort_columns = [
        column
        for column in ("sensor_day_id", "held_out_subject", "k", "method", "donor_subject")
        if column in result.columns
    ]
    if sort_columns:
        result = result.sort_values(sort_columns, kind="stable", na_position="first")
    return result.reset_index(drop=True)


def _prepare_primitive_calendar(primitives: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy only frozen sensor columns, ignoring outcome bait."""
    if not isinstance(primitives, pd.DataFrame):
        raise TypeError("primitives must be a pandas DataFrame")
    if primitives.columns.duplicated().any():
        raise ValueError("Primitive input columns must be unique")
    missing_metadata = sorted(set(_SENSOR_METADATA_COLUMNS).difference(primitives.columns))
    if missing_metadata:
        raise ValueError(f"Missing sensor-calendar columns: {missing_metadata}")

    allowed_columns = list(_SENSOR_METADATA_COLUMNS)
    missing_primitive_columns: list[str] = []
    for definition in PRIMITIVE_DEFINITIONS:
        required = (
            definition.name,
            f"{definition.name}__status",
            f"{definition.name}__availability_coverage",
            f"{definition.name}__longest_gap_minutes",
            f"{definition.name}__expected_epochs",
            f"{definition.name}__support_density",
        )
        for column in required:
            if column not in primitives.columns:
                missing_primitive_columns.append(column)
            else:
                allowed_columns.append(column)
        censoring = f"{definition.name}__censoring"
        if censoring in primitives.columns:
            allowed_columns.append(censoring)
    if missing_primitive_columns:
        raise ValueError(
            f"Missing primitive or quality columns: {sorted(missing_primitive_columns)}"
        )

    result = primitives.loc[:, allowed_columns].copy(deep=True)
    sensor_day_id = pd.to_numeric(result["sensor_day_id"], errors="coerce")
    if (
        sensor_day_id.isna().any()
        or not np.isfinite(sensor_day_id).all()
        or sensor_day_id.lt(0).any()
        or not sensor_day_id.eq(np.floor(sensor_day_id)).all()
    ):
        raise ValueError("sensor_day_id must contain finite nonnegative integers")
    result["sensor_day_id"] = sensor_day_id.astype(int)
    if result["sensor_day_id"].duplicated().any():
        raise ValueError("sensor_day_id must be unique")
    if result["subject_id"].isna().any():
        raise ValueError("subject_id must be nonmissing")

    for column in ("lifelog_date", "sleep_date", "collection_start_date"):
        parsed = pd.to_datetime(result[column], errors="raise")
        if getattr(parsed.dt, "tz", None) is not None:
            raise ValueError(f"{column} must be timezone-naive local midnight")
        if parsed.isna().any() or not parsed.eq(parsed.dt.normalize()).all():
            raise ValueError(f"{column} must be exact local midnight")
        result[column] = parsed
    if not (
        result["sleep_date"] - result["lifelog_date"]
    ).eq(pd.Timedelta(days=1)).all():
        raise ValueError("sleep_date must be exactly one day after lifelog_date")
    if result.duplicated(["subject_id", "lifelog_date", "sleep_date"]).any():
        raise ValueError("participant-day keys must be unique")

    elapsed = pd.to_numeric(result["elapsed_day_index"], errors="coerce")
    if (
        elapsed.isna().any()
        or not np.isfinite(elapsed).all()
        or elapsed.lt(1).any()
        or not elapsed.eq(np.floor(elapsed)).all()
    ):
        raise ValueError("elapsed_day_index must contain positive integers")
    result["elapsed_day_index"] = elapsed.astype(int)
    expected_elapsed = (
        result["lifelog_date"] - result["collection_start_date"]
    ).dt.days + 1
    if not result["elapsed_day_index"].eq(expected_elapsed).all():
        raise ValueError("elapsed_day_index must match the sensor collection clock")
    starts_per_subject = result.groupby("subject_id", sort=False)[
        "collection_start_date"
    ].nunique(dropna=False)
    if not starts_per_subject.eq(1).all():
        raise ValueError("collection_start_date must be constant per participant")

    for column in (
        "any_source_record_observed",
        "any_measurement_support_observed",
        "any_sensor_observed",
    ):
        if not result[column].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"{column} must contain exact boolean values")
        result[column] = result[column].astype(bool)
    if not result["any_sensor_observed"].equals(
        result["any_measurement_support_observed"]
    ):
        raise ValueError(
            "any_sensor_observed must equal the measurement-support alias"
        )

    for _, participant in result.groupby("subject_id", sort=False):
        source_observed = participant.loc[
            participant["any_source_record_observed"]
        ]
        if source_observed.empty:
            raise ValueError(
                "Each participant must have at least one source-observed calendar day"
            )
        calendar_min = participant["lifelog_date"].min()
        calendar_max = participant["lifelog_date"].max()
        source_min = source_observed["lifelog_date"].min()
        source_max = source_observed["lifelog_date"].max()
        if calendar_min != source_min or calendar_max != source_max:
            raise ValueError(
                "Calendar boundaries must equal source-observed date boundaries"
            )
        collection_start = participant["collection_start_date"].iloc[0]
        if collection_start != source_min:
            raise ValueError(
                "collection_start_date must equal the earliest source-observed date"
            )
        observed_elapsed = sorted(participant["elapsed_day_index"].tolist())
        expected_calendar = list(range(1, max(observed_elapsed) + 1))
        if observed_elapsed != expected_calendar:
            raise ValueError(
                "Each participant must retain a complete elapsed calendar without filtered rows"
            )

    primitive_names = [definition.name for definition in PRIMITIVE_DEFINITIONS]
    finite_primitives = result[primitive_names].apply(
        lambda column: pd.to_numeric(column, errors="coerce")
    ).notna().any(axis=1)
    if bool((~result["any_measurement_support_observed"] & finite_primitives).any()):
        raise ValueError("A no-support calendar row cannot contain a finite primitive")

    return result.sort_values("sensor_day_id", kind="stable").reset_index(drop=True)


def _estimate_by_primitive(
    frame: pd.DataFrame,
    confidence: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    estimates: dict[str, dict[str, float | int]] = {}
    for definition in PRIMITIVE_DEFINITIONS:
        estimates[definition.name] = robust_location_scale(
            pd.to_numeric(frame[definition.name], errors="coerce").to_numpy(dtype=float),
            confidence[definition.name].to_numpy(dtype=float),
        )
    return estimates


def _personal_baseline(
    calibration: dict[str, float | int],
    population: dict[str, float | int],
) -> dict[str, Any]:
    population_center = float(population["center"])
    population_scale = float(population["scale"])
    n_eff = float(calibration["n_eff"])
    personal_center = float(calibration["center"])
    personal_scale = float(calibration["scale"])
    if (
        not np.isfinite(population_center)
        or not np.isfinite(population_scale)
        or population_scale <= 0
    ):
        return {
            "center": float("nan"),
            "scale": float("nan"),
            "center_weight": float("nan"),
            "scale_weight": float("nan"),
            "scale_floor": float("nan"),
            "status": "degenerate",
            "scale_fallback": "population_degenerate",
        }
    scale_floor = 0.1 * population_scale
    if n_eff <= 0 or not np.isfinite(personal_center):
        return {
            "center": float("nan"),
            "scale": float("nan"),
            "center_weight": 1.0,
            "scale_weight": 0.0,
            "scale_floor": scale_floor,
            "status": "insufficient",
            "scale_fallback": "no_personal_center",
        }
    if n_eff >= 3.0 and np.isfinite(personal_scale):
        scale = max(personal_scale, scale_floor)
        scale_weight = 1.0
        fallback = "scale_floor" if personal_scale < scale_floor else "none"
    else:
        scale = population_scale
        scale_weight = 0.0
        fallback = "population_insufficient_n_eff"
    return {
        "center": personal_center,
        "scale": float(scale),
        "center_weight": 1.0,
        "scale_weight": scale_weight,
        "scale_floor": scale_floor,
        "status": "observed",
        "scale_fallback": fallback,
    }


def _method_baseline(
    method: str,
    calibration: dict[str, float | int],
    population: dict[str, float | int],
    *,
    lambda_days: float,
) -> dict[str, Any]:
    if method == "population":
        valid = (
            np.isfinite(float(population["center"]))
            and np.isfinite(float(population["scale"]))
            and float(population["scale"]) > 0
        )
        return {
            "center": float(population["center"]) if valid else float("nan"),
            "scale": float(population["scale"]) if valid else float("nan"),
            "center_weight": 0.0 if valid else float("nan"),
            "scale_weight": 0.0 if valid else float("nan"),
            "scale_floor": 0.1 * float(population["scale"]) if valid else float("nan"),
            "status": "observed" if valid else "degenerate",
            "scale_fallback": "none" if valid else "population_degenerate",
        }
    if method == "personal":
        return _personal_baseline(calibration, population)
    if method in {"shrinkage", "placebo"}:
        result = shrink_location_scale(
            personal_center=float(calibration["center"]),
            personal_scale=float(calibration["scale"]),
            population_center=float(population["center"]),
            population_scale=float(population["scale"]),
            n_eff=float(calibration["n_eff"]),
            lambda_days=lambda_days,
        )
        if result["status"] == "degenerate":
            fallback = "population_degenerate"
        elif float(calibration["n_eff"]) < 3.0:
            fallback = "population_insufficient_n_eff"
        else:
            unbounded_scale = (
                float(result["scale_weight"]) * float(calibration["scale"])
                + (1.0 - float(result["scale_weight"])) * float(population["scale"])
            )
            fallback = "scale_floor" if unbounded_scale < float(result["scale_floor"]) else "none"
        return {**result, "scale_fallback": fallback}
    raise ValueError(f"Unsupported method: {method}")


def _baseline_record(
    *,
    held_out_subject: Any,
    donor_subject: Any | None,
    k: int,
    method: str,
    primitive: str,
    population: dict[str, float | int],
    calibration: dict[str, float | int],
    estimate: dict[str, Any],
    lambda_days: float,
) -> dict[str, Any]:
    return {
        "held_out_subject": held_out_subject,
        "k": int(k),
        "method": method,
        "donor_subject": donor_subject,
        "primitive": primitive,
        "center": estimate["center"],
        "scale": estimate["scale"],
        "n_eff": (
            float(population["n_eff"])
            if method == "population"
            else float(calibration["n_eff"])
        ),
        "contributing_rows": (
            int(population["contributing_rows"])
            if method == "population"
            else int(calibration["contributing_rows"])
        ),
        "population_center": float(population["center"]),
        "population_scale": float(population["scale"]),
        "population_n_eff": float(population["n_eff"]),
        "population_contributing_rows": int(population["contributing_rows"]),
        "calibration_center": float(calibration["center"]),
        "calibration_scale": float(calibration["scale"]),
        "calibration_n_eff": float(calibration["n_eff"]),
        "calibration_contributing_rows": int(calibration["contributing_rows"]),
        "center_weight": estimate["center_weight"],
        "scale_weight": estimate["scale_weight"],
        "scale_floor": estimate["scale_floor"],
        "scale_fallback": estimate["scale_fallback"],
        "baseline_status": estimate["status"],
        "lambda_days": float(lambda_days),
    }


def _confidence_wide(
    audit: pd.DataFrame,
    column: str,
    sensor_day_ids: pd.Series,
) -> pd.DataFrame:
    wide = audit.pivot(index="sensor_day_id", columns="primitive", values=column)
    names = [definition.name for definition in PRIMITIVE_DEFINITIONS]
    return wide.reindex(index=sensor_day_ids.to_numpy(), columns=names)


def build_loso_representations(
    primitives: pd.DataFrame,
    *,
    k_values: Sequence[int] = PERSONALIZATION_K_VALUES,
    lambda_days: float = PRIMARY_LAMBDA_DAYS,
    usage_policy: str = "screen_proxy",
) -> LosoRepresentationTables:
    """Build the sparse LOSO cold-start grid without reading any outcomes."""
    requested_values: list[int] = []
    for value in k_values:
        if isinstance(value, (bool, str, bytes)):
            raise ValueError("k_values must contain exact integers")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("k_values must contain exact integers") from error
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError("k_values must contain exact integers")
        requested_values.append(int(numeric_value))
    requested_k = tuple(requested_values)
    if not requested_k or len(requested_k) != len(set(requested_k)):
        raise ValueError("k_values must be a nonempty unique subset of 3, 7, 14, 21")
    if not set(requested_k).issubset(PERSONALIZATION_K_VALUES):
        raise ValueError("k_values must be a subset of 3, 7, 14, 21")
    requested_k = tuple(value for value in PERSONALIZATION_K_VALUES if value in requested_k)
    if not np.isfinite(lambda_days) or lambda_days < 0:
        raise ValueError("lambda_days must be finite and nonnegative")

    calendar = _prepare_primitive_calendar(primitives)
    subjects = tuple(sorted(calendar["subject_id"].unique(), key=str))
    if len(subjects) < 2:
        raise ValueError("LOSO personalization requires at least two participants")
    confidence_audit = compute_primitive_confidence(
        calendar, usage_policy=usage_policy
    )
    quality = _confidence_wide(
        confidence_audit, "quality_confidence", calendar["sensor_day_id"]
    )
    coverage_only = _confidence_wide(
        confidence_audit, "coverage_only_confidence", calendar["sensor_day_id"]
    )
    value_only = _confidence_wide(
        confidence_audit, "value_only_confidence", calendar["sensor_day_id"]
    )
    quality.index = calendar.index
    coverage_only.index = calendar.index
    value_only.index = calendar.index

    baseline_records: list[dict[str, Any]] = []
    group_baselines: list[tuple[Any, int, str, Any | None, dict[str, dict[str, Any]]]] = []
    for held_out_subject in subjects:
        outer_mask = calendar["subject_id"].ne(held_out_subject)
        population_estimates = _estimate_by_primitive(
            calendar.loc[outer_mask], quality.loc[outer_mask]
        )
        population_method: dict[str, dict[str, Any]] = {}
        for definition in PRIMITIVE_DEFINITIONS:
            primitive = definition.name
            population = population_estimates[primitive]
            estimate = _method_baseline(
                "population", population, population, lambda_days=float(lambda_days)
            )
            population_method[primitive] = estimate
            baseline_records.append(
                _baseline_record(
                    held_out_subject=held_out_subject,
                    donor_subject=None,
                    k=0,
                    method="population",
                    primitive=primitive,
                    population=population,
                    calibration=population,
                    estimate=estimate,
                    lambda_days=float(lambda_days),
                )
            )
        group_baselines.append(
            (held_out_subject, 0, "population", None, population_method)
        )

        for k in requested_k:
            personal_mask = calendar["subject_id"].eq(held_out_subject) & calendar[
                "elapsed_day_index"
            ].le(k)
            personal_estimates = _estimate_by_primitive(
                calendar.loc[personal_mask], quality.loc[personal_mask]
            )
            for method in ("personal", "shrinkage"):
                method_estimates: dict[str, dict[str, Any]] = {}
                for definition in PRIMITIVE_DEFINITIONS:
                    primitive = definition.name
                    population = population_estimates[primitive]
                    calibration = personal_estimates[primitive]
                    estimate = _method_baseline(
                        method,
                        calibration,
                        population,
                        lambda_days=float(lambda_days),
                    )
                    method_estimates[primitive] = estimate
                    baseline_records.append(
                        _baseline_record(
                            held_out_subject=held_out_subject,
                            donor_subject=None,
                            k=k,
                            method=method,
                            primitive=primitive,
                            population=population,
                            calibration=calibration,
                            estimate=estimate,
                            lambda_days=float(lambda_days),
                        )
                    )
                group_baselines.append(
                    (held_out_subject, k, method, None, method_estimates)
                )

            for donor_subject in subjects:
                if donor_subject == held_out_subject:
                    continue
                donor_mask = calendar["subject_id"].eq(donor_subject) & calendar[
                    "elapsed_day_index"
                ].le(k)
                donor_estimates = _estimate_by_primitive(
                    calendar.loc[donor_mask], quality.loc[donor_mask]
                )
                placebo_estimates: dict[str, dict[str, Any]] = {}
                for definition in PRIMITIVE_DEFINITIONS:
                    primitive = definition.name
                    population = population_estimates[primitive]
                    calibration = donor_estimates[primitive]
                    estimate = _method_baseline(
                        "placebo",
                        calibration,
                        population,
                        lambda_days=float(lambda_days),
                    )
                    placebo_estimates[primitive] = estimate
                    baseline_records.append(
                        _baseline_record(
                            held_out_subject=held_out_subject,
                            donor_subject=donor_subject,
                            k=k,
                            method="placebo",
                            primitive=primitive,
                            population=population,
                            calibration=calibration,
                            estimate=estimate,
                            lambda_days=float(lambda_days),
                        )
                    )
                group_baselines.append(
                    (held_out_subject, k, "placebo", donor_subject, placebo_estimates)
                )

    metadata_columns = list(_SENSOR_METADATA_COLUMNS)
    baseline_lookup = {
        (
            record["held_out_subject"],
            record["k"],
            record["method"],
            record["donor_subject"],
            record["primitive"],
        ): record
        for record in baseline_records
    }
    representation_blocks: list[pd.DataFrame] = []
    for held_out_subject, k, method, donor_subject, method_estimates in group_baselines:
        evaluation_mask = calendar["subject_id"].eq(held_out_subject) & calendar[
            "elapsed_day_index"
        ].ge(EVALUATION_DAY_START)
        evaluation = calendar.loc[evaluation_mask]
        row_count = len(evaluation)
        block_data: dict[str, Any] = {
            column: evaluation[column].to_numpy(copy=True)
            for column in metadata_columns
        }
        block_data["held_out_subject"] = np.repeat(held_out_subject, row_count)
        block_data["k"] = np.repeat(int(k), row_count)
        block_data["method"] = np.repeat(method, row_count)
        block_data["donor_subject"] = np.repeat(donor_subject, row_count)
        for definition in PRIMITIVE_DEFINITIONS:
            primitive = definition.name
            estimate = method_estimates[primitive]
            values = pd.to_numeric(evaluation[primitive], errors="coerce").astype(float)
            primitive_confidence = quality.loc[evaluation.index, primitive].astype(float)
            block_data[primitive] = values.to_numpy()
            block_data[f"{primitive}__confidence"] = primitive_confidence.to_numpy()
            block_data[f"{primitive}__coverage_only_confidence"] = coverage_only.loc[
                evaluation.index, primitive
            ].to_numpy(dtype=float)
            block_data[f"{primitive}__value_only_confidence"] = value_only.loc[
                evaluation.index, primitive
            ].to_numpy(dtype=float)
            block_data[f"{primitive}__center"] = np.repeat(estimate["center"], row_count)
            block_data[f"{primitive}__scale"] = np.repeat(estimate["scale"], row_count)
            baseline_row = baseline_lookup[
                (held_out_subject, k, method, donor_subject, primitive)
            ]
            block_data[f"{primitive}__n_eff"] = np.repeat(
                baseline_row["n_eff"], row_count
            )
            block_data[f"{primitive}__baseline_status"] = np.repeat(
                estimate["status"], row_count
            )
            z = np.full(row_count, np.nan, dtype=float)
            value_array = values.to_numpy(dtype=float)
            confidence_array = primitive_confidence.to_numpy(dtype=float)
            usable = (
                np.isfinite(value_array)
                & (confidence_array > 0)
                & np.isfinite(float(estimate["center"]))
                & np.isfinite(float(estimate["scale"]))
                & (float(estimate["scale"]) > 0)
            )
            if bool(usable.any()):
                z[usable] = np.clip(
                    (value_array[usable] - float(estimate["center"]))
                    / float(estimate["scale"]),
                    -5.0,
                    5.0,
                )
            block_data[f"{primitive}__z"] = z
        representation_blocks.append(pd.DataFrame(block_data))

    raw_representations = pd.concat(representation_blocks, ignore_index=True)
    representations = compose_semantic_columns(raw_representations)
    method_order = pd.Categorical(
        representations["method"],
        categories=["population", "personal", "shrinkage", "placebo"],
        ordered=True,
    )
    representations = representations.assign(_method_order=method_order).sort_values(
        ["_method_order", "k", "held_out_subject", "donor_subject", "sensor_day_id"],
        kind="stable",
        na_position="first",
    ).drop(columns="_method_order").reset_index(drop=True)
    if representations.duplicated(
        ["sensor_day_id", "held_out_subject", "k", "method", "donor_subject"]
    ).any():
        raise RuntimeError("Representation keys are not unique")

    baselines = pd.DataFrame.from_records(baseline_records)
    baseline_method_order = pd.Categorical(
        baselines["method"],
        categories=["population", "personal", "shrinkage", "placebo"],
        ordered=True,
    )
    baselines = baselines.assign(_method_order=baseline_method_order).sort_values(
        ["_method_order", "k", "held_out_subject", "donor_subject", "primitive"],
        kind="stable",
        na_position="first",
    ).drop(columns="_method_order").reset_index(drop=True)
    placebo_candidates = (
        baselines.loc[
            baselines["method"].eq("placebo"),
            ["held_out_subject", "donor_subject", "k"],
        ]
        .drop_duplicates()
        .sort_values(
            ["held_out_subject", "donor_subject", "k"], kind="stable"
        )
        .reset_index(drop=True)
    )
    placebo_candidates.insert(0, "candidate_id", np.arange(len(placebo_candidates)))
    placebo_candidates["same_elapsed_window"] = True
    placebo_candidates["all_primitives_same_donor"] = True
    placebo_candidates["is_self"] = False
    return LosoRepresentationTables(
        representations=representations,
        baselines=baselines,
        confidence_audit=confidence_audit,
        placebo_candidates=placebo_candidates,
    )


def build_future_baseline_oracles(
    primitives: pd.DataFrame,
    *,
    usage_policy: str = "screen_proxy",
) -> pd.DataFrame:
    """Build evaluation-only day-22-plus oracle baselines for Task 4 scoring."""
    calendar = _prepare_primitive_calendar(primitives)
    subjects = tuple(sorted(calendar["subject_id"].unique(), key=str))
    if len(subjects) < 2:
        raise ValueError("Future oracle estimation requires at least two participants")
    audit = compute_primitive_confidence(calendar, usage_policy=usage_policy)
    quality = _confidence_wide(audit, "quality_confidence", calendar["sensor_day_id"])
    quality.index = calendar.index
    records: list[dict[str, Any]] = []
    for held_out_subject in subjects:
        population_mask = calendar["subject_id"].ne(held_out_subject)
        future_mask = calendar["subject_id"].eq(held_out_subject) & calendar[
            "elapsed_day_index"
        ].ge(EVALUATION_DAY_START)
        population = _estimate_by_primitive(
            calendar.loc[population_mask], quality.loc[population_mask]
        )
        future = _estimate_by_primitive(
            calendar.loc[future_mask], quality.loc[future_mask]
        )
        for definition in PRIMITIVE_DEFINITIONS:
            primitive = definition.name
            population_estimate = population[primitive]
            oracle = future[primitive]
            population_scale = float(population_estimate["scale"])
            population_valid = (
                np.isfinite(float(population_estimate["center"]))
                and np.isfinite(population_scale)
                and population_scale > 0
            )
            oracle_center = float(oracle["center"])
            oracle_raw_scale = float(oracle["scale"])
            n_eff = float(oracle["n_eff"])
            if not population_valid:
                oracle_scale = float("nan")
                scale_floor = float("nan")
                fallback = "population_degenerate"
            else:
                scale_floor = 0.1 * population_scale
                if n_eff < 3.0 or not np.isfinite(oracle_raw_scale):
                    oracle_scale = population_scale
                    fallback = "population_insufficient_n_eff"
                else:
                    oracle_scale = max(oracle_raw_scale, scale_floor)
                    fallback = "scale_floor" if oracle_raw_scale < scale_floor else "none"
            status = (
                "observed"
                if np.isfinite(oracle_center)
                and np.isfinite(oracle_scale)
                and oracle_scale > 0
                else "insufficient"
            )
            records.append(
                {
                    "held_out_subject": held_out_subject,
                    "primitive": primitive,
                    "oracle_center": oracle_center,
                    "oracle_raw_scale": oracle_raw_scale,
                    "oracle_scale": float(oracle_scale),
                    "oracle_n_eff": n_eff,
                    "oracle_contributing_rows": int(oracle["contributing_rows"]),
                    "oracle_scale_floor": float(scale_floor),
                    "oracle_scale_fallback": fallback,
                    "oracle_status": status,
                    "population_center": float(population_estimate["center"]),
                    "population_scale": population_scale,
                    "population_n_eff": float(population_estimate["n_eff"]),
                    "population_contributing_rows": int(
                        population_estimate["contributing_rows"]
                    ),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["held_out_subject", "primitive"], kind="stable"
    ).reset_index(drop=True)
