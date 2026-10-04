"""Post-hoc dual calibration coordinates with role-locked semantics.

This module is intentionally separate from the frozen Task 3 representation
builder.  It treats those artifacts as immutable inputs, recomputes no
calibration estimate, and exposes no outcome-dependent coordinate selector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import PRIMITIVE_DEFINITIONS
from .personalization import LosoRepresentationTables


REFERENCE_COORDINATE = "reference_z"
SELF_COORDINATE = "self_z"
RAW_TIMING_COORDINATE = "raw_timing"
DUAL_CALIBRATION_PROTOCOL = "post_hoc_exploratory_dual_v1"
CLIP_LIMIT = 5.0
EVALUATION_DAY_START = 22
FROZEN_PERSONALIZATION_K = (3, 7, 14, 21)

REPRESENTATION_KEYS = (
    "sensor_day_id",
    "held_out_subject",
    "k",
    "method",
    "donor_subject",
)
_GROUP_KEYS = ("held_out_subject", "k", "method", "donor_subject")
_ROW_METADATA = (
    "sensor_day_id",
    "subject_id",
    "lifelog_date",
    "sleep_date",
    "collection_start_date",
    "elapsed_day_index",
    "any_source_record_observed",
    "any_measurement_support_observed",
    "any_sensor_observed",
    "held_out_subject",
    "k",
    "method",
    "donor_subject",
)
_PRIMITIVE_NAMES = tuple(definition.name for definition in PRIMITIVE_DEFINITIONS)
_NULL_DONOR_TOKEN = "__DUAL_CALIBRATION_NULL_DONOR__"


@dataclass(frozen=True)
class DualCalibrationTables:
    """Separate post-hoc coordinate rows and their frozen role policy."""

    coordinates: pd.DataFrame
    coordinate_policy: pd.DataFrame


@dataclass(frozen=True)
class _SemanticPolicy:
    output_column: str
    consumer_role: str
    coordinate_policy: str
    source_columns: tuple[str, ...]
    aggregation: str
    expected_source_count: int
    clipping_state: str


_SEMANTIC_POLICIES = (
    _SemanticPolicy(
        "reference_digital_engagement_score",
        "cross_modal",
        REFERENCE_COORDINATE,
        (
            "screen_load_24h__reference_z",
            "usage_load_24h__reference_z",
        ),
        "quality_weighted_mean",
        2,
        "clipped_-5_5",
    ),
    _SemanticPolicy(
        "reference_physical_activity_score",
        "cross_modal",
        REFERENCE_COORDINATE,
        (
            "phone_activity_load_24h__reference_z",
            "step_load_24h__reference_z",
        ),
        "quality_weighted_mean",
        2,
        "clipped_-5_5",
    ),
    _SemanticPolicy(
        "reference_light_exposure_score",
        "cross_modal",
        REFERENCE_COORDINATE,
        (
            "mobile_light_exposure_24h__reference_z",
            "wearable_light_exposure_24h__reference_z",
        ),
        "quality_weighted_mean",
        2,
        "clipped_-5_5",
    ),
    _SemanticPolicy(
        "reference_physiological_settling_score",
        "cross_modal",
        REFERENCE_COORDINATE,
        (
            "heart_rate_settling_delta__reference_z",
            "heart_rate_settling_slope__reference_z",
        ),
        "quality_weighted_mean_negated_sources",
        2,
        "clipped_-5_5",
    ),
    _SemanticPolicy(
        "self_routine_deviation",
        "routine_anomaly",
        SELF_COORDINATE,
        tuple(f"{name}__self_z" for name in (
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
        )),
        "quality_weighted_rms_of_8_nonoverlapping_components",
        8,
        "clipped_-5_5",
    ),
    _SemanticPolicy(
        "raw_timing_digital_disengagement_time",
        "raw_timing",
        RAW_TIMING_COORDINATE,
        ("screen_disengagement_p90", "usage_disengagement_p90"),
        "symmetric_quality_weighted_median_minutes",
        2,
        "not_applicable",
    ),
    _SemanticPolicy(
        "raw_timing_physical_disengagement_time",
        "raw_timing",
        RAW_TIMING_COORDINATE,
        (
            "phone_activity_disengagement_p90",
            "step_disengagement_p90",
        ),
        "symmetric_quality_weighted_median_minutes",
        2,
        "not_applicable",
    ),
    _SemanticPolicy(
        "raw_timing_low_light_onset",
        "raw_timing",
        RAW_TIMING_COORDINATE,
        ("mobile_low_light_onset", "wearable_low_light_onset"),
        "symmetric_quality_weighted_median_minutes",
        2,
        "not_applicable",
    ),
    _SemanticPolicy(
        "raw_timing_winddown_center",
        "raw_timing",
        RAW_TIMING_COORDINATE,
        (
            "raw_timing_digital_disengagement_time",
            "raw_timing_physical_disengagement_time",
            "raw_timing_low_light_onset",
            "charging_onset",
        ),
        "symmetric_quality_weighted_median_minutes",
        4,
        "not_applicable",
    ),
    _SemanticPolicy(
        "raw_timing_winddown_spread",
        "raw_timing",
        RAW_TIMING_COORDINATE,
        (
            "raw_timing_digital_disengagement_time",
            "raw_timing_physical_disengagement_time",
            "raw_timing_low_light_onset",
            "charging_onset",
        ),
        "symmetric_quality_weighted_mad_minutes_min_2",
        4,
        "not_applicable",
    ),
)


def coordinate_policy_table() -> pd.DataFrame:
    """Return the fixed, outcome-independent semantic coordinate policy."""
    records: list[dict[str, Any]] = []
    for primitive in _PRIMITIVE_NAMES:
        coordinate_records = (
            (
                f"{primitive}__reference_z_unclipped",
                "reference_coordinate",
                REFERENCE_COORDINATE,
                (
                    f"representations.{primitive}",
                    f"baselines.{primitive}.center",
                    f"baselines.{primitive}.population_k0_scale",
                ),
                "(value-shared_center)/recipient_excluding_population_scale",
                "unclipped",
            ),
            (
                f"{primitive}__reference_z",
                "reference_coordinate",
                REFERENCE_COORDINATE,
                (f"{primitive}__reference_z_unclipped",),
                "clip_-5_5",
                "clipped_-5_5",
            ),
            (
                f"{primitive}__self_z_unclipped",
                "self_coordinate",
                SELF_COORDINATE,
                (
                    f"representations.{primitive}",
                    f"baselines.{primitive}.center",
                    f"baselines.{primitive}.scale",
                ),
                "(value-shared_center)/task3_estimated_self_scale",
                "unclipped",
            ),
            (
                f"{primitive}__self_z",
                "self_coordinate",
                SELF_COORDINATE,
                (f"{primitive}__self_z_unclipped",),
                "clip_-5_5",
                "clipped_-5_5",
            ),
            (
                f"{primitive}__population_scale",
                "coordinate_audit",
                "recipient_excluding_population_scale",
                (
                    f"baselines.{primitive}.population_k0_scale",
                    f"baselines.{primitive}.population_scale_redundant_copies",
                ),
                "authoritative_scale_plus_equality_audit",
                "not_applicable",
            ),
        )
        for (
            output_column,
            consumer_role,
            coordinate_policy,
            source_columns,
            aggregation,
            clipping_state,
        ) in coordinate_records:
            records.append(
                {
                    "output_column": output_column,
                    "consumer_role": consumer_role,
                    "coordinate_policy": coordinate_policy,
                    "source_columns": source_columns,
                    "aggregation": aggregation,
                    "expected_source_count": len(source_columns),
                    "clipping_state": clipping_state,
                    "post_hoc_exploratory": True,
                    "protocol": DUAL_CALIBRATION_PROTOCOL,
                }
            )
    records.extend(
        {
            "output_column": policy.output_column,
            "consumer_role": policy.consumer_role,
            "coordinate_policy": policy.coordinate_policy,
            "source_columns": policy.source_columns,
            "aggregation": policy.aggregation,
            "expected_source_count": policy.expected_source_count,
            "clipping_state": policy.clipping_state,
            "post_hoc_exploratory": True,
            "protocol": DUAL_CALIBRATION_PROTOCOL,
        }
        for policy in _SEMANTIC_POLICIES
    )
    return pd.DataFrame.from_records(records)


def _require_unique_columns(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} columns must be unique")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")


def _donor_tokens(values: pd.Series) -> pd.Series:
    return _typed_tokens(values, allow_null=True, field="donor_subject")


def _is_missing_scalar(value: Any, *, field: str) -> bool:
    missing = pd.isna(value)
    if not isinstance(missing, (bool, np.bool_)):
        raise ValueError(f"{field} values must be scalar keys")
    return bool(missing)


def _typed_key_token(value: Any, *, allow_null: bool, field: str) -> str:
    if _is_missing_scalar(value, field=field):
        if not allow_null:
            raise ValueError(f"{field} values must be nonmissing")
        return _NULL_DONOR_TOKEN
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _typed_tokens(
    values: pd.Series,
    *,
    allow_null: bool,
    field: str,
) -> pd.Series:
    return values.map(
        lambda value: _typed_key_token(
            value, allow_null=allow_null, field=field
        )
    )


def _exact_k_values(values: pd.Series, *, field: str) -> pd.Series:
    normalized: list[int] = []
    for value in values.tolist():
        if isinstance(value, (bool, np.bool_, str, bytes)):
            raise ValueError(f"{field} values must be exact integers")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} values must be exact integers") from error
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{field} values must be exact integers")
        normalized.append(int(numeric))
    return pd.Series(normalized, index=values.index, dtype=int)


def _normalized_group_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    _require_columns(frame, _GROUP_KEYS, name)
    methods = frame["method"]
    invalid_method_type = ~methods.map(lambda value: isinstance(value, str))
    if bool(invalid_method_type.any()):
        raise ValueError(f"{name} method values must be strings")
    return pd.DataFrame(
        {
            "held": _typed_tokens(
                frame["held_out_subject"],
                allow_null=False,
                field=f"{name}.held_out_subject",
            ),
            "k": _exact_k_values(frame["k"], field=f"{name}.k"),
            "method": methods.astype(object),
            "donor": _donor_tokens(frame["donor_subject"]),
        },
        index=frame.index,
    )


def _group_tuples(normalized: pd.DataFrame) -> list[tuple[str, int, str, str]]:
    return list(
        normalized.loc[:, ["held", "k", "method", "donor"]].itertuples(
            index=False, name=None
        )
    )


def _numeric_equal(left: pd.Series, right: pd.Series) -> np.ndarray:
    left_array = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_array = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return (left_array == right_array) | (np.isnan(left_array) & np.isnan(right_array))


def _object_equal(left: pd.Series, right: pd.Series) -> np.ndarray:
    left_string = left.astype("string")
    right_string = right.astype("string")
    both_missing = left_string.isna() & right_string.isna()
    return (left_string.eq(right_string).fillna(False) | both_missing).to_numpy()


def _sort_representations(representations: pd.DataFrame) -> pd.DataFrame:
    method_order = pd.Categorical(
        representations["method"],
        categories=["population", "personal", "shrinkage", "placebo"],
        ordered=True,
    )
    if bool(pd.isna(method_order).any()):
        unknown = sorted(
            set(representations.loc[pd.isna(method_order), "method"].astype(str))
        )
        raise ValueError(f"Unknown frozen representation methods: {unknown}")
    return (
        representations.assign(
            _dual_method_order=method_order,
            _dual_held_order=_typed_tokens(
                representations["held_out_subject"],
                allow_null=False,
                field="representations.held_out_subject",
            ),
            _dual_donor_order=_donor_tokens(representations["donor_subject"]),
        )
        .sort_values(
            [
                "_dual_method_order",
                "k",
                "_dual_held_order",
                "_dual_donor_order",
                "sensor_day_id",
            ],
            kind="stable",
            na_position="first",
        )
        .drop(
            columns=[
                "_dual_method_order",
                "_dual_held_order",
                "_dual_donor_order",
            ]
        )
        .reset_index(drop=True)
    )


def _validate_frozen_topology(
    primitives: pd.DataFrame,
    representations: pd.DataFrame,
    baselines: pd.DataFrame,
) -> None:
    subject_tokens = set(
        _typed_tokens(
            primitives["subject_id"],
            allow_null=False,
            field="primitive calendar.subject_id",
        )
    )
    if len(subject_tokens) < 2:
        raise ValueError("Frozen topology requires at least two participants")

    expected: set[tuple[str, int, str, str]] = set()
    for held in subject_tokens:
        expected.add((held, 0, "population", _NULL_DONOR_TOKEN))
        for k in FROZEN_PERSONALIZATION_K:
            expected.add((held, k, "personal", _NULL_DONOR_TOKEN))
            expected.add((held, k, "shrinkage", _NULL_DONOR_TOKEN))
            for donor in subject_tokens:
                if donor != held:
                    expected.add((held, k, "placebo", donor))

    representation_groups = _normalized_group_frame(
        representations, name="representations"
    )
    baseline_groups = _normalized_group_frame(baselines, name="baselines")
    actual_representation = set(_group_tuples(representation_groups))
    actual_baseline = set(_group_tuples(baseline_groups))
    if actual_representation != expected:
        raise ValueError(
            "Representation frozen topology mismatch: "
            f"missing={len(expected - actual_representation)}, "
            f"extra={len(actual_representation - expected)}"
        )
    if actual_baseline != expected:
        raise ValueError(
            "Baseline frozen topology mismatch: "
            f"missing={len(expected - actual_baseline)}, "
            f"extra={len(actual_baseline - expected)}"
        )
    if actual_representation != actual_baseline:
        raise ValueError("Representation and baseline group topology must match exactly")

    required_primitives = set(_PRIMITIVE_NAMES)
    primitives_by_group: dict[tuple[str, int, str, str], list[str]] = {}
    for group, primitive in zip(
        _group_tuples(baseline_groups), baselines["primitive"].tolist(), strict=True
    ):
        if not isinstance(primitive, str):
            raise ValueError("Baseline primitive names must be strings")
        primitives_by_group.setdefault(group, []).append(primitive)
    for group in expected:
        group_primitives = primitives_by_group.get(group, [])
        if (
            len(group_primitives) != len(_PRIMITIVE_NAMES)
            or len(set(group_primitives)) != len(group_primitives)
            or set(group_primitives) != required_primitives
        ):
            raise ValueError(
                "Every baseline group in the frozen topology must be primitive-complete"
            )


def _validate_calendar_and_grid(
    primitives: pd.DataFrame,
    representations: pd.DataFrame,
) -> None:
    _require_unique_columns(primitives, "primitive calendar")
    _require_columns(
        primitives,
        ("sensor_day_id", "subject_id", "elapsed_day_index"),
        "primitive calendar",
    )
    if primitives["sensor_day_id"].isna().any() or primitives[
        "sensor_day_id"
    ].duplicated().any():
        raise ValueError("Primitive sensor_day_id values must be nonmissing and unique")
    elapsed = pd.to_numeric(primitives["elapsed_day_index"], errors="coerce")
    if (
        elapsed.isna().any()
        or not np.isfinite(elapsed).all()
        or not elapsed.eq(np.floor(elapsed)).all()
        or not elapsed.gt(0).all()
    ):
        raise ValueError("elapsed_day_index must contain positive exact integers")

    calendar = primitives.loc[:, ["sensor_day_id", "subject_id"]].copy()
    calendar["elapsed_day_index"] = elapsed.astype(int)
    calendar["_calendar_subject_token"] = _typed_tokens(
        calendar["subject_id"],
        allow_null=False,
        field="primitive calendar.subject_id",
    )
    evaluation = calendar.loc[calendar["elapsed_day_index"].ge(EVALUATION_DAY_START)]
    if evaluation.empty:
        raise ValueError("Primitive calendar has no post-day-21 evaluation rows")
    expected_by_subject = {
        subject_token: frozenset(group["sensor_day_id"].tolist())
        for subject_token, group in evaluation.groupby(
            "_calendar_subject_token", sort=False
        )
    }

    normalized_groups = _normalized_group_frame(
        representations, name="representations"
    )
    normalized_row_keys = [
        (
            _typed_key_token(
                sensor_day_id,
                allow_null=False,
                field="representations.sensor_day_id",
            ),
            *group,
        )
        for sensor_day_id, group in zip(
            representations["sensor_day_id"].tolist(),
            _group_tuples(normalized_groups),
            strict=True,
        )
    ]
    if len(normalized_row_keys) != len(set(normalized_row_keys)):
        raise ValueError("Frozen representation keys must be unique")
    held_tokens = normalized_groups["held"].reset_index(drop=True)
    representation_subject_tokens = _typed_tokens(
        representations["subject_id"],
        allow_null=False,
        field="representations.subject_id",
    ).reset_index(drop=True)
    if not held_tokens.eq(representation_subject_tokens).all():
        raise ValueError("held_out_subject must equal the sensor-calendar subject_id")

    joined = representations.loc[
        :, ["sensor_day_id", "held_out_subject", "elapsed_day_index"]
    ].merge(
        evaluation.rename(
            columns={
                "subject_id": "_calendar_subject",
                "elapsed_day_index": "_calendar_elapsed",
            }
        ).drop(columns="_calendar_subject_token"),
        on="sensor_day_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if joined["_calendar_subject"].isna().any():
        raise ValueError("Representations contain rows outside the post-day-21 calendar")
    joined_held_tokens = _typed_tokens(
        joined["held_out_subject"],
        allow_null=False,
        field="representations.held_out_subject",
    )
    joined_calendar_tokens = _typed_tokens(
        joined["_calendar_subject"],
        allow_null=False,
        field="primitive calendar.subject_id",
    )
    if not joined_held_tokens.eq(joined_calendar_tokens).all():
        raise ValueError("Representation sensor_day_id belongs to another participant")
    if not _numeric_equal(joined["elapsed_day_index"], joined["_calendar_elapsed"]).all():
        raise ValueError("Representation elapsed_day_index disagrees with the calendar")

    grid = normalized_groups.copy()
    grid["sensor_day_id"] = representations["sensor_day_id"].to_numpy(copy=True)
    seen_subjects: set[str] = set()
    for group_key, group in grid.groupby(
        ["held", "k", "method", "donor"], dropna=False, sort=False
    ):
        held_out_subject_token = group_key[0]
        if held_out_subject_token not in expected_by_subject:
            raise ValueError("Representation held_out_subject is absent from the calendar")
        if frozenset(group["sensor_day_id"].tolist()) != expected_by_subject[
            held_out_subject_token
        ]:
            raise ValueError("Every representation group must retain the complete common row grid")
        seen_subjects.add(held_out_subject_token)
    if seen_subjects != set(expected_by_subject):
        raise ValueError("Every participant must have a complete common row grid")


def _prepare_inputs(
    primitives: pd.DataFrame,
    loso_tables: LosoRepresentationTables,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(loso_tables, LosoRepresentationTables):
        raise TypeError("loso_tables must be a LosoRepresentationTables instance")
    representations = loso_tables.representations.copy(deep=True)
    baselines = loso_tables.baselines.copy(deep=True)
    _require_unique_columns(representations, "representations")
    _require_unique_columns(baselines, "baselines")
    _require_columns(representations, _ROW_METADATA, "representation")
    _require_columns(
        baselines,
        (
            "held_out_subject",
            "k",
            "method",
            "donor_subject",
            "primitive",
            "center",
            "scale",
            "population_scale",
            "baseline_status",
        ),
        "baseline",
    )
    primitive_representation_columns: list[str] = []
    for primitive in _PRIMITIVE_NAMES:
        primitive_representation_columns.extend(
            (
                primitive,
                f"{primitive}__confidence",
                f"{primitive}__coverage_only_confidence",
                f"{primitive}__value_only_confidence",
                f"{primitive}__center",
                f"{primitive}__scale",
                f"{primitive}__baseline_status",
            )
        )
    _require_columns(
        representations,
        primitive_representation_columns,
        "primitive representation",
    )
    _validate_calendar_and_grid(primitives, representations)
    if baselines.duplicated(
        ["held_out_subject", "k", "method", "donor_subject", "primitive"]
    ).any():
        raise ValueError("Frozen baseline keys must be unique")
    unknown_primitives = sorted(
        set(baselines["primitive"].astype(str)).difference(_PRIMITIVE_NAMES)
    )
    if unknown_primitives:
        raise ValueError(f"Unknown baseline primitives: {unknown_primitives}")
    _validate_frozen_topology(primitives, representations, baselines)
    return _sort_representations(representations), baselines


def _authoritative_population_scales(baselines: pd.DataFrame) -> pd.DataFrame:
    anchors = baselines.loc[
        baselines["method"].eq("population")
        & pd.to_numeric(baselines["k"], errors="coerce").eq(0)
        & baselines["donor_subject"].isna(),
        ["held_out_subject", "primitive", "scale", "population_scale"],
    ].copy()
    anchors["_dual_held_token"] = _typed_tokens(
        anchors["held_out_subject"],
        allow_null=False,
        field="population anchor.held_out_subject",
    )
    expected_subjects = set(
        _typed_tokens(
            baselines["held_out_subject"],
            allow_null=False,
            field="baselines.held_out_subject",
        )
    )
    expected_count = len(expected_subjects) * len(_PRIMITIVE_NAMES)
    if (
        len(anchors) != expected_count
        or anchors.duplicated(["_dual_held_token", "primitive"]).any()
        or set(anchors["primitive"]) != set(_PRIMITIVE_NAMES)
    ):
        raise ValueError(
            "Population-scale provenance requires one authoritative outer-fold anchor "
            "per participant and primitive"
        )
    if not _numeric_equal(anchors["scale"], anchors["population_scale"]).all():
        raise ValueError(
            "population-scale provenance requires the redundant population_scale "
            "copy to equal the population k=0 null-donor baseline scale"
        )
    anchors = anchors.rename(columns={"scale": "_authoritative_population_scale"})
    anchor_lookup = anchors.loc[
        :, ["_dual_held_token", "primitive", "_authoritative_population_scale"]
    ]
    audited = baselines.copy()
    audited["_dual_held_token"] = _typed_tokens(
        audited["held_out_subject"],
        allow_null=False,
        field="baselines.held_out_subject",
    )
    audited = audited.merge(
        anchor_lookup,
        on=["_dual_held_token", "primitive"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if not _numeric_equal(
        audited["population_scale"], audited["_authoritative_population_scale"]
    ).all():
        raise ValueError(
            "Baseline population-scale provenance disagrees with the authoritative "
            "recipient-excluding outer-fold anchor"
        )
    return anchor_lookup


def _baseline_for_primitive(
    representations: pd.DataFrame,
    baselines: pd.DataFrame,
    anchors: pd.DataFrame,
    primitive: str,
) -> pd.DataFrame:
    left = representations.loc[:, list(REPRESENTATION_KEYS)].copy()
    left["_dual_row_position"] = np.arange(len(left))
    left["_dual_held_token"] = _typed_tokens(
        left["held_out_subject"],
        allow_null=False,
        field="representations.held_out_subject",
    )
    left["_dual_donor_token"] = _donor_tokens(left["donor_subject"])
    left["_dual_k"] = _exact_k_values(left["k"], field="representations.k")
    right = baselines.loc[
        baselines["primitive"].eq(primitive),
        [
            "held_out_subject",
            "k",
            "method",
            "donor_subject",
            "center",
            "scale",
            "population_scale",
            "baseline_status",
        ],
    ].copy()
    right["_dual_held_token"] = _typed_tokens(
        right["held_out_subject"],
        allow_null=False,
        field="baselines.held_out_subject",
    )
    right["_dual_donor_token"] = _donor_tokens(right["donor_subject"])
    right["_dual_k"] = _exact_k_values(right["k"], field="baselines.k")
    right = right.drop(columns=["held_out_subject", "k", "donor_subject"]).rename(
        columns={
            "center": "_baseline_center",
            "scale": "_baseline_self_scale",
            "population_scale": "_baseline_population_scale",
            "baseline_status": "_baseline_status",
        }
    )
    merge_keys = [
        "_dual_held_token",
        "_dual_k",
        "method",
        "_dual_donor_token",
    ]
    if right.duplicated(merge_keys).any():
        raise ValueError(f"Baseline keys for {primitive} must be unique")
    merged = left.merge(
        right,
        on=merge_keys,
        how="left",
        sort=False,
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Every representation group requires a {primitive} baseline")
    merged = merged.sort_values("_dual_row_position", kind="stable").reset_index(
        drop=True
    )
    anchor = anchors.loc[
        anchors["primitive"].eq(primitive),
        ["_dual_held_token", "_authoritative_population_scale"],
    ]
    merged = merged.merge(
        anchor,
        on="_dual_held_token",
        how="left",
        sort=False,
        validate="many_to_one",
    ).sort_values("_dual_row_position", kind="stable").reset_index(drop=True)
    if not _numeric_equal(
        merged["_baseline_population_scale"],
        merged["_authoritative_population_scale"],
    ).all():
        raise ValueError(f"{primitive} population-scale provenance is inconsistent")
    return merged


def _source_matrices(
    frame: pd.DataFrame,
    sources: Sequence[tuple[str, str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    _require_columns(
        frame,
        tuple(column for source in sources for column in source[:2]),
        "semantic source",
    )
    values = np.column_stack(
        [
            pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
            * float(direction)
            for value_column, _, direction in sources
        ]
    )
    weights = np.column_stack(
        [
            pd.to_numeric(frame[confidence_column], errors="coerce").to_numpy(
                dtype=float
            )
            for _, confidence_column, _ in sources
        ]
    )
    invalid = np.isfinite(weights) & ((weights < 0.0) | (weights > 1.0))
    if bool(invalid.any()):
        raise ValueError("Semantic source confidence must be between zero and one")
    return values, weights


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    half = float(sorted_weights.sum()) / 2.0
    index = int(np.searchsorted(cumulative, half, side="left"))
    if np.isclose(cumulative[index], half, rtol=0.0, atol=1e-12):
        if index + 1 < len(sorted_values):
            return float((sorted_values[index] + sorted_values[index + 1]) / 2.0)
    return float(sorted_values[index])


def _summarize(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    operation: str,
    expected_source_count: int,
    minimum_sources: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    effective_weights = np.where(valid, weights, 0.0)
    effective_values = np.where(valid, values, 0.0)
    modalities = valid.sum(axis=1).astype(int)
    sufficient = modalities >= int(minimum_sources)
    weight_sum = effective_weights.sum(axis=1)
    output = np.full(len(values), np.nan, dtype=float)
    if operation == "mean":
        numerator = (effective_values * effective_weights).sum(axis=1)
        np.divide(
            numerator,
            weight_sum,
            out=output,
            where=sufficient & (weight_sum > 0.0),
        )
    elif operation == "rms":
        numerator = (np.square(effective_values) * effective_weights).sum(axis=1)
        mean_square = np.full(len(values), np.nan, dtype=float)
        np.divide(
            numerator,
            weight_sum,
            out=mean_square,
            where=sufficient & (weight_sum > 0.0),
        )
        output[sufficient] = np.sqrt(mean_square[sufficient])
    elif operation == "median":
        for row_index in np.flatnonzero(sufficient):
            selected = valid[row_index]
            output[row_index] = _weighted_median(
                values[row_index, selected], weights[row_index, selected]
            )
    else:
        raise ValueError(f"Unsupported semantic operation: {operation}")
    confidence = np.zeros(len(values), dtype=float)
    confidence[sufficient] = np.minimum(
        weight_sum[sufficient] / float(expected_source_count), 1.0
    )
    status = np.where(sufficient, "derived", "insufficient")
    return output, confidence, modalities, status


def _append_semantic(
    output: dict[str, Any],
    name: str,
    summary: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    consumer_role: str,
    coordinate_policy: str,
) -> None:
    value, confidence, modalities, status = summary
    output[name] = value
    output[f"{name}__confidence"] = confidence
    output[f"{name}__modalities"] = modalities
    output[f"{name}__status"] = status
    output[f"{name}__consumer_role"] = consumer_role
    output[f"{name}__coordinate_policy"] = coordinate_policy
    output[f"{name}__post_hoc_exploratory"] = True


def _validate_role_coordinate(consumer_role: str, coordinate: str) -> None:
    required = {
        "cross_modal": REFERENCE_COORDINATE,
        "routine_anomaly": SELF_COORDINATE,
        "raw_timing": RAW_TIMING_COORDINATE,
    }
    if consumer_role not in required:
        raise ValueError(f"Unknown coordinate consumer role: {consumer_role}")
    if coordinate != required[consumer_role]:
        raise ValueError(
            f"{consumer_role} consumers require {required[consumer_role]}; "
            f"received {coordinate}"
        )


def compose_cross_modal_semantics(
    coordinates: pd.DataFrame,
    *,
    coordinate: str = REFERENCE_COORDINATE,
) -> pd.DataFrame:
    """Compose cross-modal scores while rejecting self-coordinate requests."""
    _require_unique_columns(coordinates, "dual coordinates")
    _validate_role_coordinate("cross_modal", coordinate)
    suffix = f"__{coordinate}"
    groups = {
        "reference_digital_engagement_score": (
            (
                (f"screen_load_24h{suffix}", "screen_load_24h__confidence", 1.0),
                (f"usage_load_24h{suffix}", "usage_load_24h__confidence", 1.0),
            ),
            "mean",
        ),
        "reference_physical_activity_score": (
            (
                (
                    f"phone_activity_load_24h{suffix}",
                    "phone_activity_load_24h__confidence",
                    1.0,
                ),
                (f"step_load_24h{suffix}", "step_load_24h__confidence", 1.0),
            ),
            "mean",
        ),
        "reference_light_exposure_score": (
            (
                (
                    f"mobile_light_exposure_24h{suffix}",
                    "mobile_light_exposure_24h__confidence",
                    1.0,
                ),
                (
                    f"wearable_light_exposure_24h{suffix}",
                    "wearable_light_exposure_24h__confidence",
                    1.0,
                ),
            ),
            "mean",
        ),
        "reference_physiological_settling_score": (
            (
                (
                    f"heart_rate_settling_delta{suffix}",
                    "heart_rate_settling_delta__confidence",
                    -1.0,
                ),
                (
                    f"heart_rate_settling_slope{suffix}",
                    "heart_rate_settling_slope__confidence",
                    -1.0,
                ),
            ),
            "mean",
        ),
    }
    output: dict[str, Any] = {}
    for name, (sources, operation) in groups.items():
        values, weights = _source_matrices(coordinates, sources)
        _append_semantic(
            output,
            name,
            _summarize(
                values,
                weights,
                operation=operation,
                expected_source_count=2,
            ),
            consumer_role="cross_modal",
            coordinate_policy=REFERENCE_COORDINATE,
        )
    return pd.DataFrame(output, index=coordinates.index)


def compose_self_routine_deviation(
    coordinates: pd.DataFrame,
    *,
    coordinate: str = SELF_COORDINATE,
) -> pd.DataFrame:
    """Compose personal routine anomaly while rejecting reference coordinates."""
    _require_unique_columns(coordinates, "dual coordinates")
    _validate_role_coordinate("routine_anomaly", coordinate)
    suffix = f"__{coordinate}"
    component_groups: tuple[tuple[tuple[str, str, float], ...], ...] = (
        (
            (f"screen_load_24h{suffix}", "screen_load_24h__confidence", 1.0),
            (f"usage_load_24h{suffix}", "usage_load_24h__confidence", 1.0),
        ),
        (
            (
                f"screen_disengagement_p90{suffix}",
                "screen_disengagement_p90__confidence",
                1.0,
            ),
            (
                f"usage_disengagement_p90{suffix}",
                "usage_disengagement_p90__confidence",
                1.0,
            ),
        ),
        (
            (
                f"phone_activity_load_24h{suffix}",
                "phone_activity_load_24h__confidence",
                1.0,
            ),
            (f"step_load_24h{suffix}", "step_load_24h__confidence", 1.0),
        ),
        (
            (
                f"phone_activity_disengagement_p90{suffix}",
                "phone_activity_disengagement_p90__confidence",
                1.0,
            ),
            (
                f"step_disengagement_p90{suffix}",
                "step_disengagement_p90__confidence",
                1.0,
            ),
        ),
        (
            (
                f"mobile_light_exposure_24h{suffix}",
                "mobile_light_exposure_24h__confidence",
                1.0,
            ),
            (
                f"wearable_light_exposure_24h{suffix}",
                "wearable_light_exposure_24h__confidence",
                1.0,
            ),
        ),
        (
            (
                f"mobile_low_light_onset{suffix}",
                "mobile_low_light_onset__confidence",
                1.0,
            ),
            (
                f"wearable_low_light_onset{suffix}",
                "wearable_low_light_onset__confidence",
                1.0,
            ),
        ),
        (
            (
                f"heart_rate_settling_delta{suffix}",
                "heart_rate_settling_delta__confidence",
                -1.0,
            ),
            (
                f"heart_rate_settling_slope{suffix}",
                "heart_rate_settling_slope__confidence",
                -1.0,
            ),
        ),
    )
    component_values: list[np.ndarray] = []
    component_weights: list[np.ndarray] = []
    for sources in component_groups:
        values, weights = _source_matrices(coordinates, sources)
        summary = _summarize(
            values, weights, operation="mean", expected_source_count=2
        )
        component_values.append(summary[0])
        component_weights.append(summary[1])
    charging_values, charging_weights = _source_matrices(
        coordinates,
        ((f"charging_onset{suffix}", "charging_onset__confidence", 1.0),),
    )
    component_values.append(charging_values[:, 0])
    component_weights.append(charging_weights[:, 0])
    routine = _summarize(
        np.column_stack(component_values),
        np.column_stack(component_weights),
        operation="rms",
        expected_source_count=8,
    )
    output: dict[str, Any] = {}
    _append_semantic(
        output,
        "self_routine_deviation",
        routine,
        consumer_role="routine_anomaly",
        coordinate_policy=SELF_COORDINATE,
    )
    return pd.DataFrame(output, index=coordinates.index)


def compose_raw_timing_semantics(
    coordinates: pd.DataFrame,
    *,
    coordinate: str = RAW_TIMING_COORDINATE,
) -> pd.DataFrame:
    """Compose minute-of-day summaries without standardizing clock time."""
    _require_unique_columns(coordinates, "dual coordinates")
    _validate_role_coordinate("raw_timing", coordinate)
    groups = {
        "raw_timing_digital_disengagement_time": (
            (
                (
                    "screen_disengagement_p90",
                    "screen_disengagement_p90__confidence",
                    1.0,
                ),
                (
                    "usage_disengagement_p90",
                    "usage_disengagement_p90__confidence",
                    1.0,
                ),
            ),
            "median",
        ),
        "raw_timing_physical_disengagement_time": (
            (
                (
                    "phone_activity_disengagement_p90",
                    "phone_activity_disengagement_p90__confidence",
                    1.0,
                ),
                (
                    "step_disengagement_p90",
                    "step_disengagement_p90__confidence",
                    1.0,
                ),
            ),
            "median",
        ),
        "raw_timing_low_light_onset": (
            (
                (
                    "mobile_low_light_onset",
                    "mobile_low_light_onset__confidence",
                    1.0,
                ),
                (
                    "wearable_low_light_onset",
                    "wearable_low_light_onset__confidence",
                    1.0,
                ),
            ),
            "median",
        ),
    }
    output: dict[str, Any] = {}
    for name, (sources, operation) in groups.items():
        values, weights = _source_matrices(coordinates, sources)
        _append_semantic(
            output,
            name,
            _summarize(
                values,
                weights,
                operation=operation,
                expected_source_count=2,
            ),
            consumer_role="raw_timing",
            coordinate_policy=RAW_TIMING_COORDINATE,
        )

    winddown_values = np.column_stack(
        [
            output["raw_timing_digital_disengagement_time"],
            output["raw_timing_physical_disengagement_time"],
            output["raw_timing_low_light_onset"],
            pd.to_numeric(coordinates["charging_onset"], errors="coerce").to_numpy(
                dtype=float
            ),
        ]
    )
    winddown_weights = np.column_stack(
        [
            output["raw_timing_digital_disengagement_time__confidence"],
            output["raw_timing_physical_disengagement_time__confidence"],
            output["raw_timing_low_light_onset__confidence"],
            pd.to_numeric(
                coordinates["charging_onset__confidence"], errors="coerce"
            ).to_numpy(dtype=float),
        ]
    )
    center = _summarize(
        winddown_values,
        winddown_weights,
        operation="median",
        expected_source_count=4,
    )
    _append_semantic(
        output,
        "raw_timing_winddown_center",
        center,
        consumer_role="raw_timing",
        coordinate_policy=RAW_TIMING_COORDINATE,
    )
    valid = (
        np.isfinite(winddown_values)
        & np.isfinite(winddown_weights)
        & (winddown_weights > 0.0)
    )
    spread = _summarize(
        np.abs(winddown_values - center[0][:, None]),
        winddown_weights,
        operation="median",
        expected_source_count=4,
        minimum_sources=2,
    )
    spread = (spread[0], spread[1], valid.sum(axis=1).astype(int), spread[3])
    _append_semantic(
        output,
        "raw_timing_winddown_spread",
        spread,
        consumer_role="raw_timing",
        coordinate_policy=RAW_TIMING_COORDINATE,
    )
    return pd.DataFrame(output, index=coordinates.index)


def build_dual_calibration_coordinates(
    primitives: pd.DataFrame,
    loso_tables: LosoRepresentationTables,
) -> DualCalibrationTables:
    """Generate fixed dual coordinates without modifying frozen Task 3 tables."""
    representations, baselines = _prepare_inputs(primitives, loso_tables)
    anchors = _authoritative_population_scales(baselines)

    output: dict[str, Any] = {
        column: representations[column].to_numpy(copy=True)
        for column in _ROW_METADATA
    }
    output["dual_calibration_protocol"] = np.repeat(
        DUAL_CALIBRATION_PROTOCOL, len(representations)
    )
    output["dual_calibration_post_hoc_exploratory"] = np.repeat(
        True, len(representations)
    )

    for primitive in _PRIMITIVE_NAMES:
        baseline = _baseline_for_primitive(
            representations, baselines, anchors, primitive
        )
        if not _numeric_equal(
            representations[f"{primitive}__center"], baseline["_baseline_center"]
        ).all():
            raise ValueError(f"{primitive} Task 3 center disagrees with its baseline")
        if not _numeric_equal(
            representations[f"{primitive}__scale"],
            baseline["_baseline_self_scale"],
        ).all():
            raise ValueError(f"{primitive} Task 3 scale disagrees with its baseline")
        if not _object_equal(
            representations[f"{primitive}__baseline_status"],
            baseline["_baseline_status"],
        ).all():
            raise ValueError(f"{primitive} Task 3 status disagrees with its baseline")

        values = pd.to_numeric(representations[primitive], errors="coerce").to_numpy(
            dtype=float
        )
        confidence = pd.to_numeric(
            representations[f"{primitive}__confidence"], errors="coerce"
        ).to_numpy(dtype=float)
        invalid_confidence = np.isfinite(confidence) & (
            (confidence < 0.0) | (confidence > 1.0)
        )
        if bool(invalid_confidence.any()):
            raise ValueError(f"{primitive} confidence must be between zero and one")
        center = pd.to_numeric(
            baseline["_baseline_center"], errors="coerce"
        ).to_numpy(dtype=float)
        self_scale = pd.to_numeric(
            baseline["_baseline_self_scale"], errors="coerce"
        ).to_numpy(dtype=float)
        population_scale = pd.to_numeric(
            baseline["_authoritative_population_scale"], errors="coerce"
        ).to_numpy(dtype=float)
        status_observed = baseline["_baseline_status"].astype("string").eq(
            "observed"
        ).fillna(False).to_numpy(dtype=bool)
        common = (
            np.isfinite(values)
            & np.isfinite(confidence)
            & (confidence > 0.0)
            & np.isfinite(center)
            & status_observed
        )
        reference_usable = common & np.isfinite(population_scale) & (
            population_scale > 0.0
        )
        self_usable = common & np.isfinite(self_scale) & (self_scale > 0.0)
        reference_unclipped = np.full(len(representations), np.nan, dtype=float)
        self_unclipped = np.full(len(representations), np.nan, dtype=float)
        reference_unclipped[reference_usable] = (
            values[reference_usable] - center[reference_usable]
        ) / population_scale[reference_usable]
        self_unclipped[self_usable] = (
            values[self_usable] - center[self_usable]
        ) / self_scale[self_usable]

        output[primitive] = values
        for confidence_suffix in (
            "confidence",
            "coverage_only_confidence",
            "value_only_confidence",
        ):
            output[f"{primitive}__{confidence_suffix}"] = pd.to_numeric(
                representations[f"{primitive}__{confidence_suffix}"],
                errors="coerce",
            ).to_numpy(dtype=float)
        output[f"{primitive}__reference_z_unclipped"] = reference_unclipped
        output[f"{primitive}__reference_z"] = np.clip(
            reference_unclipped, -CLIP_LIMIT, CLIP_LIMIT
        )
        output[f"{primitive}__self_z_unclipped"] = self_unclipped
        output[f"{primitive}__self_z"] = np.clip(
            self_unclipped, -CLIP_LIMIT, CLIP_LIMIT
        )
        output[f"{primitive}__population_scale"] = population_scale

    coordinate_frame = pd.DataFrame(output)
    # Preserve the frozen key dtypes and null representation exactly; the
    # DataFrame constructor may otherwise infer nullable strings from an
    # object donor column and turn Task 3 ``None`` values into ``pd.NA``.
    for column in REPRESENTATION_KEYS:
        coordinate_frame[column] = representations[column].reset_index(
            drop=True
        ).copy(deep=True)
    semantic_frames = (
        compose_cross_modal_semantics(coordinate_frame),
        compose_self_routine_deviation(coordinate_frame),
        compose_raw_timing_semantics(coordinate_frame),
    )
    coordinates = pd.concat(
        [coordinate_frame, *semantic_frames], axis=1
    ).reset_index(drop=True)
    if coordinates.duplicated(list(REPRESENTATION_KEYS)).any():
        raise RuntimeError("Dual calibration coordinate keys are not unique")
    if len(coordinates) != len(representations):
        raise RuntimeError("Dual calibration changed the frozen representation row grid")
    return DualCalibrationTables(
        coordinates=coordinates,
        coordinate_policy=coordinate_policy_table(),
    )
