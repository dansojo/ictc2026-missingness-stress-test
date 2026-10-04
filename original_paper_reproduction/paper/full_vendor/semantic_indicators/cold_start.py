"""Frozen participant-level cold-start transfer evaluation utilities.

This module is deliberately separate from the post-hoc dual-calibration layer.
It consumes only frozen Task 3 artifacts and evaluation-only oracle tables.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import PRIMITIVE_DEFINITIONS
from .dual_calibration import (
    DUAL_CALIBRATION_PROTOCOL,
    DualCalibrationTables,
    coordinate_policy_table,
)
from .personalization import (
    LosoRepresentationTables,
    build_loso_representations,
    robust_location_scale,
    shrink_location_scale,
)


DEFAULT_ROOT_SEED = 42
DEFAULT_DERANGEMENT_DRAWS = 10_000
DEFAULT_SHIFT_DRAWS = 10_000
FROZEN_PARTICIPANT_DENOMINATOR = 10
FROZEN_K_VALUES = (3, 7, 14, 21)
PRIMARY_K = 14
PRIMARY_LAMBDA_DAYS = 7.0

FROZEN_DOMAIN_PRIMITIVES = {
    "digital": (
        "screen_load_24h",
        "usage_load_24h",
        "screen_disengagement_p90",
        "usage_disengagement_p90",
    ),
    "physical": ("phone_activity_load_24h", "step_load_24h"),
    "light": ("mobile_light_exposure_24h", "wearable_light_exposure_24h"),
}
FROZEN_PAIR_SOURCES = {
    "digital_load": ("digital", "screen_load_24h", "usage_load_24h"),
    "digital_disengagement": (
        "digital",
        "screen_disengagement_p90",
        "usage_disengagement_p90",
    ),
    "physical_load": (
        "physical",
        "phone_activity_load_24h",
        "step_load_24h",
    ),
    "light_exposure": (
        "light",
        "mobile_light_exposure_24h",
        "wearable_light_exposure_24h",
    ),
}
SPECIFICITY_PAIR_MATRIX = (
    ("digital_load", "screen_load_24h", "usage_load_24h", True),
    ("screen_step", "screen_load_24h", "step_load_24h", False),
    (
        "screen_wearable_light",
        "screen_load_24h",
        "wearable_light_exposure_24h",
        False,
    ),
    (
        "phone_activity_usage",
        "phone_activity_load_24h",
        "usage_load_24h",
        False,
    ),
    ("physical_load", "phone_activity_load_24h", "step_load_24h", True),
    (
        "phone_activity_wearable_light",
        "phone_activity_load_24h",
        "wearable_light_exposure_24h",
        False,
    ),
    (
        "mobile_light_usage",
        "mobile_light_exposure_24h",
        "usage_load_24h",
        False,
    ),
    (
        "mobile_light_step",
        "mobile_light_exposure_24h",
        "step_load_24h",
        False,
    ),
    (
        "light_exposure",
        "mobile_light_exposure_24h",
        "wearable_light_exposure_24h",
        True,
    ),
)
_ALL_PRIMITIVES = tuple(definition.name for definition in PRIMITIVE_DEFINITIONS)
_CORE_PRIMITIVES = tuple(
    primitive
    for domain in FROZEN_DOMAIN_PRIMITIVES
    for primitive in FROZEN_DOMAIN_PRIMITIVES[domain]
)
_PRIMITIVE_DOMAIN = {
    primitive: domain
    for domain, primitives in FROZEN_DOMAIN_PRIMITIVES.items()
    for primitive in primitives
}
_NULL_DONOR = "__FROZEN_G3_NULL_DONOR__"


@dataclass(frozen=True)
class SignFlipResult:
    """Exact sign-flip distribution and its one-sided inclusive-tail summary."""

    distribution: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class ExactMaxStatResult:
    """Simultaneous exact participant sign-permutation max-stat audit."""

    candidate_summary: pd.DataFrame
    permutation_maxima: pd.DataFrame
    candidate_permutation_statistics: pd.DataFrame


@dataclass(frozen=True)
class FrozenColdStartResult:
    """Auditable tables for the immutable frozen Task 3 G3 evaluation."""

    primitive_transfer_errors: pd.DataFrame
    domain_transfer_errors: pd.DataFrame
    participant_transfer_deltas: pd.DataFrame
    method_transfer_summaries: pd.DataFrame
    pair_day_discrepancies: pd.DataFrame
    pair_discrepancy_summaries: pd.DataFrame
    participant_discrepancy_deltas: pd.DataFrame
    sign_flip_distribution: pd.DataFrame
    sign_flip_summary: pd.DataFrame
    coherent_derangement_assignments: pd.DataFrame
    coherent_derangement_distribution: pd.DataFrame
    coherent_derangement_summary: pd.DataFrame
    lopo_summary: pd.DataFrame
    frozen_gate_decision: pd.DataFrame


@dataclass(frozen=True)
class ExploratoryCandidateBundle:
    """One outcome-free, predeclared Task 3 candidate construction."""

    bundle_id: str
    lambda_days: float
    usage_confidence_policy: str
    usage_operator: str
    quality_threshold: float
    population_weighting: str
    loso_tables: LosoRepresentationTables
    dual_tables: Any | None
    source_calendar_sha256: str


@dataclass(frozen=True)
class ExploratoryLocationResult:
    """Auditable post-hoc G3-L location-transfer outputs."""

    candidate_ledger: pd.DataFrame
    primitive_location_errors: pd.DataFrame
    domain_location_errors: pd.DataFrame
    participant_location_deltas: pd.DataFrame
    location_curve_summary: pd.DataFrame
    location_lopo_summary: pd.DataFrame
    coherent_derangement_assignments: pd.DataFrame
    coherent_derangement_distribution: pd.DataFrame
    coherent_derangement_summary: pd.DataFrame
    max_stat_candidate_summary: pd.DataFrame
    max_stat_permutation_maxima: pd.DataFrame
    max_stat_candidate_permutation_statistics: pd.DataFrame
    location_gate_decision: pd.DataFrame


@dataclass(frozen=True)
class ExploratoryReferenceResult:
    """Auditable post-hoc G3-R reference-alignment outputs."""

    candidate_ledger: pd.DataFrame
    pair_day_discrepancies: pd.DataFrame
    pair_discrepancy_summaries: pd.DataFrame
    participant_reference_deltas: pd.DataFrame
    reference_lopo_summary: pd.DataFrame
    coherent_derangement_assignments: pd.DataFrame
    coherent_derangement_distribution: pd.DataFrame
    coherent_derangement_summary: pd.DataFrame
    max_stat_candidate_summary: pd.DataFrame
    max_stat_permutation_maxima: pd.DataFrame
    max_stat_candidate_permutation_statistics: pd.DataFrame
    reference_gate_decision: pd.DataFrame


@dataclass(frozen=True)
class CalendarShiftResult:
    """One frozen-calibration calendar-shift negative-control family."""

    shift_assignments: pd.DataFrame
    null_distribution: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class SpecificityResult:
    """Fixed 3x3 unrelated-pair specificity outputs."""

    candidate_ledger: pd.DataFrame
    pair_matrix: pd.DataFrame
    contrast_table: pd.DataFrame
    intended_pair_summary: pd.DataFrame
    participant_specificity: pd.DataFrame
    lopo_summary: pd.DataFrame
    exact_sign_flip_distribution: pd.DataFrame
    exact_sign_flip_summary: pd.DataFrame
    max_stat_candidate_summary: pd.DataFrame
    max_stat_permutation_maxima: pd.DataFrame
    max_stat_candidate_permutation_statistics: pd.DataFrame
    specificity_gate_decision: pd.DataFrame


@dataclass(frozen=True)
class ExploratoryDualColdStartResult:
    """Complete post-hoc dual analysis with the frozen result retained verbatim."""

    frozen_result: FrozenColdStartResult
    location: ExploratoryLocationResult
    reference: ExploratoryReferenceResult
    unrestricted_shift: CalendarShiftResult
    weekday_shift: CalendarShiftResult
    specificity: SpecificityResult
    scale_self_diagnostics: pd.DataFrame
    shift_summaries: pd.DataFrame
    joint_decision: pd.DataFrame


def exploratory_candidate_ledger() -> pd.DataFrame:
    """Return the immutable, result-independent exploratory candidate universe."""
    calibration_modes = ("full", "location_only", "scale_only")
    k_values = (3, 7, 14, 21)
    lambda_values = (3.0, 7.0, 14.0)
    usage_confidence_policies = ("screen_proxy", "unweighted", "value_only")
    usage_operators = ("frozen", "capped", "max_app", "record_mean")
    quality_thresholds = (0.0, 0.5, 0.7, 0.8, 0.9)
    clipping_states = ("clipped", "unclipped")
    population_weightings = ("row_weighted", "participant_balanced")
    records: list[dict[str, Any]] = []
    candidate_index = 0
    for calibration_mode in calibration_modes:
        for k in k_values:
            for lambda_days in lambda_values:
                for usage_confidence_policy in usage_confidence_policies:
                    for usage_operator in usage_operators:
                        for quality_threshold in quality_thresholds:
                            for clipping_state in clipping_states:
                                for population_weighting in population_weightings:
                                    records.append(
                                        {
                                            "candidate_id": f"dual_{candidate_index:05d}",
                                            "calibration_mode": calibration_mode,
                                            "k": k,
                                            "lambda_days": lambda_days,
                                            "usage_confidence_policy": usage_confidence_policy,
                                            "usage_operator": usage_operator,
                                            "quality_threshold": quality_threshold,
                                            "clipping_state": clipping_state,
                                            "population_weighting": population_weighting,
                                            "oracle_diagnostic": False,
                                            "deployable": True,
                                            "post_hoc_exploratory": True,
                                        }
                                    )
                                    candidate_index += 1
    records.append(
        {
            "candidate_id": "oracle_diagnostic",
            "calibration_mode": "location_only",
            "k": 14,
            "lambda_days": 7.0,
            "usage_confidence_policy": "screen_proxy",
            "usage_operator": "frozen",
            "quality_threshold": 0.0,
            "clipping_state": "clipped",
            "population_weighting": "row_weighted",
            "oracle_diagnostic": True,
            "deployable": False,
            "post_hoc_exploratory": True,
        }
    )
    return pd.DataFrame.from_records(records)


def _scale_invariant_studentized_mean(values: np.ndarray) -> float:
    """Studentize after max-absolute scaling, including exact degenerate policy."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not bool(np.isfinite(array).all()):
        return float("nan")
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return 0.0
    scaled = array / maximum
    mean = float(np.mean(scaled))
    standard_deviation = float(np.std(scaled, ddof=1))
    if standard_deviation == 0.0:
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0
    return float(mean / (standard_deviation / np.sqrt(array.size)))


def exact_max_stat_sign_flip(
    candidate_deltas: pd.DataFrame,
    *,
    candidate_column: str = "candidate_id",
    participant_column: str = "held_out_subject",
    delta_column: str = "improvement",
    expected_participants: int = FROZEN_PARTICIPANT_DENOMINATOR,
) -> ExactMaxStatResult:
    """Correct a fixed candidate family with all common participant sign vectors."""
    if isinstance(expected_participants, bool) or not isinstance(
        expected_participants, (int, np.integer)
    ):
        raise ValueError("expected_participants must be an exact positive integer")
    expected = int(expected_participants)
    if expected <= 0:
        raise ValueError("expected_participants must be an exact positive integer")
    if not isinstance(candidate_deltas, pd.DataFrame):
        raise TypeError("candidate_deltas must be a pandas DataFrame")
    if candidate_deltas.empty:
        raise ValueError("candidate_deltas must be nonempty")
    if candidate_deltas.columns.duplicated().any():
        raise ValueError("candidate_deltas columns must be unique")
    _require_columns(
        candidate_deltas,
        (candidate_column, participant_column, delta_column),
        "candidate delta",
    )
    forbidden = _forbidden_representation_columns(candidate_deltas)
    allowed = {candidate_column, participant_column, delta_column}
    forbidden = [column for column in forbidden if column not in allowed]
    if forbidden:
        raise ValueError(
            "Candidate delta table contains forbidden oracle, outcome, or argmax "
            f"columns: {forbidden}"
        )
    frame = candidate_deltas.loc[
        :, [candidate_column, participant_column, delta_column]
    ].copy(deep=True)
    if frame[[candidate_column, participant_column]].isna().any().any():
        raise ValueError("Candidate and participant identities must be nonmissing")
    frame["_candidate_token"] = frame[candidate_column].map(_identity_token)
    frame["_participant_token"] = frame[participant_column].map(_identity_token)
    if frame.duplicated(["_candidate_token", "_participant_token"]).any():
        raise ValueError("Candidate-participant rows must be unique")
    participants = (
        frame.loc[:, [participant_column, "_participant_token"]]
        .drop_duplicates("_participant_token")
        .sort_values("_participant_token", kind="stable")
        .reset_index(drop=True)
    )
    if len(participants) != expected:
        raise ValueError(f"Max-stat sign flip requires exactly {expected} participants")
    participant_tokens = participants["_participant_token"].tolist()
    participant_order = json.dumps(
        participant_tokens, ensure_ascii=False, separators=(",", ":")
    )
    candidate_identity = (
        frame.loc[:, [candidate_column, "_candidate_token"]]
        .drop_duplicates("_candidate_token")
        .sort_values("_candidate_token", kind="stable")
        .reset_index(drop=True)
    )
    candidate_tokens = candidate_identity["_candidate_token"].tolist()
    candidate_values: dict[str, np.ndarray] = {}
    for candidate_token in candidate_tokens:
        group = frame.loc[frame["_candidate_token"].eq(candidate_token)].sort_values(
            "_participant_token", kind="stable"
        )
        if len(group) != expected or group["_participant_token"].tolist() != participant_tokens:
            raise ValueError(f"Every candidate requires exactly {expected} participants")
        values = pd.to_numeric(group[delta_column], errors="coerce").to_numpy(
            dtype=float
        )
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"Every candidate requires exactly {expected} finite deltas")
        candidate_values[candidate_token] = values

    pattern_count = 1 << expected
    candidate_ids = candidate_identity[candidate_column].tolist()
    raw_matrix = np.vstack(
        [candidate_values[candidate_token] for candidate_token in candidate_tokens]
    )
    row_maximum = np.max(np.abs(raw_matrix), axis=1)
    normalized = np.zeros_like(raw_matrix)
    nonzero = row_maximum > 0.0
    normalized[nonzero] = raw_matrix[nonzero] / row_maximum[nonzero, None]
    pattern_ids = np.arange(pattern_count, dtype=np.uint64)
    bit_values = np.left_shift(np.uint64(1), np.arange(expected, dtype=np.uint64))
    sign_matrix = np.where(
        (pattern_ids[:, None] & bit_values[None, :]) != 0,
        1.0,
        -1.0,
    )
    signed = normalized[:, None, :] * sign_matrix[None, :, :]
    signed_means = np.mean(signed, axis=2)
    signed_standard_deviations = np.std(signed, axis=2, ddof=1)
    statistics = np.empty_like(signed_means)
    regular = signed_standard_deviations > 0.0
    statistics[regular] = signed_means[regular] / (
        signed_standard_deviations[regular] / np.sqrt(expected)
    )
    degenerate = ~regular
    statistics[degenerate & (signed_means > 0.0)] = np.inf
    statistics[degenerate & (signed_means < 0.0)] = -np.inf
    statistics[degenerate & (signed_means == 0.0)] = 0.0
    maxima = np.max(statistics, axis=0)
    sign_patterns = [
        "".join("+" if sign > 0 else "-" for sign in signs)
        for signs in sign_matrix
    ]
    permutation_records = [
        {
            "pattern_id": pattern_id,
            "sign_pattern": sign_patterns[pattern_id],
            "participant_order": participant_order,
            "max_studentized_mean": float(maxima[pattern_id]),
            "participant_denominator": expected,
            "candidate_denominator": len(candidate_tokens),
            "post_hoc_exploratory": True,
            "status": "observed",
        }
        for pattern_id in range(pattern_count)
    ]
    statistic_records = [
        {
            "pattern_id": pattern_id,
            "sign_pattern": sign_patterns[pattern_id],
            "participant_order": participant_order,
            "candidate_id": candidate_ids[candidate_index],
            "studentized_mean": float(statistics[candidate_index, pattern_id]),
            "participant_denominator": expected,
            "post_hoc_exploratory": True,
            "status": "observed",
        }
        for pattern_id in range(pattern_count)
        for candidate_index in range(len(candidate_tokens))
    ]

    summary_records: list[dict[str, Any]] = []
    observed_statistics = statistics[:, pattern_count - 1]
    for candidate_index, (_, row) in enumerate(candidate_identity.iterrows()):
        candidate_id = row[candidate_column]
        candidate_token = row["_candidate_token"]
        observed = float(observed_statistics[candidate_index])
        adjusted_p = float(np.mean(maxima >= observed))
        summary_records.append(
            {
                "candidate_id": candidate_id,
                "observed_mean_improvement": float(
                    np.mean(candidate_values[candidate_token])
                ),
                "observed_statistic": observed,
                "adjusted_p_value": adjusted_p,
                "tail_convention": "exact_max_stat_inclusive",
                "alternative": "larger_improvement",
                "sign_patterns": pattern_count,
                "participant_denominator": expected,
                "candidate_denominator": len(candidate_tokens),
                "participant_order": participant_order,
                "post_hoc_exploratory": True,
                "status": "observed",
                "_candidate_token": candidate_token,
            }
        )
    candidate_summary = pd.DataFrame.from_records(summary_records).sort_values(
        "_candidate_token", kind="stable"
    ).drop(columns="_candidate_token").reset_index(drop=True)
    candidate_statistics = pd.DataFrame.from_records(statistic_records)
    candidate_statistics["_candidate_token"] = candidate_statistics[
        "candidate_id"
    ].map(_identity_token)
    candidate_statistics = candidate_statistics.sort_values(
        ["pattern_id", "_candidate_token"], kind="stable"
    ).drop(columns="_candidate_token").reset_index(drop=True)
    return ExactMaxStatResult(
        candidate_summary=candidate_summary,
        permutation_maxima=pd.DataFrame.from_records(permutation_records),
        candidate_permutation_statistics=candidate_statistics,
    )


def _identity_token(value: Any) -> str:
    value_type = type(value)
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        payload = repr(value)
    return f"{value_type.__module__}.{value_type.__qualname__}:{payload}"


def standardized_sign_agreement(left: float, right: float) -> bool | float:
    """Return sign equality; zero agrees only with zero, never with nonzero."""
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        return float("nan")
    return bool(np.sign(left_value) == np.sign(right_value))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} columns must be unique")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing {name} columns: {missing}")


def _donor_token(value: Any) -> str:
    return _NULL_DONOR if pd.isna(value) else _identity_token(value)


def _exact_k(series: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if (
        numeric.isna().any()
        or not np.isfinite(numeric).all()
        or not numeric.eq(np.floor(numeric)).all()
    ):
        raise ValueError(f"{name} must contain exact integers")
    return numeric.astype(int)


def _forbidden_representation_columns(frame: pd.DataFrame) -> list[str]:
    forbidden: list[str] = []
    for column in frame.columns:
        text = str(column)
        lower = text.lower()
        if (
            re.fullmatch(r"[qs]\d+", lower)
            or "oracle" in lower
            or "__reference_z" in lower
            or "__self_z" in lower
            or "dual_calibration" in lower
            or "argmax" in lower
            or "selected_candidate" in lower
            or "selected_coordinate" in lower
            or "outcome" in lower
            or lower == "label"
            or lower.startswith("label_")
        ):
            forbidden.append(text)
    return forbidden


def _group_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["_held_token"] = frame["held_out_subject"].map(_identity_token)
    result["_k"] = _exact_k(frame["k"], name="k")
    result["_method"] = frame["method"].astype(str)
    result["_donor_token"] = frame["donor_subject"].map(_donor_token)
    return result


def _group_tuples(frame: pd.DataFrame) -> list[tuple[str, int, str, str]]:
    keys = _group_key_frame(frame)
    return list(
        zip(
            keys["_held_token"],
            keys["_k"],
            keys["_method"],
            keys["_donor_token"],
            strict=True,
        )
    )


def _numeric_equal(left: pd.Series, right: pd.Series) -> np.ndarray:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    return (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=0.0, atol=1e-12)


def _scalar_numeric_equal(left: Any, right: Any) -> bool:
    """Compare one provenance scalar with the frozen numeric tolerance."""
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    return bool(
        np.isclose(left_value, right_value, rtol=0.0, atol=1e-12)
    )


def _exact_nonnegative_support(value: Any, *, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Task 3 baseline provenance {field} must be finite and nonnegative"
        ) from error
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            f"Task 3 baseline provenance {field} must be finite and nonnegative"
        )
    return numeric


def _exact_nonnegative_rows(value: Any, *, field: str) -> int:
    numeric = _exact_nonnegative_support(value, field=field)
    if not numeric.is_integer():
        raise ValueError(
            f"Task 3 baseline provenance {field} must contain exact integers"
        )
    return int(numeric)


def _expected_task3_baseline(row: pd.Series) -> dict[str, Any]:
    """Recompute the public Task 3 method rule from its frozen audit fields."""
    method = str(row["method"])
    population_center = float(row["population_center"])
    population_scale = float(row["population_scale"])
    calibration_center = float(row["calibration_center"])
    calibration_scale = float(row["calibration_scale"])
    calibration_n_eff = float(row["calibration_n_eff"])
    lambda_days = float(row["lambda_days"])
    population_valid = bool(
        np.isfinite(population_center)
        and np.isfinite(population_scale)
        and population_scale > 0.0
    )
    if method == "population":
        return {
            "center": population_center if population_valid else np.nan,
            "scale": population_scale if population_valid else np.nan,
            "center_weight": 0.0 if population_valid else np.nan,
            "scale_weight": 0.0 if population_valid else np.nan,
            "scale_floor": 0.1 * population_scale if population_valid else np.nan,
            "baseline_status": "observed" if population_valid else "degenerate",
            "scale_fallback": "none" if population_valid else "population_degenerate",
        }
    if method == "personal":
        if not population_valid:
            return {
                "center": np.nan,
                "scale": np.nan,
                "center_weight": np.nan,
                "scale_weight": np.nan,
                "scale_floor": np.nan,
                "baseline_status": "degenerate",
                "scale_fallback": "population_degenerate",
            }
        scale_floor = 0.1 * population_scale
        if calibration_n_eff <= 0.0 or not np.isfinite(calibration_center):
            return {
                "center": np.nan,
                "scale": np.nan,
                "center_weight": 1.0,
                "scale_weight": 0.0,
                "scale_floor": scale_floor,
                "baseline_status": "insufficient",
                "scale_fallback": "no_personal_center",
            }
        if calibration_n_eff >= 3.0 and np.isfinite(calibration_scale):
            scale = max(calibration_scale, scale_floor)
            scale_weight = 1.0
            fallback = "scale_floor" if calibration_scale < scale_floor else "none"
        else:
            scale = population_scale
            scale_weight = 0.0
            fallback = "population_insufficient_n_eff"
        return {
            "center": calibration_center,
            "scale": float(scale),
            "center_weight": 1.0,
            "scale_weight": scale_weight,
            "scale_floor": scale_floor,
            "baseline_status": "observed",
            "scale_fallback": fallback,
        }
    if method not in {"shrinkage", "placebo"}:
        raise ValueError(f"Task 3 baseline provenance has unsupported method: {method}")
    estimate = shrink_location_scale(
        personal_center=calibration_center,
        personal_scale=calibration_scale,
        population_center=population_center,
        population_scale=population_scale,
        n_eff=calibration_n_eff,
        lambda_days=lambda_days,
    )
    if estimate["status"] == "degenerate":
        fallback = "population_degenerate"
    elif calibration_n_eff < 3.0:
        fallback = "population_insufficient_n_eff"
    else:
        unbounded_scale = (
            float(estimate["scale_weight"]) * calibration_scale
            + (1.0 - float(estimate["scale_weight"])) * population_scale
        )
        fallback = (
            "scale_floor"
            if unbounded_scale < float(estimate["scale_floor"])
            else "none"
        )
    return {
        **estimate,
        "baseline_status": estimate["status"],
        "scale_fallback": fallback,
    }


def _validate_task3_baseline_provenance(baselines: pd.DataFrame) -> None:
    """Reject Task 3 tables whose redundant audit fields cannot create them."""
    required = (
        "method",
        "center",
        "scale",
        "n_eff",
        "contributing_rows",
        "population_center",
        "population_scale",
        "population_n_eff",
        "population_contributing_rows",
        "calibration_center",
        "calibration_scale",
        "calibration_n_eff",
        "calibration_contributing_rows",
        "center_weight",
        "scale_weight",
        "scale_floor",
        "scale_fallback",
        "baseline_status",
        "lambda_days",
    )
    _require_columns(baselines, required, "Task 3 baseline provenance")
    provenance = baselines.copy(deep=True)
    _require_columns(
        provenance,
        ("held_out_subject", "k", "donor_subject", "primitive"),
        "Task 3 baseline provenance key",
    )
    provenance["_provenance_held"] = provenance["held_out_subject"].map(
        _identity_token
    )
    provenance["_provenance_k"] = _exact_k(
        provenance["k"], name="Task 3 baseline provenance k"
    )
    anchors = provenance.loc[
        provenance["method"].astype(str).eq("population")
        & provenance["_provenance_k"].eq(0)
        & provenance["donor_subject"].isna(),
        [
            "_provenance_held",
            "primitive",
            "population_center",
            "population_scale",
            "population_n_eff",
            "population_contributing_rows",
        ],
    ].copy()
    if anchors.duplicated(["_provenance_held", "primitive"]).any():
        raise ValueError("Task 3 baseline provenance population anchors must be unique")
    anchors = anchors.rename(
        columns={
            column: f"_anchor_{column}"
            for column in (
                "population_center",
                "population_scale",
                "population_n_eff",
                "population_contributing_rows",
            )
        }
    )
    audited = provenance.merge(
        anchors,
        on=["_provenance_held", "primitive"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(audited) != len(provenance):
        raise ValueError("Task 3 baseline provenance population anchors are incomplete")
    for field in (
        "population_center",
        "population_scale",
        "population_n_eff",
        "population_contributing_rows",
    ):
        if not _numeric_equal(audited[field], audited[f"_anchor_{field}"]).all():
            raise ValueError(
                f"Task 3 baseline provenance {field} disagrees with its population anchor"
            )
    numeric_outputs = (
        "center",
        "scale",
        "center_weight",
        "scale_weight",
        "scale_floor",
    )
    for row_index, row in baselines.iterrows():
        method = str(row["method"])
        population_n_eff = _exact_nonnegative_support(
            row["population_n_eff"], field="population_n_eff"
        )
        population_rows = _exact_nonnegative_rows(
            row["population_contributing_rows"],
            field="population_contributing_rows",
        )
        calibration_n_eff = _exact_nonnegative_support(
            row["calibration_n_eff"], field="calibration_n_eff"
        )
        calibration_rows = _exact_nonnegative_rows(
            row["calibration_contributing_rows"],
            field="calibration_contributing_rows",
        )
        if population_n_eff > population_rows + 1e-12:
            raise ValueError(
                "Task 3 baseline provenance population n_eff cannot exceed rows"
            )
        if calibration_n_eff > calibration_rows + 1e-12:
            raise ValueError(
                "Task 3 baseline provenance calibration n_eff cannot exceed rows"
            )
        expected_n_eff = population_n_eff if method == "population" else calibration_n_eff
        expected_rows = population_rows if method == "population" else calibration_rows
        if not _scalar_numeric_equal(row["n_eff"], expected_n_eff):
            raise ValueError(
                "Task 3 baseline provenance n_eff disagrees with its source estimate"
            )
        if _exact_nonnegative_rows(
            row["contributing_rows"], field="contributing_rows"
        ) != expected_rows:
            raise ValueError(
                "Task 3 baseline provenance contributing_rows disagrees with its source"
            )
        if method == "population":
            for actual, source, field in (
                (row["calibration_center"], row["population_center"], "center"),
                (row["calibration_scale"], row["population_scale"], "scale"),
                (calibration_n_eff, population_n_eff, "n_eff"),
                (calibration_rows, population_rows, "contributing_rows"),
            ):
                if not _scalar_numeric_equal(actual, source):
                    raise ValueError(
                        "Task 3 population baseline provenance requires identical "
                        f"calibration and population {field}"
                    )
        if calibration_n_eff == 0.0 and (
            not pd.isna(row["calibration_center"])
            or not pd.isna(row["calibration_scale"])
            or calibration_rows != 0
        ):
            raise ValueError(
                "Task 3 zero-support calibration provenance requires NaN center/scale "
                "and zero contributing rows"
            )
        expected = _expected_task3_baseline(row)
        for field in numeric_outputs:
            if not _scalar_numeric_equal(row[field], expected[field]):
                raise ValueError(
                    "Task 3 baseline provenance formula mismatch for "
                    f"{field} at row {row_index}"
                )
        for field in ("baseline_status", "scale_fallback"):
            if str(row[field]) != str(expected[field]):
                raise ValueError(
                    "Task 3 baseline provenance formula mismatch for "
                    f"{field} at row {row_index}"
                )


def _authoritative_numeric_equal(actual: Any, expected: Any) -> bool:
    """Use exact numeric equality for values rebuilt from the same raw audit."""
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError):
        return False
    if np.isnan(actual_value) or np.isnan(expected_value):
        return bool(np.isnan(actual_value) and np.isnan(expected_value))
    return bool(actual_value == expected_value)


def _validate_authoritative_confidence_and_calibration(
    bundle: ExploratoryCandidateBundle,
    baselines: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Rebuild every Task 3 robust source estimate from the long audit.

    The baseline table is deliberately not treated as its own evidence.  Its
    population, personal, shrinkage, and placebo source estimates must be the
    exact estimates implied by the complete confidence audit and elapsed-day
    topology.
    """
    required = (
        "sensor_day_id",
        "subject_id",
        "lifelog_date",
        "collection_start_date",
        "elapsed_day_index",
        "primitive",
        "value",
        "coverage_component",
        "gap_component",
        "validity_component",
        "censoring_component",
        "quality_confidence",
        "coverage_only_confidence",
        "value_only_confidence",
        "unweighted_confidence",
        "proxy_source",
        "primitive_status",
        "censoring",
        "usage_policy",
    )
    _require_columns(audit, required, "authoritative Task 3 confidence audit")
    audited = audit.copy(deep=True)
    audited["_sensor_token"] = audited["sensor_day_id"].map(_identity_token)
    audited["_subject_token"] = audited["subject_id"].map(_identity_token)
    audited["_elapsed"] = _exact_k(
        audited["elapsed_day_index"],
        name="authoritative confidence elapsed_day_index",
    )
    if audited["_elapsed"].le(0).any():
        raise ValueError(
            "Authoritative Task 3 confidence audit elapsed days must be positive"
        )
    audited["_local_date"] = pd.to_datetime(
        audited["lifelog_date"], errors="raise"
    )
    audited["_collection_anchor"] = pd.to_datetime(
        audited["collection_start_date"], errors="raise"
    )
    if audited[
        ["_local_date", "_collection_anchor", "_sensor_token", "_subject_token"]
    ].isna().any().any():
        raise ValueError(
            "Authoritative Task 3 confidence audit topology must be nonmissing"
        )

    sensor_metadata_records: list[dict[str, Any]] = []
    for sensor_token, sensor_rows in audited.groupby("_sensor_token", sort=False):
        for column in (
            "_subject_token",
            "_elapsed",
            "_local_date",
            "_collection_anchor",
        ):
            if sensor_rows[column].nunique(dropna=False) != 1:
                raise ValueError(
                    "Authoritative Task 3 confidence audit sensor topology is "
                    f"not one-to-one for {column}"
                )
        sensor_metadata_records.append(
            {
                "_sensor_token": sensor_token,
                "_subject_token": sensor_rows["_subject_token"].iloc[0],
                "_elapsed": int(sensor_rows["_elapsed"].iloc[0]),
                "_local_date": sensor_rows["_local_date"].iloc[0],
                "_collection_anchor": sensor_rows["_collection_anchor"].iloc[0],
            }
        )
    sensor_metadata = pd.DataFrame.from_records(sensor_metadata_records)
    if sensor_metadata.duplicated(["_subject_token", "_elapsed"]).any():
        raise ValueError(
            "Authoritative Task 3 confidence audit participant elapsed-day rows "
            "must be unique"
        )
    for _, participant_rows in sensor_metadata.groupby("_subject_token", sort=False):
        ordered = participant_rows.sort_values("_elapsed", kind="stable")
        elapsed = ordered["_elapsed"].to_numpy(dtype=int)
        if not np.array_equal(elapsed, np.arange(1, len(ordered) + 1)):
            raise ValueError(
                "Authoritative Task 3 confidence audit requires a complete elapsed "
                "calendar"
            )
        dates = ordered["_local_date"].to_numpy(dtype="datetime64[D]")
        if len(dates) > 1 and not np.array_equal(
            np.diff(dates).astype(int), np.ones(len(dates) - 1, dtype=int)
        ):
            raise ValueError(
                "Authoritative Task 3 confidence audit local dates must be consecutive"
            )
        first_date = ordered["_local_date"].iloc[0]
        if not ordered["_collection_anchor"].eq(first_date).all():
            raise ValueError(
                "Authoritative Task 3 confidence audit collection anchor must equal "
                "the first local date"
            )

    numeric_columns = (
        "coverage_component",
        "gap_component",
        "validity_component",
        "censoring_component",
        "quality_confidence",
        "coverage_only_confidence",
        "value_only_confidence",
        "unweighted_confidence",
    )
    numeric: dict[str, np.ndarray] = {}
    for column in numeric_columns:
        values = pd.to_numeric(audited[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not bool(np.isfinite(values).all()) or bool(
            ((values < 0.0) | (values > 1.0)).any()
        ):
            raise ValueError(
                f"Authoritative Task 3 confidence formula {column} must be in [0, 1]"
            )
        numeric[column] = values
    raw_values = pd.to_numeric(audited["value"], errors="coerce").to_numpy(
        dtype=float
    )
    validity = (
        np.isfinite(raw_values)
        & audited["primitive_status"].astype(str).eq("observed").to_numpy()
    ).astype(float)
    censoring_component = audited["censoring"].astype(str).eq("none").to_numpy(
        dtype=float
    )
    usage = audited["primitive"].isin(
        {"usage_load_24h", "usage_disengagement_p90"}
    ).to_numpy()
    unweighted_usage = usage & (
        audited["usage_policy"].astype(str).eq("unweighted").to_numpy()
    )
    expected_quality = (
        numeric["coverage_component"]
        * numeric["gap_component"]
        * validity
        * censoring_component
    )
    expected_coverage = numeric["coverage_component"] * validity
    expected_unweighted = validity * censoring_component
    expected_quality[unweighted_usage] = validity[unweighted_usage]
    expected_coverage[unweighted_usage] = validity[unweighted_usage]
    expected_unweighted[unweighted_usage] = validity[unweighted_usage]
    formula_expectations = (
        ("validity_component", validity),
        ("censoring_component", censoring_component),
        ("quality_confidence", np.clip(expected_quality, 0.0, 1.0)),
        ("coverage_only_confidence", np.clip(expected_coverage, 0.0, 1.0)),
        ("value_only_confidence", validity),
        ("unweighted_confidence", np.clip(expected_unweighted, 0.0, 1.0)),
    )
    for column, expected in formula_expectations:
        if not np.array_equal(numeric[column], expected, equal_nan=True):
            raise ValueError(
                f"Authoritative Task 3 confidence formula mismatch for {column}"
            )
    expected_proxy = np.where(
        usage,
        np.where(
            unweighted_usage,
            "unweighted",
            np.where(
                audited["primitive"].eq("usage_load_24h").to_numpy(),
                "screen_load_24h",
                "screen_disengagement_p90",
            ),
        ),
        audited["primitive"].astype(str).to_numpy(),
    )
    if not np.array_equal(
        audited["proxy_source"].astype(str).to_numpy(), expected_proxy
    ):
        raise ValueError(
            "Authoritative Task 3 confidence formula proxy_source mismatch"
        )

    participants = set(audited["_subject_token"])
    held_participants = set(baselines["_held_token"])
    if participants != held_participants:
        raise ValueError(
            "Authoritative Task 3 confidence audit participants must equal baseline "
            "recipients"
        )
    baseline_donors = set(
        baselines.loc[baselines["_donor_token"].ne(_NULL_DONOR), "_donor_token"]
    )
    if not baseline_donors.issubset(participants):
        raise ValueError(
            "Authoritative Task 3 placebo donor is absent from the confidence audit"
        )

    estimate_frame = audited.loc[
        :,
        [
            "_subject_token",
            "_elapsed",
            "primitive",
            "value",
            "quality_confidence",
        ],
    ].copy()
    population_estimates: dict[tuple[str, str], dict[str, float | int]] = {}
    source_estimates: dict[tuple[str, int, str], dict[str, float | int]] = {}
    for primitive, primitive_rows in estimate_frame.groupby("primitive", sort=False):
        for held_token in participants:
            population_rows = primitive_rows.loc[
                primitive_rows["_subject_token"].ne(held_token)
            ]
            population_estimates[(held_token, str(primitive))] = robust_location_scale(
                population_rows["value"].to_numpy(),
                population_rows["quality_confidence"].to_numpy(),
            )
        for source_token in participants:
            subject_rows = primitive_rows.loc[
                primitive_rows["_subject_token"].eq(source_token)
            ]
            for k in FROZEN_K_VALUES:
                source_rows = subject_rows.loc[subject_rows["_elapsed"].le(k)]
                source_estimates[(source_token, k, str(primitive))] = (
                    robust_location_scale(
                        source_rows["value"].to_numpy(),
                        source_rows["quality_confidence"].to_numpy(),
                    )
                )

    for row_index, row in baselines.iterrows():
        primitive = str(row["primitive"])
        held_token = str(row["_held_token"])
        method = str(row["_method"])
        k = int(row["_k"])
        population = population_estimates[(held_token, primitive)]
        if method == "population":
            calibration = population
        elif method in {"personal", "shrinkage"}:
            calibration = source_estimates[(held_token, k, primitive)]
        elif method == "placebo":
            donor_token = str(row["_donor_token"])
            if donor_token == _NULL_DONOR:
                raise ValueError(
                    "Authoritative Task 3 placebo calibration requires one donor"
                )
            calibration = source_estimates[(donor_token, k, primitive)]
        else:
            raise ValueError(
                f"Authoritative Task 3 calibration has unsupported method: {method}"
            )
        comparisons = (
            ("population_center", population["center"]),
            ("population_scale", population["scale"]),
            ("population_n_eff", population["n_eff"]),
            (
                "population_contributing_rows",
                population["contributing_rows"],
            ),
            ("calibration_center", calibration["center"]),
            ("calibration_scale", calibration["scale"]),
            ("calibration_n_eff", calibration["n_eff"]),
            (
                "calibration_contributing_rows",
                calibration["contributing_rows"],
            ),
        )
        for field, expected in comparisons:
            if not _authoritative_numeric_equal(row[field], expected):
                raise ValueError(
                    "Authoritative Task 3 robust estimate disagrees with confidence "
                    f"audit for {field} at baseline row {row_index}"
                )


def _validate_bundle_artifact_provenance(
    bundle: ExploratoryCandidateBundle,
    representations: pd.DataFrame,
    baselines: pd.DataFrame,
) -> None:
    """Bind bundle metadata, confidence audit, representations, and baselines."""
    _validate_task3_baseline_provenance(baselines)
    audit = bundle.loso_tables.confidence_audit.copy(deep=True)
    _require_columns(
        audit,
        (
            "sensor_day_id",
            "subject_id",
            "primitive",
            "value",
            "quality_confidence",
            "coverage_only_confidence",
            "value_only_confidence",
            "usage_policy",
        ),
        "Task 3 confidence provenance",
    )
    if audit.empty or audit[["sensor_day_id", "subject_id", "primitive"]].isna().any().any():
        raise ValueError("Task 3 confidence provenance keys must be nonempty and nonmissing")
    audit["_sensor_token"] = audit["sensor_day_id"].map(_identity_token)
    audit["_subject_token"] = audit["subject_id"].map(_identity_token)
    if audit.duplicated(["_sensor_token", "primitive"]).any():
        raise ValueError("Task 3 confidence provenance sensor/primitive keys must be unique")
    required_primitives = set(_ALL_PRIMITIVES)
    for _, group in audit.groupby("_sensor_token", sort=False):
        if len(group) != len(_ALL_PRIMITIVES) or set(group["primitive"]) != required_primitives:
            raise ValueError(
                "Task 3 confidence provenance must be 16-primitive-complete per sensor"
            )
    usage_primitives = {"usage_load_24h", "usage_disengagement_p90"}
    actual_usage = audit.loc[
        audit["primitive"].isin(usage_primitives), "usage_policy"
    ].astype(str)
    if actual_usage.empty or not actual_usage.eq(bundle.usage_confidence_policy).all():
        raise ValueError(
            "Task 3 confidence Usage policy disagrees with the declared bundle policy"
        )
    nonusage = audit.loc[
        ~audit["primitive"].isin(usage_primitives), "usage_policy"
    ].astype(str)
    if not nonusage.eq("not_applicable").all():
        raise ValueError(
            "Task 3 confidence Usage policy must be not_applicable outside Usage"
        )
    _validate_authoritative_confidence_and_calibration(bundle, baselines, audit)

    baseline_lookup = baselines.loc[
        :,
        [
            "_held_token",
            "_k",
            "_method",
            "_donor_token",
            "primitive",
            "center",
            "scale",
            "baseline_status",
        ],
    ]
    representation_keys = representations.loc[
        :, ["_held_token", "_k", "_method", "_donor_token"]
    ]
    for primitive in _ALL_PRIMITIVES:
        required_representation_columns = (
            primitive,
            f"{primitive}__confidence",
            f"{primitive}__coverage_only_confidence",
            f"{primitive}__value_only_confidence",
            f"{primitive}__center",
            f"{primitive}__scale",
            f"{primitive}__baseline_status",
        )
        _require_columns(
            representations,
            required_representation_columns,
            "Task 3 representation provenance",
        )
        primitive_baseline = baseline_lookup.loc[
            baseline_lookup["primitive"].eq(primitive)
        ].drop(columns="primitive")
        aligned = representation_keys.merge(
            primitive_baseline,
            on=["_held_token", "_k", "_method", "_donor_token"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
        if len(aligned) != len(representations):
            raise ValueError("Task 3 representation baseline provenance is incomplete")
        if not _numeric_equal(
            representations[f"{primitive}__center"], aligned["center"]
        ).all():
            raise ValueError(f"Task 3 {primitive} representation center provenance mismatch")
        if not _numeric_equal(
            representations[f"{primitive}__scale"], aligned["scale"]
        ).all():
            raise ValueError(f"Task 3 {primitive} representation scale provenance mismatch")
        if not representations[f"{primitive}__baseline_status"].astype(str).reset_index(
            drop=True
        ).equals(aligned["baseline_status"].astype(str).reset_index(drop=True)):
            raise ValueError(f"Task 3 {primitive} representation status provenance mismatch")

        audit_primitive = audit.loc[
            audit["primitive"].eq(primitive),
            [
                "_sensor_token",
                "_subject_token",
                "value",
                "quality_confidence",
                "coverage_only_confidence",
                "value_only_confidence",
            ],
        ].rename(
            columns={
                "value": "_audit_value",
                "quality_confidence": "_audit_quality",
                "coverage_only_confidence": "_audit_coverage",
                "value_only_confidence": "_audit_value_only",
            }
        )
        rep_audit = representations.loc[
            :,
            [
                "_sensor_token",
                "_held_token",
                primitive,
                f"{primitive}__confidence",
                f"{primitive}__coverage_only_confidence",
                f"{primitive}__value_only_confidence",
            ],
        ].merge(
            audit_primitive,
            on="_sensor_token",
            how="left",
            sort=False,
            validate="many_to_one",
        )
        if rep_audit["_subject_token"].isna().any() or not rep_audit[
            "_held_token"
        ].equals(rep_audit["_subject_token"]):
            raise ValueError(f"Task 3 {primitive} confidence subject provenance mismatch")
        comparisons = (
            (primitive, "_audit_value", "raw value"),
            (f"{primitive}__confidence", "_audit_quality", "quality confidence"),
            (
                f"{primitive}__coverage_only_confidence",
                "_audit_coverage",
                "coverage confidence",
            ),
            (
                f"{primitive}__value_only_confidence",
                "_audit_value_only",
                "value-only confidence",
            ),
        )
        for representation_column, audit_column, label in comparisons:
            if not _numeric_equal(
                rep_audit[representation_column], rep_audit[audit_column]
            ).all():
                raise ValueError(
                    f"Task 3 {primitive} {label} provenance disagrees with confidence audit"
                )
def _prepare_frozen_inputs(
    loso_tables: LosoRepresentationTables,
    future_oracles: pd.DataFrame,
    *,
    expected_lambda_days: float = PRIMARY_LAMBDA_DAYS,
    allow_formula_insufficient: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Any]]:
    if not isinstance(loso_tables, LosoRepresentationTables):
        raise TypeError("loso_tables must be a LosoRepresentationTables instance")
    if not isinstance(future_oracles, pd.DataFrame):
        raise TypeError("future_oracles must be a pandas DataFrame")
    representations = loso_tables.representations.copy(deep=True)
    baselines = loso_tables.baselines.copy(deep=True)
    oracles = future_oracles.copy(deep=True)
    candidates = loso_tables.placebo_candidates.copy(deep=True)
    representation_required = (
        "sensor_day_id",
        "subject_id",
        "held_out_subject",
        "elapsed_day_index",
        "k",
        "method",
        "donor_subject",
        *(f"{primitive}__z" for primitive in _ALL_PRIMITIVES),
    )
    baseline_required = (
        "held_out_subject",
        "k",
        "method",
        "donor_subject",
        "primitive",
        "center",
        "scale",
        "population_center",
        "population_scale",
        "baseline_status",
        "lambda_days",
        "n_eff",
        "contributing_rows",
    )
    oracle_required = (
        "held_out_subject",
        "primitive",
        "oracle_center",
        "oracle_scale",
        "oracle_status",
        "population_scale",
    )
    _require_columns(representations, representation_required, "representation")
    _require_columns(baselines, baseline_required, "baseline")
    _require_columns(oracles, oracle_required, "future oracle")
    forbidden = _forbidden_representation_columns(representations)
    if forbidden:
        raise ValueError(
            "Frozen representation contains forbidden oracle, dual, outcome, or "
            f"selector columns: {forbidden}"
        )
    for name, frame in (("baseline", baselines), ("future oracle", oracles)):
        bad = [
            str(column)
            for column in frame.columns
            if re.fullmatch(r"[QS]\d+", str(column), flags=re.IGNORECASE)
            or "argmax" in str(column).lower()
            or "selected_candidate" in str(column).lower()
            or "dual_calibration" in str(column).lower()
        ]
        if bad:
            raise ValueError(f"{name} contains forbidden outcome, dual, or selector columns: {bad}")

    representation_held_tokens = representations["held_out_subject"].map(_identity_token)
    representation_subject_tokens = representations["subject_id"].map(_identity_token)
    if not representation_held_tokens.equals(representation_subject_tokens):
        raise ValueError("representation subject_id must equal held_out_subject")
    participant_frame = (
        representations.loc[:, ["held_out_subject"]]
        .assign(_held_token=representation_held_tokens)
        .drop_duplicates("_held_token")
    )
    participants = [
        value
        for _, value in sorted(
            zip(
                participant_frame["_held_token"],
                participant_frame["held_out_subject"],
                strict=True,
            )
        )
    ]
    if len(participants) != FROZEN_PARTICIPANT_DENOMINATOR:
        raise ValueError("Frozen G3 requires exactly 10 participant identities")
    participant_tokens = {_identity_token(value) for value in participants}
    if set(baselines["held_out_subject"].map(_identity_token)) != participant_tokens:
        raise ValueError("Baseline subjects must match the representation subjects")
    if set(oracles["held_out_subject"].map(_identity_token)) != participant_tokens:
        raise ValueError("Future-oracle subjects must match the representation subjects")

    expected_groups: set[tuple[str, int, str, str]] = set()
    for held in participants:
        held_token = _identity_token(held)
        expected_groups.add((held_token, 0, "population", _NULL_DONOR))
        for k in FROZEN_K_VALUES:
            expected_groups.add((held_token, k, "personal", _NULL_DONOR))
            expected_groups.add((held_token, k, "shrinkage", _NULL_DONOR))
            for donor in participants:
                if _identity_token(donor) != held_token:
                    expected_groups.add(
                        (held_token, k, "placebo", _identity_token(donor))
                    )
    representation_groups = set(_group_tuples(representations))
    baseline_groups = set(_group_tuples(baselines))
    if representation_groups != expected_groups or baseline_groups != expected_groups:
        raise ValueError("Frozen Task 3 method topology is incomplete or contains extra groups")

    baseline_keys = _group_key_frame(baselines)
    baselines = pd.concat([baselines.reset_index(drop=True), baseline_keys.reset_index(drop=True)], axis=1)
    if baselines.duplicated(
        ["_held_token", "_k", "_method", "_donor_token", "primitive"]
    ).any():
        raise ValueError("Frozen baseline keys must be unique")
    required_primitives = set(_ALL_PRIMITIVES)
    for _, group in baselines.groupby(
        ["_held_token", "_k", "_method", "_donor_token"],
        sort=False,
        dropna=False,
    ):
        if len(group) != len(_ALL_PRIMITIVES) or set(group["primitive"]) != required_primitives:
            raise ValueError("Every frozen baseline group must be 16-primitive-complete")

    representation_keys = _group_key_frame(representations)
    representations = pd.concat(
        [representations.reset_index(drop=True), representation_keys.reset_index(drop=True)],
        axis=1,
    )
    representations["_sensor_token"] = representations["sensor_day_id"].map(
        _identity_token
    )
    if representations.duplicated(
        ["_held_token", "_k", "_method", "_donor_token", "_sensor_token"]
    ).any():
        raise ValueError("Frozen representation keys must be unique")
    elapsed = _exact_k(representations["elapsed_day_index"], name="elapsed_day_index")
    if not elapsed.gt(21).all():
        raise ValueError("Frozen G3 representations must contain only post-day-21 rows")
    representations["elapsed_day_index"] = elapsed
    owner_counts = representations.groupby("_sensor_token", sort=False)[
        "_held_token"
    ].nunique()
    if not owner_counts.eq(1).all():
        raise ValueError("sensor_day_id must belong to exactly one held-out participant")
    expected_rows: dict[str, frozenset[str]] = {}
    population_rows = representations.loc[
        representations["_method"].eq("population")
        & representations["_k"].eq(0)
        & representations["_donor_token"].eq(_NULL_DONOR)
    ]
    for held_token, group in population_rows.groupby("_held_token", sort=False):
        expected_rows[held_token] = frozenset(group["_sensor_token"])
    if set(expected_rows) != participant_tokens or any(not rows for rows in expected_rows.values()):
        raise ValueError("Each participant requires a nonempty population evaluation grid")
    for group_key, group in representations.groupby(
        ["_held_token", "_k", "_method", "_donor_token"],
        sort=False,
        dropna=False,
    ):
        if frozenset(group["_sensor_token"]) != expected_rows[group_key[0]]:
            raise ValueError("Every method must retain the complete common row grid")

    oracle_held = oracles["held_out_subject"].map(_identity_token)
    oracles = oracles.assign(_held_token=oracle_held)
    if oracles.duplicated(["_held_token", "primitive"]).any():
        raise ValueError("Future-oracle keys must be unique")
    if len(oracles) != len(participants) * len(_ALL_PRIMITIVES):
        raise ValueError("Future oracle must be 16-primitive-complete for every participant")
    for _, group in oracles.groupby("_held_token", sort=False):
        if set(group["primitive"]) != required_primitives:
            raise ValueError("Future oracle must be 16-primitive-complete for every participant")

    anchors = baselines.loc[
        baselines["_method"].eq("population")
        & baselines["_k"].eq(0)
        & baselines["_donor_token"].eq(_NULL_DONOR),
        ["_held_token", "primitive", "center", "scale", "population_center", "population_scale"],
    ].copy()
    if len(anchors) != len(participants) * len(_ALL_PRIMITIVES):
        raise ValueError("Recipient-excluding population anchors are incomplete")
    if not _numeric_equal(anchors["center"], anchors["population_center"]).all() or not _numeric_equal(
        anchors["scale"], anchors["population_scale"]
    ).all():
        raise ValueError("Population rows must equal the recipient-excluding population anchor")
    anchor_lookup = anchors.rename(
        columns={
            "center": "_anchor_center",
            "scale": "_anchor_scale",
        }
    ).loc[:, ["_held_token", "primitive", "_anchor_center", "_anchor_scale"]]
    audited = baselines.merge(
        anchor_lookup,
        on=["_held_token", "primitive"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if not _numeric_equal(audited["population_center"], audited["_anchor_center"]).all() or not _numeric_equal(
        audited["population_scale"], audited["_anchor_scale"]
    ).all():
        raise ValueError("Baseline disagrees with its recipient-excluding population anchor")
    baselines = audited
    if not np.isfinite(expected_lambda_days) or float(expected_lambda_days) < 0.0:
        raise ValueError("Expected lambda bundle value must be finite and nonnegative")
    bundle_lambda = pd.to_numeric(baselines["lambda_days"], errors="coerce")
    if bundle_lambda.isna().any() or not bundle_lambda.eq(
        float(expected_lambda_days)
    ).all():
        raise ValueError(
            f"Every baseline lambda must match its predeclared bundle value "
            f"{float(expected_lambda_days):g}"
        )

    core_baselines = baselines["primitive"].isin(_CORE_PRIMITIVES)
    scored_baselines = baselines.loc[core_baselines]
    baseline_centers = pd.to_numeric(scored_baselines["center"], errors="coerce")
    baseline_scales = pd.to_numeric(scored_baselines["scale"], errors="coerce")
    baseline_n_eff = pd.to_numeric(scored_baselines["n_eff"], errors="coerce")
    baseline_rows = pd.to_numeric(
        scored_baselines["contributing_rows"], errors="coerce"
    )
    baseline_status = scored_baselines["baseline_status"].astype(str)
    observed_baseline = (
        baseline_status.eq("observed")
        & np.isfinite(baseline_centers)
        & np.isfinite(baseline_scales)
        & baseline_scales.gt(0.0)
        & np.isfinite(baseline_n_eff)
        & baseline_n_eff.ge(0.0)
        & np.isfinite(baseline_rows)
        & baseline_rows.ge(0.0)
    )
    insufficient_support = (
        np.isfinite(baseline_n_eff)
        & baseline_n_eff.ge(0.0)
        & np.isfinite(baseline_rows)
        & baseline_rows.ge(0.0)
        if allow_formula_insufficient
        else baseline_n_eff.eq(0.0) & baseline_rows.eq(0.0)
    )
    insufficient_baseline = (
        baseline_status.eq("insufficient")
        & baseline_centers.isna()
        & baseline_scales.isna()
        & insufficient_support
    )
    if not (observed_baseline | insufficient_baseline).all():
        raise ValueError(
            "Every scored baseline must be either observed with a positive finite "
            "scale and valid support metadata or an authentic insufficient "
            "zero-support artifact"
        )

    oracles = oracles.merge(
        anchor_lookup,
        on=["_held_token", "primitive"],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    core_oracles = oracles["primitive"].isin(_CORE_PRIMITIVES)
    oracle_centers = pd.to_numeric(
        oracles.loc[core_oracles, "oracle_center"], errors="coerce"
    )
    oracle_scales = pd.to_numeric(
        oracles.loc[core_oracles, "oracle_scale"], errors="coerce"
    )
    if (
        oracle_centers.isna().any()
        or not np.isfinite(oracle_centers).all()
        or oracle_scales.isna().any()
        or not np.isfinite(oracle_scales).all()
        or not oracle_scales.gt(0).all()
    ):
        raise ValueError("Every scored oracle requires a finite center and positive finite scale")
    if not oracles.loc[core_oracles, "oracle_status"].astype(str).eq("observed").all():
        raise ValueError("Every scored oracle must have observed status")
    if not _numeric_equal(
        oracles.loc[core_oracles, "population_scale"],
        oracles.loc[core_oracles, "_anchor_scale"],
    ).all():
        raise ValueError("Oracle disagrees with its recipient-excluding population anchor")

    if not isinstance(candidates, pd.DataFrame):
        raise TypeError("placebo_candidates must be a pandas DataFrame")
    _require_columns(
        candidates,
        (
            "candidate_id",
            "held_out_subject",
            "donor_subject",
            "k",
            "same_elapsed_window",
            "all_primitives_same_donor",
            "is_self",
        ),
        "placebo candidate",
    )
    forbidden_candidates = _forbidden_representation_columns(candidates)
    if forbidden_candidates:
        raise ValueError(
            "Placebo candidate selector contains forbidden oracle, dual, outcome, "
            f"or argmax columns: {forbidden_candidates}"
        )
    candidate_ids = _exact_k(candidates["candidate_id"], name="candidate_id")
    if candidate_ids.lt(0).any() or candidate_ids.duplicated().any():
        raise ValueError("Placebo candidate_id values must be unique nonnegative integers")
    candidate_k = _exact_k(candidates["k"], name="placebo candidate k")
    for column, expected in (
        ("same_elapsed_window", True),
        ("all_primitives_same_donor", True),
        ("is_self", False),
    ):
        strict_boolean = candidates[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        )
        if not strict_boolean.all() or not candidates[column].eq(expected).all():
            raise ValueError(
                "Placebo candidate selector must preserve coherent, non-self, "
                "same-window assignments"
            )
    if candidates[["held_out_subject", "donor_subject"]].isna().any().any():
        raise ValueError("Placebo candidate identities must be nonmissing")
    candidate_key_rows = pd.DataFrame(
        {
            "held": candidates["held_out_subject"].map(_identity_token),
            "k": candidate_k,
            "donor": candidates["donor_subject"].map(_identity_token),
        }
    )
    if candidate_key_rows.duplicated(["held", "k", "donor"]).any():
        raise ValueError("Placebo candidate topology must not contain duplicate rows")
    candidate_keys = set(
        candidate_key_rows.itertuples(index=False, name=None)
    )
    expected_candidates = {
        (held, k, donor)
        for held, k, method, donor in expected_groups
        if method == "placebo"
    }
    if len(candidates) != len(expected_candidates) or candidate_keys != expected_candidates:
        raise ValueError(
            "Placebo candidate table must contain the exact coherent non-self donor topology"
        )

    method_order = {"population": 0, "personal": 1, "shrinkage": 2, "placebo": 3}
    baselines["_method_order"] = baselines["_method"].map(method_order)
    baselines = baselines.sort_values(
        ["_method_order", "_k", "_held_token", "_donor_token", "primitive"],
        kind="stable",
    ).reset_index(drop=True)
    representations["_method_order"] = representations["_method"].map(method_order)
    representations = representations.sort_values(
        ["_method_order", "_k", "_held_token", "_donor_token", "_sensor_token"],
        kind="stable",
    ).reset_index(drop=True)
    oracles = oracles.sort_values(["_held_token", "primitive"], kind="stable").reset_index(drop=True)
    return representations, baselines, oracles, participants


def _canonical_candidate_bundles(
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
    future_oracles: pd.DataFrame,
    *,
    trusted_source_calendar_sha256: str,
) -> list[
    tuple[
        ExploratoryCandidateBundle,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        list[Any],
    ]
]:
    if isinstance(candidate_bundles, (str, bytes)) or not isinstance(
        candidate_bundles, Sequence
    ):
        raise ValueError("candidate_bundles must be a predeclared nonempty sequence")
    bundles = list(candidate_bundles)
    if not bundles or any(
        not isinstance(bundle, ExploratoryCandidateBundle) for bundle in bundles
    ):
        raise ValueError("candidate_bundles must contain predeclared candidate bundles")
    bundle_ids = [bundle.bundle_id for bundle in bundles]
    if any(not isinstance(value, str) or not value.strip() for value in bundle_ids):
        raise ValueError("Every predeclared bundle_id must be a nonempty string")
    if len(bundle_ids) != len(set(bundle_ids)):
        raise ValueError("Predeclared bundle_id values must be unique")
    for value in bundle_ids:
        if re.search(r"oracle|outcome|argmax|selected|winner", value, flags=re.IGNORECASE):
            raise ValueError("Predeclared bundle_id cannot encode an oracle or argmax selector")

    trusted_source = _canonical_source_calendar_sha256(
        trusted_source_calendar_sha256,
        name="trusted source calendar fingerprint",
    )
    for bundle in bundles:
        declared_source = _canonical_source_calendar_sha256(
            bundle.source_calendar_sha256,
            name=f"bundle {bundle.bundle_id!r} source calendar fingerprint",
        )
        if declared_source != trusted_source:
            raise ValueError(
                "Bundle source_calendar_sha256 disagrees with the externally "
                "trusted source calendar fingerprint"
            )

    if not isinstance(future_oracles, pd.DataFrame):
        raise TypeError("future_oracles must be a pandas DataFrame")
    if future_oracles.columns.duplicated().any():
        raise ValueError("future oracle columns must be unique")
    oracle_policy_column = "usage_confidence_policy"
    requested_oracle_policies = {
        str(bundle.usage_confidence_policy) for bundle in bundles
    }
    if oracle_policy_column not in future_oracles.columns:
        if len(requested_oracle_policies) > 1:
            raise ValueError(
                "Multiple candidate Usage policies require separate future-oracle "
                "rows identified by usage_confidence_policy"
            )
        oracle_tables = {
            next(iter(requested_oracle_policies)): future_oracles.copy(deep=True)
        }
    else:
        policy_values = future_oracles[oracle_policy_column]
        if policy_values.isna().any() or not policy_values.map(
            lambda value: isinstance(value, str) and bool(value)
        ).all():
            raise ValueError(
                "Every policy-specific future oracle requires a nonempty Usage policy"
            )
        oracle_tables = {
            policy: future_oracles.loc[
                policy_values.eq(policy)
            ].copy(deep=True)
            for policy in requested_oracle_policies
        }
        missing_oracle_policies = sorted(
            policy for policy, table in oracle_tables.items() if table.empty
        )
        if missing_oracle_policies:
            raise ValueError(
                "Missing future-oracle Usage policy tables: "
                f"{missing_oracle_policies}"
            )

    allowed_lambda = {3.0, 7.0, 14.0}
    allowed_usage_confidence = {"screen_proxy", "unweighted", "value_only"}
    allowed_usage_operator = {"frozen", "capped", "max_app", "record_mean"}
    allowed_threshold = {0.0, 0.5, 0.7, 0.8, 0.9}
    allowed_weighting = {"row_weighted", "participant_balanced"}
    metadata_keys: set[tuple[float, str, str, float, str]] = set()
    prepared: list[
        tuple[
            ExploratoryCandidateBundle,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            list[Any],
        ]
    ] = []
    for bundle in bundles:
        try:
            lambda_days = float(bundle.lambda_days)
            quality_threshold = float(bundle.quality_threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("Bundle lambda and threshold must be predeclared numbers") from error
        if lambda_days not in allowed_lambda:
            raise ValueError("Bundle lambda must be one of 3, 7, or 14")
        if bundle.usage_confidence_policy not in allowed_usage_confidence:
            raise ValueError("Unknown predeclared Usage confidence policy")
        if bundle.usage_operator not in allowed_usage_operator:
            raise ValueError("Unknown predeclared Usage operator")
        if quality_threshold not in allowed_threshold:
            raise ValueError("Unknown predeclared quality threshold")
        if bundle.population_weighting not in allowed_weighting:
            raise ValueError("Unknown predeclared population weighting")
        metadata_key = (
            lambda_days,
            bundle.usage_confidence_policy,
            bundle.usage_operator,
            quality_threshold,
            bundle.population_weighting,
        )
        if metadata_key in metadata_keys:
            raise ValueError("Predeclared candidate bundle metadata must be unique")
        metadata_keys.add(metadata_key)
        representations, baselines, oracles, participants = _prepare_frozen_inputs(
            bundle.loso_tables,
            oracle_tables[bundle.usage_confidence_policy],
            expected_lambda_days=lambda_days,
            allow_formula_insufficient=True,
        )
        _validate_bundle_artifact_provenance(bundle, representations, baselines)
        prepared.append((bundle, representations, baselines, oracles, participants))

    required = {
        (lambda_days, "screen_proxy", "frozen", 0.0, "row_weighted")
        for lambda_days in allowed_lambda
    }
    if not required.issubset(metadata_keys):
        raise ValueError(
            "Predeclared lambda candidate bundles must include 3, 7, and 14 "
            "for the primary operational policy"
        )
    prepared.sort(
        key=lambda item: (
            float(item[0].lambda_days),
            item[0].usage_confidence_policy,
            item[0].usage_operator,
            float(item[0].quality_threshold),
            item[0].population_weighting,
            item[0].bundle_id,
        )
    )
    reference_representation_topology: list[tuple[Any, ...]] | None = None
    reference_baseline_topology: list[tuple[Any, ...]] | None = None
    for _, representations, baselines, _, _ in prepared:
        representation_topology = list(
            representations.loc[
                :,
                ["_held_token", "_k", "_method", "_donor_token", "_sensor_token"],
            ].itertuples(index=False, name=None)
        )
        baseline_topology = list(
            baselines.loc[
                :, ["_held_token", "_k", "_method", "_donor_token", "primitive"]
            ].itertuples(index=False, name=None)
        )
        if reference_representation_topology is None:
            reference_representation_topology = representation_topology
            reference_baseline_topology = baseline_topology
        elif (
            representation_topology != reference_representation_topology
            or baseline_topology != reference_baseline_topology
        ):
            raise ValueError(
                "Every predeclared candidate bundle must use exact common row/topology"
            )
    return prepared


def _canonical_candidate_bundles_self_declared(
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
    future_oracles: pd.DataFrame,
) -> list[
    tuple[
        ExploratoryCandidateBundle,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        list[Any],
    ]
]:
    """Canonicalize non-L/R diagnostics without granting external authority."""
    if isinstance(candidate_bundles, (str, bytes)) or not isinstance(
        candidate_bundles, Sequence
    ):
        raise ValueError("candidate_bundles must be a predeclared nonempty sequence")
    bundles = list(candidate_bundles)
    if not bundles or any(
        not isinstance(bundle, ExploratoryCandidateBundle) for bundle in bundles
    ):
        raise ValueError("candidate_bundles must contain predeclared candidate bundles")
    return _canonical_candidate_bundles(
        bundles,
        future_oracles,
        trusted_source_calendar_sha256=bundles[0].source_calendar_sha256,
    )


def _candidate_id_from_ledger(
    ledger: pd.DataFrame,
    *,
    calibration_mode: str,
    k: int,
    bundle: ExploratoryCandidateBundle,
    clipping_state: str,
) -> str:
    selected = ledger.loc[
        ledger["calibration_mode"].eq(calibration_mode)
        & ledger["k"].eq(int(k))
        & ledger["lambda_days"].eq(float(bundle.lambda_days))
        & ledger["usage_confidence_policy"].eq(bundle.usage_confidence_policy)
        & ledger["usage_operator"].eq(bundle.usage_operator)
        & ledger["quality_threshold"].eq(float(bundle.quality_threshold))
        & ledger["clipping_state"].eq(clipping_state)
        & ledger["population_weighting"].eq(bundle.population_weighting)
        & ~ledger["oracle_diagnostic"]
    ]
    if len(selected) != 1:
        raise RuntimeError("Predeclared candidate identity is not unique in the ledger")
    return str(selected["candidate_id"].iloc[0])


def _location_gate_decision(
    *,
    improved_participants: int,
    minimum_lopo: float,
    coherent_p: float,
    nonincreasing_transitions: int,
    k14_below_k3: bool,
    adjusted_p: float,
    finite_participants: int,
) -> pd.DataFrame:
    complete = int(finite_participants) == FROZEN_PARTICIPANT_DENOMINATOR
    checks = (
        (
            "at_least_8_of_10_participants_improve",
            int(improved_participants),
            8,
            ">=",
            int(improved_participants) >= 8,
        ),
        (
            "all_lopo_means_positive",
            float(minimum_lopo),
            0.0,
            ">",
            np.isfinite(minimum_lopo) and float(minimum_lopo) > 0.0,
        ),
        (
            "coherent_donor_center_p_at_most_005",
            float(coherent_p),
            0.05,
            "<=",
            np.isfinite(coherent_p) and float(coherent_p) <= 0.05,
        ),
        (
            "at_least_2_of_3_k_transitions_nonincreasing",
            int(nonincreasing_transitions),
            2,
            ">=",
            int(nonincreasing_transitions) >= 2,
        ),
        (
            "k14_location_error_below_k3",
            bool(k14_below_k3),
            True,
            "is",
            bool(k14_below_k3),
        ),
        (
            "selection_adjusted_location_p_at_most_005",
            float(adjusted_p),
            0.05,
            "<=",
            np.isfinite(adjusted_p) and float(adjusted_p) <= 0.05,
        ),
    )
    status = "observed" if complete else "insufficient"
    records = [
        {
            "gate": gate,
            "observed": observed,
            "threshold": threshold,
            "comparator": comparator,
            "passed": bool(complete and passed),
            "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
            "finite_participants": int(finite_participants),
            "status": status,
            "reason": (
                "criterion_met"
                if complete and passed
                else "fixed_participant_denominator_incomplete"
                if not complete
                else "criterion_not_met"
            ),
            "post_hoc_exploratory": True,
        }
        for gate, observed, threshold, comparator, passed in checks
    ]
    final_pass = bool(complete and all(record["passed"] for record in records))
    records.append(
        {
            "gate": "exploratory_g3_l_pass",
            "observed": int(sum(record["passed"] for record in records)),
            "threshold": len(checks),
            "comparator": "all",
            "passed": final_pass,
            "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
            "finite_participants": int(finite_participants),
            "status": status,
            "reason": (
                "all_exploratory_location_criteria_met"
                if final_pass
                else "fixed_participant_denominator_incomplete"
                if not complete
                else "one_or_more_exploratory_location_criteria_failed"
            ),
            "post_hoc_exploratory": True,
        }
    )
    return pd.DataFrame.from_records(records)


def _evaluate_exploratory_location_transfer_trusted(
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
    future_oracles: pd.DataFrame,
    *,
    validated_domains: Sequence[str],
    trusted_source_calendar_sha256: str,
    derangement_draws: int = DEFAULT_DERANGEMENT_DRAWS,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> ExploratoryLocationResult:
    """Evaluate post-hoc G3-L without allowing scale to enter its estimand."""
    domains = _canonical_validated_domains(validated_domains)
    if (
        isinstance(derangement_draws, bool)
        or not isinstance(derangement_draws, (int, np.integer))
        or int(derangement_draws) <= 0
    ):
        raise ValueError("derangement_draws must be an exact positive integer")
    if isinstance(root_seed, bool) or not isinstance(root_seed, (int, np.integer)):
        raise ValueError("root_seed must be an exact integer")
    prepared = _canonical_candidate_bundles(
        candidate_bundles,
        future_oracles,
        trusted_source_calendar_sha256=trusted_source_calendar_sha256,
    )
    ledger = exploratory_candidate_ledger()
    ledger["available"] = False
    ledger["availability_reason"] = "candidate_bundle_not_supplied"
    ledger["location_family"] = False
    ledger["primary_candidate"] = False

    primitive_records: list[dict[str, Any]] = []
    participant_frames: list[pd.DataFrame] = []
    primary_prepared: tuple[
        ExploratoryCandidateBundle,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        list[Any],
    ] | None = None
    for bundle, representations, baselines, oracles, bundle_participants in prepared:
        if (
            float(bundle.lambda_days) == 7.0
            and bundle.usage_confidence_policy == "screen_proxy"
            and bundle.usage_operator == "frozen"
            and float(bundle.quality_threshold) == 0.0
            and bundle.population_weighting == "row_weighted"
        ):
            primary_prepared = (
                bundle,
                representations,
                baselines,
                oracles,
                bundle_participants,
            )
        oracle_lookup = oracles.loc[
            oracles["primitive"].isin(_CORE_PRIMITIVES),
            ["_held_token", "primitive", "oracle_center"],
        ].rename(columns={"oracle_center": "_oracle_center"})
        population = baselines.loc[
            baselines["_method"].eq("population")
            & baselines["_k"].eq(0)
            & baselines["primitive"].isin(_CORE_PRIMITIVES),
            [
                "_held_token",
                "held_out_subject",
                "primitive",
                "_anchor_center",
                "_anchor_scale",
            ],
        ].merge(
            oracle_lookup,
            on=["_held_token", "primitive"],
            how="left",
            validate="one_to_one",
        )
        population_lookup = population.set_index(["_held_token", "primitive"])
        shrinkage = baselines.loc[
            baselines["_method"].eq("shrinkage")
            & baselines["_k"].isin(FROZEN_K_VALUES)
            & baselines["primitive"].isin(_CORE_PRIMITIVES)
        ]
        for calibration_mode in ("full", "location_only", "scale_only"):
            for k in FROZEN_K_VALUES:
                candidate_id = _candidate_id_from_ledger(
                    ledger,
                    calibration_mode=calibration_mode,
                    k=k,
                    bundle=bundle,
                    clipping_state="unclipped",
                )
                available = ledger["candidate_id"].eq(candidate_id)
                ledger.loc[available, "available"] = True
                ledger.loc[available, "availability_reason"] = "scored_g3_l"
                ledger.loc[available, "location_family"] = True
                is_primary = bool(
                    calibration_mode == "location_only"
                    and k == PRIMARY_K
                    and float(bundle.lambda_days) == PRIMARY_LAMBDA_DAYS
                    and bundle.usage_confidence_policy == "screen_proxy"
                    and bundle.usage_operator == "frozen"
                    and float(bundle.quality_threshold) == 0.0
                    and bundle.population_weighting == "row_weighted"
                )
                ledger.loc[available, "primary_candidate"] = is_primary
                selected = shrinkage.loc[shrinkage["_k"].eq(k)]
                for _, row in selected.iterrows():
                    key = (row["_held_token"], row["primitive"])
                    anchor = population_lookup.loc[key]
                    population_center = float(anchor["_anchor_center"])
                    population_scale = float(anchor["_anchor_scale"])
                    oracle_center = float(anchor["_oracle_center"])
                    candidate_center = (
                        population_center
                        if calibration_mode == "scale_only"
                        else float(row["center"])
                    )
                    population_error = abs(population_center - oracle_center) / population_scale
                    primitive_complete = bool(
                        str(row["baseline_status"]) == "observed"
                        and np.isfinite(candidate_center)
                        and np.isfinite(population_error)
                        and np.isfinite(population_scale)
                        and population_scale > 0.0
                    )
                    candidate_error = (
                        abs(candidate_center - oracle_center) / population_scale
                        if primitive_complete
                        else np.nan
                    )
                    primitive_records.append(
                        {
                            "candidate_id": candidate_id,
                            "bundle_id": bundle.bundle_id,
                            "held_out_subject": row["held_out_subject"],
                            "primitive": row["primitive"],
                            "domain": _PRIMITIVE_DOMAIN[row["primitive"]],
                            "calibration_mode": calibration_mode,
                            "k": int(k),
                            "lambda_days": float(bundle.lambda_days),
                            "usage_confidence_policy": bundle.usage_confidence_policy,
                            "usage_operator": bundle.usage_operator,
                            "quality_threshold": float(bundle.quality_threshold),
                            "population_weighting": bundle.population_weighting,
                            "population_center": population_center,
                            "candidate_center": candidate_center,
                            "oracle_center": oracle_center,
                            "population_scale": population_scale,
                            "population_location_error": population_error,
                            "candidate_location_error": candidate_error,
                            "improvement": population_error - candidate_error,
                            "scale_ignored": True,
                            "primary_candidate": is_primary,
                            "finite_count": int(primitive_complete),
                            "status": "observed" if primitive_complete else "insufficient",
                            "post_hoc_exploratory": True,
                            "_held_token": row["_held_token"],
                        }
                    )

    primitive = pd.DataFrame.from_records(primitive_records)
    primitive = primitive.loc[primitive["domain"].isin(domains)].copy()
    group_metadata = [
        "candidate_id",
        "bundle_id",
        "calibration_mode",
        "k",
        "lambda_days",
        "usage_confidence_policy",
        "usage_operator",
        "quality_threshold",
        "population_weighting",
        "primary_candidate",
        "post_hoc_exploratory",
        "held_out_subject",
        "_held_token",
    ]
    domain_records: list[dict[str, Any]] = []
    for keys, group in primitive.groupby(
        [*group_metadata, "domain"], sort=False, dropna=False
    ):
        metadata = dict(zip([*group_metadata, "domain"], keys, strict=True))
        denominator = len(FROZEN_DOMAIN_PRIMITIVES[str(metadata["domain"])])
        finite = int(np.isfinite(group["candidate_location_error"]).sum())
        complete = len(group) == denominator and finite == denominator
        domain_records.append(
            {
                **metadata,
                "population_location_error": (
                    float(group["population_location_error"].mean()) if complete else np.nan
                ),
                "candidate_location_error": (
                    float(group["candidate_location_error"].mean()) if complete else np.nan
                ),
                "improvement": (
                    float(group["improvement"].mean()) if complete else np.nan
                ),
                "primitive_denominator": denominator,
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "scale_ignored": True,
            }
        )
    domain_frame = pd.DataFrame.from_records(domain_records)
    participant_metadata = [column for column in group_metadata if column != "post_hoc_exploratory"]
    participant_records: list[dict[str, Any]] = []
    for keys, group in domain_frame.groupby(
        participant_metadata, sort=False, dropna=False
    ):
        metadata = dict(zip(participant_metadata, keys, strict=True))
        finite = int(np.isfinite(group["candidate_location_error"]).sum())
        complete = len(group) == len(domains) and finite == len(domains)
        participant_records.append(
            {
                **metadata,
                "validated_domains": json.dumps(domains),
                "population_location_error": (
                    float(group["population_location_error"].mean()) if complete else np.nan
                ),
                "candidate_location_error": (
                    float(group["candidate_location_error"].mean()) if complete else np.nan
                ),
                "improvement": float(group["improvement"].mean()) if complete else np.nan,
                "domain_denominator": len(domains),
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "scale_ignored": True,
                "post_hoc_exploratory": True,
            }
        )
    participants_frame = pd.DataFrame.from_records(participant_records)
    participants_frame = participants_frame.sort_values(
        ["candidate_id", "_held_token"], kind="stable"
    ).reset_index(drop=True)
    primary = participants_frame.loc[participants_frame["primary_candidate"]].copy()
    if len(primary) != FROZEN_PARTICIPANT_DENOMINATOR:
        raise ValueError("Primary exploratory location candidate requires exactly 10 participants")

    curve_rows: list[dict[str, Any]] = []
    curve_source = participants_frame.loc[
        participants_frame["calibration_mode"].eq("location_only")
        & participants_frame["lambda_days"].eq(PRIMARY_LAMBDA_DAYS)
        & participants_frame["usage_confidence_policy"].eq("screen_proxy")
        & participants_frame["usage_operator"].eq("frozen")
        & participants_frame["quality_threshold"].eq(0.0)
        & participants_frame["population_weighting"].eq("row_weighted")
    ]
    curve_errors: list[float] = []
    for k in FROZEN_K_VALUES:
        group = curve_source.loc[curve_source["k"].eq(k)]
        complete = len(group) == FROZEN_PARTICIPANT_DENOMINATOR and group[
            "status"
        ].eq("observed").all()
        error = float(group["candidate_location_error"].mean()) if complete else np.nan
        curve_errors.append(error)
    transitions = int(
        sum(
            np.isfinite(left) and np.isfinite(right) and right <= left
            for left, right in zip(curve_errors[:-1], curve_errors[1:], strict=True)
        )
    )
    k14_below_k3 = bool(
        np.isfinite(curve_errors[0])
        and np.isfinite(curve_errors[2])
        and curve_errors[2] < curve_errors[0]
    )
    for k, error in zip(FROZEN_K_VALUES, curve_errors, strict=True):
        curve_rows.append(
            {
                "k": k,
                "candidate_location_error": error,
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "finite_participants": int(
                    np.isfinite(
                        curve_source.loc[curve_source["k"].eq(k), "candidate_location_error"]
                    ).sum()
                ),
                "nonincreasing_transitions": transitions,
                "k14_below_k3": k14_below_k3,
                "lambda_days": PRIMARY_LAMBDA_DAYS,
                "calibration_mode": "location_only",
                "post_hoc_exploratory": True,
                "status": "observed" if np.isfinite(error) else "insufficient",
            }
        )
    curve = pd.DataFrame.from_records(curve_rows)

    lopo_records: list[dict[str, Any]] = []
    primary = primary.sort_values("_held_token", kind="stable")
    for _, excluded in primary.iterrows():
        retained = primary.loc[primary["_held_token"].ne(excluded["_held_token"])]
        finite = int(np.isfinite(retained["improvement"]).sum())
        complete = len(retained) == 9 and finite == 9
        lopo_records.append(
            {
                "excluded_subject": excluded["held_out_subject"],
                "lopo_mean_improvement": (
                    float(retained["improvement"].mean()) if complete else np.nan
                ),
                "retained_denominator": 9,
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "post_hoc_exploratory": True,
                "_excluded_token": excluded["_held_token"],
            }
        )
    lopo = pd.DataFrame.from_records(lopo_records).sort_values(
        "_excluded_token", kind="stable"
    ).drop(columns="_excluded_token").reset_index(drop=True)

    if primary_prepared is None:
        raise ValueError("Primary lambda=7 location candidate bundle is missing")
    _, _, primary_baselines, primary_oracles, primary_subjects = primary_prepared
    oracle_lookup = primary_oracles.loc[
        primary_oracles["primitive"].isin(_CORE_PRIMITIVES),
        ["_held_token", "primitive", "oracle_center"],
    ].rename(columns={"oracle_center": "_oracle_center"})
    placebo_primitives = primary_baselines.loc[
        primary_baselines["_method"].eq("placebo")
        & primary_baselines["_k"].eq(PRIMARY_K)
        & primary_baselines["primitive"].isin(_CORE_PRIMITIVES)
    ].merge(
        oracle_lookup,
        on=["_held_token", "primitive"],
        how="left",
        validate="many_to_one",
    )
    placebo_primitives["domain"] = placebo_primitives["primitive"].map(_PRIMITIVE_DOMAIN)
    placebo_primitives = placebo_primitives.loc[
        placebo_primitives["domain"].isin(domains)
    ].copy()
    placebo_primitives["candidate_location_error"] = np.abs(
        pd.to_numeric(placebo_primitives["center"], errors="coerce")
        - pd.to_numeric(placebo_primitives["_oracle_center"], errors="coerce")
    ) / pd.to_numeric(placebo_primitives["_anchor_scale"], errors="coerce")
    placebo_primitives["_primitive_complete"] = (
        placebo_primitives["baseline_status"].astype(str).eq("observed")
        & np.isfinite(placebo_primitives["candidate_location_error"])
        & np.isfinite(pd.to_numeric(placebo_primitives["_anchor_scale"], errors="coerce"))
        & pd.to_numeric(placebo_primitives["_anchor_scale"], errors="coerce").gt(0.0)
    )
    placebo_primitives.loc[
        ~placebo_primitives["_primitive_complete"], "candidate_location_error"
    ] = np.nan
    population_error_lookup = primary.set_index("_held_token")[
        "population_location_error"
    ]
    placebo_domain_records: list[dict[str, Any]] = []
    placebo_domain_keys = [
        "_held_token",
        "held_out_subject",
        "_donor_token",
        "donor_subject",
        "domain",
    ]
    for keys, group in placebo_primitives.groupby(
        placebo_domain_keys, sort=False, dropna=False
    ):
        metadata = dict(zip(placebo_domain_keys, keys, strict=True))
        denominator = len(FROZEN_DOMAIN_PRIMITIVES[str(metadata["domain"])])
        finite = int(np.isfinite(group["candidate_location_error"]).sum())
        complete = bool(
            len(group) == denominator
            and group["primitive"].nunique() == denominator
            and finite == denominator
            and group["_primitive_complete"].all()
        )
        placebo_domain_records.append(
            {
                **metadata,
                "candidate_location_error": (
                    float(group["candidate_location_error"].mean())
                    if complete
                    else np.nan
                ),
                "primitive_denominator": denominator,
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
            }
        )
    placebo_domain = pd.DataFrame.from_records(placebo_domain_records)
    placebo_participant_records: list[dict[str, Any]] = []
    placebo_participant_keys = [
        "_held_token",
        "held_out_subject",
        "_donor_token",
        "donor_subject",
    ]
    for keys, group in placebo_domain.groupby(
        placebo_participant_keys, sort=False, dropna=False
    ):
        metadata = dict(zip(placebo_participant_keys, keys, strict=True))
        finite = int(np.isfinite(group["candidate_location_error"]).sum())
        complete = bool(
            len(group) == len(domains)
            and group["domain"].nunique() == len(domains)
            and finite == len(domains)
            and group["status"].eq("observed").all()
        )
        placebo_participant_records.append(
            {
                **metadata,
                "candidate_location_error": (
                    float(group["candidate_location_error"].mean())
                    if complete
                    else np.nan
                ),
                "domain_denominator": len(domains),
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
            }
        )
    placebo_participant = pd.DataFrame.from_records(placebo_participant_records)
    placebo_participant["population_transfer_error"] = placebo_participant[
        "_held_token"
    ].map(population_error_lookup)
    placebo_participant["candidate_transfer_error"] = placebo_participant[
        "candidate_location_error"
    ]
    placebo_participant["improvement"] = (
        placebo_participant["population_transfer_error"]
        - placebo_participant["candidate_transfer_error"]
    )
    placebo_participant["_method"] = "placebo"
    placebo_participant["_k"] = PRIMARY_K
    primary_for_donor = primary.loc[
        :, ["held_out_subject", "candidate_location_error", "improvement", "status", "_held_token"]
    ].rename(columns={"candidate_location_error": "candidate_transfer_error"})
    assignments, donor_distribution, donor_summary = _score_coherent_derangements(
        primary_subjects,
        placebo_participant,
        primary_for_donor,
        draws=int(derangement_draws),
        root_seed=int(root_seed),
    )
    for table in (assignments, donor_distribution, donor_summary):
        table["endpoint"] = "location_transfer"
        table["post_hoc_exploratory"] = True

    max_stat = exact_max_stat_sign_flip(
        participants_frame.loc[
            :, ["candidate_id", "held_out_subject", "improvement"]
        ]
    )
    max_summary = max_stat.candidate_summary.merge(
        ledger.loc[:, ["candidate_id", "primary_candidate"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    primary_max = max_summary.loc[max_summary["primary_candidate"]]
    if len(primary_max) != 1:
        raise RuntimeError("Primary location max-stat candidate is not unique")
    finite_participants = int(np.isfinite(primary["improvement"]).sum())
    improved_participants = int((primary["improvement"] > 0.0).sum())
    minimum_lopo = (
        float(lopo["lopo_mean_improvement"].min())
        if lopo["status"].eq("observed").all()
        else np.nan
    )
    decision = _location_gate_decision(
        improved_participants=improved_participants,
        minimum_lopo=minimum_lopo,
        coherent_p=float(donor_summary.loc[0, "p_value"]),
        nonincreasing_transitions=transitions,
        k14_below_k3=k14_below_k3,
        adjusted_p=float(primary_max["adjusted_p_value"].iloc[0]),
        finite_participants=finite_participants,
    )
    primitive = primitive.sort_values(
        ["candidate_id", "_held_token", "domain", "primitive"], kind="stable"
    ).drop(columns="_held_token").reset_index(drop=True)
    domain_frame = domain_frame.sort_values(
        ["candidate_id", "_held_token", "domain"], kind="stable"
    ).drop(columns="_held_token").reset_index(drop=True)
    public_participants = participants_frame.drop(columns="_held_token").reset_index(drop=True)
    return ExploratoryLocationResult(
        candidate_ledger=ledger,
        primitive_location_errors=primitive,
        domain_location_errors=domain_frame,
        participant_location_deltas=public_participants,
        location_curve_summary=curve,
        location_lopo_summary=lopo,
        coherent_derangement_assignments=assignments,
        coherent_derangement_distribution=donor_distribution,
        coherent_derangement_summary=donor_summary,
        max_stat_candidate_summary=max_summary,
        max_stat_permutation_maxima=max_stat.permutation_maxima,
        max_stat_candidate_permutation_statistics=max_stat.candidate_permutation_statistics,
        location_gate_decision=decision,
    )


def evaluate_exploratory_location_transfer(
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
    future_oracles: pd.DataFrame,
    *,
    validated_domains: Sequence[str],
    trusted_full_calendar_primitives: pd.DataFrame,
    derangement_draws: int = DEFAULT_DERANGEMENT_DRAWS,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> ExploratoryLocationResult:
    """Evaluate G3-L only after an external raw-calendar Task 3 rebuild."""
    authoritative_bundles, source_sha256 = (
        _authoritative_candidate_bundles_from_calendar(
            trusted_full_calendar_primitives, candidate_bundles
        )
    )
    return _evaluate_exploratory_location_transfer_trusted(
        authoritative_bundles,
        future_oracles,
        validated_domains=validated_domains,
        trusted_source_calendar_sha256=source_sha256,
        derangement_draws=derangement_draws,
        root_seed=root_seed,
    )


def _prepare_dual_coordinates(
    bundle: ExploratoryCandidateBundle,
    frozen_representations: pd.DataFrame,
    frozen_baselines: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(bundle.dual_tables, DualCalibrationTables):
        raise ValueError("Every G3-R candidate bundle requires frozen dual coordinates")
    coordinates = bundle.dual_tables.coordinates.copy(deep=True)
    policy = bundle.dual_tables.coordinate_policy.copy(deep=True)
    coordinate_columns = tuple(
        column
        for primitive in _ALL_PRIMITIVES
        for column in (
            primitive,
            f"{primitive}__confidence",
            f"{primitive}__coverage_only_confidence",
            f"{primitive}__value_only_confidence",
            f"{primitive}__population_scale",
            f"{primitive}__reference_z_unclipped",
            f"{primitive}__reference_z",
            f"{primitive}__self_z_unclipped",
            f"{primitive}__self_z",
        )
    )
    _require_columns(
        coordinates,
        (
            "sensor_day_id",
            "subject_id",
            "held_out_subject",
            "elapsed_day_index",
            "k",
            "method",
            "donor_subject",
            "dual_calibration_protocol",
            "dual_calibration_post_hoc_exploratory",
            "task3_artifact_sha256",
            "task3_lambda_days",
            "task3_usage_confidence_policy",
            *coordinate_columns,
        ),
        "dual coordinate",
    )
    _require_columns(
        policy,
        (
            "output_column",
            "consumer_role",
            "coordinate_policy",
            "post_hoc_exploratory",
        ),
        "dual coordinate policy",
    )
    if coordinates.empty or policy.empty:
        raise ValueError("Dual coordinates and role policy must be nonempty")
    if not coordinates["dual_calibration_protocol"].astype(str).eq(
        DUAL_CALIBRATION_PROTOCOL
    ).all():
        raise ValueError("Dual provenance protocol disagrees with the Task 3 artifact")
    expected_artifact_sha256 = exploratory_bundle_artifact_sha256(bundle)
    if not coordinates["task3_artifact_sha256"].astype(str).eq(
        expected_artifact_sha256
    ).all():
        raise ValueError("Dual provenance is cross-wired to a different Task 3 artifact")
    if not pd.to_numeric(
        coordinates["task3_lambda_days"], errors="coerce"
    ).eq(float(bundle.lambda_days)).all():
        raise ValueError("Dual provenance lambda is cross-wired to another bundle")
    if not coordinates["task3_usage_confidence_policy"].astype(str).eq(
        bundle.usage_confidence_policy
    ).all():
        raise ValueError("Dual provenance Usage policy is cross-wired to another bundle")
    if not coordinates["dual_calibration_post_hoc_exploratory"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).all():
        raise ValueError("Every dual coordinate must be explicitly post-hoc exploratory")
    if not policy["post_hoc_exploratory"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    ).all():
        raise ValueError("Every dual coordinate policy must be post-hoc exploratory")
    expected_policy = coordinate_policy_table().sort_values(
        "output_column", kind="stable"
    ).reset_index(drop=True)
    actual_policy = policy.sort_values("output_column", kind="stable").reset_index(
        drop=True
    )
    try:
        pd.testing.assert_frame_equal(actual_policy, expected_policy)
    except AssertionError as error:
        raise ValueError(
            "Dual provenance coordinate policy must equal the authoritative policy"
        ) from error
    cross_modal = policy["consumer_role"].astype(str).eq("cross_modal")
    if not cross_modal.any() or not policy.loc[
        cross_modal, "coordinate_policy"
    ].astype(str).eq("reference_z").all():
        raise ValueError("Every cross-modal consumer must use reference_z")
    routine = policy["consumer_role"].astype(str).eq("routine_anomaly")
    if routine.any() and not policy.loc[routine, "coordinate_policy"].astype(str).eq(
        "self_z"
    ).all():
        raise ValueError("Every routine anomaly consumer must use self_z")
    forbidden = [
        str(column)
        for column in coordinates.columns
        if re.fullmatch(r"[QS]\d+", str(column), flags=re.IGNORECASE)
        or "oracle" in str(column).lower()
        or "outcome" in str(column).lower()
        or "argmax" in str(column).lower()
        or "selected_candidate" in str(column).lower()
    ]
    if forbidden:
        raise ValueError(f"Dual coordinates contain forbidden selector data: {forbidden}")
    held_tokens = coordinates["held_out_subject"].map(_identity_token)
    subject_tokens = coordinates["subject_id"].map(_identity_token)
    if not held_tokens.equals(subject_tokens):
        raise ValueError("Dual coordinate subject_id must equal held_out_subject")
    coordinate_keys = _group_key_frame(coordinates)
    coordinates = pd.concat(
        [coordinates.reset_index(drop=True), coordinate_keys.reset_index(drop=True)],
        axis=1,
    )
    coordinates["_sensor_token"] = coordinates["sensor_day_id"].map(_identity_token)
    coordinates["_method_order"] = coordinates["_method"].map(
        {"population": 0, "personal": 1, "shrinkage": 2, "placebo": 3}
    )
    coordinates = coordinates.sort_values(
        ["_method_order", "_k", "_held_token", "_donor_token", "_sensor_token"],
        kind="stable",
    ).reset_index(drop=True)
    expected_keys = list(
        frozen_representations.loc[
            :,
            ["_held_token", "_k", "_method", "_donor_token", "_sensor_token"],
        ].itertuples(index=False, name=None)
    )
    actual_keys = list(
        coordinates.loc[
            :,
            ["_held_token", "_k", "_method", "_donor_token", "_sensor_token"],
        ].itertuples(index=False, name=None)
    )
    if actual_keys != expected_keys:
        raise ValueError("Dual coordinates must preserve the exact frozen common topology")
    baseline_keys = ["_held_token", "_k", "_method", "_donor_token"]
    coordinate_group_keys = coordinates.loc[:, baseline_keys]
    population_anchors = frozen_baselines.loc[
        frozen_baselines["_method"].eq("population")
        & frozen_baselines["_k"].eq(0)
        & frozen_baselines["_donor_token"].eq(_NULL_DONOR),
        ["_held_token", "primitive", "scale"],
    ].rename(columns={"scale": "_authoritative_population_scale"})
    if population_anchors.duplicated(["_held_token", "primitive"]).any():
        raise ValueError("Dual provenance population anchors must be unique")
    for primitive in _ALL_PRIMITIVES:
        primitive_baseline = frozen_baselines.loc[
            frozen_baselines["primitive"].eq(primitive),
            [*baseline_keys, "center", "scale", "baseline_status"],
        ]
        aligned = coordinate_group_keys.merge(
            primitive_baseline,
            on=baseline_keys,
            how="left",
            sort=False,
            validate="many_to_one",
        ).merge(
            population_anchors.loc[
                population_anchors["primitive"].eq(primitive),
                ["_held_token", "_authoritative_population_scale"],
            ],
            on="_held_token",
            how="left",
            sort=False,
            validate="many_to_one",
        )
        if len(aligned) != len(coordinates):
            raise ValueError(f"Dual provenance {primitive} baseline alignment is incomplete")
        for column in (
            primitive,
            f"{primitive}__confidence",
            f"{primitive}__coverage_only_confidence",
            f"{primitive}__value_only_confidence",
        ):
            if not _numeric_equal(
                coordinates[column], frozen_representations[column]
            ).all():
                raise ValueError(
                    f"Dual provenance {primitive} raw/confidence copy disagrees with Task 3"
                )
        if not _numeric_equal(
            coordinates[f"{primitive}__population_scale"],
            aligned["_authoritative_population_scale"],
        ).all():
            raise ValueError(
                f"Dual provenance {primitive} population scale disagrees with Task 3"
            )
        values = pd.to_numeric(
            frozen_representations[primitive], errors="coerce"
        ).to_numpy(dtype=float)
        confidence = pd.to_numeric(
            frozen_representations[f"{primitive}__confidence"], errors="coerce"
        ).to_numpy(dtype=float)
        center = pd.to_numeric(aligned["center"], errors="coerce").to_numpy(
            dtype=float
        )
        self_scale = pd.to_numeric(aligned["scale"], errors="coerce").to_numpy(
            dtype=float
        )
        population_scale = pd.to_numeric(
            aligned["_authoritative_population_scale"], errors="coerce"
        ).to_numpy(dtype=float)
        observed = aligned["baseline_status"].astype(str).eq("observed").to_numpy()
        common = (
            np.isfinite(values)
            & np.isfinite(confidence)
            & (confidence > 0.0)
            & np.isfinite(center)
            & observed
        )
        reference_expected = np.full(len(coordinates), np.nan, dtype=float)
        self_expected = np.full(len(coordinates), np.nan, dtype=float)
        reference_usable = common & np.isfinite(population_scale) & (
            population_scale > 0.0
        )
        self_usable = common & np.isfinite(self_scale) & (self_scale > 0.0)
        reference_expected[reference_usable] = (
            values[reference_usable] - center[reference_usable]
        ) / population_scale[reference_usable]
        self_expected[self_usable] = (
            values[self_usable] - center[self_usable]
        ) / self_scale[self_usable]
        expected_coordinates = (
            (f"{primitive}__reference_z_unclipped", reference_expected, "reference formula"),
            (
                f"{primitive}__reference_z",
                np.clip(reference_expected, -5.0, 5.0),
                "reference clipping formula",
            ),
            (f"{primitive}__self_z_unclipped", self_expected, "self formula"),
            (
                f"{primitive}__self_z",
                np.clip(self_expected, -5.0, 5.0),
                "self clipping formula",
            ),
        )
        for column, expected_values, label in expected_coordinates:
            if not _numeric_equal(
                coordinates[column], pd.Series(expected_values)
            ).all():
                raise ValueError(f"Dual provenance {primitive} {label} mismatch")
    return coordinates


def _reference_pair_day_records(
    population: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    metadata: dict[str, Any],
    clipping_state: str,
) -> list[dict[str, Any]]:
    suffix = "" if clipping_state == "clipped" else "_unclipped"
    value_columns = [
        f"{primitive}__reference_z{suffix}" for primitive in _CORE_PRIMITIVES
    ]
    keys = ["_held_token", "_sensor_token"]
    population_columns = [
        *keys,
        "sensor_day_id",
        "held_out_subject",
        "elapsed_day_index",
        *value_columns,
    ]
    candidate_columns = [*keys, *value_columns]
    if population.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError("Reference coordinate row keys must be unique per candidate")
    merged = population.loc[:, population_columns].merge(
        candidate.loc[:, candidate_columns],
        on=keys,
        how="outer",
        suffixes=("_population", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Population and candidate must retain the identical row grid")
    records: list[dict[str, Any]] = []
    for pair, (domain, left, right) in FROZEN_PAIR_SOURCES.items():
        population_left = pd.to_numeric(
            merged[f"{left}__reference_z{suffix}_population"], errors="coerce"
        ).to_numpy(dtype=float)
        population_right = pd.to_numeric(
            merged[f"{right}__reference_z{suffix}_population"], errors="coerce"
        ).to_numpy(dtype=float)
        candidate_left = pd.to_numeric(
            merged[f"{left}__reference_z{suffix}_candidate"], errors="coerce"
        ).to_numpy(dtype=float)
        candidate_right = pd.to_numeric(
            merged[f"{right}__reference_z{suffix}_candidate"], errors="coerce"
        ).to_numpy(dtype=float)
        population_finite = np.isfinite(population_left) & np.isfinite(population_right)
        candidate_finite = np.isfinite(candidate_left) & np.isfinite(candidate_right)
        if not np.array_equal(population_finite, candidate_finite):
            raise ValueError(
                f"{pair} population and candidate require an identical finite row set"
            )
        for index, row in merged.iterrows():
            finite = bool(population_finite[index])
            population_discrepancy = (
                abs(population_left[index] - population_right[index])
                if finite
                else np.nan
            )
            candidate_discrepancy = (
                abs(candidate_left[index] - candidate_right[index])
                if finite
                else np.nan
            )
            records.append(
                {
                    **metadata,
                    "sensor_day_id": row["sensor_day_id"],
                    "held_out_subject": row["held_out_subject"],
                    "elapsed_day_index": int(row["elapsed_day_index"]),
                    "_held_token": row["_held_token"],
                    "_sensor_token": row["_sensor_token"],
                    "pair": pair,
                    "domain": domain,
                    "left_primitive": left,
                    "right_primitive": right,
                    "clipping_state": clipping_state,
                    "population_discrepancy": population_discrepancy,
                    "candidate_discrepancy": candidate_discrepancy,
                    "discrepancy_improvement": (
                        population_discrepancy - candidate_discrepancy
                        if finite
                        else np.nan
                    ),
                    "population_sign_agreement": (
                        float(np.sign(population_left[index]) == np.sign(population_right[index]))
                        if finite
                        else np.nan
                    ),
                    "candidate_sign_agreement": (
                        float(np.sign(candidate_left[index]) == np.sign(candidate_right[index]))
                        if finite
                        else np.nan
                    ),
                    "finite_count": int(finite),
                    "status": "observed" if finite else "common_abstention",
                    "post_hoc_exploratory": True,
                }
            )
    return records


def _summarize_reference_days(
    pair_days: pd.DataFrame,
    *,
    candidate_metadata: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_group_columns = [*candidate_metadata, "held_out_subject", "_held_token", "pair"]
    pair_records: list[dict[str, Any]] = []
    for keys, group in pair_days.groupby(pair_group_columns, sort=False, dropna=False):
        metadata = dict(zip(pair_group_columns, keys, strict=True))
        finite = int(group["finite_count"].sum())
        pair_records.append(
            {
                **metadata,
                "domain": group["domain"].iloc[0],
                "left_primitive": group["left_primitive"].iloc[0],
                "right_primitive": group["right_primitive"].iloc[0],
                "population_discrepancy": (
                    float(group["population_discrepancy"].mean()) if finite else np.nan
                ),
                "candidate_discrepancy": (
                    float(group["candidate_discrepancy"].mean()) if finite else np.nan
                ),
                "discrepancy_improvement": (
                    float(group["discrepancy_improvement"].mean()) if finite else np.nan
                ),
                "population_sign_agreement": (
                    float(group["population_sign_agreement"].mean()) if finite else np.nan
                ),
                "candidate_sign_agreement": (
                    float(group["candidate_sign_agreement"].mean()) if finite else np.nan
                ),
                "sign_agreement_delta": (
                    float(
                        group["candidate_sign_agreement"].mean()
                        - group["population_sign_agreement"].mean()
                    )
                    if finite
                    else np.nan
                ),
                "row_denominator": len(group),
                "finite_count": finite,
                "status": "observed" if finite else "insufficient",
                "post_hoc_exploratory": True,
            }
        )
    pair_summary = pd.DataFrame.from_records(pair_records)
    participant_group_columns = [
        *candidate_metadata,
        "held_out_subject",
        "_held_token",
    ]
    participant_records: list[dict[str, Any]] = []
    for keys, group in pair_summary.groupby(
        participant_group_columns, sort=False, dropna=False
    ):
        metadata = dict(zip(participant_group_columns, keys, strict=True))
        finite = int(np.isfinite(group["candidate_discrepancy"]).sum())
        complete = len(group) == len(FROZEN_PAIR_SOURCES) and finite == len(
            FROZEN_PAIR_SOURCES
        )
        participant_records.append(
            {
                **metadata,
                "population_discrepancy": (
                    float(group["population_discrepancy"].mean()) if complete else np.nan
                ),
                "candidate_discrepancy": (
                    float(group["candidate_discrepancy"].mean()) if complete else np.nan
                ),
                "discrepancy_improvement": (
                    float(group["discrepancy_improvement"].mean()) if complete else np.nan
                ),
                "population_sign_agreement": (
                    float(group["population_sign_agreement"].mean()) if complete else np.nan
                ),
                "candidate_sign_agreement": (
                    float(group["candidate_sign_agreement"].mean()) if complete else np.nan
                ),
                "sign_agreement_delta": (
                    float(group["sign_agreement_delta"].mean()) if complete else np.nan
                ),
                "pair_denominator": len(FROZEN_PAIR_SOURCES),
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "post_hoc_exploratory": True,
            }
        )
    return pair_summary, pd.DataFrame.from_records(participant_records)


def _reference_gate_decision(
    *,
    overall_improvement: float,
    improved_participants: int,
    sign_agreement_delta: float,
    coherent_p: float,
    minimum_lopo: float,
    clipped_improvement: float,
    unclipped_improvement: float,
    adjusted_p: float,
    finite_participants: int,
) -> pd.DataFrame:
    complete = int(finite_participants) == FROZEN_PARTICIPANT_DENOMINATOR
    checks = (
        ("overall_reference_discrepancy_improves", overall_improvement, 0.0, ">", overall_improvement > 0.0),
        ("at_least_8_of_10_participants_improve", improved_participants, 8, ">=", improved_participants >= 8),
        ("sign_agreement_delta_nonnegative", sign_agreement_delta, 0.0, ">=", sign_agreement_delta >= 0.0),
        ("coherent_donor_reference_p_at_most_005", coherent_p, 0.05, "<=", coherent_p <= 0.05),
        ("all_lopo_means_positive", minimum_lopo, 0.0, ">", minimum_lopo > 0.0),
        (
            "clipped_and_unclipped_improvement_same_positive_direction",
            min(clipped_improvement, unclipped_improvement),
            0.0,
            ">",
            clipped_improvement > 0.0 and unclipped_improvement > 0.0,
        ),
        ("selection_adjusted_reference_p_at_most_005", adjusted_p, 0.05, "<=", adjusted_p <= 0.05),
    )
    rows = [
        {
            "gate": gate,
            "observed": observed,
            "threshold": threshold,
            "comparator": comparator,
            "passed": bool(complete and np.isfinite(observed) and passed),
            "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
            "finite_participants": int(finite_participants),
            "status": "observed" if complete else "insufficient",
            "reason": (
                "criterion_met"
                if complete and np.isfinite(observed) and passed
                else "fixed_participant_denominator_incomplete"
                if not complete
                else "criterion_not_met"
            ),
            "post_hoc_exploratory": True,
        }
        for gate, observed, threshold, comparator, passed in checks
    ]
    final_pass = bool(complete and all(row["passed"] for row in rows))
    rows.append(
        {
            "gate": "exploratory_g3_r_pass",
            "observed": int(sum(row["passed"] for row in rows)),
            "threshold": len(checks),
            "comparator": "all",
            "passed": final_pass,
            "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
            "finite_participants": int(finite_participants),
            "status": "observed" if complete else "insufficient",
            "reason": (
                "all_exploratory_reference_criteria_met"
                if final_pass
                else "fixed_participant_denominator_incomplete"
                if not complete
                else "one_or_more_exploratory_reference_criteria_failed"
            ),
            "post_hoc_exploratory": True,
        }
    )
    return pd.DataFrame.from_records(rows)


def _evaluate_exploratory_reference_alignment_trusted(
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
    future_oracles: pd.DataFrame,
    *,
    trusted_source_calendar_sha256: str,
    derangement_draws: int = DEFAULT_DERANGEMENT_DRAWS,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> ExploratoryReferenceResult:
    """Evaluate G3-R using only role-locked Phase 3B reference coordinates."""
    if (
        isinstance(derangement_draws, bool)
        or not isinstance(derangement_draws, (int, np.integer))
        or int(derangement_draws) <= 0
    ):
        raise ValueError("derangement_draws must be an exact positive integer")
    if isinstance(root_seed, bool) or not isinstance(root_seed, (int, np.integer)):
        raise ValueError("root_seed must be an exact integer")
    prepared = _canonical_candidate_bundles(
        candidate_bundles,
        future_oracles,
        trusted_source_calendar_sha256=trusted_source_calendar_sha256,
    )
    dual_prepared = [
        (
            bundle,
            _prepare_dual_coordinates(bundle, representations, baselines),
            participants,
        )
        for bundle, representations, baselines, _, participants in prepared
    ]
    ledger = exploratory_candidate_ledger()
    ledger["available"] = False
    ledger["availability_reason"] = "candidate_bundle_not_supplied"
    ledger["reference_family"] = False
    ledger["primary_candidate"] = False
    day_records: list[dict[str, Any]] = []
    primary_bundle_coordinates: tuple[
        ExploratoryCandidateBundle, pd.DataFrame, list[Any]
    ] | None = None
    for bundle, coordinates, participants in dual_prepared:
        is_primary_bundle = bool(
            float(bundle.lambda_days) == PRIMARY_LAMBDA_DAYS
            and bundle.usage_confidence_policy == "screen_proxy"
            and bundle.usage_operator == "frozen"
            and float(bundle.quality_threshold) == 0.0
            and bundle.population_weighting == "row_weighted"
        )
        if is_primary_bundle:
            primary_bundle_coordinates = (bundle, coordinates, participants)
        population = coordinates.loc[
            coordinates["_method"].eq("population") & coordinates["_k"].eq(0)
        ]
        for k in FROZEN_K_VALUES:
            candidate = coordinates.loc[
                coordinates["_method"].eq("shrinkage") & coordinates["_k"].eq(k)
            ]
            for clipping_state in ("clipped", "unclipped"):
                candidate_id = _candidate_id_from_ledger(
                    ledger,
                    calibration_mode="location_only",
                    k=k,
                    bundle=bundle,
                    clipping_state=clipping_state,
                )
                selected = ledger["candidate_id"].eq(candidate_id)
                ledger.loc[selected, "available"] = True
                ledger.loc[selected, "availability_reason"] = "scored_g3_r_reference_z"
                ledger.loc[selected, "reference_family"] = True
                is_primary = bool(
                    is_primary_bundle
                    and k == PRIMARY_K
                    and clipping_state == "clipped"
                )
                ledger.loc[selected, "primary_candidate"] = is_primary
                day_records.extend(
                    _reference_pair_day_records(
                        population,
                        candidate,
                        metadata={
                            "candidate_id": candidate_id,
                            "bundle_id": bundle.bundle_id,
                            "calibration_mode": "location_only",
                            "k": int(k),
                            "lambda_days": float(bundle.lambda_days),
                            "usage_confidence_policy": bundle.usage_confidence_policy,
                            "usage_operator": bundle.usage_operator,
                            "quality_threshold": float(bundle.quality_threshold),
                            "population_weighting": bundle.population_weighting,
                            "primary_candidate": is_primary,
                        },
                        clipping_state=clipping_state,
                    )
                )
    pair_days = pd.DataFrame.from_records(day_records)
    candidate_metadata = [
        "candidate_id",
        "bundle_id",
        "calibration_mode",
        "k",
        "lambda_days",
        "usage_confidence_policy",
        "usage_operator",
        "quality_threshold",
        "population_weighting",
        "clipping_state",
        "primary_candidate",
    ]
    pair_summary, participant_summary = _summarize_reference_days(
        pair_days, candidate_metadata=candidate_metadata
    )
    participant_summary = participant_summary.sort_values(
        ["candidate_id", "_held_token"], kind="stable"
    ).reset_index(drop=True)
    primary = participant_summary.loc[participant_summary["primary_candidate"]].copy()
    if len(primary) != FROZEN_PARTICIPANT_DENOMINATOR:
        raise ValueError("Primary reference candidate requires exactly 10 participants")
    primary = primary.sort_values("_held_token", kind="stable")
    lopo_records: list[dict[str, Any]] = []
    for _, excluded in primary.iterrows():
        retained = primary.loc[primary["_held_token"].ne(excluded["_held_token"])]
        finite = int(np.isfinite(retained["discrepancy_improvement"]).sum())
        complete = len(retained) == 9 and finite == 9
        lopo_records.append(
            {
                "excluded_subject": excluded["held_out_subject"],
                "lopo_mean_improvement": (
                    float(retained["discrepancy_improvement"].mean()) if complete else np.nan
                ),
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "retained_denominator": 9,
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "post_hoc_exploratory": True,
                "_excluded_token": excluded["_held_token"],
            }
        )
    lopo = pd.DataFrame.from_records(lopo_records).sort_values(
        "_excluded_token", kind="stable"
    ).drop(columns="_excluded_token").reset_index(drop=True)

    if primary_bundle_coordinates is None:
        raise ValueError("Primary lambda=7 reference bundle is missing")
    _, primary_coordinates, primary_subjects = primary_bundle_coordinates
    population = primary_coordinates.loc[
        primary_coordinates["_method"].eq("population")
        & primary_coordinates["_k"].eq(0)
    ]
    placebo_records: list[dict[str, Any]] = []
    placebo = primary_coordinates.loc[
        primary_coordinates["_method"].eq("placebo")
        & primary_coordinates["_k"].eq(PRIMARY_K)
    ]
    for (held_token, donor_token), group in placebo.groupby(
        ["_held_token", "_donor_token"], sort=False, dropna=False
    ):
        placebo_records.extend(
            _reference_pair_day_records(
                population.loc[population["_held_token"].eq(held_token)],
                group,
                metadata={
                    "donor_subject": group["donor_subject"].iloc[0],
                    "_donor_token": donor_token,
                },
                clipping_state="clipped",
            )
        )
    placebo_days = pd.DataFrame.from_records(placebo_records)
    _, placebo_participants = _summarize_reference_days(
        placebo_days,
        candidate_metadata=["donor_subject", "_donor_token", "clipping_state"],
    )
    primary_population_lookup = primary.set_index("_held_token")[
        "population_discrepancy"
    ]
    placebo_participants["population_transfer_error"] = placebo_participants[
        "_held_token"
    ].map(primary_population_lookup)
    placebo_participants["candidate_transfer_error"] = placebo_participants[
        "candidate_discrepancy"
    ]
    placebo_participants["improvement"] = (
        placebo_participants["population_transfer_error"]
        - placebo_participants["candidate_transfer_error"]
    )
    placebo_participants["_method"] = "placebo"
    placebo_participants["_k"] = PRIMARY_K
    primary_for_donor = primary.loc[
        :, ["held_out_subject", "candidate_discrepancy", "discrepancy_improvement", "status", "_held_token"]
    ].rename(
        columns={
            "candidate_discrepancy": "candidate_transfer_error",
            "discrepancy_improvement": "improvement",
        }
    )
    assignments, donor_distribution, donor_summary = _score_coherent_derangements(
        primary_subjects,
        placebo_participants,
        primary_for_donor,
        draws=int(derangement_draws),
        root_seed=int(root_seed),
    )
    for table in (assignments, donor_distribution, donor_summary):
        table["endpoint"] = "reference_alignment"
        table["post_hoc_exploratory"] = True

    max_stat = exact_max_stat_sign_flip(
        participant_summary.loc[
            :, ["candidate_id", "held_out_subject", "discrepancy_improvement"]
        ].rename(columns={"discrepancy_improvement": "improvement"})
    )
    max_summary = max_stat.candidate_summary.merge(
        ledger.loc[:, ["candidate_id", "primary_candidate"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    primary_max = max_summary.loc[max_summary["primary_candidate"]]
    if len(primary_max) != 1:
        raise RuntimeError("Primary reference max-stat candidate is not unique")
    finite_participants = int(np.isfinite(primary["discrepancy_improvement"]).sum())
    overall_improvement = (
        float(primary["discrepancy_improvement"].mean())
        if finite_participants == FROZEN_PARTICIPANT_DENOMINATOR
        else np.nan
    )
    sign_delta = (
        float(primary["sign_agreement_delta"].mean())
        if finite_participants == FROZEN_PARTICIPANT_DENOMINATOR
        else np.nan
    )
    unclipped = participant_summary.loc[
        participant_summary["calibration_mode"].eq("location_only")
        & participant_summary["k"].eq(PRIMARY_K)
        & participant_summary["lambda_days"].eq(PRIMARY_LAMBDA_DAYS)
        & participant_summary["usage_confidence_policy"].eq("screen_proxy")
        & participant_summary["usage_operator"].eq("frozen")
        & participant_summary["quality_threshold"].eq(0.0)
        & participant_summary["population_weighting"].eq("row_weighted")
        & participant_summary["clipping_state"].eq("unclipped")
    ]
    unclipped_improvement = (
        float(unclipped["discrepancy_improvement"].mean())
        if len(unclipped) == FROZEN_PARTICIPANT_DENOMINATOR
        else np.nan
    )
    decision = _reference_gate_decision(
        overall_improvement=overall_improvement,
        improved_participants=int((primary["discrepancy_improvement"] > 0.0).sum()),
        sign_agreement_delta=sign_delta,
        coherent_p=float(donor_summary.loc[0, "p_value"]),
        minimum_lopo=(
            float(lopo["lopo_mean_improvement"].min())
            if lopo["status"].eq("observed").all()
            else np.nan
        ),
        clipped_improvement=overall_improvement,
        unclipped_improvement=unclipped_improvement,
        adjusted_p=float(primary_max["adjusted_p_value"].iloc[0]),
        finite_participants=finite_participants,
    )
    pair_days = pair_days.sort_values(
        ["candidate_id", "_held_token", "pair", "_sensor_token"], kind="stable"
    ).drop(columns=["_held_token", "_sensor_token"]).reset_index(drop=True)
    pair_summary = pair_summary.sort_values(
        ["candidate_id", "_held_token", "pair"], kind="stable"
    ).drop(columns="_held_token").reset_index(drop=True)
    public_participants = participant_summary.drop(columns="_held_token").reset_index(drop=True)
    return ExploratoryReferenceResult(
        candidate_ledger=ledger,
        pair_day_discrepancies=pair_days,
        pair_discrepancy_summaries=pair_summary,
        participant_reference_deltas=public_participants,
        reference_lopo_summary=lopo,
        coherent_derangement_assignments=assignments,
        coherent_derangement_distribution=donor_distribution,
        coherent_derangement_summary=donor_summary,
        max_stat_candidate_summary=max_summary,
        max_stat_permutation_maxima=max_stat.permutation_maxima,
        max_stat_candidate_permutation_statistics=max_stat.candidate_permutation_statistics,
        reference_gate_decision=decision,
    )


def evaluate_exploratory_reference_alignment(
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
    future_oracles: pd.DataFrame,
    *,
    trusted_full_calendar_primitives: pd.DataFrame,
    derangement_draws: int = DEFAULT_DERANGEMENT_DRAWS,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> ExploratoryReferenceResult:
    """Evaluate G3-R only after an external raw-calendar Task 3 rebuild."""
    authoritative_bundles, source_sha256 = (
        _authoritative_candidate_bundles_from_calendar(
            trusted_full_calendar_primitives, candidate_bundles
        )
    )
    return _evaluate_exploratory_reference_alignment_trusted(
        authoritative_bundles,
        future_oracles,
        trusted_source_calendar_sha256=source_sha256,
        derangement_draws=derangement_draws,
        root_seed=root_seed,
    )


def _identity_token_cache_key(value: Any) -> tuple[Any, ...] | None:
    """Return a safe cache key without weakening ``_identity_token`` semantics."""
    value_type = type(value)
    if isinstance(value, (float, np.floating)):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(numeric):
            # JSON normalizes every payload-level NaN to the same literal while
            # the scalar type remains part of the identity token.
            return (value_type, "nan")
        if numeric == 0.0:
            return (value_type, "zero", bool(np.signbit(numeric)))
    elif value_type not in {str, int, bool, type(None)}:
        # Equality is not a representation identity for arbitrary hashable
        # scalars.  Decimal scale and Timestamp timezone are two important
        # examples, so only exact built-in scalar classes whose equal values
        # have one canonical JSON/repr token are memoized.
        return None
    try:
        hash(value)
    except (TypeError, ValueError):
        return None
    return (value_type, value)


def _memoized_identity_token(
    value: Any, cache: dict[tuple[Any, ...], str]
) -> str:
    key = _identity_token_cache_key(value)
    if key is None:
        return _identity_token(value)
    try:
        return cache[key]
    except KeyError:
        token = _identity_token(value)
        cache[key] = token
        return token
    except (TypeError, ValueError):
        # Exotic hashable scalars can still define non-boolean equality.  They
        # remain correct by taking the uncached canonical path.
        return _identity_token(value)


def _stable_frame_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Stream the exact historical JSON token representation into SHA-256."""
    digest = hashlib.sha256()
    digest.update(b"[")
    first_row = True
    cache: dict[tuple[Any, ...], str] = {}
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        if first_row:
            first_row = False
        else:
            digest.update(b",")
        tokens = [_memoized_identity_token(value, cache) for value in row]
        digest.update(
            json.dumps(
                tokens, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        )
    digest.update(b"]")
    return digest.hexdigest()


def _canonical_raw_artifact_frame_sha256(
    frame: pd.DataFrame, *, key_columns: Sequence[str]
) -> str:
    if frame.columns.duplicated().any():
        raise ValueError("Task 3 artifact columns must be unique before fingerprinting")
    ordered = frame.copy(deep=True)
    sort_columns: list[str] = []
    for index, column in enumerate(key_columns):
        if column not in ordered.columns:
            raise ValueError(f"Task 3 artifact fingerprint is missing key: {column}")
        token_column = f"__artifact_sort_{index}"
        ordered[token_column] = ordered[column].map(_identity_token)
        sort_columns.append(token_column)
    ordered = ordered.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    columns = sorted(
        (column for column in frame.columns), key=lambda value: str(value)
    )
    return _stable_frame_sha256(ordered, columns)


def _stream_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_canonical_input_manifest(
    canonical_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Fingerprint every canonical source file with stable relative identity."""
    root = Path(canonical_root)
    if not root.is_dir():
        raise ValueError("canonical input root must be an existing directory")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise ValueError("canonical input root must contain at least one source file")
    manifest: list[dict[str, Any]] = []
    for path in paths:
        if path.is_symlink():
            raise ValueError("canonical input manifest does not accept symbolic links")
        relative_path = path.relative_to(root).as_posix()
        manifest.append(
            {
                "relative_path": relative_path,
                "size_bytes": int(path.stat().st_size),
                "sha256": _stream_file_sha256(path),
            }
        )
    return tuple(manifest)


def canonical_input_manifest_sha256(
    canonical_manifest: Sequence[Mapping[str, Any]],
) -> str:
    if isinstance(canonical_manifest, (str, bytes)) or not isinstance(
        canonical_manifest, Sequence
    ):
        raise ValueError("canonical input manifest must be a nonempty sequence")
    entries = [dict(entry) for entry in canonical_manifest]
    if not entries:
        raise ValueError("canonical input manifest must be a nonempty sequence")
    relative_paths: list[str] = []
    for entry in entries:
        if set(entry) != {"relative_path", "size_bytes", "sha256"}:
            raise ValueError("canonical input manifest entry schema is invalid")
        relative_path = entry["relative_path"]
        size_bytes = entry["size_bytes"]
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("canonical input relative_path must be nonempty")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("canonical input size_bytes must be a nonnegative integer")
        _canonical_source_calendar_sha256(
            entry["sha256"], name=f"canonical input {relative_path!r} fingerprint"
        )
        relative_paths.append(relative_path)
    if relative_paths != sorted(relative_paths) or len(relative_paths) != len(
        set(relative_paths)
    ):
        raise ValueError("canonical input manifest paths must be sorted and unique")
    payload = json.dumps(
        entries, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def production_candidate_cache_key(
    *,
    canonical_manifest: Sequence[Mapping[str, Any]],
    extraction_source_sha256: Mapping[str, str],
    extraction_policy: Mapping[str, Any],
) -> str:
    """Bind production candidate caches to data bytes, code, and extraction policy."""
    if not isinstance(extraction_source_sha256, Mapping) or not extraction_source_sha256:
        raise ValueError("extraction source fingerprints must be a nonempty mapping")
    sources: dict[str, str] = {}
    for name, digest in extraction_source_sha256.items():
        if not isinstance(name, str) or not name:
            raise ValueError("extraction source name must be nonempty")
        sources[name] = _canonical_source_calendar_sha256(
            digest, name=f"extraction source {name!r} fingerprint"
        )
    if not isinstance(extraction_policy, Mapping) or not extraction_policy:
        raise ValueError("extraction policy must be a nonempty mapping")
    payload = {
        "canonical_manifest_sha256": canonical_input_manifest_sha256(
            canonical_manifest
        ),
        "canonical_manifest": [dict(entry) for entry in canonical_manifest],
        "extraction_source_sha256": dict(sorted(sources.items())),
        "extraction_policy": dict(extraction_policy),
    }
    try:
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("extraction policy must be canonical JSON data") from error
    return hashlib.sha256(encoded).hexdigest()


def _canonical_source_calendar_sha256(value: Any, *, name: str) -> str:
    """Return one exact externally rooted SHA-256 identity or fail closed."""
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be an exact lowercase 64-hex SHA-256")
    return value


def full_calendar_task3_sha256(full_calendar_primitives: pd.DataFrame) -> str:
    """Fingerprint the exact outcome-free primitive calendar supplied to Task 3."""
    if not isinstance(full_calendar_primitives, pd.DataFrame):
        raise TypeError("full_calendar_primitives must be a pandas DataFrame")
    metadata_columns = (
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
    source_columns = list(metadata_columns)
    for definition in PRIMITIVE_DEFINITIONS:
        source_columns.extend(
            (
                definition.name,
                f"{definition.name}__status",
                f"{definition.name}__availability_coverage",
                f"{definition.name}__longest_gap_minutes",
                f"{definition.name}__expected_epochs",
                f"{definition.name}__support_density",
            )
        )
        censoring = f"{definition.name}__censoring"
        if censoring in full_calendar_primitives.columns:
            source_columns.append(censoring)
    _require_columns(
        full_calendar_primitives,
        source_columns,
        "full-calendar Task 3 source fingerprint",
    )
    return _canonical_raw_artifact_frame_sha256(
        full_calendar_primitives.loc[:, source_columns],
        key_columns=("sensor_day_id",),
    )


def exploratory_bundle_artifact_sha256(
    bundle: ExploratoryCandidateBundle,
) -> str:
    """Fingerprint every outcome-free source table plus declared bundle metadata."""
    if not isinstance(bundle, ExploratoryCandidateBundle):
        raise TypeError("bundle must be an ExploratoryCandidateBundle")
    tables = bundle.loso_tables
    source_calendar_sha256 = _canonical_source_calendar_sha256(
        bundle.source_calendar_sha256,
        name="bundle source calendar fingerprint",
    )
    payload = {
        "bundle_id": bundle.bundle_id,
        "lambda_days": float(bundle.lambda_days),
        "usage_confidence_policy": bundle.usage_confidence_policy,
        "usage_operator": bundle.usage_operator,
        "quality_threshold": float(bundle.quality_threshold),
        "population_weighting": bundle.population_weighting,
        "source_calendar_sha256": source_calendar_sha256,
        "representations": _canonical_raw_artifact_frame_sha256(
            tables.representations,
            key_columns=(
                "held_out_subject",
                "k",
                "method",
                "donor_subject",
                "sensor_day_id",
            ),
        ),
        "baselines": _canonical_raw_artifact_frame_sha256(
            tables.baselines,
            key_columns=(
                "held_out_subject",
                "k",
                "method",
                "donor_subject",
                "primitive",
            ),
        ),
        "confidence_audit": _canonical_raw_artifact_frame_sha256(
            tables.confidence_audit,
            key_columns=("sensor_day_id", "primitive"),
        ),
        "placebo_candidates": _canonical_raw_artifact_frame_sha256(
            tables.placebo_candidates,
            key_columns=("held_out_subject", "k", "donor_subject"),
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _prepare_shift_bundle_artifacts(
    bundle: ExploratoryCandidateBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Prepare and fingerprint one outcome-free bundle for calendar shifting."""
    raw_baselines = bundle.loso_tables.baselines.copy(deep=True)
    _require_columns(
        raw_baselines,
        (
            "held_out_subject",
            "k",
            "method",
            "donor_subject",
            "primitive",
            "center",
            "scale",
            "population_scale",
        ),
        "calendar-shift baseline",
    )
    population = raw_baselines.loc[
        raw_baselines["method"].astype(str).eq("population")
        & pd.to_numeric(raw_baselines["k"], errors="coerce").eq(0)
        & raw_baselines["donor_subject"].isna()
    ].copy()
    synthetic_oracles = population.loc[
        :,
        [
            "held_out_subject",
            "primitive",
            "center",
            "scale",
            "population_scale",
        ],
    ].rename(
        columns={
            "center": "oracle_center",
            "scale": "oracle_scale",
        }
    )
    synthetic_oracles["oracle_status"] = "observed"
    representations, baselines, _, _ = _prepare_frozen_inputs(
        bundle.loso_tables,
        synthetic_oracles,
        expected_lambda_days=float(bundle.lambda_days),
        allow_formula_insufficient=True,
    )
    _validate_bundle_artifact_provenance(bundle, representations, baselines)

    baseline_columns = [
        column
        for column in (
            "_held_token",
            "_k",
            "_method",
            "_donor_token",
            "primitive",
            "center",
            "scale",
            "n_eff",
            "contributing_rows",
            "population_center",
            "population_scale",
            "population_n_eff",
            "population_contributing_rows",
            "calibration_center",
            "calibration_scale",
            "calibration_n_eff",
            "calibration_contributing_rows",
            "center_weight",
            "scale_weight",
            "scale_floor",
            "scale_fallback",
            "baseline_status",
            "lambda_days",
        )
        if column in baselines.columns
    ]
    representation_columns = [
        "_held_token",
        "_k",
        "_method",
        "_donor_token",
        "_sensor_token",
        "elapsed_day_index",
        *(
            column
            for primitive in _ALL_PRIMITIVES
            for column in (
                primitive,
                f"{primitive}__confidence",
                f"{primitive}__coverage_only_confidence",
                f"{primitive}__value_only_confidence",
                f"{primitive}__center",
                f"{primitive}__scale",
                f"{primitive}__baseline_status",
            )
        ),
    ]
    audit = bundle.loso_tables.confidence_audit.copy(deep=True)
    audit["_sensor_token"] = audit["sensor_day_id"].map(_identity_token)
    audit = audit.sort_values(["_sensor_token", "primitive"], kind="stable").reset_index(
        drop=True
    )
    audit_columns = sorted(
        column for column in audit.columns if not column.startswith("_")
    )
    hashes = {
        "baseline_sha256": _stable_frame_sha256(baselines, baseline_columns),
        "representation_sha256": _stable_frame_sha256(
            representations, representation_columns
        ),
        "confidence_audit_sha256": _stable_frame_sha256(audit, audit_columns),
        "source_calendar_sha256": str(bundle.source_calendar_sha256),
    }
    artifact_payload = {
        "bundle_id": bundle.bundle_id,
        "lambda_days": float(bundle.lambda_days),
        "usage_confidence_policy": bundle.usage_confidence_policy,
        "usage_operator": bundle.usage_operator,
        "quality_threshold": float(bundle.quality_threshold),
        "population_weighting": bundle.population_weighting,
        **hashes,
    }
    hashes["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            artifact_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return representations, baselines, hashes


def _validate_full_calendar_authoritative_task3(
    full_calendar_primitives: pd.DataFrame,
    candidate_bundle: ExploratoryCandidateBundle,
    *,
    authoritative_tables: LosoRepresentationTables | None = None,
) -> LosoRepresentationTables:
    """Rebuild one bundle policy and compare every Task 3 source table exactly."""
    if authoritative_tables is None:
        try:
            authoritative = build_loso_representations(
                full_calendar_primitives,
                lambda_days=float(candidate_bundle.lambda_days),
                usage_policy=str(candidate_bundle.usage_confidence_policy),
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                "Full-calendar authoritative Task 3 rebuild failed"
            ) from error
    else:
        authoritative = authoritative_tables
    table_keys = {
        "representations": (
            "held_out_subject",
            "k",
            "method",
            "donor_subject",
            "sensor_day_id",
        ),
        "baselines": (
            "held_out_subject",
            "k",
            "method",
            "donor_subject",
            "primitive",
        ),
        "confidence_audit": ("sensor_day_id", "primitive"),
        "placebo_candidates": ("held_out_subject", "k", "donor_subject"),
    }
    for table_name, keys in table_keys.items():
        candidate = getattr(candidate_bundle.loso_tables, table_name)
        rebuilt = getattr(authoritative, table_name)
        if set(map(_identity_token, candidate.columns)) != set(
            map(_identity_token, rebuilt.columns)
        ):
            raise ValueError(
                "Full-calendar authoritative Task 3 artifact columns disagree for "
                f"{table_name}"
            )
        candidate_hash = _canonical_raw_artifact_frame_sha256(
            candidate, key_columns=keys
        )
        rebuilt_hash = _canonical_raw_artifact_frame_sha256(
            rebuilt, key_columns=keys
        )
        if candidate_hash != rebuilt_hash:
            raise ValueError(
                "Full-calendar authoritative Task 3 artifact provenance disagrees "
                f"for {table_name}"
            )
    return authoritative


def _authoritative_candidate_bundles_from_calendar(
    trusted_full_calendar_primitives: pd.DataFrame,
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
) -> tuple[tuple[ExploratoryCandidateBundle, ...], str]:
    """Bind every G3-L/R bundle to an external raw-calendar reconstruction."""
    if not isinstance(trusted_full_calendar_primitives, pd.DataFrame):
        raise TypeError("trusted raw calendar must be a pandas DataFrame")
    if isinstance(candidate_bundles, (str, bytes)) or not isinstance(
        candidate_bundles, Sequence
    ):
        raise ValueError("candidate_bundles must be a predeclared nonempty sequence")
    bundles = list(candidate_bundles)
    if not bundles or any(
        not isinstance(bundle, ExploratoryCandidateBundle) for bundle in bundles
    ):
        raise ValueError("candidate_bundles must contain predeclared candidate bundles")

    source_sha256 = full_calendar_task3_sha256(trusted_full_calendar_primitives)
    rebuilt_by_policy: dict[tuple[float, str], LosoRepresentationTables] = {}
    authoritative_bundles: list[ExploratoryCandidateBundle] = []
    for bundle in bundles:
        declared_source = _canonical_source_calendar_sha256(
            bundle.source_calendar_sha256,
            name=f"bundle {bundle.bundle_id!r} source calendar fingerprint",
        )
        if declared_source != source_sha256:
            raise ValueError(
                "Bundle source_calendar_sha256 disagrees with the trusted raw "
                "calendar fingerprint"
            )
        policy_key = (
            float(bundle.lambda_days),
            str(bundle.usage_confidence_policy),
        )
        rebuilt = rebuilt_by_policy.get(policy_key)
        rebuilt = _validate_full_calendar_authoritative_task3(
            trusted_full_calendar_primitives,
            bundle,
            authoritative_tables=rebuilt,
        )
        rebuilt_by_policy[policy_key] = rebuilt
        authoritative_bundles.append(replace(bundle, loso_tables=rebuilt))
    return tuple(authoritative_bundles), source_sha256


def calendar_shift_reference_control(
    full_calendar_primitives: pd.DataFrame,
    candidate_bundle: ExploratoryCandidateBundle,
    *,
    pair: str = "digital_load",
    draws: int = DEFAULT_SHIFT_DRAWS,
    weekday_preserving: bool = False,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> CalendarShiftResult:
    """Shift one modality's complete quality tuple while calibration stays frozen."""
    if pair not in FROZEN_PAIR_SOURCES:
        raise ValueError(f"Unknown frozen pair: {pair}")
    if (
        isinstance(draws, bool)
        or not isinstance(draws, (int, np.integer))
        or int(draws) <= 0
    ):
        raise ValueError("draws must be an exact positive integer")
    if not isinstance(weekday_preserving, (bool, np.bool_)):
        raise ValueError("weekday_preserving must be a strict boolean")
    if isinstance(root_seed, bool) or not isinstance(root_seed, (int, np.integer)):
        raise ValueError("root_seed must be an exact integer")
    if not isinstance(candidate_bundle, ExploratoryCandidateBundle):
        raise TypeError("candidate_bundle must be predeclared")
    if float(candidate_bundle.lambda_days) != PRIMARY_LAMBDA_DAYS:
        raise ValueError("Calendar shift primary candidate requires lambda=7")
    if (
        candidate_bundle.usage_confidence_policy != "unweighted"
        or candidate_bundle.usage_operator != "frozen"
        or float(candidate_bundle.quality_threshold) != 0.0
        or candidate_bundle.population_weighting != "row_weighted"
    ):
        raise ValueError(
            "Calendar shift requires the exact lambda=7 unweighted/value-only "
            "Usage policy bundle"
        )
    expected_source_sha256 = _canonical_source_calendar_sha256(
        candidate_bundle.source_calendar_sha256,
        name="Calendar shift source calendar fingerprint in source_calendar_sha256",
    )
    actual_source_sha256 = full_calendar_task3_sha256(
        full_calendar_primitives
    )
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "Full-calendar authoritative source calendar fingerprint disagrees "
            "with the predeclared Task 3 bundle"
        )
    if not isinstance(full_calendar_primitives, pd.DataFrame):
        raise TypeError("full_calendar_primitives must be a pandas DataFrame")
    calendar = full_calendar_primitives.copy(deep=True)
    if calendar.columns.duplicated().any():
        raise ValueError("Full-calendar primitive columns must be unique")
    _, left_primitive, shifted_primitive = FROZEN_PAIR_SOURCES[pair]
    tuple_columns = [
        shifted_primitive,
        f"{shifted_primitive}__status",
        f"{shifted_primitive}__availability_coverage",
        f"{shifted_primitive}__longest_gap_minutes",
        f"{shifted_primitive}__expected_epochs",
        f"{shifted_primitive}__support_density",
    ]
    censoring = f"{shifted_primitive}__censoring"
    if censoring in calendar.columns:
        tuple_columns.append(censoring)
    _require_columns(
        calendar,
        (
            "sensor_day_id",
            "subject_id",
            "lifelog_date",
            "collection_start_date",
            "elapsed_day_index",
            left_primitive,
            *tuple_columns,
        ),
        "full-calendar shift",
    )
    if calendar["sensor_day_id"].isna().any() or calendar["sensor_day_id"].duplicated().any():
        raise ValueError("Full-calendar sensor_day_id must be nonmissing and unique")
    if calendar["subject_id"].isna().any():
        raise ValueError("Full-calendar subject_id must be nonmissing")
    calendar["elapsed_day_index"] = _exact_k(
        calendar["elapsed_day_index"], name="full-calendar elapsed_day_index"
    )
    calendar["lifelog_date"] = pd.to_datetime(calendar["lifelog_date"], errors="raise")
    calendar["collection_start_date"] = pd.to_datetime(
        calendar["collection_start_date"], errors="raise"
    )
    if calendar[["lifelog_date", "collection_start_date"]].isna().any().any():
        raise ValueError("Full-calendar local date and collection anchor must be nonmissing")
    calendar["_subject_token"] = calendar["subject_id"].map(_identity_token)
    calendar["_sensor_token"] = calendar["sensor_day_id"].map(_identity_token)
    if calendar.duplicated(["_subject_token", "elapsed_day_index"]).any():
        raise ValueError("Full-calendar participant elapsed-day rows must be unique")
    calendar = calendar.sort_values(
        ["_subject_token", "elapsed_day_index"], kind="stable"
    ).reset_index(drop=True)
    for _, group in calendar.groupby("_subject_token", sort=False):
        elapsed = group["elapsed_day_index"].to_numpy(dtype=int)
        if not np.array_equal(elapsed, np.arange(1, len(group) + 1)):
            raise ValueError("Calendar shift requires each participant's full elapsed calendar")
        dates = group["lifelog_date"].to_numpy(dtype="datetime64[D]")
        if len(dates) > 1 and not bool(np.diff(dates).astype(int).tolist() == [1] * (len(dates) - 1)):
            raise ValueError("Full elapsed calendar must contain consecutive local dates")
        expected_anchor = group["lifelog_date"].iloc[0]
        if not group["collection_start_date"].eq(expected_anchor).all():
            raise ValueError(
                "Full elapsed calendar collection anchor must equal the first local date"
            )
    participant_rows = (
        calendar.loc[:, ["subject_id", "_subject_token"]]
        .drop_duplicates("_subject_token")
        .sort_values("_subject_token", kind="stable")
    )
    if len(participant_rows) != FROZEN_PARTICIPANT_DENOMINATOR:
        raise ValueError("Calendar shift requires exactly 10 participants")

    _validate_full_calendar_authoritative_task3(
        full_calendar_primitives, candidate_bundle
    )
    frozen_representations, baselines, artifact_hashes = (
        _prepare_shift_bundle_artifacts(candidate_bundle)
    )

    _require_columns(
        frozen_representations,
        (
            "lifelog_date",
            "collection_start_date",
            left_primitive,
            shifted_primitive,
        ),
        "frozen population evaluation grid",
    )
    population_grid = frozen_representations.loc[
        frozen_representations["_method"].eq("population")
        & frozen_representations["_k"].eq(0)
        & frozen_representations["_donor_token"].eq(_NULL_DONOR)
    ].copy()
    population_grid["_local_date"] = pd.to_datetime(
        population_grid["lifelog_date"], errors="raise"
    )
    population_grid["_collection_anchor"] = pd.to_datetime(
        population_grid["collection_start_date"], errors="raise"
    )
    calendar_evaluation = calendar.loc[calendar["elapsed_day_index"].gt(21)].copy()
    expected_grid = population_grid.loc[
        :,
        [
            "_held_token",
            "_sensor_token",
            "elapsed_day_index",
            "_local_date",
            "_collection_anchor",
        ],
    ].sort_values(["_held_token", "_sensor_token"], kind="stable").reset_index(drop=True)
    actual_grid = calendar_evaluation.loc[
        :,
        [
            "_subject_token",
            "_sensor_token",
            "elapsed_day_index",
            "lifelog_date",
            "collection_start_date",
        ],
    ].rename(
        columns={
            "_subject_token": "_held_token",
            "lifelog_date": "_local_date",
            "collection_start_date": "_collection_anchor",
        }
    ).sort_values(["_held_token", "_sensor_token"], kind="stable").reset_index(drop=True)
    if len(actual_grid) != len(expected_grid) or not actual_grid.equals(expected_grid):
        raise ValueError(
            "Calendar shift requires the exact frozen population evaluation grid: "
            "participant, sensor_day_id, elapsed index, local date, and collection anchor"
        )
    frozen_values = population_grid.loc[
        :, ["_sensor_token", left_primitive, shifted_primitive]
    ]
    calendar_values = calendar_evaluation.loc[
        :, ["_sensor_token", left_primitive, shifted_primitive]
    ].merge(
        frozen_values,
        on="_sensor_token",
        how="left",
        sort=False,
        validate="one_to_one",
        suffixes=("_calendar", "_frozen"),
    )
    for primitive in (left_primitive, shifted_primitive):
        if not _numeric_equal(
            calendar_values[f"{primitive}_calendar"],
            calendar_values[f"{primitive}_frozen"],
        ).all():
            raise ValueError(
                f"Calendar shift {primitive} values disagree with the exact frozen grid"
            )

    confidence_audit = candidate_bundle.loso_tables.confidence_audit.copy(deep=True)
    confidence_audit["_sensor_token"] = confidence_audit["sensor_day_id"].map(
        _identity_token
    )
    if set(confidence_audit["_sensor_token"]) != set(calendar["_sensor_token"]):
        raise ValueError(
            "Calendar shift full sensor_day_id grid must equal the confidence artifact grid"
        )
    for primitive in (left_primitive, shifted_primitive):
        audited_values = confidence_audit.loc[
            confidence_audit["primitive"].eq(primitive),
            ["_sensor_token", "value"],
        ]
        aligned_values = calendar.loc[:, ["_sensor_token", primitive]].merge(
            audited_values,
            on="_sensor_token",
            how="left",
            sort=False,
            validate="one_to_one",
        )
        if not _numeric_equal(aligned_values[primitive], aligned_values["value"]).all():
            raise ValueError(
                f"Calendar shift {primitive} values disagree with confidence provenance"
            )

    baseline_participants = set(baselines["_held_token"])
    if baseline_participants != set(participant_rows["_subject_token"]):
        raise ValueError("Calendar and frozen baseline participants must match")
    selected_baselines = baselines.loc[
        baselines["primitive"].isin([left_primitive, shifted_primitive])
        & (
            (baselines["method"].astype(str).eq("population") & baselines["_k"].eq(0))
            | (
                baselines["method"].astype(str).eq("shrinkage")
                & baselines["_k"].eq(PRIMARY_K)
                & baselines["donor_subject"].isna()
            )
        )
    ].copy()
    if len(selected_baselines) != FROZEN_PARTICIPANT_DENOMINATOR * 4:
        raise ValueError("Calendar shift requires complete population and k14 baselines")
    if not selected_baselines["baseline_status"].astype(str).eq("observed").all():
        raise ValueError("Calendar-shift baselines must be observed")
    if not pd.to_numeric(
        selected_baselines.loc[selected_baselines["method"].eq("shrinkage"), "lambda_days"],
        errors="coerce",
    ).eq(PRIMARY_LAMBDA_DAYS).all():
        raise ValueError("Calendar-shift shrinkage baseline must use lambda=7")
    calibration_columns = [
        "_held_token",
        "method",
        "k",
        "primitive",
        "center",
        "scale",
        "population_scale",
        "lambda_days",
        "baseline_status",
    ]
    calibration = selected_baselines.sort_values(
        ["_held_token", "method", "primitive"], kind="stable"
    )
    calibration_sha256 = _stable_frame_sha256(calibration, calibration_columns)
    lookup = calibration.set_index(["_held_token", "method", "primitive"])
    observed_right = pd.to_numeric(calendar[shifted_primitive], errors="coerce").to_numpy(
        dtype=float
    )
    all_left_values = pd.to_numeric(
        calendar[left_primitive], errors="coerce"
    ).to_numpy(dtype=float)
    all_elapsed = calendar["elapsed_day_index"].to_numpy(dtype=int)
    all_subject_tokens = calendar["_subject_token"].to_numpy(dtype=object)
    sensor_column_index = calendar.columns.get_loc("sensor_day_id")
    tuple_column_indices = [calendar.columns.get_loc(column) for column in tuple_columns]
    sensor_tokens = [
        _identity_token(calendar.iat[index, sensor_column_index])
        for index in range(len(calendar))
    ]
    tuple_token_rows = [
        [
            _identity_token(calendar.iat[index, column_index])
            for column_index in tuple_column_indices
        ]
        for index in range(len(calendar))
    ]
    participant_specs: list[dict[str, Any]] = []
    for _, participant in participant_rows.iterrows():
        token = participant["_subject_token"]
        indices = np.flatnonzero(all_subject_tokens == token)
        evaluation_indices = indices[all_elapsed[indices] > 21]
        population_left = lookup.loc[(token, "population", left_primitive)]
        population_right = lookup.loc[(token, "population", shifted_primitive)]
        candidate_left = lookup.loc[(token, "shrinkage", left_primitive)]
        candidate_right = lookup.loc[(token, "shrinkage", shifted_primitive)]
        participant_specs.append(
            {
                "subject_id": participant["subject_id"],
                "token": token,
                "indices": indices,
                "evaluation_indices": evaluation_indices,
                "weekdays": calendar.loc[indices, "lifelog_date"].dt.weekday.to_numpy(),
                "population_left_center": float(population_left["center"]),
                "population_right_center": float(population_right["center"]),
                "candidate_left_center": float(candidate_left["center"]),
                "candidate_right_center": float(candidate_right["center"]),
                "left_scale": float(population_left["scale"]),
                "right_scale": float(population_right["scale"]),
            }
        )

    def score(right_values: np.ndarray) -> tuple[np.ndarray, int]:
        participant_scores = np.full(FROZEN_PARTICIPANT_DENOMINATOR, np.nan, dtype=float)
        for participant_index, spec in enumerate(participant_specs):
            evaluation_indices = spec["evaluation_indices"]
            left_values = all_left_values[evaluation_indices]
            right = right_values[evaluation_indices]
            left_scale = spec["left_scale"]
            right_scale = spec["right_scale"]
            if not np.isfinite(left_scale) or left_scale <= 0.0 or not np.isfinite(right_scale) or right_scale <= 0.0:
                continue
            finite = np.isfinite(left_values) & np.isfinite(right)
            if not bool(finite.any()):
                continue
            population_discrepancy = np.abs(
                (left_values[finite] - spec["population_left_center"]) / left_scale
                - (right[finite] - spec["population_right_center"]) / right_scale
            )
            candidate_discrepancy = np.abs(
                (left_values[finite] - spec["candidate_left_center"]) / left_scale
                - (right[finite] - spec["candidate_right_center"]) / right_scale
            )
            participant_scores[participant_index] = float(
                np.mean(population_discrepancy - candidate_discrepancy)
            )
        return participant_scores, int(np.isfinite(participant_scores).sum())

    observed_participants, observed_finite = score(observed_right)
    observed = (
        float(np.mean(observed_participants))
        if observed_finite == FROZEN_PARTICIPANT_DENOMINATOR
        else np.nan
    )
    assignment_records: list[dict[str, Any]] = []
    distribution_records: list[dict[str, Any]] = []
    mode = "weekday_preserving" if weekday_preserving else "unrestricted"
    for draw_id in range(int(draws)):
        shifted_values = observed_right.copy()
        draw_hashes: list[str] = []
        draw_degenerate = False
        for spec in participant_specs:
            token = spec["token"]
            indices = spec["indices"]
            assignment_seed = _deterministic_seed(
                int(root_seed), "dual_g3_r", "calendar_shift", mode, draw_id, token
            )
            rng = np.random.default_rng(assignment_seed)
            source_indices = indices.copy()
            shift_spec: dict[str, int] = {}
            participant_degenerate = False
            if weekday_preserving:
                weekdays = spec["weekdays"]
                weekday_positions = [
                    indices[weekdays == weekday] for weekday in range(7)
                ]
                minimum_occurrences = min(map(len, weekday_positions))
                if minimum_occurrences < 2:
                    draw_degenerate = True
                    participant_degenerate = True
                    amount = 0
                else:
                    amount = int(rng.integers(1, minimum_occurrences))
                for weekday in range(7):
                    positions = weekday_positions[weekday]
                    if len(positions) < 2:
                        continue
                    source_indices[np.isin(indices, positions)] = np.roll(positions, amount)
                shift_spec["occurrence_offset"] = amount
            else:
                if len(indices) < 2:
                    draw_degenerate = True
                    participant_degenerate = True
                    amount = 0
                else:
                    amount = int(rng.integers(1, len(indices)))
                    source_indices = np.roll(indices, amount)
                shift_spec["all"] = amount
            shifted_values[indices] = observed_right[source_indices]
            mapping_payload = [
                {
                    "destination": sensor_tokens[destination],
                    "source": sensor_tokens[source],
                    "tuple": tuple_token_rows[source],
                }
                for destination, source in zip(indices, source_indices, strict=True)
            ]
            mapping_sha256 = hashlib.sha256(
                json.dumps(
                    mapping_payload, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            draw_hashes.append(mapping_sha256)
            assignment_records.append(
                {
                    "draw_id": draw_id,
                    "subject_id": spec["subject_id"],
                    "mode": mode,
                    "shift_spec": json.dumps(shift_spec, sort_keys=True, separators=(",", ":")),
                    "assignment_seed": assignment_seed,
                    "mapping_sha256": mapping_sha256,
                    "nonzero_shift": bool(shift_spec and all(value > 0 for value in shift_spec.values())),
                    "weekday_preserved": bool(weekday_preserving),
                    "tuple_columns": json.dumps(tuple_columns, separators=(",", ":")),
                    "tuple_coupling_preserved": True,
                    "full_calendar_rows": len(indices),
                    "bundle_id": candidate_bundle.bundle_id,
                    "usage_confidence_policy": candidate_bundle.usage_confidence_policy,
                    "usage_availability_policy": "value_only",
                    "artifact_sha256": artifact_hashes["artifact_sha256"],
                    "source_calendar_sha256": artifact_hashes[
                        "source_calendar_sha256"
                    ],
                    "calibration_sha256": calibration_sha256,
                    "post_hoc_exploratory": True,
                    "status": "insufficient" if participant_degenerate else "observed",
                }
            )
        null_participants, finite = score(shifted_values)
        complete = (
            not draw_degenerate
            and finite == FROZEN_PARTICIPANT_DENOMINATOR
            and np.isfinite(observed)
        )
        statistic = float(np.mean(null_participants)) if complete else np.nan
        distribution_records.append(
            {
                "draw_id": draw_id,
                "mode": mode,
                "draw_sha256": hashlib.sha256(
                    "|".join(draw_hashes).encode("utf-8")
                ).hexdigest(),
                "observed_improvement": observed,
                "null_improvement": statistic,
                "null_as_or_better": bool(complete and statistic >= observed),
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "finite_participants": finite,
                "post_hoc_exploratory": True,
                "status": "observed" if complete else "insufficient",
            }
        )
    distribution = pd.DataFrame.from_records(distribution_records)
    degenerate_draws = int(distribution["status"].ne("observed").sum())
    as_or_better = int(distribution["null_as_or_better"].sum())
    valid_null = pd.to_numeric(
        distribution.loc[distribution["status"].eq("observed"), "null_improvement"],
        errors="coerce",
    ).to_numpy(dtype=float)
    null_95 = (
        float(np.quantile(valid_null, 0.95, method="higher"))
        if valid_null.size
        else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "pair": pair,
                "shifted_primitive": shifted_primitive,
                "mode": mode,
                "observed_improvement": observed,
                "null_95th_percentile": null_95,
                "exceeds_null_95th_percentile": bool(
                    np.isfinite(observed) and np.isfinite(null_95) and observed > null_95
                ),
                "p_value": float((1 + as_or_better + degenerate_draws) / (int(draws) + 1)),
                "tail_convention": "plus_one_null_as_or_better_degenerate_conservative",
                "null_as_or_better_draws": as_or_better,
                "degenerate_draws": degenerate_draws,
                "draws": int(draws),
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "finite_participants": observed_finite,
                "usage_availability_policy": (
                    "value_only" if shifted_primitive.startswith("usage_") else "not_applicable"
                ),
                "bundle_id": candidate_bundle.bundle_id,
                "usage_confidence_policy": candidate_bundle.usage_confidence_policy,
                "artifact_sha256": artifact_hashes["artifact_sha256"],
                "baseline_sha256": artifact_hashes["baseline_sha256"],
                "representation_sha256": artifact_hashes["representation_sha256"],
                "confidence_audit_sha256": artifact_hashes[
                    "confidence_audit_sha256"
                ],
                "source_calendar_sha256": artifact_hashes[
                    "source_calendar_sha256"
                ],
                "calibration_frozen": True,
                "calibration_sha256": calibration_sha256,
                "tuple_columns": json.dumps(tuple_columns, separators=(",", ":")),
                "root_seed": int(root_seed),
                "post_hoc_exploratory": True,
                "status": (
                    "observed"
                    if observed_finite == FROZEN_PARTICIPANT_DENOMINATOR
                    and degenerate_draws == 0
                    else "insufficient"
                ),
            }
        ]
    )
    return CalendarShiftResult(
        shift_assignments=pd.DataFrame.from_records(assignment_records),
        null_distribution=distribution,
        summary=summary,
    )


def evaluate_unrelated_pair_specificity(
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
    future_oracles: pd.DataFrame,
) -> SpecificityResult:
    """Evaluate the fixed 3x3 diagonal-versus-off-diagonal specificity matrix."""
    prepared = _canonical_candidate_bundles_self_declared(
        candidate_bundles, future_oracles
    )
    dual_prepared = [
        (bundle, _prepare_dual_coordinates(bundle, representations, baselines))
        for bundle, representations, baselines, _, _ in prepared
    ]
    matrix = pd.DataFrame.from_records(
        [
            {
                "pair": pair,
                "smartphone_primitive": smartphone,
                "validator_primitive": validator,
                "intended_diagonal": intended,
                "post_hoc_exploratory": True,
            }
            for pair, smartphone, validator, intended in SPECIFICITY_PAIR_MATRIX
        ]
    )
    intended_rows = [row for row in SPECIFICITY_PAIR_MATRIX if row[3]]
    negative_rows = [row for row in SPECIFICITY_PAIR_MATRIX if not row[3]]
    ledger = exploratory_candidate_ledger()
    ledger["available"] = False
    ledger["availability_reason"] = "candidate_bundle_not_supplied"
    ledger["specificity_family"] = False
    ledger["primary_candidate"] = False
    contrast_records: list[dict[str, Any]] = []
    for bundle, coordinates in dual_prepared:
        is_primary_bundle = bool(
            float(bundle.lambda_days) == PRIMARY_LAMBDA_DAYS
            and bundle.usage_confidence_policy == "screen_proxy"
            and bundle.usage_operator == "frozen"
            and float(bundle.quality_threshold) == 0.0
            and bundle.population_weighting == "row_weighted"
        )
        population = coordinates.loc[
            coordinates["_method"].eq("population") & coordinates["_k"].eq(0)
        ]
        for k in FROZEN_K_VALUES:
            candidate = coordinates.loc[
                coordinates["_method"].eq("shrinkage") & coordinates["_k"].eq(k)
            ]
            for clipping_state in ("clipped", "unclipped"):
                suffix = "" if clipping_state == "clipped" else "_unclipped"
                candidate_id = _candidate_id_from_ledger(
                    ledger,
                    calibration_mode="location_only",
                    k=k,
                    bundle=bundle,
                    clipping_state=clipping_state,
                )
                selected_ledger = ledger["candidate_id"].eq(candidate_id)
                ledger.loc[selected_ledger, "available"] = True
                ledger.loc[selected_ledger, "availability_reason"] = "scored_specificity"
                ledger.loc[selected_ledger, "specificity_family"] = True
                is_primary = bool(
                    is_primary_bundle
                    and k == PRIMARY_K
                    and clipping_state == "clipped"
                )
                ledger.loc[selected_ledger, "primary_candidate"] = is_primary
                value_columns = [
                    f"{primitive}__reference_z{suffix}"
                    for primitive in (
                        "screen_load_24h",
                        "phone_activity_load_24h",
                        "mobile_light_exposure_24h",
                        "usage_load_24h",
                        "step_load_24h",
                        "wearable_light_exposure_24h",
                    )
                ]
                keys = ["_held_token", "_sensor_token"]
                merged = population.loc[
                    :, [*keys, "held_out_subject", "sensor_day_id", *value_columns]
                ].merge(
                    candidate.loc[:, [*keys, *value_columns]],
                    on=keys,
                    how="outer",
                    suffixes=("_population", "_candidate"),
                    indicator=True,
                    validate="one_to_one",
                )
                if not merged["_merge"].eq("both").all():
                    raise ValueError("Specificity candidates require the identical row grid")
                for held_token, participant in merged.groupby("_held_token", sort=False):
                    for intended_pair, intended_phone, intended_validator, _ in intended_rows:
                        related = [
                            row
                            for row in negative_rows
                            if row[1] == intended_phone or row[2] == intended_validator
                        ]
                        if len(related) != 4:
                            raise RuntimeError("Every intended pair requires four fixed negatives")
                        for negative_pair, negative_phone, negative_validator, _ in related:
                            channels = tuple(
                                dict.fromkeys(
                                    (
                                        intended_phone,
                                        intended_validator,
                                        negative_phone,
                                        negative_validator,
                                    )
                                )
                            )
                            if len(channels) != 3:
                                raise RuntimeError("Specificity contrast must use exactly three channels")
                            population_values = np.column_stack(
                                [
                                    pd.to_numeric(
                                        participant[f"{channel}__reference_z{suffix}_population"],
                                        errors="coerce",
                                    ).to_numpy(dtype=float)
                                    for channel in channels
                                ]
                            )
                            candidate_values = np.column_stack(
                                [
                                    pd.to_numeric(
                                        participant[f"{channel}__reference_z{suffix}_candidate"],
                                        errors="coerce",
                                    ).to_numpy(dtype=float)
                                    for channel in channels
                                ]
                            )
                            population_finite = np.isfinite(population_values).all(axis=1)
                            candidate_finite = np.isfinite(candidate_values).all(axis=1)
                            if not np.array_equal(population_finite, candidate_finite):
                                raise ValueError(
                                    "Specificity requires identical finite rows for population and candidate"
                                )
                            common = population_finite & candidate_finite
                            denominator = int(common.sum())
                            column_index = {channel: index for index, channel in enumerate(channels)}

                            def pair_improvement(phone: str, validator: str) -> float:
                                if denominator == 0:
                                    return np.nan
                                phone_index = column_index[phone]
                                validator_index = column_index[validator]
                                population_discrepancy = np.abs(
                                    population_values[common, phone_index]
                                    - population_values[common, validator_index]
                                )
                                candidate_discrepancy = np.abs(
                                    candidate_values[common, phone_index]
                                    - candidate_values[common, validator_index]
                                )
                                return float(np.mean(population_discrepancy - candidate_discrepancy))

                            intended_improvement = pair_improvement(
                                intended_phone, intended_validator
                            )
                            negative_improvement = pair_improvement(
                                negative_phone, negative_validator
                            )
                            contrast_records.append(
                                {
                                    "candidate_id": candidate_id,
                                    "bundle_id": bundle.bundle_id,
                                    "k": int(k),
                                    "lambda_days": float(bundle.lambda_days),
                                    "usage_confidence_policy": bundle.usage_confidence_policy,
                                    "usage_operator": bundle.usage_operator,
                                    "quality_threshold": float(bundle.quality_threshold),
                                    "population_weighting": bundle.population_weighting,
                                    "clipping_state": clipping_state,
                                    "primary_candidate": is_primary,
                                    "held_out_subject": participant["held_out_subject"].iloc[0],
                                    "_held_token": held_token,
                                    "intended_pair": intended_pair,
                                    "negative_pair": negative_pair,
                                    "three_channels": json.dumps(channels, separators=(",", ":")),
                                    "exact_three_channel_intersection": True,
                                    "intended_improvement": intended_improvement,
                                    "negative_improvement": negative_improvement,
                                    "specificity_contrast": intended_improvement - negative_improvement,
                                    "common_row_denominator": denominator,
                                    "intended_row_denominator": denominator,
                                    "negative_row_denominator": denominator,
                                    "status": "observed" if denominator else "insufficient",
                                    "post_hoc_exploratory": True,
                                }
                            )
    contrasts = pd.DataFrame.from_records(contrast_records)
    metadata = [
        "candidate_id",
        "bundle_id",
        "k",
        "lambda_days",
        "usage_confidence_policy",
        "usage_operator",
        "quality_threshold",
        "population_weighting",
        "clipping_state",
        "primary_candidate",
        "held_out_subject",
        "_held_token",
    ]
    intended_summary_records: list[dict[str, Any]] = []
    for keys, group in contrasts.groupby(
        [*metadata, "intended_pair"], sort=False, dropna=False
    ):
        values = dict(zip([*metadata, "intended_pair"], keys, strict=True))
        complete = len(group) == 4 and group["status"].eq("observed").all()
        intended_improvement = (
            float(group["intended_improvement"].median()) if complete else np.nan
        )
        median_negative = (
            float(group["negative_improvement"].median()) if complete else np.nan
        )
        intended_summary_records.append(
            {
                **values,
                "intended_improvement": intended_improvement,
                "median_related_negative_improvement": median_negative,
                "specificity": intended_improvement - median_negative,
                "negative_pair_denominator": 4,
                "finite_count": int(group["status"].eq("observed").sum()),
                "status": "observed" if complete else "insufficient",
                "post_hoc_exploratory": True,
            }
        )
    intended_summary = pd.DataFrame.from_records(intended_summary_records)
    participant_records: list[dict[str, Any]] = []
    for keys, group in intended_summary.groupby(metadata, sort=False, dropna=False):
        values = dict(zip(metadata, keys, strict=True))
        finite = int(np.isfinite(group["specificity"]).sum())
        complete = len(group) == 3 and finite == 3
        participant_records.append(
            {
                **values,
                "specificity": float(group["specificity"].mean()) if complete else np.nan,
                "intended_pair_denominator": 3,
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "post_hoc_exploratory": True,
            }
        )
    participant_summary = pd.DataFrame.from_records(participant_records).sort_values(
        ["candidate_id", "_held_token"], kind="stable"
    ).reset_index(drop=True)
    primary = participant_summary.loc[participant_summary["primary_candidate"]].copy()
    if len(primary) != FROZEN_PARTICIPANT_DENOMINATOR:
        raise ValueError("Primary specificity candidate requires exactly 10 participants")
    primary = primary.sort_values("_held_token", kind="stable")
    sign_flip = exact_participant_sign_flip(
        primary.loc[:, ["held_out_subject", "specificity"]].rename(
            columns={"specificity": "improvement"}
        )
    )
    lopo_records = []
    for _, excluded in primary.iterrows():
        retained = primary.loc[primary["_held_token"].ne(excluded["_held_token"])]
        finite = int(np.isfinite(retained["specificity"]).sum())
        lopo_records.append(
            {
                "excluded_subject": excluded["held_out_subject"],
                "lopo_mean_specificity": (
                    float(retained["specificity"].mean()) if finite == 9 else np.nan
                ),
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "retained_denominator": 9,
                "finite_count": finite,
                "status": "observed" if finite == 9 else "insufficient",
                "post_hoc_exploratory": True,
                "_excluded_token": excluded["_held_token"],
            }
        )
    lopo = pd.DataFrame.from_records(lopo_records).sort_values(
        "_excluded_token", kind="stable"
    ).drop(columns="_excluded_token").reset_index(drop=True)
    max_stat = exact_max_stat_sign_flip(
        participant_summary.loc[
            :, ["candidate_id", "held_out_subject", "specificity"]
        ].rename(columns={"specificity": "improvement"})
    )
    max_summary = max_stat.candidate_summary.merge(
        ledger.loc[:, ["candidate_id", "primary_candidate"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    primary_max = max_summary.loc[max_summary["primary_candidate"]]
    finite = int(np.isfinite(primary["specificity"]).sum())
    mean_specificity = (
        float(primary["specificity"].mean())
        if finite == FROZEN_PARTICIPANT_DENOMINATOR
        else np.nan
    )
    minimum_lopo = (
        float(lopo["lopo_mean_specificity"].min())
        if lopo["status"].eq("observed").all()
        else np.nan
    )
    checks = (
        ("participant_macro_specificity_positive", mean_specificity, 0.0, ">", mean_specificity > 0.0),
        ("at_least_8_of_10_specificity_directions", int((primary["specificity"] > 0).sum()), 8, ">=", int((primary["specificity"] > 0).sum()) >= 8),
        ("all_lopo_specificity_means_positive", minimum_lopo, 0.0, ">", minimum_lopo > 0.0),
        ("exact_specificity_p_at_most_005", float(sign_flip.summary.loc[0, "p_value"]), 0.05, "<=", float(sign_flip.summary.loc[0, "p_value"]) <= 0.05),
        ("selection_adjusted_specificity_p_at_most_005", float(primary_max["adjusted_p_value"].iloc[0]), 0.05, "<=", float(primary_max["adjusted_p_value"].iloc[0]) <= 0.05),
    )
    complete = finite == FROZEN_PARTICIPANT_DENOMINATOR
    decision_rows = [
        {
            "gate": gate,
            "observed": observed,
            "threshold": threshold,
            "comparator": comparator,
            "passed": bool(complete and np.isfinite(observed) and passed),
            "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
            "finite_participants": finite,
            "status": "observed" if complete else "insufficient",
            "post_hoc_exploratory": True,
        }
        for gate, observed, threshold, comparator, passed in checks
    ]
    specificity_pass = bool(complete and all(row["passed"] for row in decision_rows))
    decision_rows.append(
        {
            "gate": "specificity_pass",
            "observed": int(sum(row["passed"] for row in decision_rows)),
            "threshold": len(checks),
            "comparator": "all",
            "passed": specificity_pass,
            "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
            "finite_participants": finite,
            "status": "observed" if complete else "insufficient",
            "post_hoc_exploratory": True,
        }
    )
    contrasts = contrasts.sort_values(
        ["candidate_id", "_held_token", "intended_pair", "negative_pair"],
        kind="stable",
    ).drop(columns="_held_token").reset_index(drop=True)
    intended_summary = intended_summary.sort_values(
        ["candidate_id", "_held_token", "intended_pair"], kind="stable"
    ).drop(columns="_held_token").reset_index(drop=True)
    public_participants = participant_summary.drop(columns="_held_token").reset_index(drop=True)
    return SpecificityResult(
        candidate_ledger=ledger,
        pair_matrix=matrix,
        contrast_table=contrasts,
        intended_pair_summary=intended_summary,
        participant_specificity=public_participants,
        lopo_summary=lopo,
        exact_sign_flip_distribution=sign_flip.distribution,
        exact_sign_flip_summary=sign_flip.summary,
        max_stat_candidate_summary=max_summary,
        max_stat_permutation_maxima=max_stat.permutation_maxima,
        max_stat_candidate_permutation_statistics=max_stat.candidate_permutation_statistics,
        specificity_gate_decision=pd.DataFrame.from_records(decision_rows),
    )


def g3_scale_self_diagnostics(
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
    future_oracles: pd.DataFrame,
    *,
    g2_diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Report G3-S as descriptive rows; no result can create a pass threshold."""
    prepared = _canonical_candidate_bundles_self_declared(
        candidate_bundles, future_oracles
    )
    primary = [
        item
        for item in prepared
        if float(item[0].lambda_days) == PRIMARY_LAMBDA_DAYS
        and item[0].usage_confidence_policy == "screen_proxy"
        and item[0].usage_operator == "frozen"
        and float(item[0].quality_threshold) == 0.0
        and item[0].population_weighting == "row_weighted"
    ]
    if len(primary) != 1:
        raise ValueError("G3-S requires exactly one primary lambda=7 bundle")
    bundle, representations, baselines, oracles, _ = primary[0]
    oracle_scale = oracles.loc[
        oracles["primitive"].isin(_CORE_PRIMITIVES),
        ["_held_token", "primitive", "oracle_scale"],
    ]
    scored = baselines.loc[
        baselines["primitive"].isin(_CORE_PRIMITIVES)
        & (
            (baselines["_method"].eq("population") & baselines["_k"].eq(0))
            | (
                baselines["_method"].isin(["shrinkage", "placebo"])
                & baselines["_k"].eq(PRIMARY_K)
            )
        )
    ].merge(
        oracle_scale,
        on=["_held_token", "primitive"],
        how="left",
        validate="many_to_one",
    )
    scored["scale_error"] = np.abs(
        np.log(
            pd.to_numeric(scored["scale"], errors="coerce")
            / pd.to_numeric(scored["oracle_scale"], errors="coerce")
        )
    )
    method_participant = (
        scored.groupby(
            ["_method", "_held_token", "_donor_token"],
            sort=False,
            dropna=False,
        )["scale_error"]
        .mean()
        .reset_index()
    )
    population_error = float(
        method_participant.loc[method_participant["_method"].eq("population"), "scale_error"].mean()
    )
    candidate_error = float(
        method_participant.loc[
            method_participant["_method"].eq("shrinkage"), "scale_error"
        ].mean()
    )
    donor_error = float(
        method_participant.loc[method_participant["_method"].eq("placebo"), "scale_error"].mean()
    )
    records: list[dict[str, Any]] = [
        {
            "diagnostic": "future_scale_error",
            "value": candidate_error,
            "comparison_value": population_error,
            "improvement": population_error - candidate_error,
            "numerator": np.nan,
            "denominator": FROZEN_PARTICIPANT_DENOMINATOR,
            "status": "observed",
            "details": "k14_shrinkage_vs_recipient_excluding_population",
        },
        {
            "diagnostic": "coherent_donor_scale_error",
            "value": candidate_error,
            "comparison_value": donor_error,
            "improvement": donor_error - candidate_error,
            "numerator": np.nan,
            "denominator": FROZEN_PARTICIPANT_DENOMINATOR * 9,
            "status": "observed",
            "details": "one_donor_supplies_all_core_primitive_scales_per_recipient",
        },
    ]
    dual = _prepare_dual_coordinates(bundle, representations, baselines)
    primary_rows = dual.loc[
        dual["_method"].eq("shrinkage") & dual["_k"].eq(PRIMARY_K)
    ]
    for coordinate in ("reference", "self"):
        columns = [
            f"{primitive}__{coordinate}_z_unclipped" for primitive in _CORE_PRIMITIVES
        ]
        values = primary_rows.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )
        finite = np.isfinite(values)
        numerator = int((finite & (np.abs(values) > 5.0)).sum())
        denominator = int(finite.sum())
        records.append(
            {
                "diagnostic": f"{coordinate}_clip_rate",
                "value": float(numerator / denominator) if denominator else np.nan,
                "comparison_value": np.nan,
                "improvement": np.nan,
                "numerator": numerator,
                "denominator": denominator,
                "status": "observed" if denominator else "insufficient",
                "details": "absolute_unclipped_coordinate_above_5",
            }
        )
    g2_names = (
        "calibration_mask_stability",
        "minimum_n_eff_status",
        "fallback_rate",
        "self_anomaly_reliability",
    )
    if g2_diagnostics is None:
        supplied_lookup: dict[str, tuple[float, str]] = {}
    else:
        if not isinstance(g2_diagnostics, pd.DataFrame):
            raise TypeError("g2_diagnostics must be a pandas DataFrame")
        _require_columns(g2_diagnostics, ("diagnostic", "value", "status"), "G2 diagnostic")
        forbidden = [
            str(column)
            for column in g2_diagnostics.columns
            if re.search(r"oracle|outcome|selector|argmax", str(column), flags=re.IGNORECASE)
        ]
        if forbidden:
            raise ValueError(
                f"G2 diagnostic contains forbidden oracle, outcome, selector, or argmax: {forbidden}"
            )
        if g2_diagnostics["diagnostic"].duplicated().any():
            raise ValueError("G2 diagnostic identities must be unique")
        supplied = g2_diagnostics.set_index("diagnostic")
        if set(supplied.index) != set(g2_names):
            raise ValueError("G2 diagnostics must contain the exact four descriptive rows")
        supplied_lookup = {
            name: (float(supplied.loc[name, "value"]), str(supplied.loc[name, "status"]))
            for name in g2_names
        }
    for name in g2_names:
        value, status = supplied_lookup.get(name, (np.nan, "unavailable"))
        if status == "observed" and not np.isfinite(value):
            raise ValueError("Observed G2 diagnostics require finite values")
        records.append(
            {
                "diagnostic": name,
                "value": value,
                "comparison_value": np.nan,
                "improvement": np.nan,
                "numerator": np.nan,
                "denominator": np.nan,
                "status": status,
                "details": "supplied_by_g2" if name in supplied_lookup else "g2_not_supplied",
            }
        )
    result = pd.DataFrame.from_records(records)
    result["formal_pass_threshold"] = False
    result["post_hoc_exploratory"] = True
    return result


def exploratory_dual_decision_table(
    *,
    g0_valid: bool,
    passing_g1_g2_domains: int,
    phone_to_wearable_domain_pass: bool,
    g3_l_pass: bool,
    g3_r_pass: bool,
    location_adjusted_p: float,
    reference_adjusted_p: float,
    unrestricted_shift_pass: bool,
    weekday_shift_pass: bool,
    specificity_pass: bool,
) -> pd.DataFrame:
    """Apply the post-hoc dual AND gate without touching the frozen G3 decision."""
    booleans = {
        "g0_valid": g0_valid,
        "phone_to_wearable_domain_pass": phone_to_wearable_domain_pass,
        "g3_l_pass": g3_l_pass,
        "g3_r_pass": g3_r_pass,
        "unrestricted_shift_pass": unrestricted_shift_pass,
        "weekday_shift_pass": weekday_shift_pass,
        "specificity_pass": specificity_pass,
    }
    if any(not isinstance(value, (bool, np.bool_)) for value in booleans.values()):
        raise ValueError("Every exploratory dual gate input must be a strict boolean")
    if isinstance(passing_g1_g2_domains, bool) or not isinstance(
        passing_g1_g2_domains, (int, np.integer)
    ):
        raise ValueError("passing_g1_g2_domains must be an exact nonnegative integer")
    domain_count = int(passing_g1_g2_domains)
    if domain_count < 0:
        raise ValueError("passing_g1_g2_domains must be an exact nonnegative integer")
    p_values = (float(location_adjusted_p), float(reference_adjusted_p))
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("Selection-adjusted p-values must be finite and between zero and one")
    p_iut = max(p_values)
    checks = (
        ("g0_valid", bool(g0_valid), True, "is", bool(g0_valid)),
        ("at_least_two_g1_g2_domains", domain_count, 2, ">=", domain_count >= 2),
        ("phone_to_wearable_domain_pass", bool(phone_to_wearable_domain_pass), True, "is", bool(phone_to_wearable_domain_pass)),
        ("g3_l_pass", bool(g3_l_pass), True, "is", bool(g3_l_pass)),
        ("g3_r_pass", bool(g3_r_pass), True, "is", bool(g3_r_pass)),
        ("dual_selection_p_iut", p_iut, 0.05, "<=", p_iut <= 0.05),
        ("unrestricted_shift_pass", bool(unrestricted_shift_pass), True, "is", bool(unrestricted_shift_pass)),
        ("weekday_shift_pass", bool(weekday_shift_pass), True, "is", bool(weekday_shift_pass)),
        ("specificity_pass", bool(specificity_pass), True, "is", bool(specificity_pass)),
    )
    rows = [
        {
            "gate": gate,
            "observed": observed,
            "threshold": threshold,
            "comparator": comparator,
            "passed": bool(passed),
            "reason": "criterion_met" if passed else "criterion_not_met",
            "post_hoc_exploratory": True,
        }
        for gate, observed, threshold, comparator, passed in checks
    ]
    refined_go = bool(all(row["passed"] for row in rows))
    if refined_go:
        label = "EXPLORATORY REFINED-GO"
    elif bool(g3_l_pass) and (
        not bool(g3_r_pass)
        or not bool(unrestricted_shift_pass)
        or not bool(weekday_shift_pass)
        or not bool(specificity_pass)
    ):
        label = "STRONG BASELINE-TRANSFER SIGNAL"
    else:
        label = "EXPLORATORY DUAL-FAIL"
    rows.append(
        {
            "gate": "exploratory_refined_go",
            "observed": int(sum(row["passed"] for row in rows)),
            "threshold": len(checks),
            "comparator": "all",
            "passed": refined_go,
            "reason": "all_refined_go_criteria_met" if refined_go else "one_or_more_criteria_failed",
            "post_hoc_exploratory": True,
        }
    )
    result = pd.DataFrame.from_records(rows)
    result["location_adjusted_p"] = p_values[0]
    result["reference_adjusted_p"] = p_values[1]
    result["p_iut"] = p_iut
    result["decision_label"] = label
    return result


def evaluate_exploratory_dual_cold_start(
    candidate_bundles: Sequence[ExploratoryCandidateBundle],
    future_oracles: pd.DataFrame,
    frozen_result: FrozenColdStartResult,
    full_calendar_primitives: pd.DataFrame,
    *,
    validated_domains: Sequence[str],
    upstream_validation: pd.DataFrame,
    g2_diagnostics: pd.DataFrame | None = None,
    derangement_draws: int = DEFAULT_DERANGEMENT_DRAWS,
    shift_draws: int = DEFAULT_SHIFT_DRAWS,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> ExploratoryDualColdStartResult:
    """Run the complete dual analysis while retaining frozen B1 byte-for-byte."""
    if not isinstance(frozen_result, FrozenColdStartResult):
        raise TypeError("frozen_result must be the immutable B1 result")
    domains = _canonical_validated_domains(validated_domains)
    if not isinstance(upstream_validation, pd.DataFrame):
        raise TypeError("upstream_validation must be an explicit pandas DataFrame")
    _require_columns(
        upstream_validation,
        ("domain", "g0_valid", "g1_pass", "g2_pass", "phone_to_wearable"),
        "upstream validation",
    )
    upstream = upstream_validation.copy(deep=True)
    forbidden = [
        str(column)
        for column in upstream.columns
        if re.search(r"oracle|outcome|selector|argmax", str(column), flags=re.IGNORECASE)
    ]
    if forbidden:
        raise ValueError(f"Upstream validation contains forbidden selector data: {forbidden}")
    if upstream["domain"].isna().any() or upstream["domain"].duplicated().any():
        raise ValueError("Upstream validation domains must be nonmissing and unique")
    if set(upstream["domain"].astype(str)) != set(domains):
        raise ValueError("Upstream validation must cover the exact validated domains")
    for column in ("g0_valid", "g1_pass", "g2_pass", "phone_to_wearable"):
        if not upstream[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"Upstream {column} must contain strict booleans")
    bundles = tuple(candidate_bundles)
    analysis_bundles = tuple(
        bundle
        for bundle in bundles
        if isinstance(bundle.dual_tables, DualCalibrationTables)
    )
    authoritative_analysis_bundles, trusted_source_calendar_sha256 = (
        _authoritative_candidate_bundles_from_calendar(
            full_calendar_primitives, analysis_bundles
        )
    )
    location = _evaluate_exploratory_location_transfer_trusted(
        authoritative_analysis_bundles,
        future_oracles,
        validated_domains=domains,
        trusted_source_calendar_sha256=trusted_source_calendar_sha256,
        derangement_draws=derangement_draws,
        root_seed=root_seed,
    )
    reference = _evaluate_exploratory_reference_alignment_trusted(
        authoritative_analysis_bundles,
        future_oracles,
        trusted_source_calendar_sha256=trusted_source_calendar_sha256,
        derangement_draws=derangement_draws,
        root_seed=root_seed,
    )
    specificity = evaluate_unrelated_pair_specificity(
        authoritative_analysis_bundles, future_oracles
    )
    shift_bundles = [
        bundle
        for bundle in bundles
        if float(bundle.lambda_days) == PRIMARY_LAMBDA_DAYS
        and bundle.usage_confidence_policy == "unweighted"
        and bundle.usage_operator == "frozen"
        and float(bundle.quality_threshold) == 0.0
        and bundle.population_weighting == "row_weighted"
    ]
    if len(shift_bundles) != 1:
        raise ValueError(
            "Complete dual evaluation requires one lambda=7 unweighted shift bundle"
        )
    unrestricted = calendar_shift_reference_control(
        full_calendar_primitives,
        shift_bundles[0],
        pair="digital_load",
        draws=shift_draws,
        weekday_preserving=False,
        root_seed=root_seed,
    )
    weekday = calendar_shift_reference_control(
        full_calendar_primitives,
        shift_bundles[0],
        pair="digital_load",
        draws=shift_draws,
        weekday_preserving=True,
        root_seed=root_seed,
    )
    shift_summaries = pd.concat(
        [unrestricted.summary, weekday.summary], ignore_index=True
    )
    scale_diagnostics = g3_scale_self_diagnostics(
        authoritative_analysis_bundles,
        future_oracles,
        g2_diagnostics=g2_diagnostics,
    )
    location_pass = bool(
        location.location_gate_decision.loc[
            location.location_gate_decision["gate"].eq("exploratory_g3_l_pass"),
            "passed",
        ].item()
    )
    reference_pass = bool(
        reference.reference_gate_decision.loc[
            reference.reference_gate_decision["gate"].eq("exploratory_g3_r_pass"),
            "passed",
        ].item()
    )
    specificity_pass = bool(
        specificity.specificity_gate_decision.loc[
            specificity.specificity_gate_decision["gate"].eq("specificity_pass"),
            "passed",
        ].item()
    )

    def shift_pass(result: CalendarShiftResult) -> bool:
        row = result.summary.iloc[0]
        return bool(
            row["status"] == "observed"
            and row["p_value"] <= 0.05
            and row["exceeds_null_95th_percentile"]
        )

    passing_domains = upstream["g1_pass"].astype(bool) & upstream["g2_pass"].astype(bool)
    phone_pass = bool(
        (passing_domains & upstream["phone_to_wearable"].astype(bool)).any()
    )
    location_p = float(
        location.max_stat_candidate_summary.loc[
            location.max_stat_candidate_summary["primary_candidate"], "adjusted_p_value"
        ].item()
    )
    reference_p = float(
        reference.max_stat_candidate_summary.loc[
            reference.max_stat_candidate_summary["primary_candidate"], "adjusted_p_value"
        ].item()
    )
    decision = exploratory_dual_decision_table(
        g0_valid=bool(upstream["g0_valid"].all()),
        passing_g1_g2_domains=int(passing_domains.sum()),
        phone_to_wearable_domain_pass=phone_pass,
        g3_l_pass=location_pass,
        g3_r_pass=reference_pass,
        location_adjusted_p=location_p,
        reference_adjusted_p=reference_p,
        unrestricted_shift_pass=shift_pass(unrestricted),
        weekday_shift_pass=shift_pass(weekday),
        specificity_pass=specificity_pass,
    )
    frozen_gate = frozen_result.frozen_gate_decision.loc[
        frozen_result.frozen_gate_decision["gate"].eq("frozen_g3_pass"), "passed"
    ]
    if len(frozen_gate) != 1:
        raise ValueError("Frozen B1 result must contain exactly one frozen_g3_pass row")
    decision["frozen_g3_pass_unchanged"] = bool(frozen_gate.item())
    decision["frozen_result_can_be_retroactively_rescued"] = False
    decision["validated_domains"] = json.dumps(domains)
    return ExploratoryDualColdStartResult(
        frozen_result=frozen_result,
        location=location,
        reference=reference,
        unrestricted_shift=unrestricted,
        weekday_shift=weekday,
        specificity=specificity,
        scale_self_diagnostics=scale_diagnostics,
        shift_summaries=shift_summaries,
        joint_decision=decision,
    )


def _build_transfer_tables(
    baselines: pd.DataFrame,
    oracles: pd.DataFrame,
    validated_domains: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oracle_columns = oracles.loc[
        oracles["primitive"].isin(_CORE_PRIMITIVES),
        [
            "_held_token",
            "primitive",
            "oracle_center",
            "oracle_scale",
            "oracle_status",
            "population_scale",
        ],
    ].rename(columns={"population_scale": "oracle_population_scale"})
    primitive = baselines.loc[
        baselines["primitive"].isin(_CORE_PRIMITIVES)
    ].merge(
        oracle_columns,
        on=["_held_token", "primitive"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    center = pd.to_numeric(primitive["center"], errors="coerce").to_numpy(dtype=float)
    scale = pd.to_numeric(primitive["scale"], errors="coerce").to_numpy(dtype=float)
    oracle_center = pd.to_numeric(
        primitive["oracle_center"], errors="coerce"
    ).to_numpy(dtype=float)
    oracle_scale = pd.to_numeric(
        primitive["oracle_scale"], errors="coerce"
    ).to_numpy(dtype=float)
    population_scale = pd.to_numeric(
        primitive["_anchor_scale"], errors="coerce"
    ).to_numpy(dtype=float)
    location_error = np.abs(center - oracle_center) / population_scale
    scale_error = np.abs(np.log(scale / oracle_scale))
    transfer_error = 0.5 * location_error + 0.5 * scale_error
    primitive["domain"] = primitive["primitive"].map(_PRIMITIVE_DOMAIN)
    primitive["validated_domain"] = primitive["domain"].isin(validated_domains)
    primitive["domain_status"] = np.where(
        primitive["validated_domain"], "validated", "excluded_nonvalidated"
    )
    primitive["location_error"] = location_error
    primitive["scale_error"] = scale_error
    primitive["transfer_error"] = transfer_error
    primitive["primitive_denominator"] = 1
    primitive["finite_count"] = np.isfinite(transfer_error).astype(int)
    primitive["status"] = np.where(
        np.isfinite(transfer_error), "observed", "insufficient"
    )
    primitive["population_scale"] = population_scale
    primitive_columns = [
        "held_out_subject",
        "k",
        "method",
        "donor_subject",
        "primitive",
        "domain",
        "validated_domain",
        "domain_status",
        "center",
        "scale",
        "oracle_center",
        "oracle_scale",
        "population_scale",
        "location_error",
        "scale_error",
        "transfer_error",
        "primitive_denominator",
        "finite_count",
        "status",
        "lambda_days",
        "_held_token",
        "_k",
        "_method",
        "_donor_token",
        "_method_order",
    ]
    primitive = primitive.loc[:, primitive_columns]

    domain_records: list[dict[str, Any]] = []
    group_columns = ["_held_token", "_k", "_method", "_donor_token", "domain"]
    for keys, group in primitive.groupby(group_columns, sort=False, dropna=False):
        denominator = len(FROZEN_DOMAIN_PRIMITIVES[str(keys[4])])
        finite = int(np.isfinite(group["transfer_error"]).sum())
        first = group.iloc[0]
        complete = finite == denominator
        domain_records.append(
            {
                "held_out_subject": first["held_out_subject"],
                "k": int(first["k"]),
                "method": first["method"],
                "donor_subject": first["donor_subject"],
                "domain": first["domain"],
                "validated_domain": bool(first["validated_domain"]),
                "domain_status": first["domain_status"],
                "location_error": float(group["location_error"].mean()) if complete else np.nan,
                "scale_error": float(group["scale_error"].mean()) if complete else np.nan,
                "transfer_error": float(group["transfer_error"].mean()) if complete else np.nan,
                "primitive_denominator": denominator,
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "_held_token": keys[0],
                "_k": int(keys[1]),
                "_method": keys[2],
                "_donor_token": keys[3],
                "_method_order": int(first["_method_order"]),
            }
        )
    domains = pd.DataFrame.from_records(domain_records)

    population_lookup = domains.loc[
        domains["_method"].eq("population")
        & domains["_k"].eq(0)
        & domains["_donor_token"].eq(_NULL_DONOR)
        & domains["domain"].isin(validated_domains),
        ["_held_token", "domain", "transfer_error"],
    ].rename(columns={"transfer_error": "_population_domain_error"})
    participants_records: list[dict[str, Any]] = []
    for keys, group in domains.groupby(
        ["_held_token", "_k", "_method", "_donor_token"],
        sort=False,
        dropna=False,
    ):
        selected = group.loc[group["domain"].isin(validated_domains)].merge(
            population_lookup,
            on=["_held_token", "domain"],
            how="left",
            sort=False,
            validate="one_to_one",
        )
        denominator = len(validated_domains)
        finite = int(
            (
                np.isfinite(selected["transfer_error"])
                & np.isfinite(selected["_population_domain_error"])
            ).sum()
        )
        complete = len(selected) == denominator and finite == denominator
        first = group.iloc[0]
        population_error = (
            float(selected["_population_domain_error"].mean()) if complete else np.nan
        )
        candidate_error = float(selected["transfer_error"].mean()) if complete else np.nan
        improvement = population_error - candidate_error if complete else np.nan
        relative = (
            improvement / population_error
            if complete and population_error > 0
            else np.nan
        )
        participants_records.append(
            {
                "held_out_subject": first["held_out_subject"],
                "k": int(first["k"]),
                "method": first["method"],
                "donor_subject": first["donor_subject"],
                "validated_domains": json.dumps(validated_domains),
                "population_transfer_error": population_error,
                "candidate_transfer_error": candidate_error,
                "improvement": improvement,
                "relative_reduction": relative,
                "domain_denominator": denominator,
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "_held_token": keys[0],
                "_k": int(keys[1]),
                "_method": keys[2],
                "_donor_token": keys[3],
                "_method_order": int(first["_method_order"]),
            }
        )
    participants = pd.DataFrame.from_records(participants_records)

    summary_records: list[dict[str, Any]] = []
    for keys, group in participants.groupby(
        ["_method_order", "_k", "_method", "_donor_token"],
        sort=False,
        dropna=False,
    ):
        finite = int(np.isfinite(group["candidate_transfer_error"]).sum())
        complete = finite == FROZEN_PARTICIPANT_DENOMINATOR
        first = group.iloc[0]
        population_error = (
            float(group["population_transfer_error"].mean()) if complete else np.nan
        )
        candidate_error = (
            float(group["candidate_transfer_error"].mean()) if complete else np.nan
        )
        improvement = population_error - candidate_error if complete else np.nan
        summary_records.append(
            {
                "method": first["method"],
                "k": int(first["k"]),
                "donor_subject": first["donor_subject"],
                "population_transfer_error": population_error,
                "candidate_transfer_error": candidate_error,
                "mean_improvement": improvement,
                "relative_reduction": (
                    improvement / population_error
                    if complete and population_error > 0
                    else np.nan
                ),
                "improved_participants": int(
                    (pd.to_numeric(group["improvement"], errors="coerce") > 0).sum()
                ),
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "validated_domains": json.dumps(validated_domains),
                "_method_order": int(keys[0]),
                "_k": int(keys[1]),
                "_donor_token": keys[3],
            }
        )
    summaries = pd.DataFrame.from_records(summary_records)

    primitive = primitive.sort_values(
        ["_method_order", "_k", "_held_token", "_donor_token", "domain", "primitive"],
        kind="stable",
    ).drop(columns=["_method_order", "_held_token", "_k", "_method", "_donor_token"]).reset_index(drop=True)
    domains = domains.sort_values(
        ["_method_order", "_k", "_held_token", "_donor_token", "domain"],
        kind="stable",
    ).drop(columns=["_method_order", "_held_token", "_k", "_method", "_donor_token"]).reset_index(drop=True)
    participants = participants.sort_values(
        ["_method_order", "_k", "_held_token", "_donor_token"], kind="stable"
    ).reset_index(drop=True)
    summaries = summaries.sort_values(
        ["_method_order", "_k", "_donor_token"], kind="stable"
    ).drop(columns=["_method_order", "_k", "_donor_token"]).reset_index(drop=True)
    return primitive, domains, participants, summaries


def _build_cross_modal_tables(
    representations: pd.DataFrame,
    validated_domains: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    population = representations.loc[
        representations["_method"].eq("population")
        & representations["_k"].eq(0)
        & representations["_donor_token"].eq(_NULL_DONOR)
    ]
    candidate = representations.loc[
        representations["_method"].eq("shrinkage")
        & representations["_k"].eq(PRIMARY_K)
        & representations["_donor_token"].eq(_NULL_DONOR)
    ]
    keys = ["_held_token", "_sensor_token"]
    population_key_set = set(map(tuple, population[keys].to_numpy()))
    candidate_key_set = set(map(tuple, candidate[keys].to_numpy()))
    if population_key_set != candidate_key_set:
        raise ValueError("Population and primary candidate require an identical common row grid")

    day_records: list[dict[str, Any]] = []
    for pair, (domain, left, right) in FROZEN_PAIR_SOURCES.items():
        columns = [
            "_held_token",
            "_sensor_token",
            "sensor_day_id",
            "held_out_subject",
            f"{left}__z",
            f"{right}__z",
        ]
        pop = population.loc[:, columns].rename(
            columns={
                f"{left}__z": "population_left_z",
                f"{right}__z": "population_right_z",
            }
        )
        cand = candidate.loc[
            :, ["_held_token", "_sensor_token", f"{left}__z", f"{right}__z"]
        ].rename(
            columns={
                f"{left}__z": "candidate_left_z",
                f"{right}__z": "candidate_right_z",
            }
        )
        paired = pop.merge(
            cand,
            on=["_held_token", "_sensor_token"],
            how="outer",
            sort=False,
            validate="one_to_one",
            indicator=True,
        )
        if not paired["_merge"].eq("both").all():
            raise ValueError("Population and candidate require an identical common row grid")
        numeric_columns = (
            "population_left_z",
            "population_right_z",
            "candidate_left_z",
            "candidate_right_z",
        )
        for column in numeric_columns:
            paired[column] = pd.to_numeric(paired[column], errors="coerce")
        population_finite = np.isfinite(paired["population_left_z"]) & np.isfinite(
            paired["population_right_z"]
        )
        candidate_finite = np.isfinite(paired["candidate_left_z"]) & np.isfinite(
            paired["candidate_right_z"]
        )
        if not np.array_equal(population_finite.to_numpy(), candidate_finite.to_numpy()):
            raise ValueError(
                f"{pair} population and candidate must use the identical finite row intersection"
            )
        common = population_finite & candidate_finite
        for row_index, row in paired.iterrows():
            finite = bool(common.loc[row_index])
            if finite:
                population_discrepancy = abs(
                    float(row["population_left_z"]) - float(row["population_right_z"])
                )
                candidate_discrepancy = abs(
                    float(row["candidate_left_z"]) - float(row["candidate_right_z"])
                )
                population_sign = float(
                    bool(
                        standardized_sign_agreement(
                            row["population_left_z"], row["population_right_z"]
                        )
                    )
                )
                candidate_sign = float(
                    bool(
                        standardized_sign_agreement(
                            row["candidate_left_z"], row["candidate_right_z"]
                        )
                    )
                )
            else:
                population_discrepancy = candidate_discrepancy = float("nan")
                population_sign = candidate_sign = float("nan")
            day_records.append(
                {
                    "sensor_day_id": row["sensor_day_id"],
                    "held_out_subject": row["held_out_subject"],
                    "pair": pair,
                    "domain": domain,
                    "validated_domain": domain in validated_domains,
                    "domain_status": (
                        "validated" if domain in validated_domains else "excluded_nonvalidated"
                    ),
                    "left_primitive": left,
                    "right_primitive": right,
                    "population_method": "population",
                    "population_k": 0,
                    "population_donor_subject": None,
                    "candidate_method": "shrinkage",
                    "candidate_k": PRIMARY_K,
                    "candidate_donor_subject": None,
                    "population_left_z": row["population_left_z"],
                    "population_right_z": row["population_right_z"],
                    "candidate_left_z": row["candidate_left_z"],
                    "candidate_right_z": row["candidate_right_z"],
                    "population_discrepancy": population_discrepancy,
                    "candidate_discrepancy": candidate_discrepancy,
                    "discrepancy_improvement": (
                        population_discrepancy - candidate_discrepancy
                        if finite
                        else np.nan
                    ),
                    "population_sign_agreement": population_sign,
                    "candidate_sign_agreement": candidate_sign,
                    "sign_agreement_delta": (
                        candidate_sign - population_sign if finite else np.nan
                    ),
                    "row_denominator": 1,
                    "finite_count": int(finite),
                    "status": "observed" if finite else "insufficient",
                    "_held_token": row["_held_token"],
                    "_sensor_token": row["_sensor_token"],
                }
            )
    pair_days = pd.DataFrame.from_records(day_records)

    pair_records: list[dict[str, Any]] = []
    for keys_value, group in pair_days.groupby(
        ["_held_token", "pair", "domain"], sort=False, dropna=False
    ):
        denominator = len(group)
        finite = int(group["finite_count"].sum())
        first = group.iloc[0]
        observed = finite > 0
        finite_rows = group.loc[group["finite_count"].eq(1)]
        pair_records.append(
            {
                "held_out_subject": first["held_out_subject"],
                "pair": first["pair"],
                "domain": first["domain"],
                "population_method": "population",
                "population_k": 0,
                "population_donor_subject": None,
                "candidate_method": "shrinkage",
                "candidate_k": PRIMARY_K,
                "candidate_donor_subject": None,
                "validated_domain": bool(first["validated_domain"]),
                "domain_status": first["domain_status"],
                "population_discrepancy": (
                    float(finite_rows["population_discrepancy"].mean()) if observed else np.nan
                ),
                "candidate_discrepancy": (
                    float(finite_rows["candidate_discrepancy"].mean()) if observed else np.nan
                ),
                "discrepancy_improvement": (
                    float(finite_rows["discrepancy_improvement"].mean()) if observed else np.nan
                ),
                "population_sign_agreement": (
                    float(finite_rows["population_sign_agreement"].mean()) if observed else np.nan
                ),
                "candidate_sign_agreement": (
                    float(finite_rows["candidate_sign_agreement"].mean()) if observed else np.nan
                ),
                "sign_agreement_delta": (
                    float(finite_rows["sign_agreement_delta"].mean()) if observed else np.nan
                ),
                "row_denominator": denominator,
                "finite_count": finite,
                "status": "observed" if observed else "insufficient",
                "_held_token": keys_value[0],
            }
        )
    pair_summary = pd.DataFrame.from_records(pair_records)

    participant_records: list[dict[str, Any]] = []
    for held_token, held_pairs in pair_summary.groupby("_held_token", sort=False):
        domain_records: list[dict[str, float | int | str]] = []
        for domain in validated_domains:
            expected_pairs = [
                pair for pair, values in FROZEN_PAIR_SOURCES.items() if values[0] == domain
            ]
            selected = held_pairs.loc[held_pairs["pair"].isin(expected_pairs)]
            finite = int(np.isfinite(selected["candidate_discrepancy"]).sum())
            if len(selected) == len(expected_pairs) and finite == len(expected_pairs):
                domain_records.append(
                    {
                        "domain": domain,
                        "population_discrepancy": float(selected["population_discrepancy"].mean()),
                        "candidate_discrepancy": float(selected["candidate_discrepancy"].mean()),
                        "population_sign_agreement": float(selected["population_sign_agreement"].mean()),
                        "candidate_sign_agreement": float(selected["candidate_sign_agreement"].mean()),
                        "pair_denominator": len(expected_pairs),
                        "finite_count": finite,
                    }
                )
        domain_frame = pd.DataFrame.from_records(domain_records)
        complete = len(domain_frame) == len(validated_domains)
        first = held_pairs.iloc[0]
        population_discrepancy = (
            float(domain_frame["population_discrepancy"].mean()) if complete else np.nan
        )
        candidate_discrepancy = (
            float(domain_frame["candidate_discrepancy"].mean()) if complete else np.nan
        )
        population_sign = (
            float(domain_frame["population_sign_agreement"].mean()) if complete else np.nan
        )
        candidate_sign = (
            float(domain_frame["candidate_sign_agreement"].mean()) if complete else np.nan
        )
        participant_records.append(
            {
                "held_out_subject": first["held_out_subject"],
                "validated_domains": json.dumps(validated_domains),
                "population_method": "population",
                "population_k": 0,
                "population_donor_subject": None,
                "candidate_method": "shrinkage",
                "candidate_k": PRIMARY_K,
                "candidate_donor_subject": None,
                "population_discrepancy": population_discrepancy,
                "candidate_discrepancy": candidate_discrepancy,
                "discrepancy_improvement": (
                    population_discrepancy - candidate_discrepancy if complete else np.nan
                ),
                "population_sign_agreement": population_sign,
                "candidate_sign_agreement": candidate_sign,
                "sign_agreement_delta": candidate_sign - population_sign if complete else np.nan,
                "domain_denominator": len(validated_domains),
                "finite_count": len(domain_frame),
                "pair_denominator": sum(
                    len([1 for values in FROZEN_PAIR_SOURCES.values() if values[0] == domain])
                    for domain in validated_domains
                ),
                "finite_pair_count": int(domain_frame["finite_count"].sum()) if complete else 0,
                "status": "observed" if complete else "insufficient",
                "_held_token": held_token,
            }
        )
    participant_summary = pd.DataFrame.from_records(participant_records)
    pair_days = pair_days.sort_values(
        ["_held_token", "_sensor_token", "pair"], kind="stable"
    ).drop(columns=["_held_token", "_sensor_token"]).reset_index(drop=True)
    pair_summary = pair_summary.sort_values(
        ["_held_token", "domain", "pair"], kind="stable"
    ).drop(columns="_held_token").reset_index(drop=True)
    participant_summary = participant_summary.sort_values(
        "_held_token", kind="stable"
    ).reset_index(drop=True)
    return pair_days, pair_summary, participant_summary


def _build_lopo_summary(primary_deltas: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    ordered = primary_deltas.sort_values("_held_token", kind="stable")
    for _, excluded in ordered.iterrows():
        retained = ordered.loc[ordered["_held_token"].ne(excluded["_held_token"])]
        finite = int(np.isfinite(retained["improvement"]).sum())
        complete = len(retained) == FROZEN_PARTICIPANT_DENOMINATOR - 1 and finite == len(
            retained
        )
        records.append(
            {
                "excluded_subject": excluded["held_out_subject"],
                "method": "shrinkage",
                "k": PRIMARY_K,
                "donor_subject": None,
                "lopo_mean_improvement": (
                    float(retained["improvement"].mean()) if complete else np.nan
                ),
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "retained_denominator": FROZEN_PARTICIPANT_DENOMINATOR - 1,
                "finite_count": finite,
                "status": "observed" if complete else "insufficient",
                "_excluded_token": excluded["_held_token"],
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        "_excluded_token", kind="stable"
    ).drop(columns="_excluded_token").reset_index(drop=True)


def _score_coherent_derangements(
    participants: Sequence[Any],
    participant_transfer: pd.DataFrame,
    primary_deltas: pd.DataFrame,
    *,
    draws: int,
    root_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = generate_coherent_derangements(
        participants, draws=draws, root_seed=root_seed
    )
    placebo = participant_transfer.loc[
        participant_transfer["_method"].eq("placebo")
        & participant_transfer["_k"].eq(PRIMARY_K)
    ].copy()
    placebo["_recipient_token"] = placebo["held_out_subject"].map(_identity_token)
    placebo["_placebo_donor_token"] = placebo["donor_subject"].map(_identity_token)
    if placebo.duplicated(["_recipient_token", "_placebo_donor_token"]).any():
        raise ValueError("Primary placebo transfer keys must be unique")
    lookup = placebo.loc[
        :,
        [
            "_recipient_token",
            "_placebo_donor_token",
            "improvement",
            "candidate_transfer_error",
            "status",
        ],
    ].rename(
        columns={
            "improvement": "placebo_improvement",
            "candidate_transfer_error": "placebo_transfer_error",
            "status": "placebo_status",
        }
    )
    assignment_rows = assignments.copy(deep=True)
    assignment_rows["_recipient_token"] = assignment_rows["recipient_subject"].map(
        _identity_token
    )
    assignment_rows["_placebo_donor_token"] = assignment_rows["donor_subject"].map(
        _identity_token
    )
    scored = assignment_rows.merge(
        lookup,
        on=["_recipient_token", "_placebo_donor_token"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if scored["placebo_status"].isna().any():
        raise ValueError("Every coherent derangement requires a scored placebo donor")
    candidate_finite = int(np.isfinite(primary_deltas["improvement"]).sum())
    candidate_complete = candidate_finite == FROZEN_PARTICIPANT_DENOMINATOR
    candidate_improvement = (
        float(primary_deltas["improvement"].mean()) if candidate_complete else np.nan
    )
    records: list[dict[str, Any]] = []
    for draw_id, group in scored.groupby("draw_id", sort=True):
        finite = int(np.isfinite(group["placebo_improvement"]).sum())
        complete = bool(
            len(group) == FROZEN_PARTICIPANT_DENOMINATOR
            and finite == len(group)
            and group["placebo_status"].astype(str).eq("observed").all()
        )
        ordered = group.sort_values("_recipient_token", kind="stable")
        full_assignment = json.dumps(
            {
                _identity_token(row.recipient_subject): _identity_token(row.donor_subject)
                for row in ordered.itertuples(index=False)
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        placebo_improvement = (
            float(group["placebo_improvement"].mean()) if complete else np.nan
        )
        records.append(
            {
                "draw_id": int(draw_id),
                "assignment_hash": group["assignment_hash"].iloc[0],
                "assignment_seed": int(group["assignment_seed"].iloc[0]),
                "full_assignment": full_assignment,
                "candidate_improvement": candidate_improvement,
                "placebo_improvement": placebo_improvement,
                "candidate_minus_placebo": (
                    candidate_improvement - placebo_improvement
                    if complete and candidate_complete
                    else np.nan
                ),
                "placebo_as_or_better": bool(
                    complete
                    and candidate_complete
                    and placebo_improvement >= candidate_improvement
                ),
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "finite_count": finite,
                "status": (
                    "observed" if complete and candidate_complete else "insufficient"
                ),
                "root_seed": int(root_seed),
            }
        )
    distribution = pd.DataFrame.from_records(records).sort_values(
        "draw_id", kind="stable"
    ).reset_index(drop=True)
    degenerate = distribution["status"].ne("observed")
    exceedances = int((distribution["placebo_as_or_better"] | degenerate).sum())
    p_value = float((1 + exceedances) / (len(distribution) + 1))
    summary = pd.DataFrame(
        [
            {
                "candidate_improvement": candidate_improvement,
                "p_value": p_value,
                "tail_convention": "plus_one_placebo_as_or_better",
                "alternative": "candidate_larger_improvement",
                "draws": len(distribution),
                "participant_denominator": FROZEN_PARTICIPANT_DENOMINATOR,
                "finite_participants": candidate_finite,
                "degenerate_draws": int(degenerate.sum()),
                "root_seed": int(root_seed),
                "status": (
                    "observed"
                    if candidate_complete and not bool(degenerate.any())
                    else "insufficient"
                ),
            }
        ]
    )
    clean_assignments = assignments.sort_values(
        ["draw_id", "recipient_subject"],
        key=lambda column: column.map(_identity_token),
        kind="stable",
    ).reset_index(drop=True)
    return clean_assignments, distribution, summary


def _deterministic_seed(root_seed: int, *parts: Any) -> int:
    payload = json.dumps(
        [int(root_seed), *[_identity_token(part) for part in parts]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _ordered_unique_participants(participants: Sequence[Any]) -> list[Any]:
    values = list(participants)
    if len(values) < 2:
        raise ValueError("At least two participant identities are required")
    series = pd.Series(values, dtype=object)
    if series.isna().any() or series.duplicated().any():
        raise ValueError("Participants must contain unique participant identities")
    tokens = [_identity_token(value) for value in values]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Participants must contain unique participant identities")
    return [value for _, value in sorted(zip(tokens, values, strict=True))]


def exact_participant_sign_flip(
    participant_deltas: pd.DataFrame,
    *,
    participant_column: str = "held_out_subject",
    delta_column: str = "improvement",
    expected_participants: int = FROZEN_PARTICIPANT_DENOMINATOR,
) -> SignFlipResult:
    """Enumerate every sign pattern without dropping missing participants."""
    if isinstance(expected_participants, bool) or not isinstance(
        expected_participants, (int, np.integer)
    ):
        raise ValueError("expected_participants must be an exact positive integer")
    expected = int(expected_participants)
    if expected <= 0:
        raise ValueError("expected_participants must be an exact positive integer")
    if not isinstance(participant_deltas, pd.DataFrame):
        raise TypeError("participant_deltas must be a pandas DataFrame")
    if participant_deltas.columns.duplicated().any():
        raise ValueError("participant_deltas columns must be unique")
    missing = sorted(
        {participant_column, delta_column}.difference(participant_deltas.columns)
    )
    if missing:
        raise ValueError(f"Missing participant delta columns: {missing}")
    frame = participant_deltas.loc[:, [participant_column, delta_column]].copy(deep=True)
    if frame[participant_column].isna().any() or frame[participant_column].duplicated().any():
        raise ValueError("Participant identities must be nonmissing and unique")
    if len(frame) != expected:
        raise ValueError(f"Sign flip requires exactly {expected} participants")
    frame["_token"] = frame[participant_column].map(_identity_token)
    if frame["_token"].duplicated().any():
        raise ValueError("Participant identities must be nonmissing and unique")
    frame = frame.sort_values("_token", kind="stable").reset_index(drop=True)
    deltas = pd.to_numeric(frame[delta_column], errors="coerce").to_numpy(dtype=float)
    participant_order = json.dumps(
        frame["_token"].tolist(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    finite_count = int(np.isfinite(deltas).sum())
    complete = finite_count == expected
    pattern_count = 1 << expected
    records: list[dict[str, Any]] = []
    statistics = np.full(pattern_count, np.nan, dtype=float)
    for pattern_id in range(pattern_count):
        signs = np.array(
            [1.0 if pattern_id & (1 << index) else -1.0 for index in range(expected)],
            dtype=float,
        )
        if complete:
            statistics[pattern_id] = float(np.mean(signs * deltas))
        records.append(
            {
                "pattern_id": pattern_id,
                "sign_pattern": "".join("+" if sign > 0 else "-" for sign in signs),
                "participant_order": participant_order,
                "signed_mean": statistics[pattern_id],
                "participant_denominator": expected,
                "finite_participants": finite_count,
                "status": "observed" if complete else "insufficient",
            }
        )
    observed = float(np.mean(deltas)) if complete else float("nan")
    p_value = (
        float(np.mean(statistics >= observed))
        if complete
        else float("nan")
    )
    summary = pd.DataFrame(
        [
            {
                "observed_mean_improvement": observed,
                "p_value": p_value,
                "participant_order": participant_order,
                "tail_convention": "exact_inclusive",
                "alternative": "larger_improvement",
                "sign_patterns": pattern_count,
                "participant_denominator": expected,
                "finite_participants": finite_count,
                "status": "observed" if complete else "insufficient",
            }
        ]
    )
    return SignFlipResult(distribution=pd.DataFrame.from_records(records), summary=summary)


def generate_coherent_derangements(
    participants: Sequence[Any],
    *,
    draws: int = DEFAULT_DERANGEMENT_DRAWS,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> pd.DataFrame:
    """Generate deterministic one-to-one, non-self participant assignments."""
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or int(draws) <= 0:
        raise ValueError("draws must be an exact positive integer")
    if isinstance(root_seed, bool) or not isinstance(root_seed, (int, np.integer)):
        raise ValueError("root_seed must be an exact integer")
    ordered = _ordered_unique_participants(participants)
    tokens = [_identity_token(value) for value in ordered]
    count = len(ordered)
    records: list[dict[str, Any]] = []
    for draw_id in range(int(draws)):
        attempt = 0
        while True:
            assignment_seed = _deterministic_seed(
                int(root_seed), "frozen_g3", "coherent_derangement", draw_id, attempt
            )
            permutation = np.random.default_rng(assignment_seed).permutation(count)
            if bool(np.all(permutation != np.arange(count))):
                break
            attempt += 1
        mapping = [f"{tokens[index]}->{tokens[int(permutation[index])]}" for index in range(count)]
        assignment_hash = hashlib.sha256("|".join(mapping).encode("utf-8")).hexdigest()
        for recipient_index, donor_index in enumerate(permutation):
            records.append(
                {
                    "draw_id": draw_id,
                    "recipient_subject": ordered[recipient_index],
                    "donor_subject": ordered[int(donor_index)],
                    "assignment_seed": assignment_seed,
                    "assignment_attempt": attempt,
                    "assignment_hash": assignment_hash,
                    "participant_denominator": count,
                    "status": "observed",
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["draw_id", "recipient_subject"], key=lambda column: column.map(_identity_token),
        kind="stable",
    ).reset_index(drop=True)


def frozen_gate_decision_table(
    *,
    relative_transfer_reduction: float,
    improved_participants: int,
    sign_flip_p: float,
    coherent_placebo_p: float,
    minimum_lopo_improvement: float,
    cross_modal_discrepancy_improvement: float,
    sign_agreement_delta: float,
    participant_denominator: int = FROZEN_PARTICIPANT_DENOMINATOR,
    finite_participants: int = FROZEN_PARTICIPANT_DENOMINATOR,
) -> pd.DataFrame:
    """Emit every immutable frozen G3 gate and their conjunction."""
    if int(participant_denominator) != FROZEN_PARTICIPANT_DENOMINATOR:
        raise ValueError("Frozen G3 requires the fixed ten-participant denominator")
    denominator = int(participant_denominator)
    finite = int(finite_participants)
    complete = finite == denominator
    checks = (
        (
            "transfer_reduction_at_least_5pct",
            float(relative_transfer_reduction),
            0.05,
            ">=",
            np.isfinite(relative_transfer_reduction)
            and float(relative_transfer_reduction) >= 0.05,
        ),
        (
            "at_least_7_of_10_participants_improve",
            int(improved_participants),
            7,
            ">=",
            int(improved_participants) >= 7,
        ),
        (
            "exact_sign_flip_p_at_most_005",
            float(sign_flip_p),
            0.05,
            "<=",
            np.isfinite(sign_flip_p) and float(sign_flip_p) <= 0.05,
        ),
        (
            "coherent_placebo_p_at_most_005",
            float(coherent_placebo_p),
            0.05,
            "<=",
            np.isfinite(coherent_placebo_p) and float(coherent_placebo_p) <= 0.05,
        ),
        (
            "all_lopo_improvements_positive",
            float(minimum_lopo_improvement),
            0.0,
            ">",
            np.isfinite(minimum_lopo_improvement)
            and float(minimum_lopo_improvement) > 0.0,
        ),
        (
            "cross_modal_discrepancy_improves",
            float(cross_modal_discrepancy_improvement),
            0.0,
            ">",
            np.isfinite(cross_modal_discrepancy_improvement)
            and float(cross_modal_discrepancy_improvement) > 0.0,
        ),
        (
            "sign_agreement_delta_at_least_minus_002",
            float(sign_agreement_delta),
            -0.02,
            ">=",
            np.isfinite(sign_agreement_delta)
            and float(sign_agreement_delta) >= -0.02,
        ),
    )
    status = "observed" if complete else "insufficient"
    rows = [
        {
            "gate": name,
            "observed": observed,
            "threshold": threshold,
            "comparator": comparator,
            "passed": bool(complete and passed),
            "participant_denominator": denominator,
            "finite_participants": finite,
            "status": status,
            "reason": (
                "criterion_met"
                if complete and passed
                else "fixed_participant_denominator_incomplete"
                if not complete
                else "criterion_not_met"
            ),
        }
        for name, observed, threshold, comparator, passed in checks
    ]
    final_pass = bool(complete and all(row["passed"] for row in rows))
    rows.append(
        {
            "gate": "frozen_g3_pass",
            "observed": int(sum(row["passed"] for row in rows)),
            "threshold": len(checks),
            "comparator": "all",
            "passed": final_pass,
            "participant_denominator": denominator,
            "finite_participants": finite,
            "status": status,
            "reason": (
                "all_frozen_criteria_met"
                if final_pass
                else "fixed_participant_denominator_incomplete"
                if not complete
                else "one_or_more_frozen_criteria_failed"
            ),
        }
    )
    return pd.DataFrame.from_records(rows)


def _canonical_validated_domains(validated_domains: Sequence[str]) -> tuple[str, ...]:
    if isinstance(validated_domains, (str, bytes)) or not isinstance(
        validated_domains, Sequence
    ):
        raise ValueError(
            "validated_domains must be an explicit nonempty ordered sequence"
        )
    supplied = tuple(validated_domains)
    if not supplied or any(not isinstance(domain, str) for domain in supplied):
        raise ValueError(
            "validated_domains must be an explicit nonempty sequence of frozen domains"
        )
    if len(supplied) != len(set(supplied)):
        raise ValueError("validated_domains must not contain duplicates")
    available = tuple(FROZEN_DOMAIN_PRIMITIVES)
    unknown = sorted(set(supplied).difference(available))
    if unknown:
        raise ValueError(
            "validated_domains contains domains outside the frozen G3 core: "
            f"{unknown}"
        )
    return tuple(domain for domain in available if domain in supplied)


def evaluate_frozen_cold_start(
    loso_tables: LosoRepresentationTables,
    future_oracles: pd.DataFrame,
    *,
    validated_domains: Sequence[str],
    derangement_draws: int = DEFAULT_DERANGEMENT_DRAWS,
    root_seed: int = DEFAULT_ROOT_SEED,
) -> FrozenColdStartResult:
    """Evaluate the immutable Task 3 frozen G3 contrast.

    The only primary candidate is shrinkage at k=14/lambda=7, compared with
    the recipient-excluding population baseline. The separate future-oracle
    table is used solely here, after Task 3 representations are frozen.
    """
    domains = _canonical_validated_domains(validated_domains)
    if (
        isinstance(derangement_draws, bool)
        or not isinstance(derangement_draws, (int, np.integer))
        or int(derangement_draws) <= 0
    ):
        raise ValueError("derangement_draws must be an exact positive integer")
    if isinstance(root_seed, bool) or not isinstance(root_seed, (int, np.integer)):
        raise ValueError("root_seed must be an exact integer")
    draws = int(derangement_draws)
    seed = int(root_seed)

    representations, baselines, oracles, participants = _prepare_frozen_inputs(
        loso_tables, future_oracles
    )
    (
        primitive_transfer,
        domain_transfer,
        participant_transfer,
        method_summaries,
    ) = _build_transfer_tables(baselines, oracles, domains)
    primary = participant_transfer.loc[
        participant_transfer["_method"].eq("shrinkage")
        & participant_transfer["_k"].eq(PRIMARY_K)
        & participant_transfer["_donor_token"].eq(_NULL_DONOR)
    ].sort_values("_held_token", kind="stable")
    if len(primary) != FROZEN_PARTICIPANT_DENOMINATOR:
        raise ValueError("Primary shrinkage contrast requires exactly 10 participants")

    sign_flip = exact_participant_sign_flip(
        primary.loc[:, ["held_out_subject", "improvement"]],
        expected_participants=FROZEN_PARTICIPANT_DENOMINATOR,
    )
    lopo = _build_lopo_summary(primary)
    (
        derangement_assignments,
        derangement_distribution,
        derangement_summary,
    ) = _score_coherent_derangements(
        participants,
        participant_transfer,
        primary,
        draws=draws,
        root_seed=seed,
    )
    pair_days, pair_summaries, participant_discrepancy = _build_cross_modal_tables(
        representations, domains
    )

    transfer_finite = int(np.isfinite(primary["improvement"]).sum())
    discrepancy_finite = int(
        np.isfinite(participant_discrepancy["discrepancy_improvement"]).sum()
    )
    sign_finite = int(
        np.isfinite(participant_discrepancy["sign_agreement_delta"]).sum()
    )
    finite_participants = min(transfer_finite, discrepancy_finite, sign_finite)
    primary_complete = transfer_finite == FROZEN_PARTICIPANT_DENOMINATOR
    population_error = (
        float(primary["population_transfer_error"].mean())
        if primary_complete
        else np.nan
    )
    candidate_error = (
        float(primary["candidate_transfer_error"].mean())
        if primary_complete
        else np.nan
    )
    transfer_improvement = (
        population_error - candidate_error if primary_complete else np.nan
    )
    relative_transfer_reduction = (
        transfer_improvement / population_error
        if primary_complete and population_error > 0
        else np.nan
    )
    improved_participants = int(
        (pd.to_numeric(primary["improvement"], errors="coerce") > 0).sum()
    )
    lopo_complete = bool(lopo["status"].eq("observed").all())
    minimum_lopo = (
        float(lopo["lopo_mean_improvement"].min()) if lopo_complete else np.nan
    )
    discrepancy_complete = discrepancy_finite == FROZEN_PARTICIPANT_DENOMINATOR
    sign_complete = sign_finite == FROZEN_PARTICIPANT_DENOMINATOR
    discrepancy_improvement = (
        float(participant_discrepancy["discrepancy_improvement"].mean())
        if discrepancy_complete
        else np.nan
    )
    sign_agreement_delta = (
        float(participant_discrepancy["sign_agreement_delta"].mean())
        if sign_complete
        else np.nan
    )
    decision = frozen_gate_decision_table(
        relative_transfer_reduction=relative_transfer_reduction,
        improved_participants=improved_participants,
        sign_flip_p=float(sign_flip.summary.loc[0, "p_value"]),
        coherent_placebo_p=float(derangement_summary.loc[0, "p_value"]),
        minimum_lopo_improvement=minimum_lopo,
        cross_modal_discrepancy_improvement=discrepancy_improvement,
        sign_agreement_delta=sign_agreement_delta,
        participant_denominator=FROZEN_PARTICIPANT_DENOMINATOR,
        finite_participants=finite_participants,
    )
    metadata = {
        "primary_method": "shrinkage",
        "primary_k": PRIMARY_K,
        "primary_lambda_days": PRIMARY_LAMBDA_DAYS,
        "validated_domains": json.dumps(domains),
    }
    for table in (
        sign_flip.distribution,
        sign_flip.summary,
        derangement_assignments,
        derangement_distribution,
        derangement_summary,
        lopo,
        decision,
    ):
        for column, value in metadata.items():
            table[column] = value
    decision["derangement_draws"] = draws
    decision["root_seed"] = seed

    public_participant_transfer = participant_transfer.drop(
        columns=["_held_token", "_k", "_method", "_donor_token", "_method_order"]
    ).reset_index(drop=True)
    public_participant_discrepancy = participant_discrepancy.drop(
        columns="_held_token"
    ).reset_index(drop=True)
    return FrozenColdStartResult(
        primitive_transfer_errors=primitive_transfer,
        domain_transfer_errors=domain_transfer,
        participant_transfer_deltas=public_participant_transfer,
        method_transfer_summaries=method_summaries,
        pair_day_discrepancies=pair_days,
        pair_discrepancy_summaries=pair_summaries,
        participant_discrepancy_deltas=public_participant_discrepancy,
        sign_flip_distribution=sign_flip.distribution,
        sign_flip_summary=sign_flip.summary,
        coherent_derangement_assignments=derangement_assignments,
        coherent_derangement_distribution=derangement_distribution,
        coherent_derangement_summary=derangement_summary,
        lopo_summary=lopo,
        frozen_gate_decision=decision,
    )
