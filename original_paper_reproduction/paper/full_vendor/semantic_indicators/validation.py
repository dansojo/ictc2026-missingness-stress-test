"""Participant-aware convergent-validity statistics for frozen sensor pairs.

The module is deliberately label-free.  It treats participants, rather than
participant-days, as the resampling and direction-counting units and keeps
degenerate participants visible in every fixed-denominator result.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import PRIMARY_PAIRS, PairDefinition


DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_SHIFT_DRAWS = 1_000
DEFAULT_ROOT_SEED = 42
PRIMARY_HIGH_COVERAGE_THRESHOLD = 0.8
HIGH_COVERAGE_THRESHOLDS = (0.0, 0.5, 0.7, 0.8, 0.9)
EVALUATION_START_DAY = 22
MIN_PARTICIPANT_SPEARMAN_DAYS = 3

_NUISANCE_ROLES = (
    "left_coverage",
    "right_coverage",
    "left_gap",
    "right_gap",
    "left_support",
    "right_support",
)
_USAGE_SENSITIVITY_SLOTS = (
    # sensitivity_id, endpoint, operator, pair, left, frozen right, primary
    ("load.total", "load", "total", "digital_load", "screen_load_24h", "usage_load_24h", True),
    ("load.capped", "load", "capped", "digital_load", "screen_load_24h", None, False),
    ("load.max_app", "load", "max_app", "digital_load", "screen_load_24h", None, False),
    ("load.record_mean", "load", "record_mean", "digital_load", "screen_load_24h", None, False),
    (
        "timing.total",
        "timing",
        "total",
        "digital_disengagement",
        "screen_disengagement_p90",
        "usage_disengagement_p90",
        True,
    ),
    (
        "timing.capped",
        "timing",
        "capped",
        "digital_disengagement",
        "screen_disengagement_p90",
        None,
        False,
    ),
    (
        "timing.max_app",
        "timing",
        "max_app",
        "digital_disengagement",
        "screen_disengagement_p90",
        None,
        False,
    ),
    (
        "timing.record_mean",
        "timing",
        "record_mean",
        "digital_disengagement",
        "screen_disengagement_p90",
        None,
        False,
    ),
)
_PAIR_DOMAINS = {
    "digital_load": "digital",
    "digital_disengagement": "digital",
    "physical_load": "physical",
    "light_exposure": "light",
}


@dataclass
class ClusterBootstrapResult:
    """Auditable participant-cluster bootstrap outputs."""

    summary: pd.DataFrame
    draws: pd.DataFrame
    participant_statistics: pd.DataFrame


@dataclass
class CalendarShiftResult:
    """Auditable complete-calendar shift-null outputs."""

    summary: pd.DataFrame
    draws: pd.DataFrame
    assignments: pd.DataFrame
    row_audit: pd.DataFrame


@dataclass
class LopoAdjustmentResult:
    """LOPO nuisance fits, held-out residuals, and their aggregate estimate."""

    summary: pd.DataFrame
    rows: pd.DataFrame
    folds: pd.DataFrame
    coefficients: pd.DataFrame


@dataclass
class ConvergentValidityResult:
    """All row-, participant-, draw-, sensitivity-, and decision-level tables."""

    row_table: pd.DataFrame
    participant_table: pd.DataFrame
    adjusted_participant_table: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    bootstrap_participant_statistics: pd.DataFrame
    shift_summary: pd.DataFrame
    shift_draws: pd.DataFrame
    shift_assignments: pd.DataFrame
    lopo_rows: pd.DataFrame
    lopo_folds: pd.DataFrame
    lopo_coefficients: pd.DataFrame
    lopo_summary: pd.DataFrame
    raw_lopo: pd.DataFrame
    quality_sensitivity: pd.DataFrame
    usage_sensitivity: pd.DataFrame
    pair_summary: pd.DataFrame
    domain_summary: pd.DataFrame
    decision_table: pd.DataFrame


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def derive_deterministic_seed(root_seed: int, *parts: Any) -> int:
    """Derive a stable NumPy seed from SHA-256, independent of process state."""
    if isinstance(root_seed, (bool, np.bool_)) or int(root_seed) != root_seed:
        raise ValueError("root_seed must be an integer")
    payload = json.dumps(
        [_json_value(int(root_seed)), *(_json_value(part) for part in parts)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _ordered_participants(frame: pd.DataFrame, participant_column: str) -> list[Any]:
    _require_columns(frame, [participant_column])
    participants = frame[participant_column].drop_duplicates().tolist()
    return sorted(participants, key=lambda value: (str(type(value)), str(value)))


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _centered_sufficient_statistics(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[int, float, float, float, float, float]:
    """Return stable two-pass centered moments for finite paired arrays."""
    count = int(len(x))
    if not count:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0
    sum_x = float(x.sum())
    sum_y = float(y.sum())
    centered_x = x - float(x.mean())
    centered_y = y - float(y.mean())
    ss_x = float(np.dot(centered_x, centered_x))
    ss_y = float(np.dot(centered_y, centered_y))
    cross = float(np.dot(centered_x, centered_y))
    return count, sum_x, sum_y, ss_x, ss_y, cross


def _participant_sufficient_statistics(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    participant_column: str,
    *,
    pair_name: str,
    participants: Sequence[Any] | None = None,
) -> pd.DataFrame:
    _require_columns(frame, [participant_column, left_column, right_column])
    fixed_participants = (
        list(participants)
        if participants is not None
        else _ordered_participants(frame, participant_column)
    )
    records: list[dict[str, Any]] = []
    for participant in fixed_participants:
        subject = frame.loc[frame[participant_column].eq(participant)]
        left = _numeric(subject[left_column])
        right = _numeric(subject[right_column])
        finite = np.isfinite(left) & np.isfinite(right)
        x = left[finite]
        y = right[finite]
        count, sum_x, sum_y, ss_x, ss_y, cross = _centered_sufficient_statistics(x, y)
        records.append(
            {
                "pair": pair_name,
                participant_column: participant,
                "finite_count": count,
                "sum_left": sum_x,
                "sum_right": sum_y,
                "centered_ss_left": ss_x,
                "centered_ss_right": ss_y,
                "centered_cross_product": cross,
                "status": "ok" if count >= 2 and ss_x > 0.0 and ss_y > 0.0 else "degenerate",
                "participant_denominator": len(fixed_participants),
            }
        )
    return pd.DataFrame.from_records(records)


def _correlation_from_statistics(statistics: pd.DataFrame) -> tuple[float, int, str]:
    if statistics.empty:
        return float("nan"), 0, "insufficient_participants"
    ss_left = float(statistics["centered_ss_left"].sum())
    ss_right = float(statistics["centered_ss_right"].sum())
    cross = float(statistics["centered_cross_product"].sum())
    finite_count = int(statistics["finite_count"].sum())
    denominator = float(np.sqrt(max(ss_left, 0.0) * max(ss_right, 0.0)))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan"), finite_count, "degenerate"
    estimate = float(np.clip(cross / denominator, -1.0, 1.0))
    return estimate, finite_count, "ok"


def repeated_measures_correlation(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    participant_column: str = "subject_id",
) -> float:
    """Return the exact participant-centered cross-product correlation."""
    statistics = _participant_sufficient_statistics(
        frame,
        left_column,
        right_column,
        participant_column,
        pair_name="unspecified",
    )
    estimate, _finite_count, _status = _correlation_from_statistics(statistics)
    return estimate


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ranks_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ranks_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    centered_x = ranks_x - ranks_x.mean()
    centered_y = ranks_y - ranks_y.mean()
    denominator = float(np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.clip(np.dot(centered_x, centered_y) / denominator, -1.0, 1.0))


def participant_correlations(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    participant_column: str = "subject_id",
    *,
    pair_name: str = "unspecified",
    participants: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Return one Spearman row per fixed-denominator participant."""
    _require_columns(frame, [participant_column, left_column, right_column])
    fixed_participants = (
        list(participants)
        if participants is not None
        else _ordered_participants(frame, participant_column)
    )
    records: list[dict[str, Any]] = []
    for participant in fixed_participants:
        subject = frame.loc[frame[participant_column].eq(participant)]
        left = _numeric(subject[left_column])
        right = _numeric(subject[right_column])
        finite = np.isfinite(left) & np.isfinite(right)
        x = left[finite]
        y = right[finite]
        finite_count = int(finite.sum())
        if finite_count < MIN_PARTICIPANT_SPEARMAN_DAYS:
            estimate = float("nan")
            status = "insufficient_finite"
        else:
            constant_left = bool(np.ptp(x) == 0.0)
            constant_right = bool(np.ptp(y) == 0.0)
            if constant_left and constant_right:
                estimate = float("nan")
                status = "constant_both"
            elif constant_left:
                estimate = float("nan")
                status = "constant_left"
            elif constant_right:
                estimate = float("nan")
                status = "constant_right"
            else:
                estimate = _spearman(x, y)
                status = "ok" if np.isfinite(estimate) else "degenerate"
        records.append(
            {
                "pair": pair_name,
                participant_column: participant,
                "finite_count": finite_count,
                "spearman_rho": estimate,
                "direction_positive": bool(np.isfinite(estimate) and estimate > 0.0),
                "status": status,
                "participant_denominator": len(fixed_participants),
            }
        )
    return pd.DataFrame.from_records(records)


def cluster_bootstrap_correlation(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    participant_column: str = "subject_id",
    *,
    pair_name: str = "unspecified",
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_ROOT_SEED,
    confidence_level: float = 0.95,
    participants: Sequence[Any] | None = None,
) -> ClusterBootstrapResult:
    """Bootstrap participants with replacement from precomputed sufficient statistics."""
    draw_count = _validate_positive_integer(draws, "draws")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one")
    statistics = _participant_sufficient_statistics(
        frame,
        left_column,
        right_column,
        participant_column,
        pair_name=pair_name,
        participants=participants,
    ).reset_index(drop=True)
    participant_count = len(statistics)
    observed_r, observed_finite, observed_status = _correlation_from_statistics(statistics)
    draw_records: list[dict[str, Any]] = []
    for draw in range(draw_count):
        draw_seed = derive_deterministic_seed(seed, "cluster_bootstrap", pair_name, draw)
        if participant_count:
            sampled_indices = np.random.default_rng(draw_seed).integers(
                0, participant_count, size=participant_count
            )
            sampled = statistics.iloc[sampled_indices]
            estimate, finite_count, status = _correlation_from_statistics(sampled)
            sampled_participants = [
                _json_value(value)
                for value in sampled[participant_column].tolist()
            ]
        else:
            estimate, finite_count, status = float("nan"), 0, "insufficient_participants"
            sampled_participants = []
        draw_records.append(
            {
                "pair": pair_name,
                "draw": draw,
                "root_seed": int(seed),
                "draw_seed": draw_seed,
                "participant_denominator": participant_count,
                "finite_count": finite_count,
                "sampled_participants": json.dumps(sampled_participants, ensure_ascii=False),
                "estimate": estimate,
                "status": status,
            }
        )
    draw_table = pd.DataFrame.from_records(draw_records)
    finite_draws = pd.to_numeric(draw_table["estimate"], errors="coerce").dropna()
    alpha = (1.0 - confidence_level) / 2.0
    if finite_draws.empty:
        ci_low = ci_high = float("nan")
        summary_status = "no_finite_draws"
    else:
        ci_low, ci_high = np.quantile(finite_draws.to_numpy(dtype=float), [alpha, 1.0 - alpha])
        ci_low = float(ci_low)
        ci_high = float(ci_high)
        summary_status = "ok" if observed_status == "ok" else observed_status
    summary = pd.DataFrame.from_records(
        [
            {
                "pair": pair_name,
                "participant_denominator": participant_count,
                "finite_count": observed_finite,
                "observed_r": observed_r,
                "draws_requested": draw_count,
                "finite_draws": int(len(finite_draws)),
                "confidence_level": float(confidence_level),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "root_seed": int(seed),
                "status": summary_status,
            }
        ]
    )
    return ClusterBootstrapResult(summary, draw_table, statistics)


def holm_adjust(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return Holm FWER-adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    invalid = np.isfinite(values) & ((values < 0.0) | (values > 1.0))
    if invalid.any():
        raise ValueError("finite p-values must lie in [0,1]")
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    order = finite_indices[np.argsort(values[finite_indices], kind="stable")]
    running = 0.0
    total_hypotheses = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total_hypotheses - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _validated_calendar(
    frame: pd.DataFrame,
    participant_column: str,
    day_column: str,
    date_column: str | None,
    sensor_day_column: str | None,
) -> pd.DataFrame:
    required = [participant_column, day_column]
    if date_column is not None:
        required.append(date_column)
    if sensor_day_column is not None:
        required.append(sensor_day_column)
    _require_columns(frame, required)
    if frame.duplicated([participant_column, day_column]).any():
        raise ValueError("Each participant elapsed-day key must be unique")
    result = frame.copy()
    numeric_day = pd.to_numeric(result[day_column], errors="coerce")
    day_array = numeric_day.to_numpy(dtype=float, na_value=np.nan)
    if (
        not np.isfinite(day_array).all()
        or not np.equal(day_array, np.floor(day_array)).all()
        or (day_array < 1).any()
    ):
        raise ValueError(f"{day_column} must contain positive finite integers")
    result[day_column] = day_array.astype(np.int64)
    if date_column is not None:
        parsed_dates = pd.to_datetime(result[date_column], errors="coerce")
        if getattr(parsed_dates.dt, "tz", None) is not None:
            raise ValueError(f"{date_column} must be timezone-naive")
        if parsed_dates.isna().any() or not parsed_dates.eq(parsed_dates.dt.normalize()).all():
            raise ValueError(f"{date_column} must contain exact local calendar dates")
        result[date_column] = parsed_dates
    for participant, subject in result.groupby(participant_column, sort=False):
        ordered = subject.sort_values(day_column, kind="stable")
        days = ordered[day_column].to_numpy(dtype=int)
        expected = np.arange(int(days.min()), int(days.max()) + 1, dtype=int)
        if int(days.min()) != 1 or not np.array_equal(days, expected):
            raise ValueError(
                "Shift null requires the complete elapsed calendar, including internal gap rows; "
                f"participant {participant!r} is incomplete"
            )
        if date_column is not None:
            dates = ordered[date_column].reset_index(drop=True)
            expected_dates = dates.iloc[0] + pd.to_timedelta(days - days[0], unit="D")
            if not dates.equals(pd.Series(expected_dates, name=date_column)):
                raise ValueError(
                    "Calendar dates must advance with the complete elapsed calendar"
                )
    return result.sort_values([participant_column, day_column], kind="stable").reset_index(drop=True)


def _right_tuple_columns(
    frame: pd.DataFrame,
    right_column: str,
    requested: Sequence[str] | None,
) -> list[str]:
    if requested is None:
        columns = [
            column
            for column in frame.columns
            if column == right_column or column.startswith(f"{right_column}__")
        ]
    else:
        columns = list(requested)
    if right_column not in columns:
        columns.insert(0, right_column)
    columns = list(dict.fromkeys(columns))
    _require_columns(frame, columns)
    return columns


def _weekday_source_indices(dates: pd.Series, shift: int) -> np.ndarray:
    weekdays = dates.dt.weekday.to_numpy(dtype=int)
    source_indices = np.arange(len(dates), dtype=int)
    for weekday in range(7):
        positions = np.flatnonzero(weekdays == weekday)
        if positions.size:
            source_indices[positions] = np.roll(positions, int(shift))
    return source_indices


def circular_shift_null(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    participant_column: str = "subject_id",
    day_column: str = "elapsed_day_index",
    *,
    date_column: str | None = "lifelog_date",
    sensor_day_column: str | None = "sensor_day_id",
    right_tuple_columns: Sequence[str] | None = None,
    pair_name: str = "unspecified",
    mode: str = "unrestricted",
    evaluation_start_day: int = EVALUATION_START_DAY,
    draws: int = DEFAULT_SHIFT_DRAWS,
    seed: int = DEFAULT_ROOT_SEED,
    audit_rows: bool = False,
) -> CalendarShiftResult:
    """Shift a right primitive's complete tuple on the full participant calendar.

    ``mode='unrestricted'`` rotates all elapsed days. ``mode='weekday'`` rotates
    within each weekday stratum by one common, non-zero week offset per
    participant and draw.
    """
    draw_count = _validate_positive_integer(draws, "draws")
    start_day = _validate_positive_integer(evaluation_start_day, "evaluation_start_day")
    normalized_mode = {
        "weekday-preserving": "weekday",
        "weekday_preserving": "weekday",
    }.get(mode, mode)
    if normalized_mode not in {"unrestricted", "weekday"}:
        raise ValueError("mode must be unrestricted or weekday")
    if normalized_mode == "weekday" and date_column is None:
        raise ValueError("weekday mode requires date_column")
    calendar = _validated_calendar(
        frame,
        participant_column,
        day_column,
        date_column,
        sensor_day_column,
    )
    tuple_columns = _right_tuple_columns(calendar, right_column, right_tuple_columns)
    _require_columns(calendar, [left_column, right_column])
    participants = _ordered_participants(calendar, participant_column)
    participant_payloads: dict[Any, dict[str, Any]] = {}
    for participant in participants:
        subject = calendar.loc[calendar[participant_column].eq(participant)].reset_index(drop=True)
        participant_payloads[participant] = {
            "frame": subject,
            "left": _numeric(subject[left_column]),
            "right": _numeric(subject[right_column]),
            "day": subject[day_column].to_numpy(dtype=int),
            "dates": subject[date_column] if date_column is not None else None,
        }

    evaluation = calendar.loc[calendar[day_column].ge(start_day)]
    observed_statistics = _participant_sufficient_statistics(
        evaluation,
        left_column,
        right_column,
        participant_column,
        pair_name=pair_name,
        participants=participants,
    )
    observed_r, observed_finite, observed_status = _correlation_from_statistics(
        observed_statistics
    )

    draw_records: list[dict[str, Any]] = []
    assignment_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    for draw in range(draw_count):
        draw_seed = derive_deterministic_seed(
            seed, "calendar_shift", normalized_mode, pair_name, draw
        )
        shifted_statistics: list[dict[str, Any]] = []
        draw_valid = True
        for participant in participants:
            payload = participant_payloads[participant]
            subject = payload["frame"]
            length = len(subject)
            participant_seed = derive_deterministic_seed(
                seed,
                "calendar_shift",
                normalized_mode,
                pair_name,
                draw,
                participant,
            )
            rng = np.random.default_rng(participant_seed)
            if normalized_mode == "unrestricted":
                minimum_stratum = np.nan
                if length < 2:
                    shift = np.nan
                    assignment_status = "insufficient_calendar"
                    source_indices = np.arange(length, dtype=int)
                else:
                    shift = int(rng.integers(1, length))
                    assignment_status = "ok"
                    source_indices = np.roll(np.arange(length, dtype=int), shift)
            else:
                dates = payload["dates"]
                weekday_counts = dates.dt.weekday.value_counts().reindex(range(7), fill_value=0)
                positive_counts = weekday_counts.loc[weekday_counts.gt(0)]
                minimum_stratum = int(positive_counts.min()) if not positive_counts.empty else 0
                if minimum_stratum < 2:
                    shift = np.nan
                    assignment_status = "insufficient_weekday_stratum"
                    source_indices = np.arange(length, dtype=int)
                else:
                    shift = int(rng.integers(1, minimum_stratum))
                    assignment_status = "ok"
                    source_indices = _weekday_source_indices(dates, shift)
            if assignment_status != "ok":
                draw_valid = False
            assignment_records.append(
                {
                    "pair": pair_name,
                    "null_mode": normalized_mode,
                    "draw": draw,
                    "root_seed": int(seed),
                    "draw_seed": draw_seed,
                    participant_column: participant,
                    "participant_seed": participant_seed,
                    "shift": shift,
                    "calendar_length": length,
                    "minimum_weekday_stratum": minimum_stratum,
                    "status": assignment_status,
                }
            )
            shifted_right = payload["right"][source_indices]
            evaluation_mask = payload["day"] >= start_day
            left = payload["left"][evaluation_mask]
            right = shifted_right[evaluation_mask]
            finite = np.isfinite(left) & np.isfinite(right)
            x = left[finite]
            y = right[finite]
            count, _sum_x, _sum_y, ss_x, ss_y, cross = (
                _centered_sufficient_statistics(x, y)
            )
            shifted_statistics.append(
                {
                    "finite_count": count,
                    "centered_ss_left": ss_x,
                    "centered_ss_right": ss_y,
                    "centered_cross_product": cross,
                }
            )
            if audit_rows:
                for destination_index, source_index in enumerate(source_indices):
                    destination = subject.iloc[destination_index]
                    source = subject.iloc[int(source_index)]
                    record: dict[str, Any] = {
                        "pair": pair_name,
                        "null_mode": normalized_mode,
                        "draw": draw,
                        "root_seed": int(seed),
                        "draw_seed": draw_seed,
                        participant_column: participant,
                        day_column: destination[day_column],
                        "source_elapsed_day_index": source[day_column],
                    }
                    if sensor_day_column is not None:
                        record[sensor_day_column] = destination[sensor_day_column]
                        record[f"source_{sensor_day_column}"] = source[sensor_day_column]
                    if date_column is not None:
                        record[date_column] = destination[date_column]
                        record[f"source_{date_column}"] = source[date_column]
                    for column in tuple_columns:
                        record[column] = source[column]
                    audit_records.append(record)
        if draw_valid:
            estimate, finite_count, status = _correlation_from_statistics(
                pd.DataFrame.from_records(shifted_statistics)
            )
        else:
            estimate, finite_count, status = float("nan"), 0, "invalid_assignment"
        draw_records.append(
            {
                "pair": pair_name,
                "null_mode": normalized_mode,
                "draw": draw,
                "root_seed": int(seed),
                "draw_seed": draw_seed,
                "participant_denominator": len(participants),
                "finite_count": finite_count,
                "observed_r": observed_r,
                "null_r": estimate,
                "status": status,
            }
        )

    draw_table = pd.DataFrame.from_records(draw_records)
    finite_null = pd.to_numeric(draw_table["null_r"], errors="coerce").dropna()
    invalid_draws = draw_count - int(len(finite_null))
    if observed_status == "ok":
        finite_exceedances = int(finite_null.ge(observed_r).sum())
        exceedances = finite_exceedances + invalid_draws
        p_value = (exceedances + 1.0) / (draw_count + 1.0)
        summary_status = "ok" if invalid_draws == 0 else "incomplete_draws"
    else:
        finite_exceedances = 0
        invalid_draws = draw_count
        exceedances = draw_count
        p_value = 1.0
        summary_status = observed_status
    summary = pd.DataFrame.from_records(
        [
            {
                "pair": pair_name,
                "null_mode": normalized_mode,
                "participant_denominator": len(participants),
                "finite_count": observed_finite,
                "observed_r": observed_r,
                "draws_requested": draw_count,
                "finite_draws": int(len(finite_null)),
                "finite_exceedances": finite_exceedances,
                "conservative_invalid_draws": invalid_draws,
                "exceedances": exceedances,
                "p_value_plus_one": float(p_value),
                "evaluation_start_day": start_day,
                "root_seed": int(seed),
                "tuple_columns": json.dumps(tuple_columns, ensure_ascii=False),
                "status": summary_status,
            }
        ]
    )
    return CalendarShiftResult(
        summary=summary,
        draws=draw_table,
        assignments=pd.DataFrame.from_records(assignment_records),
        row_audit=pd.DataFrame.from_records(audit_records),
    )


def leave_one_participant_out_correlations(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    participant_column: str = "subject_id",
    *,
    pair_name: str = "unspecified",
    participants: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Return raw centered correlations after omitting each participant in turn."""
    fixed_participants = (
        list(participants)
        if participants is not None
        else _ordered_participants(frame, participant_column)
    )
    records: list[dict[str, Any]] = []
    for held_out in fixed_participants:
        retained_participants = [item for item in fixed_participants if item != held_out]
        retained = frame.loc[~frame[participant_column].eq(held_out)]
        statistics = _participant_sufficient_statistics(
            retained,
            left_column,
            right_column,
            participant_column,
            pair_name=pair_name,
            participants=retained_participants,
        )
        estimate, finite_count, status = _correlation_from_statistics(statistics)
        records.append(
            {
                "pair": pair_name,
                "held_out_subject": held_out,
                "participant_denominator": len(fixed_participants),
                "retained_participants": len(retained_participants),
                "finite_count": finite_count,
                "estimate": estimate,
                "status": status,
            }
        )
    return pd.DataFrame.from_records(records)


def _participant_centered_feature(
    values: pd.Series,
    participants: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
        index=values.index,
    )
    missing = (~finite).astype(float)
    participant_means = numeric.where(finite).groupby(participants, dropna=False).transform("mean")
    filled = numeric.where(finite, participant_means).fillna(0.0)
    centered = filled - filled.groupby(participants, dropna=False).transform("mean")
    centered_missing = missing - missing.groupby(participants, dropna=False).transform("mean")
    return centered.astype(float), centered_missing.astype(float)


def _participant_centered_response(values: pd.Series, participants: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = pd.Series(
        np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
        index=values.index,
    )
    means = numeric.where(finite).groupby(participants, dropna=False).transform("mean")
    return numeric.where(finite) - means


def _build_nuisance_design(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    participant_column: str,
    day_column: str,
    date_column: str,
    nuisance_columns: Mapping[str, str | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_roles = sorted(set(_NUISANCE_ROLES).difference(nuisance_columns))
    if missing_roles:
        raise ValueError(f"nuisance_columns is missing roles: {missing_roles}")
    participants = frame[participant_column]
    design: dict[str, pd.Series] = {}
    provenance_records: list[dict[str, Any]] = []
    for role in _NUISANCE_ROLES:
        source_column = nuisance_columns.get(role)
        if source_column is None or source_column not in frame.columns:
            source = pd.Series(np.nan, index=frame.index, dtype=float)
            source_status = "unavailable"
        else:
            source = frame[source_column]
            source_status = "available"
        centered, centered_missing = _participant_centered_feature(source, participants)
        design[role] = centered
        design[f"{role}__missing"] = centered_missing
        provenance_records.append(
            {
                "term": role,
                "source_column": source_column,
                "source_status": source_status,
            }
        )
        provenance_records.append(
            {
                "term": f"{role}__missing",
                "source_column": source_column,
                "source_status": source_status,
            }
        )
    for side, value_column in (("left", left_column), ("right", right_column)):
        numeric = pd.to_numeric(frame[value_column], errors="coerce")
        missing = (~np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))).astype(float)
        missing_series = pd.Series(missing, index=frame.index)
        design[f"{side}_value_missing"] = missing_series - missing_series.groupby(
            participants, dropna=False
        ).transform("mean")
        provenance_records.append(
            {
                "term": f"{side}_value_missing",
                "source_column": value_column,
                "source_status": "available",
            }
        )
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any() or getattr(dates.dt, "tz", None) is not None:
        raise ValueError("LOPO weekday terms require finite timezone-naive dates")
    weekday_angle = 2.0 * np.pi * dates.dt.weekday.astype(float) / 7.0
    for term, values in (
        ("weekday_sin", np.sin(weekday_angle)),
        ("weekday_cos", np.cos(weekday_angle)),
        ("elapsed_day_linear", pd.to_numeric(frame[day_column], errors="coerce")),
    ):
        values = pd.Series(values, index=frame.index, dtype=float)
        design[term] = values - values.groupby(participants, dropna=False).transform("mean")
        provenance_records.append(
            {"term": term, "source_column": date_column if term.startswith("weekday") else day_column,
             "source_status": "available"}
        )
    return pd.DataFrame(design, index=frame.index), pd.DataFrame.from_records(provenance_records)


def _fit_held_out_response(
    design: pd.DataFrame,
    centered_response: pd.Series,
    participant_values: pd.Series,
    held_out: Any,
) -> tuple[np.ndarray, str, int, int, int, dict[str, tuple[float, str]]]:
    training = ~participant_values.eq(held_out)
    response_array = centered_response.to_numpy(dtype=float, na_value=np.nan)
    valid_training = training.to_numpy(dtype=bool) & np.isfinite(response_array)
    train_design = design.loc[valid_training]
    train_response = response_array[valid_training]
    candidate_terms: list[str] = []
    term_records: dict[str, tuple[float, str]] = {}
    for term in design.columns:
        values = train_design[term].to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).all() or float(np.ptp(values)) <= 1e-12:
            term_records[term] = (0.0, "constant_or_unavailable")
        else:
            candidate_terms.append(term)
    active_terms: list[str] = []
    current_rank = 0
    for term in candidate_terms:
        proposed = active_terms + [term]
        proposed_matrix = train_design[proposed].to_numpy(dtype=float)
        proposed_rank = int(np.linalg.matrix_rank(proposed_matrix))
        if proposed_rank > current_rank:
            active_terms.append(term)
            current_rank = proposed_rank
        else:
            # Matching-screen Usage proxy terms are intentionally duplicated.
            # Keep the first frozen role and expose every deterministic drop.
            term_records[term] = (0.0, "collinear_dropped")
    training_rows = int(valid_training.sum())
    residual = np.full(len(design), np.nan, dtype=float)
    held_out_mask = participant_values.eq(held_out).to_numpy(dtype=bool)
    held_out_finite = held_out_mask & np.isfinite(response_array)
    if not active_terms:
        status = "no_varying_terms"
        rank = 0
        residual[held_out_finite] = response_array[held_out_finite]
        return residual, status, training_rows, rank, 0, term_records
    matrix = train_design[active_terms].to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(matrix))
    if training_rows <= len(active_terms):
        status = "underdetermined"
    else:
        status = "ok"
    coefficients = np.linalg.lstsq(matrix, train_response, rcond=None)[0]
    for term, coefficient in zip(active_terms, coefficients, strict=True):
        term_records[term] = (float(coefficient), "active")
    if held_out_finite.any():
        held_design = design.loc[held_out_finite, active_terms].to_numpy(dtype=float)
        residual[held_out_finite] = response_array[held_out_finite] - held_design @ coefficients
    return residual, status, training_rows, rank, len(active_terms), term_records


def lopo_nuisance_adjustment(
    frame: pd.DataFrame,
    left_column: str,
    right_column: str,
    participant_column: str = "subject_id",
    day_column: str = "elapsed_day_index",
    *,
    date_column: str = "lifelog_date",
    sensor_day_column: str = "sensor_day_id",
    nuisance_columns: Mapping[str, str | None],
    evaluation_start_day: int = EVALUATION_START_DAY,
    pair_name: str = "unspecified",
) -> LopoAdjustmentResult:
    """Fit nuisance coefficients on other participants and apply to each holdout."""
    start_day = _validate_positive_integer(evaluation_start_day, "evaluation_start_day")
    _require_columns(frame, [left_column, right_column])
    calendar = _validated_calendar(
        frame,
        participant_column,
        day_column,
        date_column,
        sensor_day_column,
    )
    working = calendar.loc[calendar[day_column].ge(start_day)].copy()
    left_values = pd.to_numeric(working[left_column], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    right_values = pd.to_numeric(working[right_column], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    # G1's frozen analysis universe is pairwise-finite before any nuisance
    # centering or coefficient fit.  Endpoint-only rows must not change the
    # adjustment applied to the common convergent-validity row set.
    pairwise_finite = np.isfinite(left_values) & np.isfinite(right_values)
    working = working.loc[pairwise_finite].copy()
    participants = _ordered_participants(calendar, participant_column)
    design, provenance = _build_nuisance_design(
        working,
        left_column,
        right_column,
        participant_column,
        day_column,
        date_column,
        nuisance_columns,
    )
    centered_responses = {
        "left": _participant_centered_response(working[left_column], working[participant_column]),
        "right": _participant_centered_response(working[right_column], working[participant_column]),
    }
    held_out_rows: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    coefficient_records: list[dict[str, Any]] = []
    for held_out in participants:
        subject_mask = working[participant_column].eq(held_out)
        subject_rows = working.loc[
            subject_mask,
            [sensor_day_column, participant_column, date_column, day_column, left_column, right_column],
        ].copy()
        subject_rows["pair"] = pair_name
        for response_name in ("left", "right"):
            residual, status, training_rows, rank, active_terms, term_records = (
                _fit_held_out_response(
                    design,
                    centered_responses[response_name],
                    working[participant_column],
                    held_out,
                )
            )
            subject_rows[f"{response_name}_residual"] = residual[subject_mask.to_numpy(dtype=bool)]
            subject_rows[f"{response_name}_fit_status"] = status
            training_subjects = [participant for participant in participants if participant != held_out]
            fold_records.append(
                {
                    "pair": pair_name,
                    "held_out_subject": held_out,
                    "response": response_name,
                    "participant_denominator": len(participants),
                    "training_subjects": json.dumps(
                        [_json_value(value) for value in training_subjects], ensure_ascii=False
                    ),
                    "training_rows": training_rows,
                    "active_terms": active_terms,
                    "design_rank": rank,
                    "status": status,
                }
            )
            provenance_by_term = provenance.set_index("term")
            for term in design.columns:
                coefficient, term_status = term_records[term]
                coefficient_records.append(
                    {
                        "pair": pair_name,
                        "held_out_subject": held_out,
                        "response": response_name,
                        "term": term,
                        "source_column": provenance_by_term.loc[term, "source_column"],
                        "source_status": provenance_by_term.loc[term, "source_status"],
                        "coefficient": coefficient,
                        "term_status": term_status,
                        "fold_status": status,
                        "participant_denominator": len(participants),
                    }
                )
        held_out_rows.append(subject_rows)
    row_table = pd.concat(held_out_rows, ignore_index=True) if held_out_rows else pd.DataFrame()
    if row_table.empty:
        adjusted_r = float("nan")
        finite_count = 0
        correlation_status = "insufficient_participants"
    else:
        adjusted_statistics = _participant_sufficient_statistics(
            row_table,
            "left_residual",
            "right_residual",
            participant_column,
            pair_name=pair_name,
            participants=participants,
        )
        adjusted_r, finite_count, correlation_status = _correlation_from_statistics(
            adjusted_statistics
        )
    fold_table = pd.DataFrame.from_records(fold_records)
    all_folds_ok = bool(not fold_table.empty and fold_table["status"].eq("ok").all())
    summary_status = correlation_status if correlation_status != "ok" else (
        "ok" if all_folds_ok else "fold_fit_incomplete"
    )
    summary = pd.DataFrame.from_records(
        [
            {
                "pair": pair_name,
                "participant_denominator": len(participants),
                "finite_count": finite_count,
                "adjusted_r": adjusted_r,
                "folds_requested": len(participants) * 2,
                "folds_ok": int(fold_table["status"].eq("ok").sum()) if not fold_table.empty else 0,
                "status": summary_status,
            }
        ]
    )
    return LopoAdjustmentResult(
        summary=summary,
        rows=row_table,
        folds=fold_table,
        coefficients=pd.DataFrame.from_records(coefficient_records),
    )


def _available_column(frame: pd.DataFrame, candidate: str) -> str | None:
    return candidate if candidate in frame.columns else None


def _pair_metadata_mapping(
    frame: pd.DataFrame,
    pair: PairDefinition,
) -> dict[str, str | None]:
    left = pair.left
    right = pair.right
    mapping: dict[str, str | None] = {
        "left_coverage": _available_column(frame, f"{left}__coverage"),
        "right_coverage": _available_column(frame, f"{right}__coverage"),
        "left_gap": _available_column(frame, f"{left}__longest_gap_minutes"),
        "right_gap": _available_column(frame, f"{right}__longest_gap_minutes"),
        "left_support": _available_column(frame, f"{left}__support_density"),
        "right_support": _available_column(frame, f"{right}__support_density"),
    }
    # The frozen validity amendment treats matching screen availability as the
    # same-phone availability proxy for UsageStats.  Usage's own event-support
    # density remains a separate right-side nuisance term.
    if right.startswith("usage_") and left.startswith("screen_"):
        mapping["right_coverage"] = mapping["left_coverage"]
        mapping["right_gap"] = mapping["left_gap"]
    return mapping


def _series_or_nan(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _quality_score(
    frame: pd.DataFrame,
    mapping: Mapping[str, str | None],
) -> pd.Series:
    left = _series_or_nan(frame, mapping.get("left_coverage"))
    right = _series_or_nan(frame, mapping.get("right_coverage"))
    values = np.column_stack(
        [
            left.to_numpy(dtype=float, na_value=np.nan),
            right.to_numpy(dtype=float, na_value=np.nan),
        ]
    )
    finite_both = np.isfinite(values).all(axis=1)
    quality = np.full(len(frame), np.nan, dtype=float)
    quality[finite_both] = np.min(values[finite_both], axis=1)
    return pd.Series(quality, index=frame.index, dtype=float)


def _pair_estimate_record(
    frame: pd.DataFrame,
    pair: PairDefinition,
    participant_column: str,
    participants: Sequence[Any],
) -> dict[str, Any]:
    statistics = _participant_sufficient_statistics(
        frame,
        pair.left,
        pair.right,
        participant_column,
        pair_name=pair.name,
        participants=participants,
    )
    estimate, finite_count, status = _correlation_from_statistics(statistics)
    return {"raw_r": estimate, "finite_count": finite_count, "status": status}


def _forbidden_inference_columns(frame: pd.DataFrame) -> list[str]:
    forbidden: list[str] = []
    for column in frame.columns:
        name = str(column)
        lowered = name.lower()
        if re.fullmatch(r"[qs]\d+", lowered) or "oracle" in lowered:
            forbidden.append(name)
    return forbidden


def _concat_tables(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [table for table in tables if not table.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def evaluate_convergent_pairs(
    primitives: pd.DataFrame,
    *,
    participant_column: str = "subject_id",
    day_column: str = "elapsed_day_index",
    date_column: str = "lifelog_date",
    sensor_day_column: str = "sensor_day_id",
    evaluation_start_day: int = EVALUATION_START_DAY,
    bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    shift_draws: int = DEFAULT_SHIFT_DRAWS,
    seed: int = DEFAULT_ROOT_SEED,
    usage_sensitivity_columns: Mapping[str, str] | None = None,
) -> ConvergentValidityResult:
    """Evaluate all four frozen pairs without observed-performance selection."""
    start_day = _validate_positive_integer(evaluation_start_day, "evaluation_start_day")
    bootstrap_count = _validate_positive_integer(bootstrap_draws, "bootstrap_draws")
    shift_count = _validate_positive_integer(shift_draws, "shift_draws")
    usage_slot_ids = {slot[0] for slot in _USAGE_SENSITIVITY_SLOTS}
    frozen_usage_slots = {slot[0] for slot in _USAGE_SENSITIVITY_SLOTS if slot[-1]}
    if usage_sensitivity_columns is not None:
        unknown_usage_definitions = sorted(
            set(usage_sensitivity_columns).difference(usage_slot_ids)
        )
        if unknown_usage_definitions:
            raise ValueError(
                f"Unknown frozen Usage sensitivity definitions: {unknown_usage_definitions}"
            )
        configured_frozen = sorted(set(usage_sensitivity_columns).intersection(frozen_usage_slots))
        if configured_frozen:
            raise ValueError(
                f"The frozen Usage primary slots cannot be configured or replaced: {configured_frozen}"
            )
    forbidden = _forbidden_inference_columns(primitives)
    if forbidden:
        raise ValueError(f"Forbidden label or oracle columns in G1 inference input: {forbidden}")
    required = [participant_column, day_column, date_column, sensor_day_column]
    for pair in PRIMARY_PAIRS:
        required.extend([pair.left, pair.right])
    _require_columns(primitives, list(dict.fromkeys(required)))
    # Validate the full collection calendar once before any inferential result.
    calendar = _validated_calendar(
        primitives,
        participant_column,
        day_column,
        date_column,
        sensor_day_column,
    )
    participants = _ordered_participants(calendar, participant_column)
    evaluation = calendar.loc[calendar[day_column].ge(start_day)].copy()

    row_tables: list[pd.DataFrame] = []
    participant_tables: list[pd.DataFrame] = []
    adjusted_participant_tables: list[pd.DataFrame] = []
    bootstrap_summaries: list[pd.DataFrame] = []
    bootstrap_draw_tables: list[pd.DataFrame] = []
    bootstrap_stat_tables: list[pd.DataFrame] = []
    shift_summaries: list[pd.DataFrame] = []
    shift_draw_tables: list[pd.DataFrame] = []
    shift_assignment_tables: list[pd.DataFrame] = []
    lopo_row_tables: list[pd.DataFrame] = []
    lopo_fold_tables: list[pd.DataFrame] = []
    lopo_coefficient_tables: list[pd.DataFrame] = []
    lopo_summaries: list[pd.DataFrame] = []
    raw_lopo_tables: list[pd.DataFrame] = []
    quality_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []

    for pair in PRIMARY_PAIRS:
        domain = _PAIR_DOMAINS[pair.name]
        mapping = _pair_metadata_mapping(calendar, pair)
        pair_rows = evaluation.loc[
            :,
            [sensor_day_column, participant_column, date_column, day_column, pair.left, pair.right],
        ].copy()
        pair_rows.insert(0, "domain", domain)
        pair_rows.insert(0, "pair", pair.name)
        pair_rows["left_value"] = pd.to_numeric(pair_rows[pair.left], errors="coerce")
        pair_rows["right_value"] = pd.to_numeric(pair_rows[pair.right], errors="coerce")
        pair_rows["pairwise_finite"] = np.isfinite(
            pair_rows["left_value"].to_numpy(dtype=float, na_value=np.nan)
        ) & np.isfinite(pair_rows["right_value"].to_numpy(dtype=float, na_value=np.nan))
        pair_rows["quality_score"] = _quality_score(evaluation, mapping).to_numpy(dtype=float)
        pair_rows["high_coverage_primary"] = pair_rows["quality_score"].ge(
            PRIMARY_HIGH_COVERAGE_THRESHOLD
        )
        for role, source_column in mapping.items():
            pair_rows[role] = _series_or_nan(evaluation, source_column).to_numpy(dtype=float)
            pair_rows[f"{role}_column"] = source_column
        for side, primitive_name in (("left", pair.left), ("right", pair.right)):
            for suffix in ("status", "censoring", "confidence"):
                source_column = f"{primitive_name}__{suffix}"
                output_column = f"{side}_{suffix}"
                pair_rows[output_column] = (
                    evaluation[source_column].to_numpy()
                    if source_column in evaluation.columns
                    else np.nan
                )
        row_tables.append(pair_rows)

        raw = _pair_estimate_record(
            evaluation, pair, participant_column, participants
        )
        participant_table = participant_correlations(
            evaluation,
            pair.left,
            pair.right,
            participant_column,
            pair_name=pair.name,
            participants=participants,
        )
        participant_table.insert(1, "domain", domain)
        participant_tables.append(participant_table)
        finite_spearman = participant_table.loc[
            participant_table["status"].eq("ok"), "spearman_rho"
        ]
        median_spearman = (
            float(finite_spearman.median()) if not finite_spearman.empty else float("nan")
        )
        positive_participants = int(participant_table["direction_positive"].sum())

        bootstrap = cluster_bootstrap_correlation(
            evaluation,
            pair.left,
            pair.right,
            participant_column,
            pair_name=pair.name,
            draws=bootstrap_count,
            seed=seed,
            participants=participants,
        )
        for table in (bootstrap.summary, bootstrap.draws, bootstrap.participant_statistics):
            table.insert(1, "domain", domain)
        bootstrap_summaries.append(bootstrap.summary)
        bootstrap_draw_tables.append(bootstrap.draws)
        bootstrap_stat_tables.append(bootstrap.participant_statistics)

        right_tuple = [
            column
            for column in calendar.columns
            if column == pair.right or column.startswith(f"{pair.right}__")
        ]
        for proxy_role in ("right_coverage", "right_gap"):
            proxy_column = mapping.get(proxy_role)
            if proxy_column is not None:
                right_tuple.append(proxy_column)
        right_tuple = list(dict.fromkeys(right_tuple))
        shift_results: dict[str, CalendarShiftResult] = {}
        for mode in ("unrestricted", "weekday"):
            shift = circular_shift_null(
                calendar,
                pair.left,
                pair.right,
                participant_column,
                day_column,
                date_column=date_column,
                sensor_day_column=sensor_day_column,
                right_tuple_columns=right_tuple,
                pair_name=pair.name,
                mode=mode,
                evaluation_start_day=start_day,
                draws=shift_count,
                seed=seed,
                audit_rows=False,
            )
            for table in (shift.summary, shift.draws, shift.assignments):
                table.insert(1, "domain", domain)
            shift_summaries.append(shift.summary)
            shift_draw_tables.append(shift.draws)
            shift_assignment_tables.append(shift.assignments)
            shift_results[mode] = shift

        raw_lopo = leave_one_participant_out_correlations(
            evaluation,
            pair.left,
            pair.right,
            participant_column,
            pair_name=pair.name,
            participants=participants,
        )
        raw_lopo.insert(1, "domain", domain)
        raw_lopo_tables.append(raw_lopo)
        lopo = lopo_nuisance_adjustment(
            calendar,
            pair.left,
            pair.right,
            participant_column,
            day_column,
            date_column=date_column,
            sensor_day_column=sensor_day_column,
            nuisance_columns=mapping,
            evaluation_start_day=start_day,
            pair_name=pair.name,
        )
        for table in (lopo.summary, lopo.rows, lopo.folds, lopo.coefficients):
            table.insert(1, "domain", domain)
        lopo_summaries.append(lopo.summary)
        lopo_row_tables.append(lopo.rows)
        lopo_fold_tables.append(lopo.folds)
        lopo_coefficient_tables.append(lopo.coefficients)
        adjusted_participant_table = participant_correlations(
            lopo.rows,
            "left_residual",
            "right_residual",
            participant_column,
            pair_name=pair.name,
            participants=participants,
        )
        adjusted_participant_table.insert(1, "domain", domain)
        adjusted_participant_tables.append(adjusted_participant_table)
        finite_adjusted_spearman = adjusted_participant_table.loc[
            adjusted_participant_table["status"].eq("ok"), "spearman_rho"
        ]
        median_adjusted_spearman = (
            float(finite_adjusted_spearman.median())
            if not finite_adjusted_spearman.empty
            else float("nan")
        )
        adjusted_positive_participants = int(
            adjusted_participant_table["direction_positive"].sum()
        )

        quality = _quality_score(evaluation, mapping)
        current_quality_records: list[dict[str, Any]] = []
        for threshold in HIGH_COVERAGE_THRESHOLDS:
            retained = evaluation.loc[quality.ge(threshold)].copy()
            estimate_record = _pair_estimate_record(
                retained, pair, participant_column, participants
            )
            quality_record = {
                    "pair": pair.name,
                    "domain": domain,
                    "threshold": threshold,
                    "is_primary_descriptive_threshold": bool(
                        threshold == PRIMARY_HIGH_COVERAGE_THRESHOLD
                    ),
                    "can_change_frozen_primary": False,
                    "participant_denominator": len(participants),
                    "retained_rows": int(len(retained)),
                    "finite_count": estimate_record["finite_count"],
                    "estimate": estimate_record["raw_r"],
                    "status": estimate_record["status"],
                    "quality_definition": "minimum_left_right_availability_coverage",
                }
            quality_records.append(quality_record)
            current_quality_records.append(quality_record)
        primary_high = next(
            record
            for record in current_quality_records
            if record["threshold"] == PRIMARY_HIGH_COVERAGE_THRESHOLD
        )
        bootstrap_summary = bootstrap.summary.iloc[0]
        unrestricted_summary = shift_results["unrestricted"].summary.iloc[0]
        weekday_summary = shift_results["weekday"].summary.iloc[0]
        lopo_summary = lopo.summary.iloc[0]
        raw_lopo_finite = pd.to_numeric(raw_lopo["estimate"], errors="coerce")
        pair_records.append(
            {
                "pair": pair.name,
                "domain": domain,
                "left": pair.left,
                "right": pair.right,
                "participant_denominator": len(participants),
                "finite_count": raw["finite_count"],
                "raw_r": raw["raw_r"],
                "raw_status": raw["status"],
                "median_participant_spearman": median_spearman,
                "finite_participant_spearman": int(finite_spearman.notna().sum()),
                "positive_participants": positive_participants,
                "bootstrap_ci_low": bootstrap_summary["ci_low"],
                "bootstrap_ci_high": bootstrap_summary["ci_high"],
                "bootstrap_finite_draws": bootstrap_summary["finite_draws"],
                "unrestricted_shift_p": unrestricted_summary["p_value_plus_one"],
                "unrestricted_shift_status": unrestricted_summary["status"],
                "weekday_shift_p": weekday_summary["p_value_plus_one"],
                "weekday_shift_status": weekday_summary["status"],
                "raw_lopo_min": (
                    float(raw_lopo_finite.min()) if raw_lopo_finite.notna().any() else float("nan")
                ),
                "raw_lopo_finite": int(raw_lopo_finite.notna().sum()),
                "adjusted_r": lopo_summary["adjusted_r"],
                "adjusted_finite_count": lopo_summary["finite_count"],
                "adjusted_status": lopo_summary["status"],
                "median_adjusted_participant_spearman": median_adjusted_spearman,
                "finite_adjusted_participant_spearman": int(
                    finite_adjusted_spearman.notna().sum()
                ),
                "adjusted_positive_participants": adjusted_positive_participants,
                "high_coverage_threshold": PRIMARY_HIGH_COVERAGE_THRESHOLD,
                "high_coverage_r": primary_high["estimate"],
                "high_coverage_finite_count": primary_high["finite_count"],
                "high_coverage_status": primary_high["status"],
                "left_coverage_column": mapping["left_coverage"],
                "right_coverage_column": mapping["right_coverage"],
                "left_gap_column": mapping["left_gap"],
                "right_gap_column": mapping["right_gap"],
                "left_support_column": mapping["left_support"],
                "right_support_column": mapping["right_support"],
                "usage_availability_policy": (
                    "matching_screen_proxy" if pair.right.startswith("usage_") else "direct_channel"
                ),
                "bootstrap_draws": bootstrap_count,
                "shift_draws": shift_count,
                "root_seed": int(seed),
            }
        )

    pair_summary = pd.DataFrame.from_records(pair_records)
    pair_summary["holm_shift_p"] = holm_adjust(
        pair_summary["unrestricted_shift_p"].to_numpy(dtype=float)
    )
    pair_summary["r_gate"] = pair_summary["raw_r"].ge(0.30) & pair_summary[
        "raw_status"
    ].eq("ok")
    pair_summary["shift_holm_gate"] = (
        pair_summary["unrestricted_shift_status"].eq("ok")
        & pair_summary["holm_shift_p"].lt(0.05)
    )
    pair_summary["direction_gate"] = (
        pair_summary["participant_denominator"].eq(10)
        & pair_summary["median_participant_spearman"].gt(0.0)
        & pair_summary["positive_participants"].ge(7)
    )
    pair_summary["raw_lopo_gate"] = (
        pair_summary["participant_denominator"].eq(10)
        & pair_summary["raw_lopo_finite"].eq(pair_summary["participant_denominator"])
        & pair_summary["raw_lopo_min"].ge(0.0)
    )
    pair_summary["adjusted_sign_gate"] = (
        pair_summary["participant_denominator"].eq(10)
        & pair_summary["adjusted_status"].eq("ok")
        & pair_summary["adjusted_r"].gt(0.0)
        & pair_summary["median_adjusted_participant_spearman"].gt(0.0)
        & pair_summary["adjusted_positive_participants"].ge(7)
    )
    pair_summary["strong_pair"] = pair_summary[
        ["r_gate", "shift_holm_gate", "direction_gate", "raw_lopo_gate", "adjusted_sign_gate"]
    ].all(axis=1)
    pair_summary["status"] = np.where(
        pair_summary["participant_denominator"].eq(10)
        & pair_summary["raw_status"].eq("ok")
        & pair_summary["unrestricted_shift_status"].eq("ok")
        & pair_summary["weekday_shift_status"].eq("ok"),
        "ok",
        "insufficient_or_degenerate",
    )

    configured_usage_columns = dict(usage_sensitivity_columns or {})
    usage_records: list[dict[str, Any]] = []
    for (
        sensitivity_id,
        endpoint,
        operator,
        pair_name,
        left_column,
        frozen_right_column,
        is_primary,
    ) in _USAGE_SENSITIVITY_SLOTS:
        if is_primary:
            right_column = frozen_right_column
            column_source = "frozen_contract"
        else:
            right_column = configured_usage_columns.get(sensitivity_id)
            column_source = "explicit_config" if right_column is not None else "unconfigured"
        if right_column is None or right_column not in evaluation.columns:
            estimate = float("nan")
            finite_count = 0
            median_spearman = float("nan")
            positive = 0
            estimate_status = "unavailable"
            status = "unavailable"
        else:
            sensitivity_pair = PairDefinition(
                f"{pair_name}__{operator}", left_column, right_column
            )
            estimate_record = _pair_estimate_record(
                evaluation, sensitivity_pair, participant_column, participants
            )
            sensitivity_participants = participant_correlations(
                evaluation,
                left_column,
                right_column,
                participant_column,
                pair_name=sensitivity_pair.name,
                participants=participants,
            )
            finite_rhos = sensitivity_participants.loc[
                sensitivity_participants["status"].eq("ok"), "spearman_rho"
            ]
            estimate = estimate_record["raw_r"]
            finite_count = estimate_record["finite_count"]
            estimate_status = estimate_record["status"]
            median_spearman = (
                float(finite_rhos.median()) if not finite_rhos.empty else float("nan")
            )
            positive = int(sensitivity_participants["direction_positive"].sum())
            status = "available_primary" if is_primary else "available_exploratory"
        usage_records.append(
            {
                "pair": pair_name,
                "domain": "digital",
                "sensitivity_id": sensitivity_id,
                "endpoint": endpoint,
                "operator": operator,
                "left_column": left_column,
                "right_column": right_column,
                "configuration_key": sensitivity_id if not is_primary else None,
                "column_source": column_source,
                "is_frozen_primary": is_primary,
                "can_change_frozen_primary": False,
                "holm_eligible": False,
                "analysis_weighting": "unweighted_value_only",
                "participant_denominator": len(participants),
                "finite_count": finite_count,
                "raw_r": estimate,
                "median_participant_spearman": median_spearman,
                "positive_participants": positive,
                "estimate_status": estimate_status,
                "status": status,
            }
        )
    usage_sensitivity = pd.DataFrame.from_records(usage_records)

    domain_records: list[dict[str, Any]] = []
    for domain in ("digital", "physical", "light"):
        domain_pairs = pair_summary.loc[pair_summary["domain"].eq(domain)]
        claimable = domain_pairs.loc[domain_pairs["strong_pair"], "pair"].tolist()
        domain_records.append(
            {
                "domain": domain,
                "pair_denominator": int(len(domain_pairs)),
                "participant_denominator": len(participants),
                "finite_pairs": int(domain_pairs["raw_status"].eq("ok").sum()),
                "strong_pairs": int(domain_pairs["strong_pair"].sum()),
                "domain_pass": bool(claimable),
                "claimable_endpoints": json.dumps(claimable, ensure_ascii=False),
                "root_seed": int(seed),
                "status": "pass" if claimable else "fail",
            }
        )
    domain_summary = pd.DataFrame.from_records(domain_records)

    decision_records: list[dict[str, Any]] = []
    gate_fields = (
        ("rmcorr_at_least_0.30", "r_gate", "raw_r", ">=", 0.30),
        ("holm_shift_below_0.05", "shift_holm_gate", "holm_shift_p", "<", 0.05),
        ("participant_direction_7_of_10", "direction_gate", "positive_participants", ">=", 7),
        ("raw_lopo_never_reverses", "raw_lopo_gate", "raw_lopo_min", ">=", 0.0),
        (
            "adjusted_participant_direction_7_of_10",
            "adjusted_sign_gate",
            "adjusted_positive_participants",
            ">=",
            7,
        ),
    )
    for row in pair_summary.itertuples(index=False):
        for gate_name, gate_field, observed_field, comparator, threshold in gate_fields:
            decision_records.append(
                {
                    "level": "pair",
                    "pair": row.pair,
                    "domain": row.domain,
                    "gate": gate_name,
                    "observed": getattr(row, observed_field),
                    "comparator": comparator,
                    "threshold": threshold,
                    "passed": bool(getattr(row, gate_field)),
                    "participant_denominator": row.participant_denominator,
                    "finite_count": row.finite_count,
                    "root_seed": int(seed),
                    "bootstrap_draws": bootstrap_count,
                    "shift_draws": shift_count,
                    "status": "pass" if getattr(row, gate_field) else "fail",
                }
            )
        decision_records.append(
            {
                "level": "pair",
                "pair": row.pair,
                "domain": row.domain,
                "gate": "all_frozen_pair_gates",
                "observed": bool(row.strong_pair),
                "comparator": "is",
                "threshold": True,
                "passed": bool(row.strong_pair),
                "participant_denominator": row.participant_denominator,
                "finite_count": row.finite_count,
                "root_seed": int(seed),
                "bootstrap_draws": bootstrap_count,
                "shift_draws": shift_count,
                "status": "pass" if row.strong_pair else "fail",
            }
        )
    for row in domain_summary.itertuples(index=False):
        decision_records.append(
            {
                "level": "domain",
                "pair": np.nan,
                "domain": row.domain,
                "gate": "at_least_one_pre_existing_pair_strong",
                "observed": row.strong_pairs,
                "comparator": ">=",
                "threshold": 1,
                "passed": bool(row.domain_pass),
                "participant_denominator": len(participants),
                "finite_count": np.nan,
                "root_seed": int(seed),
                "bootstrap_draws": bootstrap_count,
                "shift_draws": shift_count,
                "status": row.status,
            }
        )

    return ConvergentValidityResult(
        row_table=_concat_tables(row_tables),
        participant_table=_concat_tables(participant_tables),
        adjusted_participant_table=_concat_tables(adjusted_participant_tables),
        bootstrap_summary=_concat_tables(bootstrap_summaries),
        bootstrap_draws=_concat_tables(bootstrap_draw_tables),
        bootstrap_participant_statistics=_concat_tables(bootstrap_stat_tables),
        shift_summary=_concat_tables(shift_summaries),
        shift_draws=_concat_tables(shift_draw_tables),
        shift_assignments=_concat_tables(shift_assignment_tables),
        lopo_rows=_concat_tables(lopo_row_tables),
        lopo_folds=_concat_tables(lopo_fold_tables),
        lopo_coefficients=_concat_tables(lopo_coefficient_tables),
        lopo_summary=_concat_tables(lopo_summaries),
        raw_lopo=_concat_tables(raw_lopo_tables),
        quality_sensitivity=pd.DataFrame.from_records(quality_records),
        usage_sensitivity=usage_sensitivity,
        pair_summary=pair_summary,
        domain_summary=domain_summary,
        decision_table=pd.DataFrame.from_records(decision_records),
    )
