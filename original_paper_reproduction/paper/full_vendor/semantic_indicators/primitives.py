"""Pure, coverage-aware summaries for daily semantic indicator primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import InitVar, dataclass
import hashlib
import json
from numbers import Number
from os import PathLike
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .contracts import PRIMITIVE_DEFINITIONS, PrimitiveDefinition, validate_label_contract


EVENING_START_MINUTE = 18 * 60
EARLY_HR_END_MINUTE = 20 * 60
LATE_HR_START_MINUTE = 22 * 60
DAY_MINUTES = 24 * 60
OUTPUT_KEY_COLUMNS = ("row_id", "subject_id", "lifelog_date", "sleep_date", "day_index")
INTERVAL_BOUNDARY_POLICIES = (
    "measurement_support",
    "strict_source_availability",
)
SAFE_SCAFFOLD_METADATA = (
    "sensor_day_id",
    "collection_start_date",
    "elapsed_day_index",
    "any_source_record_observed",
    "any_measurement_support_observed",
    "any_sensor_observed",
)
SENSOR_OBSERVATION_FLAGS = (
    "any_source_record_observed",
    "any_measurement_support_observed",
    "any_sensor_observed",
)


FIXED_BUDGET_DEFINITIONS = (
    ("screen_fraction_00_12", "mScreenStatus_canonical.parquet", "screen_on",
     0, 12 * 60, 1, "binary_load"),
    ("screen_fraction_12_18", "mScreenStatus_canonical.parquet", "screen_on",
     12 * 60, 18 * 60, 1, "binary_load"),
    ("screen_fraction_18_24", "mScreenStatus_canonical.parquet", "screen_on",
     18 * 60, 24 * 60, 1, "binary_load"),
    ("activity_fraction_00_12", "mActivity_canonical.parquet",
     "activity_active_mobility", 0, 12 * 60, 1, "activity_load"),
    ("activity_fraction_12_18", "mActivity_canonical.parquet",
     "activity_active_mobility", 12 * 60, 18 * 60, 1, "activity_load"),
    ("activity_fraction_18_24", "mActivity_canonical.parquet",
     "activity_active_mobility", 18 * 60, 24 * 60, 1, "activity_load"),
    ("step_total_00_18", "wPedo_canonical.parquet", "step",
     0, 18 * 60, 1, "additive_load"),
    ("step_total_18_24", "wPedo_canonical.parquet", "step",
     18 * 60, 24 * 60, 1, "additive_load"),
    ("mobile_light_mean_00_18", "mLight_canonical.parquet", "m_light",
     0, 18 * 60, 10, "exposure_mean"),
    ("mobile_light_mean_18_24", "mLight_canonical.parquet", "m_light",
     18 * 60, 24 * 60, 10, "exposure_mean"),
)


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantile: float,
) -> float:
    """Return the inverse weighted empirical CDF using positive finite weights."""
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be finite and between zero and one")

    value_array = np.asarray(values, dtype=float).reshape(-1)
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if value_array.size != weight_array.size:
        raise ValueError("values and weights must have the same length")

    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    if not valid.any():
        return float("nan")

    valid_values = value_array[valid]
    valid_weights = weight_array[valid]
    order = np.argsort(valid_values, kind="stable")
    sorted_values = valid_values[order]
    cumulative_weight = np.cumsum(valid_weights[order])
    threshold = quantile * cumulative_weight[-1]
    index = int(np.searchsorted(cumulative_weight, threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def clip_lifelog_day(
    frame: pd.DataFrame,
    lifelog_date: pd.Timestamp | str,
    sleep_date: pd.Timestamp | str,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Return records in the strict local-clock interval [lifelog day, sleep day)."""
    if timestamp_column not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    start = pd.Timestamp(lifelog_date)
    end = pd.Timestamp(sleep_date)
    if start.tz is not None or end.tz is not None:
        raise ValueError("lifelog boundaries must use timezone-naive local clock timestamps")
    if start != start.normalize() or end != end.normalize():
        raise ValueError("lifelog boundaries must be exact midnight values")
    if end - start != pd.Timedelta(days=1):
        raise ValueError("sleep_date must be exactly one day after lifelog_date")

    result = frame.copy()
    timestamps = pd.to_datetime(result[timestamp_column], errors="coerce")
    if getattr(timestamps.dt, "tz", None) is not None:
        raise ValueError("sensor timestamps must use timezone-naive local clock timestamps")
    result[timestamp_column] = timestamps
    return result.loc[timestamps.ge(start) & timestamps.lt(end)].copy()


def coverage_metadata(
    frame: pd.DataFrame,
    expected_epochs: int,
    valid_mask: Iterable[bool] | None = None,
    timestamp_column: str = "timestamp",
    *,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
    cadence_minutes: int = 1,
    coverage_mode: str = "availability",
) -> dict[str, Any]:
    """Summarize cadence-grid availability or behavioral event support."""
    if isinstance(expected_epochs, bool) or int(expected_epochs) != expected_epochs or expected_epochs <= 0:
        raise ValueError("expected_epochs must be a positive integer")
    if isinstance(cadence_minutes, bool) or int(cadence_minutes) != cadence_minutes or cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be a positive integer")
    if coverage_mode not in {"availability", "event_support"}:
        raise ValueError("coverage_mode must be availability or event_support")
    if (window_start is None) != (window_end is None):
        raise ValueError("window_start and window_end must be provided together")

    if valid_mask is None:
        mask = np.ones(len(frame), dtype=bool)
    else:
        mask = np.asarray(list(valid_mask), dtype=bool).reshape(-1)
        if mask.size != len(frame):
            raise ValueError("valid_mask must have the same length as frame")

    observed_epochs = int(mask.sum())
    start: pd.Timestamp | None = None
    duration_minutes: float | None = None
    observed_indices: np.ndarray | None = None
    if window_start is not None and window_end is not None:
        start = pd.Timestamp(window_start)
        end = pd.Timestamp(window_end)
        if start.tz is not None or end.tz is not None:
            raise ValueError("coverage boundaries must be timezone-naive")
        if end <= start:
            raise ValueError("window_end must be after window_start")
        duration_minutes = (end - start).total_seconds() / 60.0
        nominal_epochs = duration_minutes / int(cadence_minutes)
        if not nominal_epochs.is_integer() or int(nominal_epochs) != int(expected_epochs):
            raise ValueError("expected_epochs must match window duration and cadence")
        observed_indices = np.array([], dtype=int)
        observed_epochs = 0

    if timestamp_column in frame.columns:
        timestamps = pd.to_datetime(frame.loc[mask, timestamp_column], errors="coerce")
        if getattr(timestamps.dt, "tz", None) is not None:
            raise ValueError("coverage timestamps must be timezone-naive local clock values")
        timestamps = timestamps.dropna()
        if start is not None and duration_minutes is not None:
            offsets = (timestamps - start).dt.total_seconds() / 60.0
            in_window = offsets.ge(0) & offsets.lt(duration_minutes)
            observed_indices = np.unique(
                np.floor(offsets.loc[in_window] / int(cadence_minutes)).astype(int)
            )
            observed_epochs = int(len(observed_indices))
        else:
            observed_epochs = int(timestamps.drop_duplicates().size)

    expected = int(expected_epochs)
    longest_gap = float("nan")
    if coverage_mode == "availability" and observed_indices is not None:
        missing = np.ones(expected, dtype=bool)
        missing[observed_indices] = False
        longest_run = current_run = 0
        for is_missing in missing:
            current_run = current_run + 1 if is_missing else 0
            longest_run = max(longest_run, current_run)
        longest_gap = float(longest_run * int(cadence_minutes))

    availability_coverage = (
        min(observed_epochs / expected, 1.0)
        if coverage_mode == "availability"
        else float("nan")
    )
    support_density = (
        min(observed_epochs / expected, 1.0)
        if coverage_mode == "event_support"
        else float("nan")
    )
    return {
        "observed_epochs": observed_epochs,
        "expected_epochs": expected,
        "coverage": availability_coverage,
        "availability_coverage": availability_coverage,
        "longest_gap_minutes": longest_gap,
        "coverage_status": (
            "unknown"
            if coverage_mode == "event_support"
            else ("observed" if observed_epochs else "missing")
        ),
        "support_epochs": observed_epochs if coverage_mode == "event_support" else float("nan"),
        "expected_support_epochs": expected if coverage_mode == "event_support" else float("nan"),
        "support_density": support_density,
        "status": "observed" if observed_epochs else "insufficient",
    }


def _numeric_column(frame: pd.DataFrame, value_column: str) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[value_column], errors="coerce")


def summarize_binary_load(
    frame: pd.DataFrame,
    value_column: str,
    expected_epochs: int,
    *,
    cadence_minutes: int = 1,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return the observed binary-on fraction, without imputing absent epochs as off."""
    values = _numeric_column(frame, value_column)
    valid = values.isin([0, 1])
    quality = coverage_metadata(
        frame, expected_epochs, valid, window_start=window_start, window_end=window_end,
        cadence_minutes=cadence_minutes,
    )
    if not valid.any():
        return float("nan"), quality
    return float(values.loc[valid].mean()), quality


def summarize_binary_active_minutes(
    frame: pd.DataFrame,
    value_column: str,
    expected_epochs: int,
    *,
    epoch_minutes: float = 1.0,
    cadence_minutes: int = 1,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return named observed active minutes without filling missing epochs as inactive."""
    values = _numeric_column(frame, value_column)
    valid = values.isin([0, 1])
    quality = coverage_metadata(
        frame, expected_epochs, valid, window_start=window_start, window_end=window_end,
        cadence_minutes=cadence_minutes,
    )
    if not valid.any():
        return float("nan"), quality
    return float(values.loc[valid].sum() * epoch_minutes), quality


def summarize_activity_load(
    frame: pd.DataFrame,
    expected_epochs: int = 1440,
    active_column: str = "activity_active_mobility",
    unknown_column: str = "activity_unknown",
    *,
    validity_equals: int | float = 0,
    cadence_minutes: int = 1,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return active mobility among known activity epochs only."""
    active = _numeric_column(frame, active_column)
    unknown = _numeric_column(frame, unknown_column)
    valid = active.isin([0, 1]) & unknown.eq(validity_equals)
    quality = coverage_metadata(
        frame, expected_epochs, valid, window_start=window_start, window_end=window_end,
        cadence_minutes=cadence_minutes,
    )
    if not valid.any():
        return float("nan"), quality
    return float(active.loc[valid].mean()), quality


def summarize_additive_load(
    frame: pd.DataFrame,
    value_column: str,
    expected_epochs: int,
    *,
    unit_scale: float = 1.0,
    minimum: float | None = None,
    cadence_minutes: int = 1,
    coverage_mode: str = "availability",
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return the sum of observed nonnegative increments or durations."""
    values = _numeric_column(frame, value_column)
    valid = values.notna() & np.isfinite(values)
    if minimum is not None:
        valid &= values.ge(minimum)
    quality = coverage_metadata(
        frame, expected_epochs, valid, window_start=window_start, window_end=window_end,
        cadence_minutes=cadence_minutes, coverage_mode=coverage_mode,
    )
    if not valid.any():
        return float("nan"), quality
    return float(values.loc[valid].sum() * unit_scale), quality


def summarize_exposure_mean(
    frame: pd.DataFrame,
    value_column: str,
    expected_epochs: int,
    *,
    minimum: float | None = None,
    transform: str | None = None,
    cadence_minutes: int = 1,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return a fixed-transform exposure mean from a raw sensor channel."""
    values = _numeric_column(frame, value_column)
    valid = values.notna() & np.isfinite(values)
    if minimum is not None:
        valid &= values.ge(minimum)
    quality = coverage_metadata(
        frame, expected_epochs, valid, window_start=window_start, window_end=window_end,
        cadence_minutes=cadence_minutes,
    )
    if not valid.any():
        return float("nan"), quality
    observed = values.loc[valid].astype(float)
    if transform == "log1p":
        observed = np.log1p(observed)
    elif transform is not None:
        raise ValueError(f"Unsupported exposure transform: {transform}")
    return float(observed.mean()), quality


def _timestamp_minutes(frame: pd.DataFrame, timestamp_column: str) -> tuple[pd.Series, pd.Series]:
    if timestamp_column not in frame.columns:
        timestamps = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    else:
        timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    minutes = timestamps.dt.hour * 60 + timestamps.dt.minute + timestamps.dt.second / 60.0
    return timestamps, minutes


def _resolve_evening_bounds(
    timestamps: pd.Series,
    window_start: pd.Timestamp | str | None,
    window_end: pd.Timestamp | str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if (window_start is None) != (window_end is None):
        raise ValueError("window_start and window_end must be provided together")
    if window_start is not None and window_end is not None:
        return pd.Timestamp(window_start), pd.Timestamp(window_end)
    observed = timestamps.dropna()
    if observed.empty:
        return None, None
    start = observed.min().normalize() + pd.Timedelta(hours=18)
    return start, start + pd.Timedelta(hours=6)


def summarize_evening_p90(
    frame: pd.DataFrame,
    value_column: str,
    expected_epochs: int = 360,
    timestamp_column: str = "timestamp",
    *,
    validity_column: str | None = None,
    validity_equals: int | float = 0,
    unit_scale: float = 1.0,
    minimum: float = 0.0,
    cadence_minutes: int = 1,
    coverage_mode: str = "availability",
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return the weighted p90 active minute in [18:00, 24:00), with no wrap."""
    timestamps, minutes = _timestamp_minutes(frame, timestamp_column)
    weights = _numeric_column(frame, value_column)
    valid = (
        timestamps.notna()
        & minutes.ge(EVENING_START_MINUTE)
        & minutes.lt(DAY_MINUTES)
        & weights.notna()
        & np.isfinite(weights)
        & weights.ge(minimum)
    )
    if validity_column is not None:
        validity = _numeric_column(frame, validity_column)
        valid &= validity.eq(validity_equals)
    bounds = _resolve_evening_bounds(timestamps, window_start, window_end)
    quality = coverage_metadata(
        frame, expected_epochs, valid, timestamp_column,
        window_start=bounds[0], window_end=bounds[1], cadence_minutes=cadence_minutes,
        coverage_mode=coverage_mode,
    )
    positive = valid & weights.gt(0)
    if not positive.any():
        return float("nan"), quality
    return weighted_quantile(
        minutes.loc[positive], weights.loc[positive] * unit_scale, 0.9
    ), quality


def summarize_low_light_onset(
    frame: pd.DataFrame,
    value_column: str,
    min_observations: int,
    expected_epochs: int = 360,
    timestamp_column: str = "timestamp",
    low_fraction_threshold: float = 0.75,
    *,
    cadence_minutes: int = 1,
    low_threshold: float = 10.0,
    bin_minutes: int = 30,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return an observed high-to-low transition without crossing censored bins."""
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    timestamps, minutes = _timestamp_minutes(frame, timestamp_column)
    values = _numeric_column(frame, value_column)
    valid = (
        timestamps.notna()
        & minutes.ge(EVENING_START_MINUTE)
        & minutes.lt(DAY_MINUTES)
        & values.notna()
        & np.isfinite(values)
        & values.ge(0)
    )
    bounds = _resolve_evening_bounds(timestamps, window_start, window_end)
    quality = coverage_metadata(
        frame, expected_epochs, valid, timestamp_column,
        window_start=bounds[0], window_end=bounds[1], cadence_minutes=cadence_minutes,
    )
    quality.update({"censoring": "none", "onset_status": "not_observed"})
    if not valid.any():
        quality.update({"censoring": "unknown", "onset_status": "insufficient"})
        return float("nan"), quality
    if bounds[0] is None or bounds[1] is None:
        quality.update({"status": "insufficient", "censoring": "unknown",
                        "onset_status": "insufficient"})
        return float("nan"), quality

    observed = pd.DataFrame(
        {
            "window_index": np.floor(
                (timestamps.loc[valid] - bounds[0]).dt.total_seconds()
                / 60.0 / bin_minutes
            ).astype(int),
            "low": values.loc[valid].le(low_threshold).astype(float),
        }
    )
    total_bins = int((bounds[1] - bounds[0]).total_seconds() / 60.0 / bin_minutes)
    windows = (
        observed.groupby("window_index", sort=True)["low"].agg(["count", "mean"])
        .reindex(range(total_bins), fill_value=0)
    )
    assessable = windows["count"].ge(min_observations)
    qualifying = assessable & windows["mean"].ge(low_fraction_threshold)
    qualifying_indices = np.flatnonzero(qualifying.to_numpy())

    if qualifying_indices.size:
        first_qualifying = int(qualifying_indices[0])
        prefix = assessable.iloc[: first_qualifying + 1]
        if first_qualifying == 0 or not bool(prefix.iloc[0]):
            quality.update({"status": "insufficient", "censoring": "left",
                            "onset_status": "left_censored"})
            return float("nan"), quality
        if not bool(prefix.all()):
            quality.update({"status": "insufficient", "censoring": "gap",
                            "onset_status": "gap_censored"})
            return float("nan"), quality
        onset_timestamp = bounds[0] + pd.Timedelta(minutes=first_qualifying * bin_minutes)
        onset = onset_timestamp.hour * 60.0 + onset_timestamp.minute
        quality["onset_status"] = "observed"
        return float(onset), quality

    if not bool(assessable.iloc[0]):
        quality.update({"status": "insufficient", "censoring": "left",
                        "onset_status": "left_censored"})
    elif not bool(assessable.all()):
        first_missing = int(np.flatnonzero(~assessable.to_numpy())[0])
        censoring = "right" if not bool(assessable.iloc[first_missing:].any()) else "gap"
        quality.update({"status": "insufficient", "censoring": censoring,
                        "onset_status": f"{censoring}_censored"})
    return float("nan"), quality


def summarize_hr_settling(
    frame: pd.DataFrame,
    value_column: str = "hr_mean",
    expected_epochs: int = 360,
    timestamp_column: str = "timestamp",
    *,
    component: str = "delta",
    early_start_minute: int = EVENING_START_MINUTE,
    early_end_minute: int = EARLY_HR_END_MINUTE,
    late_start_minute: int = LATE_HR_START_MINUTE,
    late_end_minute: int = DAY_MINUTES,
    min_segment_epochs: int = 30,
    median_bin_minutes: int = 30,
    cadence_minutes: int = 1,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return late-minus-early evening HR and the 30-minute-median slope."""
    segment_bounds = (
        early_start_minute,
        early_end_minute,
        late_start_minute,
        late_end_minute,
    )
    if not all(np.isfinite(bound) for bound in segment_bounds) or not (
        EVENING_START_MINUTE
        <= early_start_minute
        < early_end_minute
        <= late_start_minute
        < late_end_minute
        <= DAY_MINUTES
    ):
        raise ValueError(
            "HR segment bounds must be ordered, non-overlapping, and inside the evening window"
        )
    if (
        isinstance(min_segment_epochs, bool)
        or int(min_segment_epochs) != min_segment_epochs
        or min_segment_epochs <= 0
    ):
        raise ValueError("min_segment_epochs must be a positive integer")
    if (
        isinstance(median_bin_minutes, bool)
        or int(median_bin_minutes) != median_bin_minutes
        or median_bin_minutes <= 0
    ):
        raise ValueError("median_bin_minutes must be a positive integer")
    timestamps, minutes = _timestamp_minutes(frame, timestamp_column)
    values = _numeric_column(frame, value_column)
    valid = (
        timestamps.notna()
        & minutes.ge(EVENING_START_MINUTE)
        & minutes.lt(DAY_MINUTES)
        & values.notna()
        & np.isfinite(values)
    )
    if component not in {"delta", "slope"}:
        raise ValueError("component must be delta or slope")
    bounds = _resolve_evening_bounds(timestamps, window_start, window_end)
    quality = coverage_metadata(
        frame, expected_epochs, valid, timestamp_column,
        window_start=bounds[0], window_end=bounds[1], cadence_minutes=cadence_minutes,
    )
    quality["median_slope_per_hour"] = float("nan")
    quality["heart_rate_settling_delta"] = float("nan")
    quality["heart_rate_settling_slope"] = float("nan")
    quality["early_observed_epochs"] = 0
    quality["late_observed_epochs"] = 0
    if not valid.any():
        return float("nan"), quality

    early = valid & minutes.ge(early_start_minute) & minutes.lt(early_end_minute)
    late = valid & minutes.ge(late_start_minute) & minutes.lt(late_end_minute)
    quality["early_observed_epochs"] = int(
        timestamps.loc[early].dt.floor("min").nunique()
    )
    quality["late_observed_epochs"] = int(
        timestamps.loc[late].dt.floor("min").nunique()
    )
    if (
        quality["early_observed_epochs"] < min_segment_epochs
        or quality["late_observed_epochs"] < min_segment_epochs
    ):
        quality["status"] = "insufficient"
        return float("nan"), quality

    observed = pd.DataFrame(
        {
            "window_start": np.floor(minutes.loc[valid] / median_bin_minutes)
            * median_bin_minutes,
            "heart_rate": values.loc[valid].astype(float),
        }
    )
    medians = observed.groupby("window_start", sort=True)["heart_rate"].median()
    if len(medians) >= 2:
        x = medians.index.to_numpy(dtype=float) / 60.0
        y = medians.to_numpy(dtype=float)
        centered_x = x - x.mean()
        denominator = float(np.dot(centered_x, centered_x))
        if denominator > 0:
            quality["median_slope_per_hour"] = float(
                np.dot(centered_x, y - y.mean()) / denominator
            )

    quality["early_mean"] = float(values.loc[early].mean())
    quality["late_mean"] = float(values.loc[late].mean())
    quality["heart_rate_settling_delta"] = quality["late_mean"] - quality["early_mean"]
    quality["heart_rate_settling_slope"] = quality["median_slope_per_hour"]
    return float(quality[f"heart_rate_settling_{component}"]), quality


def summarize_charging_onset(
    frame: pd.DataFrame,
    value_column: str = "charging_flag",
    expected_epochs: int = 360,
    timestamp_column: str = "timestamp",
    min_duration_minutes: int = 10,
    *,
    require_off_to_on: bool = True,
    cadence_minutes: int = 1,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return an exact first onset only when its cadence-grid prefix is assessable."""
    if min_duration_minutes <= 0:
        raise ValueError("min_duration_minutes must be positive")
    if not isinstance(require_off_to_on, (bool, np.bool_)):
        raise ValueError("require_off_to_on must be boolean")
    timestamps, minutes = _timestamp_minutes(frame, timestamp_column)
    values = _numeric_column(frame, value_column)
    valid = (
        timestamps.notna()
        & minutes.ge(EVENING_START_MINUTE)
        & minutes.lt(DAY_MINUTES)
        & values.isin([0, 1])
    )
    bounds = _resolve_evening_bounds(timestamps, window_start, window_end)
    quality = coverage_metadata(
        frame, expected_epochs, valid, timestamp_column,
        window_start=bounds[0], window_end=bounds[1], cadence_minutes=cadence_minutes,
    )
    quality.update({"censoring": "none", "onset_status": "not_observed"})
    if bounds[0] is None or bounds[1] is None:
        quality.update({"censoring": "unknown", "onset_status": "insufficient"})
        return float("nan"), quality

    cadence = int(cadence_minutes)
    required_epochs = int(np.ceil(float(min_duration_minutes) / cadence))
    epochs = pd.DataFrame(
        {
            "timestamp": timestamps.loc[valid],
            "charging": values.loc[valid].astype(int),
        }
    )
    if not epochs.empty:
        offsets = (epochs["timestamp"] - bounds[0]).dt.total_seconds() / 60.0
        epochs["grid_index"] = np.floor(offsets / cadence).astype(int)
        epochs = epochs.loc[
            epochs["grid_index"].ge(0) & epochs["grid_index"].lt(expected_epochs)
        ]
        epochs = epochs.groupby("grid_index", sort=True)["charging"].max()

    states = np.full(int(expected_epochs), np.nan)
    if not epochs.empty:
        states[epochs.index.to_numpy(dtype=int)] = epochs.to_numpy(dtype=float)
    missing = np.isnan(states)
    missing_indices = np.flatnonzero(missing)
    assessable_prefix_epochs = (
        int(missing_indices[0]) if missing_indices.size else int(expected_epochs)
    )

    if (
        require_off_to_on
        and assessable_prefix_epochs > 0
        and states[0] == 1
    ):
        quality.update(
            {"status": "insufficient", "censoring": "left", "onset_status": "left_censored"}
        )
        return float("nan"), quality

    right_truncated_candidate = False
    for onset_index in range(assessable_prefix_epochs):
        if states[onset_index] != 1:
            continue
        if require_off_to_on:
            if onset_index == 0 or states[onset_index - 1] != 0:
                continue
        elif onset_index > 0 and states[onset_index - 1] == 1:
            continue

        proof_end = onset_index + required_epochs
        observed_proof_end = min(proof_end, assessable_prefix_epochs)
        proof_is_charging = bool(np.all(states[onset_index:observed_proof_end] == 1))
        if proof_end <= assessable_prefix_epochs and proof_is_charging:
            onset_timestamp = bounds[0] + pd.Timedelta(minutes=onset_index * cadence)
            onset = (
                onset_timestamp.hour * 60.0
                + onset_timestamp.minute
                + onset_timestamp.second / 60.0
            )
            quality["onset_status"] = "observed"
            return float(onset), quality
        if proof_end > assessable_prefix_epochs and proof_is_charging:
            right_truncated_candidate = True

    if missing_indices.size:
        first_missing = int(missing_indices[0])
        if first_missing == 0:
            censoring = "left"
        elif bool(missing[first_missing:].all()):
            censoring = "right"
        else:
            censoring = "gap"
        quality.update(
            {
                "status": "insufficient",
                "censoring": censoring,
                "onset_status": f"{censoring}_censored",
            }
        )
    elif right_truncated_candidate:
        quality.update(
            {"status": "insufficient", "censoring": "right", "onset_status": "right_censored"}
        )
    return float("nan"), quality


def _load_label_scaffold(
    labels: pd.DataFrame | str | PathLike[str],
) -> pd.DataFrame:
    """Return only immutable day keys from a labeled or unlabeled scaffold."""
    if isinstance(labels, pd.DataFrame):
        source = labels.copy()
    else:
        source = pd.read_csv(Path(labels))

    if source.columns.duplicated().any():
        duplicates = source.columns[source.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate input columns: {duplicates}")

    # These are deterministic products of validation, not caller-controlled features.
    source = source.drop(columns=["row_id", "day_index"], errors="ignore")
    validated = validate_label_contract(source)
    validated = _validate_sensor_scaffold_metadata(validated)
    metadata = [
        column for column in SAFE_SCAFFOLD_METADATA if column in validated.columns
    ]
    return validated.loc[:, [*OUTPUT_KEY_COLUMNS, *metadata]].copy()


def _finite_integer_values(
    values: pd.Series,
    name: str,
    *,
    minimum: int,
) -> pd.Series:
    valid_types = values.map(
        lambda value: isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
    )
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float, na_value=np.nan)
    valid = (
        valid_types.to_numpy(dtype=bool)
        & np.isfinite(array)
        & np.equal(array, np.floor(array))
        & (array >= minimum)
    )
    if not bool(valid.all()):
        qualifier = "finite nonnegative integers" if minimum == 0 else "finite integers >= 1"
        raise ValueError(f"{name} must contain {qualifier}")
    return pd.Series(array.astype(np.int64), index=values.index, name=name)


def _validate_sensor_scaffold_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the all-or-none leakage-safe collection-clock mapping."""
    present = [column for column in SAFE_SCAFFOLD_METADATA if column in frame.columns]
    if present and len(present) != len(SAFE_SCAFFOLD_METADATA):
        raise ValueError(
            "Sensor calendar metadata must provide all six columns: "
            f"{list(SAFE_SCAFFOLD_METADATA)}"
        )
    if not present:
        return frame

    result = frame.copy()
    result["sensor_day_id"] = _finite_integer_values(
        result["sensor_day_id"], "sensor_day_id", minimum=0
    )
    if result["sensor_day_id"].duplicated().any():
        raise ValueError("sensor_day_id values must be unique")

    parsed_start: list[pd.Timestamp] = []
    for value in result["collection_start_date"]:
        if isinstance(value, Number):
            raise ValueError(
                "collection_start_date must contain date-like timezone-naive "
                "midnight values, not numeric values"
            )
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "collection_start_date must contain exact timezone-naive midnight values"
            ) from error
        if (
            pd.isna(timestamp)
            or timestamp.tz is not None
            or timestamp != timestamp.normalize()
        ):
            raise ValueError(
                "collection_start_date must contain exact timezone-naive midnight values"
            )
        parsed_start.append(timestamp)
    result["collection_start_date"] = pd.Series(
        parsed_start, index=result.index
    )
    starts_per_subject = result.groupby("subject_id", dropna=False)[
        "collection_start_date"
    ].nunique(dropna=False)
    if starts_per_subject.gt(1).any():
        raise ValueError("collection_start_date must be constant per participant")
    if result["collection_start_date"].gt(result["lifelog_date"]).any():
        raise ValueError("collection_start_date cannot be after lifelog_date")

    result["elapsed_day_index"] = _finite_integer_values(
        result["elapsed_day_index"], "elapsed_day_index", minimum=1
    )
    expected_elapsed = (
        result["lifelog_date"] - result["collection_start_date"]
    ).dt.days + 1
    if not result["elapsed_day_index"].eq(expected_elapsed).all():
        raise ValueError("elapsed_day_index must exactly match the collection clock")

    for column in SENSOR_OBSERVATION_FLAGS:
        boolean_values = result[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        )
        if not bool(boolean_values.all()):
            raise ValueError(f"{column} must contain true boolean values")
        result[column] = result[column].astype(bool)
    if not result["any_sensor_observed"].equals(
        result["any_measurement_support_observed"]
    ):
        raise ValueError(
            "any_sensor_observed must be the exact measurement-support alias"
        )
    return result


def _definitions_by_sensor_file() -> dict[str, list[PrimitiveDefinition]]:
    definitions_by_file: dict[str, list[PrimitiveDefinition]] = {}
    for definition in PRIMITIVE_DEFINITIONS:
        definitions_by_file.setdefault(definition.sensor_file, []).append(definition)
    return definitions_by_file


def _temporal_contract(
    definitions: Sequence[PrimitiveDefinition],
    sensor_file: str,
) -> tuple[str, int, str]:
    contracts = {
        (
            definition.timestamp_semantics,
            definition.support_minutes,
            definition.timing_anchor,
        )
        for definition in definitions
    }
    if len(contracts) != 1:
        raise ValueError(f"Conflicting timestamp contracts for {sensor_file}")
    return contracts.pop()


def _parse_naive_sensor_timestamps(
    values: pd.Series,
    sensor_file: str,
) -> pd.Series:
    """Coerce invalid timestamps only after rejecting every timezone-aware value."""
    parsed: list[pd.Timestamp | pd.NaT] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            parsed.append(pd.NaT)
            continue
        if pd.isna(timestamp):
            parsed.append(pd.NaT)
            continue
        if timestamp.tz is not None:
            raise ValueError(
                f"{sensor_file} timestamps must be timezone-naive local clock values"
            )
        parsed.append(timestamp)
    return pd.to_datetime(
        pd.Series(parsed, index=values.index, name=values.name),
        errors="coerce",
    )


def _read_canonical_sensor(
    canonical_root: Path,
    sensor_file: str,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    sensor_path = canonical_root / sensor_file
    if not sensor_path.is_file():
        raise FileNotFoundError(f"Missing canonical sensor file: {sensor_path}")
    columns = list(dict.fromkeys(["subject_id", "timestamp", *required_columns]))
    sensor = pd.read_parquet(sensor_path, columns=columns)
    sensor["source_timestamp"] = _parse_naive_sensor_timestamps(
        sensor["timestamp"], sensor_file
    )
    return sensor


def _sensor_observation_days(
    sensor: pd.DataFrame,
    *,
    timestamp_semantics: str,
    support_minutes: int,
    timing_anchor: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = sensor["source_timestamp"]
    source_observed = pd.DataFrame(
        {
            "subject_id": sensor["subject_id"],
            "lifelog_date": timestamps.dt.normalize(),
        }
    ).dropna(subset=["subject_id", "lifelog_date"])
    source_observed = source_observed.drop_duplicates()
    if timestamp_semantics == "instant":
        if support_minutes != 0 or timing_anchor != "source_timestamp":
            raise ValueError(
                "Instant sensors require zero support and source timestamp anchor"
            )
        return source_observed, source_observed.copy()
    if timestamp_semantics != "interval_end":
        raise ValueError(f"Unsupported timestamp semantics: {timestamp_semantics}")
    if support_minutes <= 0 or timing_anchor != "support_midpoint":
        raise ValueError(
            "Interval-end sensors require positive support and support midpoint anchor"
        )
    support_end = timestamps
    support_start = support_end - pd.Timedelta(minutes=int(support_minutes))
    support_date = (support_end - pd.Timedelta(microseconds=1)).dt.normalize()
    fully_contained = support_start.ge(support_date) & support_end.le(
        support_date + pd.Timedelta(days=1)
    )
    support_observed = pd.DataFrame(
        {
            "subject_id": sensor.loc[fully_contained, "subject_id"],
            "lifelog_date": support_date.loc[fully_contained],
        }
    ).dropna(subset=["subject_id", "lifelog_date"])
    return source_observed, support_observed.drop_duplicates()


def _build_sensor_calendar_from_observations(
    source_observed_parts: Sequence[pd.DataFrame],
    support_observed_parts: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    if not source_observed_parts:
        raise ValueError("No frozen sensor files are defined")
    source_observed_days = (
        pd.concat(source_observed_parts, ignore_index=True)
        .drop_duplicates()
        .sort_values(["subject_id", "lifelog_date"], kind="stable")
        .reset_index(drop=True)
    )
    if source_observed_days.empty:
        raise ValueError("Frozen sensor files contain no valid participant timestamps")
    support_observed_days = (
        pd.concat(support_observed_parts, ignore_index=True)
        .drop_duplicates()
        .sort_values(["subject_id", "lifelog_date"], kind="stable")
        .reset_index(drop=True)
    )

    calendars: list[pd.DataFrame] = []
    for subject_id, subject_days in source_observed_days.groupby(
        "subject_id", sort=True
    ):
        collection_start = subject_days["lifelog_date"].min()
        collection_end = subject_days["lifelog_date"].max()
        dates = pd.date_range(collection_start, collection_end, freq="D")
        calendars.append(
            pd.DataFrame(
                {
                    "subject_id": subject_id,
                    "lifelog_date": dates,
                    "collection_start_date": collection_start,
                    "elapsed_day_index": np.arange(1, len(dates) + 1, dtype=int),
                }
            )
        )

    scaffold = pd.concat(calendars, ignore_index=True)
    source_index = pd.MultiIndex.from_frame(
        source_observed_days[["subject_id", "lifelog_date"]]
    )
    support_index = pd.MultiIndex.from_frame(
        support_observed_days[["subject_id", "lifelog_date"]]
    )
    scaffold_index = pd.MultiIndex.from_frame(
        scaffold[["subject_id", "lifelog_date"]]
    )
    scaffold["any_source_record_observed"] = scaffold_index.isin(source_index)
    scaffold["any_measurement_support_observed"] = scaffold_index.isin(
        support_index
    )
    scaffold["any_sensor_observed"] = scaffold[
        "any_measurement_support_observed"
    ].copy()
    scaffold["sleep_date"] = scaffold["lifelog_date"] + pd.Timedelta(days=1)
    scaffold["sensor_day_id"] = np.arange(len(scaffold), dtype=int)
    return scaffold.loc[
        :,
        [
            "subject_id",
            "lifelog_date",
            "sleep_date",
            *SAFE_SCAFFOLD_METADATA,
        ],
    ]


def _validate_sensor_metadata_provenance(
    scaffold: pd.DataFrame,
    canonical_scaffold: pd.DataFrame,
) -> None:
    if not all(column in scaffold.columns for column in SAFE_SCAFFOLD_METADATA):
        return
    keys = ["subject_id", "lifelog_date", "sleep_date"]
    supplied = scaffold.loc[:, [*keys, *SAFE_SCAFFOLD_METADATA]]
    expected = supplied.merge(
        canonical_scaffold,
        on=keys,
        how="left",
        suffixes=("", "__canonical"),
        indicator=True,
        validate="one_to_one",
    )
    if not expected["_merge"].eq("both").all():
        raise ValueError(
            "Supplied sensor metadata contains a non-canonical participant-day"
        )
    for column in SAFE_SCAFFOLD_METADATA:
        canonical_column = f"{column}__canonical"
        if not expected[column].eq(expected[canonical_column]).all():
            raise ValueError(
                f"Supplied sensor metadata does not match canonical {column}"
            )


def build_sensor_calendar_scaffold(
    canonical_root: str | PathLike[str],
) -> pd.DataFrame:
    """Build source-anchored calendars with explicit source/support observation flags.

    ``any_sensor_observed`` is an exact backward-compatible alias of
    ``any_measurement_support_observed``. Interval-end measurements are assigned
    only when their complete support is contained in a calendar day; support before
    the earliest canonical source date never expands the participant calendar.
    """
    root = Path(canonical_root)
    source_parts: list[pd.DataFrame] = []
    support_parts: list[pd.DataFrame] = []
    for sensor_file, definitions in _definitions_by_sensor_file().items():
        sensor = _read_canonical_sensor(root, sensor_file, ())
        timestamp_semantics, support_minutes, timing_anchor = _temporal_contract(
            definitions, sensor_file
        )
        source_days, support_days = _sensor_observation_days(
            sensor,
            timestamp_semantics=timestamp_semantics,
            support_minutes=support_minutes,
            timing_anchor=timing_anchor,
        )
        source_parts.append(source_days)
        support_parts.append(support_days)
    return _build_sensor_calendar_from_observations(source_parts, support_parts)


def _align_sensor(
    sensor: pd.DataFrame,
    sensor_file: str,
    scaffold: pd.DataFrame,
    *,
    timestamp_semantics: str,
    support_minutes: int,
    timing_anchor: str,
) -> pd.DataFrame:
    """Align one already-loaded canonical sensor frame to scaffold days."""
    sensor = sensor.loc[sensor["source_timestamp"].notna()].copy()
    if timestamp_semantics == "instant":
        if support_minutes != 0 or timing_anchor != "source_timestamp":
            raise ValueError("Instant sensors require zero support and source timestamp anchor")
        sensor["support_start"] = sensor["source_timestamp"]
        sensor["support_end"] = sensor["source_timestamp"]
        sensor["effective_timestamp"] = sensor["source_timestamp"]
        sensor["_lifelog_date"] = sensor["source_timestamp"].dt.normalize()
    elif timestamp_semantics == "interval_end":
        if support_minutes <= 0 or timing_anchor != "support_midpoint":
            raise ValueError(
                "Interval-end sensors require positive support and support midpoint anchor"
            )
        support = pd.Timedelta(minutes=int(support_minutes))
        sensor["support_start"] = sensor["source_timestamp"] - support
        sensor["support_end"] = sensor["source_timestamp"]
        sensor["effective_timestamp"] = sensor["source_timestamp"] - support / 2
        sensor["_lifelog_date"] = (
            sensor["support_end"] - pd.Timedelta(microseconds=1)
        ).dt.normalize()
    else:
        raise ValueError(f"Unsupported timestamp semantics: {timestamp_semantics}")
    sensor["timestamp"] = sensor["effective_timestamp"]

    day_keys = scaffold.loc[
        :, ["row_id", "subject_id", "lifelog_date", "sleep_date"]
    ]
    aligned = sensor.merge(
        day_keys,
        how="inner",
        left_on=["subject_id", "_lifelog_date"],
        right_on=["subject_id", "lifelog_date"],
        sort=False,
        validate="many_to_one",
    )
    if timestamp_semantics == "instant":
        inside_landmark = (
            aligned["source_timestamp"].ge(aligned["lifelog_date"])
            & aligned["source_timestamp"].lt(aligned["sleep_date"])
        )
    else:
        inside_landmark = (
            aligned["support_start"].ge(aligned["lifelog_date"])
            & aligned["support_end"].le(aligned["sleep_date"])
        )
    return aligned.loc[inside_landmark].drop(columns="_lifelog_date").reset_index(drop=True)


def _window_bounds(
    lifelog_date: pd.Timestamp,
    sleep_date: pd.Timestamp,
    window: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if window == "day":
        return lifelog_date, sleep_date
    if window == "evening":
        return lifelog_date + pd.Timedelta(hours=18), sleep_date
    raise ValueError(f"Unsupported primitive window: {window}")


def _dispatch_primitive(
    definition: PrimitiveDefinition,
    day_frame: pd.DataFrame,
    lifelog_date: pd.Timestamp,
    sleep_date: pd.Timestamp,
    interval_boundary_policy: str,
) -> tuple[float, dict[str, Any]]:
    """Bind a frozen primitive definition to its pure summarizer."""
    window_start, window_end = _window_bounds(
        lifelog_date, sleep_date, definition.window
    )
    if definition.timestamp_semantics == "instant":
        source_timestamps = day_frame["source_timestamp"]
        support_window_frame = day_frame.loc[
            source_timestamps.ge(window_start) & source_timestamps.lt(window_end)
        ]
        window_frame = support_window_frame
    else:
        support_window_frame = day_frame.loc[
            day_frame["support_start"].ge(window_start)
            & day_frame["support_end"].le(window_end)
        ]
        if interval_boundary_policy == "strict_source_availability":
            window_frame = support_window_frame.loc[
                support_window_frame["source_timestamp"].lt(window_end)
            ]
        else:
            window_frame = support_window_frame
    params = dict(definition.operation_params)
    common = {
        "cadence_minutes": definition.cadence_minutes,
        "window_start": window_start,
        "window_end": window_end,
    }

    if definition.operation == "binary_load":
        value, quality = summarize_binary_load(
            window_frame,
            definition.value_column,
            definition.expected_epochs,
            **common,
        )
        value = value * definition.unit_scale
    elif definition.operation == "binary_active_minutes":
        value, quality = summarize_binary_active_minutes(
            window_frame,
            definition.value_column,
            definition.expected_epochs,
            **params,
            **common,
        )
        value = value * definition.unit_scale
    elif definition.operation == "activity_load":
        if definition.validity_column is None:
            raise ValueError(f"{definition.name} requires a validity column")
        value, quality = summarize_activity_load(
            window_frame,
            definition.expected_epochs,
            definition.value_column,
            definition.validity_column,
            **params,
            **common,
        )
        value = value * definition.unit_scale
    elif definition.operation == "additive_load":
        value, quality = summarize_additive_load(
            window_frame,
            definition.value_column,
            definition.expected_epochs,
            unit_scale=definition.unit_scale,
            coverage_mode=definition.coverage_mode,
            **params,
            **common,
        )
    elif definition.operation == "exposure_mean":
        value, quality = summarize_exposure_mean(
            window_frame,
            definition.value_column,
            definition.expected_epochs,
            **params,
            **common,
        )
        value = value * definition.unit_scale
    elif definition.operation == "evening_p90":
        value, quality = summarize_evening_p90(
            window_frame,
            definition.value_column,
            definition.expected_epochs,
            validity_column=definition.validity_column,
            unit_scale=definition.unit_scale,
            coverage_mode=definition.coverage_mode,
            **params,
            **common,
        )
    elif definition.operation == "low_light_onset":
        value, quality = summarize_low_light_onset(
            window_frame,
            definition.value_column,
            expected_epochs=definition.expected_epochs,
            **params,
            **common,
        )
        value = value * definition.unit_scale
    elif definition.operation == "heart_rate_settling":
        value, quality = summarize_hr_settling(
            window_frame,
            definition.value_column,
            definition.expected_epochs,
            **params,
            **common,
        )
        value = value * definition.unit_scale
    elif definition.operation == "charging_onset":
        value, quality = summarize_charging_onset(
            window_frame,
            definition.value_column,
            definition.expected_epochs,
            **params,
            **common,
        )
        value = value * definition.unit_scale
    else:
        raise ValueError(f"Unsupported primitive operation: {definition.operation}")

    if definition.timestamp_semantics == "interval_end":
        source_boundary = support_window_frame["source_timestamp"].eq(window_end)
        crossing_after = (
            day_frame["support_start"].lt(window_end)
            & day_frame["support_end"].gt(window_end)
        )
        quality.update(
            {
                "timestamp_semantics": definition.timestamp_semantics,
                "support_minutes": definition.support_minutes,
                "timing_anchor": definition.timing_anchor,
                "interval_boundary_policy": interval_boundary_policy,
                "source_window_start_boundary_epochs": int(
                    day_frame["source_timestamp"].eq(window_start).sum()
                ),
                "source_window_end_boundary_epochs": int(source_boundary.sum()),
                "strict_source_boundary_excluded_epochs": int(
                    source_boundary.sum()
                    if interval_boundary_policy == "strict_source_availability"
                    else 0
                ),
                "support_extends_after_window_epochs": int(crossing_after.sum()),
                "source_timestamp_min": (
                    window_frame["source_timestamp"].min()
                    if not window_frame.empty else pd.NaT
                ),
                "source_timestamp_max": (
                    window_frame["source_timestamp"].max()
                    if not window_frame.empty else pd.NaT
                ),
                "effective_timestamp_min": (
                    window_frame["effective_timestamp"].min()
                    if not window_frame.empty else pd.NaT
                ),
                "effective_timestamp_max": (
                    window_frame["effective_timestamp"].max()
                    if not window_frame.empty else pd.NaT
                ),
            }
        )

    # A wholly absent sensor-day is unknown, not 0% observed. A present day with
    # an empty/invalid subwindow retains finite zero availability for that window.
    # Event-support streams deliberately retain a support density of zero.
    if int(quality.get("observed_epochs", 0)) == 0:
        quality["status"] = "insufficient"
        if definition.coverage_mode == "availability" and day_frame.empty:
            quality["coverage"] = float("nan")
            quality["availability_coverage"] = float("nan")
            quality["longest_gap_minutes"] = float("nan")
            quality["coverage_status"] = "missing"
            if "censoring" in quality:
                quality["censoring"] = "unknown"
            if "onset_status" in quality:
                quality["onset_status"] = "insufficient"
    return float(value), quality


def _forbidden_replay_columns(columns: Iterable[object]) -> list[object]:
    """Return outcome/oracle bait names using one case-insensitive policy."""
    forbidden_tokens = (
        "outcome",
        "target",
        "label",
        "oracle",
        "predict",
        "classifier",
    )
    return [
        column
        for column in columns
        if re.fullmatch(r"[qs]\d+", str(column).casefold())
        or any(token in str(column).casefold() for token in forbidden_tokens)
    ]


_PREPARED_PHASE_SCHEMA = "prepared-phase-view-v1"
_PREPARED_RECORD_SCHEMA = "prepared-phase-record-v1"
_PREPARED_CALENDAR_SCHEMA = "prepared-calendar-identity-v1"
_PREPARED_MASK_INTERVAL_SCHEMA = "prepared-mask-interval-v1"
_PREPARED_MASK_BIN_SCHEMA = "prepared-mask-bin-geometry-v1"
_PREPARED_MASK_GEOMETRY_SCHEMA = "prepared-mask-geometry-v1"

_PREPARED_CALENDAR_STRING_FIELDS = ("subject_id", "day_type")
_PREPARED_CALENDAR_INTEGER_FIELDS = ("sensor_day_id", "elapsed_day_index")
_PREPARED_CALENDAR_TIMESTAMP_FIELDS = (
    "lifelog_date",
    "sleep_date",
    "collection_start_date",
)
_PREPARED_CALENDAR_DIGEST_FIELDS = ("schema_digest", "content_digest")

_PREPARED_RECORD_STRING_FIELDS = ("subject_id",)
_PREPARED_RECORD_INTEGER_FIELDS = (
    "record_id",
    "phase_bin",
    "positive_event_count",
)
_PREPARED_RECORD_TIMESTAMP_FIELDS = (
    "source_timestamp",
    "support_start",
    "support_end",
    "effective_timestamp",
)
_PREPARED_RECORD_FLOAT_FIELDS = (
    "raw_value",
    "support_units",
    "transformed_numerator",
    "event_mass",
)
_PREPARED_RECORD_OPTIONAL_FLOAT_FIELDS = ("raw_validity", "event_minute")
_PREPARED_RECORD_BOOLEAN_FIELDS = ("valid",)
_PREPARED_RECORD_DIGEST_FIELDS = ("content_digest",)

_PREPARED_VIEW_STRING_FIELDS = (
    "primitive_name",
    "sensor_file",
    "value_column",
    "operation",
    "window",
    "coverage_mode",
    "timestamp_semantics",
    "timing_anchor",
    "unit",
    "interval_boundary_policy",
)
_PREPARED_VIEW_OPTIONAL_STRING_FIELDS = ("validity_column",)
_PREPARED_VIEW_INTEGER_FIELDS = (
    "cadence_minutes",
    "expected_epochs",
    "support_minutes",
)
_PREPARED_VIEW_FLOAT_FIELDS = ("unit_scale",)
_PREPARED_VIEW_DIGEST_FIELDS = (
    "definition_digest",
    "policy_digest",
    "source_schema_digest",
    "aligned_schema_digest",
    "aligned_frame_digest",
    "source_record_digest",
    "schema_digest",
    "content_digest",
)
_PREPARED_VIEW_NESTED_FIELDS = (
    "calendar_identity",
    "operation_params",
    "aligned_column_topology",
    "records",
)

_PREPARED_MASK_INTERVAL_INTEGER_FIELDS = ("mask_interval_id",)
_PREPARED_MASK_INTERVAL_TIMESTAMP_FIELDS = ("mask_start", "mask_end")
_PREPARED_MASK_INTERVAL_FLOAT_FIELDS = ("duration_minutes",)
_PREPARED_MASK_INTERVAL_DIGEST_FIELDS = ("content_digest",)

_PREPARED_MASK_BIN_INTEGER_FIELDS = ("phase_bin",)
_PREPARED_MASK_BIN_FLOAT_FIELDS = (
    "expected_units",
    "baseline_units",
    "observed_units",
    "induced_deleted_units",
    "masked_fraction",
)
_PREPARED_MASK_BIN_DIGEST_FIELDS = ("content_digest",)

_PREPARED_MASK_GEOMETRY_STRING_FIELDS = (
    "primitive_name",
    "operation",
    "scenario",
    "root_seed",
)
_PREPARED_MASK_GEOMETRY_INTEGER_FIELDS = ("draw",)
_PREPARED_MASK_GEOMETRY_FLOAT_FIELDS = (
    "critical_window_expected_units",
    "critical_window_baseline_units",
    "critical_window_observed_units",
    "critical_window_induced_deleted_units",
    "critical_window_observed_fraction",
)
_PREPARED_MASK_GEOMETRY_DIGEST_FIELDS = (
    "phase_view_digest",
    "calendar_digest",
    "source_record_digest",
    "definition_digest",
    "policy_digest",
    "seed",
    "mask_key_digest",
    "mask_digest",
    "schema_digest",
    "content_digest",
)
_PREPARED_MASK_GEOMETRY_NESTED_FIELDS = (
    "calendar_identity",
    "intervals",
    "deleted_record_ids",
    "bins",
)

_PREPARED_DEFINITION_STRING_FIELDS = (
    "name",
    "sensor_file",
    "value_column",
    "window",
    "coverage_mode",
    "timestamp_semantics",
    "timing_anchor",
    "unit",
    "operation",
)
_PREPARED_DEFINITION_OPTIONAL_STRING_FIELDS = ("validity_column",)
_PREPARED_DEFINITION_INTEGER_FIELDS = (
    "cadence_minutes",
    "expected_epochs",
    "support_minutes",
)
_PREPARED_DEFINITION_FLOAT_FIELDS = ("unit_scale",)
_PREPARED_DEFINITION_NESTED_FIELDS = ("operation_params",)

_PREPARED_REPLAY_STRING_FIELDS = (
    "primitive_name",
    "sensor_file",
    "interval_boundary_policy",
)
_PREPARED_REPLAY_TIMESTAMP_FIELDS = ("lifelog_date", "sleep_date")
_PREPARED_REPLAY_DIGEST_FIELDS = (
    "source_record_digest",
    "source_schema_digest",
    "aligned_schema_digest",
    "aligned_frame_digest",
    "definition_digest",
    "policy_digest",
    "prepared_content_digest",
)
_PREPARED_REPLAY_NESTED_FIELDS = (
    "definition",
    "aligned_column_topology",
    "calendar_identity",
    "_phase_view_cache",
    "source_timestamps",
    "_aligned_frame",
    "_target_positions",
    "_record_starts",
    "_record_ends",
)
_PREPARED_REPLAY_CONSTRUCTION_ONLY_FIELDS = ("_build_token",)

_PREPARED_AUTHORITY_FIELD_COVERAGE = {
    "PrimitiveDefinition": (
        *_PREPARED_DEFINITION_STRING_FIELDS,
        *_PREPARED_DEFINITION_OPTIONAL_STRING_FIELDS,
        *_PREPARED_DEFINITION_INTEGER_FIELDS,
        *_PREPARED_DEFINITION_FLOAT_FIELDS,
        *_PREPARED_DEFINITION_NESTED_FIELDS,
    ),
    "PreparedCalendarIdentity": (
        *_PREPARED_CALENDAR_STRING_FIELDS,
        *_PREPARED_CALENDAR_INTEGER_FIELDS,
        *_PREPARED_CALENDAR_TIMESTAMP_FIELDS,
        *_PREPARED_CALENDAR_DIGEST_FIELDS,
    ),
    "PreparedPhaseRecord": (
        *_PREPARED_RECORD_STRING_FIELDS,
        *_PREPARED_RECORD_INTEGER_FIELDS,
        *_PREPARED_RECORD_TIMESTAMP_FIELDS,
        *_PREPARED_RECORD_FLOAT_FIELDS,
        *_PREPARED_RECORD_OPTIONAL_FLOAT_FIELDS,
        *_PREPARED_RECORD_BOOLEAN_FIELDS,
        *_PREPARED_RECORD_DIGEST_FIELDS,
    ),
    "PreparedPhaseView": (
        *_PREPARED_VIEW_STRING_FIELDS,
        *_PREPARED_VIEW_OPTIONAL_STRING_FIELDS,
        *_PREPARED_VIEW_INTEGER_FIELDS,
        *_PREPARED_VIEW_FLOAT_FIELDS,
        *_PREPARED_VIEW_DIGEST_FIELDS,
        *_PREPARED_VIEW_NESTED_FIELDS,
    ),
    "PreparedMaskInterval": (
        *_PREPARED_MASK_INTERVAL_INTEGER_FIELDS,
        *_PREPARED_MASK_INTERVAL_TIMESTAMP_FIELDS,
        *_PREPARED_MASK_INTERVAL_FLOAT_FIELDS,
        *_PREPARED_MASK_INTERVAL_DIGEST_FIELDS,
    ),
    "PreparedMaskBinGeometry": (
        *_PREPARED_MASK_BIN_INTEGER_FIELDS,
        *_PREPARED_MASK_BIN_FLOAT_FIELDS,
        *_PREPARED_MASK_BIN_DIGEST_FIELDS,
    ),
    "PreparedMaskGeometry": (
        *_PREPARED_MASK_GEOMETRY_STRING_FIELDS,
        *_PREPARED_MASK_GEOMETRY_INTEGER_FIELDS,
        *_PREPARED_MASK_GEOMETRY_FLOAT_FIELDS,
        *_PREPARED_MASK_GEOMETRY_DIGEST_FIELDS,
        *_PREPARED_MASK_GEOMETRY_NESTED_FIELDS,
    ),
    "PreparedPrimitiveReplay": (
        *_PREPARED_REPLAY_STRING_FIELDS,
        *_PREPARED_REPLAY_TIMESTAMP_FIELDS,
        *_PREPARED_REPLAY_DIGEST_FIELDS,
        *_PREPARED_REPLAY_NESTED_FIELDS,
        *_PREPARED_REPLAY_CONSTRUCTION_ONLY_FIELDS,
    ),
}

_PREPARED_PHASE_OPERATIONS = frozenset(
    {"binary_load", "activity_load", "additive_load", "exposure_mean", "evening_p90"}
)
_PREPARED_REPLAY_OPERATIONS = frozenset(
    {
        "binary_load",
        "binary_active_minutes",
        "activity_load",
        "additive_load",
        "exposure_mean",
        "evening_p90",
        "low_light_onset",
        "heart_rate_settling",
        "charging_onset",
    }
)
_PREPARED_OPERATION_PARAMETER_TYPES = {
    "epoch_minutes": float,
    "minimum": float,
    "validity_equals": int,
    "transform": str,
    "low_threshold": float,
    "bin_minutes": int,
    "min_observations": int,
    "low_fraction_threshold": float,
    "component": str,
    "early_start_minute": int,
    "early_end_minute": int,
    "late_start_minute": int,
    "late_end_minute": int,
    "median_bin_minutes": int,
    "min_segment_epochs": int,
    "min_duration_minutes": int,
    "require_off_to_on": bool,
}
_PREPARED_OPERATION_PARAMETER_NAMES = {
    "binary_load": frozenset(),
    "binary_active_minutes": frozenset({"epoch_minutes"}),
    "activity_load": frozenset({"validity_equals"}),
    "additive_load": frozenset({"minimum"}),
    "exposure_mean": frozenset({"minimum", "transform"}),
    "evening_p90": frozenset({"minimum", "validity_equals"}),
    "low_light_onset": frozenset(
        {
            "low_threshold",
            "bin_minutes",
            "min_observations",
            "low_fraction_threshold",
        }
    ),
    "heart_rate_settling": frozenset(
        {
            "component",
            "early_start_minute",
            "early_end_minute",
            "late_start_minute",
            "late_end_minute",
            "median_bin_minutes",
            "min_segment_epochs",
        }
    ),
    "charging_onset": frozenset(
        {"min_duration_minutes", "require_off_to_on"}
    ),
}


def _make_prepared_construction_boundary():
    token = object()

    def construct(cls: type, /, **kwargs: object):
        return cls(_build_token=token, **kwargs)

    def validates(candidate: object) -> bool:
        return candidate is token

    return construct, validates


_construct_prepared_calendar, _valid_prepared_calendar_token = (
    _make_prepared_construction_boundary()
)
_construct_prepared_record, _valid_prepared_record_token = (
    _make_prepared_construction_boundary()
)
_construct_prepared_view, _valid_prepared_view_token = (
    _make_prepared_construction_boundary()
)
_construct_prepared_replay, _valid_prepared_replay_token = (
    _make_prepared_construction_boundary()
)
_construct_prepared_mask_interval, _valid_prepared_mask_interval_token = (
    _make_prepared_construction_boundary()
)
_construct_prepared_mask_bin, _valid_prepared_mask_bin_token = (
    _make_prepared_construction_boundary()
)
_construct_prepared_mask_geometry, _valid_prepared_mask_geometry_token = (
    _make_prepared_construction_boundary()
)


def _prepared_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _prepared_float_token(value: float) -> str:
    number = float(value)
    if np.isnan(number):
        return "nan"
    if np.isposinf(number):
        return "+inf"
    if np.isneginf(number):
        return "-inf"
    return number.hex()


def _prepared_timestamp(value: object, name: str) -> pd.Timestamp:
    if type(value) is not pd.Timestamp:
        raise ValueError(f"{name} must be an exact pandas Timestamp")
    if pd.isna(value) or value.tz is not None:
        raise ValueError(f"{name} must be a timezone-naive timestamp")
    return value


def _validate_prepared_exact_strings(
    candidate: object,
    field_names: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field_name in field_names:
        value = getattr(candidate, field_name)
        if type(value) is not str or not value:
            raise ValueError(
                f"{label} {field_name} must be a nonempty exact string"
            )


def _validate_prepared_optional_strings(
    candidate: object,
    field_names: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field_name in field_names:
        value = getattr(candidate, field_name)
        if value is not None and (type(value) is not str or not value):
            raise ValueError(
                f"{label} {field_name} must be None or a nonempty exact string"
            )


def _validate_prepared_digest_fields(
    candidate: object,
    field_names: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field_name in field_names:
        value = getattr(candidate, field_name)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(
                f"{label} {field_name} must be an exact lowercase SHA-256 hex string"
            )


def _validate_prepared_exact_integers(
    candidate: object,
    field_names: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field_name in field_names:
        if type(getattr(candidate, field_name)) is not int:
            raise ValueError(f"{label} {field_name} must be an exact integer")


def _validate_prepared_exact_floats(
    candidate: object,
    field_names: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field_name in field_names:
        if type(getattr(candidate, field_name)) is not float:
            raise ValueError(f"{label} {field_name} must be an exact float")


def _validate_prepared_operation_parameter(name: object, value: object) -> None:
    if type(name) is not str or not name:
        raise ValueError("definition operation parameter name must be an exact string")
    expected_type = _PREPARED_OPERATION_PARAMETER_TYPES.get(name)
    if expected_type is None:
        raise ValueError(f"unsupported definition operation parameter: {name}")
    if type(value) is not expected_type:
        raise ValueError(
            f"definition operation parameter {name} must have exact type "
            f"{expected_type.__name__}"
        )
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError(f"definition operation parameter {name} must be finite")
        if name in {"epoch_minutes", "low_fraction_threshold"} and value <= 0.0:
            raise ValueError(f"definition operation parameter {name} must be positive")
        if name in {"minimum", "low_threshold"} and value < 0.0:
            raise ValueError(
                f"definition operation parameter {name} must be nonnegative"
            )
        if name == "low_fraction_threshold" and value > 1.0:
            raise ValueError(
                "definition low_fraction_threshold must not exceed one"
            )
    elif type(value) is int:
        if name != "validity_equals" and value <= 0:
            raise ValueError(f"definition operation parameter {name} must be positive")
        if name.endswith("_minute") and value > DAY_MINUTES:
            raise ValueError(
                f"definition operation parameter {name} exceeds one day"
            )
    elif type(value) is str:
        if not value:
            raise ValueError(f"definition operation parameter {name} is empty")
        if name == "transform" and value != "log1p":
            raise ValueError("definition transform must be log1p")
        if name == "component" and value not in {"delta", "slope"}:
            raise ValueError("definition heart-rate component is invalid")


def _validate_prepared_primitive_definition_structure(
    definition: object,
) -> PrimitiveDefinition:
    """Reapply the complete frozen definition contract before digest trust."""
    if type(definition) is not PrimitiveDefinition:
        raise TypeError("definition must be an exact PrimitiveDefinition")
    _validate_prepared_exact_strings(
        definition, _PREPARED_DEFINITION_STRING_FIELDS, label="definition"
    )
    _validate_prepared_optional_strings(
        definition,
        _PREPARED_DEFINITION_OPTIONAL_STRING_FIELDS,
        label="definition",
    )
    _validate_prepared_exact_integers(
        definition, _PREPARED_DEFINITION_INTEGER_FIELDS, label="definition"
    )
    _validate_prepared_exact_floats(
        definition, _PREPARED_DEFINITION_FLOAT_FIELDS, label="definition"
    )
    if definition.window not in {"day", "evening"}:
        raise ValueError("definition window is invalid")
    if definition.coverage_mode not in {"availability", "event_support"}:
        raise ValueError("definition coverage mode is invalid")
    if definition.timestamp_semantics not in {"instant", "interval_end"}:
        raise ValueError("definition timestamp semantics are invalid")
    if definition.timing_anchor not in {"source_timestamp", "support_midpoint"}:
        raise ValueError("definition timing anchor is invalid")
    if definition.operation not in _PREPARED_REPLAY_OPERATIONS:
        raise ValueError("definition operation is unsupported")
    if definition.cadence_minutes <= 0 or definition.expected_epochs <= 0:
        raise ValueError("definition cadence and expected epochs must be positive")
    if definition.support_minutes < 0:
        raise ValueError("definition support minutes must be nonnegative")
    if not np.isfinite(definition.unit_scale) or definition.unit_scale <= 0.0:
        raise ValueError("definition unit scale must be finite and positive")
    expected_window_minutes = DAY_MINUTES if definition.window == "day" else 360
    if definition.cadence_minutes * definition.expected_epochs != expected_window_minutes:
        raise ValueError("definition cadence and expected epochs do not fill its window")
    if definition.timestamp_semantics == "instant":
        if definition.support_minutes != 0 or definition.timing_anchor != "source_timestamp":
            raise ValueError("instant definition support contract is inconsistent")
    elif (
        definition.support_minutes <= 0
        or definition.timing_anchor != "support_midpoint"
    ):
        raise ValueError("interval-end definition support contract is inconsistent")
    if type(definition.operation_params) is not tuple:
        raise ValueError("definition operation_params must be an exact tuple")
    parameter_names: list[str] = []
    for item in definition.operation_params:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError(
                "definition operation_params entries must be exact name/value tuples"
            )
        name, value = item
        _validate_prepared_operation_parameter(name, value)
        parameter_names.append(name)
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("definition operation parameter names must be unique")
    if not set(parameter_names).issubset(
        _PREPARED_OPERATION_PARAMETER_NAMES[definition.operation]
    ):
        raise ValueError("definition operation parameter topology is invalid")
    frozen_matches = tuple(
        candidate
        for candidate in PRIMITIVE_DEFINITIONS
        if candidate.name == definition.name
    )
    if len(frozen_matches) != 1 or definition != frozen_matches[0]:
        raise ValueError("definition does not match the frozen primitive contract")
    if definition.operation == "heart_rate_settling":
        params = dict(definition.operation_params)
        if not (
            0
            <= params["early_start_minute"]
            < params["early_end_minute"]
            <= params["late_start_minute"]
            < params["late_end_minute"]
            <= DAY_MINUTES
        ):
            raise ValueError("definition heart-rate windows are inconsistent")
    return definition


def _prepared_definition_payload(definition: PrimitiveDefinition) -> dict[str, object]:
    return {
        "schema": "prepared-primitive-definition-v1",
        "name": definition.name,
        "sensor_file": definition.sensor_file,
        "value_column": definition.value_column,
        "window": definition.window,
        "cadence_minutes": definition.cadence_minutes,
        "expected_epochs": definition.expected_epochs,
        "coverage_mode": definition.coverage_mode,
        "timestamp_semantics": definition.timestamp_semantics,
        "support_minutes": definition.support_minutes,
        "timing_anchor": definition.timing_anchor,
        "unit": definition.unit,
        "unit_scale": _prepared_float_token(definition.unit_scale),
        "validity_column": definition.validity_column,
        "operation": definition.operation,
        "operation_params": [
            [name, type(value).__name__, repr(value)]
            for name, value in definition.operation_params
        ],
    }


def _prepared_definition_digest(definition: PrimitiveDefinition) -> str:
    _validate_prepared_primitive_definition_structure(definition)
    return _prepared_sha256(_prepared_definition_payload(definition))


def _prepared_policy_digest(interval_boundary_policy: str) -> str:
    return _prepared_sha256(
        {
            "schema": "prepared-replay-policy-v1",
            "interval_boundary_policy": interval_boundary_policy,
            "bin_minutes": 30,
            "record_deletion": "whole_record",
            "instant_mask_rule": "start<=timestamp<end",
            "interval_mask_rule": "support_start<end and support_end>start",
        }
    )


@dataclass(frozen=True)
class PreparedCalendarIdentity:
    """Exact immutable participant-day identity derived by preparation."""

    subject_id: str
    sensor_day_id: int
    lifelog_date: pd.Timestamp
    sleep_date: pd.Timestamp
    collection_start_date: pd.Timestamp
    elapsed_day_index: int
    day_type: str
    schema_digest: str
    content_digest: str
    _build_token: InitVar[object]

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_prepared_calendar_token(_build_token):
            raise TypeError(
                "PreparedCalendarIdentity must be created by the trusted prepare boundary"
            )
        _validate_prepared_calendar_identity(self)


def _prepared_calendar_content_digest(identity: PreparedCalendarIdentity) -> str:
    return _prepared_sha256(_prepared_calendar_payload(identity))


def _prepared_calendar_payload(identity: PreparedCalendarIdentity) -> dict[str, object]:
    return _prepared_calendar_values_payload(
        subject_id=identity.subject_id,
        sensor_day_id=identity.sensor_day_id,
        lifelog_date=identity.lifelog_date,
        sleep_date=identity.sleep_date,
        collection_start_date=identity.collection_start_date,
        elapsed_day_index=identity.elapsed_day_index,
        day_type=identity.day_type,
        schema_digest=identity.schema_digest,
    )


def _prepared_calendar_values_payload(
    *,
    subject_id: str,
    sensor_day_id: int,
    lifelog_date: pd.Timestamp,
    sleep_date: pd.Timestamp,
    collection_start_date: pd.Timestamp,
    elapsed_day_index: int,
    day_type: str,
    schema_digest: str,
) -> dict[str, object]:
    return {
        "schema": _PREPARED_CALENDAR_SCHEMA,
        "subject_id": subject_id,
        "sensor_day_id": sensor_day_id,
        "lifelog_date": lifelog_date.isoformat(),
        "sleep_date": sleep_date.isoformat(),
        "collection_start_date": collection_start_date.isoformat(),
        "elapsed_day_index": elapsed_day_index,
        "day_type": day_type,
        "schema_digest": schema_digest,
    }


def _validate_prepared_calendar_identity(identity: object) -> PreparedCalendarIdentity:
    if type(identity) is not PreparedCalendarIdentity:
        raise TypeError("calendar identity must be an exact PreparedCalendarIdentity")
    _validate_prepared_exact_strings(
        identity, _PREPARED_CALENDAR_STRING_FIELDS, label="calendar"
    )
    _validate_prepared_exact_integers(
        identity, _PREPARED_CALENDAR_INTEGER_FIELDS, label="calendar"
    )
    _validate_prepared_digest_fields(
        identity, _PREPARED_CALENDAR_DIGEST_FIELDS, label="calendar"
    )
    lifelog = _prepared_timestamp(identity.lifelog_date, "lifelog_date")
    sleep = _prepared_timestamp(identity.sleep_date, "sleep_date")
    collection = _prepared_timestamp(
        identity.collection_start_date, "collection_start_date"
    )
    if identity.sensor_day_id < 0:
        raise ValueError("sensor_day_id must be an exact nonnegative integer")
    if (
        lifelog != lifelog.normalize()
        or sleep != sleep.normalize()
        or collection != collection.normalize()
    ):
        raise ValueError("calendar dates must be exact local midnights")
    if sleep - lifelog != pd.Timedelta(days=1):
        raise ValueError("sleep_date must be one day after lifelog_date")
    if identity.elapsed_day_index < 1:
        raise ValueError("elapsed_day_index must be an exact positive integer")
    expected_elapsed = int((lifelog - collection).days + 1)
    if identity.elapsed_day_index != expected_elapsed:
        raise ValueError("elapsed_day_index must match collection_start_date")
    expected_day_type = "weekend" if lifelog.weekday() >= 5 else "weekday"
    if identity.day_type != expected_day_type:
        raise ValueError("day_type must be derived from lifelog_date")
    expected_schema = _prepared_sha256({"schema": _PREPARED_CALENDAR_SCHEMA})
    if identity.schema_digest != expected_schema:
        raise ValueError("calendar schema digest does not match")
    if identity.content_digest != _prepared_calendar_content_digest(identity):
        raise ValueError("calendar content digest does not match")
    return identity


def _derive_prepared_calendar_identity(
    calendar_row: pd.DataFrame,
    scaffold: pd.DataFrame,
) -> PreparedCalendarIdentity | None:
    if any(
        column in calendar_row.columns
        for column in ("day_type", "weekday", "weekend")
    ):
        raise ValueError("day_type is derived inside the prepare boundary")
    if not all(column in calendar_row.columns for column in SAFE_SCAFFOLD_METADATA):
        return None
    raw = calendar_row.iloc[0]
    if type(raw["subject_id"]) is not str or not raw["subject_id"]:
        raise ValueError("subject_id must be a nonempty exact string")
    for name, minimum in (("sensor_day_id", 0), ("elapsed_day_index", 1)):
        series = calendar_row[name]
        if (
            not pd.api.types.is_integer_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        ):
            raise ValueError(f"{name} must have an exact integer dtype")
        value = raw[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must contain exact integers")
        if int(value) < minimum:
            raise ValueError(f"{name} is outside its valid range")
    day = scaffold.iloc[0]
    lifelog = pd.Timestamp(day["lifelog_date"])
    sleep = pd.Timestamp(day["sleep_date"])
    collection = pd.Timestamp(day["collection_start_date"])
    schema_digest = _prepared_sha256({"schema": _PREPARED_CALENDAR_SCHEMA})
    values = {
        "subject_id": raw["subject_id"],
        "sensor_day_id": int(day["sensor_day_id"]),
        "lifelog_date": lifelog,
        "sleep_date": sleep,
        "collection_start_date": collection,
        "elapsed_day_index": int(day["elapsed_day_index"]),
        "day_type": "weekend" if lifelog.weekday() >= 5 else "weekday",
        "schema_digest": schema_digest,
    }
    content_digest = _prepared_sha256(_prepared_calendar_values_payload(**values))
    return _construct_prepared_calendar(
        PreparedCalendarIdentity,
        **values,
        content_digest=content_digest,
    )


@dataclass(frozen=True)
class PreparedPhaseRecord:
    """One immutable target-window record with operation-specific statistics."""

    record_id: int
    subject_id: str
    source_timestamp: pd.Timestamp
    support_start: pd.Timestamp
    support_end: pd.Timestamp
    effective_timestamp: pd.Timestamp
    phase_bin: int
    raw_value: float
    raw_validity: float | None
    valid: bool
    support_units: float
    transformed_numerator: float
    positive_event_count: int
    event_mass: float
    event_minute: float | None
    content_digest: str
    _build_token: InitVar[object]

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_prepared_record_token(_build_token):
            raise TypeError(
                "PreparedPhaseRecord must be created by the trusted prepare boundary"
            )
        _validate_prepared_phase_record(self)


def _prepared_record_payload(record: PreparedPhaseRecord) -> dict[str, object]:
    return _prepared_record_values_payload(
        record_id=record.record_id,
        subject_id=record.subject_id,
        source_timestamp=record.source_timestamp,
        support_start=record.support_start,
        support_end=record.support_end,
        effective_timestamp=record.effective_timestamp,
        phase_bin=record.phase_bin,
        raw_value=record.raw_value,
        raw_validity=record.raw_validity,
        valid=record.valid,
        support_units=record.support_units,
        transformed_numerator=record.transformed_numerator,
        positive_event_count=record.positive_event_count,
        event_mass=record.event_mass,
        event_minute=record.event_minute,
    )


def _prepared_record_values_payload(
    *,
    record_id: int,
    subject_id: str,
    source_timestamp: pd.Timestamp,
    support_start: pd.Timestamp,
    support_end: pd.Timestamp,
    effective_timestamp: pd.Timestamp,
    phase_bin: int,
    raw_value: float,
    raw_validity: float | None,
    valid: bool,
    support_units: float,
    transformed_numerator: float,
    positive_event_count: int,
    event_mass: float,
    event_minute: float | None,
) -> dict[str, object]:
    return {
        "schema": _PREPARED_RECORD_SCHEMA,
        "record_id": record_id,
        "subject_id": subject_id,
        "source_timestamp": source_timestamp.isoformat(),
        "support_start": support_start.isoformat(),
        "support_end": support_end.isoformat(),
        "effective_timestamp": effective_timestamp.isoformat(),
        "phase_bin": phase_bin,
        "raw_value": _prepared_float_token(raw_value),
        "raw_validity": (
            None
            if raw_validity is None
            else _prepared_float_token(raw_validity)
        ),
        "valid": valid,
        "support_units": _prepared_float_token(support_units),
        "transformed_numerator": _prepared_float_token(
            transformed_numerator
        ),
        "positive_event_count": positive_event_count,
        "event_mass": _prepared_float_token(event_mass),
        "event_minute": (
            None
            if event_minute is None
            else _prepared_float_token(event_minute)
        ),
    }


def _prepared_record_content_digest(record: PreparedPhaseRecord) -> str:
    return _prepared_sha256(_prepared_record_payload(record))


def _validate_prepared_phase_record(record: object) -> PreparedPhaseRecord:
    if type(record) is not PreparedPhaseRecord:
        raise TypeError("record must be an exact PreparedPhaseRecord")
    _validate_prepared_exact_strings(
        record, _PREPARED_RECORD_STRING_FIELDS, label="record"
    )
    _validate_prepared_exact_integers(
        record, _PREPARED_RECORD_INTEGER_FIELDS, label="record"
    )
    _validate_prepared_exact_floats(
        record, _PREPARED_RECORD_FLOAT_FIELDS, label="record"
    )
    _validate_prepared_digest_fields(
        record, _PREPARED_RECORD_DIGEST_FIELDS, label="record"
    )
    for field_name in _PREPARED_RECORD_TIMESTAMP_FIELDS:
        _prepared_timestamp(getattr(record, field_name), field_name)
    for field_name in _PREPARED_RECORD_OPTIONAL_FLOAT_FIELDS:
        value = getattr(record, field_name)
        if value is not None and type(value) is not float:
            raise ValueError(f"record {field_name} must be None or an exact float")
    if type(record.valid) is not bool:
        raise ValueError("record valid must be an exact boolean")
    if record.record_id < 0:
        raise ValueError("record_id must be an exact nonnegative integer")
    if record.support_start > record.support_end:
        raise ValueError("record support interval is invalid")
    if not 0 <= record.phase_bin < 48:
        raise ValueError("phase_bin must be an exact integer in [0, 48)")
    offset_minutes = float(
        (record.effective_timestamp - record.effective_timestamp.normalize())
        / pd.Timedelta(minutes=1)
    )
    if record.phase_bin != int(np.floor(offset_minutes / 30.0)):
        raise ValueError("phase_bin must match the effective timestamp")
    for field_name in ("support_units", "transformed_numerator", "event_mass"):
        value = getattr(record, field_name)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{field_name} must be an exact finite nonnegative float"
            )
    if record.support_units not in (0.0, 1.0):
        raise ValueError("support_units must encode whole-record support")
    expected_support = 1.0 if record.valid else 0.0
    if record.support_units != expected_support:
        raise ValueError("support_units must match record validity")
    if not record.valid and record.transformed_numerator != 0.0:
        raise ValueError("invalid records cannot carry a transformed numerator")
    if record.positive_event_count not in (0, 1):
        raise ValueError("positive_event_count must be the exact integer 0 or 1")
    if record.event_minute is not None and (
        not np.isfinite(record.event_minute)
        or not 0.0 <= record.event_minute < 1440.0
    ):
        raise ValueError("event_minute must lie in [0, 1440)")
    if record.positive_event_count == 0 and (
        record.event_mass != 0.0 or record.event_minute is not None
    ):
        raise ValueError("non-event records cannot carry event mass or minute")
    if record.positive_event_count == 1 and (
        record.event_mass <= 0.0 or record.event_minute is None
    ):
        raise ValueError("positive events require separate mass and minute")
    if record.content_digest != _prepared_record_content_digest(record):
        raise ValueError("prepared record content digest is invalid")
    return record


def _construct_phase_record(**values: object) -> PreparedPhaseRecord:
    return _construct_prepared_record(
        PreparedPhaseRecord,
        **values,
        content_digest=_prepared_sha256(_prepared_record_values_payload(**values)),
    )


def _prepared_source_digest(
    records: tuple[PreparedPhaseRecord, ...],
    source_schema_digest: str,
    aligned_schema_digest: str,
    *,
    has_validity: bool,
) -> str:
    raw_validity = (
        np.asarray([record.raw_validity for record in records], dtype="<f8")
        if has_validity
        else None
    )
    if any((record.raw_validity is None) != (raw_validity is None) for record in records):
        raise ValueError("prepared records mix incompatible validity schemas")
    return _prepared_source_digest_arrays(
        source_schema_digest=source_schema_digest,
        aligned_schema_digest=aligned_schema_digest,
        subject_ids=tuple(record.subject_id for record in records),
        source_timestamps=np.asarray(
            [record.source_timestamp.value for record in records], dtype="<i8"
        ),
        support_starts=np.asarray(
            [record.support_start.value for record in records], dtype="<i8"
        ),
        support_ends=np.asarray(
            [record.support_end.value for record in records], dtype="<i8"
        ),
        effective_timestamps=np.asarray(
            [record.effective_timestamp.value for record in records], dtype="<i8"
        ),
        raw_values=np.asarray([record.raw_value for record in records], dtype="<f8"),
        raw_validity=raw_validity,
    )


def _prepared_source_digest_arrays(
    *,
    source_schema_digest: str,
    aligned_schema_digest: str,
    subject_ids: tuple[str, ...],
    source_timestamps: np.ndarray,
    support_starts: np.ndarray,
    support_ends: np.ndarray,
    effective_timestamps: np.ndarray,
    raw_values: np.ndarray,
    raw_validity: np.ndarray | None,
) -> str:
    count = int(len(source_timestamps))
    if len(subject_ids) != count or not all(
        type(subject_id) is str and subject_id for subject_id in subject_ids
    ):
        raise ValueError("prepared source subjects are invalid")
    arrays = (
        source_timestamps,
        support_starts,
        support_ends,
        effective_timestamps,
        raw_values,
    )
    if any(len(array) != count for array in arrays):
        raise ValueError("prepared source digest arrays have inconsistent lengths")
    if raw_validity is not None and len(raw_validity) != count:
        raise ValueError("prepared source validity has an inconsistent length")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": "prepared-source-records-v4",
                "count": count,
                "has_validity": raw_validity is not None,
                "source_schema_digest": source_schema_digest,
                "aligned_schema_digest": aligned_schema_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"subject_id\0")
    for subject_id in subject_ids:
        encoded = subject_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    for name, array, dtype in (
        ("source_timestamp", source_timestamps, "<i8"),
        ("support_start", support_starts, "<i8"),
        ("support_end", support_ends, "<i8"),
        ("effective_timestamp", effective_timestamps, "<i8"),
        ("raw_value", raw_values, "<f8"),
    ):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        if canonical.dtype.kind == "f":
            canonical = canonical.copy()
            canonical[np.isnan(canonical)] = np.nan
        digest.update(name.encode("ascii") + b"\0")
        digest.update(canonical.tobytes(order="C"))
    if raw_validity is not None:
        canonical_validity = np.ascontiguousarray(raw_validity, dtype="<f8").copy()
        canonical_validity[np.isnan(canonical_validity)] = np.nan
        digest.update(b"raw_validity\0")
        digest.update(canonical_validity.tobytes(order="C"))
    return digest.hexdigest()


def _prepared_source_digest_frame(
    target: pd.DataFrame,
    definition: PrimitiveDefinition,
    source_schema_digest: str,
    aligned_schema_digest: str,
) -> str:
    subject_ids = tuple(target["subject_id"].tolist())
    if not all(
        type(subject_id) is str and subject_id for subject_id in subject_ids
    ):
        raise ValueError("prepared target subjects must be exact nonempty strings")

    def timestamp_array(column: str) -> np.ndarray:
        return pd.to_datetime(target[column], errors="raise").to_numpy(
            dtype="datetime64[ns]", copy=True
        ).astype("<i8", copy=False)

    raw_values = pd.to_numeric(
        target[definition.value_column], errors="coerce"
    ).to_numpy(dtype="<f8", copy=True)
    raw_validity = (
        None
        if definition.validity_column is None
        else pd.to_numeric(
            target[definition.validity_column], errors="coerce"
        ).to_numpy(dtype="<f8", copy=True)
    )
    return _prepared_source_digest_arrays(
        source_schema_digest=source_schema_digest,
        aligned_schema_digest=aligned_schema_digest,
        subject_ids=subject_ids,
        source_timestamps=timestamp_array("source_timestamp"),
        support_starts=timestamp_array("support_start"),
        support_ends=timestamp_array("support_end"),
        effective_timestamps=timestamp_array("effective_timestamp"),
        raw_values=raw_values,
        raw_validity=raw_validity,
    )


@dataclass(frozen=True)
class PreparedPhaseView:
    """Tuple-backed public authority for one prepared primitive-day."""

    calendar_identity: PreparedCalendarIdentity
    primitive_name: str
    sensor_file: str
    value_column: str
    validity_column: str | None
    operation: str
    window: str
    cadence_minutes: int
    expected_epochs: int
    coverage_mode: str
    timestamp_semantics: str
    support_minutes: int
    timing_anchor: str
    unit: str
    unit_scale: float
    operation_params: tuple[tuple[str, str, str], ...]
    interval_boundary_policy: str
    definition_digest: str
    policy_digest: str
    source_schema_digest: str
    aligned_schema_digest: str
    aligned_column_topology: tuple[tuple[str, str], ...]
    aligned_frame_digest: str
    source_record_digest: str
    schema_digest: str
    records: tuple[PreparedPhaseRecord, ...]
    content_digest: str
    _build_token: InitVar[object]

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_prepared_view_token(_build_token):
            raise TypeError(
                "PreparedPhaseView must be created by the trusted prepare boundary"
            )
        validate_prepared_phase_view(self)


def _prepared_view_content_digest(view: PreparedPhaseView) -> str:
    return _prepared_sha256(
        _prepared_view_values_payload(
            calendar_identity=view.calendar_identity,
            primitive_name=view.primitive_name,
            sensor_file=view.sensor_file,
            value_column=view.value_column,
            validity_column=view.validity_column,
            operation=view.operation,
            window=view.window,
            cadence_minutes=view.cadence_minutes,
            expected_epochs=view.expected_epochs,
            coverage_mode=view.coverage_mode,
            timestamp_semantics=view.timestamp_semantics,
            support_minutes=view.support_minutes,
            timing_anchor=view.timing_anchor,
            unit=view.unit,
            unit_scale=view.unit_scale,
            operation_params=view.operation_params,
            interval_boundary_policy=view.interval_boundary_policy,
            definition_digest=view.definition_digest,
            policy_digest=view.policy_digest,
            source_schema_digest=view.source_schema_digest,
            aligned_schema_digest=view.aligned_schema_digest,
            aligned_column_topology=view.aligned_column_topology,
            aligned_frame_digest=view.aligned_frame_digest,
            source_record_digest=view.source_record_digest,
            schema_digest=view.schema_digest,
            records=view.records,
        )
    )


def _prepared_view_values_payload(
    *,
    calendar_identity: PreparedCalendarIdentity,
    primitive_name: str,
    sensor_file: str,
    value_column: str,
    validity_column: str | None,
    operation: str,
    window: str,
    cadence_minutes: int,
    expected_epochs: int,
    coverage_mode: str,
    timestamp_semantics: str,
    support_minutes: int,
    timing_anchor: str,
    unit: str,
    unit_scale: float,
    operation_params: tuple[tuple[str, str, str], ...],
    interval_boundary_policy: str,
    definition_digest: str,
    policy_digest: str,
    source_schema_digest: str,
    aligned_schema_digest: str,
    aligned_column_topology: tuple[tuple[str, str], ...],
    aligned_frame_digest: str,
    source_record_digest: str,
    schema_digest: str,
    records: tuple[PreparedPhaseRecord, ...],
) -> dict[str, object]:
    return {
        "schema": _PREPARED_PHASE_SCHEMA,
        "calendar_digest": calendar_identity.content_digest,
        "primitive_name": primitive_name,
        "sensor_file": sensor_file,
        "value_column": value_column,
        "validity_column": validity_column,
        "operation": operation,
        "window": window,
        "cadence_minutes": cadence_minutes,
        "expected_epochs": expected_epochs,
        "coverage_mode": coverage_mode,
        "timestamp_semantics": timestamp_semantics,
        "support_minutes": support_minutes,
        "timing_anchor": timing_anchor,
        "unit": unit,
        "unit_scale": _prepared_float_token(unit_scale),
        "operation_params": [list(item) for item in operation_params],
        "interval_boundary_policy": interval_boundary_policy,
        "definition_digest": definition_digest,
        "policy_digest": policy_digest,
        "source_schema_digest": source_schema_digest,
        "aligned_schema_digest": aligned_schema_digest,
        "aligned_column_topology": [list(item) for item in aligned_column_topology],
        "aligned_frame_digest": aligned_frame_digest,
        "source_record_digest": source_record_digest,
        "schema_digest": schema_digest,
        "records": [record.content_digest for record in records],
    }


def validate_prepared_phase_view(view: PreparedPhaseView) -> None:
    """Fail closed when any public prepared-view field was altered."""
    if type(view) is not PreparedPhaseView:
        raise TypeError("view must be an exact PreparedPhaseView")
    _validate_prepared_exact_strings(
        view, _PREPARED_VIEW_STRING_FIELDS, label="view"
    )
    _validate_prepared_optional_strings(
        view, _PREPARED_VIEW_OPTIONAL_STRING_FIELDS, label="view"
    )
    _validate_prepared_exact_integers(
        view, _PREPARED_VIEW_INTEGER_FIELDS, label="view"
    )
    _validate_prepared_exact_floats(
        view, _PREPARED_VIEW_FLOAT_FIELDS, label="view"
    )
    _validate_prepared_digest_fields(
        view, _PREPARED_VIEW_DIGEST_FIELDS, label="view"
    )
    _validate_prepared_calendar_identity(view.calendar_identity)
    if view.operation not in _PREPARED_PHASE_OPERATIONS:
        raise ValueError("view operation is not a frozen prepared operation")
    if view.window not in {"day", "evening"}:
        raise ValueError("view window must be day or evening")
    if view.operation == "evening_p90" and view.window != "evening":
        raise ValueError("evening timing operations require the evening window")
    if view.operation != "evening_p90" and view.window != "day":
        raise ValueError("prepared load/exposure operations require the day window")
    if (
        view.cadence_minutes <= 0
        or view.cadence_minutes > 30
        or 30 % view.cadence_minutes != 0
    ):
        raise ValueError("view cadence_minutes must exactly divide a half-hour bin")
    window_minutes = 1440 if view.window == "day" else 360
    if view.expected_epochs != window_minutes // view.cadence_minutes:
        raise ValueError("view expected_epochs does not match window and cadence")
    if view.coverage_mode not in {"availability", "event_support"}:
        raise ValueError("view coverage_mode is invalid")
    if view.timestamp_semantics not in {"instant", "interval_end"}:
        raise ValueError("view timestamp_semantics is invalid")
    if view.support_minutes < 0:
        raise ValueError("view support_minutes must be nonnegative")
    if view.timestamp_semantics == "instant" and (
        view.support_minutes != 0 or view.timing_anchor != "source_timestamp"
    ):
        raise ValueError("instant views require zero support and source anchoring")
    if view.timestamp_semantics == "interval_end" and (
        view.support_minutes != view.cadence_minutes
        or view.timing_anchor != "support_midpoint"
    ):
        raise ValueError("interval-end views require cadence support and midpoint anchoring")
    if not np.isfinite(view.unit_scale) or view.unit_scale <= 0.0:
        raise ValueError("view unit_scale must be an exact finite positive float")
    if view.interval_boundary_policy not in INTERVAL_BOUNDARY_POLICIES:
        raise ValueError("view interval boundary policy is invalid")
    if view.policy_digest != _prepared_policy_digest(
        view.interval_boundary_policy
    ):
        raise ValueError("view policy digest does not match its policy")
    if type(view.operation_params) is not tuple or not all(
        type(item) is tuple
        and len(item) == 3
        and all(type(value) is str and value for value in item)
        for item in view.operation_params
    ):
        raise ValueError("view operation_params must be an exact typed tuple")
    parameter_names = tuple(item[0] for item in view.operation_params)
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("view operation parameter names must be unique")
    if type(view.aligned_column_topology) is not tuple or not all(
        type(item) is tuple
        and len(item) == 2
        and all(type(value) is str and value for value in item)
        for item in view.aligned_column_topology
    ):
        raise ValueError("aligned_column_topology must be an exact typed tuple")
    topology_names = tuple(item[0] for item in view.aligned_column_topology)
    if len(topology_names) != len(set(topology_names)):
        raise ValueError("aligned_column_topology names must be unique")
    if type(view.records) is not tuple:
        raise ValueError("records must be an exact tuple of PreparedPhaseRecord")
    for record in view.records:
        _validate_prepared_phase_record(record)
        if record.subject_id != view.calendar_identity.subject_id:
            raise ValueError("view record subject is crosswired with its calendar")
    if tuple(record.record_id for record in view.records) != tuple(
        range(len(view.records))
    ):
        raise ValueError("record IDs must be canonical and consecutive")
    active_bins = range(48) if view.window == "day" else range(36, 48)
    if any(record.phase_bin not in active_bins for record in view.records):
        raise ValueError("record phase bins are outside the view window")
    if view.operation != "evening_p90" and any(
        record.positive_event_count != 0 for record in view.records
    ):
        raise ValueError("non-timing views cannot carry positive timing events")
    expected_schema = _prepared_sha256({"schema": _PREPARED_PHASE_SCHEMA})
    if view.schema_digest != expected_schema:
        raise ValueError("prepared phase schema digest does not match")
    if view.source_record_digest != _prepared_source_digest(
        view.records,
        view.source_schema_digest,
        view.aligned_schema_digest,
        has_validity=view.validity_column is not None,
    ):
        raise ValueError("prepared source digest does not match record content")
    if view.content_digest != _prepared_view_content_digest(view):
        raise ValueError("prepared view content digest does not match")


@dataclass(frozen=True)
class PreparedMaskInterval:
    """One canonical half-open interval from a reviewed G2 mask table."""

    mask_interval_id: int
    mask_start: pd.Timestamp
    mask_end: pd.Timestamp
    duration_minutes: float
    content_digest: str
    _build_token: InitVar[object]

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_prepared_mask_interval_token(_build_token):
            raise TypeError(
                "PreparedMaskInterval must be created by the trusted geometry boundary"
            )
        _validate_prepared_mask_interval(self)


def _prepared_mask_interval_payload_values(
    *,
    mask_interval_id: int,
    mask_start: pd.Timestamp,
    mask_end: pd.Timestamp,
    duration_minutes: float,
) -> dict[str, object]:
    return {
        "schema": _PREPARED_MASK_INTERVAL_SCHEMA,
        "mask_interval_id": mask_interval_id,
        "mask_start": mask_start.isoformat(),
        "mask_end": mask_end.isoformat(),
        "duration_minutes": _prepared_float_token(duration_minutes),
    }


def _prepared_mask_interval_digest_values(**values: object) -> str:
    return _prepared_sha256(_prepared_mask_interval_payload_values(**values))


def _validate_prepared_mask_interval(
    interval: object,
) -> PreparedMaskInterval:
    if type(interval) is not PreparedMaskInterval:
        raise TypeError("mask interval must be an exact PreparedMaskInterval")
    _validate_prepared_exact_integers(
        interval, _PREPARED_MASK_INTERVAL_INTEGER_FIELDS, label="mask interval"
    )
    _validate_prepared_exact_floats(
        interval, _PREPARED_MASK_INTERVAL_FLOAT_FIELDS, label="mask interval"
    )
    _validate_prepared_digest_fields(
        interval, _PREPARED_MASK_INTERVAL_DIGEST_FIELDS, label="mask interval"
    )
    start = _prepared_timestamp(interval.mask_start, "mask_start")
    end = _prepared_timestamp(interval.mask_end, "mask_end")
    if interval.mask_interval_id < 0:
        raise ValueError("mask_interval_id must be an exact nonnegative integer")
    if start >= end:
        raise ValueError("mask interval must have positive duration")
    if not np.isfinite(interval.duration_minutes) or interval.duration_minutes <= 0.0:
        raise ValueError("duration_minutes must be an exact finite positive float")
    expected = float((end - start) / pd.Timedelta(minutes=1))
    if not np.isclose(
        interval.duration_minutes, expected, rtol=0.0, atol=1e-12
    ):
        raise ValueError("mask duration does not match interval endpoints")
    if interval.content_digest != _prepared_mask_interval_digest_values(
        mask_interval_id=interval.mask_interval_id,
        mask_start=start,
        mask_end=end,
        duration_minutes=interval.duration_minutes,
    ):
        raise ValueError("mask interval content digest is invalid")
    return interval


def _prepared_mask_interval_payload(
    interval: PreparedMaskInterval,
) -> dict[str, object]:
    return _prepared_mask_interval_payload_values(
        mask_interval_id=interval.mask_interval_id,
        mask_start=interval.mask_start,
        mask_end=interval.mask_end,
        duration_minutes=interval.duration_minutes,
    )


def _construct_mask_interval(**values: object) -> PreparedMaskInterval:
    return _construct_prepared_mask_interval(
        PreparedMaskInterval,
        **values,
        content_digest=_prepared_mask_interval_digest_values(**values),
    )


@dataclass(frozen=True)
class PreparedMaskBinGeometry:
    """Whole-valid-record deletion geometry for one half-hour bin."""

    phase_bin: int
    expected_units: float
    baseline_units: float
    observed_units: float
    induced_deleted_units: float
    masked_fraction: float
    content_digest: str
    _build_token: InitVar[object]

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_prepared_mask_bin_token(_build_token):
            raise TypeError(
                "PreparedMaskBinGeometry must be created by the trusted geometry boundary"
            )
        _validate_prepared_mask_bin_geometry(self)


def _prepared_mask_bin_payload_values(
    *,
    phase_bin: int,
    expected_units: float,
    baseline_units: float,
    observed_units: float,
    induced_deleted_units: float,
    masked_fraction: float,
) -> dict[str, object]:
    return {
        "schema": _PREPARED_MASK_BIN_SCHEMA,
        "phase_bin": phase_bin,
        "expected_units": _prepared_float_token(expected_units),
        "baseline_units": _prepared_float_token(baseline_units),
        "observed_units": _prepared_float_token(observed_units),
        "induced_deleted_units": _prepared_float_token(induced_deleted_units),
        "masked_fraction": _prepared_float_token(masked_fraction),
    }


def _prepared_mask_bin_digest_values(**values: object) -> str:
    return _prepared_sha256(_prepared_mask_bin_payload_values(**values))


def _validate_prepared_mask_bin_geometry(
    item: object,
) -> PreparedMaskBinGeometry:
    if type(item) is not PreparedMaskBinGeometry:
        raise TypeError("mask bin must be an exact PreparedMaskBinGeometry")
    _validate_prepared_exact_integers(
        item, _PREPARED_MASK_BIN_INTEGER_FIELDS, label="mask bin"
    )
    _validate_prepared_exact_floats(
        item, _PREPARED_MASK_BIN_FLOAT_FIELDS, label="mask bin"
    )
    _validate_prepared_digest_fields(
        item, _PREPARED_MASK_BIN_DIGEST_FIELDS, label="mask bin"
    )
    if not 0 <= item.phase_bin < 48:
        raise ValueError("phase_bin must be an exact integer in [0, 48)")
    for field_name in _PREPARED_MASK_BIN_FLOAT_FIELDS:
        value = getattr(item, field_name)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{field_name} must be an exact finite nonnegative float"
            )
    if item.masked_fraction > 1.0:
        raise ValueError("masked_fraction must lie in [0, 1]")
    for field_name in (
        "expected_units",
        "baseline_units",
        "observed_units",
        "induced_deleted_units",
    ):
        value = getattr(item, field_name)
        if value != float(round(value)):
            raise ValueError(f"{field_name} must contain whole-record units")
    if item.baseline_units > item.expected_units:
        raise ValueError("baseline units exceed frozen expected units")
    if item.observed_units > item.baseline_units:
        raise ValueError("mask bin geometry observed units exceed baseline units")
    if item.induced_deleted_units != item.baseline_units - item.observed_units:
        raise ValueError("induced deletion geometry is inconsistent")
    expected_fraction = (
        item.induced_deleted_units / item.expected_units
        if item.expected_units > 0.0
        else 0.0
    )
    if not np.isclose(
        item.masked_fraction, expected_fraction, rtol=0.0, atol=1e-15
    ):
        raise ValueError("masked_fraction is not deleted valid units / expected units")
    if item.content_digest != _prepared_mask_bin_digest_values(
        phase_bin=item.phase_bin,
        expected_units=item.expected_units,
        baseline_units=item.baseline_units,
        observed_units=item.observed_units,
        induced_deleted_units=item.induced_deleted_units,
        masked_fraction=item.masked_fraction,
    ):
        raise ValueError("mask bin content digest is invalid")
    return item


def _prepared_mask_bin_payload(item: PreparedMaskBinGeometry) -> dict[str, object]:
    return _prepared_mask_bin_payload_values(
        phase_bin=item.phase_bin,
        expected_units=item.expected_units,
        baseline_units=item.baseline_units,
        observed_units=item.observed_units,
        induced_deleted_units=item.induced_deleted_units,
        masked_fraction=item.masked_fraction,
    )


def _construct_mask_bin(**values: object) -> PreparedMaskBinGeometry:
    return _construct_prepared_mask_bin(
        PreparedMaskBinGeometry,
        **values,
        content_digest=_prepared_mask_bin_digest_values(**values),
    )


@dataclass(frozen=True)
class PreparedMaskGeometry:
    """Immutable mask identity, deleted record IDs, and all-48-bin geometry."""

    calendar_identity: PreparedCalendarIdentity
    primitive_name: str
    operation: str
    phase_view_digest: str
    calendar_digest: str
    source_record_digest: str
    definition_digest: str
    policy_digest: str
    scenario: str
    draw: int
    root_seed: str
    seed: str
    mask_key_digest: str
    mask_digest: str
    intervals: tuple[PreparedMaskInterval, ...]
    deleted_record_ids: tuple[int, ...]
    bins: tuple[PreparedMaskBinGeometry, ...]
    critical_window_expected_units: float
    critical_window_baseline_units: float
    critical_window_observed_units: float
    critical_window_induced_deleted_units: float
    critical_window_observed_fraction: float
    schema_digest: str
    content_digest: str
    _build_token: InitVar[object]

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_prepared_mask_geometry_token(_build_token):
            raise TypeError(
                "PreparedMaskGeometry must be created by the trusted geometry boundary"
            )
        _validate_prepared_mask_geometry_structure(self)


def _prepared_mask_geometry_payload_values(
    *,
    calendar_identity: PreparedCalendarIdentity,
    primitive_name: str,
    operation: str,
    phase_view_digest: str,
    calendar_digest: str,
    source_record_digest: str,
    definition_digest: str,
    policy_digest: str,
    scenario: str,
    draw: int,
    root_seed: str,
    seed: str,
    mask_key_digest: str,
    mask_digest: str,
    intervals: tuple[PreparedMaskInterval, ...],
    deleted_record_ids: tuple[int, ...],
    bins: tuple[PreparedMaskBinGeometry, ...],
    critical_window_expected_units: float,
    critical_window_baseline_units: float,
    critical_window_observed_units: float,
    critical_window_induced_deleted_units: float,
    critical_window_observed_fraction: float,
    schema_digest: str,
) -> dict[str, object]:
    return {
        "schema": _PREPARED_MASK_GEOMETRY_SCHEMA,
        "calendar_identity": calendar_identity.content_digest,
        "primitive_name": primitive_name,
        "operation": operation,
        "phase_view_digest": phase_view_digest,
        "calendar_digest": calendar_digest,
        "source_record_digest": source_record_digest,
        "definition_digest": definition_digest,
        "policy_digest": policy_digest,
        "scenario": scenario,
        "draw": draw,
        "root_seed": root_seed,
        "seed": seed,
        "mask_key_digest": mask_key_digest,
        "mask_digest": mask_digest,
        "intervals": [item.content_digest for item in intervals],
        "deleted_record_ids": list(deleted_record_ids),
        "bins": [item.content_digest for item in bins],
        "critical_window_expected_units": _prepared_float_token(
            critical_window_expected_units
        ),
        "critical_window_baseline_units": _prepared_float_token(
            critical_window_baseline_units
        ),
        "critical_window_observed_units": _prepared_float_token(
            critical_window_observed_units
        ),
        "critical_window_induced_deleted_units": _prepared_float_token(
            critical_window_induced_deleted_units
        ),
        "critical_window_observed_fraction": _prepared_float_token(
            critical_window_observed_fraction
        ),
        "schema_digest": schema_digest,
    }


def _prepared_mask_geometry_content_digest(geometry: PreparedMaskGeometry) -> str:
    return _prepared_sha256(
        _prepared_mask_geometry_payload_values(
            calendar_identity=geometry.calendar_identity,
            primitive_name=geometry.primitive_name,
            operation=geometry.operation,
            phase_view_digest=geometry.phase_view_digest,
            calendar_digest=geometry.calendar_digest,
            source_record_digest=geometry.source_record_digest,
            definition_digest=geometry.definition_digest,
            policy_digest=geometry.policy_digest,
            scenario=geometry.scenario,
            draw=geometry.draw,
            root_seed=geometry.root_seed,
            seed=geometry.seed,
            mask_key_digest=geometry.mask_key_digest,
            mask_digest=geometry.mask_digest,
            intervals=geometry.intervals,
            deleted_record_ids=geometry.deleted_record_ids,
            bins=geometry.bins,
            critical_window_expected_units=geometry.critical_window_expected_units,
            critical_window_baseline_units=geometry.critical_window_baseline_units,
            critical_window_observed_units=geometry.critical_window_observed_units,
            critical_window_induced_deleted_units=geometry.critical_window_induced_deleted_units,
            critical_window_observed_fraction=geometry.critical_window_observed_fraction,
            schema_digest=geometry.schema_digest,
        )
    )


def _validate_prepared_mask_geometry_structure(
    geometry: object,
) -> PreparedMaskGeometry:
    from .reliability import FROZEN_MASK_SCENARIOS

    if type(geometry) is not PreparedMaskGeometry:
        raise TypeError("geometry must be an exact PreparedMaskGeometry")
    _validate_prepared_exact_strings(
        geometry,
        _PREPARED_MASK_GEOMETRY_STRING_FIELDS,
        label="mask geometry",
    )
    _validate_prepared_exact_integers(
        geometry,
        _PREPARED_MASK_GEOMETRY_INTEGER_FIELDS,
        label="mask geometry",
    )
    _validate_prepared_exact_floats(
        geometry,
        _PREPARED_MASK_GEOMETRY_FLOAT_FIELDS,
        label="mask geometry",
    )
    _validate_prepared_digest_fields(
        geometry,
        _PREPARED_MASK_GEOMETRY_DIGEST_FIELDS,
        label="mask geometry",
    )
    _validate_prepared_calendar_identity(geometry.calendar_identity)
    if geometry.operation not in _PREPARED_PHASE_OPERATIONS:
        raise ValueError("geometry operation is not a frozen prepared operation")
    if geometry.scenario not in FROZEN_MASK_SCENARIOS:
        raise ValueError("geometry scenario is not frozen")
    if geometry.draw < 0:
        raise ValueError("draw must be an exact nonnegative integer")
    if geometry.calendar_digest != geometry.calendar_identity.content_digest:
        raise ValueError("geometry calendar digest is crosswired")
    if type(geometry.intervals) is not tuple or not geometry.intervals:
        raise ValueError("geometry intervals must be an exact nonempty tuple")
    for interval in geometry.intervals:
        _validate_prepared_mask_interval(interval)
    if tuple(item.mask_interval_id for item in geometry.intervals) != tuple(
        range(len(geometry.intervals))
    ):
        raise ValueError("mask interval IDs must be canonical")
    if any(
        current.mask_start < previous.mask_end
        for previous, current in zip(geometry.intervals, geometry.intervals[1:])
    ):
        raise ValueError("mask geometry intervals overlap")
    if type(geometry.deleted_record_ids) is not tuple or not all(
        type(value) is int and value >= 0
        for value in geometry.deleted_record_ids
    ):
        raise ValueError("deleted_record_ids must be exact nonnegative integers")
    if geometry.deleted_record_ids != tuple(
        sorted(set(geometry.deleted_record_ids))
    ):
        raise ValueError("deleted_record_ids must be unique and sorted")
    if type(geometry.bins) is not tuple or len(geometry.bins) != 48:
        raise ValueError("bins must contain exactly 48 typed geometries")
    for item in geometry.bins:
        _validate_prepared_mask_bin_geometry(item)
    if tuple(item.phase_bin for item in geometry.bins) != tuple(range(48)):
        raise ValueError("geometry bins must be in canonical all-day order")
    for field_name in _PREPARED_MASK_GEOMETRY_FLOAT_FIELDS:
        value = getattr(geometry, field_name)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{field_name} must be an exact finite nonnegative float"
            )
    if geometry.critical_window_observed_fraction > 1.0:
        raise ValueError("critical-window observed fraction must lie in [0, 1]")
    for field_name in (
        "critical_window_expected_units",
        "critical_window_baseline_units",
        "critical_window_observed_units",
        "critical_window_induced_deleted_units",
    ):
        value = getattr(geometry, field_name)
        if value != float(round(value)):
            raise ValueError(f"{field_name} must contain whole-record units")
    if (
        geometry.critical_window_baseline_units
        > geometry.critical_window_expected_units
    ):
        raise ValueError("critical-window baseline exceeds expected units")
    if (
        geometry.critical_window_observed_units
        > geometry.critical_window_baseline_units
    ):
        raise ValueError("critical-window observed exceeds baseline units")
    if geometry.critical_window_induced_deleted_units != (
        geometry.critical_window_baseline_units
        - geometry.critical_window_observed_units
    ):
        raise ValueError("critical-window deletion geometry is inconsistent")
    expected_fraction = (
        geometry.critical_window_observed_units
        / geometry.critical_window_expected_units
        if geometry.critical_window_expected_units > 0.0
        else 1.0
    )
    if not np.isclose(
        geometry.critical_window_observed_fraction,
        expected_fraction,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("critical-window observed fraction is inconsistent")
    expected_schema = _prepared_sha256(
        {"schema": _PREPARED_MASK_GEOMETRY_SCHEMA}
    )
    if geometry.schema_digest != expected_schema:
        raise ValueError("mask geometry schema digest is invalid")
    if geometry.content_digest != _prepared_mask_geometry_content_digest(geometry):
        raise ValueError("mask geometry content digest is invalid")
    return geometry


@dataclass(frozen=True)
class PreparedPrimitiveReplay:
    """One validated and aligned participant-day ready for repeated masks."""

    primitive_name: str
    sensor_file: str
    definition: PrimitiveDefinition
    lifelog_date: pd.Timestamp
    sleep_date: pd.Timestamp
    interval_boundary_policy: str
    source_record_digest: str
    source_schema_digest: str
    aligned_schema_digest: str
    aligned_column_topology: tuple[tuple[str, str], ...]
    aligned_frame_digest: str
    definition_digest: str
    policy_digest: str
    calendar_identity: PreparedCalendarIdentity | None
    prepared_content_digest: str
    _phase_view_cache: PreparedPhaseView | None
    source_timestamps: np.ndarray
    _aligned_frame: pd.DataFrame
    _target_positions: np.ndarray
    _record_starts: np.ndarray
    _record_ends: np.ndarray
    _build_token: InitVar[object]

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_prepared_replay_token(_build_token):
            raise TypeError(
                "PreparedPrimitiveReplay must be created by prepare_primitive_replay"
            )
        _validate_prepared_replay_structure(self)


def _validate_prepared_replay_structure(
    prepared: object,
) -> PreparedPrimitiveReplay:
    """Validate every stored replay field before digest, frame, or cache trust."""
    if type(prepared) is not PreparedPrimitiveReplay:
        raise TypeError("prepared must be an exact PreparedPrimitiveReplay")
    if "_build_token" in prepared.__dict__:
        raise ValueError("prepared replay must not retain its construction token")
    _validate_prepared_exact_strings(
        prepared, _PREPARED_REPLAY_STRING_FIELDS, label="prepared replay"
    )
    _validate_prepared_digest_fields(
        prepared, _PREPARED_REPLAY_DIGEST_FIELDS, label="prepared replay"
    )
    definition = _validate_prepared_primitive_definition_structure(
        prepared.definition
    )
    lifelog = _prepared_timestamp(prepared.lifelog_date, "lifelog_date")
    sleep = _prepared_timestamp(prepared.sleep_date, "sleep_date")
    if (
        lifelog != lifelog.normalize()
        or sleep != sleep.normalize()
        or sleep - lifelog != pd.Timedelta(days=1)
    ):
        raise ValueError("prepared replay dates are invalid")
    if prepared.primitive_name != definition.name:
        raise ValueError("prepared primitive name is crosswired")
    if prepared.sensor_file != definition.sensor_file:
        raise ValueError("prepared sensor file is crosswired")
    if prepared.interval_boundary_policy not in INTERVAL_BOUNDARY_POLICIES:
        raise ValueError("prepared interval boundary policy is invalid")
    topology = prepared.aligned_column_topology
    if type(topology) is not tuple or not topology:
        raise ValueError("prepared aligned topology must be a nonempty exact tuple")
    for item in topology:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not str
            or not item[1]
        ):
            raise ValueError(
                "prepared aligned topology entries must be exact string tuples"
            )
    topology_columns = tuple(item[0] for item in topology)
    if len(topology_columns) != len(set(topology_columns)):
        raise ValueError("prepared aligned topology columns must be unique")
    if prepared.calendar_identity is not None:
        identity = _validate_prepared_calendar_identity(prepared.calendar_identity)
        if identity.lifelog_date != lifelog or identity.sleep_date != sleep:
            raise ValueError("prepared replay dates are crosswired with its calendar")
    cache = prepared._phase_view_cache
    if cache is not None:
        if type(cache) is not PreparedPhaseView:
            raise ValueError("prepared phase-view cache must have exact type")
        validate_prepared_phase_view(cache)
        if (
            cache.primitive_name != prepared.primitive_name
            or cache.sensor_file != prepared.sensor_file
            or cache.calendar_identity != prepared.calendar_identity
            or cache.definition_digest != prepared.definition_digest
            or cache.policy_digest != prepared.policy_digest
            or cache.source_schema_digest != prepared.source_schema_digest
            or cache.aligned_schema_digest != prepared.aligned_schema_digest
            or cache.aligned_column_topology != topology
            or cache.aligned_frame_digest != prepared.aligned_frame_digest
            or cache.source_record_digest != prepared.source_record_digest
        ):
            raise ValueError("prepared phase-view cache is crosswired")
    frame = prepared._aligned_frame
    if type(frame) is not pd.DataFrame:
        raise TypeError("prepared aligned frame must be an exact pandas DataFrame")
    if frame.columns.duplicated().any() or not all(
        type(column) is str and column for column in frame.columns
    ):
        raise ValueError("prepared aligned frame columns must be unique exact strings")

    array_contracts = (
        ("source_timestamps", prepared.source_timestamps, np.dtype("datetime64[ns]")),
        ("target_positions", prepared._target_positions, np.dtype(np.int64)),
        ("record_starts", prepared._record_starts, np.dtype("datetime64[ns]")),
        ("record_ends", prepared._record_ends, np.dtype("datetime64[ns]")),
    )
    for name, array, expected_dtype in array_contracts:
        if type(array) is not np.ndarray:
            raise TypeError(f"prepared {name} must be an exact NumPy array")
        if array.ndim != 1 or array.dtype != expected_dtype:
            raise ValueError(f"prepared {name} dtype or shape is invalid")
        if array.flags.writeable:
            raise ValueError(f"prepared {name} must be read-only")
    if len(prepared.source_timestamps) != len(prepared._target_positions):
        raise ValueError("prepared target timestamp topology is inconsistent")
    if len(prepared._record_starts) != len(frame) or len(prepared._record_ends) != len(frame):
        raise ValueError("prepared record interval topology is inconsistent")
    positions = prepared._target_positions
    if (
        (positions < 0).any()
        or (positions >= len(frame)).any()
        or len(np.unique(positions)) != len(positions)
        or (len(positions) > 1 and not np.all(positions[1:] > positions[:-1]))
    ):
        raise ValueError("prepared target positions must be unique ordered indices")
    return prepared


@dataclass(frozen=True)
class PreparedPrimitiveReplayResult:
    """Replay output only; it is never accepted as prepared authority input."""

    value: float
    quality: dict[str, Any]
    original_count: int
    deleted_count: int
    retained_count: int


def _prepared_frame_schema_digest(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    schema: str,
) -> str:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"prepared schema is missing columns: {missing}")
    return _prepared_sha256(
        {
            "schema": schema,
            "columns": [
                [column, str(frame[column].dtype)] for column in columns
            ],
        }
    )


def _prepared_aligned_column_topology(
    frame: pd.DataFrame,
) -> tuple[tuple[str, str], ...]:
    if type(frame) is not pd.DataFrame:
        raise TypeError("prepared aligned frame must be an exact pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("prepared aligned frame columns must be unique")
    if not all(type(column) is str for column in frame.columns):
        raise ValueError("prepared aligned frame columns must be exact strings")
    return tuple((column, str(frame[column].dtype)) for column in frame.columns)


def _prepared_aligned_frame_digest(
    frame: pd.DataFrame,
    topology: tuple[tuple[str, str], ...],
) -> str:
    if _prepared_aligned_column_topology(frame) != topology:
        raise ValueError("prepared aligned frame topology does not match")
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("prepared aligned frame index must be canonical")
    try:
        row_hashes = pd.util.hash_pandas_object(
            frame,
            index=True,
            categorize=False,
        ).to_numpy(dtype="<u8", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("prepared aligned frame cannot be canonically hashed") from error
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": "prepared-complete-aligned-frame-v1",
                "row_count": len(frame),
                "columns": [list(item) for item in topology],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(row_hashes, dtype="<u8").tobytes())
    return digest.hexdigest()


def _prepared_operation_params(
    definition: PrimitiveDefinition,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (name, type(value).__name__, repr(value))
        for name, value in definition.operation_params
    )


def _target_frame_for_prepared_view(
    definition: PrimitiveDefinition,
    aligned: pd.DataFrame,
    lifelog_date: pd.Timestamp,
    sleep_date: pd.Timestamp,
    interval_boundary_policy: str,
) -> pd.DataFrame:
    required = [
        "subject_id",
        "source_timestamp",
        "support_start",
        "support_end",
        "effective_timestamp",
        definition.value_column,
    ]
    if definition.validity_column is not None:
        required.append(definition.validity_column)
    missing = [column for column in required if column not in aligned.columns]
    if missing:
        raise ValueError(f"prepared aligned source is missing columns: {missing}")
    canonical = aligned.copy(deep=True)
    if canonical.duplicated(["subject_id", "source_timestamp"]).any():
        raise ValueError("prepared aligned source contains a duplicate canonical key")
    canonical = canonical.sort_values(
        ["subject_id", "source_timestamp"], kind="stable", na_position="last"
    ).reset_index(drop=True)
    window_start, window_end = _window_bounds(
        lifelog_date, sleep_date, definition.window
    )
    if definition.timestamp_semantics == "instant":
        target = canonical["source_timestamp"].ge(window_start) & canonical[
            "source_timestamp"
        ].lt(window_end)
    else:
        target = canonical["support_start"].ge(window_start) & canonical[
            "support_end"
        ].le(window_end)
        if interval_boundary_policy == "strict_source_availability":
            target &= canonical["source_timestamp"].lt(window_end)
    return canonical.loc[target].reset_index(drop=True)


def _prepared_numeric(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _prepared_phase_record_values(
    definition: PrimitiveDefinition,
    aligned: pd.DataFrame,
    lifelog_date: pd.Timestamp,
    sleep_date: pd.Timestamp,
    interval_boundary_policy: str,
) -> tuple[dict[str, object], ...]:
    target = _target_frame_for_prepared_view(
        definition,
        aligned,
        lifelog_date,
        sleep_date,
        interval_boundary_policy,
    )
    params = dict(definition.operation_params)
    records: list[dict[str, object]] = []
    for record_id, row in enumerate(target.itertuples(index=False)):
        source_timestamp = pd.Timestamp(getattr(row, "source_timestamp"))
        support_start = pd.Timestamp(getattr(row, "support_start"))
        support_end = pd.Timestamp(getattr(row, "support_end"))
        effective_timestamp = pd.Timestamp(getattr(row, "effective_timestamp"))
        raw_value = _prepared_numeric(getattr(row, definition.value_column))
        raw_validity = (
            None
            if definition.validity_column is None
            else _prepared_numeric(getattr(row, definition.validity_column))
        )
        operation = definition.operation
        valid = False
        transformed = 0.0
        if operation == "binary_load":
            valid = bool(np.isfinite(raw_value) and raw_value in (0.0, 1.0))
            transformed = raw_value * definition.unit_scale if valid else 0.0
        elif operation == "activity_load":
            validity_equals = float(params.get("validity_equals", 0))
            valid = bool(
                np.isfinite(raw_value)
                and raw_value in (0.0, 1.0)
                and raw_validity is not None
                and np.isfinite(raw_validity)
                and raw_validity == validity_equals
            )
            transformed = raw_value * definition.unit_scale if valid else 0.0
        elif operation == "additive_load":
            minimum = params.get("minimum")
            valid = bool(
                np.isfinite(raw_value)
                and (minimum is None or raw_value >= float(minimum))
            )
            transformed = raw_value * definition.unit_scale if valid else 0.0
        elif operation == "exposure_mean":
            minimum = params.get("minimum")
            valid = bool(
                np.isfinite(raw_value)
                and (minimum is None or raw_value >= float(minimum))
            )
            if valid:
                transform = params.get("transform")
                if transform == "log1p":
                    transformed = float(np.log1p(raw_value))
                elif transform is None:
                    transformed = raw_value
                else:  # pragma: no cover - frozen definitions make this unreachable.
                    raise ValueError(f"Unsupported exposure transform: {transform}")
                transformed *= definition.unit_scale
        elif operation == "evening_p90":
            minimum = float(params.get("minimum", 0.0))
            valid = bool(np.isfinite(raw_value) and raw_value >= minimum)
            if definition.validity_column is not None:
                validity_equals = float(params.get("validity_equals", 0))
                valid &= bool(
                    raw_validity is not None
                    and np.isfinite(raw_validity)
                    and raw_validity == validity_equals
                )
            transformed = raw_value * definition.unit_scale if valid else 0.0
        else:
            raise ValueError(
                f"Prepared phase views do not support operation: {operation}"
            )
        offset_minutes = float(
            (effective_timestamp - effective_timestamp.normalize())
            / pd.Timedelta(minutes=1)
        )
        phase_bin = int(np.floor(offset_minutes / 30.0))
        positive_event_count = int(operation == "evening_p90" and transformed > 0.0)
        event_mass = float(transformed if positive_event_count else 0.0)
        event_minute = float(offset_minutes) if positive_event_count else None
        records.append(
            {
                "record_id": int(record_id),
                "subject_id": getattr(row, "subject_id"),
                "source_timestamp": source_timestamp,
                "support_start": support_start,
                "support_end": support_end,
                "effective_timestamp": effective_timestamp,
                "phase_bin": phase_bin,
                "raw_value": float(raw_value),
                "raw_validity": (
                    None if raw_validity is None else float(raw_validity)
                ),
                "valid": bool(valid),
                "support_units": float(1.0 if valid else 0.0),
                "transformed_numerator": float(transformed if valid else 0.0),
                "positive_event_count": positive_event_count,
                "event_mass": event_mass,
                "event_minute": event_minute,
            }
        )
    return tuple(records)


def _prepared_phase_records(
    definition: PrimitiveDefinition,
    aligned: pd.DataFrame,
    lifelog_date: pd.Timestamp,
    sleep_date: pd.Timestamp,
    interval_boundary_policy: str,
) -> tuple[PreparedPhaseRecord, ...]:
    return tuple(
        _construct_phase_record(**values)
        for values in _prepared_phase_record_values(
            definition,
            aligned,
            lifelog_date,
            sleep_date,
            interval_boundary_policy,
        )
    )


def _prepared_view_values(
    *,
    calendar_identity: PreparedCalendarIdentity,
    definition: PrimitiveDefinition,
    sensor_file: str,
    interval_boundary_policy: str,
    source_schema_digest: str,
    aligned_schema_digest: str,
    aligned_column_topology: tuple[tuple[str, str], ...],
    aligned_frame_digest: str,
    source_record_digest: str,
    definition_digest: str,
    policy_digest: str,
    records: tuple[PreparedPhaseRecord, ...],
) -> dict[str, object]:
    return {
        "calendar_identity": calendar_identity,
        "primitive_name": definition.name,
        "sensor_file": sensor_file,
        "value_column": definition.value_column,
        "validity_column": definition.validity_column,
        "operation": definition.operation,
        "window": definition.window,
        "cadence_minutes": int(definition.cadence_minutes),
        "expected_epochs": int(definition.expected_epochs),
        "coverage_mode": definition.coverage_mode,
        "timestamp_semantics": definition.timestamp_semantics,
        "support_minutes": int(definition.support_minutes),
        "timing_anchor": definition.timing_anchor,
        "unit": definition.unit,
        "unit_scale": float(definition.unit_scale),
        "operation_params": _prepared_operation_params(definition),
        "interval_boundary_policy": interval_boundary_policy,
        "definition_digest": definition_digest,
        "policy_digest": policy_digest,
        "source_schema_digest": source_schema_digest,
        "aligned_schema_digest": aligned_schema_digest,
        "aligned_column_topology": aligned_column_topology,
        "aligned_frame_digest": aligned_frame_digest,
        "source_record_digest": source_record_digest,
        "schema_digest": _prepared_sha256({"schema": _PREPARED_PHASE_SCHEMA}),
        "records": records,
    }


def _prepared_replay_content_digest(
    *,
    calendar_identity: PreparedCalendarIdentity | None,
    definition_digest: str,
    policy_digest: str,
    source_schema_digest: str,
    aligned_schema_digest: str,
    aligned_column_topology: tuple[tuple[str, str], ...],
    aligned_frame_digest: str,
    source_record_digest: str,
    lifelog_date: pd.Timestamp,
    sleep_date: pd.Timestamp,
    primitive_name: str,
    sensor_file: str,
    record_count: int,
) -> str:
    return _prepared_sha256(
        {
            "schema": "prepared-replay-authority-v2",
            "calendar_digest": (
                None
                if calendar_identity is None
                else calendar_identity.content_digest
            ),
            "definition_digest": definition_digest,
            "policy_digest": policy_digest,
            "source_schema_digest": source_schema_digest,
            "aligned_schema_digest": aligned_schema_digest,
            "aligned_column_topology": [
                list(item) for item in aligned_column_topology
            ],
            "aligned_frame_digest": aligned_frame_digest,
            "source_record_digest": source_record_digest,
            "lifelog_date": lifelog_date.isoformat(),
            "sleep_date": sleep_date.isoformat(),
            "primitive_name": primitive_name,
            "sensor_file": sensor_file,
            "record_count": record_count,
        }
    )


@dataclass(frozen=True)
class _ValidatedPreparedReplayAuthority:
    target_positions: np.ndarray
    source_timestamps: np.ndarray
    record_starts: np.ndarray
    record_ends: np.ndarray
    target_frame: pd.DataFrame


def _prepared_datetime_array(series: pd.Series, name: str) -> np.ndarray:
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"prepared {name} values are invalid") from error
    if parsed.isna().any() or parsed.dt.tz is not None:
        raise ValueError(f"prepared {name} values must be timezone-naive timestamps")
    return parsed.to_numpy(dtype="datetime64[ns]", copy=True)


def _assert_prepared_array_exact(
    name: str,
    stored: object,
    expected: np.ndarray,
) -> None:
    if type(stored) is not np.ndarray:
        raise TypeError(f"prepared {name} must be an exact NumPy array")
    if stored.ndim != 1 or stored.dtype != expected.dtype or stored.shape != expected.shape:
        raise ValueError(f"prepared {name} dtype or shape does not match authority")
    if not np.array_equal(stored, expected):
        raise ValueError(f"prepared {name} values or ordering do not match authority")


def _validate_prepared_replay_authority(
    prepared: object,
) -> _ValidatedPreparedReplayAuthority:
    """Rebuild and bind every replay input before export or recomputation."""
    _validate_prepared_replay_structure(prepared)
    definition = prepared.definition
    definition_digest = _prepared_definition_digest(definition)
    if (
        prepared.primitive_name != definition.name
        or prepared.sensor_file != definition.sensor_file
        or prepared.definition_digest != definition_digest
    ):
        raise ValueError("prepared primitive definition is crosswired")
    if prepared.interval_boundary_policy not in INTERVAL_BOUNDARY_POLICIES:
        raise ValueError("prepared interval boundary policy is invalid")
    policy_digest = _prepared_policy_digest(prepared.interval_boundary_policy)
    if prepared.policy_digest != policy_digest:
        raise ValueError("prepared policy digest does not match")

    lifelog_date = _prepared_timestamp(prepared.lifelog_date, "lifelog_date")
    sleep_date = _prepared_timestamp(prepared.sleep_date, "sleep_date")
    if (
        lifelog_date != lifelog_date.normalize()
        or sleep_date != sleep_date.normalize()
        or sleep_date - lifelog_date != pd.Timedelta(days=1)
    ):
        raise ValueError("prepared replay dates are invalid")
    if prepared.calendar_identity is not None:
        identity = _validate_prepared_calendar_identity(prepared.calendar_identity)
        if (
            identity.lifelog_date != lifelog_date
            or identity.sleep_date != sleep_date
        ):
            raise ValueError("prepared replay dates are crosswired with its calendar")

    frame = prepared._aligned_frame
    topology = _prepared_aligned_column_topology(frame)
    bait = _forbidden_replay_columns(frame.columns)
    if bait:
        raise ValueError(f"prepared aligned frame contains outcome or oracle bait: {bait}")
    if topology != prepared.aligned_column_topology:
        raise ValueError("prepared aligned column topology does not match authority")
    if "subject_id" not in frame.columns:
        raise ValueError("prepared aligned frame is missing subject_id")
    aligned_subjects = tuple(frame["subject_id"].tolist())
    if not all(
        type(subject_id) is str and subject_id
        for subject_id in aligned_subjects
    ):
        raise ValueError(
            "prepared aligned subject_id values must be exact nonempty strings"
        )
    if prepared.calendar_identity is not None and not all(
        subject_id == prepared.calendar_identity.subject_id
        for subject_id in aligned_subjects
    ):
        raise ValueError("prepared aligned subject is crosswired with its calendar")
    aligned_schema_digest = _prepared_frame_schema_digest(
        frame,
        tuple(column for column, _dtype in topology),
        schema="prepared-aligned-source-schema-v2",
    )
    if aligned_schema_digest != prepared.aligned_schema_digest:
        raise ValueError("prepared aligned source schema digest does not match")
    aligned_frame_digest = _prepared_aligned_frame_digest(frame, topology)
    if aligned_frame_digest != prepared.aligned_frame_digest:
        raise ValueError(
            "prepared complete aligned source frame digest does not match"
        )

    required = {
        "subject_id",
        "source_timestamp",
        "support_start",
        "support_end",
        "effective_timestamp",
        definition.value_column,
    }
    if definition.validity_column is not None:
        required.add(definition.validity_column)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prepared aligned frame is missing columns: {missing}")
    if frame.duplicated(["subject_id", "source_timestamp"]).any():
        raise ValueError("prepared aligned frame contains a duplicate canonical key")
    canonical_order = frame.sort_values(
        ["subject_id", "source_timestamp"],
        kind="stable",
        na_position="last",
    ).index.to_numpy(dtype=np.int64, copy=True)
    if not np.array_equal(canonical_order, np.arange(len(frame), dtype=np.int64)):
        raise ValueError("prepared aligned frame is not in canonical order")

    source_all = _prepared_datetime_array(frame["source_timestamp"], "source_timestamp")
    if definition.timestamp_semantics == "instant":
        record_starts = source_all.copy()
        record_ends = source_all.copy()
        target = (source_all >= np.datetime64(lifelog_date)) & (
            source_all < np.datetime64(sleep_date)
        )
        if definition.window == "evening":
            target &= source_all >= np.datetime64(
                lifelog_date + pd.Timedelta(hours=18)
            )
    elif definition.timestamp_semantics == "interval_end":
        record_starts = _prepared_datetime_array(frame["support_start"], "support_start")
        record_ends = _prepared_datetime_array(frame["support_end"], "support_end")
        window_start, window_end = _window_bounds(
            lifelog_date, sleep_date, definition.window
        )
        target = (record_starts >= np.datetime64(window_start)) & (
            record_ends <= np.datetime64(window_end)
        )
        if prepared.interval_boundary_policy == "strict_source_availability":
            target &= source_all < np.datetime64(window_end)
    else:
        raise ValueError("prepared timestamp semantics are unsupported")

    target_positions = np.flatnonzero(target).astype(np.int64, copy=False)
    target_source_timestamps = source_all[target_positions].copy()
    target_frame = frame.iloc[target_positions].reset_index(drop=True)
    target_subjects = tuple(target_frame["subject_id"].tolist())
    if not all(
        type(subject_id) is str and subject_id for subject_id in target_subjects
    ):
        raise ValueError("prepared target subjects must be exact nonempty strings")
    if prepared.calendar_identity is not None and not all(
        subject_id == prepared.calendar_identity.subject_id
        for subject_id in target_subjects
    ):
        raise ValueError("prepared target subject is crosswired with its calendar")

    _assert_prepared_array_exact(
        "target positions", prepared._target_positions, target_positions
    )
    _assert_prepared_array_exact(
        "source timestamps", prepared.source_timestamps, target_source_timestamps
    )
    _assert_prepared_array_exact("record starts", prepared._record_starts, record_starts)
    _assert_prepared_array_exact("record ends", prepared._record_ends, record_ends)

    source_record_digest = _prepared_source_digest_frame(
        target_frame,
        definition,
        prepared.source_schema_digest,
        aligned_schema_digest,
    )
    if source_record_digest != prepared.source_record_digest:
        raise ValueError("prepared source record digest does not match")
    expected_content_digest = _prepared_replay_content_digest(
        calendar_identity=prepared.calendar_identity,
        definition_digest=definition_digest,
        policy_digest=policy_digest,
        source_schema_digest=prepared.source_schema_digest,
        aligned_schema_digest=aligned_schema_digest,
        aligned_column_topology=topology,
        aligned_frame_digest=aligned_frame_digest,
        source_record_digest=source_record_digest,
        lifelog_date=lifelog_date,
        sleep_date=sleep_date,
        primitive_name=prepared.primitive_name,
        sensor_file=prepared.sensor_file,
        record_count=len(target_positions),
    )
    if expected_content_digest != prepared.prepared_content_digest:
        raise ValueError("prepared replay content digest does not match")

    for array in (
        target_positions,
        target_source_timestamps,
        record_starts,
        record_ends,
    ):
        array.setflags(write=False)
    return _ValidatedPreparedReplayAuthority(
        target_positions=target_positions,
        source_timestamps=target_source_timestamps,
        record_starts=record_starts,
        record_ends=record_ends,
        target_frame=target_frame,
    )


def _construct_phase_view_from_values(values: dict[str, object]) -> PreparedPhaseView:
    content_digest = _prepared_sha256(_prepared_view_values_payload(**values))
    return _construct_prepared_view(
        PreparedPhaseView, **values, content_digest=content_digest
    )


def export_prepared_phase_view(
    prepared: PreparedPrimitiveReplay,
) -> PreparedPhaseView:
    """Export a verified tuple-only view without exposing replay internals."""
    _validate_prepared_replay_authority(prepared)
    if prepared.calendar_identity is None:
        raise ValueError(
            "prepared replay lacks the full canonical calendar identity required by PhaseGuard"
        )
    identity = prepared.calendar_identity
    definition_digest = prepared.definition_digest
    policy_digest = prepared.policy_digest
    aligned_schema_digest = prepared.aligned_schema_digest
    source_record_digest = prepared.source_record_digest
    if prepared._phase_view_cache is not None:
        if type(prepared._phase_view_cache) is not PreparedPhaseView:
            raise ValueError("prepared phase-view cache is invalid")
        validate_prepared_phase_view(prepared._phase_view_cache)
        if (
            prepared._phase_view_cache.source_record_digest
            != source_record_digest
            or prepared._phase_view_cache.calendar_identity != identity
            or prepared._phase_view_cache.definition_digest != definition_digest
            or prepared._phase_view_cache.policy_digest != policy_digest
            or prepared._phase_view_cache.aligned_column_topology
            != prepared.aligned_column_topology
            or prepared._phase_view_cache.aligned_frame_digest
            != prepared.aligned_frame_digest
        ):
            raise ValueError("prepared phase-view cache is crosswired")
        return prepared._phase_view_cache
    records = _prepared_phase_records(
        prepared.definition,
        prepared._aligned_frame,
        prepared.lifelog_date,
        prepared.sleep_date,
        prepared.interval_boundary_policy,
    )
    values = _prepared_view_values(
        calendar_identity=identity,
        definition=prepared.definition,
        sensor_file=prepared.sensor_file,
        interval_boundary_policy=prepared.interval_boundary_policy,
        source_schema_digest=prepared.source_schema_digest,
        aligned_schema_digest=aligned_schema_digest,
        aligned_column_topology=prepared.aligned_column_topology,
        aligned_frame_digest=prepared.aligned_frame_digest,
        source_record_digest=source_record_digest,
        definition_digest=definition_digest,
        policy_digest=policy_digest,
        records=records,
    )
    view = _construct_phase_view_from_values(values)
    validate_prepared_phase_view(view)
    object.__setattr__(prepared, "_phase_view_cache", view)
    return view


def prepare_primitive_replay(
    primitive_name: str,
    sensor_frame: pd.DataFrame,
    sensor_file: str,
    calendar_row: pd.DataFrame,
    *,
    interval_boundary_policy: str = "measurement_support",
) -> PreparedPrimitiveReplay:
    """Validate and align one canonical participant-day exactly once.

    This is the public preparation boundary for repeated masking.  It reuses
    the same alignment authority as :func:`extract_daily_primitives`, performs
    no sensor IO, and never mutates either caller input.
    """
    if interval_boundary_policy not in INTERVAL_BOUNDARY_POLICIES:
        raise ValueError(
            f"interval_boundary_policy must be one of {INTERVAL_BOUNDARY_POLICIES}"
        )
    definitions = {
        definition.name: definition for definition in PRIMITIVE_DEFINITIONS
    }
    if primitive_name not in definitions:
        raise ValueError(f"Unknown primitive: {primitive_name}")
    definition = definitions[primitive_name]
    _validate_prepared_primitive_definition_structure(definition)
    if sensor_file != definition.sensor_file:
        raise ValueError(
            f"{primitive_name} requires sensor file {definition.sensor_file}"
        )
    if not isinstance(sensor_frame, pd.DataFrame):
        raise TypeError("sensor_frame must be a pandas DataFrame")
    if sensor_frame.columns.duplicated().any():
        raise ValueError("sensor_frame columns must be unique")
    forbidden_bait = _forbidden_replay_columns(sensor_frame.columns)
    if forbidden_bait:
        raise ValueError(
            f"Replay sensor_frame contains outcome or oracle bait: {forbidden_bait}"
        )
    required_columns = ["subject_id", "timestamp", definition.value_column]
    if definition.validity_column is not None:
        required_columns.append(definition.validity_column)
    missing = [column for column in required_columns if column not in sensor_frame]
    if missing:
        raise ValueError(f"Missing canonical sensor columns: {missing}")
    for column in (
        definition.value_column,
        *(
            (definition.validity_column,)
            if definition.validity_column is not None
            else ()
        ),
    ):
        if (
            not pd.api.types.is_numeric_dtype(sensor_frame[column].dtype)
            or pd.api.types.is_bool_dtype(sensor_frame[column].dtype)
        ):
            raise ValueError(
                f"Canonical sensor column {column} must have a numeric dtype"
            )
    if not isinstance(calendar_row, pd.DataFrame) or len(calendar_row) != 1:
        raise ValueError("calendar_row must be a one-row pandas DataFrame")
    calendar_bait = _forbidden_replay_columns(calendar_row.columns)
    if calendar_bait:
        raise ValueError(
            f"Replay calendar_row contains outcome or oracle bait: {calendar_bait}"
        )

    scaffold = _load_label_scaffold(calendar_row)
    calendar_identity = _derive_prepared_calendar_identity(calendar_row, scaffold)
    sensor = sensor_frame.loc[:, required_columns].copy()
    if calendar_identity is not None:
        if not sensor["subject_id"].dropna().map(
            lambda value: type(value) is str
        ).all():
            raise ValueError("subject_id must contain exact string values")
    source_schema_digest = _prepared_frame_schema_digest(
        sensor,
        required_columns,
        schema="prepared-raw-source-schema-v1",
    )
    sensor["source_timestamp"] = _parse_naive_sensor_timestamps(
        sensor["timestamp"], sensor_file
    )
    valid_key = sensor["source_timestamp"].notna()
    if sensor.loc[valid_key].duplicated(["subject_id", "source_timestamp"]).any():
        raise ValueError("Replay sensor_frame contains a duplicate canonical sensor key")
    expected_subject = scaffold.iloc[0]["subject_id"]
    if not sensor.empty and not sensor["subject_id"].eq(expected_subject).any():
        raise ValueError("Replay sensor_frame does not contain the calendar subject")
    timestamp_semantics, support_minutes, timing_anchor = _temporal_contract(
        [definition], sensor_file
    )
    aligned = _align_sensor(
        sensor,
        sensor_file,
        scaffold,
        timestamp_semantics=timestamp_semantics,
        support_minutes=support_minutes,
        timing_anchor=timing_anchor,
    )
    if all(column in scaffold.columns for column in SAFE_SCAFFOLD_METADATA):
        day = scaffold.iloc[0]
        target_source_observed = bool(
            (
                sensor["subject_id"].eq(day["subject_id"])
                & sensor["source_timestamp"].dt.normalize().eq(day["lifelog_date"])
            ).any()
        )
        if target_source_observed and not bool(day["any_source_record_observed"]):
            raise ValueError(
                "Replay calendar flags contradict canonical provenance for source records"
            )
        if not aligned.empty and not bool(day["any_measurement_support_observed"]):
            raise ValueError(
                "Replay calendar flags contradict canonical provenance for measurement support"
            )
    aligned = aligned.sort_values(
        ["subject_id", "source_timestamp"], kind="stable", na_position="last"
    ).reset_index(drop=True)
    lifelog_date = scaffold.iloc[0]["lifelog_date"]
    sleep_date = scaffold.iloc[0]["sleep_date"]
    window_start, window_end = _window_bounds(
        lifelog_date, sleep_date, definition.window
    )
    source_timestamps = aligned["source_timestamp"]
    if definition.timestamp_semantics == "instant":
        record_starts = source_timestamps
        record_ends = source_timestamps
        target = source_timestamps.ge(window_start) & source_timestamps.lt(
            window_end
        )
    else:
        record_starts = aligned["support_start"]
        record_ends = aligned["support_end"]
        target = record_starts.ge(window_start) & record_ends.le(window_end)
        if interval_boundary_policy == "strict_source_availability":
            target &= source_timestamps.lt(window_end)
    target_positions = np.flatnonzero(target.to_numpy(dtype=bool))
    aligned_column_topology = _prepared_aligned_column_topology(aligned)
    aligned_schema_digest = _prepared_frame_schema_digest(
        aligned,
        tuple(column for column, _dtype in aligned_column_topology),
        schema="prepared-aligned-source-schema-v2",
    )
    aligned_frame_digest = _prepared_aligned_frame_digest(
        aligned, aligned_column_topology
    )
    target_frame = aligned.loc[target].reset_index(drop=True)
    if len(target_frame) != len(target_positions):
        raise RuntimeError("prepared phase record topology is inconsistent")
    source_record_digest = _prepared_source_digest_frame(
        target_frame, definition, source_schema_digest, aligned_schema_digest
    )
    definition_digest = _prepared_definition_digest(definition)
    policy_digest = _prepared_policy_digest(interval_boundary_policy)
    prepared_content_digest = _prepared_replay_content_digest(
        calendar_identity=calendar_identity,
        definition_digest=definition_digest,
        policy_digest=policy_digest,
        source_schema_digest=source_schema_digest,
        aligned_schema_digest=aligned_schema_digest,
        aligned_column_topology=aligned_column_topology,
        aligned_frame_digest=aligned_frame_digest,
        source_record_digest=source_record_digest,
        lifelog_date=lifelog_date,
        sleep_date=sleep_date,
        primitive_name=primitive_name,
        sensor_file=sensor_file,
        record_count=len(target_frame),
    )
    frozen_positions = target_positions.astype(np.int64, copy=True)
    frozen_source_timestamps = source_timestamps.iloc[
        target_positions
    ].to_numpy(dtype="datetime64[ns]", copy=True)
    frozen_starts = record_starts.to_numpy(dtype="datetime64[ns]", copy=True)
    frozen_ends = record_ends.to_numpy(dtype="datetime64[ns]", copy=True)
    frozen_positions.setflags(write=False)
    frozen_source_timestamps.setflags(write=False)
    frozen_starts.setflags(write=False)
    frozen_ends.setflags(write=False)
    return _construct_prepared_replay(
        PreparedPrimitiveReplay,
        primitive_name=primitive_name,
        sensor_file=sensor_file,
        definition=definition,
        lifelog_date=lifelog_date,
        sleep_date=sleep_date,
        interval_boundary_policy=interval_boundary_policy,
        source_record_digest=source_record_digest,
        source_schema_digest=source_schema_digest,
        aligned_schema_digest=aligned_schema_digest,
        aligned_column_topology=aligned_column_topology,
        aligned_frame_digest=aligned_frame_digest,
        definition_digest=definition_digest,
        policy_digest=policy_digest,
        calendar_identity=calendar_identity,
        prepared_content_digest=prepared_content_digest,
        _phase_view_cache=None,
        source_timestamps=frozen_source_timestamps,
        _aligned_frame=aligned.copy(deep=True),
        _target_positions=frozen_positions,
        _record_starts=frozen_starts,
        _record_ends=frozen_ends,
    )


def _prepared_mask_pairs(
    prepared: PreparedPrimitiveReplay,
    mask_intervals: object,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    if isinstance(mask_intervals, pd.DataFrame):
        if mask_intervals.columns.duplicated().any():
            raise ValueError("mask interval columns must be unique")
        missing = {"mask_start", "mask_end"}.difference(mask_intervals.columns)
        if missing:
            raise ValueError(
                f"mask intervals are missing columns: {sorted(missing)}"
            )
        raw_pairs = list(
            mask_intervals.loc[:, ["mask_start", "mask_end"]].itertuples(
                index=False, name=None
            )
        )
    else:
        try:
            raw_pairs = list(mask_intervals)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("mask intervals must contain start/end pairs") from error
    if not raw_pairs:
        raise ValueError("mask intervals must be nonempty")
    window_start, window_end = _window_bounds(
        prepared.lifelog_date,
        prepared.sleep_date,
        prepared.definition.window,
    )
    parsed: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
            raise ValueError("each mask interval must be a start/end pair")
        start, end = pd.Timestamp(raw_pair[0]), pd.Timestamp(raw_pair[1])
        if (
            pd.isna(start)
            or pd.isna(end)
            or start.tz is not None
            or end.tz is not None
            or start >= end
            or start < window_start
            or end > window_end
        ):
            raise ValueError("mask interval is invalid or outside the primitive window")
        parsed.append((start, end))
    parsed.sort(key=lambda pair: (pair[0], pair[1]))
    if any(
        current[0] < previous[1]
        for previous, current in zip(parsed, parsed[1:])
    ):
        raise ValueError("mask intervals must not overlap")
    return tuple(parsed)


def _canonical_prepared_mask_table(
    prepared: PreparedPrimitiveReplay,
    mask_intervals: object,
) -> pd.DataFrame:
    from .reliability import (
        FROZEN_MASK_SCENARIOS,
        MASK_TABLE_COLUMNS,
        derive_mask_seed,
        generate_mask_intervals,
    )

    if type(mask_intervals) is not pd.DataFrame or mask_intervals.empty:
        raise TypeError("mask_intervals must be an exact nonempty pandas DataFrame")
    if mask_intervals.columns.duplicated().any():
        raise ValueError("mask interval columns contain a duplicate")
    bait = _forbidden_replay_columns(mask_intervals.columns)
    if bait:
        raise ValueError(f"mask table contains outcome or oracle bait: {bait}")
    if tuple(mask_intervals.columns) != tuple(MASK_TABLE_COLUMNS):
        missing = sorted(set(MASK_TABLE_COLUMNS).difference(mask_intervals.columns))
        extra = sorted(set(mask_intervals.columns).difference(MASK_TABLE_COLUMNS))
        raise ValueError(
            f"mask table must match the exact frozen schema; missing={missing}, extra={extra}"
        )
    masks = mask_intervals.copy(deep=True)
    for name in ("sensor_day_id", "draw", "mask_interval_id"):
        if (
            not pd.api.types.is_integer_dtype(masks[name].dtype)
            or pd.api.types.is_bool_dtype(masks[name].dtype)
        ):
            raise ValueError(f"{name} must have an exact integer dtype")
    for name in (
        "subject_id",
        "primitive",
        "scenario",
        "root_seed",
        "seed",
        "mask_semantics",
        "empirical_policy",
        "status",
    ):
        if not masks[name].map(lambda value: type(value) is str).all():
            raise ValueError(f"{name} must contain exact string values")
    if (
        not pd.api.types.is_numeric_dtype(masks["mask_duration_minutes"].dtype)
        or pd.api.types.is_bool_dtype(masks["mask_duration_minutes"].dtype)
    ):
        raise ValueError("mask_duration_minutes must have a numeric dtype")
    for name in ("window_start", "window_end", "mask_start", "mask_end"):
        parsed: list[pd.Timestamp] = []
        for value in masks[name]:
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} contains an invalid timestamp") from error
            if pd.isna(timestamp) or timestamp.tz is not None:
                raise ValueError(f"{name} must contain timezone-naive timestamps")
            parsed.append(timestamp)
        masks[name] = pd.Series(parsed, index=masks.index)

    view = export_prepared_phase_view(prepared)
    identity = view.calendar_identity
    if not masks["subject_id"].eq(identity.subject_id).all():
        raise ValueError("mask subject identity is crosswired")
    if not masks["sensor_day_id"].eq(identity.sensor_day_id).all():
        raise ValueError("mask sensor_day_id identity is crosswired")
    if not masks["primitive"].eq(view.primitive_name).all():
        raise ValueError("mask primitive identity is crosswired")
    if not masks["scenario"].isin(FROZEN_MASK_SCENARIOS).all():
        raise ValueError("mask scenario is not frozen")
    identity_columns = (
        "subject_id",
        "sensor_day_id",
        "primitive",
        "scenario",
        "draw",
        "root_seed",
        "seed",
        "window_start",
        "window_end",
        "mask_semantics",
        "empirical_policy",
        "status",
    )
    if len(masks.loc[:, identity_columns].drop_duplicates()) != 1:
        raise ValueError("mask table contains more than one replay identity")
    if not masks["mask_semantics"].eq("half_open").all():
        raise ValueError("mask semantics must be half_open")
    empirical = masks["scenario"].eq("empirical_gap")
    if not masks["empirical_policy"].eq(
        np.where(empirical, "reject", "not_applicable")
    ).all():
        raise ValueError("mask empirical policy is crosswired")
    if not masks["status"].eq(
        np.where(empirical, "empirical_replay", "generated")
    ).all():
        raise ValueError("mask status is crosswired")
    window_start, window_end = _window_bounds(
        prepared.lifelog_date,
        prepared.sleep_date,
        prepared.definition.window,
    )
    if not masks["window_start"].eq(window_start).all() or not masks[
        "window_end"
    ].eq(window_end).all():
        raise ValueError("mask window is crosswired with the prepared primitive")
    if not (
        masks["mask_start"].ge(window_start)
        & masks["mask_end"].le(window_end)
        & masks["mask_start"].lt(masks["mask_end"])
    ).all():
        raise ValueError("mask interval is invalid or outside its window")
    # Match the reviewed public mask-table audit (scalar ``total_seconds``),
    # while the immutable interval below retains exact nanosecond endpoints.
    actual_duration = pd.Series(
        [
            float((end - start).total_seconds() / 60.0)
            for start, end in zip(masks["mask_start"], masks["mask_end"])
        ],
        index=masks.index,
    )
    claimed_duration = masks["mask_duration_minutes"].astype(float)
    if not np.allclose(
        actual_duration.to_numpy(dtype=float),
        claimed_duration.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("mask duration does not match endpoints")
    if masks["mask_interval_id"].duplicated().any():
        raise ValueError("mask interval IDs contain a duplicate")
    masks = masks.sort_values(
        ["mask_start", "mask_end", "mask_interval_id"], kind="stable"
    ).reset_index(drop=True)
    if not masks["mask_interval_id"].eq(
        pd.Series(np.arange(len(masks), dtype=int))
    ).all():
        raise ValueError("mask interval IDs do not match canonical order")
    if any(
        current < previous
        for previous, current in zip(
            masks["mask_end"].iloc[:-1], masks["mask_start"].iloc[1:]
        )
    ):
        raise ValueError("mask intervals overlap")
    first = masks.iloc[0]
    expected_seed = derive_mask_seed(
        first["root_seed"],
        first["scenario"],
        int(first["draw"]),
        identity.subject_id,
        identity.sensor_day_id,
        view.primitive_name,
    )
    if first["seed"] != expected_seed:
        raise ValueError("mask seed does not match its frozen key")
    if not bool(empirical.iloc[0]):
        if len(masks) != 1:
            raise ValueError("generated mask scenarios require one interval")
        regenerated = generate_mask_intervals(
            view.primitive_name,
            subject_id=identity.subject_id,
            sensor_day_id=identity.sensor_day_id,
            lifelog_date=identity.lifelog_date,
            scenario=first["scenario"],
            draw=int(first["draw"]),
            root_seed=first["root_seed"],
        ).iloc[0]
        for name in (
            "seed",
            "mask_start",
            "mask_end",
            "mask_duration_minutes",
        ):
            if first[name] != regenerated[name]:
                raise ValueError("generated mask does not match deterministic regeneration")
    return masks


def _prepared_mask_key_payload(
    *,
    view: PreparedPhaseView,
    scenario: str,
    draw: int,
    root_seed: str,
    seed: str,
) -> dict[str, object]:
    return {
        "schema": "prepared-mask-key-v1",
        "calendar_digest": view.calendar_identity.content_digest,
        "subject_id": view.calendar_identity.subject_id,
        "sensor_day_id": view.calendar_identity.sensor_day_id,
        "primitive": view.primitive_name,
        "scenario": scenario,
        "draw": draw,
        "root_seed": root_seed,
        "seed": seed,
        "mask_semantics": "half_open",
    }


def _prepared_full_mask_digest(
    mask_key_digest: str,
    intervals: tuple[PreparedMaskInterval, ...],
) -> str:
    return _prepared_sha256(
        {
            "schema": "prepared-mask-content-v1",
            "mask_key_digest": mask_key_digest,
            "intervals": [
                _prepared_mask_interval_payload(item) for item in intervals
            ],
        }
    )


def _prepared_deleted_record_ids(
    view: PreparedPhaseView,
    intervals: tuple[PreparedMaskInterval, ...],
) -> tuple[int, ...]:
    deleted: list[int] = []
    for record in view.records:
        if view.timestamp_semantics == "instant":
            is_deleted = any(
                record.source_timestamp >= interval.mask_start
                and record.source_timestamp < interval.mask_end
                for interval in intervals
            )
        else:
            is_deleted = any(
                record.support_start < interval.mask_end
                and record.support_end > interval.mask_start
                for interval in intervals
            )
        if is_deleted:
            deleted.append(record.record_id)
    return tuple(deleted)


def _prepared_geometry_bins(
    view: PreparedPhaseView,
    deleted_record_ids: tuple[int, ...],
) -> tuple[PreparedMaskBinGeometry, ...]:
    deleted = set(deleted_record_ids)
    expected_per_active_bin = float(30 // view.cadence_minutes)
    active_bins = set(range(48)) if view.window == "day" else set(range(36, 48))
    bins: list[PreparedMaskBinGeometry] = []
    for phase_bin in range(48):
        records = [record for record in view.records if record.phase_bin == phase_bin]
        baseline = float(sum(record.support_units for record in records))
        observed = float(
            sum(
                record.support_units
                for record in records
                if record.record_id not in deleted
            )
        )
        induced = float(baseline - observed)
        expected = expected_per_active_bin if phase_bin in active_bins else 0.0
        masked_fraction = induced / expected if expected > 0.0 else 0.0
        bins.append(
            _construct_mask_bin(
                phase_bin=phase_bin,
                expected_units=float(expected),
                baseline_units=baseline,
                observed_units=observed,
                induced_deleted_units=induced,
                masked_fraction=float(masked_fraction),
            )
        )
    return tuple(bins)


def _prepared_critical_geometry(
    view: PreparedPhaseView,
    bins: tuple[PreparedMaskBinGeometry, ...],
) -> tuple[float, float, float, float, float]:
    if view.operation != "evening_p90":
        return (0.0, 0.0, 0.0, 0.0, 1.0)
    critical = bins[36:48]
    expected = float(sum(item.expected_units for item in critical))
    baseline = float(sum(item.baseline_units for item in critical))
    observed = float(sum(item.observed_units for item in critical))
    induced = float(sum(item.induced_deleted_units for item in critical))
    observed_fraction = observed / expected if expected > 0.0 else 0.0
    return expected, baseline, observed, induced, float(observed_fraction)


def build_prepared_mask_geometry(
    prepared: PreparedPrimitiveReplay,
    mask_intervals: pd.DataFrame,
) -> PreparedMaskGeometry:
    """Build exact whole-record deletion geometry from one prepared authority."""
    if type(prepared) is not PreparedPrimitiveReplay:
        raise TypeError("prepared must be an exact PreparedPrimitiveReplay")
    view = export_prepared_phase_view(prepared)
    masks = _canonical_prepared_mask_table(prepared, mask_intervals)
    intervals = tuple(
        _construct_mask_interval(
            mask_interval_id=int(row.mask_interval_id),
            mask_start=pd.Timestamp(row.mask_start),
            mask_end=pd.Timestamp(row.mask_end),
            duration_minutes=float(
                (pd.Timestamp(row.mask_end) - pd.Timestamp(row.mask_start))
                / pd.Timedelta(minutes=1)
            ),
        )
        for row in masks.itertuples(index=False)
    )
    row = masks.iloc[0]
    scenario = str(row["scenario"])
    draw = int(row["draw"])
    root_seed = str(row["root_seed"])
    seed = str(row["seed"])
    mask_key_digest = _prepared_sha256(
        _prepared_mask_key_payload(
            view=view,
            scenario=scenario,
            draw=draw,
            root_seed=root_seed,
            seed=seed,
        )
    )
    mask_digest = _prepared_full_mask_digest(mask_key_digest, intervals)
    deleted_record_ids = _prepared_deleted_record_ids(view, intervals)
    bins = _prepared_geometry_bins(view, deleted_record_ids)
    critical = _prepared_critical_geometry(view, bins)
    values: dict[str, object] = {
        "calendar_identity": view.calendar_identity,
        "primitive_name": view.primitive_name,
        "operation": view.operation,
        "phase_view_digest": view.content_digest,
        "calendar_digest": view.calendar_identity.content_digest,
        "source_record_digest": view.source_record_digest,
        "definition_digest": view.definition_digest,
        "policy_digest": view.policy_digest,
        "scenario": scenario,
        "draw": draw,
        "root_seed": root_seed,
        "seed": seed,
        "mask_key_digest": mask_key_digest,
        "mask_digest": mask_digest,
        "intervals": intervals,
        "deleted_record_ids": deleted_record_ids,
        "bins": bins,
        "critical_window_expected_units": critical[0],
        "critical_window_baseline_units": critical[1],
        "critical_window_observed_units": critical[2],
        "critical_window_induced_deleted_units": critical[3],
        "critical_window_observed_fraction": critical[4],
        "schema_digest": _prepared_sha256(
            {"schema": _PREPARED_MASK_GEOMETRY_SCHEMA}
        ),
    }
    content_digest = _prepared_sha256(
        _prepared_mask_geometry_payload_values(**values)
    )
    geometry = _construct_prepared_mask_geometry(
        PreparedMaskGeometry, **values, content_digest=content_digest
    )
    validate_prepared_mask_geometry(geometry, view)
    return geometry


def validate_prepared_mask_geometry(
    geometry: PreparedMaskGeometry,
    view: PreparedPhaseView,
) -> None:
    """Rebuild all geometry from immutable public records and intervals."""
    from .reliability import derive_mask_seed

    _validate_prepared_mask_geometry_structure(geometry)
    validate_prepared_phase_view(view)
    if (
        geometry.calendar_identity != view.calendar_identity
        or geometry.calendar_digest != view.calendar_identity.content_digest
        or geometry.primitive_name != view.primitive_name
        or geometry.operation != view.operation
        or geometry.phase_view_digest != view.content_digest
        or geometry.source_record_digest != view.source_record_digest
        or geometry.definition_digest != view.definition_digest
        or geometry.policy_digest != view.policy_digest
    ):
        raise ValueError("mask geometry is crosswired with its prepared phase view")
    expected_seed = derive_mask_seed(
        geometry.root_seed,
        geometry.scenario,
        geometry.draw,
        view.calendar_identity.subject_id,
        view.calendar_identity.sensor_day_id,
        view.primitive_name,
    )
    if geometry.seed != expected_seed:
        raise ValueError("mask geometry seed does not match its key")
    expected_key_digest = _prepared_sha256(
        _prepared_mask_key_payload(
            view=view,
            scenario=geometry.scenario,
            draw=geometry.draw,
            root_seed=geometry.root_seed,
            seed=geometry.seed,
        )
    )
    if geometry.mask_key_digest != expected_key_digest:
        raise ValueError("mask key digest does not match")
    if geometry.mask_digest != _prepared_full_mask_digest(
        geometry.mask_key_digest, geometry.intervals
    ):
        raise ValueError("mask digest does not match interval content")
    expected_deleted = _prepared_deleted_record_ids(view, geometry.intervals)
    if geometry.deleted_record_ids != expected_deleted:
        raise ValueError("deleted record IDs do not match public overlap semantics")
    expected_bins = _prepared_geometry_bins(view, expected_deleted)
    if geometry.bins != expected_bins:
        raise ValueError("mask bin geometry does not match deleted valid record units")
    expected_critical = _prepared_critical_geometry(view, expected_bins)
    actual_critical = (
        geometry.critical_window_expected_units,
        geometry.critical_window_baseline_units,
        geometry.critical_window_observed_units,
        geometry.critical_window_induced_deleted_units,
        geometry.critical_window_observed_fraction,
    )
    if actual_critical != expected_critical:
        raise ValueError("critical-window geometry does not match")


def recompute_prepared_primitive(
    prepared: PreparedPrimitiveReplay,
) -> PreparedPrimitiveReplayResult:
    """Dispatch the unmasked value through the frozen primitive authority."""
    authority = _validate_prepared_replay_authority(prepared)
    value, quality = _dispatch_primitive(
        prepared.definition,
        prepared._aligned_frame,
        prepared.lifelog_date,
        prepared.sleep_date,
        prepared.interval_boundary_policy,
    )
    original_count = int(len(authority.target_positions))
    return PreparedPrimitiveReplayResult(
        value=float(value),
        quality=dict(quality),
        original_count=original_count,
        deleted_count=0,
        retained_count=original_count,
    )


def recompute_prepared_primitive_batch(
    prepared: PreparedPrimitiveReplay,
    mask_batches: Sequence[object],
) -> tuple[PreparedPrimitiveReplayResult, ...]:
    """Replay many half-open mask geometries after one canonical alignment."""
    authority = _validate_prepared_replay_authority(prepared)
    results: list[PreparedPrimitiveReplayResult] = []
    target_positions = authority.target_positions
    record_starts = authority.record_starts
    record_ends = authority.record_ends
    for mask_intervals in mask_batches:
        pairs = _prepared_mask_pairs(prepared, mask_intervals)
        deleted = np.zeros(len(prepared._aligned_frame), dtype=bool)
        for start, end in pairs:
            start64 = np.datetime64(start.to_datetime64())
            end64 = np.datetime64(end.to_datetime64())
            if prepared.definition.timestamp_semantics == "instant":
                current = (
                    (record_starts[target_positions] >= start64)
                    & (record_starts[target_positions] < end64)
                )
            else:
                current = (
                    (record_starts[target_positions] < end64)
                    & (record_ends[target_positions] > start64)
                )
            deleted[target_positions] |= current
        replay_frame = prepared._aligned_frame.loc[~deleted].reset_index(drop=True)
        value, quality = _dispatch_primitive(
            prepared.definition,
            replay_frame,
            prepared.lifelog_date,
            prepared.sleep_date,
            prepared.interval_boundary_policy,
        )
        original_count = int(len(target_positions))
        deleted_count = int(deleted[target_positions].sum())
        results.append(
            PreparedPrimitiveReplayResult(
                value=float(value),
                quality=dict(quality),
                original_count=original_count,
                deleted_count=deleted_count,
                retained_count=original_count - deleted_count,
            )
        )
    return tuple(results)


def recompute_primitive_from_frame(
    primitive_name: str,
    sensor_frame: pd.DataFrame,
    sensor_file: str,
    calendar_row: pd.DataFrame,
    *,
    interval_boundary_policy: str = "measurement_support",
) -> tuple[float, dict[str, Any]]:
    """Recompute one primitive through the prepared public replay boundary."""
    prepared = prepare_primitive_replay(
        primitive_name,
        sensor_frame,
        sensor_file,
        calendar_row,
        interval_boundary_policy=interval_boundary_policy,
    )
    result = recompute_prepared_primitive(prepared)
    return result.value, result.quality


def _safe_output(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate output columns: {duplicates}")
    frame = _validate_sensor_scaffold_metadata(frame)
    if frame["row_id"].duplicated().any():
        raise ValueError("Output row_id keys must be unique")
    if frame.duplicated(["subject_id", "lifelog_date", "sleep_date"]).any():
        raise ValueError("Output participant-day keys must be unique")

    forbidden_exact = {
        "device_id", "bssid", "BSSID", "ble_address", "address",
        "latitude", "longitude", "gps_latitude", "gps_longitude",
    }
    forbidden = [
        column
        for column in frame.columns
        if column in forbidden_exact or re.fullmatch(r"[QS]\d+", str(column))
    ]
    if forbidden:
        raise ValueError(f"Forbidden target or raw identifier columns: {forbidden}")
    if "any_measurement_support_observed" in frame.columns:
        primitive_columns = [
            definition.name
            for definition in PRIMITIVE_DEFINITIONS
            if definition.name in frame.columns
        ]
        if primitive_columns:
            unsupported = ~frame["any_measurement_support_observed"]
            finite_primitive = frame[primitive_columns].apply(
                lambda values: pd.to_numeric(values, errors="coerce")
            ).notna().any(axis=1)
            if bool((unsupported & finite_primitive).any()):
                raise ValueError(
                    "A finite primitive cannot be emitted for a day with no "
                    "measurement support"
                )
    return frame


def extract_daily_primitives(
    labels: pd.DataFrame | str | PathLike[str],
    canonical_root: str | PathLike[str],
    *,
    interval_boundary_policy: str = "measurement_support",
) -> pd.DataFrame:
    """Extract coverage-aware daily primitives on the input calendar-day scaffold."""
    if interval_boundary_policy not in INTERVAL_BOUNDARY_POLICIES:
        raise ValueError(
            f"interval_boundary_policy must be one of {INTERVAL_BOUNDARY_POLICIES}"
        )
    scaffold = _load_label_scaffold(labels)
    records = scaffold.to_dict(orient="records")
    record_by_row = {int(record["row_id"]): record for record in records}
    definitions_by_file = _definitions_by_sensor_file()
    has_sensor_metadata = all(
        column in scaffold.columns for column in SAFE_SCAFFOLD_METADATA
    )
    source_observed_parts: list[pd.DataFrame] = []
    support_observed_parts: list[pd.DataFrame] = []

    root = Path(canonical_root)
    for sensor_file, definitions in definitions_by_file.items():
        required_columns: list[str] = []
        for definition in definitions:
            required_columns.append(definition.value_column)
            if definition.validity_column is not None:
                required_columns.append(definition.validity_column)
        timestamp_semantics, support_minutes, timing_anchor = _temporal_contract(
            definitions, sensor_file
        )
        sensor = _read_canonical_sensor(
            root, sensor_file, required_columns
        )
        if has_sensor_metadata:
            source_days, support_days = _sensor_observation_days(
                sensor,
                timestamp_semantics=timestamp_semantics,
                support_minutes=support_minutes,
                timing_anchor=timing_anchor,
            )
            source_observed_parts.append(source_days)
            support_observed_parts.append(support_days)
        aligned = _align_sensor(
            sensor,
            sensor_file,
            scaffold,
            timestamp_semantics=timestamp_semantics,
            support_minutes=support_minutes,
            timing_anchor=timing_anchor,
        )
        indices_by_row = aligned.groupby("row_id", sort=False).indices
        empty = aligned.iloc[0:0]

        for definition in definitions:
            for key in scaffold.itertuples(index=False):
                positions = indices_by_row.get(key.row_id)
                day_frame = empty if positions is None else aligned.iloc[positions]
                value, quality = _dispatch_primitive(
                    definition,
                    day_frame,
                    key.lifelog_date,
                    key.sleep_date,
                    interval_boundary_policy,
                )
                record = record_by_row[int(key.row_id)]
                record[definition.name] = value
                for quality_name, quality_value in quality.items():
                    record[f"{definition.name}__{quality_name}"] = quality_value

    if has_sensor_metadata:
        canonical_scaffold = _build_sensor_calendar_from_observations(
            source_observed_parts, support_observed_parts
        )
        _validate_sensor_metadata_provenance(scaffold, canonical_scaffold)

    result = pd.DataFrame.from_records(records)
    return _safe_output(result)


def _dispatch_fixed_budget(
    operation: str,
    frame: pd.DataFrame,
    value_column: str,
    expected_epochs: int,
    cadence_minutes: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> float:
    common = {
        "cadence_minutes": cadence_minutes,
        "window_start": window_start,
        "window_end": window_end,
    }
    if operation == "binary_load":
        value, _ = summarize_binary_load(frame, value_column, expected_epochs, **common)
    elif operation == "activity_load":
        value, _ = summarize_activity_load(
            frame,
            expected_epochs,
            value_column,
            "activity_unknown",
            validity_equals=0,
            **common,
        )
    elif operation == "additive_load":
        value, _ = summarize_additive_load(
            frame,
            value_column,
            expected_epochs,
            minimum=0.0,
            unit_scale=1.0,
            **common,
        )
    elif operation == "exposure_mean":
        value, _ = summarize_exposure_mean(
            frame,
            value_column,
            expected_epochs,
            minimum=0.0,
            transform="log1p",
            **common,
        )
    else:
        raise ValueError(f"Unsupported fixed-budget operation: {operation}")
    return float(value)


def build_fixed_budget_features(
    labels: pd.DataFrame | str | PathLike[str],
    canonical_root: str | PathLike[str],
) -> pd.DataFrame:
    """Build ten missingness-preserving controls from frozen raw sensor windows."""
    scaffold = _load_label_scaffold(labels)
    records = scaffold.to_dict(orient="records")
    record_by_row = {int(record["row_id"]): record for record in records}
    definitions_by_file: dict[str, list[tuple[Any, ...]]] = {}
    for definition in FIXED_BUDGET_DEFINITIONS:
        definitions_by_file.setdefault(str(definition[1]), []).append(definition)
    primitive_definitions_by_file = _definitions_by_sensor_file()
    has_sensor_metadata = all(
        column in scaffold.columns for column in SAFE_SCAFFOLD_METADATA
    )
    source_observed_parts: list[pd.DataFrame] = []
    support_observed_parts: list[pd.DataFrame] = []

    root = Path(canonical_root)
    sensor_files = (
        tuple(primitive_definitions_by_file)
        if has_sensor_metadata
        else tuple(definitions_by_file)
    )
    for sensor_file in sensor_files:
        definitions = definitions_by_file.get(sensor_file, [])
        required_columns = [str(definition[2]) for definition in definitions]
        if any(definition[6] == "activity_load" for definition in definitions):
            required_columns.append("activity_unknown")
        sensor = _read_canonical_sensor(root, sensor_file, required_columns)
        timestamp_semantics, support_minutes, timing_anchor = _temporal_contract(
            primitive_definitions_by_file[sensor_file], sensor_file
        )
        if has_sensor_metadata:
            source_days, support_days = _sensor_observation_days(
                sensor,
                timestamp_semantics=timestamp_semantics,
                support_minutes=support_minutes,
                timing_anchor=timing_anchor,
            )
            source_observed_parts.append(source_days)
            support_observed_parts.append(support_days)
        if not definitions:
            continue
        aligned = _align_sensor(
            sensor,
            sensor_file,
            scaffold,
            timestamp_semantics=timestamp_semantics,
            support_minutes=support_minutes,
            timing_anchor=timing_anchor,
        )
        indices_by_row = aligned.groupby("row_id", sort=False).indices
        empty = aligned.iloc[0:0]

        for (
            name,
            _sensor_file,
            value_column,
            start_minute,
            end_minute,
            cadence_minutes,
            operation,
        ) in definitions:
            expected_epochs = (int(end_minute) - int(start_minute)) // int(cadence_minutes)
            for key in scaffold.itertuples(index=False):
                positions = indices_by_row.get(key.row_id)
                day_frame = empty if positions is None else aligned.iloc[positions]
                window_start = key.lifelog_date + pd.Timedelta(minutes=int(start_minute))
                window_end = key.lifelog_date + pd.Timedelta(minutes=int(end_minute))
                timestamps = day_frame["timestamp"]
                window_frame = day_frame.loc[
                    timestamps.ge(window_start) & timestamps.lt(window_end)
                ]
                record_by_row[int(key.row_id)][str(name)] = _dispatch_fixed_budget(
                    str(operation),
                    window_frame,
                    str(value_column),
                    expected_epochs,
                    int(cadence_minutes),
                    window_start,
                    window_end,
                )

    if has_sensor_metadata:
        canonical_scaffold = _build_sensor_calendar_from_observations(
            source_observed_parts, support_observed_parts
        )
        _validate_sensor_metadata_provenance(scaffold, canonical_scaffold)

    result = pd.DataFrame.from_records(records)
    expected_columns = [*OUTPUT_KEY_COLUMNS, *(item[0] for item in FIXED_BUDGET_DEFINITIONS)]
    result = result.loc[:, expected_columns]
    return _safe_output(result)
