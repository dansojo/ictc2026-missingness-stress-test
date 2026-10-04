"""Pure, coverage-aware summaries for daily semantic indicator primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
    forbidden_tokens = ("outcome", "target", "label", "oracle")
    return [
        column
        for column in columns
        if re.fullmatch(r"[qs]\d+", str(column).casefold())
        or any(token in str(column).casefold() for token in forbidden_tokens)
    ]


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
    source_timestamps: np.ndarray
    _aligned_frame: pd.DataFrame
    _target_positions: np.ndarray
    _record_starts: np.ndarray
    _record_ends: np.ndarray


@dataclass(frozen=True)
class PreparedPrimitiveReplayResult:
    value: float
    quality: dict[str, Any]
    original_count: int
    deleted_count: int
    retained_count: int


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
    if not isinstance(calendar_row, pd.DataFrame) or len(calendar_row) != 1:
        raise ValueError("calendar_row must be a one-row pandas DataFrame")
    calendar_bait = _forbidden_replay_columns(calendar_row.columns)
    if calendar_bait:
        raise ValueError(
            f"Replay calendar_row contains outcome or oracle bait: {calendar_bait}"
        )

    scaffold = _load_label_scaffold(calendar_row)
    sensor = sensor_frame.loc[:, required_columns].copy()
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
    source_payload = [
        [record_id, pd.Timestamp(source_timestamps.iloc[position]).isoformat()]
        for record_id, position in enumerate(target_positions)
    ]
    source_record_digest = hashlib.sha256(
        json.dumps(
            source_payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
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
    return PreparedPrimitiveReplay(
        primitive_name=primitive_name,
        sensor_file=sensor_file,
        definition=definition,
        lifelog_date=lifelog_date,
        sleep_date=sleep_date,
        interval_boundary_policy=interval_boundary_policy,
        source_record_digest=source_record_digest,
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


def recompute_prepared_primitive(
    prepared: PreparedPrimitiveReplay,
) -> PreparedPrimitiveReplayResult:
    """Dispatch the unmasked value through the frozen primitive authority."""
    if not isinstance(prepared, PreparedPrimitiveReplay):
        raise TypeError("prepared must be a PreparedPrimitiveReplay")
    value, quality = _dispatch_primitive(
        prepared.definition,
        prepared._aligned_frame,
        prepared.lifelog_date,
        prepared.sleep_date,
        prepared.interval_boundary_policy,
    )
    original_count = int(len(prepared._target_positions))
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
    if not isinstance(prepared, PreparedPrimitiveReplay):
        raise TypeError("prepared must be a PreparedPrimitiveReplay")
    results: list[PreparedPrimitiveReplayResult] = []
    target_positions = prepared._target_positions
    for mask_intervals in mask_batches:
        pairs = _prepared_mask_pairs(prepared, mask_intervals)
        deleted = np.zeros(len(prepared._aligned_frame), dtype=bool)
        for start, end in pairs:
            start64 = np.datetime64(start.to_datetime64())
            end64 = np.datetime64(end.to_datetime64())
            if prepared.definition.timestamp_semantics == "instant":
                current = (
                    (prepared._record_starts[target_positions] >= start64)
                    & (prepared._record_starts[target_positions] < end64)
                )
            else:
                current = (
                    (prepared._record_starts[target_positions] < end64)
                    & (prepared._record_ends[target_positions] > start64)
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
