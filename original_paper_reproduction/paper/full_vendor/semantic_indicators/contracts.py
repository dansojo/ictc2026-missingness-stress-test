"""Frozen contracts for personalized semantic indicator validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


CALIBRATION_DAYS = (0, 3, 7, 14, 21)
MAX_CALIBRATION_DAY = 21
LABEL_KEY_COLUMNS = ("subject_id", "lifelog_date", "sleep_date")


@dataclass(frozen=True)
class PrimitiveDefinition:
    name: str
    sensor_file: str
    value_column: str
    window: Literal["day", "evening"]
    cadence_minutes: int
    expected_epochs: int
    coverage_mode: Literal["availability", "event_support"]
    timestamp_semantics: Literal["instant", "interval_end"]
    support_minutes: int
    timing_anchor: Literal["source_timestamp", "support_midpoint"]
    unit: str
    unit_scale: float
    validity_column: str | None
    operation: str
    operation_params: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class PairDefinition:
    name: str
    left: str
    right: str


PRIMITIVE_DEFINITIONS = (
    PrimitiveDefinition(
        "screen_load_24h", "mScreenStatus_canonical.parquet", "screen_on", "day",
        1, 1440, "availability", "instant", 0, "source_timestamp", "fraction",
        1.0, None, "binary_load", (),
    ),
    PrimitiveDefinition(
        "screen_active_minutes_24h", "mScreenStatus_canonical.parquet", "screen_on", "day",
        1, 1440, "availability", "instant", 0, "source_timestamp", "minutes",
        1.0, None, "binary_active_minutes", (("epoch_minutes", 1.0),),
    ),
    PrimitiveDefinition(
        # ETRI Table 4 reports app durations in milliseconds. The per-bin total
        # is a summed app-usage intensity, not exclusive screen-on minutes.
        "usage_load_24h", "mUsageStats_canonical.parquet", "usage_total_usage_time", "day",
        10, 144, "event_support", "interval_end", 10, "support_midpoint",
        "reported_app_usage_minutes", 1.0 / 60000.0, None, "additive_load",
        (("minimum", 0.0),),
    ),
    PrimitiveDefinition(
        "screen_disengagement_p90", "mScreenStatus_canonical.parquet", "screen_on", "evening",
        1, 360, "availability", "instant", 0, "source_timestamp", "minute_of_day",
        1.0, None, "evening_p90", (),
    ),
    PrimitiveDefinition(
        "usage_disengagement_p90", "mUsageStats_canonical.parquet", "usage_total_usage_time",
        "evening", 10, 36, "event_support", "interval_end", 10, "support_midpoint",
        "minute_of_day", 1.0 / 60000.0, None, "evening_p90", (("minimum", 0.0),),
    ),
    PrimitiveDefinition(
        "phone_activity_load_24h", "mActivity_canonical.parquet", "activity_active_mobility",
        "day", 1, 1440, "availability", "instant", 0, "source_timestamp", "fraction",
        1.0, "activity_unknown", "activity_load", (("validity_equals", 0),),
    ),
    PrimitiveDefinition(
        "step_load_24h", "wPedo_canonical.parquet", "step", "day", 1, 1440,
        "availability", "instant", 0, "source_timestamp", "steps", 1.0, None,
        "additive_load", (("minimum", 0.0),),
    ),
    PrimitiveDefinition(
        "phone_activity_disengagement_p90", "mActivity_canonical.parquet",
        "activity_active_mobility", "evening", 1, 360, "availability", "instant", 0,
        "source_timestamp", "minute_of_day", 1.0, "activity_unknown", "evening_p90",
        (("validity_equals", 0),),
    ),
    PrimitiveDefinition(
        "step_disengagement_p90", "wPedo_canonical.parquet", "step", "evening", 1, 360,
        "availability", "instant", 0, "source_timestamp", "minute_of_day", 1.0,
        None, "evening_p90",
        (("minimum", 0.0),),
    ),
    PrimitiveDefinition(
        "mobile_light_exposure_24h", "mLight_canonical.parquet", "m_light", "day", 10, 144,
        "availability", "instant", 0, "source_timestamp", "log1p_lux", 1.0,
        None, "exposure_mean",
        (("minimum", 0.0), ("transform", "log1p")),
    ),
    PrimitiveDefinition(
        "wearable_light_exposure_24h", "wLight_canonical.parquet", "w_light", "day", 1, 1440,
        "availability", "instant", 0, "source_timestamp", "log1p_lux", 1.0,
        None, "exposure_mean",
        (("minimum", 0.0), ("transform", "log1p")),
    ),
    PrimitiveDefinition(
        "mobile_low_light_onset", "mLight_canonical.parquet", "m_light", "evening", 10, 36,
        "availability", "instant", 0, "source_timestamp", "minute_of_day", 1.0,
        None, "low_light_onset",
        (("low_threshold", 10.0), ("bin_minutes", 30), ("min_observations", 2),
         ("low_fraction_threshold", 0.75)),
    ),
    PrimitiveDefinition(
        "wearable_low_light_onset", "wLight_canonical.parquet", "w_light", "evening", 1, 360,
        "availability", "instant", 0, "source_timestamp", "minute_of_day", 1.0,
        None, "low_light_onset",
        (("low_threshold", 10.0), ("bin_minutes", 30), ("min_observations", 15),
         ("low_fraction_threshold", 0.75)),
    ),
    PrimitiveDefinition(
        "heart_rate_settling_delta", "wHr_canonical.parquet", "hr_mean", "evening", 1, 360,
        "availability", "instant", 0, "source_timestamp", "bpm", 1.0, None,
        "heart_rate_settling",
        (("component", "delta"), ("early_start_minute", 1080),
         ("early_end_minute", 1200), ("late_start_minute", 1320),
         ("late_end_minute", 1440), ("median_bin_minutes", 30),
         ("min_segment_epochs", 30)),
    ),
    PrimitiveDefinition(
        "heart_rate_settling_slope", "wHr_canonical.parquet", "hr_mean", "evening", 1, 360,
        "availability", "instant", 0, "source_timestamp", "bpm_per_hour", 1.0,
        None, "heart_rate_settling",
        (("component", "slope"), ("early_start_minute", 1080),
         ("early_end_minute", 1200), ("late_start_minute", 1320),
         ("late_end_minute", 1440), ("median_bin_minutes", 30),
         ("min_segment_epochs", 30)),
    ),
    PrimitiveDefinition(
        "charging_onset", "mACStatus_canonical.parquet", "charging_flag", "evening", 1, 360,
        "availability", "instant", 0, "source_timestamp", "minute_of_day", 1.0,
        None, "charging_onset",
        (("min_duration_minutes", 10), ("require_off_to_on", True)),
    ),
)


PRIMARY_PAIRS = (
    PairDefinition("digital_load", "screen_load_24h", "usage_load_24h"),
    PairDefinition("digital_disengagement", "screen_disengagement_p90", "usage_disengagement_p90"),
    PairDefinition("physical_load", "phone_activity_load_24h", "step_load_24h"),
    PairDefinition("light_exposure", "mobile_light_exposure_24h", "wearable_light_exposure_24h"),
)


def validate_label_contract(labels: pd.DataFrame) -> pd.DataFrame:
    """Validate next-day label keys and add stable row and chronological day IDs."""
    missing = sorted(set(LABEL_KEY_COLUMNS).difference(labels.columns))
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    result = labels.copy()
    for column in ("lifelog_date", "sleep_date"):
        parsed = pd.to_datetime(result[column], errors="raise")
        if parsed.dt.tz is not None:
            raise ValueError(f"{column} must contain timezone-naive local midnight values")
        if parsed.isna().any() or not parsed.eq(parsed.dt.normalize()).all():
            raise ValueError(f"{column} must contain exact midnight values")
        result[column] = parsed

    if result.duplicated(list(LABEL_KEY_COLUMNS)).any():
        raise ValueError("Label keys must be unique")
    if not (result["sleep_date"] - result["lifelog_date"]).eq(pd.Timedelta(days=1)).all():
        raise ValueError("sleep_date must be exactly one day after lifelog_date")

    result.insert(0, "row_id", range(len(result)))
    first_lifelog_date = result.groupby("subject_id", sort=False)["lifelog_date"].transform("min")
    result["day_index"] = (result["lifelog_date"] - first_lifelog_date).dt.days.astype(int) + 1
    return result
