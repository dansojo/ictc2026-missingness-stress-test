"""Deterministic masking and source-day selection for primitive reliability.

This module provides replay mechanics only.  It does not calculate or claim
the full fifty-draw G2 reliability gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from numbers import Number
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .contracts import PRIMARY_PAIRS, PRIMITIVE_DEFINITIONS, PrimitiveDefinition
from .personalization import (
    MAD_NORMALIZATION,
    build_loso_representations,
    compose_semantic_columns,
    compute_primitive_confidence,
    robust_location_scale,
    shrink_location_scale,
)
from .primitives import (
    PreparedPrimitiveReplay,
    prepare_primitive_replay,
    recompute_prepared_primitive,
    recompute_prepared_primitive_batch,
    recompute_primitive_from_frame,
)

CORE_G2_PRIMITIVES = (
    "screen_load_24h",
    "usage_load_24h",
    "screen_disengagement_p90",
    "usage_disengagement_p90",
    "phone_activity_load_24h",
    "step_load_24h",
    "mobile_light_exposure_24h",
    "wearable_light_exposure_24h",
)
MAX_SELECTED_DAYS = 10
DEFAULT_MASK_DRAWS = 50
FROZEN_MASK_SCENARIOS = (
    "contiguous_10pct",
    "contiguous_20pct",
    "contiguous_40pct",
    "outage_1h",
    "outage_3h",
    "outage_6h",
    "evening_critical",
    "empirical_gap",
)
G2_COMMON_FINGERPRINT_KEYS = (
    "root_seed",
    "raw_inputs_sha256",
    "canonical_inputs_sha256",
    "primitives_sha256",
    "baselines_sha256",
    "representations_sha256",
    "sensors_sha256",
    "selected_topology_sha256",
    "g1_artifact_sha256",
    "g1_config_sha256",
    "code_tree_sha256",
    "settings_sha256",
    "schema_sha256",
)
_DEFINITIONS = {definition.name: definition for definition in PRIMITIVE_DEFINITIONS}
_FRACTION_SCENARIOS = {
    "contiguous_10pct": 0.10,
    "contiguous_20pct": 0.20,
    "contiguous_40pct": 0.40,
}
_OUTAGE_SCENARIOS = {
    "outage_1h": 60,
    "outage_3h": 180,
    "outage_6h": 360,
}
MASK_TABLE_COLUMNS = (
    "subject_id",
    "sensor_day_id",
    "primitive",
    "scenario",
    "draw",
    "root_seed",
    "seed",
    "window_start",
    "window_end",
    "mask_interval_id",
    "mask_start",
    "mask_end",
    "mask_duration_minutes",
    "mask_semantics",
    "empirical_policy",
    "status",
)
DELETION_AUDIT_COLUMNS = (
    "audit_level",
    "subject_id",
    "sensor_day_id",
    "primitive",
    "scenario",
    "draw",
    "root_seed",
    "seed",
    "window_start",
    "window_end",
    "mask_interval_id",
    "mask_start",
    "mask_end",
    "mask_duration_minutes",
    "mask_semantics",
    "empirical_policy",
    "mask_status",
    "record_id",
    "source_timestamp",
    "record_interval_start",
    "record_interval_end",
    "timestamp_semantics",
    "support_minutes",
    "deleted",
    "deleted_by_any_mask",
    "usage_record_deletion_stability",
    "original_count",
    "deleted_count",
    "retained_count",
    "status",
)
_USAGE_SELECTION_PROXIES = {
    "usage_load_24h": "screen_load_24h",
    "usage_disengagement_p90": "screen_disengagement_p90",
}
SELECTION_AUDIT_COLUMNS = (
    "subject_id",
    "sensor_day_id",
    "primitive",
    "proxy_primitive",
    "availability_proxy",
    "gap_component",
    "quality_confidence",
    "source_observed",
    "finite_value",
    "primitive_status",
    "censoring",
    "eligible",
    "rank",
    "selected",
    "reason",
    "candidate_count",
    "eligible_count",
    "selected_count",
    "max_days",
    "selection_status",
    "participant_denominator",
)
SELECTED_DAY_COLUMNS = (
    "subject_id",
    "sensor_day_id",
    "primitive",
    "proxy_primitive",
    "availability_proxy",
    "gap_component",
    "quality_confidence",
    "rank",
    "selected_count",
    "selection_status",
    "participant_denominator",
)


def derive_mask_seed(
    root_seed: int | str,
    scenario: str,
    draw: int,
    subject_id: object,
    sensor_day_id: int,
    primitive: str,
) -> str:
    """Return the full SHA-256 digest of the frozen mask identity tuple."""
    identity = [
        str(root_seed),
        str(scenario),
        str(draw),
        str(subject_id),
        str(sensor_day_id),
        str(primitive),
    ]
    payload = json.dumps(
        identity, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonnegative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _local_midnight(value: object, name: str) -> pd.Timestamp:
    if isinstance(value, Number):
        raise ValueError(f"{name} must be an exact timezone-naive midnight")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must be an exact timezone-naive midnight"
        ) from error
    if (
        pd.isna(timestamp)
        or timestamp.tz is not None
        or timestamp != timestamp.normalize()
    ):
        raise ValueError(f"{name} must be an exact timezone-naive midnight")
    return timestamp


def _primitive_window(
    definition: PrimitiveDefinition, lifelog_date: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if definition.window == "day":
        return lifelog_date, lifelog_date + pd.Timedelta(days=1)
    if definition.window == "evening":
        return (
            lifelog_date + pd.Timedelta(hours=18),
            lifelog_date + pd.Timedelta(days=1),
        )
    raise ValueError(f"Unsupported frozen primitive window: {definition.window}")


def _parse_empirical_intervals(
    empirical_intervals: object,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Validate empirical gaps under the fixed reject-without-clipping policy."""
    if isinstance(empirical_intervals, pd.DataFrame):
        if empirical_intervals.columns.duplicated().any():
            raise ValueError("empirical gap columns must be unique")
        missing = {
            "gap_start",
            "gap_end",
        }.difference(empirical_intervals.columns)
        if missing:
            raise ValueError(
                "empirical gap table requires gap_start and gap_end columns"
            )
        raw_intervals = list(
            empirical_intervals.loc[:, ["gap_start", "gap_end"]].itertuples(
                index=False, name=None
            )
        )
    else:
        try:
            raw_intervals = list(empirical_intervals)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError("empirical_intervals must contain empirical gap pairs") from error

    if not raw_intervals:
        raise ValueError("empirical_intervals must contain at least one empirical gap")

    parsed: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for raw in raw_intervals:
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError("each empirical gap must be a start/end pair")
        try:
            start, end = pd.Timestamp(raw[0]), pd.Timestamp(raw[1])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("empirical gap endpoints must be valid timestamps") from error
        if start.tz is not None or end.tz is not None:
            raise ValueError("empirical gap endpoints must be timezone-naive")
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError("empirical gap must have a positive duration")
        if start < window_start or end > window_end:
            raise ValueError(
                "empirical gap is outside the primitive window; fixed policy is reject"
            )
        parsed.append((start, end))

    parsed.sort(key=lambda pair: (pair[0], pair[1]))
    for previous, current in zip(parsed, parsed[1:]):
        if current[0] < previous[1]:
            raise ValueError("empirical gap intervals must not overlap")
    return parsed


def generate_mask_intervals(
    primitive: str,
    *,
    subject_id: object,
    sensor_day_id: int,
    lifelog_date: pd.Timestamp | str,
    scenario: str,
    draw: int = 0,
    root_seed: int | str = 0,
    empirical_intervals: object | None = None,
) -> pd.DataFrame:
    """Generate deterministic half-open masks inside a frozen primitive window.

    Empirical intervals use a fixed ``reject`` policy: an invalid or
    out-of-window interval raises and is never clipped or duration-adjusted.
    """
    if primitive not in CORE_G2_PRIMITIVES:
        raise ValueError(f"primitive must be one of the eight core G2 primitives: {primitive}")
    if scenario not in FROZEN_MASK_SCENARIOS:
        raise ValueError(f"scenario must be one of {FROZEN_MASK_SCENARIOS}")
    sensor_day = _nonnegative_integer(sensor_day_id, "sensor_day_id")
    draw_id = _nonnegative_integer(draw, "draw")
    day = _local_midnight(lifelog_date, "lifelog_date")
    definition = _DEFINITIONS[primitive]
    window_start, window_end = _primitive_window(definition, day)
    seed = derive_mask_seed(
        root_seed, scenario, draw_id, subject_id, sensor_day, primitive
    )

    if scenario == "empirical_gap":
        if empirical_intervals is None:
            raise ValueError(
                "empirical_intervals is required for the empirical_gap scenario"
            )
        intervals = _parse_empirical_intervals(
            empirical_intervals, window_start, window_end
        )
        status = "empirical_replay"
        empirical_policy = "reject"
    else:
        if empirical_intervals is not None:
            raise ValueError(
                "empirical_intervals is only valid for the empirical_gap scenario"
            )
        window_minutes = int(
            (window_end - window_start).total_seconds() // 60
        )
        if scenario in _FRACTION_SCENARIOS:
            duration_minutes = int(
                window_minutes * _FRACTION_SCENARIOS[scenario]
            )
        elif scenario in _OUTAGE_SCENARIOS:
            duration_minutes = _OUTAGE_SCENARIOS[scenario]
        elif scenario == "evening_critical":
            duration_minutes = 180
        else:  # pragma: no cover - guarded by the frozen scenario tuple
            raise ValueError(f"Unsupported mask scenario: {scenario}")

        if scenario == "evening_critical":
            mask_start = day + pd.Timedelta(hours=21)
        else:
            maximum_start_minutes = window_minutes - duration_minutes
            if maximum_start_minutes < 0:
                raise ValueError("mask duration exceeds the primitive window")
            rng = np.random.default_rng(int(seed[:16], 16))
            start_offset = int(rng.integers(0, maximum_start_minutes + 1))
            mask_start = window_start + pd.Timedelta(minutes=start_offset)
        intervals = [
            (mask_start, mask_start + pd.Timedelta(minutes=duration_minutes))
        ]
        status = "generated"
        empirical_policy = "not_applicable"

    records: list[dict[str, Any]] = []
    for interval_id, (mask_start, mask_end) in enumerate(intervals):
        records.append(
            {
                "subject_id": subject_id,
                "sensor_day_id": sensor_day,
                "primitive": primitive,
                "scenario": scenario,
                "draw": draw_id,
                "root_seed": str(root_seed),
                "seed": seed,
                "window_start": window_start,
                "window_end": window_end,
                "mask_interval_id": interval_id,
                "mask_start": mask_start,
                "mask_end": mask_end,
                "mask_duration_minutes": (
                    mask_end - mask_start
                ).total_seconds()
                / 60.0,
                "mask_semantics": "half_open",
                "empirical_policy": empirical_policy,
                "status": status,
            }
        )
    return pd.DataFrame.from_records(records, columns=MASK_TABLE_COLUMNS)


def _naive_timestamp(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain valid mask timestamps") from error
    if pd.isna(timestamp) or timestamp.tz is not None:
        raise ValueError(f"{name} must contain timezone-naive mask timestamps")
    return timestamp


def _validate_mask_table(
    mask_intervals: pd.DataFrame,
    *,
    primitive: str,
    subject_id: object,
    sensor_day_id: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(mask_intervals, pd.DataFrame) or mask_intervals.empty:
        raise ValueError("mask_intervals must be a nonempty mask table")
    if mask_intervals.columns.duplicated().any():
        raise ValueError("mask table columns must be unique")
    missing = set(MASK_TABLE_COLUMNS).difference(mask_intervals.columns)
    if missing:
        raise ValueError(f"mask table is missing columns: {sorted(missing)}")

    masks = mask_intervals.loc[:, MASK_TABLE_COLUMNS].copy()
    if not masks["primitive"].eq(primitive).all():
        raise ValueError("mask primitive identity does not match replay request")
    if not masks["subject_id"].eq(subject_id).all():
        raise ValueError("mask subject identity does not match calendar row")
    numeric_day = pd.to_numeric(masks["sensor_day_id"], errors="coerce")
    if not numeric_day.eq(sensor_day_id).all():
        raise ValueError("mask sensor_day_id does not match calendar row")
    if not masks["scenario"].isin(FROZEN_MASK_SCENARIOS).all():
        raise ValueError("mask scenario is not frozen")
    if not masks["mask_semantics"].eq("half_open").all():
        raise ValueError("mask semantics must be half_open")
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
    )
    if len(masks.loc[:, identity_columns].drop_duplicates()) != 1:
        raise ValueError("mask table must contain a single replay identity")
    empirical = masks["scenario"].eq("empirical_gap")
    expected_status = np.where(empirical, "empirical_replay", "generated")
    expected_policy = np.where(empirical, "reject", "not_applicable")
    if not masks["status"].eq(expected_status).all():
        raise ValueError("mask status does not match its scenario")
    if not masks["empirical_policy"].eq(expected_policy).all():
        raise ValueError("mask empirical policy does not match its scenario")

    parsed_columns: dict[str, list[pd.Timestamp]] = {
        name: [] for name in ("window_start", "window_end", "mask_start", "mask_end")
    }
    for name in parsed_columns:
        parsed_columns[name] = [
            _naive_timestamp(value, name) for value in masks[name]
        ]
        masks[name] = pd.Series(parsed_columns[name], index=masks.index)
    if not masks["window_start"].eq(window_start).all() or not masks[
        "window_end"
    ].eq(window_end).all():
        raise ValueError("mask window does not match the frozen primitive window")
    if not (
        masks["mask_start"].ge(window_start)
        & masks["mask_end"].le(window_end)
        & masks["mask_start"].lt(masks["mask_end"])
    ).all():
        raise ValueError("mask interval must be positive and inside its window")

    actual_duration = (
        masks["mask_end"] - masks["mask_start"]
    ).dt.total_seconds() / 60.0
    claimed_duration = pd.to_numeric(
        masks["mask_duration_minutes"], errors="coerce"
    )
    if not np.isfinite(claimed_duration.to_numpy(dtype=float)).all() or not np.allclose(
        claimed_duration.to_numpy(dtype=float),
        actual_duration.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("mask duration audit does not match its endpoints")

    draws = pd.to_numeric(masks["draw"], errors="coerce")
    if not draws.map(lambda value: np.isfinite(value) and value >= 0 and value == int(value)).all():
        raise ValueError("mask draw identity must be a nonnegative integer")
    for row in masks.itertuples(index=False):
        expected_seed = derive_mask_seed(
            row.root_seed,
            row.scenario,
            int(row.draw),
            row.subject_id,
            int(row.sensor_day_id),
            row.primitive,
        )
        if row.seed != expected_seed:
            raise ValueError("mask seed does not match its frozen identity tuple")

    interval_ids = pd.to_numeric(masks["mask_interval_id"], errors="coerce")
    if (
        not interval_ids.map(
            lambda value: np.isfinite(value) and value >= 0 and value == int(value)
        ).all()
        or interval_ids.duplicated().any()
    ):
        raise ValueError("mask interval identifiers must be unique nonnegative integers")
    masks = masks.sort_values(
        ["mask_start", "mask_end", "mask_interval_id"], kind="stable"
    ).reset_index(drop=True)
    if not masks["mask_interval_id"].astype(int).eq(
        pd.Series(np.arange(len(masks), dtype=int))
    ).all():
        raise ValueError("mask interval identifiers must match canonical order")
    for previous_end, current_start in zip(
        masks["mask_end"].iloc[:-1], masks["mask_start"].iloc[1:]
    ):
        if current_start < previous_end:
            raise ValueError("mask intervals must not overlap")
    if not bool(empirical.iloc[0]):
        if len(masks) != 1:
            raise ValueError("generated mask scenarios require exactly one interval")
        row = masks.iloc[0]
        regenerated = generate_mask_intervals(
            primitive,
            subject_id=subject_id,
            sensor_day_id=sensor_day_id,
            lifelog_date=window_start.normalize(),
            scenario=row["scenario"],
            draw=int(row["draw"]),
            root_seed=row["root_seed"],
        ).iloc[0]
        for column in ("mask_start", "mask_end", "mask_duration_minutes"):
            if row[column] != regenerated[column]:
                raise ValueError(
                    "mask interval does not match deterministic regeneration"
                )
    return masks


def apply_mask_intervals(
    primitive: str,
    sensor_frame: pd.DataFrame,
    sensor_file: str,
    calendar_row: pd.DataFrame,
    mask_intervals: pd.DataFrame,
    *,
    interval_boundary_policy: str = "measurement_support",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delete canonical records under exact instant or interval-overlap rules.

    Returned values are a canonical-order copy of the retained source frame and
    a record-level deletion audit.  UsageStats results describe record-deletion
    stability only; no availability claim is made.
    """
    if primitive not in CORE_G2_PRIMITIVES:
        raise ValueError("primitive must be one of the eight core G2 primitives")
    # The public replay boundary is the single validation authority for sensor
    # file/channel, timestamps, duplicate keys, bait, and calendar provenance.
    recompute_primitive_from_frame(
        primitive,
        sensor_frame,
        sensor_file,
        calendar_row,
        interval_boundary_policy=interval_boundary_policy,
    )
    if "sensor_day_id" not in calendar_row.columns:
        raise ValueError("calendar_row must contain sensor_day_id for mask replay")

    definition = _DEFINITIONS[primitive]
    subject_id = calendar_row.iloc[0]["subject_id"]
    sensor_day_id = _nonnegative_integer(
        calendar_row.iloc[0]["sensor_day_id"], "sensor_day_id"
    )
    lifelog_date = _local_midnight(
        calendar_row.iloc[0]["lifelog_date"], "lifelog_date"
    )
    window_start, window_end = _primitive_window(definition, lifelog_date)
    masks = _validate_mask_table(
        mask_intervals,
        primitive=primitive,
        subject_id=subject_id,
        sensor_day_id=sensor_day_id,
        window_start=window_start,
        window_end=window_end,
    )

    working = sensor_frame.copy(deep=True)
    working["__source_timestamp"] = pd.to_datetime(
        working["timestamp"], errors="coerce"
    )
    working["__subject_sort"] = working["subject_id"].map(str)
    working = working.sort_values(
        ["__subject_sort", "__source_timestamp"], kind="stable", na_position="last"
    ).reset_index(drop=True)

    source_timestamp = working["__source_timestamp"]
    same_subject = working["subject_id"].eq(subject_id)
    if definition.timestamp_semantics == "instant":
        record_start = source_timestamp
        record_end = source_timestamp
        in_target = (
            same_subject
            & source_timestamp.ge(window_start)
            & source_timestamp.lt(window_end)
        )
    else:
        support = pd.Timedelta(minutes=int(definition.support_minutes))
        record_start = source_timestamp - support
        record_end = source_timestamp
        in_target = (
            same_subject
            & record_start.ge(window_start)
            & record_end.le(window_end)
        )
        if interval_boundary_policy == "strict_source_availability":
            in_target &= source_timestamp.lt(window_end)

    target_positions = np.flatnonzero(in_target.to_numpy(dtype=bool))
    delete_any = np.zeros(len(working), dtype=bool)
    per_mask_deletions: dict[tuple[int, int], bool] = {}
    for position in target_positions:
        for mask_position, mask in masks.iterrows():
            if definition.timestamp_semantics == "instant":
                deleted = bool(
                    record_start.iloc[position] >= mask["mask_start"]
                    and record_start.iloc[position] < mask["mask_end"]
                )
            else:
                deleted = bool(
                    record_start.iloc[position] < mask["mask_end"]
                    and record_end.iloc[position] > mask["mask_start"]
                )
            per_mask_deletions[(position, int(mask_position))] = deleted
            delete_any[position] |= deleted

    original_count = int(len(target_positions))
    deleted_count = int(delete_any[target_positions].sum())
    retained_count = original_count - deleted_count
    audit_records: list[dict[str, Any]] = []
    if original_count == 0:
        for mask in masks.itertuples(index=False):
            audit_records.append(
                {
                    "audit_level": "summary",
                    **{
                        name: getattr(mask, name)
                        for name in MASK_TABLE_COLUMNS
                        if name != "status"
                    },
                    "mask_status": mask.status,
                    "record_id": pd.NA,
                    "source_timestamp": pd.NaT,
                    "record_interval_start": pd.NaT,
                    "record_interval_end": pd.NaT,
                    "timestamp_semantics": definition.timestamp_semantics,
                    "support_minutes": definition.support_minutes,
                    "deleted": False,
                    "deleted_by_any_mask": False,
                    "usage_record_deletion_stability": (
                        definition.timestamp_semantics == "interval_end"
                    ),
                    "original_count": 0,
                    "deleted_count": 0,
                    "retained_count": 0,
                    "status": "no_target_records",
                }
            )
    else:
        for record_id, position in enumerate(target_positions):
            for mask_position, mask in masks.iterrows():
                deleted = per_mask_deletions[(position, int(mask_position))]
                audit_records.append(
                    {
                        "audit_level": "record",
                        **{
                            name: mask[name]
                            for name in MASK_TABLE_COLUMNS
                            if name != "status"
                        },
                        "mask_status": mask["status"],
                        "record_id": record_id,
                        "source_timestamp": source_timestamp.iloc[position],
                        "record_interval_start": record_start.iloc[position],
                        "record_interval_end": record_end.iloc[position],
                        "timestamp_semantics": definition.timestamp_semantics,
                        "support_minutes": definition.support_minutes,
                        "deleted": deleted,
                        "deleted_by_any_mask": bool(delete_any[position]),
                        "usage_record_deletion_stability": (
                            definition.timestamp_semantics == "interval_end"
                        ),
                        "original_count": original_count,
                        "deleted_count": deleted_count,
                        "retained_count": retained_count,
                        "status": "deleted" if deleted else "retained",
                    }
                )

    masked = working.loc[~delete_any].drop(
        columns=["__source_timestamp", "__subject_sort"]
    ).reset_index(drop=True)
    audit = pd.DataFrame.from_records(
        audit_records, columns=DELETION_AUDIT_COLUMNS
    )
    return masked, audit


def select_reliability_days(
    primitive_table: pd.DataFrame,
    *,
    max_days: int = MAX_SELECTED_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select at most ten source-observed high-quality days per G2 cell.

    Ranking is frozen to availability proxy descending, gap component
    descending, then ``sensor_day_id``.  UsageStats uses the matching screen
    primitive's availability and gap metadata; its magnitude, support density,
    and support count are never ranking inputs.  Outcome and oracle columns are
    ignored by construction while every participant-day candidate remains in
    the returned audit.
    """
    if (
        isinstance(max_days, (bool, np.bool_))
        or not isinstance(max_days, (int, np.integer))
        or int(max_days) <= 0
        or int(max_days) > MAX_SELECTED_DAYS
    ):
        raise ValueError("max_days must be a positive integer at most 10")
    cap = int(max_days)
    if not isinstance(primitive_table, pd.DataFrame) or primitive_table.empty:
        raise ValueError("primitive_table must be a nonempty pandas DataFrame")
    if primitive_table.columns.duplicated().any():
        raise ValueError("primitive_table columns must be unique")

    required = {
        "subject_id",
        "sensor_day_id",
        "any_source_record_observed",
    }
    for primitive in CORE_G2_PRIMITIVES:
        required.update((primitive, f"{primitive}__status"))
        proxy = _USAGE_SELECTION_PROXIES.get(primitive, primitive)
        required.update(
            (
                f"{proxy}__availability_coverage",
                f"{proxy}__longest_gap_minutes",
            )
        )
    missing = sorted(required.difference(primitive_table.columns))
    if missing:
        raise ValueError(f"primitive_table is missing required columns: {missing}")
    if primitive_table["subject_id"].isna().any():
        raise ValueError("subject_id must be present for every candidate")
    if primitive_table.duplicated(["subject_id", "sensor_day_id"]).any():
        raise ValueError("participant and sensor_day_id candidate keys must be unique")

    sensor_day_values: list[int] = []
    for value in primitive_table["sensor_day_id"]:
        sensor_day_values.append(_nonnegative_integer(value, "sensor_day_id"))
    if len(set(sensor_day_values)) != len(sensor_day_values):
        raise ValueError("sensor_day_id values must be globally unique")
    source_flag = primitive_table["any_source_record_observed"]
    if not source_flag.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("any_source_record_observed must contain true boolean values")

    working = primitive_table.copy(deep=True)
    working["sensor_day_id"] = sensor_day_values
    working["__subject_sort"] = working["subject_id"].map(str)
    working = working.sort_values(
        ["__subject_sort", "sensor_day_id"], kind="stable"
    ).reset_index(drop=True)
    participants = working[
        ["subject_id", "__subject_sort"]
    ].drop_duplicates().sort_values("__subject_sort", kind="stable")
    participant_ids = participants["subject_id"].tolist()
    participant_denominator = len(participant_ids)
    primitive_order = {
        primitive: order for order, primitive in enumerate(CORE_G2_PRIMITIVES)
    }
    audit_records: list[dict[str, Any]] = []

    for subject_id in participant_ids:
        participant_rows = working.loc[working["subject_id"].eq(subject_id)].copy()
        candidate_count = len(participant_rows)
        for primitive in CORE_G2_PRIMITIVES:
            definition = _DEFINITIONS[primitive]
            proxy_primitive = _USAGE_SELECTION_PROXIES.get(primitive, primitive)
            proxy_definition = _DEFINITIONS[proxy_primitive]
            value = pd.to_numeric(participant_rows[primitive], errors="coerce")
            value_array = value.to_numpy(dtype=float, na_value=np.nan)
            finite_value = np.isfinite(value_array)
            status = participant_rows[f"{primitive}__status"]
            valid_status = status.eq("observed").to_numpy(dtype=bool)
            censoring_column = f"{primitive}__censoring"
            if censoring_column in participant_rows:
                censoring = participant_rows[censoring_column]
                uncensored = (
                    censoring.isna() | censoring.eq("none")
                ).to_numpy(dtype=bool)
            else:
                censoring = pd.Series(
                    "none", index=participant_rows.index, dtype=object
                )
                uncensored = np.ones(candidate_count, dtype=bool)

            coverage = pd.to_numeric(
                participant_rows[
                    f"{proxy_primitive}__availability_coverage"
                ],
                errors="coerce",
            ).to_numpy(dtype=float, na_value=np.nan)
            gap_minutes = pd.to_numeric(
                participant_rows[f"{proxy_primitive}__longest_gap_minutes"],
                errors="coerce",
            ).to_numpy(dtype=float, na_value=np.nan)
            proxy_window_minutes = (
                proxy_definition.expected_epochs * proxy_definition.cadence_minutes
            )
            availability_proxy = np.where(
                np.isfinite(coverage), np.clip(coverage, 0.0, 1.0), np.nan
            )
            gap_component = np.where(
                np.isfinite(gap_minutes),
                np.clip(1.0 - gap_minutes / proxy_window_minutes, 0.0, 1.0),
                np.nan,
            )
            source_observed = participant_rows[
                "any_source_record_observed"
            ].to_numpy(dtype=bool)
            high_quality = (
                np.isfinite(availability_proxy)
                & (availability_proxy > 0.0)
                & np.isfinite(gap_component)
                & (gap_component > 0.0)
            )
            eligible = (
                source_observed
                & finite_value
                & valid_status
                & uncensored
                & high_quality
            )
            quality_confidence = np.where(
                eligible, availability_proxy * gap_component, 0.0
            )

            ranking = pd.DataFrame(
                {
                    "position": np.arange(candidate_count, dtype=int),
                    "sensor_day_id": participant_rows[
                        "sensor_day_id"
                    ].to_numpy(dtype=int),
                    "availability_proxy": availability_proxy,
                    "gap_component": gap_component,
                    "eligible": eligible,
                }
            )
            eligible_ranking = ranking.loc[ranking["eligible"]].sort_values(
                ["availability_proxy", "gap_component", "sensor_day_id"],
                ascending=[False, False, True],
                kind="stable",
            )
            rank_by_position = {
                int(position): rank
                for rank, position in enumerate(
                    eligible_ranking["position"].tolist(), start=1
                )
            }
            selected_positions = set(
                eligible_ranking.head(cap)["position"].astype(int).tolist()
            )
            eligible_count = int(eligible.sum())
            selected_count = len(selected_positions)
            if selected_count == 0:
                selection_status = "no_eligible"
            elif selected_count < cap:
                selection_status = "fewer_than_maximum"
            else:
                selection_status = "selected_maximum"

            for position, (_, candidate) in enumerate(
                participant_rows.iterrows()
            ):
                selected = position in selected_positions
                reasons: list[str] = []
                if selected:
                    reasons.append("selected")
                elif eligible[position]:
                    reasons.append(f"not_top_{cap}")
                else:
                    if not source_observed[position]:
                        reasons.append("source_not_observed")
                    if not finite_value[position]:
                        reasons.append("primitive_nonfinite")
                    if not valid_status[position]:
                        reasons.append("primitive_status_not_observed")
                    if not uncensored[position]:
                        reasons.append("censored")
                    if not np.isfinite(availability_proxy[position]):
                        reasons.append("availability_proxy_missing")
                    elif availability_proxy[position] <= 0.0:
                        reasons.append("availability_proxy_zero")
                    if not np.isfinite(gap_component[position]):
                        reasons.append("gap_component_missing")
                    elif gap_component[position] <= 0.0:
                        reasons.append("gap_component_zero")
                audit_records.append(
                    {
                        "subject_id": subject_id,
                        "sensor_day_id": int(candidate["sensor_day_id"]),
                        "primitive": primitive,
                        "proxy_primitive": proxy_primitive,
                        "availability_proxy": float(
                            availability_proxy[position]
                        ),
                        "gap_component": float(gap_component[position]),
                        "quality_confidence": float(
                            quality_confidence[position]
                        ),
                        "source_observed": bool(source_observed[position]),
                        "finite_value": bool(finite_value[position]),
                        "primitive_status": status.iloc[position],
                        "censoring": censoring.iloc[position],
                        "eligible": bool(eligible[position]),
                        "rank": rank_by_position.get(position, pd.NA),
                        "selected": selected,
                        "reason": "|".join(reasons),
                        "candidate_count": candidate_count,
                        "eligible_count": eligible_count,
                        "selected_count": selected_count,
                        "max_days": cap,
                        "selection_status": selection_status,
                        "participant_denominator": participant_denominator,
                    }
                )

    audit = pd.DataFrame.from_records(
        audit_records, columns=SELECTION_AUDIT_COLUMNS
    )
    audit["__subject_sort"] = audit["subject_id"].map(str)
    audit["__primitive_order"] = audit["primitive"].map(primitive_order)
    audit = audit.sort_values(
        ["__subject_sort", "__primitive_order", "sensor_day_id"], kind="stable"
    ).drop(columns=["__subject_sort", "__primitive_order"]).reset_index(drop=True)
    audit["rank"] = audit["rank"].astype("Int64")
    selected = audit.loc[audit["selected"], SELECTED_DAY_COLUMNS].copy()
    selected["__subject_sort"] = selected["subject_id"].map(str)
    selected["__primitive_order"] = selected["primitive"].map(primitive_order)
    selected = selected.sort_values(
        ["__subject_sort", "__primitive_order", "rank"], kind="stable"
    ).drop(columns=["__subject_sort", "__primitive_order"]).reset_index(drop=True)
    return selected, audit


@dataclass(frozen=True)
class G2ReliabilityResult:
    """Auditable tables for the frozen hierarchical G2 study."""

    selected_days: pd.DataFrame
    selected_day_audit: pd.DataFrame
    mask_intervals: pd.DataFrame
    deletion_audit: pd.DataFrame
    cell_replays: pd.DataFrame
    draw_day_metrics: pd.DataFrame
    participant_draw_metrics: pd.DataFrame
    participant_metrics: pd.DataFrame
    participant_macro_metrics: pd.DataFrame
    primitive_gates: pd.DataFrame
    pair_gates: pd.DataFrame
    domain_gates: pd.DataFrame
    decision_table: pd.DataFrame
    confidence_ablation: pd.DataFrame
    confidence_summary: pd.DataFrame
    calibration_sensitivity: pd.DataFrame
    calibration_summary: pd.DataFrame
    modality_sensitivity: pd.DataFrame
    modality_summary: pd.DataFrame
    run_manifest: pd.DataFrame


@dataclass(frozen=True)
class G2FinalizedResult:
    """Official topology-validated reconstruction of streamed G2 chunks."""

    draw_day_metrics: pd.DataFrame
    participant_draw_metrics: pd.DataFrame
    participant_metrics: pd.DataFrame
    participant_macro_metrics: pd.DataFrame
    primitive_gates: pd.DataFrame
    pair_gates: pd.DataFrame
    domain_gates: pd.DataFrame
    decision_table: pd.DataFrame
    confidence_ablation: pd.DataFrame
    confidence_summary: pd.DataFrame
    run_manifest: pd.DataFrame


def _required_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError(f"{name} columns must be unique")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalized_mad(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(finite))
    scale = float(MAD_NORMALIZATION * np.median(np.abs(finite - center)))
    return center, scale


def _population_rows(
    frozen_population_baselines: pd.DataFrame,
) -> pd.DataFrame:
    required = (
        "held_out_subject",
        "primitive",
        "method",
        "k",
        "donor_subject",
        "scale",
        "population_scale",
        "population_contributing_rows",
        "baseline_status",
    )
    _required_columns(
        frozen_population_baselines, required, "frozen_population_baselines"
    )
    methods = frozen_population_baselines["method"].astype("string")
    k_values = pd.to_numeric(frozen_population_baselines["k"], errors="coerce")
    authoritative = frozen_population_baselines.loc[
        methods.eq("population")
        & k_values.eq(0)
        & frozen_population_baselines["donor_subject"].isna()
    ].copy()
    if authoritative.duplicated(["held_out_subject", "primitive"]).any():
        raise ValueError("frozen population baseline keys must be unique")
    return authoritative


def _validate_frozen_population_anchor(
    *,
    primitive_table: pd.DataFrame,
    primitive: str,
    held_out_subject: object,
    baseline: pd.Series,
) -> float:
    if primitive not in CORE_G2_PRIMITIVES:
        raise ValueError(f"standardizer primitive is outside frozen G2: {primitive}")
    scale = float(pd.to_numeric(pd.Series([baseline["scale"]]), errors="coerce").iloc[0])
    reported_population_scale = float(
        pd.to_numeric(
            pd.Series([baseline["population_scale"]]), errors="coerce"
        ).iloc[0]
    )
    if (
        not np.isfinite(scale)
        or scale <= 0
        or not np.isfinite(reported_population_scale)
        or not np.isclose(
            scale, reported_population_scale, rtol=0.0, atol=1e-12
        )
        or str(baseline["baseline_status"]) != "observed"
    ):
        raise ValueError("frozen population anchor is nonfinite, inconsistent, or degenerate")

    confidence = compute_primitive_confidence(
        primitive_table,
        primitive_names=(primitive,),
        usage_policy="screen_proxy",
    )
    outer = confidence.loc[confidence["subject_id"].ne(held_out_subject)]
    estimate = robust_location_scale(
        pd.to_numeric(outer["value"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(
            outer["quality_confidence"], errors="coerce"
        ).to_numpy(dtype=float),
    )
    recomputed_scale = float(estimate["scale"])
    reported_rows = pd.to_numeric(
        pd.Series([baseline["population_contributing_rows"]]), errors="coerce"
    ).iloc[0]
    if (
        not np.isfinite(recomputed_scale)
        or recomputed_scale <= 0
        or not np.isclose(scale, recomputed_scale, rtol=1e-10, atol=1e-12)
        or not np.isfinite(reported_rows)
        or int(reported_rows) != int(estimate["contributing_rows"])
    ):
        raise ValueError(
            "frozen population anchor does not match the recipient-excluding Task3 k0 estimate"
        )
    return scale


def compute_reliability_standardizers(
    selected_cells: pd.DataFrame,
    *,
    frozen_population_baselines: pd.DataFrame,
    primitive_table: pd.DataFrame,
) -> pd.DataFrame:
    """Freeze within-person error scales before any masked value is inspected.

    The personal component is the normalized, unweighted MAD of the selected
    original values.  Its sole outer anchor is the audited Task3 population
    k=0 scale for the same held-out recipient and primitive.  The anchor is
    independently regenerated from all permitted other-participant rows.
    """
    _required_columns(
        selected_cells,
        ("subject_id", "sensor_day_id", "primitive", "original_value"),
        "selected_cells",
    )
    _required_columns(
        primitive_table, ("subject_id", "sensor_day_id"), "primitive_table"
    )
    if selected_cells.empty:
        return pd.DataFrame(
            columns=(
                "subject_id",
                "primitive",
                "selected_count",
                "selected_center",
                "selected_day_mad",
                "frozen_population_scale",
                "scale_floor",
                "standardizer",
                "scale_source",
                "floor_applied",
                "status",
            )
        )
    original = pd.to_numeric(selected_cells["original_value"], errors="coerce")
    if not np.isfinite(original.to_numpy(dtype=float)).all():
        raise ValueError("selected_cells original_value must be finite")
    if selected_cells.duplicated(
        ["subject_id", "sensor_day_id", "primitive"]
    ).any():
        raise ValueError("selected cell keys must be unique")

    authoritative = _population_rows(frozen_population_baselines)
    records: list[dict[str, Any]] = []
    ordered = selected_cells.assign(
        __subject_sort=selected_cells["subject_id"].map(str),
        __primitive_order=selected_cells["primitive"].map(
            {name: index for index, name in enumerate(CORE_G2_PRIMITIVES)}
        ),
    ).sort_values(
        ["__subject_sort", "__primitive_order", "sensor_day_id"], kind="stable"
    )
    for (subject_id, primitive), group in ordered.groupby(
        ["subject_id", "primitive"], sort=False, dropna=False
    ):
        matching = authoritative.loc[
            authoritative["held_out_subject"].eq(subject_id)
            & authoritative["primitive"].eq(primitive)
        ]
        if len(matching) != 1:
            raise ValueError(
                f"frozen population baseline is missing for {subject_id!r}/{primitive}"
            )
        anchor = _validate_frozen_population_anchor(
            primitive_table=primitive_table,
            primitive=str(primitive),
            held_out_subject=subject_id,
            baseline=matching.iloc[0],
        )
        values = pd.to_numeric(group["original_value"], errors="coerce").to_numpy(
            dtype=float
        )
        center, selected_mad = _normalized_mad(values)
        scale_floor = 0.1 * anchor
        if np.isfinite(selected_mad) and selected_mad > 0:
            standardizer = max(selected_mad, scale_floor)
            source = "selected_day_mad"
            floor_applied = bool(selected_mad < scale_floor)
        else:
            standardizer = anchor
            source = "frozen_outer_population"
            floor_applied = False
        records.append(
            {
                "subject_id": subject_id,
                "primitive": primitive,
                "selected_count": len(group),
                "selected_center": center,
                "selected_day_mad": selected_mad,
                "frozen_population_scale": anchor,
                "scale_floor": scale_floor,
                "standardizer": float(standardizer),
                "scale_source": source,
                "floor_applied": floor_applied,
                "status": "observed",
            }
        )
    return pd.DataFrame.from_records(records)


def _rank_correlation(
    original: np.ndarray, masked: np.ndarray
) -> tuple[float, float, int, str]:
    finite = np.isfinite(original) & np.isfinite(masked)
    count = int(finite.sum())
    if count < 3:
        return float("nan"), 0.0, count, "ineligible_fewer_than_three"
    x = original[finite]
    y = masked[finite]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), 0.0, count, "ineligible_constant"
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    rho = float(np.corrcoef(x_rank, y_rank)[0, 1])
    if not np.isfinite(rho):
        return float("nan"), 0.0, count, "ineligible_degenerate"
    return rho, rho, count, "ok"


_SELECTED_CELL_KEYS = ("subject_id", "sensor_day_id", "primitive")
_REPLAY_TOPOLOGY_KEYS = (*_SELECTED_CELL_KEYS, "scenario", "draw")
_REVIEWED_G1_SCHEMA = "reviewed_g1_pair_summary_v1"
_REVIEWED_G1_CONFIG = {
    "evaluation_start_day": 22,
    "bootstrap_draws": 10_000,
    "shift_draws": 1_000,
    "root_seed": 20260807,
    "primary_high_coverage_threshold": 0.8,
    "raw_r_minimum": 0.30,
    "holm_alpha": 0.05,
    "minimum_positive_participants": 7,
    "participant_denominator": 10,
    "usage_availability_policy": "matching_screen_proxy",
}
_G1_PAIR_DOMAINS = {
    "digital_load": "digital",
    "digital_disengagement": "digital",
    "physical_load": "physical",
    "light_exposure": "light",
}


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact must be finite canonical JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_common_g2_chunk_fingerprints(
    chunks: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    """Require exact, complete, identical provenance across stream chunks."""
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise TypeError("G2 chunks must be a sequence of mappings")
    if not chunks:
        raise ValueError("G2 chunks must not be empty")
    expected_keys = set(G2_COMMON_FINGERPRINT_KEYS)
    common: dict[str, str] | None = None
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise TypeError(f"G2 chunk {index} must be a mapping")
        fingerprints = chunk.get("common_fingerprints")
        if not isinstance(fingerprints, Mapping) or set(fingerprints) != expected_keys:
            raise ValueError(f"G2 chunk {index} common fingerprint schema is not exact")
        normalized = {key: str(fingerprints[key]) for key in G2_COMMON_FINGERPRINT_KEYS}
        if not normalized["root_seed"]:
            raise ValueError(f"G2 chunk {index} root_seed fingerprint is empty")
        for key in G2_COMMON_FINGERPRINT_KEYS:
            if key == "root_seed":
                continue
            value = normalized[key].lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"G2 chunk {index} {key} is not a SHA-256")
            normalized[key] = value
        if common is None:
            common = normalized
        elif normalized != common:
            differing = [
                key
                for key in G2_COMMON_FINGERPRINT_KEYS
                if normalized[key] != common[key]
            ]
            raise ValueError(
                "G2 chunk common fingerprint mismatch: " + ", ".join(differing)
            )
    if common is None:
        raise RuntimeError("unreachable empty G2 chunk fingerprint state")
    return common


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def compare_empirical_replay_artifacts(
    first: Mapping[str, object], second: Mapping[str, object]
) -> dict[str, object]:
    """Compare bytes from two separately preserved empirical exact runs."""
    required = {"cell_replays", "mask_intervals"}
    normalized_runs: list[dict[str, Path]] = []
    for run_name, artifacts in (("first", first), ("second", second)):
        if not isinstance(artifacts, Mapping) or set(artifacts) != required:
            raise ValueError(
                f"{run_name} independent empirical artifact schema is not exact"
            )
        paths = {name: Path(artifacts[name]).resolve() for name in sorted(required)}
        if len(set(paths.values())) != len(paths):
            raise ValueError("independent empirical artifacts require separate paths")
        for path in paths.values():
            if not path.is_file():
                raise ValueError(f"independent empirical artifact is missing: {path}")
        normalized_runs.append(paths)
    if any(
        normalized_runs[0][name] == normalized_runs[1][name] for name in required
    ):
        raise ValueError("independent empirical runs require separate paths")

    records: list[dict[str, str]] = []
    for paths in normalized_runs:
        records.append(
            {
                "cell_replays_path": str(paths["cell_replays"]),
                "cell_replays_sha256": _sha256_file(paths["cell_replays"]),
                "mask_intervals_path": str(paths["mask_intervals"]),
                "mask_intervals_sha256": _sha256_file(paths["mask_intervals"]),
            }
        )
    if (
        records[0]["cell_replays_sha256"]
        != records[1]["cell_replays_sha256"]
        or records[0]["mask_intervals_sha256"]
        != records[1]["mask_intervals_sha256"]
    ):
        raise ValueError("independent empirical exact-run artifact hashes differ")
    return {
        "independent_exact50_runs": 2,
        "hashes_identical": True,
        "first": records[0],
        "second": records[1],
    }


def canonical_g1_artifact_sha256(artifact: Mapping[str, object]) -> str:
    """Hash every reviewed G1 field except the self-declared digest."""
    if not isinstance(artifact, Mapping):
        raise TypeError("reviewed G1 artifact must be a mapping")
    payload = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    return _canonical_json_sha256(payload)


def validate_reviewed_g1_artifact(
    artifact: Mapping[str, object], *, expected_sha256: str
) -> pd.DataFrame:
    """Validate and bind the independently reviewed production G1 result."""
    if not isinstance(artifact, Mapping):
        raise TypeError("reviewed G1 artifact must be a mapping")
    required_top = {
        "schema_version",
        "config",
        "source",
        "pair_summary",
        "artifact_sha256",
    }
    if set(artifact) != required_top:
        raise ValueError("reviewed G1 artifact top-level schema is not exact")
    if artifact["schema_version"] != _REVIEWED_G1_SCHEMA:
        raise ValueError("reviewed G1 artifact schema version is not frozen")
    expected = str(expected_sha256).lower()
    embedded = str(artifact["artifact_sha256"]).lower()
    actual = canonical_g1_artifact_sha256(artifact)
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or embedded != actual
        or expected != actual
    ):
        raise ValueError("reviewed G1 artifact SHA-256 does not match the bound digest")
    if artifact["config"] != _REVIEWED_G1_CONFIG:
        raise ValueError("reviewed G1 artifact does not match the frozen G1 config")
    source = artifact["source"]
    if not isinstance(source, Mapping):
        raise ValueError("reviewed G1 artifact source must be a mapping")
    if (
        set(source) != {"review_status", "implementation_commits", "report"}
        or source["review_status"] != "CLEAN"
        or source["implementation_commits"] != ["18eadac", "f246931"]
        or source["report"] != "task-4a-report.md"
    ):
        raise ValueError("reviewed G1 artifact source provenance is not exact")
    pair_summary = artifact["pair_summary"]
    if not isinstance(pair_summary, list):
        raise ValueError("reviewed G1 pair_summary must be a JSON row list")
    frame = pd.DataFrame.from_records(pair_summary)
    required_pair = (
        "pair",
        "domain",
        "left",
        "right",
        "strong_pair",
        "r_gate",
        "shift_holm_gate",
        "direction_gate",
        "raw_lopo_gate",
        "adjusted_sign_gate",
        "status",
    )
    _required_columns(frame, required_pair, "reviewed G1 pair_summary")
    contracts = {pair.name: pair for pair in PRIMARY_PAIRS}
    if len(frame) != len(contracts) or frame["pair"].duplicated().any():
        raise ValueError("reviewed G1 pair_summary must contain four unique pairs")
    if set(frame["pair"]) != set(contracts):
        raise ValueError("reviewed G1 pair_summary pair identities are not exact")
    boolean_columns = (
        "strong_pair",
        "r_gate",
        "shift_holm_gate",
        "direction_gate",
        "raw_lopo_gate",
        "adjusted_sign_gate",
    )
    for column in boolean_columns:
        if not frame[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"reviewed G1 {column} must contain true booleans")
    for row in frame.itertuples(index=False):
        contract = contracts[str(row.pair)]
        if str(row.domain) != _G1_PAIR_DOMAINS[contract.name]:
            raise ValueError("reviewed G1 pair domain relabel is forbidden")
        if str(row.left) != contract.left or str(row.right) != contract.right:
            raise ValueError("reviewed G1 pair endpoint crosswire is forbidden")
        recomputed_strong = bool(
            row.r_gate
            and row.shift_holm_gate
            and row.direction_gate
            and row.raw_lopo_gate
            and row.adjusted_sign_gate
        )
        if bool(row.strong_pair) != recomputed_strong or str(row.status) != "ok":
            raise ValueError("reviewed G1 strong_pair/status is internally inconsistent")
    order = {pair.name: index for index, pair in enumerate(PRIMARY_PAIRS)}
    frame["__pair_order"] = frame["pair"].map(order)
    frame = frame.sort_values("__pair_order", kind="stable").drop(
        columns="__pair_order"
    ).reset_index(drop=True)
    frame["g1_artifact_sha256"] = actual
    frame["g1_config_sha256"] = _canonical_json_sha256(artifact["config"])
    frame["g1_artifact_schema"] = _REVIEWED_G1_SCHEMA
    frame["g1_pair_summary_sha256"] = _canonical_json_sha256(
        frame.drop(
            columns=[
                "g1_artifact_sha256",
                "g1_config_sha256",
                "g1_artifact_schema",
            ]
        ).to_dict(orient="records")
    )
    return frame


def validate_selected_replay_topology(
    cell_replays: pd.DataFrame,
    *,
    selected_days: pd.DataFrame,
    expected_scenarios: Sequence[str],
    expected_draws: int,
) -> None:
    """Require the exact selected-cell x scenario x draw Cartesian product."""
    _required_columns(cell_replays, _REPLAY_TOPOLOGY_KEYS, "cell_replays")
    _required_columns(selected_days, _SELECTED_CELL_KEYS, "selected_days")
    if selected_days.empty:
        raise ValueError("authoritative selected_days must not be empty")
    if selected_days.loc[:, _SELECTED_CELL_KEYS].isna().any().any():
        raise ValueError("authoritative selected_days keys must be complete")
    if selected_days.duplicated(list(_SELECTED_CELL_KEYS)).any():
        raise ValueError("authoritative selected_days keys must be unique")
    scenarios = tuple(expected_scenarios)
    if (
        not scenarios
        or len(scenarios) != len(set(scenarios))
        or any(not isinstance(value, str) or not value for value in scenarios)
    ):
        raise ValueError("expected_scenarios must be nonempty unique strings")
    if (
        isinstance(expected_draws, (bool, np.bool_))
        or int(expected_draws) != expected_draws
        or expected_draws <= 0
    ):
        raise ValueError("expected_draws must be a positive integer")
    draw_count = int(expected_draws)
    observed = cell_replays.loc[:, _REPLAY_TOPOLOGY_KEYS].copy()
    if observed.isna().any().any():
        raise ValueError("Cartesian replay topology keys must be complete")
    observed_draws = pd.to_numeric(observed["draw"], errors="coerce")
    if not observed_draws.map(
        lambda value: np.isfinite(value)
        and value >= 0
        and value == int(value)
        and value < draw_count
    ).all():
        raise ValueError("Cartesian replay topology has an invalid draw")
    observed["draw"] = observed_draws.astype(int)
    if observed.duplicated(list(_REPLAY_TOPOLOGY_KEYS)).any():
        raise ValueError("Cartesian replay topology contains duplicate cells")

    expected_records: list[tuple[Any, ...]] = []
    for selected in selected_days.loc[:, _SELECTED_CELL_KEYS].itertuples(
        index=False, name=None
    ):
        for scenario in scenarios:
            for draw in range(draw_count):
                expected_records.append((*selected, scenario, draw))
    expected = set(expected_records)
    actual = set(observed.itertuples(index=False, name=None))
    if actual != expected:
        missing = len(expected.difference(actual))
        extra = len(actual.difference(expected))
        raise ValueError(
            "Cartesian replay topology must equal authoritative selected_days "
            f"x scenarios x draws exactly (missing={missing}, extra={extra})"
        )


def aggregate_hierarchical_reliability(
    cell_replays: pd.DataFrame,
    *,
    selected_days: pd.DataFrame,
    expected_scenarios: Sequence[str],
    expected_draws: int,
    participant_roster: Sequence[object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate only cell/day -> participant-draw -> participant -> macro."""
    required = (
        "subject_id",
        "sensor_day_id",
        "primitive",
        "scenario",
        "draw",
        "original_value",
        "masked_value",
        "retained",
        "absolute_error",
        "standardized_absolute_error",
    )
    _required_columns(cell_replays, required, "cell_replays")
    if cell_replays.empty:
        raise ValueError("cell_replays must not be empty")
    topology_draws = int(expected_draws)
    validate_selected_replay_topology(
        cell_replays,
        selected_days=selected_days,
        expected_scenarios=expected_scenarios,
        expected_draws=topology_draws,
    )
    if cell_replays.duplicated(
        ["subject_id", "sensor_day_id", "primitive", "scenario", "draw"]
    ).any():
        raise ValueError("cell replay identities must be unique")
    original = pd.to_numeric(cell_replays["original_value"], errors="coerce")
    if not np.isfinite(original.to_numpy(dtype=float)).all():
        raise ValueError("every selected original cell must remain finite")
    if not cell_replays["retained"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("retained must contain true booleans")
    draw_values = pd.to_numeric(cell_replays["draw"], errors="coerce")
    if not draw_values.map(
        lambda value: np.isfinite(value) and value >= 0 and value == int(value)
    ).all():
        raise ValueError("draw must contain nonnegative integers")

    roster = (
        tuple(sorted(cell_replays["subject_id"].unique(), key=str))
        if participant_roster is None
        else tuple(participant_roster)
    )
    if not roster or len(set(map(str, roster))) != len(roster):
        raise ValueError("participant_roster must be nonempty and unique")
    observed_subjects = set(cell_replays["subject_id"].map(str))
    if not observed_subjects.issubset(set(map(str, roster))):
        raise ValueError("participant_roster is missing observed participants")
    draw_count = int(expected_draws)
    if draw_count <= 0 or not draw_values.lt(draw_count).all():
        raise ValueError("expected_draws must cover every nonnegative draw")

    day = cell_replays.copy(deep=True)
    day["draw"] = draw_values.astype(int)
    day["retained"] = day["retained"].astype(bool)
    day["success"] = (
        day["retained"]
        & pd.to_numeric(
            day["standardized_absolute_error"], errors="coerce"
        ).le(0.20)
    )
    day["aggregation_level"] = "draw_day"
    day["__subject_sort"] = day["subject_id"].map(str)
    day["__primitive_order"] = day["primitive"].map(
        {name: index for index, name in enumerate(CORE_G2_PRIMITIVES)}
    )
    day = day.sort_values(
        [
            "__primitive_order",
            "scenario",
            "__subject_sort",
            "draw",
            "sensor_day_id",
        ],
        kind="stable",
    ).drop(columns=["__subject_sort", "__primitive_order"]).reset_index(drop=True)

    combinations = day.loc[:, ["primitive", "scenario"]].drop_duplicates()
    participant_draw_records: list[dict[str, Any]] = []
    for combination in combinations.itertuples(index=False):
        primitive = combination.primitive
        scenario = combination.scenario
        subset = day.loc[
            day["primitive"].eq(primitive) & day["scenario"].eq(scenario)
        ]
        for subject_id in roster:
            subject_rows = subset.loc[subset["subject_id"].eq(subject_id)]
            for draw in range(draw_count):
                group = subject_rows.loc[subject_rows["draw"].eq(draw)]
                denominator = len(group)
                retained = group["retained"].to_numpy(dtype=bool)
                retained_count = int(retained.sum())
                masked = pd.to_numeric(
                    group["masked_value"], errors="coerce"
                ).to_numpy(dtype=float)
                original_values = pd.to_numeric(
                    group["original_value"], errors="coerce"
                ).to_numpy(dtype=float)
                rho, rho_gate, paired_count, rho_status = _rank_correlation(
                    original_values, masked
                )
                real_error = pd.to_numeric(
                    group["absolute_error"], errors="coerce"
                ).to_numpy(dtype=float)
                standardized_error = pd.to_numeric(
                    group["standardized_absolute_error"], errors="coerce"
                ).to_numpy(dtype=float)
                real_valid = retained & np.isfinite(real_error)
                standardized_valid = retained & np.isfinite(standardized_error)
                participant_draw_records.append(
                    {
                        "subject_id": subject_id,
                        "primitive": primitive,
                        "scenario": scenario,
                        "draw": draw,
                        "selected_denominator": denominator,
                        "retained_count": retained_count,
                        "retention": (
                            float(retained_count / denominator)
                            if denominator
                            else 0.0
                        ),
                        "real_error_count": int(real_valid.sum()),
                        "real_mae": (
                            float(real_error[real_valid].mean())
                            if bool(real_valid.any())
                            else float("nan")
                        ),
                        "standardized_error_count": int(
                            standardized_valid.sum()
                        ),
                        "standardized_mae": (
                            float(standardized_error[standardized_valid].mean())
                            if bool(standardized_valid.any())
                            else float("nan")
                        ),
                        "paired_count": paired_count,
                        "rank_rho": rho,
                        "rank_rho_gate": rho_gate,
                        "rho_status": rho_status,
                        "status": "ok" if denominator else "no_selected_days",
                    }
                )
    participant_draw = pd.DataFrame.from_records(participant_draw_records)
    participant_draw["__subject_sort"] = participant_draw["subject_id"].map(str)
    participant_draw["__primitive_order"] = participant_draw["primitive"].map(
        {name: index for index, name in enumerate(CORE_G2_PRIMITIVES)}
    )
    participant_draw = participant_draw.sort_values(
        ["__primitive_order", "scenario", "__subject_sort", "draw"],
        kind="stable",
    ).drop(columns=["__subject_sort", "__primitive_order"]).reset_index(drop=True)

    participant_records: list[dict[str, Any]] = []
    for (primitive, scenario, subject_id), group in participant_draw.groupby(
        ["primitive", "scenario", "subject_id"], sort=False, dropna=False
    ):
        standardized = pd.to_numeric(
            group["standardized_mae"], errors="coerce"
        ).to_numpy(dtype=float)
        real = pd.to_numeric(group["real_mae"], errors="coerce").to_numpy(
            dtype=float
        )
        finite_standardized = np.isfinite(standardized)
        finite_real = np.isfinite(real)
        complete = bool(
            len(group) == draw_count
            and group["selected_denominator"].gt(0).all()
            and finite_standardized.all()
            and finite_real.all()
        )
        participant_records.append(
            {
                "subject_id": subject_id,
                "primitive": primitive,
                "scenario": scenario,
                "draw_denominator": draw_count,
                "draw_count": len(group),
                "finite_error_draws": int(finite_standardized.sum()),
                "ineligible_draws": int(group["rho_status"].ne("ok").sum()),
                "rank_rho": float(group["rank_rho_gate"].mean()),
                "standardized_mae": (
                    float(standardized[finite_standardized].mean())
                    if bool(finite_standardized.any())
                    else float("nan")
                ),
                "real_mae": (
                    float(real[finite_real].mean())
                    if bool(finite_real.any())
                    else float("nan")
                ),
                "retention": float(group["retention"].mean()),
                "complete": complete,
                "status": "ok" if complete else "incomplete_or_degenerate",
            }
        )
    participant = pd.DataFrame.from_records(participant_records)
    participant["__subject_sort"] = participant["subject_id"].map(str)
    participant["__primitive_order"] = participant["primitive"].map(
        {name: index for index, name in enumerate(CORE_G2_PRIMITIVES)}
    )
    participant = participant.sort_values(
        ["__primitive_order", "scenario", "__subject_sort"], kind="stable"
    ).drop(columns=["__subject_sort", "__primitive_order"]).reset_index(drop=True)

    macro_records: list[dict[str, Any]] = []
    for (primitive, scenario), group in participant.groupby(
        ["primitive", "scenario"], sort=False, dropna=False
    ):
        standardized = pd.to_numeric(
            group["standardized_mae"], errors="coerce"
        ).to_numpy(dtype=float)
        real = pd.to_numeric(group["real_mae"], errors="coerce").to_numpy(
            dtype=float
        )
        finite_standardized = np.isfinite(standardized)
        finite_real = np.isfinite(real)
        complete = bool(
            len(group) == len(roster)
            and group["complete"].astype(bool).all()
            and finite_standardized.all()
            and finite_real.all()
        )
        macro_records.append(
            {
                "primitive": primitive,
                "scenario": scenario,
                "participant_denominator": len(roster),
                "participant_count": len(group),
                "finite_error_participants": int(finite_standardized.sum()),
                "ineligible_participant_draws": int(
                    group["ineligible_draws"].sum()
                ),
                "rank_rho": float(group["rank_rho"].mean()),
                "standardized_mae": (
                    float(standardized[finite_standardized].mean())
                    if bool(finite_standardized.any())
                    else float("nan")
                ),
                "real_mae": (
                    float(real[finite_real].mean())
                    if bool(finite_real.any())
                    else float("nan")
                ),
                "retention": float(group["retention"].mean()),
                "complete": complete,
                "status": "ok" if complete else "incomplete_or_degenerate",
            }
        )
    macro = pd.DataFrame.from_records(macro_records)
    return day, participant_draw, participant, macro


def evaluate_reliability_gates(
    participant_macro_metrics: pd.DataFrame,
    upstream_pair_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the immutable 20%-mask primitive, pair, and domain AND gates."""
    _required_columns(
        participant_macro_metrics,
        (
            "primitive",
            "scenario",
            "rank_rho",
            "standardized_mae",
            "retention",
            "participant_denominator",
            "complete",
            "status",
        ),
        "participant_macro_metrics",
    )
    _required_columns(
        upstream_pair_summary,
        ("pair", "domain", "strong_pair"),
        "upstream_pair_summary",
    )
    pair_contract = {pair.name: pair for pair in PRIMARY_PAIRS}
    if (
        set(upstream_pair_summary["pair"]) != set(pair_contract)
        or upstream_pair_summary["pair"].duplicated().any()
    ):
        raise ValueError("upstream_pair_summary must contain each frozen pair exactly once")
    if not upstream_pair_summary["strong_pair"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("upstream strong_pair must contain true booleans")

    primary = participant_macro_metrics.loc[
        participant_macro_metrics["scenario"].eq("contiguous_20pct")
    ]
    if primary["primitive"].duplicated().any():
        raise ValueError("primary participant macro rows must be unique by primitive")
    primitive_records: list[dict[str, Any]] = []
    for primitive in CORE_G2_PRIMITIVES:
        matching = primary.loc[primary["primitive"].eq(primitive)]
        if len(matching) == 1:
            row = matching.iloc[0]
            rho = float(row["rank_rho"])
            mae = float(row["standardized_mae"])
            retention = float(row["retention"])
            participant_denominator = int(row["participant_denominator"])
            complete = bool(
                bool(row["complete"])
                and str(row["status"]) == "ok"
                and participant_denominator == 10
            )
        else:
            rho = mae = retention = float("nan")
            complete = False
            participant_denominator = 0
        rho_gate = bool(np.isfinite(rho) and rho >= 0.85)
        mae_gate = bool(np.isfinite(mae) and mae <= 0.20)
        retention_gate = bool(np.isfinite(retention) and retention >= 0.80)
        passed = complete and rho_gate and mae_gate and retention_gate
        primitive_records.append(
            {
                "primitive": primitive,
                "scenario": "contiguous_20pct",
                "rank_rho": rho,
                "standardized_mae": mae,
                "retention": retention,
                "participant_denominator": participant_denominator,
                "complete_gate": complete,
                "rho_gate": rho_gate,
                "mae_gate": mae_gate,
                "retention_gate": retention_gate,
                "primitive_pass": passed,
                "status": "pass" if passed else "fail",
            }
        )
    primitive_gates = pd.DataFrame.from_records(primitive_records)
    primitive_lookup = primitive_gates.set_index("primitive")["primitive_pass"]

    pair_records: list[dict[str, Any]] = []
    for pair in PRIMARY_PAIRS:
        upstream = upstream_pair_summary.loc[
            upstream_pair_summary["pair"].eq(pair.name)
        ].iloc[0]
        if "left" in upstream_pair_summary and upstream["left"] != pair.left:
            raise ValueError("upstream frozen pair left endpoint is inconsistent")
        if "right" in upstream_pair_summary and upstream["right"] != pair.right:
            raise ValueError("upstream frozen pair right endpoint is inconsistent")
        left_pass = bool(primitive_lookup.loc[pair.left])
        right_pass = bool(primitive_lookup.loc[pair.right])
        both = left_pass and right_pass
        strong = bool(upstream["strong_pair"])
        passed = strong and both
        pair_records.append(
            {
                "pair": pair.name,
                "domain": upstream["domain"],
                "left": pair.left,
                "right": pair.right,
                "upstream_strong_pair": strong,
                "left_primitive_pass": left_pass,
                "right_primitive_pass": right_pass,
                "both_primitives_pass": both,
                "pair_pass": passed,
                "status": "pass" if passed else "fail",
            }
        )
    pair_gates = pd.DataFrame.from_records(pair_records)

    domain_records: list[dict[str, Any]] = []
    for domain in ("digital", "physical", "light"):
        rows = pair_gates.loc[pair_gates["domain"].eq(domain)]
        upstream_count = int(rows["upstream_strong_pair"].sum())
        passing = rows.loc[rows["pair_pass"], "pair"].tolist()
        domain_records.append(
            {
                "domain": domain,
                "pair_denominator": len(rows),
                "upstream_strong_pairs": upstream_count,
                "g2_passing_strong_pairs": len(passing),
                "passing_pairs": json.dumps(passing, ensure_ascii=False),
                "domain_pass": bool(passing),
                "status": "pass" if passing else "fail",
            }
        )
    domain_gates = pd.DataFrame.from_records(domain_records)

    decision_records: list[dict[str, Any]] = []
    for row in primitive_gates.itertuples(index=False):
        for gate, observed, comparator, threshold, passed in (
            ("rank_rho_at_least_0.85", row.rank_rho, ">=", 0.85, row.rho_gate),
            (
                "standardized_mae_at_most_0.20",
                row.standardized_mae,
                "<=",
                0.20,
                row.mae_gate,
            ),
            ("retention_at_least_0.80", row.retention, ">=", 0.80, row.retention_gate),
            ("all_primitive_gates", row.primitive_pass, "is", True, row.primitive_pass),
        ):
            decision_records.append(
                {
                    "level": "primitive",
                    "primitive": row.primitive,
                    "pair": pd.NA,
                    "domain": pd.NA,
                    "gate": gate,
                    "observed": observed,
                    "comparator": comparator,
                    "threshold": threshold,
                    "passed": bool(passed),
                    "can_change_domain_decision": True,
                    "status": "pass" if passed else "fail",
                }
            )
    for row in pair_gates.itertuples(index=False):
        decision_records.append(
            {
                "level": "pair",
                "primitive": pd.NA,
                "pair": row.pair,
                "domain": row.domain,
                "gate": "upstream_strong_and_both_primitives",
                "observed": bool(row.pair_pass),
                "comparator": "is",
                "threshold": True,
                "passed": bool(row.pair_pass),
                "can_change_domain_decision": True,
                "status": row.status,
            }
        )
    for row in domain_gates.itertuples(index=False):
        decision_records.append(
            {
                "level": "domain",
                "primitive": pd.NA,
                "pair": pd.NA,
                "domain": row.domain,
                "gate": "at_least_one_upstream_strong_pair_passes_g2",
                "observed": row.g2_passing_strong_pairs,
                "comparator": ">=",
                "threshold": 1,
                "passed": bool(row.domain_pass),
                "can_change_domain_decision": True,
                "status": row.status,
            }
        )
    passed_count = int(primitive_gates["primitive_pass"].sum())
    decision_records.append(
        {
            "level": "descriptive",
            "primitive": pd.NA,
            "pair": pd.NA,
            "domain": pd.NA,
            "gate": "descriptive_primitives_passed",
            "observed": passed_count,
            "comparator": ">=",
            "threshold": 6,
            "passed": passed_count >= 6,
            "can_change_domain_decision": False,
            "status": "descriptive_only",
        }
    )
    return (
        primitive_gates,
        pair_gates,
        domain_gates,
        pd.DataFrame.from_records(decision_records),
    )


def _binary_auc(success: np.ndarray, score: np.ndarray) -> tuple[float, str]:
    if not np.isfinite(score).all():
        return float("nan"), "incomplete_score"
    positives = success.astype(bool)
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan"), "degenerate_one_class"
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    auc = (
        float(ranks[positives].sum())
        - positive_count * (positive_count + 1) / 2.0
    ) / float(positive_count * negative_count)
    return float(auc), "ok"


def _success_spearman(
    success: np.ndarray, score: np.ndarray
) -> tuple[float, str]:
    if len(success) < 3 or not np.isfinite(score).all():
        return float("nan"), "ineligible_or_incomplete"
    if np.unique(success).size < 2 or np.unique(score).size < 2:
        return float("nan"), "degenerate_constant"
    success_rank = pd.Series(success.astype(float)).rank(method="average")
    score_rank = pd.Series(score).rank(method="average")
    value = float(np.corrcoef(success_rank, score_rank)[0, 1])
    return (value, "ok") if np.isfinite(value) else (float("nan"), "degenerate")


def evaluate_confidence_ablation(
    cell_replays: pd.DataFrame,
    *,
    selected_days: pd.DataFrame,
    expected_draws: int,
    participant_roster: Sequence[object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank primary-mask success with three frozen confidence variants."""
    required = (
        "subject_id",
        "sensor_day_id",
        "primitive",
        "scenario",
        "draw",
        "retained",
        "standardized_absolute_error",
        "masked_value_only_confidence",
        "masked_coverage_only_confidence",
        "masked_quality_confidence",
    )
    _required_columns(cell_replays, required, "cell_replays")
    primary = cell_replays.loc[
        cell_replays["scenario"].eq("contiguous_20pct")
    ].copy()
    if primary.empty:
        raise ValueError("confidence ablation requires contiguous_20pct rows")
    primary_draws = pd.to_numeric(primary["draw"], errors="coerce")
    topology_draws = int(expected_draws)
    validate_selected_replay_topology(
        primary,
        selected_days=selected_days,
        expected_scenarios=("contiguous_20pct",),
        expected_draws=topology_draws,
    )
    roster = (
        tuple(sorted(primary["subject_id"].unique(), key=str))
        if participant_roster is None
        else tuple(participant_roster)
    )
    draw_values = pd.to_numeric(primary["draw"], errors="coerce").astype(int)
    primary["draw"] = draw_values
    draw_count = (
        topology_draws
    )
    success = (
        primary["retained"].astype(bool)
        & pd.to_numeric(
            primary["standardized_absolute_error"], errors="coerce"
        ).le(0.20)
    )
    primary["success"] = success
    variants = (
        ("value_only", "masked_value_only_confidence"),
        ("coverage_only", "masked_coverage_only_confidence"),
        ("full_quality_aware", "masked_quality_confidence"),
    )
    cell_tables: list[pd.DataFrame] = []
    for variant, column in variants:
        table = primary.loc[
            :,
            [
                "subject_id",
                "sensor_day_id",
                "primitive",
                "scenario",
                "draw",
                "success",
            ],
        ].copy()
        table["variant"] = variant
        table["score"] = pd.to_numeric(primary[column], errors="coerce")
        table["aggregation_level"] = "cell"
        table["roc_auc"] = np.nan
        table["spearman"] = np.nan
        table["status"] = "cell"
        cell_tables.append(table)
    long_cells = pd.concat(cell_tables, ignore_index=True)

    participant_draw_records: list[dict[str, Any]] = []
    primitives = tuple(
        primitive
        for primitive in CORE_G2_PRIMITIVES
        if primitive in set(primary["primitive"])
    )
    for primitive in primitives:
        primitive_rows = long_cells.loc[long_cells["primitive"].eq(primitive)]
        for subject_id in roster:
            subject_rows = primitive_rows.loc[
                primitive_rows["subject_id"].eq(subject_id)
            ]
            for draw in range(draw_count):
                draw_rows = subject_rows.loc[subject_rows["draw"].eq(draw)]
                for variant, _ in variants:
                    group = draw_rows.loc[draw_rows["variant"].eq(variant)]
                    binary = group["success"].to_numpy(dtype=bool)
                    scores = pd.to_numeric(
                        group["score"], errors="coerce"
                    ).to_numpy(dtype=float)
                    auc, auc_status = _binary_auc(binary, scores)
                    spearman, spearman_status = _success_spearman(binary, scores)
                    status = (
                        "ok"
                        if auc_status == "ok" and spearman_status == "ok"
                        else f"{auc_status}|{spearman_status}"
                    )
                    participant_draw_records.append(
                        {
                            "subject_id": subject_id,
                            "sensor_day_id": pd.NA,
                            "primitive": primitive,
                            "scenario": "contiguous_20pct",
                            "draw": draw,
                            "success": pd.NA,
                            "variant": variant,
                            "score": np.nan,
                            "aggregation_level": "participant_draw",
                            "cell_denominator": len(group),
                            "success_count": int(binary.sum()),
                            "roc_auc": auc,
                            "spearman": spearman,
                            "status": status,
                        }
                    )
    participant_draw = pd.DataFrame.from_records(participant_draw_records)
    all_rows = pd.concat([long_cells, participant_draw], ignore_index=True, sort=False)

    participant_records: list[dict[str, Any]] = []
    for (primitive, subject_id, variant), group in participant_draw.groupby(
        ["primitive", "subject_id", "variant"], sort=False, dropna=False
    ):
        auc = pd.to_numeric(group["roc_auc"], errors="coerce").to_numpy(dtype=float)
        spearman = pd.to_numeric(
            group["spearman"], errors="coerce"
        ).to_numpy(dtype=float)
        complete = bool(
            len(group) == draw_count
            and np.isfinite(auc).all()
            and np.isfinite(spearman).all()
        )
        participant_records.append(
            {
                "subject_id": subject_id,
                "primitive": primitive,
                "variant": variant,
                "draw_denominator": draw_count,
                "finite_auc_draws": int(np.isfinite(auc).sum()),
                "finite_spearman_draws": int(np.isfinite(spearman).sum()),
                "roc_auc": float(auc.mean()) if complete else float("nan"),
                "spearman": float(spearman.mean()) if complete else float("nan"),
                "complete": complete,
                "status": "ok" if complete else "degenerate",
            }
        )
    participant = pd.DataFrame.from_records(participant_records)
    summary_records: list[dict[str, Any]] = []
    for (primitive, variant), group in participant.groupby(
        ["primitive", "variant"], sort=False, dropna=False
    ):
        auc = pd.to_numeric(group["roc_auc"], errors="coerce").to_numpy(dtype=float)
        spearman = pd.to_numeric(
            group["spearman"], errors="coerce"
        ).to_numpy(dtype=float)
        complete = bool(
            len(group) == len(roster)
            and np.isfinite(auc).all()
            and np.isfinite(spearman).all()
        )
        summary_records.append(
            {
                "primitive": primitive,
                "scenario": "contiguous_20pct",
                "variant": variant,
                "participant_denominator": len(roster),
                "finite_auc_participants": int(np.isfinite(auc).sum()),
                "finite_spearman_participants": int(np.isfinite(spearman).sum()),
                "roc_auc": float(auc.mean()) if complete else float("nan"),
                "spearman": float(spearman.mean()) if complete else float("nan"),
                "status": "ok" if complete else "degenerate",
            }
        )
    summary = pd.DataFrame.from_records(summary_records)
    summary["auc_delta_vs_coverage"] = np.nan
    summary["spearman_delta_vs_coverage"] = np.nan
    summary["beats_coverage_both_metrics"] = False
    for primitive in summary["primitive"].unique():
        rows = summary.loc[summary["primitive"].eq(primitive)]
        coverage = rows.loc[rows["variant"].eq("coverage_only")]
        full = rows.loc[rows["variant"].eq("full_quality_aware")]
        if len(coverage) != 1 or len(full) != 1:
            continue
        coverage_row = coverage.iloc[0]
        full_index = full.index[0]
        auc_delta = float(full.iloc[0]["roc_auc"] - coverage_row["roc_auc"])
        spearman_delta = float(
            full.iloc[0]["spearman"] - coverage_row["spearman"]
        )
        summary.loc[full_index, "auc_delta_vs_coverage"] = auc_delta
        summary.loc[full_index, "spearman_delta_vs_coverage"] = spearman_delta
        summary.loc[full_index, "beats_coverage_both_metrics"] = bool(
            full.iloc[0]["status"] == "ok"
            and coverage_row["status"] == "ok"
            and auc_delta > 0
            and spearman_delta > 0
        )
    return all_rows, summary


def _validated_g1_summary_provenance(
    upstream_pair_summary: pd.DataFrame,
) -> tuple[str, str]:
    provenance_columns = (
        "g1_artifact_sha256",
        "g1_config_sha256",
        "g1_artifact_schema",
        "g1_pair_summary_sha256",
    )
    _required_columns(
        upstream_pair_summary,
        provenance_columns,
        "validated reviewed G1 pair_summary",
    )
    values: dict[str, str] = {}
    for column in provenance_columns:
        series = upstream_pair_summary[column].astype("string")
        unique = series.loc[series.notna()].unique()
        if len(unique) != 1:
            raise ValueError(f"reviewed G1 provenance {column} must be common")
        values[column] = str(unique[0])
    if values["g1_artifact_schema"] != _REVIEWED_G1_SCHEMA:
        raise ValueError("reviewed G1 provenance schema is not frozen")
    if values["g1_config_sha256"] != _canonical_json_sha256(_REVIEWED_G1_CONFIG):
        raise ValueError("reviewed G1 config provenance is crosswired")
    artifact_digest = values["g1_artifact_sha256"]
    if len(artifact_digest) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_digest
    ):
        raise ValueError("reviewed G1 artifact provenance digest is invalid")
    order = {pair.name: index for index, pair in enumerate(PRIMARY_PAIRS)}
    payload = upstream_pair_summary.drop(columns=list(provenance_columns)).copy()
    payload["__pair_order"] = payload["pair"].map(order)
    payload = payload.sort_values("__pair_order", kind="stable").drop(
        columns="__pair_order"
    ).reset_index(drop=True)
    if _canonical_json_sha256(payload.to_dict(orient="records")) != values[
        "g1_pair_summary_sha256"
    ]:
        raise ValueError("reviewed G1 pair_summary provenance is crosswired")
    return artifact_digest, values["g1_config_sha256"]


def finalize_g2_streamed_results(
    cell_replays: pd.DataFrame,
    *,
    selected_days: pd.DataFrame,
    participant_roster: Sequence[object],
    upstream_pair_summary: pd.DataFrame,
    root_seed: int | str,
    draws: int,
    scenarios: Sequence[str],
    max_days: int,
) -> G2FinalizedResult:
    """Rebuild all official G2 metrics and gates from exact streamed cells."""
    scenarios_tuple = tuple(scenarios)
    if (
        not scenarios_tuple
        or len(scenarios_tuple) != len(set(scenarios_tuple))
        or any(value not in FROZEN_MASK_SCENARIOS for value in scenarios_tuple)
    ):
        raise ValueError("streamed scenarios must be a nonempty unique frozen subset")
    if scenarios_tuple != tuple(
        value for value in FROZEN_MASK_SCENARIOS if value in scenarios_tuple
    ):
        raise ValueError("streamed scenarios must follow frozen scenario order")
    if isinstance(draws, (bool, np.bool_)) or int(draws) != draws or draws <= 0:
        raise ValueError("streamed draws must be a positive integer")
    if (
        isinstance(max_days, (bool, np.bool_))
        or int(max_days) != max_days
        or max_days <= 0
    ):
        raise ValueError("streamed max_days must be a positive integer")
    artifact_digest, config_digest = _validated_g1_summary_provenance(
        upstream_pair_summary
    )
    (
        draw_day,
        participant_draw,
        participant,
        participant_macro,
    ) = aggregate_hierarchical_reliability(
        cell_replays,
        selected_days=selected_days,
        expected_scenarios=scenarios_tuple,
        participant_roster=participant_roster,
        expected_draws=int(draws),
    )
    production_default_grid = bool(
        int(draws) == DEFAULT_MASK_DRAWS
        and scenarios_tuple == FROZEN_MASK_SCENARIOS
        and int(max_days) == MAX_SELECTED_DAYS
    )
    gate_input = participant_macro.copy()
    if not production_default_grid:
        gate_input["complete"] = False
        gate_input["status"] = "nonproduction_grid"
    primitive_gates, pair_gates, domain_gates, decision_table = (
        evaluate_reliability_gates(gate_input, upstream_pair_summary)
    )
    for table in (primitive_gates, pair_gates, domain_gates):
        table["production_eligible"] = production_default_grid
    decision_table["production_eligible"] = production_default_grid
    decision_table["root_seed"] = str(root_seed)
    decision_table["draws"] = int(draws)

    if "contiguous_20pct" in scenarios_tuple:
        confidence_ablation, confidence_summary = evaluate_confidence_ablation(
            cell_replays,
            selected_days=selected_days,
            participant_roster=participant_roster,
            expected_draws=int(draws),
        )
    else:
        confidence_ablation = pd.DataFrame()
        confidence_summary = pd.DataFrame()

    confidence_domain_pass: dict[str, bool] = {}
    for domain_row in domain_gates.itertuples(index=False):
        domain_pairs = pair_gates.loc[
            pair_gates["domain"].eq(domain_row.domain) & pair_gates["pair_pass"]
        ]
        primitives_for_domain = sorted(
            set(domain_pairs["left"]).union(set(domain_pairs["right"])),
            key=lambda name: CORE_G2_PRIMITIVES.index(name),
        )
        full_rows = (
            confidence_summary.loc[
                confidence_summary["primitive"].isin(primitives_for_domain)
                & confidence_summary["variant"].eq("full_quality_aware")
            ]
            if not confidence_summary.empty
            else pd.DataFrame()
        )
        condition = bool(
            domain_row.domain_pass
            and len(full_rows) == len(primitives_for_domain)
            and bool(full_rows["beats_coverage_both_metrics"].all())
        )
        confidence_domain_pass[str(domain_row.domain)] = condition
        decision_table = pd.concat(
            [
                decision_table,
                pd.DataFrame(
                    [
                        {
                            "level": "confidence_domain",
                            "primitive": pd.NA,
                            "pair": pd.NA,
                            "domain": domain_row.domain,
                            "gate": "full_quality_beats_coverage_auc_and_spearman",
                            "observed": int(
                                full_rows["beats_coverage_both_metrics"].sum()
                            )
                            if not full_rows.empty
                            else 0,
                            "comparator": "==",
                            "threshold": len(primitives_for_domain),
                            "passed": condition,
                            "can_change_domain_decision": False,
                            "status": "pass" if condition else "fail",
                            "production_eligible": production_default_grid,
                            "root_seed": str(root_seed),
                            "draws": int(draws),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    domain_gates["confidence_full_beats_coverage"] = domain_gates["domain"].map(
        confidence_domain_pass
    )
    manifest = pd.DataFrame(
        [
            {
                "stage": "G2_streamed_official_finalization",
                "root_seed": str(root_seed),
                "draws": int(draws),
                "scenario_grid": json.dumps(list(scenarios_tuple)),
                "max_selected_days": int(max_days),
                "participant_denominator": len(tuple(participant_roster)),
                "selected_cell_count": len(selected_days),
                "logical_replay_count": len(cell_replays),
                "production_default_grid": production_default_grid,
                "g1_artifact_sha256": artifact_digest,
                "g1_config_sha256": config_digest,
                "status": (
                    "complete" if production_default_grid else "nonproduction_grid"
                ),
            }
        ]
    )
    return G2FinalizedResult(
        draw_day_metrics=draw_day,
        participant_draw_metrics=participant_draw,
        participant_metrics=participant,
        participant_macro_metrics=participant_macro,
        primitive_gates=primitive_gates,
        pair_gates=pair_gates,
        domain_gates=domain_gates,
        decision_table=decision_table,
        confidence_ablation=confidence_ablation,
        confidence_summary=confidence_summary,
        run_manifest=manifest,
    )


def _shrinkage_fallback(
    calibration: Mapping[str, float | int],
    estimate: Mapping[str, Any],
    population_scale: float,
) -> str:
    if estimate["status"] != "observed":
        return "population_degenerate"
    n_eff = float(calibration["n_eff"])
    if n_eff < 3.0 or not np.isfinite(float(calibration["scale"])):
        return "population_insufficient_n_eff"
    unbounded = (
        float(estimate["scale_weight"]) * float(calibration["scale"])
        + (1.0 - float(estimate["scale_weight"])) * population_scale
    )
    return "scale_floor" if unbounded < float(estimate["scale_floor"]) else "none"


def _transfer_error(
    estimate: Mapping[str, Any],
    *,
    oracle_center: float,
    oracle_scale: float,
    population_scale: float,
) -> float:
    center = float(estimate["center"])
    scale = float(estimate["scale"])
    if not (
        np.isfinite(center)
        and np.isfinite(scale)
        and scale > 0
        and np.isfinite(oracle_center)
        and np.isfinite(oracle_scale)
        and oracle_scale > 0
        and np.isfinite(population_scale)
        and population_scale > 0
    ):
        return float("nan")
    location_error = abs(center - oracle_center) / population_scale
    scale_error = abs(float(np.log(scale / oracle_scale)))
    return float(0.5 * location_error + 0.5 * scale_error)


def evaluate_calibration_sensitivity(
    primitive_table: pd.DataFrame,
    calibration_replays: pd.DataFrame,
    *,
    frozen_population_baselines: pd.DataFrame,
    draws: int = DEFAULT_MASK_DRAWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-estimate only held-out day-14 calibration under primary masks.

    Other-participant population priors and held-out post-day-21 observations
    are immutable.  Future rows are used only as the predeclared scoring oracle.
    """
    _required_columns(
        primitive_table,
        ("subject_id", "sensor_day_id", "elapsed_day_index"),
        "primitive_table",
    )
    _required_columns(
        calibration_replays,
        (
            "subject_id",
            "sensor_day_id",
            "elapsed_day_index",
            "primitive",
            "draw",
            "original_value",
            "masked_value",
            "masked_quality_confidence",
        ),
        "calibration_replays",
    )
    if isinstance(draws, (bool, np.bool_)) or int(draws) != draws or draws <= 0:
        raise ValueError("draws must be a positive integer")
    draw_count = int(draws)
    elapsed = pd.to_numeric(
        calibration_replays["elapsed_day_index"], errors="coerce"
    )
    if elapsed.isna().any() or elapsed.gt(14).any():
        raise ValueError("calibration mask rows must use elapsed_day_index <= 14")
    draw_values = pd.to_numeric(calibration_replays["draw"], errors="coerce")
    if not draw_values.map(
        lambda value: np.isfinite(value)
        and value >= 0
        and value == int(value)
        and value < draw_count
    ).all():
        raise ValueError("calibration draw rows must match the requested draw grid")
    if calibration_replays.duplicated(
        ["subject_id", "sensor_day_id", "primitive", "draw"]
    ).any():
        raise ValueError("calibration replay keys must be unique")

    primitives = tuple(
        primitive
        for primitive in CORE_G2_PRIMITIVES
        if primitive in set(calibration_replays["primitive"])
    )
    unknown = sorted(set(calibration_replays["primitive"]).difference(primitives))
    if unknown:
        raise ValueError(f"calibration replay contains non-G2 primitives: {unknown}")
    confidence_tables = [
        compute_primitive_confidence(
            primitive_table,
            primitive_names=(primitive,),
            usage_policy="screen_proxy",
        )
        for primitive in primitives
    ]
    confidence = pd.concat(confidence_tables, ignore_index=True)
    authoritative = _population_rows(frozen_population_baselines)
    calendar = primitive_table.loc[
        :, ["subject_id", "sensor_day_id", "elapsed_day_index"]
    ].copy()
    if calendar["sensor_day_id"].duplicated().any():
        raise ValueError("primitive_table sensor_day_id must be unique")
    calendar["__elapsed"] = pd.to_numeric(
        calendar["elapsed_day_index"], errors="coerce"
    )
    replay_calendar = calibration_replays.merge(
        calendar.loc[:, ["subject_id", "sensor_day_id", "__elapsed"]],
        on=["subject_id", "sensor_day_id"],
        how="left",
        validate="many_to_one",
    )
    if replay_calendar["__elapsed"].isna().any() or not replay_calendar[
        "__elapsed"
    ].eq(elapsed.to_numpy()).all():
        raise ValueError("calibration replay calendar identity is inconsistent")

    records: list[dict[str, Any]] = []
    subjects = tuple(sorted(calibration_replays["subject_id"].unique(), key=str))
    for subject_id in subjects:
        for primitive in primitives:
            group_all = calibration_replays.loc[
                calibration_replays["subject_id"].eq(subject_id)
                & calibration_replays["primitive"].eq(primitive)
            ]
            if group_all.empty:
                raise ValueError(
                    f"calibration replay grid is missing {subject_id!r}/{primitive}"
                )
            expected_ids = set(
                primitive_table.loc[
                    primitive_table["subject_id"].eq(subject_id)
                    & pd.to_numeric(
                        primitive_table["elapsed_day_index"], errors="coerce"
                    ).le(14),
                    "sensor_day_id",
                ]
            )
            for draw in range(draw_count):
                draw_rows = group_all.loc[
                    pd.to_numeric(group_all["draw"], errors="coerce").eq(draw)
                ].sort_values("sensor_day_id", kind="stable")
                if set(draw_rows["sensor_day_id"]) != expected_ids:
                    raise ValueError(
                        "calibration replay must retain the complete <=14 calendar grid"
                    )

            baseline_rows = authoritative.loc[
                authoritative["held_out_subject"].eq(subject_id)
                & authoritative["primitive"].eq(primitive)
            ]
            if len(baseline_rows) != 1:
                raise ValueError(
                    f"frozen population baseline is missing for {subject_id!r}/{primitive}"
                )
            baseline = baseline_rows.iloc[0]
            population_scale = _validate_frozen_population_anchor(
                primitive_table=primitive_table,
                primitive=primitive,
                held_out_subject=subject_id,
                baseline=baseline,
            )
            if "population_center" not in baseline.index:
                raise ValueError("frozen population baseline is missing population_center")
            population_center = float(baseline["population_center"])
            if not np.isfinite(population_center):
                raise ValueError("frozen population center must be finite")
            outer_confidence = confidence.loc[
                confidence["primitive"].eq(primitive)
                & confidence["subject_id"].ne(subject_id)
            ]
            outer_estimate = robust_location_scale(
                outer_confidence["value"].to_numpy(dtype=float),
                outer_confidence["quality_confidence"].to_numpy(dtype=float),
            )
            if not np.isclose(
                population_center,
                float(outer_estimate["center"]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError("frozen population center anchor is forged or inconsistent")

            subject_confidence = confidence.loc[
                confidence["primitive"].eq(primitive)
                & confidence["subject_id"].eq(subject_id)
            ]
            calibration_confidence = subject_confidence.loc[
                pd.to_numeric(
                    subject_confidence["elapsed_day_index"], errors="coerce"
                ).le(14)
            ].sort_values("sensor_day_id", kind="stable")
            future_confidence = subject_confidence.loc[
                pd.to_numeric(
                    subject_confidence["elapsed_day_index"], errors="coerce"
                ).ge(22)
            ]
            oracle = robust_location_scale(
                future_confidence["value"].to_numpy(dtype=float),
                future_confidence["quality_confidence"].to_numpy(dtype=float),
            )
            oracle_center = float(oracle["center"])
            oracle_raw_scale = float(oracle["scale"])
            if float(oracle["n_eff"]) < 3.0 or not np.isfinite(oracle_raw_scale):
                oracle_scale = population_scale
                oracle_fallback = "population_insufficient_n_eff"
            else:
                oracle_scale = max(oracle_raw_scale, 0.1 * population_scale)
                oracle_fallback = (
                    "scale_floor"
                    if oracle_raw_scale < 0.1 * population_scale
                    else "none"
                )
            original_calibration = robust_location_scale(
                calibration_confidence["value"].to_numpy(dtype=float),
                calibration_confidence["quality_confidence"].to_numpy(dtype=float),
            )
            original_estimate = shrink_location_scale(
                personal_center=float(original_calibration["center"]),
                personal_scale=float(original_calibration["scale"]),
                population_center=population_center,
                population_scale=population_scale,
                n_eff=float(original_calibration["n_eff"]),
                lambda_days=7.0,
            )
            original_fallback = _shrinkage_fallback(
                original_calibration, original_estimate, population_scale
            )
            original_transfer = _transfer_error(
                original_estimate,
                oracle_center=oracle_center,
                oracle_scale=oracle_scale,
                population_scale=population_scale,
            )

            for draw in range(draw_count):
                draw_rows = group_all.loc[
                    pd.to_numeric(group_all["draw"], errors="coerce").eq(draw)
                ].sort_values("sensor_day_id", kind="stable")
                masked_values = pd.to_numeric(
                    draw_rows["masked_value"], errors="coerce"
                ).to_numpy(dtype=float)
                masked_weights = pd.to_numeric(
                    draw_rows["masked_quality_confidence"], errors="coerce"
                ).to_numpy(dtype=float)
                invalid_weight = np.isfinite(masked_weights) & (
                    (masked_weights < 0) | (masked_weights > 1)
                )
                if bool(invalid_weight.any()):
                    raise ValueError("masked calibration confidence must be within [0,1]")
                if bool((~np.isfinite(masked_values) & (masked_weights > 0)).any()):
                    raise ValueError("missing masked calibration values must have zero confidence")
                masked_calibration = robust_location_scale(
                    masked_values, masked_weights
                )
                masked_estimate = shrink_location_scale(
                    personal_center=float(masked_calibration["center"]),
                    personal_scale=float(masked_calibration["scale"]),
                    population_center=population_center,
                    population_scale=population_scale,
                    n_eff=float(masked_calibration["n_eff"]),
                    lambda_days=7.0,
                )
                masked_fallback = _shrinkage_fallback(
                    masked_calibration, masked_estimate, population_scale
                )
                masked_transfer = _transfer_error(
                    masked_estimate,
                    oracle_center=oracle_center,
                    oracle_scale=oracle_scale,
                    population_scale=population_scale,
                )
                records.append(
                    {
                        "subject_id": subject_id,
                        "primitive": primitive,
                        "scenario": "contiguous_20pct_calibration_only",
                        "draw": draw,
                        "calibration_day_limit": 14,
                        "max_masked_elapsed_day": int(
                            pd.to_numeric(
                                draw_rows["elapsed_day_index"], errors="coerce"
                            ).max()
                        ),
                        "future_observations_touched": False,
                        "frozen_population_center": population_center,
                        "frozen_population_scale": population_scale,
                        "oracle_center": oracle_center,
                        "oracle_scale": oracle_scale,
                        "oracle_n_eff": float(oracle["n_eff"]),
                        "oracle_fallback": oracle_fallback,
                        "original_center": float(original_estimate["center"]),
                        "masked_center": float(masked_estimate["center"]),
                        "center_change": float(masked_estimate["center"])
                        - float(original_estimate["center"]),
                        "original_scale": float(original_estimate["scale"]),
                        "masked_scale": float(masked_estimate["scale"]),
                        "scale_change": float(masked_estimate["scale"])
                        - float(original_estimate["scale"]),
                        "original_n_eff": float(original_calibration["n_eff"]),
                        "masked_n_eff": float(masked_calibration["n_eff"]),
                        "n_eff_change": float(masked_calibration["n_eff"])
                        - float(original_calibration["n_eff"]),
                        "original_fallback": original_fallback,
                        "masked_fallback": masked_fallback,
                        "fallback_changed": original_fallback != masked_fallback,
                        "original_transfer_error": original_transfer,
                        "masked_transfer_error": masked_transfer,
                        "transfer_error_change": masked_transfer
                        - original_transfer,
                        "status": (
                            "ok"
                            if np.isfinite(masked_transfer)
                            else "insufficient"
                        ),
                    }
                )
    rows = pd.DataFrame.from_records(records)
    participant_records: list[dict[str, Any]] = []
    for (primitive, subject_id), group in rows.groupby(
        ["primitive", "subject_id"], sort=False, dropna=False
    ):
        transfer = pd.to_numeric(
            group["masked_transfer_error"], errors="coerce"
        ).to_numpy(dtype=float)
        participant_records.append(
            {
                "primitive": primitive,
                "subject_id": subject_id,
                "draw_denominator": draw_count,
                "finite_draws": int(np.isfinite(transfer).sum()),
                "original_transfer_error": float(
                    group["original_transfer_error"].iloc[0]
                ),
                "masked_transfer_error": (
                    float(transfer.mean())
                    if np.isfinite(transfer).all()
                    else float("nan")
                ),
                "n_eff_change": float(group["n_eff_change"].mean()),
                "fallback_change_rate": float(group["fallback_changed"].mean()),
                "status": "ok" if np.isfinite(transfer).all() else "incomplete",
            }
        )
    participant = pd.DataFrame.from_records(participant_records)
    summary_records: list[dict[str, Any]] = []
    for primitive, group in participant.groupby("primitive", sort=False):
        masked = pd.to_numeric(
            group["masked_transfer_error"], errors="coerce"
        ).to_numpy(dtype=float)
        complete = bool(np.isfinite(masked).all())
        original_error = float(group["original_transfer_error"].mean())
        masked_error = float(masked.mean()) if complete else float("nan")
        summary_records.append(
            {
                "primitive": primitive,
                "participant_denominator": len(subjects),
                "finite_participants": int(np.isfinite(masked).sum()),
                "original_transfer_error": original_error,
                "masked_transfer_error": masked_error,
                "transfer_error_change": masked_error - original_error,
                "n_eff_change": float(group["n_eff_change"].mean()),
                "fallback_change_rate": float(
                    group["fallback_change_rate"].mean()
                ),
                "status": "ok" if complete else "incomplete",
            }
        )
    return rows, pd.DataFrame.from_records(summary_records)


_TWO_SOURCE_SEMANTICS: Mapping[str, tuple[str, str]] = {
    "digital_engagement_score": ("screen_load_24h", "usage_load_24h"),
    "digital_disengagement_time": (
        "screen_disengagement_p90",
        "usage_disengagement_p90",
    ),
    "physical_activity_score": ("phone_activity_load_24h", "step_load_24h"),
    "physical_disengagement_time": (
        "phone_activity_disengagement_p90",
        "step_disengagement_p90",
    ),
    "light_exposure_score": (
        "mobile_light_exposure_24h",
        "wearable_light_exposure_24h",
    ),
    "low_light_onset": ("mobile_low_light_onset", "wearable_low_light_onset"),
    "physiological_settling_score": (
        "heart_rate_settling_delta",
        "heart_rate_settling_slope",
    ),
}
_FROZEN_SEMANTIC_OUTPUTS = (
    *_TWO_SOURCE_SEMANTICS,
    "winddown_center",
    "winddown_spread",
    "routine_deviation",
)
_SEMANTIC_OUTPUT_SUFFIXES = ("", "__confidence", "__modalities", "__status")


def _strip_frozen_semantic_outputs(representations: pd.DataFrame) -> pd.DataFrame:
    """Keep source/provenance columns and remove only recomposable semantics."""
    semantic_columns = {
        f"{semantic}{suffix}"
        for semantic in _FROZEN_SEMANTIC_OUTPUTS
        for suffix in _SEMANTIC_OUTPUT_SUFFIXES
    }
    return representations.drop(
        columns=[
            column for column in representations.columns if column in semantic_columns
        ]
    )


def evaluate_modality_sensitivity(
    representations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove each complete source modality from every two-source semantic."""
    _required_columns(
        representations, ("subject_id", "sensor_day_id"), "representations"
    )
    working = representations.copy(deep=True)
    if "method" in working:
        working = working.loc[working["method"].eq("shrinkage")]
    if "k" in working:
        working = working.loc[pd.to_numeric(working["k"], errors="coerce").eq(14)]
    if working.empty:
        raise ValueError("modality sensitivity has no primary shrinkage k=14 rows")
    working = _strip_frozen_semantic_outputs(working)
    baseline = compose_semantic_columns(working)
    if baseline.columns.duplicated().any():
        raise RuntimeError("semantic recomposition produced duplicate columns")
    records: list[dict[str, Any]] = []
    key_columns = [
        column
        for column in (
            "subject_id",
            "held_out_subject",
            "sensor_day_id",
            "method",
            "k",
        )
        if column in baseline.columns
    ]
    for semantic, sources in _TWO_SOURCE_SEMANTICS.items():
        for removal, removed_sources in (
            (sources[0], (sources[0],)),
            (sources[1], (sources[1],)),
            ("all_sources", sources),
        ):
            mutated = working.copy(deep=True)
            for source in removed_sources:
                confidence_column = f"{source}__confidence"
                if confidence_column not in mutated:
                    raise ValueError(
                        f"representations is missing modality confidence: {confidence_column}"
                    )
                mutated[confidence_column] = 0.0
            recomposed = compose_semantic_columns(mutated)
            if recomposed.columns.duplicated().any():
                raise RuntimeError("semantic recomposition produced duplicate columns")
            original_value = pd.to_numeric(baseline[semantic], errors="coerce")
            masked_value = pd.to_numeric(recomposed[semantic], errors="coerce")
            original_confidence = pd.to_numeric(
                baseline[f"{semantic}__confidence"], errors="coerce"
            )
            masked_confidence = pd.to_numeric(
                recomposed[f"{semantic}__confidence"], errors="coerce"
            )
            original_finite = np.isfinite(
                original_value.to_numpy(dtype=float)
            )
            masked_finite = np.isfinite(masked_value.to_numpy(dtype=float))
            for position in range(len(baseline)):
                record = {
                    column: baseline.iloc[position][column]
                    for column in key_columns
                }
                retained = bool(original_finite[position] and masked_finite[position])
                record.update(
                    {
                        "semantic": semantic,
                        "removal": removal,
                        "removed_source_count": len(removed_sources),
                        "original_value": float(original_value.iloc[position]),
                        "masked_value": float(masked_value.iloc[position]),
                        "original_confidence": float(
                            original_confidence.iloc[position]
                        ),
                        "masked_confidence": float(
                            masked_confidence.iloc[position]
                        ),
                        "original_modalities": int(
                            baseline.iloc[position][f"{semantic}__modalities"]
                        ),
                        "masked_modalities": int(
                            recomposed.iloc[position][f"{semantic}__modalities"]
                        ),
                        "retained": retained,
                        "absolute_error": (
                            abs(
                                float(masked_value.iloc[position])
                                - float(original_value.iloc[position])
                            )
                            if retained
                            else float("nan")
                        ),
                        "status": recomposed.iloc[position][
                            f"{semantic}__status"
                        ],
                    }
                )
                records.append(record)
    rows = pd.DataFrame.from_records(records)

    participant_records: list[dict[str, Any]] = []
    for (semantic, removal, subject_id), group in rows.groupby(
        ["semantic", "removal", "subject_id"], sort=False, dropna=False
    ):
        original_finite = np.isfinite(
            pd.to_numeric(group["original_value"], errors="coerce").to_numpy(
                dtype=float
            )
        )
        retained = group["retained"].to_numpy(dtype=bool) & original_finite
        errors = pd.to_numeric(group["absolute_error"], errors="coerce").to_numpy(
            dtype=float
        )
        confidence = pd.to_numeric(
            group["masked_confidence"], errors="coerce"
        ).to_numpy(dtype=float)
        denominator = int(original_finite.sum())
        participant_records.append(
            {
                "semantic": semantic,
                "removal": removal,
                "subject_id": subject_id,
                "original_denominator": denominator,
                "retained_count": int(retained.sum()),
                "retention": (
                    float(retained.sum() / denominator) if denominator else 0.0
                ),
                "absolute_error": (
                    float(errors[retained & np.isfinite(errors)].mean())
                    if bool((retained & np.isfinite(errors)).any())
                    else float("nan")
                ),
                "confidence": (
                    float(confidence[original_finite].mean())
                    if denominator
                    else float("nan")
                ),
                "status": "ok" if denominator else "no_original_values",
            }
        )
    participant = pd.DataFrame.from_records(participant_records)
    summary_records: list[dict[str, Any]] = []
    for (semantic, removal), group in participant.groupby(
        ["semantic", "removal"], sort=False, dropna=False
    ):
        errors = pd.to_numeric(group["absolute_error"], errors="coerce").to_numpy(
            dtype=float
        )
        finite_errors = np.isfinite(errors)
        summary_records.append(
            {
                "semantic": semantic,
                "removal": removal,
                "participant_denominator": len(group),
                "retention": float(group["retention"].mean()),
                "absolute_error": (
                    float(errors[finite_errors].mean())
                    if bool(finite_errors.any())
                    else float("nan")
                ),
                "confidence": float(group["confidence"].mean()),
                "status": (
                    "ok" if bool(group["status"].eq("ok").all()) else "incomplete"
                ),
            }
        )
    return rows, pd.DataFrame.from_records(summary_records)


def _calendar_row_for_replay(row: pd.Series) -> pd.DataFrame:
    columns = (
        "subject_id",
        "lifelog_date",
        "sleep_date",
        "sensor_day_id",
        "collection_start_date",
        "elapsed_day_index",
        "any_source_record_observed",
        "any_measurement_support_observed",
        "any_sensor_observed",
    )
    missing = [column for column in columns if column not in row.index]
    if missing:
        raise ValueError(f"primitive_table is missing replay calendar columns: {missing}")
    return pd.DataFrame([{column: row[column] for column in columns}])


def _source_frame_slice(
    primitive: str,
    calendar_row: pd.DataFrame,
    sensor_frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    definition = _DEFINITIONS[primitive]
    if definition.sensor_file not in sensor_frames:
        raise ValueError(f"sensor_frames is missing {definition.sensor_file}")
    frame = sensor_frames[definition.sensor_file]
    _required_columns(frame, ("subject_id", "timestamp"), definition.sensor_file)
    day = _local_midnight(calendar_row.iloc[0]["lifelog_date"], "lifelog_date")
    window_start, window_end = _primitive_window(definition, day)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.dt.tz is not None:
        raise ValueError("canonical sensor timestamps must be timezone-naive")
    subject_id = calendar_row.iloc[0]["subject_id"]
    subject = frame["subject_id"].eq(subject_id)
    if definition.timestamp_semantics == "instant":
        in_scope = subject & timestamps.ge(window_start) & timestamps.lt(window_end)
    else:
        in_scope = subject & timestamps.ge(window_start) & timestamps.le(window_end)
    return frame.loc[in_scope].copy(), definition.sensor_file


@dataclass(frozen=True)
class _IndexedSensorFrame:
    sensor_file: str
    frame: pd.DataFrame
    timestamps: np.ndarray
    positions_by_subject: Mapping[object, np.ndarray]


def _index_sensor_frame(
    frame: pd.DataFrame,
    sensor_file: str,
) -> _IndexedSensorFrame:
    """Parse and subject-sort a canonical timestamp column exactly once."""
    _required_columns(frame, ("subject_id", "timestamp"), sensor_file)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.dt.tz is not None:
        raise ValueError("canonical sensor timestamps must be timezone-naive")
    timestamp_values = timestamps.to_numpy(dtype="datetime64[ns]")
    positions_by_subject: dict[object, np.ndarray] = {}
    for subject_id, raw_positions in frame.groupby(
        "subject_id", sort=False, dropna=False
    ).indices.items():
        positions = np.asarray(raw_positions, dtype=np.int64)
        positions = positions[~np.isnat(timestamp_values[positions])]
        order = np.argsort(timestamp_values[positions], kind="stable")
        positions_by_subject[subject_id] = positions[order]
    return _IndexedSensorFrame(
        sensor_file=sensor_file,
        frame=frame,
        timestamps=timestamp_values,
        positions_by_subject=positions_by_subject,
    )


def _source_frame_slice_indexed(
    primitive: str,
    calendar_row: pd.DataFrame,
    indexed: _IndexedSensorFrame,
) -> tuple[pd.DataFrame, str]:
    """Return the reference source slice without rescanning the full sensor."""
    definition = _DEFINITIONS[primitive]
    if indexed.sensor_file != definition.sensor_file:
        raise ValueError(
            f"{primitive} requires indexed sensor file {definition.sensor_file}"
        )
    day = _local_midnight(calendar_row.iloc[0]["lifelog_date"], "lifelog_date")
    window_start, window_end = _primitive_window(definition, day)
    subject_id = calendar_row.iloc[0]["subject_id"]
    positions = indexed.positions_by_subject.get(subject_id)
    if positions is None or not len(positions):
        return indexed.frame.iloc[0:0].copy(), indexed.sensor_file
    subject_timestamps = indexed.timestamps[positions]
    start = np.datetime64(window_start.to_datetime64())
    end = np.datetime64(window_end.to_datetime64())
    left = int(np.searchsorted(subject_timestamps, start, side="left"))
    right_side = "left" if definition.timestamp_semantics == "instant" else "right"
    right = int(np.searchsorted(subject_timestamps, end, side=right_side))
    return indexed.frame.iloc[positions[left:right]].copy(), indexed.sensor_file


def _audit_record_digest(audit: pd.DataFrame) -> str:
    fields = (
        "record_id",
        "source_timestamp",
        "record_interval_start",
        "record_interval_end",
        "deleted",
        "deleted_by_any_mask",
    )
    payload: list[list[str]] = []
    for row in audit.loc[:, fields].itertuples(index=False, name=None):
        payload.append(
            [
                value.isoformat()
                if isinstance(value, pd.Timestamp)
                else "<NA>"
                if pd.isna(value)
                else str(value)
                for value in row
            ]
        )
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _compact_confidence_after_replay(
    primitive: str,
    primitive_row: pd.Series,
    masked_value: float,
    quality: Mapping[str, Any],
) -> dict[str, float]:
    validity = float(
        np.isfinite(masked_value) and str(quality.get("status")) == "observed"
    )
    proxy = _USAGE_SELECTION_PROXIES.get(primitive)
    if proxy is None:
        coverage = float(quality.get("availability_coverage", float("nan")))
        gap_minutes = float(quality.get("longest_gap_minutes", float("nan")))
        definition = _DEFINITIONS[primitive]
    else:
        coverage = float(primitive_row[f"{proxy}__availability_coverage"])
        gap_minutes = float(primitive_row[f"{proxy}__longest_gap_minutes"])
        definition = _DEFINITIONS[proxy]
    coverage_component = (
        float(np.clip(coverage, 0.0, 1.0)) if np.isfinite(coverage) else 0.0
    )
    window_minutes = definition.expected_epochs * definition.cadence_minutes
    gap_component = (
        float(np.clip(1.0 - gap_minutes / window_minutes, 0.0, 1.0))
        if np.isfinite(gap_minutes)
        else 0.0
    )
    return {
        "value_only": validity,
        "coverage_only": coverage_component * validity,
        "quality": coverage_component * gap_component * validity,
    }


def _confidence_after_replay(
    primitive: str,
    primitive_row: pd.Series,
    masked_value: float,
    quality: Mapping[str, Any],
) -> dict[str, float]:
    replay_row = pd.DataFrame([primitive_row.to_dict()])
    replay_row.loc[:, primitive] = masked_value
    for name, value in quality.items():
        replay_row.loc[:, f"{primitive}__{name}"] = value
    audit = compute_primitive_confidence(
        replay_row,
        primitive_names=(primitive,),
        usage_policy="screen_proxy",
    ).iloc[0]
    return {
        "value_only": float(audit["value_only_confidence"]),
        "coverage_only": float(audit["coverage_only_confidence"]),
        "quality": float(audit["quality_confidence"]),
    }


def _internal_gap_offsets(
    primitive: str,
    primitive_row: pd.Series,
    sensor_frames: Mapping[str, pd.DataFrame],
) -> tuple[tuple[float, float], ...]:
    proxy = _USAGE_SELECTION_PROXIES.get(primitive, primitive)
    definition = _DEFINITIONS[proxy]
    calendar = _calendar_row_for_replay(primitive_row)
    frame, _ = _source_frame_slice(proxy, calendar, sensor_frames)
    if frame.empty:
        return ()
    day = _local_midnight(primitive_row["lifelog_date"], "lifelog_date")
    window_start, window_end = _primitive_window(definition, day)
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    valid = timestamps.loc[
        timestamps.notna()
        & timestamps.ge(window_start)
        & timestamps.lt(window_end)
    ].sort_values(kind="stable").drop_duplicates()
    if len(valid) < 2:
        return ()
    cadence = pd.Timedelta(minutes=definition.cadence_minutes)
    offsets: list[tuple[float, float]] = []
    values = valid.tolist()
    for previous, current in zip(values, values[1:]):
        gap_start = pd.Timestamp(previous) + cadence
        gap_end = pd.Timestamp(current)
        if gap_start < gap_end:
            offsets.append(
                (
                    (gap_start - window_start).total_seconds() / 60.0,
                    (gap_end - window_start).total_seconds() / 60.0,
                )
            )
    return tuple(offsets)


def _internal_gap_offsets_from_prepared(
    prepared: PreparedPrimitiveReplay,
) -> tuple[tuple[float, float], ...]:
    """Extract internal cadence gaps from an already indexed daily replay."""
    timestamps = prepared.source_timestamps
    timestamps = np.unique(timestamps[~np.isnat(timestamps)])
    if len(timestamps) < 2:
        return ()
    prepared_window_start, prepared_window_end = _primitive_window(
        prepared.definition, prepared.lifelog_date
    )
    window_start = np.datetime64(prepared_window_start.to_datetime64())
    window_end = np.datetime64(prepared_window_end.to_datetime64())
    timestamps = timestamps[
        (timestamps >= window_start) & (timestamps < window_end)
    ]
    if len(timestamps) < 2:
        return ()
    cadence = np.timedelta64(
        int(prepared.definition.cadence_minutes), "m"
    )
    minute = np.timedelta64(1, "m")
    offsets: list[tuple[float, float]] = []
    for previous, current in zip(timestamps[:-1], timestamps[1:]):
        gap_start = previous + cadence
        if gap_start < current:
            offsets.append(
                (
                    float((gap_start - window_start) / minute),
                    float((current - window_start) / minute),
                )
            )
    return tuple(offsets)


def _empirical_mask_for_cell(
    *,
    primitive: str,
    target_row: pd.Series,
    draw: int,
    root_seed: int | str,
    donor_rows: Sequence[pd.Series],
    gap_offsets: Mapping[tuple[str, int], tuple[tuple[float, float], ...]],
) -> tuple[pd.DataFrame | None, int | None, str]:
    target_day_id = int(target_row["sensor_day_id"])
    eligible = [
        row
        for row in donor_rows
        if int(row["sensor_day_id"]) != target_day_id
        and gap_offsets.get((primitive, int(row["sensor_day_id"])), ())
    ]
    eligible.sort(key=lambda row: int(row["sensor_day_id"]))
    seed = derive_mask_seed(
        root_seed,
        "empirical_gap",
        draw,
        target_row["subject_id"],
        target_day_id,
        primitive,
    )
    if not eligible:
        return None, None, seed
    donor = eligible[int(seed[:16], 16) % len(eligible)]
    donor_day_id = int(donor["sensor_day_id"])
    definition = _DEFINITIONS[primitive]
    target_day = _local_midnight(target_row["lifelog_date"], "lifelog_date")
    target_window_start, _ = _primitive_window(definition, target_day)
    intervals = [
        (
            target_window_start + pd.Timedelta(minutes=start),
            target_window_start + pd.Timedelta(minutes=end),
        )
        for start, end in gap_offsets[(primitive, donor_day_id)]
    ]
    masks = generate_mask_intervals(
        primitive,
        subject_id=target_row["subject_id"],
        sensor_day_id=target_day_id,
        lifelog_date=target_day,
        scenario="empirical_gap",
        draw=draw,
        root_seed=root_seed,
        empirical_intervals=intervals,
    )
    return masks, donor_day_id, seed


def evaluate_g2_reliability(
    primitive_table: pd.DataFrame,
    sensor_frames: Mapping[str, pd.DataFrame],
    *,
    frozen_population_baselines: pd.DataFrame,
    upstream_pair_summary: pd.DataFrame,
    root_seed: int | str = 42,
    draws: int = DEFAULT_MASK_DRAWS,
    scenarios: Sequence[str] | None = None,
    max_days: int = MAX_SELECTED_DAYS,
    run_calibration: bool = True,
    run_modality: bool = True,
    representations: pd.DataFrame | None = None,
) -> G2ReliabilityResult:
    """Execute the full deterministic, participant-hierarchical G2 study."""
    started = time.perf_counter()
    if isinstance(draws, (bool, np.bool_)) or int(draws) != draws or draws <= 0:
        raise ValueError("draws must be a positive integer")
    draw_count = int(draws)
    requested_scenarios = (
        FROZEN_MASK_SCENARIOS if scenarios is None else tuple(scenarios)
    )
    if (
        not requested_scenarios
        or len(requested_scenarios) != len(set(requested_scenarios))
        or any(scenario not in FROZEN_MASK_SCENARIOS for scenario in requested_scenarios)
    ):
        raise ValueError("scenarios must be a nonempty unique frozen subset")
    requested_scenarios = tuple(
        scenario
        for scenario in FROZEN_MASK_SCENARIOS
        if scenario in requested_scenarios
    )
    if not isinstance(sensor_frames, Mapping):
        raise TypeError("sensor_frames must map canonical filenames to DataFrames")
    _required_columns(
        primitive_table,
        (
            "subject_id",
            "sensor_day_id",
            "lifelog_date",
            "sleep_date",
            "elapsed_day_index",
        ),
        "primitive_table",
    )
    primitive_before = primitive_table.copy(deep=True)
    selected_days, selected_day_audit = select_reliability_days(
        primitive_table, max_days=max_days
    )
    primitive_rows = primitive_table.copy(deep=True)
    primitive_rows["__subject_sort"] = primitive_rows["subject_id"].map(str)
    primitive_rows = primitive_rows.sort_values(
        ["__subject_sort", "sensor_day_id"], kind="stable"
    ).drop(columns="__subject_sort").reset_index(drop=True)
    row_by_day = {
        int(row["sensor_day_id"]): row
        for _, row in primitive_rows.iterrows()
    }
    if len(row_by_day) != len(primitive_rows):
        raise ValueError("primitive_table sensor_day_id must be globally unique")

    selected_cell_records: list[dict[str, Any]] = []
    selected_row_objects: dict[tuple[str, str], list[pd.Series]] = {}
    prepared_replay_cache: dict[tuple[str, int], PreparedPrimitiveReplay] = {}
    sensor_index_cache: dict[str, _IndexedSensorFrame] = {}

    def prepared_for(
        primitive: str,
        row: pd.Series,
    ) -> PreparedPrimitiveReplay:
        day_id = int(row["sensor_day_id"])
        cache_key = (primitive, day_id)
        cached = prepared_replay_cache.get(cache_key)
        if cached is not None:
            return cached
        definition = _DEFINITIONS[primitive]
        sensor_file = definition.sensor_file
        if sensor_file not in sensor_frames:
            raise ValueError(f"sensor_frames is missing {sensor_file}")
        indexed = sensor_index_cache.get(sensor_file)
        if indexed is None:
            indexed = _index_sensor_frame(sensor_frames[sensor_file], sensor_file)
            sensor_index_cache[sensor_file] = indexed
        calendar = _calendar_row_for_replay(row)
        sensor_slice, _ = _source_frame_slice_indexed(
            primitive, calendar, indexed
        )
        prepared = prepare_primitive_replay(
            primitive, sensor_slice, sensor_file, calendar
        )
        prepared_replay_cache[cache_key] = prepared
        return prepared

    for selected in selected_days.itertuples(index=False):
        day_id = int(selected.sensor_day_id)
        primitive = str(selected.primitive)
        row = row_by_day[day_id]
        if row["subject_id"] != selected.subject_id:
            raise ValueError("selected day participant identity is inconsistent")
        prepared = prepared_for(primitive, row)
        original_replay = recompute_prepared_primitive(prepared)
        replay_value = original_replay.value
        replay_quality = original_replay.quality
        original_value = float(pd.to_numeric(pd.Series([row[primitive]]), errors="coerce").iloc[0])
        if not (
            (np.isnan(replay_value) and np.isnan(original_value))
            or np.isclose(replay_value, original_value, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError(
                f"primitive_table original value does not match canonical replay for {day_id}/{primitive}"
            )
        original_status = str(row[f"{primitive}__status"])
        if str(replay_quality["status"]) != original_status:
            raise ValueError("primitive_table original status does not match canonical replay")
        selected_cell_records.append(
            {
                "subject_id": selected.subject_id,
                "sensor_day_id": day_id,
                "primitive": primitive,
                "original_value": original_value,
            }
        )
        selected_row_objects.setdefault(
            (str(selected.subject_id), primitive), []
        ).append(row)
    selected_cells = pd.DataFrame.from_records(selected_cell_records)
    standardizers = compute_reliability_standardizers(
        selected_cells,
        frozen_population_baselines=frozen_population_baselines,
        primitive_table=primitive_rows,
    )
    standardizer_lookup = {
        (str(row.subject_id), row.primitive): row
        for row in standardizers.itertuples(index=False)
    }

    original_confidence = compute_primitive_confidence(
        primitive_rows,
        primitive_names=CORE_G2_PRIMITIVES,
        usage_policy="screen_proxy",
    )
    confidence_lookup = {
        (str(row.subject_id), int(row.sensor_day_id), row.primitive): row
        for row in original_confidence.itertuples(index=False)
    }
    gap_offsets: dict[tuple[str, int], tuple[tuple[float, float], ...]] = {}
    if "empirical_gap" in requested_scenarios:
        for selected in selected_days.itertuples(index=False):
            day_id = int(selected.sensor_day_id)
            primitive = str(selected.primitive)
            proxy = _USAGE_SELECTION_PROXIES.get(primitive, primitive)
            gap_offsets[(primitive, day_id)] = _internal_gap_offsets_from_prepared(
                prepared_for(proxy, row_by_day[day_id])
            )

    geometry_cache: dict[
        tuple[str, int, tuple[tuple[int, int], ...]],
        tuple[float, dict[str, Any], int, int, int, str, str],
    ] = {}
    compact_verified_primitives: set[str] = set()
    mask_records: list[dict[str, Any]] = []
    compact_audits: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []
    unique_geometry_count = 0

    def replay_mask(
        primitive: str,
        row: pd.Series,
        scenario: str,
        draw: int,
    ) -> tuple[
        pd.DataFrame | None,
        int | None,
        str,
        float,
        dict[str, Any],
        dict[str, Any],
    ]:
        day_id = int(row["sensor_day_id"])
        if scenario == "empirical_gap":
            donors = selected_row_objects.get((str(row["subject_id"]), primitive), [])
            masks, donor_day_id, seed = _empirical_mask_for_cell(
                primitive=primitive,
                target_row=row,
                draw=draw,
                root_seed=root_seed,
                donor_rows=donors,
                gap_offsets=gap_offsets,
            )
            if masks is None:
                unavailable_quality = {"status": "insufficient"}
                compact = {
                    "original_count": 0,
                    "deleted_count": 0,
                    "retained_count": 0,
                    "record_digest": hashlib.sha256(b"empirical_gap_unavailable").hexdigest(),
                    "audit_status": "empirical_gap_unavailable",
                }
                return None, None, seed, float("nan"), unavailable_quality, compact
        else:
            donor_day_id = None
            masks = generate_mask_intervals(
                primitive,
                subject_id=row["subject_id"],
                sensor_day_id=day_id,
                lifelog_date=row["lifelog_date"],
                scenario=scenario,
                draw=draw,
                root_seed=root_seed,
            )
            seed = str(masks.iloc[0]["seed"])
        geometry = tuple(
            (
                int(pd.Timestamp(mask.mask_start).value),
                int(pd.Timestamp(mask.mask_end).value),
            )
            for mask in masks.itertuples(index=False)
        )
        cache_key = (primitive, day_id, geometry)
        if cache_key in geometry_cache:
            (
                masked_value,
                quality,
                original_count,
                deleted_count,
                retained_count,
                record_digest,
                audit_status,
            ) = geometry_cache[cache_key]
        else:
            prepared = prepared_replay_cache.get((primitive, day_id))
            if prepared is None:
                prepared = prepared_for(primitive, row)
            replay_result = recompute_prepared_primitive_batch(
                prepared, (masks,)
            )[0]
            masked_value = replay_result.value
            quality = replay_result.quality
            original_count = replay_result.original_count
            deleted_count = replay_result.deleted_count
            retained_count = replay_result.retained_count
            record_payload = [
                prepared.source_record_digest,
                prepared.definition.timestamp_semantics,
                [
                    [
                        pd.Timestamp(mask.mask_start).isoformat(),
                        pd.Timestamp(mask.mask_end).isoformat(),
                    ]
                    for mask in masks.itertuples(index=False)
                ],
            ]
            record_digest = hashlib.sha256(
                json.dumps(
                    record_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            audit_status = "ok" if original_count else "no_target_records"
            if primitive not in compact_verified_primitives:
                calendar = _calendar_row_for_replay(row)
                sensor_file = prepared.sensor_file
                indexed = sensor_index_cache[sensor_file]
                sensor_slice, _ = _source_frame_slice_indexed(
                    primitive, calendar, indexed
                )
                public_masked, public_audit = apply_mask_intervals(
                    primitive,
                    sensor_slice,
                    sensor_file,
                    calendar,
                    masks,
                )
                public_value, public_quality = recompute_primitive_from_frame(
                    primitive,
                    public_masked,
                    sensor_file,
                    calendar,
                )
                value_equal = (
                    np.isnan(masked_value) and np.isnan(public_value)
                ) or np.isclose(
                    masked_value, public_value, rtol=1e-12, atol=1e-12
                )
                quality_equal = (
                    str(quality["status"]) == str(public_quality["status"])
                    and int(quality["observed_epochs"])
                    == int(public_quality["observed_epochs"])
                    and (
                        (
                            np.isnan(float(quality["availability_coverage"]))
                            and np.isnan(
                                float(public_quality["availability_coverage"])
                            )
                        )
                        or np.isclose(
                            float(quality["availability_coverage"]),
                            float(public_quality["availability_coverage"]),
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                    and (
                        (
                            np.isnan(float(quality["longest_gap_minutes"]))
                            and np.isnan(
                                float(public_quality["longest_gap_minutes"])
                            )
                        )
                        or np.isclose(
                            float(quality["longest_gap_minutes"]),
                            float(public_quality["longest_gap_minutes"]),
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                )
                count_equal = bool(
                    original_count == int(public_audit["original_count"].max())
                    and deleted_count == int(public_audit["deleted_count"].max())
                    and retained_count == int(public_audit["retained_count"].max())
                )
                compact_confidence = _compact_confidence_after_replay(
                    primitive, row, masked_value, quality
                )
                public_confidence = _confidence_after_replay(
                    primitive, row, public_value, public_quality
                )
                confidence_equal = all(
                    np.isclose(
                        compact_confidence[name],
                        public_confidence[name],
                        rtol=0.0,
                        atol=1e-12,
                    )
                    for name in compact_confidence
                )
                if not (
                    value_equal
                    and quality_equal
                    and count_equal
                    and confidence_equal
                ):
                    raise RuntimeError(
                        f"compact replay parity failed against public C1 for {primitive}"
                    )
                compact_verified_primitives.add(primitive)
            geometry_cache[cache_key] = (
                float(masked_value),
                dict(quality),
                original_count,
                deleted_count,
                retained_count,
                record_digest,
                audit_status,
            )
        compact = {
            "original_count": original_count,
            "deleted_count": deleted_count,
            "retained_count": retained_count,
            "record_digest": record_digest,
            "audit_status": audit_status,
        }
        return masks, donor_day_id, seed, float(masked_value), dict(quality), compact

    scenario_order = {
        scenario: index for index, scenario in enumerate(FROZEN_MASK_SCENARIOS)
    }
    for selected in selected_days.itertuples(index=False):
        day_id = int(selected.sensor_day_id)
        primitive = str(selected.primitive)
        row = row_by_day[day_id]
        original_value = float(row[primitive])
        original = confidence_lookup[(str(row["subject_id"]), day_id, primitive)]
        standardizer = standardizer_lookup[(str(row["subject_id"]), primitive)]
        for scenario in requested_scenarios:
            for draw in range(draw_count):
                (
                    masks,
                    donor_day_id,
                    seed,
                    masked_value,
                    quality,
                    compact,
                ) = replay_mask(primitive, row, scenario, draw)
                if masks is None:
                    masked_confidence = {
                        "value_only": 0.0,
                        "coverage_only": 0.0,
                        "quality": 0.0,
                    }
                    masked_status = "insufficient"
                else:
                    masked_confidence = _compact_confidence_after_replay(
                        primitive, row, masked_value, quality
                    )
                    masked_status = str(quality["status"])
                retained = bool(np.isfinite(masked_value) and masked_status == "observed")
                absolute_error = (
                    abs(masked_value - original_value) if retained else float("nan")
                )
                standardized_error = (
                    absolute_error / float(standardizer.standardizer)
                    if retained
                    else float("nan")
                )
                audit_key_payload = json.dumps(
                    [seed, compact["record_digest"]],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                deletion_audit_key = hashlib.sha256(
                    audit_key_payload.encode("utf-8")
                ).hexdigest()
                compact_audits.append(
                    {
                        "deletion_audit_key": deletion_audit_key,
                        "subject_id": row["subject_id"],
                        "sensor_day_id": day_id,
                        "primitive": primitive,
                        "scenario": scenario,
                        "draw": draw,
                        "seed": seed,
                        "donor_sensor_day_id": donor_day_id,
                        "original_count": compact["original_count"],
                        "deleted_count": compact["deleted_count"],
                        "retained_count": compact["retained_count"],
                        "record_deletion_digest": compact["record_digest"],
                        "audit_representation": "normalized_source_digest_plus_mask_geometry",
                        "status": compact["audit_status"],
                    }
                )
                if masks is not None:
                    augmented = masks.copy()
                    augmented["donor_sensor_day_id"] = donor_day_id
                    augmented["deletion_audit_key"] = deletion_audit_key
                    mask_records.extend(augmented.to_dict(orient="records"))
                cell_records.append(
                    {
                        "subject_id": row["subject_id"],
                        "sensor_day_id": day_id,
                        "lifelog_date": row["lifelog_date"],
                        "elapsed_day_index": row["elapsed_day_index"],
                        "primitive": primitive,
                        "scenario": scenario,
                        "draw": draw,
                        "root_seed": str(root_seed),
                        "seed": seed,
                        "donor_sensor_day_id": donor_day_id,
                        "deletion_audit_key": deletion_audit_key,
                        "original_value": original_value,
                        "masked_value": masked_value,
                        "original_status": str(row[f"{primitive}__status"]),
                        "masked_status": masked_status,
                        "retained": retained,
                        "absolute_error": absolute_error,
                        "standardized_absolute_error": standardized_error,
                        "standardizer": float(standardizer.standardizer),
                        "standardizer_source": standardizer.scale_source,
                        "selected_day_mad": float(standardizer.selected_day_mad),
                        "frozen_population_scale": float(
                            standardizer.frozen_population_scale
                        ),
                        "scale_floor": float(standardizer.scale_floor),
                        "floor_applied": bool(standardizer.floor_applied),
                        "original_value_only_confidence": float(
                            original.value_only_confidence
                        ),
                        "original_coverage_only_confidence": float(
                            original.coverage_only_confidence
                        ),
                        "original_quality_confidence": float(
                            original.quality_confidence
                        ),
                        "masked_value_only_confidence": masked_confidence[
                            "value_only"
                        ],
                        "masked_coverage_only_confidence": masked_confidence[
                            "coverage_only"
                        ],
                        "masked_quality_confidence": masked_confidence["quality"],
                        "success": bool(retained and standardized_error <= 0.20),
                        "usage_record_deletion_stability": primitive.startswith(
                            "usage_"
                        ),
                        "reason": (
                            "retained"
                            if retained
                            else "empirical_gap_unavailable"
                            if masks is None
                            else f"masked_{masked_status}"
                        ),
                        "status": "observed" if retained else "abstained",
                    }
                )
        unique_geometry_count += len(geometry_cache)
        geometry_cache.clear()
    cell_replays = pd.DataFrame.from_records(cell_records)
    cell_replays["__subject_sort"] = cell_replays["subject_id"].map(str)
    cell_replays["__primitive_order"] = cell_replays["primitive"].map(
        {name: index for index, name in enumerate(CORE_G2_PRIMITIVES)}
    )
    cell_replays["__scenario_order"] = cell_replays["scenario"].map(scenario_order)
    cell_replays = cell_replays.sort_values(
        [
            "__primitive_order",
            "__scenario_order",
            "__subject_sort",
            "draw",
            "sensor_day_id",
        ],
        kind="stable",
    ).drop(
        columns=["__subject_sort", "__primitive_order", "__scenario_order"]
    ).reset_index(drop=True)
    mask_intervals = (
        pd.DataFrame.from_records(mask_records)
        if mask_records
        else pd.DataFrame(columns=(*MASK_TABLE_COLUMNS, "donor_sensor_day_id", "deletion_audit_key"))
    )
    if not mask_intervals.empty:
        mask_intervals["__subject_sort"] = mask_intervals["subject_id"].map(str)
        mask_intervals["__primitive_order"] = mask_intervals["primitive"].map(
            {name: index for index, name in enumerate(CORE_G2_PRIMITIVES)}
        )
        mask_intervals["__scenario_order"] = mask_intervals["scenario"].map(
            scenario_order
        )
        mask_intervals = mask_intervals.sort_values(
            [
                "__primitive_order",
                "__scenario_order",
                "__subject_sort",
                "draw",
                "sensor_day_id",
                "mask_interval_id",
            ],
            kind="stable",
        ).drop(
            columns=["__subject_sort", "__primitive_order", "__scenario_order"]
        ).reset_index(drop=True)
    deletion_audit = pd.DataFrame.from_records(compact_audits)
    deletion_audit["__subject_sort"] = deletion_audit["subject_id"].map(str)
    deletion_audit["__primitive_order"] = deletion_audit["primitive"].map(
        {name: index for index, name in enumerate(CORE_G2_PRIMITIVES)}
    )
    deletion_audit["__scenario_order"] = deletion_audit["scenario"].map(
        scenario_order
    )
    deletion_audit = deletion_audit.sort_values(
        [
            "__primitive_order",
            "__scenario_order",
            "__subject_sort",
            "draw",
            "sensor_day_id",
        ],
        kind="stable",
    ).drop(
        columns=["__subject_sort", "__primitive_order", "__scenario_order"]
    ).reset_index(drop=True)

    participants = tuple(sorted(primitive_rows["subject_id"].unique(), key=str))
    (
        draw_day_metrics,
        participant_draw_metrics,
        participant_metrics,
        participant_macro_metrics,
    ) = aggregate_hierarchical_reliability(
        cell_replays,
        selected_days=selected_days,
        expected_scenarios=requested_scenarios,
        participant_roster=participants,
        expected_draws=draw_count,
    )
    production_default_grid = bool(
        draw_count == DEFAULT_MASK_DRAWS
        and requested_scenarios == FROZEN_MASK_SCENARIOS
        and int(max_days) == MAX_SELECTED_DAYS
    )
    gate_input = participant_macro_metrics.copy()
    if not production_default_grid:
        gate_input["complete"] = False
        gate_input["status"] = "nonproduction_grid"
    primitive_gates, pair_gates, domain_gates, decision_table = (
        evaluate_reliability_gates(gate_input, upstream_pair_summary)
    )
    for table in (primitive_gates, pair_gates, domain_gates):
        table["production_eligible"] = production_default_grid
    decision_table["production_eligible"] = production_default_grid
    decision_table["root_seed"] = str(root_seed)
    decision_table["draws"] = draw_count

    if "contiguous_20pct" in requested_scenarios:
        confidence_ablation, confidence_summary = evaluate_confidence_ablation(
            cell_replays,
            selected_days=selected_days,
            participant_roster=participants,
            expected_draws=draw_count,
        )
    else:
        confidence_ablation = pd.DataFrame(
            columns=(
                "subject_id",
                "sensor_day_id",
                "primitive",
                "scenario",
                "draw",
                "success",
                "variant",
                "score",
                "aggregation_level",
                "roc_auc",
                "spearman",
                "status",
            )
        )
        confidence_summary = pd.DataFrame(
            columns=(
                "primitive",
                "scenario",
                "variant",
                "participant_denominator",
                "finite_auc_participants",
                "finite_spearman_participants",
                "roc_auc",
                "spearman",
                "status",
                "auc_delta_vs_coverage",
                "spearman_delta_vs_coverage",
                "beats_coverage_both_metrics",
            )
        )
    confidence_domain_pass: dict[str, bool] = {}
    for domain_row in domain_gates.itertuples(index=False):
        domain_pairs = pair_gates.loc[
            pair_gates["domain"].eq(domain_row.domain) & pair_gates["pair_pass"]
        ]
        primitives_for_domain = sorted(
            set(domain_pairs["left"]).union(set(domain_pairs["right"])),
            key=lambda name: CORE_G2_PRIMITIVES.index(name),
        )
        full_rows = confidence_summary.loc[
            confidence_summary["primitive"].isin(primitives_for_domain)
            & confidence_summary["variant"].eq("full_quality_aware")
        ]
        condition = bool(
            domain_row.domain_pass
            and len(full_rows) == len(primitives_for_domain)
            and bool(full_rows["beats_coverage_both_metrics"].all())
        )
        confidence_domain_pass[domain_row.domain] = condition
        decision_table = pd.concat(
            [
                decision_table,
                pd.DataFrame(
                    [
                        {
                            "level": "confidence_domain",
                            "primitive": pd.NA,
                            "pair": pd.NA,
                            "domain": domain_row.domain,
                            "gate": "full_quality_beats_coverage_auc_and_spearman",
                            "observed": int(
                                full_rows["beats_coverage_both_metrics"].sum()
                            ),
                            "comparator": "==",
                            "threshold": len(primitives_for_domain),
                            "passed": condition,
                            "can_change_domain_decision": False,
                            "status": "pass" if condition else "fail",
                            "production_eligible": production_default_grid,
                            "root_seed": str(root_seed),
                            "draws": draw_count,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    domain_gates["confidence_full_beats_coverage"] = domain_gates["domain"].map(
        confidence_domain_pass
    )

    if run_calibration:
        calibration_records: list[dict[str, Any]] = []
        calibration_rows = primitive_rows.loc[
            pd.to_numeric(
                primitive_rows["elapsed_day_index"], errors="coerce"
            ).le(14)
        ]
        for _, row in calibration_rows.iterrows():
            for primitive in CORE_G2_PRIMITIVES:
                original_value = float(
                    pd.to_numeric(pd.Series([row[primitive]]), errors="coerce").iloc[0]
                )
                for draw in range(draw_count):
                    masks, _, seed, masked_value, quality, _ = replay_mask(
                        primitive, row, "contiguous_20pct", draw
                    )
                    if masks is None:
                        raise RuntimeError("calibration contiguous mask cannot be unavailable")
                    masked_confidence = _compact_confidence_after_replay(
                        primitive, row, masked_value, quality
                    )
                    calibration_records.append(
                        {
                            "subject_id": row["subject_id"],
                            "sensor_day_id": int(row["sensor_day_id"]),
                            "elapsed_day_index": int(row["elapsed_day_index"]),
                            "primitive": primitive,
                            "scenario": "contiguous_20pct_calibration_only",
                            "draw": draw,
                            "seed": seed,
                            "original_value": original_value,
                            "masked_value": masked_value,
                            "masked_quality_confidence": masked_confidence["quality"],
                        }
                    )
        calibration_sensitivity, calibration_summary = (
            evaluate_calibration_sensitivity(
                primitive_rows,
                pd.DataFrame.from_records(calibration_records),
                frozen_population_baselines=frozen_population_baselines,
                draws=draw_count,
            )
        )
    else:
        calibration_sensitivity = pd.DataFrame()
        calibration_summary = pd.DataFrame()

    if run_modality:
        if representations is None:
            representations = build_loso_representations(
                primitive_rows, k_values=(14,)
            ).representations
        modality_sensitivity, modality_summary = evaluate_modality_sensitivity(
            representations
        )
    else:
        modality_sensitivity = pd.DataFrame()
        modality_summary = pd.DataFrame()

    runtime_seconds = time.perf_counter() - started
    run_manifest = pd.DataFrame(
        [
            {
                "stage": "G2_hierarchical_masking_reliability",
                "root_seed": str(root_seed),
                "draws": draw_count,
                "scenario_grid": json.dumps(list(requested_scenarios)),
                "max_selected_days": int(max_days),
                "production_default_grid": production_default_grid,
                "participant_denominator": len(participants),
                "selected_cell_count": len(selected_cells),
                "logical_replay_count": len(cell_replays),
                "unique_geometry_count": unique_geometry_count,
                "mask_interval_rows": len(mask_intervals),
                "deletion_audit_rows": len(deletion_audit),
                "deletion_audit_policy": "normalized_source_digest_plus_mask_geometry",
                "empirical_donor_policy": "within_participant_leave_source_day_out_sha256",
                "standardizer_policy": "selected_unmasked_1.4826MAD_then_verified_Task3_k0_outer_anchor_floor_0.1",
                "hierarchy": "draw_day_to_participant_draw_to_participant_to_equal_person_macro",
                "confidence_policy": "masked_scores_primary20_full_must_beat_coverage_in_auc_and_spearman",
                "calibration_policy": "held_out_elapsed_le14_outer_prior_and_future_oracle_frozen",
                "runtime_seconds": runtime_seconds,
                "status": "complete" if production_default_grid else "nonproduction_grid",
            }
        ]
    )
    pd.testing.assert_frame_equal(
        primitive_table,
        primitive_before,
        check_dtype=True,
        check_like=False,
    )
    return G2ReliabilityResult(
        selected_days=selected_days,
        selected_day_audit=selected_day_audit,
        mask_intervals=mask_intervals,
        deletion_audit=deletion_audit,
        cell_replays=cell_replays,
        draw_day_metrics=draw_day_metrics,
        participant_draw_metrics=participant_draw_metrics,
        participant_metrics=participant_metrics,
        participant_macro_metrics=participant_macro_metrics,
        primitive_gates=primitive_gates,
        pair_gates=pair_gates,
        domain_gates=domain_gates,
        decision_table=decision_table,
        confidence_ablation=confidence_ablation,
        confidence_summary=confidence_summary,
        calibration_sensitivity=calibration_sensitivity,
        calibration_summary=calibration_summary,
        modality_sensitivity=modality_sensitivity,
        modality_summary=modality_summary,
        run_manifest=run_manifest,
    )
