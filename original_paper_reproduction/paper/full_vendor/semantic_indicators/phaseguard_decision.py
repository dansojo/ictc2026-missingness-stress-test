"""Decision-grade bridge from reviewed prepared records to PhaseGuard rows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd

from .contracts import PRIMITIVE_DEFINITIONS
from .phaseguard import (
    PhaseBinSufficientStatistic,
    PhaseSufficientStatistics,
    build_phase_sufficient_statistics,
)
from . import phaseguard_authority as _authority
from .primitives import (
    PreparedPhaseView,
    export_prepared_phase_view,
    prepare_primitive_replay,
    recompute_prepared_primitive,
    validate_prepared_phase_view,
    weighted_quantile,
)
from .reliability import CORE_G2_PRIMITIVES


_RATE_OPERATIONS = {"binary_load", "activity_load", "exposure_mean"}
_DEFINITIONS = {definition.name: definition for definition in PRIMITIVE_DEFINITIONS}
_CALENDAR_COLUMNS = (
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


def _sha256_payload(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be an exact lowercase SHA-256")
    return value


def _float_token(value: float) -> str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        raise ValueError("infinite values cannot be audited")
    return number.hex()


def _validated_view(view: PreparedPhaseView) -> PreparedPhaseView:
    if type(view) is not PreparedPhaseView:
        raise TypeError("view must be an exact PreparedPhaseView")
    validate_prepared_phase_view(view)
    return view


def _bin_components(
    view: PreparedPhaseView, phase_bin: int
) -> tuple[float, float, float, tuple[float, ...], tuple[float, ...]]:
    records = tuple(
        record
        for record in view.records
        if record.phase_bin == phase_bin and record.valid
    )
    observed_units = float(math.fsum(record.support_units for record in records))
    numerator = float(
        math.fsum(record.transformed_numerator for record in records)
    )
    events = sorted(
        (
            (float(record.event_minute), float(record.event_mass), record.record_id)
            for record in records
            if record.positive_event_count == 1
            and record.event_minute is not None
        ),
        key=lambda item: (item[0], item[2]),
    )
    event_minutes = tuple(item[0] for item in events)
    event_weights = tuple(item[1] for item in events)
    expected_units = (
        float(30 // view.cadence_minutes)
        if view.window == "day" or 36 <= phase_bin < 48
        else 0.0
    )
    return expected_units, observed_units, numerator, event_minutes, event_weights


def _bin_value(
    view: PreparedPhaseView,
    *,
    observed_units: float,
    numerator: float,
    event_minutes: tuple[float, ...],
    event_weights: tuple[float, ...],
) -> float:
    if view.operation == "additive_load":
        return numerator if observed_units > 0.0 else float("nan")
    if view.operation in _RATE_OPERATIONS:
        return (
            numerator / observed_units
            if observed_units > 0.0
            else float("nan")
        )
    if view.operation == "evening_p90":
        return (
            float(weighted_quantile(event_minutes, event_weights, 0.9))
            if event_minutes
            else float("nan")
        )
    raise ValueError(f"unsupported prepared PhaseGuard operation: {view.operation}")


def prepared_view_to_phase_rows(view: PreparedPhaseView) -> pd.DataFrame:
    """Map one reviewed phase view to the frozen 48-row source grid."""

    frozen = _validated_view(view)
    identity = frozen.calendar_identity
    rows: list[dict[str, object]] = []
    for phase_bin in range(48):
        expected, observed, numerator, minutes, weights = _bin_components(
            frozen, phase_bin
        )
        value = _bin_value(
            frozen,
            observed_units=observed,
            numerator=numerator,
            event_minutes=minutes,
            event_weights=weights,
        )
        rows.append(
            {
                "subject_id": identity.subject_id,
                "sensor_day_id": identity.sensor_day_id,
                "elapsed_day_index": identity.elapsed_day_index,
                "phase_bin": phase_bin,
                "day_type": identity.day_type,
                "primitive": frozen.primitive_name,
                "value": float(value),
                "support": float(observed / expected) if expected > 0.0 else 0.0,
                "event_count": len(minutes),
                "event_mass": float(math.fsum(weights)),
                "representative_minute": (
                    float(value)
                    if frozen.operation == "evening_p90" and math.isfinite(value)
                    else float("nan")
                ),
                "event_minutes": minutes,
                "event_weights": weights,
            }
        )
    return pd.DataFrame(rows)


def _daily_value(view: PreparedPhaseView, rows: pd.DataFrame) -> float:
    observed = float(
        math.fsum(
            record.support_units for record in view.records if record.valid
        )
    )
    numerator = float(
        math.fsum(
            record.transformed_numerator for record in view.records if record.valid
        )
    )
    if view.operation == "additive_load":
        return numerator if observed > 0.0 else float("nan")
    if view.operation in _RATE_OPERATIONS:
        return numerator / observed if observed > 0.0 else float("nan")
    timing = rows.loc[rows["event_count"].gt(0)]
    minutes = tuple(
        minute
        for values in timing["event_minutes"]
        for minute in values
    )
    weights = tuple(
        weight
        for values in timing["event_weights"]
        for weight in values
    )
    return (
        float(weighted_quantile(minutes, weights, 0.9))
        if minutes
        else float("nan")
    )


def prepared_view_to_no_mask_statistics(
    view: PreparedPhaseView,
) -> PhaseSufficientStatistics:
    """Build PhaseGuard statistics for the reviewed empty-mask baseline."""

    frozen = _validated_view(view)
    rows = prepared_view_to_phase_rows(frozen)
    active_bins = range(36, 48) if frozen.operation == "evening_p90" else range(48)
    bins: list[PhaseBinSufficientStatistic] = []
    for phase_bin in active_bins:
        expected, observed, numerator, minutes, weights = _bin_components(
            frozen, phase_bin
        )
        bins.append(
            PhaseBinSufficientStatistic(
                phase_bin=phase_bin,
                expected_units=expected,
                observed_units=observed,
                observed_numerator=numerator,
                masked_fraction=0.0,
                baseline_units=observed,
                event_minutes=minutes,
                event_weights=weights,
            )
        )
    mask_digest = _sha256_payload(
        {
            "schema": "phaseguard-no-mask-v1",
            "phase_view_digest": frozen.content_digest,
            "intervals": [],
        }
    )
    identity = frozen.calendar_identity
    return build_phase_sufficient_statistics(
        target_subject=identity.subject_id,
        target_elapsed_day=identity.elapsed_day_index,
        sensor_day_id=identity.sensor_day_id,
        day_type=identity.day_type,
        primitive_name=frozen.primitive_name,
        operation=frozen.operation,
        masked_value=_daily_value(frozen, rows),
        source_record_digest=frozen.source_record_digest,
        mask_digest=mask_digest,
        calendar_digest=identity.content_digest,
        bins=tuple(bins),
    )


@dataclass(frozen=True)
class PhaseSourceAudit:
    """Compact decision-grade output from one no-mask source audit."""

    phase_rows: pd.DataFrame
    key_ledger: pd.DataFrame
    calibration_input_digest: str
    key_count: int
    phase_row_count: int
    parity_failure_count: int
    ledger_digest: str
    content_digest: str


def _same_float(left: object, right: object) -> bool:
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return False
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isnan(left_float) and math.isnan(right_float)
    ) or left_float.hex() == right_float.hex()


def audit_phase_source_frames(
    *,
    primitive_table: pd.DataFrame,
    sensor_frames: dict[str, pd.DataFrame],
    calibration_input_digest: str,
) -> PhaseSourceAudit:
    """Audit every reviewed day × core primitive through one prepared replay."""

    authority_digest = _require_sha256(
        calibration_input_digest, "calibration_input_digest"
    )
    if type(primitive_table) is not pd.DataFrame or primitive_table.empty:
        raise TypeError("primitive_table must be a nonempty exact DataFrame")
    if primitive_table.columns.duplicated().any():
        raise ValueError("primitive_table columns must be unique")
    required = {
        "row_id",
        *_CALENDAR_COLUMNS,
        *CORE_G2_PRIMITIVES,
        *(f"{name}__status" for name in CORE_G2_PRIMITIVES),
    }
    missing = sorted(required.difference(primitive_table.columns))
    if missing:
        raise ValueError(f"primitive_table missing audited columns: {missing}")
    if type(sensor_frames) is not dict:
        raise TypeError("sensor_frames must be an exact dict")
    required_sensor_files = tuple(
        dict.fromkeys(_DEFINITIONS[name].sensor_file for name in CORE_G2_PRIMITIVES)
    )
    if set(sensor_frames) != set(required_sensor_files):
        raise ValueError("sensor_frames must contain exactly the six frozen sensor files")
    if any(type(frame) is not pd.DataFrame for frame in sensor_frames.values()):
        raise TypeError("sensor_frames values must be exact DataFrames")

    ordered = primitive_table.sort_values("row_id", kind="mergesort").reset_index(
        drop=True
    )
    if ordered["row_id"].duplicated().any():
        raise ValueError("primitive_table row_id values must be unique")
    if not ordered["row_id"].map(
        lambda value: not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer))
    ).all():
        raise ValueError("primitive_table row_id values must be exact integers")

    phase_blocks: list[pd.DataFrame] = []
    phase_buffer: list[pd.DataFrame] = []
    ledger: list[dict[str, object]] = []
    digest_rows: list[dict[str, object]] = []
    cached_subject: str | None = None
    subject_sensors: dict[str, pd.DataFrame] = {}

    for table_row in ordered.itertuples(index=False):
        subject = getattr(table_row, "subject_id")
        if type(subject) is not str or not subject:
            raise ValueError("primitive_table subject_id must be exact nonempty strings")
        if subject != cached_subject:
            subject_sensors = {
                name: frame.loc[frame["subject_id"].eq(subject)].copy()
                for name, frame in sensor_frames.items()
            }
            cached_subject = subject
        row_id = int(getattr(table_row, "row_id"))
        calendar_row = ordered.loc[
            ordered["row_id"].eq(row_id), list(_CALENDAR_COLUMNS)
        ].copy()
        if len(calendar_row) != 1:
            raise ValueError("primitive_table calendar key is not unique")

        for primitive in CORE_G2_PRIMITIVES:
            definition = _DEFINITIONS[primitive]
            prepared = prepare_primitive_replay(
                primitive,
                subject_sensors[definition.sensor_file],
                definition.sensor_file,
                calendar_row,
            )
            view = export_prepared_phase_view(prepared)
            reviewed = recompute_prepared_primitive(prepared)
            expected_value = getattr(table_row, primitive)
            expected_status = getattr(table_row, f"{primitive}__status")
            if not _same_float(reviewed.value, expected_value):
                raise ValueError(
                    "no-mask value parity failed for "
                    f"row_id={row_id} primitive={primitive}"
                )
            if type(expected_status) is not str or (
                type(reviewed.quality.get("status")) is not str
                or reviewed.quality["status"] != expected_status
            ):
                raise ValueError(
                    "no-mask status parity failed for "
                    f"row_id={row_id} primitive={primitive}"
                )
            source_rows = prepared_view_to_phase_rows(view)
            statistics = prepared_view_to_no_mask_statistics(view)
            phase_buffer.append(source_rows)
            if len(phase_buffer) == 64:
                phase_blocks.append(pd.concat(phase_buffer, ignore_index=True))
                phase_buffer.clear()
            key_index = len(ledger)
            ledger.append(
                {
                    "key_index": key_index,
                    "row_id": row_id,
                    "subject_id": subject,
                    "sensor_day_id": int(view.calendar_identity.sensor_day_id),
                    "elapsed_day_index": int(
                        view.calendar_identity.elapsed_day_index
                    ),
                    "primitive": primitive,
                    "operation": view.operation,
                    "expected_value": float(expected_value),
                    "actual_value": float(reviewed.value),
                    "expected_status": expected_status,
                    "actual_status": reviewed.quality["status"],
                    "preparation_count": 1,
                    "phase_view_digest": view.content_digest,
                    "statistics_digest": statistics.content_digest,
                }
            )
            digest_rows.append(
                {
                    "key_index": key_index,
                    "row_id": row_id,
                    "subject_id": subject,
                    "sensor_day_id": int(view.calendar_identity.sensor_day_id),
                    "elapsed_day_index": int(
                        view.calendar_identity.elapsed_day_index
                    ),
                    "primitive": primitive,
                    "expected_value": _float_token(float(expected_value)),
                    "actual_value": _float_token(float(reviewed.value)),
                    "status": expected_status,
                    "phase_view_digest": view.content_digest,
                    "statistics_digest": statistics.content_digest,
                }
            )
    if phase_buffer:
        phase_blocks.append(pd.concat(phase_buffer, ignore_index=True))
    phase_rows = pd.concat(phase_blocks, ignore_index=True)
    key_ledger = pd.DataFrame(ledger)
    key_count = len(key_ledger)
    phase_row_count = len(phase_rows)
    if phase_row_count != key_count * 48:
        raise ValueError("phase source row topology is incomplete")
    ledger_digest = _sha256_payload(
        {"schema": "lean-phase-source-ledger-v1", "keys": digest_rows}
    )
    content_digest = _sha256_payload(
        {
            "schema": "lean-phase-source-audit-v1",
            "calibration_input_digest": authority_digest,
            "key_count": key_count,
            "phase_row_count": phase_row_count,
            "parity_failure_count": 0,
            "ledger_digest": ledger_digest,
        }
    )
    return PhaseSourceAudit(
        phase_rows=phase_rows,
        key_ledger=key_ledger,
        calibration_input_digest=authority_digest,
        key_count=key_count,
        phase_row_count=phase_row_count,
        parity_failure_count=0,
        ledger_digest=ledger_digest,
        content_digest=content_digest,
    )


def audit_calibration_phase_source_stream(
    calibration_input: _authority.CalibrationInputAuthority,
) -> PhaseSourceAudit:
    """Load only sealed inputs and run the real one-key no-mask audit."""

    if type(calibration_input) is not _authority.CalibrationInputAuthority:
        raise TypeError(
            "calibration_input must be an exact CalibrationInputAuthority"
        )
    _authority._validate_calibration_input_authority_value(calibration_input)
    primitive_table = pd.read_parquet(calibration_input.primitive_table.path)
    sensor_frames = {
        sensor_file: pd.read_parquet(calibration_input.canonical_root / sensor_file)
        for sensor_file in dict.fromkeys(
            _DEFINITIONS[name].sensor_file for name in CORE_G2_PRIMITIVES
        )
    }
    result = audit_phase_source_frames(
        primitive_table=primitive_table,
        sensor_frames=sensor_frames,
        calibration_input_digest=calibration_input.content_digest,
    )
    _authority._validate_calibration_input_authority_value(calibration_input)
    return result
