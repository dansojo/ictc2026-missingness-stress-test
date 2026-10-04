"""Outcome-free record-identity masks for the Task 12 stress experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
import pandas as pd

from .primitives import PreparedPhaseRecord, PreparedPhaseView, validate_prepared_phase_view
from .stress_contracts import MatchedDeletionTarget


SCATTERED_SCENARIO = "scattered_random_20pct"
BOUNDARY_SCENARIO = "event_boundary_20pct"


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class MatchedRecordMask:
    scenario: str
    seed_sha256: str
    target_deleted_count: int
    record_identity_digests: tuple[str, ...]
    mask_intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    status: str
    reason: str
    content_digest: str = field(init=False)

    @property
    def selected_record_count(self) -> int:
        return len(self.record_identity_digests)

    def __post_init__(self) -> None:
        if self.scenario not in (SCATTERED_SCENARIO, BOUNDARY_SCENARIO):
            raise ValueError("record-mask scenario is not frozen")
        if type(self.seed_sha256) is not str or len(self.seed_sha256) != 64:
            raise TypeError("record-mask seed must be a SHA-256 string")
        if type(self.target_deleted_count) is not int or self.target_deleted_count < 0:
            raise TypeError("target_deleted_count must be an exact nonnegative integer")
        if type(self.record_identity_digests) is not tuple or type(self.mask_intervals) is not tuple:
            raise TypeError("record-mask identities and intervals must be exact tuples")
        if self.status not in ("eligible", "ineligible", "not_applicable"):
            raise ValueError("record-mask status is invalid")
        payload = [
            "matched-record-mask-v1",
            self.scenario,
            self.seed_sha256,
            self.target_deleted_count,
            list(self.record_identity_digests),
            [[start.isoformat(), end.isoformat()] for start, end in self.mask_intervals],
            self.status,
            self.reason,
        ]
        object.__setattr__(self, "content_digest", _sha(payload))


def _seed(view: PreparedPhaseView, target: MatchedDeletionTarget, scenario: str) -> str:
    payload = [
        "structured-missingness-mask-v1",
        42,
        scenario,
        view.calendar_identity.subject_id,
        view.calendar_identity.sensor_day_id,
        view.primitive_name,
        target.draw,
        view.source_record_digest,
        target.contiguous20_deletion_audit_key,
        target.target_deleted_count,
    ]
    return _sha(payload)


def _validate_inputs(view: PreparedPhaseView, target: MatchedDeletionTarget) -> None:
    validate_prepared_phase_view(view)
    if type(target) is not MatchedDeletionTarget:
        raise TypeError("target must be an exact MatchedDeletionTarget")
    if (
        target.subject_id != view.calendar_identity.subject_id
        or target.sensor_day_id != view.calendar_identity.sensor_day_id
        or target.primitive != view.primitive_name
        or target.prepared_source_content_digest != view.source_record_digest
    ):
        raise ValueError("matched target is crosswired with its prepared phase view")


def _interval_for(record: PreparedPhaseRecord, semantics: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if semantics == "instant":
        return record.source_timestamp, record.source_timestamp + pd.Timedelta(nanoseconds=1)
    return record.support_start, record.support_end


def _mask_from_records(
    view: PreparedPhaseView,
    target: MatchedDeletionTarget,
    *,
    scenario: str,
    records: tuple[PreparedPhaseRecord, ...],
    status: str = "eligible",
    reason: str = "eligible",
) -> MatchedRecordMask:
    ordered_ids = tuple(record.content_digest for record in records)
    intervals = tuple(_interval_for(record, view.timestamp_semantics) for record in records)
    mask = MatchedRecordMask(
        scenario=scenario,
        seed_sha256=_seed(view, target, scenario),
        target_deleted_count=target.target_deleted_count,
        record_identity_digests=ordered_ids,
        mask_intervals=intervals,
        status=status,
        reason=reason,
    )
    validate_record_mask_geometry(view, mask)
    return mask


def generate_scattered_record_mask(
    view: PreparedPhaseView,
    target: MatchedDeletionTarget,
) -> MatchedRecordMask:
    """Sample K valid identities without replacement and republish canonical order."""
    _validate_inputs(view, target)
    seed = _seed(view, target, SCATTERED_SCENARIO)
    if target.eligibility_status != "eligible":
        return MatchedRecordMask(
            scenario=SCATTERED_SCENARIO,
            seed_sha256=seed,
            target_deleted_count=target.target_deleted_count,
            record_identity_digests=(),
            mask_intervals=(),
            status="ineligible",
            reason=target.reason,
        )
    candidates = tuple(record for record in view.records if record.valid)
    if target.target_deleted_count <= 0 or target.target_deleted_count > len(candidates):
        raise ValueError("valid-record target count is infeasible")
    generator = np.random.Generator(
        np.random.PCG64(int.from_bytes(bytes.fromhex(seed)[:8], "big"))
    )
    positions = sorted(
        int(value)
        for value in generator.choice(
            len(candidates), size=target.target_deleted_count, replace=False
        )
    )
    return _mask_from_records(
        view,
        target,
        scenario=SCATTERED_SCENARIO,
        records=tuple(candidates[position] for position in positions),
    )


def _adjacent(left: PreparedPhaseRecord, right: PreparedPhaseRecord, view: PreparedPhaseView) -> bool:
    if not left.valid or not right.valid:
        return False
    if view.timestamp_semantics == "instant":
        return right.source_timestamp - left.source_timestamp == pd.Timedelta(
            minutes=view.cadence_minutes
        )
    return right.support_start == left.support_end


def _positive(record: PreparedPhaseRecord, view: PreparedPhaseView) -> bool:
    if not record.valid:
        return False
    if view.operation == "evening_p90":
        return record.event_mass > 0.0
    return record.transformed_numerator > 0.0


def _boundaries(view: PreparedPhaseView) -> tuple[pd.Timestamp, ...]:
    records = view.records
    boundaries: list[pd.Timestamp] = []
    if view.operation in ("binary_load", "activity_load"):
        for left, right in zip(records, records[1:]):
            if _adjacent(left, right, view) and left.raw_value != right.raw_value:
                boundaries.append(right.effective_timestamp)
    else:
        index = 0
        while index < len(records):
            if not _positive(records[index], view):
                index += 1
                continue
            start = index
            end = index
            while (
                end + 1 < len(records)
                and _adjacent(records[end], records[end + 1], view)
                and _positive(records[end + 1], view)
            ):
                end += 1
            boundaries.extend((records[start].support_start, records[end].support_end))
            index = end + 1
    return tuple(sorted(set(boundaries)))


def _distance_ns(record: PreparedPhaseRecord, boundaries: tuple[pd.Timestamp, ...]) -> int:
    distances: list[int] = []
    for boundary in boundaries:
        if record.support_start <= boundary <= record.support_end:
            distances.append(0)
        elif boundary < record.support_start:
            distances.append(int((record.support_start - boundary).value))
        else:
            distances.append(int((boundary - record.support_end).value))
    return min(distances)


def generate_event_boundary_record_mask(
    view: PreparedPhaseView,
    target: MatchedDeletionTarget,
) -> MatchedRecordMask:
    """Rank valid identities by contract-defined boundary distance."""
    _validate_inputs(view, target)
    seed = _seed(view, target, BOUNDARY_SCENARIO)
    if target.eligibility_status != "eligible":
        return MatchedRecordMask(
            scenario=BOUNDARY_SCENARIO,
            seed_sha256=seed,
            target_deleted_count=target.target_deleted_count,
            record_identity_digests=(),
            mask_intervals=(),
            status="ineligible",
            reason=target.reason,
        )
    boundaries = _boundaries(view)
    if not boundaries:
        return MatchedRecordMask(
            scenario=BOUNDARY_SCENARIO,
            seed_sha256=seed,
            target_deleted_count=target.target_deleted_count,
            record_identity_digests=(),
            mask_intervals=(),
            status="not_applicable",
            reason="no_contract_defined_boundary",
        )
    candidates = tuple(record for record in view.records if record.valid)
    if target.target_deleted_count > len(candidates):
        raise ValueError("valid-record target count is infeasible")
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (
            _distance_ns(item[1], boundaries),
            _sha([seed, item[1].content_digest]),
        ),
    )[: target.target_deleted_count]
    selected_positions = sorted(position for position, _ in ranked)
    return _mask_from_records(
        view,
        target,
        scenario=BOUNDARY_SCENARIO,
        records=tuple(candidates[position] for position in selected_positions),
    )


def validate_record_mask_geometry(view: PreparedPhaseView, mask: MatchedRecordMask) -> None:
    """Rebuild deleted valid identities from public records and exact half-open intervals."""
    validate_prepared_phase_view(view)
    if type(mask) is not MatchedRecordMask:
        raise TypeError("mask must be an exact MatchedRecordMask")
    if mask.status != "eligible":
        if mask.record_identity_digests or mask.mask_intervals:
            raise ValueError("noneligible record mask must not contain deletion geometry")
        return
    if len(mask.record_identity_digests) != mask.target_deleted_count:
        raise ValueError("record mask does not contain exact K identities")
    if len(set(mask.record_identity_digests)) != len(mask.record_identity_digests):
        raise ValueError("record mask contains duplicate identities")
    selected: list[str] = []
    for record in view.records:
        if not record.valid:
            continue
        if any(
            (
                start <= record.source_timestamp < end
                if view.timestamp_semantics == "instant"
                else record.support_start < end and record.support_end > start
            )
            for start, end in mask.mask_intervals
        ):
            selected.append(record.content_digest)
    if tuple(selected) != mask.record_identity_digests:
        raise ValueError("record-mask intervals do not match claimed valid identities")


def record_mask_to_intervals(
    mask: MatchedRecordMask,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    if type(mask) is not MatchedRecordMask:
        raise TypeError("mask must be an exact MatchedRecordMask")
    return mask.mask_intervals
