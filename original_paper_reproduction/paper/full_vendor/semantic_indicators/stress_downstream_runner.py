"""Synthetic-only orchestration for the frozen downstream stability analysis.

This module deliberately has no real-data discovery surface.  Its inputs are sealed
in-memory authorities/frames and its only durable output is an atomic directory below
the caller-selected synthetic temporary root.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from . import outcome
from . import phaseguard_authority
from . import stress_downstream
from . import stress_downstream_statistics


_DIGEST = re.compile(r"[0-9a-f]{64}")
_FAMILIES = ("event_count", "state_ratio", "intensity", "timing")
_PRIMITIVE_FAMILY_PAIRS = (
    ("step_load_24h", "event_count"),
    ("screen_load_24h", "state_ratio"),
    ("phone_activity_load_24h", "state_ratio"),
    ("usage_load_24h", "intensity"),
    ("mobile_light_exposure_24h", "intensity"),
    ("wearable_light_exposure_24h", "intensity"),
    ("screen_disengagement_p90", "timing"),
    ("usage_disengagement_p90", "timing"),
)
_PRIMITIVES = tuple(value[0] for value in _PRIMITIVE_FAMILY_PAIRS)
_FAMILY_BY_PRIMITIVE = dict(_PRIMITIVE_FAMILY_PAIRS)
_CONDITIONS = stress_downstream_statistics.CONDITIONS
_PACK_HEADER = b"task-12-downstream-fit-state-pack-v1\n"
_ALLOWED_STATE_FORMATS = {
    "elasticnet-coefficients-v1",
    "lightgbm-model-string-v1",
    "single-class-prevalence-v1",
}


@dataclass(frozen=True, slots=True)
class DownstreamSourcePatchProjectionSeal:
    row_count: int
    logical_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamAdapterSnapshot:
    task12_authority_digest: str
    official_g2_authority_digest: str
    official_cell_evidence_row_count: int
    official_cell_evidence_projection_digest: str
    fold_roster: tuple[tuple[int, str], ...]
    primitive_family_pairs: tuple[tuple[str, str], ...]
    eligible_target_count: int
    reference_key_count: int
    max_primary_keys_per_group: int
    prediction_universe: pd.DataFrame
    ineligible_target_ledger: pd.DataFrame
    window_exclusion_ledger: pd.DataFrame
    contiguous_provenance: pd.DataFrame
    reference_tables: outcome.DownstreamReferenceFrameTables
    reference_content_digest: str
    contexts: tuple[stress_downstream.DownstreamReferenceContext, ...]
    perturbations: tuple[stress_downstream.DownstreamPerturbation, ...]
    patches: tuple[stress_downstream.DownstreamFeaturePatch, ...]
    expected_primary_key_count: int
    expected_primary_key_digest: str
    source_patch_projection: DownstreamSourcePatchProjectionSeal
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamFitStatePackSeal:
    root: Path
    outcome_build_digest: str
    fit_ledger_row_count: int
    fit_ledger_logical_digest: str
    fit_ledger_projection_digest: str
    fit_state_index_row_count: int
    fit_state_index_logical_digest: str
    fit_state_record_count: int
    fit_state_pack_byte_count: int
    fit_state_pack_sha256: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamLoadedFitStates:
    seal: DownstreamFitStatePackSeal
    ledger: pd.DataFrame
    models: tuple[outcome.FrozenFoldModel, ...]
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamRunArtifacts:
    output_dir: Path
    adapter_digest: str
    fit_authority_digest: str
    observed_prediction_pin: stress_downstream_statistics.ObservedPredictionInputPin
    statistics_manifest: stress_downstream_statistics.DownstreamStatisticsManifest
    manifest_sha256: str
    content_digest: str


_PREDICTION_UNIVERSE_COLUMNS = (
    "participant_position", "participant", "fold", "held_out_subject",
    "family_position", "family", "primitive_position", "primitive",
    "sensor_day_id", "draw", "lifelog_date", "sleep_date",
    "elapsed_day_index", "target_content_digest",
)
_INELIGIBLE_COLUMNS = (
    "subject_id", "sensor_day_id", "primitive", "draw",
    "eligibility_status", "reason", "target_content_digest",
)
_WINDOW_COLUMNS = (
    "subject_id", "sensor_day_id", "primitive", "draw",
    "eligibility_status", "reason", "elapsed_day_index",
    "downstream_exclusion_reason", "target_content_digest",
)
_CONTIGUOUS_COLUMNS = (
    "row_seq", "subject_id", "sensor_day_id", "primitive", "draw", "scenario",
    "target_content_digest", "crosslink_content_digest", "deletion_audit_key",
    "deletion_audit_digest", "record_deletion_digest",
    "official_g2_authority_digest", "official_evidence_digest", "official_status",
    "official_original_status", "official_masked_status", "official_reason",
    "retained", "normalized_replay_status", "normalized_reason", "content_digest",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_digest(value: object, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256 digest")
    return value


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{name} must be an exact built-in int >= {minimum}")
    return value


def _token(value: object) -> list[object]:
    if value is None:
        return ["null"]
    if type(value) is str:
        return ["s", value]
    if type(value) is bool:
        return ["b", value]
    if type(value) is int:
        return ["i", str(value)]
    if type(value) is float and math.isfinite(value):
        return ["f", value.hex()]
    if isinstance(value, pd.Timestamp):
        timestamp = pd.Timestamp(value)
        if timestamp.tz is not None or timestamp != timestamp.normalize():
            raise ValueError("canonical timestamp must be a timezone-naive midnight")
        return ["ts", timestamp.isoformat(timespec="nanoseconds")]
    raise ValueError(f"unsupported canonical scalar: {type(value)!r}")


class _LogicalHasher:
    def __init__(self, domain: str, columns: tuple[tuple[str, str, bool], ...]) -> None:
        self.columns = columns
        self._hash = hashlib.sha256()
        self._hash.update(domain.encode("ascii") + b"\n")
        self._hash.update(_canonical([list(value) for value in columns]) + b"\n")
        self.row_count = 0

    def update(self, row: Mapping[str, object]) -> None:
        self._hash.update(
            _canonical([_token(row[name]) for name, _type, _nullable in self.columns])
            + b"\n"
        )
        self.row_count += 1

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _row_digest(domain: str, row: Mapping[str, object], names: Sequence[str]) -> str:
    return _sha([domain, *(_token(row[name]) for name in names)])


def _float_token(value: float) -> list[str]:
    if type(value) is not float or math.isinf(value):
        raise TypeError("frozen float must be an exact finite-or-NaN built-in float")
    return ["f64be", struct.pack(">d", value).hex()]


def _decode_float_token(value: object) -> float:
    if (
        type(value) is not list
        or len(value) != 2
        or value[0] != "f64be"
        or type(value[1]) is not str
        or len(value[1]) != 16
    ):
        raise ValueError("portable float token is invalid")
    try:
        result = struct.unpack(">d", bytes.fromhex(value[1]))[0]
    except (ValueError, struct.error) as error:
        raise ValueError("portable float token is invalid") from error
    if math.isinf(result):
        raise ValueError("portable float token cannot encode infinity")
    return float(result)


def _frame_digest(domain: str, frame: pd.DataFrame) -> str:
    if type(frame) is not pd.DataFrame:
        raise TypeError("sealed table must be an exact DataFrame")
    rows = []
    for record in frame.to_dict("records"):
        rows.append([_token(_builtin_scalar(record[name])) for name in frame.columns])
    return _sha([domain, list(frame.columns), rows])


def _builtin_scalar(value: object) -> object:
    if value is None or type(value) in {str, bool, int, float, pd.Timestamp}:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return pd.Timestamp(value)
    return value


def _same_float_bits(left: object, right: object) -> bool:
    return (
        type(left) is float
        and type(right) is float
        and struct.pack(">d", left) == struct.pack(">d", right)
    )


def _file_pin(path: Path) -> tuple[int, str]:
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _write_ipc_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as stream:
        with ipc.new_stream(stream, table.schema) as writer:
            writer.write_table(table, max_chunksize=4096)


def _read_ipc_table(path: Path, expected_schema: pa.Schema) -> pa.Table:
    with path.open("rb") as stream:
        reader = ipc.open_stream(stream)
        if reader.schema != expected_schema or reader.schema.metadata is not None:
            raise ValueError(f"IPC schema mismatch: {path.name}")
        table = reader.read_all()
    return table


def _arrow_schema(columns: tuple[tuple[str, str, bool], ...]) -> pa.Schema:
    kinds = {
        "string": pa.string(), "int64": pa.int64(), "double": pa.float64(),
        "bool": pa.bool_(),
    }
    return pa.schema(
        [pa.field(name, kinds[kind], nullable=nullable) for name, kind, nullable in columns]
    )


def _baseline_digest(value: stress_downstream.DownstreamPrimitiveBaseline) -> str:
    return _sha([
        "task-12-downstream-primitive-baseline-v1", value.primitive, value.method,
        value.k, value.donor_subject, _float_token(value.center),
        _float_token(value.scale), _float_token(value.lambda_days),
    ])


def _arm_row_digest(value: stress_downstream.DownstreamReferenceArmRow) -> str:
    return _sha([
        "task-12-downstream-reference-arm-row-v1", value.arm,
        list(value.feature_names), [_float_token(item) for item in value.feature_values],
    ])


def _context_digest(value: stress_downstream.DownstreamReferenceContext) -> str:
    return _sha([
        "task-12-downstream-reference-context-v1", value.fold,
        value.held_out_subject, value.sensor_day_id,
        [[name, _float_token(raw), _float_token(z), _float_token(confidence)]
         for name, raw, z, confidence in value.primitive_basis],
        [item.content_digest for item in value.baselines],
        [item.content_digest for item in value.arm_rows],
    ])


def _perturbation_digest(value: stress_downstream.DownstreamPerturbation) -> str:
    return _sha([
        "task-12-downstream-perturbation-v1", value.fold, value.held_out_subject,
        value.sensor_day_id, value.primitive, value.draw, value.condition,
        value.replay_status, value.reason, _float_token(value.masked_value),
    ])


def _make_contexts(
    tables: outcome.DownstreamReferenceFrameTables,
    fold_roster: tuple[tuple[int, str], ...],
) -> tuple[stress_downstream.DownstreamReferenceContext, ...]:
    contexts: list[stress_downstream.DownstreamReferenceContext] = []
    seals = {(value.fold_id, value.arm_id): value for value in tables.frame_manifest}
    for fold, subject in fold_roster:
        keys = tables.prediction_keys.loc[
            tables.prediction_keys["subject_id"].eq(subject)
        ].sort_values("sensor_day_id", kind="stable")
        baseline_rows = tables.heldout_baselines.loc[
            tables.heldout_baselines["fold"].eq(fold)
        ].sort_values("primitive_position", kind="stable")
        if tuple(baseline_rows["primitive"]) != _PRIMITIVES:
            raise ValueError("held-out baseline topology mismatch")
        baselines: list[stress_downstream.DownstreamPrimitiveBaseline] = []
        for row in baseline_rows.to_dict("records"):
            fields_value = {
                "primitive": str(row["primitive"]), "method": str(row["method"]),
                "k": int(row["k"]), "donor_subject": None,
                "center": float(row["center"]), "scale": float(row["scale"]),
                "lambda_days": float(row["lambda_days"]),
            }
            unsealed = stress_downstream.DownstreamPrimitiveBaseline(
                **fields_value, content_digest="0" * 64
            )
            baselines.append(stress_downstream.DownstreamPrimitiveBaseline(
                **fields_value, content_digest=_baseline_digest(unsealed)
            ))
        for sensor_day_id_value in keys["sensor_day_id"].tolist():
            sensor_day_id = int(sensor_day_id_value)
            basis_rows = tables.primitive_basis.loc[
                tables.primitive_basis["fold"].eq(fold)
                & tables.primitive_basis["sensor_day_id"].eq(sensor_day_id)
            ].sort_values("primitive_position", kind="stable")
            if tuple(basis_rows["primitive"]) != tuple(outcome.MATCHED_SOURCE_PRIMITIVES):
                raise ValueError("reference primitive-basis topology mismatch")
            basis = tuple(
                (
                    str(row["primitive"]), float(row["raw_value"]),
                    float(row["personalized_z"]), float(row["confidence"]),
                )
                for row in basis_rows.to_dict("records")
            )
            arm_rows: list[stress_downstream.DownstreamReferenceArmRow] = []
            for arm in outcome.DOWNSTREAM_ARM_NAMES:
                seal = seals[(fold, arm)]
                frame = tables.frames[(fold, arm)]
                selected = frame.loc[frame["sensor_day_id"].eq(sensor_day_id)]
                if len(selected) != 1:
                    raise ValueError("reference frame sensor-day crosswalk is not unique")
                row = selected.iloc[0]
                feature_names = tuple(seal.feature_names)
                values = tuple(float(row[name]) for name in feature_names)
                unsealed = stress_downstream.DownstreamReferenceArmRow(
                    arm=arm, feature_names=feature_names, feature_values=values,
                    content_digest="0" * 64,
                )
                arm_rows.append(stress_downstream.DownstreamReferenceArmRow(
                    arm=arm, feature_names=feature_names, feature_values=values,
                    content_digest=_arm_row_digest(unsealed),
                ))
            context_fields = {
                "fold": fold, "held_out_subject": subject,
                "sensor_day_id": sensor_day_id, "primitive_basis": basis,
                "baselines": tuple(baselines), "arm_rows": tuple(arm_rows),
            }
            unsealed_context = stress_downstream.DownstreamReferenceContext(
                **context_fields, content_digest="0" * 64
            )
            context = stress_downstream.DownstreamReferenceContext(
                **context_fields, content_digest=_context_digest(unsealed_context)
            )
            # Public validation is exercised by the public frame constructor.
            for arm in outcome.DOWNSTREAM_ARM_NAMES:
                stress_downstream.reference_feature_frame(context, arm)
            contexts.append(context)
    return tuple(contexts)


def _validated_evidence_tuple(
    values: tuple[phaseguard_authority.OfficialCellEvidence, ...],
) -> tuple[phaseguard_authority.OfficialCellEvidence, ...]:
    if type(values) is not tuple:
        raise TypeError("official_cell_evidence must be an exact tuple")
    previous: tuple[str, int, str, str, int] | None = None
    rebuilt: list[phaseguard_authority.OfficialCellEvidence] = []
    for value in values:
        if type(value) is not phaseguard_authority.OfficialCellEvidence:
            raise TypeError("official_cell_evidence contains a forged type")
        copy = phaseguard_authority.OfficialCellEvidence(
            **{field.name: getattr(value, field.name) for field in fields(value)}
        )
        key = (
            copy.subject_id, copy.sensor_day_id, copy.primitive, copy.scenario,
            copy.draw,
        )
        if previous is not None and key <= previous:
            raise ValueError("official_cell_evidence is duplicate or out of order")
        previous = key
        rebuilt.append(copy)
    return tuple(rebuilt)


def _selected_rows_and_ledgers(
    target: pd.DataFrame,
    primitives: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    required = {
        "subject_id", "sensor_day_id", "primitive", "draw", "eligibility_status",
        "reason", "content_digest",
    }
    if not required.issubset(target.columns):
        raise ValueError("target ledger lacks downstream eligibility fields")
    primitive_lookup = primitives.set_index("sensor_day_id", drop=False)
    selected: list[dict[str, object]] = []
    ineligible: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    reasons = {
        "invalid_reference", "zero_expected_units", "no_valid_source_contribution",
        "zero_valid_deleted_count", "infeasible_valid_deleted_count",
        "invalid_standardizer", "insufficient_timing_events",
        "no_finite_nonnegative_contribution",
    }
    for source in target.to_dict("records"):
        subject = source["subject_id"]
        primitive = source["primitive"]
        sensor_day_id = source["sensor_day_id"]
        draw = source["draw"]
        if (
            type(subject) is not str or type(primitive) is not str
            or type(sensor_day_id) is not int or type(draw) is not int
            or primitive not in _FAMILY_BY_PRIMITIVE
        ):
            raise TypeError("target ledger has a coercive or unknown key")
        _require_digest(source["content_digest"], "target content digest")
        if sensor_day_id not in primitive_lookup.index:
            raise ValueError("target sensor day is absent from primitive calendar")
        calendar = primitive_lookup.loc[sensor_day_id]
        if type(calendar) is pd.DataFrame:
            raise ValueError("primitive sensor-day identity is duplicated")
        if str(calendar["subject_id"]) != subject:
            raise ValueError("target subject/calendar crosswalk mismatch")
        status = source["eligibility_status"]
        reason = source["reason"]
        if status not in {"eligible", "ineligible"} or type(reason) is not str:
            raise ValueError("target eligibility vocabulary is invalid")
        common = {
            "subject_id": subject, "sensor_day_id": sensor_day_id,
            "primitive": primitive, "draw": draw,
            "eligibility_status": status, "reason": reason,
            "target_content_digest": source["content_digest"],
        }
        elapsed = int(calendar["elapsed_day_index"])
        if status == "ineligible":
            if reason not in reasons:
                raise ValueError("producer-ineligible reason is outside the frozen set")
            ineligible.append(common)
        elif reason != "eligible":
            raise ValueError("producer-eligible row has a changed reason")
        elif elapsed < outcome.OUTCOME_EVALUATION_START_DAY:
            excluded.append({
                **common, "elapsed_day_index": elapsed,
                "downstream_exclusion_reason":
                    "outside_frozen_outcome_evaluation_window",
            })
        else:
            selected.append({
                **common, "lifelog_date": pd.Timestamp(calendar["lifelog_date"]),
                "sleep_date": pd.Timestamp(calendar["sleep_date"]),
                "elapsed_day_index": elapsed,
            })
    if not selected:
        raise ValueError("downstream prediction universe is empty")
    return (
        selected,
        pd.DataFrame.from_records(ineligible, columns=_INELIGIBLE_COLUMNS),
        pd.DataFrame.from_records(excluded, columns=_WINDOW_COLUMNS),
    )


def _make_perturbation(
    *, fold: int, subject: str, sensor_day_id: int, primitive: str, draw: int,
    condition: str, status: str, reason: str, masked_value: float,
) -> stress_downstream.DownstreamPerturbation:
    values = {
        "fold": fold, "held_out_subject": subject, "sensor_day_id": sensor_day_id,
        "primitive": primitive, "draw": draw, "condition": condition,
        "replay_status": status, "reason": reason,
        "masked_value": float(masked_value),
    }
    unsealed = stress_downstream.DownstreamPerturbation(
        **values, content_digest="0" * 64
    )
    return stress_downstream.DownstreamPerturbation(
        **values, content_digest=_perturbation_digest(unsealed)
    )


def _projection_rows(
    universe: pd.DataFrame,
    contexts: tuple[stress_downstream.DownstreamReferenceContext, ...],
    patches: tuple[stress_downstream.DownstreamFeaturePatch, ...],
) -> tuple[DownstreamSourcePatchProjectionSeal, int, str]:
    context_by_key = {
        (value.held_out_subject, value.sensor_day_id): value for value in contexts
    }
    patch_by_key = {
        (
            value.held_out_subject, value.sensor_day_id, value.primitive, value.draw,
            value.condition, value.arm,
        ): value
        for value in patches
    }
    source_hasher = _LogicalHasher(
        "task-12-downstream-source-patch-projection-digest-v1",
        stress_downstream_statistics.SOURCE_PATCH_PROJECTION_COLUMNS,
    )
    key_hasher = _LogicalHasher(
        "task-12-downstream-expected-primary-key-digest-v1",
        stress_downstream_statistics.EXPECTED_KEY_COLUMNS,
    )
    row_seq = 0
    key_count = 0
    for participant_position, (_fold, participant) in enumerate(
        sorted({(int(row.fold), str(row.participant)) for row in universe.itertuples()},
               key=lambda value: value[0])
    ):
        for family_position, family in enumerate(_FAMILIES):
            rows = universe.loc[
                universe["participant"].eq(participant)
                & universe["family"].eq(family)
            ].sort_values(
                ["primitive_position", "sensor_day_id", "draw"], kind="stable"
            )
            for arm_position, arm in enumerate(outcome.DOWNSTREAM_ARM_NAMES):
                for model_position, model_name in enumerate(outcome.MODEL_NAMES):
                    for target_position, target in enumerate(outcome.TARGET_COLUMNS):
                        for selected in rows.to_dict("records"):
                            base = {
                                "participant_position": participant_position,
                                "participant": participant, "fold": int(selected["fold"]),
                                "held_out_subject": participant,
                                "family_position": family_position, "family": family,
                                "arm_position": arm_position, "arm": arm,
                                "model_position": model_position,
                                "model_name": model_name,
                                "target_position": target_position, "target": target,
                                "primitive_position": int(selected["primitive_position"]),
                                "primitive": str(selected["primitive"]),
                                "sensor_day_id": int(selected["sensor_day_id"]),
                                "draw": int(selected["draw"]),
                            }
                            key_hasher.update(base)
                            key_count += 1
                            context = context_by_key[(participant, base["sensor_day_id"])]
                            for condition_position, condition in enumerate(_CONDITIONS):
                                patch = patch_by_key[
                                    (participant, base["sensor_day_id"], base["primitive"],
                                     base["draw"], condition, arm)
                                ]
                                source_hasher.update({
                                    "row_seq": row_seq, **base,
                                    "condition_position": condition_position,
                                    "condition": condition,
                                    "reference_context_digest": context.content_digest,
                                    "source_row_digest": patch.content_digest,
                                })
                                row_seq += 1
    logical = source_hasher.hexdigest()
    seal = DownstreamSourcePatchProjectionSeal(
        row_count=source_hasher.row_count,
        logical_digest=logical,
        content_digest=_sha([
            "task-12-downstream-source-patch-projection-seal-v1",
            source_hasher.row_count, logical,
        ]),
    )
    return seal, key_count, key_hasher.hexdigest()


def _adapter_content_digest(snapshot: DownstreamAdapterSnapshot) -> str:
    return _sha([
        "task-12-downstream-adapter-snapshot-v1",
        snapshot.task12_authority_digest, snapshot.official_g2_authority_digest,
        snapshot.official_cell_evidence_row_count,
        snapshot.official_cell_evidence_projection_digest,
        [list(value) for value in snapshot.fold_roster],
        [list(value) for value in snapshot.primitive_family_pairs],
        snapshot.eligible_target_count, snapshot.reference_key_count,
        snapshot.max_primary_keys_per_group,
        _frame_digest("task-12-downstream-prediction-universe-v1", snapshot.prediction_universe),
        _frame_digest("task-12-downstream-ineligible-target-ledger-v1", snapshot.ineligible_target_ledger),
        _frame_digest("task-12-downstream-window-exclusion-ledger-v1", snapshot.window_exclusion_ledger),
        _frame_digest("task-12-downstream-contiguous-provenance-v1", snapshot.contiguous_provenance),
        snapshot.reference_content_digest,
        [value.content_digest for value in snapshot.contexts],
        [value.content_digest for value in snapshot.perturbations],
        [value.content_digest for value in snapshot.patches],
        snapshot.expected_primary_key_count, snapshot.expected_primary_key_digest,
        snapshot.source_patch_projection.content_digest,
    ])


def build_downstream_adapter_snapshot(
    task12_authority: stress_downstream.Task12CanonicalReplayAuthority,
    official_cell_evidence: tuple[phaseguard_authority.OfficialCellEvidence, ...],
    primitives: pd.DataFrame,
    *,
    expected_task12_authority_digest: str,
    expected_official_g2_authority_digest: str,
    expected_interval_boundary_policy: str,
    expected_primitive_digest: str,
) -> DownstreamAdapterSnapshot:
    """Build the complete label-blind synthetic adapter in one bulk pass."""
    if type(task12_authority) is not stress_downstream.Task12CanonicalReplayAuthority:
        raise TypeError("task12_authority has the wrong exact type")
    expected_task12 = _require_digest(
        expected_task12_authority_digest, "expected Task 12 authority digest"
    )
    official_root = _require_digest(
        expected_official_g2_authority_digest, "expected Official G2 authority digest"
    )
    _require_digest(expected_primitive_digest, "expected primitive digest")
    if task12_authority.content_digest != expected_task12:
        raise ValueError("Task 12 authority digest mismatch")
    if type(primitives) is not pd.DataFrame:
        raise TypeError("primitives must be an exact DataFrame")

    target = stress_downstream.read_task12_replay_table(task12_authority, "target_ledger")
    crosslink = stress_downstream.read_task12_replay_table(
        task12_authority, "official_current_crosslink"
    )
    mask = stress_downstream.read_task12_replay_table(task12_authority, "mask_ledger")
    audit = stress_downstream.read_task12_replay_table(task12_authority, "deletion_audit")
    replay = stress_downstream.read_task12_replay_table(task12_authority, "cell_replays")
    del mask, audit  # Their complete graph was validated by the immutable snapshot reader.

    selected, ineligible, excluded = _selected_rows_and_ledgers(target, primitives)
    subjects = tuple(sorted({str(row["subject_id"]) for row in selected}))
    if len(subjects) != 10:
        raise ValueError("downstream adapter requires exactly ten participants")
    fold_roster = tuple((position, subject) for position, subject in enumerate(subjects))
    fold_by_subject = {subject: fold for fold, subject in fold_roster}
    prediction_records: list[dict[str, object]] = []
    for row in selected:
        subject = str(row["subject_id"])
        primitive = str(row["primitive"])
        family = _FAMILY_BY_PRIMITIVE[primitive]
        prediction_records.append({
            "participant_position": subjects.index(subject), "participant": subject,
            "fold": fold_by_subject[subject], "held_out_subject": subject,
            "family_position": _FAMILIES.index(family), "family": family,
            "primitive_position": _PRIMITIVES.index(primitive), "primitive": primitive,
            "sensor_day_id": int(row["sensor_day_id"]), "draw": int(row["draw"]),
            "lifelog_date": pd.Timestamp(row["lifelog_date"]),
            "sleep_date": pd.Timestamp(row["sleep_date"]),
            "elapsed_day_index": int(row["elapsed_day_index"]),
            "target_content_digest": str(row["target_content_digest"]),
        })
    universe = pd.DataFrame.from_records(
        prediction_records, columns=_PREDICTION_UNIVERSE_COLUMNS
    ).sort_values(
        ["participant_position", "family_position", "primitive_position",
         "sensor_day_id", "draw"], kind="stable"
    ).reset_index(drop=True)
    cells = set(zip(universe["participant"], universe["family"], strict=True))
    if cells != {(subject, family) for subject in subjects for family in _FAMILIES}:
        raise ValueError("every participant/family cell must be nonempty")
    group_sizes = universe.groupby(["participant", "family"], sort=False).size()
    max_primary_keys = int(group_sizes.max())

    reference_keys = (
        universe.loc[:, ["sensor_day_id", "participant", "lifelog_date", "sleep_date",
                         "elapsed_day_index"]]
        .rename(columns={"participant": "subject_id"})
        .drop_duplicates()
        .sort_values(["subject_id", "sensor_day_id"], kind="stable")
        .reset_index(drop=True)
    )
    reference_keys["sensor_day_id"] = reference_keys["sensor_day_id"].astype("int64")
    reference_keys["subject_id"] = reference_keys["subject_id"].astype("str")
    reference_keys["lifelog_date"] = pd.to_datetime(
        reference_keys["lifelog_date"]
    ).astype("datetime64[ns]")
    reference_keys["sleep_date"] = pd.to_datetime(
        reference_keys["sleep_date"]
    ).astype("datetime64[ns]")
    reference_keys["elapsed_day_index"] = reference_keys["elapsed_day_index"].astype("int64")
    reference_tables = outcome.build_downstream_reference_frames(
        primitives.copy(deep=True), reference_keys,
        expected_interval_boundary_policy=expected_interval_boundary_policy,
        expected_primitive_digest=expected_primitive_digest,
    )
    reference_digest = reference_tables.content_digest
    outcome.validate_downstream_reference_frames(
        reference_tables, expected_content_digest=reference_digest
    )
    contexts = _make_contexts(reference_tables, fold_roster)
    context_by_key = {
        (value.held_out_subject, value.sensor_day_id): value for value in contexts
    }

    evidence = _validated_evidence_tuple(official_cell_evidence)
    required_evidence_keys = {
        (str(row["participant"]), int(row["sensor_day_id"]), str(row["primitive"]),
         "contiguous_20pct", int(row["draw"]))
        for row in universe.to_dict("records")
    }
    evidence_keys = {
        (value.subject_id, value.sensor_day_id, value.primitive, value.scenario, value.draw)
        for value in evidence
    }
    if evidence_keys != required_evidence_keys or len(evidence) != len(required_evidence_keys):
        raise ValueError("Official-cell evidence keys do not exactly match selected targets")
    evidence_projection = _sha([
        "task-12-downstream-selected-official-cell-evidence-v1", official_root,
        [[value.subject_id, value.sensor_day_id, value.primitive, value.scenario,
          value.draw, value.content_digest] for value in evidence],
    ])
    evidence_by_key = {
        (value.subject_id, value.sensor_day_id, value.primitive, value.scenario, value.draw): value
        for value in evidence
    }

    target_by_key = {
        (str(row["subject_id"]), int(row["sensor_day_id"]), str(row["primitive"]),
         int(row["draw"])): row
        for row in target.to_dict("records")
    }
    cross_by_key = {
        (str(row["subject_id"]), int(row["sensor_day_id"]), str(row["primitive"]),
         int(row["draw"])): row
        for row in crosslink.to_dict("records")
    }
    replay_by_key = {
        (str(row["subject_id"]), int(row["sensor_day_id"]), str(row["primitive"]),
         int(row["draw"]), str(row["scenario"])): row
        for row in replay.to_dict("records")
    }
    primitive_lookup = primitives.set_index("sensor_day_id", drop=False)
    perturbations: list[stress_downstream.DownstreamPerturbation] = []
    patches: list[stress_downstream.DownstreamFeaturePatch] = []
    provenance: list[dict[str, object]] = []
    for universe_row in universe.to_dict("records"):
        subject = str(universe_row["participant"])
        sensor_day_id = int(universe_row["sensor_day_id"])
        primitive = str(universe_row["primitive"])
        draw = int(universe_row["draw"])
        fold = int(universe_row["fold"])
        base_key = (subject, sensor_day_id, primitive, draw)
        target_row = target_by_key[base_key]
        cross_row = cross_by_key[base_key]
        if (
            cross_row["target_content_digest"] != target_row["content_digest"]
            or cross_row["scenario"] != "contiguous_20pct"
        ):
            raise ValueError("contiguous target/crosslink provenance is crosswired")
        evidence_row = evidence_by_key[(*base_key[:3], "contiguous_20pct", draw)]
        for name, evidence_value, cross_name in (
            ("deletion audit key", evidence_row.deletion_audit_key, "deletion_audit_key"),
            ("record deletion digest", evidence_row.record_deletion_digest,
             "historical_record_deletion_digest"),
        ):
            if cross_name not in cross_row or cross_row[cross_name] != evidence_value:
                raise ValueError(f"contiguous {name} provenance is crosswired")
        calendar = primitive_lookup.loc[sensor_day_id]
        if (
            evidence_row.subject_id != subject
            or evidence_row.lifelog_date != pd.Timestamp(calendar["lifelog_date"])
            or evidence_row.elapsed_day_index != int(calendar["elapsed_day_index"])
        ):
            raise ValueError("Official-cell evidence calendar identity is crosswired")
        random_row = replay_by_key[(*base_key, "scattered_random_20pct")]
        event_row = replay_by_key[(*base_key, "event_boundary_20pct")]
        for source in (random_row, event_row):
            if source["target_digest"] != target_row["content_digest"]:
                raise ValueError("Task 12 replay target provenance is crosswired")
        if all(name in random_row and name in event_row for name in
               ("reference_value", "standardizer", "standardizer_source")):
            reference_values = (float(random_row["reference_value"]),
                                float(event_row["reference_value"]),
                                float(evidence_row.original_value),
                                float(calendar[primitive]))
            if not all(_same_float_bits(reference_values[0], value)
                       for value in reference_values[1:]):
                raise ValueError("contiguous reference value provenance is crosswired")
            if (
                not _same_float_bits(float(random_row["standardizer"]),
                                     float(event_row["standardizer"]))
                or not _same_float_bits(float(random_row["standardizer"]),
                                        evidence_row.standardizer)
                or random_row["standardizer_source"] != event_row["standardizer_source"]
                or random_row["standardizer_source"] != evidence_row.standardizer_source
            ):
                raise ValueError("contiguous standardizer provenance is crosswired")

        sources = (
            ("scattered_random_20pct", str(random_row["status"]),
             str(random_row["reason"]), float(random_row["masked_value"])),
            (
                "contiguous_20pct",
                "finite" if evidence_row.retained else (
                    "abstained_no_event_support" if primitive in
                    ("screen_disengagement_p90", "usage_disengagement_p90")
                    else "abstained_insufficient_support"
                ),
                "finite" if evidence_row.retained else (
                    "abstained_no_event_support" if primitive in
                    ("screen_disengagement_p90", "usage_disengagement_p90")
                    else "abstained_insufficient_support"
                ),
                float(evidence_row.masked_value) if evidence_row.retained else float("nan"),
            ),
            ("event_boundary_20pct", str(event_row["status"]),
             str(event_row["reason"]), float(event_row["masked_value"])),
        )
        context = context_by_key[(subject, sensor_day_id)]
        for condition, status, reason, masked_value in sources:
            perturbation = _make_perturbation(
                fold=fold, subject=subject, sensor_day_id=sensor_day_id,
                primitive=primitive, draw=draw, condition=condition, status=status,
                reason=reason, masked_value=masked_value,
            )
            perturbations.append(perturbation)
            for arm in outcome.DOWNSTREAM_ARM_NAMES:
                patches.append(stress_downstream.build_downstream_feature_patch(
                    context, perturbation, arm
                ))
        contiguous = perturbations[-2]
        provenance_row = {
            "row_seq": len(provenance), "subject_id": subject,
            "sensor_day_id": sensor_day_id, "primitive": primitive, "draw": draw,
            "scenario": "contiguous_20pct",
            "target_content_digest": str(target_row["content_digest"]),
            "crosslink_content_digest": str(cross_row["crosslink_content_digest"]),
            "deletion_audit_key": evidence_row.deletion_audit_key,
            "deletion_audit_digest": evidence_row.deletion_audit_digest,
            "record_deletion_digest": evidence_row.record_deletion_digest,
            "official_g2_authority_digest": official_root,
            "official_evidence_digest": evidence_row.content_digest,
            "official_status": evidence_row.status,
            "official_original_status": evidence_row.original_status,
            "official_masked_status": evidence_row.masked_status,
            "official_reason": evidence_row.reason, "retained": evidence_row.retained,
            "normalized_replay_status": contiguous.replay_status,
            "normalized_reason": contiguous.reason,
        }
        provenance_row["content_digest"] = _row_digest(
            "task-12-downstream-contiguous-provenance-row-v1", provenance_row,
            _CONTIGUOUS_COLUMNS[:-1],
        )
        provenance.append(provenance_row)

    contiguous_provenance = pd.DataFrame.from_records(
        provenance, columns=_CONTIGUOUS_COLUMNS
    )
    patch_tuple = tuple(patches)
    perturbation_tuple = tuple(perturbations)
    projection, primary_count, primary_digest = _projection_rows(
        universe, contexts, patch_tuple
    )
    n = len(universe)
    d = len(reference_keys)
    if (
        len(contexts) != d or len(reference_tables.primitive_basis) != 15 * d
        or len(reference_tables.heldout_baselines) != 80
        or len(perturbation_tuple) != 3 * n or len(patch_tuple) != 6 * n
        or primary_count != 16 * n or projection.row_count != 48 * n
    ):
        raise ValueError("downstream adapter topology is incomplete")
    values = {
        "task12_authority_digest": expected_task12,
        "official_g2_authority_digest": official_root,
        "official_cell_evidence_row_count": len(evidence),
        "official_cell_evidence_projection_digest": evidence_projection,
        "fold_roster": fold_roster,
        "primitive_family_pairs": _PRIMITIVE_FAMILY_PAIRS,
        "eligible_target_count": n, "reference_key_count": d,
        "max_primary_keys_per_group": max_primary_keys,
        "prediction_universe": universe, "ineligible_target_ledger": ineligible,
        "window_exclusion_ledger": excluded,
        "contiguous_provenance": contiguous_provenance,
        "reference_tables": reference_tables,
        "reference_content_digest": reference_digest, "contexts": contexts,
        "perturbations": perturbation_tuple, "patches": patch_tuple,
        "expected_primary_key_count": primary_count,
        "expected_primary_key_digest": primary_digest,
        "source_patch_projection": projection,
    }
    unsealed = DownstreamAdapterSnapshot(**values, content_digest="0" * 64)
    result = DownstreamAdapterSnapshot(
        **values, content_digest=_adapter_content_digest(unsealed)
    )
    validate_downstream_adapter_snapshot(
        result, expected_content_digest=result.content_digest,
        expected_source_patch_projection_digest=projection.logical_digest,
    )
    return result


def validate_downstream_adapter_snapshot(
    snapshot: DownstreamAdapterSnapshot,
    *,
    expected_content_digest: str,
    expected_source_patch_projection_digest: str,
) -> None:
    if type(snapshot) is not DownstreamAdapterSnapshot:
        raise TypeError("snapshot has the wrong exact type")
    expected = _require_digest(expected_content_digest, "expected adapter digest")
    expected_projection = _require_digest(
        expected_source_patch_projection_digest, "expected source-patch digest"
    )
    for name in (
        "task12_authority_digest", "official_g2_authority_digest",
        "official_cell_evidence_projection_digest", "reference_content_digest",
        "expected_primary_key_digest", "content_digest",
    ):
        _require_digest(getattr(snapshot, name), name)
    if snapshot.content_digest != expected:
        raise ValueError("independently retained adapter digest mismatch")
    if type(snapshot.source_patch_projection) is not DownstreamSourcePatchProjectionSeal:
        raise TypeError("source-patch seal has the wrong exact type")
    projection = snapshot.source_patch_projection
    if projection.logical_digest != expected_projection:
        raise ValueError("independently retained source-patch digest mismatch")
    if projection.content_digest != _sha([
        "task-12-downstream-source-patch-projection-seal-v1",
        projection.row_count, projection.logical_digest,
    ]):
        raise ValueError("source-patch projection seal is forged")
    if (
        type(snapshot.fold_roster) is not tuple or len(snapshot.fold_roster) != 10
        or type(snapshot.primitive_family_pairs) is not tuple
        or snapshot.primitive_family_pairs != _PRIMITIVE_FAMILY_PAIRS
        or any(type(value) is not tuple or type(value[0]) is not int
               or type(value[1]) is not str for value in snapshot.fold_roster)
    ):
        raise TypeError("adapter immutable roster topology is invalid")
    for name in (
        "official_cell_evidence_row_count", "eligible_target_count",
        "reference_key_count", "max_primary_keys_per_group",
        "expected_primary_key_count",
    ):
        _require_int(getattr(snapshot, name), name, minimum=1)
    if (
        type(snapshot.prediction_universe) is not pd.DataFrame
        or tuple(snapshot.prediction_universe.columns) != _PREDICTION_UNIVERSE_COLUMNS
        or type(snapshot.ineligible_target_ledger) is not pd.DataFrame
        or tuple(snapshot.ineligible_target_ledger.columns) != _INELIGIBLE_COLUMNS
        or type(snapshot.window_exclusion_ledger) is not pd.DataFrame
        or tuple(snapshot.window_exclusion_ledger.columns) != _WINDOW_COLUMNS
        or type(snapshot.contiguous_provenance) is not pd.DataFrame
        or tuple(snapshot.contiguous_provenance.columns) != _CONTIGUOUS_COLUMNS
    ):
        raise ValueError("adapter table topology is invalid")
    outcome.validate_downstream_reference_frames(
        snapshot.reference_tables,
        expected_content_digest=snapshot.reference_content_digest,
    )
    if (
        snapshot.official_cell_evidence_row_count != snapshot.eligible_target_count
        or len(snapshot.prediction_universe) != snapshot.eligible_target_count
        or snapshot.expected_primary_key_count != 16 * snapshot.eligible_target_count
        or projection.row_count != 48 * snapshot.eligible_target_count
        or len(snapshot.contexts) != snapshot.reference_key_count
        or len(snapshot.perturbations) != 3 * snapshot.eligible_target_count
        or len(snapshot.patches) != 6 * snapshot.eligible_target_count
    ):
        raise ValueError("adapter count topology is invalid")
    if _adapter_content_digest(snapshot) != snapshot.content_digest:
        raise ValueError("adapter content digest is forged")


_FIT_LEDGER_COLUMNS = (
    ("fit_seq", "int64", False), ("heldout_position", "int64", False),
    ("fold", "int64", False), ("held_out_subject", "string", False),
    ("arm_position", "int64", False), ("arm", "string", False),
    ("model_position", "int64", False), ("model_name", "string", False),
    ("target_position", "int64", False), ("target", "string", False),
    ("declared_feature_names_json", "string", False),
    ("training_row_count", "int64", False), ("training_digest", "string", False),
    ("preprocessor_state_json", "string", False),
    ("preprocessor_digest", "string", False),
    ("model_state_format", "string", False),
    ("model_state_byte_count", "int64", False),
    ("model_state_sha256", "string", False), ("model_digest", "string", False),
    ("fit_content_digest", "string", False),
    ("fit_state_record_digest", "string", False),
    ("fit_ledger_record_digest", "string", False),
)
_FIT_INDEX_COLUMNS = (
    ("fit_seq", "int64", False), ("byte_offset", "int64", False),
    ("byte_count", "int64", False), ("record_sha256", "string", False),
    ("record_content_digest", "string", False),
    ("fit_content_digest", "string", False),
)
_FIT_LEDGER_SCHEMA = _arrow_schema(_FIT_LEDGER_COLUMNS)
_FIT_INDEX_SCHEMA = _arrow_schema(_FIT_INDEX_COLUMNS)
_FIT_LEDGER_DIGEST_NAMES = tuple(
    name for name, _kind, _nullable in _FIT_LEDGER_COLUMNS
    if name != "fit_ledger_record_digest"
)


def _preprocessor_document(state: outcome.FrozenPreprocessorState) -> dict[str, object]:
    return {
        "declared_feature_names": list(state.declared_feature_names),
        "output_feature_names": list(state.output_feature_names),
        "medians": [_float_token(value) for value in state.medians],
        "centers": [_float_token(value) for value in state.centers],
        "scales": [_float_token(value) for value in state.scales],
        "missing_indicator_names": list(state.missing_indicator_names),
        "dropped_constant_columns": list(state.dropped_constant_columns),
        "content_digest": state.content_digest,
    }


def _preprocessor_from_document(document: object) -> outcome.FrozenPreprocessorState:
    if type(document) is not dict or set(document) != {
        "declared_feature_names", "output_feature_names", "medians", "centers",
        "scales", "missing_indicator_names", "dropped_constant_columns",
        "content_digest",
    }:
        raise ValueError("portable preprocessor document is invalid")
    for name in (
        "declared_feature_names", "output_feature_names", "missing_indicator_names",
        "dropped_constant_columns",
    ):
        values = document[name]
        if type(values) is not list or any(type(value) is not str for value in values):
            raise TypeError("portable preprocessor string topology is invalid")
    return outcome.FrozenPreprocessorState(
        declared_feature_names=tuple(document["declared_feature_names"]),
        output_feature_names=tuple(document["output_feature_names"]),
        medians=tuple(_decode_float_token(value) for value in document["medians"]),
        centers=tuple(_decode_float_token(value) for value in document["centers"]),
        scales=tuple(_decode_float_token(value) for value in document["scales"]),
        missing_indicator_names=tuple(document["missing_indicator_names"]),
        dropped_constant_columns=tuple(document["dropped_constant_columns"]),
        content_digest=_require_digest(document["content_digest"], "preprocessor digest"),
    )


def _training_evaluation(
    training_arms: outcome.OutcomeArmTables,
    training_labels: pd.DataFrame,
) -> pd.DataFrame:
    if type(training_labels) is not pd.DataFrame:
        raise TypeError("training_labels must be an exact DataFrame")
    key_names = ["subject_id", "lifelog_date", "sleep_date"]
    required = set(key_names).union(outcome.TARGET_COLUMNS)
    if not required.issubset(training_labels.columns):
        raise ValueError("training label frame lacks required columns")
    labels = training_labels.loc[:, [*key_names, *outcome.TARGET_COLUMNS]].copy()
    for name in ("lifelog_date", "sleep_date"):
        labels[name] = pd.to_datetime(labels[name], errors="raise")
        if labels[name].dt.tz is not None or not labels[name].eq(labels[name].dt.normalize()).all():
            raise ValueError("training label dates must be exact local midnights")
    if labels.duplicated(key_names).any():
        raise ValueError("training label keys are duplicated")
    for target in outcome.TARGET_COLUMNS:
        values = labels[target]
        if values.isna().any() or not values.isin((0, 1)).all():
            raise ValueError("training targets must be exact binary values")
        labels[target] = values.astype("int64")
    joined = training_arms.evaluation_keys.merge(
        labels, on=key_names, how="left", validate="one_to_one", indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("training labels do not cover the frozen evaluation keys")
    return joined.drop(columns="_merge")


def _training_digest(
    evaluation_rows: pd.DataFrame,
    target: str,
) -> str:
    records = []
    for row in evaluation_rows.to_dict("records"):
        label = row[target]
        if isinstance(label, np.integer):
            label = int(label)
        records.append([
            int(row["sensor_day_id"]), str(row["subject_id"]),
            pd.Timestamp(row["lifelog_date"]).isoformat(timespec="nanoseconds"),
            pd.Timestamp(row["sleep_date"]).isoformat(timespec="nanoseconds"),
            int(row["elapsed_day_index"]), target, label,
        ])
    return _sha(["task-12-downstream-training-digest-v1", records])


def _fit_state_payload(
    model: outcome.FrozenFoldModel,
    *,
    fit_seq: int,
    heldout_position: int,
    arm_position: int,
    model_position: int,
    target_position: int,
    outcome_build_digest: str,
    reference_content_digest: str,
) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_name": "task-12-downstream-fit-state-record-v1",
        "fit_seq": fit_seq, "heldout_position": heldout_position,
        "fold": model.fold, "held_out_subject": model.held_out_subject,
        "arm_position": arm_position, "arm": model.arm,
        "model_position": model_position, "model_name": model.model_name,
        "target_position": target_position, "target": model.target,
        "declared_feature_names": list(model.declared_feature_names),
        "preprocessor_state": _preprocessor_document(model.preprocessor_state),
        "model_state_format": model.model_state_format,
        "model_state_base64": base64.b64encode(model.model_state_bytes).decode("ascii"),
        "preprocessor_digest": model.preprocessor_digest,
        "model_digest": model.model_digest,
        "fit_content_digest": model.content_digest,
        "outcome_build_digest": outcome_build_digest,
        "reference_content_digest": reference_content_digest,
    }
    core["record_content_digest"] = _sha([
        "task-12-downstream-fit-state-record-v1", core
    ])
    return core


def _fit_authority_digest(
    *,
    outcome_build_digest: str,
    reference_content_digest: str,
    fit_ledger_row_count: int,
    fit_ledger_logical_digest: str,
    fit_ledger_projection_digest: str,
    fit_state_index_row_count: int,
    fit_state_index_logical_digest: str,
    fit_state_record_count: int,
    fit_state_pack_byte_count: int,
    fit_state_pack_sha256: str,
) -> str:
    return _sha([
        "task-12-downstream-fit-authority-v1", outcome_build_digest,
        reference_content_digest, fit_ledger_row_count, fit_ledger_logical_digest,
        fit_ledger_projection_digest, fit_state_index_row_count,
        fit_state_index_logical_digest, fit_state_record_count,
        fit_state_pack_byte_count, fit_state_pack_sha256,
    ])


def write_downstream_fit_state_pack(
    training_arms: outcome.OutcomeArmTables,
    training_labels: pd.DataFrame,
    reference_tables: outcome.DownstreamReferenceFrameTables,
    output_dir: Path,
    *,
    expected_outcome_build_digest: str,
    expected_reference_content_digest: str,
    root_seed: int,
) -> DownstreamFitStatePackSeal:
    """Fit the exact 160 LOSO states once and write a portable append-only pack."""
    expected_outcome = _require_digest(
        expected_outcome_build_digest, "expected outcome build digest"
    )
    expected_reference = _require_digest(
        expected_reference_content_digest, "expected reference content digest"
    )
    _require_int(root_seed, "root_seed")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib Path")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir()
    try:
        outcome.validate_outcome_arms(
            training_arms, expected_content_digest=expected_outcome
        )
        outcome.validate_downstream_reference_frames(
            reference_tables, expected_content_digest=expected_reference
        )
        evaluation = _training_evaluation(training_arms, training_labels)
        subjects = tuple(sorted(str(value) for value in evaluation["subject_id"].unique()))
        if len(subjects) != 10:
            raise ValueError("fit pack requires exactly ten held-out participants")
        reference_seals = {
            (value.fold_id, value.arm_id): value for value in reference_tables.frame_manifest
        }
        ledger_rows: list[dict[str, object]] = []
        index_rows: list[dict[str, object]] = []
        pack_path = output_dir / "fit_states.pack"
        with pack_path.open("xb") as pack:
            pack.write(_PACK_HEADER)
            for heldout_position, held_out_subject in enumerate(subjects):
                fold = heldout_position
                train_mask = ~evaluation["subject_id"].eq(held_out_subject)
                outer_training = evaluation.loc[train_mask].reset_index(drop=True)
                if outer_training.empty or outer_training["subject_id"].eq(held_out_subject).any():
                    raise ValueError("held-out rows leaked into downstream fitting")
                for arm_position, arm in enumerate(outcome.DOWNSTREAM_ARM_NAMES):
                    frame = (
                        training_arms.frames[(fold, arm)]
                        .set_index("sensor_day_id", drop=False)
                        .reindex(evaluation["sensor_day_id"].to_numpy())
                        .reset_index(drop=True)
                    )
                    declared = tuple(reference_seals[(fold, arm)].feature_names)
                    if any(name not in frame.columns for name in declared):
                        raise ValueError("training/reference declared features disagree")
                    train_features = frame.loc[train_mask.to_numpy(), list(declared)].reset_index(drop=True)
                    for model_position, model_name in enumerate(outcome.MODEL_NAMES):
                        for target_position, target in enumerate(outcome.TARGET_COLUMNS):
                            fit_seq = (((heldout_position * 2 + arm_position) * 2
                                        + model_position) * 4 + target_position)
                            model = outcome.fit_frozen_fold_model(
                                train_features,
                                outer_training[target].reset_index(drop=True),
                                fold=fold, held_out_subject=held_out_subject, arm=arm,
                                model_name=model_name, target=target,
                                declared_feature_names=declared, root_seed=root_seed,
                            )
                            payload = _fit_state_payload(
                                model, fit_seq=fit_seq,
                                heldout_position=heldout_position,
                                arm_position=arm_position,
                                model_position=model_position,
                                target_position=target_position,
                                outcome_build_digest=expected_outcome,
                                reference_content_digest=expected_reference,
                            )
                            payload_bytes = _canonical(payload)
                            offset = pack.tell()
                            pack.write(struct.pack(">Q", len(payload_bytes)))
                            pack.write(payload_bytes)
                            record_byte_count = 8 + len(payload_bytes)
                            index_rows.append({
                                "fit_seq": fit_seq, "byte_offset": offset,
                                "byte_count": record_byte_count,
                                "record_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                                "record_content_digest": payload["record_content_digest"],
                                "fit_content_digest": model.content_digest,
                            })
                            preprocessor_json = _canonical(
                                _preprocessor_document(model.preprocessor_state)
                            ).decode("utf-8")
                            ledger_row: dict[str, object] = {
                                "fit_seq": fit_seq,
                                "heldout_position": heldout_position, "fold": fold,
                                "held_out_subject": held_out_subject,
                                "arm_position": arm_position, "arm": arm,
                                "model_position": model_position,
                                "model_name": model_name,
                                "target_position": target_position, "target": target,
                                "declared_feature_names_json": _canonical(
                                    list(declared)
                                ).decode("utf-8"),
                                "training_row_count": len(outer_training),
                                "training_digest": _training_digest(outer_training, target),
                                "preprocessor_state_json": preprocessor_json,
                                "preprocessor_digest": model.preprocessor_digest,
                                "model_state_format": model.model_state_format,
                                "model_state_byte_count": len(model.model_state_bytes),
                                "model_state_sha256": hashlib.sha256(
                                    model.model_state_bytes
                                ).hexdigest(),
                                "model_digest": model.model_digest,
                                "fit_content_digest": model.content_digest,
                                "fit_state_record_digest": payload["record_content_digest"],
                            }
                            ledger_row["fit_ledger_record_digest"] = _row_digest(
                                "task-12-downstream-fit-ledger-row-v1", ledger_row,
                                _FIT_LEDGER_DIGEST_NAMES,
                            )
                            ledger_rows.append(ledger_row)
            if len(ledger_rows) != 160 or len(index_rows) != 160:
                raise RuntimeError("fit pack did not produce exactly 160 states")
            outcome.validate_outcome_arms(
                training_arms, expected_content_digest=expected_outcome
            )
            pack.flush()
            os.fsync(pack.fileno())

        ledger_table = pa.Table.from_pylist(ledger_rows, schema=_FIT_LEDGER_SCHEMA)
        index_table = pa.Table.from_pylist(index_rows, schema=_FIT_INDEX_SCHEMA)
        _write_ipc_table(output_dir / "fit_ledger.arrow", ledger_table)
        _write_ipc_table(output_dir / "fit_state_index.arrow", index_table)
        ledger_hasher = _LogicalHasher(
            "task-12-downstream-fit-ledger-logical-v1", _FIT_LEDGER_COLUMNS
        )
        projection_hasher = _LogicalHasher(
            "task-12-downstream-fit-ledger-projection-digest-v1",
            stress_downstream_statistics.FIT_PROJECTION_COLUMNS,
        )
        for row in ledger_rows:
            ledger_hasher.update(row)
            projection_hasher.update({
                name: row[name]
                for name, _kind, _nullable in
                stress_downstream_statistics.FIT_PROJECTION_COLUMNS
            })
        index_hasher = _LogicalHasher(
            "task-12-downstream-fit-state-index-logical-v1", _FIT_INDEX_COLUMNS
        )
        for row in index_rows:
            index_hasher.update(row)
        pack_bytes, pack_sha = _file_pin(pack_path)
        content_digest = _fit_authority_digest(
            outcome_build_digest=expected_outcome,
            reference_content_digest=expected_reference,
            fit_ledger_row_count=ledger_hasher.row_count,
            fit_ledger_logical_digest=ledger_hasher.hexdigest(),
            fit_ledger_projection_digest=projection_hasher.hexdigest(),
            fit_state_index_row_count=index_hasher.row_count,
            fit_state_index_logical_digest=index_hasher.hexdigest(),
            fit_state_record_count=len(index_rows),
            fit_state_pack_byte_count=pack_bytes,
            fit_state_pack_sha256=pack_sha,
        )
        seal = DownstreamFitStatePackSeal(
            root=output_dir, outcome_build_digest=expected_outcome,
            fit_ledger_row_count=ledger_hasher.row_count,
            fit_ledger_logical_digest=ledger_hasher.hexdigest(),
            fit_ledger_projection_digest=projection_hasher.hexdigest(),
            fit_state_index_row_count=index_hasher.row_count,
            fit_state_index_logical_digest=index_hasher.hexdigest(),
            fit_state_record_count=len(index_rows),
            fit_state_pack_byte_count=pack_bytes,
            fit_state_pack_sha256=pack_sha, content_digest=content_digest,
        )
        loaded = load_downstream_fit_state_pack(
            output_dir, expected_content_digest=content_digest
        )
        if loaded.seal != seal:
            raise ValueError("reopened fit-pack seal changed")
        return seal
    except BaseException:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        raise


def _strict_json(data: bytes) -> object:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("portable fit-state JSON is invalid") from error


def _model_from_payload(payload: object, fit_seq: int) -> outcome.FrozenFoldModel:
    required = {
        "schema_name", "fit_seq", "heldout_position", "fold", "held_out_subject",
        "arm_position", "arm", "model_position", "model_name", "target_position",
        "target", "declared_feature_names", "preprocessor_state",
        "model_state_format", "model_state_base64", "preprocessor_digest",
        "model_digest", "fit_content_digest", "outcome_build_digest",
        "reference_content_digest", "record_content_digest",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("fit-state payload topology is invalid")
    if payload["schema_name"] != "task-12-downstream-fit-state-record-v1":
        raise ValueError("fit-state payload schema mismatch")
    if payload["fit_seq"] != fit_seq:
        raise ValueError("fit-state sequence is not contiguous")
    digest = payload["record_content_digest"]
    _require_digest(digest, "fit-state record digest")
    core = dict(payload)
    del core["record_content_digest"]
    if digest != _sha(["task-12-downstream-fit-state-record-v1", core]):
        raise ValueError("fit-state record digest mismatch")
    declared = payload["declared_feature_names"]
    if type(declared) is not list or any(type(value) is not str for value in declared):
        raise TypeError("fit-state declared features are invalid")
    state_format = payload["model_state_format"]
    if type(state_format) is not str or state_format not in _ALLOWED_STATE_FORMATS:
        raise ValueError("fit-state model format is unsupported")
    encoded = payload["model_state_base64"]
    if type(encoded) is not str:
        raise TypeError("fit-state model bytes are not strict base64")
    try:
        model_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("fit-state model bytes are not strict base64") from error
    if base64.b64encode(model_bytes).decode("ascii") != encoded:
        raise ValueError("fit-state model bytes are not canonical base64")
    model = outcome.FrozenFoldModel(
        fold=_require_int(payload["fold"], "model fold"),
        held_out_subject=payload["held_out_subject"], arm=payload["arm"],
        model_name=payload["model_name"], target=payload["target"],
        declared_feature_names=tuple(declared),
        preprocessor_state=_preprocessor_from_document(payload["preprocessor_state"]),
        model_state_format=state_format, model_state_bytes=model_bytes,
        preprocessor_digest=_require_digest(
            payload["preprocessor_digest"], "model preprocessor digest"
        ),
        model_digest=_require_digest(payload["model_digest"], "model digest"),
        content_digest=_require_digest(payload["fit_content_digest"], "fit digest"),
    )
    outcome.validate_frozen_fold_model(
        model, expected_content_digest=model.content_digest
    )
    return model


def load_downstream_fit_state_pack(
    root: Path,
    *,
    expected_content_digest: str,
) -> DownstreamLoadedFitStates:
    """Reopen, cross-check, reconstruct, and validate all 160 portable states."""
    expected = _require_digest(expected_content_digest, "expected fit authority digest")
    if not isinstance(root, Path) or not root.is_dir():
        raise ValueError("fit-pack root must be an existing directory")
    if {path.name for path in root.iterdir()} != {
        "fit_ledger.arrow", "fit_state_index.arrow", "fit_states.pack"
    }:
        raise ValueError("fit-pack file set is not exact")
    ledger_table = _read_ipc_table(root / "fit_ledger.arrow", _FIT_LEDGER_SCHEMA)
    index_table = _read_ipc_table(root / "fit_state_index.arrow", _FIT_INDEX_SCHEMA)
    ledger_rows = ledger_table.to_pylist()
    index_rows = index_table.to_pylist()
    if len(ledger_rows) != 160 or len(index_rows) != 160:
        raise ValueError("fit-pack topology is not exactly 160")
    pack_path = root / "fit_states.pack"
    pack_data = pack_path.read_bytes()
    if not pack_data.startswith(_PACK_HEADER):
        raise ValueError("fit-state pack header mismatch")
    cursor = len(_PACK_HEADER)
    models: list[outcome.FrozenFoldModel] = []
    outcome_digest: str | None = None
    reference_digest: str | None = None
    for fit_seq, (index_row, ledger_row) in enumerate(
        zip(index_rows, ledger_rows, strict=True)
    ):
        if index_row["fit_seq"] != fit_seq or ledger_row["fit_seq"] != fit_seq:
            raise ValueError("fit state/index/ledger order mismatch")
        if index_row["byte_offset"] != cursor or cursor + 8 > len(pack_data):
            raise ValueError("fit-state index offset has a gap")
        payload_length = struct.unpack(">Q", pack_data[cursor:cursor + 8])[0]
        end = cursor + 8 + payload_length
        if end > len(pack_data) or index_row["byte_count"] != 8 + payload_length:
            raise ValueError("fit-state index byte count mismatch")
        payload_bytes = pack_data[cursor + 8:end]
        if hashlib.sha256(payload_bytes).hexdigest() != index_row["record_sha256"]:
            raise ValueError("fit-state physical record digest mismatch")
        payload = _strict_json(payload_bytes)
        model = _model_from_payload(payload, fit_seq)
        models.append(model)
        current_outcome = _require_digest(
            payload["outcome_build_digest"], "record outcome build digest"
        )
        current_reference = _require_digest(
            payload["reference_content_digest"], "record reference digest"
        )
        outcome_digest = current_outcome if outcome_digest is None else outcome_digest
        reference_digest = current_reference if reference_digest is None else reference_digest
        if current_outcome != outcome_digest or current_reference != reference_digest:
            raise ValueError("fit-state records crosswire authority roots")
        if (
            payload["record_content_digest"] != index_row["record_content_digest"]
            or payload["record_content_digest"] != ledger_row["fit_state_record_digest"]
            or model.content_digest != index_row["fit_content_digest"]
            or model.content_digest != ledger_row["fit_content_digest"]
            or model.preprocessor_digest != ledger_row["preprocessor_digest"]
            or model.model_digest != ledger_row["model_digest"]
            or len(model.model_state_bytes) != ledger_row["model_state_byte_count"]
            or hashlib.sha256(model.model_state_bytes).hexdigest()
                != ledger_row["model_state_sha256"]
            or _canonical(_preprocessor_document(model.preprocessor_state)).decode("utf-8")
                != ledger_row["preprocessor_state_json"]
            or _canonical(list(model.declared_feature_names)).decode("utf-8")
                != ledger_row["declared_feature_names_json"]
        ):
            raise ValueError("fit-state nested authority crosswire")
        rebuilt_ledger_digest = _row_digest(
            "task-12-downstream-fit-ledger-row-v1", ledger_row,
            _FIT_LEDGER_DIGEST_NAMES,
        )
        if rebuilt_ledger_digest != ledger_row["fit_ledger_record_digest"]:
            raise ValueError("fit ledger row digest mismatch")
        cursor = end
    if cursor != len(pack_data) or outcome_digest is None or reference_digest is None:
        raise ValueError("fit-state pack has trailing bytes or no records")
    ledger_hasher = _LogicalHasher(
        "task-12-downstream-fit-ledger-logical-v1", _FIT_LEDGER_COLUMNS
    )
    projection_hasher = _LogicalHasher(
        "task-12-downstream-fit-ledger-projection-digest-v1",
        stress_downstream_statistics.FIT_PROJECTION_COLUMNS,
    )
    for row in ledger_rows:
        ledger_hasher.update(row)
        projection_hasher.update({
            name: row[name]
            for name, _kind, _nullable in
            stress_downstream_statistics.FIT_PROJECTION_COLUMNS
        })
    index_hasher = _LogicalHasher(
        "task-12-downstream-fit-state-index-logical-v1", _FIT_INDEX_COLUMNS
    )
    for row in index_rows:
        index_hasher.update(row)
    pack_bytes = len(pack_data)
    pack_sha = hashlib.sha256(pack_data).hexdigest()
    content_digest = _fit_authority_digest(
        outcome_build_digest=outcome_digest,
        reference_content_digest=reference_digest,
        fit_ledger_row_count=ledger_hasher.row_count,
        fit_ledger_logical_digest=ledger_hasher.hexdigest(),
        fit_ledger_projection_digest=projection_hasher.hexdigest(),
        fit_state_index_row_count=index_hasher.row_count,
        fit_state_index_logical_digest=index_hasher.hexdigest(),
        fit_state_record_count=len(models), fit_state_pack_byte_count=pack_bytes,
        fit_state_pack_sha256=pack_sha,
    )
    if content_digest != expected:
        raise ValueError("independently retained fit authority digest mismatch")
    seal = DownstreamFitStatePackSeal(
        root=root, outcome_build_digest=outcome_digest,
        fit_ledger_row_count=ledger_hasher.row_count,
        fit_ledger_logical_digest=ledger_hasher.hexdigest(),
        fit_ledger_projection_digest=projection_hasher.hexdigest(),
        fit_state_index_row_count=index_hasher.row_count,
        fit_state_index_logical_digest=index_hasher.hexdigest(),
        fit_state_record_count=len(models), fit_state_pack_byte_count=pack_bytes,
        fit_state_pack_sha256=pack_sha, content_digest=content_digest,
    )
    ledger = ledger_table.to_pandas().reset_index(drop=True)
    loaded_digest = _sha([
        "task-12-downstream-loaded-fit-states-v1", content_digest,
        [model.content_digest for model in models],
    ])
    return DownstreamLoadedFitStates(
        seal=seal, ledger=ledger, models=tuple(models),
        content_digest=loaded_digest,
    )


def _observed_prediction_row(
    base: Mapping[str, object],
    patch: stress_downstream.DownstreamFeaturePatch,
    *,
    row_seq: int,
    condition_position: int,
    reference_probability: float,
    perturbed_probability: float | None,
    label: int | None,
) -> dict[str, object]:
    reference_probability = float(reference_probability)
    if not math.isfinite(reference_probability) or not 0.0 <= reference_probability <= 1.0:
        raise ValueError("reference probability is invalid")
    reference_decision = reference_probability >= 0.5
    status = patch.replay_status
    if status == "finite":
        if perturbed_probability is None:
            raise ValueError("finite patch lacks a prediction")
        perturbed_probability = float(perturbed_probability)
        change = float(abs(perturbed_probability - reference_probability))
        perturbed_decision: bool | None = perturbed_probability >= 0.5
        flip: bool | None = perturbed_decision != reference_decision
        burden: float | None = change
        applicable = coverage_eligible = True
    elif status in ("abstained_no_event_support", "abstained_insufficient_support"):
        perturbed_probability = None
        perturbed_decision = change = flip = None
        burden = 1.0
        applicable = coverage_eligible = True
    elif status == "structurally_unavailable":
        perturbed_probability = None
        perturbed_decision = change = flip = burden = None
        applicable = coverage_eligible = False
    else:
        raise ValueError("integrity-failure patch cannot become an observed row")
    if label is not None and (type(label) is not int or label not in (0, 1)):
        raise TypeError("descriptive scoring label must be an exact binary int")
    row: dict[str, object] = {
        "row_seq": row_seq, **dict(base),
        "condition_position": condition_position, "condition": patch.condition,
        "pre_mask_eligible": True,
        "reference_status": "finite_baseline_admissible",
        "replay_status": status, "replay_reason": patch.reason,
        "applicable": applicable, "coverage_eligible": coverage_eligible,
        "reference_probability": reference_probability,
        "perturbed_probability": perturbed_probability,
        "reference_decision": reference_decision,
        "perturbed_decision": perturbed_decision,
        "absolute_probability_change": change, "decision_flip": flip,
        "instability_burden": burden, "label_available": label is not None,
        "label": label, "source_row_digest": patch.content_digest,
        "reference_context_digest": patch.reference_context_digest,
    }
    names = tuple(
        name for name, _kind, _nullable in
        stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS
        if name != "row_content_digest"
    )
    row["row_content_digest"] = _row_digest(
        "task-12-downstream-observed-prediction-row-v1", row, names
    )
    return row


def _scoring_lookup(scoring_labels: pd.DataFrame | None) -> dict[tuple[str, pd.Timestamp, pd.Timestamp], tuple[int, int, int, int]]:
    if scoring_labels is None:
        return {}
    if type(scoring_labels) is not pd.DataFrame:
        raise TypeError("scoring_labels must be an exact DataFrame or None")
    keys = ["subject_id", "lifelog_date", "sleep_date"]
    if not set(keys).union(outcome.TARGET_COLUMNS).issubset(scoring_labels.columns):
        raise ValueError("scoring labels lack required columns")
    frame = scoring_labels.loc[:, [*keys, *outcome.TARGET_COLUMNS]].copy()
    for name in ("lifelog_date", "sleep_date"):
        frame[name] = pd.to_datetime(frame[name], errors="raise")
    if frame.duplicated(keys).any():
        raise ValueError("scoring label keys are duplicated")
    lookup = {}
    for row in frame.to_dict("records"):
        numeric_values: list[int] = []
        for name in outcome.TARGET_COLUMNS:
            raw = row[name]
            if not isinstance(raw, (int, float, np.integer, np.floating)):
                raise TypeError("scoring labels must be numeric binary values")
            numeric = float(raw)
            if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
                raise ValueError("scoring labels must be finite exact binary values")
            numeric_values.append(int(numeric))
        values = tuple(numeric_values)
        lookup[(str(row["subject_id"]), pd.Timestamp(row["lifelog_date"]),
                pd.Timestamp(row["sleep_date"]))] = values
    return lookup


def _statistics_authority_digest(
    value: stress_downstream_statistics.DownstreamStatisticsInputAuthority,
) -> str:
    return _sha([
        "task-12-downstream-statistics-input-authority-v1",
        [list(entry) for entry in value.fold_roster], list(value.families),
        [list(entry) for entry in value.primitive_family_pairs], list(value.arms),
        list(value.models), list(value.targets), list(value.conditions),
        value.expected_primary_key_count, value.expected_primary_key_digest,
        value.max_primary_keys_per_group, value.observed_prediction_pin.content_digest,
        value.source_patch_projection_row_count, value.source_patch_projection_digest,
        value.fit_ledger_row_count, value.fit_state_index_row_count,
        value.fit_state_record_count, value.fit_ledger_projection_digest,
        value.fit_state_index_logical_digest, value.fit_state_pack_sha256,
        value.fit_authority_content_digest,
    ])


def _snapshot_arrow_tables(snapshot: DownstreamAdapterSnapshot) -> tuple[tuple[str, pa.Table], ...]:
    universe_schema = pa.schema([
        pa.field("participant_position", pa.int64(), False),
        pa.field("participant", pa.string(), False), pa.field("fold", pa.int64(), False),
        pa.field("held_out_subject", pa.string(), False),
        pa.field("family_position", pa.int64(), False), pa.field("family", pa.string(), False),
        pa.field("primitive_position", pa.int64(), False), pa.field("primitive", pa.string(), False),
        pa.field("sensor_day_id", pa.int64(), False), pa.field("draw", pa.int64(), False),
        pa.field("lifelog_date", pa.timestamp("ns"), False),
        pa.field("sleep_date", pa.timestamp("ns"), False),
        pa.field("elapsed_day_index", pa.int64(), False),
        pa.field("target_content_digest", pa.string(), False),
    ])
    ineligible_schema = pa.schema([
        pa.field("subject_id", pa.string(), False), pa.field("sensor_day_id", pa.int64(), False),
        pa.field("primitive", pa.string(), False), pa.field("draw", pa.int64(), False),
        pa.field("eligibility_status", pa.string(), False), pa.field("reason", pa.string(), False),
        pa.field("target_content_digest", pa.string(), False),
    ])
    window_schema = pa.schema([
        *ineligible_schema,
        pa.field("elapsed_day_index", pa.int64(), False),
        pa.field("downstream_exclusion_reason", pa.string(), False),
    ])
    # Reorder the two appended fields to the frozen nine-column order.
    window_records = snapshot.window_exclusion_ledger.to_dict("records")
    window_table = pa.Table.from_pylist(window_records, schema=pa.schema([
        pa.field("subject_id", pa.string(), False), pa.field("sensor_day_id", pa.int64(), False),
        pa.field("primitive", pa.string(), False), pa.field("draw", pa.int64(), False),
        pa.field("eligibility_status", pa.string(), False), pa.field("reason", pa.string(), False),
        pa.field("elapsed_day_index", pa.int64(), False),
        pa.field("downstream_exclusion_reason", pa.string(), False),
        pa.field("target_content_digest", pa.string(), False),
    ]))
    contiguous_schema = pa.schema([
        pa.field(name, (
            pa.int64() if name in {"row_seq", "sensor_day_id", "draw"}
            else pa.bool_() if name == "retained" else pa.string()
        ), False)
        for name in _CONTIGUOUS_COLUMNS
    ])
    return (
        ("prediction_universe.arrow", pa.Table.from_pylist(
            snapshot.prediction_universe.to_dict("records"), schema=universe_schema)),
        ("ineligible_target_ledger.arrow", pa.Table.from_pylist(
            snapshot.ineligible_target_ledger.to_dict("records"), schema=ineligible_schema)),
        ("window_exclusion_ledger.arrow", window_table),
        ("contiguous_provenance.arrow", pa.Table.from_pylist(
            snapshot.contiguous_provenance.to_dict("records"), schema=contiguous_schema)),
    )


def run_downstream_snapshot(
    snapshot: DownstreamAdapterSnapshot,
    primitives: pd.DataFrame,
    training_labels: pd.DataFrame,
    scoring_labels: pd.DataFrame | None,
    output_dir: Path,
    *,
    expected_snapshot_digest: str,
    expected_primitive_digest: str,
    expected_outcome_build_digest: str,
    root_seed: int,
) -> DownstreamRunArtifacts:
    """Run one bounded synthetic snapshot and atomically hand off its artifacts."""
    expected_primitive = _require_digest(expected_primitive_digest, "expected primitive digest")
    expected_outcome = _require_digest(expected_outcome_build_digest, "expected outcome digest")
    validate_downstream_adapter_snapshot(
        snapshot, expected_content_digest=expected_snapshot_digest,
        expected_source_patch_projection_digest=snapshot.source_patch_projection.logical_digest,
    )
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise TypeError("output_dir must be an absolute pathlib Path")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    partial = output_dir.with_name(f"{output_dir.name}.{uuid.uuid4().hex}.partial")
    partial.mkdir()
    try:
        for name, table in _snapshot_arrow_tables(snapshot):
            _write_ipc_table(partial / name, table)
        training_arms = outcome.build_outcome_arms(
            primitives, training_labels,
            interval_boundary_policy="measurement_support",
        )
        outcome.validate_outcome_arms(
            training_arms, expected_content_digest=expected_outcome
        )
        manifest = training_arms.manifest
        if (
            type(manifest) is not pd.DataFrame
            or len(manifest) != 1
            or tuple(manifest.columns).count("primitive_digest") != 1
        ):
            raise ValueError("training arm primitive digest manifest is invalid")
        arm_primitive_digest = manifest.iloc[0]["primitive_digest"]
        _require_digest(arm_primitive_digest, "training arm primitive digest")
        if arm_primitive_digest != expected_primitive:
            raise ValueError("training arm primitive digest mismatch")
        for seal in snapshot.reference_tables.frame_manifest:
            reference_primitive_digest = seal.source_primitive_digest
            _require_digest(
                reference_primitive_digest, "reference frame primitive digest"
            )
            if reference_primitive_digest != expected_primitive:
                raise ValueError("reference frame primitive digest mismatch")
        fit_seal = write_downstream_fit_state_pack(
            training_arms, training_labels, snapshot.reference_tables, partial / "fit",
            expected_outcome_build_digest=expected_outcome,
            expected_reference_content_digest=snapshot.reference_content_digest,
            root_seed=root_seed,
        )
        loaded = load_downstream_fit_state_pack(
            fit_seal.root, expected_content_digest=fit_seal.content_digest
        )
        model_by_seq = {index: model for index, model in enumerate(loaded.models)}
        ledger_by_seq = {
            int(row["fit_seq"]): row for row in loaded.ledger.to_dict("records")
        }
        contexts = {(value.held_out_subject, value.sensor_day_id): value for value in snapshot.contexts}
        patches = {
            (value.held_out_subject, value.sensor_day_id, value.primitive, value.draw,
             value.condition, value.arm): value for value in snapshot.patches
        }
        labels = _scoring_lookup(scoring_labels)
        observed_path = partial / "observed_predictions.arrow"
        observed_hasher = _LogicalHasher(
            "task-12-downstream-observed-prediction-logical-v1",
            stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS,
        )
        row_seq = 0
        probability_cache: dict[tuple[int, int], float] = {}
        with observed_path.open("xb") as stream:
            with ipc.new_stream(
                stream, stress_downstream_statistics.OBSERVED_PREDICTION_SCHEMA
            ) as writer:
                for participant_position, (fold, participant) in enumerate(snapshot.fold_roster):
                    participant_rows: list[dict[str, object]] = []
                    for family_position, family in enumerate(_FAMILIES):
                        selected_rows = snapshot.prediction_universe.loc[
                            snapshot.prediction_universe["participant"].eq(participant)
                            & snapshot.prediction_universe["family"].eq(family)
                        ].sort_values(
                            ["primitive_position", "sensor_day_id", "draw"], kind="stable"
                        )
                        for arm_position, arm in enumerate(outcome.DOWNSTREAM_ARM_NAMES):
                            for model_position, model_name in enumerate(outcome.MODEL_NAMES):
                                for target_position, target in enumerate(outcome.TARGET_COLUMNS):
                                    fit_seq = (((participant_position * 2 + arm_position) * 2
                                                + model_position) * 4 + target_position)
                                    model = model_by_seq[fit_seq]
                                    ledger = ledger_by_seq[fit_seq]
                                    for selected in selected_rows.to_dict("records"):
                                        sensor_day_id = int(selected["sensor_day_id"])
                                        primitive = str(selected["primitive"])
                                        draw = int(selected["draw"])
                                        context = contexts[(participant, sensor_day_id)]
                                        reference = stress_downstream.reference_feature_frame(context, arm)
                                        cache_key = (fit_seq, sensor_day_id)
                                        if cache_key not in probability_cache:
                                            probability_cache[cache_key] = float(
                                                outcome.predict_frozen_fold_model(
                                                    model, reference,
                                                    expected_content_digest=model.content_digest,
                                                )[0]
                                            )
                                        reference_probability = probability_cache[cache_key]
                                        score_key = (
                                            participant, pd.Timestamp(selected["lifelog_date"]),
                                            pd.Timestamp(selected["sleep_date"]),
                                        )
                                        score_values = labels.get(score_key)
                                        label = None if score_values is None else int(score_values[target_position])
                                        base = {
                                            "participant_position": participant_position,
                                            "participant": participant, "fold": fold,
                                            "held_out_subject": participant,
                                            "family_position": family_position, "family": family,
                                            "primitive_position": int(selected["primitive_position"]),
                                            "primitive": primitive, "sensor_day_id": sensor_day_id,
                                            "draw": draw, "arm_position": arm_position, "arm": arm,
                                            "model_position": model_position, "model_name": model_name,
                                            "target_position": target_position, "target": target,
                                            "fit_seq": fit_seq,
                                            "fit_ledger_record_digest": str(ledger["fit_ledger_record_digest"]),
                                            "preprocessor_digest": str(ledger["preprocessor_digest"]),
                                            "model_digest": str(ledger["model_digest"]),
                                            "fit_content_digest": str(ledger["fit_content_digest"]),
                                        }
                                        for condition_position, condition in enumerate(_CONDITIONS):
                                            patch = patches[(participant, sensor_day_id, primitive, draw,
                                                             condition, arm)]
                                            patched = stress_downstream.apply_downstream_feature_patch(
                                                reference, patch,
                                                expected_context_digest=context.content_digest,
                                                expected_perturbation_digest=patch.perturbation_digest,
                                            )
                                            perturbed = None if patched is None else float(
                                                outcome.predict_frozen_fold_model(
                                                    model, patched,
                                                    expected_content_digest=model.content_digest,
                                                )[0]
                                            )
                                            observed = _observed_prediction_row(
                                                base, patch, row_seq=row_seq,
                                                condition_position=condition_position,
                                                reference_probability=reference_probability,
                                                perturbed_probability=perturbed, label=label,
                                            )
                                            participant_rows.append(observed)
                                            observed_hasher.update(observed)
                                            row_seq += 1
                    writer.write_batch(pa.RecordBatch.from_pylist(
                        participant_rows,
                        schema=stress_downstream_statistics.OBSERVED_PREDICTION_SCHEMA,
                    ))
        observed_bytes, observed_sha = _file_pin(observed_path)
        pin_values = {
            "schema_name": "task-12-downstream-observed-prediction-input-v1",
            "row_count": observed_hasher.row_count,
            "columns": stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS,
            "byte_count": observed_bytes, "physical_sha256": observed_sha,
            "logical_digest": observed_hasher.hexdigest(),
        }
        unsealed_pin = stress_downstream_statistics.ObservedPredictionInputPin(
            **pin_values, content_digest="0" * 64
        )
        pin = stress_downstream_statistics.ObservedPredictionInputPin(
            **pin_values, content_digest=_sha([
                "task-12-downstream-observed-prediction-input-pin-v1",
                unsealed_pin.schema_name, unsealed_pin.row_count,
                [list(value) for value in unsealed_pin.columns], unsealed_pin.byte_count,
                unsealed_pin.physical_sha256, unsealed_pin.logical_digest,
            ])
        )
        authority_values = {
            "fold_roster": snapshot.fold_roster, "families": _FAMILIES,
            "primitive_family_pairs": snapshot.primitive_family_pairs,
            "arms": outcome.DOWNSTREAM_ARM_NAMES, "models": outcome.MODEL_NAMES,
            "targets": outcome.TARGET_COLUMNS, "conditions": _CONDITIONS,
            "expected_primary_key_count": snapshot.expected_primary_key_count,
            "expected_primary_key_digest": snapshot.expected_primary_key_digest,
            "max_primary_keys_per_group": snapshot.max_primary_keys_per_group,
            "observed_prediction_pin": pin,
            "source_patch_projection_row_count": snapshot.source_patch_projection.row_count,
            "source_patch_projection_digest": snapshot.source_patch_projection.logical_digest,
            "fit_ledger_row_count": fit_seal.fit_ledger_row_count,
            "fit_state_index_row_count": fit_seal.fit_state_index_row_count,
            "fit_state_record_count": fit_seal.fit_state_record_count,
            "fit_ledger_projection_digest": fit_seal.fit_ledger_projection_digest,
            "fit_state_index_logical_digest": fit_seal.fit_state_index_logical_digest,
            "fit_state_pack_sha256": fit_seal.fit_state_pack_sha256,
            "fit_authority_content_digest": fit_seal.content_digest,
        }
        unsealed_authority = stress_downstream_statistics.DownstreamStatisticsInputAuthority(
            **authority_values, content_digest="0" * 64
        )
        authority = stress_downstream_statistics.DownstreamStatisticsInputAuthority(
            **authority_values, content_digest=_statistics_authority_digest(unsealed_authority)
        )
        def batches() -> Iterable[pa.RecordBatch]:
            with observed_path.open("rb") as input_stream:
                yield from ipc.open_stream(input_stream)
        handoff_dir = partial / "statistics-handoff"
        handoff = stress_downstream_statistics.analyze_downstream_statistics(
            batches(), authority, handoff_dir,
            expected_authority_digest=authority.content_digest,
        )
        statistics_manifest = stress_downstream_statistics.persist_downstream_statistics(
            handoff, partial / "statistics",
            expected_handoff_digest=handoff.content_digest,
        )
        stress_downstream_statistics.verify_persisted_downstream_statistics(
            partial / "statistics", statistics_manifest,
            expected_manifest_digest=statistics_manifest.content_digest,
        )
        shutil.rmtree(handoff_dir)
        artifacts = []
        for path in sorted(
            (value for value in partial.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(partial).as_posix(),
        ):
            byte_count, sha256 = _file_pin(path)
            relative = path.relative_to(partial).as_posix()
            artifacts.append({
                "relative_path": relative, "byte_count": byte_count,
                "physical_sha256": sha256,
                "content_digest": _sha([
                    "task-12-downstream-run-artifact-seal-v1", relative,
                    byte_count, sha256,
                ]),
            })
        manifest_core = {
            "schema_name": "task-12-downstream-run-manifest-v1",
            "adapter_digest": snapshot.content_digest,
            "official_g2_authority_digest": snapshot.official_g2_authority_digest,
            "official_cell_evidence_projection_digest":
                snapshot.official_cell_evidence_projection_digest,
            "source_patch_projection_digest": snapshot.source_patch_projection.logical_digest,
            "outcome_build_digest": expected_outcome, "selected_target_count": snapshot.eligible_target_count,
            "reference_key_count": snapshot.reference_key_count,
            "expected_primary_key_count": snapshot.expected_primary_key_count,
            "observed_row_count": pin.row_count, "fit_authority_digest": fit_seal.content_digest,
            "observed_prediction_pin_digest": pin.content_digest,
            "statistics_manifest_digest": statistics_manifest.content_digest,
            "statistics_result_digest": statistics_manifest.result_digest,
            "artifacts": artifacts,
        }
        run_digest = _sha(["task-12-downstream-run-manifest-v1", manifest_core])
        manifest_bytes = _canonical({**manifest_core, "content_digest": run_digest}) + b"\n"
        (partial / "manifest.json").write_bytes(manifest_bytes)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if output_dir.exists():
            raise FileExistsError(output_dir)
        partial.rename(output_dir)
        return DownstreamRunArtifacts(
            output_dir=output_dir, adapter_digest=snapshot.content_digest,
            fit_authority_digest=fit_seal.content_digest, observed_prediction_pin=pin,
            statistics_manifest=statistics_manifest, manifest_sha256=manifest_sha,
            content_digest=run_digest,
        )
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
