"""Result-blind authorities for the frozen downstream stability analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import outcome, personalization


_SHA256 = re.compile(r"[0-9a-f]{64}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_ARTIFACT_NAMES = (
    "target_ledger",
    "official_current_crosslink",
    "mask_ledger",
    "deletion_audit",
    "cell_replays",
    "primary_comparison",
    "participant_deltas",
    "denominators",
    "family_inference",
    "dose_response_participant",
    "dose_response_family",
    "decision",
)
_RETAINED_NAMES = _ARTIFACT_NAMES[:5]
_SCENARIOS = ("scattered_random_20pct", "event_boundary_20pct")
_PRIMITIVE_ORDER = (
    "step_load_24h",
    "screen_load_24h",
    "phone_activity_load_24h",
    "usage_load_24h",
    "mobile_light_exposure_24h",
    "wearable_light_exposure_24h",
    "screen_disengagement_p90",
    "usage_disengagement_p90",
)
_LABEL_PATH = "ETRI_Human_AI/data/raw/data_vvs/ch2026_metrics_train.csv"
_LABEL_COLUMNS = (
    "subject_id",
    "sleep_date",
    "lifelog_date",
    "Q1",
    "Q2",
    "Q3",
    "S1",
    "S2",
    "S3",
    "S4",
)
_DOWNSTREAM_PATCH_ARMS = outcome.DOWNSTREAM_ARM_NAMES
_DOWNSTREAM_PATCH_CONDITIONS = (
    "scattered_random_20pct",
    "contiguous_20pct",
    "event_boundary_20pct",
)
_DOWNSTREAM_FAMILY_PRIMITIVES = (
    ("event_count", ("step_load_24h",)),
    ("state_ratio", ("screen_load_24h", "phone_activity_load_24h")),
    (
        "intensity",
        (
            "usage_load_24h",
            "mobile_light_exposure_24h",
            "wearable_light_exposure_24h",
        ),
    ),
    ("timing", ("screen_disengagement_p90", "usage_disengagement_p90")),
)
_DOWNSTREAM_CONTEXT_PRIMITIVES = outcome.MATCHED_SOURCE_PRIMITIVES
_DOWNSTREAM_SEMANTIC_DESCENDANTS = (
    ("step_load_24h", ("physical_activity_score", "routine_deviation")),
    ("screen_load_24h", ("digital_engagement_score", "routine_deviation")),
    (
        "phone_activity_load_24h",
        ("physical_activity_score", "routine_deviation"),
    ),
    ("usage_load_24h", ("digital_engagement_score", "routine_deviation")),
    (
        "mobile_light_exposure_24h",
        ("light_exposure_score", "routine_deviation"),
    ),
    (
        "wearable_light_exposure_24h",
        ("light_exposure_score", "routine_deviation"),
    ),
    (
        "screen_disengagement_p90",
        (
            "digital_disengagement_time",
            "winddown_center",
            "winddown_spread",
            "routine_deviation",
        ),
    ),
    (
        "usage_disengagement_p90",
        (
            "digital_disengagement_time",
            "winddown_center",
            "winddown_spread",
            "routine_deviation",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class ReplayArtifactPin:
    name: str
    relative_path: str
    row_count: int
    columns: tuple[tuple[str, str], ...]
    byte_count: int
    physical_sha256: str
    logical_digest: str
    retained_snapshot: bool
    content_digest: str


@dataclass(frozen=True, slots=True)
class Task12ReplayPins:
    output_relative_path: str
    manifest_relative_path: str
    manifest_byte_count: int
    manifest_sha256: str
    release_relative_path: str
    task12_release_sha256: str
    consumption_relative_path: str
    task12_consumption_sha256: str
    draws: tuple[int, ...]
    scenarios: tuple[str, ...]
    selected_cell_count: int
    artifacts: tuple[ReplayArtifactPin, ...]
    content_digest: str


@dataclass(frozen=True, slots=True)
class ReplayArtifactSnapshot:
    pin: ReplayArtifactPin
    snapshot_bytes: bytes
    content_digest: str


@dataclass(frozen=True, slots=True)
class Task12CanonicalReplayAuthority:
    output_dir: Path
    manifest_path: Path
    manifest_bytes: bytes
    pins_digest: str
    retained_artifacts: tuple[ReplayArtifactSnapshot, ...]
    result_content_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class TrainingLabelPin:
    relative_path: str
    byte_count: int
    physical_sha256: str
    row_count: int
    columns: tuple[str, ...]
    participant_count: int
    content_digest: str


@dataclass(frozen=True, slots=True)
class TrainingLabelRow:
    subject_id: str
    sleep_date: str
    lifelog_date: str
    s1: int
    s2: int
    s3: int
    s4: int
    content_digest: str


@dataclass(frozen=True, slots=True)
class TrainingLabelAuthority:
    path: Path
    snapshot_bytes: bytes
    pin_digest: str
    rows: tuple[TrainingLabelRow, ...]
    key_digest: str
    target_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamPrimitiveBaseline:
    primitive: str
    method: str
    k: int
    donor_subject: None
    center: float
    scale: float
    lambda_days: float
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamReferenceArmRow:
    arm: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamReferenceContext:
    fold: int
    held_out_subject: str
    sensor_day_id: int
    primitive_basis: tuple[tuple[str, float, float, float], ...]
    baselines: tuple[DownstreamPrimitiveBaseline, ...]
    arm_rows: tuple[DownstreamReferenceArmRow, ...]
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamPerturbation:
    fold: int
    held_out_subject: str
    sensor_day_id: int
    primitive: str
    draw: int
    condition: str
    replay_status: str
    reason: str
    masked_value: float
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamFeaturePatch:
    fold: int
    held_out_subject: str
    sensor_day_id: int
    primitive: str
    draw: int
    condition: str
    arm: str
    replay_status: str
    reason: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    reference_context_digest: str
    reference_arm_row_digest: str
    perturbation_digest: str
    content_digest: str


def _fail(message: str) -> None:
    raise ValueError(f"DATA-INVALID: {message}")


def _compact_sha(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be an exact lowercase SHA-256")
    return value


def _require_exact_int(value: object, label: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0) or (not positive and value < 0):
        _fail(f"{label} must be an exact built-in integer")
    return value


def _artifact_pin_digest(pin: ReplayArtifactPin) -> str:
    return _compact_sha(
        [
            "task-12-replay-artifact-pin-v1",
            pin.name,
            pin.relative_path,
            pin.row_count,
            [list(item) for item in pin.columns],
            pin.byte_count,
            pin.physical_sha256,
            pin.logical_digest,
            pin.retained_snapshot,
        ]
    )


def _pins_digest(pins: Task12ReplayPins) -> str:
    return _compact_sha(
        [
            "task-12-replay-pins-v1",
            pins.output_relative_path,
            pins.manifest_relative_path,
            pins.manifest_byte_count,
            pins.manifest_sha256,
            pins.release_relative_path,
            pins.task12_release_sha256,
            pins.consumption_relative_path,
            pins.task12_consumption_sha256,
            list(pins.draws),
            list(pins.scenarios),
            pins.selected_cell_count,
            [item.content_digest for item in pins.artifacts],
        ]
    )


def _label_pin_digest(pin: TrainingLabelPin) -> str:
    return _compact_sha(
        [
            "task-12-training-label-pin-v1",
            pin.relative_path,
            pin.byte_count,
            pin.physical_sha256,
            pin.row_count,
            list(pin.columns),
            pin.participant_count,
        ]
    )


def _snapshot_digest(pin: ReplayArtifactPin, raw: bytes) -> str:
    return _compact_sha(
        [
            "task-12-replay-artifact-snapshot-v1",
            pin.content_digest,
            hashlib.sha256(raw).hexdigest(),
            pin.logical_digest,
        ]
    )


def _authority_digest(
    output_dir: Path,
    manifest_path: Path,
    manifest_bytes: bytes,
    pins_digest: str,
    retained: tuple[ReplayArtifactSnapshot, ...],
    result_content_digest: str,
) -> str:
    return _compact_sha(
        [
            "task-12-canonical-replay-authority-v1",
            str(output_dir),
            str(manifest_path),
            hashlib.sha256(manifest_bytes).hexdigest(),
            pins_digest,
            [item.content_digest for item in retained],
            result_content_digest,
        ]
    )


def _label_row_digest(row: TrainingLabelRow) -> str:
    return _compact_sha(
        [
            "task-12-training-label-row-v1",
            row.subject_id,
            row.sleep_date,
            row.lifelog_date,
            row.s1,
            row.s2,
            row.s3,
            row.s4,
        ]
    )


def _label_authority_digest(
    path: Path,
    raw: bytes,
    pin_digest: str,
    rows: tuple[TrainingLabelRow, ...],
    key_digest: str,
    target_digest: str,
) -> str:
    return _compact_sha(
        [
            "task-12-training-label-authority-v1",
            str(path),
            hashlib.sha256(raw).hexdigest(),
            pin_digest,
            [row.content_digest for row in rows],
            key_digest,
            target_digest,
        ]
    )


def _strict_relative_path(value: object, label: str, *, basename: bool = False) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value:
        _fail(f"{label} must be an exact POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in ("", ".", "..") for part in path.parts):
        _fail(f"{label} is ambiguous or traverses its root")
    if basename and (len(path.parts) != 1 or path.name != value):
        _fail(f"{label} must be a basename")
    return path


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse)


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _literal_path(root: Path, relative: PurePosixPath, label: str) -> Path:
    if not isinstance(root, Path) or not root.is_absolute():
        _fail(f"{label} root must be an absolute built-in Path")
    root_resolved = root.resolve(strict=True)
    if root_resolved != root:
        _fail(f"{label} root is not literal")
    current = root
    for part in relative.parts:
        if not current.is_dir():
            _fail(f"{label} parent is missing")
        exact_names = {entry.name for entry in current.iterdir()}
        if part not in exact_names:
            _fail(f"{label} case or component mismatch")
        current = current / part
        info = os.lstat(current)
        if _is_link_or_reparse(info):
            _fail(f"{label} contains a symlink, junction, or reparse component")
    if current.resolve(strict=True) != current:
        _fail(f"{label} resolves to an alternate path")
    return current


def _read_stable_file(path: Path, label: str) -> tuple[bytes, tuple[int, int, int, int]]:
    before = os.lstat(path)
    if _is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
        _fail(f"{label} is not a literal regular file")
    if before.st_nlink != 1:
        _fail(f"{label} is hardlinked")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        opened = os.fstat(fd)
        if _stat_identity(opened) != _stat_identity(before):
            _fail(f"{label} changed between lstat and open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after = os.lstat(path)
    if (
        _stat_identity(after_fd) != _stat_identity(opened)
        or _stat_identity(after) != _stat_identity(opened)
        or _is_link_or_reparse(after)
        or after.st_nlink != 1
    ):
        _fail(f"{label} changed while being read")
    return b"".join(chunks), _stat_identity(opened)


def _verify_stable_file(
    path: Path, expected: tuple[int, int, int, int], label: str
) -> None:
    current = os.lstat(path)
    if (
        _stat_identity(current) != expected
        or _is_link_or_reparse(current)
        or not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
    ):
        _fail(f"{label} changed during parsing")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes, label: str) -> object:
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw:
        _fail(f"{label} has a BOM or NUL")
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(f"{label} is not strict JSON: {exc}")


def _require_exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        _fail(f"{label} keys are not exact")
    return value


def _validate_artifact_pin(pin: ReplayArtifactPin, expected_name: str) -> None:
    if type(pin) is not ReplayArtifactPin or type(pin.name) is not str or pin.name != expected_name:
        _fail("artifact pin identity is forged")
    relative = _strict_relative_path(pin.relative_path, f"{pin.name} path", basename=True)
    if relative.name != f"{pin.name}.parquet":
        _fail(f"{pin.name} path is not canonical")
    _require_exact_int(pin.row_count, f"{pin.name} row count")
    _require_exact_int(pin.byte_count, f"{pin.name} byte count", positive=True)
    if type(pin.columns) is not tuple or not pin.columns:
        _fail(f"{pin.name} schema is not an exact tuple")
    seen: set[str] = set()
    for item in pin.columns:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not str
            or item[1] not in ("string", "bool", "int64", "double")
            or item[0] in seen
        ):
            _fail(f"{pin.name} schema is forged or unsupported")
        seen.add(item[0])
    _require_sha256(pin.physical_sha256, f"{pin.name} physical digest")
    if type(pin.retained_snapshot) is not bool:
        _fail(f"{pin.name} retained flag is forged")
    should_retain = pin.name in _RETAINED_NAMES
    if pin.retained_snapshot is not should_retain:
        _fail(f"{pin.name} retained flag is crosswired")
    if should_retain:
        _require_sha256(pin.logical_digest, f"{pin.name} logical digest")
    elif type(pin.logical_digest) is not str or pin.logical_digest != "":
        _fail(f"{pin.name} has an unverified logical digest")
    _require_sha256(pin.content_digest, f"{pin.name} pin digest")
    if pin.content_digest != _artifact_pin_digest(pin):
        _fail(f"{pin.name} pin is not self-consistent")


def _validate_replay_pins(pins: Task12ReplayPins, expected_pins_digest: str) -> None:
    if type(pins) is not Task12ReplayPins:
        _fail("replay pins type is forged")
    expected = _require_sha256(expected_pins_digest, "externally expected pins digest")
    _require_sha256(pins.content_digest, "replay pins digest")
    if pins.content_digest != expected:
        _fail("replay pins do not match the independently held digest")
    if type(pins.draws) is not tuple or not pins.draws:
        _fail("draws must be a nonempty tuple")
    if any(type(value) is not int or value < 0 for value in pins.draws):
        _fail("draws must contain exact built-in nonnegative integers")
    if pins.draws != tuple(range(pins.draws[0], pins.draws[0] + len(pins.draws))):
        _fail("draws are not contiguous")
    if (
        type(pins.scenarios) is not tuple
        or any(type(value) is not str for value in pins.scenarios)
        or pins.scenarios != _SCENARIOS
    ):
        _fail("scenarios are not the frozen ordered pair")
    _require_exact_int(pins.selected_cell_count, "selected cell count", positive=True)
    if type(pins.artifacts) is not tuple or len(pins.artifacts) != len(_ARTIFACT_NAMES):
        _fail("artifact pins do not contain exactly twelve entries")
    for pin, name in zip(pins.artifacts, _ARTIFACT_NAMES, strict=True):
        _validate_artifact_pin(pin, name)
    base = pins.selected_cell_count * len(pins.draws)
    expected_counts = {
        "target_ledger": base,
        "official_current_crosslink": base,
        "mask_ledger": base * len(pins.scenarios),
        "deletion_audit": base * len(pins.scenarios),
        "cell_replays": base * len(pins.scenarios),
    }
    for pin in pins.artifacts[:5]:
        if pin.row_count != expected_counts[pin.name]:
            _fail(f"{pin.name} row count violates frozen topology")
    output = _strict_relative_path(pins.output_relative_path, "output path")
    if output.parent != PurePosixPath(".superpowers/sdd") or not output.name:
        _fail("output path is outside the frozen SDD directory")
    manifest = _strict_relative_path(pins.manifest_relative_path, "manifest path")
    if str(manifest) != f"{pins.output_relative_path}/manifest.json":
        _fail("manifest path is crosswired")
    if pins.release_relative_path != ".superpowers/sdd/task-12-canonical-release-v1.json":
        _fail("release path is not the frozen literal")
    if pins.consumption_relative_path != ".superpowers/sdd/task-12-canonical-consumed-v1.json":
        _fail("consumption path is not the frozen literal")
    _require_exact_int(pins.manifest_byte_count, "manifest byte count", positive=True)
    _require_sha256(pins.manifest_sha256, "manifest digest")
    _require_sha256(pins.task12_release_sha256, "release digest")
    _require_sha256(pins.task12_consumption_sha256, "consumption digest")
    if pins.content_digest != _pins_digest(pins):
        _fail("replay pins are not self-consistent")


def _validate_manifest(document: object, pins: Task12ReplayPins, release_sha: str) -> dict[str, Any]:
    manifest = _require_exact_keys(
        document,
        {
            "schema_name",
            "mode",
            "production_eligible",
            "decision_label",
            "authority_digest",
            "result_content_digest",
            "release_seal_sha256",
            "artifacts",
        },
        "manifest",
    )
    if (
        manifest["schema_name"] != "structured-missingness-etri-output-v1"
        or manifest["mode"] != "canonical"
        or manifest["production_eligible"] is not True
        or type(manifest["decision_label"]) is not str
    ):
        _fail("manifest root contract is not canonical")
    _require_sha256(manifest["authority_digest"], "manifest authority digest")
    _require_sha256(manifest["result_content_digest"], "manifest result digest")
    if manifest["release_seal_sha256"] != release_sha:
        _fail("manifest release seal is crosswired")
    artifacts = manifest["artifacts"]
    if type(artifacts) is not list or len(artifacts) != len(pins.artifacts):
        _fail("manifest must declare exactly twelve artifacts")
    for entry, pin in zip(artifacts, pins.artifacts, strict=True):
        item = _require_exact_keys(
            entry, {"name", "path", "rows", "bytes", "sha256"}, "manifest artifact"
        )
        if (
            type(item["name"]) is not str
            or item["name"] != pin.name
            or type(item["path"]) is not str
            or item["path"] != pin.relative_path
            or type(item["rows"]) is not int
            or item["rows"] != pin.row_count
            or type(item["bytes"]) is not int
            or item["bytes"] != pin.byte_count
            or item["sha256"] != pin.physical_sha256
        ):
            _fail(f"manifest metadata is crosswired for {pin.name}")
    return manifest


def _validate_predecessors(
    release: object,
    consumption: object,
    pins: Task12ReplayPins,
    release_sha: str,
) -> None:
    if type(release) is not dict or type(consumption) is not dict:
        _fail("predecessor documents must be JSON objects")
    output_name = PurePosixPath(pins.output_relative_path).name
    if (
        release.get("schema_name") != "task-12-canonical-release-v1"
        or release.get("status") != "CLEAN"
        or release.get("canonical_etri_run_authorized") is not True
        or release.get("canonical_output_name") != output_name
        or release.get("consumption_marker_path") != pins.consumption_relative_path
    ):
        _fail("Task 12 release is not the frozen clean authorization")
    if (
        consumption.get("schema_name") != "task-12-canonical-consumption-v1"
        or consumption.get("status") != "started_one_shot"
        or consumption.get("canonical_output_name") != output_name
        or consumption.get("release_seal_sha256") != release_sha
    ):
        _fail("Task 12 consumption marker is crosswired")


def _parse_parquet(raw: bytes, pin: ReplayArtifactPin) -> tuple[pa.Table, pd.DataFrame]:
    try:
        table = pq.read_table(pa.BufferReader(raw))
    except Exception as exc:
        _fail(f"{pin.name} is not readable Parquet: {exc}")
    if any(pa.types.is_large_string(field.type) for field in table.schema):
        canonical_schema = pa.schema(
            [
                pa.field(
                    field.name,
                    pa.string() if pa.types.is_large_string(field.type) else field.type,
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
                for field in table.schema
            ],
            metadata=table.schema.metadata,
        )
        table = table.cast(canonical_schema)
    schema = tuple((field.name, str(field.type)) for field in table.schema)
    if schema != pin.columns or table.num_rows != pin.row_count:
        _fail(f"{pin.name} schema or row count differs from its pin")
    allowed = (pa.string(), pa.bool_(), pa.int64(), pa.float64())
    if any(field.type not in allowed for field in table.schema):
        _fail(f"{pin.name} contains an unsupported Arrow type")
    logical = _parquet_logical_digest(table)
    if pin.retained_snapshot and logical != pin.logical_digest:
        _fail(f"{pin.name} logical digest differs from its pin")
    return table, table.to_pandas()


def _parquet_logical_digest(table: pa.Table) -> str:
    digest = hashlib.sha256()
    digest.update(b"task-12-downstream-parquet-logical-v1\n")
    schema = [[field.name, str(field.type)] for field in table.schema]
    digest.update(
        json.dumps(
            schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )
    for batch in table.to_batches(max_chunksize=4096):
        for row_index in range(batch.num_rows):
            tokens: list[list[object]] = []
            for column_index, field in enumerate(batch.schema):
                scalar = batch.column(column_index)[row_index]
                if not scalar.is_valid:
                    token: list[object] = ["null"]
                elif pa.types.is_string(field.type):
                    value = scalar.as_py()
                    if type(value) is not str:
                        _fail("Parquet string scalar is coercive")
                    token = ["s", value]
                elif pa.types.is_boolean(field.type):
                    value = scalar.as_py()
                    if type(value) is not bool:
                        _fail("Parquet boolean scalar is coercive")
                    token = ["b", value]
                elif pa.types.is_int64(field.type):
                    value = scalar.as_py()
                    if type(value) is not int:
                        _fail("Parquet int64 scalar is coercive")
                    token = ["i", str(value)]
                elif pa.types.is_float64(field.type):
                    value = scalar.as_py()
                    if type(value) is not float or math.isinf(value):
                        _fail("Parquet float64 scalar is coercive or infinite")
                    token = ["nan"] if math.isnan(value) else ["f", value.hex()]
                else:
                    _fail("Parquet contains an unsupported Arrow type")
                tokens.append(token)
            digest.update(
                json.dumps(
                    tokens, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ).encode("utf-8")
                + b"\n"
            )
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    if not set(required).issubset(set(frame.columns)):
        _fail(f"{name} lacks required topology columns")


def _key_tuple(row: pd.Series) -> tuple[str, int, str, int]:
    subject = row["subject_id"]
    sensor_day = row["sensor_day_id"]
    primitive = row["primitive"]
    draw = row["draw"]
    if type(subject) is not str or type(primitive) is not str:
        _fail("topology string key has a nonexact type")
    if type(sensor_day) is not int or type(draw) is not int:
        _fail("topology integer key has a nonexact type")
    return subject, sensor_day, primitive, draw


def _producer_key_order(key: tuple[str, int, str, int]) -> tuple[str, int, int, int]:
    subject, sensor_day, primitive, draw = key
    try:
        primitive_rank = _PRIMITIVE_ORDER.index(primitive)
    except ValueError:
        _fail("topology primitive is outside the frozen producer order")
    return subject, primitive_rank, sensor_day, draw


def _validate_topology(frames: dict[str, pd.DataFrame], pins: Task12ReplayPins | None = None) -> str:
    target = frames["target_ledger"]
    crosslink = frames["official_current_crosslink"]
    mask = frames["mask_ledger"]
    audit = frames["deletion_audit"]
    replay = frames["cell_replays"]
    base_columns = ("subject_id", "sensor_day_id", "primitive", "draw")
    _require_columns(target, (*base_columns, "target_deleted_count", "content_digest"), "target")
    _require_columns(
        crosslink,
        (
            *base_columns,
            "scenario",
            "official_deleted_count",
            "official_retained_count",
            "target_valid_deleted_count",
            "target_content_digest",
            "crosslink_content_digest",
        ),
        "crosslink",
    )
    _require_columns(
        mask,
        (
            *base_columns,
            "scenario",
            "mask_status",
            "target_deleted_count",
            "selected_record_count",
            "content_digest",
        ),
        "mask",
    )
    _require_columns(
        audit,
        (
            *base_columns,
            "scenario",
            "target_deleted_count",
            "deleted_count",
            "retained_count",
            "exact_match",
            "mask_digest",
        ),
        "audit",
    )
    _require_columns(
        replay,
        (
            *base_columns,
            "scenario",
            "target_deleted_count",
            "deleted_count",
            "retained_count",
            "mask_digest",
            "target_digest",
            "status",
            "reason",
            "masked_value",
            "standardized_absolute_error",
        ),
        "replay",
    )
    target_rows: dict[tuple[str, int, str, int], pd.Series] = {}
    for _, row in target.iterrows():
        key = _key_tuple(row)
        if key in target_rows:
            _fail("target has a duplicate key")
        target_rows[key] = row
        _require_sha256(row["content_digest"], "target content digest")
    expected_base = sorted(target_rows, key=_producer_key_order)
    if list(target_rows) != expected_base:
        _fail("target rows are not in canonical order")
    cross_rows: dict[tuple[str, int, str, int], pd.Series] = {}
    for _, row in crosslink.iterrows():
        key = _key_tuple(row)
        if key in cross_rows or row["scenario"] != "contiguous_20pct":
            _fail("crosslink key or scenario topology is invalid")
        cross_rows[key] = row
        _require_sha256(row["crosslink_content_digest"], "crosslink content digest")
    if list(cross_rows) != expected_base:
        _fail("crosslink rows do not exactly match canonical target order")
    scenario_rows: dict[str, dict[tuple[str, int, str, int, str], pd.Series]] = {}
    for name, frame in (("mask", mask), ("audit", audit), ("replay", replay)):
        by_key: dict[tuple[str, int, str, int, str], pd.Series] = {}
        order: list[tuple[str, int, str, int, int]] = []
        for _, row in frame.iterrows():
            base = _key_tuple(row)
            scenario = row["scenario"]
            if type(scenario) is not str or scenario not in _SCENARIOS:
                _fail(f"{name} contains an undeclared scenario")
            key = (*base, scenario)
            if key in by_key:
                _fail(f"{name} contains a duplicate key")
            by_key[key] = row
            order.append((*base, _SCENARIOS.index(scenario)))
        expected = [(*key, index) for key in expected_base for index in range(2)]
        if order != expected:
            _fail(f"{name} rows do not have canonical complete scenario order")
        scenario_rows[name] = by_key
    for key in expected_base:
        target_row = target_rows[key]
        cross_row = cross_rows[key]
        if cross_row["target_content_digest"] != target_row["content_digest"]:
            _fail("crosslink target digest is crosswired")
        target_count = int(target_row["target_deleted_count"])
        if int(cross_row["target_valid_deleted_count"]) != target_count:
            _fail("target/crosslink deletion counts are crosswired")
        _require_exact_int(
            cross_row["official_deleted_count"], "crosslink historical deleted count"
        )
        _require_exact_int(
            cross_row["official_retained_count"], "crosslink historical retained count"
        )
        for scenario in _SCENARIOS:
            full = (*key, scenario)
            mask_row = scenario_rows["mask"][full]
            audit_row = scenario_rows["audit"][full]
            replay_row = scenario_rows["replay"][full]
            mask_digest = _require_sha256(mask_row["content_digest"], "mask content digest")
            if audit_row["mask_digest"] != mask_digest or replay_row["mask_digest"] != mask_digest:
                _fail("mask digest is crosswired")
            if replay_row["target_digest"] != target_row["content_digest"]:
                _fail("replay target digest is crosswired")
            if any(
                int(row["target_deleted_count"]) != target_count
                for row in (mask_row, audit_row, replay_row)
            ):
                _fail("target deletion counts are crosswired")
            selected_count = int(mask_row["selected_record_count"])
            audit_deleted = int(audit_row["deleted_count"])
            replay_deleted = int(replay_row["deleted_count"])
            audit_retained = int(audit_row["retained_count"])
            replay_retained = int(replay_row["retained_count"])
            if audit_deleted != replay_deleted or audit_retained != replay_retained:
                _fail("audit/replay deleted or retained counts are crosswired")
            exact_match = audit_row["exact_match"]
            if type(exact_match) is not bool:
                _fail("audit exact_match is not an exact boolean")
            mask_status = mask_row["mask_status"]
            if (
                type(mask_status) is not str
                or mask_status not in ("eligible", "ineligible", "not_applicable")
            ):
                _fail("mask status is not a frozen exact value")
            status = replay_row["status"]
            if type(status) is not str or not status:
                _fail("replay status is not an exact nonempty string")
            reason = replay_row["reason"]
            if type(reason) is not str or not reason:
                _fail("replay reason is not an exact nonempty string")
            masked_value = float(replay_row["masked_value"])
            standardized_error = float(replay_row["standardized_absolute_error"])
            masked_is_nan = math.isnan(masked_value)
            error_is_nan = math.isnan(standardized_error)
            if status == "structurally_unavailable":
                if (
                    mask_status not in ("ineligible", "not_applicable")
                    or selected_count != 0
                    or audit_deleted != 0
                    or audit_retained != 0
                    or exact_match is not False
                    or not masked_is_nan
                    or not error_is_nan
                ):
                    _fail("structural N/A status/count/value contract is crosswired")
            elif status == "execution_failure":
                if (
                    mask_status != "eligible"
                    or selected_count != target_count
                    or audit_deleted != 0
                    or audit_retained != 0
                    or exact_match is not False
                    or not reason.startswith("execution_failure:")
                    or reason == "execution_failure:"
                    or not masked_is_nan
                    or not error_is_nan
                ):
                    _fail("execution failure status/count/value contract is crosswired")
            elif status == "invalid_contract_or_provenance":
                if (
                    mask_status != "eligible"
                    or selected_count != target_count
                    or audit_deleted == target_count
                    or exact_match is not False
                    or reason != status
                    or not masked_is_nan
                    or not error_is_nan
                ):
                    _fail("invalid replay status/count/value contract is crosswired")
            elif status in (
                "finite",
                "abstained_no_event_support",
                "abstained_insufficient_support",
            ):
                if (
                    mask_status != "eligible"
                    or selected_count != target_count
                    or audit_deleted != target_count
                    or exact_match is not True
                    or reason != status
                ):
                    _fail("successful replay status/count contract is crosswired")
                if status == "finite":
                    if not math.isfinite(masked_value) or not math.isfinite(
                        standardized_error
                    ):
                        _fail("finite replay has a nonfinite value or error")
                elif not masked_is_nan or not error_is_nan:
                    _fail("contract abstention has a fabricated value or error")
            else:
                _fail("replay status is outside the frozen producer domain")
    if pins is not None:
        if set(key[3] for key in target_rows) != set(pins.draws):
            _fail("target draw topology differs from pins")
        cell_draws: dict[tuple[str, int, str], list[int]] = {}
        for subject, sensor_day, primitive, draw in expected_base:
            cell_draws.setdefault((subject, sensor_day, primitive), []).append(draw)
        if len(cell_draws) != pins.selected_cell_count or any(
            tuple(draws) != pins.draws for draws in cell_draws.values()
        ):
            _fail("selected-cell by draw topology differs from pins")
    return _compact_sha(
        [
            "task-12-replay-topology-v1",
            [list(key) for key in expected_base],
            [
                [*key, scenario]
                for key in expected_base
                for scenario in _SCENARIOS
            ],
        ]
    )


def load_task12_canonical_replay_authority(
    worktree_root: Path,
    pins: Task12ReplayPins,
    *,
    expected_pins_digest: str,
) -> Task12CanonicalReplayAuthority:
    _validate_replay_pins(pins, expected_pins_digest)
    root = worktree_root
    output_relative = _strict_relative_path(pins.output_relative_path, "output path")
    output_dir = _literal_path(root, output_relative, "output directory")
    if not output_dir.is_dir():
        _fail("output path is not a directory")
    expected_names = {"manifest.json", *(pin.relative_path for pin in pins.artifacts)}
    actual_names = {entry.name for entry in output_dir.iterdir()}
    if actual_names != expected_names:
        _fail("output directory has missing or undeclared artifacts")
    manifest_path = _literal_path(
        root, _strict_relative_path(pins.manifest_relative_path, "manifest path"), "manifest"
    )
    release_path = _literal_path(
        root, _strict_relative_path(pins.release_relative_path, "release path"), "release"
    )
    consumption_path = _literal_path(
        root,
        _strict_relative_path(pins.consumption_relative_path, "consumption path"),
        "consumption",
    )
    identities: set[tuple[int, int]] = set()

    def read_unique(
        path: Path, label: str
    ) -> tuple[bytes, tuple[int, int, int, int]]:
        raw, identity = _read_stable_file(path, label)
        file_identity = identity[:2]
        if file_identity in identities:
            _fail(f"{label} aliases another input")
        identities.add(file_identity)
        return raw, identity

    release_raw, release_identity = read_unique(release_path, "release")
    consumption_raw, consumption_identity = read_unique(consumption_path, "consumption")
    if hashlib.sha256(release_raw).hexdigest() != pins.task12_release_sha256:
        _fail("release physical digest differs from its pin")
    if hashlib.sha256(consumption_raw).hexdigest() != pins.task12_consumption_sha256:
        _fail("consumption physical digest differs from its pin")
    release_sha = hashlib.sha256(release_raw).hexdigest()
    release_document = _strict_json(release_raw, "release")
    consumption_document = _strict_json(consumption_raw, "consumption")
    _verify_stable_file(release_path, release_identity, "release")
    _verify_stable_file(consumption_path, consumption_identity, "consumption")
    _validate_predecessors(
        release_document,
        consumption_document,
        pins,
        release_sha,
    )
    manifest_raw, manifest_identity = read_unique(manifest_path, "manifest")
    if (
        len(manifest_raw) != pins.manifest_byte_count
        or hashlib.sha256(manifest_raw).hexdigest() != pins.manifest_sha256
    ):
        _fail("manifest bytes differ from its pin")
    manifest = _validate_manifest(_strict_json(manifest_raw, "manifest"), pins, release_sha)
    _verify_stable_file(manifest_path, manifest_identity, "manifest")
    retained: list[ReplayArtifactSnapshot] = []
    retained_frames: dict[str, pd.DataFrame] = {}
    for pin in pins.artifacts:
        artifact_path = _literal_path(
            output_dir,
            _strict_relative_path(pin.relative_path, f"{pin.name} path", basename=True),
            pin.name,
        )
        raw, artifact_identity = read_unique(artifact_path, pin.name)
        if len(raw) != pin.byte_count or hashlib.sha256(raw).hexdigest() != pin.physical_sha256:
            _fail(f"{pin.name} physical bytes differ from its pin")
        _, frame = _parse_parquet(raw, pin)
        _verify_stable_file(artifact_path, artifact_identity, pin.name)
        if pin.retained_snapshot:
            snapshot = ReplayArtifactSnapshot(
                pin=pin,
                snapshot_bytes=bytes(raw),
                content_digest=_snapshot_digest(pin, raw),
            )
            retained.append(snapshot)
            retained_frames[pin.name] = frame
    topology_digest = _validate_topology(retained_frames, pins)
    result_digest = _require_sha256(manifest["result_content_digest"], "result digest")
    retained_tuple = tuple(retained)
    content_digest = _authority_digest(
        output_dir,
        manifest_path,
        manifest_raw,
        pins.content_digest,
        retained_tuple,
        _compact_sha(["task-12-bound-result-v1", result_digest, topology_digest]),
    )
    return Task12CanonicalReplayAuthority(
        output_dir=output_dir,
        manifest_path=manifest_path,
        manifest_bytes=bytes(manifest_raw),
        pins_digest=pins.content_digest,
        retained_artifacts=retained_tuple,
        result_content_digest=result_digest,
        content_digest=content_digest,
    )


def _validate_loaded_authority(
    authority: Task12CanonicalReplayAuthority,
) -> dict[str, pd.DataFrame]:
    if type(authority) is not Task12CanonicalReplayAuthority:
        _fail("replay authority type is forged")
    if (
        not isinstance(authority.output_dir, Path)
        or not isinstance(authority.manifest_path, Path)
        or type(authority.manifest_bytes) is not bytes
        or type(authority.retained_artifacts) is not tuple
        or len(authority.retained_artifacts) != 5
    ):
        _fail("replay authority fields are forged")
    _require_sha256(authority.pins_digest, "authority pins digest")
    _require_sha256(authority.result_content_digest, "authority result digest")
    _require_sha256(authority.content_digest, "authority content digest")
    document = _strict_json(authority.manifest_bytes, "authority manifest snapshot")
    if type(document) is not dict or document.get("result_content_digest") != authority.result_content_digest:
        _fail("authority manifest snapshot is crosswired")
    frames: dict[str, pd.DataFrame] = {}
    for expected_name, snapshot in zip(_RETAINED_NAMES, authority.retained_artifacts, strict=True):
        if type(snapshot) is not ReplayArtifactSnapshot or type(snapshot.snapshot_bytes) is not bytes:
            _fail("retained snapshot type is forged")
        _validate_artifact_pin(snapshot.pin, expected_name)
        _require_sha256(snapshot.content_digest, "retained snapshot digest")
        if snapshot.content_digest != _snapshot_digest(snapshot.pin, snapshot.snapshot_bytes):
            _fail("retained snapshot digest is forged")
        _, frames[expected_name] = _parse_parquet(snapshot.snapshot_bytes, snapshot.pin)
    topology_digest = _validate_topology(frames)
    bound_result = _compact_sha(
        ["task-12-bound-result-v1", authority.result_content_digest, topology_digest]
    )
    expected = _authority_digest(
        authority.output_dir,
        authority.manifest_path,
        authority.manifest_bytes,
        authority.pins_digest,
        authority.retained_artifacts,
        bound_result,
    )
    if authority.content_digest != expected:
        _fail("replay authority content digest is forged")
    return frames


def read_task12_replay_table(
    authority: Task12CanonicalReplayAuthority,
    name: str,
) -> pd.DataFrame:
    if type(name) is not str or name not in _RETAINED_NAMES:
        _fail("requested replay table is not one of the five retained snapshots")
    frames = _validate_loaded_authority(authority)
    return frames[name].copy(deep=True).reset_index(drop=True)


def _validate_label_pin(pin: TrainingLabelPin, expected_pin_digest: str) -> None:
    if type(pin) is not TrainingLabelPin:
        _fail("training label pin type is forged")
    expected = _require_sha256(expected_pin_digest, "externally expected label pin digest")
    _require_sha256(pin.content_digest, "label pin digest")
    if pin.content_digest != expected:
        _fail("label pin does not match the independently held digest")
    if pin.relative_path != _LABEL_PATH:
        _fail("training label path is not the reviewed literal")
    _strict_relative_path(pin.relative_path, "training label path")
    _require_exact_int(pin.byte_count, "training label byte count", positive=True)
    _require_sha256(pin.physical_sha256, "training label physical digest")
    _require_exact_int(pin.row_count, "training label row count", positive=True)
    if (
        type(pin.columns) is not tuple
        or any(type(value) is not str for value in pin.columns)
        or pin.columns != _LABEL_COLUMNS
    ):
        _fail("training label columns are not exact")
    _require_exact_int(pin.participant_count, "training label participant count", positive=True)
    if pin.content_digest != _label_pin_digest(pin):
        _fail("training label pin is not self-consistent")


def _canonical_date(value: str, label: str) -> date:
    if type(value) is not str or _DATE.fullmatch(value) is None:
        _fail(f"{label} is not canonical YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        _fail(f"{label} is invalid: {exc}")
    if parsed.isoformat() != value:
        _fail(f"{label} does not round trip canonically")
    return parsed


def _parse_label_rows(raw: bytes) -> tuple[TrainingLabelRow, ...]:
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw:
        _fail("training label CSV has a BOM or NUL")
    try:
        text = raw.decode("utf-8", errors="strict")
        parsed = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as exc:
        _fail(f"training label CSV is malformed: {exc}")
    if not parsed or tuple(parsed[0]) != _LABEL_COLUMNS:
        _fail("training label header order is not exact")
    rows: list[TrainingLabelRow] = []
    seen: set[tuple[str, str, str]] = set()
    for values in parsed[1:]:
        if len(values) != len(_LABEL_COLUMNS):
            _fail("training label CSV has a ragged row")
        subject, sleep_text, lifelog_text = values[:3]
        if type(subject) is not str or not subject:
            _fail("training label subject is empty")
        sleep = _canonical_date(sleep_text, "sleep_date")
        lifelog = _canonical_date(lifelog_text, "lifelog_date")
        if sleep != lifelog + timedelta(days=1):
            _fail("sleep_date must equal lifelog_date plus one day")
        target_values = values[6:10]
        if any(value not in ("0", "1") for value in target_values):
            _fail("S1-S4 must be exact nonmissing binary values")
        key = (subject, sleep_text, lifelog_text)
        if key in seen:
            _fail("training label key is duplicated")
        seen.add(key)
        temporary = TrainingLabelRow(
            subject_id=subject,
            sleep_date=sleep_text,
            lifelog_date=lifelog_text,
            s1=int(target_values[0]),
            s2=int(target_values[1]),
            s3=int(target_values[2]),
            s4=int(target_values[3]),
            content_digest="",
        )
        rows.append(
            TrainingLabelRow(
                subject_id=temporary.subject_id,
                sleep_date=temporary.sleep_date,
                lifelog_date=temporary.lifelog_date,
                s1=temporary.s1,
                s2=temporary.s2,
                s3=temporary.s3,
                s4=temporary.s4,
                content_digest=_label_row_digest(temporary),
            )
        )
    return tuple(rows)


def _label_digests(rows: tuple[TrainingLabelRow, ...]) -> tuple[str, str]:
    keys = [[row.subject_id, row.sleep_date, row.lifelog_date] for row in rows]
    targets = [[row.s1, row.s2, row.s3, row.s4] for row in rows]
    return (
        _compact_sha(["task-12-training-label-keys-v1", keys]),
        _compact_sha(["task-12-training-label-targets-v1", targets]),
    )


def load_training_label_authority(
    workspace_root: Path,
    pin: TrainingLabelPin,
    *,
    expected_pin_digest: str,
) -> TrainingLabelAuthority:
    _validate_label_pin(pin, expected_pin_digest)
    relative = _strict_relative_path(pin.relative_path, "training label path")
    path = _literal_path(workspace_root, relative, "training label")
    raw, identity = _read_stable_file(path, "training label")
    if len(raw) != pin.byte_count or hashlib.sha256(raw).hexdigest() != pin.physical_sha256:
        _fail("training label physical bytes differ from its pin")
    rows = _parse_label_rows(raw)
    _verify_stable_file(path, identity, "training label")
    if len(rows) != pin.row_count:
        _fail("training label row count differs from its pin")
    if len({row.subject_id for row in rows}) != pin.participant_count:
        _fail("training label participant count differs from its pin")
    key_digest, target_digest = _label_digests(rows)
    content_digest = _label_authority_digest(
        path, raw, pin.content_digest, rows, key_digest, target_digest
    )
    return TrainingLabelAuthority(
        path=path,
        snapshot_bytes=bytes(raw),
        pin_digest=pin.content_digest,
        rows=rows,
        key_digest=key_digest,
        target_digest=target_digest,
        content_digest=content_digest,
    )


def _validate_label_authority(authority: TrainingLabelAuthority) -> None:
    if type(authority) is not TrainingLabelAuthority:
        _fail("training label authority type is forged")
    if (
        not isinstance(authority.path, Path)
        or type(authority.snapshot_bytes) is not bytes
        or type(authority.rows) is not tuple
        or any(type(row) is not TrainingLabelRow for row in authority.rows)
    ):
        _fail("training label authority fields are forged")
    for digest, label in (
        (authority.pin_digest, "label pin digest"),
        (authority.key_digest, "label key digest"),
        (authority.target_digest, "label target digest"),
        (authority.content_digest, "label authority digest"),
    ):
        _require_sha256(digest, label)
    rows = _parse_label_rows(authority.snapshot_bytes)
    if rows != authority.rows:
        _fail("training label snapshot and rows are crosswired")
    for row in rows:
        if row.content_digest != _label_row_digest(row):
            _fail("training label row digest is forged")
    key_digest, target_digest = _label_digests(rows)
    if key_digest != authority.key_digest or target_digest != authority.target_digest:
        _fail("training label ordered digests are forged")
    expected = _label_authority_digest(
        authority.path,
        authority.snapshot_bytes,
        authority.pin_digest,
        authority.rows,
        authority.key_digest,
        authority.target_digest,
    )
    if expected != authority.content_digest:
        _fail("training label authority digest is forged")


def training_label_frame(authority: TrainingLabelAuthority) -> pd.DataFrame:
    _validate_label_authority(authority)
    frame = pd.DataFrame.from_records(
        [
            {
                "subject_id": row.subject_id,
                "sleep_date": row.sleep_date,
                "lifelog_date": row.lifelog_date,
                "S1": row.s1,
                "S2": row.s2,
                "S3": row.s3,
                "S4": row.s4,
            }
            for row in authority.rows
        ],
        columns=["subject_id", "sleep_date", "lifelog_date", "S1", "S2", "S3", "S4"],
    )
    frame["subject_id"] = pd.Series(frame["subject_id"], dtype="string")
    frame["sleep_date"] = pd.to_datetime(
        frame["sleep_date"], format="%Y-%m-%d"
    ).astype("datetime64[ns]")
    frame["lifelog_date"] = pd.to_datetime(
        frame["lifelog_date"], format="%Y-%m-%d"
    ).astype("datetime64[ns]")
    for name in ("S1", "S2", "S3", "S4"):
        frame[name] = pd.Series(frame[name], dtype="int64")
    return frame.copy(deep=True).reset_index(drop=True)


def _validate_downstream_patch_constants() -> None:
    if _DOWNSTREAM_PATCH_ARMS != (
        "matched_fixed_personalized",
        "semantic_personalized",
    ) or _DOWNSTREAM_PATCH_ARMS != outcome.DOWNSTREAM_ARM_NAMES:
        _fail("downstream patch arms no longer match the frozen live arms")
    flattened = tuple(
        primitive
        for _family, primitives in _DOWNSTREAM_FAMILY_PRIMITIVES
        for primitive in primitives
    )
    if flattened != _PRIMITIVE_ORDER:
        _fail("downstream family primitive order no longer matches the replay order")
    if _DOWNSTREAM_CONTEXT_PRIMITIVES != outcome.MATCHED_SOURCE_PRIMITIVES:
        _fail("downstream context basis no longer matches the live matched basis")
    if len(_DOWNSTREAM_CONTEXT_PRIMITIVES) != 15:
        _fail("downstream context basis must contain exactly fifteen primitives")
    for primitive, descendants in _DOWNSTREAM_SEMANTIC_DESCENDANTS:
        live = tuple(
            semantic
            for semantic, source, _representation, _operation in outcome.SEMANTIC_SOURCE_LINEAGE
            if source == primitive
        )
        if live != descendants:
            _fail(f"semantic lineage no longer matches target {primitive}")


def _downstream_exact_string(value: object, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and value == ""):
        _fail(f"{label} must be an exact nonempty built-in string")
    return value


def _downstream_exact_float(
    value: object,
    label: str,
    *,
    allow_nan: bool,
) -> float:
    if type(value) is not float or math.isinf(value) or (math.isnan(value) and not allow_nan):
        suffix = "finite or NaN" if allow_nan else "finite"
        _fail(f"{label} must be an exact built-in float that is {suffix}")
    if math.isnan(value):
        bits = struct.unpack(">Q", struct.pack(">d", value))[0]
        if bits & 0x0008_0000_0000_0000 == 0:
            _fail(f"{label} must use a quiet NaN encoding")
    return value


def _downstream_float_token(value: object, label: str, *, allow_nan: bool = True) -> list[str]:
    checked = _downstream_exact_float(value, label, allow_nan=allow_nan)
    return ["f64be", struct.pack(">d", checked).hex()]


def _downstream_baseline_digest(baseline: DownstreamPrimitiveBaseline) -> str:
    return _compact_sha(
        [
            "task-12-downstream-primitive-baseline-v1",
            baseline.primitive,
            baseline.method,
            baseline.k,
            baseline.donor_subject,
            _downstream_float_token(baseline.center, "baseline center", allow_nan=False),
            _downstream_float_token(baseline.scale, "baseline scale", allow_nan=False),
            _downstream_float_token(
                baseline.lambda_days, "baseline lambda_days", allow_nan=False
            ),
        ]
    )


def _downstream_arm_row_digest(arm_row: DownstreamReferenceArmRow) -> str:
    return _compact_sha(
        [
            "task-12-downstream-reference-arm-row-v1",
            arm_row.arm,
            list(arm_row.feature_names),
            [
                _downstream_float_token(value, "reference feature value")
                for value in arm_row.feature_values
            ],
        ]
    )


def _downstream_context_digest(context: DownstreamReferenceContext) -> str:
    return _compact_sha(
        [
            "task-12-downstream-reference-context-v1",
            context.fold,
            context.held_out_subject,
            context.sensor_day_id,
            [
                [
                    name,
                    _downstream_float_token(raw, "context raw value"),
                    _downstream_float_token(z, "context personalized z"),
                    _downstream_float_token(
                        confidence, "context confidence", allow_nan=False
                    ),
                ]
                for name, raw, z, confidence in context.primitive_basis
            ],
            [baseline.content_digest for baseline in context.baselines],
            [arm_row.content_digest for arm_row in context.arm_rows],
        ]
    )


def _downstream_perturbation_digest(perturbation: DownstreamPerturbation) -> str:
    return _compact_sha(
        [
            "task-12-downstream-perturbation-v1",
            perturbation.fold,
            perturbation.held_out_subject,
            perturbation.sensor_day_id,
            perturbation.primitive,
            perturbation.draw,
            perturbation.condition,
            perturbation.replay_status,
            perturbation.reason,
            _downstream_float_token(perturbation.masked_value, "masked value"),
        ]
    )


def _downstream_patch_digest(patch: DownstreamFeaturePatch) -> str:
    return _compact_sha(
        [
            "task-12-downstream-feature-patch-v1",
            patch.fold,
            patch.held_out_subject,
            patch.sensor_day_id,
            patch.primitive,
            patch.draw,
            patch.condition,
            patch.arm,
            patch.replay_status,
            patch.reason,
            list(patch.feature_names),
            [
                _downstream_float_token(value, "patch feature value")
                for value in patch.feature_values
            ],
            patch.reference_context_digest,
            patch.reference_arm_row_digest,
            patch.perturbation_digest,
        ]
    )


def _downstream_matched_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for primitive in _PRIMITIVE_ORDER:
        columns.append(f"primitive__{primitive}__personalized_z")
        if primitive in ("screen_disengagement_p90", "usage_disengagement_p90"):
            columns.append(f"primitive__{primitive}__raw_minute")
    return tuple(columns)


def _downstream_semantic_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for _primitive, descendants in _DOWNSTREAM_SEMANTIC_DESCENDANTS:
        for descendant in descendants:
            column = f"semantic__{descendant}__value"
            if column not in columns:
                columns.append(column)
    return tuple(columns)


def _validate_downstream_baseline(
    baseline: DownstreamPrimitiveBaseline,
    expected_primitive: str,
) -> None:
    if type(baseline) is not DownstreamPrimitiveBaseline:
        _fail("downstream baseline type is forged")
    if type(baseline.primitive) is not str or baseline.primitive != expected_primitive:
        _fail("downstream baseline primitive order is invalid")
    if type(baseline.method) is not str or baseline.method != "shrinkage":
        _fail("downstream baseline method must be shrinkage")
    if type(baseline.k) is not int or baseline.k != outcome.PRIMARY_CALIBRATION_DAY:
        _fail("downstream baseline k does not match the frozen calibration day")
    if baseline.donor_subject is not None:
        _fail("downstream baseline donor_subject must be None")
    center = _downstream_exact_float(
        baseline.center, "downstream baseline center", allow_nan=False
    )
    scale = _downstream_exact_float(
        baseline.scale, "downstream baseline scale", allow_nan=False
    )
    lambda_days = _downstream_exact_float(
        baseline.lambda_days, "downstream baseline lambda_days", allow_nan=False
    )
    if scale <= 0.0:
        _fail("downstream baseline scale must be positive")
    if lambda_days != outcome.PRIMARY_LAMBDA_DAYS:
        _fail("downstream baseline lambda_days does not match the frozen value")
    _require_sha256(baseline.content_digest, "downstream baseline content digest")
    if baseline.content_digest != _downstream_baseline_digest(baseline):
        _fail("downstream baseline content digest is forged")
    del center


def _validate_downstream_arm_row(
    arm_row: DownstreamReferenceArmRow,
    expected_arm: str,
) -> None:
    if type(arm_row) is not DownstreamReferenceArmRow:
        _fail("downstream reference arm row type is forged")
    if type(arm_row.arm) is not str or arm_row.arm != expected_arm:
        _fail("downstream reference arm order is invalid")
    if type(arm_row.feature_names) is not tuple or type(arm_row.feature_values) is not tuple:
        _fail("downstream reference arm features must use immutable tuples")
    if not arm_row.feature_names or len(arm_row.feature_names) != len(arm_row.feature_values):
        _fail("downstream reference arm feature lengths are invalid")
    if any(type(name) is not str or name == "" for name in arm_row.feature_names):
        _fail("downstream reference feature names must be exact nonempty strings")
    if len(set(arm_row.feature_names)) != len(arm_row.feature_names):
        _fail("downstream reference feature names must be unique")
    identity_columns = {
        "fold",
        "held_out_subject",
        "subject_id",
        "sensor_day_id",
        "sleep_date",
        "lifelog_date",
        "primitive",
        "draw",
        "condition",
    }
    if identity_columns.intersection(arm_row.feature_names):
        _fail("identity columns cannot appear among downstream model features")
    for value in arm_row.feature_values:
        _downstream_exact_float(value, "downstream reference feature value", allow_nan=True)
    required = (
        _downstream_matched_columns()
        if expected_arm == "matched_fixed_personalized"
        else _downstream_semantic_columns()
    )
    if any(arm_row.feature_names.count(name) != 1 for name in required):
        _fail("downstream reference arm is missing a required patch feature")
    _require_sha256(arm_row.content_digest, "downstream reference arm digest")
    if arm_row.content_digest != _downstream_arm_row_digest(arm_row):
        _fail("downstream reference arm digest is forged")


def _validate_downstream_context(context: DownstreamReferenceContext) -> None:
    _validate_downstream_patch_constants()
    if type(context) is not DownstreamReferenceContext:
        _fail("downstream reference context type is forged")
    if type(context.fold) is not int or not 0 <= context.fold <= 9:
        _fail("downstream context fold must be an exact integer in 0..9")
    _downstream_exact_string(context.held_out_subject, "downstream held-out subject")
    if type(context.sensor_day_id) is not int or context.sensor_day_id < 0:
        _fail("downstream sensor_day_id must be an exact nonnegative integer")
    if type(context.primitive_basis) is not tuple or len(context.primitive_basis) != len(
        _DOWNSTREAM_CONTEXT_PRIMITIVES
    ):
        _fail("downstream primitive basis topology is invalid")
    for entry, expected_name in zip(
        context.primitive_basis, _DOWNSTREAM_CONTEXT_PRIMITIVES, strict=True
    ):
        if type(entry) is not tuple or len(entry) != 4:
            _fail("downstream primitive basis entries must be exact four-tuples")
        name, raw, z, confidence = entry
        if type(name) is not str or name != expected_name:
            _fail("downstream primitive basis order is invalid")
        _downstream_exact_float(raw, "downstream primitive raw value", allow_nan=True)
        _downstream_exact_float(z, "downstream primitive personalized z", allow_nan=True)
        checked_confidence = _downstream_exact_float(
            confidence, "downstream primitive confidence", allow_nan=False
        )
        if not 0.0 <= checked_confidence <= 1.0:
            _fail("downstream primitive confidence must be in [0, 1]")
    if type(context.baselines) is not tuple or len(context.baselines) != len(_PRIMITIVE_ORDER):
        _fail("downstream baseline topology is invalid")
    for baseline, expected_primitive in zip(
        context.baselines, _PRIMITIVE_ORDER, strict=True
    ):
        _validate_downstream_baseline(baseline, expected_primitive)
    if type(context.arm_rows) is not tuple or len(context.arm_rows) != len(
        _DOWNSTREAM_PATCH_ARMS
    ):
        _fail("downstream reference arm topology is invalid")
    for arm_row, expected_arm in zip(context.arm_rows, _DOWNSTREAM_PATCH_ARMS, strict=True):
        _validate_downstream_arm_row(arm_row, expected_arm)
    _require_sha256(context.content_digest, "downstream reference context digest")
    if context.content_digest != _downstream_context_digest(context):
        _fail("downstream reference context digest is forged")


def _validate_downstream_perturbation(perturbation: DownstreamPerturbation) -> None:
    if type(perturbation) is not DownstreamPerturbation:
        _fail("downstream perturbation type is forged")
    if type(perturbation.fold) is not int or not 0 <= perturbation.fold <= 9:
        _fail("downstream perturbation fold must be an exact integer in 0..9")
    _downstream_exact_string(
        perturbation.held_out_subject, "downstream perturbation held-out subject"
    )
    if type(perturbation.sensor_day_id) is not int or perturbation.sensor_day_id < 0:
        _fail("downstream perturbation sensor_day_id is invalid")
    if type(perturbation.primitive) is not str or perturbation.primitive not in _PRIMITIVE_ORDER:
        _fail("downstream perturbation primitive is invalid")
    if type(perturbation.draw) is not int or perturbation.draw < 0:
        _fail("downstream perturbation draw must be an exact nonnegative integer")
    if type(perturbation.condition) is not str or perturbation.condition not in (
        _DOWNSTREAM_PATCH_CONDITIONS
    ):
        _fail("downstream perturbation condition is invalid")
    status = _downstream_exact_string(
        perturbation.replay_status, "downstream perturbation replay_status"
    )
    reason = _downstream_exact_string(
        perturbation.reason, "downstream perturbation reason"
    )
    value = _downstream_exact_float(
        perturbation.masked_value, "downstream perturbation masked value", allow_nan=True
    )
    if status == "finite":
        if reason != "finite" or not math.isfinite(value):
            _fail("finite downstream perturbation has a status/value mismatch")
    elif status in (
        "abstained_no_event_support",
        "abstained_insufficient_support",
    ):
        if reason != status or not math.isnan(value):
            _fail("abstained downstream perturbation has a status/value mismatch")
    elif status == "structurally_unavailable":
        if perturbation.condition != "event_boundary_20pct" or not math.isnan(value):
            _fail("structural N/A is valid only for a NaN event-boundary perturbation")
    elif status == "invalid_contract_or_provenance":
        if reason != status or not math.isnan(value):
            _fail("invalid-contract perturbation has a status/value mismatch")
    elif status == "execution_failure":
        prefix = "execution_failure:"
        if not reason.startswith(prefix) or reason == prefix or not math.isnan(value):
            _fail("execution-failure perturbation has a status/value mismatch")
    else:
        _fail("downstream perturbation replay_status is unknown")
    _require_sha256(perturbation.content_digest, "downstream perturbation digest")
    if perturbation.content_digest != _downstream_perturbation_digest(perturbation):
        _fail("downstream perturbation digest is forged")


def _expected_patch_feature_names(primitive: str, arm: str) -> tuple[str, ...]:
    if arm == "matched_fixed_personalized":
        names = [f"primitive__{primitive}__personalized_z"]
        if primitive in ("screen_disengagement_p90", "usage_disengagement_p90"):
            names.append(f"primitive__{primitive}__raw_minute")
        return tuple(names)
    descendants = dict(_DOWNSTREAM_SEMANTIC_DESCENDANTS)[primitive]
    return tuple(f"semantic__{name}__value" for name in descendants)


def _validate_downstream_patch(patch: DownstreamFeaturePatch) -> None:
    _validate_downstream_patch_constants()
    if type(patch) is not DownstreamFeaturePatch:
        _fail("downstream feature patch type is forged")
    if type(patch.fold) is not int or not 0 <= patch.fold <= 9:
        _fail("downstream feature patch fold is invalid")
    _downstream_exact_string(patch.held_out_subject, "downstream patch held-out subject")
    if type(patch.sensor_day_id) is not int or patch.sensor_day_id < 0:
        _fail("downstream feature patch sensor_day_id is invalid")
    if type(patch.primitive) is not str or patch.primitive not in _PRIMITIVE_ORDER:
        _fail("downstream feature patch primitive is invalid")
    if type(patch.draw) is not int or patch.draw < 0:
        _fail("downstream feature patch draw is invalid")
    if type(patch.condition) is not str or patch.condition not in _DOWNSTREAM_PATCH_CONDITIONS:
        _fail("downstream feature patch condition is invalid")
    if type(patch.arm) is not str or patch.arm not in _DOWNSTREAM_PATCH_ARMS:
        _fail("downstream feature patch arm is invalid")
    status = _downstream_exact_string(patch.replay_status, "downstream patch replay_status")
    reason = _downstream_exact_string(patch.reason, "downstream patch reason")
    if type(patch.feature_names) is not tuple or type(patch.feature_values) is not tuple:
        _fail("downstream patch features must use immutable tuples")
    if len(patch.feature_names) != len(patch.feature_values):
        _fail("downstream patch feature lengths do not match")
    if any(type(name) is not str or name == "" for name in patch.feature_names):
        _fail("downstream patch feature names must be exact nonempty strings")
    if len(set(patch.feature_names)) != len(patch.feature_names):
        _fail("downstream patch feature names must be unique")
    for value in patch.feature_values:
        _downstream_exact_float(value, "downstream patch feature value", allow_nan=True)
    if status == "finite":
        if reason != "finite" or patch.feature_names != _expected_patch_feature_names(
            patch.primitive, patch.arm
        ):
            _fail("finite downstream patch topology is invalid")
        if patch.arm == "matched_fixed_personalized" and any(
            not math.isfinite(value) for value in patch.feature_values
        ):
            _fail("matched downstream patch values must be finite")
    elif status in (
        "abstained_no_event_support",
        "abstained_insufficient_support",
    ):
        if reason != status or patch.feature_names or patch.feature_values:
            _fail("abstained downstream patch topology is invalid")
    elif status == "structurally_unavailable":
        if (
            patch.condition != "event_boundary_20pct"
            or patch.feature_names
            or patch.feature_values
        ):
            _fail("structural downstream patch topology is invalid")
    else:
        _fail("integrity-failure or unknown status cannot be a downstream patch")
    for value, label in (
        (patch.reference_context_digest, "patch reference context digest"),
        (patch.reference_arm_row_digest, "patch reference arm-row digest"),
        (patch.perturbation_digest, "patch perturbation digest"),
        (patch.content_digest, "downstream feature patch digest"),
    ):
        _require_sha256(value, label)
    if patch.content_digest != _downstream_patch_digest(patch):
        _fail("downstream feature patch digest is forged")


def _reference_arm_row(
    context: DownstreamReferenceContext, arm: str
) -> DownstreamReferenceArmRow:
    if type(arm) is not str or arm not in _DOWNSTREAM_PATCH_ARMS:
        _fail("downstream arm is not one of the two exact frozen arms")
    return context.arm_rows[_DOWNSTREAM_PATCH_ARMS.index(arm)]


def reference_feature_frame(
    context: DownstreamReferenceContext, arm: str
) -> pd.DataFrame:
    _validate_downstream_context(context)
    arm_row = _reference_arm_row(context, arm)
    return pd.DataFrame(
        [list(arm_row.feature_values)],
        columns=list(arm_row.feature_names),
        dtype="float64",
        index=pd.RangeIndex(1),
    )


def _make_downstream_patch(
    context: DownstreamReferenceContext,
    perturbation: DownstreamPerturbation,
    arm_row: DownstreamReferenceArmRow,
    feature_names: tuple[str, ...],
    feature_values: tuple[float, ...],
) -> DownstreamFeaturePatch:
    fields = {
        "fold": context.fold,
        "held_out_subject": context.held_out_subject,
        "sensor_day_id": context.sensor_day_id,
        "primitive": perturbation.primitive,
        "draw": perturbation.draw,
        "condition": perturbation.condition,
        "arm": arm_row.arm,
        "replay_status": perturbation.replay_status,
        "reason": perturbation.reason,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "reference_context_digest": context.content_digest,
        "reference_arm_row_digest": arm_row.content_digest,
        "perturbation_digest": perturbation.content_digest,
    }
    unsealed = DownstreamFeaturePatch(**fields, content_digest="0" * 64)
    result = DownstreamFeaturePatch(
        **fields, content_digest=_downstream_patch_digest(unsealed)
    )
    _validate_downstream_patch(result)
    return result


def build_downstream_feature_patch(
    context: DownstreamReferenceContext,
    perturbation: DownstreamPerturbation,
    arm: str,
) -> DownstreamFeaturePatch:
    _validate_downstream_context(context)
    _validate_downstream_perturbation(perturbation)
    arm_row = _reference_arm_row(context, arm)
    if (
        context.fold,
        context.held_out_subject,
        context.sensor_day_id,
    ) != (
        perturbation.fold,
        perturbation.held_out_subject,
        perturbation.sensor_day_id,
    ):
        _fail("downstream context and perturbation identities do not match")
    if perturbation.replay_status in (
        "invalid_contract_or_provenance",
        "execution_failure",
    ):
        _fail("integrity-failure replay rows cannot produce feature patches")
    if perturbation.replay_status != "finite":
        return _make_downstream_patch(context, perturbation, arm_row, (), ())

    baseline = context.baselines[_PRIMITIVE_ORDER.index(perturbation.primitive)]
    basis = {
        name: (raw, z, confidence)
        for name, raw, z, confidence in context.primitive_basis
    }
    raw, _old_z, confidence = basis[perturbation.primitive]
    del raw
    if not math.isfinite(confidence) or confidence <= 0.0:
        _fail("finite replay requires positive target confidence")
    personalized_z = float(
        np.clip(
            (perturbation.masked_value - baseline.center) / baseline.scale,
            -5.0,
            5.0,
        )
    )
    if not math.isfinite(personalized_z):
        _fail("finite replay produced a nonfinite personalized z value")

    if arm == "matched_fixed_personalized":
        feature_names = [f"primitive__{perturbation.primitive}__personalized_z"]
        feature_values = [personalized_z]
        if perturbation.primitive in (
            "screen_disengagement_p90",
            "usage_disengagement_p90",
        ):
            feature_names.append(f"primitive__{perturbation.primitive}__raw_minute")
            feature_values.append(perturbation.masked_value)
    else:
        composer_columns: list[str] = []
        composer_values: list[float] = []
        for primitive, basis_raw, basis_z, basis_confidence in context.primitive_basis:
            composer_columns.extend(
                (primitive, f"{primitive}__z", f"{primitive}__confidence")
            )
            composer_values.extend(
                (
                    perturbation.masked_value
                    if primitive == perturbation.primitive
                    else basis_raw,
                    personalized_z if primitive == perturbation.primitive else basis_z,
                    basis_confidence,
                )
            )
        composer_input = pd.DataFrame(
            [composer_values], columns=composer_columns, dtype="float64"
        )
        composed = personalization.compose_semantic_columns(composer_input)
        if type(composed) is not pd.DataFrame or len(composed) != 1:
            _fail("frozen semantic composer returned an invalid frame")
        descendants = dict(_DOWNSTREAM_SEMANTIC_DESCENDANTS)[perturbation.primitive]
        feature_names = []
        feature_values = []
        for descendant in descendants:
            if list(composed.columns).count(descendant) != 1:
                _fail(f"frozen semantic composer omitted descendant {descendant}")
            value = float(composed.at[0, descendant])
            _downstream_exact_float(
                value, f"semantic descendant {descendant}", allow_nan=True
            )
            feature_names.append(f"semantic__{descendant}__value")
            feature_values.append(value)
    return _make_downstream_patch(
        context,
        perturbation,
        arm_row,
        tuple(feature_names),
        tuple(feature_values),
    )


def apply_downstream_feature_patch(
    reference: pd.DataFrame,
    patch: DownstreamFeaturePatch,
    *,
    expected_context_digest: str,
    expected_perturbation_digest: str,
) -> pd.DataFrame | None:
    _validate_downstream_patch(patch)
    _require_sha256(expected_context_digest, "expected downstream context digest")
    _require_sha256(
        expected_perturbation_digest, "expected downstream perturbation digest"
    )
    if patch.reference_context_digest != expected_context_digest:
        _fail("downstream patch is crosswired to a different reference context")
    if patch.perturbation_digest != expected_perturbation_digest:
        _fail("downstream patch is crosswired to a different perturbation")
    if type(reference) is not pd.DataFrame or reference.shape[0] != 1:
        _fail("downstream patch reference must be an exact one-row DataFrame")
    if (
        type(reference.index) is not pd.RangeIndex
        or reference.index.start != 0
        or reference.index.stop != 1
        or reference.index.step != 1
    ):
        _fail("downstream patch reference must use the exact one-row RangeIndex")
    names = tuple(reference.columns)
    if any(type(name) is not str or name == "" for name in names):
        _fail("downstream patch reference columns must be exact nonempty strings")
    if len(set(names)) != len(names):
        _fail("downstream patch reference columns must be unique")
    if any(dtype != np.dtype("float64") for dtype in reference.dtypes):
        _fail("downstream patch reference columns must all be float64")
    values = tuple(float(reference.at[0, name]) for name in names)
    for value in values:
        _downstream_exact_float(value, "downstream patch reference value", allow_nan=True)
    reference_digest = _compact_sha(
        [
            "task-12-downstream-reference-arm-row-v1",
            patch.arm,
            list(names),
            [
                _downstream_float_token(value, "downstream patch reference value")
                for value in values
            ],
        ]
    )
    if reference_digest != patch.reference_arm_row_digest:
        _fail("downstream patch reference does not reproduce its frozen arm row")
    if patch.replay_status != "finite":
        return None
    if any(names.count(name) != 1 for name in patch.feature_names):
        _fail("downstream patch target column is absent or duplicated")
    result = reference.copy(deep=True)
    for name, value in zip(patch.feature_names, patch.feature_values, strict=True):
        result.at[0, name] = value
    return result


def iter_downstream_feature_patches(
    context: DownstreamReferenceContext,
    perturbations: Iterable[DownstreamPerturbation],
) -> Iterator[DownstreamFeaturePatch]:
    _validate_downstream_context(context)
    source = iter(perturbations)
    buffered: list[DownstreamPerturbation] = []
    for index in range(4):
        try:
            item = next(source)
        except StopIteration:
            break
        if index == 3:
            _fail("a downstream perturbation base group cannot contain a fourth item")
        buffered.append(item)
    if not buffered:
        _fail("a downstream perturbation base group cannot be empty")
    for perturbation in buffered:
        _validate_downstream_perturbation(perturbation)
    first = buffered[0]
    expected_key = (
        context.fold,
        context.held_out_subject,
        context.sensor_day_id,
        first.primitive,
        first.draw,
    )
    for perturbation in buffered:
        key = (
            perturbation.fold,
            perturbation.held_out_subject,
            perturbation.sensor_day_id,
            perturbation.primitive,
            perturbation.draw,
        )
        if key != expected_key:
            _fail("downstream perturbation base group keys do not match")
    ranks = [
        _DOWNSTREAM_PATCH_CONDITIONS.index(perturbation.condition)
        for perturbation in buffered
    ]
    if len(set(ranks)) != len(ranks) or ranks != sorted(ranks):
        _fail("downstream perturbation conditions are duplicate or out of canonical order")
    patches: list[DownstreamFeaturePatch] = []
    for perturbation in buffered:
        for arm in _DOWNSTREAM_PATCH_ARMS:
            patches.append(build_downstream_feature_patch(context, perturbation, arm))
    yield from patches
