"""Synthetic-only participant statistics for the frozen downstream continuation.

The analyzer accepts a one-shot Arrow RecordBatch iterable.  It never discovers an
input path and never performs model fitting or prediction.  Its only durable products
are the eleven frozen statistics tables and their integrity manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextvars import ContextVar
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


ARMS = ("matched_fixed_personalized", "semantic_personalized")
MODELS = ("elasticnet", "lightgbm")
TARGETS = ("S1", "S2", "S3", "S4")
CONDITIONS = (
    "scattered_random_20pct",
    "contiguous_20pct",
    "event_boundary_20pct",
)
LABELS = (
    "inconclusive",
    "corroborative_stability_impact_supported",
    "corroborative_impact_not_demonstrated",
)
TABLE_NAMES = (
    "primary_cell_contrasts",
    "group_cells",
    "participant_summaries",
    "bootstrap_indices",
    "bootstrap_statistics",
    "sign_ledger",
    "sign_statistics",
    "condition_group_descriptives",
    "participant_condition_descriptives",
    "global_condition_descriptives",
    "decision",
)
ISSUE_CODES = (
    "observed_input_schema_mismatch",
    "observed_input_row_count_mismatch",
    "observed_input_order_mismatch",
    "observed_input_missing_row",
    "observed_input_duplicate_row",
    "observed_input_logical_digest_mismatch",
    "observed_row_content_digest_mismatch",
    "primary_key_digest_mismatch",
    "fit_topology_incomplete",
    "fit_identity_crosswire",
    "fit_projection_digest_mismatch",
    "source_provenance_crosswire",
    "reference_contract_invalid",
    "status_contract_invalid",
    "label_contract_invalid",
    "group_capacity_exceeded",
    "denominator_incomplete",
    "group_grid_incomplete",
    "participant_grid_incomplete",
    "required_output_invalid",
)
METRICS = (
    "finite_abs_change_median",
    "coverage",
    "abstention_rate",
    "flip_rate_finite",
    "flip_fraction_applicable",
    "brier_delta_mean",
    "log_loss_delta_mean",
)
SENSITIVITY_WORDING = (
    "Fixed-roster participant-resampling sensitivity interval (frozen 2.5/97.5 linear "
    "percentiles; not nominal frequentist coverage). The exact sign-flip p-value is "
    "conditional on participant-level sign exchangeability; LOSO training overlap means "
    "that assumption is not established, so this is participant-block sensitivity, not "
    "independent evidence of label specificity."
)
TASK12_DECISION = "stop_diagnostic_inconclusive"
_BATCH_SIZE = 4096
_STATE_AUDIT_HOOK: ContextVar[object | None] = ContextVar(
    "downstream_statistics_state_audit_hook", default=None
)
_STABLE_READ_AUDIT_HOOK: ContextVar[object | None] = ContextVar(
    "downstream_statistics_stable_read_audit_hook", default=None
)
_FILE_OPEN_AUDIT_HOOK: ContextVar[object | None] = ContextVar(
    "downstream_statistics_file_open_audit_hook", default=None
)


def _columns(*items: tuple[str, str, bool]) -> tuple[tuple[str, str, bool], ...]:
    return items


OBSERVED_PREDICTION_COLUMNS = _columns(
    ("row_seq", "int64", False),
    ("participant_position", "int64", False),
    ("participant", "string", False),
    ("fold", "int64", False),
    ("held_out_subject", "string", False),
    ("family_position", "int64", False),
    ("family", "string", False),
    ("primitive_position", "int64", False),
    ("primitive", "string", False),
    ("sensor_day_id", "int64", False),
    ("draw", "int64", False),
    ("arm_position", "int64", False),
    ("arm", "string", False),
    ("model_position", "int64", False),
    ("model_name", "string", False),
    ("target_position", "int64", False),
    ("target", "string", False),
    ("condition_position", "int64", False),
    ("condition", "string", False),
    ("fit_seq", "int64", False),
    ("pre_mask_eligible", "bool", False),
    ("reference_status", "string", False),
    ("replay_status", "string", False),
    ("replay_reason", "string", False),
    ("applicable", "bool", False),
    ("coverage_eligible", "bool", False),
    ("reference_probability", "double", False),
    ("perturbed_probability", "double", True),
    ("reference_decision", "bool", False),
    ("perturbed_decision", "bool", True),
    ("absolute_probability_change", "double", True),
    ("decision_flip", "bool", True),
    ("instability_burden", "double", True),
    ("label_available", "bool", False),
    ("label", "int64", True),
    ("fit_ledger_record_digest", "string", False),
    ("preprocessor_digest", "string", False),
    ("model_digest", "string", False),
    ("fit_content_digest", "string", False),
    ("source_row_digest", "string", False),
    ("reference_context_digest", "string", False),
    ("row_content_digest", "string", False),
)

EXPECTED_KEY_COLUMNS = _columns(
    ("participant_position", "int64", False), ("participant", "string", False),
    ("fold", "int64", False), ("held_out_subject", "string", False),
    ("family_position", "int64", False), ("family", "string", False),
    ("arm_position", "int64", False), ("arm", "string", False),
    ("model_position", "int64", False), ("model_name", "string", False),
    ("target_position", "int64", False), ("target", "string", False),
    ("primitive_position", "int64", False), ("primitive", "string", False),
    ("sensor_day_id", "int64", False), ("draw", "int64", False),
)
FIT_PROJECTION_COLUMNS = _columns(
    ("fit_seq", "int64", False), ("heldout_position", "int64", False),
    ("fold", "int64", False), ("held_out_subject", "string", False),
    ("arm_position", "int64", False), ("arm", "string", False),
    ("model_position", "int64", False), ("model_name", "string", False),
    ("target_position", "int64", False), ("target", "string", False),
    ("fit_ledger_record_digest", "string", False),
    ("preprocessor_digest", "string", False), ("model_digest", "string", False),
    ("fit_content_digest", "string", False),
)
SOURCE_PATCH_PROJECTION_COLUMNS = _columns(
    ("row_seq", "int64", False),
    ("participant_position", "int64", False),
    ("participant", "string", False),
    ("fold", "int64", False),
    ("held_out_subject", "string", False),
    ("family_position", "int64", False),
    ("family", "string", False),
    ("primitive_position", "int64", False),
    ("primitive", "string", False),
    ("sensor_day_id", "int64", False),
    ("draw", "int64", False),
    ("arm_position", "int64", False),
    ("arm", "string", False),
    ("model_position", "int64", False),
    ("model_name", "string", False),
    ("target_position", "int64", False),
    ("target", "string", False),
    ("condition_position", "int64", False),
    ("condition", "string", False),
    ("reference_context_digest", "string", False),
    ("source_row_digest", "string", False),
)

PRIMARY_COLUMNS = _columns(
    ("base_key_position", "int64", False), ("participant_position", "int64", False),
    ("participant", "string", False), ("fold", "int64", False),
    ("held_out_subject", "string", False), ("family_position", "int64", False),
    ("family", "string", False), ("arm_position", "int64", False),
    ("arm", "string", False), ("model_position", "int64", False),
    ("model_name", "string", False), ("target_position", "int64", False),
    ("target", "string", False), ("primitive_position", "int64", False),
    ("primitive", "string", False), ("sensor_day_id", "int64", False),
    ("draw", "int64", False), ("random_burden", "double", False),
    ("contiguous_burden", "double", False), ("d", "double", False),
    ("random_row_digest", "string", False), ("contiguous_row_digest", "string", False),
    ("content_digest", "string", False),
)
GROUP_COLUMNS = _columns(
    ("participant_position", "int64", False), ("participant", "string", False),
    ("family_position", "int64", False), ("family", "string", False),
    ("arm_position", "int64", False), ("arm", "string", False),
    ("model_position", "int64", False), ("model_name", "string", False),
    ("target_position", "int64", False), ("target", "string", False),
    ("contribution_count", "int64", False), ("g_median", "double", False),
    ("content_digest", "string", False),
)
PARTICIPANT_COLUMNS = _columns(
    ("participant_position", "int64", False), ("participant", "string", False),
    ("group_cell_count", "int64", False), ("p_median", "double", False),
    ("direction_positive", "bool", False), ("content_digest", "string", False),
)
BOOTSTRAP_INDEX_COLUMNS = _columns(
    ("bootstrap_draw", "int64", False), ("roster_slot", "int64", False),
    ("participant_position", "int64", False),
)
BOOTSTRAP_STAT_COLUMNS = _columns(
    ("bootstrap_draw", "int64", False), ("statistic", "double", False),
)
SIGN_LEDGER_COLUMNS = _columns(
    ("sign_index", "int64", False), ("participant_position", "int64", False),
    ("sign", "int64", False),
)
SIGN_STAT_COLUMNS = _columns(
    ("sign_index", "int64", False), ("statistic", "double", False),
    ("absolute_extreme", "bool", False),
)
CONDITION_GROUP_COLUMNS = _columns(
    ("participant_position", "int64", False), ("participant", "string", False),
    ("family_position", "int64", False), ("family", "string", False),
    ("arm_position", "int64", False), ("arm", "string", False),
    ("model_position", "int64", False), ("model_name", "string", False),
    ("target_position", "int64", False), ("target", "string", False),
    ("condition_position", "int64", False), ("condition", "string", False),
    ("expected_count", "int64", False), ("applicable_count", "int64", False),
    ("structurally_na_count", "int64", False), ("label_available_count", "int64", False),
    ("finite_count", "int64", False), ("abstained_count", "int64", False),
    ("stable_count", "int64", False), ("flipped_count", "int64", False),
    ("label_available_finite_count", "int64", False), ("coverage", "double", True),
    ("coverage_null_reason", "string", True), ("abstention_rate", "double", True),
    ("abstention_null_reason", "string", True), ("flip_rate_finite", "double", True),
    ("flip_rate_finite_null_reason", "string", True),
    ("flip_fraction_applicable", "double", True),
    ("flip_fraction_applicable_null_reason", "string", True),
    ("finite_abs_change_median", "double", True),
    ("finite_abs_change_null_reason", "string", True),
    ("reference_brier_mean", "double", True), ("perturbed_brier_mean", "double", True),
    ("brier_delta_mean", "double", True), ("reference_log_loss_mean", "double", True),
    ("perturbed_log_loss_mean", "double", True), ("log_loss_delta_mean", "double", True),
    ("performance_null_reason", "string", True), ("content_digest", "string", False),
)
PARTICIPANT_CONDITION_COLUMNS = _columns(
    ("participant_position", "int64", False), ("participant", "string", False),
    ("condition_position", "int64", False), ("condition", "string", False),
    ("metric", "string", False), ("value", "double", True),
    ("nonnull_group_count", "int64", False), ("total_group_count", "int64", False),
    ("null_reason", "string", True),
)
GLOBAL_CONDITION_COLUMNS = _columns(
    ("condition_position", "int64", False), ("condition", "string", False),
    ("metric", "string", False), ("value", "double", True),
    ("nonnull_participant_count", "int64", False),
    ("total_participant_count", "int64", False), ("null_reason", "string", True),
)
DECISION_COLUMNS = _columns(
    ("integrity_complete", "bool", False), ("label", "string", False),
    ("t_observed", "double", True), ("bootstrap_lower", "double", True),
    ("bootstrap_upper", "double", True), ("direction_positive_count", "int64", True),
    ("sign_extreme_numerator", "int64", True), ("sign_denominator", "int64", True),
    ("sign_p_value", "double", True), ("integrity_issue_count", "int64", False),
    ("integrity_issues_json", "string", False), ("sensitivity_wording", "string", False),
    ("task12_decision", "string", False), ("content_digest", "string", False),
)

TABLE_COLUMNS: Mapping[str, tuple[tuple[str, str, bool], ...]] = {
    "primary_cell_contrasts": PRIMARY_COLUMNS,
    "group_cells": GROUP_COLUMNS,
    "participant_summaries": PARTICIPANT_COLUMNS,
    "bootstrap_indices": BOOTSTRAP_INDEX_COLUMNS,
    "bootstrap_statistics": BOOTSTRAP_STAT_COLUMNS,
    "sign_ledger": SIGN_LEDGER_COLUMNS,
    "sign_statistics": SIGN_STAT_COLUMNS,
    "condition_group_descriptives": CONDITION_GROUP_COLUMNS,
    "participant_condition_descriptives": PARTICIPANT_CONDITION_COLUMNS,
    "global_condition_descriptives": GLOBAL_CONDITION_COLUMNS,
    "decision": DECISION_COLUMNS,
}


def _arrow_schema(columns: tuple[tuple[str, str, bool], ...]) -> pa.Schema:
    arrow_types = {"string": pa.string(), "bool": pa.bool_(), "int64": pa.int64(), "double": pa.float64()}
    return pa.schema([pa.field(name, arrow_types[type_name], nullable=nullable) for name, type_name, nullable in columns])


OBSERVED_PREDICTION_SCHEMA = _arrow_schema(OBSERVED_PREDICTION_COLUMNS)
TABLE_SCHEMAS = {name: _arrow_schema(columns) for name, columns in TABLE_COLUMNS.items()}


@dataclass(frozen=True, slots=True)
class ObservedPredictionInputPin:
    schema_name: str
    row_count: int
    columns: tuple[tuple[str, str, bool], ...]
    byte_count: int
    physical_sha256: str
    logical_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamStatisticsInputAuthority:
    fold_roster: tuple[tuple[int, str], ...]
    families: tuple[str, ...]
    primitive_family_pairs: tuple[tuple[str, str], ...]
    arms: tuple[str, ...]
    models: tuple[str, ...]
    targets: tuple[str, ...]
    conditions: tuple[str, ...]
    expected_primary_key_count: int
    expected_primary_key_digest: str
    max_primary_keys_per_group: int
    observed_prediction_pin: ObservedPredictionInputPin
    source_patch_projection_row_count: int
    source_patch_projection_digest: str
    fit_ledger_row_count: int
    fit_state_index_row_count: int
    fit_state_record_count: int
    fit_ledger_projection_digest: str
    fit_state_index_logical_digest: str
    fit_state_pack_sha256: str
    fit_authority_content_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class StatisticsTableSeal:
    name: str
    relative_path: str
    row_count: int
    columns: tuple[tuple[str, str, bool], ...]
    byte_count: int
    physical_sha256: str
    logical_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamStatisticsTables:
    primary_cell_contrasts: StatisticsTableSeal
    group_cells: StatisticsTableSeal
    participant_summaries: StatisticsTableSeal
    bootstrap_indices: StatisticsTableSeal
    bootstrap_statistics: StatisticsTableSeal
    sign_ledger: StatisticsTableSeal
    sign_statistics: StatisticsTableSeal
    condition_group_descriptives: StatisticsTableSeal
    participant_condition_descriptives: StatisticsTableSeal
    global_condition_descriptives: StatisticsTableSeal
    decision: StatisticsTableSeal


@dataclass(frozen=True, slots=True)
class DownstreamStatisticsResult:
    label: str
    integrity_complete: bool
    integrity_issues: tuple[str, ...]
    authority_digest: str
    tables: DownstreamStatisticsTables
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamStatisticsHandoff:
    handoff_dir: Path
    manifest_snapshot_bytes: bytes
    manifest_physical_sha256: str
    authority_digest: str
    result_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class StatisticsArtifactSeal:
    name: str
    relative_path: str
    row_count: int
    columns: tuple[tuple[str, str, bool], ...]
    byte_count: int
    physical_sha256: str
    logical_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamStatisticsManifest:
    schema_name: str
    label: str
    integrity_complete: bool
    integrity_issues: tuple[str, ...]
    authority_digest: str
    result_digest: str
    artifacts: tuple[StatisticsArtifactSeal, ...]
    content_digest: str


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_payload(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_digest(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_digest(value: object, name: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{name} must be an exact lowercase SHA-256 digest")
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
    raise ValueError(f"unsupported canonical scalar: {type(value)!r}")


class _LogicalHasher:
    def __init__(self, domain: str, columns: tuple[tuple[str, str, bool], ...]) -> None:
        self._columns = columns
        self._hash = hashlib.sha256()
        self._hash.update(domain.encode("ascii") + b"\n")
        self._hash.update(_canonical([list(column) for column in columns]) + b"\n")
        self.row_count = 0

    def update(self, row: Mapping[str, object]) -> None:
        self._hash.update(_canonical([_token(row[name]) for name, _, _ in self._columns]) + b"\n")
        self.row_count += 1

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _row_digest(tag: str, row: Mapping[str, object], names: Sequence[str]) -> str:
    return _sha_payload([tag, *(_token(row[name]) for name in names)])


_OBSERVED_DIGEST_NAMES = tuple(name for name, _, _ in OBSERVED_PREDICTION_COLUMNS if name != "row_content_digest")
_PRIMARY_DIGEST_NAMES = tuple(name for name, _, _ in PRIMARY_COLUMNS if name != "content_digest")
_GROUP_DIGEST_NAMES = tuple(name for name, _, _ in GROUP_COLUMNS if name != "content_digest")
_PARTICIPANT_DIGEST_NAMES = tuple(name for name, _, _ in PARTICIPANT_COLUMNS if name != "content_digest")
_CONDITION_DIGEST_NAMES = tuple(name for name, _, _ in CONDITION_GROUP_COLUMNS if name != "content_digest")
_DECISION_DIGEST_NAMES = tuple(name for name, _, _ in DECISION_COLUMNS if name != "content_digest")


def _audit_file_open(path: Path, mode: str) -> None:
    hook = _FILE_OPEN_AUDIT_HOOK.get()
    if hook is not None:
        hook(path, mode)


def _hash_stream(source) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    source.seek(0)
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        byte_count += len(chunk)
    return byte_count, digest.hexdigest()


def _stat_signature(file_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_nlink,
        stat.S_IFMT(file_stat.st_mode),
    )


class _StableReader:
    def __init__(
        self,
        path: Path,
        expected_signature: tuple[int, int, int, int, int, int] | None = None,
    ) -> None:
        self.path = path
        self._expected_signature = expected_signature
        self.stream = None
        self._initial_stat: os.stat_result | None = None

    def __enter__(self) -> "_StableReader":
        if not isinstance(self.path, Path):
            raise TypeError("stable-read path must be pathlib.Path")
        if _is_link_or_reparse(self.path):
            raise ValueError("stable-read path is a link or reparse point")
        try:
            path_stat = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise ValueError("stable-read path cannot be inspected") from error
        if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
            raise ValueError("stable-read path must be a single-link regular file")
        if (
            self._expected_signature is not None
            and _stat_signature(path_stat) != self._expected_signature
        ):
            raise ValueError("stable-read path changed after directory enumeration")
        hook = _STABLE_READ_AUDIT_HOOK.get()
        if hook is not None:
            hook("after_path_check_before_open", self.path, None)
        _audit_file_open(self.path, "rb")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError as error:
            raise ValueError("stable-read open failed") from error
        try:
            opened_stat = os.fstat(descriptor)
            current_path_stat = os.stat(self.path, follow_symlinks=False)
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or opened_stat.st_nlink != 1
                or _stat_signature(opened_stat) != _stat_signature(path_stat)
                or _stat_signature(current_path_stat) != _stat_signature(opened_stat)
                or _is_link_or_reparse(self.path)
            ):
                raise ValueError("stable-read pathname/open-handle identity mismatch")
            self._initial_stat = opened_stat
            self.stream = os.fdopen(descriptor, "rb", closefd=True)
        except BaseException:
            os.close(descriptor)
            raise
        return self

    @property
    def identity(self) -> tuple[int, int]:
        if self._initial_stat is None:
            raise RuntimeError("stable reader is not open")
        return self._initial_stat.st_dev, self._initial_stat.st_ino

    def checkpoint(self, phase: str) -> None:
        if self.stream is None or self._initial_stat is None:
            raise RuntimeError("stable reader is not open")
        hook = _STABLE_READ_AUDIT_HOOK.get()
        if hook is not None:
            hook(phase, self.path, self.stream)
        try:
            handle_stat = os.fstat(self.stream.fileno())
            path_stat = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise ValueError("stable-read identity recheck failed") from error
        if (
            _is_link_or_reparse(self.path)
            or _stat_signature(handle_stat) != _stat_signature(self._initial_stat)
            or _stat_signature(path_stat) != _stat_signature(self._initial_stat)
        ):
            raise ValueError("stable-read file changed during use")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type is None:
                self.checkpoint("after_hash_and_parse")
        finally:
            if self.stream is not None:
                self.stream.close()


def _stable_canonical_json(
    path: Path,
    expected_signature: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[bytes, str, object]:
    with _StableReader(path, expected_signature) as stable:
        stable.stream.seek(0)
        data = stable.stream.read()
        if not data.endswith(b"\n") or data.endswith(b"\n\n"):
            raise ValueError("manifest is not canonical LF-terminated JSON")
        document = _strict_json(data[:-1])
        if data != _canonical(document) + b"\n":
            raise ValueError("manifest bytes are not canonical JSON plus one LF")
        stable.checkpoint("after_parse_before_hash")
        physical = hashlib.sha256(data).hexdigest()
        stable.checkpoint("after_hash")
        return data, physical, document


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except AttributeError:
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _new_empty_directory(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("directory must be a pathlib.Path")
    if path.exists():
        raise ValueError("directory must be new and absent")
    if not path.parent.exists() or _is_link_or_reparse(path.parent):
        raise ValueError("directory parent must exist and be non-link")
    path.mkdir()
    return path


def _require_existing_directory(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_dir() or _is_link_or_reparse(path):
        raise ValueError("invalid artifact directory")
    return path


def _enumerate_exact_files(
    root: Path,
    expected_names: set[str],
) -> dict[str, tuple[int, int, int, int, int, int]]:
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != expected_names:
        raise ValueError("artifact file set mismatch")
    snapshots: dict[str, tuple[int, int, int, int, int, int]] = {}
    identities: set[tuple[int, int]] = set()
    for entry in entries:
        if _is_link_or_reparse(entry):
            raise ValueError("artifact directory contains a link or reparse point")
        try:
            file_stat = os.stat(entry, follow_symlinks=False)
        except OSError as error:
            raise ValueError("artifact file cannot be inspected") from error
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ValueError("artifact must be a single-link regular file")
        identity = (file_stat.st_dev, file_stat.st_ino)
        if identity in identities:
            raise ValueError("artifact directory contains aliased files")
        identities.add(identity)
        snapshots[entry.name] = _stat_signature(file_stat)
    hook = _STABLE_READ_AUDIT_HOOK.get()
    if hook is not None:
        hook("after_directory_enumeration", root, None)
    return snapshots


def _open_exclusive_output(path: Path):
    if not isinstance(path, Path):
        raise TypeError("output path must be pathlib.Path")
    if not path.parent.is_dir() or _is_link_or_reparse(path.parent):
        raise ValueError("output parent must be an existing non-link directory")
    _audit_file_open(path, "xb")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError("exclusive output creation failed") from error
    try:
        opened_stat = os.fstat(descriptor)
        path_stat = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or opened_stat.st_nlink != 1
            or _stat_signature(path_stat) != _stat_signature(opened_stat)
            or _is_link_or_reparse(path)
        ):
            raise ValueError("exclusive output pathname/open-handle identity mismatch")
        return os.fdopen(descriptor, "wb", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise


def _write_exclusive_bytes(path: Path, data: bytes) -> None:
    if type(data) is not bytes:
        raise TypeError("exclusive output payload must be exact bytes")
    with _open_exclusive_output(path) as sink:
        sink.write(data)
        sink.flush()
        os.fsync(sink.fileno())


def _ipc_options() -> ipc.IpcWriteOptions:
    return ipc.IpcWriteOptions(
        metadata_version=ipc.MetadataVersion.V5,
        allow_64bit=False,
        use_legacy_format=False,
        compression=None,
        use_threads=False,
        emit_dictionary_deltas=False,
        unify_dictionaries=False,
    )


class _CanonicalIPCWriter:
    def __init__(self, path: Path, schema: pa.Schema) -> None:
        self._sink = _open_exclusive_output(path)
        self._writer = ipc.new_stream(self._sink, schema, options=_ipc_options())
        self._schema = schema
        self._rows: list[dict[str, object]] = []
        self.row_count = 0
        self._closed = False

    @property
    def buffered_row_count(self) -> int:
        return len(self._rows)

    def add(self, row: dict[str, object]) -> None:
        if self._closed:
            raise RuntimeError("writer already closed")
        self._rows.append(row)
        self.row_count += 1
        if len(self._rows) == _BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if self._rows:
            self._writer.write_batch(pa.RecordBatch.from_pylist(self._rows, schema=self._schema))
            self._rows.clear()

    def close(self) -> None:
        if not self._closed:
            self._flush()
            self._writer.close()
            self._sink.close()
            self._closed = True


def _write_ipc_rows(path: Path, schema: pa.Schema, rows: Iterable[dict[str, object]]) -> None:
    writer = _CanonicalIPCWriter(path, schema)
    try:
        for row in rows:
            writer.add(row)
    finally:
        writer.close()


def _schema_columns(schema: pa.Schema) -> tuple[tuple[str, str, bool], ...]:
    return tuple((field.name, str(field.type), field.nullable) for field in schema)


def _require_exact_columns_tree(value: object, name: str) -> tuple[tuple[str, str, bool], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    for column in value:
        if (
            type(column) is not tuple
            or len(column) != 3
            or type(column[0]) is not str
            or type(column[1]) is not str
            or type(column[2]) is not bool
        ):
            raise TypeError(f"{name} contains a non-exact column declaration")
    return value


def _inspect_ipc(
    path: Path,
    expected_name: str,
    expected_signature: tuple[int, int, int, int, int, int] | None = None,
) -> tuple[int, str, int, str]:
    expected_schema = TABLE_SCHEMAS[expected_name]
    logical = _LogicalHasher("task-12-downstream-statistics-table-logical-v1", TABLE_COLUMNS[expected_name])
    row_count = 0
    with _StableReader(path, expected_signature) as stable:
        reader = ipc.open_stream(stable.stream)
        if reader.schema != expected_schema or reader.schema.metadata is not None:
            raise ValueError("IPC schema mismatch")
        saw_short = False
        for batch in reader:
            if saw_short or batch.num_rows <= 0 or batch.num_rows > _BATCH_SIZE:
                raise ValueError("noncanonical IPC partition")
            if batch.num_rows < _BATCH_SIZE:
                saw_short = True
            for offset in range(batch.num_rows):
                row = {name: batch.column(index)[offset].as_py() for index, (name, _, _) in enumerate(TABLE_COLUMNS[expected_name])}
                logical.update(row)
            row_count += batch.num_rows
        stable.checkpoint("after_parse_before_hash")
        byte_count, physical = _hash_stream(stable.stream)
        stable.checkpoint("after_hash")
        return row_count, logical.hexdigest(), byte_count, physical


def _table_seal(
    path: Path,
    name: str,
    expected_signature: tuple[int, int, int, int, int, int] | None = None,
) -> StatisticsTableSeal:
    row_count, logical_digest, byte_count, physical_sha256 = _inspect_ipc(
        path, name, expected_signature
    )
    relative_path = f"{name}.arrow"
    columns = TABLE_COLUMNS[name]
    content_digest = _sha_payload([
        "task-12-downstream-statistics-table-seal-v1", name, relative_path, row_count,
        [list(column) for column in columns], byte_count, physical_sha256, logical_digest,
    ])
    return StatisticsTableSeal(name, relative_path, row_count, columns, byte_count, physical_sha256, logical_digest, content_digest)


def _pin_digest(pin: ObservedPredictionInputPin) -> str:
    return _sha_payload([
        "task-12-downstream-observed-prediction-input-pin-v1", pin.schema_name,
        pin.row_count, [list(column) for column in pin.columns], pin.byte_count,
        pin.physical_sha256, pin.logical_digest,
    ])


def _authority_digest(authority: DownstreamStatisticsInputAuthority) -> str:
    return _sha_payload([
        "task-12-downstream-statistics-input-authority-v1",
        [list(entry) for entry in authority.fold_roster], list(authority.families),
        [list(entry) for entry in authority.primitive_family_pairs], list(authority.arms),
        list(authority.models), list(authority.targets), list(authority.conditions),
        authority.expected_primary_key_count, authority.expected_primary_key_digest,
        authority.max_primary_keys_per_group, authority.observed_prediction_pin.content_digest,
        authority.source_patch_projection_row_count, authority.source_patch_projection_digest,
        authority.fit_ledger_row_count, authority.fit_state_index_row_count,
        authority.fit_state_record_count, authority.fit_ledger_projection_digest,
        authority.fit_state_index_logical_digest, authority.fit_state_pack_sha256,
        authority.fit_authority_content_digest,
    ])


def _validate_authority(
    authority: DownstreamStatisticsInputAuthority,
    expected_authority_digest: str,
) -> set[str]:
    _require_digest(expected_authority_digest, "expected_authority_digest")
    if type(authority) is not DownstreamStatisticsInputAuthority:
        raise TypeError("authority has the wrong exact type")
    _require_digest(authority.content_digest, "content_digest")
    if authority.content_digest != expected_authority_digest:
        raise ValueError("independently held authority digest mismatch")

    def exact_tuple_of_strings(value: object, length: int | None = None) -> bool:
        return type(value) is tuple and (length is None or len(value) == length) and all(type(item) is str for item in value)

    if type(authority.fold_roster) is not tuple or len(authority.fold_roster) != 10:
        raise ValueError("fold_roster must contain exactly ten entries")
    for entry in authority.fold_roster:
        if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not int or type(entry[1]) is not str:
            raise TypeError("fold_roster contains a non-exact scalar")
    folds = [entry[0] for entry in authority.fold_roster]
    subjects = [entry[1] for entry in authority.fold_roster]
    if any(not subject for subject in subjects) or len(set(folds)) != 10 or len(set(subjects)) != 10:
        raise ValueError("fold_roster must contain unique folds and nonempty unique subjects")
    if not exact_tuple_of_strings(authority.families, 4):
        raise TypeError("families must be four exact strings")
    if any(not family for family in authority.families) or len(set(authority.families)) != 4:
        raise ValueError("families must be nonempty and unique")
    if type(authority.primitive_family_pairs) is not tuple or not authority.primitive_family_pairs:
        raise TypeError("primitive_family_pairs must be a nonempty tuple")
    for pair in authority.primitive_family_pairs:
        if type(pair) is not tuple or len(pair) != 2 or any(type(item) is not str for item in pair):
            raise TypeError("primitive_family_pairs contains a non-exact scalar")
        if pair[1] not in authority.families:
            raise ValueError("primitive family is outside the frozen family tuple")
    primitives = [pair[0] for pair in authority.primitive_family_pairs]
    represented_families = {pair[1] for pair in authority.primitive_family_pairs}
    if (
        any(not primitive for primitive in primitives)
        or len(set(primitives)) != len(primitives)
        or len(set(authority.primitive_family_pairs)) != len(authority.primitive_family_pairs)
        or represented_families != set(authority.families)
    ):
        raise ValueError("primitive/family mapping must be unique, nonempty, and cover every family")
    for name in ("arms", "models", "targets", "conditions"):
        if not exact_tuple_of_strings(getattr(authority, name)):
            raise TypeError(f"{name} contains a non-exact string")
    if authority.arms != ARMS or authority.models != MODELS or authority.targets != TARGETS or authority.conditions != CONDITIONS:
        raise ValueError("frozen axis mismatch")
    for name in (
        "expected_primary_key_count", "max_primary_keys_per_group",
        "source_patch_projection_row_count", "fit_ledger_row_count",
        "fit_state_index_row_count", "fit_state_record_count",
    ):
        if type(getattr(authority, name)) is not int:
            raise TypeError(f"{name} must be an exact built-in int")
    if authority.expected_primary_key_count <= 0 or authority.max_primary_keys_per_group <= 0:
        raise ValueError("authority counts must be positive")
    for name in (
        "expected_primary_key_digest", "fit_ledger_projection_digest",
        "source_patch_projection_digest",
        "fit_state_index_logical_digest", "fit_state_pack_sha256",
        "fit_authority_content_digest", "content_digest",
    ):
        _require_digest(getattr(authority, name), name)
    pin = authority.observed_prediction_pin
    if type(pin) is not ObservedPredictionInputPin:
        raise TypeError("observed prediction pin has the wrong exact type")
    if type(pin.schema_name) is not str or pin.schema_name != "task-12-downstream-observed-prediction-input-v1":
        raise ValueError("observed prediction schema name mismatch")
    if type(pin.row_count) is not int or type(pin.byte_count) is not int or pin.row_count < 0 or pin.byte_count < 0:
        raise TypeError("observed prediction pin counts must be exact nonnegative ints")
    _require_exact_columns_tree(pin.columns, "observed_prediction_pin.columns")
    if pin.columns != OBSERVED_PREDICTION_COLUMNS:
        raise ValueError("observed prediction pin columns mismatch")
    for name in ("physical_sha256", "logical_digest", "content_digest"):
        _require_digest(getattr(pin, name), f"observed_prediction_pin.{name}")
    if pin.content_digest != _pin_digest(pin):
        raise ValueError("observed prediction pin self-digest mismatch")
    if authority.content_digest != _authority_digest(authority):
        raise ValueError("authority self-digest mismatch")
    if pin.row_count != 3 * authority.expected_primary_key_count:
        raise ValueError("observed prediction pin topology mismatch")
    if authority.source_patch_projection_row_count != pin.row_count:
        raise ValueError("source-patch projection row-count mismatch")
    issues: set[str] = set()
    if (authority.fit_ledger_row_count, authority.fit_state_index_row_count, authority.fit_state_record_count) != (160, 160, 160):
        issues.add("fit_topology_incomplete")
    return issues


def _ordered_issues(issues: set[str]) -> tuple[str, ...]:
    unknown = issues.difference(ISSUE_CODES)
    if unknown:
        raise RuntimeError(f"unregistered integrity issues: {sorted(unknown)!r}")
    return tuple(code for code in ISSUE_CODES if code in issues)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _log_loss(probability: float, label: int) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return float(-(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped)))


def _same_finite_float_bits(left: object, right: object) -> bool:
    return (
        type(left) is float
        and type(right) is float
        and math.isfinite(left)
        and math.isfinite(right)
        and left.hex() == right.hex()
    )


def _new_condition_accumulator() -> dict[str, object]:
    return {
        "expected_count": 0, "applicable_count": 0, "structurally_na_count": 0,
        "label_available_count": 0, "finite_count": 0, "abstained_count": 0,
        "stable_count": 0, "flipped_count": 0, "label_available_finite_count": 0,
        "absolute_changes": [], "reference_brier": [], "perturbed_brier": [],
        "brier_delta": [], "reference_log_loss": [], "perturbed_log_loss": [],
        "log_loss_delta": [],
    }


def _condition_row(meta: Mapping[str, object], condition_position: int, acc: Mapping[str, object]) -> dict[str, object]:
    applicable = int(acc["applicable_count"])
    finite = int(acc["finite_count"])
    performance_n = int(acc["label_available_finite_count"])
    flipped = int(acc["flipped_count"])
    if applicable:
        coverage, coverage_reason = finite / applicable, None
        abstention, abstention_reason = int(acc["abstained_count"]) / applicable, None
        flip_applicable, flip_applicable_reason = flipped / applicable, None
    else:
        coverage = abstention = flip_applicable = None
        coverage_reason = abstention_reason = flip_applicable_reason = "no_applicable_predictions"
    if finite:
        flip_finite, flip_finite_reason = flipped / finite, None
        absolute_median = _median(acc["absolute_changes"])
        absolute_reason = None
    else:
        flip_finite = absolute_median = None
        flip_finite_reason = absolute_reason = "no_finite_predictions"
    if performance_n:
        reference_brier = _mean(acc["reference_brier"])
        perturbed_brier = _mean(acc["perturbed_brier"])
        brier_delta = _mean(acc["brier_delta"])
        reference_ll = _mean(acc["reference_log_loss"])
        perturbed_ll = _mean(acc["perturbed_log_loss"])
        ll_delta = _mean(acc["log_loss_delta"])
        performance_reason = None
    else:
        reference_brier = perturbed_brier = brier_delta = None
        reference_ll = perturbed_ll = ll_delta = None
        performance_reason = "no_finite_label_available_predictions"
    row: dict[str, object] = {
        "participant_position": meta["participant_position"], "participant": meta["participant"],
        "family_position": meta["family_position"], "family": meta["family"],
        "arm_position": meta["arm_position"], "arm": meta["arm"],
        "model_position": meta["model_position"], "model_name": meta["model_name"],
        "target_position": meta["target_position"], "target": meta["target"],
        "condition_position": condition_position, "condition": CONDITIONS[condition_position],
        "expected_count": int(acc["expected_count"]), "applicable_count": applicable,
        "structurally_na_count": int(acc["structurally_na_count"]),
        "label_available_count": int(acc["label_available_count"]), "finite_count": finite,
        "abstained_count": int(acc["abstained_count"]), "stable_count": int(acc["stable_count"]),
        "flipped_count": flipped, "label_available_finite_count": performance_n,
        "coverage": coverage, "coverage_null_reason": coverage_reason,
        "abstention_rate": abstention, "abstention_null_reason": abstention_reason,
        "flip_rate_finite": flip_finite, "flip_rate_finite_null_reason": flip_finite_reason,
        "flip_fraction_applicable": flip_applicable,
        "flip_fraction_applicable_null_reason": flip_applicable_reason,
        "finite_abs_change_median": absolute_median,
        "finite_abs_change_null_reason": absolute_reason,
        "reference_brier_mean": reference_brier, "perturbed_brier_mean": perturbed_brier,
        "brier_delta_mean": brier_delta, "reference_log_loss_mean": reference_ll,
        "perturbed_log_loss_mean": perturbed_ll, "log_loss_delta_mean": ll_delta,
        "performance_null_reason": performance_reason,
    }
    row["content_digest"] = _row_digest(
        "task-12-downstream-condition-group-descriptive-row-v1", row, _CONDITION_DIGEST_NAMES
    )
    return row


def _validate_position_contract(row: Mapping[str, object], authority: DownstreamStatisticsInputAuthority, issues: set[str]) -> bool:
    try:
        participant_position = row["participant_position"]
        family_position = row["family_position"]
        primitive_position = row["primitive_position"]
        arm_position = row["arm_position"]
        model_position = row["model_position"]
        target_position = row["target_position"]
        condition_position = row["condition_position"]
        valid = (
            type(participant_position) is int and 0 <= participant_position < 10
            and authority.fold_roster[participant_position] == (row["fold"], row["participant"])
            and row["held_out_subject"] == row["participant"]
            and type(family_position) is int and 0 <= family_position < 4
            and authority.families[family_position] == row["family"]
            and type(primitive_position) is int and 0 <= primitive_position < len(authority.primitive_family_pairs)
            and authority.primitive_family_pairs[primitive_position] == (row["primitive"], row["family"])
            and type(arm_position) is int and 0 <= arm_position < 2 and authority.arms[arm_position] == row["arm"]
            and type(model_position) is int and 0 <= model_position < 2 and authority.models[model_position] == row["model_name"]
            and type(target_position) is int and 0 <= target_position < 4 and authority.targets[target_position] == row["target"]
            and type(condition_position) is int and 0 <= condition_position < 3
            and authority.conditions[condition_position] == row["condition"]
        )
    except (IndexError, KeyError, TypeError):
        valid = False
    if not valid:
        issues.add("observed_input_order_mismatch")
    return valid


def _validate_row_contract(row: Mapping[str, object], issues: set[str]) -> str:
    for digest_name in (
        "fit_ledger_record_digest", "preprocessor_digest", "model_digest",
        "fit_content_digest", "source_row_digest", "reference_context_digest",
        "row_content_digest",
    ):
        if not _is_digest(row[digest_name]):
            issues.add("source_provenance_crosswire" if digest_name in {"source_row_digest", "reference_context_digest"} else "fit_identity_crosswire")
    reference_probability = row["reference_probability"]
    reference_valid = (
        row["pre_mask_eligible"] is True
        and row["reference_status"] == "finite_baseline_admissible"
        and type(reference_probability) is float and math.isfinite(reference_probability)
        and 0.0 <= reference_probability <= 1.0
        and type(row["reference_decision"]) is bool
        and row["reference_decision"] == (reference_probability >= 0.5)
    )
    if not reference_valid:
        issues.add("reference_contract_invalid")

    label = row["label"]
    label_valid = (
        (row["label_available"] is True and type(label) is int and not isinstance(label, bool) and label in (0, 1))
        or (row["label_available"] is False and label is None)
    )
    if not label_valid:
        issues.add("label_contract_invalid")

    status = row["replay_status"]
    if not reference_valid:
        return "invalid"
    reference_decision = row["reference_decision"]
    if status == "finite":
        probability = row["perturbed_probability"]
        valid = (
            row["applicable"] is True and row["coverage_eligible"] is True
            and type(probability) is float and math.isfinite(probability) and 0.0 <= probability <= 1.0
            and type(row["perturbed_decision"]) is bool
            and row["perturbed_decision"] == (probability >= 0.5)
            and type(row["absolute_probability_change"]) is float
            and _same_finite_float_bits(
                row["absolute_probability_change"], abs(probability - reference_probability)
            )
            and type(row["decision_flip"]) is bool
            and row["decision_flip"] == (row["perturbed_decision"] != reference_decision)
            and type(row["instability_burden"]) is float
            and _same_finite_float_bits(
                row["instability_burden"], abs(probability - reference_probability)
            )
        )
        category = "finite"
    elif status in ("abstained_no_event_support", "abstained_insufficient_support"):
        valid = (
            row["applicable"] is True and row["coverage_eligible"] is True
            and row["perturbed_probability"] is None and row["perturbed_decision"] is None
            and row["absolute_probability_change"] is None and row["decision_flip"] is None
            and type(row["instability_burden"]) is float and row["instability_burden"] == 1.0
        )
        category = "abstained"
    elif status == "structurally_unavailable":
        valid = (
            row["condition"] == CONDITIONS[2] and row["applicable"] is False
            and row["coverage_eligible"] is False and row["perturbed_probability"] is None
            and row["perturbed_decision"] is None and row["absolute_probability_change"] is None
            and row["decision_flip"] is None and row["instability_burden"] is None
        )
        category = "structural"
    elif status in ("invalid_contract_or_provenance", "execution_failure"):
        valid = (
            row["perturbed_probability"] is None and row["perturbed_decision"] is None
            and row["absolute_probability_change"] is None and row["decision_flip"] is None
            and row["instability_burden"] is None
        )
        category = "invalid"
        issues.add("status_contract_invalid")
    else:
        valid = False
        category = "invalid"
    if not valid:
        issues.add("status_contract_invalid")
        return "invalid"
    return category


def _update_condition_accumulator(acc: dict[str, object], row: Mapping[str, object], category: str) -> None:
    acc["expected_count"] += 1
    if row["label_available"] is True:
        acc["label_available_count"] += 1
    if category == "structural":
        acc["structurally_na_count"] += 1
        return
    if category == "invalid":
        return
    acc["applicable_count"] += 1
    if category == "abstained":
        acc["abstained_count"] += 1
        return
    acc["finite_count"] += 1
    if row["decision_flip"]:
        acc["flipped_count"] += 1
    else:
        acc["stable_count"] += 1
    acc["absolute_changes"].append(row["absolute_probability_change"])
    if row["label_available"] is True:
        acc["label_available_finite_count"] += 1
        label = row["label"]
        reference_probability = row["reference_probability"]
        perturbed_probability = row["perturbed_probability"]
        reference_brier = float((reference_probability - label) ** 2)
        perturbed_brier = float((perturbed_probability - label) ** 2)
        reference_ll = _log_loss(reference_probability, label)
        perturbed_ll = _log_loss(perturbed_probability, label)
        acc["reference_brier"].append(reference_brier)
        acc["perturbed_brier"].append(perturbed_brier)
        acc["brier_delta"].append(float(perturbed_brier - reference_brier))
        acc["reference_log_loss"].append(reference_ll)
        acc["perturbed_log_loss"].append(perturbed_ll)
        acc["log_loss_delta"].append(float(perturbed_ll - reference_ll))


def analyze_downstream_statistics(
    observed_prediction_batches: Iterable[pa.RecordBatch],
    authority: DownstreamStatisticsInputAuthority,
    handoff_dir: Path,
    *,
    expected_authority_digest: str,
) -> DownstreamStatisticsHandoff:
    issues = _validate_authority(authority, expected_authority_digest)
    if not hasattr(observed_prediction_batches, "__iter__"):
        raise TypeError("observed_prediction_batches must be an iterable")
    handoff_dir = _new_empty_directory(handoff_dir)

    observed_hasher = _LogicalHasher(
        "task-12-downstream-observed-prediction-logical-v1", OBSERVED_PREDICTION_COLUMNS
    )
    key_hasher = _LogicalHasher(
        "task-12-downstream-expected-primary-key-digest-v1", EXPECTED_KEY_COLUMNS
    )
    source_patch_hasher = _LogicalHasher(
        "task-12-downstream-source-patch-projection-digest-v1",
        SOURCE_PATCH_PROJECTION_COLUMNS,
    )
    fit_records: dict[int, dict[str, object]] = {}
    logical_invalid = False
    row_count = 0
    base_buffer: list[tuple[dict[str, object], str]] = []
    previous_order: tuple[object, ...] | None = None
    provisional_table_names = (
        "primary_cell_contrasts", "group_cells", "participant_summaries",
        "condition_group_descriptives", "participant_condition_descriptives",
    )
    provisional_writers = {
        name: _CanonicalIPCWriter(handoff_dir / f"{name}.arrow", TABLE_SCHEMAS[name])
        for name in provisional_table_names
    }
    primary_writer = provisional_writers["primary_cell_contrasts"]

    current_group_key: tuple[object, ...] | None = None
    current_group_meta: dict[str, object] | None = None
    current_group_d: list[float] = []
    current_group_accs = [_new_condition_accumulator() for _ in range(3)]
    current_group_base_count = 0
    group_ordinal = 0
    current_participant_position: int | None = None
    participant_group_buffer: list[dict[str, object]] = []
    participant_condition_buffer: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []
    participant_condition_row_count = 0
    global_values: list[list[list[float]]] = [[[] for _ in METRICS] for _ in range(3)]

    def audit_state() -> None:
        hook = _STATE_AUDIT_HOOK.get()
        if hook is not None:
            hook({
                "current_group_contributions": len(current_group_d),
                "current_participant_group_rows": len(participant_group_buffer),
                "current_participant_condition_rows": len(participant_condition_buffer),
                "fit_records": len(fit_records),
                "largest_output_buffer": max(
                    writer.buffered_row_count for writer in provisional_writers.values()
                ),
            })

    def finalize_participant() -> None:
        nonlocal current_participant_position, participant_group_buffer
        nonlocal participant_condition_buffer, participant_condition_row_count
        if current_participant_position is None:
            return
        participant = authority.fold_roster[current_participant_position][1]
        if len(participant_group_buffer) != 64 or len(participant_condition_buffer) != 192:
            issues.add("participant_grid_incomplete")
        else:
            p_median = _median([row["g_median"] for row in participant_group_buffer])
            participant_row: dict[str, object] = {
                "participant_position": current_participant_position,
                "participant": participant,
                "group_cell_count": 64,
                "p_median": p_median,
                "direction_positive": p_median > 0.0,
            }
            participant_row["content_digest"] = _row_digest(
                "task-12-downstream-participant-summary-row-v1",
                participant_row,
                _PARTICIPANT_DIGEST_NAMES,
            )
            provisional_writers["participant_summaries"].add(participant_row)
            participant_rows.append(participant_row)
            for condition_position, condition in enumerate(CONDITIONS):
                condition_groups = [
                    row for row in participant_condition_buffer
                    if row["condition_position"] == condition_position
                ]
                if len(condition_groups) != 64:
                    issues.add("participant_grid_incomplete")
                for metric_position, metric in enumerate(METRICS):
                    values = [row[metric] for row in condition_groups if row[metric] is not None]
                    value = _median(values) if values else None
                    participant_condition_row = {
                        "participant_position": current_participant_position,
                        "participant": participant,
                        "condition_position": condition_position,
                        "condition": condition,
                        "metric": metric,
                        "value": value,
                        "nonnull_group_count": len(values),
                        "total_group_count": 64,
                        "null_reason": None if values else "no_nonnull_group_values",
                    }
                    provisional_writers["participant_condition_descriptives"].add(
                        participant_condition_row
                    )
                    participant_condition_row_count += 1
                    if value is not None:
                        global_values[condition_position][metric_position].append(value)
        participant_group_buffer = []
        participant_condition_buffer = []
        current_participant_position = None
        audit_state()

    def finalize_group() -> None:
        nonlocal current_group_key, current_group_meta, current_group_d
        nonlocal current_group_accs, current_group_base_count
        nonlocal group_ordinal
        if current_group_key is None or current_group_meta is None:
            return
        remaining = group_ordinal % 64
        expected_group_key = (
            group_ordinal // 64,
            remaining // 16,
            (remaining % 16) // 8,
            (remaining % 8) // 4,
            remaining % 4,
        )
        if group_ordinal >= 640 or current_group_key != expected_group_key:
            issues.add("group_grid_incomplete")
        if not current_group_d:
            issues.add("group_grid_incomplete")
        else:
            group_row: dict[str, object] = {
                **current_group_meta,
                "contribution_count": len(current_group_d),
                "g_median": _median(current_group_d),
            }
            group_row["content_digest"] = _row_digest(
                "task-12-downstream-group-cell-row-v1", group_row, _GROUP_DIGEST_NAMES
            )
            provisional_writers["group_cells"].add(group_row)
            participant_group_buffer.append(group_row)
        for condition_position, accumulator in enumerate(current_group_accs):
            if (
                int(accumulator["expected_count"])
                != int(accumulator["applicable_count"]) + int(accumulator["structurally_na_count"])
                or int(accumulator["applicable_count"])
                != int(accumulator["finite_count"]) + int(accumulator["abstained_count"])
                or int(accumulator["finite_count"])
                != int(accumulator["stable_count"]) + int(accumulator["flipped_count"])
            ):
                issues.add("denominator_incomplete")
            condition_row = _condition_row(current_group_meta, condition_position, accumulator)
            provisional_writers["condition_group_descriptives"].add(condition_row)
            participant_condition_buffer.append(condition_row)
        group_ordinal += 1
        current_group_key = None
        current_group_meta = None
        current_group_d = []
        current_group_accs = [_new_condition_accumulator() for _ in range(3)]
        current_group_base_count = 0
        audit_state()

    def process_base(items: list[tuple[dict[str, object], str]]) -> None:
        nonlocal current_group_key, current_group_meta, current_group_base_count
        nonlocal current_participant_position
        if len(items) != 3:
            issues.add("observed_input_missing_row")
            return
        rows = [item[0] for item in items]
        categories = [item[1] for item in items]
        positions = [row["condition_position"] for row in rows]
        if positions != [0, 1, 2]:
            issues.add("observed_input_order_mismatch")
            if len(set(positions)) != len(positions):
                issues.add("observed_input_duplicate_row")
            if set(positions) != {0, 1, 2}:
                issues.add("observed_input_missing_row")
        cross_condition_names = (
            "participant_position", "participant", "fold", "held_out_subject",
            "family_position", "family", "primitive_position", "primitive",
            "sensor_day_id", "draw", "arm_position", "arm", "model_position",
            "model_name", "target_position", "target", "fit_seq", "pre_mask_eligible",
            "reference_status", "reference_decision",
            "label_available", "label", "fit_ledger_record_digest", "preprocessor_digest",
            "model_digest", "fit_content_digest", "reference_context_digest",
        )
        for name in cross_condition_names:
            if not (rows[0][name] == rows[1][name] == rows[2][name]):
                if name == "reference_context_digest":
                    issues.add("source_provenance_crosswire")
                elif name in {
                    "fit_seq", "fold", "held_out_subject", "arm_position", "arm",
                    "model_position", "model_name", "target_position", "target",
                    "fit_ledger_record_digest", "preprocessor_digest", "model_digest",
                    "fit_content_digest",
                }:
                    issues.add("fit_identity_crosswire")
                else:
                    issues.add("observed_input_order_mismatch")
        reference_probabilities = [row["reference_probability"] for row in rows]
        if not (
            _same_finite_float_bits(reference_probabilities[0], reference_probabilities[1])
            and _same_finite_float_bits(reference_probabilities[0], reference_probabilities[2])
        ):
            issues.add("reference_contract_invalid")
        source_digests = [row["source_row_digest"] for row in rows]
        if len(set(source_digests)) != 3:
            issues.add("source_provenance_crosswire")

        first = rows[0]
        key_row = {name: first[name] for name, _, _ in EXPECTED_KEY_COLUMNS}
        try:
            key_hasher.update(key_row)
        except ValueError:
            issues.add("primary_key_digest_mismatch")

        group_key = (
            first["participant_position"], first["family_position"], first["arm_position"],
            first["model_position"], first["target_position"],
        )
        if current_group_key is not None and group_key != current_group_key:
            finalize_group()
        next_participant_position = first["participant_position"]
        if (
            current_participant_position is not None
            and next_participant_position != current_participant_position
        ):
            finalize_participant()
        if current_participant_position is None and type(next_participant_position) is int:
            current_participant_position = next_participant_position
        if current_group_key is None:
            current_group_key = group_key
            current_group_meta = {
                "participant_position": first["participant_position"],
                "participant": first["participant"], "family_position": first["family_position"],
                "family": first["family"], "arm_position": first["arm_position"],
                "arm": first["arm"], "model_position": first["model_position"],
                "model_name": first["model_name"], "target_position": first["target_position"],
                "target": first["target"],
            }
        current_group_base_count += 1
        within_capacity = current_group_base_count <= authority.max_primary_keys_per_group
        if not within_capacity:
            issues.add("group_capacity_exceeded")
        if within_capacity:
            for condition_position in range(3):
                candidate = next(
                    ((row, category) for row, category in zip(rows, categories) if row["condition_position"] == condition_position),
                    None,
                )
                if candidate is None:
                    issues.add("denominator_incomplete")
                    continue
                _update_condition_accumulator(current_group_accs[condition_position], candidate[0], candidate[1])

        by_condition = {row["condition_position"]: row for row in rows if row["condition_position"] in (0, 1, 2)}
        random_row = by_condition.get(0)
        contiguous_row = by_condition.get(1)
        if (
            random_row is None or contiguous_row is None
            or type(random_row["instability_burden"]) is not float
            or type(contiguous_row["instability_burden"]) is not float
            or categories[rows.index(random_row)] == "invalid"
            or categories[rows.index(contiguous_row)] == "invalid"
        ):
            issues.add("denominator_incomplete")
            return
        random_burden = random_row["instability_burden"]
        contiguous_burden = contiguous_row["instability_burden"]
        d_value = float(contiguous_burden - random_burden)
        if within_capacity:
            current_group_d.append(d_value)
        primary_row: dict[str, object] = {
            "base_key_position": first["row_seq"] // 3 if type(first["row_seq"]) is int else -1,
            "participant_position": first["participant_position"], "participant": first["participant"],
            "fold": first["fold"], "held_out_subject": first["held_out_subject"],
            "family_position": first["family_position"], "family": first["family"],
            "arm_position": first["arm_position"], "arm": first["arm"],
            "model_position": first["model_position"], "model_name": first["model_name"],
            "target_position": first["target_position"], "target": first["target"],
            "primitive_position": first["primitive_position"], "primitive": first["primitive"],
            "sensor_day_id": first["sensor_day_id"], "draw": first["draw"],
            "random_burden": random_burden, "contiguous_burden": contiguous_burden,
            "d": d_value, "random_row_digest": random_row["row_content_digest"],
            "contiguous_row_digest": contiguous_row["row_content_digest"],
        }
        primary_row["content_digest"] = _row_digest(
            "task-12-downstream-primary-contrast-row-v1", primary_row, _PRIMARY_DIGEST_NAMES
        )
        primary_writer.add(primary_row)
        audit_state()

    try:
        for batch in observed_prediction_batches:
            if type(batch) is not pa.RecordBatch:
                raise TypeError("every input item must be an exact pyarrow.RecordBatch")
            if batch.schema != OBSERVED_PREDICTION_SCHEMA or batch.schema.metadata is not None:
                issues.add("observed_input_schema_mismatch")
                row_count += batch.num_rows
                continue
            for offset in range(batch.num_rows):
                row = {
                    name: batch.column(index)[offset].as_py()
                    for index, (name, _, _) in enumerate(OBSERVED_PREDICTION_COLUMNS)
                }
                expected_row_seq = row_count
                row_count += 1
                if row["row_seq"] != expected_row_seq:
                    issues.add("observed_input_order_mismatch")
                    if type(row["row_seq"]) is int and row["row_seq"] < expected_row_seq:
                        issues.add("observed_input_duplicate_row")
                    else:
                        issues.add("observed_input_missing_row")
                order = tuple(
                    row[name] for name in (
                        "participant_position", "family_position", "arm_position",
                        "model_position", "target_position", "primitive_position",
                        "sensor_day_id", "draw", "condition_position",
                    )
                )
                if previous_order is not None and order <= previous_order:
                    issues.add("observed_input_order_mismatch")
                previous_order = order
                _validate_position_contract(row, authority, issues)
                try:
                    observed_hasher.update(row)
                except ValueError:
                    logical_invalid = True
                try:
                    source_patch_hasher.update({
                        name: row[name] for name, _, _ in SOURCE_PATCH_PROJECTION_COLUMNS
                    })
                except ValueError:
                    issues.add("source_provenance_crosswire")
                try:
                    if row["row_content_digest"] != _row_digest(
                        "task-12-downstream-observed-prediction-row-v1", row, _OBSERVED_DIGEST_NAMES
                    ):
                        issues.add("observed_row_content_digest_mismatch")
                except ValueError:
                    issues.add("observed_row_content_digest_mismatch")
                category = _validate_row_contract(row, issues)

                fit_seq = row["fit_seq"]
                expected_fit_seq = None
                if all(type(row[name]) is int for name in (
                    "participant_position", "arm_position", "model_position", "target_position"
                )):
                    expected_fit_seq = (((row["participant_position"] * 2 + row["arm_position"]) * 2 + row["model_position"]) * 4 + row["target_position"])
                if type(fit_seq) is not int or fit_seq != expected_fit_seq:
                    issues.add("fit_identity_crosswire")
                else:
                    projection = {
                        "fit_seq": fit_seq, "heldout_position": row["participant_position"],
                        "fold": row["fold"], "held_out_subject": row["held_out_subject"],
                        "arm_position": row["arm_position"], "arm": row["arm"],
                        "model_position": row["model_position"], "model_name": row["model_name"],
                        "target_position": row["target_position"], "target": row["target"],
                        "fit_ledger_record_digest": row["fit_ledger_record_digest"],
                        "preprocessor_digest": row["preprocessor_digest"],
                        "model_digest": row["model_digest"], "fit_content_digest": row["fit_content_digest"],
                    }
                    if fit_seq in fit_records and fit_records[fit_seq] != projection:
                        issues.add("fit_identity_crosswire")
                    else:
                        fit_records.setdefault(fit_seq, projection)
                base_buffer.append((row, category))
                if len(base_buffer) == 3:
                    process_base(base_buffer)
                    base_buffer = []
    finally:
        if base_buffer:
            issues.add("observed_input_missing_row")
        finalize_group()
        finalize_participant()
        for writer in provisional_writers.values():
            writer.close()

    pin = authority.observed_prediction_pin
    if row_count != pin.row_count:
        issues.add("observed_input_row_count_mismatch")
        if row_count < pin.row_count:
            issues.add("observed_input_missing_row")
        else:
            issues.add("observed_input_duplicate_row")
    if logical_invalid or observed_hasher.row_count != row_count or observed_hasher.hexdigest() != pin.logical_digest:
        issues.add("observed_input_logical_digest_mismatch")
    if key_hasher.row_count != authority.expected_primary_key_count or key_hasher.hexdigest() != authority.expected_primary_key_digest:
        issues.add("primary_key_digest_mismatch")
    if (
        source_patch_hasher.row_count != authority.source_patch_projection_row_count
        or source_patch_hasher.hexdigest() != authority.source_patch_projection_digest
    ):
        issues.add("source_provenance_crosswire")
    if set(fit_records) != set(range(160)):
        issues.add("fit_topology_incomplete")
    fit_hasher = _LogicalHasher(
        "task-12-downstream-fit-ledger-projection-digest-v1", FIT_PROJECTION_COLUMNS
    )
    try:
        for fit_seq in sorted(fit_records):
            fit_hasher.update(fit_records[fit_seq])
    except ValueError:
        issues.add("fit_identity_crosswire")
    if fit_hasher.hexdigest() != authority.fit_ledger_projection_digest:
        issues.add("fit_projection_digest_mismatch")

    if (
        group_ordinal != 640
        or provisional_writers["group_cells"].row_count != 640
        or provisional_writers["condition_group_descriptives"].row_count != 1920
    ):
        issues.add("group_grid_incomplete")
    if (
        len(participant_rows) != 10
        or any(row["group_cell_count"] != 64 for row in participant_rows)
        or participant_condition_row_count != 210
    ):
        issues.add("participant_grid_incomplete")

    global_condition_rows: list[dict[str, object]] = []
    if not issues:
        for condition_position, condition in enumerate(CONDITIONS):
            for metric_position, metric in enumerate(METRICS):
                values = global_values[condition_position][metric_position]
                global_condition_rows.append({
                    "condition_position": condition_position, "condition": condition,
                    "metric": metric, "value": _median(values) if values else None,
                    "nonnull_participant_count": len(values), "total_participant_count": 10,
                    "null_reason": None if values else "no_nonnull_participant_values",
                })

    integrity_issues = _ordered_issues(issues)
    integrity_complete = not integrity_issues
    bootstrap_indices: np.ndarray | None = None
    bootstrap_statistics: np.ndarray | None = None
    sign_matrix: np.ndarray | None = None
    sign_statistics: np.ndarray | None = None
    extreme: np.ndarray | None = None
    t_observed = lower = upper = sign_p = None
    positive_count = numerator = None
    label = "inconclusive"
    if integrity_complete:
        participant_values = np.asarray([row["p_median"] for row in participant_rows], dtype=np.float64)
        if participant_values.shape != (10,) or not np.isfinite(participant_values).all():
            issues.add("required_output_invalid")
        else:
            t_observed = float(np.median(participant_values))
            rng = np.random.Generator(np.random.PCG64(42))
            bootstrap_indices = rng.integers(0, 10, size=(10_000, 10), dtype=np.int64, endpoint=False)
            bootstrap_statistics = np.median(participant_values[bootstrap_indices], axis=1)
            lower, upper = (
                float(value)
                for value in np.quantile(bootstrap_statistics, [0.025, 0.975], method="linear")
            )
            sign_matrix = np.empty((1024, 10), dtype=np.int64)
            sign_matrix[0, :] = 1
            for sign_index in range(1, 1024):
                code = sign_index - 1
                for participant_position in range(10):
                    sign_matrix[sign_index, participant_position] = 1 if code & (1 << participant_position) else -1
            sign_statistics = np.median(sign_matrix * participant_values, axis=1)
            extreme = np.abs(sign_statistics) >= abs(t_observed)
            numerator = int(np.count_nonzero(extreme))
            sign_p = numerator / 1024
            positive_count = int(np.count_nonzero(participant_values > 0.0))
            label = (
                "corroborative_stability_impact_supported"
                if t_observed > 0.0 and lower > 0.0 and positive_count >= 8 and sign_p <= 0.05
                else "corroborative_impact_not_demonstrated"
            )
    if issues:
        integrity_issues = _ordered_issues(issues)
        integrity_complete = False
        label = "inconclusive"
        t_observed = lower = upper = sign_p = None
        positive_count = numerator = None
        bootstrap_indices = bootstrap_statistics = sign_matrix = sign_statistics = extreme = None

    if not integrity_complete:
        for table_name in provisional_table_names:
            table_path = handoff_dir / f"{table_name}.arrow"
            table_path.unlink()
            _write_ipc_rows(table_path, TABLE_SCHEMAS[table_name], ())

    def bootstrap_index_rows() -> Iterator[dict[str, object]]:
        if bootstrap_indices is not None:
            for bootstrap_draw in range(10_000):
                for roster_slot in range(10):
                    yield {
                        "bootstrap_draw": bootstrap_draw, "roster_slot": roster_slot,
                        "participant_position": int(bootstrap_indices[bootstrap_draw, roster_slot]),
                    }

    def bootstrap_stat_rows() -> Iterator[dict[str, object]]:
        if bootstrap_statistics is not None:
            for bootstrap_draw, statistic in enumerate(bootstrap_statistics):
                yield {"bootstrap_draw": bootstrap_draw, "statistic": float(statistic)}

    def sign_ledger_rows() -> Iterator[dict[str, object]]:
        if sign_matrix is not None:
            for sign_index in range(1024):
                for participant_position in range(10):
                    yield {
                        "sign_index": sign_index, "participant_position": participant_position,
                        "sign": int(sign_matrix[sign_index, participant_position]),
                    }

    def sign_stat_rows() -> Iterator[dict[str, object]]:
        if sign_statistics is not None and extreme is not None:
            for sign_index, statistic in enumerate(sign_statistics):
                yield {
                    "sign_index": sign_index, "statistic": float(statistic),
                    "absolute_extreme": bool(extreme[sign_index]),
                }

    output_rows: Mapping[str, Iterable[dict[str, object]]] = {
        "bootstrap_indices": bootstrap_index_rows(),
        "bootstrap_statistics": bootstrap_stat_rows(),
        "sign_ledger": sign_ledger_rows(),
        "sign_statistics": sign_stat_rows(),
        "global_condition_descriptives": global_condition_rows if integrity_complete else (),
    }
    for table_name, rows in output_rows.items():
        _write_ipc_rows(handoff_dir / f"{table_name}.arrow", TABLE_SCHEMAS[table_name], rows)

    decision_row: dict[str, object] = {
        "integrity_complete": integrity_complete, "label": label,
        "t_observed": t_observed, "bootstrap_lower": lower, "bootstrap_upper": upper,
        "direction_positive_count": positive_count, "sign_extreme_numerator": numerator,
        "sign_denominator": 1024 if integrity_complete else None,
        "sign_p_value": sign_p, "integrity_issue_count": len(integrity_issues),
        "integrity_issues_json": json.dumps(
            list(integrity_issues), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ),
        "sensitivity_wording": SENSITIVITY_WORDING, "task12_decision": TASK12_DECISION,
    }
    decision_row["content_digest"] = _row_digest(
        "task-12-downstream-decision-row-v1", decision_row, _DECISION_DIGEST_NAMES
    )
    _write_ipc_rows(handoff_dir / "decision.arrow", TABLE_SCHEMAS["decision"], (decision_row,))

    seals = tuple(_table_seal(handoff_dir / f"{name}.arrow", name) for name in TABLE_NAMES)
    if integrity_complete:
        expected_counts = (authority.expected_primary_key_count, 640, 10, 100_000, 10_000, 10_240, 1_024, 1_920, 210, 21, 1)
        if tuple(seal.row_count for seal in seals) != expected_counts:
            raise RuntimeError("required output count mismatch after integrity finalization")
    result_digest = _sha_payload([
        "task-12-downstream-statistics-result-v1", label, integrity_complete,
        list(integrity_issues), authority.content_digest,
        [[seal.name, seal.logical_digest] for seal in seals],
    ])
    document = {
        "schema_name": "task-12-downstream-statistics-handoff-v1",
        "label": label, "integrity_complete": integrity_complete,
        "integrity_issues": list(integrity_issues), "authority_digest": authority.content_digest,
        "tables": [_seal_document(seal) for seal in seals], "result_digest": result_digest,
    }
    manifest_bytes = _canonical(document) + b"\n"
    manifest_path = handoff_dir / "handoff.json"
    _write_exclusive_bytes(manifest_path, manifest_bytes)
    manifest_physical_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    handoff_digest = _sha_payload([
        "task-12-downstream-statistics-handoff-content-v1", manifest_physical_sha256,
        authority.content_digest, result_digest, [seal.content_digest for seal in seals],
    ])
    return DownstreamStatisticsHandoff(
        handoff_dir=handoff_dir, manifest_snapshot_bytes=manifest_bytes,
        manifest_physical_sha256=manifest_physical_sha256,
        authority_digest=authority.content_digest, result_digest=result_digest,
        content_digest=handoff_digest,
    )


def _seal_document(seal: StatisticsTableSeal) -> dict[str, object]:
    return {
        "name": seal.name, "relative_path": seal.relative_path, "row_count": seal.row_count,
        "columns": [list(column) for column in seal.columns], "byte_count": seal.byte_count,
        "physical_sha256": seal.physical_sha256, "logical_digest": seal.logical_digest,
        "content_digest": seal.content_digest,
    }


def _artifact_document(seal: StatisticsArtifactSeal) -> dict[str, object]:
    return {
        "name": seal.name, "relative_path": seal.relative_path, "row_count": seal.row_count,
        "columns": [list(column) for column in seal.columns], "byte_count": seal.byte_count,
        "physical_sha256": seal.physical_sha256, "logical_digest": seal.logical_digest,
        "content_digest": seal.content_digest,
    }


def _strict_json(data: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"nonfinite JSON number is forbidden: {value}")

    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid canonical JSON") from error


def _table_seal_from_document(document: object) -> StatisticsTableSeal:
    if type(document) is not dict or set(document) != {
        "name", "relative_path", "row_count", "columns", "byte_count",
        "physical_sha256", "logical_digest", "content_digest",
    }:
        raise ValueError("invalid table seal document")
    name = document["name"]
    if type(name) is not str or name not in TABLE_NAMES:
        raise ValueError("invalid table name")
    columns_value = document["columns"]
    if type(columns_value) is not list:
        raise ValueError("invalid table columns")
    columns = tuple(tuple(column) for column in columns_value)
    seal = StatisticsTableSeal(
        name=name, relative_path=document["relative_path"], row_count=document["row_count"],
        columns=columns, byte_count=document["byte_count"],
        physical_sha256=document["physical_sha256"], logical_digest=document["logical_digest"],
        content_digest=document["content_digest"],
    )
    _validate_table_seal_scalars(seal)
    return seal


def _validate_table_seal_scalars(seal: StatisticsTableSeal) -> None:
    if type(seal.name) is not str or type(seal.relative_path) is not str:
        raise TypeError("table seal names must be exact strings")
    _require_exact_columns_tree(seal.columns, "table seal columns")
    if seal.relative_path != f"{seal.name}.arrow" or seal.columns != TABLE_COLUMNS[seal.name]:
        raise ValueError("table seal topology mismatch")
    if type(seal.row_count) is not int or type(seal.byte_count) is not int or seal.row_count < 0 or seal.byte_count < 0:
        raise TypeError("table seal counts must be exact nonnegative ints")
    for name in ("physical_sha256", "logical_digest", "content_digest"):
        _require_digest(getattr(seal, name), f"table seal {name}")
    expected = _sha_payload([
        "task-12-downstream-statistics-table-seal-v1", seal.name, seal.relative_path,
        seal.row_count, [list(column) for column in seal.columns], seal.byte_count,
        seal.physical_sha256, seal.logical_digest,
    ])
    if seal.content_digest != expected:
        raise ValueError("table seal self-digest mismatch")


def _validate_handoff(
    handoff: DownstreamStatisticsHandoff,
    expected_handoff_digest: str,
) -> tuple[
    DownstreamStatisticsResult,
    dict[str, tuple[int, int, int, int, int, int]],
]:
    _require_digest(expected_handoff_digest, "expected_handoff_digest")
    if type(handoff) is not DownstreamStatisticsHandoff:
        raise TypeError("handoff has the wrong exact type")
    if type(handoff.manifest_snapshot_bytes) is not bytes or not isinstance(handoff.handoff_dir, Path):
        raise TypeError("handoff contains a non-exact transport field")
    for name in ("manifest_physical_sha256", "authority_digest", "result_digest", "content_digest"):
        _require_digest(getattr(handoff, name), f"handoff.{name}")
    if handoff.content_digest != expected_handoff_digest:
        raise ValueError("independently held handoff digest mismatch")
    root = _require_existing_directory(handoff.handoff_dir)
    expected_files = {"handoff.json", *(f"{name}.arrow" for name in TABLE_NAMES)}
    snapshots = _enumerate_exact_files(root, expected_files)
    manifest_path = root / "handoff.json"
    manifest_bytes, manifest_physical, document = _stable_canonical_json(
        manifest_path, snapshots["handoff.json"]
    )
    if manifest_bytes != handoff.manifest_snapshot_bytes:
        raise ValueError("handoff manifest snapshot mismatch")
    if manifest_physical != handoff.manifest_physical_sha256:
        raise ValueError("handoff manifest physical digest mismatch")
    if type(document) is not dict or tuple(document) != (
        "authority_digest", "integrity_complete", "integrity_issues", "label",
        "result_digest", "schema_name", "tables",
    ):
        # Canonical JSON sort order is lexicographic; exact key set is checked below.
        if type(document) is not dict or set(document) != {
            "schema_name", "label", "integrity_complete", "integrity_issues",
            "authority_digest", "tables", "result_digest",
        }:
            raise ValueError("handoff manifest keys mismatch")
    if document["schema_name"] != "task-12-downstream-statistics-handoff-v1":
        raise ValueError("handoff manifest schema mismatch")
    if type(document["label"]) is not str or document["label"] not in LABELS:
        raise ValueError("handoff label mismatch")
    if type(document["integrity_complete"]) is not bool or type(document["integrity_issues"]) is not list:
        raise TypeError("handoff integrity metadata type mismatch")
    if any(type(issue) is not str for issue in document["integrity_issues"]):
        raise TypeError("handoff integrity issue codes must be exact strings")
    integrity_issues = tuple(document["integrity_issues"])
    if integrity_issues != _ordered_issues(set(integrity_issues)):
        raise ValueError("handoff integrity issue ordering mismatch")
    if document["integrity_complete"] != (not integrity_issues):
        raise ValueError("handoff integrity flag mismatch")
    _require_digest(document["authority_digest"], "handoff manifest authority_digest")
    _require_digest(document["result_digest"], "handoff manifest result_digest")
    if document["authority_digest"] != handoff.authority_digest:
        raise ValueError("handoff authority crosswire")
    tables_value = document["tables"]
    if type(tables_value) is not list or len(tables_value) != 11:
        raise ValueError("handoff table list mismatch")
    declared_seals = tuple(_table_seal_from_document(item) for item in tables_value)
    if tuple(seal.name for seal in declared_seals) != TABLE_NAMES:
        raise ValueError("handoff table order mismatch")
    verified_seals: list[StatisticsTableSeal] = []
    for seal in declared_seals:
        verified = _table_seal(
            root / seal.relative_path,
            seal.name,
            snapshots[seal.relative_path],
        )
        if verified != seal:
            raise ValueError("handoff table mutation or crosswire")
        verified_seals.append(verified)
    result_digest = _sha_payload([
        "task-12-downstream-statistics-result-v1", document["label"],
        document["integrity_complete"], list(integrity_issues), handoff.authority_digest,
        [[seal.name, seal.logical_digest] for seal in verified_seals],
    ])
    if document["result_digest"] != result_digest or handoff.result_digest != result_digest:
        raise ValueError("handoff result digest mismatch")
    handoff_digest = _sha_payload([
        "task-12-downstream-statistics-handoff-content-v1", handoff.manifest_physical_sha256,
        handoff.authority_digest, result_digest, [seal.content_digest for seal in verified_seals],
    ])
    if handoff.content_digest != handoff_digest:
        raise ValueError("handoff content digest mismatch")
    tables = DownstreamStatisticsTables(*verified_seals)
    result = DownstreamStatisticsResult(
        label=document["label"], integrity_complete=document["integrity_complete"],
        integrity_issues=integrity_issues, authority_digest=handoff.authority_digest,
        tables=tables, content_digest=result_digest,
    )
    return result, snapshots


def read_downstream_statistics_handoff(
    handoff: DownstreamStatisticsHandoff,
    *,
    expected_handoff_digest: str,
) -> DownstreamStatisticsResult:
    result, _ = _validate_handoff(handoff, expected_handoff_digest)
    return result


def _artifact_seal(
    path: Path,
    name: str,
    expected_logical_digest: str,
    expected_signature: tuple[int, int, int, int, int, int] | None = None,
) -> StatisticsArtifactSeal:
    expected_schema = TABLE_SCHEMAS[name]
    logical = _LogicalHasher("task-12-downstream-statistics-table-logical-v1", TABLE_COLUMNS[name])
    with _StableReader(path, expected_signature) as stable:
        parquet_file = pq.ParquetFile(stable.stream)
        if parquet_file.schema_arrow != expected_schema:
            raise ValueError("Parquet schema mismatch")
        row_count = 0
        for row_group in range(parquet_file.num_row_groups):
            group_count = parquet_file.metadata.row_group(row_group).num_rows
            if group_count <= 0 or group_count > _BATCH_SIZE:
                raise ValueError("noncanonical Parquet row group")
            if row_group < parquet_file.num_row_groups - 1 and group_count != _BATCH_SIZE:
                raise ValueError("nonfinal Parquet row group is short")
            seen = 0
            for batch in parquet_file.iter_batches(
                batch_size=_BATCH_SIZE,
                row_groups=[row_group],
                use_threads=False,
            ):
                seen += batch.num_rows
                for offset in range(batch.num_rows):
                    row = {
                        column_name: batch.column(index)[offset].as_py()
                        for index, (column_name, _, _) in enumerate(TABLE_COLUMNS[name])
                    }
                    logical.update(row)
            if seen != group_count:
                raise ValueError("Parquet row group read mismatch")
            row_count += group_count
        if row_count != parquet_file.metadata.num_rows:
            raise ValueError("Parquet row count mismatch")
        logical_digest = logical.hexdigest()
        if logical_digest != expected_logical_digest:
            raise ValueError("persisted logical digest mismatch")
        stable.checkpoint("after_parse_before_hash")
        byte_count, physical_sha256 = _hash_stream(stable.stream)
        stable.checkpoint("after_hash")
    relative_path = f"{name}.parquet"
    columns = TABLE_COLUMNS[name]
    content_digest = _sha_payload([
        "task-12-downstream-statistics-artifact-seal-v1", name, relative_path, row_count,
        [list(column) for column in columns], byte_count, physical_sha256, logical_digest,
    ])
    return StatisticsArtifactSeal(
        name, relative_path, row_count, columns, byte_count, physical_sha256,
        logical_digest, content_digest,
    )


def _transcode_ipc_to_parquet(
    source_path: Path,
    output_path: Path,
    table_seal: StatisticsTableSeal,
    expected_signature: tuple[int, int, int, int, int, int],
) -> None:
    name = table_seal.name
    schema = TABLE_SCHEMAS[name]
    logical = _LogicalHasher(
        "task-12-downstream-statistics-table-logical-v1", TABLE_COLUMNS[name]
    )
    with _StableReader(source_path, expected_signature) as stable:
        reader = ipc.open_stream(stable.stream)
        if reader.schema != schema or reader.schema.metadata is not None:
            raise ValueError("handoff IPC schema drift")
        sink = _open_exclusive_output(output_path)
        writer = None
        try:
            writer = pq.ParquetWriter(
                sink,
                schema,
                version="2.6",
                compression="zstd",
                compression_level=9,
                use_dictionary=False,
                write_statistics=False,
                use_byte_stream_split=False,
                data_page_version="1.0",
                store_schema=True,
            )
            saw_short = False
            row_count = 0
            for batch in reader:
                if saw_short or batch.num_rows <= 0 or batch.num_rows > _BATCH_SIZE:
                    raise ValueError("handoff IPC partition drift")
                if batch.num_rows < _BATCH_SIZE:
                    saw_short = True
                for offset in range(batch.num_rows):
                    logical.update({
                        column_name: batch.column(index)[offset].as_py()
                        for index, (column_name, _, _) in enumerate(TABLE_COLUMNS[name])
                    })
                row_count += batch.num_rows
                writer.write_batch(batch, row_group_size=batch.num_rows)
            writer.close()
            writer = None
            if not sink.closed:
                sink.flush()
                os.fsync(sink.fileno())
        finally:
            if writer is not None:
                writer.close()
            if not sink.closed:
                sink.close()
        stable.checkpoint("after_parse_before_hash")
        byte_count, physical_sha256 = _hash_stream(stable.stream)
        stable.checkpoint("after_hash")
        if (
            row_count != table_seal.row_count
            or logical.hexdigest() != table_seal.logical_digest
            or byte_count != table_seal.byte_count
            or physical_sha256 != table_seal.physical_sha256
        ):
            raise ValueError("handoff IPC changed before or during persistence")


def persist_downstream_statistics(
    handoff: DownstreamStatisticsHandoff,
    output_dir: Path,
    *,
    expected_handoff_digest: str,
) -> DownstreamStatisticsManifest:
    result, handoff_snapshots = _validate_handoff(handoff, expected_handoff_digest)
    output_dir = _new_empty_directory(output_dir)
    table_seals = tuple(getattr(result.tables, name) for name in TABLE_NAMES)
    artifacts: list[StatisticsArtifactSeal] = []
    for table_seal in table_seals:
        output_path = output_dir / f"{table_seal.name}.parquet"
        _transcode_ipc_to_parquet(
            handoff.handoff_dir / table_seal.relative_path,
            output_path,
            table_seal,
            handoff_snapshots[table_seal.relative_path],
        )
        artifact = _artifact_seal(output_path, table_seal.name, table_seal.logical_digest)
        if artifact.row_count != table_seal.row_count:
            raise ValueError("persisted row count drift")
        artifacts.append(artifact)
    result_digest = _sha_payload([
        "task-12-downstream-statistics-result-v1", result.label, result.integrity_complete,
        list(result.integrity_issues), result.authority_digest,
        [[artifact.name, artifact.logical_digest] for artifact in artifacts],
    ])
    if result_digest != result.content_digest:
        raise ValueError("transport-neutral result digest changed during persistence")
    schema_name = "task-12-downstream-statistics-manifest-v1"
    content_digest = _sha_payload([
        "task-12-downstream-statistics-manifest-v1", schema_name, result.label,
        result.integrity_complete, list(result.integrity_issues), result.authority_digest,
        result_digest, [artifact.content_digest for artifact in artifacts],
    ])
    manifest = DownstreamStatisticsManifest(
        schema_name=schema_name, label=result.label,
        integrity_complete=result.integrity_complete, integrity_issues=result.integrity_issues,
        authority_digest=result.authority_digest, result_digest=result_digest,
        artifacts=tuple(artifacts), content_digest=content_digest,
    )
    document = {
        "schema_name": manifest.schema_name, "label": manifest.label,
        "integrity_complete": manifest.integrity_complete,
        "integrity_issues": list(manifest.integrity_issues),
        "authority_digest": manifest.authority_digest, "result_digest": manifest.result_digest,
        "artifacts": [_artifact_document(artifact) for artifact in manifest.artifacts],
    }
    _write_exclusive_bytes(output_dir / "manifest.json", _canonical(document) + b"\n")
    verify_persisted_downstream_statistics(
        output_dir, manifest, expected_manifest_digest=manifest.content_digest
    )
    return manifest


def _validate_manifest_scalars(manifest: DownstreamStatisticsManifest) -> None:
    if type(manifest) is not DownstreamStatisticsManifest:
        raise TypeError("manifest has the wrong exact type")
    if type(manifest.schema_name) is not str or manifest.schema_name != "task-12-downstream-statistics-manifest-v1":
        raise ValueError("manifest schema mismatch")
    if type(manifest.label) is not str or manifest.label not in LABELS:
        raise ValueError("manifest label mismatch")
    if type(manifest.integrity_complete) is not bool or type(manifest.integrity_issues) is not tuple:
        raise TypeError("manifest integrity metadata type mismatch")
    if any(type(issue) is not str for issue in manifest.integrity_issues):
        raise TypeError("manifest issue codes must be exact strings")
    if manifest.integrity_issues != _ordered_issues(set(manifest.integrity_issues)):
        raise ValueError("manifest issue ordering mismatch")
    if manifest.integrity_complete != (not manifest.integrity_issues):
        raise ValueError("manifest integrity flag mismatch")
    if type(manifest.artifacts) is not tuple or len(manifest.artifacts) != 11:
        raise TypeError("manifest artifacts must be an exact eleven-element tuple")
    for artifact in manifest.artifacts:
        _validate_artifact_scalars(artifact)
    for name in ("authority_digest", "result_digest", "content_digest"):
        _require_digest(getattr(manifest, name), f"manifest.{name}")


def _validate_artifact_scalars(artifact: StatisticsArtifactSeal) -> None:
    if type(artifact) is not StatisticsArtifactSeal:
        raise TypeError("artifact has the wrong exact type")
    if type(artifact.name) is not str or type(artifact.relative_path) is not str:
        raise TypeError("artifact names and paths must be exact strings")
    if type(artifact.row_count) is not int or type(artifact.byte_count) is not int:
        raise TypeError("artifact counts must be exact ints")
    if artifact.row_count < 0 or artifact.byte_count < 0:
        raise ValueError("artifact counts must be nonnegative")
    _require_exact_columns_tree(artifact.columns, "artifact columns")
    for name in ("physical_sha256", "logical_digest", "content_digest"):
        _require_digest(getattr(artifact, name), f"artifact.{name}")


def verify_persisted_downstream_statistics(
    output_dir: Path,
    manifest: DownstreamStatisticsManifest,
    *,
    expected_manifest_digest: str,
) -> None:
    _require_digest(expected_manifest_digest, "expected_manifest_digest")
    _validate_manifest_scalars(manifest)
    if manifest.content_digest != expected_manifest_digest:
        raise ValueError("independently held manifest digest mismatch")
    root = _require_existing_directory(output_dir)
    expected_files = {"manifest.json", *(f"{name}.parquet" for name in TABLE_NAMES)}
    snapshots = _enumerate_exact_files(root, expected_files)
    if tuple(artifact.name for artifact in manifest.artifacts) != TABLE_NAMES:
        raise ValueError("persisted artifact order mismatch")
    verified: list[StatisticsArtifactSeal] = []
    for artifact in manifest.artifacts:
        if artifact.relative_path != f"{artifact.name}.parquet" or artifact.columns != TABLE_COLUMNS[artifact.name]:
            raise ValueError("artifact topology mismatch")
        expected_artifact_digest = _sha_payload([
            "task-12-downstream-statistics-artifact-seal-v1", artifact.name,
            artifact.relative_path, artifact.row_count,
            [list(column) for column in artifact.columns], artifact.byte_count,
            artifact.physical_sha256, artifact.logical_digest,
        ])
        if artifact.content_digest != expected_artifact_digest:
            raise ValueError("artifact self-digest mismatch")
        inspected = _artifact_seal(
            root / artifact.relative_path,
            artifact.name,
            artifact.logical_digest,
            snapshots[artifact.relative_path],
        )
        if inspected != artifact:
            raise ValueError("persisted artifact mutation or crosswire")
        verified.append(inspected)
    result_digest = _sha_payload([
        "task-12-downstream-statistics-result-v1", manifest.label,
        manifest.integrity_complete, list(manifest.integrity_issues), manifest.authority_digest,
        [[artifact.name, artifact.logical_digest] for artifact in verified],
    ])
    if result_digest != manifest.result_digest:
        raise ValueError("persisted result digest mismatch")
    manifest_digest = _sha_payload([
        "task-12-downstream-statistics-manifest-v1", manifest.schema_name, manifest.label,
        manifest.integrity_complete, list(manifest.integrity_issues), manifest.authority_digest,
        result_digest, [artifact.content_digest for artifact in verified],
    ])
    if manifest_digest != manifest.content_digest:
        raise ValueError("manifest content digest mismatch")
    document = {
        "schema_name": manifest.schema_name, "label": manifest.label,
        "integrity_complete": manifest.integrity_complete,
        "integrity_issues": list(manifest.integrity_issues),
        "authority_digest": manifest.authority_digest, "result_digest": manifest.result_digest,
        "artifacts": [_artifact_document(artifact) for artifact in manifest.artifacts],
    }
    expected_bytes = _canonical(document) + b"\n"
    manifest_path = root / "manifest.json"
    manifest_bytes, _, parsed_document = _stable_canonical_json(
        manifest_path, snapshots["manifest.json"]
    )
    if manifest_bytes != expected_bytes or parsed_document != document:
        raise ValueError("persisted manifest bytes mismatch")
