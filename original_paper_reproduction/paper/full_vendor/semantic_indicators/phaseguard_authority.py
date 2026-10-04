"""Fail-closed read-only authorities for the PhaseGuard release boundary.

This module loads the independently issued Task 8A CLEAN graph and the sealed
historical G2 official graph.  It reconstructs calibration-input authorities
through a fail-closed boundary and contains no source-index, ledger,
materialization, release, candidate, or result writer.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import BinaryIO, Iterator, Literal, NoReturn

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .personalization import build_loso_representations
from .primitives import build_sensor_calendar_scaffold, extract_daily_primitives
from .reliability import CORE_G2_PRIMITIVES


_PATH_TYPE = type(Path())
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

_EXPECTATIONS_FIELDS = {
    "content_digest",
    "expected_reviewed_tree_sha256",
    "latest_review_package",
    "predecessor_finding_chain",
    "predecessor_finding_chain_digest",
    "required_evidence_report_pins",
    "required_file_hashes",
    "required_file_hashes_digest",
    "required_finding_ids",
    "required_finding_ids_digest",
    "schema_name",
}
_REPORT_FIELDS = {
    "closed_finding_ids",
    "content_digest",
    "expectations_digest",
    "latest_review_package",
    "predecessor_finding_chain",
    "predecessor_finding_chain_digest",
    "required_finding_ids",
    "required_finding_ids_digest",
    "reviewed_evidence_report_pins",
    "reviewed_file_hashes",
    "reviewed_tree_sha256",
    "schema_name",
    "verdict",
}
_PACKAGE_FIELDS = {
    "content_digest",
    "predecessor_finding_chain",
    "required_evidence_report_pins",
    "required_file_hashes",
    "required_finding_ids",
    "reviewed_tree_spec",
    "schema_name",
}
_PIN_FIELDS = {
    "content_digest",
    "expected_sha256",
    "repo_relative_path",
    "role",
}

_CANONICAL_JSON_SPEC = {
    "allow_nan": False,
    "encoding": "utf-8",
    "ensure_ascii": True,
    "separators": [",", ":"],
    "sort_keys": True,
}
_REVIEWED_FILE_ROLES = (
    "source",
    "source",
    "test",
    "test",
    "plan",
    "plan",
    "plan",
    "plan",
)
_REVIEWED_TREE_SPEC = {
    "canonical_json": _CANONICAL_JSON_SPEC,
    "excluded_artifact_schemas": [
        "task8a-clean-expectations-v1",
        "task8a-clean-v3",
    ],
    "hash_algorithm": "sha256",
    "hash_payload": "ordered_leaf_array",
    "latest_review_package_pin_source": (
        "loader_injected_repo_relative_path_and_actual_sha256_after_"
        "expected_actual_equality"
    ),
    "leaf_payload": ["role", "repo_relative_path", "expected_sha256"],
    "leaf_sequence": [
        {
            "collection": "latest_review_package",
            "order": "singleton",
            "role": "review_package",
        },
        {
            "collection": "predecessor_finding_chain",
            "order": "array_order",
            "role": "predecessor_finding",
        },
        {
            "collection": "required_file_hashes",
            "order": "array_order",
            "roles": list(_REVIEWED_FILE_ROLES),
        },
        {
            "collection": "required_evidence_report_pins",
            "order": "array_order",
            "role": "evidence_report",
        },
    ],
    "schema_name": "phaseguard-reviewed-tree-v1",
}

TASK0B_REVIEW_ONLY_SENTINEL_DIGEST = hashlib.sha256(
    b"phaseguard:task0b_review_only:not_release:v1"
).hexdigest()

OFFICIAL_G2_MANIFEST_REPO_PATH = (
    ".superpowers/sdd/task-4c2-v2-official/manifest.json"
)
OFFICIAL_G2_MANIFEST_SHA256 = (
    "223b7b55e66d1dc8ecd522cde77c992934bde0897fbf9b7745b8bf18a361a7d1"
)
OFFICIAL_G1_REPO_PATH = (
    ".superpowers/sdd/task-4a-reviewed-g1-pair-summary.json"
)
OFFICIAL_G1_PHYSICAL_SHA256 = (
    "ac03ebba9349e750abc3d7b26d089b97cb6997ed3ba14796fc2abdde7e274596"
)
OFFICIAL_G1_ARTIFACT_SHA256 = (
    "a05199eada5a13f54e96393bd3a1a1920390bbd8aba83c7f15a4b4d790062fcb"
)
OFFICIAL_G1_CONFIG_SHA256 = (
    "5236c00952f050ca408f402e2681b47a55ffd29fdb3c8e3b0e0cf93cd02e723a"
)

OFFICIAL_G2_PRIMITIVES = (
    "screen_load_24h",
    "usage_load_24h",
    "screen_disengagement_p90",
    "usage_disengagement_p90",
    "phone_activity_load_24h",
    "step_load_24h",
    "mobile_light_exposure_24h",
    "wearable_light_exposure_24h",
)
OFFICIAL_G2_SCENARIOS = (
    "contiguous_10pct",
    "contiguous_20pct",
    "contiguous_40pct",
    "outage_1h",
    "outage_3h",
    "outage_6h",
    "evening_critical",
    "empirical_gap",
)
OFFICIAL_G2_PARTICIPANTS = tuple(f"id{index:02d}" for index in range(1, 11))
OFFICIAL_G2_DRAWS = tuple(range(50))

CALIBRATION_PRIMITIVES_REPO_PATH = (
    ".superpowers/sdd/task-4c2-real-primitives.parquet"
)
CALIBRATION_BASELINES_REPO_PATH = (
    ".superpowers/sdd/task-4c2-real-baselines.parquet"
)
CALIBRATION_PRIMITIVES_SHA256 = (
    "6490ece513f26d1056aa0e4cd2cfb9220910b3095e887d0f8e68561f07b1a230"
)
CALIBRATION_BASELINES_SHA256 = (
    "81f6458ac1661518309883f610347ef028986c978540feef1ffb842f50ebd37c"
)
CALIBRATION_SENSOR_FILES = (
    "mScreenStatus_canonical.parquet",
    "mUsageStats_canonical.parquet",
    "mActivity_canonical.parquet",
    "wPedo_canonical.parquet",
    "mLight_canonical.parquet",
    "wLight_canonical.parquet",
)
_CALIBRATION_SENSOR_FILE_HASHES = (
    (
        "mScreenStatus_canonical.parquet",
        "568393a85398c5cce82b0e87f8da14ed8bc7f6952bb5fc0476f81a3b2d57580b",
    ),
    (
        "mUsageStats_canonical.parquet",
        "a40a1c1c79ca880936a81373262f784354757461f60f71d3e70ff25df36dc36e",
    ),
    (
        "mActivity_canonical.parquet",
        "8a01873ebad0421af2d4caaef18633c08b2dc2048399799f7c6a95a01c808c07",
    ),
    (
        "wPedo_canonical.parquet",
        "47da25f672fb4a1bf2f7aebc023056ac4423726ec422b1b074bee564f583787a",
    ),
    (
        "mLight_canonical.parquet",
        "60d0bd8ed824298ac0bfd344d30a0c114df63f6860c450bc01df9813b334064e",
    ),
    (
        "wLight_canonical.parquet",
        "c647a249f2becafb95cd8c0090c3b778f186a6289c522b2635d6e900e6141e2a",
    ),
)
_CALIBRATION_OFFICIAL_G2_AUTHORITY_DIGEST = (
    "d7fba213443e03b58cd5e5fe5b2c879072faa56eda38ef9d830e3591a549739e"
)
_CALIBRATION_TASK8A_CLEAN_AUTHORITY_DIGEST = (
    "66e8214cb348197ba772573bc4dba8ae6114bc992ecb334a9ee4c12f2fe07d6b"
)
_CALIBRATION_RECONSTRUCTED_CALENDAR_DIGEST = (
    "819b8830d97e3d262893c97115ac40f4a70fb206bc6f3b3fc205db198d511729"
)
_CALIBRATION_CENSORING_CONTRACT_DIGEST = (
    "fca72bcb93d65caef4797da5193b8c9eafcbcf86adc336668520bbac3258a53a"
)
_CALIBRATION_RECONSTRUCTION_FILES = (
    "ETRI_Human_AI_v2/src/semantic_indicators/contracts.py",
    "ETRI_Human_AI_v2/src/semantic_indicators/primitives.py",
    "ETRI_Human_AI_v2/src/semantic_indicators/personalization.py",
    "ETRI_Human_AI_v2/src/semantic_indicators/reliability.py",
    "ETRI_Human_AI_v2/src/semantic_indicators/phaseguard_authority.py",
)
_CALIBRATION_HISTORICAL_FINGERPRINTS = (
    (
        "raw_inputs_sha256",
        "3cae1aeb2962dc6c15ca5756075cdfb150c1d2eb21ba368fc3761920bebb30f5",
    ),
    (
        "canonical_inputs_sha256",
        "2b270e686e025965a30fdb447a4121f531f376f5e4f8422460b9aa2769f35f83",
    ),
    (
        "sensors_sha256",
        "d558bac4736eb0ee39b48c406b7d00688206818f86d67d574ac434ecc2bc2fdb",
    ),
)

_OFFICIAL_MANIFEST_FIELDS = (
    "schema_version",
    "status",
    "common_fingerprints",
    "chunks",
    "scenario_grid",
    "draws_per_scenario",
    "primary_scenario",
    "official_run_manifest",
    "primitive_gates",
    "pair_gates",
    "domain_gates",
    "decision_table",
    "participant_macro_metrics",
    "confidence_summary",
    "calibration_summary",
    "modality_summary",
    "empirical_gap_determinism",
    "official_artifacts",
    "totals",
    "partial_artifact_policy",
    "execution_parallelism_policy",
)
_OFFICIAL_ARTIFACT_NAMES = (
    "selected_days",
    "cell_replays",
    "mask_intervals",
    "deletion_audit",
    "draw_day_metrics",
    "participant_draw_metrics",
    "participant_metrics",
    "participant_macro_metrics",
    "primitive_gates",
    "pair_gates",
    "domain_gates",
    "decision_table",
    "confidence_ablation",
    "confidence_summary",
    "run_manifest",
    "calibration_sensitivity",
    "calibration_summary",
    "modality_sensitivity",
    "modality_summary",
)
_OFFICIAL_ARTIFACT_RECORD_FIELDS = (
    "path",
    "format",
    "rows",
    "columns",
    "bytes",
    "sha256",
)
_OFFICIAL_ARTIFACT_DECLARATIONS_DIGEST = (
    "f5c690192fbcab9c4a2f0f2ba5f52b43c91932c8e641a07e3178732271181363"
)
_OFFICIAL_TOP_SEMANTICS_DIGEST = (
    "d4979d01e41fa54a793006a6e69c3a97b1adcdfe6a83a615ceaf80406773d8d7"
)
_OFFICIAL_SMALL_LOGICAL_DIGEST = (
    "dbcfe9f3398bb91cf911a883c408983a88e5faff8fb477d260c09d618cae695b"
)
_OFFICIAL_MANIFEST_LOGICAL_DIGEST = (
    "e446f18030ebc12f85ea8454a3d2fb6ea1136a3126dacd9bb84f23804fcd99e8"
)
_OFFICIAL_TOP_SEMANTIC_FIELDS = (
    "schema_version",
    "status",
    "common_fingerprints",
    "scenario_grid",
    "draws_per_scenario",
    "primary_scenario",
    "official_run_manifest",
    "totals",
    "partial_artifact_policy",
    "execution_parallelism_policy",
)
_OFFICIAL_SMALL_LOGICAL_FIELDS = (
    "primitive_gates",
    "pair_gates",
    "domain_gates",
    "decision_table",
    "participant_macro_metrics",
    "confidence_summary",
    "calibration_summary",
    "modality_summary",
)

_HISTORICAL_COMMON_FINGERPRINTS = (
    ("root_seed", "42"),
    (
        "raw_inputs_sha256",
        "3cae1aeb2962dc6c15ca5756075cdfb150c1d2eb21ba368fc3761920bebb30f5",
    ),
    (
        "canonical_inputs_sha256",
        "2b270e686e025965a30fdb447a4121f531f376f5e4f8422460b9aa2769f35f83",
    ),
    (
        "primitives_sha256",
        "6490ece513f26d1056aa0e4cd2cfb9220910b3095e887d0f8e68561f07b1a230",
    ),
    (
        "baselines_sha256",
        "81f6458ac1661518309883f610347ef028986c978540feef1ffb842f50ebd37c",
    ),
    (
        "representations_sha256",
        "7963fc3877419bd55d4135b97dd804afa7d3abf8e736c5c43d61b0f1d010e06c",
    ),
    (
        "sensors_sha256",
        "d558bac4736eb0ee39b48c406b7d00688206818f86d67d574ac434ecc2bc2fdb",
    ),
    (
        "selected_topology_sha256",
        "e9ced2d3d120bb96f181fc5b6bf42181d31f073d0e22ee59512cd2ae9375b13f",
    ),
    ("g1_artifact_sha256", OFFICIAL_G1_ARTIFACT_SHA256),
    ("g1_config_sha256", OFFICIAL_G1_CONFIG_SHA256),
    (
        "code_tree_sha256",
        "c183d9c36b0bde077ebddcdc51f801241c880e1f868026c5ed0c910f53a97c76",
    ),
    (
        "settings_sha256",
        "98c3e38c51cbe2e90c691fd81cf3c3b2ff3106387aaac15fbe5f38e87f6a9ff9",
    ),
    (
        "schema_sha256",
        "9fcd7290ba4c1aaa34a740829f79420c54de1e9e02d587af1a5c20f4b4f75371",
    ),
)

_SELECTED_COLUMNS = (
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
_CELL_COLUMNS = (
    "subject_id",
    "sensor_day_id",
    "lifelog_date",
    "elapsed_day_index",
    "primitive",
    "scenario",
    "draw",
    "root_seed",
    "seed",
    "donor_sensor_day_id",
    "deletion_audit_key",
    "original_value",
    "masked_value",
    "original_status",
    "masked_status",
    "retained",
    "absolute_error",
    "standardized_absolute_error",
    "standardizer",
    "standardizer_source",
    "selected_day_mad",
    "frozen_population_scale",
    "scale_floor",
    "floor_applied",
    "original_value_only_confidence",
    "original_coverage_only_confidence",
    "original_quality_confidence",
    "masked_value_only_confidence",
    "masked_coverage_only_confidence",
    "masked_quality_confidence",
    "success",
    "usage_record_deletion_stability",
    "reason",
    "status",
)
_MASK_COLUMNS = (
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
    "donor_sensor_day_id",
    "deletion_audit_key",
)
_DELETION_COLUMNS = (
    "deletion_audit_key",
    "subject_id",
    "sensor_day_id",
    "primitive",
    "scenario",
    "draw",
    "seed",
    "donor_sensor_day_id",
    "original_count",
    "deleted_count",
    "retained_count",
    "record_deletion_digest",
    "audit_representation",
    "status",
)
_DECISION_COLUMNS = (
    "level",
    "primitive",
    "pair",
    "domain",
    "gate",
    "observed",
    "comparator",
    "threshold",
    "passed",
    "can_change_domain_decision",
    "status",
    "production_eligible",
    "root_seed",
    "draws",
)

_PARQUET_SCHEMA_DIGESTS = {
    "selected_days": "e4c36b822e4a9b30ea279976d656296622196c1c68614feecc7dae8c71fedebf",
    "cell_replays": "900a8b387a584bcf6d5f8870eb1c07e405ec72577ea1e07354540b2f2cbe1948",
    "mask_intervals": "3e72e6994673cae1e637e487dbc79c3720fc9ab8e472c0965e3f11dd09b39475",
    "deletion_audit": "c74971636c8234813b8e2940e238250174d98da6a90bd4fc1fcdd92ea4ab4a37",
    "draw_day_metrics": "24fc86a7e423c011a6898c7420df03ee646e7c9251014de9197544ae949b53bc",
    "participant_draw_metrics": "cf48cea1ceb86e0cd5390ae8e1aecc529ab316667baebda7090da39580bf97cd",
    "participant_metrics": "f1b821f5d049f9ca62dc8cec7afc890a4d735a3656d6ac38124939c4f548b4d5",
    "participant_macro_metrics": "f992c0cf78ae252e7453feb082a330ac287b63a0e8f9f0231c9bcd5418cd47fd",
    "primitive_gates": "d691e1542cdc54dc1b9ab4865c0c1196e4edbed71f9cd8431749817ea6640ea6",
    "pair_gates": "a0584162bc9245e7f73281d92c2304cc05a4955493a48aad365263fd775bd188",
    "domain_gates": "6e09afac2cf92dc72aae9f9079a24c45e7e6e72901e45a67f50bed30ec8bec9b",
    "confidence_ablation": "90cc32df1e284501e217e7b273a365e47292445eac5f501c5a39350bcfe6776a",
    "confidence_summary": "5e2da7342b39d82145bd793af4718e845375ff57221c2edce98070dee1c1627c",
    "run_manifest": "23c7929a9413c3ff66bd288ea5cc219f3550be4d0184c5a0b7fc766c9ab42894",
    "calibration_sensitivity": "f7b603a7ea7d557c546245a7728631d5621d8154681453fb7472652ff9432d71",
    "calibration_summary": "466bd63590b7c469e78281741a18931f1e0ad86f6fac497be4d293353bfba086",
    "modality_sensitivity": "7105d586b751a8698fd3c48021e880f02e0a1481c569fa4b8b24bac336c8f848",
    "modality_summary": "e8c909d84b5cec738626b84042de4840369a91b510531fc85e443d21c65a3e4f",
}

OFFICIAL_EMPIRICAL_UNAVAILABLE_RECORD_DIGEST = hashlib.sha256(
    b"empirical_gap_unavailable"
).hexdigest()
OFFICIAL_EMPIRICAL_UNAVAILABLE_INTERVAL_DIGEST = hashlib.sha256(
    b"phaseguard:official_empirical_unavailable:intervals:v1"
).hexdigest()


class AuthorityValidationError(ValueError):
    """Raised when a physical or logical authority check fails closed."""


def _fail(message: str) -> NoReturn:
    raise AuthorityValidationError(message)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        _fail(f"{label} must be an exact lowercase 64-hex SHA-256")
    return value


def _require_exact_dict(
    value: object,
    expected_fields: set[str],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"{label} must be an exact JSON object")
    actual_fields = set(value)
    if actual_fields != expected_fields:
        _fail(
            f"{label} fields mismatch: "
            f"missing={sorted(expected_fields - actual_fields)!r} "
            f"extra={sorted(actual_fields - expected_fields)!r}"
        )
    return value


def _object_without_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> NoReturn:
    _fail(f"non-finite JSON token: {token}")


def _require_runtime_path(value: object, label: str) -> Path:
    if type(value) is not _PATH_TYPE:
        raise TypeError(f"{label} must be an exact pathlib Path")
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve {label}: {exc}")
    if not resolved.is_file():
        _fail(f"{label} must identify a regular file")
    return resolved


def _read_canonical_json(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, object], str]:
    expected = _require_sha256(expected_sha256, f"expected {label} SHA-256")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read {label}: {exc}")
    actual = _sha256_bytes(raw)
    if actual != expected:
        _fail(
            f"{label} physical hash mismatch: expected={expected} actual={actual}"
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail(f"{label} must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"{label} is not UTF-8: {exc}")
    try:
        document = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite_json,
        )
    except AuthorityValidationError:
        raise
    except json.JSONDecodeError as exc:
        _fail(
            f"{label} is not JSON: {exc.msg} at line {exc.lineno} "
            f"column {exc.colno}"
        )
    if raw != _canonical_json_bytes(document):
        _fail(f"{label} bytes are not exact canonical UTF-8 JSON")
    if type(document) is not dict:
        _fail(f"{label} must contain one exact JSON object")
    return document, actual


def _require_content_digest(document: dict[str, object], label: str) -> str:
    actual = _require_sha256(document["content_digest"], f"{label}.content_digest")
    logical = dict(document)
    del logical["content_digest"]
    expected = _sha256_bytes(_canonical_json_bytes(logical))
    if actual != expected:
        _fail(
            f"{label}.content_digest mismatch: expected={expected} actual={actual}"
        )
    return actual


def _find_repo_root(path: Path, label: str) -> Path:
    for candidate in path.parents:
        if (candidate / ".git").exists():
            return candidate.resolve()
    _fail(f"{label} is not inside a Git worktree")


def _repo_relative_path(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value or ":" in value:
        _fail(f"{label} must be a nonempty POSIX repo-relative string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{label} is not a canonical nontraversing path: {value!r}")
    if path.as_posix() != value:
        _fail(f"{label} is not canonical: {value!r}")
    return value


def _resolve_repo_file(repo_root: Path, path_text: str, label: str) -> Path:
    relative = _repo_relative_path(path_text, label)
    candidate = repo_root.joinpath(*PurePosixPath(relative).parts)
    try:
        lexical = repo_root
        for part in PurePosixPath(relative).parts:
            lexical = lexical / part
            if lexical.is_symlink() or _is_reparse_point(lexical):
                _fail(f"{label} uses a lexical symlink, junction, or reparse point")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repo_root)
    except (OSError, ValueError) as exc:
        _fail(f"{label} escapes or is absent from the repository: {exc}")
    if candidate != resolved or str(candidate) != str(resolved):
        _fail(f"{label} loses lexical identity after path resolution")
    if not resolved.is_file():
        _fail(f"{label} must identify a regular file")
    return resolved


def _rehash_repo_file(
    repo_root: Path,
    path_text: str,
    expected_sha256: str,
    label: str,
) -> Path:
    path = _resolve_repo_file(repo_root, path_text, label)
    try:
        actual = _sha256_bytes(path.read_bytes())
    except OSError as exc:
        _fail(f"cannot read {label}: {exc}")
    if actual != expected_sha256:
        _fail(
            f"{label} physical hash mismatch: "
            f"expected={expected_sha256} actual={actual}"
        )
    return path


def _require_string_array(value: object, label: str) -> list[str]:
    if type(value) is not list or any(
        type(item) is not str or not item for item in value
    ):
        _fail(f"{label} must be an exact JSON array of nonempty strings")
    if len(set(value)) != len(value):
        _fail(f"{label} must not contain duplicates")
    return value


@dataclass(frozen=True)
class ReviewArtifactPin:
    repo_relative_path: str
    expected_sha256: str
    role: Literal["review_package", "predecessor_finding", "evidence_report"]
    content_digest: str


@dataclass(frozen=True)
class Task8ACleanExpectations:
    schema_name: Literal["task8a-clean-expectations-v1"]
    expectations_path: Path
    expected_expectations_sha256: str
    expectations_sha256: str
    required_finding_ids: tuple[str, ...]
    required_finding_ids_digest: str
    latest_review_package: ReviewArtifactPin
    predecessor_finding_chain: tuple[ReviewArtifactPin, ...]
    predecessor_finding_chain_digest: str
    required_file_hashes: tuple[tuple[str, str], ...]
    required_file_hashes_digest: str
    required_evidence_report_pins: tuple[ReviewArtifactPin, ...]
    expected_reviewed_tree_sha256: str
    content_digest: str


@dataclass(frozen=True)
class Task8ACleanAuthority:
    schema_name: Literal["task8a-clean-v3"]
    report_path: Path
    expected_report_sha256: str
    report_sha256: str
    expectations: Task8ACleanExpectations
    expectations_digest: str
    verdict: Literal["CLEAN"]
    required_finding_ids: tuple[str, ...]
    required_finding_ids_digest: str
    closed_finding_ids: tuple[str, ...]
    latest_review_package: ReviewArtifactPin
    predecessor_finding_chain: tuple[ReviewArtifactPin, ...]
    predecessor_finding_chain_digest: str
    reviewed_file_hashes: tuple[tuple[str, str], ...]
    reviewed_evidence_report_pins: tuple[ReviewArtifactPin, ...]
    reviewed_tree_sha256: str
    content_digest: str


@dataclass(frozen=True)
class Task0BReviewEvidenceAuthority:
    schema_name: Literal["task0b-review-evidence-v1"]
    state: Literal["review_only"]
    review_root: Path
    reviewed_scope_ids: tuple[str, ...]
    evidence_artifact_pins: tuple[ReviewArtifactPin, ...]
    task8a_clean_authority_digest: str
    nonrelease_sentinel_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_name) is not str
            or self.schema_name != "task0b-review-evidence-v1"
        ):
            _fail("review evidence schema must be task0b-review-evidence-v1")
        if type(self.state) is not str or self.state != "review_only":
            _fail("Task 0B evidence state must remain review_only")
        if type(self.review_root) is not _PATH_TYPE or not self.review_root.is_absolute():
            _fail("review_root must be an exact absolute pathlib Path")
        try:
            resolved_review_root = self.review_root.resolve(strict=False)
        except OSError as exc:
            _fail(f"review_root cannot be resolved safely: {exc}")
        if resolved_review_root != self.review_root:
            _fail("review_root must not traverse a symlink, junction, or alias")
        forbidden = (
            ".official",
            "sealed_chunk",
            "no-mask",
            "no_mask",
            "benchmark",
            "exact-50",
            "exact_50",
            "aggregate",
        )
        review_texts = (
            self.review_root.as_posix().casefold(),
            resolved_review_root.as_posix().casefold(),
        )
        if any(token in text for text in review_texts for token in forbidden):
            _fail("review_root uses a release-only namespace")
        if type(self.reviewed_scope_ids) is not tuple or not self.reviewed_scope_ids:
            _fail("reviewed_scope_ids must be a nonempty exact tuple")
        if any(
            type(scope_id) is not str or not scope_id
            for scope_id in self.reviewed_scope_ids
        ) or len(set(self.reviewed_scope_ids)) != len(self.reviewed_scope_ids):
            _fail("reviewed_scope_ids must contain unique nonempty exact strings")
        if type(self.evidence_artifact_pins) is not tuple or any(
            type(pin) is not ReviewArtifactPin or pin.role != "evidence_report"
            for pin in self.evidence_artifact_pins
        ):
            _fail("review evidence pins must be an exact tuple of evidence reports")
        _require_sha256(
            self.task8a_clean_authority_digest,
            "task8a_clean_authority_digest",
        )
        nonrelease_sentinel = _require_sha256(
            self.nonrelease_sentinel_digest,
            "review evidence nonrelease_sentinel_digest",
        )
        if nonrelease_sentinel != TASK0B_REVIEW_ONLY_SENTINEL_DIGEST:
            _fail("review evidence nonrelease sentinel is invalid")
        _require_sha256(self.content_digest, "review evidence content_digest")


@dataclass(frozen=True, slots=True)
class OfficialArtifact:
    name: str
    manifest_relative_path: str
    format: Literal["parquet", "jsonl"]
    rows: int
    columns: tuple[str, ...]
    byte_count: int
    physical_sha256: str
    content_digest: str

    def __post_init__(self) -> None:
        _validate_official_artifact_value(self)


@dataclass(frozen=True, slots=True)
class OfficialSelectedCell:
    subject_id: str
    sensor_day_id: int
    primitive: str
    selected_row_digest: str
    selected_days_artifact_sha256: str
    content_digest: str

    def __post_init__(self) -> None:
        _validate_official_selected_cell_value(self)


@dataclass(frozen=True, slots=True)
class MaskIntervalValue:
    mask_interval_id: int
    mask_start: pd.Timestamp
    mask_end: pd.Timestamp
    mask_duration_minutes: float
    mask_semantics: Literal["half_open"]
    row_status: str
    content_digest: str

    def __post_init__(self) -> None:
        _validate_mask_interval_value(self)


@dataclass(frozen=True, slots=True)
class OfficialDeletionAudit:
    deletion_audit_key: str
    subject_id: str
    sensor_day_id: int
    primitive: str
    scenario: str
    draw: int
    seed: str
    donor_sensor_day_id: int | None
    original_count: int
    deleted_count: int
    retained_count: int
    record_deletion_digest: str
    audit_representation: Literal[
        "normalized_source_digest_plus_mask_geometry"
    ]
    status: str
    deletion_audit_artifact_sha256: str
    content_digest: str

    def __post_init__(self) -> None:
        _validate_official_deletion_audit_value(self)


@dataclass(frozen=True, slots=True)
class OfficialMaskCell:
    subject_id: str
    sensor_day_id: int
    primitive: str
    scenario: str
    draw: int
    root_seed: str
    seed: str
    status: Literal["observed_mask", "empirical_unavailable"]
    donor_sensor_day_id: int | None
    intervals: tuple[MaskIntervalValue, ...]
    interval_digest: str
    deletion_audit: OfficialDeletionAudit
    mask_intervals_artifact_sha256: str
    unavailable_sentinel_digest: str | None
    content_digest: str

    def __post_init__(self) -> None:
        _validate_official_mask_cell_value(self)


@dataclass(frozen=True, slots=True)
class OfficialCellEvidence:
    subject_id: str
    sensor_day_id: int
    lifelog_date: pd.Timestamp
    elapsed_day_index: int
    primitive: str
    scenario: str
    draw: int
    root_seed: str
    seed: str
    donor_sensor_day_id: int | None
    deletion_audit_key: str
    deletion_audit_digest: str
    record_deletion_digest: str
    original_value: float
    masked_value: float
    original_status: str
    masked_status: str
    retained: bool
    absolute_error: float
    standardized_absolute_error: float
    standardizer: float
    standardizer_source: str
    selected_day_mad: float
    frozen_population_scale: float
    scale_floor: float
    floor_applied: bool
    masked_coverage_only_confidence: float
    masked_quality_confidence: float
    reason: str
    status: Literal["observed", "abstained"]
    cell_replays_artifact_sha256: str
    content_digest: str

    def __post_init__(self) -> None:
        _validate_official_cell_evidence_value(self)


@dataclass(frozen=True, slots=True)
class OfficialG2Authority:
    manifest_path: Path
    manifest_sha256: str
    root_seed: str
    scenario_grid: tuple[str, ...]
    draws: tuple[int, ...]
    selected_cells: tuple[OfficialSelectedCell, ...]
    cell_evidence: tuple[OfficialCellEvidence, ...]
    official_masks: tuple[OfficialMaskCell, ...]
    official_artifacts: tuple[OfficialArtifact, ...]
    historical_common_fingerprints: tuple[tuple[str, str], ...]
    g1_artifact_path: Path
    g1_artifact_sha256: str
    task8a_clean: Task8ACleanAuthority
    task8a_clean_authority_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        _validate_official_g2_authority_value(self)


@dataclass(frozen=True)
class CanonicalFrameAuthority:
    path: Path
    physical_sha256: str
    row_count: int
    columns: tuple[str, ...]
    dtypes: tuple[str, ...]
    logical_rows: tuple[tuple[str, ...], ...]
    schema_digest: str
    logical_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        _validate_canonical_frame_authority_value(self)


@dataclass(frozen=True)
class CalibrationInputAuthority:
    primitive_table: CanonicalFrameAuthority
    frozen_population_baselines: CanonicalFrameAuthority
    raw_root: Path
    canonical_root: Path
    raw_inputs_sha256: str
    canonical_inputs_sha256: str
    sensors_sha256: str
    sensor_file_hashes: tuple[tuple[str, str], ...]
    reconstruction_file_hashes: tuple[tuple[str, str], ...]
    reconstructed_calendar_digest: str
    reconstructed_primitive_table_digest: str
    reconstructed_baselines_digest: str
    calibration_censoring_contract_digest: str
    official_g2_digest: str
    task8a_clean_authority_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        _validate_calibration_input_authority_value(self)


@dataclass(frozen=True, slots=True)
class _CalibrationInputPaths:
    raw_root: Path
    canonical_root: Path
    primitive_table_path: Path
    frozen_population_baselines_path: Path


def _validate_artifact_pin(
    value: object,
    *,
    required_role: Literal[
        "review_package", "predecessor_finding", "evidence_report"
    ],
    repo_root: Path,
    label: str,
) -> ReviewArtifactPin:
    pin = _require_exact_dict(value, _PIN_FIELDS, label)
    path_text = _repo_relative_path(
        pin["repo_relative_path"], f"{label}.repo_relative_path"
    )
    expected_sha256 = _require_sha256(
        pin["expected_sha256"], f"{label}.expected_sha256"
    )
    if type(pin["role"]) is not str or pin["role"] != required_role:
        _fail(f"{label}.role must equal {required_role!r}")
    content_digest = _require_sha256(
        pin["content_digest"], f"{label}.content_digest"
    )
    logical_pin = {
        "expected_sha256": expected_sha256,
        "repo_relative_path": path_text,
        "role": required_role,
    }
    expected_content_digest = _sha256_bytes(_canonical_json_bytes(logical_pin))
    if content_digest != expected_content_digest:
        _fail(
            f"{label}.content_digest mismatch: "
            f"expected={expected_content_digest} actual={content_digest}"
        )
    _rehash_repo_file(repo_root, path_text, expected_sha256, label)
    return ReviewArtifactPin(
        repo_relative_path=path_text,
        expected_sha256=expected_sha256,
        role=required_role,
        content_digest=content_digest,
    )


def _validate_pin_array(
    value: object,
    *,
    required_role: Literal["predecessor_finding", "evidence_report"],
    repo_root: Path,
    label: str,
) -> tuple[ReviewArtifactPin, ...]:
    if type(value) is not list or not value:
        _fail(f"{label} must be a nonempty exact JSON array")
    pins = tuple(
        _validate_artifact_pin(
            item,
            required_role=required_role,
            repo_root=repo_root,
            label=f"{label}[{index}]",
        )
        for index, item in enumerate(value)
    )
    if len({pin.repo_relative_path for pin in pins}) != len(pins):
        _fail(f"{label} must not contain duplicate paths")
    return pins


def _validate_file_hashes(
    value: object,
    *,
    repo_root: Path,
    label: str,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not list or not value:
        _fail(f"{label} must be a nonempty exact JSON array")
    result: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if type(item) is not list or len(item) != 2:
            _fail(f"{label}[{index}] must be [path, expected_sha256]")
        path_text = _repo_relative_path(item[0], f"{label}[{index}][0]")
        expected_sha256 = _require_sha256(item[1], f"{label}[{index}][1]")
        _rehash_repo_file(
            repo_root,
            path_text,
            expected_sha256,
            f"{label}[{index}]",
        )
        result.append((path_text, expected_sha256))
    if len({path for path, _digest in result}) != len(result):
        _fail(f"{label} must not contain duplicate paths")
    return tuple(result)


def _pin_raw(pin: ReviewArtifactPin) -> dict[str, str]:
    return {
        "content_digest": pin.content_digest,
        "expected_sha256": pin.expected_sha256,
        "repo_relative_path": pin.repo_relative_path,
        "role": pin.role,
    }


def _hash_pin_tuple(pins: tuple[ReviewArtifactPin, ...]) -> str:
    return _sha256_bytes(_canonical_json_bytes([_pin_raw(pin) for pin in pins]))


def _hash_file_tuple(values: tuple[tuple[str, str], ...]) -> str:
    return _sha256_bytes(_canonical_json_bytes([list(item) for item in values]))


def _ensure_complete_distinct_tree_paths(
    package_pin: ReviewArtifactPin,
    chain: tuple[ReviewArtifactPin, ...],
    file_hashes: tuple[tuple[str, str], ...],
    evidence: tuple[ReviewArtifactPin, ...],
) -> None:
    paths = (
        package_pin.repo_relative_path,
        *(pin.repo_relative_path for pin in chain),
        *(path for path, _digest in file_hashes),
        *(pin.repo_relative_path for pin in evidence),
    )
    if len(set(paths)) != len(paths):
        _fail("reviewed tree paths must be complete and pairwise distinct")


def _validate_expectations(
    document: dict[str, object],
    *,
    expectations_path: Path,
    expected_expectations_sha256: str,
    expectations_sha256: str,
    repo_root: Path,
) -> Task8ACleanExpectations:
    raw = _require_exact_dict(document, _EXPECTATIONS_FIELDS, "expectations")
    if raw["schema_name"] != "task8a-clean-expectations-v1":
        _fail("expectations.schema_name must be task8a-clean-expectations-v1")
    content_digest = _require_content_digest(raw, "expectations")
    finding_ids = tuple(
        _require_string_array(raw["required_finding_ids"], "required_finding_ids")
    )
    finding_digest = _require_sha256(
        raw["required_finding_ids_digest"], "required_finding_ids_digest"
    )
    expected_finding_digest = _sha256_bytes(_canonical_json_bytes(list(finding_ids)))
    if finding_digest != expected_finding_digest:
        _fail("required_finding_ids_digest does not bind the ordered finding tuple")
    package_pin = _validate_artifact_pin(
        raw["latest_review_package"],
        required_role="review_package",
        repo_root=repo_root,
        label="latest_review_package",
    )
    chain = _validate_pin_array(
        raw["predecessor_finding_chain"],
        required_role="predecessor_finding",
        repo_root=repo_root,
        label="predecessor_finding_chain",
    )
    chain_digest = _require_sha256(
        raw["predecessor_finding_chain_digest"],
        "predecessor_finding_chain_digest",
    )
    if chain_digest != _hash_pin_tuple(chain):
        _fail("predecessor_finding_chain_digest does not bind the complete chain")
    file_hashes = _validate_file_hashes(
        raw["required_file_hashes"],
        repo_root=repo_root,
        label="required_file_hashes",
    )
    file_hashes_digest = _require_sha256(
        raw["required_file_hashes_digest"], "required_file_hashes_digest"
    )
    if file_hashes_digest != _hash_file_tuple(file_hashes):
        _fail("required_file_hashes_digest does not bind the ordered files")
    evidence = _validate_pin_array(
        raw["required_evidence_report_pins"],
        required_role="evidence_report",
        repo_root=repo_root,
        label="required_evidence_report_pins",
    )
    reviewed_tree = _require_sha256(
        raw["expected_reviewed_tree_sha256"],
        "expected_reviewed_tree_sha256",
    )
    _ensure_complete_distinct_tree_paths(package_pin, chain, file_hashes, evidence)
    return Task8ACleanExpectations(
        schema_name="task8a-clean-expectations-v1",
        expectations_path=expectations_path,
        expected_expectations_sha256=expected_expectations_sha256,
        expectations_sha256=expectations_sha256,
        required_finding_ids=finding_ids,
        required_finding_ids_digest=finding_digest,
        latest_review_package=package_pin,
        predecessor_finding_chain=chain,
        predecessor_finding_chain_digest=chain_digest,
        required_file_hashes=file_hashes,
        required_file_hashes_digest=file_hashes_digest,
        required_evidence_report_pins=evidence,
        expected_reviewed_tree_sha256=reviewed_tree,
        content_digest=content_digest,
    )


def _validate_review_package(
    expectations: Task8ACleanExpectations,
    *,
    repo_root: Path,
) -> None:
    package_path = _resolve_repo_file(
        repo_root,
        expectations.latest_review_package.repo_relative_path,
        "latest review package",
    )
    package, _package_sha256 = _read_canonical_json(
        package_path,
        expectations.latest_review_package.expected_sha256,
        "latest review package",
    )
    raw = _require_exact_dict(package, _PACKAGE_FIELDS, "latest review package")
    if raw["schema_name"] != "task8a-review-package-v1":
        _fail("latest review package schema must be task8a-review-package-v1")
    _require_content_digest(raw, "latest review package")
    package_findings = tuple(
        _require_string_array(
            raw["required_finding_ids"],
            "latest review package required_finding_ids",
        )
    )
    if package_findings != expectations.required_finding_ids:
        _fail("latest review package finding tuple differs from expectations")
    chain = _validate_pin_array(
        raw["predecessor_finding_chain"],
        required_role="predecessor_finding",
        repo_root=repo_root,
        label="latest review package predecessor_finding_chain",
    )
    if chain != expectations.predecessor_finding_chain:
        _fail("latest review package predecessor chain differs from expectations")
    file_hashes = _validate_file_hashes(
        raw["required_file_hashes"],
        repo_root=repo_root,
        label="latest review package required_file_hashes",
    )
    if file_hashes != expectations.required_file_hashes:
        _fail("latest review package current files differ from expectations")
    evidence = _validate_pin_array(
        raw["required_evidence_report_pins"],
        required_role="evidence_report",
        repo_root=repo_root,
        label="latest review package required_evidence_report_pins",
    )
    if evidence != expectations.required_evidence_report_pins:
        _fail("latest review package evidence pins differ from expectations")
    if raw["reviewed_tree_spec"] != _REVIEWED_TREE_SPEC:
        _fail("latest review package reviewed_tree_spec is not exact v1")


def _validate_report(
    document: dict[str, object],
    *,
    report_path: Path,
    expected_report_sha256: str,
    report_sha256: str,
    expectations: Task8ACleanExpectations,
    repo_root: Path,
) -> Task8ACleanAuthority:
    raw = _require_exact_dict(document, _REPORT_FIELDS, "CLEAN report")
    if raw["schema_name"] != "task8a-clean-v3":
        _fail("CLEAN report schema must be task8a-clean-v3")
    if type(raw["verdict"]) is not str or raw["verdict"] != "CLEAN":
        _fail("CLEAN report verdict must be exactly CLEAN")
    content_digest = _require_content_digest(raw, "CLEAN report")
    expectations_digest = _require_sha256(
        raw["expectations_digest"], "CLEAN report expectations_digest"
    )
    if expectations_digest != expectations.content_digest:
        _fail("CLEAN report expectations_digest is stale or crosswired")
    finding_ids = tuple(
        _require_string_array(
            raw["required_finding_ids"], "CLEAN report required_finding_ids"
        )
    )
    closed_ids = tuple(
        _require_string_array(
            raw["closed_finding_ids"], "CLEAN report closed_finding_ids"
        )
    )
    if finding_ids != expectations.required_finding_ids:
        _fail("CLEAN report finding tuple differs from expectations")
    if closed_ids != finding_ids:
        _fail("CLEAN report closed findings must equal all required findings")
    finding_digest = _require_sha256(
        raw["required_finding_ids_digest"],
        "CLEAN report required_finding_ids_digest",
    )
    if finding_digest != expectations.required_finding_ids_digest:
        _fail("CLEAN report finding digest differs from expectations")
    package_pin = _validate_artifact_pin(
        raw["latest_review_package"],
        required_role="review_package",
        repo_root=repo_root,
        label="CLEAN report latest_review_package",
    )
    if package_pin != expectations.latest_review_package:
        _fail("CLEAN report package pin differs from expectations")
    chain = _validate_pin_array(
        raw["predecessor_finding_chain"],
        required_role="predecessor_finding",
        repo_root=repo_root,
        label="CLEAN report predecessor_finding_chain",
    )
    chain_digest = _require_sha256(
        raw["predecessor_finding_chain_digest"],
        "CLEAN report predecessor_finding_chain_digest",
    )
    if chain_digest != _hash_pin_tuple(chain):
        _fail("CLEAN report predecessor chain digest is invalid")
    if (
        chain != expectations.predecessor_finding_chain
        or chain_digest != expectations.predecessor_finding_chain_digest
    ):
        _fail("CLEAN report predecessor chain differs from expectations")
    file_hashes = _validate_file_hashes(
        raw["reviewed_file_hashes"],
        repo_root=repo_root,
        label="CLEAN report reviewed_file_hashes",
    )
    if file_hashes != expectations.required_file_hashes:
        _fail("CLEAN report current files differ from expectations")
    evidence = _validate_pin_array(
        raw["reviewed_evidence_report_pins"],
        required_role="evidence_report",
        repo_root=repo_root,
        label="CLEAN report reviewed_evidence_report_pins",
    )
    if evidence != expectations.required_evidence_report_pins:
        _fail("CLEAN report evidence pins differ from expectations")
    reviewed_tree = _require_sha256(
        raw["reviewed_tree_sha256"], "CLEAN report reviewed_tree_sha256"
    )
    if reviewed_tree != expectations.expected_reviewed_tree_sha256:
        _fail("CLEAN report reviewed tree differs from expectations")
    return Task8ACleanAuthority(
        schema_name="task8a-clean-v3",
        report_path=report_path,
        expected_report_sha256=expected_report_sha256,
        report_sha256=report_sha256,
        expectations=expectations,
        expectations_digest=expectations_digest,
        verdict="CLEAN",
        required_finding_ids=finding_ids,
        required_finding_ids_digest=finding_digest,
        closed_finding_ids=closed_ids,
        latest_review_package=package_pin,
        predecessor_finding_chain=chain,
        predecessor_finding_chain_digest=chain_digest,
        reviewed_file_hashes=file_hashes,
        reviewed_evidence_report_pins=evidence,
        reviewed_tree_sha256=reviewed_tree,
        content_digest=content_digest,
    )


def _recompute_reviewed_tree(authority: Task8ACleanAuthority) -> str:
    leaves: list[list[str]] = [
        [
            "review_package",
            authority.latest_review_package.repo_relative_path,
            authority.latest_review_package.expected_sha256,
        ]
    ]
    leaves.extend(
        [pin.role, pin.repo_relative_path, pin.expected_sha256]
        for pin in authority.predecessor_finding_chain
    )
    if len(authority.reviewed_file_hashes) != len(_REVIEWED_FILE_ROLES):
        _fail("reviewed current-file count differs from the v1 tree roles")
    leaves.extend(
        [role, path, expected_sha256]
        for role, (path, expected_sha256) in zip(
            _REVIEWED_FILE_ROLES,
            authority.reviewed_file_hashes,
            strict=True,
        )
    )
    leaves.extend(
        [pin.role, pin.repo_relative_path, pin.expected_sha256]
        for pin in authority.reviewed_evidence_report_pins
    )
    return _sha256_bytes(_canonical_json_bytes(leaves))


def load_task8a_clean_authority(
    report_path: Path,
    expected_report_sha256: str,
    expectations_path: Path,
    expected_expectations_sha256: str,
) -> Task8ACleanAuthority:
    """Load and independently revalidate the complete Task 8A CLEAN graph."""

    report = _require_runtime_path(report_path, "report_path")
    expectations_file = _require_runtime_path(
        expectations_path,
        "expectations_path",
    )
    expected_report = _require_sha256(
        expected_report_sha256,
        "expected_report_sha256",
    )
    expected_expectations = _require_sha256(
        expected_expectations_sha256,
        "expected_expectations_sha256",
    )
    if report == expectations_file:
        _fail("report and expectation paths must be distinct")
    report_root = _find_repo_root(report, "report_path")
    expectations_root = _find_repo_root(expectations_file, "expectations_path")
    if report_root != expectations_root:
        _fail("report and expectation paths must share one Git worktree")

    expectations_document, expectations_sha256 = _read_canonical_json(
        expectations_file,
        expected_expectations,
        "Task 8A expectations",
    )
    expectations = _validate_expectations(
        expectations_document,
        expectations_path=expectations_file,
        expected_expectations_sha256=expected_expectations,
        expectations_sha256=expectations_sha256,
        repo_root=report_root,
    )
    _validate_review_package(expectations, repo_root=report_root)

    report_document, report_sha256 = _read_canonical_json(
        report,
        expected_report,
        "Task 8A CLEAN report",
    )
    authority = _validate_report(
        report_document,
        report_path=report,
        expected_report_sha256=expected_report,
        report_sha256=report_sha256,
        expectations=expectations,
        repo_root=report_root,
    )
    recomputed_tree = _recompute_reviewed_tree(authority)
    if recomputed_tree != authority.reviewed_tree_sha256:
        _fail(
            "reviewed tree digest mismatch: "
            f"expected={authority.reviewed_tree_sha256} actual={recomputed_tree}"
        )
    return authority


def _require_nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be an exact nonempty string")
    return value


def _require_exact_integer(
    value: object,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an exact integer >= {minimum}")
    return value


def _require_exact_boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be an exact boolean")
    return value


def _float_token(
    value: object,
    label: str,
    *,
    allow_nan: bool = True,
) -> str:
    if type(value) is not float:
        _fail(f"{label} must be an exact float")
    if math.isnan(value):
        if not allow_nan:
            _fail(f"{label} must be finite")
        return "nan"
    if not math.isfinite(value):
        _fail(f"{label} must not be infinite")
    return value.hex()


def _timestamp_token(value: object, label: str) -> str:
    if type(value) is not pd.Timestamp:
        _fail(f"{label} must be an exact pandas Timestamp")
    if pd.isna(value) or value.tz is not None:
        _fail(f"{label} must be a finite timezone-naive timestamp")
    return value.isoformat(timespec="nanoseconds")


def _require_donor(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _require_exact_integer(value, label)


def _canonical_value_token(value: object, label: str) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        return _float_token(value, label)
    if type(value) is pd.Timestamp:
        return _timestamp_token(value, label)
    _fail(f"{label} has a noncanonical scalar type: {type(value).__name__}")


def _tagged_digest(schema_name: str, values: dict[str, object]) -> str:
    return _sha256_bytes(
        _canonical_json_bytes({"schema_name": schema_name, **values})
    )


def _require_content_digest_value(
    actual: object,
    expected: str,
    label: str,
) -> str:
    value = _require_sha256(actual, f"{label}.content_digest")
    if value != expected:
        _fail(
            f"{label}.content_digest mismatch: expected={expected} actual={value}"
        )
    return value


def _artifact_payload(value: OfficialArtifact) -> dict[str, object]:
    return {
        "byte_count": value.byte_count,
        "columns": list(value.columns),
        "format": value.format,
        "manifest_relative_path": value.manifest_relative_path,
        "name": value.name,
        "physical_sha256": value.physical_sha256,
        "rows": value.rows,
    }


def _validate_official_artifact_value(value: OfficialArtifact) -> None:
    _require_nonempty_string(value.name, "official artifact name")
    _require_manifest_relative_path(
        value.manifest_relative_path,
        "official artifact manifest_relative_path",
    )
    if type(value.format) is not str or value.format not in {"parquet", "jsonl"}:
        _fail("official artifact format must be exactly parquet or jsonl")
    _require_exact_integer(value.rows, "official artifact rows")
    if type(value.columns) is not tuple or not value.columns:
        _fail("official artifact columns must be a nonempty exact tuple")
    if any(type(column) is not str or not column for column in value.columns):
        _fail("official artifact columns must contain exact nonempty strings")
    if len(set(value.columns)) != len(value.columns):
        _fail("official artifact columns must be unique")
    _require_exact_integer(value.byte_count, "official artifact byte_count", minimum=1)
    _require_sha256(value.physical_sha256, "official artifact physical_sha256")
    expected = _tagged_digest("official-artifact-v1", _artifact_payload(value))
    _require_content_digest_value(value.content_digest, expected, "official artifact")


def _make_official_artifact(
    *,
    name: str,
    manifest_relative_path: str,
    format: Literal["parquet", "jsonl"],
    rows: int,
    columns: tuple[str, ...],
    byte_count: int,
    physical_sha256: str,
) -> OfficialArtifact:
    provisional = {
        "byte_count": byte_count,
        "columns": list(columns),
        "format": format,
        "manifest_relative_path": manifest_relative_path,
        "name": name,
        "physical_sha256": physical_sha256,
        "rows": rows,
    }
    return OfficialArtifact(
        name=name,
        manifest_relative_path=manifest_relative_path,
        format=format,
        rows=rows,
        columns=columns,
        byte_count=byte_count,
        physical_sha256=physical_sha256,
        content_digest=_tagged_digest("official-artifact-v1", provisional),
    )


def _selected_cell_payload(value: OfficialSelectedCell) -> dict[str, object]:
    return {
        "primitive": value.primitive,
        "selected_days_artifact_sha256": value.selected_days_artifact_sha256,
        "selected_row_digest": value.selected_row_digest,
        "sensor_day_id": value.sensor_day_id,
        "subject_id": value.subject_id,
    }


def _validate_official_selected_cell_value(value: OfficialSelectedCell) -> None:
    _require_nonempty_string(value.subject_id, "selected subject_id")
    _require_exact_integer(value.sensor_day_id, "selected sensor_day_id")
    _require_nonempty_string(value.primitive, "selected primitive")
    _require_sha256(value.selected_row_digest, "selected row digest")
    _require_sha256(
        value.selected_days_artifact_sha256,
        "selected-days artifact SHA-256",
    )
    expected = _tagged_digest(
        "official-selected-cell-v1", _selected_cell_payload(value)
    )
    _require_content_digest_value(value.content_digest, expected, "selected cell")


def _interval_payload(value: MaskIntervalValue) -> dict[str, object]:
    return {
        "mask_duration_minutes": _float_token(
            value.mask_duration_minutes,
            "mask interval duration",
            allow_nan=False,
        ),
        "mask_end": _timestamp_token(value.mask_end, "mask interval end"),
        "mask_interval_id": value.mask_interval_id,
        "mask_semantics": value.mask_semantics,
        "mask_start": _timestamp_token(value.mask_start, "mask interval start"),
        "row_status": value.row_status,
    }


def _validate_mask_interval_value(value: MaskIntervalValue) -> None:
    _require_exact_integer(value.mask_interval_id, "mask_interval_id")
    _timestamp_token(value.mask_start, "mask_start")
    _timestamp_token(value.mask_end, "mask_end")
    if value.mask_start >= value.mask_end:
        _fail("mask interval must be positive and half-open")
    duration_token = _float_token(
        value.mask_duration_minutes,
        "mask_duration_minutes",
        allow_nan=False,
    )
    if value.mask_duration_minutes <= 0.0:
        _fail("mask_duration_minutes must be positive")
    recomputed = (value.mask_end - value.mask_start).total_seconds() / 60.0
    if duration_token != float(recomputed).hex():
        _fail("mask duration does not equal its exact endpoints")
    if type(value.mask_semantics) is not str or value.mask_semantics != "half_open":
        _fail("mask semantics must be exactly half_open")
    _require_nonempty_string(value.row_status, "mask interval row_status")
    expected = _tagged_digest("official-mask-interval-v1", _interval_payload(value))
    _require_content_digest_value(value.content_digest, expected, "mask interval")


def _historical_seed(
    root_seed: str,
    scenario: str,
    draw: int,
    subject_id: str,
    sensor_day_id: int,
    primitive: str,
) -> str:
    identity = [
        root_seed,
        scenario,
        str(draw),
        subject_id,
        str(sensor_day_id),
        primitive,
    ]
    return _sha256_bytes(
        json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _historical_deletion_key(seed: str, record_digest: str) -> str:
    return _sha256_bytes(
        json.dumps(
            [seed, record_digest],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _deletion_payload(value: OfficialDeletionAudit) -> dict[str, object]:
    return {
        "audit_representation": value.audit_representation,
        "deleted_count": value.deleted_count,
        "deletion_audit_artifact_sha256": value.deletion_audit_artifact_sha256,
        "deletion_audit_key": value.deletion_audit_key,
        "donor_sensor_day_id": value.donor_sensor_day_id,
        "draw": value.draw,
        "original_count": value.original_count,
        "primitive": value.primitive,
        "record_deletion_digest": value.record_deletion_digest,
        "retained_count": value.retained_count,
        "scenario": value.scenario,
        "seed": value.seed,
        "sensor_day_id": value.sensor_day_id,
        "status": value.status,
        "subject_id": value.subject_id,
    }


def _validate_official_deletion_audit_value(
    value: OfficialDeletionAudit,
) -> None:
    key = _require_sha256(value.deletion_audit_key, "deletion_audit_key")
    _require_nonempty_string(value.subject_id, "deletion subject_id")
    _require_exact_integer(value.sensor_day_id, "deletion sensor_day_id")
    _require_nonempty_string(value.primitive, "deletion primitive")
    _require_nonempty_string(value.scenario, "deletion scenario")
    _require_exact_integer(value.draw, "deletion draw")
    seed = _require_sha256(value.seed, "deletion seed")
    _require_donor(value.donor_sensor_day_id, "deletion donor_sensor_day_id")
    original = _require_exact_integer(value.original_count, "original_count")
    deleted = _require_exact_integer(value.deleted_count, "deleted_count")
    retained = _require_exact_integer(value.retained_count, "retained_count")
    if original != deleted + retained:
        _fail("deletion audit counts must satisfy original=deleted+retained")
    record_digest = _require_sha256(
        value.record_deletion_digest,
        "record_deletion_digest",
    )
    if key != _historical_deletion_key(seed, record_digest):
        _fail("deletion_audit_key does not bind seed and record deletion digest")
    if (
        type(value.audit_representation) is not str
        or value.audit_representation
        != "normalized_source_digest_plus_mask_geometry"
    ):
        _fail("deletion audit representation is not exact")
    _require_nonempty_string(value.status, "deletion status")
    _require_sha256(
        value.deletion_audit_artifact_sha256,
        "deletion audit artifact SHA-256",
    )
    expected = _tagged_digest("official-deletion-audit-v1", _deletion_payload(value))
    _require_content_digest_value(value.content_digest, expected, "deletion audit")


def _unavailable_sentinel_payload(value: OfficialMaskCell) -> dict[str, object]:
    return {
        "deleted_count": value.deletion_audit.deleted_count,
        "deletion_audit_artifact_sha256": (
            value.deletion_audit.deletion_audit_artifact_sha256
        ),
        "deletion_audit_key": value.deletion_audit.deletion_audit_key,
        "donor_sensor_day_id": value.donor_sensor_day_id,
        "draw": value.draw,
        "interval_digest": value.interval_digest,
        "mask_intervals_artifact_sha256": value.mask_intervals_artifact_sha256,
        "original_count": value.deletion_audit.original_count,
        "primitive": value.primitive,
        "record_deletion_digest": value.deletion_audit.record_deletion_digest,
        "retained_count": value.deletion_audit.retained_count,
        "root_seed": value.root_seed,
        "scenario": value.scenario,
        "seed": value.seed,
        "sensor_day_id": value.sensor_day_id,
        "status": value.status,
        "subject_id": value.subject_id,
    }


def _mask_payload(value: OfficialMaskCell) -> dict[str, object]:
    return {
        "deletion_audit_digest": value.deletion_audit.content_digest,
        "donor_sensor_day_id": value.donor_sensor_day_id,
        "draw": value.draw,
        "interval_digest": value.interval_digest,
        "interval_row_digests": [item.content_digest for item in value.intervals],
        "mask_intervals_artifact_sha256": value.mask_intervals_artifact_sha256,
        "primitive": value.primitive,
        "root_seed": value.root_seed,
        "scenario": value.scenario,
        "seed": value.seed,
        "sensor_day_id": value.sensor_day_id,
        "status": value.status,
        "subject_id": value.subject_id,
        "unavailable_sentinel_digest": value.unavailable_sentinel_digest,
    }


def _validate_official_mask_cell_value(value: OfficialMaskCell) -> None:
    subject = _require_nonempty_string(value.subject_id, "mask subject_id")
    day = _require_exact_integer(value.sensor_day_id, "mask sensor_day_id")
    primitive = _require_nonempty_string(value.primitive, "mask primitive")
    scenario = _require_nonempty_string(value.scenario, "mask scenario")
    draw = _require_exact_integer(value.draw, "mask draw")
    if type(value.root_seed) is not str or value.root_seed != "42":
        _fail("official mask root_seed must be the exact string '42'")
    seed = _require_sha256(value.seed, "official mask seed")
    if seed != _historical_seed(
        value.root_seed, scenario, draw, subject, day, primitive
    ):
        _fail("official mask seed is crosswired")
    if type(value.status) is not str or value.status not in {
        "observed_mask",
        "empirical_unavailable",
    }:
        _fail("official mask status is invalid")
    donor = _require_donor(value.donor_sensor_day_id, "mask donor_sensor_day_id")
    if type(value.intervals) is not tuple or any(
        type(item) is not MaskIntervalValue for item in value.intervals
    ):
        _fail("official mask intervals must be an exact interval tuple")
    previous_end: pd.Timestamp | None = None
    for interval_index, interval in enumerate(value.intervals):
        _validate_mask_interval_value(interval)
        if interval.mask_interval_id != interval_index:
            _fail("official mask interval ids must be zero-based and contiguous")
        if previous_end is not None and interval.mask_start < previous_end:
            _fail("official mask intervals must be canonically ordered/nonoverlapping")
        previous_end = interval.mask_end
    interval_digest = _require_sha256(value.interval_digest, "interval_digest")
    if type(value.deletion_audit) is not OfficialDeletionAudit:
        _fail("official mask deletion audit must have the exact type")
    audit = value.deletion_audit
    _validate_official_deletion_audit_value(audit)
    if (
        (audit.subject_id, audit.sensor_day_id, audit.primitive, audit.scenario, audit.draw)
        != (subject, day, primitive, scenario, draw)
        or audit.seed != seed
        or audit.donor_sensor_day_id != donor
    ):
        _fail("official mask and nested deletion audit are crosswired")
    _require_sha256(
        value.mask_intervals_artifact_sha256,
        "mask intervals artifact SHA-256",
    )
    if value.status == "observed_mask":
        if not value.intervals:
            _fail("observed official masks require nonempty intervals")
        expected_interval_digest = _sha256_bytes(
            _canonical_json_bytes([item.content_digest for item in value.intervals])
        )
        if interval_digest != expected_interval_digest:
            _fail("observed interval digest does not bind every interval")
        if value.unavailable_sentinel_digest is not None:
            _fail("observed masks cannot carry an unavailable sentinel")
    else:
        if scenario != "empirical_gap" or value.intervals or donor is not None:
            _fail("empirical_unavailable mask topology is invalid")
        if interval_digest != OFFICIAL_EMPIRICAL_UNAVAILABLE_INTERVAL_DIGEST:
            _fail("empirical unavailable interval sentinel is invalid")
        if (
            audit.status != "empirical_gap_unavailable"
            or (audit.original_count, audit.deleted_count, audit.retained_count)
            != (0, 0, 0)
            or audit.record_deletion_digest
            != OFFICIAL_EMPIRICAL_UNAVAILABLE_RECORD_DIGEST
        ):
            _fail("empirical unavailable deletion audit is invalid")
        unavailable_digest = _require_sha256(
            value.unavailable_sentinel_digest,
            "unavailable_sentinel_digest",
        )
        expected_unavailable = _tagged_digest(
            "official-empirical-unavailable-v1",
            _unavailable_sentinel_payload(value),
        )
        if unavailable_digest != expected_unavailable:
            _fail("empirical unavailable sentinel is crosswired")
    expected = _tagged_digest("official-mask-cell-v1", _mask_payload(value))
    _require_content_digest_value(value.content_digest, expected, "official mask")


def _cell_payload(value: OfficialCellEvidence) -> dict[str, object]:
    return {
        "absolute_error": _float_token(value.absolute_error, "absolute_error"),
        "cell_replays_artifact_sha256": value.cell_replays_artifact_sha256,
        "deletion_audit_digest": value.deletion_audit_digest,
        "deletion_audit_key": value.deletion_audit_key,
        "donor_sensor_day_id": value.donor_sensor_day_id,
        "draw": value.draw,
        "elapsed_day_index": value.elapsed_day_index,
        "floor_applied": value.floor_applied,
        "frozen_population_scale": _float_token(
            value.frozen_population_scale,
            "frozen_population_scale",
            allow_nan=False,
        ),
        "lifelog_date": _timestamp_token(value.lifelog_date, "lifelog_date"),
        "masked_coverage_only_confidence": _float_token(
            value.masked_coverage_only_confidence,
            "masked_coverage_only_confidence",
            allow_nan=False,
        ),
        "masked_quality_confidence": _float_token(
            value.masked_quality_confidence,
            "masked_quality_confidence",
            allow_nan=False,
        ),
        "masked_status": value.masked_status,
        "masked_value": _float_token(value.masked_value, "masked_value"),
        "original_status": value.original_status,
        "original_value": _float_token(
            value.original_value,
            "original_value",
            allow_nan=False,
        ),
        "primitive": value.primitive,
        "reason": value.reason,
        "record_deletion_digest": value.record_deletion_digest,
        "retained": value.retained,
        "root_seed": value.root_seed,
        "scale_floor": _float_token(
            value.scale_floor,
            "scale_floor",
            allow_nan=False,
        ),
        "scenario": value.scenario,
        "seed": value.seed,
        "selected_day_mad": _float_token(
            value.selected_day_mad,
            "selected_day_mad",
            allow_nan=False,
        ),
        "sensor_day_id": value.sensor_day_id,
        "standardized_absolute_error": _float_token(
            value.standardized_absolute_error,
            "standardized_absolute_error",
        ),
        "standardizer": _float_token(
            value.standardizer,
            "standardizer",
            allow_nan=False,
        ),
        "standardizer_source": value.standardizer_source,
        "status": value.status,
        "subject_id": value.subject_id,
    }


def _validate_official_cell_evidence_value(value: OfficialCellEvidence) -> None:
    subject = _require_nonempty_string(value.subject_id, "cell subject_id")
    day = _require_exact_integer(value.sensor_day_id, "cell sensor_day_id")
    _timestamp_token(value.lifelog_date, "cell lifelog_date")
    if value.lifelog_date != value.lifelog_date.normalize():
        _fail("cell lifelog_date must be an exact local midnight")
    _require_exact_integer(value.elapsed_day_index, "elapsed_day_index", minimum=1)
    primitive = _require_nonempty_string(value.primitive, "cell primitive")
    scenario = _require_nonempty_string(value.scenario, "cell scenario")
    draw = _require_exact_integer(value.draw, "cell draw")
    if type(value.root_seed) is not str or value.root_seed != "42":
        _fail("official cell root_seed must be the exact string '42'")
    seed = _require_sha256(value.seed, "official cell seed")
    if seed != _historical_seed(
        value.root_seed, scenario, draw, subject, day, primitive
    ):
        _fail("official cell seed is crosswired")
    _require_donor(value.donor_sensor_day_id, "cell donor_sensor_day_id")
    _require_sha256(value.deletion_audit_key, "cell deletion_audit_key")
    _require_sha256(value.deletion_audit_digest, "cell deletion_audit_digest")
    _require_sha256(value.record_deletion_digest, "cell record_deletion_digest")
    _float_token(value.original_value, "original_value", allow_nan=False)
    masked_token = _float_token(value.masked_value, "masked_value")
    _require_nonempty_string(value.original_status, "original_status")
    _require_nonempty_string(value.masked_status, "masked_status")
    retained = _require_exact_boolean(value.retained, "retained")
    absolute_token = _float_token(value.absolute_error, "absolute_error")
    standardized_token = _float_token(
        value.standardized_absolute_error,
        "standardized_absolute_error",
    )
    standardizer_token = _float_token(
        value.standardizer,
        "standardizer",
        allow_nan=False,
    )
    if value.standardizer <= 0.0:
        _fail("official standardizer must be positive")
    _require_nonempty_string(value.standardizer_source, "standardizer_source")
    for field_name in (
        "selected_day_mad",
        "frozen_population_scale",
        "scale_floor",
    ):
        field_value = getattr(value, field_name)
        _float_token(field_value, field_name, allow_nan=False)
        if field_value < 0.0:
            _fail(f"{field_name} must be nonnegative")
    _require_exact_boolean(value.floor_applied, "floor_applied")
    for field_name in (
        "masked_coverage_only_confidence",
        "masked_quality_confidence",
    ):
        confidence = getattr(value, field_name)
        _float_token(confidence, field_name, allow_nan=False)
        if not 0.0 <= confidence <= 1.0:
            _fail(f"{field_name} must lie in [0,1]")
    _require_nonempty_string(value.reason, "cell reason")
    if type(value.status) is not str or value.status not in {"observed", "abstained"}:
        _fail("cell status must be exactly observed or abstained")
    if retained:
        if (
            masked_token == "nan"
            or value.masked_status != "observed"
            or value.status != "observed"
            or value.reason != "retained"
        ):
            _fail("retained official cell semantics are inconsistent")
        expected_absolute = abs(value.masked_value - value.original_value)
        if absolute_token != expected_absolute.hex():
            _fail("absolute_error is not exact")
        expected_standardized = expected_absolute / value.standardizer
        if standardized_token != expected_standardized.hex():
            _fail("standardized_absolute_error is not exact")
    else:
        if (
            absolute_token != "nan"
            or standardized_token != "nan"
            or value.status != "abstained"
        ):
            _fail("abstained official cell must retain missing errors")
    _require_sha256(
        value.cell_replays_artifact_sha256,
        "cell-replays artifact SHA-256",
    )
    expected = _tagged_digest("official-cell-evidence-v1", _cell_payload(value))
    _require_content_digest_value(value.content_digest, expected, "cell evidence")


def _digest_string_sequence(values: Iterator[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        _require_sha256(value, "sequence content digest")
        if not first:
            digest.update(b",")
        digest.update(_canonical_json_bytes(value))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _repo_relative_resolved(path: Path, label: str) -> str:
    root = _find_repo_root(path, label)
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:  # pragma: no cover - guarded by _find_repo_root
        _fail(f"{label} is not inside its resolved repository: {exc}")


def _authority_payload_parts(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    root_seed: str,
    scenario_grid: tuple[str, ...],
    draws: tuple[int, ...],
    selected_cells: tuple[OfficialSelectedCell, ...],
    cell_evidence: tuple[OfficialCellEvidence, ...],
    official_masks: tuple[OfficialMaskCell, ...],
    official_artifacts: tuple[OfficialArtifact, ...],
    historical_common_fingerprints: tuple[tuple[str, str], ...],
    g1_artifact_path: Path,
    g1_artifact_sha256: str,
    task8a_clean_authority_digest: str,
) -> dict[str, object]:
    return {
        "cell_evidence_digest": _digest_string_sequence(
            item.content_digest for item in cell_evidence
        ),
        "draws": list(draws),
        "g1_artifact_path": _repo_relative_resolved(
            g1_artifact_path,
            "G1 artifact path",
        ),
        "g1_artifact_sha256": g1_artifact_sha256,
        "g1_physical_sha256": OFFICIAL_G1_PHYSICAL_SHA256,
        "historical_common_fingerprints": [
            list(item) for item in historical_common_fingerprints
        ],
        "manifest_path": _repo_relative_resolved(
            manifest_path,
            "official manifest path",
        ),
        "manifest_sha256": manifest_sha256,
        "official_artifacts_digest": _digest_string_sequence(
            item.content_digest for item in official_artifacts
        ),
        "official_masks_digest": _digest_string_sequence(
            item.content_digest for item in official_masks
        ),
        "root_seed": root_seed,
        "scenario_grid": list(scenario_grid),
        "selected_cells_digest": _digest_string_sequence(
            item.content_digest for item in selected_cells
        ),
        "task8a_clean_authority_digest": task8a_clean_authority_digest,
    }


def _authority_payload(value: OfficialG2Authority) -> dict[str, object]:
    return _authority_payload_parts(
        manifest_path=value.manifest_path,
        manifest_sha256=value.manifest_sha256,
        root_seed=value.root_seed,
        scenario_grid=value.scenario_grid,
        draws=value.draws,
        selected_cells=value.selected_cells,
        cell_evidence=value.cell_evidence,
        official_masks=value.official_masks,
        official_artifacts=value.official_artifacts,
        historical_common_fingerprints=value.historical_common_fingerprints,
        g1_artifact_path=value.g1_artifact_path,
        g1_artifact_sha256=value.g1_artifact_sha256,
        task8a_clean_authority_digest=value.task8a_clean_authority_digest,
    )


def _revalidate_official_authority_graph(value: OfficialG2Authority) -> None:
    artifact_by_name = {artifact.name: artifact for artifact in value.official_artifacts}
    for artifact in value.official_artifacts:
        _validate_official_artifact_value(artifact)
    selected_artifact_sha256 = artifact_by_name["selected_days"].physical_sha256
    selected_days: dict[tuple[str, str], tuple[int, ...]] = {}
    selected_keys: set[tuple[str, int, str]] = set()
    selected_index = 0
    for subject in OFFICIAL_G2_PARTICIPANTS:
        for primitive in OFFICIAL_G2_PRIMITIVES:
            primitive_days: list[int] = []
            for _ in range(10):
                selected = value.selected_cells[selected_index]
                selected_index += 1
                _validate_official_selected_cell_value(selected)
                if (
                    selected.subject_id != subject
                    or selected.primitive != primitive
                    or selected.selected_days_artifact_sha256
                    != selected_artifact_sha256
                ):
                    _fail("official selected-cell topology/artifact is crosswired")
                key = (subject, selected.sensor_day_id, primitive)
                if key in selected_keys:
                    _fail("official selected-cell topology contains a duplicate key")
                selected_keys.add(key)
                primitive_days.append(selected.sensor_day_id)
            if len(set(primitive_days)) != 10:
                _fail("official selected-cell group requires ten unique days")
            selected_days[(subject, primitive)] = tuple(sorted(primitive_days))
    if selected_index != 800 or len(selected_keys) != 800:
        _fail("official selected-cell topology is incomplete or extra")

    cell_artifact_sha256 = artifact_by_name["cell_replays"].physical_sha256
    mask_artifact_sha256 = artifact_by_name["mask_intervals"].physical_sha256
    deletion_artifact_sha256 = artifact_by_name["deletion_audit"].physical_sha256
    deletion_keys: set[str] = set()
    interval_count = 0
    observed_count = 0
    unavailable_count = 0
    for expected_key, evidence, mask in zip(
        _iter_expected_topology(selected_days),
        value.cell_evidence,
        value.official_masks,
        strict=True,
    ):
        _validate_official_cell_evidence_value(evidence)
        _validate_official_mask_cell_value(mask)
        evidence_key = (
            evidence.subject_id,
            evidence.sensor_day_id,
            evidence.primitive,
            evidence.scenario,
            evidence.draw,
        )
        mask_key = (
            mask.subject_id,
            mask.sensor_day_id,
            mask.primitive,
            mask.scenario,
            mask.draw,
        )
        if evidence_key != expected_key or mask_key != expected_key:
            _fail("official evidence/mask topology is missing, extra, or reordered")
        audit = mask.deletion_audit
        if (
            evidence.root_seed != mask.root_seed
            or evidence.seed != mask.seed
            or evidence.donor_sensor_day_id != mask.donor_sensor_day_id
            or evidence.deletion_audit_key != audit.deletion_audit_key
            or evidence.deletion_audit_digest != audit.content_digest
            or evidence.record_deletion_digest != audit.record_deletion_digest
        ):
            _fail("official evidence/mask/deletion nested graph is crosswired")
        if (
            evidence.cell_replays_artifact_sha256 != cell_artifact_sha256
            or mask.mask_intervals_artifact_sha256 != mask_artifact_sha256
            or audit.deletion_audit_artifact_sha256 != deletion_artifact_sha256
        ):
            _fail("official nested graph carries a crosswired artifact SHA-256")
        if audit.deletion_audit_key in deletion_keys:
            _fail("official deletion graph contains a duplicate audit key")
        deletion_keys.add(audit.deletion_audit_key)
        interval_count += len(mask.intervals)
        observed_count += mask.status == "observed_mask"
        unavailable_count += mask.status == "empirical_unavailable"
    if (
        len(deletion_keys) != 320_000
        or interval_count != 734_907
        or observed_count != 299_350
        or unavailable_count != 20_650
    ):
        _fail("official nested graph totals are incomplete or extra")


def _validate_official_g2_authority_value(value: OfficialG2Authority) -> None:
    if type(value.manifest_path) is not _PATH_TYPE or not value.manifest_path.is_absolute():
        _fail("official manifest_path must be an exact absolute pathlib Path")
    if _repo_relative_resolved(value.manifest_path, "manifest_path") != OFFICIAL_G2_MANIFEST_REPO_PATH:
        _fail("official manifest_path is not the committed authority path")
    if value.manifest_sha256 != OFFICIAL_G2_MANIFEST_SHA256:
        _fail("official manifest SHA-256 is not the sealed value")
    if type(value.root_seed) is not str or value.root_seed != "42":
        _fail("official G2 root_seed must be the exact string '42'")
    if value.scenario_grid != OFFICIAL_G2_SCENARIOS:
        _fail("official G2 scenario grid is not exact")
    if value.draws != OFFICIAL_G2_DRAWS:
        _fail("official G2 draws are not exact")
    if type(value.selected_cells) is not tuple or len(value.selected_cells) != 800:
        _fail("official G2 requires exactly 800 selected cells")
    if any(type(item) is not OfficialSelectedCell for item in value.selected_cells):
        _fail("selected_cells contains a nonexact value")
    if type(value.cell_evidence) is not tuple or len(value.cell_evidence) != 320_000:
        _fail("official G2 requires exactly 320,000 cell evidence rows")
    if any(type(item) is not OfficialCellEvidence for item in value.cell_evidence):
        _fail("cell_evidence contains a nonexact value")
    if type(value.official_masks) is not tuple or len(value.official_masks) != 320_000:
        _fail("official G2 requires exactly 320,000 official masks")
    if any(type(item) is not OfficialMaskCell for item in value.official_masks):
        _fail("official_masks contains a nonexact value")
    if (
        type(value.official_artifacts) is not tuple
        or tuple(item.name for item in value.official_artifacts)
        != _OFFICIAL_ARTIFACT_NAMES
        or any(type(item) is not OfficialArtifact for item in value.official_artifacts)
    ):
        _fail("official G2 artifacts are not the exact ordered 19-value tuple")
    if value.historical_common_fingerprints != _HISTORICAL_COMMON_FINGERPRINTS:
        _fail("historical common fingerprints are not exact")
    if type(value.g1_artifact_path) is not _PATH_TYPE or not value.g1_artifact_path.is_absolute():
        _fail("G1 artifact path must be an exact absolute pathlib Path")
    if _repo_relative_resolved(value.g1_artifact_path, "G1 path") != OFFICIAL_G1_REPO_PATH:
        _fail("G1 artifact path is not the hard-reviewed provenance path")
    if value.g1_artifact_sha256 != OFFICIAL_G1_ARTIFACT_SHA256:
        _fail("G1 logical artifact digest is not exact")
    if type(value.task8a_clean) is not Task8ACleanAuthority:
        _fail("official G2 requires an exact Task8ACleanAuthority")
    if value.task8a_clean_authority_digest != value.task8a_clean.content_digest:
        _fail("official G2 Task 8A authority digest is crosswired")
    _revalidate_official_authority_graph(value)
    expected = _tagged_digest("official-g2-authority-v1", _authority_payload(value))
    _require_content_digest_value(value.content_digest, expected, "official G2 authority")


def _require_manifest_relative_path(value: object, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be an exact nonempty manifest-relative string")
    if "\\" in value or ":" in value or value.startswith(("/", "//")):
        _fail(f"{label} must not be absolute, drive-relative, UNC, or backslashed")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        _fail(f"{label} must be one canonical nontraversing filename")
    return value


def _require_exact_ordered_object(
    value: object,
    expected_fields: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    raw = _require_exact_dict(value, set(expected_fields), label)
    if tuple(raw) != expected_fields:
        _fail(f"{label} field order is not exact")
    return raw


def _validate_official_manifest_document(
    document: dict[str, object],
) -> tuple[OfficialArtifact, ...]:
    if _sha256_bytes(_canonical_json_bytes(document)) != _OFFICIAL_MANIFEST_LOGICAL_DIGEST:
        _fail("official G2 manifest logical graph is mutated or crosswired")
    raw = _require_exact_ordered_object(
        document,
        _OFFICIAL_MANIFEST_FIELDS,
        "official G2 manifest",
    )
    if raw["schema_version"] != "g2_streamed_official_v2":
        _fail("official G2 manifest schema_version is not exact")
    if raw["status"] != "complete_streamed_exact50":
        _fail("official G2 manifest status is not exact")
    common = raw["common_fingerprints"]
    if type(common) is not dict or tuple(common.items()) != _HISTORICAL_COMMON_FINGERPRINTS:
        _fail("official G2 historical common fingerprints are not exact")
    if raw["scenario_grid"] != list(OFFICIAL_G2_SCENARIOS):
        _fail("official G2 scenario order is not exact")
    if type(raw["draws_per_scenario"]) is not int or raw["draws_per_scenario"] != 50:
        _fail("official G2 draws_per_scenario must be the exact integer 50")
    if raw["primary_scenario"] != "contiguous_20pct":
        _fail("official G2 primary scenario is not exact")
    top_semantics = {field: raw[field] for field in _OFFICIAL_TOP_SEMANTIC_FIELDS}
    if _sha256_bytes(_canonical_json_bytes(top_semantics)) != _OFFICIAL_TOP_SEMANTICS_DIGEST:
        _fail("official G2 top-level semantics are mutated or crosswired")
    small_logical = {field: raw[field] for field in _OFFICIAL_SMALL_LOGICAL_FIELDS}
    if _sha256_bytes(_canonical_json_bytes(small_logical)) != _OFFICIAL_SMALL_LOGICAL_DIGEST:
        _fail("official G2 embedded logical summaries are mutated or crosswired")
    declarations = raw["official_artifacts"]
    if type(declarations) is not dict or tuple(declarations) != _OFFICIAL_ARTIFACT_NAMES:
        _fail("official G2 requires the exact ordered 19-artifact declaration")
    if _sha256_bytes(_canonical_json_bytes(declarations)) != _OFFICIAL_ARTIFACT_DECLARATIONS_DIGEST:
        _fail("official G2 artifact declarations are mutated or crosswired")
    artifacts: list[OfficialArtifact] = []
    for name in _OFFICIAL_ARTIFACT_NAMES:
        declaration = _require_exact_ordered_object(
            declarations[name],
            _OFFICIAL_ARTIFACT_RECORD_FIELDS,
            f"official artifact declaration {name}",
        )
        artifact_format = "jsonl" if name == "decision_table" else "parquet"
        expected_path = f"{name}.{artifact_format}"
        path_text = _require_manifest_relative_path(
            declaration["path"],
            f"official artifact {name} path",
        )
        if path_text != expected_path:
            _fail(f"official artifact {name} path/role is crosswired")
        if type(declaration["format"]) is not str or declaration["format"] != artifact_format:
            _fail(f"official artifact {name} format is crosswired")
        rows = _require_exact_integer(
            declaration["rows"],
            f"official artifact {name} rows",
        )
        columns_raw = declaration["columns"]
        if type(columns_raw) is not list or not columns_raw or any(
            type(column) is not str or not column for column in columns_raw
        ):
            _fail(f"official artifact {name} columns are invalid")
        columns = tuple(columns_raw)
        byte_count = _require_exact_integer(
            declaration["bytes"],
            f"official artifact {name} bytes",
            minimum=1,
        )
        physical_sha256 = _require_sha256(
            declaration["sha256"],
            f"official artifact {name} SHA-256",
        )
        artifacts.append(
            _make_official_artifact(
                name=name,
                manifest_relative_path=path_text,
                format=artifact_format,
                rows=rows,
                columns=columns,
                byte_count=byte_count,
                physical_sha256=physical_sha256,
            )
        )
    return tuple(artifacts)


def _hash_open_stream(stream: BinaryIO) -> tuple[str, int]:
    stream.seek(0)
    digest = hashlib.sha256()
    byte_count = 0
    while block := stream.read(1024 * 1024):
        digest.update(block)
        byte_count += len(block)
    stream.seek(0)
    return digest.hexdigest(), byte_count


def _parquet_schema_digest(path: Path) -> str:
    try:
        parquet = pq.ParquetFile(path)
    except Exception as exc:
        _fail(f"cannot open Parquet schema for {path}: {exc}")
    return _sha256_bytes(parquet.schema_arrow.serialize().to_pybytes())


def _validate_open_parquet_artifact(
    stream: BinaryIO,
    artifact: OfficialArtifact,
    *,
    expected_schema_digest: str,
) -> pq.ParquetFile:
    actual_sha256, byte_count = _hash_open_stream(stream)
    if byte_count != artifact.byte_count:
        _fail(f"official artifact {artifact.name} byte count is stale")
    if actual_sha256 != artifact.physical_sha256:
        _fail(f"official artifact {artifact.name} physical SHA-256 mismatch")
    expected_schema = _require_sha256(
        expected_schema_digest,
        f"official artifact {artifact.name} expected schema digest",
    )
    try:
        parquet = pq.ParquetFile(stream)
    except Exception as exc:
        _fail(f"official artifact {artifact.name} is not readable Parquet: {exc}")
    metadata = parquet.metadata
    if (
        metadata.num_rows != artifact.rows
        or metadata.num_row_groups != 1
        or metadata.format_version != "2.6"
        or metadata.created_by != "parquet-cpp-arrow version 24.0.0"
    ):
        _fail(f"official artifact {artifact.name} Parquet metadata is not exact")
    if tuple(parquet.schema_arrow.names) != artifact.columns:
        _fail(f"official artifact {artifact.name} Parquet columns/order mismatch")
    actual_schema = _sha256_bytes(parquet.schema_arrow.serialize().to_pybytes())
    if actual_schema != expected_schema:
        _fail(f"official artifact {artifact.name} Arrow/pandas schema drift")
    compressions = {
        metadata.row_group(group_index).column(column_index).compression
        for group_index in range(metadata.num_row_groups)
        for column_index in range(metadata.num_columns)
    }
    if compressions != {"SNAPPY"}:
        _fail(f"official artifact {artifact.name} compression is not exact")
    return parquet


def _validate_parquet_artifact(
    path: Path,
    artifact: OfficialArtifact,
    *,
    expected_schema_digest: str,
) -> None:
    if type(path) is not _PATH_TYPE:
        raise TypeError("Parquet artifact path must be an exact pathlib Path")
    try:
        with path.open("rb") as stream:
            _validate_open_parquet_artifact(
                stream,
                artifact,
                expected_schema_digest=expected_schema_digest,
            )
    except AuthorityValidationError:
        raise
    except OSError as exc:
        _fail(f"cannot read Parquet artifact {path}: {exc}")


def _decode_strict_json_bytes(raw: bytes, label: str) -> object:
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail(f"{label} must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"{label} is not UTF-8: {exc}")
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite_json,
        )
    except AuthorityValidationError:
        raise
    except json.JSONDecodeError as exc:
        _fail(
            f"{label} is not JSON: {exc.msg} at line {exc.lineno} "
            f"column {exc.colno}"
        )


def _read_strict_json_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, object], str]:
    expected = _require_sha256(expected_sha256, f"expected {label} SHA-256")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read {label}: {exc}")
    actual = _sha256_bytes(raw)
    if actual != expected:
        _fail(f"{label} physical SHA-256 mismatch")
    document = _decode_strict_json_bytes(raw, label)
    if type(document) is not dict:
        _fail(f"{label} must contain one exact JSON object")
    return document, actual


def _validate_open_decision_jsonl(
    stream: BinaryIO,
    artifact: OfficialArtifact,
) -> tuple[dict[str, object], ...]:
    actual_sha256, byte_count = _hash_open_stream(stream)
    if byte_count != artifact.byte_count or actual_sha256 != artifact.physical_sha256:
        _fail("official decision_table physical bytes are stale")
    raw = stream.read()
    stream.seek(0)
    if not raw.endswith(b"\n") or raw.startswith(b"\xef\xbb\xbf"):
        _fail("official decision_table JSONL bytes are not exact UTF-8 lines")
    lines = raw.splitlines()
    if len(lines) != artifact.rows or any(not line for line in lines):
        _fail("official decision_table requires exactly 43 nonblank lines")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        row = _decode_strict_json_bytes(line, f"decision_table line {index}")
        raw_row = _require_exact_ordered_object(
            row,
            _DECISION_COLUMNS,
            f"decision_table line {index}",
        )
        for name in ("level", "gate", "comparator", "status", "root_seed"):
            _require_nonempty_string(raw_row[name], f"decision_table.{name}")
        for name in ("primitive", "pair", "domain"):
            if raw_row[name] is not None and (
                type(raw_row[name]) is not str or not raw_row[name]
            ):
                _fail(f"decision_table.{name} must be null or an exact string")
        for name in ("observed", "threshold"):
            if type(raw_row[name]) not in {bool, int, float}:
                _fail(f"decision_table.{name} has an invalid exact scalar type")
            if type(raw_row[name]) is float and not math.isfinite(raw_row[name]):
                _fail(f"decision_table.{name} must be finite")
        for name in ("passed", "can_change_domain_decision", "production_eligible"):
            _require_exact_boolean(raw_row[name], f"decision_table.{name}")
        if raw_row["root_seed"] != "42":
            _fail("decision_table root_seed is crosswired")
        if type(raw_row["draws"]) is not int or raw_row["draws"] != 50:
            _fail("decision_table draws must be the exact integer 50")
        rows.append(raw_row)
    return tuple(rows)


_OFFICIAL_G1_CONFIG = {
    "evaluation_start_day": 22,
    "bootstrap_draws": 10000,
    "shift_draws": 1000,
    "root_seed": 20260807,
    "primary_high_coverage_threshold": 0.8,
    "raw_r_minimum": 0.3,
    "holm_alpha": 0.05,
    "minimum_positive_participants": 7,
    "participant_denominator": 10,
    "usage_availability_policy": "matching_screen_proxy",
}


def _validate_official_g1(path: Path) -> str:
    document, physical_sha256 = _read_strict_json_file(
        path,
        OFFICIAL_G1_PHYSICAL_SHA256,
        "hard-reviewed G1 artifact",
    )
    if physical_sha256 != OFFICIAL_G1_PHYSICAL_SHA256:
        _fail("hard-reviewed G1 physical provenance is stale")
    raw = _require_exact_ordered_object(
        document,
        ("schema_version", "config", "source", "pair_summary", "artifact_sha256"),
        "hard-reviewed G1 artifact",
    )
    if raw["schema_version"] != "reviewed_g1_pair_summary_v1":
        _fail("hard-reviewed G1 schema is not exact")
    if raw["config"] != _OFFICIAL_G1_CONFIG:
        _fail("hard-reviewed G1 config is not exact")
    config_digest = _sha256_bytes(
        json.dumps(
            raw["config"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    if config_digest != OFFICIAL_G1_CONFIG_SHA256:
        _fail("hard-reviewed G1 config digest is stale")
    if raw["source"] != {
        "review_status": "CLEAN",
        "implementation_commits": ["18eadac", "f246931"],
        "report": "task-4a-report.md",
    }:
        _fail("hard-reviewed G1 source provenance is not exact")
    pairs = raw["pair_summary"]
    if type(pairs) is not list or len(pairs) != 4:
        _fail("hard-reviewed G1 must contain exactly four pair rows")
    expected_pairs = (
        ("digital_load", "digital", "screen_load_24h", "usage_load_24h", True),
        (
            "digital_disengagement",
            "digital",
            "screen_disengagement_p90",
            "usage_disengagement_p90",
            True,
        ),
        (
            "physical_load",
            "physical",
            "phone_activity_load_24h",
            "step_load_24h",
            False,
        ),
        (
            "light_exposure",
            "light",
            "mobile_light_exposure_24h",
            "wearable_light_exposure_24h",
            True,
        ),
    )
    for row, expected in zip(pairs, expected_pairs, strict=True):
        if type(row) is not dict:
            _fail("hard-reviewed G1 pair row must be an exact object")
        actual = (
            row.get("pair"),
            row.get("domain"),
            row.get("left"),
            row.get("right"),
            row.get("strong_pair"),
        )
        if actual != expected or row.get("status") != "ok":
            _fail("hard-reviewed G1 pair identity or gate is crosswired")
    payload = {key: value for key, value in raw.items() if key != "artifact_sha256"}
    logical_sha256 = _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    embedded = _require_sha256(raw["artifact_sha256"], "G1 artifact_sha256")
    if logical_sha256 != OFFICIAL_G1_ARTIFACT_SHA256 or embedded != logical_sha256:
        _fail("hard-reviewed G1 logical artifact digest is stale")
    return logical_sha256


class _PayloadView:
    """Attribute view used only to compute a constructor's validated digest."""

    __slots__ = ("_values",)

    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def __getattr__(self, name: str) -> object:
        try:
            return self._values[name]
        except KeyError as exc:  # pragma: no cover - internal programming error
            raise AttributeError(name) from exc


def _require_physical_row(
    value: object,
    columns: tuple[str, ...],
    label: str,
) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != columns:
        _fail(f"{label} has missing, extra, duplicate, or reordered fields")
    return value


def _require_physical_string_fields(
    row: dict[str, object],
    fields: tuple[str, ...],
    label: str,
) -> None:
    for field in fields:
        _require_nonempty_string(row[field], f"{label}.{field}")


def _require_physical_float(
    value: object,
    label: str,
    *,
    allow_nan: bool = False,
) -> float:
    _float_token(value, label, allow_nan=allow_nan)
    return value


def _require_unit_float(value: object, label: str) -> float:
    number = _require_physical_float(value, label)
    if not 0.0 <= number <= 1.0:
        _fail(f"{label} must lie in [0,1]")
    return number


def _normalize_nullable_physical_float(value: object, label: str) -> float:
    if value is None:
        return float("nan")
    return _require_physical_float(value, label)


def _normalize_physical_double_donor(value: object, label: str) -> int | None:
    if value is None:
        return None
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        or not value.is_integer()
    ):
        _fail(f"{label} must be null or an exact integral Arrow double")
    return int(value)


def _normalize_physical_int_donor(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _require_exact_integer(value, label)


def _iter_parquet_rows(
    parquet: pq.ParquetFile,
    *,
    columns: tuple[str, ...],
    label: str,
    batch_size: int = 4096,
) -> Iterator[dict[str, object]]:
    row_index = 0
    try:
        batches = parquet.iter_batches(
            batch_size=batch_size,
            columns=list(columns),
            use_threads=False,
        )
        for batch in batches:
            for row in batch.to_pylist():
                yield _require_physical_row(row, columns, f"{label} row {row_index}")
                row_index += 1
    except AuthorityValidationError:
        raise
    except Exception as exc:
        _fail(f"cannot stream {label} rows: {exc}")


_SELECTED_PROXY_PRIMITIVES = {
    "screen_load_24h": "screen_load_24h",
    "usage_load_24h": "screen_load_24h",
    "screen_disengagement_p90": "screen_disengagement_p90",
    "usage_disengagement_p90": "screen_disengagement_p90",
    "phone_activity_load_24h": "phone_activity_load_24h",
    "step_load_24h": "step_load_24h",
    "mobile_light_exposure_24h": "mobile_light_exposure_24h",
    "wearable_light_exposure_24h": "wearable_light_exposure_24h",
}


def _selected_row_digest(row: dict[str, object]) -> str:
    return _tagged_digest(
        "official-selected-row-v1",
        {
            column: _canonical_value_token(row[column], f"selected_days.{column}")
            for column in _SELECTED_COLUMNS
        },
    )


def _load_selected_cells(
    parquet: pq.ParquetFile,
    artifact: OfficialArtifact,
) -> tuple[
    tuple[OfficialSelectedCell, ...],
    dict[tuple[str, str], tuple[int, ...]],
]:
    rows = _iter_parquet_rows(
        parquet,
        columns=_SELECTED_COLUMNS,
        label="selected_days",
    )
    selected: list[OfficialSelectedCell] = []
    physical_days: dict[tuple[str, str], list[int]] = {
        (subject, primitive): []
        for subject in OFFICIAL_G2_PARTICIPANTS
        for primitive in OFFICIAL_G2_PRIMITIVES
    }
    seen: set[tuple[str, int, str]] = set()
    for subject in OFFICIAL_G2_PARTICIPANTS:
        for primitive in OFFICIAL_G2_PRIMITIVES:
            for expected_rank in range(1, 11):
                try:
                    row = next(rows)
                except StopIteration:
                    _fail("selected_days is missing required topology rows")
                _require_physical_string_fields(
                    row,
                    (
                        "subject_id",
                        "primitive",
                        "proxy_primitive",
                        "selection_status",
                    ),
                    "selected_days",
                )
                if row["subject_id"] != subject or row["primitive"] != primitive:
                    _fail("selected_days physical order must be subject/primitive/rank")
                sensor_day_id = _require_exact_integer(
                    row["sensor_day_id"],
                    "selected_days.sensor_day_id",
                )
                if row["proxy_primitive"] != _SELECTED_PROXY_PRIMITIVES[primitive]:
                    _fail("selected_days proxy primitive is crosswired")
                for column in (
                    "availability_proxy",
                    "gap_component",
                    "quality_confidence",
                ):
                    _require_unit_float(row[column], f"selected_days.{column}")
                if (
                    type(row["rank"]) is not int
                    or row["rank"] != expected_rank
                    or type(row["selected_count"]) is not int
                    or row["selected_count"] != 10
                    or row["selection_status"] != "selected_maximum"
                    or type(row["participant_denominator"]) is not int
                    or row["participant_denominator"] != 10
                ):
                    _fail("selected_days rank/selection semantics are not exact")
                key = (subject, sensor_day_id, primitive)
                if key in seen:
                    _fail("selected_days contains a duplicate selected key")
                seen.add(key)
                physical_days[(subject, primitive)].append(sensor_day_id)
                selected_digest = _selected_row_digest(row)
                values: dict[str, object] = {
                    "subject_id": subject,
                    "sensor_day_id": sensor_day_id,
                    "primitive": primitive,
                    "selected_row_digest": selected_digest,
                    "selected_days_artifact_sha256": artifact.physical_sha256,
                }
                values["content_digest"] = _tagged_digest(
                    "official-selected-cell-v1",
                    _selected_cell_payload(_PayloadView(values)),
                )
                selected.append(OfficialSelectedCell(**values))
    try:
        next(rows)
    except StopIteration:
        pass
    else:
        _fail("selected_days contains extra topology rows")
    if len(selected) != artifact.rows or len(seen) != 800:
        _fail("selected_days does not contain exactly 800 unique selected keys")
    sorted_days: dict[tuple[str, str], tuple[int, ...]] = {}
    for key, days in physical_days.items():
        if len(days) != 10 or len(set(days)) != 10:
            _fail("selected_days requires ten unique days per participant/primitive")
        sorted_days[key] = tuple(sorted(days))
    return tuple(selected), sorted_days


def _require_expected_cell_key(
    expected_key: object,
) -> tuple[str, int, str, str, int]:
    if type(expected_key) is not tuple or len(expected_key) != 5:
        _fail("expected official cell key must be an exact five-value tuple")
    subject, sensor_day_id, primitive, scenario, draw = expected_key
    _require_nonempty_string(subject, "expected subject_id")
    _require_nonempty_string(primitive, "expected primitive")
    _require_nonempty_string(scenario, "expected scenario")
    if subject not in OFFICIAL_G2_PARTICIPANTS:
        _fail("expected official cell subject is invalid")
    _require_exact_integer(sensor_day_id, "expected sensor_day_id")
    if primitive not in OFFICIAL_G2_PRIMITIVES:
        _fail("expected official cell primitive is invalid")
    if scenario not in OFFICIAL_G2_SCENARIOS:
        _fail("expected official cell scenario is invalid")
    if type(draw) is not int or draw not in OFFICIAL_G2_DRAWS:
        _fail("expected official cell draw is invalid")
    return expected_key


def _expected_mask_window(
    lifelog_date: pd.Timestamp,
    primitive: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if primitive in {"screen_disengagement_p90", "usage_disengagement_p90"}:
        return lifelog_date + pd.Timedelta(hours=18), lifelog_date + pd.Timedelta(days=1)
    return lifelog_date, lifelog_date + pd.Timedelta(days=1)


def _build_official_pair(
    cell_row: dict[str, object],
    audit_row: dict[str, object],
    interval_rows: list[dict[str, object]],
    *,
    expected_key: tuple[str, int, str, str, int],
    cell_replays_artifact_sha256: str,
    mask_intervals_artifact_sha256: str,
    deletion_audit_artifact_sha256: str,
) -> tuple[OfficialCellEvidence, OfficialMaskCell]:
    expected = _require_expected_cell_key(expected_key)
    cell_sha = _require_sha256(
        cell_replays_artifact_sha256,
        "cell_replays artifact SHA-256",
    )
    mask_sha = _require_sha256(
        mask_intervals_artifact_sha256,
        "mask_intervals artifact SHA-256",
    )
    deletion_sha = _require_sha256(
        deletion_audit_artifact_sha256,
        "deletion_audit artifact SHA-256",
    )
    cell = _require_physical_row(cell_row, _CELL_COLUMNS, "cell_replays row")
    audit = _require_physical_row(audit_row, _DELETION_COLUMNS, "deletion_audit row")
    _require_physical_string_fields(
        cell,
        (
            "subject_id",
            "primitive",
            "scenario",
            "root_seed",
            "seed",
            "deletion_audit_key",
            "original_status",
            "masked_status",
            "standardizer_source",
            "reason",
            "status",
        ),
        "cell_replays",
    )
    _require_physical_string_fields(
        audit,
        (
            "deletion_audit_key",
            "subject_id",
            "primitive",
            "scenario",
            "seed",
            "record_deletion_digest",
            "audit_representation",
            "status",
        ),
        "deletion_audit",
    )
    if type(interval_rows) is not list:
        _fail("mask interval group must be an exact list")

    subject, sensor_day_id, primitive, scenario, draw = expected
    cell_identity = (
        cell["subject_id"],
        cell["sensor_day_id"],
        cell["primitive"],
        cell["scenario"],
        cell["draw"],
    )
    audit_identity = (
        audit["subject_id"],
        audit["sensor_day_id"],
        audit["primitive"],
        audit["scenario"],
        audit["draw"],
    )
    if cell_identity != expected or audit_identity != expected:
        _fail("official cell/deletion rows are reordered, missing, or crosswired")
    _require_exact_integer(cell["sensor_day_id"], "cell_replays.sensor_day_id")
    _require_exact_integer(cell["draw"], "cell_replays.draw")
    _require_exact_integer(audit["sensor_day_id"], "deletion_audit.sensor_day_id")
    _require_exact_integer(audit["draw"], "deletion_audit.draw")
    if cell["root_seed"] != "42":
        _fail("cell_replays root_seed must be the exact string '42'")
    expected_seed = _historical_seed("42", scenario, draw, subject, sensor_day_id, primitive)
    seed = _require_sha256(cell["seed"], "cell_replays.seed")
    audit_seed = _require_sha256(audit["seed"], "deletion_audit.seed")
    if seed != expected_seed or audit_seed != seed:
        _fail("official cell/deletion seed is stale or crosswired")
    cell_donor = _normalize_physical_double_donor(
        cell["donor_sensor_day_id"],
        "cell_replays.donor_sensor_day_id",
    )
    audit_donor = _normalize_physical_double_donor(
        audit["donor_sensor_day_id"],
        "deletion_audit.donor_sensor_day_id",
    )
    if cell_donor != audit_donor:
        _fail("official cell/deletion donors are crosswired")
    record_digest = _require_sha256(
        audit["record_deletion_digest"],
        "deletion_audit.record_deletion_digest",
    )
    deletion_key = _require_sha256(
        audit["deletion_audit_key"],
        "deletion_audit.deletion_audit_key",
    )
    if (
        deletion_key != _historical_deletion_key(seed, record_digest)
        or cell["deletion_audit_key"] != deletion_key
    ):
        _fail("official deletion key does not bind the joined seed/record digest")

    original_count = _require_exact_integer(
        audit["original_count"], "deletion_audit.original_count"
    )
    deleted_count = _require_exact_integer(
        audit["deleted_count"], "deletion_audit.deleted_count"
    )
    retained_count = _require_exact_integer(
        audit["retained_count"], "deletion_audit.retained_count"
    )
    if original_count != deleted_count + retained_count:
        _fail("deletion audit arithmetic is invalid")
    if audit["audit_representation"] != "normalized_source_digest_plus_mask_geometry":
        _fail("deletion audit representation is not exact")
    if type(audit["status"]) is not str or not audit["status"]:
        _fail("deletion audit status must be an exact nonempty string")
    deletion_values: dict[str, object] = {
        "deletion_audit_key": deletion_key,
        "subject_id": subject,
        "sensor_day_id": sensor_day_id,
        "primitive": primitive,
        "scenario": scenario,
        "draw": draw,
        "seed": seed,
        "donor_sensor_day_id": audit_donor,
        "original_count": original_count,
        "deleted_count": deleted_count,
        "retained_count": retained_count,
        "record_deletion_digest": record_digest,
        "audit_representation": audit["audit_representation"],
        "status": audit["status"],
        "deletion_audit_artifact_sha256": deletion_sha,
    }
    deletion_values["content_digest"] = _tagged_digest(
        "official-deletion-audit-v1",
        _deletion_payload(_PayloadView(deletion_values)),
    )
    deletion = OfficialDeletionAudit(**deletion_values)

    lifelog_date = cell["lifelog_date"]
    _timestamp_token(lifelog_date, "cell_replays.lifelog_date")
    if lifelog_date != lifelog_date.normalize():
        _fail("cell_replays lifelog_date must be local midnight")
    elapsed_day_index = _require_exact_integer(
        cell["elapsed_day_index"],
        "cell_replays.elapsed_day_index",
        minimum=1,
    )
    original_value = _require_physical_float(
        cell["original_value"],
        "cell_replays.original_value",
    )
    masked_value = _normalize_nullable_physical_float(
        cell["masked_value"],
        "cell_replays.masked_value",
    )
    absolute_error = _normalize_nullable_physical_float(
        cell["absolute_error"],
        "cell_replays.absolute_error",
    )
    standardized_absolute_error = _normalize_nullable_physical_float(
        cell["standardized_absolute_error"],
        "cell_replays.standardized_absolute_error",
    )
    null_error_triple = (
        cell["masked_value"] is None,
        cell["absolute_error"] is None,
        cell["standardized_absolute_error"] is None,
    )
    if len(set(null_error_triple)) != 1:
        _fail("official cell nullable value/error triple is only partially missing")
    for column in (
        "original_value_only_confidence",
        "original_coverage_only_confidence",
        "original_quality_confidence",
        "masked_value_only_confidence",
        "masked_coverage_only_confidence",
        "masked_quality_confidence",
    ):
        _require_unit_float(cell[column], f"cell_replays.{column}")
    for column in (
        "standardizer",
        "selected_day_mad",
        "frozen_population_scale",
        "scale_floor",
    ):
        _require_physical_float(cell[column], f"cell_replays.{column}")
    for column in (
        "retained",
        "floor_applied",
        "success",
        "usage_record_deletion_stability",
    ):
        _require_exact_boolean(cell[column], f"cell_replays.{column}")
    for column in (
        "original_status",
        "masked_status",
        "standardizer_source",
        "reason",
        "status",
    ):
        _require_nonempty_string(cell[column], f"cell_replays.{column}")
    if cell["original_status"] != "observed":
        _fail("official cell original_status must be observed")

    unavailable = (
        scenario == "empirical_gap"
        and cell["status"] == "abstained"
        and cell["reason"] == "empirical_gap_unavailable"
    )
    if unavailable:
        if (
            cell["masked_status"] != "insufficient"
            or cell["retained"] is not False
            or cell_donor is not None
            or not math.isnan(masked_value)
            or not math.isnan(absolute_error)
            or not math.isnan(standardized_absolute_error)
            or cell["masked_value_only_confidence"] != 0.0
            or cell["masked_coverage_only_confidence"] != 0.0
            or cell["masked_quality_confidence"] != 0.0
            or cell["success"] is not False
            or interval_rows
            or audit["status"] != "empirical_gap_unavailable"
            or (original_count, deleted_count, retained_count) != (0, 0, 0)
            or record_digest != OFFICIAL_EMPIRICAL_UNAVAILABLE_RECORD_DIGEST
        ):
            _fail("empirical unavailable cell/audit sentinel semantics are invalid")
    else:
        if (
            audit["status"] != "ok"
            or not interval_rows
        ):
            _fail("observed official cell/audit semantics are invalid")
        if cell["retained"] is True:
            if (
                cell["status"] != "observed"
                or cell["reason"] != "retained"
                or cell["masked_status"] != "observed"
                or any(null_error_triple)
            ):
                _fail("retained official cell semantics are invalid")
        elif (
            cell["retained"] is not False
            or cell["status"] != "abstained"
            or cell["reason"] not in {"masked_insufficient", "masked_observed"}
            or cell["masked_status"]
            != (
                "insufficient"
                if cell["reason"] == "masked_insufficient"
                else "observed"
            )
            or not all(null_error_triple)
        ):
            _fail("abstained observed-mask cell semantics are invalid")
        if scenario == "empirical_gap":
            if cell_donor is None:
                _fail("observed empirical masks require an exact donor")
        elif cell_donor is not None:
            _fail("non-empirical official masks cannot carry a donor")

    expected_window_start, expected_window_end = _expected_mask_window(
        lifelog_date,
        primitive,
    )
    intervals: list[MaskIntervalValue] = []
    previous_end: pd.Timestamp | None = None
    for interval_index, interval_row in enumerate(interval_rows):
        row = _require_physical_row(
            interval_row,
            _MASK_COLUMNS,
            f"mask_intervals row {interval_index}",
        )
        _require_physical_string_fields(
            row,
            (
                "subject_id",
                "primitive",
                "scenario",
                "root_seed",
                "seed",
                "mask_semantics",
                "empirical_policy",
                "status",
                "deletion_audit_key",
            ),
            "mask_intervals",
        )
        interval_identity = (
            row["subject_id"],
            row["sensor_day_id"],
            row["primitive"],
            row["scenario"],
            row["draw"],
        )
        if interval_identity != expected:
            _fail("mask interval is orphaned or crosswired")
        if row["root_seed"] != "42" or row["seed"] != seed:
            _fail("mask interval root/derived seed is crosswired")
        if row["deletion_audit_key"] != deletion_key:
            _fail("mask interval deletion audit key is crosswired")
        interval_donor = _normalize_physical_int_donor(
            row["donor_sensor_day_id"],
            "mask_intervals.donor_sensor_day_id",
        )
        if interval_donor != cell_donor:
            _fail("mask interval donor is crosswired")
        window_start = row["window_start"]
        window_end = row["window_end"]
        _timestamp_token(window_start, "mask_intervals.window_start")
        _timestamp_token(window_end, "mask_intervals.window_end")
        if window_start != expected_window_start or window_end != expected_window_end:
            _fail("mask interval measurement window is crosswired")
        if type(row["mask_interval_id"]) is not int or row["mask_interval_id"] != interval_index:
            _fail("mask interval ids/order must be exactly zero-based and contiguous")
        mask_start = row["mask_start"]
        mask_end = row["mask_end"]
        _timestamp_token(mask_start, "mask_intervals.mask_start")
        _timestamp_token(mask_end, "mask_intervals.mask_end")
        if (
            mask_start < window_start
            or mask_end > window_end
            or mask_start >= mask_end
            or (previous_end is not None and mask_start < previous_end)
        ):
            _fail("mask interval endpoints/order are invalid")
        previous_end = mask_end
        duration = _require_physical_float(
            row["mask_duration_minutes"],
            "mask_intervals.mask_duration_minutes",
        )
        if row["mask_semantics"] != "half_open":
            _fail("mask interval semantics must be half_open")
        if scenario == "empirical_gap":
            if row["empirical_policy"] != "reject" or row["status"] != "empirical_replay":
                _fail("empirical mask interval policy/status is invalid")
        elif (
            row["empirical_policy"] != "not_applicable"
            or row["status"] != "generated"
        ):
            _fail("generated mask interval policy/status is invalid")
        interval_values: dict[str, object] = {
            "mask_interval_id": interval_index,
            "mask_start": mask_start,
            "mask_end": mask_end,
            "mask_duration_minutes": duration,
            "mask_semantics": row["mask_semantics"],
            "row_status": row["status"],
        }
        interval_values["content_digest"] = _tagged_digest(
            "official-mask-interval-v1",
            _interval_payload(_PayloadView(interval_values)),
        )
        intervals.append(MaskIntervalValue(**interval_values))
    if not unavailable and scenario != "empirical_gap" and len(intervals) != 1:
        _fail("every generated non-empirical mask requires exactly one interval")

    if unavailable:
        interval_digest = OFFICIAL_EMPIRICAL_UNAVAILABLE_INTERVAL_DIGEST
        mask_status = "empirical_unavailable"
    else:
        interval_digest = _sha256_bytes(
            _canonical_json_bytes([item.content_digest for item in intervals])
        )
        mask_status = "observed_mask"
    mask_values: dict[str, object] = {
        "subject_id": subject,
        "sensor_day_id": sensor_day_id,
        "primitive": primitive,
        "scenario": scenario,
        "draw": draw,
        "root_seed": "42",
        "seed": seed,
        "status": mask_status,
        "donor_sensor_day_id": cell_donor,
        "intervals": tuple(intervals),
        "interval_digest": interval_digest,
        "deletion_audit": deletion,
        "mask_intervals_artifact_sha256": mask_sha,
        "unavailable_sentinel_digest": None,
    }
    mask_view = _PayloadView(mask_values)
    if unavailable:
        mask_values["unavailable_sentinel_digest"] = _tagged_digest(
            "official-empirical-unavailable-v1",
            _unavailable_sentinel_payload(mask_view),
        )
    mask_values["content_digest"] = _tagged_digest(
        "official-mask-cell-v1",
        _mask_payload(mask_view),
    )
    mask = OfficialMaskCell(**mask_values)

    evidence_values: dict[str, object] = {
        "subject_id": subject,
        "sensor_day_id": sensor_day_id,
        "lifelog_date": lifelog_date,
        "elapsed_day_index": elapsed_day_index,
        "primitive": primitive,
        "scenario": scenario,
        "draw": draw,
        "root_seed": "42",
        "seed": seed,
        "donor_sensor_day_id": cell_donor,
        "deletion_audit_key": deletion_key,
        "deletion_audit_digest": deletion.content_digest,
        "record_deletion_digest": record_digest,
        "original_value": original_value,
        "masked_value": masked_value,
        "original_status": cell["original_status"],
        "masked_status": cell["masked_status"],
        "retained": cell["retained"],
        "absolute_error": absolute_error,
        "standardized_absolute_error": standardized_absolute_error,
        "standardizer": cell["standardizer"],
        "standardizer_source": cell["standardizer_source"],
        "selected_day_mad": cell["selected_day_mad"],
        "frozen_population_scale": cell["frozen_population_scale"],
        "scale_floor": cell["scale_floor"],
        "floor_applied": cell["floor_applied"],
        "masked_coverage_only_confidence": cell["masked_coverage_only_confidence"],
        "masked_quality_confidence": cell["masked_quality_confidence"],
        "reason": cell["reason"],
        "status": cell["status"],
        "cell_replays_artifact_sha256": cell_sha,
    }
    evidence_values["content_digest"] = _tagged_digest(
        "official-cell-evidence-v1",
        _cell_payload(_PayloadView(evidence_values)),
    )
    evidence = OfficialCellEvidence(**evidence_values)
    if (
        evidence.deletion_audit_key != mask.deletion_audit.deletion_audit_key
        or evidence.deletion_audit_digest != mask.deletion_audit.content_digest
        or evidence.record_deletion_digest
        != mask.deletion_audit.record_deletion_digest
    ):
        _fail("official cell evidence and mask/audit graph are crosswired")
    return evidence, mask


def _require_committed_runtime_file(
    value: object,
    *,
    expected_repo_relative_path: str,
    label: str,
) -> tuple[Path, Path]:
    if type(value) is not _PATH_TYPE:
        raise TypeError(f"{label} must be an exact pathlib Path")
    if not value.is_absolute() or any(part in {".", ".."} for part in value.parts):
        _fail(f"{label} must be one canonical absolute path")
    lexical = Path(os.path.abspath(value))
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve {label}: {exc}")
    if (
        lexical != resolved
        or str(lexical) != str(resolved)
        or not resolved.is_file()
    ):
        _fail(f"{label} must not use a symlink, junction, alias, or non-file target")
    repo_root = _find_repo_root(resolved, label)
    relative = _repo_relative_resolved(resolved, label)
    if relative != expected_repo_relative_path:
        _fail(f"{label} is not the exact committed authority path")
    canonical = repo_root.joinpath(*PurePosixPath(expected_repo_relative_path).parts)
    try:
        canonical_resolved = canonical.resolve(strict=True)
    except OSError as exc:  # pragma: no cover - already guarded by resolved path
        _fail(f"cannot resolve committed {label}: {exc}")
    if (
        canonical != canonical_resolved
        or canonical_resolved != resolved
        or str(canonical) != str(resolved)
    ):
        _fail(f"{label} committed path is aliased")
    return resolved, repo_root


def _resolve_official_graph_paths(
    *,
    repo_root: Path,
    manifest_path: Path,
    g1_artifact_path: Path,
    artifacts: tuple[OfficialArtifact, ...],
) -> dict[str, Path]:
    artifact_root = manifest_path.parent
    try:
        directory_entry_names = {entry.name for entry in os.scandir(artifact_root)}
    except OSError as exc:
        _fail(f"cannot enumerate the official artifact directory exactly: {exc}")
    resolved_paths: dict[str, Path] = {}
    for artifact in artifacts:
        expected_filename = f"{artifact.name}.{artifact.format}"
        if artifact.manifest_relative_path != expected_filename:
            _fail(f"official artifact {artifact.name} filename/role spelling is not exact")
        if expected_filename not in directory_entry_names:
            case_aliases = sorted(
                name
                for name in directory_entry_names
                if name.casefold() == expected_filename.casefold()
            )
            if case_aliases:
                _fail(f"official artifact {artifact.name} uses a case-only file alias")
            _fail(f"official artifact {artifact.name} exact directory entry is absent")
        candidate = artifact_root / artifact.manifest_relative_path
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(artifact_root)
        except (OSError, ValueError) as exc:
            _fail(f"official artifact {artifact.name} escapes or is absent: {exc}")
        if candidate != resolved or not resolved.is_file():
            _fail(
                f"official artifact {artifact.name} uses a symlink, junction, or alias"
            )
        if _find_repo_root(resolved, f"official artifact {artifact.name}") != repo_root:
            _fail(f"official artifact {artifact.name} crosses Git worktrees")
        resolved_paths[artifact.name] = resolved
    all_paths = [manifest_path, g1_artifact_path, *resolved_paths.values()]
    textual = [os.path.normcase(os.path.abspath(path)) for path in all_paths]
    if len(set(textual)) != len(textual):
        _fail("official manifest, G1, and 19 artifact paths must be textually unique")
    for left_index, left in enumerate(all_paths):
        for right in all_paths[left_index + 1 :]:
            try:
                aliases = os.path.samefile(left, right)
            except OSError as exc:
                _fail(f"cannot prove official graph path identity: {exc}")
            if aliases:
                _fail("official graph contains a hardlink/symlink same-file alias")
    return resolved_paths


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return tuple(left) == tuple(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _topology_index(
    key: tuple[str, int, str, str, int],
    day_positions: dict[tuple[str, str], dict[int, int]],
) -> int:
    subject, sensor_day_id, primitive, scenario, draw = _require_expected_cell_key(key)
    try:
        scenario_index = OFFICIAL_G2_SCENARIOS.index(scenario)
        primitive_index = OFFICIAL_G2_PRIMITIVES.index(primitive)
        subject_index = OFFICIAL_G2_PARTICIPANTS.index(subject)
        day_index = day_positions[(subject, primitive)][sensor_day_id]
    except (KeyError, ValueError) as exc:
        _fail(f"official topology references a non-selected identity: {exc}")
    return (
        ((((scenario_index * 8) + primitive_index) * 10 + subject_index) * 50 + draw)
        * 10
        + day_index
    )


def _iter_expected_topology(
    selected_days: dict[tuple[str, str], tuple[int, ...]],
) -> Iterator[tuple[str, int, str, str, int]]:
    for scenario in OFFICIAL_G2_SCENARIOS:
        for primitive in OFFICIAL_G2_PRIMITIVES:
            for subject in OFFICIAL_G2_PARTICIPANTS:
                days = selected_days[(subject, primitive)]
                for draw in OFFICIAL_G2_DRAWS:
                    for sensor_day_id in days:
                        yield (subject, sensor_day_id, primitive, scenario, draw)


def _iter_mask_groups(
    parquet: pq.ParquetFile,
    *,
    day_positions: dict[tuple[str, str], dict[int, int]],
) -> Iterator[tuple[int, tuple[str, int, str, str, int], list[dict[str, object]]]]:
    rows = _iter_parquet_rows(
        parquet,
        columns=_MASK_COLUMNS,
        label="mask_intervals",
    )
    current_key: tuple[str, int, str, str, int] | None = None
    current_rows: list[dict[str, object]] = []
    current_index = -1
    last_index = -1
    for row in rows:
        key = (
            row["subject_id"],
            row["sensor_day_id"],
            row["primitive"],
            row["scenario"],
            row["draw"],
        )
        index = _topology_index(key, day_positions)
        if current_key is None:
            if index <= last_index:
                _fail("mask_intervals topology is duplicated or reordered")
            current_key = key
            current_index = index
        elif key != current_key:
            if index <= current_index:
                _fail("mask_intervals topology is duplicated or reordered")
            yield current_index, current_key, current_rows
            last_index = current_index
            current_key = key
            current_index = index
            current_rows = []
        current_rows.append(row)
    if current_key is not None:
        yield current_index, current_key, current_rows


def _next_required_row(
    rows: Iterator[dict[str, object]],
    label: str,
) -> dict[str, object]:
    try:
        return next(rows)
    except StopIteration:
        _fail(f"{label} is missing an expected topology row")


def load_official_g2_authority(
    manifest_path: Path,
    expected_manifest_sha256: str,
    task8a_clean_report_path: Path,
    expected_task8a_clean_report_sha256: str,
    task8a_clean_expectations_path: Path,
    expected_task8a_clean_expectations_sha256: str,
) -> OfficialG2Authority:
    """Load the sealed historical G2 graph under the Task 8A CLEAN authority."""

    expected_manifest = _require_sha256(
        expected_manifest_sha256,
        "expected official manifest SHA-256",
    )
    if expected_manifest != OFFICIAL_G2_MANIFEST_SHA256:
        _fail("expected official manifest SHA-256 is not the sealed value")
    manifest, repo_root = _require_committed_runtime_file(
        manifest_path,
        expected_repo_relative_path=OFFICIAL_G2_MANIFEST_REPO_PATH,
        label="official G2 manifest path",
    )
    task8a_clean = load_task8a_clean_authority(
        task8a_clean_report_path,
        expected_task8a_clean_report_sha256,
        task8a_clean_expectations_path,
        expected_task8a_clean_expectations_sha256,
    )
    if _find_repo_root(task8a_clean.report_path, "Task 8A report") != repo_root:
        _fail("official G2 manifest and Task 8A authority cross Git worktrees")
    if (
        _find_repo_root(
            task8a_clean.expectations.expectations_path,
            "Task 8A expectations",
        )
        != repo_root
    ):
        _fail("official G2 manifest and Task 8A expectations cross Git worktrees")

    manifest_document, manifest_sha256 = _read_strict_json_file(
        manifest,
        expected_manifest,
        "official G2 manifest",
    )
    artifacts = _validate_official_manifest_document(manifest_document)
    g1_candidate = repo_root.joinpath(*PurePosixPath(OFFICIAL_G1_REPO_PATH).parts)
    g1_artifact, g1_root = _require_committed_runtime_file(
        g1_candidate,
        expected_repo_relative_path=OFFICIAL_G1_REPO_PATH,
        label="hard-reviewed G1 artifact path",
    )
    if g1_root != repo_root:
        _fail("official G2 manifest and G1 artifact cross Git worktrees")
    g1_artifact_sha256 = _validate_official_g1(g1_artifact)
    artifact_paths = _resolve_official_graph_paths(
        repo_root=repo_root,
        manifest_path=manifest,
        g1_artifact_path=g1_artifact,
        artifacts=artifacts,
    )
    artifact_by_name = {artifact.name: artifact for artifact in artifacts}

    with ExitStack() as stack:
        parquet_files: dict[str, pq.ParquetFile] = {}
        decision_rows: tuple[dict[str, object], ...] | None = None
        for artifact in artifacts:
            try:
                stream = stack.enter_context(artifact_paths[artifact.name].open("rb"))
            except OSError as exc:
                _fail(f"cannot open official artifact {artifact.name}: {exc}")
            if artifact.format == "parquet":
                parquet_files[artifact.name] = _validate_open_parquet_artifact(
                    stream,
                    artifact,
                    expected_schema_digest=_PARQUET_SCHEMA_DIGESTS[artifact.name],
                )
            else:
                decision_rows = _validate_open_decision_jsonl(stream, artifact)
        if decision_rows is None or not _exact_json_equal(
            list(decision_rows),
            manifest_document["decision_table"],
        ):
            _fail("decision_table JSONL and committed manifest summary are crosswired")

        selected_cells, selected_days = _load_selected_cells(
            parquet_files["selected_days"],
            artifact_by_name["selected_days"],
        )
        day_positions = {
            key: {sensor_day_id: index for index, sensor_day_id in enumerate(days)}
            for key, days in selected_days.items()
        }
        cell_rows = _iter_parquet_rows(
            parquet_files["cell_replays"],
            columns=_CELL_COLUMNS,
            label="cell_replays",
        )
        audit_rows = _iter_parquet_rows(
            parquet_files["deletion_audit"],
            columns=_DELETION_COLUMNS,
            label="deletion_audit",
        )
        mask_groups = _iter_mask_groups(
            parquet_files["mask_intervals"],
            day_positions=day_positions,
        )
        try:
            current_mask_group = next(mask_groups)
        except StopIteration:
            current_mask_group = None

        cell_evidence: list[OfficialCellEvidence] = []
        official_masks: list[OfficialMaskCell] = []
        seen_deletion_keys: set[str] = set()
        interval_count = 0
        unavailable_count = 0
        observed_mask_count = 0
        for topology_index, expected_key in enumerate(
            _iter_expected_topology(selected_days)
        ):
            cell_row = _next_required_row(cell_rows, "cell_replays")
            audit_row = _next_required_row(audit_rows, "deletion_audit")
            intervals: list[dict[str, object]] = []
            if current_mask_group is not None:
                mask_index, mask_key, mask_rows = current_mask_group
                if mask_index < topology_index:
                    _fail("mask_intervals contains an orphaned/reordered group")
                if mask_index == topology_index:
                    if mask_key != expected_key:
                        _fail("mask_intervals topology index/key is crosswired")
                    intervals = mask_rows
                    try:
                        current_mask_group = next(mask_groups)
                    except StopIteration:
                        current_mask_group = None
            evidence, mask = _build_official_pair(
                cell_row,
                audit_row,
                intervals,
                expected_key=expected_key,
                cell_replays_artifact_sha256=artifact_by_name[
                    "cell_replays"
                ].physical_sha256,
                mask_intervals_artifact_sha256=artifact_by_name[
                    "mask_intervals"
                ].physical_sha256,
                deletion_audit_artifact_sha256=artifact_by_name[
                    "deletion_audit"
                ].physical_sha256,
            )
            if evidence.deletion_audit_key in seen_deletion_keys:
                _fail("deletion_audit contains a duplicate deletion_audit_key")
            seen_deletion_keys.add(evidence.deletion_audit_key)
            interval_count += len(mask.intervals)
            unavailable_count += mask.status == "empirical_unavailable"
            observed_mask_count += mask.status == "observed_mask"
            cell_evidence.append(evidence)
            official_masks.append(mask)
        for rows, label in (
            (cell_rows, "cell_replays"),
            (audit_rows, "deletion_audit"),
        ):
            try:
                next(rows)
            except StopIteration:
                pass
            else:
                _fail(f"{label} contains extra topology rows")
        if current_mask_group is not None:
            _fail("mask_intervals contains extra/orphaned topology rows")
        try:
            next(mask_groups)
        except StopIteration:
            pass
        else:  # pragma: no cover - current group guard catches this
            _fail("mask_intervals contains extra topology groups")
        if (
            len(cell_evidence) != 320_000
            or len(official_masks) != 320_000
            or len(seen_deletion_keys) != 320_000
            or interval_count != 734_907
            or unavailable_count != 20_650
            or observed_mask_count != 299_350
        ):
            _fail("official G2 cell/deletion/mask totals are not exact")

    evidence_tuple = tuple(cell_evidence)
    mask_tuple = tuple(official_masks)
    authority_payload = _authority_payload_parts(
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
        root_seed="42",
        scenario_grid=OFFICIAL_G2_SCENARIOS,
        draws=OFFICIAL_G2_DRAWS,
        selected_cells=selected_cells,
        cell_evidence=evidence_tuple,
        official_masks=mask_tuple,
        official_artifacts=artifacts,
        historical_common_fingerprints=_HISTORICAL_COMMON_FINGERPRINTS,
        g1_artifact_path=g1_artifact,
        g1_artifact_sha256=g1_artifact_sha256,
        task8a_clean_authority_digest=task8a_clean.content_digest,
    )
    return OfficialG2Authority(
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
        root_seed="42",
        scenario_grid=OFFICIAL_G2_SCENARIOS,
        draws=OFFICIAL_G2_DRAWS,
        selected_cells=selected_cells,
        cell_evidence=evidence_tuple,
        official_masks=mask_tuple,
        official_artifacts=artifacts,
        historical_common_fingerprints=_HISTORICAL_COMMON_FINGERPRINTS,
        g1_artifact_path=g1_artifact,
        g1_artifact_sha256=g1_artifact_sha256,
        task8a_clean=task8a_clean,
        task8a_clean_authority_digest=task8a_clean.content_digest,
        content_digest=_tagged_digest("official-g2-authority-v1", authority_payload),
    )


def _canonical_frame_components(
    frame: pd.DataFrame,
    label: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, ...], ...],
    str,
    str,
]:
    """Return the dtype-bound, order-preserving logical frame representation."""

    if type(frame) is not pd.DataFrame:
        raise TypeError(f"{label} must be an exact pandas DataFrame")
    columns = tuple(frame.columns)
    if any(type(column) is not str or not column for column in columns):
        _fail(f"{label} columns must be nonempty exact built-in strings")
    if len(set(columns)) != len(columns):
        _fail(f"{label} columns must be unique")
    dtype_objects = tuple(frame.dtypes)
    dtypes = tuple(str(dtype) for dtype in dtype_objects)
    canonical_dtype_specs = {
        "str": pd.Series([], dtype="str").dtype,
        "bool": np.dtype("bool"),
        "int64": np.dtype("int64"),
        "float64": np.dtype("float64"),
        "datetime64[ns]": np.dtype("datetime64[ns]"),
    }
    unsupported = tuple(
        dtype_name
        for dtype, dtype_name in zip(dtype_objects, dtypes, strict=True)
        if dtype_name not in canonical_dtype_specs
        or type(dtype) is not type(canonical_dtype_specs[dtype_name])
        or dtype != canonical_dtype_specs[dtype_name]
    )
    if unsupported:
        _fail(f"{label} has unsupported or coercive dtypes: {unsupported!r}")

    logical_rows: list[tuple[str, ...]] = []
    for row_index in range(len(frame)):
        row_tokens: list[str] = []
        for column_index, (column, dtype) in enumerate(zip(columns, dtypes, strict=True)):
            value = frame.iloc[row_index, column_index]
            scalar_label = f"{label}[{row_index!r},{column!r}]"
            if dtype == "str":
                if bool(pd.isna(value)):
                    token = "null"
                elif type(value) is str:
                    token = json.dumps(
                        value,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                else:
                    _fail(f"{scalar_label} must be an exact string or missing")
            elif dtype == "bool":
                if type(value) not in {bool, np.bool_}:
                    _fail(f"{scalar_label} must be an exact bool scalar")
                token = "true" if bool(value) else "false"
            elif dtype == "int64":
                if type(value) not in {int, np.int64} or type(value) is bool:
                    _fail(f"{scalar_label} must be an exact int64 scalar")
                token = str(int(value))
            elif dtype == "float64":
                if type(value) not in {float, np.float64}:
                    _fail(f"{scalar_label} must be an exact float64 scalar")
                numeric = float(value)
                if math.isnan(numeric):
                    token = "nan"
                elif not math.isfinite(numeric):
                    _fail(f"{scalar_label} must not be infinite")
                else:
                    token = numeric.hex()
            else:
                if pd.isna(value):
                    token = "nat"
                elif type(value) is pd.Timestamp and value.tz is None:
                    token = value.isoformat(timespec="nanoseconds")
                else:
                    _fail(
                        f"{scalar_label} must be an exact timezone-naive "
                        "nanosecond Timestamp or NaT"
                    )
            row_tokens.append(token)
        logical_rows.append(tuple(row_tokens))

    schema_digest = _tagged_digest(
        "canonical-frame-schema-v1",
        {"columns": list(columns), "dtypes": list(dtypes)},
    )
    logical_tuple = tuple(logical_rows)
    logical_digest = _tagged_digest(
        "canonical-frame-logical-v1",
        {
            "columns": list(columns),
            "dtypes": list(dtypes),
            "rows": [list(row) for row in logical_tuple],
        },
    )
    return columns, dtypes, logical_tuple, schema_digest, logical_digest


def _canonical_frame_payload(value: CanonicalFrameAuthority) -> dict[str, object]:
    return {
        "columns": list(value.columns),
        "dtypes": list(value.dtypes),
        "logical_digest": value.logical_digest,
        "logical_rows": [list(row) for row in value.logical_rows],
        "path": str(value.path),
        "physical_sha256": value.physical_sha256,
        "row_count": value.row_count,
        "schema_digest": value.schema_digest,
    }


def _require_canonical_logical_token(dtype: str, token: str, label: str) -> None:
    if type(token) is not str:
        _fail(f"{label} must be an exact built-in string token")
    if dtype == "str":
        if token == "null":
            return
        try:
            decoded = json.loads(token)
        except (json.JSONDecodeError, TypeError, ValueError):
            _fail(f"{label} is not a canonical JSON string token")
        if type(decoded) is not str or token != json.dumps(
            decoded,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ):
            _fail(f"{label} is not a canonical JSON string token")
        return
    if dtype == "bool":
        if token not in {"true", "false"}:
            _fail(f"{label} is not a canonical bool token")
        return
    if dtype == "int64":
        if re.fullmatch(r"(?:0|-?[1-9][0-9]*)", token) is None:
            _fail(f"{label} is not a canonical base-10 int64 token")
        numeric = int(token)
        if numeric < -(2**63) or numeric > 2**63 - 1:
            _fail(f"{label} is outside the int64 domain")
        return
    if dtype == "float64":
        if token == "nan":
            return
        try:
            numeric = float.fromhex(token)
        except (OverflowError, ValueError):
            _fail(f"{label} is not a canonical float64 token")
        if not math.isfinite(numeric) or numeric.hex() != token:
            _fail(f"{label} is not a canonical finite float64 token")
        return
    if token == "nat":
        return
    try:
        timestamp = pd.Timestamp(token)
    except (OverflowError, TypeError, ValueError):
        _fail(f"{label} is not a canonical nanosecond timestamp token")
    if (
        type(timestamp) is not pd.Timestamp
        or pd.isna(timestamp)
        or timestamp.tz is not None
        or timestamp.isoformat(timespec="nanoseconds") != token
    ):
        _fail(f"{label} is not a canonical nanosecond timestamp token")


def _validate_canonical_frame_authority_value(
    value: CanonicalFrameAuthority,
) -> None:
    if type(value) is not CanonicalFrameAuthority:
        raise TypeError("value must be an exact CanonicalFrameAuthority")
    if type(value.path) is not _PATH_TYPE or not value.path.is_absolute():
        _fail("canonical frame path must be an exact absolute pathlib Path")
    if any(part in {".", ".."} for part in value.path.parts):
        _fail("canonical frame path must be lexical and nontraversing")
    _require_sha256(value.physical_sha256, "canonical frame physical_sha256")
    _rehash_file_snapshot(
        value.path,
        value.physical_sha256,
        "canonical frame authority physical file",
    )
    if type(value.row_count) is not int or value.row_count < 0:
        _fail("canonical frame row_count must be an exact nonnegative integer")
    if type(value.columns) is not tuple or any(
        type(column) is not str or not column for column in value.columns
    ):
        _fail("canonical frame columns must be an exact tuple of strings")
    if len(set(value.columns)) != len(value.columns):
        _fail("canonical frame columns must be unique")
    if type(value.dtypes) is not tuple or len(value.dtypes) != len(value.columns):
        _fail("canonical frame dtypes must align exactly with columns")
    if any(
        type(dtype) is not str
        or dtype not in {"str", "bool", "int64", "float64", "datetime64[ns]"}
        for dtype in value.dtypes
    ):
        _fail("canonical frame dtypes contain an unsupported dtype")
    if type(value.logical_rows) is not tuple or len(value.logical_rows) != value.row_count:
        _fail("canonical frame logical_rows count is not exact")
    for row_index, row in enumerate(value.logical_rows):
        if type(row) is not tuple or len(row) != len(value.columns) or any(
            type(token) is not str for token in row
        ):
            _fail("canonical frame logical row topology is invalid")
        for column_index, (dtype, token) in enumerate(
            zip(value.dtypes, row, strict=True)
        ):
            _require_canonical_logical_token(
                dtype,
                token,
                f"canonical frame logical_rows[{row_index}][{column_index}]",
            )
    expected_schema = _tagged_digest(
        "canonical-frame-schema-v1",
        {"columns": list(value.columns), "dtypes": list(value.dtypes)},
    )
    _require_sha256(value.schema_digest, "canonical frame schema_digest")
    if value.schema_digest != expected_schema:
        _fail("canonical frame schema_digest is crosswired")
    expected_logical = _tagged_digest(
        "canonical-frame-logical-v1",
        {
            "columns": list(value.columns),
            "dtypes": list(value.dtypes),
            "rows": [list(row) for row in value.logical_rows],
        },
    )
    _require_sha256(value.logical_digest, "canonical frame logical_digest")
    if value.logical_digest != expected_logical:
        _fail("canonical frame logical_digest is crosswired")
    expected_content = _tagged_digest(
        "canonical-frame-authority-v1",
        _canonical_frame_payload(value),
    )
    _require_content_digest_value(
        value.content_digest,
        expected_content,
        "canonical frame authority",
    )


def _canonical_frame_authority_from_frame(
    path: Path,
    physical_sha256: str,
    frame: pd.DataFrame,
    label: str,
) -> CanonicalFrameAuthority:
    if type(path) is not _PATH_TYPE or not path.is_absolute():
        _fail(f"{label} path must be an exact absolute pathlib Path")
    physical = _require_sha256(physical_sha256, f"{label} physical SHA-256")
    columns, dtypes, rows, schema_digest, logical_digest = (
        _canonical_frame_components(frame, label)
    )
    payload = {
        "columns": list(columns),
        "dtypes": list(dtypes),
        "logical_digest": logical_digest,
        "logical_rows": [list(row) for row in rows],
        "path": str(path),
        "physical_sha256": physical,
        "row_count": len(frame),
        "schema_digest": schema_digest,
    }
    return CanonicalFrameAuthority(
        path=path,
        physical_sha256=physical,
        row_count=len(frame),
        columns=columns,
        dtypes=dtypes,
        logical_rows=rows,
        schema_digest=schema_digest,
        logical_digest=logical_digest,
        content_digest=_tagged_digest("canonical-frame-authority-v1", payload),
    )


def _require_canonical_range_index(frame: pd.DataFrame, label: str) -> None:
    index = frame.index
    if (
        type(index) is not pd.RangeIndex
        or index.start != 0
        or index.stop != len(frame)
        or index.step != 1
        or index.name is not None
    ):
        _fail(f"{label} must retain the canonical unnamed zero-based RangeIndex")


def _require_frame_logical_equality(
    frame: pd.DataFrame,
    authority: CanonicalFrameAuthority,
    label: str,
) -> None:
    if type(authority) is not CanonicalFrameAuthority:
        raise TypeError(f"{label} authority must be an exact CanonicalFrameAuthority")
    _validate_canonical_frame_authority_value(authority)
    _require_canonical_range_index(frame, label)
    components = _canonical_frame_components(frame, label)
    if components != (
        authority.columns,
        authority.dtypes,
        authority.logical_rows,
        authority.schema_digest,
        authority.logical_digest,
    ):
        _fail(f"{label} is not full-schema/dtype/value/status/order identical")


def _frame_logical_digest(frame: pd.DataFrame, label: str = "frame") -> str:
    _require_canonical_range_index(frame, label)
    return _canonical_frame_components(frame, label)[4]


def _is_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as exc:
        _fail(f"cannot inspect path metadata for {path}: {exc}")
    return bool(getattr(metadata, "st_file_attributes", 0) & 0x400)


def _require_canonical_directory(value: object, label: str) -> Path:
    if type(value) is not _PATH_TYPE:
        raise TypeError(f"{label} must be an exact pathlib Path")
    if not value.is_absolute() or any(part in {".", ".."} for part in value.parts):
        _fail(f"{label} must be one canonical absolute path")
    lexical = Path(os.path.abspath(value))
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve {label}: {exc}")
    if (
        lexical != resolved
        or str(lexical) != str(resolved)
        or not resolved.is_dir()
        or resolved.is_symlink()
        or _is_reparse_point(resolved)
    ):
        _fail(f"{label} must not use a symlink, junction, alias, or non-directory")
    return resolved


def _require_distinct_paths(paths: tuple[Path, ...], label: str) -> None:
    textual = tuple(os.path.normcase(os.path.abspath(path)) for path in paths)
    if len(set(textual)) != len(textual):
        _fail(f"{label} paths must be textually distinct")
    identities: set[tuple[int, int]] = set()
    for path in paths:
        try:
            metadata = path.stat()
        except OSError as exc:
            _fail(f"cannot inspect {label} path identity: {exc}")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in identities:
            _fail(f"{label} contains a same-file or hardlink alias")
        identities.add(identity)


def _resolve_calibration_input_paths(
    *,
    official_repo_root: Path,
    raw_root: Path,
    canonical_root: Path,
    primitive_table_path: Path,
    frozen_population_baselines_path: Path,
) -> _CalibrationInputPaths:
    repo_root = _require_canonical_directory(
        official_repo_root,
        "official authority repository root",
    )
    if not (repo_root / ".git").exists():
        _fail("official authority repository root is not a Git worktree")
    primitive_path, primitive_repo = _require_committed_runtime_file(
        primitive_table_path,
        expected_repo_relative_path=CALIBRATION_PRIMITIVES_REPO_PATH,
        label="sealed calibration primitive table",
    )
    baseline_path, baseline_repo = _require_committed_runtime_file(
        frozen_population_baselines_path,
        expected_repo_relative_path=CALIBRATION_BASELINES_REPO_PATH,
        label="sealed calibration Task 3 baselines",
    )
    if primitive_repo != repo_root or baseline_repo != repo_root:
        _fail("calibration caches cross the official authority Git worktree")
    raw = _require_canonical_directory(raw_root, "raw input root")
    canonical = _require_canonical_directory(canonical_root, "canonical input root")
    if (
        raw.name != "data_vvs"
        or raw.parent.name != "raw"
        or canonical.name != "canonical"
        or raw.parent.parent != canonical.parent
    ):
        _fail("raw and canonical roots are crosswired or outside the exact data layout")
    _require_distinct_paths(
        (raw, canonical, primitive_path, baseline_path),
        "calibration input graph",
    )
    return _CalibrationInputPaths(
        raw_root=raw,
        canonical_root=canonical,
        primitive_table_path=primitive_path,
        frozen_population_baselines_path=baseline_path,
    )


def _read_verified_bytes_once(
    path: Path,
    expected_sha256: str,
    label: str,
) -> bytes:
    expected = _require_sha256(expected_sha256, f"expected {label} SHA-256")
    try:
        with path.open("rb") as stream:
            payload = stream.read()
    except OSError as exc:
        _fail(f"cannot read {label}: {exc}")
    actual = _sha256_bytes(payload)
    if actual != expected:
        _fail(f"{label} physical hash mismatch: expected={expected} actual={actual}")
    return payload


def _rehash_file_snapshot(path: Path, expected_sha256: str, label: str) -> None:
    expected = _require_sha256(expected_sha256, f"expected {label} SHA-256")
    try:
        actual = _sha256_bytes(path.read_bytes())
    except OSError as exc:
        _fail(f"cannot rehash {label}: {exc}")
    if actual != expected:
        _fail(f"{label} changed after its verified snapshot")


def _load_canonical_frame_authority(
    path: Path,
    *,
    expected_physical_sha256: str,
    expected_repo_root: Path,
    expected_repo_relative_path: str,
    label: str,
) -> tuple[CanonicalFrameAuthority, pd.DataFrame, bytes]:
    resolved, repo_root = _require_committed_runtime_file(
        path,
        expected_repo_relative_path=expected_repo_relative_path,
        label=label,
    )
    if repo_root != expected_repo_root:
        _fail(f"{label} crosses the official authority Git worktree")
    raw = _read_verified_bytes_once(resolved, expected_physical_sha256, label)
    try:
        frame = pd.read_parquet(BytesIO(raw))
    except Exception as exc:
        _fail(f"cannot parse verified {label} parquet bytes: {exc}")
    if type(frame) is not pd.DataFrame:
        _fail(f"verified {label} did not decode to an exact DataFrame")
    _require_canonical_range_index(frame, label)
    authority = _canonical_frame_authority_from_frame(
        resolved,
        expected_physical_sha256,
        frame,
        label,
    )
    return authority, frame, raw


def _historical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _tree_entries(root: Path, label: str) -> tuple[Path, ...]:
    entries: list[Path] = []
    try:
        candidates = tuple(root.rglob("*"))
    except OSError as exc:
        _fail(f"cannot enumerate {label}: {exc}")
    for candidate in candidates:
        if candidate.is_symlink() or _is_reparse_point(candidate):
            _fail(f"{label} contains a symlink, junction, or reparse point")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            _fail(f"{label} contains a nonregular entry")
        entries.append(candidate)
    if not entries:
        _fail(f"{label} file tree is empty")
    return tuple(entries)


def _historical_tree_fingerprint(
    root: Path,
    files: tuple[Path, ...],
    label: str,
    *,
    require_complete: bool = True,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    canonical_root = _require_canonical_directory(root, f"{label} root")
    if type(files) is not tuple or not files:
        _fail(f"{label} files must be a nonempty exact tuple")
    resolved_files: list[Path] = []
    relative_paths: set[str] = set()
    identities: set[tuple[int, int]] = set()
    for index, candidate in enumerate(files):
        if type(candidate) is not _PATH_TYPE or not candidate.is_absolute():
            raise TypeError(f"{label} file[{index}] must be an exact absolute Path")
        lexical = Path(os.path.abspath(candidate))
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(canonical_root).as_posix()
        except (OSError, ValueError) as exc:
            _fail(f"{label} file[{index}] escapes or is absent: {exc}")
        if (
            lexical != resolved
            or str(lexical) != str(resolved)
            or resolved.is_symlink()
            or _is_reparse_point(resolved)
            or not resolved.is_file()
        ):
            _fail(f"{label} file[{index}] is aliased or nonregular")
        metadata = resolved.stat()
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if relative in relative_paths or identity in identities:
            _fail(f"{label} files contain a duplicate or same-file alias")
        relative_paths.add(relative)
        identities.add(identity)
        resolved_files.append(resolved)
    if require_complete:
        complete = _tree_entries(canonical_root, label)
        complete_relatives = {
            path.resolve().relative_to(canonical_root).as_posix() for path in complete
        }
        if relative_paths != complete_relatives:
            _fail(f"{label} files omit or add complete-tree entries")
        _require_distinct_paths(complete, label)

    records: list[dict[str, object]] = []
    ordered_hashes: list[tuple[str, str]] = []
    for path in sorted(resolved_files, key=str):
        relative = path.relative_to(canonical_root).as_posix()
        digest = _sha256_bytes(path.read_bytes())
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": digest}
        )
        ordered_hashes.append((relative, digest))
    return (
        _sha256_bytes(_historical_json_bytes(records)),
        tuple(ordered_hashes),
    )


def _complete_historical_tree_fingerprint(
    root: Path,
    label: str,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    canonical_root = _require_canonical_directory(root, f"{label} root")
    files = _tree_entries(canonical_root, label)
    return _historical_tree_fingerprint(
        canonical_root,
        files,
        label,
        require_complete=True,
    )


def _calibration_sensor_fingerprint(
    canonical_root: Path,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    root = _require_canonical_directory(canonical_root, "canonical sensor root")
    try:
        entry_names = {entry.name for entry in os.scandir(root)}
    except OSError as exc:
        _fail(f"cannot enumerate canonical sensor root: {exc}")
    for filename in CALIBRATION_SENSOR_FILES:
        if filename not in entry_names:
            aliases = sorted(
                name for name in entry_names if name.casefold() == filename.casefold()
            )
            if aliases:
                _fail(f"canonical sensor {filename} uses a case-only alias")
            _fail(f"canonical sensor {filename} is absent")
    files = tuple(root / filename for filename in CALIBRATION_SENSOR_FILES)
    digest, sorted_hashes = _historical_tree_fingerprint(
        root,
        files,
        "six calibration sensors",
        require_complete=False,
    )
    hash_by_name = dict(sorted_hashes)
    return digest, tuple((name, hash_by_name[name]) for name in CALIBRATION_SENSOR_FILES)


def _reconstruction_file_hashes(
    repo_root: Path,
) -> tuple[tuple[str, str], ...]:
    root = _require_canonical_directory(repo_root, "reconstruction repository root")
    files: list[Path] = []
    hashes: list[tuple[str, str]] = []
    for path_text in _CALIBRATION_RECONSTRUCTION_FILES:
        path = _resolve_repo_file(root, path_text, f"reconstruction source {path_text}")
        if path.is_symlink() or _is_reparse_point(path):
            _fail(f"reconstruction source {path_text} is aliased")
        files.append(path)
        hashes.append((path_text, _sha256_bytes(path.read_bytes())))
    _require_distinct_paths(tuple(files), "reconstruction sources")
    return tuple(hashes)


def _required_calibration_historical_fingerprints(
    historical: tuple[tuple[str, str], ...],
) -> tuple[str, str, str]:
    if type(historical) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not str
        or type(item[1]) is not str
        for item in historical
    ):
        _fail("historical common fingerprints must be an exact tuple graph")
    if len({name for name, _digest in historical}) != len(historical):
        _fail("historical common fingerprints contain duplicate names")
    values = dict(historical)
    result: list[str] = []
    for name, expected in _CALIBRATION_HISTORICAL_FINGERPRINTS:
        actual = _require_sha256(values.get(name), f"historical {name}")
        if actual != expected:
            _fail(f"historical {name} is not the sealed G2 value")
        result.append(actual)
    return result[0], result[1], result[2]


def _calibration_historical_fingerprints(
    raw_root: Path,
    canonical_root: Path,
) -> tuple[str, str, str, tuple[tuple[str, str], ...]]:
    raw_digest, _raw_files = _complete_historical_tree_fingerprint(
        raw_root,
        "complete raw input tree",
    )
    canonical_digest, _canonical_files = _complete_historical_tree_fingerprint(
        canonical_root,
        "complete canonical input tree",
    )
    sensors_digest, sensor_hashes = _calibration_sensor_fingerprint(canonical_root)
    return raw_digest, canonical_digest, sensors_digest, sensor_hashes


def _reconstruct_calibration_inputs(
    canonical_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaffold = build_sensor_calendar_scaffold(canonical_root)
    primitives = extract_daily_primitives(scaffold, canonical_root)
    task3 = build_loso_representations(primitives, k_values=(14,))
    return scaffold, primitives, task3.baselines


def _require_reconstructed_calibration_topology(
    primitives: pd.DataFrame,
    baselines: pd.DataFrame,
) -> None:
    _require_canonical_range_index(primitives, "reconstructed primitive table")
    _require_canonical_range_index(baselines, "reconstructed Task 3 baselines")
    if primitives.shape != (853, 231):
        _fail("reconstructed primitive table must be exactly 853x231")
    if baselines.shape != (1920, 23):
        _fail("reconstructed complete Task 3 baselines must be exactly 1920x23")
    primitive_keys = {
        "subject_id",
        "row_id",
        "sensor_day_id",
        "elapsed_day_index",
    }
    if not primitive_keys.issubset(primitives.columns):
        _fail("reconstructed primitive table omits a required identity column")
    if tuple(primitives["row_id"]) != tuple(range(853)) or tuple(
        primitives["sensor_day_id"]
    ) != tuple(range(853)):
        _fail("reconstructed primitive row/sensor-day IDs are not canonical")
    subject_counts = primitives.groupby("subject_id", sort=False).size().to_dict()
    if subject_counts != {
        "id01": 81,
        "id02": 91,
        "id03": 85,
        "id04": 91,
        "id05": 84,
        "id06": 83,
        "id07": 85,
        "id08": 87,
        "id09": 83,
        "id10": 83,
    }:
        _fail("reconstructed primitive participant topology is not exact")
    baseline_keys = {
        "held_out_subject",
        "donor_subject",
        "k",
        "method",
        "primitive",
    }
    if not baseline_keys.issubset(baselines.columns):
        _fail("reconstructed baselines omit the complete Task 3 identity")
    if baselines.duplicated(list(baseline_keys)).any():
        _fail("reconstructed baselines contain duplicate complete Task 3 keys")
    counts = baselines.groupby(["method", "k"], dropna=False, sort=False).size().to_dict()
    if counts != {
        ("population", 0): 160,
        ("personal", 14): 160,
        ("shrinkage", 14): 160,
        ("placebo", 14): 1440,
    }:
        _fail("reconstructed complete Task 3 baseline topology is not exact")


def _calibration_censoring_contract_digest(primitives: pd.DataFrame) -> str:
    if type(primitives) is not pd.DataFrame:
        raise TypeError("calibration censoring input must be an exact DataFrame")
    required = {"subject_id", "sensor_day_id", "elapsed_day_index"}
    if not required.issubset(primitives.columns):
        _fail("calibration censoring audit lacks participant-day identity")
    prefix = primitives.loc[primitives["elapsed_day_index"].le(14)].copy()
    if len(prefix) != 140:
        _fail("calibration censoring audit requires exactly 140 prefix days")
    if prefix.duplicated(["subject_id", "sensor_day_id"]).any():
        _fail("calibration censoring prefix contains duplicate participant-days")
    expected_subjects = set(OFFICIAL_G2_PARTICIPANTS)
    if set(prefix["subject_id"]) != expected_subjects:
        _fail("calibration censoring prefix participant roster is not exact")
    elapsed_by_subject = {
        subject: tuple(
            int(value)
            for value in prefix.loc[
                prefix["subject_id"].eq(subject), "elapsed_day_index"
            ]
        )
        for subject in OFFICIAL_G2_PARTICIPANTS
    }
    if any(values != tuple(range(1, 15)) for values in elapsed_by_subject.values()):
        _fail("calibration censoring prefix day order/topology is not exact")

    censoring_states: list[dict[str, object]] = []
    for primitive in CORE_G2_PRIMITIVES:
        column = f"{primitive}__censoring"
        if column not in prefix.columns:
            censoring_states.append(
                {"column": column, "primitive": primitive, "state": "absent"}
            )
            continue
        if str(prefix[column].dtype) != "str" or any(
            type(value) is not str or value != "none" for value in prefix[column]
        ):
            _fail(
                f"calibration censoring column {column} must contain exact 'none'"
            )
        censoring_states.append(
            {"column": column, "primitive": primitive, "state": "all_exact_none"}
        )
    keys = [
        [str(subject), int(sensor_day_id), int(elapsed)]
        for subject, sensor_day_id, elapsed in prefix[
            ["subject_id", "sensor_day_id", "elapsed_day_index"]
        ].itertuples(index=False, name=None)
    ]
    return _tagged_digest(
        "calibration-censoring-contract-v1",
        {
            "audited_cells": 140 * len(CORE_G2_PRIMITIVES),
            "core_primitives": list(CORE_G2_PRIMITIVES),
            "participant_days": keys,
            "states": censoring_states,
        },
    )


def _calibration_input_payload(value: CalibrationInputAuthority) -> dict[str, object]:
    return {
        "calibration_censoring_contract_digest": (
            value.calibration_censoring_contract_digest
        ),
        "canonical_inputs_sha256": value.canonical_inputs_sha256,
        "canonical_root": str(value.canonical_root),
        "frozen_population_baselines_digest": (
            value.frozen_population_baselines.content_digest
        ),
        "official_g2_digest": value.official_g2_digest,
        "primitive_table_digest": value.primitive_table.content_digest,
        "raw_inputs_sha256": value.raw_inputs_sha256,
        "raw_root": str(value.raw_root),
        "reconstructed_baselines_digest": value.reconstructed_baselines_digest,
        "reconstructed_calendar_digest": value.reconstructed_calendar_digest,
        "reconstructed_primitive_table_digest": (
            value.reconstructed_primitive_table_digest
        ),
        "reconstruction_file_hashes": [
            list(item) for item in value.reconstruction_file_hashes
        ],
        "sensor_file_hashes": [list(item) for item in value.sensor_file_hashes],
        "sensors_sha256": value.sensors_sha256,
        "task8a_clean_authority_digest": value.task8a_clean_authority_digest,
    }


def _validate_hash_pairs(
    value: object,
    expected_names: tuple[str, ...],
    label: str,
) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple or len(value) != len(expected_names):
        _fail(f"{label} must be the exact ordered tuple")
    if tuple(item[0] for item in value if type(item) is tuple and len(item) == 2) != expected_names:
        _fail(f"{label} names/order are not exact")
    for item, name in zip(value, expected_names, strict=True):
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] != name
        ):
            _fail(f"{label} entry topology is invalid")
        _require_sha256(item[1], f"{label}[{name}]")
    return value


def _validate_calibration_input_authority_value(
    value: CalibrationInputAuthority,
) -> None:
    if type(value) is not CalibrationInputAuthority:
        raise TypeError("value must be an exact CalibrationInputAuthority")
    if type(value.primitive_table) is not CanonicalFrameAuthority or type(
        value.frozen_population_baselines
    ) is not CanonicalFrameAuthority:
        _fail("calibration authority requires exact nested frame authorities")
    _validate_canonical_frame_authority_value(value.primitive_table)
    _validate_canonical_frame_authority_value(value.frozen_population_baselines)
    _primitive_path, primitive_repo_root = _require_committed_runtime_file(
        value.primitive_table.path,
        expected_repo_relative_path=CALIBRATION_PRIMITIVES_REPO_PATH,
        label="calibration primitive authority path",
    )
    _baseline_path, baseline_repo_root = _require_committed_runtime_file(
        value.frozen_population_baselines.path,
        expected_repo_relative_path=CALIBRATION_BASELINES_REPO_PATH,
        label="calibration baseline authority path",
    )
    if primitive_repo_root != baseline_repo_root:
        _fail("calibration cache authorities cross Git worktrees")
    if value.primitive_table.physical_sha256 != CALIBRATION_PRIMITIVES_SHA256:
        _fail("calibration primitive physical SHA-256 is not sealed")
    if (
        value.frozen_population_baselines.physical_sha256
        != CALIBRATION_BASELINES_SHA256
    ):
        _fail("calibration baseline physical SHA-256 is not sealed")
    _require_calibration_cache_sentinels(
        value.primitive_table,
        value.frozen_population_baselines,
    )
    _rehash_file_snapshot(
        _primitive_path,
        CALIBRATION_PRIMITIVES_SHA256,
        "calibration primitive authority live cache",
    )
    _rehash_file_snapshot(
        _baseline_path,
        CALIBRATION_BASELINES_SHA256,
        "calibration baseline authority live cache",
    )
    resolved_inputs = _resolve_calibration_input_paths(
        official_repo_root=primitive_repo_root,
        raw_root=value.raw_root,
        canonical_root=value.canonical_root,
        primitive_table_path=_primitive_path,
        frozen_population_baselines_path=_baseline_path,
    )
    raw_root = resolved_inputs.raw_root
    canonical_root = resolved_inputs.canonical_root
    expected_raw, expected_canonical, expected_sensors = (
        digest for _name, digest in _CALIBRATION_HISTORICAL_FINGERPRINTS
    )
    _require_sha256(value.raw_inputs_sha256, "calibration raw fingerprint")
    if value.raw_inputs_sha256 != expected_raw:
        _fail("calibration raw fingerprint is not historical")
    _require_sha256(
        value.canonical_inputs_sha256,
        "calibration canonical fingerprint",
    )
    if value.canonical_inputs_sha256 != expected_canonical:
        _fail("calibration canonical fingerprint is not historical")
    _require_sha256(value.sensors_sha256, "calibration six-sensor fingerprint")
    if value.sensors_sha256 != expected_sensors:
        _fail("calibration six-sensor fingerprint is not historical")
    _validate_hash_pairs(
        value.sensor_file_hashes,
        CALIBRATION_SENSOR_FILES,
        "calibration sensor_file_hashes",
    )
    if value.sensor_file_hashes != _CALIBRATION_SENSOR_FILE_HASHES:
        _fail("calibration sensor_file_hashes are not the sealed six-sensor tuple")
    live_calibration_fingerprints = _calibration_historical_fingerprints(
        raw_root,
        canonical_root,
    )
    if live_calibration_fingerprints != (
        value.raw_inputs_sha256,
        value.canonical_inputs_sha256,
        value.sensors_sha256,
        value.sensor_file_hashes,
    ):
        _fail("calibration roots or six-sensor inputs are not live-bound")
    _validate_hash_pairs(
        value.reconstruction_file_hashes,
        _CALIBRATION_RECONSTRUCTION_FILES,
        "calibration reconstruction_file_hashes",
    )
    for field_name in (
        "reconstructed_calendar_digest",
        "reconstructed_primitive_table_digest",
        "reconstructed_baselines_digest",
        "calibration_censoring_contract_digest",
        "official_g2_digest",
        "task8a_clean_authority_digest",
    ):
        _require_sha256(getattr(value, field_name), f"calibration {field_name}")
    if (
        value.reconstructed_calendar_digest
        != _CALIBRATION_RECONSTRUCTED_CALENDAR_DIGEST
    ):
        _fail("calibration reconstructed calendar digest is not sealed")
    if (
        value.calibration_censoring_contract_digest
        != _CALIBRATION_CENSORING_CONTRACT_DIGEST
    ):
        _fail("calibration censoring contract digest is not sealed")
    if value.official_g2_digest != _CALIBRATION_OFFICIAL_G2_AUTHORITY_DIGEST:
        _fail("calibration official G2 digest is not sealed")
    if (
        value.task8a_clean_authority_digest
        != _CALIBRATION_TASK8A_CLEAN_AUTHORITY_DIGEST
    ):
        _fail("calibration Task 8A CLEAN authority digest is not sealed")
    if (
        value.reconstructed_primitive_table_digest
        != value.primitive_table.logical_digest
    ):
        _fail("reconstructed primitive digest is crosswired")
    if (
        value.reconstructed_baselines_digest
        != value.frozen_population_baselines.logical_digest
    ):
        _fail("reconstructed baseline digest is crosswired")
    live_reconstruction_hashes = _reconstruction_file_hashes(primitive_repo_root)
    if value.reconstruction_file_hashes != live_reconstruction_hashes:
        _fail("calibration reconstruction source hashes are not current")
    expected_content = _tagged_digest(
        "calibration-input-authority-v1",
        _calibration_input_payload(value),
    )
    _require_content_digest_value(
        value.content_digest,
        expected_content,
        "calibration input authority",
    )


def _require_calibration_cache_sentinels(
    primitives: CanonicalFrameAuthority,
    baselines: CanonicalFrameAuthority,
) -> None:
    if (
        primitives.schema_digest
        != "8fe559ab25e880da9d772a9303364e5c7c4957471569c1c17497eccf225f3a0d"
        or primitives.logical_digest
        != "ea6249201b2a4a167b93cb2b65ee06ef0017a9a8b96c663dba9c6b97fb055db6"
        or baselines.schema_digest
        != "6907db6bf53cd3046c6b1dd84f7861981cd9a378b0a73a5745d78231d8889133"
        or baselines.logical_digest
        != "12d93d46dd7906ddb8b41cf621ca7d3f59ca2f2f2a34d6212f5854a9c4d139c8"
    ):
        _fail("independently recomputed calibration cache sentinel is not exact")


def _revalidate_calibration_inputs(
    *,
    paths: _CalibrationInputPaths,
    repo_root: Path,
    raw_inputs_sha256: str,
    canonical_inputs_sha256: str,
    sensors_sha256: str,
    sensor_file_hashes: tuple[tuple[str, str], ...],
    reconstruction_file_hashes: tuple[tuple[str, str], ...],
) -> None:
    resolved = _resolve_calibration_input_paths(
        official_repo_root=repo_root,
        raw_root=paths.raw_root,
        canonical_root=paths.canonical_root,
        primitive_table_path=paths.primitive_table_path,
        frozen_population_baselines_path=paths.frozen_population_baselines_path,
    )
    if resolved != paths:
        _fail("calibration input path graph changed during reconstruction")
    _rehash_file_snapshot(
        paths.primitive_table_path,
        CALIBRATION_PRIMITIVES_SHA256,
        "sealed calibration primitive table",
    )
    _rehash_file_snapshot(
        paths.frozen_population_baselines_path,
        CALIBRATION_BASELINES_SHA256,
        "sealed calibration Task 3 baselines",
    )
    late_raw, late_canonical, late_sensors, late_sensor_files = (
        _calibration_historical_fingerprints(paths.raw_root, paths.canonical_root)
    )
    if (
        late_raw != raw_inputs_sha256
        or late_canonical != canonical_inputs_sha256
        or late_sensors != sensors_sha256
        or late_sensor_files != sensor_file_hashes
    ):
        _fail("raw/canonical/sensor inputs changed during reconstruction")
    if _reconstruction_file_hashes(repo_root) != reconstruction_file_hashes:
        _fail("reconstruction source files changed during reconstruction")


def load_calibration_input_authority(
    official_g2: OfficialG2Authority,
    *,
    raw_root: Path,
    canonical_root: Path,
    primitive_table_path: Path,
    frozen_population_baselines_path: Path,
) -> CalibrationInputAuthority:
    """Rebuild and bind the complete historical calibration input authority."""

    if type(official_g2) is not OfficialG2Authority:
        raise TypeError("official_g2 must be an exact OfficialG2Authority")
    _validate_official_g2_authority_value(official_g2)
    repo_root = _find_repo_root(official_g2.manifest_path, "official G2 manifest")
    paths = _resolve_calibration_input_paths(
        official_repo_root=repo_root,
        raw_root=raw_root,
        canonical_root=canonical_root,
        primitive_table_path=primitive_table_path,
        frozen_population_baselines_path=frozen_population_baselines_path,
    )
    primitive_authority, sealed_primitives, _primitive_bytes = (
        _load_canonical_frame_authority(
            paths.primitive_table_path,
            expected_physical_sha256=CALIBRATION_PRIMITIVES_SHA256,
            expected_repo_root=repo_root,
            expected_repo_relative_path=CALIBRATION_PRIMITIVES_REPO_PATH,
            label="sealed calibration primitive table",
        )
    )
    baseline_authority, sealed_baselines, _baseline_bytes = (
        _load_canonical_frame_authority(
            paths.frozen_population_baselines_path,
            expected_physical_sha256=CALIBRATION_BASELINES_SHA256,
            expected_repo_root=repo_root,
            expected_repo_relative_path=CALIBRATION_BASELINES_REPO_PATH,
            label="sealed calibration Task 3 baselines",
        )
    )
    _require_calibration_cache_sentinels(primitive_authority, baseline_authority)
    _require_reconstructed_calibration_topology(sealed_primitives, sealed_baselines)

    expected_raw, expected_canonical, expected_sensors = (
        _required_calibration_historical_fingerprints(
            official_g2.historical_common_fingerprints
        )
    )
    (
        raw_inputs_sha256,
        canonical_inputs_sha256,
        sensors_sha256,
        sensor_file_hashes,
    ) = _calibration_historical_fingerprints(paths.raw_root, paths.canonical_root)
    if (
        raw_inputs_sha256 != expected_raw
        or canonical_inputs_sha256 != expected_canonical
        or sensors_sha256 != expected_sensors
    ):
        _fail("current calibration roots do not match historical G2 fingerprints")
    reconstruction_file_hashes = _reconstruction_file_hashes(repo_root)

    scaffold, reconstructed_primitives, reconstructed_baselines = (
        _reconstruct_calibration_inputs(paths.canonical_root)
    )
    _require_reconstructed_calibration_topology(
        reconstructed_primitives,
        reconstructed_baselines,
    )
    _require_frame_logical_equality(
        reconstructed_primitives,
        primitive_authority,
        "reconstructed primitive table",
    )
    _require_frame_logical_equality(
        reconstructed_baselines,
        baseline_authority,
        "reconstructed complete Task 3 baselines",
    )
    calibration_censoring_digest = _calibration_censoring_contract_digest(
        reconstructed_primitives
    )
    reconstructed_calendar_digest = _frame_logical_digest(
        scaffold,
        "reconstructed sensor calendar",
    )
    reconstructed_primitive_digest = _frame_logical_digest(
        reconstructed_primitives,
        "reconstructed primitive table",
    )
    reconstructed_baselines_digest = _frame_logical_digest(
        reconstructed_baselines,
        "reconstructed complete Task 3 baselines",
    )
    _validate_official_g2_authority_value(official_g2)

    payload = {
        "calibration_censoring_contract_digest": calibration_censoring_digest,
        "canonical_inputs_sha256": canonical_inputs_sha256,
        "canonical_root": str(paths.canonical_root),
        "frozen_population_baselines_digest": baseline_authority.content_digest,
        "official_g2_digest": official_g2.content_digest,
        "primitive_table_digest": primitive_authority.content_digest,
        "raw_inputs_sha256": raw_inputs_sha256,
        "raw_root": str(paths.raw_root),
        "reconstructed_baselines_digest": reconstructed_baselines_digest,
        "reconstructed_calendar_digest": reconstructed_calendar_digest,
        "reconstructed_primitive_table_digest": reconstructed_primitive_digest,
        "reconstruction_file_hashes": [
            list(item) for item in reconstruction_file_hashes
        ],
        "sensor_file_hashes": [list(item) for item in sensor_file_hashes],
        "sensors_sha256": sensors_sha256,
        "task8a_clean_authority_digest": official_g2.task8a_clean_authority_digest,
    }
    authority = CalibrationInputAuthority(
        primitive_table=primitive_authority,
        frozen_population_baselines=baseline_authority,
        raw_root=paths.raw_root,
        canonical_root=paths.canonical_root,
        raw_inputs_sha256=raw_inputs_sha256,
        canonical_inputs_sha256=canonical_inputs_sha256,
        sensors_sha256=sensors_sha256,
        sensor_file_hashes=sensor_file_hashes,
        reconstruction_file_hashes=reconstruction_file_hashes,
        reconstructed_calendar_digest=reconstructed_calendar_digest,
        reconstructed_primitive_table_digest=reconstructed_primitive_digest,
        reconstructed_baselines_digest=reconstructed_baselines_digest,
        calibration_censoring_contract_digest=calibration_censoring_digest,
        official_g2_digest=official_g2.content_digest,
        task8a_clean_authority_digest=official_g2.task8a_clean_authority_digest,
        content_digest=_tagged_digest("calibration-input-authority-v1", payload),
    )
    _revalidate_calibration_inputs(
        paths=paths,
        repo_root=repo_root,
        raw_inputs_sha256=raw_inputs_sha256,
        canonical_inputs_sha256=canonical_inputs_sha256,
        sensors_sha256=sensors_sha256,
        sensor_file_hashes=sensor_file_hashes,
        reconstruction_file_hashes=reconstruction_file_hashes,
    )
    return authority
