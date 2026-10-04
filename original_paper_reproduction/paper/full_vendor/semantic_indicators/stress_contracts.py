"""Fail-closed authority intake for the structured-missingness study."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat

import pandas as pd

from . import phaseguard_authority
from . import primitives


_PATH_TYPE = type(Path.cwd())
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)

PLAN_PATH = PurePosixPath(
    ".superpowers/sdd/task-12-structured-missingness-preregistered-analysis-plan.md"
)
CLARIFICATION_PATH = PurePosixPath(
    ".superpowers/sdd/task-12-preregistration-clarification-v1.md"
)
CLARIFICATION_V2_PATH = PurePosixPath(
    ".superpowers/sdd/task-12-preregistration-clarification-v2.md"
)
CLARIFICATION_V3_PATH = PurePosixPath(
    ".superpowers/sdd/task-12-preregistration-clarification-v3.md"
)
CLARIFICATION_V4_PATH = PurePosixPath(
    ".superpowers/sdd/task-12-preregistration-clarification-v4.md"
)
CLARIFICATION_V5_PATH = PurePosixPath(
    ".superpowers/sdd/task-12-preregistration-clarification-v5.md"
)
SEAL_V6_PATH = PurePosixPath(
    ".superpowers/sdd/task-12-structured-missingness-preregistration-seal-v6.json"
)
OFFICIAL_MANIFEST_PATH = PurePosixPath(
    ".superpowers/sdd/task-4c2-v2-official/manifest.json"
)
TASK8A_CLEAN_REPORT_PATH = PurePosixPath(
    ".superpowers/sdd/task-8a-clean-v3.json"
)
TASK8A_CLEAN_EXPECTATIONS_PATH = PurePosixPath(
    ".superpowers/sdd/task-8a-clean-expectations-v1.json"
)
CALIBRATION_PRIMITIVES_PATH = PurePosixPath(
    ".superpowers/sdd/task-4c2-real-primitives.parquet"
)
CALIBRATION_BASELINES_PATH = PurePosixPath(
    ".superpowers/sdd/task-4c2-real-baselines.parquet"
)

PLAN_BYTES = 21_947
PLAN_SHA256 = "b87ec6e31f1939c4df1f53e23b1e1161cfa0b244d6161413c8c566a3205ce7d7"
CLARIFICATION_BYTES = 5_685
CLARIFICATION_SHA256 = (
    "b392b2ed8d23904406e5adc8be8833e8ad1a25eecf93ff11ebba46611b519d7a"
)
CLARIFICATION_V2_BYTES = 1_187
CLARIFICATION_V2_SHA256 = (
    "b3b759d133670eb30daa642b9d685d3522b7d418d685f946d499d2234ab56a98"
)
CLARIFICATION_V3_BYTES = 3_508
CLARIFICATION_V3_SHA256 = (
    "cbfc0557746640eb7d69adfb2696e8525243ac7e75cc053614381a2600745c59"
)
CLARIFICATION_V4_BYTES = 1_434
CLARIFICATION_V4_SHA256 = (
    "818bc738a7f813b1cdf5acc039b29e06b259098585f29bc98224e8f02ff70477"
)
CLARIFICATION_V5_BYTES = 1_518
CLARIFICATION_V5_SHA256 = "900a77d70f63c05bd86704615d91b112750c5acb72e6f816457e7dcad2c37e24"
SEAL_V6_BYTES = 3_256
SEAL_V6_SHA256 = "e4f9735dc783758960c321bc954701e7a9b9e30dd9a2a40d39cf07ecd94556dc"
OFFICIAL_MANIFEST_BYTES = 92_936
OFFICIAL_MANIFEST_SHA256 = (
    "223b7b55e66d1dc8ecd522cde77c992934bde0897fbf9b7745b8bf18a361a7d1"
)
TASK8A_CLEAN_REPORT_SHA256 = (
    "00f04888de9a98f855a5ff4b5f008d0eba428bb6a54fbccc8c256a824c7f7bb3"
)
TASK8A_CLEAN_EXPECTATIONS_SHA256 = (
    "9339672f2da52335f061b314e35618b2a730d6fbfae8a9d8856d72a75b0dcc28"
)

ROOT_SEED = 42
DRAWS = 50
FEATURE_FAMILIES = (
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

_AUTHORITY_DOCUMENTS = (
    (
        "parent_plan",
        PLAN_PATH.as_posix(),
        PLAN_BYTES,
        PLAN_SHA256,
    ),
    (
        "clarification_v1",
        CLARIFICATION_PATH.as_posix(),
        CLARIFICATION_BYTES,
        CLARIFICATION_SHA256,
    ),
    (
        "clarification_v2",
        CLARIFICATION_V2_PATH.as_posix(),
        CLARIFICATION_V2_BYTES,
        CLARIFICATION_V2_SHA256,
    ),
    (
        "clarification_v3",
        CLARIFICATION_V3_PATH.as_posix(),
        CLARIFICATION_V3_BYTES,
        CLARIFICATION_V3_SHA256,
    ),
    (
        "clarification_v4",
        CLARIFICATION_V4_PATH.as_posix(),
        CLARIFICATION_V4_BYTES,
        CLARIFICATION_V4_SHA256,
    ),
    (
        "clarification_v5",
        CLARIFICATION_V5_PATH.as_posix(),
        CLARIFICATION_V5_BYTES,
        CLARIFICATION_V5_SHA256,
    ),
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be an exact lowercase SHA-256 string")
    return value


def _require_exact_keys(value: object, keys: tuple[str, ...], label: str) -> dict:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    if tuple(value) != keys:
        _fail(f"{label} fields are missing, extra, or reordered")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"strict JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    _fail(f"strict JSON contains nonfinite value {value}")


def _parse_strict_json(raw: bytes, label: str) -> dict:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not strict UTF-8 JSON: {exc}")
    if type(document) is not dict:
        raise TypeError(f"{label} must contain an exact JSON object")
    return document


def _is_alias_component(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        _fail(f"cannot inspect authority path component {path}: {exc}")
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _require_canonical_worktree_root(value: Path) -> Path:
    if type(value) is not _PATH_TYPE:
        raise TypeError("worktree_root must be an exact pathlib Path")
    if not value.is_absolute():
        _fail("worktree_root must be absolute")
    try:
        resolved = value.resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve worktree_root: {exc}")
    if value != resolved or not value.is_dir():
        _fail("worktree_root must be an existing canonical directory")
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current /= part
        if _is_alias_component(current):
            _fail("worktree_root cannot contain a symlink or reparse component")
    if not (value / ".git").exists():
        _fail("worktree_root must be a Git worktree")
    return value


def _require_literal_file(root: Path, relative: PurePosixPath, label: str) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{label} does not use a literal repository-relative path")
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current /= part
        if _is_alias_component(current):
            _fail(f"{label} contains a symlink or reparse component")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        _fail(f"cannot resolve {label}: {exc}")
    if resolved != candidate or root not in resolved.parents or not candidate.is_file():
        _fail(f"{label} is aliased or outside the worktree")
    return candidate


def _read_pinned_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> bytes:
    try:
        with path.open("rb") as stream:
            raw = stream.read()
    except OSError as exc:
        _fail(f"cannot read {label}: {exc}")
    if len(raw) != expected_bytes:
        _fail(f"{label} byte count is stale")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail(f"{label} SHA-256 is stale")
    return raw


def _validate_seal_v6(document: dict) -> None:
    _require_exact_keys(
        document,
        (
            "schema_name",
            "status",
            "frozen_at",
            "supersedes_seal",
            "authority_documents",
            "inherited_official_authority",
            "ordered_primary_families",
            "fixed_multiplicity_policy",
            "valid_record_matching",
            "result_exposure_at_freeze",
            "implementation_started",
            "tiny_run_started_but_blocked_before_new_mask_generation",
            "canonical_experiment_started",
            "authorization",
            "mutation_policy",
        ),
        "final preregistration seal v6",
    )
    if document["schema_name"] != "structured-missingness-preregistration-seal-v6":
        _fail("final preregistration seal schema is not v6")
    if document["status"] != "implementation_authority_before_new_mask_results":
        _fail("final preregistration seal is not implementation authority")
    if document["frozen_at"] != "2026-08-10T22:30:00+09:00":
        _fail("final preregistration seal freeze timestamp is stale")

    superseded = _require_exact_keys(
        document["supersedes_seal"],
        ("path", "bytes", "sha256"),
        "superseded seal pin",
    )
    if tuple(type(superseded[field]) for field in superseded) != (str, int, str):
        raise TypeError("superseded seal pin uses non-exact scalar types")
    _require_sha256(superseded["sha256"], "superseded seal SHA-256")
    if superseded != {
        "path": ".superpowers/sdd/task-12-structured-missingness-preregistration-seal-v5.json",
        "bytes": 3_117,
        "sha256": "60dda8255cce654c3e71480b90a5f20d0aa0f89d52099ee52fa47f5dd18244f0",
    }:
        _fail("superseded seal pin is stale or crosswired")
    if type(document["implementation_started"]) is not bool:
        raise TypeError("final preregistration seal implementation_started must be bool")
    if document["implementation_started"] is not True:
        _fail("final preregistration seal must record implementation_started true")
    if document["tiny_run_started_but_blocked_before_new_mask_generation"] is not True:
        _fail("blocked pre-result tiny attempt must be recorded")
    if type(document["canonical_experiment_started"]) is not bool:
        raise TypeError(
            "final preregistration seal canonical_experiment_started must be bool"
        )
    if document["canonical_experiment_started"] is not False:
        _fail("canonical experiment must not have started")

    authorities = document["authority_documents"]
    if type(authorities) is not list or len(authorities) != len(_AUTHORITY_DOCUMENTS):
        _fail("final preregistration seal must contain exactly six authority documents")
    for raw_entry, expected in zip(authorities, _AUTHORITY_DOCUMENTS, strict=True):
        entry = _require_exact_keys(
            raw_entry,
            ("role", "path", "bytes", "sha256"),
            "authority document pin",
        )
        actual = (entry["role"], entry["path"], entry["bytes"], entry["sha256"])
        if tuple(type(item) for item in actual) != (str, str, int, str):
            raise TypeError("authority document pin uses non-exact scalar types")
        _require_sha256(entry["sha256"], "authority document SHA-256")
        if actual != expected:
            _fail("authority document pin is stale, reordered, or crosswired")

    inherited = _require_exact_keys(
        document["inherited_official_authority"],
        ("path", "bytes", "sha256", "required_status"),
        "inherited official authority",
    )
    inherited_actual = (
        inherited["path"],
        inherited["bytes"],
        inherited["sha256"],
        inherited["required_status"],
    )
    if tuple(type(item) for item in inherited_actual) != (str, int, str, str):
        raise TypeError("inherited official authority uses non-exact scalar types")
    _require_sha256(inherited["sha256"], "official manifest SHA-256")
    if inherited_actual != (
        OFFICIAL_MANIFEST_PATH.as_posix(),
        OFFICIAL_MANIFEST_BYTES,
        OFFICIAL_MANIFEST_SHA256,
        "complete_streamed_exact50",
    ):
        _fail("inherited official authority is stale or crosswired")

    ordered_families = document["ordered_primary_families"]
    if type(ordered_families) is not list or any(
        type(item) is not str for item in ordered_families
    ):
        raise TypeError("ordered primary families must be an exact list of strings")
    if tuple(ordered_families) != tuple(name for name, _ in FEATURE_FAMILIES):
        _fail("ordered primary families are stale or reordered")
    multiplicity = _require_exact_keys(
        document["fixed_multiplicity_policy"],
        ("method", "family_size", "alpha", "ineligible_family_raw_p"),
        "fixed multiplicity policy",
    )
    if tuple(type(multiplicity[field]) for field in multiplicity) != (
        str,
        int,
        float,
        float,
    ):
        raise TypeError("fixed multiplicity policy uses non-exact scalar types")
    if multiplicity != {
        "method": "holm",
        "family_size": 4,
        "alpha": 0.05,
        "ineligible_family_raw_p": 1.0,
    }:
        _fail("fixed four-family Holm policy is stale")

    matching = _require_exact_keys(
        document["valid_record_matching"],
        (
            "target_count_name",
            "historical_digest_is_provenance_only",
            "current_equivalence_requires_count_triplet",
            "prepared_source_alias",
            "subject_identity_alias",
            "record_identity_alias",
            "official_deleted_count_is_valid_count",
        ),
        "valid-record matching authority",
    )
    if tuple(type(matching[field]) for field in matching) != (
        str,
        bool,
        bool,
        str,
        str,
        str,
        bool,
    ):
        raise TypeError("valid-record matching authority uses non-exact scalar types")
    if matching != {
        "target_count_name": "K_valid",
        "historical_digest_is_provenance_only": True,
        "current_equivalence_requires_count_triplet": True,
        "prepared_source_alias": "PreparedPrimitiveReplay.source_record_digest",
        "subject_identity_alias": "PreparedPrimitiveReplay.calendar_identity.subject_id",
        "record_identity_alias": "PreparedPhaseRecord.content_digest",
        "official_deleted_count_is_valid_count": False,
    }:
        _fail("valid-record matching authority is stale or crosswired")

    exposure = _require_exact_keys(
        document["result_exposure_at_freeze"],
        (
            "existing_etri_g2_results_seen",
            "new_mask_results_seen",
            "new_downstream_stability_results_seen",
            "external_transfer_results_seen",
        ),
        "result exposure state",
    )
    if any(type(value) is not bool for value in exposure.values()):
        raise TypeError("result exposure state must use exact bool values")
    if exposure != {
        "existing_etri_g2_results_seen": True,
        "new_mask_results_seen": False,
        "new_downstream_stability_results_seen": False,
        "external_transfer_results_seen": False,
    }:
        _fail("new-mask, downstream, and external results must remain unopened")
    authorization = _require_exact_keys(
        document["authorization"],
        (
            "tasks_0_through_5_authorized",
            "tiny_dry_run_authorized",
            "canonical_etri_run_authorized",
            "canonical_release_requires_fresh_clean_review",
        ),
        "implementation authorization",
    )
    if any(type(value) is not bool for value in authorization.values()):
        raise TypeError("implementation authorization must use exact bool values")
    if authorization != {
        "tasks_0_through_5_authorized": True,
        "tiny_dry_run_authorized": True,
        "canonical_etri_run_authorized": False,
        "canonical_release_requires_fresh_clean_review": True,
    }:
        _fail("implementation or canonical-run authorization is crosswired")

    mutation = _require_exact_keys(
        document["mutation_policy"],
        (
            "authority_edit_invalidates_seal",
            "release_seal_requires_exact_code_and_review_pins",
            "post_result_primary_rule_change_prohibited",
        ),
        "mutation policy",
    )
    if any(type(value) is not bool for value in mutation.values()):
        raise TypeError("mutation policy must use exact bool values")
    if any(value is not True for value in mutation.values()):
        _fail("mutation policy must remain fail-closed")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _authority_content_digest(
    *,
    worktree_root: Path,
    plan_sha256: str,
    clarification_sha256: str,
    clarification_v2_sha256: str,
    clarification_v3_sha256: str,
    clarification_v4_sha256: str,
    clarification_v5_sha256: str,
    seal_sha256: str,
    official_manifest_sha256: str,
    root_seed: int,
    draws: int,
    feature_families: tuple[tuple[str, tuple[str, ...]], ...],
) -> str:
    payload = [
        "structured-missingness-preregistration-authority-v1",
        os.fspath(worktree_root),
        plan_sha256,
        clarification_sha256,
        clarification_v2_sha256,
        clarification_v3_sha256,
        clarification_v4_sha256,
        clarification_v5_sha256,
        seal_sha256,
        official_manifest_sha256,
        root_seed,
        draws,
        [[family, list(primitives)] for family, primitives in feature_families],
    ]
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class StressPreregistrationAuthority:
    worktree_root: Path
    plan_sha256: str
    clarification_sha256: str
    clarification_v2_sha256: str
    clarification_v3_sha256: str
    clarification_v4_sha256: str
    clarification_v5_sha256: str
    seal_sha256: str
    official_manifest_sha256: str
    root_seed: int
    draws: int
    feature_families: tuple[tuple[str, tuple[str, ...]], ...]
    content_digest: str

    def __post_init__(self) -> None:
        root = _require_canonical_worktree_root(self.worktree_root)
        if self.plan_sha256 != PLAN_SHA256:
            _fail("plan SHA-256 is not the frozen value")
        if self.clarification_sha256 != CLARIFICATION_SHA256:
            _fail("clarification v1 SHA-256 is not the frozen value")
        if self.clarification_v2_sha256 != CLARIFICATION_V2_SHA256:
            _fail("clarification v2 SHA-256 is not the frozen value")
        if self.clarification_v3_sha256 != CLARIFICATION_V3_SHA256:
            _fail("clarification v3 SHA-256 is not the frozen value")
        if self.clarification_v4_sha256 != CLARIFICATION_V4_SHA256:
            _fail("clarification v4 SHA-256 is not the frozen value")
        if self.clarification_v5_sha256 != CLARIFICATION_V5_SHA256:
            _fail("clarification v5 SHA-256 is not the frozen value")
        if self.seal_sha256 != SEAL_V6_SHA256:
            _fail("seal v6 SHA-256 is not the frozen value")
        if self.official_manifest_sha256 != OFFICIAL_MANIFEST_SHA256:
            _fail("official manifest SHA-256 is not the frozen value")
        if type(self.root_seed) is not int or self.root_seed != ROOT_SEED:
            raise TypeError("root_seed must be the exact integer 42")
        if type(self.draws) is not int or self.draws != DRAWS:
            raise TypeError("draws must be the exact integer 50")
        if type(self.feature_families) is not tuple or self.feature_families != FEATURE_FAMILIES:
            _fail("feature_families must equal the frozen ordered mapping")
        expected_digest = _authority_content_digest(
            worktree_root=root,
            plan_sha256=self.plan_sha256,
            clarification_sha256=self.clarification_sha256,
            clarification_v2_sha256=self.clarification_v2_sha256,
            clarification_v3_sha256=self.clarification_v3_sha256,
            clarification_v4_sha256=self.clarification_v4_sha256,
            clarification_v5_sha256=self.clarification_v5_sha256,
            seal_sha256=self.seal_sha256,
            official_manifest_sha256=self.official_manifest_sha256,
            root_seed=self.root_seed,
            draws=self.draws,
            feature_families=self.feature_families,
        )
        _require_sha256(self.content_digest, "authority content_digest")
        if self.content_digest != expected_digest:
            _fail("authority content_digest is stale")


@dataclass(frozen=True)
class StressExecutionAuthorityGraph:
    preregistration: StressPreregistrationAuthority
    official_g2: phaseguard_authority.OfficialG2Authority
    calibration_input: phaseguard_authority.CalibrationInputAuthority


def _data_root_for_worktree(worktree_root: Path) -> Path:
    checkout_root = (
        worktree_root.parent.parent
        if worktree_root.parent.name == ".worktrees"
        else worktree_root
    )
    return checkout_root / "ETRI_Human_AI_v2" / "data"


def load_stress_execution_authority_graph(
    worktree_root: Path,
) -> StressExecutionAuthorityGraph:
    """Load the preregistration, official G2, and calibration graph exactly once."""

    root = _require_canonical_worktree_root(worktree_root)
    pinned_files = (
        (PLAN_PATH, PLAN_BYTES, PLAN_SHA256, "parent analysis plan"),
        (
            CLARIFICATION_PATH,
            CLARIFICATION_BYTES,
            CLARIFICATION_SHA256,
            "preregistration clarification v1",
        ),
        (
            CLARIFICATION_V2_PATH,
            CLARIFICATION_V2_BYTES,
            CLARIFICATION_V2_SHA256,
            "preregistration clarification v2",
        ),
        (
            CLARIFICATION_V3_PATH,
            CLARIFICATION_V3_BYTES,
            CLARIFICATION_V3_SHA256,
            "preregistration clarification v3",
        ),
        (
            CLARIFICATION_V4_PATH,
            CLARIFICATION_V4_BYTES,
            CLARIFICATION_V4_SHA256,
            "preregistration clarification v4",
        ),
        (
            CLARIFICATION_V5_PATH,
            CLARIFICATION_V5_BYTES,
            CLARIFICATION_V5_SHA256,
            "preregistration clarification v5",
        ),
        (SEAL_V6_PATH, SEAL_V6_BYTES, SEAL_V6_SHA256, "final seal v6"),
    )
    identities: set[tuple[int, int]] = set()
    raw_by_path: dict[PurePosixPath, bytes] = {}
    for relative, expected_bytes, expected_sha256, label in pinned_files:
        path = _require_literal_file(root, relative, label)
        metadata = path.stat()
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if identity in identities:
            _fail("authority files contain a same-file or hardlink alias")
        identities.add(identity)
        raw_by_path[relative] = _read_pinned_file(
            path,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            label=label,
        )

    seal_document = _parse_strict_json(raw_by_path[SEAL_V6_PATH], "final seal v6")
    _validate_seal_v6(seal_document)

    official_manifest = root.joinpath(*OFFICIAL_MANIFEST_PATH.parts)
    official = phaseguard_authority.load_official_g2_authority(
        official_manifest,
        OFFICIAL_MANIFEST_SHA256,
        root.joinpath(*TASK8A_CLEAN_REPORT_PATH.parts),
        TASK8A_CLEAN_REPORT_SHA256,
        root.joinpath(*TASK8A_CLEAN_EXPECTATIONS_PATH.parts),
        TASK8A_CLEAN_EXPECTATIONS_SHA256,
    )
    data_root = _data_root_for_worktree(root)
    calibration = phaseguard_authority.load_calibration_input_authority(
        official,
        raw_root=(data_root / "raw" / "data_vvs").resolve(),
        canonical_root=(data_root / "canonical").resolve(),
        primitive_table_path=root.joinpath(*CALIBRATION_PRIMITIVES_PATH.parts),
        frozen_population_baselines_path=root.joinpath(*CALIBRATION_BASELINES_PATH.parts),
    )

    digest = _authority_content_digest(
        worktree_root=root,
        plan_sha256=PLAN_SHA256,
        clarification_sha256=CLARIFICATION_SHA256,
        clarification_v2_sha256=CLARIFICATION_V2_SHA256,
        clarification_v3_sha256=CLARIFICATION_V3_SHA256,
        clarification_v4_sha256=CLARIFICATION_V4_SHA256,
        clarification_v5_sha256=CLARIFICATION_V5_SHA256,
        seal_sha256=SEAL_V6_SHA256,
        official_manifest_sha256=OFFICIAL_MANIFEST_SHA256,
        root_seed=ROOT_SEED,
        draws=DRAWS,
        feature_families=FEATURE_FAMILIES,
    )
    preregistration = StressPreregistrationAuthority(
        worktree_root=root,
        plan_sha256=PLAN_SHA256,
        clarification_sha256=CLARIFICATION_SHA256,
        clarification_v2_sha256=CLARIFICATION_V2_SHA256,
        clarification_v3_sha256=CLARIFICATION_V3_SHA256,
        clarification_v4_sha256=CLARIFICATION_V4_SHA256,
        clarification_v5_sha256=CLARIFICATION_V5_SHA256,
        seal_sha256=SEAL_V6_SHA256,
        official_manifest_sha256=OFFICIAL_MANIFEST_SHA256,
        root_seed=ROOT_SEED,
        draws=DRAWS,
        feature_families=FEATURE_FAMILIES,
        content_digest=digest,
    )
    return StressExecutionAuthorityGraph(
        preregistration=preregistration,
        official_g2=official,
        calibration_input=calibration,
    )


def load_stress_preregistration_authority(
    worktree_root: Path,
) -> StressPreregistrationAuthority:
    """Load and return the frozen Tasks 0–5 preregistration authority."""
    return load_stress_execution_authority_graph(worktree_root).preregistration


def _stress_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_intervals(
    value: object,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    if type(value) is not tuple or not value:
        raise TypeError("intervals must be a nonempty exact tuple")
    parsed: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("interval entries must be exact pairs")
        start, end = item
        if type(start) is not pd.Timestamp or type(end) is not pd.Timestamp:
            raise TypeError("interval endpoints must be exact pandas Timestamps")
        if start.tz is not None or end.tz is not None or start >= end:
            raise ValueError("interval endpoints are invalid")
        parsed.append((start, end))
    if parsed != sorted(parsed) or any(
        current[0] < previous[1]
        for previous, current in zip(parsed, parsed[1:])
    ):
        raise ValueError("intervals must be canonical and nonoverlapping")
    return tuple(parsed)


def official_record_deletion_digest(
    prepared: primitives.PreparedPrimitiveReplay,
    intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
) -> str:
    """Reconstruct the inherited audit digest using clarification-v3 aliases."""
    if type(prepared) is not primitives.PreparedPrimitiveReplay:
        raise TypeError("prepared must be an exact PreparedPrimitiveReplay")
    view = primitives.export_prepared_phase_view(prepared)
    exact = _exact_intervals(intervals)
    payload = [
        prepared.source_record_digest,
        view.timestamp_semantics,
        [[start.isoformat(), end.isoformat()] for start, end in exact],
    ]
    return _stress_digest(payload)


@dataclass(frozen=True)
class Contiguous20Crosslink:
    subject_id: str
    sensor_day_id: int
    primitive: str
    draw: int
    deletion_audit_key: str
    official_deleted_count: int
    official_retained_count: int
    official_record_deletion_digest: str
    intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    reference_value: float
    reference_status: str
    standardizer: float
    standardizer_source: str
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("subject_id", "primitive", "reference_status", "standardizer_source"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise TypeError(f"{name} must be an exact nonempty string")
        for name in ("sensor_day_id", "draw", "official_deleted_count", "official_retained_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact nonnegative integer")
        for name in ("deletion_audit_key", "official_record_deletion_digest"):
            _require_sha256(getattr(self, name), name)
        exact = _exact_intervals(self.intervals)
        if type(self.reference_value) is not float or type(self.standardizer) is not float:
            raise TypeError("reference value and standardizer must be exact floats")
        payload = [
            "contiguous20-crosslink-v1",
            self.subject_id,
            self.sensor_day_id,
            self.primitive,
            self.draw,
            self.deletion_audit_key,
            self.official_deleted_count,
            self.official_retained_count,
            self.official_record_deletion_digest,
            [[start.isoformat(), end.isoformat()] for start, end in exact],
            self.reference_value.hex(),
            self.reference_status,
            self.standardizer.hex(),
            self.standardizer_source,
        ]
        object.__setattr__(self, "content_digest", _stress_digest(payload))


@dataclass(frozen=True)
class MatchedDeletionTarget:
    subject_id: str
    sensor_day_id: int
    primitive: str
    draw: int
    target_deleted_count: int
    official_deleted_count: int
    contiguous20_deletion_audit_key: str
    prepared_source_content_digest: str
    current_record_deletion_digest: str
    contiguous_valid_record_identity_digests: tuple[str, ...]
    reference_value: float
    standardizer: float
    standardizer_source: str
    high_support_sensitivity: bool
    eligibility_status: str
    reason: str
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.eligibility_status not in ("eligible", "ineligible"):
            raise ValueError("eligibility status is invalid")
        payload = [
            "matched-deletion-target-v1",
            self.subject_id,
            self.sensor_day_id,
            self.primitive,
            self.draw,
            self.target_deleted_count,
            self.official_deleted_count,
            self.contiguous20_deletion_audit_key,
            self.prepared_source_content_digest,
            self.current_record_deletion_digest,
            list(self.contiguous_valid_record_identity_digests),
            self.reference_value.hex() if math.isfinite(self.reference_value) else "nan",
            self.standardizer.hex() if math.isfinite(self.standardizer) else "nan",
            self.standardizer_source,
            self.high_support_sensitivity,
            self.eligibility_status,
            self.reason,
        ]
        object.__setattr__(self, "content_digest", _stress_digest(payload))


def _record_is_deleted(
    record: primitives.PreparedPhaseRecord,
    timestamp_semantics: str,
    intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
) -> bool:
    if timestamp_semantics == "instant":
        return any(start <= record.source_timestamp < end for start, end in intervals)
    return any(record.support_start < end and record.support_end > start for start, end in intervals)


def derive_matched_deletion_target(
    prepared: primitives.PreparedPrimitiveReplay,
    crosslink: Contiguous20Crosslink,
) -> MatchedDeletionTarget:
    """Derive the valid-record K without reinterpreting the inherited audit count."""
    if type(crosslink) is not Contiguous20Crosslink:
        raise TypeError("crosslink must be an exact Contiguous20Crosslink")
    view = primitives.export_prepared_phase_view(prepared)
    identity = view.calendar_identity
    if (
        crosslink.subject_id != identity.subject_id
        or crosslink.sensor_day_id != identity.sensor_day_id
        or crosslink.primitive != view.primitive_name
    ):
        raise ValueError("contiguous20 crosslink is crosswired")
    current_record_digest = official_record_deletion_digest(prepared, crosslink.intervals)
    current_replay = primitives.recompute_prepared_primitive_batch(
        prepared, (crosslink.intervals,)
    )[0]
    if (
        current_replay.original_count != crosslink.official_deleted_count
        + crosslink.official_retained_count
        or current_replay.deleted_count != crosslink.official_deleted_count
        or current_replay.retained_count != crosslink.official_retained_count
    ):
        raise ValueError("official/current deletion count triplet does not match")
    valid_records = tuple(record for record in view.records if record.valid)
    deleted = tuple(
        record.content_digest
        for record in valid_records
        if _record_is_deleted(record, view.timestamp_semantics, crosslink.intervals)
    )
    observed_units = float(sum(record.support_units for record in valid_records))
    support_fraction = observed_units / view.expected_epochs if view.expected_epochs > 0 else 0.0
    reason = "eligible"
    eligible = True
    if crosslink.reference_status != "observed" or not math.isfinite(crosslink.reference_value):
        eligible, reason = False, "invalid_reference"
    elif view.expected_epochs <= 0:
        eligible, reason = False, "zero_expected_units"
    elif not valid_records:
        eligible, reason = False, "no_valid_source_contribution"
    elif not deleted:
        eligible, reason = False, "zero_valid_deleted_count"
    elif len(deleted) > len(valid_records):
        eligible, reason = False, "infeasible_valid_deleted_count"
    elif not math.isfinite(crosslink.standardizer) or crosslink.standardizer <= 0.0:
        eligible, reason = False, "invalid_standardizer"
    elif view.operation == "evening_p90" and (
        sum(record.positive_event_count for record in valid_records) < 3
        or sum(record.event_mass for record in valid_records) <= 0.0
    ):
        eligible, reason = False, "insufficient_timing_events"
    elif view.operation in ("additive_load", "exposure_mean") and not any(
        math.isfinite(record.transformed_numerator)
        and record.transformed_numerator >= 0.0
        for record in valid_records
    ):
        eligible, reason = False, "no_finite_nonnegative_contribution"
    return MatchedDeletionTarget(
        subject_id=identity.subject_id,
        sensor_day_id=identity.sensor_day_id,
        primitive=view.primitive_name,
        draw=crosslink.draw,
        target_deleted_count=len(deleted),
        official_deleted_count=crosslink.official_deleted_count,
        contiguous20_deletion_audit_key=crosslink.deletion_audit_key,
        prepared_source_content_digest=prepared.source_record_digest,
        current_record_deletion_digest=current_record_digest,
        contiguous_valid_record_identity_digests=deleted,
        reference_value=crosslink.reference_value,
        standardizer=crosslink.standardizer,
        standardizer_source=crosslink.standardizer_source,
        high_support_sensitivity=support_fraction >= 0.80,
        eligibility_status="eligible" if eligible else "ineligible",
        reason=reason,
    )


def build_matched_target_ledger(
    targets: list[MatchedDeletionTarget],
    *,
    expected_keys: tuple[tuple[str, int, str, int], ...],
) -> pd.DataFrame:
    """Publish an omission-detecting target ledger in canonical key order."""
    if type(targets) is not list or not all(type(item) is MatchedDeletionTarget for item in targets):
        raise TypeError("targets must be a list of exact MatchedDeletionTarget values")
    if type(expected_keys) is not tuple:
        raise TypeError("expected_keys must be an exact tuple")
    actual = tuple((item.subject_id, item.sensor_day_id, item.primitive, item.draw) for item in targets)
    if len(set(actual)) != len(actual) or set(actual) != set(expected_keys):
        raise ValueError("target ledger topology is duplicate, missing, or extra")
    ordered = sorted(targets, key=lambda item: (item.subject_id, item.primitive, item.sensor_day_id, item.draw))
    return pd.DataFrame(
        [
            {
                "subject_id": item.subject_id,
                "sensor_day_id": item.sensor_day_id,
                "primitive": item.primitive,
                "draw": item.draw,
                "target_deleted_count": item.target_deleted_count,
                "official_deleted_count": item.official_deleted_count,
                "contiguous20_deletion_audit_key": item.contiguous20_deletion_audit_key,
                "prepared_source_content_digest": item.prepared_source_content_digest,
                "contiguous_valid_record_identity_digests": item.contiguous_valid_record_identity_digests,
                "reference_value": item.reference_value,
                "standardizer": item.standardizer,
                "standardizer_source": item.standardizer_source,
                "high_support_sensitivity": item.high_support_sensitivity,
                "eligibility_status": item.eligibility_status,
                "reason": item.reason,
                "content_digest": item.content_digest,
            }
            for item in ordered
        ]
    )
