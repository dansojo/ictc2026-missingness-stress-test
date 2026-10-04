from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("external_readiness_prepare", HERE / "prepare_external_data.py")
_PREPARE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_PREPARE)

ReadinessError = _PREPARE.ReadinessError


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_fields = ["dataset", "relative_path", "bytes", "sha256"]
    if not rows or list(rows[0]) != expected_fields:
        raise ReadinessError("invalid file manifest schema")
    for row in rows:
        row["bytes"] = int(row["bytes"])
    return rows


def _resolve_relative(project_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReadinessError("dataset path must be project-relative and traversal-free")
    target = project_root / relative
    try:
        target.resolve(strict=True).relative_to(project_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ReadinessError("dataset path escapes or is missing") from exc
    return target


def verify(project_root: Path, readiness_root: Path) -> dict:
    project = _PREPARE._require_source_directory(project_root)
    locations = _load_json(readiness_root / "dataset_locations.json")
    schema_expected = _load_json(readiness_root / "schema_manifest.json")
    manifest_expected = _load_manifest(readiness_root / "file_manifest.csv")
    if locations.get("schema_version") != _PREPARE.SCHEMA_VERSION:
        raise ReadinessError("location schema version mismatch")
    expected_roles = {"ExtraSensory", "StudentLife"}
    if set(locations.get("datasets", {})) != expected_roles:
        raise ReadinessError("dataset role set mismatch")
    extra = _resolve_relative(project, locations["datasets"]["ExtraSensory"]["relative_path"])
    student = _resolve_relative(project, locations["datasets"]["StudentLife"]["relative_path"])
    current_records = sorted(
        (*_PREPARE.inventory_tree(extra, "ExtraSensory"), *_PREPARE.inventory_tree(student, "StudentLife")),
        key=lambda row: (row["dataset"], row["relative_path"]),
    )
    if current_records != manifest_expected:
        raise ReadinessError("file inventory/hash mismatch")
    schema_current = {
        "schema_version": _PREPARE.SCHEMA_VERSION,
        "ExtraSensory": _PREPARE.inspect_extrasensory(extra),
        "StudentLife": _PREPARE.inspect_studentlife(student),
    }
    if schema_current != schema_expected:
        raise ReadinessError("dataset structural schema mismatch")
    return {"status": "READY", "datasets": 2, "files": len(current_records)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--readiness-root", type=Path)
    args = parser.parse_args()
    readiness = args.readiness_root or args.project_root / "02_experiments" / "external_data" / "readiness"
    print(json.dumps(verify(args.project_root, readiness), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
