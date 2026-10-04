from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterable


SCHEMA_VERSION = "external-data-readiness-v1"
EXTRASENSORY_RELATIVE = Path("ExtraSensory.per_uuid_features_labels")
STUDENTLIFE_RELATIVE = Path("StudentLife") / "dataset"
EXTRA_UUID_RE = re.compile(
    r"^[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}\.features_labels\.csv\.gz$"
)
STUDENT_ID_RE = re.compile(r"(?:^|_)(u\d{2})(?:\.|_|$)")
STUDENT_REQUIRED_DIRS = (
    "sensing/activity",
    "sensing/conversation",
    "sensing/dark",
    "sensing/gps",
    "sensing/phonecharge",
    "sensing/phonelock",
    "sensing/wifi",
    "EMA",
    "survey",
    "education",
)


class ReadinessError(ValueError):
    pass


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReadinessError(f"cannot stat path: {path}") from exc
    attrs = getattr(info, "st_file_attributes", 0)
    return path.is_symlink() or bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _require_source_directory(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ReadinessError("source root must be an absolute pathlib.Path")
    if not path.is_dir() or _is_reparse(path):
        raise ReadinessError(f"source root must be a real directory: {path}")
    return path.resolve(strict=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_tree(root: Path, dataset: str) -> tuple[dict, ...]:
    canonical = _require_source_directory(root)
    records: list[dict] = []
    for path in sorted(canonical.rglob("*"), key=lambda item: item.relative_to(canonical).as_posix()):
        if _is_reparse(path):
            raise ReadinessError(f"symlink/reparse input is forbidden: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReadinessError(f"non-regular input is forbidden: {path}")
        records.append(
            {
                "dataset": dataset,
                "relative_path": path.relative_to(canonical).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ReadinessError(f"dataset tree is empty: {canonical}")
    return tuple(records)


def inspect_extrasensory(root: Path) -> dict:
    canonical = _require_source_directory(root)
    files = sorted(canonical.glob("*.csv.gz"), key=lambda item: item.name)
    all_files = sorted(path for path in canonical.iterdir() if path.is_file())
    if not files or files != all_files:
        raise ReadinessError("ExtraSensory root must contain only participant csv.gz files")
    if any(not EXTRA_UUID_RE.fullmatch(path.name) for path in files):
        raise ReadinessError("ExtraSensory participant filename is not a canonical UUID role")

    reference_header: tuple[str, ...] | None = None
    total_rows = 0
    per_user_rows: list[int] = []
    gaps_gt_60 = 0
    global_min: int | None = None
    global_max: int | None = None
    for path in files:
        try:
            with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = tuple(next(reader))
                if not header or header[0] != "timestamp" or header[-1] != "label_source":
                    raise ReadinessError(f"invalid ExtraSensory column boundary: {path.name}")
                if reference_header is None:
                    reference_header = header
                elif header != reference_header:
                    raise ReadinessError(f"ExtraSensory header drift: {path.name}")
                previous: int | None = None
                rows = 0
                for row in reader:
                    if len(row) != len(header):
                        raise ReadinessError(f"ExtraSensory row width drift: {path.name}")
                    value = float(row[0])
                    if not value.is_integer():
                        raise ReadinessError(f"non-integral ExtraSensory timestamp: {path.name}")
                    timestamp = int(value)
                    if previous is not None:
                        if timestamp < previous:
                            raise ReadinessError(f"non-monotonic ExtraSensory timestamp: {path.name}")
                        if timestamp - previous > 60:
                            gaps_gt_60 += 1
                    previous = timestamp
                    global_min = timestamp if global_min is None else min(global_min, timestamp)
                    global_max = timestamp if global_max is None else max(global_max, timestamp)
                    rows += 1
        except (OSError, EOFError, UnicodeError, csv.Error) as exc:
            raise ReadinessError(f"unreadable ExtraSensory gzip CSV: {path.name}") from exc
        if rows == 0:
            raise ReadinessError(f"empty ExtraSensory participant file: {path.name}")
        per_user_rows.append(rows)
        total_rows += rows

    assert reference_header is not None
    features = [name for name in reference_header if name not in {"timestamp", "label_source"} and not name.startswith("label:")]
    labels = [name for name in reference_header if name.startswith("label:")]
    return {
        "participants": len(files),
        "participant_ids": [path.name.split(".features_labels", 1)[0] for path in files],
        "rows": total_rows,
        "rows_min": min(per_user_rows),
        "rows_max": max(per_user_rows),
        "columns": list(reference_header),
        "feature_columns": len(features),
        "label_columns": len(labels),
        "timestamp_min": global_min,
        "timestamp_max": global_max,
        "gaps_gt_60_seconds": gaps_gt_60,
        "label_missing_semantics": "unknown_not_negative",
        "episode_gap_rule": "gap_gt_60_seconds_breaks_episode",
    }


def inspect_studentlife(root: Path) -> dict:
    canonical = _require_source_directory(root)
    missing = [relative for relative in STUDENT_REQUIRED_DIRS if not (canonical / relative).is_dir()]
    if missing:
        raise ReadinessError(f"StudentLife required directories missing: {missing}")
    participants: set[str] = set()
    extension_counts: dict[str, int] = {}
    files = [path for path in canonical.rglob("*") if path.is_file()]
    for path in files:
        if _is_reparse(path):
            raise ReadinessError(f"StudentLife symlink/reparse input is forbidden: {path}")
        extension_counts[path.suffix.lower() or "<none>"] = extension_counts.get(path.suffix.lower() or "<none>", 0) + 1
        match = STUDENT_ID_RE.search(path.name)
        if match:
            participants.add(match.group(1))
    if not participants:
        raise ReadinessError("StudentLife participant IDs were not found")
    return {
        "participants": sorted(participants),
        "participant_count": len(participants),
        "files": len(files),
        "required_directories": list(STUDENT_REQUIRED_DIRS),
        "extension_counts": dict(sorted(extension_counts.items())),
        "target_content_policy": "inventory_only_do_not_summarize",
        "physical_distribution_source": "Kaggle mirror; cite original StudentLife publication",
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_manifest(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "relative_path", "bytes", "sha256"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def prepare(project_root: Path, output_root: Path) -> dict:
    project = _require_source_directory(project_root)
    if not isinstance(output_root, Path) or not output_root.is_absolute():
        raise ReadinessError("output root must be an absolute pathlib.Path")
    try:
        output_root.relative_to(project)
    except ValueError as exc:
        raise ReadinessError("output root must stay within the paper project") from exc
    extra = project / EXTRASENSORY_RELATIVE
    student = project / STUDENTLIFE_RELATIVE
    extra_records = inventory_tree(extra, "ExtraSensory")
    student_records = inventory_tree(student, "StudentLife")
    schema = {
        "schema_version": SCHEMA_VERSION,
        "ExtraSensory": inspect_extrasensory(extra),
        "StudentLife": inspect_studentlife(student),
    }
    locations = {
        "schema_version": SCHEMA_VERSION,
        "path_base": "paper_project_root",
        "datasets": {
            "ExtraSensory": {"relative_path": EXTRASENSORY_RELATIVE.as_posix(), "role": "secondary_external_validation"},
            "StudentLife": {"relative_path": STUDENTLIFE_RELATIVE.as_posix(), "role": "primary_external_validation"},
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "dataset_locations.json", locations)
    _write_json(output_root / "schema_manifest.json", schema)
    records = sorted((*extra_records, *student_records), key=lambda row: (row["dataset"], row["relative_path"]))
    _write_manifest(output_root / "file_manifest.csv", records)
    return {"datasets": 2, "files": len(records), "schema_version": SCHEMA_VERSION}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output = args.output_root or args.project_root / "02_experiments" / "external_data" / "readiness"
    result = prepare(args.project_root, output)
    print(json.dumps({"status": "PREPARED", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
