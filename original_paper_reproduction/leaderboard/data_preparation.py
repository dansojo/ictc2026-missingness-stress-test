"""Rebuild the archived sensor-v2 features with explicit read-only raw inputs.

Scientific functions are loaded from hash-pinned, unchanged vendor files. Only
module Path constants are rebound. This module imports no numerical dependency
until run_pipeline() or compare_prepared() is called.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor/preprocessing"
MANIFEST = HERE / "preprocessing_manifest.json"
SENSORS = ("mScreenStatus", "mUsageStats", "wPedo", "mActivity", "wHr", "mLight",
           "wLight", "mACStatus", "mGps", "mWifi", "mBle", "mAmbience")
RAW_FILES = ("ch2026_metrics_train.csv", "ch2026_submission_sample.csv", *(
    f"ch2025_data_items/ch2025_{name}.parquet" for name in SENSORS))
STAGES = (
    "src/sensors/build_all_sensor_aligned.py",
    "src/episodes/build_sleep_episode_summary.py",
    "src/episodes/validate_sleep_episode_summary.py",
    "src/features/build_integrated_from_sensor_aligned.py",
    "src/sensors/build_test_sensor_aligned.py",
    "src/episodes/build_test_sleep_episode_summary.py",
    "src/features/build_test_integrated_from_sensor_aligned.py",
    "src/features/global_clean_sensor_v2.py",
)
FINAL_FILES = ("train_integrated_sensor_v2_cleaned.csv",
               "test_integrated_sensor_v2_cleaned.csv",
               "feature_columns_sensor_v2_cleaned.json")
KEYS = ("subject_id", "lifelog_date", "sleep_date")


def filesystem_path(path: Path) -> Path:
    """Use the Windows extended prefix for I/O without changing logical paths."""
    value = str(path)
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        if value.startswith("\\\\"):
            return Path("\\\\?\\UNC\\" + value[2:])
        return Path("\\\\?\\" + str(path.absolute()))
    return path


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with filesystem_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_inputs(data_root: Path, output_dir: Path) -> list[Path]:
    raw = data_root.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("raw input and output directories must not overlap")
    if output.exists():
        raise FileExistsError(f"refusing to reuse existing output: {output}")
    if not raw.is_dir():
        raise NotADirectoryError(raw)
    paths = [raw / name for name in RAW_FILES]
    missing = [str(path) for path in paths if not filesystem_path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing raw input files: " + "; ".join(missing))
    return paths


def verify_vendor(vendor_root: Path = VENDOR, manifest_path: Path = MANIFEST) -> list[dict]:
    root = vendor_root.resolve(strict=True)
    document = json.loads(filesystem_path(manifest_path).read_text(encoding="utf-8"))
    sources = document["sources"]
    if not sources:
        raise ValueError("vendor manifest is empty")
    for record in sources:
        relative = Path(record["vendor_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("vendor manifest path escapes vendor root")
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("vendor file resolves outside vendor root")
        if (filesystem_path(path).stat().st_size != record["bytes"]
                or file_sha256(path) != record["sha256"]):
            raise ValueError(f"vendored source changed: {relative}")
    return sources


def rebind_module_paths(module, project_root: Path, data_root: Path) -> None:
    """Move every original project-relative Path; only sensor reads stay external."""
    original = module.PROJECT_ROOT
    for name, value in tuple(vars(module).items()):
        if isinstance(value, Path) and (value == original or original in value.parents):
            setattr(module, name, project_root / value.relative_to(original))
    if hasattr(module, "SENSOR_ROOT"):
        module.SENSOR_ROOT = data_root / "ch2025_data_items"


def _load_stage(relative: str, project_root: Path, data_root: Path):
    path = filesystem_path(VENDOR / relative)
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load stage: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    rebind_module_paths(module, project_root, data_root)
    return module


def _write_json(path: Path, document: dict) -> None:
    filesystem_path(path).write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")


def write_training_input_manifest(
    data_root: Path, output_path: Path, *, provenance_path: Path | None = None,
) -> dict:
    """Pin the four fresh training inputs without requiring archive byte identity."""
    if filesystem_path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite training input manifest: {output_path}")
    names = [*("processed/" + name for name in FINAL_FILES),
             "raw/data_vvs/ch2026_submission_sample.csv"]
    document = {"source": "fresh raw preprocessing with unchanged archived scientific source",
                "files": [{"path": name,
                           "bytes": filesystem_path(data_root / name).stat().st_size,
                           "sha256": file_sha256(data_root / name)} for name in names]}
    if provenance_path is not None:
        relative = provenance_path.resolve().relative_to(output_path.parent.resolve())
        document["provenance"] = {"path": relative.as_posix(),
                                  "sha256": file_sha256(provenance_path)}
    with filesystem_path(output_path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    return document


def run_pipeline(data_root: Path, output_dir: Path) -> dict:
    raw = data_root.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    inputs = validate_inputs(raw, output)
    source_pins = verify_vendor()
    if {item["vendor_relative_path"] for item in source_pins} != set(STAGES):
        raise ValueError("vendor manifest does not pin the complete preprocessing stage set")
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    before = {str(path.relative_to(raw)).replace("\\", "/"): file_sha256(path)
              for path in inputs}
    filesystem_path(output).mkdir(parents=True, exist_ok=False)
    project = filesystem_path(output / "project")
    staged_raw = project / "data/raw/data_vvs"
    staged_raw.mkdir(parents=True)
    # Only the small key/label CSVs are copied, so original report-relative paths
    # remain valid. All twelve sensor streams are read directly from data_root.
    for filename in RAW_FILES[:2]:
        shutil.copyfile(filesystem_path(raw / filename), staged_raw / filename)
    document = {
        "schema": "ictc2026-leaderboard-preprocessing-run-v1",
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw), "output_dir": str(output),
        "input_sha256": before, "vendor_sha256": {
            record["vendor_relative_path"]: record["sha256"] for record in source_pins},
        "method_changes": [], "adaptations": [
            "Rebound module Path constants to the new project directory.",
            "Copied labels/sample keys only; sensor Parquet files are read from data_root."],
        "versions": {"python": sys.version, "numpy": np.__version__,
                     "pandas": pd.__version__, "pyarrow": pa.__version__},
        "stages": [],
    }
    manifest = output / "preprocessing_run.json"
    _write_json(manifest, document)
    started = time.perf_counter()
    try:
        for relative in STAGES:
            print(f"PREPROCESS_STAGE_START={relative}", flush=True)
            stage_start = time.perf_counter()
            stage = _load_stage(relative, project, filesystem_path(raw))
            stage.main()
            record = {"stage": relative, "status": "complete",
                      "wall_seconds": time.perf_counter() - stage_start}
            document["stages"].append(record)
            _write_json(manifest, document)
            print("PREPROCESS_STAGE_COMPLETE=" + json.dumps(record), flush=True)
        processed = project / "data/processed"
        features = json.loads((processed / FINAL_FILES[2]).read_text(encoding="utf-8"))
        train = pd.read_csv(processed / FINAL_FILES[0])
        test = pd.read_csv(processed / FINAL_FILES[1])
        original_train = pd.read_csv(filesystem_path(raw / RAW_FILES[0]))
        original_sample = pd.read_csv(filesystem_path(raw / RAW_FILES[1]))
        checks = {
            "feature_count_3155": len(features) == 3155,
            "feature_names_unique": len(features) == len(set(features)),
            "train_rows_preserved": len(train) == len(original_train),
            "test_rows_preserved": len(test) == len(original_sample),
            "train_keys_preserved": train[list(KEYS)].equals(original_train[list(KEYS)]),
            "test_keys_preserved": test[list(KEYS)].equals(original_sample[list(KEYS)]),
            "train_features_present": all(name in train for name in features),
            "test_features_present": all(name in test for name in features),
            "no_target_features": not set(features).intersection(("Q1", "Q2", "Q3", "S1", "S2", "S3", "S4")),
        }
        after = {str(path.relative_to(raw)).replace("\\", "/"): file_sha256(path)
                 for path in inputs}
        checks["raw_input_unchanged"] = before == after
        document["input_sha256_after"] = after
        verify_vendor()
        document.update({"checks": checks, "feature_count": len(features),
                         "train_shape": list(train.shape), "test_shape": list(test.shape),
                         "final_outputs": [{"path": "project/data/processed/" + name,
                            "bytes": (processed / name).stat().st_size,
                            "sha256": file_sha256(processed / name)} for name in FINAL_FILES],
                         "status": "complete" if all(checks.values()) else "validation_failed"})
        if not all(checks.values()):
            raise ValueError("preprocessing output validation failed: " + json.dumps(checks))
    except BaseException as exc:
        document.update({"status": "failed", "error_type": type(exc).__name__,
                         "error": str(exc)})
        raise
    finally:
        document["wall_seconds"] = time.perf_counter() - started
        document["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest, document)
    write_training_input_manifest(project / "data", output / "training_input_manifest.json",
                                  provenance_path=manifest)
    return document


def compare_prepared(data_root: Path, reference_processed: Path) -> dict:
    """Compare original CSV storage using the same reader; report exact and tolerant values separately."""
    import numpy as np
    import pandas as pd
    processed = filesystem_path(data_root / "processed")
    reference = filesystem_path(reference_processed)
    features = json.loads((processed / FINAL_FILES[2]).read_text(encoding="utf-8"))
    expected_features = json.loads((reference / FINAL_FILES[2]).read_text(encoding="utf-8"))
    report = {"feature_list_equal": features == expected_features,
              "feature_json_bytes_equal": file_sha256(processed / FINAL_FILES[2]) == file_sha256(reference / FINAL_FILES[2]),
              "tables": [], "comparison_tolerance": {"rtol": 1e-12, "atol": 1e-12}}
    for name in FINAL_FILES[:2]:
        actual, expected = pd.read_csv(processed / name), pd.read_csv(reference / name)
        same_columns = list(actual.columns) == list(expected.columns)
        item = {"filename": name, "shape": list(actual.shape), "reference_shape": list(expected.shape),
                "shape_equal": actual.shape == expected.shape, "columns_equal": same_columns,
                "csv_bytes_equal": file_sha256(processed / name) == file_sha256(reference / name),
                "keys_equal": actual[list(KEYS)].equals(expected[list(KEYS)]),
                "dtype_differences": [], "value_differences": []}
        if same_columns and actual.shape == expected.shape:
            for col in actual:
                a, b = actual[col], expected[col]
                if str(a.dtype) != str(b.dtype):
                    item["dtype_differences"].append({"column": col, "actual": str(a.dtype), "reference": str(b.dtype)})
                equal = a.eq(b) | (a.isna() & b.isna())
                if not bool(equal.all()):
                    detail = {"column": col, "unequal_rows": int((~equal).sum())}
                    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                        av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
                        close = np.isclose(av, bv, rtol=1e-12, atol=1e-12, equal_nan=True)
                        finite = np.isfinite(av) & np.isfinite(bv)
                        detail.update({"rows_outside_tolerance": int((~close).sum()),
                            "max_absolute_difference": float(np.max(np.abs(av[finite] - bv[finite]))) if finite.any() else None})
                    item["value_differences"].append(detail)
        item["dtypes_equal"] = same_columns and not item["dtype_differences"]
        item["values_exact"] = same_columns and item["shape_equal"] and not item["value_differences"]
        report["tables"].append(item)
    report["exact_logical_match"] = report["feature_list_equal"] and all(
        item["shape_equal"] and item["columns_equal"] and item["keys_equal"] and item["dtypes_equal"] and item["values_exact"]
        for item in report["tables"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/raw"),
                        help="Contains ch2026 CSV files and ch2025_data_items/; read only")
    parser.add_argument("--output-dir", type=Path, default=Path("/data/prepared"),
                        help="New directory; results under project/data/processed")
    parser.add_argument("--reference-processed", type=Path,
                        help="Optional archived processed directory for a post-run comparison")
    parser.add_argument("--check-only", action="store_true",
                        help="Check inputs and vendored source hashes without writes or numerical imports")
    args = parser.parse_args()
    if args.check_only:
        inputs = validate_inputs(args.data_root, args.output_dir)
        sources = verify_vendor()
        print(json.dumps({"status": "preflight_passed", "raw_files": len(inputs), "source_files": len(sources)}))
        return 0
    result = run_pipeline(args.data_root, args.output_dir)
    if args.reference_processed:
        comparison = compare_prepared(args.output_dir / "project/data", args.reference_processed)
        _write_json(args.output_dir / "reference_comparison.json", comparison)
        print("PREPROCESS_REFERENCE_EXACT=" + str(comparison["exact_logical_match"]), flush=True)
        if not comparison["exact_logical_match"]:
            return 2
    print(json.dumps({"status": result["status"], "feature_count": result["feature_count"],
                      "output_dir": str(args.output_dir), "wall_seconds": result["wall_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
