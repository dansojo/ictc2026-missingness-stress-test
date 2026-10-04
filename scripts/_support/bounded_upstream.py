"""Bounded raw-to-stress execution with complete original selection/scales.

Only orchestration is new. The raw canonicalizer and primitive, population,
selection, scale, mask and replay implementations are verified original files.
G2 metadata follows g2_vendor/reliability.py:evaluate_g2_reliability. No archive
result, selected day, mask, baseline or canonical table is a calculation input.
These stages never assert a full-study reproduction. A coordinator independently
seals their manifests into the fresh_lineage parent graph.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "original_paper_reproduction/paper"
LEADERBOARD = ROOT / "original_paper_reproduction/leaderboard"
for directory in (PAPER, LEADERBOARD):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from full_prepare import create_fresh_output, emit, save_frame, sha256, verify_vendor, write_json

PRIMITIVES = ("screen_load_24h", "phone_activity_load_24h", "usage_load_24h",
              "mobile_light_exposure_24h", "wearable_light_exposure_24h")
SCENARIO = "contiguous_20pct"
ROOT_SEED = 42
DRAWS = 50


def _manifest(stage: str, inputs: object) -> dict:
    return {"schema_name": "independent_etri_reproduction", "stage": stage,
            "status": "running", "fresh_execution": True, "scope": "bounded",
            "full_reproduction": False, "new_model_fits": 0, "inputs": inputs,
            "outputs": {}}


def _finish(output: Path, manifest: dict, started: float) -> dict:
    manifest.update(status="complete", seconds=time.perf_counter() - started)
    write_json(output / "manifest.json", manifest)
    emit(stage=manifest["stage"], scope="bounded", status="complete",
         seconds=manifest["seconds"], rows={k: v.get("rows") for k, v in manifest["outputs"].items()})
    return manifest


def _failed(output: Path, manifest: dict, error: BaseException) -> None:
    manifest.update(status="failed", error_type=type(error).__name__, error=str(error))
    write_json(output / "manifest.json", manifest)
    (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")


def _pins(paths) -> dict[str, str]:
    return {str(Path(path).resolve(strict=True)): sha256(Path(path)) for path in paths}


def _check_pins(pins: dict[str, str]) -> None:
    for name, expected in pins.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Input changed during bounded execution: {name}")


def _canonical_pins(canonical_root: Path, expected: dict | None = None) -> dict:
    files = sorted(canonical_root.glob("*_canonical.parquet"))
    if len(files) != 12:
        raise ValueError("Bounded upstream requires all twelve fresh canonical files")
    records = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}
    if expected is not None and records != expected:
        raise ValueError("Canonical inputs differ from their fresh prepare parent")
    return records


def _keys(selected: pd.DataFrame) -> list[tuple[str, int, str]]:
    return [(str(r.subject_id), int(r.sensor_day_id), str(r.primitive))
            for r in selected.itertuples(index=False)]


def run_raw(raw_root: Path, output: Path) -> Path:
    """Canonicalize all twelve sensors; do not build aligned features or fit models."""
    import data_preparation as prep

    raw_root, output = Path(raw_root).resolve(strict=True), Path(output).resolve()
    sources = prep.verify_vendor()
    inputs = prep.validate_inputs(raw_root, output)
    before = _pins(inputs)
    output = create_fresh_output(output, [raw_root, prep.VENDOR])
    started = time.perf_counter()
    manifest = _manifest("raw", {"raw_root": str(raw_root), "files": before})
    manifest.update(scientific_sources=sources,
                    consumed_raw_files=[f"ch2025_data_items/ch2025_{s}.parquet" for s in prep.SENSORS],
                    provenance_only_raw_files=list(prep.RAW_FILES[:2]),
                    adaptations=["Rebind original module Path constants to this fresh output.",
                                 "Invoke canonicalize for twelve sensors; omit aligned features and model fitting."])
    write_json(output / "manifest.json", manifest)
    try:
        module = prep._load_stage("src/sensors/build_all_sensor_aligned.py", output / "project", raw_root)
        module.CANONICAL_DIR = output / "canonical"
        canonical = module.CANONICAL_DIR
        canonical.mkdir(parents=True, exist_ok=False)
        for sensor in prep.SENSORS:
            emit(stage="raw", step="canonicalize", sensor=sensor)
            frame, audit, value_columns, state_columns = module.canonicalize(sensor)
            path = canonical / f"{sensor}_canonical.parquet"
            manifest["outputs"][sensor] = {
                "path": path.relative_to(output).as_posix(), "sha256": sha256(path),
                "bytes": path.stat().st_size, "rows": len(frame), "columns": list(frame.columns),
                "value_columns": list(value_columns), "state_columns": list(state_columns),
                "audit": audit,
            }
            write_json(output / "manifest.json", manifest)
            del frame
        _check_pins(before)
        if prep.verify_vendor() != sources:
            raise ValueError("Preprocessor source pins changed")
        manifest["canonical_root"] = str(canonical)
        _finish(output, manifest, started)
        return canonical
    except BaseException as error:
        _failed(output, manifest, error)
        raise


def run_prepare(canonical_root: Path, output: Path) -> dict:
    """Run the unchanged all-853-day preparation, marking its bounded-chain scope."""
    from full_prepare import run_prepare as original_prepare

    manifest = original_prepare(Path(canonical_root), Path(output))
    manifest.update(scope="bounded", full_reproduction=False,
                    population_scope="all 853 days / 10 participants; original LOSO baselines")
    write_json(Path(output) / "manifest.json", manifest)
    return manifest


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _original_parity(rel, prepared, primitive, row, masks, result, sensor_index) -> None:
    """Original evaluate_g2_reliability compact/public check, once per primitive."""
    calendar = rel._calendar_row_for_replay(row)
    sensor_slice, _ = rel._source_frame_slice_indexed(primitive, calendar, sensor_index)
    masked, audit = rel.apply_mask_intervals(primitive, sensor_slice, prepared.sensor_file, calendar, masks)
    value, quality = rel.recompute_primitive_from_frame(primitive, masked, prepared.sensor_file, calendar)
    if not np.isclose(result.value, value, rtol=1e-12, atol=1e-12, equal_nan=True):
        raise ValueError(f"Original compact/public replay value differs: {primitive}")
    if (str(result.quality["status"]) != str(quality["status"])
            or int(result.quality["observed_epochs"]) != int(quality["observed_epochs"])):
        raise ValueError(f"Original compact/public replay quality differs: {primitive}")
    for name in ("availability_coverage", "longest_gap_minutes"):
        if not np.isclose(float(result.quality[name]), float(quality[name]), rtol=0.0, atol=1e-12, equal_nan=True):
            raise ValueError(f"Original compact/public replay {name} differs: {primitive}")
    for name in ("original_count", "deleted_count", "retained_count"):
        if getattr(result, name) != int(audit[name].max()):
            raise ValueError(f"Original compact/public replay {name} differs: {primitive}")
    compact = rel._compact_confidence_after_replay(primitive, row, result.value, result.quality)
    public = rel._confidence_after_replay(primitive, row, value, quality)
    if not all(np.isclose(compact[k], public[k], rtol=0.0, atol=1e-12) for k in compact):
        raise ValueError(f"Original compact/public replay confidence differs: {primitive}")


def run_g2(prepare_dir: Path, canonical_root: Path, output: Path, cell_limit: int) -> list[tuple[str, int, str]]:
    """Select/normalize on all 800 cells, then replay at most five cells x 50 draws."""
    from full_sources import load_g2_modules, verify_g2_sources
    from full_stress import g2_records, read_upstream_manifest

    if type(cell_limit) is not int or not 1 <= cell_limit <= len(PRIMITIVES):
        raise ValueError("cell_limit must be an integer from one through five")
    prepare_dir, canonical_root = Path(prepare_dir).resolve(strict=True), Path(canonical_root).resolve(strict=True)
    prep = read_upstream_manifest(prepare_dir, "prepare", ("primitives", "baselines", "representations"))
    canonical_files = _canonical_pins(canonical_root, prep["canonical_files"])
    before = _pins([prepare_dir / "manifest.json", *(prepare_dir / r["path"] for r in prep["outputs"].values()),
                    *(canonical_root / name for name in canonical_files)])
    historical = load_g2_modules()
    verify_vendor()
    rel = historical.reliability
    output = create_fresh_output(Path(output), [prepare_dir, canonical_root, PAPER])
    started = time.perf_counter()
    manifest = _manifest("g2", {"canonical_root": str(canonical_root), "canonical_files": canonical_files,
                               "prepare_manifest_sha256": sha256(prepare_dir / "manifest.json"), "files": before})
    manifest.update(scientific_sources=historical.provenance,
                    parameters={"root_seed": ROOT_SEED, "draws": DRAWS, "max_days": 10,
                                "scenarios": [SCENARIO], "cell_limit": cell_limit},
                    selection_scope="all 800 original selected cells before choosing one cell per recovery primitive",
                    normalization_scope="all original selected unmasked values and all-population LOSO baselines")
    write_json(output / "manifest.json", manifest)
    try:
        primitives = pd.read_parquet(prepare_dir / "primitives.parquet")
        baselines = pd.read_parquet(prepare_dir / "baselines.parquet")
        primitives = primitives.assign(__subject_sort=primitives["subject_id"].map(str)).sort_values(
            ["__subject_sort", "sensor_day_id"], kind="stable").drop(columns="__subject_sort").reset_index(drop=True)
        rows = {int(row.sensor_day_id): pd.Series(row._asdict()) for row in primitives.itertuples(index=False)}
        if len(rows) != len(primitives):
            raise ValueError("Fresh sensor day IDs must remain globally unique")
        selected_all, selection_audit = rel.select_reliability_days(primitives, max_days=10)
        if (len(selected_all) != 800 or selected_all["subject_id"].nunique() != 10
                or selected_all["primitive"].nunique() != 8
                or not selected_all.groupby(["subject_id", "primitive"]).size().eq(10).all()):
            raise ValueError("Full original G2 selection topology differs from 800 cells")
        scale_records = []
        for selected in selected_all.itertuples(index=False):
            row = rows[int(selected.sensor_day_id)]
            if row["subject_id"] != selected.subject_id:
                raise ValueError("Fresh selected participant/day identity mismatch")
            scale_records.append({"subject_id": selected.subject_id, "sensor_day_id": int(selected.sensor_day_id),
                                  "primitive": selected.primitive, "original_value": float(row[selected.primitive])})
        standardizers = rel.compute_reliability_standardizers(pd.DataFrame(scale_records),
                            frozen_population_baselines=baselines, primitive_table=primitives)
        scale_lookup = {(str(r.subject_id), r.primitive): r for r in standardizers.itertuples(index=False)}
        confidences = rel.compute_primitive_confidence(primitives, primitive_names=rel.CORE_G2_PRIMITIVES,
                                                       usage_policy="screen_proxy")
        confidence_lookup = {(str(r.subject_id), int(r.sensor_day_id), r.primitive): r
                             for r in confidences.itertuples(index=False)}
        # Pick one first participant/day per requested primitive only AFTER full selection/scales.
        selected = pd.concat([selected_all.loc[selected_all["primitive"].eq(p)].sort_values(
            ["subject_id", "sensor_day_id"], kind="stable").head(1) for p in PRIMITIVES[:cell_limit]], ignore_index=True)
        selected = selected.sort_values(["subject_id", "primitive", "sensor_day_id"], kind="stable").reset_index(drop=True)
        if len(selected) != cell_limit:
            raise ValueError("Missing recovery primitive in full fresh selection")
        for name, table in (("selected_days", selected), ("full_selected_days", selected_all),
                            ("selected_day_audit", selection_audit), ("standardizers", standardizers)):
            manifest["outputs"][name] = save_frame(output, name, table)
        write_json(output / "manifest.json", manifest)
        cells, audits, masks_all, sensor_indexes = [], [], [], {}
        for selected_row in selected.itertuples(index=False):
            subject, day, primitive = str(selected_row.subject_id), int(selected_row.sensor_day_id), str(selected_row.primitive)
            row = rows[day]
            sensor_file = rel._DEFINITIONS[primitive].sensor_file
            if sensor_file not in sensor_indexes:
                sensor_indexes[sensor_file] = rel._index_sensor_frame(pd.read_parquet(canonical_root / sensor_file), sensor_file)
            indexed = sensor_indexes[sensor_file]
            calendar = rel._calendar_row_for_replay(row)
            sensor_slice, _ = rel._source_frame_slice_indexed(primitive, calendar, indexed)
            prepared = rel.prepare_primitive_replay(primitive, sensor_slice, sensor_file, calendar)
            original_replay = rel.recompute_prepared_primitive(prepared)
            original_value, original_status = float(row[primitive]), str(row[f"{primitive}__status"])
            if (not np.isclose(original_replay.value, original_value, rtol=1e-12, atol=1e-12, equal_nan=True)
                    or str(original_replay.quality["status"]) != original_status):
                raise ValueError(f"Fresh original canonical replay mismatch for {day}/{primitive}")
            original = confidence_lookup[subject, day, primitive]
            scale = scale_lookup[subject, primitive]
            for draw in range(DRAWS):
                masks = rel.generate_mask_intervals(primitive, subject_id=row["subject_id"], sensor_day_id=day,
                            lifelog_date=row["lifelog_date"], scenario=SCENARIO, draw=draw, root_seed=ROOT_SEED)
                seed = str(masks.iloc[0]["seed"])
                replay = rel.recompute_prepared_primitive_batch(prepared, (masks,))[0]
                if draw == 0:
                    _original_parity(rel, prepared, primitive, row, masks, replay, indexed)
                record_digest = _digest([prepared.source_record_digest, prepared.definition.timestamp_semantics,
                    [[pd.Timestamp(m.mask_start).isoformat(), pd.Timestamp(m.mask_end).isoformat()]
                     for m in masks.itertuples(index=False)]])
                audit_key = _digest([seed, record_digest])
                masked_value, quality = float(replay.value), dict(replay.quality)
                masked_status = str(quality["status"])
                confidence = rel._compact_confidence_after_replay(primitive, row, masked_value, quality)
                retained = bool(np.isfinite(masked_value) and masked_status == "observed")
                error = abs(masked_value - original_value) if retained else float("nan")
                standardized = error / float(scale.standardizer) if retained else float("nan")
                key = {"subject_id": row["subject_id"], "sensor_day_id": day, "primitive": primitive,
                       "scenario": SCENARIO, "draw": draw}
                audits.append({"deletion_audit_key": audit_key, **key, "seed": seed, "donor_sensor_day_id": None,
                    "original_count": replay.original_count, "deleted_count": replay.deleted_count,
                    "retained_count": replay.retained_count, "record_deletion_digest": record_digest,
                    "audit_representation": "normalized_source_digest_plus_mask_geometry",
                    "status": "ok" if replay.original_count else "no_target_records"})
                augmented = masks.copy()
                augmented["donor_sensor_day_id"], augmented["deletion_audit_key"] = None, audit_key
                masks_all.extend(augmented.to_dict(orient="records"))
                cells.append({**key, "lifelog_date": row["lifelog_date"], "elapsed_day_index": row["elapsed_day_index"],
                    "root_seed": str(ROOT_SEED), "seed": seed, "donor_sensor_day_id": None,
                    "deletion_audit_key": audit_key, "original_value": original_value, "masked_value": masked_value,
                    "original_status": original_status, "masked_status": masked_status, "retained": retained,
                    "absolute_error": error, "standardized_absolute_error": standardized,
                    "standardizer": float(scale.standardizer), "standardizer_source": scale.scale_source,
                    "selected_day_mad": float(scale.selected_day_mad), "frozen_population_scale": float(scale.frozen_population_scale),
                    "scale_floor": float(scale.scale_floor), "floor_applied": bool(scale.floor_applied),
                    "original_value_only_confidence": float(original.value_only_confidence),
                    "original_coverage_only_confidence": float(original.coverage_only_confidence),
                    "original_quality_confidence": float(original.quality_confidence),
                    "masked_value_only_confidence": confidence["value_only"],
                    "masked_coverage_only_confidence": confidence["coverage_only"],
                    "masked_quality_confidence": confidence["quality"],
                    "success": bool(retained and standardized <= 0.20),
                    "usage_record_deletion_stability": primitive.startswith("usage_"),
                    "reason": "retained" if retained else f"masked_{masked_status}",
                    "status": "observed" if retained else "abstained"})
            emit(stage="g2", step="bounded_cell", primitive=primitive, draws=DRAWS)
        for name, records in (("cell_replays", cells), ("deletion_audit", audits), ("mask_intervals", masks_all)):
            frame = pd.DataFrame.from_records(records)
            # Physical row order is part of the unchanged G2 record contract.
            # Assemble that layout before serializing; never reorder/repair an
            # upstream artifact at the consumer or relax its strict validator.
            columns = {"cell_replays": g2_records._CELL_COLUMNS,
                       "deletion_audit": g2_records._DELETION_COLUMNS,
                       "mask_intervals": g2_records._MASK_COLUMNS}[name]
            if len(frame.columns) != len(columns) or set(frame.columns) != set(columns):
                raise ValueError(f"Generated bounded G2 {name} physical fields differ")
            frame = frame.loc[:, list(columns)]
            order = ["__primitive_order", "__subject_sort", "draw", "sensor_day_id"]
            if name == "mask_intervals":
                order.append("mask_interval_id")
            frame = frame.assign(__primitive_order=frame["primitive"].map({p: i for i, p in enumerate(rel.CORE_G2_PRIMITIVES)}),
                                 __subject_sort=frame["subject_id"].map(str)).sort_values(order, kind="stable").drop(
                                     columns=["__primitive_order", "__subject_sort"]).reset_index(drop=True)
            manifest["outputs"][name] = save_frame(output, name, frame)
        _check_pins(before)
        verify_vendor()
        if verify_g2_sources() != historical.provenance:
            raise ValueError("Historical G2 sources changed during execution")
        manifest.update(cell_keys=_keys(selected), all_selected_cells=len(selected_all),
                        executed_cells=len(selected), executed_replays=len(cells),
                        original_unmasked_replay_validation_scope="executed bounded cells only; scale values come from full fresh primitive table")
        _finish(output, manifest, started)
        return _keys(selected)
    except BaseException as error:
        _failed(output, manifest, error)
        raise


def run_stress(prepare_dir: Path, canonical_root: Path, g2_dir: Path, output: Path) -> list[tuple[str, int, str]]:
    """Run original current-source stress replay for each fresh bounded G2 cell."""
    import full_stress as stress

    prepare_dir, canonical_root, g2_dir = (Path(p).resolve(strict=True) for p in (prepare_dir, canonical_root, g2_dir))
    sources = verify_vendor()
    prep = stress.read_upstream_manifest(prepare_dir, "prepare", ("primitives", "baselines", "representations"))
    g2 = stress.read_upstream_manifest(g2_dir, "g2", stress.G2_TABLES)
    canonical_files = _canonical_pins(canonical_root, prep["canonical_files"])
    if (g2["inputs"]["canonical_files"] != canonical_files
            or g2["inputs"]["prepare_manifest_sha256"] != sha256(prepare_dir / "manifest.json")
            or g2.get("scope") != "bounded" or g2.get("full_reproduction") is not False):
        raise ValueError("Fresh bounded G2 ancestors or scope differ")
    before = _pins([prepare_dir / "manifest.json", g2_dir / "manifest.json",
                    *(prepare_dir / r["path"] for r in prep["outputs"].values()),
                    *(g2_dir / r["path"] for r in g2["outputs"].values()),
                    *(canonical_root / name for name in canonical_files)])
    output = create_fresh_output(Path(output), [prepare_dir, canonical_root, g2_dir, PAPER])
    started = time.perf_counter()
    manifest = _manifest("stress", {"canonical_root": str(canonical_root), "canonical_files": canonical_files,
                    "prepare_manifest_sha256": sha256(prepare_dir / "manifest.json"),
                    "g2_manifest_sha256": sha256(g2_dir / "manifest.json"), "files": before})
    manifest["scientific_sources"] = sources
    write_json(output / "manifest.json", manifest)
    try:
        selected = pd.read_parquet(g2_dir / g2["outputs"]["selected_days"]["path"])
        keys = _keys(selected)
        if (not 1 <= len(keys) <= 5 or len(set(keys)) != len(keys)
                or len({key[2] for key in keys}) != len(keys) or any(key[2] not in PRIMITIVES for key in keys)
                or [list(k) for k in keys] != g2["cell_keys"]):
            raise ValueError("Bounded fresh selected-cell identity differs")
        frames = {name: pd.read_parquet(g2_dir / g2["outputs"][name]["path"])
                  for name in ("cell_replays", "deletion_audit", "mask_intervals")}
        hashes = {name: g2["outputs"][name]["sha256"] for name in frames}
        expected = tuple((subject, day, primitive, SCENARIO, draw)
                         for subject, day, primitive in keys for draw in range(DRAWS))
        pairs = stress.build_validated_pairs(**frames, artifact_hashes=hashes, expected_keys=expected)
        primitives = pd.read_parquet(prepare_dir / "primitives.parquet")
        rows = {int(r.sensor_day_id): pd.Series(r._asdict()) for r in primitives.itertuples(index=False)}
        sensor_indexes, tables = {}, {name: [] for name in stress.CELL_RESULT_TABLES}
        for subject, day, primitive in keys:
            row = rows[day]
            if str(row["subject_id"]) != subject:
                raise ValueError("Stress selected participant/day identity differs")
            prepared = stress.prepare_selected_cell(primitive, row, canonical_root=canonical_root,
                                                     sensor_indexes=sensor_indexes)
            result = stress.replay_cell(prepared, [pairs[subject, day, primitive, SCENARIO, draw] for draw in range(DRAWS)])
            for name, frame in result.items():
                tables[name].append(frame)
            emit(stage="stress", step="bounded_cell", primitive=primitive, draws=DRAWS)
        for name, chunks in tables.items():
            manifest["outputs"][name] = save_frame(output, name, pd.concat(chunks, ignore_index=True))
        _check_pins(before)
        if verify_vendor() != sources:
            raise ValueError("Stress scientific source pins changed")
        manifest.update(cell_keys=keys, executed_cells=len(keys), parameters={"root_seed": ROOT_SEED, "draws": DRAWS})
        _finish(output, manifest, started)
        return keys
    except BaseException as error:
        _failed(output, manifest, error)
        raise
