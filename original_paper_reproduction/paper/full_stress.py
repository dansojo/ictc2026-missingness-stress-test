#!/usr/bin/env python3
"""Replay the complete Task12 grid from freshly generated independent G2 inputs.

The calculation boundary is stress_runner.py:467-631. All primitive replay,
mask, K_valid, inference and decision calculations call the immutable vendor.
Legacy ``Official*`` value classes below are validated scientific records bound
to NEW artifact hashes; they do not confer historical approval or release status.
No authority graph, historical result loader, or release persistence is invoked.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from full_prepare import (PAPER, VENDOR, create_fresh_output, emit, save_frame,
                          sha256, verify_vendor, write_json)

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from semantic_indicators import phaseguard_authority as g2_records
from semantic_indicators import stress_runner as original
from semantic_indicators.stress_contracts import (
    Contiguous20Crosslink, DRAWS, FEATURE_FAMILIES, ROOT_SEED,
    derive_matched_deletion_target,
)
from semantic_indicators.stress_masks import SCATTERED_SCENARIO
from semantic_indicators.stress_statistics import (
    evaluate_internal_decision, summarize_family_inference, summarize_stress_results,
)

KEY_COLUMNS = ("subject_id", "sensor_day_id", "primitive", "scenario", "draw")
CELL_COLUMNS = ("subject_id", "sensor_day_id", "primitive")
DOSE_SCENARIOS = ("contiguous_10pct", "contiguous_20pct", "contiguous_40pct")
G2_TABLES = ("selected_days", "cell_replays", "deletion_audit", "mask_intervals")
PRIMITIVE_ORDER = tuple(primitive for _, family in FEATURE_FAMILIES for primitive in family)
CellKey = tuple[str, int, str, str, int]
ValidatedPair = tuple[g2_records.OfficialCellEvidence, g2_records.OfficialMaskCell]
prepare_selected_cell = original._prepare_selected_cell
CELL_RESULT_TABLES = ("target_ledger", "mask_ledger", "deletion_audit", "cell_replays",
                      "official_current_crosslink", "contiguous_rows")
_CELL_WORKER_CONTEXT: dict | None = None
_CELL_SENSOR_INDEXES: dict = {}


def read_upstream_manifest(directory: Path, stage: str, required_names: tuple[str, ...]) -> dict:
    """Require completed fresh provenance and verify each consumed physical table."""
    root = directory.resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    if (manifest.get("schema_name") != "independent_etri_reproduction"
            or manifest.get("stage") != stage or manifest.get("status") != "complete"
            or manifest.get("fresh_execution") is not True):
        raise ValueError(f"{stage} requires a completed fresh independent upstream manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{stage} manifest has no output hash mapping")
    for name in required_names:
        record = outputs.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"{stage} lacks output {name}")
        relative = Path(record["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"{stage} output path escapes its stage root: {name}")
        if not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"{stage} upstream output is missing or changed: {name}")
        parquet = pq.ParquetFile(path)
        if (parquet.metadata.num_rows != record.get("rows")
                or parquet.schema_arrow.names != record.get("columns")):
            raise ValueError(f"{stage} upstream schema/row manifest mismatch: {name}")
    return manifest


def _physical_rows(frame: pd.DataFrame, *, mask: bool = False) -> list[dict]:
    """Recover Arrow nulls from pandas without changing the scientific values.

    The original validator consumes Arrow rows. pandas represents nullable masks'
    int64 donor IDs as float64; restore only exact integral IDs to the original
    Arrow-int semantics. All other scalar/type checks stay in the original code.
    """
    rows = pa.Table.from_pandas(frame, preserve_index=False).to_pylist()
    for row in rows:
        for name, value in row.items():
            # pandas 3 can infer microsecond dates; Arrow then returns datetime.
            # Restore the original exact Timestamp value object (no rounding).
            if type(value) is datetime:
                row[name] = pd.Timestamp(value)
    if mask:
        for row in rows:
            donor = row.get("donor_sensor_day_id")
            if type(donor) is float:
                if not np.isfinite(donor) or not donor.is_integer():
                    raise ValueError("mask donor must be an exact nullable integer")
                row["donor_sensor_day_id"] = int(donor)
    return rows


def build_validated_pairs(
    cell_replays: pd.DataFrame,
    deletion_audit: pd.DataFrame,
    mask_intervals: pd.DataFrame,
    *,
    artifact_hashes: Mapping[str, str],
    expected_keys: tuple[CellKey, ...],
) -> dict[CellKey, ValidatedPair]:
    """Validate requested NEW G2 record pairs with the original pure row validator.

    Other scenario rows may be present in the inputs. Requested keys must be
    complete and unique; no missing, duplicate, reordered interval, crosswired
    seed/count/scale/error record is silently repaired. The caller pins the input
    artifacts before invoking this calculation-only helper.
    """
    if type(expected_keys) is not tuple or not expected_keys or len(set(expected_keys)) != len(expected_keys):
        raise ValueError("expected G2 keys must be a nonempty unique exact tuple")
    expected = set(expected_keys)
    dictionaries = []
    for name, frame, is_mask in (("cell_replays", cell_replays, False),
                                  ("deletion_audit", deletion_audit, False),
                                  ("mask_intervals", mask_intervals, True)):
        if not isinstance(frame, pd.DataFrame) or not set(KEY_COLUMNS).issubset(frame.columns):
            raise ValueError(f"{name} lacks semantic G2 keys")
        filtered = frame.loc[pd.MultiIndex.from_frame(frame[list(KEY_COLUMNS)]).isin(expected_keys)]
        unique_columns = [*KEY_COLUMNS, "mask_interval_id"] if is_mask else list(KEY_COLUMNS)
        if filtered.duplicated(unique_columns).any():
            raise ValueError(f"duplicate G2 {name} key")
        grouped: dict = {}
        for row in _physical_rows(filtered, mask=is_mask):
            key = tuple(row[column] for column in KEY_COLUMNS)
            if is_mask:
                grouped.setdefault(key, []).append(row)
            else:
                grouped[key] = row
        if not is_mask and set(grouped) != expected:
            raise ValueError(f"missing G2 {name} keys")
        dictionaries.append(grouped)
    cells, audits, masks = dictionaries
    pairs = {}
    for key in expected_keys:
        pairs[key] = g2_records._build_official_pair(
            cells[key], audits[key], masks.get(key, []), expected_key=key,
            cell_replays_artifact_sha256=artifact_hashes["cell_replays"],
            mask_intervals_artifact_sha256=artifact_hashes["mask_intervals"],
            deletion_audit_artifact_sha256=artifact_hashes["deletion_audit"],
        )
    return pairs


def replay_cell(prepared, pairs: Sequence[ValidatedPair]) -> dict[str, pd.DataFrame]:
    """Source-derived stress_runner.py:507-572, preserving all digest inputs."""
    if not pairs:
        raise ValueError("a selected cell must include its contiguous20 draws")
    targets, contiguous_rows, crosslink_rows = [], [], []
    for evidence, mask in sorted(pairs, key=lambda pair: pair[0].draw):
        if original._official_key(evidence) != original._official_key(mask) or evidence.scenario != "contiguous_20pct":
            raise ValueError("contiguous20 evidence and mask are crosswired")
        crosslink = Contiguous20Crosslink(
            subject_id=evidence.subject_id, sensor_day_id=evidence.sensor_day_id,
            primitive=evidence.primitive, draw=evidence.draw,
            deletion_audit_key=mask.deletion_audit.deletion_audit_key,
            official_deleted_count=mask.deletion_audit.deleted_count,
            official_retained_count=mask.deletion_audit.retained_count,
            official_record_deletion_digest=mask.deletion_audit.record_deletion_digest,
            intervals=tuple((interval.mask_start, interval.mask_end) for interval in mask.intervals),
            reference_value=float(evidence.original_value), reference_status=evidence.original_status,
            standardizer=float(evidence.standardizer), standardizer_source=evidence.standardizer_source,
        )
        target = derive_matched_deletion_target(prepared, crosslink)
        targets.append(target)
        crosslink_rows.append({
            "subject_id": evidence.subject_id, "sensor_day_id": evidence.sensor_day_id,
            "primitive": evidence.primitive, "scenario": "contiguous_20pct", "draw": evidence.draw,
            "deletion_audit_key": crosslink.deletion_audit_key,
            "official_original_count": crosslink.official_deleted_count + crosslink.official_retained_count,
            "official_deleted_count": crosslink.official_deleted_count,
            "official_retained_count": crosslink.official_retained_count,
            "historical_record_deletion_digest": crosslink.official_record_deletion_digest,
            "current_record_deletion_digest": target.current_record_deletion_digest,
            "prepared_source_content_digest": target.prepared_source_content_digest,
            "target_valid_deleted_count": target.target_deleted_count,
            "crosslink_content_digest": crosslink.content_digest,
            "target_content_digest": target.content_digest,
        })
        contiguous_rows.append(original._contiguous_row(evidence, target))
    replay = original.run_stress_replays(prepared, targets)
    compact_masks = replay.mask_ledger.copy()
    compact_masks["record_identity_count"] = compact_masks["record_identity_digests"].map(len)
    compact_masks["interval_count"] = compact_masks["mask_intervals"].map(len)
    compact_masks = compact_masks.drop(columns=["record_identity_digests", "mask_intervals"])
    return {"target_ledger": replay.target_ledger, "mask_ledger": compact_masks,
            "deletion_audit": replay.deletion_audit, "cell_replays": replay.cell_replays,
            "official_current_crosslink": pd.DataFrame(crosslink_rows),
             "contiguous_rows": pd.DataFrame(contiguous_rows)}


def make_cell_tasks(selected, row_by_day, frames, artifact_hashes, *, draws=tuple(range(DRAWS))):
    """Build bounded, dtype-preserving raw payloads; no sealed object crosses IPC."""
    if (type(draws) is not tuple or not draws or len(set(draws)) != len(draws)
            or any(type(draw) is not int or not 0 <= draw < DRAWS for draw in draws)):
        raise ValueError("cell draws must be unique original draw indices")
    grouped = {}
    for name in ("cell_replays", "deletion_audit", "mask_intervals"):
        frame = frames[name]
        relevant = frame.loc[frame["scenario"].eq("contiguous_20pct") & frame["draw"].isin(draws)].copy()
        grouped[name] = (relevant, relevant.groupby(list(CELL_COLUMNS), sort=False).indices)

    def generate():
        for index, (subject, day, primitive) in enumerate(selected):
            cell = (subject, int(day), primitive)
            yield {"cell_index": index, "cell": cell, "draws": draws,
                   "row": row_by_day[day].copy(deep=True), "artifact_hashes": dict(artifact_hashes),
                   "frames": {name: frame.iloc[indices.get(cell, [])].copy()
                              for name, (frame, indices) in grouped.items()}}
    return generate()


def initialize_cell_worker(context: dict) -> None:
    """Each process independently verifies immutable sources and physical inputs."""
    global _CELL_WORKER_CONTEXT, _CELL_SENSOR_INDEXES
    _CELL_WORKER_CONTEXT, _CELL_SENSOR_INDEXES = None, {}
    if verify_vendor() != context.get("scientific_sources"):
        raise ValueError("stress worker scientific source pins differ from the parent")
    prepare_dir, g2_dir, canonical_root = (Path(context[name]).resolve(strict=True)
                                          for name in ("prepare_dir", "g2_dir", "canonical_root"))
    prepared = read_upstream_manifest(prepare_dir, "prepare", ("primitives", "baselines", "representations"))
    g2 = read_upstream_manifest(g2_dir, "g2", G2_TABLES)
    if (sha256(prepare_dir / "manifest.json") != context.get("prepare_manifest_sha256")
            or sha256(g2_dir / "manifest.json") != context.get("g2_manifest_sha256")):
        raise ValueError("stress worker upstream manifest changed from parent pins")
    if (prepared.get("canonical_files") != context.get("canonical_files")
            or g2.get("inputs", {}).get("canonical_files") != context["canonical_files"]
            or g2["inputs"].get("prepare_manifest_sha256") != context["prepare_manifest_sha256"]):
        raise ValueError("stress worker prepare/canonical ancestor is crosswired")
    hashes = {name: g2["outputs"][name]["sha256"]
              for name in ("cell_replays", "deletion_audit", "mask_intervals")}
    if hashes != context.get("artifact_hashes"):
        raise ValueError("stress worker G2 artifact pins are crosswired")
    _verify_canonical(canonical_root, context["canonical_files"])
    _CELL_WORKER_CONTEXT = dict(context)
    emit(stage="stress", step="worker_ready", worker_pid=os.getpid())


def _process_memory_bytes() -> dict:
    """Read this process's memory counters for scheduling evidence only."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                          "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                          "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"peak_working_set_bytes": None, "working_set_bytes": None}
        return {"peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "working_set_bytes": int(counters.WorkingSetSize)}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_working_set_bytes": int(peak * (1 if sys.platform == "darwin" else 1024)),
            "working_set_bytes": None}


def run_cell_task(task: dict) -> dict:
    """Rebuild scientific values inside this process, then use the unchanged cell calculation."""
    if _CELL_WORKER_CONTEXT is None:
        raise RuntimeError("stress cell worker must be initialized before calculation")
    required = {"cell_index", "cell", "draws", "row", "frames", "artifact_hashes"}
    if type(task) is not dict or set(task) != required:
        raise ValueError("stress cell payload must contain only raw rows and pinned keys")
    cell, draws = task["cell"], task["draws"]
    if (type(task["cell_index"]) is not int or task["cell_index"] < 0
            or type(cell) is not tuple or len(cell) != 3
            or type(cell[0]) is not str or type(cell[1]) is not int or type(cell[2]) is not str
            or type(draws) is not tuple or not draws or len(set(draws)) != len(draws)
            or any(type(draw) is not int or not 0 <= draw < DRAWS for draw in draws)):
        raise ValueError("stress cell identity/draw topology is invalid")
    subject, day, primitive = cell
    row = task["row"]
    if (not isinstance(row, pd.Series) or row.get("subject_id") != subject
            or row.get("sensor_day_id") != day or primitive not in PRIMITIVE_ORDER):
        raise ValueError("stress cell is crosswired to the fresh primitive row")
    if task["artifact_hashes"] != _CELL_WORKER_CONTEXT["artifact_hashes"]:
        raise ValueError("stress cell G2 artifact hashes differ from verified worker inputs")
    frames = task["frames"]
    if type(frames) is not dict or set(frames) != {"cell_replays", "deletion_audit", "mask_intervals"}:
        raise ValueError("stress cell must carry all three original G2 row tables")
    expected = tuple((*cell, "contiguous_20pct", draw) for draw in draws)
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame) or not set(KEY_COLUMNS).issubset(frame.columns):
            raise ValueError(f"stress cell {name} is not a raw G2 table")
        keys = set(frame[list(KEY_COLUMNS)].itertuples(index=False, name=None))
        if not keys.issubset(expected) or (name != "mask_intervals" and keys != set(expected)):
            raise ValueError(f"stress cell {name} includes missing or crosswired keys")
    started = time.perf_counter()
    pairs = build_validated_pairs(frames["cell_replays"], frames["deletion_audit"], frames["mask_intervals"],
                                  artifact_hashes=task["artifact_hashes"], expected_keys=expected)
    prepared = prepare_selected_cell(primitive, row, canonical_root=Path(_CELL_WORKER_CONTEXT["canonical_root"]),
                                     sensor_indexes=_CELL_SENSOR_INDEXES)
    tables = replay_cell(prepared, [pairs[key] for key in expected])
    return {"cell_index": task["cell_index"], "cell": cell, "tables": tables,
            "worker_pid": os.getpid(), "seconds": time.perf_counter() - started,
            **_process_memory_bytes()}


def _bounded_cell_map(executor, function, tasks, *, max_pending: int):
    """Consume at most max_pending payloads before their results are collected."""
    if type(max_pending) is not int or max_pending < 1:
        raise ValueError("pending cell bound must be a positive integer")
    iterator, pending, exhausted = iter(tasks), set(), False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(function, task))
            if not pending:
                break
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                emit(stage="stress", step="waiting_for_cells", pending_cells=len(pending))
            for future in done:
                pending.remove(future)
                yield future.result()
    finally:
        for future in pending:
            future.cancel()


def iter_cell_results(tasks, context: dict, *, workers: int = 3, max_pending: int | None = None):
    """One worker boundary in both reference and explicitly spawned parallel modes."""
    if type(workers) is not int or workers not in (1, 2, 3):
        raise ValueError("stress workers must be 1, 2, or 3")
    bound = workers * 2 if max_pending is None else max_pending
    if type(bound) is not int or not 1 <= bound <= workers * 2:
        raise ValueError("pending cells must be bounded to at most twice the worker count")
    if workers == 1:
        initialize_cell_worker(context)
        for task in tasks:
            yield run_cell_task(task)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_cell_worker, initargs=(context,)) as executor:
        yield from _bounded_cell_map(executor, run_cell_task, tasks, max_pending=bound)


def assemble_cell_tables(results, selected) -> dict[str, pd.DataFrame]:
    """Restore canonical cell order before the original concat and global statistics."""
    ordered = [None] * len(selected)
    for result in results:
        index = result["cell_index"]
        if (type(index) is not int or not 0 <= index < len(selected) or ordered[index] is not None
                or result["cell"] != selected[index]):
            raise ValueError("stress worker result has a duplicate or crosswired cell index")
        tables = result["tables"]
        if (not isinstance(tables, dict) or set(tables) != set(CELL_RESULT_TABLES)
                or any(not isinstance(frame, pd.DataFrame) for frame in tables.values())):
            raise ValueError("stress worker result is missing an original row table")
        ordered[index] = tables
    if not ordered or any(tables is None for tables in ordered):
        raise ValueError("stress worker results are missing selected cells")
    return {name: pd.concat([tables[name] for tables in ordered], ignore_index=True)
            for name in CELL_RESULT_TABLES}


def validate_selected_topology(selected_days: pd.DataFrame) -> tuple[tuple[str, int, str], ...]:
    if not set(CELL_COLUMNS).issubset(selected_days.columns) or selected_days.duplicated(list(CELL_COLUMNS)).any():
        raise ValueError("selected G2 cells are missing semantic keys or contain duplicates")
    keys = tuple(selected_days[list(CELL_COLUMNS)].itertuples(index=False, name=None))
    roster = tuple(sorted(selected_days["subject_id"].unique()))
    if (len(keys) != 800 or roster != tuple(f"id{i:02}" for i in range(1, 11))
            or set(selected_days["primitive"]) != set(PRIMITIVE_ORDER)):
        raise ValueError("stress requires exactly 800 selected cells for the frozen 10-participant roster")
    counts = selected_days.groupby(["subject_id", "primitive"]).size()
    if len(counts) != 80 or not counts.eq(10).all():
        raise ValueError("stress requires 10 selected days per participant/primitive")
    return original._select_cells(keys, "canonical")


def _verify_canonical(canonical_root: Path, expected: dict) -> None:
    actual_names = {path.name for path in canonical_root.glob("*_canonical.parquet")}
    if not isinstance(expected, dict) or len(expected) != 12 or actual_names != set(expected):
        raise ValueError("canonical inputs must match all 12 freshly prepared source files")
    for name, record in expected.items():
        path = (canonical_root / name).resolve()
        if not path.is_relative_to(canonical_root) or sha256(path) != record.get("sha256"):
            raise ValueError(f"canonical source changed since primitive preparation: {name}")


def run_stress(prepare_dir: Path, canonical_root: Path, g2_dir: Path, output_dir: Path, *, workers: int = 3) -> dict:
    """Execute all 800 x 50 targets and 80,000 newly calculated stress replays."""
    # Refuse any existing path before loading inputs; never reuse a partial run.
    output = create_fresh_output(output_dir, [prepare_dir, canonical_root, g2_dir, VENDOR])
    started = time.perf_counter()
    manifest = {"schema_name": "independent_etri_reproduction", "stage": "stress",
                "status": "running", "fresh_execution": True, "new_model_fits": 0,
                "python": platform.python_version(), "outputs": {},
                "scheduling": {"workers": workers, "start_method": "spawn" if workers > 1 else "sequential",
                               "max_pending_cells": workers * 2, "result_order": "canonical_cell_index"},
                "parameters": {"root_seed": ROOT_SEED, "draws": DRAWS, "selected_cells": 800,
                    "scenarios": [SCATTERED_SCENARIO, "event_boundary_20pct"],
                    "primary_reference_scenario": "contiguous_20pct",
                    "dose_scenarios": list(DOSE_SCENARIOS)},
                "legacy_schema_note": "Official value-class names and historical_record_deletion_digest "
                    "are retained solely for schema compatibility. Their values bind this NEW G2 run. "
                    "No historical execution authority or release approval is asserted."}
    write_json(output / "manifest.json", manifest)
    try:
        manifest["scientific_sources"] = verify_vendor()
        manifest["adapter_sources"] = [{"path": str(path), "sha256": sha256(path)}
            for path in (Path(__file__).resolve(), PAPER / "full_prepare.py")]
        prepare_dir, canonical_root, g2_dir = (path.resolve(strict=True) for path in
                                               (prepare_dir, canonical_root, g2_dir))
        prepared_manifest = read_upstream_manifest(prepare_dir, "prepare", ("primitives", "baselines", "representations"))
        g2_manifest = read_upstream_manifest(g2_dir, "g2", G2_TABLES)
        prepare_hash, g2_hash = sha256(prepare_dir / "manifest.json"), sha256(g2_dir / "manifest.json")
        canonical_files = prepared_manifest.get("canonical_files")
        _verify_canonical(canonical_root, canonical_files)
        if (g2_manifest.get("inputs", {}).get("prepare_manifest_sha256") != prepare_hash
                or g2_manifest["inputs"].get("canonical_files") != canonical_files):
            raise ValueError("G2 does not bind the supplied fresh prepare/canonical inputs")
        manifest["inputs"] = {"prepare_dir": str(prepare_dir), "canonical_root": str(canonical_root),
            "g2_dir": str(g2_dir), "prepare_manifest_sha256": prepare_hash,
            "g2_manifest_sha256": g2_hash, "canonical_files": canonical_files,
            "prepare_outputs": prepared_manifest["outputs"], "g2_outputs": g2_manifest["outputs"]}
        write_json(output / "manifest.json", manifest)
        frames = {name: pd.read_parquet(g2_dir / g2_manifest["outputs"][name]["path"]) for name in G2_TABLES}
        selected = validate_selected_topology(frames["selected_days"])
        expected_keys = tuple((subject, day, primitive, scenario, draw)
            for subject, day, primitive in selected for scenario in DOSE_SCENARIOS for draw in range(DRAWS))
        for name in ("cell_replays", "deletion_audit"):
            relevant = frames[name].loc[frames[name]["scenario"].isin(DOSE_SCENARIOS)]
            if len(relevant) != len(expected_keys) or set(relevant[list(KEY_COLUMNS)].itertuples(index=False, name=None)) != set(expected_keys):
                raise ValueError(f"{name} dose topology differs from the complete frozen grid")
        emit(stage="stress", step="validate_new_g2", cells=len(selected), evidence_rows=len(expected_keys))
        pairs = build_validated_pairs(frames["cell_replays"], frames["deletion_audit"], frames["mask_intervals"],
            artifact_hashes={name: g2_manifest["outputs"][name]["sha256"] for name in G2_TABLES},
            expected_keys=expected_keys)
        primitive_table = pd.read_parquet(prepare_dir / prepared_manifest["outputs"]["primitives"]["path"])
        if primitive_table["sensor_day_id"].duplicated().any():
            raise ValueError("fresh primitive sensor_day_id is not unique")
        row_by_day = {int(row.sensor_day_id): pd.Series(row._asdict()) for row in primitive_table.itertuples(index=False)}
        artifact_hashes = {name: g2_manifest["outputs"][name]["sha256"]
                           for name in ("cell_replays", "deletion_audit", "mask_intervals")}
        tasks = make_cell_tasks(selected, row_by_day, frames, artifact_hashes)
        del frames
        context = {"prepare_dir": str(prepare_dir), "g2_dir": str(g2_dir), "canonical_root": str(canonical_root),
                   "prepare_manifest_sha256": prepare_hash, "g2_manifest_sha256": g2_hash,
                   "canonical_files": canonical_files, "artifact_hashes": artifact_hashes,
                   "scientific_sources": manifest["scientific_sources"]}
        results = []
        for result in iter_cell_results(tasks, context, workers=workers):
            results.append(result)
            emit(stage="stress", step="replay", subject_id=result["cell"][0], primitive=result["cell"][2],
                 completed_cells=len(results), total_cells=len(selected), cell_index=result["cell_index"],
                 worker_pid=result["worker_pid"], cell_seconds=result["seconds"],
                 worker_peak_working_set_bytes=result["peak_working_set_bytes"], seconds=time.perf_counter() - started)
        tables = assemble_cell_tables(results, selected)
        manifest["scheduling"]["worker_peak_working_set_bytes"] = {
            str(pid): max(result["peak_working_set_bytes"] or 0 for result in results if result["worker_pid"] == pid)
            for pid in sorted({result["worker_pid"] for result in results})}
        contiguous = tables.pop("contiguous_rows")
        random_rows = tables["cell_replays"].loc[tables["cell_replays"]["scenario"].eq(SCATTERED_SCENARIO)].copy()
        primary = pd.concat([contiguous, random_rows], ignore_index=True)
        roster = tuple(sorted({item[0] for item in selected}))
        dose_evidence = tuple(evidence for evidence, _ in pairs.values())
        participant_dose, family_dose = original._dose_response_tables(dose_evidence)
        summary = summarize_stress_results(primary, participant_roster=roster)
        family = summarize_family_inference(summary.participant_deltas)
        matching_complete = bool(tables["deletion_audit"].loc[tables["mask_ledger"]["mask_status"].eq("eligible")]["exact_match"].all())
        decision = evaluate_internal_decision(family, mask_matching_complete=matching_complete,
            denominator_ledger=summary.denominators, dose_response_table=family_dose)
        tables.update(primary_comparison=primary, participant_deltas=summary.participant_deltas,
            denominators=summary.denominators, family_inference=family,
            dose_response_participant=participant_dose, dose_response_family=family_dose,
            decision=pd.DataFrame([{"label": decision.label, "content_digest": decision.content_digest}]))
        for name, frame in tables.items():
            manifest["outputs"][name] = save_frame(output, name, original._storage_frame(frame))
        failures = int(tables["cell_replays"]["status"].isin(["execution_failure", "invalid_contract_or_provenance"]).sum())
        if len(tables["target_ledger"]) != 40000 or len(tables["cell_replays"]) != 80000 or failures or not matching_complete:
            raise ValueError(f"fresh stress replay failed completion checks; failed rows={failures}")
        read_upstream_manifest(prepare_dir, "prepare", ("primitives", "baselines", "representations"))
        read_upstream_manifest(g2_dir, "g2", G2_TABLES)
        if sha256(prepare_dir / "manifest.json") != prepare_hash or sha256(g2_dir / "manifest.json") != g2_hash:
            raise ValueError("upstream manifest changed during stress replay")
        _verify_canonical(canonical_root, canonical_files)
        verify_vendor()
        manifest.update(status="complete", seconds=time.perf_counter() - started,
            decision_label=decision.label, summary_content_digest=summary.content_digest,
            decision_content_digest=decision.content_digest, mask_matching_complete=matching_complete,
            fresh_replay_rows=len(tables["cell_replays"]), fresh_target_rows=len(tables["target_ledger"]))
        write_json(output / "manifest.json", manifest)
        emit(stage="stress", status="complete", seconds=manifest["seconds"], fresh_replay_rows=80000)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", seconds=time.perf_counter() - started,
                        error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", manifest)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, default=Path("/data/output/paper/prepare"))
    parser.add_argument("--canonical-root", type=Path, default=Path("/data/prepared/project/data/canonical"))
    parser.add_argument("--g2-dir", type=Path, default=Path("/data/output/paper/g2"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/output/paper/stress"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    run_stress(args.prepare_dir, args.canonical_root, args.g2_dir, args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()
