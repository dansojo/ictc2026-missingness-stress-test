"""Streaming-friendly replay orchestration for Task 12 record masks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import gc
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from . import primitives
from . import reliability
from .contracts import PRIMITIVE_DEFINITIONS
from .phaseguard_authority import OfficialCellEvidence, OfficialMaskCell
from .stress_contracts import (
    FEATURE_FAMILIES,
    Contiguous20Crosslink,
    MatchedDeletionTarget,
    StressExecutionAuthorityGraph,
    derive_matched_deletion_target,
)
from .stress_masks import (
    BOUNDARY_SCENARIO,
    SCATTERED_SCENARIO,
    MatchedRecordMask,
    generate_event_boundary_record_mask,
    generate_scattered_record_mask,
    record_mask_to_intervals,
)
from .stress_statistics import (
    evaluate_internal_decision,
    summarize_family_inference,
    summarize_stress_results,
)


_FAMILY_BY_PRIMITIVE = {
    primitive: family
    for family, primitives_in_family in FEATURE_FAMILIES
    for primitive in primitives_in_family
}
_PRIMITIVE_ORDER = tuple(
    primitive for _, family in FEATURE_FAMILIES for primitive in family
)
_DEFINITION_BY_PRIMITIVE = {
    definition.name: definition for definition in PRIMITIVE_DEFINITIONS
}


def _sha(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _float_token(value: float) -> str:
    return value.hex() if math.isfinite(value) else "nan"


@dataclass(frozen=True)
class StressReplayTables:
    target_ledger: pd.DataFrame
    mask_ledger: pd.DataFrame
    deletion_audit: pd.DataFrame
    cell_replays: pd.DataFrame
    content_digest: str


@dataclass(frozen=True)
class StressExperimentResults:
    tables: dict[str, pd.DataFrame]
    decision_label: str
    authority_digest: str
    mode: str
    content_digest: str


def _row_status(
    result: primitives.PreparedPrimitiveReplayResult,
    *,
    operation: str,
) -> str:
    if math.isfinite(result.value) and result.quality.get("status") == "observed":
        return "finite"
    if operation == "evening_p90":
        return "abstained_no_event_support"
    return "abstained_insufficient_support"


def run_stress_replays(
    prepared: primitives.PreparedPrimitiveReplay,
    targets: list[MatchedDeletionTarget],
) -> StressReplayTables:
    """Run both frozen Task 12 masks for each target while retaining every row."""
    if type(prepared) is not primitives.PreparedPrimitiveReplay:
        raise TypeError("prepared must be an exact PreparedPrimitiveReplay")
    if type(targets) is not list or not targets:
        raise TypeError("targets must be a nonempty exact list")
    if not all(type(target) is MatchedDeletionTarget for target in targets):
        raise TypeError("targets contain a non-MatchedDeletionTarget value")
    view = primitives.export_prepared_phase_view(prepared)
    ordered_targets = sorted(targets, key=lambda item: item.draw)
    if len({target.draw for target in ordered_targets}) != len(ordered_targets):
        raise ValueError("targets contain a duplicate draw")

    mask_entries: list[tuple[MatchedDeletionTarget, MatchedRecordMask]] = []
    for target in ordered_targets:
        mask_entries.extend(
            (
                (target, generate_scattered_record_mask(view, target)),
                (target, generate_event_boundary_record_mask(view, target)),
            )
        )
    eligible_entries = [entry for entry in mask_entries if entry[1].status == "eligible"]
    result_by_digest: dict[str, primitives.PreparedPrimitiveReplayResult] = {}
    failure_by_digest: dict[str, str] = {}
    if eligible_entries:
        intervals = tuple(
            record_mask_to_intervals(mask) for _, mask in eligible_entries
        )
        try:
            replay_results = primitives.recompute_prepared_primitive_batch(
                prepared, intervals
            )
            result_by_digest = {
                mask.content_digest: result
                for (_, mask), result in zip(
                    eligible_entries, replay_results, strict=True
                )
            }
        except Exception:
            for (target, mask), one_intervals in zip(
                eligible_entries, intervals, strict=True
            ):
                try:
                    result_by_digest[mask.content_digest] = (
                        primitives.recompute_prepared_primitive_batch(
                            prepared, (one_intervals,)
                        )[0]
                    )
                except Exception as exc:
                    failure_by_digest[mask.content_digest] = type(exc).__name__

    target_rows: list[dict[str, object]] = []
    for target in ordered_targets:
        target_rows.append(
            {
                "subject_id": target.subject_id,
                "sensor_day_id": target.sensor_day_id,
                "primitive": target.primitive,
                "draw": target.draw,
                "target_deleted_count": target.target_deleted_count,
                "eligibility_status": target.eligibility_status,
                "reason": target.reason,
                "prepared_source_content_digest": target.prepared_source_content_digest,
                "current_record_deletion_digest": target.current_record_deletion_digest,
                "content_digest": target.content_digest,
            }
        )

    mask_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    for target, mask in mask_entries:
        result = result_by_digest.get(mask.content_digest)
        failure = failure_by_digest.get(mask.content_digest)
        if failure is not None:
            status = "execution_failure"
            value = float("nan")
            deleted_count = 0
            retained_count = 0
            standardized_error = float("nan")
        elif result is None:
            status = "structurally_unavailable"
            value = float("nan")
            deleted_count = 0
            retained_count = 0
            standardized_error = float("nan")
        else:
            deleted_count = result.deleted_count
            retained_count = result.retained_count
            if deleted_count != target.target_deleted_count:
                status = "invalid_contract_or_provenance"
                value = float("nan")
                standardized_error = float("nan")
            else:
                status = _row_status(result, operation=view.operation)
                value = float(result.value)
                standardized_error = (
                    abs(value - target.reference_value) / target.standardizer
                    if status == "finite"
                    else float("nan")
                )
        mask_rows.append(
            {
                "subject_id": target.subject_id,
                "sensor_day_id": target.sensor_day_id,
                "primitive": target.primitive,
                "draw": target.draw,
                "scenario": mask.scenario,
                "mask_status": mask.status,
                "seed_sha256": mask.seed_sha256,
                "target_deleted_count": mask.target_deleted_count,
                "selected_record_count": mask.selected_record_count,
                "record_identity_digests": mask.record_identity_digests,
                "mask_intervals": mask.mask_intervals,
                "content_digest": mask.content_digest,
            }
        )
        audit_rows.append(
            {
                "subject_id": target.subject_id,
                "sensor_day_id": target.sensor_day_id,
                "primitive": target.primitive,
                "draw": target.draw,
                "scenario": mask.scenario,
                "target_deleted_count": target.target_deleted_count,
                "deleted_count": deleted_count,
                "retained_count": retained_count,
                "exact_match": bool(result is not None and deleted_count == target.target_deleted_count),
                "mask_digest": mask.content_digest,
            }
        )
        cell_rows.append(
            {
                "subject_id": target.subject_id,
                "sensor_day_id": target.sensor_day_id,
                "primitive": target.primitive,
                "feature_family": _FAMILY_BY_PRIMITIVE[target.primitive],
                "draw": target.draw,
                "scenario": mask.scenario,
                "status": status,
                "reason": (
                    f"execution_failure:{failure}"
                    if failure is not None
                    else mask.reason if result is None else status
                ),
                "reference_value": target.reference_value,
                "masked_value": value,
                "standardizer": target.standardizer,
                "standardizer_source": target.standardizer_source,
                "standardized_absolute_error": standardized_error,
                "target_deleted_count": target.target_deleted_count,
                "deleted_count": deleted_count,
                "retained_count": retained_count,
                "mask_digest": mask.content_digest,
                "target_digest": target.content_digest,
            }
        )
    target_ledger = pd.DataFrame(target_rows)
    mask_ledger = pd.DataFrame(mask_rows)
    deletion_audit = pd.DataFrame(audit_rows)
    cell_replays = pd.DataFrame(cell_rows)
    digest_payload = [
        "stress-replay-tables-v1",
        [row["content_digest"] for row in target_rows],
        [row["content_digest"] for row in mask_rows],
        [
            [
                row["subject_id"],
                row["sensor_day_id"],
                row["primitive"],
                row["draw"],
                row["scenario"],
                row["status"],
                _float_token(float(row["standardized_absolute_error"])),
            ]
            for row in cell_rows
        ],
    ]
    return StressReplayTables(
        target_ledger=target_ledger,
        mask_ledger=mask_ledger,
        deletion_audit=deletion_audit,
        cell_replays=cell_replays,
        content_digest=_sha(digest_payload),
    )


def _official_key(value: OfficialMaskCell | OfficialCellEvidence) -> tuple[str, int, str, str, int]:
    return (
        value.subject_id,
        value.sensor_day_id,
        value.primitive,
        value.scenario,
        value.draw,
    )


def _select_cells(
    selected: tuple[tuple[str, int, str], ...], mode: str
) -> tuple[tuple[str, int, str], ...]:
    if mode == "canonical":
        return tuple(sorted(selected, key=lambda item: (item[0], _PRIMITIVE_ORDER.index(item[2]), item[1])))
    if mode != "tiny":
        raise ValueError("mode must be tiny or canonical")
    subjects = tuple(sorted({item[0] for item in selected}))[:2]
    chosen: list[tuple[str, int, str]] = []
    for subject in subjects:
        for primitive in _PRIMITIVE_ORDER:
            candidates = [
                item for item in selected if item[0] == subject and item[2] == primitive
            ]
            if not candidates:
                raise ValueError("tiny topology is missing a subject-primitive cell")
            chosen.append(min(candidates, key=lambda item: item[1]))
    return tuple(chosen)


def _prepare_selected_cell(
    primitive: str,
    row: pd.Series,
    *,
    canonical_root: Path,
    sensor_indexes: dict[str, object],
) -> primitives.PreparedPrimitiveReplay:
    definition = _DEFINITION_BY_PRIMITIVE[primitive]
    sensor_file = definition.sensor_file
    indexed = sensor_indexes.get(sensor_file)
    if indexed is None:
        frame = pd.read_parquet(canonical_root / sensor_file)
        indexed = reliability._index_sensor_frame(frame, sensor_file)
        sensor_indexes[sensor_file] = indexed
    calendar = reliability._calendar_row_for_replay(row)
    sensor_slice, _ = reliability._source_frame_slice_indexed(
        primitive, calendar, indexed
    )
    return primitives.prepare_primitive_replay(
        primitive, sensor_slice, sensor_file, calendar
    )


def _contiguous_row(
    evidence: OfficialCellEvidence,
    target: MatchedDeletionTarget,
) -> dict[str, object]:
    finite = evidence.status == "observed" and math.isfinite(
        evidence.standardized_absolute_error
    )
    return {
        "subject_id": target.subject_id,
        "sensor_day_id": target.sensor_day_id,
        "primitive": target.primitive,
        "feature_family": _FAMILY_BY_PRIMITIVE[target.primitive],
        "draw": target.draw,
        "scenario": "contiguous_20pct",
        "status": "finite" if finite else "abstained_insufficient_support",
        "reason": evidence.reason,
        "standardized_absolute_error": (
            float(evidence.standardized_absolute_error) if finite else float("nan")
        ),
    }


def _dose_response_tables(
    evidence: tuple[OfficialCellEvidence, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_rate = {
        "contiguous_10pct": 0.10,
        "contiguous_20pct": 0.20,
        "contiguous_40pct": 0.40,
    }
    rows = [
        {
            "subject_id": item.subject_id,
            "feature_family": _FAMILY_BY_PRIMITIVE[item.primitive],
            "missingness_rate": scenario_rate[item.scenario],
            "error": (
                float(item.standardized_absolute_error)
                if item.status == "observed"
                and math.isfinite(item.standardized_absolute_error)
                else float("nan")
            ),
            "reason": (
                "finite" if item.status == "observed"
                and math.isfinite(item.standardized_absolute_error)
                else f"{item.status}:{item.reason}"
            ),
        }
        for item in evidence
        if item.scenario in scenario_rate
    ]
    frame = pd.DataFrame(rows)
    participant_rows: list[dict[str, object]] = []
    roster = tuple(sorted(frame["subject_id"].unique()))
    for subject_id in roster:
        for family in tuple(name for name, _ in FEATURE_FAMILIES):
            for rate in (0.10, 0.20, 0.40):
                group = frame.loc[
                    frame["subject_id"].eq(subject_id)
                    & frame["feature_family"].eq(family)
                    & frame["missingness_rate"].eq(rate)
                ]
                finite = pd.to_numeric(group["error"], errors="coerce")
                finite = finite[np.isfinite(finite)]
                reason_counts = tuple(
                    (str(reason), int(count))
                    for reason, count in group["reason"].value_counts().sort_index().items()
                )
                participant_rows.append({
                    "subject_id": subject_id,
                    "feature_family": family,
                    "missingness_rate": rate,
                    "total_cell_count": int(len(group)),
                    "finite_cell_count": int(len(finite)),
                    "nonfinite_cell_count": int(len(group) - len(finite)),
                    "reason_counts": reason_counts,
                    "reporting_status": "eligible" if len(finite) else "ineligible",
                    "participant_median": float(finite.median()) if len(finite) else float("nan"),
                })
    participant = pd.DataFrame(participant_rows)
    family_rows: list[dict[str, object]] = []
    for family in tuple(name for name, _ in FEATURE_FAMILIES):
        for rate in (0.10, 0.20, 0.40):
            group = participant.loc[
                participant["feature_family"].eq(family)
                & participant["missingness_rate"].eq(rate)
            ]
            values = pd.to_numeric(group["participant_median"], errors="coerce")
            finite = values[np.isfinite(values)]
            family_rows.append({
                "feature_family": family,
                "missingness_rate": rate,
                "participant_denominator": len(roster),
                "eligible_participant_count": int(len(finite)),
                "nonfinite_participant_count": int(len(roster) - len(finite)),
                "participant_median": float(finite.median()) if len(finite) else float("nan"),
            })
    family = pd.DataFrame(family_rows)
    return participant, family


def run_authority_stress_experiment(
    graph: StressExecutionAuthorityGraph,
    *,
    mode: str,
) -> StressExperimentResults:
    """Run the frozen Task 12 ETRI grid without labels, repair, or tuning."""
    if type(graph) is not StressExecutionAuthorityGraph:
        raise TypeError("graph must be an exact StressExecutionAuthorityGraph")
    official = graph.official_g2
    calibration = graph.calibration_input
    authority_digest = graph.preregistration.content_digest
    draws = tuple(range(2)) if mode == "tiny" else tuple(range(50))
    selected_all = tuple(
        (item.subject_id, item.sensor_day_id, item.primitive)
        for item in official.selected_cells
    )
    selected = _select_cells(selected_all, mode)
    selected_set = set(selected)
    masks = {
        _official_key(item): item
        for item in official.official_masks
        if item.scenario == "contiguous_20pct"
        and item.draw in draws
        and (item.subject_id, item.sensor_day_id, item.primitive) in selected_set
    }
    primary_evidence = {
        _official_key(item): item
        for item in official.cell_evidence
        if item.scenario == "contiguous_20pct"
        and item.draw in draws
        and (item.subject_id, item.sensor_day_id, item.primitive) in selected_set
    }
    dose_evidence = tuple(
        item
        for item in official.cell_evidence
        if item.scenario in ("contiguous_10pct", "contiguous_20pct", "contiguous_40pct")
    )
    expected = len(selected) * len(draws)
    if len(masks) != expected or len(primary_evidence) != expected:
        raise ValueError("official contiguous20 topology is incomplete")
    del official
    del graph
    gc.collect()

    primitive_table = pd.read_parquet(calibration.primitive_table.path)
    row_by_day = {
        int(row.sensor_day_id): pd.Series(row._asdict())
        for row in primitive_table.itertuples(index=False)
    }
    sensor_indexes: dict[str, object] = {}
    target_frames: list[pd.DataFrame] = []
    mask_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    replay_frames: list[pd.DataFrame] = []
    contiguous_rows: list[dict[str, object]] = []
    crosslink_rows: list[dict[str, object]] = []
    for subject_id, sensor_day_id, primitive in selected:
        row = row_by_day[sensor_day_id]
        if row["subject_id"] != subject_id:
            raise ValueError("selected cell is crosswired to the primitive table")
        prepared = _prepare_selected_cell(
            primitive,
            row,
            canonical_root=calibration.canonical_root,
            sensor_indexes=sensor_indexes,
        )
        targets: list[MatchedDeletionTarget] = []
        for draw in draws:
            key = (subject_id, sensor_day_id, primitive, "contiguous_20pct", draw)
            mask = masks[key]
            evidence = primary_evidence[key]
            intervals = tuple(
                (interval.mask_start, interval.mask_end) for interval in mask.intervals
            )
            crosslink = Contiguous20Crosslink(
                subject_id=subject_id,
                sensor_day_id=sensor_day_id,
                primitive=primitive,
                draw=draw,
                deletion_audit_key=mask.deletion_audit.deletion_audit_key,
                official_deleted_count=mask.deletion_audit.deleted_count,
                official_retained_count=mask.deletion_audit.retained_count,
                official_record_deletion_digest=mask.deletion_audit.record_deletion_digest,
                intervals=intervals,
                reference_value=float(evidence.original_value),
                reference_status=evidence.original_status,
                standardizer=float(evidence.standardizer),
                standardizer_source=evidence.standardizer_source,
            )
            target = derive_matched_deletion_target(prepared, crosslink)
            targets.append(target)
            crosslink_rows.append({
                "subject_id": subject_id,
                "sensor_day_id": sensor_day_id,
                "primitive": primitive,
                "scenario": "contiguous_20pct",
                "draw": draw,
                "deletion_audit_key": crosslink.deletion_audit_key,
                "official_original_count": (
                    crosslink.official_deleted_count
                    + crosslink.official_retained_count
                ),
                "official_deleted_count": crosslink.official_deleted_count,
                "official_retained_count": crosslink.official_retained_count,
                "historical_record_deletion_digest": (
                    crosslink.official_record_deletion_digest
                ),
                "current_record_deletion_digest": (
                    target.current_record_deletion_digest
                ),
                "prepared_source_content_digest": (
                    target.prepared_source_content_digest
                ),
                "target_valid_deleted_count": target.target_deleted_count,
                "crosslink_content_digest": crosslink.content_digest,
                "target_content_digest": target.content_digest,
            })
            contiguous_rows.append(_contiguous_row(evidence, target))
        replay = run_stress_replays(prepared, targets)
        target_frames.append(replay.target_ledger)
        compact_masks = replay.mask_ledger.copy()
        compact_masks["record_identity_count"] = compact_masks[
            "record_identity_digests"
        ].map(len)
        compact_masks["interval_count"] = compact_masks["mask_intervals"].map(len)
        compact_masks = compact_masks.drop(
            columns=["record_identity_digests", "mask_intervals"]
        )
        mask_frames.append(compact_masks)
        audit_frames.append(replay.deletion_audit)
        replay_frames.append(replay.cell_replays)

    target_ledger = pd.concat(target_frames, ignore_index=True)
    mask_ledger = pd.concat(mask_frames, ignore_index=True)
    deletion_audit = pd.concat(audit_frames, ignore_index=True)
    cell_replays = pd.concat(replay_frames, ignore_index=True)
    random_rows = cell_replays.loc[
        cell_replays["scenario"].eq(SCATTERED_SCENARIO)
    ].copy()
    primary = pd.concat([pd.DataFrame(contiguous_rows), random_rows], ignore_index=True)
    roster = tuple(sorted({item[0] for item in selected_all}))
    participant_dose, family_dose = _dose_response_tables(dose_evidence)
    if mode == "canonical":
        summary = summarize_stress_results(primary, participant_roster=roster)
        family = summarize_family_inference(summary.participant_deltas)
        matching_complete = bool(
            deletion_audit.loc[
                mask_ledger["mask_status"].eq("eligible")
            ]["exact_match"].all()
        )
        decision = evaluate_internal_decision(
            family,
            mask_matching_complete=matching_complete,
            denominator_ledger=summary.denominators,
            dose_response_table=family_dose,
        )
        participant_deltas = summary.participant_deltas
        denominators = summary.denominators
        decision_table = pd.DataFrame([{"label": decision.label, "content_digest": decision.content_digest}])
        label = decision.label
    else:
        family = pd.DataFrame()
        participant_deltas = pd.DataFrame()
        denominators = pd.DataFrame()
        decision_table = pd.DataFrame([{"label": "tiny_nonproduction", "content_digest": ""}])
        label = "tiny_nonproduction"
    tables = {
        "target_ledger": target_ledger,
        "official_current_crosslink": pd.DataFrame(crosslink_rows),
        "mask_ledger": mask_ledger,
        "deletion_audit": deletion_audit,
        "cell_replays": cell_replays,
        "primary_comparison": primary,
        "participant_deltas": participant_deltas,
        "denominators": denominators,
        "family_inference": family,
        "dose_response_participant": participant_dose,
        "dose_response_family": family_dose,
        "decision": decision_table,
    }
    digest = _sha([
        "stress-experiment-results-v1", mode, authority_digest,
        len(selected), len(draws), len(cell_replays), label,
    ])
    return StressExperimentResults(
        tables=tables,
        decision_label=label,
        authority_digest=authority_digest,
        mode=mode,
        content_digest=digest,
    )


def _strict_json_value(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if not all(type(key) is str for key in value):
            raise TypeError("strict JSON mappings require exact string keys")
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict_json_value(item) for item in value]
    raise TypeError(f"unsupported strict JSON value: {type(value).__name__}")


def _storage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    stored = frame.copy()
    for column in stored.columns:
        if stored[column].dtype == object and stored[column].map(
            lambda value: isinstance(value, (tuple, list, dict))
        ).any():
            stored[column] = stored[column].map(
                lambda value: json.dumps(
                    _strict_json_value(value), ensure_ascii=False,
                    allow_nan=False, separators=(",", ":"),
                ) if isinstance(value, (tuple, list, dict)) else value
            )
    return stored


def persist_stress_experiment(
    result: StressExperimentResults,
    output_dir: Path,
    *,
    release_seal_sha256: str | None = None,
) -> Path:
    """Persist a complete result atomically with strict JSON and physical hashes."""
    if type(result) is not StressExperimentResults or not isinstance(output_dir, Path):
        raise TypeError("result and output_dir types are invalid")
    if release_seal_sha256 is not None and (
        type(release_seal_sha256) is not str
        or len(release_seal_sha256) != 64
        or any(character not in "0123456789abcdef" for character in release_seal_sha256)
    ):
        raise TypeError("release_seal_sha256 must be a lowercase SHA-256 or None")
    if (result.mode == "canonical") != (release_seal_sha256 is not None):
        raise ValueError("canonical mode and release seal binding must agree")
    target = output_dir.resolve(strict=False)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.partial-{uuid.uuid4().hex}"
    partial.mkdir()
    try:
        artifacts: list[dict[str, object]] = []
        for name, frame in result.tables.items():
            path = partial / f"{name}.parquet"
            _storage_frame(frame).to_parquet(path, index=False)
            raw = path.read_bytes()
            artifacts.append({
                "name": name,
                "path": path.name,
                "rows": len(frame),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
        manifest = {
            "schema_name": "structured-missingness-etri-output-v1",
            "mode": result.mode,
            "production_eligible": result.mode == "canonical",
            "decision_label": result.decision_label,
            "authority_digest": result.authority_digest,
            "result_content_digest": result.content_digest,
            "release_seal_sha256": release_seal_sha256,
            "artifacts": artifacts,
        }
        rendered = json.dumps(
            _strict_json_value(manifest), ensure_ascii=False, allow_nan=False,
            sort_keys=True, indent=2,
        ).encode("utf-8") + b"\n"
        (partial / "manifest.json").write_bytes(rendered)
        partial.rename(target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target
