"""Fresh 160-fit ETRI downstream reproduction from independently generated tables.

The table adapter calculation is copied from the hash-pinned
semantic_indicators/stress_downstream_runner.py:677-937. Its original historical
loader is replaced with a separately verified independent stage boundary. Pure
row/value validators and model/statistics calculations remain the original code.
Legacy ``official``/``authority`` field names identify content seals here; they
never assert historical release approval. No archived fit or prediction is read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import struct
import threading
import time
import traceback

import pandas as pd
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER / 'full_vendor'))
sys.path.insert(0, str(PAPER))
from semantic_indicators import outcome, stress_downstream
from semantic_indicators import stress_downstream_runner as runner
from semantic_indicators.stress_downstream_runner import (
    DownstreamAdapterSnapshot, _selected_rows_and_ledgers, _FAMILY_BY_PRIMITIVE,
    _FAMILIES, _PRIMITIVES, _PRIMITIVE_FAMILY_PAIRS, _PREDICTION_UNIVERSE_COLUMNS,
    _CONTIGUOUS_COLUMNS, _make_contexts, _validated_evidence_tuple, _sha,
    _same_float_bits, _make_perturbation, _row_digest, _projection_rows,
    _adapter_content_digest, validate_downstream_adapter_snapshot,
)

TABLE_NAMES = ('target_ledger', 'official_current_crosslink', 'mask_ledger',
               'deletion_audit', 'cell_replays')
G2_NAMES = ('cell_replays', 'deletion_audit', 'mask_intervals')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def create_fresh_output(output_dir: Path, inputs: list[Path]) -> Path:
    root = Path(output_dir).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if root == source or root in source.parents or source in root.parents:
            raise ValueError('output overlaps input')
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    return root


def load_stage(root: Path, stage: str, names: tuple[str, ...]) -> tuple[dict, dict[str, pd.DataFrame]]:
    root = Path(root).resolve()
    document = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if (document.get('schema_name') != 'independent_etri_reproduction'
            or document.get('stage') != stage or document.get('status') != 'complete'
            or document.get('fresh_execution') is not True):
        raise ValueError(f'{stage} requires a complete fresh independent stage')
    frames = {}
    for name in names:
        pin = document['outputs'][name]
        path = (root / pin['path']).resolve()
        if root not in path.parents:
            raise ValueError('stage artifact escapes its output root')
        if sha256(path) != pin['sha256']:
            raise ValueError(f'{stage}/{name} digest mismatch')
        frame = pd.read_parquet(path)
        if len(frame) != pin['rows'] or list(frame.columns) != pin['columns']:
            raise ValueError(f'{stage}/{name} table shape mismatch')
        frames[name] = frame
    return document, frames


def load_labels(path: Path) -> pd.DataFrame:
    keys = ['subject_id', 'sleep_date', 'lifelog_date']
    targets = list(outcome.TARGET_COLUMNS)
    frame = pd.read_csv(path)
    if not set(keys + targets).issubset(frame.columns):
        raise ValueError('labels lack subject/date/S1-S4 columns')
    frame = frame.loc[:, keys + targets].copy()
    if frame.isna().any().any():
        raise ValueError('labels contain null values')
    if not frame['subject_id'].map(lambda value: type(value) is str and bool(value)).all():
        raise ValueError('label subject identity is not exact text')
    frame['subject_id'] = pd.Series(frame['subject_id'], dtype='string')
    for name in ('lifelog_date', 'sleep_date'):
        frame[name] = pd.to_datetime(frame[name], errors='raise').astype('datetime64[ns]')
        if not frame[name].eq(frame[name].dt.normalize()).all():
            raise ValueError('label dates must be local midnight')
    if not frame['sleep_date'].eq(frame['lifelog_date'] + pd.Timedelta(days=1)).all():
        raise ValueError('label sleep/lifelog date relation changed')
    if frame.duplicated(keys).any():
        raise ValueError('duplicate label key')
    for name in targets:
        numeric = pd.to_numeric(frame[name], errors='raise')
        if not numeric.isin((0, 1)).all():
            raise ValueError('labels must be binary')
        frame[name] = numeric.astype('int64')
    return frame


def count_complete_fit_records(path: Path) -> int:
    """Observe only whole records in a newly written pack; never load a model."""
    count = 0
    with path.open('rb') as stream:
        if stream.read(len(runner._PACK_HEADER)) != runner._PACK_HEADER:
            raise ValueError('new fit pack header mismatch')
        size = path.stat().st_size
        while stream.tell() + 8 <= size:
            length = struct.unpack('>Q', stream.read(8))[0]
            if stream.tell() + length > size:
                break
            stream.seek(length, 1)
            count += 1
    return count


def validate_fit_topology(ledger: pd.DataFrame) -> None:
    if len(ledger) != 160:
        raise ValueError('fit topology requires 160 newly written fits')
    subjects = tuple(sorted(ledger['held_out_subject'].unique()))
    expected = [(fold, subject, arm, model, target)
                for fold, subject in enumerate(subjects)
                for arm in outcome.DOWNSTREAM_ARM_NAMES
                for model in outcome.MODEL_NAMES for target in outcome.TARGET_COLUMNS]
    columns = ['fold', 'held_out_subject', 'arm', 'model_name', 'target']
    if (len(subjects) != 10 or ledger['fit_seq'].tolist() != list(range(160))
            or list(ledger[columns].itertuples(index=False, name=None)) != expected):
        raise ValueError('new fit ledger topology is incomplete or crosswired')


def validate_full_stress_grid(target: pd.DataFrame) -> None:
    """The complete study has 800 selected cells, each with all 50 draws."""
    columns = ['subject_id', 'primitive', 'sensor_day_id', 'draw']
    if (len(target) != 40000 or target['subject_id'].nunique() != 10
            or set(target['primitive']) != set(runner._PRIMITIVES)
            or target.duplicated(columns).any()):
        raise ValueError('full stress grid is incomplete or duplicated')
    counts = target.groupby(['subject_id', 'primitive'], sort=False)['sensor_day_id'].nunique()
    if len(counts) != 80 or not counts.eq(10).all():
        raise ValueError('full stress grid requires 10 days per participant/primitive')
    draws = target.groupby(columns[:-1], sort=False)['draw'].agg(lambda values: tuple(sorted(values)))
    if len(draws) != 800 or not draws.map(lambda values: values == tuple(range(50))).all():
        raise ValueError('full stress grid must retain all 50 draws per selected cell')


def validate_study_labels(training: pd.DataFrame, scoring: pd.DataFrame) -> None:
    if (len(training) != 450 or training['subject_id'].nunique() != 10
            or len(scoring) != 450 or scoring['subject_id'].nunique() != 10):
        raise ValueError('frozen paper labels require 450 rows and 10 subjects')
    try:
        pd.testing.assert_frame_equal(training, scoring, check_exact=True)
    except AssertionError as error:
        raise ValueError('paper scoring labels must match training labels exactly') from error


def validate_prediction_count(row_count: int, selected_target_count: int) -> None:
    if row_count != 943392:
        raise ValueError('complete study requires exactly 943392 predictions')
    if row_count != 48 * selected_target_count:
        raise ValueError('observed prediction topology is incomplete')


def build_independent_adapter_snapshot(
    frames: dict[str, pd.DataFrame], official_cell_evidence: tuple,
    primitives: pd.DataFrame, *, expected_task12_authority_digest: str,
    expected_official_g2_authority_digest: str,
    expected_interval_boundary_policy: str, expected_primitive_digest: str,
) -> DownstreamAdapterSnapshot:
    """Use NEW stage content digests, never a historical replay authority graph."""
    if type(primitives) is not pd.DataFrame:
        raise TypeError('primitives must be an exact DataFrame')
    expected_task12 = runner._require_digest(expected_task12_authority_digest, 'new stress manifest digest')
    official_root = runner._require_digest(expected_official_g2_authority_digest, 'new G2 manifest digest')
    runner._require_digest(expected_primitive_digest, 'new primitive digest')
    stress_downstream._validate_topology(frames)
    target = frames['target_ledger']
    crosslink = frames['official_current_crosslink']
    replay = frames['cell_replays']
    # SOURCE-DERIVED CALCULATION START: stress_downstream_runner.py:677-937
    selected, ineligible, excluded = _selected_rows_and_ledgers(target, primitives)
    subjects = tuple(sorted({str(row["subject_id"]) for row in selected}))
    if len(subjects) != 10:
        raise ValueError("downstream adapter requires exactly ten participants")
    fold_roster = tuple((position, subject) for position, subject in enumerate(subjects))
    fold_by_subject = {subject: fold for fold, subject in fold_roster}
    prediction_records: list[dict[str, object]] = []
    for row in selected:
        subject = str(row["subject_id"])
        primitive = str(row["primitive"])
        family = _FAMILY_BY_PRIMITIVE[primitive]
        prediction_records.append({
            "participant_position": subjects.index(subject), "participant": subject,
            "fold": fold_by_subject[subject], "held_out_subject": subject,
            "family_position": _FAMILIES.index(family), "family": family,
            "primitive_position": _PRIMITIVES.index(primitive), "primitive": primitive,
            "sensor_day_id": int(row["sensor_day_id"]), "draw": int(row["draw"]),
            "lifelog_date": pd.Timestamp(row["lifelog_date"]),
            "sleep_date": pd.Timestamp(row["sleep_date"]),
            "elapsed_day_index": int(row["elapsed_day_index"]),
            "target_content_digest": str(row["target_content_digest"]),
        })
    universe = pd.DataFrame.from_records(
        prediction_records, columns=_PREDICTION_UNIVERSE_COLUMNS
    ).sort_values(
        ["participant_position", "family_position", "primitive_position",
         "sensor_day_id", "draw"], kind="stable"
    ).reset_index(drop=True)
    cells = set(zip(universe["participant"], universe["family"], strict=True))
    if cells != {(subject, family) for subject in subjects for family in _FAMILIES}:
        raise ValueError("every participant/family cell must be nonempty")
    group_sizes = universe.groupby(["participant", "family"], sort=False).size()
    max_primary_keys = int(group_sizes.max())

    reference_keys = (
        universe.loc[:, ["sensor_day_id", "participant", "lifelog_date", "sleep_date",
                         "elapsed_day_index"]]
        .rename(columns={"participant": "subject_id"})
        .drop_duplicates()
        .sort_values(["subject_id", "sensor_day_id"], kind="stable")
        .reset_index(drop=True)
    )
    reference_keys["sensor_day_id"] = reference_keys["sensor_day_id"].astype("int64")
    reference_keys["subject_id"] = reference_keys["subject_id"].astype("str")
    reference_keys["lifelog_date"] = pd.to_datetime(
        reference_keys["lifelog_date"]
    ).astype("datetime64[ns]")
    reference_keys["sleep_date"] = pd.to_datetime(
        reference_keys["sleep_date"]
    ).astype("datetime64[ns]")
    reference_keys["elapsed_day_index"] = reference_keys["elapsed_day_index"].astype("int64")
    reference_tables = outcome.build_downstream_reference_frames(
        primitives.copy(deep=True), reference_keys,
        expected_interval_boundary_policy=expected_interval_boundary_policy,
        expected_primitive_digest=expected_primitive_digest,
    )
    reference_digest = reference_tables.content_digest
    outcome.validate_downstream_reference_frames(
        reference_tables, expected_content_digest=reference_digest
    )
    contexts = _make_contexts(reference_tables, fold_roster)
    context_by_key = {
        (value.held_out_subject, value.sensor_day_id): value for value in contexts
    }

    evidence = _validated_evidence_tuple(official_cell_evidence)
    required_evidence_keys = {
        (str(row["participant"]), int(row["sensor_day_id"]), str(row["primitive"]),
         "contiguous_20pct", int(row["draw"]))
        for row in universe.to_dict("records")
    }
    evidence_keys = {
        (value.subject_id, value.sensor_day_id, value.primitive, value.scenario, value.draw)
        for value in evidence
    }
    if evidence_keys != required_evidence_keys or len(evidence) != len(required_evidence_keys):
        raise ValueError("Official-cell evidence keys do not exactly match selected targets")
    evidence_projection = _sha([
        "task-12-downstream-selected-official-cell-evidence-v1", official_root,
        [[value.subject_id, value.sensor_day_id, value.primitive, value.scenario,
          value.draw, value.content_digest] for value in evidence],
    ])
    evidence_by_key = {
        (value.subject_id, value.sensor_day_id, value.primitive, value.scenario, value.draw): value
        for value in evidence
    }

    target_by_key = {
        (str(row["subject_id"]), int(row["sensor_day_id"]), str(row["primitive"]),
         int(row["draw"])): row
        for row in target.to_dict("records")
    }
    cross_by_key = {
        (str(row["subject_id"]), int(row["sensor_day_id"]), str(row["primitive"]),
         int(row["draw"])): row
        for row in crosslink.to_dict("records")
    }
    replay_by_key = {
        (str(row["subject_id"]), int(row["sensor_day_id"]), str(row["primitive"]),
         int(row["draw"]), str(row["scenario"])): row
        for row in replay.to_dict("records")
    }
    primitive_lookup = primitives.set_index("sensor_day_id", drop=False)
    perturbations: list[stress_downstream.DownstreamPerturbation] = []
    patches: list[stress_downstream.DownstreamFeaturePatch] = []
    provenance: list[dict[str, object]] = []
    for universe_row in universe.to_dict("records"):
        subject = str(universe_row["participant"])
        sensor_day_id = int(universe_row["sensor_day_id"])
        primitive = str(universe_row["primitive"])
        draw = int(universe_row["draw"])
        fold = int(universe_row["fold"])
        base_key = (subject, sensor_day_id, primitive, draw)
        target_row = target_by_key[base_key]
        cross_row = cross_by_key[base_key]
        if (
            cross_row["target_content_digest"] != target_row["content_digest"]
            or cross_row["scenario"] != "contiguous_20pct"
        ):
            raise ValueError("contiguous target/crosslink provenance is crosswired")
        evidence_row = evidence_by_key[(*base_key[:3], "contiguous_20pct", draw)]
        for name, evidence_value, cross_name in (
            ("deletion audit key", evidence_row.deletion_audit_key, "deletion_audit_key"),
            ("record deletion digest", evidence_row.record_deletion_digest,
             "historical_record_deletion_digest"),
        ):
            if cross_name not in cross_row or cross_row[cross_name] != evidence_value:
                raise ValueError(f"contiguous {name} provenance is crosswired")
        calendar = primitive_lookup.loc[sensor_day_id]
        if (
            evidence_row.subject_id != subject
            or evidence_row.lifelog_date != pd.Timestamp(calendar["lifelog_date"])
            or evidence_row.elapsed_day_index != int(calendar["elapsed_day_index"])
        ):
            raise ValueError("Official-cell evidence calendar identity is crosswired")
        random_row = replay_by_key[(*base_key, "scattered_random_20pct")]
        event_row = replay_by_key[(*base_key, "event_boundary_20pct")]
        for source in (random_row, event_row):
            if source["target_digest"] != target_row["content_digest"]:
                raise ValueError("Task 12 replay target provenance is crosswired")
        if all(name in random_row and name in event_row for name in
               ("reference_value", "standardizer", "standardizer_source")):
            reference_values = (float(random_row["reference_value"]),
                                float(event_row["reference_value"]),
                                float(evidence_row.original_value),
                                float(calendar[primitive]))
            if not all(_same_float_bits(reference_values[0], value)
                       for value in reference_values[1:]):
                raise ValueError("contiguous reference value provenance is crosswired")
            if (
                not _same_float_bits(float(random_row["standardizer"]),
                                     float(event_row["standardizer"]))
                or not _same_float_bits(float(random_row["standardizer"]),
                                        evidence_row.standardizer)
                or random_row["standardizer_source"] != event_row["standardizer_source"]
                or random_row["standardizer_source"] != evidence_row.standardizer_source
            ):
                raise ValueError("contiguous standardizer provenance is crosswired")

        sources = (
            ("scattered_random_20pct", str(random_row["status"]),
             str(random_row["reason"]), float(random_row["masked_value"])),
            (
                "contiguous_20pct",
                "finite" if evidence_row.retained else (
                    "abstained_no_event_support" if primitive in
                    ("screen_disengagement_p90", "usage_disengagement_p90")
                    else "abstained_insufficient_support"
                ),
                "finite" if evidence_row.retained else (
                    "abstained_no_event_support" if primitive in
                    ("screen_disengagement_p90", "usage_disengagement_p90")
                    else "abstained_insufficient_support"
                ),
                float(evidence_row.masked_value) if evidence_row.retained else float("nan"),
            ),
            ("event_boundary_20pct", str(event_row["status"]),
             str(event_row["reason"]), float(event_row["masked_value"])),
        )
        context = context_by_key[(subject, sensor_day_id)]
        for condition, status, reason, masked_value in sources:
            perturbation = _make_perturbation(
                fold=fold, subject=subject, sensor_day_id=sensor_day_id,
                primitive=primitive, draw=draw, condition=condition, status=status,
                reason=reason, masked_value=masked_value,
            )
            perturbations.append(perturbation)
            for arm in outcome.DOWNSTREAM_ARM_NAMES:
                patches.append(stress_downstream.build_downstream_feature_patch(
                    context, perturbation, arm
                ))
        contiguous = perturbations[-2]
        provenance_row = {
            "row_seq": len(provenance), "subject_id": subject,
            "sensor_day_id": sensor_day_id, "primitive": primitive, "draw": draw,
            "scenario": "contiguous_20pct",
            "target_content_digest": str(target_row["content_digest"]),
            "crosslink_content_digest": str(cross_row["crosslink_content_digest"]),
            "deletion_audit_key": evidence_row.deletion_audit_key,
            "deletion_audit_digest": evidence_row.deletion_audit_digest,
            "record_deletion_digest": evidence_row.record_deletion_digest,
            "official_g2_authority_digest": official_root,
            "official_evidence_digest": evidence_row.content_digest,
            "official_status": evidence_row.status,
            "official_original_status": evidence_row.original_status,
            "official_masked_status": evidence_row.masked_status,
            "official_reason": evidence_row.reason, "retained": evidence_row.retained,
            "normalized_replay_status": contiguous.replay_status,
            "normalized_reason": contiguous.reason,
        }
        provenance_row["content_digest"] = _row_digest(
            "task-12-downstream-contiguous-provenance-row-v1", provenance_row,
            _CONTIGUOUS_COLUMNS[:-1],
        )
        provenance.append(provenance_row)

    contiguous_provenance = pd.DataFrame.from_records(
        provenance, columns=_CONTIGUOUS_COLUMNS
    )
    patch_tuple = tuple(patches)
    perturbation_tuple = tuple(perturbations)
    projection, primary_count, primary_digest = _projection_rows(
        universe, contexts, patch_tuple
    )
    n = len(universe)
    d = len(reference_keys)
    if (
        len(contexts) != d or len(reference_tables.primitive_basis) != 15 * d
        or len(reference_tables.heldout_baselines) != 80
        or len(perturbation_tuple) != 3 * n or len(patch_tuple) != 6 * n
        or primary_count != 16 * n or projection.row_count != 48 * n
    ):
        raise ValueError("downstream adapter topology is incomplete")
    values = {
        "task12_authority_digest": expected_task12,
        "official_g2_authority_digest": official_root,
        "official_cell_evidence_row_count": len(evidence),
        "official_cell_evidence_projection_digest": evidence_projection,
        "fold_roster": fold_roster,
        "primitive_family_pairs": _PRIMITIVE_FAMILY_PAIRS,
        "eligible_target_count": n, "reference_key_count": d,
        "max_primary_keys_per_group": max_primary_keys,
        "prediction_universe": universe, "ineligible_target_ledger": ineligible,
        "window_exclusion_ledger": excluded,
        "contiguous_provenance": contiguous_provenance,
        "reference_tables": reference_tables,
        "reference_content_digest": reference_digest, "contexts": contexts,
        "perturbations": perturbation_tuple, "patches": patch_tuple,
        "expected_primary_key_count": primary_count,
        "expected_primary_key_digest": primary_digest,
        "source_patch_projection": projection,
    }
    unsealed = DownstreamAdapterSnapshot(**values, content_digest="0" * 64)
    result = DownstreamAdapterSnapshot(
        **values, content_digest=_adapter_content_digest(unsealed)
    )
    validate_downstream_adapter_snapshot(
        result, expected_content_digest=result.content_digest,
        expected_source_patch_projection_digest=projection.logical_digest,
    )
    return result
    # SOURCE-DERIVED CALCULATION END


def _progress_snapshot(root: Path, started: float, step: str) -> dict:
    state = {'stage': 'downstream', 'step': step, 'status': 'running',
             'seconds': time.perf_counter() - started,
             'updated_utc': datetime.now(timezone.utc).isoformat(),
             'persisted_new_fit_records': 0, 'persisted_prediction_batches': 0,
             'persisted_prediction_rows': 0}
    # Separate observations of sealed worker batches; these are not yet the
    # canonically joined, independently validated final prediction stream.
    state['completed_prediction_workers'] = 0
    state['worker_persisted_prediction_rows'] = 0
    work = root / 'participant_prediction_work'
    if work.is_dir():
        for path in sorted(work.glob('participant_??.arrow.json')):
            try:
                result = json.loads(path.read_text(encoding='utf-8'))
                if result.get('status') == 'complete' and type(result.get('rows')) is int:
                    state['completed_prediction_workers'] += 1
                    state['worker_persisted_prediction_rows'] += result['rows']
            except (OSError, ValueError):
                pass
    candidates = list(root.glob('run.*.partial'))
    if (root / 'run').is_dir():
        candidates.append(root / 'run')
    if len(candidates) > 1:
        raise ValueError('multiple downstream partial outputs')
    if candidates:
        result_root = candidates[0]
        pack = result_root / 'fit' / 'fit_states.pack'
        if pack.is_file():
            state['persisted_new_fit_records'] = count_complete_fit_records(pack)
        observed = result_root / 'observed_predictions.arrow'
        if observed.is_file():
            try:
                with observed.open('rb') as stream:
                    for batch in ipc.open_stream(stream):
                        state['persisted_prediction_batches'] += 1
                        state['persisted_prediction_rows'] += batch.num_rows
            except (OSError, ValueError):
                pass  # A writer may be between a batch header and its body.
    return state


def select_snapshot_runner(workers: int):
    from full_downstream_parallel import validate_worker_count, run_downstream_snapshot_parallel
    validate_worker_count(workers)
    if workers == 1:
        return runner.run_downstream_snapshot, {}
    return run_downstream_snapshot_parallel, {'workers': workers}


def run_downstream(prepare_dir: Path, stress_dir: Path, training_labels: Path,
                   scoring_labels: Path, output_dir: Path, *, workers: int = 4) -> dict:
    """Execute all 160 new fits and all selected-condition predictions once."""
    from full_prepare import verify_vendor
    from full_stress import build_validated_pairs

    snapshot_runner, scheduling_arguments = select_snapshot_runner(workers)
    source_pins = verify_vendor()
    execution_sources = {path.name: sha256(path) for path in
                         (Path(__file__), PAPER / 'full_downstream_parallel.py')}
    prepare_dir, stress_dir = Path(prepare_dir).resolve(), Path(stress_dir).resolve()
    training_labels, scoring_labels = Path(training_labels).resolve(), Path(scoring_labels).resolve()
    output = create_fresh_output(output_dir, [prepare_dir, stress_dir, training_labels,
                                               scoring_labels, PAPER / 'full_vendor'])
    started = time.perf_counter()
    manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'downstream',
                'status': 'running', 'fresh_execution': True, 'new_model_fits': 0,
                'reused_historical_model_fits': 0, 'outputs': {},
                'scientific_sources': source_pins,
                'execution': {'prediction_workers': workers, 'driver_sha256': execution_sources,
                              'scheduling': 'original_sequential' if workers == 1 else 'participant_processes',
                              'fit_and_statistics_execution': 'original_sequential'},
                'parameters': {'root_seed': 42, 'interval_boundary_policy': 'measurement_support',
                               'arms': list(outcome.DOWNSTREAM_ARM_NAMES),
                               'models': list(outcome.MODEL_NAMES), 'targets': list(outcome.TARGET_COLUMNS)},
                'provenance_note': 'Legacy official/authority field names are original content-seal schema names. '
                                   'Their values bind NEW independently generated inputs, with no historical release approval.',
                'inputs': {'prepare_dir': str(prepare_dir), 'stress_dir': str(stress_dir),
                           'training_labels': str(training_labels), 'scoring_labels': str(scoring_labels)}}
    write_json(output / 'manifest.json', manifest)
    stop = threading.Event()
    current = {'step': 'load_inputs', 'observed_fit_records': 0, 'observed_prediction_rows': 0}

    def monitor() -> None:
        while not stop.is_set():
            try:
                progress = _progress_snapshot(output, started, current['step'])
                current['observed_fit_records'] = max(current['observed_fit_records'], progress['persisted_new_fit_records'])
                current['observed_prediction_rows'] = max(current['observed_prediction_rows'], progress['persisted_prediction_rows'])
                write_json(output / 'progress.json', progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
            except Exception as error:
                # Progress is observational and cannot change the scientific run.
                print(json.dumps({'stage': 'downstream', 'progress_observation_error': str(error)}), flush=True)
            if stop.wait(30):
                break

    watcher = threading.Thread(target=monitor, name='downstream-progress', daemon=True)
    watcher.start()
    try:
        prepare_manifest, prepared = load_stage(prepare_dir, 'prepare', ('primitives',))
        stress_manifest, stress_frames = load_stage(stress_dir, 'stress', TABLE_NAMES)
        prepare_sha = sha256(prepare_dir / 'manifest.json')
        stress_sha = sha256(stress_dir / 'manifest.json')
        stress_inputs = stress_manifest['inputs']
        if stress_inputs['prepare_manifest_sha256'] != prepare_sha:
            raise ValueError('stress and downstream prepare manifests are crosswired')
        g2_dir = Path(stress_inputs['g2_dir']).resolve()
        if output == g2_dir or output in g2_dir.parents or g2_dir in output.parents:
            raise ValueError('output overlaps G2 input')
        g2_manifest, g2_frames = load_stage(g2_dir, 'g2', G2_NAMES)
        g2_sha = sha256(g2_dir / 'manifest.json')
        if stress_inputs['g2_manifest_sha256'] != g2_sha:
            raise ValueError('stress G2 manifest changed')
        input_pins = [(prepare_dir / 'manifest.json', prepare_sha),
                      (stress_dir / 'manifest.json', stress_sha), (g2_dir / 'manifest.json', g2_sha),
                      (training_labels, sha256(training_labels)), (scoring_labels, sha256(scoring_labels))]
        for root, document, names in ((prepare_dir, prepare_manifest, ('primitives',)),
                                     (stress_dir, stress_manifest, TABLE_NAMES),
                                     (g2_dir, g2_manifest, G2_NAMES)):
            for name in names:
                pin = document['outputs'][name]
                input_pins.append(((root / pin['path']).resolve(), pin['sha256']))
        manifest['inputs'].update(prepare_manifest_sha256=prepare_sha, stress_manifest_sha256=stress_sha,
                                  g2_dir=str(g2_dir), g2_manifest_sha256=g2_sha,
                                  pins=[{'path': str(path), 'sha256': digest} for path, digest in input_pins])
        train, score = load_labels(training_labels), load_labels(scoring_labels)
        validate_study_labels(train, score)
        primitives = prepared['primitives']
        validate_full_stress_grid(stress_frames['target_ledger'])
        primitive_digest = outcome._primitive_frame_digest(primitives)
        selected, _, _ = runner._selected_rows_and_ledgers(stress_frames['target_ledger'], primitives)
        expected_keys = tuple(sorted((str(row['subject_id']), int(row['sensor_day_id']),
                                      str(row['primitive']), 'contiguous_20pct', int(row['draw']))
                                     for row in selected))
        if len(set(expected_keys)) != len(expected_keys):
            raise ValueError('selected downstream target key duplicated')
        current['step'] = 'validate_new_g2_evidence'
        pairs = build_validated_pairs(g2_frames['cell_replays'], g2_frames['deletion_audit'],
                                      g2_frames['mask_intervals'], expected_keys=expected_keys,
                                      artifact_hashes={name: g2_manifest['outputs'][name]['sha256'] for name in G2_NAMES})
        evidence = tuple(pairs[key][0] for key in expected_keys)
        del g2_frames, pairs
        current['step'] = 'build_adapter'
        snapshot = build_independent_adapter_snapshot(
            stress_frames, evidence, primitives, expected_task12_authority_digest=stress_sha,
            expected_official_g2_authority_digest=g2_sha,
            expected_interval_boundary_policy='measurement_support', expected_primitive_digest=primitive_digest)
        del evidence, stress_frames
        current['step'] = 'build_training_arms'
        training_arms = outcome.build_outcome_arms(primitives, train, interval_boundary_policy='measurement_support')
        outcome_digest = str(training_arms.manifest.iloc[0]['outcome_build_digest'])
        outcome.validate_outcome_arms(training_arms, expected_content_digest=outcome_digest)
        del training_arms
        manifest['adapter'] = {'content_digest': snapshot.content_digest,
                               'source_patch_projection_digest': snapshot.source_patch_projection.logical_digest,
                               'primitive_digest': primitive_digest, 'outcome_build_digest': outcome_digest,
                               'selected_targets': snapshot.eligible_target_count,
                               'reference_keys': snapshot.reference_key_count,
                               'expected_predictions': snapshot.source_patch_projection.row_count,
                               'expected_primary_keys': snapshot.expected_primary_key_count}
        write_json(output / 'manifest.json', manifest)
        current['step'] = 'new_160_fits_predictions_and_statistics'
        artifacts = snapshot_runner(snapshot, primitives, train, score, output / 'run',
            expected_snapshot_digest=snapshot.content_digest, expected_primitive_digest=primitive_digest,
            expected_outcome_build_digest=outcome_digest, root_seed=42, **scheduling_arguments)
        current['step'] = 'verify_fresh_outputs'
        with (output / 'run/fit/fit_ledger.arrow').open('rb') as stream:
            ledger = ipc.open_stream(stream).read_all().to_pandas()
        validate_fit_topology(ledger)
        if count_complete_fit_records(output / 'run/fit/fit_states.pack') != 160:
            raise ValueError('new fit pack does not contain 160 complete records')
        validate_prediction_count(artifacts.observed_prediction_pin.row_count, snapshot.eligible_target_count)
        manifest['new_model_fits'] = 160
        for path, digest in input_pins:
            if sha256(path) != digest:
                raise ValueError(f'input changed during downstream execution: {path}')
        verify_vendor()
        if any(sha256(PAPER / name) != digest for name, digest in execution_sources.items()):
            raise ValueError('downstream execution source changed during this run')
        if workers > 1:
            schedule_path = output / 'participant_prediction_work/scheduling.json'
            schedule = json.loads(schedule_path.read_text(encoding='utf-8'))
            if (schedule.get('status') != 'complete' or schedule.get('rows') != 943392
                    or schedule.get('participant_count') != 10 or schedule.get('new_model_fits_in_workers') != 0):
                raise ValueError('parallel prediction schedule is incomplete')
            manifest['execution']['scheduling_report'] = {
                'path': schedule_path.relative_to(output).as_posix(), 'sha256': sha256(schedule_path)}
        for path in sorted((output / 'run').rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(output).as_posix()
            pin = {'path': relative, 'sha256': sha256(path), 'bytes': path.stat().st_size}
            if path.suffix == '.arrow':
                with path.open('rb') as stream:
                    reader = ipc.open_stream(stream)
                    pin['columns'] = reader.schema.names
                    pin['rows'] = sum(batch.num_rows for batch in reader)
            elif path.suffix == '.parquet':
                table = pq.ParquetFile(path)
                pin['columns'] = table.schema_arrow.names
                pin['rows'] = table.metadata.num_rows
            manifest['outputs'][relative] = pin
        manifest.update(status='complete', new_model_fits=160,
                        prediction_rows=artifacts.observed_prediction_pin.row_count,
                        observed_row_count=artifacts.observed_prediction_pin.row_count,
                        expected_primary_key_count=snapshot.expected_primary_key_count,
                        run_content_digest=artifacts.content_digest,
                        statistics_content_digest=artifacts.statistics_manifest.content_digest,
                        seconds=time.perf_counter() - started)
        write_json(output / 'manifest.json', manifest)
        return manifest
    except BaseException as error:
        manifest.update(status='failed', seconds=time.perf_counter() - started,
                        error_type=type(error).__name__, error=str(error),
                        observed_new_fit_records_lower_bound=current['observed_fit_records'],
                        observed_prediction_rows_lower_bound=current['observed_prediction_rows'])
        write_json(output / 'manifest.json', manifest)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        stop.set()
        watcher.join(timeout=5)
        progress = _progress_snapshot(output, started, current['step'])
        progress['status'] = manifest['status']
        progress['new_model_fits_verified'] = manifest['new_model_fits']
        progress['observed_new_fit_records_lower_bound'] = current['observed_fit_records']
        progress['observed_prediction_rows_lower_bound'] = current['observed_prediction_rows']
        write_json(output / 'progress.json', progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-dir', type=Path, required=True)
    parser.add_argument('--stress-dir', type=Path, required=True)
    parser.add_argument('--training-labels', type=Path, required=True)
    parser.add_argument('--scoring-labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4,
                        help='Participant prediction processes; 1 uses the original sequential runner')
    args = parser.parse_args()
    result = run_downstream(args.prepare_dir, args.stress_dir, args.training_labels,
                            args.scoring_labels, args.output_dir, workers=args.workers)
    print(json.dumps({'stage': 'downstream', 'status': result['status'],
                      'new_model_fits': result['new_model_fits'],
                      'prediction_rows': result['prediction_rows']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
