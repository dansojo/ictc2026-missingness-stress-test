"""Bounded participant-process scheduling around immutable downstream calculations.

Only locally created frozen value objects cross the multiprocessing IPC channel.
No pickle file is read or accepted. Workers write one sealed participant IPC batch;
the parent verifies and joins the ten batches in the original canonical order.
"""
from __future__ import annotations

import ast
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from full_prepare import PAPER, sha256, verify_vendor
sys.path.insert(0, str(PAPER / 'full_vendor'))
from semantic_indicators import outcome, stress_downstream, stress_downstream_statistics
from semantic_indicators import stress_downstream_runner as original
from semantic_indicators.stress_downstream_runner import (
    DownstreamAdapterSnapshot, DownstreamRunArtifacts, _require_digest,
    validate_downstream_adapter_snapshot, _snapshot_arrow_tables, _write_ipc_table,
    write_downstream_fit_state_pack, load_downstream_fit_state_pack, _scoring_lookup,
    _LogicalHasher, _file_pin, _sha, _statistics_authority_digest, _canonical,
    _FAMILIES, _CONDITIONS, _observed_prediction_row,
)

sys.dont_write_bytecode = True


def validate_worker_count(workers: int) -> None:
    if type(workers) is not int or workers not in (1, 2, 3, 4):
        raise ValueError('Prediction workers must be an exact integer from 1 through 4')


def validate_submission_order(expected: tuple[int, ...], requested: tuple[int, ...] | None) -> tuple[int, ...]:
    order = expected if requested is None else requested
    if (type(order) is not tuple or any(type(value) is not int for value in order)
            or len(order) != len(expected) or set(order) != set(expected)):
        raise ValueError('Participant dispatch must be an exact permutation')
    return order


def create_worker_workspace(output: Path, requested: Path | None) -> Path:
    output = output.resolve()
    work = (output.parent / 'participant_prediction_work' if requested is None else requested).resolve()
    if work == output or work.is_relative_to(output) or work.parent != output.parent:
        raise ValueError('Worker workspace must be a dedicated sibling of the final run')
    if work.exists():
        raise FileExistsError('Worker workspace must be fresh')
    if work.parent.is_symlink() or (hasattr(work.parent, 'is_junction') and work.parent.is_junction()):
        raise ValueError('Worker workspace parent cannot be a link')
    work.mkdir()
    return work


def _work_file(root: Path, name: str) -> Path:
    if Path(name).name != name or ':' in name or '/' in name or '\\' in name:
        raise ValueError('Worker filename escapes its workspace')
    if root.is_symlink() or (hasattr(root, 'is_junction') and root.is_junction()):
        raise ValueError('Worker workspace cannot be replaced by a link')
    root = root.resolve(strict=True)
    path = root / name
    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        raise ValueError('Worker output cannot be a link')
    return path


def _token(value):
    """Type- and float-bit-preserving seal for the trusted IPC value envelope."""
    if value is None:
        return ['none']
    if type(value) is bool:
        return ['bool', value]
    if type(value) is int:
        return ['int', value]
    if type(value) is float:
        return ['float64', struct.pack('>d', value).hex()]
    if type(value) is str:
        return ['str', value]
    if type(value) is bytes:
        return ['bytes', len(value), hashlib.sha256(value).hexdigest()]
    if type(value) is pd.Timestamp:
        return ['Timestamp', str(value.asm8.dtype), str(value.asm8.view('i8')), str(value.tz)]
    if isinstance(value, np.generic):
        return ['numpy', value.dtype.str, value.tobytes().hex()]
    if type(value) is pd.DataFrame:
        return ['DataFrame', list(value.columns), list(map(str, value.dtypes)),
                _token(tuple(value.index)), _token(tuple(value.itertuples(index=False, name=None)))]
    if type(value) in (tuple, list):
        return [type(value).__name__, [_token(child) for child in value]]
    if type(value) is dict:
        entries = [(_token(key), _token(child)) for key, child in value.items()]
        entries.sort(key=lambda item: json.dumps(item[0], separators=(',', ':'), ensure_ascii=False))
        return ['dict', entries]
    if is_dataclass(value):
        return [type(value).__module__ + '.' + type(value).__qualname__,
                [[field.name, _token(getattr(value, field.name))] for field in fields(value)]]
    raise TypeError('Unsupported IPC envelope value type: ' + type(value).__name__)


@dataclass(frozen=True)
class ParticipantJob:
    participant_position: int
    fold: int
    participant: str
    first_row: int
    universe: pd.DataFrame
    models: dict
    ledgers: dict
    contexts: dict
    patches: dict
    labels: dict
    adapter_digest: str
    fit_authority_digest: str
    source_pins: tuple
    driver_sha256: str
    content_digest: str = ''


def job_digest(job: ParticipantJob) -> str:
    return _sha(['independent-downstream-participant-ipc-v1',
                 [[field.name, _token(getattr(job, field.name))] for field in fields(job)
                  if field.name != 'content_digest']])


def _check_sources(job: ParticipantJob) -> None:
    actual = tuple((pin['path'], pin['sha256']) for pin in verify_vendor())
    if actual != job.source_pins or sha256(Path(__file__)) != job.driver_sha256:
        raise ValueError('Participant worker source pins changed')


def validate_job(job: ParticipantJob) -> None:
    if type(job) is not ParticipantJob:
        raise TypeError('Worker requires an exact locally created ParticipantJob')
    if (type(job.participant_position) is not int or not 0 <= job.participant_position < 10
            or type(job.fold) is not int or job.fold != job.participant_position
            or type(job.participant) is not str or not job.participant
            or type(job.first_row) is not int or job.first_row < 0):
        raise ValueError('Worker participant identity is invalid')
    _require_digest(job.adapter_digest, 'worker adapter digest')
    _require_digest(job.fit_authority_digest, 'worker fit authority digest')
    if (type(job.universe) is not pd.DataFrame or not len(job.universe)
            or tuple(job.universe.columns) != original._PREDICTION_UNIVERSE_COLUMNS
            or not job.universe['participant'].eq(job.participant).all()
            or job.universe.duplicated(['primitive', 'sensor_day_id', 'draw']).any()):
        raise ValueError('Worker selected universe is malformed or crosswired')
    if job.content_digest != job_digest(job):
        raise ValueError('Worker IPC envelope digest changed')
    _check_sources(job)
    expected_fits = set(range(job.participant_position * 16, (job.participant_position + 1) * 16))
    if set(job.models) != expected_fits or set(job.ledgers) != expected_fits:
        raise ValueError('Worker requires exactly the participant sixteen sealed fits')
    for fit_seq in sorted(expected_fits):
        model, ledger = job.models[fit_seq], job.ledgers[fit_seq]
        outcome.validate_frozen_fold_model(model, expected_content_digest=model.content_digest)
        local = fit_seq % 16
        arm, model_name, target = (outcome.DOWNSTREAM_ARM_NAMES[local // 8],
                                  outcome.MODEL_NAMES[(local % 8) // 4], outcome.TARGET_COLUMNS[local % 4])
        if (model.fold, model.held_out_subject, model.arm, model.model_name, model.target) != (
                job.fold, job.participant, arm, model_name, target):
            raise ValueError('Worker frozen model belongs to another fit identity')
        if (ledger['fit_seq'] != fit_seq or ledger['held_out_subject'] != job.participant
                or ledger['fit_content_digest'] != model.content_digest
                or ledger['preprocessor_digest'] != model.preprocessor_digest
                or ledger['model_digest'] != model.model_digest
                or ledger['fit_ledger_record_digest'] != original._row_digest(
                    'task-12-downstream-fit-ledger-row-v1', ledger, original._FIT_LEDGER_DIGEST_NAMES)):
            raise ValueError('Worker model and fit ledger are crosswired')
    expected_contexts = {(job.participant, int(day)) for day in job.universe['sensor_day_id'].unique()}
    if set(job.contexts) != expected_contexts:
        raise ValueError('Worker reference context topology differs')
    for key, context in job.contexts.items():
        stress_downstream._validate_downstream_context(context)
        if (context.held_out_subject, context.sensor_day_id) != key or context.fold != job.fold:
            raise ValueError('Worker reference context identity differs')
    expected_patches = {(job.participant, int(row.sensor_day_id), str(row.primitive), int(row.draw), condition, arm)
        for row in job.universe.itertuples(index=False) for condition in _CONDITIONS for arm in outcome.DOWNSTREAM_ARM_NAMES}
    if set(job.patches) != expected_patches:
        raise ValueError('Worker feature patch topology differs')
    for key, patch in job.patches.items():
        stress_downstream._validate_downstream_patch(patch)
        if ((patch.held_out_subject, patch.sensor_day_id, patch.primitive, patch.draw, patch.condition, patch.arm) != key
                or patch.fold != job.fold or patch.reference_context_digest != job.contexts[key[:2]].content_digest):
            raise ValueError('Worker patch is crosswired to another reference')
    if any(key[0] != job.participant for key in job.labels):
        raise ValueError('Worker scoring labels include another participant')


def build_jobs(snapshot, model_by_seq, ledger_by_seq, contexts, patches, labels, fit_digest: str) -> tuple[ParticipantJob, ...]:
    source_pins = tuple((pin['path'], pin['sha256']) for pin in verify_vendor())
    driver = sha256(Path(__file__))
    jobs, offset = [], 0
    for position, (fold, participant) in enumerate(snapshot.fold_roster):
        universe = snapshot.prediction_universe.loc[snapshot.prediction_universe.participant.eq(participant)].copy(deep=True)
        fit_keys = range(position * 16, (position + 1) * 16)
        job = ParticipantJob(position, fold, participant, offset, universe,
            {key: model_by_seq[key] for key in fit_keys}, {key: ledger_by_seq[key] for key in fit_keys},
            {key: value for key, value in contexts.items() if key[0] == participant},
            {key: value for key, value in patches.items() if key[0] == participant},
            {key: value for key, value in labels.items() if key[0] == participant},
            snapshot.content_digest, fit_digest, source_pins, driver)
        job = replace(job, content_digest=job_digest(job))
        validate_job(job)
        jobs.append(job)
        offset += len(universe) * 48
    if offset != len(snapshot.prediction_universe) * 48:
        raise ValueError('Participant job partition changed the prediction topology')
    return tuple(jobs)


def _peak_rss_bytes() -> int:
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
                ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = wintypes.HANDLE
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        if not query(process(), ctypes.byref(counters), counters.cb):
            raise OSError('Cannot measure worker peak resident memory')
        return int(counters.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)


def _run_worker(job: ParticipantJob, work_dir: str) -> dict:
    started = time.perf_counter()
    validate_job(job)
    root = Path(work_dir)
    name = f'participant_{job.participant_position:02d}.arrow'
    path = _work_file(root, name)
    try:
        with path.open('xb') as stream:
            with ipc.new_stream(stream, stress_downstream_statistics.OBSERVED_PREDICTION_SCHEMA) as writer:
                row_seq, logical_digest = _calculate_participant(job, writer)
        _check_sources(job)
        result = {'status': 'complete', 'participant_position': job.participant_position,
                  'participant': job.participant, 'first_row': job.first_row, 'rows': row_seq - job.first_row,
                  'job_digest': job.content_digest, 'logical_digest': logical_digest,
                  'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path),
                  'seconds': time.perf_counter() - started, 'peak_rss_bytes': _peak_rss_bytes(),
                  'process_id': os.getpid()}
        _work_file(root, name + '.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        return result
    except BaseException:
        _work_file(root, name + '.failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def _binding(job: ParticipantJob) -> tuple:
    return job.participant, job.first_row, len(job.universe) * 48, job.content_digest


def validate_result_inventory(expected: dict, results: list[dict]) -> list[dict]:
    records = {}
    for result in results:
        if type(result) is not dict or any(type(result.get(name)) is not int for name in
            ('participant_position', 'first_row', 'rows')):
            raise ValueError('Worker result counts must be exact integers')
        position = result.get('participant_position')
        if type(position) is not int or position not in expected or position in records:
            raise ValueError('Worker result has a duplicate or unknown participant')
        identity = (result.get('participant'), result.get('first_row'), result.get('rows'), result.get('job_digest'))
        if result.get('status') != 'complete' or identity != expected[position]:
            raise ValueError('Worker result is failed, malformed or crosswired')
        records[position] = result
    if set(records) != set(expected):
        raise ValueError('Worker result inventory is incomplete')
    return [records[position] for position in sorted(records)]


def _expected_rows(job: ParticipantJob):
    row_seq = job.first_row
    for family_position, family in enumerate(_FAMILIES):
        selected = job.universe.loc[job.universe.family.eq(family)].sort_values(
            ['primitive_position', 'sensor_day_id', 'draw'], kind='stable').to_dict('records')
        for arm_position, arm in enumerate(outcome.DOWNSTREAM_ARM_NAMES):
            for model_position, model_name in enumerate(outcome.MODEL_NAMES):
                for target_position, target in enumerate(outcome.TARGET_COLUMNS):
                    fit_seq = (((job.participant_position * 2 + arm_position) * 2 + model_position) * 4 + target_position)
                    model, ledger = job.models[fit_seq], job.ledgers[fit_seq]
                    for row in selected:
                        day, primitive, draw = int(row['sensor_day_id']), str(row['primitive']), int(row['draw'])
                        labels = job.labels.get((job.participant, pd.Timestamp(row['lifelog_date']), pd.Timestamp(row['sleep_date'])))
                        label = None if labels is None else int(labels[target_position])
                        for condition_position, condition in enumerate(_CONDITIONS):
                            patch = job.patches[(job.participant, day, primitive, draw, condition, arm)]
                            yield {'row_seq': row_seq, 'participant_position': job.participant_position,
                                'participant': job.participant, 'held_out_subject': job.participant, 'fold': job.fold,
                                'family_position': family_position, 'family': family,
                                'primitive_position': int(row['primitive_position']), 'primitive': primitive,
                                'sensor_day_id': day, 'draw': draw, 'arm_position': arm_position, 'arm': arm,
                                'model_position': model_position, 'model_name': model_name, 'target_position': target_position,
                                'target': target, 'fit_seq': fit_seq, 'condition_position': condition_position,
                                'condition': condition, 'fit_ledger_record_digest': ledger['fit_ledger_record_digest'],
                                'preprocessor_digest': model.preprocessor_digest, 'model_digest': model.model_digest,
                                'fit_content_digest': model.content_digest, 'source_row_digest': patch.content_digest,
                                'reference_context_digest': patch.reference_context_digest, 'label': label,
                                'replay_status': patch.replay_status, 'replay_reason': patch.reason,
                                'label_available': label is not None}
                            row_seq += 1


def read_worker_batch(root: Path, result: dict, job: ParticipantJob, global_hasher) -> pa.RecordBatch:
    if global_hasher.row_count != job.first_row:
        raise ValueError('Worker batch does not start at the canonical global row offset')
    expected_name = f'participant_{job.participant_position:02d}.arrow'
    if result.get('path') != expected_name:
        raise ValueError('Worker result path is not its assigned participant batch')
    path = _work_file(root, expected_name)
    if path.stat().st_size != result.get('bytes') or sha256(path) != result.get('sha256'):
        raise ValueError('Worker batch physical digest changed')
    with path.open('rb') as stream:
        reader = ipc.open_stream(stream)
        if not reader.schema.equals(stress_downstream_statistics.OBSERVED_PREDICTION_SCHEMA, check_metadata=True):
            raise ValueError('Worker batch schema changed')
        batch = next(reader, None)
        if batch is None or next(reader, None) is not None or batch.num_rows != len(job.universe) * 48:
            raise ValueError('Worker must produce exactly one complete participant batch')
    local_hasher = _LogicalHasher('task-12-downstream-observed-prediction-logical-v1',
                                 stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS)
    for row, expected in zip(batch.to_pylist(), _expected_rows(job), strict=True):
        if any(type(row[name]) is not type(value) or row[name] != value for name, value in expected.items()):
            raise ValueError('Worker row order, identity or provenance changed')
        issues = set()
        stress_downstream_statistics._validate_row_contract(row, issues)
        if issues:
            raise ValueError('Worker scientific row contract failed: ' + ','.join(sorted(issues)))
        names = tuple(name for name, _kind, _nullable in
                      stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS if name != 'row_content_digest')
        if row['row_content_digest'] != original._row_digest(
                'task-12-downstream-observed-prediction-row-v1', row, names):
            raise ValueError('Worker individual observed row digest changed')
        local_hasher.update(row)
        global_hasher.update(row)
    if local_hasher.hexdigest() != result['logical_digest'] or sha256(path) != result['sha256']:
        raise ValueError('Worker row digest or file changed during validation')
    return batch


def execute_jobs(jobs: tuple[ParticipantJob, ...], work: Path, *, workers: int,
                 submission_order: tuple[int, ...] | None = None) -> list[dict]:
    validate_worker_count(workers)
    by_position = {job.participant_position: job for job in jobs}
    if len(by_position) != len(jobs) or not jobs:
        raise ValueError('Participant jobs must be nonempty and unique')
    order = validate_submission_order(tuple(by_position), submission_order)
    results = []
    if workers == 1:
        for position in order:
            results.append(_run_worker(by_position[position], str(work)))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {}
            positions = iter(order)
            for position in list(order)[:workers]:
                next(positions)
                pending[pool.submit(_run_worker, by_position[position], str(work))] = position
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    results.append(future.result())
                    position = next(positions, None)
                    if position is not None:
                        pending[pool.submit(_run_worker, by_position[position], str(work))] = position
    return validate_result_inventory({job.participant_position: _binding(job) for job in jobs}, results)


def write_predictions(snapshot, model_by_seq, ledger_by_seq, contexts, patches, labels,
                      fit_digest, observed_path, observed_hasher, *, workers, work_dir=None,
                      submission_order=None, final_output_dir=None) -> int:
    started = time.perf_counter()
    validate_worker_count(workers)
    final_run = observed_path.parent if final_output_dir is None else final_output_dir
    # The original outer function writes inside run.<uuid>.partial. The dedicated
    # worker directory is a sibling and never enters the final run's artifacts.
    work = create_worker_workspace(final_run, work_dir)
    jobs = build_jobs(snapshot, model_by_seq, ledger_by_seq, contexts, patches, labels, fit_digest)
    report = {'schema_name': 'independent_downstream_prediction_schedule_v1', 'status': 'running',
              'workers': workers, 'participant_count': len(jobs), 'new_model_fits_in_workers': 0,
              'worker_ipc': 'locally created frozen objects through multiprocessing; no pickle files',
              'adapter_digest': snapshot.content_digest, 'fit_authority_digest': fit_digest}
    report_path = work / 'scheduling.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    try:
        results = execute_jobs(jobs, work, workers=workers, submission_order=submission_order)
        by_position = {job.participant_position: job for job in jobs}
        with observed_path.open('xb') as stream:
            with ipc.new_stream(stream, stress_downstream_statistics.OBSERVED_PREDICTION_SCHEMA) as writer:
                for result in results:
                    writer.write_batch(read_worker_batch(work, result, by_position[result['participant_position']], observed_hasher))
        report.update(status='complete', seconds=time.perf_counter() - started,
                      parent_peak_rss_bytes=_peak_rss_bytes(), worker_results=results,
                      rows=sum(result['rows'] for result in results), global_logical_digest=observed_hasher.hexdigest())
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return report['rows']
    except BaseException:
        report.update(status='failed', seconds=time.perf_counter() - started, error=traceback.format_exc())
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        raise


def source_equivalence_report() -> dict:
    """Audit static copied calculations against the immutable original AST."""
    vendor_path = Path(original.__file__)
    own_path = Path(__file__)
    source = ast.parse(vendor_path.read_text(encoding='utf-8'))
    own = ast.parse(own_path.read_text(encoding='utf-8'))
    functions = {node.name: node for node in own.body if isinstance(node, ast.FunctionDef)}
    old = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'run_downstream_snapshot')
    new = functions['run_downstream_snapshot_parallel']
    old_try = next(node for node in old.body if isinstance(node, ast.Try))
    new_try = next(node for node in new.body if isinstance(node, ast.Try))
    def row_start(body):
        return next(index for index, node in enumerate(body) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'row_seq' for target in node.targets))
    def suffix_start(body):
        return next(index for index, node in enumerate(body) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Tuple) for target in node.targets))
    def dump(nodes):
        return ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
    participant = next(node for node in ast.walk(old) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple) and isinstance(node.target.elts[0], ast.Name)
        and node.target.elts[0].id == 'participant_position')
    worker = functions['_calculate_participant']
    calculation_start = next(index for index, node in enumerate(worker.body) if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name) and node.target.id == 'participant_rows')
    prefix_equal = dump(old_try.body[:row_start(old_try.body)]) == dump(new_try.body[:row_start(new_try.body)])
    outside_equal = dump(old.body[:old.body.index(old_try)]) == dump(new.body[:new.body.index(new_try)])
    suffix_equal = dump(old_try.body[suffix_start(old_try.body):]) == dump(new_try.body[suffix_start(new_try.body):])
    cleanup_equal = ast.dump(old_try.handlers[0], include_attributes=False) == ast.dump(new_try.handlers[0], include_attributes=False)
    return {'schema_name': 'downstream_parallel_source_equivalence_v1',
            'original_sha256': sha256(vendor_path), 'adapter_sha256': sha256(own_path),
            'original_function_lines': [old.lineno, old.end_lineno],
            'original_participant_body_lines': [participant.body[0].lineno, participant.body[-1].end_lineno],
            'outer_prefix_ast_equal': prefix_equal and outside_equal,
            'outer_suffix_ast_equal': suffix_equal and cleanup_equal,
            'worker_calculation_ast_equal': dump(participant.body) == dump(worker.body[calculation_start:-1])}


# The source-derived calculation and original outer function are appended by the
# checked generator in validation/downstream_parallel_scheduling_01. They are
# static source, not runtime exec or a modification of the scientific vendor.
# BEGIN SOURCE-DERIVED SECTIONS


def _calculate_participant(job: ParticipantJob, writer):
    participant_position, fold, participant = job.participant_position, job.fold, job.participant
    row_seq = job.first_row
    snapshot = SimpleNamespace(prediction_universe=job.universe)
    model_by_seq, ledger_by_seq = job.models, job.ledgers
    contexts, patches, labels = job.contexts, job.patches, job.labels
    observed_hasher = _LogicalHasher(
        'task-12-downstream-observed-prediction-logical-v1',
        stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS)
    probability_cache: dict[tuple[int, int], float] = {}
    participant_rows: list[dict[str, object]] = []
    for family_position, family in enumerate(_FAMILIES):
        selected_rows = snapshot.prediction_universe.loc[
            snapshot.prediction_universe["participant"].eq(participant)
            & snapshot.prediction_universe["family"].eq(family)
        ].sort_values(
            ["primitive_position", "sensor_day_id", "draw"], kind="stable"
        )
        for arm_position, arm in enumerate(outcome.DOWNSTREAM_ARM_NAMES):
            for model_position, model_name in enumerate(outcome.MODEL_NAMES):
                for target_position, target in enumerate(outcome.TARGET_COLUMNS):
                    fit_seq = (((participant_position * 2 + arm_position) * 2
                                + model_position) * 4 + target_position)
                    model = model_by_seq[fit_seq]
                    ledger = ledger_by_seq[fit_seq]
                    for selected in selected_rows.to_dict("records"):
                        sensor_day_id = int(selected["sensor_day_id"])
                        primitive = str(selected["primitive"])
                        draw = int(selected["draw"])
                        context = contexts[(participant, sensor_day_id)]
                        reference = stress_downstream.reference_feature_frame(context, arm)
                        cache_key = (fit_seq, sensor_day_id)
                        if cache_key not in probability_cache:
                            probability_cache[cache_key] = float(
                                outcome.predict_frozen_fold_model(
                                    model, reference,
                                    expected_content_digest=model.content_digest,
                                )[0]
                            )
                        reference_probability = probability_cache[cache_key]
                        score_key = (
                            participant, pd.Timestamp(selected["lifelog_date"]),
                            pd.Timestamp(selected["sleep_date"]),
                        )
                        score_values = labels.get(score_key)
                        label = None if score_values is None else int(score_values[target_position])
                        base = {
                            "participant_position": participant_position,
                            "participant": participant, "fold": fold,
                            "held_out_subject": participant,
                            "family_position": family_position, "family": family,
                            "primitive_position": int(selected["primitive_position"]),
                            "primitive": primitive, "sensor_day_id": sensor_day_id,
                            "draw": draw, "arm_position": arm_position, "arm": arm,
                            "model_position": model_position, "model_name": model_name,
                            "target_position": target_position, "target": target,
                            "fit_seq": fit_seq,
                            "fit_ledger_record_digest": str(ledger["fit_ledger_record_digest"]),
                            "preprocessor_digest": str(ledger["preprocessor_digest"]),
                            "model_digest": str(ledger["model_digest"]),
                            "fit_content_digest": str(ledger["fit_content_digest"]),
                        }
                        for condition_position, condition in enumerate(_CONDITIONS):
                            patch = patches[(participant, sensor_day_id, primitive, draw,
                                             condition, arm)]
                            patched = stress_downstream.apply_downstream_feature_patch(
                                reference, patch,
                                expected_context_digest=context.content_digest,
                                expected_perturbation_digest=patch.perturbation_digest,
                            )
                            perturbed = None if patched is None else float(
                                outcome.predict_frozen_fold_model(
                                    model, patched,
                                    expected_content_digest=model.content_digest,
                                )[0]
                            )
                            observed = _observed_prediction_row(
                                base, patch, row_seq=row_seq,
                                condition_position=condition_position,
                                reference_probability=reference_probability,
                                perturbed_probability=perturbed, label=label,
                            )
                            participant_rows.append(observed)
                            observed_hasher.update(observed)
                            row_seq += 1
    writer.write_batch(pa.RecordBatch.from_pylist(
        participant_rows,
        schema=stress_downstream_statistics.OBSERVED_PREDICTION_SCHEMA,
    ))
    return row_seq, observed_hasher.hexdigest()


def run_downstream_snapshot_parallel(
    snapshot: DownstreamAdapterSnapshot,
    primitives: pd.DataFrame,
    training_labels: pd.DataFrame,
    scoring_labels: pd.DataFrame | None,
    output_dir: Path,
    *,
    expected_snapshot_digest: str,
    expected_primitive_digest: str,
    expected_outcome_build_digest: str,
    root_seed: int,
    workers: int = 4,
    work_dir: Path | None = None,
    submission_order: tuple[int, ...] | None = None,
) -> DownstreamRunArtifacts:
    """Run one bounded synthetic snapshot and atomically hand off its artifacts."""
    expected_primitive = _require_digest(expected_primitive_digest, "expected primitive digest")
    expected_outcome = _require_digest(expected_outcome_build_digest, "expected outcome digest")
    validate_downstream_adapter_snapshot(
        snapshot, expected_content_digest=expected_snapshot_digest,
        expected_source_patch_projection_digest=snapshot.source_patch_projection.logical_digest,
    )
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise TypeError("output_dir must be an absolute pathlib Path")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    partial = output_dir.with_name(f"{output_dir.name}.{uuid.uuid4().hex}.partial")
    partial.mkdir()
    try:
        for name, table in _snapshot_arrow_tables(snapshot):
            _write_ipc_table(partial / name, table)
        training_arms = outcome.build_outcome_arms(
            primitives, training_labels,
            interval_boundary_policy="measurement_support",
        )
        outcome.validate_outcome_arms(
            training_arms, expected_content_digest=expected_outcome
        )
        manifest = training_arms.manifest
        if (
            type(manifest) is not pd.DataFrame
            or len(manifest) != 1
            or tuple(manifest.columns).count("primitive_digest") != 1
        ):
            raise ValueError("training arm primitive digest manifest is invalid")
        arm_primitive_digest = manifest.iloc[0]["primitive_digest"]
        _require_digest(arm_primitive_digest, "training arm primitive digest")
        if arm_primitive_digest != expected_primitive:
            raise ValueError("training arm primitive digest mismatch")
        for seal in snapshot.reference_tables.frame_manifest:
            reference_primitive_digest = seal.source_primitive_digest
            _require_digest(
                reference_primitive_digest, "reference frame primitive digest"
            )
            if reference_primitive_digest != expected_primitive:
                raise ValueError("reference frame primitive digest mismatch")
        fit_seal = write_downstream_fit_state_pack(
            training_arms, training_labels, snapshot.reference_tables, partial / "fit",
            expected_outcome_build_digest=expected_outcome,
            expected_reference_content_digest=snapshot.reference_content_digest,
            root_seed=root_seed,
        )
        loaded = load_downstream_fit_state_pack(
            fit_seal.root, expected_content_digest=fit_seal.content_digest
        )
        model_by_seq = {index: model for index, model in enumerate(loaded.models)}
        ledger_by_seq = {
            int(row["fit_seq"]): row for row in loaded.ledger.to_dict("records")
        }
        contexts = {(value.held_out_subject, value.sensor_day_id): value for value in snapshot.contexts}
        patches = {
            (value.held_out_subject, value.sensor_day_id, value.primitive, value.draw,
             value.condition, value.arm): value for value in snapshot.patches
        }
        labels = _scoring_lookup(scoring_labels)
        observed_path = partial / "observed_predictions.arrow"
        observed_hasher = _LogicalHasher(
            "task-12-downstream-observed-prediction-logical-v1",
            stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS,
        )
        row_seq = write_predictions(
            snapshot, model_by_seq, ledger_by_seq, contexts, patches, labels,
            fit_seal.content_digest, observed_path, observed_hasher,
            workers=workers, work_dir=work_dir, submission_order=submission_order,
            final_output_dir=output_dir)
        observed_bytes, observed_sha = _file_pin(observed_path)
        pin_values = {
            "schema_name": "task-12-downstream-observed-prediction-input-v1",
            "row_count": observed_hasher.row_count,
            "columns": stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS,
            "byte_count": observed_bytes, "physical_sha256": observed_sha,
            "logical_digest": observed_hasher.hexdigest(),
        }
        unsealed_pin = stress_downstream_statistics.ObservedPredictionInputPin(
            **pin_values, content_digest="0" * 64
        )
        pin = stress_downstream_statistics.ObservedPredictionInputPin(
            **pin_values, content_digest=_sha([
                "task-12-downstream-observed-prediction-input-pin-v1",
                unsealed_pin.schema_name, unsealed_pin.row_count,
                [list(value) for value in unsealed_pin.columns], unsealed_pin.byte_count,
                unsealed_pin.physical_sha256, unsealed_pin.logical_digest,
            ])
        )
        authority_values = {
            "fold_roster": snapshot.fold_roster, "families": _FAMILIES,
            "primitive_family_pairs": snapshot.primitive_family_pairs,
            "arms": outcome.DOWNSTREAM_ARM_NAMES, "models": outcome.MODEL_NAMES,
            "targets": outcome.TARGET_COLUMNS, "conditions": _CONDITIONS,
            "expected_primary_key_count": snapshot.expected_primary_key_count,
            "expected_primary_key_digest": snapshot.expected_primary_key_digest,
            "max_primary_keys_per_group": snapshot.max_primary_keys_per_group,
            "observed_prediction_pin": pin,
            "source_patch_projection_row_count": snapshot.source_patch_projection.row_count,
            "source_patch_projection_digest": snapshot.source_patch_projection.logical_digest,
            "fit_ledger_row_count": fit_seal.fit_ledger_row_count,
            "fit_state_index_row_count": fit_seal.fit_state_index_row_count,
            "fit_state_record_count": fit_seal.fit_state_record_count,
            "fit_ledger_projection_digest": fit_seal.fit_ledger_projection_digest,
            "fit_state_index_logical_digest": fit_seal.fit_state_index_logical_digest,
            "fit_state_pack_sha256": fit_seal.fit_state_pack_sha256,
            "fit_authority_content_digest": fit_seal.content_digest,
        }
        unsealed_authority = stress_downstream_statistics.DownstreamStatisticsInputAuthority(
            **authority_values, content_digest="0" * 64
        )
        authority = stress_downstream_statistics.DownstreamStatisticsInputAuthority(
            **authority_values, content_digest=_statistics_authority_digest(unsealed_authority)
        )
        def batches() -> Iterable[pa.RecordBatch]:
            with observed_path.open("rb") as input_stream:
                yield from ipc.open_stream(input_stream)
        handoff_dir = partial / "statistics-handoff"
        handoff = stress_downstream_statistics.analyze_downstream_statistics(
            batches(), authority, handoff_dir,
            expected_authority_digest=authority.content_digest,
        )
        statistics_manifest = stress_downstream_statistics.persist_downstream_statistics(
            handoff, partial / "statistics",
            expected_handoff_digest=handoff.content_digest,
        )
        stress_downstream_statistics.verify_persisted_downstream_statistics(
            partial / "statistics", statistics_manifest,
            expected_manifest_digest=statistics_manifest.content_digest,
        )
        shutil.rmtree(handoff_dir)
        artifacts = []
        for path in sorted(
            (value for value in partial.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(partial).as_posix(),
        ):
            byte_count, sha256 = _file_pin(path)
            relative = path.relative_to(partial).as_posix()
            artifacts.append({
                "relative_path": relative, "byte_count": byte_count,
                "physical_sha256": sha256,
                "content_digest": _sha([
                    "task-12-downstream-run-artifact-seal-v1", relative,
                    byte_count, sha256,
                ]),
            })
        manifest_core = {
            "schema_name": "task-12-downstream-run-manifest-v1",
            "adapter_digest": snapshot.content_digest,
            "official_g2_authority_digest": snapshot.official_g2_authority_digest,
            "official_cell_evidence_projection_digest":
                snapshot.official_cell_evidence_projection_digest,
            "source_patch_projection_digest": snapshot.source_patch_projection.logical_digest,
            "outcome_build_digest": expected_outcome, "selected_target_count": snapshot.eligible_target_count,
            "reference_key_count": snapshot.reference_key_count,
            "expected_primary_key_count": snapshot.expected_primary_key_count,
            "observed_row_count": pin.row_count, "fit_authority_digest": fit_seal.content_digest,
            "observed_prediction_pin_digest": pin.content_digest,
            "statistics_manifest_digest": statistics_manifest.content_digest,
            "statistics_result_digest": statistics_manifest.result_digest,
            "artifacts": artifacts,
        }
        run_digest = _sha(["task-12-downstream-run-manifest-v1", manifest_core])
        manifest_bytes = _canonical({**manifest_core, "content_digest": run_digest}) + b"\n"
        (partial / "manifest.json").write_bytes(manifest_bytes)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if output_dir.exists():
            raise FileExistsError(output_dir)
        partial.rename(output_dir)
        return DownstreamRunArtifacts(
            output_dir=output_dir, adapter_digest=snapshot.content_digest,
            fit_authority_digest=fit_seal.content_digest, observed_prediction_pin=pin,
            statistics_manifest=statistics_manifest, manifest_sha256=manifest_sha,
            content_digest=run_digest,
        )
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
