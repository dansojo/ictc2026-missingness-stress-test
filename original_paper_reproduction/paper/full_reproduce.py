#!/usr/bin/env python3
"""Run all original ETRI paper calculations from raw data, then external/displays.

Completed steps may be explicitly resumed only after manifest/hash verification.
No historical primitive, mask, fitted-model or prediction cache is accepted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
from private_reference import REFERENCE
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

import pandas as pd
import pyarrow.parquet as pq
import pyarrow.ipc as ipc

from full_prepare import PAPER, sha256, verify_vendor, write_json, emit
from full_g2 import load_stage, G1_PATH
import reproduce as portable

G2_GROUPS = (
    (('contiguous_20pct', 1),),
    (('contiguous_10pct', 1), ('contiguous_40pct', 1)),
    (('outage_1h', 1), ('outage_3h', 1)),
    (('outage_6h', 1), ('evening_critical', 1)),
    (('empirical_gap', 1),),
    (('empirical_gap', 2),),
)
STAGE_NAMES = ('prepare', 'g1', 'g2', 'stress', 'downstream', 'evidence', 'displays')
CSV_NAMES = ('feature_contracts.csv', 'primary_feature_inference.csv', 'participant_primary_deltas.csv',
             'dose_response.csv', 'downstream_condition_summary.csv', 'downstream_feature_profiles.csv',
             'downstream_participant_primary.csv', 'model_specifications.csv')
REQUIRED_OUTPUTS = {
    'prepare': ('primitives', 'baselines', 'representations'),
    'g1': ('pair_summary',),
    'g2_scenario': ('selected_days', 'cell_replays', 'mask_intervals', 'deletion_audit'),
    'g2': ('selected_days', 'cell_replays', 'mask_intervals', 'deletion_audit'),
    'stress': ('target_ledger', 'official_current_crosslink', 'mask_ledger', 'deletion_audit',
               'cell_replays', 'decision', 'family_inference', 'participant_deltas', 'dose_response_family',
               'primary_comparison', 'denominators', 'dose_response_participant'),
    'downstream': ('run/observed_predictions.arrow', 'run/fit/fit_ledger.arrow',
                   'run/fit/fit_state_index.arrow', 'run/fit/fit_states.pack', 'run/statistics/decision.parquet'),
    'evidence': (*CSV_NAMES, 'study_design.json', 'manuscript_evidence.md'),
    'displays': ('fig2_results.pdf', 'fig2_results.png', 'table1_feature_contracts.tex', 'table2_external_transfer.tex'),
}


class ComparisonFailure(ValueError):
    pass


@contextmanager
def pipeline_lock(output: Path):
    """OS advisory lock releases after a crash; an old lock file is harmless."""
    with (output / '.pipeline.lock').open('a+b') as stream:
        if stream.seek(0, 2) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Paper pipeline output is locked by another running process') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_output_roots(output: Path, inputs: list[Path]) -> None:
    output = output.resolve()
    for root in inputs:
        root = root.resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(f'Pipeline output overlaps input: {root}')


def managed_path(output: Path, relative: str) -> Path:
    path = (output / relative).resolve()
    if not path.is_relative_to(output.resolve()) or path == output.resolve():
        raise ValueError(f'Managed output path escapes the pipeline root: {relative}')
    return path


def verify_output_records(directory: Path, document: dict, stage: str) -> None:
    outputs = document.get('outputs')
    if not isinstance(outputs, dict) or not outputs or not set(REQUIRED_OUTPUTS.get(stage, ())).issubset(outputs):
        raise ValueError(f'{stage} lacks required outputs/artifact pins')
    for name, pin in outputs.items():
        if pin.get('path') is None:
            if name in REQUIRED_OUTPUTS.get(stage, ()) or pin.get('rows') != 0 or pin.get('columns') != []:
                raise ValueError(f'{stage}/{name} has an invalid absent artifact')
            continue
        path = portable.resolve_file(directory, pin['path'])
        if sha256(path) != pin.get('sha256') or ('bytes' in pin and path.stat().st_size != pin['bytes']):
            raise ValueError(f'{stage}/{name} output digest/size changed')
        if path.suffix == '.parquet':
            table = pq.ParquetFile(path)
            rows, columns = table.metadata.num_rows, table.schema_arrow.names
        elif path.suffix == '.arrow':
            with path.open('rb') as stream:
                reader = ipc.open_stream(stream)
                columns, rows = reader.schema.names, sum(batch.num_rows for batch in reader)
        elif path.suffix in ('.csv', '.jsonl') and 'rows' in pin:
            table = pd.read_csv(path) if path.suffix == '.csv' else pd.read_json(path, lines=True)
            rows, columns = len(table), list(table.columns)
        else:
            continue
        if rows != pin.get('rows') or columns != pin.get('columns'):
            raise ValueError(f'{stage}/{name} output row/schema declaration mismatch')


def validate_g2_empirical_determinism(document: dict) -> None:
    """Require the aggregate's five checks and rebind them to independent files."""
    tables = ('selected_days', 'cell_replays', 'mask_intervals', 'deletion_audit')
    original_key = 'original_byte_and_physical_identity_validation'
    checks = document.get('empirical_determinism')
    if not isinstance(checks, dict) or set(checks) != {original_key, *tables}:
        raise ValueError('G2 empirical determinism comparison schema is incomplete')
    for table in tables:
        check = checks[table]
        if (not isinstance(check, dict) or set(check) != {'exact_values_dtypes_order', 'byte_identical'}
                or check['exact_values_dtypes_order'] is not True or check['byte_identical'] is not True):
            raise ValueError(f'G2 empirical {table} value/dtype/order/byte comparison failed or missing')
    original = checks[original_key]
    if (not isinstance(original, dict)
            or set(original) != {'independent_exact50_runs', 'hashes_identical', 'first', 'second'}
            or type(original['independent_exact50_runs']) is not int
            or original['independent_exact50_runs'] != 2 or original['hashes_identical'] is not True):
        raise ValueError('G2 empirical original independent-run evidence is incomplete')

    chunks = document['chunks']
    if any(not isinstance(pin, dict) or not isinstance(pin.get('path'), str) or not pin['path'] for pin in chunks):
        raise ValueError('G2 empirical chunk path pins are missing')
    paths = [Path(pin['path']).resolve() for pin in chunks]
    if len(set(paths)) != 9 or len({path.parent for path in paths}) != 1:
        raise ValueError('G2 empirical chunks must have distinct paths under one evidence root')
    pins = {path.name: (path, pin) for path, pin in zip(paths, chunks, strict=True)}
    if not {'empirical_gap', 'empirical_gap-run2'}.issubset(pins):
        raise ValueError('G2 empirical independent run directories are missing')
    runs = []
    for replicate, name in enumerate(('empirical_gap', 'empirical_gap-run2'), start=1):
        directory, pin = pins[name]
        manifest_path = directory / 'manifest.json'
        if not manifest_path.is_file() or sha256(manifest_path) != pin.get('manifest_sha256'):
            raise ValueError('G2 empirical chunk manifest changed or is missing')
        chunk = load_stage(directory, 'g2_scenario')
        if (chunk.get('scenario') != 'empirical_gap' or type(chunk.get('replicate')) is not int
                or chunk['replicate'] != replicate or chunk.get('parameters', {}).get('draws') != 50):
            raise ValueError('G2 empirical chunks do not prove two independent exact50 executions')
        artifacts = {}
        for table in tables:
            artifact = chunk.get('outputs', {}).get(table, {})
            if artifact.get('path') != table + '.parquet':
                raise ValueError(f'G2 empirical {table} artifact role is missing or crosswired')
            path = portable.resolve_file(directory, artifact['path'])
            artifacts[table] = (path, sha256(path))
        runs.append(artifacts)
    for table in tables:
        first, second = runs[0][table], runs[1][table]
        if first[0].samefile(second[0]) or first[1] != second[1]:
            raise ValueError(f'G2 empirical {table} files reuse identity or have different hashes')
    from full_g2 import validate_empirical_replicates
    actual = validate_empirical_replicates(pins['empirical_gap'][0], pins['empirical_gap-run2'][0], paths[0].parent)
    if original != actual:
        raise ValueError('G2 empirical original paths/hashes differ from current physical evidence')


def validate_stage_completion(stage: str, document: dict) -> None:
    expected_shapes = {
        'prepare': {'primitives': (853, 231), 'baselines': (1920, 23), 'representations': (7716, 197)},
        'g2_scenario': {'selected_days': (800, None), 'cell_replays': (40000, None), 'deletion_audit': (40000, None)},
        'g2': {'selected_days': (800, None), 'cell_replays': (320000, None), 'deletion_audit': (320000, None)},
        'stress': {'target_ledger': (40000, None), 'official_current_crosslink': (40000, None),
                   'mask_ledger': (80000, None), 'cell_replays': (80000, None), 'participant_deltas': (40, None),
                   'family_inference': (4, None), 'dose_response_family': (12, None)},
        'downstream': {'run/observed_predictions.arrow': (943392, 42), 'run/fit/fit_ledger.arrow': (160, None),
                       'run/fit/fit_state_index.arrow': (160, None), 'run/statistics/decision.parquet': (1, None)},
    }
    for name, (rows, columns) in expected_shapes.get(stage, {}).items():
        pin = document.get('outputs', {}).get(name, {})
        if pin.get('rows') != rows or (columns is not None and len(pin.get('columns', [])) != columns):
            raise ValueError(f'{stage}/{name} complete study topology is missing')
    if stage == 'g1' and document.get('comparison', {}).get('all_within_reference_precision') is not True:
        raise ValueError('G1 requires a successful reviewed comparison')
    if stage == 'g2':
        if (document.get('logical_replays') != 320000
                or document.get('main_selected_cell_replay_executions') != 360000
                or document.get('additional_calibration_primitive_replays') != 56000
                or len(document.get('chunks', [])) != 9):
            raise ValueError('G2 complete nine-execution topology missing')
        validate_g2_empirical_determinism(document)
    if stage == 'downstream' and (document.get('new_model_fits') != 160
            or document.get('reused_historical_model_fits') != 0 or document.get('prediction_rows') != 943392
            or document.get('expected_primary_key_count') != 314464):
        raise ValueError('Complete downstream study requires 160 fresh fits and 943392 predictions')
    if stage == 'evidence' and (document.get('upstream_new_model_fits') != 160
            or document.get('generated_scientific_report_files') != 10):
        raise ValueError('Evidence does not cover the complete freshly fitted study')
    if stage == 'displays':
        checks = document.get('scientific_csv_comparisons', {})
        if (set(checks) != set(CSV_NAMES) or not all(value.get('exact_csv_values') is True for value in checks.values())
                or document.get('verified_fixed_annotations') != {'event_boundary_abstained': 1104,
                                                                    'event_boundary_structurally_unavailable': 8000}):
            raise ValueError('display scientific comparisons/annotations are incomplete')


def verify_stage(directory: Path, stage: str) -> dict:
    document = load_stage(directory, stage)
    verify_output_records(directory, document, stage)
    validate_stage_completion(stage, document)
    if stage in ('g2_scenario', 'g2'):
        from full_sources import verify_g2_sources
        if document.get('g2_execution_sources') != verify_g2_sources():
            raise ValueError('G2 historical checkpoint source pins are missing or changed')
    if stage == 'g1':
        comparison = document['comparison']
        path = portable.resolve_file(directory, comparison['path'])
        if sha256(path) != comparison['sha256'] or json.loads(path.read_text(encoding='utf-8')).get('all_within_reference_precision') is not True:
            raise ValueError('G1 comparison report changed or failed')
    return document


def validate_stage_lineage(output: Path, stage: str, document: dict, context: dict) -> None:
    inputs = document.get('inputs', {})
    prepare_sha = context['prepare_sha256']
    if stage == 'prepare':
        if {name: pin['sha256'] for name, pin in document.get('canonical_files', {}).items()} != context['canonical_files']:
            raise ValueError('prepare and current canonical inputs are crosswired')
    elif stage == 'g1':
        if inputs.get('prepare_manifest', {}).get('sha256') != prepare_sha:
            raise ValueError('G1 prepare ancestor is crosswired')
    elif stage in ('g2_scenario', 'g2', 'stress', 'downstream'):
        if inputs.get('prepare_manifest_sha256') != prepare_sha:
            raise ValueError(f'{stage} prepare ancestor is crosswired')
        for ancestor in (('g2',) if stage == 'stress' else ('g2', 'stress') if stage == 'downstream' else ()):
            if inputs.get(ancestor + '_manifest_sha256') != sha256(output / ancestor / 'manifest.json'):
                raise ValueError(f'{stage} {ancestor} ancestor is crosswired')
        if stage in ('g2_scenario', 'g2'):
            if (inputs.get('reviewed_g1_file_sha256') != context['g1_file_sha256']
                    or {name: pin['sha256'] for name, pin in inputs.get('canonical_files', {}).items()} != context['canonical_files']):
                raise ValueError('G2 protocol/canonical inputs are crosswired')
        if stage == 'g2':
            expected = {str((output / 'g2_chunks' / (scenario if replicate == 1 else scenario + '-run2')).resolve())
                        for group in G2_GROUPS for scenario, replicate in group}
            chunks = document.get('chunks', [])
            if {str(Path(pin['path']).resolve()) for pin in chunks} != expected:
                raise ValueError('G2 aggregate chunk set is crosswired')
            for pin in chunks:
                if sha256(Path(pin['path']) / 'manifest.json') != pin['manifest_sha256']:
                    raise ValueError('G2 aggregate chunk manifest changed')
        if stage == 'downstream':
            pins = inputs.get('pins', [])
            labels = context.get('labels')
            if labels is not None:
                for key in ('training_labels', 'scoring_labels'):
                    if Path(inputs.get(key, '')).resolve() != labels:
                        raise ValueError('Downstream label input path changed')
                if not any(Path(pin['path']).resolve() == labels and pin['sha256'] == sha256(labels) for pin in pins):
                    raise ValueError('Downstream label input digest changed')
    elif stage == 'evidence':
        for ancestor in ('prepare', 'g2', 'stress', 'downstream'):
            root = output / ancestor
            pin = inputs.get(ancestor, {})
            if Path(pin.get('path', '')).resolve() != root.resolve() or pin.get('manifest_sha256') != sha256(root / 'manifest.json'):
                raise ValueError(f'evidence {ancestor} ancestor is crosswired')
    elif stage == 'displays':
        if (inputs.get('evidence_manifest_sha256') != sha256(output / 'evidence/manifest.json')
                or inputs.get('external_manifest_sha256') != sha256(output / 'external/raw_results/external_transfer_manifest.json')):
            raise ValueError('display evidence/external ancestor is crosswired')


def verify_preprocessing(preprocessed: Path, raw: Path) -> dict:
    sys.path.insert(0, str(PAPER.parent / 'leaderboard'))
    import data_preparation as preparation
    pre_manifest = preprocessed / 'preprocessing_run.json'
    document = json.loads(pre_manifest.read_text(encoding='utf-8'))
    checks = document.get('checks', {})
    required_checks = {'feature_count_3155', 'feature_names_unique', 'train_rows_preserved', 'test_rows_preserved',
                       'train_keys_preserved', 'test_keys_preserved', 'train_features_present', 'test_features_present',
                       'no_target_features', 'raw_input_unchanged'}
    if (document.get('schema') != 'ictc2026-leaderboard-preprocessing-run-v1' or document.get('status') != 'complete'
            or not required_checks.issubset(checks) or not all(value is True for value in checks.values())
            or document.get('method_changes') != [] or document.get('feature_count') != 3155):
        raise ValueError('raw preprocessing lacks complete verified checks')
    pins = document.get('input_sha256', {})
    if set(pins) != set(preparation.RAW_FILES) or document.get('input_sha256_after') != pins:
        raise ValueError('raw preprocessing input set is incomplete or changed')
    if [row.get('stage') for row in document.get('stages', [])] != list(preparation.STAGES) or any(
            row.get('status') != 'complete' for row in document['stages']):
        raise ValueError('raw preprocessing stage topology is incomplete')
    source_pins = preparation.verify_vendor()
    if document.get('vendor_sha256') != {pin['vendor_relative_path']: pin['sha256'] for pin in source_pins}:
        raise ValueError('raw preprocessing scientific source pins changed')
    for name, digest in pins.items():
        if sha256(portable.resolve_file(raw, name)) != digest:
            raise ValueError(f'raw preprocessing input changed: {name}')
    finals = document.get('final_outputs', [])
    if {pin['path'] for pin in finals} != {'project/data/processed/' + name for name in preparation.FINAL_FILES}:
        raise ValueError('raw preprocessing final output set is incomplete')
    for pin in finals:
        path = portable.resolve_file(preprocessed, pin['path'])
        if path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
            raise ValueError('raw preprocessing final output changed')
    canonical = preprocessed / 'project/data/canonical'
    names = {Path(name).stem.removeprefix('ch2025_') + '_canonical.parquet' for name in preparation.RAW_FILES[2:]}
    paths = list(canonical.glob('*_canonical.parquet'))
    if {path.name for path in paths} != names:
        raise ValueError('raw preprocessing canonical set is incomplete')
    return {'manifest_sha256': sha256(pre_manifest), 'canonical_files': {path.name: sha256(path) for path in paths},
            'raw_files': pins, 'final_outputs': finals}


def external_context(args) -> dict:
    roots = portable.resolve_roots(SimpleNamespace(data_root=args.data_root, inputs_json=args.external_inputs_json))
    sources = portable.verify_vendor()
    entries = json.loads((PAPER / 'input_manifest.json').read_text(encoding='utf-8'))['groups']['external']
    checked = portable.verify_entries(entries, roots)
    if not checked:
        raise ValueError('external input manifest is empty')
    return {'roots': roots, 'sources': sources, 'checked': checked}


def verify_external(directory: Path, context: dict) -> dict:
    document = json.loads((directory / 'run_manifest.json').read_text(encoding='utf-8'))
    if (document.get('status') != 'COMPLETE_WITH_STATED_SCOPE' or document.get('command') != 'external'
            or not context['checked'] or document.get('input_files_verified') != context['checked']
            or document.get('source_files_verified') != context['sources']):
        raise ValueError('external input/source provenance is incomplete or changed')
    comparison = portable.compare_results(directory / 'raw_results', REFERENCE / 'external')
    if comparison.get('scientific_equal') is not True:
        raise ComparisonFailure('external outputs fail current scientific comparison')
    return {'manifest_sha256': sha256(directory / 'run_manifest.json'), 'comparison': comparison,
            'outputs': {path.relative_to(directory).as_posix(): sha256(path)
                        for path in sorted((directory / 'raw_results').rglob('*')) if path.is_file()}}


def verify_display_science(output: Path) -> None:
    try:
        for name in CSV_NAMES:
            pd.testing.assert_frame_equal(pd.read_csv(output / 'evidence' / name),
                                          pd.read_csv(REFERENCE / 'etri' / name), check_exact=True)
        for name in ('external_transfer_summary.csv', 'external_participant_deltas.csv', 'external_denominators.csv'):
            pd.testing.assert_frame_equal(pd.read_csv(output / 'external/raw_results' / name),
                                          pd.read_csv(REFERENCE / 'external' / name), check_exact=True)
    except AssertionError as error:
        raise ComparisonFailure('Actual display scientific CSVs differ from the fixed references') from error


def completed_stage(directory: Path, stage: str, *, resume: bool) -> bool:
    if not directory.exists():
        return False
    if not resume:
        raise FileExistsError(f'Existing stage requires explicit --resume: {directory}')
    verify_stage(directory, stage)
    return True


def run_child(command: list[str], log: Path) -> None:
    if log.exists():
        raise FileExistsError(f'Refusing to overwrite execution log: {log}')
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        try:
            while process.poll() is None:
                emit(stage='paper_pipeline', child=log.stem, status='running', pid=process.pid,
                     elapsed_seconds=round(time.perf_counter() - started, 1))
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f'Child exited {process.returncode}; see {log}')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            raise
    emit(stage='paper_pipeline', child=log.stem, status='complete',
         seconds=time.perf_counter() - started)


def execute(args) -> dict:
    from full_sources import verify_g2_sources
    g2_source_pins = verify_g2_sources()
    source_pins = verify_vendor()
    output, raw = args.output_dir.resolve(), args.raw_root.resolve(strict=True)
    external_info = external_context(args)
    preprocessed = args.preprocessed_root.resolve() if args.preprocessed_root else output / 'preprocessing'
    protected = [raw, external_info['roots']['external_project'], PAPER / 'full_vendor',
                 PAPER / 'vendor', REFERENCE, PAPER / 'protocol', PAPER / 'g2_vendor',
                 PAPER.parent / 'leaderboard/vendor']
    if args.preprocessed_root and preprocessed != output / 'preprocessing':
        protected.append(preprocessed)
    if args.external_inputs_json:
        protected.append(args.external_inputs_json.resolve())
    reference_sdd = getattr(args, 'reference_sdd', None)
    if reference_sdd:
        protected.append(reference_sdd.resolve())
    validate_output_roots(output, protected)
    for name in (*STAGE_NAMES, 'g2_chunks', 'external', 'logs', 'attempts', 'comparisons', 'pipeline_manifest.json'):
        managed_path(output, name)
    if not args.preprocessed_root or preprocessed == output / 'preprocessing':
        managed_path(output, 'preprocessing')
    if output.exists() and not args.resume:
        raise FileExistsError('Use a new output directory or explicitly resume verified completed stages')
    config = {'raw_root': str(raw), 'preprocessed_root': str(preprocessed),
              'data_root': str(args.data_root.resolve()),
              'external_inputs_json': str(args.external_inputs_json.resolve()) if args.external_inputs_json else None,
              'external_inputs_json_sha256': sha256(args.external_inputs_json) if args.external_inputs_json else None,
              'reference_sdd': str(reference_sdd.resolve()) if reference_sdd else None}
    pipeline_path = output / 'pipeline_manifest.json'
    previous = None
    if pipeline_path.exists():
        previous = json.loads(pipeline_path.read_text(encoding='utf-8'))
        if previous.get('schema_name') != 'independent_etri_paper_pipeline' or previous.get('config') != config:
            raise ValueError('Resume configuration differs from this independent run')
        if previous.get('scientific_sources') != source_pins:
            raise ValueError('Resume scientific source pins differ')
        if previous.get('g2_execution_sources') != g2_source_pins:
            raise ValueError('Resume historical G2 checkpoint source pins differ')
    pre_manifest = preprocessed / 'preprocessing_run.json'
    preprocessing = verify_preprocessing(preprocessed, raw) if pre_manifest.exists() else None
    if args.preprocessed_root and preprocessing is None:
        raise ValueError('Provided preprocessed root lacks completed raw-run evidence')
    if preprocessed.exists() and preprocessing is None:
        raise ValueError('Existing preprocessing directory is partial; never overwrite it')
    canonical = preprocessed / 'project/data/canonical'

    def context() -> dict:
        return {'prepare_sha256': sha256(output / 'prepare/manifest.json') if (output / 'prepare/manifest.json').is_file() else None,
                'canonical_files': preprocessing['canonical_files'] if preprocessing else {},
                'g1_file_sha256': sha256(G1_PATH), 'labels': raw / 'ch2026_metrics_train.csv'}

    def check_stage(name: str, kind: str) -> dict:
        directory = managed_path(output, name)
        actual = verify_stage(directory, kind)
        validate_stage_lineage(output, kind, actual, context())
        if kind == 'g2_scenario':
            scenario = directory.name.removesuffix('-run2')
            replicate = 2 if directory.name.endswith('-run2') else 1
            expected_extra = scenario == 'contiguous_20pct' and replicate == 1
            if (actual.get('scenario') != scenario or actual.get('replicate') != replicate
                    or actual.get('parameters') != {'root_seed': 42, 'draws': 50, 'max_days': 10,
                         'run_calibration': expected_extra, 'run_modality': expected_extra}):
                raise ValueError('G2 scenario identity/settings changed')
        return actual

    # Read-only resume preflight precedes all child processes and manifest writes.
    if output.exists():
        if any((output / name).exists() for name in STAGE_NAMES) and preprocessing is None:
            raise ValueError('Existing scientific stages lack raw preprocessing provenance')
        for name in STAGE_NAMES:
            if (output / name).exists():
                check_stage(name, name)
        for group in G2_GROUPS:
            for scenario, replicate in group:
                name = 'g2_chunks/' + (scenario if replicate == 1 else scenario + '-run2')
                if (output / name).exists():
                    check_stage(name, 'g2_scenario')
        if (output / 'external').exists():
            verify_external(output / 'external', external_info)
        if (output / 'displays').exists():
            verify_display_science(output)
        if previous:
            if previous.get('preprocessing') and previous['preprocessing'] != preprocessing:
                raise ValueError('Previously pinned preprocessing inputs/outputs changed')
            for name, pin in previous.get('completed', {}).items():
                if sha256(output / name / 'manifest.json') != pin['manifest_sha256']:
                    raise ValueError(f'Previously completed stage manifest changed: {name}')

    preflight_pipeline_sha = sha256(pipeline_path) if pipeline_path.exists() else None
    output.mkdir(parents=True, exist_ok=True)
    with pipeline_lock(output):
        if (sha256(pipeline_path) if pipeline_path.exists() else None) != preflight_pipeline_sha:
            raise ValueError('Pipeline changed during resume preflight; restart verification')
        attempt_id = str(time.time_ns()) + '-' + uuid.uuid4().hex[:8]
        attempt_path = output / 'attempts' / (attempt_id + '.json')
        attempt_path.parent.mkdir(exist_ok=True)
        # A retry may reach a child that failed before creating its stage root.
        # Preserve the previous attempt's exclusive logs and give this one its
        # own namespace, instead of overwriting or colliding with old evidence.
        logs = output / 'logs' / attempt_id
        base = [sys.executable, '-u', '-B']
        manifest = {'schema_name': 'independent_etri_paper_pipeline', 'status': 'running',
                    'attempt_id': attempt_id, 'config': config, 'fresh_scientific_calculations_required': True,
                    'logs_directory': logs.relative_to(output).as_posix(),
                    'new_model_fits': 0, 'completed': {}, 'scientific_sources': source_pins,
                    'g2_execution_sources': g2_source_pins,
                    'previous_pipeline_manifest_sha256': sha256(pipeline_path) if pipeline_path.exists() else None,
                    'driver_sources': {p.name: sha256(p) for p in sorted(PAPER.glob('full_*.py'))}}
        if previous:
            (attempt_path.parent / (attempt_id + '.previous.json')).write_bytes(pipeline_path.read_bytes())
        started, current_step = time.perf_counter(), 'preprocessing'

        def save() -> None:
            write_json(attempt_path, manifest)
            temporary = pipeline_path.with_name('pipeline_manifest.' + attempt_id + '.tmp')
            write_json(temporary, manifest)
            temporary.replace(pipeline_path)

        def stage(name: str, kind: str, command: list[str]) -> None:
            nonlocal current_step
            current_step = name
            directory = output / name
            if not completed_stage(directory, kind, resume=args.resume):
                run_child(base + command, logs / f'{name.replace("/", "_")}.log')
            actual = check_stage(name, kind)
            manifest['completed'][name] = {'manifest_sha256': sha256(directory / 'manifest.json'),
                                           'seconds': actual.get('seconds')}
            save()

        save()
        try:
            if preprocessing is None:
                run_child(base + [str(PAPER.parent / 'leaderboard/data_preparation.py'),
                                 '--data-root', str(raw), '--output-dir', str(preprocessed)], logs / 'raw_preprocessing.log')
                preprocessing = verify_preprocessing(preprocessed, raw)
            manifest['preprocessing'] = preprocessing
            stage('prepare', 'prepare', [str(PAPER / 'full_prepare.py'), 'run',
                  '--canonical-root', str(canonical), '--output-dir', str(output / 'prepare')])
            stage('g1', 'g1', [str(PAPER / 'full_g1.py'), 'run', '--prepare-dir', str(output / 'prepare'),
                  '--output-dir', str(output / 'g1'), '--reviewed-g1', str(G1_PATH)])

            def scenario_run(item):
                scenario, replicate = item
                name = scenario if replicate == 1 else f'{scenario}-run2'
                directory = output / 'g2_chunks' / name
                if not completed_stage(directory, 'g2_scenario', resume=args.resume):
                    run_child(base + [str(PAPER / 'full_g2.py'), 'scenario', '--prepare-dir', str(output / 'prepare'),
                              '--canonical-root', str(canonical), '--output-dir', str(directory),
                              '--scenario', scenario, '--replicate', str(replicate)], logs / f'g2_{name}.log')
                check_stage('g2_chunks/' + name, 'g2_scenario')
                return name

            current_step = 'g2_scenarios'
            for group in G2_GROUPS:
                with ThreadPoolExecutor(max_workers=len(group)) as executor:
                    for name in executor.map(scenario_run, group):
                        key = 'g2_chunks/' + name
                        manifest['completed'][key] = {'manifest_sha256': sha256(output / key / 'manifest.json')}
                        save()
            stage('g2', 'g2', [str(PAPER / 'full_g2.py'), 'aggregate', '--chunks-root', str(output / 'g2_chunks'),
                              '--output-dir', str(output / 'g2')])
            stage('stress', 'stress', [str(PAPER / 'full_stress.py'), '--prepare-dir', str(output / 'prepare'),
                  '--canonical-root', str(canonical), '--g2-dir', str(output / 'g2'), '--output-dir', str(output / 'stress')])
            labels = raw / 'ch2026_metrics_train.csv'
            stage('downstream', 'downstream', [str(PAPER / 'full_downstream.py'), '--prepare-dir', str(output / 'prepare'),
                  '--stress-dir', str(output / 'stress'), '--training-labels', str(labels), '--scoring-labels', str(labels),
                  '--output-dir', str(output / 'downstream')])
            manifest['new_model_fits'] = 160
            manifest['etri_scientific_experiment_executed'] = True
            stage('evidence', 'evidence', [str(PAPER / 'full_evidence.py'), 'evidence',
                  '--prepare-dir', str(output / 'prepare'), '--g2-dir', str(output / 'g2'),
                  '--stress-dir', str(output / 'stress'), '--downstream-dir', str(output / 'downstream'),
                  '--output-dir', str(output / 'evidence')])
            current_step = 'external'
            external = output / 'external'
            if not external.exists():
                command = [str(PAPER / 'reproduce.py'), 'external', '--data-root', str(args.data_root), '--output', str(external)]
                if args.external_inputs_json:
                    command += ['--inputs-json', str(args.external_inputs_json)]
                run_child(base + command, logs / 'external.log')
            elif not args.resume:
                raise FileExistsError('Existing external result requires --resume')
            manifest['external'] = verify_external(external, external_context(args))
            manifest['scientific_experiment_executed'] = True
            save()
            verify_display_science(output)
            stage('displays', 'displays', [str(PAPER / 'full_evidence.py'), 'displays',
                  '--evidence-dir', str(output / 'evidence'), '--external-results-dir', str(external / 'raw_results'),
                  '--output-dir', str(output / 'displays')])
            manifest['scientific_experiment_executed'] = True
            if reference_sdd:
                current_step = 'detailed_comparison'
                report = output / 'comparisons' / (attempt_id + '.json')
                report.parent.mkdir(exist_ok=True)
                manifest['detailed_comparison'] = {'path': report.relative_to(output).as_posix(), 'reference_role': 'comparison_only'}
                save()
                run_child(base + [str(PAPER / 'full_compare.py'), '--actual-root', str(output),
                                  '--reference-sdd', str(reference_sdd), '--report', str(report)],
                          logs / ('comparison_' + attempt_id + '.log'))
                compared = json.loads(report.read_text(encoding='utf-8'))
                if (compared.get('schema_name') != 'independent_etri_detailed_comparison_v1'
                        or compared.get('status') != 'complete' or compared.get('full_scope') is not True):
                    raise ValueError('Detailed full comparison is incomplete or failed')
                manifest['detailed_comparison']['sha256'] = sha256(report)
            current_step = 'final_verification'
            for name, pin in manifest['completed'].items():
                kind = 'g2_scenario' if name.startswith('g2_chunks/') else name
                check_stage(name, kind)
                if sha256(output / name / 'manifest.json') != pin['manifest_sha256']:
                    raise ValueError(f'Stage changed before final completion: {name}')
            if verify_preprocessing(preprocessed, raw) != preprocessing:
                raise ValueError('Preprocessing/raw/canonical inputs changed before final completion')
            if verify_external(external, external_context(args)) != manifest['external']:
                raise ValueError('External inputs/outputs changed before final completion')
            verify_display_science(output)
            if reference_sdd and sha256(report) != manifest['detailed_comparison']['sha256']:
                raise ComparisonFailure('Detailed comparison report changed before completion')
            verify_vendor()
            if verify_g2_sources() != g2_source_pins:
                raise ValueError('G2 checkpoint sources changed during the pipeline')
            if {p.name: sha256(p) for p in sorted(PAPER.glob('full_*.py'))} != manifest['driver_sources']:
                raise ValueError('Execution driver sources changed during the pipeline')
            expected_stages = set(STAGE_NAMES) | {'g2_chunks/' + (scenario if replicate == 1 else scenario + '-run2')
                                               for group in G2_GROUPS for scenario, replicate in group}
            if set(manifest['completed']) != expected_stages:
                raise ValueError('Complete paper pipeline stage set is incomplete')
            manifest.update(status='complete', seconds=time.perf_counter() - started,
                            completion_scope='Fresh raw-to-ETRI/G1/G2/stress/160 fits/predictions/statistics; external raw results; exact scientific display comparisons and rendered files.',
                            detailed_archive_comparison_requested=bool(reference_sdd))
            save()
            return manifest
        except BaseException as error:
            manifest.update(status='comparison_failed' if current_step == 'detailed_comparison' or isinstance(error, ComparisonFailure) else 'failed',
                            failed_step=current_step, seconds=time.perf_counter() - started,
                            error_type=type(error).__name__, error=str(error))
            save()
            (output / f'pipeline_failure_{attempt_id}.txt').write_text(traceback.format_exc(), encoding='utf-8')
            raise

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=Path('/data/raw'))
    parser.add_argument('--data-root', type=Path, default=Path('/data'))
    parser.add_argument('--output-dir', type=Path, default=Path('/data/output/paper_full'))
    parser.add_argument('--preprocessed-root', type=Path, help='Explicit earlier fresh raw-run directory with its preprocessing manifest')
    parser.add_argument('--external-inputs-json', type=Path, help='Optional explicit external dataset location mapping')
    parser.add_argument('--reference-sdd', type=Path, help='Optional read-only archived SDD for strict full-table comparison; not a calculation input')
    parser.add_argument('--resume', action='store_true', help='Resume this independent run after verifying completed stages; never skip running or failed stages')
    result = execute(parser.parse_args())
    emit(stage='paper_pipeline', status=result['status'], new_model_fits=result['new_model_fits'])


if __name__ == '__main__':
    main()
