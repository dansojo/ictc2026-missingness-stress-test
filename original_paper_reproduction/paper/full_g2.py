#!/usr/bin/env python3
"""Independent fresh execution of all original G2 scenarios and checks.

Scientific calls: run_task4c2_real.py:142-164,357-370 and
aggregate_task4c2.py:335-445. Original modules are unchanged and hash checked.
Historical review metadata is a pinned protocol dependency; new results are
independent reproduction evidence, not a reissued historical release approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import pandas as pd

from full_prepare import (PAPER, create_fresh_output, emit, save_frame, sha256,
                          verify_vendor, write_json)
from full_sources import load_g2_modules, verify_g2_sources

SCENARIOS = ('contiguous_10pct', 'contiguous_20pct', 'contiguous_40pct',
             'outage_1h', 'outage_3h', 'outage_6h', 'evening_critical', 'empirical_gap')
G1_EXPECTED = 'a05199eada5a13f54e96393bd3a1a1920390bbd8aba83c7f15a4b4d790062fcb'
G1_PATH = PAPER / 'protocol/task-4a-reviewed-g1-pair-summary.json'


def load_stage(directory: Path, stage: str) -> dict:
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('schema_name') != 'independent_etri_reproduction'
            or manifest.get('stage') != stage or manifest.get('status') != 'complete'
            or manifest.get('fresh_execution') is not True):
        raise ValueError(f'{stage} must be a completed independent fresh execution')
    for name, record in manifest['outputs'].items():
        if record.get('path') is None:
            continue
        path = (directory / record['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != record['sha256']:
            raise ValueError(f'Invalid upstream artifact: {stage}/{name}')
    return manifest


def extra_calculations(scenario: str, replicate: int) -> bool:
    if scenario not in SCENARIOS or replicate not in (1, 2):
        raise ValueError('Invalid frozen G2 scenario or replicate')
    if replicate == 2 and scenario != 'empirical_gap':
        raise ValueError('Only empirical_gap has the second determinism execution')
    return scenario == 'contiguous_20pct' and replicate == 1


def save_table(directory: Path, name: str, frame: pd.DataFrame) -> dict:
    if not len(frame.columns):
        return {'path': None, 'sha256': None, 'rows': len(frame), 'columns': [], 'format': 'none'}
    mixed = any(pd.api.types.is_object_dtype(frame[col].dtype)
                and (pd.api.types.infer_dtype(frame[col], skipna=True).startswith('mixed')
                     or pd.api.types.infer_dtype(frame[col], skipna=True) in {'unknown', 'complex'})
                for col in frame.columns)
    if not mixed:
        record = save_frame(directory, name, frame)
        record['format'] = 'parquet'
        return record
    path = directory / f'{name}.jsonl'
    frame.to_json(path, orient='records', lines=True, date_format='iso', date_unit='us', force_ascii=False)
    return {'path': path.name, 'sha256': sha256(path), 'rows': len(frame),
            'columns': list(frame.columns), 'dtypes': list(map(str, frame.dtypes)),
            'bytes': path.stat().st_size, 'format': 'jsonl'}


def read_table(directory: Path, manifest: dict, name: str) -> pd.DataFrame:
    record = manifest['outputs'][name]
    if record['path'] is None:
        return pd.DataFrame(columns=record['columns'])
    path = directory / record['path']
    if sha256(path) != record['sha256']:
        raise ValueError(f'Table changed: {path}')
    if record.get('format') == 'jsonl':
        logical_dtypes = dict(zip(record['columns'], record['dtypes'], strict=True))
        frame = pd.read_json(path, orient='records', lines=True, dtype=logical_dtypes)
    else:
        frame = pd.read_parquet(path)
        # Original aggregate_task4c2.py:47-49,229-246 restores this logical
        # object column after Arrow represents its all-integer values as int64.
        if name == 'confidence_ablation' and 'sensor_day_id' in frame:
            frame['sensor_day_id'] = frame['sensor_day_id'].astype('object')
        # Prepared personalization declares donor_subject as object. Arrow
        # stores its strings, which pandas 3 reads as str; restore only this
        # known logical column before the complete dtype declaration check.
        if name == 'representations' and 'donor_subject' in frame:
            frame['donor_subject'] = frame['donor_subject'].astype('object')
    if len(frame) != record['rows'] or list(frame.columns) != record['columns']:
        raise ValueError(f'Table declaration mismatch: {path}')
    if 'dtypes' in record and list(map(str, frame.dtypes)) != record['dtypes']:
        raise ValueError(f'Table logical dtype mismatch: {path}')
    return frame


def validate_empirical_replicates(first: Path, second: Path, evidence_root: Path) -> dict:
    verify_vendor()
    from semantic_indicators.reliability import compare_empirical_replay_artifacts
    return compare_empirical_replay_artifacts(first, second, evidence_root=evidence_root)


def load_sensors(canonical_root: Path, pinned: dict, historical=None) -> dict[str, pd.DataFrame]:
    historical = historical or load_g2_modules()
    PRIMITIVE_DEFINITIONS = historical.contracts.PRIMITIVE_DEFINITIONS
    CORE_G2_PRIMITIVES = historical.reliability.CORE_G2_PRIMITIVES
    for filename, record in pinned.items():
        if sha256(canonical_root / filename) != record['sha256']:
            raise ValueError(f'Canonical input differs from fresh prepare: {filename}')
    definitions = {definition.name: definition for definition in PRIMITIVE_DEFINITIONS}
    columns_by_file = {}
    for primitive in CORE_G2_PRIMITIVES:
        definition = definitions[primitive]
        columns = columns_by_file.setdefault(definition.sensor_file, ['subject_id', 'timestamp'])
        for column in (definition.value_column, definition.validity_column):
            if column is not None and column not in columns:
                columns.append(column)
    return {filename: pd.read_parquet(canonical_root / filename, columns=columns)
            for filename, columns in columns_by_file.items()}


def run_scenario(prepare_dir: Path, canonical_root: Path, output_dir: Path,
                 scenario: str, replicate: int = 1) -> dict:
    extra = extra_calculations(scenario, replicate)
    sources = verify_vendor()
    historical = load_g2_modules()
    from semantic_indicators.reliability import validate_reviewed_g1_artifact
    prepare = load_stage(prepare_dir, 'prepare')
    g1_artifact = json.loads(G1_PATH.read_text(encoding='utf-8'))
    strict_g1 = validate_reviewed_g1_artifact(g1_artifact, expected_sha256=G1_EXPECTED)
    g1 = historical.reliability.validate_reviewed_g1_artifact(g1_artifact, expected_sha256=G1_EXPECTED)
    pd.testing.assert_frame_equal(g1, strict_g1, check_exact=True, check_dtype=True)
    sensors = load_sensors(canonical_root, prepare['canonical_files'], historical)
    frames = {name: read_table(prepare_dir, prepare, name)
              for name in ('primitives', 'baselines', 'representations')}
    output = create_fresh_output(output_dir, [prepare_dir, canonical_root, PAPER / 'full_vendor', PAPER / 'g2_vendor'])
    started = time.perf_counter()
    manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'g2_scenario',
                'status': 'running', 'fresh_execution': True, 'scenario': scenario, 'replicate': replicate,
                'parameters': {'root_seed': 42, 'draws': 50, 'max_days': 10,
                               'run_calibration': extra, 'run_modality': extra},
                'inputs': {'prepare_manifest_sha256': sha256(prepare_dir / 'manifest.json'),
                           'canonical_files': prepare['canonical_files'],
                           'reviewed_g1_file_sha256': sha256(G1_PATH), 'reviewed_g1_content_digest': G1_EXPECTED},
                'scientific_sources': sources, 'g2_execution_sources': historical.provenance, 'outputs': {},
                'driver_sources': {name: sha256(PAPER / name) for name in ('full_g2.py', 'full_prepare.py', 'full_sources.py')},
                'provenance_note': 'G2 numerical execution uses its exact 808a6f6 checkpoint; current full_vendor supplies upstream preparation and strict validation. Legacy grid flags are scientific settings, not a release approval.'}
    write_json(output / 'manifest.json', manifest)
    emit(stage='g2', scenario=scenario, replicate=replicate, status='running', draws=50)
    try:
        result = historical.reliability.evaluate_g2_reliability(frames['primitives'], sensors,
            frozen_population_baselines=frames['baselines'], upstream_pair_summary=g1,
            root_seed=42, draws=50, scenarios=(scenario,), max_days=10,
            run_calibration=extra, run_modality=extra,
            representations=frames['representations'] if extra else None)
        for name in result.__dataclass_fields__:
            frame = getattr(result, name)
            if isinstance(frame, pd.DataFrame):
                manifest['outputs'][name] = save_table(output, name, frame)
        if len(result.selected_days) != 800 or len(result.cell_replays) != 40000:
            raise ValueError('Fresh G2 scenario does not have the complete 800 x 50 topology')
        verify_vendor()
        if verify_g2_sources() != manifest['g2_execution_sources']:
            raise ValueError('G2 checkpoint sources changed during execution')
        if sha256(prepare_dir / 'manifest.json') != manifest['inputs']['prepare_manifest_sha256']:
            raise ValueError('Prepare manifest changed during G2 execution')
        manifest.update(status='complete', seconds=time.perf_counter() - started)
        write_json(output / 'manifest.json', manifest)
        emit(stage='g2', scenario=scenario, replicate=replicate, status='complete', seconds=manifest['seconds'])
        return manifest
    except BaseException as error:
        manifest.update(status='failed', seconds=time.perf_counter() - started, error=str(error))
        write_json(output / 'manifest.json', manifest)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def aggregate(chunks_root: Path, output_dir: Path, prepare_dir: Path | None = None) -> dict:
    current_sources = verify_vendor()
    historical_sources = verify_g2_sources()
    from semantic_indicators.reliability import finalize_g2_streamed_results, validate_selected_replay_topology
    roots = [chunks_root / scenario for scenario in SCENARIOS]
    roots.append(chunks_root / 'empirical_gap-run2')
    stages = [load_stage(root, 'g2_scenario') for root in roots]
    prepare_dir = prepare_dir or chunks_root.parent / 'prepare'
    current_prepare = load_stage(prepare_dir, 'prepare')
    if (sha256(prepare_dir / 'manifest.json') != stages[0]['inputs']['prepare_manifest_sha256']
            or current_prepare['canonical_files'] != stages[0]['inputs']['canonical_files']
            or current_sources != stages[0]['scientific_sources']
            or current_prepare['scientific_sources'] != current_sources):
        raise ValueError('Current physical prepare/scientific sources do not match G2 inputs')
    for record in current_prepare['inputs']:
        path = Path(record['path'])
        if sha256(path) != record['sha256']:
            raise ValueError(f'Current physical canonical input changed: {path}')
    for i, stage in enumerate(stages):
        expected_scenario, replicate = (SCENARIOS[i], 1) if i < 8 else ('empirical_gap', 2)
        extra = extra_calculations(expected_scenario, replicate)
        if (stage['scenario'] != expected_scenario or stage['replicate'] != replicate
                or stage['parameters'] != {'root_seed': 42, 'draws': 50, 'max_days': 10,
                                           'run_calibration': extra, 'run_modality': extra}
                or stage['inputs'] != stages[0]['inputs']
                or stage.get('g2_execution_sources') != historical_sources
                or stage['scientific_sources'] != stages[0]['scientific_sources']):
            raise ValueError('G2 chunks have different inputs/source/settings or incorrect identities')
    selected = read_table(roots[0], stages[0], 'selected_days')
    cell_frames, mask_frames, deletion_frames = [], [], []
    for root, stage in zip(roots[:8], stages[:8]):
        pd.testing.assert_frame_equal(selected, read_table(root, stage, 'selected_days'), check_exact=True)
        cells = read_table(root, stage, 'cell_replays')
        masks = read_table(root, stage, 'mask_intervals')
        deletion = read_table(root, stage, 'deletion_audit')
        for table in (cells, deletion):
            validate_selected_replay_topology(table, selected_days=selected,
                expected_scenarios=(stage['scenario'],), expected_draws=50)
        cell_frames.append(cells)
        mask_frames.append(masks)
        deletion_frames.append(deletion)
    determinism = {'original_byte_and_physical_identity_validation':
                   validate_empirical_replicates(roots[7], roots[8], chunks_root)}
    for name in ('selected_days', 'cell_replays', 'mask_intervals', 'deletion_audit'):
        first, second = read_table(roots[7], stages[7], name), read_table(roots[8], stages[8], name)
        pd.testing.assert_frame_equal(first, second, check_exact=True, check_dtype=True)
        determinism[name] = {'exact_values_dtypes_order': True,
                            'byte_identical': stages[7]['outputs'][name]['sha256'] == stages[8]['outputs'][name]['sha256']}
    cells = pd.concat(cell_frames, ignore_index=True)
    masks = pd.concat(mask_frames, ignore_index=True)
    deletion = pd.concat(deletion_frames, ignore_index=True)
    if len(selected) != 800 or len(cells) != 320000 or len(deletion) != 320000:
        raise ValueError('G2 aggregate is incomplete')
    output = create_fresh_output(output_dir, [chunks_root, prepare_dir, PAPER / 'full_vendor', PAPER / 'g2_vendor'])
    started = time.perf_counter()
    manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'g2',
                'status': 'running', 'fresh_execution': True, 'inputs': stages[0]['inputs'],
                'parameters': {'root_seed': 42, 'draws': 50, 'scenarios': list(SCENARIOS), 'max_days': 10},
                'chunks': [{'path': str(root), 'manifest_sha256': sha256(root / 'manifest.json')} for root in roots],
                'empirical_determinism': determinism, 'outputs': {},
                'scientific_sources': current_sources, 'g2_execution_sources': historical_sources,
                'provenance_note': 'Current strict finalizer aggregates freshly executed checkpoint-pinned G2 rows without changing their historical deletion identity.',
                'driver_sources': {name: sha256(PAPER / name) for name in ('full_g2.py', 'full_prepare.py', 'full_sources.py')}}
    write_json(output / 'manifest.json', manifest)
    try:
        finalized = finalize_g2_streamed_results(cells, selected_days=selected,
            participant_roster=tuple(sorted(selected['subject_id'].unique(), key=str)),
            reviewed_g1_artifact=json.loads(G1_PATH.read_text(encoding='utf-8')),
            expected_g1_artifact_sha256=G1_EXPECTED, root_seed='42', draws=50,
            scenarios=SCENARIOS, max_days=10)
        tables = {'selected_days': selected, 'cell_replays': cells,
                  'mask_intervals': masks, 'deletion_audit': deletion}
        tables.update({name: getattr(finalized, name) for name in finalized.__dataclass_fields__})
        for name in ('calibration_sensitivity', 'calibration_summary', 'modality_sensitivity', 'modality_summary'):
            tables[name] = read_table(roots[1], stages[1], name)
        for name, frame in tables.items():
            manifest['outputs'][name] = save_table(output, name, frame)
        for root, pin in zip(roots, manifest['chunks'], strict=True):
            if sha256(root / 'manifest.json') != pin['manifest_sha256']:
                raise ValueError('G2 chunk manifest changed during aggregate')
        if sha256(prepare_dir / 'manifest.json') != stages[0]['inputs']['prepare_manifest_sha256']:
            raise ValueError('Prepare changed during G2 aggregate')
        verify_vendor()
        if verify_g2_sources() != historical_sources:
            raise ValueError('G2 checkpoint sources changed during aggregate')
        manifest.update(status='complete', seconds=time.perf_counter() - started,
                        selected_cells=800, logical_replays=320000,
                        main_selected_cell_replay_executions=360000,
                        additional_calibration_primitive_replays=56000)
        write_json(output / 'manifest.json', manifest)
        emit(stage='g2_aggregate', status='complete', logical_replays=320000, seconds=manifest['seconds'])
        return manifest
    except BaseException as error:
        manifest.update(status='failed', seconds=time.perf_counter() - started, error=str(error))
        write_json(output / 'manifest.json', manifest)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('scenario')
    run.add_argument('--prepare-dir', type=Path, default=Path('/data/output/paper/prepare'))
    run.add_argument('--canonical-root', type=Path, default=Path('/data/prepared/project/data/canonical'))
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--scenario', choices=SCENARIOS, required=True)
    run.add_argument('--replicate', type=int, choices=(1, 2), default=1)
    final = sub.add_parser('aggregate')
    final.add_argument('--chunks-root', type=Path, default=Path('/data/output/paper/g2_chunks'))
    final.add_argument('--output-dir', type=Path, default=Path('/data/output/paper/g2'))
    final.add_argument('--prepare-dir', type=Path, help='Defaults to the prepare sibling of chunks-root')
    args = parser.parse_args()
    if args.command == 'scenario':
        run_scenario(args.prepare_dir, args.canonical_root, args.output_dir, args.scenario, args.replicate)
    else:
        aggregate(args.chunks_root, args.output_dir, args.prepare_dir)


if __name__ == '__main__':
    main()
