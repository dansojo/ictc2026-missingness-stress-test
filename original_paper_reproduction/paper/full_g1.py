#!/usr/bin/env python3
"""Fresh G1 calculation from fresh prepare output, with separate reviewed comparison.

Scientific API: original validation.py:1198-1731. Frozen reviewed parameters:
task-4a-reviewed-g1-pair-summary.json; root seed 20260807, not API default42.
The historical artifact is never used as numeric input or rewritten as fresh approval.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd

from full_prepare import (PAPER, VENDOR, create_fresh_output, emit, save_frame,
                          sha256, verify_vendor, write_json)

G1_PARAMETERS = {'evaluation_start_day': 22, 'bootstrap_draws': 10000,
                 'shift_draws': 1000, 'seed': 20260807,
                 'usage_sensitivity_columns': None}
REVIEWED_G1_DIGEST = 'a05199eada5a13f54e96393bd3a1a1920390bbd8aba83c7f15a4b4d790062fcb'
PAIRS = ('digital_load', 'digital_disengagement', 'physical_load', 'light_exposure')
# Recorded precision, not fitted tolerances: reviewed adjusted_r has10 decimals;
# raw_lopo_min and holm_shift_p have6 decimals. raw_r retains original full precision.
REFERENCE_ATOL = {'raw_r': 1e-12, 'adjusted_r': 5.1e-11,
                  'raw_lopo_min': 5.0001e-7, 'holm_shift_p': 5.0001e-7}


def save_g1_frame(output: Path, name: str, frame: pd.DataFrame) -> dict:
    """Preserve mixed boolean/numeric decisions instead of coercing their types."""
    mixed = any(pd.api.types.is_object_dtype(frame[c].dtype)
                and pd.api.types.infer_dtype(frame[c], skipna=True).startswith('mixed')
                for c in frame.columns)
    if not mixed:
        return {**save_frame(output, name, frame), 'format': 'parquet'}
    path = output / f'{name}.jsonl'

    def value(item):
        if item is None or item is pd.NA or (isinstance(item, float) and np.isnan(item)):
            return None
        if isinstance(item, np.generic):
            return value(item.item())
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        return item

    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in frame.to_dict(orient='records'):
            stream.write(json.dumps({key: value(item) for key, item in row.items()},
                                    ensure_ascii=False, allow_nan=False) + '\n')
    return {'path': path.name, 'format': 'jsonl', 'sha256': sha256(path),
            'bytes': path.stat().st_size, 'rows': len(frame),
            'columns': list(map(str, frame.columns)), 'dtypes': list(map(str, frame.dtypes))}


def compare_g1_pairs(fresh: pd.DataFrame, reviewed_g1_path: Path) -> dict:
    """Report every reviewed value, including exact versus rounded-value agreement."""
    verify_vendor()
    from semantic_indicators.reliability import validate_reviewed_g1_artifact
    reviewed_g1_path = reviewed_g1_path.resolve()
    artifact = json.loads(reviewed_g1_path.read_text(encoding='utf-8-sig'))
    validate_reviewed_g1_artifact(artifact, expected_sha256=REVIEWED_G1_DIGEST)
    reference = pd.DataFrame(artifact['pair_summary'])
    topology = ('pair' in fresh and not fresh['pair'].duplicated().any()
                and set(fresh['pair']) == set(PAIRS) and len(fresh) == 4)
    report = {'schema_name': 'independent_g1_reviewed_comparison_v1',
              'comparison_reference': {'path': str(reviewed_g1_path),
                                       'physical_sha256': sha256(reviewed_g1_path),
                                       'logical_artifact_sha256': REVIEWED_G1_DIGEST,
                                       'role': 'historical_reviewed_comparison_only'},
              'pair_topology_equal': bool(topology), 'actual_rows': len(fresh),
              'reference_rows': len(reference), 'fields': {},
              'precision_policy': {'relative_tolerance': 0, 'absolute_tolerances': REFERENCE_ATOL,
                                   'other_fields': 'exact values and scalar kinds'},
              'all_exact_values': False, 'all_within_reference_precision': False}
    if not topology:
        return report
    actual = fresh.set_index('pair').loc[list(PAIRS)]
    expected = reference.set_index('pair').loc[list(PAIRS)]
    for column in expected.columns:
        if column not in actual:
            report['fields'][column] = {'present': False, 'exact_equal': False,
                                        'within_reference_precision': False}
            continue
        left, right = actual[column], expected[column]
        if column in REFERENCE_ATOL:
            observed = pd.to_numeric(left, errors='coerce').to_numpy(dtype=float)
            reviewed = pd.to_numeric(right, errors='coerce').to_numpy(dtype=float)
            differences = np.abs(observed - reviewed)
            exact = bool(np.array_equal(observed, reviewed))
            passed = bool(np.isfinite(observed).all()
                          and np.all(differences <= REFERENCE_ATOL[column]))
            report['fields'][column] = {
                'present': True, 'exact_equal': exact, 'within_reference_precision': passed,
                'absolute_tolerance': REFERENCE_ATOL[column],
                'max_absolute_difference': float(differences.max()),
                'values': [{'pair': pair, 'actual': float(a), 'reference': float(b),
                            'absolute_difference': float(abs(a-b))}
                           for pair, a, b in zip(PAIRS, observed, reviewed)]}
        else:
            def same(a, b):
                if isinstance(b, (bool, np.bool_)):
                    return isinstance(a, (bool, np.bool_)) and bool(a) == bool(b)
                if isinstance(b, (int, np.integer)):
                    return (isinstance(a, (int, np.integer))
                            and not isinstance(a, (bool, np.bool_)) and int(a) == int(b))
                return type(a) is type(b) and a == b
            matches = [bool(same(a, b)) for a, b in zip(left, right)]
            report['fields'][column] = {
                'present': True, 'exact_equal': all(matches),
                'within_reference_precision': all(matches),
                'different_pairs': [pair for pair, match in zip(PAIRS, matches) if not match]}
    report['all_exact_values'] = all(item['exact_equal'] for item in report['fields'].values())
    report['all_within_reference_precision'] = all(
        item['within_reference_precision'] for item in report['fields'].values())
    return report


def run_g1(prepare_dir: Path, output_dir: Path, reviewed_g1_path: Path | None = None) -> dict:
    """Run the complete reviewed G1 computation; reference is optional comparison only."""
    sources = verify_vendor()
    from semantic_indicators.validation import evaluate_convergent_pairs
    prepared = prepare_dir.resolve()
    source_manifest = prepared / 'manifest.json'
    upstream = json.loads(source_manifest.read_text(encoding='utf-8-sig'))
    if (upstream.get('schema_name') != 'independent_etri_reproduction'
            or upstream.get('stage') != 'prepare' or upstream.get('status') != 'complete'
            or upstream.get('fresh_execution') is not True
            or upstream.get('parameters', {}).get('interval_boundary_policy') != 'measurement_support'):
        raise ValueError('G1 requires a complete fresh measurement_support prepare manifest')
    record = upstream['outputs']['primitives']
    primitives_path = (prepared / record['path']).resolve()
    if not primitives_path.is_relative_to(prepared) or not primitives_path.is_file():
        raise ValueError('Primitive output escapes the fresh prepare directory')
    primitive_digest = sha256(primitives_path)
    if primitive_digest != record['sha256']:
        raise ValueError('Fresh primitive digest changed from its prepare manifest')
    upstream_digest = sha256(source_manifest)
    inputs = [prepared, VENDOR]
    if reviewed_g1_path is not None:
        inputs.append(reviewed_g1_path.resolve())
    output = create_fresh_output(output_dir, inputs)
    started = time.perf_counter()
    manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'g1',
                'status': 'running', 'fresh_execution': True, 'new_model_fits': 0,
                'python': platform.python_version(), 'platform': platform.platform(),
                'parameters': dict(G1_PARAMETERS), 'scientific_sources': sources,
                'runner_sha256': sha256(Path(__file__)),
                'inputs': {'prepare_manifest': {'path': str(source_manifest), 'sha256': upstream_digest},
                           'primitives': {'path': str(primitives_path), 'sha256': primitive_digest}},
                'historical_results_used_as_numeric_input': False, 'outputs': {}}
    write_json(output / 'manifest.json', manifest)
    try:
        primitives = pd.read_parquet(primitives_path)
        if len(primitives) != 853 or primitives['subject_id'].nunique() != 10:
            raise ValueError('G1 study population requires853 primitive rows and10 participants')
        emit(stage='g1', status='running', step='convergent_pairs', parameters=G1_PARAMETERS)
        result = evaluate_convergent_pairs(primitives, **G1_PARAMETERS)
        # Exact full-data dimensions documented in task-4a-report.md:167-179.
        expected_rows = {'row_table': 2572, 'participant_table': 40,
                         'adjusted_participant_table': 40, 'bootstrap_draws': 40000,
                         'shift_draws': 8000, 'shift_assignments': 80000,
                         'lopo_rows': 2015, 'quality_sensitivity': 20,
                         'usage_sensitivity': 8, 'pair_summary': 4, 'domain_summary': 3}
        for name, count in expected_rows.items():
            if len(getattr(result, name)) != count:
                raise ValueError(f'Full G1 {name} topology differs: expected{count}')
        if (tuple(result.pair_summary['pair']) != PAIRS
                or not result.pair_summary['root_seed'].eq(20260807).all()
                or not result.pair_summary['bootstrap_draws'].eq(10000).all()
                or not result.pair_summary['shift_draws'].eq(1000).all()):
            raise ValueError('G1 pair identity or frozen parameter topology changed')
        for field in fields(result):
            manifest['outputs'][field.name] = save_g1_frame(output, field.name, getattr(result, field.name))
        if sha256(primitives_path) != primitive_digest or sha256(source_manifest) != upstream_digest:
            raise ValueError('Fresh G1 input changed during execution')
        verify_vendor()
        if reviewed_g1_path is not None:
            comparison = compare_g1_pairs(result.pair_summary, reviewed_g1_path)
            write_json(output / 'reviewed_comparison.json', comparison)
            manifest['comparison'] = {
                'path': 'reviewed_comparison.json', 'sha256': sha256(output / 'reviewed_comparison.json'),
                'all_exact_values': comparison['all_exact_values'],
                'all_within_reference_precision': comparison['all_within_reference_precision']}
        manifest.update(status='complete', seconds=time.perf_counter() - started,
                        table_count=len(manifest['outputs']))
        write_json(output / 'manifest.json', manifest)
        emit(stage='g1', status='complete', seconds=manifest['seconds'],
             table_count=manifest['table_count'], comparison=manifest.get('comparison'))
        return manifest
    except BaseException as error:
        manifest.update(status='failed', seconds=time.perf_counter() - started,
                        error_type=type(error).__name__, error=str(error))
        write_json(output / 'manifest.json', manifest)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--prepare-dir', type=Path, default=Path('/data/output/paper/prepare'))
    run.add_argument('--output-dir', type=Path, default=Path('/data/output/paper/g1'))
    run.add_argument('--reviewed-g1', type=Path, help='Optional comparison only; never a numeric input')
    compare = sub.add_parser('compare')
    compare.add_argument('--actual-dir', type=Path, required=True)
    compare.add_argument('--reviewed-g1', type=Path,
                         default=PAPER / 'protocol/task-4a-reviewed-g1-pair-summary.json')
    compare.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        result = run_g1(args.prepare_dir, args.output_dir, args.reviewed_g1)
        if result.get('comparison', {}).get('all_within_reference_precision') is False:
            raise SystemExit(2)
    else:
        report = compare_g1_pairs(pd.read_parquet(args.actual_dir / 'pair_summary.parquet'), args.reviewed_g1)
        if args.report.resolve() in {(args.actual_dir / 'pair_summary.parquet').resolve(), args.reviewed_g1.resolve()}:
            raise ValueError('Comparison report cannot overwrite its input')
        write_json(args.report, report)
        emit(**report)
        if not report['all_within_reference_precision']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
