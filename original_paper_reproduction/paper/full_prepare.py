#!/usr/bin/env python3
"""Fresh ETRI primitive extraction; never reads historical primitive caches.

Calculation calls preserve run_task4c2_real.py:125-132. Independent paths and
provenance replace the original local orchestration, not the scientific code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import pandas as pd

PAPER = Path(__file__).resolve().parent
VENDOR = PAPER / 'full_vendor'


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def emit(**value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, default=str), flush=True)


def create_fresh_output(output: Path, inputs: list[Path]) -> Path:
    output = output.resolve()
    for item in inputs:
        item = item.resolve()
        if output == item or output.is_relative_to(item) or item.is_relative_to(output):
            raise ValueError(f'Output overlaps input: {output} / {item}')
    if output.exists():
        raise FileExistsError(f'Fresh run refuses existing output: {output}')
    output.mkdir(parents=True)
    return output


def verify_vendor() -> list[dict]:
    records = json.loads((PAPER / 'full_vendor_manifest.json').read_text(encoding='utf-8-sig'))
    for record in records:
        file = VENDOR / record['path']
        if not file.is_file() or sha256(file) != record['sha256']:
            raise ValueError(f'Vendored scientific source changed: {file}')
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    return [{'path': r['path'], 'sha256': r['sha256']} for r in records]


def compare_frames(actual: pd.DataFrame, reference: pd.DataFrame) -> dict:
    result = {
        'actual_shape': list(actual.shape), 'reference_shape': list(reference.shape),
        'columns_equal': list(actual.columns) == list(reference.columns),
        'dtypes_equal': list(map(str, actual.dtypes)) == list(map(str, reference.dtypes)),
    }
    try:
        pd.testing.assert_frame_equal(actual, reference, check_exact=True, check_dtype=True, check_like=False)
        result['exact_values_dtypes_order'] = True
    except AssertionError as error:
        result['exact_values_dtypes_order'] = False
        result['difference'] = str(error)[:8000]
    return result


def save_frame(output: Path, name: str, frame: pd.DataFrame) -> dict:
    path = output / f'{name}.parquet'
    frame.to_parquet(path, index=False)
    return {'path': path.name, 'sha256': sha256(path), 'rows': len(frame),
            'columns': list(map(str, frame.columns)),
            'dtypes': list(map(str, frame.dtypes)), 'bytes': path.stat().st_size}


def run_prepare(canonical_root: Path, output_dir: Path) -> dict:
    vendor_sources = verify_vendor()
    from semantic_indicators.primitives import build_sensor_calendar_scaffold, extract_daily_primitives
    from semantic_indicators.personalization import build_loso_representations

    canonical_root = canonical_root.resolve()
    files = sorted(canonical_root.glob('*_canonical.parquet'))
    if len(files) != 12:
        raise ValueError(f'Expected all 12 canonical inputs, found {len(files)}')
    inputs = [{'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in files]
    output = create_fresh_output(output_dir, [canonical_root, VENDOR])
    started = time.perf_counter()
    manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'prepare',
                'status': 'running', 'fresh_execution': True, 'new_model_fits': 0,
                'python': platform.python_version(), 'platform': platform.platform(),
                'inputs': inputs,
                'canonical_files': {Path(r['path']).name: {'sha256': r['sha256'], 'bytes': r['bytes']} for r in inputs},
                'scientific_sources': vendor_sources,
                'parameters': {'interval_boundary_policy': 'measurement_support',
                               'k_values': [14], 'lambda_days': 7.0, 'usage_policy': 'screen_proxy'},
                'outputs': {}}
    write_json(output / 'manifest.json', manifest)
    try:
        emit(stage='prepare', step='calendar', status='running')
        scaffold = build_sensor_calendar_scaffold(canonical_root)
        emit(stage='prepare', step='primitives', calendar_rows=len(scaffold))
        primitives = extract_daily_primitives(scaffold, canonical_root, interval_boundary_policy='measurement_support')
        if len(primitives) != 853 or primitives['subject_id'].nunique() != 10:
            raise ValueError('Fresh primitive population differs from study: expected 853 days / 10 participants')
        manifest['outputs']['primitives'] = save_frame(output, 'primitives', primitives)
        write_json(output / 'manifest.json', manifest)
        emit(stage='prepare', step='loso_representations', primitive_shape=list(primitives.shape))
        task3 = build_loso_representations(primitives, k_values=(14,), lambda_days=7.0, usage_policy='screen_proxy')
        for name, frame in [('baselines', task3.baselines), ('representations', task3.representations)]:
            manifest['outputs'][name] = save_frame(output, name, frame)
        for item in inputs:
            if sha256(Path(item['path'])) != item['sha256']:
                raise ValueError(f'Input changed during execution: {item["path"]}')
        verify_vendor()
        manifest.update(status='complete', seconds=time.perf_counter() - started)
        write_json(output / 'manifest.json', manifest)
        emit(stage='prepare', status='complete', seconds=manifest['seconds'],
             tables={k: [v['rows'], len(v['columns'])] for k, v in manifest['outputs'].items()})
        return manifest
    except BaseException as error:
        manifest.update(status='failed', seconds=time.perf_counter() - started,
                        error_type=type(error).__name__, error=str(error))
        write_json(output / 'manifest.json', manifest)
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def compare_prepare(actual_dir: Path, reference_dir: Path, report_path: Path) -> dict:
    records = {}
    for name in ('primitives', 'baselines', 'representations'):
        actual = actual_dir / f'{name}.parquet'
        reference = reference_dir / f'task-4c2-real-{name}.parquet'
        records[name] = compare_frames(pd.read_parquet(actual), pd.read_parquet(reference))
        records[name].update(actual_sha256=sha256(actual), reference_sha256=sha256(reference),
                             byte_identical=sha256(actual) == sha256(reference))
    report = {'stage': 'prepare_comparison', 'tables': records,
              'all_exact_values_dtypes_order': all(r['exact_values_dtypes_order'] for r in records.values())}
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--canonical-root', type=Path, default=Path('/data/prepared/project/data/canonical'))
    run.add_argument('--output-dir', type=Path, default=Path('/data/output/paper/prepare'))
    compare = sub.add_parser('compare')
    compare.add_argument('--actual-dir', type=Path, required=True)
    compare.add_argument('--reference-dir', type=Path, required=True)
    compare.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        run_prepare(args.canonical_root, args.output_dir)
    else:
        report = compare_prepare(args.actual_dir, args.reference_dir, args.report)
        emit(**report)
        if not report['all_exact_values_dtypes_order']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
