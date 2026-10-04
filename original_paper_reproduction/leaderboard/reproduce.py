"""Portable execution of the unmodified June 11 leaderboard source.

Training never reads the reference submission. Compare is a separate command.
Default data inputs and generated outputs are under /data. No private labels
are available here, so this program does not calculate a Private score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor' / 'modeling' / 'tree_model_reliability_submit.py'
VENDOR_SHA = 'ff04ff61ae3f5ddbae357169d9545c854b8eb1816c428bc2f27146636c0c873e'
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']
INPUT_FILES = ['processed/train_integrated_sensor_v2_cleaned.csv',
               'processed/test_integrated_sensor_v2_cleaned.csv',
               'processed/feature_columns_sensor_v2_cleaned.json',
               'raw/data_vvs/ch2026_submission_sample.csv']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify_files(root, manifest):
    root = Path(root).resolve()
    result = []
    for item in manifest['files']:
        rel = Path(item['path'])
        path = (root / rel).resolve()
        if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):
            raise ValueError('Manifest paths must be relative and contained in the data root')
        actual = sha256(path)
        if actual != item['sha256']:
            raise ValueError(f'Input hash mismatch: {item["path"]}')
        result.append({'path': rel.as_posix(), 'sha256': actual, 'bytes': path.stat().st_size})
    return result


def reserve_output(output, protected):
    output = Path(output).resolve()
    for p in protected:
        p = Path(p).resolve()
        if output.is_relative_to(p) or p.is_relative_to(output):
            raise ValueError(f'Output and protected input overlap: {p}')
    output.mkdir(parents=True, exist_ok=False)
    return output


def environment():
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'catboost', 'xgboost', 'pyarrow']
    return {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'cpu_count': os.cpu_count(), 'packages': {p: importlib.metadata.version(p) for p in packages}}


def validate_numeric_features(values):
    import numpy as np
    # The original cleaning step deliberately retains NaN for native tree handling.
    # It replaces infinities with NaN. Do not introduce an imputation step here.
    if np.isinf(np.asarray(values, dtype=float)).any():
        raise ValueError('Cleaned features contain infinite values')


def load_inputs(data_root):
    import numpy as np
    import pandas as pd
    data_root = Path(data_root).resolve()
    train = pd.read_csv(data_root / INPUT_FILES[0])
    test = pd.read_csv(data_root / INPUT_FILES[1])
    features = json.loads((data_root / INPUT_FILES[2]).read_text(encoding='utf-8-sig'))
    sample = pd.read_csv(data_root / INPUT_FILES[3])
    if len(train) != 450 or len(test) != 250 or len(sample) != 250 or len(features) != 3155:
        raise ValueError('Expected 450 training rows, 250 test/sample rows and 3155 features')
    if len(set(features)) != len(features) or set(features) & set(TARGETS + KEYS):
        raise ValueError('Duplicate features or label/key leakage in feature list')
    for name, frame in [('train', train), ('test', test), ('sample', sample)]:
        for col in ['lifelog_date', 'sleep_date']:
            frame[col] = pd.to_datetime(frame[col], errors='raise')
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f'{name}: missing or duplicate row keys')
    if set(map(tuple, test[KEYS].to_numpy())) != set(map(tuple, sample[KEYS].to_numpy())):
        raise ValueError('Test and sample key sets differ')
    for frame in [train, test]:
        if not set(features).issubset(frame.columns):
            raise ValueError('Missing feature columns')
        validate_numeric_features(frame[features].to_numpy(dtype=float))
    if not train[TARGETS].isin([0, 1]).all().all():
        raise ValueError('Training labels must be binary')
    return train, test, sample, features


def compare_csv(actual, reference, *, expected_rows=250):
    import numpy as np
    import pandas as pd
    actual, reference = Path(actual), Path(reference)
    a, b = pd.read_csv(actual), pd.read_csv(reference)
    if list(a.columns) != KEYS + TARGETS or list(b.columns) != KEYS + TARGETS:
        raise ValueError('Submission columns do not match the required schema')
    if len(a) != expected_rows or len(b) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows in each submission')
    if a[KEYS].isna().any().any() or b[KEYS].isna().any().any():
        raise ValueError('Missing submission keys')
    if a.duplicated(KEYS).any() or b.duplicated(KEYS).any() or not a[KEYS].equals(b[KEYS]):
        raise ValueError('Submission key values/order differ')
    x, y = a[TARGETS].to_numpy(float), b[TARGETS].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite submission probabilities')
    if (x < 0).any() or (x > 1).any() or (y < 0).any() or (y > 1).any():
        raise ValueError('Probabilities outside [0,1]')
    diff = np.abs(x - y)
    return {'actual_sha256': sha256(actual), 'reference_sha256': sha256(reference),
            'rows': len(a), 'probabilities': int(diff.size), 'keys_and_order_equal': True,
            'byte_identical': actual.read_bytes() == reference.read_bytes(),
            'max_absolute_difference': float(diff.max()), 'mean_absolute_difference': float(diff.mean()),
            'exact_float_differences': int(np.count_nonzero(diff)),
            'all_within_1e_12': bool((diff <= 1e-12).all()),
            'within_1e_12_is_local_diagnostic_not_organizer_tolerance': True,
            'per_target_max_absolute_difference': dict(zip(TARGETS, map(float, diff.max(axis=0)))),
            'private_logloss': None,
            'private_score_note': 'Private labels unavailable. Displayed leaderboard score 0.6111 is not recalculated.'}


def save_reload_native(name, model, destination, x_test):
    """Use native formats while shielding native file APIs from Unicode paths."""
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb
    destination = Path(destination)
    with tempfile.TemporaryDirectory(prefix='ictc-native-') as tmp:
        native_path = Path(tmp) / ('model' + destination.suffix)
        if not str(native_path).isascii():
            raise ValueError('Native model API needs an ASCII temporary directory; set TMPDIR/TEMP accordingly')
        if name == 'lgbm':
            model.booster_.save_model(str(native_path))
            shutil.copyfile(native_path, destination)
            loaded = lgb.Booster(model_file=str(native_path))
            return loaded.predict(x_test)
        if name == 'cat':
            model.save_model(str(native_path))
            shutil.copyfile(native_path, destination)
            loaded = cb.CatBoostClassifier()
            loaded.load_model(str(native_path))
            return loaded.predict_proba(x_test)[:, 1]
        if name == 'xgb':
            model.save_model(str(native_path))
            shutil.copyfile(native_path, destination)
            loaded = xgb.XGBClassifier()
            loaded.load_model(str(native_path))
            return loaded.predict_proba(x_test)[:, 1]
        raise ValueError(f'Unknown model: {name}')


def record_failure(output, exc, elapsed=None):
    write_json(output / 'failure.json', {'error_type': type(exc).__name__, 'error': str(exc),
               'traceback': traceback.format_exc(), 'wall_seconds': elapsed})
    report_path = output / 'run_manifest.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    report.update({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                   'finished_utc': datetime.now(timezone.utc).isoformat()})
    if elapsed is not None:
        report['wall_seconds'] = elapsed
    write_json(report_path, report)


def train(args):
    import numpy as np
    data = args.data_root.resolve()
    manifest = json.loads(args.input_manifest.read_text(encoding='utf-8-sig'))
    if {i['path'] for i in manifest['files']} != set(INPUT_FILES):
        raise ValueError('Input manifest must name the four required training input files exactly')
    pins = verify_files(data, manifest)
    if sha256(VENDOR) != VENDOR_SHA:
        raise ValueError('Original model source hash mismatch')
    frames = load_inputs(data)
    output = reserve_output(args.output_dir, [data, ROOT])
    args._started_output = output
    start = time.perf_counter()
    write_json(output / 'run_manifest.json', {
        'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'execution': 'fresh_validation_and_full_training', 'data_root': str(data),
        'original_source_sha256': VENDOR_SHA, 'inputs': pins, 'environment': environment(),
        'driver_sha256': sha256(Path(__file__)),
        'reference_predictions_read_by_training': False, 'seed': 42,
        'historical_python': '3.14.5', 'current_python_matches_historical': sys.version_info[:3] == (3,14,5),
        'runtime_adaptations': ['explicit data/output paths', 'progress logging',
                                'save best iterations and native trained models', 'native model roundtrip checks'],
        'historical_early_stopping_iteration_conventions_preserved': True,
        'private_score_recomputed': False})
    spec = importlib.util.spec_from_file_location('historical_leaderboard_source', VENDOR)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    paths = {name: output / name for name in ['reports', 'tables', 'submissions', 'oof', 'models']}
    for p in paths.values():
        p.mkdir()
    original.load_inputs = lambda: frames
    original.ensure_dirs = lambda: paths
    events = []
    model_records = []
    original_valid = original.fit_predict_valid
    original_test = original.fit_predict_test
    original_make = original.make_model
    original_evaluate = original.evaluate_models
    created = []

    def make_model(name, n_estimators=None):
        model = original_make(name, n_estimators=n_estimators)
        created[:] = [model]
        return model

    def valid_fit(name, x, y, xv, yv):
        tick = time.perf_counter()
        prediction, best = original_valid(name, x, y, xv, yv)
        event = {'stage': 'validation', 'model': name, 'target': str(y.name),
                 'train_rows': len(x), 'valid_rows': len(xv), 'best_iteration': int(best),
                 'wall_seconds': time.perf_counter() - tick}
        events.append(event)
        write_json(output / 'fit_progress.json', events)
        print(json.dumps(event), flush=True)
        return prediction, best

    def test_fit(name, x, y, xt, n_estimators):
        tick = time.perf_counter()
        prediction = original_test(name, x, y, xt, n_estimators)
        model = created[0]
        target = str(y.name)
        suffix = {'lgbm': '.txt', 'cat': '.cbm', 'xgb': '.ubj'}[name]
        path = paths['models'] / f'{name}_{target}{suffix}'
        check = save_reload_native(name, model, path, xt)
        diff = float(np.max(np.abs(prediction - check)))
        if diff > 1e-12:
            raise ValueError(f'Native saved model roundtrip changed predictions: {name}/{target}: {diff}')
        record = {'stage': 'full_training', 'model': name, 'target': target,
                  'training_rows': len(x), 'n_estimators': int(n_estimators),
                  'model_path': path.relative_to(output).as_posix(), 'sha256': sha256(path),
                  'roundtrip_max_absolute_difference': diff,
                  'wall_seconds': time.perf_counter() - tick}
        model_records.append(record)
        write_json(output / 'model_manifest.json', model_records)
        events.append(record)
        write_json(output / 'fit_progress.json', events)
        print(json.dumps(record), flush=True)
        return prediction

    def evaluate(*call_args):
        scores, preds, iters = original_evaluate(*call_args)
        write_json(output / 'best_iterations.json', iters)
        # Freeze the precise row indices used by both historical validation methods.
        splits = {}
        for strategy in original.VALIDATION_STRATEGIES:
            tr, va = original.build_split(frames[0], frames[2], strategy)
            splits[strategy] = {'training_indices': tr.tolist(), 'validation_indices': va.tolist()}
        write_json(output / 'validation_splits.json', splits)
        return scores, preds, iters

    original.make_model = make_model
    original.fit_predict_valid = valid_fit
    original.fit_predict_test = test_fit
    original.evaluate_models = evaluate
    try:
        original.main()
        if len(events) != 63 or len(model_records) != 21:
            raise ValueError('Expected 42 validation fits and 21 full training fits')
        import pandas as pd
        weights_table = pd.read_csv(paths['tables'] / 'tree_model_reliability_ensemble_weight_search.csv')
        primary = weights_table[weights_table.strategy == 'subject_wise_last14days_holdout']
        selected = primary.sort_values('validation_logloss').iloc[0]
        selected_weights = {name: float(selected[name]) for name in ['lgbm', 'cat', 'xgb']}
        report = json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))
        report.update({'status': 'completed', 'wall_seconds': time.perf_counter() - start,
                       'validation_fits': 42, 'full_training_fits': 21,
                       'selected_weights': selected_weights,
                       'matches_historical_weights': selected_weights == {'lgbm': .5, 'cat': .5, 'xgb': 0.},
                       'validation_primary_logloss': float(selected.validation_logloss),
                       'submission_sha256': sha256(paths['submissions'] / 'submission_best_tree_ensemble_sensor_v2.csv')})
        write_json(output / 'run_manifest.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        record_failure(output, exc, time.perf_counter() - start)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train', help='Fresh historical validation, ensemble search, and full training')
    t.add_argument('--data-root', type=Path, default=Path('/data/prepared/project/data'))
    t.add_argument('--output-dir', type=Path, default=Path('/data/output/leaderboard'))
    t.add_argument('--input-manifest', type=Path, default=ROOT / 'input_manifest.json')
    c = sub.add_parser('compare', help='Compare a generated CSV after training; does not tune models')
    c.add_argument('--actual', type=Path, required=True)
    c.add_argument('--reference', type=Path, required=True)
    c.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'train':
        try:
            train(args)
        except BaseException as exc:
            # Includes failures during scientific-module import or output setup.
            # Never mark a pre-existing run failed when reservation was rejected.
            if hasattr(args, '_started_output') and not (args._started_output / 'failure.json').exists():
                record_failure(args._started_output, exc)
            raise
    else:
        if args.report.exists():
            raise FileExistsError(args.report)
        result = compare_csv(args.actual, args.reference)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, result)
        print(json.dumps(result, indent=2))
        if not result['all_within_1e_12']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
