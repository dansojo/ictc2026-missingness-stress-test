"""Inference only from saved native models; no fitting, labels or reference CSV.

Example:
  python -B predict.py --data-root /data/prepared/project/data \
    --model-dir /data/models/leaderboard --output-dir /data/output/inference

The model directory is a completed reproduce.py training run, containing
run_manifest.json, model_manifest.json and models/. This loads 7 LightGBM and
7 CatBoost models. XGBoost is not loaded because its selected weight is zero.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

import reproduce as r

ROOT = Path(__file__).resolve().parent
TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']
INPUT_FILES = ['processed/test_integrated_sensor_v2_cleaned.csv',
               'processed/feature_columns_sensor_v2_cleaned.json',
               'raw/data_vvs/ch2026_submission_sample.csv']
TRAIN_FILE = 'processed/train_integrated_sensor_v2_cleaned.csv'
WEIGHTS = {'lgbm': .5, 'cat': .5, 'xgb': 0.}
EPS = 1e-6
reserve_output = r.reserve_output


def verify_inference_inputs(data_root, input_manifest):
    """Read hashes of exactly three inference files; never open train CSV."""
    manifest = json.loads(Path(input_manifest).read_text(encoding='utf-8-sig'))
    entries = manifest['files']
    names = [item['path'] for item in entries]
    if (len(names) != len(set(names)) or not set(INPUT_FILES).issubset(names)
            or not set(names).issubset(set(INPUT_FILES + [TRAIN_FILE]))):
        raise ValueError('Input manifest must uniquely name the three inference files and optional train entry')
    return r.verify_files(data_root, {'files': [item for item in entries if item['path'] in INPUT_FILES]})


def verify_model_bundle(model_dir, input_pins):
    """Validate completed run, source/data lineage and the 14 active model hashes."""
    model_dir = Path(model_dir).resolve()
    run_path, model_path = model_dir / 'run_manifest.json', model_dir / 'model_manifest.json'
    run = json.loads(run_path.read_text(encoding='utf-8-sig'))
    if run.get('status') != 'completed':
        raise ValueError('Model run must have status completed')
    if run.get('selected_weights') != WEIGHTS:
        raise ValueError('Selected weights must be exactly lgbm=0.5, cat=0.5, xgb=0.0')
    if run.get('original_source_sha256') != r.VENDOR_SHA or r.sha256(r.VENDOR) != r.VENDOR_SHA:
        raise ValueError('Original model source hash mismatch')
    training_inputs = run.get('inputs', [])
    names = [item['path'] for item in training_inputs]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate training input pins')
    previous = {item['path']: item['sha256'] for item in training_inputs}
    if any(previous.get(item['path']) != item['sha256'] for item in input_pins):
        raise ValueError('Model training input pins do not match current inference inputs')
    records = json.loads(model_path.read_text(encoding='utf-8-sig'))
    required = {(model, target) for model in ['lgbm', 'cat'] for target in TARGETS}
    seen, active = set(), []
    suffixes = {'lgbm': '.txt', 'cat': '.cbm', 'xgb': '.ubj'}
    for record in records:
        name, target = record['model'], record['target']
        pair = (name, target)
        if pair in seen:
            raise ValueError('Duplicate model/target record')
        seen.add(pair)
        if name not in suffixes or target not in TARGETS:
            raise ValueError('Unexpected model or target')
        expected = f'models/{name}_{target}{suffixes[name]}'
        rel = Path(record['model_path'])
        resolved = (model_dir / rel).resolve()
        if (rel.as_posix() != expected or rel.is_absolute() or '..' in rel.parts
                or not resolved.is_relative_to(model_dir)):
            raise ValueError('Model path must be its expected relative path inside the model directory')
        if pair in required:
            actual = r.sha256(resolved)
            if actual != record['sha256']:
                raise ValueError(f'Model hash mismatch: {name}/{target}')
            rounds = record['n_estimators']
            if not isinstance(rounds, int) or not 30 <= rounds <= 500:
                raise ValueError('Saved model rounds must follow the historical 30..500 rule')
            active.append({'model': name, 'target': target, 'n_estimators': rounds,
                           'model_path': rel.as_posix(), 'sha256': actual,
                           'bytes': resolved.stat().st_size})
    if { (item['model'], item['target']) for item in active } != required:
        raise ValueError('Expected all 14 active LightGBM/CatBoost models')
    return {'models': active, 'selected_weights': dict(WEIGHTS),
            'run_manifest_sha256': r.sha256(run_path),
            'model_manifest_sha256': r.sha256(model_path),
            'training_environment': run.get('environment'),
            'original_source_sha256': r.VENDOR_SHA}


def validate_frames(test, sample, features):
    if len(test) != 250 or len(sample) != 250 or len(features) != 3155:
        raise ValueError('Expected 250 test/sample rows and 3155 ordered features')
    if (not all(isinstance(name, str) for name in features)
            or len(set(features)) != len(features) or set(features) & set(KEYS + TARGETS)):
        raise ValueError('Duplicate, invalid or label/key-containing feature list')
    for name, frame in [('test', test), ('sample', sample)]:
        if not set(KEYS).issubset(frame.columns):
            raise ValueError(f'{name}: missing key columns')
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError(f'{name}: missing or duplicate keys')
    if set(map(tuple, test[KEYS].to_numpy())) != set(map(tuple, sample[KEYS].to_numpy())):
        raise ValueError('Test and sample key sets differ')
    if list(sample.columns) != KEYS + TARGETS:
        raise ValueError('Sample column schema differs from the original submission')
    if not set(features).issubset(test.columns):
        raise ValueError('Missing feature columns')
    r.validate_numeric_features(test[features].to_numpy(dtype=float))


def load_inputs(data_root):
    import pandas as pd
    data_root = Path(data_root)
    test = pd.read_csv(data_root / INPUT_FILES[0])
    features = json.loads((data_root / INPUT_FILES[1]).read_text(encoding='utf-8-sig'))
    sample = pd.read_csv(data_root / INPUT_FILES[2])
    for frame in [test, sample]:
        for col in ['lifelog_date', 'sleep_date']:
            frame[col] = pd.to_datetime(frame[col], errors='raise')
    validate_frames(test, sample, features)
    return test, sample, features


def predict_native(name, path, x):
    import numpy as np
    if name == 'lgbm':
        import lightgbm as lgb
        # Python handles Unicode paths; the C++ filename API may not on Windows.
        model = lgb.Booster(model_str=Path(path).read_text(encoding='utf-8'))
        if model.feature_name() != list(x.columns):
            raise ValueError('LightGBM feature names/order mismatch')
        values = model.predict(x)
    elif name == 'cat':
        import catboost as cb
        model = cb.CatBoostClassifier()
        temp_root = Path(tempfile.gettempdir()).resolve()
        if not str(temp_root).isascii():
            raise ValueError('CatBoost native loading requires an ASCII temporary directory; set TMP/TEMP to an ASCII path')
        with tempfile.TemporaryDirectory(prefix='leaderboard-native-', dir=temp_root) as directory:
            native_path = Path(directory) / 'model.cbm'
            if not str(native_path).isascii():
                raise ValueError('CatBoost native temporary model path must be ASCII')
            native_path.write_bytes(Path(path).read_bytes())
            model.load_model(str(native_path))
        if model.feature_names_ != list(x.columns):
            raise ValueError('CatBoost feature names/order mismatch')
        values = model.predict_proba(x)[:, 1]
    else:
        raise ValueError('Only active LightGBM and CatBoost models may be loaded')
    values = np.asarray(values, dtype=float)
    if values.shape != (len(x),) or not np.isfinite(values).all():
        raise ValueError('Invalid native prediction shape or non-finite probabilities')
    if (values < 0).any() or (values > 1).any():
        raise ValueError('Native probabilities outside [0,1]')
    return values


def write_submission(sample, pred, path):
    """Same key merge, clipping and UTF-8 BOM CSV rules as the original source."""
    sub = sample[KEYS + TARGETS].copy()
    merged = sub[KEYS].merge(pred[KEYS + TARGETS], on=KEYS, how='left', validate='one_to_one')
    if merged[TARGETS].isna().any().any():
        raise ValueError('Missing predictions for submission keys')
    for target in TARGETS:
        sub[target] = merged[target].clip(EPS, 1 - EPS)
    sub.to_csv(path, index=False, encoding='utf-8-sig')


def infer(args):
    data, model_dir = args.data_root.resolve(), args.model_dir.resolve()
    pins = verify_inference_inputs(data, args.input_manifest)
    bundle = verify_model_bundle(model_dir, pins)
    test, sample, features = load_inputs(data)
    output = reserve_output(args.output_dir, [data, model_dir, ROOT])
    start = time.perf_counter()
    report_path = output / 'inference_report.json'
    report = {'status': 'running', 'execution': 'inference_only_native_model_reload',
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'training_performed': False, 'training_labels_read': False,
              'reference_predictions_read': False, 'private_score_recomputed': False,
              'historical_submission_comparison_performed': False,
              'inputs': pins, 'model_bundle': bundle,
              'source_pins': [{'path': 'predict.py', 'sha256': r.sha256(__file__)},
                              {'path': 'reproduce.py', 'sha256': r.sha256(r.__file__)},
                              {'path': 'vendor/modeling/tree_model_reliability_submit.py', 'sha256': r.VENDOR_SHA},
                              {'path': str(args.input_manifest.resolve()), 'sha256': r.sha256(args.input_manifest)}],
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'packages': {name: importlib.metadata.version(name)
                                           for name in ['numpy', 'pandas', 'lightgbm', 'catboost']}},
              'validation': {'input_hashes_verified': 3, 'active_model_hashes_verified': 14,
                             'test_rows': len(test), 'sample_rows': len(sample),
                             'ordered_features': len(features), 'keys_unique_nonmissing_and_sets_equal': True,
                             'completed_training_run_verified': True,
                             'historical_selected_weights_verified': True}}
    r.write_json(report_path, report)
    try:
        records = {(item['model'], item['target']): item for item in bundle['models']}
        x = test[features]
        prediction = test[KEYS].copy()
        for target in TARGETS:
            light = predict_native('lgbm', model_dir / records['lgbm', target]['model_path'], x)
            cat = predict_native('cat', model_dir / records['cat', target]['model_path'], x)
            # Preserve the original arithmetic order. Its XGBoost term is zero.
            prediction[target] = .5 * light + .5 * cat + 0.0
            print(f'Native inference completed: {target}', flush=True)
        submissions = output / 'submissions'
        submissions.mkdir()
        submission = submissions / 'submission_best_tree_ensemble_sensor_v2.csv'
        write_submission(sample, prediction, submission)
        report.update({'status': 'completed', 'wall_seconds': time.perf_counter() - start,
                       'native_models_loaded': 14, 'probabilities_written': 1750,
                       'submission': {'path': submission.relative_to(output).as_posix(),
                                      'sha256': r.sha256(submission), 'bytes': submission.stat().st_size},
                       'completion_note': 'Inference completed. Compare separately with the fresh-training or historical CSV; no equality claim is made here.'})
        r.write_json(report_path, report)
        return report
    except BaseException as exc:
        report.update({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                       'wall_seconds': time.perf_counter() - start})
        r.write_json(report_path, report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('/data/prepared/project/data'))
    parser.add_argument('--model-dir', type=Path, default=Path('/data/models/leaderboard'))
    parser.add_argument('--output-dir', type=Path, default=Path('/data/output/inference'))
    parser.add_argument('--input-manifest', type=Path, default=ROOT / 'input_manifest.json')
    report = infer(parser.parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
