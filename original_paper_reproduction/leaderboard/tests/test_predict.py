"""Inference boundaries, native reloads and submission ordering; tiny fixtures only."""
import hashlib
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULE = ROOT / 'predict.py'
if MODULE.exists():
    spec = importlib.util.spec_from_file_location('leaderboard_predict', MODULE)
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
else:
    p = None

TARGETS = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
KEYS = ['subject_id', 'sleep_date', 'lifelog_date']
REQUIRED = ['processed/test_integrated_sensor_v2_cleaned.csv',
            'processed/feature_columns_sensor_v2_cleaned.json',
            'raw/data_vvs/ch2026_submission_sample.csv']
SOURCE_SHA = 'ff04ff61ae3f5ddbae357169d9545c854b8eb1816c428bc2f27146636c0c873e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(p, 'The inference implementation has not been created yet')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def pinned_files(self):
        data = self.root / 'data'
        entries = []
        for rel in REQUIRED:
            path = data / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture', encoding='utf-8')
            entries.append({'path': rel, 'sha256': digest(path)})
        # The training file is deliberately absent. Inference must not open it.
        entries.insert(0, {'path': 'processed/train_integrated_sensor_v2_cleaned.csv', 'sha256': '0' * 64})
        manifest = self.root / 'input_manifest.json'
        write_json(manifest, {'files': entries})
        return data, manifest, entries[1:]

    def bundle(self, pins):
        folder = self.root / 'bundle'
        (folder / 'models').mkdir(parents=True)
        records = []
        for model, suffix in [('lgbm', '.txt'), ('cat', '.cbm')]:
            for target in TARGETS:
                path = folder / 'models' / (model + '_' + target + suffix)
                path.write_bytes(b'native fixture')
                records.append({'model': model, 'target': target, 'n_estimators': 30,
                                'model_path': path.relative_to(folder).as_posix(), 'sha256': digest(path)})
        write_json(folder / 'model_manifest.json', records)
        write_json(folder / 'run_manifest.json', {
            'status': 'completed', 'original_source_sha256': SOURCE_SHA,
            'selected_weights': {'lgbm': .5, 'cat': .5, 'xgb': 0.}, 'inputs': pins})
        return folder, records

    def test_only_three_pinned_inputs_are_read_without_training_file(self):
        data, manifest, _ = self.pinned_files()
        pins = p.verify_inference_inputs(data, manifest)
        self.assertEqual({item['path'] for item in pins}, set(REQUIRED))

    def test_changed_test_input_is_rejected(self):
        data, manifest, _ = self.pinned_files()
        (data / REQUIRED[0]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            p.verify_inference_inputs(data, manifest)

    def test_input_manifest_escape_is_rejected(self):
        data, manifest, _ = self.pinned_files()
        obj = json.loads(manifest.read_text())
        obj['files'][1]['path'] = '../outside.csv'
        write_json(manifest, obj)
        with self.assertRaises(ValueError):
            p.verify_inference_inputs(data, manifest)

    def test_native_model_hash_and_incomplete_run_are_rejected(self):
        _, _, pins = self.pinned_files()
        folder, records = self.bundle(pins)
        bundle = p.verify_model_bundle(folder, pins)
        self.assertEqual(len(bundle['models']), 14)
        path = folder / records[0]['model_path']
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            p.verify_model_bundle(folder, pins)
        run = json.loads((folder / 'run_manifest.json').read_text())
        run['status'] = 'running'
        write_json(folder / 'run_manifest.json', run)
        with self.assertRaisesRegex(ValueError, 'completed'):
            p.verify_model_bundle(folder, pins)

    def test_model_path_escape_is_rejected(self):
        _, _, pins = self.pinned_files()
        folder, records = self.bundle(pins)
        records[0]['model_path'] = '../outside.txt'
        write_json(folder / 'model_manifest.json', records)
        with self.assertRaisesRegex(ValueError, 'path'):
            p.verify_model_bundle(folder, pins)

    def test_changed_weights_and_wrong_training_input_pins_are_rejected(self):
        _, _, pins = self.pinned_files()
        folder, _ = self.bundle(pins)
        run_path = folder / 'run_manifest.json'
        run = json.loads(run_path.read_text())
        run['selected_weights'] = {'lgbm': .4, 'cat': .6, 'xgb': 0.}
        write_json(run_path, run)
        with self.assertRaisesRegex(ValueError, 'weight'):
            p.verify_model_bundle(folder, pins)
        run['selected_weights'] = {'lgbm': .5, 'cat': .5, 'xgb': 0.}
        run['inputs'][0]['sha256'] = 'f' * 64
        write_json(run_path, run)
        with self.assertRaisesRegex(ValueError, 'input'):
            p.verify_model_bundle(folder, pins)

    def test_missing_and_duplicate_active_models_are_rejected(self):
        _, _, pins = self.pinned_files()
        folder, records = self.bundle(pins)
        write_json(folder / 'model_manifest.json', records[:-1])
        with self.assertRaisesRegex(ValueError, '14'):
            p.verify_model_bundle(folder, pins)
        write_json(folder / 'model_manifest.json', records + records[:1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            p.verify_model_bundle(folder, pins)

    def frames(self):
        features = ['f' + str(i) for i in range(3155)]
        test = pd.DataFrame(np.zeros((250, 3155)), columns=features)
        test[KEYS] = pd.DataFrame({'subject_id': ['s'] * 250,
                                  'sleep_date': pd.date_range('2025-01-02', periods=250),
                                  'lifelog_date': pd.date_range('2025-01-01', periods=250)})
        sample = test[KEYS].copy()
        for target in TARGETS:
            sample[target] = 0
        return test, sample, features

    def test_missing_duplicate_and_unmatched_keys_are_rejected(self):
        for defect in ['missing', 'duplicate', 'unmatched']:
            with self.subTest(defect=defect):
                test, sample, features = self.frames()
                if defect == 'missing':
                    test.loc[0, 'subject_id'] = None
                elif defect == 'duplicate':
                    test.loc[1, KEYS] = test.loc[0, KEYS].to_numpy()
                else:
                    test.loc[0, 'subject_id'] = 'other'
                with self.assertRaisesRegex(ValueError, 'key'):
                    p.validate_frames(test, sample, features)

    def test_native_nan_is_preserved_and_infinity_rejected(self):
        test, sample, features = self.frames()
        test.loc[0, 'f0'] = np.nan
        p.validate_frames(test, sample, features)
        self.assertTrue(pd.isna(test.loc[0, 'f0']))
        test.loc[0, 'f0'] = np.inf
        with self.assertRaisesRegex(ValueError, 'infinite'):
            p.validate_frames(test, sample, features)

    def test_submission_restores_sample_order_and_original_clipping(self):
        sample = pd.DataFrame({'subject_id': ['b', 'a'], 'sleep_date': ['2025-01-02'] * 2,
                               'lifelog_date': ['2025-01-01'] * 2, **{t: [0, 0] for t in TARGETS}})
        prediction = sample[KEYS].iloc[::-1].reset_index(drop=True)
        for target in TARGETS:
            prediction[target] = [0., 1.]
        path = self.root / 'submission.csv'
        p.write_submission(sample, prediction, path)
        actual = pd.read_csv(path)
        self.assertEqual(actual['subject_id'].tolist(), ['b', 'a'])
        self.assertEqual(actual['Q1'].tolist(), [.999999, .000001])
        self.assertEqual(actual.columns.tolist(), KEYS + TARGETS)
        self.assertTrue(path.read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_output_cannot_reuse_or_overlap_data_or_models(self):
        data = self.root / 'data'
        model = self.root / 'models'
        data.mkdir()
        model.mkdir()
        for candidate in [data / 'out', model / 'out', self.root]:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                p.reserve_output(candidate, [data, model])
        out = p.reserve_output(self.root / 'out', [data, model])
        sentinel = out / 'keep'
        sentinel.write_text('keep')
        with self.assertRaises(FileExistsError):
            p.reserve_output(out, [data, model])
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_real_native_reload_matches_two_tiny_models(self):
        import catboost as cb
        import lightgbm as lgb
        x = pd.DataFrame({'f0': [0., 1., 0., 1.], 'f1': [1., 0., 0., 1.]})
        y = [0, 1, 0, 1]
        light = lgb.LGBMClassifier(n_estimators=2, min_child_samples=1, n_jobs=1, verbosity=-1)
        light.fit(x, y)
        light_path = self.root / 'light.txt'
        light.booster_.save_model(str(light_path))
        cat = cb.CatBoostClassifier(iterations=2, depth=1, thread_count=1, verbose=False, allow_writing_files=False)
        cat.fit(x, y)
        cat_path = self.root / 'cat.cbm'
        cat.save_model(str(cat_path))
        unicode_folder = self.root / '한글 모델 경로'
        unicode_folder.mkdir()
        unicode_light, unicode_cat = unicode_folder / 'light.txt', unicode_folder / 'cat.cbm'
        unicode_light.write_bytes(light_path.read_bytes())
        unicode_cat.write_bytes(cat_path.read_bytes())
        for name, path, expected in [
                ('lgbm', unicode_light, light.predict_proba(x)[:, 1]),
                ('cat', unicode_cat, cat.predict_proba(x)[:, 1])]:
            with self.subTest(model=name):
                np.testing.assert_allclose(p.predict_native(name, path, x), expected, rtol=0, atol=1e-12)

    def test_complete_inference_without_train_or_reference_file(self):
        import catboost as cb
        import lightgbm as lgb
        test, sample, features = self.frames()
        test['f0'] = np.arange(250) % 2
        sample = sample.iloc[::-1].reset_index(drop=True)
        data = self.root / 'inputs'
        for rel in REQUIRED:
            (data / rel).parent.mkdir(parents=True, exist_ok=True)
        test.to_csv(data / REQUIRED[0], index=False)
        write_json(data / REQUIRED[1], features)
        sample.to_csv(data / REQUIRED[2], index=False)
        pins = [{'path': rel, 'sha256': digest(data / rel)} for rel in REQUIRED]
        manifest = self.root / 'input_manifest.json'
        write_json(manifest, {'files': pins})
        bundle, records = self.bundle(pins)
        x, y = test[features].iloc[:8], [0, 1] * 4
        light = lgb.LGBMClassifier(n_estimators=30, min_child_samples=1, n_jobs=1, verbosity=-1)
        light.fit(x, y)
        cat = cb.CatBoostClassifier(iterations=30, depth=1, thread_count=1, verbose=False, allow_writing_files=False)
        cat.fit(x, y)
        for record in records:
            path = bundle / record['model_path']
            if record['model'] == 'lgbm':
                light.booster_.save_model(str(path))
            else:
                cat.save_model(str(path))
            record['sha256'] = digest(path)
        write_json(bundle / 'model_manifest.json', records)
        expected = ((light.predict_proba(test[features])[:, 1]
                     + cat.predict_proba(test[features])[:, 1]) / 2)[::-1]
        output = self.root / 'inference'
        with redirect_stdout(io.StringIO()):
            report = p.infer(SimpleNamespace(data_root=data, model_dir=bundle,
                                            output_dir=output, input_manifest=manifest))
        actual = pd.read_csv(output / 'submissions' / 'submission_best_tree_ensemble_sensor_v2.csv')
        self.assertEqual(len(actual), 250)
        self.assertEqual(actual['lifelog_date'].tolist(), sample['lifelog_date'].dt.strftime('%Y-%m-%d').tolist())
        np.testing.assert_allclose(actual[TARGETS].to_numpy(), np.tile(expected[:, None], (1, 7)), rtol=0, atol=1e-12)
        stored = json.loads((output / 'inference_report.json').read_text())
        self.assertEqual(stored['status'], 'completed')
        self.assertEqual(stored['native_models_loaded'], 14)
        self.assertEqual(stored['validation']['input_hashes_verified'], 3)
        self.assertFalse(report['training_performed'])
        self.assertFalse(report['reference_predictions_read'])
        self.assertFalse(report['historical_submission_comparison_performed'])

    def test_non_ascii_temp_configuration_has_an_explicit_error(self):
        # Model loading has a real filesystem dependency. Override only the
        # configured temp root to exercise the guard without changing host env.
        with patch.object(p.tempfile, 'gettempdir', return_value=str(self.root / '한글 임시')):
            with self.assertRaisesRegex(ValueError, 'ASCII.*TMP/TEMP'):
                p.predict_native('cat', self.root / 'unused.cbm', pd.DataFrame({'f0': [0.]}))


if __name__ == '__main__':
    unittest.main()
