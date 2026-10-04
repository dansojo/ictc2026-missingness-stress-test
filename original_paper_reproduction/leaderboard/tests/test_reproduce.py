"""Guard and comparison behavior; no model training in unit tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'reproduce.py'
spec = importlib.util.spec_from_file_location('leaderboard_reproduce', MODULE)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class SubmissionGuards(unittest.TestCase):
    def test_failed_run_cannot_remain_marked_running(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            r.write_json(folder/'run_manifest.json', {'status':'running','seed':42})
            r.record_failure(folder, ValueError('storage failure'), elapsed=2.)
            report=json.loads((folder/'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['seed'],42)
            self.assertTrue((folder/'failure.json').is_file())

    def test_all_native_models_roundtrip_through_unicode_output_path(self):
        import numpy as np
        import pandas as pd
        import lightgbm as lgb
        import catboost as cb
        import xgboost as xgb
        x = pd.DataFrame({'a':np.arange(30), 'b':np.arange(30)%3})
        y = pd.Series(np.arange(30)%2)
        models = {'lgbm': lgb.LGBMClassifier(n_estimators=3,min_child_samples=2,n_jobs=1,verbosity=-1),
                  'cat': cb.CatBoostClassifier(iterations=3,depth=2,verbose=False,allow_writing_files=False,thread_count=1),
                  'xgb': xgb.XGBClassifier(n_estimators=3,max_depth=2,n_jobs=1)}
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)/'모델 저장'
            folder.mkdir()
            for name, model in models.items():
                model.fit(x,y)
                path=folder/('모델'+{'lgbm':'.txt','cat':'.cbm','xgb':'.ubj'}[name])
                actual=r.save_reload_native(name, model, path, x)
                self.assertTrue(path.is_file())
                np.testing.assert_allclose(actual, model.predict_proba(x)[:,1],rtol=0,atol=1e-12)

    def test_default_submission_comparison_requires_250_rows(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'short.csv'
            pd.DataFrame({'subject_id':['a'],'sleep_date':['2026-01-02'],'lifelog_date':['2026-01-01'],
                          **{t:[.2] for t in r.TARGETS}}).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'250'):
                r.compare_csv(path,path)

    def test_native_missing_values_are_preserved_but_infinities_rejected(self):
        import numpy as np
        x = np.array([[1., np.nan], [0., 2.]])
        r.validate_numeric_features(x)
        self.assertTrue(np.isnan(x[0, 1]))
        with self.assertRaisesRegex(ValueError, 'infinite'):
            r.validate_numeric_features(np.array([[np.inf]]))

    def test_changed_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'sample.csv').write_text('a', encoding='utf-8')
            manifest = {'files': [{'path': 'sample.csv', 'sha256': r.sha256(root / 'sample.csv')}]}
            self.assertEqual(len(r.verify_files(root, manifest)), 1)
            (root / 'sample.csv').write_text('b', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash'):
                r.verify_files(root, manifest)

    def test_manifest_path_cannot_escape_input_root(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'relative'):
                r.verify_files(Path(td), {'files': [{'path': '../outside', 'sha256': '0'}]})

    def test_existing_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / 'data'
            data.mkdir()
            out = root / 'out'
            r.reserve_output(out, [data])
            sentinel = out / 'keep.txt'
            sentinel.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                r.reserve_output(out, [data])
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_output_cannot_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / 'input'
            raw.mkdir()
            with self.assertRaisesRegex(ValueError, 'overlap'):
                r.reserve_output(raw / 'output', [raw])

    def test_csv_comparison_rejects_key_reordering(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = pd.DataFrame({'subject_id': ['a','b'], 'sleep_date':['2026-01-02']*2,
                              'lifelog_date':['2026-01-01']*2, **{t:[.2,.8] for t in r.TARGETS}})
            a.to_csv(root/'a.csv', index=False)
            a.iloc[::-1].to_csv(root/'b.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'order'):
                r.compare_csv(root/'a.csv', root/'b.csv', expected_rows=2)

    def test_difference_is_measured_not_hidden_by_rounded_score(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = pd.DataFrame({'subject_id':['a'], 'sleep_date':['2026-01-02'],
                              'lifelog_date':['2026-01-01'], **{t:[.2] for t in r.TARGETS}})
            a.to_csv(root/'a.csv', index=False)
            a.loc[0,'S4'] = .21
            a.to_csv(root/'b.csv', index=False)
            result = r.compare_csv(root/'a.csv', root/'b.csv', expected_rows=1)
            self.assertFalse(result['all_within_1e_12'])
            self.assertEqual(result['probabilities'], 7)
            self.assertAlmostEqual(result['max_absolute_difference'], .01)
            self.assertIsNone(result['private_logloss'])


if __name__ == '__main__':
    unittest.main()
