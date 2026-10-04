import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))


class G2FreshBoundaryTests(unittest.TestCase):
    def setUp(self):
        path = PAPER / 'full_g2.py'
        self.assertTrue(path.is_file(), 'Independent G2 driver is required')
        spec = importlib.util.spec_from_file_location('full_g2_under_test', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_incomplete_or_frozen_upstream_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'prepare',
                        'status': 'running', 'fresh_execution': True, 'outputs': {}}
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.module.load_stage(root, 'prepare')
            manifest.update(status='complete', fresh_execution=False)
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.module.load_stage(root, 'prepare')

    def test_tampered_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({'x': [1, 2]})
            record = self.module.save_table(root, 'fixture', frame)
            manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'prepare',
                        'status': 'complete', 'fresh_execution': True, 'outputs': {'fixture': record}}
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            self.module.load_stage(root, 'prepare')
            pd.DataFrame({'x': [1, 3]}).to_parquet(root / record['path'], index=False)
            with self.assertRaises(ValueError):
                self.module.load_stage(root, 'prepare')

    def test_only_primary_runs_calibration_and_modality(self):
        self.assertTrue(self.module.extra_calculations('contiguous_20pct', 1))
        self.assertFalse(self.module.extra_calculations('empirical_gap', 2))
        with self.assertRaises(ValueError):
            self.module.extra_calculations('contiguous_20pct', 2)

    def test_serialized_logical_dtypes_are_restored_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = pd.DataFrame({'root_seed': ['42', '42'],
                                     'observed': pd.Series([True, 0.3], dtype=object)})
            pin = self.module.save_table(root, 'decision_table', original)
            actual = self.module.read_table(root, {'outputs': {'decision_table': pin}}, 'decision_table')
            self.assertEqual(list(map(str, actual.dtypes)), list(map(str, original.dtypes)))
            self.assertEqual(actual['root_seed'].tolist(), ['42', '42'])

    def test_representation_donor_object_dtype_restored_without_weakening_other_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = pd.DataFrame({'donor_subject': pd.Series(['id01', 'id02'], dtype=object), 'k': [14, 14]})
            pin = self.module.save_table(root, 'representations', original)
            actual = self.module.read_table(root, {'outputs': {'representations': pin}}, 'representations')
            pd.testing.assert_frame_equal(actual, original, check_exact=True, check_dtype=True)
            altered = dict(pin, dtypes=['object', 'float64'])
            with self.assertRaisesRegex(ValueError, 'dtype'):
                self.module.read_table(root, {'outputs': {'representations': altered}}, 'representations')

    def test_empirical_independence_and_byte_identity_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / 'one', root / 'two'
            left.mkdir()
            right.mkdir()
            for folder in (left, right):
                for name in ('cell_replays', 'mask_intervals'):
                    pd.DataFrame({'x': [1]}).to_parquet(folder / f'{name}.parquet', index=False)
            self.assertTrue(self.module.validate_empirical_replicates(left, right, root)['hashes_identical'])
            with self.assertRaises(ValueError):
                self.module.validate_empirical_replicates(left, left, root)
            pd.DataFrame({'x': [2]}).to_parquet(right / 'cell_replays.parquet', index=False)
            with self.assertRaises(ValueError):
                self.module.validate_empirical_replicates(left, right, root)


if __name__ == '__main__':
    unittest.main()
