import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd

PATH = Path(__file__).resolve().parents[1] / 'full_prepare.py'


class FullPrepareBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), 'Fresh paper prepare driver must exist')
        spec = importlib.util.spec_from_file_location('full_prepare_under_test', PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_existing_output_cannot_be_reused_as_fresh(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'existing'
            root.mkdir()
            (root / 'primitives.parquet').write_bytes(b'old result')
            with self.assertRaises(FileExistsError):
                self.module.create_fresh_output(root, [])
            self.assertEqual((root / 'primitives.parquet').read_bytes(), b'old result')

    def test_outputs_cannot_overlap_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                self.module.create_fresh_output(root / 'raw' / 'new', [root / 'raw'])

    def test_comparison_detects_changed_value_dtype_and_column_order(self):
        a = pd.DataFrame({'id': [1, 2], 'value': [float('nan'), 0.125]})
        self.assertTrue(self.module.compare_frames(a, a.copy())['exact_values_dtypes_order'])
        changed = a.copy()
        changed.loc[1, 'value'] += 1e-12
        self.assertFalse(self.module.compare_frames(changed, a)['exact_values_dtypes_order'])
        self.assertFalse(self.module.compare_frames(a.astype({'id': 'float64'}), a)['exact_values_dtypes_order'])
        self.assertFalse(self.module.compare_frames(a[['value', 'id']], a)['exact_values_dtypes_order'])


if __name__ == '__main__':
    unittest.main()
