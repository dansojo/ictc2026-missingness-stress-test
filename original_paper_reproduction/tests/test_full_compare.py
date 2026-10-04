"""Mismatch detectors for the independent full-study comparison runner."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paper'))


class FullCompareTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'paper/full_compare.py').is_file(), 'full comparison runner is missing')
        self.cmp = importlib.import_module('full_compare')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_numeric_tolerance_is_diagnostic_and_never_hides_exact_regression(self):
        reference = pd.DataFrame({'row_seq': [0, 1], 'error': [0.25, 0.5]})
        actual = reference.copy()
        actual.loc[1, 'error'] += 1e-10
        result = self.cmp.compare_frames(actual, reference, diagnostic_atol=1e-8)
        self.assertFalse(result['scientific_equal'])
        self.assertFalse(result['exact_values_order'])
        self.assertTrue(result['within_diagnostic_tolerance'])
        self.assertEqual(result['columns']['error']['different_count'], 1)
        self.assertGreater(result['columns']['error']['max_absolute_difference'], 9e-11)
        self.assertEqual(result['columns']['error']['examples'][0]['row_position'], 1)

    def test_runtime_exception_does_not_hide_scientific_status_or_count(self):
        reference = pd.DataFrame({'runtime_seconds': [10.0], 'draws': [50], 'status': ['complete']})
        actual = reference.copy()
        actual['runtime_seconds'] = 20.0
        policy = {'runtime_seconds': 'per-scenario wall clock; reliability.py:4163'}
        result = self.cmp.compare_frames(actual, reference, exceptions=policy)
        self.assertTrue(result['scientific_equal'])
        self.assertFalse(result['exact_values_order'])
        self.assertEqual(result['columns']['runtime_seconds']['different_count'], 1)
        actual['draws'] = 49
        self.assertFalse(self.cmp.compare_frames(actual, reference, exceptions=policy)['scientific_equal'])

    def test_order_schema_and_large_integer_identity_are_exact(self):
        reference = pd.DataFrame({'seed': pd.Series([2**63 + 1, 2**63 + 2], dtype='uint64'),
                                  'value': [1.0, 2.0]})
        actual = reference.copy()
        actual.loc[0, 'seed'] = 2**63 + 2
        self.assertFalse(self.cmp.compare_frames(actual, reference)['scientific_equal'])
        self.assertFalse(self.cmp.compare_frames(reference.iloc[::-1].reset_index(drop=True), reference)['scientific_equal'])
        changed_type = reference.copy()
        changed_type['value'] = changed_type['value'].astype('float32')
        self.assertFalse(self.cmp.compare_frames(changed_type, reference)['scientific_equal'])

    def test_stress_content_digests_are_not_blanket_exceptions(self):
        reference = pd.DataFrame({'mask_digest': ['a' * 64], 'standardized_absolute_error': [0.25]})
        actual = reference.copy()
        actual['mask_digest'] = 'b' * 64
        self.assertEqual(self.cmp.table_exceptions('stress', 'cell_replays'), {})
        self.assertFalse(self.cmp.compare_frames(actual, reference)['scientific_equal'])
        policy = self.cmp.table_exceptions('downstream', 'contiguous_provenance')
        self.assertIn('official_g2_authority_digest', policy)
        self.assertNotIn('record_deletion_digest', policy)
        self.assertEqual(self.cmp.table_exceptions('downstream', 'observed_predictions'), {})

    def test_nested_null_and_reason_records_compare_without_string_coercion(self):
        reference = pd.DataFrame({'reason_counts': [[{'reason': 'missing', 'count': 1}]],
                                  'value': [float('nan')]})
        actual = reference.copy(deep=True)
        self.assertTrue(self.cmp.compare_frames(actual, reference)['scientific_equal'])
        actual.at[0, 'reason_counts'] = [{'reason': 'missing', 'count': 2}]
        self.assertFalse(self.cmp.compare_frames(actual, reference)['scientific_equal'])

    def test_mixed_decision_numeric_value_reports_actual_difference(self):
        reference = pd.DataFrame({'observed': pd.Series([0.25, True, 'not_applicable'], dtype=object)})
        actual = reference.copy()
        actual.at[0, 'observed'] = 0.25000001
        result = self.cmp.compare_frames(actual, reference, diagnostic_atol=1e-6)
        self.assertFalse(result['scientific_equal'])
        self.assertTrue(result['within_diagnostic_tolerance'])
        self.assertGreater(result['columns']['observed']['max_absolute_difference'], 9e-9)

    def test_empty_table_preserves_logical_dtype_check(self):
        reference, actual = self.root / 'ref.parquet', self.root / 'actual.parquet'
        pd.DataFrame({'draw': pd.Series([], dtype='int64')}).to_parquet(reference, index=False)
        pd.DataFrame({'draw': pd.Series([], dtype='Int64')}).to_parquet(actual, index=False)
        result = self.cmp.compare_table(actual, reference)
        self.assertTrue(result['arrow_schema_equal'])
        self.assertFalse(result['dtypes_equal'])
        self.assertFalse(result['scientific_equal'])

    def write_arrow(self, path, batches):
        schema = pa.schema([('row_seq', pa.int64()), ('probability', pa.float64())])
        with path.open('wb') as stream, ipc.new_stream(stream, schema) as writer:
            for rows in batches:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))

    def test_arrow_comparison_aligns_different_batch_boundaries_and_reports_global_key(self):
        rows = [{'row_seq': i, 'probability': i / 10} for i in range(7)]
        ref, actual = self.root / 'ref.arrow', self.root / 'actual.arrow'
        self.write_arrow(ref, [rows[:2], rows[2:5], rows[5:]])
        self.write_arrow(actual, [rows[:4], rows[4:]])
        equal = self.cmp.compare_table(actual, ref, batch_size=3)
        self.assertTrue(equal['scientific_equal'])
        self.assertFalse(equal['byte_identical'])
        changed = [dict(row) for row in rows]
        changed[6]['probability'] = 0.99
        self.write_arrow(actual, [changed[:4], changed[4:]])
        result = self.cmp.compare_table(actual, ref, batch_size=3)
        self.assertFalse(result['scientific_equal'])
        self.assertEqual(result['columns']['probability']['examples'][0]['row_position'], 6)
        self.assertEqual(result['columns']['probability']['examples'][0]['key']['row_seq'], 6)

    def test_missing_stage_cannot_be_reported_complete(self):
        actual, reference = self.root / 'new', self.root / 'archive'
        actual.mkdir()
        reference.mkdir()
        report = self.cmp.run_comparisons(actual, reference, self.root / 'report.json', stages=('prepare',))
        self.assertEqual(report['status'], 'failed')
        self.assertFalse(report['full_scope'])
        self.assertFalse(report['all_scientific_equal'])
        self.assertIn('prepare', report['stages'])

    def test_single_scenario_comparison_rejects_scientific_change_after_runtime_exception(self):
        actual, reference = self.root / 'actual_scenario', self.root / 'reference_scenario'
        actual.mkdir()
        reference.mkdir()
        table = pd.DataFrame({'runtime_seconds': [1.0], 'draws': [50]})
        table.to_parquet(reference / 'run_manifest.parquet', index=False)
        table['runtime_seconds'] = 2.0
        table.to_parquet(actual / 'run_manifest.parquet', index=False)
        ar = {'path': 'run_manifest.parquet', 'rows': 1, 'sha256': self.cmp.file_sha(actual / 'run_manifest.parquet')}
        rr = {'path': 'run_manifest.parquet', 'rows': 1, 'sha256': self.cmp.file_sha(reference / 'run_manifest.parquet')}
        manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'g2_scenario',
                    'status': 'complete', 'fresh_execution': True, 'outputs': {'run_manifest': ar}}
        (actual / 'manifest.json').write_text(json.dumps(manifest))
        (reference / 'summary.json').write_text(json.dumps({'artifacts': {'run_manifest': rr}}))
        self.assertTrue(self.cmp.compare_g2_scenario(actual, reference)['scientific_equal'])
        table['draws'] = 49
        table.to_parquet(actual / 'run_manifest.parquet', index=False)
        ar['sha256'] = self.cmp.file_sha(actual / 'run_manifest.parquet')
        (actual / 'manifest.json').write_text(json.dumps(manifest))
        result = self.cmp.compare_g2_scenario(actual, reference)
        self.assertFalse(result['scientific_equal'])
        self.assertEqual(result['tables']['run_manifest']['columns']['draws']['different_count'], 1)

    def test_model_semantic_parser_preserves_tree_threshold_changes(self):
        reference = b'tree\nversion=v4\nTree=0\nthreshold=0.25 0.5\nleaf_value=1.0 2.0\n[learning_rate: 0.1]\n'
        whitespace = b'tree\nversion=v4\nTree=0\nthreshold=0.25   0.5\nleaf_value=1.0 2.0\n[learning_rate: 0.1]\n'
        changed = reference.replace(b'threshold=0.25', b'threshold=0.26')
        a = self.cmp.model_state_document('lightgbm-model-string-v1', reference)
        b = self.cmp.model_state_document('lightgbm-model-string-v1', whitespace)
        c = self.cmp.model_state_document('lightgbm-model-string-v1', changed)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == '__main__':
    unittest.main()
