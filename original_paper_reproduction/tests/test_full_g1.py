"""Portable G1 provenance and reviewed-reference comparison contracts."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'paper'))
import full_prepare


class FullG1Tests(unittest.TestCase):
    def setUp(self):
        # Missing runner should be an explicit RED assertion, not an import error.
        self.assertTrue((ROOT / 'paper/full_g1.py').is_file(), 'fresh G1 runner is missing')
        self.g1 = importlib.import_module('full_g1')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepare_fixture(self, **overrides):
        prepared = self.root / 'prepare'
        prepared.mkdir()
        frame = pd.DataFrame({'subject_id': ['id%02d' % (i % 10 + 1) for i in range(853)],
                              'sensor_day_id': range(853),
                              'elapsed_day_index': [i // 10 + 1 for i in range(853)]})
        record = full_prepare.save_frame(prepared, 'primitives', frame)
        manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'prepare',
                    'status': 'complete', 'fresh_execution': True,
                    'outputs': {'primitives': record},
                    'parameters': {'interval_boundary_policy': 'measurement_support'}}
        manifest.update(overrides)
        full_prepare.write_json(prepared / 'manifest.json', manifest)
        return prepared

    def reviewed_fixture(self):
        path = ROOT / 'paper/protocol/task-4a-reviewed-g1-pair-summary.json'
        reference = json.loads(path.read_text(encoding='utf-8'))
        return path, pd.DataFrame(reference['pair_summary'])

    def test_rejects_incomplete_or_nonfresh_upstream_before_running(self):
        for change in [{'status': 'running'}, {'fresh_execution': False}, {'stage': 'g2'}]:
            with self.subTest(change=change):
                prepared = self.prepare_fixture(**change)
                with self.assertRaises(ValueError):
                    self.g1.run_g1(prepared, self.root / 'g1')
                self.assertFalse((self.root / 'g1').exists())
                for file in prepared.iterdir():
                    file.unlink()
                prepared.rmdir()

    def test_rejects_changed_primitives_and_output_aliasing(self):
        prepared = self.prepare_fixture()
        with self.assertRaises(ValueError):
            self.g1.run_g1(prepared, prepared / 'nested')
        with (prepared / 'primitives.parquet').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'digest|changed|hash'):
            self.g1.run_g1(prepared, self.root / 'g1')
        self.assertFalse((self.root / 'g1').exists())

    def test_failure_manifest_records_frozen_seed_without_reference_reuse(self):
        prepared = self.prepare_fixture()
        full_prepare.verify_vendor()

        def stop_calculation(frame, **kwargs):
            # The actual scientific evaluator is slow (~241 historical seconds).
            # Fail only after the real runner has validated/loaded its NEW input.
            self.assertEqual(len(frame), 853)
            self.assertEqual(kwargs, {'evaluation_start_day': 22, 'bootstrap_draws': 10000,
                                     'shift_draws': 1000, 'seed': 20260807,
                                     'usage_sensitivity_columns': None})
            raise RuntimeError('deliberate evaluator failure')

        with patch('semantic_indicators.validation.evaluate_convergent_pairs', stop_calculation):
            with self.assertRaisesRegex(RuntimeError, 'deliberate evaluator failure'):
                self.g1.run_g1(prepared, self.root / 'g1')
        manifest = json.loads((self.root / 'g1/manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['schema_name'], 'independent_etri_reproduction')
        self.assertEqual(manifest['parameters']['seed'], 20260807)
        self.assertEqual(manifest['outputs'], {})
        self.assertNotIn('review_status', manifest)

    def test_comparison_preserves_rounding_difference_and_detects_changed_science(self):
        reference, fresh = self.reviewed_fixture()
        fresh.loc[0, 'adjusted_r'] = 0.184482578123
        fresh.loc[0, 'raw_lopo_min'] = 0.4710321
        fresh.loc[0, 'holm_shift_p'] = 0.003996003996
        report = self.g1.compare_g1_pairs(fresh, reference)
        self.assertTrue(report['all_within_reference_precision'])
        self.assertFalse(report['all_exact_values'])
        self.assertGreater(report['fields']['adjusted_r']['max_absolute_difference'], 0)
        changed = fresh.copy()
        changed.loc[0, 'raw_r'] += 0.0001
        self.assertFalse(self.g1.compare_g1_pairs(changed, reference)['all_within_reference_precision'])
        changed = fresh.copy()
        changed.loc[0, 'strong_pair'] = False
        self.assertFalse(self.g1.compare_g1_pairs(changed, reference)['all_within_reference_precision'])
        self.assertFalse(self.g1.compare_g1_pairs(fresh.iloc[:-1], reference)['all_within_reference_precision'])

    def test_comparison_rejects_tampered_reviewed_artifact(self):
        reference, fresh = self.reviewed_fixture()
        artifact = json.loads(reference.read_text())
        artifact['config']['root_seed'] = 42
        changed = self.root / 'changed.json'
        full_prepare.write_json(changed, artifact)
        with self.assertRaises(ValueError):
            self.g1.compare_g1_pairs(fresh, changed)

    def test_mixed_decision_scalars_persist_without_boolean_to_float_coercion(self):
        frame = pd.DataFrame({'observed': pd.Series([True, 0.25], dtype=object),
                              'threshold': pd.Series([True, 0.30], dtype=object)})
        record = self.g1.save_g1_frame(self.root, 'decision_table', frame)
        self.assertEqual(record['format'], 'jsonl')
        values = [json.loads(line) for line in (self.root / record['path']).read_text().splitlines()]
        self.assertIs(values[0]['observed'], True)
        self.assertEqual(values[1]['observed'], 0.25)


if __name__ == '__main__':
    unittest.main()
