"""Portable boundary tests; no archived predictions or model states are inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import struct
import tempfile
import unittest

import pandas as pd

PATH = Path(__file__).resolve().parents[1] / 'full_downstream.py'


class FullDownstreamBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), 'Fresh downstream implementation must exist')
        spec = importlib.util.spec_from_file_location('full_downstream_under_test', PATH)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)

    def test_existing_output_is_never_reused_or_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            saved = root / 'old.pack'
            saved.write_bytes(b'old fit')
            with self.assertRaises(FileExistsError):
                self.module.create_fresh_output(root, [])
            self.assertEqual(saved.read_bytes(), b'old fit')

    def test_output_cannot_be_inside_input_or_contain_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                self.module.create_fresh_output(root / 'input' / 'output', [root / 'input'])
            with self.assertRaises(ValueError):
                self.module.create_fresh_output(root / 'output', [root / 'output' / 'input'])

    def test_stage_pin_rejects_changed_data_and_historical_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'primitives.parquet'
            frame = pd.DataFrame({'sensor_day_id': [3], 'value': [2.0]})
            frame.to_parquet(path, index=False)
            manifest = {'schema_name': 'independent_etri_reproduction', 'stage': 'prepare',
                        'status': 'complete', 'fresh_execution': True, 'outputs': {
                            'primitives': {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                           'rows': 1, 'columns': list(frame.columns)}}}
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            loaded = self.module.load_stage(root, 'prepare', ('primitives',))
            pd.testing.assert_frame_equal(loaded[1]['primitives'], frame)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'digest'):
                self.module.load_stage(root, 'prepare', ('primitives',))
            manifest['fresh_execution'] = False
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fresh'):
                self.module.load_stage(root, 'prepare', ('primitives',))

    def test_labels_preserve_dates_and_reject_duplicate_keys_or_nonbinary_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'labels.csv'
            frame = pd.DataFrame({'subject_id': ['id01'], 'lifelog_date': ['2025-01-01'],
                                  'sleep_date': ['2025-01-02'], 'S1': [0], 'S2': [1], 'S3': [0], 'S4': [1]})
            frame.to_csv(path, index=False)
            loaded = self.module.load_labels(path)
            self.assertEqual(loaded.loc[0, 'sleep_date'], pd.Timestamp('2025-01-02'))
            self.assertEqual(tuple(str(loaded[t].dtype) for t in ('S1', 'S2', 'S3', 'S4')), ('int64',) * 4)
            pd.concat([frame, frame]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                self.module.load_labels(path)
            frame['S1'] = [0.5]
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'binary'):
                self.module.load_labels(path)

    def test_fit_topology_requires_every_model_arm_target_per_subject(self):
        rows = []
        for fold in range(10):
            for arm in ('matched_source', 'semantic_16'):
                for model in ('elasticnet', 'lightgbm'):
                    for target in ('S1', 'S2', 'S3', 'S4'):
                        rows.append({'fit_seq': len(rows), 'fold': fold, 'held_out_subject': f'id{fold+1:02d}',
                                     'arm': arm, 'model_name': model, 'target': target})
        # Actual arm vocabulary is obtained only for fixture setup; completeness is checked independently.
        for row in rows:
            row['arm'] = self.module.outcome.DOWNSTREAM_ARM_NAMES[0 if row['arm'] == 'matched_source' else 1]
        self.module.validate_fit_topology(pd.DataFrame(rows))
        with self.assertRaisesRegex(ValueError, '160|topology'):
            self.module.validate_fit_topology(pd.DataFrame(rows[:-1]))
        rows[-1] = dict(rows[-2], fit_seq=159)
        with self.assertRaisesRegex(ValueError, 'topology'):
            self.module.validate_fit_topology(pd.DataFrame(rows))

    def test_progress_counts_complete_pack_records_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fit_states.pack'
            header = b'task-12-downstream-fit-state-pack-v1\n'
            path.write_bytes(header + struct.pack('>Q', 2) + b'{}' + struct.pack('>Q', 20) + b'half')
            self.assertEqual(self.module.count_complete_fit_records(path), 1)

    def test_full_run_rejects_missing_or_duplicate_draws(self):
        rows = [{'subject_id': f'id{subject+1:02d}', 'sensor_day_id': 100 * subject + day,
                 'primitive': primitive, 'draw': draw}
                for subject in range(10) for primitive in self.module.runner._PRIMITIVES
                for day in range(10) for draw in range(50)]
        self.assertTrue(hasattr(self.module, 'validate_full_stress_grid'), 'Full run must validate every draw')
        self.module.validate_full_stress_grid(pd.DataFrame(rows))
        with self.assertRaisesRegex(ValueError, 'grid|draw'):
            self.module.validate_full_stress_grid(pd.DataFrame(rows[:-1]))
        rows[-1] = dict(rows[-2])
        with self.assertRaisesRegex(ValueError, 'grid|draw'):
            self.module.validate_full_stress_grid(pd.DataFrame(rows))

    def test_scoring_requires_the_same_450_label_rows_as_training(self):
        labels = pd.DataFrame({'subject_id': [f'id{subject+1:02d}' for subject in range(10) for _ in range(45)],
                               'S1': [0] * 450})
        self.assertTrue(hasattr(self.module, 'validate_study_labels'), 'Exact study labels must be checked')
        self.module.validate_study_labels(labels, labels.copy())
        changed = labels.copy()
        changed.loc[0, 'S1'] = 1
        with self.assertRaisesRegex(ValueError, 'scoring'):
            self.module.validate_study_labels(labels, changed)
        with self.assertRaisesRegex(ValueError, '450'):
            self.module.validate_study_labels(labels.iloc[:-1], labels.iloc[:-1])

    def test_fewer_than_the_full_study_predictions_cannot_be_complete(self):
        self.assertTrue(hasattr(self.module, 'validate_prediction_count'), 'Full study prediction total must be checked')
        self.module.validate_prediction_count(943392, 19654)
        with self.assertRaisesRegex(ValueError, '943392'):
            self.module.validate_prediction_count(48, 1)
        with self.assertRaisesRegex(ValueError, 'topology'):
            self.module.validate_prediction_count(943392, 19653)

    def test_failed_input_load_records_failure_without_claiming_new_fits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError):
                self.module.run_downstream(root / 'missing_prepare', root / 'missing_stress',
                                           root / 'labels.csv', root / 'labels.csv', root / 'output')
            manifest = json.loads((root / 'output/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['new_model_fits'], 0)
            self.assertEqual(manifest['error_type'], 'FileNotFoundError')
            self.assertTrue((root / 'output/traceback.txt').is_file())

    @unittest.skipUnless(os.environ.get('FULL_DOWNSTREAM_ORIGINAL_TEST_ROOT'),
                         'Optional read-only original synthetic-fixture equivalence test')
    def test_source_fixture_preserves_original_adapter_semantics_and_deletion_guard(self):
        sys.path.insert(0, os.environ['FULL_DOWNSTREAM_ORIGINAL_TEST_ROOT'])
        from ETRI_Human_AI_v2.tests.semantic_indicators import test_stress_downstream_runner as fixture
        from semantic_indicators import stress_downstream, stress_downstream_runner
        bundle = fixture._fixture_bundle()
        frames = {name: stress_downstream.read_task12_replay_table(bundle['authority'], name)
                  for name in self.module.TABLE_NAMES}
        expected = fixture._adapter(stress_downstream_runner)
        arguments = dict(expected_task12_authority_digest=bundle['authority'].content_digest,
                         expected_official_g2_authority_digest=bundle['official_root'],
                         expected_interval_boundary_policy='measurement_support',
                         expected_primitive_digest=fixture._primitive_digest(bundle['primitives']))
        actual = self.module.build_independent_adapter_snapshot(frames, bundle['evidence'], bundle['primitives'], **arguments)
        self.assertEqual(actual.content_digest, expected.content_digest)
        self.assertEqual(actual.source_patch_projection, expected.source_patch_projection)
        self.assertEqual(actual.eligible_target_count, 40)
        self.assertEqual(actual.expected_primary_key_count, 640)
        self.assertEqual(actual.source_patch_projection.row_count, 1920)
        self.assertEqual([value.content_digest for value in actual.contexts],
                         [value.content_digest for value in expected.contexts])
        self.assertEqual([value.content_digest for value in actual.patches],
                         [value.content_digest for value in expected.patches])
        for name in ('prediction_universe', 'ineligible_target_ledger', 'window_exclusion_ledger', 'contiguous_provenance'):
            pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=True)
        for key in actual.reference_tables.frames:
            pd.testing.assert_frame_equal(actual.reference_tables.frames[key], expected.reference_tables.frames[key], check_exact=True)
        changed = {name: frame.copy(deep=True) for name, frame in frames.items()}
        changed['deletion_audit'].loc[0, 'deleted_count'] += 1
        with self.assertRaisesRegex(ValueError, 'deleted|count'):
            self.module.build_independent_adapter_snapshot(changed, bundle['evidence'], bundle['primitives'], **arguments)


if __name__ == '__main__':
    unittest.main()
