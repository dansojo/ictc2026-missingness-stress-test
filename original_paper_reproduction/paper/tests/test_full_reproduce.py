import importlib.util
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import hashlib
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import unittest

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))


class FullOrchestrationTests(unittest.TestCase):
    def setUp(self):
        path = PAPER / 'full_reproduce.py'
        self.assertTrue(path.is_file(), 'Full paper orchestration entry point required')
        spec = importlib.util.spec_from_file_location('full_reproduce_under_test', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_grid_contains_all_nine_original_executions_with_exclusive_heavy_stages(self):
        groups = self.module.G2_GROUPS
        runs = [item for group in groups for item in group]
        self.assertEqual(len(runs), 9)
        self.assertEqual(len(set(runs)), 9)
        self.assertEqual(runs.count(('empirical_gap', 2)), 1)
        for group in groups:
            self.assertLessEqual(len(group), 2)
            if any(scenario in ('contiguous_20pct', 'empirical_gap') for scenario, _ in group):
                self.assertEqual(len(group), 1)

    def test_completed_stage_is_not_silently_reused_without_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text(json.dumps({
                'schema_name': 'independent_etri_reproduction', 'stage': 'prepare',
                'status': 'complete', 'fresh_execution': True, 'outputs': {}}), encoding='utf-8')
            with self.assertRaises(FileExistsError):
                self.module.completed_stage(root, 'prepare', resume=False)
            with self.assertRaisesRegex(ValueError, 'outputs|artifact'):
                self.module.completed_stage(root, 'prepare', resume=True)

    def test_running_stage_is_never_skipped_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text(json.dumps({
                'schema_name': 'independent_etri_reproduction', 'stage': 'prepare',
                'status': 'running', 'fresh_execution': True, 'outputs': {}}), encoding='utf-8')
            with self.assertRaises(ValueError):
                self.module.completed_stage(root, 'prepare', resume=True)

    def test_crosswired_resumed_stage_is_rejected_before_it_can_be_skipped(self):
        self.assertTrue(hasattr(self.module, 'validate_stage_lineage'), 'Resume must check ancestor hashes')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'prepare').mkdir()
            path = root / 'prepare/manifest.json'
            path.write_text('{}', encoding='utf-8')
            good = hashlib.sha256(path.read_bytes()).hexdigest()
            context = {'prepare_sha256': good, 'canonical_files': {}, 'g1_file_sha256': 'c' * 64}
            manifest = {'inputs': {'prepare_manifest_sha256': 'b' * 64}}
            with self.assertRaisesRegex(ValueError, 'prepare|ancestor'):
                self.module.validate_stage_lineage(root, 'stress', manifest, context)

    def test_resumed_display_requires_actual_comparison_evidence(self):
        self.assertTrue(hasattr(self.module, 'validate_stage_completion'), 'Completion must verify comparisons')
        with self.assertRaisesRegex(ValueError, 'comparison|display'):
            self.module.validate_stage_completion('displays', {'outputs': {'fig2_results.png': {}}})

    def test_explicit_preprocessing_and_external_inputs_cannot_overlap_output(self):
        self.assertTrue(hasattr(self.module, 'validate_output_roots'), 'All data boundaries must be disjoint')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                self.module.validate_output_roots(root / 'output', [root / 'output/raw'])
            with self.assertRaisesRegex(ValueError, 'overlap'):
                self.module.validate_output_roots(root / 'inputs/output', [root / 'inputs'])

    def test_preprocessing_empty_check_set_cannot_prove_complete(self):
        self.assertTrue(hasattr(self.module, 'verify_preprocessing'), 'Raw provenance validation required')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'preprocessing_run.json').write_text(json.dumps({'status': 'complete', 'checks': {}, 'input_sha256': {}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'preprocessing|raw'):
                self.module.verify_preprocessing(root, root / 'raw')

    def test_external_status_alone_cannot_be_resumed(self):
        self.assertTrue(hasattr(self.module, 'verify_external'), 'External resume must verify inputs and outputs')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'run_manifest.json').write_text(json.dumps({'status': 'COMPLETE_WITH_STATED_SCOPE', 'command': 'external'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'external|External'):
                self.module.verify_external(root, {'roots': {}, 'checked': [], 'sources': []})

    def test_same_output_cannot_run_two_orchestrators_at_once(self):
        self.assertTrue(hasattr(self.module, 'pipeline_lock'), 'Concurrent writes need an exclusive lock')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.module.pipeline_lock(root):
                with self.assertRaisesRegex(RuntimeError, 'running|locked'):
                    with self.module.pipeline_lock(root):
                        self.fail('second writer acquired the output lock')
            with self.module.pipeline_lock(root):
                pass

    def test_prepared_row_subset_cannot_be_complete(self):
        with self.assertRaisesRegex(ValueError, 'prepare|topology'):
            self.module.validate_stage_completion('prepare', {'outputs': {
                'primitives': {'rows': 1, 'columns': ['a']},
                'baselines': {'rows': 1, 'columns': ['a']},
                'representations': {'rows': 1, 'columns': ['a']}}})

    def test_g2_replay_count_distinguishes_main_grid_and_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.g2_completion_fixture(Path(temp))
            self.module.validate_stage_completion('g2', manifest)
            for field in ['main_selected_cell_replay_executions', 'additional_calibration_primitive_replays']:
                broken = copy.deepcopy(manifest)
                broken.pop(field)
                with self.assertRaisesRegex(ValueError, 'nine|topology|calibration'):
                    self.module.validate_stage_completion('g2', broken)

    def g2_completion_fixture(self, root):
        """Actual aggregate schema, with tiny files for the byte/identity boundary."""
        names = ['selected_days', 'cell_replays', 'mask_intervals', 'deletion_audit']
        chunks = []
        physical = {'independent_exact50_runs': 2, 'hashes_identical': True}
        for group in self.module.G2_GROUPS:
            for scenario, replicate in group:
                name = scenario if replicate == 1 else scenario + '-run2'
                directory = root / name
                directory.mkdir()
                chunk = {'schema_name': 'independent_etri_reproduction', 'stage': 'g2_scenario',
                         'status': 'complete', 'fresh_execution': True, 'scenario': scenario,
                         'replicate': replicate, 'parameters': {'draws': 50}, 'outputs': {}}
                for table in names:
                    path = directory / (table + '.parquet')
                    path.write_bytes(('independent fixture bytes for ' + table).encode())
                    chunk['outputs'][table] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                manifest_path = directory / 'manifest.json'
                manifest_path.write_text(json.dumps(chunk), encoding='utf-8')
                chunks.append({'path': str(directory), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
                if scenario == 'empirical_gap':
                    physical['first' if replicate == 1 else 'second'] = {
                        'run_directory': name,
                        **{key: value for table in ['cell_replays', 'mask_intervals'] for key, value in [
                            (table + '_path', name + '/' + table + '.parquet'),
                            (table + '_sha256', chunk['outputs'][table]['sha256'])]}}
        return {
            'outputs': {name: {'rows': count} for name, count in
                        [('selected_days', 800), ('cell_replays', 320000), ('deletion_audit', 320000)]},
            'logical_replays': 320000, 'main_selected_cell_replay_executions': 360000,
            'additional_calibration_primitive_replays': 56000, 'chunks': chunks,
            'empirical_determinism': {
                'original_byte_and_physical_identity_validation': physical,
                **{name: {'exact_values_dtypes_order': True, 'byte_identical': True} for name in names}},
        }

    def test_g2_actual_five_part_determinism_schema_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.g2_completion_fixture(Path(temp))
            self.assertEqual(len(manifest['empirical_determinism']), 5)
            self.module.validate_stage_completion('g2', manifest)

    def test_g2_requires_every_table_value_and_byte_check(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.g2_completion_fixture(Path(temp))
            for table in ['selected_days', 'cell_replays', 'mask_intervals', 'deletion_audit']:
                for field in ['exact_values_dtypes_order', 'byte_identical']:
                    for value in [None, False, 1, 'true']:
                        with self.subTest(table=table, field=field, value=value):
                            broken = copy.deepcopy(manifest)
                            check = broken['empirical_determinism'][table]
                            if value is None:
                                check.pop(field)
                            else:
                                check[field] = value
                            with self.assertRaises(ValueError):
                                self.module.validate_stage_completion('g2', broken)
            for key in manifest['empirical_determinism']:
                broken = copy.deepcopy(manifest)
                broken['empirical_determinism'].pop(key)
                with self.assertRaises(ValueError):
                    self.module.validate_stage_completion('g2', broken)

    def test_g2_requires_original_independent_run_evidence_and_matching_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.g2_completion_fixture(Path(temp))
            key = 'original_byte_and_physical_identity_validation'
            mutations = [
                lambda proof: proof.pop('first'),
                lambda proof: proof.pop('second'),
                lambda proof: proof.update(independent_exact50_runs=1),
                lambda proof: proof.update(hashes_identical=False),
                lambda proof: proof.update(second=copy.deepcopy(proof['first'])),
                lambda proof: proof['second'].update(run_directory='empirical_gap'),
                lambda proof: proof['first'].update(cell_replays_path='../cell_replays.parquet'),
                lambda proof: proof['second'].update(mask_intervals_path='empirical_gap/mask_intervals.parquet'),
                lambda proof: proof['first'].update(cell_replays_sha256='0' * 64),
                lambda proof: [record.update(mask_intervals_sha256='0' * 64) for record in [proof['first'], proof['second']]],
            ]
            for index, mutate in enumerate(mutations):
                with self.subTest(mutation=index):
                    broken = copy.deepcopy(manifest)
                    mutate(broken['empirical_determinism'][key])
                    with self.assertRaises(ValueError):
                        self.module.validate_stage_completion('g2', broken)

    def test_g2_rechecks_all_four_physical_table_hashes(self):
        for table in ['selected_days', 'cell_replays', 'mask_intervals', 'deletion_audit']:
            with self.subTest(table=table), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = self.g2_completion_fixture(root)
                path = root / 'empirical_gap-run2' / (table + '.parquet')
                path.write_bytes(path.read_bytes() + b' changed after comparison')
                with self.assertRaises(ValueError):
                    self.module.validate_stage_completion('g2', manifest)

    def test_g2_rejects_physical_file_reuse_between_independent_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.g2_completion_fixture(root)
            first = root / 'empirical_gap/cell_replays.parquet'
            second = root / 'empirical_gap-run2/cell_replays.parquet'
            second.unlink()
            os.link(first, second)
            with self.assertRaises(ValueError):
                self.module.validate_stage_completion('g2', manifest)

    def test_g2_rejects_changed_chunk_manifest_and_nonindependent_execution(self):
        for field, value in [('replicate', 1), ('scenario', 'outage_1h'), ('status', 'failed')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest = self.g2_completion_fixture(root)
                path = root / 'empirical_gap-run2/manifest.json'
                chunk = json.loads(path.read_text(encoding='utf-8'))
                chunk[field] = value
                path.write_text(json.dumps(chunk), encoding='utf-8')
                for pin in manifest['chunks']:
                    if Path(pin['path']) == path.parent:
                        pin['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    self.module.validate_stage_completion('g2', manifest)

    def test_g2_resume_requires_its_historical_execution_source_pins(self):
        expected = {'commit': 'checkpoint', 'code_tree_sha256': 'a' * 64, 'files': []}
        provider = SimpleNamespace(verify_g2_sources=lambda: expected)
        with mock.patch.dict(sys.modules, {'full_sources': provider}), \
                mock.patch.object(self.module, 'verify_output_records'), \
                mock.patch.object(self.module, 'validate_stage_completion'):
            for stage in ['g2_scenario', 'g2']:
                for pin in [None, {'commit': 'latest'}]:
                    document = {} if pin is None else {'g2_execution_sources': pin}
                    with mock.patch.object(self.module, 'load_stage', return_value=document):
                        with self.assertRaisesRegex(ValueError, 'source|checkpoint'):
                            self.module.verify_stage(Path('unused'), stage)
                document = {'g2_execution_sources': expected}
                with mock.patch.object(self.module, 'load_stage', return_value=document):
                    self.assertEqual(self.module.verify_stage(Path('unused'), stage), document)

    def test_physical_table_schema_and_rows_must_match_the_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'values.parquet'
            pd.DataFrame({'value': [1.0, 2.0]}).to_parquet(path, index=False)
            pin = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'rows': 2, 'columns': ['value']}
            self.module.verify_output_records(root, {'outputs': {'values': pin}}, 'fixture')
            pin['rows'] = 1
            with self.assertRaisesRegex(ValueError, 'row/schema'):
                self.module.verify_output_records(root, {'outputs': {'values': pin}}, 'fixture')
            pin['path'] = '../values.parquet'
            with self.assertRaises(ValueError):
                self.module.verify_output_records(root, {'outputs': {'values': pin}}, 'fixture')

    def test_resume_config_rejection_preserves_previous_manifest_and_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, output = root / 'raw', root / 'output'
            raw.mkdir()
            output.mkdir()
            prior = b'{"schema_name":"independent_etri_paper_pipeline","config":{"old":true}}'
            (output / 'pipeline_manifest.json').write_bytes(prior)
            (output / 'old.log').write_bytes(b'previous execution evidence')
            args = SimpleNamespace(output_dir=output, raw_root=raw, data_root=root, preprocessed_root=None,
                                   external_inputs_json=None, resume=True, reference_sdd=None)
            # External raw verification is read-only and is independently tested;
            # this test isolates a rejection before any pipeline write or child.
            with mock.patch.object(self.module, 'external_context', return_value={
                    'roots': {'external_project': root / 'external_inputs'}, 'sources': [], 'checked': []}):
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    self.module.execute(args)
            self.assertEqual((output / 'pipeline_manifest.json').read_bytes(), prior)
            self.assertEqual((output / 'old.log').read_bytes(), b'previous execution evidence')
            self.assertEqual({path.name for path in output.iterdir()}, {'pipeline_manifest.json', 'old.log'})


if __name__ == '__main__':
    unittest.main()
