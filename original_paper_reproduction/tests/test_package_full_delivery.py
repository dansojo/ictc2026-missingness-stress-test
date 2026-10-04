"""Packaging gates exercised only with temporary proof documents and toy files."""
import copy
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'validation/package_full_delivery.py'


class FullDeliveryTests(unittest.TestCase):
    def test_review_mode_records_a_failure_without_passing_strict_gate(self):
        pipeline, comparison, downstream = self.proof()
        pipeline.update(status='comparison_failed', failed_step='detailed_comparison')
        comparison.update(status='failed', all_scientific_equal=False)
        comparison['stages']['downstream']['scientific_equal'] = False
        self.module.require_review_proof(pipeline, comparison, downstream)
        with self.assertRaises(ValueError):
            self.module.require_complete_proof(pipeline, comparison, downstream)
        for change in ({'status': 'running'}, {'failed_step': 'downstream'}, {'new_model_fits': 0}):
            with self.assertRaises(ValueError):
                self.module.require_review_proof(dict(pipeline, **change), comparison, downstream)
        with self.assertRaises(ValueError):
            self.module.require_review_proof(pipeline, dict(comparison, full_scope=False), downstream)
        with self.assertRaises(ValueError):
            self.module.require_review_proof(pipeline, comparison, dict(downstream, prediction_rows=2))

    def test_review_binding_keeps_failed_table_and_column_differences_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            path = run / 'table.csv'
            path.write_bytes(b'toy')
            pin = {'sha256': self.module.sha256(path), 'rows': 1, 'columns': ['x']}
            columns = {'x': {'exact_equal': False, 'max_absolute_difference': 1e-17}}
            table = {'actual_path': str(path), 'actual_sha256': pin['sha256'],
                     'reference_sha256': '0'*64, 'actual_rows': 1, 'actual_columns': ['x'],
                     'scientific_equal': False, 'columns': columns}
            report = {'stages': {'downstream': {'scientific_equal': False, 'tables': {'toy': table}}}}
            with self.assertRaises(ValueError):
                self.module.comparison_bindings(run, report, {'table.csv': pin})
            result = self.module.comparison_bindings(run, report, {'table.csv': pin}, review_differences=True)
            self.assertFalse(result['downstream']['scientific_equal'])
            self.assertFalse(result['downstream']['tables']['toy']['scientific_equal'])
            self.assertEqual(result['downstream']['tables']['toy']['differing_columns'], columns)
            table['actual_sha256'] = 'a'*64
            with self.assertRaises(ValueError):
                self.module.comparison_bindings(run, report, {'table.csv': pin}, review_differences=True)

    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'Gated full-delivery packaging script required')
        spec = importlib.util.spec_from_file_location('full_delivery_under_test', SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)

    def proof(self):
        return ({'schema_name': 'independent_etri_paper_pipeline', 'status': 'complete',
                 'new_model_fits': 160, 'scientific_experiment_executed': True},
                {'schema_name': 'independent_etri_detailed_comparison_v1', 'status': 'complete',
                 'full_scope': True, 'all_scientific_equal': True,
                 'requested_stages': list(self.module.COMPARISON_STAGES),
                 'stages': {name: {'scientific_equal': True} for name in self.module.COMPARISON_STAGES}},
                {'schema_name': 'independent_etri_reproduction', 'stage': 'downstream', 'status': 'complete',
                 'fresh_execution': True, 'new_model_fits': 160, 'reused_historical_model_fits': 0,
                 'prediction_rows': 943392, 'expected_primary_key_count': 314464})

    def test_partial_or_regressed_experiment_can_never_pass_the_gate(self):
        pipeline, comparison, downstream = self.proof()
        self.module.require_complete_proof(pipeline, comparison, downstream)
        for name, value in (('status', 'running'), ('new_model_fits', 0)):
            changed = dict(pipeline, **{name: value})
            with self.assertRaises(ValueError):
                self.module.require_complete_proof(changed, comparison, downstream)
        for changed in (dict(comparison, full_scope=False), dict(comparison, all_scientific_equal=False),
                        dict(comparison, status='failed')):
            with self.assertRaises(ValueError):
                self.module.require_complete_proof(pipeline, changed, downstream)
        with self.assertRaises(ValueError):
            self.module.require_complete_proof(pipeline, comparison, dict(downstream, prediction_rows=48))

    def test_gate_rejects_failure_hidden_inside_a_complete_comparison(self):
        pipeline, comparison, downstream = self.proof()
        comparison['stages']['stress']['scientific_equal'] = False
        with self.assertRaises(ValueError):
            self.module.require_complete_proof(pipeline, comparison, downstream)

    def test_all_160_fit_comparisons_are_required_without_duplicate_sequences(self):
        report = {'both_packs_validated': True, 'actual_fit_count': 160, 'reference_fit_count': 160,
                  'scientific_equal': True,
                  'fits': [{'fit_seq': i, 'scientific_equal': True} for i in range(160)]}
        self.assertEqual(len(self.module.require_matching_fit_report(report)), 160)
        report['fits'][159]['fit_seq'] = 158
        with self.assertRaises(ValueError):
            self.module.require_matching_fit_report(report)
        report['fits'][159] = {'fit_seq': 159, 'scientific_equal': False}
        with self.assertRaises(ValueError):
            self.module.require_matching_fit_report(report)
        report['fits'].pop()
        with self.assertRaises(ValueError):
            self.module.require_matching_fit_report(report)

    def test_create_cli_refuses_unfinished_synthetic_pipeline_before_any_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'run'
            run.mkdir()
            manifest = run / 'pipeline_manifest.json'
            manifest.write_text(json.dumps({'status': 'running'}))
            original = manifest.read_bytes()
            target = root / 'deliveries'
            result = subprocess.run([sys.executable, '-B', str(SCRIPT), '--run-dir', str(run),
                '--version', 'synthetic-only', '--output-dir', str(target), '--create'],
                capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('not complete', result.stderr)
            self.assertEqual(original, manifest.read_bytes())
            self.assertFalse(target.exists())

    def test_source_path_removal_preserves_every_other_field_and_all_24_hashes(self):
        sources = [{'path': f'semantic_indicators/toy{i}.py', 'sha256': f'{i:064x}',
                    'source': 'C:' + '/Users/' + 'PrivatePerson/source.py'} for i in range(24)]
        original = copy.deepcopy(sources)
        cleaned = self.module.sanitize_manifest('paper/full_vendor_manifest.json', sources)
        self.assertEqual(sources, original)
        self.assertEqual(cleaned, [{'path': row['path'], 'sha256': row['sha256']} for row in sources])
        pre = {'sources': [{'source_path': 'C:' + '/Users/' + 'PrivatePerson/raw.py',
                            'vendor_relative_path': 'raw.py', 'sha256': 'a' * 64, 'bytes': 10}]}
        sanitized = self.module.sanitize_manifest('leaderboard/preprocessing_manifest.json', pre)
        self.assertEqual(sanitized['sources'][0], {'vendor_relative_path': 'raw.py', 'sha256': 'a' * 64, 'bytes': 10})

    def test_private_paths_are_rejected_without_rewriting_code(self):
        self.module.assert_no_private_paths('README.md', b'Use /data or C:/data as explicit inputs.')
        for text in ('C:' + '/Users/' + 'PrivatePerson/raw.csv', '/home/' + 'private_user/data.csv',
                     json.dumps({'path': 'C:' + '\\Users\\' + 'PrivatePerson\\raw.csv'})):
            with self.assertRaisesRegex(ValueError, 'private'):
                self.module.assert_no_private_paths('toy.py', text.encode())

    def test_local_roots_raw_intermediates_and_large_unlisted_files_are_excluded(self):
        for relative in ('paper/input_manifest/local_roots.json', '.venv/python.exe',
                         'paper/raw/secret.csv', 'paper/results/predictions.arrow',
                         'paper/g2_vendor/original_runner/run_task4c2_real.py',
                         'validation/paper_fresh_01/downstream/run/fit/fit_states.pack'):
            self.assertFalse(self.module.delivery_allowed(relative))
        for relative in ('tests/test_full_g1.py', 'tests/test_full_compare.py',
                         'paper/protocol/task-4a-reviewed-g1-pair-summary.json',
                         'paper/tests/fixtures/small.parquet', 'paper/full_vendor/semantic_indicators/outcome.py'):
            self.assertTrue(self.module.delivery_allowed(relative))

    def test_toy_archive_roundtrip_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            staging = root / 'staging'
            staging.mkdir()
            (staging / 'toy.py').write_text('print(1)\n', encoding='utf-8')
            records = self.module.file_manifest(staging)
            archive = root / 'toy-v1.zip'
            result = self.module.write_verified_archive(staging, archive, records)
            self.assertTrue(result['extraction_verified'])
            self.assertEqual(result['verified_payload_files'], 1)
            original = archive.read_bytes()
            with self.assertRaises(FileExistsError):
                self.module.write_verified_archive(staging, archive, records)
            self.assertEqual(archive.read_bytes(), original)
            (staging / 'toy.py').write_text('print(2)\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed|digest'):
                self.module.write_verified_archive(staging, root / 'toy-v2.zip', records)
            self.assertFalse((root / 'toy-v2.zip').exists())

    def test_manifest_pin_checks_reject_changed_files_and_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'toy.py'
            path.write_bytes(b'original')
            pin = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            self.module.verify_pin(root, pin)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'digest'):
                self.module.verify_pin(root, pin)
            with self.assertRaises(ValueError):
                self.module.verify_pin(root, dict(pin, path='../outside.py'))

    def test_comparison_must_bind_current_stage_digests_and_all_table_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            table = {'actual_path': str(root / 'prepare/toy.csv'), 'actual_sha256': 'a' * 64,
                     'reference_sha256': 'b' * 64, 'scientific_equal': True, 'actual_rows': 1, 'actual_columns': ['x']}
            report = {'stages': {'prepare': {'scientific_equal': True, 'tables': {'toy': table}}}}
            pins = {'prepare/toy.csv': {'sha256': 'a' * 64, 'rows': 1, 'columns': ['x']}}
            checked = self.module.comparison_bindings(root, report, pins, expected_roles={'prepare': ['toy']})
            self.assertEqual(checked['prepare']['tables']['toy']['artifact'], 'prepare/toy.csv')
            self.assertNotIn('actual_path', checked['prepare']['tables']['toy'])
            with self.assertRaisesRegex(ValueError, 'scope'):
                self.module.comparison_bindings(root, report, pins, expected_roles={'prepare': ['toy', 'missing']})
            table['actual_sha256'] = 'c' * 64
            with self.assertRaisesRegex(ValueError, 'digest'):
                self.module.comparison_bindings(root, report, pins)

    def test_private_pdf_metadata_is_inspected_after_decoding(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=10, height=10)
        writer.add_metadata({'/Subject': 'C:' + '/Users/' + 'PrivatePerson/private.pdf'})
        data = io.BytesIO()
        writer.write(data)
        with self.assertRaisesRegex(ValueError, 'private'):
            self.module.assert_no_private_paths('toy.pdf', data.getvalue())

    def test_display_report_requires_all_four_current_output_and_reference_pins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run, reference = root / 'run', root / 'reference'
            (run / 'displays').mkdir(parents=True)
            reference.mkdir()
            document = {'outputs': {}}
            rows = {}
            for name in self.module.DISPLAY_FILES:
                # Only hash binding is under test, no image/content claim is made
                # for these toy bytes by the real scientific comparison runner.
                data = ('toy:' + name).encode()
                (run / 'displays' / name).write_bytes(data)
                (reference / name).write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                document['outputs'][name] = {'path': name, 'sha256': digest}
                rows[name] = {'actual_sha256': digest, 'reference_sha256': digest,
                              'content_equal': True, 'byte_identical': True}
            manifest = run / 'displays/manifest.json'
            manifest.write_text(json.dumps(document))
            report = {'schema_name': 'fresh_paper_display_comparison_v1', 'status': 'complete',
                      'all_content_equal': True, 'files': rows,
                      'fresh_display_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
            path = run / 'display_comparison.json'
            path.write_text(json.dumps(report))
            self.assertTrue(self.module.verify_display_comparison(run, document, reference)['all_content_equal'])
            rows[self.module.DISPLAY_FILES[0]]['actual_sha256'] = '0' * 64
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.module.verify_display_comparison(run, document, reference)
            report['files'] = {}
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'partial'):
                self.module.verify_display_comparison(run, document, reference)

    def test_archive_rejects_unsafe_paths_extra_files_and_case_collisions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for spelling in ('../escape', '/absolute', 'C:/absolute', 'dir\\name', 'dir/NUL.py', 'dir./name'):
                with self.assertRaises(ValueError):
                    self.module.relative_path(spelling)
            (root / 'toy.py').write_bytes(b'pass\n')
            records = self.module.file_manifest(root)
            (root / 'extra.csv').write_bytes(b'x\n1\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.module.require_staging_unchanged(root, records)

    def toy_source_tree(self, root):
        def write(relative, data):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data if isinstance(data, bytes) else data.encode())
            return {'path': relative, 'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

        full = []
        for index in range(24):
            pin = write(f'paper/full_vendor/semantic_indicators/toy{index}.py', f'VALUE = {index}\n')
            full.append({'path': f'semantic_indicators/toy{index}.py', 'sha256': pin['sha256'],
                         'source': 'C:' + '/Users/' + 'PrivatePerson/archive.py'})
        write('paper/full_vendor_manifest.json', json.dumps(full))
        checkpoint = []
        for index in range(4):
            pin = write(f'paper/g2_vendor/semantic_indicators/toy{index}.py', f'VALUE = {index}\n')
            checkpoint.append(dict(pin, path=f'g2_vendor/semantic_indicators/toy{index}.py',
                                   original_path=f'src/toy{index}.py', role='scientific_module'))
        for index in range(3):
            checkpoint.append({'path': None, 'original_path': f'sdd/runner{index}.py',
                               'bytes': 10, 'sha256': 'c' * 64, 'role': 'provenance_only'})
        write('paper/g2_vendor_manifest.json', json.dumps({'schema_name': 'ictc_g2_checkpoint_sources_v1',
            'commit': self.module.G2_CHECKPOINT_COMMIT, 'code_tree_sha256': self.module.G2_CHECKPOINT_TREE,
            'files': checkpoint}))
        reporting = []
        for index in range(11):
            pin = write(f'paper/vendor/toy{index}.py', 'pass\n')
            reporting.append(dict(pin, path=f'vendor/toy{index}.py'))
        write('paper/vendor_manifest.json', json.dumps({'files': reporting}))
        pre = []
        for index in range(8):
            pin = write(f'leaderboard/vendor/preprocessing/src/toy{index}.py', 'pass\n')
            pre.append({'vendor_relative_path': f'src/toy{index}.py', 'source_path': 'C:' + '/Users/' + 'PrivatePerson/pre.py',
                        'sha256': pin['sha256'], 'vendor_sha256': pin['sha256'], 'bytes': pin['bytes']})
        write('leaderboard/preprocessing_manifest.json', json.dumps({'sources': pre,
            'stages': [pin['vendor_relative_path'] for pin in pre], 'scientific_source_changes': []}))
        model = write('leaderboard/vendor/modeling/tree_model_reliability_submit.py', 'pass\n')
        write('leaderboard/reproduce.py', f"VENDOR_SHA = '{model['sha256']}'\n")
        write('paper/full_prepare.py', "import json\nfrom pathlib import Path\ndef verify_vendor():\n return json.loads((Path(__file__).parent / 'full_vendor_manifest.json').read_text())\n")
        write('paper/full_sources.py', "import json\nfrom pathlib import Path\ndef verify_g2_sources():\n return json.loads((Path(__file__).parent / 'g2_vendor_manifest.json').read_text())\n")
        write('leaderboard/data_preparation.py', "import json\nfrom pathlib import Path\ndef verify_vendor():\n return json.loads((Path(__file__).parent / 'preprocessing_manifest.json').read_text())['sources']\n")
        write('paper/full_reproduce.py', 'pass\n')
        for relative in ('README.md', 'MODEL_DESCRIPTION.md', 'VERIFICATION_REPORT.md', 'paper/requirements.txt',
                         'leaderboard/requirements.txt', 'paper/protocol/task-4a-reviewed-g1-pair-summary.json',
                         'tests/test_full_g1.py', 'tests/test_full_compare.py', 'tests/test_package_full_delivery.py',
                         'validation/package_full_delivery.py', 'validation/compare_full_displays.py'):
            write(relative, 'pass\n' if relative.endswith('.py') else 'Toy fixture\n')
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=10, height=10)
        stream = io.BytesIO()
        writer.write(stream)
        write('MODEL_DESCRIPTION.pdf', stream.getvalue())
        write('paper/input_manifest/local_roots.json', json.dumps({'private': 'C:' + '/Users/' + 'PrivatePerson/raw'}))
        write('paper/raw/unwanted.csv', 'x\n1\n')
        write('.venv/unwanted.py', 'pass\n')
        return write

    def test_toy_staging_checks_sources_strips_only_approved_fields_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            root, staging = Path(temp) / 'source', Path(temp) / 'staging'
            root.mkdir()
            staging.mkdir()
            write = self.toy_source_tree(root)
            original = (root / 'paper/full_vendor_manifest.json').read_bytes()
            verification = {'status': 'synthetic_fixture_only', 'source_verification': self.module.verify_source_manifests(root)}
            result = self.module.stage_delivery(root, staging, verification)
            self.assertEqual(original, (root / 'paper/full_vendor_manifest.json').read_bytes())
            self.assertEqual(result['sanitized_loader_verification']['staged_full_scientific_sources_verified'], 24)
            self.assertEqual(result['sanitized_loader_verification']['staged_g2_checkpoint_records_verified'], 7)
            self.assertEqual(result['sanitized_loader_verification']['staged_g2_physical_sources_verified'], 4)
            self.assertFalse((staging / 'paper/input_manifest/local_roots.json').exists())
            self.assertFalse((staging / 'paper/raw/unwanted.csv').exists())
            self.assertFalse((staging / '.venv/unwanted.py').exists())
            self.assertTrue((staging / 'tests/test_full_g1.py').is_file())
            summary = json.loads((staging / 'FULL_DELIVERY_VERIFICATION.json').read_text())
            transform = summary['manifest_transformations']['paper/full_vendor_manifest.json']
            self.assertEqual(transform['original_sha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(len(transform['scientific_file_sha256_preserved']), 24)
            write('paper/full_vendor/semantic_indicators/toy3.py', 'MUTATED = True\n')
            with self.assertRaisesRegex(ValueError, 'digest'):
                self.module.verify_source_manifests(root)

    def test_private_path_in_code_blocks_staging_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root, staging = Path(temp) / 'source', Path(temp) / 'staging'
            root.mkdir()
            staging.mkdir()
            write = self.toy_source_tree(root)
            payload = "ROOT = 'C:" + '/Users/' + "PrivatePerson/data'\n"
            write('paper/private.py', payload)
            verification = {'status': 'synthetic_fixture_only', 'source_verification': self.module.verify_source_manifests(root)}
            with self.assertRaisesRegex(ValueError, 'private'):
                self.module.stage_delivery(root, staging, verification)
            self.assertEqual((root / 'paper/private.py').read_text(), payload)


if __name__ == '__main__':
    unittest.main()
