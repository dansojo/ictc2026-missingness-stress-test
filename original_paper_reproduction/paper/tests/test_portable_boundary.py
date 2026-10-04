"""Real filesystem boundary tests; no sensor/model packages required."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]


class PortableBoundaryTests(unittest.TestCase):
    def runner(self):
        path = PACKAGE / 'reproduce.py'
        self.assertTrue(path.is_file(), 'portable reproduction entry point is not implemented')
        spec = importlib.util.spec_from_file_location('paper_reproduce_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_relocated_unicode_input_is_resolved_without_chdir(self):
        r = self.runner()
        with tempfile.TemporaryDirectory(prefix='paper space ') as d:
            root = Path(d) / '자료'
            (root / 'a').mkdir(parents=True)
            (root / 'a/input.csv').write_bytes(b'abc')
            self.assertEqual(r.resolve_file(root, 'a/input.csv'), (root / 'a/input.csv').resolve())

    def test_manifest_cannot_escape_root_on_either_platform(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as d:
            for path in ['../outside.csv', '/etc/passwd', 'C:/data/x', 'a\\..\\outside', '\\\\server\\share\\x']:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    r.resolve_file(Path(d), path)

    def test_long_windows_archive_paths_remain_readable(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as d:
            relative = '/'.join(['long_archive_component_' + str(i) * 30 for i in range(6)]) + '/input.csv'
            target = Path(d).joinpath(*relative.split('/'))
            io_target = Path('\\\\?\\' + str(target)) if os.name == 'nt' else target
            io_target.parent.mkdir(parents=True)
            io_target.write_bytes(b'long path evidence')
            try:
                try:
                    resolved = r.resolve_file(Path(d), relative)
                except OSError as exc:
                    self.fail(f'long archive file was not resolved: {exc}')
                self.assertEqual(resolved.read_bytes(), b'long path evidence')
            finally:
                io_target.unlink()
                current = io_target.parent
                for _ in range(6):
                    current.rmdir()
                    current = current.parent

    def test_hash_mismatch_is_not_silently_accepted(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'x').write_bytes(b'changed')
            entries = [{'root': 'data', 'path': 'x', 'sha256': hashlib.sha256(b'original').hexdigest(), 'bytes': 8}]
            with self.assertRaises(ValueError):
                r.verify_entries(entries, {'data': root})

    def test_existing_output_is_preserved_even_if_empty(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'existing'
            output.mkdir()
            with self.assertRaises(FileExistsError):
                r.create_output(output, [])
            (output / 'keep.txt').write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                r.create_output(output, [])
            self.assertEqual((output / 'keep.txt').read_text(), 'keep')

    def test_output_cannot_be_nested_in_readonly_inputs(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'data'
            root.mkdir()
            with self.assertRaises(ValueError):
                r.create_output(root / 'results', [root])
            self.assertFalse((root / 'results').exists())

    def test_table_evidence_difference_is_detected(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a.csv', Path(d) / 'b.csv'
            a.write_text('key,value\nratio,0.463\n', encoding='utf-8')
            b.write_text('key,value\nratio,0.464\n', encoding='utf-8')
            self.assertFalse(r.compare_csv(a, b)['numeric_equal'])

    def test_csv_newline_or_decimal_spelling_does_not_change_numeric_result(self):
        r = self.runner()
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / 'a.csv', Path(d) / 'b.csv'
            a.write_bytes(b'key,value\nratio,0.463000\n')
            b.write_bytes(b'key,value\r\nratio,0.463\r\n')
            self.assertTrue(r.compare_csv(a, b)['numeric_equal'])

    def test_cli_relocated_manifest_verifies_and_rejects_changed_input_before_output(self):
        with tempfile.TemporaryDirectory(prefix='paper relocated ') as d:
            root = Path(d) / '자료'
            root.mkdir()
            value = root / 'input.csv'
            value.write_bytes(b'original')
            manifest = Path(d) / 'manifest.json'
            manifest.write_text(json.dumps({'groups': {'records': [{
                'root': 'etri_sdd', 'path': 'input.csv', 'bytes': 8,
                'sha256': hashlib.sha256(b'original').hexdigest(),
            }]}}), encoding='utf-8')
            mapping = Path(d) / 'roots.json'
            mapping.write_text(json.dumps({'etri_sdd': str(root)}), encoding='utf-8')
            common = ['--manifest', str(manifest), '--inputs-json', str(mapping)]
            checked = subprocess.run([sys.executable, '-B', str(PACKAGE / 'reproduce.py'),
                                      'verify-inputs', '--group', 'records', *common],
                                     cwd=d, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(json.loads(checked.stdout)['files'], 1)
            value.write_bytes(b'tampered')
            output = Path(d) / 'new-output'
            refused = subprocess.run([sys.executable, '-B', str(PACKAGE / 'reproduce.py'),
                                      'records', *common, '--output', str(output)],
                                     cwd=d, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('input byte/hash mismatch', refused.stderr)
            self.assertFalse(output.exists())

    def test_cli_refuses_existing_output_without_touching_it(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'existing'
            output.mkdir()
            sentinel = output / 'keep.txt'
            sentinel.write_bytes(b'keep exactly')
            refused = subprocess.run([
                sys.executable, '-B', str(PACKAGE / 'reproduce.py'), 'displays',
                '--evidence', str(Path(d) / 'missing-evidence'),
                '--external-results', str(Path(d) / 'missing-external'),
                '--output', str(output),
            ], cwd=d, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('output already exists', refused.stderr)
            self.assertEqual(sentinel.read_bytes(), b'keep exactly')
            self.assertEqual([p.name for p in output.iterdir()], ['keep.txt'])


if __name__ == '__main__':
    unittest.main()
