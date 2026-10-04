"""Synthetic checks for private path routing and immutable source gates."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import portable_paths as p


class PortablePathTests(unittest.TestCase):
    def test_optimized_python_is_refused_before_any_stage_can_claim_pass(self):
        result = subprocess.run([sys.executable, '-O', '-B', str(p.RELEASE / 'scripts/portable_run.py'), '--help'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Assertions must remain enabled', result.stderr)

    def test_actual_donor_tree_is_checked_even_when_archive_mapping_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); donor = root / 'cache'; donor.mkdir()
            file = donor / 'donor-day'; file.write_text('original donor')
            pins = {'Z:/study/donor_inventory/cache/donor-day': hashlib.sha256(file.read_bytes()).hexdigest()}
            self.assertEqual(p.check_consumed_tree(root, pins, '/donor_inventory/'), 1)
            file.write_text('altered non-target day')
            with self.assertRaises(ValueError): p.check_consumed_tree(root, pins, '/donor_inventory/')
            with self.assertRaises(ValueError): p.check_consumed_tree(root, {}, '/donor_inventory/')

    def test_mapping_uses_longest_component_prefix_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = root / 'config.json'
            config.write_text(json.dumps({'roles': {}, 'read_roots': [str(root)], 'mappings': [
                {'from': 'Z:/old', 'to': str(root / 'a')},
                {'from': 'Z:/old/data', 'to': str(root / 'b')}]}))
            with patch.dict(os.environ, {'ICTC_RUN_CONFIG': str(config)}):
                self.assertEqual(p.resolve_input('Z:/old/data/x'), root / 'b/x')
                self.assertEqual(p.resolve_input('Z:/old/file'), root / 'a/file')
                with self.assertRaises(ValueError):
                    p.resolve_input('Z:/old/data/../../escape')
                with self.assertRaises(FileNotFoundError):
                    p.resolve_input('Z:/oldest/x')

    def test_existing_unmapped_input_requires_an_explicit_read_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); file = root / 'f'; file.write_text('ok')
            config = root / 'config.json'
            config.write_text(json.dumps({'roles': {}, 'mappings': [], 'read_roots': []}))
            with patch.dict(os.environ, {'ICTC_RUN_CONFIG': str(config)}):
                with self.assertRaises(ValueError): p.resolve_input(file)
            config.write_text(json.dumps({'roles': {}, 'mappings': [], 'read_roots': [str(root)]}))
            with patch.dict(os.environ, {'ICTC_RUN_CONFIG': str(config)}):
                self.assertEqual(p.resolve_input(file), file)

    def test_original_source_hash_and_packaged_hash_are_distinct_gates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); old = root / 'old.py'; old.write_text('old')
            code = root / 'code.py'; code.write_text('adapted')
            digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
            manifest = root / 'docs/provenance/SOURCE_MANIFEST.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps([{'destination': 'code.py', 'sha256': digest(code),
                                            'original_sha256': digest(old)}]))
            config = root / 'config.json'
            config.write_text(json.dumps({'roles': {}, 'read_roots': [str(root)],
                'source_roots': [{'code': str(root), 'original': str(root)}]}))
            with patch.dict(os.environ, {'ICTC_RUN_CONFIG': str(config)}), patch.object(p, 'RELEASE', root):
                p.check_source(code, digest(old), original=old)
                code.write_text('tampered')
                with self.assertRaises(ValueError): p.check_source(code, digest(old), original=old)

    def test_output_overlap_and_existing_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); inputs = root / 'inputs'; inputs.mkdir()
            cfg = root / 'cfg.json'; cfg.write_text(json.dumps({'roles': {}, 'read_roots': [str(inputs)]}))
            with patch.dict(os.environ, {'ICTC_RUN_CONFIG': str(cfg)}):
                with self.assertRaises(ValueError): p.validate_output(inputs / 'run')
                with self.assertRaises(ValueError): p.validate_output(root)
                existing = root / 'existing'; existing.mkdir()
                with self.assertRaises(FileExistsError): p.validate_output(existing)
                self.assertEqual(p.validate_output(root / 'new'), root / 'new')

    def test_omitted_read_root_cannot_make_input_writable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cfg = root / 'config.json'
            cfg.write_text(json.dumps({'roles': {'m0': str(root / 'original')}, 'read_roots': []}))
            with patch.dict(os.environ, {'ICTC_RUN_CONFIG': str(cfg)}):
                with self.assertRaises(ValueError): p.role('m0')
                with self.assertRaises(ValueError): p.validate_output(root / 'original', existing=True)


if __name__ == '__main__': unittest.main()
