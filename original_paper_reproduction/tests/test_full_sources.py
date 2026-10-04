"""Stage-specific G2 source pinning cannot relabel later Task8A replay hashes."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
sys.path.insert(0, str(PAPER))


class G2CheckpointSourceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((PAPER / 'full_sources.py').is_file(), 'stage-specific source helper is required')
        self.sources = importlib.import_module('full_sources')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def clone(self):
        shutil.copytree(PAPER / 'g2_vendor', self.root / 'g2_vendor')
        shutil.copyfile(PAPER / 'g2_vendor_manifest.json', self.root / 'g2_vendor_manifest.json')
        return json.loads((self.root / 'g2_vendor_manifest.json').read_text())

    def test_exact_original_seven_file_tree_matches_actual_g2_execution(self):
        document = self.sources.verify_g2_sources()
        self.assertEqual(document['commit'], '808a6f6063026b5c4928d191cf1a480ed2886955')
        self.assertEqual(document['code_tree_sha256'], 'c183d9c36b0bde077ebddcdc51f801241c880e1f868026c5ed0c910f53a97c76')
        self.assertEqual(len(document['files']), 7)
        self.assertEqual(sum(record['role'] == 'scientific_module' for record in document['files']), 4)
        self.assertEqual(sum(record['role'] == 'provenance_only' and record['path'] is None for record in document['files']), 3)
        self.assertFalse(any(Path(record['path']).is_absolute() for record in document['files'] if record['path'] is not None))
        self.clone()
        self.assertEqual(self.sources.verify_g2_sources(self.root), document)

    def test_modified_source_fails_even_after_caller_resigns_manifest(self):
        document = self.clone()
        record = next(row for row in document['files'] if row['original_path'].endswith('/primitives.py'))
        path = self.root / record['path']
        path.write_bytes(path.read_bytes() + b'\n# changed source\n')
        record['bytes'] = path.stat().st_size
        record['sha256'] = self.sources.sha256(path)
        (self.root / 'g2_vendor_manifest.json').write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, 'tree|checkpoint'):
            self.sources.verify_g2_sources(self.root)

    def test_source_path_escape_and_added_role_are_rejected(self):
        document = self.clone()
        document['files'][0]['path'] = '../outside.py'
        (self.root / 'g2_vendor_manifest.json').write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            self.sources.verify_g2_sources(self.root)
        document = self.sources.verify_g2_sources()
        extra = copy.deepcopy(document)
        extra['files'].append(copy.deepcopy(extra['files'][0]))
        (self.root / 'g2_vendor_manifest.json').write_text(json.dumps(extra))
        with self.assertRaises(ValueError):
            self.sources.verify_g2_sources(self.root)

    def test_historical_relative_imports_do_not_replace_current_namespace(self):
        from full_prepare import verify_vendor
        verify_vendor()
        current = importlib.import_module('semantic_indicators.primitives')
        original = self.sources.load_g2_modules()
        self.assertIs(importlib.import_module('semantic_indicators.primitives'), current)
        self.assertIsNot(original.primitives, current)
        self.assertEqual(original.reliability.prepare_primitive_replay.__module__, '_ictc_g2_20260809.primitives')
        self.assertIs(original.reliability.PreparedPrimitiveReplay, original.primitives.PreparedPrimitiveReplay)
        self.assertNotIn('source_schema_digest', original.primitives.PreparedPrimitiveReplay.__dataclass_fields__)
        self.assertIn('source_schema_digest', current.PreparedPrimitiveReplay.__dataclass_fields__)

    def test_scenario_dispatches_checkpoint_evaluator_and_records_both_source_sets(self):
        g2 = importlib.import_module('full_g2')
        checkpoint = self.sources.load_g2_modules()
        marker = RuntimeError('checkpoint evaluator reached')
        evaluator = mock.Mock(side_effect=marker)
        historical = SimpleNamespace(**vars(checkpoint))
        historical.reliability = SimpleNamespace(evaluate_g2_reliability=evaluator,
            validate_reviewed_g1_artifact=checkpoint.reliability.validate_reviewed_g1_artifact)
        prepare = self.root / 'prepare'
        prepare.mkdir()
        (prepare / 'manifest.json').write_text('{}')
        output = self.root / 'scenario'
        with mock.patch.object(g2, 'load_g2_modules', return_value=historical), \
             mock.patch.object(g2, 'load_stage', return_value={'canonical_files': {}}), \
             mock.patch.object(g2, 'load_sensors', return_value={}), \
             mock.patch.object(g2, 'read_table', return_value=pd.DataFrame()):
            with self.assertRaisesRegex(RuntimeError, 'checkpoint evaluator reached'):
                g2.run_scenario(prepare, self.root / 'canonical', output, 'contiguous_10pct')
        evaluator.assert_called_once()
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['g2_execution_sources'], self.sources.verify_g2_sources())
        self.assertTrue(manifest['scientific_sources'])
        self.assertIn('full_sources.py', manifest['driver_sources'])
        self.assertEqual(manifest['status'], 'failed')

    def test_aggregate_rejects_old_run_without_checkpoint_pins_before_reading_values(self):
        g2 = importlib.import_module('full_g2')
        from full_prepare import verify_vendor
        verify_vendor()
        prepare = self.root / 'prepare'
        prepare.mkdir()
        (prepare / 'manifest.json').write_text('{}')
        inputs = {'prepare_manifest_sha256': g2.sha256(prepare / 'manifest.json'), 'canonical_files': {}}
        current_prepare = {'canonical_files': {}, 'scientific_sources': [], 'inputs': []}
        stages = {}
        for name in (*g2.SCENARIOS, 'empirical_gap-run2'):
            scenario, replicate = (name.removesuffix('-run2'), 2) if name.endswith('-run2') else (name, 1)
            extra = g2.extra_calculations(scenario, replicate)
            stages[name] = {'scenario': scenario, 'replicate': replicate, 'inputs': inputs,
                            'scientific_sources': [], 'parameters': {'root_seed': 42, 'draws': 50,
                            'max_days': 10, 'run_calibration': extra, 'run_modality': extra}}
        def load(root, stage):
            return current_prepare if stage == 'prepare' else stages[root.name]
        with mock.patch.object(g2, 'verify_vendor', return_value=[]), \
             mock.patch.object(g2, 'load_stage', side_effect=load), \
             mock.patch.object(g2, 'read_table', side_effect=RuntimeError('values read before checkpoint validation')):
            with self.assertRaisesRegex(ValueError, 'source'):
                g2.aggregate(self.root / 'g2_chunks', self.root / 'aggregate', prepare)


if __name__ == '__main__':
    unittest.main()
