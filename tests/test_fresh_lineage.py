"""Fresh lineage must bind exact bytes, parent stages, and bounded scientific scope."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fresh_lineage as f


class FreshLineageTests(unittest.TestCase):
    def fixture(self, root):
        raw = root / 'raw-input'; raw.write_bytes(b'raw')
        code = root / 'science.py'; code.write_text('original scientific function')
        old = root / 'frozen.json'; old.write_text('{"original":true}')
        run = root / 'run'; run.mkdir()
        contract = f.create_contract(run, {'raw': (raw, f.digest(raw))},
                                     {'science': (code, f.digest(code))},
                                     {'original': (old, f.digest(old))}, cell_limit=1)
        stage = run / 'raw'; stage.mkdir(); (stage / 'canonical').write_bytes(b'new canonical')
        f.seal_stage(stage, 'raw', contract, {})
        return run, contract, raw

    def test_completed_parent_and_output_bytes_are_reverified(self):
        with tempfile.TemporaryDirectory() as d:
            run, contract, raw = self.fixture(Path(d))
            prep = run / 'prepare'; prep.mkdir(); (prep / 'primitives').write_bytes(b'fresh values')
            f.seal_stage(prep, 'prepare', contract, {'raw': run / 'raw'})
            self.assertEqual(f.verify_stage(prep, 'prepare', contract)['stage'], 'prepare')
            (run / 'raw/canonical').write_bytes(b'changed donor source')
            with self.assertRaises(ValueError): f.verify_stage(prep, 'prepare', contract)

    def test_changed_raw_source_is_not_hidden_by_completed_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            run, contract, raw = self.fixture(Path(d)); raw.write_bytes(b'changed')
            with self.assertRaises(ValueError): f.verify_stage(run / 'raw', 'raw', contract)

    def test_changed_parent_manifest_rejected_even_if_parent_outputs_match(self):
        with tempfile.TemporaryDirectory() as d:
            run, contract, raw = self.fixture(Path(d))
            prep = run / 'prepare'; prep.mkdir(); (prep / 'p').write_text('p')
            f.seal_stage(prep, 'prepare', contract, {'raw': run / 'raw'})
            parent = run / 'raw/lineage.json'; value = json.loads(parent.read_text())
            value['note'] = 'different run'; parent.write_text(json.dumps(value))
            with self.assertRaises(ValueError): f.verify_stage(prep, 'prepare', contract)

    def test_missing_wrong_parent_and_full_scope_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run, contract, raw = self.fixture(Path(d))
            prep = run / 'prepare'; prep.mkdir(); (prep / 'p').write_text('p')
            with self.assertRaises(ValueError): f.seal_stage(prep, 'prepare', contract, {})
            with self.assertRaises(ValueError): f.seal_stage(prep, 'prepare', contract, {'m0': run / 'raw'})
            f.seal_stage(prep, 'prepare', contract, {'raw': run / 'raw'})
            path = prep / 'lineage.json'; value = json.loads(path.read_text())
            value['scope'] = 'full'; path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): f.verify_stage(prep, 'prepare', contract)

    def test_seed_change_is_rejected_even_if_caller_rehashes_contract(self):
        with tempfile.TemporaryDirectory() as d:
            run, contract, raw = self.fixture(Path(d))
            path = Path(contract['path']); value = json.loads(path.read_text())
            value['parameters']['root_seed'] = 43; path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): f.verify_contract({'path':str(path),'sha256':f.digest(path)})

    def test_outputs_cannot_escape_stage(self):
        with tempfile.TemporaryDirectory() as d:
            run, contract, raw = self.fixture(Path(d))
            path = run / 'raw/lineage.json'; value = json.loads(path.read_text())
            value['outputs']={'../../raw-input':f.digest(raw)}; path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): f.verify_stage(run / 'raw', 'raw', contract)

    def test_foreign_cells_and_duplicate_primitive_are_rejected(self):
        contract = {'cell_limit':1}
        keys = [['subject',1,f.PRIMITIVES[0]]]
        parent = {'stage':'m0','cell_keys':keys}
        with self.assertRaises(ValueError):
            f._keys('legacy',[['subject',2,f.PRIMITIVES[0]]],{'m0':parent},contract)
        with self.assertRaises(ValueError):
            f._keys('g2',[*keys,['subject',2,f.PRIMITIVES[0]]],{}, {'cell_limit':2})

    def test_extra_unsealed_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run, contract, raw = self.fixture(Path(d))
            (run/'raw/foreign-canonical').write_bytes(b'data added after sealing')
            with self.assertRaises(ValueError): f.verify_stage(run/'raw','raw',contract)

    def test_non_target_donor_partition_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as d:
            run, contract, raw = self.fixture(Path(d))
            keys = [['subject',1,f.PRIMITIVES[0]]]
            for stage in ('prepare','g2','stress','m0','donors'):
                path = run/stage; path.mkdir(); (path/'input').write_bytes(b'all donor dates, including non-target')
                f.seal_stage(path,stage,contract,{p:run/p for p in f.PARENTS[stage]},
                             cell_keys=[] if stage=='prepare' else keys)
            f.verify_stage(run/'donors','donors',contract)
            (run/'donors/input').write_bytes(b'non-target donor date changed')
            with self.assertRaises(ValueError): f.verify_stage(run/'donors','donors',contract)

    def test_active_config_cannot_switch_to_another_run_with_same_cells(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'active.json'
            path.write_text(json.dumps({'roles':{'m0':'run-a/m0'},'cell_keys':[['subject',1,f.PRIMITIVES[0]]]}))
            expected=f.digest(path)
            with patch.dict('os.environ',{'ICTC_RUN_CONFIG':str(path)}):
                f.verify_active_config(path,expected)
                path.write_text(json.dumps({'roles':{'m0':'run-b/m0'},'cell_keys':[['subject',1,f.PRIMITIVES[0]]]}))
                with self.assertRaises(ValueError): f.verify_active_config(path,expected)
            with patch.dict('os.environ',{'ICTC_RUN_CONFIG':str(root/'other.json')}):
                with self.assertRaises(ValueError): f.verify_active_config(path,expected)


if __name__ == '__main__': unittest.main()
