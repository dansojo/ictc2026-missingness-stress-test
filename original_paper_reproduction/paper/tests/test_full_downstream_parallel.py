"""Prediction scheduling boundary tests; no real full-study fits are run here."""
import ast
import copy
import importlib
from pathlib import Path
import sys
import tempfile
import unittest

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))


class ParallelBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((PAPER / 'full_downstream_parallel.py').is_file(), 'Parallel scheduling adapter required')
        self.module = importlib.import_module('full_downstream_parallel')

    def test_worker_count_is_bounded_and_exact_integer(self):
        for count in (1, 2, 3, 4):
            self.module.validate_worker_count(count)
        for count in (False, True, 0, 5, 2.0, '3'):
            with self.assertRaises((TypeError, ValueError)):
                self.module.validate_worker_count(count)

    def test_dispatch_order_must_be_an_exact_permutation(self):
        self.assertEqual(self.module.validate_submission_order((0, 1), None), (0, 1))
        self.assertEqual(self.module.validate_submission_order((0, 1), (1, 0)), (1, 0))
        for order in ((0,), (1, 1), (0, 2), (False, 1)):
            with self.assertRaises((TypeError, ValueError)):
                self.module.validate_submission_order((0, 1), order)

    def bindings(self):
        expected = {0: ('p01', 0, 48, 'a' * 64), 1: ('p02', 48, 96, 'b' * 64)}
        records = [dict(participant_position=i, participant=pin[0], first_row=pin[1],
                        rows=pin[2], job_digest=pin[3], status='complete') for i, pin in expected.items()]
        return expected, records

    def test_out_of_order_completion_returns_canonical_order(self):
        expected, records = self.bindings()
        actual = self.module.validate_result_inventory(expected, list(reversed(records)))
        self.assertEqual([row['participant_position'] for row in actual], [0, 1])

    def test_missing_duplicate_failed_or_crosswired_worker_never_finalizes(self):
        expected, records = self.bindings()
        for result in (records[:1], records + records[:1], [records[0], dict(records[1], status='failed')],
                       [records[0], dict(records[1], first_row=49)],
                       [records[0], dict(records[1], job_digest='c' * 64)]):
            with self.assertRaises(ValueError):
                self.module.validate_result_inventory(expected, result)

    def test_worker_workspace_cannot_overlap_final_run_or_reuse_old_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / 'run'
            work = self.module.create_worker_workspace(output, root / 'workers')
            self.assertEqual(work, root / 'workers')
            with self.assertRaises(FileExistsError):
                self.module.create_worker_workspace(output, work)
            for path in (output, output / 'workers', root.parent / 'outside_workers'):
                with self.assertRaises(ValueError):
                    self.module.create_worker_workspace(output, path)

    def test_no_untrusted_pickle_file_loading_surface(self):
        tree = ast.parse((PAPER / 'full_downstream_parallel.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertFalse(isinstance(node.func.value, ast.Name) and node.func.value.id == 'pickle'
                                 and node.func.attr in ('load', 'loads'))

    def test_invalid_job_is_rejected_before_any_worker_output(self):
        with self.assertRaises((TypeError, ValueError)):
            self.module.validate_job({'participant': 'p01'})

    def test_original_outer_prefix_and_suffix_ast_are_unchanged(self):
        expected = self.module.source_equivalence_report()
        self.assertTrue(expected['outer_prefix_ast_equal'])
        self.assertTrue(expected['outer_suffix_ast_equal'])
        self.assertTrue(expected['worker_calculation_ast_equal'])

    def test_one_worker_selects_original_scientific_runner(self):
        import full_downstream
        function, arguments = full_downstream.select_snapshot_runner(1)
        self.assertIs(function, self.module.original.run_downstream_snapshot)
        self.assertEqual(arguments, {})
        function, arguments = full_downstream.select_snapshot_runner(3)
        self.assertIs(function, self.module.run_downstream_snapshot_parallel)
        self.assertEqual(arguments, {'workers': 3})

    def test_result_exact_types_are_required(self):
        expected, records = self.bindings()
        for bad in (True, 48.0, '48'):
            with self.assertRaises((ValueError, TypeError)):
                self.module.validate_result_inventory(expected, [dict(records[0], rows=bad), records[1]])

    def test_worker_filename_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('../x', '..\\x', 'x/y', 'x:y'):
                with self.assertRaises(ValueError):
                    self.module._work_file(Path(temp), name)


if __name__ == '__main__':
    unittest.main()
