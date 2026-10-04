"""Scheduling and worker boundary tests; no full scientific run is launched."""
from __future__ import annotations

from concurrent.futures import Future
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import full_stress as adapter
from test_full_stress import fixture_frames, KEY


class CountingFuture(Future):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def result(self, timeout=None):
        self.owner.consumed += 1
        return super().result(timeout)


class ImmediateExecutor:
    def __init__(self):
        self.submitted = 0
        self.consumed = 0
        self.peak_pending = 0

    def submit(self, fn, task):
        self.submitted += 1
        self.peak_pending = max(self.peak_pending, self.submitted - self.consumed)
        future = CountingFuture(self)
        try:
            future.set_result(fn(task))
        except Exception as error:
            future.set_exception(error)
        return future


class FullStressParallelTests(unittest.TestCase):
    def frames(self):
        return dict(zip(('cell_replays', 'deletion_audit', 'mask_intervals'), fixture_frames()))

    def context(self):
        return {'canonical_root': str(PAPER), 'artifact_hashes': {name: 'a' * 64 for name in self.frames()},
                'scientific_sources': [{'path': 'one.py', 'sha256': 'b' * 64}]}

    def task(self):
        self.assertTrue(hasattr(adapter, 'make_cell_tasks'), 'Cell payload boundary must be implemented')
        return next(adapter.make_cell_tasks(
            (KEY[:3],), {1: pd.Series({'subject_id': 'id01', 'sensor_day_id': 1})},
            self.frames(), self.context()['artifact_hashes'], draws=(0,)))

    def test_task_payload_retains_raw_frame_dtypes_and_has_no_sealed_objects(self):
        self.assertTrue(hasattr(adapter, 'make_cell_tasks'))
        originals = self.frames()
        task = next(adapter.make_cell_tasks(
            (KEY[:3],), {1: pd.Series({'subject_id': 'id01', 'sensor_day_id': 1})},
            originals, self.context()['artifact_hashes'], draws=(0,)))
        self.assertEqual(task['cell'], KEY[:3])
        self.assertEqual(task['cell_index'], 0)
        self.assertEqual(task['draws'], (0,))
        self.assertEqual(set(task), {'cell_index', 'cell', 'draws', 'row', 'frames', 'artifact_hashes'})
        for name, frame in originals.items():
            pd.testing.assert_frame_equal(task['frames'][name], frame, check_exact=True)
        self.assertIsInstance(task['row'], pd.Series)
        task['frames']['cell_replays'].loc[0, 'original_value'] = -99
        self.assertNotEqual(originals['cell_replays'].loc[0, 'original_value'], -99)

    def test_worker_rejects_uninitialized_or_crosswired_payload_before_calculation(self):
        task = self.task()
        with mock.patch.object(adapter, '_CELL_WORKER_CONTEXT', None), self.assertRaisesRegex(RuntimeError, 'initialized'):
            adapter.run_cell_task(task)
        def change_deleted_count(task):
            frame = task['frames']['deletion_audit']
            frame.loc[:, 'deleted_count'] = 287
        mutations = [
            lambda t: t['artifact_hashes'].update(cell_replays='0' * 64),
            lambda t: t.update(cell=('id02', 1, 'screen_load_24h')),
            lambda t: t.update(draws=(0, 49)),
            change_deleted_count,
            lambda t: t['frames'].__setitem__('mask_intervals', t['frames']['mask_intervals'].iloc[:0]),
            lambda t: t.update(prepared=object()),
        ]
        for index, mutate in enumerate(mutations):
            attacked = copy.deepcopy(task)
            mutate(attacked)
            with self.subTest(mutation=index), mock.patch.object(adapter, '_CELL_WORKER_CONTEXT', self.context()), \
                    mock.patch.object(adapter, 'prepare_selected_cell') as prepare, self.assertRaises((ValueError, TypeError)):
                adapter.run_cell_task(attacked)
            prepare.assert_not_called()

    def test_worker_calls_original_prepare_and_replay_after_revalidating_raw_rows(self):
        task = self.task()
        sentinel = object()
        expected = {'cell_replays': pd.DataFrame({'value': [1.0]})}
        with mock.patch.object(adapter, '_CELL_WORKER_CONTEXT', self.context()), \
                mock.patch.object(adapter, '_CELL_SENSOR_INDEXES', {}) as cache, \
                mock.patch.object(adapter, 'prepare_selected_cell', return_value=sentinel) as prepare, \
                mock.patch.object(adapter, 'replay_cell', return_value=expected) as replay:
            result = adapter.run_cell_task(task)
        self.assertIs(prepare.call_args.kwargs['sensor_indexes'], cache)
        self.assertIs(replay.call_args.args[0], sentinel)
        self.assertEqual(replay.call_args.args[1][0][0].draw, 0)
        self.assertIs(result['tables'], expected)
        self.assertEqual(result['cell'], KEY[:3])

    def test_initializer_rejects_source_and_upstream_manifest_mismatch(self):
        self.assertTrue(hasattr(adapter, 'initialize_cell_worker'), 'Each spawned worker must verify inputs')
        context = self.context()
        context.update(prepare_dir=str(PAPER), g2_dir=str(PAPER), prepare_manifest_sha256='c' * 64,
                       g2_manifest_sha256='d' * 64, canonical_files={})
        with mock.patch.object(adapter, 'verify_vendor', return_value=[]), \
                mock.patch.object(adapter, 'read_upstream_manifest') as read, self.assertRaisesRegex(ValueError, 'source'):
            adapter.initialize_cell_worker(context)
        read.assert_not_called()
        with mock.patch.object(adapter, 'verify_vendor', return_value=context['scientific_sources']), \
                mock.patch.object(adapter, 'read_upstream_manifest', return_value={'outputs': {}}), \
                mock.patch.object(adapter, 'sha256', return_value='0' * 64), self.assertRaisesRegex(ValueError, 'manifest|ancestor'):
            adapter.initialize_cell_worker(context)

    def test_scheduling_bounds_unconsumed_payloads_and_propagates_failure(self):
        self.assertTrue(hasattr(adapter, '_bounded_cell_map'), 'Scheduling must use a bounded pending queue')
        executor = ImmediateExecutor()
        values = list(adapter._bounded_cell_map(executor, lambda x: x * 2, iter(range(25)), max_pending=6))
        self.assertEqual(sorted(values), list(range(0, 50, 2)))
        self.assertLessEqual(executor.peak_pending, 6)
        executor = ImmediateExecutor()
        def fail(_):
            raise RuntimeError('worker calculation failed')
        with self.assertRaisesRegex(RuntimeError, 'calculation failed'):
            list(adapter._bounded_cell_map(executor, fail, iter(range(25)), max_pending=6))
        self.assertEqual(executor.submitted, 6, 'No new work after the first worker failure')

    def test_sequential_mode_uses_the_identical_initialized_worker_boundary(self):
        self.assertTrue(hasattr(adapter, 'iter_cell_results'), 'Sequential and parallel modes need one worker boundary')
        with mock.patch.object(adapter, 'initialize_cell_worker') as initialize, \
                mock.patch.object(adapter, 'run_cell_task', side_effect=lambda x: x) as worker, \
                mock.patch.object(adapter, 'ProcessPoolExecutor') as pool:
            self.assertEqual(list(adapter.iter_cell_results(iter([1, 2]), self.context(), workers=1)), [1, 2])
        initialize.assert_called_once_with(self.context())
        self.assertEqual(worker.call_count, 2)
        pool.assert_not_called()

    def test_spawn_context_and_queue_bound_are_explicit(self):
        self.assertTrue(hasattr(adapter, 'iter_cell_results'))
        with mock.patch.object(adapter, 'ProcessPoolExecutor') as pool, \
                mock.patch.object(adapter, '_bounded_cell_map', return_value=iter([])) as bounded:
            list(adapter.iter_cell_results(iter([]), self.context(), workers=3))
        self.assertEqual(pool.call_args.kwargs['max_workers'], 3)
        self.assertEqual(pool.call_args.kwargs['mp_context'].get_start_method(), 'spawn')
        self.assertIs(pool.call_args.kwargs['initializer'], adapter.initialize_cell_worker)
        self.assertEqual(bounded.call_args.kwargs['max_pending'], 6)

    def test_assembly_uses_canonical_cell_order_and_preserves_null_dtypes(self):
        self.assertTrue(hasattr(adapter, 'assemble_cell_tables'))
        cells = (('id01', 1, 'screen_load_24h'), ('id01', 2, 'screen_load_24h'))
        frames = [pd.DataFrame({'value': pd.Series([value], dtype='Float64'),
                                'status': pd.Series([status], dtype='string')})
                  for value, status in [(1.5, 'finite'), (pd.NA, 'unavailable')]]
        results = [{'cell_index': i, 'cell': cells[i],
                    'tables': {name: frames[i] for name in adapter.CELL_RESULT_TABLES}}
                   for i in range(2)]
        actual = adapter.assemble_cell_tables(list(reversed(results)), cells)
        expected = pd.concat(frames, ignore_index=True)
        for frame in actual.values():
            pd.testing.assert_frame_equal(frame, expected, check_exact=True)
        for attacked in [results[:1], results + results[:1], [{**results[0], 'cell': cells[1]}, results[1]]]:
            with self.assertRaises(ValueError):
                adapter.assemble_cell_tables(attacked, cells)

    def test_cli_defaults_to_three_workers_and_allows_sequential_reference(self):
        for extra, expected in [([], 3), (['--workers', '1'], 1)]:
            with mock.patch.object(sys, 'argv', ['full_stress.py', *extra]), \
                    mock.patch.object(adapter, 'run_stress') as run:
                adapter.main()
            self.assertEqual(run.call_args.kwargs.get('workers'), expected)


if __name__ == '__main__':
    unittest.main()
