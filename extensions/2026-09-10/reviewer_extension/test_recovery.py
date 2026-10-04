"""Behavior checks for the new exploratory reconstruction, not empirical results."""
import importlib.util
from pathlib import Path
import numpy as np
import unittest


def core():
    path = Path(__file__).with_name('recovery_core.py')
    assert path.exists(), 'The isolated reconstruction implementation is missing'
    spec = importlib.util.spec_from_file_location('recovery_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mean_changes_sum_but_preserves_arithmetic_mean():
    c = core()
    r = c.repair(np.arange(4)*60, np.array([2., np.nan, 6., np.nan]),
                 np.array([False, True, False, True]), binary=False, method='M1', cap_seconds=180)
    np.testing.assert_array_equal(r['values'], [2, 4, 6, 4])
    assert r['values'].mean() == 4
    assert r['values'].sum() == 16


def test_hard_state_tie_is_zero_and_unknown_not_repaired():
    c = core()
    r = c.repair(np.arange(4)*60, np.array([0., 1., np.nan, np.nan]),
                 np.array([False, False, True, False]), binary=True, method='M1', cap_seconds=180)
    np.testing.assert_array_equal(r['values'], [0, 1, 0, np.nan])
    assert r['imputed_n'] == 1


def test_locf_only_real_previous_donor_cap_not_chained():
    c = core()
    r = c.repair(np.arange(7)*60, np.array([np.nan, 2., np.nan, np.nan, np.nan, np.nan, 10.]),
                 np.array([True, False, True, True, True, True, False]),
                 binary=False, method='M2', cap_seconds=180)
    np.testing.assert_array_equal(r['values'], [6, 2, 2, 2, 2, 6, 10])
    assert r['fallback_n'] == 2
    assert r['locf_n'] == 3


def test_no_donor_abstains_without_zero_fill():
    c = core()
    r = c.repair(np.array([0, 60]), np.array([np.nan, np.nan]),
                 np.array([True, False]), binary=False, method='M2', cap_seconds=180)
    assert np.isnan(r['values']).all()
    assert r['status'] == 'abstained_no_retained_donor'
    assert r['unresolved_n'] == 1


def test_rejects_deleted_value_leak():
    c = core()
    with unittest.TestCase().assertRaisesRegex(ValueError, 'hidden'):
        c.repair(np.array([0, 60]), np.array([2., 1000.]), np.array([False, True]),
                 binary=False, method='M1', cap_seconds=180)


def test_duplicate_times_fail_provenance_check():
    c = core()
    with unittest.TestCase().assertRaisesRegex(ValueError, 'strictly'):
        c.repair(np.array([0, 0]), np.array([2., np.nan]), np.array([False, True]),
                 binary=False, method='M2', cap_seconds=180)


def test_interval_overlap_differs_from_instant_membership():
    c = core()
    starts = np.array([0, 10, 20]); ends = np.array([10, 20, 30])
    np.testing.assert_array_equal(c.deleted_by_intervals(starts, ends, [(9, 10)], 'interval_end'), [True, False, False])
    np.testing.assert_array_equal(c.deleted_by_intervals(starts, starts, [(9, 10)], 'instant'), [False, False, False])
    np.testing.assert_array_equal(c.deleted_by_intervals(starts, ends, [(10, 11)], 'interval_end'), [False, True, False])


def test_light_transforms_each_value_after_raw_imputation():
    c = core()
    assert np.isclose(c.feature_value(np.array([0., 8.]), 'exposure_mean'), np.log(9)/2)
    assert c.feature_value(np.array([60000., 120000.]), 'additive_load') == 3.


def test_inputs_are_unchanged_and_natural_invalid_never_donates():
    c = core()
    x = np.array([2., np.nan, np.nan, 10.]); saved = x.copy()
    r = c.repair(np.arange(4)*60, x, np.array([False, False, True, False]),
                 binary=False, method='M2', cap_seconds=180)
    np.testing.assert_array_equal(x, saved)
    np.testing.assert_array_equal(r['values'], [2., np.nan, 2., 10.])


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(v) for k, v in list(globals().items()) if k.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
