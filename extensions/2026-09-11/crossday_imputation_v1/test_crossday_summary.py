"""Synthetic-first checks: support topology, participant weighting and denominators."""
import importlib.util
import unittest
from datetime import date
import numpy as np
import pandas as pd

if importlib.util.find_spec('summarize_crossday'):
    import summarize_crossday as summary
else:
    summary = None
if importlib.util.find_spec('verify_crossday'):
    import verify_crossday as verifier
else:
    verifier = None


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(summary, 'crossday summary implementation does not yet exist')

    def test_all_applicable_methods_share_one_support(self):
        # A per-comparator intersection would wrongly retain the second key.
        frame = pd.DataFrame({'M0': [10., 10., 10.], 'A': [5., 0., 2.],
                              'B': [4., np.nan, 1.]})
        mask, gains = summary.common_gains(frame, ['M0', 'A', 'B'], [('M0', 'A')])
        self.assertEqual(mask.tolist(), [True, False, True])
        np.testing.assert_array_equal(gains[('M0', 'A')].to_numpy(), [5., np.nan, 8.])

    def test_binary_inapplicable_methods_do_not_empty_support(self):
        applicable = summary.applicable_methods('screen_load_24h')
        for name in ('MEDIAN_RAW', 'LINEAR_RAW', 'CD_TIME_MEDIAN', 'CD_WEEKTYPE_MEDIAN'):
            self.assertNotIn(name, applicable)
        self.assertEqual(len(applicable), 14)
        self.assertEqual(len(summary.applicable_methods('usage_load_24h')), 18)

    def test_people_have_equal_weight_and_missing_roster_is_visible(self):
        # Pooling records would produce 9; participant medians produce 2.
        frame = pd.DataFrame({'subject_id': ['a'] * 101 + ['b', 'c'],
                              'gain': [9.] * 101 + [-3., 2.]})
        people, aggregate = summary.participant_summary(frame, ['a', 'b', 'c', 'd'])
        self.assertEqual(aggregate['median_gain'], 2.)
        self.assertEqual(aggregate['n_roster'], 4)
        self.assertEqual(aggregate['n_participants'], 3)
        self.assertEqual(aggregate['nonfinite_participants'], 1)
        self.assertEqual(aggregate['positive_participants'], 2)
        self.assertEqual(aggregate['negative_participants'], 1)
        self.assertEqual(aggregate['zero_participants'], 0)
        self.assertEqual(people[-1]['scheduled_keys'], 0)
        self.assertTrue(np.isnan(people[-1]['median_gain']))
        rng = np.random.default_rng(20260910)
        sample = np.array([9., -3., 2.])[rng.integers(0, 3, size=(10000, 3))]
        low, high = np.quantile(np.median(sample, axis=1), [.025, .975])
        self.assertEqual((aggregate['ci_low'], aggregate['ci_high']), (low, high))

    def test_zero_effect_and_nonfinite_keys_keep_denominators(self):
        frame = pd.DataFrame({'subject_id': ['a', 'a', 'b', 'b'],
                              'gain': [0., np.nan, -1., np.inf]})
        people, aggregate = summary.participant_summary(frame, ['a', 'b'])
        self.assertEqual(aggregate['scheduled_keys'], 4)
        self.assertEqual(aggregate['finite_keys'], 2)
        self.assertEqual(aggregate['zero_participants'], 1)
        self.assertEqual(aggregate['negative_participants'], 1)
        self.assertEqual([r['excluded_keys'] for r in people], [1, 1])
        self.assertEqual(aggregate['positive_keys'], 0)
        self.assertEqual(aggregate['zero_keys'], 1)
        self.assertEqual(aggregate['negative_keys'], 1)

    def test_geometry_intersection_spans_both_geometries(self):
        index = pd.MultiIndex.from_tuples([('a', 1), ('a', 2)], names=['subject_id', 'draw'])
        columns = pd.MultiIndex.from_product([summary.GEOMETRIES, ['M0', 'A']])
        frame = pd.DataFrame([[6., 3., 2., 1.], [5., 3., 2., np.nan]],
                             index=index, columns=columns)
        mask, deltas = summary.geometry_differences(frame, ['M0', 'A'])
        self.assertEqual(mask.tolist(), [True, False])
        np.testing.assert_array_equal(deltas['A'].to_numpy(), [2., np.nan])

    def test_fallback_fraction_is_ratio_of_counts(self):
        # Equal row-weight percentages yield 50%; correct 1/10 is 10%.
        frame = pd.DataFrame({'subject_id': ['a', 'a'], 'primitive': ['p', 'p'],
                              'geometry': ['g', 'g'], 'method': ['m', 'm'],
                              'imputed_n': [1, 9], 'fallback_n': [1, 0],
                              'deleted_valid_n': [1, 9], 'primary_n': [0, 9],
                              'fallback_slot_n': [1, 0], 'fallback_global_n': [0, 0],
                              'fallback_target_n': [0, 0], 'unresolved_n': [0, 0]})
        row = summary.fallback_table(frame, ['primitive', 'geometry', 'method']).iloc[0]
        self.assertEqual(row.fallback_fraction_of_imputed, .1)
        self.assertEqual(row.primary_fraction_of_attempted, .9)
        self.assertEqual(row.attempted_n, 10)

    def test_na_deleted_points_are_not_attempted_or_genuine_unresolved(self):
        frame = pd.DataFrame({'primitive': ['p'], 'geometry': ['g'], 'method': ['m'],
            'method_status': ['not_applicable'], 'imputed_n': [0], 'fallback_n': [0],
            'deleted_valid_n': [7], 'unresolved_n': [7]})
        row = summary.fallback_table(frame, ['primitive', 'geometry', 'method']).iloc[0]
        self.assertEqual(row.attempted_n, 0)
        self.assertEqual(row.not_applicable_deleted_valid_n, 7)
        self.assertEqual(row.attempted_unresolved_n, 0)
        self.assertTrue(np.isnan(row.unresolved_fraction_of_attempted))

    def test_complete_synthetic_reporting_grid(self):
        rows = []
        for subject in summary.ROSTER:
            for primitive in summary.PRIMITIVES:
                applicable = summary.applicable_methods(primitive)
                for geometry in summary.GEOMETRIES:
                    for method in summary.METHODS:
                        rows.append(dict(subject_id=subject, sensor_day_id=1, primitive=primitive,
                            draw=0, geometry=geometry, method=method, applicable=method in applicable,
                            family='state_ratio' if primitive in summary.PRIMITIVES[:2] else 'intensity',
                            method_status='finite' if method in applicable else 'not_applicable',
                            standardized_error=(2. if method == 'M0' else 1.) if method in applicable else np.nan))
        frame = pd.DataFrame(rows)
        table, people, common = summary.effect_tables(frame)
        geometry, geometry_people, geometry_common = summary.geometry_tables(frame)
        self.assertEqual((len(table), len(people), len(common)), (260, 2600, 100))
        self.assertEqual((len(geometry), len(geometry_people), len(geometry_common)), (90, 900, 50))
        self.assertTrue(common.common_finite.all())
        self.assertTrue(geometry_common.common_finite_both_geometries.all())
        baseline = table.loc[table.comparator.eq('M0') & table.applicable]
        self.assertTrue(baseline.median_gain.eq(1.).all())
        self.assertTrue(baseline.positive_participants.eq(10).all())
        self.assertTrue(baseline.positive_keys.eq(10).all())
        self.assertTrue(table.loc[~table.applicable, 'median_gain'].isna().all())
        self.assertTrue(geometry.loc[geometry.applicable, 'median_gain'].eq(0.).all())
        new = frame.loc[frame.method.isin(summary.NEW_METHODS)].copy()
        for name in ['donor_count_min', 'donor_count_max', 'donor_count_median', 'available_donor_dates',
                     'past_donor_dates', 'future_donor_dates', 'same_weektype_donor_dates',
                     'similarity_common_slots', 'similarity_distance', 'similarity_eligible_days',
                     'target_retained_slots', 'fallback_similarity_unavailable_n', 'fallback_copy_missing_n',
                     'copy_clock_max_lag_seconds', 'imputed_n', 'primary_n', 'fallback_n',
                     'fallback_slot_n', 'fallback_global_n', 'fallback_target_n', 'unresolved_n']:
            new[name] = 0
        new['selected_day_ordinal'] = np.nan
        donors, ledger, copy_summary = summary.donor_tables(new)
        self.assertEqual(len(ledger), 100)
        self.assertEqual(len(copy_summary), 110)
        self.assertTrue(copy_summary.selected_rows.eq(0).all())
        self.assertGreater(len(donors), 0)


class IndependentOracleTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(verifier, 'independent raw oracle does not yet exist')
        self.config = dict(slot_seconds=1800, min_days=2, recency_half_life_days=7,
                           min_similarity_slots=12, min_similarity_fraction=.5)

    def day(self, ordinal, values, clocks=None, primitive='usage_load_24h'):
        day_start = int((ordinal - date(1970, 1, 1).toordinal()) * 86400 * 10**9)
        clocks = np.asarray(clocks if clocks is not None else np.arange(len(values)) * 60, float)
        times = day_start + (clocks * 1e9).astype(np.int64)
        return dict(subject_id='a', primitive=primitive, date=date.fromordinal(ordinal).isoformat(),
                    day_start_ns=day_start, day_end_ns=day_start + 86400 * 10**9,
                    raw=np.asarray(values, float), valid=np.ones(len(values), bool),
                    starts=times, ends=times, times=times, effective_times=times,
                    local_clock_ns=times-day_start, timestamp_semantics='instant', cadence_minutes=1)

    def test_oracle_equal_dates_and_recent_weights(self):
        target = self.day(740000, [100., 0.])
        days = [target, self.day(739999, [0.]), self.day(739992, [12., 12., 12.])]
        repaired = verifier.independent_repair(days, target, np.array([False, True]), self.config)
        self.assertEqual(repaired['CD_ALL_DAY']['values'][1], 6.)
        self.assertAlmostEqual(repaired['CD_PAST_RECENT_MEAN']['values'][1], 4.)
        self.assertEqual(repaired['CD_TIME_MEAN']['primary_n'], 1)
        self.assertEqual(repaired['CD_TIME_MEAN']['donor_count_min'], 2)

    def test_oracle_binary_tie_and_numeric_na(self):
        target = self.day(740000, [1., 0.], primitive='screen_load_24h')
        days = [target, self.day(739999, [0.], primitive=target['primitive']),
                self.day(739998, [1., 1., 1.], primitive=target['primitive'])]
        repaired = verifier.independent_repair(days, target, np.array([False, True]), self.config)
        self.assertEqual(repaired['CD_ALL_DAY']['values'][1], 0.)
        self.assertEqual(repaired['CD_TIME_MEDIAN']['status'], 'not_applicable')

    def test_oracle_target_fallback_excludes_deleted_truth(self):
        target = self.day(740000, [4., 999.])
        repaired = verifier.independent_repair([target], target, np.array([False, True]), self.config)
        self.assertEqual(repaired['CD_ALL_DAY']['values'][1], 4.)
        self.assertEqual(repaired['CD_ALL_DAY']['fallback_target_n'], 1)
        self.assertEqual(repaired['CD_PAST_TIME_MEAN']['donor_count_min'], 0)

    def test_oracle_copy_uses_raw_lux_and_visible_profile(self):
        clocks = np.arange(14) * 1800 + 30
        target = self.day(740000, [2.] * 13 + [9999.], clocks, 'mobile_light_exposure_24h')
        donor = self.day(739999, [2.] * 13 + [100.], clocks, target['primitive'])
        wrong = self.day(739998, [10.] * 14, clocks, target['primitive'])
        deleted = np.arange(14) == 13
        repaired = verifier.independent_repair([target, donor, wrong], target, deleted, self.config)
        self.assertEqual(repaired['CD_SIMILAR_COPY']['values'][-1], 100.)
        self.assertEqual(repaired['CD_SIMILAR_COPY']['selected_day_ordinal'], 739999)
        self.assertEqual(repaired['CD_SIMILAR_COPY']['primary_n'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
