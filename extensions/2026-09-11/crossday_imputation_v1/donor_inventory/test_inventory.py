"""Literal boundary fixtures protect the donor cache against coordinate/identity bugs."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

MODULE = Path(__file__).with_name('inventory_core.py')
if MODULE.exists():
    import inventory_core as core
else:
    core = None


def calendar(day='2024-06-27'):
    start = pd.Timestamp(day)
    return pd.DataFrame([dict(subject_id='id01', lifelog_date=start,
        sleep_date=start+pd.Timedelta(days=1), sensor_day_id=1,
        collection_start_date=pd.Timestamp('2024-06-26'),
        elapsed_day_index=(start-pd.Timestamp('2024-06-26')).days+1,
        any_source_record_observed=True, any_measurement_support_observed=True,
        any_sensor_observed=True)])


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core, 'missing tested inventory interface')

    def test_usage_midnight_is_previous_day_and_crossing_support_is_excluded(self):
        frame = pd.DataFrame(dict(subject_id=['id01']*4,
            timestamp=pd.to_datetime(['2024-06-27 00:00','2024-06-27 00:05',
                                      '2024-06-27 00:10','2024-06-28 00:00']),
            usage_total_usage_time=[11.,22.,60000.,120000.]))
        c = core.prepare_frame('usage_load_24h', frame, calendar())
        np.testing.assert_array_equal(c['raw'], [60000.,120000.])
        np.testing.assert_array_equal(c['local_clock_ns']//core.MINUTE_NS, [5,1435])
        np.testing.assert_array_equal(c['source_local_clock_ns']//core.MINUTE_NS, [10,1440])
        np.testing.assert_array_equal((c['ends']-c['starts'])//core.MINUTE_NS,[10,10])
        self.assertEqual(core.target_overlap_count(c, dict(day_start_ns=c['day_end_ns'],
                                                        day_end_ns=c['day_end_ns']+core.DAY_NS)),0)

    def test_mobile_light_preserves_distinct_records_in_same_cadence_bin_and_invalid_raw(self):
        frame = pd.DataFrame(dict(subject_id=['id01']*4,
            timestamp=pd.to_datetime(['2024-06-27 00:02','2024-06-27 00:07',
                                      '2024-06-27 00:12','2024-06-28 00:00']),
            m_light=[10.,-4.,100.,999.]))
        c=core.prepare_frame('mobile_light_exposure_24h',frame,calendar())
        np.testing.assert_array_equal(c['raw'],[10.,-4.,100.])
        np.testing.assert_array_equal(c['valid'],[True,False,True])
        np.testing.assert_array_equal(c['local_clock_ns']//core.MINUTE_NS,[2,7,12])
        summary=core.support_summary(c)
        self.assertEqual(summary['observed_epochs'],2)
        self.assertEqual(summary['within_bin_extra_records'],1)
        self.assertEqual(summary['longest_missing_run_epochs'],142)

    def test_activity_unknown_does_not_become_valid_state(self):
        frame=pd.DataFrame(dict(subject_id=['id01']*2,
            timestamp=pd.to_datetime(['2024-06-27 12:00','2024-06-27 12:01']),
            activity_active_mobility=[1.,1.],activity_unknown=[0.,1.]))
        c=core.prepare_frame('phone_activity_load_24h',frame,calendar())
        np.testing.assert_array_equal(c['raw'],[1.,1.])
        np.testing.assert_array_equal(c['valid'],[True,False])
        np.testing.assert_array_equal(c['raw_frame'].activity_unknown,[0.,1.])

    def test_local_clock_remains_naive_and_aware_inputs_rejected(self):
        frame=pd.DataFrame(dict(subject_id=['id01'],
            timestamp=[pd.Timestamp('2024-06-27 12:00',tz='Asia/Seoul')],screen_on=[1.]))
        with self.assertRaisesRegex(ValueError,'timezone-naive'):
            core.prepare_frame('screen_load_24h',frame,calendar())

    def test_positive_overlap_and_instant_half_open_boundary(self):
        target=dict(day_start_ns=0,day_end_ns=core.DAY_NS)
        interval=dict(timestamp_semantics='interval_end',starts=np.array([-1,0,core.DAY_NS]),
                      ends=np.array([0,1,core.DAY_NS+1]),times=np.array([0,1,core.DAY_NS+1]))
        self.assertEqual(core.target_overlap_count(interval,target),1)
        instant=dict(timestamp_semantics='instant',starts=np.array([-1,0,core.DAY_NS]),
                     ends=np.array([-1,0,core.DAY_NS]),times=np.array([-1,0,core.DAY_NS]))
        self.assertEqual(core.target_overlap_count(instant,target),1)

    def test_donor_counts_exclude_same_date_and_count_distinct_dates_per_slot(self):
        target=dict(day_start_ns=3*core.DAY_NS,is_weekend=False)
        pool=[dict(day_start_ns=d*core.DAY_NS,is_weekend=w,
            valid=np.array(v),local_clock_ns=np.array(m)*core.MINUTE_NS)
            for d,w,v,m in [(1,False,[True,True],[2,5]),
                            (2,True,[False],[2]),
                            (3,False,[True],[2]),
                            (4,False,[True,True],[2,35]),
                            (5,True,[True],[2])]]
        counts,slots=core.donor_support_counts(target,pool)
        self.assertEqual(counts['other_calendar_dates'],4)
        self.assertEqual(counts['valid_donor_dates'],3)
        self.assertEqual(counts['past_valid_donor_dates'],1)
        self.assertEqual(counts['future_valid_donor_dates'],2)
        self.assertEqual(slots[0]['all_valid_donor_dates'],3)
        self.assertEqual(slots[0]['same_weektype_valid_donor_dates'],2)
        self.assertEqual(slots[0]['past_valid_donor_dates'],1)
        self.assertEqual(slots[1]['all_valid_donor_dates'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
