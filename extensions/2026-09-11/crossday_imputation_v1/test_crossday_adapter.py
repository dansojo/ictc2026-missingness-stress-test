"""Calendar support boundary checks independent of any recovery outcome."""
import unittest
import numpy as np
from run_crossday import adapt_days

class AdapterTests(unittest.TestCase):
    def day(self,start,end,semantics='interval_end'):
        return dict(subject_id='id01',primitive='usage_load_24h',day_start_ns=start,day_end_ns=end,
            starts=np.array([start]),ends=np.array([end]),timestamp_semantics=semantics,date='2026-09-10',
            local_clock_ns=np.array([10]),valid=np.array([True]),raw=np.array([123.]))
    def test_target_calendar_excluded_before_accessing_truth(self):
        target=dict(subject_id='id01',primitive='usage_load_24h',day_start_ns=100,day_end_ns=200)
        self.assertEqual(adapt_days({1:target},target),[])
    def test_previous_interval_touch_allowed(self):
        previous=self.day(0,100);target=self.day(100,200)
        result=adapt_days({1:previous,2:target},target)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['values'][0],123.)
    def test_positive_overlap_rejected(self):
        previous=self.day(0,100);previous['ends']=np.array([101])
        with self.assertRaisesRegex(ValueError,'overlaps'):adapt_days({1:previous},self.day(100,200))
    def test_instant_at_target_midnight_rejected(self):
        previous=self.day(0,100,'instant');previous['starts']=np.array([100])
        with self.assertRaisesRegex(ValueError,'overlaps'):adapt_days({1:previous},self.day(100,200))
    def test_invalid_donor_raw_redacted(self):
        previous=self.day(0,100);previous['valid']=np.array([False])
        self.assertTrue(np.isnan(adapt_days({1:previous},self.day(100,200))[0]['values'][0]))

if __name__=='__main__':unittest.main()
