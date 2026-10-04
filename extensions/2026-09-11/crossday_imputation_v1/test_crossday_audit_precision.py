"""Audit-only precision regressions; no production imputer or result changes."""
from datetime import date
from fractions import Fraction
import importlib.util
import unittest
import numpy as np
import verify_crossday as first

if importlib.util.find_spec('verify_crossday_v2'):
    import verify_crossday_v2 as revised
else:
    revised=None

ONES=[30,21,19,6,19,0,19,30,18,9,25,30,0,24,4,18,0,27,11,0,0,29,0,17,5,18,30,24,0,30,14,0,18]
COUNTS=[23 if i==16 else 30 for i in range(33)]

class AuditPrecisionTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(revised,'precision-corrected independent oracle absent')

    def test_33_date_exact_half_is_zero_despite_numpy_roundoff(self):
        fractions=[Fraction(n,k) for n,k in zip(ONES,COUNTS)]
        self.assertEqual(sum(fractions,Fraction(0))/33,Fraction(1,2))
        self.assertGreater(np.mean([float(x) for x in fractions]),.5)
        self.assertEqual(revised.precise_binary_vote(fractions),0.)

    def test_past_recency_exact_tie_and_direction(self):
        self.assertEqual(revised.precise_binary_vote([Fraction(1,2)]*3,[1,3,10]),0.)
        self.assertEqual(revised.precise_binary_vote([Fraction(1),Fraction(0)],[1,8]),1.)
        self.assertEqual(revised.precise_binary_vote([Fraction(0),Fraction(1)],[1,8]),0.)

    def test_equal_date_weight_is_not_record_count_weight(self):
        self.assertEqual(revised.precise_binary_vote([Fraction(100,100),Fraction(0,1)]),0.)

    def day(self,ordinal,values):
        start=(ordinal-date(1970,1,1).toordinal())*86400*10**9
        times=start+np.arange(len(values),dtype=np.int64)*10**9
        return dict(subject_id='synthetic',primitive='screen_load_24h',date=date.fromordinal(ordinal).isoformat(),
            raw=np.asarray(values,float),valid=np.ones(len(values),bool),times=times,starts=times,ends=times,
            effective_times=times,local_clock_ns=times-start,day_start_ns=start,day_end_ns=start+86400*10**9,
            timestamp_semantics='instant',cadence_minutes=1)

    def test_full_oracle_regression_keeps_v1_failure_visible(self):
        target=self.day(740000,[1.,0.])
        donors=[self.day(739960+i,[1.]*n+[0.]*(k-n)) for i,(n,k) in enumerate(zip(ONES,COUNTS))]
        deleted=np.array([False,True])
        original=first.independent_repair([target,*donors],target,deleted,first.SETTINGS)
        corrected=revised.independent_repair([target,*donors],target,deleted,first.SETTINGS)
        self.assertEqual(original['CD_PAST_TIME_MEAN']['values'][1],1.,'negative control reproduces the v1 oracle bug')
        self.assertEqual(corrected['CD_PAST_TIME_MEAN']['values'][1],0.)
        for method in first.NEW:
            for field in ['imputed_n','primary_n','fallback_n','fallback_slot_n','fallback_global_n','fallback_target_n','unresolved_n']:
                self.assertEqual(corrected[method][field],original[method][field])

if __name__=='__main__':unittest.main(verbosity=2)
