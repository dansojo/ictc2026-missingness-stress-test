"""Hand-derived cases; no production helper computes expected answers."""
import unittest
from datetime import date
import numpy as np
from crossday_core import prepare_archive,repair_all

D=date(2026,9,14).toordinal()  # Monday; fixtures are independent of study data.
def day(delta,clock,values,subject='id01',primitive='x'):
    return dict(subject_id=subject,primitive=primitive,day_ordinal=D+delta,
                clock_seconds=np.array(clock,dtype=float),values=np.array(values,dtype=float))

class CrossDayTests(unittest.TestCase):
    def archive(self,days,binary=False,log=False,**kw):
        result=prepare_archive('id01','x',D,days,binary=binary,profile_log=log,**kw)
        self.assertIsNotNone(result,'archive construction has not been implemented')
        return result
    def fill(self,arc,clock,values,target,cadence=60):
        result=repair_all(arc,np.array(clock,dtype=float),np.array(values,dtype=float),np.array(target,dtype=bool),cadence_seconds=cadence)
        self.assertIsNotNone(result,'cross-day filling has not been implemented')
        return result

    def test_excludes_target_date_and_other_people_or_primitives(self):
        # Break caught: including the deleted target day's truth or another person.
        a=self.archive([day(0,[100],[999]),day(-1,[100],[900],'id02'),day(-2,[100],[800],primitive='y'),day(-3,[100],[2]),day(1,[100],[6])])
        r=self.fill(a,[100,3600],[np.nan,5],[True,False])['CD_ALL_DAY']
        self.assertEqual(r['values'][0],4)
        self.assertEqual(a['day_ordinals'].tolist(),[D-3,D+1])

    def test_similar_copy_does_not_wrap_across_midnight(self):
        a=self.archive([day(-1,[1,3600,7200],[999,2,3]),day(-2,[1,3600,7200],[999,20,30])],min_similarity_slots=2)
        r=self.fill(a,[3600,7200,86399],[2,3,np.nan],[False,False,True])['CD_SIMILAR_COPY']
        self.assertEqual(r['selected_day_ordinal'],D-1)
        self.assertEqual(r['primary_n'],0)
        self.assertEqual(r['fallback_copy_missing_n'],1)
        self.assertNotEqual(r['values'][-1],999)

    def test_whole_day_control_weights_dates_not_record_density(self):
        # One dense day must not dominate: day means0 and10 ->5, not2.5.
        a=self.archive([day(-1,[1,2,3],[0,0,0]),day(-2,[1],[10])])
        self.assertEqual(self.fill(a,[100],[np.nan],[True])['CD_ALL_DAY']['values'][0],5)

    def test_clock_slot_differs_from_whole_day_control(self):
        a=self.archive([day(-1,[100,2000],[2,100]),day(-2,[100,2000],[6,100])])
        rr=self.fill(a,[100],[np.nan],[True])
        self.assertEqual(rr['CD_TIME_MEAN']['values'][0],4)
        self.assertEqual(rr['CD_ALL_DAY']['values'][0],52)

    def test_two_stage_numeric_median_is_not_pooled_record_median(self):
        a=self.archive([day(-1,[1,2,3],[1,3,100]),day(-2,[1],[11]),day(-3,[1],[20])])
        rr=self.fill(a,[100],[np.nan],[True])['CD_TIME_MEDIAN']
        self.assertEqual(rr['values'][0],11)

    def test_binary_equal_day_vote_tie_zero_and_median_na(self):
        a=self.archive([day(-1,[1],[0]),day(-2,[1],[1])],binary=True)
        rr=self.fill(a,[1],[np.nan],[True])
        self.assertEqual(rr['CD_TIME_MEAN']['values'][0],0)
        self.assertEqual(rr['CD_TIME_MEDIAN']['status'],'not_applicable')
        self.assertTrue(np.isnan(rr['CD_TIME_MEDIAN']['values'][0]))

    def test_binary_nonzero_vote_still_hard_state(self):
        a=self.archive([day(-1,[1],[1]),day(-2,[1],[1]),day(-3,[1],[0])],binary=True)
        self.assertEqual(self.fill(a,[1],[np.nan],[True])['CD_TIME_MEAN']['values'][0],1)

    def test_weektype_restriction_uses_two_matching_dates(self):
        # -3 Friday,-4 Thursday; -1 Sunday. Only weekday dates contribute.
        a=self.archive([day(-3,[1],[2]),day(-4,[1],[6]),day(-1,[1],[100])])
        r=self.fill(a,[1],[np.nan],[True])['CD_WEEKTYPE_MEAN']
        self.assertEqual(r['values'][0],4)
        self.assertEqual(r['primary_n'],1)

    def test_weektype_one_date_falls_back_to_unrestricted_time(self):
        a=self.archive([day(-3,[1],[100]),day(-1,[1],[0]),day(-2,[1],[20])])
        r=self.fill(a,[1],[np.nan],[True])['CD_WEEKTYPE_MEAN']
        self.assertEqual(r['values'][0],40)
        self.assertEqual(r['fallback_slot_n'],1)

    def test_recent_past_weighting_is_separate_from_past_pool(self):
        a=self.archive([day(-7,[1],[0]),day(-14,[1],[30]),day(1,[1],[999])])
        rr=self.fill(a,[1],[np.nan],[True])
        self.assertEqual(rr['CD_PAST_TIME_MEAN']['values'][0],15)
        self.assertEqual(rr['CD_PAST_RECENT_MEAN']['values'][0],10)

    def test_past_fallback_never_uses_future_dates(self):
        a=self.archive([day(1,[1],[100]),day(2,[1],[200])])
        rr=self.fill(a,[1,4000],[np.nan,7],[True,False])['CD_PAST_RECENT_MEAN']
        self.assertEqual(rr['values'][0],7)
        self.assertEqual(rr['fallback_target_n'],1)

    def test_slot_support_counts_distinct_dates_not_samples(self):
        a=self.archive([day(-1,[1,2,3],[100,100,100]),day(-2,[2000],[0])])
        rr=self.fill(a,[1],[np.nan],[True])['CD_TIME_MEAN']
        self.assertEqual(rr['values'][0],50)
        self.assertEqual(rr['fallback_global_n'],1)

    def test_no_archive_uses_target_retained_mean_without_chaining(self):
        a=self.archive([])
        rr=self.fill(a,[1,2,3,4],[2,np.nan,np.nan,10],[False,True,True,False])['CD_TIME_MEAN']
        np.testing.assert_array_equal(rr['values'],[2,6,6,10])
        self.assertEqual(rr['fallback_target_n'],2)

    def test_no_any_donor_abstains_with_unresolved_count(self):
        r=self.fill(self.archive([]),[1],[np.nan],[True])['CD_TIME_MEAN']
        self.assertEqual(r['status'],'abstained_unresolved')
        self.assertEqual(r['unresolved_n'],1)

    def test_similar_day_selected_only_from_visible_profile(self):
        a=self.archive([day(-1,[100,1900,3700],[10,20,123]),day(-2,[100,1900,3700],[1,2,999])],min_similarity_slots=2)
        rr=self.fill(a,[100,1900,3700],[10,20,np.nan],[False,False,True])['CD_SIMILAR_COPY']
        self.assertEqual(rr['values'][2],123)
        self.assertEqual(rr['selected_day_ordinal'],D-1)
        self.assertEqual(rr['similarity_common_slots'],2)

    def test_copy_uses_original_raw_light_even_when_similarity_is_logged(self):
        a=self.archive([day(-1,[100,1900,3700],[9,99,999]),day(-2,[100,1900,3700],[99,999,19])],log=True,min_similarity_slots=2)
        r=self.fill(a,[100,1900,3700],[9,99,np.nan],[False,False,True])['CD_SIMILAR_COPY']
        self.assertEqual(r['values'][2],999)

    def test_copy_missing_clock_falls_back_without_selecting_another_day(self):
        a=self.archive([day(-1,[100,1900,3900],[10,20,123]),day(-2,[100,1900,3700],[1,2,999])],min_similarity_slots=2)
        r=self.fill(a,[100,1900,3700],[10,20,np.nan],[False,False,True],cadence=60)['CD_SIMILAR_COPY']
        self.assertEqual(r['selected_day_ordinal'],D-1)
        self.assertEqual(r['values'][2],561)
        self.assertEqual(r['fallback_slot_n'],1)
        self.assertEqual(r['fallback_copy_missing_n'],1)

    def test_similarity_insufficient_shared_support_is_explicit(self):
        a=self.archive([day(-1,[100,1900],[2,6]),day(-2,[100,1900],[4,8])],min_similarity_slots=2)
        r=self.fill(a,[100,1900],[2,np.nan],[False,True])['CD_SIMILAR_COPY']
        self.assertIsNone(r['selected_day_ordinal'])
        self.assertEqual(r['values'][1],7)
        self.assertEqual(r['fallback_similarity_unavailable_n'],1)

    def test_deleted_truth_must_be_redacted(self):
        a=self.archive([])
        with self.assertRaisesRegex(ValueError,'redacted'):
            self.fill(a,[1],[99],[True])

    def test_clock_domain_and_binary_raw_domain_are_checked(self):
        with self.assertRaises(ValueError):self.archive([day(-1,[86400],[1])])
        with self.assertRaises(ValueError):self.archive([day(-1,[1],[.2])],binary=True)

    def test_inputs_are_not_mutated(self):
        days=[day(-1,[1],[2]),day(-2,[1],[6])]
        before=[d['values'].copy() for d in days]
        a=self.archive(days);t=np.array([1.,2000.]);x=np.array([np.nan,9.]);mask=np.array([True,False])
        xx=x.copy();tt=t.copy();mm=mask.copy()
        rr=repair_all(a,t,x,mask,cadence_seconds=60)
        self.assertIsNotNone(rr)
        for old,d in zip(before,days):np.testing.assert_array_equal(old,d['values'])
        np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(t,tt);np.testing.assert_array_equal(mask,mm)

if __name__=='__main__':unittest.main()
