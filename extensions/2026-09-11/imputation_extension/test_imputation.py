"""Hand-computed behavior tests for raw-domain restoration, independent of scores."""
import unittest
import numpy as np
from imputation_core import repair_extended

class RepairTests(unittest.TestCase):
    def call(self,t,v,targets,method,binary=False,cap=3):
        return repair_extended(t,v,targets,binary=binary,method=method,cap_seconds=cap)
    def test_median_resists_outlier(self):
        r=self.call([0,1,2,3],[1,np.nan,3,100],[0,1,0,0],'MEDIAN_RAW')
        self.assertEqual(r['values'][1],3)
    def test_even_median_in_numeric_domain(self):
        r=self.call([0,1,2],[1,np.nan,10],[0,1,0],'MEDIAN_RAW')
        self.assertEqual(r['values'][1],5.5)
    def test_linear_uses_elapsed_time_not_row_position(self):
        r=self.call([0,2,10],[0,np.nan,100],[0,1,0],'LINEAR_RAW')
        self.assertEqual(r['values'][1],20)
        self.assertEqual(r['interpolation_n'],1)
    def test_linear_edges_use_mean_without_extrapolation(self):
        r=self.call([0,1,2,3,4],[np.nan,2,np.nan,6,np.nan],[1,0,1,0,1],'LINEAR_RAW')
        np.testing.assert_array_equal(r['values'],[4,2,4,6,4])
        self.assertEqual(r['fallback_n'],2)
        self.assertEqual(r['interpolation_n'],1)
    def test_natural_unknown_is_neither_target_nor_donor(self):
        r=self.call([0,1,2,3],[0,np.nan,np.nan,12],[0,0,1,0],'LINEAR_RAW')
        self.assertTrue(np.isnan(r['values'][1]));self.assertEqual(r['values'][2],8)
    def test_binary_median_is_explicit_not_applicable(self):
        r=self.call([0,1,2],[0,np.nan,1],[0,1,0],'MEDIAN_RAW',binary=True)
        self.assertEqual(r['status'],'not_applicable');self.assertTrue(np.isnan(r['values'][1]))
    def test_binary_linear_is_explicit_not_applicable(self):
        r=self.call([0,1,2],[0,np.nan,1],[0,1,0],'LINEAR_RAW',binary=True)
        self.assertEqual(r['status'],'not_applicable');self.assertEqual(r['imputed_n'],0)
    def test_binary_locf_hard_states_and_tie_fallback_zero(self):
        r=self.call([0,1,2,4],[np.nan,1,0,np.nan],[1,0,0,1],'LOCF',binary=True,cap=3)
        np.testing.assert_array_equal(r['values'],[0,1,0,0])
        self.assertEqual(r['fallback_n'],1)
    def test_cap_boundary_is_inclusive(self):
        r=self.call([0,3,4,10],[2,np.nan,np.nan,10],[0,1,1,0],'LOCF',cap=3)
        np.testing.assert_array_equal(r['values'],[2,2,6,10])
        self.assertEqual(r['locf_n'],1);self.assertEqual(r['fallback_n'],1)
    def test_no_chained_locf_donors(self):
        r=self.call([0,2,4,6],[2,np.nan,np.nan,10],[0,1,1,0],'LOCF',cap=3)
        self.assertEqual(r['values'][2],6)
    def test_unlimited_still_has_leading_fallback(self):
        r=self.call([0,1,500,1000],[np.nan,2,np.nan,10],[1,0,1,0],'LOCF',cap=float('inf'))
        np.testing.assert_array_equal(r['values'],[6,2,2,10])
    def test_no_donors_abstains(self):
        r=self.call([0,1],[np.nan,np.nan],[1,0],'MEDIAN_RAW')
        self.assertEqual(r['status'],'abstained_no_retained_donor');self.assertEqual(r['unresolved_n'],1)
    def test_deleted_values_must_be_redacted(self):
        with self.assertRaises(ValueError):self.call([0,1],[0,1],[0,1],'MEDIAN_RAW')
    def test_invalid_binary_donors_rejected(self):
        with self.assertRaises(ValueError):self.call([0,1],[.5,np.nan],[0,1],'LOCF',binary=True)
    def test_input_unchanged(self):
        v=np.array([1,np.nan,10]);r=self.call([0,1,2],v,[0,1,0],'MEDIAN_RAW')
        self.assertTrue(np.isnan(v[1]));self.assertFalse(np.shares_memory(v,r['values']))
    def test_raw_light_then_log_is_not_log_interpolation(self):
        r=self.call([0,1,2],[0,np.nan,8],[0,1,0],'LINEAR_RAW')
        self.assertEqual(r['values'][1],4)
        self.assertAlmostEqual(float(np.log1p(r['values']).mean()),float((np.log(5)+np.log(9))/3))
        self.assertNotEqual(r['values'][1],2)
    def test_duplicate_times_rejected(self):
        with self.assertRaises(ValueError):self.call([0,0],[0,np.nan],[0,1],'MEDIAN_RAW')
    def test_nonpositive_cap_rejected(self):
        with self.assertRaises(ValueError):self.call([0,1],[0,np.nan],[0,1],'LOCF',cap=0)

if __name__=='__main__':unittest.main(verbosity=2)
