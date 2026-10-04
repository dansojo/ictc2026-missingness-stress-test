import unittest
import pandas as pd
import numpy as np
from summarize_extension import common_gains

class SummaryTests(unittest.TestCase):
    def test_all_applicable_methods_share_finite_keys(self):
        x=pd.DataFrame({'subject_id':['id01','id01'],'M0':[4.,10.],'M1':[2.,1.],'MEDIAN_RAW':[3.,np.nan]})
        y=common_gains(x,['M0','M1','MEDIAN_RAW'])
        self.assertEqual(y['gain_M1'].iloc[0],2.)
        self.assertTrue(np.isnan(y['gain_M1'].iloc[1]))
    def test_not_applicable_columns_do_not_remove_binary_cases(self):
        x=pd.DataFrame({'subject_id':['id01'],'M0':[4.],'M1':[2.],'MEDIAN_RAW':[np.nan]})
        y=common_gains(x,['M0','M1'])
        self.assertEqual(y['gain_M1'].iloc[0],2.)
    def test_negative_gain_is_preserved(self):
        x=pd.DataFrame({'subject_id':['id01'],'M0':[1.],'M1':[3.]})
        y=common_gains(x,['M0','M1'])
        self.assertEqual(y['gain_M1'].iloc[0],-2.)

if __name__=='__main__':unittest.main(verbosity=2)
