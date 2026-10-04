"""Synthetic probes of the two different definitions of nominal 20% loss."""
from pathlib import Path
import unittest,sys,importlib.util
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'original_paper_reproduction/paper/g2_vendor'))
from semantic_indicators.reliability import generate_mask_intervals
path=ROOT/'extensions/2026-09-11/external_prediction/prediction_core.py'
spec=importlib.util.spec_from_file_location('external_contract_probe',path)
external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)

class GeometryContracts(unittest.TestCase):
    def test_etri_twenty_percent_is_time_window_not_record_count(self):
        for primitive,minutes in [('screen_load_24h',288),('screen_disengagement_p90',72)]:
            masks=generate_mask_intervals(primitive,subject_id='synthetic',sensor_day_id=0,lifelog_date='2024-01-01',scenario='contiguous_20pct',draw=0,root_seed=42)
            row=masks.iloc[0]
            self.assertEqual((row.mask_end-row.mask_start).total_seconds()/60,minutes)
            self.assertGreaterEqual(row.mask_start,row.window_start)
            self.assertLessEqual(row.mask_end,row.window_end)
            repeated=generate_mask_intervals(primitive,subject_id='synthetic',sensor_day_id=0,lifelog_date='2024-01-01',scenario='contiguous_20pct',draw=0,root_seed=42)
            pd.testing.assert_frame_equal(masks,repeated)

    def test_external_twenty_percent_is_valid_record_count(self):
        random,block=external.mask_pair(13,('synthetic','person',0,0))
        self.assertEqual(len(random),2);self.assertEqual(len(block),2)
        self.assertEqual(len(set(random)),2);self.assertEqual(block[1]-block[0],1)

if __name__=='__main__':unittest.main()
