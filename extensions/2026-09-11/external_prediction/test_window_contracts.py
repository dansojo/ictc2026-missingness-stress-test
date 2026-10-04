"""Real pipeline boundary tests for temporal isolation and retained denominators."""
import unittest
import numpy as np
import pandas as pd
from run_prediction import validate_windows

def fixture():
    rows=[]
    for pid in range(11):
        for day in range(10):
            t=100000+day*3600
            rows.append(dict(dataset='ExtraSensory',participant=f'p{pid}',window_id=str(day),
                label=0 if pid==0 else day%2,window_start=t-3600,window_end=t-20,
                label_time=t,acquisition_duration=20.,times=np.linspace(t-3600,t-20,60),
                values=np.linspace(1,2,60)))
    return pd.DataFrame(rows)

class WindowContracts(unittest.TestCase):
    def test_duplicate_sensor_timestamp_is_rejected_after_canonicalization(self):
        w=fixture();w.at[0,'times'][1]=w.at[0,'times'][0]
        with self.assertRaises(ValueError):validate_windows(w)

    def test_recording_that_extends_into_target_time_is_rejected(self):
        w=fixture();w.at[0,'times'][-1]=w.at[0,'label_time']-19
        with self.assertRaises(ValueError):validate_windows(w)

    def test_single_class_person_is_kept_in_roster(self):
        audit=validate_windows(fixture())
        self.assertEqual(len(audit),11)
        person=audit[audit.participant=='p0'].iloc[0]
        self.assertEqual(person.windows,10)
        self.assertFalse(person.both_classes)

    def test_overlap_between_windows_is_rejected(self):
        w=fixture();w.at[1,'label_time']=w.at[0,'label_time']+3599
        with self.assertRaises(ValueError):validate_windows(w)

if __name__=='__main__':unittest.main(verbosity=2)
