from pathlib import Path
import importlib.util
import unittest
import numpy as np

def check():
    path=Path(__file__).with_name('recovery_support.py')
    assert path.exists(), 'Separate cadence coverage counter is missing'
    spec=importlib.util.spec_from_file_location('support',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    # Distinct records 30 seconds apart share a ten-minute coverage bin.
    times=np.array([0,30,600,86400],dtype=np.int64)*10**9
    assert m.cadence_epochs(times,np.ones(4,dtype=bool),0,86400*10**9,10,0)==2
    # Interval-end usage is placed at its support midpoint; midnight ending
    # belongs to the final bin of the preceding support day.
    times=np.array([0,600,86400],dtype=np.int64)*10**9
    assert m.cadence_epochs(times,np.ones(3,dtype=bool),0,86400*10**9,10,5)==2

if __name__=='__main__':
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([unittest.FunctionTestCase(check)])).wasSuccessful())
