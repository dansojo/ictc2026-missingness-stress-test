import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd


def module():
    path=Path(__file__).with_name('summarize_recovery.py')
    assert path.exists(), 'Participant-first summarizer is missing'
    spec=importlib.util.spec_from_file_location('summary',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_repeated_rows_do_not_increase_participant_weight():
    m=module()
    f=pd.DataFrame({'subject_id':['id01']*100+['id02','id03'], 'gain':[10.]*100+[-1.,-1.]})
    participants,summary=m.participant_summary(f,['id01','id02','id03'])
    assert summary['median_gain']==-1
    assert summary['n_participants']==3
    assert summary['positive_participants']==1


def test_all_roster_members_preserved_when_effect_unavailable():
    m=module()
    f=pd.DataFrame({'subject_id':['id01','id02'], 'gain':[1.,np.nan]})
    participants,summary=m.participant_summary(f,['id01','id02','id03'])
    assert len(participants)==3
    assert summary['n_participants']==1
    assert participants[2]['finite_keys']==0
    assert np.isnan(participants[2]['median_gain'])


def test_independent_empty_family_remains_unavailable():
    m=module()
    assert hasattr(m,'independent_point_estimate'), 'Independent empty-family handling is missing'
    f=pd.DataFrame({'subject_id':['id01'], 'M0':[np.nan], 'M1':[np.nan], 'M2':[np.nan]})
    assert np.isnan(m.independent_point_estimate(f,['id01','id02'],'M1'))


if __name__=='__main__':
    suite=unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in list(globals().items()) if k.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
