"""Check prepared windows, source immutability, label provenance and boundaries."""
from pathlib import Path
from portable_paths import role, resolve_input
import json, hashlib
import numpy as np
import pandas as pd

BASE=role('work_output')
INPUT=role('external_inventory_inputs')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    windows=pd.read_parquet(INPUT/'eligible_prediction_windows.parquet')
    sources=json.loads((INPUT/'source_manifest.json').read_text(encoding='utf-8'))
    source_checks=[]
    for row in sources:
        p=resolve_input(row['path']);ok=p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        source_checks.append(ok)
        assert ok,f'Source changed: {p}'
    assert not windows.duplicated(['dataset','participant','window_id']).any()
    counts=[];gap_min=[];gap_max=[]
    for row in windows.itertuples(index=False):
        t=np.asarray(row.times,float);v=np.asarray(row.values,float)
        assert t.shape==v.shape and t.ndim==1 and np.isfinite(t).all() and np.isfinite(v).all()
        assert (np.diff(t)>=0).all() and t.min()>=row.window_start
        assert row.label in [0,1]
        if row.dataset=='ExtraSensory':
            assert len(t)>=30 and np.ptp(t)>=2700 and (v>=0).all()
            assert t.max()<=row.window_end==row.label_time-20
            assert row.acquisition_duration==20
            gap_min.append(row.label_time-t.max())
        else:
            assert len(t)>=60 and np.ptp(t)>=43200 and np.isin(v,[0,1,2]).all()
            assert t.max()<row.window_end<=row.label_time
            day=pd.Timestamp(row.label_time,unit='s',tz='UTC').tz_convert('America/New_York').normalize()
            assert day.timestamp()==row.window_end
            assert (day-pd.DateOffset(days=1)).timestamp()==row.window_start
            gap_max.append(row.label_time-row.window_end)
        counts.append(len(t))
    for pid,part in windows.loc[windows.dataset=='ExtraSensory'].groupby('participant'):
        assert (np.diff(np.sort(part.label_time))>=3600).all()
    # Independently match every prepared label with original response/label rows.
    source_paths=[resolve_input(r['path']) for r in sources]
    for (dataset,pid),part in windows.groupby(['dataset','participant']):
        if dataset=='ExtraSensory':
            paths=[p for p in source_paths if p.name==pid+'.features_labels.csv.gz']
            assert len(paths)==1
            raw=pd.read_csv(paths[0],usecols=['timestamp','label:SLEEPING']).set_index('timestamp')
            actual=raw.loc[part.label_time,'label:SLEEPING'].to_numpy()
            assert np.array_equal(actual,part.label.to_numpy())
        else:
            paths=[p for p in source_paths if p.name=='Sleep_'+pid+'.json']
            assert len(paths)==1
            raw=json.loads(paths[0].read_text(encoding='utf-8-sig'))
            valid=[]
            for r in raw:
                try:rate=float(r.get('rate',float('nan')));ts=float(r['resp_time'])
                except (ValueError,TypeError,KeyError):continue
                if rate in [1,2,3,4] and np.isfinite(ts):valid.append({'rate':rate,'resp_time':ts})
            valid=sorted(valid,key=lambda r:r['resp_time'])
            earliest={}
            for r in valid:
                day=str(pd.Timestamp(r['resp_time'],unit='s',tz='UTC').tz_convert('America/New_York').date())
                if day not in earliest:earliest[day]=r
            for r in part.itertuples(index=False):
                day=str(pd.Timestamp(r.label_time,unit='s',tz='UTC').tz_convert('America/New_York').date())
                original=earliest[day]
                assert original['resp_time']==r.label_time and int(original['rate']>=3)==r.label
    result={'status':'PASS','sources_unchanged':all(source_checks),'source_file_checks':len(source_checks),
        'window_invariant_checks':len(windows),'label_source_matches':len(windows),
        'unique_windows':True,'ExtraSensory_nonoverlap':True,
        'ExtraSensory_smallest_label_minus_last_source_seconds':float(min(gap_min)),
        'StudentLife_largest_response_delay_after_previous_day_end_hours':float(max(gap_max)/3600),
        'eligible_prediction_windows_sha256':sha(INPUT/'eligible_prediction_windows.parquet'),
        'model_fits':0,'prediction_performance_inspected':False,
        'window_record_count_range':[int(min(counts)),int(max(counts))]}
    (BASE/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
