"""Inspect duplicate timestamp codes; no target values or predictions are used."""
from pathlib import Path
from portable_paths import role, resolve_input
import json, hashlib
import pandas as pd
import numpy as np

BASE=role('work_output')
INPUT=role('external_inventory_inputs')

def duplicates(t,v):
    d=pd.DataFrame({'t':t,'v':v});d=d.loc[d.t.duplicated(keep=False)]
    if not len(d):return {'duplicate_groups':0,'duplicate_records':0,'extra_records':0,'conflicting_groups':0,'conflicting_records':0,'identical_groups':0}
    g=d.groupby('t').v.agg(['size','nunique'])
    return {'duplicate_groups':len(g),'duplicate_records':len(d),'extra_records':int((g['size']-1).sum()),
        'conflicting_groups':int((g['nunique']>1).sum()),'conflicting_records':int(g.loc[g['nunique']>1,'size'].sum()),
        'identical_groups':int((g['nunique']==1).sum())}

def main():
    windows=pd.read_parquet(INPUT/'eligible_prediction_windows.parquet')
    rows=[]
    for r in windows.loc[windows.dataset=='StudentLife'].itertuples(index=False):
        rows.append({'participant':r.participant,'window_id':r.window_id,**duplicates(r.times,r.values)})
    frame=pd.DataFrame(rows)
    frame.to_csv(BASE/'studentlife_window_duplicate_audit.csv',index=False)
    prepared={'windows':len(frame),'windows_with_duplicate_timestamps':int((frame.duplicate_groups>0).sum()),
        'windows_with_conflicting_codes':int((frame.conflicting_groups>0).sum()),
        'totals':frame.drop(columns=['participant','window_id']).sum().astype(int).to_dict()}
    print(json.dumps({'prepared_windows':prepared}),flush=True)
    sources=json.loads((INPUT/'source_manifest.json').read_text(encoding='utf-8'))
    rows=[]
    for s in sources:
        path=resolve_input(s['path'])
        if not (path.name.startswith('activity_u') and path.suffix=='.csv'):continue
        d=pd.read_csv(path);d.columns=d.columns.str.strip()
        t=pd.to_numeric(d.timestamp,errors='coerce').to_numpy(float)
        v=pd.to_numeric(d['activity inference'],errors='coerce').to_numpy(float)
        valid=np.isfinite(t)&np.isin(v,[0,1,2])
        rows.append({'participant':path.stem.removeprefix('activity_'),**duplicates(t[valid],v[valid])})
    frame=pd.DataFrame(rows);frame.to_csv(BASE/'studentlife_source_duplicate_audit.csv',index=False)
    result={'scope':'timestamp/code metadata only; no label or performance based selection','prepared_windows':prepared,
        'all_sources':{'files':len(frame),'files_with_duplicates':int((frame.duplicate_groups>0).sum()),
        'files_with_conflicting_codes':int((frame.conflicting_groups>0).sum()),
        'totals':frame.drop(columns=['participant']).sum().astype(int).to_dict()}}
    (BASE/'ACTIVITY_DUPLICATE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
