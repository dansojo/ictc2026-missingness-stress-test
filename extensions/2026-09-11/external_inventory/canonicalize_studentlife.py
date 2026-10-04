"""Metadata-only timestamp cleanup for new external prediction windows.

Original prepared windows are preserved; no labels or model outcomes determine
which source records are retained. Run only for the agreed pre-performance design.
"""
from pathlib import Path
from portable_paths import role, resolve_input
import json,hashlib
import numpy as np
import pandas as pd

BASE=role('work_output')
INPUT=role('external_inventory_inputs')

def canonicalize(t,v):
    t=np.asarray(t,float);v=np.asarray(v,float)
    d=pd.DataFrame({'t':t,'v':v})
    g=d.groupby('t',sort=True).v.agg(['min','max','size'])
    conflicts=g['min']!=g['max']
    valid=g.loc[~conflicts]
    return valid.index.to_numpy(float),valid['min'].to_numpy(float),{
        'identical_extra_records_removed':int((g.loc[~conflicts,'size']-1).sum()),
        'conflicting_groups_removed':int(conflicts.sum()),
        'conflicting_records_removed':int(g.loc[conflicts,'size'].sum())}

def main():
    # Nontrivial duplicate checks, including conflict after an identical duplicate.
    t,v,a=canonicalize([1,1,2,2,2,3],[0,0,0,0,1,2])
    assert np.array_equal(t,[1,3]) and np.array_equal(v,[0,2])
    assert a=={'identical_extra_records_removed':1,'conflicting_groups_removed':1,'conflicting_records_removed':3}
    source=INPUT/'eligible_prediction_windows.parquet'
    df=pd.read_parquet(source);out=[];audit=[]
    for rr in df.to_dict('records'):
        if rr['dataset']=='ExtraSensory':out.append(rr);continue
        t,v,counts=canonicalize(rr['times'],rr['values'])
        keep=len(t)>=60 and np.ptp(t)>=43200
        audit.append({'participant':rr['participant'],'window_id':rr['window_id'],
            'original_records':len(rr['times']),'canonical_records':len(t),'retained':bool(keep),**counts})
        if keep:
            rr['times']=t;rr['values']=v;out.append(rr)
    result=pd.DataFrame(out);path=BASE/'eligible_prediction_windows_deduplicated.parquet'
    if path.exists():raise RuntimeError('Refusing to overwrite canonical windows')
    result.to_parquet(path,index=False)
    audits=pd.DataFrame(audit);audits.to_csv(BASE/'studentlife_canonicalization_audit.csv',index=False)
    summaries=[]
    for dataset,p in result.groupby('dataset'):
        classes=p.groupby('participant').label.agg(['size','sum','nunique'])
        summaries.append({'dataset':dataset,'participants':len(classes),'windows':len(p),
            'positive':int(p.label.sum()),'negative':int(len(p)-p.label.sum()),
            'both_classes':int((classes['nunique']==2).sum()),
            'min_windows':int(classes['size'].min()),'median_windows':float(classes['size'].median()),'max_windows':int(classes['size'].max())})
    manifest={'status':'COMPLETE','scope':'before model fit, metadata-only; original immutable windows retained',
        'rule':'For each StudentLife timestamp, retain one row if all valid0/1/2codes agree; remove every row if codes conflict. Reapply>=60records/span>=12hours.',
        'source_file':source.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'output_file':path.name,'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'groups':summaries,'StudentLife_windows_excluded':int((~audits.retained).sum()),
        'removals':audits[['identical_extra_records_removed','conflicting_groups_removed','conflicting_records_removed']].sum().astype(int).to_dict(),
        'labels_consulted_for_cleanup':False,'predictions_or_model_scores_consulted':False,'synthetic_duplicate_check':'passed'}
    (BASE/'CANONICALIZATION.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
