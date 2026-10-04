"""Design-only geometry inventory. Does not inspect raw values or error results."""
from pathlib import Path
import pickle, json, sys, hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'2026-09-10/reviewer_extension'
def q(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return dict(zip(['min','q25','median','q75','q95','max'],map(float,np.quantile(x,[0,.25,.5,.75,.95,1])))) if len(x) else {}
rows=[]
for path in sorted((OLD/'full_m0_01/cells').glob('*.pkl')):
    with path.open('rb') as f: c=pickle.load(f)
    cadence=10 if c['primitive'] in ('usage_load_24h','mobile_light_exposure_24h') else 1
    times=c['times']/1e9;valid=c['valid']
    for m in c['masks']:
        targets=m['deleted']&valid; donors=np.flatnonzero(valid&~m['deleted']); ti=np.flatnonzero(targets)
        if not len(ti):continue
        pos=np.searchsorted(times[donors],times[ti])-1
        finite=pos>=0
        gaps=(times[ti[finite]]-times[donors[pos[finite]]])/60
        rows.append(dict(primitive=c['primitive'],geometry=m['geometry'],cadence_minutes=cadence,
            n_targets=len(ti),n_without_previous=int((~finite).sum()),
            target_span_minutes=float((times[ti[-1]]-times[ti[0]])/60),
            previous_gap_minutes_median=float(np.median(gaps)) if len(gaps) else np.nan,
            previous_gap_minutes_max=float(gaps.max()) if len(gaps) else np.nan))
df=pd.DataFrame(rows);df.to_csv(HERE/'mask_geometry_metadata.csv',index=False)
out=[]
for (p,g),x in df.groupby(['primitive','geometry']):
    out.append(dict(primitive=p,geometry=g,cadence_minutes=int(x.cadence_minutes.iloc[0]),keys=len(x),
        target_span_minutes=q(x.target_span_minutes),previous_gap_max_minutes=q(x.previous_gap_minutes_max),
        n_without_previous=int(x.n_without_previous.sum()),n_targets=int(x.n_targets.sum())))
(HERE/'MASK_GEOMETRY.json').write_text(json.dumps(dict(status='complete',read_fields=['primitive','times','valid','masks.deleted','masks.geometry'],no_new_recovery_results_computed=True,groups=out),indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
