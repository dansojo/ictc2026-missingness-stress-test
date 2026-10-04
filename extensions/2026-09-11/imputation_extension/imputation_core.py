"""Raw-domain offline reconstruction from retained observations only."""
import numpy as np

def repair_extended(times_seconds, retained_values, targets, *, binary, method, cap_seconds):
    t=np.asarray(times_seconds,dtype=float)
    x=np.asarray(retained_values,dtype=float)
    target=np.asarray(targets,dtype=bool)
    if t.ndim!=1 or t.shape!=x.shape or t.shape!=target.shape:
        raise ValueError('coordinate/value/target shapes must agree')
    if not np.isfinite(t).all() or (np.diff(t)<=0).any():
        raise ValueError('strictly increasing finite coordinates required')
    if np.isfinite(x[target]).any():
        raise ValueError('deleted values must be redacted')
    if method not in ('MEDIAN_RAW','LINEAR_RAW','LOCF') or not cap_seconds>0:
        raise ValueError('unsupported method or cap')
    donors=np.flatnonzero(np.isfinite(x)&~target)
    tids=np.flatnonzero(target)
    if binary and not np.isin(x[donors],[0,1]).all():
        raise ValueError('binary donors must be hard states')
    if not binary and (x[donors]<0).any():
        raise ValueError('raw donors must be nonnegative')
    r=dict(values=x.copy(),status='finite',imputed_n=0,fallback_n=0,
           locf_n=0,interpolation_n=0,unresolved_n=len(tids),
           fallback_no_previous_n=0,fallback_cap_exceeded_n=0,
           fallback_missing_anchor_n=0)
    if binary and method in ('MEDIAN_RAW','LINEAR_RAW'):
        r['status']='not_applicable';return r
    if not len(donors):
        r['status']='abstained_no_retained_donor';return r
    fill=(float(np.count_nonzero(x[donors]==1)>len(donors)/2)
          if binary else float(np.mean(x[donors])))
    if method=='MEDIAN_RAW':
        r['values'][tids]=float(np.median(x[donors]))
    else:
        r['values'][tids]=fill
        prev=np.searchsorted(t[donors],t[tids],side='left')-1
        if method=='LOCF':
            candidate=donors[np.maximum(prev,0)]
            close=t[tids]-t[candidate]<=cap_seconds
            use=(prev>=0)&close
            r['values'][tids[use]]=x[candidate[use]]
            r['locf_n']=int(use.sum())
            r['fallback_n']=int((~use).sum())
            r['fallback_no_previous_n']=int((prev<0).sum())
            r['fallback_cap_exceeded_n']=int(((prev>=0)&~close).sum())
        else:
            nxt=prev+1
            use=(prev>=0)&(nxt<len(donors))
            left=donors[prev[use]];right=donors[nxt[use]]
            weight=(t[tids[use]]-t[left])/(t[right]-t[left])
            r['values'][tids[use]]=x[left]+weight*(x[right]-x[left])
            r['interpolation_n']=int(use.sum())
            r['fallback_n']=int((~use).sum())
            r['fallback_missing_anchor_n']=int((~use).sum())
    r['imputed_n']=len(tids);r['unresolved_n']=0
    return r
