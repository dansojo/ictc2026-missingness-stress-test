"""Other-date offline raw-value reconstruction; no reference/label input."""
from datetime import date
import numpy as np

METHODS=('CD_ALL_DAY','CD_TIME_MEAN','CD_TIME_MEDIAN','CD_WEEKTYPE_MEAN',
         'CD_WEEKTYPE_MEDIAN','CD_PAST_TIME_MEAN','CD_PAST_RECENT_MEAN','CD_SIMILAR_COPY')
MEDIAN_METHODS=('CD_TIME_MEDIAN','CD_WEEKTYPE_MEDIAN')

def _profiles(clock,values,slot_seconds,profile_log):
    nslots=86400//slot_seconds
    bins=(clock//slot_seconds).astype(int)
    counts=np.bincount(bins,minlength=nslots)
    means=np.full(nslots,np.nan);medians=means.copy();similar=means.copy()
    occupied=np.flatnonzero(counts)
    totals=np.bincount(bins,weights=values,minlength=nslots)
    means[occupied]=totals[occupied]/counts[occupied]
    transformed=np.log1p(values) if profile_log else values
    transformed_totals=np.bincount(bins,weights=transformed,minlength=nslots)
    similar[occupied]=transformed_totals[occupied]/counts[occupied]
    for i in occupied:medians[i]=np.median(values[bins==i])
    return means,medians,similar,counts

def _aggregate(matrix,weights,min_days,median=False):
    finite=np.isfinite(matrix);counts=finite.sum(axis=0)
    result=np.full(matrix.shape[1],np.nan)
    good=counts>=min_days
    if median:
        for i in np.flatnonzero(good):result[i]=np.median(matrix[finite[:,i],i])
    else:
        denominator=(finite*weights[:,None]).sum(axis=0)
        numerator=(np.where(finite,matrix,0)*weights[:,None]).sum(axis=0)
        np.divide(numerator,denominator,out=result,where=good & (denominator>0))
    return result,counts

def prepare_archive(subject_id,primitive,target_day,donor_days,*,binary,profile_log,
                    slot_seconds=1800,min_days=2,recency_half_life_days=7,
                    min_similarity_slots=12,min_similarity_fraction=.5):
    if slot_seconds<=0 or 86400%slot_seconds or min_days<1 or recency_half_life_days<=0:
        raise ValueError('invalid fixed aggregation configuration')
    if not 0<min_similarity_fraction<=1 or min_similarity_slots<1:
        raise ValueError('invalid fixed similarity support')
    nslots=86400//slot_seconds;days=[];seen=set()
    for source in donor_days:
        if source['subject_id']!=subject_id or source['primitive']!=primitive or int(source['day_ordinal'])==target_day:
            continue
        ordinal=int(source['day_ordinal'])
        if ordinal in seen:raise ValueError('duplicate donor date')
        seen.add(ordinal)
        t=np.asarray(source['clock_seconds'],dtype=float);x=np.asarray(source['values'],dtype=float)
        if t.ndim!=1 or t.shape!=x.shape or not np.isfinite(t).all() or ((t<0)|(t>=86400)).any() or (np.diff(t)<=0).any():
            raise ValueError('donor clocks must strictly increase inside the local day')
        valid=np.isfinite(x);t=t[valid].copy();x=x[valid].copy()
        if (x<0).any() or (binary and not np.isin(x,[0,1]).all()):
            raise ValueError('invalid raw donor domain')
        if not len(x):continue
        means,medians,similar,counts=_profiles(t,x,slot_seconds,profile_log)
        days.append(dict(ordinal=ordinal,clock=t,values=x,mean=means,median=medians,
                         similar=similar,counts=counts,day_mean=float(np.mean(x))))
    days.sort(key=lambda d:d['ordinal'])
    ordinals=np.array([d['ordinal'] for d in days],dtype=int)
    matrix=lambda key:np.array([d[key] for d in days],dtype=float).reshape(len(days),nslots)
    mean=matrix('mean');med=matrix('median');sim=matrix('similar')
    daymean=np.array([d['day_mean'] for d in days],dtype=float).reshape(-1,1)
    past=ordinals<target_day
    weektype=(date.fromordinal(int(target_day)).weekday()>=5)
    sameweek=np.array([(date.fromordinal(int(o)).weekday()>=5)==weektype for o in ordinals],dtype=bool)
    estimates={};counts={};globals_={};global_counts={}
    for pool,keep in [('all',np.ones(len(days),dtype=bool)),('week',sameweek),('past',past),('recent',past)]:
        weights=np.exp2(-(target_day-ordinals[keep])/recency_half_life_days) if pool=='recent' else np.ones(int(keep.sum()))
        estimates[pool+'_mean'],counts[pool+'_mean']=_aggregate(mean[keep],weights,min_days)
        estimates[pool+'_median'],counts[pool+'_median']=_aggregate(med[keep],weights,min_days,median=True)
        g,gc=_aggregate(daymean[keep],weights,min_days)
        globals_[pool]=float(g[0]);global_counts[pool]=int(gc[0])
    return dict(subject_id=subject_id,primitive=primitive,target_day=int(target_day),binary=bool(binary),
                profile_log=bool(profile_log),slot_seconds=int(slot_seconds),min_days=int(min_days),
                min_similarity_slots=int(min_similarity_slots),min_similarity_fraction=float(min_similarity_fraction),
                day_ordinals=ordinals,days=days,similar_profiles=sim,estimates=estimates,counts=counts,
                global_values=globals_,global_counts=global_counts,past_days=int(past.sum()),
                future_days=int((~past).sum()),same_weektype_days=int(sameweek.sum()))

def repair_all(archive,clock_seconds,retained_values,targets,*,cadence_seconds):
    t=np.asarray(clock_seconds,dtype=float);x=np.asarray(retained_values,dtype=float);target=np.asarray(targets,dtype=bool)
    if t.ndim!=1 or x.shape!=t.shape or target.shape!=t.shape or not np.isfinite(t).all() or ((t<0)|(t>=86400)).any() or (np.diff(t)<=0).any():
        raise ValueError('target clock/value/coordinate contract violated')
    if not cadence_seconds>0:raise ValueError('positive cadence required')
    if np.isfinite(x[target]).any():raise ValueError('deleted values must be redacted')
    retained=np.isfinite(x)&~target
    if (x[retained]<0).any() or (archive['binary'] and not np.isin(x[retained],[0,1]).all()):
        raise ValueError('invalid retained raw domain')
    tids=np.flatnonzero(target);bins=(t[tids]//archive['slot_seconds']).astype(int);n=len(tids)
    fallback=float(np.mean(x[retained])) if retained.any() else float('nan')
    if archive['binary'] and np.isfinite(fallback):fallback=float(fallback>.5)
    est=archive['estimates'];counts=archive['counts'];globalval=archive['global_values'];globalcount=archive['global_counts']

    # One donor chosen only from the visible profile, never target/deleted truth.
    _,_,profile,_=_profiles(t[retained],x[retained],archive['slot_seconds'],archive['profile_log'])
    observed_slots=int(np.isfinite(profile).sum())
    common=np.isfinite(archive['similar_profiles']) & np.isfinite(profile)[None,:]
    ncommon=common.sum(axis=1)
    eligible=(ncommon>=archive['min_similarity_slots']) & (ncommon>=archive['min_similarity_fraction']*observed_slots)
    distance=np.full(len(ncommon),np.inf)
    summed=np.where(common,np.abs(archive['similar_profiles']-profile),0).sum(axis=1)
    np.divide(summed,ncommon,out=distance,where=eligible & (ncommon>0))
    candidates=np.flatnonzero(eligible)
    selected=min(candidates,key=lambda i:(distance[i],abs(int(archive['day_ordinals'][i])-archive['target_day']),int(archive['day_ordinals'][i]>archive['target_day']),int(archive['day_ordinals'][i]))) if len(candidates) else None
    copy_values=np.full(n,np.nan);copy_lags=np.full(n,np.nan)
    if selected is not None and n:
        d=archive['days'][selected];p=np.searchsorted(d['clock'],t[tids]);left=np.maximum(p-1,0);right=np.minimum(p,len(d['clock'])-1)
        dl=np.abs(d['clock'][left]-t[tids]);dr=np.abs(d['clock'][right]-t[tids])
        chosen=np.where(dl<=dr,left,right);lag=np.minimum(dl,dr);use=lag<=cadence_seconds/2
        copy_values[use]=d['values'][chosen[use]];copy_lags[use]=lag[use]

    output={}
    for method in METHODS:
        r=dict(values=x.copy(),status='finite',imputed_n=0,primary_n=0,fallback_n=0,fallback_slot_n=0,
               fallback_global_n=0,fallback_target_n=0,unresolved_n=n,donor_count_min=0,donor_count_max=0,
               donor_count_median=0.,selected_day_ordinal=None,similarity_common_slots=0,similarity_distance=float('nan'),
               similarity_eligible_days=int(eligible.sum()),target_retained_slots=observed_slots,
               fallback_similarity_unavailable_n=0,fallback_copy_missing_n=0,copy_clock_max_lag_seconds=float('nan'))
        if archive['binary'] and method in MEDIAN_METHODS:
            r['status']='not_applicable';output[method]=r;continue
        values=np.full(n,np.nan);tier=np.zeros(n,dtype=np.int8);source_counts=np.zeros(n,dtype=int)
        def add(candidate,ct,label):
            candidate=np.broadcast_to(candidate,(n,));ct=np.broadcast_to(ct,(n,))
            use=~np.isfinite(values)&np.isfinite(candidate)
            values[use]=candidate[use];tier[use]=label;source_counts[use]=ct[use]
        if method=='CD_ALL_DAY':add(globalval['all'],globalcount['all'],1)
        elif method=='CD_SIMILAR_COPY':
            add(copy_values,1,1)
            if selected is None:r['fallback_similarity_unavailable_n']=n
            else:
                r.update(selected_day_ordinal=int(archive['day_ordinals'][selected]),similarity_common_slots=int(ncommon[selected]),similarity_distance=float(distance[selected]),fallback_copy_missing_n=int((~np.isfinite(copy_values)).sum()))
                if np.isfinite(copy_lags).any():r['copy_clock_max_lag_seconds']=float(np.nanmax(copy_lags))
            add(est['all_mean'][bins],counts['all_mean'][bins],2)
        else:
            key={'CD_TIME_MEAN':'all_mean','CD_TIME_MEDIAN':'all_median','CD_WEEKTYPE_MEAN':'week_mean',
                 'CD_WEEKTYPE_MEDIAN':'week_median','CD_PAST_TIME_MEAN':'past_mean','CD_PAST_RECENT_MEAN':'recent_mean'}[method]
            add(est[key][bins],counts[key][bins],1)
            if method.startswith('CD_WEEKTYPE'):
                fallback_key='all_median' if method.endswith('MEDIAN') else 'all_mean'
                add(est[fallback_key][bins],counts[fallback_key][bins],2)
        pool='recent' if method=='CD_PAST_RECENT_MEAN' else ('past' if method=='CD_PAST_TIME_MEAN' else 'all')
        if method!='CD_ALL_DAY':add(globalval[pool],globalcount[pool],3)
        add(fallback,0,4)
        if archive['binary']:
            good=np.isfinite(values);values[good]=(values[good]>.5).astype(float)
        r['values'][tids]=values
        good=np.isfinite(values);resolved=int(good.sum())
        r.update(status='finite' if resolved==n else 'abstained_unresolved',imputed_n=resolved,primary_n=int((tier==1).sum()),
                 fallback_n=int((tier>=2).sum()),fallback_slot_n=int((tier==2).sum()),fallback_global_n=int((tier==3).sum()),
                 fallback_target_n=int((tier==4).sum()),unresolved_n=n-resolved)
        cross=good & (source_counts>0)
        if cross.any():r.update(donor_count_min=int(source_counts[cross].min()),donor_count_max=int(source_counts[cross].max()),donor_count_median=float(np.median(source_counts[cross])))
        output[method]=r
    return output
