"""Descriptive participant-first estimates; no new confirmatory hypothesis tests."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROSTER=[f'id{i:02d}' for i in range(1,11)]


def participant_summary(frame, roster):
    rows=[]
    for subject in roster:
        gains=frame.loc[frame.subject_id.eq(subject),'gain'].to_numpy(dtype=float)
        finite=gains[np.isfinite(gains)]
        rows.append(dict(subject_id=subject,scheduled_keys=len(gains),finite_keys=len(finite),
                         median_gain=float(np.median(finite)) if len(finite) else float('nan')))
    effects=np.array([r['median_gain'] for r in rows]);effects=effects[np.isfinite(effects)]
    if len(effects):
        rng=np.random.default_rng(20260910)
        boots=np.median(effects[rng.integers(0,len(effects),size=(10000,len(effects)))],axis=1)
        low,high=np.quantile(boots,[.025,.975])
    else:
        low=high=float('nan')
    return rows, dict(n_roster=len(roster),n_participants=len(effects),
        median_gain=float(np.median(effects)) if len(effects) else float('nan'),ci_low=float(low),ci_high=float(high),
        positive_participants=int((effects>0).sum()),zero_participants=int((effects==0).sum()),
        negative_participants=int((effects<0).sum()),finite_keys=sum(r['finite_keys'] for r in rows),
        scheduled_keys=len(frame))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_point_estimate(frame, roster, method):
    estimates=[]
    for subject in roster:
        a=frame.loc[frame.subject_id.eq(subject),['M0','M1','M2']].to_numpy()
        a=a[np.isfinite(a).all(axis=1)]
        if len(a):estimates.append(float(np.median(a[:,0]-a[:,int(method[1:])])))
    return float(np.median(estimates)) if estimates else float('nan')


def run(run_dir, output):
    manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf8'))
    if manifest['status']!='complete' or manifest['scope']!='full' or manifest['rows']!=150000:
        raise ValueError('complete full-grid recovery required')
    for name,h in manifest['outputs'].items():
        if sha(run_dir/name)!=h: raise ValueError('recovery output changed')
    output.mkdir(parents=True,exist_ok=False)
    frame=pd.read_parquet(run_dir/'recovery_rows.parquet')
    keys=['subject_id','sensor_day_id','primitive','family','draw','geometry']
    pivot=frame.pivot(index=keys,columns='method',values='standardized_error').reset_index()
    common=np.isfinite(pivot[['M0','M1','M2']]).all(axis=1)
    pivot['gain_M1']=np.where(common,pivot.M0-pivot.M1,np.nan)
    pivot['gain_M2']=np.where(common,pivot.M0-pivot.M2,np.nan)
    summaries=[];participants=[]
    for level in ('family','primitive'):
        names=(['state_ratio','intensity'] if level=='family' else sorted(frame.primitive.unique()))
        for geometry in ('contiguous_20pct','scattered_random_20pct'):
            for name in names:
                subset=pivot.loc[pivot.geometry.eq(geometry) & pivot[level].eq(name)]
                for method in ('M1','M2'):
                    x=subset[['subject_id','gain_'+method]].rename(columns={'gain_'+method:'gain'})
                    ps,ss=participant_summary(x,ROSTER)
                    labels=dict(level=level,name=name,geometry=geometry,method=method)
                    summaries.append(dict(**labels,**ss));participants.extend(dict(**labels,**p) for p in ps)
    # Residual geometry contrasts use the same finite intersection across methods and geometries.
    wide=frame.pivot(index=['subject_id','sensor_day_id','primitive','family','draw'],
                     columns=['geometry','method'],values='standardized_error')
    complete=np.isfinite(wide).all(axis=1)
    residual=[]
    for family in ('state_ratio','intensity'):
        for method in ('M0','M1','M2'):
            gain=(wide[('contiguous_20pct',method)]-wide[('scattered_random_20pct',method)]).where(complete)
            q=gain.rename('gain').reset_index();q=q.loc[q.family.eq(family)]
            ps,ss=participant_summary(q,ROSTER)
            residual.append(dict(family=family,method=method,**ss))
    support=frame.groupby(['subject_id','family','primitive','geometry','method','method_status'],dropna=False).agg(
        scheduled_rows=('draw','size'),retained_observations=('retained_observed_n','sum'),
        deleted_valid=('deleted_valid_n','sum'),deleted_total=('deleted_total_n','sum'),
        imputed=('imputed_n','sum'),fallback=('fallback_n','sum'),locf=('locf_n','sum'),
        unresolved=('unresolved_n','sum')).reset_index()
    support['fallback_fraction_of_imputed']=support.fallback/support.imputed.replace(0,np.nan)
    fallback=frame.loc[frame.method.eq('M2')].groupby(['family','geometry']).agg(
        imputed=('imputed_n','sum'),fallback=('fallback_n','sum'),locf=('locf_n','sum')).reset_index()
    fallback['fallback_fraction']=fallback.fallback/fallback.imputed
    output_frames={'summary':pd.DataFrame(summaries),'participant_effects':pd.DataFrame(participants),
        'support':support,'geometry_residual':pd.DataFrame(residual),'fallback':fallback}
    for name,f in output_frames.items():f.to_csv(output/(name+'.csv'),index=False)
    # Independently recompute family point estimates with direct per-person NumPy slicing.
    for row in summaries:
        if row['level']!='family':continue
        q=pivot.loc[pivot.family.eq(row['name']) & pivot.geometry.eq(row['geometry'])]
        independent=independent_point_estimate(q,ROSTER,row['method'])
        if not (independent==row['median_gain'] or (np.isnan(independent) and np.isnan(row['median_gain']))):
            raise ValueError('independent family median cross-check failed')
    report=['# Post hoc known-support feature-recovery results','',
        'Measured ETRI extension. Positive gain means lower standardized error than no repair. This is offline reconstruction of artificially deleted originally-valid values at known timestamps, not real-time prediction or natural-missingness recovery.','',
        'All 10 original participants, 5 primitives, 10 selected days each, 50 draws, and both original 20% geometries. Original reference/scales were retained. M1: raw mean / hard-state mode (tie 0); M2: previous retained donor within 3 cadences with M1 fallback.','',
        'The selected state/intensity families were chosen after their original positive contrasts were known. Effects are exploratory; no new confirmatory p-values or change to the original overall inconclusive decision.','',
        '| Family | Method | Contiguous error reduction | Descriptive 95% bootstrap CI | Positive / zero / negative participants | Finite / scheduled keys |',
        '|---|---|---:|---:|---|---|']
    for r in summaries:
        if r['level']=='family' and r['geometry']=='contiguous_20pct':
            report.append(f"| {r['name']} | {r['method']} | {r['median_gain']:.6f} | [{r['ci_low']:.6f}, {r['ci_high']:.6f}] | {r['positive_participants']} / {r['zero_participants']} / {r['negative_participants']} | {r['finite_keys']} / {r['scheduled_keys']} |")
    report+=['','## Support and interpretation','',
        f"Scheduled rows: {len(frame):,}; finite: {frame.method_status.eq('finite').sum():,}; structurally unavailable: {frame.method_status.eq('structurally_unavailable').sum():,}.",
        '160 original paired intensity keys deleted zero valid records: 141 usage and 19 wearable-light keys. They are retained as structurally unavailable for this paired extension in both geometries and all three methods (960 scheduled rows). Their original contiguous M0 values remain in no_repair_value and in the full M0 audit; no values were relabeled in the original experiment.',
        'The finite comparison is the common intersection of M0/M1/M2 within each key. Detailed participant/primitive scheduled counts, natural-invalid support, and imputed/fallback totals accompany this report.',
        'Fallback totals count repeated mask evaluations, not independent records or participants. A bootstrap does not increase the number of independent participants.',
        'Raw mean imputation followed by a raw arithmetic mean can be algebraically redundant; our light features instead use mean(log1p(raw lux)), while usage is a sum. Hard-state mode preserves the binary domain.',
        'No external prediction model or larger-cohort predictive validation was conducted. StudentLife and ExtraSensory results in the original paper remain feature-direction analyses.',
        '', '## Implementation checks','',
        'The full M0 gate precedes recovery. Source hashes, regenerated random mask digests and matched valid deletion counts were checked. The immutable public primitive API was separately called on draw 0 of each cell, for both geometries and each method where eligible. The full-grid fast path and public audit are compared with exact equality and rtol=1e-12/atol=1e-12 separately.',
        '', 'Mean/mode baseline reference: https://scikit-learn.org/stable/modules/impute.html . The capped elapsed-time previous-donor policy is explicitly defined here and is not claimed to be an unmodified library ffill.']
    (output/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    result=dict(status='complete',run_manifest_sha256=sha(run_dir/'manifest.json'),summary_source_sha256=sha(Path(__file__)),
        bootstrap_seed=20260910,bootstrap_draws=10000,independent_family_point_estimates_match=True,
        full_roster_retained=True,rows=len(frame),outputs={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(output_frames['summary'].loc[lambda x:x.level.eq('family')].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
