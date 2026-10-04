"""Independent row, report, method and full legacy preservation checks."""
from pathlib import Path
import argparse,json,pickle,sys,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'2026-09-10/reviewer_extension'
sys.path.insert(0,str(OLD))
from imputation_core import repair_extended
from run_extension import check_pins,CONFIG,METHODS,NEW,sha

def check(run,summary,output):
    check_pins()
    m=json.loads((run/'manifest.json').read_text(encoding='utf8'))
    s=json.loads((summary/'manifest.json').read_text(encoding='utf8'))
    if m['status']!='complete' or s['status']!='complete':raise ValueError('complete run and summary needed')
    for d,meta in [(run,m),(summary,s)]:
        for name,h in meta['outputs'].items():
            if sha(d/name)!=h:raise ValueError('result changed')
    f=pd.read_parquet(run/'recovery_rows.parquet')
    old=pd.read_parquet(OLD/'full_recovery_02/recovery_rows.parquet')
    keys=['subject_id','sensor_day_id','primitive','draw','geometry','method']
    if len(f)!=500000 or f.duplicated(keys).any():raise ValueError('grid rows/unique keys')
    if sorted(f.subject_id.unique())!=CONFIG['roster']:raise ValueError('participant roster')
    if not f.groupby(['subject_id','primitive']).size().eq(10000).all():raise ValueError('all original cells/draws/methods')
    newold=f.loc[f.method.isin(['M0','M1','M2']),old.columns].sort_values(keys).reset_index(drop=True)
    old=old.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(newold,old,check_exact=True,check_dtype=True)
    if not f.loc[f.method_status.eq('finite'),'standardized_error'].ge(0).all():raise ValueError('nonnegative errors')
    finite=f.method_status.eq('finite')
    if not np.isfinite(f.loc[finite,['repaired_value','standardized_error']]).all().all():raise ValueError('finite statuses')
    if np.isfinite(f.loc[~finite,'standardized_error']).any():raise ValueError('nonfinite status contains scored row')
    if not (f.loc[finite & ~f.method.eq('M0'),'imputed_n']==f.loc[finite & ~f.method.eq('M0'),'deleted_valid_n']).all():raise ValueError('imputed support')
    if not (f.fallback_n<=f.imputed_n).all():raise ValueError('fallback counts')
    locf=f.method.str.startswith('LOCF_')&finite
    if not (f.loc[locf,'locf_n']+f.loc[locf,'fallback_n']==f.loc[locf,'imputed_n']).all():raise ValueError('LOCF count partition')
    if not (f.loc[locf,'fallback_no_previous_n']+f.loc[locf,'fallback_cap_exceeded_n']==f.loc[locf,'fallback_n']).all():raise ValueError('LOCF fallback reasons')
    linear=f.method.eq('LINEAR_RAW')&finite
    if not (f.loc[linear,'interpolation_n']+f.loc[linear,'fallback_n']==f.loc[linear,'imputed_n']).all():raise ValueError('linear partition')
    status=f.method_status.value_counts().to_dict()
    if status!={'finite':456800,'not_applicable':40000,'structurally_unavailable':3200}:raise ValueError('status topology: '+str(status))
    unavailable=f.loc[f.method_status.eq('structurally_unavailable')]
    if not unavailable.deleted_valid_n.eq(0).all():raise ValueError('unavailable reason')
    # Independent point estimates, not a call to the reporting helper.
    summary_frame=pd.read_csv(summary/'summary.csv',float_precision='round_trip')
    p=f.pivot(index=['subject_id','sensor_day_id','primitive','family','draw','geometry'],columns='method',values='standardized_error').reset_index()
    independent_checked=0
    for row in summary_frame.itertuples(index=False):
        x=p.loc[p[row.level].eq(row.name)&p.geometry.eq(row.geometry)]
        state=row.name in ('screen_load_24h','phone_activity_load_24h','state_ratio')
        applicable=METHODS if not state else [a for a in METHODS if a not in ('MEDIAN_RAW','LINEAR_RAW')]
        if row.method not in applicable:
            if not np.isnan(row.median_gain) or row.n_participants:raise ValueError('N/A summary')
            continue
        mask=np.isfinite(x[applicable].to_numpy()).all(axis=1)
        subjects=[]
        for subject in CONFIG['roster']:
            one=x.loc[mask & x.subject_id.eq(subject)]
            if len(one):subjects.append(float(np.median(one.M0.to_numpy()-one[row.method].to_numpy())))
        estimate=float(np.median(subjects))
        if estimate!=row.median_gain:raise ValueError('independent point estimate mismatch')
        rng=np.random.default_rng(20260910)
        idx=rng.integers(0,len(subjects),size=(10000,len(subjects)))
        low,high=np.quantile(np.median(np.asarray(subjects)[idx],axis=1),[.025,.975])
        if low!=row.ci_low or high!=row.ci_high:raise ValueError('bootstrap mismatch')
        if int(mask.sum())!=row.finite_keys:raise ValueError('finite denominator mismatch')
        independent_checked+=1
    # Public audit covers fixed draw0 of every eligible/applicable cell and new method.
    expected=f.loc[f.draw.eq(0)&f.method.isin(NEW)&f.eligible&f.applicable,keys]
    audits=pd.read_csv(run/'public_checks.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),audits[keys].sort_values(keys).reset_index(drop=True),check_dtype=False)
    if not audits.exact.all() or not audits.absdiff.eq(0).all():raise ValueError('public exact audit')
    if not audits.public_status.eq('observed').all():raise ValueError('public statuses')
    if not audits.public_observed_epochs.eq(audits.expected_observed_epochs).all():raise ValueError('public coverage')
    # Independent imputer calculations: NumPy piecewise interpolant, sorted median,
    # and direct previous-index scans at fixed first/middle/last target positions.
    method_checks=0;method_exact=0;method_max_absdiff=0.0
    for path in sorted((OLD/'full_m0_01/cells').glob('*.pkl')):
        with path.open('rb') as stream:c=pickle.load(stream)
        binary=c['family']=='state_ratio'
        cadence=10 if c['primitive'] in ('usage_load_24h','mobile_light_exposure_24h') else 1
        t=(c['times']-c['times'][0])/1e9
        for e in c['masks']:
            if e['draw']!=0 or not e['eligible']:continue
            target=e['deleted']&c['valid'];ti=np.flatnonzero(target)
            redacted=np.where(c['valid']&~e['deleted'],c['raw'],np.nan)
            donor=np.flatnonzero(np.isfinite(redacted))
            if not len(donor):continue
            avg=float(np.mean(redacted[donor]));mode=float(np.sum(redacted[donor]==1)>len(donor)/2)
            for name in NEW:
                if binary and name in ('MEDIAN_RAW','LINEAR_RAW'):continue
                cap=(float('inf') if name=='LOCF_INF' else cadence*60*int(name.split('_')[1])) if name.startswith('LOCF_') else 1.0
                r=repair_extended(t,redacted,target,binary=binary,method='LOCF' if name.startswith('LOCF_') else name,cap_seconds=cap)
                if binary and not np.isin(r['values'][np.isfinite(r['values'])],[0,1]).all():raise ValueError('soft binary output')
                if name=='MEDIAN_RAW':
                    sx=sorted(map(float,redacted[donor]));mid=len(sx)//2
                    med=sx[mid] if len(sx)%2 else (sx[mid-1]+sx[mid])/2
                    ids=ti;expected_values=np.full(len(ti),med)
                elif name=='LINEAR_RAW':
                    ids=ti;expected_values=np.interp(t[ti],t[donor],redacted[donor],left=avg,right=avg)
                else:
                    ids=ti[np.unique([0,len(ti)//2,len(ti)-1])];expected_values=[]
                    for targetid in ids:
                        previous=[d for d in donor if t[d]<t[targetid]]
                        expected_values.append(float(redacted[previous[-1]]) if previous and t[targetid]-t[previous[-1]]<=cap else (mode if binary else avg))
                    expected_values=np.asarray(expected_values)
                actual=r['values'][ids]
                if not np.allclose(actual,expected_values,rtol=1e-12,atol=1e-12):raise ValueError('independent method values')
                method_checks+=len(ids);method_exact+=int((actual==expected_values).sum())
                method_max_absdiff=max(method_max_absdiff,float(np.max(np.abs(actual-expected_values))))
    result=dict(status='passed',scheduled_rows=len(f),status_counts=status,roster=CONFIG['roster'],
        legacy_rows_all_columns_exact=len(old),public_checks=len(audits),public_exact=int(audits.exact.sum()),
        public_max_absdiff=float(audits.absdiff.max()),independent_summary_rows_and_bootstrap_CIs=independent_checked,
        independent_imputer_values=method_checks,independent_imputer_exact_values=method_exact,
        independent_imputer_max_absdiff=method_max_absdiff,independent_imputer_rtol=1e-12,independent_imputer_atol=1e-12,
        caveat='Independent NumPy interpolation uses a different floating evaluation order; tolerance is not exact equality. Public primitive audits and legacy regression are exact.',
        all_frozen_inputs_unchanged=True,run_manifest_sha256=sha(run/'manifest.json'),summary_manifest_sha256=sha(summary/'manifest.json'),
        verifier_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();check(args.run,args.summary,args.output)
