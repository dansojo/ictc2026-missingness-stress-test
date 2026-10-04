"""All fixed methods, all primitive outcomes and common-support denominators."""
from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'2026-09-10/reviewer_extension'
sys.path.insert(0,str(OLD))
from summarize_recovery import participant_summary,ROSTER
C=json.loads((HERE/'config.json').read_text(encoding='utf8'))
METHODS=C['old_methods']+C['new_methods']
ORDER=['M1','MEDIAN_RAW','LINEAR_RAW','LOCF_1','M2','LOCF_10','LOCF_30','LOCF_300','LOCF_INF']
PRIMITIVES=['screen_load_24h','phone_activity_load_24h','usage_load_24h','mobile_light_exposure_24h','wearable_light_exposure_24h']
KEYS=['subject_id','sensor_day_id','primitive','family','draw','geometry']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def common_gains(frame, methods):
    x=frame.copy();ok=np.isfinite(x[methods]).all(axis=1)
    x['common_finite']=ok
    for method in methods:
        if method!='M0':x['gain_'+method]=np.where(ok,x.M0-x[method],np.nan)
    return x

def run(run_dir,output):
    m=json.loads((run_dir/'manifest.json').read_text(encoding='utf8'))
    if m['status']!='complete' or m['scope']!='full' or m['rows']!=500000:raise ValueError('full fixed run required')
    for n,h in m['outputs'].items():
        if sha(run_dir/n)!=h:raise ValueError('run outputs changed')
    output.mkdir(parents=True,exist_ok=False)
    f=pd.read_parquet(run_dir/'recovery_rows.parquet')
    pivot=f.pivot(index=KEYS,columns='method',values='standardized_error').reset_index()
    parts=[]
    for p,x in pivot.groupby('primitive',sort=True):
        applicable=METHODS if p not in PRIMITIVES[:2] else [a for a in METHODS if a not in ('MEDIAN_RAW','LINEAR_RAW')]
        parts.append(common_gains(x,applicable))
    gains=pd.concat(parts,ignore_index=True)
    rows=[];participants=[]
    for level,names in [('primitive',PRIMITIVES),('family',['state_ratio','intensity'])]:
        for name in names:
            for geometry in C['geometries']:
                x=gains.loc[gains[level].eq(name)&gains.geometry.eq(geometry)]
                for method in ORDER:
                    applicable=not (method in ('MEDIAN_RAW','LINEAR_RAW') and name in [*PRIMITIVES[:2],'state_ratio'])
                    z=x[['subject_id','gain_'+method]].rename(columns={'gain_'+method:'gain'})
                    pr,s=participant_summary(z,ROSTER)
                    labels=dict(level=level,name=name,geometry=geometry,method=method,applicable=applicable)
                    rows.append(dict(**labels,**s));participants.extend(dict(**labels,**r) for r in pr)
    summary=pd.DataFrame(rows)
    support=f.groupby(['subject_id','family','primitive','geometry','method','applicable','method_status'],dropna=False).agg(
        scheduled_rows=('draw','size'),retained_observations=('retained_observed_n','sum'),
        original_valid=('original_valid_n','sum'),original_total=('original_total_n','sum'),
        deleted_valid=('deleted_valid_n','sum'),deleted_total=('deleted_total_n','sum'),
        imputed=('imputed_n','sum'),fallback=('fallback_n','sum'),locf=('locf_n','sum'),
        interpolated=('interpolation_n','sum'),unresolved=('unresolved_n','sum'),
        fallback_no_previous=('fallback_no_previous_n','sum'),
        fallback_cap_exceeded=('fallback_cap_exceeded_n','sum'),
        fallback_missing_anchor=('fallback_missing_anchor_n','sum')).reset_index()
    support['fallback_fraction_of_imputed']=support.fallback/support.imputed.replace(0,np.nan)
    fallback=[]
    for level in ('primitive','family'):
        z=f.groupby([level,'geometry','method']).agg(imputed=('imputed_n','sum'),fallback=('fallback_n','sum'),
            locf=('locf_n','sum'),interpolated=('interpolation_n','sum'),unresolved=('unresolved_n','sum'),
            scheduled_rows=('draw','size')).reset_index().rename(columns={level:'name'})
        z['level']=level;z['fallback_fraction']=z.fallback/z.imputed.replace(0,np.nan);fallback.append(z)
    # Geometry contrast: all applicable methods and both geometries share support.
    wide=f.pivot(index=['subject_id','sensor_day_id','primitive','family','draw'],
                 columns=['geometry','method'],values='standardized_error')
    residual_rows=[];residual_participants=[]
    for level,names in [('primitive',PRIMITIVES),('family',['state_ratio','intensity'])]:
        for name in names:
            x=wide.loc[wide.index.get_level_values(level)==name]
            methods=METHODS if name not in [*PRIMITIVES[:2],'state_ratio'] else [a for a in METHODS if a not in ('MEDIAN_RAW','LINEAR_RAW')]
            columns=[(g,a) for g in C['geometries'] for a in methods]
            ok=np.isfinite(x[columns]).all(axis=1)
            for method in METHODS:
                applicable=method in methods
                gain=(x[('contiguous_20pct',method)]-x[('scattered_random_20pct',method)]).where(ok & applicable)
                z=gain.rename('gain').reset_index()[['subject_id','gain']]
                pr,s=participant_summary(z,ROSTER)
                labels=dict(level=level,name=name,method=method,applicable=applicable)
                residual_rows.append(dict(**labels,**s));residual_participants.extend(dict(**labels,**r) for r in pr)
    outputs=dict(summary=summary,participant_effects=pd.DataFrame(participants),support=support,
        fallback=pd.concat(fallback,ignore_index=True),geometry_residual=pd.DataFrame(residual_rows),
        geometry_participant_effects=pd.DataFrame(residual_participants))
    for name,x in outputs.items():x.to_csv(output/(name+'.csv'),index=False)
    for geometry in C['geometries']:
        z=summary.loc[summary.level.eq('primitive')&summary.geometry.eq(geometry)]
        z.to_csv(output/('candidate_tableIII_'+geometry+'_long.csv'),index=False)
        table=z.pivot(index='name',columns='method',values='median_gain').reindex(index=PRIMITIVES,columns=ORDER)
        table.to_csv(output/('candidate_tableIII_'+geometry+'.csv'))
    # Numeric-only cap300 exceeds the24h day at10min and should equal unrestricted.
    subset=f.loc[f.primitive.isin(['usage_load_24h','mobile_light_exposure_24h']) & f.method.isin(['LOCF_300','LOCF_INF'])]
    check=subset.pivot(index=KEYS,columns='method',values='repaired_value')
    match=(check.LOCF_300==check.LOCF_INF)|(check.LOCF_300.isna()&check.LOCF_INF.isna())
    if not match.all():raise ValueError('within-day redundant cap check failed')
    report=['# Exploratory imputation sensitivity — all fixed settings','',
        'Positive gain means reduction in standardized absolute feature error relative to M0. No new confirmatory p-values. Same10participants,500cells,50draws,two20%geometries. All masks/reference/scales frozen. Added methods never see deleted values or labels.','',
        'All light filling is raw-lux-domain followed by mean(log1p). Numeric median/linear are explicitly N/A for binary states. Linear uses elapsed time and two real retained anchors, no extrapolation and raw-mean endpoint fallback; all methods are offline known-support reconstruction. LOCF uses real previous donors and rawmean/hardmode fallback, without imputed-donor chains. Cap3 is historical, not an optimum.','',
        '## Per-primitive contiguous gain','',
        '| Primitive | '+ ' | '.join(ORDER)+' |','|---|'+'---:|'*len(ORDER)]
    z=summary.loc[summary.level.eq('primitive')&summary.geometry.eq('contiguous_20pct')]
    for p in PRIMITIVES:
        cells=[]
        for method in ORDER:
            r=z.loc[z.name.eq(p)&z.method.eq(method)].iloc[0]
            cells.append(f'{r.median_gain:+.6f}' if r.applicable else 'N/A')
        report.append('| '+p+' | '+' | '.join(cells)+' |')
    report+=['','## Counts and full evidence','',
        f"Scheduled={len(f):,}; status counts={f.method_status.value_counts().to_dict()}.",
        'Numeric-inapplicable state rows remain in support output. Structurally unavailable paired keys have zero valid deletion:141usage and19wearablelight, propagated unchanged. Summary support is the finite intersection of every applicable method per primitive/key; row counts do not increase independent n=10.',
        'summary.csv reports every primitive and pooled family for both geometries, descriptive participant-bootstrap intervals,positive/zero/negative participants and denominators. participant_effects.csv provides all10participants including nonfinite cases. geometry_residual.csv reports contiguous-minus-random error on common paired support for every method. support.csv and fallback.csv retain all status/count/fallback outcomes.',
        'cap300 and unlimited match in the10-minute sensors because50h exceeds the single-day donor window. This predeclared redundancy is retained.',
        'The intensity pooled median must not be read as every intensity primitive improving. The full primitive table is the primary interpretation aid.',
        '','## Primary method documentation','',
        '- https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html (numeric mean/median, most-frequent categories).',
        '- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html (time-aware versus equally spaced linear interpolation distinction).',
        '- https://numpy.org/doc/stable/reference/generated/numpy.interp.html (piecewise linear interpolation with increasing coordinates; our explicitly defined endpoint fallback differs from endpoint-value defaults).']
    (output/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    mf=dict(status='complete',run_manifest_sha256=sha(run_dir/'manifest.json'),summary_source_sha256=sha(Path(__file__)),
        bootstrap_seed=20260910,bootstrap_draws=10000,cap300_infinity_10minute_rows=len(match),
        cap300_infinity_exact=True,outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(mf,indent=2),encoding='utf8')
    print(z[['name','method','median_gain','ci_low','ci_high','positive_participants','finite_keys']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.run,a.output)
