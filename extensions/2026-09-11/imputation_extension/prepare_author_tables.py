"""Lossless all-setting author tables and direct old-summary equality check."""
from pathlib import Path
import json,hashlib
import pandas as pd
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'2026-09-10/reviewer_extension'
OUT=HERE/'author_tables'
OUT.mkdir(exist_ok=False)
new=pd.read_csv(HERE/'summary_01/summary.csv',float_precision='round_trip')
old=pd.read_csv(OLD/'full_summary_01/summary.csv',float_precision='round_trip')
keys=['level','name','geometry','method']
comp=new.loc[new.method.isin(['M1','M2']),old.columns].sort_values(keys).reset_index(drop=True)
old=old.sort_values(keys).reset_index(drop=True)
pd.testing.assert_frame_equal(comp,old,check_exact=True)
order=['M1','MEDIAN_RAW','LINEAR_RAW','LOCF_1','M2','LOCF_10','LOCF_30','LOCF_300','LOCF_INF']
prims=['screen_load_24h','phone_activity_load_24h','usage_load_24h','mobile_light_exposure_24h','wearable_light_exposure_24h']
labels=['Screen','Phone activity','Usage','Mobile light','Wearable light']
numeric_checks=[]
for geometry in ['contiguous_20pct','scattered_random_20pct']:
    x=new.loc[new.level.eq('primitive')&new.geometry.eq(geometry)]
    header='Primitive & Mean/mode & Median & Linear & $1\\Delta$ & $3\\Delta$ & $10\\Delta$ & $30\\Delta$ & $300\\Delta$ & $\\infty$ \\\\'
    lines=['% Generated from pinned summary_01/summary.csv; every fixed method retained.',
      r'\begin{table*}[t]',r'\centering',
      r'\caption{Post hoc '+('contiguous' if geometry=='contiguous_20pct' else 'scattered-random')+r' 20\% feature-recovery gain over no repair. Positive values indicate lower standardized absolute error.}',
      r'\label{tab:recovery_sensitivity}',r'\footnotesize',r'\setlength{\tabcolsep}{4pt}',
      r'\begin{tabular}{lrrrrrrrrr}',r'\toprule',header,r'\midrule']
    for primitive,label in zip(prims,labels):
        values=[]
        for method in order:
            r=x.loc[x.name.eq(primitive)&x.method.eq(method)].iloc[0]
            value=f'{r.median_gain:+.3f}' if r.applicable else '--'
            values.append(value)
            numeric_checks.append(dict(geometry=geometry,primitive=primitive,method=method,
                source_value=float(r.median_gain) if r.applicable else None,display=value))
        lines.append(label+' & '+' & '.join(values)+r' \\')
    lines+=[r'\bottomrule',r'\end{tabular}',
      r'\par\smallskip\parbox{0.98\textwidth}{\scriptsize Participant-first medians ($n=10$). $\Delta$ is the original 1- or 10-minute cadence; LOCF caps $1,3,10,30,300,\infty$ were fixed before this extension. The historical $3\Delta$ is not an optimum. Median and elapsed-time linear interpolation apply only to numeric raw values; light uses $\log(1+x)$ after filling. Missing anchors use the retained-day mean. Finite paired keys: screen/activity/mobile light 5,000 each; usage 4,859; wearable light 4,981.}',
      r'\end{table*}']
    (OUT/('tableIII_'+geometry+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf8')
pd.DataFrame(numeric_checks).to_csv(OUT/'table_numeric_crosscheck.csv',index=False)
support=pd.read_csv(HERE/'summary_01/support.csv',float_precision='round_trip')
support['fallback_reason_detail_collected']=~support.method.isin(['M0','M1','M2'])
for col in ['fallback_no_previous','fallback_cap_exceeded','fallback_missing_anchor']:
    support.loc[~support.fallback_reason_detail_collected,col]=float('nan')
support.to_csv(OUT/'support_with_explicit_legacy_reason_NA.csv',index=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(dict(status='complete',legacy_summary_rows_all_columns_exact=len(old),
    source_summary_sha256=sha(HERE/'summary_01/summary.csv'),numeric_display_checks=len(numeric_checks),
    outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}),indent=2),encoding='utf8')
print(f'All {len(old)} legacy summary rows exact; {len(numeric_checks)} fixed table cells exported.')
