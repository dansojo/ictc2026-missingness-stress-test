"""Independent artifact audit, including public status consistency requested in review."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PREF=HERE/'full_m0_01';RUN=HERE/'full_recovery_02';REPORT=HERE/'full_summary_01'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

manifests={name:json.loads((p/'manifest.json').read_text(encoding='utf8')) for name,p in [('m0',PREF),('recovery',RUN),('summary',REPORT)]}
assert all(m['status']=='complete' for m in manifests.values())
for name,h in manifests['recovery']['source_hashes'].items():assert sha(HERE/name)==h
for name,root in [('m0',PREF),('recovery',RUN),('summary',REPORT)]:
    for filename,value in manifests[name]['outputs'].items():assert sha(root/filename)==value
rows=pd.read_parquet(RUN/'recovery_rows.parquet')
m0=pd.read_parquet(PREF/'m0_checks.parquet')
keys=['subject_id','sensor_day_id','primitive','draw','geometry']
assert len(rows)==150000 and len(m0)==50000
assert not rows.duplicated(keys+['method']).any()
assert rows.groupby('subject_id').size().eq(15000).all()
assert set(rows.subject_id)=={f'id{i:02d}' for i in range(1,11)}
assert rows.groupby(keys).size().eq(3).all()
eligible=rows.eligible
assert np.array_equal(rows.method_status.eq('finite'),eligible)
assert np.array_equal(np.isfinite(rows.standardized_error),eligible)
assert rows.loc[~eligible,'method_status'].eq('structurally_unavailable').all()
assert (rows.loc[~eligible,'deleted_valid_n']==0).all()
assert np.array_equal(m0.original_status.eq('finite'),np.isfinite(m0.actual_value))
assert m0.loc[m0.geometry.eq('contiguous_20pct'),'original_status'].eq('finite').all()
assert np.array_equal(m0.loc[m0.geometry.eq('scattered_random_20pct'),'original_status'].eq('finite'),m0.loc[m0.geometry.eq('scattered_random_20pct'),'eligible'])
finite=rows.loc[eligible]
assert np.array_equal(finite.standardized_error.to_numpy(),(abs(finite.repaired_value-finite.reference)/finite.scale).to_numpy())
for column in ('reference','scale','mask_hash','original_source_hash','retained_observed_n','deleted_valid_n'):
    assert rows.groupby(keys)[column].nunique(dropna=False).eq(1).all()
filled=rows.loc[eligible & rows.method.ne('M0')]
assert np.array_equal(filled.imputed_n,filled.deleted_valid_n)
assert filled.unresolved_n.eq(0).all()
assert np.array_equal(filled.original_valid_n,filled.retained_observed_n+filled.imputed_n)
locf=filled.loc[filled.method.eq('M2')]
assert np.array_equal(locf.imputed_n,locf.locf_n+locf.fallback_n)
audit=pd.concat([pd.read_csv(PREF/'public_m0_checks.csv'),pd.read_csv(RUN/'public_recovery_checks.csv')],ignore_index=True)
assert audit.public_status.eq('observed').all()
assert np.isfinite(audit.public_value).all() and np.isfinite(audit.array_value).all()
assert (audit.public_observed_epochs>0).all()
assert np.allclose(audit.public_value,audit.array_value,rtol=1e-12,atol=1e-12)
matched=audit.merge(rows[keys+['method','retained_observed_n','imputed_n']],on=keys+['method'],validate='one_to_one')
expected_records=matched.retained_observed_n+matched.imputed_n
# Independently count unique occupied grid intervals with pandas flooring.
# The original feature uses records; the coverage metadata uses cadence bins.
cells={}
for path in sorted((PREF/'cells').glob('*.pkl')):
    with path.open('rb') as f:c=pickle.load(f)
    cells[(c['subject_id'],c['sensor_day_id'],c['primitive'])]=c
expected_epochs=[]
for row in matched.itertuples(index=False):
    c=cells[(row.subject_id,row.sensor_day_id,row.primitive)]
    cadence=10 if row.primitive in ('usage_load_24h','mobile_light_exposure_24h') else 1
    e=next(e for e in c['masks'] if e['draw']==0 and e['geometry']==row.geometry)
    include=c['valid'] & ~e['deleted'] if row.method=='M0' else c['valid']
    ts=pd.to_datetime(c['times'][include])
    if row.primitive=='usage_load_24h':ts=ts-pd.Timedelta(minutes=5)
    cal=c['calendar'].iloc[0]
    ts=ts[(ts>=cal.lifelog_date)&(ts<cal.sleep_date)]
    expected_epochs.append(ts.floor(f'{cadence}min').nunique())
assert np.array_equal(matched.public_observed_epochs,expected_epochs)
unavailable=rows.loc[~eligible & rows.method.eq('M0') & rows.geometry.eq('contiguous_20pct')].groupby('primitive').size().to_dict()
assert unavailable=={'usage_load_24h':141,'wearable_light_exposure_24h':19}
result=dict(status='passed',scheduled_rows=len(rows),eligible_finite_rows=int(eligible.sum()),
            structurally_unavailable_rows=int((~eligible).sum()),unavailable_pairs_by_primitive=unavailable,
            m0_rows=len(m0),m0_max_value_absdiff=float(m0.value_absdiff.max()),m0_max_error_absdiff=float(m0.error_absdiff.max()),
            public_audit_rows=len(audit),public_status_and_support_consistency=True,
            audits_where_raw_records_differ_from_coverage_epochs=int((expected_records.to_numpy()!=np.asarray(expected_epochs)).sum()),
            public_exact=int(audit.exact.sum()),max_public_absdiff=float(audit.absdiff.max()),
            all_original_inputs_unchanged=all(sha(Path(p))==h for p,h in manifests['m0']['input_hashes'].items()),
            summary_manifest_sha256=sha(REPORT/'manifest.json'),auditor_sha256=sha(Path(__file__)))
assert result['all_original_inputs_unchanged']
(HERE/'FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
