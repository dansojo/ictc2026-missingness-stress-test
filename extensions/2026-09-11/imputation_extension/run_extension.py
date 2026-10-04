"""Append fixed new methods, regress all legacy rows, and audit public API."""
from pathlib import Path
from portable_paths import role, resolve_input, check_source
import argparse, hashlib, json, pickle, platform, sys, time, traceback
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'2026-09-10/reviewer_extension'
sys.path.insert(0,str(OLD))
import run_recovery_v2 as legacy
from recovery_core import repair,feature_value
from recovery_support import cell_epochs
from imputation_core import repair_extended

KEY=['subject_id','sensor_day_id','primitive','draw','geometry','method']
CONFIG=json.loads((HERE/'config.json').read_text(encoding='utf8'))
METHODS=CONFIG['old_methods']+CONFIG['new_methods']
NEW=CONFIG['new_methods']
PRE=role('m0', OLD/'full_m0_01')
LEGACY_RESULTS=role('legacy_results', OLD/'full_recovery_02')
PROTOCOL=role('same_day_protocol', HERE/'FROZEN_PROTOCOL.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def stamp():return datetime.now(timezone.utc).isoformat()
def same(a,b):return bool(a==b or (pd.isna(a) and pd.isna(b)))

def freeze():
    path=HERE/'FROZEN_PROTOCOL.json'
    if path.exists():raise ValueError('existing protocol freeze must not be overwritten')
    names=['DESIGN_PROPOSAL.md','MASK_GEOMETRY.json','config.json','imputation_core.py',
           'run_extension.py','summarize_extension.py','verify_extension.py',
           'test_imputation.py','test_summary.py','tests_green.txt','summary_tests_green.txt']
    files={name:sha(HERE/name) for name in names}
    m=json.loads((PRE/'manifest.json').read_text(encoding='utf8'))
    old_manifest=LEGACY_RESULTS/'manifest.json'
    write(path,dict(frozen_utc=stamp(),new_recovery_results_observed=False,
        prior_M0_M1_M2_results_known=True,protocol_files=files,
        old_m0_manifest_sha256=sha(PRE/'manifest.json'),
        old_recovery_manifest_sha256=sha(old_manifest),
        old_sources={name:sha(OLD/name) for name in ['run_recovery_v2.py','recovery_core.py','recovery_support.py']},
        original_input_hashes=m['input_hashes']))
    print('Frozen '+sha(path))

def check_pins():
    import fresh_lineage
    archive_pre=fresh_lineage.historical_role('m0', PRE)
    archive_legacy=fresh_lineage.historical_role('legacy_results', LEGACY_RESULTS)
    if fresh_lineage.enabled(): fresh_lineage.matching_inputs(PRE, legacy=LEGACY_RESULTS)
    frozen=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for name,h in frozen['protocol_files'].items():
        check_source(HERE/name,h)
    for name,h in frozen['old_sources'].items():
        check_source(OLD/name,h)
    if sha(archive_pre/'manifest.json')!=frozen['old_m0_manifest_sha256']:raise ValueError('M0 manifest changed')
    if sha(archive_legacy/'manifest.json')!=frozen['old_recovery_manifest_sha256']:raise ValueError('legacy manifest changed')
    m=json.loads((archive_pre/'manifest.json').read_text(encoding='utf8'))
    if m['status']!='complete' or m['m0_geometry_rows']!=50000:raise ValueError('full legacy M0 gate required')
    for name,h in m['outputs'].items():
        if sha(archive_pre/name)!=h:raise ValueError('M0 gate/cell changed: '+name)
    for path,h in m['input_hashes'].items():
        if sha(resolve_input(path))!=h:raise ValueError('canonical/upstream input changed')
    lm=json.loads((archive_legacy/'manifest.json').read_text(encoding='utf8'))
    for name,h in lm['outputs'].items():
        if sha(archive_legacy/name)!=h:raise ValueError('legacy result changed')
    legacy.verify_vendor()
    return frozen

def calculate(cell,entry,method):
    d=legacy.DEFS[cell['primitive']]
    raw,valid,deleted=cell['raw'],cell['valid'],entry['deleted']
    targets=valid&deleted
    redacted=np.where(valid&~deleted,raw,np.nan)
    numeric=d.operation not in ('binary_load','activity_load')
    counts=dict(imputed_n=0,fallback_n=0,locf_n=0,unresolved_n=int(targets.sum()),interpolation_n=0,
                fallback_no_previous_n=0,fallback_cap_exceeded_n=0,fallback_missing_anchor_n=0)
    applicable=numeric or method not in ('MEDIAN_RAW','LINEAR_RAW')
    values=redacted
    cap=0.0
    if not applicable:
        value=error=float('nan');status='not_applicable'
    elif not entry['eligible']:
        value=error=float('nan');status='structurally_unavailable'
    elif method=='M0':
        status=entry['original_status']
        value=feature_value(redacted,d.operation) if status=='finite' else float('nan')
        error=abs(value-entry['reference'])/entry['scale']
    else:
        t=(cell['times']-cell['times'][0])/1e9
        if method in ('M1','M2'):
            cap=d.cadence_minutes*60*3
            r=repair(t,redacted,targets,binary=not numeric,method=method,cap_seconds=cap)
        else:
            cap=(float('inf') if method=='LOCF_INF' else d.cadence_minutes*60*int(method.split('_')[1])) if method.startswith('LOCF_') else 1.0
            r=repair_extended(t,redacted,targets,binary=not numeric,
                method='LOCF' if method.startswith('LOCF_') else method,cap_seconds=cap)
        for k in counts:
            if k in r:counts[k]=r[k]
        values=r['values'];status=r['status']
        value=feature_value(values,d.operation)
        error=abs(value-entry['reference'])/entry['scale']
    row=dict(subject_id=cell['subject_id'],sensor_day_id=cell['sensor_day_id'],primitive=cell['primitive'],
        family=cell['family'],draw=entry['draw'],geometry=entry['geometry'],method=method,
        eligible=entry['eligible'],eligibility_reason=entry['eligibility_reason'],method_status=status,
        original_status=entry['original_status'],reference=entry['reference'],scale=entry['scale'],
        no_repair_value=entry['m0_value'],no_repair_error=entry['m0_error'],repaired_value=value,
        standardized_error=error,original_valid_n=int(valid.sum()),original_total_n=len(raw),
        retained_observed_n=int((valid&~deleted).sum()),deleted_total_n=int(deleted.sum()),
        deleted_valid_n=int(targets.sum()),original_source_hash=cell['original_source_hash'],
        source_record_digest=cell['source_record_digest'],mask_hash=entry['mask_hash'],**counts,
        applicable=applicable,cadence_minutes=float(d.cadence_minutes),cap_seconds=cap)
    return row,values

def run(output,limit=None):
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    mf=dict(status='running',started_utc=stamp(),post_hoc=True,
        scope='technical_smoke' if limit else 'full',limit=limit,outputs={},
        frozen_protocol_sha256=sha(PROTOCOL),methods=METHODS,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        source_config_frozen_before_new_results=True)
    write(output/'manifest.json',mf)
    writer=None
    try:
        check_pins()
        import fresh_lineage
        if fresh_lineage.enabled() and limit != fresh_lineage.matching_inputs(PRE, legacy=LEGACY_RESULTS):
            raise ValueError('Fresh same-day limit differs from bound topology')
        archived=pd.read_parquet(LEGACY_RESULTS/'recovery_rows.parquet')
        oldcols=list(archived.columns)
        old_index={tuple(row[k] for k in KEY):row for row in archived.to_dict('records')}
        del archived
        paths=sorted((PRE/'cells').glob('*.pkl'))
        if limit:paths=paths[:limit]
        n=0;legacy_exact=0;status_counts={};checks=[]
        for i,path in enumerate(paths):
            with path.open('rb') as f:c=pickle.load(f)
            cellrows=[]
            for entry in c['masks']:
                for method in METHODS:
                    row,values=calculate(c,entry,method)
                    if method in CONFIG['old_methods']:
                        original=old_index[tuple(row[k] for k in KEY)]
                        for k in oldcols:
                            if not same(row[k],original[k]):raise ValueError(f'legacy field mismatch {k}: {row[k]} vs {original[k]}, {tuple(row[k] for k in KEY)}')
                        legacy_exact+=1
                    if method in NEW and entry['draw']==0 and row['applicable'] and row['eligible']:
                        d=legacy.DEFS[c['primitive']];targets=c['valid']&entry['deleted']
                        restored=c['raw'].copy();restored[targets]=values[targets]
                        keep=(~entry['deleted'])|(targets&np.isfinite(values))
                        public,q=legacy.public_value(c,restored,keep)
                        exact,diff=legacy.compare(row['repaired_value'],public,f'public {method} {c["primitive"]}')
                        expected_epochs=cell_epochs(c,c['valid']&keep,d)
                        if expected_epochs!=q.get('observed_epochs'):raise ValueError('public unique-epoch support mismatch')
                        expected_status='observed' if np.isfinite(public) else q.get('status')
                        if q.get('status')!=expected_status:raise ValueError('public finite status mismatch')
                        checks.append(dict(**{k:row[k] for k in KEY},exact=exact,absdiff=diff,
                            array_value=row['repaired_value'],public_value=public,public_status=q.get('status'),
                            public_observed_epochs=q.get('observed_epochs'),expected_observed_epochs=expected_epochs,
                            valid_raw_records=int((c['valid']&keep).sum())))
                    status_counts[row['method_status']]=status_counts.get(row['method_status'],0)+1
                    cellrows.append(row)
            table=pa.Table.from_pandas(pd.DataFrame(cellrows),preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(output/'recovery_rows.parquet',table.schema)
            writer.write_table(table);n+=len(cellrows)
            if i%25==0 or i+1==len(paths):
                print(json.dumps(dict(cells=i+1,total=len(paths),rows=n,seconds=round(time.perf_counter()-started,2))),flush=True)
        writer.close();writer=None
        checks=pd.DataFrame(checks);checks.to_csv(output/'public_checks.csv',index=False)
        if n!=len(paths)*100*len(METHODS):raise ValueError('scheduled topology mismatch')
        check_pins()
        mf.update(status='complete',completed_utc=stamp(),cells=len(paths),rows=n,
            legacy_rows_exact_all_original_fields=legacy_exact,status_counts=status_counts,
            public_checks=len(checks),public_exact=int(checks.exact.sum()),
            max_public_absdiff=float(checks.absdiff.max()),all_original_inputs_unchanged=True,
            seconds=time.perf_counter()-started,
            outputs={p.name:sha(p) for p in [output/'recovery_rows.parquet',output/'public_checks.csv']})
        write(output/'manifest.json',mf)
        print(json.dumps({k:v for k,v in mf.items() if k!='outputs'},indent=2),flush=True)
    except BaseException as e:
        if writer is not None:writer.close()
        mf.update(status='failed',error=str(e),seconds=time.perf_counter()-started)
        write(output/'manifest.json',mf)
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','run'])
    p.add_argument('--output',type=Path);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.stage=='freeze':freeze()
    else:run(a.output,a.limit)
