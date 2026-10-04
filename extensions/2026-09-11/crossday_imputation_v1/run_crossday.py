"""Frozen cross-date extension; preserve old results and audit raw draw-zero fills."""
from pathlib import Path
from portable_paths import role, resolve_input, check_consumed_tree
from datetime import date,datetime,timezone
import argparse,hashlib,json,pickle,platform,sys,time,traceback
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'2026-09-10/reviewer_extension'
PRE=role('m0', OLD/'full_m0_01')
PRIOR=role('same_day_results', HERE.parent/'imputation_extension/full_01')
DONORS=role('donor_inputs', HERE/'donor_inventory')
PROTOCOL=role('cross_day_protocol', HERE/'FROZEN_PROTOCOL.json')
sys.path.insert(0,str(OLD))
import run_recovery_v2 as legacy
from recovery_core import feature_value
from recovery_support import cell_epochs
from crossday_core import METHODS,MEDIAN_METHODS,prepare_archive,repair_all
KEY=['subject_id','sensor_day_id','primitive','draw','geometry','method']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def same(a,b):return bool(a==b or (pd.isna(a) and pd.isna(b)))

def check_pins():
    import fresh_lineage
    if fresh_lineage.enabled(): fresh_lineage.matching_inputs(PRE, same_day=PRIOR, donors=DONORS)
    frozen=json.loads(PROTOCOL.read_text(encoding='utf8'))
    for path,h in frozen['files'].items():
        if sha(resolve_input(path))!=h:raise ValueError('frozen file changed')
    for root,marker in [(fresh_lineage.historical_role('m0',PRE),'/reviewer_extension/full_m0_01/'),
                        (fresh_lineage.historical_role('same_day_results',PRIOR),'/imputation_extension/full_01/'),
                        (fresh_lineage.historical_role('donor_inputs',DONORS),'/crossday_imputation_v1/donor_inventory/')]:
        check_consumed_tree(root,frozen['files'],marker)
    legacy.verify_vendor()
    return frozen

def adapt_days(partition,target):
    """Only sensor observations enter archive; exclude target day and cross-boundary support."""
    donors=[]
    for d in partition.values():
        if d['subject_id']!=target['subject_id'] or d['primitive']!=target['primitive']:raise ValueError('partition identity mismatch')
        if d['day_start_ns']==target['day_start_ns']:continue
        starts,ends=d['starts'],d['ends']
        if d['timestamp_semantics']=='instant':
            overlap=(starts>=target['day_start_ns'])&(starts<target['day_end_ns'])
        else:overlap=(starts<target['day_end_ns'])&(ends>target['day_start_ns'])
        if overlap.any():raise ValueError('donor measurement support overlaps target date')
        donors.append(dict(subject_id=d['subject_id'],primitive=d['primitive'],
            day_ordinal=date.fromisoformat(d['date']).toordinal(),clock_seconds=d['local_clock_ns']/1e9,
            values=np.where(d['valid'],d['raw'],np.nan)))
    return donors

def base_row(c,e):
    valid,deleted=c['valid'],e['deleted']
    return dict(subject_id=c['subject_id'],sensor_day_id=c['sensor_day_id'],primitive=c['primitive'],family=c['family'],
        draw=e['draw'],geometry=e['geometry'],eligible=e['eligible'],eligibility_reason=e['eligibility_reason'],
        original_status=e['original_status'],reference=e['reference'],scale=e['scale'],
        no_repair_value=e['m0_value'],no_repair_error=e['m0_error'],original_valid_n=int(valid.sum()),
        original_total_n=len(valid),retained_observed_n=int((valid&~deleted).sum()),deleted_total_n=int(deleted.sum()),
        deleted_valid_n=int((valid&deleted).sum()),original_source_hash=c['original_source_hash'],
        source_record_digest=c['source_record_digest'],mask_hash=e['mask_hash'])

def run(output,limit=None):
    check_pins()
    config=json.loads((HERE/'config.json').read_text(encoding='utf8'))
    output.mkdir(parents=True,exist_ok=False);(output/'audit_raw').mkdir()
    start=time.perf_counter();writer=None
    mf=dict(status='running',started_utc=stamp(),scope='technical_smoke' if limit else 'full',
        post_review_exploratory=True,methods=list(METHODS),frozen_protocol_sha256=sha(PROTOCOL),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,outputs={})
    write(output/'manifest.json',mf)
    try:
        old=pd.read_parquet(PRIOR/'recovery_rows.parquet',filters=[('method','==','M0')])
        index={tuple(r[k] for k in KEY[:-1]):r for r in old.to_dict('records')}
        import fresh_lineage
        expected=50000
        if fresh_lineage.enabled():
            cells=fresh_lineage.matching_inputs(PRE, same_day=PRIOR, donors=DONORS)
            if limit != cells: raise ValueError('Fresh cross-day limit differs from bound topology')
            expected=cells*100
        if len(index)!=expected:raise ValueError('M0 unique topology failure')
        paths=sorted((PRE/'cells').glob('*.pkl'));paths=paths[:limit] if limit else paths
        partition=None;partition_key=None;checks=[];statuses={};n=0;exact_baselines=0
        for i,path in enumerate(paths):
            with path.open('rb') as f:c=pickle.load(f)
            group=(c['subject_id'],c['primitive'])
            if group!=partition_key:
                with (DONORS/'cache'/('__'.join(group)+'.pkl')).open('rb') as f:partition=pickle.load(f)
                partition_key=group
            target=partition[(c['subject_id'],c['primitive'],c['sensor_day_id'])]
            for field in ['raw','valid','times']:
                if not np.array_equal(c[field],target[field],equal_nan=True):raise ValueError('target cache mismatch '+field)
            for field in ['original_source_hash','source_record_digest']:
                if c[field]!=target[field]:raise ValueError('target identity mismatch '+field)
            d=legacy.DEFS[c['primitive']];binary=d.operation in ('binary_load','activity_load')
            archive=prepare_archive(*group,date.fromisoformat(target['date']).toordinal(),adapt_days(partition,target),
                binary=binary,profile_log=d.operation=='exposure_mean',**config['archive_settings'])
            clocks=target['local_clock_ns']/1e9;cellrows=[];audit={}
            for e in c['masks']:
                base=base_row(c,e);original=index[tuple(base[k] for k in KEY[:-1])]
                for k,v in base.items():
                    if not same(v,original[k]):raise ValueError(f'M0 identity mismatch {k} at {path.name}')
                exact_baselines+=1
                redacted=np.where(c['valid']&~e['deleted'],c['raw'],np.nan);targets=c['valid']&e['deleted']
                # Structural no-valid-deletion masks remain unavailable as in the original experiment.
                results=repair_all(archive,clocks,redacted,targets,cadence_seconds=d.cadence_minutes*60) if e['eligible'] else None
                for method in METHODS:
                    applicable=not(binary and method in MEDIAN_METHODS)
                    if results is None:
                        r=dict(status='structurally_unavailable' if applicable else 'not_applicable',imputed_n=0,primary_n=0,
                            fallback_n=0,fallback_slot_n=0,fallback_global_n=0,fallback_target_n=0,unresolved_n=int(targets.sum()),
                            donor_count_min=0,donor_count_max=0,donor_count_median=0.,selected_day_ordinal=None,
                            similarity_common_slots=0,similarity_distance=float('nan'),similarity_eligible_days=0,
                            target_retained_slots=0,fallback_similarity_unavailable_n=0,fallback_copy_missing_n=0,
                            copy_clock_max_lag_seconds=float('nan'))
                        values=redacted
                    else:r=results[method].copy();values=r.pop('values')
                    status=r.pop('status');value=feature_value(values,d.operation) if status=='finite' else float('nan')
                    error=abs(value-e['reference'])/e['scale'] if np.isfinite(value) else float('nan')
                    row=dict(**base,method=method,method_status=status,applicable=applicable,repaired_value=value,
                        standardized_error=error,cadence_minutes=float(d.cadence_minutes),target_date=target['date'],
                        available_donor_dates=len(archive['days']),past_donor_dates=archive['past_days'],
                        future_donor_dates=archive['future_days'],same_weektype_donor_dates=archive['same_weektype_days'],**r)
                    if e['draw']==0 and applicable and e['eligible']:
                        audit[(e['draw'],e['geometry'],method)]=dict(target_indices=np.flatnonzero(targets),filled_values=values[targets].copy())
                        restored=c['raw'].copy();restored[targets]=values[targets]
                        keep=(~e['deleted'])|(targets&np.isfinite(values))
                        public,q=legacy.public_value(c,restored,keep)
                        exact,diff=legacy.compare(value,public,'public '+method+' '+path.name)
                        epochs=cell_epochs(c,c['valid']&keep,d)
                        if epochs!=q.get('observed_epochs'):raise ValueError('public epoch support mismatch')
                        if status=='finite' and q.get('status')!='observed':raise ValueError('public finite status mismatch')
                        checks.append(dict(**{k:row[k] for k in KEY},exact=exact,absdiff=diff,array_value=value,
                            public_value=public,public_status=q.get('status'),public_observed_epochs=q.get('observed_epochs'),
                            expected_observed_epochs=epochs,valid_raw_records=int((c['valid']&keep).sum())))
                    cellrows.append(row);statuses[status]=statuses.get(status,0)+1
            frame=pd.DataFrame(cellrows);frame['selected_day_ordinal']=pd.array(frame['selected_day_ordinal'],dtype='Int64')
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(output/'recovery_rows.parquet',table.schema)
            writer.write_table(table);n+=len(cellrows)
            with (output/'audit_raw'/path.name).open('wb') as f:pickle.dump(audit,f,protocol=5)
            if i%25==0 or i+1==len(paths):print(json.dumps(dict(cells=i+1,total=len(paths),rows=n,seconds=round(time.perf_counter()-start,2))),flush=True)
        writer.close();writer=None
        checks=pd.DataFrame(checks);checks.to_csv(output/'public_checks.csv',index=False)
        if n!=len(paths)*800:raise ValueError('new method topology mismatch')
        check_pins()
        mf.update(status='complete',completed_utc=stamp(),cells=len(paths),rows=n,status_counts=statuses,
            baseline_rows_exact_all_fields=exact_baselines,public_checks=len(checks),public_exact=int(checks.exact.sum()),
            max_public_absdiff=float(checks.absdiff.max()),all_frozen_inputs_unchanged=True,seconds=time.perf_counter()-start,
            outputs={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'})
        write(output/'manifest.json',mf);print(json.dumps({k:v for k,v in mf.items() if k!='outputs'},indent=2),flush=True)
    except BaseException as e:
        if writer is not None:writer.close()
        mf.update(status='failed',error=str(e),seconds=time.perf_counter()-start);write(output/'manifest.json',mf)
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int)
    args=p.parse_args();run(args.output,args.limit)
