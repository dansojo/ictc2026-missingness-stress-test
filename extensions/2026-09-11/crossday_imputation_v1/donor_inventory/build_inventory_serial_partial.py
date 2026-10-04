"""Build an input-only inventory; no imputation, fits, masks, or error calculations."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from portable_paths import role, resolve_input, mapped_pins, check_source, original_source
import json
import pickle
import platform
import time
import numpy as np
import pandas as pd
import inventory_core as core

SOURCE=Path(__file__).resolve().parent
HERE=role('work_output')
ROOT=core.WORKSPACE
BASE=role('base_inputs')
PRIOR=role('m0')
FREEZE=role('same_day_protocol')
CALENDAR_COLUMNS=['subject_id','lifelog_date','sleep_date','sensor_day_id',
    'collection_start_date','elapsed_day_index','any_source_record_observed',
    'any_measurement_support_observed','any_sensor_observed']
sha256=core.legacy.sha256


def emit(**value):
    print(json.dumps(value,ensure_ascii=False,default=str),flush=True)


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def main():
    started=time.perf_counter()
    out=HERE/'cache'
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('refusing to overwrite populated inventory cache')
    out.mkdir(exist_ok=True)
    manifest=dict(schema='crossday-raw-donor-inventory-v1',status='running',
        started_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
        numpy=np.__version__,pandas=pd.__version__,scope='input-only; no outcomes or fitting',
        timezone='naive local clock; no UTC conversion; IANA zone not encoded',
        cache_key_order=['subject_id','primitive','sensor_day_id'],partitions=[],outputs={})
    write_json(HERE/'manifest.json',manifest)
    frozen=json.loads(FREEZE.read_text(encoding='utf8'))
    pins=mapped_pins(frozen['original_input_hashes'])
    pins[str(FREEZE)]=sha256(FREEZE)
    old_manifest=PRIOR/'manifest.json'
    if sha256(old_manifest)!=frozen['old_m0_manifest_sha256']:
        raise ValueError('original M0 manifest changed')
    pins[str(old_manifest)]=sha256(old_manifest)
    for name,h in frozen['old_sources'].items():
        check_source(core.LEGACY/name,h)
        pins[str(original_source(core.LEGACY/name))]=h
    old=json.loads(old_manifest.read_text(encoding='utf8'))
    cell_paths={}
    for name,h in old['outputs'].items():
        if Path(name).parts[0]=='cells':
            path=PRIOR/name
            pins[str(path)]=h
            # Only identity is retained; masks/references/scales are not inspected.
            with path.open('rb') as stream:
                original=pickle.load(stream)
            key=(original['subject_id'],original['primitive'],original['sensor_day_id'])
            cell_paths[key]=path
    if len(cell_paths)!=500:
        raise ValueError('expected all 500 original target identities')
    prepare_manifest=BASE/'validation/paper_fresh_02/prepare/manifest.json'
    prep=json.loads(prepare_manifest.read_text(encoding='utf-8-sig'))
    for entry in prep['inputs']:
        pins[str(resolve_input(entry['path']))]=entry['sha256']
    vendor=core.legacy.verify_vendor()
    for entry in vendor:
        pins[str(BASE/'paper/full_vendor'/entry['path'])]=entry['sha256']
    pins[str(BASE/'paper/full_vendor_manifest.json')]=sha256(BASE/'paper/full_vendor_manifest.json')
    preprocessing=BASE/'validation/preprocessing/raw_run/preprocessing_run.json'
    pins[str(preprocessing)]=sha256(preprocessing)
    for path,h in pins.items():
        if sha256(Path(path))!=h:
            raise ValueError('input hash mismatch: '+path)
    manifest['input_hashes']=pins
    manifest['source_hashes']={p.name:sha256(p) for p in SOURCE.glob('*.py')}
    manifest['raw_lineage']=json.loads(preprocessing.read_text(encoding='utf8'))['input_sha256']
    write_json(HERE/'manifest.json',manifest)
    calendar=pd.read_parquet(BASE/'validation/paper_fresh_02/prepare/primitives.parquet',columns=CALENDAR_COLUMNS)
    if len(calendar)!=853 or sorted(calendar.subject_id.unique())!=[f'id{i:02d}' for i in range(1,11)]:
        raise ValueError('unexpected source calendar')
    calendar.to_parquet(HERE/'calendar.parquet',index=False)
    calendar.to_csv(HERE/'calendar.csv',index=False)
    canonical=Path(next(path for path in pins if path.endswith('mLight_canonical.parquet'))).parent
    daily=[]; checks=[]; counts=[]; slots=[]; summaries=[]; gaps=Counter(); phases=Counter(); sources=[]
    indexes={}
    for primitive in core.legacy.PRIMITIVES:
        definition=core.legacy.DEFS[primitive]
        # Clear indexes between primitive files to bound memory; use unchanged legacy wrapper.
        indexes.clear()
        for subject,rows in calendar.groupby('subject_id',sort=True):
            pool={}
            for row in rows.sort_values('sensor_day_id').itertuples(index=False):
                series=pd.Series(row._asdict())
                prepared=core.legacy._prepare_selected_cell(primitive,series,
                    canonical_root=canonical,sensor_indexes=indexes)
                view=core.legacy.primitives.export_prepared_phase_view(prepared)
                c=core.pack_view(view,core.legacy.reliability._calendar_row_for_replay(series))
                c['original_source_hash']=pins[str(canonical/definition.sensor_file)]
                key=(subject,primitive,int(row.sensor_day_id))
                pool[key]=c
                summary=core.support_summary(c)
                identity=dict(subject_id=subject,primitive=primitive,sensor_day_id=c['sensor_day_id'],
                    date=c['date'],weekday=c['weekday'],is_weekend=c['is_weekend'])
                daily.append(dict(**identity,**summary,day_start_ns=c['day_start_ns'],day_end_ns=c['day_end_ns']))
                for value,n in zip(*np.unique(np.diff(c['times'])//core.MINUTE_NS,return_counts=True)):
                    gaps[(subject,primitive,int(value))]+=int(n)
                for value,n in zip(*np.unique((c['times']-c['day_start_ns'])//core.MINUTE_NS%10,return_counts=True)):
                    phases[(subject,primitive,int(value))]+=int(n)
                if key in cell_paths:
                    path=cell_paths[key]
                    with path.open('rb') as stream:
                        original=pickle.load(stream)
                    for field in ['raw','valid','times']:
                        np.testing.assert_array_equal(c[field],original[field],strict=True)
                    pd.testing.assert_frame_equal(c['raw_frame'],original['raw_frame'],check_exact=True)
                    pd.testing.assert_frame_equal(c['calendar'],original['calendar'],check_exact=True)
                    for field in ['source_record_digest','original_source_hash']:
                        if c[field]!=original[field]:
                            raise ValueError(f'target {key} {field} differs')
                    expected_effective=original['times']-(5*core.MINUTE_NS if definition.timestamp_semantics=='interval_end' else 0)
                    np.testing.assert_array_equal(c['effective_times'],expected_effective)
                    expected_starts=original['times']-(10*core.MINUTE_NS if definition.timestamp_semantics=='interval_end' else 0)
                    np.testing.assert_array_equal(c['starts'],expected_starts)
                    np.testing.assert_array_equal(c['ends'],original['times'])
                    checks.append(dict(**identity,original_path=str(path),original_sha256=pins[str(path)],
                        raw_exact=True,valid_exact=True,times_exact=True,raw_frame_exact=True,
                        calendar_exact=True,source_record_digest_exact=True,source_hash_exact=True,
                        support_coordinates_exact=True,records=len(c['raw'])))
            values=list(pool.values())
            for key,target in pool.items():
                if key not in cell_paths:
                    continue
                identity=dict(subject_id=subject,primitive=primitive,sensor_day_id=target['sensor_day_id'],
                    date=target['date'],weekday=target['weekday'],is_weekend=target['is_weekend'])
                dc,ss=core.donor_support_counts(target,values)
                others=[c for c in values if c['day_start_ns']!=target['day_start_ns']]
                overlap=sum(core.target_overlap_count(c,target) for c in others)
                shared_source=sum(len(np.intersect1d(c['times'],target['times'])) for c in others)
                if overlap or shared_source:
                    raise ValueError('cross-date source overlap')
                dates=[c['date'] for c in others if c['valid'].any()]
                counts.append(dict(**identity,**dc,target_overlapping_donor_records=overlap,
                    shared_target_source_timestamps=shared_source,
                    valid_donor_dates_json=json.dumps(dates,separators=(',',':'))))
                slots.extend(dict(**identity,**s) for s in ss)
            path=out/f'{subject}__{primitive}.pkl'
            with path.open('wb') as stream:
                pickle.dump(pool,stream,protocol=5)
            record=dict(subject_id=subject,primitive=primitive,path=str(path.relative_to(HERE)),
                sha256=sha256(path),bytes=path.stat().st_size,calendar_dates=len(pool),
                raw_observed_dates=sum(len(c['raw'])>0 for c in values),
                valid_observed_dates=sum(c['valid'].any() for c in values),
                first_date=values[0]['date'],last_date=values[-1]['date'],
                records=sum(len(c['raw']) for c in values),valid_records=sum(int(c['valid'].sum()) for c in values))
            manifest['partitions'].append(record)
            summaries.append({**record,'valid_observed_dates':int(record['valid_observed_dates'])})
            emit(primitive=primitive,subject=subject,calendar_cells=len(daily),target_checks=len(checks),
                seconds=round(time.perf_counter()-started,1))
            write_json(HERE/'manifest.json',manifest)
        indexed=indexes[definition.sensor_file]
        f=indexed.frame
        subject_times=f[['subject_id','timestamp']]
        source=dict(primitive=primitive,sensor_file=definition.sensor_file,rows=len(f),
            timestamp_dtype=str(f.timestamp.dtype),min_timestamp=str(f.timestamp.min()),max_timestamp=str(f.timestamp.max()),
            exact_duplicate_keys=int(subject_times.duplicated().sum()),missing_timestamps=int(f.timestamp.isna().sum()),
            nonzero_seconds=int((f.timestamp.dt.second!=0).sum()),nonzero_subseconds=int((f.timestamp.dt.microsecond!=0).sum()),
            timestamp_semantics=definition.timestamp_semantics,cadence_minutes=definition.cadence_minutes,
            support_minutes=definition.support_minutes,unit=definition.unit,unit_scale=definition.unit_scale,
            value_column=definition.value_column,operation=definition.operation,
            raw_rows_not_in_calendar_cache=int(len(f)-sum(r['records'] for r in daily if r['primitive']==primitive)))
        sources.append(source)
    frames=dict(daily_support=pd.DataFrame(daily),target_exact_checks=pd.DataFrame(checks),
        donor_counts_by_target=pd.DataFrame(counts),slot30_donor_counts=pd.DataFrame(slots),
        donor_summary=pd.DataFrame(summaries),source_audit=pd.DataFrame(sources),
        observed_gap_counts=pd.DataFrame([dict(subject_id=s,primitive=p,gap_minutes=g,record_pairs=n)
            for (s,p,g),n in sorted(gaps.items())]),
        source_minute_mod10_counts=pd.DataFrame([dict(subject_id=s,primitive=p,minute_mod10=m,records=n)
            for (s,p,m),n in sorted(phases.items())]))
    if len(checks)!=500 or len(daily)!=4265 or len(slots)!=24000:
        raise ValueError('incomplete inventory topology')
    for name,frame in frames.items():
        frame.to_csv(HERE/(name+'.csv'),index=False)
        frame.to_parquet(HERE/(name+'.parquet'),index=False)
    for path,h in pins.items():
        if sha256(Path(path))!=h:
            raise ValueError('input changed during inventory: '+path)
    for name,h in manifest['source_hashes'].items():
        if sha256(SOURCE/name)!=h:
            raise ValueError('inventory code changed during build')
    core.legacy.verify_vendor()
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.perf_counter()-started,calendar_days=853,primitive_days=len(daily),
        target_cells_checked=len(checks),all_target_checks_exact=True,input_hashes_reverified=True,
        slot_minutes_descriptive=30,slot_count_rows=len(slots),new_fits=0,new_imputation_errors_computed=0)
    for path in HERE.glob('*'):
        if path.is_file() and path.name!='manifest.json':
            manifest['outputs'][path.name]=sha256(path)
    write_json(HERE/'manifest.json',manifest)
    emit(status='complete',seconds=round(manifest['seconds'],1),target_checks=len(checks))


if __name__=='__main__':
    main()
