"""Frozen post hoc extension with a full M0 gate before any recovery calculation."""
from pathlib import Path
from portable_paths import role, resolve_input, check_source
import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from recovery_core import repair, deleted_by_intervals, feature_value
from recovery_support import cell_epochs

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
BASE = WORKSPACE/'output/ictc2026-code-submission/2026-09-08'
PAPER = Path(__file__).resolve().parents[3]/'original_paper_reproduction/paper'
FRESH = role('base_fresh', BASE/'validation/paper_fresh_02')
sys.path.insert(0, str(PAPER))
from full_prepare import sha256, write_json, verify_vendor
verify_vendor()
from semantic_indicators import primitives, reliability
from semantic_indicators.contracts import PRIMITIVE_DEFINITIONS
from semantic_indicators.stress_runner import _prepare_selected_cell

DEFS = {d.name: d for d in PRIMITIVE_DEFINITIONS}
PRIMITIVES = ('screen_load_24h', 'phone_activity_load_24h', 'usage_load_24h',
              'mobile_light_exposure_24h', 'wearable_light_exposure_24h')
GEOMETRIES = ('contiguous_20pct', 'scattered_random_20pct')
KEY = ['subject_id', 'sensor_day_id', 'primitive', 'draw']


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')).encode('utf8')).hexdigest()


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False, default=str), flush=True)


def compare(a, b, context):
    exact = bool(a == b or (pd.isna(a) and pd.isna(b)))
    close = bool(np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))
    if not close:
        raise ValueError(f'{context}: {a!r} != {b!r}')
    return exact, 0.0 if pd.isna(a) and pd.isna(b) else abs(float(a)-float(b))


def pin_inputs():
    import fresh_lineage
    if fresh_lineage.enabled():
        for stage in ('prepare', 'g2', 'stress'):
            fresh_lineage.verify_stage(FRESH/stage, stage)
    files = {}
    for stage, names in {
        'prepare': ['primitives'], 'g2': ['selected_days', 'cell_replays', 'mask_intervals', 'deletion_audit'],
        'stress': ['cell_replays', 'mask_ledger', 'target_ledger']}.items():
        mf = FRESH/stage/'manifest.json'
        m = json.loads(mf.read_text(encoding='utf-8-sig'))
        if m['status'] != 'complete':
            raise ValueError(f'incomplete upstream {stage}')
        files[str(mf)] = sha256(mf)
        for name in names:
            rec = m['outputs'][name]; path = FRESH/stage/rec['path']
            actual = sha256(path)
            if actual != rec['sha256']:
                raise ValueError(f'changed upstream {path}')
            files[str(path)] = actual
    m = json.loads((FRESH/'stress/manifest.json').read_text(encoding='utf-8-sig'))
    canonical = resolve_input(m['inputs']['canonical_root'])
    for name in {DEFS[p].sensor_file for p in PRIMITIVES}:
        path = canonical/name
        actual = sha256(path)
        if actual != m['inputs']['canonical_files'][name]['sha256']:
            raise ValueError(f'changed canonical source {name}')
        files[str(path)] = actual
    return canonical, files


def read_selected(stage, name, scenario=None):
    filters = [('primitive', 'in', list(PRIMITIVES))]
    if scenario:
        filters.append(('scenario', '=', scenario))
    return pd.read_parquet(FRESH/stage/(name+'.parquet'), filters=filters)


def public_value(cell, raw_values, keep):
    definition = DEFS[cell['primitive']]
    frame = cell['raw_frame'].loc[keep].copy()
    frame[definition.value_column] = np.asarray(raw_values)[keep]
    return primitives.recompute_primitive_from_frame(
        cell['primitive'], frame, definition.sensor_file, cell['calendar'])


def preflight(output, limit):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = dict(status='running', stage='m0_preflight', started_utc=datetime.now(timezone.utc).isoformat(),
                    scope='technical_smoke' if limit else 'full', limit=limit,
                    recovery_results_observed=False, outputs={}, python=platform.python_version())
    write_json(output/'manifest.json', manifest)
    try:
        canonical, files = pin_inputs()
        manifest['input_hashes'] = files
        manifest['source_hashes'] = {p.name: sha256(p) for p in HERE.glob('*.py')}
        manifest['vendor_sources'] = verify_vendor()
        write_json(output/'manifest.json', manifest)
        selected = read_selected('g2', 'selected_days').sort_values(['subject_id','primitive','sensor_day_id'])
        import fresh_lineage
        if fresh_lineage.enabled():
            expected = fresh_lineage.validate_selection(selected, FRESH/'g2')
            if limit != expected: raise ValueError('Fresh M0 limit differs from bound topology')
        elif len(selected) != 500 or not selected.groupby(['subject_id','primitive']).size().eq(10).all():
            raise ValueError('selected topology does not match 500 original cells')
        if limit:
            selected = selected.head(limit)
        tables = pd.read_parquet(FRESH/'prepare/primitives.parquet')
        calendar_rows = {int(r.sensor_day_id): pd.Series(r._asdict()) for r in tables.itertuples(index=False)}
        contig = read_selected('g2','cell_replays','contiguous_20pct').set_index(KEY)
        random = read_selected('stress','cell_replays','scattered_random_20pct').set_index(KEY)
        audit = read_selected('g2','deletion_audit','contiguous_20pct').set_index(KEY)
        targets = read_selected('stress','target_ledger').set_index(KEY)
        mask_ledger = read_selected('stress','mask_ledger','scattered_random_20pct').set_index(KEY)
        intervals_frame = read_selected('g2','mask_intervals','contiguous_20pct')
        intervals = {k: list(zip(f.mask_start, f.mask_end)) for k,f in intervals_frame.sort_values('mask_interval_id').groupby(KEY)}
        indexes = {}; records = []; public_checks = []
        (output/'cells').mkdir()
        for index, row in enumerate(selected.itertuples(index=False)):
            cell_key = (row.subject_id, int(row.sensor_day_id), row.primitive)
            definition = DEFS[row.primitive]
            prepared = _prepare_selected_cell(row.primitive, calendar_rows[row.sensor_day_id], canonical_root=canonical, sensor_indexes=indexes)
            view = primitives.export_prepared_phase_view(prepared)
            rs = view.records
            raw = np.array([r.raw_value for r in rs], dtype=float)
            valid = np.array([r.valid for r in rs], dtype=bool)
            times = np.array([r.source_timestamp.value for r in rs], dtype=np.int64)
            starts = np.array([r.support_start.value for r in rs], dtype=np.int64)
            ends = np.array([r.support_end.value for r in rs], dtype=np.int64)
            if len(times) and (np.diff(times) <= 0).any():
                raise ValueError(f'duplicate/unordered source coordinates {cell_key}')
            raw_frame = pd.DataFrame({'subject_id': [r.subject_id for r in rs],
                                      'timestamp': [r.source_timestamp for r in rs], definition.value_column: raw})
            if definition.validity_column:
                raw_frame[definition.validity_column] = [r.raw_validity for r in rs]
            cell = dict(subject_id=row.subject_id, sensor_day_id=int(row.sensor_day_id), primitive=row.primitive,
                        family='state_ratio' if definition.operation in ('binary_load','activity_load') else 'intensity',
                        raw=raw, valid=valid, times=times, raw_frame=raw_frame,
                        calendar=reliability._calendar_row_for_replay(calendar_rows[row.sensor_day_id]),
                        original_source_hash=files[str(canonical/definition.sensor_file)],
                        source_record_digest=view.source_record_digest, masks=[])
            for draw in range(50):
                key = (*cell_key,draw); t=targets.loc[key]; cr=contig.loc[key]; rr=random.loc[key]; ml=mask_ledger.loc[key]
                if t.prepared_source_content_digest != view.source_record_digest:
                    raise ValueError(f'view source changed {key}')
                c_intervals = intervals.get(key, [])
                cd = deleted_by_intervals(starts, ends, [(a.value,b.value) for a,b in c_intervals], view.timestamp_semantics)
                if int((cd & valid).sum()) != int(t.target_deleted_count):
                    raise ValueError(f'K_valid mismatch {key}')
                if int(cd.sum()) != int(audit.loc[key].deleted_count):
                    raise ValueError(f'contiguous total count mismatch {key}')
                seed = digest(['structured-missingness-mask-v1',42,'scattered_random_20pct',row.subject_id,
                               int(row.sensor_day_id),row.primitive,draw,view.source_record_digest,
                               cr.deletion_audit_key,int(t.target_deleted_count)])
                if seed != ml.seed_sha256:
                    raise ValueError(f'random seed mismatch {key}')
                eligible = t.eligibility_status == 'eligible'
                candidates = np.flatnonzero(valid)
                if eligible:
                    rng = np.random.Generator(np.random.PCG64(int.from_bytes(bytes.fromhex(seed)[:8],'big')))
                    chosen = candidates[np.sort(rng.choice(len(candidates),size=int(t.target_deleted_count),replace=False))]
                else:
                    chosen = np.array([],dtype=int)
                r_intervals = [(rs[i].source_timestamp, rs[i].source_timestamp+pd.Timedelta(nanoseconds=1))
                               if view.timestamp_semantics == 'instant' else (rs[i].support_start,rs[i].support_end) for i in chosen]
                rd = np.zeros(len(rs),dtype=bool); rd[chosen] = True
                projected = deleted_by_intervals(starts, ends, [(a.value,b.value) for a,b in r_intervals], view.timestamp_semantics)
                if not np.array_equal(rd,projected):
                    raise ValueError(f'random interval geometry mismatch {key}')
                md = digest(['matched-record-mask-v1','scattered_random_20pct',seed,int(t.target_deleted_count),
                             [rs[i].content_digest for i in chosen],[[a.isoformat(),b.isoformat()] for a,b in r_intervals],
                             'eligible' if eligible else 'ineligible', 'eligible' if eligible else t.reason])
                if md != ml.content_digest or md != rr.mask_digest or int(rd.sum()) != int(ml.selected_record_count):
                    raise ValueError(f'random mask identity mismatch {key}')
                reference = float(cr.original_value); scale=float(cr.standardizer)
                compare(feature_value(raw[valid],definition.operation), reference, f'reference {key}')
                compare(scale, float(rr.standardizer), f'scale {key}')
                for geometry, deleted, source in [('contiguous_20pct',cd,cr),('scattered_random_20pct',rd,rr)]:
                    value = feature_value(raw[valid & ~deleted],definition.operation)
                    error = abs(value-reference)/scale
                    original_status = ('finite' if source.status == 'observed' else 'abstained_insufficient_support') if geometry=='contiguous_20pct' else source.status
                    actual_value = value if original_status=='finite' else float('nan')
                    actual_error = error if original_status=='finite' else float('nan')
                    ve,vd=compare(actual_value,float(source.masked_value),f'M0 value {key} {geometry}')
                    ee,ed=compare(actual_error,float(source.standardized_absolute_error),f'M0 error {key} {geometry}')
                    mask_hash = digest([[a.isoformat(),b.isoformat()] for a,b in c_intervals]) if geometry=='contiguous_20pct' else md
                    entry=dict(draw=draw,geometry=geometry,deleted=deleted,eligible=eligible,eligibility_reason=t.reason,
                               original_status=original_status,reference=reference,scale=scale,m0_value=actual_value,
                               m0_error=actual_error,mask_hash=mask_hash)
                    cell['masks'].append(entry)
                    records.append(dict(subject_id=row.subject_id,sensor_day_id=int(row.sensor_day_id),primitive=row.primitive,
                        draw=draw,geometry=geometry,original_status=original_status,eligible=eligible,
                        original_value=float(source.masked_value),actual_value=actual_value,value_exact=ve,value_absdiff=vd,
                        original_error=float(source.standardized_absolute_error),actual_error=actual_error,error_exact=ee,error_absdiff=ed,
                        deleted_total=int(deleted.sum()),deleted_valid=int((deleted & valid).sum()),mask_hash=mask_hash))
                    if draw==0:
                        public,q = public_value(cell,raw,~deleted)
                        pe,pdff=compare(value,public,f'public M0 {key} {geometry}')
                        public_checks.append(dict(subject_id=row.subject_id,sensor_day_id=int(row.sensor_day_id),primitive=row.primitive,
                            draw=draw,geometry=geometry,method='M0',exact=pe,absdiff=pdff,public_value=public,array_value=value,
                            public_status=q.get('status'),public_observed_epochs=q.get('observed_epochs')))
            path=output/'cells'/f'{index:04d}.pkl'
            with path.open('wb') as out:
                pickle.dump(cell,out,protocol=5)
            manifest['outputs'][str(path.relative_to(output))]=sha256(path)
            if index % 10 == 0 or index+1==len(selected):
                emit(stage='m0_preflight',cells=index+1,total=len(selected),seconds=round(time.perf_counter()-started,2))
        mf=pd.DataFrame(records); pf=pd.DataFrame(public_checks)
        mf.to_parquet(output/'m0_checks.parquet',index=False);pf.to_csv(output/'public_m0_checks.csv',index=False)
        for name in ('m0_checks.parquet','public_m0_checks.csv'):
            manifest['outputs'][name]=sha256(output/name)
        for path,hash_value in files.items():
            if sha256(resolve_input(path)) != hash_value: raise ValueError('upstream changed during preflight')
        verify_vendor()
        manifest.update(status='complete',cells=len(selected),m0_geometry_rows=len(mf),
                        m0_value_exact=int(mf.value_exact.sum()),m0_error_exact=int(mf.error_exact.sum()),
                        max_value_absdiff=float(mf.value_absdiff.max()),max_error_absdiff=float(mf.error_absdiff.max()),
                        all_numeric_checks_passed=True,public_m0_checks=len(pf),seconds=time.perf_counter()-started)
        write_json(output/'manifest.json',manifest);emit(**{k:manifest[k] for k in ('stage','status','cells','seconds')})
    except BaseException as error:
        manifest.update(status='failed',error=str(error),seconds=time.perf_counter()-started)
        write_json(output/'manifest.json',manifest);(output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8');raise


def execute(preflight_dir, output):
    m=json.loads((preflight_dir/'manifest.json').read_text(encoding='utf8'))
    if m['status']!='complete' or not m.get('all_numeric_checks_passed'):
        raise ValueError('M0 preflight must pass first')
    for name,h in m['outputs'].items():
        if sha256(preflight_dir/name)!=h: raise ValueError('preflight output changed')
    for path,h in m['input_hashes'].items():
        if sha256(resolve_input(path))!=h: raise ValueError('original input changed')
    for name,h in m['source_hashes'].items():
        if sha256(HERE/name)!=h: check_source(HERE/name,h)
    output.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    manifest=dict(status='running',stage='recovery',scope=m['scope'],post_hoc=True,
        condition='known originally-observed support; offline same-day reconstruction',
        started_utc=datetime.now(timezone.utc).isoformat(),preflight_sha256=sha256(preflight_dir/'manifest.json'),
        source_hashes={**m['source_hashes'], 'run_recovery_v2.py':sha256(Path(__file__)), 'recovery_support.py':sha256(HERE/'recovery_support.py')},input_hashes=m['input_hashes'],bootstrap_seed=20260910,bootstrap_draws=10000,
        methods=['M0','M1','M2'],cap_cadences=3,no_new_model_fits=True,outputs={})
    write_json(output/'manifest.json',manifest)
    try:
        rows=[];public_checks=[]
        cells=sorted((preflight_dir/'cells').glob('*.pkl'))
        for index,path in enumerate(cells):
            with path.open('rb') as f: cell=pickle.load(f)
            definition=DEFS[cell['primitive']]; raw=cell['raw']; valid=cell['valid']; times=cell['times']
            for entry in cell['masks']:
                deleted=entry['deleted']; targets=deleted & valid
                # Sole imputer input: source times, redacted valid retained values, target positions.
                redacted=np.where(valid & ~deleted,raw,np.nan)
                for method in ('M0','M1','M2'):
                    counts=dict(imputed_n=0,fallback_n=0,locf_n=0,unresolved_n=int(targets.sum()))
                    if not entry['eligible']:
                        value=error=float('nan'); status='structurally_unavailable'
                    elif method=='M0':
                        value=entry['m0_value'];error=entry['m0_error'];status=entry['original_status']
                    else:
                        rr=repair((times-times[0])/1e9,redacted,targets,
                            binary=definition.operation in ('binary_load','activity_load'),method=method,
                            cap_seconds=definition.cadence_minutes*60*3)
                        for k in counts: counts[k]=rr[k]
                        value=feature_value(rr['values'],definition.operation)
                        error=abs(value-entry['reference'])/entry['scale'];status=rr['status']
                        if entry['draw']==0:
                            # Fresh canonical frame, fresh public preparation; no sealed state is mutated.
                            restored=raw.copy();restored[targets]=rr['values'][targets]
                            keep=(~deleted) | (targets & np.isfinite(rr['values']))
                            public,q=public_value(cell,restored,keep)
                            pe,pdff=compare(value,public,f'public {cell["primitive"]} {method}')
                            if q.get('observed_epochs') != cell_epochs(cell, valid & keep, definition):
                                raise ValueError('public restored-support count mismatch')
                            public_checks.append(dict(subject_id=cell['subject_id'],sensor_day_id=cell['sensor_day_id'],
                                primitive=cell['primitive'],draw=0,geometry=entry['geometry'],method=method,
                                exact=pe,absdiff=pdff,public_value=public,array_value=value,
                                public_status=q.get('status'),public_observed_epochs=q.get('observed_epochs')))
                    rows.append(dict(subject_id=cell['subject_id'],sensor_day_id=cell['sensor_day_id'],primitive=cell['primitive'],
                        family=cell['family'],draw=entry['draw'],geometry=entry['geometry'],method=method,
                        eligible=entry['eligible'],eligibility_reason=entry['eligibility_reason'],method_status=status,
                        original_status=entry['original_status'],reference=entry['reference'],scale=entry['scale'],
                        no_repair_value=entry['m0_value'],no_repair_error=entry['m0_error'],repaired_value=value,
                        standardized_error=error,original_valid_n=int(valid.sum()),original_total_n=len(raw),
                        retained_observed_n=int((valid & ~deleted).sum()),deleted_total_n=int(deleted.sum()),
                        deleted_valid_n=int(targets.sum()),original_source_hash=cell['original_source_hash'],
                        source_record_digest=cell['source_record_digest'],mask_hash=entry['mask_hash'],**counts))
            if index % 10==0 or index+1==len(cells):
                emit(stage='recovery',cells=index+1,total=len(cells),rows=len(rows),seconds=round(time.perf_counter()-started,2))
        frame=pd.DataFrame(rows); checks=pd.DataFrame(public_checks)
        if len(frame)!=len(cells)*50*2*3 or frame.duplicated(KEY+['geometry','method']).any():
            raise ValueError('scheduled output topology mismatch')
        frame.to_parquet(output/'recovery_rows.parquet',index=False)
        checks.to_csv(output/'public_recovery_checks.csv',index=False)
        for path,h in m['input_hashes'].items():
            if sha256(resolve_input(path))!=h: raise ValueError('input changed during recovery')
        verify_vendor()
        manifest.update(status='complete',rows=len(frame),cells=len(cells),
            finite_rows=int(frame.method_status.eq('finite').sum()),
            public_recovery_checks=len(checks),public_exact=int(checks.exact.sum()),
            max_public_absdiff=float(checks.absdiff.max()),seconds=time.perf_counter()-started,
            outputs={p.name:sha256(p) for p in [output/'recovery_rows.parquet',output/'public_recovery_checks.csv']})
        write_json(output/'manifest.json',manifest);emit(stage='recovery',status='complete',rows=len(frame),seconds=manifest['seconds'])
    except BaseException as error:
        manifest.update(status='failed',error=str(error),seconds=time.perf_counter()-started)
        write_json(output/'manifest.json',manifest);(output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','execute']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',type=Path);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.stage=='preflight':preflight(a.output,a.limit)
    else:execute(a.preflight,a.output)
