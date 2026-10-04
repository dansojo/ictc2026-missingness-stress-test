"""Bounded fresh computation from trusted local archived inputs.

This is deliberately not a full-paper reproduction. It reuses archived cells,
masks, references, donor caches and external windows; new numerical reductions
are compared with stored results. No provider data or participant rows are saved.
"""
from pathlib import Path
import argparse,importlib,json,pickle,sys,time
from datetime import date
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support'))

import release_support as support

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'original_paper_reproduction/paper'
EXT=ROOT/'extensions/2026-09-11'
sys.dont_write_bytecode=True

def setup(module):
    for p in (PAPER,ROOT/'extensions/2026-09-10/reviewer_extension',EXT/module):
        sys.path.insert(0,str(p))
    import full_prepare
    full_prepare.verify_vendor()
    return importlib.import_module({'imputation_extension':'run_extension','crossday_imputation_v1':'run_crossday','external_prediction':'prediction_core'}[module])

def selected_cells(inputs,base,limit):
    old=base.parent/'2026-09-10/reviewer_extension'
    frozen=inputs.json(base/'imputation_extension/FROZEN_PROTOCOL.json')
    pre=old/'full_m0_01'
    manifest=inputs.json(pre/'manifest.json',frozen['old_m0_manifest_sha256'])
    if manifest['status']!='complete':raise ValueError('Incomplete archived cell preparation')
    cells=[];seen=set()
    # One first available cell per primitive gives bounded operator coverage.
    for rel,h in manifest['outputs'].items():
        if not rel.replace('\\','/').startswith('cells/') or not rel.endswith('.pkl'):continue
        path=inputs.output(pre,rel,manifest)
        with path.open('rb') as stream:cell=pickle.load(stream)
        if cell['primitive'] in seen:continue
        seen.add(cell['primitive']);cells.append(cell)
        if len(cells)==limit:break
    if len(cells)!=limit:raise ValueError('Requested cell count unavailable')
    return cells

def archived_rows(inputs,folder):
    mf=inputs.json(folder/'manifest.json')
    if mf['status']!='complete':raise ValueError('Archived run incomplete')
    path=inputs.output(folder,'recovery_rows.parquet',mf)
    return pd.read_parquet(path)

def compare_row(row,reference,stats):
    if row['method_status']!=reference.method_status:raise AssertionError('Method status differs')
    for name in ('repaired_value','standardized_error'):
        result=support.compare_scalar(row[name],getattr(reference,name))
        stats['scalar_checks']+=1;stats['exact_scalar_checks']+=int(result['exact'])
        stats['max_absdiff']=max(stats['max_absdiff'],result['absdiff'])
    stats['rows']+=1

def recovery(inputs,module,limit):
    base=inputs.hub/'90_ARCHIVE/output/ictc2026-final-preparation/2026-09-11'
    same_freeze=inputs.json(base/'imputation_extension/FROZEN_PROTOCOL.json')
    source_pin_checks=support.verify_selected_pins(EXT/'imputation_extension',same_freeze['protocol_files'])
    source_pin_checks+=support.verify_selected_pins(ROOT/'extensions/2026-09-10/reviewer_extension',same_freeze['old_sources'])
    if module=='crossday_imputation_v1':
        cross_freeze=inputs.json(base/module/'FROZEN_PROTOCOL.json')
        token='/crossday_imputation_v1/'
        pins={k.replace('\\','/').split(token,1)[1]:v for k,v in cross_freeze['files'].items() if token in k.replace('\\','/') and '/' not in k.replace('\\','/').split(token,1)[1]}
        source_pin_checks+=support.verify_selected_pins(EXT/module,pins)
    mod=setup(module);cells=selected_cells(inputs,base,limit)
    old=archived_rows(inputs,base/module/'full_01')
    keys=['subject_id','sensor_day_id','primitive','draw','geometry','method']
    index={tuple(getattr(r,k) for k in keys):r for r in old.itertuples(index=False)}
    stats=dict(cells=len(cells),rows=0,scalar_checks=0,exact_scalar_checks=0,max_absdiff=0.,public_checks=0,historical_source_pin_checks=source_pin_checks)
    donor_manifest=None
    if module=='crossday_imputation_v1':
        frozen=inputs.json(base/module/'FROZEN_PROTOCOL.json')
        suffix='donor_inventory/manifest.json'
        hashes=[v for k,v in frozen['files'].items() if k.replace('\\','/').endswith(suffix)]
        if len(hashes)!=1:raise ValueError('Donor manifest pin missing or ambiguous')
        donor_manifest=inputs.json(base/module/suffix,hashes[0])
    for cell in cells:
        if module=='crossday_imputation_v1':
            part=next(p for p in donor_manifest['partitions'] if p['subject_id']==cell['subject_id'] and p['primitive']==cell['primitive'])
            path=inputs.check(support.checked_file(base/module/'donor_inventory',part['path'],part['sha256']),part['sha256'])
            with path.open('rb') as stream:partition=pickle.load(stream)
            target=partition[(cell['subject_id'],cell['primitive'],cell['sensor_day_id'])]
            for field in ('raw','valid','times'):np.testing.assert_array_equal(cell[field],target[field])
            definition=mod.legacy.DEFS[cell['primitive']]
            config=json.loads((EXT/module/'config.json').read_text(encoding='utf8'))
            archive=mod.prepare_archive(cell['subject_id'],cell['primitive'],date.fromisoformat(target['date']).toordinal(),mod.adapt_days(partition,target),binary=definition.operation in ('binary_load','activity_load'),profile_log=definition.operation=='exposure_mean',**config['archive_settings'])
        for entry in cell['masks']:
            if module=='imputation_extension':
                rows=[mod.calculate(cell,entry,method) for method in mod.METHODS]
            else:
                redacted=np.where(cell['valid']&~entry['deleted'],cell['raw'],np.nan)
                targets=cell['valid']&entry['deleted']
                values=mod.repair_all(archive,target['local_clock_ns']/1e9,redacted,targets,cadence_seconds=definition.cadence_minutes*60) if entry['eligible'] else None
                rows=[]
                for method in mod.METHODS:
                    applicable=not(definition.operation in ('binary_load','activity_load') and method in mod.MEDIAN_METHODS)
                    if values is None:
                        status='structurally_unavailable' if applicable else 'not_applicable';filled=redacted
                    else:status=values[method]['status'];filled=values[method]['values']
                    value=mod.feature_value(filled,definition.operation) if status=='finite' else float('nan')
                    err=abs(value-entry['reference'])/entry['scale'] if np.isfinite(value) else float('nan')
                    row=dict(subject_id=cell['subject_id'],sensor_day_id=cell['sensor_day_id'],primitive=cell['primitive'],draw=entry['draw'],geometry=entry['geometry'],method=method,method_status=status,repaired_value=value,standardized_error=err)
                    rows.append((row,filled))
            for row,filled in rows:
                compare_row(row,index[tuple(row[k] for k in keys)],stats)
                if entry['draw']==0 and row['method_status']=='finite':
                    targets=cell['valid']&entry['deleted'];restored=cell['raw'].copy();restored[targets]=filled[targets]
                    keep=(~entry['deleted'])|(targets&np.isfinite(filled))
                    actual,_=mod.legacy.public_value(cell,restored,keep)
                    support.compare_scalar(row['repaired_value'],actual);stats['public_checks']+=1
    return stats

def external(inputs,limit):
    base=inputs.hub/'90_ARCHIVE/output/ictc2026-final-preparation/2026-09-11'
    protocol=inputs.json(base/'external_prediction/FROZEN_PROTOCOL.json')
    source_pin_checks=support.verify_selected_pins(EXT/'external_prediction',protocol['sources'])
    c=setup('external_prediction')
    path=inputs.check(base/'external_inventory/eligible_prediction_windows_deduplicated.parquet',protocol['windows_sha256'])
    windows=pd.read_parquet(path).sort_values(['dataset','participant','label_time','window_id']).reset_index(drop=True)
    archived=base/'external_prediction/run_01'
    manifest=inputs.json(archived/'manifest.json')
    observed=pd.read_parquet(inputs.output(archived,'observed_predictions.parquet',manifest))
    masked=pd.read_parquet(inputs.output(archived,'masked_predictions.parquet',manifest))
    expected_obs=support.prediction_index(observed)
    expected_mask=support.prediction_index(masked,masked=True)
    stats=dict(new_fits=0,windows_scored=0,paired_masks=0,prediction_checks=0,exact_prediction_checks=0,max_absdiff=0.,predictions_finite=True,historical_source_pin_checks=source_pin_checks,scope='bounded held-out participants per dataset; all outer-training windows; 50 original mask draws')
    # Full training pools preserve weighting and estimand; only held-out folds are bounded.
    for dataset,part in windows.groupby('dataset',sort=True):
        part=part.reset_index(drop=True);x=np.vstack([c.features(dataset,v) for v in part['values']]);y=part.label.to_numpy();g=part.participant.to_numpy()
        for person in sorted(np.unique(g))[:limit]:
            fold=c.fit_fold(x,y,g,person);stats['new_fits']+=1
            before=c.fingerprint(fold)
            for idx in fold['test_indices']:
                row=part.iloc[idx];v=np.asarray(row['values']);preds=[];expected=[]
                result=support.compare_scalar(c.predict(fold,x[idx:idx+1])[0],expected_obs[(dataset,row.participant,row.window_id)])
                stats['prediction_checks']+=1;stats['exact_prediction_checks']+=int(result['exact']);stats['max_absdiff']=max(stats['max_absdiff'],result['absdiff'])
                for draw in range(protocol['config']['draws']):
                    masks=c.mask_pair(len(v),(dataset,row.participant,row.window_id,draw))
                    if len(masks[0])!=len(masks[1]):raise AssertionError('Mask count mismatch')
                    stats['paired_masks']+=1
                    for geometry,mask in zip(('scattered_random_20pct','contiguous_20pct'),masks):
                        keep=np.ones(len(v),bool);keep[mask]=False;preds.append(c.features(dataset,v[keep]))
                        expected.append(expected_mask[(dataset,row.participant,row.window_id,draw,geometry)])
                p=c.predict(fold,np.vstack(preds))
                if not np.isfinite(p).all():raise AssertionError('Nonfinite prediction')
                for actual,ref in zip(p,expected):
                    result=support.compare_scalar(actual,ref)
                    stats['prediction_checks']+=1;stats['exact_prediction_checks']+=int(result['exact']);stats['max_absdiff']=max(stats['max_absdiff'],result['absdiff'])
                stats['windows_scored']+=1
            if c.fingerprint(fold)!=before:raise AssertionError('Scoring changed model')
    return stats

def main():
    if sys.flags.optimize:
        raise RuntimeError('Assertions must remain enabled; do not use -O or PYTHONOPTIMIZE')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module',choices=['imputation_extension','crossday_imputation_v1','external_prediction'])
    parser.add_argument('--hub',required=True,type=Path,help='Trusted local author archive; never downloaded')
    parser.add_argument('--output',required=True,type=Path,help='New private directory outside inputs and release')
    parser.add_argument('--limit',type=int,default=1)
    args=parser.parse_args()
    maximum=2 if args.module=='external_prediction' else 5
    if not 1<=args.limit<=maximum:parser.error(f'limit must be 1..{maximum}; this command cannot launch a full study')
    inputs=support.Inputs(args.hub);out=support.fresh_output(args.output,[inputs.hub],ROOT)
    started=time.perf_counter();report=dict(status='running',scope='limited_fresh_computation_from_archived_inputs',module=args.module,full_reproduction=False,strict_full_comparison_performed=False,strict_full_comparison_passed=False,source_files_verified=support.verify_sources(ROOT))
    try:
        report['checks']=external(inputs,args.limit) if args.module=='external_prediction' else recovery(inputs,args.module,args.limit)
        report['input_files_unchanged']=inputs.finish();report['status']='passed'
        support.verify_sources(ROOT)
    except BaseException as error:
        report.update(support.failure_summary(error));raise
    finally:
        report['seconds']=time.perf_counter()-started
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(json.dumps(report,indent=2),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
