"""Run the immutable external protocol; each model is fitted once per participant."""
from pathlib import Path
from portable_paths import role, resolve_input, check_source
from datetime import datetime, timezone
import argparse
import hashlib
import json
import pickle
import time
import traceback
import numpy as np
import pandas as pd
import prediction_core as c

HERE=Path(__file__).resolve().parent
GEOMETRIES=('scattered_random_20pct','contiguous_20pct')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=True)+'\n',encoding='utf8')

def emit(**obj):print(json.dumps(obj,ensure_ascii=False),flush=True)

def validate_windows(windows):
    checks=[]
    if windows.duplicated(['dataset','participant','window_id']).any():raise ValueError('Duplicate evaluated window')
    if not windows.label.isin([0,1]).all():raise ValueError('Invalid binary labels')
    for dataset,part in windows.groupby('dataset',sort=True):
        if part.participant.nunique()<=10 or len(part)<100:raise ValueError('Fixed feasibility gate not met')
        for pid,person in part.groupby('participant',sort=True):
            if dataset=='ExtraSensory' and (np.diff(person.label_time.sort_values())<3600).any():
                raise ValueError('Overlapping ExtraSensory predictor windows')
            for row in person.itertuples(index=False):
                t=np.asarray(row.times,dtype=float);v=np.asarray(row.values,dtype=float)
                if len(t)!=len(v) or not np.isfinite(t).all() or (np.diff(t)<=0).any():
                    raise ValueError('Misaligned or unordered sensor records')
                if not (t>=row.window_start).all():raise ValueError('Sensor predates stated window')
                if dataset=='StudentLife':
                    if not (t<row.window_end).all() or not row.window_end<=row.label_time:
                        raise ValueError('StudentLife sensor enters response day/future')
                    if len(t)<60 or t[-1]-t[0]<43200:raise ValueError('Insufficient StudentLife support')
                else:
                    if not (t<=row.window_end).all() or not (t+20<=row.label_time).all():
                        raise ValueError('Acquisition overlaps target')
                    if len(t)<30 or t[-1]-t[0]<2700:raise ValueError('Insufficient ExtraSensory support')
                c.features(dataset,v)
            checks.append({'dataset':dataset,'participant':pid,'windows':len(person),
                'positive':int(person.label.sum()),'negative':int((1-person.label).sum()),
                'both_classes':person.label.nunique()==2,
                'time_bounds_passed':True,'duplicates_in_windows':int(sum(len(t)-len(np.unique(t)) for t in person.times))})
    return pd.DataFrame(checks)

def prepare_dataset_features(dataset,part,directory,draws):
    rows=[]
    observed=np.vstack([c.features(dataset,v) for v in part['values']])
    if observed.shape!=(len(part),len(c.FEATURE_NAMES[dataset])):
        raise ValueError('Predictor matrix differs from declared sensor-only feature schema')
    perturbed=np.empty((len(part),draws,2,observed.shape[1]),dtype=np.float64)
    for j,r in enumerate(part.itertuples(index=False)):
        times=np.asarray(r.times,dtype=np.float64);values=np.asarray(r.values,dtype=np.float64)
        n=len(values)
        for draw in range(draws):
            masks=c.mask_pair(n,(dataset,r.participant,r.window_id,draw))
            assert len(masks[0])==len(masks[1])==int(n*.2)
            for gi,(geometry,deleted) in enumerate(zip(GEOMETRIES,masks)):
                keep=np.ones(n,dtype=bool);keep[deleted]=False
                perturbed[j,draw,gi]=c.features(dataset,values[keep])
                rows.append({'dataset':dataset,'participant':r.participant,'window_id':r.window_id,
                    'window_index':j,'draw':draw,'geometry':geometry,'original_records':n,
                    'deleted_records':len(deleted),'retained_records':int(keep.sum()),
                    'first_deleted_index':int(deleted[0]),'last_deleted_index':int(deleted[-1]),
                    'deletion_span_seconds':float(times[deleted[-1]]-times[deleted[0]]),
                    'mask_sha256':hashlib.sha256(deleted.astype('<i8').tobytes()).hexdigest()})
        if (j+1)%500==0:emit(stage='features',dataset=dataset,windows=j+1,total=len(part))
    np.save(directory/'observed_features.npy',observed,allow_pickle=False)
    np.save(directory/'masked_features.npy',perturbed,allow_pickle=False)
    pd.DataFrame(rows).to_parquet(directory/'mask_ledger.parquet',index=False)
    return observed,perturbed

def summarize(participant_metrics):
    summaries=[]
    fields=['log_loss','brier','auc','balanced_accuracy','accuracy','probability_sd',
            'predicted_positive_fraction','drift','flip_rate']
    for (dataset,condition),group in participant_metrics.groupby(['dataset','condition'],sort=True):
        for metric in fields:
            arr=group[metric].to_numpy(dtype=float)
            est,low,high=c.bootstrap_median(arr,f'{dataset}|{condition}|{metric}')
            summaries.append({'dataset':dataset,'kind':'condition','condition':condition,'metric':metric,
                'median':est,'ci_low':low,'ci_high':high,'n_participants':int(np.isfinite(arr).sum()),
                'n_roster':len(arr),'positive_differences':None,'negative_differences':None})
    for dataset,group in participant_metrics.groupby('dataset',sort=True):
        wide={name:g.set_index('participant').sort_index() for name,g in group.groupby('condition')}
        for metric in ['drift','flip_rate','log_loss','brier','auc','balanced_accuracy','accuracy']:
            arr=(wide['contiguous_20pct'][metric]-wide['scattered_random_20pct'][metric]).to_numpy()
            est,low,high=c.bootstrap_median(arr,f'{dataset}|paired|{metric}')
            summaries.append({'dataset':dataset,'kind':'paired','condition':'contiguous_minus_random','metric':metric,
                'median':est,'ci_low':low,'ci_high':high,'n_participants':int(np.isfinite(arr).sum()),
                'n_roster':len(arr),'positive_differences':int((arr>0).sum()),'negative_differences':int((arr<0).sum())})
        for metric in ['log_loss','brier']:
            arr=(wide['prior'][metric]-wide['observed'][metric]).to_numpy()
            est,low,high=c.bootstrap_median(arr,f'{dataset}|skill|{metric}')
            summaries.append({'dataset':dataset,'kind':'baseline_skill','condition':'prior_minus_model','metric':metric,
                'median':est,'ci_low':low,'ci_high':high,'n_participants':len(arr),'n_roster':len(arr),
                'positive_differences':int((arr>0).sum()),'negative_differences':int((arr<0).sum())})
    return pd.DataFrame(summaries)

def run(output):
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    manifest={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
              'model_fits':0,'new_prediction_results':True,'phase':'input_checks'}
    save_json(output/'manifest.json',manifest)
    try:
        protocol_path=role('prediction_protocol', HERE/'FROZEN_PROTOCOL.json')
        protocol=json.loads(protocol_path.read_text(encoding='utf8'))
        for rel,h in protocol['sources'].items():
            check_source(HERE/rel,h)
        input_path=resolve_input(protocol['windows_path'])
        if sha(input_path)!=protocol['windows_sha256']:raise ValueError('Prepared windows changed')
        manifest.update(protocol_sha256=sha(protocol_path),windows_sha256=sha(input_path),feature_names=c.FEATURE_NAMES,
                        scientific_config=protocol['config'])
        windows=pd.read_parquet(input_path).sort_values(['dataset','participant','label_time','window_id']).reset_index(drop=True)
        validate_windows(windows).to_csv(output/'participant_denominators.csv',index=False)
        windows.drop(columns=['times','values']).to_parquet(output/'window_index.parquet',index=False)
        save_json(output/'manifest.json',manifest)
        draw_count=protocol['config']['draws']
        observed_frames=[];prediction_frames=[];metric_rows=[];draw_metric_rows=[];fold_rows=[]
        for dataset,part in windows.groupby('dataset',sort=True):
            part=part.reset_index(drop=True)
            target=part.label.to_numpy();groups=part.participant.to_numpy()
            directory=output/dataset;directory.mkdir();(directory/'models').mkdir()
            x,xm=prepare_dataset_features(dataset,part,directory,draw_count)
            for number,pid in enumerate(sorted(np.unique(groups))):
                fold=c.fit_fold(x,target,groups,pid)
                manifest['model_fits']+=1
                idx=fold['test_indices']
                y=target[idx];base=c.predict(fold,x[idx]);prior=np.full(len(idx),fold['prior'])
                masked=c.predict(fold,xm[idx].reshape(-1,x.shape[1])).reshape(len(idx),draw_count,2)
                if c.fingerprint(fold)!=fold['fingerprint']:raise ValueError('Scoring mutated trained state')
                with (directory/'models'/f'{pid}.pkl').open('wb') as f:pickle.dump(fold,f,pickle.HIGHEST_PROTOCOL)
                info={k:v for k,v in fold.items() if k not in ['model','scaler','train_indices','test_indices']}
                info.update(dataset=dataset,unmasked_train_only=True,frozen_during_scoring=True)
                fold_rows.append(info)
                observations=part.iloc[idx].drop(columns=['times','values']).copy()
                observations['window_index']=idx;observations['p_observed']=base;observations['p_prior']=prior
                observations['model_fingerprint']=fold['fingerprint'];observed_frames.append(observations)
                predictions=pd.DataFrame({'dataset':dataset,'participant':pid,
                    'window_index':np.repeat(idx,draw_count*2),
                    'window_id':np.repeat(part.iloc[idx].window_id.to_numpy(),draw_count*2),
                    'draw':np.tile(np.repeat(np.arange(draw_count),2),len(idx)),
                    'geometry':np.tile(np.array(GEOMETRIES),len(idx)*draw_count),
                    'label':np.repeat(y,draw_count*2),'p':masked.reshape(-1),
                    'p_observed':np.repeat(base,draw_count*2),'p_prior':np.repeat(prior,draw_count*2),
                    'model_fingerprint':fold['fingerprint']})
                prediction_frames.append(predictions)
                for condition,p in [('observed',base),('prior',prior)]:
                    metric_rows.append({'dataset':dataset,'participant':pid,'condition':condition,
                        'n_windows':len(idx),'n_draws':1,**c.change_metrics(y,p,p)})
                for gi,geometry in enumerate(GEOMETRIES):
                    all_draws=[]
                    for draw in range(draw_count):
                        metric=c.change_metrics(y,base,masked[:,draw,gi]);all_draws.append(metric)
                        draw_metric_rows.append({'dataset':dataset,'participant':pid,'geometry':geometry,
                            'draw':draw,'n_windows':len(idx),**metric})
                    means={key:float(np.mean([r[key] for r in all_draws])) for key in all_draws[0]}
                    metric_rows.append({'dataset':dataset,'participant':pid,'condition':geometry,
                        'n_windows':len(idx),'n_draws':draw_count,**means})
                emit(stage='fit',dataset=dataset,heldout_number=number+1,heldout_total=len(np.unique(groups)),
                     n_train=len(fold['train_indices']),n_test=len(idx))
            # Save partial results after each whole dataset, retain if later failure.
            pd.concat(observed_frames,ignore_index=True).to_parquet(output/'observed_predictions.parquet',index=False)
            pd.concat(prediction_frames,ignore_index=True).to_parquet(output/'masked_predictions.parquet',index=False)
            save_json(output/'fold_ledger.json',fold_rows)
            save_json(output/'manifest.json',manifest)
        participant_metrics=pd.DataFrame(metric_rows)
        participant_metrics.to_csv(output/'participant_metrics.csv',index=False)
        pd.DataFrame(draw_metric_rows).to_csv(output/'participant_draw_metrics.csv',index=False)
        summary=summarize(participant_metrics);summary.to_csv(output/'summary.csv',index=False)
        guards=[]
        for dataset in sorted(windows.dataset.unique()):
            ss=summary[summary.dataset==dataset]
            skill=ss[(ss.kind=='baseline_skill')&(ss.metric=='log_loss')].iloc[0]
            auc=ss[(ss.condition=='observed')&(ss.metric=='auc')].iloc[0]
            pm=participant_metrics[(participant_metrics.dataset==dataset)&(participant_metrics.condition=='observed')]
            guards.append({'dataset':dataset,'n_participants':len(pm),'n_near_constant_probability_sd_lt_0_01':int((pm.probability_sd<.01).sum()),
                'n_single_predicted_class':int(pm.predicted_positive_fraction.isin([0.,1.]).sum()),
                'median_prior_minus_model_log_loss':float(skill['median']),
                'median_observed_auc':float(auc['median']),
                'point_estimate_skill_guard':bool(skill['median']>0 and auc['median']>.5),
                'both_descriptive_lower_bounds_above_null':bool(skill.ci_low>0 and auc.ci_low>.5),
                'participants_excluded_based_on_performance':0})
        save_json(output/'baseline_guard.json',guards)
        manifest.update(status='complete',phase='complete',elapsed_seconds=time.perf_counter()-start,
            evaluated_windows=len(windows),masked_prediction_rows=sum(len(f) for f in prediction_frames),
            induced_mask_draws=draw_count,all_folds_converged=True,all_fit_fingerprints_unchanged=True,
            participants_by_dataset=windows.groupby('dataset').participant.nunique().to_dict(),
            outputs={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'})
        save_json(output/'manifest.json',manifest)
        emit(status='complete',windows=len(windows),models=manifest['model_fits'],seconds=manifest['elapsed_seconds'])
    except BaseException:
        manifest.update(status='failed',elapsed_seconds=time.perf_counter()-start,traceback=traceback.format_exc())
        save_json(output/'manifest.json',manifest)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output)
