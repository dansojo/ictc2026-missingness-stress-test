"""Read all local external subjects; audit prospective targets/support, never fit models.

Candidate support rules are recorded before running this inventory. Counts are
feasibility information, not performance or a basis for selecting favorable effects.
"""
from pathlib import Path
from portable_paths import role
import sys, json, hashlib, gzip, csv, zipfile, platform
from collections import Counter
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
OUT = role('work_output')
ROOT = OUT.parents[2]
ARCHIVE = role('external_provider')
ORIGINAL = role('reference')/'external'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()

sources = []
prediction_windows = []
def pin(path):
    sources.append({'path':str(path), 'bytes':path.stat().st_size, 'sha256':sha(path)})

def save_json(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def class_counts(labels):
    a = np.asarray(labels, dtype=float)
    return {'n':int(np.isfinite(a).sum()), 'positive':int((a==1).sum()), 'negative':int((a==0).sum())}

def extra():
    rows=[]
    cols=['timestamp','raw_acc:magnitude_stats:mean','raw_acc:magnitude_stats:std',
          'discrete:app_state:is_active','discrete:app_state:is_inactive',
          'discrete:app_state:is_background','discrete:app_state:missing','label:SLEEPING']
    for k,path in enumerate(sorted((ARCHIVE/'ExtraSensory.per_uuid_features_labels').glob('*.csv.gz')),1):
        pin(path)
        df=pd.read_csv(path,usecols=cols).sort_values('timestamp',kind='stable')
        assert set(df['label:SLEEPING'].dropna().unique()) <= {0.,1.}
        t=df.timestamp.to_numpy(float); y=df['label:SLEEPING'].to_numpy(float)
        acc=df[['raw_acc:magnitude_stats:mean','raw_acc:magnitude_stats:std']].to_numpy(float)
        valid=np.isfinite(t)&np.isfinite(acc).all(axis=1)&(acc>=0).all(axis=1)
        ts=t[valid]
        assert (np.diff(t)>0).all(), 'duplicate/unsorted timestamps require explicit handling'
        state=df[['discrete:app_state:is_active','discrete:app_state:is_inactive','discrete:app_state:is_background']].eq(1).any(axis=1)&~df['discrete:app_state:missing'].eq(1)
        r={'participant':path.name.split('.')[0],'rows':len(df),'finite_acc_mean_std':int(valid.sum()),
           'finite_app_state':int(state.sum()),'timestamp_duplicate_count':int(df.timestamp.duplicated().sum()),
           'utc_days':int(np.unique(np.floor(t/86400)).size),
           'median_positive_gap_seconds':float(np.median(np.diff(t)[np.diff(t)>0]))}
        r.update({'sleep_'+key:val for key,val in class_counts(y).items()})
        # Select first record of each UTC hour BEFORE inspecting its label.
        first_hour=np.r_[True,np.diff(np.floor(t/3600))!=0]
        r['first_utc_hour_anchors']=int(first_hour.sum())
        for minutes in [30,60]:
            # Strictly previous recordings, >=20 s earlier, avoiding 20 s recording overlap.
            lo=np.searchsorted(ts,t-minutes*60,side='left')
            hi=np.searchsorted(ts,t-20,side='right')
            support=hi-lo
            keep=np.isfinite(y)&(support>=minutes//2)
            r.update({f'prior{minutes}_all_'+key:val for key,val in class_counts(y[keep]).items()})
            r.update({f'prior{minutes}_hour_'+key:val for key,val in class_counts(y[keep&first_hour]).items()})
            r[f'prior{minutes}_hour_missing_label']=int((first_hour&~np.isfinite(y)).sum())
            r[f'prior{minutes}_hour_insufficient_sensor_labeled']=int((first_hour&np.isfinite(y)&~keep).sum())
            r[f'prior{minutes}_hour_min_records']=int(support[keep&first_hour].min()) if (keep&first_hour).any() else None
            r[f'prior{minutes}_hour_max_records']=int(support[keep&first_hour].max()) if (keep&first_hour).any() else None
        # Parent-specified final feasibility candidate, fixed before this run:
        # mean-magnitude only, >=30 observations with >=45 min span, greedy
        # windows at least one hour apart. Label values never guide selection.
        mean=df['raw_acc:magnitude_stats:mean'].to_numpy(float)
        valid_mean=np.isfinite(t)&np.isfinite(mean)&(mean>=0)
        mt=t[valid_mean];mv=mean[valid_mean]
        chosen=[];last=-np.inf;insufficient=0;overlap=0
        for idx in np.flatnonzero(np.isfinite(y)):
            anchor=t[idx]
            a=np.searchsorted(mt,anchor-3600,side='left')
            b=np.searchsorted(mt,anchor-20,side='right')
            if b-a<30 or mt[b-1]-mt[a]<2700:
                insufficient+=1;continue
            if anchor-last<3600:
                overlap+=1;continue
            chosen.append(y[idx]);last=anchor
            prediction_windows.append({'dataset':'ExtraSensory','participant':r['participant'],
                'window_id':str(int(anchor)),'label':int(y[idx]),'window_start':float(anchor-3600),
                'window_end':float(anchor-20),'label_time':float(anchor),'acquisition_duration':20.0,
                'times':mt[a:b].tolist(),'values':mv[a:b].tolist()})
        r.update({'nonoverlap60_'+key:val for key,val in class_counts(chosen).items()})
        r['nonoverlap60_insufficient_sensor_labeled']=insufficient;r['nonoverlap60_overlap_skipped']=overlap
        rows.append(r)
        print(f'ExtraSensory {k}/60',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'extrasensory_participant_counts.csv',index=False)
    result={'raw_participants':len(frame),'raw_rows':int(frame.rows.sum()),'support_rules':
       {'windows_minutes':[30,60],'required_valid_records':'half window minutes (15 or 30)',
        'valid_record':'finite nonnegative acceleration magnitude mean AND std',
        'sensor_window':'[t-window, t-20 seconds] to avoid overlap with 20-second recording session',
        'hour_anchor':'first sensor-file record in each UTC hour, selected independently of labels',
        'no_label_imputation':True}, 'candidates':{}}
    for col in ['sleep','prior30_all','prior60_all','prior30_hour','prior60_hour','nonoverlap60']:
        result['candidates'][col]={
            'participants_with_any':int((frame[col+'_n']>0).sum()),
            'participants_with_both_classes':int(((frame[col+'_positive']>0)&(frame[col+'_negative']>0)).sum()),
            'participants_with_at_least_10':int((frame[col+'_n']>=10).sum()),
            'rows':int(frame[col+'_n'].sum()),'positive':int(frame[col+'_positive'].sum()),'negative':int(frame[col+'_negative'].sum()),
            'per_person_rows_min':int(frame.loc[frame[col+'_n']>0,col+'_n'].min()),
            'per_person_rows_median':float(frame.loc[frame[col+'_n']>0,col+'_n'].median()),
            'per_person_rows_max':int(frame.loc[frame[col+'_n']>0,col+'_n'].max())}
    zip_path=ARCHIVE/'02_experiments/external_data/extrasensory/downloads/cv5Folds.zip';pin(zip_path)
    folds=[]
    with zipfile.ZipFile(zip_path) as z:
        for fold in range(5):
            sets={}
            for part in ['train','test']:
                uuids=[]
                for osname in ['iphone','android']:
                    uuids.extend(z.read(f'cv_5_folds/fold_{fold}_{part}_{osname}_uuids.txt').decode('utf-8-sig').split())
                sets[part]=set(uuids)
            assert not sets['train']&sets['test']
            assert sets['train']|sets['test']==set(frame.participant)
            folds.append({'fold':fold,'train':sorted(sets['train']),'test':sorted(sets['test']),
                          'prior60_hour_train_rows':int(frame.loc[frame.participant.isin(sets['train']),'prior60_hour_n'].sum()),
                          'prior60_hour_test_rows':int(frame.loc[frame.participant.isin(sets['test']),'prior60_hour_n'].sum())})
    assert len(set.union(*(set(f['test']) for f in folds)))==60
    save_json('extrasensory_official_folds.json',folds)
    result['official_5fold_partition_available']=True
    pin(ARCHIVE/'02_experiments/external_data/extrasensory/downloads/README.txt')
    return result

def student():
    base=ARCHIVE/'StudentLife/dataset'
    pin(base/'EMA/EMA_definition.json')
    rows=[]; invalid_codes=Counter(); allday_hours=Counter(); total_day_rows=[]
    paths=sorted((base/'EMA/response/Sleep').glob('Sleep_u*.json'))
    for k,path in enumerate(paths,1):
        pin(path);pid=path.stem.removeprefix('Sleep_')
        recs=json.loads(path.read_text(encoding='utf-8-sig'))
        labels=[]
        for rr in recs:
            try: ts=float(rr['resp_time']);rate=float(rr.get('rate',float('nan')))
            except (ValueError,TypeError,KeyError): continue
            if not np.isfinite(ts):continue
            if rate not in (1,2,3,4):
                invalid_codes[str(rr.get('rate','missing'))]+=1; continue
            local=pd.Timestamp(ts,unit='s',tz='UTC').tz_convert('America/New_York')
            allday_hours[str(local.hour)]+=1
            labels.append({'timestamp':ts,'response_day':str(local.date()),'hour':local.hour,'y':float(rate>=3)})
        ldf=pd.DataFrame(labels).sort_values('timestamp',kind='stable')
        r={'participant':pid,'raw_sleep_records':len(recs),'valid_quality_records':len(ldf),
           'duplicate_valid_response_days':int(ldf.duplicated('response_day').sum()) if len(ldf) else 0}
        r.update({'all_valid_'+key:val for key,val in class_counts(ldf.y if len(ldf) else []).items()})
        activity=base/'sensing/activity'/f'activity_{pid}.csv'
        if not activity.exists(): raise RuntimeError(f'missing activity file for {pid}')
        pin(activity);af=pd.read_csv(activity);af.columns=af.columns.str.strip()
        t=pd.to_numeric(af.timestamp,errors='coerce').to_numpy(float)
        v=pd.to_numeric(af['activity inference'],errors='coerce').to_numpy(float)
        valid=np.isfinite(t)&np.isfinite(v)&np.isin(v,[0,1,2])
        loc=pd.to_datetime(t[valid],unit='s',utc=True).tz_convert('America/New_York')
        adf=pd.DataFrame({'t':t[valid],'local_day':loc.date.astype(str),'local_hour':loc.hour,'v':v[valid]})
        grouped=adf.groupby('local_day').agg(n=('t','size'),hours=('local_hour','nunique'),first=('t','min'),last=('t','max'))
        r['raw_activity_records']=len(af);r['valid_activity_records']=int(valid.sum());r['local_activity_days']=len(grouped)
        r['activity_invalid_records']=int((~valid).sum());r['activity_duplicate_valid_timestamps']=int(adf.t.duplicated().sum())
        for window in ['all_day_responses','daytime_04_to_18']:
            cand=ldf if window=='all_day_responses' else ldf.loc[(ldf.hour>=4)&(ldf.hour<18)]
            cand=cand.drop_duplicates('response_day',keep='first').copy()
            for threshold in ['20_records','20_records_and_12_hours']:
                kept=[]
                for rr in cand.itertuples(index=False):
                    prior=(pd.Timestamp(rr.response_day)-pd.Timedelta(days=1)).date().isoformat()
                    if prior not in grouped.index:continue
                    gr=grouped.loc[prior]
                    if gr['n']<20 or (threshold=='20_records_and_12_hours' and gr.hours<12):continue
                    kept.append(rr.y)
                    if window=='daytime_04_to_18' and threshold=='20_records_and_12_hours':
                        total_day_rows.append({'participant':pid,'response_day':rr.response_day,'sensor_day':prior,'label':int(rr.y),'n_records':int(gr['n']),'occupied_hours':int(gr.hours)})
                prefix=window+'_'+threshold
                r.update({prefix+'_'+key:val for key,val in class_counts(kept).items()})
        chosen=[];insufficient=0;no_sensor=0
        for rr in ldf.drop_duplicates('response_day',keep='first').itertuples(index=False):
            midnight=pd.Timestamp(rr.response_day,tz='America/New_York')
            previous=midnight-pd.DateOffset(days=1)
            prior=previous.date().isoformat()
            if prior not in grouped.index:no_sensor+=1;continue
            gr=grouped.loc[prior]
            if gr['n']<60 or gr['last']-gr['first']<43200:insufficient+=1;continue
            raw=adf.loc[adf.local_day==prior].sort_values('t',kind='stable')
            assert (raw.t>=previous.timestamp()).all() and (raw.t<midnight.timestamp()).all()
            chosen.append(rr.y)
            prediction_windows.append({'dataset':'StudentLife','participant':pid,
                'window_id':prior,'label':int(rr.y),'window_start':previous.timestamp(),
                'window_end':midnight.timestamp(),'label_time':float(rr.timestamp),'acquisition_duration':0.0,
                'times':raw.t.to_list(),'values':raw.v.to_list()})
        r.update({'final_prior_day_'+key:val for key,val in class_counts(chosen).items()})
        r['final_prior_day_missing_sensor_day']=no_sensor;r['final_prior_day_insufficient_sensor_day']=insufficient
        rows.append(r);print(f'StudentLife {k}/{len(paths)}',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'studentlife_participant_counts.csv',index=False)
    # No raw responses, timestamps, or per-day labels are written. Per-person aggregates suffice.
    result={'sleep_participant_files':len(frame),'raw_sleep_records':int(frame.raw_sleep_records.sum()),
      'valid_quality_records':int(frame.valid_quality_records.sum()),'invalid_rate_codes':dict(invalid_codes),
      'valid_response_local_hour_histogram':dict(sorted(allday_hours.items(),key=lambda x:int(x[0]))),
      'timezone':'America/New_York (study location assumption; participant travel not resolved)',
      'target':'Poor self-reported sleep: rate 3/4 vs 1/2; exact provided EMA definition',
      'alignment':'Previous complete local calendar day [midnight, next midnight), strictly before response calendar day; this is not inferred sleep onset/offset.',
      'deduplication':'earliest valid response per eligible response day, no favorable-label tie handling',
      'candidates':{}}
    for col in [c[:-2] for c in frame.columns if c.endswith('_n')]:
        result['candidates'][col]={'participants_with_any':int((frame[col+'_n']>0).sum()),
           'participants_with_both_classes':int(((frame[col+'_positive']>0)&(frame[col+'_negative']>0)).sum()),
           'participants_with_at_least_5':int((frame[col+'_n']>=5).sum()),
           'rows':int(frame[col+'_n'].sum()),'positive':int(frame[col+'_positive'].sum()),'negative':int(frame[col+'_negative'].sum()),
           'per_person_rows_min':int(frame.loc[frame[col+'_n']>0,col+'_n'].min()),
           'per_person_rows_median':float(frame.loc[frame[col+'_n']>0,col+'_n'].median()),
           'per_person_rows_max':int(frame.loc[frame[col+'_n']>0,col+'_n'].max())}
    return result

def main():
    save_json('INVENTORY_RULES.json',{'scope':'labels and sensor support only; no model fit, no probabilities or outcome performance inspection',
        'frozen_before_inventory':True,'ExtraSensory_windows_minutes':[30,60],'ExtraSensory_min_records':[15,30],
        'ExtraSensory_hourly_anchor':'first raw-file row per UTC hour selected before label availability',
        'StudentLife_response_hours_candidates':['all','04:00 inclusive to 18:00 exclusive'],
        'StudentLife_day_support_candidates':['20 records','20 records and 12 occupied local hours'],
        'selected_extra_candidate':'all labeled t; finite nonnegative raw_acc:magnitude_stats:mean in [t-3600,t-20] inclusive; assumed20sec acquisition ends<=t; >=30 rows; >=2700sec span; greedily retain next t at least 3600sec after previous; no class-value selection',
        'selected_student_candidate':'earliest valid rate per America/New_York local date; all response hours; previous complete local day; >=60 valid activity records spanning >=12h; values0/1/2; no class-value selection',
        'performance_based_selection_forbidden':True})
    result={'scope':'complete already-local external prediction feasibility inventory; no models trained',
            'extrasensory':extra(),'studentlife':student()}
    p=ORIGINAL/'external_transfer_summary.csv';pin(p)
    result['original_feature_analysis']=pd.read_csv(p)[['dataset','feature_family','eligible_participants','positive_participants']].to_dict('records')
    result['environment']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}
    windows=pd.DataFrame(prediction_windows)
    windows.to_parquet(OUT/'eligible_prediction_windows.parquet',index=False)
    result['prepared_windows']={'path':'eligible_prediction_windows.parquet','rows':len(windows),
       'datasets':windows.groupby('dataset').agg(windows=('window_id','size'),participants=('participant','nunique')).reset_index().to_dict('records'),
       'raw_arrays_stored_locally_only':True,'no_models_fitted':True,
       'note':'ExtraSensory sensor timestamps <= t-20 sec, current-row measurements excluded. Upper source-coordinate end inclusive. StudentLife sensor times strictly less than prior-day end.'}
    pin(Path(__file__))
    save_json('inventory.json',result);save_json('source_manifest.json',sources)
    outputs=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
    save_json('manifest.json',{'status':'COMPLETE','model_fits':0,'files_read_and_hashed':len(sources),
       'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs}})
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
