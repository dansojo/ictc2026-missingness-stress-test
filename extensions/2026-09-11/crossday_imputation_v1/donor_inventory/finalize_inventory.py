"""Read completed inventory to document feasibility without imputing any value."""
from datetime import datetime,timezone
from pathlib import Path
from portable_paths import role, resolve_input
import json
import pickle
import numpy as np
import pandas as pd
import inventory_core as core

SOURCE=Path(__file__).resolve().parent
HERE=role('work_output')
sha256=core.legacy.sha256


def table(frame):
    columns=list(frame.columns)
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(map(str,row))+' |')
    return '\n'.join(rows)


def main():
    m=json.loads((HERE/'manifest.json').read_text(encoding='utf8'))
    if m['status']!='complete' or m['target_cells_checked']!=500:
        raise ValueError('full exact cache gate required')
    checks=pd.read_csv(HERE/'target_exact_checks.csv')
    selected={(r.subject_id,r.primitive,int(r.sensor_day_id)):r for r in checks.itertuples(index=False)}
    boundary=[]; target_support=[]; partition_checks=[]; cached_usage_keys=set()
    for part in m['partitions']:
        path=HERE/part['path']
        if sha256(path)!=part['sha256']:
            raise ValueError('cache partition changed')
        with path.open('rb') as stream:
            pool=pickle.load(stream)
        for key,c in pool.items():
            if (c['subject_id'],c['primitive'],c['sensor_day_id'])!=key:
                raise ValueError('cache identity/key mismatch')
            if any(field in c for field in ['masks','reference','scale','m0_error','imputed']):
                raise ValueError('forbidden outcome field in donor cache')
            if c['primitive']=='usage_load_24h':
                cached_usage_keys.update((c['subject_id'],int(t)) for t in c['times'])
            if key not in selected:
                continue
            rec=selected[key]
            with resolve_input(rec.original_path).open('rb') as stream:
                original=pickle.load(stream)
            for field in ['raw','valid','times']:
                np.testing.assert_array_equal(c[field],original[field],strict=True)
            pd.testing.assert_frame_equal(c['raw_frame'],original['raw_frame'],check_exact=True)
            pd.testing.assert_frame_equal(c['calendar'],original['calendar'],check_exact=True)
            for field in ['source_record_digest','original_source_hash']:
                if c[field]!=original[field]:
                    raise ValueError('serialized target identity differs')
            partition_checks.append(dict(subject_id=key[0],primitive=key[1],sensor_day_id=key[2],exact=True))
            others=[d for d in pool.values() if d['day_start_ns']!=c['day_start_ns']]
            b=dict(subject_id=key[0],primitive=key[1],sensor_day_id=key[2],date=c['date'],
                positive_support_overlap_records=sum(core.target_overlap_count(d,c) for d in others),
                donor_source_timestamps_on_target_calendar_date=sum(int(((d['times']>=c['day_start_ns'])&
                    (d['times']<c['day_end_ns'])).sum()) for d in others),
                preceding_donor_support_ends_at_target_start=sum(int((d['ends']==c['day_start_ns']).sum())
                    for d in others if d['day_start_ns']<c['day_start_ns']),
                following_donor_support_starts_at_target_end=sum(int((d['starts']==c['day_end_ns']).sum())
                    for d in others if d['day_start_ns']>c['day_start_ns']))
            boundary.append(b)
            idx=c['local_clock_ns']//(30*core.MINUTE_NS)
            raw_count=np.bincount(idx,minlength=48)
            valid_count=np.bincount(idx[c['valid']],minlength=48)
            for i in range(48):
                target_support.append(dict(subject_id=key[0],primitive=key[1],sensor_day_id=key[2],slot=i,
                    target_raw_records=int(raw_count[i]),target_valid_records=int(valid_count[i])))
    if len(partition_checks)!=500 or any(r['positive_support_overlap_records'] for r in boundary):
        raise ValueError('serialized-cache verification or boundary gate failed')
    pd.DataFrame(partition_checks).to_csv(HERE/'serialized_target_checks.csv',index=False)
    pd.DataFrame(boundary).to_csv(HERE/'target_day_boundary_audit.csv',index=False)
    slot=pd.read_parquet(HERE/'slot30_donor_counts.parquet').merge(pd.DataFrame(target_support),
        on=['subject_id','primitive','sensor_day_id','slot'],how='inner',validate='one_to_one')
    slot.to_csv(HERE/'slot30_support_feasibility.csv',index=False)
    slot.to_parquet(HERE/'slot30_support_feasibility.parquet',index=False)
    feasibility=[]
    for primitive,group in slot.groupby('primitive',sort=True):
        for donor_pool in ['all','past','future','same_weektype','past_same_weektype']:
            counts=group[donor_pool+'_valid_donor_dates']
            occupied=group.target_valid_records.gt(0)
            raw_occupied=group.target_raw_records.gt(0)
            feasibility.append(dict(primitive=primitive,pool=donor_pool,target_slots=len(group),
                target_slots_with_valid_records=int(occupied.sum()),
                target_slots_with_raw_records=int(raw_occupied.sum()),
                all_slots_min_donor_dates=int(counts.min()),all_slots_median_donor_dates=float(counts.median()),
                all_slots_max_donor_dates=int(counts.max()),
                all_slots_with_at_least_2_donor_dates=int(counts.ge(2).sum()),
                raw_occupied_target_slots_with_at_least_2_donor_dates=int((raw_occupied&counts.ge(2)).sum()),
                occupied_target_slots_with_at_least_2_donor_dates=int((occupied&counts.ge(2)).sum())))
    feasibility=pd.DataFrame(feasibility)
    feasibility.to_csv(HERE/'slot30_feasibility_summary.csv',index=False)
    daily=pd.read_csv(HERE/'daily_support.csv')
    support=[]
    for primitive,group in daily.groupby('primitive',sort=True):
        observed=group[group.records.gt(0)]
        support.append(dict(primitive=primitive,calendar_dates=len(group),
            raw_observed_dates=int(group.records.gt(0).sum()),valid_observed_dates=int(group.valid_records.gt(0).sum()),
            raw_records=int(group.records.sum()),valid_records=int(group.valid_records.sum()),
            occupied_cadence_bins=int(group.observed_epochs.sum()),
            extra_records_in_occupied_bins=int(group.within_bin_extra_records.sum()),
            median_observed_day_valid_bins=float(observed.observed_epochs.median()),
            median_observed_day_longest_missing_minutes=float(observed.longest_missing_run_minutes.median()),
            max_observed_day_longest_missing_minutes=int(observed.longest_missing_run_minutes.max()),
            next_midnight_endpoint_records=int(group.source_midnight_endpoint_records.sum())))
    support=pd.DataFrame(support)
    support.to_csv(HERE/'cadence_support_summary.csv',index=False)
    source=pd.read_csv(HERE/'source_audit.csv')
    if source[['exact_duplicate_keys','missing_timestamps','nonzero_seconds','nonzero_subseconds']].to_numpy().any():
        raise ValueError('timestamp facts differ from documented inventory assumptions')
    usage_path=Path(next(path for path in m['input_hashes'] if path.endswith('mUsageStats_canonical.parquet')))
    usage_source=pd.read_parquet(usage_path,columns=['subject_id','timestamp'])
    omitted=usage_source.loc[[(r.subject_id,r.timestamp.value) not in cached_usage_keys
        for r in usage_source.itertuples(index=False)]].copy()
    if len(omitted)!=int(source.loc[source.primitive.eq('usage_load_24h'),'raw_rows_not_in_calendar_cache'].iloc[0]):
        raise ValueError('unaccounted usage source records')
    omitted['support_start']=omitted.timestamp-pd.Timedelta(minutes=10)
    omitted['support_end']=omitted.timestamp
    omitted['support_owner_date']=(omitted.timestamp-pd.Timedelta(microseconds=1)).dt.normalize()
    calendar=pd.read_parquet(HERE/'calendar.parquet')
    calendar_keys=set(zip(calendar.subject_id,calendar.lifelog_date))
    omitted['reason']=['support_owner_date_outside_original_calendar' if (r.subject_id,r.support_owner_date) not in calendar_keys
        else 'support_crosses_calendar_boundary' for r in omitted.itertuples(index=False)]
    omitted.to_csv(HERE/'source_records_outside_calendar_cache.csv',index=False)
    counts=pd.read_csv(HERE/'donor_counts_by_target.csv')
    perperson=pd.read_csv(HERE/'donor_summary.csv')
    short={'screen_load_24h':'Screen','phone_activity_load_24h':'Activity',
        'usage_load_24h':'Usage','mobile_light_exposure_24h':'Mobile light','wearable_light_exposure_24h':'Wearable light'}
    availability=perperson.pivot(index='subject_id',columns='primitive',values='valid_observed_dates').rename(columns=short)
    availability.insert(0,'Calendar days',perperson.groupby('subject_id').calendar_dates.first())
    availability=availability.reset_index().rename(columns={'subject_id':'Participant'})
    ranges=counts.groupby('primitive')[['valid_donor_dates','past_valid_donor_dates','future_valid_donor_dates',
        'same_weektype_valid_donor_dates','past_same_weektype_valid_donor_dates',
        'weekday_valid_donor_dates','weekend_valid_donor_dates']].agg(['min','median','max'])
    ranges.columns=['_'.join(c) for c in ranges.columns]
    ranges=ranges.reset_index()
    ranges.to_csv(HERE/'donor_count_ranges.csv',index=False)
    range_rows=[]
    for r in ranges.itertuples(index=False):
        range_rows.append(dict(Primitive=short[r.primitive],
            All=f'{r.valid_donor_dates_min}/{r.valid_donor_dates_median:g}/{r.valid_donor_dates_max}',
            Past=f'{r.past_valid_donor_dates_min}/{r.past_valid_donor_dates_median:g}/{r.past_valid_donor_dates_max}',
            Future=f'{r.future_valid_donor_dates_min}/{r.future_valid_donor_dates_median:g}/{r.future_valid_donor_dates_max}',
            Same_weektype=f'{r.same_weektype_valid_donor_dates_min}/{r.same_weektype_valid_donor_dates_median:g}/{r.same_weektype_valid_donor_dates_max}'))
    simple_support=support[['primitive','raw_records','valid_records','extra_records_in_occupied_bins',
                            'median_observed_day_valid_bins','median_observed_day_longest_missing_minutes']].copy()
    simple_support['primitive']=simple_support.primitive.map(short)
    simple_support.columns=['Primitive','Raw records','Valid records','Extra raw records in cadence bins',
                            'Median observed-day valid bins','Median longest missing run (min)']
    simple_feasible=feasibility[['primitive','pool',
        'target_slots_with_valid_records','occupied_target_slots_with_at_least_2_donor_dates']].copy()
    simple_feasible['primitive']=simple_feasible.primitive.map(short)
    simple_feasible.columns=['Primitive','Pool','Target slots with valid records','Those slots with >=2 donor dates']
    report=f'''# Cross-day donor input feasibility

The inventory contains **853 original calendar days × 5 primitives = 4,265 cells**, in 50 participant/primitive partitions. All **500 evaluation cells match exactly** before and after serialization: raw values, validity, source timestamps, raw frames, calendar context, source hashes and record digests. Support coordinates also match the immutable timestamp contracts. No imputation values, errors, recovery outcomes, fitted models, or performance-based exclusions were computed.

## Available dates

The following counts are dates with at least one **valid original raw record**, before excluding the target calendar day. Calendar days include sensor-empty dates. Every selected target's eligible-date list and past/future/weekday/weekend counts are in `donor_counts_by_target.csv`; date exclusion is by canonical calendar date, never outcome quality.

{table(availability)}

Across the fixed 500 targets, valid donor-date counts after excluding the target day are below. Each entry is **minimum/median/maximum across the 100 fixed target cells for that primitive**. `donor_count_ranges.csv` additionally includes past-same-weektype and separate weekday/weekend counts.

{table(pd.DataFrame(range_rows))}

All-date pools include later dates and therefore describe **offline reconstruction**. Past-only counts use strictly earlier local calendar dates. Same-weektype means weekday-to-weekday or weekend-to-weekend, not the same named weekday. Empty dates and days with zero valid records remain present in the cache but contribute zero donor support.

## Timestamp origin and day boundaries

`sensor_day_id` is the original globally sequential calendar ID; it is not a Unix day or a participant-relative day count. Participant calendars range from their earliest to latest observed canonical source date and preserve missing intervening dates. `lifelog_date` is local midnight and `sleep_date` is next local midnight. Their fixed difference is 24 hours.

The preprocessing source copies input timestamps with `pd.to_datetime`, without timezone conversion. The scientific extractor explicitly requires **timezone-naive local-clock values** and rejects timezone-aware inputs. No IANA timezone is stored; treating these encoded nanoseconds as UTC or adding an Asia/Seoul conversion would alter the original data. Calendar-day, weekday/weekend, and clock alignment use these original naive values directly.

Screen, activity, and wearable light are nominal one-minute instant streams. Mobile light is a nominal ten-minute instant stream. Their source, effective, support-start, and support-end coordinates coincide; selection is `[day_start, day_end)`. All five files have exact-minute coordinates, zero nonzero seconds/subseconds, and zero duplicate `(subject_id,timestamp)` keys.

Usage records are interval-end events with preceding ten-minute support `[t−10min,t]`, midpoint `t−5min`, and raw reported app-usage time in milliseconds. The complete support must lie inside the calendar day. A record ending at the next midnight belongs to the preceding day; a record ending at the current midnight is excluded from the current day. Crossing support such as 00:05 (23:55–00:05) is excluded by the unchanged extractor. The cache retains source endpoints (`times`) and midpoint coordinates (`effective_times`, `local_clock_ns`) separately. Source-local minutes may equal 1440, while effective-local minutes stay below 1440.

The canonical usage file contains 45,197 records and the cache contains 45,196. The single omitted record is id10 at `2024-07-06 00:00`, whose support is `2024-07-05 23:50–24:00`, before id10's original calendar begins on July6. The unchanged extractor does not expand the participant calendar to accommodate earlier support. This source-only boundary record is documented in `source_records_outside_calendar_cache.csv`; every other canonical record from the five sources is represented in the cache.

Across all 500 target/other-date combinations, positive measurement-support overlap is **zero**, as are shared source timestamps between cached target and donor records. There are **{sum(r['donor_source_timestamps_on_target_calendar_date'] for r in boundary)}** donor source timestamps on the target calendar date at the preceding usage donor's midnight endpoint. These touch the target start boundary but have no positive support in the target day. `target_day_boundary_audit.csv` documents each target explicitly.

## Cadence, alignment and gaps

{table(simple_support)}

Mobile-light timestamps span all ten minute-modulo-ten phases. Exact clock equality across dates is therefore a restrictive alignment rule. Distinct real observations within the same nominal ten-minute bin are retained individually; they are not duplicate source tuples and are not aggregated or dropped. Coverage uses unique occupied cadence bins, while primitive values use valid records. `source_minute_mod10_counts.csv` gives exact phase frequencies; `observed_gap_counts.csv` gives within-day consecutive-observation gap frequencies; `daily_support.csv` includes day-level record counts, valid-bin counts, boundary gaps and longest missing runs, including empty days.

The cache holds raw `m_light`/`w_light` values, not canonical clipped or precomputed log1p columns. Usage remains milliseconds, not scaled minutes. Activity retains `activity_unknown` and the extractor's corresponding validity flag. Operation/unit/unit-scale metadata is available for callers, but no transform is applied by this adapter.

## Descriptive 30-minute slot feasibility

A 30-minute slot is compatible with the one- and ten-minute nominal cadences and absorbs mobile-light phase variation. Ten-minute slots would preserve finer clock detail with fewer observations; sixty-minute slots would increase time pooling. These are cadence-based options, not performance-selected methods. The parent experiment owns all method decisions and the protocol freeze.

{table(simple_feasible)}

Counts are distinct donor dates with at least one valid observation in a half-hour slot; multiple donor observations on one date still count as one date. All 24,000 target/slot rows and all/past/future/same-weektype/past-same-weektype counts are in `slot30_support_feasibility.csv`. The >=2-date figures above are descriptive and do not implement an exclusion. They also do not guarantee that a nearest individual observation will meet a separate clock-distance tolerance, or that a retained-target similarity profile will have sufficient overlap; those checks belong to the frozen methods.

## Provenance and reuse

`manifest.json` pins all 12 existing canonical source files, all scientific vendor files, original calendar/input manifests, frozen legacy wrappers, all 500 original target cell files, and every output partition. Inputs were rehashed after the build and again after final cache verification. Historical raw-file hashes are carried from the unchanged preprocessing manifest as inherited lineage; original raw files were not re-read or replaced. `serialized_target_checks.csv` repeats the 500-cell source equality gate after loading saved partitions. No original masks, references, scales, or source files were edited.

See `README.md` for the precise dictionary interface. Consumers must verify partition SHA-256 values and exclude the target participant/calendar-day identity before any donor aggregation. Loading a pickle is appropriate only for this trusted local, hash-pinned cache.
'''
    (HERE/'FEASIBILITY.md').write_text(report,encoding='utf8')
    preprocessor=role('base_inputs')/'leaderboard/vendor/preprocessing/src/sensors/build_all_sensor_aligned.py'
    preprocessing_manifest=role('base_inputs')/'validation/preprocessing/raw_run/preprocessing_run.json'
    expected_preprocessor=json.loads(preprocessing_manifest.read_text(encoding='utf8'))['vendor_sha256']['src/sensors/build_all_sensor_aligned.py']
    if sha256(preprocessor)!=expected_preprocessor:
        raise ValueError('preprocessing source provenance mismatch')
    m['additional_provenance_source_hashes']={str(preprocessor):sha256(preprocessor)}
    for path,h in m['input_hashes'].items():
        if sha256(resolve_input(path))!=h:
            raise ValueError('original input changed during final verification')
    extraction_sources=m.get('extraction_source_hashes',{path:h for path,h in m['source_hashes'].items()
        if path!='finalize_inventory.py'})
    for path,h in extraction_sources.items():
        if sha256(SOURCE/path)!=h:
            raise ValueError('builder source changed during extraction')
    m['extraction_source_hashes']=extraction_sources
    m['source_hashes']={p.name:sha256(p) for p in SOURCE.glob('*.py')}
    m['outputs']={p.name:sha256(p) for p in HERE.iterdir() if p.is_file() and p.name!='manifest.json'}
    m.update(final_inventory_verified_utc=datetime.now(timezone.utc).isoformat(),serialized_target_cells_checked=500,
        zero_positive_crossday_support_overlap=True,inventory_report='FEASIBILITY.md',
        donor_raw_records=int(daily.records.sum()),donor_valid_records=int(daily.valid_records.sum()))
    (HERE/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='complete',partitions=len(m['partitions']),calendar_cells=4265,
        selected_cells_exact_after_serialization=500,raw_records=m['donor_raw_records'],
        valid_records=m['donor_valid_records'],manifest_sha256=sha256(HERE/'manifest.json'))))


if __name__=='__main__':
    main()
