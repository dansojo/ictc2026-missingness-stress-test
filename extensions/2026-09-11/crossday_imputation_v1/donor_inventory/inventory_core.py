"""Raw donor-cache adapter. Scientific extraction is exclusively the pinned legacy API."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[4]
LEGACY=HERE.parents[2]/'2026-09-10/reviewer_extension'
sys.path.insert(0,str(LEGACY))
import run_recovery_v2 as legacy

MINUTE_NS=60*10**9
DAY_NS=1440*MINUTE_NS


def pack_view(view, calendar):
    """Preserve public phase-view raw values and all three temporal coordinates."""
    records=view.records
    definition=legacy.DEFS[view.primitive_name]
    raw=np.array([r.raw_value for r in records],dtype=float)
    times=np.array([r.source_timestamp.value for r in records],dtype=np.int64)
    effective=np.array([r.effective_timestamp.value for r in records],dtype=np.int64)
    start_ns=int(calendar.iloc[0].lifelog_date.value)
    end_ns=int(calendar.iloc[0].sleep_date.value)
    frame=pd.DataFrame({'subject_id':[r.subject_id for r in records],
        'timestamp':[r.source_timestamp for r in records],definition.value_column:raw})
    if definition.validity_column:
        frame[definition.validity_column]=[r.raw_validity for r in records]
    cell=dict(subject_id=calendar.iloc[0].subject_id,
        sensor_day_id=int(calendar.iloc[0].sensor_day_id),primitive=view.primitive_name,
        raw=raw,valid=np.array([r.valid for r in records],dtype=bool),times=times,
        starts=np.array([r.support_start.value for r in records],dtype=np.int64),
        ends=np.array([r.support_end.value for r in records],dtype=np.int64),
        effective_times=effective,local_clock_ns=effective-start_ns,
        source_local_clock_ns=times-start_ns,day_start_ns=start_ns,day_end_ns=end_ns,
        date=calendar.iloc[0].lifelog_date.strftime('%Y-%m-%d'),
        weekday=int(calendar.iloc[0].lifelog_date.dayofweek),
        is_weekend=bool(calendar.iloc[0].lifelog_date.dayofweek>=5),
        raw_frame=frame,calendar=calendar.copy(),
        source_record_digest=view.source_record_digest,
        definition_digest=view.definition_digest,policy_digest=view.policy_digest,
        aligned_frame_digest=view.aligned_frame_digest,
        record_content_digests=tuple(r.content_digest for r in records),
        timestamp_semantics=view.timestamp_semantics,cadence_minutes=view.cadence_minutes,
        expected_epochs=view.expected_epochs,support_minutes=view.support_minutes,
        operation=view.operation,unit=view.unit,unit_scale=view.unit_scale,
        value_column=view.value_column,validity_column=view.validity_column,
        sensor_file=view.sensor_file,interval_boundary_policy=view.interval_boundary_policy)
    if len(times) and (np.diff(times)<=0).any():
        raise ValueError('duplicate or unordered canonical coordinates')
    if end_ns-start_ns!=DAY_NS:
        raise ValueError('unexpected calendar day length')
    if len(effective) and ((effective<start_ns)|(effective>=end_ns)).any():
        raise ValueError('effective coordinate outside day')
    return cell


def prepare_frame(primitive, frame, calendar):
    definition=legacy.DEFS[primitive]
    prepared=legacy.primitives.prepare_primitive_replay(primitive,frame,
        definition.sensor_file,calendar)
    return pack_view(legacy.primitives.export_prepared_phase_view(prepared),calendar)


def target_overlap_count(donor,target):
    """Count actual source support intersecting target's half-open calendar day."""
    if donor['timestamp_semantics']=='instant':
        included=(donor['times']>=target['day_start_ns'])&(donor['times']<target['day_end_ns'])
    else:
        included=(donor['starts']<target['day_end_ns'])&(donor['ends']>target['day_start_ns'])
    return int(included.sum())


def support_summary(cell):
    width=cell['cadence_minutes']*MINUTE_NS
    bins=cell['local_clock_ns']//width
    valid_bins=bins[cell['valid']]
    occupied=np.zeros(cell['expected_epochs'],dtype=bool)
    occupied[np.unique(valid_bins)]=True
    padded=np.r_[True,occupied,True]
    occupied_positions=np.flatnonzero(padded)
    longest=int(np.max(np.diff(occupied_positions)-1))
    deltas=np.diff(cell['times'])/MINUTE_NS
    return dict(records=len(bins),valid_records=int(cell['valid'].sum()),
        observed_epochs=int(occupied.sum()),expected_epochs=cell['expected_epochs'],
        coverage=float(occupied.mean()),
        within_bin_extra_records=int(len(bins)-len(np.unique(bins))),
        within_bin_extra_valid_records=int(len(valid_bins)-len(np.unique(valid_bins))),
        longest_missing_run_epochs=longest,
        longest_missing_run_minutes=longest*cell['cadence_minutes'],
        observed_min_gap_minutes=float(deltas.min()) if len(deltas) else None,
        observed_median_gap_minutes=float(np.median(deltas)) if len(deltas) else None,
        observed_max_gap_minutes=float(deltas.max()) if len(deltas) else None,
        valid_30min_slots=int(len(np.unique(cell['local_clock_ns'][cell['valid']]//(30*MINUTE_NS)))),
        source_midnight_endpoint_records=int((cell['times']==cell['day_end_ns']).sum()),
        support_outside_day_records=int(((cell['starts']<cell['day_start_ns'])|
            (cell['ends']>cell['day_end_ns'])).sum()))


def donor_support_counts(target,pool,slot_minutes=30):
    """Descriptive counts only: each source date contributes at most once per slot."""
    other=[c for c in pool if c['day_start_ns']!=target['day_start_ns']]
    valid=[c for c in other if c['valid'].any()]
    groups={
        'all':valid,
        'past':[c for c in valid if c['day_start_ns']<target['day_start_ns']],
        'future':[c for c in valid if c['day_start_ns']>target['day_start_ns']],
        'same_weektype':[c for c in valid if c['is_weekend']==target['is_weekend']],
        'past_same_weektype':[c for c in valid if c['day_start_ns']<target['day_start_ns']
                              and c['is_weekend']==target['is_weekend']],
        'weekday':[c for c in valid if not c['is_weekend']],
        'weekend':[c for c in valid if c['is_weekend']],
    }
    counts={'other_calendar_dates':len(other),'valid_donor_dates':len(valid),
        'raw_observed_donor_dates':sum(len(c['valid'])>0 for c in other)}
    for name,group in groups.items():
        counts[name+'_valid_donor_dates']=len(group)
    support={}
    for name,group in groups.items():
        n=np.zeros(1440//slot_minutes,dtype=int)
        for c in group:
            unique=np.unique(c['local_clock_ns'][c['valid']]//(slot_minutes*MINUTE_NS))
            n[unique]+=1
        support[name]=n
    rows=[]
    for i in range(1440//slot_minutes):
        row=dict(slot=i,start_minute=i*slot_minutes,end_minute=(i+1)*slot_minutes)
        row.update({name+'_valid_donor_dates':int(n[i]) for name,n in support.items()})
        rows.append(row)
    return counts,rows
