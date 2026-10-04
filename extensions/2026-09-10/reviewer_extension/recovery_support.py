"""Coverage counts unique cadence bins, while the feature uses valid records."""
import numpy as np

def cadence_epochs(source_times_ns, included, start_ns, end_ns, cadence_minutes, midpoint_minutes=0):
    effective=np.asarray(source_times_ns)[np.asarray(included,dtype=bool)]-int(midpoint_minutes*60*10**9)
    effective=effective[(effective>=start_ns)&(effective<end_ns)]
    return int(len(np.unique((effective-start_ns)//int(cadence_minutes*60*10**9))))

def cell_epochs(cell, included, definition):
    row=cell['calendar'].iloc[0]
    return cadence_epochs(cell['times'],included,row.lifelog_date.value,row.sleep_date.value,
                          definition.cadence_minutes,
                          definition.support_minutes/2 if definition.timestamp_semantics=='interval_end' else 0)
