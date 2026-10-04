"""Univariate known-support reconstruction. No reference or outcome input exists."""
import numpy as np


def repair(times_seconds, retained_values, targets, *, binary, method, cap_seconds):
    times = np.asarray(times_seconds)
    values = np.asarray(retained_values, dtype=float)
    targets = np.asarray(targets, dtype=bool)
    if times.ndim != 1 or values.shape != times.shape or targets.shape != times.shape:
        raise ValueError('coordinate/value/target shapes must agree')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('source times must be strictly increasing')
    if np.isfinite(values[targets]).any():
        raise ValueError('deleted values must be hidden from the imputer')
    if method not in ('M1', 'M2') or cap_seconds <= 0:
        raise ValueError('unsupported fixed method/cap')
    donors = np.flatnonzero(np.isfinite(values) & ~targets)
    if binary and not np.isin(values[donors], [0, 1]).all():
        raise ValueError('binary donors must be hard states')
    result = values.copy()
    target_ids = np.flatnonzero(targets)
    if not len(donors):
        return dict(values=result, status='abstained_no_retained_donor',
                    imputed_n=0, fallback_n=0, locf_n=0, unresolved_n=len(target_ids))
    fill = (float(np.count_nonzero(values[donors] == 1) > len(donors)/2)
            if binary else float(values[donors].mean()))
    result[target_ids] = fill
    locf_n = 0
    if method == 'M2' and len(target_ids):
        previous = np.searchsorted(times[donors], times[target_ids], side='left')-1
        candidates = donors[np.maximum(previous, 0)]
        use = (previous >= 0) & ((times[target_ids]-times[candidates]) <= cap_seconds)
        result[target_ids[use]] = values[candidates[use]]
        locf_n = int(use.sum())
    return dict(values=result, status='finite', imputed_n=len(target_ids),
                fallback_n=len(target_ids)-locf_n if method == 'M2' else 0,
                locf_n=locf_n, unresolved_n=0)


def deleted_by_intervals(starts, ends, intervals, semantics):
    starts, ends = np.asarray(starts), np.asarray(ends)
    deleted = np.zeros(len(starts), dtype=bool)
    for left, right in intervals:
        if right <= left:
            raise ValueError('mask interval must have positive width')
        if semantics == 'instant':
            deleted |= (starts >= left) & (starts < right)
        elif semantics == 'interval_end':
            deleted |= (starts < right) & (ends > left)
        else:
            raise ValueError('unknown timestamp semantics')
    return deleted


def feature_value(valid_raw_values, operation):
    values = np.asarray(valid_raw_values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float('nan')
    if operation in ('binary_load', 'activity_load'):
        if not np.isin(values, [0, 1]).all():
            raise ValueError('feature hard-state domain violated')
        return float(values.mean())
    if (values < 0).any():
        raise ValueError('nonnegative raw domain violated')
    if operation == 'additive_load':
        return float(values.sum() * (1.0/60000.0))
    if operation == 'exposure_mean':
        return float(np.log1p(values).mean())
    raise ValueError('unsupported primitive operation')
