"""Audit-only v2: exact binary vote precision; frozen v1 and outcomes unchanged.

Expected raw fills are pointwise calculations from trusted raw donor caches.
Expected reporting values are recalculated directly from immutable result rows.
"""
from pathlib import Path
from datetime import date, datetime
from collections import Counter
from statistics import median
from fractions import Fraction
from decimal import Decimal, localcontext
from functools import lru_cache
import argparse
import hashlib
import json
import math
import pickle
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRE = HERE.parents[1] / '2026-09-10/reviewer_extension/full_m0_01'
PRIOR = HERE.parent / 'imputation_extension/full_01'
ROSTER = [f'id{i:02d}' for i in range(1, 11)]
PRIMITIVES = ['screen_load_24h', 'phone_activity_load_24h', 'usage_load_24h',
              'mobile_light_exposure_24h', 'wearable_light_exposure_24h']
NEW = ['CD_ALL_DAY', 'CD_TIME_MEAN', 'CD_TIME_MEDIAN', 'CD_WEEKTYPE_MEAN',
       'CD_WEEKTYPE_MEDIAN', 'CD_PAST_TIME_MEAN', 'CD_PAST_RECENT_MEAN', 'CD_SIMILAR_COPY']
OLD = ['M0', 'M1', 'M2', 'MEDIAN_RAW', 'LINEAR_RAW', 'LOCF_1', 'LOCF_10',
       'LOCF_30', 'LOCF_300', 'LOCF_INF']
METHODS = OLD + NEW
NUMERIC = {'MEDIAN_RAW', 'LINEAR_RAW', 'CD_TIME_MEDIAN', 'CD_WEEKTYPE_MEDIAN'}
GEOMETRIES = ['contiguous_20pct', 'scattered_random_20pct']
KEY = ['subject_id', 'sensor_day_id', 'primitive', 'draw', 'geometry']
SETTINGS = dict(slot_seconds=1800, min_days=2, recency_half_life_days=7,
                min_similarity_slots=12, min_similarity_fraction=.5)
RTOL = ATOL = 1e-12


def precise_binary_vote(fractions, distances=None):
    """Mathematical hard vote, with no outcome-selected epsilon or soft states."""
    require(len(fractions) > 0, 'nonempty precise vote')
    if distances is None:
        return float(sum(fractions, Fraction(0)) / len(fractions) > Fraction(1, 2))
    require(len(distances) == len(fractions), 'recency fraction/day alignment')
    coefficients = [Fraction(0) for _ in range(7)]
    for value, distance in zip(fractions, distances):
        require(distance > 0, 'strictly past exact recency')
        quotient, remainder = divmod(int(distance), 7)
        coefficients[remainder] += (value - Fraction(1, 2)) * Fraction(1, 2 ** quotient)
    # alpha=2^(-1/7) has degree seven over Q. A rational polynomial of
    # degree <=6 is zero at alpha iff all coefficients are zero: exact tie.
    if all(c == 0 for c in coefficients):
        return 0.
    signs = []
    for precision in [100, 200]:
        with localcontext() as context:
            context.prec = precision
            alpha = Decimal(2) ** (-Decimal(1) / Decimal(7))
            value = Decimal(0)
            for coefficient in reversed(coefficients):
                value = value * alpha + Decimal(coefficient.numerator) / Decimal(coefficient.denominator)
            require(value != 0, 'increase precision to resolve nonzero recency polynomial')
            signs.append(value > 0)
    require(signs[0] == signs[1], '100/200-digit recency sign agreement')
    return float(signs[0])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1048576):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def same_scalar(expected, actual, label, exact=False):
    if pd.isna(expected) and pd.isna(actual):
        return
    if isinstance(expected, (str, bool, np.bool_)) or exact:
        require(expected == actual, f'{label}: expected {expected!r}; actual {actual!r}')
    else:
        require(np.isclose(expected, actual, rtol=RTOL, atol=ATOL),
                f'{label}: expected {expected!r}; actual {actual!r}')


def ordinal(record):
    return date.fromisoformat(record['date']).toordinal()


def independent_clock(record):
    # Derive the alignment axis from raw measurement support, not cached bins.
    starts = np.asarray(record['starts'], dtype=np.int64)
    ends = np.asarray(record['ends'], dtype=np.int64)
    if record['timestamp_semantics'] == 'instant':
        effective = np.asarray(record['times'], dtype=np.int64)
    else:
        effective = starts + (ends - starts) // 2
    require(np.array_equal(effective, record['effective_times']), 'independent effective time')
    local = effective - int(record['day_start_ns'])
    require(np.array_equal(local, record['local_clock_ns']), 'independent local-clock axis')
    require(((local >= 0) & (local < 86400 * 10**9)).all(), 'local day bounds')
    require((np.diff(local) > 0).all(), 'strictly increasing raw clocks')
    return local / 1e9


def plain_profiles(clocks, values, slot_seconds, logged):
    # Separate slices and NumPy mean/ordinary sorted median; no production bins or summaries.
    profiles = {}
    bins = np.floor(clocks / slot_seconds).astype(int)
    for slot in sorted(set(bins.tolist())):
        raw = values[bins == slot]
        profiles[slot] = (float(np.mean(raw)), float(median(raw.tolist())),
                          float(np.mean(np.log1p(raw))) if logged else float(np.mean(raw)))
    return profiles


def independent_repair(records, target, deleted, settings):
    binary = target['primitive'] in PRIMITIVES[:2]
    logged = 'light_exposure' in target['primitive']
    raw = np.asarray(target['raw'], dtype=float)
    valid = np.asarray(target['valid'], dtype=bool)
    deleted = np.asarray(deleted, dtype=bool)
    fill_ids = np.flatnonzero(valid & deleted)
    retained = valid & ~deleted
    target_clock = independent_clock(target)
    target_profile = plain_profiles(target_clock[retained], raw[retained], settings['slot_seconds'], logged)
    retained_mean = float(np.mean(raw[retained])) if retained.any() else float('nan')
    if binary and retained.any():
        retained_mean = float(Fraction(int(np.sum(raw[retained])), int(retained.sum())) > Fraction(1, 2))
    target_ordinal = ordinal(target)
    target_weekend = date.fromordinal(target_ordinal).weekday() >= 5
    archive = []
    seen = set()
    for source in records:
        require(source['subject_id'] == target['subject_id'], 'cross-subject archive entry')
        require(source['primitive'] == target['primitive'], 'cross-primitive archive entry')
        if int(source['day_start_ns']) == int(target['day_start_ns']):
            continue
        day = ordinal(source)
        require(day not in seen and day != target_ordinal, 'duplicate or target archive date')
        seen.add(day)
        starts, ends = np.asarray(source['starts']), np.asarray(source['ends'])
        if source['timestamp_semantics'] == 'instant':
            overlap = (starts >= target['day_start_ns']) & (starts < target['day_end_ns'])
        else:
            overlap = (starts < target['day_end_ns']) & (ends > target['day_start_ns'])
        require(not overlap.any(), 'source measurement support overlaps target date')
        clocks = independent_clock(source)
        mask = np.asarray(source['valid'], bool) & np.isfinite(source['raw'])
        if not mask.any():
            continue
        values = np.asarray(source['raw'], float)[mask]
        require((values >= 0).all(), 'negative raw donor')
        if binary:
            require(np.isin(values, [0., 1.]).all(), 'non-binary donor')
        exact_slot_fractions = {}
        if binary:
            slots = (clocks[mask] // settings['slot_seconds']).astype(int)
            for slot in sorted(set(slots.tolist())):
                points = values[slots == slot]
                exact_slot_fractions[slot] = Fraction(int(np.sum(points)), len(points))
        archive.append(dict(day=day, clocks=clocks[mask], raw=values,
            profile=plain_profiles(clocks[mask], values, settings['slot_seconds'], logged),
            whole=float(np.mean(values)), weekend=date.fromordinal(day).weekday() >= 5,
            exact_slots=exact_slot_fractions,
            exact_whole=Fraction(int(np.sum(values)), len(values)) if binary else None))
    archive.sort(key=lambda a: a['day'])
    pools = {'all': archive,
             'past': [a for a in archive if a['day'] < target_ordinal],
             'week': [a for a in archive if a['weekend'] == target_weekend]}
    pools['recent'] = pools['past']

    @lru_cache(None)
    def aggregate(pool, slot, use_median=False):
        donors = pools[pool]
        usable = [a for a in donors if slot is None or slot in a['profile']]
        if len(usable) < settings['min_days']:
            return float('nan'), len(usable)
        if binary:
            fractions = [a['exact_whole'] if slot is None else a['exact_slots'][slot] for a in usable]
            distances = [target_ordinal - a['day'] for a in usable] if pool == 'recent' else None
            return precise_binary_vote(fractions, distances), len(usable)
        entries = [a['whole'] if slot is None else a['profile'][slot][int(use_median)] for a in usable]
        if use_median:
            value = float(median(entries))
        elif pool == 'recent':
            weights = [2. ** (-(target_ordinal - a['day']) / settings['recency_half_life_days']) for a in usable]
            value = float(sum(w * x for w, x in zip(weights, entries)) / sum(weights))
        else:
            value = float(np.mean(entries))
        return value, len(usable)

    candidates = []
    for a in archive:
        common = sorted(set(target_profile) & set(a['profile']))
        if len(common) < settings['min_similarity_slots'] or len(common) < settings['min_similarity_fraction'] * len(target_profile):
            continue
        distance = float(np.mean([abs(target_profile[s][2] - a['profile'][s][2]) for s in common]))
        rank = (distance, abs(a['day'] - target_ordinal), int(a['day'] > target_ordinal), a['day'])
        candidates.append((rank, a, len(common)))
    selected = sorted(candidates, key=lambda c: c[0])[0] if candidates else None
    result = {}
    for method in NEW:
        value_array = np.where(retained, raw, np.nan)
        n = len(fill_ids)
        stats = dict(values=value_array, status='finite', imputed_n=0, primary_n=0, fallback_n=0,
            fallback_slot_n=0, fallback_global_n=0, fallback_target_n=0, unresolved_n=n,
            donor_count_min=0, donor_count_max=0, donor_count_median=0., selected_day_ordinal=None,
            similarity_common_slots=0, similarity_distance=float('nan'), similarity_eligible_days=len(candidates),
            target_retained_slots=len(target_profile), fallback_similarity_unavailable_n=0,
            fallback_copy_missing_n=0, copy_clock_max_lag_seconds=float('nan'),
            available_donor_dates=len(archive), past_donor_dates=len(pools['past']),
            future_donor_dates=len(archive) - len(pools['past']), same_weektype_donor_dates=len(pools['week']))
        if binary and method.endswith('MEDIAN'):
            stats['status'] = 'not_applicable'
            result[method] = stats
            continue
        if method == 'CD_SIMILAR_COPY':
            if selected:
                stats.update(selected_day_ordinal=selected[1]['day'], similarity_common_slots=selected[2],
                             similarity_distance=selected[0][0])
            else:
                stats['fallback_similarity_unavailable_n'] = n
        counts, lags = [], []
        for target_index in fill_ids:
            clock = target_clock[target_index]
            slot = int(clock // settings['slot_seconds'])
            options = []
            if method == 'CD_ALL_DAY':
                value, count = aggregate('all', None)
                options.append((value, count, 'primary_n'))
            elif method == 'CD_SIMILAR_COPY':
                if selected:
                    a = selected[1]
                    # Full argmin scan independently checks nearest-position/left-tie semantics.
                    gaps = np.abs(a['clocks'] - clock)
                    position = int(np.argmin(gaps))
                    if gaps[position] <= float(target['cadence_minutes']) * 60 / 2:
                        options.append((float(a['raw'][position]), 1, 'primary_n'))
                        lags.append(float(gaps[position]))
                    else:
                        stats['fallback_copy_missing_n'] += 1
                value, count = aggregate('all', slot)
                options.append((value, count, 'fallback_slot_n'))
            else:
                pool = 'week' if method.startswith('CD_WEEKTYPE') else (
                    'recent' if method == 'CD_PAST_RECENT_MEAN' else (
                    'past' if method == 'CD_PAST_TIME_MEAN' else 'all'))
                value, count = aggregate(pool, slot, method.endswith('MEDIAN'))
                options.append((value, count, 'primary_n'))
                if method.startswith('CD_WEEKTYPE'):
                    value, count = aggregate('all', slot, method.endswith('MEDIAN'))
                    options.append((value, count, 'fallback_slot_n'))
            if method != 'CD_ALL_DAY':
                pool = 'recent' if method == 'CD_PAST_RECENT_MEAN' else (
                    'past' if method == 'CD_PAST_TIME_MEAN' else 'all')
                value, count = aggregate(pool, None)
                options.append((value, count, 'fallback_global_n'))
            options.append((retained_mean, 0, 'fallback_target_n'))
            for value, count, tier in options:
                if not math.isfinite(value):
                    continue
                stats['values'][target_index] = float(value > .5) if binary else value
                stats[tier] += 1
                stats['imputed_n'] += 1
                if tier != 'primary_n':
                    stats['fallback_n'] += 1
                if count:
                    counts.append(count)
                break
        stats['unresolved_n'] = n - stats['imputed_n']
        if stats['unresolved_n']:
            stats['status'] = 'abstained_unresolved'
        if counts:
            stats.update(donor_count_min=min(counts), donor_count_max=max(counts), donor_count_median=float(median(counts)))
        if lags:
            stats['copy_clock_max_lag_seconds'] = max(lags)
        result[method] = stats
    return result


def independent_feature(values, primitive):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return float('nan')
    if primitive in PRIMITIVES[:2]:
        require(np.isin(finite, [0., 1.]).all(), 'soft state in independent output')
        return float(np.mean(finite))
    if primitive == 'usage_load_24h':
        return float(np.sum(finite) / 60000.)
    return float(np.mean(np.log1p(finite)))


def verify_manifests(run, summary):
    frozen_path = HERE / 'FROZEN_PROTOCOL.json'
    frozen = json.loads(frozen_path.read_text(encoding='utf8'))
    require(frozen['new_crossday_results_observed'] is False, 'before-new-outcome declaration')
    require(frozen['prior_revision4_results_known'] is True, 'prior-outcome declaration')
    for path, expected in frozen['files'].items():
        require(sha(path) == expected, 'frozen file changed: ' + path)
    old_path = PRIOR / 'recovery_rows.parquet'
    require(sha(old_path) == '90007efab549a4de69e6fdeb9e67fad731e0ca5b49f722b6a0b660043a17645e',
            'archived 500000 row bytes changed')
    manifests = []
    for folder in [run, summary, PRIOR]:
        metadata = json.loads((folder / 'manifest.json').read_text(encoding='utf8'))
        require(metadata['status'] == 'complete', 'incomplete manifest ' + str(folder))
        for name, expected in metadata['outputs'].items():
            require(sha(folder / name) == expected, 'artifact hash ' + str(folder / name))
        manifests.append(metadata)
    r, s, old = manifests
    require(r['frozen_protocol_sha256'] == sha(frozen_path), 'run freeze hash chain')
    require(s['frozen_protocol_sha256'] == sha(frozen_path), 'summary freeze hash chain')
    require(s['run_manifest_sha256'] == sha(run / 'manifest.json'), 'summary run hash chain')
    require(s['legacy_manifest_sha256'] == sha(PRIOR / 'manifest.json'), 'summary legacy manifest hash')
    require(s['legacy_rows_sha256'] == sha(old_path), 'summary legacy rows hash')
    require(s['source_sha256'] == sha(HERE / 'summarize_crossday.py'), 'summary source hash')
    require(datetime.fromisoformat(frozen['frozen_utc']) <= datetime.fromisoformat(r['started_utc']),
            'freeze precedes run start')
    config = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
    require(config['archive_settings'] == SETTINGS, 'fixed archive settings')
    require(config['new_methods'] == NEW, 'fixed new method roster')
    require(config['bootstrap_seed'] == 20260910 and config['bootstrap_draws'] == 10000, 'fixed bootstrap')
    require(r['scope'] == 'full' and r['rows'] == 400000 and r['cells'] == 500, 'full grid execution')
    amendment_path = HERE / 'AUDIT_AMENDMENT_01.json'
    amendment = json.loads(amendment_path.read_text(encoding='utf8'))
    require(amendment['kind'] == 'audit_precision_correction_only', 'audit-only amendment scope')
    require(amendment['production_or_results_changed'] is False, 'original experiment retained')
    require(amendment['original_frozen_protocol_sha256'] == sha(frozen_path), 'amendment links original freeze')
    for path, expected in amendment['files'].items():
        require(sha(path) == expected, 'audit amendment file changed: ' + path)
    return dict(frozen_files=len(frozen['files']), run_artifacts=len(r['outputs']),
                summary_artifacts=len(s['outputs']), archived_artifacts=len(old['outputs']),
                audit_amendment_files=len(amendment['files']), audit_amendment_sha256=sha(amendment_path))


def verify_rows(new, old):
    require(len(new) == 400000 and len(old) == 500000, 'complete row counts')
    require(not new.duplicated(KEY + ['method']).any(), 'unique new row keys')
    require(not old.duplicated(KEY + ['method']).any(), 'unique legacy row keys')
    require(set(new.method) == set(NEW) and set(old.method) == set(OLD), 'method sets')
    require(sorted(new.subject_id.unique()) == ROSTER, 'ten participants')
    require(set(new.draw) == set(range(50)), 'fifty draws')
    require(set(new.primitive) == set(PRIMITIVES), 'five primitives')
    require(set(new.geometry) == set(GEOMETRIES), 'two geometries')
    require(new.groupby(['subject_id', 'primitive']).sensor_day_id.nunique().eq(10).all(), 'ten dates per cell stratum')
    m0 = old.loc[old.method.eq('M0')].set_index(KEY).sort_index()
    base = ['family', 'eligible', 'eligibility_reason', 'original_status', 'reference', 'scale',
        'no_repair_value', 'no_repair_error', 'original_valid_n', 'original_total_n',
        'retained_observed_n', 'deleted_total_n', 'deleted_valid_n', 'original_source_hash',
        'source_record_digest', 'mask_hash']
    for method in NEW:
        one = new.loc[new.method.eq(method)].set_index(KEY).sort_index()
        pd.testing.assert_index_equal(one.index, m0.index, exact=True)
        pd.testing.assert_frame_equal(one[base], m0[base], check_exact=True, check_dtype=True)
    finite = new.method_status.eq('finite')
    require(np.isfinite(new.loc[finite, ['repaired_value', 'standardized_error']]).all().all(), 'finite scores')
    require((new.loc[finite, 'standardized_error'] >= 0).all(), 'nonnegative scored error')
    require(not np.isfinite(new.loc[~finite, 'standardized_error']).any(), 'nonfinite rows not scored')
    expected_error = np.abs(new.loc[finite, 'repaired_value'] - new.loc[finite, 'reference']) / new.loc[finite, 'scale']
    require(np.array_equal(expected_error.to_numpy(), new.loc[finite, 'standardized_error'].to_numpy()), 'all-row error arithmetic')
    applicable = ~(new.primitive.isin(PRIMITIVES[:2]) & new.method.isin(NUMERIC))
    require(new.applicable.eq(applicable).all(), 'method applicability')
    require(new.loc[~applicable, 'method_status'].eq('not_applicable').all(), 'explicit N/A topology')
    structural = applicable & ~new.eligible
    require(new.loc[structural, 'method_status'].eq('structurally_unavailable').all(), 'structural topology')
    require(new.loc[structural, 'deleted_valid_n'].eq(0).all(), 'structural zero-valid deletion')
    for name in ['imputed_n', 'primary_n', 'fallback_n', 'fallback_slot_n', 'fallback_global_n',
                 'fallback_target_n', 'unresolved_n']:
        require(new[name].ge(0).all(), 'nonnegative counts: ' + name)
    require((new.primary_n + new.fallback_n).eq(new.imputed_n).all(), 'primary/fallback partition')
    require((new.fallback_slot_n + new.fallback_global_n + new.fallback_target_n).eq(new.fallback_n).all(),
            'fallback tier partition')
    require((new.imputed_n + new.unresolved_n).eq(new.deleted_valid_n).all(), 'deleted-point accounting')
    require(new.loc[finite, 'imputed_n'].eq(new.loc[finite, 'deleted_valid_n']).all(), 'finite complete restoration')
    require(new.loc[~applicable, 'imputed_n'].eq(0).all(), 'N/A no actual fills')
    require(new.available_donor_dates.eq(new.past_donor_dates + new.future_donor_dates).all(), 'past/future pool partition')
    require(new.same_weektype_donor_dates.le(new.available_donor_dates).all(), 'weektype pool inclusion')
    past = new.method.isin(['CD_PAST_TIME_MEAN', 'CD_PAST_RECENT_MEAN'])
    require(new.loc[past, 'donor_count_max'].le(new.loc[past, 'past_donor_dates']).all(), 'past-only donor count bound')
    selected = new.selected_day_ordinal.notna()
    require(new.loc[selected, 'method'].eq('CD_SIMILAR_COPY').all(), 'selected date only for copy')
    require(new.loc[selected, 'similarity_common_slots'].ge(12).all(), 'minimum similarity slots')
    require(new.loc[selected, 'similarity_common_slots'].ge(.5 * new.loc[selected, 'target_retained_slots']).all(),
            'minimum target-profile overlap')
    require(new.copy_clock_max_lag_seconds.dropna().ge(0).all(), 'copy nonnegative lag')
    lag = new.copy_clock_max_lag_seconds.notna()
    require(new.loc[lag, 'copy_clock_max_lag_seconds'].le(new.loc[lag, 'cadence_minutes'] * 30).all(), 'copy half-cadence limit')
    targets = pd.to_datetime(new.loc[selected, 'target_date']).map(lambda t: t.toordinal())
    require(new.loc[selected, 'selected_day_ordinal'].ne(targets).all(), 'copy target date excluded')
    return dict(new_rows=len(new), archived_rows=len(old), exact_inherited_M0_rows=len(new),
                inherited_fields=base, status_counts=new.method_status.value_counts().to_dict())


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def independent_effect(gains, subjects):
    people = []
    all_finite = []
    for subject in ROSTER:
        values = np.asarray(gains)[np.asarray(subjects) == subject]
        finite = values[np.isfinite(values)]
        all_finite.extend(finite.tolist())
        people.append(dict(subject_id=subject, scheduled_keys=len(values), finite_keys=len(finite),
            excluded_keys=len(values) - len(finite), positive_keys=int(np.sum(finite > 0)),
            zero_keys=int(np.sum(finite == 0)), negative_keys=int(np.sum(finite < 0)),
            median_gain=float(np.median(finite)) if len(finite) else np.nan))
    values = [p['median_gain'] for p in people if np.isfinite(p['median_gain'])]
    bounds = [np.nan, np.nan]
    if values:
        rng = np.random.default_rng(20260910)
        ids = rng.integers(0, len(values), size=(10000, len(values)))
        bootstrap_medians = np.median(np.asarray(values)[ids], axis=1)
        bounds = np.quantile(bootstrap_medians, [.025, .975])
    f = np.asarray(all_finite)
    result = dict(n_roster=10, n_participants=len(values), nonfinite_participants=10 - len(values),
        median_gain=float(np.median(values)) if values else np.nan, ci_low=float(bounds[0]), ci_high=float(bounds[1]),
        positive_participants=sum(v > 0 for v in values), zero_participants=sum(v == 0 for v in values),
        negative_participants=sum(v < 0 for v in values), positive_keys=int(np.sum(f > 0)),
        zero_keys=int(np.sum(f == 0)), negative_keys=int(np.sum(f < 0)),
        finite_keys=len(f), scheduled_keys=len(gains), excluded_keys=len(gains) - len(f))
    return people, result


def check_effect_row(actual, expected, label):
    for key, value in expected.items():
        same_scalar(value, actual[key], label + '/' + key, exact=True)


def verify_effects(new, old, folder):
    columns = KEY + ['method', 'standardized_error']
    combined = pd.concat([old[columns], new[columns]], ignore_index=True)
    wide = combined.set_index(KEY + ['method']).standardized_error.unstack('method')
    effects = read_csv(folder / 'summary.csv')
    people = read_csv(folder / 'participant_effects.csv')
    support = read_csv(folder / 'common_support.csv').set_index(KEY).sort_index()
    person_index = people.set_index(['primitive', 'geometry', 'comparator', 'method', 'subject_id'])
    comparisons = [('M0', m) for m in METHODS if m != 'M0'] + [('CD_ALL_DAY', m) for m in NEW] + [
        ('CD_PAST_TIME_MEAN', 'CD_PAST_RECENT_MEAN')]
    require(len(effects) == 260 and len(people) == 2600 and len(support) == 50000, 'effect/support output topology')
    expected_labels = {(p, g, c, m) for p in PRIMITIVES for g in GEOMETRIES for c, m in comparisons}
    labels = ['primitive', 'geometry', 'comparator', 'method']
    require(not effects.duplicated(labels).any(), 'unique summary rows')
    require(set(effects[labels].itertuples(index=False, name=None)) == expected_labels, 'all fixed comparisons')
    checked = 0
    for row in effects.to_dict('records'):
        p, g = row['primitive'], row['geometry']
        one = wide.loc[(wide.index.get_level_values('primitive') == p) & (wide.index.get_level_values('geometry') == g)]
        methods = [m for m in METHODS if p not in PRIMITIVES[:2] or m not in NUMERIC]
        common = np.all(np.isfinite(one[methods].to_numpy()), axis=1)
        saved = support.reindex(one.index)
        require(np.array_equal(saved.common_finite.to_numpy(), common), 'primary common-key support')
        require(saved.applicable_methods.eq(len(methods)).all(), 'applicable method denominator')
        applicable = row['method'] in methods
        require(row['applicable'] == applicable, 'summary N/A label')
        gains = one[row['comparator']].to_numpy() - one[row['method']].to_numpy()
        gains[~common if applicable else np.ones(len(gains), bool)] = np.nan
        expected_people, result = independent_effect(gains, one.index.get_level_values('subject_id'))
        check_effect_row(row, result, str(tuple(row[k] for k in labels)))
        for person in expected_people:
            index = (p, g, row['comparator'], row['method'], person['subject_id'])
            actual = person_index.loc[index]
            check_effect_row(actual, {k: v for k, v in person.items() if k != 'subject_id'}, str(index))
        checked += 1
    geometry = read_csv(folder / 'geometry_residual.csv')
    geom_people = read_csv(folder / 'geometry_participant_effects.csv').set_index(['primitive', 'method', 'subject_id'])
    no_geometry = [k for k in KEY if k != 'geometry']
    geom_support = read_csv(folder / 'geometry_common_support.csv').set_index(no_geometry)
    geometry_wide = combined.set_index(no_geometry + ['geometry', 'method']).standardized_error.unstack(['geometry', 'method'])
    require(len(geometry) == 90 and len(geom_people) == 900 and len(geom_support) == 25000, 'geometry output topology')
    require(not geometry.duplicated(['primitive', 'method']).any(), 'unique geometry summaries')
    for row in geometry.to_dict('records'):
        p, m = row['primitive'], row['method']
        one = geometry_wide.loc[geometry_wide.index.get_level_values('primitive') == p]
        methods = [a for a in METHODS if p not in PRIMITIVES[:2] or a not in NUMERIC]
        common = np.all(np.isfinite(one[[(g, a) for g in GEOMETRIES for a in methods]].to_numpy()), axis=1)
        saved = geom_support.reindex(one.index)
        require(np.array_equal(saved.common_finite_both_geometries.to_numpy(), common), 'geometry common support')
        applicable = m in methods
        require(row['applicable'] == applicable, 'geometry applicable')
        gains = one[(GEOMETRIES[0], m)].to_numpy() - one[(GEOMETRIES[1], m)].to_numpy()
        gains[~common if applicable else np.ones(len(gains), bool)] = np.nan
        expected_people, result = independent_effect(gains, one.index.get_level_values('subject_id'))
        check_effect_row(row, result, str((p, m)))
        for person in expected_people:
            check_effect_row(geom_people.loc[(p, m, person['subject_id'])],
                {k: v for k, v in person.items() if k != 'subject_id'}, str((p, m, person['subject_id'])))
    return dict(independent_comparison_rows=checked, independent_comparison_participant_rows=len(people),
                independent_geometry_rows=len(geometry), independent_geometry_participant_rows=len(geom_people),
                independent_bootstrap_intervals=checked + len(geometry), point_CI_comparison='exact float equality')


def verify_counts(new, old, folder):
    shared = KEY + ['method', 'method_status', 'applicable', 'standardized_error']
    combined = pd.concat([new[shared], old[shared]], ignore_index=True)
    for file, group in [('status_counts.csv', ['subject_id', 'primitive', 'geometry', 'method', 'applicable', 'method_status']),
                        ('status_totals.csv', ['primitive', 'geometry', 'method', 'applicable', 'method_status'])]:
        actual = read_csv(folder / file).set_index(group)
        expected_groups = combined.groupby(group, dropna=False)
        require(len(actual) == len(expected_groups), 'status group count')
        for key, one in expected_groups:
            saved = actual.loc[key]
            require(saved.scheduled_rows == len(one), 'status scheduled count')
            require(saved.finite_error_rows == np.isfinite(one.standardized_error).sum(), 'status finite count')
    count_names = ['imputed_n', 'primary_n', 'fallback_n', 'fallback_slot_n', 'fallback_global_n',
        'fallback_target_n', 'unresolved_n', 'fallback_similarity_unavailable_n',
        'fallback_copy_missing_n', 'deleted_valid_n']
    checked = 0
    for file, group in [('fallback_by_person.csv', ['subject_id', 'primitive', 'geometry', 'method', 'method_status']),
                        ('fallback_totals.csv', ['primitive', 'geometry', 'method', 'method_status'])]:
        actual = read_csv(folder / file).set_index(group)
        require(len(actual) == len(new.groupby(group)), 'fallback grouping topology')
        for key, one in new.groupby(group):
            saved = actual.loc[key]
            totals = {name: int(np.sum(one[name].to_numpy())) for name in count_names}
            na = one.method_status.eq('not_applicable').to_numpy()
            totals['attempted_unresolved_n'] = int(np.sum(one.unresolved_n.to_numpy()[~na]))
            totals['not_applicable_deleted_valid_n'] = int(np.sum(one.deleted_valid_n.to_numpy()[na]))
            totals['attempted_n'] = totals['imputed_n'] + totals['attempted_unresolved_n']
            totals['scheduled_rows'] = len(one)
            for name, value in totals.items():
                same_scalar(value, saved[name], file + str(key) + name, exact=True)
            for name in count_names:
                fraction = name.removesuffix('_n') + '_fraction_of_attempted'
                if fraction in saved:
                    numerator = totals['attempted_unresolved_n'] if name == 'unresolved_n' else totals[name]
                    expected = numerator / totals['attempted_n'] if totals['attempted_n'] else np.nan
                    same_scalar(expected, saved[fraction], file + fraction)
            for numerator, denominator, fraction in [('fallback_n', 'imputed_n', 'fallback_fraction_of_imputed'),
                ('fallback_n', 'deleted_valid_n', 'fallback_fraction_of_deleted_valid'),
                ('imputed_n', 'deleted_valid_n', 'imputed_fraction_of_deleted_valid')]:
                expected = totals[numerator] / totals[denominator] if totals[denominator] else np.nan
                same_scalar(expected, saved[fraction], file + fraction)
            checked += 1
    ledger = read_csv(folder / 'similarity_donor_ledger.csv').set_index(KEY).sort_index()
    copy = new.loc[new.method.eq('CD_SIMILAR_COPY')].set_index(KEY).sort_index()
    pd.testing.assert_index_equal(ledger.index, copy.index)
    source_columns = [c for c in ledger.columns if c in copy.columns]
    for field in source_columns:
        pd.testing.assert_series_equal(ledger[field], copy[field], check_exact=True, check_dtype=False)
    for row in ledger.itertuples():
        expected = date.fromordinal(int(row.selected_day_ordinal)).isoformat() if pd.notna(row.selected_day_ordinal) else np.nan
        same_scalar(expected, row.selected_date, 'selected date label', exact=True)
        require(row.selected == pd.notna(row.selected_day_ordinal), 'selected flag')
        require(row.no_similarity_donor == (row.fallback_similarity_unavailable_n > 0), 'similarity unavailable flag')
        require(row.copy_missing_any == (row.fallback_copy_missing_n > 0), 'copy missing flag')
    common = read_csv(folder / 'common_support.csv')
    common_person = read_csv(folder / 'common_support_by_person.csv').set_index(['subject_id', 'primitive', 'geometry'])
    for key, one in common.groupby(['subject_id', 'primitive', 'geometry']):
        saved = common_person.loc[key]
        require(saved.scheduled_keys == len(one), 'common-person scheduled denominator')
        require(saved.common_finite_keys == one.common_finite.sum(), 'common-person finite denominator')
    old_group = ['subject_id', 'primitive', 'geometry', 'method', 'method_status']
    old_fallback = read_csv(folder / 'legacy_fallback_by_person.csv').set_index(old_group)
    for key, one in old.groupby(old_group):
        saved = old_fallback.loc[key]
        for name in [c for c in saved.index if c.endswith('_n')]:
            same_scalar(int(one[name].sum()), saved[name], 'legacy fallback ' + name, exact=True)
        numerator, denominator = int(one.fallback_n.sum()), int(one.imputed_n.sum())
        same_scalar(numerator / denominator if denominator else np.nan,
                    saved.fallback_fraction_of_imputed, 'legacy fallback fraction')
    donor_group = ['subject_id', 'primitive', 'geometry', 'method']
    donors = read_csv(folder / 'donor_support_by_person.csv').set_index(donor_group)
    donor_fields = {'donor_count_min': ('donor_count_min', 'min'),
        'donor_count_max': ('donor_count_max', 'max'),
        'median_row_donor_count_median': ('donor_count_median', 'median'),
        'median_pool_all_days': ('available_donor_dates', 'median'),
        'median_pool_past_days': ('past_donor_dates', 'median'),
        'median_pool_future_days': ('future_donor_dates', 'median'),
        'median_pool_same_weektype_days': ('same_weektype_donor_dates', 'median')}
    for key, one in new.loc[new.method_status.eq('finite')].groupby(donor_group):
        saved = donors.loc[key]
        require(saved.finite_rows == len(one), 'donor summary denominator')
        for label, (source, operation) in donor_fields.items():
            values = one[source].to_numpy()
            value = {'min': np.min, 'max': np.max, 'median': np.median}[operation](values)
            same_scalar(float(value), saved[label], 'donor summary ' + label, exact=True)
    copy_summary = read_csv(folder / 'similarity_donor_summary.csv').set_index(['subject_id', 'primitive', 'geometry'])
    copy_flat = ledger.reset_index()
    copy_fields = {
        'selected_rows': ('selected', 'sum'), 'no_similarity_donor_rows': ('no_similarity_donor', 'sum'),
        'copy_missing_any_rows': ('copy_missing_any', 'sum'),
        'distinct_selected_dates': ('selected_day_ordinal', 'nunique'),
        'median_similarity_common_slots': ('similarity_common_slots', 'median'),
        'min_similarity_common_slots': ('similarity_common_slots', 'min'),
        'median_similarity_distance': ('similarity_distance', 'median'),
        'max_similarity_distance': ('similarity_distance', 'max'),
        'median_eligible_days': ('similarity_eligible_days', 'median'),
        'copy_clock_max_lag_seconds': ('copy_clock_max_lag_seconds', 'max'),
        'copied_points': ('primary_n', 'sum'), 'fallback_points': ('fallback_n', 'sum'),
        'fallback_similarity_unavailable_points': ('fallback_similarity_unavailable_n', 'sum'),
        'fallback_copy_missing_points': ('fallback_copy_missing_n', 'sum')}
    copy_checked = 0
    for group in [['subject_id', 'primitive', 'geometry'], ['primitive', 'geometry']]:
        for key, one in copy_flat.groupby(group):
            row_key = key if len(group) == 3 else ('ALL_PARTICIPANTS_DESCRIPTIVE_COUNTS', *key)
            saved = copy_summary.loc[row_key]
            require(saved.scheduled_rows == len(one), 'copy summary denominator')
            for label, (source, operation) in copy_fields.items():
                values = one[source].dropna().to_numpy()
                if operation == 'nunique':
                    value = len(set(values.tolist()))
                elif operation == 'sum':
                    value = int(np.sum(values))
                elif len(values):
                    value = float({'median': np.median, 'min': np.min, 'max': np.max}[operation](values))
                else:
                    value = np.nan
                same_scalar(value, saved[label], 'copy summary ' + label, exact=True)
            copy_checked += 1
    return dict(independent_fallback_groups=checked, similarity_ledger_rows_exact=len(ledger),
                status_tables_independently_checked=True, independent_donor_summary_rows=len(donors),
                independent_copy_summary_rows=copy_checked, legacy_fallback_groups=len(old_fallback))


def verify_raw(new, run, output):
    index = new.set_index(KEY + ['method'])
    paths = sorted((PRE / 'cells').glob('*.pkl'))
    require(len(paths) == 500, 'all500 raw audit cells')
    cache_key = None
    partition = None
    rows = []
    total_points = exact_points = 0
    max_difference = max_feature_difference = 0.
    expected_audit_keys = []
    for number, path in enumerate(paths):
        with path.open('rb') as stream:
            cell = pickle.load(stream)
        key = (cell['subject_id'], cell['primitive'])
        if cache_key != key:
            with (HERE / 'donor_inventory/cache' / ('__'.join(key) + '.pkl')).open('rb') as stream:
                partition = pickle.load(stream)
            cache_key = key
        target = partition[(cell['subject_id'], cell['primitive'], cell['sensor_day_id'])]
        for name in ['raw', 'valid', 'times']:
            require(np.array_equal(cell[name], target[name], equal_nan=True), 'target raw cache identity ' + name)
        with (run / 'audit_raw' / path.name).open('rb') as stream:
            actual_fills = pickle.load(stream)
        expected_cell_keys = set()
        for mask in cell['masks']:
            if mask['draw'] != 0 or not mask['eligible']:
                continue
            calculation = independent_repair(list(partition.values()), target, mask['deleted'], SETTINGS)
            ids = np.flatnonzero(cell['valid'] & mask['deleted'])
            for method in NEW:
                if cell['primitive'] in PRIMITIVES[:2] and method in NUMERIC:
                    continue
                audit_key = (0, mask['geometry'], method)
                expected_cell_keys.add(audit_key)
                row_key = (cell['subject_id'], cell['sensor_day_id'], cell['primitive'], 0, mask['geometry'], method)
                expected_audit_keys.append(row_key)
                actual = actual_fills[audit_key]
                expected = calculation[method]
                require(np.array_equal(ids, actual['target_indices']), 'audit target coordinates')
                truth = expected['values'][ids]
                observed = np.asarray(actual['filled_values'])
                require(truth.shape == observed.shape, 'raw audit shape')
                require(np.array_equal(np.isfinite(truth), np.isfinite(observed)), 'raw finite support')
                require(np.allclose(truth, observed, rtol=RTOL, atol=ATOL, equal_nan=True), 'independent raw fill ' + str(row_key))
                if cell['primitive'] in PRIMITIVES[:2]:
                    require(np.array_equal(truth, observed, equal_nan=True), 'binary fills must agree exactly')
                finite = np.isfinite(truth)
                difference = float(np.max(np.abs(truth[finite] - observed[finite]))) if finite.any() else 0.
                exact = int(np.sum((truth == observed) | (np.isnan(truth) & np.isnan(observed))))
                total_points += len(truth)
                exact_points += exact
                max_difference = max(max_difference, difference)
                saved = index.loc[row_key]
                for name, value in expected.items():
                    if name == 'values':
                        continue
                    column = 'method_status' if name == 'status' else name
                    same_scalar(value, saved[column], str(row_key) + '/' + column,
                                exact=name not in ['similarity_distance', 'copy_clock_max_lag_seconds'])
                value = independent_feature(expected['values'], cell['primitive']) if expected['status'] == 'finite' else np.nan
                same_scalar(value, saved.repaired_value, str(row_key) + '/feature')
                error = abs(value - mask['reference']) / mask['scale'] if np.isfinite(value) else np.nan
                same_scalar(error, saved.standardized_error, str(row_key) + '/error')
                feature_diff = abs(value - saved.repaired_value) if np.isfinite(value) else 0.
                max_feature_difference = max(max_feature_difference, feature_diff)
                rows.append(dict(zip(KEY + ['method'], row_key), audited_points=len(truth),
                    exact_points=exact, raw_max_absdiff=difference, feature_absdiff=feature_diff,
                    status=expected['status'], selected_day_ordinal=expected['selected_day_ordinal']))
        require(set(actual_fills) == expected_cell_keys, 'exact raw audit entry coverage ' + path.name)
        if number % 25 == 0 or number == len(paths) - 1:
            print(json.dumps(dict(audited_cells=number + 1, raw_points=total_points,
                                 maximum_raw_absdiff=max_difference)), flush=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(output / 'independent_raw_checks.csv', index=False)
    public = read_csv(run / 'public_checks.csv')
    expected_keys = pd.DataFrame(expected_audit_keys, columns=KEY + ['method']).sort_values(KEY + ['method']).reset_index(drop=True)
    actual_keys = public[KEY + ['method']].sort_values(KEY + ['method']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected_keys, actual_keys, check_dtype=False, check_exact=True)
    require(public.exact.all() and public.absdiff.eq(0).all(), 'all public primitive comparisons exact')
    require(public.public_status.eq('observed').all(), 'public observed statuses')
    require(public.public_observed_epochs.eq(public.expected_observed_epochs).all(), 'public coverage checks')
    return dict(independent_raw_cells=500, independent_raw_draws=[0], independent_raw_geometries=GEOMETRIES,
                independent_raw_method_rows=len(audit), independent_raw_values=total_points,
                independent_raw_exact_values=exact_points, independent_raw_max_absdiff=max_difference,
                independent_feature_max_absdiff=max_feature_difference, public_checks=len(public),
                public_exact=len(public), rtol=RTOL, atol=ATOL)


def run_audit(run, summary, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(status='running', verifier_sha256=sha(Path(__file__)),
                  independence='No production imputation/runner/summary functions imported or called',
                  audit_version='v2: exact rational binary votes and exact recency ties with 100/200-digit nonzero sign agreement',
                  first_audit_failure_preserved=str(HERE / 'independent_audit_01/verification.json'))
    (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    try:
        result['hashes'] = verify_manifests(run, summary)
        print('All frozen and output hashes match.', flush=True)
        new = pd.read_parquet(run / 'recovery_rows.parquet')
        old = pd.read_parquet(PRIOR / 'recovery_rows.parquet')
        result.update(verify_rows(new, old))
        result.update(verify_effects(new, old, summary))
        result.update(verify_counts(new, old, summary))
        result.update(verify_raw(new, run, output))
        # Re-read every pin after the audit, guarding accidental mutation during verification.
        require(result['hashes'] == verify_manifests(run, summary), 'post-audit manifest topology')
        result.update(status='passed', seconds=time.perf_counter() - started,
            run_manifest_sha256=sha(run / 'manifest.json'), summary_manifest_sha256=sha(summary / 'manifest.json'),
            frozen_protocol_sha256=sha(HERE / 'FROZEN_PROTOCOL.json'),
            audit_outputs={'independent_raw_checks.csv': sha(output / 'independent_raw_checks.csv')},
            limitations='All new keys/metadata/status/counts and summaries cover all 50 draws; independent raw '
                        'point calculations cover only fixed draw 0 of every target cell and both geometries. '
                        'Numerical raw checks use declared tolerance, binary raw values and inherited metadata '
                        'use exact equality. This does not establish effectiveness on natural missingness.')
    except BaseException as exc:
        result.update(status='failed', seconds=time.perf_counter() - started, error=repr(exc))
        (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        raise
    (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.run, args.summary, args.output)
