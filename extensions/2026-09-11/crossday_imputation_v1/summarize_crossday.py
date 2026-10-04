"""Transparent primitive-specific reporting on one common applicable-method support.

No fitting, imputation, or result selection occurs here. Legacy files are read-only.
"""
from pathlib import Path
from datetime import date
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROSTER = [f'id{i:02d}' for i in range(1, 11)]
PRIMITIVES = ['screen_load_24h', 'phone_activity_load_24h', 'usage_load_24h',
              'mobile_light_exposure_24h', 'wearable_light_exposure_24h']
OLD_METHODS = ['M0', 'M1', 'M2', 'MEDIAN_RAW', 'LINEAR_RAW', 'LOCF_1',
               'LOCF_10', 'LOCF_30', 'LOCF_300', 'LOCF_INF']
NEW_METHODS = ['CD_ALL_DAY', 'CD_TIME_MEAN', 'CD_TIME_MEDIAN',
               'CD_WEEKTYPE_MEAN', 'CD_WEEKTYPE_MEDIAN', 'CD_PAST_TIME_MEAN',
               'CD_PAST_RECENT_MEAN', 'CD_SIMILAR_COPY']
METHODS = OLD_METHODS + NEW_METHODS
NUMERIC_ONLY = {'MEDIAN_RAW', 'LINEAR_RAW', 'CD_TIME_MEDIAN', 'CD_WEEKTYPE_MEDIAN'}
GEOMETRIES = ['contiguous_20pct', 'scattered_random_20pct']
KEYS = ['subject_id', 'sensor_day_id', 'primitive', 'draw', 'geometry']
SEED = 20260910
BOOTSTRAPS = 10000
COMPARISONS = [('M0', m) for m in METHODS if m != 'M0'] + [
    ('CD_ALL_DAY', m) for m in NEW_METHODS] + [('CD_PAST_TIME_MEAN', 'CD_PAST_RECENT_MEAN')]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def applicable_methods(primitive):
    return [m for m in METHODS if not (primitive in PRIMITIVES[:2] and m in NUMERIC_ONLY)]


def common_gains(wide, methods, comparisons):
    common = np.isfinite(wide[methods]).all(axis=1)
    result = {}
    for comparator, method in comparisons:
        applicable = comparator in methods and method in methods
        result[(comparator, method)] = (wide[comparator] - wide[method]).where(common & applicable)
    return common, result


def participant_summary(frame, roster=ROSTER):
    rows = []
    for subject in roster:
        values = frame.loc[frame.subject_id.eq(subject), 'gain'].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append(dict(subject_id=subject, scheduled_keys=len(values), finite_keys=len(finite),
                         excluded_keys=len(values) - len(finite),
                         positive_keys=int((finite > 0).sum()), zero_keys=int((finite == 0).sum()),
                         negative_keys=int((finite < 0).sum()),
                         median_gain=float(np.median(finite)) if len(finite) else float('nan')))
    estimates = np.asarray([r['median_gain'] for r in rows])
    estimates = estimates[np.isfinite(estimates)]
    low = high = float('nan')
    if len(estimates):
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(estimates), size=(BOOTSTRAPS, len(estimates)))
        low, high = np.quantile(np.median(estimates[draws], axis=1), [.025, .975])
    return rows, dict(n_roster=len(roster), n_participants=len(estimates),
        nonfinite_participants=len(roster) - len(estimates),
        median_gain=float(np.median(estimates)) if len(estimates) else float('nan'),
        ci_low=float(low), ci_high=float(high), positive_participants=int((estimates > 0).sum()),
        zero_participants=int((estimates == 0).sum()), negative_participants=int((estimates < 0).sum()),
        finite_keys=sum(r['finite_keys'] for r in rows), scheduled_keys=len(frame),
        positive_keys=sum(r['positive_keys'] for r in rows), zero_keys=sum(r['zero_keys'] for r in rows),
        negative_keys=sum(r['negative_keys'] for r in rows),
        excluded_keys=len(frame) - sum(r['finite_keys'] for r in rows))


def geometry_differences(wide, methods):
    needed = [(g, m) for g in GEOMETRIES for m in methods]
    common = np.isfinite(wide[needed]).all(axis=1)
    return common, {m: (wide[(GEOMETRIES[0], m)] - wide[(GEOMETRIES[1], m)]).where(common)
                    for m in methods}


def fallback_table(frame, group):
    frame = frame.copy()
    na = frame.method_status.eq('not_applicable') if 'method_status' in frame else pd.Series(False, index=frame.index)
    frame['attempted_unresolved_n'] = frame.unresolved_n.where(~na, 0)
    frame['not_applicable_deleted_valid_n'] = frame.deleted_valid_n.where(na, 0)
    columns = ['imputed_n', 'primary_n', 'fallback_n', 'fallback_slot_n',
               'fallback_global_n', 'fallback_target_n', 'unresolved_n',
               'fallback_similarity_unavailable_n', 'fallback_copy_missing_n', 'deleted_valid_n',
               'attempted_unresolved_n', 'not_applicable_deleted_valid_n']
    agg = {name: (name, 'sum') for name in columns if name in frame.columns}
    agg['scheduled_rows'] = ('method', 'size')
    result = frame.groupby(group, dropna=False, sort=True).agg(**agg).reset_index()
    result['attempted_n'] = result.imputed_n + result.attempted_unresolved_n
    for name in ['primary_n', 'fallback_n', 'fallback_slot_n', 'fallback_global_n',
                 'fallback_target_n', 'unresolved_n', 'fallback_similarity_unavailable_n',
                 'fallback_copy_missing_n']:
        if name in result:
            result[name.removesuffix('_n') + '_fraction_of_attempted'] = (
                result['attempted_unresolved_n' if name == 'unresolved_n' else name] / result.attempted_n.replace(0, np.nan))
    result['fallback_fraction_of_imputed'] = result.fallback_n / result.imputed_n.replace(0, np.nan)
    result['fallback_fraction_of_deleted_valid'] = result.fallback_n / result.deleted_valid_n.replace(0, np.nan)
    result['imputed_fraction_of_deleted_valid'] = result.imputed_n / result.deleted_valid_n.replace(0, np.nan)
    return result


def load_verified_run(path, expected_rows):
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf8'))
    if manifest['status'] != 'complete':
        raise ValueError(f'incomplete run: {path}')
    for name, expected in manifest['outputs'].items():
        if sha(path / name) != expected:
            raise ValueError(f'changed run artifact: {path / name}')
    frame = pd.read_parquet(path / 'recovery_rows.parquet')
    if len(frame) != expected_rows or frame.duplicated(KEYS + ['method']).any():
        raise ValueError('row count or duplicate key error')
    return frame, manifest


def effect_tables(frame):
    wide = frame.pivot(index=KEYS, columns='method', values='standardized_error').reset_index()
    rows, participants, support = [], [], []
    for primitive in PRIMITIVES:
        applicable = applicable_methods(primitive)
        for geometry in GEOMETRIES:
            one = wide.loc[wide.primitive.eq(primitive) & wide.geometry.eq(geometry)].copy()
            common, effects = common_gains(one, applicable, COMPARISONS)
            keys = one[KEYS].copy()
            keys['common_finite'] = common
            keys['applicable_methods'] = len(applicable)
            support.append(keys)
            for (comparator, method), gain in effects.items():
                x = pd.DataFrame({'subject_id': one.subject_id, 'gain': gain})
                people, result = participant_summary(x)
                labels = dict(primitive=primitive, geometry=geometry, comparator=comparator,
                              method=method, applicable=method in applicable,
                              support_policy='all_applicable_old_and_new')
                rows.append(dict(**labels, **result))
                participants.extend(dict(**labels, **p) for p in people)
    return pd.DataFrame(rows), pd.DataFrame(participants), pd.concat(support, ignore_index=True)


def geometry_tables(frame):
    index = [k for k in KEYS if k != 'geometry']
    wide = frame.pivot(index=index, columns=['geometry', 'method'], values='standardized_error')
    rows, participants, support = [], [], []
    for primitive in PRIMITIVES:
        one = wide.loc[wide.index.get_level_values('primitive') == primitive]
        applicable = applicable_methods(primitive)
        common, deltas = geometry_differences(one, applicable)
        keys = common.rename('common_finite_both_geometries').reset_index()
        keys['applicable_methods_per_geometry'] = len(applicable)
        support.append(keys)
        for method in METHODS:
            gain = deltas[method] if method in applicable else pd.Series(np.nan, index=one.index)
            x = gain.rename('gain').reset_index()[['subject_id', 'gain']]
            people, result = participant_summary(x)
            labels = dict(primitive=primitive, method=method, applicable=method in applicable,
                          contrast='contiguous_error_minus_random_error',
                          support_policy='all_applicable_methods_and_both_geometries')
            rows.append(dict(**labels, **result))
            participants.extend(dict(**labels, **p) for p in people)
    return pd.DataFrame(rows), pd.DataFrame(participants), pd.concat(support, ignore_index=True)


def donor_tables(new):
    group = ['subject_id', 'primitive', 'geometry', 'method']
    valid = new.loc[new.method_status.eq('finite')]
    donor = valid.groupby(group, sort=True).agg(
        finite_rows=('draw', 'size'), donor_count_min=('donor_count_min', 'min'),
        donor_count_max=('donor_count_max', 'max'),
        median_row_donor_count_median=('donor_count_median', 'median'),
        median_pool_all_days=('available_donor_dates', 'median'),
        median_pool_past_days=('past_donor_dates', 'median'),
        median_pool_future_days=('future_donor_dates', 'median'),
        median_pool_same_weektype_days=('same_weektype_donor_dates', 'median')).reset_index()
    fields = KEYS + ['method_status', 'selected_day_ordinal', 'similarity_common_slots',
        'similarity_distance', 'similarity_eligible_days', 'target_retained_slots',
        'fallback_similarity_unavailable_n', 'fallback_copy_missing_n',
        'copy_clock_max_lag_seconds', 'imputed_n', 'primary_n', 'fallback_n',
        'fallback_slot_n', 'fallback_global_n', 'fallback_target_n', 'unresolved_n',
        'available_donor_dates', 'past_donor_dates', 'future_donor_dates', 'same_weektype_donor_dates']
    ledger = new.loc[new.method.eq('CD_SIMILAR_COPY'), fields].copy()
    ledger['selected_date'] = ledger.selected_day_ordinal.map(
        lambda v: date.fromordinal(int(v)).isoformat() if pd.notna(v) else '')
    ledger['selected'] = ledger.selected_day_ordinal.notna()
    ledger['no_similarity_donor'] = ledger.fallback_similarity_unavailable_n.gt(0)
    ledger['copy_missing_any'] = ledger.fallback_copy_missing_n.gt(0)
    summaries = []
    for grouping in [group[:-1], ['primitive', 'geometry']]:
        z = ledger.groupby(grouping, dropna=False).agg(
            scheduled_rows=('draw', 'size'), selected_rows=('selected', 'sum'),
            no_similarity_donor_rows=('no_similarity_donor', 'sum'),
            copy_missing_any_rows=('copy_missing_any', 'sum'),
            distinct_selected_dates=('selected_day_ordinal', 'nunique'),
            median_similarity_common_slots=('similarity_common_slots', 'median'),
            min_similarity_common_slots=('similarity_common_slots', 'min'),
            median_similarity_distance=('similarity_distance', 'median'),
            max_similarity_distance=('similarity_distance', 'max'),
            median_eligible_days=('similarity_eligible_days', 'median'),
            copy_clock_max_lag_seconds=('copy_clock_max_lag_seconds', 'max'),
            copied_points=('primary_n', 'sum'), fallback_points=('fallback_n', 'sum'),
            fallback_similarity_unavailable_points=('fallback_similarity_unavailable_n', 'sum'),
            fallback_copy_missing_points=('fallback_copy_missing_n', 'sum')).reset_index()
        if 'subject_id' not in grouping:
            z['subject_id'] = 'ALL_PARTICIPANTS_DESCRIPTIVE_COUNTS'
        summaries.append(z)
    return donor, ledger, pd.concat(summaries, ignore_index=True)


def report(summary, geometry, status, output):
    text = ['# Same-person other-date reconstruction: all fixed outcomes', '',
        'Exploratory offline reconstruction, motivated after the earlier same-day results. '
        'These outcomes were not used to choose the eight methods, slot width, recency half-life, '
        'similarity threshold, dates or masks. No predictive model was fitted.', '',
        'The evaluation keeps 10 participants, 500 original target cells, 50 draws and two exact '
        '20% mask geometries. The 500,000 earlier rows remain in their original file; 400,000 '
        'new rows are joined only in memory. Earlier numerical summaries remain authoritative '
        'for their earlier support; the old-method comparisons here use the expanded shared support.', '',
        'Positive gain is comparator standardized absolute feature error minus method error. '
        'Each participant contributes the median across their finite paired days/draws; the '
        'point estimate is the median of participant medians. Intervals resample participants '
        '10,000 times with seed 20260910 (descriptive 95% percentile intervals). All 10 roster '
        'members, including any unavailable member, appear in participant_effects.csv. Repeated '
        'masks do not increase independent n beyond 10.', '',
        'Primary support is the finite intersection of M0 and all applicable old/new methods '
        'within the exact participant/day/primitive/draw/geometry key: 18 methods for numeric '
        'primitives and 14 for binary states. Binary MEDIAN_RAW, LINEAR_RAW, CD_TIME_MEDIAN and '
        'CD_WEEKTYPE_MEDIAN are explicit N/A rows. Geometry residuals additionally intersect '
        'both geometries. There is no pooled intensity primary estimate.', '',
        'All-date methods can use future dates. Both PAST methods use only earlier dates, '
        'including fallback; their retained-target fallback still uses the whole visible target '
        'day, so they are not real-time causal systems. Other evaluation dates can be donors '
        'for one corrupted target-day episode. This does not simulate simultaneous archive-wide missingness.', '',
        '## All primitive gains', '',
        'P/Z/N denotes positive/zero/negative participant medians. N/A retains scheduled keys '
        'but has no finite participant estimate. CSV files also report positive/zero/negative '
        'finite-key counts, using exact numeric zero (no tolerance). All gains, including zero and negative gains, '
        'are shown. CD_ALL_DAY versus itself is intentionally retained as a zero control.', '']
    for geom in GEOMETRIES:
        for comparator in ['M0', 'CD_ALL_DAY', 'CD_PAST_TIME_MEAN']:
            z = summary.loc[summary.geometry.eq(geom) & summary.comparator.eq(comparator)]
            text += [f'### {geom}: comparator {comparator}', '',
                '| Primitive | Method | Median gain | 95% interval | P/Z/N | Finite/scheduled keys |',
                '|---|---|---:|---|---|---|']
            for r in z.itertuples(index=False):
                estimate = f'{r.median_gain:+.6f}' if r.applicable else 'N/A'
                interval = f'[{r.ci_low:+.6f}, {r.ci_high:+.6f}]' if r.applicable else 'N/A'
                text.append(f'| {r.primitive} | {r.method} | {estimate} | {interval} | '
                    f'{r.positive_participants}/{r.zero_participants}/{r.negative_participants} | '
                    f'{r.finite_keys}/{r.scheduled_keys} |')
            text += ['']
    text += ['## Residual geometry contrast', '',
        'Positive residual means contiguous deletion retains greater error than scattered deletion. '
        'The complete 18-method, five-primitive estimates and participant intervals are in '
        'geometry_residual.csv and geometry_participant_effects.csv; N/A state methods remain visible.', '',
        '## Availability, fallback and donor evidence', '',
        'status_counts.csv contains every method status by participant, primitive and geometry; '
        'status_totals.csv gives primitive totals. common_support.csv preserves each of the '
        '50,000 scheduled keys and the primary inclusion flag. Finite/scheduled denominators '
        'are also attached to every comparison and all-person effect.', '',
        'fallback_by_person.csv and fallback_totals.csv retain new-method tier counts and ratios '
        'of summed counts. The attempted denominator is imputed + attempted_unresolved; N/A '
        'rows have no attempted fills, and their scheduled unresolved field is separately '
        'identified by not_applicable_deleted_valid_n. The '
        'deleted-valid denominator is also retained so N/A and structural cases are transparent. '
        'Fallback counts count repeated mask evaluations, not distinct original records. '
        'Legacy fallback columns are reported separately in legacy_fallback_by_person.csv.', '',
        'primary_n records the requested method; fallback_slot_n is unrestricted same-slot '
        'fallback from weektype/copy methods; fallback_global_n is other-date whole-day '
        'fallback; fallback_target_n is retained-target mean/hard-mode fallback. For '
        'CD_ALL_DAY, the other-date whole-day value is its primary tier. PAST_RECENT whole-day '
        'fallback retains the same recency weights. Similarity-unavailable and copy-missing '
        'are fallback trigger counts, not extra mutually exclusive fill tiers.', '',
        'donor_support_by_person.csv summarizes cross-day donor counts actually used for '
        'resolved fills; target-day fallback (zero other-day donors) is excluded from the '
        'within-row nonzero donor-count min/median/max. Copy uses one selected date. '
        'Pool count medians describe repeated rows and are not extra independent samples.', '',
        'similarity_donor_ledger.csv contains every scheduled copy row, selected donor date, '
        'common-slot count, discrepancy, candidate count, copy lag and fallback counts. '
        'similarity_donor_summary.csv summarizes this ledger. Similarity profiles use retained '
        'target data only: mean state fraction, mean raw usage, or mean log1p lux. Copy uses '
        'original raw values within half cadence, with no circular midnight match or warping.', '',
        'Raw lux is filled before mean(log1p); usage milliseconds before sum/60,000; state fills '
        'are hard 0/1 with ties 0. Natural invalid/missing observations stay missing. '
        'No filled value becomes a donor.', '',
        'The machine-readable output manifest pins all report files. Verification results '
        'are in the separate independent audit; this report alone is not a correctness certificate.', '']
    (output / 'RESULTS.md').write_text('\n'.join(text), encoding='utf8')


def run(run_dir, legacy_dir, output):
    new, new_manifest = load_verified_run(run_dir, 400000)
    old, old_manifest = load_verified_run(legacy_dir, 500000)
    if set(new.method.unique()) != set(NEW_METHODS) or set(old.method.unique()) != set(OLD_METHODS):
        raise ValueError('fixed method roster changed')
    shared = KEYS + ['method', 'standardized_error', 'method_status', 'applicable', 'family']
    frame = pd.concat([old[shared], new[shared]], ignore_index=True)
    if frame.duplicated(KEYS + ['method']).any() or sorted(frame.subject_id.unique()) != ROSTER:
        raise ValueError('fixed grid duplicate or roster mismatch')
    summary, people, common = effect_tables(frame)
    geometry, geometry_people, geometry_common = geometry_tables(frame)
    count_frame = frame.assign(finite_error=np.isfinite(frame.standardized_error))
    status_group = ['subject_id', 'primitive', 'geometry', 'method', 'applicable', 'method_status']
    status = count_frame.groupby(status_group, dropna=False).agg(
        scheduled_rows=('draw', 'size'), finite_error_rows=('finite_error', 'sum')).reset_index()
    status_totals = status.groupby(status_group[1:], dropna=False)[
        ['scheduled_rows', 'finite_error_rows']].sum().reset_index()
    donors, ledger, similarity = donor_tables(new)
    legacy_counts = [c for c in ['imputed_n', 'fallback_n', 'unresolved_n', 'deleted_valid_n',
        'locf_n', 'interpolation_n', 'fallback_no_previous_n', 'fallback_cap_exceeded_n',
        'fallback_missing_anchor_n'] if c in old]
    legacy_fallback = old.groupby(['subject_id', 'primitive', 'geometry', 'method', 'method_status'],
        dropna=False)[legacy_counts].sum().reset_index()
    legacy_fallback['fallback_fraction_of_imputed'] = (
        legacy_fallback.fallback_n / legacy_fallback.imputed_n.replace(0, np.nan))
    outputs = dict(summary=summary, participant_effects=people, common_support=common,
        common_support_by_person=common.groupby(['subject_id', 'primitive', 'geometry']).agg(
            scheduled_keys=('draw', 'size'), common_finite_keys=('common_finite', 'sum')).reset_index(),
        geometry_residual=geometry, geometry_participant_effects=geometry_people,
        geometry_common_support=geometry_common, status_counts=status, status_totals=status_totals,
        fallback_by_person=fallback_table(new, ['subject_id', 'primitive', 'geometry', 'method', 'method_status']),
        fallback_totals=fallback_table(new, ['primitive', 'geometry', 'method', 'method_status']),
        legacy_fallback_by_person=legacy_fallback, donor_support_by_person=donors,
        similarity_donor_ledger=ledger, similarity_donor_summary=similarity)
    output.mkdir(parents=True, exist_ok=False)
    for name, values in outputs.items():
        values.to_csv(output / (name + '.csv'), index=False)
    report(summary, geometry, status, output)
    manifest = dict(status='complete', scope='full_primitive_specific',
        legacy_rows=len(old), new_rows=len(new), combined_rows=len(frame),
        methods=METHODS, roster=ROSTER, bootstrap_seed=SEED, bootstrap_draws=BOOTSTRAPS,
        support_policy='per-primitive intersection of all applicable old and new methods',
        legacy_rows_sha256=sha(legacy_dir / 'recovery_rows.parquet'),
        legacy_manifest_sha256=sha(legacy_dir / 'manifest.json'),
        run_manifest_sha256=sha(run_dir / 'manifest.json'),
        frozen_protocol_sha256=sha(HERE / 'FROZEN_PROTOCOL.json'),
        source_sha256=sha(Path(__file__)),
        outputs={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'outputs'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, default=HERE.parent / 'imputation_extension/full_01')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.legacy, args.output)
