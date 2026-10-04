"""Independent saved-state audit. This program never calls a model fit method."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import pickle
import time
import traceback
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
EXPECTED_PROTOCOL = '34b42d076d98cf9a3ea1d2c8fed49d126af2a6fe08e2fa2eec0894c42d91faa0'
EXPECTED_INPUT = 'a4f74d30debf58bec47fa3906d14977efd834f08de8dfe3a5e8205090b843c7f'
SEED = 20260911
DRAWS = 50
BOOTSTRAPS = 10000
GEOMETRIES = ('scattered_random_20pct', 'contiguous_20pct')
FEATURES = {'StudentLife': ['active_fraction', 'mean_activity_code'],
            'ExtraSensory': ['magnitude_mean', 'magnitude_std', 'magnitude_median', 'magnitude_p90']}
METRICS = ['log_loss', 'brier', 'auc', 'balanced_accuracy', 'accuracy',
           'probability_sd', 'predicted_positive_fraction', 'drift', 'flip_rate']
RTOL, ATOL = 1e-10, 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def json_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def independent_seed(*identity):
    # Documented root/identity JSON, independently serialized here.
    return int.from_bytes(bytes.fromhex(json_digest([SEED, *identity]))[:8], 'big')


def numeric(actual, expected, label, statistics):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape, f'{label}: shape {a.shape} versus {b.shape}')
    unavailable = np.isnan(a) & np.isnan(b)
    exact = (a == b) | unavailable
    good = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
    require(good.all(), f'{label}: numerical mismatch; maximum difference '
            f'{float(np.max(np.where(np.isfinite(a-b), np.abs(a-b), np.inf))) if a.size else 0}')
    finite = np.isfinite(a) & np.isfinite(b)
    delta = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0
    item = statistics.setdefault(label, {'values': 0, 'exact_numeric_values': 0, 'matching_unavailable_values': 0, 'max_absolute_difference': 0.0})
    item['values'] += int(a.size)
    item['exact_numeric_values'] += int(exact.sum())
    item['matching_unavailable_values'] += int(unavailable.sum())
    item['max_absolute_difference'] = max(item['max_absolute_difference'], delta)


def independent_features(dataset, values):
    v = np.asarray(values, dtype=float)
    require(v.ndim == 1 and len(v) >= 2 and np.isfinite(v).all(), 'Invalid sensor values')
    if dataset == 'StudentLife':
        require(np.isin(v, [0, 1, 2]).all(), 'Unknown activity code')
        counts = np.bincount(v.astype(np.int64), minlength=3)
        n = len(v)
        return np.array([(counts[1]+counts[2])/n, (counts[1]+2*counts[2])/n], dtype=float)
    require(dataset == 'ExtraSensory' and (v >= 0).all(), 'Invalid magnitude dataset/domain')
    n = len(v)
    mean = math.fsum(v)/n
    spread = math.sqrt(math.fsum((float(x)-mean)**2 for x in v)/n)
    ordered = np.sort(v)
    median = float(ordered[n//2]) if n % 2 else float((ordered[n//2-1]+ordered[n//2])/2)
    pos = .9*(n-1)
    left = int(math.floor(pos)); right = int(math.ceil(pos))
    p90 = float(ordered[left]+(ordered[right]-ordered[left])*(pos-left))
    return np.array([mean, spread, median, p90])


def independent_masks(n, dataset, participant, window_id, draw):
    k = n//5
    random_rng = np.random.Generator(np.random.PCG64(independent_seed(dataset, participant, window_id, draw, GEOMETRIES[0])))
    selected = np.sort(random_rng.choice(n, size=k, replace=False)).astype(np.int64)
    block_rng = np.random.Generator(np.random.PCG64(independent_seed(dataset, participant, window_id, draw, GEOMETRIES[1])))
    start = int(block_rng.integers(0, n-k+1))
    return selected, np.arange(start, start+k, dtype=np.int64)


def independent_fingerprint(fold):
    s, m = fold['scaler'], fold['model']
    return json_digest({'mean': s.mean_.tolist(), 'scale': s.scale_.tolist(), 'var': s.var_.tolist(),
        'coef': m.coef_.tolist(), 'intercept': m.intercept_.tolist(), 'classes': m.classes_.tolist(), 'prior': float(fold['prior'])})


def manual_probabilities(fold, matrix):
    z = ((np.asarray(matrix)-fold['scaler'].mean_)/fold['scaler'].scale_) @ fold['model'].coef_[0] + fold['model'].intercept_[0]
    answer = np.empty_like(z)
    positive = z >= 0
    answer[positive] = 1/(1+np.exp(-z[positive]))
    exp_z = np.exp(z[~positive]); answer[~positive] = exp_z/(1+exp_z)
    return answer


def independent_metrics(y, p, observed):
    y, p, observed = np.asarray(y, int), np.asarray(p, float), np.asarray(observed, float)
    require(y.ndim == 1 and p.shape == y.shape and observed.shape == y.shape, 'Misaligned metric vectors')
    require(np.isin(y, [0, 1]).all() and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid metric data')
    n = len(y); positives = int(np.count_nonzero(y)); negatives = n-positives
    q = np.clip(p, np.finfo(float).eps, 1-np.finfo(float).eps)
    terms = np.where(y == 1, -np.log(q), -np.log1p(-q))
    decision = p >= .5
    if positives and negatives:
        ranks = rankdata(p, method='average')
        auc = (float(ranks[y == 1].sum())-positives*(positives+1)/2)/(positives*negatives)
        ba = (np.count_nonzero(decision & (y == 1))/positives + np.count_nonzero(~decision & (y == 0))/negatives)/2
    else:
        auc = ba = float('nan')
    center = float(p.mean())
    return dict(log_loss=float(terms.sum()/n), brier=float(np.dot(p-y, p-y)/n), auc=auc,
        balanced_accuracy=float(ba), accuracy=float(np.count_nonzero(decision == y)/n),
        probability_sd=float(np.sqrt(np.dot(p-center, p-center)/n)),
        predicted_positive_fraction=float(np.count_nonzero(decision)/n),
        drift=float(np.abs(p-observed).sum()/n),
        flip_rate=float(np.count_nonzero(decision != (observed >= .5))/n))


def own_median(values):
    ordered = np.sort(np.asarray(values, float)); n = len(ordered)
    return float(ordered[n//2]) if n % 2 else float((ordered[n//2-1]+ordered[n//2])/2)


def independent_bootstrap(values, identity):
    a = np.asarray(values, float); a = a[np.isfinite(a)]
    if not len(a):
        return [float('nan')]*3
    rng = np.random.Generator(np.random.PCG64(independent_seed('bootstrap', identity)))
    # Different implementation from production's integer-index/median helper.
    samples = np.sort(rng.choice(a, size=(BOOTSTRAPS, len(a)), replace=True), axis=1)
    mid = len(a)//2
    boot = samples[:, mid] if len(a) % 2 else (samples[:, mid-1]+samples[:, mid])/2
    ordered = np.sort(boot)
    bounds = []
    for probability in [.025, .975]:
        x = probability*(BOOTSTRAPS-1); lo = int(math.floor(x)); hi = int(math.ceil(x))
        bounds.append(float(ordered[lo]+(ordered[hi]-ordered[lo])*(x-lo)))
    return [own_median(a), *bounds]


def independent_summary(participant_rows):
    output = []
    datasets = sorted({r['dataset'] for r in participant_rows})
    for dataset in datasets:
        rows = [r for r in participant_rows if r['dataset'] == dataset]
        people = sorted({r['participant'] for r in rows})
        table = {(r['participant'], r['condition']): r for r in rows}
        for condition in sorted({r['condition'] for r in rows}):
            for metric in METRICS:
                a = np.array([table[p, condition][metric] for p in people])
                est, low, high = independent_bootstrap(a, f'{dataset}|{condition}|{metric}')
                output.append(dict(dataset=dataset, kind='condition', condition=condition, metric=metric,
                    median=est, ci_low=low, ci_high=high, n_participants=int(np.isfinite(a).sum()), n_roster=len(people),
                    positive_differences=float('nan'), negative_differences=float('nan')))
        for kind, condition, metrics, lhs, rhs, identity in [
            ('paired', 'contiguous_minus_random', ['drift','flip_rate','log_loss','brier','auc','balanced_accuracy','accuracy'], GEOMETRIES[1], GEOMETRIES[0], 'paired'),
            ('baseline_skill', 'prior_minus_model', ['log_loss','brier'], 'prior', 'observed', 'skill')]:
            for metric in metrics:
                a = np.array([table[p, lhs][metric]-table[p, rhs][metric] for p in people])
                est, low, high = independent_bootstrap(a, f'{dataset}|{identity}|{metric}')
                output.append(dict(dataset=dataset, kind=kind, condition=condition, metric=metric,
                    median=est, ci_low=low, ci_high=high, n_participants=int(np.isfinite(a).sum()), n_roster=len(people),
                    positive_differences=int((a > 0).sum()), negative_differences=int((a < 0).sum())))
    return pd.DataFrame(output)


def compare_tables(actual, expected, keys, numbers, label, statistics):
    require(not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), label+': duplicate keys')
    a, b = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    require(a.index.equals(b.index), label+': different row keys')
    for field in numbers:
        numeric(a[field].to_numpy(), b[field].to_numpy(), label+'.'+field, statistics)


def verify(run_dir, output):
    run_manifest = read_json(run_dir/'manifest.json')
    require(run_manifest['status'] == 'complete', 'Run must be complete before this audit starts')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stats, counters = {}, {}
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=sha(Path(__file__)), run_manifest_sha256=sha(run_dir/'manifest.json'),
        new_model_fits=0, rtol=RTOL, atol=ATOL)
    write_json(output/'manifest.json', manifest)
    try:
        protocol_path = HERE/'FROZEN_PROTOCOL.json'; protocol = read_json(protocol_path)
        require(sha(protocol_path) == EXPECTED_PROTOCOL == run_manifest['protocol_sha256'], 'Frozen protocol identity differs')
        require(protocol['config'] == run_manifest['scientific_config'], 'Manifest scientific configuration differs')
        config = protocol['config']
        expected_config = dict(seed=SEED, draws=DRAWS, deletion_fraction=.2, splits='leave_one_participant_out',
            weighting='equal_total_training_weight_per_participant_normalized_mean1', estimator='L2 LogisticRegression',
            C=1., l1_ratio=0., solver='lbfgs', max_iter=2000, tol=1e-8, decision_threshold=.5,
            bootstrap_draws=BOOTSTRAPS, near_constant_probability_sd_flag=.01,
            sample_exclusion_based_on_performance=False, hyperparameter_search=False)
        for key, value in expected_config.items():
            require(config[key] == value, f'Unexpected configuration {key}')
        require(config['features'] == FEATURES == run_manifest['feature_names'], 'Feature schema differs')
        require(config['datasets'] == ['ExtraSensory', 'StudentLife'], 'Dataset scope differs')
        for rel, h in protocol['sources'].items():
            require(sha(HERE/rel) == h, f'Frozen source changed: {rel}')
        inventory = HERE.parent/'external_inventory'
        require(sha(inventory/'DEDUPLICATED_VERIFICATION.json') == protocol['inventory_verification_sha256'], 'Inventory verification chain changed')
        require(sha(inventory/'CANONICALIZATION.json') == protocol['canonicalization_sha256'], 'Canonicalization chain changed')
        require(read_json(inventory/'DEDUPLICATED_VERIFICATION.json')['sources_unchanged'], 'Original source verification failed')
        input_path = Path(protocol['windows_path'])
        require(sha(input_path) == EXPECTED_INPUT == protocol['windows_sha256'] == run_manifest['windows_sha256'], 'Window input changed')
        for rel, h in run_manifest['outputs'].items():
            require(sha(run_dir/rel) == h, f'Run output changed: {rel}')
        counters['output_hash_checks'] = len(run_manifest['outputs'])
        windows = pd.read_parquet(input_path).sort_values(['dataset','participant','label_time','window_id']).reset_index(drop=True)
        obs = pd.read_parquet(run_dir/'observed_predictions.parquet')
        pred = pd.read_parquet(run_dir/'masked_predictions.parquet')
        require(len(windows) == len(obs) == 6211 and len(pred) == 621100, 'Prediction topology size differs')
        require(not obs.duplicated(['dataset','participant','window_id']).any(), 'Repeated observed prediction key')
        require(not pred.duplicated(['dataset','participant','window_id','draw','geometry']).any(), 'Repeated masked prediction key')
        require(run_manifest['model_fits'] == 102 and run_manifest['induced_mask_draws'] == DRAWS, 'Wrong model or draw count')
        expected_index = windows.drop(columns=['times','values'])
        pd.testing.assert_frame_equal(pd.read_parquet(run_dir/'window_index.parquet').reset_index(drop=True), expected_index, check_exact=True)
        ledger_folds = {(r['dataset'], r['heldout_participant']): r for r in read_json(run_dir/'fold_ledger.json')}
        require(len(ledger_folds) == 102, 'Fold ledger topology differs')
        participant_rows, draw_rows, fold_checks = [], [], []
        counters.update(model_folds=0, observed_probability_rows=0, masked_probability_rows=0,
            regenerated_mask_rows=0, matched_mask_pairs=0, independently_recomputed_observed_windows=0,
            independently_recomputed_draw0_geometry_features=0)
        for dataset, part in windows.groupby('dataset', sort=True):
            part = part.reset_index(drop=True); n_windows = len(part)
            directory = run_dir/dataset
            x = np.load(directory/'observed_features.npy', allow_pickle=False)
            xm = np.load(directory/'masked_features.npy', allow_pickle=False, mmap_mode='r')
            require(x.shape == (n_windows, len(FEATURES[dataset])), 'Observed feature dimensions differ')
            require(xm.shape == (n_windows, DRAWS, 2, len(FEATURES[dataset])), 'Masked feature dimensions differ')
            require(np.isfinite(x).all() and np.isfinite(xm).all(), 'Nonfinite feature output')
            masks = pd.read_parquet(directory/'mask_ledger.parquet')
            require(len(masks) == n_windows*DRAWS*2, 'Mask ledger size differs')
            require(set(masks.geometry) == set(GEOMETRIES), 'Unexpected mask geometry')
            masks = masks.assign(_geometry=masks.geometry.map(dict(zip(GEOMETRIES, [0, 1])))).sort_values(['window_index','draw','_geometry']).reset_index(drop=True)
            expected_win = np.repeat(np.arange(n_windows), DRAWS*2)
            require(np.array_equal(masks.window_index, expected_win), 'Mask window index order differs')
            require(np.array_equal(masks.draw, np.tile(np.repeat(np.arange(DRAWS), 2), n_windows)), 'Mask draw schedule differs')
            require(np.array_equal(masks.geometry, np.tile(GEOMETRIES, n_windows*DRAWS)), 'Mask geometry schedule differs')
            require(masks.dataset.eq(dataset).all(), 'Mask dataset crossed')
            require(np.array_equal(masks.participant, part.participant.to_numpy()[expected_win]), 'Mask participant crossed')
            require(np.array_equal(masks.window_id, part.window_id.to_numpy()[expected_win]), 'Mask window identity crossed')
            arrays = {name: masks[name].to_numpy() for name in ['original_records','deleted_records','retained_records','first_deleted_index','last_deleted_index','deletion_span_seconds','mask_sha256']}
            for j, row in enumerate(part.itertuples(index=False)):
                values, times = np.asarray(row.values, float), np.asarray(row.times, float)
                require(values.shape == times.shape and (np.diff(times) > 0).all(), 'Canonical source record order differs')
                numeric(x[j], independent_features(dataset, values), 'observed_features', stats)
                counters['independently_recomputed_observed_windows'] += 1
                n = len(values); k = n//5
                slots = slice(j*DRAWS*2, (j+1)*DRAWS*2)
                require((arrays['original_records'][slots] == n).all() and (arrays['deleted_records'][slots] == k).all()
                    and (arrays['retained_records'][slots] == n-k).all(), 'Mask count mismatch')
                for draw in range(DRAWS):
                    pair = independent_masks(n, dataset, row.participant, row.window_id, draw)
                    for gi, removed in enumerate(pair):
                        pos = j*DRAWS*2+draw*2+gi
                        require(len(removed) == k and len(np.unique(removed)) == k, 'Regenerated mask count mismatch')
                        require(arrays['first_deleted_index'][pos] == removed[0] and arrays['last_deleted_index'][pos] == removed[-1], 'Mask endpoints differ')
                        require(arrays['deletion_span_seconds'][pos] == times[removed[-1]]-times[removed[0]], 'Elapsed deletion span differs')
                        digest = hashlib.sha256(removed.astype('<i8').tobytes()).hexdigest()
                        require(digest == arrays['mask_sha256'][pos], f'Mask digest differs: {dataset}/{j}/{draw}/{gi}')
                        if draw == 0:
                            retained = np.ones(n, bool); retained[removed] = False
                            numeric(xm[j, draw, gi], independent_features(dataset, values[retained]), 'draw0_masked_features', stats)
                            counters['independently_recomputed_draw0_geometry_features'] += 1
                    counters['matched_mask_pairs'] += 1
                    counters['regenerated_mask_rows'] += 2
                if (j+1) % 500 == 0:
                    print(json.dumps(dict(stage='mask_audit', dataset=dataset, windows=j+1, total=n_windows)), flush=True)
            groups, target = part.participant.to_numpy(), part.label.to_numpy(dtype=int)
            people = sorted(set(groups))
            require(len(people) == {'StudentLife':49, 'ExtraSensory':53}[dataset], 'Participant roster size differs')
            model_files = list((directory/'models').glob('*.pkl'))
            require({p.stem for p in model_files} == set(people), 'Saved model roster differs')
            for pid in people:
                with (directory/'models'/f'{pid}.pkl').open('rb') as f:
                    fold = pickle.load(f)
                train = np.flatnonzero(groups != pid); held = np.flatnonzero(groups == pid)
                training_people = [p for p in people if p != pid]
                require(np.array_equal(fold['train_indices'], train) and np.array_equal(fold['test_indices'], held), 'Fold indices differ')
                require(fold['train_participants'] == training_people and fold['heldout_participant'] == pid, 'Participant leakage in fold roster')
                require(fold['n_train'] == len(train) and fold['n_test'] == len(held), 'Fold sample counts differ')
                require(set(target[train]) == {0, 1}, 'Outer training classes differ')
                per_person_features = [x[groups == p] for p in training_people]
                mean = np.mean([a.mean(axis=0) for a in per_person_features], axis=0)
                var = np.mean([np.mean((a-mean)**2, axis=0) for a in per_person_features], axis=0)
                scale = np.sqrt(var)
                eps = np.finfo(float).eps
                constant = var <= len(train)*eps*var+(len(train)*mean*eps)**2
                scale[constant] = 1
                prior = float(np.mean([target[groups == p].mean() for p in training_people]))
                numeric(fold['scaler'].mean_, mean, 'weighted_scaler_mean', stats)
                numeric(fold['scaler'].var_, var, 'weighted_scaler_variance', stats)
                numeric(fold['scaler'].scale_, scale, 'weighted_scaler_scale', stats)
                numeric(fold['scaler'].n_samples_seen_, len(train), 'weighted_scaler_total_weight', stats)
                numeric(fold['prior'], prior, 'training_only_prior', stats)
                for p in training_people:
                    numeric(fold['weight_total_by_participant'][p], len(train)/len(training_people), 'equal_training_participant_weight', stats)
                model = fold['model']; params = model.get_params()
                for key in ['C','l1_ratio','solver','max_iter','tol']:
                    require(params[key] == config[key], f'Estimator setting differs: {key}')
                require(params['random_state'] == SEED and params['class_weight'] is None and params['fit_intercept'], 'Unspecified estimator change')
                require(np.array_equal(model.classes_, [0, 1]) and model.coef_.shape == (1, x.shape[1]), 'Estimator output class/feature order differs')
                require(0 < int(model.n_iter_[0]) <= 2000, 'Invalid iteration record')
                fingerprint = independent_fingerprint(fold)
                require(fingerprint == fold['fingerprint'] == ledger_folds[dataset, pid]['fingerprint'], 'Saved fitted-state fingerprint differs')
                ledger = ledger_folds[dataset, pid]
                require(ledger['train_participants'] == training_people and ledger['unmasked_train_only'] and ledger['frozen_during_scoring'], 'Fold metadata disagrees')
                observed = obs.loc[obs.dataset.eq(dataset) & obs.participant.eq(pid)].sort_values('window_index')
                masked = pred.loc[pred.dataset.eq(dataset) & pred.participant.eq(pid)].assign(_geometry=lambda d: d.geometry.map(dict(zip(GEOMETRIES,[0,1])))).sort_values(['window_index','draw','_geometry'])
                require(np.array_equal(observed.window_index, held), 'Observed prediction indices differ')
                require(np.array_equal(observed.window_id, part.iloc[held].window_id), 'Observed window crossed')
                require(np.array_equal(observed.label, target[held]), 'Observed targets differ')
                for field in ['window_start','window_end','label_time','acquisition_duration']:
                    numeric(observed[field].to_numpy(), part.iloc[held][field].to_numpy(), 'observed_metadata.'+field, stats)
                require(np.array_equal(masked.window_index, np.repeat(held,DRAWS*2)), 'Masked prediction indices differ')
                require(np.array_equal(masked.window_id, np.repeat(part.iloc[held].window_id.to_numpy(),DRAWS*2)), 'Masked window crossed')
                require(np.array_equal(masked.label, np.repeat(target[held],DRAWS*2)), 'Masked targets differ')
                require(np.array_equal(masked.draw, np.tile(np.repeat(np.arange(DRAWS),2),len(held))), 'Masked prediction draw schedule differs')
                require(np.array_equal(masked.geometry, np.tile(GEOMETRIES,len(held)*DRAWS)), 'Masked prediction geometry schedule differs')
                require(observed.model_fingerprint.eq(fingerprint).all() and masked.model_fingerprint.eq(fingerprint).all(), 'Prediction model fingerprint crossed')
                base = model.predict_proba(fold['scaler'].transform(x[held]))[:,1]
                masked_x = np.asarray(xm[held]).reshape(-1, x.shape[1])
                probs = model.predict_proba(fold['scaler'].transform(masked_x))[:,1]
                numeric(observed.p_observed.to_numpy(), base, 'saved_state_observed_probabilities', stats)
                numeric(masked.p.to_numpy(), probs, 'saved_state_masked_probabilities', stats)
                numeric(observed.p_observed.to_numpy(), manual_probabilities(fold,x[held]), 'manual_observed_probabilities', stats)
                numeric(masked.p.to_numpy(), manual_probabilities(fold,masked_x), 'manual_masked_probabilities', stats)
                numeric(observed.p_prior.to_numpy(), np.full(len(held),prior), 'observed_prior_probabilities', stats)
                numeric(masked.p_prior.to_numpy(), np.full(len(masked),prior), 'masked_prior_probabilities', stats)
                numeric(masked.p_observed.to_numpy(), np.repeat(observed.p_observed.to_numpy(),DRAWS*2), 'masked_unmasked_pairing', stats)
                require(independent_fingerprint(fold) == fingerprint, 'Verification mutated model/scaler state')
                # Summary calculations use the saved probabilities after their state comparison.
                y = target[held]; p_base = observed.p_observed.to_numpy(); p_prior = observed.p_prior.to_numpy()
                cube = masked.p.to_numpy().reshape(len(held),DRAWS,2)
                for condition, pp in [('observed',p_base),('prior',p_prior)]:
                    participant_rows.append(dict(dataset=dataset,participant=pid,condition=condition,
                        n_windows=len(held),n_draws=1,**independent_metrics(y,pp,pp)))
                for gi, geometry in enumerate(GEOMETRIES):
                    results = []
                    for draw in range(DRAWS):
                        mm = independent_metrics(y,cube[:,draw,gi],p_base);results.append(mm)
                        draw_rows.append(dict(dataset=dataset,participant=pid,geometry=geometry,draw=draw,n_windows=len(held),**mm))
                    averages = {field: float(math.fsum(r[field] for r in results)/DRAWS) if all(np.isfinite(r[field]) for r in results) else float('nan') for field in METRICS}
                    participant_rows.append(dict(dataset=dataset,participant=pid,condition=geometry,n_windows=len(held),n_draws=DRAWS,**averages))
                fold_checks.append(dict(dataset=dataset,participant=pid,n_train=len(train),n_test=len(held),fingerprint=fingerprint,
                    rosters_disjoint=True,weighted_scaler_and_prior_match=True,all_saved_probabilities_match=True,state_unchanged=True))
                counters['model_folds'] += 1
                counters['observed_probability_rows'] += len(held)
                counters['masked_probability_rows'] += len(masked)
            print(json.dumps(dict(stage='fold_audit', dataset=dataset, models=len(people))), flush=True)
        expected_participants = pd.DataFrame(participant_rows)
        expected_draws = pd.DataFrame(draw_rows)
        compare_tables(pd.read_csv(run_dir/'participant_metrics.csv'),expected_participants,['dataset','participant','condition'],['n_windows','n_draws',*METRICS],'participant_metrics',stats)
        compare_tables(pd.read_csv(run_dir/'participant_draw_metrics.csv'),expected_draws,['dataset','participant','geometry','draw'],['n_windows',*METRICS],'draw_metrics',stats)
        expected_summary = independent_summary(participant_rows)
        compare_tables(pd.read_csv(run_dir/'summary.csv'),expected_summary,['dataset','kind','condition','metric'],
            ['median','ci_low','ci_high','n_participants','n_roster','positive_differences','negative_differences'],'summary',stats)
        for dataset, n_auc in [('StudentLife',41),('ExtraSensory',52)]:
            rr = expected_summary.loc[expected_summary.dataset.eq(dataset) & expected_summary.kind.eq('condition') & expected_summary.metric.isin(['auc','balanced_accuracy'])]
            require(rr.n_participants.eq(n_auc).all(), 'Class-dependent participant denominator differs')
        guards = {r['dataset']: r for r in read_json(run_dir/'baseline_guard.json')}
        for dataset in FEATURES:
            ss = expected_summary[expected_summary.dataset.eq(dataset)]
            skill = ss[ss.kind.eq('baseline_skill') & ss.metric.eq('log_loss')].iloc[0]
            auc = ss[ss.condition.eq('observed') & ss.metric.eq('auc')].iloc[0]
            pm = expected_participants[expected_participants.dataset.eq(dataset) & expected_participants.condition.eq('observed')]
            expected_guard = dict(n_participants=len(pm),n_near_constant_probability_sd_lt_0_01=int((pm.probability_sd<.01).sum()),
                n_single_predicted_class=int(pm.predicted_positive_fraction.isin([0.,1.]).sum()),
                median_prior_minus_model_log_loss=float(skill['median']),median_observed_auc=float(auc['median']),
                point_estimate_skill_guard=bool(skill['median']>0 and auc['median']>.5),
                both_descriptive_lower_bounds_above_null=bool(skill.ci_low>0 and auc.ci_low>.5),participants_excluded_based_on_performance=0)
            for field, value in expected_guard.items():
                if isinstance(value, float):
                    numeric(guards[dataset][field], value, 'baseline_guard.'+field, stats)
                else:
                    require(guards[dataset][field] == value, f'Baseline guard differs: {dataset}/{field}')
        # End-of-audit physical hashes distinguish data immutability from numerical agreement.
        require(sha(run_dir/'manifest.json') == manifest['run_manifest_sha256'], 'Run manifest changed during audit')
        require(sha(input_path) == EXPECTED_INPUT and sha(protocol_path) == EXPECTED_PROTOCOL, 'Frozen evidence changed during audit')
        for rel, h in run_manifest['outputs'].items():
            require(sha(run_dir/rel) == h, f'Run output changed during audit: {rel}')
        for rel, h in protocol['sources'].items():
            require(sha(HERE/rel) == h, f'Frozen source changed during audit: {rel}')
        require(sha(Path(__file__)) == manifest['audit_source_sha256'], 'Auditor changed during audit')
        pd.DataFrame(fold_checks).to_csv(output/'fold_checks.csv',index=False)
        expected_participants.to_csv(output/'independent_participant_metrics.csv',index=False)
        expected_summary.to_csv(output/'independent_summary.csv',index=False)
        write_json(output/'numeric_comparisons.json',stats)
        require(counters['model_folds']==102 and counters['regenerated_mask_rows']==621100 and
            counters['independently_recomputed_draw0_geometry_features']==12422, 'Incomplete audit topology')
        manifest.update(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
            counters=counters,source_hash_chain_verified=True,run_outputs_unchanged=True,
            auc_balanced_accuracy_participants={'StudentLife':41,'ExtraSensory':52},
            unavailable_participants_retained=True,production_summary_or_feature_helpers_used=False,
            new_model_fits=0,feature_recompute_scope='All 6211 unmasked windows; fixed draw 0, both geometries, all 6211 windows. Masks and probabilities verified for every draw.')
        report = ['# Independent external-prediction verification','',
            'PASS. No model was refitted. The audit uses independently implemented feature reductions, seed serialization/mask regeneration, weighted scaler/prior formulas, rank-based AUROC, proper losses, participant summaries, and bootstrap order statistics.','',
            f"- Saved folds checked: {counters['model_folds']}; all participant train/test rosters are disjoint and complete.",
            f"- Observed probability rows checked: {counters['observed_probability_rows']:,}; masked probability rows: {counters['masked_probability_rows']:,}.",
            f"- Original masks regenerated: {counters['regenerated_mask_rows']:,}; equal-count paired masks: {counters['matched_mask_pairs']:,}.",
            '- Independent source-value feature reductions: all 6,211 unmasked windows and all 12,422 draw-0 geometry/window combinations. Other draws have every mask and saved-state probability checked; their feature reductions were not independently recalculated.',
            '- All participant/draw metrics, participant draw averages, paired geometry contrasts, prior-comparator gains, and 10,000-resample bootstrap summaries match within the stated numerical tolerance.',
            '- AUROC/balanced-accuracy participant denominators are StudentLife 41/49 and ExtraSensory 52/53; single-class participants remain in applicable loss/stability metrics.',
            '- Frozen protocol, input, source-code hashes, inventory/canonicalization evidence chain, and every saved run-output hash match before and after verification.','',
            f'Numerical acceptance is rtol={RTOL:g}, atol={ATOL:g}, with matching unavailable NaNs. `numeric_comparisons.json` reports exact numeric counts and maximum absolute differences separately. Numerical tolerance is not byte equality; SHA-256 checks establish physical artifact identity.','',
            'This audit verifies the saved-state calculations and evidence chain. It does not independently refit logistic coefficients or turn exploratory descriptive intervals into confirmatory evidence.']
        (output/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf8')
        manifest['outputs'] = {p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
        write_json(output/'manifest.json',manifest)
        print(json.dumps(dict(status='PASS',counters=counters,elapsed_seconds=manifest['elapsed_seconds'])),flush=True)
    except BaseException as exc:
        manifest.update(status='FAIL',elapsed_seconds=time.perf_counter()-started,error=str(exc),counters=counters)
        (output/'traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        write_json(output/'numeric_comparisons_partial.json',stats)
        write_json(output/'manifest.json',manifest)
        print(json.dumps(dict(status='FAIL',error=str(exc))),flush=True)
        raise


def self_test():
    np.testing.assert_allclose(independent_features('StudentLife',[0,0,1,2]),[.5,.75])
    np.testing.assert_allclose(independent_features('ExtraSensory',[0,2,4]),[2,math.sqrt(8/3),2,3.6])
    mm = independent_metrics([0,1],[.25,.75],[.25,.75])
    require(mm['auc']==1 and mm['balanced_accuracy']==1 and mm['brier']==.0625 and mm['drift']==0, 'Synthetic metric check failed')
    require(abs(mm['log_loss']+math.log(.75))<1e-15,'Synthetic log loss check failed')
    require(math.isnan(independent_metrics([1,1],[.8,.8],[.8,.8])['auc']),'Single-class handling differs')
    tied = independent_metrics([0,1],[.3,.3],[.3,.3]); require(tied['auc']==.5,'AUC ties differ')
    for n in range(5,101):
        a,b=independent_masks(n,'Synthetic','p','w',0)
        require(len(a)==len(b)==n//5 and len(np.unique(a))==len(a) and (np.diff(b)==1).all(),'Synthetic mask check failed')
    e,l,h=independent_bootstrap([1.,2.,3.],'synthetic')
    require(e==2 and 1<=l<=e<=h<=3,'Synthetic bootstrap check failed')
    print('Independent auditor self-checks passed; no model fits and no data-result inspection.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=HERE/'run_01')
    parser.add_argument('--output',type=Path,default=HERE/'independent_verification_01')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:verify(args.run,args.output)
