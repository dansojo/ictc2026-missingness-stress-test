"""Deterministic explicit-duration Gaussian HMM used as a non-novel engine."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


_MAX_DURATION = 12
_PSEUDOCOUNT = 1e-6
_VARIANCE_FLOOR = 1e-6


def _readonly(value: np.ndarray, dtype: np.dtype | str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DurationModel:
    state_count: int
    feature_count: int
    means: np.ndarray
    variances: np.ndarray
    initial_probabilities: np.ndarray
    transition_probabilities: np.ndarray
    duration_probabilities: np.ndarray
    log_likelihood: float
    parameter_count: int
    iterations: int
    content_digest: str


@dataclass(frozen=True)
class DurationDecode:
    states: np.ndarray
    state_posterior: np.ndarray
    log_likelihood: float
    content_digest: str


@dataclass(frozen=True)
class BicEntry:
    state_count: int
    bic: float
    log_likelihood: float
    parameter_count: int
    model_digest: str


@dataclass(frozen=True)
class StateCountSelection:
    model: DurationModel
    entries: tuple[BicEntry, ...]
    content_digest: str


def _require_training_arrays(
    values: np.ndarray, availability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if type(values) is not np.ndarray or values.ndim != 3:
        raise TypeError("values must be an exact three-dimensional ndarray")
    if values.dtype != np.dtype("<f8"):
        raise TypeError("values must have exact float64 dtype")
    if type(availability) is not np.ndarray or availability.shape != values.shape:
        raise TypeError("availability must be an exact shape-matched ndarray")
    if availability.dtype != np.dtype("bool"):
        raise TypeError("availability must have exact bool dtype")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("training arrays must be nonempty")
    if not np.all(np.isfinite(values[availability])):
        raise ValueError("observed values must be finite")
    if not np.any(availability):
        raise ValueError("at least one scalar must be observed")
    return values, availability


def _require_sequence_arrays(
    values: np.ndarray, availability: np.ndarray, feature_count: int
) -> tuple[np.ndarray, np.ndarray]:
    if type(values) is not np.ndarray or values.ndim != 2:
        raise TypeError("values must be an exact two-dimensional ndarray")
    if values.dtype != np.dtype("<f8") or values.shape[1] != feature_count:
        raise TypeError("values have the wrong dtype or feature count")
    if type(availability) is not np.ndarray or availability.shape != values.shape:
        raise TypeError("availability must be an exact shape-matched ndarray")
    if availability.dtype != np.dtype("bool") or values.shape[0] == 0:
        raise TypeError("availability must have exact bool dtype")
    if not np.all(np.isfinite(values[availability])):
        raise ValueError("observed values must be finite")
    return values, availability


def _sequence_key(values: np.ndarray, availability: np.ndarray) -> bytes:
    normalized = np.where(availability, values, 0.0).astype("<f8", copy=False)
    hasher = hashlib.sha256(b"episode-hsmm-sequence-v1\0")
    hasher.update(normalized.tobytes(order="C"))
    hasher.update(availability.tobytes(order="C"))
    return hasher.digest()


def _sorted_sequences(
    values: np.ndarray, availability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(
        range(values.shape[0]),
        key=lambda index: _sequence_key(values[index], availability[index]),
    )
    return values[order].copy(), availability[order].copy()


def _emission_log_likelihood(
    values: np.ndarray,
    availability: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    safe = np.where(availability, values, 0.0)
    residual = safe[:, None, :] - means[None, :, :]
    terms = -0.5 * (
        np.log(2.0 * np.pi * variances)[None, :, :]
        + residual * residual / variances[None, :, :]
    )
    return np.sum(np.where(availability[:, None, :], terms, 0.0), axis=2)


def _logs(probabilities: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


def _viterbi_states(
    emission: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    duration: np.ndarray,
) -> np.ndarray:
    time_count, state_count = emission.shape
    log_initial = _logs(initial)
    log_transition = _logs(transition)
    log_duration = _logs(duration)
    score = log_initial[:, None] + log_duration + emission[0, :, None]
    back = np.full(
        (time_count, state_count, _MAX_DURATION), -1, dtype=np.int64
    )
    for time in range(1, time_count):
        candidate = np.full_like(score, -np.inf)
        predecessor = np.full(
            (state_count, _MAX_DURATION), -1, dtype=np.int64
        )
        candidate[:, : _MAX_DURATION - 1] = score[:, 1:]
        for state in range(state_count):
            predecessor[state, : _MAX_DURATION - 1] = (
                state * _MAX_DURATION + np.arange(1, _MAX_DURATION)
            )
        ended = score[:, 0][:, None] + log_transition
        best_previous = np.argmax(ended, axis=0)
        best_entry = ended[best_previous, np.arange(state_count)]
        entry = best_entry[:, None] + log_duration
        entry_predecessor = best_previous[:, None] * _MAX_DURATION
        use_entry = entry > candidate
        candidate = np.where(use_entry, entry, candidate)
        predecessor = np.where(use_entry, entry_predecessor, predecessor)
        score = candidate + emission[time, :, None]
        back[time] = predecessor
    expanded = int(np.argmax(score))
    states = np.empty(time_count, dtype=np.int64)
    for time in range(time_count - 1, -1, -1):
        states[time] = expanded // _MAX_DURATION
        if time:
            state = expanded // _MAX_DURATION
            remaining = expanded % _MAX_DURATION
            expanded = int(back[time, state, remaining])
    return states


def _expanded_transition(
    transition: np.ndarray, duration: np.ndarray
) -> np.ndarray:
    state_count = transition.shape[0]
    size = state_count * _MAX_DURATION
    result = np.zeros((size, size), dtype=np.float64)
    for state in range(state_count):
        for remaining in range(2, _MAX_DURATION + 1):
            source = state * _MAX_DURATION + remaining - 1
            destination = state * _MAX_DURATION + remaining - 2
            result[source, destination] = 1.0
        source = state * _MAX_DURATION
        for next_state in range(state_count):
            if transition[state, next_state] == 0.0:
                continue
            start = next_state * _MAX_DURATION
            result[source, start : start + _MAX_DURATION] = (
                transition[state, next_state] * duration[next_state]
            )
    return result


def _forward_log_likelihood(
    emission: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    duration: np.ndarray,
) -> float:
    log_transition = _logs(transition)
    log_duration = _logs(duration)
    score = _logs(initial)[:, None] + log_duration + emission[0, :, None]
    for time in range(1, emission.shape[0]):
        candidate = np.full_like(score, -np.inf)
        candidate[:, : _MAX_DURATION - 1] = score[:, 1:]
        entry = logsumexp(score[:, 0][:, None] + log_transition, axis=0)
        entry = entry[:, None] + log_duration
        score = np.logaddexp(candidate, entry) + emission[time, :, None]
    return float(logsumexp(score))


def _decode_full(
    model: DurationModel, values: np.ndarray, availability: np.ndarray
) -> DurationDecode:
    emission = _emission_log_likelihood(
        values, availability, model.means, model.variances
    )
    states = _viterbi_states(
        emission,
        model.initial_probabilities,
        model.transition_probabilities,
        model.duration_probabilities,
    )
    expanded_transition = _expanded_transition(
        model.transition_probabilities, model.duration_probabilities
    )
    log_expanded_transition = _logs(expanded_transition)
    expanded_emission = np.repeat(emission, _MAX_DURATION, axis=1)
    log_initial = (
        _logs(model.initial_probabilities)[:, None]
        + _logs(model.duration_probabilities)
    ).reshape(-1)
    alpha = np.empty_like(expanded_emission)
    alpha[0] = log_initial + expanded_emission[0]
    for time in range(1, values.shape[0]):
        alpha[time] = (
            logsumexp(alpha[time - 1][:, None] + log_expanded_transition, axis=0)
            + expanded_emission[time]
        )
    log_likelihood = float(logsumexp(alpha[-1]))
    beta = np.empty_like(alpha)
    beta[-1] = 0.0
    for time in range(values.shape[0] - 2, -1, -1):
        beta[time] = logsumexp(
            log_expanded_transition
            + expanded_emission[time + 1][None, :]
            + beta[time + 1][None, :],
            axis=1,
        )
    expanded_posterior = np.exp(alpha + beta - log_likelihood)
    posterior = expanded_posterior.reshape(
        values.shape[0], model.state_count, _MAX_DURATION
    ).sum(axis=2)
    posterior /= posterior.sum(axis=1, keepdims=True)
    states = _readonly(states, np.dtype("<i8"))
    posterior = _readonly(posterior, np.dtype("<f8"))
    hasher = hashlib.sha256(b"episode-duration-decode-v1\0")
    hasher.update(bytes.fromhex(model.content_digest))
    hasher.update(states.tobytes(order="C"))
    hasher.update(posterior.tobytes(order="C"))
    hasher.update(float(log_likelihood).hex().encode("ascii"))
    return DurationDecode(states, posterior, log_likelihood, hasher.hexdigest())


def _run_statistics(paths: tuple[np.ndarray, ...], state_count: int):
    initial = np.full(state_count, _PSEUDOCOUNT, dtype=np.float64)
    transition = np.full((state_count, state_count), _PSEUDOCOUNT, dtype=np.float64)
    np.fill_diagonal(transition, 0.0)
    duration = np.full(
        (state_count, _MAX_DURATION), _PSEUDOCOUNT, dtype=np.float64
    )
    for path in paths:
        initial[int(path[0])] += 1.0
        boundaries = np.flatnonzero(path[1:] != path[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, path.size]
        run_states = path[starts]
        run_durations = ends - starts
        for state, length in zip(run_states, run_durations, strict=True):
            duration[int(state), min(int(length), _MAX_DURATION) - 1] += 1.0
        for left, right in zip(run_states[:-1], run_states[1:], strict=True):
            transition[int(left), int(right)] += 1.0
    initial /= initial.sum()
    for state in range(state_count):
        if state_count == 1:
            transition[state, state] = 1.0
        else:
            transition[state] /= transition[state].sum()
    duration /= duration.sum(axis=1, keepdims=True)
    return initial, transition, duration


def _canonical_order(means: np.ndarray) -> np.ndarray:
    keys = tuple(means[:, column] for column in range(means.shape[1] - 1, -1, -1))
    return np.lexsort(keys)


def _model_digest(
    state_count: int,
    feature_count: int,
    arrays: tuple[np.ndarray, ...],
    log_likelihood: float,
    parameter_count: int,
    iterations: int,
) -> str:
    hasher = hashlib.sha256(b"episode-duration-model-v1\0")
    header = json.dumps(
        {
            "state_count": state_count,
            "feature_count": feature_count,
            "log_likelihood": float(log_likelihood).hex(),
            "parameter_count": parameter_count,
            "iterations": iterations,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hasher.update(header)
    for array in arrays:
        hasher.update(array.dtype.str.encode("ascii"))
        hasher.update(json.dumps(array.shape).encode("ascii"))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def fit_duration_model(
    sequences: np.ndarray,
    availability: np.ndarray,
    *,
    state_count: int,
) -> DurationModel:
    """Fit a deterministic hard-EM diagonal-Gaussian duration model."""

    sequences, availability = _require_training_arrays(sequences, availability)
    if type(state_count) is not int or not 2 <= state_count <= 16:
        raise TypeError("state_count must be an exact int in [2, 16]")
    if int(np.count_nonzero(np.any(availability, axis=2))) < state_count:
        raise ValueError("state_count exceeds usable observations")
    sequences, availability = _sorted_sequences(sequences, availability)
    feature_count = sequences.shape[2]
    observed_counts = availability.sum(axis=(0, 1))
    if np.any(observed_counts == 0):
        raise ValueError("every feature needs at least one observed scalar")
    global_means = np.sum(np.where(availability, sequences, 0.0), axis=(0, 1)) / observed_counts
    imputed = np.where(availability, sequences, global_means).reshape(-1, feature_count)
    sort_keys = tuple(
        imputed[:, column] for column in range(feature_count - 1, -1, -1)
    )
    imputed_sorted = imputed[np.lexsort(sort_keys)]
    kmeans = KMeans(n_clusters=state_count, random_state=0, n_init=20)
    kmeans.fit(imputed_sorted)
    means = kmeans.cluster_centers_.astype(np.float64)
    global_variance = np.maximum(np.var(imputed_sorted, axis=0), _VARIANCE_FLOOR)
    variances = np.tile(global_variance, (state_count, 1))
    initial = np.full(state_count, 1.0 / state_count, dtype=np.float64)
    transition = np.ones((state_count, state_count), dtype=np.float64)
    np.fill_diagonal(transition, 0.0)
    transition /= transition.sum(axis=1, keepdims=True)
    duration = np.full(
        (state_count, _MAX_DURATION), 1.0 / _MAX_DURATION, dtype=np.float64
    )
    previous_paths: tuple[np.ndarray, ...] | None = None
    iterations = 0
    for iteration in range(1, 21):
        paths = tuple(
            _viterbi_states(
                _emission_log_likelihood(sequence, observed, means, variances),
                initial,
                transition,
                duration,
            )
            for sequence, observed in zip(sequences, availability, strict=True)
        )
        iterations = iteration
        if previous_paths is not None and all(
            np.array_equal(left, right)
            for left, right in zip(paths, previous_paths, strict=True)
        ):
            break
        previous_paths = tuple(path.copy() for path in paths)
        old_means = means.copy()
        old_variances = variances.copy()
        for state in range(state_count):
            state_rows = np.stack([path == state for path in paths], axis=0)
            for feature in range(feature_count):
                selected = state_rows & availability[:, :, feature]
                observed_values = sequences[:, :, feature][selected]
                if observed_values.size:
                    means[state, feature] = float(np.mean(observed_values))
                    variances[state, feature] = max(
                        float(np.mean((observed_values - means[state, feature]) ** 2)),
                        _VARIANCE_FLOOR,
                    )
                else:
                    means[state, feature] = old_means[state, feature]
                    variances[state, feature] = old_variances[state, feature]
        initial, transition, duration = _run_statistics(paths, state_count)

    order = _canonical_order(means)
    means = means[order]
    variances = variances[order]
    initial = initial[order]
    transition = transition[order][:, order]
    duration = duration[order]
    total_log_likelihood = 0.0
    for sequence, observed in zip(sequences, availability, strict=True):
        emission = _emission_log_likelihood(sequence, observed, means, variances)
        total_log_likelihood += _forward_log_likelihood(
            emission, initial, transition, duration
        )
    parameter_count = (
        2 * state_count * feature_count
        + (state_count - 1)
        + state_count * max(state_count - 2, 0)
        + state_count * (_MAX_DURATION - 1)
    )
    means = _readonly(means, np.dtype("<f8"))
    variances = _readonly(variances, np.dtype("<f8"))
    initial = _readonly(initial, np.dtype("<f8"))
    transition = _readonly(transition, np.dtype("<f8"))
    duration = _readonly(duration, np.dtype("<f8"))
    digest = _model_digest(
        state_count,
        feature_count,
        (means, variances, initial, transition, duration),
        total_log_likelihood,
        parameter_count,
        iterations,
    )
    return DurationModel(
        state_count,
        feature_count,
        means,
        variances,
        initial,
        transition,
        duration,
        total_log_likelihood,
        parameter_count,
        iterations,
        digest,
    )


def decode_duration_model(
    model: DurationModel,
    values: np.ndarray,
    availability: np.ndarray,
) -> DurationDecode:
    if type(model) is not DurationModel:
        raise TypeError("model must be an exact DurationModel")
    values, availability = _require_sequence_arrays(
        values, availability, model.feature_count
    )
    return _decode_full(model, values, availability)


def select_state_count(
    sequences: np.ndarray, availability: np.ndarray
) -> StateCountSelection:
    sequences, availability = _require_training_arrays(sequences, availability)
    observed_scalar_count = int(np.count_nonzero(availability))
    entries: list[BicEntry] = []
    models: dict[int, DurationModel] = {}
    for state_count in (4, 6, 8):
        model = fit_duration_model(
            sequences, availability, state_count=state_count
        )
        bic = -2.0 * model.log_likelihood + model.parameter_count * math.log(
            observed_scalar_count
        )
        models[state_count] = model
        entries.append(
            BicEntry(
                state_count,
                float(bic),
                model.log_likelihood,
                model.parameter_count,
                model.content_digest,
            )
        )
    chosen_entry = min(entries, key=lambda item: (item.bic, item.state_count))
    hasher = hashlib.sha256(b"episode-state-count-selection-v1\0")
    for entry in entries:
        hasher.update(
            json.dumps(
                {
                    "state_count": entry.state_count,
                    "bic": entry.bic.hex(),
                    "log_likelihood": entry.log_likelihood.hex(),
                    "parameter_count": entry.parameter_count,
                    "model_digest": entry.model_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return StateCountSelection(
        models[chosen_entry.state_count], tuple(entries), hasher.hexdigest()
    )


__all__ = [
    "BicEntry",
    "DurationDecode",
    "DurationModel",
    "StateCountSelection",
    "decode_duration_model",
    "fit_duration_model",
    "select_state_count",
]
