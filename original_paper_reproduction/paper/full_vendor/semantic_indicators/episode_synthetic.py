"""Outcome-free synthetic benchmark for episode-stability validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


_PARTICIPANTS = 10
_DAYS = 14
_BINS = 48
_DOMAINS = 6
_STATES = 4
_SCENARIOS = ("contiguous_20pct", "outage_1h")


def _readonly(array: np.ndarray, dtype: np.dtype | str) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def _hash_array(hasher: "hashlib._Hash", name: str, array: np.ndarray) -> None:
    header = json.dumps(
        {"name": name, "shape": array.shape, "dtype": array.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hasher.update(len(header).to_bytes(8, "big"))
    hasher.update(header)
    hasher.update(array.tobytes(order="C"))


@dataclass(frozen=True)
class SyntheticMask:
    scenario: str
    draw: int
    withheld: np.ndarray
    content_digest: str


@dataclass(frozen=True)
class SyntheticEpisodeBatch:
    participant_ids: tuple[str, ...]
    values: np.ndarray
    availability: np.ndarray
    true_states: np.ndarray
    induced_masks: tuple[SyntheticMask, ...]
    content_digest: str


def _six_durations(rng: np.random.Generator) -> tuple[int, ...]:
    remaining = _BINS
    durations: list[int] = []
    for segment in range(5):
        remaining_segments = 6 - segment - 1
        minimum = max(1, remaining - 12 * remaining_segments)
        maximum = min(12, remaining - remaining_segments)
        duration = int(rng.integers(minimum, maximum + 1))
        durations.append(duration)
        remaining -= duration
    durations.append(remaining)
    return tuple(durations)


def _state_sequences(rng: np.random.Generator) -> np.ndarray:
    states = np.empty((_PARTICIPANTS, _DAYS, _BINS), dtype="<i8")
    for participant in range(_PARTICIPANTS):
        for day in range(_DAYS):
            cursor = 0
            state = (participant + day) % _STATES
            for duration in _six_durations(rng):
                states[participant, day, cursor : cursor + duration] = state
                cursor += duration
                choices = tuple(candidate for candidate in range(_STATES) if candidate != state)
                state = choices[int(rng.integers(0, len(choices)))]
    return states


def _mask_digest(scenario: str, draw: int, withheld: np.ndarray) -> str:
    hasher = hashlib.sha256(b"episode-synthetic-mask-v1\0")
    hasher.update(scenario.encode("ascii"))
    hasher.update(draw.to_bytes(2, "big"))
    _hash_array(hasher, "withheld", withheld)
    return hasher.hexdigest()


def _induced_masks(seed: int) -> tuple[SyntheticMask, ...]:
    masks: list[SyntheticMask] = []
    for scenario in _SCENARIOS:
        for draw in range(4):
            material = f"episode-mask-v1|{seed}|{scenario}|{draw}".encode("ascii")
            derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            rng = np.random.default_rng(derived_seed)
            withheld = np.zeros(
                (_PARTICIPANTS, _DAYS, _BINS, _DOMAINS), dtype=np.bool_
            )
            for participant in range(_PARTICIPANTS):
                for day in range(_DAYS):
                    if scenario == "contiguous_20pct":
                        for domain in range(_DOMAINS):
                            start = int(rng.integers(0, _BINS - 10 + 1))
                            withheld[participant, day, start : start + 10, domain] = True
                    else:
                        domain = int(rng.integers(0, _DOMAINS))
                        start = int(rng.integers(0, _BINS - 2 + 1))
                        withheld[participant, day, start : start + 2, domain] = True
            withheld = _readonly(withheld, np.dtype("bool"))
            masks.append(
                SyntheticMask(
                    scenario=scenario,
                    draw=draw,
                    withheld=withheld,
                    content_digest=_mask_digest(scenario, draw, withheld),
                )
            )
    return tuple(masks)


def _batch_digest(
    seed: int,
    participant_shift: tuple[int, float] | None,
    values: np.ndarray,
    availability: np.ndarray,
    true_states: np.ndarray,
    masks: tuple[SyntheticMask, ...],
) -> str:
    hasher = hashlib.sha256(b"episode-synthetic-batch-v1\0")
    header = json.dumps(
        {"seed": seed, "participant_shift": participant_shift},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hasher.update(header)
    _hash_array(hasher, "values", values)
    _hash_array(hasher, "availability", availability)
    _hash_array(hasher, "true_states", true_states)
    for mask in masks:
        hasher.update(bytes.fromhex(mask.content_digest))
    return hasher.hexdigest()


def generate_episode_batch(
    seed: int,
    *,
    participant_shift: tuple[int, float] | None = None,
) -> SyntheticEpisodeBatch:
    """Generate the frozen 10 x 14 x 48 x 6 outcome-free benchmark."""

    if type(seed) is not int or seed < 0:
        raise TypeError("seed must be a nonnegative exact int")
    normalized_shift: tuple[int, float] | None = None
    if participant_shift is not None:
        if type(participant_shift) is not tuple or len(participant_shift) != 2:
            raise TypeError("participant_shift must be an exact pair")
        participant, amount = participant_shift
        if type(participant) is not int or not 0 <= participant < _PARTICIPANTS:
            raise ValueError("participant_shift participant is out of range")
        if type(amount) not in (int, float) or type(amount) is bool:
            raise TypeError("participant_shift amount must be a built-in number")
        amount_float = float(amount)
        if not math.isfinite(amount_float):
            raise ValueError("participant_shift amount must be finite")
        normalized_shift = (participant, amount_float)

    rng = np.random.default_rng(seed)
    true_states = _state_sequences(rng)
    centers = np.array(
        [
            [-1.5, -1.0, -0.8, 0.5, -0.7, -0.4],
            [0.8, 1.2, -0.2, -0.7, 0.4, 0.8],
            [-0.4, -0.3, 1.5, 1.2, 0.8, -0.5],
            [1.3, 0.5, 0.6, -1.0, 1.4, 1.0],
        ],
        dtype="<f8",
    )
    participant_offsets = rng.normal(0.0, 0.2, size=(_PARTICIPANTS, _DOMAINS))
    noise_scale = np.array([0.35, 0.45, 0.40, 0.30, 0.50, 0.40])
    clock = np.arange(_BINS, dtype=np.float64)
    clock_effect = np.stack(
        [
            0.15 * np.sin((clock + domain * 2.0) * 2.0 * np.pi / _BINS)
            for domain in range(_DOMAINS)
        ],
        axis=1,
    )
    values = centers[true_states] + participant_offsets[:, None, None, :]
    values = values + clock_effect[None, None, :, :]
    values = values + rng.normal(0.0, noise_scale, size=values.shape)

    missing_probability = np.empty(values.shape, dtype=np.float64)
    for participant in range(_PARTICIPANTS):
        for domain in range(_DOMAINS):
            probability = 0.04 + 0.015 * (participant % 3)
            if domain >= 4:
                probability += 0.03
            missing_probability[participant, :, :, domain] = probability
    edge = (clock < 6) | (clock > 43)
    missing_probability[:, :, edge, :] += 0.04
    availability = rng.random(values.shape) >= missing_probability
    values[~availability] = np.nan
    if normalized_shift is not None:
        participant, amount = normalized_shift
        finite = np.isfinite(values[participant])
        values[participant][finite] += amount

    values = _readonly(values, np.dtype("<f8"))
    availability = _readonly(availability, np.dtype("bool"))
    true_states = _readonly(true_states, np.dtype("<i8"))
    masks = _induced_masks(seed)
    return SyntheticEpisodeBatch(
        participant_ids=tuple(f"s{i:02d}" for i in range(_PARTICIPANTS)),
        values=values,
        availability=availability,
        true_states=true_states,
        induced_masks=masks,
        content_digest=_batch_digest(
            seed,
            normalized_shift,
            values,
            availability,
            true_states,
            masks,
        ),
    )


def _boundaries(value: tuple[int, ...], name: str) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if any(type(item) is not int or not 0 < item < _BINS for item in value):
        raise ValueError(f"{name} entries must be exact interior-bin ints")
    if tuple(sorted(set(value))) != value:
        raise ValueError(f"{name} must be strictly increasing and unique")
    return value


def boundary_f1(reference: tuple[int, ...], predicted: tuple[int, ...]) -> float:
    """One-to-one boundary F1 with the frozen +/- one-bin tolerance."""

    reference = _boundaries(reference, "reference")
    predicted = _boundaries(predicted, "predicted")
    if not reference and not predicted:
        return 1.0
    unmatched = set(range(len(predicted)))
    matches = 0
    for boundary in reference:
        candidates = [
            index for index in unmatched if abs(predicted[index] - boundary) <= 1
        ]
        if candidates:
            chosen = min(candidates, key=lambda index: (abs(predicted[index] - boundary), index))
            unmatched.remove(chosen)
            matches += 1
    if matches == 0:
        return 0.0
    precision = matches / len(predicted)
    recall = matches / len(reference)
    return 2.0 * precision * recall / (precision + recall)


def _label_array(value: np.ndarray, name: str) -> np.ndarray:
    if type(value) is not np.ndarray or value.ndim != 1 or value.size == 0:
        raise TypeError(f"{name} must be a nonempty one-dimensional ndarray")
    if value.dtype != np.dtype("<i8"):
        raise TypeError(f"{name} must have exact int64 dtype")
    return value


def segment_iou(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Label-permutation-invariant mean state occupancy IoU."""

    reference = _label_array(reference, "reference")
    predicted = _label_array(predicted, "predicted")
    if reference.shape != predicted.shape:
        raise ValueError("reference and predicted must have equal shape")
    reference_labels = np.unique(reference)
    predicted_labels = np.unique(predicted)
    score = np.zeros((len(reference_labels), len(predicted_labels)), dtype=np.float64)
    for row, reference_label in enumerate(reference_labels):
        reference_mask = reference == reference_label
        for column, predicted_label in enumerate(predicted_labels):
            predicted_mask = predicted == predicted_label
            union = int(np.count_nonzero(reference_mask | predicted_mask))
            intersection = int(np.count_nonzero(reference_mask & predicted_mask))
            score[row, column] = intersection / union if union else 0.0
    rows, columns = linear_sum_assignment(-score)
    return float(score[rows, columns].sum() / max(len(reference_labels), len(predicted_labels)))


def risk_at_coverage(
    errors: np.ndarray,
    confidence: np.ndarray,
    fractions: tuple[float, ...],
) -> tuple[float, ...]:
    """Return risk after retaining the most-confident requested fractions."""

    for value, name in ((errors, "errors"), (confidence, "confidence")):
        if type(value) is not np.ndarray or value.ndim != 1 or value.dtype != np.dtype("<f8"):
            raise TypeError(f"{name} must be an exact one-dimensional float64 ndarray")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
    if errors.size == 0 or errors.shape != confidence.shape:
        raise ValueError("errors and confidence must be nonempty and shape-matched")
    if type(fractions) is not tuple or not fractions:
        raise TypeError("fractions must be a nonempty exact tuple")
    normalized: list[float] = []
    for fraction in fractions:
        if type(fraction) is not float or not 0.0 < fraction <= 1.0:
            raise ValueError("coverage fractions must be built-in floats in (0, 1]")
        normalized.append(fraction)
    order = np.lexsort((np.arange(errors.size, dtype=np.int64), -confidence))
    risks: list[float] = []
    for fraction in normalized:
        retained = int(math.ceil(fraction * errors.size))
        risks.append(float(np.mean(errors[order[:retained]])))
    return tuple(risks)


__all__ = [
    "SyntheticEpisodeBatch",
    "SyntheticMask",
    "boundary_f1",
    "generate_episode_batch",
    "risk_at_coverage",
    "segment_iou",
]
