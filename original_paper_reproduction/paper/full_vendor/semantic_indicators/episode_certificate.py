"""Structured-loss stability certificate for outcome-free episode tokens."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from .episode_hsmm import DurationDecode


_AGREEMENT_THRESHOLD = 0.80
_PERTURBATION_COUNT = 8


def _readonly(value: np.ndarray, dtype: np.dtype | str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EpisodeToken:
    start: int
    end: int
    label: int | str
    state_confidence: float
    start_boundary_confidence: float
    end_boundary_confidence: float
    evidence_quality: float


@dataclass(frozen=True)
class EpisodeCertificate:
    tokens: tuple[EpisodeToken, ...]
    aligned_states: np.ndarray
    state_agreement: np.ndarray
    boundary_positions: tuple[int, ...]
    boundary_agreement: np.ndarray
    certificate_confidence: np.ndarray
    ordinary_confidence: np.ndarray
    coverage_confidence: np.ndarray
    serialized_token_count: int
    content_digest: str


def _validate_decode(value: DurationDecode, length: int | None = None) -> int:
    if type(value) is not DurationDecode:
        raise TypeError("decode must be an exact DurationDecode")
    if type(value.states) is not np.ndarray or value.states.ndim != 1:
        raise TypeError("decode states must be one-dimensional")
    if value.states.dtype != np.dtype("<i8") or value.states.size == 0:
        raise TypeError("decode states must be nonempty exact int64")
    if np.any(value.states < 0):
        raise ValueError("decode states must be nonnegative")
    if type(value.state_posterior) is not np.ndarray:
        raise TypeError("decode posterior must be an exact ndarray")
    if value.state_posterior.dtype != np.dtype("<f8"):
        raise TypeError("decode posterior must have exact float64 dtype")
    if value.state_posterior.ndim != 2 or value.state_posterior.shape[0] != value.states.size:
        raise ValueError("decode posterior has the wrong topology")
    if not np.all(np.isfinite(value.state_posterior)):
        raise ValueError("decode posterior must be finite")
    if np.any(value.state_posterior < 0.0):
        raise ValueError("decode posterior must be nonnegative")
    if not np.allclose(value.state_posterior.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("decode posterior rows must be normalized")
    if length is not None and value.states.size != length:
        raise ValueError("all decodes must have equal length")
    return int(value.states.size)


def _align_states(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference_labels = np.unique(reference)
    candidate_labels = np.unique(candidate)
    counts = np.zeros((len(reference_labels), len(candidate_labels)), dtype=np.int64)
    for row, reference_label in enumerate(reference_labels):
        for column, candidate_label in enumerate(candidate_labels):
            counts[row, column] = int(
                np.count_nonzero((reference == reference_label) & (candidate == candidate_label))
            )
    rows, columns = linear_sum_assignment(-counts)
    mapping = {
        int(candidate_labels[column]): int(reference_labels[row])
        for row, column in zip(rows, columns, strict=True)
    }
    next_unmatched = int(reference_labels.max()) + 1
    aligned = np.empty_like(candidate)
    for label in candidate_labels:
        integer_label = int(label)
        if integer_label not in mapping:
            mapping[integer_label] = next_unmatched
            next_unmatched += 1
        aligned[candidate == label] = mapping[integer_label]
    return aligned


def _boundary_matches(reference: tuple[int, ...], candidate: np.ndarray) -> np.ndarray:
    result = np.zeros(len(reference), dtype=np.bool_)
    candidate_boundaries = tuple(
        int(value) for value in (np.flatnonzero(candidate[1:] != candidate[:-1]) + 1)
    )
    unmatched = set(range(len(candidate_boundaries)))
    for reference_index, boundary in enumerate(reference):
        choices = [
            index
            for index in unmatched
            if abs(candidate_boundaries[index] - boundary) <= 1
        ]
        if choices:
            chosen = min(
                choices,
                key=lambda index: (abs(candidate_boundaries[index] - boundary), index),
            )
            unmatched.remove(chosen)
            result[reference_index] = True
    return result


def _certificate_digest(
    tokens: tuple[EpisodeToken, ...],
    arrays: tuple[np.ndarray, ...],
    boundaries: tuple[int, ...],
) -> str:
    hasher = hashlib.sha256(b"episode-stability-certificate-v1\0")
    token_payload = [
        {
            "start": token.start,
            "end": token.end,
            "label": token.label,
            "state_confidence": token.state_confidence.hex(),
            "start_boundary_confidence": token.start_boundary_confidence.hex(),
            "end_boundary_confidence": token.end_boundary_confidence.hex(),
            "evidence_quality": token.evidence_quality.hex(),
        }
        for token in tokens
    ]
    hasher.update(
        json.dumps(
            {"tokens": token_payload, "boundaries": boundaries},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for array in arrays:
        hasher.update(array.dtype.str.encode("ascii"))
        hasher.update(json.dumps(array.shape).encode("ascii"))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def certify_episode_tokens(
    original: DurationDecode,
    perturbations: tuple[DurationDecode, ...],
    availability: np.ndarray,
) -> EpisodeCertificate:
    """Certify original runs by agreement across exactly eight loss perturbations."""

    length = _validate_decode(original)
    if type(perturbations) is not tuple or len(perturbations) != _PERTURBATION_COUNT:
        raise ValueError("perturbations must be an exact tuple of length eight")
    if type(availability) is not np.ndarray or availability.ndim != 2:
        raise TypeError("availability must be an exact two-dimensional ndarray")
    if availability.dtype != np.dtype("bool") or availability.shape[0] != length:
        raise TypeError("availability must be exact bool with matching time length")
    if availability.shape[1] == 0:
        raise ValueError("availability must contain at least one domain")
    aligned_rows: list[np.ndarray] = []
    for perturbation in perturbations:
        _validate_decode(perturbation, length)
        aligned_rows.append(_align_states(original.states, perturbation.states))
    aligned_rows.sort(key=lambda row: row.tobytes(order="C"))
    aligned = np.stack(aligned_rows, axis=0)
    state_agreement = np.mean(aligned == original.states[None, :], axis=0)
    boundaries = tuple(
        int(value)
        for value in (np.flatnonzero(original.states[1:] != original.states[:-1]) + 1)
    )
    if boundaries:
        boundary_hits = np.stack(
            [_boundary_matches(boundaries, row) for row in aligned], axis=0
        )
        boundary_agreement = np.mean(boundary_hits, axis=0)
    else:
        boundary_agreement = np.empty(0, dtype=np.float64)
    boundary_lookup = {
        boundary: float(boundary_agreement[index])
        for index, boundary in enumerate(boundaries)
    }
    coverage_confidence = np.mean(availability, axis=1, dtype=np.float64)
    ordinary_confidence = np.max(original.state_posterior, axis=1)
    certificate_confidence = np.zeros(length, dtype=np.float64)
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, length]
    tokens: list[EpisodeToken] = []
    has_internal_boundary = bool(boundaries)
    for start_value, end_value in zip(starts, ends, strict=True):
        start = int(start_value)
        end = int(end_value)
        state_confidence = float(np.mean(state_agreement[start:end]))
        start_boundary = 1.0 if start == 0 else boundary_lookup[start]
        end_boundary = 1.0 if end == length else boundary_lookup[end]
        evidence_quality = float(np.mean(availability[start:end]))
        confidence = (
            state_agreement[start:end]
            * min(start_boundary, end_boundary)
            * coverage_confidence[start:end]
        )
        certificate_confidence[start:end] = confidence
        accepted = (
            has_internal_boundary
            and state_confidence >= _AGREEMENT_THRESHOLD
            and start_boundary >= _AGREEMENT_THRESHOLD
            and end_boundary >= _AGREEMENT_THRESHOLD
            and evidence_quality > 0.0
        )
        label: int | str = int(original.states[start]) if accepted else "AMBIGUOUS"
        tokens.append(
            EpisodeToken(
                start,
                end,
                label,
                state_confidence,
                start_boundary,
                end_boundary,
                evidence_quality,
            )
        )
    aligned = _readonly(aligned, np.dtype("<i8"))
    state_agreement = _readonly(state_agreement, np.dtype("<f8"))
    boundary_agreement = _readonly(boundary_agreement, np.dtype("<f8"))
    certificate_confidence = _readonly(certificate_confidence, np.dtype("<f8"))
    ordinary_confidence = _readonly(ordinary_confidence, np.dtype("<f8"))
    coverage_confidence = _readonly(coverage_confidence, np.dtype("<f8"))
    frozen_tokens = tuple(tokens)
    digest = _certificate_digest(
        frozen_tokens,
        (
            aligned,
            state_agreement,
            boundary_agreement,
            certificate_confidence,
            ordinary_confidence,
            coverage_confidence,
        ),
        boundaries,
    )
    return EpisodeCertificate(
        frozen_tokens,
        aligned,
        state_agreement,
        boundaries,
        boundary_agreement,
        certificate_confidence,
        ordinary_confidence,
        coverage_confidence,
        len(frozen_tokens),
        digest,
    )


__all__ = [
    "EpisodeCertificate",
    "EpisodeToken",
    "certify_episode_tokens",
]
