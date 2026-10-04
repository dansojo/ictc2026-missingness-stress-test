"""Outcome-free personal phase templates for selective semantic compression.

This module starts with the pure, auditable PhaseGuard calibration boundary.
Raw replay integration is intentionally separate: a template can be trusted
only after its online prefix, recipient exclusion, and uncertainty arithmetic
are independently testable.

Threat boundary: underscore construction closures prevent accidental public
construction and ``dataclasses.replace`` token reuse; they are not a security
boundary against arbitrary in-process Python code.  Canonical authority belongs
to the exact study runner, which must rebuild these values from reviewed public
prepared-phase, mask-geometry, and sensor-calendar views.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
import json
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


_PHASE_REQUIRED_COLUMNS = (
    "subject_id",
    "sensor_day_id",
    "elapsed_day_index",
    "phase_bin",
    "day_type",
    "primitive",
    "value",
    "support",
)
_PHASE_KEY_COLUMNS = (
    "subject_id",
    "sensor_day_id",
    "primitive",
    "phase_bin",
    "day_type",
)
_PHASE_CANONICAL_ORDER = (
    "subject_id",
    "elapsed_day_index",
    "sensor_day_id",
    "primitive",
    "day_type",
    "phase_bin",
)
_DAY_TYPES = ("weekday", "weekend")
_CORE_PRIMITIVE_OPERATIONS = {
    "screen_load_24h": "binary_load",
    "usage_load_24h": "additive_load",
    "screen_disengagement_p90": "evening_p90",
    "usage_disengagement_p90": "evening_p90",
    "phone_activity_load_24h": "activity_load",
    "step_load_24h": "additive_load",
    "mobile_light_exposure_24h": "exposure_mean",
    "wearable_light_exposure_24h": "exposure_mean",
}
_RATE_OPERATIONS = {"binary_load", "activity_load", "exposure_mean"}
_EXPECTED_UNITS_PER_BIN = {
    "screen_load_24h": 30.0,
    "usage_load_24h": 3.0,
    "screen_disengagement_p90": 30.0,
    "usage_disengagement_p90": 3.0,
    "phone_activity_load_24h": 30.0,
    "step_load_24h": 30.0,
    "mobile_light_exposure_24h": 3.0,
    "wearable_light_exposure_24h": 30.0,
}
_FROZEN_METHODS = (
    "personal_global",
    "population_phase",
    "personal_phase",
    "phaseguard",
)


def _make_construction_boundary():
    token = object()

    def construct(cls: type, /, **kwargs: object):
        return cls(_build_token=token, **kwargs)

    def validates(candidate: object) -> bool:
        return candidate is token

    return construct, validates


_construct_statistics, _valid_statistics_token = _make_construction_boundary()
_construct_template_bundle, _valid_template_token = _make_construction_boundary()
_construct_replay_result, _valid_result_token = _make_construction_boundary()


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_token(value: float) -> str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return number.hex()


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _policy_digest(policy: "PhaseGuardPolicy") -> str:
    return _sha256_payload(
        {
            "schema": "phaseguard-policy-v1",
            "bin_minutes": policy.bin_minutes,
            "tau": _float_token(policy.tau),
            "min_personal_support": policy.min_personal_support,
            "max_calibration_day": policy.max_calibration_day,
        }
    )


def _validate_policy_integrity(policy: object) -> "PhaseGuardPolicy":
    if type(policy) is not PhaseGuardPolicy:
        raise TypeError("policy must be an exact PhaseGuardPolicy")
    if (
        type(policy.bin_minutes) is not int
        or policy.bin_minutes != 30
        or type(policy.tau) is not float
        or policy.tau != 7.0
        or type(policy.min_personal_support) is not int
        or policy.min_personal_support != 3
        or type(policy.max_calibration_day) is not int
        or policy.max_calibration_day != 14
        or type(policy.policy_digest) is not str
        or _policy_digest(policy) != policy.policy_digest
    ):
        raise ValueError("policy integrity does not match frozen PhaseGuard")
    return policy


@dataclass(frozen=True)
class PhaseGuardPolicy:
    """Frozen outcome-free calibration policy."""

    bin_minutes: int = 30
    tau: float = 7.0
    min_personal_support: int = 3
    max_calibration_day: int = 14
    policy_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.bin_minutes) is not int or self.bin_minutes <= 0:
            raise ValueError("bin_minutes must be an exact positive integer")
        if 1440 % self.bin_minutes != 0:
            raise ValueError("bin_minutes must divide 1440 exactly")
        if self.bin_minutes != 30:
            raise ValueError("PhaseGuard bin_minutes is frozen at 30")
        if type(self.tau) is not float or self.tau != 7.0:
            raise ValueError("PhaseGuard tau must be the exact frozen float 7.0")
        if type(self.min_personal_support) is not int or self.min_personal_support != 3:
            raise ValueError("PhaseGuard min_personal_support is frozen at exact integer 3")
        if type(self.max_calibration_day) is not int or self.max_calibration_day != 14:
            raise ValueError("PhaseGuard max_calibration_day is frozen at exact integer 14")
        object.__setattr__(self, "policy_digest", _policy_digest(self))


@dataclass(frozen=True)
class PhaseEstimate:
    """One online personal/population phase estimate with provenance."""

    value: float
    template_mad: float
    personal_value: float
    personal_mad: float
    population_value: float
    population_mad: float
    n_eff: float
    personal_observations: int
    population_participants: int
    personal_weight: float
    source_status: str
    personal_source_days: tuple[int, ...]
    population_subjects: tuple[str, ...]


@dataclass(frozen=True)
class PhaseGuardReliability:
    """Components of the preregistered PhaseGuard reliability score."""

    reliability: float
    coverage_component: float
    support_component: float
    uncertainty_component: float
    critical_component: float


def _exact_finite_real(value: object, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be an exact finite number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _deterministic_fsum(values: Iterable[float]) -> float:
    """Sum one finite nonnegative multiset independently of caller order."""

    return float(math.fsum(sorted(float(value) for value in values)))


@dataclass(frozen=True)
class PhaseBinSufficientStatistic:
    """Immutable observed statistics and exact mask geometry for one phase bin."""

    phase_bin: int
    expected_units: float
    observed_units: float
    observed_numerator: float
    masked_fraction: float
    baseline_units: float | None = None
    event_minutes: tuple[float, ...] = ()
    event_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.phase_bin, (bool, np.bool_))
            or not isinstance(self.phase_bin, (int, np.integer))
            or not 0 <= int(self.phase_bin) < 48
        ):
            raise ValueError("phase_bin must be an exact integer in [0, 48)")
        expected = _exact_finite_real(
            self.expected_units, "expected_units", minimum=np.nextafter(0.0, 1.0)
        )
        observed = _exact_finite_real(self.observed_units, "observed_units")
        numerator = _exact_finite_real(
            self.observed_numerator, "observed_numerator"
        )
        masked = _exact_finite_real(self.masked_fraction, "masked_fraction")
        if masked > 1.0:
            raise ValueError("masked_fraction must be in [0, 1]")
        induced_deleted_units = expected * masked
        inferred_baseline = observed + induced_deleted_units
        if self.baseline_units is None:
            baseline = inferred_baseline
        else:
            baseline = _exact_finite_real(self.baseline_units, "baseline_units")
            if not math.isclose(
                baseline, inferred_baseline, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "baseline_units contradict observed support and induced mask"
                )
        if baseline > expected + 1e-12:
            raise ValueError("observed_units contradict exact masked geometry")
        if type(self.event_minutes) is not tuple or type(self.event_weights) is not tuple:
            raise ValueError("event arrays must be exact tuples")
        if len(self.event_minutes) != len(self.event_weights):
            raise ValueError("event minutes and weights must have equal length")
        minutes = tuple(
            _exact_finite_real(value, "event minute") for value in self.event_minutes
        )
        weights = tuple(
            _exact_finite_real(
                value, "event weight", minimum=np.nextafter(0.0, 1.0)
            )
            for value in self.event_weights
        )
        if any(value >= 1440.0 for value in minutes):
            raise ValueError("event minutes must be in [0, 1440)")
        lower = int(self.phase_bin) * 30.0
        if any(not lower <= value < lower + 30.0 for value in minutes):
            raise ValueError("event minutes must lie inside phase_bin")
        if minutes != tuple(sorted(minutes)):
            raise ValueError("event minutes must be deterministically sorted")
        if (observed == 0.0 or masked == 1.0) and minutes:
            raise ValueError("fully deleted or unobserved bins cannot contain events")
        if observed == 0.0 and numerator != 0.0:
            raise ValueError("positive numerator requires observed units")
        object.__setattr__(self, "phase_bin", int(self.phase_bin))
        object.__setattr__(self, "expected_units", expected)
        object.__setattr__(self, "observed_units", observed)
        object.__setattr__(self, "observed_numerator", numerator)
        object.__setattr__(self, "masked_fraction", masked)
        object.__setattr__(self, "baseline_units", baseline)
        object.__setattr__(self, "event_minutes", minutes)
        object.__setattr__(self, "event_weights", weights)


def _bin_digest_payload(item: PhaseBinSufficientStatistic) -> dict[str, object]:
    return {
        "phase_bin": item.phase_bin,
        "expected_units": _float_token(item.expected_units),
        "observed_units": _float_token(item.observed_units),
        "observed_numerator": _float_token(item.observed_numerator),
        "masked_fraction": _float_token(item.masked_fraction),
        "baseline_units": _float_token(float(item.baseline_units)),
        "event_minutes": [_float_token(value) for value in item.event_minutes],
        "event_weights": [_float_token(value) for value in item.event_weights],
    }


@dataclass(frozen=True)
class PhaseSufficientStatistics:
    """Trusted masked-day statistics bound to target, source and mask geometry."""

    target_subject: str
    target_elapsed_day: int
    sensor_day_id: int
    day_type: str
    primitive_name: str
    operation: str
    masked_value: float
    source_record_digest: str
    mask_digest: str
    calendar_digest: str
    policy_digest: str
    bins: tuple[PhaseBinSufficientStatistic, ...]
    _build_token: InitVar[object]
    observed_fraction: float = field(init=False)
    induced_mask_fraction: float = field(init=False)
    critical_window_coverage: float = field(init=False)
    content_digest: str = field(init=False)

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_statistics_token(_build_token):
            raise TypeError(
                "PhaseSufficientStatistics must be created by the trusted factory"
            )
        if type(self.target_subject) is not str or not self.target_subject:
            raise ValueError("target_subject must be a nonempty exact string")
        target_day = _exact_positive_int(
            self.target_elapsed_day, "target_elapsed_day"
        )
        if type(self.sensor_day_id) is not int or self.sensor_day_id < 0:
            raise ValueError("sensor_day_id must be an exact nonnegative integer")
        if type(self.day_type) is not str or self.day_type not in _DAY_TYPES:
            raise ValueError(f"day_type must be one of {_DAY_TYPES}")
        if type(self.primitive_name) is not str or self.primitive_name not in (
            _CORE_PRIMITIVE_OPERATIONS
        ):
            raise ValueError("primitive_name must be one frozen G2 primitive")
        if type(self.operation) is not str or self.operation != (
            _CORE_PRIMITIVE_OPERATIONS[self.primitive_name]
        ):
            raise ValueError("operation is crosswired with primitive_name")
        if isinstance(self.masked_value, (bool, np.bool_)) or not isinstance(
            self.masked_value, (int, float, np.integer, np.floating)
        ):
            raise ValueError("masked_value must be an exact numeric value")
        masked_value = float(self.masked_value)
        if np.isinf(masked_value):
            raise ValueError("masked_value cannot be infinite")
        _require_sha256(self.source_record_digest, "source_record_digest")
        _require_sha256(self.mask_digest, "mask_digest")
        _require_sha256(self.calendar_digest, "calendar_digest")
        _require_sha256(self.policy_digest, "policy_digest")
        if self.policy_digest != _validate_policy_integrity(PhaseGuardPolicy()).policy_digest:
            raise ValueError("policy_digest does not identify frozen PhaseGuard")
        if type(self.bins) is not tuple or not self.bins:
            raise ValueError("bins must be a nonempty exact tuple")
        if not all(type(item) is PhaseBinSufficientStatistic for item in self.bins):
            raise ValueError("bins must contain exact PhaseBinSufficientStatistic values")
        phase_bins = tuple(item.phase_bin for item in self.bins)
        if len(set(phase_bins)) != len(phase_bins):
            raise ValueError("phase bins must be unique")
        if phase_bins != tuple(sorted(phase_bins)):
            raise ValueError("phase bins must be sorted")
        expected_grid = (
            tuple(range(36, 48))
            if self.operation == "evening_p90"
            else tuple(range(48))
        )
        if phase_bins != expected_grid:
            raise ValueError("bins must contain the complete frozen phase grid")
        expected_units = _EXPECTED_UNITS_PER_BIN[self.primitive_name]
        if any(item.expected_units != expected_units for item in self.bins):
            raise ValueError("expected_units contradict frozen primitive cadence")
        if self.operation == "evening_p90":
            for item in self.bins:
                if not math.isclose(
                    _deterministic_fsum(item.event_weights),
                    item.observed_numerator,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("event weight must equal observed event mass")
        elif any(item.event_minutes or item.event_weights for item in self.bins):
            raise ValueError("non-event operation cannot contain event arrays")
        if any(item.observed_numerator < 0.0 for item in self.bins):
            raise ValueError("observed numerator must be nonnegative")
        total_numerator = float(
            sum(item.observed_numerator for item in self.bins)
        )
        total_observed_units = float(sum(item.observed_units for item in self.bins))
        if self.operation == "additive_load":
            if np.isfinite(masked_value):
                if not math.isclose(
                    masked_value,
                    total_numerator,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "additive masked_value contradicts retained contributions"
                    )
            elif total_observed_units > 0.0:
                raise ValueError("additive masked_value must be finite with support")
        elif self.operation in _RATE_OPERATIONS:
            if self.operation in {"binary_load", "activity_load"} and any(
                item.observed_numerator > item.observed_units + 1e-12
                for item in self.bins
            ):
                raise ValueError("binary/activity numerator exceeds observed units")
            if total_observed_units > 0.0:
                expected_rate = total_numerator / total_observed_units
                if not np.isfinite(masked_value) or not math.isclose(
                    masked_value,
                    expected_rate,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "rate masked_value contradicts retained numerator/support"
                    )
                if (
                    self.operation in {"binary_load", "activity_load"}
                    and not 0.0 <= masked_value <= 1.0
                ):
                    raise ValueError("binary/activity masked_value must be in [0, 1]")
            elif np.isfinite(masked_value):
                raise ValueError("rate masked_value must be missing without support")
        elif self.operation == "evening_p90":
            event_minutes = np.asarray(
                [minute for item in self.bins for minute in item.event_minutes],
                dtype=float,
            )
            event_weights = np.asarray(
                [weight for item in self.bins for weight in item.event_weights],
                dtype=float,
            )
            if event_weights.size:
                order = np.argsort(event_minutes, kind="stable")
                cumulative = np.cumsum(event_weights[order])
                index = int(
                    np.searchsorted(cumulative, 0.9 * cumulative[-1], side="left")
                )
                expected_p90 = float(event_minutes[order][index])
                if not np.isfinite(masked_value) or not math.isclose(
                    masked_value,
                    expected_p90,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "timing masked_value contradicts retained weighted p90"
                    )
            elif np.isfinite(masked_value):
                raise ValueError("timing masked_value must be missing without events")
        total = float(sum(item.expected_units for item in self.bins))
        observed_geometry = float(sum(item.observed_units for item in self.bins))
        induced_deleted = float(
            sum(
                float(item.baseline_units) - item.observed_units
                for item in self.bins
            )
        )
        observed_fraction = observed_geometry / total
        induced_mask_fraction = induced_deleted / total
        critical = observed_fraction if self.operation == "evening_p90" else 1.0
        object.__setattr__(self, "target_elapsed_day", target_day)
        object.__setattr__(self, "masked_value", masked_value)
        object.__setattr__(self, "observed_fraction", float(observed_fraction))
        object.__setattr__(
            self, "induced_mask_fraction", float(induced_mask_fraction)
        )
        object.__setattr__(self, "critical_window_coverage", critical)
        object.__setattr__(self, "content_digest", _statistics_content_digest(self))


def _statistics_content_digest(statistics: PhaseSufficientStatistics) -> str:
    return _sha256_payload(
        {
            "schema": "phaseguard-statistics-v1",
            "target_subject": statistics.target_subject,
            "target_elapsed_day": statistics.target_elapsed_day,
            "sensor_day_id": statistics.sensor_day_id,
            "day_type": statistics.day_type,
            "primitive_name": statistics.primitive_name,
            "operation": statistics.operation,
            "masked_value": _float_token(statistics.masked_value),
            "source_record_digest": statistics.source_record_digest,
            "mask_digest": statistics.mask_digest,
            "calendar_digest": statistics.calendar_digest,
            "policy_digest": statistics.policy_digest,
            "observed_fraction": _float_token(statistics.observed_fraction),
            "induced_mask_fraction": _float_token(
                statistics.induced_mask_fraction
            ),
            "critical_window_coverage": _float_token(
                statistics.critical_window_coverage
            ),
            "bins": [_bin_digest_payload(item) for item in statistics.bins],
        }
    )


@dataclass(frozen=True)
class PhaseTemplateBin:
    """One frozen personal/population template component used for repair."""

    phase_bin: int
    value: float
    template_mad: float
    population_scale: float
    n_eff: float
    source_status: str
    representative_minute: float | None = None
    expected_event_count: float = 0.0
    expected_event_mass: float = 0.0
    event_minutes: tuple[float, ...] = ()
    event_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.phase_bin, (bool, np.bool_))
            or not isinstance(self.phase_bin, (int, np.integer))
            or not 0 <= int(self.phase_bin) < 48
        ):
            raise ValueError("phase_bin must be an exact integer in [0, 48)")
        value = _exact_finite_real(self.value, "template value")
        mad = _exact_finite_real(self.template_mad, "template_mad")
        scale = _exact_finite_real(self.population_scale, "population_scale")
        n_eff = _exact_finite_real(self.n_eff, "n_eff")
        events = _exact_finite_real(self.expected_event_count, "expected_event_count")
        event_mass = _exact_finite_real(
            self.expected_event_mass, "expected_event_mass"
        )
        if (events > 0.0) != (event_mass > 0.0):
            raise ValueError(
                "expected event count and mass must be jointly positive or zero"
            )
        allowed_status = {
            "shrunk_personal",
            "personal_only",
            "population_fallback",
        }
        if type(self.source_status) is not str or self.source_status not in allowed_status:
            raise ValueError("source_status is not a supported template provenance")
        minute: float | None
        if self.representative_minute is None:
            minute = None
        else:
            minute = _exact_finite_real(
                self.representative_minute, "representative_minute"
            )
            lower = int(self.phase_bin) * 30.0
            if not lower <= minute < lower + 30.0:
                raise ValueError("representative_minute must lie inside phase_bin")
        if type(self.event_minutes) is not tuple or type(self.event_weights) is not tuple:
            raise ValueError("template event arrays must be exact tuples")
        if len(self.event_minutes) != len(self.event_weights):
            raise ValueError("template event arrays must have equal length")
        event_minutes = tuple(
            _exact_finite_real(value, "template event minute")
            for value in self.event_minutes
        )
        event_weights = tuple(
            _exact_finite_real(
                value,
                "template event weight",
                minimum=np.nextafter(0.0, 1.0),
            )
            for value in self.event_weights
        )
        lower = int(self.phase_bin) * 30.0
        if any(not lower <= value < lower + 30.0 for value in event_minutes):
            raise ValueError("template event minutes must lie inside phase_bin")
        if event_minutes != tuple(sorted(event_minutes)):
            raise ValueError("template event minutes must be deterministically sorted")
        if (events > 0.0 or event_mass > 0.0) and not event_minutes:
            raise ValueError(
                "positive expected event count/mass requires an event distribution"
            )
        if event_minutes and not math.isclose(
            _deterministic_fsum(event_weights), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("template event weights must sum to one")
        object.__setattr__(self, "phase_bin", int(self.phase_bin))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "template_mad", mad)
        object.__setattr__(self, "population_scale", scale)
        object.__setattr__(self, "n_eff", n_eff)
        object.__setattr__(self, "representative_minute", minute)
        object.__setattr__(self, "expected_event_count", events)
        object.__setattr__(self, "expected_event_mass", event_mass)
        object.__setattr__(self, "event_minutes", event_minutes)
        object.__setattr__(self, "event_weights", event_weights)


def _template_bin_digest_payload(item: PhaseTemplateBin | None) -> object:
    if item is None:
        return None
    return {
        "phase_bin": item.phase_bin,
        "value": _float_token(item.value),
        "template_mad": _float_token(item.template_mad),
        "population_scale": _float_token(item.population_scale),
        "n_eff": _float_token(item.n_eff),
        "source_status": item.source_status,
        "representative_minute": (
            None
            if item.representative_minute is None
            else _float_token(item.representative_minute)
        ),
        "expected_event_count": _float_token(item.expected_event_count),
        "expected_event_mass": _float_token(item.expected_event_mass),
        "event_minutes": [_float_token(value) for value in item.event_minutes],
        "event_weights": [_float_token(value) for value in item.event_weights],
    }


@dataclass(frozen=True)
class PhaseTemplateBundle:
    """One complete, immutable comparator template with online provenance."""

    target_subject: str
    target_elapsed_day: int
    target_sensor_day_id: int
    day_type: str
    primitive_name: str
    operation: str
    method: str
    policy_digest: str
    phase_source_digest: str
    calendar_digest: str
    recipient_exclusion_digest: str
    personal_source_days: tuple[int, ...]
    population_subjects: tuple[str, ...]
    bins: tuple[PhaseTemplateBin | None, ...]
    _build_token: InitVar[object]
    content_digest: str = field(init=False)

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_template_token(_build_token):
            raise TypeError("PhaseTemplateBundle must be created by the trusted factory")
        if type(self.target_subject) is not str or not self.target_subject:
            raise ValueError("target_subject must be a nonempty exact string")
        target_day = _exact_positive_int(
            self.target_elapsed_day, "target_elapsed_day"
        )
        if type(self.target_sensor_day_id) is not int or self.target_sensor_day_id < 0:
            raise ValueError("target_sensor_day_id must be an exact nonnegative integer")
        if type(self.day_type) is not str or self.day_type not in _DAY_TYPES:
            raise ValueError(f"day_type must be one of {_DAY_TYPES}")
        if (
            type(self.primitive_name) is not str
            or self.primitive_name not in _CORE_PRIMITIVE_OPERATIONS
        ):
            raise ValueError("primitive_name must be one frozen G2 primitive")
        if self.operation != _CORE_PRIMITIVE_OPERATIONS[self.primitive_name]:
            raise ValueError("operation is crosswired with primitive_name")
        if type(self.method) is not str or self.method not in _FROZEN_METHODS:
            raise ValueError("method must be one frozen PhaseGuard comparator")
        for name in (
            "policy_digest",
            "phase_source_digest",
            "calendar_digest",
            "recipient_exclusion_digest",
        ):
            _require_sha256(getattr(self, name), name)
        if self.policy_digest != _validate_policy_integrity(PhaseGuardPolicy()).policy_digest:
            raise ValueError("policy_digest does not identify frozen PhaseGuard")
        if type(self.personal_source_days) is not tuple or any(
            type(day) is not int for day in self.personal_source_days
        ):
            raise ValueError("personal_source_days must be an exact integer tuple")
        permitted_days = online_calibration_days(target_day)
        if (
            self.personal_source_days != tuple(sorted(set(self.personal_source_days)))
            or any(day not in permitted_days for day in self.personal_source_days)
        ):
            raise ValueError("personal_source_days violate the frozen online prefix")
        if type(self.population_subjects) is not tuple or any(
            type(subject) is not str or not subject
            for subject in self.population_subjects
        ):
            raise ValueError("population_subjects must be an exact string tuple")
        if self.population_subjects != tuple(sorted(set(self.population_subjects))):
            raise ValueError("population_subjects must be unique and sorted")
        if self.target_subject in self.population_subjects:
            raise ValueError("population_subjects contain the held-out recipient")
        expected_grid = (
            tuple(range(36, 48))
            if self.operation == "evening_p90"
            else tuple(range(48))
        )
        if type(self.bins) is not tuple or len(self.bins) != len(expected_grid):
            raise ValueError("bundle bins must contain the complete frozen phase grid")
        for expected_phase, item in zip(expected_grid, self.bins, strict=True):
            if item is not None and type(item) is not PhaseTemplateBin:
                raise ValueError("bundle bins contain a forged template type")
            if item is not None and item.phase_bin != expected_phase:
                raise ValueError("bundle template phase is crosswired")
            if (
                item is not None
                and self.operation in {"binary_load", "activity_load"}
                and not 0.0 <= item.value <= 1.0
            ):
                raise ValueError("binary/activity template value must be in [0, 1]")
        expected_recipient_digest = _sha256_payload(
            {
                "schema": "phaseguard-recipient-audit-v1",
                "target_subject": self.target_subject,
                "target_elapsed_day": target_day,
                "method": self.method,
                "personal_source_days": list(self.personal_source_days),
                "population_subjects": list(self.population_subjects),
                "phase_source_digest": self.phase_source_digest,
                "policy_digest": self.policy_digest,
            }
        )
        if self.recipient_exclusion_digest != expected_recipient_digest:
            raise ValueError("recipient exclusion audit digest is inconsistent")
        object.__setattr__(self, "target_elapsed_day", target_day)
        object.__setattr__(self, "content_digest", _template_bundle_content_digest(self))


def _template_bundle_content_digest(bundle: PhaseTemplateBundle) -> str:
    return _sha256_payload(
        {
            "schema": "phaseguard-template-bundle-v2",
            "target_subject": bundle.target_subject,
            "target_elapsed_day": bundle.target_elapsed_day,
            "target_sensor_day_id": bundle.target_sensor_day_id,
            "day_type": bundle.day_type,
            "primitive_name": bundle.primitive_name,
            "operation": bundle.operation,
            "method": bundle.method,
            "policy_digest": bundle.policy_digest,
            "phase_source_digest": bundle.phase_source_digest,
            "calendar_digest": bundle.calendar_digest,
            "recipient_exclusion_digest": bundle.recipient_exclusion_digest,
            "personal_source_days": list(bundle.personal_source_days),
            "population_subjects": list(bundle.population_subjects),
            "bins": [_template_bin_digest_payload(item) for item in bundle.bins],
        }
    )


def build_phase_sufficient_statistics(
    *,
    target_subject: str,
    target_elapsed_day: int,
    sensor_day_id: int,
    day_type: str,
    primitive_name: str,
    operation: str,
    masked_value: float,
    source_record_digest: str,
    mask_digest: str,
    calendar_digest: str,
    bins: tuple[PhaseBinSufficientStatistic, ...],
    policy: PhaseGuardPolicy | None = None,
) -> PhaseSufficientStatistics:
    """Build statistics whose digest binds reviewed target and mask geometry."""

    frozen = policy if policy is not None else PhaseGuardPolicy()
    _validate_policy_integrity(frozen)
    return _construct_statistics(
        PhaseSufficientStatistics,
        target_subject=target_subject,
        target_elapsed_day=target_elapsed_day,
        sensor_day_id=sensor_day_id,
        day_type=day_type,
        primitive_name=primitive_name,
        operation=operation,
        masked_value=masked_value,
        source_record_digest=source_record_digest,
        mask_digest=mask_digest,
        calendar_digest=calendar_digest,
        policy_digest=frozen.policy_digest,
        bins=bins,
    )


def _canonical_phase_frame_digest(rows: pd.DataFrame) -> str:
    ordered = rows.sort_values(
        list(_PHASE_CANONICAL_ORDER),
        kind="mergesort",
    )
    records: list[dict[str, object]] = []
    for row in ordered.itertuples(index=False):
        record: dict[str, object] = {
            "subject_id": str(row.subject_id),
            "elapsed_day_index": int(row.elapsed_day_index),
            "sensor_day_id": int(row.sensor_day_id),
            "primitive": str(row.primitive),
            "day_type": str(row.day_type),
            "phase_bin": int(row.phase_bin),
            "value": _float_token(float(row.value)),
            "support": _float_token(float(row.support)),
        }
        if hasattr(row, "event_count"):
            record["event_count"] = int(row.event_count)
        if hasattr(row, "event_mass"):
            record["event_mass"] = _float_token(float(row.event_mass))
        if hasattr(row, "event_minutes"):
            record["event_minutes"] = [
                _float_token(value) for value in row.event_minutes
            ]
            record["event_weights"] = [
                _float_token(value) for value in row.event_weights
            ]
        records.append(record)
    return _sha256_payload({"schema": "phaseguard-source-v2", "records": records})


def _canonical_calendar_digest(rows: pd.DataFrame) -> str:
    calendar = (
        rows.loc[
            :, ["subject_id", "elapsed_day_index", "sensor_day_id", "day_type"]
        ]
        .drop_duplicates()
        .sort_values(
            ["subject_id", "elapsed_day_index", "sensor_day_id", "day_type"],
            kind="mergesort",
        )
    )
    records = [
        [str(row.subject_id), int(row.elapsed_day_index), int(row.sensor_day_id), str(row.day_type)]
        for row in calendar.itertuples(index=False)
    ]
    return _sha256_payload({"schema": "phaseguard-calendar-v1", "records": records})


def fit_phase_template_bundle(
    phase_rows: pd.DataFrame,
    *,
    target_subject: str,
    target_elapsed_day: int,
    target_sensor_day_id: int,
    primitive: str,
    day_type: str,
    policy: PhaseGuardPolicy | None = None,
    method: str = "phaseguard",
) -> PhaseTemplateBundle:
    """Fit one complete comparator bundle through a single frozen source view."""

    frozen = policy if policy is not None else PhaseGuardPolicy()
    _validate_policy_integrity(frozen)
    if type(target_subject) is not str or not target_subject:
        raise ValueError("target_subject must be a nonempty exact string")
    target_day = _exact_positive_int(target_elapsed_day, "target_elapsed_day")
    if type(target_sensor_day_id) is not int or target_sensor_day_id < 0:
        raise ValueError("target_sensor_day_id must be an exact nonnegative integer")
    if type(primitive) is not str or primitive not in _CORE_PRIMITIVE_OPERATIONS:
        raise ValueError("primitive must be one frozen G2 primitive")
    if type(day_type) is not str or day_type not in _DAY_TYPES:
        raise ValueError(f"day_type must be one of {_DAY_TYPES}")
    if type(method) is not str or method not in _FROZEN_METHODS:
        raise ValueError("method must be one frozen PhaseGuard comparator")
    rows = _validate_phase_rows(phase_rows, frozen)
    operation = _CORE_PRIMITIVE_OPERATIONS[primitive]
    if operation == "evening_p90":
        rows = _validate_timing_event_rows(
            rows, primitive=primitive, day_type=day_type, policy=frozen
        )
    prefix = online_calibration_days(target_day)
    forbidden_target_rows = rows.loc[
        rows["subject_id"].eq(target_subject)
        & ~rows["elapsed_day_index"].isin(prefix)
    ]
    if not forbidden_target_rows.empty:
        raise ValueError("phase_rows contains target or future target rows")
    phase_source_digest = _canonical_phase_frame_digest(rows)
    calendar_digest = _canonical_calendar_digest(rows)
    relevant = rows.loc[
        rows["primitive"].eq(primitive) & rows["day_type"].eq(day_type)
    ]
    personal = _finite_supported(
        relevant.loc[
            relevant["subject_id"].eq(target_subject)
            & relevant["elapsed_day_index"].isin(prefix)
        ]
    )
    personal_source_days = tuple(
        sorted(personal["elapsed_day_index"].astype(int).unique().tolist())
    )
    population_subjects = tuple(
        sorted(
            _finite_supported(
                relevant.loc[~relevant["subject_id"].eq(target_subject)]
            )["subject_id"]
            .astype(str)
            .unique()
            .tolist()
        )
    )
    phase_grid = range(36, 48) if operation == "evening_p90" else range(48)
    templates = tuple(
        fit_phase_template_bin(
            rows,
            target_subject=target_subject,
            target_elapsed_day=target_day,
            primitive=primitive,
            phase_bin=phase_bin,
            day_type=day_type,
            policy=frozen,
            method=method,
        )
        for phase_bin in phase_grid
    )
    recipient_exclusion_digest = _sha256_payload(
        {
            "schema": "phaseguard-recipient-audit-v1",
            "target_subject": target_subject,
            "target_elapsed_day": target_day,
            "method": method,
            "personal_source_days": list(personal_source_days),
            "population_subjects": list(population_subjects),
            "phase_source_digest": phase_source_digest,
            "policy_digest": frozen.policy_digest,
        }
    )
    return _construct_template_bundle(
        PhaseTemplateBundle,
        target_subject=target_subject,
        target_elapsed_day=target_day,
        target_sensor_day_id=target_sensor_day_id,
        day_type=day_type,
        primitive_name=primitive,
        operation=operation,
        method=method,
        policy_digest=frozen.policy_digest,
        phase_source_digest=phase_source_digest,
        calendar_digest=calendar_digest,
        recipient_exclusion_digest=recipient_exclusion_digest,
        personal_source_days=personal_source_days,
        population_subjects=population_subjects,
        bins=templates,
    )


def fit_phase_template_bundles(
    phase_rows: pd.DataFrame,
    *,
    target_subject: str,
    target_elapsed_day: int,
    target_sensor_day_id: int,
    primitive: str,
    day_type: str,
    policy: PhaseGuardPolicy | None = None,
) -> tuple[PhaseTemplateBundle, ...]:
    """Build every frozen repair comparator from one immutable source snapshot."""

    if not isinstance(phase_rows, pd.DataFrame):
        raise TypeError("phase_rows must be a pandas DataFrame")
    snapshot = phase_rows.copy(deep=True)
    frozen = policy if policy is not None else PhaseGuardPolicy()
    _validate_policy_integrity(frozen)
    bundles = tuple(
        fit_phase_template_bundle(
            snapshot,
            target_subject=target_subject,
            target_elapsed_day=target_elapsed_day,
            target_sensor_day_id=target_sensor_day_id,
            primitive=primitive,
            day_type=day_type,
            policy=frozen,
            method=method,
        )
        for method in _FROZEN_METHODS
    )
    source_identities = {
        (bundle.phase_source_digest, bundle.calendar_digest, bundle.policy_digest)
        for bundle in bundles
    }
    if len(source_identities) != 1:  # pragma: no cover - fail closed on mutation.
        raise RuntimeError("comparator bundles do not share one frozen source identity")
    return bundles


@dataclass(frozen=True)
class PhaseGuardReplayResult:
    """One repaired primitive with reconstructable reliability provenance."""

    target_subject: str
    target_elapsed_day: int
    sensor_day_id: int
    day_type: str
    primitive_name: str
    operation: str
    method: str
    repaired_value: float
    original_masked_value: float
    observed_fraction: float
    imputed_fraction: float
    reliability: PhaseGuardReliability
    personal_support: float
    population_fallback: bool
    template_mad: float
    population_scale: float
    critical_window_coverage: float
    template_source_statuses: tuple[tuple[int, str], ...]
    status: str
    reason: str
    source_record_digest: str
    mask_digest: str
    calendar_digest: str
    policy_digest: str
    phase_source_digest: str
    statistics_digest: str
    template_digest: str
    _build_token: InitVar[object]
    result_digest: str = field(init=False)

    def __post_init__(self, _build_token: object) -> None:
        if not _valid_result_token(_build_token):
            raise TypeError(
                "PhaseGuardReplayResult must be created by the trusted repair boundary"
            )
        if type(self.reliability) is not PhaseGuardReliability:
            raise TypeError("reliability must be an exact PhaseGuardReliability")
        if type(self.target_subject) is not str or not self.target_subject:
            raise ValueError("result target_subject must be a nonempty exact string")
        _exact_positive_int(self.target_elapsed_day, "result target_elapsed_day")
        if type(self.sensor_day_id) is not int or self.sensor_day_id < 0:
            raise ValueError("result sensor_day_id must be an exact nonnegative integer")
        if type(self.day_type) is not str or self.day_type not in _DAY_TYPES:
            raise ValueError("result day_type is not frozen")
        if (
            type(self.primitive_name) is not str
            or self.primitive_name not in _CORE_PRIMITIVE_OPERATIONS
            or self.operation != _CORE_PRIMITIVE_OPERATIONS[self.primitive_name]
        ):
            raise ValueError("result primitive/operation identity is invalid")
        if type(self.method) is not str or self.method not in (
            *_FROZEN_METHODS,
            "no_repair",
        ):
            raise ValueError("result method is not frozen")
        repaired = _exact_result_number(self.repaired_value, "repaired_value")
        original = _exact_result_number(
            self.original_masked_value, "original_masked_value"
        )
        observed = _finite_number(self.observed_fraction, "observed_fraction")
        imputed = _finite_number(self.imputed_fraction, "imputed_fraction")
        support = _finite_number(self.personal_support, "personal_support")
        mad = _finite_number(self.template_mad, "template_mad")
        scale = _finite_number(self.population_scale, "population_scale")
        critical = _finite_number(
            self.critical_window_coverage, "critical_window_coverage"
        )
        if not 0.0 <= observed <= 1.0 or not 0.0 <= imputed <= 1.0:
            raise ValueError("result coverage fractions must be in [0, 1]")
        if observed + imputed > 1.0 + 1e-12:
            raise ValueError("result observed and induced-mask fractions overlap")
        if support < 0.0 or mad < 0.0 or scale < 0.0:
            raise ValueError("result reliability inputs must be nonnegative")
        if not 0.0 <= critical <= 1.0:
            raise ValueError("result critical coverage must be in [0, 1]")
        for name in (
            "source_record_digest",
            "mask_digest",
            "calendar_digest",
            "policy_digest",
            "phase_source_digest",
            "statistics_digest",
            "template_digest",
        ):
            _require_sha256(getattr(self, name), name)
        if type(self.population_fallback) is not bool:
            raise ValueError("population_fallback must be an exact boolean")
        if type(self.template_source_statuses) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not int
            or type(item[1]) is not str
            for item in self.template_source_statuses
        ):
            raise ValueError("template_source_statuses must be exact typed pairs")
        allowed_status_reasons = {
            ("observed_exact", "no_masked_phase"),
            ("observed_incomplete", "natural_missingness_no_injected_mask"),
            ("repaired", "phase_template_repair"),
            ("abstain", "no_observed_support"),
            ("abstain", "missing_phase_template"),
            ("abstain", "no_effective_support"),
            ("abstain", "missing_event_location"),
            ("abstain", "insufficient_expected_events"),
            ("abstain", "no_positive_event_mass"),
            ("abstain", "nonfinite_repair"),
        }
        if (self.status, self.reason) not in allowed_status_reasons:
            raise ValueError("result status/reason pair is not frozen")
        timing = self.operation == "evening_p90"
        if self.status == "abstain":
            if not math.isnan(repaired):
                raise ValueError("abstention must carry a missing repaired value")
        elif not np.isfinite(repaired):
            raise ValueError("non-abstention must carry a finite repaired value")
        if self.reason == "no_observed_support" and observed != 0.0:
            raise ValueError("no_observed_support requires zero observed coverage")
        if self.reason == "no_positive_event_mass" and (
            not timing or observed <= 0.0
        ):
            raise ValueError(
                "no_positive_event_mass requires observed timing coverage"
            )
        if self.status == "repaired":
            expected_reliability = compute_phaseguard_reliability(
                observed_fraction=observed,
                n_eff=support,
                template_mad=mad,
                population_scale=scale,
                critical_window_coverage=critical,
                timing_primitive=timing,
            )
        elif self.status == "abstain":
            expected_reliability = _abstention_reliability(
                observed, critical_component=critical if timing else 1.0
            )
        else:
            critical_component = critical if timing else 1.0
            expected_reliability = PhaseGuardReliability(
                reliability=float((observed * critical_component) ** 0.25),
                coverage_component=float(observed),
                support_component=1.0,
                uncertainty_component=1.0,
                critical_component=float(critical_component),
            )
        if self.reliability != expected_reliability:
            raise ValueError("result reliability formula does not match bound inputs")
        if self.population_fallback != any(
            status == "population_fallback"
            for _, status in self.template_source_statuses
        ):
            raise ValueError("result population fallback contradicts source statuses")
        object.__setattr__(self, "repaired_value", repaired)
        object.__setattr__(self, "original_masked_value", original)
        object.__setattr__(self, "result_digest", _replay_result_digest(self))


def _exact_result_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be an exact numeric value")
    result = float(value)
    if np.isinf(result):
        raise ValueError(f"{name} cannot be infinite")
    return result


def _replay_result_digest(result: PhaseGuardReplayResult) -> str:
    return _sha256_payload(
        {
            "schema": "phaseguard-replay-result-v1",
            "target_subject": result.target_subject,
            "target_elapsed_day": result.target_elapsed_day,
            "sensor_day_id": result.sensor_day_id,
            "day_type": result.day_type,
            "primitive_name": result.primitive_name,
            "operation": result.operation,
            "method": result.method,
            "repaired_value": _float_token(result.repaired_value),
            "original_masked_value": _float_token(result.original_masked_value),
            "observed_fraction": _float_token(result.observed_fraction),
            "imputed_fraction": _float_token(result.imputed_fraction),
            "reliability": {
                "reliability": _float_token(result.reliability.reliability),
                "coverage_component": _float_token(
                    result.reliability.coverage_component
                ),
                "support_component": _float_token(
                    result.reliability.support_component
                ),
                "uncertainty_component": _float_token(
                    result.reliability.uncertainty_component
                ),
                "critical_component": _float_token(
                    result.reliability.critical_component
                ),
            },
            "personal_support": _float_token(result.personal_support),
            "population_fallback": result.population_fallback,
            "template_mad": _float_token(result.template_mad),
            "population_scale": _float_token(result.population_scale),
            "critical_window_coverage": _float_token(
                result.critical_window_coverage
            ),
            "template_source_statuses": [
                [phase, status] for phase, status in result.template_source_statuses
            ],
            "status": result.status,
            "reason": result.reason,
            "source_record_digest": result.source_record_digest,
            "mask_digest": result.mask_digest,
            "calendar_digest": result.calendar_digest,
            "policy_digest": result.policy_digest,
            "phase_source_digest": result.phase_source_digest,
            "statistics_digest": result.statistics_digest,
            "template_digest": result.template_digest,
        }
    )


def _exact_positive_int(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def online_calibration_days(
    target_elapsed_day: int, *, max_calibration_day: int = 14
) -> tuple[int, ...]:
    """Return the strict online prefix, never including target or future."""

    target = _exact_positive_int(target_elapsed_day, "target_elapsed_day")
    maximum = _exact_positive_int(max_calibration_day, "max_calibration_day")
    if type(max_calibration_day) is not int or maximum != 14:
        raise ValueError("max_calibration_day is frozen at exact integer 14")
    stop = min(maximum, target - 1)
    return tuple(range(1, stop + 1))


def _compact_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _forbidden_phase_columns(columns: Iterable[object]) -> tuple[object, ...]:
    forbidden: list[object] = []
    for column in columns:
        compact = _compact_name(column)
        if (
            any(
                token in compact
                for token in (
                    "outcome",
                    "target",
                    "label",
                    "oracle",
                    "prediction",
                    "probability",
                    "predscore",
                )
            )
            or re.search(r"(?:^|[^a-z])[qs][1-4](?:$|[^a-z])", str(column).casefold())
            or re.search(r"[qs][1-4](?:probability|score|label|target)", compact)
        ):
            forbidden.append(column)
    return tuple(forbidden)


def _validate_phase_rows(rows: pd.DataFrame, policy: PhaseGuardPolicy) -> pd.DataFrame:
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("phase_rows must be a pandas DataFrame")
    if rows.columns.duplicated().any():
        raise ValueError("phase_rows columns must be unique")
    missing = sorted(set(_PHASE_REQUIRED_COLUMNS).difference(rows.columns))
    if missing:
        raise ValueError(f"phase_rows missing required columns: {missing}")
    forbidden = _forbidden_phase_columns(rows.columns)
    if forbidden:
        raise ValueError(f"phase_rows contains outcome or prediction bait: {forbidden}")

    optional_timing = tuple(
        column
        for column in (
            "event_count",
            "event_mass",
            "representative_minute",
            "event_minutes",
            "event_weights",
        )
        if column in rows.columns
    )
    result = rows.loc[:, _PHASE_REQUIRED_COLUMNS + optional_timing].copy(deep=True)
    if result.duplicated(list(_PHASE_KEY_COLUMNS)).any():
        raise ValueError("duplicate phase source key")
    canonical_key = [
        "subject_id",
        "elapsed_day_index",
        "primitive",
        "phase_bin",
        "day_type",
    ]
    if result.duplicated(canonical_key).any():
        raise ValueError("duplicate canonical phase source key")
    if not result["subject_id"].map(lambda value: type(value) is str).all():
        raise ValueError("subject_id values must be exact strings")
    if not result["primitive"].map(lambda value: type(value) is str).all():
        raise ValueError("primitive values must be exact strings")
    if not result["day_type"].map(lambda value: type(value) is str).all():
        raise ValueError("day_type values must be exact strings")
    if not result["day_type"].isin(_DAY_TYPES).all():
        raise ValueError(f"day_type must be one of {_DAY_TYPES}")

    def exact_integer(value: object) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not result["elapsed_day_index"].map(exact_integer).all():
        raise ValueError("elapsed_day_index values must be exact integers")
    if not result["sensor_day_id"].map(exact_integer).all():
        raise ValueError("sensor_day_id values must be exact integers")
    if not result["phase_bin"].map(exact_integer).all():
        raise ValueError("phase_bin values must be exact integers")
    result["elapsed_day_index"] = result["elapsed_day_index"].astype(int)
    result["sensor_day_id"] = result["sensor_day_id"].astype(int)
    result["phase_bin"] = result["phase_bin"].astype(int)
    if result["elapsed_day_index"].le(0).any():
        raise ValueError("elapsed_day_index values must be positive")
    if result["sensor_day_id"].lt(0).any():
        raise ValueError("sensor_day_id values must be nonnegative")
    elapsed_calendar = result.groupby(
        ["subject_id", "elapsed_day_index"], sort=False, dropna=False
    ).agg(sensor_days=("sensor_day_id", "nunique"), day_types=("day_type", "nunique"))
    sensor_calendar = result.groupby(
        ["subject_id", "sensor_day_id"], sort=False, dropna=False
    ).agg(elapsed_days=("elapsed_day_index", "nunique"), day_types=("day_type", "nunique"))
    if (
        elapsed_calendar["sensor_days"].gt(1).any()
        or elapsed_calendar["day_types"].gt(1).any()
        or sensor_calendar["elapsed_days"].gt(1).any()
        or sensor_calendar["day_types"].gt(1).any()
    ):
        raise ValueError("phase_rows calendar identity is crosswired")
    phase_count = 1440 // int(policy.bin_minutes)
    if result["phase_bin"].lt(0).any() or result["phase_bin"].ge(phase_count).any():
        raise ValueError("phase_bin is outside the frozen phase grid")

    exact_value = result["value"].map(
        lambda value: value is None
        or value is pd.NA
        or (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
        )
    )
    if not bool(exact_value.all()):
        raise ValueError("value must contain exact numeric values or missing values")
    values = pd.to_numeric(result["value"], errors="coerce")
    invalid_value = result["value"].notna() & (~np.isfinite(values))
    if invalid_value.any():
        raise ValueError("value must be finite or missing")
    exact_support = result["support"].map(
        lambda value: (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
        )
    )
    if not bool(exact_support.all()):
        raise ValueError("support must contain exact numeric values")
    support = pd.to_numeric(result["support"], errors="coerce")
    if support.isna().any() or (~np.isfinite(support)).any():
        raise ValueError("support must be finite")
    if support.lt(0).any() or support.gt(1).any():
        raise ValueError("support must be in [0, 1]")
    result["value"] = values.astype(float)
    result["support"] = support.astype(float)
    bounded_primitives = tuple(
        primitive
        for primitive, operation in _CORE_PRIMITIVE_OPERATIONS.items()
        if operation in {"binary_load", "activity_load"}
    )
    bounded_rows = result["primitive"].isin(bounded_primitives) & result[
        "value"
    ].notna()
    if (
        bounded_rows
        & (result["value"].lt(0.0) | result["value"].gt(1.0))
    ).any():
        raise ValueError("binary/activity source value must be in [0, 1]")
    return result.sort_values(
        list(_PHASE_CANONICAL_ORDER), kind="mergesort"
    ).reset_index(drop=True)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    order = np.lexsort((np.arange(values.size), values))
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = _deterministic_fsum(ordered_weights) / 2.0
    cumulative = np.cumsum(ordered_weights)
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    index = min(index, ordered_values.size - 1)
    if (
        index + 1 < ordered_values.size
        and math.isclose(float(cumulative[index]), threshold, rel_tol=0.0, abs_tol=1e-15)
    ):
        return float((ordered_values[index] + ordered_values[index + 1]) / 2.0)
    return float(ordered_values[index])


def _weighted_mad(values: np.ndarray, weights: np.ndarray, center: float) -> float:
    if values.size == 0 or not np.isfinite(center):
        return float("nan")
    return _weighted_median(np.abs(values - center), weights)


def _finite_supported(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["value"].notna() & frame["support"].gt(0)].copy()


def fit_phase_estimate(
    phase_rows: pd.DataFrame,
    *,
    target_subject: str,
    target_elapsed_day: int,
    primitive: str,
    phase_bin: int,
    day_type: str,
    policy: PhaseGuardPolicy | None = None,
) -> PhaseEstimate:
    """Fit one strict-online personal phase estimate without outcomes."""

    frozen = policy if policy is not None else PhaseGuardPolicy()
    _validate_policy_integrity(frozen)
    if type(target_subject) is not str or not target_subject:
        raise ValueError("target_subject must be a nonempty exact string")
    target_day = _exact_positive_int(target_elapsed_day, "target_elapsed_day")
    if type(primitive) is not str or primitive not in _CORE_PRIMITIVE_OPERATIONS:
        raise ValueError("primitive must be one frozen G2 primitive")
    if isinstance(phase_bin, bool) or not isinstance(phase_bin, (int, np.integer)):
        raise ValueError("phase_bin must be an exact integer")
    phase = int(phase_bin)
    if phase < 0 or phase >= 1440 // int(frozen.bin_minutes):
        raise ValueError("phase_bin is outside the frozen phase grid")
    if type(day_type) is not str or day_type not in _DAY_TYPES:
        raise ValueError(f"day_type must be one of {_DAY_TYPES}")

    rows = _validate_phase_rows(phase_rows, frozen)
    eligible = rows.loc[
        rows["primitive"].eq(primitive)
        & rows["phase_bin"].eq(phase)
        & rows["day_type"].eq(day_type)
    ].copy()
    prefix = online_calibration_days(
        target_day, max_calibration_day=int(frozen.max_calibration_day)
    )
    personal = _finite_supported(
        eligible.loc[
            eligible["subject_id"].eq(target_subject)
            & eligible["elapsed_day_index"].isin(prefix)
        ]
    )
    personal_values = personal["value"].to_numpy(dtype=float)
    personal_weights = personal["support"].to_numpy(dtype=float)
    n_eff = _deterministic_fsum(personal_weights)
    personal_value = _weighted_median(personal_values, personal_weights)
    personal_mad = _weighted_mad(personal_values, personal_weights, personal_value)
    personal_supported = (
        np.isfinite(personal_value)
        and n_eff >= float(frozen.min_personal_support)
    )

    donor_rows = _finite_supported(
        eligible.loc[~eligible["subject_id"].eq(target_subject)]
    )
    donor_centers: list[tuple[str, float]] = []
    for subject, group in donor_rows.groupby("subject_id", sort=True):
        center = _weighted_median(
            group["value"].to_numpy(dtype=float),
            group["support"].to_numpy(dtype=float),
        )
        if np.isfinite(center):
            donor_centers.append((str(subject), float(center)))
    population_subjects = tuple(subject for subject, _ in donor_centers)
    population_values = np.asarray([value for _, value in donor_centers], dtype=float)
    if population_values.size:
        population_value = float(np.median(population_values))
        population_mad = float(np.median(np.abs(population_values - population_value)))
    else:
        population_value = float("nan")
        population_mad = float("nan")
    population_supported = np.isfinite(population_value)

    if personal_supported and population_supported:
        personal_weight = n_eff / (n_eff + float(frozen.tau))
        if personal_value == population_value:
            value = personal_value
        else:
            value = (
                personal_weight * personal_value
                + (1.0 - personal_weight) * population_value
            )
        template_mad = (
            personal_weight * personal_mad
            + (1.0 - personal_weight) * population_mad
        )
        status = "shrunk_personal"
    elif personal_supported:
        personal_weight = 1.0
        value = personal_value
        template_mad = personal_mad
        status = "personal_only"
    elif population_supported:
        personal_weight = 0.0
        value = population_value
        template_mad = population_mad
        status = "population_fallback"
    else:
        personal_weight = 0.0
        value = float("nan")
        template_mad = float("nan")
        status = "abstain"

    return PhaseEstimate(
        value=float(value),
        template_mad=float(template_mad),
        personal_value=float(personal_value),
        personal_mad=float(personal_mad),
        population_value=float(population_value),
        population_mad=float(population_mad),
        n_eff=n_eff,
        personal_observations=int(len(personal)),
        population_participants=int(len(population_subjects)),
        personal_weight=float(personal_weight),
        source_status=status,
        personal_source_days=tuple(
            sorted(personal["elapsed_day_index"].astype(int).unique().tolist())
        ),
        population_subjects=population_subjects,
    )


def _phase_metric_rows(
    phase_rows: pd.DataFrame,
    *,
    metric: str,
    primitive: str,
    phase_bin: int,
    day_type: str,
    positive_event_only: bool,
) -> pd.DataFrame:
    if metric not in phase_rows.columns:
        raise ValueError(f"timing phase_rows missing {metric}")
    rows = phase_rows.copy(deep=True)
    relevant = (
        rows["primitive"].eq(primitive)
        & rows["phase_bin"].eq(phase_bin)
        & rows["day_type"].eq(day_type)
    )
    exact_numeric = rows.loc[relevant, metric].map(
        lambda value: (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
        )
    )
    numeric = pd.to_numeric(rows.loc[relevant, metric], errors="coerce")
    if metric == "event_count":
        exact_integer = rows.loc[relevant, metric].map(
            lambda value: (
                not isinstance(value, (bool, np.bool_))
                and isinstance(value, (int, np.integer))
            )
        )
        valid = (
            exact_integer
            & numeric.notna()
            & np.isfinite(numeric)
            & numeric.ge(0)
        )
        if not bool(valid.all()):
            raise ValueError(
                "event_count must contain exact nonnegative integer values"
            )
    elif metric == "event_mass":
        valid = exact_numeric & numeric.notna() & np.isfinite(numeric) & numeric.ge(0)
        if not bool(valid.all()):
            raise ValueError("event_mass must contain exact finite nonnegative values")
    else:
        event_count = pd.to_numeric(
            rows.loc[relevant, "event_count"], errors="coerce"
        )
        needs_location = event_count.gt(0)
        valid_location = exact_numeric & numeric.notna() & np.isfinite(numeric)
        if not bool((~needs_location | valid_location).all()):
            raise ValueError(
                "representative_minute must be finite when event_count is positive"
            )
        lower = float(phase_bin * 30)
        inside = numeric.ge(lower) & numeric.lt(lower + 30.0)
        if not bool((~needs_location | inside).all()):
            raise ValueError("representative_minute must lie inside phase_bin")
    rows["value"] = pd.to_numeric(rows[metric], errors="coerce")
    if positive_event_only:
        event_count = pd.to_numeric(rows["event_count"], errors="coerce")
        rows.loc[event_count.le(0) | event_count.isna(), "value"] = np.nan
        rows.loc[event_count.le(0) | event_count.isna(), "support"] = 0.0
    return rows


@dataclass(frozen=True)
class _TemplateScalar:
    value: float
    mad: float
    population_scale: float
    n_eff: float
    personal_weight: float
    source_status: str


def _select_phase_scalar(
    phase_rows: pd.DataFrame,
    *,
    target_subject: str,
    target_elapsed_day: int,
    primitive: str,
    phase_bin: int,
    day_type: str,
    policy: PhaseGuardPolicy,
    method: str,
) -> _TemplateScalar | None:
    if method == "personal_global":
        rows = _validate_phase_rows(phase_rows, policy)
        prefix = online_calibration_days(
            target_elapsed_day,
            max_calibration_day=int(policy.max_calibration_day),
        )
        eligible = _finite_supported(
            rows.loc[
                rows["subject_id"].eq(target_subject)
                & rows["primitive"].eq(primitive)
                & rows["day_type"].eq(day_type)
                & rows["elapsed_day_index"].isin(prefix)
            ]
        )
        values = eligible["value"].to_numpy(dtype=float)
        weights = eligible["support"].to_numpy(dtype=float)
        n_eff = float(weights.sum())
        value = _weighted_median(values, weights)
        if (
            not np.isfinite(value)
            or n_eff < float(policy.min_personal_support)
        ):
            return None
        mad = _weighted_mad(values, weights, value)
        donor_centers: list[float] = []
        donors = _finite_supported(
            rows.loc[
                ~rows["subject_id"].eq(target_subject)
                & rows["primitive"].eq(primitive)
                & rows["day_type"].eq(day_type)
            ]
        )
        for _, group in donors.groupby("subject_id", sort=True):
            center = _weighted_median(
                group["value"].to_numpy(dtype=float),
                group["support"].to_numpy(dtype=float),
            )
            if np.isfinite(center):
                donor_centers.append(center)
        if donor_centers:
            population_center = float(np.median(donor_centers))
            population_scale = float(
                np.median(np.abs(np.asarray(donor_centers) - population_center))
            )
        else:
            population_scale = float(mad)
        return _TemplateScalar(
            value=float(value),
            mad=float(mad),
            population_scale=float(population_scale),
            n_eff=n_eff,
            personal_weight=1.0,
            source_status="personal_only",
        )

    estimate = fit_phase_estimate(
        phase_rows,
        target_subject=target_subject,
        target_elapsed_day=target_elapsed_day,
        primitive=primitive,
        phase_bin=phase_bin,
        day_type=day_type,
        policy=policy,
    )
    if method == "phaseguard":
        value = estimate.value
        mad = estimate.template_mad
        n_eff = estimate.n_eff
        status = estimate.source_status
        personal_weight = estimate.personal_weight
    elif method == "personal_phase":
        if (
            not np.isfinite(estimate.personal_value)
            or estimate.n_eff < float(policy.min_personal_support)
        ):
            return None
        value = estimate.personal_value
        mad = estimate.personal_mad
        n_eff = estimate.n_eff
        status = "personal_only"
        personal_weight = 1.0
    elif method == "population_phase":
        if not np.isfinite(estimate.population_value):
            return None
        value = estimate.population_value
        mad = estimate.population_mad
        n_eff = 0.0
        status = "population_fallback"
        personal_weight = 0.0
    else:  # pragma: no cover - caller validates the frozen method family.
        raise RuntimeError("unsupported frozen template method")
    if not np.isfinite(value) or not np.isfinite(mad):
        return None
    population_scale = (
        estimate.population_mad
        if np.isfinite(estimate.population_mad)
        else estimate.personal_mad
    )
    if not np.isfinite(population_scale):
        population_scale = 0.0
    return _TemplateScalar(
        value=float(value),
        mad=float(mad),
        population_scale=float(population_scale),
        n_eff=float(n_eff),
        personal_weight=float(personal_weight),
        source_status=status,
    )


def _validate_timing_event_rows(
    phase_rows: pd.DataFrame,
    *,
    primitive: str,
    day_type: str,
    policy: PhaseGuardPolicy,
) -> pd.DataFrame:
    rows = _validate_phase_rows(phase_rows, policy)
    required = ("event_count", "event_mass", "event_minutes", "event_weights")
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"timing phase_rows missing exact event distribution: {missing}")
    relevant = rows.loc[
        rows["primitive"].eq(primitive) & rows["day_type"].eq(day_type)
    ].copy()
    for row in relevant.itertuples(index=False):
        if (
            isinstance(row.event_count, (bool, np.bool_))
            or not isinstance(row.event_count, (int, np.integer))
            or int(row.event_count) < 0
        ):
            raise ValueError("event_count must be an exact nonnegative integer")
        count = int(row.event_count)
        mass = _exact_finite_real(row.event_mass, "event_mass")
        if type(row.event_minutes) is not tuple or type(row.event_weights) is not tuple:
            raise ValueError("timing event arrays must be exact tuples")
        if len(row.event_minutes) != len(row.event_weights):
            raise ValueError("timing event arrays must have equal length")
        if len(row.event_minutes) != count:
            raise ValueError("event_count must equal event array length")
        minutes = tuple(
            _exact_finite_real(value, "event minute") for value in row.event_minutes
        )
        weights = tuple(
            _exact_finite_real(
                value, "event weight", minimum=np.nextafter(0.0, 1.0)
            )
            for value in row.event_weights
        )
        lower = int(row.phase_bin) * 30.0
        if any(not lower <= value < lower + 30.0 for value in minutes):
            raise ValueError("timing event minutes must lie inside phase_bin")
        if minutes != tuple(sorted(minutes)):
            raise ValueError("timing event minutes must be deterministically sorted")
        if not math.isclose(
            _deterministic_fsum(weights), mass, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("event_mass must equal exact event weight mass")
        if count == 0 and minutes:
            raise ValueError("zero event_count cannot contain event endpoints")
    return rows


def _normalized_distribution(
    rows: pd.DataFrame,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    terms_by_minute: dict[float, list[float]] = {}
    for row in rows.itertuples(index=False):
        support = float(row.support)
        for minute, weight in zip(row.event_minutes, row.event_weights, strict=True):
            terms_by_minute.setdefault(float(minute), []).append(
                float(weight) * support
            )
    masses = {
        minute: _deterministic_fsum(terms)
        for minute, terms in sorted(terms_by_minute.items())
    }
    total = _deterministic_fsum(masses.values())
    if total <= 0.0:
        return (), ()
    minutes = tuple(sorted(masses))
    weights = tuple(float(masses[minute] / total) for minute in minutes)
    return minutes, weights


def _population_event_distribution(
    rows: pd.DataFrame,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    donor_distributions: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for _, group in rows.groupby("subject_id", sort=True):
        distribution = _normalized_distribution(group)
        if distribution[0]:
            donor_distributions.append(distribution)
    if not donor_distributions:
        return (), ()
    terms_by_minute: dict[float, list[float]] = {}
    donor_weight = 1.0 / len(donor_distributions)
    for minutes, weights in donor_distributions:
        for minute, weight in zip(minutes, weights, strict=True):
            terms_by_minute.setdefault(minute, []).append(donor_weight * weight)
    masses = {
        minute: _deterministic_fsum(terms)
        for minute, terms in sorted(terms_by_minute.items())
    }
    minutes = tuple(masses)
    weights = tuple(float(masses[minute]) for minute in minutes)
    return minutes, weights


def _blend_event_distributions(
    personal: tuple[tuple[float, ...], tuple[float, ...]],
    population: tuple[tuple[float, ...], tuple[float, ...]],
    *,
    personal_weight: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if personal_weight >= 1.0 or not population[0]:
        return personal
    if personal_weight <= 0.0 or not personal[0]:
        return population
    terms_by_minute: dict[float, list[float]] = {}
    for minute, weight in zip(*personal, strict=True):
        terms_by_minute.setdefault(minute, []).append(personal_weight * weight)
    for minute, weight in zip(*population, strict=True):
        terms_by_minute.setdefault(minute, []).append(
            (1.0 - personal_weight) * weight
        )
    masses = {
        minute: _deterministic_fsum(terms)
        for minute, terms in sorted(terms_by_minute.items())
    }
    minutes = tuple(masses)
    weights = tuple(float(masses[minute]) for minute in minutes)
    return minutes, weights


def _select_event_distribution(
    phase_rows: pd.DataFrame,
    *,
    target_subject: str,
    target_elapsed_day: int,
    primitive: str,
    phase_bin: int,
    day_type: str,
    policy: PhaseGuardPolicy,
    method: str,
    scalar: _TemplateScalar,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rows = _validate_timing_event_rows(
        phase_rows, primitive=primitive, day_type=day_type, policy=policy
    )
    relevant = rows.loc[
        rows["primitive"].eq(primitive)
        & rows["day_type"].eq(day_type)
        & rows["phase_bin"].eq(phase_bin)
    ]
    prefix = online_calibration_days(target_elapsed_day)
    personal_rows = _finite_supported(
        relevant.loc[
            relevant["subject_id"].eq(target_subject)
            & relevant["elapsed_day_index"].isin(prefix)
        ]
    )
    donor_rows = _finite_supported(
        relevant.loc[~relevant["subject_id"].eq(target_subject)]
    )
    if method == "personal_global":
        if scalar.value <= 0.0:
            return (), ()
        return (phase_bin * 30.0 + 15.0,), (1.0,)
    personal = _normalized_distribution(personal_rows)
    population = _population_event_distribution(donor_rows)
    if method == "personal_phase":
        return personal
    if method == "population_phase":
        return population
    return _blend_event_distributions(
        personal,
        population,
        personal_weight=scalar.personal_weight,
    )


def fit_phase_template_bin(
    phase_rows: pd.DataFrame,
    *,
    target_subject: str,
    target_elapsed_day: int,
    primitive: str,
    phase_bin: int,
    day_type: str,
    policy: PhaseGuardPolicy | None = None,
    method: str = "phaseguard",
) -> PhaseTemplateBin | None:
    """Convert strict-online phase rows into one repair-ready template bin."""

    frozen = policy if policy is not None else PhaseGuardPolicy()
    _validate_policy_integrity(frozen)
    if type(target_subject) is not str or not target_subject:
        raise ValueError("target_subject must be a nonempty exact string")
    _exact_positive_int(target_elapsed_day, "target_elapsed_day")
    if type(primitive) is not str or primitive not in _CORE_PRIMITIVE_OPERATIONS:
        raise ValueError("primitive must be one frozen G2 primitive")
    if isinstance(phase_bin, (bool, np.bool_)) or not isinstance(
        phase_bin, (int, np.integer)
    ):
        raise ValueError("phase_bin must be an exact integer")
    if not 0 <= int(phase_bin) < 48:
        raise ValueError("phase_bin is outside the frozen phase grid")
    if type(day_type) is not str or day_type not in _DAY_TYPES:
        raise ValueError(f"day_type must be one of {_DAY_TYPES}")
    allowed_methods = {
        "personal_global",
        "population_phase",
        "personal_phase",
        "phaseguard",
    }
    if type(method) is not str or method not in allowed_methods:
        raise ValueError("method must be one frozen PhaseGuard comparator")
    base = _select_phase_scalar(
        phase_rows,
        target_subject=target_subject,
        target_elapsed_day=target_elapsed_day,
        primitive=primitive,
        phase_bin=phase_bin,
        day_type=day_type,
        policy=frozen,
        method=method,
    )
    if base is None:
        return None

    representative_minute: float | None = None
    expected_event_count = 0.0
    expected_event_mass = 0.0
    event_minutes: tuple[float, ...] = ()
    event_weights: tuple[float, ...] = ()
    if _CORE_PRIMITIVE_OPERATIONS[primitive] == "evening_p90":
        if not isinstance(phase_rows, pd.DataFrame):
            raise TypeError("phase_rows must be a pandas DataFrame")
        event_rows = _phase_metric_rows(
            phase_rows,
            metric="event_count",
            primitive=primitive,
            phase_bin=int(phase_bin),
            day_type=day_type,
            positive_event_only=False,
        )
        event_estimate = _select_phase_scalar(
            event_rows,
            target_subject=target_subject,
            target_elapsed_day=target_elapsed_day,
            primitive=primitive,
            phase_bin=phase_bin,
            day_type=day_type,
            policy=frozen,
            method=method,
        )
        expected_event_count = event_estimate.value if event_estimate else 0.0
        event_mass_rows = _phase_metric_rows(
            phase_rows,
            metric="event_mass",
            primitive=primitive,
            phase_bin=int(phase_bin),
            day_type=day_type,
            positive_event_only=False,
        )
        event_mass_estimate = _select_phase_scalar(
            event_mass_rows,
            target_subject=target_subject,
            target_elapsed_day=target_elapsed_day,
            primitive=primitive,
            phase_bin=phase_bin,
            day_type=day_type,
            policy=frozen,
            method=method,
        )
        expected_event_mass = (
            event_mass_estimate.value if event_mass_estimate else 0.0
        )
        event_minutes, event_weights = _select_event_distribution(
            phase_rows,
            target_subject=target_subject,
            target_elapsed_day=target_elapsed_day,
            primitive=primitive,
            phase_bin=int(phase_bin),
            day_type=day_type,
            policy=frozen,
            method=method,
            scalar=base,
        )
        if (
            expected_event_count > 0.0 or expected_event_mass > 0.0
        ) and not event_minutes:
            return None
        if event_minutes:
            representative_minute = _weighted_median(
                np.asarray(event_minutes, dtype=float),
                np.asarray(event_weights, dtype=float),
            )

    return PhaseTemplateBin(
        phase_bin=int(phase_bin),
        value=float(base.value),
        template_mad=float(base.mad),
        population_scale=float(base.population_scale),
        n_eff=float(base.n_eff),
        source_status=base.source_status,
        representative_minute=representative_minute,
        expected_event_count=expected_event_count,
        expected_event_mass=expected_event_mass,
        event_minutes=event_minutes,
        event_weights=event_weights,
    )


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def compute_phaseguard_reliability(
    *,
    observed_fraction: float,
    n_eff: float,
    template_mad: float,
    population_scale: float,
    critical_window_coverage: float,
    timing_primitive: bool,
) -> PhaseGuardReliability:
    """Compute the preregistered four-component geometric reliability."""

    observed = _finite_number(observed_fraction, "observed_fraction")
    support = _finite_number(n_eff, "n_eff")
    mad = _finite_number(template_mad, "template_mad")
    scale = _finite_number(population_scale, "population_scale")
    critical_coverage = _finite_number(
        critical_window_coverage, "critical_window_coverage"
    )
    if not 0.0 <= observed <= 1.0:
        raise ValueError("observed_fraction must be in [0, 1]")
    if support < 0:
        raise ValueError("n_eff must be nonnegative")
    if mad < 0:
        raise ValueError("template_mad must be nonnegative")
    if scale < 0:
        raise ValueError("population_scale must be nonnegative")
    if not 0.0 <= critical_coverage <= 1.0:
        raise ValueError("critical_window_coverage must be in [0, 1]")
    if type(timing_primitive) is not bool:
        raise ValueError("timing_primitive must be an exact boolean")

    coverage_component = observed
    support_component = min(1.0, support / 7.0)
    uncertainty_component = math.exp(-mad / max(scale, 1e-12))
    critical_component = critical_coverage if timing_primitive else 1.0
    product = (
        coverage_component
        * support_component
        * uncertainty_component
        * critical_component
    )
    reliability = product ** 0.25
    return PhaseGuardReliability(
        reliability=float(reliability),
        coverage_component=float(coverage_component),
        support_component=float(support_component),
        uncertainty_component=float(uncertainty_component),
        critical_component=float(critical_component),
    )


def _abstention_reliability(
    observed_fraction: float, *, critical_component: float
) -> PhaseGuardReliability:
    return PhaseGuardReliability(
        reliability=0.0,
        coverage_component=float(observed_fraction),
        support_component=0.0,
        uncertainty_component=0.0,
        critical_component=float(critical_component),
    )


def _phaseguard_abstention(
    statistics: PhaseSufficientStatistics,
    bundle: PhaseTemplateBundle,
    *,
    observed_fraction: float,
    personal_support: float,
    template_mad: float,
    population_scale: float,
    source_statuses: tuple[tuple[int, str], ...],
    reason: str,
) -> PhaseGuardReplayResult:
    critical_component = statistics.critical_window_coverage
    return _construct_replay_result(
        PhaseGuardReplayResult,
        target_subject=statistics.target_subject,
        target_elapsed_day=statistics.target_elapsed_day,
        sensor_day_id=statistics.sensor_day_id,
        day_type=statistics.day_type,
        primitive_name=statistics.primitive_name,
        operation=statistics.operation,
        method=bundle.method,
        repaired_value=float("nan"),
        original_masked_value=float(statistics.masked_value),
        observed_fraction=float(observed_fraction),
        imputed_fraction=float(statistics.induced_mask_fraction),
        reliability=_abstention_reliability(
            observed_fraction, critical_component=critical_component
        ),
        personal_support=float(personal_support),
        population_fallback=any(
            status == "population_fallback" for _, status in source_statuses
        ),
        template_mad=float(template_mad),
        population_scale=float(population_scale),
        critical_window_coverage=float(critical_component),
        template_source_statuses=source_statuses,
        status="abstain",
        reason=reason,
        source_record_digest=statistics.source_record_digest,
        mask_digest=statistics.mask_digest,
        calendar_digest=statistics.calendar_digest,
        policy_digest=statistics.policy_digest,
        phase_source_digest=bundle.phase_source_digest,
        statistics_digest=statistics.content_digest,
        template_digest=bundle.content_digest,
    )


def _validate_statistics_integrity(statistics: PhaseSufficientStatistics) -> None:
    if type(statistics) is not PhaseSufficientStatistics:
        raise TypeError("statistics must be an exact PhaseSufficientStatistics")
    if _statistics_content_digest(statistics) != statistics.content_digest:
        raise ValueError("statistics content digest does not match bound content")


def _validate_template_integrity(bundle: PhaseTemplateBundle) -> None:
    if type(bundle) is not PhaseTemplateBundle:
        raise TypeError("templates must be an exact PhaseTemplateBundle")
    if _template_bundle_content_digest(bundle) != bundle.content_digest:
        raise ValueError("template bundle content digest does not match bound content")


def _validate_replay_binding(
    statistics: PhaseSufficientStatistics, bundle: PhaseTemplateBundle
) -> None:
    _validate_template_integrity(bundle)
    target_identity = (
        statistics.target_subject,
        statistics.target_elapsed_day,
        statistics.sensor_day_id,
        statistics.day_type,
    )
    bundle_identity = (
        bundle.target_subject,
        bundle.target_elapsed_day,
        bundle.target_sensor_day_id,
        bundle.day_type,
    )
    if target_identity != bundle_identity:
        raise ValueError("template target identity does not match statistics")
    if (
        statistics.primitive_name != bundle.primitive_name
        or statistics.operation != bundle.operation
    ):
        raise ValueError("template primitive identity does not match statistics")
    if statistics.policy_digest != bundle.policy_digest:
        raise ValueError("template policy identity does not match statistics")
    if statistics.calendar_digest != bundle.calendar_digest:
        raise ValueError("template calendar identity does not match statistics")


def _repair_phaseguard_statistics_impl(
    statistics: PhaseSufficientStatistics,
    templates: PhaseTemplateBundle | None,
) -> PhaseGuardReplayResult:
    """Repair one masked primitive from cached phase statistics.

    The mask geometry, rather than the hidden unmasked target value, controls
    every imputed contribution.  A zero-mask call returns the reviewed value
    bit-for-bit before consulting any template.
    """

    _validate_statistics_integrity(statistics)

    total_units = float(sum(item.expected_units for item in statistics.bins))
    imputed_fraction = statistics.induced_mask_fraction
    imputed_units = float(imputed_fraction * total_units)
    observed_fraction = statistics.observed_fraction
    timing = statistics.operation == "evening_p90"
    if imputed_units == 0.0:
        if templates is not None:
            _validate_replay_binding(statistics, templates)
            method = templates.method
            phase_source_digest = templates.phase_source_digest
            template_digest = templates.content_digest
        else:
            method = "no_repair"
            phase_source_digest = _sha256_payload(
                {"schema": "phaseguard-no-template-v1"}
            )
            template_digest = phase_source_digest
        critical_component = (
            statistics.critical_window_coverage if timing else 1.0
        )
        if observed_fraction == 1.0 and np.isfinite(statistics.masked_value):
            status = "observed_exact"
            reason = "no_masked_phase"
        elif observed_fraction == 0.0:
            status = "abstain"
            reason = "no_observed_support"
        elif timing and not np.isfinite(statistics.masked_value):
            status = "abstain"
            reason = "no_positive_event_mass"
        elif not np.isfinite(statistics.masked_value):
            status = "abstain"
            reason = "no_observed_support"
        else:
            status = "observed_incomplete"
            reason = "natural_missingness_no_injected_mask"
        if status == "abstain":
            natural_reliability = _abstention_reliability(
                observed_fraction, critical_component=critical_component
            )
        else:
            natural_reliability = PhaseGuardReliability(
                reliability=float(
                    (observed_fraction * critical_component) ** 0.25
                ),
                coverage_component=float(observed_fraction),
                support_component=1.0,
                uncertainty_component=1.0,
                critical_component=float(critical_component),
            )
        return _construct_replay_result(
            PhaseGuardReplayResult,
            target_subject=statistics.target_subject,
            target_elapsed_day=statistics.target_elapsed_day,
            sensor_day_id=statistics.sensor_day_id,
            day_type=statistics.day_type,
            primitive_name=statistics.primitive_name,
            operation=statistics.operation,
            method=method,
            repaired_value=statistics.masked_value,
            original_masked_value=statistics.masked_value,
            observed_fraction=float(observed_fraction),
            imputed_fraction=0.0,
            reliability=natural_reliability,
            personal_support=0.0,
            population_fallback=False,
            template_mad=0.0,
            population_scale=0.0,
            critical_window_coverage=float(
                statistics.critical_window_coverage
            ),
            template_source_statuses=(),
            status=status,
            reason=reason,
            source_record_digest=statistics.source_record_digest,
            mask_digest=statistics.mask_digest,
            calendar_digest=statistics.calendar_digest,
            policy_digest=statistics.policy_digest,
            phase_source_digest=phase_source_digest,
            statistics_digest=statistics.content_digest,
            template_digest=template_digest,
        )

    if type(templates) is not PhaseTemplateBundle:
        raise TypeError("templates must be an exact PhaseTemplateBundle")
    _validate_replay_binding(statistics, templates)
    critical = statistics.critical_window_coverage

    template_by_bin: dict[int, PhaseTemplateBin] = {}
    statistic_bins = {item.phase_bin for item in statistics.bins}
    for template in templates.bins:
        if template is None:
            continue
        if template.phase_bin in template_by_bin:
            raise ValueError("template phase bins must be unique")
        if template.phase_bin not in statistic_bins:
            raise ValueError("template phase bin is not present in statistics")
        template_by_bin[template.phase_bin] = template
    masked_bins = [item for item in statistics.bins if item.masked_fraction > 0.0]
    if any(item.phase_bin not in template_by_bin for item in masked_bins):
        return _phaseguard_abstention(
            statistics,
            templates,
            observed_fraction=observed_fraction,
            personal_support=0.0,
            template_mad=0.0,
            population_scale=0.0,
            source_statuses=(),
            reason="missing_phase_template",
        )

    weighted_templates = [
        (
            template_by_bin[item.phase_bin],
            item.expected_units * item.masked_fraction,
        )
        for item in masked_bins
    ]
    template_weight = float(sum(weight for _, weight in weighted_templates))
    n_eff = float(
        sum(template.n_eff * weight for template, weight in weighted_templates)
        / template_weight
    )
    template_mad = float(
        sum(
            template.template_mad * weight
            for template, weight in weighted_templates
        )
        / template_weight
    )
    population_scale = float(
        sum(
            template.population_scale * weight
            for template, weight in weighted_templates
        )
        / template_weight
    )
    source_statuses = tuple(
        (item.phase_bin, template_by_bin[item.phase_bin].source_status)
        for item in masked_bins
    )
    reliability = compute_phaseguard_reliability(
        observed_fraction=observed_fraction,
        n_eff=n_eff,
        template_mad=template_mad,
        population_scale=population_scale,
        critical_window_coverage=critical,
        timing_primitive=timing,
    )

    if statistics.operation == "additive_load":
        observed_value = (
            statistics.masked_value
            if np.isfinite(statistics.masked_value)
            else float(sum(item.observed_numerator for item in statistics.bins))
        )
        repaired = observed_value + sum(
            template_by_bin[item.phase_bin].value * item.masked_fraction
            for item in masked_bins
        )
    elif statistics.operation in _RATE_OPERATIONS:
        numerator = float(sum(item.observed_numerator for item in statistics.bins))
        denominator = float(sum(item.observed_units for item in statistics.bins))
        for item in masked_bins:
            filled_units = item.expected_units * item.masked_fraction
            numerator += template_by_bin[item.phase_bin].value * filled_units
            denominator += filled_units
        if denominator <= 0.0:
            return _phaseguard_abstention(
                statistics,
                templates,
                observed_fraction=observed_fraction,
                personal_support=n_eff,
                template_mad=template_mad,
                population_scale=population_scale,
                source_statuses=source_statuses,
                reason="no_effective_support",
            )
        repaired = numerator / denominator
    elif timing:
        minutes: list[float] = []
        weights: list[float] = []
        effective_event_count = 0.0
        for item in statistics.bins:
            minutes.extend(item.event_minutes)
            weights.extend(item.event_weights)
            effective_event_count += float(len(item.event_minutes))
        for item in masked_bins:
            template = template_by_bin[item.phase_bin]
            effective_event_count += (
                template.expected_event_count * item.masked_fraction
            )
            imputed_mass = template.expected_event_mass * item.masked_fraction
            if imputed_mass <= 0.0:
                continue
            if not template.event_minutes:
                return _phaseguard_abstention(
                    statistics,
                    templates,
                    observed_fraction=observed_fraction,
                    personal_support=n_eff,
                    template_mad=template_mad,
                    population_scale=population_scale,
                    source_statuses=source_statuses,
                    reason="missing_event_location",
                )
            for minute, relative_weight in zip(
                template.event_minutes, template.event_weights, strict=True
            ):
                minutes.append(minute)
                weights.append(imputed_mass * relative_weight)
        if observed_fraction == 0.0 and effective_event_count < 3.0:
            return _phaseguard_abstention(
                statistics,
                templates,
                observed_fraction=observed_fraction,
                personal_support=n_eff,
                template_mad=template_mad,
                population_scale=population_scale,
                source_statuses=source_statuses,
                reason="insufficient_expected_events",
            )
        if not weights:
            return _phaseguard_abstention(
                statistics,
                templates,
                observed_fraction=observed_fraction,
                personal_support=n_eff,
                template_mad=template_mad,
                population_scale=population_scale,
                source_statuses=source_statuses,
                reason="no_positive_event_mass",
            )
        value_array = np.asarray(minutes, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        order = np.argsort(value_array, kind="stable")
        sorted_values = value_array[order]
        cumulative = np.cumsum(weight_array[order])
        index = int(np.searchsorted(cumulative, 0.9 * cumulative[-1], side="left"))
        repaired = float(sorted_values[min(index, len(sorted_values) - 1)])
    else:  # pragma: no cover - the typed constructor makes this unreachable.
        raise RuntimeError("unsupported frozen operation")

    if (
        statistics.operation in {"binary_load", "activity_load"}
        and np.isfinite(repaired)
        and not -1e-12 <= repaired <= 1.0 + 1e-12
    ):
        raise ValueError("bounded repair lies outside [0, 1]")
    if not np.isfinite(repaired):
        return _phaseguard_abstention(
            statistics,
            templates,
            observed_fraction=observed_fraction,
            personal_support=n_eff,
            template_mad=template_mad,
            population_scale=population_scale,
            source_statuses=source_statuses,
            reason="nonfinite_repair",
        )
    return _construct_replay_result(
        PhaseGuardReplayResult,
        target_subject=statistics.target_subject,
        target_elapsed_day=statistics.target_elapsed_day,
        sensor_day_id=statistics.sensor_day_id,
        day_type=statistics.day_type,
        primitive_name=statistics.primitive_name,
        operation=statistics.operation,
        method=templates.method,
        repaired_value=float(repaired),
        original_masked_value=float(statistics.masked_value),
        observed_fraction=float(observed_fraction),
        imputed_fraction=float(imputed_fraction),
        reliability=reliability,
        personal_support=n_eff,
        population_fallback=any(
            status == "population_fallback" for _, status in source_statuses
        ),
        template_mad=template_mad,
        population_scale=population_scale,
        critical_window_coverage=float(critical),
        template_source_statuses=source_statuses,
        status="repaired",
        reason="phase_template_repair",
        source_record_digest=statistics.source_record_digest,
        mask_digest=statistics.mask_digest,
        calendar_digest=statistics.calendar_digest,
        policy_digest=statistics.policy_digest,
        phase_source_digest=templates.phase_source_digest,
        statistics_digest=statistics.content_digest,
        template_digest=templates.content_digest,
    )


def repair_phaseguard_statistics(
    statistics: PhaseSufficientStatistics,
    templates: PhaseTemplateBundle | None,
) -> PhaseGuardReplayResult:
    """Recompute one trusted replay result from bound statistics and templates."""

    return _repair_phaseguard_statistics_impl(statistics, templates)


def validate_phaseguard_replay_result(
    result: PhaseGuardReplayResult,
    statistics: PhaseSufficientStatistics,
    templates: PhaseTemplateBundle | None,
) -> None:
    """Rebuild and compare a result before an artifact manifest accepts it."""

    if type(result) is not PhaseGuardReplayResult:
        raise TypeError("result must be an exact PhaseGuardReplayResult")
    if _replay_result_digest(result) != result.result_digest:
        raise ValueError("replay result digest does not match bound content")
    expected = _repair_phaseguard_statistics_impl(statistics, templates)
    if result.result_digest != expected.result_digest:
        raise ValueError("replay result does not match authoritative recomputation")


__all__ = [
    "PhaseBinSufficientStatistic",
    "PhaseEstimate",
    "PhaseGuardPolicy",
    "PhaseGuardReliability",
    "PhaseGuardReplayResult",
    "PhaseSufficientStatistics",
    "PhaseTemplateBin",
    "PhaseTemplateBundle",
    "build_phase_sufficient_statistics",
    "compute_phaseguard_reliability",
    "fit_phase_estimate",
    "fit_phase_template_bin",
    "fit_phase_template_bundle",
    "fit_phase_template_bundles",
    "online_calibration_days",
    "repair_phaseguard_statistics",
    "validate_phaseguard_replay_result",
]
