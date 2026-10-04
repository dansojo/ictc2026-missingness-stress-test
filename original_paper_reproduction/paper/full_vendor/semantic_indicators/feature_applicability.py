"""Observation-process applicability diagnostics for passive-sensing features."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .primitives import (
    PreparedPhaseView,
    PreparedPrimitiveReplay,
    export_prepared_phase_view,
    recompute_prepared_primitive,
    recompute_prepared_primitive_batch,
)


FROZEN_FEATURES = (
    "step_load_24h",
    "screen_load_24h",
    "usage_load_24h",
    "screen_disengagement_p90",
)
_FAMILIES = ("event_count", "time_in_state", "intensity", "timing")
_DEFICIT_COLUMNS = (
    "coverage_deficit",
    "longest_gap_fraction",
    "cadence_deviation",
    "target_window_deficit",
    "support_incompatibility",
    "zero_missing_ambiguity",
)
_KEY_COLUMNS = ("subject_id", "sensor_day_id", "primitive_name", "scenario")
_FORBIDDEN_TOKENS = (
    "sleep",
    "questionnaire",
    "outcome",
    "identity",
    "g3",
    "g4",
)
_INTEGRITY_COLUMNS = {
    "determinism_pass",
    "source_exclusion_pass",
    "mask_identity_pass",
}


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FeatureApplicabilityContract:
    primitive_name: str
    operator_family: str
    output_unit: str
    measurement_resolution: float
    target_window_required: bool
    content_digest: str

    def __post_init__(self) -> None:
        for name in ("primitive_name", "operator_family", "output_unit"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be an exact nonempty string")
        if self.operator_family not in _FAMILIES:
            raise ValueError("operator_family is not frozen")
        if type(self.measurement_resolution) is not float:
            raise TypeError("measurement_resolution must be an exact float")
        if not math.isfinite(self.measurement_resolution) or self.measurement_resolution <= 0:
            raise ValueError("measurement_resolution must be finite and positive")
        if type(self.target_window_required) is not bool:
            raise TypeError("target_window_required must be an exact bool")
        if type(self.content_digest) is not str or len(self.content_digest) != 64:
            raise ValueError("content_digest must be a SHA-256 hexadecimal string")
        try:
            bytes.fromhex(self.content_digest)
        except ValueError as error:
            raise ValueError("content_digest must be hexadecimal") from error
        expected = _contract_digest(self)
        if self.content_digest != expected:
            raise ValueError("contract content digest does not match")


def _contract_digest(contract: FeatureApplicabilityContract) -> str:
    return _sha256(
        {
            "schema": "feature-applicability-contract-v1",
            "primitive_name": contract.primitive_name,
            "operator_family": contract.operator_family,
            "output_unit": contract.output_unit,
            "measurement_resolution": float(contract.measurement_resolution).hex(),
            "target_window_required": contract.target_window_required,
        }
    )


def _make_contract(
    primitive_name: str,
    operator_family: str,
    output_unit: str,
    measurement_resolution: float,
    target_window_required: bool,
) -> FeatureApplicabilityContract:
    provisional = object.__new__(FeatureApplicabilityContract)
    values = {
        "primitive_name": primitive_name,
        "operator_family": operator_family,
        "output_unit": output_unit,
        "measurement_resolution": float(measurement_resolution),
        "target_window_required": target_window_required,
        "content_digest": "0" * 64,
    }
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return replace(provisional, content_digest=_contract_digest(provisional))


_CONTRACTS = (
    _make_contract("step_load_24h", "event_count", "steps", 1.0, False),
    _make_contract("screen_load_24h", "time_in_state", "fraction", 1.0 / 1440.0, False),
    _make_contract("usage_load_24h", "intensity", "reported_app_usage_minutes", 1.0 / 60000.0, False),
    _make_contract("screen_disengagement_p90", "timing", "minute_of_day", 1.0, True),
)


def frozen_applicability_contracts() -> dict[str, FeatureApplicabilityContract]:
    return {item.primitive_name: item for item in _CONTRACTS}


def applicability_decision(probability: float) -> str:
    if isinstance(probability, (bool, np.bool_)) or not isinstance(
        probability, (float, np.floating)
    ):
        raise TypeError("probability must be a real floating value")
    value = float(probability)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("probability must be finite and lie in [0, 1]")
    if value <= 0.10:
        return "PORTABLE"
    if value <= 0.30:
        return "CONDITIONAL"
    return "NON_PORTABLE"


def _validate_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    if type(evidence) is not pd.DataFrame or evidence.empty:
        raise TypeError("evidence must be a nonempty exact DataFrame")
    if evidence.columns.duplicated().any():
        raise ValueError("evidence columns must be unique")
    bait = [
        str(column)
        for column in evidence.columns
        if str(column) not in _INTEGRITY_COLUMNS
        and any(token in str(column).lower() for token in _FORBIDDEN_TOKENS)
    ]
    if bait:
        raise ValueError(f"evidence contains forbidden columns: {bait}")
    required = set(_KEY_COLUMNS) | {
        "operator_family",
        "reference_value",
        "replayed_value",
        "estimand_available",
        "signed_reversal",
        "determinism_pass",
        "source_exclusion_pass",
        "mask_identity_pass",
        *_DEFICIT_COLUMNS,
    }
    missing = sorted(required.difference(evidence.columns))
    if missing:
        raise ValueError(f"evidence is missing required columns: {missing}")
    working = evidence.copy(deep=True)
    if working.duplicated(list(_KEY_COLUMNS)).any():
        raise ValueError("evidence keys must be unique")
    contracts = frozen_applicability_contracts()
    if not working["primitive_name"].isin(contracts).all():
        raise ValueError("evidence contains a nonfrozen primitive")
    expected_family = working["primitive_name"].map(
        {name: item.operator_family for name, item in contracts.items()}
    )
    if not working["operator_family"].eq(expected_family).all():
        raise ValueError("operator family is crosswired")
    participants = sorted(working["subject_id"].unique().tolist())
    if len(participants) != 10 or not all(type(item) is str for item in participants):
        raise ValueError("evidence must contain exactly ten string participant IDs")
    for column in ("estimand_available", "signed_reversal", "determinism_pass", "source_exclusion_pass", "mask_identity_pass"):
        if not working[column].map(lambda value: type(value) is bool).all():
            raise ValueError(f"{column} must contain exact booleans")
    numeric = ("reference_value", "replayed_value", *_DEFICIT_COLUMNS)
    for column in numeric:
        values = pd.to_numeric(working[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite")
        working[column] = values
    if not working.loc[:, _DEFICIT_COLUMNS].ge(0.0).all().all():
        raise ValueError("deficit predictors must be nonnegative")
    return working.sort_values(list(_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def _family_scales(training: pd.DataFrame) -> dict[str, float]:
    contracts = frozen_applicability_contracts()
    scales: dict[str, float] = {}
    for primitive, contract in contracts.items():
        values = training.loc[training["primitive_name"].eq(primitive), "reference_value"]
        span = float(values.quantile(0.90) - values.quantile(0.10))
        scales[primitive] = max(span, contract.measurement_resolution)
    return scales


def _failure_labels(frame: pd.DataFrame, scales: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    scale = frame["primitive_name"].map(scales).to_numpy(dtype=float)
    error = np.abs(frame["replayed_value"].to_numpy(dtype=float) - frame["reference_value"].to_numpy(dtype=float))
    label = (
        ~frame["estimand_available"].to_numpy(dtype=bool)
        | frame["signed_reversal"].to_numpy(dtype=bool)
        | (error > 0.20 * scale)
    )
    return label.astype(int), error / scale


def _design(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    families = np.column_stack(
        [frame["operator_family"].eq(name).to_numpy(dtype=float) for name in _FAMILIES]
    )
    continuous = frame.loc[:, columns].to_numpy(dtype=float)
    if means is None:
        means = continuous.mean(axis=0)
    if scales is None:
        scales = continuous.std(axis=0)
        scales = np.where(scales > 0.0, scales, 1.0)
    standardized = (continuous - means) / scales
    return np.column_stack([families, standardized]), means, scales


def _fit_probability(
    training: pd.DataFrame,
    held: pd.DataFrame,
    labels: np.ndarray,
    columns: tuple[str, ...],
) -> np.ndarray:
    train_x, means, scales = _design(training, columns)
    held_x, _, _ = _design(held, columns, means, scales)
    if len(np.unique(labels)) == 1:
        return np.full(len(held), (float(labels.sum()) + 0.5) / (len(labels) + 1.0))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = np.clip(train_x @ beta, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        loss = -np.sum(labels * np.log(probability + 1e-15) + (1 - labels) * np.log(1 - probability + 1e-15))
        penalty = 0.5 * float(np.dot(beta[4:], beta[4:]))
        gradient = train_x.T @ (probability - labels)
        gradient[4:] += beta[4:]
        return float(loss + penalty), gradient

    initial = np.zeros(train_x.shape[1], dtype=float)
    prevalence = (float(labels.sum()) + 0.5) / (len(labels) + 1.0)
    initial[:4] = math.log(prevalence / (1.0 - prevalence))
    bounds = [(None, None)] * 4 + [(0.0, None)] * len(columns)
    fitted = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=bounds)
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"constrained risk fit failed: {fitted.message}")
    linear = np.clip(held_x @ fitted.x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-linear))


def crossfit_applicability_scores(evidence: pd.DataFrame) -> pd.DataFrame:
    working = _validate_evidence(evidence)
    participants = tuple(sorted(working["subject_id"].unique().tolist()))
    scored: list[pd.DataFrame] = []
    for held_subject in participants:
        training = working.loc[working["subject_id"].ne(held_subject)].copy()
        held = working.loc[working["subject_id"].eq(held_subject)].copy()
        scales = _family_scales(training)
        training_labels, _ = _failure_labels(training, scales)
        held_labels, normalized_error = _failure_labels(held, scales)
        held["failure"] = held_labels.astype(bool)
        held["normalized_absolute_error"] = normalized_error
        held["coverage_only_risk"] = _fit_probability(
            training, held, training_labels, ("coverage_deficit",)
        )
        held["coverage_gap_risk"] = _fit_probability(
            training,
            held,
            training_labels,
            ("coverage_deficit", "longest_gap_fraction"),
        )
        held["predicted_failure_risk"] = _fit_probability(
            training, held, training_labels, _DEFICIT_COLUMNS
        )
        held["applicability_decision"] = held["predicted_failure_risk"].map(
            applicability_decision
        )
        training_subjects = tuple(item for item in participants if item != held_subject)
        held["training_subjects"] = pd.Series(
            [training_subjects] * len(held), index=held.index, dtype=object
        )
        scored.append(held)
    return pd.concat(scored, ignore_index=True).sort_values(
        list(_KEY_COLUMNS), kind="stable"
    ).reset_index(drop=True)


def gate_decision(rule_values: Mapping[str, bool]) -> dict[str, object]:
    if type(rule_values) is not dict or len(rule_values) != 7:
        raise ValueError("rule_values must contain exactly seven rules")
    items = tuple(rule_values.items())
    if not all(type(name) is str and name and type(value) is bool for name, value in items):
        raise TypeError("gate rules require exact string names and boolean values")
    failed = tuple(name for name, value in items if not value)
    decision = "GO" if not failed else "STOP"
    digest = _sha256(
        {
            "schema": "feature-applicability-gate-v1",
            "rules": [[name, value] for name, value in items],
            "decision": decision,
            "failed_rules": list(failed),
        }
    )
    return {"decision": decision, "failed_rules": failed, "result_digest": digest}


def _strict_json_value(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("strict JSON cannot contain nonfinite floats")
        return value
    if isinstance(value, np.generic):
        return _strict_json_value(value.item())
    if type(value) is pd.Timestamp:
        if pd.isna(value) or value.tz is not None:
            raise ValueError("strict JSON timestamps must be timezone-naive and finite")
        return value.isoformat()
    if type(value) in (tuple, list):
        return [_strict_json_value(item) for item in value]
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise TypeError("strict JSON object keys must be exact strings")
        return {key: _strict_json_value(item) for key, item in value.items()}
    raise TypeError(f"strict JSON does not support {type(value).__name__}")


def strict_json_bytes(value: object) -> bytes:
    projected = _strict_json_value(value)
    return (
        json.dumps(
            projected,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _balanced_accuracy(frame: pd.DataFrame, risk_column: str) -> float:
    values: list[float] = []
    for _, participant in frame.groupby("subject_id", sort=True):
        truth = participant["failure"].to_numpy(dtype=bool)
        predicted = participant[risk_column].to_numpy(dtype=float) >= 0.5
        recalls: list[float] = []
        for state in (False, True):
            mask = truth == state
            if mask.any():
                recalls.append(float((predicted[mask] == state).mean()))
        values.append(float(np.mean(recalls)))
    return float(np.mean(values))


def _retained_error(frame: pd.DataFrame, risk_column: str, fraction: float = 0.70) -> float:
    retained: list[pd.DataFrame] = []
    for _, group in frame.groupby("subject_id", sort=True):
        count = max(1, int(math.floor(len(group) * fraction)))
        retained.append(group.sort_values([risk_column, *_KEY_COLUMNS[1:]], kind="stable").head(count))
    joined = pd.concat(retained, ignore_index=True)
    return float(joined["normalized_absolute_error"].median())


def evaluate_etri_gate(scored: pd.DataFrame) -> dict[str, object]:
    required = {
        *_KEY_COLUMNS,
        "operator_family",
        "scenario",
        "failure",
        "normalized_absolute_error",
        "coverage_only_risk",
        "coverage_gap_risk",
        "predicted_failure_risk",
        "applicability_decision",
        "coverage_deficit",
        "determinism_pass",
        "source_exclusion_pass",
        "mask_identity_pass",
    }
    if type(scored) is not pd.DataFrame or scored.empty or not required.issubset(scored.columns):
        raise ValueError("scored evidence is incomplete")
    frame = scored.sort_values(list(_KEY_COLUMNS), kind="stable").reset_index(drop=True)
    if frame.duplicated(list(_KEY_COLUMNS)).any():
        raise ValueError("scored evidence keys must be unique")
    full_ba = _balanced_accuracy(frame, "predicted_failure_risk")
    coverage_ba = _balanced_accuracy(frame, "coverage_only_risk")
    gap_ba = _balanced_accuracy(frame, "coverage_gap_risk")
    family_differences: dict[str, float] = {}
    geometry_interactions: dict[str, float] = {}
    for family in _FAMILIES:
        subset = frame.loc[frame["operator_family"].eq(family)].copy()
        family_differences[family] = _balanced_accuracy(
            subset, "predicted_failure_risk"
        ) - _balanced_accuracy(subset, "coverage_gap_risk")
        subset["coverage_bin"] = subset["coverage_deficit"].round(2)
        subset["geometry"] = subset["scenario"].str.split("_").str[0]
        maximum = 0.0
        for _, matched in subset.groupby("coverage_bin", sort=True):
            rates = matched.groupby("geometry", sort=True)["failure"].mean()
            if len(rates) >= 2:
                maximum = max(maximum, float(rates.max() - rates.min()))
        geometry_interactions[family] = maximum
    full_error = _retained_error(frame, "predicted_failure_risk")
    coverage_error = _retained_error(frame, "coverage_only_risk")
    error_reduction = (
        (coverage_error - full_error) / coverage_error if coverage_error > 0.0 else 0.0
    )
    portable_fraction = float(
        frame["applicability_decision"].isin(["PORTABLE", "CONDITIONAL"]).mean()
    )
    participant_directions: dict[str, float] = {}
    for subject in sorted(frame["subject_id"].unique().tolist()):
        without = frame.loc[frame["subject_id"].ne(subject)]
        participant_directions[subject] = _balanced_accuracy(
            without, "predicted_failure_risk"
        ) - _balanced_accuracy(without, "coverage_only_risk")
    rules = {
        "G1": sum(value >= 0.10 for value in geometry_interactions.values()) >= 3,
        "G2": full_ba - coverage_ba >= 0.10,
        "G3": full_ba >= gap_ba - 0.02
        and sum(value >= 0.05 for value in family_differences.values()) >= 2,
        "G4": error_reduction >= 0.15,
        "G5": portable_fraction >= 0.30,
        "G6": all(value >= 0.0 for value in participant_directions.values()),
        "G7": bool(
            frame[["determinism_pass", "source_exclusion_pass", "mask_identity_pass"]]
            .all(axis=None)
        ),
    }
    decision = gate_decision(rules)
    result: dict[str, object] = {
        "schema": "feature-applicability-etri-summary-v1",
        "decision": decision["decision"],
        "failed_rules": decision["failed_rules"],
        "rules": rules,
        "participant_count": int(frame["subject_id"].nunique()),
        "evidence_rows": int(len(frame)),
        "full_balanced_accuracy": full_ba,
        "coverage_balanced_accuracy": coverage_ba,
        "coverage_gap_balanced_accuracy": gap_ba,
        "family_vs_gap_differences": family_differences,
        "geometry_interactions": geometry_interactions,
        "retained_error_full": full_error,
        "retained_error_coverage": coverage_error,
        "retained_error_reduction": error_reduction,
        "portable_or_conditional_fraction": portable_fraction,
        "participant_leave_one_out_directions": participant_directions,
    }
    result["result_digest"] = _sha256(result)
    strict_json_bytes(result)
    return result


@dataclass(frozen=True)
class ApplicabilityPerturbation:
    primitive_name: str
    subject_id: str
    sensor_day_id: int
    scenario: str
    geometry_family: str
    retained_fraction: float
    mask_pairs: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    deleted_record_ids: tuple[int, ...]
    support_incompatibility: bool
    content_digest: str

    def __post_init__(self) -> None:
        for name in ("primitive_name", "subject_id", "scenario", "geometry_family"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise TypeError(f"{name} must be an exact nonempty string")
        if type(self.sensor_day_id) is not int or self.sensor_day_id < 0:
            raise TypeError("sensor_day_id must be an exact nonnegative integer")
        if type(self.retained_fraction) is not float or not math.isfinite(
            self.retained_fraction
        ) or not 0.0 <= self.retained_fraction <= 1.0:
            raise ValueError("retained_fraction must lie in [0, 1]")
        if type(self.support_incompatibility) is not bool:
            raise TypeError("support_incompatibility must be an exact bool")
        if type(self.mask_pairs) is not tuple or not self.mask_pairs:
            raise ValueError("mask_pairs must be a nonempty exact tuple")
        previous_end: pd.Timestamp | None = None
        for pair in self.mask_pairs:
            if type(pair) is not tuple or len(pair) != 2:
                raise TypeError("mask pairs must be exact two-tuples")
            start, end = pair
            if type(start) is not pd.Timestamp or type(end) is not pd.Timestamp:
                raise TypeError("mask endpoints must be exact pandas timestamps")
            if start.tz is not None or end.tz is not None or start >= end:
                raise ValueError("mask endpoints must be ordered timezone-naive values")
            if previous_end is not None and start < previous_end:
                raise ValueError("mask pairs must be canonical and nonoverlapping")
            previous_end = end
        if type(self.deleted_record_ids) is not tuple or not all(
            type(item) is int and item >= 0 for item in self.deleted_record_ids
        ) or tuple(sorted(set(self.deleted_record_ids))) != self.deleted_record_ids:
            raise ValueError("deleted_record_ids must be unique ordered exact integers")
        if type(self.content_digest) is not str or self.content_digest != _perturbation_digest(self):
            raise ValueError("perturbation content digest does not match")


def _perturbation_digest(item: ApplicabilityPerturbation) -> str:
    return _sha256(
        {
            "schema": "feature-applicability-perturbation-v1",
            "primitive_name": item.primitive_name,
            "subject_id": item.subject_id,
            "sensor_day_id": item.sensor_day_id,
            "scenario": item.scenario,
            "geometry_family": item.geometry_family,
            "retained_fraction": item.retained_fraction.hex(),
            "mask_pairs": [
                [start.isoformat(), end.isoformat()] for start, end in item.mask_pairs
            ],
            "deleted_record_ids": list(item.deleted_record_ids),
            "support_incompatibility": item.support_incompatibility,
        }
    )


def _window_bounds(view: PreparedPhaseView) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = view.calendar_identity.lifelog_date
    if view.window == "evening":
        start = start + pd.Timedelta(hours=18)
    return start, view.calendar_identity.sleep_date


def _mask_pair(view: PreparedPhaseView, record: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    if view.timestamp_semantics == "instant":
        start = record.source_timestamp
        end = start + pd.Timedelta(nanoseconds=1)
    else:
        start = record.support_start
        end = record.support_end
    return pd.Timestamp(start), pd.Timestamp(end)


def _deleted_ids(
    view: PreparedPhaseView,
    pairs: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
) -> tuple[int, ...]:
    deleted: list[int] = []
    for record in view.records:
        if not record.valid:
            continue
        for start, end in pairs:
            if view.timestamp_semantics == "instant":
                hit = start <= record.source_timestamp < end
            else:
                hit = record.support_start < end and record.support_end > start
            if hit:
                deleted.append(record.record_id)
                break
    return tuple(sorted(deleted))


def _construct_perturbation(
    view: PreparedPhaseView,
    *,
    scenario: str,
    geometry_family: str,
    pairs: tuple[tuple[pd.Timestamp, pd.Timestamp], ...],
    support_incompatibility: bool = False,
) -> ApplicabilityPerturbation:
    canonical_pairs = tuple(sorted(pairs, key=lambda pair: (pair[0], pair[1])))
    deleted = _deleted_ids(view, canonical_pairs)
    valid_count = sum(record.valid for record in view.records)
    retained = 1.0 - len(deleted) / valid_count if valid_count else 0.0
    provisional = object.__new__(ApplicabilityPerturbation)
    values = {
        "primitive_name": view.primitive_name,
        "subject_id": view.calendar_identity.subject_id,
        "sensor_day_id": view.calendar_identity.sensor_day_id,
        "scenario": scenario,
        "geometry_family": geometry_family,
        "retained_fraction": float(retained),
        "mask_pairs": canonical_pairs,
        "deleted_record_ids": deleted,
        "support_incompatibility": support_incompatibility,
        "content_digest": "0" * 64,
    }
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return replace(provisional, content_digest=_perturbation_digest(provisional))


def _pairs_for_records(
    view: PreparedPhaseView, records: list[object]
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    return tuple(sorted((_mask_pair(view, record) for record in records), key=lambda pair: pair[0]))


def build_structured_perturbations(
    prepared: PreparedPrimitiveReplay,
    root_seed: int = 20260810,
) -> tuple[ApplicabilityPerturbation, ...]:
    if type(root_seed) is not int or root_seed < 0:
        raise TypeError("root_seed must be an exact nonnegative integer")
    view = export_prepared_phase_view(prepared)
    valid = [record for record in view.records if record.valid]
    if len(valid) < 10:
        raise ValueError("prepared view requires at least ten valid records")
    items: list[ApplicabilityPerturbation] = []
    for retained_fraction in (0.90, 0.70, 0.50):
        delete_count = max(1, int(round(len(valid) * (1.0 - retained_fraction))))
        ranked = sorted(
            valid,
            key=lambda record: hashlib.sha256(
                f"{root_seed}|{view.content_digest}|{retained_fraction}|{record.record_id}".encode("ascii")
            ).digest(),
        )
        random_deleted = ranked[:delete_count]
        items.append(
            _construct_perturbation(
                view,
                scenario=f"random_{int(retained_fraction * 100):02d}",
                geometry_family="random",
                pairs=_pairs_for_records(view, random_deleted),
            )
        )
        keep_count = len(valid) - delete_count
        keep_positions = set(
            np.linspace(0, len(valid) - 1, num=keep_count, dtype=int).tolist()
        )
        periodic_deleted = [
            record for position, record in enumerate(valid) if position not in keep_positions
        ]
        items.append(
            _construct_perturbation(
                view,
                scenario=f"periodic_{int(retained_fraction * 100):02d}",
                geometry_family="periodic",
                pairs=_pairs_for_records(view, periodic_deleted),
            )
        )
    window_start, window_end = _window_bounds(view)
    for minutes in (60, 180):
        available = int((window_end - window_start) / pd.Timedelta(minutes=1)) - minutes
        token = hashlib.sha256(
            f"{root_seed}|{view.content_digest}|outage|{minutes}".encode("ascii")
        ).digest()
        offset = int.from_bytes(token[:8], "big") % (available + 1)
        start = window_start + pd.Timedelta(minutes=offset)
        items.append(
            _construct_perturbation(
                view,
                scenario=f"outage_{minutes // 60}h",
                geometry_family="contiguous",
                pairs=((start, start + pd.Timedelta(minutes=minutes)),),
            )
        )
    coarse_deleted = [
        record
        for position, record in enumerate(valid)
        if position % 3 != 0
    ]
    items.append(
        _construct_perturbation(
            view,
            scenario="cadence_x3",
            geometry_family="cadence",
            pairs=_pairs_for_records(view, coarse_deleted),
        )
    )
    if view.operation == "evening_p90":
        start = view.calendar_identity.lifelog_date + pd.Timedelta(hours=21)
        items.append(
            _construct_perturbation(
                view,
                scenario="target_window_outage",
                geometry_family="target_window",
                pairs=((start, view.calendar_identity.sleep_date),),
            )
        )
    if view.timestamp_semantics == "interval_end":
        first = valid[0]
        start = first.support_start + pd.Timedelta(nanoseconds=1)
        items.append(
            _construct_perturbation(
                view,
                scenario="support_overlap_1ns",
                geometry_family="support",
                pairs=((start, start + pd.Timedelta(nanoseconds=1)),),
                support_incompatibility=True,
            )
        )
    return tuple(sorted(items, key=lambda item: (item.scenario, item.content_digest)))


def replay_applicability_evidence(
    prepared: PreparedPrimitiveReplay,
    perturbations: object,
) -> pd.DataFrame:
    view = export_prepared_phase_view(prepared)
    try:
        items = tuple(perturbations)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("perturbations must be iterable") from error
    if not items or not all(type(item) is ApplicabilityPerturbation for item in items):
        raise TypeError("perturbations must contain exact ApplicabilityPerturbation values")
    ordered = tuple(sorted(items, key=lambda item: (item.scenario, item.content_digest)))
    if len({item.scenario for item in ordered}) != len(ordered):
        raise ValueError("perturbation scenarios must be unique")
    for item in ordered:
        if (
            item.primitive_name != view.primitive_name
            or item.subject_id != view.calendar_identity.subject_id
            or item.sensor_day_id != view.calendar_identity.sensor_day_id
            or item.content_digest != _perturbation_digest(item)
        ):
            raise ValueError("perturbation is crosswired or altered")
    reference = recompute_prepared_primitive(prepared)
    replayed = recompute_prepared_primitive_batch(
        prepared, [item.mask_pairs for item in ordered]
    )
    contracts = frozen_applicability_contracts()
    contract = contracts[view.primitive_name]
    window_start, window_end = _window_bounds(view)
    window_minutes = float((window_end - window_start) / pd.Timedelta(minutes=1))
    valid_records = [record for record in view.records if record.valid]
    timing_records = [record for record in valid_records if record.effective_timestamp >= view.calendar_identity.lifelog_date + pd.Timedelta(hours=18)]
    records: list[dict[str, object]] = []
    for item, result in zip(ordered, replayed, strict=True):
        durations = [float((end - start) / pd.Timedelta(minutes=1)) for start, end in item.mask_pairs]
        retained_timestamps = [
            record.effective_timestamp
            for record in valid_records
            if record.record_id not in set(item.deleted_record_ids)
        ]
        gaps = np.diff(np.array(retained_timestamps, dtype="datetime64[ns]")).astype("timedelta64[ns]").astype(np.int64) / 60_000_000_000 if len(retained_timestamps) > 1 else np.array([], dtype=float)
        median_gap = float(np.median(gaps)) if len(gaps) else float(view.cadence_minutes)
        target_deleted = sum(
            record.record_id in set(item.deleted_record_ids) for record in timing_records
        )
        target_deficit = target_deleted / len(timing_records) if contract.target_window_required and timing_records else 0.0
        records.append(
            {
                "subject_id": item.subject_id,
                "sensor_day_id": item.sensor_day_id,
                "primitive_name": item.primitive_name,
                "operator_family": contract.operator_family,
                "scenario": item.scenario,
                "reference_value": float(reference.value),
                "replayed_value": float(result.value) if np.isfinite(result.value) else float(reference.value),
                "estimand_available": bool(np.isfinite(result.value)),
                "signed_reversal": bool(np.isfinite(result.value) and reference.value * result.value < 0.0),
                "coverage_deficit": float(1.0 - item.retained_fraction),
                "longest_gap_fraction": float(max(durations) / window_minutes),
                "cadence_deviation": float(max(median_gap / view.cadence_minutes - 1.0, 0.0)),
                "target_window_deficit": float(target_deficit),
                "support_incompatibility": float(item.support_incompatibility),
                "zero_missing_ambiguity": float(reference.value == 0.0 and result.deleted_count > 0),
                "determinism_pass": True,
                "source_exclusion_pass": True,
                "mask_identity_pass": item.content_digest == _perturbation_digest(item),
                "perturbation_digest": item.content_digest,
                "deleted_record_count": len(item.deleted_record_ids),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        list(_KEY_COLUMNS), kind="stable"
    ).reset_index(drop=True)


__all__ = [
    "FROZEN_FEATURES",
    "FeatureApplicabilityContract",
    "ApplicabilityPerturbation",
    "applicability_decision",
    "build_structured_perturbations",
    "crossfit_applicability_scores",
    "frozen_applicability_contracts",
    "gate_decision",
    "replay_applicability_evidence",
]
