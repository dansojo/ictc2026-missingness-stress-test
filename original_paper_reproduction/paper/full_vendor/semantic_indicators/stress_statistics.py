"""Preregistered participant-level inference for structured missingness stress tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math

import numpy as np
import pandas as pd

from .validation import holm_adjust


FEATURE_FAMILY_ORDER = ("event_count", "state_ratio", "intensity", "timing")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _participant_values(
    frame: pd.DataFrame,
    *,
    participant_column: str,
    delta_column: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    if type(frame) is not pd.DataFrame:
        raise TypeError("participant_deltas must be an exact pandas DataFrame")
    if participant_column not in frame or delta_column not in frame:
        raise ValueError("participant delta columns are incomplete")
    if frame.empty:
        raise ValueError("participant delta table must be nonempty")
    rows: list[tuple[str, float]] = []
    for participant, delta in frame[[participant_column, delta_column]].itertuples(
        index=False, name=None
    ):
        if type(participant) is not str or not participant:
            raise TypeError("participant identities must be exact nonempty strings")
        if type(delta) not in (float, int):
            raise TypeError("participant deltas must be exact numeric scalars")
        value = float(delta)
        if not math.isfinite(value):
            raise ValueError("participant deltas must be finite")
        rows.append((participant, value))
    if len({participant for participant, _ in rows}) != len(rows):
        raise ValueError("participant identities must be unique")
    rows.sort(key=lambda item: item[0])
    return tuple(item[0] for item in rows), np.asarray([item[1] for item in rows])


@dataclass(frozen=True)
class ExactTwoSidedSignFlip:
    participant_count: int
    pattern_count: int
    observed_statistic: float
    arithmetic_mean: float
    participant_median: float
    p_value: float
    alternative: str
    content_digest: str


@dataclass(frozen=True)
class ParticipantPercentileInterval:
    feature_family: str
    participant_count: int
    draws: int
    root_seed: int
    median: float
    ci_low: float
    ci_high: float
    seed_sha256: str
    content_digest: str


@dataclass(frozen=True)
class InternalDecision:
    label: str
    exact_matching_complete: bool
    positive_family_count: int
    confirmatory_family_count: int
    dose_response_family_count: int
    denominators_complete: bool
    content_digest: str


@dataclass(frozen=True)
class StressSummaryTables:
    participant_deltas: pd.DataFrame
    denominators: pd.DataFrame
    content_digest: str


def exact_two_sided_sign_flip(
    participant_deltas: pd.DataFrame,
    *,
    participant_column: str = "subject_id",
    delta_column: str = "delta",
    minimum_participants: int = 8,
    maximum_participants: int = 10,
) -> ExactTwoSidedSignFlip:
    """Enumerate all signs and use the inclusive absolute-mean tail."""
    participants, values = _participant_values(
        participant_deltas,
        participant_column=participant_column,
        delta_column=delta_column,
    )
    count = len(participants)
    if count < minimum_participants or count > maximum_participants:
        raise ValueError("eligible participant count must be between 8 and 10")
    observed = abs(float(np.mean(values)))
    tail_count = 0
    pattern_count = 2**count
    for signs in itertools.product((-1.0, 1.0), repeat=count):
        statistic = abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        if statistic >= observed:
            tail_count += 1
    payload = [
        "exact-two-sided-sign-flip-v1",
        list(participants),
        [float(value).hex() for value in values],
        pattern_count,
        tail_count,
    ]
    return ExactTwoSidedSignFlip(
        participant_count=count,
        pattern_count=pattern_count,
        observed_statistic=observed,
        arithmetic_mean=float(np.mean(values)),
        participant_median=float(np.median(values)),
        p_value=tail_count / pattern_count,
        alternative="two_sided",
        content_digest=_digest(payload),
    )


def participant_percentile_interval(
    participant_deltas: pd.DataFrame,
    *,
    feature_family: str,
    draws: int = 10_000,
    root_seed: int = 42,
    participant_column: str = "subject_id",
    delta_column: str = "delta",
) -> ParticipantPercentileInterval:
    """Return the frozen participant-cluster percentile interval for the median."""
    if type(feature_family) is not str or feature_family not in FEATURE_FAMILY_ORDER:
        raise ValueError("feature_family is not in the frozen four-family order")
    if type(draws) is not int or draws != 10_000:
        raise ValueError("bootstrap draws must equal 10000")
    if type(root_seed) is not int or type(root_seed) is bool or root_seed != 42:
        raise ValueError("root_seed must equal 42")
    participants, values = _participant_values(
        participant_deltas,
        participant_column=participant_column,
        delta_column=delta_column,
    )
    seed_payload = ["participant-cluster-bootstrap-v1", root_seed, feature_family]
    seed_sha256 = _digest(seed_payload)
    seed_uint64 = int.from_bytes(bytes.fromhex(seed_sha256)[:8], "big", signed=False)
    generator = np.random.Generator(np.random.PCG64(seed_uint64))
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    ci_low, ci_high = np.quantile(medians, (0.025, 0.975), method="linear")
    payload = [
        "participant-percentile-interval-v1",
        feature_family,
        list(participants),
        [float(value).hex() for value in values],
        draws,
        root_seed,
        float(ci_low).hex(),
        float(ci_high).hex(),
    ]
    return ParticipantPercentileInterval(
        feature_family=feature_family,
        participant_count=len(values),
        draws=draws,
        root_seed=root_seed,
        median=float(np.median(values)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        seed_sha256=seed_sha256,
        content_digest=_digest(payload),
    )


def summarize_family_inference(participant_deltas: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen four-family tests and one fixed Holm correction family."""
    required = {"feature_family", "subject_id", "delta"}
    if type(participant_deltas) is not pd.DataFrame or not required.issubset(
        participant_deltas.columns
    ):
        raise ValueError("participant family delta table is incomplete")
    unknown = set(participant_deltas["feature_family"].unique()) - set(
        FEATURE_FAMILY_ORDER
    )
    if unknown:
        raise ValueError("unknown feature family")
    rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for family in FEATURE_FAMILY_ORDER:
        family_frame = participant_deltas.loc[
            participant_deltas["feature_family"].eq(family), ["subject_id", "delta"]
        ].copy()
        if family_frame.empty:
            participants, values = (), np.asarray([], dtype=float)
        else:
            participants, values = _participant_values(
                family_frame, participant_column="subject_id", delta_column="delta"
            )
        eligible = 8 <= len(values) <= 10
        interval = (
            participant_percentile_interval(family_frame, feature_family=family)
            if len(values)
            else None
        )
        sign_flip = exact_two_sided_sign_flip(family_frame) if eligible else None
        raw_p = sign_flip.p_value if sign_flip is not None else 1.0
        raw_p_values.append(raw_p)
        rows.append(
            {
                "feature_family": family,
                "status": "eligible" if eligible else "ineligible",
                "eligible_participant_count": len(participants),
                "participant_median": float(np.median(values)) if len(values) else np.nan,
                "participant_mean": float(np.mean(values)) if len(values) else np.nan,
                "direction_agreement_count": int(np.sum(values > 0.0)),
                "ci_low": interval.ci_low if interval is not None else np.nan,
                "ci_high": interval.ci_high if interval is not None else np.nan,
                "raw_p_value": raw_p,
            }
        )
    adjusted = holm_adjust(np.asarray(raw_p_values, dtype=float))
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p_value"] = float(value)
    return pd.DataFrame(rows)


def summarize_stress_results(
    cell_replays: pd.DataFrame,
    *,
    participant_roster: tuple[str, ...],
) -> StressSummaryTables:
    """Pair contiguous/random cells, retain denominators, then aggregate participants."""
    required = {
        "subject_id",
        "sensor_day_id",
        "primitive",
        "feature_family",
        "draw",
        "scenario",
        "status",
        "standardized_absolute_error",
    }
    if type(cell_replays) is not pd.DataFrame or not required.issubset(
        cell_replays.columns
    ):
        raise ValueError("cell replay table is incomplete")
    if (
        type(participant_roster) is not tuple
        or not participant_roster
        or len(set(participant_roster)) != len(participant_roster)
        or not all(type(value) is str and value for value in participant_roster)
    ):
        raise ValueError("participant_roster must be a unique exact string tuple")
    if not set(cell_replays["subject_id"]).issubset(set(participant_roster)):
        raise ValueError("cell replay table contains a subject outside the roster")
    scenarios = ("contiguous_20pct", "scattered_random_20pct")
    if not cell_replays["scenario"].isin(scenarios).all():
        raise ValueError("primary stress summary contains an unknown scenario")
    keys = [
        "subject_id",
        "sensor_day_id",
        "primitive",
        "feature_family",
        "draw",
    ]
    if cell_replays.duplicated([*keys, "scenario"]).any():
        raise ValueError("cell replay table contains duplicate scenario keys")
    counts = cell_replays.groupby(keys, sort=False)["scenario"].nunique()
    if not counts.eq(2).all():
        raise ValueError("every primary cell key must contain both scenarios")
    indexed = cell_replays.set_index([*keys, "scenario"])
    contiguous = indexed.xs("contiguous_20pct", level="scenario")
    random = indexed.xs("scattered_random_20pct", level="scenario")
    if not contiguous.index.equals(random.index):
        contiguous = contiguous.sort_index()
        random = random.sort_index()
        if not contiguous.index.equals(random.index):
            raise ValueError("primary scenario keys are crosswired")
    finite = contiguous["status"].eq("finite") & random["status"].eq("finite")
    c_error = pd.to_numeric(contiguous["standardized_absolute_error"], errors="coerce")
    r_error = pd.to_numeric(random["standardized_absolute_error"], errors="coerce")
    finite &= np.isfinite(c_error) & np.isfinite(r_error)
    paired = contiguous.reset_index().loc[:, keys]
    paired["paired_finite"] = finite.to_numpy(dtype=bool)
    paired["delta"] = np.where(
        paired["paired_finite"],
        c_error.to_numpy(dtype=float) - r_error.to_numpy(dtype=float),
        np.nan,
    )
    contiguous_status = contiguous["status"].astype(str).to_numpy()
    random_status = random["status"].astype(str).to_numpy()
    paired["paired_reason"] = np.where(
        paired["paired_finite"],
        "finite_primary_delta",
        np.char.add(np.char.add(contiguous_status, "|"), random_status),
    )
    finite_rows = paired.loc[paired["paired_finite"]].copy()
    participant_deltas = (
        finite_rows.groupby(["feature_family", "subject_id"], sort=True)["delta"]
        .median()
        .reset_index()
    )
    denominator_rows: list[dict[str, object]] = []
    for family in FEATURE_FAMILY_ORDER:
        for subject_id in participant_roster:
            subject_rows = paired.loc[
                paired["feature_family"].eq(family)
                & paired["subject_id"].eq(subject_id)
            ]
            finite_count = int(subject_rows["paired_finite"].sum())
            reason_counts = tuple(
                (str(reason), int(count))
                for reason, count in subject_rows["paired_reason"]
                .value_counts(sort=False)
                .sort_index()
                .items()
            )
            denominator_rows.append(
                {
                    "feature_family": family,
                    "subject_id": subject_id,
                    "paired_total": int(len(subject_rows)),
                    "paired_finite": finite_count,
                    "paired_abstained_or_unavailable": int(
                        len(subject_rows) - finite_count
                    ),
                    "reporting_status": (
                        "eligible" if finite_count > 0 else "ineligible"
                    ),
                    "reason": (
                        "finite_primary_delta"
                        if finite_count > 0
                        else "no_paired_finite_delta"
                    ),
                    "reason_counts": reason_counts,
                }
            )
    denominators = pd.DataFrame(denominator_rows)
    digest_rows = [
        [row.feature_family, row.subject_id, float(row.delta).hex()]
        for row in participant_deltas.itertuples(index=False)
    ]
    denominator_payload = denominators.to_dict(orient="records")
    return StressSummaryTables(
        participant_deltas=participant_deltas,
        denominators=denominators,
        content_digest=_digest(
            ["stress-summary-tables-v1", digest_rows, denominator_payload]
        ),
    )


def evaluate_internal_decision(
    family_results: pd.DataFrame,
    *,
    mask_matching_complete: bool,
    denominator_ledger: pd.DataFrame,
    dose_response_table: pd.DataFrame,
) -> InternalDecision:
    """Evaluate the frozen strong/bounded/STOP hierarchy without result tuning."""
    required = {
        "feature_family",
        "participant_median",
        "holm_p_value",
        "direction_agreement_count",
        "eligible_participant_count",
    }
    if type(family_results) is not pd.DataFrame or not required.issubset(
        family_results.columns
    ):
        raise ValueError("family decision table is incomplete")
    ordered = family_results.set_index("feature_family").reindex(FEATURE_FAMILY_ORDER)
    if ordered.isna().all(axis=1).any() or len(family_results) != 4:
        raise ValueError("decision table must contain exactly the four frozen families")
    denominator_required = {
        "feature_family", "subject_id", "reporting_status", "reason",
        "paired_total", "paired_finite", "paired_abstained_or_unavailable",
        "reason_counts",
    }
    if type(denominator_ledger) is not pd.DataFrame or not denominator_required.issubset(
        denominator_ledger.columns
    ):
        raise ValueError("denominator ledger is incomplete")
    if denominator_ledger.duplicated(["feature_family", "subject_id"]).any():
        raise ValueError("denominator ledger contains duplicate family-subject keys")
    denominator_families = tuple(
        family
        for family in FEATURE_FAMILY_ORDER
        if family in set(denominator_ledger["feature_family"])
    )
    rosters = [
        tuple(sorted(denominator_ledger.loc[
            denominator_ledger["feature_family"].eq(family), "subject_id"
        ]))
        for family in FEATURE_FAMILY_ORDER
    ]
    counts_valid = True
    for row in denominator_ledger.itertuples(index=False):
        if not all(
            type(value) is int and value >= 0
            for value in (
                row.paired_total,
                row.paired_finite,
                row.paired_abstained_or_unavailable,
            )
        ):
            counts_valid = False
            break
        reason_counts = row.reason_counts
        reason_map = dict(reason_counts) if type(reason_counts) is tuple else {}
        if (
            type(reason_counts) is not tuple
            or not all(
                type(item) is tuple
                and len(item) == 2
                and type(item[0]) is str
                and bool(item[0])
                and type(item[1]) is int
                and item[1] >= 0
                for item in reason_counts
            )
            or sum(item[1] for item in reason_counts) != row.paired_total
            or len(reason_map) != len(reason_counts)
            or row.paired_finite + row.paired_abstained_or_unavailable
            != row.paired_total
            or reason_map.get("finite_primary_delta", 0) != row.paired_finite
            or sum(
                count for reason, count in reason_counts
                if reason != "finite_primary_delta"
            ) != row.paired_abstained_or_unavailable
            or (row.reporting_status == "eligible") != (row.paired_finite > 0)
            or row.reason
            != (
                "finite_primary_delta"
                if row.paired_finite > 0
                else "no_paired_finite_delta"
            )
        ):
            counts_valid = False
            break
    eligible_by_family = denominator_ledger.loc[
        denominator_ledger["reporting_status"].eq("eligible")
    ].groupby("feature_family").size()
    denominators = bool(
        denominator_families == FEATURE_FAMILY_ORDER
        and all(len(roster) == 10 for roster in rosters)
        and all(roster == rosters[0] for roster in rosters[1:])
        and denominator_ledger["reporting_status"].isin(("eligible", "ineligible")).all()
        and denominator_ledger["reason"].map(lambda value: type(value) is str and bool(value)).all()
        and counts_valid
        and all(
            int(eligible_by_family.get(family, 0))
            == int(ordered.loc[family, "eligible_participant_count"])
            for family in FEATURE_FAMILY_ORDER
        )
    )
    if not denominators:
        raise ValueError("denominator ledger does not preserve the full ten-person roster")
    dose_required = {
        "feature_family", "missingness_rate", "participant_median",
        "participant_denominator", "eligible_participant_count",
        "nonfinite_participant_count",
    }
    if type(dose_response_table) is not pd.DataFrame or not dose_required.issubset(
        dose_response_table.columns
    ):
        raise ValueError("dose-response table is incomplete")
    if dose_response_table.duplicated(["feature_family", "missingness_rate"]).any():
        raise ValueError("dose-response table contains duplicate keys")
    dose_flags: dict[str, bool] = {}
    for family in FEATURE_FAMILY_ORDER:
        rows = dose_response_table.loc[
            dose_response_table["feature_family"].eq(family)
        ].sort_values("missingness_rate", kind="stable")
        rates = pd.to_numeric(rows["missingness_rate"], errors="coerce").to_numpy(float)
        medians = pd.to_numeric(rows["participant_median"], errors="coerce").to_numpy(float)
        dose_denominators = pd.to_numeric(
            rows["participant_denominator"], errors="coerce"
        ).to_numpy(float)
        eligible_counts = pd.to_numeric(
            rows["eligible_participant_count"], errors="coerce"
        ).to_numpy(float)
        nonfinite_counts = pd.to_numeric(
            rows["nonfinite_participant_count"], errors="coerce"
        ).to_numpy(float)
        if (
            len(rows) != 3
            or not np.array_equal(rates, np.array([0.10, 0.20, 0.40]))
            or not np.array_equal(dose_denominators, np.array([10.0, 10.0, 10.0]))
            or not np.isfinite(eligible_counts).all()
            or not np.isfinite(nonfinite_counts).all()
            or not np.array_equal(
                eligible_counts + nonfinite_counts, dose_denominators
            )
        ):
            raise ValueError("dose-response table does not contain the frozen denominator grid")
        complete = bool(
            np.array_equal(eligible_counts, dose_denominators)
            and np.array_equal(nonfinite_counts, np.zeros(3))
            and np.isfinite(medians).all()
        )
        dose_flags[family] = bool(complete and np.all(np.diff(medians) >= 0.0))

    positive_count = int((ordered["participant_median"] > 0.0).sum())
    confirmatory_count = 0
    for row in ordered.itertuples():
        n = int(row.eligible_participant_count)
        threshold = max(8, math.ceil(0.80 * n))
        if (
            8 <= n <= 10
            and float(row.holm_p_value) <= 0.05
            and int(row.direction_agreement_count) >= threshold
        ):
            confirmatory_count += 1
    dose_count = sum(dose_flags.values())
    exact = type(mask_matching_complete) is bool and mask_matching_complete
    strong = (
        exact
        and positive_count >= 3
        and confirmatory_count >= 2
        and dose_count >= 3
        and denominators
    )
    if strong:
        label = "strong_internal_go"
    elif exact and positive_count >= 3 and denominators:
        label = "bounded_internal_go"
    else:
        label = "stop_diagnostic_inconclusive"
    payload = [
        "structured-missingness-internal-decision-v1",
        label,
        exact,
        positive_count,
        confirmatory_count,
        dose_count,
        denominators,
    ]
    return InternalDecision(
        label=label,
        exact_matching_complete=exact,
        positive_family_count=positive_count,
        confirmatory_family_count=confirmatory_count,
        dose_response_family_count=dose_count,
        denominators_complete=denominators,
        content_digest=_digest(payload),
    )
