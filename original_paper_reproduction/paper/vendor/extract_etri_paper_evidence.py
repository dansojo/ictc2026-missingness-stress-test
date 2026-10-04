from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FAMILY_ORDER = ["event_count", "state_ratio", "intensity", "timing"]
CONDITION_ORDER = [
    "scattered_random_20pct",
    "contiguous_20pct",
    "event_boundary_20pct",
]

FEATURE_CONTRACTS = {
    "event_count": {
        "primitives": "step_load_24h",
        "contract_interpretation": "Sum of valid nonnegative event contributions",
        "operation_specific_eligibility": "At least one valid event contribution",
    },
    "state_ratio": {
        "primitives": "screen_load_24h; phone_activity_load_24h",
        "contract_interpretation": "Valid positive-state units divided by valid state units",
        "operation_specific_eligibility": "Positive valid-state denominator",
    },
    "intensity": {
        "primitives": (
            "usage_load_24h; mobile_light_exposure_24h; "
            "wearable_light_exposure_24h"
        ),
        "contract_interpretation": (
            "Valid nonnegative duration or transformed intensity contribution"
        ),
        "operation_specific_eligibility": (
            "At least one finite valid nonnegative contribution"
        ),
    },
    "timing": {
        "primitives": "screen_disengagement_p90; usage_disengagement_p90",
        "contract_interpretation": "Weighted timing quantile over positive event mass",
        "operation_specific_eligibility": (
            "At least three positive source events and positive total event mass"
        ),
    },
}

MODEL_PARAMETERS = {
    "elasticnet_logistic_regression": {
        "solver": "saga",
        "penalty": "elasticnet",
        "l1_ratio": 0.5,
        "C": 1.0,
        "max_iter": 2000,
        "tol": 1e-4,
        "class_weight": None,
    },
    "lightgbm": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 10,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.9g")


def ordered(frame: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.Categorical(result[column], categories=order, ordered=True)
    return result.sort_values(column).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdd-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sdd = args.sdd_dir.resolve()
    source_root = args.source_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    diagnostic = sdd / "task-12-etri-production_20260810_230202"
    downstream = sdd / "task-12-downstream-etri-production-v1"
    diagnostic_manifest = diagnostic / "manifest.json"
    downstream_manifest = downstream / "manifest.json"
    diagnostic_report = sdd / "task-12-etri-result-report.md"
    downstream_report = sdd / "task-12-downstream-v11-independent-result-report.md"
    preregistration = sdd / "task-12-structured-missingness-preregistered-analysis-plan.md"
    downstream_protocol = sdd / "task-12-downstream-continuation-protocol-v1.md"
    outcome_source = source_root / "ETRI_Human_AI_v2/src/semantic_indicators/outcome.py"

    required = [
        diagnostic_manifest,
        downstream_manifest,
        diagnostic_report,
        downstream_report,
        preregistration,
        downstream_protocol,
        outcome_source,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required evidence: {missing}")

    diagnostic_decision = pd.read_parquet(diagnostic / "decision.parquet")
    family_inference = pd.read_parquet(diagnostic / "family_inference.parquet")
    denominators = pd.read_parquet(diagnostic / "denominators.parquet")
    dose_response = pd.read_parquet(diagnostic / "dose_response_family.parquet")
    participant_deltas = pd.read_parquet(diagnostic / "participant_deltas.parquet")
    primary = pd.read_parquet(
        diagnostic / "primary_comparison.parquet",
        columns=[
            "subject_id",
            "sensor_day_id",
            "primitive",
            "feature_family",
            "draw",
            "scenario",
            "status",
        ],
    )
    mask_ledger = pd.read_parquet(
        diagnostic / "mask_ledger.parquet",
        columns=["scenario", "mask_status"],
    )

    statistics = downstream / "statistics"
    downstream_decision = pd.read_parquet(statistics / "decision.parquet")
    condition_groups = pd.read_parquet(statistics / "condition_group_descriptives.parquet")
    participant_summaries = pd.read_parquet(statistics / "participant_summaries.parquet")
    global_conditions = pd.read_parquet(statistics / "global_condition_descriptives.parquet")

    assert diagnostic_decision.loc[0, "label"] == "stop_diagnostic_inconclusive"
    assert family_inference["feature_family"].nunique() == 4
    assert set(family_inference["eligible_participant_count"]) == {10}
    assert len(participant_deltas) == 40
    assert len(dose_response) == 12
    assert downstream_decision.loc[0, "integrity_complete"] == True  # noqa: E712
    assert downstream_decision.loc[0, "label"] == "corroborative_impact_not_demonstrated"
    assert len(condition_groups) == 1920
    assert len(participant_summaries) == 10

    selected_cells = (
        primary.loc[primary["scenario"] == "contiguous_20pct"]
        .drop_duplicates(["subject_id", "sensor_day_id", "primitive"])
    )
    assert len(selected_cells) == 800
    assert primary["subject_id"].nunique() == 10
    assert primary["primitive"].nunique() == 8
    assert primary["draw"].nunique() == 50

    denominator_family = (
        denominators.groupby("feature_family", as_index=False)[
            ["paired_total", "paired_finite", "paired_abstained_or_unavailable"]
        ]
        .sum()
    )
    selected_family = (
        selected_cells.groupby("feature_family", as_index=False)
        .size()
        .rename(columns={"size": "selected_cell_count"})
    )
    contracts = []
    for family in FAMILY_ORDER:
        row = {"feature_family": family, **FEATURE_CONTRACTS[family]}
        contracts.append(row)
    feature_contracts = pd.DataFrame(contracts)
    feature_contracts = feature_contracts.merge(selected_family, on="feature_family")
    feature_contracts = feature_contracts.merge(denominator_family, on="feature_family")
    feature_contracts.insert(4, "draws_per_condition", 50)
    feature_contracts["primary_finite_fraction"] = (
        feature_contracts["paired_finite"] / feature_contracts["paired_total"]
    )
    feature_contracts = ordered(feature_contracts, "feature_family", FAMILY_ORDER)

    inference = ordered(family_inference, "feature_family", FAMILY_ORDER)
    inference = inference.merge(denominator_family, on="feature_family", how="left")
    inference["direction_interpretation"] = inference["participant_median"].map(
        lambda value: (
            "contiguous_more_distorting" if value > 0 else "no_positive_median"
        )
    )
    inference["confirmatory_interpretation"] = inference["holm_p_value"].map(
        lambda value: "holm_significant" if value <= 0.05 else "not_holm_significant"
    )

    participant_deltas = ordered(participant_deltas, "feature_family", FAMILY_ORDER)
    participant_deltas = participant_deltas.sort_values(
        ["feature_family", "subject_id"]
    ).reset_index(drop=True)
    dose_response = ordered(dose_response, "feature_family", FAMILY_ORDER)
    dose_response = dose_response.sort_values(
        ["feature_family", "missingness_rate"]
    ).reset_index(drop=True)

    counts = (
        condition_groups.groupby("condition", as_index=False)[
            [
                "expected_count",
                "applicable_count",
                "structurally_na_count",
                "finite_count",
                "abstained_count",
                "stable_count",
                "flipped_count",
                "label_available_count",
                "label_available_finite_count",
            ]
        ]
        .sum()
    )
    participant_abs_change = global_conditions.loc[
        global_conditions["metric"] == "finite_abs_change_median",
        ["condition", "value", "nonnull_participant_count", "total_participant_count"],
    ].rename(columns={"value": "participant_first_median_abs_probability_change"})
    condition_summary = counts.merge(participant_abs_change, on="condition", how="left")
    condition_summary["coverage"] = (
        condition_summary["finite_count"] / condition_summary["applicable_count"]
    )
    condition_summary["abstention_rate"] = (
        condition_summary["abstained_count"] / condition_summary["applicable_count"]
    )
    condition_summary["flip_rate_finite"] = (
        condition_summary["flipped_count"] / condition_summary["finite_count"]
    )
    condition_summary["flip_fraction_applicable"] = (
        condition_summary["flipped_count"] / condition_summary["applicable_count"]
    )
    condition_summary = ordered(condition_summary, "condition", CONDITION_ORDER)

    family_condition_participant = (
        condition_groups.groupby(
            ["participant", "family", "condition"], as_index=False, sort=False
        )["finite_abs_change_median"]
        .median()
    )
    family_profile = (
        family_condition_participant.groupby(
            ["family", "condition"], as_index=False, sort=False
        )["finite_abs_change_median"]
        .median()
        .pivot(index="family", columns="condition", values="finite_abs_change_median")
        .reset_index()
        .rename(columns={"family": "feature_family"})
    )
    family_profile["contiguous_minus_random"] = (
        family_profile["contiguous_20pct"]
        - family_profile["scattered_random_20pct"]
    )
    family_profile = ordered(family_profile, "feature_family", FAMILY_ORDER)

    participant_summary = participant_summaries[
        ["participant_position", "participant", "group_cell_count", "p_median", "direction_positive"]
    ].copy()

    model_rows = []
    for model, parameters in MODEL_PARAMETERS.items():
        model_rows.append(
            {
                "model": model,
                "parameters_json": json.dumps(
                    parameters, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    model_specifications = pd.DataFrame(model_rows)

    output_files = {
        "feature_contracts.csv": feature_contracts,
        "primary_feature_inference.csv": inference,
        "participant_primary_deltas.csv": participant_deltas,
        "dose_response.csv": dose_response,
        "downstream_condition_summary.csv": condition_summary,
        "downstream_feature_profiles.csv": family_profile,
        "downstream_participant_primary.csv": participant_summary,
        "model_specifications.csv": model_specifications,
    }
    for name, frame in output_files.items():
        write_csv(frame, output / name)

    mask_counts = (
        mask_ledger.groupby(["scenario", "mask_status"])
        .size()
        .rename("count")
        .reset_index()
    )
    mask_status = {
        f"{row.scenario}:{row.mask_status}": int(row.count)
        for row in mask_counts.itertuples(index=False)
    }
    downstream_root = json.loads(downstream_manifest.read_text(encoding="utf-8"))
    study_design = {
        "schema_name": "ictc-2026-etri-paper-evidence-v1",
        "diagnostic": {
            "participants": 10,
            "primitives": 8,
            "selected_participant_day_primitive_cells": 800,
            "selected_days_per_participant_and_primitive": 10,
            "root_seed": 42,
            "draws_per_scenario": 50,
            "primary_conditions": ["contiguous_20pct", "scattered_random_20pct"],
            "secondary_event_condition": "event_boundary_20pct",
            "dose_conditions": ["contiguous_10pct", "contiguous_20pct", "contiguous_40pct"],
            "new_mask_rows": int(len(mask_ledger)),
            "mask_status_counts": mask_status,
            "inferential_unit": "participant",
            "primary_estimand": (
                "participant median of standardized absolute error under contiguous 20% "
                "minus matched scattered-random 20%"
            ),
            "bootstrap_resamples": 10000,
            "primary_test": "exact two-sided participant sign-flip",
            "multiplicity": "Holm adjustment over four fixed feature families",
            "decision_label": str(diagnostic_decision.loc[0, "label"]),
        },
        "downstream": {
            "participants": 10,
            "split": "leave-one-subject-out",
            "arms": ["matched_fixed_personalized", "semantic_personalized"],
            "models": ["elasticnet", "lightgbm"],
            "targets": ["S1", "S2", "S3", "S4"],
            "fit_count": 160,
            "fit_identity_formula": "10 folds x 2 arms x 2 models x 4 targets",
            "fit_policy": "fit once on unmasked outer-training rows; reuse across conditions",
            "reporting_threshold": 0.5,
            "conditions": CONDITION_ORDER,
            "observed_prediction_rows": int(downstream_root["observed_row_count"]),
            "expected_primary_keys": int(downstream_root["expected_primary_key_count"]),
            "condition_group_rows": int(len(condition_groups)),
            "group_cells": 640,
            "participant_summaries": 10,
            "bootstrap_resamples": 10000,
            "sign_flip_patterns": 1024,
            "recovery_new_model_fits": 0,
            "decision_label": str(downstream_decision.loc[0, "label"]),
        },
    }
    (output / "study_design.json").write_text(
        json.dumps(study_design, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    evidence_text = f"""# ETRI manuscript evidence inventory

## Scientific position

The ETRI evidence supports a diagnostic-guideline contribution, not a universal
performance-improvement or four-family structure-effect claim. The preregistered
diagnostic decision was `{diagnostic_decision.loc[0, 'label']}` because only two of
four families had a positive participant median, although exact matching, two
confirmatory family results, all four monotonic dose responses, and complete
denominators were retained. The independently verified downstream label was
`{downstream_decision.loc[0, 'label']}`.

## Candidate methods facts

- Ten ETRI participants, eight primitives, and 800 selected participant-day-primitive
  cells were evaluated with 50 deterministic draws per scenario and root seed 42.
- The primary contrast held the valid deleted-record count fixed and compared
  contiguous 20% loss with matched scattered-random 20% loss.
- Event-boundary 20% loss was secondary because a contract-defined boundary was not
  available for every cell.
- The observed-support primitive value was used as the reference. It is not unobserved
  ground truth; natural missingness remains part of the provenance.
- Participant-level medians were the inferential observations. Each feature family used
  a deterministic 10,000-resample participant bootstrap interval, an exact two-sided
  participant sign-flip test, and Holm correction over the fixed four-family set.
- Downstream analysis used ten LOSO folds, two feature arms, ElasticNet logistic
  regression and LightGBM, and four binary targets (S1-S4), giving 160 fit identities.
  Each model was fitted once on unmasked outer-training data and reused for all three
  held-out missingness conditions.

## Candidate diagnostic results

- State ratio: participant median = {inference.loc[inference.feature_family == 'state_ratio', 'participant_median'].iloc[0]:.6f},
  95% bootstrap interval [{inference.loc[inference.feature_family == 'state_ratio', 'ci_low'].iloc[0]:.6f},
  {inference.loc[inference.feature_family == 'state_ratio', 'ci_high'].iloc[0]:.6f}],
  10/10 positive participants, Holm-adjusted p =
  {inference.loc[inference.feature_family == 'state_ratio', 'holm_p_value'].iloc[0]:.7f}.
- Intensity: participant median = {inference.loc[inference.feature_family == 'intensity', 'participant_median'].iloc[0]:.6f},
  95% bootstrap interval [{inference.loc[inference.feature_family == 'intensity', 'ci_low'].iloc[0]:.6f},
  {inference.loc[inference.feature_family == 'intensity', 'ci_high'].iloc[0]:.6f}],
  10/10 positive participants, Holm-adjusted p =
  {inference.loc[inference.feature_family == 'intensity', 'holm_p_value'].iloc[0]:.7f}.
- Event count and timing both had a participant median of 0. Their Holm-adjusted
  p-values were 0.9375, so the data did not show a reliable contiguous-loss penalty for
  these two families.
- Standardized error increased monotonically from 10% to 20% to 40% contiguous
  missingness in all four feature families.

## Candidate downstream results

- The independently verified output contained {downstream_root['observed_row_count']:,}
  predictions. The frozen participant-first statistic was 0 with bootstrap sensitivity
  interval [0, 0], 1/10 positive participant directions, and exact sign-flip p = 1.0.
- Descriptively, participant-first median absolute probability change was
  {condition_summary.loc[condition_summary.condition == 'scattered_random_20pct', 'participant_first_median_abs_probability_change'].iloc[0]:.6f}
  for scattered random, {condition_summary.loc[condition_summary.condition == 'contiguous_20pct', 'participant_first_median_abs_probability_change'].iloc[0]:.6f}
  for contiguous, and {condition_summary.loc[condition_summary.condition == 'event_boundary_20pct', 'participant_first_median_abs_probability_change'].iloc[0]:.6f}
  for event-boundary loss.
- Aggregate decision flips were 4,384/314,464 (1.394%) for random,
  7,747/314,464 (2.464%) for contiguous, and 8,636/306,464 (2.818%) for
  event-boundary loss. Event-boundary loss also retained 8,000 structurally
  unavailable and 1,104 abstained rows.
- State-ratio and intensity features had positive descriptive contiguous-minus-random
  probability-change contrasts (+0.003170 and +0.007785). Timing was near zero
  (+0.000018), while event count was negative for contiguous versus random (-0.001462)
  but larger under event-boundary deletion.

## Claim boundary

Supported:

1. Equal missingness rates can yield different feature distortions under different
   loss geometries.
2. Feature families have heterogeneous structure-sensitivity profiles.
3. In ETRI, state-ratio and intensity features were more distorted by contiguous than
   matched scattered-random loss; event count and timing did not show the same reliable
   primary effect.
4. The workflow retains abstention, structural unavailability, and denominator
   topology instead of silently filtering them.
5. Downstream summaries provide descriptive context but did not meet the frozen
   corroborative-impact criterion.

Not supported:

1. All four feature families are consistently structure-sensitive.
2. Structured missingness reliably changes downstream decisions across participants.
3. The procedure improves predictive accuracy, reconstructs missing observations, or
   identifies unobserved ground truth.
4. External transfer; StudentLife and ExtraSensory remain pending.

## Display mapping

- Table I: `feature_contracts.csv` plus `study_design.json`.
- Fig. 2a: `participant_primary_deltas.csv` with family medians and intervals from
  `primary_feature_inference.csv`.
- Fig. 2b: `dose_response.csv`.
- Fig. 2c: `downstream_feature_profiles.csv`; condition-level flip, abstention, and
  structural-unavailability annotations come from `downstream_condition_summary.csv`.
"""
    (output / "manuscript_evidence.md").write_text(evidence_text, encoding="utf-8")

    source_files = {}
    for path in required:
        source_files[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    generated = {}
    for path in sorted(output.iterdir()):
        if path.name == "evidence_manifest.json" or not path.is_file():
            continue
        generated[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema_name": "ictc-2026-etri-paper-evidence-manifest-v1",
        "source_files": source_files,
        "generated_files": generated,
        "external_validation_opened": False,
        "notes": [
            "The downstream root manifest transitively binds the large Arrow and Parquet artifacts.",
            "This package performs descriptive extraction only and does not refit models or rerun masks.",
        ],
    }
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output),
                "generated_file_count": len(generated) + 1,
                "diagnostic_label": diagnostic_decision.loc[0, "label"],
                "downstream_label": downstream_decision.loc[0, "label"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
