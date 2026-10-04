#!/usr/bin/env python3
"""Regenerate ten paper reports from fresh independent ETRI stage outputs.

Eight CSV calculations are source-derived without alteration from the immutable
vendor/extract_etri_paper_evidence.py:176-304. Historical loader assumptions,
release identities and hardcoded result prose are replaced by current lineage,
measured counts and data-derived narrative. Reference files are comparison only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from private_reference import REFERENCE
import shutil
import time
import traceback

import pandas as pd

from full_prepare import PAPER, VENDOR, create_fresh_output, emit, sha256, verify_vendor, write_json
from full_stress import G2_TABLES, read_upstream_manifest, validate_selected_topology

STRESS_TABLES = ("decision", "family_inference", "denominators", "dose_response_family",
                 "participant_deltas", "primary_comparison", "mask_ledger")
DOWNSTREAM_TABLES = ("decision", "condition_group_descriptives", "participant_summaries",
                     "global_condition_descriptives")
CSV_NAMES = ("feature_contracts.csv", "primary_feature_inference.csv", "participant_primary_deltas.csv",
             "dose_response.csv", "downstream_condition_summary.csv", "downstream_feature_profiles.csv",
             "downstream_participant_primary.csv", "model_specifications.csv")


def load_reporting_vendor(filename: str):
    """Load immutable reporting functions/constants without executing legacy CLI."""
    records = json.loads((PAPER / "vendor_manifest.json").read_text(encoding="utf-8-sig"))["files"]
    relative = "vendor/" + filename
    expected = next(record["sha256"] for record in records if record["path"] == relative)
    path = PAPER / relative
    if sha256(path) != expected:
        raise ValueError(f"reporting vendor changed: {filename}")
    spec = importlib.util.spec_from_file_location("independent_reporting_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load_reporting_vendor("extract_etri_paper_evidence.py")
FAMILY_ORDER = original.FAMILY_ORDER
CONDITION_ORDER = original.CONDITION_ORDER
FEATURE_CONTRACTS = original.FEATURE_CONTRACTS
MODEL_PARAMETERS = original.MODEL_PARAMETERS
ordered = original.ordered


def validate_lineage(manifests: dict, hashes: dict) -> None:
    for stage in ("prepare", "g2", "stress", "downstream"):
        manifest = manifests[stage]
        if (manifest.get("schema_name") != "independent_etri_reproduction"
                or manifest.get("stage") != stage or manifest.get("status") != "complete"
                or manifest.get("fresh_execution") is not True):
            raise ValueError(f"evidence requires a fresh complete {stage} stage")
    for stage, ancestors in (("g2", ("prepare",)), ("stress", ("prepare", "g2")),
                             ("downstream", ("prepare", "g2", "stress"))):
        for ancestor in ancestors:
            if manifests[stage].get("inputs", {}).get(ancestor + "_manifest_sha256") != hashes[ancestor]:
                raise ValueError(f"{stage} and {ancestor} manifests are crosswired")
    downstream = manifests["downstream"]
    if (downstream.get("new_model_fits") != 160
            or downstream.get("reused_historical_model_fits") != 0):
        raise ValueError("full paper evidence requires 160 NEW model fits and no historical reuse")
    rows = downstream.get("prediction_rows", downstream.get("observed_row_count"))
    primary = downstream.get("expected_primary_key_count", downstream.get("adapter", {}).get("expected_primary_keys"))
    if type(rows) is not int or type(primary) is not int or primary <= 0 or rows != 3 * primary:
        raise ValueError("fresh downstream prediction topology is incomplete")


def build_reporting_tables(stress: dict[str, pd.DataFrame], downstream: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Original extractor:176-304, with new DataFrame input boundary only."""
    family_inference = stress["family_inference"]
    denominators = stress["denominators"]
    dose_response = stress["dose_response_family"]
    participant_deltas = stress["participant_deltas"]
    primary = stress["primary_comparison"]
    condition_groups = downstream["condition_group_descriptives"]
    participant_summaries = downstream["participant_summaries"]
    global_conditions = downstream["global_condition_descriptives"]
    selected_cells = primary.loc[primary["scenario"] == "contiguous_20pct"].drop_duplicates(
        ["subject_id", "sensor_day_id", "primitive"])
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
    return output_files


def build_study_design(stress: dict, downstream: dict, downstream_manifest: dict) -> dict:
    primary = stress["primary_comparison"]
    selected = primary.loc[primary.scenario.eq("contiguous_20pct")].drop_duplicates(
        ["subject_id", "sensor_day_id", "primitive"])
    mask_ledger = stress["mask_ledger"]
    mask_counts = mask_ledger.groupby(["scenario", "mask_status"]).size()
    participants = int(primary.subject_id.nunique())
    observed = int(downstream_manifest.get("prediction_rows", downstream_manifest.get("observed_row_count")))
    expected = int(downstream_manifest.get("expected_primary_key_count",
                    downstream_manifest.get("adapter", {}).get("expected_primary_keys")))
    return {
        "schema_name": "ictc-2026-etri-paper-evidence-independent-v1",
        "provenance": "fresh independent canonical-to-model reproduction",
        "diagnostic": {
            "participants": participants, "primitives": int(primary.primitive.nunique()),
            "selected_participant_day_primitive_cells": int(len(selected)),
            "selected_days_per_participant_and_primitive": 10, "root_seed": 42,
            "draws_per_scenario": int(primary.draw.nunique()),
            "primary_conditions": ["contiguous_20pct", "scattered_random_20pct"],
            "secondary_event_condition": "event_boundary_20pct",
            "dose_conditions": ["contiguous_10pct", "contiguous_20pct", "contiguous_40pct"],
            "new_mask_rows": int(len(mask_ledger)),
            "mask_status_counts": {f"{scenario}:{status}": int(count)
                                   for (scenario, status), count in mask_counts.items()},
            "inferential_unit": "participant",
            "primary_estimand": "participant median of standardized absolute error under contiguous 20% "
                                "minus matched scattered-random 20%",
            "bootstrap_resamples": 10000, "primary_test": "exact two-sided participant sign-flip",
            "multiplicity": "Holm adjustment over four fixed feature families",
            "decision_label": str(stress["decision"].loc[0, "label"]),
        },
        "downstream": {
            "participants": participants, "split": "leave-one-subject-out",
            "arms": ["matched_fixed_personalized", "semantic_personalized"],
            "models": ["elasticnet", "lightgbm"], "targets": ["S1", "S2", "S3", "S4"],
            "fit_count": int(downstream_manifest["new_model_fits"]),
            "new_model_fits": int(downstream_manifest["new_model_fits"]),
            "reused_historical_model_fits": int(downstream_manifest["reused_historical_model_fits"]),
            "fit_identity_formula": "10 folds x 2 arms x 2 models x 4 targets",
            "fit_policy": "fit once on unmasked outer-training rows; reuse across conditions",
            "reporting_threshold": 0.5, "conditions": CONDITION_ORDER,
            "observed_prediction_rows": observed, "expected_primary_keys": expected,
            "condition_group_rows": int(len(downstream["condition_group_descriptives"])),
            "group_cells": int(downstream["participant_summaries"].group_cell_count.sum()),
            "participant_summaries": int(len(downstream["participant_summaries"])),
            "bootstrap_resamples": 10000,
            "sign_flip_patterns": int(downstream["decision"].loc[0, "sign_denominator"]),
            "decision_label": str(downstream["decision"].loc[0, "label"]),
        },
    }


def render_manuscript_evidence(tables: dict, study: dict, downstream_decision: pd.DataFrame) -> str:
    """Every result number/label comes from this run, including null outcomes."""
    diag, model = study["diagnostic"], study["downstream"]
    decision = downstream_decision.iloc[0]
    lines = ["# ETRI manuscript evidence from a fresh independent run", "",
        "This report uses newly generated primitive, G2, matched stress and downstream outputs. "
        "Historical records are comparison references only.", "", "## Methods and observed execution", "",
        f"- {diag['participants']} participants, {diag['primitives']} primitives, "
        f"{diag['selected_participant_day_primitive_cells']} selected participant-day-primitive cells; "
        f"{diag['draws_per_scenario']} draws per scenario, root seed {diag['root_seed']}.",
        "- The primary comparison holds valid deleted-record counts fixed between contiguous 20% "
        "and scattered-random 20% loss. Event-boundary loss is a secondary condition.",
        "- Observed-support primitive values are the reference; natural missingness remains part "
        "of the provenance and the reference is not unobserved ground truth.",
        "- Participant medians are the inferential observations: 10,000 deterministic bootstrap "
        "resamples, exact two-sided sign-flip tests and four-family Holm adjustment.",
        f"- {model['new_model_fits']} NEW fits; {model['reused_historical_model_fits']} historical fits reused. "
        f"{model['observed_prediction_rows']:,} predictions from {model['expected_primary_keys']:,} primary keys.",
        "- Ten LOSO folds, two feature arms, ElasticNet logistic regression and LightGBM, "
        "four binary targets S1–S4. Each fitted model is reused across held-out missingness conditions.",
        "", "## Diagnostic results", "", f"Decision label: `{diag['decision_label']}`.", ""]
    for row in tables["primary_feature_inference.csv"].itertuples(index=False):
        lines.append(f"- {row.feature_family}: median {row.participant_median:.6f}, "
            f"95% bootstrap interval [{row.ci_low:.6f}, {row.ci_high:.6f}], "
            f"{int(row.direction_agreement_count)}/{int(row.eligible_participant_count)} direction agreements, "
            f"Holm p={row.holm_p_value:.7g}; {row.direction_interpretation}; {row.confirmatory_interpretation}.")
    for family in FAMILY_ORDER:
        dose = tables["dose_response.csv"].loc[lambda frame: frame.feature_family.eq(family)].sort_values("missingness_rate")
        values = dose.participant_median.tolist()
        rates = ", ".join(f"{float(rate):.0%}: {float(value):.6f}" for rate, value in
                          zip(dose.missingness_rate, values, strict=True))
        monotonic = len(values) == 3 and all(pd.notna(value) for value in values) and all(
            left <= right for left, right in zip(values, values[1:]))
        lines.append(f"- {family} dose response ({rates}); nondecreasing={str(monotonic).lower()}.")
    lines += ["", "## Downstream results", "", f"Decision label: `{model['decision_label']}`.", "",
        f"Participant-first statistic {decision.t_observed:.6f}; bootstrap sensitivity interval "
        f"[{decision.bootstrap_lower:.6f}, {decision.bootstrap_upper:.6f}]; "
        f"{int(decision.direction_positive_count)}/{model['participants']} positive participant directions; "
        f"exact sign-flip p={decision.sign_p_value:.7g}.", ""]
    for row in tables["downstream_condition_summary.csv"].itertuples(index=False):
        lines.append(f"- {row.condition}: participant-first median absolute probability change "
            f"{row.participant_first_median_abs_probability_change:.6f}; flips "
            f"{int(row.flipped_count):,}/{int(row.applicable_count):,} applicable predictions "
            f"({100 * row.flip_fraction_applicable:.3f}%); {int(row.abstained_count):,} abstained; "
            f"{int(row.structurally_na_count):,} structurally unavailable.")
    for row in tables["downstream_feature_profiles.csv"].itertuples(index=False):
        lines.append(f"- {row.feature_family}: descriptive contiguous-minus-random probability-change "
                     f"contrast {row.contiguous_minus_random:+.6f}.")
    lines += ["", "## Interpretation limits", "",
        "The frozen diagnostic and downstream decision labels above determine the study conclusions. "
        "Descriptive contrasts do not demonstrate predictive improvement, reconstruction of missing "
        "observations, or recovery of unobserved ground truth. This ETRI report does not infer the "
        "execution status of separately analyzed external datasets.", "", "## Display mapping", "",
        "- Table I: feature_contracts.csv and study_design.json.",
        "- Figure 2a: participant_primary_deltas.csv and primary_feature_inference.csv.",
        "- Figure 2b: dose_response.csv.",
        "- Figure 2c: downstream_condition_summary.csv; family profiles remain available separately.", ""]
    return "\n".join(lines)


def _output_record(path: Path, root: Path, frame: pd.DataFrame | None = None) -> dict:
    record = {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
    if frame is not None:
        record.update(rows=len(frame), columns=list(frame.columns))
    return record


def _verify_output_pins(root: Path, manifest: dict, names: tuple[str, ...]) -> None:
    for name in names:
        record = manifest.get("outputs", {}).get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"missing fresh artifact pin: {name}")
        relative = Path(record["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file() or sha256(path) != record.get("sha256"):
            raise ValueError(f"fresh artifact missing, escaped or changed: {name}")


def run_evidence(prepare_dir: Path, g2_dir: Path, stress_dir: Path,
                 downstream_dir: Path, output_dir: Path) -> dict:
    output = create_fresh_output(output_dir, [prepare_dir, g2_dir, stress_dir, downstream_dir, REFERENCE, VENDOR])
    started = time.perf_counter()
    manifest = {"schema_name": "independent_etri_reproduction", "stage": "evidence",
                "status": "running", "fresh_execution": True, "new_model_fits": 0,
                "outputs": {}, "provenance_note": "Reports are calculated from the linked fresh stages; "
                "the evidence stage does not perform additional fits or mask replays."}
    write_json(output / "manifest.json", manifest)
    try:
        manifest["scientific_sources"] = verify_vendor()
        load_reporting_vendor("extract_etri_paper_evidence.py")
        directories = {name: path.resolve(strict=True) for name, path in
            (("prepare", prepare_dir), ("g2", g2_dir), ("stress", stress_dir), ("downstream", downstream_dir))}
        required = {"prepare": ("primitives", "baselines", "representations"), "g2": G2_TABLES,
                    "stress": STRESS_TABLES,
                    "downstream": tuple(f"run/statistics/{name}.parquet" for name in DOWNSTREAM_TABLES)}
        manifests = {stage: read_upstream_manifest(root, stage, required[stage]) for stage, root in directories.items()}
        hashes = {stage: sha256(root / "manifest.json") for stage, root in directories.items()}
        validate_lineage(manifests, hashes)
        downstream_manifest = manifests["downstream"]
        _verify_output_pins(directories["downstream"], downstream_manifest,
                           ("run/observed_predictions.arrow", "run/fit/fit_ledger.arrow", "run/fit/fit_states.pack"))
        manifest["inputs"] = {stage: {"path": str(directories[stage]), "manifest_sha256": hashes[stage],
                                      "outputs": manifests[stage]["outputs"]} for stage in directories}
        manifest["adapter_sources"] = [_output_record(path, PAPER) for path in
            (Path(__file__).resolve(), PAPER / "vendor/extract_etri_paper_evidence.py")]
        stress = {name: pd.read_parquet(directories["stress"] / manifests["stress"]["outputs"][name]["path"])
                  for name in STRESS_TABLES}
        downstream = {name: pd.read_parquet(directories["downstream"] /
            downstream_manifest["outputs"][f"run/statistics/{name}.parquet"]["path"]) for name in DOWNSTREAM_TABLES}
        selected = stress["primary_comparison"].loc[lambda frame: frame.scenario.eq("contiguous_20pct")].drop_duplicates(
            ["subject_id", "sensor_day_id", "primitive"])
        keys = validate_selected_topology(selected)
        g2_selected = pd.read_parquet(directories["g2"] / manifests["g2"]["outputs"]["selected_days"]["path"])
        if keys != validate_selected_topology(g2_selected):
            raise ValueError("stress and G2 selected cells are crosswired")
        if (len(stress["primary_comparison"]) != 80000 or len(stress["mask_ledger"]) != 80000
                or len(stress["participant_deltas"]) != 40 or len(stress["dose_response_family"]) != 12
                or len(stress["family_inference"]) != 4
                or len(downstream["condition_group_descriptives"]) != 1920
                or len(downstream["participant_summaries"]) != 10
                or len(downstream["decision"]) != 1
                or not bool(downstream["decision"].loc[0, "integrity_complete"])):
            raise ValueError("fresh report inputs do not have the complete scientific topology")
        tables = build_reporting_tables(stress, downstream)
        for name, frame in tables.items():
            path = output / name
            original.write_csv(frame, path)
            manifest["outputs"][name] = _output_record(path, output, frame)
        study = build_study_design(stress, downstream, downstream_manifest)
        write_json(output / "study_design.json", study)
        (output / "manuscript_evidence.md").write_text(
            render_manuscript_evidence(tables, study, downstream["decision"]), encoding="utf-8")
        for name in ("study_design.json", "manuscript_evidence.md"):
            manifest["outputs"][name] = _output_record(output / name, output)
        for stage, root in directories.items():
            if sha256(root / "manifest.json") != hashes[stage]:
                raise ValueError(f"{stage} changed while producing evidence")
            read_upstream_manifest(root, stage, required[stage])
        verify_vendor()
        load_reporting_vendor("extract_etri_paper_evidence.py")
        manifest.update(status="complete", seconds=time.perf_counter() - started,
                        generated_scientific_report_files=len(manifest["outputs"]), upstream_new_model_fits=160)
        write_json(output / "manifest.json", manifest)
        write_json(output / "evidence_manifest.json", manifest)
        emit(stage="evidence", status="complete", files=len(manifest["outputs"]))
        return manifest
    except BaseException as error:
        manifest.update(status="failed", seconds=time.perf_counter() - started,
                        error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", manifest)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def validate_display_annotations(condition_summary: pd.DataFrame) -> None:
    rows = condition_summary.loc[condition_summary.condition.eq("event_boundary_20pct")]
    if len(rows) != 1 or (int(rows.iloc[0].abstained_count), int(rows.iloc[0].structurally_na_count)) != (1104, 8000):
        raise ValueError("unchanged renderer requires verified boundary annotations: 1104 abstained, 8000 unavailable")


def run_displays(evidence_dir: Path, external_results_dir: Path, output_dir: Path) -> dict:
    """Render fresh evidence only after checking every fixed scientific annotation."""
    output = create_fresh_output(output_dir, [evidence_dir, external_results_dir, REFERENCE, VENDOR])
    started = time.perf_counter()
    manifest = {"schema_name": "independent_etri_reproduction", "stage": "displays",
                "status": "running", "fresh_execution": True, "new_model_fits": 0, "outputs": {}}
    write_json(output / "manifest.json", manifest)
    try:
        evidence_dir, external_results_dir = evidence_dir.resolve(strict=True), external_results_dir.resolve(strict=True)
        evidence_manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8-sig"))
        if (evidence_manifest.get("schema_name") != "independent_etri_reproduction"
                or evidence_manifest.get("stage") != "evidence" or evidence_manifest.get("status") != "complete"
                or evidence_manifest.get("fresh_execution") is not True
                or evidence_manifest.get("upstream_new_model_fits") != 160):
            raise ValueError("displays require complete evidence bound to 160 fresh upstream fits")
        _verify_output_pins(evidence_dir, evidence_manifest, CSV_NAMES)
        comparisons = {}
        for name in CSV_NAMES:
            actual, reference = pd.read_csv(evidence_dir / name), pd.read_csv(REFERENCE / 'etri' / name)
            pd.testing.assert_frame_equal(actual, reference, check_exact=True)
            comparisons[name] = {"exact_csv_values": True, "reference_role": "comparison_only"}
        validate_display_annotations(pd.read_csv(evidence_dir / "downstream_condition_summary.csv"))
        external_names = ("external_transfer_summary.csv", "external_participant_deltas.csv", "external_denominators.csv")
        for name in external_names:
            pd.testing.assert_frame_equal(pd.read_csv(external_results_dir / name),
                pd.read_csv(REFERENCE / 'external' / name), check_exact=True)
        external_manifest_path = external_results_dir / "external_transfer_manifest.json"
        external_manifest = json.loads(external_manifest_path.read_text(encoding="utf-8-sig"))
        if (external_manifest.get("external_labels_opened") is not False
                or external_manifest.get("decision") != {"label": "full_transfer", "passed_primary_pairs": 4, "total_primary_pairs": 4}):
            raise ValueError("external display source must have the separately verified four-pair label-free result")
        cache_dir = output / ".matplotlib"
        cache_dir.mkdir()
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
        renderer = load_reporting_vendor("generate_final_displays.py")
        stage = output / "display_inputs/paper_ictc2026/02_experiments"
        destination = stage / "etri_paper_evidence"
        destination.mkdir(parents=True)
        for name in CSV_NAMES:
            shutil.copyfile(evidence_dir / name, destination / name)
        destination = stage / "external_transfer/results"
        destination.mkdir(parents=True)
        for name in (*external_names, "external_transfer_manifest.json"):
            shutil.copyfile(external_results_dir / name, destination / name)
        display_manifest = renderer.build(output / "display_inputs", output / "figures", output / "tables")
        manifest["inputs"] = {"evidence_dir": str(evidence_dir), "evidence_manifest_sha256": sha256(evidence_dir / "manifest.json"),
            "external_results_dir": str(external_results_dir), "external_manifest_sha256": sha256(external_manifest_path)}
        manifest["scientific_csv_comparisons"] = comparisons
        manifest["verified_fixed_annotations"] = {"event_boundary_abstained": 1104, "event_boundary_structurally_unavailable": 8000}
        manifest["original_renderer_result"] = display_manifest
        for directory, name in (("figures", "fig2_results.pdf"), ("figures", "fig2_results.png"),
                                ("tables", "table1_feature_contracts.tex"), ("tables", "table2_external_transfer.tex")):
            manifest["outputs"][name] = _output_record(output / directory / name, output)
        _verify_output_pins(evidence_dir, evidence_manifest, CSV_NAMES)
        manifest.update(status="complete", seconds=time.perf_counter() - started)
        write_json(output / "manifest.json", manifest)
        write_json(output / "display_manifest.json", manifest)
        emit(stage="displays", status="complete", files=len(manifest["outputs"]))
        return manifest
    except BaseException as error:
        manifest.update(status="failed", seconds=time.perf_counter() - started,
                        error_type=type(error).__name__, error=str(error))
        write_json(output / "manifest.json", manifest)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evidence = sub.add_parser("evidence")
    for name in ("prepare", "g2", "stress", "downstream"):
        evidence.add_argument(f"--{name}-dir", type=Path, default=Path(f"/data/output/paper/{name}"))
    evidence.add_argument("--output-dir", type=Path, default=Path("/data/output/paper/evidence"))
    displays = sub.add_parser("displays")
    displays.add_argument("--evidence-dir", type=Path, default=Path("/data/output/paper/evidence"))
    displays.add_argument("--external-results-dir", type=Path, default=Path("/data/output/paper/external/raw_results"))
    displays.add_argument("--output-dir", type=Path, default=Path("/data/output/paper/displays"))
    args = parser.parse_args()
    if args.command == "evidence":
        run_evidence(args.prepare_dir, args.g2_dir, args.stress_dir, args.downstream_dir, args.output_dir)
    else:
        run_displays(args.evidence_dir, args.external_results_dir, args.output_dir)


if __name__ == "__main__":
    main()
