from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DATASETS = ("StudentLife", "ExtraSensory")
FAMILIES = ("event_count", "state_ratio", "intensity", "timing")
PRIMARY_FAMILIES = ("state_ratio", "intensity")
DELETION_FRACTION = 0.20
DRAWS = 50
ROOT_SEED = 42
MAX_DAYS = 10
MIN_DAYS = 5
BOOTSTRAP_RESAMPLES = 10_000


def validate_settings(*, deletion_fraction: float, draws: int) -> None:
    if type(deletion_fraction) is not float or deletion_fraction != DELETION_FRACTION:
        raise ValueError("deletion_fraction must be exact 0.20")
    if type(draws) is not int or draws != DRAWS:
        raise ValueError("draws must be exact 50")


def _seed(text: str) -> int:
    digest = hashlib.sha256(f"external-transfer-v1|{ROOT_SEED}|{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def matched_mask_indices(
    *, record_count: int, deletion_fraction: float, seed_material: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    validate_settings(deletion_fraction=deletion_fraction, draws=DRAWS)
    if type(record_count) is not int or record_count < 2:
        raise ValueError("record_count must be an exact integer >= 2")
    k = max(1, int(math.floor(record_count * deletion_fraction + 0.5)))
    k = min(k, record_count - 1)
    rng_random = random.Random(_seed(seed_material + "|random"))
    rng_contiguous = random.Random(_seed(seed_material + "|contiguous"))
    scattered = tuple(sorted(rng_random.sample(range(record_count), k)))
    start = rng_contiguous.randrange(record_count - k + 1)
    contiguous = tuple(range(start, start + k))
    return scattered, contiguous


def compute_feature(family: str, records: Sequence[tuple[float, float]]) -> float:
    if family not in FAMILIES:
        raise ValueError(f"unknown feature family: {family}")
    clean = [(float(t), float(v)) for t, v in records if math.isfinite(t) and math.isfinite(v)]
    if family == "event_count":
        return float(sum(v for _, v in clean if v > 0))
    if not clean:
        return math.nan
    if family == "state_ratio":
        return float(statistics.fmean(v for _, v in clean))
    if family == "intensity":
        return float(statistics.fmean(v for _, v in clean))
    positive = sorted((t, v) for t, v in clean if v > 0)
    if len(positive) < 3:
        return math.nan
    total = sum(v for _, v in positive)
    if total <= 0:
        return math.nan
    threshold = 0.90 * total
    cumulative = 0.0
    for timestamp, weight in positive:
        cumulative += weight
        if cumulative >= threshold:
            return float(timestamp)
    return float(positive[-1][0])


def compute_masked_feature(
    family: str,
    records: Sequence[tuple[float, float]],
    deleted_indices: Sequence[int],
) -> float:
    if family not in FAMILIES:
        raise ValueError(f"unknown feature family: {family}")
    deleted = np.asarray(deleted_indices, dtype=np.int64)
    if family in ("event_count", "state_ratio", "intensity"):
        values = np.fromiter((float(value) for _, value in records), dtype=np.float64)
        finite = np.isfinite(values)
        retained = finite.copy()
        retained[deleted] = False
        kept = values[retained]
        if family == "event_count":
            return float(kept[kept > 0].sum())
        if kept.size == 0:
            return math.nan
        return float(kept.mean())
    deleted_set = set(int(index) for index in deleted_indices)
    retained_records = [record for index, record in enumerate(records) if index not in deleted_set]
    return compute_feature(family, retained_records)


def exact_two_sided_sign_p(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v)) and float(v) != 0.0]
    n = len(clean)
    if n == 0:
        return 1.0
    positive = sum(v > 0 for v in clean)
    tail = min(positive, n - positive)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, probability))


def participant_first_summary(rows: Sequence[dict]) -> tuple[list[dict], dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        delta = float(row["delta"])
        if math.isfinite(delta):
            grouped[str(row["participant"])].append(delta)
    participants = [
        {"participant": participant, "delta": float(statistics.median(values))}
        for participant, values in sorted(grouped.items())
        if values
    ]
    deltas = [row["delta"] for row in participants]
    summary = {
        "eligible_participants": len(deltas),
        "positive_participants": sum(v > 0 for v in deltas),
        "zero_participants": sum(v == 0 for v in deltas),
        "negative_participants": sum(v < 0 for v in deltas),
        "participant_median": float(statistics.median(deltas)) if deltas else math.nan,
        "raw_sign_p": exact_two_sided_sign_p(deltas),
    }
    return participants, summary


def classify_transfer(rows: Sequence[dict]) -> dict:
    keys = {(str(r["dataset"]), str(r["feature_family"])): r for r in rows}
    expected = {(d, f) for d in DATASETS for f in PRIMARY_FAMILIES}
    if set(keys) != expected:
        raise ValueError("transfer classification requires exactly four primary pairs")
    passed = 0
    for key in sorted(expected):
        row = keys[key]
        if float(row["participant_median"]) > 0 and float(row["positive_fraction"]) >= 0.60:
            passed += 1
    if passed == 4:
        label = "full_transfer"
    elif passed >= 2:
        label = "partial_transfer"
    else:
        label = "transfer_not_demonstrated"
    return {"label": label, "passed_primary_pairs": passed, "total_primary_pairs": 4}


def verify_readiness(project_root: Path) -> dict:
    verifier = project_root / "02_experiments" / "external_data" / "readiness" / "verify_external_data.py"
    if not verifier.is_file():
        raise RuntimeError("readiness verifier is missing")
    completed = subprocess.run(
        [sys.executable, str(verifier), "--project-root", str(project_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    if result != {"datasets": 2, "files": 2044, "status": "READY"}:
        raise RuntimeError(f"unexpected readiness result: {result}")
    return result


def _utc_day_and_second(timestamp: float) -> tuple[int, float]:
    day = math.floor(timestamp / 86400.0)
    return int(day), float(timestamp - day * 86400.0)


def _read_csv_clean(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skip_blank_lines=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _studentlife_participants(dataset_root: Path) -> list[str]:
    activity = dataset_root / "sensing" / "activity"
    return sorted(p.stem.removeprefix("activity_") for p in activity.glob("activity_u*.csv"))


def load_studentlife_participant(dataset_root: Path, participant: str) -> dict[str, dict[int, list[tuple[float, float]]]]:
    result = {family: defaultdict(list) for family in FAMILIES}
    activity_path = dataset_root / "sensing" / "activity" / f"activity_{participant}.csv"
    conversation_path = dataset_root / "sensing" / "conversation" / f"conversation_{participant}.csv"
    if activity_path.is_file():
        frame = _read_csv_clean(activity_path)
        timestamp_col, value_col = "timestamp", "activity inference"
        for timestamp, value in frame[[timestamp_col, value_col]].itertuples(index=False, name=None):
            try:
                t, v = float(timestamp), float(value)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(t) and math.isfinite(v) and v in (0.0, 1.0, 2.0)):
                continue
            day, second = _utc_day_and_second(t)
            result["state_ratio"][day].append((second, 1.0 if v > 0 else 0.0))
            result["intensity"][day].append((second, v))
    if conversation_path.is_file():
        frame = _read_csv_clean(conversation_path)
        for start, end in frame[["start_timestamp", "end_timestamp"]].itertuples(index=False, name=None):
            try:
                start_f, end_f = float(start), float(end)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(start_f) and math.isfinite(end_f) and end_f >= start_f):
                continue
            day, second = _utc_day_and_second(start_f)
            result["event_count"][day].append((second, 1.0))
            result["timing"][day].append((second, 1.0))
    return {family: {day: sorted(records) for day, records in days.items()} for family, days in result.items()}


EXTRA_USECOLS = (
    "timestamp",
    "discrete:app_state:is_active",
    "discrete:app_state:is_inactive",
    "discrete:app_state:is_background",
    "discrete:app_state:missing",
    "raw_acc:magnitude_stats:mean",
)


def load_extrasensory_participant(path: Path) -> dict[str, dict[int, list[tuple[float, float]]]]:
    frame = pd.read_csv(path, compression="gzip", usecols=list(EXTRA_USECOLS))
    frame = frame.loc[:, list(EXTRA_USECOLS)]
    result = {family: defaultdict(list) for family in FAMILIES}
    states_by_day: dict[int, list[tuple[float, float]]] = defaultdict(list)
    intensity_by_day: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in frame.itertuples(index=False, name=None):
        timestamp = row[0]
        try:
            t = float(timestamp)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(t):
            continue
        day, second = _utc_day_and_second(t)
        active, inactive, background, missing, acceleration = row[1:]
        known_values = [active, inactive, background]
        missing_flag = 0.0 if pd.isna(missing) else float(missing)
        if missing_flag != 1.0 and any(not pd.isna(v) and float(v) == 1.0 for v in known_values):
            state = 1.0 if not pd.isna(active) and float(active) == 1.0 else 0.0
            states_by_day[day].append((second, state))
        if not pd.isna(acceleration):
            value = float(acceleration)
            if math.isfinite(value) and value >= 0:
                intensity_by_day[day].append((second, value))
    for day, states in states_by_day.items():
        ordered = sorted(states)
        result["state_ratio"][day] = ordered
        previous = None
        for second, state in ordered:
            if state == 1.0 and previous == 0.0:
                result["event_count"][day].append((second, 1.0))
                result["timing"][day].append((second, 1.0))
            previous = state
    for day, values in intensity_by_day.items():
        result["intensity"][day] = sorted(values)
    return {family: dict(days) for family, days in result.items()}


def _minimum_records(family: str) -> int:
    return 3 if family in ("event_count", "timing") else 20


def _select_days(dataset: str, participant: str, family: str, days: dict[int, list]) -> list[int]:
    eligible = []
    for day, records in days.items():
        if len(records) < _minimum_records(family):
            continue
        if math.isfinite(compute_feature(family, records)):
            eligible.append(day)
    eligible.sort(key=lambda day: hashlib.sha256(f"day-selection-v1|42|{dataset}|{participant}|{family}|{day}".encode()).hexdigest())
    return sorted(eligible[:MAX_DAYS])


def _robust_scale(values: Sequence[float]) -> float:
    array = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return 1.0
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median))) * 1.4826
    q25, q75 = np.percentile(array, [25, 75])
    iqr_scale = float(q75 - q25) / 1.349
    fallback = abs(median) * 0.10
    for candidate in (mad, iqr_scale, fallback, 1.0):
        if math.isfinite(candidate) and candidate > 1e-12:
            return candidate
    return 1.0


def analyze_participant_family(
    *, dataset: str, participant: str, family: str, days: dict[int, list[tuple[float, float]]]
) -> tuple[list[dict], dict]:
    selected = _select_days(dataset, participant, family, days)
    expected = len(selected) * DRAWS
    if len(selected) < MIN_DAYS:
        return [], {"selected_days": len(selected), "expected_draws": expected, "finite_draws": 0, "status": "ineligible_lt_5_days"}
    references = {day: compute_feature(family, days[day]) for day in selected}
    scale = _robust_scale(list(references.values()))
    rows = []
    abstained = 0
    for day in selected:
        records = days[day]
        reference = references[day]
        prepared_values = np.fromiter((float(value) for _, value in records), dtype=np.float64)
        prepared_total = float(prepared_values.sum())
        for draw in range(DRAWS):
            scattered, contiguous = matched_mask_indices(
                record_count=len(records),
                deletion_fraction=DELETION_FRACTION,
                seed_material=f"{dataset}|{participant}|{family}|{day}|{draw}",
            )
            if family in ("state_ratio", "intensity"):
                retained_count = len(records) - len(scattered)
                random_feature = (prepared_total - float(prepared_values[list(scattered)].sum())) / retained_count
                contiguous_feature = (prepared_total - float(prepared_values[list(contiguous)].sum())) / retained_count
            elif family == "event_count":
                positive_values = np.maximum(prepared_values, 0.0)
                positive_total = float(positive_values.sum())
                random_feature = positive_total - float(positive_values[list(scattered)].sum())
                contiguous_feature = positive_total - float(positive_values[list(contiguous)].sum())
            else:
                random_feature = compute_masked_feature(family, records, scattered)
                contiguous_feature = compute_masked_feature(family, records, contiguous)
            if not (math.isfinite(random_feature) and math.isfinite(contiguous_feature)):
                abstained += 1
                continue
            random_error = abs(random_feature - reference) / scale
            contiguous_error = abs(contiguous_feature - reference) / scale
            rows.append(
                {
                    "participant": participant,
                    "day": day,
                    "draw": draw,
                    "delta": contiguous_error - random_error,
                    "random_error": random_error,
                    "contiguous_error": contiguous_error,
                    "deleted_count": len(scattered),
                }
            )
    status = "eligible" if len(rows) >= math.ceil(expected * 0.80) else "ineligible_lt_80pct_finite"
    if status != "eligible":
        rows = []
    return rows, {
        "selected_days": len(selected),
        "expected_draws": expected,
        "finite_draws": len(rows),
        "abstained_draws": abstained,
        "status": status,
        "scale": scale,
    }


def _bootstrap_median_ci(values: Sequence[float], seed_material: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(_seed(seed_material))
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size))
    medians = np.median(array[indices], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def _holm(rows: list[dict]) -> None:
    ordered = sorted(range(len(rows)), key=lambda i: float(rows[i]["raw_sign_p"]))
    running = 0.0
    m = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * float(rows[index]["raw_sign_p"]))
        running = max(running, adjusted)
        rows[index]["holm_p"] = running


def analyze_dataset(project_root: Path, dataset: str) -> tuple[list[dict], list[dict], list[dict]]:
    if dataset == "StudentLife":
        root = project_root / "StudentLife" / "dataset"
        participants = _studentlife_participants(root)
        loader = lambda participant: load_studentlife_participant(root, participant)
    elif dataset == "ExtraSensory":
        root = project_root / "ExtraSensory.per_uuid_features_labels"
        paths = sorted(root.glob("*.features_labels.csv.gz"))
        participants = [path.name.split(".features_labels.csv.gz")[0] for path in paths]
        path_by_participant = dict(zip(participants, paths))
        loader = lambda participant: load_extrasensory_participant(path_by_participant[participant])
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    participant_rows: list[dict] = []
    denominator_rows: list[dict] = []
    draw_rows_by_family: dict[str, list[dict]] = defaultdict(list)
    for position, participant in enumerate(participants, 1):
        family_days = loader(participant)
        for family in FAMILIES:
            rows, denominator = analyze_participant_family(
                dataset=dataset,
                participant=participant,
                family=family,
                days=family_days.get(family, {}),
            )
            denominator_rows.append({"dataset": dataset, "participant": participant, "feature_family": family, **denominator})
            draw_rows_by_family[family].extend(rows)
        print(f"PROGRESS {dataset} {position}/{len(participants)}", flush=True)
    family_summaries: list[dict] = []
    for family in FAMILIES:
        participants_first, summary = participant_first_summary(draw_rows_by_family[family])
        for row in participants_first:
            participant_rows.append({"dataset": dataset, "feature_family": family, **row})
        deltas = [row["delta"] for row in participants_first]
        ci_low, ci_high = _bootstrap_median_ci(deltas, f"bootstrap|{dataset}|{family}")
        count = summary["eligible_participants"]
        family_summaries.append(
            {
                "dataset": dataset,
                "feature_family": family,
                **summary,
                "positive_fraction": summary["positive_participants"] / count if count else math.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    _holm(family_summaries)
    return family_summaries, participant_rows, denominator_rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path.name}")
    fields = list(rows[0])
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(project_root: Path, output_dir: Path) -> dict:
    validate_settings(deletion_fraction=DELETION_FRACTION, draws=DRAWS)
    readiness = verify_readiness(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, participants, denominators = [], [], []
    for dataset in DATASETS:
        s, p, d = analyze_dataset(project_root, dataset)
        summaries.extend(s)
        participants.extend(p)
        denominators.extend(d)
    primary_rows = [row for row in summaries if row["feature_family"] in PRIMARY_FAMILIES]
    decision = classify_transfer(primary_rows)
    _write_csv(output_dir / "external_transfer_summary.csv", summaries)
    _write_csv(output_dir / "external_participant_deltas.csv", participants)
    _write_csv(output_dir / "external_denominators.csv", denominators)
    report = [
        "# External transfer validation report",
        "",
        f"- Decision: `{decision['label']}` ({decision['passed_primary_pairs']}/4 primary dataset-family pairs)",
        f"- Inputs: {readiness['datasets']} datasets and {readiness['files']} pinned files",
        "- Contrast: matched contiguous 20% minus scattered-random 20% standardized absolute error",
        "- Unit: participant-first median; 50 deterministic draws and up to 10 days per participant",
        "- External labels and outcome models: not accessed",
        "",
        "## Dataset-family results",
        "",
    ]
    for row in summaries:
        report.append(
            f"- {row['dataset']} / {row['feature_family']}: median={row['participant_median']:.6g}, "
            f"95% bootstrap [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"positive={row['positive_participants']}/{row['eligible_participants']}, "
            f"exact sign p={row['raw_sign_p']:.6g}, Holm p={row['holm_p']:.6g}."
        )
    report.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This analysis tests directional transfer of feature-distortion geometry only. It does not test predictive performance, external outcome associations, or equality of effect magnitudes across datasets.",
            "",
        ]
    )
    report_path = output_dir / "external_transfer_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    generated = sorted(p for p in output_dir.iterdir() if p.is_file() and p.name != "external_transfer_manifest.json")
    readiness_manifest = project_root / "02_experiments" / "external_data" / "readiness" / "file_manifest.csv"
    manifest = {
        "schema_name": "external-transfer-evidence-v1",
        "decision": decision,
        "settings": {
            "deletion_fraction": DELETION_FRACTION,
            "draws": DRAWS,
            "root_seed": ROOT_SEED,
            "max_days_per_participant": MAX_DAYS,
            "minimum_days_per_participant_family": MIN_DAYS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        "readiness_manifest": {"bytes": readiness_manifest.stat().st_size, "sha256": _sha256(readiness_manifest)},
        "generated_files": {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in generated},
        "external_labels_opened": False,
        "outcome_models_fitted": False,
    }
    manifest_path = output_dir / "external_transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = run(args.project_root.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": "COMPLETE", **manifest["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
