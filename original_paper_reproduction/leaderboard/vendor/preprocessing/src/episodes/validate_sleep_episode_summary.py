"""Validate the train-row-aligned sleep episode summary.

This script does not modify the sleep episode CSV. It writes a quality report
and feature summary so the estimated episode columns can be reviewed before
final integration and global feature cleaning.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_VENV_SITE = PROJECT_ROOT.parents[0] / "ETRI_Human_AI" / ".venv" / "Lib" / "site-packages"
if V1_VENV_SITE.exists():
    sys.path.insert(0, str(V1_VENV_SITE))

import numpy as np
import pandas as pd


TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "data_vvs" / "ch2026_metrics_train.csv"
SLEEP_PATH = PROJECT_ROOT / "data" / "interim" / "train_sleep_episode_summary.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "sleep_episode_cleaning_report.md"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "sleep_episode_feature_summary.csv"
SUBJECT_PATH = PROJECT_ROOT / "outputs" / "tables" / "sleep_episode_quality_by_subject.csv"

KEY_COLS = ["subject_id", "lifelog_date", "sleep_date"]
TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def setup_dirs() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBJECT_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing key columns: {missing}")
    for col in ["lifelog_date", "sleep_date"]:
        df[col] = pd.to_datetime(df[col]).dt.date.astype(str)
    return df


def classify_column(col: str) -> str:
    if col in KEY_COLS:
        return "key"
    if col.endswith("_time") or col.endswith("_start") or col.endswith("_end") or col.startswith("estimated_"):
        return "time_or_estimate"
    if "confidence" in col:
        return "confidence"
    if "fallback" in col or "plausibility" in col or col.startswith("is_"):
        return "quality_flag"
    if "subject_median" in col or "diff_subject" in col:
        return "subject_relative"
    return "episode_feature"


def recommended_action(col: str, missing_ratio: float, nunique: int) -> str:
    if col in KEY_COLS:
        return "keep_key"
    if missing_ratio == 1.0:
        return "drop_candidate_all_nan"
    if missing_ratio > 0.95:
        return "drop_candidate_high_missing"
    if nunique <= 1:
        return "drop_candidate_constant"
    if "confidence" in col:
        return "keep_quality_weight_candidate"
    if "fallback" in col or "plausibility" in col or col.startswith("is_"):
        return "keep_quality_flag_candidate"
    return "keep_candidate"


def build_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        missing_ratio = float(df[col].isna().mean())
        nunique = int(df[col].nunique(dropna=True))
        rows.append(
            {
                "column": col,
                "group": classify_column(col),
                "dtype": str(df[col].dtype),
                "missing_ratio": missing_ratio,
                "nunique": nunique,
                "recommended_action": recommended_action(col, missing_ratio, nunique),
            }
        )
    return pd.DataFrame(rows)


def build_subject_quality(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "estimated_sleep_duration_minutes",
        "sleep_start_confidence",
        "wake_confidence",
        "episode_confidence",
        "estimated_night_awake_event_count",
        "estimated_night_awake_minutes",
        "used_fallback_sleep_start",
        "used_fallback_wake_time",
        "is_short_sleep_episode_lt4h",
        "is_long_sleep_episode_gt12h",
    ]
    rows = []
    for subject_id, part in df.groupby("subject_id", sort=True):
        row = {"subject_id": subject_id, "rows": len(part)}
        for col in num_cols:
            if col in part.columns:
                vals = pd.to_numeric(part[col], errors="coerce")
                row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
                if col.startswith("used_") or col.startswith("is_"):
                    row[f"{col}_sum"] = int(vals.fillna(0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    setup_dirs()
    train = read_table(TRAIN_PATH, "train")
    sleep = read_table(SLEEP_PATH, "sleep_episode")

    train_keys = train[KEY_COLS].copy()
    merged_key_check = train_keys.merge(sleep[KEY_COLS], on=KEY_COLS, how="left", indicator=True)
    matched_rows = int((merged_key_check["_merge"] == "both").sum())
    match_rate = matched_rows / len(train_keys)

    target_overlap = [c for c in TARGET_COLS if c in sleep.columns]
    duplicate_keys = int(sleep.duplicated(KEY_COLS).sum())

    for col in ["estimated_sleep_start", "estimated_wake_time"]:
        if col in sleep.columns:
            sleep[col] = pd.to_datetime(sleep[col], errors="coerce")

    sleep_start = sleep.get("estimated_sleep_start")
    wake_time = sleep.get("estimated_wake_time")
    lifelog_dt = pd.to_datetime(sleep["lifelog_date"])
    sleep_dt = pd.to_datetime(sleep["sleep_date"])

    sleep_start_in_range = (
        (sleep_start >= lifelog_dt + pd.Timedelta(hours=21))
        & (sleep_start <= sleep_dt + pd.Timedelta(hours=4))
        if sleep_start is not None
        else pd.Series(False, index=sleep.index)
    )
    wake_in_range = (
        (wake_time >= sleep_dt + pd.Timedelta(hours=5))
        & (wake_time <= sleep_dt + pd.Timedelta(hours=11))
        if wake_time is not None
        else pd.Series(False, index=sleep.index)
    )

    duration = pd.to_numeric(sleep.get("estimated_sleep_duration_minutes"), errors="coerce")
    short_rows = int(pd.to_numeric(sleep.get("is_short_sleep_episode_lt4h"), errors="coerce").fillna(0).sum())
    long_rows = int(pd.to_numeric(sleep.get("is_long_sleep_episode_gt12h"), errors="coerce").fillna(0).sum())
    sleep_fallback_rows = int(pd.to_numeric(sleep.get("used_fallback_sleep_start"), errors="coerce").fillna(0).sum())
    wake_fallback_rows = int(pd.to_numeric(sleep.get("used_fallback_wake_time"), errors="coerce").fillna(0).sum())

    feature_summary = build_feature_summary(sleep)
    feature_summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    subject_quality = build_subject_quality(sleep)
    subject_quality.to_csv(SUBJECT_PATH, index=False, encoding="utf-8-sig")

    all_nan = int((feature_summary["missing_ratio"] == 1.0).sum())
    high_missing = int((feature_summary["missing_ratio"] > 0.95).sum())
    constant = int((feature_summary["nunique"] <= 1).sum())

    lines = [
        "# Sleep Episode Cleaning Report",
        "",
        "This report validates `data/interim/train_sleep_episode_summary.csv`.",
        "No columns are deleted here; deletion is deferred to global feature cleaning.",
        "",
        "## Key Validation",
        "",
        f"- Sleep rows: `{len(sleep)}`",
        f"- Train rows: `{len(train)}`",
        f"- Matched train keys: `{matched_rows}`",
        f"- Match rate: `{match_rate:.6f}`",
        f"- Duplicate sleep keys: `{duplicate_keys}`",
        f"- Target overlap columns: `{target_overlap}`",
        "",
        "## Episode Boundary Validation",
        "",
        f"- Sleep-start in expected range rows: `{int(sleep_start_in_range.sum())}`",
        f"- Wake-time in expected range rows: `{int(wake_in_range.sum())}`",
        f"- Mean duration minutes: `{duration.mean():.2f}`",
        f"- Median duration minutes: `{duration.median():.2f}`",
        f"- Min duration minutes: `{duration.min():.2f}`",
        f"- Max duration minutes: `{duration.max():.2f}`",
        f"- Short episode rows (<4h): `{short_rows}`",
        f"- Long episode rows (>12h): `{long_rows}`",
        f"- Sleep fallback rows: `{sleep_fallback_rows}`",
        f"- Wake fallback rows: `{wake_fallback_rows}`",
        "",
        "## Feature Quality",
        "",
        f"- Columns: `{sleep.shape[1]}`",
        f"- All-NaN columns: `{all_nan}`",
        f"- High-missing columns (>0.95): `{high_missing}`",
        f"- Constant columns: `{constant}`",
        f"- Overall missing rate: `{sleep.isna().mean().mean():.6f}`",
        "",
        "## Output Tables",
        "",
        f"- Feature summary: `{SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Subject quality: `{SUBJECT_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Guardrails",
        "",
        "- Target columns are not used.",
        "- The sleep episode CSV is not modified.",
        "- Low-confidence rows are flagged, not deleted.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("==================================")
    print("SLEEP EPISODE VALIDATION")
    print("==================================")
    print(f"Rows : {len(sleep)}")
    print(f"Match Rate : {match_rate * 100:.2f} %")
    print(f"Columns : {sleep.shape[1]}")
    print(f"All-NaN : {all_nan}")
    print(f"High Missing : {high_missing}")
    print(f"Constant : {constant}")
    print(f"Short <4h : {short_rows}")
    print(f"Long >12h : {long_rows}")
    print(f"Wake Fallback : {wake_fallback_rows}")
    print(f"Report : {REPORT_PATH}")
    print("==================================")


if __name__ == "__main__":
    main()
