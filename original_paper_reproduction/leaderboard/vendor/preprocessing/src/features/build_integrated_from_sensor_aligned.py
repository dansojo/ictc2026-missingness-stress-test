"""Build final v2 integrated table from 12 sensor-aligned files.

Inputs:
- ch2026_metrics_train.csv
- data/interim/sensor_aligned/*_train_aligned.csv
- data/interim/train_sleep_episode_summary.csv

The output is a new processed candidate table. No model training, validation
split, or submission is created.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_VENV_SITE = PROJECT_ROOT.parents[0] / "ETRI_Human_AI" / ".venv" / "Lib" / "site-packages"
if V1_VENV_SITE.exists():
    sys.path.insert(0, str(V1_VENV_SITE))

import pandas as pd


TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "data_vvs" / "ch2026_metrics_train.csv"
SENSOR_ALIGNED_DIR = PROJECT_ROOT / "data" / "interim" / "sensor_aligned"
SLEEP_PATH = PROJECT_ROOT / "data" / "interim" / "train_sleep_episode_summary.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "train_integrated_sensor_v2.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "train_integrated_sensor_v2_report.md"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "train_integrated_sensor_v2_columns.csv"

KEY_COLS = ["subject_id", "lifelog_date", "sleep_date"]
TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EXPECTED_SENSORS = [
    "mScreenStatus",
    "mUsageStats",
    "wPedo",
    "mActivity",
    "wHr",
    "mLight",
    "wLight",
    "mACStatus",
    "mGps",
    "mWifi",
    "mBle",
    "mAmbience",
]


def setup_dirs() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


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


def validate_key(df: pd.DataFrame, name: str) -> None:
    duplicated = int(df.duplicated(KEY_COLS).sum())
    if duplicated:
        raise ValueError(f"{name} has duplicated key rows: {duplicated}")


def source_group(col: str, sensor_feature_cols: dict[str, set[str]], sleep_cols: set[str]) -> str:
    if col in KEY_COLS:
        return "key"
    if col in TARGET_COLS:
        return "target"
    if col in sleep_cols:
        return "sleep_episode"
    for sensor, cols in sensor_feature_cols.items():
        if col in cols:
            return sensor
    return "unknown"


def build_column_summary(
    df: pd.DataFrame,
    sensor_feature_cols: dict[str, set[str]],
    sleep_cols: set[str],
) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        missing_ratio = float(df[col].isna().mean())
        nunique = int(df[col].nunique(dropna=True))
        if col in KEY_COLS:
            action = "keep_key"
        elif col in TARGET_COLS:
            action = "keep_target"
        elif missing_ratio == 1.0:
            action = "drop_candidate_all_nan"
        elif missing_ratio > 0.95:
            action = "drop_candidate_high_missing"
        elif nunique <= 1:
            action = "drop_candidate_constant"
        else:
            action = "keep_candidate"
        rows.append(
            {
                "column": col,
                "source_group": source_group(col, sensor_feature_cols, sleep_cols),
                "dtype": str(df[col].dtype),
                "missing_ratio": missing_ratio,
                "nunique": nunique,
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    setup_dirs()
    train = read_table(TRAIN_PATH, "train")
    validate_key(train, "train")
    merged = train.copy()

    sensor_feature_cols: dict[str, set[str]] = {}
    sensor_shapes = []
    for sensor in EXPECTED_SENSORS:
        path = SENSOR_ALIGNED_DIR / f"{sensor}_train_aligned.csv"
        sensor_df = read_table(path, sensor)
        validate_key(sensor_df, sensor)
        feature_cols = [c for c in sensor_df.columns if c not in KEY_COLS]
        sensor_feature_cols[sensor] = set(feature_cols)
        sensor_shapes.append((sensor, sensor_df.shape[0], sensor_df.shape[1], len(feature_cols)))
        merged = merged.merge(sensor_df, on=KEY_COLS, how="left", validate="one_to_one")

    sleep = read_table(SLEEP_PATH, "sleep_episode")
    validate_key(sleep, "sleep_episode")
    sleep_feature_cols = set(c for c in sleep.columns if c not in KEY_COLS)
    merged = merged.merge(sleep, on=KEY_COLS, how="left", validate="one_to_one")

    if len(merged) != len(train):
        raise RuntimeError(f"Row count changed: train={len(train)} merged={len(merged)}")

    merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    summary = build_column_summary(merged, sensor_feature_cols, sleep_feature_cols)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    feature_cols = [c for c in merged.columns if c not in KEY_COLS + TARGET_COLS]
    all_nan = int((summary["recommended_action"] == "drop_candidate_all_nan").sum())
    high_missing = int((summary["recommended_action"] == "drop_candidate_high_missing").sum())
    constant = int((summary["recommended_action"] == "drop_candidate_constant").sum())

    lines = [
        "# Train Integrated Sensor V2 Report",
        "",
        "This integrated table is rebuilt from sensor-specific aligned CSV files.",
        "",
        "## Inputs",
        "",
        f"- Train labels: `{TRAIN_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Sensor aligned directory: `{SENSOR_ALIGNED_DIR.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Sleep episode summary: `{SLEEP_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Output",
        "",
        f"- CSV: `{OUT_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Shape: `{merged.shape}`",
        f"- Rows preserved: `{len(merged) == len(train)}`",
        f"- Train rows: `{len(train)}`",
        f"- Integrated rows: `{len(merged)}`",
        f"- Trainable candidate columns: `{len(feature_cols)}`",
        f"- Overall missing rate: `{merged.isna().mean().mean():.6f}`",
        "",
        "## Pre-Cleaning Candidate Flags",
        "",
        f"- All-NaN columns: `{all_nan}`",
        f"- High-missing columns (>0.95): `{high_missing}`",
        f"- Constant columns: `{constant}`",
        "",
        "## Sensor Files",
        "",
    ]
    for sensor, rows, cols, features in sensor_shapes:
        lines.append(f"- `{sensor}`: rows={rows}, cols={cols}, features={features}")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Raw data was not modified.",
            "- Sensor-aligned inputs were not modified.",
            "- No model training was performed.",
            "- No validation split or submission was created.",
            "- Final deletion is deferred to global feature cleaning.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("==================================")
    print("TRAIN INTEGRATED SENSOR V2")
    print("==================================")
    print(f"Train Rows : {len(train)}")
    print(f"Merged Rows : {len(merged)}")
    print(f"Columns : {merged.shape[1]}")
    print(f"Trainable Candidate Columns : {len(feature_cols)}")
    print(f"Missing Rate : {merged.isna().mean().mean() * 100:.2f} %")
    print(f"All-NaN : {all_nan}")
    print(f"High Missing : {high_missing}")
    print(f"Constant : {constant}")
    print(f"Output : {OUT_PATH}")
    print("==================================")


if __name__ == "__main__":
    main()
