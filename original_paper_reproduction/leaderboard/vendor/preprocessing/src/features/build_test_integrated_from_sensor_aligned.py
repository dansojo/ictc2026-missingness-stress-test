"""Build test integrated table from sample submission and test-aligned sensors."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_VENV_SITE = PROJECT_ROOT.parents[0] / "ETRI_Human_AI" / ".venv" / "Lib" / "site-packages"
if V1_VENV_SITE.exists():
    sys.path.insert(0, str(V1_VENV_SITE))

import pandas as pd


SAMPLE_PATH = PROJECT_ROOT / "data" / "raw" / "data_vvs" / "ch2026_submission_sample.csv"
TEST_SENSOR_DIR = PROJECT_ROOT / "data" / "interim" / "sensor_aligned_test"
TEST_SLEEP_PATH = PROJECT_ROOT / "data" / "interim" / "test_sleep_episode_summary.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "test_integrated_sensor_v2.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "test_integrated_sensor_v2_report.md"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "test_integrated_sensor_v2_columns.csv"

KEY_COLS = ["subject_id", "lifelog_date", "sleep_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SENSORS = [
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


def read_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path)
    for col in ["lifelog_date", "sleep_date"]:
        df[col] = pd.to_datetime(df[col]).dt.date.astype(str)
    if df.duplicated(KEY_COLS).any():
        raise ValueError(f"{name} key is duplicated.")
    return df


def write_column_summary(df: pd.DataFrame) -> None:
    rows = []
    for col in df.columns:
        if col in KEY_COLS:
            group = "key"
        elif col in TARGETS:
            group = "target_placeholder"
        elif col.startswith("estimated_") or "sleep" in col or "wake" in col or "episode" in col:
            group = "sleep_episode"
        else:
            group = "sensor_feature"
        rows.append(
            {
                "column": col,
                "group": group,
                "dtype": str(df[col].dtype),
                "missing_ratio": float(df[col].isna().mean()),
                "nunique": int(df[col].nunique(dropna=True)),
            }
        )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample = read_table(SAMPLE_PATH, "sample_submission")
    merged = sample.copy()
    sensor_shapes = []
    for sensor in SENSORS:
        path = TEST_SENSOR_DIR / f"{sensor}_test_aligned.csv"
        sensor_df = read_table(path, sensor)
        sensor_shapes.append((sensor, sensor_df.shape[0], sensor_df.shape[1]))
        merged = merged.merge(sensor_df, on=KEY_COLS, how="left", validate="one_to_one")

    sleep = read_table(TEST_SLEEP_PATH, "test_sleep_episode")
    merged = merged.merge(sleep, on=KEY_COLS, how="left", validate="one_to_one")

    if len(merged) != len(sample):
        raise RuntimeError(f"Row count changed: sample={len(sample)} merged={len(merged)}")

    merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    write_column_summary(merged)

    lines = [
        "# Test Integrated Sensor V2 Report",
        "",
        f"- Sample rows: `{len(sample)}`",
        f"- Integrated rows: `{len(merged)}`",
        f"- Shape: `{merged.shape}`",
        f"- CSV: `{OUT_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Overall missing rate: `{merged.isna().mean().mean():.6f}`",
        "",
        "## Sensor Files",
        "",
    ]
    for sensor, rows, cols in sensor_shapes:
        lines.append(f"- `{sensor}`: rows={rows}, cols={cols}")
    lines.extend(["", "Target columns are placeholders from sample submission."])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("==================================")
    print("TEST INTEGRATED SENSOR V2")
    print("==================================")
    print(f"Sample Rows : {len(sample)}")
    print(f"Merged Rows : {len(merged)}")
    print(f"Columns : {merged.shape[1]}")
    print(f"Missing Rate : {merged.isna().mean().mean() * 100:.2f} %")
    print(f"Output : {OUT_PATH}")
    print("==================================")


if __name__ == "__main__":
    main()
