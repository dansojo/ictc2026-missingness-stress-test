"""Build sleep episode summary for submission sample rows."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_VENV_SITE = PROJECT_ROOT.parents[0] / "ETRI_Human_AI" / ".venv" / "Lib" / "site-packages"
if V1_VENV_SITE.exists():
    sys.path.insert(0, str(V1_VENV_SITE))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "episodes"))

import numpy as np
import pandas as pd

import build_sleep_episode_summary as base


SAMPLE_PATH = PROJECT_ROOT / "data" / "raw" / "data_vvs" / "ch2026_submission_sample.csv"
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "test_sleep_episode_summary.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "test_sleep_episode_summary_report.md"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "test_sleep_episode_summary_columns.csv"


def load_sample_keys() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_PATH)
    out = sample[base.KEY_COLS].copy()
    out["lifelog_date"] = pd.to_datetime(out["lifelog_date"]).dt.normalize()
    out["sleep_date"] = pd.to_datetime(out["sleep_date"]).dt.normalize()
    out["row_order"] = np.arange(len(out))
    if out.duplicated(base.KEY_COLS).any():
        raise ValueError("Sample key is not unique.")
    return out


def write_summary(result: pd.DataFrame) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in result.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(result[col].dtype),
                "missing_ratio": float(result[col].isna().mean()),
                "nunique": int(result[col].nunique(dropna=True)),
            }
        )
    pd.DataFrame(rows).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_keys = load_sample_keys()
    sensors = base.load_core_sensors()
    rows = []
    for idx, row in sample_keys.iterrows():
        if idx % 50 == 0:
            print(f"[test episode] {idx}/{len(sample_keys)}")
        rows.append(base.reconstruct_episode(row, sensors))
    result = pd.DataFrame(rows)
    result = base.add_subject_baseline_columns(result)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    write_summary(result)

    lines = [
        "# Test Sleep Episode Summary Report",
        "",
        f"- CSV: `{OUT_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Shape: `{result.shape}`",
        f"- Rows match sample submission: `{len(result) == len(sample_keys)}`",
        f"- Mean duration minutes: `{result['estimated_sleep_duration_minutes'].mean():.2f}`",
        f"- Median duration minutes: `{result['estimated_sleep_duration_minutes'].median():.2f}`",
        f"- Sleep fallback rows: `{int(result['used_fallback_sleep_start'].sum())}`",
        f"- Wake fallback rows: `{int(result['used_fallback_wake_time'].sum())}`",
        "",
        "Raw files are read only. Target placeholders are not used.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("==================================")
    print("TEST SLEEP EPISODE SUMMARY")
    print("==================================")
    print(f"Rows : {result.shape[0]}")
    print(f"Columns : {result.shape[1]}")
    print(f"Output : {OUT_PATH}")
    print("Targets used : NO")
    print("==================================")


if __name__ == "__main__":
    main()
