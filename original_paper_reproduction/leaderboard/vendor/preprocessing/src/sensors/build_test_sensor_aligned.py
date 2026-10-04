"""Build sensor-aligned CSV files for the submission sample rows.

This mirrors the train sensor-aligned pipeline but uses
ch2026_submission_sample.csv as the key skeleton. Raw sensor files are read only.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_VENV_SITE = PROJECT_ROOT.parents[0] / "ETRI_Human_AI" / ".venv" / "Lib" / "site-packages"
if V1_VENV_SITE.exists():
    sys.path.insert(0, str(V1_VENV_SITE))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "sensors"))

import pandas as pd

import build_all_sensor_aligned as base


SAMPLE_PATH = PROJECT_ROOT / "data" / "raw" / "data_vvs" / "ch2026_submission_sample.csv"
TEST_ALIGNED_DIR = PROJECT_ROOT / "data" / "interim" / "sensor_aligned_test"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "test_sensor_aligned_build_status.md"
STATUS_PATH = PROJECT_ROOT / "outputs" / "tables" / "test_sensor_aligned_build_status.csv"


def load_sample_keys() -> pd.DataFrame:
    sample = pd.read_csv(SAMPLE_PATH)
    keys = sample[base.KEY_COLS].copy()
    keys["lifelog_date"] = pd.to_datetime(keys["lifelog_date"]).dt.normalize()
    keys["sleep_date"] = pd.to_datetime(keys["sleep_date"]).dt.normalize()
    keys["row_order"] = range(len(keys))
    if keys.duplicated(base.KEY_COLS).any():
        raise ValueError("Sample key is not unique.")
    return keys


def write_test_aligned(sensor: str, aligned: pd.DataFrame) -> None:
    TEST_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    path = TEST_ALIGNED_DIR / f"{sensor}_test_aligned.csv"
    aligned.to_csv(path, index=False, encoding="utf-8-sig")


def dataframe_to_markdown(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return lines


def main() -> None:
    base.setup_dirs()
    TEST_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)

    sample_keys = load_sample_keys()
    rows = []
    for sensor in base.SENSORS:
        print(f"[test sensor] {sensor}")
        canonical, audit, value_cols, state_cols = base.canonicalize(sensor)
        aligned = base.aggregate_sensor(sensor, canonical, value_cols, state_cols, sample_keys)
        write_test_aligned(sensor, aligned)
        rows.append(
            {
                "sensor": sensor,
                "test_rows": len(aligned),
                "test_cols": aligned.shape[1],
                "feature_cols": aligned.shape[1] - len(base.KEY_COLS),
                "missing_rate": float(aligned.isna().mean().mean()),
                "canonical_rows": len(canonical),
            }
        )

    status = pd.DataFrame(rows)
    status.to_csv(STATUS_PATH, index=False, encoding="utf-8-sig")
    lines = ["# Test Sensor Aligned Build Status", ""]
    lines.extend(dataframe_to_markdown(status))
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("==================================")
    print("TEST SENSOR ALIGNED BUILD")
    print("==================================")
    print(status.to_string(index=False))
    print("Raw files modified : NO")
    print("Targets used : NO")
    print("==================================")


if __name__ == "__main__":
    main()
