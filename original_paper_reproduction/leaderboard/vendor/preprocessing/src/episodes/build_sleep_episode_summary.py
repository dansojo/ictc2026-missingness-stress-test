"""Reconstruct train-row-aligned sleep episode interpretation columns.

The output is a separate CSV aligned to ch2026_metrics_train.csv rows.
It uses raw sensor parquet files directly and does not read the previous
train_sensor_aligned_summary.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_VENV_SITE = PROJECT_ROOT.parents[0] / "ETRI_Human_AI" / ".venv" / "Lib" / "site-packages"
if V1_VENV_SITE.exists():
    sys.path.insert(0, str(V1_VENV_SITE))

import numpy as np
import pandas as pd

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "data_vvs"
SENSOR_ROOT = RAW_ROOT / "ch2025_data_items"
TRAIN_PATH = RAW_ROOT / "ch2026_metrics_train.csv"
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "train_sleep_episode_summary.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "train_sleep_episode_summary_report.md"
COLUMN_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "train_sleep_episode_summary_columns.csv"

KEY_COLS = ["subject_id", "lifelog_date", "sleep_date"]
TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def setup_dirs() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    COLUMN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_train() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH)
    missing = [c for c in KEY_COLS if c not in train.columns]
    if missing:
        raise ValueError(f"Missing train key columns: {missing}")
    out = train[KEY_COLS].copy()
    out["lifelog_date"] = pd.to_datetime(out["lifelog_date"]).dt.normalize()
    out["sleep_date"] = pd.to_datetime(out["sleep_date"]).dt.normalize()
    out["row_order"] = np.arange(len(out))
    return out


def load_sensor(file_name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = SENSOR_ROOT / file_name
    df = pd.read_parquet(path, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["subject_id", "timestamp"]).reset_index(drop=True)


def load_core_sensors() -> dict[str, dict[str, pd.DataFrame]]:
    print("[load] mScreenStatus")
    screen = load_sensor("ch2025_mScreenStatus.parquet", ["subject_id", "timestamp", "m_screen_use"])
    screen["screen_on"] = pd.to_numeric(screen["m_screen_use"], errors="coerce").fillna(0).astype(int)
    screen = screen[["subject_id", "timestamp", "screen_on"]]

    print("[load] wPedo")
    pedo = load_sensor("ch2025_wPedo.parquet", ["subject_id", "timestamp", "step", "speed", "distance"])
    for col in ["step", "speed", "distance"]:
        pedo[col] = pd.to_numeric(pedo[col], errors="coerce").fillna(0)
    pedo["pedo_active"] = ((pedo["step"] > 0) | (pedo["speed"] > 0) | (pedo["distance"] > 0)).astype(int)
    pedo = pedo[["subject_id", "timestamp", "step", "speed", "distance", "pedo_active"]]

    print("[load] mActivity")
    activity = load_sensor("ch2025_mActivity.parquet", ["subject_id", "timestamp", "m_activity"])
    code = pd.to_numeric(activity["m_activity"], errors="coerce")
    activity["activity_still"] = code.eq(3).astype(int)
    activity["activity_unknown"] = code.eq(4).astype(int)
    activity["activity_active"] = code.isin([0, 1, 7, 8]).astype(int)
    activity = activity[["subject_id", "timestamp", "activity_still", "activity_unknown", "activity_active"]]

    print("[load] mLight")
    mlight = load_sensor("ch2025_mLight.parquet", ["subject_id", "timestamp", "m_light"])
    mlight["m_light"] = pd.to_numeric(mlight["m_light"], errors="coerce")
    mlight["m_low_light"] = mlight["m_light"].le(10).astype(int)
    mlight["m_bright_light"] = mlight["m_light"].ge(100).astype(int)
    mlight = mlight[["subject_id", "timestamp", "m_light", "m_low_light", "m_bright_light"]]

    print("[load] wLight")
    wlight = load_sensor("ch2025_wLight.parquet", ["subject_id", "timestamp", "w_light"])
    wlight["w_light"] = pd.to_numeric(wlight["w_light"], errors="coerce")
    wlight["w_low_light"] = wlight["w_light"].le(10).astype(int)
    wlight["w_bright_light"] = wlight["w_light"].ge(100).astype(int)
    wlight = wlight[["subject_id", "timestamp", "w_light", "w_low_light", "w_bright_light"]]

    print("[load] mACStatus")
    charging = load_sensor("ch2025_mACStatus.parquet", ["subject_id", "timestamp", "m_charging"])
    charging["charging"] = pd.to_numeric(charging["m_charging"], errors="coerce").fillna(0).astype(int)
    charging = charging[["subject_id", "timestamp", "charging"]]

    return {
        "screen": dict(tuple(screen.groupby("subject_id", sort=False))),
        "pedo": dict(tuple(pedo.groupby("subject_id", sort=False))),
        "activity": dict(tuple(activity.groupby("subject_id", sort=False))),
        "mlight": dict(tuple(mlight.groupby("subject_id", sort=False))),
        "wlight": dict(tuple(wlight.groupby("subject_id", sort=False))),
        "charging": dict(tuple(charging.groupby("subject_id", sort=False))),
    }


def subset_sensor(
    by_subject: dict[str, pd.DataFrame],
    subject_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    df = by_subject.get(subject_id)
    if df is None:
        return pd.DataFrame()
    mask = (df["timestamp"] >= start) & (df["timestamp"] < end)
    return df.loc[mask].copy()


def make_minute_grid(row: pd.Series, sensors: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    subject_id = row["subject_id"]
    start = row["lifelog_date"] + pd.Timedelta(hours=18)
    end = row["sleep_date"] + pd.Timedelta(hours=12)
    grid = pd.DataFrame({"timestamp": pd.date_range(start, end, freq="1min", inclusive="left")})

    for name, by_subject in sensors.items():
        part = subset_sensor(by_subject, subject_id, start, end)
        if part.empty:
            continue
        part["minute"] = part["timestamp"].dt.floor("min")
        value_cols = [c for c in part.columns if c not in ["subject_id", "timestamp", "minute"]]
        agg = part.groupby("minute", sort=False)[value_cols].mean().reset_index()
        grid = grid.merge(agg, left_on="timestamp", right_on="minute", how="left").drop(columns=["minute"])

    defaults = {
        "screen_on": 0,
        "pedo_active": 0,
        "step": 0,
        "speed": 0,
        "distance": 0,
        "activity_still": 0,
        "activity_unknown": 0,
        "activity_active": 0,
        "charging": 0,
    }
    for col, default in defaults.items():
        if col not in grid.columns:
            grid[col] = default
        grid[col] = grid[col].fillna(default)

    for col in ["m_light", "w_light", "m_low_light", "m_bright_light", "w_low_light", "w_bright_light"]:
        if col not in grid.columns:
            grid[col] = np.nan

    grid["sensor_active_score"] = (
        grid["screen_on"].gt(0).astype(int)
        + grid["pedo_active"].gt(0).astype(int)
        + grid["activity_active"].gt(0).astype(int)
    )
    grid["inactivity_score"] = (
        grid["screen_on"].eq(0).astype(int)
        + grid["pedo_active"].eq(0).astype(int)
        + grid["activity_still"].gt(0).astype(int)
    )
    grid["is_inactive_minute"] = grid["inactivity_score"].ge(2)
    grid["is_active_minute"] = grid["sensor_active_score"].ge(1)
    return grid


def first_time(grid: pd.DataFrame, mask: pd.Series) -> pd.Timestamp | pd.NaT:
    if mask.any():
        return grid.loc[mask, "timestamp"].iloc[0]
    return pd.NaT


def last_time(grid: pd.DataFrame, mask: pd.Series) -> pd.Timestamp | pd.NaT:
    if mask.any():
        return grid.loc[mask, "timestamp"].iloc[-1]
    return pd.NaT


def longest_inactive_segment(grid: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT, int]:
    part = grid[(grid["timestamp"] >= start) & (grid["timestamp"] < end)].copy()
    if part.empty:
        return pd.NaT, pd.NaT, 0
    inactive = part["is_inactive_minute"].fillna(False).to_numpy()
    timestamps = part["timestamp"].to_numpy()
    best_start = None
    best_end = None
    best_len = 0
    cur_start = None
    cur_len = 0
    for idx, val in enumerate(inactive):
        if val:
            if cur_start is None:
                cur_start = idx
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = idx
        else:
            cur_start = None
            cur_len = 0
    if best_start is None:
        return pd.NaT, pd.NaT, 0
    return pd.Timestamp(timestamps[best_start]), pd.Timestamp(timestamps[best_end]) + pd.Timedelta(minutes=1), int(best_len)


def find_first_sustained_active(grid: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp | pd.NaT, int]:
    part = grid[(grid["timestamp"] >= start) & (grid["timestamp"] < end)].copy()
    if part.empty:
        return pd.NaT, 0
    active = part["is_active_minute"].fillna(False).astype(bool).to_numpy()
    timestamps = part["timestamp"].to_numpy()
    for idx in range(len(active)):
        window = active[idx : idx + 10]
        if len(window) < 5:
            break
        if window.sum() >= 2:
            return pd.Timestamp(timestamps[idx]), int(window.sum())
    return pd.NaT, 0


def count_active_bursts(part: pd.DataFrame) -> tuple[int, int]:
    if part.empty:
        return 0, 0
    active = part["is_active_minute"].fillna(False).astype(bool).to_numpy()
    bursts = 0
    minutes = 0
    in_burst = False
    cur_len = 0
    for val in active:
        if val:
            cur_len += 1
            minutes += 1
            if not in_burst:
                bursts += 1
                in_burst = True
        else:
            cur_len = 0
            in_burst = False
    return bursts, minutes


def calculate_confidence(*signals: bool) -> float:
    available = sum(bool(s) for s in signals)
    return round(min(1.0, available / max(1, len(signals))), 3)


def reconstruct_episode(row: pd.Series, sensors: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
    grid = make_minute_grid(row, sensors)
    lifelog_date = row["lifelog_date"]
    sleep_date = row["sleep_date"]

    sleep_search_start = lifelog_date + pd.Timedelta(hours=21)
    sleep_search_end = sleep_date + pd.Timedelta(hours=4)
    wake_search_start = sleep_date + pd.Timedelta(hours=5)
    wake_search_end = sleep_date + pd.Timedelta(hours=11)

    inactive_start, inactive_end, longest_inactive = longest_inactive_segment(grid, sleep_search_start, wake_search_end)

    screen_last_on = last_time(
        grid,
        (grid["timestamp"] >= sleep_search_start)
        & (grid["timestamp"] < sleep_date + pd.Timedelta(hours=4))
        & grid["screen_on"].gt(0),
    )
    pedo_last_active = last_time(
        grid,
        (grid["timestamp"] >= sleep_search_start)
        & (grid["timestamp"] < sleep_date + pd.Timedelta(hours=4))
        & grid["pedo_active"].gt(0),
    )
    activity_last_active = last_time(
        grid,
        (grid["timestamp"] >= sleep_search_start)
        & (grid["timestamp"] < sleep_date + pd.Timedelta(hours=4))
        & grid["activity_active"].gt(0),
    )

    sleep_candidates = []
    for name, ts in [
        ("longest_inactive_start", inactive_start),
        ("screen_last_on_plus_15m", screen_last_on + pd.Timedelta(minutes=15) if pd.notna(screen_last_on) else pd.NaT),
        ("pedo_last_active_plus_10m", pedo_last_active + pd.Timedelta(minutes=10) if pd.notna(pedo_last_active) else pd.NaT),
        ("activity_last_active_plus_10m", activity_last_active + pd.Timedelta(minutes=10) if pd.notna(activity_last_active) else pd.NaT),
    ]:
        if pd.notna(ts) and sleep_search_start <= ts <= sleep_search_end:
            sleep_candidates.append((name, ts))
    if sleep_candidates:
        estimated_sleep_start = sorted(sleep_candidates, key=lambda x: x[1])[0][1]
        sleep_start_source = sorted(sleep_candidates, key=lambda x: x[1])[0][0]
    else:
        estimated_sleep_start = lifelog_date + pd.Timedelta(hours=23, minutes=30)
        sleep_start_source = "fallback_23_30"

    wake_candidate, sustained_count = find_first_sustained_active(grid, wake_search_start, wake_search_end)
    screen_first_on = first_time(
        grid,
        (grid["timestamp"] >= wake_search_start) & (grid["timestamp"] < wake_search_end) & grid["screen_on"].gt(0),
    )
    pedo_first_active = first_time(
        grid,
        (grid["timestamp"] >= wake_search_start) & (grid["timestamp"] < wake_search_end) & grid["pedo_active"].gt(0),
    )
    activity_first_active = first_time(
        grid,
        (grid["timestamp"] >= wake_search_start) & (grid["timestamp"] < wake_search_end) & grid["activity_active"].gt(0),
    )
    bright_first = first_time(
        grid,
        (grid["timestamp"] >= wake_search_start)
        & (grid["timestamp"] < wake_search_end)
        & (grid["m_bright_light"].fillna(0).gt(0) | grid["w_bright_light"].fillna(0).gt(0)),
    )

    wake_candidates = []
    for name, ts in [
        ("sustained_activity", wake_candidate),
        ("screen_first_on", screen_first_on),
        ("pedo_first_active", pedo_first_active),
        ("activity_first_active", activity_first_active),
        ("first_bright_light", bright_first),
    ]:
        if pd.notna(ts):
            wake_candidates.append((name, ts))
    if wake_candidates:
        # Use the earliest robust behavioral candidate; bright light only if no behavior exists.
        behavior = [(n, t) for n, t in wake_candidates if n != "first_bright_light"]
        selected = sorted(behavior or wake_candidates, key=lambda x: x[1])[0]
        estimated_wake_time = selected[1]
        wake_source = selected[0]
    else:
        estimated_wake_time = sleep_date + pd.Timedelta(hours=7, minutes=30)
        wake_source = "fallback_07_30"

    if estimated_wake_time <= estimated_sleep_start:
        estimated_wake_time = max(estimated_wake_time, sleep_date + pd.Timedelta(hours=7, minutes=30))
        wake_source = wake_source + "_adjusted"

    sleep_window = grid[(grid["timestamp"] >= estimated_sleep_start) & (grid["timestamp"] < estimated_wake_time)].copy()
    sleep_duration = (estimated_wake_time - estimated_sleep_start).total_seconds() / 60
    night_bursts, night_active_minutes = count_active_bursts(sleep_window)
    is_short_episode = sleep_duration < 240
    is_long_episode = sleep_duration > 720
    if is_short_episode:
        plausibility = "short_lt_4h"
    elif is_long_episode:
        plausibility = "long_gt_12h"
    else:
        plausibility = "normal_4h_to_12h"

    sleep_start_conf = calculate_confidence(
        pd.notna(inactive_start),
        pd.notna(screen_last_on),
        pd.notna(pedo_last_active) or pd.notna(activity_last_active),
        longest_inactive >= 180,
    )
    wake_conf = calculate_confidence(
        pd.notna(wake_candidate),
        pd.notna(screen_first_on),
        pd.notna(pedo_first_active) or pd.notna(activity_first_active),
        pd.notna(bright_first),
    )
    if is_short_episode or is_long_episode:
        sleep_start_conf = round(sleep_start_conf * 0.7, 3)
        wake_conf = round(wake_conf * 0.7, 3)

    return {
        "subject_id": row["subject_id"],
        "lifelog_date": lifelog_date.date().isoformat(),
        "sleep_date": sleep_date.date().isoformat(),
        "estimated_sleep_start": estimated_sleep_start.isoformat(sep=" ") if pd.notna(estimated_sleep_start) else "",
        "estimated_wake_time": estimated_wake_time.isoformat(sep=" ") if pd.notna(estimated_wake_time) else "",
        "estimated_sleep_duration_minutes": round(float(sleep_duration), 2),
        "estimated_sleep_start_hour": round(estimated_sleep_start.hour + estimated_sleep_start.minute / 60, 3),
        "estimated_wake_hour": round(estimated_wake_time.hour + estimated_wake_time.minute / 60, 3),
        "sleep_start_source": sleep_start_source,
        "wake_source": wake_source,
        "sleep_start_confidence": sleep_start_conf,
        "wake_confidence": wake_conf,
        "episode_confidence": round((sleep_start_conf + wake_conf) / 2, 3),
        "episode_plausibility": plausibility,
        "is_short_sleep_episode_lt4h": int(is_short_episode),
        "is_long_sleep_episode_gt12h": int(is_long_episode),
        "longest_inactive_streak_minutes_21_11": longest_inactive,
        "longest_inactive_start": inactive_start.isoformat(sep=" ") if pd.notna(inactive_start) else "",
        "longest_inactive_end": inactive_end.isoformat(sep=" ") if pd.notna(inactive_end) else "",
        "screen_last_on_before_sleep": screen_last_on.isoformat(sep=" ") if pd.notna(screen_last_on) else "",
        "screen_first_on_after_04": screen_first_on.isoformat(sep=" ") if pd.notna(screen_first_on) else "",
        "pedo_last_active_before_sleep": pedo_last_active.isoformat(sep=" ") if pd.notna(pedo_last_active) else "",
        "pedo_first_active_after_04": pedo_first_active.isoformat(sep=" ") if pd.notna(pedo_first_active) else "",
        "activity_last_active_before_sleep": activity_last_active.isoformat(sep=" ") if pd.notna(activity_last_active) else "",
        "activity_first_active_after_04": activity_first_active.isoformat(sep=" ") if pd.notna(activity_first_active) else "",
        "first_bright_light_after_04": bright_first.isoformat(sep=" ") if pd.notna(bright_first) else "",
        "sleep_window_minutes": int(len(sleep_window)),
        "sleep_window_screen_on_minutes": int(sleep_window["screen_on"].gt(0).sum()),
        "sleep_window_pedo_active_minutes": int(sleep_window["pedo_active"].gt(0).sum()),
        "sleep_window_activity_active_minutes": int(sleep_window["activity_active"].gt(0).sum()),
        "sleep_window_activity_still_minutes": int(sleep_window["activity_still"].gt(0).sum()),
        "sleep_window_charging_minutes": int(sleep_window["charging"].gt(0).sum()),
        "sleep_window_mlight_low_ratio": round(float(sleep_window["m_low_light"].mean()), 4)
        if sleep_window["m_low_light"].notna().any()
        else np.nan,
        "sleep_window_wlight_low_ratio": round(float(sleep_window["w_low_light"].mean()), 4)
        if sleep_window["w_low_light"].notna().any()
        else np.nan,
        "estimated_night_awake_event_count": int(night_bursts),
        "estimated_night_awake_minutes": int(night_active_minutes),
        "wake_sustained_activity_count_10min": int(sustained_count),
        "used_fallback_sleep_start": int(sleep_start_source.startswith("fallback")),
        "used_fallback_wake_time": int(wake_source.startswith("fallback")),
    }


def add_subject_baseline_columns(result: pd.DataFrame) -> pd.DataFrame:
    out = result.copy()
    out["estimated_sleep_start_dt"] = pd.to_datetime(out["estimated_sleep_start"])
    out["estimated_wake_time_dt"] = pd.to_datetime(out["estimated_wake_time"])
    out["sleep_start_minutes_from_lifelog_midnight"] = (
        (out["estimated_sleep_start_dt"] - pd.to_datetime(out["lifelog_date"])).dt.total_seconds() / 60
    )
    out["wake_minutes_from_sleep_midnight"] = (
        (out["estimated_wake_time_dt"] - pd.to_datetime(out["sleep_date"])).dt.total_seconds() / 60
    )
    out["subject_median_sleep_start_minute"] = out.groupby("subject_id")[
        "sleep_start_minutes_from_lifelog_midnight"
    ].transform("median")
    out["subject_median_wake_minute"] = out.groupby("subject_id")["wake_minutes_from_sleep_midnight"].transform("median")
    out["sleep_start_diff_subject_median_minutes"] = (
        out["sleep_start_minutes_from_lifelog_midnight"] - out["subject_median_sleep_start_minute"]
    )
    out["wake_diff_subject_median_minutes"] = out["wake_minutes_from_sleep_midnight"] - out["subject_median_wake_minute"]
    out = out.drop(columns=["estimated_sleep_start_dt", "estimated_wake_time_dt"])
    return out


def write_report(result: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "column": result.columns,
            "dtype": [str(result[c].dtype) for c in result.columns],
            "missing_ratio": [float(result[c].isna().mean()) for c in result.columns],
            "nunique": [int(result[c].nunique(dropna=True)) for c in result.columns],
        }
    )
    summary.to_csv(COLUMN_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    lines = [
        "# Sleep Episode Summary Report",
        "",
        "This file reconstructs sleep episode interpretation columns aligned to train rows.",
        "It is not a final train merge file and does not include target columns.",
        "",
        "## Output",
        "",
        f"- CSV: `{OUT_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Shape: `{result.shape}`",
        f"- Rows match train: `{len(result) == 450}`",
        "",
        "## Core Sensors",
        "",
        "- Primary: `mScreenStatus`, `wPedo`, `mActivity`",
        "- Auxiliary: `mLight`, `wLight`, `mACStatus`",
        "- Not used for boundary in this version: `mGps`, `mWifi`, `mBle`, `mAmbience`, `mUsageStats`",
        "",
        "## Reconstruction Logic",
        "",
        "- Episode date is `lifelog_date`.",
        "- Raw minute grid covers 18:00 on `lifelog_date` to 12:00 on `sleep_date`.",
        "- Sleep-start candidates are constrained to 21:00 on `lifelog_date` through 04:00 on `sleep_date`.",
        "- Wake candidates are constrained to 05:00 through 11:00 on `sleep_date`.",
        "- Sleep start is inferred from long inactivity, last screen use, last pedometer movement, and last activity signal.",
        "- Wake time is inferred from first sustained activity, first screen use, first pedometer movement, first active activity state, and bright light as auxiliary.",
        "- Night awake is counted as active bursts inside estimated sleep start and wake time.",
        "",
        "## Quality Snapshot",
        "",
        f"- Mean sleep duration minutes: `{result['estimated_sleep_duration_minutes'].mean():.2f}`",
        f"- Median sleep duration minutes: `{result['estimated_sleep_duration_minutes'].median():.2f}`",
        f"- Mean sleep start confidence: `{result['sleep_start_confidence'].mean():.3f}`",
        f"- Mean wake confidence: `{result['wake_confidence'].mean():.3f}`",
        f"- Sleep fallback rows: `{int(result['used_fallback_sleep_start'].sum())}`",
        f"- Wake fallback rows: `{int(result['used_fallback_wake_time'].sum())}`",
        f"- Short episode rows (<4h): `{int(result['is_short_sleep_episode_lt4h'].sum())}`",
        f"- Long episode rows (>12h): `{int(result['is_long_sleep_episode_gt12h'].sum())}`",
        "",
        "## Guardrails",
        "",
        "- Target columns Q1/Q2/Q3/S1/S2/S3/S4 are not used.",
        "- The previous train_sensor_aligned_summary.csv is not used.",
        "- Raw sensor files are read only.",
        "- Boundary estimates should be audited before modeling.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_dirs()
    train = load_train()
    sensors = load_core_sensors()
    rows = []
    for idx, row in train.iterrows():
        if idx % 50 == 0:
            print(f"[episode] {idx}/{len(train)}")
        rows.append(reconstruct_episode(row, sensors))
    result = pd.DataFrame(rows)
    result = add_subject_baseline_columns(result)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    write_report(result)

    print("==================================")
    print("SLEEP EPISODE SUMMARY")
    print("==================================")
    print(f"Rows : {result.shape[0]}")
    print(f"Columns : {result.shape[1]}")
    print(f"Output : {OUT_PATH}")
    print(f"Report : {REPORT_PATH}")
    print(f"Mean Duration : {result['estimated_sleep_duration_minutes'].mean():.2f} min")
    print(f"Sleep Fallback Rows : {int(result['used_fallback_sleep_start'].sum())}")
    print(f"Wake Fallback Rows : {int(result['used_fallback_wake_time'].sum())}")
    print("Targets included : NO")
    print("Raw files modified : NO")
    print("==================================")


if __name__ == "__main__":
    main()
