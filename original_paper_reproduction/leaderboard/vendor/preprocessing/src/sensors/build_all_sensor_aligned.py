"""Build sensor-specific canonical and train-aligned tables for all sensors.

This script follows the v2 sensor policy:
- raw parquet files are read-only
- each sensor gets its own canonical table
- each sensor gets its own train-row-aligned CSV
- target columns are never used
- sensor-level reports mark quality issues, while final feature deletion is
  reserved for global cleaning
"""

from __future__ import annotations

import ast
import math
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
CANONICAL_DIR = PROJECT_ROOT / "data" / "canonical"
ALIGNED_DIR = PROJECT_ROOT / "data" / "interim" / "sensor_aligned"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

KEY_COLS = ["subject_id", "lifelog_date", "sleep_date"]

WINDOWS = {
    "day": ("lifelog_date", "00:00", "sleep_date", "00:00", 1440),
    "evening": ("lifelog_date", "18:00", "sleep_date", "00:00", 360),
    "pre_sleep": ("lifelog_date", "21:00", "sleep_date", "00:00", 180),
    "night": ("sleep_date", "00:00", "sleep_date", "06:00", 360),
    "morning": ("sleep_date", "06:00", "sleep_date", "12:00", 360),
    "episode_proxy": ("lifelog_date", "20:00", "sleep_date", "11:00", 900),
}

SENSORS = {
    "mScreenStatus": "ch2025_mScreenStatus.parquet",
    "mUsageStats": "ch2025_mUsageStats.parquet",
    "wPedo": "ch2025_wPedo.parquet",
    "mActivity": "ch2025_mActivity.parquet",
    "wHr": "ch2025_wHr.parquet",
    "mLight": "ch2025_mLight.parquet",
    "wLight": "ch2025_wLight.parquet",
    "mACStatus": "ch2025_mACStatus.parquet",
    "mGps": "ch2025_mGps.parquet",
    "mWifi": "ch2025_mWifi.parquet",
    "mBle": "ch2025_mBle.parquet",
    "mAmbience": "ch2025_mAmbience.parquet",
}


def setup_dirs() -> None:
    for path in [CANONICAL_DIR, ALIGNED_DIR, REPORT_DIR, TABLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_train_keys() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_PATH)
    keys = train[KEY_COLS].copy()
    keys["lifelog_date"] = pd.to_datetime(keys["lifelog_date"]).dt.normalize()
    keys["sleep_date"] = pd.to_datetime(keys["sleep_date"]).dt.normalize()
    keys["row_order"] = np.arange(len(keys))
    if keys.duplicated(KEY_COLS).any():
        raise ValueError("Train key is not unique.")
    return keys


def parse_object(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, dict)):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def safe_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return len(value)
    parsed = parse_object(value)
    if isinstance(parsed, dict):
        return len(parsed)
    if isinstance(parsed, (list, tuple, np.ndarray)):
        return len(parsed)
    return 1 if parsed is not None else 0


def clip_series(s: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if lower is not None:
        out = out.mask(out < lower)
    if upper is not None:
        out = out.clip(upper=upper)
    return out


def robust_upper(s: pd.Series, q: float = 0.995) -> float:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.nan
    return float(vals.quantile(q))


def assign_lifelog_date(ts: pd.Series, window: str) -> pd.Series:
    date = ts.dt.normalize()
    hour = ts.dt.hour
    if window == "day":
        return date
    if window == "evening":
        return date.where(hour >= 18)
    if window == "pre_sleep":
        return date.where(hour >= 21)
    if window == "night":
        return (date - pd.Timedelta(days=1)).where(hour < 6)
    if window == "morning":
        return (date - pd.Timedelta(days=1)).where((hour >= 6) & (hour < 12))
    if window == "episode_proxy":
        out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns]")
        out.loc[hour >= 20] = date.loc[hour >= 20]
        out.loc[hour < 11] = date.loc[hour < 11] - pd.Timedelta(days=1)
        return out
    raise ValueError(f"Unknown window: {window}")


def longest_streak(values: pd.Series, target: int) -> int:
    best = 0
    cur = 0
    for val in values.fillna(0).astype(int):
        if val == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def transition_count(values: pd.Series) -> int:
    vals = values.dropna().astype(int)
    if vals.empty:
        return 0
    return int(vals.diff().fillna(0).ne(0).sum())


def summarize_scan(value: Any, id_key: str | None = None) -> dict[str, float]:
    parsed = parse_object(value)
    if isinstance(parsed, dict):
        items = list(parsed.values())
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []
    ids = set()
    classes = set()
    rssis = []
    for item in items:
        obj = item.tolist() if isinstance(item, np.ndarray) else item
        if isinstance(obj, dict):
            if id_key and obj.get(id_key) is not None:
                ids.add(str(obj.get(id_key)))
            if obj.get("device_class") is not None:
                classes.add(str(obj.get("device_class")))
            if obj.get("rssi") is not None:
                rssis.append(to_float(obj.get("rssi")))
        elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
            rssis.append(to_float(obj[-1]))
    rssi = pd.Series(rssis, dtype="float64").dropna()
    return {
        "scan_count": len(items),
        "unique_id_count": len(ids),
        "class_count": len(classes),
        "rssi_mean": float(rssi.mean()) if len(rssi) else np.nan,
        "rssi_std": float(rssi.std(ddof=0)) if len(rssi) else np.nan,
        "rssi_min": float(rssi.min()) if len(rssi) else np.nan,
        "rssi_max": float(rssi.max()) if len(rssi) else np.nan,
        "strong_signal_count": int((rssi >= -60).sum()) if len(rssi) else 0,
    }


def summarize_gps(value: Any) -> dict[str, float]:
    parsed = parse_object(value)
    items = parsed if isinstance(parsed, list) else []
    speeds, lats, lons, alts = [], [], [], []
    for item in items:
        obj = item.tolist() if isinstance(item, np.ndarray) else item
        if isinstance(obj, dict):
            speeds.append(to_float(obj.get("speed")))
            lats.append(to_float(obj.get("latitude")))
            lons.append(to_float(obj.get("longitude")))
            alts.append(to_float(obj.get("altitude")))
    speed = pd.Series(speeds, dtype="float64").dropna()
    lat = pd.Series(lats, dtype="float64").dropna()
    lon = pd.Series(lons, dtype="float64").dropna()
    alt = pd.Series(alts, dtype="float64").dropna()
    speed_clipped = speed.clip(lower=0, upper=15) if len(speed) else speed
    return {
        "point_count": len(items),
        "speed_mean": float(speed_clipped.mean()) if len(speed_clipped) else np.nan,
        "speed_max": float(speed_clipped.max()) if len(speed_clipped) else np.nan,
        "speed_outlier_count": int((speed > 15).sum()) if len(speed) else 0,
        "stationary_ratio": float((speed <= 0.5).mean()) if len(speed) else np.nan,
        "moving_ratio": float((speed > 1.0).mean()) if len(speed) else np.nan,
        "lat_std": float(lat.std(ddof=0)) if len(lat) else np.nan,
        "lon_std": float(lon.std(ddof=0)) if len(lon) else np.nan,
        "altitude_mean": float(alt.mean()) if len(alt) else np.nan,
    }


def summarize_usage(value: Any) -> dict[str, float]:
    parsed = parse_object(value)
    items = parsed if isinstance(parsed, list) else []
    times = []
    for item in items:
        obj = item.tolist() if isinstance(item, np.ndarray) else item
        if isinstance(obj, dict):
            for key in ["total_time", "totalTime", "usage_time", "time"]:
                if key in obj:
                    times.append(to_float(obj.get(key)))
                    break
        elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
            times.append(to_float(obj[-1]))
    vals = pd.Series(times, dtype="float64").dropna().clip(lower=0)
    return {
        "app_count": len(items),
        "total_usage_time": float(vals.sum()) if len(vals) else 0.0,
        "max_app_usage_time": float(vals.max()) if len(vals) else np.nan,
        "usage_record_count": int(len(vals)),
    }


def summarize_ambience(value: Any) -> dict[str, float]:
    parsed = parse_object(value)
    items = parsed if isinstance(parsed, list) else []
    out = {
        "valid_class_count": 0,
        "top_prob": np.nan,
        "quiet_proxy": 0.0,
        "noise_prob": 0.0,
        "speech_prob": 0.0,
        "music_prob": 0.0,
        "vehicle_prob": 0.0,
        "human_sound_prob": 0.0,
    }
    top_prob = -np.inf
    for item in items:
        obj = item.tolist() if isinstance(item, np.ndarray) else item
        if not isinstance(obj, (list, tuple)) or len(obj) < 2:
            continue
        label = str(obj[0]).lower()
        prob = to_float(obj[1])
        if np.isnan(prob):
            continue
        prob = max(0.0, min(1.0, prob))
        out["valid_class_count"] += 1
        top_prob = max(top_prob, prob)
        if any(k in label for k in ["silence", "quiet"]):
            out["quiet_proxy"] += prob
        if any(k in label for k in ["noise", "sound", "bang", "clatter", "siren"]):
            out["noise_prob"] += prob
        if any(k in label for k in ["speech", "conversation", "talk", "voice", "human"]):
            out["speech_prob"] += prob
            out["human_sound_prob"] += prob
        if "music" in label:
            out["music_prob"] += prob
        if any(k in label for k in ["vehicle", "motor", "traffic", "train", "bus", "car"]):
            out["vehicle_prob"] += prob
    out["top_prob"] = top_prob if np.isfinite(top_prob) else np.nan
    return out


def summarize_hr(value: Any) -> dict[str, float]:
    parsed = parse_object(value)
    arr = np.array(parsed if parsed is not None else [], dtype=float).ravel()
    valid = arr[(arr >= 35) & (arr <= 220)]
    return {
        "hr_raw_count": int(arr.size),
        "hr_valid_count": int(valid.size),
        "hr_invalid_count": int(arr.size - valid.size),
        "hr_invalid_ratio": float((arr.size - valid.size) / arr.size) if arr.size else np.nan,
        "hr_mean": float(valid.mean()) if valid.size else np.nan,
        "hr_std": float(valid.std(ddof=0)) if valid.size else np.nan,
        "hr_min": float(valid.min()) if valid.size else np.nan,
        "hr_max": float(valid.max()) if valid.size else np.nan,
        "hr_range": float(valid.max() - valid.min()) if valid.size else np.nan,
    }


def raw_audit(raw: pd.DataFrame) -> dict[str, Any]:
    ts = pd.to_datetime(raw["timestamp"])
    return {
        "raw_rows": len(raw),
        "raw_columns": len(raw.columns),
        "raw_missing_values": int(raw.isna().sum().sum()),
        "raw_duplicate_key_rows": int(raw.duplicated(["subject_id", "timestamp"]).sum()),
        "subjects": int(raw["subject_id"].nunique()),
        "timestamp_min": str(ts.min()),
        "timestamp_max": str(ts.max()),
    }


def canonicalize(sensor: str) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[str]]:
    raw = pd.read_parquet(SENSOR_ROOT / SENSORS[sensor])
    audit = raw_audit(raw)
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    value_cols: list[str] = []
    state_cols: list[str] = []

    if sensor == "mScreenStatus":
        vals = pd.to_numeric(df["m_screen_use"], errors="coerce")
        audit["invalid_value_rows"] = int((~vals.isin([0, 1])).sum())
        df = df[vals.isin([0, 1])].copy()
        df["screen_on"] = vals.loc[df.index].astype(int)
        df["screen_off"] = 1 - df["screen_on"]
        value_cols = ["screen_on", "screen_off"]
        state_cols = ["screen_on"]

    elif sensor == "mACStatus":
        vals = pd.to_numeric(df["m_charging"], errors="coerce")
        audit["invalid_value_rows"] = int((~vals.isin([0, 1])).sum())
        df = df[vals.isin([0, 1])].copy()
        df["charging_flag"] = vals.loc[df.index].astype(int)
        value_cols = ["charging_flag"]
        state_cols = ["charging_flag"]

    elif sensor == "mActivity":
        code = pd.to_numeric(df["m_activity"], errors="coerce")
        valid_codes = [0, 1, 2, 3, 4, 5, 7, 8]
        audit["invalid_activity_code_rows"] = int((~code.isin(valid_codes)).sum())
        df["activity_code"] = code
        df["activity_still"] = code.eq(3).astype(int)
        df["activity_unknown"] = code.eq(4).astype(int)
        df["activity_active_mobility"] = code.isin([0, 1, 7, 8]).astype(int)
        df["activity_walking"] = code.eq(7).astype(int)
        df["activity_running"] = code.eq(8).astype(int)
        value_cols = [
            "activity_still",
            "activity_unknown",
            "activity_active_mobility",
            "activity_walking",
            "activity_running",
        ]
        state_cols = ["activity_still", "activity_active_mobility"]

    elif sensor == "wPedo":
        drop_cols = [c for c in ["running_step", "walking_step"] if c in df.columns]
        audit["dropped_constant_zero_columns"] = ",".join(
            c for c in drop_cols if pd.to_numeric(df[c], errors="coerce").fillna(0).abs().sum() == 0
        )
        for col in ["step", "step_frequency", "distance", "speed", "burned_calories"]:
            upper = robust_upper(df[col])
            audit[f"{col}_clip_upper_p995"] = upper
            df[f"{col}_clean"] = clip_series(df[col], lower=0, upper=upper)
            df[f"{col}_invalid_negative"] = pd.to_numeric(df[col], errors="coerce").lt(0).astype(int)
        df["pedo_active"] = (
            df["step_clean"].fillna(0).gt(0) | df["distance_clean"].fillna(0).gt(0) | df["speed_clean"].fillna(0).gt(0)
        ).astype(int)
        value_cols = [
            "step_clean",
            "step_frequency_clean",
            "distance_clean",
            "speed_clean",
            "burned_calories_clean",
            "pedo_active",
        ]
        state_cols = ["pedo_active"]

    elif sensor in ["mLight", "wLight"]:
        raw_col = "m_light" if sensor == "mLight" else "w_light"
        prefix = "mlight" if sensor == "mLight" else "wlight"
        upper = robust_upper(df[raw_col])
        audit["light_clip_upper_p995"] = upper
        df[f"{prefix}_raw"] = pd.to_numeric(df[raw_col], errors="coerce")
        df[f"{prefix}_invalid_negative"] = df[f"{prefix}_raw"].lt(0).astype(int)
        df[f"{prefix}_clipped"] = clip_series(df[raw_col], lower=0, upper=upper)
        df[f"{prefix}_log1p"] = np.log1p(df[f"{prefix}_clipped"])
        df[f"{prefix}_very_low"] = df[f"{prefix}_clipped"].le(1).astype(int)
        df[f"{prefix}_low"] = df[f"{prefix}_clipped"].le(10).astype(int)
        df[f"{prefix}_bright"] = df[f"{prefix}_clipped"].ge(100).astype(int)
        value_cols = [
            f"{prefix}_clipped",
            f"{prefix}_log1p",
            f"{prefix}_very_low",
            f"{prefix}_low",
            f"{prefix}_bright",
        ]
        state_cols = [f"{prefix}_low", f"{prefix}_bright"]

    elif sensor == "wHr":
        parsed = pd.DataFrame([summarize_hr(v) for v in df["heart_rate"]])
        df = pd.concat([df[["subject_id", "timestamp"]], parsed], axis=1)
        audit["hr_total_invalid_samples"] = int(df["hr_invalid_count"].sum())
        value_cols = list(parsed.columns)

    elif sensor == "mGps":
        # GPS is a low-dependency auxiliary channel. Deep point-level parsing of
        # 800k rows is intentionally deferred to a dedicated mobility experiment
        # to avoid making GPS dominate the first sensor-cleaning pass.
        df["gps_point_count"] = df["m_gps"].map(safe_len)
        df["gps_has_record"] = df["gps_point_count"].gt(0).astype(int)
        audit["gps_deep_point_parsing"] = "deferred_auxiliary_only"
        audit["gps_raw_lat_lon_direct_features"] = "blocked"
        value_cols = ["gps_point_count", "gps_has_record"]

    elif sensor == "mWifi":
        parsed = pd.DataFrame([summarize_scan(v, id_key="bssid") for v in df["m_wifi"]]).add_prefix("wifi_")
        df = pd.concat([df[["subject_id", "timestamp"]], parsed], axis=1)
        value_cols = list(parsed.columns)

    elif sensor == "mBle":
        parsed = pd.DataFrame([summarize_scan(v, id_key="address") for v in df["m_ble"]]).add_prefix("ble_")
        df = pd.concat([df[["subject_id", "timestamp"]], parsed], axis=1)
        value_cols = list(parsed.columns)

    elif sensor == "mUsageStats":
        parsed = pd.DataFrame([summarize_usage(v) for v in df["m_usage_stats"]]).add_prefix("usage_")
        df = pd.concat([df[["subject_id", "timestamp"]], parsed], axis=1)
        value_cols = list(parsed.columns)

    elif sensor == "mAmbience":
        parsed = pd.DataFrame([summarize_ambience(v) for v in df["m_ambience"]]).add_prefix("ambience_")
        df = pd.concat([df[["subject_id", "timestamp"]], parsed], axis=1)
        value_cols = list(parsed.columns)

    else:
        raise ValueError(sensor)

    df = df.drop_duplicates(["subject_id", "timestamp"], keep="last").sort_values(["subject_id", "timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    audit["canonical_rows"] = len(df)
    canonical_path = CANONICAL_DIR / f"{sensor}_canonical.parquet"
    df.to_parquet(canonical_path, index=False)
    return df, audit, value_cols, state_cols


def aggregate_sensor(sensor: str, canonical: pd.DataFrame, value_cols: list[str], state_cols: list[str], train: pd.DataFrame) -> pd.DataFrame:
    out = train[KEY_COLS + ["row_order"]].copy()
    for window, (_, _, _, _, expected_minutes) in WINDOWS.items():
        temp = canonical[["subject_id", "timestamp"] + value_cols].copy()
        temp["lifelog_date"] = assign_lifelog_date(temp["timestamp"], window)
        temp = temp.dropna(subset=["lifelog_date"])
        grouped = temp.groupby(["subject_id", "lifelog_date"], sort=False)
        block = grouped.size().rename(f"{sensor}_{window}_obs_count").to_frame()
        block[f"{sensor}_{window}_coverage_ratio"] = (block[f"{sensor}_{window}_obs_count"] / expected_minutes).clip(upper=1.0)
        for col in value_cols:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
            agg = grouped[col].agg(["mean", "std", "median", "min", "max", "sum"])
            agg.columns = [f"{sensor}_{col}_{window}_{stat}" for stat in agg.columns]
            block = block.join(agg)
        for col in state_cols:
            if col in value_cols:
                block[f"{sensor}_{col}_{window}_transition_count"] = grouped[col].apply(transition_count)
                block[f"{sensor}_{col}_{window}_longest_1_streak"] = grouped[col].apply(lambda s: longest_streak(s, 1))
                block[f"{sensor}_{col}_{window}_longest_0_streak"] = grouped[col].apply(lambda s: longest_streak(s, 0))
        out = out.merge(block.reset_index(), on=["subject_id", "lifelog_date"], how="left", validate="one_to_one")

    for col in [c for c in out.columns if c.endswith("_obs_count")]:
        out[col] = out[col].fillna(0).astype(int)
    for col in [c for c in out.columns if c.endswith("_coverage_ratio")]:
        out[col] = out[col].fillna(0)

    rel = {}
    numeric_cols = [
        c
        for c in out.columns
        if c not in KEY_COLS + ["row_order"]
        and pd.api.types.is_numeric_dtype(out[c])
        and any(k in c for k in ["mean", "sum", "ratio", "transition_count", "longest"])
    ]
    for col in numeric_cols[:80]:
        subject_mean = out.groupby("subject_id")[col].transform("mean")
        subject_std = out.groupby("subject_id")[col].transform("std").replace(0, np.nan)
        rel[f"{col}_diff_subject_mean"] = out[col] - subject_mean
        rel[f"{col}_zscore_subject"] = (out[col] - subject_mean) / subject_std
    if rel:
        out = pd.concat([out, pd.DataFrame(rel, index=out.index)], axis=1)
    out = out.sort_values("row_order").drop(columns=["row_order"]).reset_index(drop=True)
    for col in ["lifelog_date", "sleep_date"]:
        out[col] = pd.to_datetime(out[col]).dt.date.astype(str)
    return out


def write_sensor_outputs(sensor: str, aligned: pd.DataFrame, audit: dict[str, Any]) -> None:
    aligned_path = ALIGNED_DIR / f"{sensor}_train_aligned.csv"
    summary_path = TABLE_DIR / f"{sensor}_train_aligned_summary.csv"
    report_path = REPORT_DIR / f"{sensor}_cleaning_report.md"
    aligned.to_csv(aligned_path, index=False, encoding="utf-8-sig")

    rows = []
    for col in aligned.columns:
        if col in KEY_COLS:
            group = "key"
        elif "zscore_subject" in col or "diff_subject_mean" in col:
            group = "subject_relative"
        elif "coverage" in col or "obs_count" in col:
            group = "coverage"
        else:
            group = "sensor_feature"
        missing_ratio = float(aligned[col].isna().mean())
        nunique = int(aligned[col].nunique(dropna=True))
        if col in KEY_COLS:
            action = "keep_key"
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
                "group": group,
                "dtype": str(aligned[col].dtype),
                "missing_ratio": missing_ratio,
                "nunique": nunique,
                "recommended_action": action,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    feature_cols = [c for c in aligned.columns if c not in KEY_COLS]
    all_nan = int((summary["missing_ratio"] == 1.0).sum())
    high_missing = int((summary["missing_ratio"] > 0.95).sum())
    constant = int((summary["nunique"] <= 1).sum())
    lines = [
        f"# {sensor} Cleaning Report",
        "",
        "## Raw And Canonical Audit",
        "",
    ]
    for key, val in audit.items():
        lines.append(f"- {key}: `{val}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Canonical: `data/canonical/{sensor}_canonical.parquet`",
            f"- Train aligned: `data/interim/sensor_aligned/{sensor}_train_aligned.csv`",
            f"- Summary: `outputs/tables/{sensor}_train_aligned_summary.csv`",
            f"- Aligned shape: `{aligned.shape}`",
            f"- Rows match train: `{len(aligned) == 450}`",
            f"- Feature columns: `{len(feature_cols)}`",
            "",
            "## Feature Quality",
            "",
            f"- Overall missing rate: `{aligned.isna().mean().mean():.6f}`",
            f"- All-NaN columns: `{all_nan}`",
            f"- High-missing columns (>0.95): `{high_missing}`",
            f"- Constant columns: `{constant}`",
            "",
            "## Guardrails",
            "",
            "- Raw data was read only.",
            "- Target columns were not used.",
            "- Final deletion is deferred to global feature cleaning.",
            "- Key columns are preserved.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all() -> pd.DataFrame:
    setup_dirs()
    train = load_train_keys()
    status_rows = []
    for sensor in SENSORS:
        print(f"[sensor] {sensor}")
        canonical, audit, value_cols, state_cols = canonicalize(sensor)
        aligned = aggregate_sensor(sensor, canonical, value_cols, state_cols, train)
        write_sensor_outputs(sensor, aligned, audit)
        summary = pd.read_csv(TABLE_DIR / f"{sensor}_train_aligned_summary.csv")
        status_rows.append(
            {
                "sensor": sensor,
                "canonical_rows": len(canonical),
                "aligned_rows": len(aligned),
                "aligned_cols": aligned.shape[1],
                "missing_rate": float(aligned.isna().mean().mean()),
                "all_nan_cols": int((summary["missing_ratio"] == 1.0).sum()),
                "high_missing_cols": int((summary["missing_ratio"] > 0.95).sum()),
                "constant_cols": int((summary["nunique"] <= 1).sum()),
            }
        )
    status = pd.DataFrame(status_rows)
    status.to_csv(TABLE_DIR / "all_sensor_aligned_build_status.csv", index=False, encoding="utf-8-sig")
    report_lines = ["# All Sensor Aligned Build Status", ""]
    report_lines.extend(dataframe_to_markdown(status))
    report_lines.append("")
    (REPORT_DIR / "all_sensor_aligned_build_status.md").write_text("\n".join(report_lines), encoding="utf-8")
    return status


def dataframe_to_markdown(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    status = build_all()
    print("==================================")
    print("ALL SENSOR CLEANING COMPLETE")
    print("==================================")
    print(status.to_string(index=False))
    print("Raw files modified : NO")
    print("Targets included : NO")
    print("==================================")


if __name__ == "__main__":
    main()
