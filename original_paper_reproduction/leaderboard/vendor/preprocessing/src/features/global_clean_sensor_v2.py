"""Global train feature cleaning and deterministic application to test."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_VENV_SITE = PROJECT_ROOT.parents[0] / "ETRI_Human_AI" / ".venv" / "Lib" / "site-packages"
if V1_VENV_SITE.exists():
    sys.path.insert(0, str(V1_VENV_SITE))

import numpy as np
import pandas as pd


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_integrated_sensor_v2.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_integrated_sensor_v2.csv"
TRAIN_OUT = PROJECT_ROOT / "data" / "processed" / "train_integrated_sensor_v2_cleaned.csv"
TEST_OUT = PROJECT_ROOT / "data" / "processed" / "test_integrated_sensor_v2_cleaned.csv"
FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "feature_columns_sensor_v2_cleaned.json"
RULES_OUT = PROJECT_ROOT / "data" / "processed" / "cleaning_rules_sensor_v2.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "global_feature_cleaning_sensor_v2_report.md"
REMOVED_PATH = PROJECT_ROOT / "outputs" / "tables" / "global_removed_features_sensor_v2.csv"
CLEAN_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "global_cleaned_feature_summary_sensor_v2.csv"

KEY_COLS = ["subject_id", "lifelog_date", "sleep_date"]
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
DATE_LIKE_HINTS = ["date", "time", "_start", "_end"]


def setup_dirs() -> None:
    for path in [TRAIN_OUT, TEST_OUT, FEATURES_OUT, RULES_OUT, REPORT_PATH, REMOVED_PATH, CLEAN_SUMMARY_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path)
    for col in ["lifelog_date", "sleep_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date.astype(str)
    return df.replace([np.inf, -np.inf], np.nan)


def is_trainable(col: str, df: pd.DataFrame) -> bool:
    if col in KEY_COLS or col in TARGETS:
        return False
    lower = col.lower()
    if any(hint in lower for hint in DATE_LIKE_HINTS):
        return False
    return pd.api.types.is_numeric_dtype(df[col])


def most_common_ratio(s: pd.Series) -> float:
    vals = s.dropna()
    if vals.empty:
        return np.nan
    return float(vals.value_counts(normalize=True, dropna=True).iloc[0])


def remove_features(
    features: list[str],
    reason: str,
    condition: pd.Series | list[bool],
    stats: pd.DataFrame,
    removed: list[dict],
) -> list[str]:
    condition = pd.Series(condition, index=features)
    to_remove = [f for f in features if bool(condition.loc[f])]
    for f in to_remove:
        row = stats.loc[f]
        removed.append(
            {
                "feature_name": f,
                "removed_reason": reason,
                "missing_ratio": row["missing_ratio"],
                "nunique": row["nunique"],
                "most_common_ratio": row["most_common_ratio"],
            }
        )
    return [f for f in features if f not in set(to_remove)]


def build_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for col in features:
        s = df[col]
        rows.append(
            {
                "feature_name": col,
                "missing_ratio": float(s.isna().mean()),
                "nunique": int(s.nunique(dropna=True)),
                "most_common_ratio": most_common_ratio(s),
                "zero_or_nan_ratio": float((s.fillna(0) == 0).mean()),
                "dtype": str(s.dtype),
            }
        )
    return pd.DataFrame(rows).set_index("feature_name")


def high_corr_filter(df: pd.DataFrame, features: list[str], threshold: float = 0.995) -> tuple[list[str], list[dict]]:
    if len(features) <= 1:
        return features, []
    num = df[features].astype("float32")
    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    removed = []
    for col in upper.columns:
        high = upper.index[upper[col] >= threshold].tolist()
        if high:
            to_drop.add(col)
            removed.append(
                {
                    "feature_name": col,
                    "removed_reason": f"high_correlation_gt_{threshold}",
                    "missing_ratio": float(df[col].isna().mean()),
                    "nunique": int(df[col].nunique(dropna=True)),
                    "most_common_ratio": most_common_ratio(df[col]),
                }
            )
    kept = [f for f in features if f not in to_drop]
    return kept, removed


def main() -> None:
    setup_dirs()
    train = read_table(TRAIN_PATH, "train_integrated")
    test = read_table(TEST_PATH, "test_integrated")

    original_features = [c for c in train.columns if is_trainable(c, train)]
    stats = build_stats(train, original_features)
    features = original_features.copy()
    removed: list[dict] = []

    features = remove_features(features, "all_nan", stats.loc[features]["missing_ratio"].eq(1.0), stats, removed)
    features = remove_features(features, "high_missing_gt_0.95", stats.loc[features]["missing_ratio"].gt(0.95), stats, removed)
    features = remove_features(features, "constant_nunique_le_1", stats.loc[features]["nunique"].le(1), stats, removed)
    features = remove_features(features, "near_constant_ge_0.995", stats.loc[features]["most_common_ratio"].ge(0.995), stats, removed)

    duplicate_cols = []
    if len(features) > 1:
        duplicated = train[features].T.duplicated()
        duplicate_cols = duplicated.index[duplicated].tolist()
    features = remove_features(features, "duplicate_feature", pd.Series([f in duplicate_cols for f in features], index=features), stats, removed)

    corr_features = [f for f in features if train[f].isna().mean() < 0.5]
    kept_corr, corr_removed = high_corr_filter(train, corr_features, threshold=0.995)
    corr_drop = set(corr_features) - set(kept_corr)
    features = [f for f in features if f not in corr_drop]
    removed.extend(corr_removed)

    meta_cols = KEY_COLS + TARGETS
    train_clean = pd.concat([train[meta_cols], train[features]], axis=1)

    test_meta_cols = [c for c in KEY_COLS + TARGETS if c in test.columns]
    test_feature_data = {}
    for col in features:
        if col in test.columns:
            test_feature_data[col] = test[col].to_numpy()
        else:
            test_feature_data[col] = np.full(len(test), np.nan)
    test_feature_frame = pd.DataFrame(test_feature_data, index=test.index)
    test_clean = pd.concat([test[test_meta_cols].copy(), test_feature_frame], axis=1)

    train_clean.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
    test_clean.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")
    FEATURES_OUT.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    RULES_OUT.write_text(
        json.dumps(
            {
                "input_train": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
                "input_test": str(TEST_PATH.relative_to(PROJECT_ROOT)),
                "original_feature_count": len(original_features),
                "cleaned_feature_count": len(features),
                "removed_feature_count": len(removed),
                "rules": [
                    "exclude key/target/date-like/non-numeric columns",
                    "remove all_nan",
                    "remove missing_ratio > 0.95",
                    "remove nunique <= 1",
                    "remove most_common_ratio >= 0.995",
                    "remove duplicate features",
                    "remove high correlation >= 0.995 among features with missing_ratio < 0.5",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    removed_df = pd.DataFrame(removed)
    removed_df.to_csv(REMOVED_PATH, index=False, encoding="utf-8-sig")
    clean_summary = build_stats(train_clean, features).reset_index()
    clean_summary.to_csv(CLEAN_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    reason_counts = removed_df["removed_reason"].value_counts().to_dict() if not removed_df.empty else {}
    lines = [
        "# Global Feature Cleaning Sensor V2 Report",
        "",
        "Cleaning rules are learned from train only and applied deterministically to test.",
        "",
        "## Inputs",
        "",
        f"- Train: `{TRAIN_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Test: `{TEST_PATH.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Outputs",
        "",
        f"- Clean train: `{TRAIN_OUT.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Clean test: `{TEST_OUT.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Feature list: `{FEATURES_OUT.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Cleaning rules: `{RULES_OUT.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## Counts",
        "",
        f"- Original trainable features: `{len(original_features)}`",
        f"- Cleaned features: `{len(features)}`",
        f"- Removed features: `{len(removed)}`",
        f"- Clean train shape: `{train_clean.shape}`",
        f"- Clean test shape: `{test_clean.shape}`",
        f"- Clean train missing rate: `{train_clean[features].isna().mean().mean():.6f}`",
        f"- Clean test missing rate: `{test_clean[features].isna().mean().mean():.6f}`",
        "",
        "## Removed Reason Counts",
        "",
    ]
    for reason, count in reason_counts.items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Test cleaning is not learned independently.",
            "- Target columns are not used as features.",
            "- Key/date-like columns are preserved as metadata but excluded from feature list.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("==================================")
    print("GLOBAL FEATURE CLEANING SENSOR V2")
    print("==================================")
    print(f"Original Features : {len(original_features)}")
    print(f"Cleaned Features : {len(features)}")
    print(f"Removed Features : {len(removed)}")
    print(f"Clean Train Shape : {train_clean.shape}")
    print(f"Clean Test Shape : {test_clean.shape}")
    print(f"Train Missing Rate : {train_clean[features].isna().mean().mean() * 100:.2f} %")
    print(f"Test Missing Rate : {test_clean[features].isna().mean().mean() * 100:.2f} %")
    print("==================================")


if __name__ == "__main__":
    main()
