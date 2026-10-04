"""Tree model reliability test and submission candidates.

Purpose:
- Keep current cleaned v2 features fixed.
- Compare LGBM/CatBoost/XGBoost under the agreed validation policy.
- Create only two submission candidates:
  1. best non-baseline single model among CatBoost/XGBoost
  2. best tree ensemble using LGBM/CatBoost/XGBoost

This is not a tuning script and does not add features.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SITE_PACKAGES = PROJECT_ROOT.parent / "ETRI_Human_AI" / ".venv" / "Lib" / "site-packages"
if LEGACY_SITE_PACKAGES.exists():
    sys.path.insert(0, str(LEGACY_SITE_PACKAGES))

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss


TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEY_COLS = ["subject_id", "sleep_date", "lifelog_date"]
SEED = 42
EPS = 1e-6
PUBLIC_LB_BASELINE = 0.6312543803


VALIDATION_STRATEGIES = [
    "subject_wise_last14days_holdout",
    "subject_wise_sample_row_count_mimic_holdout",
]


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
        else:
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text_df.columns.astype(str)) + " |"
    divider = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.astype(str).to_numpy()]
    return "\n".join([header, divider, *rows])


def ensure_dirs() -> dict[str, Path]:
    paths = {
        "reports": PROJECT_ROOT / "outputs" / "reports",
        "tables": PROJECT_ROOT / "outputs" / "tables",
        "submissions": PROJECT_ROOT / "outputs" / "submissions",
        "oof": PROJECT_ROOT / "outputs" / "oof",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    raw_dir = PROJECT_ROOT / "data" / "raw" / "data_vvs"
    train = pd.read_csv(processed_dir / "train_integrated_sensor_v2_cleaned.csv")
    test = pd.read_csv(processed_dir / "test_integrated_sensor_v2_cleaned.csv")
    sample = pd.read_csv(raw_dir / "ch2026_submission_sample.csv")
    features = json.loads((processed_dir / "feature_columns_sensor_v2_cleaned.json").read_text(encoding="utf-8"))
    for frame in [train, test, sample]:
        for col in ["lifelog_date", "sleep_date"]:
            frame[col] = pd.to_datetime(frame[col])
    return train, test, sample, features


def build_split(train: pd.DataFrame, sample: pd.DataFrame, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    if strategy == "subject_wise_last14days_holdout":
        valid_mask = pd.Series(False, index=train.index)
        for _, sub_df in train.groupby("subject_id"):
            max_date = sub_df["lifelog_date"].max()
            valid_mask.loc[sub_df.index] = sub_df["lifelog_date"] >= (max_date - pd.Timedelta(days=13))
    elif strategy == "subject_wise_last21days_holdout":
        valid_mask = pd.Series(False, index=train.index)
        for _, sub_df in train.groupby("subject_id"):
            max_date = sub_df["lifelog_date"].max()
            valid_mask.loc[sub_df.index] = sub_df["lifelog_date"] >= (max_date - pd.Timedelta(days=20))
    elif strategy == "subject_wise_last25pct_holdout":
        valid_indices = []
        for _, sub_df in train.sort_values(["subject_id", "lifelog_date"]).groupby("subject_id"):
            n_valid = max(1, int(np.ceil(len(sub_df) * 0.25)))
            valid_indices.extend(sub_df.tail(n_valid).index.tolist())
        valid_mask = train.index.isin(valid_indices)
    elif strategy == "subject_wise_sample_row_count_mimic_holdout":
        valid_indices = []
        sample_counts = sample["subject_id"].value_counts()
        for subject, sub_train in train.groupby("subject_id"):
            sub_sample = sample[sample["subject_id"] == subject]
            if sub_sample.empty:
                continue
            sample_mid = sub_sample["lifelog_date"].min() + (sub_sample["lifelog_date"].max() - sub_sample["lifelog_date"].min()) / 2
            n_valid = min(len(sub_train) - 1, max(1, int(sample_counts.get(subject, 0))))
            dist = (sub_train["lifelog_date"] - sample_mid).abs().dt.days
            valid_indices.extend(dist.sort_values().head(n_valid).index.tolist())
        valid_mask = train.index.isin(valid_indices)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    valid_arr = np.asarray(valid_mask)
    train_idx = np.flatnonzero(~valid_arr)
    valid_idx = np.flatnonzero(valid_arr)
    return train_idx, valid_idx


def make_model(model_name: str, n_estimators: int | None = None):
    if model_name == "lgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.03,
            num_leaves=16,
            max_depth=4,
            min_child_samples=10,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=1.0,
            reg_lambda=5.0,
            n_estimators=n_estimators or 1200,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    if model_name == "cat":
        return cb.CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5.0,
            iterations=n_estimators or 1000,
            random_seed=SEED,
            allow_writing_files=False,
            verbose=False,
        )
    if model_name == "xgb":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.65,
            gamma=0.0,
            reg_alpha=1.0,
            reg_lambda=5.0,
            n_estimators=n_estimators or 1000,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=100 if n_estimators is None else None,
        )
    raise ValueError(f"Unknown model: {model_name}")


def fit_predict_valid(model_name: str, x_train, y_train, x_valid, y_valid) -> tuple[np.ndarray, int]:
    model = make_model(model_name)
    if model_name == "lgbm":
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict_proba(x_valid)[:, 1]
        best_iter = int(model.best_iteration_ or model.n_estimators)
    elif model_name == "cat":
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid), early_stopping_rounds=100, use_best_model=True)
        pred = model.predict_proba(x_valid)[:, 1]
        best_iter = int(model.get_best_iteration() or model.tree_count_ or 2000)
    else:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        pred = model.predict_proba(x_valid)[:, 1]
        best_iter = int(getattr(model, "best_iteration", None) or model.n_estimators)
    return pred, max(best_iter, 1)


def fit_predict_test(model_name: str, x, y, x_test, n_estimators: int) -> np.ndarray:
    model = make_model(model_name, n_estimators=n_estimators)
    if model_name == "cat":
        model.fit(x, y)
    else:
        model.fit(x, y)
    return model.predict_proba(x_test)[:, 1]


def score_targets(y_true: pd.DataFrame, pred: pd.DataFrame) -> float:
    return float(np.mean([log_loss(y_true[t], pred[t].clip(EPS, 1 - EPS), labels=[0, 1]) for t in TARGETS]))


def evaluate_models(train: pd.DataFrame, sample: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict, dict]:
    x = train[features]
    results = []
    holdout_preds: dict[str, dict[str, pd.DataFrame]] = {m: {} for m in ["lgbm", "cat", "xgb"]}
    best_iters: dict[str, dict[str, list[int]]] = {m: {t: [] for t in TARGETS} for m in ["lgbm", "cat", "xgb"]}

    for strategy in VALIDATION_STRATEGIES:
        tr_idx, va_idx = build_split(train, sample, strategy)
        y_valid_frame = train.iloc[va_idx][TARGETS].reset_index(drop=True)
        for model_name in ["lgbm", "cat", "xgb"]:
            pred_frame = train.iloc[va_idx][KEY_COLS].reset_index(drop=True).copy()
            target_scores = {}
            for target in TARGETS:
                y = train[target].astype(int)
                pred, best_iter = fit_predict_valid(
                    model_name,
                    x.iloc[tr_idx],
                    y.iloc[tr_idx],
                    x.iloc[va_idx],
                    y.iloc[va_idx],
                )
                pred_frame[target] = pred
                best_iters[model_name][target].append(best_iter)
                target_scores[target] = log_loss(y.iloc[va_idx], np.clip(pred, EPS, 1 - EPS), labels=[0, 1])
            avg_score = score_targets(y_valid_frame, pred_frame[TARGETS])
            holdout_preds[model_name][strategy] = pred_frame
            row = {
                "model": model_name,
                "strategy": strategy,
                "validation_logloss": avg_score,
                "valid_rows": len(va_idx),
                "train_rows": len(tr_idx),
            }
            for target, score in target_scores.items():
                row[f"{target}_logloss"] = score
            results.append(row)
            print(f"{model_name:4s} | {strategy:45s} | {avg_score:.6f}")
    return pd.DataFrame(results), holdout_preds, best_iters


def ensemble_search(eval_df: pd.DataFrame, holdout_preds: dict, train: pd.DataFrame, sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    best_by_strategy = {}
    for strategy in VALIDATION_STRATEGIES:
        _, va_idx = build_split(train, sample, strategy)
        y_valid = train.iloc[va_idx][TARGETS].reset_index(drop=True)
        pred_frames = {m: holdout_preds[m][strategy][TARGETS].reset_index(drop=True) for m in ["lgbm", "cat", "xgb"]}
        for wl in np.arange(0.0, 1.01, 0.1):
            for wc in np.arange(0.0, 1.01 - wl, 0.1):
                wx = round(1.0 - wl - wc, 10)
                if wx < -1e-9:
                    continue
                weights = {"lgbm": float(wl), "cat": float(wc), "xgb": float(wx)}
                pred = sum(weights[m] * pred_frames[m] for m in weights)
                score = score_targets(y_valid, pred)
                rows.append({"strategy": strategy, "lgbm": weights["lgbm"], "cat": weights["cat"], "xgb": weights["xgb"], "validation_logloss": score})
        strat_df = pd.DataFrame([r for r in rows if r["strategy"] == strategy]).sort_values("validation_logloss")
        best_by_strategy[strategy] = strat_df.iloc[0].to_dict()
    return pd.DataFrame(rows), best_by_strategy


def train_test_predictions(train: pd.DataFrame, test: pd.DataFrame, features: list[str], best_iters: dict) -> dict[str, pd.DataFrame]:
    x = train[features]
    x_test = test[features]
    pred_by_model = {}
    for model_name in ["lgbm", "cat", "xgb"]:
        pred_frame = test[KEY_COLS].copy()
        for target in TARGETS:
            y = train[target].astype(int)
            median_iter = int(np.median(best_iters[model_name][target])) if best_iters[model_name][target] else 300
            # Keep final fits compact to avoid over-training from a single tiny validation set.
            n_estimators = int(np.clip(median_iter, 30, 500))
            pred_frame[target] = fit_predict_test(model_name, x, y, x_test, n_estimators)
        pred_by_model[model_name] = pred_frame
    return pred_by_model


def write_submission(sample: pd.DataFrame, pred: pd.DataFrame, path: Path) -> None:
    sub = sample[KEY_COLS + TARGETS].copy()
    merged = sub[KEY_COLS].merge(pred[KEY_COLS + TARGETS], on=KEY_COLS, how="left", validate="one_to_one")
    if merged[TARGETS].isna().any().any():
        raise ValueError(f"Missing predictions for {path.name}")
    for target in TARGETS:
        sub[target] = merged[target].clip(EPS, 1 - EPS)
    sub.to_csv(path, index=False, encoding="utf-8-sig")


def write_report(eval_df: pd.DataFrame, ensemble_df: pd.DataFrame, best_single: str, best_weights: dict, paths: dict[str, Path]) -> None:
    primary = eval_df[eval_df["strategy"] == "subject_wise_last14days_holdout"].sort_values("validation_logloss")
    lines = [
        "# Tree Model Reliability Test",
        "",
        f"- Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Purpose: validate whether model changes still align with the accepted v2 validation policy.",
        "- Features: feature_columns_sensor_v2_cleaned.json",
        "- No new feature engineering or hyperparameter tuning.",
        f"- Baseline Public LB reference: {PUBLIC_LB_BASELINE:.10f}",
        "",
        "## Primary Validation Ranking",
        "",
        df_to_markdown(primary[["model", "strategy", "validation_logloss", "valid_rows"]]),
        "",
        "## All Model Validation Scores",
        "",
        df_to_markdown(eval_df.sort_values(["strategy", "validation_logloss"])),
        "",
        "## Ensemble Weight Search Top 20",
        "",
        df_to_markdown(ensemble_df.sort_values("validation_logloss").head(20)),
        "",
        "## Submission Candidates",
        "",
        f"- Best non-baseline single model: {best_single}",
        f"- Best ensemble weights from primary validation: {best_weights}",
        "",
        "## Decision Rule",
        "",
        "- Submit best_single and best_ensemble only.",
        "- Compare LB movement against the primary and secondary validation scores before starting heavy feature engineering.",
    ]
    (paths["reports"] / "tree_model_reliability_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    paths = ensure_dirs()
    train, test, sample, features = load_inputs()
    eval_df, holdout_preds, best_iters = evaluate_models(train, sample, features)
    ensemble_df, best_by_strategy = ensemble_search(eval_df, holdout_preds, train, sample)

    primary_scores = eval_df[eval_df["strategy"] == "subject_wise_last14days_holdout"].set_index("model")["validation_logloss"]
    non_baseline = primary_scores.drop(index="lgbm", errors="ignore")
    best_single = str(non_baseline.idxmin())
    best_weights = best_by_strategy["subject_wise_last14days_holdout"]

    test_preds = train_test_predictions(train, test, features, best_iters)
    ensemble_pred = test[KEY_COLS].copy()
    for target in TARGETS:
        ensemble_pred[target] = (
            best_weights["lgbm"] * test_preds["lgbm"][target]
            + best_weights["cat"] * test_preds["cat"][target]
            + best_weights["xgb"] * test_preds["xgb"][target]
        )

    eval_df.to_csv(paths["tables"] / "tree_model_reliability_validation_scores.csv", index=False, encoding="utf-8-sig")
    ensemble_df.to_csv(paths["tables"] / "tree_model_reliability_ensemble_weight_search.csv", index=False, encoding="utf-8-sig")
    for model_name, pred in test_preds.items():
        pred.to_csv(paths["tables"] / f"tree_model_reliability_{model_name}_test_predictions.csv", index=False, encoding="utf-8-sig")
    ensemble_pred.to_csv(paths["tables"] / "tree_model_reliability_best_ensemble_test_predictions.csv", index=False, encoding="utf-8-sig")

    write_submission(sample, test_preds[best_single], paths["submissions"] / f"submission_best_single_{best_single}_sensor_v2.csv")
    write_submission(sample, ensemble_pred, paths["submissions"] / "submission_best_tree_ensemble_sensor_v2.csv")
    write_report(eval_df, ensemble_df, best_single, best_weights, paths)

    print("=" * 34)
    print("TREE MODEL RELIABILITY SUBMISSIONS")
    print("=" * 34)
    print("Primary validation scores")
    for model_name, score in primary_scores.sort_values().items():
        print(f"{model_name:4s}: {score:.6f}")
    print(f"Best single candidate : {best_single}")
    print(f"Best ensemble weights : LGBM {best_weights['lgbm']:.1f} CAT {best_weights['cat']:.1f} XGB {best_weights['xgb']:.1f}")
    print(f"Best ensemble primary : {best_weights['validation_logloss']:.6f}")
    print(f"Single submission     : {paths['submissions'] / f'submission_best_single_{best_single}_sensor_v2.csv'}")
    print(f"Ensemble submission   : {paths['submissions'] / 'submission_best_tree_ensemble_sensor_v2.csv'}")
    print("=" * 34)


if __name__ == "__main__":
    main()
