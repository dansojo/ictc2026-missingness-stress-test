"""Leakage-safe secondary S1-S4 evaluation for frozen sensor representations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import re
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .contracts import PRIMITIVE_DEFINITIONS
from .personalization import LosoRepresentationTables, build_loso_representations
from .validation import derive_deterministic_seed, holm_adjust


TARGET_COLUMNS = ("S1", "S2", "S3", "S4")
ARM_NAMES = (
    "calendar",
    "matched_fixed_raw",
    "matched_fixed_personalized",
    "semantic_raw",
    "semantic_personalized",
    "confidence_status_only",
)
MODEL_NAMES = ("elasticnet", "lightgbm")
PRIMARY_BASELINE_ARM = "matched_fixed_personalized"
PRIMARY_CANDIDATE_ARM = "semantic_personalized"
DOWNSTREAM_ARM_NAMES = (PRIMARY_BASELINE_ARM, PRIMARY_CANDIDATE_ARM)
PRIMARY_CONTRAST_ID = f"{PRIMARY_CANDIDATE_ARM}__vs__{PRIMARY_BASELINE_ARM}"
DEFAULT_OUTCOME_BOOTSTRAP_DRAWS = 10_000
INTERVAL_BOUNDARY_POLICIES = (
    "measurement_support",
    "strict_source_availability",
)
BOUNDARY_POLICY_PROVENANCE_COLUMNS = tuple(
    f"{definition.name}__interval_boundary_policy"
    for definition in PRIMITIVE_DEFINITIONS
    if definition.timestamp_semantics == "interval_end"
)
_ARM_SOURCE_POLICY_COLUMN = "__source_interval_boundary_policy"
_ARM_SOURCE_DIGEST_COLUMN = "__source_primitive_digest"
_FRAME_SCHEMA_DIGEST_ALGORITHM = "sha256-json-column-dtype-v1"
_FRAME_CONTENT_DIGEST_ALGORITHM = "sha256-schema-pandas-hash-v1"
_FRAME_MANIFEST_DIGEST_ALGORITHM = "sha256-canonical-frame-seals-v1"
_OUTCOME_BUILD_DIGEST_ALGORITHM = "sha256-trusted-g4-build-v1"
_DOWNSTREAM_REFERENCE_FRAME_SEAL_DOMAIN = (
    "task-12-downstream-reference-frame-seal-v1"
)
_DOWNSTREAM_REFERENCE_TABLES_DOMAIN = (
    "task-12-downstream-reference-frame-tables-v1"
)
PRIMARY_CALIBRATION_DAY = 14
PRIMARY_LAMBDA_DAYS = 7.0
OUTCOME_EVALUATION_START_DAY = 22
SCORE_EPSILON = 1e-6
DATA_INVALID_STATUS = "DATA-INVALID/FIX-AND-RERUN"

MATCHED_SOURCE_PRIMITIVES = tuple(
    definition.name
    for definition in PRIMITIVE_DEFINITIONS
    if definition.name != "screen_active_minutes_24h"
)
SEMANTIC_COLUMNS = (
    "digital_engagement_score",
    "digital_disengagement_time",
    "physical_activity_score",
    "physical_disengagement_time",
    "light_exposure_score",
    "low_light_onset",
    "physiological_settling_score",
    "winddown_center",
    "winddown_spread",
    "routine_deviation",
)
RAW_TIMING_PRIMITIVES = (
    "screen_disengagement_p90",
    "usage_disengagement_p90",
    "phone_activity_disengagement_p90",
    "step_disengagement_p90",
    "mobile_low_light_onset",
    "wearable_low_light_onset",
    "charging_onset",
)

_DOWNSTREAM_REFERENCE_KEY_COLUMNS = (
    "sensor_day_id",
    "subject_id",
    "lifelog_date",
    "sleep_date",
    "elapsed_day_index",
)
_DOWNSTREAM_REFERENCE_KEY_DTYPES = {
    "sensor_day_id": "int64",
    "subject_id": "str",
    "lifelog_date": "datetime64[ns]",
    "sleep_date": "datetime64[ns]",
    "elapsed_day_index": "int64",
}
_DOWNSTREAM_REFERENCE_BASELINE_PRIMITIVES = (
    "step_load_24h",
    "screen_load_24h",
    "phone_activity_load_24h",
    "usage_load_24h",
    "mobile_light_exposure_24h",
    "wearable_light_exposure_24h",
    "screen_disengagement_p90",
    "usage_disengagement_p90",
)
_DOWNSTREAM_REFERENCE_PROVENANCE_COLUMNS = (
    _ARM_SOURCE_POLICY_COLUMN,
    _ARM_SOURCE_DIGEST_COLUMN,
)

_RECIPIENT_AUDIT_COLUMNS = (
    "outer_fold_subject",
    "recipient_subject",
    "recipient_role",
    "population_donors",
    "population_donor_count",
    "personal_calibration_max_elapsed_day",
    "lambda_days",
    "future_rows_used_for_personal_baseline",
    "outer_heldout_used_for_training_recipient_prior",
    "fold",
)
_RECIPIENT_AUDIT_DTYPES = {
    "outer_fold_subject": "str",
    "recipient_subject": "str",
    "recipient_role": "str",
    "population_donors": "object",
    "population_donor_count": "int64",
    "personal_calibration_max_elapsed_day": "int64",
    "lambda_days": "float64",
    "future_rows_used_for_personal_baseline": "bool",
    "outer_heldout_used_for_training_recipient_prior": "bool",
    "fold": "int64",
}
_FOLD_BASELINE_COLUMNS = (
    "outer_fold_subject",
    "recipient_subject",
    "k",
    "method",
    "donor_subject",
    "primitive",
    "center",
    "scale",
    "n_eff",
    "contributing_rows",
    "population_center",
    "population_scale",
    "population_n_eff",
    "population_contributing_rows",
    "calibration_center",
    "calibration_scale",
    "calibration_n_eff",
    "calibration_contributing_rows",
    "center_weight",
    "scale_weight",
    "scale_floor",
    "scale_fallback",
    "baseline_status",
    "lambda_days",
    "fold",
)
_FOLD_BASELINE_DTYPES = {
    "outer_fold_subject": "str",
    "recipient_subject": "str",
    "k": "int64",
    "method": "str",
    "donor_subject": "str",
    "primitive": "str",
    "center": "float64",
    "scale": "float64",
    "n_eff": "float64",
    "contributing_rows": "int64",
    "population_center": "float64",
    "population_scale": "float64",
    "population_n_eff": "float64",
    "population_contributing_rows": "int64",
    "calibration_center": "float64",
    "calibration_scale": "float64",
    "calibration_n_eff": "float64",
    "calibration_contributing_rows": "int64",
    "center_weight": "float64",
    "scale_weight": "float64",
    "scale_floor": "float64",
    "scale_fallback": "str",
    "baseline_status": "str",
    "lambda_days": "float64",
    "fold": "int64",
}

_DIRECT_SEMANTIC_LINEAGE = (
    ("digital_engagement_score", "screen_load_24h", "z", "weighted_mean"),
    ("digital_engagement_score", "usage_load_24h", "z", "weighted_mean"),
    (
        "digital_disengagement_time",
        "screen_disengagement_p90",
        "raw_minute",
        "weighted_median",
    ),
    (
        "digital_disengagement_time",
        "usage_disengagement_p90",
        "raw_minute",
        "weighted_median",
    ),
    ("physical_activity_score", "phone_activity_load_24h", "z", "weighted_mean"),
    ("physical_activity_score", "step_load_24h", "z", "weighted_mean"),
    (
        "physical_disengagement_time",
        "phone_activity_disengagement_p90",
        "raw_minute",
        "weighted_median",
    ),
    (
        "physical_disengagement_time",
        "step_disengagement_p90",
        "raw_minute",
        "weighted_median",
    ),
    ("light_exposure_score", "mobile_light_exposure_24h", "z", "weighted_mean"),
    (
        "light_exposure_score",
        "wearable_light_exposure_24h",
        "z",
        "weighted_mean",
    ),
    ("low_light_onset", "mobile_low_light_onset", "raw_minute", "weighted_median"),
    (
        "low_light_onset",
        "wearable_low_light_onset",
        "raw_minute",
        "weighted_median",
    ),
    (
        "physiological_settling_score",
        "heart_rate_settling_delta",
        "negated_z",
        "weighted_mean",
    ),
    (
        "physiological_settling_score",
        "heart_rate_settling_slope",
        "negated_z",
        "weighted_mean",
    ),
)
_ROUTINE_NEGATED_PRIMITIVES = {
    "heart_rate_settling_delta",
    "heart_rate_settling_slope",
}
SEMANTIC_SOURCE_LINEAGE = (
    _DIRECT_SEMANTIC_LINEAGE
    + tuple(
        (
            "winddown_center",
            primitive,
            "raw_minute",
            "hierarchical_weighted_median",
        )
        for primitive in RAW_TIMING_PRIMITIVES
    )
    + tuple(
        (
            "winddown_spread",
            primitive,
            "raw_minute",
            "hierarchical_weighted_mad",
        )
        for primitive in RAW_TIMING_PRIMITIVES
    )
    + tuple(
        (
            "routine_deviation",
            primitive,
            "negated_z" if primitive in _ROUTINE_NEGATED_PRIMITIVES else "z",
            "weighted_rms",
        )
        for primitive in MATCHED_SOURCE_PRIMITIVES
    )
)

ELASTICNET_PARAMETERS: dict[str, Any] = {
    "solver": "saga",
    "penalty": "elasticnet",
    "l1_ratio": 0.5,
    "C": 1.0,
    "max_iter": 2000,
    "tol": 1e-4,
    "class_weight": None,
}
LIGHTGBM_PARAMETERS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
}

_FIXED_STATUS_VOCABULARY = (
    "observed",
    "derived",
    "insufficient",
    "degenerate",
    "none",
    "left",
    "right",
    "gap",
    "unknown",
    "other",
)


@dataclass(frozen=True)
class LosoFold:
    """One exact participant-held-out outer fold."""

    fold: int
    held_out_subject: Any
    train_positions: np.ndarray
    test_positions: np.ndarray


@dataclass(frozen=True)
class FoldPreprocessor:
    """Outer-training-only numeric transformation and its fitted audit state."""

    train: np.ndarray
    test: np.ndarray
    feature_names: tuple[str, ...]
    train_medians: dict[str, float]
    train_scales: dict[str, float]
    status_vocabulary: tuple[str, ...]
    dropped_constant_columns: tuple[str, ...]


@dataclass(frozen=True)
class FrozenPreprocessorState:
    """Complete train-derived preprocessing authority for later prediction."""

    declared_feature_names: tuple[str, ...]
    output_feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    missing_indicator_names: tuple[str, ...]
    dropped_constant_columns: tuple[str, ...]
    content_digest: str


@dataclass(frozen=True)
class FrozenFoldModel:
    """One non-executable, replayable outer-fold model authority."""

    fold: int
    held_out_subject: str
    arm: str
    model_name: str
    target: str
    declared_feature_names: tuple[str, ...]
    preprocessor_state: FrozenPreprocessorState
    model_state_format: str
    model_state_bytes: bytes
    preprocessor_digest: str
    model_digest: str
    content_digest: str


@dataclass(frozen=True)
class FixedModelPrediction:
    """Predictions plus the frozen fit status needed for row-level auditing."""

    probabilities: np.ndarray
    status: str
    training_count: int
    training_prevalence: float
    feature_count: int
    converged: bool


@dataclass(frozen=True)
class FoldSafeRepresentations:
    """Nested recipient-excluding Task 3 representations for one outer fold."""

    outer_held_out_subject: Any
    representations: pd.DataFrame
    baselines: pd.DataFrame
    recipient_audit: pd.DataFrame


@dataclass(frozen=True)
class OutcomeFrameSeal:
    """Immutable construction-time identity and digest for one fold-arm frame."""

    fold_id: int
    arm_id: str
    held_out_subject: Any
    source_policy: str
    source_primitive_digest: str
    schema_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamReferenceFrameSeal:
    """Immutable identity for one arbitrary-key held-out downstream frame."""

    fold_id: int
    arm_id: str
    held_out_subject: str
    row_count: int
    feature_names: tuple[str, ...]
    prediction_key_digest: str
    source_policy: str
    source_primitive_digest: str
    schema_digest: str
    frame_content_digest: str
    content_digest: str


@dataclass(frozen=True, slots=True)
class DownstreamReferenceFrameTables:
    """Synthetic-safe arbitrary-key reference frames and their full audit seal."""

    prediction_keys: pd.DataFrame
    frames: dict[tuple[int, str], pd.DataFrame]
    frame_manifest: tuple[DownstreamReferenceFrameSeal, ...]
    primitive_basis: pd.DataFrame
    heldout_baselines: pd.DataFrame
    feature_dictionary: pd.DataFrame
    semantic_lineage: pd.DataFrame
    content_digest: str


@dataclass(frozen=True)
class OutcomeArmTables:
    """All fold-specific feature frames with one frozen feature dictionary."""

    evaluation_keys: pd.DataFrame
    frames: dict[tuple[int, str], pd.DataFrame]
    frame_manifest: tuple[OutcomeFrameSeal, ...]
    feature_dictionary: pd.DataFrame
    semantic_lineage: pd.DataFrame
    fold_baselines: pd.DataFrame
    recipient_audit: pd.DataFrame
    manifest: pd.DataFrame


@dataclass(frozen=True)
class OutcomeScoreTables:
    """Row-balanced and participant-target-balanced predictive scores."""

    row_scores: pd.DataFrame
    participant_target_scores: pd.DataFrame
    participant_scores: pd.DataFrame
    target_scores: pd.DataFrame
    macro_scores: pd.DataFrame
    calibration: pd.DataFrame


@dataclass(frozen=True)
class ParticipantDeltaTables:
    """Participant bootstrap, exact sign flip, LOPO and concentration audit."""

    summary: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    lopo: pd.DataFrame


@dataclass(frozen=True)
class OutcomeContrastTables:
    """All fixed arm contrasts and the corrected secondary G4 decision."""

    paired_rows: pd.DataFrame
    participant_target: pd.DataFrame
    participant: pd.DataFrame
    target: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    lopo: pd.DataFrame
    contrast_summary: pd.DataFrame
    target_holm: pd.DataFrame
    decision_table: pd.DataFrame
    g4_summary: pd.DataFrame


@dataclass(frozen=True)
class OutcomeEvaluationResult:
    """OOF predictions, fold preprocessors, scores, contrasts and manifest."""

    predictions: pd.DataFrame
    preprocessor_audit: pd.DataFrame
    scores: OutcomeScoreTables
    contrasts: OutcomeContrastTables
    manifest: pd.DataFrame


@dataclass(frozen=True)
class OutcomeSensitivityResult:
    """Primary and strict-source results that can never replace one another."""

    primary: OutcomeEvaluationResult
    strict_source_availability: OutcomeEvaluationResult
    macro_comparison: pd.DataFrame
    decision_comparison: pd.DataFrame
    manifest: pd.DataFrame


def json_safe_audit_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively normalize audit scalars for strict standard-library JSON."""

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return normalize(value[()])
            return [normalize(item) for item in value]
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                return None
            return pd.Timestamp(value).isoformat()
        if isinstance(value, np.generic):
            return normalize(value.item())
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            return value.isoformat()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    return normalize(payload)


def _primitive_frame_digest(primitives: pd.DataFrame) -> str:
    """Hash the exact primitive schema and sensor-key ordered row values."""
    if "sensor_day_id" not in primitives.columns:
        raise ValueError("DATA-INVALID: primitive provenance lacks sensor_day_id")
    ordered = primitives.sort_values("sensor_day_id", kind="stable").reset_index(
        drop=True
    )
    schema = json.dumps(
        [(str(column), str(dtype)) for column, dtype in ordered.dtypes.items()],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    row_hashes = pd.util.hash_pandas_object(
        ordered,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha256()
    digest.update(schema)
    digest.update(row_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_frame_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or "sensor_day_id" not in frame.columns:
        raise ValueError("DATA-INVALID: frame seal lacks sensor_day_id")
    if frame["sensor_day_id"].isna().any() or frame["sensor_day_id"].duplicated().any():
        raise ValueError("DATA-INVALID: frame seal requires unique sensor_day_id")
    return frame.sort_values("sensor_day_id", kind="stable").reset_index(drop=True)


def _frame_schema_bytes(frame: pd.DataFrame) -> bytes:
    return json.dumps(
        [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _frame_schema_digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(_frame_schema_bytes(frame)).hexdigest()


def _frame_content_digest(frame: pd.DataFrame) -> str:
    """Hash complete values after canonical sensor-key ordering, ignoring attrs."""
    ordered = _canonical_frame_rows(frame)
    row_hashes = pd.util.hash_pandas_object(
        ordered,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha256()
    digest.update(_frame_schema_bytes(ordered))
    digest.update(row_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _frame_manifest_digest(frame_manifest: tuple[OutcomeFrameSeal, ...]) -> str:
    records = [
        {
            "fold_id": seal.fold_id,
            "arm_id": seal.arm_id,
            "held_out_subject_type": (
                f"{type(seal.held_out_subject).__module__}."
                f"{type(seal.held_out_subject).__qualname__}"
            ),
            "held_out_subject": str(seal.held_out_subject),
            "source_policy": seal.source_policy,
            "source_primitive_digest": seal.source_primitive_digest,
            "schema_digest": seal.schema_digest,
            "content_digest": seal.content_digest,
        }
        for seal in frame_manifest
    ]
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_table_rows(
    table: pd.DataFrame,
    *,
    sort_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise ValueError("DATA-INVALID: trusted build table is not a DataFrame")
    missing = sorted(set(sort_columns).difference(table.columns))
    if missing:
        raise ValueError(f"DATA-INVALID: trusted build table lacks keys {missing}")
    return table.sort_values(
        list(sort_columns),
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)


def _table_schema_digest(table: pd.DataFrame) -> str:
    return hashlib.sha256(_frame_schema_bytes(table)).hexdigest()


def _table_content_digest(
    table: pd.DataFrame,
    *,
    sort_columns: tuple[str, ...],
) -> str:
    ordered = _canonical_table_rows(table, sort_columns=sort_columns)
    row_hashes = pd.util.hash_pandas_object(
        ordered,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64, copy=False)
    digest = hashlib.sha256()
    digest.update(_frame_schema_bytes(ordered))
    digest.update(row_hashes.tobytes(order="C"))
    return digest.hexdigest()


def _outcome_build_digest_components(arms: OutcomeArmTables) -> dict[str, str]:
    return {
        "frame_manifest_digest": _frame_manifest_digest(arms.frame_manifest),
        "evaluation_keys_schema_digest": _table_schema_digest(arms.evaluation_keys),
        "evaluation_keys_content_digest": _table_content_digest(
            arms.evaluation_keys,
            sort_columns=("sensor_day_id",),
        ),
        "fold_baselines_schema_digest": _table_schema_digest(arms.fold_baselines),
        "fold_baselines_content_digest": _table_content_digest(
            arms.fold_baselines,
            sort_columns=("fold", "recipient_subject", "method", "primitive"),
        ),
        "recipient_audit_schema_digest": _table_schema_digest(arms.recipient_audit),
        "recipient_audit_content_digest": _table_content_digest(
            arms.recipient_audit,
            sort_columns=("fold", "recipient_subject"),
        ),
    }


def _outcome_build_root_digest(arms: OutcomeArmTables) -> str:
    """Digest the complete internally rebuilt G4 feature/audit artifact."""
    if len(arms.manifest) != 1:
        raise ValueError("DATA-INVALID: trusted G4 build manifest must have one row")
    manifest = arms.manifest.iloc[0]
    required = {"interval_boundary_policy", "primitive_digest"}
    if not required.issubset(arms.manifest.columns):
        raise ValueError("DATA-INVALID: trusted G4 build manifest lacks source identity")
    payload = {
        "algorithm": _OUTCOME_BUILD_DIGEST_ALGORITHM,
        "source_policy": str(manifest["interval_boundary_policy"]),
        "source_primitive_digest": str(manifest["primitive_digest"]),
        **_outcome_build_digest_components(arms),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_primitive_extraction_provenance(
    primitives: pd.DataFrame,
    requested_policy: str,
) -> str:
    missing = sorted(
        set(BOUNDARY_POLICY_PROVENANCE_COLUMNS).difference(primitives.columns)
    )
    if missing:
        raise ValueError(
            "DATA-INVALID: extraction boundary policy provenance is missing "
            f"{missing}"
        )
    for column in BOUNDARY_POLICY_PROVENANCE_COLUMNS:
        values = primitives[column].drop_duplicates()
        if (
            primitives[column].isna().any()
            or len(values) != 1
            or values.iloc[0] != requested_policy
        ):
            raise ValueError(
                "DATA-INVALID: extraction boundary policy provenance disagrees "
                f"with requested policy for {column}"
            )
    return _primitive_frame_digest(primitives)


def _validated_arm_source_provenance(arms: OutcomeArmTables) -> tuple[str, str]:
    required_manifest = {
        "interval_boundary_policy",
        "extraction_interval_boundary_policy",
        "primitive_digest",
    }
    missing = sorted(required_manifest.difference(arms.manifest.columns))
    if missing or len(arms.manifest) != 1:
        raise ValueError(
            "DATA-INVALID: arm source provenance manifest is incomplete or nonunique"
        )
    manifest = arms.manifest.iloc[0]
    policy = manifest["interval_boundary_policy"]
    extraction_policy = manifest["extraction_interval_boundary_policy"]
    digest = manifest["primitive_digest"]
    if (
        type(policy) is not str
        or type(extraction_policy) is not str
        or type(digest) is not str
        or policy not in INTERVAL_BOUNDARY_POLICIES
        or extraction_policy != policy
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError("DATA-INVALID: arm source provenance manifest is forged")
    if not arms.frames:
        raise ValueError("DATA-INVALID: arm source provenance has no frames")
    for key, frame in arms.frames.items():
        missing_frame = {
            _ARM_SOURCE_POLICY_COLUMN,
            _ARM_SOURCE_DIGEST_COLUMN,
        }.difference(frame.columns)
        if missing_frame:
            raise ValueError(
                f"DATA-INVALID: arm source provenance is missing for frame {key}"
            )
        if (
            not frame[_ARM_SOURCE_POLICY_COLUMN].eq(policy).all()
            or not frame[_ARM_SOURCE_DIGEST_COLUMN].eq(digest).all()
        ):
            raise ValueError(
                f"DATA-INVALID: arm source provenance disagrees for frame {key}"
            )
    return policy, digest


def _ordered_subjects(values: pd.Series) -> list[Any]:
    subjects = values.drop_duplicates().tolist()
    return sorted(subjects, key=lambda value: (str(type(value)), str(value)))


def make_loso_splits(rows: pd.DataFrame) -> tuple[LosoFold, ...]:
    """Return the immutable ten-fold LOSO partition in input-row coordinates."""
    required = {"sensor_day_id", "subject_id"}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Missing LOSO columns: {missing}")
    if rows["sensor_day_id"].isna().any() or rows["sensor_day_id"].duplicated().any():
        raise ValueError("sensor_day_id must be nonmissing and unique")
    if rows["subject_id"].isna().any():
        raise ValueError("subject_id must be nonmissing")
    subjects = _ordered_subjects(rows["subject_id"])
    if len(subjects) != 10:
        raise ValueError("Exact G4 LOSO requires exactly 10 participants")
    folds: list[LosoFold] = []
    subject_array = rows["subject_id"].to_numpy()
    for fold, held_out in enumerate(subjects):
        test = np.flatnonzero(subject_array == held_out)
        train = np.flatnonzero(subject_array != held_out)
        if test.size == 0 or train.size == 0:
            raise ValueError("Every LOSO fold must contain nonempty train and test rows")
        folds.append(
            LosoFold(
                fold=fold,
                held_out_subject=held_out,
                train_positions=train,
                test_positions=test,
            )
        )
    return tuple(folds)


def outcome_model_seed(
    root_seed: int,
    model: str,
    target: str,
    fold: int,
    held_out_subject: Any,
) -> int:
    """Return the common arm-invariant seed for one frozen outer-fold fit."""
    return derive_deterministic_seed(
        root_seed,
        "g4_fixed_model",
        model,
        target,
        fold,
        held_out_subject,
    )


def _forbidden_feature_columns(columns: list[Any]) -> list[str]:
    forbidden: list[str] = []
    for value in columns:
        name = str(value)
        separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
        normalized = re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")
        tokens = tuple(token for token in normalized.split("_") if token)
        compact = "".join(tokens)
        reserved_token = any(
            token
            in {
                "subject",
                "participant",
                "donor",
                "recipient",
                "row",
                "fold",
                "oracle",
                "outcome",
                "target",
                "label",
                "probability",
                "probabilities",
                "prediction",
                "predictions",
                "pred",
            }
            for token in tokens
        )
        target_token = any(re.fullmatch(r"[sq]\d+", token) for token in tokens)
        forbidden_compound = compact in {
            "sensordayid",
            "dayindex",
            "lifelogdate",
            "sleepdate",
            "heldout",
            "heldoutsubject",
        }
        reserved_compact_concept = any(
            concept in compact
            for concept in (
                "subject",
                "participant",
                "row",
                "fold",
                "donor",
                "recipient",
                "oracle",
                "outcome",
                "target",
                "probability",
                "prediction",
            )
        )
        compact_target_or_prediction = re.search(
            r"(?:[sq]\d+|pred)",
            compact,
        )
        if (
            reserved_token
            or target_token
            or forbidden_compound
            or reserved_compact_concept
            or compact_target_or_prediction is not None
            or "held_out" in normalized
        ):
            forbidden.append(name)
    return forbidden


def fit_fold_preprocessor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    declared_feature_names: tuple[str, ...],
) -> FoldPreprocessor:
    """Fit finiteness, imputation, missing flags, constants and scaling on train."""
    if train.columns.duplicated().any() or test.columns.duplicated().any():
        raise ValueError("Feature columns must be unique")
    if (
        not isinstance(declared_feature_names, tuple)
        or not declared_feature_names
        or any(not isinstance(name, str) or not name for name in declared_feature_names)
        or len(set(declared_feature_names)) != len(declared_feature_names)
    ):
        raise ValueError("declared_feature_names must be a nonempty unique tuple")
    train_names = tuple(str(column) for column in train.columns)
    test_names = tuple(str(column) for column in test.columns)
    if train_names != declared_feature_names or test_names != declared_feature_names:
        raise ValueError("Feature columns must exactly match the declared allowlist")
    forbidden = sorted(
        set(
            _forbidden_feature_columns(list(train.columns))
            + _forbidden_feature_columns(list(test.columns))
        )
    )
    if forbidden:
        raise ValueError(f"Feature matrix contains forbidden columns: {forbidden}")

    train_numeric = train.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    test_numeric = (
        test.reindex(columns=train.columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    medians = train_numeric.median(axis=0).fillna(0.0)
    train_filled = train_numeric.fillna(medians)
    test_filled = test_numeric.fillna(medians)

    augmented_train: dict[str, np.ndarray] = {}
    augmented_test: dict[str, np.ndarray] = {}
    for column in train_numeric.columns:
        augmented_train[str(column)] = train_filled[column].to_numpy(dtype=float)
        augmented_test[str(column)] = test_filled[column].to_numpy(dtype=float)
        if bool(train_numeric[column].isna().any()):
            missing_name = f"{column}__missing"
            augmented_train[missing_name] = train_numeric[column].isna().to_numpy(dtype=float)
            augmented_test[missing_name] = test_numeric[column].isna().to_numpy(dtype=float)

    kept_names: list[str] = []
    dropped: list[str] = []
    train_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    scales: dict[str, float] = {}
    for name, values in augmented_train.items():
        scale = float(np.std(values, ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            dropped.append(name)
            continue
        center = float(np.mean(values))
        kept_names.append(name)
        scales[name] = scale
        train_columns.append((values - center) / scale)
        test_columns.append((augmented_test[name] - center) / scale)

    if kept_names:
        train_array = np.column_stack(train_columns).astype(float, copy=False)
        test_array = np.column_stack(test_columns).astype(float, copy=False)
    else:
        kept_names = ["constant_fallback"]
        scales = {"constant_fallback": 1.0}
        train_array = np.zeros((len(train), 1), dtype=float)
        test_array = np.zeros((len(test), 1), dtype=float)

    return FoldPreprocessor(
        train=train_array,
        test=test_array,
        feature_names=tuple(kept_names),
        train_medians={str(column): float(value) for column, value in medians.items()},
        train_scales=scales,
        status_vocabulary=_FIXED_STATUS_VOCABULARY,
        dropped_constant_columns=tuple(dropped),
    )


def fit_predict_fixed_model(
    model_name: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    y_train: np.ndarray | pd.Series,
    *,
    seed: int,
) -> FixedModelPrediction:
    """Fit one frozen model without any outer-test feedback."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown frozen model: {model_name}")
    train_array = np.asarray(train_x, dtype=float)
    test_array = np.asarray(test_x, dtype=float)
    if train_array.ndim != 2 or test_array.ndim != 2:
        raise ValueError("train_x and test_x must be two-dimensional")
    if train_array.shape[1] != test_array.shape[1]:
        raise ValueError("train_x and test_x must have identical feature counts")
    if not np.isfinite(train_array).all() or not np.isfinite(test_array).all():
        raise ValueError("Model matrices must be finite after preprocessing")
    y = np.asarray(y_train, dtype=float).reshape(-1)
    if len(y) != len(train_array) or not np.isfinite(y).all():
        raise ValueError("y_train must be finite and align with train_x")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_train must be binary")
    y_integer = y.astype(int)
    training_count = int(len(y_integer))
    if training_count == 0:
        raise ValueError("Training fold cannot be empty")
    prevalence = float((int(y_integer.sum()) + 1.0) / (training_count + 2.0))
    feature_count = int(train_array.shape[1])
    if np.unique(y_integer).size < 2:
        return FixedModelPrediction(
            probabilities=np.full(len(test_array), prevalence, dtype=float),
            status="single_class_smoothed_prevalence",
            training_count=training_count,
            training_prevalence=prevalence,
            feature_count=feature_count,
            converged=True,
        )

    random_state = int(seed) % (2**31 - 1)
    if model_name == "elasticnet":
        model = LogisticRegression(
            **ELASTICNET_PARAMETERS,
            random_state=random_state,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_array, y_integer)
        convergence_warning = any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        )
        probabilities = model.predict_proba(test_array)[:, 1]
        status = "convergence_warning" if convergence_warning else "ok"
        converged = not convergence_warning
    else:
        model = lgb.LGBMClassifier(
            objective="binary",
            **LIGHTGBM_PARAMETERS,
            random_state=random_state,
            verbosity=-1,
        )
        model.fit(train_array, y_integer)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            probabilities = model.predict_proba(test_array)[:, 1]
        status = "ok"
        converged = True

    return FixedModelPrediction(
        probabilities=np.asarray(probabilities, dtype=float),
        status=status,
        training_count=training_count,
        training_prevalence=prevalence,
        feature_count=feature_count,
        converged=converged,
    )


_FROZEN_MODEL_STATE_FORMATS = (
    "elasticnet-coefficients-v1",
    "lightgbm-model-string-v1",
    "single-class-prevalence-v1",
)
_FROZEN_MODEL_SEED_POLICY = "derive-deterministic-seed-frozen-fold-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_frozen_text(value: Any, name: str) -> None:
    if type(value) is not str or not value:
        raise TypeError(f"Frozen model authority {name} must be a nonempty built-in str")


def _validate_declared_frozen_features(
    declared_feature_names: tuple[str, ...],
) -> None:
    if (
        type(declared_feature_names) is not tuple
        or not declared_feature_names
        or any(type(name) is not str or not name for name in declared_feature_names)
        or len(set(declared_feature_names)) != len(declared_feature_names)
    ):
        raise TypeError(
            "Frozen model authority declared_feature_names must be a nonempty "
            "unique built-in tuple of built-in str"
        )
    forbidden = _forbidden_feature_columns(list(declared_feature_names))
    if forbidden:
        raise ValueError(
            f"Frozen model authority contains forbidden feature columns: {forbidden}"
        )


def _numeric_frozen_frame(
    frame: pd.DataFrame,
    declared_feature_names: tuple[str, ...],
    *,
    role: str,
) -> np.ndarray:
    if type(frame) is not pd.DataFrame:
        raise TypeError(f"Frozen model {role} features must be an exact DataFrame")
    if any(type(column) is not str for column in frame.columns):
        raise TypeError(
            f"Frozen model {role} feature columns must be exact built-in str"
        )
    if frame.columns.duplicated().any():
        raise ValueError(f"Frozen model {role} feature columns must be unique")
    if tuple(frame.columns) != declared_feature_names:
        raise ValueError(
            f"Frozen model {role} feature columns must exactly match declared order"
        )
    forbidden = _forbidden_feature_columns(list(frame.columns))
    if forbidden:
        raise ValueError(
            f"Frozen model {role} features contain forbidden columns: {forbidden}"
        )
    for column, dtype in frame.dtypes.items():
        kind = getattr(dtype, "kind", None)
        if kind not in {"i", "u", "f"}:
            raise TypeError(
                f"Frozen model {role} feature {column!r} has coercive dtype {dtype}"
            )
    try:
        values = frame.to_numpy(dtype=float, na_value=np.nan, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Frozen model {role} features must be numeric") from exc
    if np.isinf(values).any():
        raise ValueError(f"Frozen model {role} features contain infinity")
    return values


def _frozen_preprocessor_digest(state: FrozenPreprocessorState) -> str:
    if type(state) is not FrozenPreprocessorState:
        raise TypeError("Frozen preprocessor authority has the wrong exact type")
    return _canonical_digest(
        {
            "algorithm": "sha256-frozen-preprocessor-state-v1",
            "declared_feature_names": list(state.declared_feature_names),
            "output_feature_names": list(state.output_feature_names),
            "medians": list(state.medians),
            "centers": list(state.centers),
            "scales": list(state.scales),
            "missing_indicator_names": list(state.missing_indicator_names),
            "dropped_constant_columns": list(state.dropped_constant_columns),
        }
    )


def _validate_frozen_preprocessor_state(state: FrozenPreprocessorState) -> None:
    if type(state) is not FrozenPreprocessorState:
        raise TypeError("Frozen preprocessor authority has the wrong exact type")
    _validate_declared_frozen_features(state.declared_feature_names)
    tuple_fields = (
        ("output_feature_names", state.output_feature_names, str),
        ("medians", state.medians, float),
        ("centers", state.centers, float),
        ("scales", state.scales, float),
        ("missing_indicator_names", state.missing_indicator_names, str),
        ("dropped_constant_columns", state.dropped_constant_columns, str),
    )
    for name, values, scalar_type in tuple_fields:
        if type(values) is not tuple or any(type(value) is not scalar_type for value in values):
            raise TypeError(
                f"Frozen preprocessor authority {name} has a non-built-in type"
            )
    if (
        not state.output_feature_names
        or len(set(state.output_feature_names)) != len(state.output_feature_names)
        or any(not value for value in state.output_feature_names)
        or len(set(state.missing_indicator_names)) != len(state.missing_indicator_names)
        or len(set(state.dropped_constant_columns))
        != len(state.dropped_constant_columns)
    ):
        raise ValueError("Frozen preprocessor authority topology is invalid")
    if len(state.medians) != len(state.declared_feature_names):
        raise ValueError("Frozen preprocessor authority median count is invalid")
    if len(state.centers) != len(state.output_feature_names) or len(state.scales) != len(
        state.output_feature_names
    ):
        raise ValueError("Frozen preprocessor authority scaling count is invalid")
    if (
        not np.isfinite(np.asarray(state.medians, dtype=float)).all()
        or not np.isfinite(np.asarray(state.centers, dtype=float)).all()
        or not np.isfinite(np.asarray(state.scales, dtype=float)).all()
        or any(scale <= 0.0 for scale in state.scales)
    ):
        raise ValueError("Frozen preprocessor authority statistics are invalid")

    missing_set = set(state.missing_indicator_names)
    expected_missing_names = {
        f"{name}__missing" for name in state.declared_feature_names
    }
    if not missing_set.issubset(expected_missing_names):
        raise ValueError("Frozen preprocessor authority missing topology is invalid")
    augmented_names: list[str] = []
    for name in state.declared_feature_names:
        augmented_names.append(name)
        missing_name = f"{name}__missing"
        if missing_name in missing_set:
            if missing_name in augmented_names:
                raise ValueError("Frozen preprocessor authority topology collides")
            augmented_names.append(missing_name)
    if len(set(augmented_names)) != len(augmented_names):
        raise ValueError("Frozen preprocessor authority topology collides")

    dropped = set(state.dropped_constant_columns)
    if not dropped.issubset(augmented_names):
        raise ValueError("Frozen preprocessor authority constant drops are invalid")
    if state.output_feature_names == ("constant_fallback",):
        if (
            dropped != set(augmented_names)
            or state.centers != (0.0,)
            or state.scales != (1.0,)
        ):
            raise ValueError("Frozen preprocessor authority fallback is invalid")
    else:
        expected_output = tuple(name for name in augmented_names if name not in dropped)
        if state.output_feature_names != expected_output:
            raise ValueError("Frozen preprocessor authority output topology is invalid")
    if type(state.content_digest) is not str or not _SHA256_PATTERN.fullmatch(
        state.content_digest
    ):
        raise ValueError("Frozen preprocessor authority digest is invalid")
    if _frozen_preprocessor_digest(state) != state.content_digest:
        raise ValueError("Frozen preprocessor authority digest mismatch")


def _fit_frozen_preprocessor(
    train_features: pd.DataFrame,
    declared_feature_names: tuple[str, ...],
) -> tuple[FrozenPreprocessorState, np.ndarray]:
    values = _numeric_frozen_frame(
        train_features,
        declared_feature_names,
        role="training",
    )
    if len(values) == 0:
        raise ValueError("Frozen model training fold cannot be empty")
    medians = tuple(
        0.0 if np.isnan(values[:, index]).all() else float(np.nanmedian(values[:, index]))
        for index in range(values.shape[1])
    )
    filled = np.where(np.isnan(values), np.asarray(medians, dtype=float), values)
    missing_indicator_names = tuple(
        f"{name}__missing"
        for index, name in enumerate(declared_feature_names)
        if np.isnan(values[:, index]).any()
    )

    augmented_names: list[str] = []
    augmented_columns: list[np.ndarray] = []
    for index, name in enumerate(declared_feature_names):
        augmented_names.append(name)
        augmented_columns.append(filled[:, index])
        missing_name = f"{name}__missing"
        if missing_name in missing_indicator_names:
            if missing_name in augmented_names:
                raise ValueError("Frozen preprocessor missing-indicator name collides")
            augmented_names.append(missing_name)
            augmented_columns.append(np.isnan(values[:, index]).astype(float))

    output_names: list[str] = []
    dropped: list[str] = []
    centers: list[float] = []
    scales: list[float] = []
    transformed: list[np.ndarray] = []
    for name, column in zip(augmented_names, augmented_columns, strict=True):
        scale = float(np.std(column, ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            dropped.append(name)
            continue
        center = float(np.mean(column))
        output_names.append(name)
        centers.append(center)
        scales.append(scale)
        transformed.append((column - center) / scale)
    if transformed:
        train_array = np.column_stack(transformed).astype(float, copy=False)
    else:
        output_names = ["constant_fallback"]
        centers = [0.0]
        scales = [1.0]
        train_array = np.zeros((len(values), 1), dtype=float)

    unsealed = FrozenPreprocessorState(
        declared_feature_names=declared_feature_names,
        output_feature_names=tuple(output_names),
        medians=medians,
        centers=tuple(centers),
        scales=tuple(scales),
        missing_indicator_names=missing_indicator_names,
        dropped_constant_columns=tuple(dropped),
        content_digest="",
    )
    state = FrozenPreprocessorState(
        declared_feature_names=unsealed.declared_feature_names,
        output_feature_names=unsealed.output_feature_names,
        medians=unsealed.medians,
        centers=unsealed.centers,
        scales=unsealed.scales,
        missing_indicator_names=unsealed.missing_indicator_names,
        dropped_constant_columns=unsealed.dropped_constant_columns,
        content_digest=_frozen_preprocessor_digest(unsealed),
    )
    _validate_frozen_preprocessor_state(state)
    return state, train_array


def _transform_frozen_features(
    state: FrozenPreprocessorState,
    features: pd.DataFrame,
) -> np.ndarray:
    _validate_frozen_preprocessor_state(state)
    values = _numeric_frozen_frame(
        features,
        state.declared_feature_names,
        role="prediction",
    )
    filled = np.where(np.isnan(values), np.asarray(state.medians, dtype=float), values)
    augmented: dict[str, np.ndarray] = {}
    missing_names = set(state.missing_indicator_names)
    for index, name in enumerate(state.declared_feature_names):
        augmented[name] = filled[:, index]
        missing_name = f"{name}__missing"
        if missing_name in missing_names:
            augmented[missing_name] = np.isnan(values[:, index]).astype(float)
    if state.output_feature_names == ("constant_fallback",):
        return np.zeros((len(values), 1), dtype=float)
    output = [
        (augmented[name] - center) / scale
        for name, center, scale in zip(
            state.output_feature_names,
            state.centers,
            state.scales,
            strict=True,
        )
    ]
    return np.column_stack(output).astype(float, copy=False)


def _frozen_fold_model_seed(
    root_seed: int,
    *,
    fold: int,
    held_out_subject: str,
    arm: str,
    model_name: str,
    target: str,
) -> int:
    """Derive the fit seed from the complete downstream fit identity."""
    return derive_deterministic_seed(
        root_seed,
        _FROZEN_MODEL_SEED_POLICY,
        fold,
        held_out_subject,
        arm,
        model_name,
        target,
    )


def _frozen_model_digest(model_state_format: str, model_state_bytes: bytes) -> str:
    return _canonical_digest(
        {
            "algorithm": "sha256-frozen-model-state-v1",
            "model_state_format": model_state_format,
            "model_state_sha256": hashlib.sha256(model_state_bytes).hexdigest(),
        }
    )


def _frozen_fold_content_digest(model: FrozenFoldModel) -> str:
    if type(model) is not FrozenFoldModel:
        raise TypeError("Frozen fold model authority has the wrong exact type")
    return _canonical_digest(
        {
            "algorithm": "sha256-frozen-fold-model-authority-v1",
            "seed_policy": _FROZEN_MODEL_SEED_POLICY,
            "fold": model.fold,
            "held_out_subject": model.held_out_subject,
            "arm": model.arm,
            "model_name": model.model_name,
            "target": model.target,
            "declared_feature_names": list(model.declared_feature_names),
            "preprocessor_digest": model.preprocessor_digest,
            "model_state_format": model.model_state_format,
            "model_digest": model.model_digest,
        }
    )


def _json_model_state(model_state_bytes: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(model_state_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Frozen model state is invalid") from exc
    if type(decoded) is not dict:
        raise ValueError("Frozen model state is invalid")
    return decoded


def _validate_frozen_fold_model(model: FrozenFoldModel) -> Any:
    if type(model) is not FrozenFoldModel:
        raise TypeError("Frozen fold model authority has the wrong exact type")
    if type(model.fold) is not int or model.fold < 0:
        raise TypeError("Frozen fold model authority fold is invalid")
    for name, value in (
        ("held_out_subject", model.held_out_subject),
        ("arm", model.arm),
        ("model_name", model.model_name),
        ("target", model.target),
    ):
        _require_frozen_text(value, name)
    if model.arm not in DOWNSTREAM_ARM_NAMES:
        raise ValueError("Frozen fold model authority arm is invalid")
    if model.model_name not in MODEL_NAMES:
        raise ValueError("Frozen fold model authority model is invalid")
    if model.target not in TARGET_COLUMNS:
        raise ValueError("Frozen fold model authority target is invalid")
    _validate_declared_frozen_features(model.declared_feature_names)
    _validate_frozen_preprocessor_state(model.preprocessor_state)
    if model.preprocessor_state.declared_feature_names != model.declared_feature_names:
        raise ValueError("Frozen fold model authority schema crosswire")
    if (
        type(model.preprocessor_digest) is not str
        or model.preprocessor_digest != model.preprocessor_state.content_digest
    ):
        raise ValueError("Frozen fold model authority preprocessor digest mismatch")
    if type(model.model_state_format) is not str or (
        model.model_state_format not in _FROZEN_MODEL_STATE_FORMATS
    ):
        raise ValueError("Frozen fold model authority state format is unsupported")
    if type(model.model_state_bytes) is not bytes or not model.model_state_bytes:
        raise TypeError("Frozen fold model authority state bytes are invalid")
    expected_format = {
        "elasticnet": "elasticnet-coefficients-v1",
        "lightgbm": "lightgbm-model-string-v1",
    }[model.model_name]
    if model.model_state_format not in {
        expected_format,
        "single-class-prevalence-v1",
    }:
        raise ValueError("Frozen fold model authority model/state crosswire")
    if (
        type(model.model_digest) is not str
        or not _SHA256_PATTERN.fullmatch(model.model_digest)
        or model.model_digest
        != _frozen_model_digest(model.model_state_format, model.model_state_bytes)
    ):
        raise ValueError("Frozen fold model authority model digest mismatch")
    if (
        type(model.content_digest) is not str
        or not _SHA256_PATTERN.fullmatch(model.content_digest)
        or model.content_digest != _frozen_fold_content_digest(model)
    ):
        raise ValueError("Frozen fold model authority content digest mismatch")

    feature_count = len(model.preprocessor_state.output_feature_names)
    if model.model_state_format == "single-class-prevalence-v1":
        state = _json_model_state(model.model_state_bytes)
        if set(state) != {"prevalence"} or type(state["prevalence"]) is not float:
            raise ValueError("Frozen model state is invalid")
        if not np.isfinite(state["prevalence"]) or not 0.0 < state["prevalence"] < 1.0:
            raise ValueError("Frozen model state is invalid")
        return state
    if model.model_state_format == "elasticnet-coefficients-v1":
        state = _json_model_state(model.model_state_bytes)
        if set(state) != {"classes", "coefficients", "intercept"}:
            raise ValueError("Frozen model state is invalid")
        if state["classes"] != [0, 1] or type(state["coefficients"]) is not list:
            raise ValueError("Frozen model state is invalid")
        if len(state["coefficients"]) != feature_count:
            raise ValueError("Frozen model state has the wrong feature count")
        numeric = (*state["coefficients"], state["intercept"])
        if any(type(value) is not float or not np.isfinite(value) for value in numeric):
            raise ValueError("Frozen model state is invalid")
        return state
    try:
        return model.model_state_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Frozen model state is invalid") from exc


def _reconstruct_frozen_model_state(
    model: FrozenFoldModel,
    validated_state: Any,
) -> Any:
    """Construct an estimator only after structural and ledger validation."""
    if model.model_state_format != "lightgbm-model-string-v1":
        return validated_state
    if type(validated_state) is not str or not validated_state:
        raise ValueError("Frozen model state is invalid")
    try:
        booster = lgb.Booster(model_str=validated_state)
    except lgb.basic.LightGBMError as exc:
        raise ValueError("Frozen model state is invalid") from exc
    feature_count = len(model.preprocessor_state.output_feature_names)
    if booster.num_feature() != feature_count:
        raise ValueError("Frozen model state has the wrong feature count")
    return booster


def fit_frozen_fold_model(
    train_features: pd.DataFrame,
    y_train: pd.Series,
    *,
    fold: int,
    held_out_subject: str,
    arm: str,
    model_name: str,
    target: str,
    declared_feature_names: tuple[str, ...],
    root_seed: int,
) -> FrozenFoldModel:
    """Fit once on outer-training rows and seal replayable fitted state."""
    if type(fold) is not int or fold < 0:
        raise TypeError("Frozen model fold must be a nonnegative built-in int")
    if type(root_seed) is not int or root_seed < 0:
        raise TypeError("Frozen model root_seed must be a nonnegative built-in int")
    for name, value in (
        ("held_out_subject", held_out_subject),
        ("arm", arm),
        ("model_name", model_name),
        ("target", target),
    ):
        _require_frozen_text(value, name)
    if arm not in DOWNSTREAM_ARM_NAMES:
        raise ValueError(f"Unknown frozen arm: {arm}")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown frozen model: {model_name}")
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown frozen target: {target}")
    _validate_declared_frozen_features(declared_feature_names)
    if type(y_train) is not pd.Series:
        raise TypeError("Frozen model y_train must be an exact Series")
    if getattr(y_train.dtype, "kind", None) not in {"i", "u", "f"}:
        raise TypeError("Frozen model y_train must have a non-coercive numeric dtype")

    preprocessor, train_array = _fit_frozen_preprocessor(
        train_features,
        declared_feature_names,
    )
    try:
        y = y_train.to_numpy(dtype=float, na_value=np.nan, copy=True).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TypeError("Frozen model y_train must be numeric") from exc
    if len(y) != len(train_array) or not np.isfinite(y).all():
        raise ValueError("Frozen model y_train must be finite and align with training")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Frozen model y_train must be binary")
    y_integer = y.astype(int)
    seed = _frozen_fold_model_seed(
        root_seed,
        fold=fold,
        held_out_subject=held_out_subject,
        arm=arm,
        model_name=model_name,
        target=target,
    )
    prevalence = float((int(y_integer.sum()) + 1.0) / (len(y_integer) + 2.0))
    if np.unique(y_integer).size < 2:
        state_format = "single-class-prevalence-v1"
        state_bytes = json.dumps(
            {"prevalence": prevalence},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    elif model_name == "elasticnet":
        estimator = LogisticRegression(
            **ELASTICNET_PARAMETERS,
            random_state=int(seed) % (2**31 - 1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", FutureWarning)
            estimator.fit(train_array, y_integer)
        state_format = "elasticnet-coefficients-v1"
        state_bytes = json.dumps(
            {
                "classes": [int(value) for value in estimator.classes_],
                "coefficients": [float(value) for value in estimator.coef_[0]],
                "intercept": float(estimator.intercept_[0]),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    else:
        estimator = lgb.LGBMClassifier(
            objective="binary",
            **LIGHTGBM_PARAMETERS,
            random_state=int(seed) % (2**31 - 1),
            verbosity=-1,
        )
        estimator.fit(train_array, y_integer)
        state_format = "lightgbm-model-string-v1"
        state_bytes = estimator.booster_.model_to_string().encode("utf-8")

    model_digest = _frozen_model_digest(state_format, state_bytes)
    unsealed = FrozenFoldModel(
        fold=fold,
        held_out_subject=held_out_subject,
        arm=arm,
        model_name=model_name,
        target=target,
        declared_feature_names=declared_feature_names,
        preprocessor_state=preprocessor,
        model_state_format=state_format,
        model_state_bytes=state_bytes,
        preprocessor_digest=preprocessor.content_digest,
        model_digest=model_digest,
        content_digest="",
    )
    model = FrozenFoldModel(
        fold=unsealed.fold,
        held_out_subject=unsealed.held_out_subject,
        arm=unsealed.arm,
        model_name=unsealed.model_name,
        target=unsealed.target,
        declared_feature_names=unsealed.declared_feature_names,
        preprocessor_state=unsealed.preprocessor_state,
        model_state_format=unsealed.model_state_format,
        model_state_bytes=unsealed.model_state_bytes,
        preprocessor_digest=unsealed.preprocessor_digest,
        model_digest=unsealed.model_digest,
        content_digest=_frozen_fold_content_digest(unsealed),
    )
    validated_state = _validate_frozen_fold_model(model)
    _reconstruct_frozen_model_state(model, validated_state)
    return model


def validate_frozen_fold_model(
    model: FrozenFoldModel,
    *,
    expected_content_digest: str,
) -> None:
    """Validate sealed training state without fitting or predicting."""
    state = _validate_frozen_fold_model(model)
    if (
        type(expected_content_digest) is not str
        or not _SHA256_PATTERN.fullmatch(expected_content_digest)
    ):
        raise TypeError(
            "Frozen model expected content digest must be an exact built-in "
            "lowercase SHA-256 str"
        )
    if expected_content_digest != model.content_digest:
        raise ValueError("Frozen model expected content digest mismatch")
    _reconstruct_frozen_model_state(model, state)


def predict_frozen_fold_model(
    model: FrozenFoldModel,
    test_features: pd.DataFrame,
    *,
    expected_content_digest: str,
) -> np.ndarray:
    """Predict from sealed training state without fitting or topology changes."""
    validate_frozen_fold_model(
        model,
        expected_content_digest=expected_content_digest,
    )
    state = _validate_frozen_fold_model(model)
    state = _reconstruct_frozen_model_state(model, state)
    test_array = _transform_frozen_features(model.preprocessor_state, test_features)
    if len(test_array) == 0:
        return np.empty(0, dtype=float)
    if model.model_state_format == "single-class-prevalence-v1":
        probabilities = np.full(len(test_array), state["prevalence"], dtype=float)
    elif model.model_state_format == "elasticnet-coefficients-v1":
        coefficients = np.asarray(state["coefficients"], dtype=float)
        logits = test_array @ coefficients + float(state["intercept"])
        probabilities = np.empty_like(logits, dtype=float)
        nonnegative = logits >= 0.0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
        exponentials = np.exp(logits[~nonnegative])
        probabilities[~nonnegative] = exponentials / (1.0 + exponentials)
    else:
        probabilities = np.asarray(state.predict(test_array), dtype=float)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if (
        len(probabilities) != len(test_array)
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0.0) | (probabilities > 1.0)).any()
    ):
        raise ValueError("Frozen model state emitted invalid probabilities")
    return probabilities


def _selected_task3_rows(
    tables: LosoRepresentationTables,
    recipients: set[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    representation_mask = tables.representations["held_out_subject"].isin(recipients) & (
        (tables.representations["method"].eq("population") & tables.representations["k"].eq(0))
        | (
            tables.representations["method"].eq("shrinkage")
            & tables.representations["k"].eq(PRIMARY_CALIBRATION_DAY)
        )
    )
    baseline_mask = tables.baselines["held_out_subject"].isin(recipients) & (
        (tables.baselines["method"].eq("population") & tables.baselines["k"].eq(0))
        | (
            tables.baselines["method"].eq("shrinkage")
            & tables.baselines["k"].eq(PRIMARY_CALIBRATION_DAY)
        )
    )
    return (
        tables.representations.loc[representation_mask].copy(),
        tables.baselines.loc[baseline_mask].copy(),
    )


def _combine_fold_representations(
    primitives: pd.DataFrame,
    outer_held_out_subject: Any,
    full_tables: LosoRepresentationTables,
) -> FoldSafeRepresentations:
    subjects = _ordered_subjects(primitives["subject_id"])
    if outer_held_out_subject not in subjects:
        raise ValueError("outer_held_out_subject is absent from the sensor calendar")
    outer_training_subjects = [
        subject for subject in subjects if subject != outer_held_out_subject
    ]
    if len(outer_training_subjects) < 2:
        raise ValueError("Nested recipient exclusion requires at least three participants")
    outer_training_primitives = primitives.loc[
        primitives["subject_id"].isin(outer_training_subjects)
    ].copy()
    nested_tables = build_loso_representations(
        outer_training_primitives,
        k_values=(PRIMARY_CALIBRATION_DAY,),
        lambda_days=PRIMARY_LAMBDA_DAYS,
    )

    test_representations, test_baselines = _selected_task3_rows(
        full_tables, {outer_held_out_subject}
    )
    train_representations, train_baselines = _selected_task3_rows(
        nested_tables, set(outer_training_subjects)
    )
    representations = pd.concat(
        [train_representations, test_representations], ignore_index=True
    ).sort_values(
        ["method", "subject_id", "sensor_day_id"], kind="stable"
    ).reset_index(drop=True)
    baselines = pd.concat([train_baselines, test_baselines], ignore_index=True)
    baselines = baselines.rename(columns={"held_out_subject": "recipient_subject"})
    baselines.insert(0, "outer_fold_subject", outer_held_out_subject)
    baselines = baselines.sort_values(
        ["recipient_subject", "method", "primitive"], kind="stable"
    ).reset_index(drop=True)

    audit_records: list[dict[str, Any]] = []
    for recipient in subjects:
        if recipient == outer_held_out_subject:
            donors = [subject for subject in subjects if subject != recipient]
            role = "outer_test"
        else:
            donors = [
                subject
                for subject in outer_training_subjects
                if subject != recipient
            ]
            role = "outer_train"
        audit_records.append(
            {
                "outer_fold_subject": outer_held_out_subject,
                "recipient_subject": recipient,
                "recipient_role": role,
                "population_donors": tuple(donors),
                "population_donor_count": len(donors),
                "personal_calibration_max_elapsed_day": PRIMARY_CALIBRATION_DAY,
                "lambda_days": PRIMARY_LAMBDA_DAYS,
                "future_rows_used_for_personal_baseline": False,
                "outer_heldout_used_for_training_recipient_prior": False,
            }
        )

    expected_methods = {"population", "shrinkage"}
    if set(representations["method"].unique()) != expected_methods:
        raise RuntimeError("Fold-safe representation methods are incomplete")
    if representations.duplicated(["sensor_day_id", "method"]).any():
        raise RuntimeError("Fold-safe representation keys are not unique")
    return FoldSafeRepresentations(
        outer_held_out_subject=outer_held_out_subject,
        representations=representations,
        baselines=baselines,
        recipient_audit=pd.DataFrame.from_records(audit_records),
    )


def build_fold_safe_representations(
    primitives: pd.DataFrame,
    *,
    outer_held_out_subject: Any,
) -> FoldSafeRepresentations:
    """Rebuild Task 3 priors so the outer heldout is absent from train priors."""
    full_tables = build_loso_representations(
        primitives,
        k_values=(PRIMARY_CALIBRATION_DAY,),
        lambda_days=PRIMARY_LAMBDA_DAYS,
    )
    return _combine_fold_representations(
        primitives,
        outer_held_out_subject,
        full_tables,
    )


def _validated_midnight(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if getattr(parsed.dt, "tz", None) is not None:
        raise ValueError(f"{name} provenance must be timezone-naive")
    if parsed.isna().any() or not parsed.eq(parsed.dt.normalize()).all():
        raise ValueError(f"{name} provenance must contain exact local dates")
    return parsed


def join_outcome_labels(
    sensor_keys: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    require_real_contract: bool = True,
) -> pd.DataFrame:
    """Join binary outcomes onto immutable sensor keys and then select day 22+."""
    key_columns = [
        "sensor_day_id",
        "subject_id",
        "lifelog_date",
        "sleep_date",
        "elapsed_day_index",
    ]
    missing_sensor = sorted(set(key_columns).difference(sensor_keys.columns))
    label_key_columns = ["subject_id", "lifelog_date", "sleep_date"]
    missing_labels = sorted(
        set(label_key_columns + list(TARGET_COLUMNS)).difference(labels.columns)
    )
    if missing_sensor:
        raise ValueError(f"Missing frozen sensor provenance columns: {missing_sensor}")
    if missing_labels:
        raise ValueError(f"Missing outcome label columns: {missing_labels}")

    sensor = sensor_keys.loc[:, key_columns].copy()
    if sensor["sensor_day_id"].isna().any() or sensor["sensor_day_id"].duplicated().any():
        raise ValueError("Frozen sensor_day_id provenance must be unique")
    if sensor.duplicated(label_key_columns).any():
        raise ValueError("Frozen participant-date provenance must be unique")
    outcomes = labels.copy()
    for frame in (sensor, outcomes):
        frame["lifelog_date"] = _validated_midnight(frame["lifelog_date"], "lifelog_date")
        frame["sleep_date"] = _validated_midnight(frame["sleep_date"], "sleep_date")
        if not (frame["sleep_date"] - frame["lifelog_date"]).eq(
            pd.Timedelta(days=1)
        ).all():
            raise ValueError("sleep_date provenance must be exactly lifelog_date plus one day")
    if outcomes.duplicated(label_key_columns).any():
        raise ValueError("Outcome participant-date provenance must be unique")

    provided_sensor_id: pd.Series | None = None
    if "sensor_day_id" in outcomes.columns:
        provided_sensor_id = pd.to_numeric(outcomes.pop("sensor_day_id"), errors="coerce")
    provided_elapsed: pd.Series | None = None
    if "elapsed_day_index" in outcomes.columns:
        provided_elapsed = pd.to_numeric(
            outcomes.pop("elapsed_day_index"), errors="coerce"
        )
    for target in TARGET_COLUMNS:
        numeric = pd.to_numeric(outcomes[target], errors="coerce")
        if numeric.isna().any() or not numeric.isin((0, 1)).all():
            raise ValueError(f"{target} must be an exact binary outcome")
        outcomes[target] = numeric.astype(int)

    joined = outcomes[label_key_columns + list(TARGET_COLUMNS)].merge(
        sensor,
        on=label_key_columns,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Outcome dates do not match frozen sensor provenance")
    joined = joined.drop(columns="_merge")
    if provided_sensor_id is not None:
        expected_ids = pd.to_numeric(joined["sensor_day_id"], errors="coerce")
        provided_array = provided_sensor_id.to_numpy(dtype=float, na_value=np.nan)
        if not np.array_equal(provided_array, expected_ids.to_numpy(dtype=float)):
            raise ValueError("Provided sensor_day_id does not match frozen provenance")
    if provided_elapsed is not None:
        provided_array = provided_elapsed.to_numpy(dtype=float, na_value=np.nan)
        expected_array = pd.to_numeric(joined["elapsed_day_index"]).to_numpy(dtype=float)
        if not np.array_equal(provided_array, expected_array):
            raise ValueError("Provided elapsed_day_index does not match sensor provenance")

    evaluation = joined.loc[
        pd.to_numeric(joined["elapsed_day_index"], errors="coerce").ge(
            OUTCOME_EVALUATION_START_DAY
        )
    ].copy()
    evaluation = evaluation.sort_values("sensor_day_id", kind="stable").reset_index(drop=True)
    if require_real_contract:
        if len(labels) != 450:
            raise ValueError("Canonical G4 requires exactly 450 outcome labels")
        if len(evaluation) != 275:
            raise ValueError("Canonical G4 requires exactly 275 post-day21 labeled rows")
        if evaluation["subject_id"].nunique() != 10:
            raise ValueError("Canonical G4 requires exactly 10 participants")
    output_columns = key_columns + list(TARGET_COLUMNS)
    return evaluation.loc[:, output_columns]


_PRIMITIVE_STATUS_VOCABULARY = ("observed", "insufficient", "other")
_CENSORING_VOCABULARY = ("none", "left", "right", "gap", "unknown", "other")
_SEMANTIC_STATUS_VOCABULARY = ("derived", "insufficient", "other")


def _category_block(
    values: pd.Series,
    prefix: str,
    vocabulary: tuple[str, ...],
) -> dict[str, np.ndarray]:
    normalized = values.astype("string").fillna("unknown").str.lower()
    known = set(vocabulary).difference({"other"})
    result: dict[str, np.ndarray] = {}
    for category in vocabulary:
        if category == "other":
            indicator = ~normalized.isin(known)
        else:
            indicator = normalized.eq(category)
        result[f"{prefix}__{category}"] = indicator.to_numpy(dtype=float)
    return result


def _feature_record(
    arm: str,
    feature: str,
    *,
    source: str,
    role: str,
    behavioral_value: bool,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "feature": feature,
        "source": source,
        "role": role,
        "behavioral_value": bool(behavioral_value),
        "calendar_base_included": True,
        "fixed_before_outcomes": True,
    }


def _calendar_block(keys: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    dates = pd.to_datetime(keys["lifelog_date"])
    angle = 2.0 * np.pi * dates.dt.weekday.to_numpy(dtype=float) / 7.0
    data = {
        "calendar__weekday_sin": np.sin(angle),
        "calendar__weekday_cos": np.cos(angle),
        "calendar__elapsed_day_trend": pd.to_numeric(
            keys["elapsed_day_index"], errors="coerce"
        ).to_numpy(dtype=float),
    }
    records = [
        _feature_record(
            "calendar", feature, source="calendar", role="calendar", behavioral_value=False
        )
        for feature in data
    ]
    return pd.DataFrame(data), records


def _quality_block(
    representation: pd.DataFrame,
    primitive_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    data: dict[str, np.ndarray] = {}
    metadata: list[tuple[str, str, str]] = []
    for primitive in MATCHED_SOURCE_PRIMITIVES:
        numeric_sources = (
            (f"{primitive}__confidence", "quality_confidence"),
            (f"{primitive}__coverage_only_confidence", "coverage_only_confidence"),
            (f"{primitive}__value_only_confidence", "value_only_confidence"),
        )
        for source_column, suffix in numeric_sources:
            feature = f"quality__{primitive}__{suffix}"
            data[feature] = pd.to_numeric(
                representation[source_column], errors="coerce"
            ).to_numpy(dtype=float)
            metadata.append((feature, primitive, "quality"))
        for suffix in (
            "availability_coverage",
            "longest_gap_minutes",
            "support_density",
        ):
            feature = f"quality__{primitive}__{suffix}"
            data[feature] = pd.to_numeric(
                primitive_rows[f"{primitive}__{suffix}"], errors="coerce"
            ).to_numpy(dtype=float)
            metadata.append((feature, primitive, "quality"))
        status_features = _category_block(
            primitive_rows[f"{primitive}__status"],
            f"status__{primitive}",
            _PRIMITIVE_STATUS_VOCABULARY,
        )
        for feature, values in status_features.items():
            data[feature] = values
            metadata.append((feature, primitive, "status"))
        censoring_column = f"{primitive}__censoring"
        censoring = (
            primitive_rows[censoring_column]
            if censoring_column in primitive_rows.columns
            else pd.Series("unknown", index=primitive_rows.index)
        )
        censoring_features = _category_block(
            censoring,
            f"censoring__{primitive}",
            _CENSORING_VOCABULARY,
        )
        for feature, values in censoring_features.items():
            data[feature] = values
            metadata.append((feature, primitive, "status"))
    return pd.DataFrame(data), metadata


def _semantic_block(
    representation: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str, str, bool]]]:
    data: dict[str, np.ndarray] = {}
    metadata: list[tuple[str, str, str, bool]] = []
    for semantic in SEMANTIC_COLUMNS:
        value_feature = f"semantic__{semantic}__value"
        data[value_feature] = pd.to_numeric(
            representation[semantic], errors="coerce"
        ).to_numpy(dtype=float)
        metadata.append((value_feature, semantic, "semantic_value", True))
        for suffix in ("confidence", "modalities"):
            feature = f"semantic__{semantic}__{suffix}"
            data[feature] = pd.to_numeric(
                representation[f"{semantic}__{suffix}"], errors="coerce"
            ).to_numpy(dtype=float)
            metadata.append((feature, semantic, "quality", False))
        status_features = _category_block(
            representation[f"{semantic}__status"],
            f"semantic_status__{semantic}",
            _SEMANTIC_STATUS_VOCABULARY,
        )
        for feature, values in status_features.items():
            data[feature] = values
            metadata.append((feature, semantic, "status", False))
    return pd.DataFrame(data), metadata


def _arm_frames_for_fold(
    bundle: FoldSafeRepresentations,
    primitives: pd.DataFrame,
    evaluation_keys: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    key_columns = [
        "sensor_day_id",
        "subject_id",
        "lifelog_date",
        "sleep_date",
        "elapsed_day_index",
    ]
    sensor_ids = evaluation_keys["sensor_day_id"]
    primitive_lookup = primitives.set_index("sensor_day_id", drop=False)
    primitive_rows = primitive_lookup.reindex(sensor_ids.to_numpy()).reset_index(drop=True)
    method_frames: dict[str, pd.DataFrame] = {}
    for method in ("population", "shrinkage"):
        lookup = bundle.representations.loc[
            bundle.representations["method"].eq(method)
        ].set_index("sensor_day_id", drop=False)
        rows = lookup.reindex(sensor_ids.to_numpy()).reset_index(drop=True)
        if rows["sensor_day_id"].isna().any():
            raise ValueError("Fold representation is missing a labeled sensor_day_id")
        method_frames[method] = rows

    calendar, calendar_records = _calendar_block(evaluation_keys)
    quality, quality_metadata = _quality_block(method_frames["population"], primitive_rows)
    raw_values = pd.DataFrame(
        {
            f"primitive__{primitive}__raw_value": pd.to_numeric(
                primitive_rows[primitive], errors="coerce"
            ).to_numpy(dtype=float)
            for primitive in MATCHED_SOURCE_PRIMITIVES
        }
    )
    personalized_values = pd.DataFrame(
        {
            f"primitive__{primitive}__personalized_z": pd.to_numeric(
                method_frames["shrinkage"][f"{primitive}__z"], errors="coerce"
            ).to_numpy(dtype=float)
            for primitive in MATCHED_SOURCE_PRIMITIVES
        }
    )
    personalized_timing = pd.DataFrame(
        {
            f"primitive__{primitive}__raw_minute": pd.to_numeric(
                method_frames["shrinkage"][primitive], errors="coerce"
            ).to_numpy(dtype=float)
            for primitive in RAW_TIMING_PRIMITIVES
        }
    )
    semantic_raw, semantic_raw_metadata = _semantic_block(method_frames["population"])
    semantic_personalized, semantic_personalized_metadata = _semantic_block(
        method_frames["shrinkage"]
    )

    feature_blocks = {
        "calendar": [calendar],
        "matched_fixed_raw": [calendar, raw_values, quality],
        "matched_fixed_personalized": [
            calendar,
            personalized_values,
            personalized_timing,
            quality,
        ],
        "semantic_raw": [calendar, semantic_raw],
        "semantic_personalized": [calendar, semantic_personalized],
        "confidence_status_only": [calendar, quality],
    }
    frames: dict[str, pd.DataFrame] = {}
    for arm in ARM_NAMES:
        values = pd.concat(feature_blocks[arm], axis=1)
        frames[arm] = pd.concat(
            [evaluation_keys[key_columns].reset_index(drop=True), values], axis=1
        )

    dictionary_records: list[dict[str, Any]] = []
    calendar_features = [record["feature"] for record in calendar_records]
    for arm in ARM_NAMES:
        for feature in calendar_features:
            dictionary_records.append(
                _feature_record(
                    arm,
                    feature,
                    source="calendar",
                    role="calendar",
                    behavioral_value=False,
                )
            )
    for arm in ("matched_fixed_raw", "matched_fixed_personalized", "confidence_status_only"):
        for feature, source, role in quality_metadata:
            dictionary_records.append(
                _feature_record(
                    arm,
                    feature,
                    source=source,
                    role=role,
                    behavioral_value=False,
                )
            )
    for primitive in MATCHED_SOURCE_PRIMITIVES:
        dictionary_records.append(
            _feature_record(
                "matched_fixed_raw",
                f"primitive__{primitive}__raw_value",
                source=primitive,
                role="primitive_value",
                behavioral_value=True,
            )
        )
        dictionary_records.append(
            _feature_record(
                "matched_fixed_personalized",
                f"primitive__{primitive}__personalized_z",
                source=primitive,
                role="primitive_z",
                behavioral_value=True,
            )
        )
    for primitive in RAW_TIMING_PRIMITIVES:
        dictionary_records.append(
            _feature_record(
                "matched_fixed_personalized",
                f"primitive__{primitive}__raw_minute",
                source=primitive,
                role="timing_raw",
                behavioral_value=True,
            )
        )
    for arm, metadata in (
        ("semantic_raw", semantic_raw_metadata),
        ("semantic_personalized", semantic_personalized_metadata),
    ):
        for feature, source, role, behavioral in metadata:
            dictionary_records.append(
                _feature_record(
                    arm,
                    feature,
                    source=source,
                    role=role,
                    behavioral_value=behavioral,
                )
            )
    dictionary = pd.DataFrame.from_records(dictionary_records)
    return frames, dictionary


def _expected_production_feature_dictionary() -> pd.DataFrame:
    """Derive the frozen dictionary independently from immutable contracts."""
    calendar_features = (
        "calendar__weekday_sin",
        "calendar__weekday_cos",
        "calendar__elapsed_day_trend",
    )
    records: list[dict[str, Any]] = []
    for arm in ARM_NAMES:
        for feature in calendar_features:
            records.append(
                _feature_record(
                    arm,
                    feature,
                    source="calendar",
                    role="calendar",
                    behavioral_value=False,
                )
            )

    quality_records: list[tuple[str, str, str]] = []
    for primitive in MATCHED_SOURCE_PRIMITIVES:
        for suffix in (
            "quality_confidence",
            "coverage_only_confidence",
            "value_only_confidence",
            "availability_coverage",
            "longest_gap_minutes",
            "support_density",
        ):
            quality_records.append(
                (f"quality__{primitive}__{suffix}", primitive, "quality")
            )
        for category in _PRIMITIVE_STATUS_VOCABULARY:
            quality_records.append(
                (f"status__{primitive}__{category}", primitive, "status")
            )
        for category in _CENSORING_VOCABULARY:
            quality_records.append(
                (f"censoring__{primitive}__{category}", primitive, "status")
            )
    for arm in (
        "matched_fixed_raw",
        "matched_fixed_personalized",
        "confidence_status_only",
    ):
        for feature, source, role in quality_records:
            records.append(
                _feature_record(
                    arm,
                    feature,
                    source=source,
                    role=role,
                    behavioral_value=False,
                )
            )

    for primitive in MATCHED_SOURCE_PRIMITIVES:
        records.append(
            _feature_record(
                "matched_fixed_raw",
                f"primitive__{primitive}__raw_value",
                source=primitive,
                role="primitive_value",
                behavioral_value=True,
            )
        )
        records.append(
            _feature_record(
                "matched_fixed_personalized",
                f"primitive__{primitive}__personalized_z",
                source=primitive,
                role="primitive_z",
                behavioral_value=True,
            )
        )
    for primitive in RAW_TIMING_PRIMITIVES:
        records.append(
            _feature_record(
                "matched_fixed_personalized",
                f"primitive__{primitive}__raw_minute",
                source=primitive,
                role="timing_raw",
                behavioral_value=True,
            )
        )
    for arm in ("semantic_raw", "semantic_personalized"):
        for semantic in SEMANTIC_COLUMNS:
            records.append(
                _feature_record(
                    arm,
                    f"semantic__{semantic}__value",
                    source=semantic,
                    role="semantic_value",
                    behavioral_value=True,
                )
            )
            for suffix in ("confidence", "modalities"):
                records.append(
                    _feature_record(
                        arm,
                        f"semantic__{semantic}__{suffix}",
                        source=semantic,
                        role="quality",
                        behavioral_value=False,
                    )
                )
            for category in _SEMANTIC_STATUS_VOCABULARY:
                records.append(
                    _feature_record(
                        arm,
                        f"semantic_status__{semantic}__{category}",
                        source=semantic,
                        role="status",
                        behavioral_value=False,
                    )
                )
    return pd.DataFrame.from_records(records)


def _expected_outer_fold_subjects(arms: OutcomeArmTables) -> dict[int, Any]:
    try:
        folds = make_loso_splits(arms.evaluation_keys)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "DATA-INVALID: outer-fold heldout identity cannot be reconstructed"
        ) from error
    return {fold.fold: fold.held_out_subject for fold in folds}


def _validate_exact_table_dtypes(
    table: pd.DataFrame,
    expected: dict[str, str],
    *,
    label: str,
) -> None:
    observed = {str(column): str(dtype) for column, dtype in table.dtypes.items()}
    if observed != expected:
        raise ValueError(
            f"DATA-INVALID: {label} dtype contract mismatch; "
            f"expected={expected}, observed={observed}"
        )


def _validate_outer_fold_exclusion(
    arms: OutcomeArmTables,
    expected_heldout: dict[int, Any],
) -> None:
    audit = arms.recipient_audit
    subjects = _ordered_subjects(arms.evaluation_keys["subject_id"])
    expected_keys = {
        (fold, recipient) for fold in expected_heldout for recipient in subjects
    }
    if (
        not isinstance(audit, pd.DataFrame)
        or tuple(audit.columns) != _RECIPIENT_AUDIT_COLUMNS
        or len(audit) != len(expected_keys)
        or audit.duplicated(["fold", "recipient_subject"]).any()
    ):
        raise ValueError("DATA-INVALID: outer-fold heldout exclusion audit is incomplete")
    _validate_exact_table_dtypes(
        audit,
        _RECIPIENT_AUDIT_DTYPES,
        label="recipient audit",
    )
    actual_keys = set(zip(audit["fold"], audit["recipient_subject"], strict=True))
    if actual_keys != expected_keys:
        raise ValueError("DATA-INVALID: outer-fold heldout exclusion keys mismatch")

    for row in audit.itertuples(index=False):
        fold = row.fold
        recipient = row.recipient_subject
        if fold not in expected_heldout:
            raise ValueError("DATA-INVALID: outer-fold heldout exclusion has unknown fold")
        heldout = expected_heldout[fold]
        expected_role = "outer_test" if recipient == heldout else "outer_train"
        expected_donors = tuple(
            subject
            for subject in subjects
            if subject != recipient and (recipient == heldout or subject != heldout)
        )
        exact_false_flags = all(
            type(value) is bool and not value
            for value in (
                row.future_rows_used_for_personal_baseline,
                row.outer_heldout_used_for_training_recipient_prior,
            )
        )
        if (
            type(fold) is not int
            or type(row.recipient_role) is not str
            or type(row.population_donor_count) is not int
            or type(row.personal_calibration_max_elapsed_day) is not int
            or type(row.lambda_days) is not float
            or type(row.outer_fold_subject) is not type(heldout)
            or row.outer_fold_subject != heldout
            or row.recipient_role != expected_role
            or not isinstance(row.population_donors, tuple)
            or row.population_donors != expected_donors
            or row.population_donor_count != len(expected_donors)
            or row.personal_calibration_max_elapsed_day != PRIMARY_CALIBRATION_DAY
            or row.lambda_days != PRIMARY_LAMBDA_DAYS
            or not exact_false_flags
        ):
            raise ValueError("DATA-INVALID: outer-fold heldout exclusion was forged")

    baselines = arms.fold_baselines
    if (
        not isinstance(baselines, pd.DataFrame)
        or tuple(baselines.columns) != _FOLD_BASELINE_COLUMNS
    ):
        raise ValueError("DATA-INVALID: fold baseline schema is incomplete")
    _validate_exact_table_dtypes(
        baselines,
        _FOLD_BASELINE_DTYPES,
        label="fold baseline",
    )
    primitives = tuple(definition.name for definition in PRIMITIVE_DEFINITIONS)
    expected_baseline_keys = {
        (fold, recipient, method, k, primitive)
        for fold in expected_heldout
        for recipient in subjects
        for method, k in (
            ("population", 0),
            ("shrinkage", PRIMARY_CALIBRATION_DAY),
        )
        for primitive in primitives
    }
    key_columns = ["fold", "recipient_subject", "method", "k", "primitive"]
    actual_baseline_keys = set(
        baselines.loc[:, key_columns].itertuples(index=False, name=None)
    )
    if (
        len(baselines) != len(expected_baseline_keys)
        or baselines.duplicated(key_columns).any()
        or actual_baseline_keys != expected_baseline_keys
        or not baselines["donor_subject"].isna().all()
        or not baselines["lambda_days"].eq(PRIMARY_LAMBDA_DAYS).all()
    ):
        raise ValueError("DATA-INVALID: fold baseline topology is incomplete or duplicated")
    for row in baselines.itertuples(index=False):
        heldout = expected_heldout.get(row.fold)
        if (
            type(row.fold) is not int
            or type(row.k) is not int
            or type(row.method) is not str
            or type(row.primitive) is not str
            or heldout is None
            or type(row.outer_fold_subject) is not type(heldout)
            or row.outer_fold_subject != heldout
        ):
            raise ValueError("DATA-INVALID: fold baseline identity was forged")


def _validate_frame_manifest(
    arms: OutcomeArmTables,
    *,
    trusted_build_root: str,
) -> None:
    if (
        type(trusted_build_root) is not str
        or re.fullmatch(r"[0-9a-f]{64}", trusted_build_root) is None
    ):
        raise ValueError("DATA-INVALID: trusted G4 build root is not exact")
    expected_sequence = tuple(
        (fold, arm) for fold in range(10) for arm in ARM_NAMES
    )
    frame_manifest = arms.frame_manifest
    if (
        not isinstance(frame_manifest, tuple)
        or len(frame_manifest) != len(expected_sequence)
        or any(type(seal) is not OutcomeFrameSeal for seal in frame_manifest)
    ):
        raise ValueError("DATA-INVALID: frame manifest identity is incomplete or forged")
    if any(
        type(seal.fold_id) is not int
        or type(seal.arm_id) is not str
        or type(seal.source_policy) is not str
        or type(seal.source_primitive_digest) is not str
        or type(seal.schema_digest) is not str
        or type(seal.content_digest) is not str
        for seal in frame_manifest
    ):
        raise ValueError("DATA-INVALID: frame manifest field types are forged")
    if (
        tuple((seal.fold_id, seal.arm_id) for seal in frame_manifest)
        != expected_sequence
    ):
        raise ValueError("DATA-INVALID: frame manifest identity is incomplete or forged")
    if not isinstance(arms.frames, dict) or any(
        type(key) is not tuple
        or len(key) != 2
        or type(key[0]) is not int
        or type(key[1]) is not str
        for key in arms.frames
    ):
        raise ValueError("DATA-INVALID: frame dictionary key types are forged")
    if set(arms.frames) != set(expected_sequence):
        raise ValueError("DATA-INVALID: frame manifest keys mismatch frame dictionary")

    required_global = {
        "frame_manifest_rows",
        "frame_schema_digest_algorithm",
        "frame_content_digest_algorithm",
        "frame_manifest_digest_algorithm",
        "frame_manifest_digest",
        "evaluation_keys_schema_digest",
        "evaluation_keys_content_digest",
        "fold_baselines_schema_digest",
        "fold_baselines_content_digest",
        "recipient_audit_schema_digest",
        "recipient_audit_content_digest",
        "outcome_build_digest_algorithm",
        "outcome_build_digest",
    }
    if len(arms.manifest) != 1 or not required_global.issubset(arms.manifest.columns):
        raise ValueError("DATA-INVALID: frame manifest root seal is incomplete")
    global_manifest = arms.manifest.iloc[0]
    if (
        global_manifest["frame_manifest_rows"] != len(expected_sequence)
        or global_manifest["frame_schema_digest_algorithm"]
        != _FRAME_SCHEMA_DIGEST_ALGORITHM
        or global_manifest["frame_content_digest_algorithm"]
        != _FRAME_CONTENT_DIGEST_ALGORITHM
        or global_manifest["frame_manifest_digest_algorithm"]
        != _FRAME_MANIFEST_DIGEST_ALGORITHM
        or global_manifest["frame_manifest_digest"]
        != _frame_manifest_digest(frame_manifest)
        or global_manifest["outcome_build_digest_algorithm"]
        != _OUTCOME_BUILD_DIGEST_ALGORITHM
    ):
        raise ValueError("DATA-INVALID: frame manifest root seal is forged")

    policy, primitive_digest = _validated_arm_source_provenance(arms)
    expected_heldout = _expected_outer_fold_subjects(arms)
    _validate_outer_fold_exclusion(arms, expected_heldout)
    components = _outcome_build_digest_components(arms)
    if any(
        type(global_manifest[name]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", global_manifest[name]) is None
        or global_manifest[name] != digest
        for name, digest in components.items()
    ):
        raise ValueError("DATA-INVALID: trusted G4 build component digest mismatch")
    rebuilt_root = _outcome_build_root_digest(arms)
    if (
        type(global_manifest["outcome_build_digest"]) is not str
        or global_manifest["outcome_build_digest"] != rebuilt_root
        or rebuilt_root != trusted_build_root
    ):
        raise ValueError("DATA-INVALID: trusted G4 build root mismatch")
    key_columns = [
        "sensor_day_id",
        "subject_id",
        "lifelog_date",
        "sleep_date",
        "elapsed_day_index",
    ]
    canonical_keys = _canonical_frame_rows(arms.evaluation_keys)[key_columns]
    for seal in frame_manifest:
        key = (seal.fold_id, seal.arm_id)
        heldout = expected_heldout[seal.fold_id]
        frame = arms.frames[key]
        if (
            type(seal.held_out_subject) is not type(heldout)
            or seal.held_out_subject != heldout
        ):
            raise ValueError("DATA-INVALID: frame manifest heldout identity mismatch")
        if (
            seal.source_policy != policy
            or seal.source_primitive_digest != primitive_digest
            or re.fullmatch(r"[0-9a-f]{64}", seal.schema_digest) is None
            or re.fullmatch(r"[0-9a-f]{64}", seal.content_digest) is None
            or seal.schema_digest != _frame_schema_digest(frame)
            or seal.content_digest != _frame_content_digest(frame)
        ):
            raise ValueError(f"DATA-INVALID: frame manifest digest mismatch for {key}")
        try:
            pd.testing.assert_frame_equal(
                _canonical_frame_rows(frame)[key_columns],
                canonical_keys,
                check_dtype=False,
            )
        except AssertionError as error:
            raise ValueError(
                f"DATA-INVALID: frame manifest row provenance mismatch for {key}"
            ) from error


def _derive_production_feature_allowlists(
    arms: OutcomeArmTables,
    *,
    trusted_build_root: str,
) -> dict[str, tuple[str, ...]]:
    """Validate an internally rebuilt artifact against its separate trusted root."""
    lineage_columns = (
        "semantic_output",
        "source_primitive",
        "source_transform",
        "composition",
    )
    expected_lineage = pd.DataFrame.from_records(
        SEMANTIC_SOURCE_LINEAGE,
        columns=lineage_columns,
    )
    try:
        if tuple(arms.semantic_lineage.columns) != lineage_columns:
            raise AssertionError("lineage columns")
        pd.testing.assert_frame_equal(
            arms.semantic_lineage.reset_index(drop=True),
            expected_lineage,
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError("DATA-INVALID: semantic lineage mismatches frozen contract") from error

    expected = _expected_production_feature_dictionary()
    dictionary_columns = tuple(expected.columns)
    actual = arms.feature_dictionary
    try:
        if tuple(actual.columns) != dictionary_columns:
            raise AssertionError("dictionary columns")
        sort_columns = ["arm", "feature"]
        pd.testing.assert_frame_equal(
            actual.sort_values(sort_columns, kind="stable").reset_index(drop=True),
            expected.sort_values(sort_columns, kind="stable").reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError("DATA-INVALID: feature dictionary mismatches frozen contract") from error

    allowlists = {
        arm: tuple(expected.loc[expected["arm"].eq(arm), "feature"])
        for arm in ARM_NAMES
    }
    expected_frame_keys = {
        (fold, arm) for fold in range(10) for arm in ARM_NAMES
    }
    if set(arms.frames) != expected_frame_keys:
        raise ValueError("DATA-INVALID: fold-arm frame topology mismatches allowlist")
    metadata = {
        "sensor_day_id",
        "subject_id",
        "lifelog_date",
        "sleep_date",
        "elapsed_day_index",
        _ARM_SOURCE_POLICY_COLUMN,
        _ARM_SOURCE_DIGEST_COLUMN,
    }
    for (fold, arm), frame in arms.frames.items():
        if frame.columns.duplicated().any():
            raise ValueError(
                f"DATA-INVALID: fold {fold} arm {arm} has duplicate feature columns"
            )
        expected_columns = metadata.union(allowlists[arm])
        if set(frame.columns) != expected_columns:
            raise ValueError(
                f"DATA-INVALID: fold {fold} arm {arm} crosswires feature allowlist"
            )
    _validate_frame_manifest(arms, trusted_build_root=trusted_build_root)
    return allowlists


def validate_outcome_arms(
    arms: OutcomeArmTables,
    *,
    expected_content_digest: str,
) -> None:
    """Validate a sealed outcome-arm authority without building or fitting."""
    if type(arms) is not OutcomeArmTables:
        raise TypeError("Outcome arm authority has the wrong exact type")
    if (
        type(expected_content_digest) is not str
        or not _SHA256_PATTERN.fullmatch(expected_content_digest)
    ):
        raise TypeError(
            "Outcome arms expected content digest must be an exact built-in "
            "lowercase SHA-256 str"
        )
    _derive_production_feature_allowlists(
        arms,
        trusted_build_root=expected_content_digest,
    )


_DOWNSTREAM_REFERENCE_BASIS_COLUMNS = (
    "fold",
    "held_out_subject",
    "sensor_day_id",
    "primitive_position",
    "primitive",
    "raw_value",
    "personalized_z",
    "confidence",
)
_DOWNSTREAM_REFERENCE_BASIS_DTYPES = {
    "fold": "int64",
    "held_out_subject": "str",
    "sensor_day_id": "int64",
    "primitive_position": "int64",
    "primitive": "str",
    "raw_value": "float64",
    "personalized_z": "float64",
    "confidence": "float64",
}
_DOWNSTREAM_REFERENCE_BASELINE_COLUMNS = (
    "fold",
    "held_out_subject",
    "primitive_position",
    "primitive",
    "method",
    "k",
    "donor_subject",
    "center",
    "scale",
    "lambda_days",
)
_DOWNSTREAM_REFERENCE_BASELINE_DTYPES = {
    "fold": "int64",
    "held_out_subject": "str",
    "primitive_position": "int64",
    "primitive": "str",
    "method": "str",
    "k": "int64",
    "donor_subject": "object",
    "center": "float64",
    "scale": "float64",
    "lambda_days": "float64",
}
_DOWNSTREAM_REFERENCE_LINEAGE_COLUMNS = (
    "semantic_output",
    "source_primitive",
    "source_transform",
    "composition",
)


def _downstream_reference_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("DATA-INVALID: reference seal payload is not strict JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_reference_digest(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise TypeError(
            f"DATA-INVALID: {label} must be an exact built-in lowercase SHA-256 str"
        )
    return value


def _downstream_reference_seal_digest(
    seal: DownstreamReferenceFrameSeal,
) -> str:
    return _downstream_reference_sha256(
        [
            _DOWNSTREAM_REFERENCE_FRAME_SEAL_DOMAIN,
            seal.fold_id,
            seal.arm_id,
            seal.held_out_subject,
            seal.row_count,
            list(seal.feature_names),
            seal.prediction_key_digest,
            seal.source_policy,
            seal.source_primitive_digest,
            seal.schema_digest,
            seal.frame_content_digest,
        ]
    )


def _downstream_reference_root_digest(
    tables: DownstreamReferenceFrameTables,
    *,
    source_policy: str,
    source_primitive_digest: str,
) -> str:
    return _downstream_reference_sha256(
        [
            _DOWNSTREAM_REFERENCE_TABLES_DOMAIN,
            source_policy,
            source_primitive_digest,
            _table_schema_digest(tables.prediction_keys),
            _table_content_digest(
                tables.prediction_keys,
                sort_columns=("subject_id", "sensor_day_id"),
            ),
            [seal.content_digest for seal in tables.frame_manifest],
            _table_schema_digest(tables.primitive_basis),
            _table_content_digest(
                tables.primitive_basis,
                sort_columns=("fold", "sensor_day_id", "primitive_position"),
            ),
            _table_schema_digest(tables.heldout_baselines),
            _table_content_digest(
                tables.heldout_baselines,
                sort_columns=("fold", "primitive_position"),
            ),
            _table_schema_digest(tables.feature_dictionary),
            _table_content_digest(
                tables.feature_dictionary,
                sort_columns=("arm", "feature"),
            ),
            _table_schema_digest(tables.semantic_lineage),
            _table_content_digest(
                tables.semantic_lineage,
                sort_columns=_DOWNSTREAM_REFERENCE_LINEAGE_COLUMNS,
            ),
        ]
    )


def _exact_reference_strings(values: pd.Series, *, label: str) -> None:
    if any(type(value) is not str for value in values.tolist()):
        raise TypeError(f"DATA-INVALID: {label} values must be exact built-in str")


def _validated_reference_prediction_keys(
    prediction_keys: pd.DataFrame,
    *,
    require_canonical_order: bool,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if type(prediction_keys) is not pd.DataFrame:
        raise TypeError("DATA-INVALID: prediction_keys must be an exact DataFrame")
    if tuple(prediction_keys.columns) != _DOWNSTREAM_REFERENCE_KEY_COLUMNS:
        raise ValueError("DATA-INVALID: prediction key columns are not exact")
    _validate_exact_table_dtypes(
        prediction_keys,
        _DOWNSTREAM_REFERENCE_KEY_DTYPES,
        label="prediction keys",
    )
    if (
        prediction_keys.empty
        or prediction_keys.isna().any().any()
        or prediction_keys.duplicated().any()
        or prediction_keys["sensor_day_id"].duplicated().any()
        or prediction_keys.duplicated(
            ["subject_id", "lifelog_date", "sleep_date"]
        ).any()
    ):
        raise ValueError("DATA-INVALID: prediction keys are empty, null, or duplicated")
    _exact_reference_strings(prediction_keys["subject_id"], label="prediction subject")
    try:
        lifelog_dates = _validated_midnight(
            prediction_keys["lifelog_date"], "prediction lifelog_date"
        )
        sleep_dates = _validated_midnight(
            prediction_keys["sleep_date"], "prediction sleep_date"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "DATA-INVALID: prediction dates must be timezone-naive local midnights"
        ) from error
    if not (sleep_dates - lifelog_dates).eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "DATA-INVALID: prediction sleep_date must equal lifelog_date plus one day"
        )
    if not prediction_keys["elapsed_day_index"].ge(
        OUTCOME_EVALUATION_START_DAY
    ).all():
        raise ValueError("DATA-INVALID: prediction keys contain a pre-day-22 row")
    subjects = tuple(_ordered_subjects(prediction_keys["subject_id"]))
    if len(subjects) != 10 or any(type(subject) is not str for subject in subjects):
        raise ValueError("DATA-INVALID: prediction key roster must be ten exact strings")
    if any(
        not prediction_keys["subject_id"].eq(subject).any() for subject in subjects
    ):
        raise ValueError("DATA-INVALID: every subject needs a prediction key")
    subject_position = {subject: position for position, subject in enumerate(subjects)}
    canonical = (
        prediction_keys.assign(
            __subject_position=prediction_keys["subject_id"].map(subject_position)
        )
        .sort_values(["__subject_position", "sensor_day_id"], kind="stable")
        .drop(columns="__subject_position")
        .reset_index(drop=True)
    )
    if require_canonical_order:
        try:
            pd.testing.assert_frame_equal(
                prediction_keys.reset_index(drop=True),
                canonical,
                check_dtype=True,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(
                "DATA-INVALID: prediction keys are not in canonical order"
            ) from error
    return canonical, subjects


def _validated_reference_build_inputs(
    primitives: pd.DataFrame,
    prediction_keys: pd.DataFrame,
    *,
    expected_interval_boundary_policy: str,
    expected_primitive_digest: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...], str]:
    if type(primitives) is not pd.DataFrame:
        raise TypeError("DATA-INVALID: primitives must be an exact DataFrame")
    if (
        type(expected_interval_boundary_policy) is not str
        or expected_interval_boundary_policy not in INTERVAL_BOUNDARY_POLICIES
    ):
        raise TypeError("DATA-INVALID: expected interval boundary policy is not exact")
    expected_digest = _require_exact_reference_digest(
        expected_primitive_digest,
        label="expected primitive digest",
    )
    primitive_input = primitives.copy(deep=True)
    key_input = prediction_keys.copy(deep=True)
    required_primitive_columns = set(_DOWNSTREAM_REFERENCE_KEY_COLUMNS).union(
        BOUNDARY_POLICY_PROVENANCE_COLUMNS
    )
    missing = sorted(required_primitive_columns.difference(primitive_input.columns))
    if missing:
        raise ValueError(
            f"DATA-INVALID: primitive calendar lacks reference columns {missing}"
        )
    if (
        primitive_input.empty
        or primitive_input["sensor_day_id"].isna().any()
        or primitive_input["sensor_day_id"].duplicated().any()
        or primitive_input["subject_id"].isna().any()
    ):
        raise ValueError("DATA-INVALID: primitive calendar key topology is invalid")
    _exact_reference_strings(primitive_input["subject_id"], label="primitive subject")
    primitive_subjects = tuple(_ordered_subjects(primitive_input["subject_id"]))
    if len(primitive_subjects) != 10:
        raise ValueError("DATA-INVALID: primitive roster must contain ten subjects")
    provenance_digest = _validate_primitive_extraction_provenance(
        primitive_input,
        expected_interval_boundary_policy,
    )
    independently_recomputed_digest = _primitive_frame_digest(primitive_input)
    if (
        provenance_digest != independently_recomputed_digest
        or independently_recomputed_digest != expected_digest
    ):
        raise ValueError("DATA-INVALID: expected primitive digest mismatch")

    canonical_keys, key_subjects = _validated_reference_prediction_keys(
        key_input,
        require_canonical_order=False,
    )
    if key_subjects != primitive_subjects:
        raise ValueError("DATA-INVALID: prediction and primitive subject rosters differ")
    primitive_lookup = primitive_input.set_index("sensor_day_id", drop=False)
    matched = primitive_lookup.reindex(
        canonical_keys["sensor_day_id"].to_numpy()
    ).reset_index(drop=True)
    if matched["sensor_day_id"].isna().any():
        raise ValueError("DATA-INVALID: prediction sensor_day_id is absent from primitives")
    try:
        pd.testing.assert_frame_equal(
            canonical_keys.loc[:, list(_DOWNSTREAM_REFERENCE_KEY_COLUMNS)],
            matched.loc[:, list(_DOWNSTREAM_REFERENCE_KEY_COLUMNS)],
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "DATA-INVALID: prediction key calendar crosswalk disagrees with primitives"
        ) from error
    return (
        primitive_input,
        canonical_keys,
        primitive_subjects,
        independently_recomputed_digest,
    )


def _reference_subset_bundle(
    bundle: FoldSafeRepresentations,
    fold_keys: pd.DataFrame,
) -> FoldSafeRepresentations:
    sensor_ids = fold_keys["sensor_day_id"].to_numpy()
    selected: list[pd.DataFrame] = []
    for method in ("population", "shrinkage"):
        method_rows = bundle.representations.loc[
            bundle.representations["method"].eq(method)
        ]
        if method_rows["sensor_day_id"].duplicated().any():
            raise ValueError("DATA-INVALID: fold representation keys are duplicated")
        rows = (
            method_rows.set_index("sensor_day_id", drop=False)
            .reindex(sensor_ids)
            .reset_index(drop=True)
        )
        if (
            rows["sensor_day_id"].isna().any()
            or not rows["subject_id"].eq(bundle.outer_held_out_subject).all()
        ):
            raise ValueError("DATA-INVALID: requested held-out representation is absent")
        selected.append(rows)
    return FoldSafeRepresentations(
        outer_held_out_subject=bundle.outer_held_out_subject,
        representations=pd.concat(selected, ignore_index=True),
        baselines=bundle.baselines,
        recipient_audit=bundle.recipient_audit,
    )


def _reference_feature_dictionary() -> pd.DataFrame:
    expected = _expected_production_feature_dictionary()
    return expected.loc[
        expected["arm"].isin(DOWNSTREAM_ARM_NAMES)
    ].reset_index(drop=True)


def _reference_semantic_lineage() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        SEMANTIC_SOURCE_LINEAGE,
        columns=_DOWNSTREAM_REFERENCE_LINEAGE_COLUMNS,
    )


def _cast_reference_basis(records: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame.from_records(records, columns=_DOWNSTREAM_REFERENCE_BASIS_COLUMNS)
    return table.astype(_DOWNSTREAM_REFERENCE_BASIS_DTYPES)


def _cast_reference_baselines(records: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame.from_records(
        records,
        columns=_DOWNSTREAM_REFERENCE_BASELINE_COLUMNS,
    )
    table = table.astype(
        {
            key: value
            for key, value in _DOWNSTREAM_REFERENCE_BASELINE_DTYPES.items()
            if key != "donor_subject"
        }
    )
    table["donor_subject"] = pd.Series(
        [None] * len(table),
        index=table.index,
        dtype="object",
    )
    return table.loc[:, list(_DOWNSTREAM_REFERENCE_BASELINE_COLUMNS)]


def _reference_quiet_nan_only(values: pd.Series) -> bool:
    array = values.to_numpy(dtype=np.float64, copy=True)
    nan_mask = np.isnan(array)
    if not nan_mask.any():
        return True
    bits = array.view(np.uint64)
    return bool((bits[nan_mask] & np.uint64(0x0008_0000_0000_0000)).all())


def _reference_equal_float_bits(left: Any, right: Any) -> bool:
    left_bits = np.asarray([left], dtype=np.float64).view(np.uint64)[0]
    right_bits = np.asarray([right], dtype=np.float64).view(np.uint64)[0]
    return bool(left_bits == right_bits)


def _validate_reference_dictionary_and_lineage(
    tables: DownstreamReferenceFrameTables,
) -> dict[str, tuple[str, ...]]:
    expected_dictionary = _reference_feature_dictionary()
    expected_lineage = _reference_semantic_lineage()
    if (
        type(tables.feature_dictionary) is not pd.DataFrame
        or type(tables.semantic_lineage) is not pd.DataFrame
    ):
        raise TypeError("DATA-INVALID: reference registries must be exact DataFrames")
    try:
        pd.testing.assert_frame_equal(
            tables.feature_dictionary.reset_index(drop=True),
            expected_dictionary,
            check_dtype=True,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            tables.semantic_lineage.reset_index(drop=True),
            expected_lineage,
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "DATA-INVALID: reference feature dictionary or lineage mismatch"
        ) from error
    for column in ("arm", "feature", "source", "role"):
        _exact_reference_strings(
            tables.feature_dictionary[column],
            label=f"feature dictionary {column}",
        )
    for column in _DOWNSTREAM_REFERENCE_LINEAGE_COLUMNS:
        _exact_reference_strings(
            tables.semantic_lineage[column],
            label=f"semantic lineage {column}",
        )
    if tables.feature_dictionary.duplicated(["arm", "feature"]).any():
        raise ValueError("DATA-INVALID: reference feature dictionary is duplicated")
    return {
        arm: tuple(
            expected_dictionary.loc[
                expected_dictionary["arm"].eq(arm), "feature"
            ]
        )
        for arm in DOWNSTREAM_ARM_NAMES
    }


def _validate_reference_frames_and_seals(
    tables: DownstreamReferenceFrameTables,
    *,
    subjects: tuple[str, ...],
    allowlists: dict[str, tuple[str, ...]],
) -> tuple[str, str]:
    expected_sequence = tuple(
        (fold, arm)
        for fold in range(10)
        for arm in DOWNSTREAM_ARM_NAMES
    )
    if (
        type(tables.frames) is not dict
        or any(
            type(key) is not tuple
            or len(key) != 2
            or type(key[0]) is not int
            or type(key[1]) is not str
            for key in tables.frames
        )
        or tuple(tables.frames) != expected_sequence
    ):
        raise ValueError("DATA-INVALID: reference frame dictionary topology mismatch")
    if (
        type(tables.frame_manifest) is not tuple
        or len(tables.frame_manifest) != len(expected_sequence)
        or any(
            type(seal) is not DownstreamReferenceFrameSeal
            for seal in tables.frame_manifest
        )
    ):
        raise TypeError("DATA-INVALID: reference frame manifest identity mismatch")
    policies: set[str] = set()
    primitive_digests: set[str] = set()
    for position, (key, seal) in enumerate(
        zip(expected_sequence, tables.frame_manifest, strict=True)
    ):
        fold, arm = key
        heldout = subjects[fold]
        if (
            type(seal.fold_id) is not int
            or type(seal.arm_id) is not str
            or type(seal.held_out_subject) is not str
            or type(seal.row_count) is not int
            or type(seal.feature_names) is not tuple
            or any(type(feature) is not str for feature in seal.feature_names)
            or type(seal.source_policy) is not str
        ):
            raise TypeError(
                f"DATA-INVALID: reference frame seal field type mismatch at {position}"
            )
        for label, value in (
            ("prediction key digest", seal.prediction_key_digest),
            ("source primitive digest", seal.source_primitive_digest),
            ("frame schema digest", seal.schema_digest),
            ("frame content digest", seal.frame_content_digest),
            ("frame seal digest", seal.content_digest),
        ):
            _require_exact_reference_digest(value, label=label)
        if (
            (seal.fold_id, seal.arm_id, seal.held_out_subject) !=
            (fold, arm, heldout)
            or seal.feature_names != allowlists[arm]
            or len(seal.feature_names) != len(set(seal.feature_names))
        ):
            raise ValueError("DATA-INVALID: reference frame seal identity mismatch")
        frame = tables.frames[key]
        if type(frame) is not pd.DataFrame:
            raise TypeError("DATA-INVALID: reference frame must be an exact DataFrame")
        expected_columns = (
            *_DOWNSTREAM_REFERENCE_KEY_COLUMNS,
            *allowlists[arm],
            *_DOWNSTREAM_REFERENCE_PROVENANCE_COLUMNS,
        )
        if tuple(frame.columns) != expected_columns or frame.columns.duplicated().any():
            raise ValueError("DATA-INVALID: reference frame column order mismatch")
        if any(str(frame[feature].dtype) != "float64" for feature in allowlists[arm]):
            raise ValueError("DATA-INVALID: reference frame feature dtype mismatch")
        fold_keys = tables.prediction_keys.loc[
            tables.prediction_keys["subject_id"].eq(heldout),
            list(_DOWNSTREAM_REFERENCE_KEY_COLUMNS),
        ].sort_values("sensor_day_id", kind="stable").reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                frame.loc[:, list(_DOWNSTREAM_REFERENCE_KEY_COLUMNS)].reset_index(
                    drop=True
                ),
                fold_keys,
                check_dtype=True,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError("DATA-INVALID: reference frame key crosswire") from error
        if seal.row_count != len(fold_keys) or seal.row_count <= 0:
            raise ValueError("DATA-INVALID: reference frame row count mismatch")
        if (
            str(frame[_ARM_SOURCE_POLICY_COLUMN].dtype) != "str"
            or str(frame[_ARM_SOURCE_DIGEST_COLUMN].dtype) != "str"
        ):
            raise ValueError("DATA-INVALID: reference provenance dtype mismatch")
        _exact_reference_strings(
            frame[_ARM_SOURCE_POLICY_COLUMN], label="frame source policy"
        )
        _exact_reference_strings(
            frame[_ARM_SOURCE_DIGEST_COLUMN], label="frame source primitive digest"
        )
        if (
            not frame[_ARM_SOURCE_POLICY_COLUMN].eq(seal.source_policy).all()
            or not frame[_ARM_SOURCE_DIGEST_COLUMN]
            .eq(seal.source_primitive_digest)
            .all()
            or seal.source_policy not in INTERVAL_BOUNDARY_POLICIES
        ):
            raise ValueError("DATA-INVALID: reference frame provenance mismatch")
        expected_key_digest = _table_content_digest(
            fold_keys,
            sort_columns=("sensor_day_id",),
        )
        if (
            seal.prediction_key_digest != expected_key_digest
            or seal.schema_digest != _frame_schema_digest(frame)
            or seal.frame_content_digest != _frame_content_digest(frame)
            or seal.content_digest != _downstream_reference_seal_digest(seal)
        ):
            raise ValueError("DATA-INVALID: reference frame seal digest mismatch")
        policies.add(seal.source_policy)
        primitive_digests.add(seal.source_primitive_digest)
    if len(policies) != 1 or len(primitive_digests) != 1:
        raise ValueError("DATA-INVALID: reference source provenance is nonunique")
    policy = next(iter(policies))
    primitive_digest = next(iter(primitive_digests))
    _require_exact_reference_digest(
        primitive_digest,
        label="reference source primitive digest",
    )
    return policy, primitive_digest


def _validate_reference_primitive_basis(
    tables: DownstreamReferenceFrameTables,
    *,
    subjects: tuple[str, ...],
) -> None:
    basis = tables.primitive_basis
    if type(basis) is not pd.DataFrame or tuple(basis.columns) != (
        _DOWNSTREAM_REFERENCE_BASIS_COLUMNS
    ):
        raise ValueError("DATA-INVALID: primitive basis columns mismatch")
    _validate_exact_table_dtypes(
        basis,
        _DOWNSTREAM_REFERENCE_BASIS_DTYPES,
        label="primitive basis",
    )
    expected_records: list[dict[str, Any]] = []
    for fold, heldout in enumerate(subjects):
        fold_ids = tables.prediction_keys.loc[
            tables.prediction_keys["subject_id"].eq(heldout), "sensor_day_id"
        ].tolist()
        for sensor_day_id in fold_ids:
            for primitive_position, primitive in enumerate(MATCHED_SOURCE_PRIMITIVES):
                expected_records.append(
                    {
                        "fold": fold,
                        "held_out_subject": heldout,
                        "sensor_day_id": sensor_day_id,
                        "primitive_position": primitive_position,
                        "primitive": primitive,
                    }
                )
    identity_columns = _DOWNSTREAM_REFERENCE_BASIS_COLUMNS[:5]
    expected_identity = pd.DataFrame.from_records(
        expected_records,
        columns=identity_columns,
    ).astype(
        {
            key: _DOWNSTREAM_REFERENCE_BASIS_DTYPES[key]
            for key in identity_columns
        }
    )
    try:
        pd.testing.assert_frame_equal(
            basis.loc[:, list(identity_columns)].reset_index(drop=True),
            expected_identity,
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError("DATA-INVALID: primitive basis topology mismatch") from error
    raw = basis["raw_value"].to_numpy(dtype=np.float64)
    personalized = basis["personalized_z"].to_numpy(dtype=np.float64)
    confidence = basis["confidence"].to_numpy(dtype=np.float64)
    if (
        np.isinf(raw).any()
        or np.isinf(personalized).any()
        or not np.isfinite(confidence).all()
        or ((confidence < 0.0) | (confidence > 1.0)).any()
        or not _reference_quiet_nan_only(basis["raw_value"])
        or not _reference_quiet_nan_only(basis["personalized_z"])
    ):
        raise ValueError("DATA-INVALID: primitive basis value domain mismatch")
    _exact_reference_strings(basis["held_out_subject"], label="basis heldout subject")
    _exact_reference_strings(basis["primitive"], label="basis primitive")
    for row in basis.itertuples(index=False):
        frame = tables.frames[(row.fold, PRIMARY_BASELINE_ARM)]
        matches = frame.loc[frame["sensor_day_id"].eq(row.sensor_day_id)]
        if len(matches) != 1:
            raise ValueError("DATA-INVALID: primitive basis frame crosslink mismatch")
        frame_row = matches.iloc[0]
        if not _reference_equal_float_bits(
            row.personalized_z,
            frame_row[f"primitive__{row.primitive}__personalized_z"],
        ) or not _reference_equal_float_bits(
            row.confidence,
            frame_row[f"quality__{row.primitive}__quality_confidence"],
        ):
            raise ValueError("DATA-INVALID: primitive basis feature crosslink mismatch")
        if row.primitive in RAW_TIMING_PRIMITIVES and not _reference_equal_float_bits(
            row.raw_value,
            frame_row[f"primitive__{row.primitive}__raw_minute"],
        ):
            raise ValueError("DATA-INVALID: primitive basis raw timing crosslink mismatch")


def _validate_reference_heldout_baselines(
    tables: DownstreamReferenceFrameTables,
    *,
    subjects: tuple[str, ...],
) -> None:
    baselines = tables.heldout_baselines
    if type(baselines) is not pd.DataFrame or tuple(baselines.columns) != (
        _DOWNSTREAM_REFERENCE_BASELINE_COLUMNS
    ):
        raise ValueError("DATA-INVALID: heldout baseline columns mismatch")
    _validate_exact_table_dtypes(
        baselines,
        _DOWNSTREAM_REFERENCE_BASELINE_DTYPES,
        label="heldout baselines",
    )
    expected_records = [
        {
            "fold": fold,
            "held_out_subject": heldout,
            "primitive_position": position,
            "primitive": primitive,
            "method": "shrinkage",
            "k": PRIMARY_CALIBRATION_DAY,
            "donor_subject": None,
            "lambda_days": PRIMARY_LAMBDA_DAYS,
        }
        for fold, heldout in enumerate(subjects)
        for position, primitive in enumerate(
            _DOWNSTREAM_REFERENCE_BASELINE_PRIMITIVES
        )
    ]
    identity_columns = (
        "fold",
        "held_out_subject",
        "primitive_position",
        "primitive",
        "method",
        "k",
        "donor_subject",
        "lambda_days",
    )
    expected_identity = pd.DataFrame.from_records(
        expected_records,
        columns=identity_columns,
    )
    for column, dtype in _DOWNSTREAM_REFERENCE_BASELINE_DTYPES.items():
        if column in expected_identity.columns and column != "donor_subject":
            expected_identity[column] = expected_identity[column].astype(dtype)
    expected_identity["donor_subject"] = pd.Series(
        [None] * len(expected_identity), dtype="object"
    )
    try:
        pd.testing.assert_frame_equal(
            baselines.loc[:, list(identity_columns)].reset_index(drop=True),
            expected_identity,
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError("DATA-INVALID: heldout baseline topology mismatch") from error
    if (
        len(baselines) != 80
        or not np.isfinite(baselines["center"].to_numpy(dtype=float)).all()
        or not np.isfinite(baselines["scale"].to_numpy(dtype=float)).all()
        or not baselines["scale"].gt(0.0).all()
        or not all(value is None for value in baselines["donor_subject"].tolist())
    ):
        raise ValueError("DATA-INVALID: heldout baseline value domain mismatch")
    for column in ("held_out_subject", "primitive", "method"):
        _exact_reference_strings(baselines[column], label=f"baseline {column}")


def validate_downstream_reference_frames(
    tables: DownstreamReferenceFrameTables,
    *,
    expected_content_digest: str,
) -> None:
    """Validate all mutable nested objects against an independently held root."""
    expected_root = _require_exact_reference_digest(
        expected_content_digest,
        label="expected reference content digest",
    )
    if type(tables) is not DownstreamReferenceFrameTables:
        raise TypeError("DATA-INVALID: reference tables identity is forged")
    stored_root = _require_exact_reference_digest(
        tables.content_digest,
        label="stored reference content digest",
    )
    if any(
        type(table) is not pd.DataFrame
        for table in (
            tables.prediction_keys,
            tables.primitive_basis,
            tables.heldout_baselines,
            tables.feature_dictionary,
            tables.semantic_lineage,
        )
    ):
        raise TypeError("DATA-INVALID: reference table identity is forged")
    _, subjects = _validated_reference_prediction_keys(
        tables.prediction_keys,
        require_canonical_order=True,
    )
    allowlists = _validate_reference_dictionary_and_lineage(tables)
    policy, primitive_digest = _validate_reference_frames_and_seals(
        tables,
        subjects=subjects,
        allowlists=allowlists,
    )
    _validate_reference_primitive_basis(tables, subjects=subjects)
    _validate_reference_heldout_baselines(tables, subjects=subjects)
    rebuilt_root = _downstream_reference_root_digest(
        tables,
        source_policy=policy,
        source_primitive_digest=primitive_digest,
    )
    if stored_root != rebuilt_root or rebuilt_root != expected_root:
        raise ValueError("DATA-INVALID: downstream reference root digest mismatch")


def build_downstream_reference_frames(
    primitives: pd.DataFrame,
    prediction_keys: pd.DataFrame,
    *,
    expected_interval_boundary_policy: str,
    expected_primitive_digest: str,
) -> DownstreamReferenceFrameTables:
    """Build arbitrary post-day-21 held-out frames without labels or fitting."""
    (
        primitive_input,
        canonical_keys,
        subjects,
        primitive_digest,
    ) = _validated_reference_build_inputs(
        primitives,
        prediction_keys,
        expected_interval_boundary_policy=expected_interval_boundary_policy,
        expected_primitive_digest=expected_primitive_digest,
    )
    feature_dictionary = _reference_feature_dictionary()
    semantic_lineage = _reference_semantic_lineage()
    allowlists = {
        arm: tuple(
            feature_dictionary.loc[feature_dictionary["arm"].eq(arm), "feature"]
        )
        for arm in DOWNSTREAM_ARM_NAMES
    }
    canonical_primitive_lookup = primitive_input.set_index(
        "sensor_day_id",
        drop=False,
    )
    full_tables = build_loso_representations(
        primitive_input,
        k_values=(PRIMARY_CALIBRATION_DAY,),
        lambda_days=PRIMARY_LAMBDA_DAYS,
    )
    frames: dict[tuple[int, str], pd.DataFrame] = {}
    seals: list[DownstreamReferenceFrameSeal] = []
    basis_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    for fold, heldout in enumerate(subjects):
        bundle = _combine_fold_representations(
            primitive_input,
            heldout,
            full_tables,
        )
        fold_keys = canonical_keys.loc[
            canonical_keys["subject_id"].eq(heldout),
            list(_DOWNSTREAM_REFERENCE_KEY_COLUMNS),
        ].sort_values("sensor_day_id", kind="stable").reset_index(drop=True)
        fold_bundle = _reference_subset_bundle(bundle, fold_keys)
        built_frames, built_dictionary = _arm_frames_for_fold(
            fold_bundle,
            primitive_input,
            fold_keys,
        )
        built_downstream_dictionary = built_dictionary.loc[
            built_dictionary["arm"].isin(DOWNSTREAM_ARM_NAMES)
        ].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                built_downstream_dictionary,
                feature_dictionary,
                check_dtype=True,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError(
                "DATA-INVALID: constructed feature dictionary mismatches authority"
            ) from error
        for arm in DOWNSTREAM_ARM_NAMES:
            declared = allowlists[arm]
            frame = built_frames[arm]
            expected_columns = (
                *_DOWNSTREAM_REFERENCE_KEY_COLUMNS,
                *declared,
            )
            if set(frame.columns) != set(expected_columns):
                raise ValueError("DATA-INVALID: constructed reference frame crosswire")
            frame = frame.loc[:, list(expected_columns)].copy()
            for feature in declared:
                frame[feature] = frame[feature].astype("float64")
            frame[_ARM_SOURCE_POLICY_COLUMN] = pd.Series(
                [expected_interval_boundary_policy] * len(frame),
                dtype="str",
            )
            frame[_ARM_SOURCE_DIGEST_COLUMN] = pd.Series(
                [primitive_digest] * len(frame),
                dtype="str",
            )
            key = (fold, arm)
            frames[key] = frame
            unsealed = DownstreamReferenceFrameSeal(
                fold_id=fold,
                arm_id=arm,
                held_out_subject=heldout,
                row_count=int(len(frame)),
                feature_names=declared,
                prediction_key_digest=_table_content_digest(
                    fold_keys,
                    sort_columns=("sensor_day_id",),
                ),
                source_policy=expected_interval_boundary_policy,
                source_primitive_digest=primitive_digest,
                schema_digest=_frame_schema_digest(frame),
                frame_content_digest=_frame_content_digest(frame),
                content_digest="0" * 64,
            )
            seals.append(
                DownstreamReferenceFrameSeal(
                    fold_id=unsealed.fold_id,
                    arm_id=unsealed.arm_id,
                    held_out_subject=unsealed.held_out_subject,
                    row_count=unsealed.row_count,
                    feature_names=unsealed.feature_names,
                    prediction_key_digest=unsealed.prediction_key_digest,
                    source_policy=unsealed.source_policy,
                    source_primitive_digest=unsealed.source_primitive_digest,
                    schema_digest=unsealed.schema_digest,
                    frame_content_digest=unsealed.frame_content_digest,
                    content_digest=_downstream_reference_seal_digest(unsealed),
                )
            )

        shrinkage = fold_bundle.representations.loc[
            fold_bundle.representations["method"].eq("shrinkage")
        ].set_index("sensor_day_id", drop=False)
        for sensor_day_id in fold_keys["sensor_day_id"].tolist():
            representation = shrinkage.loc[sensor_day_id]
            for primitive_position, primitive in enumerate(MATCHED_SOURCE_PRIMITIVES):
                raw_value = representation[primitive]
                canonical_raw_value = canonical_primitive_lookup.at[
                    sensor_day_id,
                    primitive,
                ]
                if not _reference_equal_float_bits(raw_value, canonical_raw_value):
                    raise ValueError(
                        "DATA-INVALID: shrinkage raw primitive bits disagree with "
                        "the canonical primitive calendar"
                    )
                basis_records.append(
                    {
                        "fold": fold,
                        "held_out_subject": heldout,
                        "sensor_day_id": sensor_day_id,
                        "primitive_position": primitive_position,
                        "primitive": primitive,
                        "raw_value": raw_value,
                        "personalized_z": representation[f"{primitive}__z"],
                        "confidence": representation[f"{primitive}__confidence"],
                    }
                )

        heldout_baselines = bundle.baselines.loc[
            bundle.baselines["outer_fold_subject"].eq(heldout)
            & bundle.baselines["recipient_subject"].eq(heldout)
            & bundle.baselines["method"].eq("shrinkage")
            & bundle.baselines["k"].eq(PRIMARY_CALIBRATION_DAY)
        ]
        if heldout_baselines["primitive"].duplicated().any():
            raise ValueError("DATA-INVALID: heldout baseline primitive is duplicated")
        baseline_lookup = heldout_baselines.set_index("primitive", drop=False)
        for primitive_position, primitive in enumerate(
            _DOWNSTREAM_REFERENCE_BASELINE_PRIMITIVES
        ):
            if primitive not in baseline_lookup.index:
                raise ValueError("DATA-INVALID: heldout baseline primitive is absent")
            baseline = baseline_lookup.loc[primitive]
            baseline_records.append(
                {
                    "fold": fold,
                    "held_out_subject": heldout,
                    "primitive_position": primitive_position,
                    "primitive": primitive,
                    "method": "shrinkage",
                    "k": PRIMARY_CALIBRATION_DAY,
                    "donor_subject": None,
                    "center": baseline["center"],
                    "scale": baseline["scale"],
                    "lambda_days": baseline["lambda_days"],
                }
            )

    primitive_basis = _cast_reference_basis(basis_records)
    heldout_baselines = _cast_reference_baselines(baseline_records)
    unsealed_tables = DownstreamReferenceFrameTables(
        prediction_keys=canonical_keys,
        frames=frames,
        frame_manifest=tuple(seals),
        primitive_basis=primitive_basis,
        heldout_baselines=heldout_baselines,
        feature_dictionary=feature_dictionary,
        semantic_lineage=semantic_lineage,
        content_digest="0" * 64,
    )
    content_digest = _downstream_reference_root_digest(
        unsealed_tables,
        source_policy=expected_interval_boundary_policy,
        source_primitive_digest=primitive_digest,
    )
    result = DownstreamReferenceFrameTables(
        prediction_keys=unsealed_tables.prediction_keys,
        frames=unsealed_tables.frames,
        frame_manifest=unsealed_tables.frame_manifest,
        primitive_basis=unsealed_tables.primitive_basis,
        heldout_baselines=unsealed_tables.heldout_baselines,
        feature_dictionary=unsealed_tables.feature_dictionary,
        semantic_lineage=unsealed_tables.semantic_lineage,
        content_digest=content_digest,
    )
    if (
        primitive_digest != expected_primitive_digest
        or any(
            seal.source_policy != expected_interval_boundary_policy
            or seal.source_primitive_digest != expected_primitive_digest
            for seal in result.frame_manifest
        )
    ):
        raise ValueError("DATA-INVALID: independently supplied source authority mismatch")
    validate_downstream_reference_frames(
        result,
        expected_content_digest=content_digest,
    )
    return result


def build_outcome_arms(
    primitives: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    interval_boundary_policy: str = "measurement_support",
) -> OutcomeArmTables:
    """Build all nested fold features before attaching any outcome values."""
    if not isinstance(primitives, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        raise TypeError("primitives and labels must be pandas DataFrames")
    if interval_boundary_policy not in INTERVAL_BOUNDARY_POLICIES:
        raise ValueError(
            f"interval_boundary_policy must be one of {INTERVAL_BOUNDARY_POLICIES}"
        )
    primitive_input = primitives.copy(deep=True)
    primitive_digest = _validate_primitive_extraction_provenance(
        primitive_input,
        interval_boundary_policy,
    )
    subjects = _ordered_subjects(primitive_input["subject_id"])
    if len(subjects) != 10:
        raise ValueError("Canonical G4 feature construction requires exactly 10 participants")

    full_tables = build_loso_representations(
        primitive_input,
        k_values=(PRIMARY_CALIBRATION_DAY,),
        lambda_days=PRIMARY_LAMBDA_DAYS,
    )
    bundles = [
        _combine_fold_representations(
            primitive_input,
            outer_held_out_subject,
            full_tables,
        )
        for outer_held_out_subject in subjects
    ]

    sensor_keys = primitive_input[
        [
            "sensor_day_id",
            "subject_id",
            "lifelog_date",
            "sleep_date",
            "elapsed_day_index",
        ]
    ]
    joined = join_outcome_labels(sensor_keys, labels, require_real_contract=True)
    evaluation_keys = joined[
        [
            "sensor_day_id",
            "subject_id",
            "lifelog_date",
            "sleep_date",
            "elapsed_day_index",
        ]
    ].copy()

    frames: dict[tuple[int, str], pd.DataFrame] = {}
    frame_seals: list[OutcomeFrameSeal] = []
    dictionaries: list[pd.DataFrame] = []
    baseline_tables: list[pd.DataFrame] = []
    recipient_tables: list[pd.DataFrame] = []
    for fold, bundle in enumerate(bundles):
        current_frames, dictionary = _arm_frames_for_fold(
            bundle,
            primitive_input,
            evaluation_keys,
        )
        for arm, frame in current_frames.items():
            frame_with_provenance = frame.copy()
            frame_with_provenance[_ARM_SOURCE_POLICY_COLUMN] = (
                interval_boundary_policy
            )
            frame_with_provenance[_ARM_SOURCE_DIGEST_COLUMN] = primitive_digest
            frames[(fold, arm)] = frame_with_provenance
            frame_seals.append(
                OutcomeFrameSeal(
                    fold_id=fold,
                    arm_id=arm,
                    held_out_subject=bundle.outer_held_out_subject,
                    source_policy=interval_boundary_policy,
                    source_primitive_digest=primitive_digest,
                    schema_digest=_frame_schema_digest(frame_with_provenance),
                    content_digest=_frame_content_digest(frame_with_provenance),
                )
            )
        dictionaries.append(dictionary)
        baseline_tables.append(bundle.baselines.assign(fold=fold))
        recipient_tables.append(bundle.recipient_audit.assign(fold=fold))

    feature_dictionary = dictionaries[0].reset_index(drop=True)
    for dictionary in dictionaries[1:]:
        pd.testing.assert_frame_equal(
            feature_dictionary,
            dictionary.reset_index(drop=True),
            check_dtype=False,
        )
    if feature_dictionary.duplicated(["arm", "feature"]).any():
        raise RuntimeError("Feature dictionary contains duplicate arm features")
    expected_ids = tuple(evaluation_keys["sensor_day_id"])
    for (fold, arm), frame in frames.items():
        if tuple(frame["sensor_day_id"]) != expected_ids:
            raise RuntimeError(f"Fold {fold} arm {arm} changed the G4 row universe")

    frame_manifest = tuple(frame_seals)
    fold_baselines = pd.concat(baseline_tables, ignore_index=True)
    recipient_audit = pd.concat(recipient_tables, ignore_index=True)
    manifest = pd.DataFrame.from_records(
        [
            {
                "sensor_calendar_rows": len(primitive_input),
                "outcome_label_rows": len(labels),
                "evaluation_rows": len(evaluation_keys),
                "participant_denominator": len(subjects),
                "outer_folds": len(subjects),
                "arms": len(ARM_NAMES),
                "targets": len(TARGET_COLUMNS),
                "semantic_lineage_rows": len(SEMANTIC_SOURCE_LINEAGE),
                "primary_k": PRIMARY_CALIBRATION_DAY,
                "lambda_days": PRIMARY_LAMBDA_DAYS,
                "nested_recipient_exclusion": True,
                "labels_joined_after_sensor_features": True,
                "interval_boundary_policy": interval_boundary_policy,
                "extraction_interval_boundary_policy": interval_boundary_policy,
                "primitive_digest": primitive_digest,
                "primitive_provenance_columns": json.dumps(
                    BOUNDARY_POLICY_PROVENANCE_COLUMNS
                ),
                "frame_manifest_rows": len(frame_manifest),
                "frame_schema_digest_algorithm": _FRAME_SCHEMA_DIGEST_ALGORITHM,
                "frame_content_digest_algorithm": _FRAME_CONTENT_DIGEST_ALGORITHM,
                "frame_manifest_digest_algorithm": _FRAME_MANIFEST_DIGEST_ALGORITHM,
                "frame_manifest_digest": _frame_manifest_digest(frame_manifest),
                "outcome_build_digest_algorithm": _OUTCOME_BUILD_DIGEST_ALGORITHM,
                "strict_source_availability_sensitivity": (
                    "separate_required"
                    if interval_boundary_policy == "measurement_support"
                    else "this_is_sensitivity_not_primary"
                ),
                "g4_is_secondary": True,
                "can_change_g1_g3": False,
                "status": "ready",
            }
        ]
    )
    result = OutcomeArmTables(
        evaluation_keys=evaluation_keys.reset_index(drop=True),
        frames=frames,
        frame_manifest=frame_manifest,
        feature_dictionary=feature_dictionary,
        semantic_lineage=pd.DataFrame.from_records(
            SEMANTIC_SOURCE_LINEAGE,
            columns=(
                "semantic_output",
                "source_primitive",
                "source_transform",
                "composition",
            ),
        ),
        fold_baselines=fold_baselines,
        recipient_audit=recipient_audit,
        manifest=manifest,
    )
    for name, digest in _outcome_build_digest_components(result).items():
        manifest.loc[:, name] = digest
    trusted_build_root = _outcome_build_root_digest(result)
    manifest.loc[:, "outcome_build_digest"] = trusted_build_root
    _derive_production_feature_allowlists(
        result,
        trusted_build_root=trusted_build_root,
    )
    return result


def _clipped_probability(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all() or bool(((array < 0.0) | (array > 1.0)).any()):
        raise ValueError("Probabilities must be finite and lie in [0,1]")
    return np.clip(array, SCORE_EPSILON, 1.0 - SCORE_EPSILON)


def _binary_truth(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all() or not np.isin(array, (0.0, 1.0)).all():
        raise ValueError("Outcome truth must be exact binary values")
    return array.astype(int)


def _logloss_rows(truth: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return -(truth * np.log(probability) + (1 - truth) * np.log(1.0 - probability))


def calibration_summary(
    truth: pd.Series | np.ndarray,
    probability: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Return pooled calibration intercept/slope only when identifiable."""
    y = _binary_truth(truth)
    prediction = _clipped_probability(probability)
    if len(y) != len(prediction):
        raise ValueError("Calibration truth and probability must align")
    if np.unique(y).size < 2:
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "status": "nonidentifiable_single_class",
        }
    logit = np.log(prediction / (1.0 - prediction))
    if float(np.ptp(logit)) <= 1e-12:
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "status": "nonidentifiable_constant_prediction",
        }
    design = np.column_stack([np.ones(len(logit), dtype=float), logit])
    beta = np.zeros(2, dtype=float)
    converged = False
    for _ in range(100):
        linear = np.clip(design @ beta, -30.0, 30.0)
        fitted = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(fitted * (1.0 - fitted), 1e-9)
        hessian = design.T @ (weights[:, None] * design)
        gradient = design.T @ (y - fitted)
        try:
            increment = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            increment = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta += increment
        if float(np.max(np.abs(increment))) < 1e-10:
            converged = True
            break
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "status": "ok" if converged else "fit_not_converged",
    }


def score_outcome_predictions(predictions: pd.DataFrame) -> OutcomeScoreTables:
    """Score participant-target cells first, then weight people and targets equally."""
    required = {
        "arm",
        "model",
        "target",
        "fold",
        "sensor_day_id",
        "subject_id",
        "truth",
        "probability",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    key_columns = ["arm", "model", "target", "sensor_day_id"]
    if predictions.duplicated(key_columns).any():
        raise ValueError("Prediction arm/model/target/sensor rows must be unique")
    rows = predictions.copy()
    truth = _binary_truth(rows["truth"])
    probability = _clipped_probability(rows["probability"])
    rows["scoring_probability"] = probability
    rows["row_logloss"] = _logloss_rows(truth, probability)
    rows["row_brier"] = np.square(truth - probability)

    participant_target = (
        rows.groupby(
            ["arm", "model", "subject_id", "target"],
            sort=False,
            dropna=False,
        )
        .agg(
            rows=("sensor_day_id", "size"),
            logloss=("row_logloss", "mean"),
            brier=("row_brier", "mean"),
        )
        .reset_index()
    )
    participant = (
        participant_target.groupby(
            ["arm", "model", "subject_id"], sort=False, dropna=False
        )
        .agg(
            target_count=("target", "nunique"),
            participant_logloss=("logloss", "mean"),
            participant_brier=("brier", "mean"),
        )
        .reset_index()
    )
    target = (
        participant_target.groupby(
            ["arm", "model", "target"], sort=False, dropna=False
        )
        .agg(
            participant_count=("subject_id", "nunique"),
            target_logloss=("logloss", "mean"),
            target_brier=("brier", "mean"),
        )
        .reset_index()
    )
    macro_cell = (
        participant_target.groupby(["arm", "model"], sort=False, dropna=False)
        .agg(
            participant_target_cells=("logloss", "size"),
            participant_target_logloss=("logloss", "mean"),
            participant_target_brier=("brier", "mean"),
        )
        .reset_index()
    )
    day_weighted = (
        rows.groupby(["arm", "model"], sort=False, dropna=False)
        .agg(
            prediction_rows=("sensor_day_id", "size"),
            day_weighted_logloss=("row_logloss", "mean"),
            day_weighted_brier=("row_brier", "mean"),
        )
        .reset_index()
    )
    macro = macro_cell.merge(
        day_weighted, on=["arm", "model"], how="inner", validate="one_to_one"
    )

    calibration_records: list[dict[str, Any]] = []
    for (arm, model, target_name), group in rows.groupby(
        ["arm", "model", "target"], sort=False, dropna=False
    ):
        calibration = calibration_summary(group["truth"], group["probability"])
        calibration_records.append(
            {
                "arm": arm,
                "model": model,
                "target": target_name,
                "rows": len(group),
                **calibration,
            }
        )
    return OutcomeScoreTables(
        row_scores=rows,
        participant_target_scores=participant_target,
        participant_scores=participant,
        target_scores=target,
        macro_scores=macro,
        calibration=pd.DataFrame.from_records(calibration_records),
    )


def pair_arm_predictions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    """Pair two arms only after proving exact equality of all OOF row keys."""
    keys = [
        "model",
        "target",
        "fold",
        "sensor_day_id",
        "subject_id",
        "lifelog_date",
        "sleep_date",
        "elapsed_day_index",
        "truth",
    ]
    required = set(keys + ["arm", "probability"])
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} predictions are missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} prediction rows are not unique")
    left_keys = baseline[keys].sort_values(keys, kind="stable").reset_index(drop=True)
    right_keys = candidate[keys].sort_values(keys, kind="stable").reset_index(drop=True)
    if len(left_keys) != len(right_keys) or not left_keys.equals(right_keys):
        raise ValueError("Baseline and candidate must have identical OOF row keys")
    paired = baseline[keys + ["probability"]].merge(
        candidate[keys + ["probability"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_candidate"),
    )
    truth = _binary_truth(paired["truth"])
    baseline_probability = _clipped_probability(paired["probability_baseline"])
    candidate_probability = _clipped_probability(paired["probability_candidate"])
    paired["baseline_logloss"] = _logloss_rows(truth, baseline_probability)
    paired["candidate_logloss"] = _logloss_rows(truth, candidate_probability)
    paired["row_delta"] = paired["baseline_logloss"] - paired["candidate_logloss"]
    paired["baseline_arm"] = baseline["arm"].iloc[0] if len(baseline) else np.nan
    paired["candidate_arm"] = candidate["arm"].iloc[0] if len(candidate) else np.nan
    return paired


def _exact_signflip_p(values: np.ndarray) -> tuple[float, int]:
    count = len(values)
    patterns = 1 << count
    bit_positions = np.arange(count, dtype=np.uint64)
    pattern_ids = np.arange(patterns, dtype=np.uint64)[:, None]
    signs = np.where(((pattern_ids >> bit_positions) & 1) == 1, 1.0, -1.0)
    null_means = (signs * values[None, :]).mean(axis=1)
    observed = float(values.mean())
    p_value = float(np.mean(null_means >= observed - 1e-15))
    return p_value, patterns


def summarize_participant_deltas(
    values: pd.Series | np.ndarray,
    *,
    draws: int = DEFAULT_OUTCOME_BOOTSTRAP_DRAWS,
    seed: int,
    contrast_id: str,
) -> ParticipantDeltaTables:
    """Apply fixed-ten participant bootstrap, sign flip, LOPO and concentration."""
    deltas = np.asarray(values, dtype=float).reshape(-1)
    if len(deltas) != 10:
        raise ValueError("G4 paired uncertainty requires exactly 10 participants")
    if not np.isfinite(deltas).all():
        raise ValueError("Participant deltas must all be finite")
    if isinstance(draws, (bool, np.bool_)) or int(draws) != draws or draws <= 0:
        raise ValueError("draws must be a positive integer")
    draw_count = int(draws)
    bootstrap_records: list[dict[str, Any]] = []
    for draw in range(draw_count):
        draw_seed = derive_deterministic_seed(
            seed, "g4_participant_bootstrap", contrast_id, draw
        )
        indices = np.random.default_rng(draw_seed).integers(0, 10, size=10)
        bootstrap_records.append(
            {
                "draw": draw,
                "draw_seed": draw_seed,
                "mean_delta": float(deltas[indices].mean()),
                "participant_denominator": 10,
            }
        )
    bootstrap = pd.DataFrame.from_records(bootstrap_records)
    ci_low, ci_high = np.quantile(
        bootstrap["mean_delta"].to_numpy(dtype=float), [0.025, 0.975]
    )
    signflip_p, signflip_patterns = _exact_signflip_p(deltas)
    lopo_records = [
        {
            "held_out_position": held_out,
            "lopo_mean_delta": float(np.delete(deltas, held_out).mean()),
            "participant_denominator": 10,
            "retained_participants": 9,
        }
        for held_out in range(10)
    ]
    lopo = pd.DataFrame.from_records(lopo_records)
    positive = np.maximum(deltas, 0.0)
    positive_sum = float(positive.sum())
    concentration = (
        float(positive.max() / positive_sum)
        if positive_sum > 0.0
        else float("nan")
    )
    summary = pd.DataFrame.from_records(
        [
            {
                "contrast_id": contrast_id,
                "participant_denominator": 10,
                "macro_delta": float(deltas.mean()),
                "improved_participants": int((deltas > 0.0).sum()),
                "bootstrap_draws": draw_count,
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "signflip_patterns": signflip_patterns,
                "signflip_p_one_sided": signflip_p,
                "finite_lopo": int(np.isfinite(lopo["lopo_mean_delta"]).sum()),
                "minimum_lopo_delta": float(lopo["lopo_mean_delta"].min()),
                "positive_gain_sum": positive_sum,
                "contribution_concentration": concentration,
                "concentration_gate": bool(
                    np.isfinite(concentration) and concentration <= 0.5
                ),
                "root_seed": int(seed),
                "status": "ok",
            }
        ]
    )
    return ParticipantDeltaTables(summary=summary, bootstrap_draws=bootstrap, lopo=lopo)


def _invalid_participant_delta_tables(
    values: np.ndarray,
    *,
    draws: int,
    seed: int,
    contrast_id: str,
) -> ParticipantDeltaTables:
    """Return an auditable invalid result without fabricating fixed-ten inference."""
    deltas = np.asarray(values, dtype=float).reshape(-1)
    finite = deltas[np.isfinite(deltas)]
    positive = np.maximum(finite, 0.0)
    positive_sum = float(positive.sum())
    concentration = (
        float(positive.max() / positive_sum)
        if positive_sum > 0.0
        else float("nan")
    )
    summary = pd.DataFrame.from_records(
        [
            {
                "contrast_id": contrast_id,
                "participant_denominator": len(deltas),
                "macro_delta": float(finite.mean()) if len(finite) else float("nan"),
                "improved_participants": int((finite > 0.0).sum()),
                "bootstrap_draws": 0,
                "requested_bootstrap_draws": int(draws),
                "bootstrap_ci_low": float("nan"),
                "bootstrap_ci_high": float("nan"),
                "signflip_patterns": 0,
                "signflip_p_one_sided": float("nan"),
                "finite_lopo": 0,
                "minimum_lopo_delta": float("nan"),
                "positive_gain_sum": positive_sum,
                "contribution_concentration": concentration,
                "concentration_gate": False,
                "root_seed": int(seed),
                "status": DATA_INVALID_STATUS,
            }
        ]
    )
    bootstrap = pd.DataFrame(
        columns=("draw", "draw_seed", "mean_delta", "participant_denominator")
    )
    lopo = pd.DataFrame(
        columns=("held_out_position", "lopo_mean_delta", "participant_denominator")
    )
    return ParticipantDeltaTables(summary=summary, bootstrap_draws=bootstrap, lopo=lopo)


_TARGET_COMPLETE_KEY_COLUMNS = (
    "fold",
    "sensor_day_id",
    "subject_id",
    "lifelog_date",
    "sleep_date",
    "elapsed_day_index",
)


def _target_grid_fields(paired: pd.DataFrame) -> dict[str, Any]:
    observed_targets = set(paired["target"].drop_duplicates())
    exact_target_identity = observed_targets == set(TARGET_COLUMNS)
    target_rows = paired.groupby("target", sort=False, dropna=False).size()
    minimum_rows = int(target_rows.min()) if len(target_rows) else 0
    maximum_rows = int(target_rows.max()) if len(target_rows) else 0
    identical_keys = exact_target_identity
    reference: pd.DataFrame | None = None
    if exact_target_identity:
        for target in TARGET_COLUMNS:
            current = paired.loc[
                paired["target"].eq(target),
                _TARGET_COMPLETE_KEY_COLUMNS,
            ].sort_values(list(_TARGET_COMPLETE_KEY_COLUMNS), kind="stable")
            current = current.reset_index(drop=True)
            if current.duplicated(list(_TARGET_COMPLETE_KEY_COLUMNS)).any():
                identical_keys = False
                break
            if reference is None:
                reference = current
            elif not current.equals(reference):
                identical_keys = False
                break
    return {
        "exact_target_identity": bool(exact_target_identity),
        "minimum_target_sensor_rows": minimum_rows,
        "maximum_target_sensor_rows": maximum_rows,
        "identical_complete_target_keys": bool(identical_keys),
    }


def _model_gate_fields(row: pd.Series) -> dict[str, bool]:
    return {
        "participant_denominator_gate": int(row["participant_denominator"]) == 10,
        "target_denominator_gate": int(row["target_denominator"]) == 4,
        "data_grid_gate": (
            int(row["common_sensor_rows"]) == 275
            and int(row["participant_target_cells"]) == 40
            and int(row["participants_with_four_targets"]) == 10
            and int(row["minimum_target_participants"]) == 10
            and int(row["maximum_target_participants"]) == 10
            and bool(row["exact_target_identity"])
            and int(row["minimum_target_sensor_rows"]) == 275
            and int(row["maximum_target_sensor_rows"]) == 275
            and bool(row["identical_complete_target_keys"])
        ),
        "macro_improvement_gate": float(row["macro_delta"]) >= 0.005,
        "participant_improvement_gate": int(row["improved_participants"]) >= 7,
        "bootstrap_gate": float(row["bootstrap_ci_low"]) > 0.0,
        "signflip_gate": float(row["signflip_p_one_sided"]) <= 0.05,
        "lopo_gate": (
            int(row["finite_lopo"]) == 10
            and float(row["minimum_lopo_delta"]) > 0.0
        ),
        "concentration_gate": (
            np.isfinite(float(row["contribution_concentration"]))
            and float(row["contribution_concentration"]) <= 0.5
        ),
}


def evaluate_g4_decision(contrast_summary: pd.DataFrame) -> pd.DataFrame:
    """Fail closed on exact primary identity/grid, then require every G4 gate."""
    required = {
        "model",
        "is_primary_contrast",
        "contrast_id",
        "baseline_arm",
        "candidate_arm",
        "participant_denominator",
        "target_denominator",
        "common_sensor_rows",
        "participant_target_cells",
        "participants_with_four_targets",
        "minimum_target_participants",
        "maximum_target_participants",
        "exact_target_identity",
        "minimum_target_sensor_rows",
        "maximum_target_sensor_rows",
        "identical_complete_target_keys",
        "macro_delta",
        "improved_participants",
        "bootstrap_ci_low",
        "signflip_p_one_sided",
        "finite_lopo",
        "minimum_lopo_delta",
        "contribution_concentration",
    }
    missing = sorted(required.difference(contrast_summary.columns))
    invalid_reasons: list[str] = []
    if missing:
        invalid_reasons.append(f"missing_columns={','.join(missing)}")
        flagged = contrast_summary.iloc[0:0].copy()
        primary = flagged
    else:
        flagged = contrast_summary.loc[
            contrast_summary["is_primary_contrast"].eq(True)
        ].copy()
        identity = (
            flagged["contrast_id"].eq(PRIMARY_CONTRAST_ID)
            & flagged["baseline_arm"].eq(PRIMARY_BASELINE_ARM)
            & flagged["candidate_arm"].eq(PRIMARY_CANDIDATE_ARM)
        )
        if len(flagged) != 2 or not bool(identity.all()):
            invalid_reasons.append("primary_identity_or_row_count")
        primary = flagged.loc[identity].copy()

    model_passes: dict[str, bool] = {}
    model_data_valid: dict[str, bool] = {}
    for model in MODEL_NAMES:
        rows = (
            primary.loc[primary["model"].eq(model)]
            if "model" in primary.columns
            else primary.iloc[0:0]
        )
        if len(rows) != 1:
            model_passes[model] = False
            model_data_valid[model] = False
            invalid_reasons.append(f"{model}_primary_row_count")
            continue
        gates = _model_gate_fields(rows.iloc[0])
        model_data_valid[model] = bool(
            gates["participant_denominator_gate"]
            and gates["target_denominator_gate"]
            and gates["data_grid_gate"]
        )
        if not model_data_valid[model]:
            failed_data_gates = ",".join(
                gate
                for gate in (
                    "participant_denominator_gate",
                    "target_denominator_gate",
                    "data_grid_gate",
                )
                if not gates[gate]
            )
            invalid_reasons.append(f"{model}_primary_grid={failed_data_gates}")
        model_passes[model] = bool(model_data_valid[model] and all(gates.values()))
    both_models_present = set(primary["model"]) == set(MODEL_NAMES) and len(primary) == 2
    if not both_models_present:
        invalid_reasons.append("both_models_not_exact")
    data_valid = bool(
        not invalid_reasons
        and both_models_present
        and all(model_data_valid.get(model, False) for model in MODEL_NAMES)
    )
    g4_pass = bool(data_valid and all(model_passes.values()))
    status = DATA_INVALID_STATUS if not data_valid else ("pass" if g4_pass else "fail")
    return pd.DataFrame.from_records(
        [
            {
                "primary_contrast": PRIMARY_CONTRAST_ID,
                "elasticnet_pass": model_passes.get("elasticnet", False),
                "lightgbm_pass": model_passes.get("lightgbm", False),
                "both_models_present": both_models_present,
                "data_valid": data_valid,
                "data_invalid_reasons": ";".join(dict.fromkeys(invalid_reasons)),
                "g4_pass": g4_pass,
                "g4_is_secondary": True,
                "can_rescue_measurement": False,
                "status": status,
            }
        ]
    )


def _empty_contrast_tables() -> OutcomeContrastTables:
    empty = pd.DataFrame()
    summary = pd.DataFrame.from_records(
        [
            {
                "data_valid": False,
                "data_invalid_reasons": "primary_predictions_unavailable",
                "g4_pass": False,
                "g4_is_secondary": True,
                "can_rescue_measurement": False,
                "status": DATA_INVALID_STATUS,
            }
        ]
    )
    return OutcomeContrastTables(
        paired_rows=empty,
        participant_target=empty,
        participant=empty,
        target=empty,
        bootstrap_draws=empty,
        lopo=empty,
        contrast_summary=empty,
        target_holm=empty,
        decision_table=empty,
        g4_summary=summary,
    )


def summarize_outcome_deltas(
    predictions: pd.DataFrame,
    *,
    bootstrap_draws: int = DEFAULT_OUTCOME_BOOTSTRAP_DRAWS,
    seed: int = 42,
) -> OutcomeContrastTables:
    """Retain every fixed arm comparison with participant-paired uncertainty."""
    available_arms = [arm for arm in ARM_NAMES if arm in set(predictions["arm"])]
    if len(available_arms) < 2:
        return _empty_contrast_tables()
    paired_tables: list[pd.DataFrame] = []
    participant_target_tables: list[pd.DataFrame] = []
    participant_tables: list[pd.DataFrame] = []
    target_tables: list[pd.DataFrame] = []
    bootstrap_tables: list[pd.DataFrame] = []
    lopo_tables: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    target_p_records: list[dict[str, Any]] = []

    for model in [name for name in MODEL_NAMES if name in set(predictions["model"])]:
        model_predictions = predictions.loc[predictions["model"].eq(model)]
        for baseline_arm, candidate_arm in combinations(available_arms, 2):
            contrast_id = f"{candidate_arm}__vs__{baseline_arm}"
            baseline = model_predictions.loc[model_predictions["arm"].eq(baseline_arm)]
            candidate = model_predictions.loc[model_predictions["arm"].eq(candidate_arm)]
            paired = pair_arm_predictions(baseline, candidate)
            paired.insert(0, "contrast_id", contrast_id)
            paired_tables.append(paired)
            participant_target = (
                paired.groupby(["subject_id", "target"], sort=False, dropna=False)
                .agg(
                    rows=("sensor_day_id", "size"),
                    baseline_logloss=("baseline_logloss", "mean"),
                    candidate_logloss=("candidate_logloss", "mean"),
                    delta=("row_delta", "mean"),
                )
                .reset_index()
            )
            participant_target.insert(0, "model", model)
            participant_target.insert(0, "candidate_arm", candidate_arm)
            participant_target.insert(0, "baseline_arm", baseline_arm)
            participant_target.insert(0, "contrast_id", contrast_id)
            participant_target_tables.append(participant_target)
            participant = (
                participant_target.groupby("subject_id", sort=False, dropna=False)
                .agg(
                    target_count=("target", "nunique"),
                    baseline_logloss=("baseline_logloss", "mean"),
                    candidate_logloss=("candidate_logloss", "mean"),
                    delta=("delta", "mean"),
                )
                .reset_index()
            )
            participant.insert(0, "model", model)
            participant.insert(0, "candidate_arm", candidate_arm)
            participant.insert(0, "baseline_arm", baseline_arm)
            participant.insert(0, "contrast_id", contrast_id)
            participant_tables.append(participant)
            ordered_participants = participant.sort_values("subject_id", key=lambda s: s.map(str))
            participant_deltas = ordered_participants["delta"].to_numpy(dtype=float)
            if len(participant_deltas) == 10 and np.isfinite(participant_deltas).all():
                uncertainty = summarize_participant_deltas(
                    participant_deltas,
                    draws=bootstrap_draws,
                    seed=seed,
                    contrast_id=f"{model}:{contrast_id}",
                )
            else:
                uncertainty = _invalid_participant_delta_tables(
                    participant_deltas,
                    draws=bootstrap_draws,
                    seed=seed,
                    contrast_id=f"{model}:{contrast_id}",
                )
            uncertainty_summary = uncertainty.summary.iloc[0].to_dict()
            target_denominator = int(participant_target["target"].nunique())
            target_participant_counts = participant_target.groupby(
                "target", sort=False, dropna=False
            )["subject_id"].nunique()
            participant_target_counts = participant_target.groupby(
                "subject_id", sort=False, dropna=False
            )["target"].nunique()
            target_sensor_sets = [
                set(group["sensor_day_id"])
                for _, group in paired.groupby("target", sort=False, dropna=False)
            ]
            common_sensor_rows = (
                len(set.intersection(*target_sensor_sets)) if target_sensor_sets else 0
            )
            target_grid = _target_grid_fields(paired)
            record = {
                "contrast_id": contrast_id,
                "baseline_arm": baseline_arm,
                "candidate_arm": candidate_arm,
                "model": model,
                "is_primary_contrast": bool(
                    baseline_arm == PRIMARY_BASELINE_ARM
                    and candidate_arm == PRIMARY_CANDIDATE_ARM
                ),
                "is_personalization_contrast": bool(
                    baseline_arm == "semantic_raw"
                    and candidate_arm == "semantic_personalized"
                ),
                "target_denominator": target_denominator,
                "common_sensor_rows": common_sensor_rows,
                "participant_target_cells": len(participant_target),
                "participants_with_four_targets": int(
                    participant_target_counts.eq(4).sum()
                ),
                "minimum_target_participants": int(
                    target_participant_counts.min()
                ),
                "maximum_target_participants": int(
                    target_participant_counts.max()
                ),
                **target_grid,
                **{
                    key: value
                    for key, value in uncertainty_summary.items()
                    if key != "contrast_id"
                },
            }
            gates = _model_gate_fields(pd.Series(record))
            record.update(gates)
            record["model_pass"] = bool(all(gates.values()))
            summary_records.append(record)
            current_bootstrap = uncertainty.bootstrap_draws.copy()
            current_bootstrap.insert(0, "model", model)
            current_bootstrap.insert(0, "contrast_id", contrast_id)
            bootstrap_tables.append(current_bootstrap)
            current_lopo = uncertainty.lopo.copy()
            current_lopo.insert(0, "model", model)
            current_lopo.insert(0, "contrast_id", contrast_id)
            lopo_tables.append(current_lopo)

            target_summary = (
                participant_target.groupby("target", sort=False, dropna=False)
                .agg(
                    participant_denominator=("subject_id", "nunique"),
                    baseline_logloss=("baseline_logloss", "mean"),
                    candidate_logloss=("candidate_logloss", "mean"),
                    delta=("delta", "mean"),
                )
                .reset_index()
            )
            target_summary.insert(0, "model", model)
            target_summary.insert(0, "candidate_arm", candidate_arm)
            target_summary.insert(0, "baseline_arm", baseline_arm)
            target_summary.insert(0, "contrast_id", contrast_id)
            target_tables.append(target_summary)
            for target_name, target_group in participant_target.groupby(
                "target", sort=False, dropna=False
            ):
                ordered = target_group.sort_values("subject_id", key=lambda s: s.map(str))
                target_values = ordered["delta"].to_numpy(dtype=float)
                p_value, patterns = _exact_signflip_p(target_values)
                target_p_records.append(
                    {
                        "contrast_id": contrast_id,
                        "model": model,
                        "target": target_name,
                        "participant_denominator": len(target_values),
                        "target_delta": float(target_values.mean()),
                        "signflip_patterns": patterns,
                        "raw_p": p_value,
                    }
                )

    contrast_summary = pd.DataFrame.from_records(summary_records)
    target_holm = pd.DataFrame.from_records(target_p_records)
    if not target_holm.empty:
        adjusted_groups: list[pd.DataFrame] = []
        for (_contrast, _model), group in target_holm.groupby(
            ["contrast_id", "model"], sort=False, dropna=False
        ):
            current = group.copy()
            current["holm_p"] = holm_adjust(current["raw_p"].to_numpy(dtype=float))
            current["holm_gate"] = current["holm_p"].lt(0.05)
            adjusted_groups.append(current)
        target_holm = pd.concat(adjusted_groups, ignore_index=True)
    g4_summary = evaluate_g4_decision(contrast_summary)
    decision_records: list[dict[str, Any]] = []
    gate_names = (
        "participant_denominator_gate",
        "target_denominator_gate",
        "data_grid_gate",
        "macro_improvement_gate",
        "participant_improvement_gate",
        "bootstrap_gate",
        "signflip_gate",
        "lopo_gate",
        "concentration_gate",
    )
    for row in contrast_summary.loc[contrast_summary["is_primary_contrast"]].itertuples(
        index=False
    ):
        for gate in gate_names:
            passed = bool(getattr(row, gate))
            decision_records.append(
                {
                    "level": "model",
                    "model": row.model,
                    "contrast_id": row.contrast_id,
                    "gate": gate,
                    "passed": passed,
                    "status": "pass" if passed else "fail",
                    "g4_is_secondary": True,
                    "can_rescue_measurement": False,
                }
            )
    decision_records.append(
        {
            "level": "g4",
            "model": "both",
            "contrast_id": PRIMARY_CONTRAST_ID,
            "gate": "both_fixed_models_all_conditions",
            "passed": bool(g4_summary.loc[0, "g4_pass"]),
            "status": g4_summary.loc[0, "status"],
            "g4_is_secondary": True,
            "can_rescue_measurement": False,
        }
    )
    return OutcomeContrastTables(
        paired_rows=pd.concat(paired_tables, ignore_index=True),
        participant_target=pd.concat(participant_target_tables, ignore_index=True),
        participant=pd.concat(participant_tables, ignore_index=True),
        target=pd.concat(target_tables, ignore_index=True),
        bootstrap_draws=pd.concat(bootstrap_tables, ignore_index=True),
        lopo=pd.concat(lopo_tables, ignore_index=True),
        contrast_summary=contrast_summary,
        target_holm=target_holm,
        decision_table=pd.DataFrame.from_records(decision_records),
        g4_summary=g4_summary,
    )


def _attach_targets_to_evaluation(
    evaluation_keys: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    label_keys = ["subject_id", "lifelog_date", "sleep_date"]
    missing = sorted(set(label_keys + list(TARGET_COLUMNS)).difference(labels.columns))
    if missing:
        raise ValueError(f"Missing evaluation label columns: {missing}")
    outcomes = labels.copy()
    for column in ("lifelog_date", "sleep_date"):
        outcomes[column] = _validated_midnight(outcomes[column], column)
    if not (outcomes["sleep_date"] - outcomes["lifelog_date"]).eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError("sleep_date provenance must be exactly one day after lifelog_date")
    if outcomes.duplicated(label_keys).any():
        raise ValueError("Outcome participant-date provenance must be unique")
    for target in TARGET_COLUMNS:
        numeric = pd.to_numeric(outcomes[target], errors="coerce")
        if numeric.isna().any() or not numeric.isin((0, 1)).all():
            raise ValueError(f"{target} must be an exact binary outcome")
        outcomes[target] = numeric.astype(int)
    joined = evaluation_keys.merge(
        outcomes[label_keys + list(TARGET_COLUMNS)],
        on=label_keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Evaluation labels do not match the frozen 275 sensor rows")
    return joined.drop(columns="_merge")


def _evaluate_outcome_arms(
    labels: pd.DataFrame,
    arms: OutcomeArmTables,
    *,
    trusted_build_root: str,
    models: tuple[str, ...] = MODEL_NAMES,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    bootstrap_draws: int = DEFAULT_OUTCOME_BOOTSTRAP_DRAWS,
    seed: int = 42,
) -> OutcomeEvaluationResult:
    """Run exact LOSO on six fixed arms with fold-local preprocessing."""
    interval_boundary_policy, primitive_digest = _validated_arm_source_provenance(
        arms
    )
    production_allowlists = _derive_production_feature_allowlists(
        arms,
        trusted_build_root=trusted_build_root,
    )
    requested_models = tuple(models)
    requested_targets = tuple(targets)
    if not requested_models or len(set(requested_models)) != len(requested_models):
        raise ValueError("models must be a nonempty unique frozen subset")
    if not set(requested_models).issubset(MODEL_NAMES):
        raise ValueError("models contains an unknown model")
    if not requested_targets or len(set(requested_targets)) != len(requested_targets):
        raise ValueError("targets must be a nonempty unique S1-S4 subset")
    if not set(requested_targets).issubset(TARGET_COLUMNS):
        raise ValueError("targets contains a non-S1-S4 outcome")
    evaluation = _attach_targets_to_evaluation(arms.evaluation_keys, labels)
    if len(evaluation) != 275 or evaluation["subject_id"].nunique() != 10:
        raise ValueError("G4 evaluation must retain exactly 275 rows and 10 participants")
    folds = make_loso_splits(evaluation)

    prediction_records: list[dict[str, Any]] = []
    preprocessor_records: list[dict[str, Any]] = []
    for fold in folds:
        for arm in ARM_NAMES:
            frame = (
                arms.frames[(fold.fold, arm)]
                .set_index("sensor_day_id", drop=False)
                .reindex(evaluation["sensor_day_id"].to_numpy())
                .reset_index(drop=True)
            )
            if not frame["sensor_day_id"].equals(evaluation["sensor_day_id"]):
                raise ValueError("Every fold arm must use the identical 275 sensor rows")
            feature_names = production_allowlists[arm]
            selected_features = list(feature_names)
            transformed = fit_fold_preprocessor(
                frame.iloc[fold.train_positions][selected_features],
                frame.iloc[fold.test_positions][selected_features],
                declared_feature_names=feature_names,
            )
            preprocessor_records.append(
                {
                    "fold": fold.fold,
                    "held_out_subject": fold.held_out_subject,
                    "arm": arm,
                    "training_rows": len(fold.train_positions),
                    "test_rows": len(fold.test_positions),
                    "input_feature_count": len(feature_names),
                    "fitted_feature_count": len(transformed.feature_names),
                    "feature_names": json.dumps(transformed.feature_names),
                    "train_medians": json.dumps(transformed.train_medians, sort_keys=True),
                    "train_scales": json.dumps(transformed.train_scales, sort_keys=True),
                    "status_vocabulary": json.dumps(transformed.status_vocabulary),
                    "status": "ok",
                }
            )
            for target in requested_targets:
                y_train = evaluation.iloc[fold.train_positions][target].to_numpy(dtype=int)
                y_test = evaluation.iloc[fold.test_positions][target].to_numpy(dtype=int)
                for model in requested_models:
                    model_seed = outcome_model_seed(
                        seed,
                        model,
                        target,
                        fold.fold,
                        fold.held_out_subject,
                    )
                    fitted = fit_predict_fixed_model(
                        model,
                        transformed.train,
                        transformed.test,
                        y_train,
                        seed=model_seed,
                    )
                    test_rows = evaluation.iloc[fold.test_positions]
                    for position, probability, truth in zip(
                        range(len(test_rows)),
                        fitted.probabilities,
                        y_test,
                        strict=True,
                    ):
                        row = test_rows.iloc[position]
                        prediction_records.append(
                            {
                                "sensor_day_id": row["sensor_day_id"],
                                "subject_id": row["subject_id"],
                                "lifelog_date": row["lifelog_date"],
                                "sleep_date": row["sleep_date"],
                                "elapsed_day_index": row["elapsed_day_index"],
                                "target": target,
                                "arm": arm,
                                "model": model,
                                "fold": fold.fold,
                                "held_out_subject": fold.held_out_subject,
                                "truth": int(truth),
                                "probability": float(probability),
                                "training_count": fitted.training_count,
                                "training_prevalence": fitted.training_prevalence,
                                "feature_count": fitted.feature_count,
                                "seed": model_seed,
                                "status": fitted.status,
                                "converged": fitted.converged,
                            }
                        )
    predictions = pd.DataFrame.from_records(prediction_records)
    predictions = predictions.sort_values(
        ["arm", "model", "target", "sensor_day_id"], kind="stable"
    ).reset_index(drop=True)
    expected_rows = len(evaluation) * len(ARM_NAMES) * len(requested_models) * len(
        requested_targets
    )
    if len(predictions) != expected_rows:
        raise RuntimeError("Outcome prediction table changed the fixed row denominator")
    scores = score_outcome_predictions(predictions)
    contrasts = summarize_outcome_deltas(
        predictions,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )
    production_eligible = bool(
        requested_models == MODEL_NAMES
        and requested_targets == TARGET_COLUMNS
        and int(bootstrap_draws) == DEFAULT_OUTCOME_BOOTSTRAP_DRAWS
    )
    manifest = pd.DataFrame.from_records(
        [
            {
                "evaluation_rows": len(evaluation),
                "participant_denominator": evaluation["subject_id"].nunique(),
                "outer_folds": len(folds),
                "arms": len(ARM_NAMES),
                "models": len(requested_models),
                "targets": len(requested_targets),
                "prediction_rows": len(predictions),
                "bootstrap_draws": int(bootstrap_draws),
                "interval_boundary_policy": interval_boundary_policy,
                "primitive_digest": primitive_digest,
                "frame_manifest_digest": arms.manifest.loc[
                    0, "frame_manifest_digest"
                ],
                "outcome_build_digest": trusted_build_root,
                "production_defaults": production_eligible,
                "g4_is_secondary": True,
                "can_rescue_measurement": False,
                "status": "complete" if production_eligible else "nonproduction_grid",
            }
        ]
    )
    return OutcomeEvaluationResult(
        predictions=predictions,
        preprocessor_audit=pd.DataFrame.from_records(preprocessor_records),
        scores=scores,
        contrasts=contrasts,
        manifest=manifest,
    )


def evaluate_outcomes(
    primitives: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    interval_boundary_policy: str = "measurement_support",
    models: tuple[str, ...] = MODEL_NAMES,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    bootstrap_draws: int = DEFAULT_OUTCOME_BOOTSTRAP_DRAWS,
    seed: int = 42,
) -> OutcomeEvaluationResult:
    """Rebuild trusted G4 arms from primitives, then run exact LOSO scoring."""
    if not isinstance(primitives, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        raise TypeError(
            "Production G4 evaluation requires primitive and label DataFrames; "
            "prebuilt arm tables are not accepted"
        )
    arms = build_outcome_arms(
        primitives,
        labels,
        interval_boundary_policy=interval_boundary_policy,
    )
    trusted_build_root = arms.manifest.loc[0, "outcome_build_digest"]
    return _evaluate_outcome_arms(
        labels,
        arms,
        trusted_build_root=trusted_build_root,
        models=models,
        targets=targets,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )


_SENSITIVITY_KEY_COLUMNS = (
    "arm",
    "model",
    "target",
    "fold",
    "held_out_subject",
    "sensor_day_id",
    "subject_id",
    "lifelog_date",
    "sleep_date",
    "elapsed_day_index",
    "truth",
)
_SENSITIVITY_PRIMITIVE_KEY_COLUMNS = (
    "sensor_day_id",
    "subject_id",
    "lifelog_date",
    "sleep_date",
    "elapsed_day_index",
)


def _sorted_sensitivity_keys(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_SENSITIVITY_KEY_COLUMNS).difference(predictions.columns))
    if missing:
        raise ValueError(f"DATA-INVALID: sensitivity predictions lack keys {missing}")
    if predictions.duplicated(list(_SENSITIVITY_KEY_COLUMNS)).any():
        raise ValueError("DATA-INVALID: sensitivity OOF keys are duplicated")
    return predictions.loc[:, _SENSITIVITY_KEY_COLUMNS].sort_values(
        list(_SENSITIVITY_KEY_COLUMNS), kind="stable"
    ).reset_index(drop=True)


def _canonical_sensitivity_primitive_keys(primitives: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(
        set(_SENSITIVITY_PRIMITIVE_KEY_COLUMNS).difference(primitives.columns)
    )
    if missing:
        raise ValueError(
            f"DATA-INVALID: sensitivity primitives lack full-grid keys {missing}"
        )
    keys = primitives.loc[:, _SENSITIVITY_PRIMITIVE_KEY_COLUMNS].copy()
    if (
        keys.isna().any().any()
        or keys["sensor_day_id"].duplicated().any()
        or keys.duplicated(
            ["subject_id", "lifelog_date", "sleep_date", "elapsed_day_index"]
        ).any()
    ):
        raise ValueError(
            "DATA-INVALID: sensitivity primitive full-grid keys are incomplete"
        )
    return keys.sort_values("sensor_day_id", kind="stable").reset_index(drop=True)


def evaluate_outcome_sensitivity(
    primary_primitives: pd.DataFrame,
    strict_source_primitives: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    models: tuple[str, ...] = MODEL_NAMES,
    targets: tuple[str, ...] = TARGET_COLUMNS,
    bootstrap_draws: int = DEFAULT_OUTCOME_BOOTSTRAP_DRAWS,
    seed: int = 42,
) -> OutcomeSensitivityResult:
    """Internally rebuild primary and strict arms before separate evaluation."""
    if (
        not isinstance(primary_primitives, pd.DataFrame)
        or not isinstance(strict_source_primitives, pd.DataFrame)
        or not isinstance(labels, pd.DataFrame)
    ):
        raise TypeError(
            "G4 sensitivity requires primary/strict primitive and label DataFrames; "
            "prebuilt arm tables are not accepted"
        )
    primary_primitive_keys = _canonical_sensitivity_primitive_keys(
        primary_primitives
    )
    strict_primitive_keys = _canonical_sensitivity_primitive_keys(
        strict_source_primitives
    )
    if not primary_primitive_keys.equals(strict_primitive_keys):
        raise ValueError(
            "DATA-INVALID: primary and strict primitive full-grid keys differ"
        )
    primary_arms = build_outcome_arms(
        primary_primitives,
        labels,
        interval_boundary_policy="measurement_support",
    )
    strict_source_arms = build_outcome_arms(
        strict_source_primitives,
        labels,
        interval_boundary_policy="strict_source_availability",
    )
    if not primary_arms.evaluation_keys.equals(strict_source_arms.evaluation_keys):
        raise ValueError("DATA-INVALID: primary and strict evaluation keys differ")
    for name, arms, expected_policy in (
        ("primary", primary_arms, "measurement_support"),
        ("strict", strict_source_arms, "strict_source_availability"),
    ):
        observed_policy, _digest = _validated_arm_source_provenance(arms)
        if observed_policy != expected_policy:
            raise ValueError(
                f"DATA-INVALID: {name} extraction provenance policy is not exact"
            )

    primary = _evaluate_outcome_arms(
        labels,
        primary_arms,
        trusted_build_root=primary_arms.manifest.loc[0, "outcome_build_digest"],
        models=models,
        targets=targets,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )
    strict = _evaluate_outcome_arms(
        labels,
        strict_source_arms,
        trusted_build_root=strict_source_arms.manifest.loc[
            0, "outcome_build_digest"
        ],
        models=models,
        targets=targets,
        bootstrap_draws=bootstrap_draws,
        seed=seed,
    )
    primary_keys = _sorted_sensitivity_keys(primary.predictions)
    strict_keys = _sorted_sensitivity_keys(strict.predictions)
    if not primary_keys.equals(strict_keys):
        raise ValueError("DATA-INVALID: primary and strict OOF keys differ")

    score_keys = ["arm", "model"]
    primary_macro = primary.scores.macro_scores.copy()
    strict_macro = strict.scores.macro_scores.copy()
    primary_macro = primary_macro.rename(
        columns={
            column: f"primary__{column}"
            for column in primary_macro.columns
            if column not in score_keys
        }
    )
    strict_macro = strict_macro.rename(
        columns={
            column: f"strict__{column}"
            for column in strict_macro.columns
            if column not in score_keys
        }
    )
    macro_comparison = primary_macro.merge(
        strict_macro,
        on=score_keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not macro_comparison["_merge"].eq("both").all():
        raise ValueError("DATA-INVALID: primary and strict score grids differ")
    macro_comparison = macro_comparison.drop(columns="_merge")
    for metric in (
        "participant_target_logloss",
        "participant_target_brier",
        "day_weighted_logloss",
        "day_weighted_brier",
    ):
        macro_comparison[f"strict_minus_primary__{metric}"] = (
            macro_comparison[f"strict__{metric}"]
            - macro_comparison[f"primary__{metric}"]
        )

    primary_g4 = primary.contrasts.g4_summary.iloc[0]
    strict_g4 = strict.contrasts.g4_summary.iloc[0]
    decision_comparison = pd.DataFrame.from_records(
        [
            {
                "analysis": "primary",
                "interval_boundary_policy": "measurement_support",
                "data_valid": bool(primary_g4.get("data_valid", False)),
                "g4_pass": bool(primary_g4["g4_pass"]),
                "status": primary_g4["status"],
                "can_replace_primary": True,
            },
            {
                "analysis": "sensitivity",
                "interval_boundary_policy": "strict_source_availability",
                "data_valid": bool(strict_g4.get("data_valid", False)),
                "g4_pass": bool(strict_g4["g4_pass"]),
                "status": strict_g4["status"],
                "can_replace_primary": False,
            },
        ]
    )
    primary_production = bool(primary.manifest.loc[0, "production_defaults"])
    strict_production = bool(strict.manifest.loc[0, "production_defaults"])
    if primary_production != strict_production:
        raise ValueError("DATA-INVALID: primary and strict production grids differ")
    if primary_production and (
        len(primary.predictions) != 13_200 or len(strict.predictions) != 13_200
    ):
        raise ValueError("DATA-INVALID: production sensitivity requires 13,200 rows each")
    manifest = pd.DataFrame.from_records(
        [
            {
                "primary_interval_boundary_policy": "measurement_support",
                "sensitivity_interval_boundary_policy": (
                    "strict_source_availability"
                ),
                "primary_prediction_rows": len(primary.predictions),
                "strict_prediction_rows": len(strict.predictions),
                "primary_outcome_build_digest": primary.manifest.loc[
                    0, "outcome_build_digest"
                ],
                "strict_outcome_build_digest": strict.manifest.loc[
                    0, "outcome_build_digest"
                ],
                "identical_oof_keys": True,
                "primary_g4_pass": bool(primary_g4["g4_pass"]),
                "strict_g4_pass": bool(strict_g4["g4_pass"]),
                "reported_g4_pass": bool(primary_g4["g4_pass"]),
                "primary_decision_unchanged": True,
                "strict_can_replace_primary": False,
                "production_defaults": primary_production,
                "status": "complete" if primary_production else "nonproduction_grid",
            }
        ]
    )
    return OutcomeSensitivityResult(
        primary=primary,
        strict_source_availability=strict,
        macro_comparison=macro_comparison,
        decision_comparison=decision_comparison,
        manifest=manifest,
    )
