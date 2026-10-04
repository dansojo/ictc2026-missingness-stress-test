"""Fail-closed copy-resume primitives for interrupted downstream prediction streams."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil
import struct
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from semantic_indicators import outcome
from semantic_indicators import stress_downstream
from semantic_indicators import stress_downstream_runner
from semantic_indicators import stress_downstream_statistics


@dataclass(frozen=True, slots=True)
class RecoveryArtifactPin:
    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.relative_path) is not str or not self.relative_path:
            raise TypeError("recovery artifact path must be an exact nonempty string")
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise TypeError("recovery artifact byte count must be an exact nonnegative integer")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or self.sha256.lower() != self.sha256
            or any(value not in "0123456789abcdef" for value in self.sha256)
        ):
            raise TypeError("recovery artifact SHA-256 must be exact lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class RecoveryPrefixSeal:
    root: Path
    participants: tuple[str, ...]
    batch_row_counts: tuple[int, ...]
    row_count: int
    next_row_seq: int
    observed_sha256: str


@dataclass(frozen=True, slots=True)
class RecoveryObservedSeal:
    path: Path
    participants: tuple[str, ...]
    batch_count: int
    row_count: int
    physical_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_tuple(values: object, name: str) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(value) is not str or not value for value in values):
        raise TypeError(f"{name} must be an exact tuple of nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates")
    return values


def _validate_artifacts(
    root: Path,
    artifact_pins: tuple[RecoveryArtifactPin, ...],
) -> None:
    if type(artifact_pins) is not tuple or any(
        type(value) is not RecoveryArtifactPin for value in artifact_pins
    ):
        raise TypeError("artifact_pins must contain exact RecoveryArtifactPin values")
    expected = tuple(value.relative_path for value in artifact_pins)
    if expected != tuple(sorted(expected)) or len(set(expected)) != len(expected):
        raise ValueError("recovery artifact pin paths are not unique canonical order")
    observed = tuple(
        sorted(
            value.relative_to(root).as_posix()
            for value in root.rglob("*")
            if value.is_file()
        )
    )
    if observed != expected:
        raise ValueError("recovery source artifact set mismatch")
    for pin in artifact_pins:
        path = root / Path(pin.relative_path)
        if not path.is_file() or path.is_symlink():
            raise ValueError("recovery artifact is absent or aliased")
        if path.stat().st_size != pin.byte_count or _file_sha256(path) != pin.sha256:
            raise ValueError(f"recovery artifact pin mismatch: {pin.relative_path}")


def _validate_batch(
    batch: pa.RecordBatch,
    *,
    expected_schema: pa.Schema,
    expected_participant: str,
    next_row_seq: int,
    row_validator: Callable[[dict[str, object]], None],
) -> int:
    if batch.schema != expected_schema:
        raise ValueError("recovery observed schema mismatch")
    records = batch.to_pylist()
    if not records:
        raise ValueError("recovery participant batch is empty")
    participants = {value["participant"] for value in records}
    if participants != {expected_participant}:
        raise ValueError("recovery participant batch order mismatch")
    for record in records:
        if type(record["row_seq"]) is not int or record["row_seq"] != next_row_seq:
            raise ValueError("recovery row_seq is not contiguous")
        row_validator(record)
        next_row_seq += 1
    return next_row_seq


def inspect_recovery_prefix(
    root: Path,
    *,
    artifact_pins: tuple[RecoveryArtifactPin, ...],
    expected_schema: pa.Schema,
    expected_participants: tuple[str, ...],
    expected_prefix_rows: int,
    row_validator: Callable[[dict[str, object]], None],
) -> RecoveryPrefixSeal:
    """Validate the immutable source files and every complete observed batch."""
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ValueError("recovery source root must be an existing unaliased directory")
    if not isinstance(expected_schema, pa.Schema):
        raise TypeError("expected_schema must be a PyArrow schema")
    participants = _require_exact_tuple(expected_participants, "expected_participants")
    if type(expected_prefix_rows) is not int or expected_prefix_rows < 1:
        raise TypeError("expected_prefix_rows must be a positive exact integer")
    if not callable(row_validator):
        raise TypeError("row_validator must be callable")
    _validate_artifacts(root, artifact_pins)
    observed = root / "observed_predictions.arrow"
    next_row_seq = 0
    row_counts: list[int] = []
    with observed.open("rb") as stream:
        reader = ipc.open_stream(stream)
        batches = list(reader)
    if len(batches) != len(participants):
        raise ValueError("recovery prefix batch count mismatch")
    for expected_participant, batch in zip(participants, batches, strict=True):
        before = next_row_seq
        next_row_seq = _validate_batch(
            batch,
            expected_schema=expected_schema,
            expected_participant=expected_participant,
            next_row_seq=next_row_seq,
            row_validator=row_validator,
        )
        row_counts.append(next_row_seq - before)
    if next_row_seq != expected_prefix_rows:
        raise ValueError("recovery prefix row count mismatch")
    return RecoveryPrefixSeal(
        root=root,
        participants=participants,
        batch_row_counts=tuple(row_counts),
        row_count=next_row_seq,
        next_row_seq=next_row_seq,
        observed_sha256=_file_sha256(observed),
    )


def recover_observed_stream(
    source_root: Path,
    destination: Path,
    *,
    artifact_pins: tuple[RecoveryArtifactPin, ...],
    expected_schema: pa.Schema,
    expected_prefix_participants: tuple[str, ...],
    expected_all_participants: tuple[str, ...],
    expected_prefix_rows: int,
    expected_total_rows: int,
    suffix_batch_factory: Callable[[int], Iterable[pa.RecordBatch]],
    row_validator: Callable[[dict[str, object]], None],
) -> RecoveryObservedSeal:
    """Copy a validated prefix and append only the canonical missing suffix."""
    if not isinstance(destination, Path) or destination.exists():
        raise ValueError("recovery destination must be an absent pathlib Path")
    all_participants = _require_exact_tuple(
        expected_all_participants, "expected_all_participants"
    )
    prefix_participants = _require_exact_tuple(
        expected_prefix_participants, "expected_prefix_participants"
    )
    if all_participants[: len(prefix_participants)] != prefix_participants:
        raise ValueError("recovery prefix is not a canonical participant prefix")
    if type(expected_total_rows) is not int or expected_total_rows <= expected_prefix_rows:
        raise TypeError("expected_total_rows must exceed the exact prefix row count")
    if not callable(suffix_batch_factory):
        raise TypeError("suffix_batch_factory must be callable")
    prefix = inspect_recovery_prefix(
        source_root,
        artifact_pins=artifact_pins,
        expected_schema=expected_schema,
        expected_participants=prefix_participants,
        expected_prefix_rows=expected_prefix_rows,
        row_validator=row_validator,
    )
    next_row_seq = prefix.next_row_seq
    observed_participants: list[str] = []
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output_stream:
            with ipc.new_stream(output_stream, expected_schema) as writer:
                with (source_root / "observed_predictions.arrow").open("rb") as input_stream:
                    for batch in ipc.open_stream(input_stream):
                        writer.write_batch(batch)
                        observed_participants.append(
                            str(batch.column(batch.schema.get_field_index("participant"))[0].as_py())
                        )
                missing = all_participants[len(prefix_participants) :]
                suffix = iter(suffix_batch_factory(next_row_seq))
                for expected_participant in missing:
                    try:
                        batch = next(suffix)
                    except StopIteration as error:
                        raise ValueError("recovery suffix batch count mismatch") from error
                    next_row_seq = _validate_batch(
                        batch,
                        expected_schema=expected_schema,
                        expected_participant=expected_participant,
                        next_row_seq=next_row_seq,
                        row_validator=row_validator,
                    )
                    writer.write_batch(batch)
                    observed_participants.append(expected_participant)
                try:
                    next(suffix)
                except StopIteration:
                    pass
                else:
                    raise ValueError("recovery suffix batch count mismatch")
        if tuple(observed_participants) != all_participants:
            raise ValueError("recovery final participant order mismatch")
        if next_row_seq != expected_total_rows:
            raise ValueError("recovery final row count mismatch")
        return RecoveryObservedSeal(
            path=destination,
            participants=tuple(observed_participants),
            batch_count=len(observed_participants),
            row_count=next_row_seq,
            physical_sha256=_file_sha256(destination),
        )
    except BaseException:
        if destination.exists():
            destination.unlink()
        raise


def derive_fit_authority_digest(root: Path) -> str:
    """Reconstruct the saved fit authority root before reopening all models."""
    if not isinstance(root, Path) or not root.is_dir():
        raise ValueError("fit-pack root must be an existing pathlib Path")
    ledger = stress_downstream_runner._read_ipc_table(
        root / "fit_ledger.arrow", stress_downstream_runner._FIT_LEDGER_SCHEMA
    ).to_pylist()
    index = stress_downstream_runner._read_ipc_table(
        root / "fit_state_index.arrow", stress_downstream_runner._FIT_INDEX_SCHEMA
    ).to_pylist()
    if len(ledger) != 160 or len(index) != 160:
        raise ValueError("recovery fit pack topology is not exactly 160")
    pack = (root / "fit_states.pack").read_bytes()
    header = stress_downstream_runner._PACK_HEADER
    if not pack.startswith(header) or len(pack) < len(header) + 8:
        raise ValueError("recovery fit pack header is invalid")
    length = struct.unpack(">Q", pack[len(header) : len(header) + 8])[0]
    end = len(header) + 8 + length
    if end > len(pack):
        raise ValueError("recovery first fit record is truncated")
    payload = stress_downstream_runner._strict_json(pack[len(header) + 8 : end])
    if type(payload) is not dict:
        raise ValueError("recovery first fit payload is invalid")
    outcome_digest = stress_downstream_runner._require_digest(
        payload.get("outcome_build_digest"), "recovery outcome build digest"
    )
    reference_digest = stress_downstream_runner._require_digest(
        payload.get("reference_content_digest"), "recovery reference digest"
    )
    ledger_hasher = stress_downstream_runner._LogicalHasher(
        "task-12-downstream-fit-ledger-logical-v1",
        stress_downstream_runner._FIT_LEDGER_COLUMNS,
    )
    projection_hasher = stress_downstream_runner._LogicalHasher(
        "task-12-downstream-fit-ledger-projection-digest-v1",
        stress_downstream_statistics.FIT_PROJECTION_COLUMNS,
    )
    index_hasher = stress_downstream_runner._LogicalHasher(
        "task-12-downstream-fit-state-index-logical-v1",
        stress_downstream_runner._FIT_INDEX_COLUMNS,
    )
    for row in ledger:
        ledger_hasher.update(row)
        projection_hasher.update(
            {
                name: row[name]
                for name, _kind, _nullable in stress_downstream_statistics.FIT_PROJECTION_COLUMNS
            }
        )
    for row in index:
        index_hasher.update(row)
    return stress_downstream_runner._fit_authority_digest(
        outcome_build_digest=outcome_digest,
        reference_content_digest=reference_digest,
        fit_ledger_row_count=ledger_hasher.row_count,
        fit_ledger_logical_digest=ledger_hasher.hexdigest(),
        fit_ledger_projection_digest=projection_hasher.hexdigest(),
        fit_state_index_row_count=index_hasher.row_count,
        fit_state_index_logical_digest=index_hasher.hexdigest(),
        fit_state_record_count=len(index),
        fit_state_pack_byte_count=len(pack),
        fit_state_pack_sha256=hashlib.sha256(pack).hexdigest(),
    )


def _validate_observed_row(row: dict[str, object]) -> None:
    expected = stress_downstream_statistics._row_digest(
        "task-12-downstream-observed-prediction-row-v1",
        row,
        stress_downstream_statistics._OBSERVED_DIGEST_NAMES,
    )
    if row.get("row_content_digest") != expected:
        raise ValueError("recovery observed row self-digest mismatch")


def _suffix_batches(
    snapshot: stress_downstream_runner.DownstreamAdapterSnapshot,
    loaded: stress_downstream_runner.DownstreamLoadedFitStates,
    scoring_labels: pd.DataFrame | None,
    *,
    completed_count: int,
    first_row_seq: int,
) -> Iterable[pa.RecordBatch]:
    model_by_seq = {index: model for index, model in enumerate(loaded.models)}
    ledger_by_seq = {
        int(row["fit_seq"]): row for row in loaded.ledger.to_dict("records")
    }
    contexts = {
        (value.held_out_subject, value.sensor_day_id): value for value in snapshot.contexts
    }
    patches = {
        (
            value.held_out_subject,
            value.sensor_day_id,
            value.primitive,
            value.draw,
            value.condition,
            value.arm,
        ): value
        for value in snapshot.patches
    }
    labels = stress_downstream_runner._scoring_lookup(scoring_labels)
    row_seq = first_row_seq
    probability_cache: dict[tuple[int, int], float] = {}
    for participant_position, (fold, participant) in enumerate(snapshot.fold_roster):
        if participant_position < completed_count:
            continue
        participant_rows: list[dict[str, object]] = []
        for family_position, family in enumerate(stress_downstream_runner._FAMILIES):
            selected_rows = snapshot.prediction_universe.loc[
                snapshot.prediction_universe["participant"].eq(participant)
                & snapshot.prediction_universe["family"].eq(family)
            ].sort_values(
                ["primitive_position", "sensor_day_id", "draw"], kind="stable"
            )
            for arm_position, arm in enumerate(outcome.DOWNSTREAM_ARM_NAMES):
                for model_position, model_name in enumerate(outcome.MODEL_NAMES):
                    for target_position, target in enumerate(outcome.TARGET_COLUMNS):
                        fit_seq = (
                            ((participant_position * 2 + arm_position) * 2 + model_position)
                            * 4
                            + target_position
                        )
                        model = model_by_seq[fit_seq]
                        ledger = ledger_by_seq[fit_seq]
                        for selected in selected_rows.to_dict("records"):
                            sensor_day_id = int(selected["sensor_day_id"])
                            primitive = str(selected["primitive"])
                            draw = int(selected["draw"])
                            context = contexts[(participant, sensor_day_id)]
                            reference = stress_downstream.reference_feature_frame(context, arm)
                            cache_key = (fit_seq, sensor_day_id)
                            if cache_key not in probability_cache:
                                probability_cache[cache_key] = float(
                                    outcome.predict_frozen_fold_model(
                                        model,
                                        reference,
                                        expected_content_digest=model.content_digest,
                                    )[0]
                                )
                            reference_probability = probability_cache[cache_key]
                            score_key = (
                                participant,
                                pd.Timestamp(selected["lifelog_date"]),
                                pd.Timestamp(selected["sleep_date"]),
                            )
                            score_values = labels.get(score_key)
                            label = (
                                None
                                if score_values is None
                                else int(score_values[target_position])
                            )
                            base = {
                                "participant_position": participant_position,
                                "participant": participant,
                                "fold": fold,
                                "held_out_subject": participant,
                                "family_position": family_position,
                                "family": family,
                                "primitive_position": int(selected["primitive_position"]),
                                "primitive": primitive,
                                "sensor_day_id": sensor_day_id,
                                "draw": draw,
                                "arm_position": arm_position,
                                "arm": arm,
                                "model_position": model_position,
                                "model_name": model_name,
                                "target_position": target_position,
                                "target": target,
                                "fit_seq": fit_seq,
                                "fit_ledger_record_digest": str(
                                    ledger["fit_ledger_record_digest"]
                                ),
                                "preprocessor_digest": str(ledger["preprocessor_digest"]),
                                "model_digest": str(ledger["model_digest"]),
                                "fit_content_digest": str(ledger["fit_content_digest"]),
                            }
                            for condition_position, condition in enumerate(
                                stress_downstream_runner._CONDITIONS
                            ):
                                patch = patches[
                                    (participant, sensor_day_id, primitive, draw, condition, arm)
                                ]
                                patched = stress_downstream.apply_downstream_feature_patch(
                                    reference,
                                    patch,
                                    expected_context_digest=context.content_digest,
                                    expected_perturbation_digest=patch.perturbation_digest,
                                )
                                perturbed = (
                                    None
                                    if patched is None
                                    else float(
                                        outcome.predict_frozen_fold_model(
                                            model,
                                            patched,
                                            expected_content_digest=model.content_digest,
                                        )[0]
                                    )
                                )
                                observed = stress_downstream_runner._observed_prediction_row(
                                    base,
                                    patch,
                                    row_seq=row_seq,
                                    condition_position=condition_position,
                                    reference_probability=reference_probability,
                                    perturbed_probability=perturbed,
                                    label=label,
                                )
                                participant_rows.append(observed)
                                row_seq += 1
        yield pa.RecordBatch.from_pylist(
            participant_rows,
            schema=stress_downstream_statistics.OBSERVED_PREDICTION_SCHEMA,
        )


def _observed_pin(
    path: Path,
) -> stress_downstream_statistics.ObservedPredictionInputPin:
    hasher = stress_downstream_runner._LogicalHasher(
        "task-12-downstream-observed-prediction-logical-v1",
        stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS,
    )
    with path.open("rb") as stream:
        for batch in ipc.open_stream(stream):
            for row in batch.to_pylist():
                hasher.update(row)
    byte_count, physical_sha256 = stress_downstream_runner._file_pin(path)
    values = {
        "schema_name": "task-12-downstream-observed-prediction-input-v1",
        "row_count": hasher.row_count,
        "columns": stress_downstream_statistics.OBSERVED_PREDICTION_COLUMNS,
        "byte_count": byte_count,
        "physical_sha256": physical_sha256,
        "logical_digest": hasher.hexdigest(),
    }
    unsealed = stress_downstream_statistics.ObservedPredictionInputPin(
        **values, content_digest="0" * 64
    )
    return stress_downstream_statistics.ObservedPredictionInputPin(
        **values,
        content_digest=stress_downstream_runner._sha(
            [
                "task-12-downstream-observed-prediction-input-pin-v1",
                unsealed.schema_name,
                unsealed.row_count,
                [list(value) for value in unsealed.columns],
                unsealed.byte_count,
                unsealed.physical_sha256,
                unsealed.logical_digest,
            ]
        ),
    )


def recover_downstream_snapshot(
    snapshot: stress_downstream_runner.DownstreamAdapterSnapshot,
    scoring_labels: pd.DataFrame | None,
    source_partial: Path,
    artifact_pins: tuple[RecoveryArtifactPin, ...],
    output_dir: Path,
    *,
    expected_snapshot_digest: str,
    expected_fit_authority_digest: str,
    expected_prefix_participants: tuple[str, ...],
    expected_prefix_rows: int,
    expected_total_rows: int,
) -> stress_downstream_runner.DownstreamRunArtifacts:
    """Publish a complete run by copying a pinned prefix and predicting its suffix."""
    stress_downstream_runner.validate_downstream_adapter_snapshot(
        snapshot,
        expected_content_digest=expected_snapshot_digest,
        expected_source_patch_projection_digest=(
            snapshot.source_patch_projection.logical_digest
        ),
    )
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise TypeError("recovery output_dir must be an absolute pathlib Path")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    all_participants = tuple(value for _fold, value in snapshot.fold_roster)
    if all_participants[: len(expected_prefix_participants)] != expected_prefix_participants:
        raise ValueError("saved participants are not the snapshot's canonical prefix")
    derived_fit_digest = derive_fit_authority_digest(source_partial / "fit")
    if derived_fit_digest != expected_fit_authority_digest:
        raise ValueError("recovery fit authority pin mismatch")
    loaded = stress_downstream_runner.load_downstream_fit_state_pack(
        source_partial / "fit",
        expected_content_digest=expected_fit_authority_digest,
    )
    if len(loaded.models) != 160:
        raise ValueError("recovery model topology is not exactly 160")
    regenerated_tables = dict(stress_downstream_runner._snapshot_arrow_tables(snapshot))
    for name, table in regenerated_tables.items():
        with (source_partial / name).open("rb") as stream:
            saved = ipc.open_stream(stream).read_all()
        if not saved.equals(table):
            raise ValueError(f"saved adapter artifact disagrees with snapshot: {name}")
    partial = output_dir.with_name(
        f"{output_dir.name}.{uuid.uuid4().hex}.recovery.partial"
    )
    partial.mkdir()
    try:
        for pin in artifact_pins:
            if pin.relative_path == "observed_predictions.arrow":
                continue
            source = source_partial / Path(pin.relative_path)
            destination = partial / Path(pin.relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        observed_path = partial / "observed_predictions.arrow"
        observed_seal = recover_observed_stream(
            source_partial,
            observed_path,
            artifact_pins=artifact_pins,
            expected_schema=stress_downstream_statistics.OBSERVED_PREDICTION_SCHEMA,
            expected_prefix_participants=expected_prefix_participants,
            expected_all_participants=all_participants,
            expected_prefix_rows=expected_prefix_rows,
            expected_total_rows=expected_total_rows,
            suffix_batch_factory=lambda first: _suffix_batches(
                snapshot,
                loaded,
                scoring_labels,
                completed_count=len(expected_prefix_participants),
                first_row_seq=first,
            ),
            row_validator=_validate_observed_row,
        )
        if observed_seal.row_count != expected_total_rows:
            raise ValueError("recovery observed row count mismatch")
        pin = _observed_pin(observed_path)
        authority_values = {
            "fold_roster": snapshot.fold_roster,
            "families": stress_downstream_runner._FAMILIES,
            "primitive_family_pairs": snapshot.primitive_family_pairs,
            "arms": outcome.DOWNSTREAM_ARM_NAMES,
            "models": outcome.MODEL_NAMES,
            "targets": outcome.TARGET_COLUMNS,
            "conditions": stress_downstream_runner._CONDITIONS,
            "expected_primary_key_count": snapshot.expected_primary_key_count,
            "expected_primary_key_digest": snapshot.expected_primary_key_digest,
            "max_primary_keys_per_group": snapshot.max_primary_keys_per_group,
            "observed_prediction_pin": pin,
            "source_patch_projection_row_count": snapshot.source_patch_projection.row_count,
            "source_patch_projection_digest": snapshot.source_patch_projection.logical_digest,
            "fit_ledger_row_count": loaded.seal.fit_ledger_row_count,
            "fit_state_index_row_count": loaded.seal.fit_state_index_row_count,
            "fit_state_record_count": loaded.seal.fit_state_record_count,
            "fit_ledger_projection_digest": loaded.seal.fit_ledger_projection_digest,
            "fit_state_index_logical_digest": loaded.seal.fit_state_index_logical_digest,
            "fit_state_pack_sha256": loaded.seal.fit_state_pack_sha256,
            "fit_authority_content_digest": loaded.seal.content_digest,
        }
        unsealed_authority = stress_downstream_statistics.DownstreamStatisticsInputAuthority(
            **authority_values, content_digest="0" * 64
        )
        authority = stress_downstream_statistics.DownstreamStatisticsInputAuthority(
            **authority_values,
            content_digest=stress_downstream_runner._statistics_authority_digest(
                unsealed_authority
            ),
        )

        def batches() -> Iterable[pa.RecordBatch]:
            with observed_path.open("rb") as stream:
                yield from ipc.open_stream(stream)

        handoff_dir = partial / "statistics-handoff"
        handoff = stress_downstream_statistics.analyze_downstream_statistics(
            batches(),
            authority,
            handoff_dir,
            expected_authority_digest=authority.content_digest,
        )
        statistics_manifest = stress_downstream_statistics.persist_downstream_statistics(
            handoff,
            partial / "statistics",
            expected_handoff_digest=handoff.content_digest,
        )
        stress_downstream_statistics.verify_persisted_downstream_statistics(
            partial / "statistics",
            statistics_manifest,
            expected_manifest_digest=statistics_manifest.content_digest,
        )
        shutil.rmtree(handoff_dir)
        artifacts: list[dict[str, object]] = []
        for path in sorted(
            (value for value in partial.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(partial).as_posix(),
        ):
            byte_count, sha256 = stress_downstream_runner._file_pin(path)
            relative = path.relative_to(partial).as_posix()
            artifacts.append(
                {
                    "relative_path": relative,
                    "byte_count": byte_count,
                    "physical_sha256": sha256,
                    "content_digest": stress_downstream_runner._sha(
                        [
                            "task-12-downstream-run-artifact-seal-v1",
                            relative,
                            byte_count,
                            sha256,
                        ]
                    ),
                }
            )
        manifest_core = {
            "schema_name": "task-12-downstream-run-manifest-v1",
            "adapter_digest": snapshot.content_digest,
            "official_g2_authority_digest": snapshot.official_g2_authority_digest,
            "official_cell_evidence_projection_digest": (
                snapshot.official_cell_evidence_projection_digest
            ),
            "source_patch_projection_digest": snapshot.source_patch_projection.logical_digest,
            "outcome_build_digest": loaded.seal.outcome_build_digest,
            "selected_target_count": snapshot.eligible_target_count,
            "reference_key_count": snapshot.reference_key_count,
            "expected_primary_key_count": snapshot.expected_primary_key_count,
            "observed_row_count": pin.row_count,
            "fit_authority_digest": loaded.seal.content_digest,
            "observed_prediction_pin_digest": pin.content_digest,
            "statistics_manifest_digest": statistics_manifest.content_digest,
            "statistics_result_digest": statistics_manifest.result_digest,
            "artifacts": artifacts,
        }
        run_digest = stress_downstream_runner._sha(
            ["task-12-downstream-run-manifest-v1", manifest_core]
        )
        manifest_bytes = stress_downstream_runner._canonical(
            {**manifest_core, "content_digest": run_digest}
        ) + b"\n"
        (partial / "manifest.json").write_bytes(manifest_bytes)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if output_dir.exists():
            raise FileExistsError(output_dir)
        partial.rename(output_dir)
        return stress_downstream_runner.DownstreamRunArtifacts(
            output_dir=output_dir,
            adapter_digest=snapshot.content_digest,
            fit_authority_digest=loaded.seal.content_digest,
            observed_prediction_pin=pin,
            statistics_manifest=statistics_manifest,
            manifest_sha256=manifest_sha,
            content_digest=run_digest,
        )
    except BaseException:
        raise
