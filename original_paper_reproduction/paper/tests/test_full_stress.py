"""Independent stress boundary guards plus one bounded archived-fixture replay.

The literal G2 fixture is id01/day 1/screen_load_24h/contiguous20/draw 0.
It tests the adapter only; it is never an input to run_stress.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
KEY = ("id01", 1, "screen_load_24h", "contiguous_20pct", 0)
SEED = "f21a35fb9d2619831ca4404cf70d91f5a17ae4ff0f7c9f11c48d0ec23a10f6d3"
AUDIT_KEY = "6b3dd0624c1031fb235f1cf1a52b21daf90cb151eb35863f2afe724b61f41383"
RECORD_DIGEST = "223ad8555ac9ccd0a08fa1ebd805c73729091bc45a667dbcfe895ac8d893eea0"


def fixture_frames():
    cell = dict(subject_id="id01", sensor_day_id=1,
                lifelog_date=pd.Timestamp("2024-06-27"), elapsed_day_index=2,
                primitive="screen_load_24h", scenario="contiguous_20pct", draw=0,
                root_seed="42", seed=SEED, donor_sensor_day_id=float("nan"),
                deletion_audit_key=AUDIT_KEY, original_value=0.3657342657342657,
                masked_value=0.2968476357267951, original_status="observed",
                masked_status="observed", retained=True,
                absolute_error=0.06888663000747058,
                standardized_absolute_error=3.353089820799653,
                standardizer=0.020544224488159502, standardizer_source="selected_day_mad",
                selected_day_mad=0.020544224488159502,
                frozen_population_scale=0.11691247340425537,
                scale_floor=0.011691247340425537, floor_applied=False,
                original_value_only_confidence=1.0,
                original_coverage_only_confidence=0.9930555555555556,
                original_quality_confidence=0.9861593364197532,
                masked_value_only_confidence=1.0,
                masked_coverage_only_confidence=0.7930555555555555,
                masked_quality_confidence=0.6344444444444445, success=False,
                usage_record_deletion_stability=False, reason="retained", status="observed")
    audit = dict(deletion_audit_key=AUDIT_KEY, subject_id="id01", sensor_day_id=1,
                 primitive="screen_load_24h", scenario="contiguous_20pct", draw=0,
                 seed=SEED, donor_sensor_day_id=float("nan"), original_count=1430,
                 deleted_count=288, retained_count=1142, record_deletion_digest=RECORD_DIGEST,
                 audit_representation="normalized_source_digest_plus_mask_geometry", status="ok")
    mask = dict(subject_id="id01", sensor_day_id=1, primitive="screen_load_24h",
                scenario="contiguous_20pct", draw=0, root_seed="42", seed=SEED,
                window_start=pd.Timestamp("2024-06-27"), window_end=pd.Timestamp("2024-06-28"),
                mask_interval_id=0, mask_start=pd.Timestamp("2024-06-27 07:12"),
                mask_end=pd.Timestamp("2024-06-27 12:00"), mask_duration_minutes=288.0,
                mask_semantics="half_open", empirical_policy="not_applicable", status="generated",
                donor_sensor_day_id=float("nan"), deletion_audit_key=AUDIT_KEY)
    return pd.DataFrame([cell]), pd.DataFrame([audit]), pd.DataFrame([mask])


class FullStressTests(unittest.TestCase):
    def adapter(self):
        self.assertIsNotNone(importlib.util.find_spec("full_stress"),
                             "full_stress independent adapter has not been implemented")
        return importlib.import_module("full_stress")

    def pairs(self, frames=None):
        return self.adapter().build_validated_pairs(
            *(fixture_frames() if frames is None else frames),
            artifact_hashes={name: "a" * 64 for name in
                             ("cell_replays", "deletion_audit", "mask_intervals")},
            expected_keys=(KEY,))

    def test_g2_row_adapter_preserves_exact_scalar_and_interval_values(self):
        evidence, mask = self.pairs()[KEY]
        self.assertEqual(evidence.standardized_absolute_error, 3.353089820799653)
        self.assertEqual(evidence.standardizer, 0.020544224488159502)
        self.assertEqual(mask.deletion_audit.deleted_count, 288)
        self.assertEqual(mask.intervals[0].mask_start, pd.Timestamp("2024-06-27 07:12"))
        self.assertEqual(mask.intervals[0].mask_end, pd.Timestamp("2024-06-27 12:00"))

    def test_duplicate_or_missing_g2_rows_are_rejected(self):
        for index in (0, 1, 2):
            for duplicate in (True, False):
                frames = list(fixture_frames())
                frames[index] = (pd.concat([frames[index], frames[index]], ignore_index=True)
                                 if duplicate else frames[index].iloc[:0])
                with self.subTest(table=index, duplicate=duplicate), self.assertRaises(ValueError):
                    self.pairs(frames)

    def test_crosswired_seed_count_interval_and_scale_are_rejected(self):
        attacks = ((0, "seed", "0" * 64), (1, "deleted_count", 287),
                   (2, "mask_end", pd.Timestamp("2024-06-29")),
                   (0, "standardizer", 0.2), (1, "record_deletion_digest", "0" * 64))
        for table, column, value in attacks:
            frames = list(fixture_frames())
            frames[table].loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.pairs(frames)

    def test_missing_values_keep_abstention_instead_of_fabricating_errors(self):
        cell, audit, mask = fixture_frames()
        for name in ("masked_value", "absolute_error", "standardized_absolute_error"):
            cell.loc[0, name] = float("nan")
        cell.loc[0, "status"] = "abstained"
        cell.loc[0, "reason"] = "masked_insufficient"
        cell.loc[0, "masked_status"] = "insufficient"
        cell.loc[0, "retained"] = False
        evidence, _ = self.pairs((cell, audit, mask))[KEY]
        self.assertTrue(pd.isna(evidence.standardized_absolute_error))
        self.assertEqual(evidence.status, "abstained")

    def make_upstream(self, root):
        path = root / "cell_replays.parquet"
        fixture_frames()[0].to_parquet(path, index=False)
        manifest = {"schema_name": "independent_etri_reproduction", "stage": "g2",
                    "status": "complete", "fresh_execution": True,
                    "outputs": {"cell_replays": {"path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "rows": 1, "columns": list(fixture_frames()[0].columns)}}}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_upstream_requires_fresh_completed_stage_and_current_hash(self):
        adapter = self.adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_upstream(root)
            accepted = adapter.read_upstream_manifest(root, "g2", ("cell_replays",))
            self.assertEqual(accepted["outputs"]["cell_replays"]["rows"], 1)
            for field, bad in (("schema_name", "structured-missingness-etri-output-v1"),
                               ("stage", "prepare"), ("status", "running"),
                               ("fresh_execution", False)):
                changed = {**manifest, field: bad}
                (root / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    adapter.read_upstream_manifest(root, "g2", ("cell_replays",))
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "cell_replays.parquet").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                adapter.read_upstream_manifest(root, "g2", ("cell_replays",))

    def test_upstream_path_escape_is_rejected(self):
        adapter = self.adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_upstream(root)
            manifest["outputs"]["cell_replays"]["path"] = "../escape.parquet"
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                adapter.read_upstream_manifest(root, "g2", ("cell_replays",))

    def test_existing_output_is_not_overwritten_even_with_missing_inputs(self):
        adapter = self.adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                adapter.run_stress(root / "prepare", root / "canonical", root / "g2", output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_failed_upstream_validation_records_failure_without_completing(self):
        adapter = self.adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "failed"
            with self.assertRaises(FileNotFoundError):
                adapter.run_stress(root / "missing-prepare", root / "missing-canonical",
                                   root / "missing-g2", output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["outputs"], {})
            self.assertNotIn("production_eligible", manifest)
            self.assertNotIn("release_seal_sha256", manifest)

    def test_selected_grid_rejects_missing_duplicate_and_wrong_participant(self):
        adapter = self.adapter()
        selected = pd.DataFrame([(f"id{i:02}", (i - 1) * 10 + day, primitive)
                                 for i in range(1, 11)
                                 for primitive in adapter.PRIMITIVE_ORDER
                                 for day in range(1, 11)],
                                columns=["subject_id", "sensor_day_id", "primitive"])
        self.assertEqual(len(adapter.validate_selected_topology(selected)), 800)
        wrong_roster = selected.copy()
        wrong_roster.loc[wrong_roster.subject_id.eq("id10"), "subject_id"] = "id11"
        for attacked in (selected.iloc[:-1], pd.concat([selected, selected.iloc[:1]]), wrong_roster):
            with self.assertRaises(ValueError):
                adapter.validate_selected_topology(attacked)

    def test_reordered_original_schema_is_rejected(self):
        frames = list(fixture_frames())
        frames[0] = frames[0][list(reversed(frames[0].columns))]
        with self.assertRaises(ValueError):
            self.pairs(frames)

    def test_small_replay_matches_original_archived_target_and_two_fresh_masks(self):
        adapter = self.adapter()
        roots_path = Path(os.environ.get("ICTC_TEST_ROOTS", str(PAPER / "input_manifest/local_roots.json")))
        if not roots_path.exists():
            self.skipTest("optional local archived scientific fixture is unavailable")
        roots = json.loads(roots_path.read_text(encoding="utf-8"))
        source = Path(roots["etri_source"])
        canonical = PAPER.parent / "validation/preprocessing/raw_run/project/data/canonical"
        if not canonical.is_dir():
            canonical = source / "ETRI_Human_AI_v2/data/canonical"
        primitive_file = Path(roots["etri_sdd"]) / "task-4c2-real-primitives.parquet"
        if sys.platform == "win32":
            primitive_file = Path("\\\\?\\" + str(primitive_file))
        if not canonical.is_dir() or not primitive_file.is_file():
            self.skipTest("optional local canonical/primitive fixture is unavailable")
        primitives = pd.read_parquet(primitive_file)
        row = primitives.loc[primitives["sensor_day_id"].eq(1)].iloc[0]
        prepared = adapter.prepare_selected_cell("screen_load_24h", row,
                                                canonical_root=canonical, sensor_indexes={})
        tables = adapter.replay_cell(prepared, [self.pairs()[KEY]])
        target = tables["target_ledger"].iloc[0]
        self.assertEqual(target.target_deleted_count, 288)
        self.assertEqual(target.content_digest,
                         "b4aa787fa6adb4ab7825f66928b3d9c2ae2583c1f7455bf46d8b30700478104f")
        self.assertEqual(len(tables["cell_replays"]), 2)
        self.assertTrue(tables["deletion_audit"].exact_match.all())
        from semantic_indicators import stress_runner
        from semantic_indicators.stress_contracts import Contiguous20Crosslink, derive_matched_deletion_target
        evidence, mask = self.pairs()[KEY]
        crosslink = Contiguous20Crosslink(
            subject_id="id01", sensor_day_id=1, primitive="screen_load_24h", draw=0,
            deletion_audit_key=AUDIT_KEY, official_deleted_count=288, official_retained_count=1142,
            official_record_deletion_digest=RECORD_DIGEST,
            intervals=((pd.Timestamp("2024-06-27 07:12"), pd.Timestamp("2024-06-27 12:00")),),
            reference_value=0.3657342657342657, reference_status="observed",
            standardizer=0.020544224488159502, standardizer_source="selected_day_mad")
        direct = stress_runner.run_stress_replays(prepared, [derive_matched_deletion_target(prepared, crosslink)])
        for name in ("target_ledger", "deletion_audit", "cell_replays"):
            pd.testing.assert_frame_equal(tables[name], getattr(direct, name), check_exact=True)


if __name__ == "__main__":
    unittest.main()
