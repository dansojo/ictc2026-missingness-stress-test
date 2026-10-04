"""Tests of fresh-evidence extraction; archived fixtures are comparison only."""
from __future__ import annotations

import copy
import os
import importlib
import importlib.util
import io
import json
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.environ.get('ICTC_PRIVATE_REFERENCE', str(PAPER / 'reference')))
sys.path.insert(0, str(PAPER))


class FullEvidenceTests(unittest.TestCase):
    def adapter(self):
        self.assertIsNotNone(importlib.util.find_spec("full_evidence"),
                             "fresh evidence adapter has not been implemented")
        return importlib.import_module("full_evidence")

    def lineage(self):
        hashes = {stage: digit * 64 for stage, digit in
                  (("prepare", "1"), ("g2", "2"), ("stress", "3"), ("downstream", "4"))}
        manifests = {stage: {"schema_name": "independent_etri_reproduction", "stage": stage,
                             "status": "complete", "fresh_execution": True, "inputs": {}}
                     for stage in hashes}
        manifests["g2"]["inputs"]["prepare_manifest_sha256"] = hashes["prepare"]
        manifests["stress"]["inputs"].update(prepare_manifest_sha256=hashes["prepare"],
                                               g2_manifest_sha256=hashes["g2"])
        manifests["downstream"].update(new_model_fits=160, reused_historical_model_fits=0,
                                         prediction_rows=943392, expected_primary_key_count=314464)
        manifests["downstream"]["inputs"].update(prepare_manifest_sha256=hashes["prepare"],
            g2_manifest_sha256=hashes["g2"], stress_manifest_sha256=hashes["stress"])
        return manifests, hashes

    def test_evidence_rejects_crosswired_chain_and_reused_models(self):
        adapter = self.adapter()
        manifests, hashes = self.lineage()
        adapter.validate_lineage(manifests, hashes)
        for stage, field, bad in (("downstream", "new_model_fits", 0),
                                  ("downstream", "reused_historical_model_fits", 160),
                                  ("stress", "fresh_execution", False)):
            changed = copy.deepcopy(manifests)
            changed[stage][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.validate_lineage(changed, hashes)
        changed = copy.deepcopy(manifests)
        changed["downstream"]["inputs"]["stress_manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            adapter.validate_lineage(changed, hashes)

    def fixture_tables(self):
        adapter = self.adapter()
        roots_path = Path(os.environ.get("ICTC_TEST_ROOTS", str(PAPER / "input_manifest/local_roots.json")))
        if not roots_path.exists():
            self.skipTest("optional archived schema fixture is unavailable")
        roots = json.loads(roots_path.read_text(encoding="utf-8"))
        root = Path(roots["etri_sdd"])
        if sys.platform == "win32":
            root = Path("\\\\?\\" + str(root))
        diagnostic = root / "task-12-etri-production_20260810_230202"
        downstream = root / "task-12-downstream-etri-production-v1/statistics"
        if not (diagnostic / "family_inference.parquet").is_file():
            self.skipTest("optional archived comparison tables are unavailable")
        return ({name: pd.read_parquet(diagnostic / f"{name}.parquet") for name in adapter.STRESS_TABLES},
                {name: pd.read_parquet(downstream / f"{name}.parquet") for name in adapter.DOWNSTREAM_TABLES})

    def test_eight_csv_formulas_match_archived_comparison_exactly(self):
        adapter = self.adapter()
        tables = adapter.build_reporting_tables(*self.fixture_tables())
        self.assertEqual(set(tables), set(adapter.CSV_NAMES))
        for name, frame in tables.items():
            rendered = frame.to_csv(index=False, lineterminator="\n", float_format="%.9g")
            actual = pd.read_csv(io.StringIO(rendered))
            expected = pd.read_csv(REFERENCE / "etri" / name)
            with self.subTest(name=name):
                pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_result_prose_and_design_derive_changed_values(self):
        adapter = self.adapter()
        stress, downstream = self.fixture_tables()
        stress["family_inference"].loc[stress["family_inference"].feature_family.eq("event_count"),
                                         "participant_median"] = 1.234567
        downstream["decision"].loc[0, "t_observed"] = 0.123456
        downstream["decision"].loc[0, "sign_p_value"] = 0.25
        downstream["decision"].loc[0, "direction_positive_count"] = 7
        tables = adapter.build_reporting_tables(stress, downstream)
        manifests, hashes = self.lineage()
        design = adapter.build_study_design(stress, downstream, manifests["downstream"])
        prose = adapter.render_manuscript_evidence(tables, design, downstream["decision"])
        self.assertIn("1.234567", prose)
        self.assertIn("0.123456", prose)
        self.assertIn("0.25", prose)
        self.assertIn("7/10", prose)
        self.assertEqual(design["downstream"]["new_model_fits"], 160)
        self.assertNotIn("recovery_new_model_fits", design["downstream"])
        self.assertNotIn("External transfer; StudentLife and ExtraSensory remain pending", prose)

    def test_fixed_display_counts_are_checked_against_evidence(self):
        adapter = self.adapter()
        if not (REFERENCE / 'etri/downstream_condition_summary.csv').exists():
            self.skipTest('Private archived reference not configured')
        frame = pd.read_csv(REFERENCE / "etri/downstream_condition_summary.csv")
        adapter.validate_display_annotations(frame)
        frame.loc[frame.condition.eq("event_boundary_20pct"), "abstained_count"] = 0
        with self.assertRaises(ValueError):
            adapter.validate_display_annotations(frame)

    def test_existing_evidence_or_display_output_is_preserved(self):
        adapter = self.adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                adapter.run_evidence(root/"prepare", root/"g2", root/"stress", root/"downstream", output)
            with self.assertRaises(FileExistsError):
                adapter.run_displays(root/"evidence", root/"external", output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_display_adapter_renders_pinned_comparison_fixture(self):
        """This is a temporary reference-fixture test, never a fresh experiment."""
        if not (REFERENCE / 'etri/feature_contracts.csv').exists():
            self.skipTest('Private archived reference not configured')
        adapter = self.adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "comparison-fixture"
            evidence.mkdir()
            pins = {}
            for name in adapter.CSV_NAMES:
                path = evidence / name
                shutil.copyfile(REFERENCE / "etri" / name, path)
                pins[name] = {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            document = {"schema_name": "independent_etri_reproduction", "stage": "evidence",
                        "status": "complete", "fresh_execution": True,
                        "upstream_new_model_fits": 160, "outputs": pins,
                        "test_fixture_only": True}
            (evidence / "manifest.json").write_text(json.dumps(document), encoding="utf-8")
            manifest = adapter.run_displays(evidence, REFERENCE / "external", root / "display-test")
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["outputs"]), 4)
            self.assertTrue((root / "display-test/.matplotlib").is_dir(),
                            "renderer cache must remain inside the writable output")
            from PIL import Image
            with Image.open(root / "display-test/figures/fig2_results.png") as actual:
                with Image.open(REFERENCE / "displays/fig2_results.png") as expected:
                    self.assertEqual(actual.size, expected.size)
                    self.assertEqual(actual.convert("RGBA").tobytes(), expected.convert("RGBA").tobytes())

    def test_failed_evidence_input_is_recorded_without_reports(self):
        adapter = self.adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                adapter.run_evidence(root/"missing-prepare", root/"g2", root/"stress", root/"downstream", root/"output")
            manifest = json.loads((root/"output/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["outputs"], {})


if __name__ == "__main__":
    unittest.main()
