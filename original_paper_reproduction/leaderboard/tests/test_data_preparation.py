"""Standard-library safety tests for the raw-feature driver.

These catch raw/output overlap, partial overwrite, absent raw streams, and
changed vendored source before any scientific preprocessing can write files.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
from types import SimpleNamespace
import unittest

DRIVER = Path(__file__).resolve().parents[1] / "data_preparation.py"
RAW_NAMES = [
    "ch2026_metrics_train.csv", "ch2026_submission_sample.csv",
    *[f"ch2025_data_items/ch2025_{name}.parquet" for name in (
        "mScreenStatus", "mUsageStats", "wPedo", "mActivity", "wHr", "mLight",
        "wLight", "mACStatus", "mGps", "mWifi", "mBle", "mAmbience")],
]


class PreparationGuards(unittest.TestCase):
    def setUp(self):
        self.assertTrue(DRIVER.is_file(), "portable data_preparation.py is not implemented")
        spec = importlib.util.spec_from_file_location("tested_preparation_driver", DRIVER)
        self.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.driver)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        for name in RAW_NAMES:
            path = self.raw / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic fixture; never read as a sensor")

    def test_missing_stream_rejected_before_creating_output(self):
        (self.raw / RAW_NAMES[-1]).unlink()
        out = self.root / "new-output"
        with self.assertRaises(FileNotFoundError):
            self.driver.validate_inputs(self.raw, out)
        self.assertFalse(out.exists())

    def test_output_inside_raw_is_rejected(self):
        with self.assertRaises(ValueError):
            self.driver.validate_inputs(self.raw, self.raw / "outputs")
        self.assertFalse((self.raw / "outputs").exists())

    def test_raw_inside_output_is_rejected(self):
        with self.assertRaises(ValueError):
            self.driver.validate_inputs(self.raw, self.root)

    def test_existing_output_is_never_reused(self):
        out = self.root / "existing-output"
        out.mkdir()
        marker = out / "keep.txt"
        marker.write_text("original result", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.driver.validate_inputs(self.raw, out)
        self.assertEqual(marker.read_text(), "original result")

    def test_valid_input_validation_is_read_only(self):
        out = self.root / "new-output"
        paths = self.driver.validate_inputs(self.raw, out)
        self.assertEqual(len(paths), 14)
        self.assertEqual(set(paths), {self.raw / name for name in RAW_NAMES})
        self.assertFalse(out.exists())

    def test_vendored_file_digest_change_is_rejected(self):
        vendor = self.root / "vendor"
        vendor.mkdir()
        script = vendor / "stage.py"
        script.write_bytes(b"original stage\n")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"sources": [{
            "vendor_relative_path": "stage.py", "bytes": 15,
            "sha256": hashlib.sha256(b"original stage\n").hexdigest(),
        }]}), encoding="utf-8")
        self.driver.verify_vendor(vendor, manifest)
        script.write_bytes(b"modified stage\n")
        with self.assertRaises(ValueError):
            self.driver.verify_vendor(vendor, manifest)

    def test_vendor_manifest_cannot_escape_vendor_root(self):
        vendor = self.root / "vendor"
        vendor.mkdir()
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"sources": [{
            "vendor_relative_path": "../outside.py", "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }]}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.driver.verify_vendor(vendor, manifest)

    def test_rebinding_routes_outputs_and_keeps_external_sensor_input(self):
        old = self.root / "vendor-project"
        stage = SimpleNamespace(PROJECT_ROOT=old,
            RAW_ROOT=old / "data/raw/data_vvs",
            SENSOR_ROOT=old / "data/raw/data_vvs/ch2025_data_items",
            TRAIN_PATH=old / "data/raw/data_vvs/ch2026_metrics_train.csv",
            OUT_PATH=old / "data/processed/features.csv", THRESHOLD=0.995)
        project = self.root / "run/project"
        self.driver.rebind_module_paths(stage, project, self.raw)
        self.assertEqual(stage.OUT_PATH, project / "data/processed/features.csv")
        self.assertEqual(stage.TRAIN_PATH, project / "data/raw/data_vvs/ch2026_metrics_train.csv")
        self.assertEqual(stage.SENSOR_ROOT, self.raw / "ch2025_data_items")
        self.assertEqual(stage.THRESHOLD, 0.995)

    def test_training_manifest_contains_only_four_relative_input_pins(self):
        data = self.root / "prepared/project/data"
        names = ["processed/train_integrated_sensor_v2_cleaned.csv",
                 "processed/test_integrated_sensor_v2_cleaned.csv",
                 "processed/feature_columns_sensor_v2_cleaned.json",
                 "raw/data_vvs/ch2026_submission_sample.csv"]
        for index, name in enumerate(names):
            path = data / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture-{index}\n".encode())
        provenance = self.root / "prepared/preprocessing_run.json"
        provenance.write_text('{"status":"complete"}', encoding="utf-8")
        destination = provenance.parent / "training_input_manifest.json"
        document = self.driver.write_training_input_manifest(data, destination, provenance_path=provenance)
        self.assertEqual([item["path"] for item in document["files"]], names)
        self.assertNotIn("data_root", document)
        for index, record in enumerate(document["files"]):
            expected = f"fixture-{index}\n".encode()
            self.assertEqual(record["bytes"], len(expected))
            self.assertEqual(record["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(document["provenance"]["path"], "preprocessing_run.json")
        self.assertEqual(document["provenance"]["sha256"], hashlib.sha256(provenance.read_bytes()).hexdigest())
        self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), document)
        original = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            self.driver.write_training_input_manifest(data, destination)
        self.assertEqual(destination.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
