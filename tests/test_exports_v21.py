from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.io import loadmat

from epoch_builder import WINDOW_COLUMNS
from mat_exporter import export_session_to_mat
from preprocess_eeg import preprocess_session
from session_reports import build_session_qc_report
from session_recorder import PROTOCOL_CONFIG


ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    def test_windows_schema_contains_analysis_covariates_and_dataset_masks(self):
        required = {
            "probe_id", "probe_attention", "probe_confidence", "seconds_to_probe",
            "mental_effort", "course_attention_rating", "video_interest", "video_difficulty",
            "start_number", "reported_final_number", "subtraction_compliance",
            "self_caught_nearby", "quality_label", "reject_reason", "dataset_membership",
            "training_eligible", "start_sample", "end_sample", "start_timestamp", "end_timestamp",
        }
        self.assertTrue(required <= set(WINDOW_COLUMNS))

    def test_offline_notch_writes_a_copy_and_preserves_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            sample_rate = 250
            count = 1000
            t = np.arange(count) / sample_rate
            signal = np.sin(2 * np.pi * 10 * t) + 4 * np.sin(2 * np.pi * 50 * t)
            frame = pd.DataFrame({
                "device_sample_number": np.arange(count), "stream_segment": 0,
                "channel_0_uv": signal, "channel_1_uv": signal,
            })
            frame.to_csv(session / "eeg.csv", index=False)
            (session / "metadata.json").write_text(json.dumps({
                "session": {"sample_rate_hz": sample_rate}
            }), encoding="utf-8")
            original = (session / "eeg.csv").read_bytes()
            output, report = preprocess_session(session)
            self.assertEqual((session / "eeg.csv").read_bytes(), original)
            self.assertTrue(output.exists())
            details = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(details["raw_source_overwritten"])
            self.assertEqual(details["filter"]["center_hz"], 50.0)
            filtered = pd.read_csv(output)["channel_0_uv_notch50"].to_numpy()
            self.assertLess(np.nanstd(filtered), np.std(signal))

    def test_integrity_report_detects_missing_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            pd.DataFrame(columns=["event_type", "probe_id", "block_id"]).to_csv(session / "events.csv", index=False)
            pd.DataFrame(columns=["device_sample_number"]).to_csv(session / "eeg.csv", index=False)
            pd.DataFrame().to_csv(session / "qc.csv", index=False)
            (session / "metadata.json").write_text(json.dumps({
                "run_id": "run-x", "software_version": "2.1.0", "protocol_version": "2.1",
                "protocol_config_version": "test", "session": {
                    "sample_rate_hz": 250, "counterbalance_group": "G01", "study_phase": "pilot",
                    "b_start_numbers": {}, "baseline_override": False,
                }, "qc": {"dropouts": []}, "preprocessing": PROTOCOL_CONFIG["preprocessing"],
            }), encoding="utf-8")
            report = json.loads(build_session_qc_report(session).read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(any("probes" in item for item in report["failures"]))
            self.assertTrue(any("eeg.csv" in item for item in report["failures"]))

    def test_mat_contains_normalized_tables_and_integrity_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            pd.DataFrame([{
                "received_timestamp": 1, "received_order": 0, "device_sample_number": 0,
                "sample_index": 0, "sample_time_sec": 0, "packet_gap_before": 0,
                "block_id": "", "group_id": "", "condition": "", "condition_type": "",
                "condition_code": -1, "weak_label": -1, "phase": "idle", "base_valid_for_training": 0,
                "channel_0_raw": 1, "channel_1_raw": -1, "channel_0_uv": 1.0, "channel_1_uv": -1.0,
                "quality_flag": "ok", "is_formal_experiment": 0, "stream_segment": 0,
                "sample_time_status": "packet_counter",
            }]).to_csv(session / "eeg.csv", index=False)
            pd.DataFrame([{"event_type": "session_start", "block_id": "", "sample_time_sec": ""}]).to_csv(session / "events.csv", index=False)
            pd.DataFrame(columns=["status"]).to_csv(session / "qc.csv", index=False)
            metadata = {
                "run_id": "run-x", "labels": {"-1": "none"},
                "session": {"sample_rate_hz": 250, "channel_names": ["c0", "c1"], "conditions": {},
                            "planned_sequence": [], "counterbalance_group": "G01"},
            }
            (session / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            for filename, columns in (
                ("probes.csv", ["probe_id"]), ("block_ratings.csv", ["block_id"]),
                ("quiz_responses.csv", ["question_id"]), ("probe_epochs.csv", ["probe_id"]),
                ("windows.csv", ["window_id"]),
            ):
                pd.DataFrame(columns=columns).to_csv(session / filename, index=False)
            (session / "session_qc_report.json").write_text('{"status":"FAIL"}', encoding="utf-8")
            mat = loadmat(export_session_to_mat(session))
            self.assertIn("session_qc_report_json", mat)
            self.assertIn("probes", mat)
            self.assertIn("block_ratings", mat)
            self.assertIn("quiz_responses", mat)


if __name__ == "__main__":
    unittest.main()
