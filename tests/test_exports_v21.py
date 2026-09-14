from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from scipy.io import loadmat

from session_recorder import ExperimentRecorder, SessionConfig


def packet(index: int, first: int = 100, second: int = -100) -> bytes:
    return (b"\xa0" + bytes([index & 255]) + first.to_bytes(3, "big", signed=True)
            + second.to_bytes(3, "big", signed=True) + bytes(24) + b"\xc0")


class RawAcquisitionExportTests(unittest.TestCase):
    def make_recorder(self, root: Path) -> ExperimentRecorder:
        rec = ExperimentRecorder(
            SessionConfig(subject_id="sub-raw", counterbalance_group="G01", study_phase="pilot"),
            root,
        )
        rec.start()
        return rec

    @staticmethod
    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_raw_packets_and_adc_counts_round_trip_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_recorder(Path(tmp))
            packets = [packet(0, 123456, -654321), packet(1, -1, 1)]
            for raw in packets:
                rec.record_packet(raw, raw)
            rec.stop(export_mat=False)
            self.assertEqual(rec.raw_path.read_bytes(), b"".join(packets))
            rows = self.read_csv(rec.eeg_path)
            self.assertEqual([(int(r["channel_0_raw"]), int(r["channel_1_raw"])) for r in rows],
                             [(123456, -654321), (-1, 1)])

    def test_gap_is_marked_without_filling_and_duplicate_is_marked(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_recorder(Path(tmp))
            for index in (10, 13, 13):
                raw = packet(index)
                rec.record_packet(raw, raw)
            rec.stop(export_mat=False)
            rows = self.read_csv(rec.eeg_path)
            self.assertEqual(len(rows), 3)
            self.assertEqual([int(r["device_sample_number"]) for r in rows], [0, 3, 3])
            self.assertEqual(rows[1]["packet_gap_before"], "2")
            self.assertIn("duplicate_index", rows[2]["quality_flag"])

    def test_saturated_sample_is_flagged_but_still_saved_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_recorder(Path(tmp))
            raw = packet(0, 8_388_607, -8_388_608)
            rec.record_packet(raw, raw)
            rec.stop(export_mat=False)
            row = self.read_csv(rec.eeg_path)[0]
            self.assertEqual(int(row["channel_0_raw"]), 8_388_607)
            self.assertEqual(int(row["channel_1_raw"]), -8_388_608)
            self.assertIn("adc_saturation", row["quality_flag"])
            self.assertEqual(rec.raw_path.read_bytes(), raw)

    def test_reconnect_starts_new_stream_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_recorder(Path(tmp))
            raw = packet(1); rec.record_packet(raw, raw)
            rec.mark_stream_discontinuity()
            raw = packet(2); rec.record_packet(raw, raw)
            rec.stop(export_mat=False)
            self.assertEqual([r["stream_segment"] for r in self.read_csv(rec.eeg_path)], ["0", "1"])

    def test_events_align_or_remain_explicitly_unaligned_and_browser_retries_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_recorder(Path(tmp))
            anchor = rec.clock_time()
            with patch.object(rec, "clock_time", return_value=anchor):
                raw = packet(0); rec.record_packet(raw, raw)
            with patch.object(rec, "clock_time", return_value=anchor + 0.1):
                rec.log_event("sync_near")
            with patch.object(rec, "clock_time", return_value=anchor + 1.0):
                rec.log_event("sync_far")
            event_id = str(uuid.uuid4())
            payload = {
                "client_event_id": event_id, "client_id": "browser-test",
                "recorder_run_id": rec.run_id, "subject_id": "sub-raw", "session_id": "ses-001",
                "counterbalance_group": "G01", "event_type": "sync_marker",
            }
            self.assertTrue(rec.handle_browser_event(payload))
            self.assertFalse(rec.handle_browser_event(payload))
            rec.browser_clients.clear()
            rec.stop(export_mat=False)
            rows = self.read_csv(rec.events_path)
            near = next(row for row in rows if row["event_type"] == "sync_near")
            far = next(row for row in rows if row["event_type"] == "sync_far")
            self.assertEqual(near["alignment_status"], "host_receive_nearest")
            self.assertEqual(far["alignment_status"], "outside_sample_tolerance")
            self.assertEqual(far["device_sample_number"], "")
            self.assertEqual(sum(row["event_id"] == event_id for row in rows), 1)

    def test_stop_creates_hashes_report_and_no_derived_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_recorder(Path(tmp))
            raw = packet(0); rec.record_packet(raw, raw)
            rec.stop(export_mat=True)
            self.assertTrue(rec.checksums_path.exists())
            self.assertTrue((rec.session_dir / "session_acquisition_report.json").exists())
            self.assertTrue((rec.session_dir / "session_raw.mat").exists())
            self.assertFalse((rec.session_dir / "windows.csv").exists())
            self.assertFalse((rec.session_dir / "probe_epochs.csv").exists())
            self.assertFalse((rec.session_dir / "eeg_preprocessed.csv").exists())
            metadata = json.loads(rec.metadata_path.read_text(encoding="utf-8"))
            for path in (rec.eeg_path, rec.raw_path):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(metadata["raw_file_sha256"][path.name], digest)
            forbidden = (rec.session_dir / "session_acquisition_report.json").read_text(encoding="utf-8")
            self.assertNotIn("training_eligible", forbidden)
            self.assertNotIn("valid_window", forbidden)
            mat = loadmat(rec.session_dir / "session_raw.mat")
            self.assertIn("session_acquisition_report_json", mat)
            self.assertNotIn("windows", mat)
            self.assertNotIn("probe_epochs", mat)

    def test_report_failure_does_not_damage_raw_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_recorder(Path(tmp))
            raw = packet(0, 7, -7); rec.record_packet(raw, raw)
            with patch("session_reports.build_session_acquisition_report", side_effect=RuntimeError("report boom")):
                rec.stop(export_mat=False)
            self.assertEqual(rec.raw_path.read_bytes(), raw)
            self.assertEqual(len(self.read_csv(rec.eeg_path)), 1)
            self.assertIn("report boom", rec.integrity_status)

    def test_mat_failure_is_nonfatal_for_raw_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.make_recorder(Path(tmp))
            raw = packet(0, 9, -9); rec.record_packet(raw, raw)
            with patch("mat_exporter.export_session_to_mat", side_effect=RuntimeError("mat boom")):
                result = rec.stop(export_mat=True)
            self.assertIsNone(result)
            self.assertEqual(rec.raw_path.read_bytes(), raw)
            self.assertTrue(rec.checksums_path.exists())
            self.assertIn("mat boom", rec.export_status)


if __name__ == "__main__":
    unittest.main()
