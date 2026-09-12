"""Regression tests using generated data only; no hardware or experiment data."""
from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.io import loadmat

import mat_exporter
from qc_metrics import channel_metrics
from session_recorder import ExperimentRecorder, SessionConfig


def packet(index: int, first: int = 1, second: int = -1) -> bytes:
    return b"\xa0" + bytes([index & 255]) + first.to_bytes(3, "big", signed=True) + second.to_bytes(3, "big", signed=True) + bytes(24) + b"\xc0"


class RecorderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rec = ExperimentRecorder(SessionConfig(subject_id="sub-test"), self.root)
        self.rec.start()
        self.anchor = self.rec.clock_time()

    def tearDown(self):
        if self.rec._active:
            self.rec.browser_clients.clear()
            self.rec.stop(export_mat=False)
        self.tmp.cleanup()

    def feed(self, count=10, start=0, offset=0.0):
        for i in range(count):
            raw = packet(start + i, i + 1, -i - 1)
            with patch.object(self.rec, "clock_time", return_value=self.anchor + offset + i / 250):
                self.rec.record_packet(raw, raw)

    def event(self, **changes):
        return {
            "subject_id": "sub-test", "session_id": "ses-001", "recorder_run_id": self.rec.run_id,
            "block_id": "1", "condition": "A", "client_event_id": "event-1",
            "event_type": "manual_sync_mark", "event_timestamp": self.anchor + 0.008,
            "clock_offset_sec": 0.0, "is_correct": True, **changes,
        }

    def rows(self):
        with self.rec.events_path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


class EventTests(RecorderCase):
    def test_delayed_event_keeps_source_block_and_occurrence_sample(self):
        self.feed(100)
        self.rec.current_block = 2
        self.rec.current_group = 1
        self.rec.current_condition = "B"
        self.rec.phase = "video"
        self.rec.handle_browser_event(self.event())
        row = self.rows()[-1]
        self.assertEqual((row["block_id"], row["condition"], row["group_id"]), ("1", "A", "1"))
        self.assertEqual(int(row["eeg_received_order"]), 2)
        self.assertEqual(float(row["sample_time_sec"]), 2 / 250)
        self.assertEqual(row["timestamp_source"], "browser_calibrated")
        self.assertEqual(self.rec.current_block, 2)
        self.assertEqual(self.rec.phase, "video")

    def test_wrong_run_or_session_rejected_without_row(self):
        for change in ({"recorder_run_id": "another-run"}, {"session_id": "ses-other"}, {"subject_id": "other"}):
            before = self.rec.event_count
            with self.assertRaises(ValueError):
                self.rec.handle_browser_event(self.event(**change))
            self.assertEqual(self.rec.event_count, before)

    def test_duplicate_event_is_written_once(self):
        self.feed()
        self.assertTrue(self.rec.handle_browser_event(self.event()))
        before = self.rec.event_count
        self.assertFalse(self.rec.handle_browser_event(self.event()))
        self.assertEqual(before, self.rec.event_count)

    def test_legacy_iso_timestamp_and_missing_identity_are_supported(self):
        self.feed()
        iso = datetime.fromtimestamp(self.anchor + 0.008, timezone.utc).isoformat()
        self.rec.handle_browser_event({"event_type": "manual_sync_mark", "iso_time": iso})
        row = self.rows()[-1]
        self.assertEqual(row["identity_status"], "legacy_partial")
        self.assertEqual(row["timestamp_source"], "browser_iso_unverified")
        self.assertEqual(int(row["eeg_received_order"]), 2)

    def test_event_in_dropout_does_not_get_invented_sample(self):
        self.feed()
        self.rec.handle_browser_event(self.event(event_timestamp=self.anchor + 3))
        row = self.rows()[-1]
        self.assertEqual(row["sample_time_sec"], "")
        self.assertEqual(row["eeg_received_order"], "")
        self.assertEqual(row["alignment_status"], "outside_sample_tolerance")

    def test_invalid_timestamp_rejected_without_row(self):
        before = self.rec.event_count
        with self.assertRaises(ValueError):
            self.rec.handle_browser_event(self.event(event_timestamp=float("nan")))
        self.assertEqual(before, self.rec.event_count)

    def test_late_lifecycle_event_does_not_rewind_live_phase(self):
        self.feed()
        self.rec.current_block = 2
        self.rec.current_condition = "B"
        self.rec.phase = "video"
        self.rec.handle_browser_event(self.event(event_type="quiz_submit", score=2, total=3))
        self.assertEqual((self.rec.current_block, self.rec.phase), (2, "video"))
        row = self.rows()[-1]
        self.assertEqual(row["event_type"], "course_quiz_summary")
        self.assertEqual(row["block_id"], "1")
        self.assertEqual(row["state_applied"], "0")

    def test_debug_block_selection_remains_unrestricted(self):
        self.rec.baseline_complete = self.rec.baseline_passed = True
        with patch.object(self.rec, "get_qc_status", return_value={"status": "good"}):
            self.rec.handle_browser_event(self.event(event_type="block_start", topic_key="a"))
            self.rec.handle_browser_event(self.event(event_type="block_start", topic_key="a", block_id="6", condition="B", client_event_id="event-2"))
        self.assertEqual((self.rec.current_block, self.rec.current_video_id), (6, "a"))

    def test_reconnect_starts_segment_without_fake_packet_loss(self):
        self.feed(2)
        self.rec.mark_stream_discontinuity()
        self.feed(1, start=0, offset=3)
        self.assertEqual(self.rec.stream_segment, 1)
        self.assertEqual(self.rec.total_missing_packets, 0)
        self.rec._flush_files()
        eeg = pd.read_csv(self.rec.eeg_path)
        self.assertEqual(eeg.iloc[-1]["sample_time_status"], "counter_elapsed_ambiguous")
        self.assertIn("stream_discontinuity", eeg.iloc[-1]["quality_flag"])

    def test_manual_event_uses_stable_host_clock(self):
        self.feed()
        with patch.object(self.rec, "clock_time", return_value=self.anchor + 0.008):
            self.rec.mark_artifact("test")
        row = self.rows()[-1]
        self.assertEqual(row["timestamp_source"], "server")
        self.assertEqual(row["eeg_received_order"], "2")
        self.assertEqual(float(row["exclude_before_sec"]), 1.5)


class FlatlineTests(RecorderCase):
    def metrics(self, values):
        return channel_metrics(values, sample_rate_hz=250, scale_uv_per_count=0.000001)

    def test_constant_and_near_zero_are_identified(self):
        for value in (0, 123, -321):
            metrics = self.metrics([value] * 751)
            self.assertTrue(metrics["constant_value"])
            self.assertTrue(metrics["near_zero_rms"])
            self.assertTrue(metrics["invalid_flatline"])

    def test_one_adc_count_signal_is_not_rejected_by_amplitude(self):
        metrics = self.metrics([0, 1] * 500)
        self.assertFalse(metrics["near_zero_rms"])
        self.assertFalse(metrics["invalid_flatline"])
        self.assertAlmostEqual(metrics["rms_counts"], 0.5)

    def test_long_frozen_interval_detected_in_otherwise_changing_window(self):
        metrics = self.metrics(list(range(100)) + [123] * 751 + list(range(100)))
        self.assertFalse(metrics["constant_value"])
        self.assertFalse(metrics["near_zero_rms"])
        self.assertTrue(metrics["invalid_flatline"])

    def test_short_constant_fragment_is_not_flagged_invalid(self):
        self.assertFalse(self.metrics([0] * 100)["invalid_flatline"])

    def test_low_amplitude_baseline_still_passes(self):
        self.rec.start_baseline(30)
        end = self.rec.clock_time()
        start = end - 30.1
        self.rec.baseline_started_timestamp = start
        for i in range(7526):
            raw = round(math.sin(2 * math.pi * 10 * i / 250))
            self.rec._qc_samples.append((start + i / 250, raw, raw, 0))
        self.rec._last_packet_timestamp = end
        self.rec.update_qc_snapshot()
        self.assertTrue(self.rec.baseline_passed)

    def test_constant_baseline_fails_and_existing_metrics_remain(self):
        self.rec.start_baseline(30)
        end = self.rec.clock_time()
        start = end - 30.1
        self.rec.baseline_started_timestamp = start
        for i in range(7526):
            self.rec._qc_samples.append((start + i / 250, 0, 123, 0))
        self.rec._last_packet_timestamp = end
        status = self.rec.update_qc_snapshot()
        self.assertFalse(self.rec.baseline_passed)
        self.assertEqual(status["status"], "bad")
        for key in ("saturation_rate_pct", "line_50hz_ratio_pct", "rms_uv"):
            self.assertIn(key, status["channels"][0])
        saved = json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))
        self.assertTrue(saved["qc"]["baseline_complete"])
        self.assertFalse(saved["qc"]["baseline_passed"])


class MatTests(RecorderCase):
    def prepare_csv(self):
        self.rec.current_block = 1
        self.rec.current_condition = "A"
        self.rec.phase = "video"
        self.feed()
        for i, value in enumerate((True, False, "", 0.75, "TRUE", "false", "NA", "unknown")):
            self.rec.log_event("test", is_correct=value, event_value="001")
        self.rec.stop(export_mat=False)

    def test_boolean_numeric_empty_and_original_text_roundtrip(self):
        self.prepare_csv()
        result = loadmat(mat_exporter.export_session_to_mat(self.rec.session_dir), simplify_cells=True)
        values = result["events"]["is_correct"]
        np.testing.assert_allclose(values[1:7], [1, 0, np.nan, .75, 1, 0], equal_nan=True)
        self.assertTrue(np.isnan(values[7:]).all())
        table = result["csv_text"]["events"]
        columns = list(table["columns"])
        self.assertEqual(table["values"][7, columns.index("is_correct")], "NA")
        self.assertEqual(table["values"][8, columns.index("is_correct")], "unknown")
        self.assertEqual(result["events"]["event_value"][1], "001")

    def test_legacy_without_group_or_qc_remains_exportable(self):
        self.prepare_csv()
        for path in (self.rec.eeg_path, self.rec.events_path):
            frame = pd.read_csv(path, keep_default_na=False)
            frame.drop(columns=[c for c in ("group_id", "stream_segment", "sample_time_status") if c in frame]).to_csv(path, index=False)
        self.rec.qc_path.unlink()
        result = loadmat(mat_exporter.export_session_to_mat(self.rec.session_dir), simplify_cells=True)
        np.testing.assert_equal(result["eeg"]["group_id"], np.ones(10))
        self.assertNotIn("qc", result)

    def test_missing_eeg_values_remain_nan_and_original_is_retained(self):
        self.prepare_csv()
        frame = pd.read_csv(self.rec.eeg_path, dtype=str, keep_default_na=False)
        frame.loc[0, "channel_0_raw"] = ""
        frame.loc[1, "base_valid_for_training"] = "False"
        frame["legacy_identifier"] = "0007"
        frame.to_csv(self.rec.eeg_path, index=False)
        result = loadmat(mat_exporter.export_session_to_mat(self.rec.session_dir), simplify_cells=True)
        self.assertTrue(np.isnan(result["eeg"]["raw"][0, 0]))
        self.assertEqual(result["eeg"]["base_valid_for_training"][1], 0)
        original = result["csv_text"]["eeg"]
        self.assertIn("legacy_identifier", original["columns"])

    def test_unknown_long_column_name_and_text_survive(self):
        self.prepare_csv()
        frame = pd.read_csv(self.rec.events_path, dtype=str, keep_default_na=False)
        column = "legacy_" + "x" * 70
        frame[column] = "None"
        frame.to_csv(self.rec.events_path, index=False)
        result = loadmat(mat_exporter.export_session_to_mat(self.rec.session_dir), simplify_cells=True)
        self.assertIn(column, result["csv_text"]["events"]["columns"])

    def test_empty_recording_exports(self):
        self.rec.stop(export_mat=True)
        self.assertTrue(self.rec.mat_path.exists())
        result = loadmat(self.rec.mat_path)
        self.assertIn("eeg", result)

    def test_failed_export_keeps_previous_mat_and_can_retry(self):
        self.feed()
        self.rec.mat_path.write_bytes(b"previous-export")
        with patch("mat_exporter.savemat", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.rec.stop(export_mat=True)
        self.assertEqual(self.rec.mat_path.read_bytes(), b"previous-export")
        metadata = json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "export_failed")
        self.assertFalse(self.rec.mat_path.with_suffix(".mat.tmp").exists())
        self.rec.stop(export_mat=True)
        metadata = json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "stopped")
        self.assertEqual(metadata["export_status"], "complete")


class ShutdownTests(RecorderCase):
    def final_event(self, client="browser-1", event_id="final"):
        self.rec.handle_browser_event(self.event(client_id=client, client_event_id=event_id,
                                                event_type="browser_shutdown_ready"))

    def acknowledgement(self, client="browser-1", event_id="final"):
        return {"run_id": self.rec.run_id, "request_id": self.rec.shutdown_request_id,
                "client_id": client, "pending": 0, "last_event_id": event_id}

    def test_stop_waits_for_final_event_and_acknowledgement(self):
        self.rec.register_browser("browser-1", 2)
        self.rec.request_shutdown()
        with self.assertRaises(RuntimeError):
            self.rec.stop(export_mat=False)
        with self.assertRaises(ValueError):
            self.rec.acknowledge_shutdown(self.acknowledgement())
        self.final_event()
        self.rec.acknowledge_shutdown(self.acknowledgement())
        self.assertTrue(self.rec.shutdown_ready)
        self.rec.stop(export_mat=True)
        rows = self.rows()
        self.assertEqual(rows[-2]["event_type"], "browser_shutdown_ready")
        self.assertEqual(rows[-1]["event_type"], "session_end")
        metadata = json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "stopped")
        self.assertEqual(metadata["export_status"], "complete")
        mat = loadmat(self.rec.mat_path, simplify_cells=True)
        self.assertEqual(json.loads(mat["metadata_json"])["status"], "stopped")

    def test_all_connected_clients_must_ack_their_own_final_event(self):
        for client in ("browser-1", "browser-2"):
            self.rec.register_browser(client)
        self.rec.request_shutdown()
        self.final_event()
        self.rec.acknowledge_shutdown(self.acknowledgement())
        self.assertFalse(self.rec.shutdown_ready)
        with self.assertRaises(ValueError):
            self.rec.acknowledge_shutdown(self.acknowledgement(client="browser-2"))
        self.final_event("browser-2", "final-2")
        self.rec.acknowledge_shutdown(self.acknowledgement("browser-2", "final-2"))
        self.assertTrue(self.rec.shutdown_ready)

    def test_events_after_ack_rejected_but_retries_still_idempotent(self):
        self.rec.register_browser("browser-1")
        self.rec.request_shutdown()
        self.final_event()
        self.rec.acknowledge_shutdown(self.acknowledgement())
        with self.assertRaises(ValueError):
            self.rec.handle_browser_event(self.event(client_id="browser-1", client_event_id="too-late"))
        self.assertFalse(self.rec.handle_browser_event(self.event(client_id="browser-1", client_event_id="final")))

    def test_fsync_failure_does_not_report_complete(self):
        self.feed()
        with patch.object(self.rec, "_flush_files", side_effect=OSError("fsync failed")):
            with self.assertRaises(OSError):
                self.rec.stop(export_mat=True)
        self.assertTrue(self.rec._active)
        self.assertFalse(self.rec.mat_path.exists())
        self.assertEqual(json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))["status"], "recording")

    def test_complete_only_when_experiment_was_complete(self):
        self.rec.phase = "complete"
        self.rec.stop(export_mat=True)
        metadata = json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "complete")
        self.assertTrue(metadata["experiment_complete"])

    def test_unresolved_payload_is_fsynced_and_not_reported_as_complete(self):
        self.rec.register_browser("browser-1")
        self.rec.request_shutdown()
        self.final_event()
        unresolved = [{"event": {"client_event_id": "rejected", "session_id": "wrong"}, "reason": "wrong session"}]
        self.rec.acknowledge_shutdown({**self.acknowledgement(), "unresolved_events": unresolved})
        recovery = json.loads((self.rec.session_dir / "pending_browser_events.json").read_text(encoding="utf-8"))
        self.assertEqual(recovery["clients"]["browser-1"], unresolved)
        self.rec.phase = "complete"
        self.rec.stop(export_mat=True)
        metadata = json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "saved_with_unresolved_events")
        self.assertEqual(metadata["unresolved_event_count"], 1)

    def test_save_retry_appends_final_end_after_new_events(self):
        with patch.object(self.rec, "_flush_files", side_effect=OSError("temporary failure")):
            with self.assertRaises(OSError):
                self.rec.stop(export_mat=False)
        self.rec.cancel_shutdown()
        self.rec.log_event("continued_debugging")
        self.rec.stop(export_mat=False)
        rows = self.rows()
        self.assertEqual(rows[-2]["event_type"], "continued_debugging")
        self.assertEqual(rows[-1]["event_type"], "session_end")


if __name__ == "__main__":
    unittest.main()
