from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from qc_metrics import channel_metrics
from session_recorder import (
    EEG_CSV_COLUMNS, EVENT_CSV_COLUMNS, QC_CSV_COLUMNS,
    ExperimentRecorder, SessionConfig,
)


def packet(index: int, first: int = 100, second: int = -100) -> bytes:
    return b"\xa0" + bytes([index & 255]) + first.to_bytes(3, "big", signed=True) + second.to_bytes(3, "big", signed=True) + bytes(24) + b"\xc0"


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rec = ExperimentRecorder(SessionConfig(subject_id="sub-test", counterbalance_group="G01", study_phase="pilot"), self.root)
        self.rec.start()

    def tearDown(self):
        if self.rec._active:
            # Individual unit tests call the browser-event handler directly;
            # queue-drain handshaking is covered separately by integration
            # tests, so no live browser remains at teardown.
            self.rec.browser_clients.clear()
            self.rec.stop(export_mat=False)
        self.tmp.cleanup()

    def payload(self, event_type: str, **extra):
        return {
            "client_event_id": str(uuid.uuid4()), "client_id": "browser-test",
            "recorder_run_id": self.rec.run_id, "subject_id": "sub-test", "session_id": "ses-001",
            "counterbalance_group": "G01", "event_type": event_type,
            **extra,
        }

    def unlock(self):
        self.rec.subtraction_practice["confirmed"] = True
        self.rec.baseline_complete = True
        self.rec.baseline_passed = True
        self.rec._cached_qc = {"status": "good", "signal_alive": True, "messages": [], "channels": []}

    def test_baseline_is_fixed_to_thirty_seconds_and_practice_gated(self):
        with self.assertRaisesRegex(RuntimeError, "连续减7练习"):
            self.rec.start_baseline(30)
        self.rec.subtraction_practice["confirmed"] = True
        with self.assertRaisesRegex(ValueError, "固定为30秒"):
            self.rec.start_baseline(60)
        self.rec.start_baseline(30)
        self.assertEqual(self.rec.baseline_target_sec, 30)

    def test_three_unique_prime_start_numbers_are_pre_generated(self):
        values = [item["start_number"] for item in self.rec.b_start_numbers.values()]
        self.assertEqual(len(values), 3)
        self.assertEqual(len(set(values)), 3)
        self.assertTrue(all(800 <= value <= 1000 for value in values))
        self.assertTrue(all(all(value % divisor for divisor in range(2, int(value ** 0.5) + 1)) for value in values))

    def test_block_video_and_order_are_locked(self):
        self.unlock()
        with self.assertRaisesRegex(ValueError, "must use V1"):
            self.rec.begin_block(1, "V2")
        self.rec.begin_block(1, "V1")
        self.rec.start_video()
        self.rec.end_video()
        self.rec.end_block()
        with self.assertRaisesRegex(RuntimeError, "下一个必须是Block 2"):
            self.rec.begin_block(3, "V3")

    def test_self_caught_has_motor_and_sensitivity_masks(self):
        self.unlock(); self.rec.begin_block(1, "V1"); self.rec.start_video()
        self.rec.handle_browser_event(self.payload("self_caught", block_id=1, condition="A", video_id="V1"))
        self.rec._event_handle.flush()
        with self.rec.events_path.open(encoding="utf-8", newline="") as handle:
            row = [row for row in csv.DictReader(handle) if row["event_type"] == "self_caught"][-1]
        self.assertEqual(row["exclude_before_sec"], "2.0")
        self.assertEqual(row["exclude_after_sec"], "2.0")
        self.assertEqual(row["sensitivity_exclude_before_sec"], "10.0")

    def test_complete_rating_fields_are_first_class_columns(self):
        self.unlock(); self.rec.begin_block(1, "V1"); self.rec.start_video(); self.rec.end_video()
        self.rec.handle_browser_event(self.payload(
            "rating_end", block_id=1, condition="A", video_id="V1",
            course_attention_rating=4, mental_effort=5, video_interest=3, video_difficulty=2,
        ))
        self.rec._event_handle.flush()
        with self.rec.events_path.open(encoding="utf-8", newline="") as handle:
            row = [row for row in csv.DictReader(handle) if row["event_type"] == "rating_end"][-1]
        self.assertEqual(row["mental_effort"], "5")
        self.assertEqual(row["video_interest"], "3")
        self.assertEqual(row["video_difficulty"], "2")

    def test_b_rating_requires_all_three_subtraction_fields(self):
        self.unlock(); self.rec.begin_block(1, "V1"); self.rec.start_video(); self.rec.end_video()
        self.rec.current_condition = "B"
        payload = self.payload("rating_end", block_id=1, condition="B", video_id="V1",
                               course_attention_rating=2, mental_effort=8, video_interest=3, video_difficulty=2)
        with self.assertRaises((TypeError, ValueError)):
            self.rec.handle_browser_event(payload)

    def test_event_and_qc_schemas_include_required_direct_fields(self):
        for field in ("event_name", "event_type", "recorder_timestamp", "browser_timestamp",
                      "device_sample_number", "subject_id", "session_id", "block_order",
                      "video_id", "condition_label", "client_event_id", "counterbalance_group"):
            self.assertIn(field, EVENT_CSV_COLUMNS)
        self.assertIn("signal_alive", QC_CSV_COLUMNS)
        self.assertNotIn("channel_0_line_50hz_ratio_pct", QC_CSV_COLUMNS)
        self.assertIn("counterbalance_group", EEG_CSV_COLUMNS)

    def test_channel_qc_has_no_50hz_ratio(self):
        metrics = channel_metrics(range(600), sample_rate_hz=250, scale_uv_per_count=1.0)
        self.assertNotIn("line_50hz_ratio_pct", metrics)
        self.assertIn("high_frequency_ratio_pct", metrics)

    def test_two_second_dropout_and_recovery_are_recorded(self):
        self.unlock(); self.rec.begin_block(1, "V1"); self.rec.start_video()
        anchor = self.rec.clock_time()
        with patch.object(self.rec, "clock_time", return_value=anchor):
            raw = packet(0); self.rec.record_packet(raw, raw)
        with patch.object(self.rec, "clock_time", return_value=anchor + 2.1):
            status = self.rec.update_qc_snapshot()
        self.assertFalse(status["signal_alive"])
        self.assertEqual(self.rec.phase, "paused")
        with patch.object(self.rec, "clock_time", return_value=anchor + 2.2):
            raw = packet(1); self.rec.record_packet(raw, raw)
        self.assertEqual(len(self.rec.dropout_records), 1)
        self.assertGreaterEqual(self.rec.dropout_records[0]["duration_sec"], 2.0)

    def test_override_requires_operator_and_reason(self):
        self.rec.baseline_complete = True
        self.rec.baseline_passed = False
        with self.assertRaises(ValueError):
            self.rec.approve_baseline_override("", "reason")
        with self.assertRaises(ValueError):
            self.rec.approve_baseline_override("op-1", "")
        self.rec.approve_baseline_override("op-1", "electrode checked")
        self.assertEqual(self.rec.baseline_override_details["operator_id"], "op-1")


if __name__ == "__main__":
    unittest.main()
