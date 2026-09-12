from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from order_balance import recommend_order
from qc_metrics import ADC_LIMIT, window_metrics
from session_recorder import ExperimentRecorder, SessionConfig


class QcMetricTests(unittest.TestCase):
    def _samples(self, frequency_hz: float, seconds: float = 10.0, missing: int = 0):
        sample_rate = 250
        count = int(sample_rate * seconds)
        return [
            (
                index / sample_rate,
                int(10_000 * math.sin(2 * math.pi * frequency_hz * index / sample_rate)),
                int(8_000 * math.sin(2 * math.pi * frequency_hz * index / sample_rate)),
                missing if index == count // 2 else 0,
            )
            for index in range(count)
        ]

    def test_detects_50_hz_dominance(self):
        result = window_metrics(
            self._samples(50.0), sample_rate_hz=250, scale_uv_per_count=0.01
        )
        self.assertGreater(result["channels"][0]["line_50hz_ratio_pct"], 95.0)

    def test_non_line_signal_has_low_50_hz_ratio(self):
        result = window_metrics(
            self._samples(10.0), sample_rate_hz=250, scale_uv_per_count=0.01
        )
        self.assertLess(result["channels"][0]["line_50hz_ratio_pct"], 1.0)

    def test_reports_loss_and_near_rail_saturation(self):
        samples = self._samples(10.0, missing=10)
        timestamp, _ch0, ch1, gap = samples[0]
        samples[0] = (timestamp, int(ADC_LIMIT * 0.995), ch1, gap)
        result = window_metrics(samples, sample_rate_hz=250, scale_uv_per_count=0.01)
        self.assertGreater(result["packet_loss_rate_pct"], 0)
        self.assertGreater(result["channels"][0]["saturation_rate_pct"], 0)


class OrderBalanceTests(unittest.TestCase):
    def _write_metadata(self, root: Path, subject: str, order: int, started: float):
        path = root / "data" / subject / "ses-001" / f"run-{int(started)}" / "metadata.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"session": {"subject_id": subject, "order": order, "started_timestamp": started}}),
            encoding="utf-8",
        )

    def test_first_new_v13_subject_starts_with_order_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(recommend_order(Path(tmp), "sub-001")["order"], 2)

    def test_new_subject_alternates_and_repeat_subject_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_metadata(root, "sub-001", 2, 1.0)
            self.assertEqual(recommend_order(root, "sub-002")["order"], 1)
            self.assertEqual(recommend_order(root, "sub-001")["order"], 2)
            self.assertTrue(recommend_order(root, "sub-001")["existing_subject"])


class BaselineGateTests(unittest.TestCase):
    def test_clean_30_second_baseline_unlocks_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = ExperimentRecorder(
                SessionConfig(subject_id="sub-test", output_root=str(root / "data")), root
            )
            recorder.start()
            try:
                recorder.start_baseline(30)
                end = time.time()
                start = end - 30.1
                recorder.baseline_started_timestamp = start
                for index in range(int(30.1 * 250) + 1):
                    timestamp = start + index / 250
                    raw = int(10_000 * math.sin(2 * math.pi * 10 * index / 250))
                    recorder._qc_samples.append((timestamp, raw, raw, 0))
                recorder._last_packet_timestamp = end
                result = recorder.update_qc_snapshot()
                self.assertTrue(result["baseline"]["complete"])
                self.assertTrue(result["baseline"]["passed"])
                self.assertTrue(recorder.qc_ready_for_experiment)
                self.assertEqual(recorder.begin_block(1, "V1"), "A")
            finally:
                recorder.stop(export_mat=False)

    def test_block_is_locked_before_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = ExperimentRecorder(
                SessionConfig(subject_id="sub-test", output_root=str(root / "data")), root
            )
            recorder.start()
            with self.assertRaises(RuntimeError):
                recorder.begin_block(1, "V1")
            recorder.stop(export_mat=False)

    def test_three_seconds_without_any_eeg_is_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = ExperimentRecorder(
                SessionConfig(subject_id="sub-test", output_root=str(root / "data")), root
            )
            recorder.start()
            try:
                recorder.started_timestamp = time.time() - 3.5
                status = recorder.get_qc_status()
                self.assertEqual(status["status"], "bad")
                self.assertIn("重新连接", status["messages"][0])
            finally:
                recorder.stop(export_mat=False)


class LineNoisePolicyTests(unittest.TestCase):
    """50 Hz is measured and reported, but never rejects a window on its own."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.recorder = ExperimentRecorder(
            SessionConfig(subject_id="sub-line", output_root=str(root / "data")), root
        )
        self.recorder.start()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.recorder.stop, export_mat=False)

    def fill(self, frequency_hz: float, *, seconds: float = 12.0, amplitude: int = 10_000,
             saturate: bool = False, flat: bool = False, missing: int = 0) -> None:
        end = time.time()
        start = end - seconds
        count = int(seconds * 250) + 1
        for index in range(count):
            timestamp = start + index / 250
            raw = 0 if flat else int(amplitude * math.sin(2 * math.pi * frequency_hz * index / 250))
            # Keep the defects inside the QC rolling window, not at the oldest sample.
            if saturate and index == count - 1:
                raw = int(ADC_LIMIT * 0.995)
            gap = missing if index == count - 1 else 0
            self.recorder._qc_samples.append((timestamp, raw, raw, gap))
        self.recorder._last_packet_timestamp = end

    def snapshot(self, *, age_sec: float = 0.0) -> dict:
        return self.recorder.update_qc_snapshot()

    def qc_rows(self) -> list[dict]:
        self.recorder._flush_files()
        import csv
        with self.recorder.qc_path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_pure_50_hz_is_a_warning_and_not_bad(self):
        self.fill(50.0)
        result = self.snapshot()
        self.assertEqual(result["status"], "warning")
        self.assertNotEqual(result["status"], "bad")
        self.assertTrue(result["warning_50hz"])
        self.assertTrue(result["needs_notch"])
        self.assertTrue(any("50 Hz" in message for message in result["messages"]))

    def test_the_50_hz_ratio_is_still_measured_and_written_to_qc_csv(self):
        self.fill(50.0)
        self.snapshot()
        row = self.qc_rows()[-1]
        self.assertGreater(float(row["channel_0_line_50hz_ratio_pct"]), 50.0)
        self.assertGreater(float(row["channel_1_line_50hz_ratio_pct"]), 50.0)
        self.assertEqual(row["warning_50hz"], "True")
        self.assertEqual(row["needs_notch"], "True")
        self.assertEqual(row["status"], "warning")
        self.assertIn("50 Hz", row["messages"])

    def test_persistent_50_hz_still_never_becomes_bad(self):
        self.fill(50.0)
        first = self.snapshot()
        self.assertEqual(first["status"], "warning")
        # Push the first sighting far enough back to exceed the persistence window.
        started = self.recorder._line_noise_started_at
        self.recorder._line_noise_started_at = [
            value - self.recorder.config.line_noise_persistence_sec * 3 for value in started
        ]
        second = self.snapshot()
        self.assertEqual(second["status"], "warning")
        self.assertTrue(second["needs_notch"])
        self.assertTrue(any("持续" in message for message in second["messages"]))

    def test_clean_signal_clears_the_notch_flag(self):
        self.fill(10.0)
        result = self.snapshot()
        self.assertEqual(result["status"], "good")
        self.assertFalse(result["warning_50hz"])
        self.assertFalse(result["needs_notch"])

    def test_saturation_is_still_bad_even_without_line_noise(self):
        self.fill(10.0, saturate=True)
        result = self.snapshot()
        self.assertEqual(result["status"], "bad")
        self.assertFalse(result["needs_notch"])

    def test_a_flat_channel_is_still_bad(self):
        self.fill(10.0, flat=True)
        self.assertEqual(self.snapshot()["status"], "bad")

    def test_heavy_packet_loss_is_still_bad(self):
        self.fill(10.0, missing=5_000)
        self.assertEqual(self.snapshot()["status"], "bad")

    def test_saturation_together_with_line_noise_is_still_bad(self):
        self.fill(50.0, saturate=True)
        result = self.snapshot()
        self.assertEqual(result["status"], "bad")
        self.assertTrue(result["needs_notch"])

    def test_a_baseline_with_only_high_50_hz_still_passes(self):
        self.recorder.start_baseline(30)
        end = time.time()
        self.recorder.baseline_started_timestamp = end - 30.1
        self.fill(50.0, seconds=30.1)
        result = self.recorder.update_qc_snapshot()
        self.assertTrue(result["baseline"]["complete"])
        self.assertTrue(result["baseline"]["passed"], result["messages"])
        self.assertTrue(self.recorder.qc_ready_for_experiment)
        self.assertEqual(self.recorder.begin_block(1, "V1"), "A")

    def test_a_baseline_with_saturation_still_fails(self):
        self.recorder.start_baseline(30)
        end = time.time()
        self.recorder.baseline_started_timestamp = end - 30.1
        self.fill(10.0, seconds=30.1, saturate=True)
        result = self.recorder.update_qc_snapshot()
        self.assertTrue(result["baseline"]["complete"])
        self.assertFalse(result["baseline"]["passed"])

    def test_metadata_records_the_line_noise_policy(self):
        self.fill(10.0)
        self.snapshot()
        metadata = json.loads(self.recorder.metadata_path.read_text(encoding="utf-8"))
        policy = metadata["qc"]["line_noise_policy"]
        self.assertIn("never marks a window reject", policy["recording_stage"])
        self.assertEqual(policy["flags"], ["warning_50hz", "needs_notch"])
        self.assertIn("never filtered", policy["raw_eeg"])
        self.assertIn("notch", policy["analysis_stage"])
        self.assertIn("saturation", policy["reject_reasons"])
        self.assertIn("sample discontinuity", policy["reject_reasons"])


if __name__ == "__main__":
    unittest.main()
