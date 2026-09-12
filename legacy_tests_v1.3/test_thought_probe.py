"""Recorder-side tests for the stage 3 thought probe and confidence rating."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scipy.io import loadmat

import mat_exporter
from session_recorder import (
    PROBE_ATTENTION_MAP,
    PROBE_CONFIG,
    PROBE_DEBUG_CONFIG,
    PROBE_OPTIONS,
    ExperimentRecorder,
    SessionConfig,
    feasible_probe_count,
    normalize_probe_config,
    validate_probe_schedule,
)


def packet(index: int, first: int = 1, second: int = -1) -> bytes:
    return (
        b"\xa0" + bytes([index & 255])
        + first.to_bytes(3, "big", signed=True)
        + second.to_bytes(3, "big", signed=True)
        + bytes(24) + b"\xc0"
    )


class ProbeScheduleValidationTests(unittest.TestCase):
    """The hard constraints are enforced on whatever the browser reports."""

    def test_protocol_default_fits_four_probes_in_a_six_minute_video(self):
        self.assertEqual(PROBE_CONFIG["probes_per_block"], 4)
        self.assertEqual(feasible_probe_count(360.0, PROBE_CONFIG), 4)
        times, config = validate_probe_schedule([50.0, 120.0, 200.0, 300.0], 360.0, None)
        self.assertEqual(len(times), 4)
        self.assertEqual(config, PROBE_CONFIG)

    def test_first_probe_before_45_seconds_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_probe_schedule([44.0, 120.0, 200.0, 300.0], 360.0, None)

    def test_probe_inside_the_final_30_seconds_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_probe_schedule([50.0, 120.0, 200.0, 331.0], 360.0, None)

    def test_probes_closer_than_45_seconds_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_probe_schedule([50.0, 90.0, 200.0, 300.0], 360.0, None)

    def test_schedule_must_contain_every_feasible_probe(self):
        with self.assertRaises(ValueError):
            validate_probe_schedule([50.0, 120.0, 200.0], 360.0, None)
        with self.assertRaises(ValueError):
            validate_probe_schedule([50.0, 120.0, 200.0, 260.0, 320.0], 360.0, None)

    def test_short_clip_shrinks_the_count_instead_of_breaking_the_gaps(self):
        self.assertEqual(feasible_probe_count(120.0, PROBE_CONFIG), 2)
        validate_probe_schedule([45.0, 90.0], 120.0, None)
        self.assertEqual(feasible_probe_count(60.0, PROBE_CONFIG), 0)
        validate_probe_schedule([], 60.0, None)

    def test_debug_configuration_is_validated_against_its_own_margins(self):
        config = normalize_probe_config(PROBE_DEBUG_CONFIG)
        self.assertEqual(feasible_probe_count(40.0, config), 4)
        validate_probe_schedule([5.0, 12.0, 20.0, 30.0], 40.0, PROBE_DEBUG_CONFIG)
        with self.assertRaises(ValueError):
            validate_probe_schedule([4.0, 12.0, 20.0, 30.0], 40.0, PROBE_DEBUG_CONFIG)

    def test_three_decimal_rounding_does_not_fail_the_boundaries(self):
        validate_probe_schedule([44.9996, 90.0, 135.0, 269.9998], 300.0, None)

    def test_invalid_durations_and_configs_are_refused(self):
        with self.assertRaises(ValueError):
            validate_probe_schedule([50.0], 0.0, None)
        with self.assertRaises(ValueError):
            validate_probe_schedule([float("nan")], 360.0, None)
        with self.assertRaises(ValueError):
            normalize_probe_config({"min_gap_sec": 0})


class ProbeRecorderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rec = ExperimentRecorder(SessionConfig(subject_id="sub-probe"), Path(self.tmp.name))
        self.rec.start()
        self.anchor = self.rec.clock_time()
        self.step = 0
        self.rec.baseline_complete = self.rec.baseline_passed = True
        with patch.object(self.rec, "get_qc_status", return_value={"status": "good"}):
            self.rec.begin_block(1, "a")
            self.rec.start_video()
        raw = packet(1, 20, -20)
        self.rec.record_packet(raw, raw)

    def tearDown(self):
        if self.rec._active:
            self.rec.browser_clients.clear()
            self.rec.stop(export_mat=False)
        self.tmp.cleanup()

    def send(self, event_type: str, **changes):
        self.step += 1
        payload = {
            "subject_id": "sub-probe",
            "session_id": "ses-001",
            "recorder_run_id": self.rec.run_id,
            "client_event_id": f"probe-event-{self.step}",
            "client_id": "browser-probe",
            "event_type": event_type,
            "event_timestamp": self.anchor + self.step * 0.01,
            "client_timestamp": self.anchor + self.step * 0.01 + 0.05,
            "clock_offset_sec": -0.05,
            "block_id": "1",
            "condition": "A",
            "topic_key": "a",
        }
        payload.update(changes)
        return self.rec.handle_browser_event(payload)

    def schedule(self, times=(50.0, 120.0, 200.0, 300.0), duration=360.0, **changes):
        return self.send(
            "probe_schedule", probe_schedule_id="sched-1", probe_times=list(times),
            video_duration_sec=duration, video_id="a", **changes,
        )

    def full_probe(self, probe_id="1_1", response=1, confidence=3, planned=50.0):
        self.send("probe_onset", probe_id=probe_id, probe_index=1, probe_schedule_id="sched-1",
                  probe_planned_time_sec=planned, video_time_sec=planned, pause_reason="thought_probe")
        self.send("attention_response", probe_id=probe_id, probe_index=1, probe_schedule_id="sched-1",
                  probe_planned_time_sec=planned, video_time_sec=planned, response=response,
                  probe_attention=PROBE_ATTENTION_MAP[response], response_time_ms=1200)
        self.send("confidence_response", probe_id=probe_id, probe_index=1, probe_schedule_id="sched-1",
                  probe_planned_time_sec=planned, video_time_sec=planned, confidence=confidence,
                  response_time_ms=800)
        self.send("video_resume", probe_id=probe_id, probe_index=1, probe_schedule_id="sched-1",
                  video_time_sec=planned, pause_reason="thought_probe")

    def rows(self, *event_types):
        with self.rec.events_path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [row for row in rows if not event_types or row["event_type"] in event_types]


class ProbeEventTests(ProbeRecorderCase):
    def test_schedule_is_stored_on_the_block_and_in_metadata(self):
        self.schedule()
        stored = self.rec.probe_schedules[1]
        self.assertEqual(stored["probe_times_sec"], [50.0, 120.0, 200.0, 300.0])
        self.assertTrue(stored["uses_protocol_default_config"])
        session = json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))["session"]
        self.assertEqual(session["probe_config"], PROBE_CONFIG)
        self.assertEqual(session["probe_schedules"]["1"]["probe_times_sec"], [50.0, 120.0, 200.0, 300.0])
        row = self.rows("probe_schedule")[0]
        self.assertEqual(row["event_value"], "50,120,200,300")
        self.assertEqual(row["probe_index"], "4")

    def test_recorder_refuses_a_schedule_that_breaks_the_constraints(self):
        with self.assertRaises(ValueError):
            self.schedule(times=(10.0, 120.0, 200.0, 300.0))
        self.assertNotIn(1, self.rec.probe_schedules)

    def test_probe_chain_is_recorded_with_every_required_field(self):
        self.schedule()
        self.full_probe(response=2, confidence=4)
        chain = self.rows("probe_onset", "attention_response", "confidence_response", "video_resume")
        self.assertEqual([row["event_type"] for row in chain],
                         ["probe_onset", "attention_response", "confidence_response", "video_resume"])
        for row in chain:
            self.assertEqual(row["probe_id"], "1_1")
            self.assertEqual(row["block_id"], "1")
            self.assertEqual(row["video_id"], "a")
            self.assertEqual(row["condition_label"], "A")
            self.assertEqual(row["condition_type"], "focused")
            self.assertEqual(row["subject_id"], "sub-probe")
            self.assertEqual(row["session_id"], "ses-001")
            self.assertEqual(float(row["video_time_sec"]), 50.0)
            self.assertTrue(row["browser_timestamp"])
            self.assertTrue(row["recorder_timestamp"])
            self.assertEqual(row["device_sample_number"], row["eeg_sample_number"])
            self.assertNotEqual(row["device_sample_number"], "")
        onset, attention, confidence, resume = chain
        self.assertEqual(onset["pause_reason"], "thought_probe")
        self.assertEqual(attention["response"], "2")
        self.assertEqual(attention["response_label"], PROBE_OPTIONS[2])
        self.assertEqual(attention["probe_attention"], "OFF")
        self.assertEqual(attention["response_time_ms"], "1200")
        self.assertEqual(confidence["confidence"], "4")
        self.assertEqual(confidence["response_time_ms"], "800")
        self.assertEqual(resume["pause_reason"], "thought_probe")

    def test_attention_answers_map_to_on_off_off_ambiguous(self):
        self.schedule()
        for response in (1, 2, 3, 4):
            self.send("probe_onset", probe_id=f"1_{response}", probe_index=response)
            self.send("attention_response", probe_id=f"1_{response}", probe_index=response, response=response)
            self.send("confidence_response", probe_id=f"1_{response}", probe_index=response, confidence=1)
        answers = self.rows("attention_response")
        self.assertEqual([row["probe_attention"] for row in answers], ["ON", "OFF", "OFF", "AMBIGUOUS"])

    def test_condition_a_keeps_a_subtraction_answer_untouched(self):
        self.schedule()
        self.full_probe(response=2)
        row = self.rows("attention_response")[0]
        self.assertEqual(row["condition_label"], "A")
        self.assertEqual(row["response"], "2")
        self.assertEqual(row["response_label"], "连续减7心算任务")
        self.assertEqual(row["probe_attention"], "OFF")
        stored = self.rec.probe_schedules[1]["responses"]["1_1"]
        self.assertEqual(stored, {"response": 2, "probe_attention": "OFF", "confidence": 3})

    def test_confidence_outside_one_to_four_is_rejected(self):
        self.schedule()
        self.send("probe_onset", probe_id="1_1", probe_index=1)
        self.send("attention_response", probe_id="1_1", probe_index=1, response=1)
        for bad in (0, 5, -1):
            with self.assertRaises(ValueError):
                self.send("confidence_response", probe_id="1_1", probe_index=1, confidence=bad)
        self.assertEqual(self.rows("confidence_response"), [])
        self.send("confidence_response", probe_id="1_1", probe_index=1, confidence=2)
        self.assertEqual(self.rows("confidence_response")[0]["confidence"], "2")

    def test_probe_response_outside_the_four_options_is_rejected(self):
        self.schedule()
        self.send("probe_onset", probe_id="1_1", probe_index=1)
        for bad in (0, 5):
            with self.assertRaises(ValueError):
                self.send("attention_response", probe_id="1_1", probe_index=1, response=bad)

    def test_the_chain_cannot_be_skipped_or_reordered(self):
        self.schedule()
        with self.assertRaises(ValueError):
            self.send("attention_response", probe_id="1_1", probe_index=1, response=1)
        self.send("probe_onset", probe_id="1_1", probe_index=1)
        with self.assertRaises(ValueError):
            self.send("probe_onset", probe_id="1_1", probe_index=1)
        with self.assertRaises(ValueError):
            self.send("confidence_response", probe_id="1_1", probe_index=1, confidence=3)
        with self.assertRaises(ValueError):
            self.send("video_resume", probe_id="1_1", pause_reason="thought_probe")
        self.send("attention_response", probe_id="1_1", probe_index=1, response=1)
        self.send("confidence_response", probe_id="1_1", probe_index=1, confidence=3)
        self.send("video_resume", probe_id="1_1", pause_reason="thought_probe")
        self.assertEqual(self.rec._probe_stages["1_1"], "resumed")

    def test_probe_events_require_an_identifier(self):
        self.schedule()
        with self.assertRaises(ValueError):
            self.send("probe_onset", probe_index=1)

    def test_probe_pauses_the_recorded_phase_and_resume_restores_it(self):
        self.schedule()
        self.assertEqual(self.rec.phase, "video")
        self.send("probe_onset", probe_id="1_1", probe_index=1)
        self.assertEqual(self.rec.phase, "thought_probe")
        raw = packet(2, 30, -30)
        self.rec.record_packet(raw, raw)
        self.send("attention_response", probe_id="1_1", probe_index=1, response=1)
        self.send("confidence_response", probe_id="1_1", probe_index=1, confidence=3)
        self.send("video_resume", probe_id="1_1", pause_reason="thought_probe")
        self.assertEqual(self.rec.phase, "video")
        self.rec._flush_files()
        with self.rec.eeg_path.open(encoding="utf-8") as handle:
            eeg = list(csv.DictReader(handle))
        self.assertEqual(eeg[-1]["phase"], "thought_probe")
        self.assertEqual(eeg[-1]["is_formal_experiment"], "0")
        self.assertEqual(eeg[-1]["base_valid_for_training"], "0")
        self.assertEqual(eeg[-1]["weak_label"], "-1")

    def test_cancelled_probe_keeps_the_partial_chain_auditable(self):
        self.schedule()
        self.send("probe_onset", probe_id="1_1", probe_index=1)
        self.send("probe_cancelled", probe_id="1_1", probe_index=1, reason="实验员紧急结束")
        self.assertEqual(self.rec._probe_stages["1_1"], "cancelled")
        self.assertEqual(self.rec.phase, "paused")
        self.assertEqual(self.rows("probe_cancelled")[0]["event_value"], "实验员紧急结束")


class ProbeExportTests(ProbeRecorderCase):
    def test_probe_columns_survive_the_mat_export(self):
        self.schedule()
        self.full_probe(response=4, confidence=2)
        self.rec.browser_clients.clear()
        self.rec.stop(export_mat=True)
        result = loadmat(self.rec.mat_path, simplify_cells=True)
        events = result["events"]
        types = list(events["event_type"])
        attention = types.index("attention_response")
        confidence = types.index("confidence_response")
        self.assertEqual(events["probe_id"][attention], "1_1")
        self.assertEqual(events["probe_attention"][attention], "AMBIGUOUS")
        self.assertEqual(events["response"][attention], 4)
        self.assertEqual(events["response_time_ms"][attention], 1200)
        self.assertEqual(events["confidence"][confidence], 2)
        self.assertEqual(events["pause_reason"][types.index("probe_onset")], "thought_probe")
        self.assertEqual(events["video_time_sec"][attention], 50.0)
        metadata = json.loads(result["metadata_json"])
        self.assertEqual(metadata["thought_probe"]["attention_map"],
                         {str(key): value for key, value in PROBE_ATTENTION_MAP.items()})

    def test_legacy_sessions_without_probe_columns_still_export(self):
        self.rec.browser_clients.clear()
        self.rec.stop(export_mat=False)
        import pandas as pd

        frame = pd.read_csv(self.rec.events_path, dtype=str, keep_default_na=False)
        frame = frame.drop(columns=[column for column in frame.columns if column.startswith("probe_")])
        frame.to_csv(self.rec.events_path, index=False)
        result = loadmat(mat_exporter.export_session_to_mat(self.rec.session_dir), simplify_cells=True)
        self.assertIn("block_start", list(result["events"]["event_type"]))
        self.assertNotIn("probe_id", result["events"])


if __name__ == "__main__":
    unittest.main()
