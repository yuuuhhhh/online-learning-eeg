"""Regression tests for the v2.1 development-stage experiment skeleton."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from scipy.io import loadmat

import mat_exporter
from session_recorder import (
    CONDITIONS,
    LEGACY_ARITHMETIC_EVENTS,
    REST_DURATIONS_AFTER_BLOCK,
    SEQUENCES,
    ExperimentRecorder,
    SessionConfig,
    _is_prime,
    generate_unique_b_start_numbers,
)


def packet(index: int, first: int = 1, second: int = -1) -> bytes:
    return (
        b"\xa0" + bytes([index & 255])
        + first.to_bytes(3, "big", signed=True)
        + second.to_bytes(3, "big", signed=True)
        + bytes(24) + b"\xc0"
    )


class V21RecorderCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rec = ExperimentRecorder(SessionConfig(subject_id="sub-v21"), self.root)
        self.rec.start()
        self.anchor = self.rec.clock_time()

    def tearDown(self):
        if self.rec._active:
            self.rec.browser_clients.clear()
            self.rec.stop(export_mat=False)
        self.tmp.cleanup()

    def browser_event(self, event_type: str, **changes):
        payload = {
            "subject_id": "sub-v21",
            "session_id": "ses-001",
            "recorder_run_id": self.rec.run_id,
            "client_event_id": f"event-{event_type}-{len(changes)}",
            "client_id": "browser-v21",
            "event_type": event_type,
            "event_timestamp": self.anchor + 0.01,
            "clock_offset_sec": 0.0,
        }
        payload.update(changes)
        return payload

    def rows(self):
        with self.rec.events_path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


class ConditionAndNumberTests(V21RecorderCase):
    def test_condition_types_are_protocol_values(self):
        self.assertEqual(CONDITIONS["A"]["condition_type"], "focused")
        self.assertEqual(CONDITIONS["B"]["condition_type"], "bbbd_subtraction")

    def test_each_sequence_gets_three_unique_primes_in_range(self):
        for sequence in SEQUENCES.values():
            values = list(generate_unique_b_start_numbers(sequence).values())
            self.assertEqual(len(values), 3)
            self.assertEqual(len(set(values)), 3)
            self.assertTrue(all(800 <= value <= 1000 and _is_prime(value) for value in values))

    def test_session_metadata_pre_generates_only_b_block_numbers(self):
        metadata = json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))
        numbers = metadata["session"]["b_start_numbers"]
        self.assertEqual(set(map(int, numbers)), {2, 4, 6})
        values = [item["start_number"] for item in numbers.values()]
        self.assertEqual(len(set(values)), 3)
        self.assertTrue(all(800 <= value <= 1000 and _is_prime(value) for value in values))
        self.assertTrue(all(item["generated_timestamp"] for item in numbers.values()))

    def test_eeg_rows_store_condition_type_without_attention_truth_label(self):
        self.rec.baseline_complete = self.rec.baseline_passed = True
        with patch.object(self.rec, "get_qc_status", return_value={"status": "good"}):
            self.rec.begin_block(1, "a")
            self.rec.start_video()
        raw = packet(1, 20, -20)
        self.rec.record_packet(raw, raw)
        self.rec._flush_files()
        eeg = pd.read_csv(self.rec.eeg_path, keep_default_na=False)
        row = eeg.iloc[-1]
        self.assertEqual(row["condition"], "A")
        self.assertEqual(row["condition_type"], "focused")
        self.assertEqual(int(row["condition_code"]), 0)
        self.assertEqual(int(row["weak_label"]), -1)

    def test_b_eeg_rows_use_subtraction_condition_type(self):
        self.rec.baseline_complete = self.rec.baseline_passed = True
        with patch.object(self.rec, "get_qc_status", return_value={"status": "good"}):
            self.rec.begin_block(2, "b")
            self.rec.start_video()
        raw = packet(1, 20, -20)
        self.rec.record_packet(raw, raw)
        self.rec._flush_files()
        row = pd.read_csv(self.rec.eeg_path, keep_default_na=False).iloc[-1]
        self.assertEqual(row["condition"], "B")
        self.assertEqual(row["condition_type"], "bbbd_subtraction")
        self.assertEqual(int(row["condition_code"]), 1)
        self.assertEqual(int(row["weak_label"]), -1)
        self.rec.stop(export_mat=True)
        result = loadmat(self.rec.mat_path, simplify_cells=True)
        self.assertEqual(list(result["condition_types"]), ["focused", "bbbd_subtraction"])
        self.assertEqual(int(result["eeg"]["weak_label"]), -1)
        self.assertEqual(result["eeg"]["condition_type"], "bbbd_subtraction")


class EventAndRestTests(V21RecorderCase):
    def test_new_session_rejects_every_legacy_arithmetic_event(self):
        before = self.rec.event_count
        for index, event_type in enumerate([*sorted(LEGACY_ARITHMETIC_EVENTS), "arithmetic_custom", "math_custom"]):
            with self.assertRaises(ValueError):
                self.rec.handle_browser_event(self.browser_event(
                    event_type, client_event_id=f"legacy-{index}", block_id="2", condition="B"
                ))
            with self.assertRaises(ValueError):
                self.rec.log_event(event_type)
        self.assertEqual(self.rec.event_count, before)

    def test_practice_events_are_nonformal_and_leave_no_block_context(self):
        for index, event_type in enumerate((
            "subtraction_practice_start", "subtraction_practice_end", "practice_confirmed"
        )):
            self.rec.handle_browser_event(self.browser_event(
                event_type, client_event_id=f"practice-{index}", practice_number=100,
                block_id="", condition="", group_id="", is_formal_experiment=0,
                event_timestamp=self.anchor + index * 0.01,
            ))
        rows = [row for row in self.rows() if row["event_type"].startswith("subtraction_") or row["event_type"] == "practice_confirmed"]
        self.assertEqual([row["event_type"] for row in rows], [
            "subtraction_practice_start", "subtraction_practice_end", "practice_confirmed"
        ])
        self.assertTrue(all(row["block_id"] == "" and row["condition"] == "" for row in rows))
        self.assertTrue(all(row["is_formal_experiment"] == "0" for row in rows))
        self.assertTrue(self.rec.subtraction_practice["confirmed"])

    def test_rest_plan_and_recorded_duration_fields(self):
        self.assertEqual(list(REST_DURATIONS_AFTER_BLOCK.values()), [30, 30, 180, 30, 30])
        self.rec.handle_browser_event(self.browser_event(
            "rest_start", client_event_id="rest-start", block_id="3", condition="A",
            after_block_id=3, planned_duration=180, actual_duration="",
        ))
        self.rec.handle_browser_event(self.browser_event(
            "rest_end", client_event_id="rest-end", block_id="3", condition="A",
            after_block_id=3, planned_duration=180, actual_duration=2.5,
            event_timestamp=self.anchor + 2.5,
        ))
        rows = [row for row in self.rows() if row["event_type"] in {"rest_start", "rest_end"}]
        self.assertEqual([row["planned_duration"] for row in rows], ["180", "180"])
        self.assertEqual(rows[1]["actual_duration"], "2.5")
        self.assertEqual(rows[1]["after_block_id"], "3")
        self.assertTrue(all(row["is_formal_experiment"] == "0" for row in rows))

    def test_start_number_events_keep_generation_display_and_hide_times(self):
        info = self.rec.b_start_numbers[2]
        for event_type, extra in (
            ("subtraction_start_number_displayed", {"displayed_timestamp": self.anchor + 1}),
            ("subtraction_start_number_hidden", {
                "displayed_timestamp": self.anchor + 1, "hidden_timestamp": self.anchor + 6,
            }),
        ):
            self.rec.handle_browser_event(self.browser_event(
                event_type, client_event_id=event_type, block_id="2", condition="B",
                start_number=info["start_number"], generated_timestamp=info["generated_timestamp"],
                **extra,
            ))
        rows = [row for row in self.rows() if row["event_type"].startswith("subtraction_start_number_")]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["block_id"] == "2" and row["condition_type"] == "bbbd_subtraction" for row in rows))
        self.assertEqual(int(rows[0]["start_number"]), info["start_number"])
        self.assertTrue(rows[0]["generated_timestamp"])
        self.assertTrue(rows[0]["displayed_timestamp"])
        self.assertTrue(rows[1]["hidden_timestamp"])


class LegacyMatCompatibilityTests(V21RecorderCase):
    def test_old_arithmetic_rows_remain_readable_in_mat(self):
        self.rec.stop(export_mat=False)
        frame = pd.read_csv(self.rec.events_path, dtype=str, keep_default_na=False)
        legacy = {column: "" for column in frame.columns}
        legacy.update({
            "event_timestamp": "123.5", "block_id": "2", "condition": "B",
            "event_type": "arithmetic_response", "event_value": "8 + 4 = 12",
            "participant_answer": "F", "is_correct": "True", "reaction_time_ms": "650",
        })
        frame = pd.concat([frame, pd.DataFrame([legacy])], ignore_index=True)
        frame.to_csv(self.rec.events_path, index=False)
        result = loadmat(mat_exporter.export_session_to_mat(self.rec.session_dir), simplify_cells=True)
        event_types = list(result["events"]["event_type"])
        legacy_index = event_types.index("arithmetic_response")
        self.assertEqual(result["events"]["participant_answer"][legacy_index], "F")
        self.assertEqual(result["events"]["is_correct"][legacy_index], 1)
        self.assertEqual(result["events"]["reaction_time_ms"][legacy_index], 650)

    def test_legacy_label_names_are_not_reinterpreted_as_new_conditions(self):
        self.rec.stop(export_mat=False)
        metadata = json.loads(self.rec.metadata_path.read_text(encoding="utf-8"))
        metadata["labels"] = {"-1": "unlabeled", "0": "on_task", "1": "off_task"}
        metadata["session"].pop("conditions", None)
        self.rec.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        result = loadmat(mat_exporter.export_session_to_mat(self.rec.session_dir), simplify_cells=True)
        self.assertEqual(list(result["label_names"]), ["on_task", "off_task"])
        self.assertEqual(list(result["condition_types"]), ["legacy_A", "legacy_B"])


if __name__ == "__main__":
    unittest.main()
