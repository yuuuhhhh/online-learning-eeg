"""Stage 4 tests: the 10 s pre-probe epoch and its 4 s / 2 s window index."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scipy.io import loadmat

import mat_exporter
from epoch_builder import (
    ATTENTION_LABELS,
    EPOCH_DURATION_SEC,
    WINDOW_LENGTH_SEC,
    WINDOW_STEP_SEC,
    build_session_epochs,
)
from session_recorder import EEG_CSV_COLUMNS, EVENT_CSV_COLUMNS

SAMPLE_RATE = 250.0
CLOCK_ORIGIN = 1_700_000_000.0


def metadata_document() -> dict:
    return {
        "schema_version": "2.1-dev",
        "run_id": "run-epoch-test",
        "status": "stopped",
        "labels": {"-1": "no_instantaneous_attention_label"},
        "training_rules": {"exclude_video_start_sec": 0.0, "exclude_video_end_sec": 0.0},
        "attention_epochs": {
            "epoch_duration_sec": EPOCH_DURATION_SEC,
            "window_length_sec": WINDOW_LENGTH_SEC,
            "window_step_sec": WINDOW_STEP_SEC,
            "attention_labels": list(ATTENTION_LABELS),
        },
        "session": {
            "subject_id": "sub-epoch",
            "session_id": "ses-001",
            "order": 1,
            "sample_rate_hz": SAMPLE_RATE,
            "channel_names": ["ear_channel_0", "ear_channel_1"],
            "planned_sequence": ["A", "B", "A", "B", "A", "B"],
            "conditions": {"A": {"condition_type": "focused"}, "B": {"condition_type": "bbbd_subtraction"}},
        },
    }


def eeg_rows(total_samples: int, *, missing: set[int] = frozenset(),
             page_spans: tuple[tuple[int, int], ...] = (), segment_break_at: int | None = None) -> list[dict]:
    rows = []
    segment = 0
    for number in range(total_samples):
        if number in missing:
            continue
        if segment_break_at is not None and number >= segment_break_at:
            segment = 1
        on_page = any(low <= number <= high for low, high in page_spans)
        rows.append({
            "received_timestamp": CLOCK_ORIGIN + number / SAMPLE_RATE,
            "received_order": number,
            "device_sample_number": number,
            "sample_index": number % 256,
            "sample_time_sec": number / SAMPLE_RATE,
            "packet_gap_before": 0,
            "block_id": 1,
            "group_id": 1,
            "condition": "" if on_page else "A",
            "condition_type": "" if on_page else "focused",
            "condition_code": -1 if on_page else 0,
            "weak_label": -1,
            "phase": "thought_probe" if on_page else "video",
            "base_valid_for_training": 0 if on_page else 1,
            "channel_0_raw": 10, "channel_1_raw": -10,
            "channel_0_uv": 0.24, "channel_1_uv": -0.24,
            "quality_flag": "ok",
            "is_formal_experiment": 0 if on_page else 1,
            "stream_segment": segment,
            "sample_time_status": "packet_counter",
        })
    return rows


def event_row(event_type: str, **values) -> dict:
    row = {column: "" for column in EVENT_CSV_COLUMNS}
    row.update({
        "event_type": event_type,
        "block_id": 1,
        "condition": "A",
        "condition_label": "A",
        "condition_type": "focused",
        "video_id": "a",
        "subject_id": "sub-epoch",
        "session_id": "ses-001",
        "run_id": "run-epoch-test",
        "alignment_status": "host_receive_nearest",
    })
    row.update(values)
    return row


def probe_events(probe_id: str, onset_sample: int, resume_sample: int, *,
                 response: int = 1, attention: str = "ON", confidence: int = 3,
                 aligned: bool = True) -> list[dict]:
    shared = {"probe_id": probe_id, "probe_schedule_id": "sched-1", "probe_index": 1}
    alignment = {"device_sample_number": onset_sample, "eeg_sample_number": onset_sample} if aligned else {
        "device_sample_number": "", "alignment_status": "outside_sample_tolerance"}
    return [
        event_row("probe_onset", **shared, **alignment, pause_reason="thought_probe",
                  recorder_timestamp=CLOCK_ORIGIN + onset_sample / SAMPLE_RATE,
                  video_time_sec=onset_sample / SAMPLE_RATE),
        event_row("attention_response", **shared, device_sample_number=onset_sample + 10,
                  response=response, response_label=f"option-{response}", probe_attention=attention),
        event_row("confidence_response", **shared, device_sample_number=onset_sample + 20, confidence=confidence),
        event_row("video_resume", **shared, device_sample_number=resume_sample, pause_reason="thought_probe"),
    ]


class EpochCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, eeg: list[dict], events: list[dict], qc: list[dict] | None = None) -> Path:
        with (self.dir / "eeg.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EEG_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(eeg)
        with (self.dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVENT_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(events)
        if qc is not None:
            with (self.dir / "qc.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["qc_timestamp", "window_duration_sec", "status"])
                writer.writeheader()
                writer.writerows(qc)
        (self.dir / "metadata.json").write_text(json.dumps(metadata_document(), ensure_ascii=False), encoding="utf-8")
        return self.dir

    def standard_session(self, **changes):
        """One aligned probe at sample 3000 with its page running to sample 3500."""
        eeg = eeg_rows(changes.pop("total_samples", 5000),
                       missing=changes.pop("missing", frozenset()),
                       page_spans=((3000, 3500),),
                       segment_break_at=changes.pop("segment_break_at", None))
        events = [event_row("block_start", device_sample_number=0),
                  *probe_events("1_1_abcd1234", 3000, 3500, **changes)]
        return self.write(eeg, events)

    def read(self, name: str) -> list[dict]:
        with (self.dir / name).open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def epochs(self):
        return self.read("probe_epochs.csv")

    def windows(self, kind: str | None = None):
        rows = self.read("windows.csv")
        return [row for row in rows if kind is None or row["window_kind"] == kind]


class EpochBoundaryTests(EpochCase):
    def test_epoch_covers_exactly_ten_seconds_of_samples_at_250hz(self):
        self.standard_session()
        build_session_epochs(self.dir)
        epoch = self.epochs()[0]
        self.assertEqual(epoch["status"], "valid")
        self.assertEqual(int(epoch["start_sample"]), 500)
        self.assertEqual(int(epoch["end_sample"]), 2999)
        self.assertEqual(int(epoch["sample_count"]), 2500)
        self.assertEqual(
            (int(epoch["end_sample"]) - int(epoch["start_sample"]) + 1) / SAMPLE_RATE, EPOCH_DURATION_SEC)

    def test_epoch_ends_on_the_last_sample_before_the_real_onset(self):
        self.standard_session()
        build_session_epochs(self.dir)
        epoch = self.epochs()[0]
        onset = int(epoch["probe_onset_sample"])
        self.assertEqual(onset, 3000)
        self.assertEqual(int(epoch["end_sample"]), onset - 1)
        self.assertEqual(int(epoch["start_sample"]), onset - int(EPOCH_DURATION_SEC * SAMPLE_RATE))

    def test_planned_probe_time_is_never_used_as_the_anchor(self):
        eeg = eeg_rows(5000, page_spans=((3000, 3500),))
        events = [event_row("block_start", device_sample_number=0),
                  *probe_events("1_1_abcd1234", 3000, 3500)]
        # A planned time far from the real onset must not move the epoch.
        for row in events:
            if row["event_type"] == "probe_onset":
                row["probe_planned_time_sec"] = 5.0
        self.write(eeg, events)
        build_session_epochs(self.dir)
        epoch = self.epochs()[0]
        self.assertEqual(int(epoch["end_sample"]), 2999)

    def test_epoch_and_windows_exclude_the_probe_page(self):
        self.standard_session()
        build_session_epochs(self.dir)
        epoch = self.epochs()[0]
        self.assertLess(int(epoch["end_sample"]), 3000)
        for window in self.windows("probe_epoch"):
            self.assertLess(int(window["end_sample"]), 3000)
        page_samples = {int(row["device_sample_number"]) for row in self.read("eeg.csv")
                        if row["phase"] == "thought_probe"}
        self.assertTrue(page_samples)
        # No window of any kind may overlap the seconds the probe page was on screen.
        for window in self.windows():
            covered = set(range(int(window["start_sample"]), int(window["end_sample"]) + 1))
            self.assertFalse(covered & page_samples, window["window_id"])

    def test_epoch_timestamps_come_from_the_indexed_eeg_rows(self):
        self.standard_session()
        build_session_epochs(self.dir)
        epoch = self.epochs()[0]
        self.assertAlmostEqual(float(epoch["start_timestamp"]), CLOCK_ORIGIN + 500 / SAMPLE_RATE, places=6)
        self.assertAlmostEqual(float(epoch["end_timestamp"]), CLOCK_ORIGIN + 2999 / SAMPLE_RATE, places=6)
        self.assertAlmostEqual(
            float(epoch["end_timestamp"]) - float(epoch["start_timestamp"]),
            (2500 - 1) / SAMPLE_RATE, places=6)


class WindowIndexTests(EpochCase):
    def test_four_second_windows_step_by_two_seconds(self):
        self.standard_session()
        build_session_epochs(self.dir)
        windows = self.windows("probe_epoch")
        self.assertEqual(len(windows), 4)
        self.assertEqual([int(w["start_sample"]) for w in windows], [500, 1000, 1500, 2000])
        self.assertEqual([int(w["end_sample"]) for w in windows], [1499, 1999, 2499, 2999])
        for window in windows:
            self.assertEqual(int(window["sample_count"]), int(WINDOW_LENGTH_SEC * SAMPLE_RATE))
            span = (int(window["end_sample"]) - int(window["start_sample"]) + 1) / SAMPLE_RATE
            self.assertEqual(span, WINDOW_LENGTH_SEC)
        steps = [int(windows[i + 1]["start_sample"]) - int(windows[i]["start_sample"]) for i in range(3)]
        self.assertEqual(steps, [int(WINDOW_STEP_SEC * SAMPLE_RATE)] * 3)

    def test_seconds_to_probe_counts_back_from_the_onset(self):
        self.standard_session()
        build_session_epochs(self.dir)
        self.assertEqual([float(w["seconds_to_probe"]) for w in self.windows("probe_epoch")], [6.0, 4.0, 2.0, 0.0])

    def test_all_windows_of_one_probe_share_its_probe_id(self):
        self.standard_session()
        build_session_epochs(self.dir)
        windows = self.windows("probe_epoch")
        self.assertEqual({w["probe_id"] for w in windows}, {"1_1_abcd1234"})
        self.assertEqual(len({w["window_id"] for w in windows}), 4)
        self.assertEqual([w["probe_attention"] for w in windows], ["ON"] * 4)

    def test_windows_carry_the_grouping_and_condition_identity(self):
        self.standard_session()
        build_session_epochs(self.dir)
        window = self.windows("probe_epoch")[0]
        self.assertEqual(window["subject_id"], "sub-epoch")
        self.assertEqual(window["session_id"], "ses-001")
        self.assertEqual(window["video_id"], "a")
        self.assertEqual(window["block_id"], "1")
        self.assertEqual(window["block_order"], "1")
        self.assertEqual(window["sequence_order"], "1")
        self.assertEqual(window["condition_label"], "A")
        self.assertEqual(window["condition_type"], "focused")
        self.assertEqual(window["quality_label"], "good")
        self.assertEqual(window["reject_reason"], "")

    def test_windows_only_index_samples_and_never_copy_eeg(self):
        self.standard_session()
        build_session_epochs(self.dir)
        header = self.read("windows.csv")[0].keys()
        self.assertFalse([name for name in header if "channel" in name or name.endswith("_uv")])


class AttentionLabelTests(EpochCase):
    def test_on_off_ambiguous_labels_are_carried_through_unchanged(self):
        for response, attention in ((1, "ON"), (2, "OFF"), (3, "OFF"), (4, "AMBIGUOUS")):
            with self.subTest(response=response), tempfile.TemporaryDirectory() as tmp:
                self.dir = Path(tmp)
                self.standard_session(response=response, attention=attention)
                build_session_epochs(self.dir)
                self.assertEqual(self.epochs()[0]["probe_attention"], attention)
                self.assertEqual(self.epochs()[0]["probe_response"], str(response))
                self.assertEqual({w["probe_attention"] for w in self.windows("probe_epoch")}, {attention})

    def test_ordinary_video_outside_a_probe_epoch_is_labelled_none(self):
        self.standard_session()
        build_session_epochs(self.dir)
        condition_windows = self.windows("condition")
        self.assertTrue(condition_windows)
        self.assertEqual({w["probe_attention"] for w in condition_windows}, {"NONE"})
        self.assertEqual({w["probe_id"] for w in condition_windows}, {""})
        self.assertEqual({w["condition_label"] for w in condition_windows}, {"A"})
        epoch_range = range(500, 3000)
        for window in condition_windows:
            self.assertFalse(set(range(int(window["start_sample"]), int(window["end_sample"]) + 1)) & set(epoch_range))

    def test_only_the_four_documented_labels_are_emitted(self):
        self.standard_session(response=4, attention="AMBIGUOUS")
        build_session_epochs(self.dir)
        self.assertLessEqual({w["probe_attention"] for w in self.windows()}, set(ATTENTION_LABELS))

    def test_condition_windows_never_overlap_the_probe_page(self):
        self.standard_session()
        build_session_epochs(self.dir)
        for window in self.windows("condition"):
            self.assertFalse(3000 <= int(window["end_sample"]) and int(window["start_sample"]) <= 3500)


class RejectionTests(EpochCase):
    def assert_rejected(self, reason: str):
        build_session_epochs(self.dir)
        epoch = self.epochs()[0]
        self.assertEqual(epoch["status"], "rejected")
        self.assertEqual(epoch["reject_reason"], reason)
        self.assertEqual(self.windows("probe_epoch"), [])
        return epoch

    def test_missing_samples_inside_the_ten_seconds_are_rejected(self):
        self.standard_session(missing={1500, 1501})
        self.assert_rejected("incomplete_eeg")

    def test_a_stream_discontinuity_inside_the_epoch_is_rejected(self):
        self.standard_session(segment_break_at=1800)
        self.assert_rejected("incomplete_eeg")

    def test_a_probe_without_ten_seconds_of_history_is_rejected(self):
        eeg = eeg_rows(3000, page_spans=((1000, 1200),))
        events = [*probe_events("1_1_abcd1234", 1000, 1200)]
        self.write(eeg, events)
        self.assert_rejected("insufficient_history")

    def test_an_unaligned_probe_onset_is_rejected_instead_of_guessed(self):
        self.standard_session(aligned=False)
        epoch = self.assert_rejected("probe_onset_not_aligned")
        self.assertEqual(epoch["start_sample"], "")
        self.assertEqual(epoch["end_sample"], "")

    def test_an_epoch_overlapping_an_earlier_probe_page_is_rejected(self):
        eeg = eeg_rows(8000, page_spans=((3000, 3500), (4000, 4400)))
        events = [event_row("block_start", device_sample_number=0),
                  *probe_events("1_1_abcd1234", 3000, 3500),
                  *probe_events("1_2_abcd1234", 4000, 4400, response=2, attention="OFF")]
        self.write(eeg, events)
        build_session_epochs(self.dir)
        first, second = sorted(self.epochs(), key=lambda row: row["probe_id"])
        self.assertEqual(first["status"], "valid")
        self.assertEqual(second["status"], "rejected")
        self.assertEqual(second["reject_reason"], "overlaps_probe_page")

    def test_an_incomplete_probe_chain_is_rejected(self):
        eeg = eeg_rows(5000, page_spans=((3000, 3500),))
        events = probe_events("1_1_abcd1234", 3000, 3500)
        self.write(eeg, [row for row in events if row["event_type"] != "confidence_response"])
        self.assert_rejected("incomplete_probe_chain")

    def test_a_cancelled_probe_is_rejected(self):
        eeg = eeg_rows(5000, page_spans=((3000, 3500),))
        events = [*probe_events("1_1_abcd1234", 3000, 3500),
                  event_row("probe_cancelled", probe_id="1_1_abcd1234", device_sample_number=3400)]
        self.write(eeg, events)
        self.assert_rejected("incomplete_probe_chain")


class WindowQualityTests(EpochCase):
    def test_a_window_overlapping_a_bad_qc_report_is_not_a_training_window(self):
        self.standard_session()
        # One bad 10 s QC report covering the host time of the first window.
        self.write_qc([{"qc_timestamp": CLOCK_ORIGIN + 1499 / SAMPLE_RATE,
                        "window_duration_sec": 10.0, "status": "bad"}])
        build_session_epochs(self.dir)
        windows = self.windows("probe_epoch")
        self.assertEqual(windows[0]["quality_label"], "reject")
        self.assertIn("qc_reject", windows[0]["reject_reason"])
        self.assertEqual(windows[0]["qc_status"], "bad")
        self.assertEqual(windows[-1]["quality_label"], "good")

    def test_a_window_with_only_high_50_hz_stays_usable(self):
        self.standard_session()
        self.write_qc([{"qc_timestamp": CLOCK_ORIGIN + 1499 / SAMPLE_RATE, "window_duration_sec": 10.0,
                        "status": "warning", "needs_notch": "True"}])
        build_session_epochs(self.dir)
        flagged = self.windows("probe_epoch")[0]
        self.assertEqual(flagged["quality_label"], "warning")
        self.assertEqual(flagged["reject_reason"], "")
        self.assertEqual(flagged["qc_status"], "warning")
        self.assertEqual(flagged["warning_50hz"], "1")
        self.assertEqual(flagged["needs_notch"], "1")
        clean = self.windows("probe_epoch")[-1]
        self.assertEqual(clean["quality_label"], "good")
        self.assertEqual(clean["needs_notch"], "0")

    def test_high_50_hz_does_not_rescue_a_window_with_a_real_defect(self):
        eeg = eeg_rows(5000, page_spans=((3000, 3500),))
        for row in eeg:
            if row["device_sample_number"] == 700:
                row["quality_flag"] = "adc_saturation"
        self.write(eeg, [*probe_events("1_1_abcd1234", 3000, 3500)])
        self.write_qc([{"qc_timestamp": CLOCK_ORIGIN + 1499 / SAMPLE_RATE, "window_duration_sec": 10.0,
                        "status": "warning", "needs_notch": "True"}])
        build_session_epochs(self.dir)
        window = self.windows("probe_epoch")[0]
        self.assertEqual(window["quality_label"], "reject")
        self.assertIn("eeg_quality_flag", window["reject_reason"])
        self.assertEqual(window["needs_notch"], "1")

    def test_a_session_recorded_before_the_notch_flag_still_indexes(self):
        self.standard_session()
        # Old qc.csv files have no needs_notch column at all.
        self.write_qc([{"qc_timestamp": CLOCK_ORIGIN + 1499 / SAMPLE_RATE,
                        "window_duration_sec": 10.0, "status": "good"}])
        build_session_epochs(self.dir)
        self.assertEqual({row["needs_notch"] for row in self.windows()}, {"0"})
        self.assertEqual({row["quality_label"] for row in self.windows("probe_epoch")}, {"good"})

    def test_a_window_touching_a_non_formal_phase_is_not_a_training_window(self):
        eeg = eeg_rows(5000, page_spans=((3000, 3500),))
        for row in eeg:
            if 600 <= row["device_sample_number"] <= 900:
                row["phase"] = "paused"
                row["is_formal_experiment"] = 0
        self.write(eeg, [*probe_events("1_1_abcd1234", 3000, 3500)])
        build_session_epochs(self.dir)
        windows = self.windows("probe_epoch")
        self.assertEqual(windows[0]["quality_label"], "reject")
        self.assertIn("non_formal_phase", windows[0]["reject_reason"])
        self.assertEqual([w["quality_label"] for w in windows[1:]], ["good"] * 3)

    def test_a_window_with_an_eeg_quality_flag_is_not_a_training_window(self):
        eeg = eeg_rows(5000, page_spans=((3000, 3500),))
        for row in eeg:
            if row["device_sample_number"] == 2600:
                row["quality_flag"] = "adc_saturation"
        self.write(eeg, [*probe_events("1_1_abcd1234", 3000, 3500)])
        build_session_epochs(self.dir)
        flagged = [w for w in self.windows("probe_epoch")
                   if int(w["start_sample"]) <= 2600 <= int(w["end_sample"])]
        self.assertTrue(flagged)
        self.assertTrue(all(w["quality_label"] == "reject" for w in flagged))
        self.assertTrue(all("eeg_quality_flag" in w["reject_reason"] for w in flagged))

    def write_qc(self, rows):
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (self.dir / "qc.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


class MatExportTests(EpochCase):
    def test_mat_carries_the_epoch_and_window_indices(self):
        self.standard_session(response=2, attention="OFF", confidence=4)
        result = loadmat(mat_exporter.export_session_to_mat(self.dir), simplify_cells=True)
        self.assertIn("probe_epochs", result)
        self.assertIn("windows", result)
        epochs = result["probe_epochs"]
        self.assertEqual(epochs["probe_id"], "1_1_abcd1234")
        self.assertEqual(epochs["probe_attention"], "OFF")
        self.assertEqual(epochs["probe_confidence"], 4)
        self.assertEqual(epochs["start_sample"], 500)
        self.assertEqual(epochs["end_sample"], 2999)
        self.assertEqual(epochs["status"], "valid")
        windows = result["windows"]
        probe_rows = [index for index, kind in enumerate(windows["window_kind"]) if kind == "probe_epoch"]
        self.assertEqual(len(probe_rows), 4)
        self.assertEqual([windows["start_sample"][index] for index in probe_rows], [500, 1000, 1500, 2000])
        self.assertEqual([windows["seconds_to_probe"][index] for index in probe_rows], [6.0, 4.0, 2.0, 0.0])
        self.assertEqual({windows["quality_label"][index] for index in probe_rows}, {"good"})
        self.assertIn("NONE", set(windows["probe_attention"]))
        self.assertIn("probe_epochs", result["csv_text"])
        self.assertIn("windows", result["csv_text"])

    def test_a_session_without_probes_still_exports_without_the_new_keys(self):
        eeg = eeg_rows(1000)
        self.write(eeg, [event_row("block_start", device_sample_number=0),
                         event_row("video_play", device_sample_number=1)])
        result = loadmat(mat_exporter.export_session_to_mat(self.dir), simplify_cells=True)
        self.assertNotIn("probe_epochs", result)
        self.assertNotIn("windows", result)
        self.assertFalse((self.dir / "windows.csv").exists())
        self.assertIn("block_start", list(result["events"]["event_type"]))


class OnsetAlignmentTests(unittest.TestCase):
    def test_an_event_on_a_tied_burst_anchors_to_the_newest_sample(self):
        """A BLE burst shares one receive time; the epoch must not start a burst early."""
        from unittest.mock import patch

        from session_recorder import ExperimentRecorder, SessionConfig

        def ble_packet(index: int) -> bytes:
            return (b"\xa0" + bytes([index & 255]) + (5).to_bytes(3, "big", signed=True)
                    + (-5).to_bytes(3, "big", signed=True) + bytes(24) + b"\xc0")

        with tempfile.TemporaryDirectory() as tmp:
            recorder = ExperimentRecorder(SessionConfig(subject_id="sub-tie"), Path(tmp))
            recorder.start()
            frozen = recorder.clock_time() + 1.0
            with patch.object(recorder, "clock_time", return_value=frozen):
                for index in range(50):
                    recorder.record_packet(ble_packet(index), ble_packet(index))
                self.assertEqual(len(set(recorder._sample_timestamps)), 1)
                recorder.log_event("manual_sync_mark")
                recorder._flush_files()
            with recorder.events_path.open(encoding="utf-8") as handle:
                mark = [row for row in csv.DictReader(handle) if row["event_type"] == "manual_sync_mark"][0]
            self.assertEqual(mark["alignment_status"], "host_receive_nearest")
            self.assertEqual(int(mark["device_sample_number"]), 49)
            recorder.browser_clients.clear()
            recorder.stop(export_mat=False)


class RecorderIntegrationTests(unittest.TestCase):
    def test_a_real_recorded_probe_produces_a_valid_indexed_epoch(self):
        """The whole chain: BLE packets, aligned probe events, epoch, windows, MAT."""
        from unittest.mock import patch

        from session_recorder import ExperimentRecorder, SessionConfig

        def ble_packet(index: int) -> bytes:
            return (b"\xa0" + bytes([index & 255]) + (15).to_bytes(3, "big", signed=True)
                    + (-15).to_bytes(3, "big", signed=True) + bytes(24) + b"\xc0")

        with tempfile.TemporaryDirectory() as tmp:
            recorder = ExperimentRecorder(SessionConfig(subject_id="sub-live"), Path(tmp))
            recorder.start()
            recorder.baseline_complete = recorder.baseline_passed = True
            with patch.object(recorder, "get_qc_status", return_value={"status": "good"}):
                recorder.begin_block(1, "a")
                recorder.start_video()
            for index in range(3000):
                recorder.record_packet(ble_packet(index), ble_packet(index))

            probe_id = "1_1_live0001"
            step = [0]

            def browser(event_type, **values):
                step[0] += 1
                recorder.handle_browser_event({
                    "subject_id": "sub-live", "session_id": "ses-001",
                    "recorder_run_id": recorder.run_id, "client_id": "browser-live",
                    "client_event_id": f"live-{step[0]}", "event_type": event_type,
                    "event_timestamp": recorder.clock_time(), "clock_offset_sec": 0.0,
                    "block_id": "1", "condition": "A", "topic_key": "a",
                    "probe_id": probe_id, "probe_index": 1, "probe_schedule_id": "sched-live",
                    **values,
                })

            browser("probe_onset", pause_reason="thought_probe")
            browser("attention_response", response=1, probe_attention="ON", response_time_ms=900)
            browser("confidence_response", confidence=3, response_time_ms=500)
            browser("video_resume", pause_reason="thought_probe")
            for index in range(3000, 3100):
                recorder.record_packet(ble_packet(index), ble_packet(index))

            recorder.browser_clients.clear()
            recorder.stop(export_mat=True)

            with (recorder.session_dir / "probe_epochs.csv").open(encoding="utf-8") as handle:
                epoch = list(csv.DictReader(handle))[0]
            self.assertEqual(epoch["status"], "valid", epoch["reject_reason"])
            self.assertEqual(epoch["probe_id"], probe_id)
            self.assertEqual(epoch["probe_attention"], "ON")
            self.assertEqual(int(epoch["sample_count"]), 2500)
            self.assertEqual(int(epoch["end_sample"]), int(epoch["probe_onset_sample"]) - 1)
            self.assertEqual(int(epoch["window_count"]), 4)

            with (recorder.session_dir / "windows.csv").open(encoding="utf-8") as handle:
                windows = [row for row in csv.DictReader(handle) if row["window_kind"] == "probe_epoch"]
            self.assertEqual(len(windows), 4)
            self.assertEqual({row["probe_id"] for row in windows}, {probe_id})
            self.assertEqual([float(row["seconds_to_probe"]) for row in windows], [6.0, 4.0, 2.0, 0.0])
            result = loadmat(recorder.mat_path, simplify_cells=True)
            self.assertEqual(result["probe_epochs"]["probe_id"], probe_id)

    def test_stopping_a_session_writes_both_index_files(self):
        from session_recorder import ExperimentRecorder, SessionConfig

        with tempfile.TemporaryDirectory() as tmp:
            recorder = ExperimentRecorder(SessionConfig(subject_id="sub-index"), Path(tmp))
            recorder.start()
            recorder.browser_clients.clear()
            recorder.stop(export_mat=True)
            self.assertTrue((recorder.session_dir / "probe_epochs.csv").exists())
            self.assertTrue((recorder.session_dir / "windows.csv").exists())
            self.assertEqual(recorder.epoch_status, "complete")
            metadata = json.loads(recorder.metadata_path.read_text(encoding="utf-8"))
            rules = metadata["attention_epochs"]
            self.assertEqual(rules["epoch_duration_sec"], 10.0)
            self.assertEqual(rules["window_length_sec"], 4.0)
            self.assertEqual(rules["window_step_sec"], 2.0)
            self.assertEqual(rules["attention_labels"], ["ON", "OFF", "AMBIGUOUS", "NONE"])
            self.assertEqual(metadata["files"]["windows_csv"], "windows.csv")


if __name__ == "__main__":
    unittest.main()
