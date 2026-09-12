"""Locate the 10 s EEG epoch before each thought probe and index its windows.

This module only derives sample indices and labels. The EEG itself stays in
eeg.csv / eeg_raw.bin exactly once; nothing here copies or filters a waveform.
The acquisition program intentionally does not measure 50 Hz line-noise power.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

EPOCH_DURATION_SEC = 10.0
WINDOW_LENGTH_SEC = 4.0
WINDOW_STEP_SEC = 2.0

# The probe answer is a self report about the seconds before the page appeared.
# NONE marks ordinary EEG that no probe ever asked about.
ATTENTION_LABELS = ("ON", "OFF", "AMBIGUOUS", "NONE")

PROBE_EPOCH_COLUMNS = [
    "probe_id", "subject_id", "session_id", "run_id",
    "block_id", "block_order", "sequence_order", "video_id",
    "condition_label", "condition_type",
    "probe_attention", "probe_response", "probe_response_label", "probe_confidence",
    "probe_onset_sample", "probe_onset_timestamp", "probe_onset_video_time_sec",
    "epoch_duration_sec", "start_sample", "end_sample",
    "start_timestamp", "end_timestamp", "sample_count",
    "status", "reject_reason", "window_count",
]

WINDOW_COLUMNS = [
    "window_id", "probe_id", "window_kind", "window_index",
    "subject_id", "session_id", "run_id", "video_id",
    "block_id", "block_order", "sequence_order",
    "condition_label", "condition_type",
    "probe_attention", "probe_response", "probe_confidence", "seconds_to_probe",
    "mental_effort", "course_attention_rating", "video_interest", "video_difficulty",
    "start_number", "reported_final_number", "subtraction_compliance",
    "self_caught_nearby", "dataset_membership", "training_eligible",
    "start_sample", "end_sample", "start_timestamp", "end_timestamp", "sample_count",
    "quality_label", "reject_reason", "qc_status",
]

WINDOW_QUALITY_LABELS = ("GOOD", "USABLE", "REJECT")

_EEG_COLUMNS = (
    "device_sample_number", "received_timestamp", "stream_segment",
    "is_formal_experiment", "quality_flag", "block_id", "phase",
)


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> Path:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _numeric(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sample_number(value: Any) -> Optional[int]:
    """Sample numbers are counts, so keep them out of floating point."""
    number = _numeric(value)
    return None if number is None else int(round(number))


class SampleIndex:
    """Map device sample numbers onto EEG rows without assuming row == sample."""

    def __init__(self, eeg: pd.DataFrame):
        self.numbers = pd.to_numeric(eeg["device_sample_number"], errors="coerce").to_numpy(dtype=np.float64)
        self.timestamps = pd.to_numeric(eeg["received_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        self.segments = pd.to_numeric(eeg.get("stream_segment", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        formal = pd.to_numeric(eeg.get("is_formal_experiment", 0), errors="coerce").fillna(0)
        self.formal = formal.to_numpy(dtype=np.int64)
        self.quality = eeg.get("quality_flag", pd.Series([""] * len(eeg))).fillna("").astype(str).to_numpy()
        self.blocks = pd.to_numeric(eeg.get("block_id", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        self.phases = eeg.get("phase", pd.Series([""] * len(eeg))).fillna("").astype(str).to_numpy()
        # device_sample_number never decreases, so a sorted search is exact here.
        self.usable = np.isfinite(self.numbers)

    def __len__(self) -> int:
        return int(self.numbers.size)

    def slice_for(self, start: float, end: float) -> Optional[slice]:
        """Rows covering sample numbers start..end inclusive, or None if incomplete."""
        expected = int(round(end - start)) + 1
        if expected <= 0 or not len(self):
            return None
        low = int(np.searchsorted(self.numbers, start, side="left"))
        high = int(np.searchsorted(self.numbers, end, side="right"))
        if high - low != expected:
            return None
        span = self.numbers[low:high]
        if span[0] != start or span[-1] != end or not np.all(np.diff(span) == 1):
            return None
        if self.segments[low] != self.segments[high - 1]:
            return None
        return slice(low, high)


def _probe_page_intervals(events: pd.DataFrame) -> list[tuple[float, float]]:
    """Sample ranges during which a probe page was on screen."""
    intervals: list[tuple[float, float]] = []
    onsets = events[events["event_type"] == "probe_onset"]
    closing = {"video_resume", "confidence_response", "probe_cancelled", "block_end", "video_ended"}
    for _, onset in onsets.iterrows():
        start = _sample_number(onset.get("device_sample_number"))
        if start is None:
            continue
        probe_id = str(onset.get("probe_id", ""))
        later = events[
            (events["event_type"].isin(closing))
            & (events["probe_id"].astype(str) == probe_id)
        ]
        ends = [value for value in (_sample_number(row.get("device_sample_number")) for _, row in later.iterrows())
                if value is not None and value >= start]
        intervals.append((start, max(ends) if ends else start))
    return intervals


def _qc_intervals(session_dir: Path) -> list[tuple[float, float, str, tuple[str, ...]]]:
    qc_path = session_dir / "qc.csv"
    if not qc_path.exists() or qc_path.stat().st_size == 0:
        return []
    try:
        qc = pd.read_csv(qc_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return []
    intervals = []
    for _, row in qc.iterrows():
        stamp = _numeric(row.get("qc_timestamp"))
        duration = _numeric(row.get("window_duration_sec")) or 0.0
        if stamp is None:
            continue
        reasons: list[str] = []
        if any(str(row.get(f"channel_{channel}_invalid_flatline", "")).lower() in {"true", "1"} for channel in range(2)):
            reasons.append("flat")
        if any((_numeric(row.get(f"channel_{channel}_saturation_rate_pct")) or 0) > 0 for channel in range(2)):
            reasons.append("saturation")
        if (_numeric(row.get("packet_loss_rate_pct")) or 0) > 0:
            reasons.append("missing")
        messages = str(row.get("messages", ""))
        if "极端" in messages:
            reasons.append("extreme_amplitude")
        if "高频污染" in messages:
            reasons.append("high_frequency_contamination")
        intervals.append((stamp - duration, stamp, str(row.get("status", "")), tuple(dict.fromkeys(reasons))))
    return intervals


def _overlaps(start: float, end: float, ranges: list[tuple[float, float]]) -> bool:
    return any(start <= other_end and other_start <= end for other_start, other_end in ranges)


def _overlapping_qc(start_ts: float, end_ts: float,
                    intervals: list[tuple[float, float, str, tuple[str, ...]]]) -> tuple[str, list[str]]:
    """Return the worst QC status and all direct rejection reasons touching a window."""
    ranking = {"good": 0, "waiting": 1, "warning": 2, "bad": 3}
    worst = ""
    reasons: list[str] = []
    for other_start, other_end, status, interval_reasons in intervals:
        if start_ts <= other_end and other_start <= end_ts:
            reasons.extend(interval_reasons)
            if ranking.get(status, 0) >= ranking.get(worst, -1):
                worst = status
    return worst, list(dict.fromkeys(reasons))


def _collect_probes(events: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Join every probe's onset, attention answer and confidence answer."""
    probes: dict[str, dict[str, Any]] = {}
    interesting = {"probe_onset", "attention_response", "confidence_response", "probe_cancelled"}
    for _, row in events[events["event_type"].isin(interesting)].iterrows():
        probe_id = str(row.get("probe_id", "")).strip()
        if not probe_id:
            continue
        probe = probes.setdefault(probe_id, {"probe_id": probe_id})
        probe[str(row["event_type"])] = row
    return probes


def _event_sample_ranges(events: pd.DataFrame, event_type: str, before_sec: float, after_sec: float,
                         sample_rate: float) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    selected = events[events["event_type"] == event_type]
    for _, row in selected.iterrows():
        sample = _sample_number(row.get("device_sample_number"))
        if sample is None:
            continue
        ranges.append((sample - before_sec * sample_rate, sample + after_sec * sample_rate))
    return ranges


def _row_defined_ranges(events: pd.DataFrame, event_types: set[str], sample_rate: float) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for _, row in events[events["event_type"].isin(event_types)].iterrows():
        sample = _sample_number(row.get("device_sample_number"))
        if sample is None:
            continue
        before = _numeric(row.get("exclude_before_sec")) or 0.0
        after = _numeric(row.get("exclude_after_sec")) or 0.0
        ranges.append((sample - before * sample_rate, sample + after * sample_rate))
    return ranges


def _block_ratings(session_dir: Path) -> dict[int, dict[str, Any]]:
    path = session_dir / "block_ratings.csv"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return {}
    result: dict[int, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        block = int(_numeric(row.get("block_id")) or 0)
        if block:
            result[block] = row.to_dict()
    return result


def build_session_epochs(session_dir: Path | str) -> tuple[Path, Path]:
    """Write probe_epochs.csv and windows.csv for one recorded session."""
    session_dir = Path(session_dir).resolve()
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    session = metadata.get("session", {})
    sample_rate = float(session.get("sample_rate_hz", 250.0))
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("metadata.json must provide a positive sample_rate_hz")

    epoch_samples = int(round(EPOCH_DURATION_SEC * sample_rate))
    window_samples = int(round(WINDOW_LENGTH_SEC * sample_rate))
    step_samples = int(round(WINDOW_STEP_SEC * sample_rate))

    eeg = pd.read_csv(session_dir / "eeg.csv", keep_default_na=False, low_memory=False,
                      encoding="utf-8-sig", dtype=str)
    for column in _EEG_COLUMNS:
        if column not in eeg:
            eeg[column] = ""
    events = pd.read_csv(session_dir / "events.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    for column in ("event_type", "probe_id", "device_sample_number"):
        if column not in events:
            events[column] = ""

    index = SampleIndex(eeg)
    page_ranges = _probe_page_intervals(events)
    qc_ranges = _qc_intervals(session_dir)
    planned = list(session.get("planned_sequence", []))
    planned_block_rows = list(session.get("planned_blocks", []))
    identity = {
        "subject_id": session.get("subject_id", ""),
        "session_id": session.get("session_id", ""),
        "run_id": metadata.get("run_id", ""),
        "sequence_order": session.get("counterbalance_group", session.get("order", "")),
    }

    flow = metadata.get("protocol_flow", {})
    motor_before = float(flow.get("self_caught_motor_mask_before_sec", 2.0))
    motor_after = float(flow.get("self_caught_motor_mask_after_sec", 2.0))
    sensitivity_before = float(flow.get("self_caught_sensitivity_before_sec", 10.0))
    transition_after = float(flow.get("probe_transition_mask_after_sec", 2.0))
    motor_ranges = _event_sample_ranges(events, "self_caught", motor_before, motor_after, sample_rate)
    motion_ranges = _row_defined_ranges(events, {"artifact", "manual_artifact"}, sample_rate)
    self_caught_ranges = _event_sample_ranges(events, "self_caught", sensitivity_before, 0.0, sample_rate)
    transition_events = events[(events["event_type"] == "video_resume") & (events.get("pause_reason", "") == "thought_probe")]
    transition_ranges: list[tuple[float, float]] = []
    for _, row in transition_events.iterrows():
        sample = _sample_number(row.get("device_sample_number"))
        if sample is not None:
            transition_ranges.append((sample, sample + transition_after * sample_rate))
    ratings_by_block = _block_ratings(session_dir)

    epoch_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    claimed: list[tuple[float, float]] = []

    def quality_for(rows: slice, start_ts: float, end_ts: float) -> dict[str, Any]:
        reasons: list[str] = []
        if not np.all(index.formal[rows] == 1):
            reasons.append("non_formal_phase")
        flags = ";".join(index.quality[rows])
        if "adc_saturation" in flags:
            reasons.append("saturation")
        if any(flag in flags for flag in ("packet_gap", "stream_discontinuity", "duplicate_index")):
            reasons.append("missing")
        if any(flag in flags for flag in ("artifact", "motion")):
            reasons.append("motion")
        window_start = float(index.numbers[rows.start])
        window_end = float(index.numbers[rows.stop - 1])
        if _overlaps(window_start, window_end, motor_ranges):
            reasons.append("motor_event")
        if _overlaps(window_start, window_end, motion_ranges):
            reasons.append("motion")
        if _overlaps(window_start, window_end, transition_ranges):
            reasons.append("probe_transition")
        qc_status, qc_reasons = _overlapping_qc(start_ts, end_ts, qc_ranges)
        reasons.extend(qc_reasons)
        if qc_status == "bad":
            reasons.append("qc_reject")
        reasons = list(dict.fromkeys(reasons))
        label = "REJECT" if reasons else "USABLE" if qc_status in {"warning", "waiting", ""} else "GOOD"
        return {
            "quality_label": label,
            "reject_reason": ";".join(reasons),
            "qc_status": qc_status,
            "self_caught_nearby": int(_overlaps(window_start, window_end, self_caught_ranges)),
            "training_eligible": int(label != "REJECT"),
        }

    for probe_id, probe in sorted(_collect_probes(events).items()):
        onset = probe.get("probe_onset")
        attention = probe.get("attention_response")
        confidence = probe.get("confidence_response")
        block_id = int(_numeric(onset.get("block_id")) or 0) if onset is not None else 0
        record = {
            **identity,
            "probe_id": probe_id,
            "block_id": block_id or "",
            "block_order": block_id or "",
            "video_id": (onset.get("video_id", "") if onset is not None else ""),
            "condition_label": (onset.get("condition_label", "") if onset is not None else ""),
            "condition_type": (onset.get("condition_type", "") if onset is not None else ""),
            "probe_attention": (attention.get("probe_attention", "") if attention is not None else ""),
            "probe_response": (attention.get("response", "") if attention is not None else ""),
            "probe_response_label": (attention.get("response_label", "") if attention is not None else ""),
            "probe_confidence": (confidence.get("confidence", "") if confidence is not None else ""),
            "epoch_duration_sec": EPOCH_DURATION_SEC,
            "status": "rejected",
            "reject_reason": "",
            "window_count": 0,
        }
        if planned and 1 <= block_id <= len(planned):
            record.setdefault("condition_label", planned[block_id - 1])

        if onset is None or attention is None or confidence is None or "probe_cancelled" in probe:
            record["reject_reason"] = "incomplete_probe_chain"
            epoch_rows.append(record)
            continue

        onset_sample = _sample_number(onset.get("device_sample_number"))
        record["probe_onset_timestamp"] = onset.get("recorder_timestamp", "")
        record["probe_onset_video_time_sec"] = onset.get("video_time_sec", "")
        if onset_sample is None or str(onset.get("alignment_status", "")) != "host_receive_nearest":
            record["reject_reason"] = "probe_onset_not_aligned"
            epoch_rows.append(record)
            continue
        record["probe_onset_sample"] = onset_sample

        # End the epoch on the last sample strictly before the probe page appeared.
        end_sample = onset_sample - 1
        start_sample = end_sample - (epoch_samples - 1)
        record.update(start_sample=start_sample, end_sample=end_sample, sample_count=epoch_samples)
        if start_sample < 0:
            record["reject_reason"] = "insufficient_history"
            epoch_rows.append(record)
            continue
        rows = index.slice_for(start_sample, end_sample)
        if rows is None:
            record["reject_reason"] = "incomplete_eeg"
            epoch_rows.append(record)
            continue
        if _overlaps(start_sample, end_sample, page_ranges):
            record["reject_reason"] = "overlaps_probe_page"
            epoch_rows.append(record)
            continue

        record["start_timestamp"] = index.timestamps[rows.start]
        record["end_timestamp"] = index.timestamps[rows.stop - 1]
        record["status"] = "valid"
        claimed.append((start_sample, end_sample))

        window_index = 0
        offset = 0
        while offset + window_samples <= epoch_samples:
            window_start = start_sample + offset
            window_end = window_start + window_samples - 1
            window_rows_slice = index.slice_for(window_start, window_end)
            start_ts = index.timestamps[window_rows_slice.start]
            end_ts = index.timestamps[window_rows_slice.stop - 1]
            rating = ratings_by_block.get(block_id, {})
            start_number = session.get("b_start_numbers", {}).get(str(block_id), session.get("b_start_numbers", {}).get(block_id, {}))
            window_rows.append({
                **{key: record[key] for key in (
                    "subject_id", "session_id", "run_id", "video_id", "block_id", "block_order",
                    "sequence_order", "condition_label", "condition_type",
                    "probe_attention", "probe_response", "probe_confidence",
                )},
                "window_id": f"{probe_id}#w{window_index + 1}",
                "probe_id": probe_id,
                "window_kind": "probe_epoch",
                "dataset_membership": "Dataset B",
                "window_index": window_index + 1,
                # Seconds between the end of this window and the probe onset.
                "seconds_to_probe": round((onset_sample - (window_end + 1)) / sample_rate, 3),
                "start_sample": window_start, "end_sample": window_end,
                "start_timestamp": start_ts, "end_timestamp": end_ts,
                "sample_count": window_samples,
                "mental_effort": rating.get("mental_effort", ""),
                "course_attention_rating": rating.get("course_attention_rating", ""),
                "video_interest": rating.get("video_interest", ""),
                "video_difficulty": rating.get("video_difficulty", ""),
                "start_number": start_number.get("start_number", "") if isinstance(start_number, dict) else "",
                "reported_final_number": rating.get("reported_final_number", ""),
                "subtraction_compliance": rating.get("subtraction_compliance", ""),
                **quality_for(window_rows_slice, start_ts, end_ts),
            })
            window_index += 1
            offset += step_samples
        record["window_count"] = window_index
        epoch_rows.append(record)

    window_rows.extend(_condition_windows(
        index, identity, planned, planned_block_rows, session, ratings_by_block, qc_ranges, page_ranges, claimed,
        sample_rate, window_samples, step_samples, quality_for,
    ))

    return (
        _write_csv(session_dir / "probe_epochs.csv", PROBE_EPOCH_COLUMNS, epoch_rows),
        _write_csv(session_dir / "windows.csv", WINDOW_COLUMNS, window_rows),
    )


def _condition_windows(index, identity, planned, planned_block_rows, session, ratings_by_block, qc_ranges, page_ranges, claimed,
                       sample_rate, window_samples, step_samples, quality_for) -> list[dict[str, Any]]:
    """Tile the formal video regions that no probe asked about, labelled NONE."""
    rows: list[dict[str, Any]] = []
    conditions = session.get("conditions", {})
    formal = (index.formal == 1) & index.usable
    if not formal.any():
        return rows
    for block_id in sorted({int(value) for value in index.blocks[formal] if value > 0}):
        selected = np.flatnonzero(formal & (index.blocks == block_id))
        if not selected.size:
            continue
        numbers = index.numbers[selected]
        segments = index.segments[selected]
        breaks = np.flatnonzero((np.diff(numbers) != 1) | (np.diff(segments) != 0)) + 1
        condition = planned[block_id - 1] if planned and 1 <= block_id <= len(planned) else ""
        condition_type = conditions.get(condition, {}).get("condition_type", "")
        planned_block = next((item for item in planned_block_rows if int(item.get("block_id", 0)) == block_id), {})
        video_id = planned_block.get("video_id", "")
        rating = ratings_by_block.get(block_id, {})
        start_number = session.get("b_start_numbers", {}).get(str(block_id), session.get("b_start_numbers", {}).get(block_id, {}))
        counter = 0
        for group in np.split(np.arange(selected.size), breaks):
            if group.size < window_samples:
                continue
            first = int(numbers[group[0]])
            offset = 0
            while offset + window_samples <= group.size:
                window_start = first + offset
                window_end = window_start + window_samples - 1
                if _overlaps(window_start, window_end, claimed) or _overlaps(window_start, window_end, page_ranges):
                    offset += step_samples
                    continue
                window_slice = index.slice_for(window_start, window_end)
                if window_slice is None:
                    offset += step_samples
                    continue
                start_ts = index.timestamps[window_slice.start]
                end_ts = index.timestamps[window_slice.stop - 1]
                counter += 1
                rows.append({
                    **identity,
                    "window_id": f"b{block_id}#c{counter}",
                    "probe_id": "",
                    "window_kind": "condition",
                    "dataset_membership": "Dataset A",
                    "window_index": counter,
                    "video_id": video_id,
                    "block_id": block_id, "block_order": block_id,
                    "condition_label": condition, "condition_type": condition_type,
                    # No probe asked about this stretch, so there is no attention report.
                    "probe_attention": "NONE",
                    "probe_response": "", "probe_confidence": "", "seconds_to_probe": "",
                    "start_sample": window_start, "end_sample": window_end,
                    "start_timestamp": start_ts, "end_timestamp": end_ts,
                    "sample_count": window_samples,
                    "mental_effort": rating.get("mental_effort", ""),
                    "course_attention_rating": rating.get("course_attention_rating", ""),
                    "video_interest": rating.get("video_interest", ""),
                    "video_difficulty": rating.get("video_difficulty", ""),
                    "start_number": start_number.get("start_number", "") if isinstance(start_number, dict) else "",
                    "reported_final_number": rating.get("reported_final_number", ""),
                    "subtraction_compliance": rating.get("subtraction_compliance", ""),
                    **quality_for(window_slice, start_ts, end_ts),
                })
                offset += step_samples
    return rows


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python epoch_builder.py <session_directory>")
    for output in build_session_epochs(sys.argv[1]):
        print(output)
