"""Build normalized behavioral tables and the end-of-session integrity report."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


PROBE_COLUMNS = [
    "probe_id", "subject_id", "session_id", "run_id", "counterbalance_group",
    "block_id", "block_order", "video_id", "condition_label",
    "probe_index", "probe_planned_time_sec", "probe_attention", "probe_response",
    "probe_response_label", "probe_confidence", "onset_timestamp", "onset_sample",
    "onset_video_time_sec", "attention_response_timestamp", "attention_response_time_ms",
    "confidence_response_timestamp", "confidence_response_time_ms", "resume_timestamp",
    "resume_sample", "complete",
]

RATING_COLUMNS = [
    "subject_id", "session_id", "run_id", "counterbalance_group", "block_id", "block_order",
    "video_id", "condition_label", "condition_type", "course_attention_rating", "mental_effort",
    "video_interest", "video_difficulty", "subtraction_compliance",
    "subtraction_compliance_low_pct", "subtraction_compliance_high_pct",
    "subtraction_difficulty", "reported_final_number", "duration_sec", "recorder_timestamp",
    "device_sample_number",
]

QUIZ_COLUMNS = [
    "subject_id", "session_id", "run_id", "counterbalance_group", "block_id", "block_order",
    "video_id", "condition_label", "question_id", "question_text", "response_index",
    "participant_answer", "response_text", "correct_answer", "is_correct", "reaction_time_ms",
    "recorder_timestamp", "device_sample_number",
]


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> Path:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict(orient="records") if not frame.empty else []


def _event_name(frame: pd.DataFrame) -> pd.Series:
    if "event_name" in frame:
        return frame["event_name"].astype(str)
    return frame.get("event_type", pd.Series([""] * len(frame))).astype(str)


def build_normalized_tables(session_dir: Path | str) -> dict[str, Path]:
    session_dir = Path(session_dir).resolve()
    events = _read_csv(session_dir / "events.csv")
    names = _event_name(events)
    probe_rows: list[dict[str, Any]] = []
    if not events.empty and "probe_id" in events:
        for probe_id in sorted(value for value in events["probe_id"].unique() if value):
            part = events[events["probe_id"] == probe_id]
            by_name = {name: part[names.loc[part.index] == name].iloc[-1] for name in set(names.loc[part.index])}
            onset = by_name.get("probe_onset")
            attention = by_name.get("attention_response")
            confidence = by_name.get("confidence_response")
            resume = by_name.get("video_resume")
            source = onset if onset is not None else part.iloc[0]
            probe_rows.append({
                "probe_id": probe_id,
                **{key: source.get(key, "") for key in (
                    "subject_id", "session_id", "run_id", "counterbalance_group", "block_id",
                    "block_order", "video_id", "condition_label", "probe_index", "probe_planned_time_sec",
                )},
                "probe_attention": "" if attention is None else attention.get("probe_attention", ""),
                "probe_response": "" if attention is None else attention.get("response", ""),
                "probe_response_label": "" if attention is None else attention.get("response_label", ""),
                "probe_confidence": "" if confidence is None else confidence.get("confidence", ""),
                "onset_timestamp": "" if onset is None else onset.get("recorder_timestamp", ""),
                "onset_sample": "" if onset is None else onset.get("device_sample_number", ""),
                "onset_video_time_sec": "" if onset is None else onset.get("video_time_sec", ""),
                "attention_response_timestamp": "" if attention is None else attention.get("recorder_timestamp", ""),
                "attention_response_time_ms": "" if attention is None else attention.get("response_time_ms", ""),
                "confidence_response_timestamp": "" if confidence is None else confidence.get("recorder_timestamp", ""),
                "confidence_response_time_ms": "" if confidence is None else confidence.get("response_time_ms", ""),
                "resume_timestamp": "" if resume is None else resume.get("recorder_timestamp", ""),
                "resume_sample": "" if resume is None else resume.get("device_sample_number", ""),
                "complete": int(all(item is not None for item in (onset, attention, confidence, resume))),
            })

    rating_events = events[names == "rating_end"] if not events.empty else events
    rating_rows = [
        {key: row.get(key, "") for key in RATING_COLUMNS}
        for row in _rows(rating_events)
    ]
    quiz_events = events[names == "quiz_item_response"] if not events.empty else events
    quiz_rows = [
        {key: row.get(key, "") for key in QUIZ_COLUMNS}
        for row in _rows(quiz_events)
    ]

    outputs = {
        "probes": _write_csv(session_dir / "probes.csv", PROBE_COLUMNS, probe_rows),
        "block_ratings": _write_csv(session_dir / "block_ratings.csv", RATING_COLUMNS, rating_rows),
        "quiz_responses": _write_csv(session_dir / "quiz_responses.csv", QUIZ_COLUMNS, quiz_rows),
    }
    return outputs


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _counts_by_block(frame: pd.DataFrame, *, mask: pd.Series | None = None) -> dict[str, int]:
    if frame.empty or "block_id" not in frame:
        return {str(block): 0 for block in range(1, 7)}
    selected = frame if mask is None else frame.loc[mask]
    numeric = pd.to_numeric(selected["block_id"], errors="coerce")
    return {str(block): int((numeric == block).sum()) for block in range(1, 7)}


def build_session_qc_report(session_dir: Path | str) -> Path:
    session_dir = Path(session_dir).resolve()
    build_normalized_tables(session_dir)
    metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
    session = metadata.get("session", {})
    events = _read_csv(session_dir / "events.csv")
    eeg = _read_csv(session_dir / "eeg.csv")
    qc = _read_csv(session_dir / "qc.csv")
    windows = _read_csv(session_dir / "windows.csv")
    probes = _read_csv(session_dir / "probes.csv")
    ratings = _read_csv(session_dir / "block_ratings.csv")
    quiz = _read_csv(session_dir / "quiz_responses.csv")
    names = _event_name(events)

    completed_blocks = sorted({int(value) for value in events.loc[names == "block_end", "block_id"] if str(value).isdigit()}) if not events.empty else []
    complete_probes = int((probes.get("complete", pd.Series(dtype=str)) == "1").sum())
    start_numbers = [item.get("start_number") for item in session.get("b_start_numbers", {}).values()]
    b_ratings = ratings[ratings.get("condition_label", pd.Series(dtype=str)) == "B"] if not ratings.empty else ratings
    b_complete = 0
    for _, row in b_ratings.iterrows():
        if all(str(row.get(field, "")).strip() for field in (
            "subtraction_compliance", "subtraction_difficulty", "reported_final_number"
        )):
            b_complete += 1

    pause_names = {"video_pause", "probe_onset"}
    resume_names = {"video_resume"}
    pause_count = int(names.isin(pause_names).sum()) if not events.empty else 0
    resume_count = int(names.isin(resume_names).sum()) if not events.empty else 0
    expected = {
        "blocks": 6,
        "probes": 24,
        "ratings": 6,
        "quiz_responses": 24,
        "b_start_numbers": 3,
        "b_specific_ratings": 3,
    }
    actual = {
        "blocks": len(completed_blocks),
        "probes": complete_probes,
        "ratings": len(ratings),
        "quiz_responses": len(quiz),
        "b_start_numbers": len(start_numbers),
        "b_specific_ratings": b_complete,
    }
    failures = [f"{key}: expected {value}, got {actual[key]}" for key, value in expected.items() if actual[key] != value]
    block_end_counts = _counts_by_block(events, mask=names == "block_end") if not events.empty else _counts_by_block(events)
    probe_counts = _counts_by_block(probes, mask=probes.get("complete", pd.Series(dtype=str)) == "1")
    rating_counts = _counts_by_block(ratings)
    quiz_counts = _counts_by_block(quiz)
    for block in range(1, 7):
        key = str(block)
        for label, counts, wanted in (
            ("block_end", block_end_counts, 1),
            ("complete probes", probe_counts, 4),
            ("ratings", rating_counts, 1),
            ("quiz responses", quiz_counts, 4),
        ):
            if counts[key] != wanted:
                failures.append(f"block {block} {label}: expected {wanted}, got {counts[key]}")
    if len(set(start_numbers)) != len(start_numbers):
        failures.append("B start numbers are not unique")
    for value in start_numbers:
        number = int(_number(value, 0))
        if number < 800 or number > 1000 or any(number % divisor == 0 for divisor in range(2, int(number ** 0.5) + 1)):
            failures.append(f"invalid B start number: {value}")
    pause_resume_by_reason: dict[str, dict[str, int]] = {}
    if not events.empty:
        for index, row in events.iterrows():
            name = str(names.loc[index])
            if name not in pause_names | resume_names:
                continue
            reason = "thought_probe" if name == "probe_onset" else str(row.get("pause_reason", "")).strip() or "unspecified"
            counts = pause_resume_by_reason.setdefault(reason, {"pause": 0, "resume": 0})
            counts["resume" if name in resume_names else "pause"] += 1
    for reason, counts in pause_resume_by_reason.items():
        if counts["pause"] != counts["resume"]:
            failures.append(
                f"pause/resume mismatch for {reason}: {counts['pause']}/{counts['resume']}"
            )
    if not events.empty and names.isin({"browser_recovery_blocked", "block_incomplete"}).any():
        failures.append("unsafe browser recovery or incomplete block was recorded")
    if eeg.empty:
        failures.append("eeg.csv contains no EEG samples")

    warnings: list[str] = []
    if session.get("baseline_override"):
        warnings.append("Baseline QC was overridden; inspect operator and reason")
    if not metadata.get("protocol_config_version"):
        warnings.append("Protocol config traceability is missing")
    if session.get("study_phase") != "formal":
        warnings.append(f"Study phase is {session.get('study_phase', 'unknown')}, not formal")

    sample_rate = _number(session.get("sample_rate_hz"), 250.0)
    packet_gaps = pd.to_numeric(eeg.get("packet_gap_before", pd.Series(dtype=str)), errors="coerce").fillna(0) if not eeg.empty else pd.Series(dtype=float)
    missing_packets = int(packet_gaps.sum()) if len(packet_gaps) else 0
    expected_packets = len(eeg) + missing_packets
    quality_flags = eeg.get("quality_flag", pd.Series(dtype=str)).astype(str) if not eeg.empty else pd.Series(dtype=str)
    session_metrics = {
        "received_samples": len(eeg),
        "total_missing_packets": missing_packets,
        "total_packet_loss_rate_pct": round(missing_packets / expected_packets * 100.0, 6) if expected_packets else None,
        "longest_packet_gap_sec": round((_number(packet_gaps.max()) / sample_rate), 6) if len(packet_gaps) else 0.0,
        "dropout_count": len(metadata.get("qc", {}).get("dropouts", [])),
        "dropout_total_duration_sec": round(sum(_number(item.get("duration_sec")) for item in metadata.get("qc", {}).get("dropouts", [])), 6),
        "saturated_sample_ratio_pct": round(quality_flags.str.contains("adc_saturation", regex=False).mean() * 100.0, 6) if len(quality_flags) else None,
        "channel_0_rms_uv_median": _number(pd.to_numeric(qc.get("channel_0_rms_uv", pd.Series(dtype=str)), errors="coerce").median(), None) if not qc.empty else None,
        "channel_1_rms_uv_median": _number(pd.to_numeric(qc.get("channel_1_rms_uv", pd.Series(dtype=str)), errors="coerce").median(), None) if not qc.empty else None,
    }

    per_block_valid_ratio = {}
    if not windows.empty and "block_id" in windows:
        for block_id, part in windows.groupby("block_id"):
            labels = part.get("quality_label", pd.Series([""] * len(part))).str.upper()
            per_block_valid_ratio[str(block_id)] = round(float(labels.isin(["GOOD", "USABLE"]).mean()), 6)
    session_metrics["per_block_valid_window_ratio"] = per_block_valid_ratio

    status = "FAIL" if failures else "WARN" if warnings else "PASS"
    report = {
        "status": status,
        "generated_for_run_id": metadata.get("run_id", ""),
        "software_version": metadata.get("software_version", ""),
        "protocol_version": metadata.get("protocol_version", ""),
        "protocol_config_version": metadata.get("protocol_config_version", ""),
        "protocol_config_sha256": metadata.get("protocol_config_sha256", ""),
        "counterbalance_group": session.get("counterbalance_group", ""),
        "study_phase": session.get("study_phase", ""),
        "expected": expected,
        "actual": actual,
        "completed_blocks": completed_blocks,
        "pause_count": pause_count,
        "resume_count": resume_count,
        "pause_resume_by_reason": pause_resume_by_reason,
        "per_block_actual": {
            "block_end": block_end_counts,
            "complete_probes": probe_counts,
            "ratings": rating_counts,
            "quiz_responses": quiz_counts,
        },
        "failures": failures,
        "warnings": warnings,
        "session_metrics": session_metrics,
        "line_noise_policy": {
            "measured_during_acquisition": False,
            "offline_filter": metadata.get("preprocessing", {}).get("line_noise_filter", {}),
        },
    }
    path = session_dir / "session_qc_report.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path
