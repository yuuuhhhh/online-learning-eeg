"""Convert one completed EEG session directory to a MATLAB .mat file."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


NUMERIC_EVENT_COLUMNS = [
    "event_timestamp", "sample_time_sec", "received_order", "block_id", "group_id", "duration_sec",
    "exclude_before_sec", "exclude_after_sec", "is_correct", "reaction_time_ms",
    "received_timestamp", "client_timestamp", "client_monotonic_ms", "clock_offset_sec",
    "clock_round_trip_ms", "eeg_received_order", "eeg_sample_number", "eeg_stream_segment", "alignment_error_ms",
    "context_matches_current", "state_applied",
    "start_number", "generated_timestamp", "displayed_timestamp", "hidden_timestamp",
    "after_block_id", "planned_duration", "actual_duration", "is_formal_experiment",
    "video_time_sec", "browser_timestamp", "recorder_timestamp", "device_sample_number",
    "probe_index", "probe_planned_time_sec", "response", "response_time_ms", "confidence",
    "block_order", "session_half", "sensitivity_exclude_before_sec",
    "course_attention_rating", "mental_effort", "video_interest", "video_difficulty",
    "subtraction_compliance_low_pct", "subtraction_compliance_high_pct",
    "subtraction_difficulty", "reported_final_number", "rating_value",
    "quiz_score", "quiz_total", "response_index",
]

QC_TEXT_COLUMNS = {"context", "status", "messages"}
NUMERIC_INDEX_COLUMNS = {
    "block_id", "block_order", "sequence_order", "window_index",
    "probe_response", "probe_confidence", "seconds_to_probe",
    "probe_onset_sample", "probe_onset_timestamp", "probe_onset_video_time_sec",
    "epoch_duration_sec", "start_sample", "end_sample",
    "start_timestamp", "end_timestamp", "sample_count", "window_count",
    "course_attention_rating", "mental_effort", "video_interest", "video_difficulty",
    "subtraction_compliance_low_pct", "subtraction_compliance_high_pct",
    "subtraction_difficulty", "reported_final_number", "response_index", "is_correct",
    "reaction_time_ms", "complete",
}
EEG_NUMERIC_COLUMNS = {
    "received_timestamp", "received_order", "device_sample_number", "sample_index",
    "sample_time_sec", "packet_gap_before", "weak_label", "base_valid_for_training",
    "channel_0_raw", "channel_1_raw", "channel_0_uv", "channel_1_uv", "stream_segment",
    "condition_code", "is_formal_experiment",
}


def _string_array(series: pd.Series) -> np.ndarray:
    return series.fillna("").astype(str).to_numpy(dtype=object)


def _numeric_array(series: pd.Series, dtype=np.float64, fill=np.nan) -> np.ndarray:
    # CSV can contain booleans alongside numeric summaries and empty cells.
    normalized = series.map(
        lambda value: {"true": 1, "false": 0}.get(str(value).strip().lower(), value)
    )
    return pd.to_numeric(normalized, errors="coerce").fillna(fill).to_numpy(dtype=dtype)


def _csv_text(path: Path) -> pd.DataFrame:
    # Keep identifiers (001), literal NA/None, and empty cells intact. Numeric
    # projections are accompanied by an exact text representation in the MAT.
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _text_table(frame: pd.DataFrame) -> dict:
    # A matrix also preserves unknown/long CSV column names without MATLAB's
    # structure-field naming restrictions.
    return {"columns": np.asarray(frame.columns, dtype=object),
            "values": frame.fillna("").to_numpy(dtype=object)}


def _read_eeg(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    # Let pandas retain efficient numeric storage for the high-volume EEG table.
    # Text/unknown columns stay strings, including identifiers with leading zeros.
    return pd.read_csv(path, keep_default_na=False, low_memory=False, encoding="utf-8-sig",
                       dtype={column: str for column in columns if column not in EEG_NUMERIC_COLUMNS})


def _compatible_numeric(series: pd.Series, dtype) -> np.ndarray:
    values = _numeric_array(series)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        if not np.all(np.isfinite(values) & (values >= limits.min) & (values <= limits.max) & (values == np.floor(values))):
            return values  # Keep missing values as NaN, never silently replace with zero.
    return values.astype(dtype)


def _valid_mat_field(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,62}", name))


def build_training_mask(eeg: pd.DataFrame, events: pd.DataFrame, metadata: dict) -> np.ndarray:
    """Apply protocol exclusions while preserving block-level labels."""
    sample_time = _numeric_array(eeg["sample_time_sec"])
    mask = _numeric_array(eeg["base_valid_for_training"], fill=0).astype(bool)
    mask &= np.isfinite(sample_time)
    for column in ("channel_0_raw", "channel_1_raw"):
        mask &= np.isfinite(_numeric_array(eeg[column]))
    rules = {"exclude_video_start_sec": 10.0, "exclude_video_end_sec": 5.0,
             **metadata.get("training_rules", {})}

    eeg_block_ids = pd.to_numeric(eeg["block_id"], errors="coerce").fillna(0).astype(int)
    event_block_ids = pd.to_numeric(events["block_id"], errors="coerce").fillna(0).astype(int)
    for block_id in sorted(x for x in eeg_block_ids.unique() if x > 0):
        block_rows = eeg_block_ids == block_id
        block_events = events[event_block_ids == block_id]
        starts = pd.to_numeric(block_events.loc[block_events["event_type"] == "video_start", "sample_time_sec"], errors="coerce").dropna()
        ends = pd.to_numeric(block_events.loc[block_events["event_type"] == "video_end", "sample_time_sec"], errors="coerce").dropna()
        if len(starts):
            start = float(starts.iloc[0]) + float(rules["exclude_video_start_sec"])
            mask[block_rows & (sample_time < start)] = False
        if len(ends):
            end = float(ends.iloc[-1]) - float(rules["exclude_video_end_sec"])
            mask[block_rows & (sample_time > end)] = False

    for _, event in events.iterrows():
        before = pd.to_numeric(pd.Series([event.get("exclude_before_sec")]), errors="coerce").iloc[0]
        after = pd.to_numeric(pd.Series([event.get("exclude_after_sec")]), errors="coerce").iloc[0]
        if pd.isna(before):
            before = 0.0
        if pd.isna(after):
            after = 0.0
        if before > 0 or after > 0:
            center = pd.to_numeric(event["sample_time_sec"], errors="coerce")
            if pd.isna(center):
                continue
            mask[(sample_time >= center - float(before)) & (sample_time <= center + float(after))] = False

    return mask.astype(np.uint8)


def _ensure_attention_index(session_dir: Path, events: pd.DataFrame) -> None:
    """Build the probe index for a probe session that has not been indexed yet."""
    if (session_dir / "windows.csv").exists():
        return
    if "probe_onset" not in set(events.get("event_type", pd.Series(dtype=str))):
        return
    try:
        from .epoch_builder import build_session_epochs
    except ImportError:
        from epoch_builder import build_session_epochs
    try:
        build_session_epochs(session_dir)
    except Exception:
        pass  # An index failure must never stop the session from exporting.


def export_session_to_mat(session_dir: Path | str, *, metadata_override: dict | None = None) -> Path:
    session_dir = Path(session_dir).resolve()
    eeg_path = session_dir / "eeg.csv"
    events_path = session_dir / "events.csv"
    metadata_path = session_dir / "metadata.json"
    qc_path = session_dir / "qc.csv"
    output_path = session_dir / "session.mat"

    if not eeg_path.exists() or not events_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Session directory must contain eeg.csv, events.csv and metadata.json")

    eeg = _read_eeg(eeg_path)
    events = _csv_text(events_path)
    for column in ("block_id", "event_type", "sample_time_sec"):
        if column not in events:
            events[column] = ""
    metadata = metadata_override if metadata_override is not None else json.loads(metadata_path.read_text(encoding="utf-8"))
    _ensure_attention_index(session_dir, events)
    training_mask = build_training_mask(eeg, events, metadata)

    block_ids = pd.to_numeric(eeg["block_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int16)
    if "group_id" in eeg.columns:
        group_ids = pd.to_numeric(eeg["group_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    else:
        group_ids = np.where(block_ids > 0, (block_ids + 1) // 2, 0).astype(np.int8)

    numeric_eeg = {
        "received_timestamp": np.float64, "received_order": np.int64,
        "device_sample_number": np.int64, "sample_index": np.uint8,
        "sample_time_sec": np.float64, "packet_gap_before": np.uint16,
        "weak_label": np.int8, "base_valid_for_training": np.uint8,
    }
    if "stream_segment" in eeg:
        numeric_eeg["stream_segment"] = np.uint32
    if "condition_code" in eeg:
        numeric_eeg["condition_code"] = np.int8
    if "is_formal_experiment" in eeg:
        numeric_eeg["is_formal_experiment"] = np.uint8
    eeg_struct = {
        "raw": np.column_stack([_compatible_numeric(eeg[f"channel_{i}_raw"], np.int32) for i in range(2)]),
        "uv": np.column_stack([_compatible_numeric(eeg[f"channel_{i}_uv"], np.float32) for i in range(2)]),
        **{column: _compatible_numeric(eeg[column], dtype) for column, dtype in numeric_eeg.items()},
        "block_id": block_ids,
        "group_id": group_ids,
        "condition": _string_array(eeg["condition"]),
        "phase": _string_array(eeg["phase"]),
        "quality_flag": _string_array(eeg["quality_flag"]),
        "training_mask": training_mask,
    }
    if "sample_time_status" in eeg:
        eeg_struct["sample_time_status"] = _string_array(eeg["sample_time_status"])
    if "condition_type" in eeg:
        eeg_struct["condition_type"] = _string_array(eeg["condition_type"])

    event_struct = {}
    for column in events.columns:
        if not _valid_mat_field(column):
            continue  # Original names and values remain available in csv_text.
        if column in NUMERIC_EVENT_COLUMNS:
            event_struct[column] = _numeric_array(events[column])
        else:
            event_struct[column] = _string_array(events[column])

    # Avoid duplicating every normal EEG numeric cell as a MATLAB cell string.
    # Preserve unknown columns and original values in columns needing coercion.
    known_eeg = set(numeric_eeg) | {"block_id", "group_id", "condition", "phase", "quality_flag",
                                  "channel_0_raw", "channel_1_raw", "channel_0_uv", "channel_1_uv"}
    text_eeg = {"condition", "condition_type", "phase", "quality_flag", "sample_time_status"}
    known_eeg |= text_eeg
    preserved_columns = [column for column in eeg if column not in known_eeg]
    original_rows, original_columns, original_values = [], [], []
    for column in eeg:
        if column in known_eeg and column not in text_eeg:
            values = eeg[column]
            changed = ~np.isfinite(_numeric_array(values))
            if values.dtype == object or values.dtype == bool:
                changed |= values.map(lambda value: str(value).strip().lower() in {"true", "false"}).to_numpy(dtype=bool)
            for row_index in np.flatnonzero(changed):
                original_rows.append(row_index)
                original_columns.append(column)
                original_values.append(str(values.iloc[row_index]))
    labels = metadata.get("labels", {})
    legacy_label_names = [labels[key] for key in ("0", "1") if key in labels]
    conditions = metadata.get("session", {}).get("conditions", metadata.get("conditions", {}))
    mat_payload = {
        "eeg": eeg_struct,
        "events": event_struct,
        "csv_text": {
            "eeg": _text_table(eeg[preserved_columns]), "events": _text_table(events),
            "eeg_numeric_original": {"row_index": np.asarray(original_rows, dtype=np.int64),
                                     "column": np.asarray(original_columns, dtype=object),
                                     "value": np.asarray(original_values, dtype=object)},
        },
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "sample_rate_hz": float(metadata["session"]["sample_rate_hz"]),
        "channel_names": np.asarray(metadata["session"]["channel_names"], dtype=object),
        "label_names": np.asarray(legacy_label_names or [labels.get("-1", "no_instantaneous_attention_label")], dtype=object),
        "condition_labels": np.asarray(["A", "B"], dtype=object),
        "condition_types": np.asarray([
            conditions.get("A", {}).get("condition_type", "legacy_A"),
            conditions.get("B", {}).get("condition_type", "legacy_B"),
        ], dtype=object),
    }
    # Derived attention indices are optional: sessions recorded before stage 4
    # simply do not carry these keys.
    for name, path in (
        ("probe_epochs", session_dir / "probe_epochs.csv"),
        ("windows", session_dir / "windows.csv"),
        ("probes", session_dir / "probes.csv"),
        ("block_ratings", session_dir / "block_ratings.csv"),
        ("quiz_responses", session_dir / "quiz_responses.csv"),
    ):
        if not path.exists() or path.stat().st_size == 0:
            continue
        table = _csv_text(path)
        if table.empty and not len(table.columns):
            continue
        struct = {}
        for column in table.columns:
            if not _valid_mat_field(column):
                continue
            struct[column] = (_numeric_array(table[column]) if column in NUMERIC_INDEX_COLUMNS
                              else _string_array(table[column]))
        mat_payload[name] = struct
        mat_payload["csv_text"][name] = _text_table(table)

    if qc_path.exists() and qc_path.stat().st_size > 0:
        qc = _csv_text(qc_path)
        qc_struct = {}
        for column in qc.columns:
            if not _valid_mat_field(column):
                continue
            if column in QC_TEXT_COLUMNS:
                qc_struct[column] = _string_array(qc[column])
            else:
                qc_struct[column] = _numeric_array(qc[column])
        mat_payload["qc"] = qc_struct
        mat_payload["csv_text"]["qc"] = _text_table(qc)
    report_path = session_dir / "session_qc_report.json"
    if report_path.exists():
        mat_payload["session_qc_report_json"] = report_path.read_text(encoding="utf-8")
    temporary_path = output_path.with_suffix(".mat.tmp")
    try:
        with temporary_path.open("wb") as handle:
            savemat(handle, mat_payload, do_compression=True, long_field_names=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python mat_exporter.py <session_directory>")
    print(export_session_to_mat(sys.argv[1]))
