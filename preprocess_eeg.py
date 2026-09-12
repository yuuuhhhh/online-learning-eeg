"""Offline 50 Hz line-noise removal on a copy of the recorded EEG.

The acquisition files are never changed. Filtering is applied separately to
continuous stream segments so a dropout is never bridged by the filter.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, iirnotch

try:
    from .protocol_config import load_protocol_config
except ImportError:
    from protocol_config import load_protocol_config


def preprocess_session(session_dir: Path | str) -> tuple[Path, Path]:
    session_dir = Path(session_dir).resolve()
    metadata_path = session_dir / "metadata.json"
    eeg_path = session_dir / "eeg.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    protocol, protocol_hash, _ = load_protocol_config(Path(__file__).resolve().parent)
    filter_config = protocol["preprocessing"]["line_noise_filter"]
    if not filter_config.get("enabled") or filter_config.get("type") != "notch_bandstop":
        raise ValueError("protocol does not enable the documented 50 Hz notch/band-stop filter")

    sample_rate = float(metadata["session"]["sample_rate_hz"])
    center_hz = float(filter_config["center_hz"])
    if center_hz >= sample_rate / 2:
        raise ValueError("50 Hz notch must be below the Nyquist frequency")
    quality_factor = 30.0
    b, a = iirnotch(center_hz, quality_factor, fs=sample_rate)
    frame = pd.read_csv(eeg_path, keep_default_na=False, low_memory=False, encoding="utf-8-sig")
    for column in ("channel_0_uv", "channel_1_uv", "device_sample_number"):
        if column not in frame:
            raise ValueError(f"eeg.csv is missing {column}")
    if "stream_segment" not in frame:
        frame["stream_segment"] = 0

    filtered = frame.copy()
    filtered_columns = ["channel_0_uv_notch50", "channel_1_uv_notch50"]
    for column in filtered_columns:
        filtered[column] = np.nan
    skipped_segments: list[dict] = []
    processed_segments = 0
    for segment_id, segment in frame.groupby("stream_segment", sort=False):
        indices = segment.index.to_numpy()
        numbers = pd.to_numeric(segment["device_sample_number"], errors="coerce").to_numpy()
        breaks = np.flatnonzero(np.diff(numbers) != 1) + 1
        for part in np.split(indices, breaks):
            if len(part) <= 3 * max(len(a), len(b)):
                skipped_segments.append({"stream_segment": str(segment_id), "rows": int(len(part)), "reason": "too_short"})
                continue
            for source, target in zip(("channel_0_uv", "channel_1_uv"), filtered_columns):
                values = pd.to_numeric(frame.loc[part, source], errors="coerce").to_numpy(dtype=float)
                if not np.all(np.isfinite(values)):
                    skipped_segments.append({"stream_segment": str(segment_id), "rows": int(len(part)), "reason": f"non_finite_{source}"})
                    break
                filtered.loc[part, target] = filtfilt(b, a, values)
            else:
                processed_segments += 1

    output_path = session_dir / "eeg_preprocessed.csv"
    temporary = output_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        filtered.to_csv(handle, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)

    report = {
        "source_file": eeg_path.name,
        "output_file": output_path.name,
        "raw_source_overwritten": False,
        "filter": {**filter_config, "quality_factor": quality_factor, "zero_phase": True},
        "sample_rate_hz": sample_rate,
        "processed_continuous_segments": processed_segments,
        "skipped_segments": skipped_segments,
        "protocol_config_version": protocol["config_version"],
        "protocol_config_sha256": protocol_hash,
    }
    report_path = session_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, report_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python preprocess_eeg.py <session_directory>")
    for output in preprocess_session(sys.argv[1]):
        print(output)

