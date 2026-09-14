"""Acquisition-health metrics that never classify scientific EEG usability."""

from __future__ import annotations

from typing import Iterable

import numpy as np


ADC_LIMIT = 8_388_607
SATURATION_FRACTION = 0.99
# Work in raw ADC counts: the hardware's microvolt scale is not yet verified.
# RMS is used only to detect a numerically constant stream, never as a
# physiological threshold or training label.
NEAR_ZERO_RMS_COUNTS = 1e-6
FLATLINE_DURATION_SEC = 2.0


def channel_metrics(
    raw_values: Iterable[int],
    *,
    sample_rate_hz: float,
    scale_uv_per_count: float | None = None,
) -> dict[str, float | int | None]:
    """Return data-flow diagnostics in raw ADC counts only."""
    raw = np.asarray(list(raw_values), dtype=np.float64)
    if raw.size == 0:
        return {
            "sample_count": 0,
            "saturation_rate_pct": None,
            "rms_counts": None,
            "constant_value": None,
            "near_zero_rms": None,
            "longest_unchanged_sec": 0.0,
            "invalid_flatline": False,
        }

    saturation_threshold = ADC_LIMIT * SATURATION_FRACTION
    saturation_rate_pct = float(np.mean(np.abs(raw) >= saturation_threshold) * 100.0)
    rms_counts = float(np.sqrt(np.mean(np.square(raw - np.mean(raw)))))
    changes = np.flatnonzero(np.diff(raw) != 0) + 1
    longest_run = int(np.max(np.diff(np.r_[0, changes, raw.size])))
    unchanged_sec = max(0, longest_run - 1) / float(sample_rate_hz)
    constant = bool(np.all(raw == raw[0]))
    near_zero = bool(rms_counts <= NEAR_ZERO_RMS_COUNTS)
    enough_samples = raw.size >= int(round(FLATLINE_DURATION_SEC * sample_rate_hz)) + 1
    invalid_flatline = (enough_samples and (constant or near_zero)) or unchanged_sec >= FLATLINE_DURATION_SEC

    return {
        "sample_count": int(raw.size),
        "saturation_rate_pct": saturation_rate_pct,
        "rms_counts": rms_counts,
        "constant_value": constant,
        "near_zero_rms": near_zero,
        "longest_unchanged_sec": unchanged_sec,
        "invalid_flatline": bool(invalid_flatline),
    }


def window_metrics(
    samples: Iterable[tuple[float, int, int, int]],
    *,
    sample_rate_hz: float,
    scale_uv_per_count: float,
) -> dict:
    """Calculate two-channel QC metrics from (timestamp, ch0, ch1, missing_before)."""
    rows = list(samples)
    if not rows:
        return {
            "sample_count": 0,
            "window_duration_sec": 0.0,
            "missing_packets": 0,
            "packet_loss_rate_pct": None,
            "estimated_sample_rate_hz": None,
            "channels": [
                channel_metrics([], sample_rate_hz=sample_rate_hz, scale_uv_per_count=scale_uv_per_count),
                channel_metrics([], sample_rate_hz=sample_rate_hz, scale_uv_per_count=scale_uv_per_count),
            ],
        }

    missing = sum(max(0, int(row[3])) for row in rows)
    expected = len(rows) + missing
    duration = max(0.0, float(rows[-1][0]) - float(rows[0][0]))
    return {
        "sample_count": len(rows),
        "window_duration_sec": duration,
        "missing_packets": missing,
        "packet_loss_rate_pct": (float(missing / expected * 100.0) if expected else None),
        "estimated_sample_rate_hz": (float((len(rows) - 1) / duration) if duration > 0 and len(rows) > 1 else None),
        "channels": [
            channel_metrics(
                (row[1] for row in rows),
                sample_rate_hz=sample_rate_hz,
                scale_uv_per_count=scale_uv_per_count,
            ),
            channel_metrics(
                (row[2] for row in rows),
                sample_rate_hz=sample_rate_hz,
                scale_uv_per_count=scale_uv_per_count,
            ),
        ],
    }
