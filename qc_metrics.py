"""Reusable EEG quality-control calculations for the two-channel recorder."""

from __future__ import annotations

from typing import Iterable

import numpy as np


ADC_LIMIT = 8_388_607
SATURATION_FRACTION = 0.99
TOTAL_POWER_LOW_HZ = 1.0
TOTAL_POWER_HIGH_HZ = 100.0
HIGH_FREQUENCY_LOW_HZ = 70.0
HIGH_FREQUENCY_HIGH_HZ = 100.0
# Work in raw ADC counts: the hardware's microvolt scale is not yet verified.
# This is numerical zero, not a physiological amplitude cutoff.
NEAR_ZERO_RMS_COUNTS = 1e-6
FLATLINE_DURATION_SEC = 2.0


def channel_metrics(
    raw_values: Iterable[int],
    *,
    sample_rate_hz: float,
    scale_uv_per_count: float,
) -> dict[str, float | int | None]:
    """Return acquisition QC metrics without measuring the 50 Hz line-noise ratio."""
    raw = np.asarray(list(raw_values), dtype=np.float64)
    if raw.size == 0:
        return {
            "sample_count": 0,
            "saturation_rate_pct": None,
            "rms_uv": None,
            "rms_counts": None,
            "dc_offset_counts": None,
            "peak_abs_counts": None,
            "high_frequency_ratio_pct": None,
            "constant_value": None,
            "near_zero_rms": None,
            "longest_unchanged_sec": 0.0,
            "invalid_flatline": False,
        }

    saturation_threshold = ADC_LIMIT * SATURATION_FRACTION
    saturation_rate_pct = float(np.mean(np.abs(raw) >= saturation_threshold) * 100.0)
    demeaned_uv = (raw - np.mean(raw)) * float(scale_uv_per_count)
    rms_uv = float(np.sqrt(np.mean(np.square(demeaned_uv))))
    rms_counts = float(np.sqrt(np.mean(np.square(raw - np.mean(raw)))))
    changes = np.flatnonzero(np.diff(raw) != 0) + 1
    longest_run = int(np.max(np.diff(np.r_[0, changes, raw.size])))
    unchanged_sec = max(0, longest_run - 1) / float(sample_rate_hz)
    constant = bool(np.all(raw == raw[0]))
    near_zero = bool(rms_counts <= NEAR_ZERO_RMS_COUNTS)
    enough_samples = raw.size >= int(round(FLATLINE_DURATION_SEC * sample_rate_hz)) + 1
    invalid_flatline = (enough_samples and (constant or near_zero)) or unchanged_sec >= FLATLINE_DURATION_SEC

    high_frequency_ratio_pct: float | None = None
    if raw.size >= max(32, int(round(sample_rate_hz * 2.0))):
        windowed = demeaned_uv * np.hanning(raw.size)
        spectrum = np.fft.rfft(windowed)
        power = np.square(np.abs(spectrum))
        frequencies = np.fft.rfftfreq(raw.size, d=1.0 / float(sample_rate_hz))
        total_mask = (frequencies >= TOTAL_POWER_LOW_HZ) & (
            frequencies <= min(TOTAL_POWER_HIGH_HZ, sample_rate_hz / 2.0)
        )
        high_frequency_mask = (frequencies >= HIGH_FREQUENCY_LOW_HZ) & (
            frequencies <= min(HIGH_FREQUENCY_HIGH_HZ, sample_rate_hz / 2.0)
        )
        total_power = float(np.sum(power[total_mask]))
        if total_power > 0:
            high_frequency_ratio_pct = float(np.sum(power[high_frequency_mask]) / total_power * 100.0)
        elif rms_uv == 0:
            high_frequency_ratio_pct = 0.0

    return {
        "sample_count": int(raw.size),
        "saturation_rate_pct": saturation_rate_pct,
        "rms_uv": rms_uv,
        "rms_counts": rms_counts,
        "dc_offset_counts": float(np.mean(raw)),
        "peak_abs_counts": float(np.max(np.abs(raw))),
        "high_frequency_ratio_pct": high_frequency_ratio_pct,
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
