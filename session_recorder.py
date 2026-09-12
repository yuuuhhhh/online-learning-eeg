"""Thread-safe EEG session recording and experiment event logging."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import secrets
import threading
import time
import uuid
from array import array
from bisect import bisect_right
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from .qc_metrics import ADC_LIMIT, SATURATION_FRACTION, FLATLINE_DURATION_SEC, NEAR_ZERO_RMS_COUNTS, window_metrics
    from .protocol_config import load_protocol_config, planned_blocks
except ImportError:
    from qc_metrics import ADC_LIMIT, SATURATION_FRACTION, FLATLINE_DURATION_SEC, NEAR_ZERO_RMS_COUNTS, window_metrics
    from protocol_config import load_protocol_config, planned_blocks


PROTOCOL_CONFIG, PROTOCOL_CONFIG_HASH, PROTOCOL_CONFIG_PATH = load_protocol_config(Path(__file__).resolve().parent)


QC_CSV_COLUMNS = [
    "qc_timestamp",
    "sample_time_sec",
    "context",
    "window_duration_sec",
    "sample_count",
    "missing_packets",
    "packet_loss_rate_pct",
    "channel_0_saturation_rate_pct",
    "channel_1_saturation_rate_pct",
    "channel_0_rms_uv",
    "channel_1_rms_uv",
    "channel_0_dc_offset_counts",
    "channel_1_dc_offset_counts",
    "channel_0_peak_abs_counts",
    "channel_1_peak_abs_counts",
    "channel_0_high_frequency_ratio_pct",
    "channel_1_high_frequency_ratio_pct",
    "signal_alive",
    "status",
    "messages",
]
QC_SIGNAL_FIELDS = ("rms_counts", "constant_value", "near_zero_rms", "longest_unchanged_sec", "invalid_flatline")
QC_CSV_COLUMNS += [f"channel_{channel}_{field}" for channel in range(2) for field in QC_SIGNAL_FIELDS]


EEG_CSV_COLUMNS = [
    "received_timestamp",
    "received_order",
    "device_sample_number",
    "sample_index",
    "sample_time_sec",
    "packet_gap_before",
    "block_id",
    "group_id",
    "block_order",
    "session_half",
    "counterbalance_group",
    "video_id",
    "condition",
    "condition_label",
    "condition_type",
    "condition_code",
    "weak_label",
    "phase",
    "base_valid_for_training",
    "channel_0_raw",
    "channel_1_raw",
    "channel_0_uv",
    "channel_1_uv",
    "quality_flag",
]
EEG_CSV_COLUMNS += ["is_formal_experiment", "stream_segment", "sample_time_status"]

EVENT_CSV_COLUMNS = [
    "event_name",
    "event_timestamp",
    "sample_time_sec",
    "received_order",
    "block_id",
    "block_order",
    "group_id",
    "session_half",
    "counterbalance_group",
    "condition",
    "condition_type",
    "event_type",
    "event_value",
    "duration_sec",
    "exclude_before_sec",
    "exclude_after_sec",
    "trial_id",
    "question_id",
    "correct_answer",
    "participant_answer",
    "is_correct",
    "reaction_time_ms",
    "notes",
]
EVENT_CSV_COLUMNS += [
    "run_id", "subject_id", "session_id", "client_event_id", "client_id",
    "received_timestamp", "client_timestamp", "client_monotonic_ms",
    "timestamp_source", "clock_offset_sec", "clock_round_trip_ms",
    "eeg_received_order", "eeg_sample_number", "eeg_stream_segment", "alignment_error_ms", "alignment_status",
    "identity_status", "context_matches_current", "state_applied",
    "start_number", "generated_timestamp", "displayed_timestamp", "hidden_timestamp",
    "after_block_id", "planned_duration", "actual_duration", "is_formal_experiment",
    "operator_id", "override_reason", "sensitivity_exclude_before_sec",
    "course_attention_rating", "mental_effort", "video_interest", "video_difficulty",
    "subtraction_compliance", "subtraction_compliance_low_pct", "subtraction_compliance_high_pct",
    "subtraction_difficulty", "reported_final_number", "rating_item", "rating_value",
    "quiz_score", "quiz_total", "question_text", "response_index", "response_text",
]
PROBE_EVENT_COLUMNS = [
    "probe_id", "probe_index", "probe_schedule_id", "probe_planned_time_sec",
    "response", "response_label", "response_time_ms", "confidence",
    "probe_attention", "pause_reason",
]
# browser_timestamp/recorder_timestamp/device_sample_number restate existing
# quantities under the protocol's field names so probe rows are self-contained.
EVENT_CSV_COLUMNS += [
    "video_id", "condition_label", "video_time_sec",
    "browser_timestamp", "recorder_timestamp", "device_sample_number",
] + PROBE_EVENT_COLUMNS

CONDITION_CODES = {"": -1, "A": 0, "B": 1}
CONDITIONS = PROTOCOL_CONFIG["conditions"]
REST_DURATIONS_AFTER_BLOCK = {
    int(key): int(value)
    for key, value in PROTOCOL_CONFIG["flow"]["rest_durations_after_block_sec"].items()
}

PROBE_QUESTION = PROTOCOL_CONFIG["thought_probe"]["question"]
PROBE_CONFIDENCE_QUESTION = PROTOCOL_CONFIG["thought_probe"]["confidence_question"]
PROBE_OPTIONS = {item["value"]: item["label"] for item in PROTOCOL_CONFIG["thought_probe"]["options"]}
# The response is stored exactly as given. A条件选择"连续减7"同样记录为 OFF，
# 程序不会因为当前条件自动纠正被试的回答。
PROBE_ATTENTION_MAP = {item["value"]: item["attention"] for item in PROTOCOL_CONFIG["thought_probe"]["options"]}
CONFIDENCE_CHOICES = tuple(PROTOCOL_CONFIG["thought_probe"]["confidence_choices"])
PROBE_CONFIG = {
    key: PROTOCOL_CONFIG["thought_probe"][key]
    for key in ("probes_per_block", "min_first_onset_sec", "min_end_margin_sec", "min_gap_sec")
}
# Development-only configuration so probes can be exercised with short test clips.
PROBE_DEBUG_CONFIG = {
    "probes_per_block": 4,
    "min_first_onset_sec": 5.0,
    "min_end_margin_sec": 3.0,
    "min_gap_sec": 5.0,
}
# Schedules are transported with three decimals; absorb that rounding only.
PROBE_TIME_TOLERANCE_SEC = 1e-3
PROBE_EVENTS = {"probe_schedule", "probe_onset", "attention_response", "confidence_response", "probe_cancelled"}

LEGACY_ARITHMETIC_EVENTS = {
    "arithmetic_onset", "arithmetic_response", "arithmetic_timeout", "arithmetic_cancelled",
    "math_onset", "math_response", "math_timeout", "math_summary",
}


def _epoch_rules() -> dict[str, Any]:
    """Describe the derived attention index; epoch_builder owns the values."""
    try:
        from .epoch_builder import ATTENTION_LABELS, EPOCH_DURATION_SEC, WINDOW_LENGTH_SEC, WINDOW_STEP_SEC
    except ImportError:
        from epoch_builder import ATTENTION_LABELS, EPOCH_DURATION_SEC, WINDOW_LENGTH_SEC, WINDOW_STEP_SEC
    return {
        "epoch_duration_sec": EPOCH_DURATION_SEC,
        "window_length_sec": WINDOW_LENGTH_SEC,
        "window_step_sec": WINDOW_STEP_SEC,
        "attention_labels": list(ATTENTION_LABELS),
        "anchor": "probe_onset device_sample_number; the epoch ends on the last sample before the probe page",
        "files": ["probe_epochs.csv", "windows.csv"],
        "storage": "sample indices and labels only; EEG stays once in eeg.csv and eeg_raw.bin",
        "grouping": "split train/test by probe_id, block and subject; never split one probe's windows",
        "none_policy": "ordinary video EEG that no probe asked about is labelled NONE, never ON or OFF",
    }


def _is_legacy_arithmetic_event(event_type: Any) -> bool:
    value = str(event_type).strip().lower()
    return value in LEGACY_ARITHMETIC_EVENTS or value.startswith(("arithmetic_", "math_"))


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    for divisor in range(2, int(value ** 0.5) + 1):
        if value % divisor == 0:
            return False
    return True


def generate_unique_b_start_numbers(sequence: list[str]) -> dict[int, int]:
    """Generate one unique prime in 800..1000 for each B block."""
    blocks = [index for index, condition in enumerate(sequence, start=1) if condition == "B"]
    candidates = [value for value in range(800, 1001) if _is_prime(value)]
    return dict(zip(blocks, secrets.SystemRandom().sample(candidates, len(blocks))))


def normalize_probe_config(raw: Any = None) -> dict[str, float]:
    """Return a probe configuration with the protocol defaults filled in."""
    config = dict(PROBE_CONFIG)
    for key, default in PROBE_CONFIG.items():
        if not isinstance(raw, dict) or key not in raw:
            continue
        value = float(raw[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"probe config {key} must be a finite non-negative number")
        config[key] = int(value) if key == "probes_per_block" else value
    config["probes_per_block"] = int(config["probes_per_block"])
    if config["probes_per_block"] < 0:
        raise ValueError("probes_per_block cannot be negative")
    if config["min_gap_sec"] <= 0:
        raise ValueError("min_gap_sec must be positive")
    return config


def feasible_probe_count(duration_sec: float, config: dict[str, float]) -> int:
    """How many probes fit in a video once the edge margins are removed."""
    span = (duration_sec - config["min_end_margin_sec"]) - config["min_first_onset_sec"]
    if not math.isfinite(span) or span < 0:
        return 0
    return max(0, min(config["probes_per_block"], int(span // config["min_gap_sec"]) + 1))


def validate_probe_schedule(times: Any, duration_sec: Any, raw_config: Any = None) -> tuple[list[float], dict[str, float]]:
    """Check one block's probe onsets against the hard scheduling constraints."""
    config = normalize_probe_config(raw_config)
    duration = float(duration_sec)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("probe schedule requires a positive video duration")
    if not isinstance(times, (list, tuple)):
        raise ValueError("probe schedule times must be a list")
    values = [float(value) for value in times]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("probe onset times must be finite")
    feasible = feasible_probe_count(duration, config)
    expected = config["probes_per_block"]
    if feasible < expected:
        raise ValueError(
            f"video duration {duration:g}s cannot fit all {expected} probes under the hard constraints"
        )
    if len(values) != expected:
        raise ValueError(f"probe schedule must contain {expected} onsets for a {duration:g}s video, got {len(values)}")
    earliest = config["min_first_onset_sec"] - PROBE_TIME_TOLERANCE_SEC
    latest = duration - config["min_end_margin_sec"] + PROBE_TIME_TOLERANCE_SEC
    for index, value in enumerate(values):
        if value < earliest:
            raise ValueError(f"probe {index + 1} starts before {config['min_first_onset_sec']:g}s of video")
        if value > latest:
            raise ValueError(f"probe {index + 1} starts inside the final {config['min_end_margin_sec']:g}s of video")
        if index and value - values[index - 1] < config["min_gap_sec"] - PROBE_TIME_TOLERANCE_SEC:
            raise ValueError(f"probes {index} and {index + 1} are closer than {config['min_gap_sec']:g}s")
    return values, config


def _safe_id(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{field_name} may contain only letters, numbers, '_' and '-'")
    return value


def parse_24bit_signed(data: bytes) -> int:
    if len(data) != 3:
        raise ValueError("A 24-bit EEG value must contain exactly 3 bytes")
    value = (data[0] << 16) | (data[1] << 8) | data[2]
    if value & 0x800000:
        value -= 0x1000000
    return value


@dataclass
class SessionConfig:
    subject_id: str
    session_id: str = "ses-001"
    counterbalance_group: str = "G01"
    study_phase: str = "pilot"
    operator_id: str = ""
    notes: str = ""
    sample_rate_hz: float = 250.0
    device_name: str = "MindBridge-v3.11"
    channel_names: list[str] = field(default_factory=lambda: ["ear_channel_0", "ear_channel_1"])
    eeg_scale_uv_per_count: float = 0.288486 / 12.0
    scale_status: str = "unverified; confirm gain, reference voltage and units with device provider"
    experiment_name: str = "online_learning_focused_vs_bbbd_subtraction"
    output_root: str = "data"
    qc_interval_sec: float = 2.0
    qc_window_sec: float = 10.0
    eeg_absence_alert_sec: float = float(PROTOCOL_CONFIG["qc"]["eeg_absence_alert_sec"])
    packet_loss_warning_pct: float = float(PROTOCOL_CONFIG["qc"]["packet_loss_warning_pct"])
    packet_loss_fail_pct: float = float(PROTOCOL_CONFIG["qc"]["packet_loss_fail_pct"])
    extreme_dc_offset_counts: float = float(PROTOCOL_CONFIG["qc"]["extreme_dc_offset_counts"])
    extreme_amplitude_counts: float = float(PROTOCOL_CONFIG["qc"]["extreme_amplitude_counts"])
    high_frequency_ratio_warning_pct: float = float(PROTOCOL_CONFIG["qc"]["high_frequency_ratio_warning_pct"])
    high_frequency_ratio_fail_pct: float = float(PROTOCOL_CONFIG["qc"]["high_frequency_ratio_fail_pct"])

    def validate(self) -> None:
        self.subject_id = _safe_id(self.subject_id, "subject_id")
        self.session_id = _safe_id(self.session_id, "session_id")
        if self.operator_id:
            self.operator_id = _safe_id(self.operator_id, "operator_id")
        self.counterbalance_group = self.counterbalance_group.strip().upper()
        if self.counterbalance_group not in PROTOCOL_CONFIG["counterbalance_groups"]:
            raise ValueError("counterbalance_group must be G01 through G12")
        if self.study_phase not in PROTOCOL_CONFIG["study_phases"]:
            raise ValueError("study_phase must be smoke, pilot, or formal")
        if self.study_phase == "formal" and not PROTOCOL_CONFIG["frozen_for_formal"]:
            raise ValueError("正式采集被阻止：当前协议配置尚未在 pilot 后冻结")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if len(self.channel_names) != 2:
            raise ValueError("This device bridge currently supports exactly 2 real EEG channels")
        if self.qc_interval_sec <= 0 or self.qc_window_sec < 2:
            raise ValueError("QC interval/window settings are invalid")


class ExperimentRecorder:
    """Save EEG, raw packets, metadata and synchronized experiment events."""

    def __init__(self, config: SessionConfig, project_dir: Path):
        config.validate()
        self.config = config
        self.project_dir = Path(project_dir).resolve()
        output_root = Path(config.output_root)
        if not output_root.is_absolute():
            output_root = self.project_dir / output_root
        run_name = datetime.now().strftime("run-%Y%m%d_%H%M%S")
        self.session_dir = output_root / config.subject_id / config.session_id / run_name
        self.run_id = str(uuid.uuid4())

        self.eeg_path = self.session_dir / "eeg.csv"
        self.events_path = self.session_dir / "events.csv"
        self.raw_path = self.session_dir / "eeg_raw.bin"
        self.qc_path = self.session_dir / "qc.csv"
        self.metadata_path = self.session_dir / "metadata.json"
        self.mat_path = self.session_dir / "session.mat"

        self._lock = threading.RLock()
        self._eeg_handle = None
        self._event_handle = None
        self._raw_handle = None
        self._qc_handle = None
        self._eeg_writer = None
        self._event_writer = None
        self._qc_writer = None
        self._active = False

        self.received_order = 0
        self.device_sample_number = 0
        self.last_sample_index: Optional[int] = None
        self.total_missing_packets = 0
        self.duplicate_packets = 0
        self.saturated_samples = 0
        self.event_count = 0
        self._seen_browser_event_ids: set[str] = set()
        self._event_context: dict[str, Any] = {}
        self._browser_apply_state = True
        self._last_browser_state_timestamp = float("-inf")
        self._clock_wall = time.time()
        self._clock_monotonic = time.monotonic()
        self._sample_timestamps = array("d")
        self._sample_numbers = array("q")
        self._sample_orders = array("q")
        self._sample_segments = array("I")
        self.stream_segment = 0
        self._stream_discontinuity = False
        self._sample_time_status = "packet_counter"
        self._last_browser_event_ids: dict[str, str] = {}
        self.unresolved_browser_events: dict[str, list[dict[str, Any]]] = {}
        self.browser_clients: dict[str, dict[str, Any]] = {}
        self.shutdown_request_id: Optional[str] = None
        self._shutdown_clients: set[str] = set()
        self._shutdown_acks: set[str] = set()
        self._session_end_logged = False
        self.export_status = "not_started"
        self.epoch_status = "not_started"
        self.integrity_status = "not_started"
        self.started_timestamp: Optional[float] = None
        self.ended_timestamp: Optional[float] = None

        self.current_block = 0
        self.current_session_half = 0
        self.current_condition = ""
        self.current_condition_type = ""
        self.current_video_id = ""
        self.phase = "idle"
        self.completed_blocks: set[int] = set()

        generated_at = self.clock_time()
        self.b_start_numbers = {
            block_id: {"start_number": value, "generated_timestamp": generated_at}
            for block_id, value in generate_unique_b_start_numbers(self.planned_sequence).items()
        }
        self.subtraction_practice = {
            "practice_number": 100,
            "started": False,
            "confirmed": False,
            "started_timestamp": None,
            "ended_timestamp": None,
        }
        self.probe_config = dict(PROBE_CONFIG)
        self.probe_schedules: dict[int, dict[str, Any]] = {}
        self._probe_stages: dict[str, str] = {}

        qc_capacity = max(1, int(round(config.sample_rate_hz * 65.0)))
        self._qc_samples: deque[tuple[float, int, int, int]] = deque(maxlen=qc_capacity)
        self._last_packet_timestamp: Optional[float] = None
        self._cached_qc: dict[str, Any] = {}
        self._rms_instability_started_at: list[Optional[float]] = [None, None]
        self._last_qc_status = "waiting"
        self.qc_snapshot_count = 0

        self.baseline_started_timestamp: Optional[float] = None
        self.baseline_target_sec = float(PROTOCOL_CONFIG["flow"]["baseline_duration_sec"])
        self.baseline_complete = False
        self.baseline_passed = False
        self.baseline_override = False
        self.baseline_result: dict[str, Any] = {}
        self.baseline_rms_uv: list[Optional[float]] = [None, None]
        self.baseline_override_details: dict[str, Any] = {}
        self._dropout_started_timestamp: Optional[float] = None
        self.dropout_records: list[dict[str, Any]] = []

    def clock_time(self) -> float:
        """Unix-anchored monotonic host clock, shared by samples and event sync."""
        return self._clock_wall + (time.monotonic() - self._clock_monotonic)

    def mark_stream_discontinuity(self) -> None:
        with self._lock:
            self._stream_discontinuity = True

    def register_browser(self, client_id: str, pending: int = 0) -> None:
        if not client_id:
            return
        with self._lock:
            self.browser_clients[client_id] = {"last_seen": self.clock_time(), "pending": max(0, pending)}
            if self.shutdown_request_id:
                self._shutdown_clients.add(client_id)

    def request_shutdown(self) -> str:
        with self._lock:
            self.shutdown_request_id = str(uuid.uuid4())
            self._shutdown_clients = set(self.browser_clients)
            self._shutdown_acks.clear()
            return self.shutdown_request_id

    def cancel_shutdown(self) -> None:
        with self._lock:
            self.shutdown_request_id = None
            self._shutdown_acks.clear()
            if self._active:
                self._session_end_logged = False
                self.ended_timestamp = None

    def acknowledge_shutdown(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if payload.get("run_id") != self.run_id or payload.get("request_id") != self.shutdown_request_id or not self.shutdown_request_id:
                raise ValueError("Shutdown request does not match this recording")
            client_id = str(payload.get("client_id", ""))
            if client_id not in self._shutdown_clients or payload.get("pending") != 0:
                raise ValueError("Browser queue is not confirmed empty")
            last_id = payload.get("last_event_id")
            if not last_id or self._last_browser_event_ids.get(client_id) != last_id:
                raise ValueError("Final browser event has not been written")
            unresolved = payload.get("unresolved_events", [])
            if not isinstance(unresolved, list) or any(not isinstance(item, dict) for item in unresolved):
                raise ValueError("Invalid unresolved event backup")
            if unresolved:
                self.unresolved_browser_events[client_id] = unresolved
                path = self.session_dir / "pending_browser_events.json"
                temporary = path.with_suffix(".json.tmp")
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump({"run_id": self.run_id, "clients": self.unresolved_browser_events}, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(path)
            self._flush_files(sync=True)
            self._shutdown_acks.add(client_id)

    @property
    def shutdown_ready(self) -> bool:
        with self._lock:
            return bool(self.shutdown_request_id) and self._shutdown_clients <= self._shutdown_acks

    def _flush_files(self, *, sync: bool = False) -> None:
        for handle in (self._eeg_handle, self._event_handle, self._raw_handle, self._qc_handle):
            if handle is not None and not handle.closed:
                handle.flush()
                if sync:
                    os.fsync(handle.fileno())

    def _event_alignment(self, timestamp: float) -> dict[str, Any]:
        result = {"sample_time_sec": "", "eeg_received_order": "", "eeg_sample_number": "", "eeg_stream_segment": "",
                  "device_sample_number": "", "alignment_error_ms": "", "alignment_status": "no_eeg"}
        if not self._sample_timestamps:
            return result
        # A BLE burst shares one host receive timestamp. Those samples were produced
        # in the window ending at that instant, so an event landing on the tie belongs
        # with the most recent sample of the burst, not the oldest one.
        index = bisect_right(self._sample_timestamps, timestamp)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(self._sample_timestamps)]
        nearest = min(candidates, key=lambda i: (abs(self._sample_timestamps[i] - timestamp), -i))
        error = self._sample_timestamps[nearest] - timestamp
        result["alignment_error_ms"] = error * 1000.0
        # Do not invent an EEG position inside a dropout or far outside the recording.
        if abs(error) > 0.25:
            result["alignment_status"] = "outside_sample_tolerance"
            return result
        result.update(sample_time_sec=self._sample_numbers[nearest] / self.config.sample_rate_hz,
                      eeg_received_order=self._sample_orders[nearest],
                      eeg_sample_number=self._sample_numbers[nearest],
                      device_sample_number=self._sample_numbers[nearest],
                      eeg_stream_segment=self._sample_segments[nearest],
                      alignment_status="host_receive_nearest")
        return result

    @property
    def planned_sequence(self) -> list[str]:
        return [block["condition_label"] for block in self.planned_blocks]

    @property
    def planned_blocks(self) -> list[dict[str, Any]]:
        return planned_blocks(PROTOCOL_CONFIG, self.config.counterbalance_group)

    @staticmethod
    def session_half_for_block(block_id: int) -> int:
        if not 1 <= int(block_id) <= 6:
            return 0
        return 1 if int(block_id) <= 3 else 2

    @property
    def current_sample_time_sec(self) -> float:
        return self.device_sample_number / self.config.sample_rate_hz

    def condition_for_block(self, block_id: int) -> str:
        if not 1 <= block_id <= 6:
            raise ValueError("block_id must be between 1 and 6")
        return self.planned_sequence[block_id - 1]

    def condition_type_for_block(self, block_id: int) -> str:
        return CONDITIONS[self.condition_for_block(block_id)]["condition_type"]

    @property
    def qc_ready_for_experiment(self) -> bool:
        return self.baseline_complete and (self.baseline_passed or self.baseline_override)

    def start_baseline(self, duration_sec: float = 60.0) -> None:
        duration_sec = float(duration_sec)
        if duration_sec != float(PROTOCOL_CONFIG["flow"]["baseline_duration_sec"]):
            raise ValueError("正式协议的采前睁眼静息基线固定为60秒")
        with self._lock:
            if not self._active:
                raise RuntimeError("Session has not started")
            if not self.subtraction_practice.get("confirmed"):
                raise RuntimeError("必须先完成连续减7练习并由实验员确认")
            if self.current_block:
                raise RuntimeError("Cannot start baseline while a block is active")
            self.baseline_target_sec = duration_sec
            self.baseline_started_timestamp = self.clock_time()
            self.baseline_complete = False
            self.baseline_passed = False
            self.baseline_override = False
            self.baseline_result = {}
            self.baseline_rms_uv = [None, None]
            self._rms_instability_started_at = [None, None]
            self._qc_samples.clear()
            self.phase = "baseline"
            self.log_event("qc_baseline_start", duration_sec=duration_sec)

    def approve_baseline_override(self, operator_id: str, reason: str) -> None:
        with self._lock:
            if not self.baseline_complete:
                raise RuntimeError("Baseline QC has not completed")
            if self.baseline_passed:
                return
            operator_id = _safe_id(operator_id, "operator_id")
            reason = str(reason).strip()
            if not reason:
                raise ValueError("人工QC override必须填写原因")
            self.baseline_override = True
            self.baseline_override_details = {
                "operator_id": operator_id,
                "reason": reason,
                "timestamp": self.clock_time(),
            }
            self.log_event(
                "qc_baseline_override", operator_id=operator_id, override_reason=reason,
                notes=f"OVERRIDE: {reason}",
            )
            self._write_metadata("recording")

    def _initial_qc_status(self) -> dict[str, Any]:
        return {
            "qc_timestamp": self.clock_time(),
            "context": "baseline" if self.phase == "baseline" else "runtime",
            "status": "waiting",
            "status_label": "等待EEG数据",
            "messages": ["尚未收到足够的EEG数据。"],
            "sample_count": 0,
            "window_duration_sec": 0.0,
            "missing_packets": 0,
            "packet_loss_rate_pct": None,
            "data_age_sec": None,
            "signal_alive": False,
            "channels": [
                {"sample_count": 0, "saturation_rate_pct": None, "rms_uv": None,
                 "dc_offset_counts": None, "peak_abs_counts": None, "high_frequency_ratio_pct": None},
                {"sample_count": 0, "saturation_rate_pct": None, "rms_uv": None,
                 "dc_offset_counts": None, "peak_abs_counts": None, "high_frequency_ratio_pct": None},
            ],
        }

    def get_qc_status(self) -> dict[str, Any]:
        with self._lock:
            snapshot = json.loads(json.dumps(self._cached_qc or self._initial_qc_status()))
            now = self.clock_time()
            data_anchor = self._last_packet_timestamp or self.started_timestamp
            data_age = None if data_anchor is None else max(0.0, now - data_anchor)
            snapshot["data_age_sec"] = data_age
            snapshot["signal_alive"] = bool(data_age is not None and data_age < self.config.eeg_absence_alert_sec)
            if data_age is not None and data_age >= self.config.eeg_absence_alert_sec:
                snapshot["status"] = "bad"
                snapshot["status_label"] = "EEG中断"
                snapshot["messages"] = [f"已连续 {data_age:.1f} 秒没有EEG，请立即重新连接耳机。"]
            baseline_elapsed = 0.0
            if self.baseline_started_timestamp is not None:
                baseline_elapsed = max(0.0, now - self.baseline_started_timestamp)
            snapshot["baseline"] = {
                "started": self.baseline_started_timestamp is not None,
                "target_sec": self.baseline_target_sec,
                "elapsed_sec": baseline_elapsed,
                "recorded_sec": (
                    self.baseline_result.get("window_duration_sec", 0.0)
                    if self.baseline_complete
                    else snapshot.get("window_duration_sec", 0.0)
                ),
                "complete": self.baseline_complete,
                "passed": self.baseline_passed,
                "override": self.baseline_override,
                "ready_for_experiment": self.qc_ready_for_experiment,
            }
            return snapshot

    def update_qc_snapshot(self) -> dict[str, Any]:
        """Calculate and persist one periodic QC snapshot."""
        with self._lock:
            if not self._active:
                return self.get_qc_status()
            now = self.clock_time()
            baseline_active = self.phase == "baseline" and not self.baseline_complete
            if baseline_active and self.baseline_started_timestamp is not None:
                cutoff = self.baseline_started_timestamp
                context = "baseline"
            else:
                cutoff = now - self.config.qc_window_sec
                context = "runtime"
            samples = [row for row in self._qc_samples if row[0] >= cutoff]
            metrics = window_metrics(
                samples,
                sample_rate_hz=self.config.sample_rate_hz,
                scale_uv_per_count=self.config.eeg_scale_uv_per_count,
            )
            data_anchor = self._last_packet_timestamp or self.started_timestamp
            data_age = None if data_anchor is None else max(0.0, now - data_anchor)
            messages: list[str] = []
            status = "good"
            has_warning = False

            if data_age is None or metrics["sample_count"] < int(self.config.sample_rate_hz * 2):
                status = "waiting"
                messages.append("等待至少2秒EEG数据。")
            if data_age is not None and data_age >= self.config.eeg_absence_alert_sec:
                status = "bad"
                messages = [f"已连续 {data_age:.1f} 秒没有EEG，请立即重新连接耳机。"]
                if self.current_block and self._dropout_started_timestamp is None:
                    self._dropout_started_timestamp = data_anchor
                    self.phase = "paused"
                    self.log_event(
                        "eeg_dropout", event_value="signal_lost", pause_reason="eeg_dropout",
                        notes=f"absence_threshold_sec={self.config.eeg_absence_alert_sec:g}",
                    )

            loss = metrics["packet_loss_rate_pct"]
            if loss is not None and loss >= self.config.packet_loss_fail_pct:
                status = "bad"
                messages.append(f"丢包率 {loss:.2f}% 过高，请检查蓝牙距离、供电和连接。")
            elif loss is not None and loss >= self.config.packet_loss_warning_pct and status == "good":
                status = "warning"
                has_warning = True
                messages.append(f"丢包率 {loss:.2f}% 偏高。")

            for channel_index, channel in enumerate(metrics["channels"]):
                if channel["invalid_flatline"]:
                    status = "bad"
                    messages.append(f"通道{channel_index + 1}信号恒定、RMS接近数值零或连续至少{FLATLINE_DURATION_SEC:g}秒无变化，请检查数据流和电极。")
                saturation = channel["saturation_rate_pct"]
                if saturation is not None and saturation > 0:
                    status = "bad"
                    messages.append(f"通道{channel_index + 1}出现饱和（{saturation:.3f}%）。")

                dc_offset = channel["dc_offset_counts"]
                if dc_offset is not None and abs(dc_offset) >= self.config.extreme_dc_offset_counts:
                    status = "bad"
                    messages.append(f"通道{channel_index + 1}出现极端DC漂移，请检查电极接触。")
                peak = channel["peak_abs_counts"]
                if peak is not None and peak >= self.config.extreme_amplitude_counts:
                    status = "bad"
                    messages.append(f"通道{channel_index + 1}出现极端幅值。")
                high_frequency = channel["high_frequency_ratio_pct"]
                if high_frequency is not None and high_frequency >= self.config.high_frequency_ratio_fail_pct:
                    status = "bad"
                    messages.append(f"通道{channel_index + 1}高频污染严重（{high_frequency:.2f}%）。")
                elif high_frequency is not None and high_frequency >= self.config.high_frequency_ratio_warning_pct:
                    if status == "good":
                        status = "warning"
                    has_warning = True
                    messages.append(f"通道{channel_index + 1}高频污染偏高（{high_frequency:.2f}%）。")

                baseline_rms = self.baseline_rms_uv[channel_index]
                current_rms = channel["rms_uv"]
                if context == "runtime" and baseline_rms and current_rms is not None:
                    ratio = current_rms / baseline_rms
                    unstable = ratio < 0.2 or ratio > 5.0
                    if unstable:
                        if self._rms_instability_started_at[channel_index] is None:
                            self._rms_instability_started_at[channel_index] = now
                        if now - self._rms_instability_started_at[channel_index] >= 6.0:
                            status = "bad"
                            messages.append(
                                f"通道{channel_index + 1} RMS相对基线突变，可能接触不稳，请检查并重新连接。"
                            )
                        elif status == "good":
                            status = "warning"
                            has_warning = True
                            messages.append(f"通道{channel_index + 1} RMS相对基线异常，正在确认是否持续。")
                    else:
                        self._rms_instability_started_at[channel_index] = None

            if not messages and status == "good":
                messages.append("当前QC指标正常。")

            metrics.update({
                "qc_timestamp": now,
                "context": context,
                "signal_alive": bool(data_age is not None and data_age < self.config.eeg_absence_alert_sec),
                "status": status,
                "status_label": {"good": "正常", "warning": "注意", "bad": "异常", "waiting": "等待数据"}[status],
                "messages": messages,
                "data_age_sec": data_age,
            })

            if baseline_active and metrics["window_duration_sec"] >= self.baseline_target_sec:
                saturation_ok = all(
                    channel["saturation_rate_pct"] == 0 for channel in metrics["channels"]
                )
                loss_ok = loss is not None and loss < self.config.packet_loss_fail_pct
                self.baseline_complete = True
                self.baseline_passed = bool(
                    saturation_ok and loss_ok and status != "bad" and status != "waiting"
                    and not has_warning
                )
                self.baseline_result = json.loads(json.dumps(metrics))
                self.baseline_rms_uv = [channel["rms_uv"] for channel in metrics["channels"]]
                self.phase = "idle"
                self.log_event(
                    "qc_baseline_complete",
                    event_value="pass" if self.baseline_passed else "fail",
                    duration_sec=metrics["window_duration_sec"],
                    notes="; ".join(messages),
                )

            self._cached_qc = metrics
            self._write_qc_row(metrics)
            if baseline_active and self.baseline_complete:
                self._write_metadata("recording")
            if status != self._last_qc_status:
                self.log_event("qc_status_change", event_value=status, notes="; ".join(messages))
                self._last_qc_status = status
            return self.get_qc_status()

    def _write_qc_row(self, metrics: dict[str, Any]) -> None:
        if self._qc_writer is None:
            return
        channels = metrics["channels"]
        self._qc_writer.writerow({
            "qc_timestamp": metrics["qc_timestamp"],
            "sample_time_sec": self.current_sample_time_sec,
            "context": metrics["context"],
            "window_duration_sec": metrics["window_duration_sec"],
            "sample_count": metrics["sample_count"],
            "missing_packets": metrics["missing_packets"],
            "packet_loss_rate_pct": metrics["packet_loss_rate_pct"],
            "channel_0_saturation_rate_pct": channels[0]["saturation_rate_pct"],
            "channel_1_saturation_rate_pct": channels[1]["saturation_rate_pct"],
            "channel_0_rms_uv": channels[0]["rms_uv"],
            "channel_1_rms_uv": channels[1]["rms_uv"],
            "channel_0_dc_offset_counts": channels[0]["dc_offset_counts"],
            "channel_1_dc_offset_counts": channels[1]["dc_offset_counts"],
            "channel_0_peak_abs_counts": channels[0]["peak_abs_counts"],
            "channel_1_peak_abs_counts": channels[1]["peak_abs_counts"],
            "channel_0_high_frequency_ratio_pct": channels[0]["high_frequency_ratio_pct"],
            "channel_1_high_frequency_ratio_pct": channels[1]["high_frequency_ratio_pct"],
            "signal_alive": bool(metrics.get("signal_alive", False)),
            "status": metrics["status"],
            "messages": "; ".join(metrics["messages"]),
            **{f"channel_{index}_{field}": channel[field]
               for index, channel in enumerate(channels) for field in QC_SIGNAL_FIELDS},
        })
        self._qc_handle.flush()
        self.qc_snapshot_count += 1

    def start(self) -> Path:
        with self._lock:
            if self._active:
                return self.session_dir
            self.session_dir.mkdir(parents=True, exist_ok=False)
            self._eeg_handle = self.eeg_path.open(
                "w", newline="", encoding="utf-8", buffering=128 * 1024
            )
            self._event_handle = self.events_path.open(
                "w", newline="", encoding="utf-8", buffering=16 * 1024
            )
            self._raw_handle = self.raw_path.open("wb", buffering=128 * 1024)
            self._qc_handle = self.qc_path.open(
                "w", newline="", encoding="utf-8", buffering=16 * 1024
            )
            self._eeg_writer = csv.DictWriter(self._eeg_handle, fieldnames=EEG_CSV_COLUMNS)
            self._event_writer = csv.DictWriter(self._event_handle, fieldnames=EVENT_CSV_COLUMNS)
            self._qc_writer = csv.DictWriter(self._qc_handle, fieldnames=QC_CSV_COLUMNS)
            self._eeg_writer.writeheader()
            self._event_writer.writeheader()
            self._qc_writer.writeheader()
            self.started_timestamp = self.clock_time()
            self._active = True
            self._write_metadata(status="recording")
            self.log_event("session_start", event_value=self.config.experiment_name)
            return self.session_dir

    def _metadata(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": PROTOCOL_CONFIG["schema_version"],
            "software_version": PROTOCOL_CONFIG["software_version"],
            "protocol_version": PROTOCOL_CONFIG["protocol_version"],
            "protocol_config_version": PROTOCOL_CONFIG["config_version"],
            "protocol_config_sha256": PROTOCOL_CONFIG_HASH,
            "protocol_config_file": "config/protocol_v2.1.json",
            "run_id": self.run_id,
            "status": status,
            "experiment_complete": self.phase == "complete",
            "export_status": self.export_status,
            "epoch_status": self.epoch_status,
            "integrity_status": self.integrity_status,
            "unresolved_event_count": sum(len(items) for items in self.unresolved_browser_events.values()),
            "unresolved_events_file": "pending_browser_events.json" if self.unresolved_browser_events else None,
            "event_timing": {
                "clock": "Unix-anchored host monotonic clock",
                "sample_alignment": "nearest BLE receive timestamp, latest sample of a tied burst; not hardware-trigger precision",
                "max_alignment_distance_sec": 0.25,
                "received_order": "sample count when event received (legacy meaning)",
                "eeg_received_order": "zero-based EEG row linked to event occurrence",
                "missing_identity_policy": "legacy payload accepted and explicitly marked",
                "stream_segment": "increments on reconnect or receive gap >= one 8-bit counter period; do not join EEG windows across segments",
                "sample_time_status": "after a discontinuity packet-counter elapsed time is ambiguous; use EEG row, stream segment and received_timestamp",
            },
            "session": {
                **asdict(self.config),
                "output_root": str(Path(self.config.output_root)),
                "counterbalance_group": self.config.counterbalance_group,
                "study_phase": self.config.study_phase,
                "planned_blocks": self.planned_blocks,
                "planned_sequence": self.planned_sequence,
                "session_halves": [[1, 2, 3], [4, 5, 6]],
                "conditions": CONDITIONS,
                "b_start_numbers": self.b_start_numbers,
                "subtraction_practice": self.subtraction_practice,
                "rest_durations_after_block_sec": REST_DURATIONS_AFTER_BLOCK,
                "probe_config": self.probe_config,
                "probe_schedules": self.probe_schedules,
                "materials_version": PROTOCOL_CONFIG["materials"]["version"],
                "materials": PROTOCOL_CONFIG["materials"],
                "started_timestamp": self.started_timestamp,
                "ended_timestamp": self.ended_timestamp,
                "current_state": {
                    "block_id": self.current_block,
                    "block_order": self.current_block,
                    "session_half": self.current_session_half,
                    "video_id": self.current_video_id,
                    "condition_label": self.current_condition,
                    "phase": self.phase,
                    "completed_blocks": sorted(self.completed_blocks),
                },
            },
            "thought_probe": {
                "question": PROBE_QUESTION,
                "confidence_question": PROBE_CONFIDENCE_QUESTION,
                "options": PROBE_OPTIONS,
                "attention_map": PROBE_ATTENTION_MAP,
                "confidence_range": list(CONFIDENCE_CHOICES),
                "event_chain": ["probe_onset", "attention_response", "confidence_response", "video_resume"],
                "trigger": "video.currentTime; probe time does not advance while the video is paused",
                "response_policy": "stored exactly as answered; never corrected against the block condition",
            },
            "attention_epochs": _epoch_rules(),
            "labels": {
                "-1": "no_instantaneous_attention_label",
            },
            "condition_codes": {"-1": "no_condition", "0": "A", "1": "B"},
            "phases": [
                "idle", "baseline", "subtraction_practice", "prompt", "video", "paused", "thought_probe",
                "rating", "quiz", "inter_block", "rest", "post_video", "complete",
            ],
            "training_rules": {
                "window_length_sec": 4.0,
                "window_step_sec": 2.0,
                "exclude_video_start_sec": 10.0,
                "exclude_video_end_sec": 5.0,
                "instantaneous_attention_labels": "ON/OFF/AMBIGUOUS only in probe-confirmed epochs; NONE elsewhere",
                "condition_policy": "A/B and condition_type describe the block manipulation, not momentary attention truth.",
                "split_unit": "subject/session/block; never randomly split windows from one block",
            },
            "qc": {
                "interval_sec": self.config.qc_interval_sec,
                "rolling_window_sec": self.config.qc_window_sec,
                "baseline_target_sec": self.baseline_target_sec,
                "baseline_complete": self.baseline_complete,
                "baseline_passed": self.baseline_passed,
                "baseline_override": self.baseline_override,
                "baseline_override_details": self.baseline_override_details,
                "absence_alert_sec": self.config.eeg_absence_alert_sec,
                "acquisition_line_noise_measurement": False,
                "thresholds": PROTOCOL_CONFIG["qc"],
                "reject_reasons": [
                    "saturation", "flat", "missing", "extreme_amplitude",
                    "high_frequency_contamination", "motion", "motor_event",
                ],
                "packet_loss_warning_pct": self.config.packet_loss_warning_pct,
                "packet_loss_fail_pct": self.config.packet_loss_fail_pct,
                "saturation_definition": "absolute ADC count >= 99% of 24-bit full scale",
                "rms_definition": "demeaned time-domain RMS in scaled microvolts",
                "runtime_contact_rule": "RMS below 0.2x or above 5x baseline for 6 seconds",
                "flatline_duration_sec": FLATLINE_DURATION_SEC,
                "near_zero_rms_counts": NEAR_ZERO_RMS_COUNTS,
                "baseline_result": self.baseline_result,
                "dropouts": self.dropout_records,
            },
            "preprocessing": PROTOCOL_CONFIG["preprocessing"],
            "protocol_flow": PROTOCOL_CONFIG["flow"],
            "hardware_note": (
                "The experiment design mentions 4 channels, but the current BLE/OpenBCI bridge "
                "provides 2 real 24-bit EEG channels. No zero-padded channels are stored."
            ),
            "files": {
                "eeg_csv": self.eeg_path.name,
                "events_csv": self.events_path.name,
                "raw_packets": self.raw_path.name,
                "qc_csv": self.qc_path.name,
                "probe_epochs_csv": "probe_epochs.csv",
                "windows_csv": "windows.csv",
                "probes_csv": "probes.csv",
                "block_ratings_csv": "block_ratings.csv",
                "quiz_responses_csv": "quiz_responses.csv",
                "session_qc_report": "session_qc_report.json",
                "mat": self.mat_path.name,
            },
            "counts": {
                "received_samples": self.received_order,
                "missing_packets": self.total_missing_packets,
                "duplicate_packets": self.duplicate_packets,
                "saturated_samples": self.saturated_samples,
                "events": self.event_count,
                "qc_snapshots": self.qc_snapshot_count,
            },
        }

    def _write_metadata(self, status: str) -> None:
        temporary = self.metadata_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self._metadata(status), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.metadata_path)

    def record_packet(self, original_packet: bytes, normalized_packet: bytes) -> None:
        """Record one validated 33-byte packet and its two EEG channels."""
        with self._lock:
            if not self._active:
                return
            if len(original_packet) != 33 or len(normalized_packet) != 33:
                raise ValueError("EEG packets must be 33 bytes")

            received_timestamp = self.clock_time()
            sample_index = int(normalized_packet[1])
            packet_gap_before = 0
            quality_flags: list[str] = []
            discontinuity = self._stream_discontinuity or (
                self._last_packet_timestamp is not None and
                received_timestamp - self._last_packet_timestamp >= 256 / self.config.sample_rate_hz
            )
            if discontinuity and self.last_sample_index is not None:
                self.stream_segment += 1
                self._sample_time_status = "counter_elapsed_ambiguous"
                quality_flags.append("stream_discontinuity")
                self.last_sample_index = None
                self.device_sample_number += 1
            self._stream_discontinuity = False

            if self.last_sample_index is None:
                if not self.received_order:
                    self.device_sample_number = 0
            else:
                delta = (sample_index - self.last_sample_index) & 0xFF
                if delta == 0:
                    self.duplicate_packets += 1
                    quality_flags.append("duplicate_index")
                else:
                    self.device_sample_number += delta
                    packet_gap_before = max(delta - 1, 0)
                    if packet_gap_before:
                        self.total_missing_packets += packet_gap_before
                        quality_flags.append("packet_gap")
            self.last_sample_index = sample_index
            self._sample_timestamps.append(received_timestamp)
            self._sample_numbers.append(self.device_sample_number)
            self._sample_orders.append(self.received_order)
            self._sample_segments.append(self.stream_segment)

            if self._dropout_started_timestamp is not None:
                duration = max(0.0, received_timestamp - self._dropout_started_timestamp)
                dropout = {
                    "started_timestamp": self._dropout_started_timestamp,
                    "recovered_timestamp": received_timestamp,
                    "duration_sec": duration,
                    "block_id": self.current_block or "",
                    "video_id": self.current_video_id,
                    "device_sample_number": self.device_sample_number,
                }
                self.dropout_records.append(dropout)
                self._dropout_started_timestamp = None
                self.log_event(
                    "eeg_recovery", event_value="signal_recovered", duration_sec=duration,
                    pause_reason="eeg_dropout", notes="等待实验员重新确认同步后恢复视频",
                )

            channel_0_raw = parse_24bit_signed(normalized_packet[2:5])
            channel_1_raw = parse_24bit_signed(normalized_packet[5:8])
            saturation_threshold = ADC_LIMIT * SATURATION_FRACTION
            if abs(channel_0_raw) >= saturation_threshold or abs(channel_1_raw) >= saturation_threshold:
                self.saturated_samples += 1
                quality_flags.append("adc_saturation")

            self._last_packet_timestamp = received_timestamp
            self._qc_samples.append(
                (received_timestamp, channel_0_raw, channel_1_raw, packet_gap_before)
            )

            condition = self.current_condition if self.phase == "video" else ""
            condition_type = self.current_condition_type if condition else ""
            condition_code = CONDITION_CODES[condition]
            # A/B is an experimental condition, not an instantaneous attention label.
            weak_label = -1
            base_valid = int(self.phase == "video" and not quality_flags)

            self._eeg_writer.writerow({
                "received_timestamp": received_timestamp,
                "received_order": self.received_order,
                "device_sample_number": self.device_sample_number,
                "sample_index": sample_index,
                "sample_time_sec": self.current_sample_time_sec,
                "packet_gap_before": packet_gap_before,
                "block_id": self.current_block if self.current_block else "",
                "group_id": "",
                "block_order": self.current_block if self.current_block else "",
                "session_half": self.current_session_half if self.current_session_half else "",
                "counterbalance_group": self.config.counterbalance_group,
                "video_id": self.current_video_id if self.current_block else "",
                "condition": condition,
                "condition_label": condition,
                "condition_type": condition_type,
                "condition_code": condition_code,
                "weak_label": weak_label,
                "phase": self.phase,
                "base_valid_for_training": base_valid,
                "channel_0_raw": channel_0_raw,
                "channel_1_raw": channel_1_raw,
                "channel_0_uv": channel_0_raw * self.config.eeg_scale_uv_per_count,
                "channel_1_uv": channel_1_raw * self.config.eeg_scale_uv_per_count,
                "quality_flag": ";".join(quality_flags) if quality_flags else "ok",
                "is_formal_experiment": int(self.phase == "video"),
                "stream_segment": self.stream_segment,
                "sample_time_status": self._sample_time_status,
            })
            self._raw_handle.write(original_packet)
            self.received_order += 1

            flush_interval = max(1, int(round(self.config.sample_rate_hz)))
            if self.received_order % flush_interval == 0:
                self._eeg_handle.flush()
                self._event_handle.flush()
                self._raw_handle.flush()

    def log_event(
        self,
        event_type: str,
        *,
        event_value: Any = "",
        duration_sec: Any = "",
        exclude_before_sec: Any = 0.0,
        exclude_after_sec: Any = 0.0,
        trial_id: Any = "",
        question_id: Any = "",
        correct_answer: Any = "",
        participant_answer: Any = "",
        is_correct: Any = "",
        reaction_time_ms: Any = "",
        group_id: Any = "",
        start_number: Any = "",
        generated_timestamp: Any = "",
        displayed_timestamp: Any = "",
        hidden_timestamp: Any = "",
        after_block_id: Any = "",
        planned_duration: Any = "",
        actual_duration: Any = "",
        is_formal_experiment: Any = "",
        video_id: Any = None,
        video_time_sec: Any = None,
        probe_id: Any = "",
        probe_index: Any = "",
        probe_schedule_id: Any = "",
        probe_planned_time_sec: Any = "",
        response: Any = "",
        response_label: Any = "",
        response_time_ms: Any = "",
        confidence: Any = "",
        probe_attention: Any = "",
        pause_reason: Any = "",
        operator_id: Any = "",
        override_reason: Any = "",
        sensitivity_exclude_before_sec: Any = "",
        course_attention_rating: Any = "",
        mental_effort: Any = "",
        video_interest: Any = "",
        video_difficulty: Any = "",
        subtraction_compliance: Any = "",
        subtraction_compliance_low_pct: Any = "",
        subtraction_compliance_high_pct: Any = "",
        subtraction_difficulty: Any = "",
        reported_final_number: Any = "",
        rating_item: Any = "",
        rating_value: Any = "",
        quiz_score: Any = "",
        quiz_total: Any = "",
        question_text: Any = "",
        response_index: Any = "",
        response_text: Any = "",
        notes: str = "",
    ) -> None:
        if _is_legacy_arithmetic_event(event_type):
            raise ValueError("Legacy screen-arithmetic events are not recorded in new v2.1 sessions")
        with self._lock:
            if not self._active:
                return
            received_timestamp = self.clock_time()
            timestamp = self._event_context.get("event_timestamp", received_timestamp)
            alignment = self._event_alignment(timestamp)
            context = self._event_context
            block_value = context.get("block_id", self.current_block or "")
            condition_value = context.get("condition", self.current_condition)
            condition_type = context.get(
                "condition_type", CONDITIONS.get(condition_value, {}).get("condition_type", "")
            )
            if is_formal_experiment == "":
                is_formal_experiment = int(bool(block_value) and event_type not in {
                    "subtraction_practice_start", "subtraction_practice_end", "practice_confirmed",
                    "rest_start", "rest_end",
                })
            self._event_writer.writerow({
                "event_name": str(event_type),
                "event_timestamp": timestamp,
                **alignment,
                "received_order": self.received_order,
                "block_id": block_value,
                "block_order": block_value,
                "group_id": "",
                "session_half": self.session_half_for_block(int(block_value)) if str(block_value).isdigit() else "",
                "counterbalance_group": self.config.counterbalance_group,
                "condition": condition_value,
                "condition_type": condition_type,
                "event_type": str(event_type),
                "event_value": event_value,
                "duration_sec": duration_sec,
                "exclude_before_sec": exclude_before_sec,
                "exclude_after_sec": exclude_after_sec,
                "trial_id": trial_id,
                "question_id": question_id,
                "correct_answer": correct_answer,
                "participant_answer": participant_answer,
                "is_correct": is_correct,
                "reaction_time_ms": reaction_time_ms,
                "notes": notes,
                "start_number": start_number,
                "generated_timestamp": generated_timestamp,
                "displayed_timestamp": displayed_timestamp,
                "hidden_timestamp": hidden_timestamp,
                "after_block_id": after_block_id,
                "planned_duration": planned_duration,
                "actual_duration": actual_duration,
                "is_formal_experiment": is_formal_experiment,
                "video_id": (context.get("video_id") or self.current_video_id or "") if video_id is None else video_id,
                "condition_label": condition_value,
                "video_time_sec": context.get("video_time_sec", "") if video_time_sec is None else video_time_sec,
                "browser_timestamp": context.get("client_timestamp", ""),
                "recorder_timestamp": received_timestamp,
                "probe_id": probe_id,
                "probe_index": probe_index,
                "probe_schedule_id": probe_schedule_id,
                "probe_planned_time_sec": probe_planned_time_sec,
                "response": response,
                "response_label": response_label,
                "response_time_ms": response_time_ms,
                "confidence": confidence,
                "probe_attention": probe_attention,
                "pause_reason": pause_reason,
                "operator_id": operator_id,
                "override_reason": override_reason,
                "sensitivity_exclude_before_sec": sensitivity_exclude_before_sec,
                "course_attention_rating": course_attention_rating,
                "mental_effort": mental_effort,
                "video_interest": video_interest,
                "video_difficulty": video_difficulty,
                "subtraction_compliance": subtraction_compliance,
                "subtraction_compliance_low_pct": subtraction_compliance_low_pct,
                "subtraction_compliance_high_pct": subtraction_compliance_high_pct,
                "subtraction_difficulty": subtraction_difficulty,
                "reported_final_number": reported_final_number,
                "rating_item": rating_item,
                "rating_value": rating_value,
                "quiz_score": quiz_score,
                "quiz_total": quiz_total,
                "question_text": question_text,
                "response_index": response_index,
                "response_text": response_text,
                "run_id": self.run_id,
                "subject_id": self.config.subject_id,
                "session_id": self.config.session_id,
                "received_timestamp": context.get("received_timestamp", received_timestamp),
                "timestamp_source": context.get("timestamp_source", "server"),
                **{key: context.get(key, "") for key in (
                    "client_event_id", "client_id", "client_timestamp", "client_monotonic_ms",
                    "clock_offset_sec", "clock_round_trip_ms", "identity_status",
                    "context_matches_current", "state_applied",
                )},
            })
            self.event_count += 1
            self._event_handle.flush()

    def begin_block(self, block_id: int, video_id: str) -> str:
        with self._lock:
            if self.current_block:
                raise RuntimeError("已有Block正在进行，不能重复开始")
            if self.config.study_phase != "smoke":
                expected_next = len(self.completed_blocks) + 1
                if block_id != expected_next:
                    raise RuntimeError(f"不能跳步：下一个必须是Block {expected_next}")
            if not self.qc_ready_for_experiment:
                raise RuntimeError(
                    "Pre-experiment baseline QC is incomplete or has unresolved warnings"
                )
            if not self.subtraction_practice.get("confirmed"):
                raise RuntimeError("连续减7练习尚未确认")
            if self.get_qc_status().get("status") == "bad":
                raise RuntimeError("Current EEG QC is abnormal; reconnect/check electrodes before starting")
            condition = self.condition_for_block(block_id)
            video_id = _safe_id(video_id, "video_id")
            expected_video = self.planned_blocks[block_id - 1]["video_id"]
            if video_id != expected_video:
                raise ValueError(
                    f"Block {block_id} must use {expected_video} in {self.config.counterbalance_group}, got {video_id}"
                )
            self.current_block = block_id
            self.current_session_half = self.session_half_for_block(block_id)
            self.current_condition = condition
            self.current_condition_type = CONDITIONS[condition]["condition_type"]
            self.current_video_id = video_id
            self.phase = "prompt"
            number = self.b_start_numbers.get(block_id, {})
            self.log_event(
                "block_start", event_value=video_id,
                start_number=number.get("start_number", ""),
                generated_timestamp=number.get("generated_timestamp", ""),
                notes=f"condition={condition}; condition_type={self.current_condition_type}",
            )
            self.log_event("condition_prompt_start", event_value=condition)
            return condition

    def start_video(self) -> None:
        with self._lock:
            if not self.current_block:
                raise RuntimeError("Start a block before starting its video")
            self.phase = "video"
            self.log_event("video_start", event_value=self.current_video_id)

    def end_video(self) -> None:
        with self._lock:
            if self.phase not in {"video", "paused", "thought_probe"}:
                raise RuntimeError("No video is currently active")
            self.log_event("video_end", event_value=self.current_video_id)
            self.phase = "post_video"

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase
            self.log_event("phase_change", event_value=phase)

    def log_attention_rating(self, score: int) -> None:
        if score not in {1, 2, 3, 4, 5}:
            raise ValueError("attention score must be between 1 and 5")
        with self._lock:
            self.phase = "rating"
            self.log_event("attention_rating", event_value=score)

    def log_summary(
        self,
        event_type: str,
        correct: int,
        total: int,
        mean_reaction_time_ms: Any = "",
        notes: str = "",
    ) -> None:
        if total < 0 or correct < 0 or correct > total:
            raise ValueError("correct/total values are invalid")
        self.log_event(
            event_type,
            event_value=f"{correct}/{total}",
            is_correct=(correct / total if total else ""),
            reaction_time_ms=mean_reaction_time_ms,
            notes=notes,
        )

    def mark_artifact(self, artifact_type: str, notes: str = "") -> None:
        self.log_event(
            "artifact",
            event_value=artifact_type,
            exclude_before_sec=1.5,
            exclude_after_sec=1.5,
            notes=notes,
        )

    def log_external_event(self, payload: dict[str, Any]) -> None:
        """Log a supported external event through the HTTP trigger endpoint."""
        allowed = {
            "event_value", "duration_sec", "exclude_before_sec", "exclude_after_sec",
            "trial_id", "question_id", "correct_answer", "participant_answer",
            "is_correct", "reaction_time_ms", "notes",
            "start_number", "generated_timestamp", "displayed_timestamp", "hidden_timestamp",
            "after_block_id", "planned_duration", "actual_duration", "is_formal_experiment",
            "video_id", "video_time_sec", "pause_reason",
            "operator_id", "override_reason", "sensitivity_exclude_before_sec",
            "course_attention_rating", "mental_effort", "video_interest", "video_difficulty",
            "subtraction_compliance", "subtraction_compliance_low_pct", "subtraction_compliance_high_pct",
            "subtraction_difficulty", "reported_final_number", "rating_item", "rating_value",
            "quiz_score", "quiz_total", "question_text", "response_index", "response_text",
            *PROBE_EVENT_COLUMNS,
        }
        event_type = str(payload.get("event_type", "external_trigger"))
        if _is_legacy_arithmetic_event(event_type):
            raise ValueError("Legacy screen-arithmetic events are not recorded in new v2.1 sessions")
        kwargs = {key: payload[key] for key in allowed if key in payload}
        self.log_event(event_type, **kwargs)

    @staticmethod
    def _browser_block_id(value: Any) -> int:
        match = re.search(r"(\d+)", str(value))
        if not match:
            raise ValueError(f"Cannot parse block_id from {value!r}")
        block_id = int(match.group(1))
        if not 1 <= block_id <= 6:
            raise ValueError("Browser block_id must be between 1 and 6")
        return block_id

    def handle_browser_event(self, payload: dict[str, Any]) -> bool:
        """Process one browser event exactly once, even if HTTP retries it."""
        client_event_id = str(payload.get("client_event_id", "")).strip()
        with self._lock:
            if not self._active:
                raise RuntimeError("Recording has already closed")
            run_id = str(payload.get("recorder_run_id") or payload.get("run_id") or "")
            if run_id and run_id not in {self.run_id, self.session_dir.name}:
                raise ValueError("Event run_id does not match this recording")
            for key, expected in (("subject_id", self.config.subject_id), ("session_id", self.config.session_id)):
                if payload.get(key) and str(payload[key]).strip() != expected:
                    raise ValueError(f"Event {key} does not match this recording")
            if payload.get("counterbalance_group") and str(payload["counterbalance_group"]).upper() != self.config.counterbalance_group:
                raise ValueError("Event counterbalance_group does not match this recording")
            if client_event_id and client_event_id in self._seen_browser_event_ids:
                return False
            client_id = str(payload.get("client_id", ""))
            if client_id in self._shutdown_acks:
                raise ValueError("Browser already confirmed its final event")
            received = self.clock_time()
            timestamp = received
            source = "server_fallback"
            client_timestamp = payload.get("client_timestamp", "")
            if payload.get("event_timestamp") is not None:
                timestamp = float(payload["event_timestamp"])
                source = "browser_calibrated" if payload.get("clock_offset_sec") is not None else "browser_unverified"
            elif payload.get("iso_time"):
                parsed = datetime.fromisoformat(str(payload["iso_time"]).replace("Z", "+00:00"))
                # Historical browser exports use ISO UTC. Reject ambiguous local timestamps.
                if parsed.tzinfo is None:
                    raise ValueError("Event ISO timestamp must include a timezone")
                timestamp = parsed.timestamp()
                client_timestamp = timestamp
                source = "browser_iso_unverified"
            if not math.isfinite(timestamp):
                raise ValueError("Event timestamp must be finite")
            block_id = self._browser_block_id(payload["block_id"]) if payload.get("block_id") else self.current_block
            condition = str(payload.get("condition") or (self.condition_for_block(block_id) if block_id else self.current_condition))
            event_type = str(payload.get("event_type", "external_trigger"))
            if _is_legacy_arithmetic_event(event_type):
                raise ValueError("Legacy screen-arithmetic events are not recorded in new v2.1 sessions")
            matches = block_id == self.current_block
            state_events = {"block_start", "video_play", "video_resume", "video_pause", "video_ended", "block_end",
                            "rating_start", "rating_item_response", "rating_end",
                            "attention_rating_submit", "quiz_start", "quiz_item_response", "quiz_end",
                            "quiz_response", "quiz_submit", "rest_start", "rest_end", "self_caught",
                            "subtraction_practice_start", "subtraction_practice_end", "practice_confirmed",
                            *PROBE_EVENTS}
            # Debug block selection remains unrestricted. Delayed events are retained
            # under their source block, but cannot rewind the current live state.
            applies = timestamp >= self._last_browser_state_timestamp and (
                matches or event_type in {"block_start", "rest_start", "rest_end",
                                          "subtraction_practice_start", "subtraction_practice_end", "practice_confirmed"}
            )
            self._event_context = {
                "event_timestamp": timestamp, "received_timestamp": received,
                "timestamp_source": source, "client_timestamp": client_timestamp,
                "block_id": block_id or "", "group_id": "",
                "condition": condition, "client_event_id": client_event_id, "client_id": client_id,
                "condition_type": payload.get("condition_type") or CONDITIONS.get(condition, {}).get("condition_type", ""),
                "video_id": payload.get("video_id") or payload.get("topic_key") or "",
                "video_time_sec": payload.get("video_time_sec", ""),
                "identity_status": "verified" if run_id == self.run_id and payload.get("subject_id") and payload.get("session_id") else "legacy_partial",
                "context_matches_current": int(matches), "state_applied": int(applies and event_type in state_events),
                **{key: payload.get(key, "") for key in ("client_monotonic_ms", "clock_offset_sec", "clock_round_trip_ms")},
            }
            self._browser_apply_state = applies
            try:
                self._handle_browser_event_impl(payload)
                if applies and event_type in state_events:
                    self._last_browser_state_timestamp = timestamp
            finally:
                self._event_context = {}
                self._browser_apply_state = True
            if client_event_id:
                self._seen_browser_event_ids.add(client_event_id)
            if client_id:
                self._last_browser_event_ids[client_id] = client_event_id
                self.register_browser(client_id)
            return True

    def _handle_browser_event_impl(self, payload: dict[str, Any]) -> None:
        """Synchronize browser experiment events with EEG labels and event rows."""
        browser_subject = str(payload.get("subject_id", "")).strip()
        if browser_subject and browser_subject != self.config.subject_id:
            raise ValueError(
                f"Browser subject_id={browser_subject!r} does not match active EEG "
                f"session {self.config.subject_id!r}"
            )
        browser_session = str(payload.get("session_id", "")).strip()
        if browser_session and browser_session != self.config.session_id:
            raise ValueError(
                f"Browser session_id={browser_session!r} does not match active EEG "
                f"session {self.config.session_id!r}"
            )

        event_type = str(payload.get("event_type", "external_trigger"))
        compact_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        if event_type == "self_caught":
            condition = self._event_context.get("condition", self.current_condition)
            if condition not in CONDITIONS or not self.current_block:
                raise ValueError("self_caught is only valid during an active A/B block")
            self.log_event(
                "self_caught",
                event_value=(
                    "noticed_attention_left_course_return_to_video" if condition == "A"
                    else "noticed_subtraction_stopped_restart_subtraction"
                ),
                exclude_before_sec=PROTOCOL_CONFIG["flow"]["self_caught_motor_mask_before_sec"],
                exclude_after_sec=PROTOCOL_CONFIG["flow"]["self_caught_motor_mask_after_sec"],
                sensitivity_exclude_before_sec=PROTOCOL_CONFIG["flow"]["self_caught_sensitivity_before_sec"],
                notes=compact_payload,
            )
            return

        if event_type == "subtraction_practice_start":
            self.subtraction_practice["started"] = True
            self.subtraction_practice["started_timestamp"] = self._event_context.get("event_timestamp")
            if self._browser_apply_state and self.phase == "idle":
                self.phase = "subtraction_practice"
            self.log_event(
                event_type,
                event_value=("confirmed" if event_type == "practice_confirmed" else
                             payload.get("practice_number", self.subtraction_practice["practice_number"])),
                is_formal_experiment=0,
                notes=compact_payload,
            )
            self._write_metadata("recording")
            return

        if event_type in {"subtraction_practice_end", "practice_confirmed"}:
            if event_type == "subtraction_practice_end":
                self.subtraction_practice["ended_timestamp"] = self._event_context.get("event_timestamp")
            else:
                self.subtraction_practice["confirmed"] = True
            if self._browser_apply_state and self.phase == "subtraction_practice":
                self.phase = "idle"
            self.log_event(
                event_type,
                event_value=payload.get("practice_number", self.subtraction_practice["practice_number"]),
                is_formal_experiment=0,
                notes=compact_payload,
            )
            self._write_metadata("recording")
            return

        if event_type in {"subtraction_start_number_displayed", "subtraction_start_number_hidden"}:
            self.log_event(
                event_type,
                event_value=payload.get("start_number", ""),
                start_number=payload.get("start_number", ""),
                generated_timestamp=payload.get("generated_timestamp", ""),
                displayed_timestamp=payload.get("displayed_timestamp", ""),
                hidden_timestamp=payload.get("hidden_timestamp", ""),
                notes=compact_payload,
            )
            return

        if event_type in PROBE_EVENTS:
            self._handle_probe_event(event_type, payload, compact_payload)
            return

        if event_type == "block_start":
            block_id = self._browser_block_id(payload.get("block_id", ""))
            condition = str(payload.get("condition", "")).upper()
            expected = self.condition_for_block(block_id)
            if condition != expected:
                raise ValueError(
                    f"Block {block_id} should use condition {expected} for "
                    f"{self.config.counterbalance_group}, but browser sent {condition}"
                )
            video_id = str(payload.get("topic_key") or payload.get("topic_name") or f"V{block_id}")
            if self._browser_apply_state:
                self.begin_block(block_id, video_id)
            else:
                self.log_event("block_start", event_value=video_id, notes="Delayed event; live state unchanged")
            self.log_event("browser_block_config", event_value=video_id, notes=compact_payload)
            return

        if event_type in {"video_play", "video_resume"}:
            pause_reason = str(payload.get("pause_reason", ""))
            probe_id = str(payload.get("probe_id", "")).strip()
            if pause_reason == "thought_probe":
                if self._probe_stages.get(probe_id) != "confidence":
                    raise ValueError("A probe resume must follow its confidence_response")
                self._probe_stages[probe_id] = "resumed"
            if self._browser_apply_state:
                if event_type == "video_play" and self.phase != "video":
                    self.start_video()
                else:
                    self.set_phase("video")
            elif event_type == "video_play":
                self.log_event("video_start", event_value=payload.get("topic_key", ""))
            self.log_event(
                event_type, event_value=payload.get("video_time_sec", ""),
                probe_id=probe_id, pause_reason=pause_reason, notes=compact_payload,
            )
            return

        if event_type == "video_pause":
            if self._browser_apply_state:
                self.set_phase("paused")
            self.log_event(
                event_type, event_value=payload.get("video_time_sec", ""),
                pause_reason=payload.get("pause_reason", ""), notes=compact_payload,
            )
            return

        if event_type in {"video_ended", "block_end"}:
            if self._browser_apply_state and self.phase in {"video", "paused"}:
                self.end_video()
            elif not self._browser_apply_state and event_type == "video_ended":
                self.log_event("video_end", event_value=payload.get("topic_key", ""))
            self.log_event(f"browser_{event_type}", event_value=payload.get("reason", ""), notes=compact_payload)
            return

        if event_type == "rating_start":
            if self._browser_apply_state:
                self.phase = "rating"
            self.log_event("rating_start", notes=compact_payload)
            return

        if event_type == "rating_item_response":
            item = str(payload.get("rating_item", ""))
            value = payload.get("rating_value", "")
            limits = {
                "course_attention_rating": (1, 5), "mental_effort": (1, 9),
                "video_interest": (1, 5), "video_difficulty": (1, 5),
                "subtraction_difficulty": (1, 5),
            }
            if item in limits:
                numeric = int(value)
                if not limits[item][0] <= numeric <= limits[item][1]:
                    raise ValueError(f"{item} is outside its allowed range")
            self.log_event(
                "rating_item_response", event_value=value, rating_item=item, rating_value=value,
                response_time_ms=payload.get("response_time_ms", ""), notes=compact_payload,
            )
            return

        if event_type == "rating_end":
            attention = int(payload.get("course_attention_rating"))
            effort = int(payload.get("mental_effort"))
            interest = int(payload.get("video_interest"))
            difficulty = int(payload.get("video_difficulty"))
            if attention not in range(1, 6) or effort not in range(1, 10) or interest not in range(1, 6) or difficulty not in range(1, 6):
                raise ValueError("A rating field is outside its allowed range")
            condition = self._event_context.get("condition", self.current_condition)
            extra = {}
            if condition == "B":
                compliance = str(payload.get("subtraction_compliance", "")).strip()
                subtraction_difficulty = int(payload.get("subtraction_difficulty"))
                final_number = int(payload.get("reported_final_number"))
                allowed_compliance = {f"{low}-{100 if low == 90 else low + 10}%" for low in range(0, 100, 10)}
                if compliance not in allowed_compliance:
                    raise ValueError("subtraction_compliance must be a documented 10% interval")
                if subtraction_difficulty not in range(1, 6):
                    raise ValueError("subtraction_difficulty must be 1 through 5")
                low, high = [int(value) for value in re.findall(r"\d+", compliance)]
                extra = {
                    "subtraction_compliance": compliance,
                    "subtraction_compliance_low_pct": low,
                    "subtraction_compliance_high_pct": high,
                    "subtraction_difficulty": subtraction_difficulty,
                    "reported_final_number": final_number,
                }
            self.log_event(
                "rating_end", course_attention_rating=attention, mental_effort=effort,
                video_interest=interest, video_difficulty=difficulty,
                duration_sec=payload.get("duration_sec", ""), notes=compact_payload, **extra,
            )
            return

        if event_type == "attention_rating_submit":
            score = int(payload.get("rating"))
            if self._browser_apply_state:
                self.log_attention_rating(score)
            else:
                self.log_event("attention_rating", event_value=score)
            self.log_event("browser_attention_detail", notes=compact_payload)
            return

        if event_type == "quiz_start":
            if self._browser_apply_state:
                self.phase = "quiz"
            self.log_event("quiz_start", notes=compact_payload)
            return

        if event_type == "quiz_item_response":
            if self._browser_apply_state and self.phase != "quiz":
                self.set_phase("quiz")
            self.log_event(
                "quiz_item_response", event_value=payload.get("question_text", ""),
                question_id=payload.get("question_id", ""), question_text=payload.get("question_text", ""),
                correct_answer=payload.get("correct_letter", ""), participant_answer=payload.get("response_letter", ""),
                response_index=payload.get("response_index", ""), response_text=payload.get("response_text", ""),
                is_correct=payload.get("is_correct", ""), reaction_time_ms=payload.get("reaction_time_ms", ""),
                notes=compact_payload,
            )
            return

        if event_type == "quiz_end":
            score = int(payload.get("score", 0))
            total = int(payload.get("total", 0))
            if total != 4 or not 0 <= score <= total:
                raise ValueError("Every block quiz must contain exactly four questions")
            self.log_event(
                "quiz_end", event_value=f"{score}/{total}", quiz_score=score, quiz_total=total,
                duration_sec=payload.get("duration_sec", ""), notes=compact_payload,
            )
            if self._browser_apply_state and self.current_block:
                self.end_block()
            return

        if event_type == "quiz_response":
            if self._browser_apply_state and self.phase != "quiz":
                self.set_phase("quiz")
            self.log_event(
                "course_response",
                event_value=payload.get("question_text", ""),
                question_id=payload.get("question_id", ""),
                correct_answer=payload.get("correct_letter", ""),
                participant_answer=payload.get("response_letter", ""),
                is_correct=payload.get("is_correct", ""),
                notes=compact_payload,
            )
            return

        if event_type == "quiz_submit":
            if self._browser_apply_state and self.phase != "quiz":
                self.set_phase("quiz")
            self.log_summary(
                "course_quiz_summary",
                int(payload.get("score", 0)),
                int(payload.get("total", 0)),
                notes=compact_payload,
            )
            if self._browser_apply_state and self.current_block:
                self.end_block()
            return

        if event_type in {"rest_start", "rest_end"}:
            after_block_id = int(payload.get("after_block_id") or payload.get("block_id") or 0)
            if after_block_id not in REST_DURATIONS_AFTER_BLOCK:
                raise ValueError("Rest is only configured after blocks 1 through 5")
            planned_rest = REST_DURATIONS_AFTER_BLOCK[after_block_id]
            if event_type == "rest_end":
                actual = float(payload.get("actual_duration", payload.get("duration_sec", 0)) or 0)
                if actual + 0.25 < planned_rest and self.config.study_phase != "smoke":
                    raise ValueError("pilot/formal模式不能提前结束休息")
            if self._browser_apply_state and event_type == "rest_start":
                self.phase = "rest"
            self.log_event(
                event_type,
                event_value=f"after-block-{after_block_id}",
                duration_sec=payload.get("actual_duration", payload.get("duration_sec", "")),
                group_id="",
                after_block_id=after_block_id,
                planned_duration=payload.get(
                    "planned_duration", planned_rest
                ),
                actual_duration=payload.get("actual_duration", ""),
                is_formal_experiment=0,
                notes=compact_payload,
            )
            if self._browser_apply_state and event_type == "rest_end":
                self.phase = "idle"
            return

        self.log_event(
            event_type,
            event_value=payload.get("event_value", payload.get("video_time_sec", "")),
            notes=compact_payload,
        )

    def _handle_probe_event(self, event_type: str, payload: dict[str, Any], compact_payload: str) -> None:
        """Record one thought-probe step and enforce the probe event chain."""
        probe_block = self._browser_block_id(payload["block_id"]) if payload.get("block_id") else self.current_block
        schedule_id = str(payload.get("probe_schedule_id", "")).strip()

        if event_type == "probe_schedule":
            if not probe_block:
                raise ValueError("A probe schedule must name its block")
            times, config = validate_probe_schedule(
                payload.get("probe_times", []),
                payload.get("video_duration_sec"),
                payload.get("probe_config"),
            )
            self.probe_schedules[probe_block] = {
                "probe_schedule_id": schedule_id,
                "block_id": probe_block,
                "video_id": payload.get("video_id") or payload.get("topic_key", ""),
                "video_duration_sec": float(payload["video_duration_sec"]),
                "probe_times_sec": times,
                "probe_config": config,
                "uses_protocol_default_config": config == PROBE_CONFIG,
                "generated_timestamp": self._event_context.get("event_timestamp"),
            }
            self.log_event(
                "probe_schedule",
                event_value=",".join(f"{value:g}" for value in times),
                probe_schedule_id=schedule_id,
                probe_index=len(times),
                notes=compact_payload,
            )
            self._write_metadata("recording")
            return

        probe_id = str(payload.get("probe_id", "")).strip()
        if not probe_id:
            raise ValueError("Thought probe events require a probe_id")
        stage = self._probe_stages.get(probe_id)
        common = {
            "probe_id": probe_id,
            "probe_index": payload.get("probe_index", ""),
            "probe_schedule_id": schedule_id,
            "probe_planned_time_sec": payload.get("probe_planned_time_sec", ""),
            "notes": compact_payload,
        }

        if event_type == "probe_onset":
            if stage is not None:
                raise ValueError(f"Probe {probe_id} already has an onset")
            self._probe_stages[probe_id] = "onset"
            if self._browser_apply_state and self.phase in {"video", "paused"}:
                self.phase = "thought_probe"
            self.log_event("probe_onset", event_value=probe_id, pause_reason="thought_probe", **common)
            return

        if event_type == "attention_response":
            if stage != "onset":
                raise ValueError(f"attention_response for {probe_id} must follow its probe_onset")
            response = int(payload["response"])
            if response not in PROBE_ATTENTION_MAP:
                raise ValueError("Probe response must be one of options 1 through 4")
            # Stored as answered. B-task answers under condition A are kept unchanged.
            attention = PROBE_ATTENTION_MAP[response]
            self._probe_stages[probe_id] = "attention"
            self._record_probe_answer(probe_block, probe_id, response=response, probe_attention=attention)
            self.log_event(
                "attention_response",
                event_value=response,
                response=response,
                response_label=PROBE_OPTIONS[response],
                probe_attention=attention,
                response_time_ms=payload.get("response_time_ms", ""),
                **common,
            )
            return

        if event_type == "confidence_response":
            if stage != "attention":
                raise ValueError(f"confidence_response for {probe_id} must follow its attention_response")
            confidence = int(payload["confidence"])
            if confidence not in CONFIDENCE_CHOICES:
                raise ValueError("Probe confidence must be an integer between 1 and 4")
            self._probe_stages[probe_id] = "confidence"
            self._record_probe_answer(probe_block, probe_id, confidence=confidence)
            self.log_event(
                "confidence_response",
                event_value=confidence,
                confidence=confidence,
                response_time_ms=payload.get("response_time_ms", ""),
                **common,
            )
            return

        # probe_cancelled: the block stopped before the chain completed.
        if stage is None:
            raise ValueError(f"Probe {probe_id} was never started")
        self._probe_stages[probe_id] = "cancelled"
        if self._browser_apply_state and self.phase == "thought_probe":
            self.phase = "paused"
        self.log_event("probe_cancelled", event_value=payload.get("reason", ""), **common)

    def _record_probe_answer(self, block_id: Any, probe_id: str, **values: Any) -> None:
        schedule = self.probe_schedules.setdefault(block_id, {"block_id": block_id, "probe_times_sec": []})
        responses = schedule.setdefault("responses", {})
        responses.setdefault(probe_id, {}).update(values)
        self._write_metadata("recording")

    def end_block(self) -> None:
        with self._lock:
            if not self.current_block:
                raise RuntimeError("No block is active")
            self.log_event("block_end", event_value=self.current_video_id)
            finished_block = self.current_block
            self.completed_blocks.add(finished_block)
            if finished_block in REST_DURATIONS_AFTER_BLOCK:
                self.phase = "rest"
            elif finished_block == 6:
                self.phase = "complete"
            else:
                self.phase = "inter_block"
            self.current_block = 0
            self.current_condition = ""
            self.current_condition_type = ""
            self.current_video_id = ""
            self.current_session_half = 0

    def stop(self, export_mat: bool = True) -> Optional[Path]:
        with self._lock:
            if self._active and self.browser_clients and not self.shutdown_ready:
                raise RuntimeError("Browser event queues have not finished saving")
            if self._active:
                if not self._session_end_logged:
                    self.log_event("session_end")
                    self._session_end_logged = True
                self.ended_timestamp = self.clock_time()
                # Only close after all four files have passed flush + fsync.
                self._flush_files(sync=True)
                for handle in (self._eeg_handle, self._event_handle, self._raw_handle, self._qc_handle):
                    handle.close()
                self._active = False
            final_status = ("saved_with_unresolved_events" if self.unresolved_browser_events else
                            "complete" if self.phase == "complete" else "stopped")
            self.export_status = "exporting" if export_mat else "skipped"
            self._write_metadata(status="exporting" if export_mat else final_status)

        try:
            from .session_reports import build_normalized_tables, build_session_qc_report
        except ImportError:
            from session_reports import build_normalized_tables, build_session_qc_report
        try:
            build_normalized_tables(self.session_dir)
        except Exception as error:
            self.integrity_status = f"normalized tables failed: {error}"

        try:
            from .epoch_builder import build_session_epochs
        except ImportError:
            from epoch_builder import build_session_epochs
        # Derived indices must never block the raw session from being saved.
        try:
            build_session_epochs(self.session_dir)
            self.epoch_status = "complete"
        except Exception as error:
            self.epoch_status = f"failed: {error}"

        try:
            report_path = build_session_qc_report(self.session_dir)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.integrity_status = str(report.get("status", "unknown"))
        except Exception as error:
            self.integrity_status = f"failed: {error}"

        if export_mat:
            try:
                from .mat_exporter import export_session_to_mat
            except ImportError:
                from mat_exporter import export_session_to_mat

            final_metadata = self._metadata(final_status)
            final_metadata["export_status"] = "complete"
            try:
                path = export_session_to_mat(self.session_dir, metadata_override=final_metadata)
            except Exception:
                self.export_status = "failed"
                self._write_metadata(status="export_failed")
                raise
            self.export_status = "complete"
            self._write_metadata(status=final_status)
            return path
        return None
