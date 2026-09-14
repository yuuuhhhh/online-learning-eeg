"""Load and validate the single versioned protocol configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONFIG_RELATIVE_PATH = Path("config") / "protocol_v2.1.json"


def load_protocol_config(project_dir: Path | str | None = None) -> tuple[dict[str, Any], str, Path]:
    root = Path(project_dir).resolve() if project_dir is not None else Path(__file__).resolve().parent
    path = root / CONFIG_RELATIVE_PATH
    raw = path.read_bytes()
    config = json.loads(raw.decode("utf-8"))
    validate_protocol_config(config)
    return config, hashlib.sha256(raw).hexdigest(), path


def validate_protocol_config(config: dict[str, Any]) -> None:
    for field in ("schema_version", "config_version", "protocol_version", "software_version"):
        if not str(config.get(field, "")).strip():
            raise ValueError(f"{field} is required")
    if config.get("study_phases") != ["smoke", "pilot", "formal"]:
        raise ValueError("study_phases must be smoke, pilot, formal")
    conditions = config.get("conditions", {})
    if conditions.get("A", {}).get("condition_type") != "focused":
        raise ValueError("condition A must use condition_type=focused")
    if conditions.get("B", {}).get("condition_type") != "bbbd_subtraction":
        raise ValueError("condition B must use condition_type=bbbd_subtraction")

    groups = config.get("counterbalance_groups", {})
    expected_groups = [f"G{index:02d}" for index in range(1, 13)]
    if list(groups) != expected_groups:
        raise ValueError("counterbalance_groups must contain G01 through G12 in order")
    for group, entries in groups.items():
        if len(entries) != 6:
            raise ValueError(f"{group} must contain six blocks")
        videos = []
        conditions = []
        for entry in entries:
            video_id, condition = str(entry).split("-", 1)
            videos.append(video_id)
            conditions.append(condition)
        if sorted(videos) != [f"V{index}" for index in range(1, 7)]:
            raise ValueError(f"{group} must use V1 through V6 exactly once")
        if conditions.count("A") != 3 or conditions.count("B") != 3:
            raise ValueError(f"{group} must contain 3 A and 3 B blocks")
        if any(left == right for left, right in zip(conditions, conditions[1:])):
            raise ValueError(f"{group} conditions must alternate")

    flow = config.get("flow", {})
    if flow.get("block_count") != 6 or flow.get("baseline_duration_sec") != 30:
        raise ValueError("flow must contain six blocks and a fixed 30-second baseline")
    if flow.get("rest_durations_after_block_sec") != {
        "1": 30, "2": 30, "3": 30, "4": 30, "5": 30
    }:
        raise ValueError("rest schedule must be 30/30/30/30/30 seconds")

    probe = config.get("thought_probe", {})
    if probe.get("probes_per_block") != 4:
        raise ValueError("pilot protocol must schedule four probes per block")
    if [item.get("attention") for item in probe.get("options", [])] != ["ON", "OFF", "OFF", "AMBIGUOUS"]:
        raise ValueError("thought-probe option mapping is invalid")
    if probe.get("confidence_choices") != [1, 2, 3, 4]:
        raise ValueError("probe confidence choices must be 1 through 4")

    materials = config.get("materials", {})
    videos = materials.get("videos", [])
    if [video.get("video_id") for video in videos] != [f"V{index}" for index in range(1, 7)]:
        raise ValueError("materials must contain V1 through V6 in order")
    for video in videos:
        questions = video.get("questions", [])
        if len(questions) != 4:
            raise ValueError(f"{video.get('video_id')} must have exactly four questions")
        for question in questions:
            if len(question.get("options", [])) != 4:
                raise ValueError(f"{question.get('question_id')} must have exactly four options")
            if question.get("answer") not in range(4):
                raise ValueError(f"{question.get('question_id')} has an invalid answer index")
        duration = float(video.get("duration_sec", 0))
        required = (
            float(probe["min_first_onset_sec"])
            + (int(probe["probes_per_block"]) - 1) * float(probe["min_gap_sec"])
            + float(probe["min_end_margin_sec"])
        )
        if duration < required:
            raise ValueError(f"{video.get('video_id')} is too short for all four probes")

    qc = config.get("qc", {})
    if qc.get("affects_recording_or_labels") is not False:
        raise ValueError("acquisition QC must not alter recording or labels")
    raw_data = config.get("raw_data", {})
    for field in ("packet_protocol_version", "parser_version", "eeg_schema_version", "event_schema_version"):
        if not str(raw_data.get(field, "")).strip():
            raise ValueError(f"raw_data.{field} is required")
    if raw_data.get("missing_sample_policy") != "never_fill_or_interpolate":
        raise ValueError("missing EEG samples must never be filled or interpolated")


def planned_blocks(config: dict[str, Any], counterbalance_group: str) -> list[dict[str, Any]]:
    entries = config["counterbalance_groups"][counterbalance_group]
    return [
        {
            "block_id": index,
            "block_order": index,
            "session_half": 1 if index <= 3 else 2,
            "video_id": entry.split("-", 1)[0],
            "condition_label": entry.split("-", 1)[1],
        }
        for index, entry in enumerate(entries, start=1)
    ]
