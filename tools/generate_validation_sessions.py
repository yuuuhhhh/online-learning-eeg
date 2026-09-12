"""Generate two tiny synthetic runs for integrity-report acceptance testing.

These files exercise table counts and report logic only.  They are not EEG
recordings and must never be used for scientific analysis.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protocol_config import load_protocol_config, planned_blocks
from epoch_builder import PROBE_EPOCH_COLUMNS
from mat_exporter import export_session_to_mat
from session_reports import build_session_qc_report
from session_recorder import EEG_CSV_COLUMNS


OUT = ROOT / "validation_samples"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_header(path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def build_rows() -> list[dict]:
    config, _, _ = load_protocol_config(ROOT)
    blocks = planned_blocks(config, "G01")
    rows: list[dict] = []
    timestamp = 1_800_000_000.0

    def add(event_name: str, block: dict, **extra) -> None:
        nonlocal timestamp
        timestamp += 0.01
        rows.append({
            "event_name": event_name,
            "event_type": event_name,
            "recorder_timestamp": timestamp,
            "device_sample_number": int((timestamp - 1_800_000_000) * 250),
            "subject_id": "synthetic-complete",
            "session_id": "ses-validation",
            "run_id": "synthetic-run",
            "counterbalance_group": "G01",
            "block_id": block["block_order"],
            "block_order": block["block_order"],
            "video_id": block["video_id"],
            "condition_label": block["condition_label"],
            "condition_type": config["conditions"][block["condition_label"]]["condition_type"],
            **extra,
        })

    for block in blocks:
        add("block_start", block)
        for probe_index in range(1, 5):
            probe_id = f"B{block['block_order']}P{probe_index}"
            common = {"probe_id": probe_id, "probe_index": probe_index,
                      "probe_planned_time_sec": 45 + probe_index * 55}
            add("probe_onset", block, video_time_sec=common["probe_planned_time_sec"], **common)
            add("attention_response", block, response=1, response_label="课程内容",
                probe_attention="ON", response_time_ms=800, **common)
            add("confidence_response", block, confidence=3, response_time_ms=400, **common)
            add("video_resume", block, pause_reason="thought_probe", **common)
        rating = {
            "course_attention_rating": 4,
            "mental_effort": 5,
            "video_interest": 4,
            "video_difficulty": 3,
        }
        if block["condition_label"] == "B":
            rating.update({
                "subtraction_compliance": "70-80%",
                "subtraction_compliance_low_pct": 70,
                "subtraction_compliance_high_pct": 80,
                "subtraction_difficulty": 3,
                "reported_final_number": 137,
            })
        add("rating_end", block, **rating)
        for question_index in range(1, 5):
            add("quiz_item_response", block,
                question_id=f"{block['video_id']}Q{question_index}",
                question_text="synthetic validation question",
                response_index=1, participant_answer=1, response_text="synthetic",
                correct_answer=1, is_correct=1, reaction_time_ms=900)
        add("block_end", block)
    return rows


def write_run(path: Path, rows: list[dict], label: str) -> None:
    config, digest, _ = load_protocol_config(ROOT)
    path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "synthetic_validation_only": True,
        "run_id": f"synthetic-{label}",
        "software_version": config["software_version"],
        "protocol_version": config["protocol_version"],
        "protocol_config_version": config["config_version"],
        "protocol_config_sha256": digest,
        "session": {
            "subject_id": f"synthetic-{label}",
            "session_id": "ses-validation",
            "study_phase": "formal",
            "counterbalance_group": "G01",
            "sample_rate_hz": 250,
            "channel_names": ["synthetic_channel_0", "synthetic_channel_1"],
            "conditions": config["conditions"],
            "b_start_numbers": {
                "2": {"start_number": 809},
                "4": {"start_number": 811},
                "6": {"start_number": 821},
            },
        },
        "preprocessing": config["preprocessing"],
        "labels": {"-1": "no_instantaneous_attention_label"},
        "notice": "Synthetic integrity-test fixture; not scientific EEG data.",
    }
    (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(path / "events.csv", rows)
    eeg_row = {field: "" for field in EEG_CSV_COLUMNS}
    eeg_row.update({
        "device_sample_number": 1, "received_timestamp": 1_800_000_000.0,
        "received_order": 0, "sample_index": 1, "sample_time_sec": 0.004,
        "packet_gap_before": 0, "quality_flag": "", "block_id": 1,
        "block_order": 1, "session_half": 1, "counterbalance_group": "G01",
        "video_id": "V1", "condition": "A", "condition_label": "A",
        "condition_type": "focused", "condition_code": 0, "weak_label": -1,
        "phase": "video", "base_valid_for_training": 1,
        "channel_0_raw": 100, "channel_1_raw": -100,
        "channel_0_uv": 2.4, "channel_1_uv": -2.4,
        "is_formal_experiment": 1, "stream_segment": 0,
        "sample_time_status": "synthetic",
    })
    write_csv(path / "eeg.csv", [eeg_row])
    (path / "eeg_raw.bin").write_bytes(b"SYNTHETIC_VALIDATION_ONLY")
    write_csv(path / "qc.csv", [{"channel_0_rms_uv": 4.2, "channel_1_rms_uv": 4.5, "signal_alive": 1}])
    write_csv(path / "windows.csv", [
        {"block_id": block_id, "quality_label": "GOOD"} for block_id in range(1, 7)
    ])
    build_session_qc_report(path)
    write_header(path / "probe_epochs.csv", PROBE_EPOCH_COLUMNS)
    export_session_to_mat(path)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    full_rows = build_rows()
    write_run(OUT / "完整会话_PASS", full_rows, "complete")

    incomplete_rows = [
        row for row in full_rows
        if not (row["event_name"] == "confidence_response" and row.get("probe_id") == "B2P3")
        and not (row["event_name"] == "rating_end" and row.get("block_id") == 6)
        and not (row["event_name"] == "quiz_item_response" and row.get("question_id") == "V4Q4")
    ]
    write_run(OUT / "缺项会话_FAIL", incomplete_rows, "incomplete")
    (OUT / "README.txt").write_text(
        "仅用于验证完整性报告：完整会话应为 PASS，缺项会话应为 FAIL。所有 EEG/事件均为合成数据，禁止用于分析。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
