"""G01-G12 counterbalance allocation with phase-separated recruitment queues."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .protocol_config import load_protocol_config
except ImportError:
    from protocol_config import load_protocol_config


def recommend_counterbalance_group(project_dir: Path, subject_id: str, study_phase: str) -> dict:
    """Cycle G01..G12 within one study phase and preserve repeat assignments."""
    subject_id = subject_id.strip()
    # The data directory may be redirected to a temporary/external root.  The
    # protocol itself remains part of the application package in that case.
    config_root = Path(project_dir)
    if not (config_root / "config" / "protocol_v2.1.json").exists():
        config_root = Path(__file__).resolve().parent
    protocol, _, _ = load_protocol_config(config_root)
    phases = set(protocol["study_phases"])
    if study_phase not in phases:
        raise ValueError(f"study_phase must be one of {sorted(phases)}")
    group_ids = list(protocol["counterbalance_groups"])

    assignments: dict[str, tuple[float, str]] = {}
    data_dir = Path(project_dir) / "data"
    if data_dir.exists():
        for metadata_path in data_dir.rglob("metadata.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                session = metadata.get("session", {})
                if str(session.get("study_phase", "")).strip() != study_phase:
                    continue
                saved_subject = str(session.get("subject_id", "")).strip()
                saved_group = str(session.get("counterbalance_group", "")).strip().upper()
                started = float(session.get("started_timestamp") or 0.0)
                if not saved_subject or saved_group not in group_ids:
                    continue
                previous = assignments.get(saved_subject)
                if previous is None or started < previous[0]:
                    assignments[saved_subject] = (started, saved_group)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

    if subject_id and subject_id in assignments:
        group = assignments[subject_id][1]
        return {
            "counterbalance_group": group,
            "existing_subject": True,
            "assigned_subjects": len(assignments),
            "study_phase": study_phase,
        }

    ordered = sorted(assignments.items(), key=lambda item: (item[1][0], item[0]))
    group = group_ids[len(ordered) % len(group_ids)]
    return {
        "counterbalance_group": group,
        "existing_subject": False,
        "assigned_subjects": len(assignments),
        "study_phase": study_phase,
    }
