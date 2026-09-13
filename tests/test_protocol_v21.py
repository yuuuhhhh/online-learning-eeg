from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from order_balance import recommend_counterbalance_group
from protocol_config import load_protocol_config, planned_blocks
from session_recorder import PROTOCOL_CONFIG, SessionConfig, validate_probe_schedule


ROOT = Path(__file__).resolve().parents[1]


class ProtocolConfigTests(unittest.TestCase):
    def test_config_is_valid_and_traceable(self):
        config, digest, path = load_protocol_config(ROOT)
        self.assertEqual(config["protocol_version"], "2.1")
        self.assertEqual(config["software_version"], "2.1.1")
        self.assertEqual(config["flow"]["baseline_duration_sec"], 30)
        self.assertEqual(
            config["flow"]["rest_durations_after_block_sec"],
            {str(block_id): 30 for block_id in range(1, 6)},
        )
        self.assertEqual(len(digest), 64)
        self.assertEqual(path.name, "protocol_v2.1.json")

    def test_all_twelve_counterbalance_templates_are_complete(self):
        groups = PROTOCOL_CONFIG["counterbalance_groups"]
        self.assertEqual(list(groups), [f"G{i:02d}" for i in range(1, 13)])
        for group in groups:
            blocks = planned_blocks(PROTOCOL_CONFIG, group)
            self.assertEqual([row["block_order"] for row in blocks], list(range(1, 7)))
            self.assertEqual({row["video_id"] for row in blocks}, {f"V{i}" for i in range(1, 7)})
            conditions = [row["condition_label"] for row in blocks]
            self.assertEqual(conditions.count("A"), 3)
            self.assertEqual(conditions.count("B"), 3)
            self.assertTrue(all(a != b for a, b in zip(conditions, conditions[1:])))

    def test_each_video_has_balanced_condition_and_position(self):
        all_blocks = [planned_blocks(PROTOCOL_CONFIG, group) for group in PROTOCOL_CONFIG["counterbalance_groups"]]
        for video_id in [f"V{i}" for i in range(1, 7)]:
            rows = [row for blocks in all_blocks for row in blocks if row["video_id"] == video_id]
            self.assertEqual(sum(row["condition_label"] == "A" for row in rows), 6)
            self.assertEqual(sum(row["condition_label"] == "B" for row in rows), 6)
            self.assertEqual(sorted(row["block_order"] for row in rows), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

    def test_assignments_cycle_and_are_separate_by_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(12):
                session = root / "data" / f"sub-{index:03d}" / "ses-001" / f"run-{index:03d}"
                session.mkdir(parents=True)
                (session / "metadata.json").write_text(json.dumps({
                    "session": {"subject_id": f"sub-{index:03d}", "study_phase": "formal",
                                "counterbalance_group": f"G{index + 1:02d}", "started_timestamp": index + 1}
                }), encoding="utf-8")
            self.assertEqual(recommend_counterbalance_group(root, "sub-new", "formal")["counterbalance_group"], "G01")
            self.assertEqual(recommend_counterbalance_group(root, "sub-003", "formal")["counterbalance_group"], "G04")
            self.assertEqual(recommend_counterbalance_group(root, "sub-new", "pilot")["counterbalance_group"], "G01")

    def test_formal_mode_is_blocked_until_config_is_frozen(self):
        with self.assertRaisesRegex(ValueError, "尚未.*冻结"):
            SessionConfig(subject_id="sub-001", counterbalance_group="G01", study_phase="formal").validate()

    def test_materials_are_six_real_videos_with_four_questions(self):
        for video in PROTOCOL_CONFIG["materials"]["videos"]:
            path = ROOT / "materials" / "videos" / video["filename"]
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 1_000_000)
            self.assertGreaterEqual(video["duration_sec"], 300)
            self.assertEqual(len(video["questions"]), 4)
            self.assertTrue(all(len(question["options"]) == 4 for question in video["questions"]))

    def test_short_video_is_rejected_not_silently_reduced(self):
        with self.assertRaisesRegex(ValueError, "cannot fit all 4 probes"):
            validate_probe_schedule([45, 90], 120)

    def test_acquisition_does_not_measure_50hz(self):
        self.assertFalse(PROTOCOL_CONFIG["qc"]["acquisition_line_noise_measurement"])
        self.assertEqual(PROTOCOL_CONFIG["preprocessing"]["line_noise_filter"]["type"], "notch_bandstop")


if __name__ == "__main__":
    unittest.main()
