from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import davis_curated_common as common
import build_davis_curated_dataset as builder
import summarize_davis_curated_results as summary


class DavisCuratedCommonTests(unittest.TestCase):
    def test_stress_preset_excludes_baseline_sequences(self) -> None:
        scenarios = {item.scenario for item in common.iter_sequences_for_preset("stress")}
        self.assertIn("davis__person_occlusion__parkour", scenarios)
        self.assertIn("davis__bear_occlusion__bear", scenarios)
        self.assertNotIn("davis__dog_baseline__dog", scenarios)
        self.assertNotIn("davis__elephant_baseline__elephant", scenarios)

    def test_unknown_preset_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            common.iter_sequences_for_preset("unknown")


class DavisCuratedBuilderTests(unittest.TestCase):
    def test_build_manifest_row_marks_missing_sequence_without_shelling_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            davis_root = Path(tmpdir) / "DAVIS" / "JPEGImages" / "480p"
            output_dir = Path(tmpdir) / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)

            row = builder.build_manifest_row(
                item=common.CuratedSequence(
                    sequence="missing-sequence",
                    scenario="missing__scenario",
                    group="stress",
                    expected_classes=("person",),
                    notes="missing input",
                ),
                davis_root=davis_root,
                output_dir=output_dir,
                fps=24,
                force=False,
            )

            self.assertEqual(row.status, "missing_sequence")
            self.assertEqual(row.frames, 0)
            self.assertEqual(row.video_path, str(output_dir / "missing__scenario.mp4"))


class DavisCuratedSummaryTests(unittest.TestCase):
    def test_summarize_item_computes_expected_shares_and_eval_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            enrichment_root = root / "activation"
            linking_root = root / "linking"
            scenario = "davis__person_occlusion__parkour"

            scenario_enrichment_dir = enrichment_root / scenario
            scenario_linking_dir = linking_root / scenario
            trace_summary_dir = scenario_linking_dir / "trace_summary"
            trace_summary_dir.mkdir(parents=True, exist_ok=True)
            (trace_summary_dir / "track_1.jpg").write_bytes(b"jpg")
            (trace_summary_dir / "track_2.jpg").write_bytes(b"jpg")

            frames = [
                {
                    "frame_num": 0,
                    "detections": [
                        {"class_name": "person"},
                        {"class_name": "skateboard"},
                    ],
                },
                {
                    "frame_num": 1,
                    "detections": [
                        {"class_name": "person"},
                    ],
                },
            ]
            tracks_payload = {
                "tracks": [
                    {
                        "track_id": 1,
                        "class_name": "person",
                        "hits": 2,
                        "start_frame": 0,
                        "end_frame": 1,
                        "valid_track": True,
                    },
                    {
                        "track_id": 2,
                        "class_name": "skateboard",
                        "hits": 1,
                        "start_frame": 0,
                        "end_frame": 0,
                        "valid_track": False,
                    },
                ]
            }
            relink_payload = {"stats": {"num_accepted_edges": 1, "relink_dino_coverage": 0.5}}

            (scenario_enrichment_dir).mkdir(parents=True, exist_ok=True)
            (scenario_linking_dir).mkdir(parents=True, exist_ok=True)
            (scenario_enrichment_dir / "enriched_detections.json").write_text(json.dumps(frames), encoding="utf-8")
            (scenario_linking_dir / "tracks.json").write_text(json.dumps(tracks_payload), encoding="utf-8")
            (scenario_linking_dir / "relink_manifest.json").write_text(json.dumps(relink_payload), encoding="utf-8")

            row = summary.summarize_item(
                {
                    "scenario": scenario,
                    "sequence": "parkour",
                    "group": "occlusion",
                    "expected_classes": "person",
                    "status": "existing_video",
                },
                enrichment_root,
                linking_root,
            )

            self.assertEqual(row.sampled_frames, 2)
            self.assertEqual(row.frames_with_detections, 2)
            self.assertEqual(row.detections_total, 3)
            self.assertEqual(row.trace_summary_images, 2)
            self.assertAlmostEqual(row.expected_detection_share, 2.0 / 3.0, places=6)
            self.assertAlmostEqual(row.expected_track_share, 0.5, places=6)
            self.assertAlmostEqual(row.best_expected_track_coverage, 1.0, places=6)
            self.assertEqual(row.class_eval, "FAIL")
            self.assertEqual(row.track_eval, "PASS")
            self.assertEqual(row.overall_eval, "FAIL")


if __name__ == "__main__":
    unittest.main()
