from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import build_track_docs as track_doc_builder


def _write_test_video(video_path: str, *, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (100, 100))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV video writer is unavailable in this environment.")
    for frame_num in range(frame_count):
        frame = np.full((100, 100, 3), 30 * frame_num, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class BuildTrackDocsScriptTests(unittest.TestCase):
    def test_main_builds_docs_for_requested_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            linking_root = os.path.join(tmpdir, "linking")
            enrichment_root = os.path.join(tmpdir, "activation")
            scenario = "demo_scenario"
            linking_dir = os.path.join(linking_root, scenario)
            enrichment_dir = os.path.join(enrichment_root, scenario)
            os.makedirs(linking_dir, exist_ok=True)
            os.makedirs(enrichment_dir, exist_ok=True)

            video_path = os.path.join(tmpdir, "demo.avi")
            _write_test_video(video_path)

            with open(os.path.join(enrichment_dir, "enriched_detections.json"), "w", encoding="utf-8") as f:
                json.dump([], f)
            with open(os.path.join(enrichment_dir, "projection_manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"input_video_path": video_path}, f)
            with open(os.path.join(linking_dir, "tracks.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "tracks": [
                            {
                                "track_id": 1,
                                "class_id": 0,
                                "class_name": "person",
                                "start_frame": 0,
                                "end_frame": 1,
                                "hits": 2,
                                "valid_track": True,
                                "events": [],
                                "observations": [
                                    {
                                        "frame_num": 0,
                                        "det_index": 0,
                                        "bbox": [5.0, 5.0, 35.0, 35.0],
                                        "confidence": 0.9,
                                        "fragment_track_id": 1,
                                        "class_id": 0,
                                        "class_name": "person",
                                        "visual_similarity": None,
                                    }
                                ],
                            }
                        ]
                    },
                    f,
                )
            with open(os.path.join(linking_dir, "trace_summary.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "items": [
                            {
                                "track_id": 1,
                                "class_id": 0,
                                "class_name": "person",
                                "start_frame": 0,
                                "end_frame": 1,
                                "hits": 2,
                                "valid_track": True,
                                "relinked_from": [],
                                "had_recovery": False,
                                "frames": [
                                    {
                                        "frame_num": 0,
                                        "det_index": 0,
                                        "bbox": [5.0, 5.0, 35.0, 35.0],
                                        "fragment_track_id": 1,
                                        "canonical_track_id": 1,
                                        "role": "start",
                                    }
                                ],
                            }
                        ]
                    },
                    f,
                )

            rc = track_doc_builder.main(
                [
                    "--linking-root",
                    linking_root,
                    "--enrichment-root",
                    enrichment_root,
                    "--scenario",
                    scenario,
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(linking_dir, "track_docs", "README.md")))
            self.assertTrue(os.path.exists(os.path.join(linking_root, "track_docs_overview.md")))


if __name__ == "__main__":
    unittest.main()
