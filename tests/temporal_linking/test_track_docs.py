from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.track_docs import build_track_docs


def _write_test_video(video_path: str, *, frame_count: int = 4) -> None:
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (100, 100))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV video writer is unavailable in this environment.")
    for frame_num in range(frame_count):
        frame = np.full((100, 100, 3), 40 * frame_num, dtype=np.uint8)
        cv2.putText(frame, str(frame_num), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


class TrackDocsTests(unittest.TestCase):
    def test_build_track_docs_writes_summary_frames_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.avi")
            _write_test_video(video_path)

            tracks_payload = {
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 32,
                        "class_name": "sports ball",
                        "start_frame": 0,
                        "end_frame": 3,
                        "hits": 3,
                        "valid_track": True,
                        "avg_visual_similarity": 0.975,
                        "total_misses": 1,
                        "max_miss_streak": 1,
                        "relinked_from": [7],
                        "events": [
                            {"frame_num": 0, "type": "created"},
                            {"frame_num": 3, "type": "recovered"},
                        ],
                        "observations": [
                            {
                                "frame_num": 0,
                                "det_index": 0,
                                "bbox": [10.0, 10.0, 35.0, 35.0],
                                "confidence": 0.9,
                                "visual_similarity": None,
                                "fragment_track_id": 1,
                                "class_id": 32,
                                "class_name": "sports ball",
                            },
                            {
                                "frame_num": 1,
                                "det_index": 0,
                                "bbox": [20.0, 10.0, 45.0, 35.0],
                                "confidence": 0.8,
                                "visual_similarity": 0.98,
                                "fragment_track_id": 1,
                                "class_id": 32,
                                "class_name": "sports ball",
                            },
                            {
                                "frame_num": 3,
                                "det_index": 0,
                                "bbox": [55.0, 10.0, 80.0, 35.0],
                                "confidence": 0.3,
                                "visual_similarity": 0.95,
                                "fragment_track_id": 7,
                                "class_id": 75,
                                "class_name": "vase",
                            },
                        ],
                    }
                ]
            }
            trace_summary_payload = {
                "items": [
                    {
                        "track_id": 1,
                        "class_id": 32,
                        "class_name": "sports ball",
                        "start_frame": 0,
                        "end_frame": 3,
                        "hits": 3,
                        "valid_track": True,
                        "relinked_from": [7],
                        "had_recovery": True,
                        "frames": [
                            {
                                "frame_num": 0,
                                "det_index": 0,
                                "bbox": [10.0, 10.0, 35.0, 35.0],
                                "fragment_track_id": 1,
                                "canonical_track_id": 1,
                                "role": "start",
                            },
                            {
                                "frame_num": 1,
                                "det_index": 0,
                                "bbox": [20.0, 10.0, 45.0, 35.0],
                                "fragment_track_id": 1,
                                "canonical_track_id": 1,
                                "role": "middle",
                            },
                            {
                                "frame_num": 3,
                                "det_index": 0,
                                "bbox": [55.0, 10.0, 80.0, 35.0],
                                "fragment_track_id": 7,
                                "canonical_track_id": 1,
                                "role": "end",
                            },
                        ],
                    }
                ]
            }
            relink_manifest_payload = {
                "accepted_edges": [
                    {"predecessor_id": 1, "successor_id": 7, "score": 0.81, "method": "dino"}
                ]
            }

            output = build_track_docs(
                scenario="demo_track",
                tracks_payload=tracks_payload,
                trace_summary_payload=trace_summary_payload,
                output_dir=os.path.join(tmpdir, "track_docs"),
                trace_summary_dir=os.path.join(tmpdir, "trace_summary"),
                relink_manifest_payload=relink_manifest_payload,
                video_path=video_path,
            )

            self.assertEqual(output.num_tracks, 1)
            self.assertEqual(output.num_summary_images, 1)
            self.assertEqual(output.num_frame_images, 3)

            track_dir = os.path.join(output.output_dir, "track_1")
            self.assertTrue(os.path.exists(os.path.join(track_dir, "summary.jpg")))
            self.assertTrue(os.path.exists(os.path.join(track_dir, "frames", "start_f0000.jpg")))
            self.assertTrue(os.path.exists(os.path.join(track_dir, "frames", "middle_f0001.jpg")))
            self.assertTrue(os.path.exists(os.path.join(track_dir, "frames", "end_f0003.jpg")))

            with open(os.path.join(track_dir, "track.json"), "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["derived"]["fragment_track_ids"], [1, 7])
            self.assertEqual(len(payload["derived"]["accepted_relink_edges"]), 1)
            self.assertEqual(payload["artifacts"]["summary_image"], "summary.jpg")
            self.assertEqual(len(payload["artifacts"]["frame_images"]), 3)
            self.assertEqual(payload["derived"]["observed_classes"][0]["class_name"], "sports ball")

            with open(os.path.join(track_dir, "README.md"), "r", encoding="utf-8") as f:
                readme = f.read()
            self.assertIn("sports ball", readme)
            self.assertIn("1 -> 7 via dino", readme)
            self.assertIn("[track metadata](track.json)", readme)

            with open(output.index_path, "r", encoding="utf-8") as f:
                index_readme = f.read()
            self.assertIn("[track_1](track_1/README.md)", index_readme)


if __name__ == "__main__":
    unittest.main()
