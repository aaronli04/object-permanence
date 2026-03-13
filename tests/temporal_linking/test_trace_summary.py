from __future__ import annotations

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.config import TemporalLinkingConfig
from temporal_linking.trace_summary import build_trace_summary, render_trace_summary, summary_color, summary_color_index


def _write_video(path: str, *, frame_count: int = 12) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 72))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV video writer is unavailable in this environment.")
    for frame_num in range(frame_count):
        frame = np.full((72, 96, 3), 30 + (frame_num * 10), dtype=np.uint8)
        cv2.putText(frame, str(frame_num), (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


class TraceSummaryTests(unittest.TestCase):
    def test_build_trace_summary_creates_one_item_per_final_track(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.7, min_hits_to_activate=1)
        payload = build_trace_summary(
            tracks_payload={
                "tracks": [
                    {
                        "track_id": 2,
                        "class_id": 32,
                        "class_name": "sports ball",
                        "start_frame": 1,
                        "end_frame": 10,
                        "hits": 5,
                        "valid_track": True,
                        "relinked_from": [7],
                        "events": [
                            {"frame_num": 2, "type": "created"},
                            {"frame_num": 8, "type": "recovered"},
                        ],
                        "observations": [
                            {"frame_num": 1, "det_index": 0, "bbox": [10, 10, 30, 30], "visual_similarity": None, "fragment_track_id": 2},
                            {"frame_num": 3, "det_index": 0, "bbox": [10, 10, 30, 30], "visual_similarity": 0.9, "fragment_track_id": 2},
                            {"frame_num": 5, "det_index": 0, "bbox": [10, 10, 30, 30], "visual_similarity": 0.9, "fragment_track_id": 2},
                            {"frame_num": 8, "det_index": 0, "bbox": [10, 10, 30, 30], "visual_similarity": 0.9, "fragment_track_id": 7},
                            {"frame_num": 10, "det_index": 0, "bbox": [10, 10, 30, 30], "visual_similarity": 0.9, "fragment_track_id": 7},
                        ],
                    }
                ]
            },
            cfg=cfg,
            enriched_json_path="/tmp/enriched.json",
        )

        self.assertEqual(payload["schema_version"], "temporal_linking_trace_summary_v1")
        self.assertEqual(len(payload["items"]), 1)
        item = payload["items"][0]
        self.assertEqual(item["track_id"], 2)
        self.assertEqual(item["relinked_from"], [7])
        self.assertTrue(bool(item["had_recovery"]))
        self.assertEqual([frame["role"] for frame in item["frames"]], ["start", "middle", "end"])
        self.assertEqual([frame["frame_num"] for frame in item["frames"]], [1, 5, 10])

    def test_build_trace_summary_uses_available_frames_without_padding(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.7, min_hits_to_activate=1)
        payload = build_trace_summary(
            tracks_payload={
                "tracks": [
                    {
                        "track_id": 3,
                        "class_id": 32,
                        "class_name": "sports ball",
                        "start_frame": 2,
                        "end_frame": 7,
                        "hits": 2,
                        "valid_track": True,
                        "relinked_from": [],
                        "events": [{"frame_num": 2, "type": "created"}],
                        "observations": [
                            {"frame_num": 2, "det_index": 0, "bbox": [10, 10, 30, 30], "visual_similarity": None, "fragment_track_id": 3},
                            {"frame_num": 7, "det_index": 0, "bbox": [10, 10, 30, 30], "visual_similarity": 0.9, "fragment_track_id": 3},
                        ],
                    }
                ]
            },
            cfg=cfg,
            enriched_json_path="/tmp/enriched.json",
        )

        item = payload["items"][0]
        self.assertEqual([frame["role"] for frame in item["frames"]], ["start", "end"])
        self.assertEqual([frame["frame_num"] for frame in item["frames"]], [2, 7])

    def test_summary_color_mapping_is_stable_and_not_direct_modulo(self) -> None:
        ids = [1, 2, 3, 4, 25]
        first_pass = [summary_color_index(track_id) for track_id in ids]
        second_pass = [summary_color_index(track_id) for track_id in ids]

        self.assertEqual(first_pass, second_pass)
        self.assertNotEqual(first_pass, [track_id % 24 for track_id in ids])
        self.assertEqual(summary_color(42), summary_color(42))

    def test_render_trace_summary_writes_one_jpeg_per_track(self) -> None:
        payload = {
            "schema_version": "temporal_linking_trace_summary_v1",
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "config_hash_sha256": "abc",
            "input_enriched_json": "enriched.json",
            "items": [
                {
                    "track_id": 2,
                    "class_id": 32,
                    "class_name": "sports ball",
                    "start_frame": 2,
                    "end_frame": 7,
                    "hits": 4,
                    "valid_track": True,
                    "relinked_from": [7],
                    "had_recovery": True,
                    "frames": [
                        {"frame_num": 2, "det_index": 0, "bbox": [10.0, 10.0, 30.0, 30.0], "fragment_track_id": 2, "canonical_track_id": 2, "role": "start"},
                        {"frame_num": 4, "det_index": 0, "bbox": [10.0, 10.0, 30.0, 30.0], "fragment_track_id": 2, "canonical_track_id": 2, "role": "middle"},
                        {"frame_num": 7, "det_index": 0, "bbox": [10.0, 10.0, 30.0, 30.0], "fragment_track_id": 7, "canonical_track_id": 2, "role": "end"},
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.avi")
            _write_video(video_path)
            rendered = render_trace_summary(
                trace_summary_payload=payload,
                video_path=video_path,
                output_dir=os.path.join(tmpdir, "trace_summary"),
            )

            self.assertEqual(rendered, 1)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "trace_summary", "track_2.jpg")))


if __name__ == "__main__":
    unittest.main()
