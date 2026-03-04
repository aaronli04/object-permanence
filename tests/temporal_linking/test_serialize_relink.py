from __future__ import annotations

import os
import sys
import unittest

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.config import TemporalLinkingConfig
from temporal_linking.serialize import apply_merges_to_tracks_payload, remap_linked_frames_track_ids


class SerializeRelinkTests(unittest.TestCase):
    def test_apply_merges_recomputes_track_summary_fields(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.65, min_track_length=2)
        tracks = [
            {
                "track_id": 1,
                "class_id": 32,
                "class_name": "sports ball",
                "status": "closed",
                "start_frame": 0,
                "end_frame": 5,
                "hits": 2,
                "total_misses": 3,
                "max_miss_streak": 3,
                "avg_visual_similarity": 0.8,
                "valid_track": True,
                "events": [{"frame_num": 0, "type": "created"}, {"frame_num": 5, "type": "closed"}],
                "observations": [
                    {"frame_num": 0, "det_index": 0, "bbox": [0, 0, 1, 1], "visual_similarity": None},
                    {"frame_num": 5, "det_index": 0, "bbox": [0, 0, 1, 1], "visual_similarity": 0.8},
                ],
            },
            {
                "track_id": 2,
                "class_id": 32,
                "class_name": "sports ball",
                "status": "closed",
                "start_frame": 10,
                "end_frame": 10,
                "hits": 1,
                "total_misses": 2,
                "max_miss_streak": 2,
                "avg_visual_similarity": None,
                "valid_track": False,
                "events": [{"frame_num": 10, "type": "created"}, {"frame_num": 10, "type": "closed"}],
                "observations": [
                    {"frame_num": 10, "det_index": 0, "bbox": [0, 0, 1, 1], "visual_similarity": None},
                ],
            },
        ]

        merged = apply_merges_to_tracks_payload(tracks, merge_map={2: 1}, cfg=cfg)

        self.assertEqual(len(merged), 1)
        track = merged[0]
        self.assertEqual(track["track_id"], 1)
        self.assertEqual(track["start_frame"], 0)
        self.assertEqual(track["end_frame"], 10)
        self.assertEqual(track["hits"], 3)
        self.assertEqual(track["total_misses"], 5)
        self.assertEqual(track["max_miss_streak"], 3)
        self.assertEqual(track["relinked_from"], [2])
        self.assertAlmostEqual(float(track["avg_visual_similarity"]), 0.8, places=5)
        self.assertTrue(bool(track["valid_track"]))

    def test_remap_linked_frames_track_ids(self) -> None:
        linked_frames = [
            {
                "frame_num": 0,
                "detections": [
                    {"det_index": 0, "temporal_link": {"track_id": 1}},
                    {"det_index": 1, "temporal_link": {"track_id": 2}},
                ],
            }
        ]

        remapped = remap_linked_frames_track_ids(linked_frames, merge_map={2: 1})
        det_ids = [det["temporal_link"]["track_id"] for det in remapped[0]["detections"]]
        self.assertEqual(det_ids, [1, 1])


if __name__ == "__main__":
    unittest.main()
