from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.config import TemporalLinkingConfig
from temporal_linking.tracker import TrackManager
from temporal_linking.types import Assignment, Detection, TrackStatus


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _make_det(frame_num: int, det_index: int, vec: np.ndarray) -> Detection:
    vec_n = _normalize(vec)
    return Detection(
        frame_num=frame_num,
        det_index=det_index,
        class_id=32,
        class_name="sports ball",
        bbox_xyxy=np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32),
        confidence=0.9,
        activation_vec=vec_n,
        small_crop_flag=False,
        raw_payload={
            "class_id": 32,
            "class_name": "sports ball",
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "confidence": 0.9,
            "activation": {"vector": vec_n.tolist(), "dim": 256, "small_crop_flag": False},
        },
    )


class TrackerLifecycleTests(unittest.TestCase):
    def test_track_transitions_tentative_to_lost_to_active_to_closed(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.8, min_hits_to_activate=2, max_lost_frames=1)
        manager = TrackManager(cfg)

        first = _make_det(frame_num=0, det_index=0, vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        track = manager.spawn(first, frame_num=0)
        self.assertEqual(track.status, TrackStatus.TENTATIVE)

        manager.mark_unmatched([track], matched_track_ids=set(), frame_num=1)
        self.assertEqual(track.status, TrackStatus.LOST)

        recovered_det = _make_det(frame_num=2, det_index=0, vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        assignment = Assignment(
            track_id=track.track_id,
            det_index=0,
            visual_similarity=0.95,
            spatial_score=0.8,
            total_score=1.05,
            source_track_status=TrackStatus.LOST,
        )
        manager.apply_match(track.track_id, recovered_det, assignment, frame_num=2)
        self.assertEqual(track.status, TrackStatus.ACTIVE)

        manager.close(track, end_frame=2)
        self.assertEqual(track.status, TrackStatus.CLOSED)
        self.assertIn(track.track_id, manager.state.closed)


if __name__ == "__main__":
    unittest.main()
