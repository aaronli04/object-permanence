from __future__ import annotations

from collections import deque
import os
import sys
import unittest

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.assignment import assign_frame
from temporal_linking.config import TemporalLinkingConfig
from temporal_linking.types import Detection, Track, TrackStatus


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _make_track(
    track_id: int,
    vec: np.ndarray,
    class_id: int = 32,
    *,
    bbox: np.ndarray | None = None,
    frame_width: float | None = 100.0,
    frame_height: float | None = 100.0,
) -> Track:
    vec_n = _normalize(vec)
    bbox_xyxy = bbox if bbox is not None else np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32)
    track = Track(
        track_id=track_id,
        class_id=class_id,
        class_name="sports ball",
        status=TrackStatus.ACTIVE,
        start_frame=0,
        last_seen_frame=0,
        last_bbox_xyxy=bbox_xyxy,
        last_vec=vec_n.copy(),
        ema_vec=vec_n.copy(),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    track.vec_history = deque([vec_n.copy()], maxlen=5)
    return track


def _make_det(
    det_index: int,
    vec: np.ndarray,
    class_id: int = 32,
    *,
    bbox: np.ndarray | None = None,
    frame_width: float | None = 100.0,
    frame_height: float | None = 100.0,
) -> Detection:
    vec_n = _normalize(vec)
    bbox_xyxy = bbox if bbox is not None else np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32)
    return Detection(
        frame_num=1,
        det_index=det_index,
        class_id=class_id,
        class_name="sports ball",
        bbox_xyxy=bbox_xyxy,
        confidence=0.9,
        activation_vec=vec_n,
        small_crop_flag=False,
        raw_payload={
            "class_id": class_id,
            "class_name": "sports ball",
            "bbox": [float(v) for v in bbox_xyxy.tolist()],
            "confidence": 0.9,
            "activation": {"vector": vec_n.tolist(), "dim": int(vec_n.shape[0]), "small_crop_flag": False},
        },
        frame_width=frame_width,
        frame_height=frame_height,
    )


class AssignmentTests(unittest.TestCase):
    def test_hungarian_picks_global_best_pairing(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.6, assignment_method="hungarian")

        tracks = [
            _make_track(track_id=1, vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
            _make_track(track_id=2, vec=np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
        ]
        detections = [
            _make_det(det_index=0, vec=np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
            _make_det(det_index=1, vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32)),
        ]

        assignments = assign_frame(tracks, detections, cfg)
        got = {(a.track_id, a.det_index) for a in assignments}

        self.assertEqual(got, {(1, 1), (2, 0)})

    def test_visual_threshold_pre_gates_assignment(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.95)

        track = _make_track(track_id=1, vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        below_threshold = _make_det(det_index=0, vec=np.asarray([0.8, 0.6, 0.0], dtype=np.float32))
        above_threshold = _make_det(det_index=1, vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))

        assignments = assign_frame([track], [below_threshold, above_threshold], cfg)

        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].det_index, 1)
        self.assertGreaterEqual(assignments[0].visual_similarity, cfg.similarity_threshold)

    def test_class_policy_blocks_mismatch(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.0, match_within_class=True)
        track = _make_track(track_id=1, vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32), class_id=32)
        det = _make_det(det_index=0, vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32), class_id=1)

        assignments = assign_frame([track], [det], cfg)
        self.assertEqual(assignments, [])

    def test_spatial_gate_blocks_implausible_high_cosine_match(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.0, max_centroid_distance=0.10)
        vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        track = _make_track(
            track_id=1,
            vec=vec,
            bbox=np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
            frame_width=100.0,
            frame_height=100.0,
        )
        far_det = _make_det(
            det_index=0,
            vec=vec,
            bbox=np.asarray([90.0, 90.0, 100.0, 100.0], dtype=np.float32),
            frame_width=100.0,
            frame_height=100.0,
        )
        near_det = _make_det(
            det_index=1,
            vec=vec,
            bbox=np.asarray([2.0, 2.0, 12.0, 12.0], dtype=np.float32),
            frame_width=100.0,
            frame_height=100.0,
        )

        assignments = assign_frame([track], [far_det, near_det], cfg)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].det_index, 1)


if __name__ == "__main__":
    unittest.main()
