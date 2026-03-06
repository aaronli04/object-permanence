from __future__ import annotations

from collections import deque
import os
import sys
import unittest

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.config import TemporalLinkingConfig
from temporal_linking.similarity import compute_pair_scores
from temporal_linking.types import Detection, Track, TrackStatus


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _make_det(
    frame_num: int,
    det_index: int,
    class_id: int,
    vec: np.ndarray,
    *,
    bbox: np.ndarray | None = None,
    frame_width: float | None = 100.0,
    frame_height: float | None = 100.0,
) -> Detection:
    vec_n = _normalize(vec)
    bbox_xyxy = bbox if bbox is not None else np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32)
    return Detection(
        frame_num=frame_num,
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


def _make_track(
    track_id: int,
    class_id: int,
    vec: np.ndarray,
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
    track.sim_history = deque([0.95], maxlen=5)
    return track


class SimilarityMatrixTests(unittest.TestCase):
    def test_compute_pair_scores_returns_expected_similarity(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.8)
        vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        track = _make_track(track_id=1, class_id=32, vec=vec)
        det = _make_det(frame_num=1, det_index=0, class_id=32, vec=vec)

        scores = compute_pair_scores([track], [det], cfg)

        self.assertEqual(scores.visual.shape, (1, 1))
        self.assertEqual(scores.spatial.shape, (1, 1))
        self.assertEqual(scores.assignment.shape, (1, 1))
        self.assertEqual(scores.eligible.shape, (1, 1))
        self.assertAlmostEqual(float(scores.visual[0, 0]), 1.0, places=5)
        self.assertTrue(float(scores.spatial[0, 0]) > 0.0)
        self.assertTrue(bool(scores.eligible[0, 0]))

    def test_visual_threshold_applies_to_eligibility(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.95)
        track = _make_track(track_id=1, class_id=32, vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        det = _make_det(frame_num=1, det_index=0, class_id=32, vec=np.asarray([0.8, 0.6, 0.0], dtype=np.float32))

        scores = compute_pair_scores([track], [det], cfg)

        self.assertTrue(float(scores.visual[0, 0]) < 0.95)
        self.assertFalse(bool(scores.eligible[0, 0]))
        self.assertTrue(np.isneginf(float(scores.assignment[0, 0])))

    def test_class_mismatch_is_invalid_when_policy_enabled(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.8, match_within_class=True)
        vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        track = _make_track(track_id=1, class_id=32, vec=vec)
        det = _make_det(frame_num=1, det_index=0, class_id=1, vec=vec)

        scores = compute_pair_scores([track], [det], cfg)

        self.assertTrue(np.isneginf(float(scores.visual[0, 0])))
        self.assertFalse(bool(scores.eligible[0, 0]))
        self.assertTrue(np.isneginf(float(scores.assignment[0, 0])))

    def test_spatial_gate_blocks_far_pairs_before_cosine(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.5, max_centroid_distance=0.10)
        vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        track = _make_track(
            track_id=1,
            class_id=32,
            vec=vec,
            bbox=np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
            frame_width=100.0,
            frame_height=100.0,
        )
        det = _make_det(
            frame_num=1,
            det_index=0,
            class_id=32,
            vec=vec,
            bbox=np.asarray([90.0, 90.0, 100.0, 100.0], dtype=np.float32),
            frame_width=100.0,
            frame_height=100.0,
        )

        scores = compute_pair_scores([track], [det], cfg)
        self.assertEqual(float(scores.visual[0, 0]), 0.0)
        self.assertFalse(bool(scores.eligible[0, 0]))
        self.assertTrue(np.isneginf(float(scores.assignment[0, 0])))

    def test_spatial_gate_is_skipped_when_dimensions_missing(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.5, max_centroid_distance=0.10)
        vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        track = _make_track(
            track_id=1,
            class_id=32,
            vec=vec,
            bbox=np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
            frame_width=None,
            frame_height=None,
        )
        det = _make_det(
            frame_num=1,
            det_index=0,
            class_id=32,
            vec=vec,
            bbox=np.asarray([90.0, 90.0, 100.0, 100.0], dtype=np.float32),
            frame_width=None,
            frame_height=None,
        )

        scores = compute_pair_scores([track], [det], cfg)
        self.assertAlmostEqual(float(scores.visual[0, 0]), 1.0, places=5)
        self.assertTrue(bool(scores.eligible[0, 0]))


if __name__ == "__main__":
    unittest.main()
