from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.config import TemporalLinkingConfig
from temporal_linking.relink import run_relink
from temporal_linking.serialize import build_relink_manifest
from temporal_linking.tracker import TrackManager
from temporal_linking.types import Assignment, Detection, Track, TrackStatus


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _make_detection(
    *,
    frame_num: int,
    det_index: int,
    yolo_vec: np.ndarray,
    dino_vec: np.ndarray | None,
) -> Detection:
    yolo_unit = _normalize(yolo_vec)
    dino_unit = _normalize(dino_vec) if dino_vec is not None else None
    return Detection(
        frame_num=frame_num,
        det_index=det_index,
        class_id=32,
        class_name="sports ball",
        bbox_xyxy=np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32),
        confidence=0.95,
        activation_vec=yolo_unit,
        small_crop_flag=False,
        raw_payload={
            "class_id": 32,
            "class_name": "sports ball",
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "confidence": 0.95,
            "activation": {
                "vector": yolo_unit.tolist(),
                "dim": int(yolo_unit.shape[0]),
                "small_crop_flag": False,
                "dino_vector": None if dino_unit is None else dino_unit.tolist(),
                "dino_available": bool(dino_unit is not None),
            },
        },
        frame_width=100.0,
        frame_height=100.0,
        dino_vector=dino_unit,
    )


def _make_closed_track(
    *,
    track_id: int,
    frame_start: int,
    frame_end: int,
    yolo_vec: np.ndarray,
    dino_vec: np.ndarray | None,
) -> Track:
    yolo_unit = _normalize(yolo_vec)
    track = Track(
        track_id=track_id,
        class_id=32,
        class_name="sports ball",
        status=TrackStatus.CLOSED,
        start_frame=frame_start,
        last_seen_frame=frame_end,
        hits=2,
    )
    track.obs_vecs = [yolo_unit.copy(), yolo_unit.copy()]
    track.obs_positions = [
        (10.0, 10.0, frame_start),
        (10.0, 10.0, frame_end),
    ]
    track.dino_vector = _normalize(dino_vec) if dino_vec is not None else None
    return track


class RelinkDinoTests(unittest.TestCase):
    def test_close_builds_track_dino_vector_when_min_observations_met(self) -> None:
        cfg = TemporalLinkingConfig(
            similarity_threshold=0.7,
            min_hits_to_activate=1,
            relink_dino_min_detections=2,
        )
        manager = TrackManager(cfg)

        det0 = _make_detection(
            frame_num=0,
            det_index=0,
            yolo_vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            dino_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        track = manager.spawn(det0, frame_num=0)
        det1 = _make_detection(
            frame_num=1,
            det_index=0,
            yolo_vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            dino_vec=np.asarray([0.9, 0.1], dtype=np.float32),
        )
        assignment = Assignment(
            track_id=track.track_id,
            det_index=0,
            visual_similarity=0.95,
            spatial_score=0.8,
            total_score=1.0,
            source_track_status=track.status,
        )
        manager.apply_match(track.track_id, det1, assignment, frame_num=1)
        manager.close(track, end_frame=1)

        self.assertIsNotNone(track.dino_vector)
        self.assertAlmostEqual(float(np.linalg.norm(track.dino_vector)), 1.0, places=6)

    def test_close_sets_track_dino_vector_none_when_below_min_observations(self) -> None:
        cfg = TemporalLinkingConfig(
            similarity_threshold=0.7,
            min_hits_to_activate=1,
            relink_dino_min_detections=2,
        )
        manager = TrackManager(cfg)

        det0 = _make_detection(
            frame_num=0,
            det_index=0,
            yolo_vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            dino_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        track = manager.spawn(det0, frame_num=0)
        det1 = _make_detection(
            frame_num=1,
            det_index=0,
            yolo_vec=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            dino_vec=None,
        )
        assignment = Assignment(
            track_id=track.track_id,
            det_index=0,
            visual_similarity=0.95,
            spatial_score=0.8,
            total_score=1.0,
            source_track_status=track.status,
        )
        manager.apply_match(track.track_id, det1, assignment, frame_num=1)
        manager.close(track, end_frame=1)

        self.assertIsNone(track.dino_vector)

    def test_relink_uses_dino_when_both_track_vectors_available(self) -> None:
        cfg = TemporalLinkingConfig(
            similarity_threshold=0.7,
            relink_min_track_hits=1,
            relink_threshold=0.95,
            relink_dino_threshold=0.80,
            relink_fallback_threshold=1.0,
            relink_use_dino=True,
        )
        pred = _make_closed_track(
            track_id=1,
            frame_start=0,
            frame_end=5,
            yolo_vec=np.asarray([1.0, 0.0], dtype=np.float32),
            dino_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        succ = _make_closed_track(
            track_id=2,
            frame_start=10,
            frame_end=15,
            yolo_vec=np.asarray([-1.0, 0.0], dtype=np.float32),
            dino_vec=np.asarray([0.95, 0.05], dtype=np.float32),
        )
        _merge_map, relink_result = run_relink([pred, succ], cfg)
        accepted = relink_result["accepted_edges"]
        stats = relink_result["stats"]

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["method"], "dino")
        self.assertEqual(int(stats["relink_dino_accepted"]), 1)
        self.assertEqual(int(stats["relink_yolo_accepted"]), 0)
        self.assertAlmostEqual(float(stats["relink_dino_coverage"]), 1.0, places=6)

    def test_relink_falls_back_to_yolo_when_dino_missing(self) -> None:
        cfg = TemporalLinkingConfig(
            similarity_threshold=0.7,
            relink_min_track_hits=1,
            relink_threshold=0.80,
            relink_dino_threshold=0.95,
            relink_fallback_threshold=1.0,
            relink_use_dino=True,
        )
        pred = _make_closed_track(
            track_id=1,
            frame_start=0,
            frame_end=5,
            yolo_vec=np.asarray([1.0, 0.0], dtype=np.float32),
            dino_vec=None,
        )
        succ = _make_closed_track(
            track_id=2,
            frame_start=10,
            frame_end=15,
            yolo_vec=np.asarray([0.95, 0.05], dtype=np.float32),
            dino_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        _merge_map, relink_result = run_relink([pred, succ], cfg)
        accepted = relink_result["accepted_edges"]
        stats = relink_result["stats"]

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["method"], "yolo")
        self.assertEqual(int(stats["relink_dino_accepted"]), 0)
        self.assertEqual(int(stats["relink_yolo_accepted"]), 1)
        self.assertAlmostEqual(float(stats["relink_dino_coverage"]), 0.0, places=6)

    def test_relink_no_dino_flag_forces_yolo_path(self) -> None:
        cfg = TemporalLinkingConfig(
            similarity_threshold=0.7,
            relink_min_track_hits=1,
            relink_threshold=0.80,
            relink_dino_threshold=0.80,
            relink_fallback_threshold=1.0,
            relink_use_dino=False,
        )
        pred = _make_closed_track(
            track_id=1,
            frame_start=0,
            frame_end=5,
            yolo_vec=np.asarray([1.0, 0.0], dtype=np.float32),
            dino_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        succ = _make_closed_track(
            track_id=2,
            frame_start=10,
            frame_end=15,
            yolo_vec=np.asarray([0.95, 0.05], dtype=np.float32),
            dino_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        )
        _merge_map, relink_result = run_relink([pred, succ], cfg)
        accepted = relink_result["accepted_edges"]
        stats = relink_result["stats"]

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["method"], "yolo")
        self.assertAlmostEqual(float(stats["relink_dino_coverage"]), 0.0, places=6)

    def test_relink_manifest_preserves_dino_coverage_float(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.7)
        manifest = build_relink_manifest(
            cfg=cfg,
            relink_result={
                "stats": {
                    "num_accepted_edges": 3,
                    "relink_dino_accepted": 2,
                    "relink_yolo_accepted": 1,
                    "relink_dino_coverage": 0.5,
                },
                "accepted_edges": [],
            },
            merge_map={},
        )
        self.assertAlmostEqual(float(manifest["stats"]["relink_dino_coverage"]), 0.5, places=6)
        self.assertIn("relink_use_dino", manifest["config"])
        self.assertIn("relink_dino_threshold", manifest["config"])


if __name__ == "__main__":
    unittest.main()
