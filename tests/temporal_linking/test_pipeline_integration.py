from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.config import TemporalLinkingConfig
from temporal_linking.pipeline import link_video_frames, run_temporal_linking
from temporal_linking.types import Detection, FrameDetections


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _make_detection(frame_num: int, det_index: int, vec: np.ndarray) -> Detection:
    vec_n = _normalize(vec)
    return Detection(
        frame_num=frame_num,
        det_index=det_index,
        class_id=32,
        class_name="sports ball",
        bbox_xyxy=np.asarray([10.0, 10.0, 20.0, 20.0], dtype=np.float32),
        confidence=0.95,
        activation_vec=vec_n,
        small_crop_flag=False,
        raw_payload={
            "class_id": 32,
            "class_name": "sports ball",
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "confidence": 0.95,
            "activation": {"vector": vec_n.tolist(), "small_crop_flag": False},
        },
    )


class PipelineIntegrationTests(unittest.TestCase):
    def test_recovers_lost_track_with_same_similarity_threshold(self) -> None:
        cfg = TemporalLinkingConfig(
            similarity_threshold=0.9,
            max_lost_frames=2,
            min_hits_to_activate=1,
        )
        vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        frames = [
            FrameDetections(frame_num=0, detections=[_make_detection(0, 0, vec)]),
            FrameDetections(frame_num=1, detections=[]),
            FrameDetections(frame_num=2, detections=[_make_detection(2, 0, vec)]),
        ]

        result = link_video_frames(frames, cfg, enriched_json_path="synthetic.json")

        first_track = result.linked_frames[0]["detections"][0]["temporal_link"]["track_id"]
        second_track = result.linked_frames[2]["detections"][0]["temporal_link"]["track_id"]
        self.assertEqual(first_track, second_track)
        self.assertEqual(result.manifest_payload["stats"]["num_recoveries"], 1)

    def test_closes_track_after_max_lost_and_spawns_new_track(self) -> None:
        cfg = TemporalLinkingConfig(
            similarity_threshold=0.9,
            max_lost_frames=1,
            min_hits_to_activate=1,
        )
        vec = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        frames = [
            FrameDetections(frame_num=0, detections=[_make_detection(0, 0, vec)]),
            FrameDetections(frame_num=1, detections=[]),
            FrameDetections(frame_num=2, detections=[]),
            FrameDetections(frame_num=3, detections=[_make_detection(3, 0, vec)]),
        ]

        result = link_video_frames(frames, cfg, enriched_json_path="synthetic.json")

        first_track = result.linked_frames[0]["detections"][0]["temporal_link"]["track_id"]
        later_track = result.linked_frames[3]["detections"][0]["temporal_link"]["track_id"]
        self.assertNotEqual(first_track, later_track)

    def test_noop_relink_thresholds_keep_single_sweep_behavior(self) -> None:
        cfg_noop = TemporalLinkingConfig(
            similarity_threshold=0.7,
            max_lost_frames=1,
            min_hits_to_activate=1,
            relink_min_track_hits=1,
            relink_threshold=1.0,
            relink_fallback_threshold=1.0,
        )
        cfg_merge = TemporalLinkingConfig(
            similarity_threshold=0.7,
            max_lost_frames=1,
            min_hits_to_activate=1,
            relink_min_track_hits=1,
            relink_threshold=0.55,
            relink_fallback_threshold=0.40,
        )

        vec_first = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        vec_late = np.asarray([0.95, 0.31, 0.0], dtype=np.float32)
        frames = [
            FrameDetections(frame_num=0, detections=[_make_detection(0, 0, vec_first)]),
            FrameDetections(frame_num=1, detections=[]),
            FrameDetections(frame_num=2, detections=[]),
            FrameDetections(frame_num=3, detections=[_make_detection(3, 0, vec_late)]),
        ]

        result_noop = link_video_frames(frames, cfg_noop, enriched_json_path="synthetic.json")
        first_track_noop = result_noop.linked_frames[0]["detections"][0]["temporal_link"]["track_id"]
        later_track_noop = result_noop.linked_frames[3]["detections"][0]["temporal_link"]["track_id"]
        self.assertNotEqual(first_track_noop, later_track_noop)

        result_merge = link_video_frames(frames, cfg_merge, enriched_json_path="synthetic.json")
        first_track_merge = result_merge.linked_frames[0]["detections"][0]["temporal_link"]["track_id"]
        later_track_merge = result_merge.linked_frames[3]["detections"][0]["temporal_link"]["track_id"]
        self.assertEqual(first_track_merge, later_track_merge)

    def test_run_temporal_linking_writes_relink_manifest(self) -> None:
        vec = np.zeros((256,), dtype=np.float32)
        vec[0] = 1.0
        enriched = [
            {
                "frame_num": 0,
                "detections": [
                    {
                        "class_id": 32,
                        "class_name": "sports ball",
                        "bbox": [10.0, 10.0, 20.0, 20.0],
                        "confidence": 0.9,
                        "activation": {"vector": vec.tolist(), "dim": 256, "small_crop_flag": False},
                    }
                ],
            },
            {"frame_num": 1, "detections": []},
            {"frame_num": 2, "detections": []},
            {
                "frame_num": 3,
                "detections": [
                    {
                        "class_id": 32,
                        "class_name": "sports ball",
                        "bbox": [10.0, 10.0, 20.0, 20.0],
                        "confidence": 0.9,
                        "activation": {"vector": vec.tolist(), "dim": 256, "small_crop_flag": False},
                    }
                ],
            },
        ]

        cfg = TemporalLinkingConfig(
            similarity_threshold=0.7,
            max_lost_frames=1,
            min_hits_to_activate=1,
            relink_min_track_hits=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            enriched_path = os.path.join(tmpdir, "enriched.json")
            with open(enriched_path, "w", encoding="utf-8") as f:
                json.dump(enriched, f)

            outputs = run_temporal_linking(
                enriched_json_path=enriched_path,
                output_dir=tmpdir,
                config=cfg,
            )

            self.assertIn("relink_manifest", outputs)
            self.assertTrue(os.path.exists(outputs["relink_manifest"]))

            with open(outputs["relink_manifest"], "r", encoding="utf-8") as f:
                relink_manifest = json.load(f)
            self.assertEqual(relink_manifest["schema_version"], "temporal_linking_relink_manifest_v1")
            self.assertIn("stats", relink_manifest)


if __name__ == "__main__":
    unittest.main()
