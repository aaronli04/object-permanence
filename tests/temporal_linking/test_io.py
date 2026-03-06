from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.io import load_enriched_frames


class TemporalLinkingIOTests(unittest.TestCase):
    def test_load_enriched_frames_accepts_dynamic_activation_dim(self) -> None:
        vec = [0.0] * 128
        vec[0] = 1.0
        payload = [
            {
                "frame_num": 0,
                "detections": [
                    {
                        "class_id": 32,
                        "class_name": "sports ball",
                        "bbox": [0.0, 0.0, 128.0, 96.0],
                        "confidence": 0.9,
                        "activation": {"vector": vec, "dim": 128, "small_crop_flag": False},
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "enriched.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            frames = load_enriched_frames(path)

        self.assertEqual(len(frames), 1)
        self.assertEqual(len(frames[0].detections), 1)
        det = frames[0].detections[0]
        self.assertEqual(int(det.activation_vec.shape[0]), 128)
        self.assertEqual(det.frame_width, 128.0)
        self.assertEqual(det.frame_height, 96.0)

    def test_load_enriched_frames_rejects_mismatched_vector_dims(self) -> None:
        vec_128 = [0.0] * 128
        vec_128[0] = 1.0
        vec_64 = [0.0] * 64
        vec_64[0] = 1.0
        payload = [
            {
                "frame_num": 0,
                "detections": [
                    {
                        "class_id": 32,
                        "class_name": "sports ball",
                        "bbox": [0.0, 0.0, 64.0, 64.0],
                        "confidence": 0.9,
                        "activation": {"vector": vec_128, "dim": 128, "small_crop_flag": False},
                    }
                ],
            },
            {
                "frame_num": 1,
                "detections": [
                    {
                        "class_id": 32,
                        "class_name": "sports ball",
                        "bbox": [0.0, 0.0, 64.0, 64.0],
                        "confidence": 0.9,
                        "activation": {"vector": vec_64, "dim": 64, "small_crop_flag": False},
                    }
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "enriched.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            with self.assertRaises(ValueError):
                load_enriched_frames(path)

    def test_load_enriched_frames_rejects_declared_dim_mismatch(self) -> None:
        vec = [0.0] * 128
        vec[0] = 1.0
        payload = [
            {
                "frame_num": 0,
                "detections": [
                    {
                        "class_id": 32,
                        "class_name": "sports ball",
                        "bbox": [0.0, 0.0, 64.0, 64.0],
                        "confidence": 0.9,
                        "activation": {"vector": vec, "dim": 64, "small_crop_flag": False},
                    }
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "enriched.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            with self.assertRaises(ValueError):
                load_enriched_frames(path)


if __name__ == "__main__":
    unittest.main()
