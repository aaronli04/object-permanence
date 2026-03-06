from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from trace_enrichment.pipeline import build_enriched_payload, build_manifest, fit_pca_and_project
from trace_enrichment.types import CollectedDetection, CollectedFrame, CollectionStats, HookConfig, RunConfig


class _FakePCA:
    def __init__(self, n_components: int, svd_solver: str, random_state: int) -> None:
        self.n_components = int(n_components)
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        return matrix[:, : self.n_components]


class ProjectionBehaviorTests(unittest.TestCase):
    def test_fit_pca_and_project_does_not_pad_beyond_effective_dim(self) -> None:
        frames = [
            CollectedFrame(
                frame_num=0,
                detections=[
                    CollectedDetection(
                        class_id=32,
                        class_name="sports ball",
                        bbox=[0.0, 0.0, 1.0, 1.0],
                        confidence=0.9,
                        raw_vector=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
            ),
            CollectedFrame(
                frame_num=1,
                detections=[
                    CollectedDetection(
                        class_id=32,
                        class_name="sports ball",
                        bbox=[0.0, 0.0, 1.0, 1.0],
                        confidence=0.9,
                        raw_vector=np.asarray([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
            ),
            CollectedFrame(
                frame_num=2,
                detections=[
                    CollectedDetection(
                        class_id=32,
                        class_name="sports ball",
                        bbox=[0.0, 0.0, 1.0, 1.0],
                        confidence=0.9,
                        raw_vector=np.asarray([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                    )
                ],
            ),
        ]

        with mock.patch("trace_enrichment.pipeline.PCA", _FakePCA):
            _pca, effective_dim = fit_pca_and_project(frames, pca_dim=4)

        self.assertEqual(effective_dim, 3)
        projected = frames[0].detections[0].projected_vector
        assert projected is not None
        self.assertEqual(int(projected.shape[0]), 3)
        self.assertAlmostEqual(float(np.linalg.norm(projected)), 1.0, places=6)

    def test_build_enriched_payload_uses_actual_projection_dim(self) -> None:
        frame = CollectedFrame(
            frame_num=0,
            detections=[
                CollectedDetection(
                    class_id=32,
                    class_name="sports ball",
                    bbox=[1.0, 2.0, 3.0, 4.0],
                    confidence=0.9,
                    projected_vector=np.asarray([0.6, 0.8], dtype=np.float32),
                )
            ],
        )
        hook_config = HookConfig(layer="15", stride=8, requested_layer="15", layers=("15",), layer_weights=(1.0,))

        payload = build_enriched_payload([frame], projection_dim=2, hook_config=hook_config)
        activation = payload[0]["detections"][0]["activation"]
        self.assertEqual(activation["dim"], 2)
        self.assertEqual(activation["projection"], "pca_2")
        self.assertEqual(len(activation["vector"]), 2)

    def test_build_manifest_records_actual_and_requested_projection_dims(self) -> None:
        hook_config = HookConfig(layer="15", stride=8, requested_layer="15", layers=("15",), layer_weights=(1.0,))
        frame = CollectedFrame(
            frame_num=0,
            detections=[
                CollectedDetection(
                    class_id=32,
                    class_name="sports ball",
                    bbox=[0.0, 0.0, 1.0, 1.0],
                    confidence=0.9,
                    raw_vector=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    projected_vector=np.asarray([1.0, 0.0], dtype=np.float32),
                )
            ],
        )
        stats = CollectionStats(total_sampled_frames=1, frames_with_detections=1, total_detections=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            with open(video_path, "wb") as f:
                f.write(b"video")
            run_config = RunConfig(
                video_path=video_path,
                model_name="yolov8n.pt",
                sample_rate=5,
                batch_size=8,
                pca_dim=128,
                output_dir=tmpdir,
            )
            manifest = build_manifest(
                run_config=run_config,
                hook_config=hook_config,
                effective_pca_dim=2,
                frames=[frame],
                stats=stats,
            )

        self.assertEqual(manifest["projection_dim"], 2)
        self.assertEqual(manifest["projection_dim_requested"], 128)
        self.assertEqual(manifest["fitted_pca_components"], 2)


if __name__ == "__main__":
    unittest.main()
