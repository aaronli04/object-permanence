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

from trace_enrichment.pipeline import run_trace_enrichment
from trace_enrichment.dino import DinoUnavailableError
from trace_enrichment.types import CollectedDetection, CollectedFrame, CollectionStats, HookConfig


def _build_frames(raw_dim: int) -> tuple[list[CollectedFrame], CollectionStats]:
    frames = [
        CollectedFrame(
            frame_num=0,
            detections=[
                CollectedDetection(
                    class_id=32,
                    class_name="sports ball",
                    bbox=[0.0, 0.0, 32.0, 32.0],
                    confidence=0.9,
                    raw_vector=np.ones((raw_dim,), dtype=np.float32),
                    projected_vector=np.asarray([1.0, 0.0], dtype=np.float32),
                )
            ],
        )
    ]
    stats = CollectionStats(total_sampled_frames=1, frames_with_detections=1, total_detections=1)
    return frames, stats


class PipelineDinoIntegrationTests(unittest.TestCase):
    def _run_with_mocks(
        self,
        *,
        raw_dim: int,
        hook_config: HookConfig,
        dino_loader_side_effect=None,
        env: dict[str, str] | None = None,
    ) -> dict[str, object]:
        frames, stats = _build_frames(raw_dim)
        manifests: list[dict[str, object]] = []
        video_fd, video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(video_fd)
        with open(video_path, "wb") as f:
            f.write(b"video")

        def _fake_collect(**_kwargs):
            return frames, stats

        def _fake_fit(_frames, pca_dim):
            self.assertGreaterEqual(int(pca_dim), 2)
            return object(), 2

        def _capture_write(_path, payload):
            if isinstance(payload, dict) and "schema_version" in payload:
                manifests.append(payload)

        patchers = [
            mock.patch("trace_enrichment.pipeline.load_yolo", return_value=object()),
            mock.patch("trace_enrichment.pipeline._build_hook_config", return_value=hook_config),
            mock.patch("trace_enrichment.pipeline.collect_single_pass_trace", side_effect=_fake_collect),
            mock.patch("trace_enrichment.pipeline.fit_pca_and_project", side_effect=_fake_fit),
            mock.patch("trace_enrichment.pipeline.joblib.dump"),
            mock.patch("trace_enrichment.pipeline.write_json", side_effect=_capture_write),
            mock.patch("trace_enrichment.pipeline.load_dino_embedder", side_effect=dino_loader_side_effect),
        ]
        env_patch = mock.patch.dict(os.environ, env or {}, clear=False)
        with env_patch:
            with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5], patchers[6]:
                with tempfile.TemporaryDirectory() as tmpdir:
                    run_trace_enrichment(
                        video_path=video_path,
                        model_name="yolov8n.pt",
                        output_dir=tmpdir,
                        sample_rate=5,
                        layer_name=hook_config.requested_layer,
                        stride=8,
                        batch_size=1,
                        pca_dim=128,
                    )
        os.remove(video_path)
        self.assertTrue(manifests)
        return manifests[-1]

    def test_manifest_raw_dim_is_592_when_dino_enabled(self) -> None:
        hook = HookConfig(
            layer="4.cv1",
            stride=8,
            requested_layer="multi_layer_embedding",
            layers=("4.cv1", "15", "22.cv3.0"),
            layer_weights=(0.549, 0.351, 0.100),
            multi_layer_enabled=True,
        )
        manifest = self._run_with_mocks(raw_dim=592, hook_config=hook, dino_loader_side_effect=lambda **_: object())
        self.assertEqual(int(manifest["raw_embedding_dim"]), 592)
        self.assertEqual(int(manifest["raw_activation_dim"]), 592)
        self.assertTrue(bool(manifest["dino_enabled"]))
        self.assertLessEqual(int(manifest["projection_dim"]), 128)

    def test_manifest_records_dino_load_error_and_falls_back(self) -> None:
        hook = HookConfig(
            layer="4.cv1",
            stride=8,
            requested_layer="multi_layer_embedding",
            layers=("4.cv1", "15", "22.cv3.0"),
            layer_weights=(0.549, 0.351, 0.100),
            multi_layer_enabled=True,
        )
        manifest = self._run_with_mocks(
            raw_dim=208,
            hook_config=hook,
            dino_loader_side_effect=DinoUnavailableError("hub unavailable"),
        )
        self.assertEqual(int(manifest["raw_embedding_dim"]), 208)
        self.assertFalse(bool(manifest["dino_enabled"]))
        self.assertIn("hub unavailable", str(manifest["dino_load_error"]))

    def test_trace_disable_dino_forces_yolo_only_raw_dim(self) -> None:
        hook = HookConfig(
            layer="4.cv1",
            stride=8,
            requested_layer="multi_layer_embedding",
            layers=("4.cv1", "15", "22.cv3.0"),
            layer_weights=(0.549, 0.351, 0.100),
            multi_layer_enabled=True,
        )
        manifest = self._run_with_mocks(
            raw_dim=208,
            hook_config=hook,
            dino_loader_side_effect=lambda **_: object(),
            env={"TRACE_DISABLE_DINO": "1"},
        )
        self.assertEqual(int(manifest["raw_embedding_dim"]), 208)
        self.assertFalse(bool(manifest["dino_enabled"]))
        self.assertEqual(str(manifest["dino_load_error"]), "")

    def test_single_layer_with_dino_has_single_plus_384_dim(self) -> None:
        hook = HookConfig(
            layer="7",
            stride=8,
            requested_layer="7",
            layers=("7",),
            layer_weights=(1.0,),
            multi_layer_enabled=False,
        )
        manifest = self._run_with_mocks(raw_dim=416, hook_config=hook, dino_loader_side_effect=lambda **_: object())
        self.assertEqual(int(manifest["raw_embedding_dim"]), 416)
        self.assertTrue(bool(manifest["dino_enabled"]))


if __name__ == "__main__":
    unittest.main()
