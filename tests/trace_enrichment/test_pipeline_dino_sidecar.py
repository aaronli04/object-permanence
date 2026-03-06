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

from trace_enrichment.dino import DinoUnavailableError
from trace_enrichment.pipeline import run_trace_enrichment
from trace_enrichment.types import CollectedDetection, CollectedFrame, CollectionStats, HookConfig


def _unit_dino_vec() -> np.ndarray:
    vec = np.ones((384,), dtype=np.float32)
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def _build_frames(*, raw_dim: int, dino_available: bool) -> tuple[list[CollectedFrame], CollectionStats]:
    dino_vec = _unit_dino_vec() if dino_available else None
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
                    dino_vector=dino_vec,
                    dino_available=dino_available,
                )
            ],
        )
    ]
    stats = CollectionStats(total_sampled_frames=1, frames_with_detections=1, total_detections=1)
    return frames, stats


class PipelineDinoSidecarTests(unittest.TestCase):
    def _run_with_mocks(
        self,
        *,
        frames: list[CollectedFrame],
        stats: CollectionStats,
        hook_config: HookConfig,
        dino_loader_side_effect=None,
        env: dict[str, str] | None = None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        manifests: list[dict[str, object]] = []
        enriched_payloads: list[dict[str, object]] = []
        video_fd, video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(video_fd)
        with open(video_path, "wb") as f:
            f.write(b"video")

        def _fake_collect(**_kwargs):
            return frames, stats

        def _fake_fit(_frames, pca_dim):
            self.assertGreaterEqual(int(pca_dim), 2)
            return object(), 2

        def _capture_write(path, payload):
            if path.endswith("projection_manifest.json") and isinstance(payload, dict):
                manifests.append(payload)
            elif path.endswith("enriched_detections.json") and isinstance(payload, list):
                enriched_payloads.extend(payload)

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
        self.assertTrue(enriched_payloads)
        return manifests[-1], enriched_payloads

    def test_dino_sidecar_present_and_excluded_from_pca_input(self) -> None:
        hook = HookConfig(
            layer="4.cv1",
            stride=8,
            requested_layer="multi_layer_embedding",
            layers=("4.cv1", "15", "22.cv3.0"),
            layer_weights=(0.549, 0.351, 0.100),
            multi_layer_enabled=True,
        )
        frames, stats = _build_frames(raw_dim=208, dino_available=True)
        manifest, enriched_payload = self._run_with_mocks(
            frames=frames,
            stats=stats,
            hook_config=hook,
            dino_loader_side_effect=lambda **_: object(),
        )

        self.assertEqual(int(manifest["raw_embedding_dim"]), 208)
        self.assertEqual(str(manifest["dino_role"]), "relink_sidecar")
        self.assertTrue(bool(manifest["dino_enabled"]))
        self.assertNotIn("dino_weight", manifest)
        self.assertNotIn("dino_feature_dim", manifest)

        activation = enriched_payload[0]["detections"][0]["activation"]
        self.assertTrue(bool(activation["dino_available"]))
        self.assertEqual(len(activation["dino_vector"]), 384)

    def test_dino_disabled_stores_null_sidecar_and_runs(self) -> None:
        hook = HookConfig(
            layer="4.cv1",
            stride=8,
            requested_layer="multi_layer_embedding",
            layers=("4.cv1", "15", "22.cv3.0"),
            layer_weights=(0.549, 0.351, 0.100),
            multi_layer_enabled=True,
        )
        frames, stats = _build_frames(raw_dim=208, dino_available=False)
        manifest, enriched_payload = self._run_with_mocks(
            frames=frames,
            stats=stats,
            hook_config=hook,
            dino_loader_side_effect=lambda **_: object(),
            env={"TRACE_DISABLE_DINO": "1"},
        )

        self.assertEqual(int(manifest["raw_embedding_dim"]), 208)
        self.assertFalse(bool(manifest["dino_enabled"]))
        self.assertIsNone(manifest["dino_model"])
        self.assertIsNone(manifest["dino_load_error"])

        activation = enriched_payload[0]["detections"][0]["activation"]
        self.assertFalse(bool(activation["dino_available"]))
        self.assertIsNone(activation["dino_vector"])

    def test_dino_load_error_recorded_with_null_sidecar(self) -> None:
        hook = HookConfig(
            layer="4.cv1",
            stride=8,
            requested_layer="multi_layer_embedding",
            layers=("4.cv1", "15", "22.cv3.0"),
            layer_weights=(0.549, 0.351, 0.100),
            multi_layer_enabled=True,
        )
        frames, stats = _build_frames(raw_dim=208, dino_available=False)
        manifest, enriched_payload = self._run_with_mocks(
            frames=frames,
            stats=stats,
            hook_config=hook,
            dino_loader_side_effect=DinoUnavailableError("hub unavailable"),
        )

        self.assertFalse(bool(manifest["dino_enabled"]))
        self.assertIn("hub unavailable", str(manifest["dino_load_error"]))
        self.assertIsNotNone(manifest["dino_model"])
        activation = enriched_payload[0]["detections"][0]["activation"]
        self.assertFalse(bool(activation["dino_available"]))
        self.assertIsNone(activation["dino_vector"])


if __name__ == "__main__":
    unittest.main()
