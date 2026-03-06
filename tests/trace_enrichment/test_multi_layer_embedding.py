from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from common.numeric import l2_normalize
from common.warn_once import WarnOnce
from trace_enrichment.constants import DEFAULT_HEAD_LAYER
from trace_enrichment.pipeline import _build_hook_config, _build_weighted_embedding
from trace_enrichment.types import HookConfig


class MultiLayerEmbeddingTests(unittest.TestCase):
    def test_weighted_embedding_concatenates_and_final_l2_normalizes(self) -> None:
        layer_names = ("a", "b", "c")
        layer_weights = (0.4, 0.2, 0.4)
        layer_outputs = {
            "a": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "b": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "c": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
        }
        vectors = [
            np.asarray([3.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 4.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float32),
        ]

        with mock.patch(
            "trace_enrichment.pipeline.build_raw_activation_vector",
            side_effect=[(vectors[0], False), (vectors[1], False), (vectors[2], True)],
        ):
            combined, small = _build_weighted_embedding(
                layer_outputs=layer_outputs,
                layer_names=layer_names,
                layer_weights=layer_weights,
                batch_index=0,
                bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
                pool=object(),
                frame_h=10,
                frame_w=10,
                warn=WarnOnce(),
            )

        parts = [
            l2_normalize(vectors[0]) * 0.4,
            l2_normalize(vectors[1]) * 0.2,
            l2_normalize(vectors[2]) * 0.4,
        ]
        expected = l2_normalize(np.concatenate(parts, axis=0))
        self.assertEqual(combined.shape[0], 7)
        self.assertTrue(np.allclose(combined, expected, atol=1e-6))
        self.assertTrue(np.isclose(float(np.linalg.norm(combined)), 1.0, atol=1e-6))
        self.assertTrue(small)

    def test_weighted_embedding_reweights_when_one_layer_missing(self) -> None:
        layer_names = ("a", "b", "c")
        layer_weights = (0.5, 0.3, 0.2)
        layer_outputs = {
            "a": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "b": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "c": None,
        }
        vec_a = np.asarray([2.0, 0.0], dtype=np.float32)
        vec_b = np.asarray([0.0, 3.0], dtype=np.float32)
        warn = WarnOnce()

        with mock.patch(
            "trace_enrichment.pipeline.build_raw_activation_vector",
            side_effect=[(vec_a, False), (vec_b, False)],
        ):
            combined, _small = _build_weighted_embedding(
                layer_outputs=layer_outputs,
                layer_names=layer_names,
                layer_weights=layer_weights,
                batch_index=0,
                bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
                pool=object(),
                frame_h=10,
                frame_w=10,
                warn=warn,
            )

        # Missing layer c => weights renormalize from [0.5, 0.3] to [0.625, 0.375].
        expected = l2_normalize(
            np.concatenate(
                [
                    l2_normalize(vec_a) * 0.625,
                    l2_normalize(vec_b) * 0.375,
                ],
                axis=0,
            )
        )
        self.assertTrue(np.allclose(combined, expected, atol=1e-6))
        self.assertTrue(warn.seen("missing_output:c"))

    def test_weighted_embedding_raises_when_less_than_two_layers_available(self) -> None:
        layer_names = ("a", "b", "c")
        layer_weights = (0.5, 0.3, 0.2)
        layer_outputs = {
            "a": torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            "b": None,
            "c": None,
        }
        with mock.patch(
            "trace_enrichment.pipeline.build_raw_activation_vector",
            side_effect=[(np.asarray([1.0, 0.0], dtype=np.float32), False)],
        ):
            with self.assertRaises(RuntimeError):
                _build_weighted_embedding(
                    layer_outputs=layer_outputs,
                    layer_names=layer_names,
                    layer_weights=layer_weights,
                    batch_index=0,
                    bbox_xyxy=[0.0, 0.0, 1.0, 1.0],
                    pool=object(),
                    frame_h=10,
                    frame_w=10,
                    warn=WarnOnce(),
                )

    def test_build_hook_config_uses_single_layer_when_custom_head_requested(self) -> None:
        with mock.patch("trace_enrichment.pipeline.resolve_hook_layer_name", return_value="7") as resolve:
            cfg = _build_hook_config(yolo=object(), layer_name="7", stride=8)
        self.assertFalse(cfg.multi_layer_enabled)
        self.assertEqual(cfg.layer, "7")
        self.assertEqual(cfg.layers, ("7",))
        resolve.assert_called_once()

    def test_build_hook_config_uses_multi_layer_by_default(self) -> None:
        expected = HookConfig(
            layer="4.cv1",
            stride=8,
            requested_layer="multi_layer_embedding",
            layers=("4.cv1", "15", "22.cv3.0"),
            layer_weights=(0.4, 0.25, 0.35),
            multi_layer_enabled=True,
        )
        with (
            mock.patch("trace_enrichment.pipeline._multi_layer_embedding_disabled", return_value=False),
            mock.patch("trace_enrichment.pipeline._resolve_multi_layer_hook_config", return_value=expected) as resolver,
        ):
            cfg = _build_hook_config(yolo=object(), layer_name=DEFAULT_HEAD_LAYER, stride=8)
        self.assertTrue(cfg.multi_layer_enabled)
        self.assertEqual(cfg.layers, ("4.cv1", "15", "22.cv3.0"))
        resolver.assert_called_once()


if __name__ == "__main__":
    unittest.main()
