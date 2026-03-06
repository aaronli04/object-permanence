from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from trace_enrichment.constants import DINO_EMBEDDING_DIM
from trace_enrichment.dino import DinoEmbedder, extract_dino_embedding


class _FakeDinoModel:
    def __call__(self, _tensor):
        return torch.ones((1, DINO_EMBEDDING_DIM), dtype=torch.float32)


class DinoTests(unittest.TestCase):
    def _embedder(self) -> DinoEmbedder:
        return DinoEmbedder(
            model=_FakeDinoModel(),
            device=torch.device("cpu"),
            model_name="fake_dino",
            feature_dim=DINO_EMBEDDING_DIM,
            input_size=224,
            tiny_crop_min=32,
            crop_padding_ratio=0.05,
        )

    def test_extract_dino_embedding_returns_normalized_vector_for_valid_crop(self) -> None:
        frame = np.full((96, 96, 3), 127, dtype=np.uint8)
        result = extract_dino_embedding(
            frame_bgr=frame,
            bbox_xyxy=[10.0, 10.0, 80.0, 80.0],
            embedder=self._embedder(),
        )

        self.assertTrue(result.valid_crop)
        self.assertFalse(result.tiny_crop)
        self.assertEqual(int(result.vector.shape[0]), DINO_EMBEDDING_DIM)
        self.assertAlmostEqual(float(np.linalg.norm(result.vector)), 1.0, places=6)

    def test_extract_dino_embedding_returns_zero_vector_for_invalid_crop(self) -> None:
        frame = np.full((96, 96, 3), 127, dtype=np.uint8)
        result = extract_dino_embedding(
            frame_bgr=frame,
            bbox_xyxy=[20.0, 20.0, 20.0, 40.0],  # invalid width
            embedder=self._embedder(),
        )

        self.assertFalse(result.valid_crop)
        self.assertFalse(result.tiny_crop)
        self.assertEqual(int(result.vector.shape[0]), DINO_EMBEDDING_DIM)
        self.assertEqual(float(np.linalg.norm(result.vector)), 0.0)

    def test_extract_dino_embedding_returns_zero_vector_for_tiny_crop(self) -> None:
        frame = np.full((96, 96, 3), 127, dtype=np.uint8)
        result = extract_dino_embedding(
            frame_bgr=frame,
            bbox_xyxy=[10.0, 10.0, 20.0, 20.0],  # tiny crop
            embedder=self._embedder(),
        )

        self.assertTrue(result.valid_crop)
        self.assertTrue(result.tiny_crop)
        self.assertEqual(int(result.vector.shape[0]), DINO_EMBEDDING_DIM)
        self.assertEqual(float(np.linalg.norm(result.vector)), 0.0)


if __name__ == "__main__":
    unittest.main()

