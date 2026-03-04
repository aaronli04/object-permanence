from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from common.numeric import topk_l2_renorm, topk_l2_renorm_pad


class TopKProjectionTests(unittest.TestCase):
    def test_padding_after_truncation_preserves_cosine(self) -> None:
        a = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        k = 3

        a_k = topk_l2_renorm(a, topk=k)
        b_k = topk_l2_renorm(b, topk=k)
        cos_k = float(np.dot(a_k, b_k))

        a_pad = topk_l2_renorm_pad(a, topk=k, target_dim=8)
        b_pad = topk_l2_renorm_pad(b, topk=k, target_dim=8)
        cos_pad = float(np.dot(a_pad, b_pad))

        self.assertAlmostEqual(cos_k, cos_pad, places=6)


if __name__ == "__main__":
    unittest.main()
