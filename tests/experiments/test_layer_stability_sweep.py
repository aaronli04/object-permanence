from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stderr

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import layer_stability_sweep as lss


class _FakeBox:
    def __init__(self, cls_value: int, conf_value: float, xyxy: list[float], track_id: int | None) -> None:
        self.cls = torch.tensor([cls_value], dtype=torch.float32)
        self.conf = torch.tensor([conf_value], dtype=torch.float32)
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.id = None if track_id is None else torch.tensor([track_id], dtype=torch.float32)


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class LayerStabilitySweepTests(unittest.TestCase):
    def test_compute_separability_high_when_between_var_dominates(self) -> None:
        vectors_by_group = {
            1: [
                np.asarray([0.0, 0.0], dtype=np.float32),
                np.asarray([0.1, -0.1], dtype=np.float32),
            ],
            2: [
                np.asarray([10.0, 10.0], dtype=np.float32),
                np.asarray([10.1, 9.9], dtype=np.float32),
            ],
        }
        within_var, between_var, separability, group_count = lss._compute_separability(vectors_by_group)
        self.assertEqual(group_count, 2)
        self.assertGreater(between_var, within_var)
        self.assertGreater(separability, 1.0)

    def test_compute_separability_zero_when_group_means_identical(self) -> None:
        vectors_by_group = {
            1: [
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([1.0, 2.0], dtype=np.float32),
            ],
            2: [
                np.asarray([1.0, 2.0], dtype=np.float32),
                np.asarray([1.0, 2.0], dtype=np.float32),
            ],
        }
        within_var, between_var, separability, group_count = lss._compute_separability(vectors_by_group)
        self.assertEqual(group_count, 2)
        self.assertEqual(within_var, 0.0)
        self.assertEqual(between_var, 0.0)
        self.assertEqual(separability, 0.0)

    def test_compute_separability_single_group_is_zero(self) -> None:
        vectors_by_group = {
            7: [
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 1.0], dtype=np.float32),
            ]
        }
        within_var, between_var, separability, group_count = lss._compute_separability(vectors_by_group)
        self.assertEqual(group_count, 1)
        self.assertEqual(within_var, 0.0)
        self.assertEqual(between_var, 0.0)
        self.assertEqual(separability, 0.0)

    def test_layer_rows_sort_by_separability_before_cosine(self) -> None:
        high_sep = lss.LayerAccum(module_type="X", feature_dim=2)
        high_sep.norms = [1.0, 1.1]
        high_sep.vectors_by_frame = {
            0: [np.asarray([1.0, 0.0], dtype=np.float32)],
            1: [np.asarray([0.0, 1.0], dtype=np.float32)],
        }
        high_sep.vectors_by_group = {
            1: [
                np.asarray([0.0, 0.0], dtype=np.float32),
                np.asarray([0.1, -0.1], dtype=np.float32),
            ],
            2: [
                np.asarray([8.0, 8.0], dtype=np.float32),
                np.asarray([8.1, 7.9], dtype=np.float32),
            ],
        }

        high_cos_low_sep = lss.LayerAccum(module_type="Y", feature_dim=2)
        high_cos_low_sep.norms = [1.0, 1.2]
        high_cos_low_sep.vectors_by_frame = {
            0: [np.asarray([1.0, 0.0], dtype=np.float32)],
            1: [np.asarray([1.0, 0.0], dtype=np.float32)],
        }
        high_cos_low_sep.vectors_by_group = {
            1: [np.asarray([1.0, 1.0], dtype=np.float32), np.asarray([1.0, 1.0], dtype=np.float32)],
            2: [np.asarray([1.0, 1.0], dtype=np.float32), np.asarray([1.0, 1.0], dtype=np.float32)],
        }

        layer_data = {"high_sep": high_sep, "high_cos_low_sep": high_cos_low_sep}
        rows = lss._layer_rows(layer_data=layer_data, sampled_frames=[0, 1])
        self.assertEqual(rows[0]["layer_name"], "high_sep")
        self.assertGreater(rows[0]["separability"], rows[1]["separability"])

    def test_layer_rows_warns_on_single_group(self) -> None:
        accum = lss.LayerAccum(module_type="X", feature_dim=2)
        accum.norms = [1.0, 1.2]
        accum.vectors_by_frame = {
            0: [np.asarray([1.0, 0.0], dtype=np.float32)],
            1: [np.asarray([1.0, 0.0], dtype=np.float32)],
        }
        accum.vectors_by_group = {
            5: [np.asarray([1.0, 1.0], dtype=np.float32), np.asarray([1.0, 1.0], dtype=np.float32)]
        }
        err = io.StringIO()
        with redirect_stderr(err):
            rows = lss._layer_rows(layer_data={"single": accum}, sampled_frames=[0, 1])
        self.assertEqual(rows[0]["separability"], 0.0)
        self.assertIn("WARNING:", err.getvalue())

    def test_iter_target_detections_supports_all_classes_and_track_fallback(self) -> None:
        result = _FakeResult(
            boxes=[
                _FakeBox(cls_value=32, conf_value=0.9, xyxy=[0, 0, 5, 5], track_id=101),
                _FakeBox(cls_value=0, conf_value=0.8, xyxy=[1, 1, 6, 6], track_id=None),
                _FakeBox(cls_value=1, conf_value=0.1, xyxy=[2, 2, 7, 7], track_id=5),
            ]
        )
        all_rows = lss._iter_target_detections(result, class_id=-1, min_confidence=0.25)
        self.assertEqual(len(all_rows), 2)
        self.assertEqual(all_rows[0][2], 32)
        self.assertEqual(all_rows[0][3], 101)
        self.assertEqual(all_rows[1][2], 0)
        self.assertIsNone(all_rows[1][3])

        class_rows = lss._iter_target_detections(result, class_id=32, min_confidence=0.25)
        self.assertEqual(len(class_rows), 1)
        self.assertEqual(class_rows[0][2], 32)


if __name__ == "__main__":
    unittest.main()
