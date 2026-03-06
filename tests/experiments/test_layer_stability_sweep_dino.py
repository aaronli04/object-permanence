from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import layer_stability_sweep as lss


class _FakeBox:
    def __init__(self) -> None:
        self.cls = torch.tensor([32.0], dtype=torch.float32)
        self.conf = torch.tensor([0.95], dtype=torch.float32)
        self.xyxy = torch.tensor([[10.0, 10.0, 50.0, 50.0]], dtype=torch.float32)
        self.id = torch.tensor([101.0], dtype=torch.float32)


class _FakeResult:
    def __init__(self) -> None:
        self.boxes = [_FakeBox()]


class _FakeYolo:
    def __call__(self, _frame, verbose: bool = False):  # noqa: ARG002
        return [_FakeResult()]


class LayerStabilitySweepDinoTests(unittest.TestCase):
    def test_dino_flag_adds_dino_cls_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "sweep.csv")
            argv = [
                "layer_stability_sweep.py",
                "--video",
                "dummy.mp4",
                "--output-csv",
                output_csv,
                "--max-sampled-frames",
                "2",
                "--dino",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("layer_stability_sweep.load_yolo", return_value=_FakeYolo()),
                mock.patch("layer_stability_sweep.get_module_map", return_value={}),
                mock.patch(
                    "layer_stability_sweep.FrameSampler",
                    return_value=[
                        (0, np.zeros((80, 80, 3), dtype=np.uint8)),
                        (1, np.zeros((80, 80, 3), dtype=np.uint8)),
                    ],
                ),
                mock.patch("layer_stability_sweep.load_dino_embedder", return_value=object()),
                mock.patch(
                    "layer_stability_sweep.extract_dino_embedding",
                    return_value=SimpleNamespace(
                        vector=np.ones((384,), dtype=np.float32) / np.sqrt(384.0),
                        valid_crop=True,
                        tiny_crop=False,
                    ),
                ),
            ):
                rc = lss.main()

            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(output_csv))
            with open(output_csv, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(any(row["layer_name"] == "dino_cls" for row in rows))

    def test_dino_preflight_failure_writes_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = os.path.join(tmpdir, "sweep.csv")
            argv = [
                "layer_stability_sweep.py",
                "--video",
                "dummy.mp4",
                "--output-csv",
                output_csv,
                "--dino",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("layer_stability_sweep.load_yolo", return_value=_FakeYolo()),
                mock.patch(
                    "layer_stability_sweep.load_dino_embedder",
                    side_effect=lss.DinoUnavailableError("offline"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    lss.main()
            self.assertFalse(os.path.exists(output_csv))


if __name__ == "__main__":
    unittest.main()
