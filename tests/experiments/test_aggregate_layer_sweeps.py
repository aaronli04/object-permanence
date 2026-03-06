from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

import aggregate_layer_sweeps as ag


def _write_sweep_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_name",
                "module_type",
                "feature_dim",
                "mean_consecutive_cosine",
                "norm_std",
                "within_var",
                "between_var",
                "separability",
                "track_id_coverage",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class AggregateLayerSweepsTests(unittest.TestCase):
    def test_read_csvs_requires_track_id_coverage_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layer_stability_sweep_old.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "layer_name",
                        "module_type",
                        "feature_dim",
                        "mean_consecutive_cosine",
                        "norm_std",
                        "within_var",
                        "between_var",
                        "separability",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "layer_name": "A",
                        "module_type": "Conv",
                        "feature_dim": 32,
                        "mean_consecutive_cosine": 0.9,
                        "norm_std": 0.1,
                        "within_var": 0.1,
                        "between_var": 1.0,
                        "separability": 10.0,
                    }
                )
            with self.assertRaises(ValueError):
                ag._read_csvs([path])

    def test_build_rows_and_candidates_rank_by_mean_separability_then_cosine(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sweep_csv(
                root / "layer_stability_sweep_v1.csv",
                [
                    {
                        "layer_name": "A",
                        "module_type": "Conv",
                        "feature_dim": 32,
                        "mean_consecutive_cosine": 0.70,
                        "norm_std": 0.20,
                        "within_var": 1.0,
                        "between_var": 3.0,
                        "separability": 3.0,
                        "track_id_coverage": 0.9,
                    },
                    {
                        "layer_name": "B",
                        "module_type": "Conv",
                        "feature_dim": 32,
                        "mean_consecutive_cosine": 0.95,
                        "norm_std": 0.30,
                        "within_var": 1.0,
                        "between_var": 2.0,
                        "separability": 2.0,
                        "track_id_coverage": 0.4,
                    },
                ],
            )
            _write_sweep_csv(
                root / "layer_stability_sweep_v2.csv",
                [
                    {
                        "layer_name": "A",
                        "module_type": "Conv",
                        "feature_dim": 32,
                        "mean_consecutive_cosine": 0.75,
                        "norm_std": 0.20,
                        "within_var": 1.0,
                        "between_var": 5.0,
                        "separability": 5.0,
                        "track_id_coverage": 0.8,
                    },
                    {
                        "layer_name": "B",
                        "module_type": "Conv",
                        "feature_dim": 32,
                        "mean_consecutive_cosine": 0.99,
                        "norm_std": 0.30,
                        "within_var": 1.0,
                        "between_var": 1.5,
                        "separability": 1.5,
                        "track_id_coverage": 0.3,
                    },
                ],
            )

            aggregates = ag._read_csvs(sorted(root.glob("layer_stability_sweep_*.csv")))
            rows = ag._build_rows(aggregates)
            candidates = ag._winner_candidates(rows, min_feature_dim=32)
            ranked = ag._rank_rows(candidates)
            self.assertGreater(len(ranked), 1)
            self.assertEqual(ranked[0]["layer_name"], "A")
            self.assertEqual(ranked[0]["rank"], 1)
            self.assertEqual(ranked[0]["videos_present"], 2)
            self.assertGreater(float(ranked[0]["mean_separability"]), float(ranked[1]["mean_separability"]))
            self.assertAlmostEqual(float(ranked[0]["mean_track_id_coverage"]), 0.85, places=6)

    def test_candidates_drop_low_dim_and_conv_duplicates(self) -> None:
        rows = [
            {
                "layer_name": "0",
                "module_type": "Conv",
                "feature_dim": 16,
                "videos_present": 8,
                "mean_separability": 99.0,
                "mean_between_var": 1.0,
                "mean_within_var": 0.01,
                "mean_mean_consecutive_cosine": 0.9,
                "mean_norm_std": 0.1,
                "mean_track_id_coverage": 0.1,
            },
            {
                "layer_name": "2.cv1",
                "module_type": "Conv",
                "feature_dim": 32,
                "videos_present": 8,
                "mean_separability": 10.0,
                "mean_between_var": 1.0,
                "mean_within_var": 0.1,
                "mean_mean_consecutive_cosine": 0.9,
                "mean_norm_std": 0.1,
                "mean_track_id_coverage": 0.9,
            },
            {
                "layer_name": "2.cv1.conv",
                "module_type": "Conv2d",
                "feature_dim": 32,
                "videos_present": 8,
                "mean_separability": 10.0,
                "mean_between_var": 1.0,
                "mean_within_var": 0.1,
                "mean_mean_consecutive_cosine": 0.9,
                "mean_norm_std": 0.1,
                "mean_track_id_coverage": 0.9,
            },
            {
                "layer_name": "22.cv3.0",
                "module_type": "Sequential",
                "feature_dim": 80,
                "videos_present": 8,
                "mean_separability": 9.0,
                "mean_between_var": 1.0,
                "mean_within_var": 0.1,
                "mean_mean_consecutive_cosine": 0.95,
                "mean_norm_std": 0.2,
                "mean_track_id_coverage": 0.2,
            },
            {
                "layer_name": "seq_parent",
                "module_type": "Sequential",
                "feature_dim": 64,
                "videos_present": 8,
                "mean_separability": 8.0,
                "mean_between_var": 1.0,
                "mean_within_var": 0.1,
                "mean_mean_consecutive_cosine": 0.90,
                "mean_norm_std": 0.2,
                "mean_track_id_coverage": 0.7,
            },
            {
                "layer_name": "seq_parent.conv",
                "module_type": "Conv2d",
                "feature_dim": 64,
                "videos_present": 8,
                "mean_separability": 8.0,
                "mean_between_var": 1.0,
                "mean_within_var": 0.1,
                "mean_mean_consecutive_cosine": 0.90,
                "mean_norm_std": 0.2,
                "mean_track_id_coverage": 0.7,
            },
        ]
        candidates = ag._winner_candidates(rows, min_feature_dim=32)
        names = [str(r["layer_name"]) for r in candidates]
        self.assertNotIn("0", names)
        self.assertIn("2.cv1", names)
        self.assertNotIn("2.cv1.conv", names)
        self.assertIn("22.cv3.0", names)
        self.assertIn("seq_parent.conv", names)

    def test_main_writes_aggregate_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sweep_csv(
                root / "layer_stability_sweep_left.csv",
                [
                    {
                        "layer_name": "winner",
                        "module_type": "Conv",
                        "feature_dim": 32,
                        "mean_consecutive_cosine": 0.90,
                        "norm_std": 0.10,
                        "within_var": 0.20,
                        "between_var": 1.00,
                        "separability": 5.0,
                        "track_id_coverage": 0.45,
                    }
                ],
            )
            _write_sweep_csv(
                root / "layer_stability_sweep_right.csv",
                [
                    {
                        "layer_name": "winner",
                        "module_type": "Conv",
                        "feature_dim": 32,
                        "mean_consecutive_cosine": 0.80,
                        "norm_std": 0.10,
                        "within_var": 0.30,
                        "between_var": 1.20,
                        "separability": 4.0,
                        "track_id_coverage": 0.40,
                    }
                ],
            )

            output_csv = root / "aggregate_separability.csv"
            argv = [
                "aggregate_layer_sweeps.py",
                "--input-glob",
                str(root / "layer_stability_sweep_*.csv"),
                "--output-csv",
                str(output_csv),
                "--top-n",
                "5",
                "--winner-min-feature-dim",
                "32",
            ]
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", argv), redirect_stderr(stderr):
                rc = ag.main()
            self.assertEqual(rc, 0)
            self.assertTrue(output_csv.exists())
            self.assertIn("mean_track_id_coverage < 0.5", stderr.getvalue())

            with output_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["rank"], "1")
            self.assertEqual(rows[0]["layer_name"], "winner")
            self.assertEqual(rows[0]["videos_present"], "2")
            self.assertIn("mean_track_id_coverage", rows[0])


if __name__ == "__main__":
    unittest.main()
