#!/usr/bin/env python3
"""Sweep top-k PCA dimensions and evaluate cosine stability across frame groups.

Expected input JSON format:
[
  {"frame": 0, "vec": [0.1, -0.2, ...]},
  {"frame": 5, "vec": [0.05, 0.03, ...]},
  ...
]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from common.numeric import l2_normalize_rows


def _parse_frame_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Frame list cannot be empty.")
    return [int(item) for item in values]


def _mean_upper_triangle(sim_matrix: np.ndarray, indices: Sequence[int]) -> float:
    if len(indices) < 2:
        return float("nan")
    sub = sim_matrix[np.ix_(indices, indices)]
    tri_i, tri_j = np.triu_indices(sub.shape[0], k=1)
    if tri_i.size == 0:
        return float("nan")
    return float(np.mean(sub[tri_i, tri_j]))


def _mean_cross(sim_matrix: np.ndarray, left_idx: Sequence[int], right_idx: Sequence[int]) -> float:
    if not left_idx or not right_idx:
        return float("nan")
    sub = sim_matrix[np.ix_(left_idx, right_idx)]
    return float(np.mean(sub))


def _collect_indices(frames: Sequence[int], wanted: set[int]) -> list[int]:
    return [idx for idx, frame in enumerate(frames) if frame in wanted]


def load_vectors(path: Path) -> tuple[list[int], np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of {'frame', 'vec'} objects.")

    frames: list[int] = []
    vectors: list[np.ndarray] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not an object.")
        if "frame" not in item or "vec" not in item:
            raise ValueError(f"Item {idx} must contain 'frame' and 'vec'.")
        frame = int(item["frame"])
        vec_raw = item["vec"]
        if not isinstance(vec_raw, list):
            raise ValueError(f"Item {idx} field 'vec' must be a list.")
        vec = np.asarray(vec_raw, dtype=np.float64)
        if vec.ndim != 1:
            raise ValueError(f"Item {idx} vector must be 1D.")
        frames.append(frame)
        vectors.append(vec)

    if not vectors:
        raise ValueError("No vectors found in input JSON.")

    dim = vectors[0].shape[0]
    for idx, vec in enumerate(vectors):
        if vec.shape[0] != dim:
            raise ValueError(f"Vector dim mismatch at item {idx}: expected {dim}, got {vec.shape[0]}.")

    matrix = np.stack(vectors, axis=0)
    return frames, matrix


def sweep_topk(
    *,
    frames: Sequence[int],
    vectors: np.ndarray,
    k_values: Iterable[int],
    early_frames: Sequence[int],
    late_frames: Sequence[int],
) -> tuple[list[int], list[float], list[float], list[float]]:
    early_idx = _collect_indices(frames, set(early_frames))
    late_idx = _collect_indices(frames, set(late_frames))

    if not early_idx:
        raise ValueError("None of the requested early frames were found in input.")
    if not late_idx:
        raise ValueError("None of the requested late frames were found in input.")

    ks: list[int] = []
    within_early: list[float] = []
    within_late: list[float] = []
    cross_scores: list[float] = []

    dim = vectors.shape[1]
    for k in k_values:
        if k <= 0 or k > dim:
            continue
        sliced = vectors[:, :k]
        sliced = l2_normalize_rows(sliced)
        sim = sliced @ sliced.T

        ks.append(k)
        within_early.append(_mean_upper_triangle(sim, early_idx))
        within_late.append(_mean_upper_triangle(sim, late_idx))
        cross_scores.append(_mean_cross(sim, early_idx, late_idx))

    if not ks:
        raise ValueError("No valid k values were produced. Check min/max/step arguments.")
    return ks, within_early, within_late, cross_scores


def plot_curves(
    *,
    ks: Sequence[int],
    within_early: Sequence[float],
    within_late: Sequence[float],
    cross_scores: Sequence[float],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - dependency/environment guard
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: python3 -m pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, within_early, marker="o", label="within_early")
    ax.plot(ks, within_late, marker="o", label="within_late")
    ax.plot(ks, cross_scores, marker="o", label="cross")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Top-k PCA Dimensions (first k components)")
    ax.set_ylabel("Average Cosine Similarity")
    ax.set_title("Top-k Dimensional Sweep: Similarity Stability")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep top-k PCA dims and compare cosine similarity aggregates.")
    parser.add_argument("--input-json", required=True, help="Path to JSON list of {'frame','vec'} objects.")
    parser.add_argument(
        "--output-plot",
        default="experiments/results/topk_similarity_sweep.png",
        help="Path to output plot PNG.",
    )
    parser.add_argument("--skip-plot", action="store_true", help="Skip plot generation (still computes metrics).")
    parser.add_argument("--output-csv", help="Optional path to output CSV with k,within_early,within_late,cross.")
    parser.add_argument("--min-k", type=int, default=2, help="Minimum k (inclusive).")
    parser.add_argument("--max-k", type=int, default=40, help="Maximum k (inclusive).")
    parser.add_argument("--step-k", type=int, default=2, help="Step size for k sweep.")
    parser.add_argument("--early-frames", default="0,5,10,15", help="Comma-separated early frame numbers.")
    parser.add_argument("--late-frames", default="70,75,80,85,90,95,100", help="Comma-separated late frame numbers.")
    args = parser.parse_args()

    if args.step_k <= 0:
        raise ValueError("--step-k must be > 0.")

    frames, vectors = load_vectors(Path(args.input_json))
    k_values = range(args.min_k, args.max_k + 1, args.step_k)
    early_frames = _parse_frame_list(args.early_frames)
    late_frames = _parse_frame_list(args.late_frames)

    ks, within_early, within_late, cross_scores = sweep_topk(
        frames=frames,
        vectors=vectors,
        k_values=k_values,
        early_frames=early_frames,
        late_frames=late_frames,
    )

    if not args.skip_plot:
        plot_curves(
            ks=ks,
            within_early=within_early,
            within_late=within_late,
            cross_scores=cross_scores,
            output_path=Path(args.output_plot),
        )

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["k", "within_early", "within_late", "cross"])
            for k, w_e, w_l, c in zip(ks, within_early, within_late, cross_scores):
                writer.writerow([int(k), float(w_e), float(w_l), float(c)])

    best_idx = int(np.nanargmax(np.asarray(cross_scores, dtype=np.float64)))
    best_k = ks[best_idx]
    best_cross = cross_scores[best_idx]

    print("k,within_early,within_late,cross")
    for k, w_e, w_l, c in zip(ks, within_early, within_late, cross_scores):
        print(f"{k},{w_e:.6f},{w_l:.6f},{c:.6f}")
    print(f"best_k_for_cross={best_k}")
    print(f"best_cross_similarity={best_cross:.6f}")
    if not args.skip_plot:
        print(f"plot_saved={Path(args.output_plot)}")
    if args.output_csv:
        print(f"csv_saved={Path(args.output_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
