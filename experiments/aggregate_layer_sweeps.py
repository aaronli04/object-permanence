#!/usr/bin/env python3
"""Aggregate per-video layer sweep CSVs into a single separability leaderboard."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LayerAggregate:
    module_types: set[str] = field(default_factory=set)
    feature_dims: set[int] = field(default_factory=set)
    videos_present: set[str] = field(default_factory=set)
    separability: list[float] = field(default_factory=list)
    between_var: list[float] = field(default_factory=list)
    within_var: list[float] = field(default_factory=list)
    mean_consecutive_cosine: list[float] = field(default_factory=list)
    norm_std: list[float] = field(default_factory=list)
    track_id_coverage: list[float] = field(default_factory=list)


def _mean_finite(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float("nan")
    return float(np.mean(arr))


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Invalid numeric value for '{key}': {value!r}") from exc


def _to_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Invalid int value for '{key}': {value!r}") from exc


def _read_csvs(paths: list[Path]) -> dict[str, LayerAggregate]:
    required = {
        "layer_name",
        "module_type",
        "feature_dim",
        "mean_consecutive_cosine",
        "norm_std",
        "within_var",
        "between_var",
        "separability",
        "track_id_coverage",
    }

    aggregates: dict[str, LayerAggregate] = {}
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: CSV has no header.")
            missing = required.difference(set(reader.fieldnames))
            if missing:
                raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
            for row in reader:
                layer_name = (row.get("layer_name") or "").strip()
                if not layer_name:
                    raise ValueError(f"{path}: row missing layer_name.")
                agg = aggregates.setdefault(layer_name, LayerAggregate())
                agg.module_types.add((row.get("module_type") or "unknown").strip() or "unknown")
                agg.feature_dims.add(_to_int(row, "feature_dim"))
                agg.videos_present.add(path.stem)
                agg.separability.append(_to_float(row, "separability"))
                agg.between_var.append(_to_float(row, "between_var"))
                agg.within_var.append(_to_float(row, "within_var"))
                agg.mean_consecutive_cosine.append(_to_float(row, "mean_consecutive_cosine"))
                agg.norm_std.append(_to_float(row, "norm_std"))
                agg.track_id_coverage.append(_to_float(row, "track_id_coverage"))
    return aggregates


def _build_rows(aggregates: dict[str, LayerAggregate]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_name, agg in aggregates.items():
        rows.append(
            {
                "layer_name": layer_name,
                "module_type": sorted(agg.module_types)[0] if agg.module_types else "unknown",
                "feature_dim": sorted(agg.feature_dims)[0] if agg.feature_dims else 0,
                "videos_present": len(agg.videos_present),
                "mean_separability": _mean_finite(agg.separability),
                "mean_between_var": _mean_finite(agg.between_var),
                "mean_within_var": _mean_finite(agg.within_var),
                "mean_mean_consecutive_cosine": _mean_finite(agg.mean_consecutive_cosine),
                "mean_norm_std": _mean_finite(agg.norm_std),
                "mean_track_id_coverage": _mean_finite(agg.track_id_coverage),
            }
        )

    rows.sort(
        key=lambda row: (
            1 if not math.isfinite(float(row["mean_separability"])) else 0,
            -float(row["mean_separability"]) if math.isfinite(float(row["mean_separability"])) else 0.0,
            1 if not math.isfinite(float(row["mean_mean_consecutive_cosine"])) else 0,
            -float(row["mean_mean_consecutive_cosine"])
            if math.isfinite(float(row["mean_mean_consecutive_cosine"]))
            else 0.0,
            str(row["layer_name"]),
        )
    )
    return rows


def _rank_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        ranked_row = dict(row)
        ranked_row["rank"] = idx
        ranked.append(ranked_row)
    return ranked


def _winner_candidates(
    rows: list[dict[str, object]],
    *,
    min_feature_dim: int,
) -> list[dict[str, object]]:
    if min_feature_dim <= 0:
        raise ValueError("min_feature_dim must be > 0")

    eligible = [row for row in rows if int(row["feature_dim"]) >= int(min_feature_dim)]
    eligible_by_name = {str(row["layer_name"]): row for row in eligible}

    deduped: list[dict[str, object]] = []
    for row in eligible:
        name = str(row["layer_name"])
        if name.endswith(".conv"):
            parent = name[: -len(".conv")]
            parent_row = eligible_by_name.get(parent)
            if parent_row is not None and str(parent_row.get("module_type", "")) == "Conv":
                continue
        deduped.append(row)

    return deduped


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "layer_name",
                "module_type",
                "feature_dim",
                "videos_present",
                "mean_separability",
                "mean_between_var",
                "mean_within_var",
                "mean_mean_consecutive_cosine",
                "mean_norm_std",
                "mean_track_id_coverage",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-video layer sweep CSVs by mean separability.")
    parser.add_argument(
        "--input-glob",
        default="experiments/results/layer_selection/per_video/layer_stability_sweep_*.csv",
        help="Glob pattern for per-video sweep CSV files.",
    )
    parser.add_argument(
        "--output-csv",
        default="experiments/results/layer_selection/aggregate/aggregate_separability.csv",
        help="Output aggregate CSV path.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of top rows to print.")
    parser.add_argument(
        "--winner-min-feature-dim",
        type=int,
        default=32,
        help="Minimum feature dim eligible for winner selection (default: 32).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not paths:
        raise FileNotFoundError(f"No CSV files matched input glob: {args.input_glob}")

    aggregates = _read_csvs(paths)
    rows = _build_rows(aggregates)
    if not rows:
        raise RuntimeError("No aggregate rows were produced.")

    candidates = _winner_candidates(rows, min_feature_dim=int(args.winner_min_feature_dim))
    if not candidates:
        raise RuntimeError(
            "No winner candidates were produced after constraints "
            f"(min_feature_dim={args.winner_min_feature_dim}, dedupe_parent_conv=True)."
        )
    ranked = _rank_rows(candidates)

    output_csv = Path(args.output_csv)
    _write_csv(output_csv, ranked)

    winner = ranked[0]
    winner_track_coverage = float(winner["mean_track_id_coverage"])
    if math.isfinite(winner_track_coverage) and winner_track_coverage < 0.5:
        print(
            "WARNING: selected winner has mean_track_id_coverage < 0.5; "
            "separability may be primarily class-level.",
            file=sys.stderr,
        )
    print(f"input_files={len(paths)}")
    print(f"layers_aggregated={len(rows)}")
    print(f"layers_after_constraints={len(ranked)}")
    print(f"winner_min_feature_dim={int(args.winner_min_feature_dim)}")
    print("dedupe_parent_conv=True")
    print(f"csv_saved={output_csv}")
    print(
        "selected_winner="
        f"{winner['layer_name']}"
        f" mean_separability={float(winner['mean_separability']):.6f}"
        f" mean_cos={float(winner['mean_mean_consecutive_cosine']):.6f}"
        f" mean_track_cov={winner_track_coverage:.6f}"
    )
    print("")
    print(f"Top {args.top_n} aggregate layers by mean_separability:")
    for row in ranked[: args.top_n]:
        print(
            f"{int(row['rank']):>2}. {str(row['layer_name']):<28} "
            f"type={str(row['module_type']):<14} "
            f"dim={int(row['feature_dim']):<4} "
            f"videos={int(row['videos_present']):<2} "
            f"mean_separability={float(row['mean_separability']):.6f} "
            f"mean_cos={float(row['mean_mean_consecutive_cosine']):.6f} "
            f"mean_track_cov={float(row['mean_track_id_coverage']):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
