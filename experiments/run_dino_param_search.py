#!/usr/bin/env python3
"""Run relink-only DINO threshold sweep (R0..R6) on enriched scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScenarioContext:
    name: str
    enriched_json: Path


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    relink_use_dino: bool
    relink_dino_threshold: float | None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run relink-only DINO threshold parameter search.")
    parser.add_argument(
        "--enrichment-root",
        type=Path,
        default=repo_root / "experiments" / "results" / "activation_enrichment",
        help="Directory containing per-scenario enriched_detections.json artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "experiments" / "results" / "param_search",
        help="Directory where per-run linking outputs and summary.csv are written.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=repo_root / ".venv" / "bin" / "python",
        help="Python executable used to run src/run_temporal_linking.py.",
    )
    parser.add_argument(
        "--linking-script",
        type=Path,
        default=repo_root / "src" / "run_temporal_linking.py",
        help="Path to temporal linking entrypoint.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.70,
        help="Frame-to-frame similarity threshold passed to temporal linking.",
    )
    parser.add_argument(
        "--relink-threshold",
        type=float,
        default=0.55,
        help="YOLO centroid relink threshold used for non-DINO relink path.",
    )
    parser.add_argument(
        "--relink-fallback-threshold",
        type=float,
        default=0.40,
        help="Spatial fallback threshold used by the relink third pass.",
    )
    parser.add_argument(
        "--activation-topk",
        type=int,
        default=64,
        help="Top-k projected dimensions used for temporal linking.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_scenarios(root: Path) -> list[ScenarioContext]:
    if not root.exists():
        raise FileNotFoundError(f"Enrichment root not found: {root}")

    contexts: list[ScenarioContext] = []
    for scenario_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        enriched_json = scenario_dir / "enriched_detections.json"
        if not enriched_json.exists():
            continue
        contexts.append(ScenarioContext(name=scenario_dir.name, enriched_json=enriched_json))

    if not contexts:
        raise RuntimeError(f"No scenario enriched_detections.json files found under {root}")
    return contexts


def _run_temporal_linking(
    *,
    python_exe: Path,
    linking_script: Path,
    enriched_json: Path,
    output_dir: Path,
    similarity_threshold: float,
    relink_threshold: float,
    relink_fallback_threshold: float,
    relink_use_dino: bool,
    relink_dino_threshold: float | None,
    activation_topk: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe),
        str(linking_script),
        "--enriched-json",
        str(enriched_json),
        "--output-dir",
        str(output_dir),
        "--activation-topk",
        str(int(activation_topk)),
        "--similarity-threshold",
        f"{float(similarity_threshold):.3f}",
        "--relink-threshold",
        f"{float(relink_threshold):.3f}",
        "--relink-fallback-threshold",
        f"{float(relink_fallback_threshold):.3f}",
    ]
    if relink_use_dino:
        if relink_dino_threshold is None:
            raise ValueError("relink_dino_threshold is required when relink_use_dino=True")
        cmd.extend(["--relink-dino-threshold", f"{float(relink_dino_threshold):.3f}"])
    else:
        cmd.append("--no-relink-dino")

    subprocess.run(cmd, check=True)


def _summarize_run(*, run_spec: RunSpec, output_root: Path, scenarios: list[ScenarioContext]) -> dict[str, Any]:
    total_tracks = 0
    valid_tracks = 0
    relink_edges = 0
    relink_dino_accepted = 0
    relink_yolo_accepted = 0
    dino_pairs_estimate = 0.0
    total_candidates = 0

    for scenario in scenarios:
        scenario_dir = output_root / run_spec.run_id / scenario.name
        linking_manifest = _load_json(scenario_dir / "linking_manifest.json")
        relink_manifest = _load_json(scenario_dir / "relink_manifest.json")

        linking_stats = linking_manifest.get("stats", {})
        relink_stats = relink_manifest.get("stats", {})

        scenario_total = int(linking_stats.get("num_tracks_total", 0))
        scenario_valid = int(linking_stats.get("num_tracks_valid", 0))
        scenario_edges = int(relink_stats.get("num_accepted_edges", len(relink_manifest.get("accepted_edges", []))))
        scenario_dino_accepted = int(relink_stats.get("relink_dino_accepted", 0))
        scenario_yolo_accepted = int(relink_stats.get("relink_yolo_accepted", 0))
        scenario_candidates = int(relink_stats.get("num_candidates", 0))
        scenario_coverage = float(relink_stats.get("relink_dino_coverage", 0.0))

        total_tracks += scenario_total
        valid_tracks += scenario_valid
        relink_edges += scenario_edges
        relink_dino_accepted += scenario_dino_accepted
        relink_yolo_accepted += scenario_yolo_accepted
        total_candidates += scenario_candidates
        dino_pairs_estimate += float(scenario_coverage) * float(scenario_candidates)

    relink_dino_coverage = (dino_pairs_estimate / float(total_candidates)) if total_candidates > 0 else 0.0
    fragmentation_ratio = float(total_tracks / valid_tracks) if valid_tracks > 0 else float("inf")

    return {
        "run_id": run_spec.run_id,
        "relink_dino_threshold": (
            "" if run_spec.relink_dino_threshold is None else float(run_spec.relink_dino_threshold)
        ),
        "relink_use_dino": bool(run_spec.relink_use_dino),
        "total_tracks": int(total_tracks),
        "valid_tracks": int(valid_tracks),
        "relink_edges": int(relink_edges),
        "relink_dino_accepted": int(relink_dino_accepted),
        "relink_yolo_accepted": int(relink_yolo_accepted),
        "relink_dino_coverage": float(relink_dino_coverage),
        "fragmentation_ratio": float(fragmentation_ratio),
        "delta_valid_vs_baseline": 0,
    }


def _write_summary_csv(rows: list[dict[str, Any]], summary_csv: Path) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run_id",
        "relink_dino_threshold",
        "relink_use_dino",
        "total_tracks",
        "valid_tracks",
        "relink_edges",
        "relink_dino_accepted",
        "relink_yolo_accepted",
        "relink_dino_coverage",
        "fragmentation_ratio",
        "delta_valid_vs_baseline",
    ]

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["relink_dino_coverage"] = f"{float(row['relink_dino_coverage']):.6f}"
            out_row["fragmentation_ratio"] = f"{float(row['fragmentation_ratio']):.6f}"
            writer.writerow(out_row)


def _ranking_key(row: dict[str, Any]) -> tuple[int, int, float]:
    return (
        -int(row["relink_edges"]),
        -int(row["valid_tracks"]),
        float(row["fragmentation_ratio"]),
    )


def run_experiment_grid(*, args: argparse.Namespace, scenarios: list[ScenarioContext]) -> list[dict[str, Any]]:
    run_specs = [
        RunSpec(run_id="R0", relink_use_dino=False, relink_dino_threshold=None),
        RunSpec(run_id="R1", relink_use_dino=True, relink_dino_threshold=0.40),
        RunSpec(run_id="R2", relink_use_dino=True, relink_dino_threshold=0.45),
        RunSpec(run_id="R3", relink_use_dino=True, relink_dino_threshold=0.50),
        RunSpec(run_id="R4", relink_use_dino=True, relink_dino_threshold=0.55),
        RunSpec(run_id="R5", relink_use_dino=True, relink_dino_threshold=0.60),
        RunSpec(run_id="R6", relink_use_dino=True, relink_dino_threshold=0.65),
    ]

    rows: list[dict[str, Any]] = []
    for spec in run_specs:
        print(
            f"[run {spec.run_id}] relink_use_dino={spec.relink_use_dino} "
            f"relink_dino_threshold={spec.relink_dino_threshold if spec.relink_dino_threshold is not None else '-'}"
        )
        for scenario in scenarios:
            scenario_out = args.output_root / spec.run_id / scenario.name
            _run_temporal_linking(
                python_exe=args.python,
                linking_script=args.linking_script,
                enriched_json=scenario.enriched_json,
                output_dir=scenario_out,
                similarity_threshold=float(args.similarity_threshold),
                relink_threshold=float(args.relink_threshold),
                relink_fallback_threshold=float(args.relink_fallback_threshold),
                relink_use_dino=bool(spec.relink_use_dino),
                relink_dino_threshold=spec.relink_dino_threshold,
                activation_topk=int(args.activation_topk),
            )

        row = _summarize_run(run_spec=spec, output_root=args.output_root, scenarios=scenarios)
        _write_json(args.output_root / spec.run_id / "run_metrics.json", row)
        rows.append(row)
        print(
            f"[run {spec.run_id}] total_tracks={row['total_tracks']} valid_tracks={row['valid_tracks']} "
            f"relink_edges={row['relink_edges']} dino_cov={row['relink_dino_coverage']:.3f}"
        )

    baseline_row = next((row for row in rows if row["run_id"] == "R0"), None)
    if baseline_row is None:
        raise RuntimeError("Missing baseline run R0")
    baseline_valid = int(baseline_row["valid_tracks"])
    baseline_total = int(baseline_row["total_tracks"])

    for row in rows:
        row["delta_valid_vs_baseline"] = int(row["valid_tracks"]) - baseline_valid

    eligible = [row for row in rows if int(row["total_tracks"]) <= baseline_total]
    pool = eligible if eligible else rows
    winner = sorted(pool, key=_ranking_key)[0]

    print("")
    print(
        "Winner (maximize relink_edges and valid_tracks with total_tracks <= baseline): "
        f"{winner['run_id']}"
    )
    print(
        f"  relink_edges={winner['relink_edges']} valid_tracks={winner['valid_tracks']} "
        f"total_tracks={winner['total_tracks']} fragmentation_ratio={winner['fragmentation_ratio']:.4f}"
    )
    if not eligible:
        print("  NOTE: no run met total_tracks <= baseline; selected best unconstrained run.")

    return rows


def main() -> int:
    args = parse_args()
    scenarios = load_scenarios(args.enrichment_root)
    rows = run_experiment_grid(args=args, scenarios=scenarios)

    summary_csv = args.output_root / "summary.csv"
    _write_summary_csv(rows, summary_csv)
    print("")
    print(f"Wrote summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
