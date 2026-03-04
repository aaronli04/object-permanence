"""CLI entrypoint for offline temporal linking."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

try:
    from common.paths import scenario_name_from_enriched_json
except ImportError:  # pragma: no cover - import-path compatibility
    from src.common.paths import scenario_name_from_enriched_json  # type: ignore

from .config import TemporalLinkingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run temporal linking on enriched detections JSON.")
    parser.add_argument("--enriched-json", required=True, help="Path to enriched_detections.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for linked_detections.json, tracks.json, linking_manifest.json, relink_manifest.json. "
        "Default: experiments/results/linking/<scenario_name_from_enriched_json_parent>.",
    )

    parser.add_argument(
        "--similarity-threshold",
        required=True,
        type=float,
        help="Single similarity threshold used for both normal linking and lost-track recovery.",
    )

    parser.add_argument("--max-lost-frames", type=int, default=6, help="Maximum consecutive missed frames before closing a track.")
    parser.add_argument("--min-hits-to-activate", type=int, default=2, help="Detections required to promote tentative tracks.")
    parser.add_argument("--min-track-length", type=int, default=2, help="Minimum hits for track to be marked valid.")

    parser.add_argument("--history-size", type=int, default=5, help="Per-track vector/similarity history length.")
    parser.add_argument("--ema-alpha", type=float, default=0.35, help="EMA alpha for descriptor smoothing.")

    parser.add_argument("--w-last", type=float, default=0.55, help="Weight for last descriptor in reference vector.")
    parser.add_argument("--w-ema", type=float, default=0.30, help="Weight for EMA descriptor in reference vector.")
    parser.add_argument("--w-hist", type=float, default=0.15, help="Weight for history-mean descriptor in reference vector.")
    parser.add_argument("--w-spatial", type=float, default=0.05, help="Weight for spatial secondary score.")
    parser.add_argument("--w-consistency", type=float, default=0.10, help="Weight for rolling similarity consistency.")
    parser.add_argument("--w-age", type=float, default=0.05, help="Weight for lost-age decay bonus.")

    parser.add_argument(
        "--assignment-method",
        choices=["hungarian", "greedy"],
        default="hungarian",
        help="Pairing solver used each frame.",
    )
    parser.add_argument(
        "--match-within-class",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict matching to same class_id pairs.",
    )
    parser.add_argument(
        "--filter-short-tracks-in-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only count tracks with hits >= min_track_length as valid in summary stats.",
    )
    parser.add_argument(
        "--activation-topk",
        type=int,
        default=64,
        help="Keep first K activation dims and L2-renormalize before linking (default: 64).",
    )

    parser.add_argument(
        "--relink-threshold",
        type=float,
        default=0.55,
        help="Centroid similarity gate used in post-hoc relinking.",
    )
    parser.add_argument(
        "--relink-max-gap-frames",
        type=int,
        default=120,
        help="Maximum frame gap for relink candidates (default: 120; use -1 for unlimited).",
    )
    parser.add_argument(
        "--relink-min-track-hits",
        type=int,
        default=2,
        help="Minimum hits required for tracks to be relink candidates.",
    )
    parser.add_argument(
        "--relink-max-pixels-per-frame",
        type=float,
        default=15.0,
        help="Maximum plausible drift rate (pixels/frame) used by spatial relink fallback scoring.",
    )
    parser.add_argument(
        "--relink-fallback-threshold",
        type=float,
        default=0.40,
        help="Fallback score threshold used after centroid-based relinking.",
    )
    return parser


def _load_runner():
    from .pipeline import run_temporal_linking

    return run_temporal_linking


def _default_linking_output_dir(enriched_json_path: str) -> str:
    scenario = scenario_name_from_enriched_json(enriched_json_path)
    return os.path.join("experiments", "results", "linking", scenario)


def _build_config(args: argparse.Namespace) -> TemporalLinkingConfig:
    return TemporalLinkingConfig(
        similarity_threshold=args.similarity_threshold,
        max_lost_frames=args.max_lost_frames,
        min_hits_to_activate=args.min_hits_to_activate,
        min_track_length=args.min_track_length,
        history_size=args.history_size,
        ema_alpha=args.ema_alpha,
        w_last=args.w_last,
        w_ema=args.w_ema,
        w_hist=args.w_hist,
        w_spatial=args.w_spatial,
        w_consistency=args.w_consistency,
        w_age=args.w_age,
        assignment_method=args.assignment_method,
        match_within_class=bool(args.match_within_class),
        filter_short_tracks_in_summary=bool(args.filter_short_tracks_in_summary),
        activation_topk=args.activation_topk,
        relink_threshold=args.relink_threshold,
        relink_max_gap_frames=args.relink_max_gap_frames,
        relink_min_track_hits=args.relink_min_track_hits,
        relink_max_pixels_per_frame=args.relink_max_pixels_per_frame,
        relink_fallback_threshold=args.relink_fallback_threshold,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir if args.output_dir else _default_linking_output_dir(args.enriched_json)

    run_temporal_linking = _load_runner()
    config = _build_config(args)

    try:
        outputs = run_temporal_linking(
            enriched_json_path=args.enriched_json,
            output_dir=output_dir,
            config=config,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved linked detections to {outputs['linked_detections']}")
    print(f"Saved tracks to {outputs['tracks']}")
    print(f"Saved linking manifest to {outputs['linking_manifest']}")
    print(f"Saved relink manifest to {outputs['relink_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
