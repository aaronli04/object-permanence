"""CLI entrypoint for offline temporal linking."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from .config import TemporalLinkingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run temporal linking on enriched detections JSON.")
    parser.add_argument("--enriched-json", required=True, help="Path to enriched_detections.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for linked_detections.json, tracks.json, linking_manifest.json. "
        "Default: directory containing enriched JSON.",
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
    return parser


def _load_runner():
    from .pipeline import run_temporal_linking

    return run_temporal_linking


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
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir if args.output_dir else (os.path.dirname(args.enriched_json) or ".")

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
