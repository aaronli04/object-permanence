"""CLI entrypoint for offline temporal linking."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from common.paths import scenario_name_from_enriched_json

from .config import TemporalLinkingConfig

_CFG_DEFAULTS = TemporalLinkingConfig.defaults()


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

    parser.add_argument(
        "--max-lost-frames",
        type=int,
        default=int(_CFG_DEFAULTS["max_lost_frames"]),
        help="Maximum consecutive missed frames before closing a track.",
    )
    parser.add_argument(
        "--min-hits-to-activate",
        type=int,
        default=int(_CFG_DEFAULTS["min_hits_to_activate"]),
        help="Detections required to promote tentative tracks.",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=int(_CFG_DEFAULTS["min_track_length"]),
        help="Minimum hits for track to be marked valid.",
    )

    parser.add_argument(
        "--history-size",
        type=int,
        default=int(_CFG_DEFAULTS["history_size"]),
        help="Per-track vector/similarity history length.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=float(_CFG_DEFAULTS["ema_alpha"]),
        help="EMA alpha for descriptor smoothing.",
    )

    parser.add_argument(
        "--w-last",
        type=float,
        default=float(_CFG_DEFAULTS["w_last"]),
        help="Weight for last descriptor in reference vector.",
    )
    parser.add_argument(
        "--w-ema",
        type=float,
        default=float(_CFG_DEFAULTS["w_ema"]),
        help="Weight for EMA descriptor in reference vector.",
    )
    parser.add_argument(
        "--w-hist",
        type=float,
        default=float(_CFG_DEFAULTS["w_hist"]),
        help="Weight for history-mean descriptor in reference vector.",
    )
    parser.add_argument(
        "--w-spatial",
        type=float,
        default=float(_CFG_DEFAULTS["w_spatial"]),
        help="Weight for spatial secondary score.",
    )
    parser.add_argument(
        "--w-consistency",
        type=float,
        default=float(_CFG_DEFAULTS["w_consistency"]),
        help="Weight for rolling similarity consistency.",
    )
    parser.add_argument(
        "--w-age",
        type=float,
        default=float(_CFG_DEFAULTS["w_age"]),
        help="Weight for lost-age decay bonus.",
    )

    parser.add_argument(
        "--assignment-method",
        choices=["hungarian", "greedy"],
        default=str(_CFG_DEFAULTS["assignment_method"]),
        help="Pairing solver used each frame.",
    )
    parser.add_argument(
        "--match-within-class",
        action=argparse.BooleanOptionalAction,
        default=bool(_CFG_DEFAULTS["match_within_class"]),
        help="Restrict matching to same class_id pairs.",
    )
    parser.add_argument(
        "--filter-short-tracks-in-summary",
        action=argparse.BooleanOptionalAction,
        default=bool(_CFG_DEFAULTS["filter_short_tracks_in_summary"]),
        help="Only count tracks with hits >= min_track_length as valid in summary stats.",
    )
    parser.add_argument(
        "--activation-topk",
        type=int,
        default=int(_CFG_DEFAULTS["activation_topk"]),
        help="Keep first K activation dims and L2-renormalize before linking (default: 64).",
    )
    parser.add_argument(
        "--max-centroid-distance",
        type=float,
        default=float(_CFG_DEFAULTS["max_centroid_distance"]),
        help=(
            "Maximum normalized centroid distance allowed for frame-to-frame matches, as a fraction "
            "of frame diagonal (default: 0.40)."
        ),
    )

    parser.add_argument(
        "--relink-threshold",
        type=float,
        default=float(_CFG_DEFAULTS["relink_threshold"]),
        help="YOLO centroid similarity gate used in post-hoc relinking.",
    )
    parser.add_argument(
        "--relink-max-gap-frames",
        type=int,
        default=int(_CFG_DEFAULTS["relink_max_gap_frames"]),
        help="Maximum frame gap for relink candidates (default: -1 for unlimited).",
    )
    parser.add_argument(
        "--relink-min-track-hits",
        type=int,
        default=int(_CFG_DEFAULTS["relink_min_track_hits"]),
        help="Minimum hits required for tracks to be relink candidates.",
    )
    parser.add_argument(
        "--relink-max-pixels-per-frame",
        type=float,
        default=float(_CFG_DEFAULTS["relink_max_pixels_per_frame"]),
        help="Maximum plausible drift rate (pixels/frame) used by spatial relink fallback scoring.",
    )
    parser.add_argument(
        "--relink-fallback-threshold",
        type=float,
        default=float(_CFG_DEFAULTS["relink_fallback_threshold"]),
        help="Fallback score threshold used after centroid-based relinking.",
    )
    parser.add_argument(
        "--relink-dino-threshold",
        type=float,
        default=float(_CFG_DEFAULTS["relink_dino_threshold"]),
        help="DINO similarity gate used when both relink fragments have DINO representatives.",
    )
    parser.add_argument(
        "--no-relink-dino",
        action="store_true",
        help="Disable DINO relink scoring and force YOLO centroid relink path.",
    )
    return parser


def _load_runner():
    from .pipeline import run_temporal_linking

    return run_temporal_linking


def _default_linking_output_dir(enriched_json_path: str) -> str:
    scenario = scenario_name_from_enriched_json(enriched_json_path)
    return os.path.join("experiments", "results", "linking", scenario)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir if args.output_dir else _default_linking_output_dir(args.enriched_json)

    run_temporal_linking = _load_runner()
    config = TemporalLinkingConfig.from_cli_namespace(args)

    try:
        outputs = run_temporal_linking(
            enriched_json_path=args.enriched_json,
            output_dir=output_dir,
            config=config,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Saved linked detections to {outputs.linked_detections}")
    print(f"Saved tracks to {outputs.tracks}")
    print(f"Saved linking manifest to {outputs.linking_manifest}")
    print(f"Saved relink manifest to {outputs.relink_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
