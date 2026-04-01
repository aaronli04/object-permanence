#!/usr/bin/env python3
"""Run enrichment, linking, and track docs for a single video."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
from typing import Any, Callable, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_SRC = os.path.join(PROJECT_ROOT, "src")
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from common.paths import video_stem
from temporal_linking.config import TemporalLinkingConfig
from temporal_linking.track_docs import TrackDocOutputs, build_track_docs_from_paths
from trace_enrichment.constants import DEFAULT_BATCH_SIZE, DEFAULT_HEAD_LAYER, DEFAULT_HEAD_STRIDE, DEFAULT_SAMPLE_RATE, OUTPUT_VECTOR_DIM


@dataclass(frozen=True)
class SingleVideoRunOutputs:
    scenario: str
    enrichment_dir: str
    linking_dir: str
    enriched_detections: str
    tracks: str
    trace_summary: str
    relink_manifest: str
    track_docs_dir: str | None


def build_parser() -> argparse.ArgumentParser:
    cfg_defaults = TemporalLinkingConfig.defaults()
    parser = argparse.ArgumentParser(description="Run the full object permanence pipeline on one video.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLOv8 model weights or model identifier.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("experiments", "results"),
        help="Root directory for activation_enrichment/ and linking/ outputs.",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sample every N frames.")
    parser.add_argument("--head-layer", default=DEFAULT_HEAD_LAYER, help="Fallback single-layer hook target.")
    parser.add_argument("--head-stride", type=int, default=DEFAULT_HEAD_STRIDE, help="Hook stride metadata.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="YOLO inference batch size.")
    parser.add_argument("--pca-dim", type=int, default=OUTPUT_VECTOR_DIM, help="Target PCA dimension.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.70,
        help="Frame-to-frame cosine similarity gate.",
    )
    parser.add_argument(
        "--activation-topk",
        type=int,
        default=int(cfg_defaults["activation_topk"]),
        help="Keep first K activation dims before linking.",
    )
    parser.add_argument(
        "--max-centroid-distance",
        type=float,
        default=float(cfg_defaults["max_centroid_distance"]),
        help="Maximum normalized centroid distance for frame-to-frame matches.",
    )
    parser.add_argument(
        "--relink-threshold",
        type=float,
        default=float(cfg_defaults["relink_threshold"]),
        help="YOLO relink similarity gate.",
    )
    parser.add_argument(
        "--relink-dino-threshold",
        type=float,
        default=float(cfg_defaults["relink_dino_threshold"]),
        help="DINO relink similarity gate.",
    )
    parser.add_argument(
        "--relink-max-gap-frames",
        type=int,
        default=int(cfg_defaults["relink_max_gap_frames"]),
        help="Maximum relink gap in sampled frames (-1 means unlimited).",
    )
    parser.add_argument(
        "--relink-fallback-threshold",
        type=float,
        default=float(cfg_defaults["relink_fallback_threshold"]),
        help="Fallback spatial relink threshold.",
    )
    parser.add_argument(
        "--render-trace-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render track summary JPEGs (default: enabled).",
    )
    parser.add_argument(
        "--build-track-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build per-track doc folders after linking (default: enabled).",
    )
    parser.add_argument(
        "--no-relink-dino",
        action="store_true",
        help="Disable DINO relink scoring and force YOLO relink only.",
    )
    return parser


def _load_enrichment_runner():
    from trace_enrichment.pipeline import run_trace_enrichment

    return run_trace_enrichment


def _load_linking_runner():
    from temporal_linking.pipeline import run_temporal_linking

    return run_temporal_linking


def run_single_video(
    args: argparse.Namespace,
    *,
    run_trace_enrichment_fn: Callable[..., Any] | None = None,
    run_temporal_linking_fn: Callable[..., Any] | None = None,
    build_track_docs_fn: Callable[..., TrackDocOutputs] | None = None,
) -> SingleVideoRunOutputs:
    run_trace_enrichment = run_trace_enrichment_fn or _load_enrichment_runner()
    run_temporal_linking = run_temporal_linking_fn or _load_linking_runner()
    build_track_docs = build_track_docs_fn or build_track_docs_from_paths

    scenario = video_stem(args.video)
    enrichment_dir = os.path.join(args.output_root, "activation_enrichment", scenario)
    linking_dir = os.path.join(args.output_root, "linking", scenario)

    enrichment_outputs = run_trace_enrichment(
        video_path=args.video,
        model_name=args.model,
        output_dir=enrichment_dir,
        sample_rate=int(args.sample_rate),
        layer_name=args.head_layer,
        stride=int(args.head_stride),
        batch_size=int(args.batch_size),
        pca_dim=int(args.pca_dim),
    )

    linking_outputs = run_temporal_linking(
        enriched_json_path=enrichment_outputs.enriched_detections,
        output_dir=linking_dir,
        config=TemporalLinkingConfig.from_cli_namespace(args),
        render_trace_summary_artifacts=bool(args.render_trace_summary),
        video_path=args.video,
    )

    track_docs_dir: str | None = None
    if args.build_track_docs:
        track_doc_outputs = build_track_docs(
            scenario=scenario,
            tracks_path=linking_outputs.tracks,
            trace_summary_path=linking_outputs.trace_summary,
            output_dir=os.path.join(linking_dir, "track_docs"),
            trace_summary_dir=os.path.join(linking_dir, "trace_summary"),
            relink_manifest_path=linking_outputs.relink_manifest,
            video_path=args.video,
            render_trace_summary_images=bool(args.render_trace_summary),
        )
        track_docs_dir = track_doc_outputs.output_dir

    return SingleVideoRunOutputs(
        scenario=scenario,
        enrichment_dir=enrichment_dir,
        linking_dir=linking_dir,
        enriched_detections=enrichment_outputs.enriched_detections,
        tracks=linking_outputs.tracks,
        trace_summary=linking_outputs.trace_summary,
        relink_manifest=linking_outputs.relink_manifest,
        track_docs_dir=track_docs_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        outputs = run_single_video(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Scenario: {outputs.scenario}")
    print(f"Enrichment dir: {outputs.enrichment_dir}")
    print(f"Linking dir: {outputs.linking_dir}")
    print(f"Saved enriched detections to {outputs.enriched_detections}")
    print(f"Saved tracks to {outputs.tracks}")
    print(f"Saved relink manifest to {outputs.relink_manifest}")
    print(f"Saved trace summary to {outputs.trace_summary}")
    if outputs.track_docs_dir:
        print(f"Saved track docs to {outputs.track_docs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
