"""CLI for single-pass enriched trace generation."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HEAD_LAYER,
    DEFAULT_HEAD_STRIDE,
    DEFAULT_OUTPUT_ROOT,
    OUTPUT_VECTOR_DIM,
)
from .io import build_enriched_output_dir, find_videos
from .types import TraceEnrichmentOutputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-pass YOLO detection + activation trace enrichment on one video or a batch."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", help="Path to a single input video.")
    inputs.add_argument("--video-dir", help="Directory containing input videos for batch mode.")

    parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob pattern for batch mode (default: *.mp4). Ignored in single-video mode.",
    )
    parser.add_argument("--model", required=True, help="Ultralytics YOLOv8 model weights or model identifier.")
    parser.add_argument("--sample-rate", type=int, default=5, help="Sample every N frames (default: 5).")

    parser.add_argument(
        "--head-layer",
        default=DEFAULT_HEAD_LAYER,
        help=(
            "Single-layer hook target for fallback mode. Accepts exact module names (e.g. '15'), "
            "model-style paths (e.g. 'model.model[15]'), or aliases 'neck.C2f.15' / 'neck.C2f.mid'. "
            "Default run uses EMBEDDING_LAYERS unless TRACE_DISABLE_MULTI_LAYER_EMBEDDING=1. "
            "DINO sidecar extraction is enabled by default unless TRACE_DISABLE_DINO=1."
        ),
    )
    parser.add_argument(
        "--head-stride",
        type=int,
        default=DEFAULT_HEAD_STRIDE,
        help="Hook layer stride metadata stored in manifest (default: 8).",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="YOLO inference batch size.")
    parser.add_argument("--pca-dim", type=int, default=OUTPUT_VECTOR_DIM, help="Target PCA dimension.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for enriched outputs.")
    return parser


def _load_runner():
    from .pipeline import run_trace_enrichment

    return run_trace_enrichment


def _run_for_video(run_trace_enrichment, args: argparse.Namespace, video_path: str) -> TraceEnrichmentOutputs:
    output_dir = build_enriched_output_dir(args.output_root, video_path)
    return run_trace_enrichment(
        video_path=video_path,
        model_name=args.model,
        output_dir=output_dir,
        sample_rate=args.sample_rate,
        layer_name=args.head_layer,
        stride=args.head_stride,
        batch_size=args.batch_size,
        pca_dim=args.pca_dim,
    )


def _print_artifacts(outputs: TraceEnrichmentOutputs) -> None:
    print(f"Saved enriched detections to {outputs.enriched_detections}")
    print(f"Saved PCA projection to {outputs.pca_projection}")
    print(f"Saved projection manifest to {outputs.projection_manifest}")


def _run_single(run_trace_enrichment, args: argparse.Namespace) -> int:
    try:
        outputs = _run_for_video(run_trace_enrichment, args, args.video)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_artifacts(outputs)
    return 0


def _run_batch(run_trace_enrichment, args: argparse.Namespace) -> int:
    videos = find_videos(args.video_dir, args.pattern)
    if not videos:
        print(f"No videos found in {args.video_dir} matching {args.pattern}")
        return 0

    failures: list[tuple[str, str]] = []
    success_count = 0
    total = len(videos)

    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{total}] Processing {video_path}")
        try:
            _run_for_video(run_trace_enrichment, args, video_path)
        except Exception as exc:
            failures.append((video_path, str(exc)))
            print(f"[{index}/{total}] ERROR {video_path}: {exc}", file=sys.stderr)
            continue
        success_count += 1
        print(f"[{index}/{total}] OK {video_path} -> {build_enriched_output_dir(args.output_root, video_path)}")

    print("")
    print("Batch summary")
    print(f"  total: {total}")
    print(f"  succeeded: {success_count}")
    print(f"  failed: {len(failures)}")
    if failures:
        print("  failed videos:")
        for video_path, error in failures:
            print(f"    - {video_path}: {error}")

    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_trace_enrichment = _load_runner()
    if args.video:
        return _run_single(run_trace_enrichment, args)
    return _run_batch(run_trace_enrichment, args)


if __name__ == "__main__":
    raise SystemExit(main())
