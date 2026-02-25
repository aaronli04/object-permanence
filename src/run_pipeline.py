#!/usr/bin/env python3
"""Unified Phase 1 pipeline CLI (single-pass detection + enrichment)."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Sequence, Tuple


DEFAULT_OUTPUT_ROOT = os.path.join("experiments", "results", "enriched")


def _import_pipeline():
    """Import heavy pipeline module lazily so `--help` works without runtime deps."""
    if __package__ in (None, ""):
        sys.path.append(os.path.dirname(__file__))
        from detection.enrichment.pipeline import run_phase1_enrichment  # type: ignore
    else:
        from .detection.enrichment.pipeline import run_phase1_enrichment
    return run_phase1_enrichment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 single-pass YOLO detection + activation enrichment on one video or a batch."
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

    parser.add_argument("--deep-layer", default="8", help="Named module for deep backbone C2f hook (default: 8).")
    parser.add_argument("--mid-layer", default="6", help="Named module for mid backbone C2f hook (default: 6).")
    parser.add_argument("--deep-stride", type=int, default=32, help="Feature stride for deep hook layer.")
    parser.add_argument("--mid-stride", type=int, default=16, help="Feature stride for mid hook layer.")
    parser.add_argument("--batch-size", type=int, default=8, help="YOLO inference batch size (default: 8).")
    parser.add_argument("--pca-dim", type=int, default=256, help="Target PCA dimension (default: 256).")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory for per-video enriched artifacts.",
    )
    return parser


def _video_name(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def _output_dir(output_root: str, video_path: str) -> str:
    return os.path.join(output_root, _video_name(video_path))


def _run_one(args: argparse.Namespace, video_path: str) -> Dict[str, str]:
    run_phase1_enrichment = _import_pipeline()
    return run_phase1_enrichment(
        video_path=video_path,
        model_name=args.model,
        output_dir=_output_dir(args.output_root, video_path),
        sample_rate=args.sample_rate,
        deep_layer_name=args.deep_layer,
        mid_layer_name=args.mid_layer,
        stride_deep=args.deep_stride,
        stride_mid=args.mid_stride,
        batch_size=args.batch_size,
        pca_dim=args.pca_dim,
    )


def _print_artifacts(outputs: Dict[str, str]) -> None:
    print(f"Saved enriched detections to {outputs['enriched_detections']}")
    print(f"Saved PCA projection to {outputs['pca_projection']}")
    print(f"Saved projection manifest to {outputs['projection_manifest']}")


def _run_single(args: argparse.Namespace) -> int:
    try:
        outputs = _run_one(args, args.video)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _print_artifacts(outputs)
    return 0


def _find_videos(video_dir: str, pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(video_dir, pattern)))


def _run_batch(args: argparse.Namespace) -> int:
    videos = _find_videos(args.video_dir, args.pattern)
    if not videos:
        print(f"No videos found in {args.video_dir} matching {args.pattern}")
        return 0

    successes: List[Tuple[str, Dict[str, str]]] = []
    failures: List[Tuple[str, str]] = []
    total = len(videos)

    for idx, video_path in enumerate(videos, start=1):
        print(f"[{idx}/{total}] Processing {video_path}")
        try:
            outputs = _run_one(args, video_path)
        except Exception as exc:
            failures.append((video_path, str(exc)))
            print(f"[{idx}/{total}] ERROR {video_path}: {exc}", file=sys.stderr)
            continue

        successes.append((video_path, outputs))
        print(f"[{idx}/{total}] OK {video_path} -> {_output_dir(args.output_root, video_path)}")

    print("")
    print("Batch summary")
    print(f"  total: {total}")
    print(f"  succeeded: {len(successes)}")
    print(f"  failed: {len(failures)}")
    if failures:
        print("  failed videos:")
        for video_path, error in failures:
            print(f"    - {video_path}: {error}")

    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video:
        return _run_single(args)
    return _run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())

