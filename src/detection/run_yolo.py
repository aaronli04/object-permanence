#!/usr/bin/env python3
"""CLI entry point for YOLOv8 frame detection."""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from detection.pipeline import run_pipeline  # type: ignore
else:
    from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 on sampled video frames.")
    parser.add_argument(
        "--video",
        default="data/raw_videos/3sec_Left_to_Right.mp4",
        help="Path to input video.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=30,
        help="Sample every N frames (default: 30).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLOv8 model name or path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_pipeline(args.video, args.sample_rate, args.model)
    print(f"Saved detections to {output_path}")


if __name__ == "__main__":
    main()
