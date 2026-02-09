"""Command-line interfaces for detection pipelines."""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

from .pipeline import run_pipeline


def build_parser(mode: str) -> argparse.ArgumentParser:
    """Build an argument parser for single or batch execution."""
    if mode not in ("single", "batch"):
        raise ValueError("mode must be 'single' or 'batch'")

    description = "Run YOLOv8 on sampled video frames."
    if mode == "batch":
        description = "Run YOLOv8 on all videos in a directory."

    parser = argparse.ArgumentParser(description=description)
    if mode == "single":
        parser.add_argument(
            "--video",
            default="data/raw_videos/3sec_Left_to_Right.mp4",
            help="Path to input video.",
        )
    else:
        parser.add_argument(
            "--video-dir",
            default="data/raw_videos",
            help="Directory containing input videos.",
        )
        parser.add_argument(
            "--pattern",
            default="*.mp4",
            help="Glob pattern for video files (default: *.mp4).",
        )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=5,
        help="Sample every N frames (default: 5).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLOv8 model name or path.",
    )
    return parser


def find_videos(video_dir: str, pattern: str) -> List[str]:
    """Return a sorted list of video paths matching a glob pattern."""
    glob_pattern = os.path.join(video_dir, pattern)
    return sorted(glob.glob(glob_pattern))


def run_single(args: argparse.Namespace) -> str:
    """Run the pipeline for a single video."""
    return run_pipeline(args.video, args.sample_rate, args.model)


def run_batch(args: argparse.Namespace) -> List[str]:
    """Run the pipeline for every matching video in a directory."""
    videos = find_videos(args.video_dir, args.pattern)
    outputs: List[str] = []
    for video_path in videos:
        outputs.append(run_pipeline(video_path, args.sample_rate, args.model))
    return outputs
