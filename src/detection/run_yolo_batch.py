#!/usr/bin/env python3
"""Run YOLOv8 on all videos in a directory."""

from __future__ import annotations

import os
import sys
 
if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from detection.cli import build_parser, run_batch  # type: ignore
else:
    from .cli import build_parser, run_batch


def main() -> None:
    """Parse args and run batch detection."""
    print(
        "DEPRECATION: `src/detection/run_yolo_batch.py` is detection-only. "
        "Use `python3 src/run_pipeline.py --video-dir <dir> --pattern \"*.mp4\" --model yolov8n.pt --sample-rate 5` "
        "for Phase 1 outputs.",
        file=sys.stderr,
    )
    parser = build_parser(mode="batch")
    args = parser.parse_args()
    outputs = run_batch(args)
    if not outputs:
        print(f"No videos found in {args.video_dir} matching {args.pattern}")
        return
    for output_path in outputs:
        print(f"Saved detections to {output_path}")


if __name__ == "__main__":
    main()
