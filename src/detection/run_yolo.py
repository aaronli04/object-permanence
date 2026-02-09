#!/usr/bin/env python3
"""CLI entry point for YOLOv8 frame detection."""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from detection.cli import build_parser, run_single  # type: ignore
else:
    from .cli import build_parser, run_single


def main() -> None:
    """Parse args and run single-video detection."""
    parser = build_parser(mode="single")
    args = parser.parse_args()
    output_path = run_single(args)
    print(f"Saved detections to {output_path}")


if __name__ == "__main__":
    main()
