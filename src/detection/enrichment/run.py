#!/usr/bin/env python3
"""Deprecated CLI entry point for the retired two-pass enrichment pipeline."""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "DEPRECATION: The two-pass enrichment CLI (`src/detection/enrichment/run.py`) is retired. "
        "Phase 1 is now single-pass and does not use an intermediate detections JSON. "
        "Use `python3 src/run_pipeline.py --video <path> --model yolov8n.pt --sample-rate 5` "
        "or batch mode with `--video-dir`.",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
