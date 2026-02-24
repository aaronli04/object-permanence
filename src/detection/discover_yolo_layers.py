#!/usr/bin/env python3
"""Compatibility wrapper for YOLO layer discovery.

Prefer `python -m src.detection.enrichment.discover_layers`.
"""

from .enrichment.discover_layers import main


if __name__ == "__main__":
    main()
