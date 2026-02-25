#!/usr/bin/env python3
"""Top-level runner for enriched trace generation."""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(__file__))
    from trace_enrichment.cli import main  # type: ignore
else:
    from .trace_enrichment.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
