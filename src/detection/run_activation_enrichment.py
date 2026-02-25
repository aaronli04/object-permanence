#!/usr/bin/env python3
"""Deprecated compatibility wrapper for the retired two-pass enrichment CLI."""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from detection.enrichment.run import main  # type: ignore
else:
    from .enrichment.run import main


if __name__ == "__main__":
    main()
