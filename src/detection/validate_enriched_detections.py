#!/usr/bin/env python3
"""Compatibility wrapper for enriched detection validation CLI.

Prefer `python -m src.detection.enrichment.validate`.
"""

from .enrichment.validate import main


if __name__ == "__main__":
    main()
