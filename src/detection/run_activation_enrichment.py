#!/usr/bin/env python3
"""Compatibility wrapper for activation enrichment CLI.

Prefer `python -m src.detection.enrichment.run`.
"""

from .enrichment.run import main


if __name__ == "__main__":
    main()
