#!/usr/bin/env python3
"""Top-level runner for temporal linking."""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(__file__))
    from temporal_linking.cli import main  # type: ignore
else:
    from .temporal_linking.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
