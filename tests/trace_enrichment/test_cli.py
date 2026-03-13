from __future__ import annotations

import os
import sys
import unittest

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from trace_enrichment.cli import build_parser
from trace_enrichment.constants import DEFAULT_SAMPLE_RATE


class TraceEnrichmentCliTests(unittest.TestCase):
    def test_sample_rate_defaults_to_frame_by_frame(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--video", "video.mp4", "--model", "yolov8n.pt"])
        self.assertEqual(args.sample_rate, DEFAULT_SAMPLE_RATE)

    def test_sample_rate_can_still_be_overridden(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--video", "video.mp4", "--model", "yolov8n.pt", "--sample-rate", "3"])
        self.assertEqual(args.sample_rate, 3)


if __name__ == "__main__":
    unittest.main()
