from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import run_single_video as single_video
from temporal_linking.track_docs import TrackDocOutputs


class RunSingleVideoTests(unittest.TestCase):
    def test_run_single_video_wires_stage_outputs_into_expected_dirs(self) -> None:
        parser = single_video.build_parser()
        args = parser.parse_args(["--video", "/tmp/demo.mp4", "--output-root", "/tmp/results"])

        calls: dict[str, dict[str, object]] = {}

        def fake_enrichment(**kwargs):
            calls["enrichment"] = kwargs
            return SimpleNamespace(enriched_detections="/tmp/results/activation_enrichment/demo/enriched_detections.json")

        def fake_linking(**kwargs):
            calls["linking"] = kwargs
            return SimpleNamespace(
                tracks="/tmp/results/linking/demo/tracks.json",
                trace_summary="/tmp/results/linking/demo/trace_summary.json",
                relink_manifest="/tmp/results/linking/demo/relink_manifest.json",
            )

        def fake_track_docs(**kwargs):
            calls["track_docs"] = kwargs
            return TrackDocOutputs(
                scenario="demo",
                output_dir="/tmp/results/linking/demo/track_docs",
                index_path="/tmp/results/linking/demo/track_docs/README.md",
                num_tracks=1,
                num_summary_images=1,
                num_frame_images=3,
            )

        outputs = single_video.run_single_video(
            args,
            run_trace_enrichment_fn=fake_enrichment,
            run_temporal_linking_fn=fake_linking,
            build_track_docs_fn=fake_track_docs,
        )

        self.assertEqual(outputs.scenario, "demo")
        self.assertEqual(outputs.track_docs_dir, "/tmp/results/linking/demo/track_docs")
        self.assertEqual(calls["enrichment"]["output_dir"], "/tmp/results/activation_enrichment/demo")
        self.assertEqual(calls["linking"]["output_dir"], "/tmp/results/linking/demo")
        self.assertEqual(calls["linking"]["enriched_json_path"], calls["enrichment"]["output_dir"] + "/enriched_detections.json")
        self.assertAlmostEqual(calls["linking"]["config"].similarity_threshold, 0.70, places=6)
        self.assertEqual(calls["track_docs"]["scenario"], "demo")
        self.assertEqual(calls["track_docs"]["output_dir"], "/tmp/results/linking/demo/track_docs")
        self.assertEqual(calls["track_docs"]["video_path"], "/tmp/demo.mp4")


if __name__ == "__main__":
    unittest.main()
