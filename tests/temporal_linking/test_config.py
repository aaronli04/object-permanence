from __future__ import annotations

import argparse
import os
import sys
import unittest

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.config import TemporalLinkingConfig


class TemporalLinkingConfigTests(unittest.TestCase):
    def test_defaults_exposes_dataclass_defaults(self) -> None:
        defaults = TemporalLinkingConfig.defaults()
        self.assertEqual(int(defaults["max_lost_frames"]), 6)
        self.assertEqual(float(defaults["ema_alpha"]), 0.35)
        self.assertNotIn("similarity_threshold", defaults)

    def test_from_cli_namespace_maps_no_relink_dino_flag(self) -> None:
        args = argparse.Namespace(
            similarity_threshold=0.7,
            max_lost_frames=4,
            min_hits_to_activate=2,
            min_track_length=2,
            history_size=5,
            ema_alpha=0.35,
            w_last=0.55,
            w_ema=0.30,
            w_hist=0.15,
            w_spatial=0.05,
            w_consistency=0.10,
            w_age=0.05,
            assignment_method="hungarian",
            match_within_class=True,
            filter_short_tracks_in_summary=True,
            activation_topk=64,
            max_centroid_distance=0.40,
            relink_threshold=0.55,
            relink_max_gap_frames=-1,
            relink_min_track_hits=2,
            relink_max_pixels_per_frame=15.0,
            relink_fallback_threshold=0.40,
            relink_dino_threshold=0.55,
            no_relink_dino=True,
        )

        cfg = TemporalLinkingConfig.from_cli_namespace(args)
        self.assertFalse(bool(cfg.relink_use_dino))
        self.assertEqual(int(cfg.relink_dino_min_detections), 2)

    def test_from_cli_values_requires_similarity_threshold(self) -> None:
        with self.assertRaises(ValueError):
            TemporalLinkingConfig.from_cli_values({"max_lost_frames": 3})


if __name__ == "__main__":
    unittest.main()
