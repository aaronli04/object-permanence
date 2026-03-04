from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.relink import build_candidates, build_fragments, score_centroid, score_fallback
from temporal_linking.types import Track, TrackStatus


def _load_track_fixture() -> list[Track]:
    base = Path("experiments/results/enriched/Right_to_left")
    tracks_path = base / "tracks.json"
    linked_path = base / "linked_detections.json"
    if not tracks_path.exists() or not linked_path.exists():
        raise unittest.SkipTest("Right_to_left fixture is unavailable in experiments/results/enriched.")

    tracks_payload = json.loads(tracks_path.read_text()).get("tracks", [])
    linked_payload = json.loads(linked_path.read_text())

    vec_map: dict[tuple[int, int], np.ndarray] = {}
    for frame in linked_payload:
        frame_num = int(frame["frame_num"])
        for det in frame.get("detections", []):
            vec = np.asarray(det["activation"]["vector"], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec = vec / norm
            vec_map[(frame_num, int(det["det_index"]))] = vec

    tracks: list[Track] = []
    for item in tracks_payload:
        status = TrackStatus(str(item.get("status", "closed")))
        track = Track(
            track_id=int(item["track_id"]),
            class_id=int(item["class_id"]),
            class_name=str(item.get("class_name", "")),
            status=status,
            start_frame=int(item["start_frame"]),
            last_seen_frame=int(item["end_frame"]),
            hits=int(item["hits"]),
            total_misses=int(item.get("total_misses", 0)),
            max_miss_streak=int(item.get("max_miss_streak", 0)),
        )
        track.events = list(item.get("events", []))
        track.observations = list(item.get("observations", []))
        track.obs_vecs = []
        track.obs_positions = []
        for obs in track.observations:
            key = (int(obs["frame_num"]), int(obs["det_index"]))
            vec = vec_map.get(key)
            if vec is not None:
                track.obs_vecs.append(vec.copy())
            bbox = obs.get("bbox", [0.0, 0.0, 0.0, 0.0])
            x1, y1, x2, y2 = [float(v) for v in bbox]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            track.obs_positions.append((cx, cy, int(obs["frame_num"])))
        tracks.append(track)

    return tracks


class RelinkCalibrationTests(unittest.TestCase):
    def test_long_gap_candidate_scores_are_finite(self) -> None:
        tracks = _load_track_fixture()
        fragments = build_fragments(tracks, min_hits=1)
        candidates = build_candidates(fragments, max_gap_frames=-1)

        target_pair = None
        for pred, succ in candidates:
            if pred.class_id != 32 or succ.class_id != 32:
                continue
            if pred.last_frame == 15 and succ.first_frame == 70:
                target_pair = (pred, succ)
                break

        self.assertIsNotNone(target_pair, "Expected sports-ball long-gap candidate (15 -> 70) to exist.")
        pred, succ = target_pair  # type: ignore[misc]

        centroid_edge = score_centroid([target_pair])[0]  # type: ignore[arg-type]
        fallback_edge = score_fallback([target_pair], max_pixels_per_frame=15.0)[0]  # type: ignore[arg-type]

        self.assertTrue(np.isfinite(centroid_edge.score))
        self.assertTrue(np.isfinite(fallback_edge.score))

        print("CALIBRATION_PROBE")
        print(f"track_ids={pred.track_id}->{succ.track_id}")
        print(f"gap_frames={succ.first_frame - pred.last_frame}")
        print(f"hits={pred.hits}->{succ.hits}")
        print(f"centroid_similarity={centroid_edge.score:.6f}")
        print(f"spatial_plausibility_score={fallback_edge.score:.6f}")


if __name__ == "__main__":
    unittest.main()
