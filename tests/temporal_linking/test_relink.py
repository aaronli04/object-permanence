from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.relink import (
    build_candidates,
    build_fragments,
    greedy_assign,
    resolve_chains,
    score_centroid,
    score_fallback,
)
from temporal_linking.types import RelinkEdge, Track, TrackFragment, TrackStatus


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _make_closed_track(track_id: int, class_id: int, frames: list[int], vecs: list[np.ndarray]) -> Track:
    if len(frames) != len(vecs):
        raise ValueError("frames and vecs must be the same length")

    track = Track(
        track_id=track_id,
        class_id=class_id,
        class_name="sports ball" if class_id == 32 else "other",
        status=TrackStatus.CLOSED,
        start_frame=frames[0],
        last_seen_frame=frames[-1],
        hits=len(frames),
    )
    track.observations = []
    track.obs_vecs = []
    for idx, (frame_num, vec) in enumerate(zip(frames, vecs)):
        track.observations.append(
            {
                "frame_num": int(frame_num),
                "det_index": int(idx),
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "visual_similarity": None if idx == 0 else 0.9,
            }
        )
        track.obs_vecs.append(_normalize(vec))
    return track


class RelinkUnitTests(unittest.TestCase):
    def test_candidate_filtering_respects_hits_class_and_gap(self) -> None:
        e1 = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        tracks = [
            _make_closed_track(1, 32, [0, 5], [e1, e1]),
            _make_closed_track(2, 32, [20, 25], [e1, e1]),
            _make_closed_track(3, 0, [30, 35], [e1, e1]),
            _make_closed_track(4, 32, [10], [e1]),  # excluded by min_hits=2
        ]

        fragments = build_fragments(tracks, min_hits=2)
        candidates = build_candidates(fragments, max_gap_frames=20)
        pairs = {(a.track_id, b.track_id) for a, b in candidates}

        self.assertEqual({f.track_id for f in fragments}, {1, 2, 3})
        self.assertEqual(pairs, {(1, 2)})

    def test_centroid_and_fallback_scoring(self) -> None:
        e1 = _normalize(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        e2 = _normalize(np.asarray([0.9, 0.1, 0.0], dtype=np.float32))
        a = TrackFragment(
            track_id=10,
            class_id=32,
            first_frame=0,
            last_frame=5,
            hits=2,
            centroid=e1,
            frame_vecs=np.stack([e1, e1], axis=0),
        )
        b = TrackFragment(
            track_id=11,
            class_id=32,
            first_frame=10,
            last_frame=15,
            hits=2,
            centroid=e2,
            frame_vecs=np.stack([e2, e2], axis=0),
        )

        centroid_score = score_centroid([(a, b)])[0].score
        fallback_score = score_fallback([(a, b)], percentile=90.0)[0].score

        self.assertGreater(centroid_score, 0.9)
        self.assertGreater(fallback_score, 0.9)

    def test_greedy_assign_prioritizes_centroid_then_fallback(self) -> None:
        centroid_edges = [
            RelinkEdge(predecessor_id=1, successor_id=2, score=0.82, method="centroid"),
            RelinkEdge(predecessor_id=1, successor_id=3, score=0.78, method="centroid"),
        ]
        fallback_edges = [
            RelinkEdge(predecessor_id=1, successor_id=3, score=0.99, method="fallback"),
            RelinkEdge(predecessor_id=4, successor_id=5, score=0.60, method="fallback"),
        ]

        accepted = greedy_assign(
            centroid_edges=centroid_edges,
            fallback_edges=fallback_edges,
            relink_threshold=0.8,
            fallback_threshold=0.4,
        )
        got = {(edge.predecessor_id, edge.successor_id, edge.method) for edge in accepted}

        self.assertIn((1, 2, "centroid"), got)
        self.assertNotIn((1, 3, "fallback"), got)
        self.assertIn((4, 5, "fallback"), got)

    def test_resolve_chains_is_deterministic(self) -> None:
        fragments = [
            TrackFragment(
                track_id=10,
                class_id=32,
                first_frame=5,
                last_frame=6,
                hits=2,
                centroid=np.asarray([1.0, 0.0], dtype=np.float32),
                frame_vecs=np.asarray([[1.0, 0.0]], dtype=np.float32),
            ),
            TrackFragment(
                track_id=2,
                class_id=32,
                first_frame=0,
                last_frame=1,
                hits=2,
                centroid=np.asarray([1.0, 0.0], dtype=np.float32),
                frame_vecs=np.asarray([[1.0, 0.0]], dtype=np.float32),
            ),
            TrackFragment(
                track_id=7,
                class_id=32,
                first_frame=0,
                last_frame=2,
                hits=2,
                centroid=np.asarray([1.0, 0.0], dtype=np.float32),
                frame_vecs=np.asarray([[1.0, 0.0]], dtype=np.float32),
            ),
        ]
        accepted = [
            RelinkEdge(predecessor_id=10, successor_id=2, score=0.9, method="centroid"),
            RelinkEdge(predecessor_id=2, successor_id=7, score=0.9, method="centroid"),
        ]

        merge_map = resolve_chains(accepted, fragments)
        self.assertEqual(merge_map, {7: 2, 10: 2})


if __name__ == "__main__":
    unittest.main()
