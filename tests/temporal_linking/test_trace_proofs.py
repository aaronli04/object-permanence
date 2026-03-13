from __future__ import annotations

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.config import TemporalLinkingConfig
from temporal_linking.trace_proofs import build_trace_references, proof_color, proof_color_index, render_trace_proofs
from temporal_linking.types import SerializedTrackObservation, Track, TrackStatus


def _make_closed_track(
    track_id: int,
    *,
    frames: list[int],
    events: list[dict[str, object]] | None = None,
) -> Track:
    track = Track(
        track_id=track_id,
        class_id=32,
        class_name="sports ball",
        status=TrackStatus.CLOSED,
        start_frame=frames[0],
        last_seen_frame=frames[-1],
        hits=len(frames),
    )
    track.events = list(events or [])
    track.observations = [
        SerializedTrackObservation(
            frame_num=frame_num,
            det_index=index,
            bbox=[10.0, 10.0, 30.0, 30.0],
            visual_similarity=None if index == 0 else 0.9,
            fragment_track_id=track_id,
        )
        for index, frame_num in enumerate(frames)
    ]
    return track


def _write_video(path: str, *, frame_count: int = 12) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 72))
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV video writer is unavailable in this environment.")
    for frame_num in range(frame_count):
        frame = np.full((72, 96, 3), 30 + (frame_num * 10), dtype=np.uint8)
        cv2.putText(frame, str(frame_num), (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


class TraceProofTests(unittest.TestCase):
    def test_build_trace_references_creates_deterministic_relink_item(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.7, min_hits_to_activate=1, relink_min_track_hits=1)
        predecessor = _make_closed_track(2, frames=[1, 2, 3, 4], events=[{"frame_num": 1, "type": "created"}])
        successor = _make_closed_track(7, frames=[8, 9, 10], events=[{"frame_num": 8, "type": "created"}])

        payload = build_trace_references(
            closed_tracks=[predecessor, successor],
            cfg=cfg,
            enriched_json_path="/tmp/enriched.json",
            relink_result={
                "accepted_edges": [
                    {"predecessor_id": 2, "successor_id": 7, "method": "dino", "score": 0.88}
                ]
            },
            merge_map={7: 2},
        )

        self.assertEqual(payload["schema_version"], "temporal_linking_trace_references_v1")
        item = payload["items"][0]
        self.assertEqual(item["proof_id"], "relink_2_7")
        self.assertEqual(item["canonical_track_id"], 2)
        self.assertEqual(item["fragment_track_ids"], [2, 7])
        self.assertEqual([group["name"] for group in item["frame_groups"]], ["pred_tail", "succ_head"])
        self.assertEqual(
            [frame["frame_num"] for frame in item["frame_groups"][0]["frames"]],
            [2, 3, 4],
        )
        self.assertEqual(
            [frame["frame_num"] for frame in item["frame_groups"][1]["frames"]],
            [8, 9, 10],
        )

    def test_build_trace_references_creates_recovery_gap_metadata(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.7, min_hits_to_activate=1)
        track = _make_closed_track(
            5,
            frames=[0, 1, 4, 5],
            events=[
                {"frame_num": 0, "type": "created"},
                {"frame_num": 2, "type": "lost"},
                {"frame_num": 4, "type": "recovered"},
                {"frame_num": 5, "type": "closed"},
            ],
        )

        payload = build_trace_references(
            closed_tracks=[track],
            cfg=cfg,
            enriched_json_path="/tmp/enriched.json",
            relink_result={"accepted_edges": []},
            merge_map={},
        )

        self.assertEqual(len(payload["items"]), 1)
        item = payload["items"][0]
        self.assertEqual(item["proof_id"], "recovery_5_4")
        self.assertEqual(item["gap_frames"], 2)
        self.assertEqual(
            [frame["frame_num"] for frame in item["frame_groups"][0]["frames"]],
            [0, 1],
        )
        self.assertEqual(
            [frame["frame_num"] for frame in item["frame_groups"][1]["frames"]],
            [4, 5],
        )

    def test_build_trace_references_emits_only_available_frames_without_padding(self) -> None:
        cfg = TemporalLinkingConfig(similarity_threshold=0.7, min_hits_to_activate=1, relink_min_track_hits=1)
        predecessor = _make_closed_track(3, frames=[2], events=[{"frame_num": 2, "type": "created"}])
        successor = _make_closed_track(9, frames=[6, 7], events=[{"frame_num": 6, "type": "created"}])

        payload = build_trace_references(
            closed_tracks=[predecessor, successor],
            cfg=cfg,
            enriched_json_path="/tmp/enriched.json",
            relink_result={
                "accepted_edges": [
                    {"predecessor_id": 3, "successor_id": 9, "method": "yolo", "score": 0.65}
                ]
            },
            merge_map={9: 3},
        )

        item = payload["items"][0]
        self.assertEqual(len(item["frame_groups"][0]["frames"]), 1)
        self.assertEqual(len(item["frame_groups"][1]["frames"]), 2)

    def test_proof_color_mapping_is_stable_and_not_direct_modulo(self) -> None:
        ids = [1, 2, 3, 4, 25]
        first_pass = [proof_color_index(track_id) for track_id in ids]
        second_pass = [proof_color_index(track_id) for track_id in ids]

        self.assertEqual(first_pass, second_pass)
        self.assertNotEqual(first_pass, [track_id % 24 for track_id in ids])
        self.assertEqual(proof_color(42), proof_color(42))

    def test_render_trace_proofs_writes_jpegs(self) -> None:
        payload = {
            "schema_version": "temporal_linking_trace_references_v1",
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            "config_hash_sha256": "abc",
            "input_enriched_json": "enriched.json",
            "items": [
                {
                    "kind": "relink",
                    "proof_id": "relink_2_7",
                    "canonical_track_id": 2,
                    "fragment_track_ids": [2, 7],
                    "method": "dino",
                    "score": 0.88,
                    "frame_groups": [
                        {
                            "name": "pred_tail",
                            "frames": [
                                {
                                    "frame_num": 2,
                                    "det_index": 0,
                                    "bbox": [10.0, 10.0, 30.0, 30.0],
                                    "fragment_track_id": 2,
                                    "canonical_track_id": 2,
                                    "role": "pred_tail",
                                }
                            ],
                        },
                        {
                            "name": "succ_head",
                            "frames": [
                                {
                                    "frame_num": 7,
                                    "det_index": 0,
                                    "bbox": [10.0, 10.0, 30.0, 30.0],
                                    "fragment_track_id": 7,
                                    "canonical_track_id": 2,
                                    "role": "succ_head",
                                }
                            ],
                        },
                    ],
                },
                {
                    "kind": "recovery",
                    "proof_id": "recovery_2_7",
                    "canonical_track_id": 2,
                    "fragment_track_ids": [2],
                    "gap_frames": 3,
                    "frame_groups": [
                        {
                            "name": "before_gap",
                            "frames": [
                                {
                                    "frame_num": 3,
                                    "det_index": 0,
                                    "bbox": [10.0, 10.0, 30.0, 30.0],
                                    "fragment_track_id": 2,
                                    "canonical_track_id": 2,
                                    "role": "before_gap",
                                }
                            ],
                        },
                        {
                            "name": "after_gap",
                            "frames": [
                                {
                                    "frame_num": 7,
                                    "det_index": 0,
                                    "bbox": [10.0, 10.0, 30.0, 30.0],
                                    "fragment_track_id": 2,
                                    "canonical_track_id": 2,
                                    "role": "after_gap",
                                }
                            ],
                        },
                    ],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.avi")
            _write_video(video_path)
            rendered = render_trace_proofs(
                trace_references_payload=payload,
                video_path=video_path,
                output_dir=os.path.join(tmpdir, "proofs"),
            )

            self.assertEqual(rendered, 2)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "proofs", "relink_2_7.jpg")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "proofs", "recovery_2_7.jpg")))


if __name__ == "__main__":
    unittest.main()
