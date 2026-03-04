#!/usr/bin/env python3
"""Validate temporal linking output schema and consistency."""

from __future__ import annotations

import argparse
import json
from typing import Any


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_linked_detections(
    linked_frames: list[dict[str, Any]],
    *,
    similarity_threshold: float | None = None,
) -> dict[str, int]:
    if not isinstance(linked_frames, list):
        raise AssertionError("linked_detections.json must be a top-level list")

    frame_count = len(linked_frames)
    det_count = 0

    for frame in linked_frames:
        if "frame_num" not in frame or "detections" not in frame:
            raise AssertionError("Each frame must contain frame_num and detections")
        detections = frame["detections"]
        if not isinstance(detections, list):
            raise AssertionError("detections must be a list")

        seen_track_ids: set[int] = set()
        for det in detections:
            det_count += 1
            if "det_index" not in det:
                raise AssertionError(f"Missing det_index in frame {frame['frame_num']}")
            temporal_link = det.get("temporal_link")
            if not isinstance(temporal_link, dict):
                raise AssertionError(f"Missing temporal_link in frame {frame['frame_num']} det {det.get('det_index')}")

            track_id = temporal_link.get("track_id")
            if not isinstance(track_id, int):
                raise AssertionError(f"track_id must be int in frame {frame['frame_num']} det {det.get('det_index')}")
            if track_id in seen_track_ids:
                raise AssertionError(f"Duplicate track_id={track_id} in frame {frame['frame_num']}")
            seen_track_ids.add(track_id)

            source_status = temporal_link.get("source_track_status")
            visual_similarity = temporal_link.get("visual_similarity")
            if source_status != "new":
                if not isinstance(visual_similarity, (int, float)):
                    raise AssertionError(
                        f"visual_similarity must be numeric for matched detections in frame {frame['frame_num']}"
                    )
                if similarity_threshold is not None and float(visual_similarity) < float(similarity_threshold):
                    raise AssertionError(
                        f"visual_similarity {visual_similarity} below configured threshold "
                        f"{similarity_threshold} in frame {frame['frame_num']}"
                    )

    return {"num_frames": frame_count, "num_detections": det_count}


def validate_tracks_payload(tracks_payload: dict[str, Any]) -> dict[str, int]:
    if not isinstance(tracks_payload, dict):
        raise AssertionError("tracks.json must be a top-level object")
    tracks = tracks_payload.get("tracks")
    if not isinstance(tracks, list):
        raise AssertionError("tracks.json must contain list field 'tracks'")

    track_ids: set[int] = set()
    valid_count = 0
    for track in tracks:
        if not isinstance(track, dict):
            raise AssertionError("each track entry must be an object")
        track_id = track.get("track_id")
        if not isinstance(track_id, int):
            raise AssertionError("track_id must be int")
        if track_id in track_ids:
            raise AssertionError(f"Duplicate track_id={track_id} in tracks.json")
        track_ids.add(track_id)

        if bool(track.get("valid_track")):
            valid_count += 1

    return {"num_tracks_total": len(tracks), "num_tracks_valid": valid_count}


def validate_manifest_payload(manifest_payload: dict[str, Any], expected: dict[str, int]) -> None:
    if not isinstance(manifest_payload, dict):
        raise AssertionError("linking_manifest.json must be a top-level object")

    stats = manifest_payload.get("stats")
    if not isinstance(stats, dict):
        raise AssertionError("linking_manifest.json missing stats object")

    for key, value in expected.items():
        manifest_value = stats.get(key)
        if manifest_value is None:
            raise AssertionError(f"manifest missing stats.{key}")
        if int(manifest_value) != int(value):
            raise AssertionError(
                f"manifest stats mismatch for {key}: expected={value}, got={manifest_value}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate temporal linking outputs.")
    parser.add_argument("linked_json", help="Path to linked_detections.json")
    parser.add_argument("tracks_json", help="Path to tracks.json")
    parser.add_argument("manifest_json", help="Path to linking_manifest.json")
    args = parser.parse_args()

    manifest_payload = _load_json(args.manifest_json)
    cfg = manifest_payload.get("config", {}) if isinstance(manifest_payload, dict) else {}
    similarity_threshold = cfg.get("similarity_threshold")

    linked_stats = validate_linked_detections(
        _load_json(args.linked_json),
        similarity_threshold=float(similarity_threshold) if similarity_threshold is not None else None,
    )
    tracks_stats = validate_tracks_payload(_load_json(args.tracks_json))

    expected_stats = {
        "num_frames": linked_stats["num_frames"],
        "num_detections": linked_stats["num_detections"],
        "num_tracks_total": tracks_stats["num_tracks_total"],
        "num_tracks_valid": tracks_stats["num_tracks_valid"],
    }
    validate_manifest_payload(manifest_payload, expected=expected_stats)

    print(f"num_frames={linked_stats['num_frames']}")
    print(f"num_detections={linked_stats['num_detections']}")
    print(f"num_tracks_total={tracks_stats['num_tracks_total']}")
    print(f"num_tracks_valid={tracks_stats['num_tracks_valid']}")
    print("Validation passed")


if __name__ == "__main__":
    main()
