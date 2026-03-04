"""Serialization helpers for temporal linking outputs."""

from __future__ import annotations

from dataclasses import asdict
import os
from typing import Any

from .config import TemporalLinkingConfig
from .types import Assignment, FrameDetections, Track

SCHEMA_VERSION_TRACKS = "temporal_linking_v1"
SCHEMA_VERSION_MANIFEST = "temporal_linking_manifest_v1"


def match_link_meta(
    *,
    assignment: Assignment,
    track_id: int,
    track_status: str,
    age_since_seen: int,
) -> dict[str, Any]:
    return {
        "track_id": int(track_id),
        "track_status": track_status,
        "source_track_status": assignment.source_track_status.value,
        "visual_similarity": float(assignment.visual_similarity),
        "spatial_score": float(assignment.spatial_score),
        "total_score": float(assignment.total_score),
        "age_since_seen": int(age_since_seen),
    }


def new_track_link_meta(*, track_id: int, track_status: str) -> dict[str, Any]:
    return {
        "track_id": int(track_id),
        "track_status": track_status,
        "source_track_status": "new",
        "visual_similarity": None,
        "spatial_score": None,
        "total_score": None,
        "age_since_seen": 0,
    }


def serialize_linked_frame(
    frame: FrameDetections,
    per_det_link_meta: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    detections_payload: list[dict[str, Any]] = []
    for det in frame.detections:
        link_meta = per_det_link_meta.get(det.det_index)
        if link_meta is None:
            raise RuntimeError(f"Missing temporal link metadata for frame={frame.frame_num} det_index={det.det_index}")

        det_payload = dict(det.raw_payload)
        det_payload["det_index"] = int(det.det_index)
        det_payload["temporal_link"] = link_meta
        detections_payload.append(det_payload)

    return {
        "frame_num": int(frame.frame_num),
        "detections": detections_payload,
    }


def _track_payload(track: Track, cfg: TemporalLinkingConfig) -> dict[str, Any]:
    avg_visual_similarity: float | None = None
    if track.visual_similarity_count > 0:
        avg_visual_similarity = track.visual_similarity_sum / float(track.visual_similarity_count)

    return {
        "track_id": int(track.track_id),
        "class_id": int(track.class_id),
        "class_name": track.class_name,
        "status": track.status.value,
        "start_frame": int(track.start_frame),
        "end_frame": int(track.last_seen_frame),
        "hits": int(track.hits),
        "total_misses": int(track.total_misses),
        "max_miss_streak": int(track.max_miss_streak),
        "avg_visual_similarity": None if avg_visual_similarity is None else float(avg_visual_similarity),
        "valid_track": bool(track.hits >= cfg.min_track_length),
        "events": list(track.events),
        "observations": list(track.observations),
    }


def serialize_tracks(closed_tracks: list[Track], cfg: TemporalLinkingConfig) -> dict[str, Any]:
    tracks = [_track_payload(track, cfg) for track in closed_tracks]

    if cfg.filter_short_tracks_in_summary:
        valid_tracks = [item for item in tracks if item["valid_track"]]
    else:
        valid_tracks = tracks

    return {
        "schema_version": SCHEMA_VERSION_TRACKS,
        "tracks": tracks,
        "summary": {
            "num_tracks_total": len(tracks),
            "num_tracks_valid": len(valid_tracks),
        },
    }


def serialize_manifest(
    *,
    enriched_json_path: str,
    linked_frames: list[dict[str, Any]],
    tracks_payload: dict[str, Any],
    cfg: TemporalLinkingConfig,
) -> dict[str, Any]:
    tracks = tracks_payload.get("tracks", [])

    num_recoveries = 0
    for track in tracks:
        for event in track.get("events", []):
            if event.get("type") == "recovered":
                num_recoveries += 1

    num_detections = sum(len(frame.get("detections", [])) for frame in linked_frames)
    num_valid_tracks = sum(1 for track in tracks if bool(track.get("valid_track")))

    return {
        "schema_version": SCHEMA_VERSION_MANIFEST,
        "input_enriched_json": os.path.basename(enriched_json_path),
        "config": asdict(cfg),
        "stats": {
            "num_frames": int(len(linked_frames)),
            "num_detections": int(num_detections),
            "num_tracks_total": int(len(tracks)),
            "num_tracks_valid": int(num_valid_tracks),
            "num_recoveries": int(num_recoveries),
        },
    }
