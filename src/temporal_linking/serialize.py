"""Serialization helpers for temporal linking outputs."""

from __future__ import annotations

import copy
from dataclasses import asdict
import os
from typing import Any

from .config import TemporalLinkingConfig
from .types import Assignment, FrameDetections, RelinkManifest, Track

SCHEMA_VERSION_TRACKS = "temporal_linking_v1"
SCHEMA_VERSION_MANIFEST = "temporal_linking_manifest_v1"
SCHEMA_VERSION_RELINK_MANIFEST = "temporal_linking_relink_manifest_v1"


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


def _canonical_track_id(track_id: int, merge_map: dict[int, int]) -> int:
    seen: set[int] = set()
    current = int(track_id)
    while current in merge_map and current not in seen:
        seen.add(current)
        nxt = int(merge_map[current])
        if nxt == current:
            break
        current = nxt
    return current


def apply_merges_to_tracks_payload(
    tracks: list[dict[str, Any]],
    merge_map: dict[int, int],
    cfg: TemporalLinkingConfig,
) -> list[dict[str, Any]]:
    if not merge_map:
        return copy.deepcopy(tracks)

    tracks_by_id = {int(track["track_id"]): copy.deepcopy(track) for track in tracks}

    groups: dict[int, list[int]] = {}
    for track_id in tracks_by_id:
        canonical_id = _canonical_track_id(track_id, merge_map)
        if canonical_id not in tracks_by_id:
            canonical_id = track_id
        groups.setdefault(canonical_id, []).append(track_id)

    merged_tracks: list[dict[str, Any]] = []
    for canonical_id, members in groups.items():
        members_sorted = sorted(
            set(members),
            key=lambda track_id: (int(tracks_by_id[track_id].get("start_frame", 0)), int(track_id)),
        )
        canonical_payload = copy.deepcopy(tracks_by_id[canonical_id])

        if len(members_sorted) == 1:
            merged_tracks.append(canonical_payload)
            continue

        observations: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        total_misses = 0
        max_miss_streak = 0
        start_frame = int(canonical_payload.get("start_frame", 0))
        end_frame = int(canonical_payload.get("end_frame", 0))

        for member_id in members_sorted:
            payload = tracks_by_id[member_id]
            observations.extend(copy.deepcopy(payload.get("observations", [])))
            events.extend(copy.deepcopy(payload.get("events", [])))
            total_misses += int(payload.get("total_misses", 0))
            max_miss_streak = max(max_miss_streak, int(payload.get("max_miss_streak", 0)))
            start_frame = min(start_frame, int(payload.get("start_frame", start_frame)))
            end_frame = max(end_frame, int(payload.get("end_frame", end_frame)))

        observations.sort(key=lambda item: (int(item.get("frame_num", 0)), int(item.get("det_index", -1))))
        events.sort(key=lambda item: (int(item.get("frame_num", 0)), str(item.get("type", ""))))

        sim_values = [float(obs["visual_similarity"]) for obs in observations if obs.get("visual_similarity") is not None]
        avg_visual_similarity: float | None = None
        if sim_values:
            avg_visual_similarity = float(sum(sim_values) / float(len(sim_values)))

        hits = len(observations)

        canonical_payload["status"] = "closed"
        canonical_payload["start_frame"] = int(start_frame)
        canonical_payload["end_frame"] = int(end_frame)
        canonical_payload["hits"] = int(hits)
        canonical_payload["total_misses"] = int(total_misses)
        canonical_payload["max_miss_streak"] = int(max_miss_streak)
        canonical_payload["avg_visual_similarity"] = avg_visual_similarity
        canonical_payload["valid_track"] = bool(hits >= cfg.min_track_length)
        canonical_payload["events"] = events
        canonical_payload["observations"] = observations
        canonical_payload["relinked_from"] = [int(track_id) for track_id in members_sorted if track_id != canonical_id]
        merged_tracks.append(canonical_payload)

    merged_tracks.sort(key=lambda track: int(track.get("track_id", 0)))
    return merged_tracks


def remap_linked_frames_track_ids(
    linked_frames: list[dict[str, Any]],
    merge_map: dict[int, int],
) -> list[dict[str, Any]]:
    if not merge_map:
        return copy.deepcopy(linked_frames)

    remapped = copy.deepcopy(linked_frames)
    for frame in remapped:
        for det in frame.get("detections", []):
            temporal_link = det.get("temporal_link")
            if not isinstance(temporal_link, dict):
                continue
            raw_track_id = temporal_link.get("track_id")
            if not isinstance(raw_track_id, int):
                continue
            canonical_id = _canonical_track_id(raw_track_id, merge_map)
            temporal_link["track_id"] = int(canonical_id)
    return remapped


def serialize_tracks(
    closed_tracks: list[Track],
    cfg: TemporalLinkingConfig,
    merge_map: dict[int, int] | None = None,
) -> dict[str, Any]:
    tracks = [_track_payload(track, cfg) for track in closed_tracks]
    if merge_map:
        tracks = apply_merges_to_tracks_payload(tracks=tracks, merge_map=merge_map, cfg=cfg)

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


def build_relink_manifest(
    *,
    cfg: TemporalLinkingConfig,
    relink_result: dict[str, Any],
    merge_map: dict[int, int],
) -> dict[str, Any]:
    raw_stats = relink_result.get("stats", {})
    if not isinstance(raw_stats, dict):
        raw_stats = {}

    stats = {str(key): int(value) for key, value in raw_stats.items()}
    accepted_edges = relink_result.get("accepted_edges", [])
    if not isinstance(accepted_edges, list):
        accepted_edges = []

    relink_cfg = {
        "relink_threshold": float(cfg.relink_threshold),
        "relink_max_gap_frames": int(cfg.relink_max_gap_frames),
        "relink_min_track_hits": int(cfg.relink_min_track_hits),
        "relink_max_pixels_per_frame": float(cfg.relink_max_pixels_per_frame),
        "relink_fallback_threshold": float(cfg.relink_fallback_threshold),
    }
    manifest = RelinkManifest(
        schema_version=SCHEMA_VERSION_RELINK_MANIFEST,
        config=relink_cfg,
        stats=stats,
        accepted_edges=accepted_edges,
        merge_map={str(track_id): int(canonical_id) for track_id, canonical_id in sorted(merge_map.items())},
    )
    return asdict(manifest)


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
