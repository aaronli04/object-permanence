"""Offline temporal linking pipeline."""

from __future__ import annotations

from .assignment import assign_frame
from .config import TemporalLinkingConfig
from .io import build_output_paths, load_enriched_frames, write_linking_outputs
from .serialize import (
    match_link_meta,
    new_track_link_meta,
    serialize_linked_frame,
    serialize_manifest,
    serialize_tracks,
)
from .tracker import TrackManager
from .types import FrameDetections, TemporalLinkingResult


def link_video_frames(
    frames: list[FrameDetections],
    cfg: TemporalLinkingConfig,
    *,
    enriched_json_path: str | None = None,
) -> TemporalLinkingResult:
    manager = TrackManager(cfg)
    linked_frames: list[dict[str, object]] = []

    for frame in sorted(frames, key=lambda item: item.frame_num):
        candidates = manager.candidates()
        assignments = assign_frame(candidates, frame.detections, cfg)

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        per_det_link_meta: dict[int, dict[str, object]] = {}

        for assignment in assignments:
            if assignment.track_id in matched_track_ids or assignment.det_index in matched_det_indices:
                continue

            track = manager.get(assignment.track_id)
            age_since_seen = int(track.miss_streak)
            det = frame.detections[assignment.det_index]

            updated_track = manager.apply_match(track.track_id, det, assignment, frame.frame_num)

            matched_track_ids.add(updated_track.track_id)
            matched_det_indices.add(det.det_index)
            per_det_link_meta[det.det_index] = match_link_meta(
                assignment=assignment,
                track_id=updated_track.track_id,
                track_status=updated_track.status.value,
                age_since_seen=age_since_seen,
            )

        manager.mark_unmatched(candidates, matched_track_ids, frame.frame_num)

        for det in frame.detections:
            if det.det_index in matched_det_indices:
                continue
            new_track = manager.spawn(det, frame.frame_num)
            per_det_link_meta[det.det_index] = new_track_link_meta(
                track_id=new_track.track_id,
                track_status=new_track.status.value,
            )

        linked_frames.append(serialize_linked_frame(frame, per_det_link_meta))

    manager.close_remaining()

    effective_enriched_json = enriched_json_path or "enriched_detections.json"
    closed_tracks = manager.closed_tracks()
    tracks_payload = serialize_tracks(closed_tracks, cfg)
    manifest_payload = serialize_manifest(
        enriched_json_path=effective_enriched_json,
        linked_frames=linked_frames,
        tracks_payload=tracks_payload,
        cfg=cfg,
    )

    return TemporalLinkingResult(
        linked_frames=linked_frames,
        tracks_payload=tracks_payload,
        manifest_payload=manifest_payload,
    )


def run_temporal_linking(
    *,
    enriched_json_path: str,
    output_dir: str,
    config: TemporalLinkingConfig,
) -> dict[str, str]:
    frames = load_enriched_frames(enriched_json_path)
    result = link_video_frames(frames, config, enriched_json_path=enriched_json_path)

    artifacts = build_output_paths(output_dir)
    write_linking_outputs(
        artifacts=artifacts,
        linked_frames=result.linked_frames,
        tracks_payload=result.tracks_payload,
        manifest_payload=result.manifest_payload,
    )

    return {
        "linked_detections": artifacts.linked_detections_path,
        "tracks": artifacts.tracks_path,
        "linking_manifest": artifacts.manifest_path,
    }
