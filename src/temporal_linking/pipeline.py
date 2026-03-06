"""Offline temporal linking pipeline."""

from __future__ import annotations

import os

from .assignment import assign_frame
from .config import TemporalLinkingConfig
from .io import build_output_paths, load_enriched_frames, write_json, write_linking_outputs
from .relink import run_relink
from .serialize import (
    build_relink_manifest,
    match_link_meta,
    new_track_link_meta,
    remap_linked_frames_track_ids,
    serialize_linked_frame,
    serialize_manifest,
    serialize_tracks,
)
from .tracker import TrackManager
from .types import FrameDetections, TemporalLinkingOutputs, TemporalLinkingResult


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

    closed_tracks = manager.finalize()
    merge_map, relink_result = run_relink(closed_tracks, cfg)
    linked_frames = remap_linked_frames_track_ids(linked_frames, merge_map)
    effective_enriched_json = enriched_json_path or "enriched_detections.json"
    tracks_payload = serialize_tracks(closed_tracks, cfg, merge_map=merge_map)
    manifest_payload = serialize_manifest(
        enriched_json_path=effective_enriched_json,
        linked_frames=linked_frames,
        tracks_payload=tracks_payload,
        cfg=cfg,
    )
    relink_manifest_payload = build_relink_manifest(
        cfg=cfg,
        relink_result=relink_result,
        merge_map=merge_map,
    )

    return TemporalLinkingResult(
        linked_frames=linked_frames,
        tracks_payload=tracks_payload,
        manifest_payload=manifest_payload,
        relink_manifest_payload=relink_manifest_payload,
    )


def run_temporal_linking(
    *,
    enriched_json_path: str,
    output_dir: str,
    config: TemporalLinkingConfig,
) -> TemporalLinkingOutputs:
    frames = load_enriched_frames(enriched_json_path, activation_topk=config.activation_topk)
    result = link_video_frames(frames, config, enriched_json_path=enriched_json_path)

    artifacts = build_output_paths(output_dir)
    write_linking_outputs(
        artifacts=artifacts,
        linked_frames=result.linked_frames,
        tracks_payload=result.tracks_payload,
        manifest_payload=result.manifest_payload,
    )
    relink_manifest_path = os.path.join(output_dir, "relink_manifest.json")
    write_json(relink_manifest_path, result.relink_manifest_payload)

    return TemporalLinkingOutputs(
        linked_detections=artifacts.linked_detections_path,
        tracks=artifacts.tracks_path,
        linking_manifest=artifacts.manifest_path,
        relink_manifest=relink_manifest_path,
    )
