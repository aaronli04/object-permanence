"""Poster-friendly track documentation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
from typing import Any

import cv2

from common.io import load_json, write_json

from .trace_summary import (
    annotate_trace_frame,
    collect_trace_summary_frame_nums,
    load_video_frames,
    render_trace_summary,
    summary_color,
)


@dataclass(frozen=True)
class TrackDocOutputs:
    scenario: str
    output_dir: str
    index_path: str
    num_tracks: int
    num_summary_images: int
    num_frame_images: int


def _write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _sorted_tracks(tracks_payload: dict[str, Any]) -> list[dict[str, Any]]:
    tracks = tracks_payload.get("tracks", [])
    if not isinstance(tracks, list):
        return []
    filtered = [track for track in tracks if isinstance(track, dict)]
    filtered.sort(key=lambda item: int(item.get("track_id", 0)))
    return filtered


def _trace_items_by_track(trace_summary_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    items = trace_summary_payload.get("items", [])
    if not isinstance(items, list):
        return {}
    return {
        int(item.get("track_id", -1)): item
        for item in items
        if isinstance(item, dict) and int(item.get("track_id", -1)) >= 0
    }


def _family_track_ids(track_payload: dict[str, Any]) -> set[int]:
    family = {int(track_payload.get("track_id", 0))}
    relinked_from = track_payload.get("relinked_from", [])
    if isinstance(relinked_from, list):
        family.update(int(track_id) for track_id in relinked_from if isinstance(track_id, int))
    observations = track_payload.get("observations", [])
    if isinstance(observations, list):
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            fragment_track_id = obs.get("fragment_track_id")
            if isinstance(fragment_track_id, int):
                family.add(int(fragment_track_id))
    return family


def _observed_classes(track_payload: dict[str, Any]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int | None, str], dict[str, Any]] = {}
    observations = track_payload.get("observations", [])
    if not isinstance(observations, list):
        return []

    for obs in observations:
        if not isinstance(obs, dict):
            continue
        raw_class_id = obs.get("class_id")
        class_id = int(raw_class_id) if isinstance(raw_class_id, int) else None
        class_name = str(obs.get("class_name", "")) if obs.get("class_name") is not None else ""
        key = (class_id, class_name)
        bucket = by_key.setdefault(
            key,
            {
                "class_id": class_id,
                "class_name": class_name,
                "detections": 0,
                "confidence_sum": 0.0,
            },
        )
        bucket["detections"] = int(bucket["detections"]) + 1
        bucket["confidence_sum"] = float(bucket["confidence_sum"]) + float(obs.get("confidence", 0.0) or 0.0)

    rows = list(by_key.values())
    rows.sort(
        key=lambda item: (
            -float(item["confidence_sum"]),
            -int(item["detections"]),
            str(item["class_name"]),
        )
    )
    return rows


def _accepted_edges_for_track(
    track_payload: dict[str, Any],
    relink_manifest_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(relink_manifest_payload, dict):
        return []
    accepted_edges = relink_manifest_payload.get("accepted_edges", [])
    if not isinstance(accepted_edges, list):
        return []

    family_ids = _family_track_ids(track_payload)
    rows: list[dict[str, Any]] = []
    for edge in accepted_edges:
        if not isinstance(edge, dict):
            continue
        predecessor_id = edge.get("predecessor_id")
        successor_id = edge.get("successor_id")
        if not isinstance(predecessor_id, int) or not isinstance(successor_id, int):
            continue
        if predecessor_id not in family_ids and successor_id not in family_ids:
            continue
        rows.append(
            {
                "predecessor_id": int(predecessor_id),
                "successor_id": int(successor_id),
                "score": float(edge.get("score", 0.0) or 0.0),
                "method": str(edge.get("method", "")),
            }
        )

    rows.sort(key=lambda item: (item["predecessor_id"], item["successor_id"]))
    return rows


def _events(track_payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = track_payload.get("events", [])
    if not isinstance(events, list):
        return []
    rows = [event for event in events if isinstance(event, dict)]
    rows.sort(key=lambda item: (int(item.get("frame_num", 0)), str(item.get("type", ""))))
    return rows


def _track_span(track_payload: dict[str, Any]) -> int:
    start_frame = int(track_payload.get("start_frame", 0))
    end_frame = int(track_payload.get("end_frame", 0))
    return max(0, end_frame - start_frame + 1)


def _copy_summary_image(track_id: int, trace_summary_dir: str | None, track_dir: str) -> str | None:
    if not trace_summary_dir:
        return None
    source_path = os.path.join(trace_summary_dir, f"track_{track_id}.jpg")
    if not os.path.exists(source_path):
        return None
    target_path = os.path.join(track_dir, "summary.jpg")
    shutil.copyfile(source_path, target_path)
    return os.path.basename(target_path)


def _render_frame_images(
    *,
    track_id: int,
    item: dict[str, Any] | None,
    track_dir: str,
    frames_by_num: dict[int, Any],
) -> list[dict[str, Any]]:
    if item is None or not frames_by_num:
        return []

    refs = item.get("frames", [])
    if not isinstance(refs, list):
        return []

    role_dir = os.path.join(track_dir, "frames")
    os.makedirs(role_dir, exist_ok=True)
    color = summary_color(track_id)

    artifacts: list[dict[str, Any]] = []
    for frame_ref in refs:
        if not isinstance(frame_ref, dict):
            continue
        frame_num = int(frame_ref.get("frame_num", 0))
        if frame_num not in frames_by_num:
            continue
        role = str(frame_ref.get("role", "frame")).strip() or "frame"
        filename = f"{role}_f{frame_num:04d}.jpg"
        rel_path = os.path.join("frames", filename)
        output_path = os.path.join(track_dir, rel_path)
        annotated = annotate_trace_frame(frames_by_num[frame_num], frame_ref, color)
        if not cv2.imwrite(output_path, annotated):
            raise RuntimeError(f"Failed to write track frame image {output_path}")
        artifacts.append(
            {
                "role": role,
                "frame_num": frame_num,
                "fragment_track_id": int(frame_ref.get("fragment_track_id", track_id)),
                "path": rel_path,
            }
        )
    return artifacts


def _build_track_doc_payload(
    *,
    scenario: str,
    track_payload: dict[str, Any],
    trace_item: dict[str, Any] | None,
    relink_manifest_payload: dict[str, Any] | None,
    summary_image: str | None,
    frame_images: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_classes = _observed_classes(track_payload)
    accepted_edges = _accepted_edges_for_track(track_payload, relink_manifest_payload)
    fragment_track_ids = sorted(_family_track_ids(track_payload))
    track_id = int(track_payload.get("track_id", 0))

    return {
        "scenario": scenario,
        "track": track_payload,
        "trace_summary": trace_item,
        "derived": {
            "track_id": track_id,
            "span_frames": _track_span(track_payload),
            "fragment_track_ids": fragment_track_ids,
            "observed_classes": observed_classes,
            "accepted_relink_edges": accepted_edges,
        },
        "artifacts": {
            "summary_image": summary_image,
            "frame_images": frame_images,
        },
    }


def _format_yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _track_readme(doc_payload: dict[str, Any]) -> str:
    scenario = str(doc_payload.get("scenario", ""))
    track = doc_payload.get("track", {})
    if not isinstance(track, dict):
        track = {}
    derived = doc_payload.get("derived", {})
    if not isinstance(derived, dict):
        derived = {}
    artifacts = doc_payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    trace_item = doc_payload.get("trace_summary", {})
    if not isinstance(trace_item, dict):
        trace_item = {}

    track_id = int(track.get("track_id", 0))
    relinked_from = track.get("relinked_from", [])
    relinked_from_text = ", ".join(str(track_id) for track_id in relinked_from) if isinstance(relinked_from, list) and relinked_from else "none"
    fragment_track_ids = derived.get("fragment_track_ids", [])
    fragment_ids_text = ", ".join(str(track_id) for track_id in fragment_track_ids) if isinstance(fragment_track_ids, list) and fragment_track_ids else "none"
    avg_similarity = track.get("avg_visual_similarity")
    avg_similarity_text = f"{float(avg_similarity):.4f}" if avg_similarity is not None else "n/a"

    lines = [f"# {scenario} / track_{track_id}", ""]
    summary_image = artifacts.get("summary_image")
    if isinstance(summary_image, str) and summary_image:
        lines.extend([f"![track summary]({summary_image})", ""])

    lines.extend(
        [
            "## Summary",
            f"- class: {track.get('class_name', '')} ({track.get('class_id', -1)})",
            f"- frames: {track.get('start_frame', 0)}-{track.get('end_frame', 0)} ({derived.get('span_frames', 0)} frame span)",
            f"- hits: {track.get('hits', 0)}",
            f"- valid track: {_format_yes_no(bool(track.get('valid_track')))}",
            f"- had recovery: {_format_yes_no(bool(trace_item.get('had_recovery')))}",
            f"- relinked from: {relinked_from_text}",
            f"- fragment track ids: {fragment_ids_text}",
            f"- avg visual similarity: {avg_similarity_text}",
            f"- total misses: {track.get('total_misses', 0)}",
            f"- max miss streak: {track.get('max_miss_streak', 0)}",
            "",
            "## Label Mix",
        ]
    )

    observed_classes = derived.get("observed_classes", [])
    if isinstance(observed_classes, list) and observed_classes:
        for item in observed_classes:
            if not isinstance(item, dict):
                continue
            lines.append(
                "- "
                f"{item.get('class_name', '')} ({item.get('class_id', 'n/a')}): "
                f"{item.get('detections', 0)} detections, confidence sum {float(item.get('confidence_sum', 0.0)):.3f}"
            )
    else:
        lines.append("- no per-observation class metadata")

    lines.extend(["", "## Relink Edges"])
    accepted_edges = derived.get("accepted_relink_edges", [])
    if isinstance(accepted_edges, list) and accepted_edges:
        for edge in accepted_edges:
            if not isinstance(edge, dict):
                continue
            lines.append(
                "- "
                f"{edge.get('predecessor_id', '?')} -> {edge.get('successor_id', '?')} "
                f"via {edge.get('method', '')} (score {float(edge.get('score', 0.0)):.4f})"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Event Timeline"])
    events = _events(track)
    if events:
        for event in events:
            lines.append(f"- frame {int(event.get('frame_num', 0))}: {event.get('type', '')}")
    else:
        lines.append("- none")

    lines.extend(["", "## Key Frames"])
    frame_images = artifacts.get("frame_images", [])
    if isinstance(frame_images, list) and frame_images:
        for frame_image in frame_images:
            if not isinstance(frame_image, dict):
                continue
            lines.append(
                "- "
                f"{frame_image.get('role', 'frame')}: frame {frame_image.get('frame_num', 0)}, "
                f"fragment {frame_image.get('fragment_track_id', track_id)}, "
                f"[image]({frame_image.get('path', '')})"
            )
    else:
        lines.append("- no frame images available")

    lines.extend(["", "[track metadata](track.json)", ""])
    return "\n".join(lines)


def _scenario_index_readme(
    *,
    scenario: str,
    track_docs: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Track docs for {scenario}",
        "",
        "Per-track folders contain a rendered summary image, annotated key frames, and machine-readable metadata.",
        "",
        "## Tracks",
    ]

    if not track_docs:
        lines.extend(["", "- no tracks available", ""])
        return "\n".join(lines)

    for doc_payload in track_docs:
        track = doc_payload.get("track", {})
        if not isinstance(track, dict):
            continue
        derived = doc_payload.get("derived", {})
        if not isinstance(derived, dict):
            derived = {}
        track_id = int(track.get("track_id", 0))
        relinked_from = track.get("relinked_from", [])
        relinked_from_text = (
            ", ".join(str(track_id) for track_id in relinked_from)
            if isinstance(relinked_from, list) and relinked_from
            else "none"
        )
        lines.append(
            "- "
            f"[track_{track_id}](track_{track_id}/README.md): "
            f"{track.get('class_name', '')}, "
            f"hits {track.get('hits', 0)}, "
            f"frames {track.get('start_frame', 0)}-{track.get('end_frame', 0)} "
            f"({derived.get('span_frames', 0)} span), "
            f"valid {_format_yes_no(bool(track.get('valid_track')))}, "
            f"recovered {_format_yes_no(bool(doc_payload.get('trace_summary', {}) and doc_payload['trace_summary'].get('had_recovery')))}, "
            f"relinked from {relinked_from_text}"
        )

    lines.append("")
    return "\n".join(lines)


def build_track_docs(
    *,
    scenario: str,
    tracks_payload: dict[str, Any],
    trace_summary_payload: dict[str, Any],
    output_dir: str,
    trace_summary_dir: str | None = None,
    relink_manifest_payload: dict[str, Any] | None = None,
    video_path: str | None = None,
    render_trace_summary_images: bool = True,
) -> TrackDocOutputs:
    os.makedirs(output_dir, exist_ok=True)

    if trace_summary_dir and render_trace_summary_images and video_path:
        render_trace_summary(
            trace_summary_payload=trace_summary_payload,
            video_path=video_path,
            output_dir=trace_summary_dir,
        )

    tracks = _sorted_tracks(tracks_payload)
    items_by_track = _trace_items_by_track(trace_summary_payload)
    frames_by_num = (
        load_video_frames(video_path, collect_trace_summary_frame_nums(trace_summary_payload))
        if video_path
        else {}
    )

    num_summary_images = 0
    num_frame_images = 0
    doc_payloads: list[dict[str, Any]] = []

    for track_payload in tracks:
        track_id = int(track_payload.get("track_id", 0))
        track_dir = os.path.join(output_dir, f"track_{track_id}")
        os.makedirs(track_dir, exist_ok=True)

        trace_item = items_by_track.get(track_id)
        summary_image = _copy_summary_image(track_id, trace_summary_dir, track_dir)
        if summary_image is not None:
            num_summary_images += 1
        frame_images = _render_frame_images(
            track_id=track_id,
            item=trace_item,
            track_dir=track_dir,
            frames_by_num=frames_by_num,
        )
        num_frame_images += len(frame_images)

        doc_payload = _build_track_doc_payload(
            scenario=scenario,
            track_payload=track_payload,
            trace_item=trace_item,
            relink_manifest_payload=relink_manifest_payload,
            summary_image=summary_image,
            frame_images=frame_images,
        )
        write_json(os.path.join(track_dir, "track.json"), doc_payload)
        _write_text(os.path.join(track_dir, "README.md"), _track_readme(doc_payload))
        doc_payloads.append(doc_payload)

    index_path = os.path.join(output_dir, "README.md")
    _write_text(index_path, _scenario_index_readme(scenario=scenario, track_docs=doc_payloads))

    return TrackDocOutputs(
        scenario=scenario,
        output_dir=output_dir,
        index_path=index_path,
        num_tracks=len(doc_payloads),
        num_summary_images=num_summary_images,
        num_frame_images=num_frame_images,
    )


def build_track_docs_from_paths(
    *,
    scenario: str,
    tracks_path: str,
    trace_summary_path: str,
    output_dir: str,
    trace_summary_dir: str | None = None,
    relink_manifest_path: str | None = None,
    video_path: str | None = None,
    render_trace_summary_images: bool = True,
) -> TrackDocOutputs:
    tracks_payload = load_json(tracks_path)
    trace_summary_payload = load_json(trace_summary_path)
    relink_manifest_payload = load_json(relink_manifest_path) if relink_manifest_path and os.path.exists(relink_manifest_path) else None
    return build_track_docs(
        scenario=scenario,
        tracks_payload=tracks_payload,
        trace_summary_payload=trace_summary_payload,
        output_dir=output_dir,
        trace_summary_dir=trace_summary_dir,
        relink_manifest_payload=relink_manifest_payload,
        video_path=video_path,
        render_trace_summary_images=render_trace_summary_images,
    )
