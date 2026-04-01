"""Final-track trace summary generation and rendering."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from typing import Any

import cv2
import numpy as np

from common.io import load_json

from .config import TemporalLinkingConfig
from .serialize import SCHEMA_VERSION_TRACE_SUMMARY, serialize_trace_summary
from .types import TraceFrameReference, TraceSummaryItem, TraceSummaryPayload

_TILE_WIDTH = 240
_TILE_HEIGHT = 160
_TOP_HEADER_HEIGHT = 48
_GROUP_HEADER_HEIGHT = 28
_BORDER = 8
_PALETTE_HUE_ORDER = [0, 11, 5, 17, 2, 14, 8, 20, 4, 16, 10, 22, 1, 13, 7, 19, 3, 15, 9, 21, 6, 18, 12, 23]


def _config_hash(cfg: TemporalLinkingConfig) -> str:
    payload = json.dumps(cfg.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_observations(track_payload: dict[str, Any]) -> list[dict[str, Any]]:
    observations = track_payload.get("observations", [])
    if not isinstance(observations, list):
        return []
    ordered = [observation for observation in observations if isinstance(observation, dict)]
    ordered.sort(key=lambda item: (int(item.get("frame_num", 0)), int(item.get("det_index", -1))))
    return ordered


def _select_summary_frames(track_payload: dict[str, Any]) -> list[TraceFrameReference]:
    observations = _ordered_observations(track_payload)
    if not observations:
        return []

    indices = [0]
    if len(observations) >= 3:
        indices.append(len(observations) // 2)
    if len(observations) >= 2:
        indices.append(len(observations) - 1)

    deduped_indices: list[int] = []
    for index in indices:
        if index not in deduped_indices:
            deduped_indices.append(index)

    roles = ["start", "end"] if len(deduped_indices) == 2 else ["start", "middle", "end"]
    if len(deduped_indices) == 1:
        roles = ["start"]

    refs: list[TraceFrameReference] = []
    track_id = int(track_payload.get("track_id", 0))
    for role, index in zip(roles, deduped_indices):
        observation = observations[index]
        refs.append(
            TraceFrameReference(
                frame_num=int(observation["frame_num"]),
                det_index=int(observation["det_index"]),
                bbox=[float(v) for v in observation.get("bbox", [])],
                fragment_track_id=int(observation.get("fragment_track_id", track_id)),
                canonical_track_id=track_id,
                role=role,
            )
        )
    return refs


def build_trace_summary(
    *,
    tracks_payload: dict[str, Any],
    cfg: TemporalLinkingConfig,
    enriched_json_path: str,
) -> dict[str, Any]:
    raw_tracks = tracks_payload.get("tracks", [])
    if not isinstance(raw_tracks, list):
        raw_tracks = []

    items: list[TraceSummaryItem] = []
    for track_payload in raw_tracks:
        if not isinstance(track_payload, dict):
            continue
        events = track_payload.get("events", [])
        if not isinstance(events, list):
            events = []
        relinked_from = track_payload.get("relinked_from", [])
        if not isinstance(relinked_from, list):
            relinked_from = []

        items.append(
            TraceSummaryItem(
                track_id=int(track_payload.get("track_id", 0)),
                class_id=int(track_payload.get("class_id", -1)),
                class_name=str(track_payload.get("class_name", "")),
                start_frame=int(track_payload.get("start_frame", 0)),
                end_frame=int(track_payload.get("end_frame", 0)),
                hits=int(track_payload.get("hits", 0)),
                valid_track=bool(track_payload.get("valid_track")),
                relinked_from=[int(track_id) for track_id in relinked_from if isinstance(track_id, int)],
                had_recovery=any(
                    isinstance(event, dict) and str(event.get("type", "")) == "recovered"
                    for event in events
                ),
                frames=_select_summary_frames(track_payload),
            )
        )

    payload = TraceSummaryPayload(
        schema_version=SCHEMA_VERSION_TRACE_SUMMARY,
        generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        config_hash_sha256=_config_hash(cfg),
        input_enriched_json=os.path.basename(enriched_json_path),
        items=items,
    )
    return serialize_trace_summary(payload)


def resolve_video_path(*, enriched_json_path: str, video_path_override: str | None = None) -> str | None:
    if video_path_override:
        return video_path_override if os.path.exists(video_path_override) else None

    projection_manifest_path = os.path.join(os.path.dirname(enriched_json_path), "projection_manifest.json")
    if not os.path.exists(projection_manifest_path):
        return None

    manifest = load_json(projection_manifest_path)
    if not isinstance(manifest, dict):
        return None
    video_path = manifest.get("input_video_path")
    if not isinstance(video_path, str) or not video_path:
        return None
    return video_path if os.path.exists(video_path) else None


def _build_palette() -> list[tuple[int, int, int]]:
    palette: list[tuple[int, int, int]] = []
    for hue_index in _PALETTE_HUE_ORDER:
        hsv = np.asarray([[[int(round((180.0 * float(hue_index)) / 24.0)), 210, 235]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette


_SUMMARY_PALETTE = _build_palette()


def summary_color_index(track_id: int) -> int:
    digest = hashlib.sha256(f"track:{track_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:2], byteorder="big", signed=False)
    return int(seed % len(_SUMMARY_PALETTE))


def summary_color(track_id: int) -> tuple[int, int, int]:
    return _SUMMARY_PALETTE[summary_color_index(track_id)]


def collect_trace_summary_frame_nums(trace_summary_payload: dict[str, Any]) -> set[int]:
    items = trace_summary_payload.get("items", [])
    if not isinstance(items, list):
        return set()

    frame_nums: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        for frame_ref in item.get("frames", []):
            if not isinstance(frame_ref, dict):
                continue
            frame_nums.add(int(frame_ref["frame_num"]))
    return frame_nums


def load_video_frames(video_path: str, frame_nums: set[int]) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    if not frame_nums:
        return frames

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        for frame_num in sorted(frame_nums):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_num))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not load frame {frame_num} from {video_path}")
            frames[frame_num] = frame
    finally:
        cap.release()
    return frames


def annotate_trace_frame(frame: np.ndarray, frame_ref: dict[str, Any], color: tuple[int, int, int]) -> np.ndarray:
    annotated = frame.copy()
    bbox = [int(round(float(v))) for v in frame_ref.get("bbox", [0, 0, 0, 0])]
    x1, y1, x2, y2 = bbox
    height, width = annotated.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    label = f"{frame_ref.get('role', '')}  f{int(frame_ref['frame_num'])}"
    cv2.rectangle(annotated, (0, 0), (min(width - 1, 140), 24), (20, 20, 20), thickness=-1)
    cv2.putText(annotated, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return annotated


def _resize_and_pad(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(float(_TILE_WIDTH) / float(max(width, 1)), float(_TILE_HEIGHT) / float(max(height, 1)))
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((_TILE_HEIGHT, _TILE_WIDTH, 3), 245, dtype=np.uint8)
    y_off = (_TILE_HEIGHT - resized.shape[0]) // 2
    x_off = (_TILE_WIDTH - resized.shape[1]) // 2
    canvas[y_off : y_off + resized.shape[0], x_off : x_off + resized.shape[1]] = resized
    return canvas


def _top_header(item: dict[str, Any], width: int, color: tuple[int, int, int]) -> np.ndarray:
    header = np.full((_TOP_HEADER_HEIGHT, width, 3), 235, dtype=np.uint8)
    title = f"track_{int(item['track_id'])}  {item.get('class_name', '')}".strip()
    meta = f"frames {int(item['start_frame'])}-{int(item['end_frame'])}  hits {int(item['hits'])}"
    flags: list[str] = []
    if bool(item.get("valid_track")):
        flags.append("valid")
    if bool(item.get("had_recovery")):
        flags.append("recovered")
    relinked_from = item.get("relinked_from", [])
    if isinstance(relinked_from, list) and relinked_from:
        flags.append("relinked")
    if flags:
        meta = f"{meta}  {' '.join(flags)}"
    cv2.putText(header, title, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(header, meta, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 55, 55), 1, cv2.LINE_AA)
    return header


def _frame_strip(item: dict[str, Any], frames_by_num: dict[int, np.ndarray], color: tuple[int, int, int]) -> np.ndarray:
    refs = item.get("frames", [])
    if not isinstance(refs, list) or not refs:
        body = np.full((_TILE_HEIGHT, _TILE_WIDTH, 3), 250, dtype=np.uint8)
        cv2.putText(body, "No frames", (65, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1, cv2.LINE_AA)
        return body

    tiles: list[np.ndarray] = []
    for frame_ref in refs:
        if not isinstance(frame_ref, dict):
            continue
        frame_num = int(frame_ref["frame_num"])
        annotated = annotate_trace_frame(frames_by_num[frame_num], frame_ref, color)
        tile = _resize_and_pad(annotated)
        label_band = np.full((_GROUP_HEADER_HEIGHT, _TILE_WIDTH, 3), 232, dtype=np.uint8)
        cv2.putText(
            label_band,
            str(frame_ref.get("role", "")),
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        tiles.append(np.vstack([label_band, tile]))

    if not tiles:
        body = np.full((_TILE_HEIGHT + _GROUP_HEADER_HEIGHT, _TILE_WIDTH, 3), 250, dtype=np.uint8)
        return body
    return np.hstack(tiles)


def _render_item(item: dict[str, Any], frames_by_num: dict[int, np.ndarray], output_path: str) -> None:
    color = summary_color(int(item["track_id"]))
    strip = _frame_strip(item, frames_by_num, color)
    header = _top_header(item, strip.shape[1], color)
    body = np.vstack([header, strip])
    canvas = np.full((body.shape[0] + (_BORDER * 2), body.shape[1] + (_BORDER * 2), 3), 255, dtype=np.uint8)
    canvas[_BORDER : _BORDER + body.shape[0], _BORDER : _BORDER + body.shape[1]] = body
    if not cv2.imwrite(output_path, canvas):
        raise RuntimeError(f"Failed to write trace summary {output_path}")


def render_trace_summary(
    *,
    trace_summary_payload: dict[str, Any],
    video_path: str,
    output_dir: str,
) -> int:
    items = trace_summary_payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("trace_summary_payload.items must be a list")

    os.makedirs(output_dir, exist_ok=True)
    if not items:
        return 0

    frames_by_num = load_video_frames(video_path, collect_trace_summary_frame_nums(trace_summary_payload))
    rendered = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        track_id = int(item.get("track_id", -1))
        if track_id < 0:
            continue
        output_path = os.path.join(output_dir, f"track_{track_id}.jpg")
        _render_item(item, frames_by_num, output_path)
        rendered += 1
    return rendered
