"""Trace proof reference generation and rendering."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from typing import Any, Iterable

import cv2
import numpy as np

from common.io import load_json

from .config import TemporalLinkingConfig
from .serialize import SCHEMA_VERSION_TRACE_REFERENCES, canonical_track_id, serialize_trace_references
from .types import (
    TraceFrameGroup,
    TraceFrameReference,
    TraceProofItem,
    TraceReferencesPayload,
    Track,
    serialize_observation,
)

_MAX_GROUP_OBSERVATIONS = 3
_TILE_WIDTH = 240
_TILE_HEIGHT = 160
_HEADER_HEIGHT = 28
_DIVIDER_WIDTH = 120
_BORDER = 8
_PALETTE_HUE_ORDER = [0, 11, 5, 17, 2, 14, 8, 20, 4, 16, 10, 22, 1, 13, 7, 19, 3, 15, 9, 21, 6, 18, 12, 23]


def _config_hash(cfg: TemporalLinkingConfig) -> str:
    payload = json.dumps(cfg.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _track_observations(track: Track) -> list[dict[str, Any]]:
    return [serialize_observation(observation, default_fragment_track_id=track.track_id) for observation in track.observations]


def _frame_refs(
    observations: Iterable[dict[str, Any]],
    *,
    role: str,
    fragment_track_id: int,
    canonical_track_id_value: int,
) -> list[TraceFrameReference]:
    refs: list[TraceFrameReference] = []
    for observation in observations:
        refs.append(
            TraceFrameReference(
                frame_num=int(observation["frame_num"]),
                det_index=int(observation["det_index"]),
                bbox=[float(v) for v in observation.get("bbox", [])],
                fragment_track_id=int(observation.get("fragment_track_id", fragment_track_id)),
                canonical_track_id=int(canonical_track_id_value),
                role=role,
            )
        )
    return refs


def _build_relink_items(
    *,
    closed_tracks: list[Track],
    accepted_edges: list[dict[str, Any]],
    merge_map: dict[int, int],
) -> list[TraceProofItem]:
    tracks_by_id = {int(track.track_id): track for track in closed_tracks}
    items: list[TraceProofItem] = []

    for edge in accepted_edges:
        predecessor_id = int(edge["predecessor_id"])
        successor_id = int(edge["successor_id"])
        predecessor = tracks_by_id.get(predecessor_id)
        successor = tracks_by_id.get(successor_id)
        if predecessor is None or successor is None:
            continue

        canonical_id_value = canonical_track_id(predecessor_id, merge_map)
        pred_observations = _track_observations(predecessor)
        succ_observations = _track_observations(successor)
        pred_tail = pred_observations[-_MAX_GROUP_OBSERVATIONS:]
        succ_head = succ_observations[:_MAX_GROUP_OBSERVATIONS]

        items.append(
            TraceProofItem(
                kind="relink",
                proof_id=f"relink_{predecessor_id}_{successor_id}",
                canonical_track_id=int(canonical_id_value),
                fragment_track_ids=[predecessor_id, successor_id],
                frame_groups=[
                    TraceFrameGroup(
                        name="pred_tail",
                        frames=_frame_refs(
                            pred_tail,
                            role="pred_tail",
                            fragment_track_id=predecessor_id,
                            canonical_track_id_value=canonical_id_value,
                        ),
                    ),
                    TraceFrameGroup(
                        name="succ_head",
                        frames=_frame_refs(
                            succ_head,
                            role="succ_head",
                            fragment_track_id=successor_id,
                            canonical_track_id_value=canonical_id_value,
                        ),
                    ),
                ],
                method=str(edge.get("method")) if edge.get("method") is not None else None,
                score=None if edge.get("score") is None else float(edge["score"]),
            )
        )
    return items


def _build_recovery_items(
    *,
    closed_tracks: list[Track],
    merge_map: dict[int, int],
) -> list[TraceProofItem]:
    items: list[TraceProofItem] = []
    for track in sorted(closed_tracks, key=lambda item: item.track_id):
        canonical_id_value = canonical_track_id(track.track_id, merge_map)
        observations = _track_observations(track)
        pending_lost_frame: int | None = None

        for event in track.events:
            event_type = str(event.get("type", ""))
            if event_type == "lost":
                pending_lost_frame = int(event["frame_num"])
                continue
            if event_type != "recovered" or pending_lost_frame is None:
                continue

            recovered_frame = int(event["frame_num"])
            before_gap_all = [obs for obs in observations if int(obs["frame_num"]) < pending_lost_frame]
            after_gap_all = [obs for obs in observations if int(obs["frame_num"]) >= recovered_frame]
            before_gap = before_gap_all[-_MAX_GROUP_OBSERVATIONS:]
            after_gap = after_gap_all[:_MAX_GROUP_OBSERVATIONS]
            if not before_gap and not after_gap:
                pending_lost_frame = None
                continue

            gap_frames: int | None = None
            if before_gap and after_gap:
                gap_frames = max(0, int(after_gap[0]["frame_num"]) - int(before_gap[-1]["frame_num"]) - 1)

            items.append(
                TraceProofItem(
                    kind="recovery",
                    proof_id=f"recovery_{track.track_id}_{recovered_frame}",
                    canonical_track_id=int(canonical_id_value),
                    fragment_track_ids=[int(track.track_id)],
                    frame_groups=[
                        TraceFrameGroup(
                            name="before_gap",
                            frames=_frame_refs(
                                before_gap,
                                role="before_gap",
                                fragment_track_id=track.track_id,
                                canonical_track_id_value=canonical_id_value,
                            ),
                        ),
                        TraceFrameGroup(
                            name="after_gap",
                            frames=_frame_refs(
                                after_gap,
                                role="after_gap",
                                fragment_track_id=track.track_id,
                                canonical_track_id_value=canonical_id_value,
                            ),
                        ),
                    ],
                    gap_frames=gap_frames,
                )
            )
            pending_lost_frame = None

    return items


def build_trace_references(
    *,
    closed_tracks: list[Track],
    cfg: TemporalLinkingConfig,
    enriched_json_path: str,
    relink_result: dict[str, Any],
    merge_map: dict[int, int],
) -> dict[str, Any]:
    accepted_edges = relink_result.get("accepted_edges", [])
    if not isinstance(accepted_edges, list):
        accepted_edges = []

    items = _build_relink_items(
        closed_tracks=closed_tracks,
        accepted_edges=accepted_edges,
        merge_map=merge_map,
    )
    items.extend(_build_recovery_items(closed_tracks=closed_tracks, merge_map=merge_map))

    payload = TraceReferencesPayload(
        schema_version=SCHEMA_VERSION_TRACE_REFERENCES,
        generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        config_hash_sha256=_config_hash(cfg),
        input_enriched_json=os.path.basename(enriched_json_path),
        items=items,
    )
    return serialize_trace_references(payload)


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


_PROOF_PALETTE = _build_palette()


def proof_color_index(canonical_track_id: int) -> int:
    digest = hashlib.sha256(f"track:{canonical_track_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:2], byteorder="big", signed=False)
    return int(seed % len(_PROOF_PALETTE))


def proof_color(canonical_track_id: int) -> tuple[int, int, int]:
    return _PROOF_PALETTE[proof_color_index(canonical_track_id)]


def _load_frames(video_path: str, frame_nums: set[int]) -> dict[int, np.ndarray]:
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


def _annotate_frame(frame: np.ndarray, frame_ref: dict[str, Any], color: tuple[int, int, int]) -> np.ndarray:
    annotated = frame.copy()
    bbox = [int(round(float(v))) for v in frame_ref.get("bbox", [0, 0, 0, 0])]
    x1, y1, x2, y2 = bbox
    height, width = annotated.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    label = f"f{int(frame_ref['frame_num'])} d{int(frame_ref['det_index'])}"
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


def _group_header(label: str, width: int) -> np.ndarray:
    header = np.full((_HEADER_HEIGHT, width, 3), 232, dtype=np.uint8)
    cv2.putText(header, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return header


def _stack_group(
    group: dict[str, Any],
    frames_by_num: dict[int, np.ndarray],
    *,
    color: tuple[int, int, int],
) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for frame_ref in group.get("frames", []):
        frame_num = int(frame_ref["frame_num"])
        annotated = _annotate_frame(frames_by_num[frame_num], frame_ref, color)
        tiles.append(_resize_and_pad(annotated))

    if not tiles:
        panel_width = _TILE_WIDTH
        body = np.full((_TILE_HEIGHT, panel_width, 3), 250, dtype=np.uint8)
        cv2.putText(body, "No frames", (65, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1, cv2.LINE_AA)
    else:
        body = np.hstack(tiles)
        panel_width = body.shape[1]
    return np.vstack([_group_header(str(group.get("name", "")), panel_width), body])


def _divider_panel(item: dict[str, Any], height: int) -> np.ndarray:
    divider = np.full((height, _DIVIDER_WIDTH, 3), 215, dtype=np.uint8)
    if str(item.get("kind")) == "relink":
        fragment_track_ids = item.get("fragment_track_ids", [])
        if not isinstance(fragment_track_ids, list):
            fragment_track_ids = []
        pred_id = fragment_track_ids[0] if fragment_track_ids else "?"
        succ_id = fragment_track_ids[-1] if fragment_track_ids else "?"
        lines = [
            "RELINK",
            f"{pred_id} -> {succ_id}",
        ]
        if item.get("method") is not None:
            score = item.get("score")
            score_text = "n/a" if score is None else f"{float(score):.3f}"
            lines.append(f"{item['method']} {score_text}")
    else:
        gap_frames = item.get("gap_frames")
        gap_text = "gap n/a" if gap_frames is None else f"gap {int(gap_frames)}f"
        lines = ["RECOVERY", gap_text]

    for index, line in enumerate(lines):
        cv2.putText(
            divider,
            str(line),
            (12, 38 + (index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
    return divider


def _render_item(item: dict[str, Any], frames_by_num: dict[int, np.ndarray], output_path: str) -> None:
    color = proof_color(int(item["canonical_track_id"]))
    groups = item.get("frame_groups", [])
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError(f"Trace proof item {item.get('proof_id')} missing required frame_groups")

    left = _stack_group(groups[0], frames_by_num, color=color)
    right = _stack_group(groups[1], frames_by_num, color=color)
    panel_height = max(left.shape[0], right.shape[0])

    if left.shape[0] != panel_height:
        pad = np.full((panel_height - left.shape[0], left.shape[1], 3), 245, dtype=np.uint8)
        left = np.vstack([left, pad])
    if right.shape[0] != panel_height:
        pad = np.full((panel_height - right.shape[0], right.shape[1], 3), 245, dtype=np.uint8)
        right = np.vstack([right, pad])

    divider = _divider_panel(item, panel_height)
    body = np.hstack([left, divider, right])
    canvas = np.full((body.shape[0] + (_BORDER * 2), body.shape[1] + (_BORDER * 2), 3), 255, dtype=np.uint8)
    canvas[_BORDER : _BORDER + body.shape[0], _BORDER : _BORDER + body.shape[1]] = body
    if not cv2.imwrite(output_path, canvas):
        raise RuntimeError(f"Failed to write proof sheet {output_path}")


def render_trace_proofs(
    *,
    trace_references_payload: dict[str, Any],
    video_path: str,
    output_dir: str,
) -> int:
    items = trace_references_payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("trace_references_payload.items must be a list")

    os.makedirs(output_dir, exist_ok=True)
    if not items:
        return 0

    frame_nums: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        for group in item.get("frame_groups", []):
            if not isinstance(group, dict):
                continue
            for frame_ref in group.get("frames", []):
                if not isinstance(frame_ref, dict):
                    continue
                frame_nums.add(int(frame_ref["frame_num"]))

    frames_by_num = _load_frames(video_path, frame_nums)
    rendered = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        proof_id = str(item.get("proof_id", "")).strip()
        if not proof_id:
            continue
        output_path = os.path.join(output_dir, f"{proof_id}.jpg")
        _render_item(item, frames_by_num, output_path)
        rendered += 1
    return rendered
