"""I/O utilities for temporal linking."""

from __future__ import annotations

import copy
import os
import sys
from typing import Any

import numpy as np

from common.io import load_json, write_json
from common.numeric import topk_l2_renorm
from common.warn_once import WarnOnce

from .types import Detection, FrameDetections, TemporalLinkArtifacts

_DINO_VECTOR_DIM = 384


_FRAME_DIMENSION_KEY_PAIRS: tuple[tuple[str, str], ...] = (
    ("frame_width", "frame_height"),
    ("width", "height"),
    ("image_width", "image_height"),
)


def _to_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0.0:
        return None
    return float(number)


def _extract_explicit_frame_dims(
    frame_item: dict[str, Any],
    det_item: dict[str, Any],
    activation_item: dict[str, Any],
) -> tuple[float | None, float | None]:
    for source in (frame_item, det_item, activation_item):
        for width_key, height_key in _FRAME_DIMENSION_KEY_PAIRS:
            width = _to_positive_float(source.get(width_key))
            height = _to_positive_float(source.get(height_key))
            if width is not None and height is not None:
                return width, height
    return None, None


def _infer_global_bbox_dims(raw: list[Any]) -> tuple[float | None, float | None]:
    max_x2 = 0.0
    max_y2 = 0.0
    found = False
    for frame_item in raw:
        if not isinstance(frame_item, dict):
            continue
        detections = frame_item.get("detections")
        if not isinstance(detections, list):
            continue
        for det in detections:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x2 = _to_positive_float(bbox[2])
            y2 = _to_positive_float(bbox[3])
            if x2 is None or y2 is None:
                continue
            max_x2 = max(max_x2, x2)
            max_y2 = max(max_y2, y2)
            found = True
    if not found or max_x2 <= 0.0 or max_y2 <= 0.0:
        return None, None
    return float(max_x2), float(max_y2)


def _parse_activation_vector(
    *,
    activation_item: dict[str, Any],
    frame_num: int,
    det_index: int,
    expected_activation_dim: int | None,
    activation_topk: int | None,
) -> tuple[np.ndarray, int]:
    vec = activation_item.get("vector")
    if not isinstance(vec, list):
        raise ValueError(f"Detection at frame {frame_num} index {det_index} activation.vector must be a list")
    activation_vec = np.asarray(vec, dtype=np.float32)
    if activation_vec.ndim != 1:
        raise ValueError(f"Detection at frame {frame_num} index {det_index} activation vector must be 1D")

    activation_dim = int(activation_vec.shape[0])
    if expected_activation_dim is not None and activation_dim != expected_activation_dim:
        raise ValueError(
            f"Detection at frame {frame_num} index {det_index} activation vector dim must be "
            f"{expected_activation_dim}, got {activation_dim}"
        )

    declared_dim = activation_item.get("dim")
    if declared_dim is not None and int(declared_dim) != activation_dim:
        raise ValueError(
            f"Detection at frame {frame_num} index {det_index} activation.dim must be "
            f"{activation_dim}, got {declared_dim}"
        )

    if activation_topk is not None:
        activation_vec = topk_l2_renorm(activation_vec, topk=int(activation_topk))
    return activation_vec, activation_dim


def _parse_dino_vector(
    activation_item: dict[str, Any],
    *,
    warn: WarnOnce,
) -> np.ndarray | None:
    dino_raw = activation_item.get("dino_vector")
    dino_available_raw = activation_item.get("dino_available")
    dino_available = bool(dino_available_raw) if dino_available_raw is not None else (dino_raw is not None)
    if not dino_available:
        return None

    if not isinstance(dino_raw, list):
        warn.warn(
            key="dino_non_list",
            message="Encountered non-list dino_vector in enriched detections; treating those entries as unavailable.",
        )
        return None

    dino_arr = np.asarray(dino_raw, dtype=np.float32)
    if (
        dino_arr.ndim != 1
        or int(dino_arr.shape[0]) != _DINO_VECTOR_DIM
        or not bool(np.isfinite(dino_arr).all())
    ):
        warn.warn(
            key="dino_invalid_shape",
            message=(
                "Encountered invalid dino_vector shape/content in enriched detections; "
                "treating those entries as unavailable."
            ),
        )
        return None

    dino_norm = float(np.linalg.norm(dino_arr))
    if dino_norm <= 0.0:
        warn.warn(
            key="dino_zero_norm",
            message="Encountered zero-norm dino_vector in enriched detections; treating those entries as unavailable.",
        )
        return None
    return (dino_arr / dino_norm).astype(np.float32, copy=False)


def _resolve_frame_dims(
    *,
    frame_item: dict[str, Any],
    det_item: dict[str, Any],
    activation_item: dict[str, Any],
    fallback_width: float | None,
    fallback_height: float | None,
    warn: WarnOnce,
) -> tuple[float | None, float | None]:
    frame_width, frame_height = _extract_explicit_frame_dims(frame_item, det_item, activation_item)
    if frame_width is not None and frame_height is not None:
        return frame_width, frame_height

    if fallback_width is not None and fallback_height is not None:
        warn.warn(
            key="inferred_dims",
            message=(
                "frame dimensions missing in enriched detections; using global bbox-inferred dimensions "
                f"w={fallback_width:.1f}, h={fallback_height:.1f}."
            ),
        )
        return fallback_width, fallback_height

    warn.warn(
        key="missing_dims",
        message=(
            "frame dimensions unavailable in enriched detections and cannot be inferred from bbox ranges; "
            "centroid-distance gating will be skipped for affected pairs."
        ),
    )
    return None, None


def load_enriched_frames(path: str, *, activation_topk: int | None = None) -> list[FrameDetections]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError("Expected enriched detections JSON top-level list")

    expected_activation_dim: int | None = None
    fallback_width, fallback_height = _infer_global_bbox_dims(raw)
    warn = WarnOnce()
    valid_dino_vectors = 0
    total_detections = 0

    frames: list[FrameDetections] = []
    for frame_item in raw:
        if not isinstance(frame_item, dict):
            raise ValueError("Each frame entry must be an object")
        if "frame_num" not in frame_item or "detections" not in frame_item:
            raise ValueError("Each frame entry must include frame_num and detections")

        frame_num = int(frame_item["frame_num"])
        detections_raw = frame_item["detections"]
        if not isinstance(detections_raw, list):
            raise ValueError(f"detections for frame {frame_num} must be a list")

        detections: list[Detection] = []
        for det_index, det in enumerate(detections_raw):
            if not isinstance(det, dict):
                raise ValueError(f"Detection at frame {frame_num} index {det_index} must be an object")

            activation = det.get("activation")
            if not isinstance(activation, dict):
                raise ValueError(f"Detection at frame {frame_num} index {det_index} missing activation")

            activation_vec, activation_dim = _parse_activation_vector(
                activation_item=activation,
                frame_num=frame_num,
                det_index=det_index,
                expected_activation_dim=expected_activation_dim,
                activation_topk=activation_topk,
            )
            if expected_activation_dim is None:
                expected_activation_dim = activation_dim
            dino_vec = _parse_dino_vector(activation, warn=warn)

            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Detection at frame {frame_num} index {det_index} bbox must be list[4]")
            bbox_xyxy = np.asarray(bbox, dtype=np.float32)

            frame_width, frame_height = _resolve_frame_dims(
                frame_item=frame_item,
                det_item=det,
                activation_item=activation,
                fallback_width=fallback_width,
                fallback_height=fallback_height,
                warn=warn,
            )

            detections.append(
                Detection(
                    frame_num=frame_num,
                    det_index=int(det_index),
                    class_id=int(det.get("class_id", -1)),
                    class_name=str(det.get("class_name", "")),
                    bbox_xyxy=bbox_xyxy,
                    confidence=float(det.get("confidence", 0.0)),
                    activation_vec=activation_vec,
                    small_crop_flag=bool(activation.get("small_crop_flag", False)),
                    raw_payload=copy.deepcopy(det),
                    frame_width=frame_width,
                    frame_height=frame_height,
                    dino_vector=dino_vec,
                )
            )
            total_detections += 1
            if dino_vec is not None:
                valid_dino_vectors += 1

        frames.append(FrameDetections(frame_num=frame_num, detections=detections))

    frames.sort(key=lambda item: item.frame_num)
    coverage = (
        (float(valid_dino_vectors) / float(total_detections)) if total_detections > 0 else 0.0
    )
    print(
        "INFO: DINO sidecar coverage "
        f"{valid_dino_vectors}/{total_detections} detections ({coverage:.1%}).",
        file=sys.stderr,
    )
    return frames


def build_output_paths(output_dir: str) -> TemporalLinkArtifacts:
    return TemporalLinkArtifacts(
        linked_detections_path=os.path.join(output_dir, "linked_detections.json"),
        tracks_path=os.path.join(output_dir, "tracks.json"),
        manifest_path=os.path.join(output_dir, "linking_manifest.json"),
    )


def write_linking_outputs(
    artifacts: TemporalLinkArtifacts,
    linked_frames: list[dict[str, Any]],
    tracks_payload: dict[str, Any],
    manifest_payload: dict[str, Any],
) -> None:
    write_json(artifacts.linked_detections_path, linked_frames)
    write_json(artifacts.tracks_path, tracks_payload)
    write_json(artifacts.manifest_path, manifest_payload)
