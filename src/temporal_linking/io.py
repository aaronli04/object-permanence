"""I/O utilities for temporal linking."""

from __future__ import annotations

import copy
import os
import sys
from typing import Any

import numpy as np

try:
    from common.io import load_json, write_json
    from common.numeric import topk_l2_renorm
except ImportError:  # pragma: no cover - import-path compatibility
    from src.common.io import load_json, write_json  # type: ignore
    from src.common.numeric import topk_l2_renorm  # type: ignore

from .types import Detection, FrameDetections, TemporalLinkArtifacts


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


def load_enriched_frames(path: str, *, activation_topk: int | None = None) -> list[FrameDetections]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError("Expected enriched detections JSON top-level list")

    expected_activation_dim: int | None = None
    fallback_width, fallback_height = _infer_global_bbox_dims(raw)
    warned_inferred_dims = False
    warned_missing_dims = False

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

            vec = activation.get("vector")
            if not isinstance(vec, list):
                raise ValueError(f"Detection at frame {frame_num} index {det_index} activation.vector must be a list")
            activation_vec = np.asarray(vec, dtype=np.float32)
            if activation_vec.ndim != 1:
                raise ValueError(f"Detection at frame {frame_num} index {det_index} activation vector must be 1D")
            activation_dim = int(activation_vec.shape[0])
            if expected_activation_dim is None:
                expected_activation_dim = activation_dim
            elif activation_dim != expected_activation_dim:
                raise ValueError(
                    f"Detection at frame {frame_num} index {det_index} activation vector dim must be "
                    f"{expected_activation_dim}, got {activation_dim}"
                )
            declared_dim = activation.get("dim")
            if declared_dim is not None and int(declared_dim) != activation_dim:
                raise ValueError(
                    f"Detection at frame {frame_num} index {det_index} activation.dim must be "
                    f"{activation_dim}, got {declared_dim}"
                )
            if activation_topk is not None:
                activation_vec = topk_l2_renorm(activation_vec, topk=int(activation_topk))

            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Detection at frame {frame_num} index {det_index} bbox must be list[4]")
            bbox_xyxy = np.asarray(bbox, dtype=np.float32)

            frame_width, frame_height = _extract_explicit_frame_dims(frame_item, det, activation)
            if frame_width is None or frame_height is None:
                if fallback_width is not None and fallback_height is not None:
                    frame_width = fallback_width
                    frame_height = fallback_height
                    if not warned_inferred_dims:
                        print(
                            "WARNING: frame dimensions missing in enriched detections; "
                            f"using global bbox-inferred dimensions w={fallback_width:.1f}, h={fallback_height:.1f}.",
                            file=sys.stderr,
                        )
                        warned_inferred_dims = True
                elif not warned_missing_dims:
                    print(
                        "WARNING: frame dimensions unavailable in enriched detections and cannot be inferred from "
                        "bbox ranges; centroid-distance gating will be skipped for affected pairs.",
                        file=sys.stderr,
                    )
                    warned_missing_dims = True

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
                )
            )

        frames.append(FrameDetections(frame_num=frame_num, detections=detections))

    frames.sort(key=lambda item: item.frame_num)
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
