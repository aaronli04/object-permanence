"""I/O utilities for temporal linking."""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import numpy as np

from .types import Detection, FrameDetections, TemporalLinkArtifacts

EXPECTED_ACTIVATION_DIM = 256


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_enriched_frames(path: str) -> list[FrameDetections]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError("Expected enriched detections JSON top-level list")

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
            if activation_vec.shape[0] != EXPECTED_ACTIVATION_DIM:
                raise ValueError(
                    f"Detection at frame {frame_num} index {det_index} activation vector dim must be "
                    f"{EXPECTED_ACTIVATION_DIM}, got {activation_vec.shape[0]}"
                )
            activation_dim = activation.get("dim")
            if activation_dim is not None and int(activation_dim) != EXPECTED_ACTIVATION_DIM:
                raise ValueError(
                    f"Detection at frame {frame_num} index {det_index} activation.dim must be "
                    f"{EXPECTED_ACTIVATION_DIM}, got {activation_dim}"
                )

            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Detection at frame {frame_num} index {det_index} bbox must be list[4]")
            bbox_xyxy = np.asarray(bbox, dtype=np.float32)

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
