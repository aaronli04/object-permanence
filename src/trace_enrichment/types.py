"""Typed internal models for trace enrichment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HookConfig:
    layer: str
    stride: int
    requested_layer: str
    layers: tuple[str, ...] = ()
    layer_weights: tuple[float, ...] = ()
    multi_layer_enabled: bool = False


@dataclass(frozen=True)
class RunConfig:
    video_path: str
    model_name: str
    sample_rate: int
    batch_size: int
    pca_dim: int
    output_dir: str


@dataclass(frozen=True)
class OutputArtifacts:
    enriched_json_path: str
    pca_path: str
    manifest_path: str


@dataclass
class CollectedDetection:
    class_id: int
    class_name: str
    bbox: list[float]
    confidence: float
    small_crop_flag: bool = False
    raw_vector: Any | None = None
    projected_vector: Any | None = None


@dataclass
class CollectedFrame:
    frame_num: int
    detections: list[CollectedDetection] = field(default_factory=list)


@dataclass
class CollectionStats:
    total_sampled_frames: int = 0
    frames_with_detections: int = 0
    total_detections: int = 0
