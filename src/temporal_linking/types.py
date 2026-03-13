"""Typed models for temporal linking."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    CLOSED = "closed"


@dataclass(frozen=True)
class Detection:
    frame_num: int
    det_index: int
    class_id: int
    class_name: str
    bbox_xyxy: np.ndarray
    confidence: float
    activation_vec: np.ndarray
    small_crop_flag: bool
    raw_payload: dict[str, Any]
    frame_width: float | None = None
    frame_height: float | None = None
    dino_vector: np.ndarray | None = None


@dataclass(frozen=True)
class FrameDetections:
    frame_num: int
    detections: list[Detection]


@dataclass(frozen=True)
class SerializedTrackObservation:
    frame_num: int
    det_index: int
    bbox: list[float]
    visual_similarity: float | None
    fragment_track_id: int | None = None


ObservationLike: TypeAlias = SerializedTrackObservation | Mapping[str, Any]


def serialize_observation(
    observation: ObservationLike,
    *,
    default_fragment_track_id: int | None = None,
) -> dict[str, Any]:
    if isinstance(observation, SerializedTrackObservation):
        payload = asdict(observation)
    else:
        payload = {
            "frame_num": int(observation["frame_num"]),
            "det_index": int(observation["det_index"]),
            "bbox": [float(v) for v in observation.get("bbox", [])],
            "visual_similarity": (
                None
                if observation.get("visual_similarity") is None
                else float(observation["visual_similarity"])
            ),
        }
        raw_fragment_track_id = observation.get("fragment_track_id")
        if raw_fragment_track_id is not None:
            payload["fragment_track_id"] = int(raw_fragment_track_id)

    if payload.get("fragment_track_id") is None and default_fragment_track_id is not None:
        payload["fragment_track_id"] = int(default_fragment_track_id)
    return payload


@dataclass(frozen=True)
class SerializedTemporalLink:
    track_id: int
    track_status: str
    source_track_status: str
    visual_similarity: float | None
    spatial_score: float | None
    total_score: float | None
    age_since_seen: int
    fragment_track_id: int | None = None


TemporalLinkLike: TypeAlias = SerializedTemporalLink | Mapping[str, Any]


def serialize_temporal_link(
    temporal_link: TemporalLinkLike,
    *,
    default_fragment_track_id: int | None = None,
) -> dict[str, Any]:
    if isinstance(temporal_link, SerializedTemporalLink):
        payload = asdict(temporal_link)
    else:
        payload = {
            "track_id": int(temporal_link["track_id"]),
            "track_status": str(temporal_link["track_status"]),
            "source_track_status": str(temporal_link["source_track_status"]),
            "visual_similarity": (
                None if temporal_link.get("visual_similarity") is None else float(temporal_link["visual_similarity"])
            ),
            "spatial_score": None if temporal_link.get("spatial_score") is None else float(temporal_link["spatial_score"]),
            "total_score": None if temporal_link.get("total_score") is None else float(temporal_link["total_score"]),
            "age_since_seen": int(temporal_link["age_since_seen"]),
        }
        raw_fragment_track_id = temporal_link.get("fragment_track_id")
        if raw_fragment_track_id is not None:
            payload["fragment_track_id"] = int(raw_fragment_track_id)

    if payload.get("fragment_track_id") is None and default_fragment_track_id is not None:
        payload["fragment_track_id"] = int(default_fragment_track_id)
    return payload


@dataclass(frozen=True)
class TraceFrameReference:
    frame_num: int
    det_index: int
    bbox: list[float]
    fragment_track_id: int
    canonical_track_id: int
    role: str


@dataclass(frozen=True)
class TraceSummaryItem:
    track_id: int
    class_id: int
    class_name: str
    start_frame: int
    end_frame: int
    hits: int
    valid_track: bool
    relinked_from: list[int]
    had_recovery: bool
    frames: list[TraceFrameReference]


@dataclass(frozen=True)
class TraceSummaryPayload:
    schema_version: str
    generated_at_utc: str
    config_hash_sha256: str
    input_enriched_json: str
    items: list[TraceSummaryItem]


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    status: TrackStatus
    start_frame: int
    last_seen_frame: int
    hits: int = 1
    miss_streak: int = 0
    total_misses: int = 0
    max_miss_streak: int = 0

    last_bbox_xyxy: np.ndarray = field(default_factory=lambda: np.zeros((4,), dtype=np.float32))
    last_vec: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    ema_vec: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    vec_history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=5))
    sim_history: deque[float] = field(default_factory=lambda: deque(maxlen=5))

    visual_similarity_sum: float = 0.0
    visual_similarity_count: int = 0

    observations: list[SerializedTrackObservation | dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    obs_vecs: list[np.ndarray] = field(default_factory=list)
    obs_dino_vecs: list[np.ndarray] = field(default_factory=list)
    obs_positions: list[tuple[float, float, int]] = field(default_factory=list)
    frame_width: float | None = None
    frame_height: float | None = None
    dino_vector: np.ndarray | None = None


@dataclass(frozen=True)
class TrackFragment:
    track_id: int
    class_id: int
    first_frame: int
    last_frame: int
    hits: int
    centroid: np.ndarray
    frame_vecs: np.ndarray
    last_positions: list[tuple[float, float, int]]
    first_position: tuple[float, float, int]
    dino_vector: np.ndarray | None = None


@dataclass(frozen=True)
class RelinkEdge:
    predecessor_id: int
    successor_id: int
    score: float
    method: Literal["dino", "yolo", "spatial"]


@dataclass(frozen=True)
class RelinkManifest:
    schema_version: str
    config: dict[str, Any]
    stats: dict[str, Any]
    accepted_edges: list[dict[str, Any]]
    merge_map: dict[str, int]


@dataclass(frozen=True)
class PairScores:
    visual: np.ndarray
    spatial: np.ndarray
    tie_break: np.ndarray
    assignment: np.ndarray
    eligible: np.ndarray


@dataclass(frozen=True)
class Assignment:
    track_id: int
    det_index: int
    visual_similarity: float
    spatial_score: float
    total_score: float
    source_track_status: TrackStatus


@dataclass
class TrackerState:
    next_track_id: int = 1
    active: dict[int, Track] = field(default_factory=dict)
    lost: dict[int, Track] = field(default_factory=dict)
    closed: dict[int, Track] = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalLinkArtifacts:
    linked_detections_path: str
    tracks_path: str
    manifest_path: str
    trace_summary_path: str
    trace_summary_dir: str


@dataclass(frozen=True)
class TemporalLinkingResult:
    linked_frames: list[dict[str, Any]]
    tracks_payload: dict[str, Any]
    manifest_payload: dict[str, Any]
    relink_manifest_payload: dict[str, Any]
    trace_summary_payload: dict[str, Any]


@dataclass(frozen=True)
class TemporalLinkingOutputs:
    linked_detections: str
    tracks: str
    linking_manifest: str
    relink_manifest: str
    trace_summary: str
    trace_summary_dir: str | None = None
