"""Shared data models for detection outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    """Single object detection from a frame."""
    class_id: int
    class_name: str
    bbox: List[float]
    confidence: float


@dataclass
class FrameDetections:
    """Detections for a single frame."""
    frame_num: int
    detections: List[Detection]
