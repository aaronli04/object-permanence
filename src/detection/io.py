"""I/O utilities for detection results."""

from __future__ import annotations

import json
import os
from typing import Dict, List

from .types import FrameDetections


def build_output_path(video_path: str) -> str:
    """Build the JSON output path for a given input video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(
        "experiments",
        "results",
        "detections",
        f"{video_name}_detections.json",
    )


class OutputWriter:
    """Serialize detections to a JSON file."""
    def __init__(self, output_path: str) -> None:
        """Create a writer that writes to the given path."""
        self.output_path = output_path

    def write(self, frames: List[FrameDetections]) -> None:
        """Write detections to disk as JSON."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        payload: List[Dict[str, object]] = []
        for frame in frames:
            payload.append(
                {
                    "frame_num": frame.frame_num,
                    "detections": [
                        {
                            "class_id": d.class_id,
                            "class_name": d.class_name,
                            "bbox": d.bbox,
                            "confidence": d.confidence,
                        }
                        for d in frame.detections
                    ],
                }
            )

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
