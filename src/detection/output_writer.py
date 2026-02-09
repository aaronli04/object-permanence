"""Output writers for detection results."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

from .yolo_runner import Detection


@dataclass
class FrameDetections:
    frame_num: int
    detections: List[Detection]


class OutputWriter:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def write(self, frames: List[FrameDetections]) -> None:
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
