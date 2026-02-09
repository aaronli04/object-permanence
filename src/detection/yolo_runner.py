"""YOLOv8 inference wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    bbox: List[float]
    confidence: float


class YoloRunner:
    def __init__(self, model_name: str) -> None:
        self.model = YOLO(model_name)

    def run(self, frame: "cv2.Mat") -> List[Detection]:
        results = self.model(frame, verbose=False)
        if not results:
            return []

        result = results[0]
        names = result.names or {}
        detections: List[Detection] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            xyxy = box.xyxy[0].tolist() if box.xyxy is not None else [0.0, 0.0, 0.0, 0.0]
            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=names.get(cls_id, "unknown"),
                    bbox=[float(v) for v in xyxy],
                    confidence=conf,
                )
            )

        return detections
