"""YOLOv8 inference wrapper."""

from __future__ import annotations

from typing import List

import cv2
from ultralytics import YOLO

from .types import Detection


class YoloRunner:
    """Runs YOLOv8 inference on frames and normalizes outputs."""
    def __init__(self, model_name: str) -> None:
        """Load a YOLOv8 model by name or local path."""
        self.model = YOLO(model_name)

    def run(self, frame: "cv2.Mat") -> List[Detection]:
        """Run inference on a frame and return detections."""
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
