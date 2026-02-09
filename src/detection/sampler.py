"""Frame sampling utilities."""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2


class FrameSampler:
    """Iterate through a video and yield every Nth frame."""
    def __init__(self, video_path: str, sample_rate: int) -> None:
        """Create a sampler for a video path and sampling interval."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        self.video_path = video_path
        self.sample_rate = sample_rate

    def __iter__(self) -> Iterable[Tuple[int, "cv2.Mat"]]:
        """Yield (frame_num, frame) tuples from the input video."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.video_path}")

        frame_num = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_num % self.sample_rate == 0:
                    yield frame_num, frame
                frame_num += 1
        finally:
            cap.release()
