"""End-to-end detection pipeline."""

from __future__ import annotations

from typing import List, Optional

from .io import OutputWriter, build_output_path
from .sampler import FrameSampler
from .types import FrameDetections
from .yolo_runner import YoloRunner


def run_pipeline(
    video_path: str,
    sample_rate: int,
    model_name: str,
    writer: Optional[OutputWriter] = None,
) -> str:
    """Run sampling, inference, and writing for a single video."""
    sampler = FrameSampler(video_path, sample_rate)
    yolo = YoloRunner(model_name)
    output_path = build_output_path(video_path)
    writer = writer or OutputWriter(output_path)

    frames: List[FrameDetections] = []
    for frame_num, frame in sampler:
        detections = yolo.run(frame)
        frames.append(FrameDetections(frame_num=frame_num, detections=detections))

    writer.write(frames)
    return output_path
