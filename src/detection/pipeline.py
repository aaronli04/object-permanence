"""End-to-end detection pipeline."""

from __future__ import annotations

import os
from typing import List

from .output_writer import FrameDetections, OutputWriter
from .sampler import FrameSampler
from .yolo_runner import YoloRunner


def build_output_path(video_path: str) -> str:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(
        "experiments",
        "results",
        "detections",
        f"{video_name}_detections.json",
    )


def run_pipeline(video_path: str, sample_rate: int, model_name: str) -> str:
    sampler = FrameSampler(video_path, sample_rate)
    yolo = YoloRunner(model_name)
    output_path = build_output_path(video_path)
    writer = OutputWriter(output_path)

    frames: List[FrameDetections] = []
    for frame_num, frame in sampler:
        detections = yolo.run(frame)
        frames.append(FrameDetections(frame_num=frame_num, detections=detections))

    writer.write(frames)
    return output_path
