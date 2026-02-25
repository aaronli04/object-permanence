"""Filesystem and JSON helpers for trace enrichment."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Any

from .types import OutputArtifacts


def write_json(path: str, payload: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def video_stem(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def find_videos(video_dir: str, pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(video_dir, pattern)))


def build_enriched_output_dir(output_root: str, video_path: str) -> str:
    return os.path.join(output_root, video_stem(video_path))


def build_output_artifacts(output_dir: str) -> OutputArtifacts:
    return OutputArtifacts(
        enriched_json_path=os.path.join(output_dir, "enriched_detections.json"),
        pca_path=os.path.join(output_dir, "pca_projection.pkl"),
        manifest_path=os.path.join(output_dir, "projection_manifest.json"),
    )
