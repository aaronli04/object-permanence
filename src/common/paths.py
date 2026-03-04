"""Shared path naming helpers."""

from __future__ import annotations

import os


def video_stem(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def scenario_name_from_enriched_json(enriched_json_path: str) -> str:
    parent = os.path.basename(os.path.dirname(enriched_json_path))
    return parent if parent else "run"
