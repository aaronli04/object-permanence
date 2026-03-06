#!/usr/bin/env python3
"""Validate enriched detection trace schema and activation vectors."""

from __future__ import annotations

import argparse
import math
from typing import Any

try:
    from common.io import load_json
except ImportError:  # pragma: no cover - import-path compatibility
    from src.common.io import load_json  # type: ignore


def _infer_expected_dim(frames: list[dict[str, Any]]) -> int:
    for frame in frames:
        detections = frame.get("detections")
        if not isinstance(detections, list):
            continue
        for det in detections:
            if not isinstance(det, dict):
                continue
            act = det.get("activation")
            if not isinstance(act, dict):
                continue
            vec = act.get("vector")
            if isinstance(vec, list) and len(vec) > 0:
                return int(len(vec))
            dim = act.get("dim")
            if isinstance(dim, (int, float)) and int(dim) > 0:
                return int(dim)
    raise AssertionError("Could not infer expected activation dim from enriched detections")


def validate_enriched_frames(frames: list[dict[str, Any]], expected_dim: int | None) -> dict[str, float | int]:
    dim = int(expected_dim) if expected_dim is not None else _infer_expected_dim(frames)
    frame_count = len(frames)
    det_count = 0
    small_crop_count = 0
    norms: list[float] = []

    for frame in frames:
        if "frame_num" not in frame or "detections" not in frame:
            raise AssertionError("Each frame record must include frame_num and detections")
        detections = frame["detections"]
        if not isinstance(detections, list):
            raise AssertionError("detections must be a list")

        for det in detections:
            det_count += 1
            if "activation" not in det:
                raise AssertionError(f"Missing activation field on detection in frame {frame['frame_num']}")
            act = det["activation"]
            if int(act.get("dim", -1)) != dim:
                raise AssertionError(
                    f"Expected activation dim {dim}, got {act.get('dim')} in frame {frame['frame_num']}"
                )
            vec = act.get("vector")
            if not isinstance(vec, list) or len(vec) != dim:
                got_len = len(vec) if isinstance(vec, list) else "non-list"
                raise AssertionError(
                    f"Activation vector length mismatch in frame {frame['frame_num']}: {got_len}"
                )
            if any((not isinstance(v, (int, float))) or (not math.isfinite(float(v))) for v in vec):
                raise AssertionError(f"Activation vector contains NaN/Inf in frame {frame['frame_num']}")
            norm = math.sqrt(sum(float(v) * float(v) for v in vec))
            norms.append(norm)
            if bool(act.get("small_crop_flag")):
                small_crop_count += 1

    mean_norm = (sum(norms) / len(norms)) if norms else 0.0
    return {
        "frames": frame_count,
        "detections": det_count,
        "expected_dim": dim,
        "small_crop_flag_count": small_crop_count,
        "mean_l2_norm": mean_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate enriched detections JSON.")
    parser.add_argument("enriched_json", help="Path to enriched_detections.json")
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=None,
        help="Expected activation dimension. If omitted, infer from the first valid detection.",
    )
    args = parser.parse_args()

    frames = load_json(args.enriched_json)
    if not isinstance(frames, list):
        raise ValueError("Expected top-level list")
    stats = validate_enriched_frames(frames, args.expected_dim)
    print(f"frames={stats['frames']}")
    print(f"detections={stats['detections']}")
    print(f"expected_dim={stats['expected_dim']}")
    print(f"small_crop_flag_count={stats['small_crop_flag_count']}")
    print(f"mean_l2_norm={stats['mean_l2_norm']:.6f}")
    print("Validation passed")


if __name__ == "__main__":
    main()
