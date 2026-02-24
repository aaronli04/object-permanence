#!/usr/bin/env python3
"""Validate enriched detection trace schema and activation vectors."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level list")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate enriched detections JSON.")
    parser.add_argument("enriched_json", help="Path to enriched_detections.json")
    parser.add_argument("--expected-dim", type=int, default=256, help="Expected activation dimension")
    args = parser.parse_args()

    frames = load_json(args.enriched_json)
    frame_count = len(frames)
    det_count = 0
    small_crop_count = 0
    norms: List[float] = []

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
            if int(act.get("dim", -1)) != args.expected_dim:
                raise AssertionError(
                    f"Expected activation dim {args.expected_dim}, got {act.get('dim')} in frame {frame['frame_num']}"
                )
            vec = act.get("vector")
            if not isinstance(vec, list) or len(vec) != args.expected_dim:
                raise AssertionError(
                    f"Activation vector length mismatch in frame {frame['frame_num']}: {len(vec) if isinstance(vec, list) else 'non-list'}"
                )
            if any((not isinstance(v, (int, float))) or (not math.isfinite(float(v))) for v in vec):
                raise AssertionError(f"Activation vector contains NaN/Inf in frame {frame['frame_num']}")
            norm = math.sqrt(sum(float(v) * float(v) for v in vec))
            norms.append(norm)
            if bool(act.get("small_crop_flag")):
                small_crop_count += 1

    mean_norm = (sum(norms) / len(norms)) if norms else 0.0
    print(f"frames={frame_count}")
    print(f"detections={det_count}")
    print(f"expected_dim={args.expected_dim}")
    print(f"small_crop_flag_count={small_crop_count}")
    print(f"mean_l2_norm={mean_norm:.6f}")
    print("Validation passed")


if __name__ == "__main__":
    main()
