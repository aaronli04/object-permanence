#!/usr/bin/env python3
"""Extract class-filtered activation vectors from enriched detections JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_vectors(
    payload: list[dict[str, Any]],
    *,
    class_id: int,
    min_confidence: float,
    max_per_frame: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_item in payload:
        frame_num = int(frame_item.get("frame_num", -1))
        detections = frame_item.get("detections") or []
        if not isinstance(detections, list):
            continue

        accepted = 0
        for det in detections:
            if int(det.get("class_id", -1)) != class_id:
                continue
            confidence = float(det.get("confidence", 0.0))
            if confidence <= min_confidence:
                continue

            activation = det.get("activation") or {}
            vec = activation.get("vector")
            if not isinstance(vec, list):
                continue

            rows.append({"frame": frame_num, "vec": [float(v) for v in vec]})
            accepted += 1
            if max_per_frame > 0 and accepted >= max_per_frame:
                break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract activation vectors for a target class from enriched JSON.")
    parser.add_argument("--enriched-json", required=True, help="Path to enriched_detections.json.")
    parser.add_argument("--output-json", required=True, help="Path to output JSON list of {'frame','vec'} objects.")
    parser.add_argument("--class-id", type=int, required=True, help="Target class id to extract.")
    parser.add_argument("--min-confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument(
        "--max-per-frame",
        type=int,
        default=0,
        help="Max vectors to keep per frame (0 keeps all; default: 0).",
    )
    args = parser.parse_args()

    enriched_path = Path(args.enriched_json)
    output_path = Path(args.output_json)

    payload = _load_json(enriched_path)
    if not isinstance(payload, list):
        raise ValueError("Expected enriched JSON root to be a list of frame objects.")

    rows = _extract_vectors(
        payload,
        class_id=int(args.class_id),
        min_confidence=float(args.min_confidence),
        max_per_frame=int(args.max_per_frame),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"saved={output_path}")
    print(f"vectors={len(rows)}")
    if rows:
        frames = sorted({int(row['frame']) for row in rows})
        print(f"first_frame={frames[0]}")
        print(f"last_frame={frames[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
