#!/usr/bin/env python3
"""Run per-layer temporal stability sweep on YOLOv8 module outputs."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from trace_enrichment.model import get_module_map, load_yolo  # noqa: E402
from trace_enrichment.sampler import FrameSampler  # noqa: E402

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError(f"torch import failed: {exc}") from exc


@dataclass
class LayerAccum:
    module_type: str
    feature_dim: int | None = None
    norms: list[float] = field(default_factory=list)
    vectors_by_frame: dict[int, list[np.ndarray]] = field(default_factory=dict)


class HookBank:
    """Collect last forward output tensor per hooked module."""

    def __init__(self, module_map: dict[str, Any]) -> None:
        self.module_map = module_map
        self.outputs: dict[str, Any] = {}
        self._handles: list[Any] = []

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (list, tuple)) else output
            if isinstance(tensor, torch.Tensor):
                self.outputs[name] = tensor.detach().cpu()
            else:
                self.outputs[name] = None

        return hook

    def register(self) -> None:
        for name, module in self.module_map.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def clear(self) -> None:
        self.outputs.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "HookBank":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


def _bbox_to_feature_roi(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    frame_h: int,
    frame_w: int,
    fmap_h: int,
    fmap_w: int,
) -> tuple[int, int, int, int] | None:
    if frame_h <= 0 or frame_w <= 0 or fmap_h <= 0 or fmap_w <= 0:
        return None

    x1, y1, x2, y2 = bbox_xyxy
    fx1 = int(math.floor((x1 / frame_w) * fmap_w))
    fy1 = int(math.floor((y1 / frame_h) * fmap_h))
    fx2 = int(math.ceil((x2 / frame_w) * fmap_w))
    fy2 = int(math.ceil((y2 / frame_h) * fmap_h))

    fx1 = max(0, min(fx1, fmap_w))
    fy1 = max(0, min(fy1, fmap_h))
    fx2 = max(0, min(fx2, fmap_w))
    fy2 = max(0, min(fy2, fmap_h))

    if fx2 - fx1 < 1 or fy2 - fy1 < 1:
        return None
    return fx1, fy1, fx2, fy2


def _iter_target_detections(result: Any, *, class_id: int, min_confidence: float) -> list[tuple[tuple[float, float, float, float], float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    rows: list[tuple[tuple[float, float, float, float], float]] = []
    for box in boxes:
        cls_value = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
        conf_value = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
        if cls_value != class_id or conf_value <= min_confidence:
            continue
        xyxy = box.xyxy[0].tolist() if getattr(box, "xyxy", None) is not None else [0.0, 0.0, 0.0, 0.0]
        rows.append(((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])), conf_value))
    return rows


def _layer_rows(
    *,
    layer_data: dict[str, LayerAccum],
    sampled_frames: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_name, accum in layer_data.items():
        if accum.feature_dim is None:
            continue
        pair_scores: list[float] = []
        for prev_frame, curr_frame in zip(sampled_frames, sampled_frames[1:]):
            prev_vecs = accum.vectors_by_frame.get(prev_frame) or []
            curr_vecs = accum.vectors_by_frame.get(curr_frame) or []
            if not prev_vecs or not curr_vecs:
                continue
            prev_matrix = np.stack(prev_vecs, axis=0)
            curr_matrix = np.stack(curr_vecs, axis=0)
            pair_scores.append(float(np.mean(prev_matrix @ curr_matrix.T)))

        if not pair_scores:
            continue

        norms = np.asarray(accum.norms, dtype=np.float64)
        norm_std = float(np.std(norms)) if norms.size > 0 else float("nan")
        rows.append(
            {
                "layer_name": layer_name,
                "module_type": accum.module_type,
                "feature_dim": int(accum.feature_dim),
                "mean_consecutive_cosine": float(np.mean(np.asarray(pair_scores, dtype=np.float64))),
                "norm_std": norm_std,
            }
        )
    rows.sort(
        key=lambda row: (
            1 if not math.isfinite(row["mean_consecutive_cosine"]) else 0,
            -row["mean_consecutive_cosine"] if math.isfinite(row["mean_consecutive_cosine"]) else 0.0,
        )
    )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer_name",
                "module_type",
                "feature_dim",
                "mean_consecutive_cosine",
                "norm_std",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer temporal stability sweep for YOLOv8 module outputs.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights path or identifier.")
    parser.add_argument("--sample-rate", type=int, default=5, help="Sample every N frames.")
    parser.add_argument("--max-sampled-frames", type=int, default=20, help="Max sampled frames to process.")
    parser.add_argument("--class-id", type=int, default=32, help="Target class id (default: sports ball=32).")
    parser.add_argument("--min-confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument(
        "--output-csv",
        default="experiments/results/layer_stability_sweep.csv",
        help="CSV output path.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of top rows to print.")
    args = parser.parse_args()

    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be > 0")
    if args.max_sampled_frames <= 0:
        raise ValueError("--max-sampled-frames must be > 0")

    yolo = load_yolo(args.model)
    module_map = get_module_map(yolo)
    module_types = {name: module.__class__.__name__ for name, module in module_map.items()}
    layer_data: dict[str, LayerAccum] = {}

    sampled_frames: list[int] = []
    target_detection_count = 0

    with HookBank(module_map) as hooks:
        for frame_num, frame in FrameSampler(args.video, args.sample_rate):
            if len(sampled_frames) >= args.max_sampled_frames:
                break

            hooks.clear()
            with torch.no_grad():
                results = yolo(frame, verbose=False)
            if not results:
                continue

            result = results[0]
            frame_h = int(frame.shape[0])
            frame_w = int(frame.shape[1])
            sampled_frames.append(int(frame_num))

            target_detections = _iter_target_detections(
                result,
                class_id=int(args.class_id),
                min_confidence=float(args.min_confidence),
            )
            if not target_detections:
                continue

            for bbox_xyxy, _confidence in target_detections:
                target_detection_count += 1
                for layer_name, output in hooks.outputs.items():
                    if not isinstance(output, torch.Tensor):
                        continue
                    if output.ndim != 4:
                        continue
                    if output.shape[0] < 1:
                        continue

                    fmap = output[0]
                    if fmap.ndim != 3:
                        continue
                    channels, fmap_h, fmap_w = map(int, fmap.shape)
                    if fmap_h < 4 or fmap_w < 4:
                        continue

                    roi = _bbox_to_feature_roi(
                        bbox_xyxy=bbox_xyxy,
                        frame_h=frame_h,
                        frame_w=frame_w,
                        fmap_h=fmap_h,
                        fmap_w=fmap_w,
                    )
                    if roi is None:
                        continue

                    x1, y1, x2, y2 = roi
                    crop = fmap[:, y1:y2, x1:x2]
                    if crop.numel() <= 0:
                        continue

                    pooled = F.adaptive_avg_pool2d(crop.unsqueeze(0), (1, 1)).reshape(channels)
                    raw_vec = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                    norm = float(np.linalg.norm(raw_vec))
                    if not math.isfinite(norm) or norm <= 0.0:
                        continue
                    vec = raw_vec / norm

                    accum = layer_data.setdefault(
                        layer_name,
                        LayerAccum(module_type=module_types.get(layer_name, "unknown")),
                    )
                    if accum.feature_dim is None:
                        accum.feature_dim = int(vec.shape[0])
                    accum.norms.append(norm)
                    accum.vectors_by_frame.setdefault(int(frame_num), []).append(vec)

    rows = _layer_rows(layer_data=layer_data, sampled_frames=sampled_frames)
    output_csv = Path(args.output_csv)
    _write_csv(output_csv, rows)

    print(f"sampled_frames={len(sampled_frames)}")
    print(f"target_detections={target_detection_count}")
    print(f"named_modules_scanned={len(module_map)}")
    print(f"eligible_layers={len(rows)}")
    print(f"csv_saved={output_csv}")
    print("")
    print(f"Top {args.top_n} layers by mean_consecutive_cosine:")
    for idx, row in enumerate(rows[: args.top_n], start=1):
        print(
            f"{idx:>2}. {row['layer_name']:<28} "
            f"type={row['module_type']:<14} "
            f"dim={row['feature_dim']:<4} "
            f"mean_cos={row['mean_consecutive_cosine']:.6f} "
            f"norm_std={row['norm_std']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
