#!/usr/bin/env python3
"""Run per-layer temporal stability sweep on YOLOv8 module outputs."""

# NOTE: If this sweep is re-run and rankings materially change, update
# src/trace_enrichment/constants.py EMBEDDING_LAYERS weights/layers accordingly.

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
from trace_enrichment.constants import (  # noqa: E402
    DINO_CROP_PADDING_RATIO,
    DINO_EMBEDDING_DIM,
    DINO_INPUT_SIZE,
    DINO_LOAD_TIMEOUT_SECONDS,
    DINO_MODEL_NAME,
    DINO_MODEL_REPO,
    DINO_TINY_CROP_MIN_SIZE,
)
from trace_enrichment.dino import (  # noqa: E402
    DinoEmbedder,
    DinoUnavailableError,
    extract_dino_embedding,
    load_dino_embedder,
)

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
    vectors_by_group: dict[int, list[np.ndarray]] = field(default_factory=dict)
    grouped_total: int = 0
    grouped_with_track_id: int = 0
    grouped_with_class_fallback: int = 0


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


def _extract_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if int(value.numel()) <= 0:
                return None
            scalar = float(value.reshape(-1)[0].item())
        elif hasattr(value, "item"):
            scalar = float(value.item())
        else:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size <= 0:
                return None
            scalar = float(arr[0])
    except Exception:
        return None
    if not math.isfinite(scalar):
        return None
    return int(scalar)


def _iter_target_detections(
    result: Any, *, class_id: int, min_confidence: float
) -> list[tuple[tuple[float, float, float, float], float, int, int | None]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    rows: list[tuple[tuple[float, float, float, float], float, int, int | None]] = []
    for box in boxes:
        cls_value = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
        conf_value = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
        if conf_value <= min_confidence:
            continue
        if class_id >= 0 and cls_value != class_id:
            continue
        xyxy = box.xyxy[0].tolist() if getattr(box, "xyxy", None) is not None else [0.0, 0.0, 0.0, 0.0]
        track_value = _extract_optional_int(getattr(box, "id", None))
        rows.append(
            (
                (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                conf_value,
                cls_value,
                track_value,
            )
        )
    return rows


def _resolve_group_key(*, track_id: int | None, class_id: int, require_track_id: bool) -> int | None:
    if track_id is not None:
        return int(track_id)
    if require_track_id:
        return None
    return int(class_id)


def _infer_model_device(yolo: Any) -> str | None:
    try:
        params = getattr(getattr(yolo, "model", None), "parameters", None)
        if callable(params):
            first = next(params(), None)
            if first is not None and hasattr(first, "device"):
                return str(first.device)
    except Exception:
        return None
    return None


def _compute_separability(
    vectors_by_group: dict[int, list[np.ndarray]],
) -> tuple[float, float, float, int]:
    group_vectors = [vecs for vecs in vectors_by_group.values() if vecs]
    group_count = len(group_vectors)
    if group_count < 2:
        return 0.0, 0.0, 0.0, group_count

    group_within_vars: list[float] = []
    group_mean_vectors: list[np.ndarray] = []
    for vecs in group_vectors:
        mat = np.stack(vecs, axis=0).astype(np.float64, copy=False)
        per_dim_var = np.var(mat, axis=0, dtype=np.float64)
        group_within_vars.append(float(np.mean(per_dim_var)))
        group_mean_vectors.append(np.mean(mat, axis=0, dtype=np.float64))

    within_var = float(np.mean(np.asarray(group_within_vars, dtype=np.float64))) if group_within_vars else 0.0
    group_means = np.stack(group_mean_vectors, axis=0).astype(np.float64, copy=False)
    between_var = float(np.mean(np.var(group_means, axis=0, dtype=np.float64)))
    separability = float(between_var / (within_var + 1e-8))
    if not math.isfinite(separability):
        separability = 0.0
    return within_var, between_var, separability, group_count


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
        within_var, between_var, separability, group_count = _compute_separability(accum.vectors_by_group)
        track_id_coverage = (
            float(accum.grouped_with_track_id) / float(accum.grouped_total) if accum.grouped_total > 0 else 0.0
        )
        if group_count < 2:
            print(
                f"WARNING: layer '{layer_name}' has only {group_count} distinct groups; separability set to 0.0",
                file=sys.stderr,
            )
        if track_id_coverage < 0.30:
            print(
                f"WARNING: layer '{layer_name}' track_id coverage is low ({track_id_coverage:.1%}); "
                "separability may be dominated by class-level signal.",
                file=sys.stderr,
            )
        rows.append(
            {
                "layer_name": layer_name,
                "module_type": accum.module_type,
                "feature_dim": int(accum.feature_dim),
                "mean_consecutive_cosine": float(np.mean(np.asarray(pair_scores, dtype=np.float64))),
                "norm_std": norm_std,
                "within_var": within_var,
                "between_var": between_var,
                "separability": separability,
                "track_id_coverage": track_id_coverage,
            }
        )
    rows.sort(
        key=lambda row: (
            1 if not math.isfinite(row["separability"]) else 0,
            -row["separability"] if math.isfinite(row["separability"]) else 0.0,
            1 if not math.isfinite(row["mean_consecutive_cosine"]) else 0,
            -row["mean_consecutive_cosine"] if math.isfinite(row["mean_consecutive_cosine"]) else 0.0,
            row["layer_name"],
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
                "within_var",
                "between_var",
                "separability",
                "track_id_coverage",
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
    parser.add_argument("--class-id", type=int, default=-1, help="Target class id (-1 includes all classes).")
    parser.add_argument("--min-confidence", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument(
        "--require-track-id",
        action="store_true",
        help="When set, only use detections with finite track_id for grouping; do not fallback to class_id.",
    )
    parser.add_argument(
        "--output-csv",
        default="experiments/results/layer_selection/per_video/layer_stability_sweep.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--dino",
        action="store_true",
        help="When set, include DINO CLS embeddings (dino_vits8) as an additional sweep candidate.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of top rows to print.")
    args = parser.parse_args()

    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be > 0")
    if args.max_sampled_frames <= 0:
        raise ValueError("--max-sampled-frames must be > 0")

    yolo = load_yolo(args.model)
    dino_embedder: DinoEmbedder | None = None
    if bool(args.dino):
        preferred_device = _infer_model_device(yolo)
        try:
            dino_embedder = load_dino_embedder(
                model_name=DINO_MODEL_NAME,
                feature_dim=DINO_EMBEDDING_DIM,
                hub_repo=DINO_MODEL_REPO,
                preferred_device=preferred_device,
                load_timeout_seconds=DINO_LOAD_TIMEOUT_SECONDS,
                input_size=DINO_INPUT_SIZE,
                tiny_crop_min=DINO_TINY_CROP_MIN_SIZE,
                crop_padding_ratio=DINO_CROP_PADDING_RATIO,
            )
        except DinoUnavailableError as exc:
            raise RuntimeError(
                "DINO preflight load failed before sweep start. "
                "No output CSV was written for this run. "
                f"Error: {exc}"
            ) from exc

    module_map = get_module_map(yolo)
    module_types = {name: module.__class__.__name__ for name, module in module_map.items()}
    layer_data: dict[str, LayerAccum] = {}
    warn_once: set[str] = set()

    sampled_frames: list[int] = []
    target_detection_count = 0
    target_detection_with_track_id_count = 0
    target_detection_class_fallback_count = 0

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

            for bbox_xyxy, _confidence, cls_value, track_id in target_detections:
                target_detection_count += 1
                if track_id is not None:
                    target_detection_with_track_id_count += 1
                else:
                    target_detection_class_fallback_count += 1
                group_key = _resolve_group_key(
                    track_id=track_id,
                    class_id=int(cls_value),
                    require_track_id=bool(args.require_track_id),
                )
                if group_key is None:
                    continue
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
                    accum.grouped_total += 1
                    if track_id is not None:
                        accum.grouped_with_track_id += 1
                    else:
                        accum.grouped_with_class_fallback += 1
                    accum.norms.append(norm)
                    accum.vectors_by_frame.setdefault(int(frame_num), []).append(vec)
                    accum.vectors_by_group.setdefault(group_key, []).append(vec)

                if dino_embedder is not None:
                    try:
                        dino_result = extract_dino_embedding(
                            frame_bgr=frame,
                            bbox_xyxy=bbox_xyxy,
                            embedder=dino_embedder,
                        )
                        dino_raw_vec = dino_result.vector
                    except Exception as exc:
                        if "dino_extract_fail" not in warn_once:
                            print(
                                "WARNING: DINO extraction failed for one or more detections; "
                                f"skipping those samples. Error: {exc}",
                                file=sys.stderr,
                            )
                            warn_once.add("dino_extract_fail")
                        continue

                    if dino_result.tiny_crop and "dino_tiny_crop" not in warn_once:
                        print(
                            f"WARNING: One or more DINO crops were smaller than {DINO_TINY_CROP_MIN_SIZE}x"
                            f"{DINO_TINY_CROP_MIN_SIZE}; sweep embeddings may be noisy.",
                            file=sys.stderr,
                        )
                        warn_once.add("dino_tiny_crop")

                    dino_norm = float(np.linalg.norm(dino_raw_vec))
                    if not math.isfinite(dino_norm) or dino_norm <= 0.0:
                        continue
                    dino_vec = dino_raw_vec / dino_norm

                    dino_accum = layer_data.setdefault("dino_cls", LayerAccum(module_type="DINO_ViT_CLS"))
                    if dino_accum.feature_dim is None:
                        dino_accum.feature_dim = int(dino_vec.shape[0])
                    dino_accum.grouped_total += 1
                    if track_id is not None:
                        dino_accum.grouped_with_track_id += 1
                    else:
                        dino_accum.grouped_with_class_fallback += 1
                    dino_accum.norms.append(dino_norm)
                    dino_accum.vectors_by_frame.setdefault(int(frame_num), []).append(dino_vec.astype(np.float32))
                    dino_accum.vectors_by_group.setdefault(group_key, []).append(dino_vec.astype(np.float32))

    rows = _layer_rows(layer_data=layer_data, sampled_frames=sampled_frames)
    output_csv = Path(args.output_csv)
    _write_csv(output_csv, rows)

    print(f"sampled_frames={len(sampled_frames)}")
    print(f"target_detections={target_detection_count}")
    print(f"target_detections_with_valid_track_id={target_detection_with_track_id_count}")
    print(f"target_detections_class_fallback={target_detection_class_fallback_count}")
    print(f"named_modules_scanned={len(module_map)}")
    print(f"eligible_layers={len(rows)}")
    print(f"csv_saved={output_csv}")
    print("")
    print(f"Top {args.top_n} layers by separability (tie-break: mean_consecutive_cosine):")
    for idx, row in enumerate(rows[: args.top_n], start=1):
        print(
            f"{idx:>2}. {row['layer_name']:<28} "
            f"type={row['module_type']:<14} "
            f"dim={row['feature_dim']:<4} "
            f"separability={row['separability']:.6f} "
            f"between_var={row['between_var']:.6f} "
            f"within_var={row['within_var']:.6f} "
            f"mean_cos={row['mean_consecutive_cosine']:.6f} "
            f"norm_std={row['norm_std']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
