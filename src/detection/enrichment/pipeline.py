"""Phase 1 single-pass YOLO detection + activation enrichment."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..sampler import FrameSampler
from .introspection import get_module_map

try:
    import joblib
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from ultralytics import YOLO

    _IMPORT_ERROR: Optional[ModuleNotFoundError] = None
except ModuleNotFoundError as exc:
    joblib = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    PCA = None  # type: ignore[assignment]
    YOLO = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


POOL_SIZE = (3, 3)
POOL_STRATEGY = "adaptive_avg_3x3"
OUTPUT_VECTOR_DIM = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_DEEP_LAYER = "8"
DEFAULT_MID_LAYER = "6"
DEFAULT_DEEP_STRIDE = 32
DEFAULT_MID_STRIDE = 16
SCHEMA_VERSION = "activation_enrichment_v2_single_pass"


def _ensure_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        missing = getattr(_IMPORT_ERROR, "name", "unknown")
        raise RuntimeError(
            f"Missing dependency '{missing}'. Install requirements.txt before running Phase 1 enrichment."
        ) from _IMPORT_ERROR


@dataclass(frozen=True)
class LayerHookConfig:
    """Selected hook layers and their feature strides."""

    deep_name: str
    mid_name: str
    deep_stride: int
    mid_stride: int


@dataclass(frozen=True)
class OutputArtifacts:
    """Filesystem destinations for enrichment outputs."""

    enriched_json_path: str
    pca_path: str
    manifest_path: str


@dataclass
class RawActivationRecord:
    """Raw activation vector collected for one detection."""

    frame_num: int
    det_index: int
    raw_vector: "np.ndarray"
    small_crop_flag: bool


@dataclass
class SinglePassCollection:
    """Trace and raw activation vectors collected from one video run."""

    frame_order: List[Dict[str, Any]]
    raw_records: List[RawActivationRecord]
    total_sampled_frames: int
    frames_with_detections: int
    total_detections: int


class FeatureHookCollector:
    """Capture feature maps from named modules during forward passes."""

    def __init__(self, module_map: Dict[str, "torch.nn.Module"], layer_names: Sequence[str]) -> None:
        self.module_map = module_map
        self.layer_names = list(layer_names)
        self._handles: List[Any] = []
        self.outputs: Dict[str, "torch.Tensor"] = {}

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (list, tuple)) else output
            self.outputs[name] = tensor.detach()

        return hook

    def register(self) -> None:
        for name in self.layer_names:
            if name not in self.module_map:
                raise KeyError(f"Hook layer '{name}' not found in model modules")
            self._handles.append(self.module_map[name].register_forward_hook(self._make_hook(name)))

    def clear(self) -> None:
        self.outputs = {}

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "FeatureHookCollector":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _bbox_to_feature_roi(
    bbox_xyxy: Sequence[float],
    stride: int,
    fmap_h: int,
    fmap_w: int,
) -> Tuple[int, int, int, int, bool]:
    x1, y1, x2, y2 = bbox_xyxy
    fx1 = int(math.floor(x1 / stride))
    fy1 = int(math.floor(y1 / stride))
    fx2 = int(math.ceil(x2 / stride))
    fy2 = int(math.ceil(y2 / stride))

    fx1 = max(0, min(fx1, fmap_w - 1))
    fy1 = max(0, min(fy1, fmap_h - 1))
    fx2 = max(0, min(fx2, fmap_w))
    fy2 = max(0, min(fy2, fmap_h))

    small = False
    if fx2 <= fx1:
        small = True
        fx2 = min(fmap_w, fx1 + 1)
        fx1 = max(0, fx2 - 1)
    if fy2 <= fy1:
        small = True
        fy2 = min(fmap_h, fy1 + 1)
        fy1 = max(0, fy2 - 1)

    return fx1, fy1, fx2, fy2, small


def crop_and_pool_feature(
    fmap: "torch.Tensor",
    bbox_xyxy: Sequence[float],
    stride: int,
    pool: "torch.nn.Module",
) -> Tuple["np.ndarray", bool]:
    if fmap.ndim != 3:
        raise ValueError(f"Expected feature map [C,H,W], got {tuple(fmap.shape)}")
    c, h, w = fmap.shape
    x1, y1, x2, y2, small = _bbox_to_feature_roi(bbox_xyxy, stride, h, w)
    crop = fmap[:, y1:y2, x1:x2].unsqueeze(0)
    pooled = pool(crop).reshape(c * POOL_SIZE[0] * POOL_SIZE[1])
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False), small


def build_raw_activation_vector(
    fmap_mid: "torch.Tensor",
    fmap_deep: "torch.Tensor",
    bbox_xyxy: Sequence[float],
    pool: "torch.nn.Module",
    stride_mid: int,
    stride_deep: int,
) -> Tuple["np.ndarray", bool]:
    mid_vec, mid_small = crop_and_pool_feature(fmap_mid, bbox_xyxy, stride_mid, pool)
    deep_vec, deep_small = crop_and_pool_feature(fmap_deep, bbox_xyxy, stride_deep, pool)
    return np.concatenate([deep_vec, mid_vec], axis=0), (mid_small or deep_small)


def _l2_normalize(vec: "np.ndarray") -> "np.ndarray":
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        return np.zeros_like(vec)
    return vec / norm


def _default_layer_aliases() -> List[str]:
    return ["backbone.C2f_deep", "backbone.C2f_mid"]


def _build_artifact_paths(output_dir: str) -> OutputArtifacts:
    return OutputArtifacts(
        enriched_json_path=os.path.join(output_dir, "enriched_detections.json"),
        pca_path=os.path.join(output_dir, "pca_projection.pkl"),
        manifest_path=os.path.join(output_dir, "projection_manifest.json"),
    )


def _layer_manifest_section(cfg: LayerHookConfig) -> Dict[str, Any]:
    return {
        "aliases": _default_layer_aliases(),
        "actual": {"deep": cfg.deep_name, "mid": cfg.mid_name},
        "strides": {"deep": cfg.deep_stride, "mid": cfg.mid_stride},
    }


def _pad_and_normalize_projection(vec: "np.ndarray", source_dim: int, target_dim: int) -> "np.ndarray":
    if source_dim < target_dim:
        padded = np.zeros((target_dim,), dtype=np.float32)
        padded[:source_dim] = vec
        vec = padded
    return _l2_normalize(vec)


def _build_projected_index(
    raw_records: List[RawActivationRecord],
    projected: "np.ndarray",
    effective_pca_dim: int,
    target_pca_dim: int,
) -> Dict[Tuple[int, int], Tuple["np.ndarray", bool]]:
    index: Dict[Tuple[int, int], Tuple["np.ndarray", bool]] = {}
    for rec, proj in zip(raw_records, projected):
        index[(rec.frame_num, rec.det_index)] = (
            _pad_and_normalize_projection(proj, effective_pca_dim, target_pca_dim),
            rec.small_crop_flag,
        )
    return index


def _build_enriched_payload(
    frame_order: List[Dict[str, Any]],
    projected_index: Dict[Tuple[int, int], Tuple["np.ndarray", bool]],
    pca_dim: int,
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for frame in frame_order:
        enriched_dets: List[Dict[str, Any]] = []
        for det_index, det in enumerate(frame.get("detections", [])):
            key = (int(frame["frame_num"]), det_index)
            if key not in projected_index:
                raise RuntimeError(f"Missing projected activation for frame={key[0]} det_index={key[1]}")
            vec, small_crop_flag = projected_index[key]
            enriched_dets.append(
                {
                    "class_id": int(det["class_id"]),
                    "class_name": det["class_name"],
                    "bbox": [float(v) for v in det["bbox"]],
                    "confidence": float(det["confidence"]),
                    "activation": {
                        "vector": [float(v) for v in vec.tolist()],
                        "dim": int(pca_dim),
                        "layers": _default_layer_aliases(),
                        "pool": POOL_STRATEGY,
                        "projection": f"pca_{pca_dim}",
                        "small_crop_flag": bool(small_crop_flag),
                    },
                }
            )
        payload.append(
            {
                "frame_num": int(frame["frame_num"]),
                "detections": sorted(enriched_dets, key=lambda d: float(d["confidence"]), reverse=True),
            }
        )
    return payload


def _extract_sorted_result_detections(result: Any) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    names = getattr(result, "names", None) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        cls_id = int(box.cls.item()) if box.cls is not None else -1
        conf = float(box.conf.item()) if box.conf is not None else 0.0
        xyxy = [float(v) for v in box.xyxy[0].tolist()] if box.xyxy is not None else [0.0, 0.0, 0.0, 0.0]
        detections.append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id, "unknown"),
                "bbox": xyxy,
                "confidence": conf,
            }
        )

    detections.sort(key=lambda d: float(d["confidence"]), reverse=True)
    return detections


def _collect_single_pass_records(
    *,
    yolo: "YOLO",
    video_path: str,
    sample_rate: int,
    module_map: Dict[str, "torch.nn.Module"],
    layer_cfg: LayerHookConfig,
    batch_size: int,
    pool: "torch.nn.Module",
) -> SinglePassCollection:
    frame_order: List[Dict[str, Any]] = []
    raw_records: List[RawActivationRecord] = []
    total_sampled_frames = 0
    frames_with_detections = 0
    total_detections = 0

    def process_batch(batch_items: List[Tuple[int, Any]], hooks: FeatureHookCollector) -> None:
        nonlocal total_sampled_frames, frames_with_detections, total_detections
        if not batch_items:
            return

        hooks.clear()
        frames = [frame for _, frame in batch_items]
        results = yolo(frames, verbose=False)

        if layer_cfg.deep_name not in hooks.outputs or layer_cfg.mid_name not in hooks.outputs:
            raise RuntimeError("Expected hook outputs were not captured during YOLO forward pass")

        deep_batch = hooks.outputs[layer_cfg.deep_name].detach().cpu()
        mid_batch = hooks.outputs[layer_cfg.mid_name].detach().cpu()
        if deep_batch.shape[0] != len(batch_items) or mid_batch.shape[0] != len(batch_items):
            raise RuntimeError("Hook output batch size does not match inference batch size")

        for i, (frame_num, _frame) in enumerate(batch_items):
            detections = _extract_sorted_result_detections(results[i])
            frame_order.append({"frame_num": int(frame_num), "detections": detections})
            total_sampled_frames += 1
            if detections:
                frames_with_detections += 1
            total_detections += len(detections)

            for det_index, det in enumerate(detections):
                raw_vec, small_crop_flag = build_raw_activation_vector(
                    fmap_mid=mid_batch[i],
                    fmap_deep=deep_batch[i],
                    bbox_xyxy=det["bbox"],
                    pool=pool,
                    stride_mid=layer_cfg.mid_stride,
                    stride_deep=layer_cfg.deep_stride,
                )
                raw_records.append(
                    RawActivationRecord(
                        frame_num=int(frame_num),
                        det_index=det_index,
                        raw_vector=raw_vec,
                        small_crop_flag=bool(small_crop_flag),
                    )
                )

    pending: List[Tuple[int, Any]] = []
    with FeatureHookCollector(module_map=module_map, layer_names=[layer_cfg.deep_name, layer_cfg.mid_name]) as hooks:
        for frame_num, frame in FrameSampler(video_path, sample_rate):
            pending.append((int(frame_num), frame))
            if len(pending) >= batch_size:
                process_batch(pending, hooks)
                pending = []
        if pending:
            process_batch(pending, hooks)

    return SinglePassCollection(
        frame_order=frame_order,
        raw_records=raw_records,
        total_sampled_frames=total_sampled_frames,
        frames_with_detections=frames_with_detections,
        total_detections=total_detections,
    )


def run_phase1_enrichment(
    *,
    video_path: str,
    model_name: str,
    output_dir: str,
    sample_rate: int,
    deep_layer_name: str = DEFAULT_DEEP_LAYER,
    mid_layer_name: str = DEFAULT_MID_LAYER,
    stride_deep: int = DEFAULT_DEEP_STRIDE,
    stride_mid: int = DEFAULT_MID_STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pca_dim: int = OUTPUT_VECTOR_DIM,
) -> Dict[str, str]:
    """Run single-pass Phase 1 detection+enrichment and write artifacts."""
    _ensure_runtime_dependencies()

    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if pca_dim <= 0:
        raise ValueError("pca_dim must be > 0")

    layer_cfg = LayerHookConfig(
        deep_name=deep_layer_name,
        mid_name=mid_layer_name,
        deep_stride=stride_deep,
        mid_stride=stride_mid,
    )
    artifacts = _build_artifact_paths(output_dir)

    yolo = YOLO(model_name)
    module_map = get_module_map(yolo)
    pool = torch.nn.AdaptiveAvgPool2d(POOL_SIZE)

    print(
        f"Running Phase 1 single-pass enrichment on '{video_path}' "
        f"(sample_rate={sample_rate}, model={model_name})"
    )
    print(
        f"Using hook layers: deep='{layer_cfg.deep_name}' (stride={layer_cfg.deep_stride}), "
        f"mid='{layer_cfg.mid_name}' (stride={layer_cfg.mid_stride})"
    )

    collection = _collect_single_pass_records(
        yolo=yolo,
        video_path=video_path,
        sample_rate=sample_rate,
        module_map=module_map,
        layer_cfg=layer_cfg,
        batch_size=batch_size,
        pool=pool,
    )

    if collection.total_detections <= 0:
        raise RuntimeError(
            f"No detections found in sampled frames for '{video_path}'. Phase 1 enrichment requires detections."
        )

    raw_matrix = np.stack([r.raw_vector for r in collection.raw_records], axis=0)
    effective_pca_dim = min(pca_dim, raw_matrix.shape[0], raw_matrix.shape[1])
    if effective_pca_dim <= 0:
        raise RuntimeError("No activation vectors collected for PCA")

    pca = PCA(n_components=effective_pca_dim, svd_solver="auto", random_state=0)
    projected = pca.fit_transform(raw_matrix).astype(np.float32, copy=False)

    projected_index = _build_projected_index(collection.raw_records, projected, effective_pca_dim, pca_dim)
    enriched_payload = _build_enriched_payload(collection.frame_order, projected_index, pca_dim)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pca, artifacts.pca_path)
    save_json(artifacts.enriched_json_path, enriched_payload)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "projection_file": os.path.basename(artifacts.pca_path),
        "projection_dim": int(pca_dim),
        "fitted_pca_components": int(effective_pca_dim),
        "fit_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_video_hash_sha256": sha256_file(video_path),
        "input_video_file": os.path.basename(video_path),
        "pool": POOL_STRATEGY,
        "pool_size": list(POOL_SIZE),
        "layers": _layer_manifest_section(layer_cfg),
        "raw_activation_dim": int(raw_matrix.shape[1]),
        "num_vectors_fit": int(raw_matrix.shape[0]),
        "model_name": model_name,
        "batch_size": int(batch_size),
        "sample_rate": int(sample_rate),
        "total_sampled_frames": int(collection.total_sampled_frames),
        "frames_with_detections": int(collection.frames_with_detections),
        "total_detections": int(collection.total_detections),
        "artifacts": {
            "enriched_detections_json": os.path.basename(artifacts.enriched_json_path),
            "projection_manifest_json": os.path.basename(artifacts.manifest_path),
        },
    }
    save_json(artifacts.manifest_path, manifest)

    return {
        "enriched_detections": artifacts.enriched_json_path,
        "pca_projection": artifacts.pca_path,
        "projection_manifest": artifacts.manifest_path,
    }


def run_activation_enrichment(*args: Any, **kwargs: Any) -> Dict[str, str]:
    """Compatibility shim for the retired two-pass enrichment API."""
    raise RuntimeError(
        "run_activation_enrichment() is deprecated and was removed with the single-pass Phase 1 refactor. "
        "Use run_phase1_enrichment() or `python3 src/run_pipeline.py --video <path> --model yolov8n.pt --sample-rate 5`."
    )


def build_enrichment_parser() -> argparse.ArgumentParser:
    """Compatibility shim for the retired enrichment CLI parser."""
    raise RuntimeError(
        "The two-pass enrichment CLI is retired. Use `python3 src/run_pipeline.py --video <path> --model yolov8n.pt --sample-rate 5`."
    )

