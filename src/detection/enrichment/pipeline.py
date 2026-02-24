"""Phase 1 post-processing enrichment with YOLOv8 backbone activations."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
from .introspection import get_module_map, print_c2f_candidates

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
IOU_THRESHOLD = 0.5
SCHEMA_VERSION = "activation_enrichment_v1"


def _ensure_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        missing = getattr(_IMPORT_ERROR, "name", "unknown")
        raise RuntimeError(
            f"Missing dependency '{missing}'. Install requirements.txt before running activation enrichment."
        ) from _IMPORT_ERROR


@dataclass
class MatchedDetection:
    """Link an input JSON detection to a YOLO rerun detection index."""

    input_det: Dict[str, Any]
    yolo_det_index: Optional[int]


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
    """Raw activation vector collected for one input detection."""

    frame_num: int
    det_index: int
    raw_vector: "np.ndarray"
    small_crop_flag: bool
    match_found: bool


class VideoFrameReader:
    """Random-access frame reader for a video file."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

    def read_frame(self, frame_num: int) -> "cv2.Mat":
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = self.cap.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_num} from {self.video_path}")
        return frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class FeatureHookCollector:
    """Capture feature maps from named modules during forward passes."""

    def __init__(self, module_map: Dict[str, torch.nn.Module], layer_names: Sequence[str]) -> None:
        self.module_map = module_map
        self.layer_names = list(layer_names)
        self._handles: List[Any] = []
        self.outputs: Dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (list, tuple)):
                tensor = output[0]
            else:
                tensor = output
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


def load_detection_trace(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input detections JSON must be a list of frame records")
    return data


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def xyxy_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def extract_yolo_boxes(result) -> List[Dict[str, Any]]:
    boxes = []
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        cls_id = int(box.cls.item()) if box.cls is not None else -1
        conf = float(box.conf.item()) if box.conf is not None else 0.0
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        boxes.append({"class_id": cls_id, "confidence": conf, "bbox": xyxy})
    return boxes


def match_input_to_yolo(input_dets: List[Dict[str, Any]], yolo_dets: List[Dict[str, Any]]) -> List[MatchedDetection]:
    matched: List[MatchedDetection] = []
    used_indices: set[int] = set()

    for det in input_dets:
        best_idx: Optional[int] = None
        best_iou = 0.0
        for idx, ydet in enumerate(yolo_dets):
            if idx in used_indices:
                continue
            if int(ydet["class_id"]) != int(det["class_id"]):
                continue
            iou = xyxy_iou(det["bbox"], ydet["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx is not None and best_iou >= IOU_THRESHOLD:
            used_indices.add(best_idx)
            matched.append(MatchedDetection(input_det=det, yolo_det_index=best_idx))
        else:
            matched.append(MatchedDetection(input_det=det, yolo_det_index=None))

    return matched


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
    fmap: torch.Tensor,
    bbox_xyxy: Sequence[float],
    stride: int,
    pool: torch.nn.Module,
) -> Tuple[np.ndarray, bool]:
    if fmap.ndim != 3:
        raise ValueError(f"Expected feature map [C,H,W], got {tuple(fmap.shape)}")
    c, h, w = fmap.shape
    x1, y1, x2, y2, small = _bbox_to_feature_roi(bbox_xyxy, stride, h, w)
    crop = fmap[:, y1:y2, x1:x2].unsqueeze(0)
    pooled = pool(crop).reshape(c * POOL_SIZE[0] * POOL_SIZE[1])
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False), small


def build_raw_activation_vector(
    fmap_mid: torch.Tensor,
    fmap_deep: torch.Tensor,
    bbox_xyxy: Sequence[float],
    pool: torch.nn.Module,
    stride_mid: int,
    stride_deep: int,
) -> Tuple[np.ndarray, bool]:
    mid_vec, mid_small = crop_and_pool_feature(fmap_mid, bbox_xyxy, stride_mid, pool)
    deep_vec, deep_small = crop_and_pool_feature(fmap_deep, bbox_xyxy, stride_deep, pool)
    return np.concatenate([deep_vec, mid_vec], axis=0), (mid_small or deep_small)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        return np.zeros_like(vec)
    return vec / norm


def _frame_batches(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _sort_frame_detections_by_conf(frame_record: Dict[str, Any]) -> None:
    frame_record["detections"] = sorted(
        frame_record.get("detections", []),
        key=lambda d: float(d.get("confidence", 0.0)),
        reverse=True,
    )


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


def _normalize_trace_in_place(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for frame in trace:
        frame.setdefault("detections", [])
        _sort_frame_detections_by_conf(frame)
    return sorted(trace, key=lambda f: int(f["frame_num"]))


def _empty_enriched_payload(frame_order: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"frame_num": int(f["frame_num"]), "detections": []} for f in frame_order]


def _pad_and_normalize_projection(vec: "np.ndarray", source_dim: int, target_dim: int) -> "np.ndarray":
    if source_dim < target_dim:
        padded = np.zeros((target_dim,), dtype=np.float32)
        padded[:source_dim] = vec
        vec = padded
    return _l2_normalize(vec)


def _collect_raw_activation_records(
    *,
    yolo: "YOLO",
    frame_order: List[Dict[str, Any]],
    video_path: str,
    module_map: Dict[str, torch.nn.Module],
    layer_cfg: LayerHookConfig,
    batch_size: int,
    pool: torch.nn.Module,
) -> List[RawActivationRecord]:
    records: List[RawActivationRecord] = []
    with VideoFrameReader(video_path) as frame_reader, FeatureHookCollector(
        module_map=module_map, layer_names=[layer_cfg.deep_name, layer_cfg.mid_name]
    ) as hooks:
        for batch in _frame_batches(frame_order, batch_size):
            frames = [frame_reader.read_frame(int(fr["frame_num"])) for fr in batch]
            hooks.clear()
            results = yolo(frames, verbose=False)

            if layer_cfg.deep_name not in hooks.outputs or layer_cfg.mid_name not in hooks.outputs:
                raise RuntimeError("Expected hook outputs were not captured during YOLO forward pass")

            deep_batch = hooks.outputs[layer_cfg.deep_name].detach().cpu()
            mid_batch = hooks.outputs[layer_cfg.mid_name].detach().cpu()
            if deep_batch.shape[0] != len(batch) or mid_batch.shape[0] != len(batch):
                raise RuntimeError("Hook output batch size does not match inference batch size")

            for i, frame_record in enumerate(batch):
                input_dets = frame_record.get("detections", [])
                yolo_dets = extract_yolo_boxes(results[i])
                for det_idx, match in enumerate(match_input_to_yolo(input_dets, yolo_dets)):
                    raw_vec, small_crop_flag = build_raw_activation_vector(
                        fmap_mid=mid_batch[i],
                        fmap_deep=deep_batch[i],
                        bbox_xyxy=match.input_det["bbox"],
                        pool=pool,
                        stride_mid=layer_cfg.mid_stride,
                        stride_deep=layer_cfg.deep_stride,
                    )
                    records.append(
                        RawActivationRecord(
                            frame_num=int(frame_record["frame_num"]),
                            det_index=det_idx,
                            raw_vector=raw_vec,
                            small_crop_flag=bool(small_crop_flag),
                            match_found=match.yolo_det_index is not None,
                        )
                    )
    return records


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


def run_activation_enrichment(
    *,
    input_json_path: str,
    video_path: str,
    model_name: str,
    output_dir: str,
    deep_layer_name: str = "8",
    mid_layer_name: str = "6",
    stride_deep: int = 32,
    stride_mid: int = 16,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pca_dim: int = OUTPUT_VECTOR_DIM,
) -> Dict[str, str]:
    """Run the two-pass enrichment pipeline and write artifacts."""
    _ensure_runtime_dependencies()

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    trace = load_detection_trace(input_json_path)
    frame_order = _normalize_trace_in_place(trace)

    os.makedirs(output_dir, exist_ok=True)
    artifacts = _build_artifact_paths(output_dir)
    layer_cfg = LayerHookConfig(
        deep_name=deep_layer_name,
        mid_name=mid_layer_name,
        deep_stride=stride_deep,
        mid_stride=stride_mid,
    )

    yolo = YOLO(model_name)
    print_c2f_candidates(yolo)
    print(
        f"Selected hook layers: deep='{layer_cfg.deep_name}' (stride={layer_cfg.deep_stride}), "
        f"mid='{layer_cfg.mid_name}' (stride={layer_cfg.mid_stride})"
    )

    module_map = get_module_map(yolo)
    pool = torch.nn.AdaptiveAvgPool2d(POOL_SIZE)

    total_detections = sum(len(frame.get("detections", [])) for frame in frame_order)
    if total_detections == 0:
        save_json(artifacts.enriched_json_path, _empty_enriched_payload(frame_order))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "projection_file": os.path.basename(artifacts.pca_path),
            "projection_dim": pca_dim,
            "fit_timestamp_utc": None,
            "input_file_hash_sha256": sha256_file(input_json_path),
            "pool": POOL_STRATEGY,
            "pool_size": list(POOL_SIZE),
            "layers": _layer_manifest_section(layer_cfg),
            "note": "No detections in input trace; PCA was not fitted.",
        }
        save_json(artifacts.manifest_path, manifest)
        return {
            "enriched_detections": artifacts.enriched_json_path,
            "pca_projection": artifacts.pca_path,
            "projection_manifest": artifacts.manifest_path,
        }

    # Pass 1: collect raw activation vectors for all detections.
    raw_records = _collect_raw_activation_records(
        yolo=yolo,
        frame_order=frame_order,
        video_path=video_path,
        module_map=module_map,
        layer_cfg=layer_cfg,
        batch_size=batch_size,
        pool=pool,
    )

    raw_matrix = np.stack([r.raw_vector for r in raw_records], axis=0)
    effective_pca_dim = min(pca_dim, raw_matrix.shape[0], raw_matrix.shape[1])
    if effective_pca_dim <= 0:
        raise RuntimeError("No activation vectors collected for PCA")

    pca = PCA(n_components=effective_pca_dim, svd_solver="auto", random_state=0)
    projected = pca.fit_transform(raw_matrix).astype(np.float32, copy=False)
    joblib.dump(pca, artifacts.pca_path)

    # Pass 2: write enriched JSON using projected + normalized vectors.
    projected_index = _build_projected_index(raw_records, projected, effective_pca_dim, pca_dim)
    enriched_payload = _build_enriched_payload(frame_order, projected_index, pca_dim)
    save_json(artifacts.enriched_json_path, enriched_payload)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "projection_file": os.path.basename(artifacts.pca_path),
        "projection_dim": int(pca_dim),
        "fitted_pca_components": int(effective_pca_dim),
        "fit_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_file_hash_sha256": sha256_file(input_json_path),
        "input_json_file": os.path.basename(input_json_path),
        "pool": POOL_STRATEGY,
        "pool_size": list(POOL_SIZE),
        "layers": _layer_manifest_section(layer_cfg),
        "raw_activation_dim": int(raw_matrix.shape[1]),
        "num_vectors_fit": int(raw_matrix.shape[0]),
        "model_name": model_name,
        "batch_size": int(batch_size),
        "iou_match_threshold": IOU_THRESHOLD,
        "unmatched_input_detections": int(sum(1 for r in raw_records if not r.match_found)),
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


def build_enrichment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich YOLO detections with backbone activations + PCA.")
    parser.add_argument("--input-json", required=True, help="Input detections JSON trace.")
    parser.add_argument("--video", required=True, help="Source video used to regenerate frames.")
    parser.add_argument("--model", required=True, help="Local YOLOv8 weights path or model identifier.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for enriched_detections.json, pca_projection.pkl, projection_manifest.json.",
    )
    parser.add_argument("--deep-layer", default="8", help="Named module for deep backbone C2f (default: 8 for YOLOv8m).")
    parser.add_argument("--mid-layer", default="6", help="Named module for mid backbone C2f (default: 6 for YOLOv8m).")
    parser.add_argument("--deep-stride", type=int, default=32, help="Feature stride for deep hook layer.")
    parser.add_argument("--mid-stride", type=int, default=16, help="Feature stride for mid hook layer.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size (default: 8).")
    parser.add_argument("--pca-dim", type=int, default=OUTPUT_VECTOR_DIM, help="Target PCA dimension (default: 256).")
    return parser
