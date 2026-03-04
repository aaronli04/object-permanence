"""YOLO/model-specific helpers for trace enrichment."""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Sequence

from .constants import POOL_SIZE
from .types import CollectedDetection

try:
    import numpy as np
    import torch
    from ultralytics import YOLO

    _IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    YOLO = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def ensure_model_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        missing = getattr(_IMPORT_ERROR, "name", "unknown")
        raise RuntimeError(
            "Trace enrichment runtime dependencies failed to import "
            f"(module={missing}). Install/repair requirements.txt environment. Original error: {_IMPORT_ERROR}"
        ) from _IMPORT_ERROR


def load_yolo(model_name: str):
    ensure_model_runtime_dependencies()
    return YOLO(model_name)


def get_module_map(yolo) -> dict[str, object]:
    return dict(yolo.model.model.named_modules())


def list_c2f_module_names(yolo) -> list[str]:
    return [
        name
        for name, module in yolo.model.model.named_modules()
        if module.__class__.__name__ == "C2f"
    ]


def _top_level_c2f_indices(yolo) -> list[int]:
    module_map = get_module_map(yolo)
    indices: list[int] = []
    for name, module in module_map.items():
        if not name.isdigit():
            continue
        if module.__class__.__name__ != "C2f":
            continue
        indices.append(int(name))
    indices.sort()
    return indices


def resolve_hook_layer_name(yolo, requested_layer: str) -> str:
    """Resolve user-facing layer identifiers to a concrete named module key."""
    module_map = get_module_map(yolo)
    requested = str(requested_layer).strip()
    if not requested:
        raise ValueError("requested_layer cannot be empty")

    if requested in module_map:
        return requested

    if requested.isdigit():
        normalized = str(int(requested))
        if normalized in module_map:
            return normalized

    bracket_match = re.fullmatch(r"model\.model\[(\d+)\]", requested)
    if bracket_match:
        idx = str(int(bracket_match.group(1)))
        if idx in module_map:
            return idx

    c2f_indices = _top_level_c2f_indices(yolo)
    if requested in {"neck.C2f.15", "neck_c2f_15", "neck.C2f.mid", "neck_c2f_mid"}:
        if 15 in c2f_indices:
            return "15"
        if len(c2f_indices) >= 3:
            return str(c2f_indices[-2])
        if c2f_indices:
            return str(c2f_indices[-1])

    c2f_names = list_c2f_module_names(yolo)
    available_hint = ", ".join(c2f_names[:12])
    if len(c2f_names) > 12:
        available_hint += ", ..."
    raise KeyError(
        f"Hook layer '{requested_layer}' not found. "
        f"Run discover_layers to inspect names. C2f candidates: {available_hint or '(none)'}"
    )


class FeatureHookCollector:
    """Capture feature maps from named modules during forward passes."""

    def __init__(self, module_map: dict[str, "torch.nn.Module"], layer_names: Sequence[str]) -> None:
        self.module_map = module_map
        self.layer_names = list(layer_names)
        self._handles: list[Any] = []
        self.outputs: dict[str, "torch.Tensor"] = {}

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (list, tuple)) else output
            self.outputs[name] = tensor

        return hook

    def register(self) -> None:
        for name in self.layer_names:
            if name not in self.module_map:
                raise KeyError(f"Hook layer '{name}' not found in model modules")
            self._handles.append(self.module_map[name].register_forward_hook(self._make_hook(name)))

    def clear(self) -> None:
        self.outputs.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "FeatureHookCollector":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


def extract_detections_from_result(result: Any, *, sort_by_confidence: bool = True) -> list[CollectedDetection]:
    detections: list[CollectedDetection] = []
    names = getattr(result, "names", None) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        cls_id = int(box.cls.item()) if getattr(box, "cls", None) is not None else -1
        conf = float(box.conf.item()) if getattr(box, "conf", None) is not None else 0.0
        xyxy = (
            [float(v) for v in box.xyxy[0].tolist()]
            if getattr(box, "xyxy", None) is not None
            else [0.0, 0.0, 0.0, 0.0]
        )
        detections.append(
            CollectedDetection(
                class_id=cls_id,
                class_name=names.get(cls_id, "unknown"),
                bbox=xyxy,
                confidence=conf,
            )
        )

    if sort_by_confidence:
        detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _bbox_to_feature_roi(
    bbox_xyxy: Sequence[float],
    frame_h: int,
    frame_w: int,
    fmap_h: int,
    fmap_w: int,
) -> tuple[int, int, int, int, bool]:
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError(f"Invalid frame shape for ROI mapping: h={frame_h}, w={frame_w}")

    x1, y1, x2, y2 = bbox_xyxy
    fx1 = int(math.floor((x1 / frame_w) * fmap_w))
    fy1 = int(math.floor((y1 / frame_h) * fmap_h))
    fx2 = int(math.ceil((x2 / frame_w) * fmap_w))
    fy2 = int(math.ceil((y2 / frame_h) * fmap_h))

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
    frame_h: int,
    frame_w: int,
    pool: "torch.nn.Module",
):
    if fmap.ndim != 3:
        raise ValueError(f"Expected feature map [C,H,W], got {tuple(fmap.shape)}")
    c, h, w = fmap.shape
    x1, y1, x2, y2, small = _bbox_to_feature_roi(bbox_xyxy, frame_h, frame_w, h, w)
    crop = fmap[:, y1:y2, x1:x2].unsqueeze(0)
    pooled = pool(crop).reshape(c * POOL_SIZE[0] * POOL_SIZE[1])
    vec = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
    return vec, small


def build_raw_activation_vector(
    fmap: "torch.Tensor",
    bbox_xyxy: Sequence[float],
    pool: "torch.nn.Module",
    frame_h: int,
    frame_w: int,
):
    vec, small = crop_and_pool_feature(fmap, bbox_xyxy, frame_h, frame_w, pool)
    return vec, small
