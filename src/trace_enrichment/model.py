"""YOLO/model-specific helpers for trace enrichment."""

from __future__ import annotations

import math
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
    stride: int,
    fmap_h: int,
    fmap_w: int,
) -> tuple[int, int, int, int, bool]:
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
):
    if fmap.ndim != 3:
        raise ValueError(f"Expected feature map [C,H,W], got {tuple(fmap.shape)}")
    c, h, w = fmap.shape
    x1, y1, x2, y2, small = _bbox_to_feature_roi(bbox_xyxy, stride, h, w)
    crop = fmap[:, y1:y2, x1:x2].unsqueeze(0)
    pooled = pool(crop).reshape(c * POOL_SIZE[0] * POOL_SIZE[1])
    vec = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
    return vec, small


def build_raw_activation_vector(
    fmap_mid: "torch.Tensor",
    fmap_deep: "torch.Tensor",
    bbox_xyxy: Sequence[float],
    pool: "torch.nn.Module",
    stride_mid: int,
    stride_deep: int,
):
    mid_vec, mid_small = crop_and_pool_feature(fmap_mid, bbox_xyxy, stride_mid, pool)
    deep_vec, deep_small = crop_and_pool_feature(fmap_deep, bbox_xyxy, stride_deep, pool)
    return np.concatenate([deep_vec, mid_vec], axis=0), (mid_small or deep_small)
