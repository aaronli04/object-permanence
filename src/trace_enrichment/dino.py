"""DINO feature extraction helpers for trace enrichment."""

from __future__ import annotations

import math
import socket
import threading
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from common.numeric import l2_normalize

try:
    import cv2
    import numpy as np
    import torch

    _IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover - runtime dependency guard
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


class DinoUnavailableError(RuntimeError):
    """Raised when DINO cannot be loaded or used."""


@dataclass(frozen=True)
class DinoEmbeddingResult:
    vector: "np.ndarray"
    valid_crop: bool
    tiny_crop: bool


_MODEL_CACHE: dict[tuple[str, str, str], "DinoEmbedder"] = {}
_CACHE_LOCK = threading.Lock()


def ensure_dino_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        missing = getattr(_IMPORT_ERROR, "name", "unknown")
        raise DinoUnavailableError(
            "DINO runtime dependencies failed to import "
            f"(module={missing}). Install/repair requirements.txt environment. Original error: {_IMPORT_ERROR}"
        ) from _IMPORT_ERROR


def _resolve_device(preferred_device: str | None) -> "torch.device":
    if preferred_device:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _padded_crop_box(
    *,
    bbox_xyxy: Sequence[float],
    frame_h: int,
    frame_w: int,
    pad_ratio: float,
) -> tuple[int, int, int, int, bool]:
    if frame_h <= 0 or frame_w <= 0:
        return 0, 0, 0, 0, False
    if len(bbox_xyxy) != 4:
        return 0, 0, 0, 0, False

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0.0 or height <= 0.0:
        return 0, 0, 0, 0, False

    pad = float(max(width, height) * max(0.0, float(pad_ratio)))
    crop_x1 = max(0, int(math.floor(x1 - pad)))
    crop_y1 = max(0, int(math.floor(y1 - pad)))
    crop_x2 = min(int(frame_w), int(math.ceil(x2 + pad)))
    crop_y2 = min(int(frame_h), int(math.ceil(y2 + pad)))
    valid = bool(crop_x2 > crop_x1 and crop_y2 > crop_y1)
    return crop_x1, crop_y1, crop_x2, crop_y2, valid


def _zero_vector(dim: int) -> "np.ndarray":
    return np.zeros((int(dim),), dtype=np.float32)


class DinoEmbedder:
    def __init__(
        self,
        *,
        model: Any,
        device: "torch.device",
        model_name: str,
        feature_dim: int,
        input_size: int,
        tiny_crop_min: int,
        crop_padding_ratio: float,
    ) -> None:
        self.model = model
        self.device = device
        self.model_name = model_name
        self.feature_dim = int(feature_dim)
        self.input_size = int(input_size)
        self.tiny_crop_min = int(tiny_crop_min)
        self.crop_padding_ratio = float(crop_padding_ratio)
        self._mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def extract(self, *, frame_bgr: "np.ndarray", bbox_xyxy: Sequence[float]) -> DinoEmbeddingResult:
        frame_h = int(frame_bgr.shape[0]) if hasattr(frame_bgr, "shape") and len(frame_bgr.shape) >= 2 else 0
        frame_w = int(frame_bgr.shape[1]) if hasattr(frame_bgr, "shape") and len(frame_bgr.shape) >= 2 else 0
        x1, y1, x2, y2, valid = _padded_crop_box(
            bbox_xyxy=bbox_xyxy,
            frame_h=frame_h,
            frame_w=frame_w,
            pad_ratio=self.crop_padding_ratio,
        )
        if not valid:
            return DinoEmbeddingResult(vector=_zero_vector(self.feature_dim), valid_crop=False, tiny_crop=False)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop is None or int(getattr(crop, "size", 0)) <= 0:
            return DinoEmbeddingResult(vector=_zero_vector(self.feature_dim), valid_crop=False, tiny_crop=False)

        crop_h = int(crop.shape[0])
        crop_w = int(crop.shape[1])
        tiny_crop = bool(crop_h < self.tiny_crop_min or crop_w < self.tiny_crop_min)
        if tiny_crop:
            return DinoEmbeddingResult(vector=_zero_vector(self.feature_dim), valid_crop=True, tiny_crop=True)
        resized = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32, copy=False) / 255.0
        normalized = (rgb - self._mean) / self._std
        chw = np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, (list, tuple)):
            if not output:
                raise RuntimeError("DINO model returned an empty output tuple")
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"DINO model returned unsupported output type: {type(output)!r}")

        if output.ndim == 1:
            vec_tensor = output
        elif output.ndim >= 2:
            flattened = output.reshape(int(output.shape[0]), -1)
            if int(flattened.shape[0]) <= 0:
                raise RuntimeError("DINO model output batch is empty")
            vec_tensor = flattened[0]
        else:
            raise RuntimeError(f"DINO model returned invalid tensor shape: {tuple(output.shape)}")

        vec = vec_tensor.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        if int(vec.shape[0]) != self.feature_dim:
            raise RuntimeError(
                f"DINO feature dimension mismatch: expected {self.feature_dim}, got {int(vec.shape[0])}"
            )
        return DinoEmbeddingResult(vector=l2_normalize(vec), valid_crop=True, tiny_crop=tiny_crop)


def load_dino_embedder(
    *,
    model_name: str,
    feature_dim: int,
    hub_repo: str,
    preferred_device: str | None = None,
    load_timeout_seconds: float = 20.0,
    input_size: int = 224,
    tiny_crop_min: int = 32,
    crop_padding_ratio: float = 0.05,
) -> DinoEmbedder:
    ensure_dino_runtime_dependencies()
    device = _resolve_device(preferred_device)
    cache_key = (hub_repo, model_name, str(device))
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    prior_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(float(load_timeout_seconds))
    try:
        model = torch.hub.load(hub_repo, model_name)
    except Exception as exc:
        raise DinoUnavailableError(
            "Failed to load DINO model via torch.hub "
            f"(repo={hub_repo}, model={model_name}, device={device}). "
            f"This can happen when internet is unavailable and weights are not cached. Error: {exc}"
        ) from exc
    finally:
        socket.setdefaulttimeout(prior_timeout)

    try:
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    except Exception as exc:
        raise DinoUnavailableError(f"DINO model initialization failed: {exc}") from exc

    embedder = DinoEmbedder(
        model=model,
        device=device,
        model_name=model_name,
        feature_dim=feature_dim,
        input_size=input_size,
        tiny_crop_min=tiny_crop_min,
        crop_padding_ratio=crop_padding_ratio,
    )
    with _CACHE_LOCK:
        _MODEL_CACHE[cache_key] = embedder
    return embedder


def extract_dino_embedding(
    *,
    frame_bgr: "np.ndarray",
    bbox_xyxy: Sequence[float],
    embedder: DinoEmbedder,
) -> DinoEmbeddingResult:
    return embedder.extract(frame_bgr=frame_bgr, bbox_xyxy=bbox_xyxy)
