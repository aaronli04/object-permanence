"""Shared numeric/vector helpers."""

from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector, returning zeros for degenerate inputs."""
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32, copy=False)


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return (matrix / safe_norms).astype(np.float32, copy=False)


def topk_l2_renorm_pad(vec: np.ndarray, *, topk: int, target_dim: int) -> np.ndarray:
    """Keep first K dims, L2-renormalize, and zero-pad to target_dim."""
    if target_dim <= 0:
        raise ValueError("target_dim must be > 0")
    k = min(int(topk), int(vec.shape[0]), int(target_dim))
    if k <= 0:
        raise ValueError("topk must be > 0")

    sliced = l2_normalize(vec[:k].astype(np.float32, copy=False))
    out = np.zeros((int(target_dim),), dtype=np.float32)
    out[:k] = sliced
    return out
