"""Assignment solvers for frame-to-frame linking."""

from __future__ import annotations

from typing import Final

import numpy as np

from .config import TemporalLinkingConfig
from .similarity import compute_pair_scores
from .types import Assignment, Detection, Track

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None  # type: ignore[assignment]


_INVALID_COST: Final[float] = 1e9


def _hungarian_pairs(score: np.ndarray, eligible: np.ndarray) -> list[tuple[int, int]]:
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is unavailable for Hungarian assignment")

    if not eligible.any():
        return []

    cost = np.where(eligible, -score, _INVALID_COST).astype(np.float64, copy=False)
    rows, cols = linear_sum_assignment(cost)

    pairs: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if eligible[r, c]:
            pairs.append((r, c))
    return pairs


def _greedy_pairs(score: np.ndarray, eligible: np.ndarray) -> list[tuple[int, int]]:
    if not eligible.any():
        return []

    n_tracks, n_dets = score.shape
    candidates: list[tuple[float, int, int]] = []
    for i in range(n_tracks):
        for j in range(n_dets):
            if not eligible[i, j]:
                continue
            candidates.append((float(score[i, j]), i, j))

    candidates.sort(key=lambda item: item[0], reverse=True)

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for _score, i, j in candidates:
        if i in used_tracks or j in used_dets:
            continue
        used_tracks.add(i)
        used_dets.add(j)
        pairs.append((i, j))

    return pairs


def solve_pairs(
    score: np.ndarray,
    eligible: np.ndarray,
    method: str,
) -> list[tuple[int, int]]:
    if method == "hungarian":
        if linear_sum_assignment is not None:
            return _hungarian_pairs(score, eligible)
        return _greedy_pairs(score, eligible)

    if method == "greedy":
        return _greedy_pairs(score, eligible)

    raise ValueError(f"Unsupported assignment method: {method}")


def assign_frame(
    tracks: list[Track],
    detections: list[Detection],
    cfg: TemporalLinkingConfig,
) -> list[Assignment]:
    """Assign tracks to detections for one frame."""
    if not tracks or not detections:
        return []

    scores = compute_pair_scores(tracks, detections, cfg)
    pairs = solve_pairs(scores.assignment, scores.eligible, cfg.assignment_method)

    assignments: list[Assignment] = []
    for track_idx, det_idx in pairs:
        track = tracks[track_idx]
        assignments.append(
            Assignment(
                track_id=track.track_id,
                det_index=int(det_idx),
                visual_similarity=float(scores.visual[track_idx, det_idx]),
                spatial_score=float(scores.spatial[track_idx, det_idx]),
                total_score=float(scores.assignment[track_idx, det_idx]),
                source_track_status=track.status,
            )
        )

    assignments.sort(key=lambda a: a.total_score, reverse=True)
    return assignments
