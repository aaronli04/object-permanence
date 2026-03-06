"""Similarity and score matrix construction for temporal linking."""

from __future__ import annotations

import math

import numpy as np

try:
    from common.numeric import l2_normalize
except ImportError:  # pragma: no cover - import-path compatibility
    from src.common.numeric import l2_normalize  # type: ignore

from .config import TemporalLinkingConfig
from .types import Detection, PairScores, Track


def _bbox_center(bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def _bbox_wh(bbox_xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def _bbox_diag(bbox_xyxy: np.ndarray) -> float:
    w, h = _bbox_wh(bbox_xyxy)
    return float(math.sqrt((w * w) + (h * h)))


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)
    union = (aw * ah) + (bw * bh) - inter

    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _history_mean(track: Track) -> np.ndarray:
    if not track.vec_history:
        return track.last_vec
    matrix = np.stack(list(track.vec_history), axis=0)
    return np.mean(matrix, axis=0).astype(np.float32, copy=False)


def build_reference_vector(track: Track, cfg: TemporalLinkingConfig) -> np.ndarray:
    hist_mean = _history_mean(track)
    mixed = (cfg.w_last * track.last_vec) + (cfg.w_ema * track.ema_vec) + (cfg.w_hist * hist_mean)
    return l2_normalize(mixed)


def _spatial_score(track: Track, det: Detection) -> float:
    iou = bbox_iou(track.last_bbox_xyxy, det.bbox_xyxy)
    center_delta = _bbox_center(track.last_bbox_xyxy) - _bbox_center(det.bbox_xyxy)
    center_dist = float(np.linalg.norm(center_delta))
    scale = max(_bbox_diag(track.last_bbox_xyxy), _bbox_diag(det.bbox_xyxy), 1.0)
    center_term = float(math.exp(-center_dist / scale))
    return float((0.5 * iou) + (0.5 * center_term))


def _pair_frame_dims(track: Track, det: Detection) -> tuple[float, float] | None:
    width = det.frame_width if det.frame_width is not None else track.frame_width
    height = det.frame_height if det.frame_height is not None else track.frame_height
    if width is None or height is None:
        return None
    width = float(width)
    height = float(height)
    if width <= 0.0 or height <= 0.0:
        return None
    return width, height


def _normalized_centroid_distance(track: Track, det: Detection) -> float | None:
    dims = _pair_frame_dims(track, det)
    if dims is None:
        return None
    width, height = dims
    center_delta = _bbox_center(track.last_bbox_xyxy) - _bbox_center(det.bbox_xyxy)
    dx = float(center_delta[0])
    dy = float(center_delta[1])
    return float(math.sqrt(((dx / width) ** 2) + ((dy / height) ** 2)) / math.sqrt(2.0))


def compute_pair_scores(
    tracks: list[Track],
    detections: list[Detection],
    cfg: TemporalLinkingConfig,
) -> PairScores:
    """Compute visual, tie-break, and assignment score matrices.

    Eligibility is determined by class policy, spatial plausibility, and visual similarity threshold.
    """
    n_tracks = len(tracks)
    n_dets = len(detections)

    visual = np.full((n_tracks, n_dets), np.float32(-np.inf), dtype=np.float32)
    spatial = np.zeros((n_tracks, n_dets), dtype=np.float32)
    tie_break = np.zeros((n_tracks, n_dets), dtype=np.float32)
    assignment = np.full((n_tracks, n_dets), np.float32(-np.inf), dtype=np.float32)
    eligible = np.zeros((n_tracks, n_dets), dtype=bool)

    if n_tracks == 0 or n_dets == 0:
        return PairScores(visual=visual, spatial=spatial, tie_break=tie_break, assignment=assignment, eligible=eligible)

    for i, track in enumerate(tracks):
        ref_vec = build_reference_vector(track, cfg)
        consistency = float(np.mean(track.sim_history)) if track.sim_history else None
        age_decay = float(math.exp(-float(track.miss_streak) / float(max(cfg.max_lost_frames, 1))))

        for j, det in enumerate(detections):
            if cfg.match_within_class and track.class_id != det.class_id:
                continue

            centroid_distance = _normalized_centroid_distance(track, det)
            if centroid_distance is not None and centroid_distance > cfg.max_centroid_distance:
                visual[i, j] = np.float32(0.0)
                spatial[i, j] = np.float32(0.0)
                tie_break[i, j] = np.float32(0.0)
                eligible[i, j] = False
                continue

            vis = float(np.dot(ref_vec, det.activation_vec))
            sp = _spatial_score(track, det)
            cons = consistency if consistency is not None else vis
            tie = (cfg.w_spatial * sp) + (cfg.w_consistency * cons) + (cfg.w_age * age_decay)

            is_eligible = vis >= cfg.similarity_threshold

            visual[i, j] = np.float32(vis)
            spatial[i, j] = np.float32(sp)
            tie_break[i, j] = np.float32(tie)
            eligible[i, j] = bool(is_eligible)
            if is_eligible:
                assignment[i, j] = np.float32(vis + tie)

    return PairScores(visual=visual, spatial=spatial, tie_break=tie_break, assignment=assignment, eligible=eligible)
