"""Post-hoc relinking pass for closed track fragments."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.numeric import l2_normalize

from .config import TemporalLinkingConfig
from .types import RelinkEdge, Track, TrackFragment, TrackStatus


def _empty_gallery() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float32)


def _build_dino_gallery(track: Track, gallery_size: int) -> np.ndarray:
    if gallery_size <= 0:
        return _empty_gallery()
    if not track.obs_dino_samples:
        return _empty_gallery()

    ordered_samples = sorted(track.obs_dino_samples, key=lambda sample: (sample.frame_num, -sample.confidence))
    if len(ordered_samples) <= gallery_size:
        return np.stack([sample.vector for sample in ordered_samples], axis=0).astype(np.float32, copy=False)

    chosen: list[np.ndarray] = []
    sample_count = len(ordered_samples)
    for idx in range(gallery_size):
        start = int(np.floor((idx * sample_count) / gallery_size))
        end = int(np.floor(((idx + 1) * sample_count) / gallery_size))
        end = max(end, start + 1)
        bucket = ordered_samples[start:end]
        if not bucket:
            continue
        chosen_sample = max(bucket, key=lambda sample: (sample.confidence, -sample.frame_num))
        chosen.append(chosen_sample.vector)

    if not chosen:
        return _empty_gallery()
    return np.stack(chosen, axis=0).astype(np.float32, copy=False)


def _gallery_similarity(gallery_a: np.ndarray, gallery_b: np.ndarray, *, topk: int) -> float:
    if gallery_a.ndim != 2 or gallery_b.ndim != 2:
        return 0.0
    if int(gallery_a.shape[0]) <= 0 or int(gallery_b.shape[0]) <= 0:
        return 0.0

    pairwise = np.matmul(gallery_a, gallery_b.T).reshape(-1)
    if int(pairwise.shape[0]) <= 0:
        return 0.0

    k = max(1, min(int(topk), int(pairwise.shape[0])))
    top_scores = np.sort(pairwise)[-k:]
    return float(np.mean(top_scores))


def build_fragments(tracks: list[Track], min_hits: int, dino_gallery_size: int = 20) -> list[TrackFragment]:
    """Build relink-eligible fragments from closed tracks."""
    fragments: list[TrackFragment] = []
    for track in tracks:
        if track.status != TrackStatus.CLOSED:
            continue
        if track.hits < min_hits:
            continue
        if not track.obs_vecs:
            continue
        if not track.obs_positions:
            continue

        frame_vecs = np.stack([l2_normalize(v) for v in track.obs_vecs], axis=0).astype(np.float32, copy=False)
        centroid = l2_normalize(np.mean(frame_vecs, axis=0))
        positions = list(track.obs_positions)
        dino_gallery = _build_dino_gallery(track, dino_gallery_size)
        dino_vec = None
        if track.dino_vector is not None:
            dino_vec = l2_normalize(np.asarray(track.dino_vector, dtype=np.float32))
            if float(np.linalg.norm(dino_vec)) <= 0.0:
                dino_vec = None
        fragments.append(
            TrackFragment(
                track_id=int(track.track_id),
                class_id=int(track.class_id),
                first_frame=int(track.start_frame),
                last_frame=int(track.last_seen_frame),
                hits=int(track.hits),
                centroid=centroid,
                frame_vecs=frame_vecs,
                last_positions=positions[-3:],
                first_position=positions[0],
                dino_gallery=dino_gallery,
                dino_vector=dino_vec,
            )
        )

    fragments.sort(key=lambda f: (f.first_frame, f.track_id))
    return fragments


def build_candidates(
    fragments: list[TrackFragment],
    max_gap_frames: int,
) -> list[tuple[TrackFragment, TrackFragment]]:
    """Enumerate temporally ordered same-class fragment pairs."""
    candidates: list[tuple[TrackFragment, TrackFragment]] = []
    for i, pred in enumerate(fragments):
        for j, succ in enumerate(fragments):
            if i == j:
                continue
            if pred.class_id != succ.class_id:
                continue
            if pred.last_frame >= succ.first_frame:
                continue

            gap = succ.first_frame - pred.last_frame
            if max_gap_frames != -1 and gap > max_gap_frames:
                continue
            candidates.append((pred, succ))
    return candidates


def score_centroid(candidates: list[tuple[TrackFragment, TrackFragment]]) -> list[RelinkEdge]:
    """YOLO centroid cosine score for each candidate pair."""
    edges: list[RelinkEdge] = []
    for pred, succ in candidates:
        score = float(np.dot(pred.centroid, succ.centroid))
        edges.append(
            RelinkEdge(
                predecessor_id=pred.track_id,
                successor_id=succ.track_id,
                score=score,
                method="yolo",
            )
        )
    return edges


def score_identity(
    candidates: list[tuple[TrackFragment, TrackFragment]],
    *,
    relink_use_dino: bool,
    relink_dino_min_detections: int = 2,
    relink_dino_gallery_topk: int = 3,
) -> tuple[list[RelinkEdge], float]:
    """Score each candidate using DINO when available, else YOLO centroid."""
    edges: list[RelinkEdge] = []
    dino_scored = 0

    for pred, succ in candidates:
        use_dino = bool(
            relink_use_dino
            and pred.dino_gallery.ndim == 2
            and succ.dino_gallery.ndim == 2
            and int(pred.dino_gallery.shape[0]) >= int(relink_dino_min_detections)
            and int(succ.dino_gallery.shape[0]) >= int(relink_dino_min_detections)
        )
        if use_dino:
            score = _gallery_similarity(
                pred.dino_gallery,
                succ.dino_gallery,
                topk=relink_dino_gallery_topk,
            )
            method: str = "dino"
            dino_scored += 1
        else:
            score = float(np.dot(pred.centroid, succ.centroid))
            method = "yolo"
        edges.append(
            RelinkEdge(
                predecessor_id=pred.track_id,
                successor_id=succ.track_id,
                score=score,
                method=method,
            )
        )

    coverage = (float(dino_scored) / float(len(candidates))) if candidates else 0.0
    return edges, coverage


def score_fallback(
    candidates: list[tuple[TrackFragment, TrackFragment]],
    max_pixels_per_frame: float,
) -> list[RelinkEdge]:
    """Spatial plausibility score for each candidate pair."""
    edges: list[RelinkEdge] = []
    for pred, succ in candidates:
        score = _spatial_plausibility_score(pred, succ, max_pixels_per_frame)
        edges.append(
            RelinkEdge(
                predecessor_id=pred.track_id,
                successor_id=succ.track_id,
                score=score,
                method="spatial",
            )
        )
    return edges


def _spatial_plausibility_score(
    frag_a: TrackFragment,
    frag_b: TrackFragment,
    max_pixels_per_frame: float,
) -> float:
    pts = frag_a.last_positions
    if len(pts) >= 2:
        frames = np.asarray([p[2] for p in pts], dtype=np.float64)
        xs = np.asarray([p[0] for p in pts], dtype=np.float64)
        ys = np.asarray([p[1] for p in pts], dtype=np.float64)
        vx = float(np.polyfit(frames, xs, 1)[0])
        vy = float(np.polyfit(frames, ys, 1)[0])
    else:
        vx = 0.0
        vy = 0.0

    last_cx, last_cy, last_frame = frag_a.last_positions[-1]
    gap = max(1, int(frag_b.first_frame - last_frame))
    pred_cx = float(last_cx) + (vx * float(gap))
    pred_cy = float(last_cy) + (vy * float(gap))

    actual_cx, actual_cy, _ = frag_b.first_position
    pixel_error = float(np.hypot(pred_cx - float(actual_cx), pred_cy - float(actual_cy)))
    pixels_per_frame_error = pixel_error / float(gap)
    return 1.0 - (pixels_per_frame_error / float(max_pixels_per_frame))


def greedy_assign(
    identity_edges: list[RelinkEdge],
    fallback_edges: list[RelinkEdge],
    yolo_threshold: float,
    dino_threshold: float,
    fallback_threshold: float,
) -> list[RelinkEdge]:
    """Accept one-to-one edges: identity pass first, then fallback for unresolved nodes."""
    accepted: list[RelinkEdge] = []
    used_as_predecessor: set[int] = set()
    used_as_successor: set[int] = set()

    sorted_identity = sorted(
        identity_edges,
        key=lambda edge: (-edge.score, edge.predecessor_id, edge.successor_id),
    )
    for edge in sorted_identity:
        threshold = dino_threshold if edge.method == "dino" else yolo_threshold
        if edge.score < threshold:
            continue
        if edge.predecessor_id in used_as_predecessor:
            continue
        if edge.successor_id in used_as_successor:
            continue
        accepted.append(edge)
        used_as_predecessor.add(edge.predecessor_id)
        used_as_successor.add(edge.successor_id)

    sorted_fallback = sorted(
        fallback_edges,
        key=lambda edge: (-edge.score, edge.predecessor_id, edge.successor_id),
    )
    for edge in sorted_fallback:
        if edge.score < fallback_threshold:
            continue
        if edge.predecessor_id in used_as_predecessor:
            continue
        if edge.successor_id in used_as_successor:
            continue
        accepted.append(edge)
        used_as_predecessor.add(edge.predecessor_id)
        used_as_successor.add(edge.successor_id)

    return accepted


def resolve_chains(
    accepted: list[RelinkEdge],
    fragments: list[TrackFragment],
) -> dict[int, int]:
    """Resolve accepted links into canonical chains using union-find."""
    if not accepted:
        return {}

    first_frame_by_id = {fragment.track_id: fragment.first_frame for fragment in fragments}

    parent: dict[int, int] = {}

    def find(node: int) -> int:
        if node not in parent:
            parent[node] = node
            return node
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        parent[rb] = ra

    for edge in accepted:
        union(edge.predecessor_id, edge.successor_id)

    groups: dict[int, list[int]] = {}
    for edge in accepted:
        for track_id in (edge.predecessor_id, edge.successor_id):
            root = find(track_id)
            groups.setdefault(root, []).append(track_id)

    merge_map: dict[int, int] = {}
    for members in groups.values():
        uniq_members = sorted(set(members))
        canonical = min(
            uniq_members,
            key=lambda track_id: (first_frame_by_id.get(track_id, int(1e9)), track_id),
        )
        for track_id in uniq_members:
            if track_id != canonical:
                merge_map[track_id] = canonical
    return merge_map


def run_relink(
    tracks: list[Track],
    cfg: TemporalLinkingConfig,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Execute full relink pass and return merge map + stats payload."""
    num_closed_tracks = sum(1 for track in tracks if track.status == TrackStatus.CLOSED)
    fragments = build_fragments(
        tracks,
        cfg.relink_min_track_hits,
        dino_gallery_size=cfg.relink_dino_gallery_size,
    )
    candidates = build_candidates(fragments, cfg.relink_max_gap_frames)

    identity_edges, dino_coverage = score_identity(
        candidates,
        relink_use_dino=bool(cfg.relink_use_dino),
        relink_dino_min_detections=int(cfg.relink_dino_min_detections),
        relink_dino_gallery_topk=int(cfg.relink_dino_gallery_topk),
    )
    fallback_edges = score_fallback(candidates, cfg.relink_max_pixels_per_frame)

    yolo_threshold = float(cfg.relink_threshold)
    dino_threshold = float(cfg.relink_dino_threshold)
    fallback_threshold = float(cfg.relink_fallback_threshold)
    # Treat 1.0 as an explicit no-merge setting for each pass.
    if yolo_threshold >= 1.0:
        yolo_threshold = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
    if dino_threshold >= 1.0:
        dino_threshold = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
    if fallback_threshold >= 1.0:
        fallback_threshold = float(np.nextafter(np.float32(1.0), np.float32(2.0)))

    accepted_edges = greedy_assign(
        identity_edges=identity_edges,
        fallback_edges=fallback_edges,
        yolo_threshold=yolo_threshold,
        dino_threshold=dino_threshold,
        fallback_threshold=fallback_threshold,
    )
    merge_map = resolve_chains(accepted_edges, fragments)

    yolo_edges_above = sum(
        1 for edge in identity_edges if edge.method == "yolo" and edge.score >= yolo_threshold
    )
    dino_edges_above = sum(
        1 for edge in identity_edges if edge.method == "dino" and edge.score >= dino_threshold
    )
    fallback_edges_above = sum(1 for edge in fallback_edges if edge.score >= fallback_threshold)

    accepted_dino = sum(1 for edge in accepted_edges if edge.method == "dino")
    accepted_yolo = sum(1 for edge in accepted_edges if edge.method == "yolo")
    accepted_fallback = sum(1 for edge in accepted_edges if edge.method == "spatial")

    stats: dict[str, int | float] = {
        "num_closed_tracks": int(num_closed_tracks),
        "num_fragments": int(len(fragments)),
        "num_candidates": int(len(candidates)),
        "num_yolo_edges_above_threshold": int(yolo_edges_above),
        "num_dino_edges_above_threshold": int(dino_edges_above),
        "num_fallback_edges_above_threshold": int(fallback_edges_above),
        "num_accepted_edges": int(len(accepted_edges)),
        "num_accepted_yolo": int(accepted_yolo),
        "num_accepted_dino": int(accepted_dino),
        "num_accepted_fallback": int(accepted_fallback),
        "num_absorbed_tracks": int(len(merge_map)),
        "relink_dino_coverage": float(dino_coverage),
        "relink_dino_accepted": int(accepted_dino),
        "relink_yolo_accepted": int(accepted_yolo),
    }

    accepted_payload = [
        {
            "predecessor_id": int(edge.predecessor_id),
            "successor_id": int(edge.successor_id),
            "score": float(edge.score),
            "method": edge.method,
        }
        for edge in accepted_edges
    ]
    return merge_map, {"stats": stats, "accepted_edges": accepted_payload}
