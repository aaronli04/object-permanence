"""Track lifecycle manager for temporal linking."""

from __future__ import annotations

from collections import deque
import numpy as np

from common.numeric import l2_normalize

from .config import TemporalLinkingConfig
from .types import Assignment, Detection, Track, TrackStatus, TrackerState


class TrackManager:
    """Owns track lifecycle transitions and state containers."""

    def __init__(self, cfg: TemporalLinkingConfig) -> None:
        self.cfg = cfg
        self.state = TrackerState()

    def candidates(self) -> list[Track]:
        tracks = list(self.state.active.values()) + list(self.state.lost.values())
        tracks.sort(key=lambda track: track.track_id)
        return tracks

    def get(self, track_id: int) -> Track:
        track = self.state.active.get(track_id)
        if track is None:
            track = self.state.lost.get(track_id)
        if track is None:
            track = self.state.closed.get(track_id)
        if track is None:
            raise KeyError(f"Unknown track_id={track_id}")
        return track

    def spawn(self, det: Detection, frame_num: int) -> Track:
        track_id = self.state.next_track_id
        self.state.next_track_id += 1

        status = TrackStatus.ACTIVE if self.cfg.min_hits_to_activate <= 1 else TrackStatus.TENTATIVE
        track = Track(
            track_id=track_id,
            class_id=det.class_id,
            class_name=det.class_name,
            status=status,
            start_frame=frame_num,
            last_seen_frame=frame_num,
            last_bbox_xyxy=det.bbox_xyxy.copy(),
            last_vec=det.activation_vec.copy(),
            ema_vec=det.activation_vec.copy(),
            frame_width=det.frame_width,
            frame_height=det.frame_height,
        )
        track.vec_history = deque([det.activation_vec.copy()], maxlen=self.cfg.history_size)
        track.sim_history = deque(maxlen=self.cfg.history_size)
        track.events.append({"frame_num": int(frame_num), "type": "created"})
        if status == TrackStatus.ACTIVE:
            track.events.append({"frame_num": int(frame_num), "type": "activated"})
        track.observations.append(
            {
                "frame_num": int(frame_num),
                "det_index": int(det.det_index),
                "bbox": [float(v) for v in det.bbox_xyxy.tolist()],
                "visual_similarity": None,
            }
        )
        track.obs_vecs.append(det.activation_vec.copy())
        self._append_dino_observation(track, det)
        cx = (float(det.bbox_xyxy[0]) + float(det.bbox_xyxy[2])) * 0.5
        cy = (float(det.bbox_xyxy[1]) + float(det.bbox_xyxy[3])) * 0.5
        track.obs_positions.append((cx, cy, int(frame_num)))

        self._set_active(track)
        return track

    def apply_match(self, track_id: int, det: Detection, assignment: Assignment, frame_num: int) -> Track:
        track = self.get(track_id)
        source_status = track.status

        track.last_seen_frame = int(frame_num)
        track.hits += 1
        track.miss_streak = 0

        track.last_bbox_xyxy = det.bbox_xyxy.copy()
        track.last_vec = det.activation_vec.copy()
        track.ema_vec = l2_normalize(
            (self.cfg.ema_alpha * det.activation_vec) + ((1.0 - self.cfg.ema_alpha) * track.ema_vec)
        )
        track.frame_width = det.frame_width
        track.frame_height = det.frame_height

        if track.vec_history.maxlen != self.cfg.history_size:
            track.vec_history = deque(track.vec_history, maxlen=self.cfg.history_size)
        track.vec_history.append(det.activation_vec.copy())

        if track.sim_history.maxlen != self.cfg.history_size:
            track.sim_history = deque(track.sim_history, maxlen=self.cfg.history_size)
        track.sim_history.append(float(assignment.visual_similarity))

        track.visual_similarity_sum += float(assignment.visual_similarity)
        track.visual_similarity_count += 1

        track.observations.append(
            {
                "frame_num": int(frame_num),
                "det_index": int(det.det_index),
                "bbox": [float(v) for v in det.bbox_xyxy.tolist()],
                "visual_similarity": float(assignment.visual_similarity),
            }
        )
        track.obs_vecs.append(det.activation_vec.copy())
        self._append_dino_observation(track, det)
        cx = (float(det.bbox_xyxy[0]) + float(det.bbox_xyxy[2])) * 0.5
        cy = (float(det.bbox_xyxy[1]) + float(det.bbox_xyxy[3])) * 0.5
        track.obs_positions.append((cx, cy, int(frame_num)))

        if source_status == TrackStatus.LOST:
            track.status = TrackStatus.ACTIVE
            track.events.append({"frame_num": int(frame_num), "type": "recovered"})
        elif source_status == TrackStatus.TENTATIVE and track.hits >= self.cfg.min_hits_to_activate:
            track.status = TrackStatus.ACTIVE
            track.events.append({"frame_num": int(frame_num), "type": "activated"})

        self._set_active(track)
        return track

    def mark_unmatched(self, candidates: list[Track], matched_track_ids: set[int], frame_num: int) -> None:
        for track in candidates:
            if track.track_id in matched_track_ids:
                continue

            track.miss_streak += 1
            track.total_misses += 1
            track.max_miss_streak = max(track.max_miss_streak, track.miss_streak)

            if track.status in (TrackStatus.ACTIVE, TrackStatus.TENTATIVE):
                track.status = TrackStatus.LOST
                track.events.append({"frame_num": int(frame_num), "type": "lost"})

            if track.status == TrackStatus.LOST:
                self._set_lost(track)

            if track.miss_streak > self.cfg.max_lost_frames:
                self.close(track, end_frame=track.last_seen_frame)

    def close(self, track: Track, end_frame: int) -> None:
        if track.status == TrackStatus.CLOSED:
            return

        track.dino_vector = self._build_track_dino_vector(track)
        track.status = TrackStatus.CLOSED
        track.events.append({"frame_num": int(end_frame), "type": "closed"})
        self._remove_open(track.track_id)
        self.state.closed[track.track_id] = track

    def close_remaining(self) -> None:
        for track in list(self.state.active.values()) + list(self.state.lost.values()):
            self.close(track, end_frame=track.last_seen_frame)

    def finalize(self) -> list[Track]:
        self.close_remaining()
        return self.closed_tracks()

    def closed_tracks(self) -> list[Track]:
        tracks = list(self.state.closed.values())
        tracks.sort(key=lambda track: track.track_id)
        return tracks

    def _remove_open(self, track_id: int) -> None:
        self.state.active.pop(track_id, None)
        self.state.lost.pop(track_id, None)

    def _set_active(self, track: Track) -> None:
        self._remove_open(track.track_id)
        self.state.active[track.track_id] = track

    def _set_lost(self, track: Track) -> None:
        self._remove_open(track.track_id)
        self.state.lost[track.track_id] = track

    def _append_dino_observation(self, track: Track, det: Detection) -> None:
        if det.dino_vector is None:
            return
        vec = np.asarray(det.dino_vector, dtype=np.float32)
        if vec.ndim != 1 or int(vec.shape[0]) <= 0:
            return
        if not bool(np.isfinite(vec).all()):
            return
        norm = float(np.linalg.norm(vec))
        if norm <= 0.0:
            return
        track.obs_dino_vecs.append((vec / norm).astype(np.float32, copy=False))

    def _build_track_dino_vector(self, track: Track) -> np.ndarray | None:
        if len(track.obs_dino_vecs) < int(self.cfg.relink_dino_min_detections):
            return None
        stacked = np.stack(track.obs_dino_vecs, axis=0)
        mean_vec = np.mean(stacked, axis=0).astype(np.float32, copy=False)
        mean_vec = l2_normalize(mean_vec)
        if float(np.linalg.norm(mean_vec)) <= 0.0:
            return None
        return mean_vec
