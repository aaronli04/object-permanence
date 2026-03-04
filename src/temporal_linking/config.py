"""Configuration models for temporal linking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TemporalLinkingConfig:
    """Tunable parameters for offline temporal linking."""

    # Single similarity gating threshold used for all matches and recoveries.
    similarity_threshold: float

    # Track lifecycle controls.
    max_lost_frames: int = 6
    min_hits_to_activate: int = 2
    min_track_length: int = 2

    # Noise/stability controls.
    history_size: int = 5
    ema_alpha: float = 0.35

    # Score blending weights.
    w_last: float = 0.55
    w_ema: float = 0.30
    w_hist: float = 0.15
    w_spatial: float = 0.05
    w_consistency: float = 0.10
    w_age: float = 0.05

    # Assignment and matching policy.
    assignment_method: Literal["hungarian", "greedy"] = "hungarian"
    match_within_class: bool = True
    filter_short_tracks_in_summary: bool = True
    activation_topk: int | None = 64

    # Post-hoc relinking controls (always evaluated in sweep 2).
    relink_threshold: float = 0.55
    relink_max_gap_frames: int = 120  # finite by default to reduce long-gap false merges
    relink_min_track_hits: int = 2
    relink_max_pixels_per_frame: float = 15.0
    relink_fallback_threshold: float = 0.40

    def __post_init__(self) -> None:
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [-1.0, 1.0]")
        if not -1.0 <= self.relink_threshold <= 1.0:
            raise ValueError("relink_threshold must be in [-1.0, 1.0]")
        if self.relink_max_gap_frames < -1:
            raise ValueError("relink_max_gap_frames must be -1 or >= 0")
        if self.relink_min_track_hits <= 0:
            raise ValueError("relink_min_track_hits must be > 0")
        if self.relink_max_pixels_per_frame <= 0.0:
            raise ValueError("relink_max_pixels_per_frame must be positive")
        if not -1.0 <= self.relink_fallback_threshold <= 1.0:
            raise ValueError("relink_fallback_threshold must be in [-1.0, 1.0]")
        if self.max_lost_frames < 0:
            raise ValueError("max_lost_frames must be >= 0")
        if self.min_hits_to_activate <= 0:
            raise ValueError("min_hits_to_activate must be > 0")
        if self.min_track_length <= 0:
            raise ValueError("min_track_length must be > 0")
        if self.history_size <= 0:
            raise ValueError("history_size must be > 0")
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0.0, 1.0]")

        for name, value in (
            ("w_last", self.w_last),
            ("w_ema", self.w_ema),
            ("w_hist", self.w_hist),
            ("w_spatial", self.w_spatial),
            ("w_consistency", self.w_consistency),
            ("w_age", self.w_age),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0.0")

        if (self.w_last + self.w_ema + self.w_hist) <= 0.0:
            raise ValueError("w_last + w_ema + w_hist must be > 0.0")
        if self.activation_topk is not None and self.activation_topk <= 0:
            raise ValueError("activation_topk must be > 0 when provided")
