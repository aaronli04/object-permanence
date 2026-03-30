#!/usr/bin/env python3
"""Shared metadata for curated DAVIS benchmark scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


MANIFEST_SCHEMA_VERSION: Final[str] = "davis_curated_manifest_v1"
SUMMARY_SCHEMA_VERSION: Final[str] = "davis_curated_summary_v1"


@dataclass(frozen=True)
class CuratedSequence:
    sequence: str
    scenario: str
    group: str
    expected_classes: tuple[str, ...]
    notes: str

    def expected_classes_csv(self) -> str:
        return ",".join(self.expected_classes)


CURATED_SEQUENCES: Final[tuple[CuratedSequence, ...]] = (
    CuratedSequence(
        sequence="parkour",
        scenario="davis__person_occlusion__parkour",
        group="occlusion",
        expected_classes=("person",),
        notes="Full-body occlusion and reappearance while moving.",
    ),
    CuratedSequence(
        sequence="bmx-bumps",
        scenario="davis__person_fast_motion__bmx_bumps",
        group="fast_motion",
        expected_classes=("person", "bicycle"),
        notes="Fast rider motion with viewpoint change and heavy occlusion.",
    ),
    CuratedSequence(
        sequence="breakdance",
        scenario="davis__person_rotation__breakdance",
        group="rotation",
        expected_classes=("person",),
        notes="Rapid pose changes and self-occlusion.",
    ),
    CuratedSequence(
        sequence="dance-twirl",
        scenario="davis__person_distractor__dance_twirl",
        group="distractor",
        expected_classes=("person",),
        notes="Human motion with rotation and cluttered background.",
    ),
    CuratedSequence(
        sequence="horsejump-high",
        scenario="davis__horse_person_occlusion__horsejump_high",
        group="occlusion",
        expected_classes=("person", "horse"),
        notes="Joint rider and horse motion with partial occlusion.",
    ),
    CuratedSequence(
        sequence="dog",
        scenario="davis__dog_baseline__dog",
        group="baseline",
        expected_classes=("dog",),
        notes="Clear single-object animal baseline.",
    ),
    CuratedSequence(
        sequence="blackswan",
        scenario="davis__bird_baseline__blackswan",
        group="baseline",
        expected_classes=("bird",),
        notes="High-contrast single-object baseline.",
    ),
    CuratedSequence(
        sequence="bear",
        scenario="davis__bear_occlusion__bear",
        group="occlusion",
        expected_classes=("bear",),
        notes="Large animal with appearance change and partial occlusion.",
    ),
    CuratedSequence(
        sequence="elephant",
        scenario="davis__elephant_baseline__elephant",
        group="baseline",
        expected_classes=("elephant",),
        notes="Large object with stable detector footprint.",
    ),
    CuratedSequence(
        sequence="drift-chicane",
        scenario="davis__car_speed__drift_chicane",
        group="vehicle",
        expected_classes=("car",),
        notes="Fast vehicle with motion blur and direction changes.",
    ),
    CuratedSequence(
        sequence="scooter-black",
        scenario="davis__scooter_exit_reenter__scooter_black",
        group="vehicle",
        expected_classes=("person", "motorcycle"),
        notes="Rider and scooter with scale and boundary changes.",
    ),
    CuratedSequence(
        sequence="motocross-jump",
        scenario="davis__motorcycle_jump__motocross_jump",
        group="vehicle",
        expected_classes=("person", "motorcycle"),
        notes="Rider and motorcycle airborne with strong motion.",
    ),
)

PRESET_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "all": ("baseline", "occlusion", "fast_motion", "rotation", "distractor", "vehicle"),
    "stress": ("occlusion", "fast_motion", "rotation", "distractor", "vehicle"),
    "baseline": ("baseline",),
}


def iter_sequences_for_preset(preset: str) -> tuple[CuratedSequence, ...]:
    if preset not in PRESET_GROUPS:
        available = ", ".join(sorted(PRESET_GROUPS))
        raise ValueError(f"Unknown DAVIS preset {preset!r}. Available presets: {available}")
    groups = set(PRESET_GROUPS[preset])
    return tuple(sequence for sequence in CURATED_SEQUENCES if sequence.group in groups)
