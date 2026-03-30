#!/usr/bin/env python3
"""Build a curated DAVIS video subset for object-permanence evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from davis_curated_common import MANIFEST_SCHEMA_VERSION, PRESET_GROUPS, CuratedSequence, iter_sequences_for_preset


@dataclass(frozen=True)
class ManifestRow:
    scenario: str
    sequence: str
    group: str
    expected_classes: str
    notes: str
    frames: int
    status: str
    video_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--davis-root",
        default="DAVIS/JPEGImages/480p",
        help="Root directory containing DAVIS frame folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw_videos",
        help="Directory where converted MP4 files will be written.",
    )
    parser.add_argument(
        "--manifest-json",
        default="data/davis_curated_manifest.json",
        help="Output JSON manifest path.",
    )
    parser.add_argument(
        "--manifest-csv",
        default="data/davis_curated_manifest.csv",
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frame rate used for reconstructed MP4 files.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_GROUPS),
        default="all",
        help="Subset preset to build.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild MP4 files even if they already exist.",
    )
    return parser.parse_args()


def run_ffmpeg(input_pattern: Path, output_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(input_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def count_frames(sequence_dir: Path) -> int:
    return sum(1 for _ in sequence_dir.glob("*.jpg"))


def build_manifest_row(
    *,
    item: CuratedSequence,
    davis_root: Path,
    output_dir: Path,
    fps: int,
    force: bool,
) -> ManifestRow:
    sequence_dir = davis_root / item.sequence
    frame_count = count_frames(sequence_dir) if sequence_dir.exists() else 0
    output_path = output_dir / f"{item.scenario}.mp4"

    status = "missing_sequence"
    if sequence_dir.exists():
        status = "existing_video" if output_path.exists() and not force else "generated"
        if status == "generated":
            run_ffmpeg(sequence_dir / "%05d.jpg", output_path, fps)

    return ManifestRow(
        scenario=item.scenario,
        sequence=item.sequence,
        group=item.group,
        expected_classes=item.expected_classes_csv(),
        notes=item.notes,
        frames=frame_count,
        status=status,
        video_path=str(output_path),
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "sequence",
        "group",
        "expected_classes",
        "notes",
        "frames",
        "status",
        "video_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def main() -> int:
    args = parse_args()
    davis_root = Path(args.davis_root)
    output_dir = Path(args.output_dir)
    manifest_json = Path(args.manifest_json)
    manifest_csv = Path(args.manifest_csv)
    sequences = iter_sequences_for_preset(args.preset)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[ManifestRow] = []

    for item in sequences:
        row = build_manifest_row(
            item=item,
            davis_root=davis_root,
            output_dir=output_dir,
            fps=args.fps,
            force=bool(args.force),
        )
        rows.append(row)
        print(f"{row.status:>16}  {row.sequence:<16} -> {row.video_path}")

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "preset": args.preset,
        "davis_root": str(davis_root),
        "output_dir": str(output_dir),
        "items": [row.to_dict() for row in rows],
    }
    write_json(manifest_json, payload)
    write_csv(manifest_csv, rows)

    generated = sum(1 for row in rows if row.status == "generated")
    existing = sum(1 for row in rows if row.status == "existing_video")
    missing = sum(1 for row in rows if row.status == "missing_sequence")
    print("")
    print(f"Generated videos: {generated}")
    print(f"Reused videos:   {existing}")
    print(f"Missing inputs:  {missing}")
    print(f"Manifest JSON:   {manifest_json}")
    print(f"Manifest CSV:    {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
