#!/usr/bin/env python3
"""Build poster-friendly per-track documentation from linking outputs."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_SRC = os.path.join(PROJECT_ROOT, "src")
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from temporal_linking.track_docs import TrackDocOutputs, build_track_docs_from_paths
from temporal_linking.trace_summary import resolve_video_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build per-track markdown docs and image folders.")
    parser.add_argument(
        "--linking-root",
        default=os.path.join("experiments", "results", "linking"),
        help="Root directory containing per-scenario linking results.",
    )
    parser.add_argument(
        "--enrichment-root",
        default=os.path.join("experiments", "results", "activation_enrichment"),
        help="Root directory containing per-scenario enrichment results.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario name to process. Repeat to limit to multiple scenarios. Default: process all scenarios under linking-root.",
    )
    parser.add_argument(
        "--render-trace-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render or refresh trace summary JPEGs when source video is available (default: enabled).",
    )
    return parser


def discover_scenarios(linking_root: str, requested: list[str]) -> list[str]:
    if requested:
        return sorted(set(str(item) for item in requested))

    if not os.path.isdir(linking_root):
        return []

    scenarios: list[str] = []
    for name in sorted(os.listdir(linking_root)):
        scenario_dir = os.path.join(linking_root, name)
        if not os.path.isdir(scenario_dir):
            continue
        if os.path.exists(os.path.join(scenario_dir, "tracks.json")) and os.path.exists(
            os.path.join(scenario_dir, "trace_summary.json")
        ):
            scenarios.append(name)
    return scenarios


def build_track_docs_for_scenario(
    *,
    scenario: str,
    linking_root: str,
    enrichment_root: str,
    render_trace_summary_images: bool = True,
) -> TrackDocOutputs:
    linking_dir = os.path.join(linking_root, scenario)
    enrichment_dir = os.path.join(enrichment_root, scenario)
    tracks_path = os.path.join(linking_dir, "tracks.json")
    trace_summary_path = os.path.join(linking_dir, "trace_summary.json")
    relink_manifest_path = os.path.join(linking_dir, "relink_manifest.json")
    enriched_json_path = os.path.join(enrichment_dir, "enriched_detections.json")

    if not os.path.exists(tracks_path):
        raise FileNotFoundError(f"Missing tracks.json for scenario '{scenario}': {tracks_path}")
    if not os.path.exists(trace_summary_path):
        raise FileNotFoundError(f"Missing trace_summary.json for scenario '{scenario}': {trace_summary_path}")

    video_path = None
    if os.path.exists(enriched_json_path):
        video_path = resolve_video_path(enriched_json_path=enriched_json_path)

    return build_track_docs_from_paths(
        scenario=scenario,
        tracks_path=tracks_path,
        trace_summary_path=trace_summary_path,
        output_dir=os.path.join(linking_dir, "track_docs"),
        trace_summary_dir=os.path.join(linking_dir, "trace_summary"),
        relink_manifest_path=relink_manifest_path if os.path.exists(relink_manifest_path) else None,
        video_path=video_path,
        render_trace_summary_images=render_trace_summary_images,
    )


def write_overview(linking_root: str, outputs: list[TrackDocOutputs]) -> str:
    overview_path = os.path.join(linking_root, "track_docs_overview.md")
    lines = [
        "# Track Docs Overview",
        "",
        "Scenario-level indexes live beside each linking result directory.",
        "",
        "## Scenarios",
    ]
    if not outputs:
        lines.extend(["", "- no track docs were generated", ""])
    else:
        for output in sorted(outputs, key=lambda item: item.scenario):
            rel_index_path = os.path.relpath(output.index_path, linking_root)
            lines.append(
                "- "
                f"[{output.scenario}]({rel_index_path}): "
                f"{output.num_tracks} tracks, "
                f"{output.num_summary_images} summary images, "
                f"{output.num_frame_images} annotated frame images"
            )
        lines.append("")

    with open(overview_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return overview_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    scenarios = discover_scenarios(args.linking_root, args.scenario)
    if not scenarios:
        print(f"No scenarios found under {args.linking_root}", file=sys.stderr)
        return 1

    outputs: list[TrackDocOutputs] = []
    failures: list[tuple[str, str]] = []

    for scenario in scenarios:
        try:
            output = build_track_docs_for_scenario(
                scenario=scenario,
                linking_root=args.linking_root,
                enrichment_root=args.enrichment_root,
                render_trace_summary_images=bool(args.render_trace_summary),
            )
        except Exception as exc:
            failures.append((scenario, str(exc)))
            print(f"[track-docs] ERROR {scenario}: {exc}", file=sys.stderr)
            continue
        outputs.append(output)
        print(
            f"[track-docs] {scenario}: {output.num_tracks} tracks -> {output.output_dir} "
            f"({output.num_summary_images} summaries, {output.num_frame_images} frame images)"
        )

    overview_path = write_overview(args.linking_root, outputs)
    print(f"[track-docs] overview -> {overview_path}")

    if failures:
        print("[track-docs] failures:", file=sys.stderr)
        for scenario, error in failures:
            print(f"  - {scenario}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
