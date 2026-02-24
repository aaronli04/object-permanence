#!/usr/bin/env python3
"""CLI entry point for activation enrichment pipeline."""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from detection.enrichment.pipeline import build_enrichment_parser, run_activation_enrichment  # type: ignore
else:
    from .pipeline import build_enrichment_parser, run_activation_enrichment


def main() -> None:
    parser = build_enrichment_parser()
    args = parser.parse_args()
    outputs = run_activation_enrichment(
        input_json_path=args.input_json,
        video_path=args.video,
        model_name=args.model,
        output_dir=args.output_dir,
        deep_layer_name=args.deep_layer,
        mid_layer_name=args.mid_layer,
        stride_deep=args.deep_stride,
        stride_mid=args.mid_stride,
        batch_size=args.batch_size,
        pca_dim=args.pca_dim,
    )
    print(f"Saved enriched detections to {outputs['enriched_detections']}")
    print(f"Saved PCA projection to {outputs['pca_projection']}")
    print(f"Saved projection manifest to {outputs['projection_manifest']}")


if __name__ == "__main__":
    main()
