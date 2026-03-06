"""Lightweight constants for trace enrichment."""

from __future__ import annotations

import os

POOL_SIZE = (1, 1)
POOL_STRATEGY = "adaptive_avg_1x1"
ACTIVATION_LAYER_ALIASES = ["2.cv1"]
OUTPUT_VECTOR_DIM = 256
DEFAULT_BATCH_SIZE = 8
# Selected from multi-video separability calibration with winner constraints
# stored at experiments/results/layer_selection/aggregate/aggregate_separability.csv:
# feature_dim >= 32 and dedupe .conv layers when parent Conv module exists.
DEFAULT_HEAD_LAYER = "2.cv1"
# Multi-layer embedding: (layer_name, separability_weight)
# Weights derived from aggregate separability sweep across all scenario videos
# 4.cv1=15.495, 15=9.926, 22.cv3.0=13.902 -> normalized to sum=1.0
EMBEDDING_LAYERS = [
    ("4.cv1", 0.398),
    ("15", 0.255),
    ("22.cv3.0", 0.357),
]
# Feature flag: set TRACE_DISABLE_MULTI_LAYER_EMBEDDING=1 to force single-layer extraction.
DISABLE_MULTI_LAYER_EMBEDDING_ENV = "TRACE_DISABLE_MULTI_LAYER_EMBEDDING"
DEFAULT_HEAD_STRIDE = 8
SCHEMA_VERSION = "activation_enrichment_v2_single_pass"
DEFAULT_OUTPUT_ROOT = os.path.join("experiments", "results", "activation_enrichment")
