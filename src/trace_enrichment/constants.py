"""Lightweight constants for trace enrichment."""

from __future__ import annotations

import os

POOL_SIZE = (1, 1)
POOL_STRATEGY = "adaptive_avg_1x1"
ACTIVATION_LAYER_ALIASES = ["2.cv1"]
# PCA compresses the raw YOLO embedding (default 208-D) to this target dimension.
OUTPUT_VECTOR_DIM = 128
DEFAULT_BATCH_SIZE = 8
# Selected from multi-video separability calibration with winner constraints
# stored at experiments/results/layer_selection/aggregate/aggregate_separability.csv:
# feature_dim >= 32 and dedupe .conv layers when parent Conv module exists.
DEFAULT_HEAD_LAYER = "2.cv1"
# Multi-layer embedding: (layer_name, separability_weight)
# Weights derived from aggregate separability sweep across all scenario videos.
# 22.cv3.0 is intentionally downweighted: it is retained as a class-consistency
# gate but provides limited within-class instance discrimination.
EMBEDDING_LAYERS = [
    ("4.cv1", 0.549),
    ("15", 0.351),
    ("22.cv3.0", 0.100),
]
DINO_EMBEDDING_DIM = 384
DINO_MODEL_REPO = "facebookresearch/dino:main"
DINO_MODEL_NAME = "dino_vits8"
DINO_INPUT_SIZE = 224
DINO_TINY_CROP_MIN_SIZE = 32
DINO_CROP_PADDING_RATIO = 0.05
DINO_LOAD_TIMEOUT_SECONDS = 20.0
DISABLE_DINO_ENV = "TRACE_DISABLE_DINO"
# Feature flag: set TRACE_DISABLE_MULTI_LAYER_EMBEDDING=1 to force single-layer extraction.
DISABLE_MULTI_LAYER_EMBEDDING_ENV = "TRACE_DISABLE_MULTI_LAYER_EMBEDDING"
DEFAULT_HEAD_STRIDE = 8
SCHEMA_VERSION = "activation_enrichment_v2_single_pass"
DEFAULT_OUTPUT_ROOT = os.path.join("experiments", "results", "activation_enrichment")
