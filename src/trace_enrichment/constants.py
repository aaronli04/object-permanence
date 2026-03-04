"""Lightweight constants for trace enrichment."""

from __future__ import annotations

import os

POOL_SIZE = (1, 1)
POOL_STRATEGY = "adaptive_avg_1x1"
ACTIVATION_LAYER_ALIASES = ["neck.C2f.15"]
OUTPUT_VECTOR_DIM = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_HEAD_LAYER = "15"
DEFAULT_HEAD_STRIDE = 8
SCHEMA_VERSION = "activation_enrichment_v2_single_pass"
DEFAULT_OUTPUT_ROOT = os.path.join("experiments", "results", "activation_enrichment")
