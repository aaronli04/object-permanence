"""Lightweight constants for trace enrichment."""

from __future__ import annotations

import os

POOL_SIZE = (3, 3)
POOL_STRATEGY = "adaptive_avg_3x3"
ACTIVATION_LAYER_ALIASES = ["head.cv3[2]"]
OUTPUT_VECTOR_DIM = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_HEAD_LAYER = "22.cv3.2.1"
DEFAULT_HEAD_STRIDE = 32
SCHEMA_VERSION = "activation_enrichment_v2_single_pass"
DEFAULT_OUTPUT_ROOT = os.path.join("experiments", "results", "enriched")
