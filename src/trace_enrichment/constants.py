"""Lightweight constants for trace enrichment."""

from __future__ import annotations

import os

POOL_SIZE = (3, 3)
POOL_STRATEGY = "adaptive_avg_3x3"
ACTIVATION_LAYER_ALIASES = ["backbone.C2f_deep", "backbone.C2f_mid"]
OUTPUT_VECTOR_DIM = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_DEEP_LAYER = "8"
DEFAULT_MID_LAYER = "6"
DEFAULT_DEEP_STRIDE = 32
DEFAULT_MID_STRIDE = 16
SCHEMA_VERSION = "activation_enrichment_v2_single_pass"
DEFAULT_OUTPUT_ROOT = os.path.join("experiments", "results", "enriched")
