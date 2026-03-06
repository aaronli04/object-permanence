# Object Permanence

An offline two-stage pipeline for identity-preserving object tracking using YOLOv8. The system builds rich multi-layer identity embeddings from YOLO's internal feature representations and links detections across frames, including through occlusions, using cosine similarity and constrained assignment.

---

## Overview

Most object detectors treat each frame independently. This pipeline adds a temporal identity layer on top of YOLOv8: every detection is assigned a stable identity that persists across frames, survives occlusion, and can be relinked after a track is lost.

The pipeline runs in two offline stages:

**Stage 1 - Trace Enrichment** (`src/run_pipeline.py`)
Samples frames from video, runs YOLOv8, extracts multi-layer feature embeddings per detection, and projects all embeddings to a fixed 256-D output schema via PCA.

**Stage 2 - Temporal Linking** (`src/run_temporal_linking.py`)
Links detections across sampled frames using cosine similarity on normalized embeddings. Enforces one-to-one assignment via the Hungarian algorithm and runs a relink pass to recover fragmented tracks after occlusions.

---

## Multi-Layer Identity Embedding

Each detection's identity vector is a weighted combination of feature activations from three complementary layers of the YOLOv8 backbone and head. Layers were selected and weighted using a separability-driven calibration sweep across all scenario videos (see [Layer Calibration](#layer-calibration)).

| Layer | Tier | Raw Dim | Sweep Separability | Weight |
|---|---|---:|---:|---:|
| `4.cv1` | Appearance | 64 | 15.495 | 0.398 |
| `15` | Semantic | 64 | 9.926 | 0.255 |
| `22.cv3.0` | Class-level | 80 | 13.902 | 0.357 |

**Why three tiers?**
- **Appearance (4.cv1):** Early backbone activations encode texture and color patterns, a strong signal for distinguishing objects that look different.
- **Semantic (15):** Mid-network neck activations encode spatial context and object structure, which stays stable across viewpoint changes and partial occlusion.
- **Class-level (22.cv3.0):** Detection-head activations encode class probability space, which anchors identity to semantic type.

**Embedding construction per detection:**
1. Register forward hooks on all configured embedding layers.
2. Run YOLO forward pass and map each detection ROI onto each hooked feature map.
3. Adaptive-average pool each layer ROI to a single vector.
4. L2-normalize each vector independently.
5. Multiply each normalized vector by its separability-derived weight.
6. Concatenate the vectors to a 208-D combined embedding (`64 + 64 + 80`).
7. L2-normalize the final concatenated embedding.

**Resilience:**
- Hooks are registered and removed atomically around each forward pass to prevent memory leaks in long-running sessions.
- If a layer is unavailable in a model variant, remaining layer weights are renormalized to sum to 1.
- If fewer than 2 embedding layers are available, enrichment raises a clear error rather than silently degrading.
- Single-layer fallback is available via `TRACE_DISABLE_MULTI_LAYER_EMBEDDING=1`.

Embedding configuration is stored in `src/trace_enrichment/constants.py` as `EMBEDDING_LAYERS` (list of `(layer_name, weight)` tuples). Weights should be recalibrated by re-running the aggregate sweep if the model or video distribution changes.

---

## Temporal Linking

Frame-to-frame linking operates on cosine similarity between normalized projected embeddings.

**Matching:**
- Similarity gate: `visual_similarity >= similarity_threshold` (default `0.65`).
- Assignment: Hungarian algorithm for globally consistent one-to-one matching per frame pair.

**Track state machine:**
```text
TENTATIVE -> ACTIVE -> LOST -> CLOSED
```

Reference descriptors blend last, EMA, and history vectors for stability against appearance drift.

**Relink pass:**
After the primary linking run, a second pass evaluates pairs of closed track fragments to recover identities split by occlusion:
- Enforces class consistency and temporal ordering constraints.
- Accepts high-confidence fragment pairs by centroid-cosine score.
- Falls back to spatial proximity when cosine score is below threshold.
- Merges accepted chains into canonical track IDs.

---

## Layer Calibration

Layer selection is driven by a separability metric, a Fisher-style ratio measuring how well each layer's activations separate different objects while remaining consistent within an object.

**Metrics computed per layer:**
- `within_var`: mean per-group variance across feature dimensions.
- `between_var`: variance of per-group mean vectors across dimensions.
- `separability = between_var / (within_var + 1e-8)`.

**Grouping policy:** use `track_id` when available on YOLO boxes; fallback to `class_id`.

**Ranking policy:** descending `separability` -> descending `mean_consecutive_cosine` -> ascending `layer_name`.

**Degenerate handling:** layers with fewer than 2 distinct groups receive `separability = 0.0` and emit a warning.

**Winner selection constraints:**
- Exclude layers with `feature_dim < 32`.
- Deduplicate `.conv` child entries when the parent `Conv` module is already present.

**Current aggregate leaderboard (top 5, constrained):**

| Rank | Layer | Type | Feature Dim | Mean Separability | Mean Cosine |
|---|---|---:|---:|---:|---:|
| 1 | `2.cv1` | Conv | 32 | 15.911 | 0.9776 |
| 2 | `1` | Conv | 32 | 15.675 | 0.9440 |
| 3 | `4.cv1` | Conv | 64 | 15.495 | 0.9673 |
| 4 | `22.cv3.0` | Sequential | 80 | 13.902 | 0.9981 |
| 5 | `15` | C2f | 64 | 9.926 | 0.9809 |

Calibration artifacts:
```text
experiments/results/layer_selection/per_video/layer_stability_sweep_<scenario>.csv
experiments/results/layer_selection/aggregate/aggregate_separability.csv
```

---

## Experiment Results

### Top-k Linking Evaluation (`Right_to_left`)

| k | within_early | within_late | cross | ball_tracks | total_tracks | valid_tracks |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.8380 | 0.7886 | 0.7141 | 1 | 5 | 3 |
| 64 | 0.8288 | 0.7711 | 0.7044 | 1 | 5 | 3 |

Default: `--activation-topk 64`.

### End-to-End Scenario Results

Configuration: embedding layers `4.cv1 + 15 + 22.cv3.0`, raw dim `208`, `activation_topk=64`, `similarity_threshold=0.65`, `relink_threshold=0.55`, `relink_max_gap_frames=-1`, `relink_fallback_threshold=0.40`.

| Scenario | Frames | Detections | Ball Tracks | Total Tracks | Valid Tracks | Relink Edges |
|---|---:|---:|---:|---:|---:|---:|
| 10sec_Left_to_Right | 133 | 160 | 1 | 6 | 5 | 1 |
| 3sec_Left_to_Right | 49 | 77 | 1 | 6 | 5 | 1 |
| Exit_frame_while_occluded | 53 | 72 | 1 | 5 | 4 | 0 |
| Left_bounce_back | 64 | 105 | 1 | 5 | 4 | 2 |
| Left_to_right | 25 | 52 | 1 | 6 | 5 | 1 |
| No_occlusion_ball_removed | 34 | 37 | 1 | 10 | 5 | 0 |
| Occlusion_ball_removed | 48 | 114 | 1 | 14 | 8 | 2 |
| Right_to_left | 21 | 40 | 1 | 4 | 3 | 1 |
| **Totals** | **427** | **657** | **8** | **56** | **39** | **8** |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

---

## Run Commands

### 1. Per-video layer sweeps
```bash
for v in data/raw_videos/*.mp4; do
  stem="$(basename "$v" .mp4)"
  python3 experiments/layer_stability_sweep.py \
    --video "$v" \
    --model yolov8n.pt \
    --sample-rate 5 \
    --max-sampled-frames 20 \
    --class-id -1 \
    --min-confidence 0.25 \
    --output-csv "experiments/results/layer_selection/per_video/layer_stability_sweep_${stem}.csv"
done
```

### 2. Aggregate layer sweeps
```bash
python3 experiments/aggregate_layer_sweeps.py \
  --input-glob "experiments/results/layer_selection/per_video/layer_stability_sweep_*.csv" \
  --output-csv experiments/results/layer_selection/aggregate/aggregate_separability.csv \
  --winner-min-feature-dim 32 \
  --top-n 20
```

### 3. Activation enrichment (batch)
```bash
python3 src/run_pipeline.py \
  --video-dir data/raw_videos \
  --pattern "*.mp4" \
  --sample-rate 5 \
  --model yolov8n.pt
```

### 4. Temporal linking (batch)
```bash
for f in experiments/results/activation_enrichment/*/enriched_detections.json; do
  python3 src/run_temporal_linking.py \
    --enriched-json "$f" \
    --activation-topk 64 \
    --similarity-threshold 0.65 \
    --relink-threshold 0.55 \
    --relink-max-gap-frames -1 \
    --relink-fallback-threshold 0.40
done
```

---

## Output Layout

```text
experiments/results/
  layer_selection/
    per_video/
      layer_stability_sweep_<scenario>.csv
    aggregate/
      aggregate_separability.csv
  activation_enrichment/
    <scenario>/
      enriched_detections.json
      pca_projection.pkl
      projection_manifest.json
  linking/
    <scenario>/
      linked_detections.json
      tracks.json
      linking_manifest.json
      relink_manifest.json
```
