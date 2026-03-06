# Object Permanence

Object permanence pipeline for YOLOv8 detections with identity-preserving temporal linking.

## What Is Implemented

The system runs in two offline stages:

- Stage 1: `trace_enrichment` (`src/run_pipeline.py`) samples frames, runs YOLOv8, builds identity embeddings per detection, and projects embeddings to a fixed 256-D output schema with PCA.
- Stage 2: `temporal_linking` (`src/run_temporal_linking.py`) links detections across sampled frames with cosine similarity, enforces one-to-one assignment, and relinks fragmented tracks after occlusions.

## Multi-Layer Identity Embedding (Current Baseline)

Identity vectors are no longer extracted from one layer only. The implemented embedding is a weighted combination of three complementary tiers:

| Layer | Tier | Raw Dim | Sweep Separability | Weight |
|---|---|---:|---:|---:|
| `4.cv1` | appearance | 64 | 15.495 | 0.398 |
| `15` | semantic | 64 | 9.926 | 0.255 |
| `22.cv3.0` | class-level | 80 | 13.902 | 0.357 |

Embedding construction per detection:

1. Extract ROI feature vector from each configured layer.
2. Adaptive-average pool to one vector per layer.
3. L2-normalize each layer vector.
4. Multiply each layer vector by its separability-derived weight.
5. Concatenate vectors (`64 + 64 + 80 = 208`).
6. L2-normalize the final concatenated embedding.

Resilience behavior:

- Hooks for all configured layers are registered/removed atomically per forward pass.
- If a layer is missing in a model variant, available-layer weights are renormalized.
- If fewer than 2 embedding layers are available, enrichment raises a clear error.
- Single-layer fallback remains available with `TRACE_DISABLE_MULTI_LAYER_EMBEDDING=1`.

Constants source:
- `src/trace_enrichment/constants.py`
- `EMBEDDING_LAYERS` stores `(layer_name, weight)`.
- `DEFAULT_HEAD_LAYER` is retained for fallback compatibility.
- `DEFAULT_HEAD_LAYER` is the single-layer fallback (`2.cv1`); the default production embedding uses `EMBEDDING_LAYERS`.

## Layer Calibration and Selection

Layer selection is calibrated using separability, not pooled norm standard deviation.

Per-layer metrics:

- `within_var`: mean per-group variance across feature dimensions.
- `between_var`: variance of group mean vectors across dimensions.
- `separability = between_var / (within_var + 1e-8)`.

Grouping policy during sweep:

- Prefer `track_id` when available.
- Fallback to `class_id` when track ID is not available.

Ranking policy:

1. Descending separability.
2. Descending mean consecutive cosine (tie-break).
3. Ascending layer name (deterministic final tie-break).

Degenerate safety:

- If a layer has fewer than 2 distinct groups, `separability=0.0` and a warning is logged.

Calibration artifacts are persisted under:

- `experiments/results/layer_selection/per_video/layer_stability_sweep_<scenario>.csv`
- `experiments/results/layer_selection/aggregate/aggregate_separability.csv`

Aggregate winner constraints:

- Exclude layers with `feature_dim < 32`.
- Deduplicate `.conv` child candidates when the parent Conv module entry exists.

Current constrained aggregate leaderboard (top rows):

| Rank | Layer | Type | Feature Dim | Mean Separability | Mean Consecutive Cosine |
|---|---|---|---:|---:|---:|
| 1 | `2.cv1` | `Conv` | 32 | 15.911515 | 0.977638 |
| 2 | `1` | `Conv` | 32 | 15.674581 | 0.944006 |
| 3 | `4.cv1` | `Conv` | 64 | 15.495382 | 0.967329 |
| 4 | `22.cv3.0` | `Sequential` | 80 | 13.901690 | 0.998128 |
| 5 | `22.cv3.0.2` | `Conv2d` | 80 | 13.901690 | 0.998128 |

## Linking Approach

Frame-to-frame linking uses cosine similarity on normalized activation vectors:

- Primary gate: `visual_similarity >= similarity_threshold`.
- Assignment: Hungarian (default) for globally consistent one-to-one matching.
- Track state machine: `TENTATIVE -> ACTIVE -> LOST -> CLOSED`.
- Reference descriptor blends last/EMA/history vectors for stability.

Relink pass:

1. Evaluates closed-track fragment pairs with class + time-order constraints.
2. Accepts high-confidence centroid-cosine links.
3. Uses spatial fallback when centroid score is below threshold.
4. Merges accepted chains into canonical track IDs.

## Experiment Results

### Top-k Linking Evaluation (Right_to_left)

| k | within_early | within_late | cross | ball_tracks | total_tracks | valid_tracks |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.837967 | 0.788560 | 0.714120 | 1 | 5 | 3 |
| 64 | 0.828755 | 0.771111 | 0.704412 | 1 | 5 | 3 |

Chosen default: `--activation-topk 64`.

### End-to-End Scenario Results (Current Multi-Layer Baseline)

Snapshot: fresh full batch run with current defaults on March 6, 2026.

Configuration:

- embedding layers: `4.cv1, 15, 22.cv3.0`
- raw embedding dim: `208`
- linking: `activation_topk=64`, `similarity_threshold=0.65`
- relink: `relink_threshold=0.55`, `relink_max_gap_frames=-1`, `relink_fallback_threshold=0.40`

Measured outcomes:

| Scenario | Frames | Detections | Ball Tracks | Total Tracks | Valid Tracks | Relink Edges |
|---|---:|---:|---:|---:|---:|---:|
| `10sec_Left_to_Right` | 133 | 160 | 1 | 6 | 5 | 1 |
| `3sec_Left_to_Right` | 49 | 77 | 1 | 6 | 5 | 1 |
| `Exit_frame_while_occluded` | 53 | 72 | 1 | 5 | 4 | 0 |
| `Left_bounce_back` | 64 | 105 | 1 | 5 | 4 | 2 |
| `Left_to_right` | 25 | 52 | 1 | 6 | 5 | 1 |
| `No_occlusion_ball_removed` | 34 | 37 | 1 | 10 | 5 | 0 |
| `Occlusion_ball_removed` | 48 | 114 | 1 | 14 | 8 | 2 |
| `Right_to_left` | 21 | 40 | 1 | 4 | 3 | 1 |

Totals: `427` sampled frames, `657` detections, `8` accepted relink edges across `8` scenarios.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

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
