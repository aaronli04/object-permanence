# Object Permanence

An offline two-stage pipeline for identity-preserving object tracking using YOLOv8. The system builds multi-layer YOLO identity embeddings for frame-to-frame linking and uses DINO CLS vectors as relink-only sidecar evidence to verify identity across occlusion gaps.

---

## Overview

Most object detectors treat each frame independently. This pipeline adds a temporal identity layer on top of YOLOv8: every detection is assigned a stable identity that persists across frames, survives occlusion, and can be relinked after a track is lost.

The pipeline runs in two offline stages:

**Stage 1 - Trace Enrichment** (`src/run_pipeline.py`)
Samples frames from video, runs YOLOv8, extracts multi-layer feature embeddings per detection, and projects all embeddings to a 128-D target embedding via PCA (actual per-run output dim can be lower when sample count is small).

**Stage 2 - Temporal Linking** (`src/run_temporal_linking.py`)
Links detections across sampled frames using cosine similarity on normalized embeddings. Enforces one-to-one assignment via the Hungarian algorithm and runs a relink pass to recover fragmented tracks after occlusions.

---

## Multi-Layer Identity Embedding

Frame-to-frame linking uses a YOLO-only multi-layer composite embedding. Layer weights are calibration-driven (see [Layer Calibration](#layer-calibration)).

| Layer | Tier | Raw Dim | Sweep Separability | Weight |
|---|---|---:|---:|---:|
| `4.cv1` | Appearance | 64 | 15.495 | 0.549 |
| `15` | Semantic | 64 | 9.926 | 0.351 |
| `22.cv3.0` | Class-level | 80 | 13.902 | 0.100 |

**Why these YOLO tiers?**
- **Appearance (4.cv1):** Early backbone activations encode texture and color patterns, a strong signal for distinguishing objects that look different.
- **Semantic (15):** Mid-network neck activations encode spatial context and object structure, which stays stable across viewpoint changes and partial occlusion.
- **Class-level (22.cv3.0):** Detection-head activations encode class probability space. It is retained as a class-consistency gate but weighted conservatively because same-class instances are often near-identical in this space.

### DINO Methodology (Relink-Only Re-identification)

DINO is intentionally not used for frame-to-frame assignment. Consecutive-frame linking benefits from **stability** under small pose/lighting jitter, while relinking fragmented tracks needs **discriminability** to avoid false identity merges.

The implementation therefore treats DINO CLS as relink-only evidence:
- Extract DINO per detection during enrichment and store it as sidecar metadata.
- Aggregate sidecar vectors at track close time into a track-level DINO representative.
- Use DINO cosine only in relink candidate scoring when both fragments have valid DINO representatives.
- Fall back to YOLO relink scoring when DINO is missing, and keep spatial fallback as a third pass.

### Implementation Details

**Embedding construction per detection (enrichment):**
1. Register forward hooks on configured YOLO embedding layers.
2. Run YOLO forward pass and map each detection ROI onto each hooked feature map.
3. Adaptive-average pool each YOLO layer ROI, L2-normalize per-layer vectors, apply YOLO layer weights, concatenate, then L2-normalize to a YOLO composite vector (`208-D`).
4. Fit PCA on run detections and reduce raw YOLO embeddings to a `128-D` target projection (or lower effective dim when detections are fewer than 128). PCA is used for compression, not expansion.
5. Separately extract DINO CLS (`384-D`, L2-normalized) from padded detection crops and store as sidecar:
   - `activation.dino_vector` (`list[float]` or `null`)
   - `activation.dino_available` (`bool`)
6. `projection_manifest.json` records DINO sidecar role and runtime state:
   - `dino_role = "relink_sidecar"`
   - `dino_enabled`, `dino_model`, `dino_load_error`

**Resilience:**
- Hooks are registered and removed atomically around each forward pass to prevent memory leaks in long-running sessions.
- If a YOLO layer is unavailable in a model variant, remaining layer weights are renormalized to sum to 1.
- If fewer than 2 embedding layers are available, enrichment raises a clear error rather than silently degrading.
- Single-layer fallback is available via `TRACE_DISABLE_MULTI_LAYER_EMBEDDING=1`.
- DINO can be disabled via `TRACE_DISABLE_DINO=1`.
- If DINO load fails (e.g., offline and uncached), enrichment logs a warning and marks DINO sidecars unavailable while preserving YOLO enrichment outputs.
- Tiny/invalid DINO crops are treated as unavailable sidecars (`dino_vector = null`, `dino_available = false`).
- `projection_manifest.json` records both actual `projection_dim` and requested `projection_dim_requested`.

Embedding configuration lives in `src/trace_enrichment/constants.py`:
- `EMBEDDING_LAYERS`: YOLO per-layer weights.
- `DINO_EMBEDDING_DIM`: sidecar DINO vector dimension (`384`).

---

## Temporal Linking

Frame-to-frame linking operates on cosine similarity between normalized projected embeddings.

**Matching:**
- Similarity gate: `visual_similarity >= similarity_threshold` (recommended `0.70`).
- Spatial plausibility gate: centroid distance must be <= `max_centroid_distance` (default `0.40`, normalized by frame diagonal) before cosine scoring.
- Assignment: Hungarian algorithm for globally consistent one-to-one matching per frame pair.

**Track state machine:**
```text
TENTATIVE -> ACTIVE -> LOST -> CLOSED
```

Reference descriptors blend last, EMA, and history vectors for stability against appearance drift.

**Relink pass:**
After the primary linking run, a second pass evaluates pairs of closed track fragments to recover identities split by occlusion:
- Enforces class consistency and temporal ordering constraints.
- Scores identity by method:
  - `dino`: cosine on track-level DINO representatives when both are available and `relink_use_dino=true` (gate: `relink_dino_threshold`).
  - `yolo`: cosine on YOLO fragment centroids when DINO is unavailable or disabled (gate: `relink_threshold`).
- Falls back to spatial plausibility (`spatial`) as a third pass for unresolved pairs (gate: `relink_fallback_threshold`).
- Merges accepted chains into canonical track IDs.
- Records DINO contribution metrics in `relink_manifest.json`: `relink_dino_coverage`, `relink_dino_accepted`, `relink_yolo_accepted`.

---

## Layer Calibration

Layer selection is driven by a separability metric, a Fisher-style ratio measuring how well each layer's activations separate different objects while remaining consistent within an object.
When `--dino` is enabled in the sweep script, `dino_cls` is evaluated as an additional candidate alongside YOLO module layers.

**Metrics computed per layer:**
- `within_var`: mean per-group variance across feature dimensions.
- `between_var`: variance of per-group mean vectors across dimensions.
- `separability = between_var / (within_var + 1e-8)`.
- `track_id_coverage`: fraction of grouped detections using `track_id` (vs class fallback).

**Grouping policy:** use `track_id` when available on detections; fallback to `class_id` unless `--require-track-id` is set.
Instance-level grouping is preferred for re-identification calibration. When `track_id_coverage` is low, separability mostly reflects class-level separation.

**Current calibration context:** the aggregate leaderboard below was produced with class-level fallback (`track_id_coverage = 0.0` across layers). Treat it as class-separability guidance, not instance-level ID calibration.

**Ranking policy:** descending `separability` -> descending `mean_consecutive_cosine` -> ascending `layer_name`.

**Degenerate handling:** layers with fewer than 2 distinct groups receive `separability = 0.0` and emit a warning.
Layers with `track_id_coverage < 0.30` emit warnings, and aggregate winner selection emits a warning if `mean_track_id_coverage < 0.50`.

**Winner selection constraints:**
- Exclude layers with `feature_dim < 32`.
- Deduplicate `.conv` child entries when the parent `Conv` module is already present.

**Current aggregate leaderboard (top 5, constrained):**

| Rank | Layer | Type | Feature Dim | Mean Separability | Mean Cosine | Mean Track Coverage |
|---|---|---:|---:|---:|---:|---:|
| 1 | `2.cv1` | Conv | 32 | 15.911 | 0.9776 | 0.0000 |
| 2 | `1` | Conv | 32 | 15.675 | 0.9440 | 0.0000 |
| 3 | `4.cv1` | Conv | 64 | 15.495 | 0.9673 | 0.0000 |
| 4 | `22.cv3.0` | Sequential | 80 | 13.902 | 0.9981 | 0.0000 |
| 5 | `15` | C2f | 64 | 9.926 | 0.9809 | 0.0000 |

If sweeps are run without tracking-enabled detections, `track_id_coverage` can be `0.0` across layers. In that case, separability rankings should be treated as class-level approximations until sweeps are rerun with stable track IDs (recommended: `--dino --require-track-id`).

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

Configuration below reflects the DINO relink-sidecar run (`R4`): embedding layers `4.cv1 + 15 + 22.cv3.0` with weights `0.549/0.351/0.100`, raw YOLO dim `208`, PCA target `128` (effective dim may be lower on small runs), DINO sidecar dim `384`, `activation_topk=64`, `similarity_threshold=0.70`, `max_centroid_distance=0.40`, `relink_use_dino=true`, `relink_dino_threshold=0.55`, `relink_threshold=0.55`, `relink_max_gap_frames=-1`, `relink_fallback_threshold=0.40`.

| Scenario | Frames | Detections | Ball Tracks | Total Tracks | Valid Tracks | Relink Edges |
|---|---:|---:|---:|---:|---:|---:|
| 10sec_Left_to_Right | 133 | 160 | 1 | 6 | 5 | 1 |
| 3sec_Left_to_Right | 49 | 77 | 1 | 6 | 5 | 1 |
| Exit_frame_while_occluded | 53 | 72 | 1 | 5 | 4 | 0 |
| Left_bounce_back | 64 | 105 | 1 | 5 | 4 | 2 |
| Left_to_right | 25 | 52 | 1 | 6 | 5 | 1 |
| No_occlusion_ball_removed | 34 | 37 | 1 | 9 | 4 | 1 |
| Occlusion_ball_removed | 48 | 114 | 1 | 14 | 8 | 2 |
| Right_to_left | 21 | 40 | 1 | 4 | 3 | 1 |
| **Totals** | **427** | **657** | **8** | **55** | **38** | **9** |

### DINO Relink Threshold Sweep (totals)

All runs used the same configuration as above except `relink_use_dino` / `relink_dino_threshold`.

| Run | relink_use_dino | relink_dino_threshold | Total Tracks | Valid Tracks | Relink Edges | relink_dino_coverage |
|---|---:|---:|---:|---:|---:|---:|
| R0 | false | — | 56 | 39 | 8 | 0.000 |
| R1 | true | 0.40 | 55 | 38 | 9 | 1.000 |
| R2 | true | 0.45 | 55 | 38 | 9 | 1.000 |
| R3 | true | 0.50 | 55 | 38 | 9 | 1.000 |
| R4 | true | 0.55 | 55 | 38 | 9 | 1.000 |
| R5 | true | 0.60 | 56 | 39 | 8 | 1.000 |
| R6 | true | 0.65 | 56 | 39 | 8 | 1.000 |

Winner under constraint (`total_tracks <= R0`) is `R1`; `R1` through `R4` tie on aggregate metrics.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### DINO installation (one-time cache warmup)
```bash
export TORCH_HOME="$PWD/.torch_cache"
export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"

python3 - <<'PY'
import torch
_ = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
print("DINO cached in:", torch.hub.get_dir())
PY
```

Verify weights exist:
```bash
find .torch_cache/hub/checkpoints -name "dino_deitsmall8_pretrain.pth"
```

---

## Run Commands

### 1. Per-video layer sweeps
Instance-level sweep (recommended when detections include stable `track_id`):
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
    --dino \
    --require-track-id \
    --output-csv "experiments/results/layer_selection/per_video/layer_stability_sweep_${stem}.csv"
done
```
Class-level fallback sweep (when `track_id` is unavailable) is the same command without `--require-track-id`.

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
Set `TRACE_DISABLE_DINO=1` to disable DINO sidecar extraction. On first DINO-enabled run, `torch.hub` may download model weights; if unavailable offline and uncached, enrichment logs a clear warning and marks DINO sidecars unavailable for that run while preserving YOLO enrichment.

### 4. Temporal linking (batch)
```bash
for f in experiments/results/activation_enrichment/*/enriched_detections.json; do
  python3 src/run_temporal_linking.py \
    --enriched-json "$f" \
    --activation-topk 64 \
    --similarity-threshold 0.70 \
    --max-centroid-distance 0.40 \
    --relink-threshold 0.55 \
    --relink-dino-threshold 0.55 \
    --relink-max-gap-frames -1 \
    --relink-fallback-threshold 0.40
done
```
Add `--no-relink-dino` to force YOLO relink scoring.

### 5. DINO relink threshold sweep (R0..R6)
```bash
python3 experiments/run_dino_param_search.py \
  --enrichment-root experiments/results/activation_enrichment \
  --output-root experiments/results/param_search
```
Outputs:
- `summary.csv` with `run_id,relink_dino_threshold,relink_use_dino,total_tracks,valid_tracks,relink_edges,relink_dino_accepted,relink_yolo_accepted,relink_dino_coverage,fragmentation_ratio,delta_valid_vs_baseline`
- Per-run scenario artifacts under `param_search/R0` ... `param_search/R6`

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
  param_search/
    summary.csv
    R0/
      <scenario>/...
    R1/
      <scenario>/...
    ...
    R6/
      <scenario>/...
```
