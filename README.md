# Object Permanence

Single-pass YOLOv8 activation enrichment plus offline temporal linking for object permanence experiments.

## How This System Works

This project runs a strict two-stage offline pipeline:

1. `trace_enrichment`: detect objects and attach activation descriptors to each detection.
2. `temporal_linking`: connect detections across sampled frames into track IDs.

High-level data flow:

1. Sample every `N`th frame from a video.
2. Run YOLOv8 on sampled frames.
3. For each detection, extract a feature vector from a hooked internal layer.
4. Fit PCA on collected per-detection vectors and project to 256-D.
5. Link detections over time using cosine similarity plus tie-break scores.
6. Run a second relink pass for fragmented closed tracks.

## Intuition Behind Linking

### Vectors

When a detection is produced, the pipeline also extracts an internal appearance descriptor from YOLO features.  
By default, enrichment uses a multi-layer weighted embedding built from three tiers:
- `4.cv1` (appearance tier, 64 dims)
- `15` (semantic tier, 64 dims)
- `22.cv3.0` (class tier, 80 dims)

Each layer vector is pooled, L2-normalized, weighted by separability-derived weight, concatenated, and L2-normalized again.  
Raw combined descriptor dim is `208` (`64 + 64 + 80`).

At enrichment time:
- Raw vectors are projected with PCA and stored in a fixed 256-D schema.
- Effective signal is capped by fitted PCA components (at most 208, often lower on short clips).

At linking time (default):
- `--activation-topk 64` keeps the first 64 projected dims.
- Vectors are L2-normalized before cosine scoring.

### Cosine similarity

Each descriptor is treated as a vector in feature space. Cosine similarity asks whether two vectors point in the same direction:

- `1.0`: same direction (very similar appearance)
- `0.0`: orthogonal (unrelated)
- `-1.0`: opposite direction

Because vectors are L2-normalized, cosine becomes a dot product:

`cos(a, b) = dot(a, b)`

That is the primary visual identity score used by the tracker.

### Frame-to-frame matching

For every sampled frame pair, the tracker scores all candidate track-detection pairs, then applies two constraints:

- Hard gate first: pair is ineligible if `visual_similarity < 0.65` (`--similarity-threshold`).
- One-to-one matching: each detection can match at most one track, and each track can match at most one detection.

Assignment is solved with Hungarian by default (`--assignment-method hungarian`), which maximizes total assignment quality globally rather than picking greedy local matches.

### Track memory

A track reference vector is not only "last seen descriptor". It is:

`ref = normalize(w_last * last + w_ema * ema + w_hist * history_mean)`

Defaults:
- `w_last = 0.55`
- `w_ema = 0.30`
- `w_hist = 0.15`

This reduces brittleness from single-frame artifacts (blur, partial occlusion, odd pose).

### Relinking

If a track is closed and the object later reappears, relink tries to merge fragments:

1. Same class and temporal order (successor starts after predecessor ends).
2. Centroid descriptor cosine gate (`--relink-threshold`, default `0.55`).
3. If centroid gate fails, spatial fallback score (`--relink-fallback-threshold`, default `0.40`).
4. Gap control via `--relink-max-gap-frames` (default `-1`, unlimited).

Accepted edges are merged into canonical track IDs.

### End-to-end summary

Every sampled frame:

detect -> extract descriptor -> compare by cosine -> Hungarian assignment -> lifecycle updates (`TENTATIVE/ACTIVE/LOST/CLOSED`) -> relink closed fragments.

Goal: maintain identity continuity through motion, misses, and short occlusions.

## Pipeline Details

### Stage 1: Activation Enrichment

Input:
- Raw video frames from `data/raw_videos/*.mp4`

Core mechanics:
- Default embedding layers and weights are defined in `src/trace_enrichment/constants.py` (`EMBEDDING_LAYERS`).
- Single-layer fallback remains available via `--head-layer` when multi-layer embedding is disabled.
- For each detection bbox `[x1, y1, x2, y2]` on a frame `(frame_w, frame_h)`, map to feature-map ROI `(fmap_w, fmap_h)`:
  - `fx1 = floor((x1 / frame_w) * fmap_w)`
  - `fy1 = floor((y1 / frame_h) * fmap_h)`
  - `fx2 = ceil((x2 / frame_w) * fmap_w)`
  - `fy2 = ceil((y2 / frame_h) * fmap_h)`
- Crop feature tensor `[C, H, W]` to ROI.
- Adaptive average pool to `1x1`, yielding a `C`-dim raw vector.
- Fit PCA over all raw vectors in the run, project to 256-D.
- L2-normalize the projected vectors before writing output JSON.

Output per scenario:
- `experiments/results/activation_enrichment/<scenario>/enriched_detections.json`
- `experiments/results/activation_enrichment/<scenario>/pca_projection.pkl`
- `experiments/results/activation_enrichment/<scenario>/projection_manifest.json`

Entrypoint:
- `src/run_pipeline.py`

### Stage 2: Temporal Linking

Input:
- `enriched_detections.json` from Stage 1

Core mechanics:
- Optional descriptor truncation via `--activation-topk K`:
  - Keep first `K` dims of each 256-D vector
  - L2-renormalize
  - Link directly in K-dimensional space (no internal zero-padding)
- Track reference descriptor is a weighted blend:
  - `ref = normalize(w_last * last_vec + w_ema * ema_vec + w_hist * hist_mean)`
- Match score components:
  - Visual cosine similarity (`dot(ref, det_vec)`)
  - Spatial tie-break score (`IoU + center-distance term`)
  - Consistency and lost-age tie-break terms
- Eligibility gate:
  - Pair is eligible only if `visual_similarity >= similarity_threshold`
- Assignment:
  - Hungarian (default) or greedy over eligible pairs
- Lifecycle:
  - `TENTATIVE -> ACTIVE -> LOST -> CLOSED`
- Relink sweep:
  - Evaluate closed fragment pairs (same class, temporal order)
  - Pass 1: centroid cosine with `relink_threshold`
  - Pass 2: spatial plausibility fallback with `relink_fallback_threshold`
  - One-to-one greedy acceptance + chain resolution

#### Frame-to-Frame Link Score

For a track `t` and detection `d`, the score is built as:

`h_t = mean(vec_history_t)` (or `last_t` if no history)  
`ref_t = normalize(w_last * last_t + w_ema * ema_t + w_hist * h_t)`

`v_{t,d} = ref_t . a_d`  
where `a_d` is the detection activation vector.

Spatial term:

`IoU_{t,d} = IoU(last_bbox_t, bbox_d)`  
`dist_{t,d} = ||center(last_bbox_t) - center(bbox_d)||_2`  
`scale_{t,d} = max(diag(last_bbox_t), diag(bbox_d), 1)`  
`c_{t,d} = exp(-dist_{t,d} / scale_{t,d})`  
`s_{t,d} = 0.5 * IoU_{t,d} + 0.5 * c_{t,d}`

Consistency and age terms:

`k_{t,d} = mean(sim_history_t)` (or `v_{t,d}` if no history)  
`age_t = exp(-miss_streak_t / max(max_lost_frames, 1))`

Tie-break:

`tie_{t,d} = w_spatial * s_{t,d} + w_consistency * k_{t,d} + w_age * age_t`

Eligibility gate:

`eligible_{t,d} = class_ok AND (v_{t,d} >= similarity_threshold)`

Final pair score:

`score_{t,d} = v_{t,d} + tie_{t,d}` if eligible, else `-inf`.

How scores become links:
- Hungarian (default): solve a one-to-one assignment that maximizes total score.
  Equivalent implementation is min-cost with `cost = -score` for eligible pairs.
- Greedy: sort eligible pairs by descending score and keep non-conflicting pairs.

#### Relink Internals

Relinking runs after all tracks are closed and only uses fragments with:
- closed status
- at least `relink_min_track_hits` observations
- descriptor history and position history present.

Candidate pair `(p -> q)` must satisfy:

`same_class`  
`p.last_frame < q.first_frame`  
`gap_ok = (relink_max_gap_frames == -1) OR ((q.first_frame - p.last_frame) <= relink_max_gap_frames)`

Centroid relink score:

`centroid_f = normalize(mean(normalize(obs_vec_i for fragment f)))`  
`centroid_score_{p,q} = centroid_p . centroid_q`

Spatial fallback score:

Estimate predecessor velocity `(vx, vy)` from its recent positions over time.  
For gap `g = max(1, q.first_frame - p.last_frame)`:

`x_hat = x_last + vx * g`  
`y_hat = y_last + vy * g`

`e = ||(x_hat, y_hat) - (x_q_start, y_q_start)||_2`  
`spatial_score_{p,q} = 1 - ((e / g) / relink_max_pixels_per_frame)`

Acceptance logic:
- Pass 1: accept centroid edges with `centroid_score >= relink_threshold`, one-to-one greedy.
- Pass 2: for unresolved nodes, accept spatial edges with `spatial_score >= relink_fallback_threshold`, one-to-one greedy.

Chain merge:
- Accepted edges are unioned into chains.
- Canonical track ID is the member with earliest `first_frame` (tie-break: smaller `track_id`).
- `merge_map[absorbed] = canonical`.

Serialization after relink:
- `linked_detections.json`: remap `temporal_link.track_id` through `merge_map`.
- `tracks.json`: merge absorbed members into canonical track and record absorbed IDs in `relinked_from`.
- `relink_manifest.json`: stores relink config, candidate/acceptance stats, accepted edges, and `merge_map`.

Output per scenario:
- `experiments/results/linking/<scenario>/linked_detections.json`
- `experiments/results/linking/<scenario>/tracks.json`
- `experiments/results/linking/<scenario>/linking_manifest.json`
- `experiments/results/linking/<scenario>/relink_manifest.json`

Entrypoint:
- `src/run_temporal_linking.py`

## Why These Defaults

### Layer Selection Evidence

A separability-first layer sweep is now used for layer selection.

Each layer reports:
- `within_var`: mean per-group variance across feature dimensions.
- `between_var`: variance of group mean vectors, averaged across dimensions.
- `separability = between_var / (within_var + 1e-8)`.

Grouping key:
- `track_id` when available.
- fallback to class label otherwise.

Calibration setup:
- Videos: all `data/raw_videos/*.mp4` (8 scenarios)
- Sweep config: `sample_rate=5`, first `20` sampled frames, `class-id=-1`, `conf > 0.25`
- Winner constraints: `feature_dim >= 32` and drop `.conv` layers when parent Conv layer exists
- Per-video outputs: `experiments/results/layer_selection/per_video/layer_stability_sweep_<video>.csv`
- Aggregate artifact: `experiments/results/layer_selection/aggregate/aggregate_separability.csv`
- Ranking: descending `separability`, tie-break by `mean_consecutive_cosine`

Top aggregate layers:

| Rank | Layer | Type | Feature Dim | Mean Separability | Mean Consecutive Cosine |
|---|---|---|---:|---:|---:|
| 1 | `2.cv1` | `Conv` | 32 | 15.911515 | 0.977638 |
| 2 | `1` | `Conv` | 32 | 15.674581 | 0.944006 |
| 3 | `4.cv1` | `Conv` | 64 | 15.495382 | 0.967329 |
| 4 | `22.cv3.0` | `Sequential` | 80 | 13.901690 | 0.998128 |
| 5 | `22.cv3.0.2` | `Conv2d` | 80 | 13.901690 | 0.998128 |

Default multi-layer embedding (stored as `EMBEDDING_LAYERS` in `src/trace_enrichment/constants.py`):
- `4.cv1` (appearance): separability `15.495` -> weight `0.398`
- `15` (semantic): separability `9.926` -> weight `0.255`
- `22.cv3.0` (class): separability `13.902` -> weight `0.357`
- Weights are re-normalized at runtime to sum to `1.0` after any layer-availability fallback.

Selection rationale:
- Separability directly captures between-object discriminability versus within-object compactness.
- Layer tiers contribute complementary information (texture/appearance, semantic structure, class-level cues).
- If calibration rankings materially change, update `EMBEDDING_LAYERS` from the aggregate sweep artifact.

### Top-K Selection Evidence

Top-k cosine sweep on `Right_to_left` sports-ball descriptors:
- Ball vectors used: `11` (frames `0,5,10,15,70,75,80,85,90,95,100`)
- Metrics:
  - `within_early`: mean pairwise cosine among early frames
  - `within_late`: mean pairwise cosine among late frames
  - `cross`: mean cosine between early vs late groups

Sweeps were run to `k=256` to test both low-dimensional and high-dimensional regimes.

Top-k experiment summary:

Right_to_left linking outcomes for each `k`:

| k | within_early | within_late | cross | ball_tracks | total_tracks | valid_tracks | Notes |
|---:|---:|---:|---:|---:|---:|---:|---|
| 12 | 0.837967 | 0.788560 | 0.714120 | 1 | 5 | 3 | Best `cross` among `k >= 12` |
| 64 | 0.828755 | 0.771111 | 0.704412 | 1 | 5 | 3 | Legacy single-layer calibration reference (`neck.C2f.15`) |

Current dimension cap with default multi-layer embedding:
- `raw_activation_dim = 208` (`64 + 64 + 80`) when all embedding layers are available.
- Effective information is capped by fitted PCA components per run.

Cross-scenario linking behavior (`k=64` vs prior `k=12` outputs):
- Ball-track count changed in `2/8` scenarios:
  - `Left_bounce_back`: `1 -> 2`
  - `No_occlusion_ball_removed`: `2 -> 1`
- `k=64`, `k=128`, and `k=256` matched across all tested scenarios.

Chosen default:
- `--activation-topk 64`

### Reproduce Calibration Commands

Per-video separability sweeps (all classes):

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

Aggregate all per-video sweep CSVs and persist calibration record:

```bash
python3 experiments/aggregate_layer_sweeps.py \
  --input-glob "experiments/results/layer_selection/per_video/layer_stability_sweep_*.csv" \
  --output-csv experiments/results/layer_selection/aggregate/aggregate_separability.csv \
  --winner-min-feature-dim 32 \
  --top-n 20
```

After re-running calibration, update `EMBEDDING_LAYERS` in `src/trace_enrichment/constants.py`.

### Scenario Outcomes with Current Baseline

Current baseline:
- Enrichment: `embedding_layers=[4.cv1,15,22.cv3.0]` with separability weights `[0.398,0.255,0.357]`, pool `1x1`
- Linking: `activation_topk=64`, `similarity_threshold=0.65`, `relink_threshold=0.55`, `relink_max_gap_frames=-1`, `relink_fallback_threshold=0.40`

### Caveats and Guardrails

- Top-k truncation math:
  - Cosine is computed on normalized truncated vectors directly (K dims).
  - If vectors are both truncated to K and then zero-padded equally, cosine is unchanged; we keep K-dim compute for clarity.
- PCA fit scope:
  - PCA is fit per run over all detections in that run (not globally across scenarios).
  - Low detection counts can make projections noisier; this is now flagged in `projection_manifest.json` via `projection_fit_scope` and `projection_caveats`.
- Relink gap:
  - Default `--relink-max-gap-frames` is `-1` (unlimited relinking distance).
  - Use a finite value when you want to constrain long-gap merges.

Results across scenarios:

| Scenario | Frames | Detections | Ball Tracks | Tracks Total | Tracks Valid | Relink Edges Accepted |
|---|---:|---:|---:|---:|---:|---:|
| `10sec_Left_to_Right` | 133 | 160 | 1 | 6 | 5 | 1 |
| `3sec_Left_to_Right` | 49 | 77 | 1 | 7 | 5 | 1 |
| `Exit_frame_while_occluded` | 53 | 72 | 1 | 5 | 4 | 0 |
| `Left_bounce_back` | 64 | 105 | 1 | 5 | 4 | 2 |
| `Left_to_right` | 25 | 52 | 1 | 7 | 5 | 1 |
| `No_occlusion_ball_removed` | 34 | 37 | 1 | 8 | 5 | 2 |
| `Occlusion_ball_removed` | 48 | 114 | 1 | 14 | 8 | 2 |
| `Right_to_left` | 21 | 40 | 1 | 5 | 3 | 1 |

## Threshold and Parameter Reference

### Linking Gating and Relink Thresholds

- `--similarity-threshold`:
  - Primary hard gate in frame-to-frame assignment.
  - If visual cosine is below this value, pair is ineligible regardless of tie-break terms.
- `--relink-threshold`:
  - Centroid cosine threshold for closed-track relink pass.
- `--relink-fallback-threshold`:
  - Spatial plausibility threshold for fallback relink scoring.
- `--relink-max-gap-frames`:
  - Max allowed frame gap for relink candidate pairs (default `-1` for unlimited).
- `--relink-min-track-hits`:
  - Min observations required for track fragment to participate in relinking.
- `--relink-max-pixels-per-frame`:
  - Drift-rate cap used to convert motion prediction error into fallback score.

### Linking Score Weight Parameters

- `--w-last`: weight for last observed descriptor.
- `--w-ema`: weight for descriptor EMA.
- `--w-hist`: weight for mean descriptor history.
- `--w-spatial`: weight of spatial tie-break signal.
- `--w-consistency`: weight of similarity-history consistency.
- `--w-age`: weight of age decay bonus for lost tracks.

### Lifecycle and Assignment Controls

- `--activation-topk`:
  - Activation descriptor truncation dimension used for linking (default `64`; effectively capped by raw descriptor dim, currently `208` for the default multi-layer embedding).
- `--max-lost-frames`: close track after this many missed sampled frames.
- `--min-hits-to-activate`: tentative -> active promotion threshold.
- `--min-track-length`: validity threshold in summary reporting.
- `--history-size`: history window for descriptor/similarity buffers.
- `--ema-alpha`: EMA update coefficient.
- `--assignment-method`: `hungarian` or `greedy`.
- `--match-within-class`: class-constrained matching toggle.
- `--filter-short-tracks-in-summary`: whether summary excludes short tracks.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Commands

### Activation Enrichment (Single Scenario)

```bash
python3 src/run_pipeline.py \
  --video data/raw_videos/3sec_Left_to_Right.mp4 \
  --sample-rate 5 \
  --model yolov8n.pt \
  --head-layer 2.cv1 \
  --head-stride 8
```

### Activation Enrichment (Batch over Raw Videos)

```bash
python3 src/run_pipeline.py \
  --video-dir data/raw_videos \
  --pattern "*.mp4" \
  --sample-rate 5 \
  --model yolov8n.pt \
  --head-layer 2.cv1 \
  --head-stride 8
```

### Activation Enrichment (Force Single-Layer Fallback)

```bash
TRACE_DISABLE_MULTI_LAYER_EMBEDDING=1 \
python3 src/run_pipeline.py \
  --video data/raw_videos/3sec_Left_to_Right.mp4 \
  --sample-rate 5 \
  --model yolov8n.pt \
  --head-layer 2.cv1 \
  --head-stride 8
```

### Temporal Linking (Single Scenario)

```bash
python3 src/run_temporal_linking.py \
  --enriched-json experiments/results/activation_enrichment/3sec_Left_to_Right/enriched_detections.json \
  --activation-topk 64 \
  --similarity-threshold 0.65 \
  --relink-threshold 0.55 \
  --relink-max-gap-frames -1 \
  --relink-fallback-threshold 0.40
```

### Temporal Linking (Batch over All Enriched Scenarios)

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

### End-to-End Batch (Both Stages)

```bash
python3 src/run_pipeline.py \
  --video-dir data/raw_videos \
  --pattern "*.mp4" \
  --sample-rate 5 \
  --model yolov8n.pt \
  --head-layer 2.cv1 \
  --head-stride 8

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

### Optional No-Merge Relink Mode

```bash
python3 src/run_temporal_linking.py \
  --enriched-json experiments/results/activation_enrichment/3sec_Left_to_Right/enriched_detections.json \
  --activation-topk 64 \
  --similarity-threshold 0.65 \
  --relink-threshold 1.0 \
  --relink-fallback-threshold 1.0
```

## Validation Commands

### Validate Enrichment Output

```bash
python3 -m src.trace_enrichment.validate \
  experiments/results/activation_enrichment/3sec_Left_to_Right/enriched_detections.json \
  --expected-dim 256
```

### Validate Linking Output

```bash
python3 -m src.temporal_linking.validate \
  experiments/results/linking/3sec_Left_to_Right/linked_detections.json \
  experiments/results/linking/3sec_Left_to_Right/tracks.json \
  experiments/results/linking/3sec_Left_to_Right/linking_manifest.json
```

## Output Layout

```text
experiments/results/
  layer_selection/
    per_video/layer_stability_sweep_<scenario>.csv
    aggregate/aggregate_separability.csv
  activation_enrichment/<scenario>/
    enriched_detections.json
    pca_projection.pkl
    projection_manifest.json
  linking/<scenario>/
    linked_detections.json
    tracks.json
    linking_manifest.json
    relink_manifest.json
```

## Output Schema Highlights

Enriched detection payload (core fields):

```json
{
  "class_id": 32,
  "class_name": "sports ball",
  "bbox": [41.71, 532.64, 322.24, 779.72],
  "confidence": 0.818,
  "activation": {
    "vector": [0.013, -0.224],
    "dim": 256,
    "layers": ["4.cv1", "15", "22.cv3.0"],
    "pool": "adaptive_avg_1x1",
    "projection": "pca_256",
    "small_crop_flag": false
  }
}
```

Linked detection additions:

```json
{
  "det_index": 0,
  "temporal_link": {
    "track_id": 12,
    "track_status": "active",
    "source_track_status": "lost",
    "visual_similarity": 0.744,
    "spatial_score": 0.602,
    "total_score": 0.901,
    "age_since_seen": 0
  }
}
```
