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

## Technical Pipeline Details

### Stage 1: Activation Enrichment

Input:
- Raw video frames from `data/raw_videos/*.mp4`

Core mechanics:
- Hook target defaults to `model.model[15]` (neck C2f block).
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
  - Zero-pad back to 256-D (for schema/compatibility)
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

Output per scenario:
- `experiments/results/linking/<scenario>/linked_detections.json`
- `experiments/results/linking/<scenario>/tracks.json`
- `experiments/results/linking/<scenario>/linking_manifest.json`
- `experiments/results/linking/<scenario>/relink_manifest.json`

Entrypoint:
- `src/run_temporal_linking.py`

## Why These Defaults (Data-Backed)

### Layer Selection Evidence

A layer-stability sweep was run on `Right_to_left` (`sample_rate=5`, first 20 sampled frames, sports ball class only, conf > 0.25).

Observed:
- Sampled frames: `20`
- Sports-ball detections used: `10`
- Named modules scanned: `224`
- Eligible layers: `154`

Top layers by temporal stability (`mean_consecutive_cosine`):

| Rank | Layer | Type | Feature Dim | Mean Consecutive Cosine | Norm Std |
|---|---|---|---:|---:|---:|
| 1 | `22.dfl.conv` | `Conv2d` | 1 | 1.000000 | 0.321435 |
| 2 | `22.cv3.0` | `Sequential` | 80 | 0.999654 | 4.387048 |
| 3 | `22.cv3.0.2` | `Conv2d` | 80 | 0.999654 | 4.387048 |
| 4 | `0` | `Conv` | 16 | 0.999556 | 0.201582 |
| 5 | `0.conv` | `Conv2d` | 16 | 0.999556 | 0.201582 |

Selected default layer:
- `model.model[15]` (`neck.C2f.15`)
- Stability result: mean cosine `0.995373`, norm std `0.042314`, feature dim `64` (typically ranked in the 30s/154 on this sweep; exact rank shifts with tied scores)

Selection rationale:
- Not the absolute max cosine layer, but a better trade-off between descriptor dimensionality and stability than very compressed heads (for example dim `1`).
- Produced strong downstream linking behavior across scenarios.

### Top-K Selection Evidence

Top-k cosine sweep on `Right_to_left` sports-ball descriptors:
- Ball vectors used: `11` (frames `0,5,10,15,70,75,80,85,90,95,100`)
- Metrics:
  - `within_early`: mean pairwise cosine among early frames
  - `within_late`: mean pairwise cosine among late frames
  - `cross`: mean cosine between early vs late groups

To avoid low-dimensional over-bias, we enforced `k >= 12`. Under that constraint:
- Best `cross` occurs at `k = 12`
- `within_early = 0.837967`
- `within_late = 0.788560`
- `cross = 0.714120`

Chosen default:
- `--activation-topk 12`

### Reproduce Calibration Commands

Layer stability sweep (all named modules, sports ball only, first 20 sampled frames):

```bash
python3 experiments/layer_stability_sweep.py \
  --video data/raw_videos/Right_to_left.mp4 \
  --model yolov8n.pt \
  --sample-rate 5 \
  --max-sampled-frames 20 \
  --class-id 32 \
  --min-confidence 0.25 \
  --output-csv experiments/results/layer_stability_sweep_right_to_left.csv
```

Top-k sweep input extraction (class-filtered layer 15 vectors):

```bash
python3 experiments/extract_object_vectors.py \
  --enriched-json experiments/results/activation_enrichment/Right_to_left/enriched_detections.json \
  --output-json experiments/results/activation_enrichment/Right_to_left/object_vectors_layer15.json \
  --class-id 32 \
  --min-confidence 0.25
```

Top-k cosine sweep CSV (`k=2..40`, step 2):

```bash
python3 experiments/analyze_topk_dims.py \
  --input-json experiments/results/activation_enrichment/Right_to_left/object_vectors_layer15.json \
  --output-csv experiments/results/topk_similarity_sweep_layer15.csv \
  --skip-plot \
  --min-k 2 \
  --max-k 40 \
  --step-k 2 \
  --early-frames 0,5,10,15 \
  --late-frames 70,75,80,85,90,95,100
```

### Scenario Outcomes with Current Baseline

Current baseline:
- Enrichment: `head-layer=15`, `head-stride=8`, pool `1x1`
- Linking: `activation_topk=12`, `similarity_threshold=0.65`, `relink_threshold=0.55`, `relink_fallback_threshold=0.40`

Results across scenarios:

| Scenario | Frames | Detections | Ball Tracks | Tracks Total | Tracks Valid |
|---|---:|---:|---:|---:|---:|
| `10sec_Left_to_Right` | 133 | 160 | 1 | 6 | 5 |
| `3sec_Left_to_Right` | 49 | 77 | 1 | 7 | 5 |
| `Exit_frame_while_occluded` | 53 | 72 | 1 | 5 | 4 |
| `Left_bounce_back` | 64 | 105 | 1 | 5 | 4 |
| `Left_to_right` | 25 | 52 | 1 | 7 | 5 |
| `No_occlusion_ball_removed` | 34 | 37 | 2 | 9 | 6 |
| `Occlusion_ball_removed` | 48 | 114 | 1 | 14 | 8 |
| `Right_to_left` | 21 | 40 | 1 | 5 | 3 |

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
  - Max allowed frame gap for relink candidate pairs (`-1` means unlimited).
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
  --head-layer 15 \
  --head-stride 8
```

### Activation Enrichment (Batch over Raw Videos)

```bash
python3 src/run_pipeline.py \
  --video-dir data/raw_videos \
  --pattern "*.mp4" \
  --sample-rate 5 \
  --model yolov8n.pt \
  --head-layer 15 \
  --head-stride 8
```

### Temporal Linking (Single Scenario)

```bash
python3 src/run_temporal_linking.py \
  --enriched-json experiments/results/activation_enrichment/3sec_Left_to_Right/enriched_detections.json \
  --activation-topk 12 \
  --similarity-threshold 0.65 \
  --relink-threshold 0.55 \
  --relink-fallback-threshold 0.40
```

### Temporal Linking (Batch over All Enriched Scenarios)

```bash
for f in experiments/results/activation_enrichment/*/enriched_detections.json; do
  python3 src/run_temporal_linking.py \
    --enriched-json "$f" \
    --activation-topk 12 \
    --similarity-threshold 0.65 \
    --relink-threshold 0.55 \
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
  --head-layer 15 \
  --head-stride 8

for f in experiments/results/activation_enrichment/*/enriched_detections.json; do
  python3 src/run_temporal_linking.py \
    --enriched-json "$f" \
    --activation-topk 12 \
    --similarity-threshold 0.65 \
    --relink-threshold 0.55 \
    --relink-fallback-threshold 0.40
done
```

### Optional No-Merge Relink Mode

```bash
python3 src/run_temporal_linking.py \
  --enriched-json experiments/results/activation_enrichment/3sec_Left_to_Right/enriched_detections.json \
  --activation-topk 12 \
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
    "layers": ["neck.C2f.15"],
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

## Repository Hygiene

- `experiments/results/` is intentionally gitignored.
- Keep benchmark/progress narrative in this README + output manifests.
- Keep ad-hoc analysis scratch files out of repo root.
