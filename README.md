# Object Permanence

Single-pass YOLOv8 detection enrichment and offline temporal linking for object permanence experiments.

## Project Scope

This repository currently runs two stages:

1. **Trace Enrichment** (`src/trace_enrichment/`)
2. **Temporal Linking** (`src/temporal_linking/`)

The system is designed for offline video processing over complete recorded videos.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Architecture

### Stage 1: Trace Enrichment

Input:
- Raw video frames

Core responsibilities:
- Sample every `N`th frame
- Run YOLOv8 detections
- Capture detection-head activations from a configured hook layer
- Build per-detection activation vectors
- Fit PCA and project vectors to 256 dims
- L2-normalize projected vectors

Output artifacts per video:
- `experiments/results/enriched/<video_name>/enriched_detections.json`
- `experiments/results/enriched/<video_name>/pca_projection.pkl`
- `experiments/results/enriched/<video_name>/projection_manifest.json`

Entrypoint:
- `src/run_pipeline.py`

### Stage 2: Temporal Linking

Input:
- `enriched_detections.json` from Stage 1

Core responsibilities:
- Sweep 1: walk frames in temporal order and assign detections to tracks
- Keep track lifecycle state (`TENTATIVE`, `ACTIVE`, `LOST`, `CLOSED`)
- Sweep 2: relink compatible closed fragments across gaps
- Produce final merged track IDs in output artifacts

Matching policy:
- **Single gating threshold:** `similarity_threshold`
- Pair is eligible only if visual similarity is above threshold
- Same threshold applies to normal links and lost-track recovery
- Spatial and consistency terms are secondary tie-break terms
- **Relink controls:** `relink_threshold`, `relink_max_gap_frames`, `relink_min_track_hits`, `relink_max_pixels_per_frame`, `relink_fallback_threshold`

Output artifacts per video:
- `experiments/results/enriched/<video_name>/linked_detections.json`
- `experiments/results/enriched/<video_name>/tracks.json`
- `experiments/results/enriched/<video_name>/linking_manifest.json`
- `experiments/results/enriched/<video_name>/relink_manifest.json`

Entrypoint:
- `src/run_temporal_linking.py`

## How It Works

### End-to-End Flow

1. Run enrichment on a video (or batch of videos).
2. Inspect/validate enriched activation outputs.
3. Run temporal linking on each enriched trace.
4. Validate linked outputs and track statistics.

### Temporal Linking Sweeps

1. **Primary sweep (frame-by-frame):**
- Builds track continuity over adjacent sampled frames using visual similarity plus tie-break features.

2. **Relink sweep (post-hoc):**
- Evaluates closed track fragments of the same class.
- Uses centroid similarity first, then fallback spatial plausibility scoring for unresolved pairs.
- Applies accepted links as merged final track IDs.

### Enriched Detection Schema (core fields)

```json
{
  "class_id": 32,
  "class_name": "sports ball",
  "bbox": [41.71, 532.64, 322.24, 779.72],
  "confidence": 0.818,
  "activation": {
    "vector": [0.013, -0.224],
    "dim": 256,
    "layers": ["head.cv3[2]"],
    "pool": "adaptive_avg_3x3",
    "projection": "pca_256",
    "small_crop_flag": false
  }
}
```

### Linked Detection Additions (core fields)

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

## Usage

### Run Trace Enrichment

Single video:

```bash
python3 src/run_pipeline.py \
  --video data/raw_videos/3sec_Left_to_Right.mp4 \
  --sample-rate 5 \
  --model yolov8n.pt
```

Batch videos:

```bash
python3 src/run_pipeline.py \
  --video-dir data/raw_videos \
  --pattern "*.mp4" \
  --sample-rate 5 \
  --model yolov8n.pt
```

### Validate Enriched Output

```bash
python3 -m src.trace_enrichment.validate \
  experiments/results/enriched/3sec_Left_to_Right/enriched_detections.json \
  --expected-dim 256
```

### Run Temporal Linking

```bash
python3 src/run_temporal_linking.py \
  --enriched-json experiments/results/enriched/3sec_Left_to_Right/enriched_detections.json \
  --similarity-threshold 0.65 \
  --relink-threshold 0.55 \
  --relink-fallback-threshold 0.40
```

Disable relink merges (no-op sweep) by setting strict thresholds:

```bash
python3 src/run_temporal_linking.py \
  --enriched-json experiments/results/enriched/3sec_Left_to_Right/enriched_detections.json \
  --similarity-threshold 0.65 \
  --relink-threshold 1.0 \
  --relink-fallback-threshold 1.0
```

### Relink Options

- `--relink-threshold` (default `0.55`): centroid similarity gate for relink acceptance.
- `--relink-max-gap-frames` (default `-1`): max allowed frame gap between fragments (`-1` means no cap).
- `--relink-min-track-hits` (default `2`): minimum observations required for a fragment to be relink-eligible.
- `--relink-max-pixels-per-frame` (default `15.0`): maximum plausible drift rate (pixels/frame) for spatial fallback scoring.
- `--relink-fallback-threshold` (default `0.40`): fallback score gate after centroid pass.

### Calibration Note

Long-gap calibration on `Right_to_left` showed that cross-track percentile cosine scores remained unstable for the sports-ball 8→18 gap.
The fallback pass therefore uses spatial plausibility (trajectory extrapolation + drift-rate normalization) rather than percentile cosine.

### Output Files

- `linked_detections.json`: per-detection temporal link metadata with final track IDs.
- `tracks.json`: merged final tracks and per-track summary fields.
- `linking_manifest.json`: run config and aggregate stage statistics.
- `relink_manifest.json`: relink diagnostics (candidate counts, accepted links, merge map).

### Validate Temporal Linking Output

```bash
python3 -m src.temporal_linking.validate \
  experiments/results/enriched/3sec_Left_to_Right/linked_detections.json \
  experiments/results/enriched/3sec_Left_to_Right/tracks.json \
  experiments/results/enriched/3sec_Left_to_Right/linking_manifest.json
```

## What We Are Working On Next

1. **Occlusion continuity:** improve identity recovery when objects re-emerge after long occlusions.
2. **Threshold calibration workflow:** define repeatable tuning protocol for `similarity_threshold` across videos.
3. **Sampling strategy evaluation:** compare `sample-rate` settings to reduce identity fragmentation.
4. **Tracking quality metrics:** add aggregate metrics for ID continuity and track fragmentation in manifests/reports.
