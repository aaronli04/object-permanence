# Object Permanence

Single-pass YOLOv8 detection + activation trace enrichment for object permanence experiments.

This repository's current scope is only enriched trace generation:
- sampled-frame detections
- backbone activation embeddings
- PCA projection artifacts and provenance

It does not implement temporal linking, tracking, re-identification, occlusion handling, or any later reasoning stage.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run Trace Enrichment

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

Batch mode processes videos sequentially, logs progress per video, continues on failures, and exits nonzero if any video fails.

### Optional: Discover Hook Layers

```bash
python3 -m src.trace_enrichment.discover_layers --model yolov8n.pt
```

Typical YOLOv8n/YOLOv8m defaults used by the pipeline:
- deep layer: `8` (stride `32`)
- mid layer: `6` (stride `16`)

### Validate Enriched Output

```bash
python3 -m src.trace_enrichment.validate \
  experiments/results/enriched/3sec_Left_to_Right/enriched_detections.json \
  --expected-dim 256
```

## Outputs

For input video `<video_name>.mp4`, the pipeline writes:

- `experiments/results/enriched/<video_name>/enriched_detections.json`
- `experiments/results/enriched/<video_name>/pca_projection.pkl`
- `experiments/results/enriched/<video_name>/projection_manifest.json`

`enriched_detections.json` is the main downstream artifact and preserves per-frame detections with activation metadata.

`pca_projection.pkl` is the fitted PCA model for that video's activation vectors.

`projection_manifest.json` records provenance and projection settings (model, hook layers/strides, pooling strategy, PCA dims, sample rate, counts, and input video hash).

## How It Works

The pipeline is single-pass with respect to YOLO inference on sampled frames:

1. Sample every Nth frame from the input video.
2. Run YOLOv8 on batches of sampled frames.
3. Capture backbone feature maps via forward hooks during the same forward pass.
4. For each detection, crop feature ROIs from the hooked maps using the same detection bbox.
5. Adaptive-average-pool each ROI to `3x3`, flatten, and concatenate (deep + mid) into a raw activation vector.
6. Fit PCA across all detections in the trace.
7. Project to 256 dimensions (or fewer fitted components padded to 256), then L2-normalize.
8. Write the enriched trace and PCA provenance artifacts.

No IoU reconciliation or internal artifact "linking" is used. Detection decisions and activation vectors come from the same forward pass.

## Enriched Detection Schema (Activation Payload)

Each detection in `enriched_detections.json` includes:

```json
{
  "class_id": 32,
  "class_name": "sports ball",
  "bbox": [41.71, 532.64, 322.24, 779.72],
  "confidence": 0.818,
  "activation": {
    "vector": [0.013, -0.224],
    "dim": 256,
    "layers": ["backbone.C2f_deep", "backbone.C2f_mid"],
    "pool": "adaptive_avg_3x3",
    "projection": "pca_256",
    "small_crop_flag": false
  }
}
```
