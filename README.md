# Object Permanence

YOLOv8-based video detection and Phase 1 activation enrichment for object permanence reasoning.

Phase 1 ends at an enriched detection trace: sampled-frame detections plus activation vectors from YOLOv8 backbone hooks. It intentionally does not include temporal linking, tracking, re-identification, or occlusion reasoning (those belong to a future Phase 2).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run

### 1. Run Phase 1 (single pass)

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

Each video writes:

- `experiments/results/enriched/<video_name>/enriched_detections.json`
- `experiments/results/enriched/<video_name>/pca_projection.pkl`
- `experiments/results/enriched/<video_name>/projection_manifest.json`

Batch mode runs videos sequentially, logs per-video progress, continues on failure, and exits nonzero if any video fails.

### 2. Discover hook layers (optional, for overrides)

```bash
python3 -m src.detection.enrichment.discover_layers --model yolov8n.yaml
```

Typical YOLOv8n/YOLOv8m hook choices:

- deep layer: `8` (stride `32`)
- mid layer: `6` (stride `16`)

### 3. Validate enriched outputs

Single file:

```bash
python3 -m src.detection.enrichment.validate \
  experiments/results/enriched/3sec_Left_to_Right/enriched_detections.json \
  --expected-dim 256
```

Batch validation:

```bash
for f in experiments/results/enriched/*/enriched_detections.json; do
  python3 -m src.detection.enrichment.validate "$f" --expected-dim 256 || break
done
```

## How It Works

Phase 1 uses a single YOLOv8 inference pass per sampled frame batch to produce both detection decisions and activation vectors. There is no rerun and no internal IoU-based artifact reconciliation step.

### Detection

The pipeline samples video frames at a fixed interval and runs YOLOv8 inference. For each sampled frame it records detections (`class_id`, `class_name`, `bbox`, `confidence`) in the enriched output payload.

In practice, this means the detection stage is responsible only for answering: "What did YOLO detect in this frame?" It is intentionally narrow in scope. It does not attempt identity tracking, temporal smoothing, or object permanence inference. Those concerns are deferred to downstream stages.

### Enrichment

During the same YOLO forward pass, the pipeline captures backbone feature maps with forward hooks and computes an activation embedding for each detection from the exact same detection bbox.

This stage exists because the final detection output is often too compressed for reasoning tasks. A class label and bounding box are sufficient for detection, but they do not preserve much of the model's internal representation of appearance. The enrichment stage recovers that information from the backbone and turns it into a compact vector that can be compared across frames.

It uses two backbone feature maps captured with forward hooks:

- a deep `C2f` layer (semantic features, stride 32)
- a mid `C2f` layer (higher spatial resolution, stride 16)

For each detection:

1. The image-space bounding box is mapped into feature-map coordinates using the layer stride.
2. A region is cropped from each hooked feature map.
3. `AdaptiveAvgPool2d((3, 3))` is applied to each crop.
4. The pooled tensors are flattened and concatenated into a raw activation vector.
5. PCA is fit across all detections collected for the video trace.
6. Each vector is projected to 256 dimensions and L2-normalized.
7. The projected vector is stored in the detection as `activation`.

The enriched trace is the Phase 1 output artifact used by later stages.

## Repository Structure

The codebase keeps detection utilities and enrichment logic in the `src/detection/` area, with a unified top-level Phase 1 entrypoint.

The `src/detection/` package contains detection utilities (frame sampling, YOLO wrappers, legacy detection-only CLIs) used by the unified Phase 1 pipeline.

The `src/detection/enrichment/` package contains the Phase 1 reasoning-layer implementation. It includes:

- layer discovery utilities for hook placement
- hook-based feature capture during YOLO inference
- ROI pooling and activation vector construction
- PCA fitting and projection
- output validation for enriched traces

Use `src/run_pipeline.py` as the primary Phase 1 interface. Compatibility wrappers remain in `src/detection/` for legacy module paths and currently emit deprecation notices.

## Outputs

Phase 1 writes enriched outputs per video:

- `experiments/results/enriched/<video_name>/enriched_detections.json`
- `experiments/results/enriched/<video_name>/pca_projection.pkl`
- `experiments/results/enriched/<video_name>/projection_manifest.json`

`enriched_detections.json` is the main downstream artifact. It contains per-frame detections and activation metadata for each detection.

`pca_projection.pkl` is the fitted dimensionality-reduction model for that video run. It captures how raw activation vectors were projected into the 256-dimensional embedding space.

`projection_manifest.json` records provenance for reproducibility, including selected hook layers, stride configuration, pooling strategy, projection dimension, fit timestamp, and input video hash.

## Enriched Detection Schema

Each detection in `enriched_detections.json` includes an `activation` payload:

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

The `activation.vector` field is the compact embedding used by downstream reasoning. It is produced from backbone feature crops, projected with PCA, and L2-normalized so it can be compared with similarity-based methods in later phases.

The remaining activation fields describe how that vector was produced. Together, they make the enriched JSON a self-describing intermediate artifact rather than just a raw tensor dump.
