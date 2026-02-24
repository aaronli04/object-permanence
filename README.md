# Object Permanence

YOLOv8-based video detection and Phase 1 activation enrichment for object permanence reasoning.

The repository is structured around a simple principle: keep raw detector output stable, then layer richer reasoning features on top of it as a separate post-processing step. This keeps the system easy to debug, easy to rerun, and suitable for later phases that need reproducible intermediate artifacts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run

### 1. Generate detections

Single video:

```bash
python3 src/detection/run_yolo.py \
  --video data/raw_videos/3sec_Left_to_Right.mp4 \
  --sample-rate 5 \
  --model yolov8n.pt
```

Batch videos:

```bash
python3 src/detection/run_yolo_batch.py \
  --video-dir data/raw_videos \
  --pattern "*.mp4" \
  --sample-rate 5 \
  --model yolov8n.pt
```

### 2. Discover hook layers

```bash
python3 -m src.detection.enrichment.discover_layers --model yolov8n.yaml
```

Typical YOLOv8n/YOLOv8m hook choices:

- deep layer: `8` (stride `32`)
- mid layer: `6` (stride `16`)

### 3. Enrich detections with activation vectors

Single video:

```bash
python3 -m src.detection.enrichment.run \
  --input-json experiments/results/detections/3sec_Left_to_Right_detections.json \
  --video data/raw_videos/3sec_Left_to_Right.mp4 \
  --model yolov8n.pt \
  --output-dir experiments/results/enriched/3sec_Left_to_Right \
  --deep-layer 8 \
  --mid-layer 6 \
  --deep-stride 32 \
  --mid-stride 16 \
  --batch-size 8 \
  --pca-dim 256
```

Batch all generated detection traces:

```bash
for json in experiments/results/detections/*_detections.json; do
  base="$(basename "$json" _detections.json)"
  video="data/raw_videos/${base}.mp4"
  outdir="experiments/results/enriched/${base}"

  [ -f "$video" ] || continue

  python3 -m src.detection.enrichment.run \
    --input-json "$json" \
    --video "$video" \
    --model yolov8n.pt \
    --output-dir "$outdir" \
    --deep-layer 8 \
    --mid-layer 6 \
    --deep-stride 32 \
    --mid-stride 16 \
    --batch-size 8 \
    --pca-dim 256 || break
done
```

### 4. Validate enriched outputs

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

The system is organized into two stages:

### Detection

The detection pipeline samples video frames at a fixed interval and runs YOLOv8 inference. The result is a JSON trace containing frame numbers and detections (`class_id`, `class_name`, `bbox`, `confidence`).

This trace is the stable intermediate artifact used by later reasoning stages.

In practice, this means the detection stage is responsible only for answering: "What did YOLO detect in this frame?" It is intentionally narrow in scope. It does not attempt identity tracking, temporal smoothing, or object permanence inference. Those concerns are deferred to downstream stages.

### Enrichment

The enrichment pipeline re-runs YOLOv8 on the original frames and attaches an activation embedding to each existing detection.

This stage exists because the final detection output is often too compressed for reasoning tasks. A class label and bounding box are sufficient for detection, but they do not preserve much of the model's internal representation of appearance. The enrichment stage recovers that information from the backbone and turns it into a compact vector that can be compared across frames.

It uses two backbone feature maps captured with forward hooks:

- a deep `C2f` layer (semantic features, stride 32)
- a mid `C2f` layer (higher spatial resolution, stride 16)

For each detection:

1. The image-space bounding box is mapped into feature-map coordinates using the layer stride.
2. A region is cropped from each hooked feature map.
3. `AdaptiveAvgPool2d((3, 3))` is applied to each crop.
4. The pooled tensors are flattened and concatenated into a raw activation vector.
5. PCA is fit across all vectors in the input trace.
6. Each vector is projected to 256 dimensions and L2-normalized.
7. The projected vector is stored in the detection as `activation`.

The enrichment step preserves the original detections and only augments them.

This separation is a design choice, not just an implementation detail. It allows the project to maintain a versioned detection schema while evolving the reasoning layer independently. If enrichment logic changes (different hook layers, pooling, projection, or matching policy), the original detection trace remains untouched.

### Linking

Because YOLO is re-run during enrichment, detections are linked back to the original JSON entries by:

- `class_id` match
- bounding-box IoU threshold (`>= 0.5`)

This keeps the original detection trace as the source of truth while attaching features from the rerun inference.

The linking step is the bridge between the two stages. It ensures the activation vector is written onto the correct detection record even when rerun YOLO output order differs from the original JSON. Using class and IoU matching provides a deterministic and inspectable rule for this association.

## Repository Structure

The codebase is split between baseline detection and post-processing enrichment.

The `src/detection/` package contains the detection pipeline used to sample video frames, run YOLOv8, and write JSON traces. These modules define the minimal artifact that later stages consume.

The `src/detection/enrichment/` package contains the Phase 1 reasoning-layer implementation. It includes:

- layer discovery utilities for hook placement
- hook-based feature capture during YOLO reruns
- ROI pooling and activation vector construction
- PCA fitting and projection
- output validation for enriched traces

Compatibility wrappers remain in `src/detection/` for legacy module paths, but the `src/detection/enrichment/` package is the primary interface for the activation pipeline.

## Outputs

The output layout mirrors the two-stage architecture.

Detection output:

- `experiments/results/detections/<video_name>_detections.json`

Enrichment output (per video):

- `experiments/results/enriched/<video_name>/enriched_detections.json`
- `experiments/results/enriched/<video_name>/pca_projection.pkl`
- `experiments/results/enriched/<video_name>/projection_manifest.json`

`enriched_detections.json` is the main downstream artifact. It preserves the original per-frame detection records and adds activation metadata for each detection.

`pca_projection.pkl` is the fitted dimensionality-reduction model for that enrichment run. It captures how raw activation vectors were projected into the 256-dimensional embedding space.

`projection_manifest.json` records provenance for reproducibility, including selected hook layers, stride configuration, pooling strategy, projection dimension, fit timestamp, and input file hash.

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
