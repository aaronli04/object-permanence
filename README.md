# Object Permanence

This repo includes:

- a YOLOv8 video detection pipeline that samples frames, runs inference, and saves JSON results
- a Phase 1 post-processing activation enrichment pipeline that re-runs YOLOv8 with forward hooks and attaches backbone activation vectors to each detection

## Quickstart

1. Create and activate a virtual environment (recommended default):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

2. Run a single-video detection from the repo root:

```bash
python3 src/detection/run_yolo.py --video data/raw_videos/3sec_Left_to_Right.mp4 --sample-rate 5 --model yolov8n.pt
```

3. Or run batch detection over a directory:

```bash
python3 src/detection/run_yolo_batch.py --video-dir data/raw_videos --pattern "*.mp4" --sample-rate 5 --model yolov8n.pt
```

4. Discover hook layers for your YOLO variant (used by enrichment):

```bash
python3 -m src.detection.enrichment.discover_layers --model yolov8n.yaml
```

5. Run enrichment for one video first (recommended sanity check):

```bash
python3 -m src.detection.enrichment.run \
  --input-json "experiments/results/detections/3sec_Left_to_Right_detections.json" \
  --video "data/raw_videos/3sec_Left_to_Right.mp4" \
  --model yolov8n.pt \
  --output-dir "experiments/results/enriched/3sec_Left_to_Right" \
  --deep-layer 8 \
  --mid-layer 6 \
  --deep-stride 32 \
  --mid-stride 16 \
  --batch-size 8 \
  --pca-dim 256
```

6. Batch-enrich all generated detection JSONs:

```bash
for json in experiments/results/detections/*_detections.json; do
  base="$(basename "$json" _detections.json)"
  video="data/raw_videos/${base}.mp4"
  outdir="experiments/results/enriched/${base}"

  if [ ! -f "$video" ]; then
    echo "Skipping (missing video): $video"
    continue
  fi

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

7. Validate outputs:

```bash
for f in experiments/results/enriched/*/enriched_detections.json; do
  python3 -m src.detection.enrichment.validate "$f" --expected-dim 256 || break
done
```

Notes:

- If your environment is offline, use a locally available weights file (for example `yolov8n.pt` or `/path/to/yolov8n.pt`)
- Avoid shell placeholders like `<video_name>` when pasting commands directly into bash, because `<...>` is interpreted as input redirection

## Output

Detections are saved to:

```
experiments/results/detections/[video_name]_detections.json
```

Each entry includes the frame number and an array of detections with class id, class name, bounding box coordinates (xyxy), and confidence.

## Phase 1 Activation Enrichment

This pipeline takes:

- an existing detections JSON trace (sampled every 5 frames)
- the source video
- a YOLOv8 model (re-run for feature extraction)

It writes:

- `enriched_detections.json`
- `pca_projection.pkl`
- `projection_manifest.json`

Batching support:

- yes, the YOLO rerun path processes frames in batches (default `--batch-size 8`)
- adjust with `--batch-size` depending on GPU/CPU memory

## How It Works (Two Components)

The Phase 1 reasoning layer has two conceptual components:

### 1. Enriching (feature extraction + embedding)

This component adds an `activation` payload to each existing detection by:

- re-running YOLOv8 on the original frame images
- capturing two backbone feature maps using forward hooks (mid + deep `C2f`)
- cropping the detection ROI in feature-map space (using stride-based coordinate scaling)
- applying `AdaptiveAvgPool2d((3, 3))`
- flattening + concatenating pooled features
- fitting PCA across all detections in the input trace
- projecting to a 256-d vector and L2-normalizing it

Output of this component:

- `enriched_detections.json` (per-detection activation vectors)
- `pca_projection.pkl` (fitted PCA object)
- `projection_manifest.json` (provenance metadata)

### 2. Linking (input detections <-> rerun YOLO detections)

This component ensures activations are attached to the correct original JSON entries.

It does not add or remove detections. Instead, for each frame it:

- reruns YOLOv8 and reads the rerun detections
- matches each input JSON detection to a rerun YOLO detection by:
- `class_id` equality
- bbox IoU threshold (`>= 0.5`)
- keeps original input detections as the source-of-truth records
- uses the matched (or fallback original bbox if unmatched) entry to compute the activation crop

Why this matters:

- preserves the original sampled trace as a stable contract
- enables deterministic downstream indexing (detections are sorted by confidence before writing)
- provides a clear boundary between raw detection generation and post-processing reasoning

### Layer Discovery (C2f modules)

Use the discovery utility to inspect candidate backbone `C2f` layers before hooking:

```bash
python3 -m src.detection.enrichment.discover_layers --model yolov8n.yaml
```

`discover_layers` is lightweight and does not require `scikit-learn` to be importable.

For `yolov8n` and `yolov8m`, the typical backbone choices are:

- deep (stride 32): module `8`
- mid (stride 16): module `6`

### Run Enrichment

Behavior:

- re-runs YOLO and captures two backbone feature maps with forward hooks
- matches rerun detections back to input JSON by `class_id` + bbox IoU (`>= 0.5`)
- crops feature-map ROIs by stride (`16` and `32`)
- applies `AdaptiveAvgPool2d((3, 3))`, concatenates vectors, fits PCA in a first pass
- projects to `256` dimensions, L2-normalizes, and writes enriched JSON in a second pass
- sorts detections by confidence descending within each frame

### Validate Enriched Output

The validator asserts:

- every detection has an `activation` field
- `activation.dim == 256`
- no NaN/Inf values in activation vectors
- prints a summary including `small_crop_flag` counts

## Recommended Process (End-to-End)

1. Run detection batch over `data/raw_videos/`
2. Run `discover_layers` for your YOLO variant (`yolov8n.yaml`, `yolov8m.yaml`, etc.)
3. Run enrichment (single video first to verify settings)
4. Run the enrichment shell loop for all detection JSONs
5. Validate one output, then run the validation loop for all outputs

## Structure

- `src/detection/types.py`: Shared dataclasses for detections
- `src/detection/io.py`: Output path utilities and JSON writer
- `src/detection/sampler.py`: Frame sampling from video
- `src/detection/yolo_runner.py`: YOLOv8 inference wrapper
- `src/detection/pipeline.py`: Pipeline orchestration
- `src/detection/cli.py`: Shared CLI argument parsing and dispatch
- `src/detection/run_yolo.py`: Single-video CLI entry point
- `src/detection/run_yolo_batch.py`: Batch CLI entry point
- `src/detection/enrichment/pipeline.py`: Phase 1 activation enrichment pipeline (hooks, ROI pooling, PCA, manifests)
- `src/detection/enrichment/discover_layers.py`: YOLOv8 `C2f` layer discovery utility
- `src/detection/enrichment/run.py`: Activation enrichment CLI entry point
- `src/detection/enrichment/validate.py`: Enriched JSON validation script
- `src/detection/enrichment/introspection.py`: Shared YOLO model introspection helpers
- `src/detection/activation_enrichment.py`: Compatibility wrapper (legacy import path)
- `src/detection/discover_yolo_layers.py`: Compatibility wrapper (legacy CLI module path)
- `src/detection/run_activation_enrichment.py`: Compatibility wrapper (legacy CLI module path)
- `src/detection/validate_enriched_detections.py`: Compatibility wrapper (legacy CLI module path)

## Config

- `--video`: input video path (single mode)
- `--video-dir`: input directory (batch mode)
- `--pattern`: glob for batch videos (default `*.mp4`)
- `--sample-rate`: sample every N frames (default 5)
- `--model`: YOLOv8 model name or path (default `yolov8n.pt`)

## Enriched Detection Schema (Phase 1)

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
