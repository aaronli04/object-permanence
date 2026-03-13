# Object Permanence

**Identity-preserving object tracking on top of YOLOv8 — fully offline.**

Most detectors treat each frame in isolation. Object Permanence adds a temporal identity layer: every detection gets a stable ID that persists across frames, survives occlusion, and relinks after a track is lost — no cloud, no fine-tuning required.

---

## How it works

The pipeline runs in two offline stages:

**Stage 1 — Enrichment** &nbsp;`src/run_pipeline.py`
Runs YOLOv8 on sampled frames, extracts multi-layer feature embeddings per detection, and compresses them to a 128-D identity vector via PCA.

**Stage 2 — Linking** &nbsp;`src/run_temporal_linking.py`
Links detections across frames using cosine similarity + the Hungarian algorithm for one-to-one assignment, then runs a relink pass to recover tracks fragmented by occlusion.

### Identity embedding

Frame-to-frame matching uses a weighted composite of three YOLO layers, each chosen for a different role:

| Layer | Role | Why |
|---|---|---|
| `4.cv1` | Appearance | Early backbone — texture, color |
| `15` | Structure | Mid-neck — spatial context, pose stability |
| `22.cv3.0` | Class gate | Detection head — consistency check |

[DINO](https://github.com/facebookresearch/dino) `ViT-S/8` CLS vectors are extracted as a **relink-only sidecar** — not used for frame-to-frame matching (where stability matters) but brought in during the relink pass (where discriminability matters most).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Single video:**
```bash
python3 src/run_pipeline.py \
  --video data/raw_videos/my_clip.mp4 \
  --model yolov8n.pt \
  --sample-rate 1

python3 src/run_temporal_linking.py \
  --enriched-json experiments/results/activation_enrichment/my_clip/enriched_detections.json \
  --render-trace-proofs
```

**All videos at once:**
```bash
bash scripts/run_full_pipeline.sh
```

> **First run with DINO?** Warm the cache once before going offline:
> ```bash
> export TORCH_HOME="$PWD/.torch_cache"
> python3 -c "import torch; torch.hub.load('facebookresearch/dino:main', 'dino_vits8')"
> ```

---

## Results

Evaluated across 8 scenarios (occlusion, removal, direction changes, bounce-back). Default config: `similarity_threshold=0.70`, `relink_dino_threshold=0.55`, `relink_fallback_threshold=0.40`.

| Scenario | Frames | Detections | Ball Tracks | Valid Tracks | Relinks |
|---|---:|---:|---:|---:|---:|
| 10sec Left→Right | 133 | 160 | 1 | 5 | 1 |
| 3sec Left→Right | 49 | 77 | 1 | 5 | 1 |
| Exit while occluded | 53 | 72 | 1 | 4 | 0 |
| Left bounce back | 64 | 105 | 1 | 4 | 2 |
| Left→Right | 25 | 52 | 1 | 5 | 1 |
| Ball removed (clear) | 34 | 37 | 1 | 4 | 1 |
| Ball removed (occluded) | 48 | 114 | 1 | 8 | 2 |
| Right→Left | 21 | 40 | 1 | 3 | 1 |
| **Total** | **427** | **657** | **8** | **38** | **9** |

DINO relinking (`R1`–`R4`) consistently adds one relink edge vs. the YOLO-only baseline (`R0`) without increasing total track count.

---

## Output

```
experiments/results/
  activation_enrichment/<scenario>/
    enriched_detections.json
    pca_projection.pkl
    projection_manifest.json          ← DINO sidecar role + runtime state
  linking/<scenario>/
    linked_detections.json
    tracks.json
    relink_manifest.json              ← DINO/YOLO relink coverage metrics
    trace_proofs/                     ← visual proof images per relink
```

---

## Configuration

All defaults live in `src/trace_enrichment/constants.py`. Key knobs:

| Parameter | Default | What it controls |
|---|---:|---|
| `similarity_threshold` | `0.70` | Frame-to-frame match gate |
| `max_centroid_distance` | `0.40` | Spatial plausibility gate (normalized) |
| `relink_dino_threshold` | `0.55` | DINO relink acceptance |
| `relink_threshold` | `0.55` | YOLO relink fallback |
| `relink_fallback_threshold` | `0.40` | Spatial-only last resort |

Disable DINO entirely: `TRACE_DISABLE_DINO=1`
Force single-layer mode: `TRACE_DISABLE_MULTI_LAYER_EMBEDDING=1`

---

## Known limitations

- **Spurious YOLO detections** (e.g. logos misclassified as objects) enter the pipeline as valid candidates and can fragment nearby real tracks. The relink pass has no way to distinguish them.
- **Layer calibration is class-level, not instance-level.** The separability sweep ran without stable `track_id` labels, so weights reflect class separation — the easier problem. Scenes with multiple instances of the same class (several people, several balls) may need re-calibration with `--require-track-id`.
- **Thresholds were tuned on 8 controlled scenarios.** Faster motion, denser scenes, or different class distributions may need a re-sweep.
- **PCA basis is fit per run, not per object.** Rare objects or late-appearing objects may be projected into a subspace dominated by more common categories.

---

## Advanced usage

<details>
<summary>Re-run the layer calibration sweep</summary>

```bash
# Per-video sweep
for v in data/raw_videos/*.mp4; do
  stem="$(basename "$v" .mp4)"
  python3 experiments/layer_stability_sweep.py \
    --video "$v" --model yolov8n.pt --sample-rate 1 \
    --dino --require-track-id \
    --output-csv "experiments/results/layer_selection/per_video/layer_stability_sweep_${stem}.csv"
done

# Aggregate
python3 experiments/aggregate_layer_sweeps.py \
  --input-glob "experiments/results/layer_selection/per_video/*.csv" \
  --output-csv experiments/results/layer_selection/aggregate/aggregate_separability.csv
```
</details>

<details>
<summary>DINO relink threshold sweep (R0–R6)</summary>

```bash
python3 experiments/run_dino_param_search.py \
  --enrichment-root experiments/results/activation_enrichment \
  --output-root experiments/results/param_search
```

Outputs `summary.csv` plus per-run artifacts under `param_search/R0` … `param_search/R6`.
</details>