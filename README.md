## Object Permanence

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

## Known Limitations

The main failure modes are detector-driven rather than linker-driven:
- **Spurious detections remain trackable.** If YOLO emits a persistent false positive, the tracker can keep it internally consistent.
- **Heavy occlusion is only partially recoverable.** Gallery relink recovers many splits, but cannot resolve cases where the occluder itself creates a competing detection.
- **Calibration is scenario-specific.** Current thresholds and layer weights were tuned on eight controlled videos, and the layer sweep was effectively class-level (`track_id_coverage = 0.0`) rather than instance-level.

---

## Methodology

Frame-to-frame linking uses a YOLO-only multi-layer composite embedding. Layer weights come from a separability sweep over candidate YOLO layers.

| Layer | Tier | Raw Dim | Sweep Separability | Weight |
|---|---|---:|---:|---:|
| `4.cv1` | Appearance | 64 | 15.495 | 0.549 |
| `15` | Semantic | 64 | 9.926 | 0.351 |
| `22.cv3.0` | Class-level | 80 | 13.902 | 0.100 |

**Why these YOLO tiers?**
- **Appearance (4.cv1):** Early backbone activations encode texture and color patterns, a strong signal for distinguishing objects that look different.
- **Semantic (15):** Mid-network neck activations encode spatial context and object structure, which stays stable across viewpoint changes and partial occlusion.
- **Class-level (22.cv3.0):** Detection-head activations encode class probability space. It is retained as a class-consistency gate but weighted conservatively because same-class instances are often near-identical in this space.

Layer weights were chosen from a Fisher-style separability sweep over YOLO layers. The current sweep used class fallback rather than stable instance IDs, so these weights should be interpreted as strong implementation guidance rather than a final instance-level optimum.

### DINO Gallery Relink

DINO is used only at relink time, where discriminability matters more than frame-to-frame stability:
- Extract DINO per detection during enrichment and store it as sidecar metadata.
- Retain valid DINO observations on each track and build a representative gallery per closed fragment at relink time.
- Score DINO relink candidates by the mean of the strongest gallery-to-gallery cosine matches, rather than a single fragment mean vector.
- Fall back to YOLO relink scoring when DINO is missing, and keep spatial fallback as a third pass.

The gallery reducer is designed to preserve temporal coverage rather than only the highest-confidence frames:
- Each closed fragment keeps a temporally ordered set of valid DINO samples gathered during tracking.
- At relink time, those samples are reduced to a fixed-size representative gallery (`relink_dino_gallery_size`, default `20`) by dividing the fragment lifetime into temporal buckets and keeping the highest-confidence sample from each bucket.
- Fragment similarity is computed from the pairwise cosine matrix between the two galleries and reduced with a top-k mean (`relink_dino_gallery_topk`, default `3`).
- This favors fragments that agree on several strong appearance matches and reduces the chance that a single noisy frame causes a false merge.

### Class Resolution Under Label Noise

YOLO class labels are treated as **noisy metadata**, not a hard identity gate:
- Frame-to-frame matching applies a configurable cross-class penalty (`class_mismatch_penalty`) instead of rejecting class mismatches outright.
- Relink applies a smaller cross-class penalty (`relink_class_mismatch_penalty`) for the same reason: a fragment labeled `vase` can still be a valid continuation of a `sports ball` track if appearance and motion evidence are strong.
- Each track resolves its displayed `class_id` / `class_name` from a confidence-weighted vote across all of its observations.
- Track observations keep the raw per-detection class label and confidence so mislabeled frames remain inspectable in `tracks.json` and the rendered trace summaries.

### Linking Pipeline

Frame-to-frame linking operates on cosine similarity between normalized projected embeddings.

**Matching**
- Similarity gate: `visual_similarity >= similarity_threshold` (recommended `0.70`).
- Spatial plausibility gate: centroid distance must be <= `max_centroid_distance` (default `0.40`, normalized by frame diagonal) before cosine scoring.
- Class mismatch is not a hard reject. When `match_within_class=true`, cross-class frame matches receive a soft assignment penalty (`class_mismatch_penalty`, default `0.20`).
- Assignment: Hungarian algorithm for globally consistent one-to-one matching per frame pair.

**Track state machine**
```text
TENTATIVE -> ACTIVE -> LOST -> CLOSED
```

Reference descriptors blend last, EMA, and history vectors for stability against appearance drift.
Track class is resolved from the confidence-weighted vote of its linked observations, not from the most recent detector label.

**Relink pass**
After the primary linking run, a second pass evaluates pairs of closed track fragments to recover identities split by occlusion:
- Enforces temporal ordering constraints; cross-class fragment pairs are allowed but penalized.
- Scores identity by method:
  - `dino`: top-k mean cosine over fragment DINO galleries when both fragments have enough valid DINO samples and `relink_use_dino=true` (gate: `relink_dino_threshold`).
  - `yolo`: cosine on YOLO fragment centroids when DINO is unavailable or disabled (gate: `relink_threshold`).
- Applies a relink class mismatch penalty (`relink_class_mismatch_penalty`, default `0.10`) to identity and spatial fallback scores when fragment-level resolved classes disagree.
- Falls back to spatial plausibility (`spatial`) as a third pass for unresolved pairs (gate: `relink_fallback_threshold`).
- Merges accepted chains into canonical track IDs.
- Records DINO contribution metrics in `relink_manifest.json`: `relink_dino_coverage`, `relink_dino_accepted`, `relink_yolo_accepted`.

---

## Experiment Results

`activation_topk=64` is the default operating point. On `Right_to_left`, `k=12` and `k=64` produced the same ball track count and total track count, so the larger value was retained for stability.

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

### Qualitative Trace Summary Examples

The rendered trace summary sheets below come directly from `experiments/results/linking/*/trace_summary/` and illustrate both successful recoveries and detector-driven failure modes.

**Successful recovery: long left-to-right ball track**

`10sec_Left_to_Right` track `1` shows the sports ball surviving fragmentation and being merged back into a single canonical trajectory.

![10sec_Left_to_Right recovery](experiments/results/linking/10sec_Left_to_Right/trace_summary/track_1.jpg)

**Successful recovery: bounce-back relink**

`Left_bounce_back` track `1` is a representative case where the ball leaves the immediate neighborhood of its prior detections, fragments, and is later recovered into the same final track.

![Left_bounce_back recovery](experiments/results/linking/Left_bounce_back/trace_summary/track_1.jpg)

**Limitation: spurious non-ball recovery**

`Occlusion_ball_removed` track `5` is a good example of a detector-driven false positive. The temporal linker can keep this track internally consistent, but it cannot correct the fact that YOLO emitted a persistent non-ball object.

![Occlusion_ball_removed limitation](experiments/results/linking/Occlusion_ball_removed/trace_summary/track_5.jpg)

**Limitation: stable false-positive track**

`No_occlusion_ball_removed` track `8` shows that even without heavy occlusion, consistent false detections can still form coherent tracks and survive relink.

![No_occlusion_ball_removed limitation](experiments/results/linking/No_occlusion_ball_removed/trace_summary/track_8.jpg)

---

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

**Minimal reproduction**
```bash
bash scripts/run_full_pipeline.sh
```

This README is intentionally poster-facing. For full operational detail, use `src/run_pipeline.py`, `src/run_temporal_linking.py`, and `scripts/run_full_pipeline.sh`.
