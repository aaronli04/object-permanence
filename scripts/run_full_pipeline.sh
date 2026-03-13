#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-yolov8n.pt}"
VIDEO_GLOB="${VIDEO_GLOB:-data/raw_videos/*.mp4}"
SAMPLE_RATE="${SAMPLE_RATE:-5}"
ACTIVATION_TOPK="${ACTIVATION_TOPK:-64}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.70}"
MAX_CENTROID_DISTANCE="${MAX_CENTROID_DISTANCE:-0.40}"
RELINK_THRESHOLD="${RELINK_THRESHOLD:-0.55}"
RELINK_DINO_THRESHOLD="${RELINK_DINO_THRESHOLD:-0.55}"
RELINK_MAX_GAP_FRAMES="${RELINK_MAX_GAP_FRAMES:--1}"
RELINK_FALLBACK_THRESHOLD="${RELINK_FALLBACK_THRESHOLD:-0.40}"

shopt -s nullglob
videos=(${VIDEO_GLOB})
shopt -u nullglob

if [ "${#videos[@]}" -eq 0 ]; then
  echo "No videos matched: ${VIDEO_GLOB}" >&2
  exit 1
fi

for video in "${videos[@]}"; do
  scenario="$(basename "${video}" .mp4)"

  echo "[enrichment] ${video}"
  python3 src/run_pipeline.py \
    --video "${video}" \
    --model "${MODEL}" \
    --sample-rate "${SAMPLE_RATE}"

  echo "[linking] ${scenario}"
  python3 src/run_temporal_linking.py \
    --enriched-json "experiments/results/activation_enrichment/${scenario}/enriched_detections.json" \
    --activation-topk "${ACTIVATION_TOPK}" \
    --similarity-threshold "${SIMILARITY_THRESHOLD}" \
    --max-centroid-distance "${MAX_CENTROID_DISTANCE}" \
    --relink-threshold "${RELINK_THRESHOLD}" \
    --relink-dino-threshold "${RELINK_DINO_THRESHOLD}" \
    --relink-max-gap-frames "${RELINK_MAX_GAP_FRAMES}" \
    --relink-fallback-threshold "${RELINK_FALLBACK_THRESHOLD}" \
    --render-trace-proofs
done
