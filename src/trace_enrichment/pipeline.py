"""Single-pass enriched trace pipeline."""

from __future__ import annotations

import datetime as dt
import os
from typing import Any, Iterable, Optional

from .constants import (
    ACTIVATION_LAYER_ALIASES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEEP_LAYER,
    DEFAULT_DEEP_STRIDE,
    DEFAULT_MID_LAYER,
    DEFAULT_MID_STRIDE,
    OUTPUT_VECTOR_DIM,
    POOL_SIZE,
    POOL_STRATEGY,
    SCHEMA_VERSION,
)
from .io import build_output_artifacts, sha256_file, write_json
from .model import (
    FeatureHookCollector,
    build_raw_activation_vector,
    ensure_model_runtime_dependencies,
    extract_detections_from_result,
    get_module_map,
    load_yolo,
)
from .sampler import FrameSampler
from .types import CollectedDetection, CollectedFrame, CollectionStats, HookConfig, RunConfig

try:
    import joblib
    import numpy as np
    from sklearn.decomposition import PCA

    _PIPELINE_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    joblib = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    PCA = None  # type: ignore[assignment]
    _PIPELINE_IMPORT_ERROR = exc


def ensure_pipeline_runtime_dependencies() -> None:
    ensure_model_runtime_dependencies()
    if _PIPELINE_IMPORT_ERROR is not None:
        missing = getattr(_PIPELINE_IMPORT_ERROR, "name", "unknown")
        raise RuntimeError(
            "Trace enrichment projection dependencies failed to import "
            f"(module={missing}). Install/repair requirements.txt environment. Original error: {_PIPELINE_IMPORT_ERROR}"
        ) from _PIPELINE_IMPORT_ERROR


def _iter_detections(frames: list[CollectedFrame]) -> Iterable[CollectedDetection]:
    for frame in frames:
        for det in frame.detections:
            yield det


def _layer_manifest_section(cfg: HookConfig) -> dict[str, Any]:
    return {
        "aliases": list(ACTIVATION_LAYER_ALIASES),
        "actual": {"deep": cfg.deep_layer, "mid": cfg.mid_layer},
        "strides": {"deep": cfg.deep_stride, "mid": cfg.mid_stride},
    }


def _l2_normalize(vec: "np.ndarray") -> "np.ndarray":
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        return np.zeros_like(vec)
    return vec / norm


def _pad_and_normalize_projection(vec: "np.ndarray", source_dim: int, target_dim: int) -> "np.ndarray":
    if source_dim < target_dim:
        padded = np.zeros((target_dim,), dtype=np.float32)
        padded[:source_dim] = vec
        vec = padded
    return _l2_normalize(vec)


def collect_single_pass_trace(
    *,
    yolo: Any,
    video_path: str,
    sample_rate: int,
    hook_config: HookConfig,
    batch_size: int,
) -> tuple[list[CollectedFrame], CollectionStats]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    module_map = get_module_map(yolo)
    # Pool on the active device, then move pooled vectors to CPU in model helpers.
    import torch  # local import keeps top-level import cost low in non-runtime paths

    pool = torch.nn.AdaptiveAvgPool2d(POOL_SIZE)
    frames: list[CollectedFrame] = []
    stats = CollectionStats()

    def process_batch(batch_items: list[tuple[int, Any]], hooks: FeatureHookCollector) -> None:
        if not batch_items:
            return

        hooks.clear()
        batch_images = [frame for _, frame in batch_items]
        results = yolo(batch_images, verbose=False)
        if len(results) != len(batch_items):
            raise RuntimeError("YOLO result count does not match inference batch size")
        if hook_config.deep_layer not in hooks.outputs or hook_config.mid_layer not in hooks.outputs:
            raise RuntimeError("Expected hook outputs were not captured during YOLO forward pass")

        deep_batch = hooks.outputs[hook_config.deep_layer].detach().cpu()
        mid_batch = hooks.outputs[hook_config.mid_layer].detach().cpu()
        if deep_batch.shape[0] != len(batch_items) or mid_batch.shape[0] != len(batch_items):
            raise RuntimeError("Hook output batch size does not match inference batch size")

        for index, (frame_num, _frame) in enumerate(batch_items):
            detections = extract_detections_from_result(results[index], sort_by_confidence=True)
            for det in detections:
                raw_vec, small_crop_flag = build_raw_activation_vector(
                    fmap_mid=mid_batch[index],
                    fmap_deep=deep_batch[index],
                    bbox_xyxy=det.bbox,
                    pool=pool,
                    stride_mid=hook_config.mid_stride,
                    stride_deep=hook_config.deep_stride,
                )
                det.raw_vector = raw_vec
                det.small_crop_flag = bool(small_crop_flag)

            frames.append(CollectedFrame(frame_num=int(frame_num), detections=detections))
            stats.total_sampled_frames += 1
            if detections:
                stats.frames_with_detections += 1
            stats.total_detections += len(detections)

    pending: list[tuple[int, Any]] = []
    with FeatureHookCollector(module_map=module_map, layer_names=[hook_config.deep_layer, hook_config.mid_layer]) as hooks:
        for frame_num, frame in FrameSampler(video_path, sample_rate):
            pending.append((int(frame_num), frame))
            if len(pending) >= batch_size:
                process_batch(pending, hooks)
                pending = []
        if pending:
            process_batch(pending, hooks)

    return frames, stats


def fit_pca_and_project(frames: list[CollectedFrame], pca_dim: int):
    detections = list(_iter_detections(frames))
    if not detections:
        raise RuntimeError("No detections collected for PCA")

    raw_vectors = []
    for det in detections:
        if det.raw_vector is None:
            raise RuntimeError("Missing raw activation vector on collected detection")
        raw_vectors.append(det.raw_vector)

    raw_matrix = np.stack(raw_vectors, axis=0)
    effective_pca_dim = min(pca_dim, raw_matrix.shape[0], raw_matrix.shape[1])
    if effective_pca_dim <= 0:
        raise RuntimeError("No activation vectors collected for PCA")

    pca = PCA(n_components=effective_pca_dim, svd_solver="auto", random_state=0)
    projected = pca.fit_transform(raw_matrix).astype(np.float32, copy=False)
    for det, proj in zip(detections, projected):
        det.projected_vector = _pad_and_normalize_projection(proj, effective_pca_dim, pca_dim)

    return pca, int(effective_pca_dim)


def build_enriched_payload(frames: list[CollectedFrame], pca_dim: int) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for frame in frames:
        detections_payload: list[dict[str, Any]] = []
        for det in frame.detections:
            if det.projected_vector is None:
                raise RuntimeError(f"Missing projected activation for frame={frame.frame_num}")
            detections_payload.append(
                {
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "bbox": [float(v) for v in det.bbox],
                    "confidence": float(det.confidence),
                    "activation": {
                        "vector": [float(v) for v in det.projected_vector.tolist()],
                        "dim": int(pca_dim),
                        "layers": list(ACTIVATION_LAYER_ALIASES),
                        "pool": POOL_STRATEGY,
                        "projection": f"pca_{pca_dim}",
                        "small_crop_flag": bool(det.small_crop_flag),
                    },
                }
            )
        payload.append({"frame_num": frame.frame_num, "detections": detections_payload})
    return payload


def build_manifest(
    *,
    run_config: RunConfig,
    hook_config: HookConfig,
    effective_pca_dim: int,
    frames: list[CollectedFrame],
    stats: CollectionStats,
) -> dict[str, Any]:
    first_detection = next(_iter_detections(frames), None)
    if first_detection is None or first_detection.raw_vector is None:
        raise RuntimeError("Cannot build manifest without at least one activation vector")

    artifacts = build_output_artifacts(run_config.output_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "projection_file": os.path.basename(artifacts.pca_path),
        "projection_dim": int(run_config.pca_dim),
        "fitted_pca_components": int(effective_pca_dim),
        "fit_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_video_hash_sha256": sha256_file(run_config.video_path),
        "input_video_file": os.path.basename(run_config.video_path),
        "pool": POOL_STRATEGY,
        "pool_size": list(POOL_SIZE),
        "layers": _layer_manifest_section(hook_config),
        "raw_activation_dim": int(len(first_detection.raw_vector)),
        "num_vectors_fit": int(stats.total_detections),
        "model_name": run_config.model_name,
        "batch_size": int(run_config.batch_size),
        "sample_rate": int(run_config.sample_rate),
        "total_sampled_frames": int(stats.total_sampled_frames),
        "frames_with_detections": int(stats.frames_with_detections),
        "total_detections": int(stats.total_detections),
        "artifacts": {
            "enriched_detections_json": os.path.basename(artifacts.enriched_json_path),
            "projection_manifest_json": os.path.basename(artifacts.manifest_path),
        },
    }


def run_trace_enrichment(
    *,
    video_path: str,
    model_name: str,
    output_dir: str,
    sample_rate: int,
    deep_layer_name: str = DEFAULT_DEEP_LAYER,
    mid_layer_name: str = DEFAULT_MID_LAYER,
    stride_deep: int = DEFAULT_DEEP_STRIDE,
    stride_mid: int = DEFAULT_MID_STRIDE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pca_dim: int = OUTPUT_VECTOR_DIM,
) -> dict[str, str]:
    ensure_pipeline_runtime_dependencies()

    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if pca_dim <= 0:
        raise ValueError("pca_dim must be > 0")

    run_config = RunConfig(
        video_path=video_path,
        model_name=model_name,
        sample_rate=sample_rate,
        batch_size=batch_size,
        pca_dim=pca_dim,
        output_dir=output_dir,
    )
    hook_config = HookConfig(
        deep_layer=deep_layer_name,
        mid_layer=mid_layer_name,
        deep_stride=stride_deep,
        mid_stride=stride_mid,
    )
    artifacts = build_output_artifacts(output_dir)

    yolo = load_yolo(model_name)
    frames, stats = collect_single_pass_trace(
        yolo=yolo,
        video_path=video_path,
        sample_rate=sample_rate,
        hook_config=hook_config,
        batch_size=batch_size,
    )

    if stats.total_detections <= 0:
        raise RuntimeError(
            f"No detections found in sampled frames for '{video_path}'. Trace enrichment requires detections."
        )

    pca, effective_pca_dim = fit_pca_and_project(frames, pca_dim)
    enriched_payload = build_enriched_payload(frames, pca_dim)
    manifest = build_manifest(
        run_config=run_config,
        hook_config=hook_config,
        effective_pca_dim=effective_pca_dim,
        frames=frames,
        stats=stats,
    )

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pca, artifacts.pca_path)
    write_json(artifacts.enriched_json_path, enriched_payload)
    write_json(artifacts.manifest_path, manifest)

    return {
        "enriched_detections": artifacts.enriched_json_path,
        "pca_projection": artifacts.pca_path,
        "projection_manifest": artifacts.manifest_path,
    }
