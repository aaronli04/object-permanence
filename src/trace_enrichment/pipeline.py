"""Single-pass enriched trace pipeline."""

from __future__ import annotations

import datetime as dt
import os
import sys
from typing import Any, Iterable, Optional

try:
    from common.numeric import l2_normalize
except ImportError:  # pragma: no cover - import-path compatibility
    from src.common.numeric import l2_normalize  # type: ignore

from .constants import (
    ACTIVATION_LAYER_ALIASES,
    DINO_CROP_PADDING_RATIO,
    DINO_EMBEDDING_DIM,
    DINO_INPUT_SIZE,
    DINO_LOAD_TIMEOUT_SECONDS,
    DINO_MODEL_NAME,
    DINO_MODEL_REPO,
    DINO_TINY_CROP_MIN_SIZE,
    DISABLE_DINO_ENV,
    DISABLE_MULTI_LAYER_EMBEDDING_ENV,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HEAD_LAYER,
    DEFAULT_HEAD_STRIDE,
    EMBEDDING_LAYERS,
    OUTPUT_VECTOR_DIM,
    POOL_SIZE,
    POOL_STRATEGY,
    SCHEMA_VERSION,
)
from .dino import DinoEmbedder, DinoUnavailableError, extract_dino_embedding, load_dino_embedder
from .io import build_output_artifacts, sha256_file, write_json
from .model import (
    FeatureHookCollector,
    build_raw_activation_vector,
    ensure_model_runtime_dependencies,
    extract_detections_from_result,
    get_module_map,
    load_yolo,
    resolve_hook_layer_name,
)
from .sampler import FrameSampler
from .types import CollectedDetection, CollectedFrame, CollectionStats, HookConfig, RunConfig

import numpy as np

try:
    import joblib
    from sklearn.decomposition import PCA

    _PIPELINE_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    joblib = None  # type: ignore[assignment]
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


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _multi_layer_embedding_disabled() -> bool:
    return _is_truthy_env(os.environ.get(DISABLE_MULTI_LAYER_EMBEDDING_ENV))


def _dino_disabled() -> bool:
    return _is_truthy_env(os.environ.get(DISABLE_DINO_ENV))


def _infer_model_device(yolo: Any) -> str | None:
    try:
        params = getattr(getattr(yolo, "model", None), "parameters", None)
        if callable(params):
            first = next(params(), None)
            if first is not None and hasattr(first, "device"):
                return str(first.device)
    except Exception:
        return None
    return None


def _normalize_weights(items: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not items:
        return []
    total = float(sum(weight for _, weight in items))
    if total <= 0.0:
        raise ValueError("Embedding layer weights must sum to > 0")
    return [(name, float(weight / total)) for name, weight in items]


def _resolve_multi_layer_hook_config(yolo: Any, stride: int) -> HookConfig:
    resolved_weight_by_layer: dict[str, float] = {}
    resolved_order: list[str] = []
    missing_layers: list[str] = []
    for configured_name, configured_weight in EMBEDDING_LAYERS:
        try:
            resolved_name = resolve_hook_layer_name(yolo, str(configured_name))
        except KeyError:
            missing_layers.append(str(configured_name))
            continue
        if resolved_name not in resolved_weight_by_layer:
            resolved_order.append(resolved_name)
            resolved_weight_by_layer[resolved_name] = 0.0
        resolved_weight_by_layer[resolved_name] += float(configured_weight)

    if missing_layers:
        print(
            "WARNING: multi-layer embedding missing configured layers: "
            + ", ".join(missing_layers)
            + ". Falling back to available layers.",
            file=sys.stderr,
        )

    normalized = _normalize_weights([(name, resolved_weight_by_layer[name]) for name in resolved_order])
    if len(normalized) < 2:
        raise RuntimeError(
            "Multi-layer embedding requires at least 2 available layers after resolution. "
            f"Configured={len(EMBEDDING_LAYERS)} available={len(normalized)}"
        )

    layers = tuple(name for name, _ in normalized)
    weights = tuple(float(weight) for _, weight in normalized)
    return HookConfig(
        layer=layers[0],
        stride=stride,
        requested_layer="multi_layer_embedding",
        layers=layers,
        layer_weights=weights,
        multi_layer_enabled=True,
    )


def _build_weighted_embedding(
    *,
    layer_outputs: dict[str, Any],
    layer_names: tuple[str, ...],
    layer_weights: tuple[float, ...],
    batch_index: int,
    bbox_xyxy: list[float],
    pool: "torch.nn.Module",
    frame_h: int,
    frame_w: int,
    warn_once: set[str],
) -> tuple["np.ndarray", bool]:
    available_vectors: list["np.ndarray"] = []
    available_weights: list[float] = []
    small_flags: list[bool] = []

    for layer_name, layer_weight in zip(layer_names, layer_weights):
        output = layer_outputs.get(layer_name)
        if output is None:
            warn_key = f"missing_output:{layer_name}"
            if warn_key not in warn_once:
                print(
                    f"WARNING: layer '{layer_name}' did not produce hook output; skipping this layer for embedding.",
                    file=sys.stderr,
                )
                warn_once.add(warn_key)
            continue
        if not hasattr(output, "ndim") or int(output.ndim) != 4:
            warn_key = f"invalid_ndim:{layer_name}"
            if warn_key not in warn_once:
                shape = tuple(output.shape) if hasattr(output, "shape") else "(unknown)"
                print(
                    f"WARNING: layer '{layer_name}' output shape {shape} is not [B,C,H,W]; skipping this layer.",
                    file=sys.stderr,
                )
                warn_once.add(warn_key)
            continue
        if int(output.shape[0]) <= int(batch_index):
            warn_key = f"short_batch:{layer_name}"
            if warn_key not in warn_once:
                print(
                    f"WARNING: layer '{layer_name}' output batch ({int(output.shape[0])}) is smaller than "
                    f"requested index {batch_index}; skipping this layer.",
                    file=sys.stderr,
                )
                warn_once.add(warn_key)
            continue

        try:
            raw_vec, small_crop = build_raw_activation_vector(
                fmap=output[batch_index],
                bbox_xyxy=bbox_xyxy,
                pool=pool,
                frame_h=frame_h,
                frame_w=frame_w,
            )
        except Exception as exc:
            warn_key = f"extract_fail:{layer_name}"
            if warn_key not in warn_once:
                print(
                    f"WARNING: failed to extract pooled vector for layer '{layer_name}': {exc}",
                    file=sys.stderr,
                )
                warn_once.add(warn_key)
            continue

        vec_n = l2_normalize(raw_vec)
        if float(np.linalg.norm(vec_n)) <= 0.0:
            warn_key = f"zero_norm:{layer_name}"
            if warn_key not in warn_once:
                print(
                    f"WARNING: layer '{layer_name}' produced zero-norm pooled vector; skipping this layer.",
                    file=sys.stderr,
                )
                warn_once.add(warn_key)
            continue

        available_vectors.append(vec_n.astype(np.float32, copy=False))
        available_weights.append(float(layer_weight))
        small_flags.append(bool(small_crop))

    if len(available_vectors) < 2:
        raise RuntimeError(
            "Multi-layer embedding requires at least 2 valid layer vectors per detection. "
            f"Got {len(available_vectors)} from configured {len(layer_names)} layers."
        )

    weight_sum = float(sum(available_weights))
    if weight_sum <= 0.0:
        raise RuntimeError("Multi-layer embedding weights collapsed to zero after per-detection fallback.")

    scaled_parts = [
        vec * float(weight / weight_sum) for vec, weight in zip(available_vectors, available_weights)
    ]
    combined = np.concatenate(scaled_parts, axis=0).astype(np.float32, copy=False)
    return l2_normalize(combined), any(small_flags)


def _layer_manifest_section(cfg: HookConfig) -> dict[str, Any]:
    resolved_layers = list(cfg.layers) if cfg.layers else [cfg.layer]
    layer_weights = list(cfg.layer_weights) if cfg.layer_weights else [1.0]
    embedding_layers_payload = [
        {"layer": layer_name, "weight": float(weight)}
        for layer_name, weight in zip(resolved_layers, layer_weights)
    ]
    return {
        "aliases": list(ACTIVATION_LAYER_ALIASES),
        "requested": cfg.requested_layer,
        "resolved": cfg.layer,
        "actual": {"resolved_hook_layer": cfg.layer, "resolved_hook_layers": resolved_layers},
        "strides": {"resolved_hook_layer": cfg.stride},
        "embedding": {
            "enabled": bool(cfg.multi_layer_enabled),
            "resolved_layers": resolved_layers,
            "layer_weights": layer_weights,
            "layers": embedding_layers_payload,
        },
    }


def _projection_caveats(*, total_detections: int, effective_pca_dim: int) -> list[str]:
    caveats: list[str] = []
    # Per-run PCA can be unstable when fitted on a small number of detections relative to components.
    if total_detections < max(50, (3 * max(1, effective_pca_dim))):
        caveats.append("low_sample_count_for_per_run_pca_fit")
    return caveats


def collect_single_pass_trace(
    *,
    yolo: Any,
    video_path: str,
    sample_rate: int,
    hook_config: HookConfig,
    batch_size: int,
    dino_embedder: DinoEmbedder | None = None,
) -> tuple[list[CollectedFrame], CollectionStats]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    module_map = get_module_map(yolo)
    # Pool on the active device, then move pooled vectors to CPU in model helpers.
    import torch  # local import keeps top-level import cost low in non-runtime paths

    pool = torch.nn.AdaptiveAvgPool2d(POOL_SIZE)
    frames: list[CollectedFrame] = []
    stats = CollectionStats()
    layer_names = tuple(hook_config.layers) if hook_config.layers else (hook_config.layer,)
    layer_weights = tuple(hook_config.layer_weights) if hook_config.layer_weights else (1.0,)
    is_multi = bool(hook_config.multi_layer_enabled and len(layer_names) > 1)
    warn_once: set[str] = set()
    dino_active = bool(dino_embedder is not None)

    def process_batch(batch_items: list[tuple[int, Any]]) -> None:
        if not batch_items:
            return

        # Register and remove all hooks around each forward pass to avoid stale handles in long runs.
        with FeatureHookCollector(module_map=module_map, layer_names=list(layer_names)) as hooks:
            hooks.clear()
            batch_images = [frame for _, frame in batch_items]
            results = yolo(batch_images, verbose=False)
            if len(results) != len(batch_items):
                raise RuntimeError("YOLO result count does not match inference batch size")

            for layer_name in layer_names:
                if layer_name not in hooks.outputs:
                    raise RuntimeError(f"Expected hook output for layer '{layer_name}' was not captured.")

            layer_outputs: dict[str, Any] = {}
            for layer_name in layer_names:
                output = hooks.outputs.get(layer_name)
                if output is None:
                    layer_outputs[layer_name] = None
                    continue
                if hasattr(output, "detach"):
                    output = output.detach()
                if hasattr(output, "cpu"):
                    output = output.cpu()
                layer_outputs[layer_name] = output

            for index, (frame_num, _frame) in enumerate(batch_items):
                detections = extract_detections_from_result(results[index], sort_by_confidence=True)
                for det in detections:
                    if is_multi:
                        yolo_vec, small_crop_flag = _build_weighted_embedding(
                            layer_outputs=layer_outputs,
                            layer_names=layer_names,
                            layer_weights=layer_weights,
                            batch_index=index,
                            bbox_xyxy=det.bbox,
                            pool=pool,
                            frame_h=int(_frame.shape[0]),
                            frame_w=int(_frame.shape[1]),
                            warn_once=warn_once,
                        )
                    else:
                        single_output = layer_outputs.get(hook_config.layer)
                        if single_output is None:
                            raise RuntimeError(
                                f"Expected hook output for single layer '{hook_config.layer}' is missing."
                            )
                        yolo_vec, small_crop_flag = build_raw_activation_vector(
                            fmap=single_output[index],
                            bbox_xyxy=det.bbox,
                            pool=pool,
                            frame_h=int(_frame.shape[0]),
                            frame_w=int(_frame.shape[1]),
                        )

                    det.raw_vector = yolo_vec
                    det.small_crop_flag = bool(small_crop_flag)
                    det.dino_vector = None
                    det.dino_available = False
                    if not dino_active:
                        continue

                    dino_small_crop = False
                    dino_valid_crop = False
                    try:
                        dino_result = extract_dino_embedding(
                            frame_bgr=_frame,
                            bbox_xyxy=det.bbox,
                            embedder=dino_embedder,
                        )
                        dino_vec = np.asarray(dino_result.vector, dtype=np.float32)
                        dino_small_crop = bool(dino_result.tiny_crop)
                        dino_valid_crop = bool(dino_result.valid_crop)
                    except Exception as exc:
                        warn_key = "dino_extract_fail"
                        if warn_key not in warn_once:
                            print(
                                "WARNING: DINO embedding extraction failed; "
                                "marking DINO sidecar unavailable for affected detections. "
                                f"Error: {exc}",
                                file=sys.stderr,
                            )
                            warn_once.add(warn_key)
                        continue

                    if not dino_valid_crop:
                        warn_key = "dino_invalid_crop"
                        if warn_key not in warn_once:
                            print(
                                "WARNING: One or more detection crops were invalid for DINO; "
                                "marking DINO sidecar unavailable for those detections.",
                                file=sys.stderr,
                            )
                            warn_once.add(warn_key)
                        continue

                    if dino_small_crop:
                        warn_key = "dino_tiny_crop"
                        if warn_key not in warn_once:
                            print(
                                f"WARNING: One or more DINO crops were smaller than {DINO_TINY_CROP_MIN_SIZE}x"
                                f"{DINO_TINY_CROP_MIN_SIZE}; marking DINO sidecar unavailable for those detections.",
                                file=sys.stderr,
                            )
                            warn_once.add(warn_key)
                        continue

                    if int(dino_vec.shape[0]) != int(DINO_EMBEDDING_DIM):
                        warn_key = "dino_dim_mismatch"
                        if warn_key not in warn_once:
                            print(
                                "WARNING: DINO embedding had unexpected dimension; "
                                "marking DINO sidecar unavailable for affected detections.",
                                file=sys.stderr,
                            )
                            warn_once.add(warn_key)
                        continue

                    dino_vec = l2_normalize(dino_vec)
                    if float(np.linalg.norm(dino_vec)) <= 0.0:
                        warn_key = "dino_zero_norm"
                        if warn_key not in warn_once:
                            print(
                                "WARNING: DINO embedding had zero norm; marking DINO sidecar unavailable.",
                                file=sys.stderr,
                            )
                            warn_once.add(warn_key)
                        continue

                    det.dino_vector = dino_vec.astype(np.float32, copy=False)
                    det.dino_available = True

                frames.append(CollectedFrame(frame_num=int(frame_num), detections=detections))
                stats.total_sampled_frames += 1
                if detections:
                    stats.frames_with_detections += 1
                stats.total_detections += len(detections)

    pending: list[tuple[int, Any]] = []
    for frame_num, frame in FrameSampler(video_path, sample_rate):
        pending.append((int(frame_num), frame))
        if len(pending) >= batch_size:
            process_batch(pending)
            pending = []
    if pending:
        process_batch(pending)

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
        det.projected_vector = l2_normalize(proj)

    return pca, int(effective_pca_dim)


def build_enriched_payload(
    frames: list[CollectedFrame],
    projection_dim: int,
    hook_config: HookConfig,
) -> list[dict[str, Any]]:
    activation_layers = list(hook_config.layers) if hook_config.layers else [hook_config.layer]
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
                        "dim": int(projection_dim),
                        "layers": activation_layers,
                        "pool": POOL_STRATEGY,
                        "projection": f"pca_{projection_dim}",
                        "small_crop_flag": bool(det.small_crop_flag),
                        "dino_vector": (
                            [float(v) for v in np.asarray(det.dino_vector, dtype=np.float32).tolist()]
                            if det.dino_vector is not None
                            else None
                        ),
                        "dino_available": bool(det.dino_available and det.dino_vector is not None),
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
    dino_enabled: bool = False,
    dino_model: str | None = None,
    dino_load_error: str | None = None,
) -> dict[str, Any]:
    first_detection = next(_iter_detections(frames), None)
    if first_detection is None or first_detection.raw_vector is None:
        raise RuntimeError("Cannot build manifest without at least one activation vector")

    artifacts = build_output_artifacts(run_config.output_dir)
    raw_dim = int(len(first_detection.raw_vector))
    return {
        "schema_version": SCHEMA_VERSION,
        "projection_file": os.path.basename(artifacts.pca_path),
        "projection_dim": int(effective_pca_dim),
        "projection_dim_requested": int(run_config.pca_dim),
        "fitted_pca_components": int(effective_pca_dim),
        "projection_fit_scope": "per_run_all_detections",
        "projection_caveats": _projection_caveats(
            total_detections=int(stats.total_detections),
            effective_pca_dim=int(effective_pca_dim),
        ),
        "fit_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_video_hash_sha256": sha256_file(run_config.video_path),
        "input_video_file": os.path.basename(run_config.video_path),
        "pool": POOL_STRATEGY,
        "pool_size": list(POOL_SIZE),
        "layers": _layer_manifest_section(hook_config),
        "raw_activation_dim": raw_dim,
        "raw_embedding_dim": raw_dim,
        "dino_enabled": bool(dino_enabled),
        "dino_role": "relink_sidecar",
        "dino_model": str(dino_model) if dino_model is not None else None,
        "dino_load_error": str(dino_load_error) if dino_load_error is not None else None,
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


def _build_hook_config(yolo: Any, *, layer_name: str, stride: int) -> HookConfig:
    use_multi = bool(
        EMBEDDING_LAYERS
        and not _multi_layer_embedding_disabled()
        and str(layer_name).strip() == str(DEFAULT_HEAD_LAYER)
    )
    if use_multi:
        return _resolve_multi_layer_hook_config(yolo, stride)

    resolved_layer = resolve_hook_layer_name(yolo, layer_name)
    return HookConfig(
        layer=resolved_layer,
        stride=stride,
        requested_layer=layer_name,
        layers=(resolved_layer,),
        layer_weights=(1.0,),
        multi_layer_enabled=False,
    )


def run_trace_enrichment(
    *,
    video_path: str,
    model_name: str,
    output_dir: str,
    sample_rate: int,
    layer_name: str = DEFAULT_HEAD_LAYER,
    stride: int = DEFAULT_HEAD_STRIDE,
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
    artifacts = build_output_artifacts(output_dir)

    yolo = load_yolo(model_name)
    dino_load_error: str | None = None
    dino_embedder: DinoEmbedder | None = None
    dino_requested = bool(not _dino_disabled())
    if dino_requested:
        preferred_device = _infer_model_device(yolo)
        try:
            dino_embedder = load_dino_embedder(
                model_name=DINO_MODEL_NAME,
                feature_dim=DINO_EMBEDDING_DIM,
                hub_repo=DINO_MODEL_REPO,
                preferred_device=preferred_device,
                load_timeout_seconds=DINO_LOAD_TIMEOUT_SECONDS,
                input_size=DINO_INPUT_SIZE,
                tiny_crop_min=DINO_TINY_CROP_MIN_SIZE,
                crop_padding_ratio=DINO_CROP_PADDING_RATIO,
            )
        except DinoUnavailableError as exc:
            dino_load_error = str(exc)
            dino_embedder = None
            print(
                "WARNING: DINO model failed to load; DINO sidecar will be unavailable for this run. "
                f"Error: {exc}",
                file=sys.stderr,
            )
    dino_enabled = bool(dino_embedder is not None)

    hook_config = _build_hook_config(yolo, layer_name=layer_name, stride=stride)
    frames, stats = collect_single_pass_trace(
        yolo=yolo,
        video_path=video_path,
        sample_rate=sample_rate,
        hook_config=hook_config,
        batch_size=batch_size,
        dino_embedder=dino_embedder,
    )

    if stats.total_detections <= 0:
        raise RuntimeError(
            f"No detections found in sampled frames for '{video_path}'. Trace enrichment requires detections."
        )

    pca, effective_pca_dim = fit_pca_and_project(frames, pca_dim)
    enriched_payload = build_enriched_payload(frames, effective_pca_dim, hook_config)
    manifest = build_manifest(
        run_config=run_config,
        hook_config=hook_config,
        effective_pca_dim=effective_pca_dim,
        frames=frames,
        stats=stats,
        dino_enabled=dino_enabled,
        dino_model=(f"{DINO_MODEL_REPO}/{DINO_MODEL_NAME}" if dino_requested else None),
        dino_load_error=dino_load_error,
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
