#!/usr/bin/env python3
"""List YOLOv8 C2f layer candidates for hook selection."""

from __future__ import annotations

import argparse

from .model import ensure_model_runtime_dependencies, get_module_map, list_c2f_module_names, load_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="List YOLOv8 C2f modules for hook discovery.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLOv8 model name or local weights path.",
    )
    args = parser.parse_args()

    ensure_model_runtime_dependencies()
    yolo = load_yolo(args.model)
    core = yolo.model

    print(f"Model: {args.model}")
    print(f"Core model type: {core.__class__.__name__}")
    print("C2f modules:")
    module_map = get_module_map(yolo)
    names = list_c2f_module_names(yolo)
    if not names:
        print("  (none found)")
        return
    for idx, name in enumerate(names):
        module = module_map[name]
        ch = getattr(module, "cv2", None)
        out_channels = getattr(ch, "out_channels", "unknown")
        print(f"  [{idx}] {name} | type={module.__class__.__name__} | out_channels={out_channels}")


if __name__ == "__main__":
    main()
