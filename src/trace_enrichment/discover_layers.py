#!/usr/bin/env python3
"""List YOLOv8 detection-head layer candidates for hook selection."""

from __future__ import annotations

import argparse

from .model import ensure_model_runtime_dependencies, get_module_map, load_yolo


def main() -> None:
    parser = argparse.ArgumentParser(description="List YOLOv8 detection-head modules for hook discovery.")
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
    print("Detection head cv3 branch modules:")
    module_map = get_module_map(yolo)
    names = [name for name in module_map if ".cv3." in name]
    if not names:
        print("  (no cv3 modules found)")
        return
    for idx, name in enumerate(sorted(names)):
        module = module_map[name]
        print(f"  [{idx}] {name} | type={module.__class__.__name__}")
    print("")
    print("Recommended hook target:")
    print("  15  (C2f block in the neck, accessed as model.model[15])")


if __name__ == "__main__":
    main()
