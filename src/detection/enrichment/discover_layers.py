"""Utility to inspect YOLOv8 modules and list C2f layer candidates."""

from __future__ import annotations

import argparse

from ultralytics import YOLO
from .introspection import get_module_map, list_c2f_module_names


def main() -> None:
    parser = argparse.ArgumentParser(description="List YOLOv8 C2f modules for hook discovery.")
    parser.add_argument(
        "--model",
        default="yolov8m.pt",
        help="Ultralytics YOLOv8 model name or local weights path.",
    )
    args = parser.parse_args()

    yolo = YOLO(args.model)
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
