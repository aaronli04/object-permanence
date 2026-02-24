"""Shared helpers for inspecting YOLOv8 module structure."""

from __future__ import annotations

from typing import Dict, List


def get_module_map(yolo) -> Dict[str, object]:
    """Return a name->module map for the underlying Ultralytics model."""
    return dict(yolo.model.model.named_modules())


def list_c2f_module_names(yolo) -> List[str]:
    """List all named C2f modules in model traversal order."""
    return [
        name
        for name, module in yolo.model.model.named_modules()
        if module.__class__.__name__ == "C2f"
    ]


def print_c2f_candidates(yolo) -> None:
    """Print C2f module names with stable indices for hook selection."""
    print("C2f candidate modules (for hook placement):")
    names = list_c2f_module_names(yolo)
    if not names:
        print("  (none found)")
        return
    for idx, name in enumerate(names):
        print(f"  [{idx}] {name}")
