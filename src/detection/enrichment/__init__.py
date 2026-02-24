"""Activation-enrichment subpackage for post-processing YOLO detections.

Keep this package init lightweight so subcommands like layer discovery do not
pull in heavy runtime dependencies (e.g., sklearn) unnecessarily.
"""

__all__: list[str] = []
