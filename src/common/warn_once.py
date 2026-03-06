"""Helpers for emitting process-local warnings only once."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class WarnOnce:
    """Track warning keys and emit each warning at most once."""

    stream: TextIO = sys.stderr
    _seen: set[str] = field(default_factory=set)

    def warn(self, key: str, message: str) -> None:
        if key in self._seen:
            return
        print(f"WARNING: {message}", file=self.stream)
        self._seen.add(key)

    def seen(self, key: str) -> bool:
        return key in self._seen
