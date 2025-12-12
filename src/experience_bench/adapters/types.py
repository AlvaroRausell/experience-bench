from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletionResult:
    text: str
    raw_usage: dict[str, Any] | None
    usage_derived: dict[str, Any] | None
