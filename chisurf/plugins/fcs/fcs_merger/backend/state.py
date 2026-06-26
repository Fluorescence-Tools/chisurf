"""Persisted state schema for the FCS-Merger plugin."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict


@dataclasses.dataclass
class MergerState:
    """Last-used correlation folder."""

    correlation_folder: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MergerState":
        return cls(correlation_folder=str((d or {}).get("correlation_folder", "")))
