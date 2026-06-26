"""Persisted state schema for the FCS confocal calculator plugin."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict


@dataclasses.dataclass
class ConfocalState:
    """Last-used calculator settings."""

    settings: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfocalState":
        return cls(settings=dict((d or {}).get("settings", {})))
