"""Plugin state namespace for the FRET Calculator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FretCalculatorState:
    """Persistent state for the ``fret_calculator`` namespace."""

    last_fret_settings: dict[str, Any] = field(default_factory=dict)
    last_homo_fret_settings: dict[str, Any] = field(default_factory=dict)
    last_fret_result: dict[str, Any] | None = None
    last_homo_fret_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "last_fret_settings": dict(self.last_fret_settings),
            "last_homo_fret_settings": dict(self.last_homo_fret_settings),
            "last_fret_result": self.last_fret_result,
            "last_homo_fret_result": self.last_homo_fret_result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FretCalculatorState:
        """Construct from a JSON-compatible dictionary."""
        return cls(
            last_fret_settings=dict(data.get("last_fret_settings", {})),
            last_homo_fret_settings=dict(data.get("last_homo_fret_settings", {})),
            last_fret_result=data.get("last_fret_result"),
            last_homo_fret_result=data.get("last_homo_fret_result"),
        )
