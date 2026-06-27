"""State model for the 2D-FLCS plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlcTwoDState:
    """Persisted state for the 2D-FLCS plugin."""

    tttr_path: str | None = None
    irf_path: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    last_result_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible state payload."""
        return {
            "tttr_path": self.tttr_path,
            "irf_path": self.irf_path,
            "settings": dict(self.settings),
            "last_result_summary": self.last_result_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlcTwoDState":
        """Create state from a JSON-compatible payload."""
        return cls(
            tttr_path=data.get("tttr_path"),
            irf_path=data.get("irf_path"),
            settings=dict(data.get("settings", {})),
            last_result_summary=data.get("last_result_summary"),
        )
