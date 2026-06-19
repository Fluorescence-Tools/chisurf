"""BVA plugin state namespace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BvaState:
    """Plugin state for burst_bva namespace."""

    last_analysis_folder: str | None = None
    last_settings: dict[str, Any] = field(default_factory=dict)
    last_result_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_analysis_folder": self.last_analysis_folder,
            "last_settings": dict(self.last_settings),
            "last_result_summary": self.last_result_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BvaState:
        return cls(
            last_analysis_folder=data.get("last_analysis_folder"),
            last_settings=dict(data.get("last_settings", {})),
            last_result_summary=data.get("last_result_summary"),
        )
