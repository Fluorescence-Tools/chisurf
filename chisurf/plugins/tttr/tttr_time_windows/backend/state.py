"""Time Window Bins plugin state namespace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimeWindowState:
    """Plugin state for tttr_time_windows namespace.

    Lives under ``session_state.plugins["tttr_time_windows"]``.
    """

    selected_files: list[str] = field(default_factory=list)
    time_window_ms: float = 10.0
    output_dir: str | None = None
    last_result_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_files": list(self.selected_files),
            "time_window_ms": self.time_window_ms,
            "output_dir": self.output_dir,
            "last_result_summary": self.last_result_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeWindowState:
        return cls(
            selected_files=list(data.get("selected_files", [])),
            time_window_ms=float(data.get("time_window_ms", 10.0)),
            output_dir=data.get("output_dir"),
            last_result_summary=data.get("last_result_summary"),
        )
