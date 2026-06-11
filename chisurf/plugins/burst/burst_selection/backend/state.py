"""Burst Selection plugin state namespace.

All state is JSON-serializable. Large objects (TTTR readers, NumPy arrays)
are referenced by UID and stored in the server's private object store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BurstSelectionState:
    """Plugin state for burst_selection namespace.

    Lives under ``session_state.plugins["burst_selection"]``.
    """

    selected_files: List[str] = field(default_factory=list)
    last_job_id: Optional[str] = None
    last_job_status: Optional[str] = None
    last_result_summary: Optional[Dict[str, Any]] = None
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_files": list(self.selected_files),
            "last_job_id": self.last_job_id,
            "last_job_status": self.last_job_status,
            "last_result_summary": self.last_result_summary,
            "settings": dict(self.settings),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BurstSelectionState:
        return cls(
            selected_files=list(data.get("selected_files", [])),
            last_job_id=data.get("last_job_id"),
            last_job_status=data.get("last_job_status"),
            last_result_summary=data.get("last_result_summary"),
            settings=dict(data.get("settings", {})),
        )
