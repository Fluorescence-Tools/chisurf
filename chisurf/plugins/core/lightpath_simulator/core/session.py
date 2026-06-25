"""Server-session state for the light-path simulator plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .workflow import get_probes_info, resolve_db_path


def _utc_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _catalogue_key(db_path: str | None) -> str:
    """Return a cache key for a resolved spectra database."""
    resolved = Path(resolve_db_path(db_path)).expanduser()
    try:
        stat = resolved.stat()
        return f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return f"{resolved}:missing"


@dataclass
class LightPathSessionState:
    """JSON-safe runtime state owned by the ChiSurf server session."""

    probe_catalogues: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_probe_db_path: str | None = None
    last_graph: dict[str, Any] | None = None
    last_result: dict[str, Any] | None = None
    last_operation_id: str | None = None
    updated_at: str | None = None

    def get_probe_catalogue(self, db_path: str | None = None) -> dict[str, Any]:
        """Return cached probe metadata for the current spectra database."""
        key = _catalogue_key(db_path)
        cached = self.probe_catalogues.get(key)
        if cached is not None:
            self.last_probe_db_path = cached["db_path"]
            return cached["payload"]

        payload = get_probes_info(db_path=db_path)
        resolved = str(Path(resolve_db_path(db_path)).expanduser())
        self.probe_catalogues = {
            key: {
                "db_path": resolved,
                "payload": payload,
                "loaded_at": _utc_iso(),
            }
        }
        self.last_probe_db_path = resolved
        self.updated_at = _utc_iso()
        return payload

    def record_simulation(
        self,
        graph: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Store the latest simulation payload in the session namespace."""
        self.last_graph = graph
        self.last_result = result
        self.updated_at = _utc_iso()

    def record_save(self, result: dict[str, Any]) -> None:
        """Store the latest persisted lightpath operation id."""
        operation_id = result.get("operation_id")
        if operation_id is not None:
            self.last_operation_id = str(operation_id)
        self.updated_at = _utc_iso()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe session snapshot."""
        return {
            "probe_catalogue_count": len(self.probe_catalogues),
            "last_probe_db_path": self.last_probe_db_path,
            "last_operation_id": self.last_operation_id,
            "has_last_graph": self.last_graph is not None,
            "has_last_result": self.last_result is not None,
            "updated_at": self.updated_at,
        }
