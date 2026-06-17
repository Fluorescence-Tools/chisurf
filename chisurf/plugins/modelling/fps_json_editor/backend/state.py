"""Plugin state namespace for FPS JSON Editor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FpsJsonEditorState:
    """State for the ``fps_json_editor`` namespace."""

    last_pdb_id: str | None = None
    last_pdb_path: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable state."""
        return {
            "last_pdb_id": self.last_pdb_id,
            "last_pdb_path": self.last_pdb_path,
            "settings": dict(self.settings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FpsJsonEditorState:
        """Build state from JSON-serializable data."""
        return cls(
            last_pdb_id=data.get("last_pdb_id"),
            last_pdb_path=data.get("last_pdb_path"),
            settings=dict(data.get("settings", {})),
        )
