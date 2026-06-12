"""Plugin state namespace for {{ cookiecutter.plugin_display_name }}.

All state is JSON-serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PluginState:
    """State for the ``{{ cookiecutter.plugin_name }}`` namespace."""

    settings: dict[str, Any] = field(default_factory=dict)
    last_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": dict(self.settings),
            "last_result": self.last_result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginState:
        return cls(
            settings=dict(data.get("settings", {})),
            last_result=data.get("last_result"),
        )
