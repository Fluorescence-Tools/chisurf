"""JSON-serializable DTOs for {{ cookiecutter.plugin_display_name }}."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PluginSettings:
    """Example plugin settings DTO."""
    example_param: str = "default"
    example_number: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginSettings:
        return cls(**{k: v for k, v in data.items() if k in (
            "example_param", "example_number"
        )})
