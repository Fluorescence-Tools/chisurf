"""JSON serialization helpers for {{ cookiecutter.plugin_display_name }}."""

from __future__ import annotations

from typing import Any

from .models import PluginSettings


def settings_from_dict(data: dict[str, Any]) -> PluginSettings:
    """Deserialize plugin settings from a JSON-compatible dict."""
    return PluginSettings.from_dict(data)
