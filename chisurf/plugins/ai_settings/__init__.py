"""
AI Settings plugin for configuring API providers and backends.

This plugin provides a GUI for managing centralized AI API settings,
including provider selection, base URL, model selection, and API key.
"""
from chisurf.plugins.ai_settings.plugin import AISettingsWidget

name = "Tools:AI Settings"
icon = "🤖"  # Robot emoji for AI

def load():
    """Return the plugin's main widget instance."""
    return AISettingsWidget()

__all__ = ["name", "load", "icon", "AISettingsWidget"]

if __name__ == "plugin":
    window = AISettingsWidget()
    window.show()
