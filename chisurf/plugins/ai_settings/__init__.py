from __future__ import annotations

from pathlib import Path
import chisurf as cs
from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness
from chisurf.plugins.ai_settings.gui.tool import AISettingsWidget

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Tools:AI Settings"

description = "AI Settings plugin for configuring API providers and backends."
icon = "🤖"


def load():
    """Return the plugin's main widget instance."""
    return AISettingsWidget()


__all__ = ["name", "load", "icon", "AISettingsWidget"]

if __name__ == "plugin":
    try:
        parent = getattr(cs, "cs", None)
        window = AISettingsWidget(parent=parent)
        if _manifest is not None:
            apply_manifest_statefulness(window, _manifest)
        window.show()
    except Exception as exc:
        print(f"Failed to open AI Settings: {exc}")
