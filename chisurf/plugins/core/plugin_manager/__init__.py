from __future__ import annotations

from pathlib import Path
import chisurf as cs
from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness
from chisurf.plugins.core.plugin_manager.gui.tool import (
    PluginManagerWidget,
    read_module_docstring,
)

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Setup:Plugins"

description = (
    "This tool manages the list of active/disabled plugins in ChiSurf."
)
icon = "🔌"


def load():
    """Return the plugin manager widget."""
    return PluginManagerWidget()


__all__ = [
    "PluginManagerWidget",
    "read_module_docstring",
    "load",
    "name",
    "description",
    "icon",
]

if __name__ == "plugin":
    try:
        parent = getattr(cs, "cs", None)
        window = PluginManagerWidget(parent=parent)
        if _manifest is not None:
            apply_manifest_statefulness(window, _manifest)
        window.show()
        window.raise_()
        window.activateWindow()
    except Exception as exc:
        print(f"Failed to open Plugin Manager: {exc}")
        import traceback
        traceback.print_exc()
