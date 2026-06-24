from __future__ import annotations

from pathlib import Path
import chisurf as cs
from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness
from chisurf.plugins.core.setup.gui.tool import UnifiedSettingsTool

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Setup:Settings"

description = "Unified Settings Dialog for ChiSurf"
icon = "⚙️"


def load():
    """Return the unified settings window."""
    return UnifiedSettingsTool()


__all__ = [
    "UnifiedSettingsTool",
    "load",
    "name",
    "description",
    "icon",
]

if __name__ == "plugin":
    try:
        parent = getattr(cs, "cs", None)
        window = UnifiedSettingsTool(parent=parent)
        if _manifest is not None:
            apply_manifest_statefulness(window, _manifest)
        window.show()
        window.raise_()
        window.activateWindow()
    except Exception as exc:
        print(f"Failed to open Settings: {exc}")
        import traceback
        traceback.print_exc()
