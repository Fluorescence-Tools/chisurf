from __future__ import annotations

from pathlib import Path
import chisurf as cs
from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness
from chisurf.plugins.core.model_manager.gui.tool import ModelManagerWidget

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Setup:Models"

description = "Model Manager Plugin for ChiSurf"
icon = "⚙️"


def load():
    """Return the model manager widget."""
    return ModelManagerWidget()


__all__ = [
    "ModelManagerWidget",
    "load",
    "name",
    "description",
    "icon",
]

if __name__ == "plugin":
    try:
        parent = getattr(cs, "cs", None)
        window = ModelManagerWidget(parent=parent)
        if _manifest is not None:
            apply_manifest_statefulness(window, _manifest)
        window.show()
        window.raise_()
        window.activateWindow()
    except Exception as exc:
        print(f"Failed to open Model Manager: {exc}")
        import traceback
        traceback.print_exc()
