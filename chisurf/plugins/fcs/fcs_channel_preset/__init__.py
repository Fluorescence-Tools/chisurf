from __future__ import annotations

from pathlib import Path
import chisurf as cs
from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness
from chisurf.plugins.fcs.fcs_channel_preset.gui.tool import FCSChannelWidget, FCSChannelDialog

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Setup:FCS Definitions"

description = "FCS channel definition plugin per detector setup"
icon = "📡"


def load():
    """Return the FCS channel definition widget."""
    return FCSChannelWidget()


__all__ = [
    "FCSChannelWidget",
    "FCSChannelDialog",
    "load",
    "name",
    "description",
    "icon",
]

if __name__ == "plugin":
    try:
        parent = getattr(cs, "cs", None)
        window = FCSChannelWidget(parent=parent)
        if _manifest is not None:
            apply_manifest_statefulness(window, _manifest)
        window.show()
    except Exception as exc:
        print(f"Failed to open FCS Channel preset: {exc}")
