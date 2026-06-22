from __future__ import annotations

from pathlib import Path
import chisurf as cs
from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness
from chisurf.plugins.core.setup_channel_definition.gui.tool import SetupChannelDefinitionWidget

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Setup:Channel Definition"

description = (
    "This tool defines and configures detector channels and time windows."
)
icon = "🔢"


def load():
    """Return the setup channel definition widget."""
    return SetupChannelDefinitionWidget()


__all__ = [
    "SetupChannelDefinitionWidget",
    "load",
    "name",
    "description",
    "icon",
]

if __name__ == "plugin":
    try:
        parent = getattr(cs, "cs", None)
        window = SetupChannelDefinitionWidget(parent=parent)
        if _manifest is not None:
            apply_manifest_statefulness(window, _manifest)
        window.show()
    except Exception as exc:
        print(f"Failed to open Setup Channel Definition: {exc}")
