"""Combined FRET / HomoFRET Calculator Plugin.

Provides heteroFRET and homoFRET parameter calculations in a single tabbed
window, backed by the new client-server architecture.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness

# Load manifest as source of truth
_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Main:Tools:FRET-Calculator"

from .gui.tool import FretCalculatorTool  # noqa: E402

__all__ = ["FretCalculatorTool"]

if __name__ == "plugin":
    window = FretCalculatorTool()
    if _manifest is not None:
        apply_manifest_statefulness(window, _manifest)
    window.show()
    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass
