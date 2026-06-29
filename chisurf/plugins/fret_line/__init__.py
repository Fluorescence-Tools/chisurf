"""FRET line generator plugin — static, dynamic, WLC, and mixture modes.

Provides a GUI tool (``FRETLineTool``) for computing FRET efficiency lines
over parameter ranges and overlaying them on smFRET 2D histograms in ndxplorer.
The widget is built via AutoForm (PRD-40) from ``fret_line.view.json``.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "FRET:FRET Line Generator"

from .gui.tool import FRETLineTool  # noqa: E402

__all__ = ["FRETLineTool"]

if __name__ == "plugin":
    window = FRETLineTool()
    if _manifest is not None:
        apply_manifest_statefulness(window, _manifest)
    window.show()
    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass
