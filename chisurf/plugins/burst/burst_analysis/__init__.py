"""Integrated burst analysis plugin."""

from __future__ import annotations

from pathlib import Path

import chisurf as cs
from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Spectroscopy:Single-Molecule:Burst Analysis"

description = "Integrated burst analysis workflow."


def load():
    """Return the integrated burst analysis widget."""
    from chisurf.plugins.burst.burst_analysis.gui.tool import BurstAnalysisTool

    return BurstAnalysisTool()


__all__ = ["BurstAnalysisTool", "load", "name", "description"]


def __getattr__(attr_name: str):
    """Lazily expose Qt GUI classes."""
    if attr_name == "BurstAnalysisTool":
        from chisurf.plugins.burst.burst_analysis.gui.tool import BurstAnalysisTool

        return BurstAnalysisTool
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


if __name__ == "plugin":
    try:
        from chisurf.plugins.burst.burst_analysis.gui.tool import BurstAnalysisTool

        parent = getattr(cs, "cs", None)
        window = BurstAnalysisTool(parent=parent)
        if _manifest is not None:
            apply_manifest_statefulness(window, _manifest)
        window.show()
        window.raise_()
        window.activateWindow()
    except Exception as exc:  # pragma: no cover - GUI launch path
        print(f"Failed to open Burst Analysis: {exc}")
        import traceback

        traceback.print_exc()
