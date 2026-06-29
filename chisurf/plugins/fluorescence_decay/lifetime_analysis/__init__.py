"""Integrated fluorescence lifetime analysis plugin."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = (
    _manifest.display_name
    if _manifest is not None
    else "Spectroscopy:Fluorescence decay:Lifetime Analysis"
)


def load():
    """Return the integrated lifetime analysis tool."""
    from chisurf.plugins.fluorescence_decay.lifetime_analysis.gui.tool import (
        LifetimeAnalysisTool,
    )

    return LifetimeAnalysisTool()


if __name__ == "plugin":
    window = load()
    window.show()
    window.raise_()
    window.activateWindow()


__all__ = ["LifetimeAnalysisTool", "load", "name"]
