"""Jordi G-Factor Calculator Plugin.

Exposes the Jordi G-factor GUI tool and registers backend services.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness

# Load manifest as source of truth
_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
    cli_entrypoint = _manifest.entrypoints.cli or ""
else:
    name = "Spectroscopy:Fluorescence decay:Jordi G-Factor Calculator"
    cli_entrypoint = ""

# Re-export the main widget class
from .gui.tool import JordiGFactorCalculator

__all__ = ["JordiGFactorCalculator"]


if __name__ == "plugin":
    window = JordiGFactorCalculator()
    if _manifest is not None:
        apply_manifest_statefulness(window, _manifest)
    window.show()
    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass
