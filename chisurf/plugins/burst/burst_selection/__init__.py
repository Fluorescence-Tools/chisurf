"""Single Molecule Burst Selection Plugin."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

# Load manifest as source of truth
_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
    cli_entrypoint = _manifest.entrypoints.cli or ""
else:
    # Legacy fallback
    name = "Spectroscopy:Single-Molecule:Burst-Selection"
    cli_entrypoint = "burst-selection=chisurf.plugins.burst.burst_selection.cli:cli"

USE_LEGACY_GUI = False

__all__ = ["BurstSelectionTool", "USE_LEGACY_GUI"]


def __getattr__(name: str):
    """Lazily import the GUI tool (PRD-23: no Qt import as a package side effect).

    Keeps ``from ...burst_selection import BurstSelectionTool`` working while the
    Qt-free ``api``/``cli`` submodules can be imported headlessly (no Qt needed).
    """
    if name == "BurstSelectionTool":
        if USE_LEGACY_GUI:
            from .gui.legacy.burst_selector import BurstSelectionTool
        else:
            from .gui.tool import BurstSelectionTool
        return BurstSelectionTool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
