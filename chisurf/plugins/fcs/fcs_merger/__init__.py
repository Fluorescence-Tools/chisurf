"""FCS Merger plugin.

Merge / average multiple fluorescence correlation spectroscopy (FCS) curves to
improve signal-to-noise. The compute core is Qt-free (the shared
``chisurf.core.fluorescence.fcs.merge`` primitives, re-exported via :mod:`.core`
and :mod:`.api`), exposed through backend RPC services (:mod:`.backend`) and
reached from the GUI through :class:`.gui.FcsMergerClient`. The GUI is the
``ChisurfWizard`` wrapper around the shared ``WizardFcsMerger`` page.
"""

from __future__ import annotations

from pathlib import Path as _Path

from chisurf.core.plugin import load_manifest as _load_manifest

from .api import compute_average_correlations, merge_folder  # noqa: F401
from .gui.client import FcsMergerClient  # noqa: F401

_manifest = _load_manifest(_Path(__file__).with_name("manifest.json"))
name = (
    _manifest.display_name if _manifest is not None
    else "Spectroscopy:Fluorescence Correlation Spectroscopy:FCS-Merger"
)

# Hidden from the menu: surfaced inside the FCS Toolbox meta tool.
menu_hidden = True

__all__ = [
    "FcsMergerClient",
    "compute_average_correlations",
    "merge_folder",
    "menu_hidden",
    "name",
]
