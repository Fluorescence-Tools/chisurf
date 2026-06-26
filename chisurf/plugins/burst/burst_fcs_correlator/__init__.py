"""Burst-wise FCS Correlator plugin.

Computes fluorescence correlation functions on a per-burst basis from Burst-ID
``.bst`` / BUR files. The compute core is Qt-free (:mod:`.core`), exposed via
backend RPC services (:mod:`.backend`) and reached from the GUI through
:class:`.gui.BurstFcsClient`. The GUI (:class:`.gui.BurstFcsTool`) renders its
settings and dockable plots declaratively via AutoForm. The legacy
:class:`.wizard.BurstWiseFCSWizard` is kept for reference.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.core.plugin.registry import apply_manifest_statefulness

from .gui.tool import BurstFcsTool
from .wizard import BurstWiseFCSWizard

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Spectroscopy:Fluorescence Correlation Spectroscopy:Burst-wise FCS"

__all__ = ["BurstFcsTool", "BurstWiseFCSWizard", "name"]


if __name__ == "plugin":  # pragma: no cover
    win = BurstFcsTool()
    if _manifest is not None:
        apply_manifest_statefulness(win, _manifest)
    win.show()
