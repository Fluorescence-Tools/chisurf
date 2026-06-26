"""FCS Tools plugin — a meta tool hosting several FCS tools behind an icon rail.

Bundles the detector/correlation setup wizards, 2D-FLCS, Burst-wise FCS, the
diffusion/volume calculator, the filter calculator and the FCS merger. Built on
the reusable ``MetaToolWindow``. The ribbon execs this file with
``__name__ == "plugin"``.
"""

from __future__ import annotations

from .tool import FcsToolboxTool

name = "Spectroscopy:Fluorescence Correlation Spectroscopy:FCS Tools"

__all__ = ["FcsToolboxTool", "name"]


if __name__ == "plugin":  # pragma: no cover
    _fcs_toolbox_window = FcsToolboxTool()
    _fcs_toolbox_window.show()
