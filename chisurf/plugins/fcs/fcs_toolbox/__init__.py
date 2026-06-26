"""FCS Toolbox plugin — a meta tool hosting several FCS tools behind an icon rail.

Currently bundles 2D-FLCS, Burst-wise FCS and the Diffusion/Volume calculator.
The ribbon execs this file with ``__name__ == "plugin"``.
"""

from __future__ import annotations

from .tool import FcsToolboxTool

name = "Spectroscopy:Fluorescence Correlation Spectroscopy:FCS Toolbox"

__all__ = ["FcsToolboxTool", "name"]


if __name__ == "plugin":  # pragma: no cover
    _fcs_toolbox_window = FcsToolboxTool()
    _fcs_toolbox_window.show()
