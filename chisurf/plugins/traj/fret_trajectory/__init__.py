"""FRET analysis from molecular dynamics trajectories."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.plugins.traj.fret_trajectory.gui import Structure2Transfer

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Structure:Trajectory:FRET"

__all__ = ["Structure2Transfer"]

if __name__ == "plugin":
    window = Structure2Transfer()
    window.show()
