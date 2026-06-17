"""Potential energy calculator plugin."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.plugins.traj.potential_energy.widget import PotentialEnergyWidget

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Structure:Trajectory:Energy Calculator"

__all__ = ["PotentialEnergyWidget"]

if __name__ == "plugin":
    window = PotentialEnergyWidget()
    window.show()
