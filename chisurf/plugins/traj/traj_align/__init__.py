"""Trajectory alignment tool."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.plugins.traj.traj_align.widget import AlignTrajectoryWidget

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Structure:Trajectory:Align"

__all__ = ["AlignTrajectoryWidget"]

if __name__ == "plugin":
    window = AlignTrajectoryWidget()
    window.show()
