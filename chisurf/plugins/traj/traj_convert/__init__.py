"""Trajectory converter tool."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest
from chisurf.plugins.traj.traj_convert.widget import MDConverter

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Structure:Trajectory:Convert"

__all__ = ["MDConverter"]

if __name__ == "plugin":
    window = MDConverter()
    window.show()
