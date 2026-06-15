"""
Global View

This plugin provides a graphical interface for visualizing and managing parameter relationships
in fitting models through an interactive network graph representation.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Main:Tools:🕸️ Global View"
