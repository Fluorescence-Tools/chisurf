"""Thin wrapper for BVA plugin (new-style DockArea tool)."""

from __future__ import annotations

from chisurf.plugins.burst.burst_bva.gui.tool import BVATool


class MainWindow(BVATool):
    """Legacy-compatible alias."""
    pass


if __name__ == "plugin":
    wizard = MainWindow()
    wizard.show()
