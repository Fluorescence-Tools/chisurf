"""Thin Chimol ChiSurf plugin shim.

This module is intentionally very small. It exposes the plugin menu name and
creates a :class:`MolViewPluginWindow` from the core viewer package when
loaded as a plugin or when run as a standalone application.
"""

from __future__ import annotations

import sys

from qtpy import QtWidgets
from chisurf.plugins.chimol.chimol.app import MolViewPluginWindow

# Public launcher metadata (no plugin menu registration here)
__version__ = "0.0.1"

name = "Structure:Chimol"


def _create_window() -> MolViewPluginWindow:
    win = MolViewPluginWindow()
    try:
        win.resize(1000, 700)
    except Exception:
        pass
    return win


def main() -> None:
    """Launch Chimol as a standalone application."""

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    win = _create_window()
    win.show()
    if owns_app:
        sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()


if __name__ == "plugin":
    win = MolViewPluginWindow()
    win.show()
