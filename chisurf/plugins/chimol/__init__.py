"""Chimol plugin — molecular structure viewer for ChiSurf."""

from __future__ import annotations

import sys
from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Structure:Structure:ChiMOL"

__version__ = "0.2.0"

from chisurf.plugins.chimol.chimol.app import MolViewPluginWindow


def _create_window() -> MolViewPluginWindow:
    win = MolViewPluginWindow()
    try:
        win.resize(1000, 700)
    except Exception:
        pass
    return win


def main() -> None:
    """Launch Chimol as a standalone application."""
    from qtpy import QtWidgets
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    win = _create_window()
    win.show()
    if owns_app:
        sys.exit(app.exec())


if __name__ == "__main__":
    main()


if __name__ == "plugin":
    win = MolViewPluginWindow()
    win.show()
