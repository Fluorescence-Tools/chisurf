"""Standalone launcher for the anisotropy wizard (``csg_anisotropy``)."""

from __future__ import annotations


def main() -> None:
    """Launch the anisotropy wizard as a standalone Qt application."""
    from qtpy import QtWidgets

    from .gui.tool import AnisotropyWizard

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = AnisotropyWizard()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
