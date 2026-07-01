"""Standalone launcher for the Wizards hub (``csg_wizards``)."""

from __future__ import annotations


def main() -> None:
    """Launch the Wizards hub as a standalone Qt application."""
    from qtpy import QtWidgets

    from .gui.tool import WizardHub

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = WizardHub()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
