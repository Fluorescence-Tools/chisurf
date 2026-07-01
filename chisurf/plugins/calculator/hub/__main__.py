"""Standalone launcher for the Calculators hub (``csg_calculators``)."""

from __future__ import annotations


def main() -> None:
    """Launch the Calculators hub as a standalone Qt application."""
    from qtpy import QtWidgets

    from .gui.tool import CalculatorHub

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = CalculatorHub()
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
