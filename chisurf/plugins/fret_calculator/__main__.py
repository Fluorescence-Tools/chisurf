"""
Entry point for the FRET Calculator tool.

This module provides a command-line entry point for the FRET Calculator tool,
which allows users to calculate and convert between various FRET-related quantities.
"""

import sys
from qtpy import QtWidgets
from .tau2r import FRETCalculator

def main():
    """
    Main entry point for the FRET Calculator tool.
    Creates and shows the FRETCalculator widget.
    """
    app = QtWidgets.QApplication(sys.argv)
    win = FRETCalculator()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()