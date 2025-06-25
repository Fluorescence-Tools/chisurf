"""
Entry point for the Kappa2 Distribution tool.

This module provides a command-line entry point for the Kappa2 Distribution tool,
which allows users to calculate and visualize the distribution of the orientation
factor κ² for Förster Resonance Energy Transfer (FRET) experiments.
"""

import sys
from qtpy import QtWidgets
from .k2dgui import Kappa2Dist

def main():
    """
    Main entry point for the Kappa2 Distribution tool.
    Creates and shows the Kappa2Dist widget.
    """
    app = QtWidgets.QApplication(sys.argv)
    win = Kappa2Dist()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()