"""
Filtered FCS Lifetime Filter Calculator

This plugin provides a small GUI tool to calculate filtered-FCS (fFCS) lifetime
filters from microtime decay patterns, similar to the Calc_fFCS_Filters
functionality in PAM. It operates on precomputed decay histograms for several
species and a total decay histogram and computes:

- Species-specific filters
- Reconstructed total decay
- Weighted residuals

The plugin is intentionally file-based and self-contained so that it can be
used independently of other wizards. Histogram files are expected to contain a
single column of counts (one value per TAC/microtime bin).
"""

import sys
from qtpy import QtWidgets

from .widget import FcsFilterCalculatorWidget


# Define the plugin name - this will appear in the Plugins menu
name = "Fluorescence Correlation Spectroscopy:Lifetime Filter Calculator"


# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed by the ChiSurf plugin system.
if __name__ == "plugin":
    window = FcsFilterCalculatorWidget()
    window.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FcsFilterCalculatorWidget()
    win.show()
    sys.exit(app.exec_())
