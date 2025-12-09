"""
HomoFRET Calculator

This plugin provides an anisotropy-based HomoFRET calculator to estimate
homo-FRET exchange rate (k_homo) and an effective donor-acceptor distance
using t_RM (anisotropy relaxation time) and ρ (rotational correlation time).

Inputs:
- τ0 (donor lifetime without FRET) [ns]
- R0 (Förster radius) [same units as desired output distance]
- t_RM (anisotropy relaxation time) [s]
- ρ (rotational correlation time, no homoFRET) [s]

Outputs:
- k_homo [1/s]
- R_DA (from homoFRET)
"""

import sys
from qtpy import QtWidgets
from .hydrogui import HydroGui

# Plugin display name
name = "Tools:HydroPro"

if __name__ == "plugin":
    window = HydroGui()
    window.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = HydroGui()
    win.show()
    sys.exit(app.exec_())
