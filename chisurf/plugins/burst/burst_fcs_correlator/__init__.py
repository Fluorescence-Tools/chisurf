"""Burst-wise FCS Correlator plugin.

This plugin provides a wizard for computing fluorescence correlation
functions on a per-burst basis, using Burst-ID ``.bst`` files as
input and the FCS presets/correlator settings defined elsewhere in
ChiSurf.
"""

from chisurf.gui import QtWidgets, QtCore

from .wizard import BurstWiseFCSWizard


# Plugin category/name for the ChiSurf menu
name = "Single-Molecule:Burst-wise FCS Correlator"


if __name__ == "plugin":  # pragma: no cover
    dlg = BurstWiseFCSWizard()
    dlg.show()

