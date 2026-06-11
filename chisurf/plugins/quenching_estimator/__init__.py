"""
Quenching Estimator Plugin (QuEst)

This plugin is part of the ChiSurf application and provides tools for simulating 
fluorescence quenching processes in macromolecules.

The plugin implements a diffusion simulation approach to analyze time-resolved FRET 
measurements of labeled macromolecules. It simulates the diffusion of fluorescent dyes 
around a macromolecule and calculates quenching effects based on the proximity to 
quencher residues (such as Tryptophan, Tyrosine, and Histidine).

Key features:
- Simulation of dye diffusion using accessible volume (AV) calculations
- Calculation of fluorescence quenching based on proximity to quencher residues
- Generation of fluorescence decay histograms
- Visualization of diffusion trajectories and protein structures

The diffusion simulation methodology is based on the approach described in:
Peulen, T. O., Opanasyuk, O., & Seidel, C. A. M. (2017). 
"Combining Graphical and Analytical Methods with Molecular Simulations To Analyze 
Time-Resolved FRET Measurements of Labeled Macromolecules Accurately." 
The Journal of Physical Chemistry B, 121(35), 8211-8241.
https://pubs.acs.org/doi/10.1021/acs.jpcb.7b03441

The plugin provides a graphical interface for setting up and running these simulations,
as well as for analyzing and visualizing the results.
"""

name = "Structure:Computation:QuEst"

import sys

import chisurf as cs
from quest.lib.tools.dye_diffusion import TransientDecayGenerator

from qtpy import QtWidgets

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

log = cs.logging.info


@persist_plugin_state("quenching_estimator")
class QuEstWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dg = TransientDecayGenerator()
        self.setCentralWidget(self.dg)
        self._create_menus()
        try:
            self.resize(1000, 600)
        except Exception:
            pass

    def _create_menus(self):
        mbar = self.menuBar()
        file_menu = mbar.addMenu("&File")

        load_pdb_action = file_menu.addAction("Load PDB…")
        load_pdb_action.triggered.connect(self.dg.onLoadPDB)

        file_menu.addSeparator()
        save_project_action = file_menu.addAction("Save project…")
        save_project_action.triggered.connect(self.dg.onSaveProject)
        load_project_action = file_menu.addAction("Load project…")
        load_project_action.triggered.connect(self.dg.onLoadProject)

        file_menu.addSeparator()
        close_action = file_menu.addAction("Close")
        close_action.triggered.connect(self.close)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = QuEstWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "plugin":
    win = QuEstWindow()
    win.show()
