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

name = "Tools:QuEst (Quenching estimator)"

import sys

import chisurf
from quest.lib.tools.dye_diffusion import TransientDecayGenerator

from PyQt5.QtWidgets import *

log = chisurf.logging.info


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ndx = TransientDecayGenerator()
    ndx.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    ndx = TransientDecayGenerator()
    ndx.show()
