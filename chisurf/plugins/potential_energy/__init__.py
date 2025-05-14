"""
Potential Energy Calculator

This plugin provides a graphical interface for calculating potential energy
components in molecular structures. Features include:

- Calculation of various energy terms (bond, angle, dihedral, non-bonded)
- Support for different force fields and energy functions
- Analysis of energy contributions from specific residues or atoms
- Visualization of energy distributions
- Export of energy data for further analysis

The potential energy calculator is useful for assessing the stability of
molecular structures and identifying strained or unfavorable conformations.
"""

import sys
from chisurf.gui.tools.structure.potential_energy import PotentialEnergyWidget

# Define the plugin name - this will appear in the Plugins menu
name = "Structure:Potential Energy Calculator"



# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the PotentialEnergyWidget class
    window = PotentialEnergyWidget()
    # Show the window
    window.show()