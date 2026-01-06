"""
Trajectory Energy Analysis Tool

This plugin calculates and analyzes energy components from molecular dynamics
trajectories. Features include:
- Calculation of various energy terms (bond, angle, dihedral, non-bonded)
- Support for different force fields and energy functions
- Time series analysis of energy components
- Statistical analysis of energy distributions
- Visualization of energy profiles over the trajectory

Energy analysis is crucial for assessing the stability of simulations and
identifying potential issues in molecular models or simulation parameters.
"""

import sys
from chisurf.plugins.traj.potential_energy.widget import PotentialEnergyWidget

# Define the plugin name - this will appear in the Plugins menu
name = "Structure:Trajectory:Trajectory Energy"



# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the PotentialEnergyWidget class
    window = PotentialEnergyWidget()
    # Show the window
    window.show()
