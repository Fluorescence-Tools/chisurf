import sys
from chisurf.gui.tools.structure.fret_trajectory.gui import Structure2Transfer

# Define the plugin name - this will appear in the Plugins menu
name = "Structure:FRET from Trajectory"

"""
FRET Analysis from Molecular Dynamics Trajectories

This plugin calculates FRET efficiency values from molecular dynamics (MD) 
trajectories by analyzing the distances between specified residues over time.

Features:
- Load and analyze MD trajectories in various formats
- Select donor and acceptor positions for FRET calculations
- Calculate inter-residue distances and corresponding FRET efficiencies
- Apply accessible volume (AV) models to account for dye flexibility
- Generate FRET efficiency histograms and time traces
- Export results for further analysis

Ideal for comparing experimental FRET data with structural models from 
molecular dynamics simulations.
"""

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the Structure2Transfer class
    window = Structure2Transfer()
    # Show the window
    window.show()
