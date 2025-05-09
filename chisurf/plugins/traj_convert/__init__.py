import sys
from chisurf.gui.tools.structure.convert_trajectory import MDConverter

# Define the plugin name - this will appear in the Plugins menu
name = "Structure:Trajectory Converter"

"""
Trajectory Converter for Molecular Dynamics Data

This plugin provides tools for converting molecular dynamics trajectory files 
between different formats. Features include:
- Support for multiple input formats (PDB, XTC, DCD, TRR, etc.)
- Support for multiple output formats
- Options to select specific frames or time ranges
- Ability to filter atoms or residues
- Support for topology conversion

The converter is essential for working with trajectories from different 
simulation packages or for preparing data for specific analysis tools.
"""

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the MDConverter class
    window = MDConverter()
    # Show the window
    window.show()
