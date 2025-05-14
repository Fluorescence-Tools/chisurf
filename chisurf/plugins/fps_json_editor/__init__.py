"""
FPS JSON Editor for Accessible Volume Calculations

This plugin provides a graphical interface for creating and editing FPS (Fast
Positioning and Screening) JSON files used in accessible volume (AV) calculations
for fluorescent labels.

Features:
- Load and visualize PDB structures
- Define labeling positions on proteins
- Configure AV simulation parameters
- Specify experimental FRET distances between labels
- Save and load FPS JSON configuration files

The editor is essential for preparing input files for FRET-based structural modeling
and accessible volume simulations.
"""

import sys
from chisurf.gui.tools.structure.create_av_json.label_structure import LabelStructure

# Define the plugin name - this will appear in the Plugins menu
name = "Structure:FPS JSON Editor"



# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the LabelStructure class
    window = LabelStructure()
    # Show the window
    window.show()
