"""
Trajectory Rotation and Translation Tool

This plugin provides functionality to apply rigid body transformations
(rotations and translations) to molecular dynamics trajectories. Features include:
- Interactive 3D manipulation of structures
- Precise numerical control of rotation angles and translation vectors
- Ability to align structures based on selected atoms or residues
- Support for multiple coordinate reference frames
- Batch processing of multiple trajectory files

These transformations are essential for preparing structures for analysis,
comparing different conformations, or setting up new simulations with
specific molecular orientations.
"""

import sys
from chisurf.gui.tools.structure.rotate_translate_trajectory import RotateTranslateTrajectoryWidget

# Define the plugin name - this will appear in the Plugins menu
name = "Structure:Rotate/Translate Trajectory"



# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the RotateTranslateTrajectoryWidget class
    window = RotateTranslateTrajectoryWidget()
    # Show the window
    window.show()
