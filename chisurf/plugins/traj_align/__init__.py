import sys
from chisurf.gui.tools.structure.align_trajectory import AlignTrajectoryWidget

# Define the plugin name - this will appear in the Plugins menu
name = "Structure:Trajectory Alignment"

"""
Trajectory Alignment Tool

This plugin provides tools for aligning molecular dynamics trajectories to 
reference structures or to specific frames within the trajectory.

Features:
- Load trajectories in various formats (PDB, XTC, DCD, etc.)
- Select specific atoms or residues for alignment
- Align to a reference structure or to a specific frame
- Apply RMSD-based alignment algorithms
- Visualize alignment quality through RMSD plots
- Save aligned trajectories for further analysis

Proper alignment is essential for analyzing conformational changes, 
calculating order parameters, and comparing different simulations.
"""

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the AlignTrajectoryWidget class
    window = AlignTrajectoryWidget()
    # Show the window
    window.show()
