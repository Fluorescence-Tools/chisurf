import sys
from chisurf.gui.tools.structure.join_trajectories import JoinTrajectoriesWidget

# Define the plugin name - this will appear in the Plugins menu
name = "Structure:Join Trajectories"

"""
Trajectory Joining Tool

This plugin provides functionality to combine multiple molecular dynamics 
trajectory files into a single continuous trajectory. Features include:
- Support for multiple input trajectory formats
- Options to align frames between trajectories
- Ability to remove duplicate frames at trajectory boundaries
- Support for concatenating trajectories with different topologies
- Options to filter or select specific parts of each trajectory

Joining trajectories is useful for analyzing simulations that were run in 
multiple segments or for combining related simulations into a single dataset.
"""

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the JoinTrajectoriesWidget class
    window = JoinTrajectoriesWidget()
    # Show the window
    window.show()
