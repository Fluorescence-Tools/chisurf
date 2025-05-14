"""
Global View

This plugin provides a graphical interface for visualizing and managing parameter relationships 
in fitting models through an interactive network graph representation.

Features:
- Visualization of fits and parameters as nodes in a network graph
- Interactive linking of parameters across different fits for global analysis
- Multiple graph layout algorithms for optimal visualization
- Selection and editing of parameter properties directly from the graph
- Saving and loading of graph configurations for reproducible analysis
- Color-coded nodes to distinguish between fits, fixed parameters, linked parameters, and free parameters

The Global View plugin is particularly useful for global fitting analysis, where parameters 
need to be shared or linked across multiple datasets. It provides an intuitive visual way to 
create and manage these relationships, making complex fitting scenarios more manageable.

Users can select nodes to view and edit parameter properties, link parameters by selecting 
source and target nodes, and adjust the graph's appearance to better visualize the 
relationships between fits and parameters.
"""

name = "Tools:Global View"
