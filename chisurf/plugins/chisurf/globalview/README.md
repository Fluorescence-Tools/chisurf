# Global View Plugin

This plugin provides a graphical interface for visualizing and managing parameter relationships in fitting models 
through an interactive network graph representation.

## Features

- Visualization of fits and parameters as nodes in a network graph
- Interactive linking of parameters across different fits for global analysis
- Multiple graph layout algorithms for optimal visualization
- Selection and editing of parameter properties directly from the graph
- Saving and loading of graph configurations for reproducible analysis
- Color-coded nodes to distinguish between fits, fixed parameters, linked parameters, and free parameters

## Overview

The Global View plugin is particularly useful for global fitting analysis, where parameters need to be shared or 
linked across multiple datasets. It provides an intuitive visual way to create and manage these relationships, making 
complex fitting scenarios more manageable.

Users can select nodes to view and edit parameter properties, link parameters by selecting source and target nodes, 
and adjust the graph's appearance to better visualize the relationships between fits and parameters.

Global fitting is a powerful approach in data analysis where multiple datasets are analyzed simultaneously with shared 
parameters. This approach can significantly improve parameter estimation by leveraging information across different 
experimental conditions or measurements.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - networkx (for graph representation)
  - matplotlib (for visualization)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Global View
2. Load existing fits or create new fits in ChiSurf
3. Visualize the parameter relationships:
   - Fits appear as larger nodes
   - Parameters appear as smaller nodes connected to their respective fits
4. Create parameter links:
   - Select a source parameter node
   - Select a target parameter node
   - Confirm the link creation
5. Edit parameter properties:
   - Select a parameter node
   - Modify properties (fixed/free, bounds, value)
6. Adjust graph layout:
   - Choose from different layout algorithms
   - Manually adjust node positions if needed
7. Save the graph configuration for future use

## Applications

- Global analysis of multiple datasets
- Sharing parameters across different experimental conditions
- Creating complex parameter relationships for sophisticated models
- Visualizing the structure of fitting models
- Documenting parameter relationships for reproducible analysis
- Teaching and explaining global fitting concepts

## Benefits

- Intuitive visual representation of parameter relationships
- Simplified management of complex global fitting scenarios
- Reduced errors in parameter linking
- Improved understanding of model structure
- Enhanced reproducibility through saved configurations
- Streamlined workflow for global analysis

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.