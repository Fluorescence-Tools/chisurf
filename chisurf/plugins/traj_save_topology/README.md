# Topology File Generator Plugin

This plugin extracts and saves topology information from the first frame of molecular dynamics trajectories.

## Features

- Extraction of topology from the first frame of a trajectory
- Saving topology as a PDB file
- Simple and intuitive interface

## Overview

The Topology File Generator plugin enables users to extract and save topology information from the first frame of a 
molecular trajectory. Topology files are essential for many molecular dynamics simulations and analysis tools, 
providing the connectivity and parameter information needed to interpret coordinate data correctly.

The plugin provides a simple interface for loading a trajectory file and saving its first frame as a PDB file. 
This is useful for extracting a representative structure from a trajectory for further analysis or visualization.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - mdtraj (for trajectory handling)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Structure > Save Topology
2. Load a trajectory file:
   - Click "..." to select an input H5 trajectory file
   - The file path will appear in the text field
3. Save the topology file:
   - Click "save" to choose a save location
   - The first frame of the trajectory will be saved as a PDB file

## Applications

The Topology File Generator plugin can be used for:
- Extracting a representative structure from a trajectory for visualization
- Saving the first frame of a trajectory as a PDB file for further analysis
- Creating input files for molecular viewers or analysis tools that require PDB format
- Preserving the initial structure of a simulation for reference

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.
