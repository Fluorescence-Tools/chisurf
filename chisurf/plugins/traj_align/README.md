# Trajectory Alignment Plugin

This plugin provides tools for aligning molecular dynamics trajectories to the first frame of the trajectory.

## Features

- Load trajectories in H5 format
- Select specific atoms for alignment
- Align to the first frame of the trajectory as reference
- Apply RMSD-based alignment algorithms
- Save aligned trajectories for further analysis

## Overview

The Trajectory Alignment plugin enables users to align molecular dynamics trajectories to the first frame of the 
trajectory. Proper alignment is essential for analyzing conformational changes, calculating order parameters, and 
comparing different simulations.

The plugin provides a user-friendly interface for loading trajectories and selecting alignment criteria. Users 
can choose specific atoms to use for alignment by providing a comma-separated list of atom IDs, which is particularly 
useful when focusing on specific structural elements while allowing other regions to move freely. The RMSD-based 
alignment algorithms ensure optimal superposition of the selected structural elements.

The aligned trajectories can be saved for further analysis, ensuring that all subsequent calculations are performed 
in a consistent reference frame.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - mdtraj (for trajectory handling)
  - tables (for HDF5 file handling)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Structure > Trajectory Alignment
2. Load a trajectory file:
   - Click the "..." button to select an input H5 trajectory file
   - The file path will appear in the text field
3. Specify atoms for alignment:
   - Enter a comma-separated list of atom IDs in the "Atom-selection" text area
   - These atoms will be used for calculating the alignment
4. Set the stride value:
   - Adjust the "stride" value to control frame sampling (higher values process fewer frames)
5. Save the aligned trajectory:
   - Click the "save" button to choose a save location
   - The aligned trajectory will be saved in H5 format

## Applications

The Trajectory Alignment plugin can be used for:
- Removing overall rotational and translational motion from trajectories
- Focusing analysis on specific structural changes by aligning stable regions
- Preparing trajectories for principal component analysis or other conformational analyses
- Creating a consistent reference frame for trajectory analysis
- Improving visualization of molecular dynamics by removing global motions
- Preparing aligned trajectories for further analysis in ChiSurf or other software

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.
