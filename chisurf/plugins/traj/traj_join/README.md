# Trajectory Joining Plugin

This plugin provides functionality to combine two molecular dynamics trajectory files into a single continuous trajectory.

## Features

- Support for joining two H5 trajectory files
- Two join modes: 'time' (concatenate frames) and 'atoms' (stack atoms)
- Options to reverse the order of either trajectory
- Chunk size setting for memory management

## Overview

The Trajectory Joining plugin enables users to combine two molecular trajectory files into a single 
continuous trajectory. This functionality is useful for analyzing simulations that were run in multiple segments or 
for combining related simulations into a single dataset.

The plugin provides a simple interface for loading two trajectory files, configuring how they should be joined, and 
saving the resulting combined trajectory. Users can choose between two joining modes: joining along the time axis 
(concatenating frames) or joining along the atom axis (stacking atoms). Additionally, users can reverse the order of 
either trajectory if needed.

By combining trajectory segments, users can perform analyses that span longer timescales or compare different 
simulation conditions within a unified framework.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - mdtraj (for trajectory handling)
  - tables (for HDF5 file handling)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Structure > Join Trajectories
2. Load trajectory files:
   - Click the "..." button next to "Trajectory 1" to select the first input trajectory file
   - Click the "..." button next to "Trajectory 2" to select the second input trajectory file
   - Both files must be in H5 format
3. Configure joining options:
   - Select the join mode: "time" (concatenate frames) or "atoms" (stack atoms)
   - Check the box next to a trajectory to reverse its order (optional)
   - Set the chunk size for memory management
4. Perform the joining:
   - Click "join" to begin the process
   - Select a save location for the joined trajectory
   - The joined trajectory will be saved in H5 format

## Applications

The Trajectory Joining plugin can be used for:
- Combining multiple simulation segments into a single continuous trajectory
- Merging trajectories from replica exchange or parallel tempering simulations
- Creating longer trajectories for improved statistical sampling
- Comparing different simulation conditions within a unified analysis framework
- Preparing trajectories for analyses that require long timescales
- Consolidating related simulations for simplified data management
- Creating custom trajectory ensembles from selected simulation segments

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.
