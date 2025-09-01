# Trajectory Rotation and Translation Plugin

This plugin provides functionality to apply rigid body transformations (rotations and translations) to molecular 
dynamics trajectories.

## Features

- Interactive 3D manipulation of structures
- Precise numerical control of rotation angles and translation vectors
- Ability to align structures based on selected atoms or residues
- Support for multiple coordinate reference frames
- Batch processing of multiple trajectory files
- Visualization of transformations in real-time
- Support for various trajectory file formats

## Overview

The Trajectory Rotation and Translation plugin enables users to apply rigid body transformations (rotations and 
translations) to molecular dynamics trajectories. These transformations are essential for preparing structures for 
analysis, comparing different conformations, or setting up new simulations with specific molecular orientations.

The plugin provides both interactive 3D manipulation tools and precise numerical controls for applying rotations and 
translations to molecular structures. Users can rotate structures around specific axes, translate them in any direction, 
or align them based on selected atoms or residues. The plugin supports multiple coordinate reference frames, 
allowing for transformations relative to different parts of the structure or external reference points.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - mdtraj (for trajectory handling)
  - matplotlib or pyqtgraph (for visualization)
  - scipy (for transformation calculations)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Structure > Rotate/Translate Trajectory
2. Load a trajectory file:
   - Click "Browse" to select an input trajectory file
   - The file format will be automatically detected
3. Select transformation mode:
   - Choose between interactive manipulation or numerical input
   - Select the reference frame for transformations
4. Apply transformations:
   - If using interactive mode:
     - Use the 3D controls to rotate and translate the structure
     - Fine-tune the orientation using the adjustment sliders
   - If using numerical input:
     - Enter rotation angles (in degrees) for each axis
     - Enter translation distances (in Angstroms) for each direction
   - If aligning structures:
     - Select atoms or residues to use for alignment
     - Choose the target structure or orientation
5. Preview the transformation:
   - View the transformed structure in the 3D viewer
   - Compare with the original structure if needed
6. Save the transformed trajectory:
   - Choose a directory and filename for the output
   - Select the desired output format
   - Click "Save" to write the transformed trajectory to disk

## Applications

The Trajectory Rotation and Translation plugin can be used for:
- Preparing structures for docking or binding site analysis
- Aligning multiple structures for comparison
- Setting up initial configurations for molecular dynamics simulations
- Orienting molecules for optimal visualization or presentation
- Standardizing the orientation of structures for consistent analysis
- Creating symmetry-related copies of molecular structures
- Educational demonstrations of molecular geometry and symmetry
- Preparing structures for specific analytical techniques that require particular orientations

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.