# Trajectory Clash Removal Plugin

This plugin identifies and removes frames from molecular dynamics trajectories that contain steric clashes or 
other structural problems.

## Features

- Detection of atom-atom clashes based on van der Waals radii
- Customizable clash criteria and thresholds
- Ability to focus on specific regions or residues
- Options to repair clashed frames or remove them entirely
- Statistical reporting on the number and types of clashes
- Visualization of clash locations in 3D structures
- Batch processing capabilities

## Overview

The Trajectory Clash Removal plugin enables users to identify and remove frames from molecular dynamics trajectories 
that contain steric clashes or other structural problems. Removing clashed frames is important for preparing clean 
trajectories for further analysis, especially when working with modeled structures or trajectories from enhanced 
sampling methods.

The plugin provides a comprehensive interface for detecting clashes based on atom-atom distances and van der Waals 
radii. Users can customize the clash criteria and thresholds to suit their specific needs, and can focus the analysis 
on particular regions or residues of interest. When clashes are detected, users have the option to either repair the 
clashed frames using energy minimization or other methods, or remove them entirely from the trajectory.

The statistical reporting features provide insights into the number and types of clashes present in the trajectory, 
helping users understand the nature of the structural problems. This information can be valuable for improving modeling 
or simulation protocols in future work.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - mdtraj (for trajectory handling)
  - matplotlib (for visualization)
  - scipy (for clash detection algorithms)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Structure > Remove Clashed Frames
2. Load a trajectory file:
   - Click "Browse" to select an input trajectory file
   - The file format will be automatically detected
3. Configure clash detection settings:
   - Set distance thresholds for clash detection
   - Choose atom types to include in the analysis
   - Select specific regions or residues to analyze (optional)
   - Configure clash criteria (number of clashes, severity, etc.)
4. Run the clash detection:
   - Click "Detect Clashes" to analyze the trajectory
   - View the results in the clash report panel
5. Handle clashed frames:
   - Choose to repair or remove clashed frames
   - If repairing, configure the repair method
   - If removing, select which frames to exclude
6. Save the processed trajectory:
   - Choose a directory and filename for the output
   - Select the desired output format
   - Click "Save" to write the cleaned trajectory to disk

## Applications

The Trajectory Clash Removal plugin can be used for:
- Cleaning trajectories from homology modeling or structure prediction
- Removing artifacts from enhanced sampling simulations
- Preparing trajectories for docking or other structure-based analyses
- Identifying problematic regions in protein models
- Quality control of molecular dynamics simulations
- Improving the reliability of structural analysis by removing physically unrealistic conformations
- Educational demonstrations of molecular structure validation
- Preprocessing trajectories for visualization or presentation

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.