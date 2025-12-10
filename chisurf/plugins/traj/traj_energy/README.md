# Trajectory Energy Analysis Plugin

This plugin calculates and analyzes energy components from molecular dynamics trajectories.

## Features

- Calculation of various energy terms (bond, angle, dihedral, non-bonded)
- Support for different force fields and energy functions
- Export of energy data as time series for external analysis
- Support for trajectory frame selection via stride parameter
- Batch processing capabilities

## Overview

The Trajectory Energy Analysis plugin enables users to calculate and analyze energy components from molecular 
trajectories. Energy analysis is crucial for assessing the stability of simulations and identifying potential issues 
in molecular models or simulation parameters.

The plugin provides an interface for loading trajectories and selecting energy terms to calculate. Users can calculate 
various energy components including bonded interactions (bonds, angles, dihedrals) and non-bonded interactions 
(electrostatics, van der Waals) through different potential types. The plugin supports different force fields and 
energy functions, allowing for flexibility in the analysis approach.

The plugin exports energy data to a CSV file, which can be used for further analysis in external tools. Users can 
control which frames are processed using the stride parameter, making it possible to analyze specific portions of a 
trajectory or reduce processing time for large trajectories.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - mdtraj (for trajectory handling)
  - tables (for HDF5 file handling)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Structure > Trajectory Energy Calculator
2. Load a trajectory file:
   - Click "..." to select an input H5 trajectory file
   - The file path will appear in the text field
3. Configure energy calculation settings:
   - Select energy terms from the dropdown menu
   - Click "add" to include the term in the calculation
   - Adjust parameters for each potential term as needed
   - Set the stride value to control frame sampling
4. Perform the energy calculation:
   - Click "process" to begin the calculation
   - Select a location to save the CSV file with energy data
   - The calculation will process each frame and write energy values to the CSV file
5. Analyze the results:
   - Open the CSV file in external software for further analysis
   - The file contains energy values for each frame and potential term

## Applications

The Trajectory Energy Analysis plugin can be used for:
- Validating the stability of molecular dynamics simulations
- Identifying frames with high-energy conformations or clashes
- Analyzing the contribution of different energy terms to overall stability
- Comparing energy profiles between different simulations or systems
- Investigating the energetic effects of mutations or ligand binding
- Identifying potential issues in force field parameters
- Educational demonstrations of molecular energetics
- Quality control of simulation outputs

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.
