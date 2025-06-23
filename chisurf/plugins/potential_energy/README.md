# Potential Energy Calculator Plugin

This plugin provides a graphical interface for calculating potential energy components in molecular structures.

## Features

- Calculation of various energy terms including:
  - H-Bond potential
  - AV-Potential (Accessible Volume)
  - Iso-UNRES potential
  - Miyazawa-Jernigan potential
  - Go-Potential
  - ASA-Calpha (Accessible Surface Area)
  - Radius of Gyration
  - Clash potential
- Export of energy data for further analysis
- Processing of molecular dynamics trajectories
- Customizable potential energy terms with adjustable weights

## Overview

The Potential Energy Calculator is a tool for assessing the stability of molecular structures and identifying strained 
or unfavorable conformations. It allows researchers to analyze the energy landscape of molecules and molecular dynamics 
trajectories, providing insights into structural stability and conformational changes.

The plugin supports multiple potential energy terms that can be combined with custom weights to create tailored energy 
functions for specific analysis needs. Results can be exported to CSV files for further processing or visualization in 
external software.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - mdtraj (for trajectory handling)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Structure > Potential Energy Calculator
2. Load a molecular structure or trajectory:
   - Click "Open trajectory" to load an H5 trajectory file
3. Configure potential energy calculations:
   - Select potential energy terms from the dropdown menu
   - Adjust parameters for each potential term
   - Click "Add potential" to include the term in the calculation
4. Process the trajectory:
   - Set the stride value to control frame sampling
   - Click "Process trajectory" to calculate energies for all frames
   - Save results to a CSV file for further analysis
5. View and analyze results:
   - Examine energy values for each frame
   - Identify high-energy conformations or interactions

## Applications

- Evaluating the stability of protein structures
- Analyzing conformational changes in molecular dynamics simulations
- Identifying strained or unfavorable molecular geometries
- Studying energy distributions in molecular ensembles
- Validating structural models through energy assessment
- Educational tool for understanding molecular energetics

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.
