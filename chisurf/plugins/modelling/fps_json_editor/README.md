# FPS JSON Editor for Accessible Volume Calculations

This plugin provides a graphical interface for creating and editing FPS (Fast Positioning and Screening) JSON files used in accessible volume (AV) calculations for fluorescent labels.

## Features

- Load and visualize PDB structures
- Define labeling positions on proteins
- Configure AV simulation parameters
- Specify experimental FRET distances between labels
- Save and load FPS JSON configuration files
- Interactive 3D visualization of protein structures
- User-friendly interface for parameter configuration

## Overview

Accessible Volume (AV) calculations are essential for interpreting FRET (Förster Resonance Energy Transfer) data in structural biology. These calculations model the spatial distribution of fluorescent dyes attached to proteins, accounting for the flexible linkers that connect the dyes to specific amino acids.

The FPS JSON Editor plugin simplifies the process of creating input files for the FPS (Fast Positioning and Screening) algorithm, which performs AV calculations and integrates them with structural modeling. By providing a graphical interface for defining labeling positions, simulation parameters, and experimental constraints, the plugin makes advanced structural modeling accessible to researchers without requiring extensive programming knowledge.

The editor is essential for preparing input files for FRET-based structural modeling and accessible volume simulations, bridging the gap between experimental FRET measurements and structural interpretation.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - matplotlib
  - biopython (for PDB handling)
  - mdtraj (for molecular structure processing)
  - json (for file handling)

## Usage

1. Launch the plugin from the ChiSurf menu: Structure > FPS JSON Editor
2. Load a PDB structure:
   - Import from file
   - Fetch from PDB database
3. Define labeling positions:
   - Select amino acids for labeling
   - Specify dye properties (linker length, dye radius)
4. Configure AV simulation parameters:
   - Grid spacing
   - Clash distance
   - Simulation boundaries
5. Add experimental FRET distances:
   - Define donor-acceptor pairs
   - Enter measured distances and uncertainties
6. Save the configuration as a JSON file for use with FPS
7. Visualize the protein structure with labeled positions

## Applications

- Preparation for FRET-based structural modeling
- Integration of FRET data with molecular structures
- Planning of optimal labeling strategies for FRET experiments
- Interpretation of experimental FRET efficiency values
- Validation of structural models against FRET measurements
- Ensemble modeling of flexible protein regions

## Benefits

- Simplifies the creation of complex JSON configuration files
- Provides visual feedback for labeling positions
- Ensures correct formatting for FPS input files
- Streamlines the workflow from experimental design to structural modeling
- Reduces errors in parameter specification
- Makes advanced structural modeling techniques accessible to non-specialists

## Related Resources

- FPS (Fast Positioning and Screening) software
- Accessible Volume (AV) theory and applications
- FRET-based structural modeling approaches
- Integrative structural biology methods

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.