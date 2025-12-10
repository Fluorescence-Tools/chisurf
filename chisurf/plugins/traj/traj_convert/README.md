# Trajectory Converter Plugin

This plugin provides tools for converting molecular dynamics trajectory files between different formats.

## Features

- Support for multiple output formats
- Options to select specific frames or time ranges
- Ability to filter atoms or residues
- Support for topology conversion
- Batch processing capabilities

## Overview

The Trajectory Converter plugin enables users to convert molecular dynamics trajectory files between different 
formats. This functionality is essential for working with trajectories from different simulation packages or for 
preparing data for specific analysis tools.

The plugin provides a user-friendly interface for selecting input files, specifying output formats, and configuring 
conversion parameters. Users can select specific frames or time ranges to extract from larger trajectories, filter 
specific atoms or residues of interest, and ensure that topology information is properly preserved during conversion.

By supporting a wide range of file formats, the plugin serves as a bridge between different molecular dynamics software 
packages, allowing researchers to leverage the strengths of multiple tools in their analysis workflows.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - mdtraj (for trajectory handling)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: Structure > Trajectory Converter
2. Load a trajectory file:
   - Click "Browse" to select an input trajectory file
   - The file format will be automatically detected
3. Configure conversion settings:
   - Select the desired output format
   - Choose frames to include (all, range, or specific frames)
   - Select atoms or residues to include (optional)
   - Specify topology handling options
4. Set the output location:
   - Choose a directory and filename for the converted trajectory
5. Start the conversion:
   - Click "Convert" to begin the process
   - A progress bar will show the status of the conversion
   - The converted file will be saved to the specified location

## Applications

The Trajectory Converter plugin can be used for:
- Converting trajectories between different molecular dynamics software formats
- Extracting specific portions of trajectories for focused analysis
- Preparing trajectories for visualization in different software packages
- Reducing file sizes by selecting only relevant atoms or frames
- Standardizing trajectory formats across a research project
- Creating input files for specialized analysis tools
- Archiving trajectories in preferred formats

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.