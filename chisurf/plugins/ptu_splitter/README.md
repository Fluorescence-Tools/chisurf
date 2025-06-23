# PTU Splitter Plugin

This plugin provides functionality for splitting large Time-Tagged Time-Resolved (TTTR) files, particularly those 
in the PicoQuant PTU format, into smaller segments.

## Features

- Support for various TTTR file formats with automatic detection
- Configurable splitting parameters (photons per file)
- Options for micro-time binning to reduce file size
- Ability to reset macro-times in the output files
- Selection of output container formats (file format conversion)
- Drag-and-drop interface for easy file loading
- Batch processing capabilities for multiple files

## Overview

The PTU Splitter plugin enables users to break down large TTTR datasets into smaller, more manageable segments. This 
is particularly valuable for handling data from long-duration single-molecule or imaging experiments, where file 
sizes can become unwieldy for analysis software.

The plugin provides an intuitive interface for configuring how files should be split, with options to specify the 
number of photons per output file, apply micro-time binning for size reduction, reset macro-times for easier 
analysis, and even convert between different TTTR file formats during the splitting process.

By splitting large files into smaller segments, users can reduce memory requirements for analysis, enable parallel 
processing of data chunks, extract specific time segments of interest, and generally make their data more 
manageable for subsequent processing steps.

## Requirements

- Python packages:
  - PyQt5/QtPy
  - tttrlib
  - numpy
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: TTTR > PTU-Splitter
2. Load a TTTR file:
   - Click "Browse..." to select an input file
   - Or drag and drop a file into the input field
   - The file format will be automatically detected
3. Configure splitting parameters:
   - Set the number of photons per output file
   - Choose whether to reset macro-times in output files
   - Select the output container format (PTU, HT3, etc.)
   - Configure micro-time binning options if desired
4. Select an output folder:
   - Click "Browse..." to choose a destination folder
   - Or drag and drop a folder into the output field
5. Start the splitting process:
   - Click "Split File" to begin processing
   - A progress indicator will show the status of the operation
   - Output files will be created in the specified folder with sequential numbering

## Applications

The PTU Splitter plugin can be used for:
- Breaking down large datasets into manageable chunks for analysis
- Extracting specific time segments from long measurements
- Creating subsets of data for parallel processing
- Reducing memory requirements for analysis software
- Converting between different TTTR file formats
- Preparing data for specialized analysis pipelines
- Archiving or sharing smaller portions of large datasets

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.