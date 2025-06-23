# Single-Molecule MLE Plugin

This plugin provides a graphical interface to the PTU processing CLI for analyzing single-molecule fluorescence data. 
It enables users to process multiple PTU files with customizable parameters and view the results in an integrated 
browser.

## Features

- Drag-and-drop interface for selecting PTU files
- Configurable processing parameters through a user-friendly GUI
- Real-time progress tracking and log output
- Integrated browser for viewing processed molecule images and decay plots
- Support for batch processing multiple files
- Customizable segmentation parameters for molecule detection
- Lifetime fitting with adjustable parameters

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - matplotlib
  - tttrlib (for PTU file handling)

## Usage

1. Launch the plugin from the ChiSurf menu: Imaging > Single-Molecule MLE
2. Drag and drop PTU files into the file list
3. Select an IRF file using the Browse button
4. Configure processing parameters in the Settings tab
5. Click Process to start the analysis
6. View the results in the Browser tab after processing is complete

## Processing Parameters

The plugin provides a comprehensive set of parameters for PTU processing:

### Basic Parameters
- Detector channels: Specify which detector channels to use
- Micro-time range: Set the range of micro-time channels to analyze
- Micro-time binning: Control the binning of micro-time data
- Normalization options: Choose how to normalize counts

### Fitting Parameters
- Threshold: Set the threshold for molecule detection
- Min length: Specify the minimum histogram length
- Shift parameters: Adjust SP and SS shifts
- IRF threshold fraction: Control IRF processing
- Initial values and fixed flags for fitting

### Segmentation Parameters
- Segmentation σ: Controls Gaussian blur for segmentation
- Segmentation threshold: Sets the threshold for peak detection
- Peak footprint size: Defines the minimum size of detected peaks

## Output

The plugin generates several outputs for each processed PTU file:
- Molecule images showing spatial distribution
- Decay plots with fitted lifetime curves
- Combined TSV file with analysis results
- Log output showing processing details

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.