# TTTR Histogram Plugin

This plugin provides a graphical interface for generating fluorescence decay histograms from Time-Tagged Time-Resolved 
(TTTR) data.

## Features

- Configurable histogram parameters (binning, time range)
- Channel selection for multi-channel TTTR data
- Interactive visualization of decay curves
- Export of histogram data for further analysis
- Support for various TTTR file formats
- Background subtraction capabilities
- Logarithmic and linear display options

## Overview

The TTTR Histogram plugin enables users to generate fluorescence decay histograms from Time-Tagged Time-Resolved (TTTR) 
data. These histograms represent the distribution of photon arrival times relative to the excitation pulse, which is 
essential for analyzing fluorescence lifetimes and time-resolved spectroscopy.

The plugin provides a flexible interface for configuring histogram parameters such as bin width, time range, and channel 
selection. Users can visualize the resulting decay curves in real-time and adjust parameters to optimize the 
representation of their data. The interactive display allows for easy assessment of data quality and extraction of 
key features.

Generated histograms can be exported for further analysis, such as fitting with exponential decay models to extract 
fluorescence lifetimes. This plugin serves as a critical bridge between raw TTTR data acquisition and advanced 
fluorescence lifetime analysis.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - tttrlib
  - matplotlib or pyqtgraph (for visualization)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: TTTR > Generate Decay
2. Load TTTR data:
   - Click "Open" to select a TTTR file
   - The file information will be displayed
3. Configure histogram settings:
   - Select detection channels to include
   - Set the time range for the histogram
   - Adjust the bin width or number of bins
   - Configure any additional parameters (background subtraction, etc.)
4. Generate the histogram:
   - Click "Generate" to create the decay histogram
   - The decay curve will be displayed in the plot area
5. Analyze and export:
   - Inspect the decay curve
   - Toggle between linear and logarithmic display if needed
   - Export the histogram data for further analysis
   - Save the plot as an image if desired

## Applications

The TTTR Histogram plugin can be used for:
- Fluorescence lifetime analysis
- Time-resolved fluorescence spectroscopy
- Quality assessment of TTTR data
- Preparation of data for lifetime fitting
- Investigating photophysical processes
- Characterizing instrument response functions
- Educational demonstrations of time-resolved fluorescence
- Comparing decay profiles across different experimental conditions

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.