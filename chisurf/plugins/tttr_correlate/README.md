# TTTR Correlate Plugin

This plugin provides a graphical interface for calculating correlation functions from Time-Tagged Time-Resolved 
(TTTR) data.

## Features

- Support for various correlation types (auto, cross)
- Configurable correlation parameters
- Interactive visualization of correlation curves
- Export of correlation results
- Multiple correlation algorithm options
- Channel selection for specific analysis

## Overview

The TTTR Correlate plugin enables users to calculate correlation functions from Time-Tagged Time-Resolved (TTTR) data. 
Correlation analysis is a fundamental technique in fluorescence correlation spectroscopy (FCS) and related methods, 
providing insights into molecular dynamics, diffusion processes, and interactions.

This plugin implements efficient algorithms for computing auto-correlation and cross-correlation functions from photon 
arrival times in TTTR files. Users can configure correlation parameters such as lag time ranges, binning options, and 
normalization methods to optimize the analysis for specific experimental conditions.

The interactive visualization tools allow for immediate inspection of correlation curves, making it easy to assess data 
quality and extract meaningful parameters. Results can be exported for further analysis or publication.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - tttrlib
  - matplotlib or pyqtgraph (for visualization)
  - chisurf core modules

## Usage

1. Launch the plugin from the ChiSurf menu: TTTR > Correlate
2. Load TTTR data:
   - Click "Open" to select a TTTR file
   - The file information will be displayed
3. Configure correlation settings:
   - Select correlation type (auto-correlation or cross-correlation)
   - Choose channels for correlation
   - Set correlation parameters (lag time range, binning, etc.)
   - Select normalization method
4. Calculate correlation:
   - Click "Calculate" to compute the correlation function
   - The correlation curve will be displayed in the plot area
5. Analyze results:
   - Inspect the correlation curve
   - Adjust display options if needed
   - Export results for further analysis
6. Batch processing (optional):
   - Set up multiple correlation calculations
   - Process them sequentially or in parallel

## Applications

The TTTR Correlate plugin can be used for:
- Analyzing molecular diffusion in solution
- Measuring molecular brightness and concentration
- Investigating binding kinetics and molecular interactions
- Characterizing flow and directed transport
- Studying conformational dynamics of biomolecules
- Analyzing photophysical processes like blinking and bleaching
- Quality control of fluorescence measurements

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.