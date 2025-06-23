# ndXplorer Plugin

This plugin provides a powerful interface for analyzing and visualizing multidimensional fluorescence data within 
ChiSurf.

## Features

- Burst analysis for single-molecule fluorescence experiments
- Multiparameter fluorescence detection (MFD) analysis
- Interactive selection and filtering of burst events
- Visualization of multidimensional data through histograms and plots
- Support for FRET efficiency calculations and proximity ratio analysis
- Application to both solution-based measurements and image spectroscopy data

## Overview

The ndXplorer tool is particularly useful for analyzing complex fluorescence datasets where multiple parameters need 
to be correlated, such as fluorescence intensity, lifetime, anisotropy, and spectral information. It provides an 
intuitive interface for exploring relationships between different fluorescence parameters.

For single-molecule experiments, ndXplorer enables detailed burst analysis with capabilities to select, filter, and 
categorize individual molecule detection events based on multiple criteria. The tool also supports advanced FRET 
analysis with various correction factors and calculation methods.

When working with image spectroscopy data, ndXplorer allows pixel-by-pixel analysis of multiparameter fluorescence 
information, enabling spatial correlation of spectroscopic properties.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - matplotlib
  - ndxplorer (core functionality)
  - tttrlib (for TTTR file handling)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > ndXplorer
2. Load data:
   - Import TTTR files for single-molecule analysis
   - Load image data for spatial analysis
3. Configure analysis parameters:
   - Set burst selection criteria
   - Define parameter calculations
   - Configure visualization options
4. Explore data through interactive visualizations:
   - Create histograms and scatter plots
   - Apply filters and selections
   - Analyze correlations between parameters
5. Perform specialized analyses:
   - FRET efficiency calculations
   - Lifetime analysis
   - Anisotropy measurements
6. Export results for further analysis or publication

## Applications

- Single-molecule FRET studies
- Protein conformational dynamics
- Molecular interaction analysis
- Fluorescence lifetime imaging
- Multiparameter fluorescence detection
- Correlation of multiple fluorescence parameters
- Spatial mapping of spectroscopic properties

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.