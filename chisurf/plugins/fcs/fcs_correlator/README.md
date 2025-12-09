# FCS Correlator Plugin

This plugin provides a wizard interface for correlating and merging fluorescence correlation spectroscopy (FCS) data.

## Features

- Selection and filtering of TTTR (Time-Tagged Time-Resolved) photon data
- Calculation of correlation functions with configurable parameters
- Merging of multiple correlation curves for improved signal-to-noise ratio
- Export of correlation results for further analysis

## Overview

Fluorescence Correlation Spectroscopy (FCS) is a powerful technique for studying molecular dynamics, diffusion, and interactions at the single-molecule level. This plugin provides the essential tools for processing raw photon data into correlation curves that can be analyzed to extract parameters such as diffusion coefficients, concentrations, and binding kinetics.

The correlator is essential for analyzing molecular dynamics and diffusion processes in fluorescence correlation spectroscopy experiments.

## Requirements

- Python packages:
  - numpy
  - tttrlib (for TTTR file handling)
  - PyQt5 (for GUI components)
  - matplotlib (for plotting)

## Usage

1. Launch the plugin from the ChiSurf menu: FCS > Correlator
2. Load TTTR data files
3. Configure correlation parameters:
   - Select detection channels
   - Set correlation time range
   - Choose binning and normalization options
4. Calculate correlation functions
5. Merge multiple correlation curves if needed
6. Export results for further analysis or fitting

## Applications

- Measuring diffusion coefficients of molecules in solution
- Determining molecular concentrations
- Analyzing binding kinetics and molecular interactions
- Characterizing molecular size and shape
- Studying conformational dynamics of biomolecules
- Investigating cellular processes at the molecular level

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.