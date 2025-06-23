# Microtime Histogram Plugin

This plugin provides tools for creating, visualizing, and analyzing microtime histograms from Time-Tagged Time-Resolved 
(TTTR) fluorescence data.

## Features

- Loading and processing of TTTR files from various formats
- Creation of microtime histograms for parallel and perpendicular detection channels
- Support for burst selection using BID/BUR files for single-molecule analysis
- Cumulative histogram generation from multiple files
- Export of histogram data for further analysis
- Integration with ChiSurf for advanced data processing

## Overview

Microtime histograms represent the distribution of photon arrival times relative to the excitation pulse, providing 
valuable information about fluorescence lifetimes and molecular dynamics. In polarization-resolved measurements, 
separate histograms for parallel and perpendicular detection channels enable fluorescence anisotropy analysis.

The plugin is particularly useful for time-resolved fluorescence spectroscopy, fluorescence lifetime imaging (FLIM), 
and single-molecule experiments where temporal information about photon arrival is critical for understanding molecular 
properties and dynamics.

## Requirements

- Python packages:
  - numpy
  - matplotlib
  - tttrlib (for TTTR file handling)
  - PyQt5 (for GUI components)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Histogram-Microtime
2. Load TTTR data files
3. Configure histogram parameters:
   - Select detection channels
   - Set microtime range
   - Choose binning options
4. Generate and visualize microtime histograms
5. Export data for further analysis or fitting

## Applications

- Fluorescence lifetime analysis
- Time-resolved fluorescence spectroscopy
- Fluorescence anisotropy measurements
- Single-molecule FRET experiments
- Fluorescence lifetime imaging (FLIM)
- Molecular dynamics studies

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.