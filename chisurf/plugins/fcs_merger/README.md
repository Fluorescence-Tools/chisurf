# FCS Merger Plugin

This plugin provides a graphical interface for merging multiple fluorescence correlation spectroscopy (FCS) curves.

## Features

- Loading and visualization of multiple correlation curves
- Averaging of correlation data with proper weighting
- Statistical analysis of merged data
- Export of merged correlation curves for further analysis
- Interactive visualization of individual and merged curves
- Customizable merging parameters and weights

## Overview

Fluorescence Correlation Spectroscopy (FCS) is a powerful technique for studying molecular dynamics, diffusion, and interactions at the single-molecule level. However, individual FCS measurements can sometimes suffer from noise or artifacts that affect data quality.

The FCS Merger plugin addresses this challenge by allowing users to combine multiple correlation curves from repeated measurements of the same sample. By properly averaging these curves, the plugin helps improve the signal-to-noise ratio and provides more reliable data for subsequent analysis.

The merger is useful for improving signal-to-noise ratio in FCS experiments by combining data from multiple measurements, resulting in more robust and reliable correlation curves for fitting and interpretation.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - matplotlib (for plotting)
  - scipy (for statistical analysis)

## Usage

1. Launch the plugin from the ChiSurf menu: FCS > Merger
2. Load multiple correlation curves:
   - Import from files
   - Select from currently open datasets
3. Visualize individual curves to assess quality
4. Configure merging parameters:
   - Weighting method (equal, statistical, custom)
   - Normalization options
   - Time range selection
5. Generate the merged correlation curve
6. View statistical analysis of the merged data
7. Export the merged curve for further analysis

## Applications

- Improving data quality in FCS experiments
- Combining repeated measurements for more reliable results
- Reducing the impact of outliers or artifacts in individual measurements
- Preparing high-quality correlation data for model fitting
- Standardizing correlation curves from different measurement sessions
- Creating reference datasets from multiple experiments

## Benefits

- Enhances signal-to-noise ratio in correlation data
- Provides statistical measures of data quality and reliability
- Streamlines the process of combining multiple measurements
- Facilitates more accurate fitting of correlation models
- Reduces the impact of experimental artifacts on data interpretation
- Saves time by automating the merging process with proper weighting

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.