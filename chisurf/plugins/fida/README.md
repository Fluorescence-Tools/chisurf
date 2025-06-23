# Fluorescence Intensity Distribution Analysis (FIDA) Plugin

**NOTE: This plugin is currently broken and may not function properly.**

This plugin provides a graphical interface for analyzing the distribution of fluorescence intensities in single-molecule experiments.

## Features

- Loading and processing of photon data from TTTR files
- Calculation of photon count histograms
- Fitting of theoretical models to experimental data
- Determination of concentrations and specific brightnesses of fluorescent species
- Visualization of results with interactive plots

## Overview

FIDA is a powerful method for characterizing heterogeneous samples and resolving different molecular species based on their brightness. The plugin implements the complete FIDA theory, including the calculation of photon count distributions from first principles using the spatial profile of the detection volume.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - pyqtgraph
  - tttrlib (for TTTR file handling)
  - numba (for accelerated calculations)

## Usage

1. Launch the plugin from the ChiSurf menu: Single-Molecule > FIDA
2. Load a TTTR file containing photon data
3. Configure data settings:
   - Select routing channels
   - Set bin time
   - Define microtime range
4. Compute the intensity trace and photon count distribution
5. Configure FIDA optical parameters:
   - Set B(0) and coefficients for the detection volume profile
   - Adjust background rate
   - Set integration parameters
6. Define the number of species and their initial parameters
7. Fit the distribution using the FIDA model
8. View and save the results

## Theory

FIDA analyzes the distribution of photon counts in time bins to extract information about the number, concentration, and specific brightness of different fluorescent species in the sample. The method accounts for the spatial profile of the detection volume and the Poisson statistics of photon detection.

The specific brightness (q) represents the average number of photons detected per molecule per bin time, while the concentration (c) represents the average number of molecules in the detection volume.

## Applications

- Characterizing heterogeneous samples with multiple fluorescent species
- Determining concentrations and brightness values without calibration
- Studying molecular interactions and complex formation
- Analyzing aggregation processes
- Resolving species with different brightness values

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.