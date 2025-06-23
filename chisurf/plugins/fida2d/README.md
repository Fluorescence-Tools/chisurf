# 2D Fluorescence Intensity Distribution Analysis (FIDA2D) Plugin

**NOTE: This plugin is currently broken and may not function properly.**

This plugin provides a graphical interface for analyzing the joint distribution of fluorescence intensities in two detection channels.

## Features

- Loading and processing of photon data from TTTR files
- Calculation of 2D photon count histograms
- Fitting of theoretical models to experimental data
- Determination of concentrations and specific brightnesses of fluorescent species
- Visualization of results with interactive 2D plots

## Overview

FIDA2D extends the capabilities of standard FIDA by analyzing correlations between different detection channels, making it powerful for studying species with different brightness characteristics in multiple channels, such as FRET-labeled molecules.

The plugin implements the theoretical framework for 2D-FIDA, which analyzes the joint probability distribution of photon counts in two detection channels to extract information about molecular species with different brightness characteristics.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - pyqtgraph
  - matplotlib
  - tttrlib (for TTTR file handling)

## Usage

1. Launch the plugin from the ChiSurf menu: Single-Molecule > FIDA-2D
2. Load a TTTR file containing photon data
3. Configure data settings:
   - Select routing channels for both detection channels
   - Set bin time
4. Compute the experimental 2D histogram
5. Define the number of species and their initial parameters:
   - Concentration (c)
   - Specific brightness in channel 1 (q1)
   - Specific brightness in channel 2 (q2)
6. Fit the 2D distribution using the FIDA2D model
7. View and interpret the results

## Theory

FIDA2D analyzes the joint distribution of photon counts in two detection channels to extract information about the number, concentration, and specific brightness of different fluorescent species in the sample. The method accounts for the spatial profile of the detection volume and the Poisson statistics of photon detection.

The theoretical model is based on the generating function approach, which relates the joint probability distribution to the molecular properties through Fourier transformation.

## Applications

- Studying FRET-labeled molecules with different energy transfer efficiencies
- Analyzing samples with multiple fluorescent species
- Characterizing molecular interactions and complex formation
- Resolving species with different brightness ratios in two channels
- Investigating binding and conformational changes

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.