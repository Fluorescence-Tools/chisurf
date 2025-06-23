# Photon Counting Histogram (PCH) Analysis Plugin

This plugin provides tools for analyzing the distribution of photon counts in fluorescence time traces.

## Features

- Loading and processing of TTTR files
- Calculation of photon count histograms
- Fitting of theoretical models to experimental data
- Support for single or multiple species analysis
- Visualization of results with interactive plots
- Statistical analysis of fit quality

## Overview

PCH analysis can reveal information about:
- Molecular brightness (ε)
- Number of molecules in the detection volume (⟨N⟩)
- Presence of multiple species with different brightness values

The plugin implements the theoretical framework for PCH analysis, which uses the statistical fluctuations in 
fluorescence intensity to extract information about the concentration and brightness of fluorescent molecules.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - pyqtgraph
  - tttrlib (for TTTR file handling)
  - numba (for accelerated calculations)

## Usage

1. Launch the plugin from the ChiSurf menu: Single-Molecule > Photon Counting Histogram
2. Load a TTTR file containing photon data
3. Configure data settings:
   - Select routing channels
   - Set bin time
   - Define microtime range
4. Compute the intensity trace and photon count histogram
5. Define the number of species and their initial parameters:
   - Molecular brightness (ε)
   - Average number of molecules (⟨N⟩)
6. Fit the histogram using the PCH model
7. View and save the results

## Theory

PCH analysis is based on the statistical analysis of fluorescence intensity fluctuations. The method accounts for the 
spatial profile of the detection volume and the Poisson statistics of photon detection.

The theoretical model describes the probability distribution of photon counts based on:
- The spatial distribution of the excitation profile
- The molecular brightness of the fluorophores
- The number of molecules in the detection volume

## Applications

- Determining molecular brightness and concentration
- Distinguishing between species with different brightness values
- Studying molecular interactions and complex formation
- Analyzing heterogeneous samples
- Investigating molecular aggregation

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.