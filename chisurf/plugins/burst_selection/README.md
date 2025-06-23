# Single Molecule Burst Selection Plugin

This plugin provides tools for analyzing single-molecule fluorescence bursts in TTTR data.

## Features

- Load and process TTTR files for burst analysis
- Visualize burst data in histograms
- Fit Gaussian mixtures to the histograms
- Display fit results in a table
- Support for drag-and-drop file loading
- Interactive parameter adjustment

## Overview

The plugin supports drag-and-drop file loading and provides interactive visualization of proximity ratio distributions, 
which is particularly useful for single-molecule FRET experiments. It can fit multiple Gaussian components to identify 
different conformational states or populations in the data.

## Technical Details

### Burst Selection Process

The plugin uses a multi-step process for burst selection:

1. **Channel Definition**: Users define detector channels for donor (green) and acceptor (red) fluorescence.
2. **Photon Filtering**: The `WizardTTTRPhotonFilter` component filters photons based on user-defined parameters.
3. **Burst Identification**: Photon bursts are identified using time-window and photon count thresholds.
4. **File Processing**: The plugin processes TTTR files and generates .bur files containing burst information.
5. **Data Analysis**: Burst data is loaded from .bur files and analyzed to calculate parameters like proximity ratios.

### Proximity Ratio Calculation

The proximity ratio (PR) is calculated for each burst using the formula:

```
PR = N_red / (N_red + N_green)
```

Where:
- N_red is the number of photons detected in the acceptor (red) channel
- N_green is the number of photons detected in the donor (green) channel

This ratio is a measure of FRET efficiency and ranges from 0 (no FRET) to 1 (high FRET).

### Gaussian Mixture Fitting

The plugin can fit multiple Gaussian components to histograms of burst parameters (e.g., proximity ratios):

1. Users specify the number of Gaussian components (k)
2. Initial guesses for amplitude, mean, and standard deviation are automatically generated
3. The `curve_fit` function from SciPy is used to optimize the parameters
4. Individual Gaussian components and their sum are plotted over the histogram
5. Fit results (amplitude, mean, standard deviation) with uncertainties are displayed in a table

The multi-Gaussian function is defined as:
```
f(x) = ∑(i=1 to k) A_i * exp(-0.5 * ((x - μ_i) / σ_i)²)
```

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy (for Gaussian fitting)
  - matplotlib (for plotting)
  - tttrlib (for TTTR file handling)
  - pandas (for data handling)

## Usage

1. Launch the plugin from the ChiSurf menu: Single-Molecule > Burst-Selection
2. Load TTTR data files via drag-and-drop or file browser
3. Configure burst selection parameters:
   - Set photon count thresholds
   - Define time windows
   - Select detection channels
4. Process files to generate burst data
5. Generate histograms of proximity ratios or other parameters
6. Adjust histogram parameters (bins, range)
7. Select the number of Gaussian components to fit
8. View fit results in the table display
9. Optionally edit the data using the spreadsheet view

## File Formats

- **Input**: TTTR (Time-Tagged Time-Resolved) files containing photon arrival times
- **Intermediate**: .bur files containing burst information (first/last photon, duration, photon counts)
- **Analysis**: The plugin works with pandas DataFrames for data manipulation and analysis

## Applications

- Single-molecule FRET analysis
- Identifying conformational states in biomolecules
- Studying molecular heterogeneity
- Analyzing dynamic processes at the single-molecule level
- Quantifying population distributions in complex samples

## Background

Single-molecule burst analysis involves detecting photon bursts from individual molecules as they diffuse through a 
confocal detection volume. By analyzing the properties of these bursts (such as intensity, duration, and proximity 
ratio), researchers can gain insights into molecular structure, dynamics, and interactions at the single-molecule 
level.

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.
