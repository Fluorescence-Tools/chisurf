# Spectra Viewer Plugin

This plugin provides a comprehensive tool for visualizing, comparing, and analyzing absorption and emission spectra of fluorophores.

## Features

- Load and display multiple absorption and emission spectra simultaneously
- Normalize spectra for easier comparison
- Calculate Förster radius (R₀) for FRET pairs with customizable parameters
- View and edit fluorophore metadata including optical properties
- Display molecular structures and chemical diagrams of fluorophores
- Add custom dyes with their spectra and properties
- Download spectral data for commercial fluorophores

## Overview

The plugin features an intuitive interface with interactive plots and supports side-by-side comparison of multiple fluorophores. This makes it particularly useful for:
- Selecting optimal FRET pairs for experiments
- Designing multiplexed fluorescence experiments
- Comparing spectral properties across different fluorophore families
- Educational purposes to understand spectral characteristics and FRET theory

The Förster radius calculator implements the complete overlap integral calculation with all relevant parameters (orientation factor, quantum yield, refractive index), providing accurate R₀ values for FRET experimental design.

## Requirements

- Python packages:
  - PyQt5
  - pyqtgraph
  - numpy
  - matplotlib (for plotting)
  - pillow (for image handling)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Spectra Viewer
2. Browse and select fluorophores from the database
3. View absorption and emission spectra in the interactive plot
4. Add spectra to the comparison list for side-by-side analysis
5. Calculate Förster radius for potential FRET pairs
6. Add custom fluorophores with their spectral data and properties
7. Export spectral data to CSV for further analysis

## Database Features

The plugin maintains a database of fluorophores with:
- Absorption and emission spectra
- Extinction coefficients
- Quantum yields
- Molecular structures
- Literature references
- Other relevant metadata

Users can add custom fluorophores to the database, making it a valuable resource for planning fluorescence experiments.

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.