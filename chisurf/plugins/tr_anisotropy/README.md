# Time-Resolved Anisotropy Plugin

This plugin provides tools for analyzing time-resolved fluorescence anisotropy data.

## Features

- Load and process polarization-resolved fluorescence decay data
- Set up and visualize rotation spectra and lifetime components
- Create and manage anisotropy fits with multiple rotation correlation times
- Analyze rotational diffusion of fluorophores in different environments

## Overview

Time-resolved anisotropy is a powerful technique for studying the rotational motion of fluorophores, providing insights 
into molecular size, shape, flexibility, and interactions. This plugin implements a wizard-based interface that 
guides users through the process of setting up and analyzing anisotropy decay data, from data loading to model fitting 
and result visualization.

The plugin supports multiple rotation correlation times and lifetime components, making it suitable for analyzing 
complex systems with heterogeneous rotational dynamics or multiple fluorophore populations.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - matplotlib (for plotting)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Anisotropy-Wizard
2. Load parallel and perpendicular fluorescence decay data
3. Configure analysis parameters:
   - Set up lifetime components
   - Define rotation correlation times
   - Adjust fitting parameters
4. Perform anisotropy decay fitting
5. Visualize and interpret results
6. Export analysis for further use

## Applications

- Determining the size and shape of macromolecules
- Studying protein-protein interactions
- Analyzing membrane fluidity and microviscosity
- Investigating conformational changes in biomolecules
- Characterizing molecular dynamics in complex environments
- Monitoring binding events through changes in rotational diffusion

## Theory

Fluorescence anisotropy decay is described by the equation:
r(t) = r₀ × Σ βᵢ × exp(-t/φᵢ)

Where:
- r(t) is the anisotropy at time t
- r₀ is the fundamental anisotropy
- βᵢ are the pre-exponential factors
- φᵢ are the rotational correlation times

The plugin implements this model with support for multiple correlation times to account for complex rotational dynamics.

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.