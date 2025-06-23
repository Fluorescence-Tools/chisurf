# Kappa2 Distribution Plugin

This plugin provides tools for calculating and visualizing the distribution of the orientation factor κ² for Förster 
Resonance Energy Transfer (FRET) experiments.

## Features

- Calculate κ² distributions using different models (Wobbling-in-Cone, Diffusion-during-Lifetime)
- Visualize the distribution of κ² values
- Calculate the effect of κ² uncertainty on apparent FRET distances
- Support for known and unknown donor-acceptor orientations
- Incorporate steady-state anisotropy measurements to estimate fluorophore mobility

## Background

The orientation factor κ² is a critical parameter in FRET that describes the relative orientation of the donor emission 
dipole and the acceptor absorption dipole. It affects the calculation of the Förster radius (R₀) and consequently the 
distance measurements derived from FRET experiments.

In most FRET applications, κ² is assumed to be 2/3 (≈0.667), which is valid only when both fluorophores undergo 
isotropic rotational diffusion that is much faster than the fluorescence lifetime. However, in many biological systems, 
this assumption may not hold due to restricted rotational mobility of the fluorophores.

This plugin allows researchers to model more realistic κ² distributions based on experimental anisotropy data, providing 
more accurate distance measurements in FRET experiments where the standard assumptions about fluorophore mobility may 
not apply.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - matplotlib (for plotting)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Kappa2 Distribution
2. Select the model for κ² distribution calculation:
   - Wobbling-in-Cone model
   - Diffusion-during-Lifetime model
3. Input parameters:
   - Steady-state anisotropy values for donor and acceptor
   - Fundamental anisotropy values
   - Fluorescence lifetimes
   - Rotational correlation times (if known)
4. Calculate and visualize the κ² distribution
5. Analyze the effect on FRET distance measurements
6. Export results for further analysis or publication

## Theory

The orientation factor κ² is given by:
κ² = (cos θT - 3 cos θD cos θA)²

Where:
- θT is the angle between the donor emission dipole and the acceptor absorption dipole
- θD is the angle between the donor emission dipole and the line connecting the donor and acceptor
- θA is the angle between the acceptor absorption dipole and the line connecting the donor and acceptor

The plugin implements various models to calculate the distribution of κ² values based on the rotational mobility of the 
fluorophores, which can be estimated from anisotropy measurements.

## Applications

- Improving the accuracy of FRET-derived distance measurements
- Estimating uncertainty in FRET measurements due to orientation effects
- Studying systems with restricted fluorophore mobility
- Validating the κ² = 2/3 assumption in specific experimental setups
- Educational tool for understanding the impact of orientation on FRET

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.