# FRET Calculator Plugin

This plugin provides a calculator for Förster Resonance Energy Transfer (FRET) parameters. It allows users to calculate and convert between various FRET-related quantities.

## Features

The calculator enables conversion and calculation of:
- FRET efficiency (E)
- Distance between fluorophores (R)
- Förster radius (R0)
- Donor lifetime in the presence of acceptor (τDA)
- Donor lifetime in the absence of acceptor (τD)
- FRET rate constant (kFRET)

The calculator automatically updates all values when any parameter is changed, making it easy to explore the relationships between different FRET parameters.

## Background

Förster Resonance Energy Transfer (FRET) is a distance-dependent physical process by which energy is transferred non-radiatively from an excited donor molecule to an acceptor molecule. FRET is widely used as a "spectroscopic ruler" to measure distances between biomolecules labeled with appropriate fluorophores.

The relationship between FRET efficiency (E) and distance (R) is given by:
E = 1 / [1 + (R/R0)^6]

Where R0 is the Förster radius, the distance at which the FRET efficiency is 50%.

## Requirements

- Python packages:
  - qtpy (for GUI components)
  - numpy
  - matplotlib (for plotting)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > FRET Calculator
2. Enter known values in the appropriate fields
3. The calculator will automatically compute and update all related parameters
4. Explore how changing one parameter affects the others

## Applications

- Planning FRET experiments by selecting appropriate donor-acceptor pairs
- Interpreting FRET data to determine molecular distances
- Educational tool for understanding FRET relationships
- Converting between different FRET-related parameters
- Estimating expected FRET efficiencies for known distances

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.