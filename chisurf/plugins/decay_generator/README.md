# Dye Diffusion Decay Generator Plugin

This plugin provides a graphical interface for simulating fluorescence decay curves based on dye diffusion models with dynamic quenching effects.

## Features

- Simulation of fluorescence decays with various diffusion parameters
- Modeling of dynamic quenching processes between fluorescent dyes and protein quenchers
- Calculation of fluorescence lifetime changes due to collisional quenching
- Configuration of dye properties, diffusion coefficients, and quenching parameters
- Accessible volume (AV) calculations to model dye movement constraints
- Interactive 3D visualization of dye trajectories and quencher positions
- Generation of realistic fluorescence decay histograms
- Export of simulated data for further analysis

## Overview

Dynamic quenching occurs when a fluorescent dye in its excited state collides with a quencher molecule (typically certain amino acids like tryptophan, tyrosine, or histidine), resulting in non-radiative energy transfer that shortens the fluorescence lifetime. This process is distance-dependent and requires molecular diffusion.

The plugin simulates the Brownian motion of dye molecules attached to proteins, calculates collision events with quenchers, and generates the resulting fluorescence decay curves. These simulations are particularly valuable for:

- Interpreting experimental fluorescence decay data from protein-dye systems
- Understanding the relationship between protein structure and fluorescence quenching
- Designing fluorescence experiments with optimal dye-label positions
- Validating analytical models of dynamic quenching processes
- Studying the effects of diffusion constraints on fluorescence properties

The generated decay curves incorporate realistic photophysics including diffusion coefficients, quenching rates, and spatial constraints, providing a powerful tool for both experimental design and data interpretation in fluorescence spectroscopy.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - matplotlib
  - quest (for TransientDecayGenerator)
  - pytraj (for trajectory handling)
  - mdtraj (for molecular structure processing)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Dye Diffusion
2. Load or create a protein structure
3. Configure simulation parameters:
   - Dye properties (lifetime, diffusion coefficient)
   - Quencher positions and properties
   - Accessible volume constraints
   - Simulation time and resolution
4. Run the simulation
5. Visualize the results:
   - 3D trajectories of dye molecules
   - Fluorescence decay curves
   - Quenching event statistics
6. Export the simulated decay data for further analysis

## Applications

- Fluorescence lifetime studies of labeled proteins
- Investigation of protein conformational dynamics
- Optimization of fluorescent labeling strategies
- Interpretation of complex fluorescence decay kinetics
- Validation of analytical models for dynamic quenching
- Educational tool for understanding fluorescence quenching mechanisms
- Design of fluorescence-based sensors and probes

## Theory

The simulation is based on the Stern-Volmer relationship for dynamic quenching:
τ₀/τ = 1 + k_q × τ₀ × [Q]

Where:
- τ₀ is the unquenched fluorescence lifetime
- τ is the quenched fluorescence lifetime
- k_q is the bimolecular quenching constant
- [Q] is the quencher concentration

The plugin extends this model by incorporating spatial constraints, diffusion dynamics, and multiple quenching sites to provide more realistic simulations of fluorescence decay in complex biomolecular systems.

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.