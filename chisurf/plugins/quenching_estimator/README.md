# Quenching Estimator Plugin (QuEst)

This plugin provides tools for simulating fluorescence quenching processes in macromolecules and analyzing time-resolved FRET measurements.

## Features

- Simulation of dye diffusion using accessible volume (AV) calculations
- Calculation of fluorescence quenching based on proximity to quencher residues
- Generation of fluorescence decay histograms
- Visualization of diffusion trajectories and protein structures
- Analysis of time-resolved FRET measurements of labeled macromolecules
- Integration with molecular structure data

## Overview

The Quenching Estimator Plugin (QuEst) implements a diffusion simulation approach to analyze time-resolved FRET 
measurements of labeled macromolecules. It simulates the diffusion of fluorescent dyes around a macromolecule and 
calculates quenching effects based on the proximity to quencher residues such as Tryptophan, Tyrosine, and Histidine.

The plugin provides a graphical interface for setting up and running these simulations, as well as for analyzing and 
visualizing the results. By simulating the movement of fluorescent dyes and their interactions with quencher residues, 
researchers can gain insights into the dynamic behavior of macromolecules and interpret experimental FRET data more 
accurately.

The diffusion simulation methodology is based on the approach described in:
Peulen, T. O., Opanasyuk, O., & Seidel, C. A. M. (2017). "Combining Graphical and Analytical Methods with Molecular 
Simulations To Analyze Time-Resolved FRET Measurements of Labeled Macromolecules Accurately." 
The Journal of Physical Chemistry B, 121(35), 8211-8241.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - quest (for dye diffusion simulation)
  - chisurf core modules
  - Molecular visualization libraries

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > QuEst (Quenching estimator)
2. Load a molecular structure:
   - Import a PDB file or other supported molecular structure format
   - The structure will be displayed in the visualization panel
3. Configure simulation parameters:
   - Define dye attachment points on the macromolecule
   - Set diffusion parameters (step size, number of steps, etc.)
   - Identify quencher residues (Trp, Tyr, His) or specify custom quenchers
4. Run the simulation:
   - Click the "Run Simulation" button to start the diffusion simulation
   - Monitor the progress in the status panel
5. Analyze results:
   - View the generated fluorescence decay histograms
   - Examine the diffusion trajectories of the dyes
   - Analyze quenching effects based on proximity to quencher residues
6. Export data:
   - Save simulation results for further analysis
   - Export visualizations and histograms

## Applications

The Quenching Estimator plugin can be used for:
- Interpreting time-resolved FRET measurements of labeled macromolecules
- Studying the effects of quencher residues on fluorescence properties
- Investigating the dynamic behavior of fluorescent dyes attached to macromolecules
- Validating experimental FRET data through simulation
- Optimizing labeling strategies for FRET experiments
- Educational demonstrations of fluorescence quenching principles
- Research into protein structure and dynamics

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.