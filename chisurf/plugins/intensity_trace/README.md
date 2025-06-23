# Intensity Trace Analysis Plugin

This plugin provides tools for analyzing fluorescence intensity time traces from single-molecule experiments. It 
enables researchers to extract dynamic information from photon counting data, particularly for studying conformational 
changes, molecular interactions, and reaction kinetics at the single-molecule level.

## Features

- Loading and displaying Time-Tagged Time-Resolved (TTTR) data as intensity traces
- Histogram analysis of photon counts with customizable binning
- Hidden Markov Model (HMM) analysis for state detection and classification
- Bayesian Information Criterion (BIC) calculation for optimal state number determination
- Dwell time analysis for extracting kinetic information and rate constants
- FRET efficiency calculation and state-specific distribution analysis
- Transition probability matrix visualization and analysis
- Exponential fitting of dwell time distributions
- Interactive visualization with adjustable parameters
- Support for multi-channel data analysis (donor/acceptor channels)

## Workflow

The plugin implements a comprehensive workflow for single-molecule state analysis:
1. Load TTTR data and convert to binned intensity traces
2. Visualize traces and photon count distributions
3. Apply HMM to identify discrete states in noisy data
4. Analyze state transitions and dwell times to extract kinetic information
5. For FRET data, calculate efficiency distributions for each state

## Requirements

- Python packages:
  - PyQt5
  - pyqtgraph
  - numpy
  - scipy
  - tttrlib
  - hmmlearn

## Usage

1. Launch the plugin from the ChiSurf menu: Single-Molecule > Intensity trace
2. Load a TTTR file (e.g., .ptu file)
3. Configure the analysis parameters:
   - Set the time window for binning
   - Select the channels to display
   - Adjust histogram binning and range
4. Perform HMM analysis to identify states
5. Analyze dwell times, transition matrices, and FRET distributions
6. Save the results

## Applications

Ideal for analyzing single-molecule FRET, protein folding/unfolding, enzyme dynamics, ligand binding, blinking 
behavior, or any other dynamic processes that can be observed in fluorescence intensity traces. The HMM approach is 
particularly powerful for detecting states in noisy data with overlapping distributions.

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.