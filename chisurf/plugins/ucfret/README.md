# UCFRET: Bayesian FRET Analysis Plugin

This plugin provides tools for analyzing fluorescence decays to extract FRET efficiency distributions using 
Bayesian inference. It implements the UCFRET methodology for analyzing time-resolved fluorescence data.

## Features

- Load and analyze fluorescence decay data
- Fit donor and acceptor decays
- Sample posterior distributions of FRET parameters
- Visualize FRET efficiency distributions
- Export results for further analysis
- Edit settings with a graphical editor
- Run ucfret CLI in a separate process
- Display sampling and analysis output in real-time

## Applications

Ideal for extracting detailed information about conformational states from time-resolved FRET experiments. The 
Bayesian approach provides robust uncertainty quantification and can reveal heterogeneity in molecular systems.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - matplotlib
  - ucfret (core analysis library)

## Usage

1. Launch the plugin from the ChiSurf menu: Tools > Bayesian FRET Analysis
2. Load fluorescence decay data
3. Configure analysis parameters using the graphical editor
4. Run the Bayesian sampling process
5. Visualize and interpret the resulting FRET efficiency distributions
6. Export results for further analysis or publication

## Methodology

The plugin uses Bayesian inference to analyze time-resolved fluorescence data. This approach:
- Accounts for measurement uncertainties
- Provides posterior probability distributions rather than point estimates
- Can reveal heterogeneity in the sample
- Allows for model comparison and selection

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.