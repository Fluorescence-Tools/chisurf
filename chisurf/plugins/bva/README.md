# Single Molecule Burst Variance Analysis (BVA) Plugin

This plugin implements Burst Variance Analysis for single-molecule FRET experiments.

## Features

- Load burst data from previous burst analysis
- Configure BVA parameters (donor/acceptor channels, micro-time ranges, etc.)
- Compute BVA metrics for each burst
- Visualize results with static FRET line for comparison
- Save results to files for further analysis

## Overview

BVA is a technique that analyzes the variance of FRET efficiency within individual bursts to distinguish between static 
and dynamic heterogeneity in the sample. It is particularly useful for identifying conformational dynamics in 
biomolecules that occur on timescales comparable to the burst duration.

By comparing the measured variance of FRET efficiency within bursts to the expected shot-noise limited variance, BVA 
can identify molecules undergoing conformational changes during the observation time.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - matplotlib (for plotting)
  - tttrlib (for TTTR file handling)

## Usage

1. Launch the plugin from the ChiSurf menu: Single-Molecule > Burst-Variance Analysis
2. Load burst data from previous burst analysis
3. Configure analysis parameters:
   - Select donor and acceptor channels
   - Define micro-time ranges
   - Set binning parameters
4. Compute BVA metrics for each burst
5. Visualize results with the static FRET line for comparison
6. Save results for further analysis

## Theory

BVA works by dividing each burst into sub-bursts and calculating the FRET efficiency for each sub-burst. The 
variance of these FRET efficiencies is then compared to the expected shot-noise limited variance:

- If the measured variance matches the expected shot-noise variance, the molecule is likely in a static conformation
- If the measured variance exceeds the expected shot-noise variance, the molecule is likely undergoing conformational changes during the observation

The plugin implements this analysis and provides visualization tools to interpret the results.

## Applications

- Identifying conformational dynamics in biomolecules
- Distinguishing between static and dynamic heterogeneity in samples
- Studying protein folding/unfolding events
- Investigating enzyme dynamics
- Analyzing molecular interactions with temporal resolution

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.