# Lazy Lifetime Fitter (lltf)

A Python module for fitting fluorescence lifetime data using advanced exponential decay models.

## Overview

Lazy Lifetime Fitter (lltf) is a specialized tool for analyzing time-resolved fluorescence spectroscopy data. It provides robust algorithms for extracting fluorescence lifetimes from time-correlated single photon counting (TCSPC) measurements. The module implements state-of-the-art fitting procedures with comprehensive error analysis and model selection capabilities.

Fluorescence lifetime measurements provide valuable information about the molecular environment and dynamics of fluorophores. This information is crucial in various applications including:
- Protein conformation and dynamics studies
- Membrane biophysics
- Cell imaging
- Material science
- Photophysics research

## Installation

```bash
pip install lltf
```

Or install from source:

```bash
git clone https://github.com/yourusername/lltf.git
cd lltf
pip install -e .
```

## Features

- Fit fluorescence lifetime data with multiple exponential components
- Automatic determination of optimal number of lifetime components
- Convolution with instrument response function (IRF)
- Background estimation and correction
- IRF shift estimation and correction
- Pile-up correction
- Command-line interface for easy use
- Visualization of fits and residuals
- Integration with ChiSurf for GUI-based analysis

## Theoretical Background

Fluorescence lifetime decay is typically modeled as a sum of exponential decays:

I(t) = ∑ Aᵢ * exp(-t/τᵢ)

Where:
- I(t) is the fluorescence intensity at time t
- Aᵢ is the amplitude of the i-th component
- τᵢ is the lifetime of the i-th component

In practice, the measured signal is a convolution of this decay with the instrument response function (IRF):

I_measured(t) = IRF(t) ⊗ I(t) + Background

LLTF handles this convolution and provides tools for estimating and correcting for background and IRF shift.

## Basic Usage

### Command Line Interface

The simplest way to use lltf is through the command line interface:

```bash
lltf fit example/5-44_D0.dat example/IRF_D0.dat
```

This will fit a single-exponential decay to the data and save the results to `decay_data_fit.json` and a plot to `decay_data_fit.png`.

### Options

```
Options:
  -o, --output TEXT           Output JSON file
  -p, --plot TEXT             Output plot file
  -sp, --save-path TEXT       Path to save all output files (creates a subfolder
                              named "lltf_output" in data location if not specified)
  -c, --config PATH           Configuration YAML file
  -n, --n-lifetimes INTEGER   Number of lifetimes to fit (ignored if
                              --find-optimal is used)
  -f, --find-optimal          Find optimal number of lifetimes automatically
  -m, --max-lifetimes INTEGER
                              Maximum number of lifetimes to try when finding
                              optimal
  -pt, --prob-threshold FLOAT
                              Probability threshold for selecting the best
                              number of lifetimes
  -s, --skiprows INTEGER      Number of rows to skip in data files
  -d, --delimiter TEXT        Delimiter used in data files
  -t, --time-column INTEGER   Column index for time data
  -y, --counts-column INTEGER
                              Column index for counts data
  -v, --verbose               Print verbose output
  -si, --save-intermediate    Save intermediate results when finding optimal
                              number of lifetimes (default: enabled)
  -ib, --intermediate-base TEXT
                              Base filename for intermediate results (defaults
                              to output filename without extension)
  --help                      Show this message and exit.
```

### Examples

#### Fit with a single lifetime component

```bash
lltf fit example/5-44_D0.dat example/IRF_D0.dat -v
```

#### Fit with multiple lifetime components

```bash
lltf fit example/5-44_D0.dat example/IRF_D0.dat -n 2 -v
```

#### Find the optimal number of lifetime components

```bash
lltf fit example/5-44_D0.dat example/IRF_D0.dat -f -m 4 -v
```

#### Use a configuration file

```bash
lltf fit example/5-44_D0.dat example/IRF_D0.dat -c example/config.yml -v
```

#### Specify a custom save path for output files

```bash
lltf fit example/5-44_D0.dat example/IRF_D0.dat -sp ./results -v
```

This will save all output files (JSON, PNG, intermediate results) to the `./results` directory. If the directory doesn't exist, it will be created automatically.

## Python API

You can also use lltf as a Python module:

```python
from lltf import fit_lifetime, Decay
import numpy as np
import matplotlib.pyplot as plt

# Load data
decay_data = np.genfromtxt('example/5-44_D0.dat')
irf_data = np.genfromtxt('example/IRF_D0.dat')

# Create Decay object
decay = Decay(
    decay=decay_data[:, 1],
    irf=irf_data[:, 1],
    time_axis=decay_data[:, 0]
)

# Set analysis range
decay.get_analysis_range(verbose=True)

# Estimate background
decay.estimate_background(verbose=True)

# Estimate IRF shift
decay.estimate_irf_shift(verbose=True)

# Fit with 2 lifetime components
result = decay.fit(n_lifetimes=2, verbose=True)

# Plot the results and save to file
decay.plot('fit_result.png')

# Save the results to JSON
decay.to_json('fit_result.json')

# Or use the fit_lifetime function directly
result = fit_lifetime(
    decay_file='example/5-44_D0.dat',
    irf_file='example/IRF_D0.dat',
    n_lifetimes=2,
    verbose=True
)
```

## Integration with ChiSurf

LLTF integrates with the ChiSurf application, providing a graphical user interface for fluorescence lifetime analysis. The integration allows users to:

1. Load and visualize TCSPC data directly in ChiSurf
2. Configure fitting parameters through an intuitive interface
3. Run lifetime fitting with real-time progress display
4. Visualize fitting results with interactive plots
5. Export results in various formats

To access the LLTF functionality in ChiSurf, use the "Tools:Lifetime Analysis" menu option.

## Configuration

You can customize the fitting process using a YAML configuration file:

```yaml
verbose: true
estimate_background_parameter:
  enabled: true
  initial_irf_background: 0.0
  fit_irf_background: true
  average_window: 10
analysis_range_parameter:
  count_threshold: 10.0
  area: 0.999
  start_at_peak: false
  start_fraction: 0.1
  skip_first_channels: 0
  skip_last_channels: 0
estimate_irf_shift_parameters:
  enabled: true
  apply_shift: true
  irf_time_shift_scan_range: [-8.0, 8.0]
  irf_time_shift_scan_n_steps: 20
lifetime_fit_parameter:
  find_optimal: false
  maximum_number_of_lifetimes: 6
  prob_threshold: 0.68
  plot_probabilities: true
  plot_weighted_residuals: true
  randomize_initial_values:
    enabled: false
    min_lifetime: 0.5
    max_lifetime: 5.0
    amplitude_variation: 0.5
pile_up_correction:
  enabled: false
  rep_rate: 80.0
  dead_time: 85.0
  measurement_time: 60.0
plot_resulting_fit: false  # Note: This parameter is ignored. Plots are always saved to file, never shown on screen.
```

## Advanced Usage

### Model Selection

LLTF uses Bayesian Information Criterion (BIC) for model selection when finding the optimal number of lifetime components. The BIC balances goodness of fit with model complexity to avoid overfitting.

### Error Analysis

Standard errors for fitted parameters are calculated from the covariance matrix of the fit. These provide an estimate of the uncertainty in the fitted lifetimes and amplitudes.

### Handling Complex Decays

For complex decay profiles, consider:
1. Starting with a single-exponential model and gradually increasing complexity
2. Using the `--find-optimal` option to automatically determine the best model
3. Examining residuals for systematic deviations that might indicate an inadequate model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License