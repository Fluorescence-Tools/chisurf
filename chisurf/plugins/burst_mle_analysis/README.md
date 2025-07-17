# Burst MLE Lifetime Analysis Plugin

This plugin provides a powerful interface for analyzing fluorescence lifetime data from single-molecule experiments 
using Maximum Likelihood Estimation (MLE).

## Features

- Analysis of burst data from Time-Tagged Time-Resolved (TTTR) measurements
- Maximum Likelihood Estimation for accurate fluorescence lifetime determination
- Handling of Instrument Response Function (IRF) and background corrections
- Support for multiple detector channels and burst selection criteria
- Interactive visualization of lifetime fits and results
- Comprehensive parameter adjustment for optimizing analysis

## Overview

Fluorescence lifetime analysis at the single-molecule level presents unique challenges due to the limited number of 
photons available for analysis. Traditional least-squares fitting methods often perform poorly in these 
low-photon-count scenarios, leading to biased or imprecise parameter estimates.

Maximum Likelihood Estimation (MLE) is particularly advantageous for single-molecule fluorescence lifetime analysis as 
it provides more accurate parameter estimates than traditional least-squares methods when dealing with low photon 
counts, which is typical in single-molecule experiments.

This plugin implements MLE-based lifetime analysis specifically optimized for burst data from single-molecule 
experiments, allowing researchers to extract reliable lifetime information even from datasets with limited photon 
statistics.

## Requirements

- Python packages:
  - PyQt5
  - numpy
  - scipy
  - matplotlib
  - tttrlib (for TTTR file handling)
  - lmfit (for optimization algorithms)

## Usage

1. Launch the plugin from the ChiSurf menu: Single-Molecule > Burst MLE Lifetime Analysis
2. Load burst data:
   - Import TTTR files
   - Load or create burst selection files
3. Configure analysis parameters:
   - Select detector channels
   - Set microtime ranges
   - Configure IRF handling
   - Adjust background correction
4. Set up MLE parameters:
   - Define lifetime model (single or multi-exponential)
   - Configure optimization settings
5. Run the analysis
6. Visualize and evaluate results:
   - Lifetime distributions
   - Fit quality metrics
   - Parameter confidence intervals
7. Export results for further analysis or reporting

## Applications

- Single-molecule FRET studies
- Conformational dynamics of biomolecules
- Protein folding and unfolding kinetics
- Enzyme reaction mechanisms
- Molecular interaction studies
- Characterization of heterogeneous samples
- Investigation of dynamic processes at the single-molecule level

## Theory

The Maximum Likelihood Estimation approach for fluorescence lifetime analysis is based on finding the parameter values 
that maximize the likelihood function:

L(θ|data) = ∏ p(t_i|θ)

Where:
- L is the likelihood function
- θ represents the model parameters (lifetimes, amplitudes)
- t_i are the individual photon arrival times
- p(t_i|θ) is the probability of observing a photon at time t_i given parameters θ

For computational efficiency, the negative log-likelihood is typically minimized:

-ln(L) = -∑ ln(p(t_i|θ))

The plugin implements this approach with appropriate handling of the instrument response function and background 
contributions, providing robust lifetime estimates even for low photon counts.

## References

The methodology implemented in this plugin is based on the approach described in:

Maus, M., Cotlet, M., Hofkens, J., Gensch, T., De Schryver, F. C., Schaffer, J., & Seidel, C. A. (2001). An Experimental 
Comparison of the Maximum Likelihood Estimation and Nonlinear Least-Squares Fluorescence Lifetime Analysis of Single 
Molecules. Analytical Chemistry, 73(9), 2078-2086. https://doi.org/10.1021/ac000877g

## License

This plugin is part of the ChiSurf package and is distributed under the same license.

## Author

This plugin was created as part of the ChiSurf project.