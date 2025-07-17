"""
Lifetime MLE Analysis

This plugin provides a powerful interface for analyzing fluorescence lifetime data 
from imaging experiments using Maximum Likelihood Estimation (MLE).

Features:
- Analysis of image data from Time-Tagged Time-Resolved (TTTR) measurements
- Maximum Likelihood Estimation for accurate fluorescence lifetime determination
- Handling of Instrument Response Function (IRF) and background corrections
- Support for multiple detector channels
- Interactive visualization of lifetime fits and results
- Comprehensive parameter adjustment for optimizing analysis
- Export of results in various formats

MLE is particularly advantageous for fluorescence lifetime analysis
as it provides more accurate parameter estimates than traditional least-squares methods
when dealing with low photon counts.

The methodology implemented in this plugin is based on the approach described in:
Maus, M., Cotlet, M., Hofkens, J., Gensch, T., De Schryver, F. C., Schaffer, J., & Seidel, C. A. (2001).
An Experimental Comparison of the Maximum Likelihood Estimation and Nonlinear Least-Squares
Fluorescence Lifetime Analysis of Single Molecules.
Analytical Chemistry, 73(9), 2078-2086. https://doi.org/10.1021/ac000877g
"""

name = "Imaging:Lifetime MLE Analysis"

from .imgmle import LifetimeMleAnalysisWizard

def run():
    """
    Run the Lifetime MLE Analysis wizard.
    """
    wizard = LifetimeMleAnalysisWizard()
    wizard.show()
    return wizard

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the Kappa2Dist class
    window = run()
