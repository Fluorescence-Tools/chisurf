"""
LLTF: Lazy Lifetime Analysis

This plugin provides tools for analyzing fluorescence lifetime data using the LLTF module.
It implements advanced fitting procedures for extracting fluorescence lifetimes from
time-correlated single photon counting (TCSPC) measurements.

Features:
- Load and analyze TCSPC data
- Fit decay curves with multiple exponential components
- Automatic determination of optimal number of lifetime components
- Convolution with instrument response function (IRF)
- Background estimation and correction
- IRF shift estimation and correction
- Pile-up correction
- Visualization of fits and residuals
- Export results for further analysis

Ideal for extracting detailed information about fluorophore environments and dynamics
from time-resolved fluorescence experiments.
"""

import sys
from chisurf.plugins.lltf.lltf_gui import LLTFGUIWizard

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:Lazy Lifetime Analysis"

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the LLTFGUIWizard class
    window = LLTFGUIWizard()
    # Show the window
    window.show()