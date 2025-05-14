"""

Ucfret Plugin

This plugin is part of the ChiSurf application.

"""

import sys
from chisurf.gui.widgets.ucfret.wizard import UCFRETWizard

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:Bayesian FRET Analysis"

"""
UCFRET: Bayesian FRET Analysis

This plugin provides tools for analyzing fluorescence decays to extract FRET efficiency 
distributions using Bayesian inference. It implements the UCFRET methodology for 
analyzing time-resolved fluorescence data.

Features:
- Load and analyze fluorescence decay data
- Fit donor and acceptor decays
- Sample posterior distributions of FRET parameters
- Visualize FRET efficiency distributions
- Export results for further analysis

Ideal for extracting detailed information about conformational states from 
time-resolved FRET experiments.
"""

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the UCFRETWizard class
    window = UCFRETWizard()
    # Show the window
    window.show()