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
- Edit settings with a graphical editor
- Run ucfret CLI in a separate process
- Display sampling and analysis output in real-time

Ideal for extracting detailed information about conformational states from
time-resolved FRET experiments.
"""

import sys
from chisurf.plugins.ucfret.ucfret_gui import UCFRETGUIWizard

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:Bayesian FRET Analysis"



# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the UCFRETGUIWizard class
    window = UCFRETGUIWizard()
    # Show the window
    window.show()
