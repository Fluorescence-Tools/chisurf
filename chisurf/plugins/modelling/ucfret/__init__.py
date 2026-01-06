"""UCFRET plugin module for ChiSurf.

This plugin provides a GUI for the ucfret module, which implements Bayesian analysis
of time-resolved FRET data.
"""

from .wizard import UCFRETWizard

# Define the plugin name - this will appear in the Plugins menu
name = "Dev:Spectroscopy:ucFRET"


if __name__ == "plugin":
    window = UCFRETWizard()
    window.show()
