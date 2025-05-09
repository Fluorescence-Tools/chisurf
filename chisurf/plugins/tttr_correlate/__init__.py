import sys
from chisurf.gui.tools.tttr.correlate import CorrelateTTTR

# Define the plugin name - this will appear in the Plugins menu
name = "TTTR:Correlate"

"""
TTTR Correlate

This plugin provides a graphical interface for calculating correlation functions
from Time-Tagged Time-Resolved (TTTR) data.

Features:
- Support for various correlation types (auto, cross)
- Configurable correlation parameters
- Interactive visualization of correlation curves
- Export of correlation results

The correlator is essential for analyzing dynamic processes in fluorescence
correlation spectroscopy (FCS) and related techniques.
"""

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the CorrelateTTTR class
    window = CorrelateTTTR()
    # Show the window
    window.show()