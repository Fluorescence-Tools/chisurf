import sys
from chisurf.gui.tools.tttr.histogram import HistogramTTTR

# Define the plugin name - this will appear in the Plugins menu
name = "TTTR:Generate Decay"

"""
TTTR Histogram (Generate Decay)

This plugin provides a graphical interface for generating fluorescence decay histograms
from Time-Tagged Time-Resolved (TTTR) data.

Features:
- Configurable histogram parameters (binning, time range)
- Channel selection for multi-channel TTTR data
- Interactive visualization of decay curves
- Export of histogram data for further analysis

The histogram generator is useful for time-resolved fluorescence spectroscopy and
lifetime analysis.
"""

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the HistogramTTTR class
    window = HistogramTTTR()
    # Show the window
    window.show()