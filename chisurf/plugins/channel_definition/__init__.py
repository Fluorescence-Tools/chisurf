"""
Channel Definition

This plugin provides a graphical interface for defining and configuring detector channels 
and time windows for Time-Tagged Time-Resolved (TTTR) fluorescence spectroscopy data.

Features:
- Definition of detector channels with specific routing channel numbers
- Configuration of Pulsed Interleaved Excitation (PIE) time windows
- Association of microtime ranges with specific detectors
- Import and export of channel definitions via JSON files
- Interactive wizard interface for intuitive setup

The Channel Definition plugin is essential for preprocessing TTTR data before analysis, 
allowing researchers to properly map physical detector channels to their experimental 
setup and define time windows for techniques like PIE. These definitions are used by 
other analysis plugins to correctly interpret the raw photon data.

The plugin is particularly useful for multi-detector setups and advanced fluorescence 
techniques such as single-molecule FRET, fluorescence lifetime imaging, and 
fluorescence correlation spectroscopy, where proper channel assignment is critical 
for accurate data analysis.
"""

name = "Setup:Channel Definition"
