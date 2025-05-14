"""
Histogram-Microtime

This plugin provides tools for creating, visualizing, and analyzing microtime histograms 
from Time-Tagged Time-Resolved (TTTR) fluorescence data.

Features:
- Loading and processing of TTTR files from various formats
- Creation of microtime histograms for parallel and perpendicular detection channels
- Support for burst selection using BID/BUR files for single-molecule analysis
- Cumulative histogram generation from multiple files
- Export of histogram data for further analysis
- Integration with ChiSurf for advanced data processing

Microtime histograms represent the distribution of photon arrival times relative to the 
excitation pulse, providing valuable information about fluorescence lifetimes and 
molecular dynamics. In polarization-resolved measurements, separate histograms for 
parallel and perpendicular detection channels enable fluorescence anisotropy analysis.

The plugin is particularly useful for time-resolved fluorescence spectroscopy, 
fluorescence lifetime imaging (FLIM), and single-molecule experiments where 
temporal information about photon arrival is critical for understanding molecular 
properties and dynamics.
"""

name = "Tools:Histogram-Microtime"
