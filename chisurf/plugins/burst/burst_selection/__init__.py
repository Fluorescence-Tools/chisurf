"""
Single Molecule Burst Selection Plugin

This plugin provides tools for analyzing single-molecule fluorescence bursts in TTTR data.
It allows users to:
1. Load and process TTTR files for burst analysis
2. Visualize burst data in histograms
3. Fit Gaussian mixtures to the histograms
4. Display fit results in a table

The plugin supports drag-and-drop file loading and provides interactive visualization
of proximity ratio distributions, which is particularly useful for single-molecule
FRET experiments. It can fit multiple Gaussian components to identify different
conformational states or populations in the data.
"""

name = "Single-Molecule:Burst-Selection"

