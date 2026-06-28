"""
FCS Correlator

This plugin provides a two-pane navigation-based correlator tool for computing
and merging fluorescence correlation spectroscopy (FCS) data. Features include:

- Detector and PIE window definition
- TTTR file selection with drag-and-drop
- Optional photon/burst filtering
- Multi-tau correlation with configurable parameters (bins, cascades, fine grid)
- FCS curve merging and export

The tool replaces the legacy QWizard with a modern navigation panel layout
(left step list, right view/display), built using the AutoForm declarative
UI framework for the correlator settings panel.
"""

name = "Spectroscopy:Fluorescence Correlation Spectroscopy:Correlator"
