"""
PTU Microtime Shifter Plugin

This plugin provides tools for adjusting the micro-time channels in TTTR (Time-Tagged Time-Resolved)
files, particularly those in the PicoQuant PTU format. Micro-time shifting is useful for:

1. Correcting timing offsets between different detection channels
2. Aligning fluorescence decay curves from different detectors
3. Compensating for electronic delays in the detection system
4. Preparing data for multi-detector analysis

The plugin features:
- Loading and visualizing TTTR files
- Applying global or channel-specific micro-time shifts
- Real-time visualization of micro-time histograms
- Saving the modified TTTR files

This tool is particularly valuable for multi-color FRET experiments where precise
temporal alignment of detection channels is critical for accurate analysis.
"""

name = "TTTR:PTU Microtime Shifter"

