"""
PTU Splitter Plugin

This plugin provides functionality for splitting large TTTR (Time-Tagged Time-Resolved)
files, particularly those in the PicoQuant PTU format, into smaller segments. This is
useful for:

1. Breaking down large datasets into manageable chunks
2. Extracting specific time segments from long measurements
3. Creating subsets of data for parallel processing
4. Reducing memory requirements for analysis

The plugin features:
- Support for various TTTR file formats
- Configurable splitting parameters (photons per file, time segments)
- Options for micro-time binning to reduce file size
- Ability to reset macro-times in the output files
- Selection of output container formats (file format conversion)

This tool is particularly valuable for handling large datasets from long-duration
single-molecule or imaging experiments, making them more manageable for subsequent analysis.
"""

name = "TTTR:PTU-Splitter"

