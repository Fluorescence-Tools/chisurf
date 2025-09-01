"""
ALEX Creator

This plugin provides tools for converting and processing Alternating Laser Excitation (ALEX) 
data in Time-Tagged Time-Resolved (TTTR) files. It allows users to:

1. Load TTTR files, including SM format files
2. Convert SM files to PTU format
3. Apply ALEX period and shift parameters to transform macro time information into micro time
4. Visualize the resulting micro time histogram
5. Save the processed data as a new PTU file

The plugin implements essential module operations on macro time stored in micro time, 
enabling ALEX data to be processed with the same pipelines as Pulsed Interleaved Excitation 
(PIE) data. This is particularly useful for:

- Converting between different TTTR file formats
- Preparing ALEX data for analysis with standard PIE analysis tools
- Visualizing the distribution of photons in the ALEX period
- Optimizing ALEX period and shift parameters for specific experiments

The conversion process preserves all event data while transforming the time information
to make ALEX data compatible with PIE analysis workflows.
"""

name = "TTTR:ALEX Creator"
