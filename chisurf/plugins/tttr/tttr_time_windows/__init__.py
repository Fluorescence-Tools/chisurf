"""
TTTR Time-Window Splitter

A simple wizard plugin to split TTTR files into fixed-duration time windows (time windows, tws)
and save them as BID files (start, stop photon indices). The wizard has three pages:

1) Setup: choose the time-window length (ms) and the output folder.
2) Files: drag-and-drop TTTR files (.ptu, .ht2, .ht3, .phu, .pt3, .t3r).
3) Process: preview a single file's intensity trace with window boundaries and process all files.

The resulting BID files are saved in the chosen output folder, one per TTTR input file.
"""

name = "Single-Molecule:TTTR→Time-Window BIDs"

# Optional icon, if available later we can set it here
icon = None
