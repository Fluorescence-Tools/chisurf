"""
Processing Plugin

This plugin provides a specialized browser interface for running and visualizing
Processing language sketches within ChiSurf. Processing is a flexible software
sketchbook and language for learning how to code within the context of the visual arts.

The plugin allows users to:
1. Load and run Processing sketches from HTML files
2. Interact with the sketches in real-time
3. Navigate between different sketches
4. Create custom visualizations using the Processing language

This serves as a simple demonstration of how to use Processing within ChiSurf
for creating interactive data visualizations. Users can create their own Processing
sketches to visualize scientific data in novel ways.
"""

from .wizard import *

name = "Miscellaneous:Processing"
