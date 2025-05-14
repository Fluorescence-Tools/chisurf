"""
TTTR Convert

This plugin provides a graphical interface for converting Time-Tagged Time-Resolved (TTTR)
data files between different formats.

Features:
- Support for various TTTR file formats
- Batch conversion capabilities
- Configuration of conversion parameters
- Preview of file contents

The converter is useful for preparing TTTR data for analysis in different software packages.
"""

import sys
from chisurf.gui.tools.tttr.convert import TTTRConvert

# Define the plugin name - this will appear in the Plugins menu
name = "TTTR:Convert"



# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the TTTRConvert class
    window = TTTRConvert()
    # Show the window
    window.show()