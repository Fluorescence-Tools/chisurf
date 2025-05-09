import sys
from quest.lib.tools.dye_diffusion import TransientDecayGenerator

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:Dye Diffusion"

"""
Dye Diffusion Decay Generator

This plugin provides a graphical interface for generating fluorescence decay curves
based on dye diffusion models.

Features:
- Simulation of fluorescence decays with various diffusion parameters
- Configuration of dye properties and environment
- Interactive visualization of decay curves
- Export of simulated data for further analysis

The decay generator is useful for studying the effects of diffusion on fluorescence
lifetime measurements and for generating test data for analysis methods.
"""

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the TransientDecayGenerator class
    window = TransientDecayGenerator()
    # Show the window
    window.show()