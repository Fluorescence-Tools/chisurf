"""
FRET Calculator

This plugin provides a calculator for Förster Resonance Energy Transfer (FRET) parameters.
It allows users to calculate and convert between various FRET-related quantities such as:
- FRET efficiency (E)
- Distance between fluorophores (R)
- Förster radius (R0)
- Donor lifetime in the presence of acceptor (τDA)
- Donor lifetime in the absence of acceptor (τD)
- FRET rate constant (kFRET)

The calculator automatically updates all values when any parameter is changed,
making it easy to explore the relationships between different FRET parameters.
"""

from chisurf.gui.tools.fret.calculator.tau2r import FRETCalculator

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:FRET Calculator"



# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the FRETCalculator class
    window = FRETCalculator()
    # Show the window
    window.show()