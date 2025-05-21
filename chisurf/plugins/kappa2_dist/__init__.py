"""
Kappa2 Distribution

This plugin provides tools for calculating and visualizing the distribution of the orientation factor κ² 
for Förster Resonance Energy Transfer (FRET) experiments.

Features:
- Calculate κ² distributions using different models (Wobbling-in-Cone, Diffusion-during-Lifetime)
- Visualize the distribution of κ² values
- Calculate the effect of κ² uncertainty on apparent FRET distances
- Support for known and unknown donor-acceptor orientations
- Incorporate steady-state anisotropy measurements to estimate fluorophore mobility

The orientation factor κ² is a critical parameter in FRET that describes the relative orientation 
of the donor emission dipole and the acceptor absorption dipole. It affects the calculation of 
the Förster radius (R₀) and consequently the distance measurements derived from FRET experiments.

In most FRET applications, κ² is assumed to be 2/3 (≈0.667), which is valid only when both 
fluorophores undergo isotropic rotational diffusion that is much faster than the fluorescence 
lifetime. However, in many biological systems, this assumption may not hold due to restricted 
rotational mobility of the fluorophores.

This plugin allows researchers to model more realistic κ² distributions based on experimental 
anisotropy data, providing more accurate distance measurements in FRET experiments where the 
standard assumptions about fluorophore mobility may not apply.
"""

import sys
from .k2dgui import Kappa2Dist

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:Kappa2 Distribution"

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the Kappa2Dist class
    window = Kappa2Dist()
    # Show the window
    window.show()
