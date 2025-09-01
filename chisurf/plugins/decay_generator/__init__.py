"""
Dye Diffusion Decay Generator

This plugin provides a graphical interface for simulating fluorescence decay curves 
based on dye diffusion models with dynamic quenching effects.

Features:
- Simulation of fluorescence decays with various diffusion parameters
- Modeling of dynamic quenching processes between fluorescent dyes and protein quenchers
- Calculation of fluorescence lifetime changes due to collisional quenching
- Configuration of dye properties, diffusion coefficients, and quenching parameters
- Accessible volume (AV) calculations to model dye movement constraints
- Interactive 3D visualization of dye trajectories and quencher positions
- Generation of realistic fluorescence decay histograms
- Export of simulated data for further analysis

Dynamic quenching occurs when a fluorescent dye in its excited state collides with 
a quencher molecule (typically certain amino acids like tryptophan, tyrosine, or 
histidine), resulting in non-radiative energy transfer that shortens the fluorescence 
lifetime. This process is distance-dependent and requires molecular diffusion.

The plugin simulates the Brownian motion of dye molecules attached to proteins, 
calculates collision events with quenchers, and generates the resulting fluorescence 
decay curves. These simulations are particularly valuable for:

- Interpreting experimental fluorescence decay data from protein-dye systems
- Understanding the relationship between protein structure and fluorescence quenching
- Designing fluorescence experiments with optimal dye-label positions
- Validating analytical models of dynamic quenching processes
- Studying the effects of diffusion constraints on fluorescence properties

The generated decay curves incorporate realistic photophysics including diffusion 
coefficients, quenching rates, and spatial constraints, providing a powerful tool 
for both experimental design and data interpretation in fluorescence spectroscopy.
"""
import sys
from quest.lib.tools.dye_diffusion import TransientDecayGenerator

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:Dye Diffusion"


# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the TransientDecayGenerator class
    window = TransientDecayGenerator()
    # Show the window
    window.show()
