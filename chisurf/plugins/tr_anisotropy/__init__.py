"""
Time-Resolved Anisotropy Plugin

This plugin provides tools for analyzing time-resolved fluorescence anisotropy data.
It allows users to:
1. Load and process polarization-resolved fluorescence decay data
2. Set up and visualize rotation spectra and lifetime components
3. Create and manage anisotropy fits with multiple rotation correlation times
4. Analyze rotational diffusion of fluorophores in different environments

Time-resolved anisotropy is a powerful technique for studying the rotational motion
of fluorophores, providing insights into molecular size, shape, flexibility, and
interactions. This plugin implements a wizard-based interface that guides users
through the process of setting up and analyzing anisotropy decay data, from data
loading to model fitting and result visualization.

The plugin supports multiple rotation correlation times and lifetime components,
making it suitable for analyzing complex systems with heterogeneous rotational
dynamics or multiple fluorophore populations.
"""

name = "Tools:Anisotropy-Wizard"

from .wizard import *
