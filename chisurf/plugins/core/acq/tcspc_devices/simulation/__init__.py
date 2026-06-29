"""
Simulation TCSPC Device Module

This module provides a simulation wrapper for TCSPC hardware using the Burbulator
single molecule diffusion simulator.
"""

from .wrapper import SimulationDevice, BurbulatorSimulator
from .setup_dialog import EnhancedSimulationSetupDialog as SimulationSetupDialog

__all__ = [
    'SimulationDevice',
    'BurbulatorSimulator',
    'SimulationSetupDialog'
]
