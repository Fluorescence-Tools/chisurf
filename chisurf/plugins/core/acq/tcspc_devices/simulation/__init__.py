"""
Simulation TCSPC Device Module

This module provides a simulated TCSPC acquisition device using tttrlib's photon
simulator.
"""

from .wrapper import BurbulatorSimulator, SimulationDevice
from .core.streaming import TttrlibSimulator
from .setup_dialog import EnhancedSimulationSetupDialog as SimulationSetupDialog

__all__ = [
    'SimulationDevice',
    'TttrlibSimulator',
    'BurbulatorSimulator',
    'SimulationSetupDialog'
]
