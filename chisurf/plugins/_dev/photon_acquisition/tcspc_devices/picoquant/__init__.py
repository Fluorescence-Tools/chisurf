"""
PicoQuant Wrapper Module

This module provides a wrapper for PicoQuant TCSPC hardware using snAPI.
It includes classes and functions for initializing and controlling
PicoQuant devices, as well as for acquiring data from them.

The module is designed to be manufacturer-agnostic, allowing for
future extension to other hardware.
"""

from .wrapper import PicoQuantAPI, PicoQuantDevice
from .setup_dialog import PicoQuantSetupDialog

__all__ = [
    'PicoQuantAPI',
    'PicoQuantDevice',
    'PicoQuantSetupDialog'
]
