"""
BH SPC Wrapper Module

This module provides a wrapper for the Becker & Hickl SPC hardware.
It includes classes and functions for initializing and controlling
BH SPC devices, as well as for acquiring data from them.

The module is designed to be manufacturer-agnostic, allowing for
future extension to other hardware (e.g., Picoquant).
"""

from .wrapper import (
    DLLOperationMode,
    InitStatus,
    ParID,
    SPCMError,
    BHSPC,
    minimal_spcm_ini,
    ini_file,
    BHSPCDevice
)
from .card_setup_dialog import BHSPCCardSetupDialog
from .reader import BeckerHicklSPCSetupReader

__all__ = [
    'DLLOperationMode',
    'InitStatus',
    'ParID',
    'SPCMError',
    'BHSPC',
    'minimal_spcm_ini',
    'ini_file',
    'BHSPCDevice',
    'BHSPCCardSetupDialog',
    'BeckerHicklSPCSetupReader'
]
