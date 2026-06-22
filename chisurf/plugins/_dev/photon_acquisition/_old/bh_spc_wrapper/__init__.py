"""
BH SPC Wrapper Module

This module provides a wrapper for the Becker & Hickl SPC hardware.
It includes classes and functions for initializing and controlling
BH SPC devices, as well as for acquiring data from them.

The module is designed to be manufacturer-agnostic, allowing for
future extension to other hardware (e.g., Picoquant).
"""

try:
    from .wrapper import (
        DLLOperationMode,
        InitStatus,
        ParID,
        SPCMError,
        BHSPC,
        minimal_spcm_ini,
        ini_file,
        TCSPCDevice
    )
except Exception:
    # Legacy/dev-only hardware wrapper; allow import to succeed even if the
    # underlying implementation cannot be imported in this environment.
    DLLOperationMode = InitStatus = ParID = SPCMError = BHSPC = None
    minimal_spcm_ini = ini_file = TCSPCDevice = None

try:
    from .card_setup_dialog import BHSPCCardSetupDialog
except Exception:
    BHSPCCardSetupDialog = None

try:
    from .reader import BeckerHicklSPCSetupReader
except Exception:
    BeckerHicklSPCSetupReader = None

__all__ = [
    'DLLOperationMode',
    'InitStatus',
    'ParID',
    'SPCMError',
    'BHSPC',
    'minimal_spcm_ini',
    'ini_file',
    'TCSPCDevice',
    'BHSPCCardSetupDialog',
    'BeckerHicklSPCSetupReader'
]
