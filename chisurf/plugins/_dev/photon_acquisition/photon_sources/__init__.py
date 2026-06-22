"""Photon acquisition sources for the SM acquisition plugin.

This package provides a user-facing alias for the older
``tcspc_devices`` package. It re-exports the common abstractions and
setup dialogs so that new code can import from
``chisurf.plugins._dev.photon_acquisition.photon_sources`` while existing
code that still uses ``tcspc_devices`` continues to work.
"""

from __future__ import annotations

from ..tcspc_devices import (
    TCSPCDeviceABC,
    TCSPCDevice,
    BHSPCCardSetupDialog,
    SimulationSetupDialog,
    PicoQuantSetupDialog,
)

__all__ = [
    "TCSPCDeviceABC",
    "TCSPCDevice",
    "BHSPCCardSetupDialog",
    "SimulationSetupDialog",
    "PicoQuantSetupDialog",
]
