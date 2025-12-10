"""Compatibility wrapper for the MaxEnt core solvers.

This module re-exports the implementation from the plugin root so code can
import from::

    chisurf.plugins.fluorescence_decay.maxent_decay.fmem.core

without changing the actual numerical implementation location.
"""

from __future__ import annotations

from ..core import *  # noqa: F401,F403
