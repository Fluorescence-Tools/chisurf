"""Expose the MaxEnt Python API under the ``fmem`` namespace.

The actual implementation currently lives in
:mod:`chisurf.plugins.fluorescence_decay.maxent_decay.api`; this module
re-exports its public helpers so callers can rely on the stable
``...maxent_decay.fmem.api`` path.
"""

from __future__ import annotations

from ..api import (
    build_tau_grid,
    build_distance_grid,
    run_lifetime_mem_from_arrays,
    run_lifetime_mem_from_files,
    run_fret_mem_from_arrays,
    run_fret_mem_from_files,
)

__all__ = [
    "build_tau_grid",
    "build_distance_grid",
    "run_lifetime_mem_from_arrays",
    "run_lifetime_mem_from_files",
    "run_fret_mem_from_arrays",
    "run_fret_mem_from_files",
]
