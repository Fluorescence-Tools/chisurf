"""Core MaxEnt MEM implementation package (fmem).

This subpackage provides a stable namespace for the maximum-entropy TCSPC
solvers, CLI, and Python API used by the MaxEnt decay plugin.

Typical imports::

    from chisurf.plugins.fluorescence_decay.maxent_decay.fmem import (
        solve_lifetime_mem,
        solve_fret_mem,
        run_lifetime_mem_from_arrays,
        run_fret_mem_from_arrays,
    )
"""

from __future__ import annotations

from .core import *  # noqa: F401,F403
from .api import (
    build_tau_grid,
    build_distance_grid,
    run_lifetime_mem_from_arrays,
    run_lifetime_mem_from_files,
    run_fret_mem_from_arrays,
    run_fret_mem_from_files,
)
from .sampling import sample_mem_distribution_emcee

__all__ = [
    # Core-level symbols (re-exported from .core)
    "MIN_PROB",
    "load_tcspc_two_column",
    "auto_fit_range_tcspc",
    "solve_lifetime_mem",
    "solve_fret_mem",
    # Python API helpers
    "build_tau_grid",
    "build_distance_grid",
    "run_lifetime_mem_from_arrays",
    "run_lifetime_mem_from_files",
    "run_fret_mem_from_arrays",
    "run_fret_mem_from_files",
    "sample_mem_distribution_emcee",
]
