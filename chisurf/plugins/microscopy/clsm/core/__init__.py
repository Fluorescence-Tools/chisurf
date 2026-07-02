"""Qt-free computation layer for the CLSM plugin.

Everything in this package is pure NumPy/SciPy/Numba plus ``tttrlib`` (imported
lazily at call time).  No Qt, no chisurf-GUI imports — safe to call from the
CLI, the RPC services, and headless tests.
"""

from __future__ import annotations

from .frc import compute_frc, counting_noise, gaussian_kernel
from .imaging import (
    brush_kernel,
    build_clsm_image,
    decay_of_selection,
    reduce_frames,
    representation,
)
from .setups import builtin_setups, read_clsm_markers

__all__ = [
    "compute_frc",
    "counting_noise",
    "gaussian_kernel",
    "brush_kernel",
    "build_clsm_image",
    "decay_of_selection",
    "reduce_frames",
    "representation",
    "builtin_setups",
    "read_clsm_markers",
]
