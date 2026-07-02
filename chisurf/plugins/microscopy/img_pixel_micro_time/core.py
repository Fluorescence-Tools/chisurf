"""Qt-free compute for the mean-micro-time imaging plugin."""

from __future__ import annotations

from typing import Any

import numpy as np

from chisurf.core.fluorescence.imaging import (
    add_maps_to_hdf5,
    build_clsm,
    get_tttr,
)


def compute_mean_micro_time(
    filename: str,
    channels=(0,),
    n_ph_min: int = 2,
    microtime_resolution: float = -1.0,
) -> dict[str, Any]:
    """Compute a per-pixel mean-micro-time map from a TTTR imaging file.

    Parameters
    ----------
    filename : str
        Path to a TTTR imaging file (PTU/HT3/...); CLSM markers are auto-detected.
    channels : sequence of int
        Detector channel(s) to include.
    n_ph_min : int
        Minimum photons per pixel; pixels below are set to 0.
    microtime_resolution : float
        Micro-time resolution in ns for the output (``-1`` derives it from the
        header so the map is in nanoseconds; ``< 0`` after that keeps raw channel
        units).

    Returns
    -------
    dict
        ``{"maps": {mean_micro_time, intensity}, "shape": (ny, nx)}``.
    """
    tttr = get_tttr(filename)
    clsm = build_clsm(tttr, channels=tuple(channels))
    res_ns = float(microtime_resolution)
    if res_ns < 0.0:
        micro_res = float(getattr(tttr.header, "micro_time_resolution", 0.0) or 0.0)
        res_ns = micro_res * 1e9 if micro_res > 0.0 else -1.0
    mt = np.asarray(
        clsm.get_mean_micro_time(tttr, res_ns, int(n_ph_min), True), dtype=float
    )
    if mt.ndim == 3:
        mt = mt[0]
    mt = np.nan_to_num(mt)
    intensity = np.asarray(clsm.get_intensity(), dtype=float)
    intensity = intensity.sum(axis=0) if intensity.ndim == 3 else intensity
    maps = {"mean_micro_time": mt, "intensity": intensity}
    return {"maps": maps, "shape": mt.shape}


def add_mean_micro_time_to_hdf5(maps: dict[str, np.ndarray], path: str) -> list[str]:
    """Add the mean-micro-time field to a standard imaging HDF5 in place."""
    keep = {k: maps[k] for k in ("mean_micro_time",) if k in maps}
    return add_maps_to_hdf5(path, keep)
