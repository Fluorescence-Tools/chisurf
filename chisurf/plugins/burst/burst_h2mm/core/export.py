"""Defined, portable H2MM result tables — openable directly in ndxplorer (ndX).

The analysis produces a Viterbi state per photon; joined with the per-photon
metadata (:class:`~.photons.PhotonMeta`) this becomes a **per-photon table** and,
aggregated, a **per-burst table**. Both are written as plain **numeric** tables
(one row per point) that ndX opens through its generic CSV / MFD-HDF5 readers:

* the time axis is named ``"Mean Macro Time (s)"`` — the column ndX auto-selects
  as an axis;
* ``State`` (the H2MM Viterbi assignment) is an integer column that ndX reads as
  a categorical colour / Z axis;
* everything is numeric, so nothing is dropped on import.

HDF5 is written with pandas ``HDFStore`` under key ``"results"`` (the key ndX's
reader looks for first); CSV is a comma-delimited header + numeric rows.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

from .h2mm import BurstPhotons
from .photons import PhotonMeta

NDX_HDF5_KEY = "results"


@dataclass
class H2mmTables:
    """The per-photon and per-burst result tables (as pandas DataFrames)."""

    photons: object  # pandas.DataFrame — one row per photon
    bursts: object   # pandas.DataFrame — one row per burst


def build_tables(
    data: BurstPhotons,
    meta: PhotonMeta,
    path: np.ndarray,
    fret: np.ndarray,
    base_time_s: float,
) -> H2mmTables:
    """Assemble the per-photon and per-burst ndX tables.

    Parameters
    ----------
    data : BurstPhotons
        Engine-layout photon data (provides ``streams``/burst offsets).
    meta : PhotonMeta
        Per-photon macro/micro/channel/burst arrays aligned with ``data``.
    path : numpy.ndarray
        Per-photon Viterbi state (length ``N``, from :func:`~.h2mm.viterbi`).
    fret : numpy.ndarray
        Per-state apparent FRET efficiency (for the burst-level mean-E column).
    base_time_s : float
        Seconds per base time unit (macro-time → seconds).

    Returns
    -------
    H2mmTables
        ``photons`` (one row per photon) and ``bursts`` (one row per burst).
    """
    import pandas as pd

    macro_s = meta.macro_time.astype(np.float64) * float(base_time_s)
    photons = pd.DataFrame(
        {
            "Mean Macro Time (s)": macro_s,   # ndX auto-axis name
            "Macro Time": meta.macro_time,
            "Micro Time": meta.micro_time,
            "Channel": meta.channel,
            "Stream": data.streams.astype(np.int64),
            "State": path.astype(np.int64),
            "Burst": meta.burst_id,
        }
    )

    # Per-burst aggregation.
    offsets = data.burst_offsets
    n_bursts = data.n_bursts
    n_states = int(fret.shape[0])
    rows = []
    fret_arr = np.asarray(fret, dtype=np.float64)
    for b in range(n_bursts):
        s = int(offsets[b])
        e = int(offsets[b + 1])
        seg = path[s:e]
        occ = np.bincount(seg, minlength=n_states).astype(np.float64)
        dominant = int(np.argmax(occ))
        n_trans = int(np.count_nonzero(np.diff(seg)))
        mean_e = float((occ * fret_arr).sum() / occ.sum()) if occ.sum() > 0 else np.nan
        t0 = float(meta.macro_time[s]) * base_time_s
        rows.append((b, e - s, t0, dominant, n_trans, mean_e))
    bursts = pd.DataFrame(
        rows,
        columns=[
            "Burst",
            "Number of Photons",     # ndX auto-uses this as a histogram weight
            "Mean Macro Time (s)",
            "Dominant State",
            "Number of Transitions",
            "Mean FRET E",
        ],
    )
    return H2mmTables(photons=photons, bursts=bursts)


def write_hdf5(df, path: str | pathlib.Path, key: str = NDX_HDF5_KEY) -> str:
    """Write a table to an ndX-openable HDF5 file (``key='results'``)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(str(path), key=key, format="table", mode="w")
    return str(path)


def write_csv(df, path: str | pathlib.Path) -> str:
    """Write a table to an ndX-openable CSV file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=False)
    return str(path)
