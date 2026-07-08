"""Extract per-burst photon streams for H2MM from ``.bur`` data + TTTR files.

A burst is stored as a ``.bur`` table row carrying ``First File`` /
``First Photon`` / ``Last Photon`` — integer photon indices into the raw
``tttrlib.TTTR`` object.  This module slices each burst, maps every photon to
an H2MM *stream* index from its routing channel and micro-time, and returns
the arrays the Numba engine consumes.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tttrlib

from .h2mm import BurstPhotons, prepare_bursts


@dataclass
class StreamDef:
    """Definition of one H2MM photon stream (a detector category).

    Attributes
    ----------
    name : str
        Human-readable stream name (e.g. ``"green"``, ``"red"``).
    channels : list of int
        TCSPC routing channel numbers assigned to this stream.
    micro_time_ranges : list of tuple[int, int]
        Inclusive micro-time windows (``[]`` accepts any micro time).
    """

    name: str
    channels: List[int]
    micro_time_ranges: List[Tuple[int, int]] = field(default_factory=list)


def default_streams() -> List[StreamDef]:
    """Return the canonical 2-colour donor/acceptor stream definition."""
    return [
        StreamDef("green", [0, 8], []),
        StreamDef("red", [1, 9], []),
    ]


def streams_from_dicts(items: Sequence[dict]) -> List[StreamDef]:
    """Build :class:`StreamDef` objects from JSON-compatible dictionaries."""
    out: List[StreamDef] = []
    for it in items:
        ranges = [(int(a), int(b)) for a, b in (it.get("micro_time_ranges") or [])]
        out.append(
            StreamDef(
                name=str(it.get("name", f"stream{len(out)}")),
                channels=[int(c) for c in it.get("channels", [])],
                micro_time_ranges=ranges,
            )
        )
    return out


def _stream_index_arrays(
    channels: np.ndarray,
    micro_times: np.ndarray,
    streams: Sequence[StreamDef],
) -> np.ndarray:
    """Map each photon to a stream index (``-1`` where no stream matches)."""
    idx = np.full(channels.shape[0], -1, dtype=np.int32)
    # Assign in reverse so earlier stream definitions take precedence.
    for s_i in range(len(streams) - 1, -1, -1):
        s = streams[s_i]
        mask = np.isin(channels, np.asarray(s.channels, dtype=channels.dtype))
        if s.micro_time_ranges:
            mt_mask = np.zeros(channels.shape[0], dtype=bool)
            for lo, hi in s.micro_time_ranges:
                mt_mask |= (micro_times >= lo) & (micro_times <= hi)
            mask &= mt_mask
        idx[mask] = s_i
    return idx


def extract_burst_photons(
    df: pd.DataFrame,
    tttrs: Dict[str, "tttrlib.TTTR"],
    streams: Sequence[StreamDef],
    time_scale: int = 1,
    min_photons: int = 3,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Slice bursts into per-burst ``(times, stream_index)`` arrays.

    Parameters
    ----------
    df : pandas.DataFrame
        Burst table with ``First File`` / ``First Photon`` / ``Last Photon``.
    tttrs : dict
        Maps the ``First File`` value to a ``tttrlib.TTTR`` object.
    streams : sequence of StreamDef
        Photon-stream definitions; photons matching no stream are dropped.
    time_scale : int
        Optional integer down-scaling of macro times (coarser base unit).
    min_photons : int
        Bursts with fewer assigned photons than this are skipped.

    Returns
    -------
    times : list of numpy.ndarray
        Per-burst monotonically non-decreasing integer macro times.
    stream_idx : list of numpy.ndarray
        Per-burst photon stream indices in ``[0, len(streams))``.
    """
    col_ff = df.columns.get_loc("First File")
    col_fp = df.columns.get_loc("First Photon")
    col_lp = df.columns.get_loc("Last Photon")

    cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ff, tttr in tttrs.items():
        cache[ff] = (
            np.asarray(tttr.macro_times),
            np.asarray(tttr.routing_channels),
            np.asarray(tttr.micro_times),
        )

    times_out: List[np.ndarray] = []
    streams_out: List[np.ndarray] = []
    for row in df.itertuples(index=False, name=None):
        ff = row[col_ff]
        if ff not in cache:
            continue
        first = int(row[col_fp])
        last = int(row[col_lp])
        if last <= first:
            continue
        macro, chan, micro = cache[ff]

        mt = macro[first:last]
        ch = chan[first:last]
        mi = micro[first:last]

        s_idx = _stream_index_arrays(ch, mi, streams)
        keep = s_idx >= 0
        if keep.sum() < min_photons:
            continue

        t = mt[keep].astype(np.int64)
        if time_scale > 1:
            t = t // time_scale
        s = s_idx[keep].astype(np.int32)

        # Enforce monotonicity (macro times are already sorted, but guard).
        order = np.argsort(t, kind="stable")
        times_out.append(t[order])
        streams_out.append(s[order])

    return times_out, streams_out


def bursts_from_dataframe(
    df: pd.DataFrame,
    tttrs: Dict[str, "tttrlib.TTTR"],
    streams: Sequence[StreamDef],
    time_scale: int = 1,
    min_photons: int = 3,
) -> BurstPhotons:
    """Return engine-ready :class:`BurstPhotons` from a burst DataFrame."""
    times, stream_idx = extract_burst_photons(
        df, tttrs, streams, time_scale=time_scale, min_photons=min_photons
    )
    if not times:
        raise ValueError("no bursts with enough stream-assigned photons")
    return prepare_bursts(times, stream_idx, n_streams=len(streams))


def load_bur_dataframe(paths: Sequence[str | pathlib.Path]) -> pd.DataFrame:
    """Read and concatenate one or more ``.bur`` files into a DataFrame."""
    from chisurf.core.fio.fluorescence.burst import read_bur_file

    frames = [read_bur_file(p) for p in paths]
    if not frames:
        raise ValueError("no .bur files provided")
    return pd.concat(frames, ignore_index=True)


def load_tttrs_for_dataframe(
    df: pd.DataFrame,
    data_dir: str | pathlib.Path,
    file_type: str = "SPC-130",
) -> Dict[str, "tttrlib.TTTR"]:
    """Load the TTTR object referenced by each unique ``First File`` value."""
    data_dir = pathlib.Path(data_dir)
    tttrs: Dict[str, "tttrlib.TTTR"] = {}
    for ff in df["First File"].unique():
        if ff in tttrs:
            continue
        candidate = pathlib.Path(ff)
        path = candidate if candidate.is_absolute() and candidate.exists() else data_dir / ff
        ftype = file_type
        if not ftype or ftype.lower() == "auto":
            ftype = tttrlib.inferTTTRFileType(str(path))
        tttrs[ff] = tttrlib.TTTR(str(path), ftype)
    return tttrs
