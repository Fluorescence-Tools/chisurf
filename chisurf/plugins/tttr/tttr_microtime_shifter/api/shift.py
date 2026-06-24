"""Pure micro-time shift logic (no GUI, no MFDB)."""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import tttrlib


def safe_tttr_path(path: str) -> str:
    """Resolve a TTTR file path for tttrlib, handling non-ASCII paths.

    On Windows, if the path contains non-ASCII characters, the file is
    temporarily renamed to an ASCII-only name so that tttrlib can open it.

    Parameters
    ----------
    path : str
        Original file path.

    Returns
    -------
    str
        A path that tttrlib can open.

    """
    if sys.platform != 'win32' or all(ord(c) < 128 for c in path):
        return path
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    ext = os.path.splitext(path)[1]
    temp_path = os.path.join(folder, f"tttr_{uuid.uuid4().hex}{ext}")
    try:
        os.rename(path, temp_path)
    except Exception:
        return path
    return temp_path


def restore_tttr_path(original: str, temp: str) -> None:
    """Restore a TTTR file path after ``safe_tttr_path``.

    Parameters
    ----------
    original : str
        Original file path.
    temp : str
        Temporary file path returned by ``safe_tttr_path``.

    """
    if original == temp:
        return
    try:
        os.rename(temp, original)
    except Exception:
        pass


def _load_tttr(path: str, filetype: str | None = None) -> tttrlib.TTTR:
    """Load a TTTR file, handling non-ASCII paths.

    Parameters
    ----------
    path : str
        File path.
    filetype : str, optional
        Explicit file type for tttrlib.

    Returns
    -------
    tttrlib.TTTR
        Loaded TTTR object.

    """
    safe = safe_tttr_path(path)
    try:
        tt = tttrlib.TTTR(safe, filetype=filetype)
    finally:
        if safe != path:
            restore_tttr_path(path, safe)
    return tt


def _compute_effective_shifts(
    global_shift: int,
    channel_shifts: dict[int, int],
    routing_channels: np.ndarray,
    n_mt: int,
) -> dict[int, int]:
    """Compute the effective per-channel shift modulo *n_mt*.

    Parameters
    ----------
    global_shift : int
        Global shift applied to all channels.
    channel_shifts : dict
        Per-channel shifts.
    routing_channels : numpy.ndarray
        Array of routing channel values.
    n_mt : int
        Number of micro-time channels.

    Returns
    -------
    dict
        Mapping from routing channel to effective shift value.

    """
    used = sorted(set(int(c) for c in routing_channels))
    effective: dict[int, int] = {}
    for ch in used:
        per_ch = channel_shifts.get(ch, 0)
        effective[ch] = (global_shift + per_ch) % n_mt
    return effective


def _apply_shifts(
    tt: tttrlib.TTTR,
    global_shift: int,
    channel_shifts: dict[int, int],
) -> dict[int, int]:
    """Apply micro-time shifts to a TTTR object in place.

    Parameters
    ----------
    tt : tttrlib.TTTR
        TTTR object to shift.
    global_shift : int
        Global shift applied to all channels.
    channel_shifts : dict
        Per-channel shifts.

    Returns
    -------
    dict
        Mapping from routing channel to the effective shift applied.

    """
    n_mt = tt.header.get_effective_number_of_micro_time_channels()
    routing = tt.routing_channels
    used = sorted(set(int(c) for c in routing))
    effective: dict[int, int] = {}
    for ch in used:
        per_ch = channel_shifts.get(ch, 0)
        tot = (global_shift + per_ch) % n_mt
        if tot != 0:
            tt.shift_micro_time_by_channel(int(ch), int(tot))
        effective[int(ch)] = int(tot)
    return effective


def shift_file(
    path: str,
    *,
    global_shift: int = 0,
    channel_shifts: dict[int, int] | None = None,
    filetype: str | None = None,
    output_dir: str | None = None,
) -> tuple[str, dict[int, int]]:
    """Shift micro-times in a TTTR file and write the result.

    This is the pure API entry point.  No GUI or MFDB imports.

    Parameters
    ----------
    path : str
        Input TTTR file path.
    global_shift : int
        Global shift applied to all routing channels.
    channel_shifts : dict, optional
        Per-channel shifts.
    filetype : str, optional
        Explicit tttrlib file type.
    output_dir : str, optional
        Output directory.  Defaults to the input file directory.

    Returns
    -------
    tuple of (str, dict)
        Output file path and the ``{channel: effective_shift}`` mapping.

    """
    channel_shifts = channel_shifts or {}
    tt = _load_tttr(path, filetype=filetype)
    applied = _apply_shifts(tt, global_shift, channel_shifts)

    stem = Path(path).stem
    ext = Path(path).suffix
    out_dir = Path(output_dir) if output_dir else Path(path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = str(out_dir / f"{stem}_shifted{ext}")

    temp_out = str(out_dir / f"save_{uuid.uuid4().hex}{ext}")
    tt.write(temp_out)
    if os.path.exists(out_path):
        os.remove(out_path)
    shutil.move(temp_out, out_path)

    return out_path, applied


def load_file_metadata(path: str) -> dict[str, Any]:
    """Load metadata from a TTTR file (no shifts applied).

    Parameters
    ----------
    path : str
        TTTR file path.

    Returns
    -------
    dict
        Metadata dict with keys ``routing_channels``, ``n_mt``, ``n_photons``.

    """
    tt = _load_tttr(path)
    return {
        "routing_channels": [int(c) for c in sorted(set(int(x) for x in tt.routing_channels))],
        "n_mt": int(tt.header.get_effective_number_of_micro_time_channels()),
        "n_photons": int(len(tt)),
    }


def load_histogram(
    path: str | list[str],
    *,
    global_shift: int = 0,
    channel_shifts: dict[int, int] | None = None,
    filetype: str | None = None,
) -> dict[str, Any]:
    """Load shifted micro-time histogram data for GUI preview.

    Parameters
    ----------
    path : str or list of str
        TTTR file path(s).
    global_shift : int
        Global shift applied to all routing channels.
    channel_shifts : dict, optional
        Per-channel shifts.
    filetype : str, optional
        Explicit tttrlib file type.

    Returns
    -------
    dict
        Histogram payload with ``n_mt``, ``routing_channels``, and
        ``histograms`` keys.

    """
    paths = [path] if isinstance(path, str) else list(path)
    channel_shifts = channel_shifts or {}
    
    total_histograms: dict[str, np.ndarray] = {}
    n_mt = 0
    all_used = set()
    
    for p in paths:
        try:
            tt = _load_tttr(p, filetype=filetype)
            file_n_mt = int(tt.header.get_effective_number_of_micro_time_channels())
            if n_mt == 0:
                n_mt = file_n_mt
            elif n_mt != file_n_mt:
                continue
                
            routing = tt.routing_channels
            micro_times = tt.micro_times
            used = set(int(c) for c in routing)
            all_used.update(used)
            
            for ch in used:
                mask = routing == ch
                if not np.any(mask):
                    continue
                effective = (global_shift + int(channel_shifts.get(ch, 0))) % n_mt
                shifted = (micro_times[mask] + effective) % n_mt
                hist = np.bincount(shifted, minlength=n_mt)
                
                ch_str = str(ch)
                if ch_str in total_histograms:
                    total_histograms[ch_str] += hist
                else:
                    total_histograms[ch_str] = hist
        except Exception:
            continue

    histograms_serialized = {
        k: [int(v) for v in arr]
        for k, arr in total_histograms.items()
    }

    return {
        "n_mt": n_mt,
        "routing_channels": sorted(list(all_used)),
        "histograms": histograms_serialized,
    }
