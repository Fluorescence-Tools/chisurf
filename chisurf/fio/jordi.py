"""
Utilities for reading and writing Jordi files.

Jordi files are simple ASCII text files containing a single 1D numeric vector.
Convention used throughout ChiSurf:
- First half: Parallel (VV) decay histogram
- Second half: Perpendicular (VH) decay histogram

This module provides a single entry point for writing Jordi files to ensure
consistent formatting and future extensibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray, Iterable[float]]


def _normalize_jordi_array(data: ArrayLike) -> np.ndarray:
    """
    Normalize input into a 1D numpy array that follows the Jordi convention
    (first half = parallel, second half = perpendicular).

    Accepted input shapes:
    - 1D array: assumed already in (parallel, perpendicular) concatenated form.
    - 2D array with shape (2, N): rows are (parallel, perpendicular).
    - 2D array with shape (N, 2): columns are (parallel, perpendicular).

    Returns
    -------
    np.ndarray
        1D array of length 2*N.
    """
    arr = np.asarray(data)
    if arr.ndim == 1:
        # assume already concatenated [cp, cs]
        return arr.astype(float)

    if arr.ndim == 2:
        r, c = arr.shape
        if r == 2:
            # rows: [cp; cs]
            return np.hstack([arr[0], arr[1]]).astype(float)
        if c == 2:
            # cols: [cp, cs]
            return np.hstack([arr[:, 0], arr[:, 1]]).astype(float)

    raise ValueError(
        "Jordi data must be 1D or a 2D array with shape (2, N) or (N, 2). "
        f"Got shape {arr.shape}."
    )


def write_jordi(
    data: ArrayLike,
    filename: Union[str, Path],
    fmt: Optional[str] = None,
    overwrite: bool = True,
    create_dirs: bool = True,
    newline: str = "\n",
    comments: str = "",
    header: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> Path:
    """
    Write Jordi data to an ASCII file.

    Parameters
    ----------
    data
        Jordi data. Can be 1D array in concatenated form, or 2D as (2, N) or (N, 2).
    filename
        Target file path. Directory is created if create_dirs is True.
    fmt
        Format string for numpy.savetxt. If None, numpy default is used.
    overwrite
        If False and file exists, raises FileExistsError.
    create_dirs
        Create parent directories if they do not exist.
    newline, comments, header
        Passed to numpy.savetxt for flexibility.

    Returns
    -------
    Path
        The path to the written file.
    """
    out_path = Path(filename)
    if create_dirs:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite=False: {out_path}")

    vec = _normalize_jordi_array(data)

    # Prepare kwargs for numpy.savetxt
    kwargs = {
        "newline": newline,
        "comments": comments,
        "header": header or "",
    }
    if fmt is not None:
        kwargs["fmt"] = fmt
    if delimiter is not None:
        kwargs["delimiter"] = delimiter

    # Use numpy.savetxt with optional formatting
    # Keep defaults close to prior usage (no specific fmt provided previously)
    np.savetxt(out_path.as_posix(), vec, **kwargs)

    return out_path



def read_jordi(
    filename,
    split: bool = False,
    dtype=float,
    delimiter: str | None = None,
    comments: str = "#",
):
    """
    Read Jordi data from an ASCII file.

    Parameters
    ----------
    filename : str | Path
        Path to the input file.
    split : bool, default False
        If True, return a tuple (parallel, perpendicular) where each is 1D array.
        Otherwise return the 1D concatenated Jordi vector.
    dtype : data-type, optional
        Data type of the resulting array.
    delimiter : str, optional
        The string used to separate values. If None, will try to infer.
    comments : str, default '#'
        The character used to indicate the start of a comment.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        The full Jordi vector or a tuple of (parallel, perpendicular).
    """
    path = Path(filename)
    arr = np.loadtxt(path.as_posix(), dtype=dtype, delimiter=delimiter, comments=comments)

    # Handle possible 2D shapes gracefully: (N,2) or (2,N) → concatenate [cp, cs]
    if arr.ndim == 2:
        r, c = arr.shape
        if c == 2:
            arr = np.hstack([arr[:, 0], arr[:, 1]])
        elif r == 2:
            arr = np.hstack([arr[0, :], arr[1, :]])
        else:
            # Flatten as fallback
            arr = arr.ravel()

    if arr.ndim != 1:
        arr = arr.ravel()

    if len(arr) % 2 != 0:
        # If odd length, we can't split reliably; still return raw if split=False
        if split:
            raise ValueError(f"Jordi vector must have even length to split. Got length {len(arr)} from {path}.")
        return arr

    if split:
        half = len(arr) // 2
        return arr[:half], arr[half:]
    else:
        return arr
