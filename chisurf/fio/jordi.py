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

import io
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
    footer: Optional[Union[str, Iterable[str]]] = None,
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
    footer
        Optional footer text to append after a separator row containing a negative number ("-1").
        If a sequence of strings is provided, each element is written as a line.

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

    # If a footer is provided, append a separator with a negative number and then the footer text
    if footer is not None:
        with out_path.open("a", encoding="utf-8", newline="") as fh:
            # Ensure there is a newline before the separator if the file does not end with one
            # numpy.savetxt typically ends with a newline already, so we just write the separator
            fh.write(f"-1{newline}")
            if isinstance(footer, str):
                # Write as-is, but ensure it ends with a newline
                if footer and not footer.endswith(("\n", "\r")):
                    fh.write(footer + newline)
                else:
                    fh.write(footer)
            else:
                for line in footer:
                    line = "" if line is None else str(line)
                    if line.endswith(("\n", "\r")):
                        fh.write(line)
                    else:
                        fh.write(line + newline)

    return out_path



def read_jordi(
    filename,
    split: bool = False,
    dtype=float,
    delimiter: str | None = None,
    comments: str = "#",
    return_footer: bool = False,
):
    """
    Read Jordi data from an ASCII file with optional footer.

    The file is split at the first non-comment data row that contains any negative
    numeric value. All rows before that separator are parsed as conventional Jordi
    numeric data. All rows after the separator constitute the footer (can contain
    arbitrary metadata). The separator row itself is not included in the data nor
    the footer.

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
    return_footer : bool, default False
        When True, also return the footer string. If split is True, returns
        (cp, cs, footer); otherwise returns (arr, footer).

    Returns
    -------
    np.ndarray | tuple
        The full Jordi vector or a tuple of (parallel, perpendicular). If
        return_footer=True, the footer string is appended to the return value.
    """
    path = Path(filename)

    # Read raw lines
    try:
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except UnicodeDecodeError:
        # Fallback without explicit encoding
        with path.open("r") as fh:
            lines = fh.readlines()

    # Detect the first non-comment row that contains a negative number
    sep_idx = None
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue  # ignore blank rows for detection
        if comments and s.startswith(comments):
            continue  # ignore comment rows for detection
        # Tokenize and try to parse numbers
        tokens = s.split(delimiter) if delimiter else s.split()
        if not tokens:
            continue
        neg_found = False
        all_numeric = True
        for tok in tokens:
            try:
                val = float(tok)
            except Exception:
                all_numeric = False
                break
            if val < 0:
                neg_found = True
        if all_numeric and neg_found:
            sep_idx = idx
            break

    # Build data and footer sections (separator line excluded from both)
    if sep_idx is None:
        data_lines = lines
        footer_lines: list[str] = []
    else:
        data_lines = lines[:sep_idx]
        footer_lines = lines[sep_idx + 1 :]

    footer_text = "".join(footer_lines)

    # Parse numeric data section with numpy.loadtxt using an in-memory buffer
    data_text = "".join(data_lines)
    if data_text.strip():
        try:
            arr = np.loadtxt(io.StringIO(data_text), dtype=dtype, delimiter=delimiter, comments=comments)
        except Exception:
            # If parsing fails (e.g., no numeric data), return empty array
            arr = np.array([], dtype=dtype)
    else:
        arr = np.array([], dtype=dtype)

    # Normalize shape: accept (N,2) or (2,N) and convert to 1D concatenated
    if arr.ndim == 2:
        r, c = arr.shape
        if c == 2:
            arr = np.hstack([arr[:, 0], arr[:, 1]])
        elif r == 2:
            arr = np.hstack([arr[0, :], arr[1, :]])
        else:
            arr = arr.ravel()
    elif arr.ndim != 1:
        arr = arr.ravel()

    # If odd length, splitting is not reliable
    if len(arr) % 2 != 0:
        if split:
            raise ValueError(f"Jordi vector must have even length to split. Got length {len(arr)} from {path}.")
        if return_footer:
            return arr, footer_text
        return arr

    if split:
        half = len(arr) // 2
        cp, cs = arr[:half], arr[half:]
        if return_footer:
            return cp, cs, footer_text
        return cp, cs
    else:
        if return_footer:
            return arr, footer_text
        return arr
