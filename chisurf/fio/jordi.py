"""
Utilities for reading and writing Jordi files.

Jordi files are simple ASCII text files containing numeric data with optional metadata.
Supports flexible polarization channels (VV, VH, VM) with automatic format detection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, overload

import io
import numpy as np

ArrayLike = Union[np.ndarray, Iterable[float]]


def _parse_footer_metadata(footer_text: str) -> Dict[str, Any]:
    """Parse key-value pairs from footer text."""
    if not footer_text:
        return {}
    
    metadata = {}
    for line in footer_text.strip().split('\n'):
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass
            metadata[key] = value
    return metadata


def _normalize_jordi_array(data: ArrayLike) -> np.ndarray:
    """
    Normalize input into a 1D numpy array.
    
    For backward compatibility, supports legacy format where first half is VV
    and second half is VH. For the new format, channels are stored sequentially.
    """
    arr = np.asarray(data, dtype=float)
    return arr.reshape(-1)


def write_jordi(
    filename: Union[str, Path],
    vv: Optional[ArrayLike] = None,
    vh: Optional[ArrayLike] = None,
    vm: Optional[ArrayLike] = None,
    data: Optional[ArrayLike] = None,
    g_factor: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    fmt: Optional[str] = None,
    overwrite: bool = True,
    create_dirs: bool = True,
    newline: str = "\n",
    header: Optional[str] = None,
    delimiter: Optional[str] = None,
    footer: Optional[Union[str, Iterable[str]]] = None,
) -> Path:
    """
    Write Jordi data with flexible channel support.

    Parameters
    ----------
    filename : str or Path
        Output file path.
    vv, vh, vm : array-like, optional
        Individual channel data. At least one must be provided unless using 'data'.
    data : array-like, optional
        Legacy format: concatenated array (for backward compatibility).
    g_factor : float, optional
        G-factor used for anisotropy calculations.
    metadata : dict, optional
        Additional metadata to store in the footer.
    **kwargs
        Additional arguments passed to numpy.savetxt.

    Returns
    -------
    Path
        The path to the written file.
    """
    # Collect provided channels
    channels = {}
    if vv is not None:
        channels['VV'] = np.asarray(vv, dtype=float)
    if vh is not None:
        channels['VH'] = np.asarray(vh, dtype=float)
    if vm is not None:
        channels['VM'] = np.asarray(vm, dtype=float)
    
    # Handle legacy data parameter
    if data is not None:
        if channels:
            raise ValueError("Cannot specify both 'data' and individual channels (vv/vh/vm)")
        arr = _normalize_jordi_array(data)
        half = len(arr) // 2
        channels = {'VV': arr[:half], 'VH': arr[half:]}
    
    if not channels:
        raise ValueError("At least one channel (vv, vh, vm) or data must be provided")
    
    # Validate lengths
    lengths = [len(v) for v in channels.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"All channels must have same length. Got: {dict(zip(channels.keys(), lengths))}")
    
    # Build metadata
    channel_list = list(channels.keys())
    format_version = "2.0" if len(channel_list) != 2 or set(channel_list) != {'VV', 'VH'} else "1.0"
    
    footer_lines = [
        f"format_version: {format_version}",
        f"channels: {', '.join(channel_list)}"
    ]
    
    if g_factor is not None:
        footer_lines.append(f"g_factor: {g_factor}")
    
    if metadata:
        for key, value in metadata.items():
            footer_lines.append(f"{key}: {value}")
    
    # Concatenate data in channel order
    vec = np.concatenate([channels[ch] for ch in channel_list])
    
    # Ensure parent directories exist
    path = Path(filename)
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and handle overwrite
    if path.exists() and not overwrite:
        raise FileExistsError(f"File {path} already exists and overwrite=False")
    
    # Prepare header and footer
    if header is None:
        header = ""
    
    # Process footer
    if footer is not None:
        if isinstance(footer, (list, tuple)):
            footer = newline.join(str(line) for line in footer)
        footer = newline.join(f"#{line}" for line in footer_lines) + newline + footer if footer else newline.join(f"#{line}" for line in footer_lines)
    else:
        footer = newline.join(f"#{line}" for line in footer_lines) if footer_lines else None
    
    # Write the data to file
    np.savetxt(
        path,
        vec.reshape(-1, 1),  # Save as column vector
        fmt=fmt or '%.6f',
        delimiter=delimiter or '\t',
        header=header,
        footer=footer if footer else '',  # Ensure empty string instead of None
        comments='# ',
        newline=newline
    )
    
    return path


@overload
def read_jordi(
    filename: Union[str, Path],
    channels: None = None,
    split: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]: ...

@overload
def read_jordi(
    filename: Union[str, Path],
    channels: Union[str, List[str]],
    split: bool = True,
    return_metadata: bool = False,
    **kwargs
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, Any]]]: ...

def read_jordi(
    filename: Union[str, Path],
    channels: Optional[Union[str, List[str]]] = None,
    split: bool = False,
    return_metadata: bool = False,
    **kwargs
):
    """
    Read Jordi data with flexible channel support.

    Parameters
    ----------
    filename : str or Path
        Input file path.
    channels : str or list of str, optional
        Which channels to return. If None, returns all channels.
    split : bool, default False
        If True, returns a dictionary of {channel_name: array}.
        If False, returns a concatenated array.
    return_metadata : bool, default False
        If True, returns a tuple of (data, metadata).
    **kwargs
        Additional arguments passed to numpy.loadtxt.

    Returns
    -------
    np.ndarray or dict or tuple
        If split=False: concatenated 1D array
        If split=True: dict like {'VV': array, 'VH': array, ...}
        If return_metadata=True: returns (data, metadata)
    """
    path = Path(filename)
    
    # Read file with legacy function
    arr, footer_text = _read_jordi_legacy(path, return_footer=True, **kwargs)
    
    # Parse metadata from footer
    metadata = _parse_footer_metadata(footer_text)
    
    # Get channel list from metadata or infer from array length
    channel_list = metadata.get('channels', None)
    if channel_list is None:
        # Legacy format: try to infer channels
        if len(arr) % 2 == 0:
            channel_list = ['VV', 'VH']  # Default legacy format
        else:
            channel_list = ['VM']  # Single channel if odd length
    
    # Handle case where channel_list is a string
    if isinstance(channel_list, str):
        channel_list = [ch.strip() for ch in channel_list.split(',') if ch.strip()]
    
    # Calculate points per channel
    n_channels = len(channel_list)
    if n_channels == 0:
        raise ValueError("No channels found in file")
    
    n_points = len(arr) // n_channels
    if n_points * n_channels != len(arr):
        raise ValueError(
            f"Array length {len(arr)} is not divisible by number of channels {n_channels}"
        )
    
    # Split array into channels
    channel_data = {
        ch: arr[i * n_points:(i + 1) * n_points].copy()
        for i, ch in enumerate(channel_list)
    }
    
    # Filter channels if requested
    if channels is not None:
        if isinstance(channels, str):
            channels = [channels]
        channel_data = {k: v for k, v in channel_data.items() if k in channels}
    
    # Handle return format
    if not split:
        # Return concatenated array
        result = np.concatenate(list(channel_data.values()))
    else:
        # Return dictionary of channels
        result = channel_data
    
    # Backward compatibility: return tuple for VV/VH if in legacy format
    if (not return_metadata and split and 
        set(channel_list) == {'VV', 'VH'} and 
        channels is None and
        metadata.get('format_version', '1.0') == '1.0'):
        return channel_data['VV'], channel_data['VH']
    
    if return_metadata:
        return result, metadata
    return result


def _read_jordi_legacy(
    filename: Union[str, Path],
    split: bool = False,
    dtype=float,
    delimiter: Optional[str] = None,
    comments: str = "#",
    return_footer: bool = False,
):
    """
    Original read_jordi implementation, now for internal use.
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
        footer_lines: List[str] = []
    else:
        data_lines = lines[:sep_idx]
        footer_lines = lines[sep_idx + 1:]

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

    # Normalize to 1D array
    arr = _normalize_jordi_array(arr)

    if return_footer:
        return arr, footer_text
    return arr
