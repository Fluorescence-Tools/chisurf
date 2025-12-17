from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def build_idx_map(
    intervals_by_file: Dict[str, Sequence[Tuple[int, int]]],
    *,
    inclusive_stop: bool = True,     # True -> [start, stop]; False -> [start, stop)
    dtype=np.int64                   # use np.int32 if you want to save RAM
) -> Dict[str, np.ndarray]:
    """Build per-file index arrays from non-overlapping intervals.

    Parameters
    ----------
    intervals_by_file : dict
        Mapping ``filename`` to a sequence of ``(start, stop)`` integer
        pairs. Intervals are assumed to be non-overlapping and
        ``start < stop``.
    inclusive_stop : bool, optional
        If ``True`` (default), the stop index is treated as inclusive,
        i.e. an interval ``(0, 2)`` expands to indices ``[0, 1, 2]``.
        If ``False``, the stop index is exclusive as in standard Python
        slicing.
    dtype : data-type, optional
        Integer dtype for the resulting index arrays (default
        ``numpy.int64``).

    Returns
    -------
    dict
        A dictionary mapping each input filename to a one-dimensional
        :class:`numpy.ndarray` of indices. Files with no intervals are
        mapped to empty arrays.

    Examples
    --------
    Build index arrays for two files with simple intervals::

        >>> from chisurf.experiments.pda import build_idx_map
        >>> ivals = {"a": [(0, 2)], "b": [(5, 6)]}
        >>> out = build_idx_map(ivals)
        >>> sorted(out.keys())
        ['a', 'b']
        >>> out['a'].tolist()
        [0, 1, 2]
        >>> out['b'].tolist()
        [5, 6]
    """
    if not intervals_by_file:
        return {}

    # Prepare a stable ordering of files (preserve input dict order)
    file_names = list(intervals_by_file.keys())

    # Pre-create output with empty arrays for files that may have no intervals
    out: Dict[str, np.ndarray] = {fn: np.empty(0, dtype=dtype) for fn in file_names}

    # Flatten all intervals into global arrays
    starts_list = []
    stops_list = []
    file_ids = []  # one id per interval

    for fid, fname in enumerate(file_names):
        ivals = intervals_by_file.get(fname) or []
        if not ivals:
            continue
        n = len(ivals)
        starts_list.append(np.fromiter((int(s) for s, _ in ivals), count=n, dtype=dtype))
        stops_list.append(np.fromiter((int(e) for _, e in ivals), count=n, dtype=dtype))
        file_ids.append(np.full(n, fid, dtype=np.int64))

    if not starts_list:
        return out  # nothing to do

    starts = np.concatenate(starts_list, axis=0)
    stops = np.concatenate(stops_list, axis=0)
    file_ids_per_interval = np.concatenate(file_ids, axis=0)

    # Compute segment lengths (inclusive or exclusive stop)
    plus = 1 if inclusive_stop else 0
    lens = (stops - starts + plus).astype(np.int64)

    # Filter out non-positive lengths defensively
    valid = lens > 0
    if not np.all(valid):
        starts = starts[valid]
        lens = lens[valid]
        file_ids_per_interval = file_ids_per_interval[valid]

    total = int(lens.sum())
    if total <= 0:
        return out

    # Vectorized expansion across all intervals
    rep_starts = np.repeat(starts, lens)                  # length == total
    seg_offsets = (np.cumsum(lens) - lens).astype(np.int64)
    within = np.arange(total, dtype=dtype) - np.repeat(seg_offsets, lens)
    expanded_indices = rep_starts + within                # global expanded indices

    # For each expanded index, the owning file id
    file_ids_expanded = np.repeat(file_ids_per_interval, lens)

    # Group expanded indices by file without Python loops over files
    order = np.argsort(file_ids_expanded, kind='stable')
    sorted_ids = file_ids_expanded[order]
    sorted_idx = expanded_indices[order]

    # Locate boundaries for unique file ids
    unique_ids, first_pos = np.unique(sorted_ids, return_index=True)

    # Split sorted_idx into chunks per unique file id
    # Build slicing indices (add end sentinel)
    boundaries = np.concatenate([first_pos, np.array([sorted_idx.size], dtype=first_pos.dtype)])

    for uidx, start_pos, end_pos in zip(unique_ids, boundaries[:-1], boundaries[1:]):
        if start_pos == end_pos:
            continue
        fname = file_names[int(uidx)]
        out[fname] = sorted_idx[start_pos:end_pos].astype(dtype, copy=False)

    return out
