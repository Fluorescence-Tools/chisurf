from __future__ import annotations

"""Photon distribution analysis (PDA) experiment utilities.

This module provides helper functions and the :class:`PdaReader` used
to construct PDA histograms from time-tagged single-photon (TTTR) data.

The central pieces are:

* :func:`build_idx_map` – vectorized helper that expands per-file
  index intervals into NumPy index arrays.
* :class:`PdaReader` – experiment reader that uses :mod:`tttrlib` to
  compute experimental S1S2 histograms and attach PDA metadata to
  :class:`chisurf.data.DataCurve` objects.

The examples in this module avoid touching real TTTR files; all
heavy I/O and :mod:`tttrlib` calls are only shown in skipped doctests.
"""

import pathlib
import numpy as np
from typing import Dict, Sequence, Tuple

from chisurf import typing

import tttrlib

import chisurf.settings
import chisurf.fluorescence.tcspc
import chisurf.experiments
import chisurf.base
import chisurf.fluorescence
import chisurf.data
import chisurf.curve
import chisurf.fio.fluorescence
from chisurf import logging

from chisurf.experiments import reader


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


class PdaReader(reader.ExperimentReader):

    """Experiment reader for PDA TTTR data.

    This reader uses :mod:`tttrlib` to construct experimental S1S2
    histograms from TTTR files and attaches PDA metadata to
    :class:`chisurf.data.DataCurve` objects.

    Parameters
    ----------
    channels : tuple of list of int
        Channel numbers for donor/acceptor detection (green, red).
    micro_time_ranges : list of (int, int)
        Micro-time windows used for photon selection.
    reading_routine : str, optional
        :mod:`tttrlib` reader type (e.g. ``'PTU'``, default).
    maximum_number_of_photons : int, optional
        Maximum photon number used for the S1S2 histogram support.
    minimum_number_of_photons : int, optional
        Minimum total photon number for bursts.
    minimum_time_window_length : float, optional
        Minimum time window length for bursts in seconds.

    Notes
    -----
    The :meth:`read` method performs real file I/O and depends on
    :mod:`tttrlib`. A minimal usage pattern (omitting real files) is::

        >>> from chisurf.experiments.pda import PdaReader  # doctest: +SKIP
        >>> r = PdaReader(  # doctest: +SKIP
        ...     channels=([0], [1]),
        ...     micro_time_ranges=[(0, 4096)],
        ...     reading_routine='PTU'
        ... )  # doctest: +SKIP
    """

    def __init__(
            self,
            channels: typing.Tuple[typing.List[int], typing.List[int]],
            micro_time_ranges: typing.List[typing.Tuple[int, int]],
            reading_routine: str = 'PTU',
            maximum_number_of_photons: int = 150,
            minimum_number_of_photons: int = 5,
            minimum_time_window_length: float = 5e-4,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.reading_routine = reading_routine
        self.micro_time_ranges = micro_time_ranges
        self.maximum_number_of_photons = maximum_number_of_photons
        self.minimum_number_of_photons = minimum_number_of_photons
        self.minimum_time_window_length = minimum_time_window_length
        self.channels = channels

    def autofitrange(self, data, **kwargs) -> typing.Tuple[int, int]:
        logging.warning("PDA autofitrange not yet implemented")
        return 0, len(data.y.flatten())

    def read(self, filename: typing.List[str] = None, *args, **kwargs) -> chisurf.data.ExperimentDataGroup:
        if isinstance(filename, str):
            filename = [filename]

        filename.sort()
        fn = pathlib.Path(filename[0])
        data_group = chisurf.data.ExperimentDataGroup([])

        # Optional burst slicing: dict[str, List[Tuple[int,int]]]
        burst_slices = kwargs.get('burst_slices', None)
        if isinstance(burst_slices, dict) and len(burst_slices) == 0:
            burst_slices = None

        logging.debug({
            'reader': 'PdaReader',
            'reading_routine': self.reading_routine,
            'n_input_files': len(filename),
            'first_file': str(fn)
        })

        t = None
        if fn.is_file():
            if burst_slices:
                # Build a TTTR consisting only of specified intervals using vectorized indices
                logging.info(f"PDA.read: Applying burst_slices to TTTR data for {len(filename)} file(s) using index maps.")
                # Prepare intervals per actually provided filenames (match keys by full path, name, or stem)
                intervals_by_file: Dict[str, Sequence[Tuple[int, int]]] = {}
                # Normalize keys in a helper for quick access
                def _get_intervals_for_path(p: pathlib.Path):
                    return (
                        burst_slices.get(str(p), [])
                        or burst_slices.get(p.name, [])
                        or burst_slices.get(p.stem, [])
                    )
                for f in filename:
                    pf = pathlib.Path(f)
                    if not pf.is_file():
                        continue
                    ivals = _get_intervals_for_path(pf)
                    if ivals:
                        intervals_by_file[str(pf)] = ivals
                if not intervals_by_file:
                    t = None
                else:
                    # Build indices; our intervals are [start, stop) so inclusive_stop=False
                    idx_map = build_idx_map(intervals_by_file, inclusive_stop=False, dtype=np.int64)
                    t_sel = None
                    for f in filename:
                        pf = pathlib.Path(f)
                        if not pf.is_file():
                            continue
                        key = str(pf)
                        idxs = idx_map.get(key)
                        if idxs is None or idxs.size == 0:
                            continue
                        try:
                            tt = tttrlib.TTTR(pf.as_posix(), self.reading_routine)
                            # Use vectorized selection once per file
                            ds = tt[idxs]
                        except Exception:
                            continue
                        if ds is None:
                            continue
                        if t_sel is None:
                            t_sel = ds
                        else:
                            t_sel.append(ds)
                    t = t_sel
            else:
                # Default: load entire files and append
                t = tttrlib.TTTR(fn.as_posix(), self.reading_routine)
                for fn in filename[1:]:
                    fn = pathlib.Path(fn)
                    if fn.is_file():
                        d = tttrlib.TTTR(fn.as_posix(), self.reading_routine)
                        t.append(d)

        if t is not None:
            channels_1 = self.channels[0]
            channels_2 = self.channels[1]
            logging.debug({'channels_1': channels_1, 'channels_2': channels_2,
                           'max_photons': self.maximum_number_of_photons,
                           'min_photons': self.minimum_number_of_photons,
                           'min_tw_len_ms': self.minimum_time_window_length})
            s1s2_e, ps, tttr_indices = tttrlib.Pda.compute_experimental_histograms(
                tttr_data=t,
                channels_1=channels_1,
                channels_2=channels_2,
                maximum_number_of_photons=self.maximum_number_of_photons,
                minimum_number_of_photons=self.minimum_number_of_photons,
                minimum_time_window_length=self.minimum_time_window_length
            )

            # Align experimental S1S2 orientation with the theoretical model.
            #
            # The tttrlib.Pda model S1S2 matrix (from the probability spectrum)
            # uses rows for channel 1 (green) and columns for channel 2 (red).
            # The experimental histogram produced by compute_experimental_histograms
            # is effectively stored with rows corresponding to channel 2 and
            # columns to channel 1. For consistent comparison (2D residuals and
            # 1D projections), we transpose the experimental matrix here so that
            # both share the same (green,row; red,col) convention.
            try:
                import numpy as _np
                s1s2_e = _np.asarray(s1s2_e)
                if s1s2_e.ndim == 2:
                    s1s2_e = s1s2_e.T
            except Exception:
                pass

            # Attach PDA-specific metadata describing the 2D S1S2 grid and
            # the 1D flattening used for fitting. We now use a standard
            # row-major flattening of the full 2D support so that the
            # resulting 1D vector can be treated in the same way as other
            # grid-based datasets (e.g. RICS).
            s1s2_shape = tuple(getattr(s1s2_e, 'shape', (0, 0)))
            ny, nx = s1s2_shape if len(s1s2_shape) == 2 else (0, 0)

            # Precompute row/column indices consistent with row-major
            # flattening so PDA-specific code (e.g. photon-number gating)
            # can still access N = row + col for each 1D bin.
            if ny > 0 and nx > 0:
                rr, cc = np.indices((ny, nx))
                row_indices = rr.ravel().tolist()
                col_indices = cc.ravel().tolist()
            else:
                row_indices, col_indices = [], []

            d = {
                'maximum_number_of_photons': self.maximum_number_of_photons,
                'minimum_number_of_photons': self.minimum_number_of_photons,
                'minimum_time_window_length': self.minimum_time_window_length,
                'channels': self.channels,
                's1s2': s1s2_e,
                'ps': ps,
                'row_indices': row_indices,
                'col_indices': col_indices,
                # Dimensionality/meta for 2D handling (kept here for PDA-
                # specific consumers, but GUI should prefer the generic
                # meta_data['grid'] entry added below).
                'ndim': 2,
                'shape': s1s2_shape,
                # Total number of 1D points in the PDA-specific flattening
                'size': int(len(row_indices)),
            }

            # Generic grid metadata describing the 2D S1S2 support and the
            # mapping from the 2D grid to the 1D histogram used for fitting.
            # This is consumed by GUI components in a model-agnostic manner
            # and uses a standard NumPy-style row-major order ('C').
            grid_meta = {
                'ndim': 2,
                'shape': s1s2_shape,
                # NumPy-style order string: 'C' -> row-major, 'F' -> column-major
                'order': 'C',
                # Total number of 1D points in the flattened representation
                'size': int(np.prod(s1s2_shape)) if len(s1s2_shape) == 2 else 0,
            }

            meta_all = {
                'grid': grid_meta,
            }

            # Use the full S1S2 matrix as data, flattened in row-major
            # order. Elements outside the physically populated triangular
            # support (if any) will naturally carry zero counts.
            y = np.asarray(s1s2_e, dtype=float).ravel(order='C')
            x = np.arange(y.size)
            data = chisurf.data.DataCurve(
                name=fn.stem,
                data_reader=self,
                pda=d,
                meta_data=meta_all,
                y=y, x=x,
                ey=chisurf.fluorescence.tcspc.counting_noise(y)
            )
            data_group.append(data)
            logging.debug({'s1s2_shape': s1s2_e.shape if hasattr(s1s2_e, 'shape') else None,
                           'y_len': len(y)})
        else:
            logging.warning("PDA.read: No TTTR data could be constructed from the provided files.")

        data_group.data_reader = self
        return data_group
