from __future__ import annotations

"""Photon distribution analysis (PDA) experiment utilities.

This module provides helper functions and the :class:`PdaReader` used
to construct PDA histograms from time-tagged single-photon (TTTR) data.

The central pieces are:

* :func:`build_idx_map` – vectorized helper that expands per-file
  index intervals into NumPy index arrays.
* :class:`PdaReader` – experiment reader that uses :mod:`tttrlib` to
  compute experimental S1S2 histograms and attach PDA metadata to
  :class:`chisurf.core.data.DataCurve` objects.

The examples in this module avoid touching real TTTR files; all
heavy I/O and :mod:`tttrlib` calls are only shown in skipped doctests.
"""

import pathlib
import numpy as np
from typing import Dict, Sequence, Tuple

from chisurf import typing

import json
import tttrlib

import chisurf.core.settings
import chisurf.core.fluorescence.tcspc
import chisurf.core.base
import chisurf.core.fluorescence
import chisurf.core.data
import chisurf.core.curve
import chisurf.core.fio.fluorescence
from chisurf import logging

from chisurf.core.experiments.core.reader import ExperimentReader

from .index_map import build_idx_map


class PdaReader(ExperimentReader):
    operation_type = "pda_histogram_computation"
    artifact_kind_source = "raw_data"
    artifact_kind_derived = "pda_histogram"
    derived_data_format = "json"
    derived_mime_type = "application/json"

    """Experiment reader for PDA TTTR data.

    This reader uses :mod:`tttrlib` to construct experimental S1S2
    histograms from TTTR files and attaches PDA metadata to
    :class:`chisurf.core.data.DataCurve` objects.

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

        >>> from chisurf.core.experiments.pda import PdaReader  # doctest: +SKIP
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
            minimum_time_window_length: float = 2e-3,
            tw_configs=None,
            *args,
            **kwargs
    ):
        """Initialize a PDA reader.

        Parameters
        ----------
        channels : tuple of list of int
            Channel numbers for donor/acceptor detection ``(green, red)``.
        micro_time_ranges : list of tuple of int
            Micro-time windows ``(start, stop)`` for photon selection.
        reading_routine : str
            tttrlib reader type (e.g. ``'PTU'``).
        maximum_number_of_photons : int
            Maximum photon number for the S1S2 histogram support.
        minimum_number_of_photons : int
            Minimum total photon count per burst.
        minimum_time_window_length : float
            Minimum time window length for bursts in seconds.
        tw_configs : list, optional
            List of ``(min_photons, min_tw_len_s)`` configurations.
        """
        super().__init__(*args, **kwargs)
        self.reading_routine = reading_routine
        self.micro_time_ranges = micro_time_ranges
        self.maximum_number_of_photons = maximum_number_of_photons
        self.minimum_number_of_photons = minimum_number_of_photons
        self.minimum_time_window_length = minimum_time_window_length
        self.tw_configs = tw_configs
        self.channels = channels

    def autofitrange(self, data, **kwargs) -> typing.Tuple[int, int]:
        """Return the full flattened data range as the default fit interval.

        Parameters
        ----------
        data : chisurf.core.base.Data
            The experimental PDA data.

        Returns
        -------
        tuple of int
            ``(0, len(y))`` for the flattened y data.
        """
        logging.warning("PDA autofitrange not yet implemented")
        return 0, len(data.y.flatten())

    def read(self, filename: typing.List[str] = None, *args, **kwargs) -> chisurf.core.data.ExperimentDataGroup:
        """Read PDA TTTR data and return S1S2 histograms.

        Parameters
        ----------
        filename : list of str, optional
            Path(s) to the TTTR data file(s).

        Returns
        -------
        chisurf.core.data.ExperimentDataGroup
            Group containing :class:`DataCurve` objects with S1S2 histograms.
            Each curve carries the following metadata:

            - ``curve.pda`` — PDA-specific dict with keys:
              ``maximum_number_of_photons``, ``minimum_number_of_photons``,
              ``minimum_time_window_length``, ``channels``, ``s1s2``,
              ``ps``, ``row_indices``, ``col_indices``, ``ndim``, ``shape``,
              ``size``, ``reading_routine``, ``micro_time_ranges``,
              ``tttr_indices``.
            - ``curve.meta_data`` — generic metadata dict with keys:
              ``filenames`` (list of source file paths), ``grid`` (2D grid
              description), ``tttr_header_json`` (TTTR file header JSON
              string), ``tw_configs`` (list of time-window configurations
              used).
        """
        if isinstance(filename, str):
            filename = [filename]

        try:
            logging.info(
                "PDA TRACE: PdaReader.read called with %d filename(s)",
                len(filename) if filename is not None else -1,
            )
        except Exception:
            pass

        filename.sort()
        from chisurf.core.file_formats import FILE_FORMATS as _FILE_FORMATS
        source_filenames = [
            {
                'path': str(p),
                'format': _FILE_FORMATS.get(p.suffix.lower(), {}).get('name', ''),
            }
            for p in (pathlib.Path(f) for f in filename)
            if p.is_file()
        ]
        fn = pathlib.Path(source_filenames[0]['path']) if source_filenames else pathlib.Path(filename[0])
        data_group = chisurf.core.data.ExperimentDataGroup([])

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
                try:
                    logging.info(
                        "PDA TRACE: burst_slices mode enabled with %d key(s)",
                        len(burst_slices) if burst_slices is not None else 0,
                    )
                except Exception:
                    pass
                # Prepare intervals per actually provided filenames (match keys by full path, name, or stem)
                intervals_by_file: Dict[str, Sequence[Tuple[int, int]]] = {}
                # Normalize keys in a helper for quick access
                def _get_intervals_for_path(p: pathlib.Path):
                    """Return burst-slice intervals for a file path.

                    Tries lookups by full path, basename, and stem.

                    Parameters
                    ----------
                    p : pathlib.Path
                        The file path to look up.

                    Returns
                    -------
                    list
                        The matching burst intervals, or an empty list.
                    """
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
                try:
                    logging.info(
                        "PDA TRACE: intervals_by_file built for %d file(s)",
                        len(intervals_by_file),
                    )
                except Exception:
                    pass
                if not intervals_by_file:
                    t = None
                else:
                    # Build indices; our intervals are [start, stop) so inclusive_stop=False
                    try:
                        logging.info(
                            "PDA TRACE: calling build_idx_map for %d file(s)",
                            len(intervals_by_file),
                        )
                    except Exception:
                        pass
                    idx_map = build_idx_map(intervals_by_file, inclusive_stop=False, dtype=np.int64)
                    try:
                        logging.info(
                            "PDA TRACE: build_idx_map finished; idx_map has %d key(s)",
                            len(idx_map),
                        )
                    except Exception:
                        pass
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
                            try:
                                logging.info(
                                    "PDA TRACE: loading TTTR with burst slices: %s (n_indices=%d)",
                                    key,
                                    int(idxs.size),
                                )
                            except Exception:
                                pass
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
                try:
                    logging.info(
                        "PDA TRACE: loading full TTTR data for %d file(s)",
                        len(filename),
                    )
                except Exception:
                    pass
                t = tttrlib.TTTR(fn.as_posix(), self.reading_routine)
                for fn in filename[1:]:
                    fn = pathlib.Path(fn)
                    if fn.is_file():
                        try:
                            logging.info("PDA TRACE: appending TTTR file %s", str(fn))
                        except Exception:
                            pass
                        d = tttrlib.TTTR(fn.as_posix(), self.reading_routine)
                        t.append(d)

        if t is not None:
            # Passthrough TTTR metadata: extract header JSON once for reuse
            # across all time-window configurations.
            tttr_header_json = None
            try:
                tttr_header_json = t.get_header().get_json()
            except Exception:
                pass

            channels_1 = self.channels[0]
            channels_2 = self.channels[1]

            # Determine the list of (minimum_number_of_photons, minimum_time_window_length)
            # configurations to run. If no explicit list was provided, fall back to the
            # single reader-level thresholds for backward compatibility.
            configs = []
            tw_cfgs = getattr(self, 'tw_configs', None)
            if tw_cfgs:
                for cfg in tw_cfgs:
                    try:
                        n_ph, tw_len = int(cfg[0]), float(cfg[1])
                    except Exception:
                        continue
                    if n_ph <= 0 or tw_len <= 0.0:
                        continue
                    configs.append((n_ph, tw_len))
            if not configs:
                try:
                    configs = [
                        (
                            int(self.minimum_number_of_photons),
                            float(self.minimum_time_window_length),
                        )
                    ]
                except Exception:
                    configs = []

            base_name = fn.stem
            multi = len(configs) > 1

            for n_ph_cfg, tw_len_cfg in configs:
                logging.debug({
                    'channels_1': channels_1,
                    'channels_2': channels_2,
                    'max_photons': self.maximum_number_of_photons,
                    'min_photons': n_ph_cfg,
                    'min_tw_len_s': tw_len_cfg,
                })
                try:
                    logging.info(
                        "PDA TRACE: calling tttrlib.Pda.compute_experimental_histograms (min_photons=%d, min_tw_len_s=%g)",
                        int(n_ph_cfg), float(tw_len_cfg),
                    )
                except Exception:
                    pass
                s1s2_e, ps, tttr_indices = tttrlib.Pda.compute_experimental_histograms(
                    tttr_data=t,
                    channels_1=channels_1,
                    channels_2=channels_2,
                    maximum_number_of_photons=self.maximum_number_of_photons,
                    minimum_number_of_photons=n_ph_cfg,
                    minimum_time_window_length=tw_len_cfg
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
                    'minimum_number_of_photons': n_ph_cfg,
                    'minimum_time_window_length': tw_len_cfg,
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
                    'tttr_indices': tttr_indices,
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
                    'tttr_header_json': tttr_header_json,
                    'tw_configs': getattr(self, 'tw_configs', None),
                    'filenames': source_filenames,
                    'reading_routine': self.reading_routine,
                    'micro_time_ranges': self.micro_time_ranges,
                }

                # Use the full S1S2 matrix as data, flattened in row-major
                # order. Elements outside the physically populated triangular
                # support (if any) will naturally carry zero counts.
                y = np.asarray(s1s2_e, dtype=float).ravel(order='C')
                x = np.arange(y.size)

                name = base_name
                if multi:
                    try:
                        tw_ms_val = float(tw_len_cfg) * 1.0e3
                        name = f"{base_name}_TW{tw_ms_val:g}ms"
                    except Exception:
                        name = base_name

                try:
                    logging.info(
                        "PDA TRACE: constructing DataCurve (name=%s, y_len=%d)",
                        name,
                        int(y.size) if hasattr(y, 'size') else -1,
                    )
                except Exception:
                    pass
                data = chisurf.core.data.DataCurve(
                    name=name,
                    filename=source_filenames[0]['path'] if source_filenames else str(fn),
                    load_filename_on_init=False,
                    data_reader=self,
                    pda=d,
                    meta_data=meta_all,
                    y=y, x=x,
                    ey=chisurf.core.fluorescence.tcspc.counting_noise(y)
                )
                data_group.append(data)
                logging.debug({'s1s2_shape': s1s2_e.shape if hasattr(s1s2_e, 'shape') else None,
                               'y_len': len(y)})
        else:
            logging.warning("PDA.read: No TTTR data could be constructed from the provided files.")

        try:
            logging.info(
                "PDA TRACE: PdaReader.read returning ExperimentDataGroup with %d entry(ies)",
                len(data_group),
            )
        except Exception:
            pass
        data_group.data_reader = self
        return data_group
