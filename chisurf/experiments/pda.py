from __future__ import annotations

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
    """
    Return {filename: np.ndarray(indices)} where each array contains the stacked
    indices of all (non-overlapping) ranges for that file.

    This implementation vectorizes across all files at once to minimize Python
    overhead. It builds one global expansion of all intervals, then groups the
    expanded indices back by file using NumPy operations.

    Assumes start < stop and bursts are non-overlapping. Keeps the original
    file order of keys provided by `intervals_by_file`.
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

    def __init__(
            self,
            channels: typing.Tuple[typing.List[int], typing.List[int]],
            micro_time_ranges: typing.List[typing.Tuple[int, int]],
            reading_routine: str = 'PTU',
            maximum_number_of_photons: int = 148,
            minimum_number_of_photons: int = 20,
            minimum_time_window_length: float = 1e-4,
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

            row_indices, col_indices = list(), list()
            for r in range(self.maximum_number_of_photons):
                for c in range(self.maximum_number_of_photons - r):
                    row_indices.append(r)
                    col_indices.append(c)

            d = {
                'maximum_number_of_photons': self.maximum_number_of_photons,
                'minimum_number_of_photons': self.minimum_number_of_photons,
                'minimum_time_window_length': self.minimum_time_window_length,
                'channels': self.channels,
                's1s2': s1s2_e,
                'ps': ps,
                'row_indices': row_indices,
                'col_indices': col_indices
            }
            # Use s1s2 matrix as data only use upper left triangle of matrix
            y = s1s2_e[row_indices, col_indices]
            x = np.arange(len(y))
            data = chisurf.data.DataCurve(
                name=fn.stem,
                data_reader=self,
                pda=d,
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
