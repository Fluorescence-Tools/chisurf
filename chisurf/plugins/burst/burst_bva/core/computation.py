"""BVA computation functions."""

from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from chisurf import logging
except ImportError:
    import logging

import tttrlib


class ProgressWindow:
    """Minimal progress reporter (can be a QDialog or a callable wrapper)."""

    def __init__(self, title="Progress", message="Processing...", max_value=100, parent=None):
        self._max = max_value
        self._value = 0
        self._message = message

    def set_value(self, value: int):
        self._value = value

    def set_maximum(self, value: int):
        self._max = value


def read_burst_analysis(
        paris_path: pathlib.Path,
        tttr_file_type: str,
        pattern: str = 'b*4*',
        row_stride: int = 2
) -> Tuple[pd.DataFrame, Dict[str, tttrlib.TTTR]]:
    """Read burst analysis data files and return DataFrame + TTTR dict."""

    data_path = paris_path.parent
    dfs = []
    is_first_file = True
    for path in paris_path.glob(pattern):
        frames = []
        for fn in sorted(path.glob('*')):
            with open(fn) as f:
                t = f.read().splitlines()
                h = t[0].rstrip('\t').split('\t')
                d = [line.rstrip('\t').split('\t') for line in t[2::row_stride]]
                frames.append(pd.DataFrame(d, columns=h))
        dfs.append(pd.concat(frames, ignore_index=True))
    df = pd.concat(dfs, axis=1)

    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            if not is_first_file:
                logging.info(f"read_burst_analysis: Could not convert {column} to numeric")
        is_first_file = False

    tttrs: Dict[str, tttrlib.TTTR] = {}
    for ff in df['First File']:
        if ff not in tttrs:
            fn = str(data_path / ff)
            tttrs[ff] = tttrlib.TTTR(fn, tttr_file_type)

    return df, tttrs


def compute_static_bva_line(
        prox_mean_bins: np.ndarray,
        number_of_photons_per_slice: int = 4,
        n_samples: int = 10_000
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate BVA static line via binomial sampling."""
    binom_samples = np.random.binomial(
        number_of_photons_per_slice,
        prox_mean_bins,
        size=(n_samples, len(prox_mean_bins))
    )
    ratios = binom_samples / number_of_photons_per_slice
    return ratios.mean(axis=0), ratios.std(axis=0)


def compute_bva(
        df: pd.DataFrame,
        tttrs: Dict[str, tttrlib.TTTR],
        donor_channels: List[int] = (0, 8),
        donor_micro_time_ranges: List[Tuple[int, int]] = ((0, 4096),),
        acceptor_channels: List[int] = (1, 9),
        acceptor_micro_time_ranges: List[Tuple[int, int]] = ((0, 4096),),
        minimum_window_length: float = 0.01,
        number_of_photons_per_slice: int = -1,
        progress_window=None,
) -> pd.DataFrame:
    """Compute BVA: proximity ratio mean and std per burst."""
    prox_means: List[float] = []
    proxt_stds: List[float] = []

    tttr_arrays = {}
    time_calibrations = {}
    for ff, tttr in tttrs.items():
        micro_arr = np.asarray(tttr.micro_times)
        channel_arr = np.asarray(tttr.routing_channels)
        macro_arr = np.asarray(tttr.macro_times)
        tttr_arrays[ff] = (micro_arr, channel_arr, macro_arr)
        time_calibrations[ff] = tttr.header.tag('MeasDesc_GlobalResolution')['value']

    col_ff = df.columns.get_loc("First File")
    col_fp = df.columns.get_loc("First Photon")
    col_lp = df.columns.get_loc("Last Photon")

    for i, row in enumerate(df.itertuples(index=False, name=None)):
        ff = row[col_ff]
        first_photon = int(row[col_fp])
        last_photon = int(row[col_lp])
        if ff not in tttr_arrays:
            continue

        micro_arr, channel_arr, macro_arr = tttr_arrays[ff]
        time_calib = time_calibrations[ff]

        mt_burst = micro_arr[first_photon:last_photon]
        ch_burst = channel_arr[first_photon:last_photon]
        macro_burst = macro_arr[first_photon:last_photon]

        n_events = len(macro_burst)
        if n_events == 0:
            prox_means.append(np.nan)
            proxt_stds.append(np.nan)
            continue

        if number_of_photons_per_slice < 0:
            window_duration = minimum_window_length / time_calib
            windows = []
            start_idx = 0
            while start_idx < n_events:
                end_idx = np.searchsorted(macro_burst, macro_burst[start_idx] + window_duration, side='right')
                if end_idx <= start_idx:
                    end_idx = start_idx + 1
                windows.append((start_idx, end_idx))
                start_idx = end_idx
        else:
            chunk_size = number_of_photons_per_slice
            windows = [(i, min(i + chunk_size, n_events)) for i in range(0, n_events, chunk_size)]

        donor_mask = np.logical_or.reduce(
            [(mt_burst >= start) & (mt_burst <= stop) for start, stop in donor_micro_time_ranges]
        )
        donor_mask &= np.isin(ch_burst, donor_channels)
        acceptor_mask = np.logical_or.reduce(
            [(mt_burst >= start) & (mt_burst <= stop) for start, stop in acceptor_micro_time_ranges]
        )
        acceptor_mask &= np.isin(ch_burst, acceptor_channels)

        donor_cs = np.cumsum(donor_mask.astype(np.int64))
        acceptor_cs = np.cumsum(acceptor_mask.astype(np.int64))

        def _count(cs, s, e):
            return cs[e - 1] if s == 0 else cs[e - 1] - cs[s - 1]

        window_ratios = []
        for s, e in windows:
            if s >= e:
                continue
            dc = _count(donor_cs, s, e)
            ac = _count(acceptor_cs, s, e)
            total = dc + ac
            window_ratios.append(ac / total if total > 0 else 0.0)

        if window_ratios:
            prox_means.append(float(np.nanmean(window_ratios)))
            proxt_stds.append(float(np.nanstd(window_ratios)))
        else:
            prox_means.append(np.nan)
            proxt_stds.append(np.nan)

        if progress_window:
            progress_window.set_value(i + 1)

    df['Proximity Ratio Mean'] = np.array(prox_means)
    df['Proximity Ratio Std'] = np.array(proxt_stds)
    return df


def write_bv4_analysis(df: pd.DataFrame, analysis_folder: str = "analysis", progress_window=None):
    """Write BVA results to .bv4 files in a bv4/ subfolder."""
    bv4_folder = pathlib.Path(analysis_folder) / "bv4"
    bv4_folder.mkdir(parents=True, exist_ok=True)

    groups = list(df.groupby("First File"))
    for i, (tttr_file, group) in enumerate(groups, start=1):
        tttr_stem = pathlib.Path(tttr_file).stem
        bv4_filename = bv4_folder / f"{tttr_stem}_0.bv4"

        mini_df = group[['Proximity Ratio Mean', 'Proximity Ratio Std']].copy()
        n = len(mini_df)
        columns_list = list(mini_df.columns) + [""]
        new_df = pd.DataFrame(np.zeros((2 * n + 1, len(columns_list))), columns=columns_list)
        new_df[""] = ""
        new_df.loc[1::2, mini_df.columns] = mini_df.values
        new_df.to_csv(bv4_filename, sep='\t', index=False)

        if progress_window:
            progress_window.set_value(i)

    logging.info(f"BVA results written to {bv4_folder}")
