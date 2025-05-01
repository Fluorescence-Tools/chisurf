import sys
import pathlib
import ast
import tqdm
from typing import Dict, List, Tuple, Union

try:
    from chisurf import logging
except ImportError:
    import logging

import tttrlib
import json
import numpy as np
import pandas as pd

# Import PyQt5 modules
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QLineEdit, QMessageBox, QGroupBox, QFormLayout, QProgressBar
)
from PyQt5.QtCore import Qt

# For embedding matplotlib plots in PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from PyQt5.QtWidgets import QDialog, QVBoxLayout, QProgressBar, QLabel
from PyQt5.QtCore import Qt, QCoreApplication

class ProgressWindow(QDialog):
    def __init__(self, title="Progress", message="Processing...", max_value=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)  # Keeps this dialog in front

        self.layout = QVBoxLayout()
        self.label = QLabel(message)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max_value)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def set_value(self, value: int):
        """Update the progress bar value and allow the GUI to refresh."""
        self.progress_bar.setValue(value)
        QCoreApplication.processEvents()


def write_bv4_analysis(df: pd.DataFrame, analysis_folder: str = "analysis", progress_window=None):
    """
    Writes Burst Variance Analysis (BVA) results to .bv4 files in a 'bv4' subfolder.
    If a progress_window is provided, it is updated for each file written.
    """
    bv4_folder = pathlib.Path(analysis_folder) / "bv4"
    bv4_folder.mkdir(parents=True, exist_ok=True)

    # Convert groups to a list to know the total number of iterations.
    groups = list(df.groupby("First File"))
    total = len(groups)

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

        # Update the progress window if provided.
        if progress_window:
            progress_window.set_value(i)

    logging.info(f"BVA results have been written to .bv4 files in the '{bv4_folder}' directory.")


def read_burst_analysis(
        paris_path: pathlib.Path,
        tttr_file_type: str,
        pattern: str = 'b*4*',
        row_stride: int = 2
) -> (pd.DataFrame, Dict[str, tttrlib.TTTR]):
    """
    Reads burst analysis data files from the specified directory and returns a
    concatenated DataFrame along with a dictionary of TTTR data objects.
    (Docstring omitted for brevity.)
    """
    def update_tttr_dict(data_path, tttrs: Dict[str, tttrlib.TTTR] = dict()):
        for ff, fl in zip(df['First File'], df['Last File']):
            if ff not in tttrs:
                fn = str(data_path / ff)
                tttrs[ff] = tttrlib.TTTR(fn, tttr_file_type)
        return tttrs

    info_path = paris_path / 'Info'
    data_path = paris_path.parent

    dfs = []
    is_first_file = True
    for path in paris_path.glob(pattern):
        frames = []
        for fn in sorted(path.glob('*')):
            with open(fn) as f:
                t = f.read().splitlines()
                # build header (drop trailing empty field)
                h = t[0].rstrip('\t').split('\t')
                # build rows, stripping any trailing tabs
                d = [line.rstrip('\t').split('\t') for line in t[2::row_stride]]
                # now each row has exactly len(h) entries
                frames.append(pd.DataFrame(d, columns=h))
        dfs.append(pd.concat(frames, ignore_index=True))
    df = pd.concat(dfs, axis=1)

    # Convert columns to numeric where possible
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            if not is_first_file:
                logging.info(f"read_burst_analysis: Could not convert {column} to numeric")
        is_first_file = False

    tttrs = dict()
    update_tttr_dict(data_path, tttrs)
    return df, tttrs


def compute_static_bva_line(
        prox_mean_bins: np.ndarray,  # Proximity ratio bins
        number_of_photons_per_slice: int = 4,
        n_samples: int = 10_000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates BVA by calculating the mean and standard deviation of proximity ratios.
    (Docstring omitted for brevity.)
    """
    # Vectorized simulation: generate all samples in one call
    binom_samples = np.random.binomial(
        number_of_photons_per_slice,
        prox_mean_bins,
        size=(n_samples, len(prox_mean_bins))
    )
    ratios = binom_samples / number_of_photons_per_slice
    prox_mean = ratios.mean(axis=0)
    prox_sd = ratios.std(axis=0)
    return prox_mean, prox_sd



def compute_bva_optimized_2(
        df: pd.DataFrame,
        tttrs: Dict[str, 'tttrlib.TTTR'],
        donor_channels: List[int] = [0, 8],
        donor_micro_time_ranges: List[Tuple[int, int]] = [(0, 4096)],
        acceptor_channels: List[int] = [1, 9],
        acceptor_micro_time_ranges: List[Tuple[int, int]] = [(0, 4096)],
        minimum_window_length: float = 0.01,
        number_of_photons_per_slice: int = -1,
        progress_window=None  # New optional parameter
) -> pd.DataFrame:
    """
    Optimized computation of Burst Variance Analysis (BVA) that minimizes TTTR slicing.

    For each burst, the function:

      1. Precomputes donor and acceptor binary masks (using vectorized micro time and channel tests)
         and then computes cumulative sums for fast window counting.
      2. Splits each burst into windows either by fixed time (using np.searchsorted) or by fixed photon count.
      3. Uses the cumulative sums to quickly compute the number of donor and acceptor events in each window.
      4. Calculates per-burst proximity ratio mean and standard deviation.

    Parameters:
    -----------
    df : pd.DataFrame
        Burst data containing at least the columns "First File", "First Photon", "Last Photon".
    tttrs : Dict[str, tttrlib.TTTR]
        Dictionary mapping file names to TTTR objects.
    donor_channels : List[int]
        Donor channel identifiers.
    donor_micro_time_ranges : List[Tuple[int, int]]
        Allowed micro time ranges for donor photons.
    acceptor_channels : List[int]
        Acceptor channel identifiers.
    acceptor_micro_time_ranges : List[Tuple[int, int]]
        Allowed micro time ranges for acceptor photons.
    minimum_window_length : float
        Minimum burst window length in seconds (used if number_of_photons_per_slice < 0).
    number_of_photons_per_slice : int
        If set to a positive value, bursts are divided into chunks containing a fixed number of photons.
        If -1, bursts are split based on time windows.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with new columns 'Proximity Ratio Mean' and 'Proximity Ratio Std'.
    """
    import numpy as np

    proximity_ratios_mean = []
    proximity_ratios_sd = []

    # Precompute numpy arrays for each TTTR and cache the global time calibration.
    tttr_arrays = {}
    time_calibrations = {}
    for ff, tttr in tttrs.items():
        micro_arr = np.asarray(tttr.micro_times)
        channel_arr = np.asarray(tttr.routing_channels)
        macro_arr = np.asarray(tttr.macro_times)  # Needed for time window splitting.
        tttr_arrays[ff] = (micro_arr, channel_arr, macro_arr)
        time_calibrations[ff] = tttr.header.tag('MeasDesc_GlobalResolution')['value']

    # Get column indices for required columns.
    col_ff = df.columns.get_loc("First File")
    col_fp = df.columns.get_loc("First Photon")
    col_lp = df.columns.get_loc("Last Photon")

    for i, row in enumerate(df.itertuples(index=False, name=None)):
        ff = row[col_ff]
        first_photon = int(row[col_fp])
        last_photon = int(row[col_lp])
        if ff not in tttr_arrays:
            # Skip burst if TTTR file not precomputed.
            continue

        micro_arr, channel_arr, macro_arr = tttr_arrays[ff]
        time_calib = time_calibrations[ff]

        # Slice out the burst events once.
        mt_burst = micro_arr[first_photon:last_photon]
        ch_burst = channel_arr[first_photon:last_photon]
        macro_burst = macro_arr[first_photon:last_photon]

        # Determine burst windows.
        windows = []
        n_events = len(macro_burst)
        if n_events == 0:
            windows = []
        elif number_of_photons_per_slice < 0:
            # Time window splitting: convert minimum window length (sec) to macro time units.
            window_duration = minimum_window_length / time_calib
            start_idx = 0
            while start_idx < n_events:
                target_time = macro_burst[start_idx] + window_duration
                end_idx = np.searchsorted(macro_burst, target_time, side='right')
                if end_idx <= start_idx:
                    end_idx = start_idx + 1
                windows.append((start_idx, end_idx))
                start_idx = end_idx
        else:
            # Fixed number of photons per slice.
            chunk_size = number_of_photons_per_slice
            windows = [(i, min(i + chunk_size, n_events)) for i in range(0, n_events, chunk_size)]

        if not windows:
            # If no windows computed, skip this burst.
            proximity_ratios_mean.append(np.nan)
            proximity_ratios_sd.append(np.nan)
            continue

        # Precompute donor and acceptor binary masks for the entire burst.
        donor_mask = np.logical_or.reduce(
            [(mt_burst >= start) & (mt_burst <= stop) for start, stop in donor_micro_time_ranges]
        )
        donor_mask &= np.isin(ch_burst, donor_channels)
        acceptor_mask = np.logical_or.reduce(
            [(mt_burst >= start) & (mt_burst <= stop) for start, stop in acceptor_micro_time_ranges]
        )
        acceptor_mask &= np.isin(ch_burst, acceptor_channels)

        # Compute cumulative sums for donor and acceptor masks.
        donor_cumsum = np.cumsum(donor_mask.astype(np.int64))
        acceptor_cumsum = np.cumsum(acceptor_mask.astype(np.int64))

        # Function to get count in window [s, e) using cumulative sum.
        def count_in_window(cumsum, s, e):
            if s == 0:
                return cumsum[e - 1]
            else:
                return cumsum[e - 1] - cumsum[s - 1]

        # Compute counts and ratios for each window.
        window_ratios = []
        for s, e in windows:
            # Safety check: ensure window is non-empty.
            if s >= e:
                continue
            donor_count = count_in_window(donor_cumsum, s, e)
            acceptor_count = count_in_window(acceptor_cumsum, s, e)
            total = donor_count + acceptor_count
            ratio = (acceptor_count / total) if total > 0 else 0
            window_ratios.append(ratio)

        # Compute mean and std over the burst windows.
        if window_ratios:
            proximity_ratios_mean.append(np.nanmean(window_ratios))
            proximity_ratios_sd.append(np.nanstd(window_ratios))
        else:
            proximity_ratios_mean.append(np.nan)
            proximity_ratios_sd.append(np.nan)

        if progress_window:
            progress_window.set_value(i + 1)


    df['Proximity Ratio Mean'] = np.array(proximity_ratios_mean)
    df['Proximity Ratio Std'] = np.array(proximity_ratios_sd)
    return df


def compute_bva_optimized(
        df: pd.DataFrame,
        tttrs: Dict[str, 'tttrlib.TTTR'],
        donor_channels: List[int] = [0, 8],
        donor_micro_time_ranges: List[Tuple[int, int]] = [(0, 4096)],
        acceptor_channels: List[int] = [1, 9],
        acceptor_micro_time_ranges: List[Tuple[int, int]] = [(0, 4096)],
        minimum_window_length: float = 0.01,
        number_of_photons_per_slice: int = -1
) -> pd.DataFrame:
    """
    Optimized computation of burst variance analysis (BVA) that minimizes repeated TTTR slicing.
    Pre-extracts the macro, micro, and channel arrays from each TTTR and uses numpy operations to split
    bursts into time windows (or fixed photon chunks) and count donor/acceptor events.
    """
    proximity_ratios_mean = []
    proximity_ratios_sd = []

    # Precompute numpy arrays for each TTTR and cache time calibrations.
    tttr_arrays = {}
    time_calibrations = {}
    for ff, tttr in tttrs.items():
        micro_arr = np.asarray(tttr.micro_times)
        channel_arr = np.asarray(tttr.routing_channels)
        # Assuming macro_times is available for window splitting.
        macro_arr = np.asarray(tttr.macro_times)
        tttr_arrays[ff] = (micro_arr, channel_arr, macro_arr)
        time_calibrations[ff] = tttr.header.tag('MeasDesc_GlobalResolution')['value']

    # Get column indices (assuming column names: "First File", "First Photon", "Last Photon")
    col_ff = df.columns.get_loc("First File")
    col_fp = df.columns.get_loc("First Photon")
    col_lp = df.columns.get_loc("Last Photon")

    # Iterate over bursts using itertuples with plain tuples for speed.
    for row in tqdm.tqdm(df.itertuples(index=False, name=None), total=len(df)):
        ff = row[col_ff]
        first_photon = int(row[col_fp])
        last_photon = int(row[col_lp])

        # Skip if the TTTR arrays for this file aren't available.
        if ff not in tttr_arrays:
            continue

        micro_arr, channel_arr, macro_arr = tttr_arrays[ff]
        time_calib = time_calibrations[ff]

        # Slice the burst directly from the precomputed arrays.
        mt_burst = micro_arr[first_photon:last_photon]
        ch_burst = channel_arr[first_photon:last_photon]
        macro_burst = macro_arr[first_photon:last_photon]

        # Determine burst windows.
        windows = []
        if number_of_photons_per_slice < 0:
            # Use time windows: convert the minimum window length into macro time units.
            window_duration = minimum_window_length / time_calib
            start_idx = 0
            n_events = len(macro_burst)
            while start_idx < n_events:
                target_time = macro_burst[start_idx] + window_duration
                # Find the first index where macro time exceeds the target.
                end_idx = np.searchsorted(macro_burst, target_time, side='right')
                if end_idx <= start_idx:
                    end_idx = start_idx + 1
                windows.append((start_idx, end_idx))
                start_idx = end_idx
        else:
            # Use fixed-size photon slices.
            chunk_size = number_of_photons_per_slice
            n_events = len(mt_burst)
            windows = [(i, min(i + chunk_size, n_events)) for i in range(0, n_events, chunk_size)]

        # Count donor and acceptor photons for each window.
        donor_counts = []
        acceptor_counts = []
        for s, e in windows:
            if s >= e:
                continue
            window_mt = mt_burst[s:e]
            window_ch = ch_burst[s:e]
            # Build masks for donor channels within the allowed micro time ranges.
            donor_mask = np.logical_or.reduce(
                [(window_mt >= start) & (window_mt <= stop) for start, stop in donor_micro_time_ranges]
            )
            donor_mask &= np.isin(window_ch, donor_channels)
            # Build masks for acceptor channels.
            acceptor_mask = np.logical_or.reduce(
                [(window_mt >= start) & (window_mt <= stop) for start, stop in acceptor_micro_time_ranges]
            )
            acceptor_mask &= np.isin(window_ch, acceptor_channels)
            donor_counts.append(donor_mask.sum())
            acceptor_counts.append(acceptor_mask.sum())

        donor_counts_arr = np.array(donor_counts)
        acceptor_counts_arr = np.array(acceptor_counts)
        total = donor_counts_arr + acceptor_counts_arr
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(total > 0, acceptor_counts_arr / total, 0)
        proximity_ratios_mean.append(np.nanmean(ratios))
        proximity_ratios_sd.append(np.nanstd(ratios))

    df['Proximity Ratio Mean'] = np.array(proximity_ratios_mean)
    df['Proximity Ratio Std'] = np.array(proximity_ratios_sd)
    return df


def compute_bva(
        df: pd.DataFrame,  # Target data frame
        tttrs: Dict[str, 'tttrlib.TTTR'],  # Dictionary of TTTR data
        donor_channels: List[int] = [0, 8],
        donor_micro_time_ranges: List[Tuple[int, int]] = [(0, 4096)],
        acceptor_channels: List[int] = [1, 9],
        acceptor_micro_time_ranges: List[Tuple[int, int]] = [(0, 4096)],
        minimum_window_length: float = 0.01,
        number_of_photons_per_slice: int = -1
) -> pd.DataFrame:
    """
    Computes proximity ratio statistics for bursts using BVA.
    (Docstring omitted for brevity.)
    """
    proximity_ratios_mean = []
    proximity_ratios_sd = []

    # Precompute time calibrations (avoiding repeated header lookups)
    time_calibrations = {ff: tttr.header.tag('MeasDesc_GlobalResolution')['value']
                         for ff, tttr in tttrs.items()}

    # Precompute column indices:
    col_ff = df.columns.get_loc("First File")
    col_lf = df.columns.get_loc("Last File")
    col_fp = df.columns.get_loc("First Photon")
    col_lp = df.columns.get_loc("Last Photon")

    for row in tqdm.tqdm(df.itertuples(index=False, name=None), total=len(df)):
        # Assuming df has columns: 'First File', 'Last File', 'First Photon', 'Last Photon'
        ff = row[col_ff]
        fl = row[col_lf]
        first_photon = int(row[col_fp])
        last_photon = int(row[col_lp])

        tttr = tttrs.get(ff, None)
        if tttr is None:
            continue

        time_calibration = time_calibrations[ff]
        burst_tttr = tttr[first_photon:last_photon]

        if number_of_photons_per_slice < 0:
            burst_tws = burst_tttr.get_ranges_by_time_window(
                minimum_window_length,
                macro_time_calibration=time_calibration
            )
            burst_tws = burst_tws.reshape((-1, 2))
            burst_tws_tttr = [burst_tttr[start:stop] for start, stop in burst_tws]
        else:
            chunk_size = number_of_photons_per_slice
            burst_tws_tttr = [burst_tttr[i:i + chunk_size] for i in range(0, len(burst_tttr), chunk_size)]

        n_donor = []
        n_acceptor = []
        for tw_tttr in burst_tws_tttr:
            mt = tw_tttr.micro_times
            ch = tw_tttr.routing_channels

            # Combine microtime ranges using np.logical_or.reduce (vectorized)
            donor_time_mask = np.logical_or.reduce(
                [(mt >= start) & (mt <= stop) for start, stop in donor_micro_time_ranges]
            )
            donor_channel_mask = np.isin(ch, donor_channels)
            combined_donor_mask = donor_time_mask & donor_channel_mask

            acceptor_time_mask = np.logical_or.reduce(
                [(mt >= start) & (mt <= stop) for start, stop in acceptor_micro_time_ranges]
            )
            acceptor_channel_mask = np.isin(ch, acceptor_channels)
            combined_acceptor_mask = acceptor_time_mask & acceptor_channel_mask

            n_donor.append(np.sum(combined_donor_mask))
            n_acceptor.append(np.sum(combined_acceptor_mask))

        tw_n_acceptor = np.array(n_acceptor)
        tw_n_donor = np.array(n_donor)
        tw_total = tw_n_acceptor + tw_n_donor
        with np.errstate(divide='ignore', invalid='ignore'):
            tw_proximity_ratios = np.where(tw_total > 0, tw_n_acceptor / tw_total, 0)

        proximity_ratios_mean.append(np.nanmean(tw_proximity_ratios))
        proximity_ratios_sd.append(np.nanstd(tw_proximity_ratios))

    df['Proximity Ratio Mean'] = np.array(proximity_ratios_mean)
    df['Proximity Ratio Std'] = np.array(proximity_ratios_sd)
    return df


def make_2d_plot_on_canvas(
        canvas,  # the MplCanvas instance
        x, y,
        range_x,
        range_y,
        xlabel: str = "x",
        ylabel: str = "y",
        bins_x: int = 100,
        bins_y: int = 100,
        log_x: bool = False,
        log_y: bool = False,
        vmin: float = None,
        vmax: float = None,
        cmap: str = 'jet',
        overlays: list = None
):
    canvas.fig.clear()

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height, width, 0.2]
    rect_histy = [left + width, bottom, 0.2, height]

    ax_scatter = canvas.fig.add_axes(rect_scatter)
    ax_histx = canvas.fig.add_axes(rect_histx, sharex=ax_scatter)
    ax_histy = canvas.fig.add_axes(rect_histy, sharey=ax_scatter)

    ax_scatter.set_xlabel(xlabel, fontsize=16, fontweight='bold')
    ax_scatter.set_ylabel(ylabel, fontsize=16, fontweight='bold')

    if log_x:
        ax_scatter.set_xscale("log")
        ax_histx.set_xscale("log")
    if log_y:
        ax_scatter.set_yscale("log")
        ax_histy.set_yscale("log")

    for label in (ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels() +
                  ax_histx.get_xticklabels() + ax_histx.get_yticklabels()):
        label.set_fontsize(14)

    if log_x:
        bins_x_vals = np.logspace(np.log10(range_x[0]), np.log10(range_x[1]), bins_x)
    else:
        bins_x_vals = np.linspace(range_x[0], range_x[1], bins_x)
    if log_y:
        bins_y_vals = np.logspace(np.log10(range_y[0]), np.log10(range_y[1]), bins_y)
    else:
        bins_y_vals = np.linspace(range_y[0], range_y[1], bins_y)

    def scatter_hist(x, y, ax, ax_histx, ax_histy):
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        ax.hist2d(x, y, bins=[bins_x_vals, bins_y_vals], cmap=cmap, density=False, vmin=vmin, vmax=vmax)
        ax_histx.hist(x, bins=bins_x_vals, color="dimgrey")
        ax_histy.hist(y, bins=bins_y_vals, color="darkorange", orientation='horizontal')

    scatter_hist(x, y, ax_scatter, ax_histx, ax_histy)

    if overlays:
        for overlay in overlays:
            x_overlay, y_overlay, color = overlay
            ax_scatter.plot(x_overlay, y_overlay, color=color)

    canvas.draw()


# ----------------------------
# Default BVA settings
# ----------------------------
DEFAULT_FILE_TYPE = 'SPC-130'
DEFAULT_BVA_SETTINGS: Dict[str, Union[List[int], List[Tuple[int, int]], float, int]] = {
    "donor_channels": [0, 8],
    "donor_micro_time_ranges": [(0, 32768)],
    "acceptor_channels": [1, 9],
    "acceptor_micro_time_ranges": [(0, 32768)],
    "minimum_window_length": 0.01,
    "number_of_photons_per_slice": 10
}


# ----------------------------
# PyQt GUI Application
# ----------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("smFRET BVA Analysis")
        self.resize(400, 800)
        self.data_folder = None
        self.analysis_folder = None
        self.file_type = DEFAULT_FILE_TYPE
        self.bva_settings = DEFAULT_BVA_SETTINGS.copy()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        folder_layout = QHBoxLayout()
        self.btn_select_folder = QPushButton("Select Data Folder")
        self.btn_select_folder.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.btn_select_folder)
        self.folder_label = QLabel("No folder selected")
        folder_layout.addWidget(self.folder_label)
        self.main_layout.addLayout(folder_layout)

        self.bva_group = QGroupBox("BVA Settings")
        bva_layout = QFormLayout()
        self.le_donor_channels = QLineEdit("0,8")
        bva_layout.addRow("Donor Channels (comma-separated):", self.le_donor_channels)
        self.le_donor_micro = QLineEdit("[(0,32768)]")
        bva_layout.addRow("Donor Micro Time Ranges:", self.le_donor_micro)
        self.le_acceptor_channels = QLineEdit("1,9")
        bva_layout.addRow("Acceptor Channels (comma-separated):", self.le_acceptor_channels)
        self.le_acceptor_micro = QLineEdit("[(0,32768)]")
        bva_layout.addRow("Acceptor Micro Time Ranges:", self.le_acceptor_micro)
        self.le_minimum_window_length = QLineEdit("0.01")
        bva_layout.addRow("Minimum Window Length:", self.le_minimum_window_length)
        self.le_photons_per_slice = QLineEdit("10")
        bva_layout.addRow("Number of Photons per Slice:", self.le_photons_per_slice)
        self.bva_group.setLayout(bva_layout)
        self.main_layout.addWidget(self.bva_group)

        self.btn_run = QPushButton("Run BVA Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.main_layout.addWidget(self.btn_run)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.main_layout.addWidget(self.canvas)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.data_folder = pathlib.Path(folder)
            self.analysis_folder = self.data_folder
            self.folder_label.setText(f"Data Folder: {self.data_folder}")

    def get_bva_settings(self) -> Dict[str, Union[List[int], List[Tuple[int, int]], float, int]]:
        try:
            donor_channels = [int(x.strip()) for x in self.le_donor_channels.text().split(',')]
        except Exception as e:
            raise ValueError("Error parsing Donor Channels: " + str(e))
        try:
            donor_micro = ast.literal_eval(self.le_donor_micro.text())
            if not isinstance(donor_micro, list):
                donor_micro = [donor_micro]
        except Exception as e:
            raise ValueError("Error parsing Donor Micro Time Ranges: " + str(e))
        try:
            acceptor_channels = [int(x.strip()) for x in self.le_acceptor_channels.text().split(',')]
        except Exception as e:
            raise ValueError("Error parsing Acceptor Channels: " + str(e))
        try:
            acceptor_micro = ast.literal_eval(self.le_acceptor_micro.text())
            if not isinstance(acceptor_micro, list):
                acceptor_micro = [acceptor_micro]
        except Exception as e:
            raise ValueError("Error parsing Acceptor Micro Time Ranges: " + str(e))
        try:
            min_window = float(self.le_minimum_window_length.text())
        except Exception as e:
            raise ValueError("Error parsing Minimum Window Length: " + str(e))
        try:
            photons_slice = int(self.le_photons_per_slice.text())
        except Exception as e:
            raise ValueError("Error parsing Number of Photons per Slice: " + str(e))

        settings = {
            "donor_channels": donor_channels,
            "donor_micro_time_ranges": donor_micro,
            "acceptor_channels": acceptor_channels,
            "acceptor_micro_time_ranges": acceptor_micro,
            "minimum_window_length": min_window,
            "number_of_photons_per_slice": photons_slice
        }
        return settings

    def run_analysis(self):
        if not self.data_folder:
            QMessageBox.warning(self, "Error", "Please select a data folder first.")
            return

        try:
            self.bva_settings = self.get_bva_settings()
        except Exception as e:
            QMessageBox.critical(self, "BVA Settings Error", str(e))
            return

        QApplication.processEvents()

        # Optionally create and show a dedicated progress dialog.
        # For example, if you expect 'df' to have N rows:
        # (You might need to estimate the total work units beforehand.)
        progress_dialog = ProgressWindow(
            title="BVA Analysis",
            message="Computing BVA...",
            max_value=100,  # You can update this after you know the number of iterations (e.g., len(df))
            parent=self
        )
        progress_dialog.show()

        # Read burst analysis and run BVA computations
        df, tttrs = read_burst_analysis(self.analysis_folder, self.file_type, pattern='bi4_bur')

        # Adjust the progress dialog maximum to match the number of bursts:
        progress_dialog.progress_bar.setMaximum(len(df))

        # Pass the progress_dialog to the computation function
        df_v = compute_bva_optimized_2(df, tttrs, **self.bva_settings, progress_window=progress_dialog)

        progress_dialog.close()

        df_selected = df[df['Proximity Ratio Std'] > 0.0]
        x = df_selected['Proximity Ratio Mean']
        y = df_selected['Proximity Ratio Std']

        x_axis = np.linspace(0, 1, 131)
        number_of_photons_per_slice = self.bva_settings['number_of_photons_per_slice']
        if number_of_photons_per_slice < 0:
            number_of_photons_per_slice = 100
        prox_mean_sim, prox_sd_sim = compute_static_bva_line(
            x_axis, number_of_photons_per_slice=number_of_photons_per_slice
        )
        overlays = [(prox_mean_sim, prox_sd_sim, 'r')]

        make_2d_plot_on_canvas(
            self.canvas,
            x, y,
            range_x=(-0.05, 1.05),
            range_y=(-0.01, 0.44),
            xlabel='Mean Proximity Ratio',
            ylabel='Std Proximity Ratio',
            bins_x=51,
            bins_y=51,
            vmin=0.1,
            vmax=None,
            cmap='Purples',
            overlays=overlays
        )

        progress_dialog = ProgressWindow(
            title="Writing BV4 Files",
            message="Writing BV4 files...",
            max_value=len(df.groupby("First File"))
        )
        progress_dialog.show()

        write_bv4_analysis(df_v, self.analysis_folder, progress_window=progress_dialog)

        progress_dialog.close()

        bv4_folder = pathlib.Path(self.analysis_folder) / "bv4"
        bv4_folder.mkdir(parents=True, exist_ok=True)
        settings_path = bv4_folder / "bva_settings.json"

        with open(settings_path, "w") as f:
            json.dump(self.bva_settings, f, indent=4)

        logging.info(f"BVA settings saved to {settings_path}")

# ----------------------------
# Entry Points for Plugin and Standalone Mode
# ----------------------------
if __name__ == "plugin":
    wizard = MainWindow()
    wizard.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
