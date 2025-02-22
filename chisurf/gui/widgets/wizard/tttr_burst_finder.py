import os
import pathlib
import typing

import tttrlib
import json
import time
import numpy as np

import matplotlib
import pyqtgraph as pg
import pandas as pd

from typing import TypedDict
from collections import OrderedDict

import chisurf.fio
import chisurf.gui.decorators
from chisurf.gui import QtGui, QtWidgets, QtCore, uic

QValidator = QtGui.QValidator

colors = chisurf.settings.gui['plot']['colors']


def get_indices_in_ranges(rout, mt, chs, micro_time_ranges):
    # Create a boolean mask for the rout values in chs
    rout_mask = np.isin(rout, chs)

    # Create a boolean mask for the mt values in micro_time_ranges
    mt_mask = np.zeros(mt.shape, dtype=bool)
    for start, end in micro_time_ranges:
        mt_mask |= (mt >= start) & (mt <= end)

    # Get indices where both masks are true
    indices = np.where(rout_mask & mt_mask)[0]

    return indices.tolist()


def create_mti_summary(
        filename: pathlib.Path,
        analysis_dir: pathlib.Path,
        max_macro_time,
        append: bool = True
):
    """
    Creates or appends to an MTI file in the 'Info' folder. If any MTI file exists, appends to it;
    otherwise, creates a new one based on the given filename.

    Args:
        filename (pathlib.Path): Path to the original file (e.g., '.ht3').
        analysis_dir (pathlib.Path): Directory where the 'Info' folder will be created if missing.
        max_macro_time: The maximum macro time (last photon time) to log.
        append (bool): If True, appends to the first existing MTI file found, otherwise creates a new file.

    Example:
        Creates or appends to an MTI file in:
        c:/analysis_directory/Info/Split_60_132_tween0p00001-0000.mti
        With entry:
        c:/data/Split_60_132_tween0p00001-0000.ht3   74860.905977
    """
    # Create the 'Info' directory if it doesn't exist
    parent_directory = analysis_dir / 'Info'
    parent_directory.mkdir(exist_ok=True, parents=True)

    # Search for any existing .mti files in the 'Info' folder
    existing_mti_files = list(parent_directory.glob("*.mti"))

    # If there are any existing .mti files, append to the first one found
    if existing_mti_files and append:
        mti_filename = existing_mti_files[0]
        mode = 'a'
    else:
        # If no .mti files are found or append is False, create a new file based on the filename
        mti_filename = parent_directory / f"{filename.stem}.mti"
        mode = 'w'

    # Write the filename and max_macro_time to the .mti file
    with open(mti_filename, mode) as mti_file:
        mti_file.write(f"{filename}\t{max_macro_time:.6f}\n")


def create_bur_summary(start_stop, filename, tttr, windows, detectors):
    """
    A vectorized approach to compute burst summary information.

    :param start_stop: List of tuples (start_index, stop_index)
    :param filename: String representing the file name
    :param tttr: TTTR object with attributes:
                 - macro_times
                 - micro_times
                 - routing_channel
                 - header.macro_time_resolution
    :param windows: Dictionary {window_name: [(r_start, r_stop), ...]}
    :param detectors: Dictionary {det_name: {"chs": [...], "micro_time_ranges": [(mt_start, mt_stop), ...]}}
    :return: A pandas DataFrame with one row per burst, including detector + window stats.
    """

    # Extract main arrays once
    n_max = len(tttr)
    macro_times = tttr.macro_times
    micro_times = tttr.micro_times
    routing_channels = tttr.routing_channel
    res = tttr.header.macro_time_resolution

    summary_data = []

    # Loop over each burst defined by (start_index, stop_index)
    for start_index, stop_index in start_stop:
        if start_index > n_max or stop_index > n_max:
            continue

        # Slice once for this burst
        burst_rout = routing_channels[start_index:stop_index]
        burst_micro = micro_times[start_index:stop_index]
        burst_macro = macro_times[start_index:stop_index]

        # Compute global burst stats
        if stop_index <= start_index:
            # Handle corner cases (e.g., empty slice)
            duration = 0.0
            mean_macro_time = 0.0
            n_photons = 0
        else:
            duration = (macro_times[stop_index] - macro_times[start_index]) * res
            mean_macro_time = ((macro_times[stop_index] + macro_times[start_index]) / 2.0) * res
            n_photons = stop_index - start_index

        # Avoid division by zero
        count_rate = n_photons / duration if duration > 0 else np.nan

        # Initialize the row dictionary
        row_data = {
            "First Photon": start_index,
            "Last Photon": stop_index,
            "Duration (ms)": duration * 1e3,
            "Mean Macro Time (ms)": mean_macro_time * 1e3,
            "Number of Photons": n_photons,
            "Count Rate (KHz)": count_rate / 1e3,
            "First File": filename,
            "Last File": filename
        }

        # ----------------------------------------------------------------------
        # 1) Precompute a boolean mask for each detector (channels + micro_time_ranges)
        # ----------------------------------------------------------------------
        detector_masks = {}
        for det_name, det_info in detectors.items():
            chs = det_info["chs"]
            micro_time_ranges = det_info["micro_time_ranges"]

            # (a) channel mask (is routing_channel in the allowed channels?)
            ch_mask = np.isin(burst_rout, chs)

            # (b) microtime mask (is micro_time within one of the specified ranges?)
            micro_mask = np.zeros(len(burst_micro), dtype=bool)
            for (mts, mtp) in micro_time_ranges:
                micro_mask |= (burst_micro >= mts) & (burst_micro < mtp)

            # Combined detector mask
            det_mask = ch_mask & micro_mask
            detector_masks[det_name] = det_mask

        # ----------------------------------------------------------------------
        # 2) Compute per-detector stats for the entire burst
        # ----------------------------------------------------------------------
        for det_name, mask in detector_masks.items():
            idx = np.nonzero(mask)[0]
            if len(idx) == 0:
                # No photons in this detector for the entire burst
                row_data[f"First Photon ({det_name})"] = -1
                row_data[f"Last Photon ({det_name})"] = -1
                row_data[f"Duration ({det_name}) (ms)"] = -1.0
                row_data[f"Mean Macrotime ({det_name}) (ms)"] = -1.0
                row_data[f"Number of Photons ({det_name})"] = 0
                row_data[f"{det_name.capitalize()} Count Rate (KHz)"] = -1.0
            else:
                first_idx = idx[0]
                last_idx = idx[-1]
                num_ph = len(idx)
                dur_color_ms = (burst_macro[last_idx] - burst_macro[first_idx]) * res * 1e3
                mean_mt_color_ms = ((burst_macro[last_idx] + burst_macro[first_idx]) / 2.0) * res * 1e3
                rate_color_khz = (num_ph / dur_color_ms) if dur_color_ms > 0 else np.nan

                row_data[f"First Photon ({det_name})"] = start_index + first_idx
                row_data[f"Last Photon ({det_name})"] = start_index + last_idx
                row_data[f"Duration ({det_name}) (ms)"] = dur_color_ms
                row_data[f"Mean Macrotime ({det_name}) (ms)"] = mean_mt_color_ms
                row_data[f"Number of Photons ({det_name})"] = num_ph
                row_data[f"{det_name.capitalize()} Count Rate (KHz)"] = rate_color_khz

        # ----------------------------------------------------------------------
        # 3) Compute per-detector, per-window stats
        # ----------------------------------------------------------------------
        for window_name, w_ranges in windows.items():
            # If you only need the first (r_start, r_stop) in windows[window_name]:
            (r_start, r_stop) = w_ranges[0]

            # Build a mask for the window's microtime range
            w_mask = (burst_micro >= r_start) & (burst_micro < r_stop)

            for det_name in detectors:
                # Intersection of detector mask with the window mask
                combined_mask = detector_masks[det_name] & w_mask
                idx = np.nonzero(combined_mask)[0]
                if len(idx) == 0:
                    row_data[f"S {window_name} {det_name} (kHz) | {r_start}-{r_stop}"] = -1.0
                else:
                    num_ph = len(idx)
                    dur_window_ms = (burst_macro[idx[-1]] - burst_macro[idx[0]]) * res * 1e3
                    rate_window_khz = (num_ph / dur_window_ms) if dur_window_ms > 0 else np.nan
                    row_data[f"S {window_name} {det_name} (kHz) | {r_start}-{r_stop}"] = rate_window_khz

        # Append row data
        summary_data.append(row_data)

    # Build dataframe
    summary_df = pd.DataFrame(summary_data)
    return summary_df


class CommaSeparatedIntegersValidator(QValidator):

    def validate(self, input_str, pos):
        # Allow empty input
        if not input_str:
            return QValidator.Intermediate, input_str, pos

        # Split the input by commas
        parts = input_str.split(',')

        for part in parts:
            part = part.strip()

            # Allow empty parts (for intermediate states like entering a comma)
            if part == '':
                continue

            # Check if the part is a digit
            if not part.isdigit():
                return QValidator.Intermediate, input_str, pos

            # Convert to integer and check range
            num = int(part)
            if num < 0 or num > 255:
                return QValidator.Invalid, input_str, pos

        # If the input ends with a comma, allow it as intermediate input
        if input_str.endswith(','):
            return QValidator.Intermediate, input_str, pos

        # If all parts are valid, return Acceptable
        return QValidator.Acceptable, input_str, pos

    def fixup(self, input_str):
        # Remove trailing commas
        input_str = input_str.rstrip(',')

        # Optionally, fix invalid parts (in case there are out-of-range numbers)
        parts = input_str.split(',')
        valid_parts = []

        for part in parts:
            part = part.strip()
            if part.isdigit():
                num = int(part)
                if 0 <= num <= 255:
                    valid_parts.append(str(num))

        return ', '.join(valid_parts)


def fill_small_gaps_in_array(arr, max_gap):
    # Identify where the array changes from 1 to 0 and 0 to 1
    is_burst = np.diff(arr, prepend=0, append=0)
    starts = np.where(is_burst == 1)[0]
    stops = np.where(is_burst == -1)[0]

    # Calculate gap sizes between consecutive bursts
    gaps = starts[1:] - stops[:-1] - 1

    # Identify which gaps are small enough to fill
    small_gaps = np.where(gaps <= max_gap)[0]

    # Fill small gaps by setting the values in those gaps to 1
    for idx in small_gaps:
        arr[stops[idx]:starts[idx + 1]] = 1

    return arr


def find_bursts(arr, max_gap=0):
    # Find where the array changes from 0 to 1 (start of burst) and 1 to 0 (end of burst)
    is_burst = np.diff(arr, prepend=0, append=0)
    starts = np.where(is_burst == 1)[0]
    stops = np.where(is_burst == -1)[0]

    # If max_gap is greater than 0, merge small gaps
    if max_gap > 0:
        merged_starts = [starts[0]]  # Initialize with the first start
        merged_stops = []

        for i in range(1, len(starts)):
            # Check if the gap between current stop and next start is small enough to merge
            if starts[i] - stops[i - 1] - 1 <= max_gap:
                continue  # Skip this start, effectively merging
            else:
                merged_stops.append(stops[i - 1])
                merged_starts.append(starts[i])

        # Append the final stop
        merged_stops.append(stops[-1])

        # Convert merged lists to NumPy arrays
        starts = np.array(merged_starts)
        stops = np.array(merged_stops)

    # Stack the starts and stops into a 2D array
    bursts = np.column_stack((starts, stops - 1))  # stop is exclusive, so subtract 1

    return bursts


class CountRateFilterSettings(TypedDict):
    n_ph_max: int
    time_window: float
    invert: bool


class DeltaMacroTimeFilterSettings(TypedDict):
    dT_min: float
    dT_max: float
    dT_min_active: bool
    dT_max_active: bool


class PhotonFilterSettings(TypedDict):
    count_rate_filter: CountRateFilterSettings
    delta_macro_time: DeltaMacroTimeFilterSettings


class WizardTTTRBurstFinder(QtWidgets.QWizardPage):

    @property
    def photon_number_threshold(self):
        return self.spinBox.value()

    @property
    def target_path(self) -> pathlib.Path:
        return pathlib.Path(self.lineEdit_2.text())

    @property
    def filename(self) -> pathlib.Path:
        return pathlib.Path(self.lineEdit.text())

    @property
    def filetype(self) -> str:
        txt = self.comboBox.currentText()
        if txt == 'Auto':
            return None
        return txt

    @property
    def trace_bin_width(self) -> float:
        return float(self.doubleSpinBox_4.value())

    @property
    def use_lower(self) -> bool:
        return bool(self.checkBox_2.isChecked())

    @property
    def use_upper(self) -> bool:
        return bool(self.checkBox_3.isChecked())

    @property
    def use_gap_fill(self):
        return bool(self.checkBox_5.isChecked())

    @property
    def plot_min(self):
        return self.spinBox_2.value()

    @property
    def plot_max(self):
        return self.spinBox_3.value()

    @property
    def current_tttr_filename(self):
        v = self.spinBox_4.value()
        if v < len(self.settings['tttr_filenames']):
            return self.settings['tttr_filenames'][v]
        else:
            return None

    @property
    def decay_coarse(self):
        return self.spinBox_5.value()

    @property
    def max_gap(self):
        return self.spinBox_7.value()

    @property
    def number_of_burst_bins(self):
        return self.spinBox_6.value()

    @property
    def channels(self) -> typing.List[int]:
        s = self.lineEdit_4.text()
        if len(s) > 0:
            return [int(x) for x in s.split(',')]
        return []

    @property
    def microtime_ranges(self) -> typing.Optional[typing.List[typing.Tuple[int, int]]]:
        s = self.lineEdit_5.text()

        # Check if the input string is empty
        if not s:
            chisurf.logging.log(0, "::microtime_ranges: Warning - Input string is empty.")
            return None

        try:
            ranges = [tuple(map(int, item.split('-'))) for item in s.split(';')]

            # Check if each range has exactly two values
            if all(len(r) == 2 for r in ranges):
                return ranges
            else:
                chisurf.logging.log(1, "::microtime_ranges: Invalid format for microsecond ranges.")
                return None

        except (ValueError, TypeError):
            chisurf.logging.log(1, "::microtime_ranges: Invalid values in microsecond ranges.")
            return None

    @property
    def dT(self):
        if isinstance(self.tttr, tttrlib.TTTR):
            mT = self.tttr.get_macro_times()
            d = np.diff(mT, prepend=mT[0])
            h = self.tttr.header
            d = d * h.macro_time_resolution * 1000.0
            return d

    @property
    def dT_min(self) -> float:
        return self._dT_min

    @property
    def dT_max(self) -> float:
        return self._dT_max

    @property
    def selected(self):
        start_time = time.time()

        dT = self.dT
        if dT is None:
            return list()
        s = np.ones_like(dT, dtype=bool)

        chs = self.channels
        if len(chs) > 0:
            mask = tttrlib.TTTRMask()
            mask.select_channels(self.tttr, chs, mask=True)
            m = mask.get_mask()
            s = np.logical_and(s, m)

        if self.microtime_ranges:
            mask = tttrlib.TTTRMask()
            mask.select_microtime_ranges(self.tttr, self.microtime_ranges)
            mask.flip()
            m = mask.get_mask()
            s = np.logical_and(s, m)

        if self.use_lower:
            s = np.logical_and(s, dT >= self.dT_min)
        if self.use_upper:
            s = np.logical_and(s, dT <= self.dT_max)

        if self.settings['count_rate_filter_active']:
            filter_options = self.settings['count_rate_filter']
            selection_idx = self.tttr.get_selection_by_count_rate(**filter_options, make_mask=True)
            s = np.logical_and(s[:-1], selection_idx >= 0)

        if self.max_gap > 0 and self.use_gap_fill:
            s = fill_small_gaps_in_array(s, max_gap=self.max_gap)

        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        chisurf.logging.log(0, "Elapsed time: %s" % elapsed_time)

        return s.astype(dtype=np.uint8)

    @property
    def burst_start_stop(self):
        selected = self.selected
        max_gap = 4
        if len(selected) > max_gap:
            return find_bursts(selected, max_gap=max_gap)
        else:
            return np.array([], dtype=np.uint64)

    @property
    def burst_lengths(self):
        brst = self.burst_start_stop
        if len(brst) > 2:
            n = brst.T[1] - brst.T[0]
        else:
            n = np.array([], dtype=np.uint64)
        return n

    @property
    def save_sl5(self):
        return self.checkBox_6.isChecked()

    @property
    def save_bur(self):
        return self.checkBox_7.isChecked()

    def update_filter_plot(self):
        n_min = self.plot_min
        n_max = self.plot_max

        x = np.arange(n_min, n_max)
        y = self.selected[n_min:n_max]

        self.plot_select.setData(x=x, y=y)

    def update_dt_plot(self):
        dT = self.dT
        if isinstance(dT, np.ndarray):
            n_min = self.plot_min
            n_max = self.plot_max

            mask = self.selected[n_min:n_max]
            mask = mask.astype(bool)
            y = dT[n_min:n_max]
            x = np.arange(n_min, n_max)

            mx = np.ma.masked_array(x, mask=~mask)
            my = np.ma.masked_array(y, mask=~mask)
            self.plot_selected.setData(x=mx.compressed(), y=my.compressed())

            mx = np.ma.masked_array(x, mask=mask)
            my = np.ma.masked_array(y, mask=mask)
            self.plot_unselected.setData(x=mx.compressed(), y=my.compressed())
        else:
            print("Issue with dT:", dT)


    def update_mcs_plot(self):
        if self.toolButton_2.isChecked():
            if isinstance(self.tttr, tttrlib.TTTR):
                idx = np.where(self.selected)[0]
                tw = self.trace_bin_width / 1000.0

                trace_selected = self.tttr[idx].get_intensity_trace(time_window_length=tw)
                y1 = np.copy(trace_selected)
                x1 = np.arange(len(trace_selected), dtype=np.float64) * tw
                self.plot_mcs_selected.setData(x1, y1)

                trace_all = self.tttr.get_intensity_trace(time_window_length=tw)
                y2 = np.copy(trace_all)
                x2 = np.arange(len(trace_all), dtype=np.float64) * tw
                self.plot_mcs_all.setData(x2, y2)
        else:
            self.plot_mcs_all.setData(x=[1.0], y=[1.0])
            self.plot_mcs_selected.setData(x=[1.0], y=[1.0])

    def update_burst_histogram(self):
        if self.toolButton_7.isChecked():
            burst_lengths = self.burst_lengths
            # Create a histogram of burst lengths
            num_bins = self.number_of_burst_bins
            hist, bin_edges = np.histogram(burst_lengths, bins=num_bins)

            # Clear the previous plot
            self.pw_burst_histogram.clear()

            # Fill area under the histogram
            self.pw_burst_histogram.addItem(pg.BarGraphItem(x0=bin_edges[:-1], x1=bin_edges[1:],
                                                            y0=0, y1=hist, brush='b', pen='w'))

            # Plot with steps
            self.plot_burst_histogram = self.pw_burst_histogram.plot(bin_edges, hist, pen='b', stepMode=True)

            # Add total number of bursts as a label directly on the plot
            total_bursts = np.sum(hist)
            pos_x = 0.5 * (bin_edges[0] + bin_edges[-1])  # x position for the text
            pos_y = hist.max() * 0.9  # y position for the text, slightly below max height
            self.total_bursts_label = pg.TextItem(f"Total bursts: {total_bursts}", anchor=(0, 0), color='w')
            self.total_bursts_label.setPos(pos_x, pos_y)
            self.pw_burst_histogram.addItem(self.total_bursts_label)

            # Set the y-axis to logarithmic scale if needed
            self.pw_burst_histogram.setYRange(0.0, max(hist))
            self.pw_burst_histogram.setXRange(0.0, max(bin_edges))

    def update_decay_plot(self):
        if self.toolButton_3.isChecked():
            if isinstance(self.tttr, tttrlib.TTTR):
                idx = np.where(self.selected)[0]
                y, x = self.tttr[idx].get_microtime_histogram(self.decay_coarse)
                if len(x) > 0:
                    idx_max = np.where(y > 0)[0][-1]
                    x = x[:idx_max]
                    y = y[:idx_max]
                    x *= 1e9  # units in nano seconds
                    self.plot_decay_selected.setData(x=x, y=y)
                    y, x = self.tttr.get_microtime_histogram(self.decay_coarse)
                    idx_max = np.where(y > 0)[0][-1]
                    x = x[:idx_max]
                    y = y[:idx_max]
                    x *= 1e9
                    self.plot_decay_all.setData(x=x, y=y)
        else:
            self.plot_decay_all.setData(x=[1.0], y=[1.0])
            self.plot_decay_selected.setData(x=[1.0], y=[1.0])

    def update_plots(self, selection: str = "all"):
        print("updating plots:", selection)
        try:
            if callable(self.callback_function):
                self.callback_function()
        except AttributeError:
            pass
        if 'mcs' in selection:
            self.update_mcs_plot()
        if 'decay' in selection:
            self.update_decay_plot()
        if 'dT' in selection:
            self.update_dt_plot()
        if 'filter' in selection:
            self.update_filter_plot()
        else:
            self.update_dt_plot()
            self.update_decay_plot()
            self.update_mcs_plot()
            self.update_filter_plot()
            self.update_burst_histogram()

    def read_tttr(self):
        fn = self.current_tttr_filename
        if fn:
            if pathlib.Path(fn).exists():
                n = len(self.settings['tttr_filenames'])
                self.spinBox_4.setMaximum(n - 1)
                self.comboBox.setEnabled(False)
                if isinstance(self.filetype, str):
                    self.tttr = tttrlib.TTTR(fn, self.filetype)
                else:
                    self.tttr = tttrlib.TTTR(fn)
                header = self.tttr.get_header()
                s = header.json
                d = json.loads(s)
                self.settings['header'] = d
                self.update_plots()

    def update_output_path(self):
        if len(self.channels) > 0:
            chs = ','.join([str(x) for x in self.channels])
        else:
            chs = 'All'
        s = f'burstwise_{chs} {self.dT_max:.4f}#{self.photon_number_threshold}'
        self.lineEdit_2.setText(s)

    def update_parameter(self):
        lb, ub = self.region_selector.getRegion()
        self.settings['count_rate_filter_active'] = self.checkBox_4.isChecked()
        self.settings['count_rate_filter']['n_ph_max'] = int(self.spinBox.value())
        self.settings['count_rate_filter']['time_window'] = max(0.05, float(self.doubleSpinBox.value())) * 1e-3
        self.settings['count_rate_filter']['invert'] = bool(self.checkBox.isChecked())

        self.settings['delta_macro_time_filter']['dT_min'] = 10.0**lb
        self.settings['delta_macro_time_filter']['dT_max'] = 10.0**ub
        self.settings['delta_macro_time_filter']['dT_min_active'] = self.checkBox_2.isChecked()
        self.settings['delta_macro_time_filter']['dT_max_active'] = self.checkBox_3.isChecked()

        self.update_plots()
        self.update_output_path()

    def onClearFiles(self):
        self.settings['tttr_filenames'].clear()
        self.comboBox.setEnabled(True)
        self.lineEdit.clear()
        self.tttr = None

        # Clear each plot item
        self.plot_unselected.setData([], [])
        self.plot_selected.setData([], [])
        self.plot_mcs_all.setData([], [])
        self.plot_mcs_selected.setData([], [])
        self.plot_decay_all.setData([], [])
        self.plot_decay_selected.setData([], [])
        self.plot_select.setData([], [])

        # Clear the burst histogram entirely (removes bars/text)
        self.pw_burst_histogram.clear()

    def updateUI(self):
        self.lineEdit.setText(self.current_tttr_filename)

    def onRegionUpdate(self):
        lb, ub = self.doubleSpinBox_2.value(), self.doubleSpinBox_3.value()
        if self.pw_dT.getAxis('left').logMode:
            lb, ub = np.log10(lb), np.log10(ub)
        self.region_selector.setRegion(rgn=(lb, ub))

    @property
    def parent_directories(self) -> typing.List[pathlib.Path]:
        r = list()
        for filename in self.settings['tttr_filenames']:
            # Remove null characters from the filename
            filename = filename.replace('\x00', '')

            fn: pathlib.Path = pathlib.Path(filename).absolute()
            t = fn.parent / self.target_path
            r.append(t)
        return r

    def save_bi4_bur_first_last(self):
        for t, filename in zip(self.parent_directories, self.settings['tttr_filenames']):
            fn: pathlib.Path = pathlib.Path(filename)
            self.tttr = tttrlib.TTTR(fn.as_posix(), self.filetype)

            # Use basename to get the base name of the file
            base_name = fn.stem

            start_stop = self.burst_start_stop
            df = create_bur_summary(
                start_stop=start_stop,
                tttr=self.tttr,
                filename=fn,
                windows=self.windows,
                detectors=self.detectors
            )
            bur_directory = t / 'bi4_bur'
            bur_directory.mkdir(exist_ok=True, parents=True)
            output_bur = bur_directory / f"{base_name}.bur"
            df.to_csv(output_bur, sep='\t', index=False)

            mt = self.tttr.macro_times[-1] * self.tttr.header.macro_time_resolution
            create_mti_summary(
                filename=fn,
                analysis_dir=t,
                max_macro_time=mt,
                append=True
            )

    def save_filter_data(self):
        for t, filename in zip(self.parent_directories, self.settings['tttr_filenames']):
            parent_directory = t / 'sl5'
            parent_directory.mkdir(exist_ok=True, parents=True)
            parent_directory = parent_directory.absolute()
            fn: pathlib.Path = pathlib.Path(filename).absolute()
            self.tttr = tttrlib.TTTR(fn.as_posix(), self.filetype)

            d = {
                'filename': os.path.relpath(fn, t),
                'filetype': self.filetype,
                'count_rate_filter': self.settings['count_rate_filter'],
                'delta_macro_time_filter': self.settings['delta_macro_time_filter'],
                'filter': chisurf.fio.compress_numpy_array(self.selected)
            }

            # Use basename to get the base name of the file
            base_name = fn.stem

            # Modify the output file name to use the base name in the same folder
            output_filename = parent_directory / f"{base_name}.json.gz"

            with chisurf.fio.open_maybe_zipped(output_filename, "w") as outfile:
                packed = json.dumps(d)
                outfile.write(packed)

        self.filter_data_saved = True

    def save_selection(self):
        if self.save_bur:
            self.save_bi4_bur_first_last()
        if self.save_sl5:
            self.save_filter_data()

    def fill_pie_windows(self, k):
        self.windows = k
        self.comboBox_3.addItems(k.keys())

    def fill_detectors(self, k):
        self.detectors = k
        self.comboBox_2.addItems(k.keys())

    def update_detectors(self):
        key = self.comboBox_2.currentText()
        s = ", ".join([str(i) for i in self.detectors[key]["chs"]])
        self.lineEdit_4.setText(s)
        self.update_parameter()

    def update_pie_windows(self):
        key = self.comboBox_3.currentText()
        pie_win = self.windows[key]
        s = ";".join([f"{i[0]}-{i[1]}" for i in pie_win])
        self.lineEdit_5.setText(s)
        self.update_parameter()

    @chisurf.gui.decorators.init_with_ui("tttr_burst_finder.ui")
    def __init__(self, *args,
                 windows,
                 detectors,
                 callback_function=None,
                 show_dT=True,
                 show_burst_histogram=True,
                 show_mcs=True,
                 show_decay=True,
                 show_filter=True,
                 initial_trace_bin_width: float = 1.0,
                 initial_photon_threshold: int = 100,
                 initial_tw_size: float = 1.0,
                 initial_max_gap: int = 6,
                 initial_decay_coarse: int = 16,
                 initial_number_of_burst_bins: int = 50,
                 initial_dT_min: float = 0.0001,
                 initial_dT_max: float = 0.15,
                 **kwargs):
        self.setTitle("Photon filter / burst finder")

        # Store the windows and detectors dictionaries and update corresponding UI elements.
        self.windows = windows
        self.detectors = detectors
        self.fill_detectors(detectors)
        self.fill_pie_windows(windows)
        self.callback_function = callback_function

        self.comboBox.clear()
        self.comboBox.insertItem(0, "Auto")  # Insert "Auto" at index 0
        self.comboBox.insertItems(1, list(tttrlib.TTTR.get_supported_container_names()))

        self.settings: dict = dict()
        tttr_filenames: typing.List[pathlib.Path] = list()
        self.settings['tttr_filenames'] = tttr_filenames
        self.settings['count_rate_filter']: CountRateFilterSettings = dict()
        self.settings['delta_macro_time_filter']: DeltaMacroTimeFilterSettings = dict()
        self.filter_data_saved = False

        def cc():
            self.spinBox_4.setMaximum(len(self.settings['tttr_filenames']) - 1)
            self.spinBox_4.setValue(len(self.settings['tttr_filenames']) - 1)
            self.read_tttr()

        self.textEdit.setVisible(False)
        chisurf.gui.decorators.lineEdit_dragFile_injector(
            self.lineEdit,
            call=cc,
            target=self.settings['tttr_filenames']
        )
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sizePolicy)

        self.tttr = None

        # Set default dT values (which will later be overridden by any passed initial_dT_min/max)
        self._dT_min = 0.0001
        self._dT_max = 0.15

        # Define common pens for the plots.
        color_all = QtGui.QColor(255, 255, 0, 64)
        color_selected = QtGui.QColor(0, 255, 255, 255)
        pen2 = pg.mkPen(color_all, width=1, style=QtCore.Qt.SolidLine)
        pen1 = pg.mkPen(color_selected, width=1, style=QtCore.Qt.SolidLine)

        # Plot widget: Burst size histogram
        self.pw_burst_histogram = pg.plot()
        self.plot_burst_histogram = self.pw_burst_histogram.getPlotItem()
        self.pw_burst_histogram.resize(100, 80)
        self.pw_burst_histogram.setVisible(show_burst_histogram)

        # Plot widget: Delta macro time plot
        self.pw_dT = pg.plot()
        self.plot_item_dt = self.pw_dT.getPlotItem()
        self.plot_unselected = self.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen2)
        self.plot_selected = self.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen1)
        self.pw_dT.resize(200, 40)
        self.pw_dT.setVisible(show_dT)

        # Plot widget: MCS trace
        self.pw_mcs = pg.plot()
        self.plot_item_mcs = self.pw_mcs.getPlotItem()
        # Set axis labels for MCS trace plot
        self.plot_item_mcs.setLabel('bottom', 'Time (s)')
        self.plot_item_mcs.setLabel('left', 'Intensity')
        self.plot_mcs_all = self.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen2)
        self.plot_mcs_selected = self.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen1)
        self.pw_mcs.resize(200, 80)
        self.pw_mcs.setVisible(show_mcs)

        # Plot widget: Fluorescence decay
        self.pw_decay = pg.plot()
        self.plot_item_decay = self.pw_decay.getPlotItem()
        # Set axis labels for decay plot
        self.plot_item_decay.setLabel('bottom', 'Time (ns)')
        self.plot_item_decay.setLabel('left', 'Counts')
        self.plot_decay_all = self.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen2)
        self.plot_decay_selected = self.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen1)
        self.pw_decay.resize(200, 80)
        self.pw_decay.setVisible(show_decay)

        # Plot widget: Filtered photons
        self.pw_filter = pg.plot()
        self.pw_filter.setXLink(self.pw_dT)
        self.pw_dT.setMouseEnabled(x=False, y=False)
        self.pw_filter.setMouseEnabled(x=False, y=False)
        self.plot_item_sel = self.pw_filter.getPlotItem()
        self.plot_select = self.plot_item_sel.plot(x=[1.0], y=[1.0])
        self.pw_filter.resize(200, 20)
        self.pw_filter.setVisible(show_filter)

        self.plot_item_dt.setLogMode(False, True)
        self.plot_item_decay.setLogMode(False, True)
        self.plot_item_sel.setLogMode(False, False)

        ca = list(matplotlib.colors.hex2color(colors["region_selector"]))
        co = [ca[0] * 255, ca[1] * 255, ca[2] * 255, colors["region_selector_alpha"]]
        self.region_selector = pg.LinearRegionItem(brush=co, orientation='horizontal',
                                                   values=(np.log10(0.001), np.log10(0.15)))
        self.pw_dT.addItem(self.region_selector)

        def onRegionUpdate(evt):
            lb, ub = self.region_selector.getRegion()
            if self.pw_dT.getAxis('left').logMode:
                lb, ub = 10 ** lb, 10 ** ub
            self._dT_min = lb
            self._dT_max = ub

            self.doubleSpinBox_2.blockSignals(True)
            self.doubleSpinBox_3.blockSignals(True)
            self.doubleSpinBox_2.setValue(lb)
            self.doubleSpinBox_3.setValue(ub)
            self.doubleSpinBox_2.blockSignals(False)
            self.doubleSpinBox_3.blockSignals(False)

            self.update_plots()
            self.update_output_path()

        self.region_selector.sigRegionChangeFinished.connect(onRegionUpdate)

        # --- Add new initial settings for burst selection parameters ---
        # Set the initial trace bin width (time window) for the MCS trace.
        self.doubleSpinBox_4.setValue(initial_trace_bin_width)
        # Set the photon number threshold.
        self.spinBox.setValue(initial_photon_threshold)
        # Set the maximum gap (for gap filling in burst selection).
        self.spinBox_7.setValue(initial_max_gap)
        # Set the decay coarse binning.
        self.spinBox_5.setValue(initial_decay_coarse)
        # Set the number of burst bins for the histogram.
        self.spinBox_6.setValue(initial_number_of_burst_bins)
        # Set the lower and upper limits for dT (delta macro time) filtering.
        self._dT_min = initial_dT_min
        self._dT_max = initial_dT_max
        self.doubleSpinBox_2.setValue(initial_dT_min)
        self.doubleSpinBox_3.setValue(initial_dT_max)
        # Update the region selector to match the new dT limits.
        self.doubleSpinBox.setValue(initial_tw_size)
        self.region_selector.setRegion((np.log10(initial_dT_min), np.log10(initial_dT_max)))

        # Add widgets to layout.
        self.gridLayout_6.addWidget(self.pw_dT, 0, 0, 1, 3)
        self.gridLayout_6.addWidget(self.pw_filter, 1, 0, 1, 3)
        self.gridLayout_6.addWidget(self.pw_mcs, 2, 0, 1, 1)
        self.gridLayout_6.addWidget(self.pw_decay, 2, 1, 1, 1)
        self.gridLayout_6.addWidget(self.pw_burst_histogram, 0, 1, 2, 1)

        # Connect actions.
        self.actionUpdate_Values.triggered.connect(self.update_parameter)
        self.actionUpdateUI.triggered.connect(self.updateUI)
        self.actionFile_changed.triggered.connect(self.read_tttr)
        self.actionRegionUpdate.triggered.connect(self.onRegionUpdate)

        self.toolButton_2.toggled.connect(self.pw_mcs.setVisible)
        self.toolButton_3.toggled.connect(self.pw_decay.setVisible)
        self.toolButton_4.toggled.connect(self.pw_filter.setVisible)
        self.toolButton_7.toggled.connect(self.pw_burst_histogram.setVisible)
        self.toolButton_5.clicked.connect(self.save_selection)
        self.toolButton_6.clicked.connect(self.onClearFiles)

        self.comboBox_2.currentTextChanged.connect(self.update_detectors)
        self.comboBox_3.currentTextChanged.connect(self.update_pie_windows)

        # Set the custom validator.
        validator = CommaSeparatedIntegersValidator()
        self.lineEdit_4.setValidator(validator)

        self.update_parameter()
