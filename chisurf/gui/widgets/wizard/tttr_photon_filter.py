import os
import pathlib
import typing
import zipfile
import shutil
from datetime import datetime

import tttrlib
import json
import time
import numpy as np
import pandas as pd

import pyqtgraph as pg
import matplotlib

import chisurf.fio as io
import chisurf.fio.fluorescence
import chisurf.math
import chisurf.gui.decorators
from chisurf.gui import QtGui, QtWidgets, QtCore, uic
from chisurf.math.signal import fill_small_gaps_in_array
from chisurf.settings.path_utils import get_path
from chisurf.settings.file_utils import safe_open_file
from chisurf.gui.widgets.wizard.tttr_channel_definition import save_detector_setups
from chisurf.gui.widgets.progress import EnhancedProgressDialog

# Path to the central detector setups file
DETECTOR_SETUPS_FILE = get_path('settings') / 'detector_setups.json'

def load_detector_setups(file_path=None):
    """Load detector setups from the central settings file or a custom file.

    Args:
        file_path: Optional custom path to load from. If None, uses DETECTOR_SETUPS_FILE.
    """
    return safe_open_file(
        file_path=file_path or DETECTOR_SETUPS_FILE,
        processor=json.load,
        default_value={"setups": {}}
    )

QValidator = QtGui.QValidator
colors = chisurf.settings.gui['plot']['colors']


def create_array_with_ones(start_stop_pairs, length):
    """
    Create a boolean array of the given length, set to True (1)
    in the intervals [start, stop) defined by start_stop_pairs.
    """
    arr = np.zeros(length, dtype=bool)
    for start, stop in start_stop_pairs:
        arr[start:stop] = 1
    return arr


class ProgressWindow(QtWidgets.QDialog):
    def __init__(self, title="Processing Files", message="Loading files...", max_value=100, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel(message)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, max_value)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def set_value(self, value: int):
        self.progress_bar.setValue(value)
        QtWidgets.QApplication.processEvents()


class CommaSeparatedIntegersValidator(QValidator):

    """
    QValidator to ensure input is a comma-separated list of valid
    integers between 0 and 255.
    """
    def validate(self, input_str, pos):
        if not input_str:
            return QValidator.Intermediate, input_str, pos

        parts = input_str.split(',')
        for part in parts:
            part = part.strip()
            if part == '':
                continue
            if not part.isdigit():
                return QValidator.Intermediate, input_str, pos
            num = int(part)
            if num < 0 or num > 255:
                return QValidator.Invalid, input_str, pos

        if input_str.endswith(','):
            return QValidator.Intermediate, input_str, pos

        return QValidator.Acceptable, input_str, pos

    def fixup(self, input_str):
        input_str = input_str.rstrip(',')
        parts = input_str.split(',')
        valid_parts = []
        for part in parts:
            part = part.strip()
            if part.isdigit():
                num = int(part)
                if 0 <= num <= 255:
                    valid_parts.append(str(num))
        return ', '.join(valid_parts)


class WizardTTTRPhotonFilter(QtWidgets.QWizardPage):
    """
    WizardTTTRPhotonFilter loads TTTR files and applies photon filtering
    based on count-rate or burst-search criteria, channel selection,
    microtime ranges, and optional gap-filling.

    Parameters
    ----------
    windows : dict
        Dictionary of predefined PIE windows. Passed to the relevant UI comboBox.
    detectors : dict
        Dictionary of predefined detectors. Passed to the relevant UI comboBox.
    show_dT : bool, default=True
        Whether to show the dT plot (between consecutive photons).
    show_filter : bool, default=True
        Whether to show the filter/selection plot.
    show_mcs : bool, default=True
        Whether to show the intensity trace (MCS) plot.
    show_decay : bool, default=True
        Whether to show the decay plot (microtime histogram).
    show_burst : bool, default=True
        Whether to show the burst histogram plot.
    default_dT_min : float, default=0.0001
        Initial lower bound for delta macro-time filter (in ms).
    default_dT_max : float, default=0.15
        Initial upper bound for delta macro-time filter (in ms).
    use_dT_min : bool, default=True
        Whether the lower bound of delta macro-time filter is active initially.
    use_dT_max : bool, default=True
        Whether the upper bound of delta macro-time filter is active initially.
    default_photon_threshold : int, default=160
        Initial threshold for count rate or burst search.
    default_count_rate_window_ms : float, default=1.0
        Initial time window (ms) for count-rate based filtering.
    invert_count_rate_filter : bool, default=False
        Whether to invert the count-rate filter initially.
    default_filter_mode : str, default='count_rate'
        Selects which radio button filter mode is active at initialization.
        Valid options: 'count_rate' or 'burst'.
        - If 'count_rate', the `radioButton` (count-rate filter) is checked.
        - If 'burst', the `radioButton_2` (burst-search filter) is checked.

    Notes
    -----
    The filter mode can also be changed by the user at runtime. The
    current mode can be queried from the `used_filter` property.
    """

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
    def filetype(self) -> str | None:
        setup_name = self.comboBox.currentText()
        if not setup_name or setup_name == "No setups available":
            # Display warning message if no setup is selected
            QtWidgets.QMessageBox.warning(
                self, 
                "No Setup Selected", 
                "Please define a setup first in the Detector Configuration page."
            )
            return None

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the file type from the setup's tttr_reading section
            setup_data = setups["setups"][setup_name]
            if "tttr_reading" in setup_data and "file_type" in setup_data["tttr_reading"]:
                file_type = setup_data["tttr_reading"]["file_type"]

                # If file type is Auto, try to infer from current file
                if file_type == 'Auto':
                    current_file = self.current_tttr_filename
                    if current_file and pathlib.Path(current_file).exists():
                        file_type_int = tttrlib.inferTTTRFileType(current_file)
                        if file_type_int is not None and file_type_int >= 0:
                            container_names = tttrlib.TTTR.get_supported_container_names()
                            if 0 <= file_type_int < len(container_names):
                                return container_names[file_type_int]
                    return None
                return file_type

        # If setup doesn't exist or doesn't have file type, show warning
        QtWidgets.QMessageBox.warning(
            self, 
            "Invalid Setup", 
            f"The selected setup '{setup_name}' is not properly defined. Please check your setup configuration."
        )
        return None

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

    @plot_max.setter
    def plot_max(self, v):
        self.spinBox_3.blockSignals(True)
        self.spinBox_3.setValue(v)
        self.spinBox_3.blockSignals(False)

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
    def min_ph(self):
        """
        The minimum number of photons for burst detection (or any other usage).
        This property gets the current spinBox value.
        """
        return self.spinBox.value()

    @min_ph.setter
    def min_ph(self, value):
        """
        Sets the spinBox value to the specified integer/float.
        """
        self.spinBox.setValue(value)

    @property
    def ph_window(self):
        """
        The photon window size for burst detection (or other usage).
        This property gets the current spinBox_8 value.
        """
        return self.spinBox_8.value()

    @ph_window.setter
    def ph_window(self, value):
        """
        Sets the spinBox_8 value to the specified integer/float.
        """
        self.spinBox_8.setValue(value)

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
    def used_filter(self):
        """
        Returns the currently active filter mode:
        - 'count_rate' if `radioButton` is checked
        - 'burst' if `radioButton_2` is checked
        """
        if self.radioButton.isChecked():
            return 'count_rate'
        elif self.radioButton_2.isChecked():
            return 'burst'

    @property
    def selected(self):
        """
        Returns a boolean mask (as np.uint8 array) of selected photons
        based on all applied filters: channel/microtime filter,
        dT thresholding, count-rate filter (if active),
        or burst filter (if active), plus optional gap-filling.
        """
        start_time = time.time()
        dT = self.dT
        tttr = self.tttr
        if dT is None:
            return list()

        s = np.ones_like(dT, dtype=bool)
        chs = self.channels
        if len(chs) > 0:
            mask = tttrlib.TTTRMask()
            mask.select_channels(tttr, chs, mask=True)
            m = mask.get_mask()
            s = np.logical_and(s, m)

        if self.microtime_ranges:
            mask = tttrlib.TTTRMask()
            mask.select_microtime_ranges(tttr, self.microtime_ranges)
            mask.flip()
            m = mask.get_mask()
            s = np.logical_and(s, m)

        if self.use_lower:
            s = np.logical_and(s, dT >= self.dT_min)
        if self.use_upper:
            s = np.logical_and(s, dT <= self.dT_max)

        # Apply either count_rate or burst filter depending on radio button
        if self.used_filter == 'count_rate':
            if self.settings['count_rate_filter_active']:
                filter_options = self.settings['count_rate_filter']
                selection_idx = self.tttr.get_selection_by_count_rate(
                    **filter_options, make_mask=True
                )
                s = np.logical_and(s, selection_idx >= 0)

        elif self.used_filter == 'burst':
            min_ph = self.min_ph
            ph_window = self.ph_window
            tw = self.dT_max / 1000.0
            start_stop = tttr.burst_search(min_ph, ph_window, tw)
            start_stop = np.array(start_stop).reshape((-1, 2))
            n = len(tttr)
            sel = create_array_with_ones(start_stop, n)
            s = np.logical_and(s, sel)

        if self.max_gap > 0 and self.use_gap_fill:
            s = fill_small_gaps_in_array(s, max_gap=self.max_gap)

        end_time = time.time()
        chisurf.logging.log(0, f"Elapsed time: {end_time - start_time}")
        return s.astype(dtype=np.uint8)

    @property
    def burst_start_stop(self):
        """
        Returns the start-stop indices of bursts in the selected mask,
        allowing for small gaps with max_gap=4 by default.
        """
        selected = self.selected
        max_gap = self.max_gap
        if len(selected) > max_gap:
            return chisurf.math.signal.find_bursts(selected, max_gap=max_gap)
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
        
    @property
    def save_hdf5(self):
        # For testing purposes, return True
        # In a production environment, this would check a checkbox or configuration setting
        return True  # Change to False to disable HDF5 output

    def update_filter_plot(self):
        """
        Update the plot showing which photons are selected (1) vs unselected (0).
        """
        try:
            selected = self.selected
            if isinstance(selected, (np.ndarray, list)) and len(selected) > 0:
                n_min = self.plot_min
                n_max = self.plot_max
                x = np.arange(n_min, n_max)
                y = selected[n_min:n_max]
                self.plot_select.setData(x=x, y=y)
            else:
                # If selected is not available, clear the plot
                self.plot_select.setData([], [])
        except (AttributeError, IndexError, TypeError):
            # Handle the case when self.selected is not available or returns an error
            self.plot_select.setData([], [])

    def update_dt_plot(self):
        """
        Update the plot of dT vs photon index, highlighting selected vs unselected points.
        """
        try:
            dT = self.dT
            if isinstance(dT, np.ndarray):
                n_min = self.plot_min
                n_max = self.plot_max
                mask = self.selected[n_min:n_max].astype(bool)
                y = dT[n_min:n_max]
                x = np.arange(n_min, n_max)

                mx = np.ma.masked_array(x, mask=~mask)
                my = np.ma.masked_array(y, mask=~mask)
                self.plot_selected.setData(x=mx.compressed(), y=my.compressed())

                mx = np.ma.masked_array(x, mask=mask)
                my = np.ma.masked_array(y, mask=mask)
                self.plot_unselected.setData(x=mx.compressed(), y=my.compressed())
            else:
                # If dT is not available, clear the plots
                self.plot_selected.setData([], [])
                self.plot_unselected.setData([], [])
        except AttributeError:
            # Handle the case when self.dT is not available
            self.plot_selected.setData([], [])
            self.plot_unselected.setData([], [])

    @property
    def do_mcs_plot(self) -> bool:
        return self.toolButton_2.isChecked()

    @do_mcs_plot.setter
    def do_mcs_plot(self, v):
        self.toolButton_2.setChecked(v)

    @property
    def do_burst_photon_plot(self) -> bool:
        return self.toolButton_7.isChecked()

    @do_burst_photon_plot.setter
    def do_burst_photon_plot(self, v):
        self.toolButton_7.setChecked(v)

    @property
    def do_microtime_plot(self) -> bool:
        return self.toolButton_3.isChecked()

    @do_microtime_plot.setter
    def do_microtime_plot(self, v):
        self.toolButton_3.setChecked(v)

    def update_mcs_plot(self):
        """
        Update the intensity trace plot (MCS) for both all photons and the selected subset.
        """
        if self.do_mcs_plot:
            try:
                tttr = self.tttr
                if isinstance(tttr, tttrlib.TTTR):
                    try:
                        selected = self.selected
                        if isinstance(selected, (np.ndarray, list)) and len(selected) > 0:
                            idx = np.where(selected)[0]
                            tw = self.trace_bin_width / 1000.0
                            trace_selected = tttr[idx].get_intensity_trace(time_window_length=tw)
                            x1 = np.arange(len(trace_selected)) * tw
                            self.plot_mcs_selected.setData(x1, trace_selected)
                        else:
                            self.plot_mcs_selected.setData([], [])
                    except (AttributeError, IndexError, TypeError):
                        self.plot_mcs_selected.setData([], [])

                    trace_all = tttr.get_intensity_trace(time_window_length=self.trace_bin_width / 1000.0)
                    x2 = np.arange(len(trace_all)) * (self.trace_bin_width / 1000.0)
                    self.plot_mcs_all.setData(x2, trace_all)
                else:
                    self.plot_mcs_all.setData([], [])
                    self.plot_mcs_selected.setData([], [])
            except (AttributeError, IndexError, TypeError):
                self.plot_mcs_all.setData([], [])
                self.plot_mcs_selected.setData([], [])
        else:
            self.plot_mcs_all.setData(x=[1.0], y=[1.0])
            self.plot_mcs_selected.setData(x=[1.0], y=[1.0])

    def update_burst_histogram(self):
        """
        Update the histogram of burst lengths (only relevant if burst mode is active).
        """
        if self.do_burst_photon_plot:
            try:
                burst_lengths = self.burst_lengths
                if isinstance(burst_lengths, np.ndarray) and len(burst_lengths) > 0:
                    num_bins = self.number_of_burst_bins
                    hist, bin_edges = np.histogram(burst_lengths, bins=num_bins)

                    self.pw_burst_histogram.clear()
                    self.pw_burst_histogram.addItem(
                        pg.BarGraphItem(
                            x0=bin_edges[:-1],
                            x1=bin_edges[1:],
                            y0=0,
                            y1=hist,
                            brush='b',
                            pen='w'
                        )
                    )
                    self.plot_burst_histogram = self.pw_burst_histogram.plot(
                        bin_edges,
                        hist,
                        pen='b',
                        stepMode=True
                    )

                    total_bursts = np.sum(hist)
                    pos_x = 0.5 * (bin_edges[0] + bin_edges[-1]) if bin_edges.size > 1 else 0
                    pos_y = (hist.max() * 0.9) if hist.size > 0 else 1.0
                    self.total_bursts_label = pg.TextItem(
                        f"Total bursts: {total_bursts}",
                        anchor=(0, 0),
                        color='w'
                    )
                    self.total_bursts_label.setPos(pos_x, pos_y)
                    self.pw_burst_histogram.addItem(self.total_bursts_label)
                    self.pw_burst_histogram.setYRange(0.0, max(hist) if hist.size > 0 else 1.0)
                    self.pw_burst_histogram.setXRange(0.0, max(bin_edges) if bin_edges.size > 0 else 1.0)
                else:
                    self.pw_burst_histogram.clear()
                    self.pw_burst_histogram.addItem(
                        pg.TextItem(
                            "No burst data available",
                            anchor=(0.5, 0.5),
                            color='w'
                        )
                    )
            except (AttributeError, IndexError, TypeError, ValueError):
                self.pw_burst_histogram.clear()
                self.pw_burst_histogram.addItem(
                    pg.TextItem(
                        "Error processing burst data",
                        anchor=(0.5, 0.5),
                        color='w'
                    )
                )

    def update_decay_plot(self):
        """
        Update the microtime decay plot (histogram), comparing all photons vs selected photons.
        """
        if self.do_microtime_plot:
            try:
                tttr = self.tttr
                if isinstance(tttr, tttrlib.TTTR):
                    try:
                        selected = self.selected
                        if isinstance(selected, (np.ndarray, list)) and len(selected) > 0:
                            idx = np.where(selected)[0]
                            y, x = tttr[idx].get_microtime_histogram(self.decay_coarse)
                            if np.any(y > 0):
                                idx_max = np.where(y > 0)[0][-1]
                                x = x[:idx_max]
                                y = y[:idx_max]
                                x *= 1e9
                                self.plot_decay_selected.setData(x=x, y=y)
                            else:
                                self.plot_decay_selected.setData([], [])
                        else:
                            self.plot_decay_selected.setData([], [])
                    except (AttributeError, IndexError, TypeError):
                        self.plot_decay_selected.setData([], [])

                    y, x = tttr.get_microtime_histogram(self.decay_coarse)
                    if np.any(y > 0):
                        idx_max = np.where(y > 0)[0][-1]
                        x = x[:idx_max]
                        y = y[:idx_max]
                        x *= 1e9
                        self.plot_decay_all.setData(x=x, y=y)
                    else:
                        self.plot_decay_all.setData([], [])
                else:
                    self.plot_decay_all.setData([], [])
                    self.plot_decay_selected.setData([], [])
            except (AttributeError, IndexError, TypeError):
                self.plot_decay_all.setData([], [])
                self.plot_decay_selected.setData([], [])
        else:
            self.plot_decay_all.setData(x=[1.0], y=[1.0])
            self.plot_decay_selected.setData(x=[1.0], y=[1.0])

    def update_plots(self, selection: str = "all"):
        """
        Convenience method to update different sets of plots
        ('mcs', 'decay', 'dT', 'filter', 'burst').
        """
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
        """
        Called when the user changes the active file.
        Looks up the preloaded TTTR object in self.tttr_objects.
        """
        fn = self.current_tttr_filename
        if not fn:
            return

        p = pathlib.Path(fn).resolve()
        p_str = str(p)

        if p_str in self.tttr_objects:
            try:
                self.tttr = self.tttr_objects[p_str]
                self.plot_max = len(self.tttr) - 1
                header = self.tttr.get_header()
                s = header.json
                d = json.loads(s)
                self.settings['header'] = d
                self.update_plots()
            except Exception as e:
                # If there's an error accessing the TTTR object's properties,
                # display an error message and exit early
                QtWidgets.QMessageBox.critical(
                    self, 
                    "Error Reading File", 
                    f"Failed to read file '{p.name}' with the selected setup.\n\n"
                    f"Error: {str(e)}\n\n"
                    f"Please check that you have selected the correct setup for this file type."
                )
                # Remove the problematic TTTR object from the dictionary
                if p_str in self.tttr_objects:
                    del self.tttr_objects[p_str]
                self.tttr_objects = dict()
                self.onClearFiles()
                return  # Exit early to prevent undefined state

    def update_output_path(self):
        """
        Update the suggested output path (lineEdit_2) based on the current settings.
        """
        if len(self.channels) > 0:
            chs = ','.join([str(x) for x in self.channels])
        else:
            chs = 'All'

        # Determine path format based on filter mode
        if self.used_filter == "count_rate":
            path_prefix = f"countrate"
        else:
            path_prefix = "burstwise"

        s = f"{path_prefix}_{chs} {self.dT_max:.4f}#{self.photon_number_threshold}"
        self.lineEdit_2.setText(s)

    def update_parameter(self):
        """
        Update internal settings whenever the user changes filters or region selectors.
        """
        lb, ub = self.region_selector.getRegion()
        self.settings['count_rate_filter_active'] = self.checkBox_4.isChecked()
        self.settings['count_rate_filter']['n_ph_max'] = int(self.spinBox.value())
        self.settings['count_rate_filter']['time_window'] = float(self.doubleSpinBox.value()) * 1e-3
        self.settings['count_rate_filter']['invert'] = bool(self.checkBox.isChecked())

        self.settings['delta_macro_time_filter']['dT_min'] = 10.0 ** lb if self.pw_dT.getAxis('left').logMode else lb
        self.settings['delta_macro_time_filter']['dT_max'] = 10.0 ** ub if self.pw_dT.getAxis('left').logMode else ub
        self.settings['delta_macro_time_filter']['dT_min_active'] = self.checkBox_2.isChecked()
        self.settings['delta_macro_time_filter']['dT_max_active'] = self.checkBox_3.isChecked()

        self.update_plots()
        self.update_output_path()

    def onClearFiles(self):
        """
        Clears the list of filenames, resets the spinBox, clears the lineEdit,
        unsets the current TTTR object, and also clears all plots.
        """
        self.settings['tttr_filenames'].clear()
        self.spinBox_4.setMaximum(0)
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
        """
        Updates the lineEdit with the currently active file name.
        """
        self.lineEdit.setText(self.current_tttr_filename)

    def onRegionUpdate(self):
        """
        Sync the numeric spinBoxes (doubleSpinBox_2/3) with the region item in the dT plot.
        """
        lb, ub = self.doubleSpinBox_2.value(), self.doubleSpinBox_3.value()
        if self.pw_dT.getAxis('left').logMode:
            lb, ub = np.log10(lb), np.log10(ub)
        self.region_selector.setRegion(rgn=(lb, ub))

    @property
    def parent_directories(self) -> typing.List[pathlib.Path]:
        """
        For each TTTR filename, get the parent directory with the configured target path.
        """
        r = []
        for filename in self.settings['tttr_filenames']:
            filename = filename.replace('\x00', '')
            fn = pathlib.Path(filename).absolute()
            t = fn.parent / self.target_path
            r.append(t)
        return r

    def save_selection(self, output_types = None, zip_output = False, remove_folder = False):
        """
        Save the selection data in .bur or .json.gz, depending on user checkboxes or specified output_types,
        and display a progress bar while saving. Optionally zip the output folder after saving and remove
        the original folder if requested.
        
        Parameters:
        -----------
        output_types : set, optional
            Set of output types to save. If provided, this overrides the checkbox settings.
            Possible values: "bur", "sl5", "hdf5"
        zip_output : bool, optional
            Whether to zip the output folder after saving. Default is False.
        remove_folder : bool, optional
            Whether to remove the original folder after zipping. Default is False.
            This parameter is only used if zip_output is True.
        """
        total_files = len(self.settings['tttr_filenames'])
        total_tasks = 0
        
        # If output_types is provided, use it; otherwise, use checkbox settings
        if output_types is None:
            output_types = set()
            if self.save_bur:
                output_types.add("bur")
            if self.save_sl5:
                output_types.add("sl5")
            if self.save_hdf5:
                output_types.add("hdf5")
        
        # Calculate total tasks based on output_types
        if "bur" in output_types:
            total_tasks += total_files
        if "sl5" in output_types:
            total_tasks += total_files
        if "hdf5" in output_types:
            total_tasks += 1  # Only one task for HDF5 as we create a single file
        
        # Add additional tasks for zipping if needed
        if zip_output:
            total_tasks += 2  # Add tasks for zipping and potential folder removal

        # Create enhanced progress dialog
        progress = EnhancedProgressDialog("Saving Selection", "Initializing...", 0, total_tasks, self)
        progress.show()

        current_task = 0
        
        # For HDF5 output, we'll collect DataFrames from all files
        all_dfs = []
        
        # Process each file
        for t, filename in zip(self.parent_directories, self.settings['tttr_filenames']):
            fn = pathlib.Path(filename)
            resolved_path = str(fn.resolve())
            self.tttr = self.tttr_objects[resolved_path]
            start_stop = self.burst_start_stop

            # Generate DataFrame with or without interleaved zeros based on needs
            # If only HDF5 is needed, skip interleaved zeros
            # If both are needed, include interleaved zeros for BUR compatibility
            include_zeros = "bur" in output_types

            df = io.fluorescence.burst.generate_burst_dataframe(
                start_stop=start_stop,
                filename=fn,
                tttr=self.tttr,
                windows=self.windows,
                detectors=self.detectors,
                include_interleaved_zeros=include_zeros
            )
            
            # Handle BUR output
            if "bur" in output_types:
                base_name = fn.stem
                bur_directory = t / 'bi4_bur'
                bur_directory.mkdir(exist_ok=True, parents=True)
                bur_filename = bur_directory / f"{base_name}.bur"
                
                # Update progress with current file info
                progress.update_text(f"Saving BUR file: {base_name}.bur")
                
                # Write the DataFrame to BUR file
                io.fluorescence.burst.write_dataframe_to_bur(df, bur_filename)
                
                # Write MTI summary
                mt = self.tttr.macro_times[-1] * self.tttr.header.macro_time_resolution
                io.fluorescence.burst.write_mti_summary(
                    filename=fn,
                    analysis_dir=t,
                    max_macro_time=mt,
                    append=True
                )
                
                current_task += 1
                progress.update_progress(current_task)
                if progress.wasCanceled():
                    break
            
            # Collect DataFrame for HDF5 if needed
            if "hdf5" in output_types:
                # If we're only using HDF5 (not BUR), we need to filter out interleaved zeros
                if not "bur" in output_types and df is not None and include_zeros:
                    # Filter out rows where all numeric columns are zero
                    # This removes the interleaved zero rows
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        df = df[~(df[numeric_cols] == 0).all(axis=1)]
                
                if df is not None:
                    # Add a column to identify the source file
                    df_copy = df.copy()
                    df_copy['Source File'] = str(fn)
                    all_dfs.append(df_copy)
        
        # Write HDF5 file if needed
        if "hdf5" in output_types and all_dfs:
            # Update progress
            progress.update_text("Creating combined HDF5 file...")
            
            # Combine all DataFrames into a single DataFrame
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Create the HDF5 directory
            hdf5_directory = self.parent_directories[0] / 'hdf5'
            hdf5_directory.mkdir(exist_ok=True, parents=True)
            
            # Create a single HDF5 file with a timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            hdf5_filename = hdf5_directory / f"burst_data_{timestamp}.h5"
            
            # Convert problematic columns to string type to avoid serialization issues
            # The 'First File' and 'Last File' columns may contain mixed types
            if 'First File' in combined_df.columns:
                combined_df['First File'] = combined_df['First File'].astype(str)
            if 'Last File' in combined_df.columns:
                combined_df['Last File'] = combined_df['Last File'].astype(str)
            
            # Update progress with file info
            progress.update_text(f"Writing HDF5 file: {hdf5_filename.name}")
            
            # Write the combined DataFrame to the HDF5 file
            # Using 'results' as the key for compatibility with mfd-hdf format expected by ndxplorer
            combined_df.to_hdf(
                hdf5_filename, 
                key='results',
                mode='w',
                complevel=9,
                complib='blosc',
                format='table'
            )
        
            # Update progress
            current_task += 1
            progress.update_progress(current_task, "HDF5 file created successfully")
            if progress.wasCanceled():
                return

        if "sl5" in output_types:
            for t, filename in zip(self.parent_directories, self.settings['tttr_filenames']):
                parent_directory = t / 'sl5'
                parent_directory.mkdir(exist_ok=True, parents=True)
                parent_directory = parent_directory.absolute()
                fn = pathlib.Path(filename)
                base_name = fn.stem
                
                # Update progress with current file info
                progress.update_text(f"Saving SL5 file: {base_name}.json.gz")
                
                resolved_path = str(fn.resolve())
                self.tttr = self.tttr_objects[resolved_path]

                d = {
                    'filename': os.path.relpath(fn, t),
                    'filetype': self.filetype,
                    'count_rate_filter': self.settings['count_rate_filter'],
                    'delta_macro_time_filter': self.settings['delta_macro_time_filter'],
                    'filter': chisurf.fio.compress_numpy_array(self.selected)
                }
                output_filename = parent_directory / f"{base_name}.json.gz"
                with io.open_maybe_zipped(output_filename, "w") as outfile:
                    packed = json.dumps(d)
                    outfile.write(packed)
                current_task += 1
                progress.update_progress(current_task)
                if progress.wasCanceled():
                    break

        # If we're not zipping, finish and close the dialog
        # Otherwise just mark data as saved and continue with the same dialog
        if not zip_output:
            progress.finish("Selection saved successfully")
        else:
            # Just hide the dialog temporarily if we're going to zip
            progress.setValue(progress.maximum())
            
        self.filter_data_saved = True
        
        # Zip the output folder if requested
        if zip_output and self.parent_directories:
            # Get the first output directory (they should all be in the same parent directory)
            output_folder = self.parent_directories[0]
            if output_folder.exists():
                # Update progress dialog for zipping
                progress.update_text(f"Creating ZIP archive of: {output_folder}")
                current_task += 1
                progress.update_progress(current_task)
                if progress.wasCanceled():
                    return
                
                # Zip the output folder
                zip_file = self.zip_output_folder(output_folder, progress)
                
                if zip_file and zip_file.exists():
                    if remove_folder:
                        # Remove the original folder after successful zipping if requested
                        try:
                            progress.update_text(f"Removing original folder: {output_folder}")
                            shutil.rmtree(output_folder)
                            progress.update_text(f"ZIP archive created: {zip_file}\nOriginal folder removed")
                        except Exception as e:
                            progress.update_text(f"ZIP archive created: {zip_file}\nFailed to remove folder: {str(e)}")
                    else:
                        progress.update_text(f"ZIP archive created: {zip_file}")
                    
                    # Give user a moment to see the final status and then finish
                    progress.finish(auto_close=True)

    def fill_pie_windows(self, k):
        self.windows = k
        self.comboBox_3.addItems(k.keys())

    def zip_output_folder(self, output_folder, existing_progress=None, add_timestamp=False):
        """
        Zip the output folder and its contents.
        
        Parameters:
        -----------
        output_folder : pathlib.Path
            Path to the folder to be zipped.
        existing_progress : QtWidgets.QProgressDialog, optional
            An existing progress dialog to use instead of creating a new one.
        add_timestamp : bool, optional
            Whether to add a timestamp to the zip filename. Default is False.
        
        Returns:
        --------
        pathlib.Path
            Path to the created zip file.
        """
        if not output_folder.exists() or not output_folder.is_dir():
            return None
        
        if add_timestamp:
            # Create a timestamp for the zip filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            zip_filename = output_folder.parent / f"{output_folder.name}_{timestamp}.zip"
        else:
            zip_filename = output_folder.parent / f"{output_folder.name}.zip"
        
        # Use existing progress dialog if provided, otherwise create a new one
        using_existing_progress = existing_progress is not None
        if not using_existing_progress:
            progress = EnhancedProgressDialog("Creating ZIP Archive", "Zipping output folder...", 0, 100, self)
            progress.show()
        else:
            progress = existing_progress
            progress.update_text("Zipping output folder...")
        
        progress.setValue(10)  # Show some initial progress
        QtWidgets.QApplication.processEvents()
        
        try:
            # Create the zip file
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Get total number of files for better progress tracking
                total_files = sum([len(files) for _, _, files in os.walk(output_folder)])
                processed_files = 0
                
                # Walk through all files and subdirectories in the output folder
                for root, dirs, files in os.walk(output_folder):
                    # Convert root path to a pathlib.Path for easier manipulation
                    root_path = pathlib.Path(root)
                    
                    # Add each file to the zip
                    for file in files:
                        file_path = root_path / file
                        # Calculate the relative path for the file in the zip
                        rel_path = file_path.relative_to(output_folder)
                        # Add the file to the zip
                        zipf.write(file_path, rel_path)
                        
                        # Update progress based on files processed
                        processed_files += 1
                        progress_value = 10 + int(80 * processed_files / total_files) if total_files > 0 else 90
                        if isinstance(progress, EnhancedProgressDialog):
                            progress.update_progress(progress_value, f"Zipping: {rel_path}")
                        else:
                            progress.setValue(progress_value)
                            progress.setLabelText(f"Zipping: {rel_path}")
                            QtWidgets.QApplication.processEvents()
                        
                        if progress.wasCanceled():
                            return None
            
            # Final progress update
            if isinstance(progress, EnhancedProgressDialog):
                progress.update_progress(100, "ZIP archive completed")
            else:
                progress.setValue(100)
                progress.setLabelText("ZIP archive completed")
                QtWidgets.QApplication.processEvents()
            
            return zip_filename
            
        except Exception as e:
            # Update progress dialog instead of showing a message box
            error_message = f"Error creating ZIP: {str(e)}"
            if isinstance(progress, EnhancedProgressDialog):
                progress.update_text(error_message)
                QtWidgets.QApplication.processEvents()
                time.sleep(2)  # Give user time to see the error
            elif using_existing_progress:
                progress.setLabelText(error_message)
                QtWidgets.QApplication.processEvents()
                time.sleep(2)  # Give user time to see the error
            else:
                # Only show message box if we created our own progress dialog and it's not enhanced
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error Creating ZIP",
                    f"Failed to create ZIP archive: {str(e)}"
                )
            return None
        finally:
            # Close the progress dialog only if we created it
            if not using_existing_progress:
                progress.finish("ZIP operation completed")
    
    def fill_detectors(self, k):
        self.detectors = k
        # Add "All" option at the beginning
        self.comboBox_2.addItem("All")
        self.comboBox_2.addItems(k.keys())

    def update_detectors(self):
        """
        Sync the comboBox_2 selection to the channel lineEdit_4.
        If "All" is selected, empty the channel numbers line widget.
        """
        key = self.comboBox_2.currentText()
        if key == "All":
            # Empty the channel numbers line widget
            self.lineEdit_4.setText("")
        else:
            # Set the channel numbers from the selected detector
            s = ", ".join([str(i) for i in self.detectors[key]["chs"]])
            self.lineEdit_4.setText(s)
        self.update_parameter()

    def update_pie_windows(self):
        """
        Sync the comboBox_3 selection to the lineEdit_5 for microtime ranges.
        """
        key = self.comboBox_3.currentText()
        pie_win = self.windows[key]

        # Check if pie_win is a list/tuple or a single integer
        if isinstance(pie_win, (list, tuple)):
            # Handle list/tuple of windows
            formatted_windows = []
            # If pie_win is a list with exactly 2 elements and both are integers,
            # it's likely a single window definition [start, end]
            if len(pie_win) == 2 and all(isinstance(x, int) for x in pie_win):
                formatted_windows.append(f"{pie_win[0]}-{pie_win[1]}")
            else:
                # Otherwise, process each item in the list
                for i in pie_win:
                    if isinstance(i, (list, tuple)) and len(i) >= 2:
                        # Normal case: i is a tuple/list with at least two elements
                        formatted_windows.append(f"{i[0]}-{i[1]}")
                    elif isinstance(i, int):
                        # Handle case where i is a single integer
                        formatted_windows.append(f"{i}-{i}")
                    else:
                        # Skip invalid entries
                        continue
            s = ";".join(formatted_windows)
        elif isinstance(pie_win, int):
            # Handle case where pie_win is a single integer
            s = f"{pie_win}-{pie_win}"
        else:
            # Default to empty string for unexpected types
            s = ""

        self.lineEdit_5.setText(s)
        self.update_parameter()

    def load_available_setups(self):
        """
        Load available setups from the detector setups file and populate the comboBox.
        If no setups are available, display a message in the comboBox.
        """
        self.comboBox.clear()

        # Load setups from the detector setups file
        setups = load_detector_setups()
        setup_names = list(setups.get("setups", {}).keys())

        if not setup_names:
            # If no setups are available, add a placeholder item
            self.comboBox.addItem("No setups available")
        else:
            # Add all available setups to the comboBox
            self.comboBox.addItems(setup_names)

            # If there's a last used setup, select it
            if setups.get("last_used") and setups["last_used"] in setup_names:
                index = self.comboBox.findText(setups["last_used"])
                if index >= 0:
                    self.comboBox.setCurrentIndex(index)

    def get_burst_selection_parameters(self):
        """
        Collect all burst selection parameters from the UI.

        Returns:
            dict: A dictionary containing all burst selection parameters.
        """
        # Get the region selector values
        lb, ub = self.region_selector.getRegion()
        dT_min = 10.0 ** lb if self.pw_dT.getAxis('left').logMode else lb
        dT_max = 10.0 ** ub if self.pw_dT.getAxis('left').logMode else ub

        # Collect all parameters
        return {
            "dT_min": dT_min,
            "dT_max": dT_max,
            "use_dT_min": self.checkBox_2.isChecked(),
            "use_dT_max": self.checkBox_3.isChecked(),
            "photon_threshold": self.spinBox.value(),
            "count_rate_window_ms": self.doubleSpinBox.value(),
            "invert_count_rate_filter": self.checkBox.isChecked(),
            "filter_mode": self.used_filter,
            "use_gap_fill": self.checkBox_5.isChecked(),
            "max_gap": self.spinBox_7.value(),
            "trace_bin_width": self.doubleSpinBox_4.value(),
            "number_of_burst_bins": self.spinBox_6.value()
        }

    def save_burst_selection_parameters(self):
        """
        Save the burst selection parameters to the current setup.
        """
        setup_name = self.comboBox.currentText()
        if not setup_name or setup_name == "No setups available":
            QtWidgets.QMessageBox.warning(
                self, 
                "No Setup Selected", 
                "Please select a setup first to save burst selection parameters."
            )
            return False

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the burst selection parameters
            burst_params = self.get_burst_selection_parameters()

            # Add the burst selection parameters to the setup
            setup_data = setups["setups"][setup_name]
            setup_data["burst_selection"] = burst_params

            # Save the updated setups
            if save_detector_setups(setups, DETECTOR_SETUPS_FILE):
                QtWidgets.QMessageBox.information(
                    self, 
                    "Success", 
                    f"Burst selection parameters saved to setup '{setup_name}' successfully."
                )
                return True
            else:
                QtWidgets.QMessageBox.critical(
                    self, 
                    "Error", 
                    f"Failed to save burst selection parameters to setup '{setup_name}'."
                )
                return False
        else:
            QtWidgets.QMessageBox.warning(
                self, 
                "Invalid Setup", 
                f"The selected setup '{setup_name}' does not exist."
            )
            return False

    def update_micro_time_binning(self, setup_name=None):
        """
        Update the micro time binning (Decay bin) from the selected setup.

        Args:
            setup_name (str, optional): The name of the setup to use. If None, uses the currently selected setup.
        """
        if setup_name is None:
            setup_name = self.comboBox.currentText()

        if not setup_name or setup_name == "No setups available":
            return

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the micro time binning from the setup's tttr_reading section
            setup_data = setups["setups"][setup_name]
            if "tttr_reading" in setup_data and "micro_time_binning" in setup_data["tttr_reading"]:
                micro_time_binning = setup_data["tttr_reading"]["micro_time_binning"]
                # Update the spinBox_5 value (decay_coarse)
                self.spinBox_5.setValue(micro_time_binning)
                # Update the plots
                self.update_plots()

    def update_channel_routing(self, setup_name=None):
        """
        Update the channel routing (detectors) from the selected setup.

        Args:
            setup_name (str, optional): The name of the setup to use. If None, uses the currently selected setup.
        """
        if setup_name is None:
            setup_name = self.comboBox.currentText()

        if not setup_name or setup_name == "No setups available":
            return

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the detectors from the setup
            setup_data = setups["setups"][setup_name]
            if "detectors" in setup_data:
                # Update the detectors attribute
                self.detectors = setup_data["detectors"]

                # Clear and repopulate comboBox_2
                self.comboBox_2.blockSignals(True)
                self.comboBox_2.clear()

                # Add "All" option at the beginning
                self.comboBox_2.addItem("All")

                # Add detector names
                self.comboBox_2.addItems(self.detectors.keys())

                # Select the first detector by default (or "All" if no detectors)
                if self.comboBox_2.count() > 0:
                    self.comboBox_2.setCurrentIndex(0)

                self.comboBox_2.blockSignals(False)

                # Update the channel numbers in lineEdit_4
                self.update_detectors()

    def update_pie_windows_from_setup(self, setup_name=None):
        """
        Update the PIE windows from the selected setup.

        Args:
            setup_name (str, optional): The name of the setup to use. If None, uses the currently selected setup.
        """
        if setup_name is None:
            setup_name = self.comboBox.currentText()

        if not setup_name or setup_name == "No setups available":
            return

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the windows from the setup
            setup_data = setups["setups"][setup_name]
            if "windows" in setup_data:
                # Update the windows attribute
                self.windows = setup_data["windows"]

                # Clear and repopulate comboBox_3
                self.comboBox_3.blockSignals(True)
                self.comboBox_3.clear()

                # Add window names
                self.comboBox_3.addItems(self.windows.keys())

                # Select the first window by default
                if self.comboBox_3.count() > 0:
                    self.comboBox_3.setCurrentIndex(0)

                self.comboBox_3.blockSignals(False)

                # Update the microtime ranges in lineEdit_5
                self.update_pie_windows()

    def update_burst_selection_parameters(self, setup_name=None):
        """
        Update the burst selection parameters from the selected setup.

        Args:
            setup_name (str, optional): The name of the setup to use. If None, uses the currently selected setup.
        """
        if setup_name is None:
            setup_name = self.comboBox.currentText()

        if not setup_name or setup_name == "No setups available":
            return

        # Load setups from the detector setups file
        setups = load_detector_setups()

        # Check if the selected setup exists
        if setup_name in setups.get("setups", {}):
            # Get the burst selection parameters from the setup
            setup_data = setups["setups"][setup_name]
            if "burst_selection" in setup_data:
                burst_params = setup_data["burst_selection"]

                # Update the UI with the burst selection parameters
                if "dT_min" in burst_params and "dT_max" in burst_params:
                    dT_min = burst_params["dT_min"]
                    dT_max = burst_params["dT_max"]
                    if self.pw_dT.getAxis('left').logMode:
                        dT_min = np.log10(dT_min) if dT_min > 0 else -4  # Default to -4 if dT_min is 0 or negative
                        dT_max = np.log10(dT_max) if dT_max > 0 else 0   # Default to 0 if dT_max is 0 or negative
                    self.region_selector.setRegion((dT_min, dT_max))
                    self._dT_min = burst_params["dT_min"]
                    self._dT_max = burst_params["dT_max"]
                    self.doubleSpinBox_2.setValue(burst_params["dT_min"])
                    self.doubleSpinBox_3.setValue(burst_params["dT_max"])

                if "use_dT_min" in burst_params:
                    self.checkBox_2.setChecked(burst_params["use_dT_min"])

                if "use_dT_max" in burst_params:
                    self.checkBox_3.setChecked(burst_params["use_dT_max"])

                if "photon_threshold" in burst_params:
                    self.spinBox.setValue(burst_params["photon_threshold"])

                if "count_rate_window_ms" in burst_params:
                    self.doubleSpinBox.setValue(burst_params["count_rate_window_ms"])

                if "invert_count_rate_filter" in burst_params:
                    self.checkBox.setChecked(burst_params["invert_count_rate_filter"])

                if "filter_mode" in burst_params:
                    if burst_params["filter_mode"] == "count_rate":
                        self.radioButton.setChecked(True)
                    else:
                        self.radioButton_2.setChecked(True)

                if "use_gap_fill" in burst_params:
                    self.checkBox_5.setChecked(burst_params["use_gap_fill"])

                if "max_gap" in burst_params:
                    self.spinBox_7.setValue(burst_params["max_gap"])

                if "trace_bin_width" in burst_params:
                    self.doubleSpinBox_4.setValue(burst_params["trace_bin_width"])

                if "number_of_burst_bins" in burst_params:
                    self.spinBox_6.setValue(burst_params["number_of_burst_bins"])

                # Update the plots
                self.update_plots()

    @chisurf.gui.decorators.init_with_ui("tttr_photon_filter.ui")
    def __init__(
            self,
            *args,
            windows,
            detectors,
            show_dT: bool = True,
            show_filter: bool = True,
            show_mcs: bool = True,
            show_decay: bool = True,
            show_burst: bool = True,
            default_mcs_dT: float = 1.0,
            default_dT_min: float = 0.0001,
            default_dT_max: float = 0.15,
            use_dT_min: bool = False,
            use_dT_max: bool = True,
            default_photon_threshold: int = 60,
            default_count_rate_window_ms: float = 1.0,
            invert_count_rate_filter: bool = True,
            default_filter_mode: str = 'burst',
            use_gap_fill: bool = False,
            default_max_gap: int = 3,
            **kwargs
    ):
        """
        Initialize the photon-filter wizard.

        Parameters
        ----------
        windows : dict
            Dictionary of predefined PIE windows. Passed to the relevant UI comboBox.
        detectors : dict
            Dictionary of predefined detectors. Passed to the relevant UI comboBox.
        show_dT : bool, default=True
            Whether to show the dT plot (between consecutive photons).
        show_filter : bool, default=True
            Whether to show the filter/selection plot.
        show_mcs : bool, default=True
            Whether to show the intensity trace (MCS) plot.
        show_decay : bool, default=True
            Whether to show the microtime histogram (decay plot).
        show_burst : bool, default=True
            Whether to show the burst histogram.
        default_dT_min : float, default=0.0001
            Initial lower bound for delta macro-time filter (in ms).
        default_dT_max : float, default=0.15
            Initial upper bound for delta macro-time filter (in ms).
        default_mcs_dT : float, default=1.0
            Default / initial bin width value of intensity trace (in ms).
        use_dT_min : bool, default=False
            Whether the lower bound of delta macro-time filter is active initially.
        use_dT_max : bool, default=True
            Whether the upper bound of delta macro-time filter is active initially.
        default_photon_threshold : int, default=60
            Initial threshold for count rate or burst search.
        default_count_rate_window_ms : float, default=1.0
            Initial time window (ms) for count-rate based filtering.
        invert_count_rate_filter : bool, default=True
            Whether to invert the count-rate filter initially.
        default_filter_mode : str, default='burst'
            Which filter mode radio button is selected by default.
            Valid: 'count_rate' or 'burst'.
        use_gap_fill : bool, default=False
            Whether gap-filling is enabled by default.
        default_max_gap : int, default=3
            Maximum number of consecutive unselected photons that can be 'filled'
            when gap-filling is applied.

        Returns
        -------
        None. Initializes the UI and sets default widget states.
        """
        self.setTitle("Photon filter")
        self.windows = windows
        self.detectors = detectors
        self.fill_detectors(detectors)
        self.fill_pie_windows(windows)

        # Dictionary to store all TTTR objects
        self.tttr_objects = dict()

        # Load available setups for comboBox
        self.load_available_setups()

        # Main settings
        self.settings: dict = {}
        self.settings['tttr_filenames'] = []
        self.settings['count_rate_filter'] = {}
        self.settings['delta_macro_time_filter'] = {}
        self.filter_data_saved = False

        def after_file_drop():
            """
            Callback to load *all* TTTR files after they're dropped or specified.
            Ensures that files with restricted extensions require explicit file type selection.
            If a folder is dropped, all files with supported extensions (found directly in that folder)
            are added.
            Uses tttrlib.inferTTTRFileType to automatically detect file types when possible.
            """
            # Generate allowed extensions dynamically from tttrlib
            allowed_extensions = {
                f".{ext.lower()}" if not ext.startswith('.') else ext.lower()
                for ext in tttrlib.TTTR.get_supported_container_names()
            }

            # Expand directories: if an entry in tttr_filenames is a folder,
            # replace it with all files (in that folder only) with allowed extensions.
            expanded_files = []
            for path_str in self.settings['tttr_filenames']:
                p = pathlib.Path(path_str).resolve()
                if p.is_dir():
                    for child in p.iterdir():
                        if child.is_file() and child.suffix.lower() in allowed_extensions:
                            expanded_files.append(str(child.resolve()))
                else:
                    expanded_files.append(str(p))
            self.settings['tttr_filenames'] = expanded_files

            # List of restricted extensions requiring manual selection (if needed)
            RESTRICTED_EXTENSIONS = [".spc"]  # Extend or modify as required

            requires_filetype_selection = False
            restricted_files = []

            for fn in self.settings['tttr_filenames']:
                p = pathlib.Path(fn).resolve()
                file_extension = p.suffix.lower()

                # Check if the file extension requires explicit selection
                if file_extension in RESTRICTED_EXTENSIONS:
                    if self.filetype == "Auto":  # Only warn if no file type is preselected
                        requires_filetype_selection = True
                        restricted_files.append(p.name)

            if requires_filetype_selection:
                QtWidgets.QMessageBox.warning(
                    self, "File Type Required",
                    "The following files require an explicit file type selection before loading:\n\n"
                    + "\n".join(restricted_files)
                    + "\n\nPlease select the correct file type from the dropdown menu."
                )
                self.onClearFiles()
                return  # Prevent loading any files

            # Get file type from the selected setup
            file_type = self.filetype
            # If no setup is selected or the setup doesn't have a file type,
            # we've already shown a warning in the filetype property

            # Proceed with loading the files and showing progress
            total_files = len(self.settings['tttr_filenames'])
            progress_window = ProgressWindow(title="Loading Files", message="Processing files...",
                                             max_value=total_files, parent=self)
            progress_window.show()

            for i, fn in enumerate(self.settings['tttr_filenames'], start=1):
                p = pathlib.Path(fn).resolve()
                p_str = str(p)

                if p_str not in self.tttr_objects:
                    if p.exists() and p.is_file():
                        file_type = self.filetype
                        try:
                            if isinstance(file_type, str):
                                self.tttr_objects[p_str] = tttrlib.TTTR(p_str, file_type)
                            elif p.suffix.lower() not in RESTRICTED_EXTENSIONS:
                                # Use inferTTTRFileType for better auto-detection
                                file_type_int = tttrlib.inferTTTRFileType(p_str)
                                if file_type_int is not None and file_type_int >= 0:
                                    self.tttr_objects[p_str] = tttrlib.TTTR(p_str, file_type_int)
                                else:
                                    # Fall back to default auto-detection if inference fails
                                    self.tttr_objects[p_str] = tttrlib.TTTR(p_str)
                        except Exception as e:
                            progress_window.close()
                            QtWidgets.QMessageBox.critical(
                                self, 
                                "Error Loading File", 
                                f"Failed to load file '{p.name}' with the selected setup.\n\n"
                                f"Error: {str(e)}\n\n"
                                f"Please check that you have selected the correct setup for this file type."
                            )
                            self.onClearFiles()
                            return  # Exit early to prevent undefined state

                progress_window.set_value(i)

            progress_window.close()

            n_files = len(self.settings['tttr_filenames'])
            self.spinBox_4.setMaximum(n_files - 1)
            if n_files > 0:
                self.spinBox_4.setValue(n_files - 1)
            self.read_tttr()

        # Inject file-drop logic
        self.textEdit.setVisible(False)
        chisurf.gui.decorators.lineEdit_dragFile_injector(
            self.lineEdit, call=after_file_drop, target=self.settings['tttr_filenames']
        )

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.setSizePolicy(sizePolicy)

        # Internal TTTR reference
        self.tttr = None

        # Initialize dT_min/dT_max
        self._dT_min = default_dT_min
        self._dT_max = default_dT_max

        # Create & configure plots
        self.pw_burst_histogram = pg.PlotWidget(parent=self, title="Burst histogram")
        self.plot_burst_histogram = self.pw_burst_histogram.getPlotItem()
        self.pw_burst_histogram.setLabel('left', 'Counts')
        self.pw_burst_histogram.setLabel('bottom', 'Burst size (Nbr. Photons)')
        self.pw_burst_histogram.resize(100, 80)

        color_all = QtGui.QColor(255, 255, 0, 64)
        color_selected = QtGui.QColor(0, 255, 255, 255)
        pen2 = pg.mkPen(color_all, width=1, style=QtCore.Qt.SolidLine)
        pen1 = pg.mkPen(color_selected, width=1, style=QtCore.Qt.SolidLine)

        self.pw_dT = pg.PlotWidget(parent=self, title="Delta macro-time")
        self.pw_dT.setLabel('left', 'dT (ms)')
        self.pw_dT.setLabel('bottom', 'Photon Index')
        self.plot_item_dt = self.pw_dT.getPlotItem()
        self.plot_unselected = self.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen2)
        self.plot_selected = self.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen1)
        self.pw_dT.resize(200, 40)

        self.pw_mcs = pg.PlotWidget(parent=self, title="Count rate display")
        self.pw_mcs.setLabel('left', 'Intensity (kHz)')
        self.pw_mcs.setLabel('bottom', 'Time (s)')
        self.plot_item_mcs = self.pw_mcs.getPlotItem()
        self.plot_mcs_all = self.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen2)
        self.plot_mcs_selected = self.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen1)
        self.pw_mcs.resize(200, 80)

        self.pw_decay = pg.PlotWidget(parent=self, title="Microtime histogram")
        self.pw_decay.setLabel('left', 'Counts')
        self.pw_decay.setLabel('bottom', 'Microtime (ns)')
        self.plot_item_decay = self.pw_decay.getPlotItem()
        self.plot_decay_all = self.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen2)
        self.plot_decay_selected = self.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen1)
        self.pw_decay.resize(200, 80)

        self.pw_filter = pg.PlotWidget(parent=self, title="Filter/selection")
        self.pw_filter.setLabel('left', 'Selected (1) / Unselected (0)')
        self.pw_filter.setLabel('bottom', 'Photon Index')
        self.pw_filter.setXLink(self.pw_dT)
        self.pw_dT.setMouseEnabled(x=False, y=False)
        self.pw_filter.setMouseEnabled(x=False, y=False)
        self.plot_item_sel = self.pw_filter.getPlotItem()
        self.plot_select = self.plot_item_sel.plot(x=[1.0], y=[1.0])
        self.pw_filter.resize(200, 20)

        self.plot_item_dt.setLogMode(False, True)
        self.plot_item_decay.setLogMode(False, True)
        self.plot_item_sel.setLogMode(False, False)

        ca = list(matplotlib.colors.hex2color(colors["region_selector"]))
        co = [ca[0] * 255, ca[1] * 255, ca[2] * 255, colors["region_selector_alpha"]]
        self.region_selector = pg.LinearRegionItem(
            brush=co, orientation='horizontal',
            values=(np.log10(self._dT_min), np.log10(self._dT_max))
        )
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

        # Place plots in layout
        self.gridLayout_6.addWidget(self.pw_dT, 0, 0, 1, 3)
        self.gridLayout_6.addWidget(self.pw_filter, 1, 0, 1, 3)
        self.gridLayout_6.addWidget(self.pw_mcs, 2, 0, 1, 1)
        self.gridLayout_6.addWidget(self.pw_decay, 2, 1, 1, 1)
        self.gridLayout_6.addWidget(self.pw_burst_histogram, 0, 1, 2, 1)

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
        self.comboBox.currentTextChanged.connect(self.update_micro_time_binning)
        self.comboBox.currentTextChanged.connect(self.update_burst_selection_parameters)
        self.comboBox.currentTextChanged.connect(self.update_channel_routing)
        self.comboBox.currentTextChanged.connect(self.update_pie_windows_from_setup)

        # Custom validator
        validator = CommaSeparatedIntegersValidator()
        self.lineEdit_4.setValidator(validator)

        # Initialize defaults
        self.spinBox.setValue(default_photon_threshold)
        self.doubleSpinBox.setValue(default_count_rate_window_ms)
        self.checkBox.setChecked(invert_count_rate_filter)
        self.checkBox_2.setChecked(use_dT_min)
        self.checkBox_3.setChecked(use_dT_max)
        self.doubleSpinBox_4.setValue(default_mcs_dT)

        # -- NEW: set default gap-fill checkbox/spinbox --
        self.checkBox_5.setChecked(use_gap_fill)  # <--- gap-fill checkbox
        self.spinBox_7.setValue(default_max_gap)  # <--- max-gap spinbox

        # Control initial plot visibility
        self.pw_dT.setVisible(show_dT)
        self.toolButton_4.setChecked(show_filter)
        self.toolButton_2.setChecked(show_mcs)
        self.toolButton_3.setChecked(show_decay)
        self.toolButton_7.setChecked(show_burst)

        # -- Set the default filter mode (radio buttons) --
        # radioButton = count_rate; radioButton_2 = burst
        if default_filter_mode == 'burst':
            self.radioButton_2.setChecked(True)
        else:
            # Fallback or default to 'count_rate'
            self.radioButton.setChecked(True)

        # Add a button to save burst selection parameters
        self.save_burst_params_button = QtWidgets.QPushButton("Save Parameters", self)
        self.save_burst_params_button.clicked.connect(self.save_burst_selection_parameters)
        self.save_burst_params_button.setToolTip("Save burst parameters as default to setup.")
        self.gridLayout_3.addWidget(self.save_burst_params_button, 5, 0, 1, 2)

        # Update micro time binning and burst selection parameters from the selected setup
        self.update_micro_time_binning()
        self.update_burst_selection_parameters()

        # Final initial update
        self.update_parameter()
