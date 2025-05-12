import os
import pathlib
import typing

import tttrlib
import json
import time
import numpy as np

import pyqtgraph as pg
import matplotlib

import chisurf.fio as io
import chisurf.fio.fluorescence
import chisurf.math
import chisurf.gui.decorators
from chisurf.gui import QtGui, QtWidgets, QtCore, uic
from chisurf.math.signal import fill_small_gaps_in_array

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
        txt = self.comboBox.currentText()
        if txt == 'Auto':
            # Try to infer file type from the current file
            current_file = self.current_tttr_filename
            if current_file and pathlib.Path(current_file).exists():
                file_type_int = tttrlib.inferTTTRFileType(current_file)
                # Update comboBox if a file type is recognized
                if file_type_int is not None and file_type_int >= 0:
                    container_names = tttrlib.TTTR.get_supported_container_names()
                    if 0 <= file_type_int < len(container_names):
                        # Return the inferred file type name
                        return container_names[file_type_int]
            # If we can't infer, return None to let tttrlib try auto-detection
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

    def update_filter_plot(self):
        """
        Update the plot showing which photons are selected (1) vs unselected (0).
        """
        n_min = self.plot_min
        n_max = self.plot_max
        x = np.arange(n_min, n_max)
        y = self.selected[n_min:n_max]
        self.plot_select.setData(x=x, y=y)

    def update_dt_plot(self):
        """
        Update the plot of dT vs photon index, highlighting selected vs unselected points.
        """
        if isinstance(self.dT, np.ndarray):
            dT = self.dT
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
            tttr = self.tttr
            if isinstance(tttr, tttrlib.TTTR):
                idx = np.where(self.selected)[0]
                tw = self.trace_bin_width / 1000.0
                trace_selected = tttr[idx].get_intensity_trace(time_window_length=tw)
                x1 = np.arange(len(trace_selected)) * tw
                self.plot_mcs_selected.setData(x1, trace_selected)

                trace_all = tttr.get_intensity_trace(time_window_length=tw)
                x2 = np.arange(len(trace_all)) * tw
                self.plot_mcs_all.setData(x2, trace_all)
        else:
            self.plot_mcs_all.setData(x=[1.0], y=[1.0])
            self.plot_mcs_selected.setData(x=[1.0], y=[1.0])

    def update_burst_histogram(self):
        """
        Update the histogram of burst lengths (only relevant if burst mode is active).
        """
        if self.do_burst_photon_plot:
            burst_lengths = self.burst_lengths
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

    def update_decay_plot(self):
        """
        Update the microtime decay plot (histogram), comparing all photons vs selected photons.
        """
        if self.do_microtime_plot:
            if isinstance(self.tttr, tttrlib.TTTR):
                idx = np.where(self.selected)[0]
                y, x = self.tttr[idx].get_microtime_histogram(self.decay_coarse)
                idx_max = np.where(y > 0)[0][-1] if np.any(y > 0) else 1
                x = x[:idx_max]
                y = y[:idx_max]
                x *= 1e9
                self.plot_decay_selected.setData(x=x, y=y)

                y, x = self.tttr.get_microtime_histogram(self.decay_coarse)
                idx_max = np.where(y > 0)[0][-1] if np.any(y > 0) else 1
                x = x[:idx_max]
                y = y[:idx_max]
                x *= 1e9
                self.plot_decay_all.setData(x=x, y=y)
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
            self.tttr = self.tttr_objects[p_str]
            self.plot_max = len(self.tttr) - 1
            header = self.tttr.get_header()
            s = header.json
            d = json.loads(s)
            self.settings['header'] = d
            self.update_plots()
        else:
            # Possibly fallback logic or a message
            chisurf.logging.log(1, f"Path not in dictionary: {p_str}")

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

    def save_selection(self):
        """
        Save the selection data in .bur or .json.gz, depending on user checkboxes,
        and display a progress bar while saving.
        """
        total_files = len(self.settings['tttr_filenames'])
        total_tasks = 0
        if self.save_bur:
            total_tasks += total_files
        if self.save_sl5:
            total_tasks += total_files

        progress = QtWidgets.QProgressDialog("Saving selection...", "Cancel", 0, total_tasks, self)
        progress.setWindowTitle("Saving selection")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()

        current_task = 0

        if self.save_bur:
            for t, filename in zip(self.parent_directories, self.settings['tttr_filenames']):
                fn = pathlib.Path(filename)
                resolved_path = str(fn.resolve())
                self.tttr = self.tttr_objects[resolved_path]

                base_name = fn.stem
                bur_directory = t / 'bi4_bur'
                bur_directory.mkdir(exist_ok=True, parents=True)
                bur_filename = bur_directory / f"{base_name}.bur"

                start_stop = self.burst_start_stop
                io.fluorescence.burst.write_bur_file(
                    bur_filename,
                    start_stop=start_stop,
                    tttr=self.tttr,
                    filename=fn,
                    windows=self.windows,
                    detectors=self.detectors
                )
                mt = self.tttr.macro_times[-1] * self.tttr.header.macro_time_resolution
                io.fluorescence.burst.write_mti_summary(
                    filename=fn,
                    analysis_dir=t,
                    max_macro_time=mt,
                    append=True
                )
                current_task += 1
                progress.setValue(current_task)
                QtWidgets.QApplication.processEvents()
                if progress.wasCanceled():
                    break

        if self.save_sl5:
            for t, filename in zip(self.parent_directories, self.settings['tttr_filenames']):
                parent_directory = t / 'sl5'
                parent_directory.mkdir(exist_ok=True, parents=True)
                parent_directory = parent_directory.absolute()
                fn = pathlib.Path(filename)
                resolved_path = str(fn.resolve())
                self.tttr = self.tttr_objects[resolved_path]

                d = {
                    'filename': os.path.relpath(fn, t),
                    'filetype': self.filetype,
                    'count_rate_filter': self.settings['count_rate_filter'],
                    'delta_macro_time_filter': self.settings['delta_macro_time_filter'],
                    'filter': chisurf.fio.compress_numpy_array(self.selected)
                }
                base_name = fn.stem
                output_filename = parent_directory / f"{base_name}.json.gz"
                with io.open_maybe_zipped(output_filename, "w") as outfile:
                    packed = json.dumps(d)
                    outfile.write(packed)
                current_task += 1
                progress.setValue(current_task)
                QtWidgets.QApplication.processEvents()
                if progress.wasCanceled():
                    break

        progress.close()
        self.filter_data_saved = True

    def fill_pie_windows(self, k):
        self.windows = k
        self.comboBox_3.addItems(k.keys())

    def fill_detectors(self, k):
        self.detectors = k
        self.comboBox_2.addItems(k.keys())

    def update_detectors(self):
        """
        Sync the comboBox_2 selection to the channel lineEdit_4.
        """
        key = self.comboBox_2.currentText()
        s = ", ".join([str(i) for i in self.detectors[key]["chs"]])
        self.lineEdit_4.setText(s)
        self.update_parameter()

    def update_pie_windows(self):
        """
        Sync the comboBox_3 selection to the lineEdit_5 for microtime ranges.
        """
        key = self.comboBox_3.currentText()
        pie_win = self.windows[key]
        s = ";".join([f"{i[0]}-{i[1]}" for i in pie_win])
        self.lineEdit_5.setText(s)
        self.update_parameter()

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

        # Filetype combos
        self.comboBox.clear()
        self.comboBox.insertItem(0, "Auto")
        self.comboBox.insertItems(1, list(tttrlib.TTTR.get_supported_container_names()))

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

            # Try to infer file type from the first file if set to Auto
            if self.filetype == "Auto" and self.settings['tttr_filenames']:
                first_file = self.settings['tttr_filenames'][0]
                file_type_int = tttrlib.inferTTTRFileType(first_file)

                # Update comboBox if a file type is recognized
                if file_type_int is not None and file_type_int >= 0:
                    container_names = tttrlib.TTTR.get_supported_container_names()
                    if 0 <= file_type_int < len(container_names):
                        # Add 1 to account for 'Auto' at index 0
                        idx = file_type_int + 1
                        if 0 <= idx < self.comboBox.count():
                            self.comboBox.setCurrentIndex(idx)

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
                        if isinstance(self.filetype, str) and self.filetype != "Auto":
                            self.tttr_objects[p_str] = tttrlib.TTTR(p_str, self.filetype)
                        elif p.suffix.lower() not in RESTRICTED_EXTENSIONS:
                            # Use inferTTTRFileType for better auto-detection
                            file_type = tttrlib.inferTTTRFileType(p_str)
                            if file_type is not None and file_type >= 0:
                                self.tttr_objects[p_str] = tttrlib.TTTR(p_str, file_type)
                            else:
                                # Fall back to default auto-detection if inference fails
                                self.tttr_objects[p_str] = tttrlib.TTTR(p_str)

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

        # Final initial update
        self.update_parameter()
