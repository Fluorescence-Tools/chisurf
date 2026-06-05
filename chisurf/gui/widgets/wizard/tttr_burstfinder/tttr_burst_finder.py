import os
import pathlib
import typing

import tttrlib
import json
import time
import numpy as np

import pyqtgraph as pg
import pandas as pd

from collections import OrderedDict

import chisurf.core.fio
import chisurf.gui.decorators
from chisurf.gui import QtGui, QtWidgets, QtCore, uic

from .tttr_burst_finder_utils import (
    get_indices_in_ranges,
    create_mti_summary,
    create_bur_summary,
    CommaSeparatedIntegersValidator,
    fill_small_gaps_in_array,
    find_bursts,
    CountRateFilterSettings,
    DeltaMacroTimeFilterSettings,
    PhotonFilterSettings,
)
from .tttr_burst_finder_ui import setup_ui as _setup_ui

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
            text = str(s).strip()
            if not text:
                return None

            # Allow both ';' and ',' as range separators.
            segments = []
            for item in text.replace(',', ';').split(';'):
                item = item.strip()
                if item:
                    segments.append(item)

            if not segments:
                chisurf.logging.log(0, "::microtime_ranges: No usable ranges after parsing.")
                return None

            ranges: typing.List[typing.Tuple[int, int]] = []
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue

                # Support ":" or "-" as min-max separator while allowing negative values.
                if ':' in seg:
                    a_txt, b_txt = seg.split(':', 1)
                else:
                    pos = seg.rfind('-')
                    if pos <= 0:
                        a_txt = seg
                        b_txt = seg
                    else:
                        a_txt = seg[:pos]
                        b_txt = seg[pos + 1 :]

                a = int(a_txt.strip())
                b = int(b_txt.strip())
                if a <= b:
                    ranges.append((a, b))
                else:
                    ranges.append((b, a))

            return ranges if ranges else None

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

        if self.settings.get('filter_active', True):
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
        self.settings['filter_active'] = self.checkBox_4.isChecked()
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
                'filter': chisurf.core.fio.compress_numpy_array(self.selected)
            }

            # Use basename to get the base name of the file
            base_name = fn.stem

            # Modify the output file name to use the base name in the same folder
            output_filename = parent_directory / f"{base_name}.json.gz"

            with chisurf.core.fio.open_maybe_zipped(output_filename, "w") as outfile:
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
        _setup_ui(
            self,
            windows=windows,
            detectors=detectors,
            callback_function=callback_function,
            show_dT=show_dT,
            show_burst_histogram=show_burst_histogram,
            show_mcs=show_mcs,
            show_decay=show_decay,
            show_filter=show_filter,
            initial_trace_bin_width=initial_trace_bin_width,
            initial_photon_threshold=initial_photon_threshold,
            initial_tw_size=initial_tw_size,
            initial_max_gap=initial_max_gap,
            initial_decay_coarse=initial_decay_coarse,
            initial_number_of_burst_bins=initial_number_of_burst_bins,
            initial_dT_min=initial_dT_min,
            initial_dT_max=initial_dT_max,
        )
