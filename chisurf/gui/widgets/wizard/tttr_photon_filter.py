import os
import pathlib
import typing

import tttrlib
import json
import time
import numpy as np

import matplotlib
import pyqtgraph as pg

from typing import TypedDict

import chisurf.fio
import chisurf.gui.decorators
from chisurf.gui import QtGui, QtWidgets, QtCore, uic

colors = chisurf.settings.gui['plot']['colors']


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


class WizardTTTRPhotonFilter(QtWidgets.QWizardPage):

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
        return self.comboBox.currentText()

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

        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time
        chisurf.logging.log(0, "Elapsed time: %s" % elapsed_time)

        return s.astype(dtype=np.uint8)

    def update_filter_plot(self):
        n_min = self.plot_min
        n_max = self.plot_max

        x = np.arange(n_min, n_max)
        y = self.selected[n_min:n_max]

        self.plot_select.setData(x=x, y=y)

    def update_dt_plot(self):
        if isinstance(self.dT, np.ndarray):
            dT = self.dT
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

    def update_decay_plot(self):
        if self.toolButton_3.isChecked():
            if isinstance(self.tttr, tttrlib.TTTR):
                idx = np.where(self.selected)[0]
                y, x = self.tttr[idx].get_microtime_histogram(self.decay_coarse)
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

    @property
    def plot_selection(self):

        return "all"

    def update_plots(self, selection: str = "all"):
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

    def read_tttr(self):
        fn = self.current_tttr_filename
        if fn:
            if pathlib.Path(fn).exists():
                n = len(self.settings['tttr_filenames'])
                self.spinBox_4.setMaximum(n - 1)
                self.comboBox.setEnabled(False)
                self.tttr = tttrlib.TTTR(fn, self.filetype)
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
        s = f'{chs}_{self.dT_max:.3f}#{self.photon_number_threshold}'
        self.lineEdit_2.setText(s)

    def update_parameter(self):
        lb, ub = self.region_selector.getRegion()
        self.settings['count_rate_filter_active'] = self.checkBox_4.isChecked()
        self.settings['count_rate_filter']['n_ph_max'] = int(self.spinBox.value())
        self.settings['count_rate_filter']['time_window'] = float(self.doubleSpinBox.value()) * 1e-3
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

    def save_filter_data(self):
        for t, filename in zip(self.parent_directories, self.settings['tttr_filenames']):
            parent_directory = t / 'sl5'
            parent_directory.mkdir(exist_ok=True, parents=True)
            parent_directory = parent_directory.absolute()
            fn: pathlib.Path = pathlib.Path(filename).absolute()

            d = {
                'filename': os.path.relpath(fn, t),
                'filetype': self.filetype,
                'count_rate_filter': self.settings['count_rate_filter'],
                'delta_macro_time_filter': self.settings['delta_macro_time_filter'],
                'filter': chisurf.fio.compress_numpy_array(self.selected)
            }

            # Use basename to get the base name of the file
            base_name = fn.stem

            # Get the parent directory of the original file

            # Modify the output file name to use the base name in the same folder
            output_filename = parent_directory / f"{base_name}.json.gz"

            with chisurf.fio.open_maybe_zipped(output_filename, "w") as outfile:
                packed = json.dumps(d)
                outfile.write(packed)
        self.filter_data_saved = True

    @chisurf.gui.decorators.init_with_ui("tttr_photon_filter.ui")
    def __init__(self, *args, **kwargs):
        self.setTitle("Photon filter")

        self.settings: dict = dict()
        tttr_filenames: typing.List[pathlib.Path] = list()
        self.settings['tttr_filenames'] = tttr_filenames
        self.settings['count_rate_filter']: CountRateFilterSettings = dict()
        self.settings['delta_macro_time_filter']: DeltaMacroTimeFilterSettings = dict()
        self.filter_data_saved = False

        self.textEdit.setVisible(False)
        chisurf.gui.decorators.lineEdit_dragFile_injector(
            self.lineEdit,
            call=self.read_tttr,
            target=self.settings['tttr_filenames']
        )
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sizePolicy)

        self.tttr = None
        self._dT_min = 0.0001
        self._dT_max = 0.15

        # Create plots
        self.pw_dT = pg.plot()
        self.pw_filter = pg.plot()
        self.pw_mcs = pg.plot()
        self.pw_decay = pg.plot()

        color_all = QtGui.QColor(255, 255, 0, 64)
        color_selected = QtGui.QColor(0, 255, 255, 255)
        pen2 = pg.mkPen(color_all, width=1, style=QtCore.Qt.SolidLine)
        pen1 = pg.mkPen(color_selected, width=1, style=QtCore.Qt.SolidLine)

        self.plot_item_dt = self.pw_dT.getPlotItem()
        self.plot_unselected = self.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen2)
        self.plot_selected = self.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen1)

        self.plot_item_mcs = self.pw_mcs.getPlotItem()
        self.plot_mcs_all = self.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen2)
        self.plot_mcs_selected = self.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen1)

        self.plot_item_decay = self.pw_decay.getPlotItem()
        self.plot_decay_all = self.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen2)
        self.plot_decay_selected = self.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen1)

        self.plot_item_sel = self.pw_filter.getPlotItem()
        self.plot_select = self.plot_item_sel.plot(x=[1.0], y=[1.0])

        self.pw_filter.setXLink(self.pw_dT)
        self.pw_dT.setMouseEnabled(x=False, y=False)
        self.pw_filter.setMouseEnabled(x=False, y=False)
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

        self.pw_dT.resize(200, 40)
        self.pw_filter.resize(200, 20)

        self.pw_mcs.resize(200, 80)
        self.pw_decay.resize(200, 80)

        self.gridLayout_6.addWidget(self.pw_dT, 0, 0, 1, 2)  # (widget, row, column, rowSpan, columnSpan)
        self.gridLayout_6.addWidget(self.pw_filter, 1, 0, 1, 2)  # (widget, row, column, rowSpan, columnSpan)
        self.gridLayout_6.addWidget(self.pw_mcs, 2, 0, 1, 1)  # (widget, row, column, rowSpan, columnSpan)
        self.gridLayout_6.addWidget(self.pw_decay, 2, 1, 1, 1)  # (widget, row, column, rowSpan, columnSpan)

        # Connect actions
        self.actionUpdate_Values.triggered.connect(self.update_parameter)
        self.actionUpdateUI.triggered.connect(self.updateUI)
        self.actionFile_changed.triggered.connect(self.read_tttr)
        self.actionRegionUpdate.triggered.connect(self.onRegionUpdate)

        self.toolButton_2.toggled.connect(self.pw_mcs.setVisible)
        self.toolButton_3.toggled.connect(self.pw_decay.setVisible)
        self.toolButton_4.toggled.connect(self.pw_filter.setVisible)
        self.toolButton_5.clicked.connect(self.save_filter_data)
        self.toolButton_6.clicked.connect(self.onClearFiles)

        self.update_parameter()

