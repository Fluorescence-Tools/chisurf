import os
import pathlib
import typing

import json
import numpy as np

import pyqtgraph as pg

import chisurf.fio
import chisurf.data
import chisurf.fluorescence.fcs
import chisurf.gui.decorators
from chisurf.gui import QtGui, QtWidgets, QtCore, uic

colors = chisurf.settings.gui['plot']['colors']


class WizardFcsMerger(QtWidgets.QWizardPage):

    @property
    def correlation_folder(self):
        return pathlib.Path(self.lineEdit.text())

    @property
    def current_curve_idx(self) -> int:
        table = self.tableWidget
        idx = int(table.currentIndex().row())
        return idx

    @staticmethod
    def compute_average_correlations(correlations: typing.List[dict]) -> dict:
        taus = []
        cors = []
        ws = []
        acquisition_time = 0.0
        count_rate = 0.0
        for correlation in correlations:
            tau = np.array(correlation['x'])
            cor = np.array(correlation['y'])
            acquisition_time += correlation['duration']
            counts = correlation['channel_a']['counts'] + correlation['channel_b']['counts']
            count_rate += counts / acquisition_time
            taus.append(tau)
            cors.append(cor)
            w = chisurf.fluorescence.fcs.noise(tau, cor, acquisition_time, count_rate, weight_type='suren')
            ws.append(w)
        ys = np.array(cors)
        n_curves = len(correlations)
        ey = np.std(ys, axis=0) / np.sqrt(n_curves)
        correlation = {
            'x': np.array(taus).mean(axis=0)[1:],
            'y': ys.mean(axis=0)[1:],
            'ey': ey[1:], #np.array(ws).mean(axis=0)[1:],
            'duration': acquisition_time,
            'count_rate': count_rate
        }
        return correlation

    @property
    def mean_correlation(self) -> dict:
        data = self.compute_average_correlations(self.correlations)
        return data

    @property
    def merge_folder(self):
        return pathlib.Path('..')

    def update_plots(self, *args, **kwargs):
        chisurf.logging.log(0, 'WizardTTTRCorrelator::Updating plots')
        self.pw_fcs.clear()
        idx = self.current_curve_idx
        for i, cor in enumerate(self.correlations):
            if i == idx:
                width = 3.0
            else:
                width = 1.0
            pen = pg.mkPen(chisurf.settings.colors[i % len(chisurf.settings.colors)]['hex'], width=width)
            self.plot_item_fcs.plot(x=cor['x'], y=cor['y'], pen=pen)

        self.pw_fcs_mean.clear()
        corr_mean = self.mean_correlation
        self.plot_item_fcs_mean.plot(x=corr_mean['x'], y=corr_mean['y'])

    def onClearFiles(self):
        print( "WizardTTTRCorrelator::onClearFiles")
        self.settings['tttr_filenames'].clear()
        self.comboBox.setEnabled(True)
        self.lineEdit.clear()

    def writeTableValues(self, table: QtWidgets.QTableWidget, arr):
        table.setRowCount(0)
        for row in arr:
            rc = table.rowCount()
            table.insertRow(rc)
            for i, col in enumerate(row):
                table.setItem(rc, i, QtWidgets.QTableWidgetItem(f"{col: 0.2f}"))
                table.resizeRowsToContents()
            table.resizeRowsToContents()

    def append_correlation(self, filename: pathlib.Path, correlation_dict):
        self.correlations.append(correlation_dict)
        table = self.tableWidget
        rc = table.rowCount()
        table.insertRow(rc)
        duration = correlation_dict['duration']
        count_rate_a = correlation_dict['channel_a']['counts'] / duration
        count_rate_b = correlation_dict['channel_b']['counts'] / duration

        fnw = QtWidgets.QTableWidgetItem(f"{filename.stem}")
        fnw.setToolTip(f'{filename.as_posix()}')
        table.setItem(rc, 0, fnw)
        table.setItem(rc, 1, QtWidgets.QTableWidgetItem(f"{count_rate_a: 0.2f}"))
        table.setItem(rc, 2, QtWidgets.QTableWidgetItem(f"{count_rate_b: 0.2f}"))
        table.setItem(rc, 3, QtWidgets.QTableWidgetItem(f"{duration: 0.2f}"))

    def open_correlation_folder(self, folder: pathlib.Path = None):
        chisurf.logging.log(0, "WizardFcsMerger::open_correlation_folder")
        self.tableWidget.setRowCount(0)
        if folder is None:
            folder = self.correlation_folder
        selected_files = sorted(list(folder.glob('*.json.gz')))
        self.correlations.clear()
        for file in selected_files:
            with chisurf.fio.open_maybe_zipped(file) as fp:
                d = json.load(fp)
                self.append_correlation(file, d)
        chisurf.logging.log(0, 'Opening analysis folder')
        print(list(selected_files))
        self.lineEdit_2.setText(self.target_filepath.as_posix())
        self.update_plots()

    @property
    def target_filepath(self) -> pathlib.Path:
        correlation_filename = self.correlation_folder.stem + '.cor'
        filename = self.correlation_folder.parent / correlation_filename
        return filename

    def save_mean_correlation(self, evt = None, filename: pathlib.Path = None):
        chisurf.logging.log(0, "WizardFcsMerger::save_mean_correlation")
        correlation = self.mean_correlation
        if filename is None:
            filename = self.target_filepath
        chisurf.logging.log(0, 'Saving:', filename)
        suren_column = np.zeros_like(correlation['x'])
        suren_column[0] = correlation['duration']
        suren_column[1] = correlation['count_rate']
        c = np.vstack([correlation['x'] * 1000.0, correlation['y'], suren_column, correlation['ey']])
        np.savetxt(filename.as_posix(), c.T, delimiter='\t')

    def onRemoveRow(self):
        table = self.tableWidget
        rc = table.rowCount()
        idx = int(table.currentIndex().row())
        if rc >= 0:
            if idx < 0:
                idx = 0
            table.removeRow(idx)
            self.correlations.pop(idx)
            self.update_plots()

    @chisurf.gui.decorators.init_with_ui("fcs_merger.ui")
    def __init__(self, *args, **kwargs):
        self.setTitle("Correlation merging")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sizePolicy)
        self.textEdit.setVisible(False)

        self.correlations: typing.List[dict] = list()

        chisurf.gui.decorators.lineEdit_dragFile_injector(self.lineEdit, call=self.open_correlation_folder)

        # Create plots
        self.pw_fcs = pg.plot()
        self.pw_fcs.resize(100, 150)
        self.plot_item_fcs = self.pw_fcs.getPlotItem()
        self.plot_item_fcs.setLogMode(True, False)
        self.horizontalLayout_3.addWidget(self.pw_fcs)

        self.pw_fcs_mean = pg.plot()
        self.pw_fcs_mean.resize(100, 150)
        self.plot_item_fcs_mean = self.pw_fcs_mean.getPlotItem()
        self.plot_item_fcs_mean.setLogMode(True, False)
        self.horizontalLayout_3.addWidget(self.pw_fcs_mean)

        # Connect actions
        self.actionRowDoubleClicked.triggered.connect(self.onRemoveRow)
        self.actionRowSingleClick.triggered.connect(self.update_plots)
        self.toolButton_3.clicked.connect(self.save_mean_correlation)

        # self.actionUpdate_Values.triggered.connect(self.update_parameter)
        # self.actionUpdateUI.triggered.connect(self.updateUI)
        # self.actionFile_changed.triggered.connect(self.read_tttr)
        # self.actionRegionUpdate.triggered.connect(self.onRegionUpdate)
        #
        # self.toolButton_2.toggled.connect(self.pw_mcs.setVisible)
        # self.toolButton_3.toggled.connect(self.pw_decay.setVisible)
        # self.toolButton_4.toggled.connect(self.pw_filter.setVisible)
        # self.toolButton_5.clicked.connect(self.save_filter_data)
        # self.toolButton_3.clicked.connect(self.correlate_data)

