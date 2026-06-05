import typing

import matplotlib
import numpy as np
import pyqtgraph as pg
import tttrlib

import chisurf
import chisurf.gui.decorators
from chisurf.gui import QtCore, QtGui, QtWidgets

from .tttr_burst_finder_utils import (
    CommaSeparatedIntegersValidator,
    CountRateFilterSettings,
    DeltaMacroTimeFilterSettings,
)


def setup_ui(
    page,
    *,
    windows,
    detectors,
    callback_function=None,
    show_dT: bool = True,
    show_burst_histogram: bool = True,
    show_mcs: bool = True,
    show_decay: bool = True,
    show_filter: bool = True,
    initial_trace_bin_width: float = 1.0,
    initial_photon_threshold: int = 100,
    initial_tw_size: float = 1.0,
    initial_max_gap: int = 6,
    initial_decay_coarse: int = 16,
    initial_number_of_burst_bins: int = 50,
    initial_dT_min: float = 0.0001,
    initial_dT_max: float = 0.15,
):
    page.setTitle("Photon filter / burst finder")

    page.windows = windows
    page.detectors = detectors
    page.fill_detectors(detectors)
    page.fill_pie_windows(windows)
    page.callback_function = callback_function

    page.comboBox.clear()
    page.comboBox.insertItem(0, "Auto")
    page.comboBox.insertItems(1, list(tttrlib.TTTR.get_supported_container_names()))

    page.settings: dict = dict()
    tttr_filenames: typing.List[typing.Any] = list()
    page.settings['tttr_filenames'] = tttr_filenames
    page.settings['count_rate_filter']: CountRateFilterSettings = dict()
    page.settings['delta_macro_time_filter']: DeltaMacroTimeFilterSettings = dict()
    page.filter_data_saved = False

    def cc():
        page.spinBox_4.setMaximum(len(page.settings['tttr_filenames']) - 1)
        page.spinBox_4.setValue(len(page.settings['tttr_filenames']) - 1)
        page.read_tttr()

    page.textEdit.setVisible(False)
    chisurf.gui.decorators.lineEdit_dragFile_injector(
        page.lineEdit,
        call=cc,
        target=page.settings['tttr_filenames']
    )
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    page.setSizePolicy(sizePolicy)

    page.tttr = None

    page._dT_min = 0.0001
    page._dT_max = 0.15

    color_all = QtGui.QColor(255, 255, 0, 64)
    color_selected = QtGui.QColor(0, 255, 255, 255)
    pen2 = pg.mkPen(color_all, width=1, style=QtCore.Qt.SolidLine)
    pen1 = pg.mkPen(color_selected, width=1, style=QtCore.Qt.SolidLine)

    page.pw_burst_histogram = pg.PlotWidget(parent=page, title='Burst size distribution')
    page.plot_burst_histogram = page.pw_burst_histogram.getPlotItem()
    page.pw_burst_histogram.resize(100, 80)
    page.pw_burst_histogram.setVisible(show_burst_histogram)

    page.pw_dT = pg.PlotWidget(parent=page, title='Delta macrotime')
    page.plot_item_dt = page.pw_dT.getPlotItem()
    page.plot_unselected = page.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen2)
    page.plot_selected = page.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen1)
    page.pw_dT.resize(200, 40)
    page.pw_dT.setVisible(show_dT)

    page.pw_mcs = pg.PlotWidget(parent=page, title='Count-rate display')
    page.plot_item_mcs = page.pw_mcs.getPlotItem()
    page.plot_item_mcs.setLabel('bottom', 'Time (s)')
    page.plot_item_mcs.setLabel('left', 'Intensity')
    page.plot_mcs_all = page.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen2)
    page.plot_mcs_selected = page.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen1)
    page.pw_mcs.resize(200, 80)
    page.pw_mcs.setVisible(show_mcs)

    page.pw_decay = pg.PlotWidget(parent=page, title='Microtime histogram')
    page.plot_item_decay = page.pw_decay.getPlotItem()
    page.plot_item_decay.setLabel('bottom', 'Time (ns)')
    page.plot_item_decay.setLabel('left', 'Counts')
    page.plot_decay_all = page.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen2)
    page.plot_decay_selected = page.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen1)
    page.pw_decay.resize(200, 80)
    page.pw_decay.setVisible(show_decay)

    page.pw_filter = pg.PlotWidget(parent=page, title='Filered/Photons')
    page.pw_filter.setXLink(page.pw_dT)
    page.pw_dT.setMouseEnabled(x=False, y=False)
    page.pw_filter.setMouseEnabled(x=False, y=False)
    page.plot_item_sel = page.pw_filter.getPlotItem()
    page.plot_select = page.plot_item_sel.plot(x=[1.0], y=[1.0])
    page.pw_filter.resize(200, 20)
    page.pw_filter.setVisible(show_filter)

    page.plot_item_dt.setLogMode(False, True)
    page.plot_item_decay.setLogMode(False, True)
    page.plot_item_sel.setLogMode(False, False)

    colors = chisurf.core.settings.gui['plot']['colors']
    ca = list(matplotlib.colors.hex2color(colors["region_selector"]))
    co = [ca[0] * 255, ca[1] * 255, ca[2] * 255, colors["region_selector_alpha"]]
    page.region_selector = pg.LinearRegionItem(
        brush=co,
        orientation='horizontal',
        values=(np.log10(0.001), np.log10(0.15))
    )
    page.pw_dT.addItem(page.region_selector)

    def _on_region_change_finished(evt):
        lb, ub = page.region_selector.getRegion()
        if page.pw_dT.getAxis('left').logMode:
            lb, ub = 10 ** lb, 10 ** ub
        page._dT_min = lb
        page._dT_max = ub

        page.doubleSpinBox_2.blockSignals(True)
        page.doubleSpinBox_3.blockSignals(True)
        page.doubleSpinBox_2.setValue(lb)
        page.doubleSpinBox_3.setValue(ub)
        page.doubleSpinBox_2.blockSignals(False)
        page.doubleSpinBox_3.blockSignals(False)

        page.update_plots()
        page.update_output_path()

    page.region_selector.sigRegionChangeFinished.connect(_on_region_change_finished)

    page.doubleSpinBox_4.setValue(initial_trace_bin_width)
    page.spinBox.setValue(initial_photon_threshold)
    page.spinBox_7.setValue(initial_max_gap)
    page.spinBox_5.setValue(initial_decay_coarse)
    page.spinBox_6.setValue(initial_number_of_burst_bins)
    page._dT_min = initial_dT_min
    page._dT_max = initial_dT_max
    page.doubleSpinBox_2.setValue(initial_dT_min)
    page.doubleSpinBox_3.setValue(initial_dT_max)
    page.doubleSpinBox.setValue(initial_tw_size)
    page.region_selector.setRegion((np.log10(initial_dT_min), np.log10(initial_dT_max)))

    page.gridLayout_6.addWidget(page.pw_dT, 0, 0, 1, 3)
    page.gridLayout_6.addWidget(page.pw_filter, 1, 0, 1, 3)
    page.gridLayout_6.addWidget(page.pw_mcs, 2, 0, 1, 1)
    page.gridLayout_6.addWidget(page.pw_decay, 2, 1, 1, 1)
    page.gridLayout_6.addWidget(page.pw_burst_histogram, 0, 1, 2, 1)

    page.actionUpdate_Values.triggered.connect(page.update_parameter)
    page.actionUpdateUI.triggered.connect(page.updateUI)
    page.actionFile_changed.triggered.connect(page.read_tttr)
    page.actionRegionUpdate.triggered.connect(page.onRegionUpdate)

    page.toolButton_2.toggled.connect(page.pw_mcs.setVisible)
    page.toolButton_3.toggled.connect(page.pw_decay.setVisible)
    page.toolButton_4.toggled.connect(page.pw_filter.setVisible)
    page.toolButton_7.toggled.connect(page.pw_burst_histogram.setVisible)
    page.toolButton_5.clicked.connect(page.save_selection)
    page.toolButton_6.clicked.connect(page.onClearFiles)

    page.comboBox_2.currentTextChanged.connect(page.update_detectors)
    page.comboBox_3.currentTextChanged.connect(page.update_pie_windows)

    validator = CommaSeparatedIntegersValidator()
    page.lineEdit_4.setValidator(validator)

    page.update_parameter()
