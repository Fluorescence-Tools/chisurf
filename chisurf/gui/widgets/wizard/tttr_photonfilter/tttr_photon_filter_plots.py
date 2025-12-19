import matplotlib
import numpy as np
import pyqtgraph as pg

import chisurf

from chisurf.gui import QtGui, QtWidgets, QtCore


def create_plots(page, colors):
    # Create & configure plots
    page.pw_burst_histogram = pg.PlotWidget(parent=page, title="Burst histogram")
    page.plot_burst_histogram = page.pw_burst_histogram.getPlotItem()
    page.pw_burst_histogram.setLabel('left', 'Counts')
    page.pw_burst_histogram.setLabel('bottom', 'Burst size (Nbr. Photons)')
    page.pw_burst_histogram.resize(100, 80)

    color_all = QtGui.QColor(255, 255, 0, 64)
    color_selected = QtGui.QColor(0, 255, 255, 255)
    pen2 = pg.mkPen(color_all, width=1, style=QtCore.Qt.SolidLine)
    pen1 = pg.mkPen(color_selected, width=1, style=QtCore.Qt.SolidLine)

    page.pw_dT = pg.PlotWidget(parent=page, title="Delta macro-time")
    page.pw_dT.setLabel('left', 'dT (ms)')
    page.pw_dT.setLabel('bottom', 'Photon Index')
    page.plot_item_dt = page.pw_dT.getPlotItem()
    page.plot_unselected = page.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen2)
    page.plot_selected = page.plot_item_dt.plot(x=[1.0], y=[1.0], pen=pen1)
    page.pw_dT.resize(200, 40)

    page.pw_mcs = pg.PlotWidget(parent=page, title="Count rate display")
    page.pw_mcs.setLabel('left', 'Intensity (kHz)')
    page.pw_mcs.setLabel('bottom', 'Time (s)')
    page.plot_item_mcs = page.pw_mcs.getPlotItem()
    page.plot_mcs_all = page.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen2)
    page.plot_mcs_selected = page.plot_item_mcs.plot(x=[1.0], y=[1.0], pen=pen1)
    page.pw_mcs.resize(200, 80)

    page.pw_decay = pg.PlotWidget(parent=page, title="Microtime histogram")
    page.pw_decay.setLabel('left', 'Counts')
    page.pw_decay.setLabel('bottom', 'Microtime (ns)')
    page.plot_item_decay = page.pw_decay.getPlotItem()
    page.plot_decay_all = page.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen2)
    page.plot_decay_selected = page.plot_item_decay.plot(x=[1.0], y=[1.0], pen=pen1)
    page.pw_decay.resize(200, 80)

    page.pw_filter = pg.PlotWidget(parent=page, title="Filter/selection")
    page.pw_filter.setLabel('left', 'Selected (1) / Unselected (0)')
    page.pw_filter.setLabel('bottom', 'Photon Index')
    page.pw_filter.setXLink(page.pw_dT)
    page.pw_dT.setMouseEnabled(x=False, y=False)
    page.pw_filter.setMouseEnabled(x=False, y=False)
    page.plot_item_sel = page.pw_filter.getPlotItem()
    page.plot_select = page.plot_item_sel.plot(x=[1.0], y=[1.0])
    page.pw_filter.resize(200, 20)

    page.plot_item_dt.setLogMode(False, True)
    page.plot_item_decay.setLogMode(False, True)
    page.plot_item_sel.setLogMode(False, False)

    ca = list(matplotlib.colors.hex2color(colors["region_selector"]))
    co = [ca[0] * 255, ca[1] * 255, ca[2] * 255, colors["region_selector_alpha"]]
    page.region_selector = pg.LinearRegionItem(
        brush=co, orientation='horizontal',
        values=(np.log10(page._dT_min), np.log10(page._dT_max))
    )
    page.pw_dT.addItem(page.region_selector)

    def onRegionUpdate(evt):
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

    page.region_selector.sigRegionChangeFinished.connect(onRegionUpdate)


def place_plots(page):
    # Place plots in layout
    page.gridLayout_6.addWidget(page.pw_dT, 0, 0, 1, 3)
    page.gridLayout_6.addWidget(page.pw_filter, 1, 0, 1, 3)
    page.gridLayout_6.addWidget(page.pw_mcs, 2, 0, 1, 1)
    page.gridLayout_6.addWidget(page.pw_decay, 2, 1, 1, 1)
    page.gridLayout_6.addWidget(page.pw_burst_histogram, 0, 1, 2, 1)
