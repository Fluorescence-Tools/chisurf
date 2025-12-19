import typing

import pyqtgraph as pg

import chisurf
import chisurf.gui.decorators
from chisurf.gui import QtWidgets


def setup_ui(page):
    page.setTitle("Correlation merging")
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    page.setSizePolicy(sizePolicy)
    page.textEdit.setVisible(False)

    page.correlations: typing.List[dict] = list()

    chisurf.gui.decorators.lineEdit_dragFile_injector(page.lineEdit, call=page.open_correlation_folder)

    # Setup plots
    page.pw_fcs = pg.PlotWidget(parent=page, title='FCS')
    page.pw_fcs.resize(100, 150)
    page.plot_item_fcs = page.pw_fcs.getPlotItem()
    page.plot_item_fcs.setLogMode(True, False)
    page.horizontalLayout_3.addWidget(page.pw_fcs)

    page.pw_fcs_mean = pg.PlotWidget(parent=page, title='FCS Merged')
    page.pw_fcs_mean.resize(100, 150)
    page.plot_item_fcs_mean = page.pw_fcs_mean.getPlotItem()
    page.plot_item_fcs_mean.setLogMode(True, False)
    page.horizontalLayout_3.addWidget(page.pw_fcs_mean)

    # Setup table widget with an extra column for the merge checkbox.
    page.tableWidget.setColumnCount(5)
    page.tableWidget.setHorizontalHeaderLabels(["Use", "File", "CR A (kHz)", "CR B (kHz)", "Duration (s)"])

    # Remove the double-click deletion action and instead toggle the checkbox on double click.
    # self.actionRowDoubleClicked.triggered.connect(self.onRemoveRow)  <-- Removed!
    page.tableWidget.itemDoubleClicked.connect(page.onRowDoubleClicked)
    page.actionRowSingleClick.triggered.connect(page.update_plots)
    page.toolButton_3.clicked.connect(page.save_mean_correlation)
