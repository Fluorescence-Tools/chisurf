from __future__ import annotations
from chisurf import typing

import sys
import re

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

import chisurf.decorators
import chisurf.curve
import chisurf.data
import chisurf.gui.decorators
import chisurf.gui.widgets.experiments.widgets
import chisurf.fluorescence.tcspc
import chisurf.gui.widgets.fio

plot_settings = chisurf.settings.gui['plot']
lw = plot_settings['line_width']


class HistogramTTTR(
    QtWidgets.QWidget,
    chisurf.curve.CurveGroup
):

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="tttr_histogram.ui"
    )
    def __init__(self):
        self.setContentsMargins(0, 0, 0, 0)
        self._curves = list()
        self.tcspc_setup_widget = TcspcTTTRWidget()
        self.verticalLayout.addWidget(
            self.tcspc_setup_widget
        )
        self.curve_selector = chisurf.gui.widgets.experiments.widgets.ExperimentalDataSelector(
            get_data_sets=self.get_data_curves,
            click_close=False
        )
        self.verticalLayout_6.addWidget(self.curve_selector)
        self.plot = pg.PlotWidget()
        plot = self.plot.getPlotItem()
        self.verticalLayout_9.addWidget(self.plot)
        self.legend = plot.addLegend()
        self.curve_selector.onRemoveDataset = self.remove_curve

        # Actions
        self.tcspc_setup_widget.pushButton.clicked.connect(
            self.add_curve
        )

    @property
    def curve_name(self):
        s = str(self.lineEdit.text())
        if len(s) == 0:
            return "no-name"
        else:
            return s

    def remove_curve(
            self,
            selected_index: typing.List[int] = None
    ):
        if selected_index is None:
            selected_index = [
                i.row() for i in self.curve_selector.selectedIndexes()
            ]
        super().remove_curve(selected_index)
        self.curve_selector.update()
        self.plot_curves()

    def clear_curves(self):
        super().clear_curves()
        plot = self.plot.getPlotItem()
        plot.clear()

    def plot_curves(self):
        self.legend.close()
        plot = self.plot.getPlotItem()
        plot.clear()
        self.legend = plot.addLegend()

        current_curve = self.curve_selector.selected_curve_index
        for i, curve in enumerate(self._curves):
            l = lw * 0.5 if i != current_curve else 1.5 * lw
            color = chisurf.settings.colors[i % len(chisurf.settings.colors)]['hex']
            plot.plot(x=curve.x, y=curve.y,
                      pen=pg.mkPen(color, width=l),
                      name=curve.name)

        plot.setLogMode(x=False, y=True)
        plot.showGrid(True, True, 1.0)

    def add_curve(
            self,
            v: chisurf.curve.Curve = None
    ):
        if v is None:
            v = self.tcspc_setup_widget.load_data()
            self.curve_selector.update()
            self.plot_curves()
        super().add_curve(v)


class TcspcTTTRWidget(
    QtWidgets.QWidget
):

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="tcspcTTTRWidget.ui"
    )
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        self.spcFileWidget = chisurf.gui.widgets.fio.SpcFileWidget(self)
        self.layout().insertWidget(0, self.spcFileWidget)

        # Actions
        self.comboBox.currentIndexChanged[int].connect(self.onTacDivChanged)
        self.spcFileWidget.actionLoad_sample.triggered.connect(self.onLoadFile)
        self.spcFileWidget.actionLoad_sample.triggered.connect(self.onTacDivChanged)
        self.spcFileWidget.actionDt_changed.triggered.connect(self.onTacDivChanged)

    @property
    def nPh(
            self
    ) -> int:
        return int(self.lineEdit_5.text())

    @nPh.setter
    def nPh(self, v):
        self.lineEdit_5.setText("%d" % v)

    @property
    def div(
            self
    ) -> int:
        return int(
            self.comboBox.currentText()
        )

    @property
    def rep_rate(self):
        return self.spcFileWidget.rep_rate

    @property
    def dt_min(self):
        return float(self.doubleSpinBox.value())

    @property
    def use_dtmin(self):
        return self.checkBox.isChecked()

    @property
    def histSelection(self):
        return str(
            self.lineEdit.text()
        ).replace(" ", "").upper()

    @property
    def inverted_selection(self):
        return self.checkBox_2.isChecked()

    @property
    def nTAC(self):
        return int(self.lineEdit_4.text())

    @nTAC.setter
    def nTAC(self, v):
        self.lineEdit_4.setText("%d" % v)

    def make_histogram(self):
        # get right data
        table = self.spcFileWidget.photons.photon_table
        selection_tac = np.ma.array(
            [
                row['TAC'] for row in table.where(self.histSelection)
            ]
        )[:-1]

        if self.use_dtmin:
            if self.inverted_selection:
                selection_mask = np.diff(
                    np.array(
                        [row['MT'] for row in table.where(self.histSelection)]
                    )
                ) < self.dt_min
            else:
                selection_mask = np.diff(
                    np.array(
                        [row['MT'] for row in table.where(self.histSelection)]
                    )
                ) > self.dt_min
            selection_tac.mask = selection_mask
            self.nPh = selection_mask.sum()
        else:
            self.nPh = selection_tac.shape[0]

        ta = selection_tac.compressed().astype(np.int32)
        ta //= self.div
        hist = np.bincount(
            ta,
            minlength=self.spcFileWidget.photons.n_tac
        )
        self.y = hist.astype(np.float64)
        self.x = np.arange(len(hist), dtype=np.float64) + 1.0
        self.x *= self.spcFileWidget.photons.dt
        self.xt = self.x

        ex = r'(ROUT==\d+)'
        routCh = re.findall(ex, self.histSelection)
        self.chs = [int(ch.split('==')[1]) for ch in routCh]
        self.lineEdit_3.setText("%s" % self.chs)

    def onTacDivChanged(self):
        self.dtBase = self.spcFileWidget.dt
        self.tacDiv = float(self.comboBox.currentText())
        self.nTAC = (self.spcFileWidget.nTAC + 1) / self.tacDiv
        self.dt = self.dtBase * self.tacDiv
        self.lineEdit_2.setText("%.3f" % self.dt)

    def onLoadFile(self):
        #self.nROUT = self.spcFileWidget.nROUT
        self.onTacDivChanged()

    def load_data(
            self,
            *args,
            **kwargs
    ):
        self.make_histogram()
        x = self.x
        y = self.y
        name = self.spcFileWidget.sample_name + "_" + str(self.chs)
        d = chisurf.data.DataCurve(
            x=x,
            y=y,
            ey=chisurf.fluorescence.tcspc.counting_noise(
                decay=y
            ),
            name=name
        )
        return d


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = HistogramTTTR()
    win.show()
    sys.exit(app.exec_())
