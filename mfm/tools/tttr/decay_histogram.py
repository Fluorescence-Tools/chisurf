from __future__ import annotations

import sys
import os
import copy
import re

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, uic
import qdarkstyle

import mfm
import mfm.experiments
import mfm.experiments.data
from mfm.fluorescence.tcspc import weights
from mfm.io.widgets import SpcFileWidget

plot_settings = mfm.settings.gui['plot']
pyqtgraph_settings = plot_settings["pyqtgraph"]
lw = plot_settings['line_width']


class HistogramTTTR(QtWidgets.QWidget):

    @property
    def curve_name(self):
        s = str(self.lineEdit.text())
        if len(s) == 0:
            return "no-name"
        else:
            return s

    def onRemoveDataset(self):
        selected_index = [i.row() for i in self.cs.selectedIndexes()]
        l = list()
        for i, c in enumerate(self._curves):
            if i not in selected_index:
                l.append(c)
        self._curves = l
        self.cs.update()
        self.plot_curves()

    def clear_curves(self):
        self._curves = list()
        plot = self.plot.getPlotItem()
        plot.clear()

    def get_data_curves(self, **kwargs):
        return self._curves

    def plot_curves(self):
        self.legend.close()
        plot = self.plot.getPlotItem()
        plot.clear()
        self.legend = plot.addLegend()

        current_curve = self.cs.selected_curve_index
        for i, curve in enumerate(self._curves):
            l = lw * 0.5 if i != current_curve else 1.5 * lw
            color = mfm.settings.colors[i % len(mfm.settings.colors)]['hex']
            plot.plot(x=curve.x, y=curve.y,
                      pen=pg.mkPen(color, width=l),
                      name=curve.name)

        plot.setLogMode(x=False, y=True)
        plot.showGrid(True, True, 1.0)

    def add_curve(self):
        d = self.decay.load_data()

        self._curves.append(copy.deepcopy(d))
        self.cs.update()
        self.plot_curves()

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self._curves = list()
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tttr_histogram.ui"
            ),
            self
        )
        w = TCSPCSetupTTTRWidget()
        self.decay = w
        self.verticalLayout.addWidget(w)
        w.show()

        self.cs = mfm.widgets.curve.ExperimentalDataSelector(
            get_data_sets=self.get_data_curves,
            click_close=False
        )
        self.verticalLayout_6.addWidget(self.cs)

        w.tcspcTTTR.tcspcTTTRWidget.pushButton.clicked.connect(self.add_curve)
        #self.cs.itemClicked[QListWidgetItem].connect(self.plot_curves)

        self.plot = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])
        plot = self.plot.getPlotItem()
        self.verticalLayout_9.addWidget(self.plot)
        self.legend = plot.addLegend()
        self.cs.onRemoveDataset = self.onRemoveDataset


class TCSPCSetupTTTRWidget(QtWidgets.QWidget):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self, **kwargs)
        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout

        self.tcspcTTTR = TcspcTTTRWidget(self)
        self.rep_rate = self.tcspcTTTR.rep_rate
        self.layout.addWidget(self.tcspcTTTR)

    def load_data(self, **kwargs):
        d = mfm.experiments.data.DataCurve()
        self.tcspcTTTR.makeHist()
        d.filename = self.tcspcTTTR.spcFileWidget.sample_name
        d.name = self.tcspcTTTR.spcFileWidget.sample_name + "_" + str(self.tcspcTTTR.chs)
        x = self.tcspcTTTR.x
        y = self.tcspcTTTR.y
        w = weights(y)
        d.x, d.y = x, y
        d.set_weights(w)
        return d


class TcspcTTTRWidget(QtWidgets.QWidget):
    histDone = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.tcspcTTTRWidget = QtWidgets.QWidget(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tcspcTTTRWidget.ui"
            ),
            self.tcspcTTTRWidget
        )
        self.spcFileWidget = SpcFileWidget(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.spcFileWidget)
        layout.addWidget(self.tcspcTTTRWidget)

        self.tcspcTTTRWidget.comboBox.currentIndexChanged[int].connect(self.onTacDivChanged)
        self.spcFileWidget.actionLoad_sample.triggered.connect(self.onLoadFile)
        self.spcFileWidget.actionDt_changed.triggered.connect(self.onTacDivChanged)

    @property
    def nPh(self):
        return int(self.tcspcTTTRWidget.lineEdit_5.text())

    @nPh.setter
    def nPh(self, v):
        self.tcspcTTTRWidget.lineEdit_5.setText("%d" % v)

    @property
    def div(self):
        return int(self.tcspcTTTRWidget.comboBox.currentText())

    @property
    def rep_rate(self):
        return self.spcFileWidget.rep_rate

    @property
    def dt_min(self):
        return float(self.tcspcTTTRWidget.doubleSpinBox.value())

    @property
    def use_dtmin(self):
        return self.tcspcTTTRWidget.checkBox.isChecked()

    @property
    def histSelection(self):
        return str(self.tcspcTTTRWidget.lineEdit.text()).replace(" ", "").upper()

    @property
    def inverted_selection(self):
        return self.tcspcTTTRWidget.checkBox_2.isChecked()

    @property
    def nTAC(self):
        return int(self.tcspcTTTRWidget.lineEdit_4.text())

    @nTAC.setter
    def nTAC(self, v):
        self.tcspcTTTRWidget.lineEdit_4.setText("%d" % v)

    def makeHist(self):
        # get right data
        h5 = self.spcFileWidget._photons.h5
        nodeName = self.spcFileWidget._photons.sample_names[0] #str(self.spcFileWidget.comboBox.currentText())
        table = h5.get_node('/' + nodeName, 'photons')
        selection_tac = np.ma.array([row['TAC'] for row in table.where(self.histSelection)])[:-1]

        if self.use_dtmin:
            if self.inverted_selection:
                print("inverted selection")
                selection_mask = np.diff(np.array([row['MT'] for row in table.where(self.histSelection)])) < self.dt_min
            else:
                print("normal selection")
                selection_mask = np.diff(np.array([row['MT'] for row in table.where(self.histSelection)])) > self.dt_min
            print("dMTmin: %s" % self.dt_min)
            selection_tac.mask = selection_mask
            self.nPh = selection_mask.sum()
        else:
            self.nPh = selection_tac.shape[0]
        print("nPh: %s" % self.nPh)

        ta = selection_tac.compressed().astype(np.int32)
        ta /= self.div
        hist = np.bincount(ta, minlength=self.nTAC)
        self.y = hist.astype(np.float64)
        self.x = np.arange(len(hist), dtype=np.float64) + 1.0
        self.x *= self.dt
        self.xt = self.x

        ex = r'(ROUT==\d+)'
        routCh = re.findall(ex, self.histSelection)
        self.chs = [int(ch.split('==')[1]) for ch in routCh]
        self.tcspcTTTRWidget.lineEdit_3.setText("%s" % self.chs)
        curve = mfm.experiments.data.DataCurve(setup=self)
        curve.x = self.x
        curve.y = self.y
        self.histDone.emit(self.nROUT, self.nTAC, self.chs, curve)

    def onTacDivChanged(self):
        self.dtBase = self.spcFileWidget.dt
        self.tacDiv = float(self.tcspcTTTRWidget.comboBox.currentText())
        self.nTAC = (self.spcFileWidget.nTAC + 1) / self.tacDiv
        self.dt = self.dtBase * self.tacDiv
        self.tcspcTTTRWidget.lineEdit_2.setText("%.3f" % self.dt)

    def onLoadFile(self):
        self.nROUT = self.spcFileWidget.nROUT
        self.onTacDivChanged()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = HistogramTTTR()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
