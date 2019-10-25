from __future__ import annotations

import copy
from copy import deepcopy
import os
import sys

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, uic
import qdarkstyle

import chisurf.settings as mfm
import chisurf.experiments.widgets
import chisurf.tools
import chisurf.fluorescence
import chisurf.experiments.data
import chisurf.settings as settings
import chisurf.widgets
from chisurf.fio.widgets import SpcFileWidget

settings = chisurf.settings.cs_settings['correlator']
plot_settings = chisurf.settings.gui['plot']
lw = plot_settings['line_width']


class CorrelateTTTR(QtWidgets.QWidget):

    name = "tttr-correlate"

    @property
    def curve_name(self):
        s = str(self.lineEdit.text())
        if len(s) == 0:
            return "no-name"
        else:
            return s

    def onRemoveDataset(self):
        selected_index = [
            i.row() for i in self.cs.selectedIndexes()
        ]
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
        plot.setLogMode(x=True, y=False)
        plot.showGrid(True, True, 1.0)

        current_curve = self.cs.selected_curve_index
        for i, curve in enumerate(self._curves):
            l = lw * 0.5 if i != current_curve else 1.5 * lw
            color = chisurf.settings.colors[
                i % len(chisurf.settings.colors)
                ]['hex']
            plot.plot(x=curve.x, y=curve.y,
                      pen=pg.mkPen(color, width=l),
                      name=curve.name)

    def add_curve(self):
        d = self.corr.correlator.data
        d.setup = self.corr
        d.name = self.corr.fileWidget.sample_name
        self._curves.append(copy.deepcopy(d))
        self.cs.update()
        self.plot_curves()

    @chisurf.decorators.init_with_ui(ui_filename="tttr_correlate.ui")
    def __init__(self, *args, **kwargs):
        self._curves = list()
        w = chisurf.tools.tttr.correlate.FCStttr()
        self.corr = w
        self.verticalLayout.addWidget(w)
        w.show()

        self.cs = chisurf.experiments.widgets.ExperimentalDataSelector(
            get_data_sets=self.get_data_curves,
            click_close=False
        )
        self.verticalLayout_6.addWidget(self.cs)

        w.correlator.pushButton_3.clicked.connect(
            w.correlator.correlator_thread.start
        )
        w.correlator.correlator_thread.finished.connect(self.add_curve)
        #self.curve_selector.itemClicked[QListWidgetItem].connect(self.plot_curves)

        self.plot = pg.PlotWidget()
        plot = self.plot.getPlotItem()
        self.verticalLayout_9.addWidget(self.plot)
        self.legend = plot.addLegend()
        self.cs.onRemoveDataset = self.onRemoveDataset


class Correlator(QtCore.QThread):

    procDone = QtCore.pyqtSignal(bool)
    partDone = QtCore.pyqtSignal(int)

    @property
    def data(self):
        if isinstance(
                self._data,
                chisurf.experiments.data.DataCurve
        ):
            return self._data
        else:
            return chisurf.experiments.data.DataCurve(
                setup=self
            )

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.p = kwargs.get('parent', None)
        self.exiting = False
        self._data = None
        self._results = list()
        self._dt1 = 0
        self._dt2 = 0

    def getWeightStream(
            self,
            tacWeighting
    ):
        """
        :param tacWeighting: is either a list of integers or a numpy-array.
        If it's a list of integers the integers correspond to channel-numbers.
        In this case all photons have an equal weight of one. If tacWeighting
        is a numpy-array it should be of shape [max-routing, number of TAC
        channels]. The array contains np.floats with weights for photons
        arriving at different TAC-times.
        :return: numpy-array with same length as photon-stream, each photon
        is associated to one weight.
        """
        print("Correlator:getWeightStream")
        photons = self.p.photon_source.photons
        if type(tacWeighting) is list:
            print("channel-wise selection")
            print("Max-Rout: %s" % photons.n_rout)
            wt = np.zeros([photons.n_rout, photons.n_tac], dtype=np.float32)
            wt[tacWeighting] = 1.0
        elif type(tacWeighting) is np.ndarray:
            print("TAC-weighted")
            wt = tacWeighting
        w = chisurf.fluorescence.fcs.correlate.get_weights(
            photons.rout,
            photons.tac,
            wt,
            photons.nPh
        )
        return w

    def run(self):
        data = chisurf.experiments.data.DataCurve()

        w1 = self.getWeightStream(self.p.ch1)
        w2 = self.getWeightStream(self.p.ch2)
        print("Correlation running...")
        print("Correlation method: %s" % self.p.method)
        print("Fine-correlation: %s" % self.p.fine)
        print("Nbr. of correlations: %s" % self.p.split)
        photons = self.p.photon_source.photons

        self._results = list()
        n = len(photons)
        nGroup = n / self.p.split
        self.partDone.emit(0.0)
        for i in range(0, n - n % nGroup, nGroup):
            nbr = ((i + 1) / nGroup + 1)
            print("Correlation Nbr.: %s" % nbr)
            p = photons[i:i + nGroup]
            wi1, wi2 = w1[i:i + nGroup], w2[i:i + nGroup]
            if self.p.method == 'tp':
                results = chisurf.fluorescence.fcs.correlate.log_corr(
                    p.mt, p.tac, p.rout, p.cr_filter,
                    wi1, wi2, self.p.B, self.p.nCasc,
                    self.p.fine, photons.n_tac
                )
                np_1 = results['number_of_photons_ch1']
                np_2 = results['number_of_photons_ch2']
                dt_1 = results['measurement_time_ch1']
                dt_2 = results['measurement_time_ch2']
                tau = results['correlation_time_axis']
                corr = results['correlation_amplitude']
                cr = chisurf.fluorescence.fcs.correlate.normalize(
                    np_1, np_2, dt_1, dt_2, tau, corr, self.p.B
                )
                cr /= self.p.dt
                dur = float(min(dt_1, dt_2)) * self.p.dt / 1000  # seconds
                tau = tau.astype(np.float64)
                tau *= self.p.dt
                self._results.append([cr, dur, tau, corr])
            self.partDone.emit(float(nbr) / self.p.split * 100)

        # Calculate average correlations
        cors = list()
        taus = list()
        weights = list()
        for c in self._results:
            cr, dur, tau, corr = c
            weight = self.weight(tau, corr, dur, cr)
            weights.append(weight)
            cors.append(corr)
            taus.append(tau)

        cor = np.array(cors)
        w = np.array(weights)

        data.x = np.array(taus).mean(axis=0)[1:]
        data.y = cor.mean(axis=0)[1:]
        data.ey = 1. / w.mean(axis=0)[1:]

        print("correlation done")
        self._data = data
        self.procDone.emit(True)
        self.exiting = True

    def weight(self, tau, cor, dur, cr):
        """
        tau-axis in milliseconds
        correlation amplitude
        dur = duration in seconds
        cr = count-rate in kHz
        """
        if self.p.weighting == 1:
            return chisurf.fluorescence.fcs.weights(
                tau, cor, dur, cr, type='uniform'
            )
        elif self.p.weighting == 0:
            return chisurf.fluorescence.fcs.weights(
                tau, cor, dur, cr, type='suren'
            )


class CorrelatorWidget(QtWidgets.QWidget):

    def __init__(
            self,
            parent,
            photon_source,
            ch1: int = '0',
            ch2: int = '8',
            setup=None,
            nCasc: int = settings['nCasc'],
            B: int = settings['B'],
            split: int = settings['split'],
            weighting: str = settings['weighting'],
            fine: bool = settings['fine'],
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "correlatorWidget.ui"
            ),
            self
        )

        self.nCasc = nCasc
        self.B = B
        self.split = split
        self.weighting = weighting
        self.fine = fine
        self.setup = setup
        self.parent = parent
        self.cr = 0.0
        self.ch1 = ch1
        self.ch2 = ch2

        self.photon_source = photon_source
        self.correlator_thread = Correlator(self)

        # fill widgets
        self.comboBox_3.addItems(chisurf.fluorescence.fcs.weightCalculations)
        self.comboBox_2.addItems(chisurf.fluorescence.fcs.correlationMethods)
        self.checkBox.setChecked(True)
        self.checkBox.setChecked(False)
        self.progressBar.setValue(0.0)

        # connect widgets
        self.pushButton_3.clicked.connect(self.correlator_thread.start)
        self.correlator_thread.partDone.connect(self.updatePBar)

    def updatePBar(self, val):
        self.progressBar.setValue(val)

    @property
    def data(self):
        return self.correlator_thread.data

    @property
    def dt(self):
        dt = self.photon_source.photons.mt_clk
        if self.fine:
            dt /= self.photon_source.photons.n_tac
        return dt

    @property
    def weighting(self):
        return self.comboBox_3.currentIndex()

    @weighting.setter
    def weighting(self, v):
        self.comboBox_3.setCurrentIndex(int(v))

    @property
    def ch1(self):
        return [int(x) for x in str(self.lineEdit_4.text()).split()]

    @ch1.setter
    def ch1(self, v):
        self.lineEdit_4.setText(str(v))

    @property
    def ch2(self):
        return [int(x) for x in str(self.lineEdit_5.text()).split()]

    @ch2.setter
    def ch2(self, v):
        self.lineEdit_5.setText(str(v))

    @property
    def fine(self):
        return int(self.checkBox.isChecked())

    @fine.setter
    def fine(self, v):
        self.checkBox.setCheckState(v)

    @property
    def B(self):
        return int(self.spinBox_3.value())

    @B.setter
    def B(self, v):
        return self.spinBox_3.setValue(v)

    @property
    def nCasc(self):
        return int(self.spinBox_2.value())

    @nCasc.setter
    def nCasc(self, v):
        return self.spinBox_2.setValue(v)

    @property
    def method(self):
        return str(self.comboBox_2.currentText())

    @property
    def split(self):
        return int(self.spinBox.value())

    @split.setter
    def split(self, v):
        self.spinBox.setValue(v)


class CrFilterWidget(QtWidgets.QWidget):

    def __init__(
            self,
            parent,
            photon_source,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'fcs-cr-filter.ui'
            ),
            self
        )
        self.photon_source = photon_source
        self.parent = parent

        self.verbose = kwargs.get('verbose', chisurf.verbose)
        self.time_window = kwargs.get('time_window', settings['time_window'])
        self.max_count_rate = kwargs.get('max_count_rate', settings['max_count_rate'])

        self.sample_name = self.photon_source.sample_name

    @property
    def max_count_rate(self):
        return float(self.lineEdit_6.text())

    @max_count_rate.setter
    def max_count_rate(self, v):
        self.lineEdit_6.setText(str(v))

    @property
    def time_window(self):
        return float(self.lineEdit_8.text())

    @time_window.setter
    def time_window(self, v):
        self.lineEdit_8.setText(str(v))

    @property
    def cr_filter_on(self):
        return bool(self.groupBox_2.isChecked())

    @property
    def photons(self):
        photons = self.photon_source.photons
        if self.cr_filter_on:
            dt = photons.mt_clk
            tw = int(self.time_window / dt)
            n_ph_max = int(self.max_count_rate * self.time_window)
            if self.verbose:
                print("Using count-rate filter:")
                print("Window-size [ms]: %s" % self.time_window)
                print("max_count_rate [kHz]: %s" % self.max_count_rate)
                print("n_ph_max in window [#]: %s" % n_ph_max)
                print("Window-size [n(MTCLK)]: %s" % tw)
                print("---------------------------------")

            mt = photons.mt
            n_ph = mt.shape[0]
            w = np.ones(n_ph, dtype=np.float32)
            chisurf.fluorescence.fcs.correlate.count_rate_filter(
                mt,
                tw,
                n_ph_max,
                w,
                n_ph
            )
            photons.cr_filter = w
            return photons
        else:
            return photons


class FCStttr(QtWidgets.QWidget):

    name = 'fcs-tttr'

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.parent = parent
        self.layout = layout
        self.fileWidget = SpcFileWidget(self)
        self.countrateFilterWidget = CrFilterWidget(self, self.fileWidget)
        self.correlator = CorrelatorWidget(self, self.countrateFilterWidget)
        self.layout.addWidget(self.fileWidget)
        self.layout.addWidget(self.countrateFilterWidget)
        self.layout.addWidget(self.correlator)

    def load_data(self, **kwargs):
        d = self.correlator.data
        d.setup = self
        d.name = self.fileWidget.sample_name
        return deepcopy(d)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = CorrelateTTTR()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    win.show()
    sys.exit(app.exec_())
