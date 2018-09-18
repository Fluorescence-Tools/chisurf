from copy import deepcopy
import copy

from PyQt4 import QtCore, QtGui, uic
import numpy as np
import pyqtgraph as pg

import mfm
from mfm.experiments import settings
from mfm.io.widgets import SpcFileWidget

plot_settings = mfm.settings['gui']['plot']
pyqtgraph_settings = plot_settings["pyqtgraph"]
color_scheme = mfm.colors
lw = plot_settings['line_width']


class CorrelateTTTR(QtGui.QWidget):

    name = "tttr-correlate"

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
        plot.setLogMode(x=True, y=False)
        plot.showGrid(True, True, 1.0)

        current_curve = self.cs.selected_curve_index
        for i, curve in enumerate(self._curves):
            l = lw * 0.5 if i != current_curve else 1.5 * lw
            plot.plot(x=curve.x, y=curve.y,
                      pen=pg.mkPen(color_scheme[i % len(color_scheme)]['hex'], width=l),
                      name=curve.name)

    def add_curve(self):
        d = self.corr.correlator.data
        d.setup = self.corr
        d.name = self.corr.fileWidget.sample_name
        self._curves.append(copy.deepcopy(d))
        self.cs.update()
        self.plot_curves()

    def __init__(self):
        QtGui.QWidget.__init__(self)
        self._curves = list()
        uic.loadUi('./mfm/ui/tools/tttr_correlate.ui', self)
        w = mfm.tools.tttr.correlate.FCStttr()
        self.corr = w
        self.verticalLayout.addWidget(w)
        w.show()

        self.cs = mfm.widgets.CurveSelector(get_data_curves=self.get_data_curves, click_close=False)
        self.verticalLayout_6.addWidget(self.cs)

        self.connect(w.correlator.pushButton_3, QtCore.SIGNAL('clicked()'), w.correlator.correlator_thread.start)
        self.connect(w.correlator.correlator_thread, QtCore.SIGNAL("finished()"), self.add_curve)
        self.connect(self.cs, QtCore.SIGNAL('itemClicked(QListWidgetItem *)'), self.plot_curves)

        self.plot = pg.PlotWidget(useOpenGL=pyqtgraph_settings['useOpenGL'])
        plot = self.plot.getPlotItem()
        self.verticalLayout_9.addWidget(self.plot)
        self.legend = plot.addLegend()
        self.cs.onRemoveDataset = self.onRemoveDataset


class Correlator(QtCore.QThread):

    procDone = QtCore.pyqtSignal(bool)
    partDone = QtCore.pyqtSignal(int)

    @property
    def data(self):
        if isinstance(self._data, mfm.curve.DataCurve):
            return self._data
        else:
            return mfm.curve.DataCurve(setup=self)

    def __init__(self, parent):
        QtCore.QThread.__init__(self, parent)
        self.p = parent
        self.exiting = False
        self._data = None
        self._results = []
        self._dt1 = 0
        self._dt2 = 0

    def getWeightStream(self, tacWeighting):
        """
        :param tacWeighting: is either a list of integers or a numpy-array. If it's a list of integers\
        the integers correspond to channel-numbers. In this case all photons have an equal weight of one.\
        If tacWeighting is a numpy-array it should be of shape [max-routing, number of TAC channels]. The\
        array contains np.floats with weights for photons arriving at different TAC-times.
        :return: numpy-array with same length as photon-stream, each photon is associated to one weight.
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
        w = mfm.fluorescence.fcs.get_weights(photons.rout, photons.tac, wt, photons.nPh)
        return w

    def run(self):
        data = mfm.curve.DataCurve()

        w1 = self.getWeightStream(self.p.ch1)
        w2 = self.getWeightStream(self.p.ch2)
        print("Correlation running...")
        print("Correlation method: %s" % self.p.method)
        print("Fine-correlation: %s" % self.p.fine)
        print("Nbr. of correlations: %s" % self.p.split)
        photons = self.p.photon_source.photons

        self._results = []
        n = len(photons)
        nGroup = n / self.p.split
        self.partDone.emit(0.0)
        for i in range(0, n - n % nGroup, nGroup):
            nbr = ((i + 1) / nGroup + 1)
            print("Correlation Nbr.: %s" % nbr)
            p = photons[i:i + nGroup]
            wi1, wi2 = w1[i:i + nGroup], w2[i:i + nGroup]
            if self.p.method == 'tp':
                np1, np2, dt1, dt2, tau, corr = mfm.fluorescence.fcs.log_corr(p.mt, p.tac, p.rout, p.cr_filter,
                                                          wi1, wi2, self.p.B, self.p.nCasc,
                                                          self.p.fine, photons.n_tac)
                cr = mfm.fluorescence.fcs.normalize(np1, np2, dt1, dt2, tau, corr, self.p.B)
                cr /= self.p.dt
                dur = float(min(dt1, dt2)) * self.p.dt / 1000  # seconds
                tau = tau.astype(np.float64)
                tau *= self.p.dt
                self._results.append([cr, dur, tau, corr])
            self.partDone.emit(float(nbr) / self.p.split * 100)

        # Calculate average correlations
        cors = []
        taus = []
        weights = []
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
            return mfm.fluorescence.fcs.weights(tau, cor, dur, cr, type='uniform')
        elif self.p.weighting == 0:
            return mfm.fluorescence.fcs.weights(tau, cor, dur, cr, type='suren')


class CorrelatorWidget(QtGui.QWidget):

    def __init__(self, parent, photon_source, ch1='0', ch2='8', setup=None, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/experiments/correlatorWidget.ui', self)

        self.nCasc = kwargs.get('nCasc', settings['nCasc'])
        self.B = kwargs.get('B', settings['B'])
        self.split = kwargs.get('split', settings['split'])
        self.weighting = kwargs.get('weighting', settings['weighting'])
        self.fine = kwargs.get('fine', settings['fine'])

        self.setup = setup
        self.parent = parent
        self.cr = 0.0
        self.ch1 = ch1
        self.ch2 = ch2

        self.photon_source = photon_source
        self.correlator_thread = Correlator(self)

        # fill widgets
        self.comboBox_3.addItems(mfm.fluorescence.fcs.weightCalculations)
        self.comboBox_2.addItems(mfm.fluorescence.fcs.correlationMethods)
        self.checkBox.setChecked(True)
        self.checkBox.setChecked(False)
        self.progressBar.setValue(0.0)

        # connect widgets
        self.connect(self.pushButton_3, QtCore.SIGNAL("clicked()"), self.correlator_thread.start)
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


class CrFilterWidget(QtGui.QWidget):

    def __init__(self, parent, photon_source, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/experiments/fcs-cr-filter.ui', self)
        self.photon_source = photon_source
        self.parent = parent

        self.verbose = kwargs.get('verbose', mfm.verbose)
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
            mfm.fluorescence.fcs.count_rate_filter(mt, tw, n_ph_max, w, n_ph)
            photons.cr_filter = w
            return photons
        else:
            return photons


class FCStttr(QtGui.QWidget):

    name = 'FCS-tttr'

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        layout = QtGui.QVBoxLayout(self)
        self.parent = parent
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
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