from PyQt4 import QtGui, QtCore, uic
import numpy as np

import mfm
from mfm.curve import Base
from mfm.experiments import Setup
from mfm.fluorescence.tcspc import weights, fitrange, weights_ps
from mfm.io.ascii import Csv
from mfm.io import sdtfile
from mfm.io.widgets import CsvWidget


class CsvTCSPC(object):

    def __init__(self, **kwargs):
        self.dt = kwargs.get('dt', 1.0)
        self.rep_rate = kwargs.get('rep_rate', 1.0)
        self.is_jordi = kwargs.get('is_jordi', False)
        self.polarization = kwargs.get('mode', 'vm')
        self.g_factor = kwargs.get('g_factor', 1.0)
        self.rebin = kwargs.get('rebin', 1.0)


class CsvTCSPCWidget(CsvTCSPC, QtGui.QWidget):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        self.parent = kwargs.get('parent', None)

        uic.loadUi('mfm/ui/experiments/csvTCSPCWidget.ui', self)
        self.connect(self.actionDtChanged, QtCore.SIGNAL('triggered()'), self.onDtChanged)
        self.connect(self.actionRebinChanged, QtCore.SIGNAL('triggered()'), self.onRebinChanged)
        self.connect(self.actionRepratechange, QtCore.SIGNAL('triggered()'), self.onRepratechange)
        self.connect(self.actionPolarizationChange, QtCore.SIGNAL('triggered()'), self.onPolarizationChange)
        self.connect(self.actionGfactorChanged, QtCore.SIGNAL('triggered()'), self.onGfactorChanged)
        self.connect(self.actionIsjordiChanged, QtCore.SIGNAL('triggered()'), self.onIsjordiChanged)

    def onIsjordiChanged(self):
        is_jordi = bool(self.checkBox_3.isChecked())
        mfm.run("cs.current_setup.is_jordi = %s" % is_jordi)
        mfm.run("cs.current_setup.use_header = %s" % (not is_jordi))

    def onGfactorChanged(self):
        gfactor = float(self.doubleSpinBox_3.value())
        mfm.run("cs.current_setup.g_factor = %f" % gfactor)

    def onPolarizationChange(self):
        pol = 'vm'
        if self.radioButton_3.isChecked():
            pol = 'vv'
        elif self.radioButton_2.isChecked():
            pol = 'vh'
        elif self.radioButton.isChecked():
            pol = 'vm'
        mfm.run("cs.current_setup.polarization = '%s'" % pol)

    def onRepratechange(self):
        rep_rate = self.doubleSpinBox.value()
        mfm.run("cs.current_setup.rep_rate = %s" % rep_rate)

    def onRebinChanged(self):
        rebin = int(self.comboBox.currentText())
        mfm.run("cs.current_setup.rebin = %s" % rebin)
        self.onDtChanged()

    def onDtChanged(self):
        rebin = int(self.comboBox.currentText())
        dt = float(self.doubleSpinBox_2.value()) * rebin if self.checkBox_2.isChecked() else 1.0 * rebin
        mfm.run("cs.current_setup.dt = %s" % dt)


class TCSPCSetup(Setup):

    def __init__(self, *args, **kwargs):
        Setup.__init__(self, *args, **kwargs)
        self.csvSetup = kwargs.pop('csvSetup', Csv(*args, **kwargs))
        self.skiprows = kwargs.pop('skiprows', 7)
        self.dt = kwargs.get('dt', mfm.settings['tcspc']['dt'])
        self.rep_rate = kwargs.get('rep_rate', mfm.settings['tcspc']['rep_rate'])
        self.g_factor = kwargs.get('g_factor', mfm.settings['tcspc']['g_factor'])
        self.polarization = 'vm'
        self.rebin = kwargs.get('rebin', 1.0)
        self.is_jordi = kwargs.get('is_jordi', False)
        self.use_header = True

    @staticmethod
    def autofitrange(data, **kwargs):
        area = kwargs.get('area', mfm.settings['tcspc']['fit_area'])
        threshold = kwargs.get('threshold', mfm.settings['tcspc']['fit_count_threshold'])
        return fitrange(data.y, threshold, area)

    def load_data(self, *args, **kwargs):
        filename = kwargs.pop('filename', None)
        skiprows = kwargs.get('skiprows', self.skiprows)

        # Load data
        if self.is_jordi:
            self.csvSetup.x_on = False
            self.csvSetup.skiprows = 0

        rebin = self.rebin
        dt = self.dt

        if self.is_jordi:
            self.csvSetup.load(filename, skiprows=0, use_header=False, guess_dialect=False)
            d = self.csvSetup.data[0]
            channels = d.shape[0] / 2
            new_channels = int(channels / rebin)

            c1, c2 = d.reshape([2, channels])
            c1 = c1.reshape([new_channels, rebin]).sum(axis=1)
            c2 = c2.reshape([new_channels, rebin]).sum(axis=1)

            if self.polarization == 'vv':
                y = c1
                ey = weights(c1)
            elif self.polarization == 'vh':
                y = c2
                ey = weights(c2)
            else:
                f2 = 2.0 * self.g_factor
                y = c1 + f2 * c2
                ey = weights_ps(c1, c2, f2)
            x = np.arange(c1.shape[0], dtype=np.float64) * dt
        else:
            self.csvSetup.load(filename, skiprows=skiprows, use_header=self.use_header)
            d = self.csvSetup.data.T
            channels = d.shape[0]
            new_channels = int(channels / rebin)

            x = self.csvSetup.data[0] * dt
            y = self.csvSetup.data[1]
            y = y.reshape([new_channels, rebin]).sum(axis=1)
            ey = weights(y)
            x = np.average(x.reshape([new_channels, rebin]), axis=1) / rebin

        # TODO: in future adaptive binning of time axis
        #from scipy.stats import binned_statistic
        #dt = xn[1]-xn[0]
        #xb = np.logspace(np.log10(dt), np.log10(np.max(xn)), 512)
        #tmp = binned_statistic(xn, yn, statistic='sum', bins=xb)
        #xn = xb[:-1]
        #print xn
        #yn = tmp[0]
        #print tmp[1].shape

        x = x[channels % rebin:]
        y = y[channels % rebin:]
        ex = np.zeros(x.shape)
        d = mfm.curve.DataCurve(x, y, ex, ey, setup=self, **kwargs)
        d.filename = self.csvSetup.filename
        return d

    def __str__(self):
        s = Setup.__str__(self)
        s += 'dt [ns]: %.2f \n' % self.dt
        s += 'repetion rate [MHz]: %.1f \n' % self.rep_rate
        s += 'TAC channels: %s \n' % self.csvSetup.n_points
        return s


class TCSPCSetupWidget(TCSPCSetup, mfm.io.ascii.Csv, QtGui.QWidget):

    # @property
    # def skiprows(self):
    #     return self.csvSetup.skiprows
    #
    # @skiprows.setter
    # def skiprows(self, v):
    #     self.csvSetup.skiprows = v

    def load_data(self, *args, **kwargs):
        filename = kwargs.pop('filename', None)
        if filename is None:
            filename = mfm.widgets.open_file(description='CSV-TCSPC file', file_type='All files (*.*)', working_path=None)
        kwargs['filename'] = filename
        return TCSPCSetup.load_data(self, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self)
        TCSPCSetup.__init__(self, *args, **kwargs)

        csvSetup = CsvWidget(parent=self)
        csvTCSPC = CsvTCSPCWidget(**kwargs)
        csvSetup.widget.hide()

        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.layout.addWidget(csvTCSPC)
        self.layout.addWidget(csvSetup)

        TCSPCSetup.__init__(self, *args, **kwargs)
        # Overwrite non-widget attributes by widgets
        self.csvTCSPC = csvTCSPC


class TcspcSDTWidget(QtGui.QWidget):

    @property
    def name(self):
        return self.filename + " _ " + str(self.curve_number)

    @property
    def n_curves(self):
        n_data_curves = len(self._sdt.data)
        return n_data_curves

    @property
    def curve_number(self):
        """
        The currently selected curves
        """
        return int(self.comboBox.currentIndex())

    @curve_number.setter
    def curve_number(self, v):
        self.comboBox.setCurrentIndex(int(v))

    @property
    def filename(self):
        return str(self.lineEdit.text())

    @filename.setter
    def filename(self, v):
        self._sdt = sdtfile.SdtFile(v)
        # refresh GUI
        self.comboBox.clear()
        l = [str(i) for i in range(self.n_curves)]
        self.comboBox.addItems(l)
        self.lineEdit.setText(str(v))
        self.lineEdit_3.setText(str(self.rep_rate))

    @property
    def sdt(self):
        if self._sdt is None:
            self.onOpenFile()
        return self._sdt

    @property
    def times(self):
        """
        The time-array in nano-seconds
        """
        x = self._sdt.times[0] * 1e9
        return np.array(x, dtype=np.float64)

    @property
    def ph_counts(self):
        y = self._sdt.data[self.curve_number][0]
        return np.array(y, dtype=np.float64)

    @property
    def rep_rate(self):
        return 1. / (self._sdt.measure_info[self.curve_number]['rep_t'] * 1e-3)[0]

    @rep_rate.setter
    def rep_rate(self, v):
        pass

    @property
    def curve(self):
        y = self.ph_counts
        w = weights(y)
        d = mfm.curve.DataCurve(setup=self, x=self.times, y=y, weights=w, name=self.name)
        return d

    def onOpenFile(self, **kwargs):

        fn = kwargs.get('filename', None)
        if fn is None:
            self.filename = mfm.widgets.open_file('Open BH-SDT file', 'SDT-files (*.sdt)')
        else:
            self.filename = fn

        self.curve_number = kwargs.get('curve_nbr', 0)
        self.plainTextEdit.setPlainText(str(self.sdt.info))

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/experiments/sdtfile.ui', self)
        self._sdt = None
        self.rep_rate = kwargs.get('rep_rate', 16.0)
        self.connect(self.actionOpen_SDT_file, QtCore.SIGNAL('triggered()'), self.onOpenFile)


class TCSPCSetupSDTWidget(QtGui.QWidget, TCSPCSetup):

    @property
    def rep_rate(self):
        return self.tcspcSDT.rep_rate

    @rep_rate.setter
    def rep_rate(self, v):
        pass

    @property
    def dt(self):
        dt = self.tcspcSDT.times[1] - self.tcspcSDT.times[0]
        return dt

    @dt.setter
    def dt(self, v):
        pass

    @property
    def curve_nbr(self):
        return self.tcspcSDT.curve_number

    @curve_nbr.setter
    def curve_nbr(self, v):
        self.tcspcSDT.curve_number = v

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self, **kwargs)
        TCSPCSetup.__init__(self, **kwargs)
        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)

        self.tcspcSDT = TcspcSDTWidget()
        #self.tcspcSDT.widget.hide()

        self.layout.addWidget(self.tcspcSDT)
        self.name = kwargs.get('name', 'Becker SDT')

        #self.connect(self.tcspcSDT.actionAdd_curve, QtCore.SIGNAL('triggered()'), )

    def load_data(self, *args, **kwargs):
        curves = list()
        self.tcspcSDT.onOpenFile(**kwargs)
        for curve_nbr in range(self.tcspcSDT.n_curves):
            self.tcspcSDT.curve_number = curve_nbr
            curves.append(self.tcspcSDT.curve)
        return curves


