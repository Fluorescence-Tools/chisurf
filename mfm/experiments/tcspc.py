from PyQt5 import QtCore, QtGui, QtWidgets, uic
import numpy as np
import os

import mfm
from mfm.experiments import Reader
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
        #self.g_factor = kwargs.get('g_factor', 1.0)
        self.rebin = kwargs.get('rebin', (1, 1))
        self.matrix_columns = kwargs.get('matrix_columns', [0, 1])


class CsvTCSPCWidget(CsvTCSPC, QtWidgets.QWidget):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        self.parent = kwargs.get('parent', None)
        uic.loadUi('mfm/ui/experiments/csvTCSPCWidget.ui', self)
        self.actionDtChanged.triggered.connect(self.onDtChanged)
        self.actionRebinChanged.triggered.connect(self.onRebinChanged)
        self.actionRepratechange.triggered.connect(self.onRepratechange)
        self.actionPolarizationChange.triggered.connect(self.onPolarizationChange)
        self.actionGfactorChanged.triggered.connect(self.onGfactorChanged)
        self.actionIsjordiChanged.triggered.connect(self.onIsjordiChanged)
        self.actionMatrixColumnsChanged.triggered.connect(self.onMatrixColumnsChanged)
        self._matrix_columns = list()

    def onMatrixColumnsChanged(self):
        s = str(self.lineEdit.text())
        matrix_columns = map(int, s.strip().split(' '))
        mfm.run("cs.current_setup.matrix_columns = %s" % matrix_columns)

    def onIsjordiChanged(self):
        is_jordi = bool(self.checkBox_3.isChecked())
        mfm.run("cs.current_setup.is_jordi = %s" % is_jordi)
        mfm.run("cs.current_setup.use_header = %s" % (not is_jordi))

    def onGfactorChanged(self):
        gfactor = float(self.doubleSpinBox_3.value())
        mfm.run("cs.current_setup.g_factor = %f" % gfactor)

    def onPolarizationChange(self):
        pol = 'vm'
        if self.radioButton_4.isChecked():
            pol = 'vv/vh'
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
        rebin_y = int(self.comboBox.currentText())
        rebin_x = int(self.comboBox_2.currentText())
        mfm.run("cs.current_setup.rebin = (%s, %s)" % (rebin_x, rebin_y))
        self.onDtChanged()

    def onDtChanged(self):
        rebin = int(self.comboBox.currentText())
        dt = float(self.doubleSpinBox_2.value()) * rebin if self.checkBox_2.isChecked() else 1.0 * rebin
        mfm.run("cs.current_setup.dt = %s" % dt)


class TCSPCReader(Reader):

    def __init__(self, *args, **kwargs):
        super(TCSPCReader, self).__init__(self, *args, **kwargs)
        self.csvSetup = kwargs.pop('csvSetup', Csv(*args, **kwargs))
        #self.skiprows = kwargs.pop('skiprows', 7)
        #self.dt = kwargs.get('dt', mfm.cs_settings['tcspc']['dt'])
        #self.rep_rate = kwargs.get('rep_rate', mfm.cs_settings['tcspc']['rep_rate'])
        #self.g_factor = kwargs.get('g_factor', mfm.cs_settings['tcspc']['g_factor'])
        #self.polarization = 'vm'
        #self.rebin = kwargs.get('rebin', (1, 1))
        #self.is_jordi = kwargs.get('is_jordi', False)
        self.matrix_columns = kwargs.get('matrix_columns', None)
        #self.use_header = True

    @staticmethod
    def autofitrange(data, **kwargs):
        area = kwargs.get('area', mfm.cs_settings['tcspc']['fit_area'])
        threshold = kwargs.get('threshold', mfm.cs_settings['tcspc']['fit_count_threshold'])
        return fitrange(data.y, threshold, area)

    def read(self, *args, **kwargs):
        filename = kwargs.pop('filename', None)
        skiprows = kwargs.get('skiprows', self.skiprows)

        # Load data
        rebin_x, rebin_y = self.rebin
        dt = self.dt
        mc = self.matrix_columns

        self.csvSetup.load(filename, skiprows=skiprows, use_header=self.use_header, usecols=mc)
        data = self.csvSetup.data

        if self.is_jordi:
            if data.ndim == 1:
                data = data.reshape(1, len(data))
            n_data_sets, n_vv_vh = data.shape
            n_data_points = n_vv_vh / 2
            c1, c2 = data[:, :n_data_points], data[:, n_data_points:]

            new_channels = int(n_data_points / rebin_y)
            c1 = c1.reshape([n_data_sets, new_channels, rebin_y]).sum(axis=2)
            c2 = c2.reshape([n_data_sets, new_channels, rebin_y]).sum(axis=2)
            n_data_points = c1.shape[1]

            if self.polarization == 'vv':
                y = c1
                ey = weights(c1)
            elif self.polarization == 'vh':
                y = c2
                ey = weights(c2)
            elif self.polarization == 'vv/vh':
                e1, e2 = weights(c1), weights(c2)
                y = np.vstack([c1, c2])
                ey = np.vstack([e1, e2])
            else:
                f2 = 2.0 * self.g_factor
                y = c1 + f2 * c2
                ey = weights_ps(c1, c2, f2)
            x = np.arange(n_data_points, dtype=np.float64) * dt
        else:
            x = data[0] * dt
            y = data[1:]

            n_datasets, n_data_points = y.shape
            n_data_points = int(n_data_points / rebin_y)
            y = y.reshape([n_datasets, n_data_points, rebin_y]).sum(axis=2)
            ey = weights(y)
            x = np.average(x.reshape([n_data_points, rebin_y]), axis=1) / rebin_y

        # TODO: in future adaptive binning of time axis
        #from scipy.stats import binned_statistic
        #dt = xn[1]-xn[0]
        #xb = np.logspace(np.log10(dt), np.log10(np.max(xn)), 512)
        #tmp = binned_statistic(xn, yn, statistic='sum', bins=xb)
        #xn = xb[:-1]
        #print xn
        #yn = tmp[0]
        #print tmp[1].shape
        x = x[n_data_points % rebin_y:]
        y = y[n_data_points % rebin_y:]

        # rebin along x-axis
        y_rebin = np.zeros_like(y)
        ib = 0
        for ix in range(0, y.shape[0], rebin_x):
            y_rebin[ib] += y[ix:ix+rebin_x, :].sum(axis=0)
            ib += 1
        y_rebin = y_rebin[:ib, :]
        ex = np.zeros(x.shape)
        data_curves = list()
        n_data_sets = y_rebin.shape[0]
        fn = self.csvSetup.filename
        for i, yi in enumerate(y_rebin):
            eyi = ey[i]
            if n_data_sets > 1:
                name = '{} {:d}_{:d}'.format(fn, i, n_data_sets)
            else:
                name = filename
            data = mfm.curve.DataCurve(x=x, y=yi, ex=ex, ey=1./eyi, setup=self, name=name, **kwargs)
            data.filename = filename
            data_curves.append(data)
        data_group = mfm.curve.DataCurveGroup(data_curves, filename)
        return data_group


class TCSPCSetupWidget(TCSPCReader, mfm.io.ascii.Csv, QtWidgets.QWidget):

    # @property
    # def skiprows(self):
    #     return self.csvSetup.skiprows
    #
    # @skiprows.setter
    # def skiprows(self, v):
    #     self.csvSetup.skiprows = v

    def read(self, *args, **kwargs):
        filename = kwargs.pop('filename', None)
        if filename is None:
            filename = mfm.widgets.get_filename(description='CSV-TCSPC file', file_type='All files (*.*)', working_path=None)
        kwargs['filename'] = filename
        return TCSPCReader.read(self, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super(TCSPCSetupWidget, self).__init__()
        QtWidgets.QWidget.__init__(self)
        #TCSPCReader.__init__(self, *args, **kwargs)

        csvSetup = CsvWidget(parent=self)
        csvTCSPC = CsvTCSPCWidget(**kwargs)
        csvSetup.widget.hide()

        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        #self.layout.setMargin(0)
        #self.layout.setSpacing(0)
        self.layout.addWidget(csvTCSPC)
        self.layout.addWidget(csvSetup)

        TCSPCReader.__init__(self, *args, **kwargs)
        # Overwrite non-widget attributes by widgets
        self.csvTCSPC = csvTCSPC


class TcspcSDTWidget(QtWidgets.QWidget):

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
        d = mfm.curve.DataCurve(setup=self, x=self.times, y=y, ey=1./w, name=self.name)
        return d

    def onOpenFile(self, **kwargs):

        fn = kwargs.get('filename', None)
        if fn is None:
            self.filename = mfm.widgets.get_filename('Open BH-SDT file', 'SDT-files (*.sdt)')
        else:
            self.filename = fn

        self.curve_number = kwargs.get('curve_nbr', 0)
        self.textBrowser.setPlainText(str(self.sdt.info))

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('mfm/ui/experiments/sdtfile.ui', self)
        self._sdt = None
        self.rep_rate = kwargs.get('rep_rate', 16.0)
        self.actionOpen_SDT_file.triggered.connect(self.onOpenFile)


class TCSPCSetupSDTWidget(TCSPCReader, QtWidgets.QWidget):

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

    def __str__(self):
        return "Dummy TCSPC"

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self, **kwargs)
        TCSPCReader.__init__(self, **kwargs)
        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        #self.layout.setMargin(0)
        #self.layout.setSpacing(0)

        self.tcspcSDT = TcspcSDTWidget()
        #self.tcspcSDT.widget.hide()

        self.layout.addWidget(self.tcspcSDT)
        self.name = kwargs.get('name', 'Becker SDT')

        #self.connect(self.tcspcSDT.actionAdd_curve, QtCore.SIGNAL('triggered()'), )

    def read(self, *args, **kwargs):
        curves = list()
        self.tcspcSDT.onOpenFile(**kwargs)
        for curve_nbr in range(self.tcspcSDT.n_curves):
            self.tcspcSDT.curve_number = curve_nbr
            curves.append(self.tcspcSDT.curve)
        return curves


class TCSPCSetupDummy(TCSPCReader):

    name = "Dummy-TCSPC"

    def __init__(self, **kwargs):
        TCSPCReader.__init__(self, **kwargs)
        self.parent = kwargs.get('parent', None)
        self.sample_name = kwargs.get('sample_name', 'Dummy-sample')
        self.name = kwargs.get('name', "Dummy")
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.lifetime = kwargs.get('lifetime', 4.1)
        self.n_TAC = kwargs.get('n_TAC', 4096)
        self.dt = kwargs.get('dt', 0.0141)
        self.p0 = kwargs.get('p0', 10000.0)
        self.rep_rate = 10.0  # TODO - dirty!!!

    def read(self, filename=None, **kwargs):
        x = np.arange(self.n_TAC) * self.dt
        y = np.exp(-x/self.lifetime) * self.p0
        ey = 1./weights(y)

        d = mfm.curve.DataCurve(x=x, y=y, ey=ey, setup=self, name="TCSPC-Dummy")
        d.setup = self

        return d

    def __str__(self):
        s = 'TCSPCSetup: Dummy\n'
        return s


class TCSPCSetupDummyWidget(QtWidgets.QWidget, TCSPCSetupDummy):

    @property
    def sample_name(self):
        name = str(self.lineEdit.text())
        return name

    @sample_name.setter
    def sample_name(self, v):
        pass

    @property
    def p0(self):
        return self.spinBox_2.value()

    @p0.setter
    def p0(self, v):
        pass

    @property
    def lifetime(self):
        return self.doubleSpinBox_2.value()

    @lifetime.setter
    def lifetime(self, v):
        pass

    @property
    def n_TAC(self):
        return self.spinBox.value()

    @n_TAC.setter
    def n_TAC(self, v):
        pass

    @property
    def dt(self):
        return self.doubleSpinBox.value()

    @dt.setter
    def dt(self, v):
        pass

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        TCSPCSetupDummy.__init__(self, **kwargs)
        uic.loadUi('mfm/ui/experiments/tcspcDummy.ui', self)

