from __future__ import annotations
from typing import Tuple

import numpy as np
from qtpy import  QtWidgets, uic
import os

import mfm
import mfm.experiments
import mfm.experiments.data
import mfm.widgets
from mfm.experiments.reader import ExperimentReader
from mfm.fluorescence.tcspc import weights, fitrange, weights_ps
from mfm.io import sdtfile
from mfm.io.ascii import Csv
from mfm.io.widgets import CsvWidget


class CsvTCSPC(object):

    def __init__(
            self,
            *args,
            dt: float = 1.0,
            rep_rate: float = 1.0,
            is_jordi: bool = False,
            mode: str = 'vm',
            g_factor: float = 1.0,
            rebin: Tuple[int, int] = (1, 1),
            matrix_columns: Tuple[int, int] = (0, 1),
            **kwargs
    ):
        super(CsvTCSPC, self).__init__(*args, **kwargs)
        self.dt = dt
        self.excitation_repetition_rate = rep_rate
        self.is_jordi = is_jordi
        self.polarization = mode
        self.g_factor = g_factor
        self.rebin = rebin
        self.matrix_columns = matrix_columns


class CsvTCSPCWidget(CsvTCSPC, QtWidgets.QWidget):

    def __init__(self, **kwargs):
        QtWidgets.QWidget.__init__(self)
        self.parent = kwargs.get('parent', None)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "csvTCSPCWidget.ui"
            ),
            self
        )

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


class TCSPCReader(ExperimentReader):

    def __init__(
            self,
            *args,
            skiprows: int = 7,
            is_jordi: bool = False,
            rebin: Tuple[int, int] = (1, 1),
            use_header: bool = True,
            polarization: str = 'vm',
            **kwargs
    ):
        super(TCSPCReader, self).__init__(self, *args, **kwargs)
        self.csvSetup = kwargs.pop('csvSetup', Csv(*args, **kwargs))
        self.skiprows = skiprows
        self.is_jordi = is_jordi
        self.rebin = rebin
        self.use_header = use_header
        self.polarization = polarization
        self.dt = kwargs.get('dt', mfm.settings.cs_settings['tcspc']['dt'])
        self.rep_rate = kwargs.get('rep_rate', mfm.settings.cs_settings['tcspc']['rep_rate'])
        self.g_factor = kwargs.get('g_factor', mfm.settings.cs_settings['tcspc']['g_factor'])
        self.matrix_columns = kwargs.get('matrix_columns', None)

    @staticmethod
    def autofitrange(
            data,
            **kwargs
    ) -> Tuple[float, float]:
        area = kwargs.get('area', mfm.settings.cs_settings['tcspc']['fit_area'])
        threshold = kwargs.get('threshold', mfm.settings.cs_settings['tcspc']['fit_count_threshold'])
        return fitrange(data.y, threshold, area)

    def read(
            self,
            *args,
            filename: str = None,
            **kwargs
    ) -> mfm.experiments.data.DataCurveGroup:
        filename = filename
        skiprows = kwargs.get('skiprows', self.skiprows)

        # Load data
        rebin_x, rebin_y = self.rebin
        dt = self.dt
        mc = self.matrix_columns

        self.csvSetup.load(
            filename,
            skiprows=skiprows,
            use_header=self.use_header,
            usecols=mc
        )
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
            data = mfm.experiments.data.DataCurve(
                x=x,
                y=yi,
                ex=ex,
                ey=1. / eyi,
                setup=self,
                name=name,
                **kwargs
            )
            data.filename = filename
            data_curves.append(data)
        data_group = mfm.experiments.data.DataCurveGroup(data_curves, filename)
        return data_group


class TCSPCSetupWidget(TCSPCReader, mfm.io.ascii.Csv, QtWidgets.QWidget):

    def read(
            self,
            *args,
            filename: str = None,
            **kwargs
    ) -> mfm.experiments.data.DataCurveGroup:
        if filename is None:
            filename = mfm.widgets.get_filename(
                description='CSV-TCSPC file',
                file_type='All files (*.*)',
                working_path=None
            )
        if os.path.isfile(filename):
            kwargs['filename'] = filename
            return TCSPCReader.read(self, *args, **kwargs)
        else:
            return None

    def __init__(self, *args, **kwargs):
        super(TCSPCSetupWidget, self).__init__()
        QtWidgets.QWidget.__init__(self)
        #TCSPCReader.__init__(self, *args, **kwargs)

        csvSetup = CsvWidget(parent=self)
        csvTCSPC = CsvTCSPCWidget(**kwargs)
        csvSetup.widget.hide()

        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        self.layout.addWidget(csvTCSPC)
        self.layout.addWidget(csvSetup)

        TCSPCReader.__init__(self, *args, **kwargs)
        # Overwrite non-widget attributes by widgets
        self.csvTCSPC = csvTCSPC


class TcspcSDTWidget(QtWidgets.QWidget):

    @property
    def name(self) -> str:
        return self.filename + " _ " + str(self.curve_number)

    @property
    def n_curves(self) -> int:
        n_data_curves = len(self._sdt.data)
        return n_data_curves

    @property
    def curve_number(self) -> int:
        """
        The number of the currently selected curve
        """
        return int(self.comboBox.currentIndex())

    @curve_number.setter
    def curve_number(
            self,
            v: int
    ):
        self.comboBox.setCurrentIndex(int(v))

    @property
    def filename(self) -> str:
        return str(self.lineEdit.text())

    @filename.setter
    def filename(
            self,
            v: str
    ):
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
    def times(self) -> np.array:
        """
        The time-array in nano-seconds
        """
        x = self._sdt.times[0] * 1e9
        return np.array(x, dtype=np.float64)

    @property
    def ph_counts(self) -> np.array:
        y = self._sdt.data[self.curve_number][0]
        return np.array(y, dtype=np.float64)

    @property
    def rep_rate(self) -> float:
        return 1. / (self._sdt.measure_info[self.curve_number]['rep_t'] * 1e-3)[0]

    @rep_rate.setter
    def rep_rate(self, v):
        pass

    @property
    def curve(self) -> mfm.experiments.data.DataCurve:
        y = self.ph_counts
        w = weights(y)
        d = mfm.experiments.data.DataCurve(setup=self, x=self.times, y=y, ey=1. / w, name=self.name)
        return d

    def onOpenFile(self, **kwargs):

        fn = kwargs.get('filename', None)
        if fn is None:
            filename = mfm.widgets.get_filename('Open BH-SDT file', 'SDT-files (*.sdt)')
            self.filename = filename
            print(self.filename)
        else:
            self.filename = fn

        self.curve_number = kwargs.get('curve_nbr', 0)
        self.textBrowser.setPlainText(str(self.sdt.info))

    def __init__(self, **kwargs):
        super(TcspcSDTWidget, self).__init__()
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "sdtfile.ui"
            ),
            self
        )
        self._sdt = None
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

    def __init__(
            self,
            *args,
            name: str = 'Becker SDT',
            **kwargs
    ):
        super(TCSPCSetupSDTWidget, self).__init__(*args, **kwargs)

        layout = QtWidgets.QVBoxLayout(self)
        self.layout = layout
        self.tcspcSDT = TcspcSDTWidget()
        self.layout.addWidget(self.tcspcSDT)
        self.name = name
        #self.connect(self.tcspcSDT.actionAdd_curve, QtCore.SIGNAL('triggered()'), )

    def read(
            self,
            *args,
            **kwargs
    ):
        curves = list()
        self.tcspcSDT.onOpenFile(**kwargs)
        for curve_nbr in range(self.tcspcSDT.n_curves):
            self.tcspcSDT.curve_number = curve_nbr
            curves.append(self.tcspcSDT.curve)
        return curves


class TCSPCSetupDummy(TCSPCReader):

    name = "Dummy-TCSPC"

    def __init__(
            self,
            *args,
            n_tac: int = 4096,
            dt: float = 0.0141,
            p0: float = 10000.0,
            rep_rate: float = 10.0,
            lifetime: float = 4.1,
            name: str = 'Dummy',
            sample_name: str = 'TCSPC-Dummy',
            parent: QtWidgets.QWidget = None,
            verbose: bool = None,
            **kwargs
    ):
        super(TCSPCSetupDummy, self).__init__(*args, **kwargs)
        TCSPCReader.__init__(self, **kwargs)
        self.parent = parent

        if verbose is None:
            verbose = mfm.verbose
        self.verbose = verbose

        self.sample_name = sample_name
        self.name = name
        self.lifetime = lifetime
        self.n_tac = n_tac
        self.dt = dt
        self.p0 = p0
        self.rep_rate = rep_rate

    def read(
            self,
            filename: str = None,
            **kwargs
    ):
        if filename is None:
            filename = self.sample_name

        x = np.arange(self.n_tac) * self.dt
        y = np.exp(-x/self.lifetime) * self.p0
        ey = 1./weights(y)

        d = mfm.experiments.data.DataCurve(
            x=x,
            y=y,
            ey=ey,
            setup=self,
            name=filename
        )
        d.setup = self

        return d

    def __str__(self):
        s = 'TCSPCSetup: Dummy\n'
        return s


class TCSPCSetupDummyWidget(QtWidgets.QWidget, TCSPCSetupDummy):

    @property
    def sample_name(
            self
    ) -> str:
        name = str(self.lineEdit.text())
        return name

    @sample_name.setter
    def sample_name(
            self,
            v: str
    ):
        pass

    @property
    def p0(
            self
    ) -> int:
        return self.spinBox_2.value()

    @p0.setter
    def p0(
            self,
            v: int
    ):
        pass

    @property
    def lifetime(
            self
    ) -> float:
        return self.doubleSpinBox_2.value()

    @lifetime.setter
    def lifetime(
            self,
            v: float
    ):
        pass

    @property
    def n_tac(
            self
    ) -> int:
        return self.spinBox.value()

    @n_tac.setter
    def n_tac(
            self,
            v: int
    ):
        pass

    @property
    def dt(
            self
    ) -> float:
        return self.doubleSpinBox.value()

    @dt.setter
    def dt(
            self,
            v: float
    ):
        pass

    def __init__(self, **kwargs):
        super(TCSPCSetupDummyWidget, self).__init__(**kwargs)
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "tcspcDummy.ui"
            ),
            self
        )


