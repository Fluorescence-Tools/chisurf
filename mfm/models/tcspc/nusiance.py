import numpy as np
from PyQt5 import QtCore, QtWidgets, uic

import mfm
from mfm.curve import Curve
from mfm.fitting.parameter import FittingParameterGroup, FittingParameter
from mfm.widgets import CurveSelector



class Generic(FittingParameterGroup):

    @property
    def n_ph_bg(self):
        """Number of background photons
        """
        if isinstance(self.background_curve, Curve):
            return self._background_curve.y.sum() / self.t_bg * self.t_exp
        else:
            return 0

    @property
    def n_ph_exp(self):
        """Number of experimental photons
        """
        if isinstance(self.fit.data, Curve):
            return self.fit.data.y.sum()
        else:
            return 0

    @property
    def n_ph_fl(self):
        """Number of fluorescence photons
        """
        return self.n_ph_exp - self.n_ph_bg

    @property
    def scatter(self):
        # Scatter amplitude
        return self._sc.value

    @scatter.setter
    def scatter(self, v):
        self._sc.value = v

    @property
    def background(self):
        # Constant background in fluorescence decay curve
        return self._bg.value

    @background.setter
    def background(self, v):
        self._bg.value = v

    @property
    def background_curve(self):
        # Background curve
        if isinstance(self._background_curve, Curve):
            return self._background_curve
        else:
            return None

    @background_curve.setter
    def background_curve(self, v):
        if isinstance(v, Curve):
            self._background_curve = v

    @property
    def t_bg(self):
        """Measurement time of background-measurement
        """
        return self._tmeas_bg.value

    @t_bg.setter
    def t_bg(self, v):
        self._tmeas_bg.value = v

    @property
    def t_exp(self):
        """Measurement time of experiment
        """
        return self._tmeas_exp.value

    @t_exp.setter
    def t_exp(self, v):
        self._tmeas_exp.value = v

    def __init__(self, **kwargs):
        kwargs['name'] = 'Nuisance'
        FittingParameterGroup.__init__(self, **kwargs)

        self._background_curve = None
        self._sc = FittingParameter(value=0.0, name='sc', model=self.model)
        self._bg = FittingParameter(value=0.0, name='bg', model=self.model)
        self._tmeas_bg = FittingParameter(value=1.0, name='tBg', lb=0.001, ub=10000000, fixed=True)
        self._tmeas_exp = FittingParameter(value=1.0, name='tMeas', lb=0.001, ub=10000000, fixed=True)

        self.background_curve = kwargs.get('background_curve', None)


class GenericWidget(Generic, QtWidgets.QWidget):

    def change_bg_curve(self, background_index=None):
        if isinstance(background_index, int):
            self.background_select.selected_curve_index = background_index
        self._background_curve = self.background_select.selected_dataset

        self.lineEdit.setText(self.background_select.curve_name)
        self.fit.model.update()

    def update_widget(self):
        self.lineedit_nphBg.setText("%i" % self.n_ph_bg)
        self.lineedit_nphFl.setText("%i" % self.n_ph_fl)

    def __init__(self, **kwargs):
        Generic.__init__(self, **kwargs)
        QtWidgets.QWidget.__init__(self)

        self.parent = kwargs.get('parent', None)
        if kwargs.get('hide_generic', False):
            self.hide()

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        gb = QtWidgets.QGroupBox()
        gb.setTitle("Generic")
        self.layout.addWidget(gb)

        gbl = QtWidgets.QVBoxLayout()
        gbl.setSpacing(0)
        gbl.setContentsMargins(0, 0, 0, 0)

        gb.setLayout(gbl)
        # Generic parameters
        l = QtWidgets.QGridLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)
        gbl.addLayout(l)

        sc_w = self._sc.make_widget(update_function=self.update_widget, text='Sc')
        bg_w = self._bg.make_widget(update_function=self.update_widget, text='Bg')
        tmeas_bg_w = self._tmeas_bg.make_widget(update_function=self.update_widget, text='t<sub>Bg</sub>')
        tmeas_exp_w = self._tmeas_exp.make_widget(update_function=self.update_widget, text='t<sub>Meas</sub>')

        l.addWidget(sc_w, 1, 0)
        l.addWidget(bg_w, 1, 1)
        l.addWidget(tmeas_bg_w, 2, 0)
        l.addWidget(tmeas_exp_w, 2, 1)

        ly = QtWidgets.QHBoxLayout()
        l.addLayout(ly, 0, 0, 1, 2)
        ly.addWidget(QtWidgets.QLabel('Background file:'))
        self.lineEdit = QtWidgets.QLineEdit()
        ly.addWidget(self.lineEdit)

        open_bg = QtWidgets.QPushButton()
        open_bg.setText('...')
        ly.addWidget(open_bg)

        self.background_select = CurveSelector(parent=None, change_event=self.change_bg_curve, fit=self.fit)
        open_bg.clicked.connect(self.background_select.show)

        a = QtWidgets.QHBoxLayout()
        a.addWidget(QtWidgets.QLabel('nPh(Bg)'))
        self.lineedit_nphBg = QtWidgets.QLineEdit()
        a.addWidget(self.lineedit_nphBg)
        l.addLayout(a, 3, 0, 1, 1)

        a = QtWidgets.QHBoxLayout()
        a.addWidget(QtWidgets.QLabel('nPh(Fl)'))
        self.lineedit_nphFl = QtWidgets.QLineEdit()
        a.addWidget(self.lineedit_nphFl)
        l.addLayout(a, 3, 1, 1, 1)


class Corrections(FittingParameterGroup):

    @property
    def lintable(self):
        if self._lintable is None:
            self._lintable = np.ones_like(self.fit.data.y)
        return self._lintable[::-1] if self.reverse else self._lintable

    @lintable.setter
    def lintable(self, v):
        self._curve = v
        self._lintable = self.calc_lintable(v.y)

    @property
    def window_length(self):
        return int(self._window_length.value)

    @window_length.setter
    def window_length(self, v):
        self._window_length.value = v
        self._lintable = self.calc_lintable(self._curve.y)

    @property
    def window_function(self):
        return self._window_function

    @window_function.setter
    def window_function(self, v):
        self._window_function = v
        self._lintable = self.calc_lintable(self._curve.y)

    @property
    def reverse(self):
        return self._reverse

    @reverse.setter
    def reverse(self, v):
        self._reverse = v

    def calc_lintable(self, y, **kwargs):
        window_function = kwargs.get('window_function', self.window_function)
        window_length = kwargs.get('window_length', self.window_length)
        xmin = kwargs.get('xmin', self.fit.xmin)
        xmax = kwargs.get('xmax', self.fit.xmax)
        return mfm.fluorescence.tcspc.dnl_table(y, window_length, window_function, xmin, xmax)

    @property
    def measurement_time(self):
        return self.fit.model.generic.t_exp

    @measurement_time.setter
    def measurement_time(self, v):
        self.fit.model.generic.t_exp = v

    @property
    def rep_rate(self):
        return self.fit.model.convolve.rep_rate

    @rep_rate.setter
    def rep_rate(self, v):
        self.fit.model.convolve.rep_rate = v

    @property
    def dead_time(self):
        return self._dead_time.value

    @dead_time.setter
    def dead_time(self, v):
        self._dead_time.value = v

    def pileup(self, decay, **kwargs):
        data = kwargs.get('data', self.fit.data.y)
        rep_rate = kwargs.get('rep_rate', self.rep_rate)
        dead_time = kwargs.get('dead_time', self.dead_time)
        meas_time = kwargs.get('meas_time', self.measurement_time)
        if self.correct_pile_up:
            mfm.fluorescence.tcspc.pile_up(data, decay, rep_rate, dead_time, meas_time, verbose=self.verbose)

    def linearize(self, decay, **kwargs):
        lintable = kwargs.get('lintable', self.lintable)
        if lintable is not None and self.correct_dnl:
            return decay * lintable
        return decay

    def __init__(self, fit, **kwargs):
        kwargs['fit'] = fit
        kwargs['name'] = 'Corrections'
        FittingParameterGroup.__init__(self, **kwargs)

        self._lintable = None
        self._curve = None
        self._reverse = kwargs.get('reverse', False)
        self.correct_dnl = kwargs.get('correct_dnl', False)
        self._window_length = FittingParameter(value=17.0, name='win-size', fixed=True, decimals=0)
        self._window_function = kwargs.get('window_function', 'hanning')
        self.correct_pile_up = kwargs.get('correct_pile_up', False)

        self._auto_range = kwargs.get('lin_auto_range', True)
        self._dead_time = FittingParameter(value=85.0, name='tDead', fixed=True, decimals=1)


class CorrectionsWidget(Corrections, QtWidgets.QWidget):

    def __init__(self, fit, **kwargs):
        super(CorrectionsWidget, self).__init__(fit=fit, threshold=0.9, reverse=False, enabled=False)
        QtWidgets.QWidget.__init__(self)
        uic.loadUi("mfm/ui/fitting/models/tcspc/tcspcCorrections.ui", self)
        self.groupBox.setChecked(False)
        self.comboBox.addItems(mfm.math.signal.windowTypes)
        if kwargs.get('hide_corrections', False):
            self.hide()

        self._dead_time.make_widget(layout=self.horizontalLayout_2, text='t<sub>dead</sub>[ns]')
        self._window_length.make_widget(layout=self.horizontalLayout_2)

        self.lin_select = CurveSelector(parent=None,
                                        change_event=self.onChangeLin,
                                        fit=self.fit,
                                        setup=mfm.experiments.tcspc.TCSPCReader)

        self.actionSelect_lintable.triggered.connect(self.lin_select.show)

        self.checkBox_3.toggled.connect(lambda: mfm.run(
            "cs.current_fit.models.corrections.correct_pile_up = %s\n" % self.checkBox_3.isChecked())
        )

        self.checkBox_2.toggled.connect(lambda: mfm.run(
            "cs.current_fit.models.corrections.reverse = %s" % self.checkBox_2.isChecked())
        )

        self.checkBox.toggled.connect(lambda: mfm.run(
            "cs.current_fit.models.corrections.correct_dnl = %s" % self.checkBox.isChecked())
        )

        self.comboBox.currentIndexChanged.connect(lambda: mfm.run(
            "cs.current_fit.models.corrections.window_function = '%s'" % self.comboBox.currentText())
        )

    def onChangeLin(self):
        idx = self.lin_select.selected_curve_index
        t = "lin_table = cs.current_fit.models.corrections.lin_select.datasets[%s]\n" \
            "for f in cs.current_fit[cs.current_fit._selected_fit:]:\n" \
            "   f.models.corrections.lintable = mfm.curve.DataCurve(x=lin_table.x, y=lin_table.y)\n" \
            "   f.models.corrections.correct_dnl = True\n" % idx
        mfm.run(t)

        lin_name = self.lin_select.curve_name
        t = "for f in cs.current_fit[cs.current_fit._selected_fit:]:\n" \
            "   f.models.corrections.lineEdit.setText('%s')\n" \
            "   f.models.corrections.checkBox.setChecked(True)\n" % lin_name
        mfm.run(t)

        mfm.run("cs.current_fit.update()")


class Convolve(FittingParameterGroup):

    @property
    def dt(self):
        return self._dt.value

    @dt.setter
    def dt(self, v):
        self._dt.value = v

    @property
    def lamp_background(self):
        return self._lb.value / self.n_photons_irf

    @lamp_background.setter
    def lamp_background(self, v):
        self._lb.value = v

    @property
    def timeshift(self):
        return self._ts.value

    @timeshift.setter
    def timeshift(self, v):
        self._ts.value = v

    @property
    def start(self):
        return int(self._start.value / self.dt)

    @start.setter
    def start(self, v):
        self._start.value = v

    @property
    def stop(self):
        stop = int(self._stop.value / self.dt)
        return stop

    @stop.setter
    def stop(self, v):
        self._stop.value = v

    @property
    def rep_rate(self):
        return self._rep.value

    @rep_rate.setter
    def rep_rate(self, v):
        self._rep.value = float(v)

    @property
    def do_convolution(self):
        return self._do_convolution

    @do_convolution.setter
    def do_convolution(self, v):
        self._do_convolution = bool(v)

    @property
    def n0(self):
        return self._n0.value

    @n0.setter
    def n0(self, v):
        self._n0.value = v

    @property
    def irf(self):
        irf = self._irf
        if isinstance(irf, Curve):
            irf = self._irf
            irf = (irf - self.lamp_background) << self.timeshift
            irf.y = np.maximum(irf.y, 0)
            return irf
        else:
            x = np.copy(self.data.x)
            y = np.zeros_like(self.data.y)
            y[0] = 1.0
            curve = mfm.curve.Curve(x=x, y=y)
            return curve

    @property
    def _irf(self):
        return self.__irf

    @_irf.setter
    def _irf(self, v):
        self.n_photons_irf = v.norm(mode="sum")
        self.__irf = v
        try:
            # Approximate n0 the initial number of donor molecules in the
            # excited state
            data = self.data
            # Detect in which channel IRF starts
            x_irf = np.argmax(v.y > 0.005)
            x_min = data.x[x_irf]
            # Shift the time-axis by the number of channels
            x = data.x[x_irf:] - x_min
            y = data.y[x_irf:]
            # Using the average arrival time estimate the initial
            # number of molecules in the excited state
            tau0 = np.dot(x, y).sum() / y.sum()
            self.n0 = y.sum() / tau0
        except AttributeError:
            self.n0 = 1000.

    @property
    def data(self):
        if self._data is None:
            try:
                return self.fit.data
            except AttributeError:
                return None
        else:
            return self._data

    @data.setter
    def data(self, v):
        self._data = v

    def scale(self, decay, **kwargs):
        start = kwargs.get('start', min(0, self.start))
        stop = kwargs.get('stop', min(self.stop, len(decay)))
        bg = kwargs.get('bg', 0.0)
        autoscale = kwargs.get('autoscale', self._n0.fixed)
        data = kwargs.get('data', self.data)

        if autoscale:
            weights = 1./data.ey
            self.n0 = float(mfm.fluorescence.tcspc.rescale_w_bg(decay, data.y, weights, bg, start, stop))
        else:
            decay *= self.n0

        return decay

    def convolve(self, data, **kwargs):
        verbose = kwargs.get('verbose', mfm.verbose)
        mode = kwargs.get('mode', self.mode)
        dt = kwargs.get('dt', self.dt)
        rep_rate = kwargs.get('rep_rate', self.rep_rate)
        irf = kwargs.get('irf', self.irf)
        scatter = kwargs.get('scatter', 0.0)
        decay = kwargs.get('decay', np.zeros(self.data.y.shape))

        # Make sure used IRF is of same size as data-array
        irf_y = np.resize(irf.y, self.data.y.shape)
        n_points = irf_y.shape[0]
        stop = min(self.stop, n_points)
        start = min(0, self.start)

        if mode == "per":
            period = 1000. / rep_rate
            mfm.fluorescence.tcspc.fconv_per_cs(decay, data, irf_y, start, stop, n_points, period, dt, n_points)
            # TODO: in future non linear time-axis (better suited for exponentially decaying data)
            # time = fit.data._x
            # mfm.fluorescence.tcspc.fconv_per_dt(decay, lifetime_spectrum, irf_y, start, stop, n_points, period, time)
        elif mode == "exp":
            t = self.data.x
            mfm.fluorescence.tcspc.fconv(decay, data, irf_y, stop, t)
        elif mode == "full":
            decay = np.convolve(data, irf_y, mode="full")[:n_points]

        if verbose:
            print("------------")
            print("Convolution:")
            print("Lifetimes: %s" % data)
            print("dt: %s" % dt)
            print("Irf: %s" % irf.name)
            print("Stop: %s" % stop)
            print("dt: %s" % dt)
            print("Convolution mode: %s" % mode)

        decay += (scatter * irf_y)

        return decay

    def __init__(self, fit, **kwargs):
        kwargs['name'] = 'Convolution'
        kwargs['fit'] = fit
        FittingParameterGroup.__init__(self, **kwargs)

        self._data = None
        try:
            data = kwargs.get('data', fit.data)
            dt = data.dt[0]
            rep_rate = data.setup.rep_rate
            stop = len(data) * dt
            self.data = data
        except AttributeError:
            dt = kwargs.get('dt', 1.0)
            rep_rate = kwargs.get('rep_rate', 1.0)
            stop = 1
            data = kwargs.get('data', None)
        self.data = data

        self._n0 = FittingParameter(value=mfm.settings.cs_settings['tcspc']['n0'],
                                    name='n0',
                                    fixed=mfm.settings.cs_settings['tcspc']['autoscale'],
                                    decimals=2)
        self._dt = FittingParameter(value=dt, name='dt', fixed=True, digits=4)
        self._rep = FittingParameter(value=rep_rate, name='rep', fixed=True)
        self._start = FittingParameter(value=0.0, name='start', fixed=True)
        self._stop = FittingParameter(value=stop, name='stop', fixed=True)
        self._lb = FittingParameter(value=0.0, name='lb')
        self._ts = FittingParameter(value=0.0, name='ts')
        self._do_convolution = mfm.settings.cs_settings['tcspc']['convolution_on_by_default']
        self.mode = mfm.settings.cs_settings['tcspc']['default_convolution_mode']
        self.n_photons_irf = 1.0
        self.__irf = kwargs.get('irf', None)
        if self.__irf is not None:
            self._irf = self.__irf


class ConvolveWidget(Convolve, QtWidgets.QWidget):

    def __init__(self, fit, **kwargs):
        Convolve.__init__(self, fit, **kwargs)
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('mfm/ui/fitting/models/tcspc/convolveWidget.ui', self)

        hide_curve_convolution = kwargs.get('hide_curve_convolution', True)
        if hide_curve_convolution:
            self.radioButton_3.setVisible(not hide_curve_convolution)

        l = QtWidgets.QHBoxLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)
        self._dt.make_widget(layout=l, hide_bounds=True)
        self._n0.make_widget(layout=l)
        self.verticalLayout_2.addLayout(l)

        l = QtWidgets.QHBoxLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)
        self._start.make_widget(layout=l)
        self._stop.make_widget(layout=l)
        self.verticalLayout_2.addLayout(l)

        l = QtWidgets.QHBoxLayout()
        l.setSpacing(0)
        l.setContentsMargins(0, 0, 0, 0)
        self._lb.make_widget(layout=l)
        self._ts.make_widget(layout=l)
        self.verticalLayout_2.addLayout(l)

        self._rep.make_widget(layout=self.horizontalLayout_3, text='r[MHz]')

        self.irf_select = CurveSelector(parent=None,
                                        change_event=self.change_irf,
                                        fit=self.fit,
                                        setup=mfm.experiments.tcspc.TCSPCReader)
        self.actionSelect_IRF.triggered.connect(self.irf_select.show)

        self.radioButton_3.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton_2.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton.clicked.connect(self.onConvolutionModeChanged)
        self.groupBox.toggled.connect(self.onDoConvolutionChanged)

    def onConvolutionModeChanged(self):
        t = "for f in cs.current_fit:\n" \
            "   f.models.convolve.mode = '%s'\n" % self.gui_mode
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def onDoConvolutionChanged(self):
        mfm.run("cs.current_fit.models.convolve.do_convolution = %s" % self.groupBox.isChecked())

    def change_irf(self):
        idx = self.irf_select.selected_curve_index
        t = "irf = cs.current_fit.models.convolve.irf_select.datasets[%s]\n" \
            "for f in cs.current_fit[cs.current_fit._selected_fit:]:\n" \
            "   f.models.convolve._irf = mfm.curve.DataCurve(x=irf.x, y=irf.y)\n" % idx
        t += "cs.current_fit.update()"
        mfm.run(t)
        self.fwhm = self._irf.fwhm
        current_fit = mfm.cs.current_fit
        for f in current_fit[current_fit._selected_fit:]:
            f.model.convolve.lineEdit.setText(self.irf_select.curve_name)

    @property
    def fwhm(self):
        return self._irf.fwhm

    @fwhm.setter
    def fwhm(self, v):
        self._fwhm = v
        if isinstance(v, float):
            self.lineEdit_2.setText("%.3f" % v)
        else:
            self.lineEdit_2.setText(None)

    @property
    def gui_mode(self):
        if self.radioButton_2.isChecked():
            return "exp"
        elif self.radioButton.isChecked():
            return "per"
        elif self.radioButton_3.isChecked():
            return "full"