import numpy as np
from PyQt5 import QtWidgets, uic

import mfm
from mfm.curve import Curve
from mfm.parameter import ParameterGroup, FittingParameter
from mfm.widgets.widgets import CurveSelector


class Convolve(ParameterGroup):

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
        ParameterGroup.__init__(self, **kwargs)

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

        self._n0 = FittingParameter(value=mfm.cs_settings['tcspc']['n0'],
                                    name='n0',
                                    fixed=mfm.cs_settings['tcspc']['autoscale'],
                                    decimals=2)
        self._dt = FittingParameter(value=dt, name='dt', fixed=True, digits=4)
        self._rep = FittingParameter(value=rep_rate, name='rep', fixed=True)
        self._start = FittingParameter(value=0.0, name='start', fixed=True)
        self._stop = FittingParameter(value=stop, name='stop', fixed=True)
        self._lb = FittingParameter(value=0.0, name='lb')
        self._ts = FittingParameter(value=0.0, name='ts')
        self._do_convolution = mfm.cs_settings['tcspc']['convolution_on_by_default']
        self.mode = mfm.cs_settings['tcspc']['default_convolution_mode']
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
            "   f.model.convolve.mode = '%s'\n" % self.gui_mode
        mfm.run(t)
        mfm.run("cs.current_fit.update()")

    def onDoConvolutionChanged(self):
        mfm.run("cs.current_fit.model.convolve.do_convolution = %s" % self.groupBox.isChecked())

    def change_irf(self):
        idx = self.irf_select.selected_curve_index
        t = "irf = cs.current_fit.model.convolve.irf_select.datasets[%s]\n" \
            "for f in cs.current_fit[cs.current_fit._selected_fit:]:\n" \
            "   f.model.convolve._irf = mfm.curve.DataCurve(x=irf.x, y=irf.y)\n" % idx
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

