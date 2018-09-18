from PyQt4 import QtGui, uic
import numpy as np
import mfm
from mfm.curve import Curve
from mfm.parameter import AggregatedParameters, FittingParameter
from mfm.widgets import CurveSelector


class Convolve(AggregatedParameters):

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
            x = np.copy(self.fit.data.x)
            y = np.zeros_like(self.fit.data.y)
            y[0] = 1.0
            curve = mfm.curve.Curve(x, y)
            return curve

    @property
    def _irf(self):
        return self.__irf

    @_irf.setter
    def _irf(self, v):
        self.n_photons_irf = v.norm(mode="sum")
        self.__irf = v
        # Approximate n0 the initial number of donor molecules in the
        # excited state
        data = self.fit.data
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

    def scale(self, decay, **kwargs):
        fit = kwargs.get('fit', self.fit)
        start = kwargs.get('start', min(0, self.start))
        stop = kwargs.get('stop', min(self.stop, len(decay)))
        bg = kwargs.get('bg', fit.model.generic.background)
        autoscale = kwargs.get('autoscale', self._n0.fixed)
        data = kwargs.get('data', fit.data)

        if autoscale:
            self.n0 = float(mfm.fluorescence.tcspc.rescale_w_bg(decay, data.y, data.weights, bg, start, stop))
        else:
            decay *= self.n0

        return decay

    def convolve(self, lifetime_spectrum, **kwargs):
        verbose = kwargs.get('verbose', mfm.verbose)
        mode = kwargs.get('mode', self.mode)
        dt = kwargs.get('dt', self.dt)
        rep_rate = kwargs.get('rep_rate', self.rep_rate)
        irf = kwargs.get('irf', self.irf)
        scatter = kwargs.get('scatter', 0.0)
        fit = kwargs.get('fit', self.fit)
        decay = kwargs.get('decay', np.zeros(fit.data._y.shape))
        #print decay

        # Make sure used IRF is of same size as data-array
        irf_y = np.resize(irf.y, fit.data.y.shape)
        n_points = irf_y.shape[0]
        stop = min(self.stop, n_points)
        start = min(0, self.start)
        #print start
        #print stop

        if mode == "per":
            period = 1000. / rep_rate
            mfm.fluorescence.tcspc.fconv_per(decay, lifetime_spectrum, irf_y, start, stop, n_points, period, dt)

            #mfm.fluorescence.tcspc.fconv_per_cs(decay, lifetime_spectrum, irf_y, start, stop, n_points, period, dt, n_points)
            # TODO: in future non linear time-axis (better suited for exponentially decaying data)
            # time = fit.data._x
            # mfm.fluorescence.tcspc.fconv_per_dt(decay, lifetime_spectrum, irf_y, start, stop, n_points, period, time)
        elif mode == "exp":
            t = fit.data._x
            mfm.fluorescence.tcspc.fconv(decay, lifetime_spectrum, irf_y, stop, t)
        elif mode == "full":
            lifetime_spectrum = np.convolve(decay, irf_y, mode="full")[:n_points]
        #print decay

        if verbose:
            print("------------")
            print("Convolution:")
            print("Lifetimes: %s" % lifetime_spectrum)
            print("dt: %s" % dt)
            print("Irf: %s" % irf.name)
            print("Stop: %s" % stop)
            print("dt: %s" % dt)
            print("Convolution mode: %s" % mode)

        decay += (scatter * irf_y)
        #print decay

        return decay

    def __init__(self, fit, **kwargs):
        kwargs['name'] = 'Convolution'
        kwargs['fit'] = fit
        AggregatedParameters.__init__(self, **kwargs)

        self._n0 = FittingParameter(value=mfm.settings['tcspc']['n0'],
                                    name='n0',
                                    fixed=mfm.settings['tcspc']['autoscale'],
                                    decimals=2)
        self._dt = FittingParameter(value=fit.data.dt[0], name='dt', fixed=True, digits=4)
        self._rep = FittingParameter(value=fit.data.setup.rep_rate, name='rep', fixed=True)
        self._start = FittingParameter(value=0.0, name='start', fixed=True)
        self._stop = FittingParameter(value=len(fit.data) * self.dt, name='stop', fixed=True)
        self._lb = FittingParameter(value=0.0, name='lb')
        self._ts = FittingParameter(value=0.0, name='ts')
        self._do_convolution = mfm.settings['tcspc']['convolution_on_by_default']
        self.mode = mfm.settings['tcspc']['default_convolution_mode']
        self.n_photons_irf = 1.0
        self.__irf = None


class ConvolveWidget(Convolve, QtGui.QWidget):

    def __init__(self, fit, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/fitting/models/tcspc/convolveWidget.ui', self)
        Convolve.__init__(self, fit, **kwargs)

        hide_curve_convolution = kwargs.get('hide_curve_convolution', True)
        if hide_curve_convolution:
            self.radioButton_3.setVisible(not hide_curve_convolution)

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        self._dt = self._dt.widget(layout=l, hide_bounds=True)
        self._n0 = self._n0.widget(layout=l)
        self.verticalLayout_2.addLayout(l)

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        self._start = self._start.widget(layout=l)
        self._stop = self._stop.widget(layout=l)
        self.verticalLayout_2.addLayout(l)

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        self._lb = self._lb.widget(layout=l)
        self._ts = self._ts.widget(layout=l)
        self.verticalLayout_2.addLayout(l)

        self._rep = self._rep.widget(layout=self.horizontalLayout_3, text='r[MHz]')

        self.irf_select = CurveSelector(parent=None,
                                        change_event=self.change_irf,
                                        fit=self.fit,
                                        setup=mfm.experiments.tcspc.TCSPCSetup)
        self.actionSelect_IRF.triggered.connect(self.irf_select.show)

        self.radioButton_3.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton_2.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton.clicked.connect(self.onConvolutionModeChanged)
        self.groupBox.toggled.connect(self.onDoConvolutionChanged)

    def onConvolutionModeChanged(self):
        #mfm.run("cs.current_fit.model = '%s'" % self.gui_mode)
        t = "for f in cs.current_fit:\n" \
            "   f.model.convolve.mode = '%s'\n" % self.gui_mode
        mfm.run(t)

        #t = "for f in cs.current_fit:\n" \
        #    "   f.model.convolve.radioButton.setChecked(True)\n"
        #mfm.run(t)

        mfm.run("cs.current_fit.update()")

    def onDoConvolutionChanged(self):
        mfm.run("cs.current_fit.model.convolve.do_convolution = %s" % self.groupBox.isChecked())

    def change_irf(self):
        idx = self.irf_select.selected_curve_index
        t = "irf = cs.current_fit.model.convolve.irf_select.datasets[%s]\n" \
            "for f in cs.current_fit[cs.current_fit._current_fit:]:\n" \
            "   f.model.convolve._irf = mfm.curve.DataCurve(x=irf.x, y=irf.y)\n" % idx
        mfm.run(t)

        irf_name = self.irf_select.curve_name
        t = "for f in cs.current_fit[cs.current_fit._current_fit:]:\n" \
            "   f.model.convolve.lineEdit.setText('%s')\n" % irf_name
        mfm.run(t)

        mfm.run("cs.current_fit.update()")
        self.fwhm = self._irf.fwhm

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

