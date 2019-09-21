"""

"""
from __future__ import annotations

import numpy as np

import mfm
import mfm.cmd
import mfm.fluorescence.tcspc.convolve
import mfm.fluorescence.tcspc.corrections
import mfm.math
import mfm.fluorescence
from mfm.curve import Curve
from mfm.fitting.parameter import FittingParameterGroup, FittingParameter


class Generic(FittingParameterGroup):
    """

    """

    @property
    def n_ph_bg(
            self
    ) -> float:
        """Number of background photons
        """
        if isinstance(self.background_curve, Curve):
            return self._background_curve.y.sum() / self.t_bg * self.t_exp
        else:
            return 0.0

    @property
    def n_ph_exp(
            self
    ) -> int:
        """Number of experimental photons
        """
        if isinstance(self.fit.data, Curve):
            return self.fit.data.y.sum()
        else:
            return 0

    @property
    def n_ph_fl(
            self
    ) -> float:
        """Number of fluorescence photons
        """
        return self.n_ph_exp - self.n_ph_bg

    @property
    def scatter(
            self
    ) -> float:
        # Scatter amplitude
        return self._sc.value

    @scatter.setter
    def scatter(
            self,
            v: float
    ):
        self._sc.value = v

    @property
    def background(
            self
    ) -> float:
        # Constant background in fluorescence decay curve
        return self._bg.value

    @background.setter
    def background(
            self,
            v: float
    ):
        self._bg.value = v

    @property
    def background_curve(
            self
    ) -> mfm.curve.Curve:
        # Background curve
        if isinstance(self._background_curve, Curve):
            return self._background_curve
        else:
            return None

    @background_curve.setter
    def background_curve(
            self,
            v: float
    ):
        if isinstance(v, Curve):
            self._background_curve = v

    @property
    def t_bg(
            self
    ) -> float:
        """Measurement time of background-measurement
        """
        return self._tmeas_bg.value

    @t_bg.setter
    def t_bg(
            self,
            v: float
    ):
        self._tmeas_bg.value = v

    @property
    def t_exp(
            self
    ) -> float:
        """Measurement time of experiment
        """
        return self._tmeas_exp.value

    @t_exp.setter
    def t_exp(
            self,
            v: float
    ):
        self._tmeas_exp.value = v

    def __init__(
            self,
            background_curve: mfm.experiments.data.DataCurve = None,
            name: str = 'Nuisance',
            **kwargs
    ):
        """

        :param background_curve:
        :param name:
        :param kwargs:
        """
        super(Generic, self).__init__(
            name=name,
            **kwargs
        )
        self._background_curve = background_curve
        self._sc = FittingParameter(value=0.0, name='sc', model=self.model)
        self._bg = FittingParameter(value=0.0, name='bg', model=self.model)
        self._tmeas_bg = FittingParameter(value=1.0, name='tBg', lb=0.001, ub=10000000, fixed=True)
        self._tmeas_exp = FittingParameter(value=1.0, name='tMeas', lb=0.001, ub=10000000, fixed=True)


class Corrections(FittingParameterGroup):
    """

    """

    @property
    def lintable(
            self
    ) -> np.array:
        if self._lintable is None:
            self._lintable = np.ones_like(self.fit.data.y)
        return self._lintable[::-1] if self.reverse else self._lintable

    @lintable.setter
    def lintable(
            self,
            v: np.array
    ):
        self._curve = v
        self._lintable = self.calc_lintable(v.y)

    @property
    def window_length(
            self
    ) -> int:
        return int(self._window_length.value)

    @window_length.setter
    def window_length(
            self,
            v: int
    ):
        self._window_length.value = v
        self._lintable = self.calc_lintable(self._curve.y)

    @property
    def window_function(
            self
    ) -> str:
        return self._window_function

    @window_function.setter
    def window_function(
            self,
            v: str
    ):
        self._window_function = v
        self._lintable = self.calc_lintable(self._curve.y)

    @property
    def reverse(
            self
    ) -> bool:
        return self._reverse

    @reverse.setter
    def reverse(
            self,
            v: bool
    ):
        self._reverse = v

    def calc_lintable(
            self,
            y,
            xmin: int = None,
            xmax: int = None,
            window_function: str = None,
            window_length: int = None
    ):
        """

        :param y:
        :param xmin:
        :param xmax:
        :param window_function:
        :param window_length:
        :return:
        """
        if xmin is None:
            xmin = self.fit.xmin
        if xmax is None:
            xmax = self.fit.xmax
        if window_function is None:
            window_function = self.window_function
        if window_length is None:
            window_length = self.window_length
        return mfm.fluorescence.tcspc.corrections.compute_linearization_table(
            y,
            window_length,
            window_function,
            xmin,
            xmax
        )

    @property
    def measurement_time(
            self
    ) -> float:
        return self.fit.model.generic.t_exp

    @measurement_time.setter
    def measurement_time(
            self,
            v: float
    ):
        self.fit.model.generic.t_exp = v

    @property
    def rep_rate(self) -> float:
        return self.fit.model.convolve.rep_rate

    @rep_rate.setter
    def rep_rate(
            self,
            v: float
    ):
        self.fit.model.convolve.rep_rate = v

    @property
    def dead_time(
            self
    ) -> float:
        return self._dead_time.value

    @dead_time.setter
    def dead_time(
            self,
            v: float
    ):
        self._dead_time.value = v

    def pileup(
            self,
            decay: np.array,
            **kwargs
    ):
        """

        :param decay:
        :param kwargs:
        :return:
        """
        data = kwargs.get('data', self.fit.data.y)
        rep_rate = kwargs.get('rep_rate', self.rep_rate)
        dead_time = kwargs.get('dead_time', self.dead_time)
        meas_time = kwargs.get('meas_time', self.measurement_time)
        if self.correct_pile_up:
            mfm.fluorescence.tcspc.corrections.correct_model_for_pile_up(
                data,
                decay,
                rep_rate,
                dead_time,
                meas_time,
                verbose=self.verbose
            )

    def linearize(
            self,
            decay: np.array,
            **kwargs
    ):
        lintable = kwargs.get('lintable', self.lintable)
        if lintable is not None and self.correct_dnl:
            return decay * lintable
        return decay

    def __init__(
            self,
            fit: mfm.fitting.fit.Fit,
            name: str = 'Corrections',
            reverse: bool = False,
            correct_dnl: bool = False,
            window_function: str = 'hanning',
            correct_pile_up: bool = False,
            lin_auto_range: bool = True,
            **kwargs
    ):
        super().__init__(
            fit=fit,
            name=name,
            **kwargs
        )
        self._lintable = None
        self._curve = None
        self._reverse = reverse
        self.correct_dnl = correct_dnl
        self.correct_pile_up = correct_pile_up
        self._window_function = window_function
        self._auto_range = lin_auto_range
        self._dead_time = FittingParameter(value=85.0, name='tDead', fixed=True, decimals=1)
        self._window_length = FittingParameter(value=17.0, name='win-size', fixed=True, decimals=0)


class Convolve(FittingParameterGroup):
    """

    """

    @property
    def dt(
            self
    ) -> float:
        return self._dt.value

    @dt.setter
    def dt(
            self,
            v: float
    ):
        self._dt.value = v

    @property
    def lamp_background(
            self
    ) -> float:
        return self._lb.value / self.n_photons_irf

    @lamp_background.setter
    def lamp_background(
            self,
            v: float
    ):
        self._lb.value = v

    @property
    def timeshift(
            self
    ) -> float:
        return self._ts.value

    @timeshift.setter
    def timeshift(
            self,
            v: float
    ):
        self._ts.value = v

    @property
    def start(
            self
    ) -> int:
        return int(self._start.value // self.dt)

    @start.setter
    def start(
            self,
            v: int
    ):
        self._start.value = v

    @property
    def stop(
            self
    ) -> int:
        stop = int(self._stop.value // self.dt)
        return stop

    @stop.setter
    def stop(
            self,
            v: int
    ):
        self._stop.value = v

    @property
    def rep_rate(
            self
    ) -> float:
        return self._rep.value

    @rep_rate.setter
    def rep_rate(
            self,
            v: float
    ):
        self._rep.value = float(v)

    @property
    def do_convolution(
            self
    ) -> bool:
        return self._do_convolution

    @do_convolution.setter
    def do_convolution(
            self,
            v: bool
    ):
        self._do_convolution = bool(v)

    @property
    def n0(
            self
    ) -> float:
        return self._n0.value

    @n0.setter
    def n0(
            self,
            v: float
    ):
        self._n0.value = v

    @property
    def irf(
            self
    ) -> mfm.curve.Curve:
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
    def _irf(
            self
    ) -> mfm.curve.Curve:
        return self.__irf

    @_irf.setter
    def _irf(
            self,
            v: mfm.curve.Curve
    ):
        self.n_photons_irf = v.normalize(mode="sum")
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
    def data(
            self
    ) -> mfm.experiments.data.DataCurve:
        if self._data is None:
            try:
                return self.fit.data
            except AttributeError:
                return None
        else:
            return self._data

    @data.setter
    def data(
            self,
            v: mfm.experiments.data.DataCurve
    ):
        self._data = v

    def scale(
            self,
            decay: mfm.experiments.data.DataCurve,
            start: int = None,
            stop: int = None,
            bg: float = 0.0,
            data: np.array = None,
            autoscale: bool = None
    ) -> np.array:
        if start is None:
            start = min(0, self.start)
        if stop is None:
            stop = min(self.stop, len(decay))
        if autoscale is None:
            autoscale = self._n0.fixed
        if data is None:
            data = self.data

        if autoscale:
            weights = 1.0 / data.ey
            self.n0 = float(mfm.fluorescence.tcspc.rescale_w_bg(decay, data.y, weights, bg, start, stop))
        else:
            decay *= self.n0

        return decay

    def convolve(
            self,
            data: mfm.experiments.data.DataCurve,
            verbose: bool = None,
            mode: str = None,
            dt: float = None,
            rep_rate: float = None,
            irf: mfm.curve.Curve = None,
            scatter: float = 0.0,
            decay: np.array = None
    ) -> np.array:
        if verbose is None:
            verbose = mfm.verbose
        if mode is None:
            mode = self.mode
        if dt is None:
            dt = self.dt
        if rep_rate is None:
            rep_rate = self.rep_rate
        if irf is None:
            irf = self.irf
        if decay is None:
            decay = np.zeros(self.data.y.shape)

        # Make sure used IRF is of same size as data-array
        irf_y = np.resize(irf.y, self.data.y.shape)
        n_points = irf_y.shape[0]
        stop = min(self.stop, n_points)
        start = min(0, self.start)

        if mode == "per":
            period = 1000. / rep_rate
            mfm.fluorescence.tcspc.convolve.convolve_lifetime_spectrum_periodic(decay, data, irf_y, start, stop, n_points, period, dt, n_points)
            # TODO: in future non linear time-axis (better suited for exponentially decaying data)
            # time = fit.data._x
            # mfm.fluorescence.tcspc.fconv_per_dt(decay, lifetime_spectrum, irf_y, start, stop, n_points, period, time)
        elif mode == "exp":
            t = self.data.x
            mfm.fluorescence.tcspc.convolve.convolve_lifetime_spectrum(
                decay,
                data,
                irf_y,
                stop,
                t
            )
        elif mode == "full":
            decay = np.convolve(
                data,
                irf_y,
                mode="full"
            )[:n_points]

        if verbose:
            print("------------")
            print("Convolution:")
            print("Lifetimes: %s" % data)
            print("dt: %s" % dt)
            print("Irf: %s" % irf.name)
            print("Stop: %s" % stop)
            print("dt: %s" % dt)
            print("Convolution file_type: %s" % mode)

        decay += (scatter * irf_y)
        return decay

    def __init__(
            self,
            fit: mfm.fitting.fit.Fit,
            name: str = 'Convolution',
            irf: mfm.curve.Curve = None,
            **kwargs
    ):
        super(Convolve, self).__init__(
            fit=fit,
            name=name,
            **kwargs
        )

        self._data = None
        try:
            data = kwargs.get('data', fit.data)
            dt = data.dx[0]
            rep_rate = data.setup.rep_rate
            stop = len(data) * dt
            self.data = data
        except AttributeError:
            dt = kwargs.get('dt', 1.0)
            rep_rate = kwargs.get('rep_rate', 1.0)
            stop = 1
            data = kwargs.get('data', None)
        self.data = data

        self._n0 = FittingParameter(
            value=mfm.settings.cs_settings['tcspc']['n0'],
            name='n0',
            fixed=mfm.settings.cs_settings['tcspc']['autoscale'],
            decimals=2
        )
        self._dt = FittingParameter(
            value=dt,
            name='dt',
            fixed=True,
            digits=4
        )
        self._rep = FittingParameter(
            value=rep_rate,
            name='rep',
            fixed=True
        )
        self._start = FittingParameter(
            value=0.0,
            name='start',
            fixed=True
        )
        self._stop = FittingParameter(
            value=stop,
            name='stop',
            fixed=True
        )
        self._lb = FittingParameter(
            value=0.0,
            name='lb'
        )
        self._ts = FittingParameter(
            value=0.0,
            name='ts'
        )
        self._do_convolution = mfm.settings.cs_settings['tcspc']['convolution_on_by_default']
        self.mode = mfm.settings.cs_settings['tcspc']['default_convolution_mode']
        self.n_photons_irf = 1.0

        self.__irf = irf
        if self.__irf is not None:
            self._irf = self.__irf


