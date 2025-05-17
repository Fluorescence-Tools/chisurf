from __future__ import annotations

import numpy as np
import scipy.stats

import chisurf.data
import chisurf.experiments
import chisurf.macros
import chisurf.fluorescence.tcspc.convolve
import chisurf.fluorescence.tcspc.corrections
import chisurf.math
import chisurf.fluorescence
from chisurf.curve import Curve
from chisurf.fitting.parameter import (
    FittingParameterGroup, FittingParameter
)


class Generic(FittingParameterGroup):

    @property
    def n_ph_bg(self) -> float:
        """Number of background photons
        """
        n_bg = 0.0
        if isinstance(self.background_curve, Curve):
            a = self._background_curve.y.sum() / self.t_bg * self.t_exp
            if not np.isnan(a):
                n_bg += a
        n_bg += self._bg.value * len(self.fit.data.x)
        return n_bg

    @property
    def n_ph_exp(self) -> int:
        """Number of fluorescence photons
        """
        if isinstance(self.fit.data, Curve):
            return self.fit.data.y.sum()
        else:
            return 0

    @property
    def n_ph_fl(self) -> float:
        """Number of fluorescence photons
        """
        return max(self.n_ph_exp - self.n_ph_bg, 1.0)

    @property
    def scatter(self) -> float:
        # Scatter amplitude
        return self._sc.value

    @scatter.setter
    def scatter(self, v: float):
        self._sc.value = v

    @property
    def background(self) -> float:
        # Constant background in fluorescence decay curve
        return self._bg.value

    @background.setter
    def background(self, v: float):
        self._bg.value = v

    @property
    def background_curve(self) -> chisurf.curve.Curve:
        if isinstance(self._background_curve, Curve):
            return self._background_curve
        else:
            return None

    @background_curve.setter
    def background_curve(self, v: float):
        if isinstance(v, Curve):
            self._background_curve = v

    def unload_background_curve(self):
        """Unload the background curve and reset it to default (None)
        """
        self._background_curve = None

    @property
    def t_bg(self) -> float:
        """Measurement time of background-measurement
        """
        return self._tmeas_bg.value

    @t_bg.setter
    def t_bg(self, v: float):
        self._tmeas_bg.value = v

    @property
    def t_exp(self) -> float:
        """Measurement time of experiment
        """
        return self._tmeas_exp.value

    @t_exp.setter
    def t_exp(self, v: float):
        self._tmeas_exp.value = v

    def __init__(
            self,
            background_curve: chisurf.data.DataCurve = None,
            name: str = 'Nuisance',
            **kwargs
    ):

        super().__init__(
            name=name,
            **kwargs
        )
        self._background_curve = background_curve
        self._sc = FittingParameter(
            value=0.0,
            name='sc',
            lb=0.0,
            ub=100.0,
            bounds_on=True
        )
        self._bg = FittingParameter(
            value=0.0,
            name='bg'
        )
        self._tmeas_bg = FittingParameter(
            value=1.0,
            name='tBg',
            lb=1e-6,
            ub=1e9,
            fixed=True,
            bounds_on=True
        )
        self._tmeas_exp = FittingParameter(
            value=1.0,
            name='tMeas',
            fixed=True,
            lb=1e-6,
            ub=1e9,
            bounds_on=True
        )


class Corrections(FittingParameterGroup):

    @property
    def lintable(self) -> np.array:
        if self._lintable is None:
            self._lintable = np.ones_like(self.fit.data.y)
        return self._lintable[::-1] if self.reverse else self._lintable

    @lintable.setter
    def lintable(self, v: np.array):
        self._curve = v
        self._lintable = self.calc_lintable(v.y)

    def unload_lintable(self):
        """Unload the linearization table and reset it to default (array of ones)
        """
        self._curve = None
        self._lintable = None

    @property
    def window_length(self) -> int:
        return int(self._window_length.value)

    @window_length.setter
    def window_length(self, v: int):
        self._window_length.value = v
        self._lintable = self.calc_lintable(self._curve.y)

    @property
    def window_function(self) -> str:
        return self._window_function

    @window_function.setter
    def window_function(self, v: str):
        self._window_function = v
        self._lintable = self.calc_lintable(self._curve.y)

    @property
    def reverse(self) -> bool:
        return self._reverse

    @reverse.setter
    def reverse(self, v: bool):
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
        return chisurf.fluorescence.tcspc.corrections.compute_linearization_table(
            y,
            window_length,
            window_function,
            xmin,
            xmax
        )

    @property
    def measurement_time(self) -> float:
        try:
            return self.fit.model.generic.t_exp
        except (AttributeError, KeyError):
            return 1.0

    @measurement_time.setter
    def measurement_time(
            self,
            v: float
    ):
        self.fit.model.generic.t_exp = v

    @property
    def rep_rate(self) -> float:
        try:
            return self.fit.model.convolve.rep_rate
        except (AttributeError, KeyError):
            return 1.0

    @rep_rate.setter
    def rep_rate(self, v: float):
        self.fit.model.convolve.rep_rate = v

    @property
    def dead_time(self) -> float:
        return self._dead_time.value

    @dead_time.setter
    def dead_time(self, v: float):
        self._dead_time.value = v

    def pileup(self, decay: np.array, **kwargs):
        data = kwargs.get('data', self.fit.data.y)
        rep_rate = kwargs.get('rep_rate', self.rep_rate)
        dead_time = kwargs.get('dead_time', self.dead_time)
        meas_time = kwargs.get('meas_time', self.measurement_time)
        if self.correct_pile_up:
            chisurf.fluorescence.tcspc.corrections.add_pile_up_to_model(
                data,
                decay,
                rep_rate,
                dead_time,
                meas_time,
                modify_inplace=True
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
            fit: chisurf.fitting.fit.Fit,
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

    @property
    def dt(self) -> float:
        return self._dt.value

    @dt.setter
    def dt(self, v: float):
        self._dt.value = v

    @property
    def lamp_background(self) -> float:
        return self._lb.value # / self.n_photons_irf

    @lamp_background.setter
    def lamp_background(self, v: float):
        self._lb.value = v

    @property
    def timeshift(self) -> float:
        return self._ts.value

    @timeshift.setter
    def timeshift(self, v: float):
        self._ts.value = v

    @property
    def start(self) -> int:
        return int(self._start.value // self.dt)

    @start.setter
    def start(self, v: int):
        self._start.value = v

    @property
    def stop(self) -> int:
        stop = int(self._stop.value // self.dt)
        return stop

    @stop.setter
    def stop(self, v: int):
        self._stop.value = v

    @property
    def irf_start(self) -> int:
        return int(self._irf_start.value // self.dt)

    @irf_start.setter
    def irf_start(self, v: int):
        # Convert to numpy array of long integers
        v_array = np.array([v], dtype=np.int64)
        self._irf_start.value = v_array

    @property
    def irf_stop(self) -> int:
        stop = int(self._irf_stop.value // self.dt)
        return stop

    @irf_stop.setter
    def irf_stop(self, v: int):
        # Convert to numpy array of long integers
        v_array = np.array([v], dtype=np.int64)
        self._irf_stop.value = v_array

    @property
    def rep_rate(self) -> float:
        return self._rep.value

    @rep_rate.setter
    def rep_rate(self, v: float):
        self._rep.value = float(v)

    @property
    def do_convolution(self) -> bool:
        return self._do_convolution

    @do_convolution.setter
    def do_convolution(self, v: bool):
        self._do_convolution = bool(v)

    @property
    def n0(self) -> float:
        return self._n0.value

    @n0.setter
    def n0(self, v: float):
        self._n0.value = v

    @property
    def irf(self) -> chisurf.curve.Curve:
        if isinstance(self._irf, chisurf.curve.Curve):
            irf = self._irf
            irf -= self.lamp_background
            irf.y = np.clip(irf.y, 0, None)
        else:
            start_fraction = 0.1
            x = np.copy(self.data.x)
            x_min = np.where(self.data.y > start_fraction * np.max(self.data.y))[0][0]
            loc = self.data.x[x_min]
            scale = self._iw.value
            shape = self._ik.value
            y = chisurf.math.functions.distributions.generalized_normal_distribution(x, loc, scale, shape, True)
            y *= np.sum(self.data.y)
            irf = chisurf.curve.Curve(x=x, y=y)
            irf.y[irf.y < 1] = 0.0

        # Zero out the IRF outside the specified range
        irf_start_idx = self.irf_start
        irf_stop_idx = self.irf_stop
        if irf_start_idx > 0 or irf_stop_idx < len(irf.y):
            # Create a copy to avoid modifying the original
            irf_y = np.copy(irf.y)
            # Zero out before irf_start
            if irf_start_idx > 0:
                irf_y[:irf_start_idx] = 0.0
            # Zero out after irf_stop
            if irf_stop_idx < len(irf_y):
                irf_y[irf_stop_idx:] = 0.0
            # Create a new curve with the modified y values
            irf = chisurf.curve.Curve(x=irf.x, y=irf_y)

        irf = irf << float(self.timeshift)
        return irf

    @property
    def _irf(self) -> chisurf.curve.Curve:
        re = self.__irf
        # if re is None:
        #     x = self.fit.data.x
        #     y = np.zeros_like(self.fit.data.y)
        #     re = chisurf.curve.Curve(x, y)
        return re

    @_irf.setter
    def _irf(self, v: chisurf.curve.Curve):
        self.n_photons_irf = v.normalize(mode="sum", inplace=False)
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

            # Update the upper bound of the lamp background parameter to half the lamp height
            if hasattr(v, 'y') and len(v.y) > 0:
                lamp_height = np.max(v.y)
                if lamp_height > 0:
                    # Get current bounds and update only the upper bound
                    current_bounds = self._lb.bounds
                    self._lb.bounds = current_bounds[0], lamp_height / 2
        except AttributeError:
            self.n0 = 1000.

    @property
    def data(self) -> chisurf.data.DataCurve:
        if self._data is None:
            try:
                return self.fit.data
            except AttributeError:
                return None
        else:
            return self._data

    @data.setter
    def data(self, v: chisurf.data.DataCurve):
        self._data = v

    def scale(
            self,
            decay: chisurf.data.DataCurve,
            start: int = None,
            stop: int = None,
            bg: float = 0.0,
            data: np.ndarray = None,
            autoscale: bool = None
    ) -> np.ndarray:
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
            n0 = chisurf.fluorescence.tcspc.rescale_w_bg(
                model_decay=decay,
                experimental_decay=data.y,
                experimental_weights=weights,
                experimental_background=bg,
                start=start,
                stop=stop
            )
            self._n0.fixed = False
            self._n0.value = n0
            self._n0.fixed = True
        decay *= self.n0

        return decay

    def unload_irf(self):
        """Unload the IRF and reset it to default (None)
        """
        self.__irf = None

    def convolve(
            self,
            data: chisurf.data.DataCurve,
            verbose: bool = False,
            mode: str = None,
            dt: float = None,
            rep_rate: float = None,
            irf: chisurf.curve.Curve = None,
            scatter: float = 0.0,
            decay: np.array = None
    ) -> np.array:
        if verbose is None:
            verbose = chisurf.settings.cs_settings['verbose']
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
            chisurf.fluorescence.tcspc.convolve.convolve_lifetime_spectrum_periodic_nb(
                decay, data, irf_y,
                start, stop, n_points,
                period, dt, n_points
            )
            # TODO: in future non linear time-axis (better suited for exponentially decaying data)
            # time = fit.data._x
            # chisurf.fluorescence.tcspc.fconv_per_dt(model_decay, lifetime_spectrum, irf_y, start, stop, n_points, period,
            # time)
        elif mode == "exp":
            t = self.data.x
            chisurf.fluorescence.tcspc.convolve.convolve_lifetime_spectrum_nb(
                output_decay=decay,
                lifetime_spectrum=data,
                instrument_response_function=irf_y,
                convolution_stop=stop,
                time_axis=t
            )
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

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            name: str = 'Convolution',
            irf: chisurf.curve.Curve = None,
            **kwargs
    ):
        super().__init__(fit=fit, name=name, **kwargs)

        self._data = None
        try:
            data = kwargs.get('data', fit.data)
            dt = data.dx[0]
            rep_rate = data.data_reader.rep_rate
            stop = len(data) * dt
            self.data = data
        except (AttributeError, TypeError):
            dt = kwargs.get('dt', 1.0)
            rep_rate = kwargs.get('rep_rate', 1.0)
            stop = 1
            data = kwargs.get('data', None)
        self.data = data

        self._n0 = FittingParameter(
            value=chisurf.settings.cs_settings['tcspc']['n0'],
            name='n0',
            label_text="n<sub>0</sub>",
            fixed=chisurf.settings.cs_settings['tcspc']['autoscale'],
            decimals=4
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
        self._irf_start = FittingParameter(
            value=0.0,
            name='irf_start',
            label_text="IRF Start",
            fixed=True
        )
        self._irf_stop = FittingParameter(
            value=stop,
            name='irf_stop',
            label_text="IRF Stop",
            fixed=True
        )
        # Set bounds for lamp background to be between 0 and half the lamp height
        # Default upper bound will be updated when IRF is set
        self._lb = FittingParameter(
            value=0.0,
            name='lb',
            fixed=True,
            bounds_on=True,
            lb=0.0,
            ub=1.0  # Default upper bound, will be updated when IRF is set
        )
        self._ts = FittingParameter(
            value=0.0,
            name='ts',
            bounds_on=False
        )

        self._iw = FittingParameter(
            value=0.10,
            name='iw',
            fixed=True,
            label_text='IRF<sub>w</sub>'
        )
        self._ik = FittingParameter(
            value=-0.31,
            name='ik',
            fixed=True,
            label_text='IRF<sub>k</sub>'
        )

        self._do_convolution = chisurf.settings.cs_settings['tcspc']['convolution_on_by_default']
        self.mode = chisurf.settings.cs_settings['tcspc']['default_convolution_mode']
        self.n_photons_irf = 1.0

        self.__irf = irf
        if self.__irf is not None:
            self._irf = self.__irf
