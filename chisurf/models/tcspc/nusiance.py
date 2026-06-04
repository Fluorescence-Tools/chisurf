from __future__ import annotations

import logging

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
        """Scatter amplitude."""
        # Scatter amplitude
        return self._sc.value

    @scatter.setter
    def scatter(self, v: float):
        """Scatter amplitude."""
        self._sc.value = v

    @property
    def background(self) -> float:
        """Constant background in fluorescence decay curve."""
        # Constant background in fluorescence decay curve
        return self._bg.value

    @background.setter
    def background(self, v: float):
        """Constant background in fluorescence decay curve."""
        self._bg.value = v

    @property
    def background_curve(self) -> chisurf.curve.Curve:
        """Background curve used for background subtraction."""
        if isinstance(self._background_curve, Curve):
            return self._background_curve
        else:
            return None

    @background_curve.setter
    def background_curve(self, v: float):
        """Background curve used for background subtraction."""
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
        """Measurement time of background measurement."""
        self._tmeas_bg.value = v

    @property
    def t_exp(self) -> float:
        """Measurement time of experiment
        """
        return self._tmeas_exp.value

    @t_exp.setter
    def t_exp(self, v: float):
        """Measurement time of experiment."""
        self._tmeas_exp.value = v

    # TODO: needs docstring
    def __init__(
            self,
            background_curve: chisurf.data.DataCurve = None,
            name: str = 'Nuisance',
            **kwargs
    ):
        """Initialize the instance."""

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
        """Linearization table for DNL correction."""
        if self._lintable is None:
            self._lintable = np.ones_like(self.fit.data.y)
        return self._lintable[::-1] if self.reverse else self._lintable

    @lintable.setter
    def lintable(self, v: np.array):
        """Linearization table for DNL correction."""
        self._curve = v
        self._lintable = self.calc_lintable(v.y)

    def unload_lintable(self):
        """Unload the linearization table and reset it to default (array of ones)
        """
        self._curve = None
        self._lintable = None

    @property
    def window_length(self) -> int:
        """Window length for the smoothing window function."""
        return int(self._window_length.value)

    @window_length.setter
    def window_length(self, v: int):
        """Window length for the smoothing window function."""
        self._window_length.value = v
        self._lintable = self.calc_lintable(self._curve.y)

    @property
    def window_function(self) -> str:
        """Name of the window function used for smoothing."""
        return self._window_function

    @window_function.setter
    def window_function(self, v: str):
        """Name of the window function used for smoothing."""
        self._window_function = v
        self._lintable = self.calc_lintable(self._curve.y)

    @property
    def reverse(self) -> bool:
        """Whether to reverse the linearization table."""
        return self._reverse

    @reverse.setter
    def reverse(self, v: bool):
        """Whether to reverse the linearization table."""
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
        """Measurement time of the experiment in seconds."""
        try:
            return self.fit.model.generic.t_exp
        except (AttributeError, KeyError):
            return 1.0

    @measurement_time.setter
    def measurement_time(
            self,
            v: float
    ):
        """Measurement time of the experiment in seconds."""
        self.fit.model.generic.t_exp = v

    @property
    def rep_rate(self) -> float:
        """Laser repetition rate in MHz."""
        try:
            return self.fit.model.convolve.rep_rate
        except (AttributeError, KeyError):
            return 1.0

    @rep_rate.setter
    def rep_rate(self, v: float):
        """Laser repetition rate in MHz."""
        self.fit.model.convolve.rep_rate = v

    @property
    def dead_time(self) -> float:
        """Dead time of the detector in ns."""
        return self._dead_time.value

    @dead_time.setter
    def dead_time(self, v: float):
        """Dead time of the detector in ns."""
        self._dead_time.value = v

    # TODO: needs docstring
    def pileup(self, decay: np.array, **kwargs):
        """Apply pile-up correction to the decay."""
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

    # TODO: needs docstring
    def linearize(
            self,
            decay: np.array,
            **kwargs
    ):
        """Apply DNL linearization to the decay."""
        lintable = kwargs.get('lintable', self.lintable)
        if lintable is not None and self.correct_dnl:
            return decay * lintable
        return decay

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of linearization state.

        This intentionally captures only lightweight, non-Qt attributes
        that are not already handled via :class:`FittingParameter` (those
        are serialized by the generic fit_state helpers).
        """

        state: dict = {
            "correct_dnl": bool(self.correct_dnl),
            "correct_pile_up": bool(self.correct_pile_up),
            "reverse": bool(self.reverse),
            "window_function": str(self.window_function),
            "lin_auto_range": bool(getattr(self, "_auto_range", True)),
        }

        # Persist the linearization table itself if it has been computed.
        # This allows project round-trips to reuse the same correction even
        # if the underlying helper curve is no longer available.
        try:
            lt = getattr(self, "_lintable", None)
            if isinstance(lt, np.ndarray) and lt.size > 0:
                state["lintable"] = lt.tolist()
        except Exception:
            pass

        return state

    def set_state(self, state: dict) -> None:
        """Restore linearization state from :meth:`get_state` output."""

        if not isinstance(state, dict):
            return

        try:
            self.correct_dnl = bool(state.get("correct_dnl", self.correct_dnl))
        except Exception:
            pass
        try:
            self.correct_pile_up = bool(state.get("correct_pile_up", self.correct_pile_up))
        except Exception:
            pass
        try:
            self.reverse = bool(state.get("reverse", self.reverse))
        except Exception:
            pass

        wf = state.get("window_function")
        if isinstance(wf, str):
            try:
                self.window_function = wf
            except Exception:
                pass

        if "lin_auto_range" in state:
            try:
                self._auto_range = bool(state["lin_auto_range"])
            except Exception:
                pass

        # Restore linearization table if present. We store the raw
        # (non-reversed) table and let the ``lintable`` property handle
        # orientation via the ``reverse`` flag.
        lt = state.get("lintable")
        if isinstance(lt, (list, tuple)):
            try:
                arr = np.asarray(lt, dtype=float)
                if arr.size > 0:
                    self._lintable = arr
            except Exception:
                pass

    # TODO: needs docstring
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
        """Initialize the instance."""
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
        """Time step per channel in ns."""
        return self._dt.value

    @dt.setter
    def dt(self, v: float):
        """Time step per channel in ns."""
        self._dt.value = v

    @property
    def lamp_background(self) -> float:
        """Lamp background offset."""
        return self._lb.value # / self.n_photons_irf

    @lamp_background.setter
    def lamp_background(self, v: float):
        """Lamp background offset."""
        self._lb.value = v

    @property
    def timeshift(self) -> float:
        """Time shift of the IRF relative to the decay."""
        return self._ts.value

    @timeshift.setter
    def timeshift(self, v: float):
        """Time shift of the IRF relative to the decay."""
        self._ts.value = v

    @property
    def start(self) -> int:
        """Start channel for convolution (in indices)."""
        return int(self._start.value // self.dt)

    @start.setter
    def start(self, v: int):
        """Start channel for convolution (in indices)."""
        self._start.value = v

    @property
    def stop(self) -> int:
        """Stop channel for convolution (in indices)."""
        stop = int(self._stop.value // self.dt)
        return stop

    @stop.setter
    def stop(self, v: int):
        """Stop channel for convolution (in indices)."""
        self._stop.value = v

    @property
    def irf_start(self) -> int:
        """Start channel for zeroing IRF (in indices)."""
        return int(self._irf_start.value // self.dt)

    @irf_start.setter
    def irf_start(self, v: int):
        """Start channel for zeroing IRF (in indices)."""
        # Convert to numpy array of long integers
        v_array = np.array([v], dtype=np.int64)
        self._irf_start.value = v_array

    @property
    def irf_stop(self) -> int:
        """Stop channel for zeroing IRF (in indices)."""
        stop = int(self._irf_stop.value // self.dt)
        return stop

    @irf_stop.setter
    def irf_stop(self, v: int):
        """Stop channel for zeroing IRF (in indices)."""
        # Convert to numpy array of long integers
        v_array = np.array([v], dtype=np.int64)
        self._irf_stop.value = v_array

    @property
    def rep_rate(self) -> float:
        """Laser repetition rate in MHz."""
        return self._rep.value

    @rep_rate.setter
    def rep_rate(self, v: float):
        """Laser repetition rate in MHz."""
        self._rep.value = float(v)

    @property
    def do_convolution(self) -> bool:
        """Whether convolution with IRF is enabled."""
        return self._do_convolution

    @do_convolution.setter
    def do_convolution(self, v: bool):
        """Whether convolution with IRF is enabled."""
        self._do_convolution = bool(v)

    @property
    def n0(self) -> float:
        """Initial number of excited donor molecules."""
        return self._n0.value

    @n0.setter
    def n0(self, v: float):
        """Initial number of excited donor molecules."""
        self._n0.value = v

    def _process_irf(self, normalize: bool = True) -> chisurf.curve.Curve:
        """Helper method to process IRF with common operations.
        
        This method handles the common operations for both normalized and unnormalized IRF:
        1. Get the IRF from self._irf
        2. Subtract lamp background
        3. Clip negative values
        4. Zero out IRF values outside the specified range
        5. Optionally normalize or scale to original height
        6. Apply timeshift
        
        Args:
            normalize: If True, normalize the IRF (unless truncated). If False, scale to original height.
            verbose: If True, log processing steps.
            
        Returns:
            chisurf.curve.Curve: The processed IRF curve.
        """
        if isinstance(self._irf, chisurf.curve.Curve):
            irf = self._irf
            irf -= self.lamp_background
            irf.y = np.clip(irf.y, 0, None)
        else:
            start_fraction = 0.1
            x = np.copy(self.data.x)
            x_min = np.where(self.data.y > start_fraction * np.max(self.data.y))[0][0]
            loc = self.data.x[x_min]
            # Width is magnitude-only: sign in IRF asymmetry is controlled by
            # kappa/shape (ik), not by width (iw).
            scale = max(abs(float(self._iw.value)), np.finfo(float).eps)
            shape = self._ik.value
            y = chisurf.math.functions.distributions.generalized_normal_distribution(x, loc, scale, shape, True)
            y *= np.sum(self.data.y)
            irf = chisurf.curve.Curve(x=x, y=y)
            irf.y[irf.y < 1] = 0.0

        irf -= self.lamp_background
        irf.y = np.clip(irf.y, 0, None)

        # Zero out the IRF outside the specified range
        irf_start_idx = self.irf_start
        irf_stop_idx = self.irf_stop
        
        logging.debug(f'Zeroing out IRF y-values. Start: {irf_start_idx}, Stop: {irf_stop_idx}, Total: {len(irf.y)}')
            
        if irf_start_idx > 0 or irf_stop_idx < len(irf.y):
            # Create a copy to avoid modifying the original
            irf_y = np.copy(irf.y)
            # Zero out before irf_start
            if irf_start_idx > 0:
                irf_y[:irf_start_idx] = 0.0
                logging.debug(f'Zeroed out IRF from 0 to {irf_start_idx}')
            # Zero out after irf_stop
            if irf_stop_idx < len(irf_y):
                irf_y[irf_stop_idx:] = 0.0
                logging.debug(f'Zeroed out IRF from {irf_stop_idx} to {len(irf_y)}')
            # Create a new curve with the modified y values
            irf = chisurf.curve.Curve(x=irf.x, y=irf_y)
            logging.debug(f'Created new IRF curve with truncated values')
        
        # Handle normalization or scaling
        is_truncated = irf_start_idx > 0 or irf_stop_idx < len(irf.y)
        
        if normalize:
            # Skip normalization if we've truncated the IRF
            if is_truncated:
                logging.debug(f'Skipping normalization for truncated IRF')
            else:
                # Normalize the IRF only if we haven't truncated it
                irf.normalize(mode="sum", inplace=True)
                logging.debug(f'Normalized non-truncated IRF')
        else:
            logging.debug(f'No IRF scaling')
        
        # Apply timeshift
        irf = irf << float(self.timeshift)
        return irf

    @property
    def irf(self) -> chisurf.curve.Curve:
        """Returns the normalized IRF for convolution calculations.
        
        Returns:
            chisurf.curve.Curve: The normalized IRF curve.
        """
        return self._process_irf(normalize=True)

    @property
    def unnormalized_irf(self) -> chisurf.curve.Curve:
        """Returns the IRF at its original height for plotting purposes.
        
        This method is similar to the `irf` property but scales the IRF by the
        `n_photons_irf` factor to restore its original height.
        
        Returns:
            chisurf.curve.Curve: The unnormalized IRF curve.
        """
        return self._process_irf(normalize=False)

    @property
    def _irf(self) -> chisurf.curve.Curve:
        """Stored IRF curve (private)."""
        re = self.__irf
        # if re is None:
        #     x = self.fit.data.x
        #     y = np.zeros_like(self.fit.data.y)
        #     re = chisurf.curve.Curve(x, y)
        return re

    @_irf.setter
    def _irf(self, v: chisurf.curve.Curve):
        """Stored IRF curve (private)."""
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
        """Data curve used for convolution."""
        if self._data is None:
            try:
                return self.fit.data
            except AttributeError:
                return None
        else:
            return self._data

    @data.setter
    def data(self, v: chisurf.data.DataCurve):
        """Data curve used for convolution."""
        self._data = v

    # TODO: needs docstring
    def scale(
            self,
            decay: chisurf.data.DataCurve,
            start: int = None,
            stop: int = None,
            bg: float = 0.0,
            data: np.ndarray = None,
            autoscale: bool = None
    ) -> np.ndarray:
        """Scale the model decay to match experimental data."""
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

    # TODO: needs docstring
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
        """Convolve a lifetime spectrum with the IRF."""
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
        
        # Normalize IRF to unity before convolution
        if np.sum(irf_y) > 0:
            irf_y = irf_y / np.sum(irf_y)
        
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
            logging.debug("------------")
            logging.debug("Convolution:")
            logging.debug("Lifetimes: %s" % data)
            logging.debug("dt: %s" % dt)
            logging.debug("Irf: %s" % irf.name)
            logging.debug("Stop: %s" % stop)
            logging.debug("dt: %s" % dt)
            logging.debug("Convolution mode: %s" % mode)

        decay += (scatter * irf_y)
        return decay

    # TODO: needs docstring
    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            name: str = 'Convolution',
            irf: chisurf.curve.Curve = None,
            **kwargs
    ):
        """Initialize the instance."""
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
            label_text='IRF<sub>start</sub>',
            fixed=True
        )
        self._irf_stop = FittingParameter(
            value=stop,
            name='irf_stop',
            label_text='IRF<sub>stop</sub>',
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

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of convolution/IRF state.

        Only small, non-Qt pieces of state are captured here. All scalar
        parameters that are :class:`FittingParameter` instances are already
        handled by the generic fit_state serializer.
        """

        state: dict = {
            "do_convolution": bool(self.do_convolution),
            "mode": str(getattr(self, "mode", "")),
        }

        # Persist the IRF curve (x/y) if available so TCSPC fits round-trip
        # with their instrument response function.
        try:
            irf = getattr(self, "_irf", None)
            if isinstance(irf, chisurf.curve.Curve):
                x = getattr(irf, "x", None)
                y = getattr(irf, "y", None)
                if x is not None and y is not None:
                    x_arr = np.asarray(x, dtype=float).ravel()
                    y_arr = np.asarray(y, dtype=float).ravel()
                    if x_arr.size and x_arr.size == y_arr.size:
                        state["irf"] = {
                            "x": x_arr.tolist(),
                            "y": y_arr.tolist(),
                        }
            # If we are in the GUI widget subclass, also persist the IRF
            # filename/label shown in the line edit so the user can see
            # which IRF was selected after a project reload.
            le = getattr(self, "lineEdit", None)
            if le is not None:
                try:
                    txt = str(le.text())
                except Exception:
                    txt = ""
                if txt:
                    state["irf_name"] = txt
        except Exception:
            pass

        return state

    def set_state(self, state: dict) -> None:
        """Restore convolution/IRF state from :meth:`get_state` output."""

        if not isinstance(state, dict):
            return

        try:
            self.do_convolution = bool(state.get("do_convolution", self.do_convolution))
        except Exception:
            pass

        mode = state.get("mode")
        if isinstance(mode, str) and mode:
            try:
                self.mode = mode
            except Exception:
                pass

        irf_state = state.get("irf")
        if isinstance(irf_state, dict):
            try:
                x = np.asarray(irf_state.get("x", []), dtype=float)
                y = np.asarray(irf_state.get("y", []), dtype=float)
                if x.size and x.size == y.size:
                    curve = chisurf.curve.Curve(x=x, y=y)
                    # Use the public setter so n0 and lamp background bounds
                    # are updated consistently.
                    self._irf = curve
            except Exception:
                pass

        # Restore IRF filename label in the GUI, if present.
        irf_name = state.get("irf_name")
        if isinstance(irf_name, str) and irf_name:
            le = getattr(self, "lineEdit", None)
            if le is not None:
                try:
                    le.setText(irf_name)
                except Exception:
                    pass
