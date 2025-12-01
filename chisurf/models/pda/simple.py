from __future__ import annotations

"""PDA model classes for fluorescence photon distribution analysis.

This module contains discrete and Gaussian-distance PDA models that
operate on S1S2 histograms produced by :mod:`tttrlib` and the
``chisurf.experiments.pda.PdaReader``. The models are built from
small :class:`~chisurf.fitting.parameter.FittingParameterGroup`
containers such as :class:`ProbCh0`, and
 :class:`Background`.

Only a subset of the functionality is exercised in doctests; all
examples avoid real TTTR files and heavy computation.
"""

from chisurf import typing

import math
import tttrlib

import numpy as np

import chisurf
import chisurf.plots
import chisurf.curve
import chisurf.math.datatools

from chisurf.fitting.parameter import FittingParameterGroup, FittingParameter
from chisurf.models.model import ModelCurve
from chisurf.models.pda.nusiance import Background
from .common import mask_zero_photon_bins, pda_1d_residuals_from_s1s2


class ProbCh0(FittingParameterGroup):

    """Probability and channel-0 parameters for discrete PDA species.

    The group holds per-species amplitudes and ``pch0`` (probability of
    a photon going to channel 0). Amplitudes can be forced to be
    positive (``absolute_amplitudes``) and normalized to sum to one
    (``normalize_amplitudes``). The convenience property
    :attr:`pch0_spectrum` exposes the interleaved format expected by
    :class:`tttrlib.Pda`.

    Examples
    --------
    Create a small two-species probability group and inspect the
    normalized amplitudes and probability spectrum::

        >>> from chisurf.models.pda.simple import ProbCh0
        >>> p = ProbCh0()
        >>> p.append(amplitude=1.0, pch0=0.25)
        >>> p.append(amplitude=3.0, pch0=0.75)
        >>> a = p.amplitudes
        >>> round(float(a.sum()), 7)
        1.0
        >>> spec = p.pch0_spectrum
        >>> spec.size
        4
        >>> spec[0] + spec[2]
        1.0
    """

    @property
    def absolute_amplitudes(self) -> bool:
        return self._abs_amplitudes

    @absolute_amplitudes.setter
    def absolute_amplitudes(self, v: bool):
        self._abs_amplitudes = v

    @property
    def normalize_amplitudes(self) -> bool:
        return self._normalize_amplitudes

    @normalize_amplitudes.setter
    def normalize_amplitudes(self, v: bool):
        self._normalize_amplitudes = v

    @property
    def amplitudes(self) -> np.array:
        vs = np.array([x.value for x in self._amplitudes])
        if self.absolute_amplitudes:
            vs = np.sqrt(vs**2)
        if self.normalize_amplitudes:
            vs /= abs(vs.sum())
        return vs

    @amplitudes.setter
    def amplitudes(self, vs: typing.List[float]):
        for i, v in enumerate(vs):
            self._amplitudes[i].value = v

    @property
    def pch0(self) -> np.array:
        vs = np.array([math.sqrt(x.value ** 2) for x in self._pch0])
        for i, v in enumerate(vs):
            self._pch0[i].value = v
        return vs

    @pch0.setter
    def pch0(self, vs: typing.List[float]):
        for i, v in enumerate(vs):
            self._pch0[i].value = v

    @property
    def pch0_spectrum(self) -> np.array:
        if self._link is None:
            return chisurf.math.datatools.two_column_to_interleaved(
                self.amplitudes,
                self.pch0
            )
        else:
            return self._link.pch0

    @pch0_spectrum.setter
    def pch0_spectrum(self, v: np.array):
        for i in range(len(v) // 2):
            self._amplitudes[2 * i + 0].value = v[2 * i + 0]
            self._pch0[2 * i + 1].value = v[2 * i + 1]

    @property
    def n(self) -> int:
        return len(self._amplitudes)

    @property
    def link(self) -> chisurf.fitting.parameter.FittingParameter:
        return self._link

    @link.setter
    def link(self, v: chisurf.fitting.parameter.FittingParameter):
        if isinstance(v, ProbCh0) or v is None:
            self._link = v

    def update(self):
        amplitudes = self.amplitudes
        for i, a in enumerate(self._amplitudes):
            a.value = amplitudes[i]

    def finalize(self):
        self.update()

    def append(
            self,
            amplitude: float = 1.0,
            pch0: float = 0.1,
            lower_bound_amplitude: float = 0.0,
            upper_bound_amplitude: float = 1.0,
            fixed: bool = False,
            bound_on: bool = True,
            lower_bound_pch0: float = 0.000001,
            upper_bound_pch0: float = 0.999999,
            **kwargs
    ):
        n = len(self)
        i = n + 1
        amplitude = FittingParameter(
            lb=lower_bound_amplitude,
            ub=upper_bound_amplitude,
            value=amplitude,
            name=f'x{self.short}{i}',
            label_text=f'x<sub>{self.short},{i}</sub>',
            fixed=fixed,
            bounds_on=bound_on
        )
        pch0 = FittingParameter(
            lb=lower_bound_pch0,
            ub=upper_bound_pch0,
            value=pch0,
            name=f't{self.short}{i}',
            label_text=f'p<sub>{self.short},{i}</sub>',
            fixed=fixed,
            bounds_on=bound_on
        )
        self._amplitudes.append(amplitude)
        self._pch0.append(pch0)

    def pop(self) -> typing.Tuple[
        chisurf.fitting.parameter.FittingParameter,
        chisurf.fitting.parameter.FittingParameter
    ]:
        amplitude = self._amplitudes.pop()
        lifetime = self._pch0.pop()
        return amplitude, lifetime

    def __init__(
            self,
            short: str = '0',
            absolute_amplitudes: bool = True,
            normalize_amplitudes: bool = True,
            amplitudes: typing.List[chisurf.fitting.parameter.FittingParameter] = None,
            pch0: typing.List[chisurf.fitting.parameter.FittingParameter] = None,
            name: str = 'pch0',
            link: FittingParameter = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.short = short
        self._abs_amplitudes = absolute_amplitudes
        self._normalize_amplitudes = normalize_amplitudes
        self._pch0 = None
        self._name = name
        self._link = link

        if amplitudes is None:
            amplitudes = list()
        self._amplitudes = amplitudes

        if pch0 is None:
            pch0 = list()
        self._pch0 = pch0

    def __len__(self):
        return self.n


class PdaSimpleModel(ModelCurve):
    """Discrete-species PDA model working on S1S2 histograms.

    This model combines a :class:`Background` group with a
    :class:`ProbCh0` group describing discrete species with
    probabilities and channel-0 fractions. The underlying
    :class:`tttrlib.Pda` instance is configured from the attached
    ``fit.data.pda`` metadata.

    Notes
    -----
    The full model requires experimental PDA metadata and a working
    :mod:`tttrlib` installation. A minimal usage sketch is::

        >>> from chisurf.models.pda.simple import PdaSimpleModel  # doctest: +SKIP
        >>> m = PdaSimpleModel(fit)  # doctest: +SKIP
        >>> m.update_model()         # doctest: +SKIP
    """

    name = "PDA-discrete"

    def __str__(self):
        s = super().__str__()
        return s

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            background: Background = None,
            pch0: ProbCh0 = None,
            kw_hist: dict = None,
            **kwargs
    ):
        super().__init__(fit, **kwargs)

        if background is None:
            background = Background(name='background', fit=fit, **kwargs)
        if pch0 is None:
            pch0 = ProbCh0(name='pCh0', fit=fit, **kwargs)
        self.background = background
        self.pch0 = pch0

        if kw_hist is None:
            kw_hist = {
                "x_max": 500.0,
                "x_min": 0.05,
                "log_x": True,
                "n_bins": 81,
                "n_min": 10
            }
        self.kw_hist = kw_hist

        kw_pda = {
            "hist2d_nmax": fit.data.pda['maximum_number_of_photons'],
            "hist2d_nmin": fit.data.pda['minimum_number_of_photons'],
            "pF": fit.data.pda['ps']
        }
        self.pda = tttrlib.Pda(**kw_pda)
        self.residual_mode = "1D"

    def update_model(
            self,
            pch0: np.array = None,
            verbose: bool = None,
            **kwargs
    ):
        if verbose is None:
            verbose = chisurf.settings.cs_settings['verbose']
        self.pda.background_ch1 = self.background.bg0
        self.pda.background_ch2 = self.background.bg1
        p = self.pch0.pch0_spectrum
        # Debug: log probability spectrum passed to tttrlib for discrete PDA
        try:
            chisurf.logging.debug(
                {
                    'model': 'PdaSimpleModel',
                    'len_prob_spectrum': int(len(p)),
                    'first_entries': [float(x) for x in p[:8]],
                    'BG': float(self.background.bg0),
                    'BR': float(self.background.bg1),
                }
            )
        except Exception:
            pass
        self.pda.set_probability_spectrum_ch1(p)
        # Use upper left triangle for fitting
        row_indices = self.fit.data.pda['row_indices']
        col_indices = self.fit.data.pda['col_indices']
        y = self.pda.s1s2[row_indices, col_indices]
        y *= np.sum(self.fit.data.y) / y.sum()
        x = np.arange(len(y))
        self.d = np.vstack((x, y))

    def _get_1d_residuals(
            self,
            fit: chisurf.fitting.fit.Fit,
    ) -> np.ndarray:
        wres = pda_1d_residuals_from_s1s2(
            fit=fit,
            pda_obj=self.pda,
            nuisance=getattr(self, "nuisance", None),
        )
        try:
            self._last_1d_residual_size = int(wres.size)
        except Exception:
            pass
        return wres

    def get_wres(
            self,
            fit: chisurf.fitting.fit.Fit,
            xmin: int = None,
            xmax: int = None
    ) -> np.ndarray:
        import chisurf.fitting as _fitting

        mode = getattr(self, "residual_mode", "1D")
        if mode == "1D":
            return self._get_1d_residuals(fit)

        if xmin is None:
            xmin = fit.xmin
        if xmax is None:
            xmax = fit.xmax

        wres = _fitting.calculate_weighted_residuals(
            fit.data,
            self,
            xmin=xmin,
            xmax=xmax,
        )

        return mask_zero_photon_bins(fit, xmin, wres)

    @property
    def n_points(self) -> int:
        mode = getattr(self, "residual_mode", "1D")
        if mode == "1D":
            try:
                n = int(getattr(self, "_last_1d_residual_size", 0) or 0)
            except Exception:
                n = 0
            if n > 0:
                return n
        return super().n_points

    def get_state(self) -> dict:
        state = super().get_state()
        if not isinstance(state, dict):
            state = {}
        extra = state.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            state["extra"] = extra
        pch0 = getattr(self, "pch0", None)
        try:
            if pch0 is not None:
                extra["pda_probch0_n"] = int(len(pch0))
        except Exception:
            pass
        return state

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        extra = state.get("extra") or {}
        pch0 = getattr(self, "pch0", None)
        try:
            target_n = extra.get("pda_probch0_n")
            if pch0 is not None and target_n is not None:
                target_n = int(target_n)
                while len(pch0) < target_n:
                    try:
                        pch0.append(amplitude=1.0, pch0=0.1)
                    except TypeError:
                        pch0.append()
                while len(pch0) > target_n:
                    pch0.pop()
        except Exception:
            pass
        super().set_state(state)

# Backwards-compatible re-exports of Gaussian-distance PDA models.
from .pdagauss import PdaGaussianDistances, PdaGaussianDistanceModel
