from __future__ import annotations

from pygments.styles import vs

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


class ProbCh0(FittingParameterGroup):

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

    def update_model(
            self,
            pch0: np.array = None,
            verbose: bool = chisurf.verbose,
            **kwargs
    ):
        self.pda.background_ch1 = self.background.bg0
        self.pda.background_ch2 = self.background.bg1
        p = self.pch0.pch0_spectrum
        self.pda.set_probability_spectrum_ch1(p)
        # Use upper left triangle for fitting
        row_indices = self.fit.data.pda['row_indices']
        col_indices = self.fit.data.pda['col_indices']
        y = self.pda.s1s2[row_indices, col_indices]
        y *= np.sum(self.fit.data.y) / y.sum()
        x = np.arange(len(y))
        self.d = np.vstack((x, y))
