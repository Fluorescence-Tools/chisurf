from __future__ import annotations
import typing

import math

import numpy as np

import chisurf.curve
import chisurf.math.datatools
from chisurf.fitting.parameter import FittingParameterGroup, FittingParameter
from chisurf.models.model import ModelCurve
from chisurf.models.tcspc.nusiance import Generic, Corrections, Convolve
from chisurf.models.tcspc.anisotropy import Anisotropy
from chisurf.fluorescence.general import species_averaged_lifetime, fluorescence_averaged_lifetime


class Lifetime(FittingParameterGroup):

    @property
    def absolute_amplitudes(self) -> bool:
        return self._abs_amplitudes

    @absolute_amplitudes.setter
    def absolute_amplitudes(
            self,
            v: bool
    ):
        self._abs_amplitudes = v

    @property
    def normalize_amplitudes(self) -> bool:
        return self._normalize_amplitudes

    @normalize_amplitudes.setter
    def normalize_amplitudes(
            self,
            v: bool
    ):
        self._normalize_amplitudes = v

    @property
    def species_averaged_lifetime(self) -> float:
        a = self.amplitudes
        a /= a.sum()
        return species_averaged_lifetime(
            chisurf.math.datatools.two_column_to_interleaved(a, self.lifetimes)
        )

    @property
    def fluorescence_averaged_lifetime(self) -> float:
        a = self.amplitudes
        a /= a.sum()
        return fluorescence_averaged_lifetime(
            chisurf.math.datatools.two_column_to_interleaved(a, self.lifetimes)
        )

    @property
    def amplitudes(self) -> np.array:
        vs = np.array([x.value for x in self._amplitudes])
        if self.absolute_amplitudes:
            vs = np.sqrt(vs**2)
        if self.normalize_amplitudes:
            vs /= vs.sum()
        return vs

    @amplitudes.setter
    def amplitudes(
            self,
            vs: typing.List[float]
    ):
        for i, v in enumerate(vs):
            self._amplitudes[i].value = v

    @property
    def lifetimes(self) -> np.array:
        vs = np.array([math.sqrt(x.value ** 2) for x in self._lifetimes])
        for i, v in enumerate(vs):
            self._lifetimes[i].value = v
        return vs

    @lifetimes.setter
    def lifetimes(
            self,
            vs: typing.List[float]
    ):
        for i, v in enumerate(vs):
            self._lifetimes[i].value = v

    @property
    def lifetime_spectrum(self) -> np.array:
        if self._link is None:
            if self._lifetime_spectrum is None:
                return chisurf.math.datatools.two_column_to_interleaved(
                    self.amplitudes,
                    self.lifetimes
                )
            else:
                return self._lifetime_spectrum
        else:
            return self._link.lifetime_spectrum

    @lifetime_spectrum.setter
    def lifetime_spectrum(
            self,
            v: np.array
    ):
        self._lifetime_spectrum = v
        for p in self.parameters_all:
            p.fixed = True

    @property
    def rate_spectrum(self) -> np.array:
        return chisurf.math.datatools.invert_interleaved(
            self.lifetime_spectrum
        )

    @property
    def n(self) -> int:
        return len(self._amplitudes)

    @property
    def link(self) -> chisurf.fitting.parameter.FittingParameter:
        return self._link

    @link.setter
    def link(
            self,
            v: chisurf.fitting.parameter.FittingParameter
    ):
        if isinstance(v, Lifetime) or v is None:
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
            lifetime: float = 4.0,
            lower_bound_amplitude: float = None,
            upper_bound_amplitude: float = None,
            fixed: bool = False,
            bound_on: bool = False,
            lower_bound_lifetime: float = None,
            upper_bound_lifetime: float = None,
            **kwargs
    ):
        n = len(self)
        amplitude = FittingParameter(
            lb=lower_bound_amplitude,
            ub=upper_bound_amplitude,
            value=amplitude,
            name='x%s%i' % (self.short, n + 1),
            fixed=fixed,
            bounds_on=bound_on
        )
        lifetime = FittingParameter(
            lb=lower_bound_lifetime,
            ub=upper_bound_lifetime,
            value=lifetime,
            name='t%s%i' % (self.short, n + 1),
            fixed=fixed,
            bounds_on=bound_on
        )
        self._amplitudes.append(amplitude)
        self._lifetimes.append(lifetime)

    def pop(self) -> typing.Tuple[
        chisurf.fitting.parameter.FittingParameter,
        chisurf.fitting.parameter.FittingParameter
    ]:
        amplitude = self._amplitudes.pop()
        lifetime = self._lifetimes.pop()
        return amplitude, lifetime

    def __init__(
            self,
            short: str = 'L',
            absolute_amplitudes: bool = True,
            normalize_amplitudes: bool = True,
            amplitudes: typing.List[chisurf.fitting.parameter.FittingParameter] = None,
            lifetimes: typing.List[chisurf.fitting.parameter.FittingParameter] = None,
            name: str = 'lifetimes',
            link: FittingParameter = None,
            **kwargs
    ):
        super().__init__(
            name=name,
            **kwargs
        )
        self.short = short
        self._abs_amplitudes = absolute_amplitudes
        self._normalize_amplitudes = normalize_amplitudes
        self._lifetime_spectrum = None
        self._name = name
        self._link = link

        if amplitudes is None:
            amplitudes = list()
        self._amplitudes = amplitudes

        if lifetimes is None:
            lifetimes = list()
        self._lifetimes = lifetimes

    def __len__(self):
        return self.n


class LifetimeModel(ModelCurve):
    """

    """

    name = "Lifetime fit"

    def __str__(self):
        s = super().__str__()
        s += "\nLifetimes"
        s += "\n------------------\n"
        s += "\nAverage Lifetimes:\n"
        s += "<tau>x: %.3f\n<tau>F: %.3f\n" % (
            self.species_averaged_lifetime,
            self.fluorescence_averaged_lifetime
        )
        return s

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            generic: Generic = None,
            corrections: Corrections = None,
            anisotropy: Anisotropy = None,
            lifetimes: Lifetime = None,
            convolve: Convolve = None,
            **kwargs
    ):
        super().__init__(fit, **kwargs)
        if generic is None:
            generic = Generic(name='generic', fit=fit, **kwargs)
        self.generic = generic

        if corrections is None:
            corrections = Corrections(name='corrections', fit=fit, **kwargs)
        self.corrections = corrections

        if anisotropy is None:
            anisotropy = Anisotropy(name='anisotropy', **kwargs)
        self.anisotropy = anisotropy

        if lifetimes is None:
            lifetimes = Lifetime(name='lifetimes', fit=fit, **kwargs)
        self.lifetimes = lifetimes

        if convolve is None:
            convolve = Convolve(name='convolve', fit=fit, **kwargs)
        self.convolve = convolve

    @property
    def species_averaged_lifetime(
            self
    ) -> float:
        return species_averaged_lifetime(self.lifetime_spectrum)

    @property
    def var_lifetime(
            self
    ) -> float:
        lx = self.species_averaged_lifetime
        lf = self.fluorescence_averaged_lifetime
        return lx*(lf-lx)

    @property
    def fluorescence_averaged_lifetime(
            self
    ) -> float:
        return fluorescence_averaged_lifetime(
            self.lifetime_spectrum,
            self.species_averaged_lifetime
        )

    @property
    def lifetime_spectrum(
            self
    ) -> np.array:
        return self.lifetimes.lifetime_spectrum

    def get_curves(
            self
    ) -> typing.Dict[str, chisurf.curve.Curve]:
        d = super().get_curves()
        d['IRF'] = self.convolve.irf
        return d

    def decay(
            self,
            time: np.array
    ) -> np.array:
        amplitudes, lifetimes = chisurf.math.datatools.interleaved_to_two_columns(
            self.lifetime_spectrum
        )
        return np.array([np.dot(amplitudes, np.exp(- t / lifetimes)) for t in time])

    def update_model(
            self,
            shift_bg_with_irf: bool = None,
            lifetime_spectrum: np.array = None,
            scatter: float = None,
            verbose: bool = chisurf.verbose,
            background: float = None,
            background_curve: chisurf.curve.Curve = None,
            **kwargs
    ):
        if lifetime_spectrum is None:
            lifetime_spectrum = self.lifetime_spectrum
        if scatter is None:
            scatter = self.generic.scatter
        if background is None:
            background = self.generic.background
        if shift_bg_with_irf is None:
            shift_bg_with_irf = chisurf.settings.cs_settings['tcspc']['shift_bg_with_irf']
        if background_curve is None:
            background_curve = self.generic.background_curve

        lifetime_spectrum = self.anisotropy.get_decay(
            lifetime_spectrum
        )
        decay = self.convolve.convolve(
            lifetime_spectrum,
            verbose=verbose,
            scatter=scatter,
            **kwargs
        )

        # Calculate background curve from reference measurement
        if isinstance(
                background_curve,
                chisurf.curve.Curve
        ):
            if shift_bg_with_irf:
                background_curve = background_curve << self.convolve.timeshift

            bg_y = np.copy(background_curve.y)
            bg_y /= bg_y.sum()
            bg_y *= self.generic.n_ph_bg

            decay *= self.generic.n_ph_fl
            decay += bg_y

        self.convolve.scale(
            decay,
            bg=self.generic.background
        )
        self.corrections.pileup(decay)
        decay += background
        decay = self.corrections.linearize(decay)
        self.y = np.maximum(decay, 0)

