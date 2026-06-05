from __future__ import annotations
from chisurf import typing

import math

import numpy as np

import chisurf.core.curve
import chisurf.core.math.datatools
from chisurf import logging
from chisurf.core.fitting.parameter import FittingParameterGroup, FittingParameter
from chisurf.core.models.model import ModelCurve
from chisurf.core.models.tcspc.nusiance import Generic, Corrections, Convolve
from chisurf.core.models.tcspc.anisotropy import Anisotropy
from chisurf.core.fluorescence.general import species_averaged_lifetime, fluorescence_averaged_lifetime


class Lifetime(FittingParameterGroup):

    @property
    def absolute_amplitudes(self) -> bool:
        """Whether absolute amplitude values are used."""
        return self._abs_amplitudes

    @absolute_amplitudes.setter
    def absolute_amplitudes(self, v: bool):
        """Whether absolute amplitude values are used."""
        self._abs_amplitudes = v

    @property
    def normalize_amplitudes(self) -> bool:
        """Whether amplitudes are normalized to sum to one."""
        return self._normalize_amplitudes

    @normalize_amplitudes.setter
    def normalize_amplitudes(self, v: bool):
        """Whether amplitudes are normalized to sum to one."""
        self._normalize_amplitudes = v

    @property
    def species_averaged_lifetime(self) -> float:
        """Species-averaged (amplitude-weighted) lifetime <tau>x."""
        a = self.amplitudes
        a /= a.sum()
        return species_averaged_lifetime(
            chisurf.core.math.datatools.two_column_to_interleaved(a, self.lifetimes)
        )

    @property
    def fluorescence_averaged_lifetime(self) -> float:
        """Fluorescence-averaged (intensity-weighted) lifetime <tau>F."""
        a = self.amplitudes
        a /= a.sum()
        return fluorescence_averaged_lifetime(
            chisurf.core.math.datatools.two_column_to_interleaved(a, self.lifetimes)
        )

    @property
    def amplitudes(self) -> np.array:
        """Array of amplitude values."""
        vs = np.array([x.value for x in self._amplitudes])
        if self.absolute_amplitudes:
            vs = np.sqrt(vs**2)
        if self.normalize_amplitudes:
            vs /= abs(vs.sum())
        return vs

    @amplitudes.setter
    def amplitudes(self, vs: typing.List[float]):
        """Array of amplitude values."""
        for i, v in enumerate(vs):
            self._amplitudes[i].value = v

    @property
    def lifetimes(self) -> np.array:
        """Array of lifetime values (always positive)."""
        vs = np.array([math.sqrt(x.value ** 2) for x in self._lifetimes])
        for i, v in enumerate(vs):
            self._lifetimes[i].value = v
        return vs

    @lifetimes.setter
    def lifetimes(self, vs: typing.List[float]):
        """Array of lifetime values (always positive)."""
        for i, v in enumerate(vs):
            self._lifetimes[i].value = v

    @property
    def lifetime_spectrum(self) -> np.array:
        """Interleaved (amplitude, lifetime, ...) array."""
        if self._link is None:
            if self._lifetime_spectrum is None:
                return chisurf.core.math.datatools.two_column_to_interleaved(
                    self.amplitudes,
                    self.lifetimes
                )
            else:
                return self._lifetime_spectrum
        else:
            return self._link.lifetime_spectrum

    @lifetime_spectrum.setter
    def lifetime_spectrum(self, v: np.array):
        """Interleaved (amplitude, lifetime, ...) array."""
        self._lifetime_spectrum = v
        for p in self.parameters_all:
            p.fixed = True

    @property
    def rate_spectrum(self) -> np.array:
        """Interleaved (amplitude, rate, ...) array (inverse of lifetimes)."""
        return chisurf.core.math.datatools.invert_interleaved(
            self.lifetime_spectrum
        )

    @property
    def n(self) -> int:
        """Number of exponential components."""
        try:
            amplitudes = getattr(self, "_amplitudes", None)
            if amplitudes is not None:
                return len(amplitudes)
        except Exception:
            pass
            
        # Fallback: infer component count from parameter names only if the
        # internal lists are missing (can happen after partial GUI construction).
        params = getattr(self, "parameters_all_dict", {}) or {}
        if not params:
            try:
                self.find_parameters()
                params = getattr(self, "parameters_all_dict", {}) or {}
            except Exception:
                params = {}
        short = getattr(self, "short", "")
        x_prefix = f"x{short}"
        t_prefix = f"t{short}"
        n_x = len([name for name in params if name.startswith(x_prefix)])
        n_t = len([name for name in params if name.startswith(t_prefix)])
        return max(n_x, n_t, 0)

    @property
    def link(self) -> chisurf.core.fitting.parameter.FittingParameter:
        """Linked Lifetime object for spectrum sharing."""
        return self._link

    @link.setter
    def link(self, v: chisurf.core.fitting.parameter.FittingParameter):
        """Linked Lifetime object for spectrum sharing."""
        if isinstance(v, Lifetime) or v is None:
            self._link = v

    # TODO: needs docstring
    def update(self):
        """Update the state and emit signals."""
        amplitudes = self.amplitudes
        for i, a in enumerate(self._amplitudes):
            a.value = amplitudes[i]

    # TODO: needs docstring
    def finalize(self):
        """Finalize the component state."""
        self.update()

    # TODO: needs docstring
    def append(
            self,
            amplitude: float = 1.0,
            lifetime: float = 4.0,
            lower_bound_amplitude: float = 0.0,
            upper_bound_amplitude: float = 1.0,
            fixed: bool = False,
            bound_on: bool = False,
            lower_bound_lifetime: float = 0.001,
            upper_bound_lifetime: float = 100.0,
            **kwargs
    ):
        """Add a new component."""
        n = len(self)
        i = n + 1
        amplitude = FittingParameter(
            lb=lower_bound_amplitude,
            ub=upper_bound_amplitude,
            value=amplitude,
            name=f'x{self.short}{i}',
            label_text=f'x<sub>{self.short}, {i}</sub>',
            fixed=fixed,
            bounds_on=bound_on
        )
        lifetime = FittingParameter(
            lb=lower_bound_lifetime,
            ub=upper_bound_lifetime,
            value=lifetime,
            name=f't{self.short}{i}',
            label_text=f'&tau;<sub>{self.short},{i}</sub>',
            fixed=fixed,
            bounds_on=bound_on
        )
        self._amplitudes.append(amplitude)
        self._lifetimes.append(lifetime)
        if getattr(self, "_parameters", None) is not None:
            self.append_parameter(amplitude)
            self.append_parameter(lifetime)

    # TODO: needs docstring
    def pop(self) -> typing.Tuple[
        chisurf.core.fitting.parameter.FittingParameter,
        chisurf.core.fitting.parameter.FittingParameter
    ]:
        """Remove the last component."""
        amplitude = self._amplitudes.pop()
        lifetime = self._lifetimes.pop()
        if getattr(self, "_parameters", None) is not None:
            self._parameters = [
                p for p in self._parameters 
                if p is not amplitude and p is not lifetime
            ]
        return amplitude, lifetime

    # TODO: needs docstring
    def __init__(
            self,
            short: str = 'L',
            absolute_amplitudes: bool = True,
            normalize_amplitudes: bool = True,
            amplitudes: typing.List[chisurf.core.fitting.parameter.FittingParameter] = None,
            lifetimes: typing.List[chisurf.core.fitting.parameter.FittingParameter] = None,
            name: str = 'lifetimes',
            link: FittingParameter = None,
            **kwargs
    ):
        """Initialize the instance."""
        super().__init__(name=name, **kwargs)
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
        """Return the number of components."""
        return self.n


class LifetimeModel(ModelCurve):

    name = "Lifetime "

    def __str__(self):
        """Return a string representation."""
        s = super().__str__()
        s += "\nLifetimes"
        s += "\n------------------\n"
        s += "\nAverage Lifetimes:\n"
        s += (
            f"<tau>x: {self.species_averaged_lifetime:.3f}\n"
            f"<tau>F: {self.fluorescence_averaged_lifetime:.3f}\n"
            f"Steady state anisotropy: {self.steady_state_anisotropy:.3f}\n"
        )
        s += "\nAnisotropy"
        s += "\n------------------\n"
        s += f"Steady state anisotropy: {self.steady_state_anisotropy:.3f}\n"
        return s

    # TODO: needs docstring
    def __init__(
            self,
            fit: chisurf.core.fitting.fit.Fit,
            generic: Generic = None,
            corrections: Corrections = None,
            anisotropy: Anisotropy = None,
            lifetimes: Lifetime = None,
            convolve: Convolve = None,
            **kwargs
    ):
        """Initialize the instance."""
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
        # Automatically set polarization type for fits
        logging.info("Checking for polarization type setup.")
        # Use the unified method to set polarization based on group position
        polarization_set = self.anisotropy.set_polarization_by_group_position(fit, self)
        if polarization_set:
            logging.info(f"Polarization type set to {self.anisotropy.polarization_type}")
        if anisotropy is None:
            anisotropy = Anisotropy(name='anisotropy', **kwargs)
        self.anisotropy = anisotropy

        if lifetimes is None:
            # Preserve an existing lifetimes object (e.g. injected by a widget)
            # instead of overwriting it with a fresh instance.
            existing = getattr(self, "lifetimes", None)
            if isinstance(existing, Lifetime):
                lifetimes = existing
            else:
                lifetimes = Lifetime(name='lifetimes', fit=fit, **kwargs)
        self.lifetimes = lifetimes

        if convolve is None:
            convolve = Convolve(name='convolve', fit=fit, **kwargs)
        self.convolve = convolve

    @property
    def species_averaged_lifetime(self) -> float:
        """Species-averaged (amplitude-weighted) lifetime <tau>x."""
        return species_averaged_lifetime(self.lifetime_spectrum)

    @property
    def var_lifetime(self) -> float:
        """Variance of the lifetime distribution."""
        lx = self.species_averaged_lifetime
        lf = self.fluorescence_averaged_lifetime
        return lx*(lf-lx)

    @property
    def fluorescence_averaged_lifetime(self) -> float:
        """Fluorescence-averaged (intensity-weighted) lifetime <tau>F."""
        return fluorescence_averaged_lifetime(
            self.lifetime_spectrum,
            self.species_averaged_lifetime
        )

    @property
    def steady_state_anisotropy(self) -> float:
        """Steady-state anisotropy from lifetime and rotation spectra."""
        ls = self.lifetime_spectrum
        rs = self.anisotropy.rotation_spectrum
        lrs = chisurf.core.math.datatools.elte2(ls, rs)
        lrx, lrt = chisurf.core.math.datatools.interleaved_to_two_columns(lrs)
        lx, lt = chisurf.core.math.datatools.interleaved_to_two_columns(ls)
        nom = lrx @ lrt
        denom = lx @ lt
        if denom == 0:
            return float("NAN")
        return nom / denom

    @property
    def lifetime_spectrum(self) -> np.array:
        """Interleaved (amplitude, lifetime, ...) array."""
        return self.lifetimes.lifetime_spectrum

    # TODO: needs docstring
    def get_curves(self, copy_curves: bool = False) -> typing.Dict[str, chisurf.core.curve.Curve]:
        """Return a dictionary of curves for plotting."""
        d = super().get_curves(copy_curves)
        # Use unnormalized IRF for plotting to display it at its original height
        d['IRF'] = self.convolve.unnormalized_irf
        return d

    # TODO: needs docstring
    def decay(self, time: np.array) -> np.array:
        """Compute the fluorescence decay."""
        amplitudes, lifetimes = chisurf.core.math.datatools.interleaved_to_two_columns(
            self.lifetime_spectrum
        )
        return np.array([np.dot(amplitudes, np.exp(- t / lifetimes)) for t in time])

    # TODO: needs docstring
    def update_model(
            self,
            shift_bg_with_irf: bool = None,
            lifetime_spectrum: np.array = None,
            scatter: float = None,
            verbose: bool = None,
            background: float = None,
            background_curve: chisurf.core.curve.Curve = None,
            **kwargs
    ):
        """Recompute the model decay."""
        if verbose is None:
            verbose = chisurf.core.settings.cs_settings['verbose']
        if lifetime_spectrum is None:
            lifetime_spectrum = self.lifetime_spectrum
        if scatter is None:
            scatter = self.generic.scatter
        if background is None:
            background = self.generic.background
        if shift_bg_with_irf is None:
            shift_bg_with_irf = chisurf.core.settings.cs_settings['tcspc']['shift_bg_with_irf']
        if background_curve is None:
            background_curve = self.generic.background_curve

        lifetime_spectrum = self.anisotropy.get_decay(lifetime_spectrum)
        decay = self.convolve.convolve(
            lifetime_spectrum,
            verbose=verbose,
            scatter=scatter,
            **kwargs
        )

        # Calculate background curve from reference measurement
        if isinstance(background_curve, chisurf.core.curve.Curve):

            if shift_bg_with_irf:
                background_curve = background_curve << self.convolve.timeshift

            bg_y = np.copy(background_curve.y)
            bg_y *= self.generic.n_ph_bg / bg_y.sum()
            decay *= self.generic.n_ph_fl / decay.sum()

            decay += bg_y

        self.corrections.pileup(decay)
        self.convolve.scale(decay, bg=self.generic.background)
        decay += background
        decay = self.corrections.linearize(decay)
        self.y = np.maximum(decay, 0)

    # TODO: needs docstring
    def get_state(self) -> dict:
        """Return a JSON-serializable state snapshot."""
        state = super().get_state()
        if not isinstance(state, dict):
            state = {}
        extra = state.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            state["extra"] = extra
        lifetimes = getattr(self, "lifetimes", None)
        try:
            if lifetimes is not None:
                extra["lifetimes_n"] = int(len(lifetimes))
        except Exception:
            pass
        return state

    # TODO: needs docstring
    def set_state(self, state: dict) -> None:
        """Restore state from a JSON-serializable snapshot."""
        if not isinstance(state, dict):
            return
        extra = state.get("extra") or {}
        lifetimes = getattr(self, "lifetimes", None)
        try:
            target_n = extra.get("lifetimes_n")
            if lifetimes is not None and target_n is not None:
                target_n = int(target_n)
                while True:
                    try:
                        current_n = int(len(lifetimes))
                    except Exception:
                        break
                    if current_n >= target_n:
                        break
                    try:
                        lifetimes.append()
                    except TypeError:
                        try:
                            lifetimes.append(amplitude=1.0, lifetime=4.0)
                        except Exception:
                            break
                try:
                    while len(lifetimes) > target_n:
                        lifetimes.pop()
                except Exception:
                    pass
        except Exception:
            pass
        super().set_state(state)


class LifetimeMixtureModel(LifetimeModel):

    name = "Lifetime mixer"

    @property
    def lifetime_fits(self) -> typing.List[chisurf.core.fitting.fit.Fit]:
        """List of fits with LifetimeModel instances (excluding self)."""
        return [
            s for s in chisurf.fits
            if isinstance(s, chisurf.core.fitting.fit.Fit) and isinstance(s.model, LifetimeModel) and s.model is not self
        ]

    # TODO: needs docstring
    def __init__(
            self,
            fit: chisurf.core.fitting.fit.Fit,
            **kwargs
    ):
        """Initialize the instance."""
        super().__init__(fit, **kwargs)
        self.lifetime_model_instances: typing.List[LifetimeModel] = list()
        self._fractions: typing.List[FittingParameter] = list()

    # TODO: needs docstring
    def append_model(self, model_instance: LifetimeModel, name: str = None):
        """Append a LifetimeModel instance with a mixing fraction."""
        self.lifetime_model_instances.append(model_instance)
        if name is None:
            name = 'x(' + model_instance.fit.name + ')'
        self._fractions.append(
            FittingParameter(
                value=1.0,
                name=name,
                bounds_on=True,
                lb=0.0, ub=1.0
            )
        )

    # TODO: needs docstring
    def pop_model(self, idx: int = None):
        """Remove a model instance by index."""
        if idx is None:
            m = self.lifetime_model_instances.pop()
            f = self._fractions.pop()
            return f, m
        else:
            m = self.lifetime_model_instances[idx]
            f = self._fractions[idx]
            self.lifetime_model_instances = self.lifetime_model_instances[:idx - 1] + self.lifetime_model_instances[idx + 1:]
            self._fractions = self._fractions[:idx] + self._fractions[idx + 1:]
            return f, m

    @property
    def n_model(self):
        """Number of model instances in the mixture."""
        return len(self.lifetime_model_instances)

    @property
    def factions(self) -> np.ndarray:
        """Normalized mixing fractions of the model instances."""
        if self.n_model > 0:
            v = np.array([x.value for x in self._fractions])
            v /= np.sum(v)
            return v
        else:
            return np.array([1.0], dtype=np.float64)

    @property
    def lifetime_spectrum(self) -> np.array:
        """Interleaved (amplitude, lifetime, ...) array."""
        lts = list()
        for model, fraction in zip(self.lifetime_model_instances, self.factions):
            lt = np.copy(model.lifetime_spectrum)
            lt[::2] *= fraction
            lts.append(lt)
        if len(lts) > 0:
            return np.hstack(lts)
        else:
            return np.array([1., 4.], dtype=np.float64)
