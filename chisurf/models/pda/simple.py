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
from chisurf.models.pda.nusiance import Background, PdaFretNuisance
import chisurf.math.functions.distributions
import chisurf.models.tcspc.fret
from chisurf.fluorescence.general import distance_to_fret_efficiency


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


class PdaGaussianDistances(FittingParameterGroup):

    @property
    def means(self) -> np.array:
        try:
            return np.array([p.value for p in self._means])
        except AttributeError:
            return np.array([])

    @property
    def sigmas(self) -> np.array:
        try:
            return np.array([p.value for p in self._sigmas])
        except AttributeError:
            return np.array([])

    @property
    def amplitudes(self) -> np.array:
        try:
            a = np.sqrt(np.array([p.value for p in self._amplitudes]) ** 2)
            s = a.sum()
            if s > 0.0:
                a /= s
            return a
        except AttributeError:
            return np.array([])

    @property
    def distribution(self) -> np.array:
        means = self.means
        sigmas = self.sigmas
        amplitudes = self.amplitudes
        if means.size == 0:
            return np.zeros((2, 0), dtype=np.float64)
        r = chisurf.models.tcspc.fret.rda_axis

        def _gauss(x, loc, scale):
            if scale <= 0.0:
                return np.zeros_like(x)
            return chisurf.math.functions.distributions.normal_distribution(
                x=x,
                loc=loc,
                scale=scale,
                norm=False
            )

        dist_args = [[float(m), float(s)] for m, s in zip(means, sigmas)]
        p = chisurf.math.functions.distributions.combine_distributions(
            x_axis=r,
            dist_function=_gauss,
            dist_args=dist_args,
            weights=amplitudes.tolist(),
            accumulate=True,
            normalize=True
        )
        return np.vstack((r, p))

    def finalize(self):
        amplitudes = self.amplitudes
        for i, p in enumerate(self._amplitudes):
            p.value = amplitudes[i]

    def append(
            self,
            mean: float,
            sigma: float,
            amplitude: float
    ):
        n = len(self)
        i = n + 1
        m = FittingParameter(
            value=mean,
            name=f'R(P,{i})'
        )
        s = FittingParameter(
            value=sigma,
            name=f's(P,{i})'
        )
        a = FittingParameter(
            value=amplitude,
            name=f'x(P,{i})'
        )
        self._means.append(m)
        self._sigmas.append(s)
        self._amplitudes.append(a)

    def pop(self):
        self._means.pop()
        self._sigmas.pop()
        self._amplitudes.pop()

    def __len__(self):
        return len(self._amplitudes)

    def __init__(
            self,
            name: str = 'pda_gaussians',
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._means = list()
        self._sigmas = list()
        self._amplitudes = list()


class PdaGaussianDistanceModel(ModelCurve):

    name = "PDA-Gaussian-distance"

    def __str__(self):
        s = super().__str__()
        return s

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            nuisance: PdaFretNuisance = None,
            distances: PdaGaussianDistances = None,
            kw_hist: dict = None,
            **kwargs
    ):
        super().__init__(fit, **kwargs)

        if nuisance is None:
            nuisance = PdaFretNuisance(name='pda_fret_nuisance', fit=fit, **kwargs)
        if distances is None:
            distances = PdaGaussianDistances(name='pda_distances', fit=fit, **kwargs)
        self.nuisance = nuisance
        self.distances = distances

        # FRET parameters (R0, tau0, kappa2, etc.) shared with TCSPC FRET models
        # Use defaults from chisurf.settings.fret but expose them as fitting
        # parameters so R0 and, if desired, kappa2 can be adjusted. For PDA we
        # disable the additional E_FRET fitting parameter as it is not used.
        self.fret_parameters = chisurf.models.tcspc.fret.FRETParameters(
            enable_fret_efficiency=False
        )

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
            verbose: bool = None,
            **kwargs
    ):
        if verbose is None:
            verbose = chisurf.settings.cs_settings['verbose']

        dist = self.distances.distribution
        if dist.shape[1] == 0:
            x = np.arange(self.fit.data.y.size)
            y = np.zeros_like(x, dtype=np.float64)
            self.d = np.vstack((x, y))
            return

        # distance distribution (bound species)
        r = dist[0]
        p_r = dist[1]

        # FRET efficiency from distance for bound species
        R0 = self.fret_parameters.forster_radius
        E = distance_to_fret_efficiency(r, R0)

        # nuisance parameters
        alpha = self.nuisance.alpha      # spectral crosstalk (as in eq. 6: red leak / green)
        BG    = self.nuisance.BG
        BR    = self.nuisance.BR
        gG    = self.nuisance.gG
        gR    = self.nuisance.gR
        QYD   = self.nuisance.QYD       # donor QY (Φ_FD(0))
        QYA   = self.nuisance.QYA       # acceptor QY (Φ_FA)

        # gamma from eq. 6: γ = g_R Φ_FA / (g_G Φ_FD(0))
        # here Φ_FA ≈ QYA, Φ_FD(0) ≈ QYD
        gamma = (gR * QYA) / (gG * QYD)

        # avoid division by zero for E -> 1
        eps = 1e-12
        E_safe = np.clip(E, 0.0 + eps, 1.0 - eps)

        # p_G as in the paper for bound species: p_G = (1 + α + γ E / (1-E))^-1
        p_G_bound = 1.0 / (1.0 + alpha + gamma * E_safe / (1.0 - E_safe))

        # donor-only fraction xDOnly: treat as separate species with p_G≈0.99999
        xD0 = float(self.fret_parameters.xDOnly)
        xD0 = np.clip(xD0, 0.0, 1.0)

        # effective amplitudes for bound species are renormalized to (1 - xD0)
        # while their *shape* follows p_r. We keep the same r grid, but scale
        # the overall weight of the bound ensemble.
        if p_r.size > 0:
            # normalize p_r to 1 and then scale to (1 - xD0)
            pr_sum = float(np.sum(p_r))
            if pr_sum > 0.0:
                p_r_eff = (1.0 - xD0) * (p_r / pr_sum)
            else:
                p_r_eff = np.zeros_like(p_r, dtype=np.float64)
        else:
            p_r_eff = p_r

        # channel-1 probability for bound species
        p_ch1_bound = p_G_bound.astype(np.float64)

        # donor-only species: no acceptor, so nearly all photons go to channel 1
        # Use a tiny red leak so tttrlib gets a finite p_ch1 strictly < 1.
        p_ch1_d0 = 0.99999

        # Construct combined probability spectrum:
        # [xD0, p_ch1_d0, p_r_eff0, p_ch1_bound0, p_r_eff1, p_ch1_bound1, ...]
        n_bound = int(p_r_eff.size)
        prob_spectrum = np.empty((n_bound + 1) * 2, dtype=np.float64)

        # first entry: donor-only species
        prob_spectrum[0] = xD0
        prob_spectrum[1] = p_ch1_d0

        # remaining entries: bound distance species (renormalized to 1 - xD0)
        if n_bound > 0:
            prob_spectrum[2::2] = p_r_eff
            prob_spectrum[3::2] = p_ch1_bound
        else:
            # if no bound species, just donor-only
            prob_spectrum[2::2] = []
            prob_spectrum[3::2] = []

        # Debug: log probability spectrum and key parameters for Gaussian PDA
        try:
            chisurf.logging.debug(
                {
                    'model': 'PdaGaussianDistanceModel',
                    'len_prob_spectrum': int(len(prob_spectrum)),
                    'first_entries': [float(x) for x in prob_spectrum[:8]],
                    'R0': float(R0),
                    'alpha': float(alpha),
                    'BG': float(BG),
                    'BR': float(BR),
                    'gG': float(gG),
                    'gR': float(gR),
                    'QYD': float(QYD),
                    'QYA': float(QYA),
                    'E_min': float(E.min()),
                    'E_max': float(E.max()),
                    'pG_bound_min': float(p_G_bound.min()),
                    'pG_bound_max': float(p_G_bound.max()),
                    'xD0': float(xD0),
                    'p_ch1_d0': float(p_ch1_d0),
                }
            )
        except Exception:
            pass

        # set backgrounds and probability spectrum in PDA object
        self.pda.background_ch1 = BG
        self.pda.background_ch2 = BR
        self.pda.set_probability_spectrum_ch1(prob_spectrum.tolist())

        # compute model histogram on the same bins as the data
        row_indices = self.fit.data.pda['row_indices']
        col_indices = self.fit.data.pda['col_indices']
        y = self.pda.s1s2[row_indices, col_indices]

        # normalize total model counts to total data counts
        total_data = np.sum(self.fit.data.y)
        total_model = y.sum()
        if total_model > 0.0:
            y *= total_data / total_model

        x = np.arange(len(y))
        self.d = np.vstack((x, y))
