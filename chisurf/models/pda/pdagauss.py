from __future__ import annotations

"""Gaussian-distance PDA models.

This module contains PDA models that use Gaussian distance distributions
for donor–acceptor separations. The core classes
:class:`PdaGaussianDistances` and :class:`PdaGaussianDistanceModel`
were previously defined in :mod:`chisurf.models.pda.simple` and have
been moved here for clarity.
"""

from chisurf import typing

import math
import tttrlib

import numpy as np

import chisurf
import chisurf.math.datatools

from chisurf.fitting.parameter import FittingParameterGroup, FittingParameter
from chisurf.models.model import ModelCurve
from chisurf.models.pda.nusiance import PdaFretNuisance
from chisurf.models.pda.common import mask_zero_photon_bins, pda_1d_residuals_from_s1s2
import chisurf.math.functions.distributions
import chisurf.models.tcspc.fret
from chisurf.fluorescence.general import distance_to_fret_efficiency


class PdaGaussianDistances(FittingParameterGroup):
    """Gaussian distance components for PDA distance models.

    The group stores means, standard deviations and amplitudes for a
    small number of Gaussian distance components. The
    :attr:`distribution` property turns these parameters into a
    normalized distance distribution on the grid defined by
    ``chisurf.models.tcspc.fret.rda_axis``.
    """

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

        # Optional limited-width mode: interpret stored sigmas as
        # percentages of the mean distance, so that for a component with
        # mean R and stored value w we use sigma = (w / 100) * R.
        if getattr(self, "limited_width", False):
            sigmas = (sigmas / 100.0) * means

        def _gauss(x, loc, scale):
            if scale <= 0.0:
                return np.zeros_like(x)
            return chisurf.math.functions.distributions.normal_distribution(
                x=x,
                loc=loc,
                scale=scale,
                norm=False,
            )

        dist_args = [[float(m), float(s)] for m, s in zip(means, sigmas)]
        p = chisurf.math.functions.distributions.combine_distributions(
            x_axis=r,
            dist_function=_gauss,
            dist_args=dist_args,
            weights=amplitudes.tolist(),
            accumulate=True,
            normalize=True,
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
        amplitude: float,
    ):
        n = len(self)
        i = n + 1
        m = FittingParameter(
            value=mean,
            name=f"R(P,{i})",
            lb=0.0,
            ub=100.0,
            bounds_on=True,
        )
        s = FittingParameter(
            value=sigma,
            name=f"s(P,{i})",
            lb=1.0,
            ub=20.0,
            bounds_on=True,
        )
        a = FittingParameter(
            value=amplitude,
            name=f"x(P,{i})",
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
        name: str = "pda_gaussians",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._means = list()
        self._sigmas = list()
        self._amplitudes = list()
        # When True, interpret the stored sigma parameters as *fractions*
        # of the corresponding mean distances in the distribution
        # property above.
        self.limited_width = False


class PdaGaussianDistanceModel(ModelCurve):
    """PDA model with a Gaussian distance distribution."""

    name = "PDA-Gaussian-distance"

    def __str__(self):
        s = super().__str__()
        return s

    def __init__(
        self,
        fit: "chisurf.fitting.fit.Fit",
        nuisance: PdaFretNuisance | None = None,
        distances: PdaGaussianDistances | None = None,
        kw_hist: dict | None = None,
        **kwargs,
    ):
        super().__init__(fit, **kwargs)

        if nuisance is None:
            nuisance = PdaFretNuisance(name="pda_fret_nuisance", fit=fit, **kwargs)
        if distances is None:
            distances = PdaGaussianDistances(name="pda_distances", fit=fit, **kwargs)
        self.nuisance = nuisance
        self.distances = distances

        self.fret_parameters = chisurf.models.tcspc.fret.FRETParameters(
            enable_fret_efficiency=False
        )

        if kw_hist is None:
            kw_hist = {
                "x_max": 500.0,
                "x_min": 0.05,
                "log_x": True,
                "n_bins": 81,
                "n_min": 10,
            }
        self.kw_hist = kw_hist

        kw_pda = {
            "hist2d_nmax": fit.data.pda["maximum_number_of_photons"],
            "hist2d_nmin": fit.data.pda["minimum_number_of_photons"],
            "pF": fit.data.pda["ps"],
        }
        self.pda = tttrlib.Pda(**kw_pda)

        # Default to 1D residuals for PDA Gaussian-distance models.
        self.residual_mode = "1D"

    def update_model(
        self,
        verbose: bool | None = None,
        **kwargs,
    ):
        if verbose is None:
            verbose = chisurf.settings.cs_settings["verbose"]

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

        # Nuisance parameters
        n = self.nuisance
        BG = n.BG
        BR = n.BR
        QYD = n.QYD
        QYA = n.QYA

        # Excitation / emission description via absolute excitation
        # probabilities and a full 2x2 emission/detection matrix g_{channel|species}.
        ExDG = getattr(n, "ExDG", 0.0)
        ExAG = getattr(n, "ExAG", 0.0)
        gGD = getattr(n, "gGD", 0.0)
        gGA = getattr(n, "gGA", 0.0)
        gRD = getattr(n, "gRD", 0.0)
        gRA = getattr(n, "gRA", 0.0)

        eps = 1e-12
        E_safe = np.clip(E, eps, 1.0 - eps)

        ExDG_val = float(ExDG)
        ExAG_val = float(ExAG)
        gGD_val = float(gGD)
        gGA_val = float(gGA)
        gRD_val = float(gRD)
        gRA_val = float(gRA)
        QYD_val = float(QYD)
        QYA_val = float(QYA)

        # DA species: donor excitation weight ExDG·(1-E), acceptor excitation
        # weight ExDG·E + ExAG (direct acceptor excitation). Quantum yields
        # scale the donor/acceptor emission independently.
        S_D = ExDG_val * (1.0 - E_safe)
        S_A = ExDG_val * E_safe + ExAG_val
        S_DQ = QYD_val * S_D
        S_AQ = QYA_val * S_A

        G_DA = gGD_val * S_DQ + gGA_val * S_AQ
        R_DA = gRD_val * S_DQ + gRA_val * S_AQ
        denom = G_DA + R_DA
        with np.errstate(divide="ignore", invalid="ignore"):
            p_G_bound = np.where(denom > 0.0, G_DA / denom, 0.5)

        # Donor-only species: only donor emission contributes, scaled by QYD.
        S_D0 = ExDG_val
        S_D0Q = QYD_val * S_D0
        G_D0 = gGD_val * S_D0Q
        R_D0 = gRD_val * S_D0Q
        denom0 = G_D0 + R_D0
        if denom0 > 0.0:
            p_ch1_d0 = float(G_D0 / denom0)
        else:
            p_ch1_d0 = 0.5

        xD0 = float(self.fret_parameters.xDOnly)
        xD0 = np.clip(xD0, 0.0, 1.0)

        if p_r.size > 0:
            pr_sum = float(np.sum(p_r))
            if pr_sum > 0.0:
                p_r_eff = (1.0 - xD0) * (p_r / pr_sum)
            else:
                p_r_eff = np.zeros_like(p_r, dtype=np.float64)
        else:
            p_r_eff = p_r

        p_ch1_bound = p_G_bound.astype(np.float64)

        n_bound = int(p_r_eff.size)
        prob_spectrum = np.empty((n_bound + 1) * 2, dtype=np.float64)

        prob_spectrum[0] = xD0
        prob_spectrum[1] = p_ch1_d0

        if n_bound > 0:
            prob_spectrum[2::2] = p_r_eff
            prob_spectrum[3::2] = p_ch1_bound
        else:
            prob_spectrum[2::2] = []
            prob_spectrum[3::2] = []

        try:
            chisurf.logging.debug(
                {
                    "model": "PdaGaussianDistanceModel",
                    "len_prob_spectrum": int(len(prob_spectrum)),
                    "first_entries": [float(x) for x in prob_spectrum[:8]],
                    "R0": float(R0),
                    "BG": float(BG),
                    "BR": float(BR),
                    "QYD": float(QYD),
                    "QYA": float(QYA),
                    "E_min": float(E.min()),
                    "E_max": float(E.max()),
                    "pG_bound_min": float(p_G_bound.min()),
                    "pG_bound_max": float(p_G_bound.max()),
                    "xD0": float(xD0),
                    "p_ch1_d0": float(p_ch1_d0),
                }
            )
        except Exception:
            pass

        self.pda.background_ch1 = BG
        self.pda.background_ch2 = BR
        self.pda.set_probability_spectrum_ch1(prob_spectrum.tolist())

        s1s2_model = np.asarray(self.pda.s1s2, dtype=float)
        try:
            pda_meta = getattr(self.fit.data, "pda", None) or {}
            shp = pda_meta.get("shape")
            if shp is not None and len(shp) == 2:
                ny, nx = int(shp[0]), int(shp[1])
                s1s2_model = s1s2_model[:ny, :nx]
        except Exception:
            pass

        y = s1s2_model.ravel(order="C")

        total_data = float(np.sum(self.fit.data.y))
        total_model = float(np.sum(y))
        if total_model > 0.0:
            y *= total_data / total_model

        x = np.arange(y.size)
        self.d = np.vstack((x, y))

    def _get_1d_residuals(
        self,
        fit: "chisurf.fitting.fit.Fit",
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
        fit: "chisurf.fitting.fit.Fit",
        xmin: int | None = None,
        xmax: int | None = None,
    ) -> np.ndarray:
        import chisurf.fitting as _fitting

        mode = getattr(self, "residual_mode", "1D")
        if mode == "1D":
            return self._get_1d_residuals(fit)

        if xmin is None:
            xmin = fit.xmin
        if xmax is None:
            xmax = fit.xmax

        # Base weighted residuals using the standard machinery
        wres = _fitting.calculate_weighted_residuals(
            fit.data,
            self,
            xmin=xmin,
            xmax=xmax,
        )

        masked = wres

        # Apply photon-number mask based on nuisance parameters, if available
        try:
            pda_meta = getattr(fit.data, "pda", None)
            if isinstance(pda_meta, dict):
                row_indices = np.asarray(pda_meta.get("row_indices"), dtype=np.int64)
                col_indices = np.asarray(pda_meta.get("col_indices"), dtype=np.int64)
                if row_indices.size and col_indices.size:
                    n_points = masked.size
                    start = int(max(0, xmin))
                    stop = int(min(start + n_points, row_indices.size))
                    if stop > start:
                        N = row_indices[start:stop] + col_indices[start:stop]

                        # Dataset defaults
                        pda_nmin = int(pda_meta.get("minimum_number_of_photons", 0) or 0)
                        pda_nmax = int(pda_meta.get("maximum_number_of_photons", 0) or 0)
                        if pda_nmax <= 0:
                            pda_nmax = int(N.max()) if N.size > 0 else 0

                        # Nuisance overrides (0 => use dataset default)
                        try:
                            nmin_param = int(round(float(self.nuisance.nPh_min)))
                        except Exception:
                            nmin_param = 0
                        try:
                            nmax_param = int(round(float(self.nuisance.nPh_max)))
                        except Exception:
                            nmax_param = 0

                        if nmin_param != 0 or nmax_param != 0:
                            nmin = nmin_param if nmin_param > 0 else pda_nmin
                            nmax = nmax_param if nmax_param > 0 else pda_nmax
                            if nmax >= nmin:
                                mask = (N >= nmin) & (N <= nmax)
                                mlen = min(mask.size, masked.size)
                                if mlen > 0:
                                    masked = np.array(masked, copy=True)
                                    masked[:mlen][~mask[:mlen]] = 0.0
        except Exception:
            pass

        # Additionally mask out 2D bins with zero experimental photons so they
        # do not contribute to chi2. The PDA DataCurve stores the flattened
        # S1S2 counts in ``fit.data.y``.
        return mask_zero_photon_bins(fit, xmin, masked)

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
        distances = getattr(self, "distances", None)
        try:
            if distances is not None:
                extra["pda_gaussians_n"] = int(len(distances))
        except Exception:
            pass
        return state

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        extra = state.get("extra") or {}
        distances = getattr(self, "distances", None)
        try:
            target_n = extra.get("pda_gaussians_n")
            if distances is not None and target_n is not None:
                target_n = int(target_n)
                while len(distances) < target_n:
                    try:
                        distances.append(mean=5.0, sigma=0.5, amplitude=1.0)
                    except TypeError:
                        distances.append(5.0, 0.5, 1.0)
                while len(distances) > target_n:
                    distances.pop()
        except Exception:
            pass
        super().set_state(state)

