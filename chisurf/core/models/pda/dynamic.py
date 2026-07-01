from __future__ import annotations

"""Dynamic two-state PDA model (dual-color).

This model describes single-molecule FRET Photon Distribution Analysis of a
system that interconverts between **two conformational states** during the
observation (burst / time window). It is the ChiSurf counterpart of PAM's
``PDAFit`` *dynamic model* (``DynamicSystem = 1``), restricted to two colours.

Physics
-------
Each state ``i`` has a Gaussian inter-dye distance distribution
(``R_i`` ± ``s_i``) and therefore a mean per-photon green (donor-channel)
probability ``pG_i`` obtained from the FRET efficiency and the usual
excitation/emission/crosstalk description (see
:func:`chisurf.core.models.pda.common.green_probability_from_efficiency`).

For a molecule observed over a window, the *fraction of time* ``f`` spent in
state 1 is a random variable whose distribution follows the exact occupation
time of a two-state Markov (telegraph) process with stationary initial
condition. It has two boundary masses (spent the whole window in one state) and
an interior density expressed through modified Bessel functions:

    w(f) = e^{-a f - b (1-f)} [ (p1 b + p2 a) I0(z)
             + sqrt(ab / (f(1-f))) (p1 (1-f) + p2 f) I1(z) ],
    z = 2 sqrt(a b f (1-f)),
    a = k1 T = K (1 - p1),   b = k2 T = K p1,

with steady-state occupancy ``p1`` of state 1 and a single dimensionless
exchange parameter ``K = (k1 + k2) T`` (mean number of transitions per window).

The time-averaged per-photon green probability for a molecule with time
fraction ``f`` is ``pG(f) = f pG1 + (1 - f) pG2`` (equal-brightness
assumption). The resulting amplitude/probability spectrum is handed to
:class:`tttrlib.Pda`.

Limiting behaviour (used as the headless acceptance test):

* ``K -> 0`` (slow exchange): only the boundary masses survive, giving two
  static populations at ``pG1`` and ``pG2`` with weights ``p1``/``p2`` — i.e.
  the static two-state result.
* ``K -> inf`` (fast exchange): the distribution collapses onto ``f = p1``,
  giving a single averaged population at ``p1 pG1 + p2 pG2``.
"""

import numpy as np
import tttrlib
from scipy.special import i0, i1

import chisurf as cs
import chisurf.core.models.tcspc.fret
from chisurf.core.fitting.parameter import FittingParameter, FittingParameterGroup
from chisurf.core.fluorescence.general import distance_to_fret_efficiency
from chisurf.core.math.functions.distributions import normal_distribution
from chisurf.core.models.model import ModelCurve
from chisurf.core.models.pda.common import (
    green_probability_from_efficiency,
    mask_zero_photon_bins,
    pda_1d_residuals_from_s1s2,
)
from chisurf.core.models.pda.nusiance import PdaFretNuisance


class PdaDynamicStates(FittingParameterGroup):
    """Two exchanging states (R, sigma) plus occupancy and exchange rate."""

    def __init__(self, name: str = "pda_dynamic_states", **kwargs):
        """Initialize the two-state parameter group."""
        super().__init__(name=name, **kwargs)
        self._R1 = FittingParameter(value=40.0, name="R1", lb=1.0, ub=200.0, bounds_on=True,
                                    label_text="R<sub>1</sub>")
        self._s1 = FittingParameter(value=6.0, name="s1", lb=0.5, ub=50.0, bounds_on=True,
                                    label_text="s<sub>1</sub>")
        self._R2 = FittingParameter(value=60.0, name="R2", lb=1.0, ub=200.0, bounds_on=True,
                                    label_text="R<sub>2</sub>")
        self._s2 = FittingParameter(value=6.0, name="s2", lb=0.5, ub=50.0, bounds_on=True,
                                    label_text="s<sub>2</sub>")
        self._x1 = FittingParameter(value=0.5, name="x1", lb=0.0, ub=1.0, bounds_on=True,
                                    label_text="x<sub>1</sub>")
        # Dimensionless exchange K = (k1+k2)*T_window (mean transitions/window).
        self._kex = FittingParameter(value=1.0, name="k_ex", lb=0.0, ub=1e4, bounds_on=True,
                                     label_text="K<sub>ex</sub>")

    R1 = property(lambda s: s._R1.value)
    s1 = property(lambda s: s._s1.value)
    R2 = property(lambda s: s._R2.value)
    s2 = property(lambda s: s._s2.value)
    x1 = property(lambda s: float(np.clip(s._x1.value, 0.0, 1.0)))
    k_ex = property(lambda s: max(0.0, float(s._kex.value)))


def two_state_time_fraction_pdf(f: np.ndarray, p1: float, K: float) -> np.ndarray:
    """Interior occupation-time-fraction density of a two-state Markov process.

    Parameters
    ----------
    f : numpy.ndarray
        Time fractions in ``(0, 1)`` spent in state 1.
    p1 : float
        Steady-state occupancy of state 1.
    K : float
        Dimensionless exchange rate ``(k1 + k2) * T``.

    Returns
    -------
    numpy.ndarray
        Unnormalized interior density ``w(f)`` (boundary masses handled
        separately by the caller).
    """
    f = np.asarray(f, dtype=float)
    p2 = 1.0 - p1
    a = K * p2  # k1 * T  (exit rate out of state 1)
    b = K * p1  # k2 * T  (exit rate out of state 2)
    fm = np.clip(f, 1e-9, 1.0 - 1e-9)
    fp = fm
    fn = 1.0 - fm
    z = 2.0 * np.sqrt(a * b * fp * fn)
    with np.errstate(divide="ignore", invalid="ignore"):
        term0 = (p1 * b + p2 * a) * i0(z)
        term1 = np.sqrt(a * b / (fp * fn)) * (p1 * fn + p2 * fp) * i1(z)
    w = np.exp(-a * fp - b * fn) * (term0 + term1)
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


class PdaDynamicTwoStateModel(ModelCurve):
    """Dynamic two-state (dual-color) PDA model."""

    name = "PDA-dynamic-2-state"

    #: Declarative AutoForm layout (PRD-38 model/view-spec split).
    view_spec_file = "dynamic.view.json"

    def __init__(
        self,
        fit: cs.core.fitting.fit.Fit,
        nuisance: PdaFretNuisance | None = None,
        states: PdaDynamicStates | None = None,
        n_grid: int = 41,
        **kwargs,
    ):
        """Initialize the dynamic two-state PDA model.

        Parameters
        ----------
        fit : cs.core.fitting.fit.Fit
            Fit object holding the experimental PDA data.
        nuisance : PdaFretNuisance, optional
            Correction/nuisance parameter group.
        states : PdaDynamicStates, optional
            Two-state distance / occupancy / exchange group.
        n_grid : int
            Number of interior time-fraction grid points.
        **kwargs
            Forwarded to the parent constructor.
        """
        super().__init__(fit, **kwargs)
        self.nuisance = nuisance or PdaFretNuisance(name="pda_fret_nuisance", fit=fit, **kwargs)
        self.states = states or PdaDynamicStates(name="pda_dynamic_states", fit=fit, **kwargs)
        self.fret_parameters = chisurf.core.models.tcspc.fret.FRETParameters(
            enable_fret_efficiency=False
        )
        self.n_grid = int(n_grid)
        kw_pda = {
            "hist2d_nmax": fit.data.pda["maximum_number_of_photons"],
            "hist2d_nmin": fit.data.pda["minimum_number_of_photons"],
            "pF": fit.data.pda["ps"],
        }
        self.pda = tttrlib.Pda(**kw_pda)
        self.residual_mode = "1D"

    # -- helpers ------------------------------------------------------------
    def _mean_green_probability(self, R: float, sigma: float, r, E, pG) -> float:
        """Return the state's Gaussian-averaged per-photon green probability."""
        if sigma <= 0.0:
            e = distance_to_fret_efficiency(np.array([R]), self.fret_parameters.forster_radius)
            return float(green_probability_from_efficiency(e, self.nuisance)[0])
        g = normal_distribution(x=r, loc=float(R), scale=float(sigma), norm=False)
        s = float(np.sum(g))
        if s <= 0.0:
            return 0.5
        return float(np.sum((g / s) * pG))

    def update_model(self, verbose: bool | None = None, **kwargs):
        """Build the two-state dynamic probability spectrum and update the curve."""
        st = self.states
        r = chisurf.core.models.tcspc.fret.rda_axis
        R0 = self.fret_parameters.forster_radius
        E = distance_to_fret_efficiency(r, R0)
        pG = green_probability_from_efficiency(E, self.nuisance)

        pG1 = self._mean_green_probability(st.R1, st.s1, r, E, pG)
        pG2 = self._mean_green_probability(st.R2, st.s2, r, E, pG)

        p1 = st.x1
        p2 = 1.0 - p1
        K = st.k_ex

        # Interior time-fraction grid + exact two-state occupation density.
        f = np.linspace(1e-4, 1.0 - 1e-4, max(3, self.n_grid))
        w = two_state_time_fraction_pdf(f, p1, K)
        df = f[1] - f[0]
        interior_w = w * df
        interior_pG = f * pG1 + (1.0 - f) * pG2

        # Boundary masses: entire window spent in one state.
        a = K * p2
        b = K * p1
        mass_f1 = p1 * float(np.exp(-a))  # f = 1 -> pG1
        mass_f0 = p2 * float(np.exp(-b))  # f = 0 -> pG2

        amps = np.concatenate(([mass_f0], interior_w, [mass_f1]))
        pch1 = np.concatenate(([pG2], interior_pG, [pG1]))

        total = float(np.sum(amps))
        if total > 0.0:
            amps = amps / total

        # Optional donor-only fraction (shares the Gaussian model's semantics).
        xD0 = float(np.clip(self.fret_parameters.xDOnly, 0.0, 1.0))
        if xD0 > 0.0:
            amps = amps * (1.0 - xD0)
            # Donor-only per-photon green probability at E = 0.
            pG_d0 = float(green_probability_from_efficiency(np.array([1e-9]), self.nuisance)[0])
            amps = np.concatenate(([xD0], amps))
            pch1 = np.concatenate(([pG_d0], pch1))

        prob_spectrum = np.empty(amps.size * 2, dtype=np.float64)
        prob_spectrum[0::2] = amps
        prob_spectrum[1::2] = pch1

        try:
            self.nuisance.update_correction_factors()
        except Exception:
            pass

        self.pda.background_ch1 = self.nuisance.BG
        self.pda.background_ch2 = self.nuisance.BR
        self.pda.set_probability_spectrum_ch1(prob_spectrum.tolist())

        s1s2_model = np.asarray(self.pda.s1s2, dtype=float)
        try:
            shp = (getattr(self.fit.data, "pda", None) or {}).get("shape")
            if shp is not None and len(shp) == 2:
                ny, nx = int(shp[0]), int(shp[1])
                s1s2_model = s1s2_model[:ny, :nx]
        except Exception:
            pass

        y = s1s2_model.ravel(order="C")
        total_data = float(np.sum(self.fit.data.y))
        total_model = float(np.sum(y))
        if total_model > 0.0:
            y = y * (total_data / total_model)
        x = np.arange(y.size)
        self.d = np.vstack((x, y))

    def _get_1d_residuals(self, fit) -> np.ndarray:
        """Compute 1D weighted residuals from the S1S2 histogram."""
        wres = pda_1d_residuals_from_s1s2(
            fit=fit, pda_obj=self.pda, nuisance=getattr(self, "nuisance", None)
        )
        try:
            self._last_1d_residual_size = int(wres.size)
        except Exception:
            pass
        return wres

    def get_wres(self, fit, xmin: int | None = None, xmax: int | None = None) -> np.ndarray:
        """Return weighted residuals (1D projection by default)."""
        import chisurf.core.fitting as _fitting

        if getattr(self, "residual_mode", "1D") == "1D":
            return self._get_1d_residuals(fit)
        if xmin is None:
            xmin = fit.xmin
        if xmax is None:
            xmax = fit.xmax
        wres = _fitting.calculate_weighted_residuals(fit.data, self, xmin=xmin, xmax=xmax)
        return mask_zero_photon_bins(fit, xmin, wres)

    @property
    def n_points(self) -> int:
        """Number of data points for chi-squared (1D projection size)."""
        if getattr(self, "residual_mode", "1D") == "1D":
            n = int(getattr(self, "_last_1d_residual_size", 0) or 0)
            if n > 0:
                return n
        return super().n_points
