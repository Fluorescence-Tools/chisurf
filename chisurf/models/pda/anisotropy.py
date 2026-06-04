from __future__ import annotations

import numpy as np
import tttrlib

import chisurf
import chisurf.math.datatools

from chisurf.models.model import ModelCurve
from chisurf.fitting.parameter import FittingParameterGroup, FittingParameter
from chisurf.models.pda.common import mask_zero_photon_bins, pda_1d_residuals_from_s1s2


class PdaAnisotropyNuisance(FittingParameterGroup):
    """Nuisance parameters for anisotropy-PDA.

    Keeps backgrounds in the two polarization channels and
    optical calibration parameters G, l1, l2.
    """

    def __init__(self, name: str = "pda_aniso_nuisance", **kwargs):
        """Initialize the anisotropy nuisance parameter group.

        Parameters
        ----------
        name : str
            Name of the parameter group.
        **kwargs
            Forwarded to the parent constructor.
        """
        super().__init__(name=name, **kwargs)

        # Backgrounds per time window (parallel / perpendicular)
        self.B_par = FittingParameter(
            value=1.0,
            name="B_par",
            lb=0.0,
            ub=1e6,
            bounds_on=True,
        )
        self.B_perp = FittingParameter(
            value=1.0,
            name="B_perp",
            lb=0.0,
            ub=1e6,
            bounds_on=True,
        )
        # Detection ratio and mixing factors (fixed or very tightly bounded)
        self.G = FittingParameter(
            value=1.0,
            name="G",
            lb=0.1,
            ub=10.0,
            bounds_on=True,
            label_text="g<sub>perp</sub>/g<sub>par</sub>",
        )
        self.l1 = FittingParameter(
            value=0.0,
            name="l1",
            lb=0.0,
            ub=0.33,
            bounds_on=True,
        )
        self.l2 = FittingParameter(
            value=0.0,
            name="l2",
            lb=0.0,
            ub=0.33,
            bounds_on=True,
        )

        try:
            FittingParameterGroup.find_parameters(self)
        except Exception:
            pass


class PdaAnisotropySpecies(FittingParameterGroup):
    """Discrete anisotropy species (amplitude + r_i) for anisotropy-PDA."""

    @property
    def amplitudes(self) -> np.ndarray:
        """Return normalized per-species amplitudes."""
        try:
            vs = np.array([p.value for p in self._amplitudes], dtype=float)
        except Exception:
            return np.zeros(0, dtype=np.float64)
        # Optional absolute and normalization semantics mirroring ProbCh0
        if getattr(self, "_abs_amplitudes", True):
            vs = np.sqrt(vs ** 2)
        if getattr(self, "_normalize_amplitudes", True):
            s = float(np.sum(vs))
            if s != 0.0:
                vs /= s
        return vs

    @amplitudes.setter
    def amplitudes(self, vs) -> None:
        """Set per-species amplitudes from an iterable of floats."""
        try:
            values = list(vs)
        except Exception:
            return
        for i, v in enumerate(values):
            try:
                self._amplitudes[i].value = float(v)
            except Exception:
                continue

    @property
    def anisotropies(self) -> np.ndarray:
        """Per-species anisotropies r_i."""
        try:
            rs = np.array([p.value for p in self._anisotropies], dtype=float)
            return rs
        except Exception:
            return np.zeros(0, dtype=np.float64)

    def finalize(self) -> None:
        """Synchronize internal amplitudes with any normalization rules."""
        amps = self.amplitudes
        try:
            for i, p in enumerate(self._amplitudes):
                p.value = amps[i]
        except Exception:
            pass

    def append(
        self,
        amplitude: float = 1.0,
        r: float = 0.3,
        lower_bound_amplitude: float = 0.0,
        upper_bound_amplitude: float = 1.0,
        lower_bound_r: float = -0.2,
        upper_bound_r: float = 0.5,
        fixed: bool = False,
        bounds_on: bool = False,
        **kwargs,
    ) -> None:
        """Append a new anisotropy species (amplitude, r)."""

        n = len(self)
        i = n + 1
        amp_param = FittingParameter(
            lb=lower_bound_amplitude,
            ub=upper_bound_amplitude,
            value=amplitude,
            name=f"xA{i}",
            label_text=f"x<sub>A,{i}</sub>",
            fixed=fixed,
            bounds_on=bounds_on,
        )
        r_param = FittingParameter(
            lb=lower_bound_r,
            ub=upper_bound_r,
            value=r,
            name=f"rA{i}",
            label_text=f"r<sub>A,{i}</sub>",
            fixed=fixed,
            bounds_on=bounds_on,
        )
        self._amplitudes.append(amp_param)
        self._anisotropies.append(r_param)

    def pop(self):
        """Remove the last anisotropy species, if any."""
        try:
            amp = self._amplitudes.pop()
            r = self._anisotropies.pop()
            return amp, r
        except Exception:
            return None, None

    def __len__(self) -> int:
        """Return the number of anisotropy species."""
        try:
            return len(self._amplitudes)
        except Exception:
            return 0

    def build_probability_spectrum(self, G: float, l1: float, l2: float) -> np.ndarray:
        """Return interleaved (amplitude, p_par, ...) spectrum for tttrlib.Pda."""

        amps = self.amplitudes
        rs = self.anisotropies
        if amps.size == 0 or rs.size == 0:
            return np.zeros(0, dtype=np.float64)

        p_pars = []
        for r in rs:
            p_pars.append(
                anisotropy_to_p_parallel(
                    float(r),
                    float(G),
                    float(l1),
                    float(l2),
                )
            )
        p_pars = np.asarray(p_pars, dtype=np.float64)

        try:
            spec = chisurf.math.datatools.two_column_to_interleaved(amps, p_pars)
        except Exception:
            spec = np.zeros(0, dtype=np.float64)
        return spec

    def __init__(self, name: str = "pda_aniso_species", **kwargs):
        """Initialize the anisotropy species group.

        Parameters
        ----------
        name : str
            Name of the parameter group.
        **kwargs
            Forwarded to the parent constructor.
        """
        super().__init__(name=name, **kwargs)
        self._amplitudes: list[FittingParameter] = []
        self._anisotropies: list[FittingParameter] = []
        # Amplitude handling flags (mirroring ProbCh0 semantics)
        self._abs_amplitudes = True
        self._normalize_amplitudes = True


def anisotropy_to_p_parallel(r: float, G: float, l1: float, l2: float) -> float:
    """Kalinin et al. 2007 eq. (4) – mapping anisotropy to p_parallel.

    Parameters
    ----------
    r : float
        Fluorescence anisotropy.
    G : float
        Detection ratio g_perp/g_par.
    l1, l2 : float
        Polarization mixing parameters.
    """
    r = float(r)
    G = float(G)
    l1 = float(l1)
    l2 = float(l2)

    num = 1.0 + r * (2.0 - 3.0 * l1)
    den = num + G - G * r * (1.0 - 3.0 * l2)
    if den <= 0.0:
        return 0.5

    eps = 1e-12
    return float(np.clip(num / den, eps, 1.0 - eps))


class PdaAnisotropyModel(ModelCurve):
    """PDA model for single-species fluorescence anisotropy.

    Uses :class:`tttrlib.Pda` with channel-1 = parallel, channel-2 =
    perpendicular. The underlying S1S2 matrix is generated by
    :mod:`tttrlib` from experimental photon streams and then converted
    to an anisotropy distribution either as a 2D S_par/S_perp map or a
    1D histogram via :func:`pda_1d_residuals_from_s1s2`.
    """

    name = "PDA-anisotropy"

    def __init__(
        self,
        fit: "chisurf.fitting.fit.Fit",
        nuisance: PdaAnisotropyNuisance | None = None,
        species: "PdaAnisotropySpecies | None" = None,
        kw_hist: dict | None = None,
        **kwargs,
    ):
        """Initialize the anisotropy PDA model.

        Parameters
        ----------
        fit : chisurf.fitting.fit.Fit
            The fit object holding experimental data.
        nuisance : PdaAnisotropyNuisance, optional
            Nuisance parameter group.
        species : PdaAnisotropySpecies, optional
            Anisotropy species parameter group.
        kw_hist : dict, optional
            Histogram settings for 1D residuals.
        **kwargs
            Forwarded to the parent constructor.
        """
        super().__init__(fit, **kwargs)

        if nuisance is None:
            nuisance = PdaAnisotropyNuisance(
                name="pda_aniso_nuisance",
                fit=fit,
                **kwargs,
            )
        self.nuisance = nuisance

        if species is None:
            species = PdaAnisotropySpecies(
                name="pda_aniso_species",
                fit=fit,
                **kwargs,
            )
            # Ensure at least one default anisotropy species.
            try:
                species.append(amplitude=1.0, r=0.3)
            except TypeError:
                try:
                    species.append()
                except Exception:
                    pass
        self.species = species

        # Histogram / PDA settings from the dataset
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
            # For anisotropy-PDA, pF is still the fluorescence intensity
            # distribution extracted from the data (same as in FRET-PDA).
            "pF": fit.data.pda["ps"],
        }
        self.pda = tttrlib.Pda(**kw_pda)

        # 1D residuals are usually preferred (histogram of r or r_exp)
        self.residual_mode = "1D"

    def update_model(self, verbose: bool | None = None, **kwargs):
        """Update the model curve from current nuisance and species parameters.

        Parameters
        ----------
        verbose : bool, optional
            Whether to log debug output.
        **kwargs
            Forwarded to the parent method.
        """
        if verbose is None:
            verbose = chisurf.settings.cs_settings["verbose"]

        # --- 1) Read nuisance params and anisotropy species -----------
        G = float(self.nuisance.G.value)
        l1 = float(self.nuisance.l1.value)
        l2 = float(self.nuisance.l2.value)
        B_par = float(self.nuisance.B_par.value)
        B_perp = float(self.nuisance.B_perp.value)

        species = getattr(self, "species", None)

        # --- 2) Build channel-1 (parallel) probability spectrum -------
        prob_spectrum = np.zeros(0, dtype=np.float64)
        try:
            if species is not None:
                prob_spectrum = species.build_probability_spectrum(
                    G=G,
                    l1=l1,
                    l2=l2,
                )
        except Exception:
            prob_spectrum = np.zeros(0, dtype=np.float64)

        if prob_spectrum.size == 0:
            # Fallback: single default species at r = 0.3 with unit amplitude
            p_par = anisotropy_to_p_parallel(0.3, G, l1, l2)
            prob_spectrum = np.array([1.0, p_par], dtype=np.float64)

        # --- 3) Configure tttrlib.Pda --------------------------------
        self.pda.background_ch1 = B_par
        self.pda.background_ch2 = B_perp
        self.pda.set_probability_spectrum_ch1(prob_spectrum.tolist())

        # 2D model S_parallel / S_perp distribution
        s1s2_model = np.asarray(self.pda.s1s2, dtype=float)

        # When the data P(S∥, S⊥) has been flattened into fit.data.y,
        # we need to match the shape from the metadata (same as in
        # FRET PDA).
        try:
            pda_meta = getattr(self.fit.data, "pda", None) or {}
            shp = pda_meta.get("shape")
            if shp is not None and len(shp) == 2:
                ny, nx = int(shp[0]), int(shp[1])
                s1s2_model = s1s2_model[:ny, :nx]
        except Exception:
            pass

        y = s1s2_model.ravel(order="C")

        # Normalize total counts of model to experimental counts
        total_data = float(np.sum(self.fit.data.y))
        total_model = float(np.sum(y))
        if total_model > 0.0:
            y *= total_data / total_model

        x = np.arange(y.size)
        self.d = np.vstack((x, y))

    # --- Residuals (can reuse the existing PDA machinery) -------------

    def _get_1d_residuals(self, fit: "chisurf.fitting.fit.Fit") -> np.ndarray:
        """Return 1D weighted residuals for the anisotropy histogram.

        The helper :func:`pda_1d_residuals_from_s1s2` constructs a 1D
        histogram from the 2D S_par/S_perp distributions and returns
        chi-square-style residuals between experiment and model.
        """
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
        """Compute weighted residuals for the anisotropy PDA model.

        Parameters
        ----------
        fit : chisurf.fitting.fit.Fit
            The fit object.
        xmin : int, optional
            Start index for the residual window.
        xmax : int, optional
            End index for the residual window.

        Returns
        -------
        numpy.ndarray
            Weighted residuals.
        """
        import chisurf.fitting as _fitting

        mode = getattr(self, "residual_mode", "1D")
        if mode == "1D":
            return self._get_1d_residuals(fit)

        # 2D residuals mode – identical in spirit to the FRET PDA models
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

        # Apply photon-number mask and zero-photon bin mask as in FRET PDA
        masked = wres
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
                        pda_nmin = int(pda_meta.get("minimum_number_of_photons", 0) or 0)
                        pda_nmax = int(pda_meta.get("maximum_number_of_photons", 0) or 0)
                        if pda_nmax <= 0:
                            pda_nmax = int(N.max()) if N.size > 0 else 0

                        mask = (N >= pda_nmin) & (N <= pda_nmax)
                        mlen = min(mask.size, masked.size)
                        if mlen > 0:
                            masked = np.array(masked, copy=True)
                            masked[:mlen][~mask[:mlen]] = 0.0
        except Exception:
            pass

        return mask_zero_photon_bins(fit, xmin, masked)

    @property
    def n_points(self) -> int:
        """Return the number of data points for chi-squared calculation."""
        mode = getattr(self, "residual_mode", "1D")
        if mode == "1D":
            try:
                n = int(getattr(self, "_last_1d_residual_size", 0) or 0)
            except Exception:
                n = 0
            if n > 0:
                return n
        return super().n_points
