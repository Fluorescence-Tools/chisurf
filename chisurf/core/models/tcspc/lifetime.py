from __future__ import annotations
from chisurf import typing

import math

import numpy as np

import chisurf.core.curve
import chisurf.core.plot_transforms as plot_transforms
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

    def _lifetime_parameter_rows(self) -> list:
        """Return lifetime parameters interleaved as (amplitude_i, lifetime_i) pairs.

        Used by the data-driven editor's dynamic group so each row pairs an
        amplitude with its lifetime (x1, tau1, x2, tau2, ...).
        """
        rows = []
        for amplitude, lifetime in zip(self._amplitudes, self._lifetimes):
            rows.append(amplitude)
            rows.append(lifetime)
        return rows

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
        # A lifetime model needs at least one component to be valid; seed one
        # here (compute side) so the model is usable headlessly and any view
        # renders a row without the GUI having to bootstrap it.
        if len(self.lifetimes) == 0:
            self.lifetimes.append()

        if convolve is None:
            convolve = Convolve(name='convolve', fit=fit, **kwargs)
        self.convolve = convolve

    #: User-editable editor layout, loaded by Model.view_spec(). Lives next to
    #: this module as lifetime.view.json so the GUI is data-driven, not coded.
    view_spec_file = "lifetime.view.json"

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
    # --- plot-reference overlay modes (relocated from LifetimeModelWidgetBase, PRD-38 4c) ---

    def _tcspc_reference_window(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> np.ndarray:
        """Return the y-window used for photon normalization.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        numpy.ndarray
            Finite y-values used for the denominator.
        """
        y = np.asarray(context.y, dtype=float)
        if not bool(context.parameters.get("fit_range_only", False)):
            return y[np.isfinite(y)]
        try:
            data_x = np.asarray(getattr(getattr(context.fit, "data", None), "x", []), dtype=float)
            if y.size == data_x.size:
                xmin = int(getattr(context.fit, "xmin", 0))
                xmax = int(getattr(context.fit, "xmax", y.size))
                y = y[max(0, xmin):min(y.size, xmax)]
        except Exception:
            pass
        return y[np.isfinite(y)]

    def _tcspc_total_photons_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize TCSPC counts by total photons.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Photon-normalized curve.
        """
        window = self._tcspc_reference_window(context)
        denominator = float(np.nansum(window))
        if not np.isfinite(denominator) or denominator == 0.0:
            raise ValueError("total photon count is zero")
        return plot_transforms.PlotReferenceResult(
            x=context.x,
            y=np.asarray(context.y, dtype=float) / denominator,
            y_label="counts / total photons",
        )

    def _tcspc_peak_photons_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize TCSPC counts by the peak photon count.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Peak-normalized curve.
        """
        window = self._tcspc_reference_window(context)
        if window.size == 0:
            raise ValueError("peak photon count is unavailable")
        denominator = float(np.nanmax(window))
        if not np.isfinite(denominator) or denominator == 0.0:
            raise ValueError("peak photon count is zero")
        return plot_transforms.PlotReferenceResult(
            x=context.x,
            y=np.asarray(context.y, dtype=float) / denominator,
            y_label="counts / peak photons",
        )

    def _donor_reference_curve(self, context: plot_transforms.PlotReferenceContext) -> np.ndarray:
        """Return a donor-reference curve for the current context.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        numpy.ndarray
            Reference curve.
        """
        raw_ref = None
        ref_model = getattr(self, "_reference", None)
        if ref_model is not None:
            try:
                ref_model.update_model()
                raw_ref = np.maximum(np.asarray(ref_model.y, dtype=float), 0.0)
            except Exception:
                raw_ref = None
        if raw_ref is None:
            raw_ref = np.asarray(getattr(self, "reference"), dtype=float)

        scale_mode = str(context.parameters.get("scale", "data_peak"))
        ref = np.asarray(raw_ref, dtype=float).copy()
        peak = float(np.nanmax(ref)) if ref.size else 0.0
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError("donor reference peak is unavailable")
        if scale_mode == "data_peak":
            y_peak = float(np.nanmax(np.asarray(context.y, dtype=float)))
            if np.isfinite(y_peak) and y_peak > 0.0:
                ref *= y_peak / peak
        elif scale_mode == "reference_peak":
            ref /= peak
        return ref

    def _tcspc_donor_reference_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Normalize TCSPC FRET curves by donor-reference decay.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Donor-reference-normalized curve.
        """
        ref = self._donor_reference_curve(context)
        y = np.asarray(context.y, dtype=float)
        x = np.asarray(context.x, dtype=float)
        n = min(y.size, ref.size, x.size)
        if n <= 0:
            raise ValueError("donor reference length is zero")
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(np.abs(ref[:n]) > 1e-15, y[:n] / ref[:n], np.nan)
        return plot_transforms.PlotReferenceResult(
            x=x[:n],
            y=out,
            y_label="counts / donor reference",
        )

    def _plot_anisotropy_widget(self):
        """Return the anisotropy component used for r(t) plotting.

        Returns
        -------
        object or None
            Anisotropy widget/component.
        """
        aniso = getattr(self, "anisotropy", None)
        if aniso is not None:
            return aniso
        for name in ("fret_rates", "fret", "distance_distribution"):
            candidate = getattr(getattr(self, name, None), "anisotropy", None)
            if candidate is not None:
                return candidate
        return None

    @staticmethod
    def _tcspc_rt_curves(t, vv, vh, g: float, l1: float, l2: float):
        """Compute uncorrected and corrected anisotropy curves.

        Parameters
        ----------
        t : array_like
            Time axis.
        vv : array_like
            Parallel channel.
        vh : array_like
            Perpendicular channel.
        g : float
            G-factor.
        l1 : float
            Leakage correction l1.
        l2 : float
            Leakage correction l2.

        Returns
        -------
        tuple
            ``(t, r_uncorrected, r_corrected)``.
        """
        t = np.asarray(t, dtype=float)
        vv = np.asarray(vv, dtype=float)
        vh = np.asarray(vh, dtype=float)
        det = (1.0 - l1) * (1.0 - l2) - l1 * l2
        if abs(det) < 1e-12:
            raise ValueError("anisotropy leakage correction is singular")
        den_unc = g * vv + 2.0 * vh
        with np.errstate(divide="ignore", invalid="ignore"):
            r_unc = np.where(np.abs(den_unc) > 1e-12, (vv - vh) / den_unc, np.nan)
        vv_u = ((1.0 - l2) * vv - l1 * vh) / det
        vh_u = (-l2 * vv + (1.0 - l1) * vh) / det
        den_cor = g * vv_u + 2.0 * vh_u
        with np.errstate(divide="ignore", invalid="ignore"):
            r_cor = np.where(np.abs(den_cor) > 1e-12, (vv_u - vh_u) / den_cor, np.nan)
        finite = np.isfinite(t) & np.isfinite(r_unc) & np.isfinite(r_cor)
        if np.any(finite):
            return t[finite], r_unc[finite], r_cor[finite]
        return t, r_unc, r_cor

    def _tcspc_anisotropy_rt_mode(
            self,
            context: plot_transforms.PlotReferenceContext
    ) -> plot_transforms.PlotReferenceResult:
        """Plot time-resolved anisotropy from VV/VH channels.

        Parameters
        ----------
        context : PlotReferenceContext
            Current plot-transform context.

        Returns
        -------
        PlotReferenceResult
            Anisotropy curve, or hidden result for non-primary group members.
        """
        if context.curve_key not in ("data", "model"):
            return plot_transforms.PlotReferenceResult(context.x, context.y, visible=False)
        if context.group_index is not None and context.selected_group_index is not None:
            if int(context.group_index) != int(context.selected_group_index):
                return plot_transforms.PlotReferenceResult(context.x, context.y, visible=False)

        aniso = self._plot_anisotropy_widget()
        if aniso is None:
            raise ValueError("anisotropy component is unavailable")

        if context.curve_key == "model":
            t, vv, vh = aniso._extract_vv_vh_model_for_diag()
        else:
            t, vv, vh, _defaults = aniso._extract_vv_vh_raw_for_diag()
        if t is None or vv is None or vh is None:
            return plot_transforms.PlotReferenceResult(context.x, context.y, visible=False)

        vv = np.asarray(vv, dtype=float) - float(context.parameters.get("bg_vv", 0.0))
        vh = np.asarray(vh, dtype=float) - float(context.parameters.get("bg_vh", 0.0))
        shift = float(context.parameters.get("vh_shift", 0.0))
        if hasattr(aniso, "_shift_trace_to_reference"):
            vh = aniso._shift_trace_to_reference(np.asarray(t, dtype=float), vh, shift)

        tt, r_unc, r_cor = self._tcspc_rt_curves(
            t=t,
            vv=vv,
            vh=vh,
            g=float(context.parameters.get("g", getattr(aniso, "g", 1.0))),
            l1=float(context.parameters.get("l1", getattr(aniso, "l1", 0.0))),
            l2=float(context.parameters.get("l2", getattr(aniso, "l2", 0.0))),
        )
        variant = str(context.parameters.get("variant", "corrected"))
        y = r_unc if variant == "uncorrected" else r_cor
        return plot_transforms.PlotReferenceResult(x=tt, y=y, y_label="r(t)")

    def _tcspc_anisotropy_defaults(self) -> typing.Dict[str, float]:
        """Return default plot-only anisotropy parameters.

        Returns
        -------
        dict
            Defaults for r(t) correction controls.
        """
        aniso = self._plot_anisotropy_widget()
        defaults = {
            "g": 1.0,
            "l1": 0.0,
            "l2": 0.0,
            "bg_vv": 0.0,
            "bg_vh": 0.0,
            "vh_shift": 0.0,
        }
        if aniso is None:
            return defaults
        for key in ("g", "l1", "l2"):
            try:
                defaults[key] = float(getattr(aniso, key))
            except Exception:
                pass
        try:
            _t, _vv, _vh, diag_defaults = aniso._extract_vv_vh_raw_for_diag()
            if isinstance(diag_defaults, dict):
                defaults["bg_vv"] = float(diag_defaults.get("bg_vv", defaults["bg_vv"]))
                defaults["bg_vh"] = float(diag_defaults.get("bg_vh", defaults["bg_vh"]))
                defaults["vh_shift"] = float(diag_defaults.get("shift_vh", 0.0)) - float(diag_defaults.get("shift_vv", 0.0))
        except Exception:
            pass
        return defaults

    def get_plot_reference_modes(self) -> typing.List[plot_transforms.PlotReferenceMode]:
        """Return TCSPC reference modes for the line plot.

        Returns
        -------
        list
            Plot reference modes.
        """
        fit_range_param = plot_transforms.PlotReferenceParameter(
            key="fit_range_only",
            label="fit range",
            kind="bool",
            default=False,
        )
        modes = [
            plot_transforms.PlotReferenceMode(
                key="tcspc_total_photons",
                label="Total photons",
                callback=self._tcspc_total_photons_mode,
                parameters=(fit_range_param,),
                applies_to=("data", "model"),
                y_label="counts / total photons",
                y_range=(0, 1.0),
                y_padding=0.05,
            ),
            plot_transforms.PlotReferenceMode(
                key="tcspc_peak_photons",
                label="Peak photons",
                callback=self._tcspc_peak_photons_mode,
                parameters=(fit_range_param,),
                applies_to=("data", "model"),
                y_label="counts / peak photons",
                y_range=(0, 1.0),
                y_padding=0.05,
            ),
        ]
        if hasattr(self, "_reference") or hasattr(type(self), "reference"):
            modes.append(
                plot_transforms.PlotReferenceMode(
                    key="tcspc_donor_reference",
                    label="Donor reference",
                    callback=self._tcspc_donor_reference_mode,
                    parameters=(
                        plot_transforms.PlotReferenceParameter(
                            key="scale",
                            label="scale",
                            kind="choice",
                            default="data_peak",
                            choices=(
                                ("data_peak", "data peak"),
                                ("reference_peak", "reference peak"),
                                ("none", "none"),
                            ),
                        ),
                    ),
                    applies_to=("data", "model"),
                    y_label="counts / donor reference",
                    y_range=(0, 1.0),
                    y_padding=0.05,
                )
            )
        if self._plot_anisotropy_widget() is not None:
            defaults = self._tcspc_anisotropy_defaults()
            modes.append(
                plot_transforms.PlotReferenceMode(
                    key="tcspc_anisotropy_rt",
                    label="r(t) anisotropy",
                    callback=self._tcspc_anisotropy_rt_mode,
                    parameters=(
                        plot_transforms.PlotReferenceParameter("g", "g", "float", defaults["g"], step=0.01),
                        plot_transforms.PlotReferenceParameter("l1", "l1", "float", defaults["l1"], step=0.001),
                        plot_transforms.PlotReferenceParameter("l2", "l2", "float", defaults["l2"], step=0.001),
                        plot_transforms.PlotReferenceParameter("bg_vv", "BgVV", "float", defaults["bg_vv"], step=1.0),
                        plot_transforms.PlotReferenceParameter("bg_vh", "BgVH", "float", defaults["bg_vh"], step=1.0),
                        plot_transforms.PlotReferenceParameter("vh_shift", "dVH", "float", defaults["vh_shift"], step=0.01),
                        plot_transforms.PlotReferenceParameter(
                            "variant",
                            "variant",
                            "choice",
                            "corrected",
                            choices=(("corrected", "corrected"), ("uncorrected", "uncorrected")),
                        ),
                    ),
                    applies_to=("data", "model"),
                    y_label="r(t)",
                    y_range=(-0.05, 0.45),
                    y_padding=0.0,
                )
            )
        return modes

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


class LifetimeNewModel(LifetimeModel):
    """Lifetime model whose editor is auto-generated by the new typed-widget
    framework (PRD-38/PRD-40).

    Identical computation to :class:`LifetimeModel`, registered as a *pure*
    (Qt-free) model so the GUI renders its editor via ``AutoModelWidget`` from
    ``lifetime.view.json``. Registered **additively** in the GUI as a separate
    "Lifetime (new)" entry so the framework can be exercised live without
    touching the proven hand-written ``LifetimeModelWidget``. Do not make this
    the primary entry until the framework is signed off.
    """

    name = "Lifetime (new)"


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
    def fractions(self) -> np.ndarray:
        """Normalized mixing fractions of the model instances."""
        if self.n_model > 0:
            v = np.array([x.value for x in self._fractions])
            v /= np.sum(v)
            return v
        else:
            return np.array([1.0], dtype=np.float64)

    @property
    def model_names(self) -> typing.List[str]:
        """Names of the model instances in the mixture."""
        return [f'x{i + 1}' for i in range(len(self._fractions))]

    @property
    def lifetime_spectrum(self) -> np.array:
        """Interleaved (amplitude, lifetime, ...) array."""
        lts = list()
        for model, fraction in zip(self.lifetime_model_instances, self.fractions):
            lt = np.copy(model.lifetime_spectrum)
            lt[::2] *= fraction
            lts.append(lt)
        if len(lts) > 0:
            return np.hstack(lts)
        else:
            return np.array([1., 4.], dtype=np.float64)
