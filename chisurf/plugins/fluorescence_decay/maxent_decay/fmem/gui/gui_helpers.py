from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .qt_stack import ensure_qt_stack


class _MaxentHelpersMixin:
    def _build_tau_grid(self) -> np.ndarray:
        tmin = float(self.spin_tau_min.value())
        tmax = float(self.spin_tau_max.value())
        bins = int(self.spin_tau_bins.value())
        if tmax <= tmin or bins < 2:
            raise ValueError("Invalid tau grid parameters")
        return np.linspace(tmin, tmax, bins, dtype=float)

    def _get_fitrange_arg(self, n: int) -> Optional[Tuple[int, int]]:
        fitrange = self._fit_range
        if fitrange is None:
            fit = self._current_fit()
            if fit is not None:
                try:
                    xmin = int(getattr(fit, "xmin", 0))
                    xmax = int(getattr(fit, "xmax", n - 1))
                    fitrange = (xmin, xmax)
                except Exception:
                    fitrange = None

        if fitrange is None or n <= 0:
            return None
        fs, fe = int(fitrange[0]), int(fitrange[1])
        if fe < fs:
            fe = fs
        fs = max(0, min(fs, n - 1))
        fe = max(fs, min(fe, n - 1))
        return (fs, fe)

    def _build_fret_r_axis(self, R0_val: float) -> np.ndarray:
        r_min_frac = float(self.spin_R_min.value())
        r_max_frac = float(self.spin_R_max.value())
        n_points = int(self.spin_R_points.value())
        if n_points < 2:
            n_points = 2
        if r_max_frac <= r_min_frac:
            r_max_frac = r_min_frac + 1e-3
        return np.linspace(
            r_min_frac * float(R0_val),
            r_max_frac * float(R0_val),
            n_points,
            dtype=float,
        )

    def _get_fret_donor_spectrum(self, tau0_val: float) -> np.ndarray:
        if self._donly_vec is not None:
            return self._donly_vec
        return np.array([1.0, float(tau0_val)], dtype=float)

    def _get_dist_prior_for_axis(self, r_axis: np.ndarray) -> Optional[np.ndarray]:
        if self._dist_prior_vec is None:
            return None
        if self._dist_prior_vec.size != np.asarray(r_axis, dtype=float).ravel().size:
            return None
        return self._dist_prior_vec

    def _get_period_arg(self, use_periodic: bool) -> Optional[float]:
        if not use_periodic:
            return None
        return max(float(self.spin_period.value()), 1e-3)

    def _get_nuisance_settings(
        self,
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[float], Optional[float]]:
        optimize_nuisance = bool(self.chk_fit_nuisance.isChecked())

        fix_ts = bool(
            getattr(self, "chk_fix_timeshift", None) is not None
            and self.chk_fix_timeshift.isChecked()
        )
        fix_bg = bool(
            getattr(self, "chk_fix_background", None) is not None
            and self.chk_fix_background.isChecked()
        )
        fix_irf_bg = bool(
            getattr(self, "chk_fix_irf_bg", None) is not None
            and self.chk_fix_irf_bg.isChecked()
        )
        fix_x_donly = bool(
            getattr(self, "chk_fix_x_donly", None) is not None
            and self.chk_fix_x_donly.isChecked()
        )

        nuisance_step_timeshift = 0.0 if fix_ts else None
        nuisance_step_background = 0.0 if fix_bg else None
        nuisance_step_irf_background = 0.0 if fix_irf_bg else None
        nuisance_step_x_donly = 0.0 if fix_x_donly else None

        return (
            optimize_nuisance,
            nuisance_step_timeshift,
            nuisance_step_background,
            nuisance_step_irf_background,
            nuisance_step_x_donly,
        )

    def _get_decay_and_dt(self) -> Tuple[np.ndarray, float, np.ndarray]:
        fit = self._current_fit()
        if fit is None or getattr(fit, "data", None) is None:
            raise RuntimeError("No current fit / data available")
        data = fit.data
        y = np.asarray(getattr(data, "y", []), dtype=float).ravel()
        if y.size == 0:
            raise RuntimeError("Current data has no counts")
        try:
            x = np.asarray(getattr(data, "x", None), dtype=float).ravel()
        except Exception:
            x = np.zeros_like(y)
        if x.size != y.size or x.size == 0:
            try:
                dx = float(np.asarray(data.dx).ravel()[0])
            except Exception:
                dx = 1.0
            x = dx * np.arange(y.size, dtype=float)
        else:
            dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
        self._t_axis = x
        return y, dx, x

    def _build_irf_array(self, n: int, t: np.ndarray, dt: float) -> np.ndarray:
        if self._irf_dataset is not None:
            try:
                x_irf = np.asarray(getattr(self._irf_dataset, "x", []), dtype=float).ravel()
                y_irf = np.asarray(getattr(self._irf_dataset, "y", []), dtype=float).ravel()
            except Exception:
                x_irf = np.zeros(0, dtype=float)
                y_irf = np.zeros(0, dtype=float)
            if x_irf.size > 1 and y_irf.size == x_irf.size:
                order = np.argsort(x_irf)
                x_sorted = x_irf[order]
                y_sorted = y_irf[order]
                lamp = np.interp(t, x_sorted, y_sorted, left=0.0, right=0.0)
                return lamp.astype(float)

        fit = self._current_fit()
        if fit is not None:
            try:
                model = fit.model
                if hasattr(model, "convolve"):
                    convolve = model.convolve
                    irf_curve = None
                    if hasattr(convolve, "unnormalized_irf"):
                        irf_curve = convolve.unnormalized_irf
                    elif hasattr(convolve, "irf"):
                        irf_curve = convolve.irf
                    if irf_curve is not None:
                        y_irf = np.asarray(irf_curve.y, dtype=float).ravel()
                        if y_irf.size > 0:
                            lamp = np.resize(y_irf, n)
                            return lamp
            except Exception:
                pass

        lamp = np.zeros(n, dtype=float)
        lamp[0] = 1.0
        return lamp

    def _estimate_irf_fwhm(self, t: np.ndarray, lamp: np.ndarray) -> Optional[float]:
        t_arr = np.asarray(t, dtype=float).ravel()
        lamp_arr = np.asarray(lamp, dtype=float).ravel()
        if t_arr.size != lamp_arr.size or t_arr.size < 3:
            return None
        max_val = float(np.max(lamp_arr))
        if not np.isfinite(max_val) or max_val <= 0.0:
            return None
        half = 0.5 * max_val
        above = np.nonzero(lamp_arr >= half)[0]
        if above.size < 2:
            if t_arr.size >= 2:
                dt_local = float(np.mean(np.diff(t_arr)))
                return dt_local if dt_local > 0.0 else None
            return None
        fwhm = float(t_arr[above[-1]] - t_arr[above[0]]) / 2.0
        if fwhm <= 0.0 and t_arr.size >= 2:
            dt_local = float(np.mean(np.diff(t_arr)))
            fwhm = dt_local
        return fwhm if fwhm > 0.0 else None

    def _plot_decay_and_irf(
        self,
        decay: np.ndarray,
        t: np.ndarray,
        lamp: Optional[np.ndarray],
    ) -> None:
        ensure_qt_stack()

        self.plot_decay.clear()
        if getattr(self, "_fit_region", None) is not None:
            self.plot_decay.addItem(self._fit_region)

        decay_plot = np.maximum(np.asarray(decay, dtype=float).ravel(), 1.0)

        irf_plot = None
        if lamp is not None:
            lamp_arr = np.asarray(lamp, dtype=float).ravel()
            if lamp_arr.size == decay_plot.size and np.any(lamp_arr > 0.0):
                scale = float(np.max(decay_plot)) / float(np.max(lamp_arr))
                irf_plot = lamp_arr * scale

        self.plot_decay.plot(t, decay_plot, pen="w", name="data")
        if irf_plot is not None:
            self.plot_decay.plot(t, np.maximum(irf_plot, 1.0), pen="r", name="IRF")

        self.plot_wres.clear()
        self.plot_dist.clear()
