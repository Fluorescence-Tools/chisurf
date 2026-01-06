from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from chisurf.plugins.fluorescence_decay.maxent_decay.fmem.core import solve_fret_mem, solve_lifetime_mem
from .qt_stack import ensure_qt_stack


class _MaxentRunMixin:
    def _on_lcurve_clicked(self) -> None:
        try:
            self._run_lcurve()
        except Exception as exc:
            _, QtWidgets, _, _, _ = ensure_qt_stack()
            QtWidgets.QMessageBox.critical(self, "L-curve error", str(exc))

    def _run_lcurve(self) -> None:
        _, QtWidgets, QtCore, chisurf, _ = ensure_qt_stack()

        decay, dt, t = self._get_decay_and_dt()
        lamp = self._build_irf_array(decay.size, t, dt)

        is_fret = bool(self._mode_fret)

        fit_start_fraction = float(self.spin_start_frac.value())
        use_periodic = bool(self.chk_use_periodic.isChecked())

        ts_channels = float(self.spin_timeshift.value())
        ts_val = float(ts_channels)
        bg_val = float(self.spin_background.value())
        irf_bg_input = float(self.spin_irf_bg.value())
        irf_bg_arg: Optional[float]
        if irf_bg_input > 0.0:
            irf_bg_arg = irf_bg_input
        else:
            irf_bg_arg = None
        lamp_scatter_val = float(self.spin_lamp_scatter.value())

        fitrange_arg = self._get_fitrange_arg(decay.size)

        r_axis = None
        tau0_val = None
        R0_val = None
        x_donly_val = None
        donly_vec = None
        tau = None
        prior_vec: Optional[Sequence[float]] = None

        if is_fret:
            tau0_val = float(self.spin_tau0.value())
            R0_val = float(self.spin_R0.value())
            x_donly_val = float(self.spin_x_donly.value())
            r_axis = self._build_fret_r_axis(R0_val)
            donly_vec = self._get_fret_donor_spectrum(tau0_val)
            prior = self._get_dist_prior_for_axis(r_axis)
        else:
            tau = self._build_tau_grid()
            if self._prior_vec is not None and self._prior_vec.size == tau.size:
                prior_vec = self._prior_vec

        (
            optimize_nuisance,
            nuisance_step_timeshift,
            nuisance_step_background,
            nuisance_step_irf_background,
            nuisance_step_x_donly,
        ) = self._get_nuisance_settings()

        nu0 = float(self.spin_nu.value())
        if nu0 <= 0.0:
            nu0 = 1e-5
        log10_center = float(np.log10(nu0))
        try:
            dec_left = float(self.spin_lcurve_dec_left.value())
            dec_right = float(self.spin_lcurve_dec_right.value())
        except Exception:
            dec_left = 2.0
            dec_right = 2.0
        if dec_left < 0.0:
            dec_left = 0.0
        if dec_right < 0.0:
            dec_right = 0.0
        log10_min = log10_center - dec_left
        log10_max = log10_center + dec_right
        if (not np.isfinite(log10_min)) or (not np.isfinite(log10_max)) or log10_min >= log10_max:
            log10_min = log10_center - 2.0
            log10_max = log10_center + 2.0
        n_nu = 16
        log_grid = np.linspace(log10_min, log10_max, n_nu)
        nu_grid = 10.0 ** log_grid

        chi2_vals = np.empty_like(nu_grid, dtype=float)
        sol_vals = np.empty_like(nu_grid, dtype=float)

        progress = QtWidgets.QProgressDialog("Computing L-curve...", "Cancel", 0, int(n_nu), self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        try:
            for i, nu_val in enumerate(nu_grid):
                if progress.wasCanceled():
                    raise RuntimeError("L-curve computation cancelled")
                try:
                    if is_fret:
                        period_arg = self._get_period_arg(use_periodic)

                        res = solve_fret_mem(
                            decay=decay,
                            lamp=lamp,
                            dt=dt,
                            period=period_arg,
                            R=r_axis,
                            tau0=float(tau0_val),
                            R0=float(R0_val),
                            donly=donly_vec,
                            x_donly=float(x_donly_val) if x_donly_val is not None else 0.0,
                            timeshift=float(ts_val),
                            background=float(bg_val),
                            lamp_scatter=float(lamp_scatter_val),
                            irf_background=irf_bg_arg,
                            fitrange=fitrange_arg,
                            fit_start_fraction=fit_start_fraction,
                            nu=float(nu_val),
                            max_iter=200,
                            prior=prior,
                            optimize_nuisance=optimize_nuisance,
                            nuisance_step_timeshift=nuisance_step_timeshift,
                            nuisance_step_background=nuisance_step_background,
                            nuisance_step_irf_background=nuisance_step_irf_background,
                            nuisance_step_x_donly=nuisance_step_x_donly,
                            progress_cb=None,
                        )
                    else:
                        res = solve_lifetime_mem(
                            decay=decay,
                            lamp=lamp,
                            dt=dt,
                            tau=tau,
                            lamp_scatter=float(lamp_scatter_val),
                            fitrange=fitrange_arg,
                            fit_start_fraction=fit_start_fraction,
                            nu=float(nu_val),
                            optimize_nuisance=optimize_nuisance,
                            nuisance_step_timeshift=nuisance_step_timeshift,
                            nuisance_step_background=nuisance_step_background,
                            nuisance_step_irf_background=nuisance_step_irf_background,
                            prior=prior_vec,
                            max_iter=200,
                            progress_cb=None,
                        )
                    chi2_vals[i] = float(res.get("chisq", np.nan))
                    p_res = np.asarray(res.get("p", []), dtype=float).ravel()
                    if p_res.size == 0:
                        sol_vals[i] = np.nan
                    else:
                        sol_vals[i] = float(np.linalg.norm(p_res))
                except Exception:
                    chi2_vals[i] = np.nan
                    sol_vals[i] = np.nan
                progress.setValue(int(i + 1))
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
        finally:
            progress.close()

        corner_idx = None
        try:
            mask = np.isfinite(chi2_vals) & np.isfinite(sol_vals)
            if np.any(mask) and getattr(chisurf, "math", None) is not None:
                try:
                    corner_idx = chisurf.math.regularization.discrete_lcurve_corner(chi2_vals, sol_vals)
                except Exception:
                    corner_idx = None
        except Exception:
            corner_idx = None

        if corner_idx is not None and 0 <= int(corner_idx) < nu_grid.size:
            try:
                self.spin_nu.setValue(float(nu_grid[int(corner_idx)]))
            except Exception:
                pass

        self._update_lcurve_plot(chi2_vals, sol_vals, nu_grid, corner_idx)

    def _update_lcurve_plot(
        self,
        chi2_vals: np.ndarray,
        sol_vals: np.ndarray,
        nu_grid: np.ndarray,
        corner_idx: Optional[int],
    ) -> None:
        if getattr(self, "plot_lcurve", None) is None:
            return
        try:
            chi2_arr = np.asarray(chi2_vals, dtype=float).ravel()
            sol_arr = np.asarray(sol_vals, dtype=float).ravel()
        except Exception:
            chi2_arr = np.array([], dtype=float)
            sol_arr = np.array([], dtype=float)
        if chi2_arr.size == 0 or sol_arr.size == 0:
            self._lcurve_curve.setData([], [])
            try:
                self._lcurve_corner.hide()
            except Exception:
                pass
            return
        mask = np.isfinite(chi2_arr) & np.isfinite(sol_arr) & (chi2_arr > 0.0) & (sol_arr > 0.0)
        if not np.any(mask):
            self._lcurve_curve.setData([], [])
            try:
                self._lcurve_corner.hide()
            except Exception:
                pass
            return
        idx_all = np.nonzero(mask)[0]
        chi2_plot = chi2_arr[mask]
        sol_plot = sol_arr[mask]
        eps = np.finfo(float).tiny
        chi2_plot = np.clip(chi2_plot, eps, np.inf)
        sol_plot = np.clip(sol_plot, eps, np.inf)
        self._lcurve_curve.setData(chi2_plot, sol_plot)

        try:
            self._lcurve_corner.hide()
        except Exception:
            pass

        if corner_idx is None:
            return
        try:
            matches = np.nonzero(idx_all == int(corner_idx))[0]
            if matches.size == 0:
                return
            local_idx = int(matches[0])
            if 0 <= local_idx < chi2_plot.size:
                self._lcurve_corner.setData(
                    [float(chi2_plot[local_idx])],
                    [float(sol_plot[local_idx])],
                )
                try:
                    self._lcurve_corner.show()
                except Exception:
                    pass
        except Exception:
            return

    def _run_mem(self) -> None:
        _, QtWidgets, QtCore, _, _ = ensure_qt_stack()

        decay, dt, t = self._get_decay_and_dt()
        lamp = self._build_irf_array(decay.size, t, dt)

        is_fret = bool(self._mode_fret)

        if is_fret and (not self._has_donor_spectrum_loaded()):
            raise RuntimeError("Load donor spectrum first (FRET mode requires donor spectrum).")

        if not is_fret:
            tau = self._build_tau_grid()
        nu = float(self.spin_nu.value())
        fit_start_fraction = float(self.spin_start_frac.value())
        use_periodic = bool(self.chk_use_periodic.isChecked())

        prior_vec: Optional[Sequence[float]] = None
        if (not is_fret) and self._prior_vec is not None:
            if self._prior_vec.size != tau.size:
                raise ValueError(
                    f"Prior length {self._prior_vec.size} does not match tau grid length {tau.size}"
                )
            prior_vec = self._prior_vec

        (
            optimize_nuisance,
            nuisance_step_timeshift,
            nuisance_step_background,
            nuisance_step_irf_background,
            nuisance_step_x_donly,
        ) = self._get_nuisance_settings()

        max_iter = 200

        fitrange_arg = self._get_fitrange_arg(decay.size)

        r_axis = None
        tau0_val = None
        R0_val = None
        x_donly_val = None
        if is_fret:
            tau0_val = float(self.spin_tau0.value())
            R0_val = float(self.spin_R0.value())
            x_donly_val = float(self.spin_x_donly.value())
            r_axis = self._build_fret_r_axis(R0_val)

        ts_channels = float(self.spin_timeshift.value())
        ts_val = float(ts_channels)
        bg_val = float(self.spin_background.value())
        irf_bg_input = float(self.spin_irf_bg.value())
        irf_bg_arg: Optional[float]
        if irf_bg_input > 0.0:
            irf_bg_arg = irf_bg_input
        else:
            irf_bg_arg = None
        lamp_scatter_val = float(self.spin_lamp_scatter.value())

        progress = QtWidgets.QProgressDialog("Running MEM...", "Cancel", 0, max_iter, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def _progress_cb(iter_idx, chisq, S, Q, dgrad):
            val = int(iter_idx)
            if val < 0:
                val = 0
            if val > max_iter:
                val = max_iter
            progress.setValue(val)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents)
            if progress.wasCanceled():
                raise RuntimeError("MEM computation cancelled")

        try:
            if is_fret:
                donly_vec = self._get_fret_donor_spectrum(float(tau0_val))
                prior = self._get_dist_prior_for_axis(np.asarray(r_axis, dtype=float))

                period_arg = self._get_period_arg(use_periodic)

                result = solve_fret_mem(
                    decay=decay,
                    lamp=lamp,
                    dt=dt,
                    period=period_arg,
                    R=r_axis,
                    tau0=float(tau0_val),
                    R0=float(R0_val),
                    donly=donly_vec,
                    x_donly=float(x_donly_val) if x_donly_val is not None else 0.0,
                    timeshift=float(ts_val),
                    background=float(bg_val),
                    lamp_scatter=float(lamp_scatter_val),
                    irf_background=irf_bg_arg,
                    fitrange=fitrange_arg,
                    fit_start_fraction=fit_start_fraction,
                    nu=nu,
                    max_iter=max_iter,
                    prior=prior,
                    progress_cb=_progress_cb,
                    optimize_nuisance=optimize_nuisance,
                    nuisance_step_timeshift=nuisance_step_timeshift,
                    nuisance_step_background=nuisance_step_background,
                    nuisance_step_irf_background=nuisance_step_irf_background,
                    nuisance_step_x_donly=nuisance_step_x_donly,
                )
            else:
                tau = self._build_tau_grid()
                result = solve_lifetime_mem(
                    decay=decay,
                    lamp=lamp,
                    dt=dt,
                    tau=tau,
                    timeshift=float(ts_val),
                    background=float(bg_val),
                    lamp_scatter=float(lamp_scatter_val),
                    irf_background=irf_bg_arg,
                    fitrange=fitrange_arg,
                    fit_start_fraction=fit_start_fraction,
                    nu=nu,
                    optimize_nuisance=optimize_nuisance,
                    nuisance_step_timeshift=nuisance_step_timeshift,
                    nuisance_step_background=nuisance_step_background,
                    nuisance_step_irf_background=nuisance_step_irf_background,
                    prior=prior_vec,
                    max_iter=max_iter,
                    progress_cb=_progress_cb,
                )
        finally:
            progress.close()
        self._last_result = result
        if getattr(self, "btn_save", None) is not None:
            self.btn_save.setEnabled(True)
        try:
            if not is_fret:
                if "timeshift" in result:
                    self.spin_timeshift.setValue(float(result["timeshift"]))
                if "background" in result:
                    self.spin_background.setValue(float(result["background"]))
                if "lamp_scatter" in result:
                    self.spin_lamp_scatter.setValue(float(result["lamp_scatter"]))
                if "irf_background" in result:
                    self.spin_irf_bg.setValue(float(result["irf_background"]))
            else:
                if "timeshift" in result:
                    self.spin_timeshift.setValue(float(result["timeshift"]))
                if "background" in result:
                    self.spin_background.setValue(float(result["background"]))
                if "lamp_scatter" in result:
                    self.spin_lamp_scatter.setValue(float(result["lamp_scatter"]))
                if "x_donly" in result:
                    self.spin_x_donly.setValue(float(result["x_donly"]))
                if "irf_background" in result:
                    self.spin_irf_bg.setValue(float(result["irf_background"]))
        except Exception:
            pass
        self._update_plots_from_result(decay, t, result)
