from __future__ import annotations

import numpy as np


class _MaxentModeMixin:
    def _has_donor_spectrum_loaded(self) -> bool:
        arr = getattr(self, "_donly_vec", None)
        if arr is None:
            return False
        try:
            v = np.asarray(arr, dtype=float).ravel()
        except Exception:
            return False
        if v.size < 2:
            return False
        if v.size % 2 != 0:
            return False
        if not np.all(np.isfinite(v)):
            return False
        return True

    def _update_donor_requirement_ui(self) -> None:
        is_fret = bool(getattr(self, "_mode_fret", False))
        has_donor = self._has_donor_spectrum_loaded()
        required_missing = bool(is_fret and (not has_donor))

        enabled = (not is_fret) or has_donor
        try:
            self.btn_run.setEnabled(bool(enabled))
        except Exception:
            pass
        try:
            self.btn_lcurve.setEnabled(bool(enabled))
        except Exception:
            pass

        tooltip = ""
        if required_missing:
            tooltip = "Load donor spectrum to enable MEM-FRET."
        try:
            self.btn_run.setToolTip(tooltip)
        except Exception:
            pass
        try:
            self.btn_lcurve.setToolTip(tooltip)
        except Exception:
            pass

        if required_missing:
            try:
                self.btn_load_donor.setStyleSheet(self._donor_style_missing)
                self.btn_load_donor_fit.setStyleSheet(self._donor_style_missing)
                self.label_donor_info.setStyleSheet(self._donor_label_style_missing)
            except Exception:
                pass
        else:
            try:
                self.btn_load_donor.setStyleSheet(self._donor_btn_style_normal or "")
                self.btn_load_donor_fit.setStyleSheet(self._donor_btn_fit_style_normal or "")
                self.label_donor_info.setStyleSheet(self._donor_label_style_normal or "")
            except Exception:
                pass

    def _reset_mem_result_state(self) -> None:
        if getattr(self, "_lcurve_curve", None) is not None:
            try:
                self._lcurve_curve.setData([], [])
            except Exception:
                pass
        if getattr(self, "_lcurve_corner", None) is not None:
            try:
                self._lcurve_corner.hide()
            except Exception:
                pass
        if getattr(self, "plot_dist", None) is not None:
            try:
                self.plot_dist.clear()
            except Exception:
                pass
        if getattr(self, "plot_wres", None) is not None:
            try:
                self.plot_wres.clear()
            except Exception:
                pass
        self._last_result = None
        self._sample_stats = None
        self._sample_band_lower = None
        self._sample_band_upper = None
        self._sample_band_fill = None
        self._sample_hist_item = None
        if getattr(self, "btn_save", None) is not None:
            try:
                self.btn_save.setEnabled(False)
            except Exception:
                pass

    def _on_mode_changed(self, index: int) -> None:
        self._mode_fret = bool(index == 1)
        self._update_mode_ui()
        self._update_donor_requirement_ui()

    def _update_mode_ui(self) -> None:
        is_fret = bool(self._mode_fret)
        lifetime_visible = not is_fret
        fret_visible = is_fret
        use_periodic = bool(self.chk_use_periodic.isChecked())

        try:
            self.spin_start_frac.setVisible(False)
        except Exception:
            pass
        label_start_frac = getattr(self, "_start_frac_label", None)
        if label_start_frac is not None:
            try:
                label_start_frac.setVisible(False)
            except Exception:
                pass

        # Lifetime-specific controls: tau grid.
        for w in (
            self.spin_tau_min,
            self.spin_tau_max,
            self.spin_tau_bins,
        ):
            w.setVisible(lifetime_visible)
        label_tau = getattr(self, "_tau_grid_label", None)
        if label_tau is not None:
            label_tau.setVisible(lifetime_visible)

        # FRET-specific controls (excluding lamp scatter, which is
        # meaningful in both modes).
        for w in (
            self.spin_tau0,
            self.spin_R0,
            self.spin_R_min,
            self.spin_R_max,
            self.spin_R_points,
            self.btn_load_donor,
            self.btn_load_donor_fit,
            self.label_donor_info,
            self._row_x_donly,
        ):
            w.setVisible(fret_visible)
        label_rda = getattr(self, "_rda_grid_label", None)
        if label_rda is not None:
            label_rda.setVisible(fret_visible)

        # tau0/R0 labels should also be hidden in lifetime mode.
        label_tau0 = getattr(self, "_tau0_label", None)
        if label_tau0 is not None:
            label_tau0.setVisible(fret_visible)
        label_R0 = getattr(self, "_R0_label", None)
        if label_R0 is not None:
            label_R0.setVisible(fret_visible)
        label_x_donly = getattr(self, "_x_donly_label", None)
        if label_x_donly is not None:
            label_x_donly.setVisible(fret_visible)

        # Period spinbox is relevant whenever periodic convolution is
        # enabled, independent of mode.
        period_visible = bool(use_periodic)
        try:
            self.spin_period.setVisible(period_visible)
        except Exception:
            pass
        label_period = getattr(self, "_period_label", None)
        if label_period is not None:
            label_period.setVisible(period_visible)

        # Combined prior button is always visible; the two labels are
        # mode specific for clarity.
        self.btn_load_prior.setVisible(True)
        self.label_prior_info.setVisible(lifetime_visible)
        self.label_dist_prior_info.setVisible(fret_visible)

        self._update_donor_requirement_ui()
