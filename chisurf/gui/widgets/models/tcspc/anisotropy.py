from __future__ import annotations

import numpy as np

import chisurf as cs
from qtpy import QtWidgets, QtCore, QtGui
import chisurf.gui.widgets.fitting
from chisurf.core.fitting.parameter import FittingParameter

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover
    pg = None

from chisurf.core.models.tcspc.anisotropy import Anisotropy


ADD_BUTTON_STYLE = (
    "QPushButton { background-color: #1f7a1f; color: white; border: 1px solid #166016; "
    "border-radius: 3px; padding: 2px 8px; }"
    "QPushButton:hover { background-color: #249124; }"
    "QPushButton:pressed { background-color: #155815; }"
)

REMOVE_BUTTON_STYLE = (
    "QPushButton { background-color: #a82020; color: white; border: 1px solid #7d1717; "
    "border-radius: 3px; padding: 2px 8px; }"
    "QPushButton:hover { background-color: #bf2626; }"
    "QPushButton:pressed { background-color: #7d1717; }"
)


class AnisotropyWidget(Anisotropy, QtWidgets.QGroupBox):






    def _compute_anisotropy_traces_from_channels(self, t, vv_m, vh_m):
        """Compute uncorrected and corrected anisotropy traces from VV/VH.

        Parameters
        ----------
        t : np.ndarray
            Time axis.
        vv_m : np.ndarray
            VV channel data.
        vh_m : np.ndarray
            VH channel data.

        Returns
        -------
        tuple of np.ndarray or None
            (time, r_uncorrected, r_corrected).
        """
        if t is None or vv_m is None or vh_m is None:
            return None, None, None
        try:
            g = float(self.g)
            l1 = float(self.l1)
            l2 = float(self.l2)
            det = (1.0 - l1) * (1.0 - l2) - l1 * l2
            if abs(det) < 1e-12:
                return t, None, None

            den_unc = g * vv_m + 2.0 * vh_m
            with np.errstate(divide='ignore', invalid='ignore'):
                r_unc = np.where(np.abs(den_unc) > 1e-12, (vv_m - vh_m) / den_unc, np.nan)

            vv = ((1.0 - l2) * vv_m - l1 * vh_m) / det
            vh = (-l2 * vv_m + (1.0 - l1) * vh_m) / det
            den_cor = g * vv + 2.0 * vh
            with np.errstate(divide='ignore', invalid='ignore'):
                r_cor = np.where(np.abs(den_cor) > 1e-12, (vv - vh) / den_cor, np.nan)

            finite = np.isfinite(t) & np.isfinite(r_unc) & np.isfinite(r_cor)
            if np.any(finite):
                return t[finite], r_unc[finite], r_cor[finite]
            return t, r_unc, r_cor
        except Exception:
            return t, None, None

    def _compute_anisotropy_traces(self):
        """Compute anisotropy traces from background-corrected VV/VH data.

        Returns
        -------
        tuple of np.ndarray or None
            (time, r_uncorrected, r_corrected).
        """
        t, vv_m, vh_m = self._extract_vv_vh_bg_corrected()
        return self._compute_anisotropy_traces_from_channels(t, vv_m, vh_m)

    def _show_anisotropy_decay_dialog(self) -> None:
        """Show a dialog with interactive anisotropy decay plots (data and model)."""
        if pg is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Anisotropy decays",
                "pyqtgraph is not available, cannot plot anisotropy decays.",
            )
            return

        t_raw, vv_raw, vh_raw, defaults = self._extract_vv_vh_raw_for_diag()
        if t_raw is None or vv_raw is None or vh_raw is None or defaults is None:
            QtWidgets.QMessageBox.information(
                self,
                "Anisotropy decays",
                "VV/VH channels are not available for anisotropy-decay diagnostics.",
            )
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Anisotropy decays (corrected vs uncorrected)")
        layout = QtWidgets.QVBoxLayout(dialog)

        controls = QtWidgets.QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(2)

        def _spin(value: float, step: float, decimals: int = 4):
            """Create a QDoubleSpinBox with the given range and step.

            Parameters
            ----------
            value : float
                Initial value.
            step : float
                Single step increment.
            decimals : int
                Number of decimal places.

            Returns
            -------
            QDoubleSpinBox
            """
            sb = QtWidgets.QDoubleSpinBox(dialog)
            sb.setRange(-100000.0, 100000.0)
            sb.setDecimals(decimals)
            sb.setSingleStep(step)
            sb.setValue(float(value))
            return sb

        g_sb = _spin(defaults['g'], 0.01, 6)
        l1_sb = _spin(defaults['l1'], 0.001, 6)
        l2_sb = _spin(defaults['l2'], 0.001, 6)
        bg_vv_sb = _spin(defaults['bg_vv'], 1.0, 3)
        bg_vh_sb = _spin(defaults['bg_vh'], 1.0, 3)
        rel_shift_default = float(defaults['shift_vh']) - float(defaults['shift_vv'])
        rel_shift_sb = _spin(rel_shift_default, 0.01, 4)

        controls.addWidget(QtWidgets.QLabel("g:"), 0, 0)
        controls.addWidget(g_sb, 0, 1)
        controls.addWidget(QtWidgets.QLabel("l1:"), 0, 2)
        controls.addWidget(l1_sb, 0, 3)
        controls.addWidget(QtWidgets.QLabel("l2:"), 0, 4)
        controls.addWidget(l2_sb, 0, 5)

        controls.addWidget(QtWidgets.QLabel("BgVV:"), 1, 0)
        controls.addWidget(bg_vv_sb, 1, 1)
        controls.addWidget(QtWidgets.QLabel("BgVH:"), 1, 2)
        controls.addWidget(bg_vh_sb, 1, 3)
        controls.addWidget(QtWidgets.QLabel("dVH-VV:"), 1, 4)
        controls.addWidget(rel_shift_sb, 1, 5)

        controls.addWidget(QtWidgets.QLabel(" "), 2, 0)
        link_l_chk = QtWidgets.QCheckBox("link l1/l2", dialog)
        link_l_chk.setChecked(True)
        controls.addWidget(link_l_chk, 2, 2, 1, 1)
        reset_btn = QtWidgets.QToolButton(dialog)
        reset_btn.setText("Reset")
        controls.addWidget(reset_btn, 2, 3, 1, 1)
        layout.addLayout(controls)

        pw = pg.PlotWidget(dialog)
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setLabel("bottom", "Time")
        pw.setLabel("left", "r(t)")
        pw.setYRange(0.0, 0.5, padding=0.0)
        pw.addLegend()
        c_data_unc = pw.plot([], [], pen=pg.mkPen((100, 116, 139), width=2), name="data raw")
        c_data_cor = pw.plot([], [], pen=pg.mkPen((22, 163, 74), width=2), name="data corr")
        c_model_unc = pw.plot([], [], pen=pg.mkPen((59, 130, 246), width=2, style=QtCore.Qt.DashLine), name="model raw")
        c_model_cor = pw.plot([], [], pen=pg.mkPen((16, 185, 129), width=2, style=QtCore.Qt.DashLine), name="model corr")
        layout.addWidget(pw)

        t_model, vv_model_raw, vh_model_raw = self._extract_vv_vh_model_for_diag()

        state = {
            't': None,
            'r_data_unc': None,
            'r_data_cor': None,
            't_model': None,
            'r_model_unc': None,
            'r_model_cor': None,
        }

        def _rt_curves(t, vv, vh, g, l1, l2):
            """Compute uncorrected and corrected anisotropy from channels.

            Parameters
            ----------
            t : np.ndarray
            vv : np.ndarray
            vh : np.ndarray
            g : float
            l1 : float
            l2 : float

            Returns
            -------
            tuple of np.ndarray or None
                (t, r_uncorrected, r_corrected).
            """
            det = (1.0 - l1) * (1.0 - l2) - l1 * l2
            if abs(det) < 1e-12:
                return t, None, None
            den_unc = g * vv + 2.0 * vh
            with np.errstate(divide='ignore', invalid='ignore'):
                r_unc = np.where(np.abs(den_unc) > 1e-12, (vv - vh) / den_unc, np.nan)
            vv_u = ((1.0 - l2) * vv - l1 * vh) / det
            vh_u = (-l2 * vv + (1.0 - l1) * vh) / det
            den_cor = g * vv_u + 2.0 * vh_u
            with np.errstate(divide='ignore', invalid='ignore'):
                r_cor = np.where(np.abs(den_cor) > 1e-12, (vv_u - vh_u) / den_cor, np.nan)
            finite = np.isfinite(t) & np.isfinite(r_unc) & np.isfinite(r_cor)
            if np.any(finite):
                return t[finite], r_unc[finite], r_cor[finite]
            return t, r_unc, r_cor

        # TODO: needs docstring
        def recompute():
            """Recompute and update the anisotropy decay plot."""
            t = np.asarray(t_raw, dtype=np.float64)
            vv = np.asarray(vv_raw, dtype=np.float64) - float(bg_vv_sb.value())
            vh = np.asarray(vh_raw, dtype=np.float64) - float(bg_vh_sb.value())
            vh = self._shift_trace_to_reference(t, vh, float(rel_shift_sb.value()))

            g = float(g_sb.value())
            l1 = float(l1_sb.value())
            l2 = float(l2_sb.value())
            tt, r_unc, r_cor = _rt_curves(t, vv, vh, g, l1, l2)
            state['t'] = tt
            state['r_data_unc'] = r_unc
            state['r_data_cor'] = r_cor
            if tt is None or r_unc is None or r_cor is None:
                c_data_unc.setData([], [])
                c_data_cor.setData([], [])
            else:
                c_data_unc.setData(tt, r_unc)
                c_data_cor.setData(tt, r_cor)

            # Model curves (raw/corrected) with same control parameters.
            if t_model is None or vv_model_raw is None or vh_model_raw is None:
                state['t_model'] = None
                state['r_model_unc'] = None
                state['r_model_cor'] = None
                c_model_unc.setData([], [])
                c_model_cor.setData([], [])
                return

            tm = np.asarray(t_model, dtype=np.float64)
            vvm = np.asarray(vv_model_raw, dtype=np.float64) - float(bg_vv_sb.value())
            vhm = np.asarray(vh_model_raw, dtype=np.float64) - float(bg_vh_sb.value())
            vhm = self._shift_trace_to_reference(tm, vhm, float(rel_shift_sb.value()))
            ttm, rmu, rmc = _rt_curves(tm, vvm, vhm, g, l1, l2)
            state['t_model'] = ttm
            state['r_model_unc'] = rmu
            state['r_model_cor'] = rmc
            if ttm is None or rmu is None or rmc is None:
                c_model_unc.setData([], [])
                c_model_cor.setData([], [])
            else:
                c_model_unc.setData(ttm, rmu)
                c_model_cor.setData(ttm, rmc)

        def _sync_l2_from_l1():
            """Synchronise l2 value from l1 when the link checkbox is checked."""
            if not link_l_chk.isChecked():
                return
            v = float(l1_sb.value())
            if abs(float(l2_sb.value()) - v) > 1e-15:
                l2_sb.blockSignals(True)
                l2_sb.setValue(v)
                l2_sb.blockSignals(False)

        def _on_link_toggle(checked: bool):
            """Handle the l1/l2 link checkbox toggle.

            Parameters
            ----------
            checked : bool
                Whether the link is active.
            """
            l2_sb.setEnabled(not bool(checked))
            if checked:
                _sync_l2_from_l1()
            recompute()

        link_l_chk.toggled.connect(_on_link_toggle)
        l1_sb.valueChanged.connect(lambda _: _sync_l2_from_l1())
        _on_link_toggle(link_l_chk.isChecked())

        for sb in (g_sb, l1_sb, l2_sb, bg_vv_sb, bg_vh_sb, rel_shift_sb):
            sb.valueChanged.connect(lambda _: recompute())

        # TODO: needs docstring
        def on_reset():
            """Reset diagnostic spin boxes to defaults."""
            g_sb.setValue(float(defaults['g']))
            l1_sb.setValue(float(defaults['l1']))
            l2_sb.setValue(float(defaults['l2']))
            bg_vv_sb.setValue(float(defaults['bg_vv']))
            bg_vh_sb.setValue(float(defaults['bg_vh']))
            rel_shift_sb.setValue(float(defaults['shift_vh']) - float(defaults['shift_vv']))
            link_l_chk.setChecked(True)

        reset_btn.clicked.connect(on_reset)
        recompute()

        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        save_btn = box.addButton("Save CSV", QtWidgets.QDialogButtonBox.ActionRole)

        # TODO: needs docstring
        def on_save_csv():
            """Save anisotropy decay data to CSV."""
            tt = state.get('t')
            r_du = state.get('r_data_unc')
            r_dc = state.get('r_data_cor')
            if tt is None or r_du is None or r_dc is None:
                QtWidgets.QMessageBox.information(dialog, "Save anisotropy decay", "No anisotropy decay data to save.")
                return
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                dialog,
                "Save anisotropy decay",
                "anisotropy_decay.csv",
                "CSV files (*.csv);;All files (*)",
            )
            if not filename:
                return
            try:
                tm = state.get('t_model')
                r_mu = state.get('r_model_unc')
                r_mc = state.get('r_model_cor')
                if tm is None or r_mu is None or r_mc is None:
                    arr = np.column_stack([tt, r_du, r_dc])
                    header = "time_data,r_data_uncorrected,r_data_corrected"
                else:
                    # Save both data and model on their own time axes.
                    n = max(len(tt), len(tm))
                    out = np.full((n, 6), np.nan, dtype=float)
                    out[:len(tt), 0] = tt
                    out[:len(tt), 1] = r_du
                    out[:len(tt), 2] = r_dc
                    out[:len(tm), 3] = tm
                    out[:len(tm), 4] = r_mu
                    out[:len(tm), 5] = r_mc
                    arr = out
                    header = "time_data,r_data_uncorrected,r_data_corrected,time_model,r_model_uncorrected,r_model_corrected"
                np.savetxt(filename, arr, delimiter=",", header=header, comments="")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(dialog, "Save anisotropy decay", f"Failed to save file:\n{exc}")

        save_btn.clicked.connect(on_save_csv)
        box.accepted.connect(dialog.accept)
        layout.addWidget(box)
        dialog.resize(560, 380)
        dialog.exec_()

    def _update_consistency_label(self) -> None:
        """Update the colour-coded consistency label comparing r_S,L and r_S,I."""
        if not hasattr(self, '_quality_label'):
            return
        r_l = getattr(self, '_last_rsl', float('nan'))
        r_i = getattr(self, '_last_rsi', float('nan'))
        if not (np.isfinite(r_l) and np.isfinite(r_i)):
            self._quality_label.setText("Diag: waiting")
            self._quality_label.setStyleSheet("color: #6b7280;")
            return

        denom = max(abs(r_l), abs(r_i), 1e-12)
        delta_pct = 100.0 * abs(r_l - r_i) / denom
        if delta_pct > 20.0:
            self._quality_label.setText(f"Diag: red (Δ={delta_pct:.1f}%)")
            self._quality_label.setStyleSheet("color: #b91c1c; font-weight: 600;")
        elif delta_pct > 10.0:
            self._quality_label.setText(f"Diag: yellow (Δ={delta_pct:.1f}%)")
            self._quality_label.setStyleSheet("color: #a16207; font-weight: 600;")
        else:
            self._quality_label.setText(f"Diag: green (Δ={delta_pct:.1f}%)")
            self._quality_label.setStyleSheet("color: #166534;")

    def _toggle_diag_visibility(self, show: bool) -> None:
        """Show or hide the VV/VH background integral diagnostic widgets.

        Parameters
        ----------
        show : bool
            Whether to show the widgets.
        """
        show = bool(show)
        for w in (getattr(self, '_w_vv_bg', None), getattr(self, '_w_vh_bg', None)):
            if w is not None:
                w.setVisible(show)

    def _intensity_diag_key(self):
        """Generate a cache key for intensity diagnostics.

        Returns
        -------
        tuple
            Hashable key based on fit, data, group, polarization, g, l1, l2.
        """
        fit = getattr(self, 'fit', None)
        data = getattr(fit, 'data', None)
        group = getattr(fit, 'group', None)
        pol = str(getattr(self, 'polarization_type', 'vm')).lower()
        return (
            id(fit),
            id(data),
            id(group),
            pol,
            round(float(self.g), 10),
            round(float(self.l1), 10),
            round(float(self.l2), 10),
        )

    def _compute_intensity_diagnostics(self):
        """Compute intensity diagnostic values (sums and steady-state r).

        Returns
        -------
        dict or None
            Dictionary with keys ``sum_vv_m``, ``sum_vh_m``, ``r_si``.
        """
        t, vv_m, vh_m = self._extract_vv_vh_bg_corrected()
        if t is None or vv_m is None or vh_m is None:
            return None

        out = {
            'sum_vv_m': float(np.nansum(vv_m)),
            'sum_vh_m': float(np.nansum(vh_m)),
            'r_si': float('nan'),
        }
        try:
            l1 = float(self.l1)
            l2 = float(self.l2)
            g = float(self.g)
            det = (1.0 - l1) * (1.0 - l2) - l1 * l2
            if abs(det) < 1e-12:
                return out
            vv = ((1.0 - l2) * vv_m - l1 * vh_m) / det
            vh = (-l2 * vv_m + (1.0 - l1) * vh_m) / det
            num = np.nansum(vv - vh)
            den = np.nansum(g * vv + 2.0 * vh)
            if den != 0.0 and np.isfinite(den):
                out['r_si'] = float(num / den)
        except Exception:
            pass
        return out

    def _get_intensity_diagnostics(self):
        """Return cached intensity diagnostics or compute them if stale.

        Returns
        -------
        dict or None
        """
        key = self._intensity_diag_key()
        if getattr(self, '_intensity_diag_cache_key', None) == key:
            return getattr(self, '_intensity_diag_cache', None)
        diag = self._compute_intensity_diagnostics()
        self._intensity_diag_cache_key = key
        self._intensity_diag_cache = diag
        return diag


    def _extract_vv_vh_bg_corrected(self):
        """Extract VV and VH channel data with background correction.

        Handles stacked datasets, fit-group VV/VH pairs, and fallback
        single-dataset extraction.

        Returns
        -------
        tuple of np.ndarray or None
            (time, vv_bg_corrected, vh_bg_corrected).
        """
        fit = getattr(self, 'fit', None)
        data = getattr(fit, 'data', None)
        if data is None:
            return None, None, None

        pol = str(getattr(self, 'polarization_type', 'vm')).lower()

        # Grouped VV/VH datasets (preferred path for dual-channel anisotropy).
        if pol in ('vv/vh', 'vvvh') and hasattr(data, '__len__') and hasattr(data, '__getitem__'):
            try:
                if len(data) >= 2:
                    d_vv = data[0]
                    d_vh = data[1]
                    t_vv = np.asarray(getattr(d_vv, 'x', None), dtype=np.float64)
                    t_vh = np.asarray(getattr(d_vh, 'x', None), dtype=np.float64)
                    y_vv = np.asarray(getattr(d_vv, 'y', None), dtype=np.float64)
                    y_vh = np.asarray(getattr(d_vh, 'y', None), dtype=np.float64)
                    if y_vv.ndim > 1:
                        y_vv = y_vv[0]
                    if y_vh.ndim > 1:
                        y_vh = y_vh[0]
                    n = min(t_vv.size, t_vh.size, y_vv.size, y_vh.size)
                    if n >= 2:
                        t = t_vv[:n]
                        if np.max(np.abs(t - t_vh[:n])) > 1e-12:
                            t = 0.5 * (t + t_vh[:n])

                        bg_vv = 0.0
                        bg_vh = 0.0
                        try:
                            m_vv = getattr(d_vv, 'meta_data', None)
                            if isinstance(m_vv, dict):
                                bg_vv = float(m_vv.get('bg', m_vv.get('background', 0.0)))
                        except Exception:
                            pass
                        try:
                            m_vh = getattr(d_vh, 'meta_data', None)
                            if isinstance(m_vh, dict):
                                bg_vh = float(m_vh.get('bg', m_vh.get('background', 0.0)))
                        except Exception:
                            pass
                        try:
                            m_grp = getattr(data, 'meta_data', None)
                            if isinstance(m_grp, dict):
                                bg_vv = float(m_grp.get('bg_vv', m_grp.get('BgVV', bg_vv)))
                                bg_vh = float(m_grp.get('bg_vh', m_grp.get('BgVH', bg_vh)))
                        except Exception:
                            pass

                        vv_m = y_vv[:n] - bg_vv
                        vh_m = y_vh[:n] - bg_vh
                        # Single fit / stacked channels: one shared timeshift,
                        # therefore no relative alignment needed here.
                        return t, vv_m, vh_m
            except Exception:
                pass

        # Fit-group VV/VH datasets: each local fit carries one polarization curve.
        fit = getattr(self, 'fit', None)
        group = getattr(fit, 'group', None)
        if group is not None:
            try:
                if len(group) >= 2:
                    vv_fit = None
                    vh_fit = None
                    for local_fit in group:
                        local_model = getattr(local_fit, 'model', None)
                        local_aniso = getattr(local_model, 'anisotropy', None)
                        local_pol = str(getattr(local_aniso, 'polarization_type', '')).lower()
                        if local_pol == 'vv' and vv_fit is None:
                            vv_fit = local_fit
                        elif local_pol == 'vh' and vh_fit is None:
                            vh_fit = local_fit
                    if vv_fit is None or vh_fit is None:
                        vv_fit = group[0]
                        vh_fit = group[1]

                    d_vv = getattr(vv_fit, 'data', None)
                    d_vh = getattr(vh_fit, 'data', None)
                    if d_vv is not None and d_vh is not None:
                        t_vv = np.asarray(getattr(d_vv, 'x', None), dtype=np.float64)
                        t_vh = np.asarray(getattr(d_vh, 'x', None), dtype=np.float64)
                        y_vv = np.asarray(getattr(d_vv, 'y', None), dtype=np.float64)
                        y_vh = np.asarray(getattr(d_vh, 'y', None), dtype=np.float64)
                        # If a local fit carries a stacked 2-channel dataset,
                        # explicitly pick VV (row 0) and VH (row 1) channels.
                        if y_vv.ndim > 1:
                            y_vv = y_vv[0]
                        if y_vh.ndim > 1:
                            y_vh = y_vh[1] if y_vh.shape[0] > 1 else y_vh[0]
                        n = min(t_vv.size, t_vh.size, y_vv.size, y_vh.size)
                        if n >= 2:
                            t = t_vv[:n]
                            if np.max(np.abs(t - t_vh[:n])) > 1e-12:
                                t = 0.5 * (t + t_vh[:n])
                            bg_vv = self._curve_bg_level(d_vv, 0.0)
                            bg_vh = self._curve_bg_level(d_vh, 0.0)
                            vv_m = y_vv[:n] - bg_vv
                            vh_m = y_vh[:n] - bg_vh

                            # Align VV/VH to a common IRF position using their
                            # individual local-fit timeshifts.
                            shift_vv = self._fit_timeshift(vv_fit)
                            shift_vh = self._fit_timeshift(vh_fit)
                            # Reference: top/first VV trace in the group.
                            vh_m = self._shift_trace_to_reference(
                                t=t,
                                y=vh_m,
                                delta_t=(shift_vh - shift_vv)
                            )
                            return t, vv_m, vh_m
            except Exception:
                pass

        # Fallback: current dataset only.
        try:
            x = np.asarray(getattr(data, 'x', None), dtype=np.float64)
            y = np.asarray(getattr(data, 'y', None), dtype=np.float64)
        except Exception:
            return None, None, None

        if y.size == 0:
            return None, None, None
        if y.ndim == 1:
            y = y.reshape(1, -1)
        n = min(x.size, y.shape[-1])
        if n < 2:
            return None, None, None

        t = x[:n]
        y = y[:, :n]
        _, bg_vv, bg_vh = self._background_levels(data, y)
        if y.shape[0] >= 2:
            return t, y[0] - bg_vv, y[1] - bg_vh
        if pol == 'vv':
            vv_ch = y[0]
            return t, vv_ch - bg_vv, None
        if pol == 'vh':
            vh_ch = y[1] if y.shape[0] > 1 else y[0]
            return t, None, vh_ch - bg_vh
        return t, None, None

    def _compute_vv_bg_corrected_integral(self) -> float:
        """Return the VV background-corrected integrated intensity.

        Returns
        -------
        float
        """
        diag = self._get_intensity_diagnostics()
        if not isinstance(diag, dict):
            return float('nan')
        return float(diag.get('sum_vv_m', float('nan')))

    def _compute_vh_bg_corrected_integral(self) -> float:
        """Return the VH background-corrected integrated intensity.

        Returns
        -------
        float
        """
        diag = self._get_intensity_diagnostics()
        if not isinstance(diag, dict):
            return float('nan')
        return float(diag.get('sum_vh_m', float('nan')))

    def _compute_rt(self, time_axis: np.ndarray) -> np.ndarray:
        """Compute the time-dependent anisotropy r(t) from the rotation spectrum.

        Parameters
        ----------
        time_axis : np.ndarray
            Time values.

        Returns
        -------
        np.ndarray
            Anisotropy decay r(t).
        """
        t = np.asarray(time_axis, dtype=np.float64)
        rt = np.zeros_like(t)
        rs = np.asarray(self.rotation_spectrum, dtype=np.float64)
        if rs.size < 2:
            return rt
        b = rs[0::2]
        rho = rs[1::2]
        for bi, ri in zip(b, rho):
            try:
                r = float(ri)
                if r > 0.0:
                    rt += float(bi) * np.exp(-t / r)
            except Exception:
                continue
        return rt

    def _background_levels(self, data, y: np.ndarray) -> tuple[float, float, float]:
        """Determine background levels for VV and VH channels.

        Parameters
        ----------
        data : object
            Data curve or group with optional metadata.
        y : np.ndarray
            Channel data array.

        Returns
        -------
        tuple of float
            (bg, bg_vv, bg_vh).
        """
        bg = 0.0
        try:
            fit = getattr(self, 'fit', None)
            model = getattr(fit, 'model', None)
            generic = getattr(model, 'generic', None)
            if generic is not None and hasattr(generic, 'background'):
                bg = float(generic.background)
        except Exception:
            bg = 0.0

        bg_vv = bg
        bg_vh = bg
        try:
            meta = getattr(data, 'meta_data', None)
            if isinstance(meta, dict):
                bg_vv = float(meta.get('bg_vv', meta.get('BgVV', bg_vv)))
                bg_vh = float(meta.get('bg_vh', meta.get('BgVH', bg_vh)))
                bg = float(meta.get('bg', meta.get('background', bg)))
        except Exception:
            pass

        if y.shape[0] >= 2:
            return bg, bg_vv, bg_vh
        return bg, bg, bg

    def _compute_steady_state_anisotropy_lifetime(self) -> float:
        """Compute the steady-state anisotropy from the lifetime model.

        Returns
        -------
        float
            Steady-state anisotropy value, or NaN if unavailable.
        """
        model = getattr(self, 'model', None)
        if model is None:
            fit = getattr(self, 'fit', None)
            model = getattr(fit, 'model', None)
        if model is not None and hasattr(model, 'steady_state_anisotropy'):
            try:
                v = float(model.steady_state_anisotropy)
                self._last_rsl = v
                self._update_consistency_label()
                return v
            except Exception:
                pass
        self._last_rsl = float('nan')
        self._update_consistency_label()
        return float('nan')

    def _compute_steady_state_anisotropy_intensity(self) -> float:
        """Compute the steady-state anisotropy from the intensity data.

        Returns
        -------
        float
            Steady-state anisotropy value, or NaN if unavailable.
        """
        model = getattr(self, 'model', None)
        if model is None:
            fit = getattr(self, 'fit', None)
            model = getattr(fit, 'model', None)

        fit = getattr(self, 'fit', None)
        data = getattr(fit, 'data', None)
        if data is None:
            return float('nan')

        pol = str(getattr(self, 'polarization_type', 'vm')).lower()
        diag = self._get_intensity_diagnostics()
        if isinstance(diag, dict):
            r_si = float(diag.get('r_si', float('nan')))
            if np.isfinite(r_si):
                self._last_rsi = r_si
                self._update_consistency_label()
                return r_si

        try:
            x = np.asarray(getattr(data, 'x', None), dtype=np.float64)
        except Exception:
            x = np.array([], dtype=np.float64)
        if x.size == 0:
            try:
                y_tmp = np.asarray(getattr(data, 'y', None), dtype=np.float64)
                n = y_tmp.shape[-1] if y_tmp.size else 0
                x = np.arange(max(0, int(n)), dtype=np.float64)
            except Exception:
                return float('nan')

        y = np.asarray(getattr(data, 'y', None), dtype=np.float64)
        if y.size == 0:
            return float('nan')
        if y.ndim == 1:
            y = y.reshape(1, -1)

        n = min(x.size, y.shape[-1])
        if n < 2:
            return float('nan')
        t = x[:n]
        y = y[:, :n]

        pol = str(getattr(self, 'polarization_type', 'vm')).lower()
        g = float(self.g)
        bg, bg_vv, bg_vh = self._background_levels(data, y)

        rt = self._compute_rt(t)
        vm = None
        if pol == 'vm':
            vm = np.clip(y[0] - bg, 0.0, None)
        elif pol == 'vv':
            vv = np.clip(y[0] - bg_vv, 0.0, None)
            denom = 1.0 + 2.0 * rt
            mask = np.abs(denom) > 1e-12
            if np.any(mask):
                vm = np.zeros_like(vv)
                vm[mask] = vv[mask] / denom[mask]
        elif pol == 'vh':
            vh = np.clip(y[0] - bg_vh, 0.0, None)
            denom = 1.0 - g * rt
            mask = np.abs(denom) > 1e-12
            if np.any(mask):
                vm = np.zeros_like(vh)
                vm[mask] = vh[mask] / denom[mask]
        elif y.shape[0] >= 2:
            # Best effort if polarization metadata is unavailable.
            vv = np.clip(y[0] - bg_vv, 0.0, None)
            vh = np.clip(y[1] - bg_vh, 0.0, None)
            vm = vv + 2.0 * g * vh
        else:
            vm = np.clip(y[0] - bg, 0.0, None)

        if vm is None:
            return float('nan')

        num = np.nansum(vm * rt)
        den = np.nansum(vm)
        if den == 0.0 or not np.isfinite(den):
            self._last_rsi = float('nan')
            self._update_consistency_label()
            return float('nan')
        v = float(num / den)
        self._last_rsi = v
        self._update_consistency_label()
        return v

    @property
    def polarization_type(self) -> str:
        """Current polarization type (vm, vv, vh, vv/vh)."""
        return self._polarization_type

    @polarization_type.setter
    def polarization_type(self, v: str):
        """Set current polarization type."""
        Anisotropy.polarization_type.fset(self, v)

        if not all(hasattr(self, name) for name in ('radioButtonVM', 'radioButtonVV', 'radioButtonVH')):
            return

        # Block signals to prevent unwanted signal emissions
        self.radioButtonVM.blockSignals(True)
        self.radioButtonVV.blockSignals(True)
        self.radioButtonVH.blockSignals(True)

        # Update the radio buttons to reflect the current polarization type
        if self._polarization_type == 'vm':
            self.radioButtonVM.setChecked(True)
        elif self._polarization_type == 'vv':
            self.radioButtonVV.setChecked(True)
        elif self._polarization_type == 'vh':
            self.radioButtonVH.setChecked(True)

        # Unblock signals
        self.radioButtonVM.blockSignals(False)
        self.radioButtonVV.blockSignals(False)
        self.radioButtonVH.blockSignals(False)

        # Show/hide rotation parameters based on the selected polarization type
        self.hide_roation_parameters()

    # TODO: needs docstring
    def __init__(self, *args, **kwargs):
        """Initialize the instance."""
        super().__init__(*args, **kwargs)

        if getattr(self, 'model', None) is None:
            parent = kwargs.get('model', None)
            if parent is not None:
                self.model = parent
        if getattr(self, 'fit', None) is None:
            fit = kwargs.get('fit', None)
            if fit is None and getattr(self, 'model', None) is not None:
                fit = getattr(self.model, 'fit', None)
            if fit is not None:
                self.fit = fit

        self.setTitle("Rotational-times")
        self._last_rsl = float('nan')
        self._last_rsi = float('nan')
        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)

        self.setLayout(self.lh)
        self.rot_vis = False
        self._rho_widgets = list()
        self._b_widgets = list()

        self.radioButtonVM = QtWidgets.QRadioButton("VM")
        self.radioButtonVM.setToolTip(
            "Excitation: Vertical\nDetection: Magic-Angle"
        )
        self.radioButtonVM.setChecked(True)
        self.radioButtonVM.clicked.connect(
            lambda: self.set_polarization_type('vm')
        )

        self.radioButtonVV = QtWidgets.QRadioButton("VV")
        self.radioButtonVV.setToolTip(
            "Excitation: Vertical\nDetection: Vertical"
        )
        self.radioButtonVV.clicked.connect(
            lambda: self.set_polarization_type('vv')
        )

        self.radioButtonVH = QtWidgets.QRadioButton("VH")
        self.radioButtonVH.setToolTip(
            "Excitation: Vertical\nDetection: Horizontal"
        )
        self.radioButtonVH.clicked.connect(
            lambda: self.set_polarization_type('vh')
        )

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        add_rho = QtWidgets.QPushButton()
        add_rho.setText("add")
        add_rho.setCheckable(False)
        add_rho.setStyleSheet(ADD_BUTTON_STYLE)
        add_rho.clicked.connect(self.onAddRotation)

        remove_rho = QtWidgets.QPushButton()
        remove_rho.setText("del")
        remove_rho.setCheckable(False)
        remove_rho.setStyleSheet(REMOVE_BUTTON_STYLE)
        remove_rho.clicked.connect(self.onRemoveRotation)

        spacerItem = QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        layout.addItem(spacerItem)

        layout.addWidget(self.radioButtonVM)
        layout.addWidget(self.radioButtonVV)
        layout.addWidget(self.radioButtonVH)

        self.lh.addLayout(layout)

        self.gb = QtWidgets.QGroupBox()
        self.lh.addWidget(self.gb)

        self.lh = QtWidgets.QVBoxLayout()
        self.lh.setContentsMargins(0, 0, 0, 0)
        self.lh.setSpacing(0)
        self.gb.setLayout(self.lh)

        param_grid = QtWidgets.QGridLayout()
        param_grid.setContentsMargins(0, 0, 0, 0)
        param_grid.setHorizontalSpacing(6)
        param_grid.setVerticalSpacing(0)

        w_r0 = cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._r0,
            label_text='r<sub>0</sub>'
        )
        param_grid.addWidget(w_r0, 0, 0)
        w_g = cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._g,
            label_text='g'
        )
        param_grid.addWidget(w_g, 0, 1)

        w_l1 = cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._l1,
            label_text='l<sub>1</sub>',
            decimals=4
        )
        param_grid.addWidget(w_l1, 1, 0)
        w_l2 = cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._l2,
            label_text='l<sub>2</sub>',
            decimals=4
        )
        param_grid.addWidget(w_l2, 1, 1)

        self.lh.addLayout(param_grid)

        output_grid = QtWidgets.QGridLayout()
        output_grid.setContentsMargins(0, 0, 0, 0)
        output_grid.setHorizontalSpacing(6)
        output_grid.setVerticalSpacing(0)

        self._rss_l = FittingParameter(
            name='r_ss_l',
            value=self._compute_steady_state_anisotropy_lifetime,
            fixed=True,
            is_output=True,
            label_text='r<sub>S,L</sub>'
        )
        w_rss_l = cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._rss_l,
            label_text='r<sub>S,L</sub>',
            decimals=5
        )
        output_grid.addWidget(w_rss_l, 0, 0)

        self._rss_i = FittingParameter(
            name='r_ss_i',
            value=self._compute_steady_state_anisotropy_intensity,
            fixed=True,
            is_output=True,
            label_text='r<sub>S,I</sub>'
        )
        w_rss_i = cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._rss_i,
            label_text='r<sub>S,I</sub>',
            decimals=5
        )
        output_grid.addWidget(w_rss_i, 0, 1)

        self._quality_label = QtWidgets.QLabel()
        self._quality_label.setWordWrap(False)
        output_grid.addWidget(self._quality_label, 1, 0, 1, 2)
        self._update_consistency_label()

        control_row = QtWidgets.QHBoxLayout()
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(4)
        control_row.addWidget(add_rho)
        control_row.addWidget(remove_rho)
        control_row.addItem(
            QtWidgets.QSpacerItem(
                0,
                0,
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Minimum,
            )
        )

        diag_toggle = QtWidgets.QToolButton()
        diag_toggle.setText("VV/VH diag")
        diag_toggle.setCheckable(True)
        diag_toggle.setChecked(False)
        diag_toggle.toggled.connect(self._toggle_diag_visibility)
        control_row.addWidget(diag_toggle)

        decay_btn = QtWidgets.QToolButton()
        decay_btn.setText("show r(t)")
        decay_btn.clicked.connect(self._show_anisotropy_decay_dialog)
        control_row.addWidget(decay_btn)
        output_grid.addLayout(control_row, 2, 0, 1, 2)

        self._vv_bg_int = FittingParameter(
            name='vv_bg_int',
            value=self._compute_vv_bg_corrected_integral,
            fixed=True,
            is_output=True,
            label_text='VV<sub>bg-corr</sub>'
        )
        w_vv_bg = cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._vv_bg_int,
            label_text='VV<sub>bg-corr</sub>',
            decimals=3
        )
        self._w_vv_bg = w_vv_bg
        output_grid.addWidget(w_vv_bg, 3, 0)

        self._vh_bg_int = FittingParameter(
            name='vh_bg_int',
            value=self._compute_vh_bg_corrected_integral,
            fixed=True,
            is_output=True,
            label_text='VH<sub>bg-corr</sub>'
        )
        w_vh_bg = cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._vh_bg_int,
            label_text='VH<sub>bg-corr</sub>',
            decimals=3
        )
        self._w_vh_bg = w_vh_bg
        output_grid.addWidget(w_vh_bg, 3, 1)
        self._toggle_diag_visibility(False)

        self.lh.addLayout(output_grid)
        self.add_rotation()

        # Initialize radio buttons based on current polarization_type
        # Block signals during initialization
        self.radioButtonVM.blockSignals(True)
        self.radioButtonVV.blockSignals(True)
        self.radioButtonVH.blockSignals(True)

        if self._polarization_type.lower() == 'vv':
            self.radioButtonVV.setChecked(True)
        elif self._polarization_type.lower() == 'vh':
            self.radioButtonVH.setChecked(True)
        else:  # Default to VM
            self.radioButtonVM.setChecked(True)

        # Unblock signals after initialization
        self.radioButtonVM.blockSignals(False)
        self.radioButtonVV.blockSignals(False)
        self.radioButtonVH.blockSignals(False)

        self.hide_roation_parameters()

        try:
            self._install_code_badge()
        except Exception:
            pass

    def _install_code_badge(self):
        """Install a code badge for dev mode source jumping."""
        try:
            import chisurf.core.settings
            if not cs.core.settings.is_dev_mode():
                return
            if hasattr(self, '_chisurf_code_badge_installed'):
                return
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import resolve_object_source
            resolver = lambda: resolve_object_source(self)
            install_code_badge(self, resolver, corner='top-right', margin=4)
            self._chisurf_code_badge_installed = True
        except Exception:
            pass

    def set_polarization_type(self, pol_type: str):
        """
        Set the polarization type and update both the model and GUI.

        Parameters
        ----------
        pol_type : str
            The polarization type ('vm', 'vv', or 'vh')
        """
        # Update the widget's property
        Anisotropy.polarization_type.fset(self, pol_type)

        # Block signals to prevent unwanted signal emissions
        self.radioButtonVM.blockSignals(True)
        self.radioButtonVV.blockSignals(True)
        self.radioButtonVH.blockSignals(True)

        # Explicitly set the active radiobox
        if self._polarization_type == 'vm':
            self.radioButtonVM.setChecked(True)
        elif self._polarization_type == 'vv':
            self.radioButtonVV.setChecked(True)
        elif self._polarization_type == 'vh':
            self.radioButtonVH.setChecked(True)

        # Unblock signals
        self.radioButtonVM.blockSignals(False)
        self.radioButtonVV.blockSignals(False)
        self.radioButtonVH.blockSignals(False)

        # Show/hide rotation parameters based on the selected polarization type
        self.hide_roation_parameters()

    # TODO: needs docstring
    def hide_roation_parameters(self):
        """Show/hide rotation parameters based on polarization."""
        # Hide rotation parameters when VM is selected, show otherwise
        if self.radioButtonVM.isChecked():
            self.gb.hide()
        else:
            self.gb.show()

    # TODO: needs docstring
    def onAddRotation(self):
        """Handle add rotation button click."""
        cs.core.actions.dispatch(
            name="model.add_component",
            payload={"component_name": "anisotropy"},
        )

    # TODO: needs docstring
    def onRemoveRotation(self):
        """Handle remove rotation button click."""
        cs.core.actions.dispatch(
            name="model.remove_component",
            payload={"component_name": "anisotropy"},
        )

    # TODO: needs docstring
    def add_rotation(self, **kwargs):
        """Add a rotation component with GUI widget."""
        super().add_rotation(**kwargs)
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.lh.addLayout(layout)
        self._b_widgets.append(
            cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                fitting_parameter=self._bs[-1],
                decimals=4,
                layout=layout
            )
        )
        self._rho_widgets.append(
            cs.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
                fitting_parameter=self._rhos[-1],
                decimals=4,
                layout=layout
            )
        )

    # TODO: needs docstring
    def remove_rotation(self):
        """Remove the last rotation component and widget."""
        self._rhos.pop()
        self._bs.pop()
        self._rho_widgets.pop().close()
        self._b_widgets.pop().close()

    # TODO: needs docstring
    def append(self, *args, **kwargs):
        """Add a new component."""
        self.add_rotation(*args, **kwargs)

    # TODO: needs docstring
    def pop(self):
        """Remove the last component."""
        self.remove_rotation()
