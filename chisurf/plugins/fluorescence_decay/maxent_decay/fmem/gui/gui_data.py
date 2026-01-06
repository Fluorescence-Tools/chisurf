from __future__ import annotations

import json

import numpy as np

from chisurf.plugins.fluorescence_decay.maxent_decay.fmem import settings as maxent_settings
from .qt_stack import ensure_qt_stack


class _MaxentDataMixin:
    def _current_fit(self):
        _, _, _, chisurf, _ = ensure_qt_stack()
        try:
            return chisurf.cs.current_fit
        except Exception:
            return None

    def _describe_current_data(self) -> str:
        fit = self._current_fit()
        if fit is None or getattr(fit, "data", None) is None:
            return "No current fit / data"
        data = fit.data
        try:
            name = getattr(data, "name", None) or getattr(data, "filename", None)
        except Exception:
            name = None
        n_points = len(getattr(data, "y", []))
        return f"Current fit data: {name or 'unnamed'} (N={n_points})"

    def _on_refresh_data(self) -> None:
        # Changing the sample should clear any existing MEM results so the
        # user does not accidentally interpret old distributions or L-curve
        # diagnostics as belonging to the new data.
        try:
            self._reset_mem_result_state()
        except Exception:
            pass
        self.label_data_source.setText(self._describe_current_data())
        try:
            decay, dt, t = self._get_decay_and_dt()
        except Exception:
            return
        fit = self._current_fit()
        if fit is None:
            return

        try:
            xmin = int(getattr(fit, "xmin", 0))
            xmax = int(getattr(fit, "xmax", decay.size - 1))
        except Exception:
            xmin = 0
            xmax = decay.size - 1
        if decay.size > 0:
            xmin = max(0, min(xmin, decay.size - 1))
            xmax = max(xmin, min(xmax, decay.size - 1))
            self._fit_range = (xmin, xmax)
            if getattr(self, "_fit_region", None) is not None:
                self._fit_region.blockSignals(True)
                self._fit_region.setRegion((float(t[xmin]), float(t[xmax])))
                self._fit_region.blockSignals(False)

        try:
            model = fit.model
        except Exception:
            model = None
        if model is not None and hasattr(model, "convolve"):
            irf_curve = None
            try:
                convolve = model.convolve
                # Prefer the unnormalized IRF used for plotting in the
                # standard lifetime views; fall back to the normalized IRF
                # if needed.
                if hasattr(convolve, "unnormalized_irf"):
                    irf_curve = convolve.unnormalized_irf
                elif hasattr(convolve, "irf"):
                    irf_curve = convolve.irf
            except Exception:
                irf_curve = None

            if irf_curve is not None:
                self._irf_dataset = irf_curve
                try:
                    name = getattr(irf_curve, "name", None) or getattr(irf_curve, "filename", None)
                except Exception:
                    name = None
                if name:
                    self.label_irf_source.setText(f"IRF: model.convolve.irf ({name})")
                else:
                    self.label_irf_source.setText("IRF: model.convolve.irf (from current fit)")

        if model is not None:
            convolve = getattr(model, "convolve", None)
            generic = getattr(model, "generic", None)

            if convolve is not None:
                try:
                    ts_val = float(convolve.timeshift)
                except Exception:
                    ts_val = None
                if ts_val is not None:
                    try:
                        self.spin_timeshift.setValue(ts_val)
                    except Exception:
                        pass

                try:
                    irf_bg_val = float(convolve.lamp_background)
                except Exception:
                    irf_bg_val = None
                if irf_bg_val is not None:
                    try:
                        self.spin_irf_bg.setValue(irf_bg_val)
                    except Exception:
                        pass

            if generic is not None:
                try:
                    bg_val = float(generic.background)
                except Exception:
                    bg_val = None
                if bg_val is not None:
                    try:
                        self.spin_background.setValue(bg_val)
                    except Exception:
                        pass

                try:
                    scatter_val = float(generic.scatter)
                except Exception:
                    scatter_val = None
                if scatter_val is not None:
                    try:
                        self.spin_lamp_scatter.setValue(scatter_val)
                    except Exception:
                        pass

        try:
            lamp = self._build_irf_array(decay.size, t, dt)
        except Exception:
            lamp = None
        try:
            self._plot_decay_and_irf(decay, t, lamp)
        except Exception:
            pass
        if lamp is not None:
            try:
                fwhm = self._estimate_irf_fwhm(t, lamp)
            except Exception:
                fwhm = None
            if fwhm is not None and fwhm > 0.0:
                try:
                    self.spin_tau_min.setValue(float(fwhm))
                except Exception:
                    pass

        # For FRET mode, if no explicit period has been set by the user,
        # initialize the period spin box from the decay time axis
        # (approximate excitation period as the acquisition window).
        try:
            if t is not None and np.asarray(t, dtype=float).size > 1:
                t_arr = np.asarray(t, dtype=float).ravel()
                period_guess = float(t_arr[-1] - t_arr[0])
                if period_guess > 0.0:
                    self.spin_period.setValue(period_guess)
        except Exception:
            pass

    def _ensure_irf_selector(self):
        _, _, _, _, ExperimentalDataSelector = ensure_qt_stack()
        if getattr(self, "_irf_selector", None) is not None:
            return
        self._irf_selector = ExperimentalDataSelector(
            parent=None,
            change_event=self._on_irf_selection_changed,
            context_menu_enabled=False,
        )

    def _on_edit_settings_clicked(self) -> None:
        """Open a simple JSON editor for the MaxEnt settings file.

        This loads the current settings JSON into a text editor with JSON
        highlighting (when the generic ChiSurf text editor is available) or
        a plain text fallback. On save, the JSON is validated and written
        back to the MaxEnt settings file and ``self._settings`` is
        reloaded. UI widgets keep their current values until the user
        restarts the widget or re-opens it; the JSON mainly controls base
        defaults.
        """

        _, QtWidgets, _, _, _ = ensure_qt_stack()

        # Load the raw file contents, creating the file from defaults if
        # needed.
        try:
            path = maxent_settings.get_settings_file()
        except Exception:
            path = None

        current_dict: dict = {}
        if path is not None:
            try:
                current_dict = maxent_settings.load_maxent_settings()
            except Exception:
                current_dict = {}

        # Serialize with nice formatting for editing.
        try:
            initial_text = json.dumps(current_dict, indent=2, sort_keys=True)
        except Exception:
            initial_text = "{}"

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Edit MaxEnt JSON settings")
        layout = QtWidgets.QVBoxLayout(dialog)

        editor = None
        try:
            # Prefer the generic ChiSurf text editor with JSON highlighting
            # if available.
            import chisurf.plugins.misc.code_editor.text_editor as _te  # type: ignore

            editor = _te.TextEditor(dialog, language="json")
            editor.setText(initial_text)
        except Exception:
            editor = QtWidgets.QPlainTextEdit(dialog)
            editor.setPlainText(initial_text)

        layout.addWidget(editor)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        layout.addWidget(buttons)

        def _on_accept() -> None:
            text = editor.text() if hasattr(editor, "text") else editor.toPlainText()
            try:
                data = json.loads(text)
                if not isinstance(data, dict):
                    raise ValueError("Top-level JSON value must be an object")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    dialog,
                    "Invalid JSON",
                    f"The settings file must contain valid JSON object data.\n\nError: {exc}",
                )
                return

            # Persist via helper so we reuse the same logic everywhere.
            ok = maxent_settings.save_maxent_settings(data)
            if not ok:
                QtWidgets.QMessageBox.critical(
                    dialog,
                    "Save error",
                    "Failed to write MaxEnt settings.json.",
                )
                return

            # Refresh cached settings for future runs.
            try:
                self._settings = maxent_settings.load_maxent_settings()
            except Exception:
                self._settings = {}

            dialog.accept()

        buttons.accepted.connect(_on_accept)
        buttons.rejected.connect(dialog.reject)

        dialog.exec_()

    def _on_select_irf_clicked(self) -> None:
        self._ensure_irf_selector()
        try:
            self._irf_selector.show()
        except Exception:
            pass

    def _on_clear_irf_clicked(self) -> None:
        self._irf_dataset = None
        self.label_irf_source.setText("IRF: model.convolve.irf (default)")

    def _on_irf_selection_changed(self) -> None:
        try:
            ds = self._irf_selector.selected_dataset
        except Exception:
            ds = None
        self._irf_dataset = ds
        if ds is None:
            self.label_irf_source.setText("IRF: model.convolve.irf (default)")
        else:
            try:
                name = getattr(ds, "name", None) or getattr(ds, "filename", None)
            except Exception:
                name = None
            self.label_irf_source.setText(f"IRF: {name or 'dataset'}")

    def _on_fit_region_changed(self) -> None:
        if getattr(self, "_fit_region", None) is None:
            return
        if self._t_axis is None:
            return
        t = self._t_axis
        if t.size == 0:
            return
        lb, ub = self._fit_region.getRegion()
        start = int(np.searchsorted(t, lb, side="left"))
        stop = int(np.searchsorted(t, ub, side="right") - 1)
        n = t.size
        if n <= 0:
            return
        start = max(0, min(start, n - 1))
        stop = max(start, min(stop, n - 1))
        self._fit_range = (start, stop)
