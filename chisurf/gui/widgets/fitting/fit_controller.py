from __future__ import annotations

import os
import time
import typing
import pathlib
import textwrap

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, uic, QtCore, QtGui
import matplotlib.colors as mcolors

import chisurf as cs
import chisurf.logging
import chisurf.core.data
import chisurf.core.fitting
import chisurf.core.decorators
import chisurf.gui.decorators
import chisurf.core.settings

import chisurf.gui.widgets
import chisurf.gui.widgets.experiments.widgets
from chisurf.gui.widgets.general import Controller
from chisurf.core.math.optimization.leastsqbound import OptimizationCancelled
from chisurf.core.actions import record_action
from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client


class FittingControllerWidget(Controller):

    @staticmethod
    def _iter_fit_parameters_to_finalize(fit):
        """Yield parameters owned by the current fit or fit group."""
        local_fits = list(getattr(fit, "grouped_fits", []))
        if not local_fits:
            local_fits = [fit]
        for local_fit in local_fits:
            model = getattr(local_fit, "model", None)
            yield from getattr(model, "parameters_all", [])

    def _collect_parameter_snapshot(self) -> typing.List[typing.Dict[str, typing.Any]]:
        snapshot: typing.List[typing.Dict[str, typing.Any]] = []
        try:
            fit_group_name = str(getattr(self.fit, "name", ""))
            local_fits = list(getattr(self.fit, "grouped_fits", []))
            if not local_fits:
                local_fits = [self.fit]
            for local_fit in local_fits:
                local_fit_name = str(getattr(local_fit, "name", ""))
                model = getattr(local_fit, "model", None)
                if model is None:
                    continue
                for param in getattr(model, "parameters_all", []):
                    try:
                        bounds = getattr(param, "bounds", None)
                        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                            lb = float(bounds[0])
                            ub = float(bounds[1])
                        else:
                            lb = None
                            ub = None
                    except Exception:
                        lb = None
                        ub = None
                    snapshot.append({
                        "fit_group": fit_group_name,
                        "local_fit": local_fit_name,
                        "parameter_name": str(getattr(param, "name", "")),
                        "value": float(getattr(param, "value", 0.0)),
                        "fixed": bool(getattr(param, "fixed", False)),
                        "bounds_on": bool(getattr(param, "bounds_on", False)),
                        "lower": lb,
                        "upper": ub,
                    })
        except Exception:
            return []
        return snapshot

    def _collect_fit_range_snapshot(self) -> typing.List[typing.Dict[str, typing.Any]]:
        snapshot: typing.List[typing.Dict[str, typing.Any]] = []
        try:
            fit_group_name = str(getattr(self.fit, "name", ""))
            local_fits = list(getattr(self.fit, "grouped_fits", []))
            if not local_fits:
                local_fits = [self.fit]
            for local_fit in local_fits:
                xmin, xmax = getattr(local_fit, "fit_range", (None, None))
                snapshot.append({
                    "fit_group": fit_group_name,
                    "local_fit": str(getattr(local_fit, "name", "")),
                    "xmin": int(xmin),
                    "xmax": int(xmax),
                })
        except Exception:
            return []
        return snapshot

    def _record_history(self, action_type: str, summary: str, payload: typing.Optional[typing.Dict[str, typing.Any]] = None) -> None:
        try:
            source_uid = str(getattr(self.fit, "unique_identifier", ""))
            if str(action_type) in {"fit_run_start", "fit_run_finish", "fit_run_abort"}:
                cs.core.actions.dispatch(
                    name=str(action_type).replace("_", "."),
                    payload=payload or {},
                )
                return
            record_action(
                action_type=action_type,
                summary=summary,
                payload=payload,
                source_uid=source_uid or None,
            )
            return
        except Exception:
            pass
        try:
            cs.logging.info(f"# HIST {action_type}: {summary}")
        except Exception:
            pass

    @property
    def selected_fit(self) -> int:
        return int(self.comboBox.currentIndex())

    @selected_fit.setter
    def selected_fit(
            self,
            v: int
    ):
        self.comboBox.setCurrentIndex(int(v))

    @property
    def current_fit_type(self) -> str:
        return str(self.comboBox.currentText())

    @property
    def local_first(self) -> bool:
        return self.checkBox.isChecked()

    @property
    def n_steps(self) -> int:
        return int(self.doubleSpinBox.value() * 1000)

    @property
    def n_runs(self) -> int:
        return self.spinBox_5.value()

    def _format_dataset_label(self, name: str, max_length: int = 40) -> str:
        try:
            s = str(name)
        except Exception:
            return name
        if not s:
            return s
        try:
            max_len = int(max_length)
        except Exception:
            max_len = 40
        if max_len < 7 or len(s) <= max_len:
            return s
        keep_total = max_len - 4  # reserve 4 characters for '....'
        start_keep = keep_total // 2
        end_keep = keep_total - start_keep
        return f"{s[:start_keep]}....{s[-end_keep:]}"

    def _update_combo_tooltip(self, index: int) -> None:
        try:
            full_name = self.comboBox.itemData(index, QtCore.Qt.ToolTipRole)
        except Exception:
            full_name = None
        if not full_name:
            try:
                full_name = self.comboBox.itemText(index)
            except Exception:
                full_name = ""
        try:
            self.comboBox.setToolTip(str(full_name))
        except Exception:
            pass

    def change_dataset(self) -> None:
        dataset = self.curve_select.selected_dataset
        fc = get_fitting_client()
        if fc is not None:
            fit_index = int(getattr(self.fit, "fit_idx", 0))
            dataset_uid = str(getattr(dataset, "unique_identifier", "") or "")
            fc.set_fit_dataset(
                fit_index=fit_index,
                dataset_uid=dataset_uid,
            )
        full_name = os.path.basename(
            getattr(dataset, 'name', getattr(dataset, 'filename', ''))
        )
        display_name = self._format_dataset_label(full_name)
        idx = self.comboBox.currentIndex()
        self.comboBox.setItemText(idx, display_name)
        try:
            self.comboBox.setItemData(idx, full_name, QtCore.Qt.ToolTipRole)
        except Exception:
            pass
        self._update_combo_tooltip(idx)

    def show_selector(self):
        self.curve_select.show()
        self.curve_select.update()

    def __init__(
            self,
            fit: cs.core.fitting.fit.FitGroup = None,
            hide_fit_button: bool = False,
            hide_range: bool = False,
            hide_fitting: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.fit = fit
        self.curve_select = cs.gui.widgets.experiments.widgets.ExperimentalDataSelector(
            parent=None,
            fit=fit,
            change_event=self.change_dataset,
            experiment=fit.data.experiment.__class__
        )

        uic.loadUi(pathlib.Path(__file__).parent / "fittingWidget.ui", self)

        self.curve_select.hide()
        if fit is not None:
            for f in fit:
                data = getattr(f, 'data', None)
                try:
                    base_name = os.path.basename(
                        getattr(data, 'name', getattr(data, 'filename', ''))
                    )
                except Exception:
                    base_name = getattr(data, 'name', 'Unknown')
                display_name = self._format_dataset_label(base_name)
                self.comboBox.addItem(display_name)
                idx = self.comboBox.count() - 1
                try:
                    self.comboBox.setItemData(idx, base_name, QtCore.Qt.ToolTipRole)
                except Exception:
                    pass
        try:
            self.comboBox.currentIndexChanged.connect(self._update_combo_tooltip)
            self._update_combo_tooltip(self.comboBox.currentIndex())
        except Exception:
            pass

        # decorate the update method of the fit
        # after decoration it should also call the update of
        # the fitting widget
        def wrapper(f):

            def update_new(*args, **kwargs):
                f(*args, **kwargs)
                self.update(*args)
            return update_new

        self.fit.run = wrapper(self.fit.run)

        self.actionFit.triggered.connect(self.onRunFit)
        self.actionAutoFitRange.triggered.connect(self.onAutoFitRange)
        self.actionFit_range_changed.triggered.connect(self.onFitRangeChanged)
        self.actionChange_dataset.triggered.connect(self.show_selector)
        self.actionSelectionChanged.triggered.connect(self.onDatasetChanged)
        self.actionErrorEstimate.triggered.connect(self.onErrorEstimate)

        self.spinBox_3.valueChanged.connect(self._result_changed)

        # Detect whether the current dataset is intrinsically multidimensional
        # based on generic grid metadata (data.meta_data['grid']). For such
        # datasets we enable all four range spin boxes and interpret them as
        # 2D bounds; for purely 1D datasets we keep the original two spin
        # boxes and hide/disable the extra pair. This keeps the controller
        # agnostic of specific experiments/models.
        self._is_2d_dataset = False
        self._2d_shape = None
        self._grid_meta = {}
        self._grid_order = None
        self._init_dimensionality()

        if hide_fit_button:
            self.button_fit.hide()
        if hide_range:
            self.button_auto_fit_range.hide()
            self.spinBox.hide()
            self.spinBox_2.hide()
        if hide_fitting:
            self.hide()

        self._apply_proteinmc_controls()

        try:
            self.comboBox.currentIndexChanged.connect(lambda *_args: self._apply_proteinmc_controls())
        except Exception:
            pass

        try:
            self._install_code_badge()
        except Exception:
            pass

    def _apply_proteinmc_controls(self) -> None:
        """Disable generic fitting controls when this controller hosts ProteinMC."""

        if self._is_proteinmc_fit():
            self.actionFit.setEnabled(False)
            self.groupBox.hide()
        else:
            self.groupBox.show()
            self.actionFit.setEnabled(True)
            self.button_fit.setEnabled(True)

        if self._model_sampling_handler() is not None or self._is_proteinmc_fit():
            self.button_sample.setToolTip(
                "Start model-defined sampling. For ProteinMC, run length is controlled "
                "by the MC trials, Save every, and Max frames fields below."
            )
        if self._is_proteinmc_fit():
            self.doubleSpinBox.setEnabled(True)
            self.doubleSpinBox.setToolTip("ProteinMC MC trials, in thousands.")
            self.spinBox_5.setEnabled(True)
            self.spinBox_5.setToolTip("Number of independent ProteinMC runs to launch.")
            steps_label = getattr(self, "label", None)
            if steps_label is not None:
                steps_label.setEnabled(True)
                steps_label.setToolTip("ProteinMC MC trials, in thousands.")
            runs_label = getattr(self, "label_2", None)
            if runs_label is not None:
                runs_label.setEnabled(True)
                runs_label.setToolTip("Number of independent ProteinMC runs to launch.")
        else:
            for widget in (self.doubleSpinBox, self.spinBox_5):
                widget.setEnabled(True)
                widget.setToolTip("")
            for widget_name in ("label", "label_2"):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.setEnabled(True)
                    widget.setToolTip("")
    def _candidate_sampling_models(self) -> list:
        """Return models that may handle the Sampling button themselves."""

        models = []

        direct_model = getattr(self.fit, "model", None)
        if direct_model is not None:
            models.append(direct_model)
        selected_fit = getattr(self.fit, "selected_fit", None)
        selected_model = getattr(selected_fit, "model", None)
        if selected_model is not None and selected_model not in models:
            models.append(selected_model)

        fits = []
        grouped_fits = getattr(self.fit, "grouped_fits", None)
        if grouped_fits:
            try:
                fits.extend(list(grouped_fits))
            except Exception:
                pass
        if not fits:
            try:
                fits = list(self.fit)
            except Exception:
                fits = [self.fit]
        try:
            index = int(self.selected_fit)
        except Exception:
            index = 0
        if index < 0 or index >= len(fits):
            index = 0
        fit = fits[index] if fits else self.fit
        model = getattr(fit, "model", None)
        if model is not None and model not in models:
            models.append(model)
        for fit in fits:
            model = getattr(fit, "model", None)
            if model is not None and model not in models:
                models.append(model)
        return models

    def _model_sampling_handler(self):
        """Return a model-defined Sampling-button handler if one exists."""

        for model in self._candidate_sampling_models():
            for method_name in ("run_sampling", "sample", "on_sample", "start_sampling"):
                method = getattr(model, method_name, None)
                if callable(method):
                    return method
        return None

    def _is_proteinmc_fit(self) -> bool:
        """Return True if any candidate model is ProteinMC."""

        for model in self._candidate_sampling_models():
            model_name = str(getattr(model, "name", "") or getattr(model.__class__, "name", ""))
            class_name = str(getattr(model.__class__, "__name__", ""))
            if model_name == "ProteinMC" or class_name == "ProteinMCModelWidget":
                return True
        return False

    def _proteinmc_model_widget(self):
        """Return the active ProteinMC model widget, if this fit uses one."""

        for model in self._candidate_sampling_models():
            model_name = str(getattr(model, "name", "") or getattr(model.__class__, "name", ""))
            class_name = str(getattr(model.__class__, "__name__", ""))
            if model_name == "ProteinMC" or class_name == "ProteinMCModelWidget":
                return model
        return None

    def _install_code_badge(self):
        """Install a code badge for dev mode source jumping."""
        try:
            import chisurf.core.settings
            if not cs.core.settings.is_dev_mode():
                return
            if hasattr(self, '_chisurf_code_badge_installed'):
                return
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import resolve_fit_window_source
            resolver = lambda: resolve_fit_window_source(self)
            install_code_badge(self, resolver, corner='top-right', margin=4)
            self._chisurf_code_badge_installed = True
        except Exception:
            pass

    def _result_changed(self):
        result_idx = self.spinBox_3.value() - 1
        fc = get_fitting_client()
        if fc is not None:
            fc.set_fit_result_idx(
                fit_index=getattr(self.fit, "fit_idx", 0),
                result_idx=result_idx,
            )

    def onDatasetChanged(self):
        fc = get_fitting_client()
        if fc is not None:
            fc.group_select_member(
                fit_uid=str(getattr(self.fit, "unique_identifier", "") or ""),
                member_index=self.selected_fit,
            )

    def onErrorEstimate(self):
        sampling_handler = self._model_sampling_handler()
        if sampling_handler is not None:
            target_dir, _ = cs.gui.widgets.get_directory(caption="Select Sampling Output Folder")
            if target_dir is None:
                cs.logging.info("Model-defined sampling canceled!")
                return
            # Model-specific sampling (e.g. ProteinMC) must run its own
            # algorithm instead of the generic emcee server path, which
            # assumes a curve-based model.
            try:
                sampling_handler(
                    output_directory=target_dir,
                    run_count=self.n_runs,
                    n_iter=self.n_steps,
                )
                cs.logging.info("Model-defined sampling started.")
            except Exception:
                cs.logging.exception("Model-defined sampling failed")
            return
        if self._is_proteinmc_fit():
            cs.logging.warning("ProteinMC must handle Sampling itself; refusing to run generic emcee sampling.")
            return

        fit_name = str(getattr(self.fit, "name", ""))
        cs.logging.info(f"Sampling analysis: {fit_name}")
        target_dir, _ = cs.gui.widgets.get_directory(caption="Select Target Folder for Sampling Results")
        if target_dir is None:
            cs.logging.info("Sampling canceled!")
            return
        
        target_dir_str = str(target_dir)
        
        kw = cs.core.settings.cs_settings['optimization']['sampling'].copy()
        kw['n_runs'] = self.n_runs
        kw['steps'] = self.n_steps
        
        fc = get_fitting_client()
        if fc is not None:
            fc.start_sampling(
                fit_uid=str(getattr(self.fit, "unique_identifier", "") or ""),
                n_steps=self.n_steps,
                n_runs=self.n_runs,
                target_directory=target_dir_str,
            )
            cs.logging.info("Sampling started on server.")

    def _run_fit_impl(self):
        if self._proteinmc_model_widget() is not None:
            self._apply_proteinmc_controls()
            cs.logging.info("ProteinMC does not use generic Fit. Use Sampling to start ProteinMC.")
            return
        try:
            fit_name = str(getattr(self.fit, "name", ""))
        except Exception:
            fit_name = ""
        cs.logging.info(f"Please wait fitting: {fit_name}")

        try:
            wrapped_name = cs.gui.widgets.progress.wrap_text(fit_name, width=48, max_lines=3)
        except Exception:
            wrapped_name = fit_name
        if "\n" in wrapped_name:
            base_label = f"Fitting:\n{wrapped_name}"
        else:
            base_label = f"Fitting {wrapped_name}..." if wrapped_name else "Fitting..."
        t0 = time.perf_counter()
        before_snapshot = self._collect_parameter_snapshot()
        before_fit_range = self._collect_fit_range_snapshot()
        self._record_history(
            action_type="fit_run_start",
            summary=f"start fit run: {fit_name}",
            payload={
                "fit_name": fit_name,
                "local_first": bool(self.local_first),
                "n_steps": int(self.n_steps),
                "n_runs": int(self.n_runs),
                "parameter_snapshot_before": before_snapshot,
                "fit_range_snapshot_before": before_fit_range,
            },
        )

        dialog = None
        success = False
        try:
            # Create a modal progress dialog if the GUI helpers are available.
            try:
                dialog = cs.gui.widgets.progress.EnhancedProgressDialog(
                    title="Fitting",
                    label_text=base_label,
                    min_value=0,
                    max_value=100,
                    parent=self,
                )
                dialog.show()
                dialog.update_progress(0)
            except Exception:
                dialog = None

            def _on_progress(done: int, total: int, chi2=None, chi2r=None, **_kwargs) -> None:
                """Update the progress dialog from least-squares callbacks.

                The callback receives the number of completed residual
                evaluations (done) and an estimated total evaluation
                budget (total). It maps this to a 0–100 percentage.

                When the user presses the Cancel button on the progress
                dialog, this callback raises :class:`OptimizationCancelled`
                so that the optimizer aborts cleanly while keeping the
                current parameter values.
                """

                if dialog is None:
                    return

                # Honour user cancellation as soon as possible. The
                # least-squares wrapper treats this exception specially
                # and propagates it back to :meth:`onRunFit`.
                try:
                    if dialog.wasCanceled():
                        raise OptimizationCancelled()
                except OptimizationCancelled:
                    raise
                except Exception:
                    # Ignore unexpected UI errors when checking cancel state.
                    pass

                try:
                    total_val = float(total) if total else 0.0
                except Exception:
                    total_val = 0.0
                if total_val <= 0.0:
                    value = 0
                else:
                    try:
                        frac = float(done) / total_val
                    except Exception:
                        frac = 0.0
                    if frac < 0.0:
                        frac = 0.0
                    if frac > 1.0:
                        frac = 1.0
                    value = int(round(100.0 * frac))
                # Build an informative status line including objective values
                # when available.
                parts = [f"eval {done}/{int(total) if total else '?'}"]
                if done > 0 and total:
                    elapsed = time.perf_counter() - t0
                    remaining = (elapsed / float(done)) * (float(total) - done)
                    if remaining > 3600:
                        parts.append(f"ETA: {int(remaining // 3600)}h {int((remaining % 3600) // 60)}m")
                    elif remaining > 60:
                        parts.append(f"ETA: {int(remaining // 60)}m {int(remaining % 60)}s")
                    else:
                        parts.append(f"ETA: {int(remaining)}s")

                if chi2 is not None:
                    try:
                        parts.append(f"chi2={float(chi2):.3g}")
                    except Exception:
                        pass
                if chi2r is not None:
                    try:
                        parts.append(f"chi2r={float(chi2r):.3g}")
                    except Exception:
                        pass

                label_text = base_label + "\n" + "  |  ".join(parts)

                try:
                    dialog.update_progress(value, text=label_text)
                except Exception:
                    # Never let UI errors break the optimizer.
                    pass

            # Run the fit synchronously, allowing the optimizer to invoke
            # the progress callback from within the residual evaluations.
            try:
                fc = get_fitting_client()
                if fc is not None:
                    fc.run_fit(
                        fit_uid=str(getattr(self.fit, "unique_identifier", "") or ""),
                    )
            except OptimizationCancelled:
                cs.logging.info("Fitting cancelled by user.")
                success = False
            else:
                if fc is not None:
                    fc.model_finalize(
                        fit_uid=str(getattr(self.fit, "unique_identifier", "") or ""),
                    )
                for pa in self._iter_fit_parameters_to_finalize(self.fit):
                    controller = getattr(pa, "controller", None)
                    if controller is None:
                        continue
                    try:
                        controller.finalize()
                    except (AttributeError, RuntimeError, TypeError):
                        cs.logging.warning(
                            f"Fitting parameter {pa.name} failed to update its controller."
                        )
                cs.logging.info("Fitting finished!")
                success = True

                # Update fit result selector
                self.spinBox_3.setMaximum(len(self.fit.results))
                self.spinBox_3.setMinimum(1)
                self.spinBox_3.setValue(1)
        finally:
            if dialog is not None:
                try:
                    final_text = "Fitting finished!" if success else "Fitting aborted."
                    # Close immediately by default; user can override via settings.
                    try:
                        delay_ms = int(cs.core.settings.gui.get('fit_progress_close_delay_ms', 0))
                    except Exception:
                        delay_ms = 0
                    dialog.finish(final_text=final_text, auto_close=True, close_delay_ms=delay_ms)
                except Exception:
                    try:
                        dialog.finalize(force_auto_close=True)
                    except Exception:
                        pass

        elapsed_ms = int(round((time.perf_counter() - t0) * 1000.0))
        after_snapshot = self._collect_parameter_snapshot()
        after_fit_range = self._collect_fit_range_snapshot()
        self._record_history(
            action_type="fit_run_finish" if success else "fit_run_abort",
            summary=(
                f"fit {'finished' if success else 'aborted'}: {self.fit.name} "
                f"({elapsed_ms} ms)"
            ),
            payload={
                "fit_name": str(getattr(self.fit, "name", "")),
                "success": bool(success),
                "elapsed_ms": int(elapsed_ms),
                "result_count": int(len(getattr(self.fit, "results", []))),
                "parameter_snapshot_after": after_snapshot,
                "fit_range_snapshot_after": after_fit_range,
            },
        )

    def onRunFit(self):
        proteinmc_model = self._proteinmc_model_widget()
        if proteinmc_model is not None:
            self._apply_proteinmc_controls()
            cs.logging.info("ProteinMC does not use generic Fit. Use Sampling to start ProteinMC.")
            return
        fc = get_fitting_client()
        if fc is not None:
            self._run_fit_impl()

    @property
    def xmin(self):
        return int(self.spinBox_2.value())

    @xmin.setter
    def xmin(self, v: int):
        self.spinBox_2.setValue(v)

    @property
    def xmax(self):
        return int(self.spinBox.value())

    @xmax.setter
    def xmax(self, v: int):
        self.spinBox.setValue(v)

    @property
    def xmin2(self) -> int:
        return int(self.spinBox_4.value())

    @xmin2.setter
    def xmin2(self, v: int):
        self.spinBox_4.setValue(v)

    @property
    def xmax2(self) -> int:
        return int(self.spinBox_6.value())

    @xmax2.setter
    def xmax2(self, v: int):
        self.spinBox_6.setValue(v)

    def onFitRangeChanged(self, event, xmin: int = None, xmax: int = None):
        cs.logging.info(f'onFitRangeChanged: {xmin, xmax}')
        if xmin is not None:
            self.xmin = xmin
        if xmax is not None:
            self.xmax = xmax
        fc = get_fitting_client()
        if fc is not None:
            fc.set_fit_range(
                fit_uid=str(getattr(self.fit, "unique_identifier", "") or ""),
                xmin=self.xmin,
                xmax=self.xmax,
            )
        if getattr(self, '_is_2d_dataset', False):
            try:
                self._update_2d_mask_from_spinboxes()
            except Exception as e:
                cs.logging.warning(f'Failed to update 2D mask from spinboxes: {e}')
        if getattr(self, '_auto_fit_range_in_progress', False):
            return
        if fc is not None:
            fc.update_fit(
                fit_uid=str(getattr(self.fit, "unique_identifier", "") or ""),
            )


    def onAutoFitRange(self):
        """Apply the reader-provided default fit range and update the fit."""
        try:
            fc = get_fitting_client()
            fit_uid_val = str(getattr(self.fit, "unique_identifier", "") or "")
            range_applied_by_rpc = False
            if fc is not None:
                result = fc.auto_fit_range(fit_uid=fit_uid_val)
                if result.get("ok"):
                    xmin_1d, xmax_1d = result.get("xmin", 0), result.get("xmax", 0)
                    range_applied_by_rpc = bool(result.get("applied"))
                else:
                    try:
                        xmin_1d, xmax_1d = self.fit.data.data_reader.autofitrange(self.fit.data)
                    except Exception:
                        return
            else:
                try:
                    xmin_1d, xmax_1d = self.fit.data.data_reader.autofitrange(self.fit.data)
                except Exception:
                    return

            cs.logging.info(f'onAutoFitRange: {xmin_1d, xmax_1d}')

            try:
                self._auto_fit_range_in_progress = True
            except Exception:
                pass

            deferred_update_scheduled = False
            blocked_widgets = []
            for widget in (self.spinBox_2, self.spinBox_4, self.spinBox, self.spinBox_6):
                try:
                    blocked_widgets.append((widget, widget.blockSignals(True)))
                except Exception:
                    pass

            try:
                if getattr(self, '_is_2d_dataset', False) and self._2d_shape is not None:
                    ny, nx = int(self._2d_shape[0]), int(self._2d_shape[1])
                    self.spinBox_2.setRange(0, max(0, nx - 1))
                    self.spinBox_4.setRange(0, max(0, nx - 1))
                    self.spinBox.setRange(0, max(0, ny - 1))
                    self.spinBox_6.setRange(0, max(0, ny - 1))
                    self.spinBox_2.setValue(0)
                    self.spinBox_4.setValue(max(0, nx - 1))
                    self.spinBox.setValue(0)
                    self.spinBox_6.setValue(max(0, ny - 1))
                    try:
                        n_flat = int(len(self.fit.data.y))
                    except Exception:
                        n_flat = max(0, int(xmax_1d))
                    if fc is not None:
                        fc.set_fit_range(fit_uid=fit_uid_val, xmin=0, xmax=n_flat)
                    else:
                        self.fit.fit_range = (0, n_flat)
                    try:
                        self._update_2d_mask_from_spinboxes()
                    except Exception as e:
                        cs.logging.warning(f'Failed to update 2D mask after 2D autofitrange: {e}')
                else:
                    self.xmin, self.xmax = (xmin_1d, xmax_1d)
                    if fc is not None and not range_applied_by_rpc:
                        fc.set_fit_range(fit_uid=fit_uid_val, xmin=xmin_1d, xmax=xmax_1d)
                    elif fc is None:
                        self.fit.fit_range = (int(xmin_1d), int(xmax_1d))

                fit = self.fit
                xmin_val = int(self.xmin)
                xmax_val = int(self.xmax)
                is_2d = bool(getattr(self, "_is_2d_dataset", False))
                xmin2_val = int(self.xmin2) if is_2d else 0
                xmax2_val = int(self.xmax2) if is_2d else 0

                def _do_deferred_update():
                    try:
                        if fc is None:
                            try:
                                fit.update()
                            except Exception as e:
                                cs.logging.warning(f"Local fit update after auto fit range failed: {e}")
                        try:
                            payload = {
                                "fit_group": str(getattr(fit, "name", "")),
                                "xmin": xmin_val,
                                "xmax": xmax_val,
                                "source": "auto_fit_range",
                                "is_2d": is_2d,
                            }
                            if is_2d:
                                payload.update({
                                    "x_min": xmin_val,
                                    "x_max": xmin2_val,
                                    "y_min": xmax_val,
                                    "y_max": xmax2_val,
                                })
                            self._record_history(
                                action_type="fit_range_set",
                                summary=f"auto fit range for '{getattr(fit, 'name', '')}' to [{xmin_val}, {xmax_val})",
                                payload=payload,
                            )
                        except Exception:
                            pass
                    finally:
                        try:
                            self._auto_fit_range_in_progress = False
                        except Exception:
                            pass

                QtCore.QTimer.singleShot(0, _do_deferred_update)
                deferred_update_scheduled = True
            finally:
                for widget, previous_state in reversed(blocked_widgets):
                    try:
                        widget.blockSignals(previous_state)
                    except Exception:
                        pass
                if not deferred_update_scheduled:
                    try:
                        self._auto_fit_range_in_progress = False
                    except Exception:
                        pass
        except Exception as e:
            cs.logging.warning(f"onAutoFitRange failed: {e}")
            try:
                self._auto_fit_range_in_progress = False
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Dimensionality and 2D mask helpers
    # ------------------------------------------------------------------

    def _init_dimensionality(self) -> None:
        """Detect whether the attached dataset exposes a generic grid.

        Detection is based solely on ``data.meta_data['grid']``, which is a
        dictionary with at least the following keys when present:

        - ``ndim``: int
            Number of logical grid dimensions. Only ``ndim == 2`` is
            currently supported by this widget.
        - ``shape``: tuple
            Grid shape ``(ny, nx)`` used when reconstructing 2D selections
            from the 1D flattened data arrays.
        - ``order``: str, optional
            NumPy-style memory order string used to map the 2D grid to the
            1D data vector. Known values are::

                'C'  # row-major flattening (default)
                'F'  # column-major flattening

            Some experiments may additionally provide explicit index arrays
            (e.g. ``row_indices`` / ``col_indices``) when the 1D vector is a
            sparse view of the grid. These are used by
            :meth:`_update_2d_mask_from_spinboxes` to translate 2D rectangles
            into 1D masks when present.

        Experiment-specific readers (e.g. PDA, RICS) are responsible for
        populating this metadata; the controller itself stays agnostic of the
        concrete experiment/model types.
        """

        data = None
        try:
            data = self.fit.data
        except Exception:
            pass

        # Reset cached dimensionality state
        self._is_2d_dataset = False
        self._2d_shape = None
        self._grid_meta = {}
        self._grid_flattening = None

        if data is None:
            return

        try:
            meta_all = getattr(data, 'meta_data', {}) or {}
        except Exception:
            meta_all = {}
        grid_meta = meta_all.get('grid', {}) or {}

        try:
            ndim = int(grid_meta.get('ndim', 1))
        except Exception:
            ndim = 1
        shape = grid_meta.get('shape', None)

        if ndim == 2 and shape is not None:
            try:
                ny, nx = int(shape[0]), int(shape[1])
                if ny > 0 and nx > 0:
                    self._is_2d_dataset = True
                    self._2d_shape = (ny, nx)
                    self._grid_meta = grid_meta
                    self._grid_order = grid_meta.get('order', 'C')
            except Exception:
                pass

        # Configure spin boxes according to dimensionality
        try:
            if self._is_2d_dataset and self._2d_shape is not None:
                ny, nx = int(self._2d_shape[0]), int(self._2d_shape[1])
                # x-axis: columns (0 .. nx-1)
                self.spinBox_2.setRange(0, max(0, nx - 1))
                self.spinBox_4.setRange(0, max(0, nx - 1))
                # y-axis: rows (0 .. ny-1)
                self.spinBox.setRange(0, max(0, ny - 1))
                self.spinBox_6.setRange(0, max(0, ny - 1))

                # Default to full extents if not yet initialized
                if self.spinBox_4.value() == 0:
                    self.spinBox_2.setValue(0)
                    self.spinBox_4.setValue(max(0, nx - 1))
                if self.spinBox_6.value() == 0:
                    self.spinBox.setValue(0)
                    self.spinBox_6.setValue(max(0, ny - 1))

                # Make sure the secondary spin boxes are visible and enabled
                self.spinBox_4.setEnabled(True)
                self.spinBox_6.setEnabled(True)
                self.spinBox_4.show()
                self.spinBox_6.show()

                # Update mask when any of the 2D range spin boxes changes.
                try:
                    self.spinBox_2.editingFinished.connect(lambda: self._update_2d_mask_from_spinboxes())
                    self.spinBox_4.editingFinished.connect(lambda: self._update_2d_mask_from_spinboxes())
                    self.spinBox.editingFinished.connect(lambda: self._update_2d_mask_from_spinboxes())
                    self.spinBox_6.editingFinished.connect(lambda: self._update_2d_mask_from_spinboxes())
                except Exception:
                    pass
            else:
                # 1D datasets: keep only the original two spin boxes active
                # for the fit range; the extra pair is disabled to avoid
                # suggesting a 2D selection.
                self.spinBox_4.setEnabled(False)
                self.spinBox_6.setEnabled(False)
                self.spinBox_4.hide()
                self.spinBox_6.hide()
        except Exception:
            pass

    def _update_2d_mask_from_spinboxes(self) -> None:
        if not getattr(self, '_is_2d_dataset', False):
            return

        try:
            data = self.fit.data
        except Exception:
            return

        x_min = int(self.xmin)
        x_max = int(self.xmin2)
        y_min = int(self.xmax)
        y_max = int(self.xmax2)

        if self._2d_shape is None:
            return
        ny, nx = int(self._2d_shape[0]), int(self._2d_shape[1])
        if ny <= 0 or nx <= 0:
            return

        x0 = max(0, min(x_min, x_max))
        x1 = min(nx - 1, max(x_min, x_max))
        y0 = max(0, min(y_min, y_max))
        y1 = min(ny - 1, max(y_min, y_max))

        if x1 < x0 or y1 < y0:
            fc = get_fitting_client()
            if fc is not None:
                fc.set_fit_mask(
                    mask=[],
                    fit_uid=str(getattr(self.fit, "unique_identifier", "") or ""),
                )
            return

        grid_meta = getattr(self, '_grid_meta', {}) or {}

        mask_data = None
        if 'row_indices' in grid_meta and 'col_indices' in grid_meta:
            try:
                row_indices = np.asarray(grid_meta.get('row_indices'), dtype=np.int64)
                col_indices = np.asarray(grid_meta.get('col_indices'), dtype=np.int64)
            except Exception:
                return
            if row_indices.size == 0 or col_indices.size == 0:
                return
            n = min(row_indices.size, col_indices.size)
            mask = (
                (col_indices[:n] >= x0) & (col_indices[:n] <= x1) &
                (row_indices[:n] >= y0) & (row_indices[:n] <= y1)
            )
            mask_data = mask.astype(float)
        else:
            try:
                ny_img, nx_img = int(ny), int(nx)
            except Exception:
                ny_img, nx_img = ny, nx
            yy, xx = np.indices((ny_img, nx_img))
            mask_2d = (
                (xx >= x0) & (xx <= x1) &
                (yy >= y0) & (yy <= y1)
            )
            mask_data = mask_2d.ravel().astype(float)

        if mask_data is not None:
            fc = get_fitting_client()
            if fc is not None:
                fc.set_fit_mask(
                    mask=mask_data.tolist(),
                    fit_uid=str(getattr(self.fit, "unique_identifier", "") or ""),
                )
