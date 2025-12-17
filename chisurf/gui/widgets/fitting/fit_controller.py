from __future__ import annotations

import os
import typing
import pathlib
import textwrap

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, uic, QtCore, QtGui
import matplotlib.colors as mcolors

import chisurf.data
import chisurf.fitting
import chisurf.decorators
import chisurf.gui.decorators
import chisurf.settings

import chisurf.gui.widgets
import chisurf.gui.widgets.experiments.widgets
from chisurf.gui.widgets import Controller
from chisurf.math.optimization.leastsqbound import OptimizationCancelled


class FittingControllerWidget(Controller):

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
        self.fit.data = dataset
        self.fit.update()
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
            fit: chisurf.fitting.fit.FitGroup = None,
            hide_fit_button: bool = False,
            hide_range: bool = False,
            hide_fitting: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.fit = fit
        self.curve_select = chisurf.gui.widgets.experiments.widgets.ExperimentalDataSelector(
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

    def _result_changed(self):
        result_idx = self.spinBox_3.value() - 1
        chisurf.run(f"chisurf.fits[{self.fit.fit_idx}].set_result_idx({result_idx})")

    def onDatasetChanged(self):
        chisurf.run(f"chisurf.macros.change_selected_fit_of_group({self.selected_fit})")

    def onErrorEstimate(self):
        chisurf.logging.info(f"Sampling analysis: {self.fit.name}")
        filename = chisurf.gui.widgets.save_file('Error estimate', '*.er4')
        if filename is None:
            chisurf.logging.info("Sampling canceled!")
            return
        else:
            kw = chisurf.settings.cs_settings['optimization']['sampling']
            kw['n_runs'] = self.n_runs
            kw['steps'] = self.n_steps
            chisurf.fitting.fit.sample_fit(self.fit, filename, **kw)
            chisurf.logging.info("Sampling done!")

    def onRunFit(self):
        chisurf.logging.info(f"Please wait fitting: {self.fit.name}")

        dialog = None
        success = False
        try:
            # Create a modal progress dialog if the GUI helpers are available.
            try:
                dialog = chisurf.gui.widgets.progress.EnhancedProgressDialog(
                    title="Fitting",
                    label_text=f"Fitting {self.fit.name}...",
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
                try:
                    base_label = f"Fitting {self.fit.name}..."
                except Exception:
                    base_label = "Fitting..."

                parts = [base_label, f"eval {done}/{int(total) if total else '?'}"]
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

                label_text = "  |  ".join(parts)

                try:
                    dialog.update_progress(value, text=label_text)
                except Exception:
                    # Never let UI errors break the optimizer.
                    pass

            # Run the fit synchronously, allowing the optimizer to invoke
            # the progress callback from within the residual evaluations.
            try:
                self.fit.run(
                    local_first=self.local_first,
                    progress_callback=_on_progress,
                )
            except OptimizationCancelled:
                chisurf.logging.info("Fitting cancelled by user.")
                success = False
            else:
                # Finalize model and parameter controllers as before.
                self.fit.model.finalize()
                for pa in chisurf.fitting.parameter.FittingParameter.get_instances():
                    try:
                        pa.controller.finalize()
                    except (AttributeError, RuntimeError, TypeError):
                        chisurf.logging.warning(
                            f"Fitting parameter {pa.name} does not have a controller to update."
                        )
                chisurf.logging.info("Fitting finished!")
                success = True

                # Update fit result selector
                self.spinBox_3.setMaximum(len(self.fit.results))
                self.spinBox_3.setMinimum(1)
                self.spinBox_3.setValue(1)
        finally:
            if dialog is not None:
                try:
                    final_text = "Fitting finished!" if success else "Fitting aborted."
                    try:
                        delay_ms = int(chisurf.settings.gui.get('fit_progress_close_delay_ms', 500))
                    except Exception:
                        delay_ms = 500
                    dialog.finish(final_text=final_text, auto_close=True, close_delay_ms=delay_ms)
                except Exception:
                    try:
                        dialog.finalize(force_auto_close=True)
                    except Exception:
                        pass

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
        chisurf.logging.info(f'onFitRangeChanged: {xmin, xmax}')
        if xmin is not None:
            self.xmin = xmin
        if xmax is not None:
            self.xmax = xmax
        try:
            # Apply directly to this widget's fit to avoid depending on cs.current_fit
            self.fit.fit_range = (self.xmin, self.xmax)
        except Exception as e:
            chisurf.logging.warning(f'Failed to set fit range directly: {e}')
        # For intrinsically 2D datasets (PDA/RICS), update the Fit/FitGroup
        # mask from the four spin boxes interpreted as (x_min, x_max,
        # y_min, y_max) indices on the underlying 2D grid.
        if getattr(self, '_is_2d_dataset', False):
            try:
                self._update_2d_mask_from_spinboxes()
            except Exception as e:
                chisurf.logging.warning(f'Failed to update 2D mask from spinboxes: {e}')
        # Avoid deep re-entrant updates when auto-fit-range is already
        # driving a fit update.
        if getattr(self, '_auto_fit_range_in_progress', False):
            return
        self.fit.update()

    def onAutoFitRange(self):
        data = getattr(self.fit, "data", None)
        reader = getattr(data, "data_reader", None)
        if reader is None or not hasattr(reader, "autofitrange"):
            return
        try:
            fit_range = reader.autofitrange(data)
            chisurf.logging.info(f'onAutoFitRange: {fit_range}')
            xmin_1d, xmax_1d = fit_range

            # Guard against re-entrant updates when auto-fit-range is itself
            # driving a fit update.
            try:
                self._auto_fit_range_in_progress = True
            except Exception:
                pass

            try:
                if getattr(self, '_is_2d_dataset', False) and self._2d_shape is not None:
                    # For intrinsically 2D datasets (e.g. PDA, RICS) we treat
                    # autofitrange as a suggestion for the *flattened* 1D
                    # extent, but the UI spin boxes encode 2D index bounds.
                    # Here we default the 2D selection to the full grid and
                    # use the full 1D range for the fit; any further
                    # restriction is expressed via the 2D mask only.

                    ny, nx = int(self._2d_shape[0]), int(self._2d_shape[1])

                    # Ensure spin box ranges match the grid shape
                    self.spinBox_2.setRange(0, max(0, nx - 1))
                    self.spinBox_4.setRange(0, max(0, nx - 1))
                    self.spinBox.setRange(0, max(0, ny - 1))
                    self.spinBox_6.setRange(0, max(0, ny - 1))

                    # Full 2D extents in index space
                    self.spinBox_2.setValue(0)
                    self.spinBox_4.setValue(max(0, nx - 1))
                    self.spinBox.setValue(0)
                    self.spinBox_6.setValue(max(0, ny - 1))

                    # Full 1D range over the flattened data vector
                    try:
                        n_flat = int(len(self.fit.data.y))
                    except Exception:
                        n_flat = max(0, int(xmax_1d))
                    try:
                        self.fit.fit_range = (0, n_flat)
                    except Exception as e:
                        chisurf.logging.warning(f'Failed to set 1D fit range during 2D auto-fit: {e}')

                    # Refresh the 2D mask from the full-extent spin boxes
                    try:
                        self._update_2d_mask_from_spinboxes()
                    except Exception as e:
                        chisurf.logging.warning(f'Failed to update 2D mask after 2D autofitrange: {e}')
                else:
                    # 1D datasets: keep the original semantics where the two
                    # spin boxes encode [xmin, xmax) directly.
                    self.xmin, self.xmax = (xmin_1d, xmax_1d)
                    try:
                        self.fit.fit_range = (xmin_1d, xmax_1d)
                    except Exception as e:
                        chisurf.logging.warning(f'Failed to set fit range during auto-fit: {e}')

                # Trigger a single fit update for the new range / mask
                self.fit.update()
                # Allow models to react to the completed auto-fit range via
                # an optional hook. This keeps the controller generic while
                # enabling model-specific post-processing (e.g. MaxEnt L-curves).
                try:
                    grouped = getattr(self.fit, "grouped_fits", None)
                    if isinstance(grouped, (list, tuple)):
                        models = [getattr(f, "model", None) for f in grouped]
                    else:
                        models = [getattr(self.fit, "model", None)]
                    for m in models:
                        hook = getattr(m, "on_auto_fit_range_completed", None)
                        if callable(hook):
                            try:
                                hook()
                            except Exception as e:
                                chisurf.logging.warning(
                                    f"FittingControllerWidget.onAutoFitRange: model hook on_auto_fit_range_completed failed: {e}"
                                )
                except Exception:
                    pass
            finally:
                try:
                    self._auto_fit_range_in_progress = False
                except Exception:
                    pass
        except Exception as e:
            chisurf.logging.warning(f"onAutoFitRange failed: {e}")

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
        """Build a 1D mask from 2D bounds for PDA/RICS datasets.

        Spin box mapping:
            spinBox_2 -> x_min
            spinBox_4 -> x_max
            spinBox   -> y_min
            spinBox_6 -> y_max

        The resulting 1D mask is stored on ``self.fit.mask`` so that the
        abstract Fit/FitGroup machinery can remain unaware of PDA/RICS
        specifics while still respecting the 2D selection.
        """

        if not getattr(self, '_is_2d_dataset', False):
            return

        try:
            data = self.fit.data
        except Exception:
            return

        # Read bounds and normalize order
        x_min = int(self.xmin)
        x_max = int(self.xmin2)
        y_min = int(self.xmax)
        y_max = int(self.xmax2)

        if self._2d_shape is None:
            return
        ny, nx = int(self._2d_shape[0]), int(self._2d_shape[1])
        if ny <= 0 or nx <= 0:
            return

        # Clamp to valid index ranges and ensure min <= max
        x0 = max(0, min(x_min, x_max))
        x1 = min(nx - 1, max(x_min, x_max))
        y0 = max(0, min(y_min, y_max))
        y1 = min(ny - 1, max(y_min, y_max))

        if x1 < x0 or y1 < y0:
            # Degenerate rectangle -> clear mask
            try:
                self.fit.mask = None
            except Exception:
                pass
            return

        # Use the generic grid metadata to map 2D bounds back to the 1D
        # flattened representation. By default we assume a dense 2D grid
        # flattened in NumPy 'C' (row-major) order. Experiments may
        # optionally provide explicit index arrays (row_indices/col_indices)
        # to describe sparse or non-rectangular supports.

        grid_meta = getattr(self, '_grid_meta', {}) or {}

        # Prefer explicit index arrays when available
        if 'row_indices' in grid_meta and 'col_indices' in grid_meta:
            # 1D flattening via explicit (row_indices, col_indices)
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
            try:
                self.fit.mask = mask.astype(float)
            except Exception:
                pass
            return

        # Default: full 2D grid, flattened in NumPy 'C' (row-major) order
        try:
            ny_img, nx_img = int(ny), int(nx)
        except Exception:
            ny_img, nx_img = ny, nx

        yy, xx = np.indices((ny_img, nx_img))
        mask_2d = (
            (xx >= x0) & (xx <= x1) &
            (yy >= y0) & (yy <= y1)
        )
        mask_1d = mask_2d.ravel()
        try:
            self.fit.mask = mask_1d.astype(float)
        except Exception:
            pass


