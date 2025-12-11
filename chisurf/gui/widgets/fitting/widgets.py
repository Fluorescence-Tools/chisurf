from __future__ import annotations

import os
import typing
import pathlib

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

parameter_settings = chisurf.settings.parameter


@chisurf.decorators.register
class ModelDataRepresentationSelector(QtWidgets.QTreeWidget):

    @property
    def selected_fit_index(self) -> int:
        if self.currentIndex().parent().isValid():
            return self.currentIndex().parent().row()
        else:
            return self.currentIndex().row()

    @selected_fit_index.setter
    def selected_fit_index(self, v: int):
        self.setCurrentItem(self.topLevelItem(v))

    @property
    def selected_fit(self) -> chisurf.fitting.fit.FitGroup:
        return chisurf.fits[self.selected_fit_index]

    @property
    def selected_fits(self) -> typing.List[chisurf.fitting.fit.FitGroup]:
        return [chisurf.fits[i] for i in self.selected_fit_idx]

    @property
    def selected_fit_idx(self) -> typing.List[int]:
        return [r.row() for r in self.selectedIndexes()]

    def selectedIndexes(self) -> typing.List[QtCore.QModelIndex]:
        idx = super().selectedIndexes()[::3]
        return idx

    def keyPressEvent(self, event):
        key = event.key()
        if key in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
            self.onRemoveFit()

    def onCurveChanged(self):
        for fit_window in chisurf.cs.mdiarea.subWindowList():
            if fit_window.fit == self.selected_fit:
                chisurf.cs.mdiarea.setActiveSubWindow(fit_window)
                break
        self.change_event()

    def onChangeCurveName(self):
        # select current curve and change its name
        pass

    def onRemoveFit(self):
        fit_idxs = [selected_index.row() for selected_index in self.selectedIndexes()]
        for fit_idx in fit_idxs:
            chisurf.run(f'chisurf.macros.close_fit({fit_idx})')
        self.update(update_others=True)

    def onSaveFit(self, event: QtCore.QEvent = None, **kwargs):
        for fit_window in chisurf.cs.mdiarea.subWindowList():
            chisurf.cs.mdiarea.setActiveSubWindow(fit_window)
            chisurf.cs.onSaveFit()

    def contextMenuEvent(self, event):
        if self.context_menu_enabled:
            menu = QtWidgets.QMenu(self)
            menu.setTitle("Fits")
            menu.addAction("Save").triggered.connect(self.onSaveFit)
            menu.addAction("Close").triggered.connect(self.onRemoveFit)
            menu.addAction("Update").triggered.connect(self.update)
            menu.exec_(event.globalPos())

    def update(self, *args, update_others=True, **kwargs):
        # Optimize: avoid triggering expensive fit.update() on every list rebuild
        # and minimize signal/paint churn during population.
        try:
            self.blockSignals(True)
            self.setUpdatesEnabled(False)
            super().update()
            self.clear()

            for nbr, fit in enumerate(chisurf.fits):
                # Only use lightweight data to populate the list; do not call fit.update() here.
                try:
                    widget_name = pathlib.Path(fit.data.name).name
                except Exception:
                    widget_name = getattr(fit.data, 'name', 'Unknown')
                try:
                    model_name = fit.model.__class__.name
                except Exception:
                    model_name = getattr(fit.model.__class__, '__name__', 'Model')
                item = QtWidgets.QTreeWidgetItem(self, [str(nbr), widget_name, model_name])
                item.setToolTip(1, getattr(fit, 'name', widget_name))
                item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        finally:
            self.setUpdatesEnabled(True)
            self.blockSignals(False)

    def onItemChanged(self):
        if self.selected_fits:
            ds = self.selected_fits[0]

            # Find the index of the selected dataset
            index_of_ds = chisurf.fits.index(ds)

            # Remove "c" from its current position
            chisurf.fits.pop(index_of_ds)

            # Insert "c" at position 1
            idx_new = int(self.currentItem().text(0))
            chisurf.fits.insert(idx_new, ds)

            self.update(update_others=True)

    def change_event(self):
        pass

    def show(self):
        self.update()
        QtWidgets.QTreeWidget.show(self)

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit = None,
            experiment=None,
            drag_enabled: bool = False,
            click_close: bool = False,
            change_event: typing.Callable = None,
            curve_types: str = 'experiment',
            get_data_sets: typing.Callable = None,
            parent: QtWidgets.QWidget = None,
            icon: QtGui.QIcon = None,
            context_menu_enabled: bool = True
    ):
        if get_data_sets is None:
            def get_data_sets(**kwargs):
                return chisurf.data.get_data(
                    data_set=chisurf.imported_datasets,
                    **kwargs
                )
            self.get_data_sets = get_data_sets
        else:
            self.get_data_sets = get_data_sets

        if change_event is not None:
            self.change_event = change_event

        if icon is None:
            icon = QtGui.QIcon(":/icons/icons/list-add.png")

        self.curve_type = curve_types
        self.click_close = click_close
        self.fit = fit
        self.experiment = experiment
        self.context_menu_enabled = context_menu_enabled

        super().__init__(parent)
        self.setWindowIcon(icon)
        self.setWordWrap(True)
        self.setAlternatingRowColors(True)

        if drag_enabled:
            self.setAcceptDrops(True)
            self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        # http://python.6.x6.nabble.com/Drag-and-drop-editing-in-QListWidget-or-QListView-td1792540.html
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.drag_item = None
        self.drag_row = None

        self.clicked.connect(self.onCurveChanged)
        self.itemChanged.connect(self.onItemChanged)

        self.setHeaderHidden(False)
        self.setColumnCount(3)
        self.setHeaderLabels(('#', 'Data name', 'Model type'))
        header = self.header()

        # Set resize mode for the first and third columns
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

        header.setSectionsClickable(True)


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


class FitSubWindow(QtWidgets.QMdiSubWindow):

    def update(self, *args):
        super().update(self, *args)
        self.plot_tab_widget.update(*args)

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            control_layout: QtWidgets.QLayout,
            fit_widget: chisurf.gui.widgets.fitting.widgets.FittingControllerWidget = None,
            *args,
            **kwargs
    ):
        super().__init__(*args,  **kwargs)

        self.fit = fit
        self.fit_widget = fit_widget
        w = QtWidgets.QWidget(None)
        self.setWidget(w)

        # Set the focus policy of the subwindow
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        w.setFocusPolicy(QtCore.Qt.ClickFocus)

        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.plot_tab_widget = QtWidgets.QTabWidget(self)
        layout.addWidget(self.plot_tab_widget)
        rect = self.plot_tab_widget.geometry()
        self.setGeometry(rect)

        self.current_plot_controller = QtWidgets.QWidget(self)
        self.current_plot_controller.hide()

        # Lazy plot instantiation: create lightweight tab containers now, build plots on demand
        self._control_layout = control_layout
        self._plot_specs = list(fit.model.plot_classes)
        self._plot_containers = []
        self._plots_all = [None] * len(self._plot_specs)      # positional storage
        self._created_plots = []                               # actual created plots (shared)
        # Create empty containers per tab
        for (plot_class, kwargs) in self._plot_specs:
            container = QtWidgets.QWidget()
            container.setLayout(QtWidgets.QVBoxLayout())
            container.layout().setContentsMargins(0, 0, 0, 0)
            container.layout().setSpacing(0)
            self._plot_containers.append(container)
            tab_name = getattr(plot_class, 'name', None)
            if not isinstance(tab_name, str):
                tab_name = getattr(plot_class, '__name__', str(plot_class))
            self.plot_tab_widget.addTab(container, tab_name)
        # Share created plot list with FitGroup and its member Fits
        fit.plots = self._created_plots
        for f in fit:
            f.plots = self._created_plots

        # Instantiate the initially visible plot after the event loop returns
        # to avoid re-entrancy issues during fit creation; this may introduce
        # a tiny visual delay but is safer.
        def _ensure_initial_plot():
            idx = self.plot_tab_widget.currentIndex()
            self.ensure_plot_created(idx)
            self.on_change_plot()
        QtCore.QTimer.singleShot(0, _ensure_initial_plot)

        self.plot_tab_widget.currentChanged.connect(self.on_change_plot)

        # Use RubberBandResize / RubberBandMove
        self.setOption(
            chisurf.gui.QtWidgets.QMdiSubWindow.RubberBandResize,
            chisurf.settings.gui['RubberBandResize']
        )
        self.setOption(
            chisurf.gui.QtWidgets.QMdiSubWindow.RubberBandMove,
            chisurf.settings.gui['RubberBandMove']
        )

        # Set windows icon
        try:
            icon = fit.model.icon
        except AttributeError:
            icon = chisurf.gui.QtGui.QIcon(":/icons/icons/list-add.png")
        self.setWindowIcon(icon)

        # Set global style sheet
        # window_style = chisurf.settings.gui['fit_window_style']
        # self.setStyleSheet(chisurf.settings.style_sheet)

        self.setAttribute(chisurf.gui.QtCore.Qt.WA_DeleteOnClose, True)

        # Resize window
        xs, ys = chisurf.settings.gui['fit_windows_size']
        self.resize(xs, ys)

    def ensure_plot_created(self, idx: int):
        # Create plot for given index if not yet created
        if idx < 0 or idx >= len(self._plot_specs):
            return None
        if self._plots_all[idx] is not None:
            return self._plots_all[idx]
        plot_class, kwargs = self._plot_specs[idx]
        try:
            plot = plot_class(self.fit, **kwargs)
        except Exception as e:
            # Provide a fallback widget to avoid breaking the tab UI
            fallback = QtWidgets.QLabel(f"Failed to create plot: {getattr(plot_class, 'name', plot_class.__name__)}\n{e}")
            self._plot_containers[idx].layout().addWidget(fallback)
            self._plots_all[idx] = fallback
            return fallback
        # Attach to container and control layout
        plot.plot_controller.hide()
        self._plot_containers[idx].layout().addWidget(plot)
        self._control_layout.addWidget(plot.plot_controller)
        # Track in storage lists
        self._plots_all[idx] = plot
        self._created_plots.append(plot)
        
        # Connect LinePlot region changes to the Fit widget's range selector
        try:
            region_changed = getattr(plot, 'regionChanged', None)
            if region_changed is not None and hasattr(region_changed, 'connect') and self.fit_widget is not None:
                def _sync_fit_widget_range(xmin: int, xmax: int, fw=self.fit_widget):
                    # Update only the UI of the fit widget to reflect the plot's region
                    # The underlying fit_range is already updated inside the plot via chisurf.run
                    try:
                        fw.blockSignals(True)
                        fw.xmin = xmin
                        fw.xmax = xmax
                    finally:
                        fw.blockSignals(False)
                region_changed.connect(_sync_fit_widget_range)
        except Exception:
            pass
        
        return plot

    def on_change_plot(self):
        idx = self.plot_tab_widget.currentIndex()
        # Ensure the selected tab's plot exists
        plot = self.ensure_plot_created(idx)
        # Toggle controllers
        try:
            self.current_plot_controller.hide()
        except Exception:
            pass
        if plot is None or not hasattr(plot, 'plot_controller'):
            return
        self.current_plot_controller = plot.plot_controller
        self.current_plot_controller.show()
        # Ensure the newly visible plot refreshes its content; we defer the
        # heavy update to the next event-loop turn to avoid deep re-entrancy
        # during fit creation.
        try:
            update_all = getattr(plot, 'update_all', None)
            if callable(update_all):
                QtCore.QTimer.singleShot(0, update_all)
            elif hasattr(plot, 'update'):
                QtCore.QTimer.singleShot(0, plot.update)
        except Exception:
            try:
                plot.update()
            except Exception:
                pass

    def updateStatusBar(self, msg: str):
        self.statusBar().showMessage(msg)

    def closeEvent(self, event: QtCore.QEvent):
        # Honour a per-window opt-out flag (used by macros/app shutdown) as
        # well as the global confirm_close_fit setting.
        if getattr(self, 'close_confirm', True) and chisurf.settings.gui['confirm_close_fit']:
            reply = chisurf.gui.widgets.MyMessageBox.question(
                self,
                'Message',
                "Are you sure to close this fit?:\n%s" % self.fit.name,
                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                chisurf.console.execute('chisurf.macros.close_fit()')
                chisurf.gui.widgets.hide_items_in_layout(chisurf.cs.modelLayout)
                header_layout = getattr(chisurf.cs, "analysisHeaderLayout", None)
                if header_layout is not None:
                    chisurf.gui.widgets.hide_items_in_layout(header_layout)
                chisurf.gui.widgets.hide_items_in_layout(chisurf.cs.plotOptionsLayout)
            else:
                event.ignore()
        else:
            event.accept()


class FittingParameterDetailPopup(QtWidgets.QDialog):

    def __init__(self, controller: 'FittingParameterWidget'):
        super().__init__(controller)
        # Use Popup flag so clicks outside cause deactivation; we then hide on focus loss
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Popup)
        self.controller = controller
        self.setObjectName('FittingParameterDetailPopup')
        # Ensure we hide if the window deactivates (extra safety beyond Qt.Popup)
        self.installEventFilter(self)
        # Ensure the popup can take focus and is activated when shown
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, False)
        # Counter to temporarily suspend auto-hide on focus loss (e.g., while link menu is open)
        self._suspend_auto_hide = 0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        self.lbl_title = QtWidgets.QLabel(f"{controller.fitting_parameter.name}")
        font = self.lbl_title.font()
        font.setBold(True)
        self.lbl_title.setFont(font)
        layout.addWidget(self.lbl_title)

        # Optional human-readable description of the parameter, taken from
        # the underlying Parameter/FittingParameter "description" attribute.
        self.lbl_description = QtWidgets.QLabel("")
        self.lbl_description.setWordWrap(True)
        self.lbl_description.setStyleSheet("color: gray; font-size: 9pt")
        layout.addWidget(self.lbl_description)

        # Link info and actions
        link_row = QtWidgets.QHBoxLayout()
        self.lbl_link = QtWidgets.QLabel("")
        self.btn_change_link = QtWidgets.QToolButton()
        self.btn_change_link.setText("Link…")
        self.btn_unlink = QtWidgets.QToolButton()
        self.btn_unlink.setText("Unlink")
        link_row.addWidget(self.lbl_link, 1)
        link_row.addWidget(self.btn_change_link)
        link_row.addWidget(self.btn_unlink)
        layout.addLayout(link_row)

        # Value editor
        val_row = QtWidgets.QHBoxLayout()
        val_row.addWidget(QtWidgets.QLabel("Value:"))
        self.sb_value = pg.SpinBox(dec=True, decimals=self.controller.widget_value.opts.get('decimals', 6), finite=False)
        val_row.addWidget(self.sb_value)
        layout.addLayout(val_row)

        # Fixed checkbox
        self.cb_fixed = QtWidgets.QCheckBox("Fixed")
        layout.addWidget(self.cb_fixed)

        # Bounds group
        bounds_group = QtWidgets.QGroupBox("Bounds")
        b_layout = QtWidgets.QGridLayout(bounds_group)
        self.cb_bounds_on = QtWidgets.QCheckBox("Enable bounds")
        b_layout.addWidget(self.cb_bounds_on, 0, 0, 1, 2)
        b_layout.addWidget(QtWidgets.QLabel("Lower:"), 1, 0)
        self.sb_lb = pg.SpinBox(dec=True, decimals=self.controller.widget_lower_bound.opts.get('decimals', 6))
        b_layout.addWidget(self.sb_lb, 1, 1)
        b_layout.addWidget(QtWidgets.QLabel("Upper:"), 2, 0)
        self.sb_ub = pg.SpinBox(dec=True, decimals=self.controller.widget_upper_bound.opts.get('decimals', 6))
        b_layout.addWidget(self.sb_ub, 2, 1)
        layout.addWidget(bounds_group)


        # Close hint
        hint = QtWidgets.QLabel("Click outside to close")
        hint.setStyleSheet("color: gray; font-size: 9pt")
        layout.addWidget(hint)

        # Connections
        self.btn_change_link.clicked.connect(self._on_change_link)
        self.btn_unlink.clicked.connect(self._on_unlink)
        self.cb_fixed.toggled.connect(self._on_fixed_toggled)
        self.cb_bounds_on.toggled.connect(self._on_bounds_on_toggled)
        self.sb_lb.editingFinished.connect(self._on_bounds_changed)
        self.sb_ub.editingFinished.connect(self._on_bounds_changed)
        self.sb_value.editingFinished.connect(self._on_value_changed)

        self.refresh_from_model()

    def _begin_suspend_auto_hide(self):
        try:
            self._suspend_auto_hide += 1
        except Exception:
            self._suspend_auto_hide = 1

    def _end_suspend_auto_hide(self):
        try:
            self._suspend_auto_hide -= 1
            if self._suspend_auto_hide < 0:
                self._suspend_auto_hide = 0
        except Exception:
            self._suspend_auto_hide = 0

    def eventFilter(self, obj, event):
        # Hide the popup when it loses focus or the window deactivates, unless suspended
        if event is not None:
            et = int(event.type())
            if et == int(QtCore.QEvent.FocusOut) or et == int(QtCore.QEvent.WindowDeactivate):
                if getattr(self, '_suspend_auto_hide', 0) > 0:
                    # Do not hide; let event pass through
                    return False
                # Use hide (not close) as requested
                self.hide()
                return True
        return super().eventFilter(obj, event)

    def focusOutEvent(self, event: QtGui.QFocusEvent):
        # Extra safety: hide on focus out unless suspended
        try:
            if getattr(self, '_suspend_auto_hide', 0) == 0:
                self.hide()
        finally:
            event.accept()

    def _on_change_link(self):
        menu = self.controller.build_link_menu()
        # Show menu under the button
        pos = self.btn_change_link.mapToGlobal(QtCore.QPoint(0, self.btn_change_link.height()))
        # While the menu is open and linking occurs, do not auto-hide the popup
        self._begin_suspend_auto_hide()
        try:
            menu.exec_(pos)
            # After possible changes, refresh UI/model
            self.controller.finalize()
            self.refresh_from_model()
        finally:
            self._end_suspend_auto_hide()
            # Keep the popup open and focused after linking
            try:
                self.raise_()
                self.activateWindow()
                self.setFocus(QtCore.Qt.PopupFocusReason)
            except Exception:
                pass

    def _on_unlink(self):
        fp = self.controller.fitting_parameter
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].link = None\n"
            f"chisurf.fits[{fp.fit_idx}].update()"
        )
        self.controller.finalize()
        self.refresh_from_model()

    def _on_fixed_toggled(self):
        fp = self.controller.fitting_parameter
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].fixed = {self.cb_fixed.isChecked()}\n"
            f"chisurf.fits[{fp.fit_idx}].update()"
        )
        self.controller.finalize()

    def _on_bounds_on_toggled(self):
        fp = self.controller.fitting_parameter
        checked = self.cb_bounds_on.isChecked()
        # Toggle bounds_on in the model
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds_on = {checked}"
        )
        # Enable/disable editors immediately for better UX
        self.sb_lb.setEnabled(checked)
        self.sb_ub.setEnabled(checked)
        # If turning ON and current bounds are invalid/missing, initialize them from the UI spin boxes
        if checked:
            bounds_valid = False
            try:
                b = getattr(fp, 'bounds', None)
                bounds_valid = isinstance(b, (tuple, list)) and len(b) == 2
            except Exception:
                bounds_valid = False
            if not bounds_valid:
                chisurf.run(
                    f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds = ({self.sb_lb.value()}, {self.sb_ub.value()})"
                )
        # Refresh UI/model without risking unpack errors
        self.controller.finalize()
        self.refresh_from_model()

    def _on_bounds_changed(self):
        fp = self.controller.fitting_parameter
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds = ({self.sb_lb.value()}, {self.sb_ub.value()})"
        )
        self.controller.finalize()
        self.refresh_from_model()

    def _on_value_changed(self):
        fp = self.controller.fitting_parameter
        chisurf.run(
            f"parameter = chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}']\n"
            f"fixed = parameter.fixed \n"
            f"parameter.fixed = False\n"
            f"parameter.value = {self.sb_value.value()} \n"
            f"parameter.fixed = fixed\n"
            f"chisurf.fits[{fp.fit_idx}].finalize()"
        )
        self.controller.finalize()

    def refresh_from_model(self):
        fp = self.controller.fitting_parameter
        # Description text (may be empty)
        try:
            desc = getattr(fp, 'description', "")
        except Exception:
            desc = ""
        self.lbl_description.setVisible(bool(desc))
        if desc:
            self.lbl_description.setText(str(desc))
        # Update link label
        if getattr(fp, 'link', None) is not None:
            self.lbl_link.setText(f"Linked to: {fp.link.name}")
            self.btn_unlink.setEnabled(True)
        else:
            self.lbl_link.setText("Not linked")
            self.btn_unlink.setEnabled(False)
        # Value
        try:
            v = float(fp.value)
        except Exception:
            v = self.controller.widget_value.value()
        self.sb_value.setValue(v)
        # Fixed
        self.cb_fixed.blockSignals(True)
        self.cb_fixed.setChecked(bool(fp.fixed))
        self.cb_fixed.blockSignals(False)
        # Bounds
        self.cb_bounds_on.blockSignals(True)
        self.sb_lb.blockSignals(True)
        self.sb_ub.blockSignals(True)
        self.cb_bounds_on.setChecked(bool(fp.bounds_on))
        # Enable/disable editors based on bounds_on
        self.sb_lb.setEnabled(bool(fp.bounds_on))
        self.sb_ub.setEnabled(bool(fp.bounds_on))
        try:
            b = getattr(fp, 'bounds', None)
            if isinstance(b, (tuple, list)) and len(b) == 2:
                lb, ub = b
                self.sb_lb.setValue(float(lb))
                self.sb_ub.setValue(float(ub))
        except Exception:
            pass
        self.cb_bounds_on.blockSignals(False)
        self.sb_lb.blockSignals(False)
        self.sb_ub.blockSignals(False)




class FittingParameterWidget(Controller):

    def make_linkcall(self, fit_idx: int, parameter_name: str):
        def linkcall():
            try:
                self.blockSignals(True)

                # Fetch current and target parameters
                param_self = chisurf.fits[self.fitting_parameter.fit_idx].model.parameters_all_dict[self.fitting_parameter.name]
                param_other = chisurf.fits[fit_idx].model.parameters_all_dict[parameter_name]

                # Check for recursion using the Parameter class method
                if param_self.check_recursive_link(param_other, param_self):
                    QtWidgets.QMessageBox.warning(
                        self,  # Parent widget
                        "Linking Error",
                        "Recursion detected: Cannot link a parameter to itself or create a cyclic dependency.",
                        QtWidgets.QMessageBox.Ok
                    )
                else:
                    tooltip = " linked to " + parameter_name
                    s = (
                        f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['{self.fitting_parameter.name}'].link = "
                        f"chisurf.fits[{fit_idx}].model.parameters_all_dict['{parameter_name}'] \n"
                        f"chisurf.fits[{self.fitting_parameter.fit_idx}].update()"
                    )
                    # Execute the link assignment in the global chisurf
                    # context; the Parameter.link setter will update the
                    # follower's controller state via set_linked(True).
                    chisurf.run(s)

                    # Refresh this widget from the underlying parameter so it
                    # reflects the follower/linked role. The target parameter
                    # (master) remains visually unchanged (no check mark), so
                    # the user can always use this row's checkbox to unlink.
                    self.widget_link.setToolTip(tooltip)
                    try:
                        self.finalize()
                    except Exception:
                        pass

            finally:
                self.blockSignals(False)

        return linkcall

    def build_link_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        menu.setTitle(
            "Link " + self.fitting_parameter.name + " to:"
        )

        for fit_idx, f in enumerate(chisurf.fits):
            for fs in f:
                submenu = QtWidgets.QMenu(menu)
                submenu.setTitle(fs.name)

                # Sorted by "Aggregation"
                for a in fs.model.aggregated_parameters:
                    action_submenu = QtWidgets.QMenu(submenu)
                    action_submenu.setTitle(a.name)
                    ut = a.parameters_all
                    ut.sort(key=lambda x: x.name, reverse=False)
                    for p in ut:
                        if p is not self.fitting_parameter:
                            Action = action_submenu.addAction(p.name)
                            Action.triggered.connect(
                                self.make_linkcall(fit_idx, p.name)
                            )
                    submenu.addMenu(action_submenu)
                action_submenu = QtWidgets.QMenu(submenu)

                # Simply all parameters
                action_submenu.setTitle("All parameters")
                keys = list(fs.model.parameters_all_dict.keys())
                sorted_keys = sorted(keys)
                for key in sorted_keys:
                    p = fs.model.parameters_all_dict[key]
                    if p is not self:
                        Action = action_submenu.addAction(p.name)
                        Action.triggered.connect(self.make_linkcall(fit_idx, p.name))
                submenu.addMenu(action_submenu)

                menu.addMenu(submenu)
        return menu

    def contextMenuEvent(self, event: QtGui.QCloseEvent):

        menu = self.build_link_menu()
        menu.exec_(event.globalPos())

    def __str__(self):
        return ""

    @chisurf.gui.decorators.init_with_ui("variable_widget.ui")
    def __init__(
            self,
            fitting_parameter: chisurf.fitting.parameter.FittingParameter,
            layout: QtWidgets.QLayout = None,
            decimals: int = None,
            hide_label: bool = None,
            hide_error: bool = None,
            fixable: bool = None,
            hide_bounds: bool = None,
            name: str = None,
            label_text: str = None,
            hide_link: bool = None,
            suffix: str = "",
            callback: typing.Callable = None
    ):
        if hide_link is None:
            hide_link = parameter_settings['hide_link']
        if hide_bounds is None:
            hide_bounds = parameter_settings['hide_bounds']
        if name is None:
            name = self.__class__.__name__
        if label_text is None:
            label_text = name
        if fixable is None:
            fixable = parameter_settings['fixable']
        hide_fix_checkbox = fixable
        if hide_error is None:
            hide_error = parameter_settings['hide_error']
        if hide_label is None:
            hide_label = parameter_settings['hide_label']
        if decimals is None:
            decimals = parameter_settings['decimals']

        self.callback = callback
        self.name = fitting_parameter.name
        self.fitting_parameter = fitting_parameter
        self._details_popup = None  # created lazily on first label click
        self._is_output_param = bool(getattr(fitting_parameter, "is_output", False))

        # Allow HTML/RichText labels (e.g. "cpm<sub>all</sub>") so that
        # parameter names can be decorated with subscripts/superscripts
        # while keeping the underlying parameter name unchanged.
        try:
            self.label.setTextFormat(QtCore.Qt.RichText)
        except Exception:
            pass

        self.widget_value = pg.SpinBox(
            dec=True,
            decimals=decimals,
            suffix=suffix,
            finite=False
        )
        self.widget_value.opts['compactHeight'] = False
        self.horizontalLayout.addWidget(self.widget_value)

        self.widget_lower_bound = pg.SpinBox(
            dec=True,
            decimals=decimals
        )
        self.horizontalLayout_2.addWidget(self.widget_lower_bound)

        self.widget_upper_bound = pg.SpinBox(
            dec=True,
            decimals=decimals
        )
        self.horizontalLayout_2.addWidget(self.widget_upper_bound)

        # Hide and disable widgets
        self.label.setVisible(not hide_label)
        self.lineEdit.setVisible(not hide_error)
        self.widget_bounds_on.setDisabled(hide_bounds)
        self.widget_fix.setVisible(fixable or not hide_fix_checkbox)
        self.widget.setHidden(hide_bounds)
        self.widget_link.setDisabled(hide_link)

        if self._is_output_param:
            # Output parameters are displayed as read-only result cells.
            # Keep the row layout identical (checkboxes stay visible) but
            # prevent any user interaction and remove spin buttons so the
            # value looks like a plain, non-editable field.
            try:
                # Try to hide spin buttons directly on the SpinBox.
                self.widget_value.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            except Exception:
                # Fallback for pyqtgraph.SpinBox implementations that expose
                # an inner "spin" widget.
                try:
                    spin = getattr(self.widget_value, "spin", None)
                    if spin is not None and hasattr(spin, "setButtonSymbols"):
                        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
                except Exception:
                    pass
            try:
                self.widget_value.setReadOnly(True)
            except Exception:
                pass
            try:
                # Avoid focus so wheel / keyboard cannot change the value.
                self.widget_value.setFocusPolicy(QtCore.Qt.NoFocus)
            except Exception:
                pass
            try:
                self.widget_fix.setEnabled(False)
            except Exception:
                pass
            try:
                self.widget_bounds_on.setEnabled(False)
            except Exception:
                pass
            try:
                self.widget_link.setEnabled(False)
            except Exception:
                pass
            # Hide the lower/upper bound spin boxes for outputs; only keep
            # the (disabled) bounds checkbox for alignment.
            self.widget.setHidden(True)

        # Make label interactive: clicking opens a details popup
        try:
            self.label.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self.label.setToolTip("Click to view and edit details")
            # install a mousePress handler
            self.label.mousePressEvent = self._on_label_mouse_press  # type: ignore
        except Exception:
            pass

        # Display of values
        try:
            _init_v = float(fitting_parameter.value)
        except Exception:
            _init_v = self.widget_value.value() if hasattr(self, 'widget_value') else 0.0
        self.widget_value.setValue(_init_v)
        # Do not left-pad HTML labels with spaces; this breaks rich text.
        # For plain-text labels we keep the original padding.
        try:
            if "<" in label_text or ">" in label_text:
                self.label.setText(label_text)
            else:
                self.label.setText(label_text.ljust(5))
        except Exception:
            self.label.setText(label_text)

        # variable bounds
        if not fitting_parameter.bounds_on:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Checked)

        # variable fixed
        if fitting_parameter.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        self.widget.hide()

        # The variable value
        self.widget_value.editingFinished.connect(self._on_main_value_changed)
        if callback:
            self.widget_value.editingFinished.connect(self.callback)

        self.widget_fix.toggled.connect(
            lambda: chisurf.run(
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['{fitting_parameter.name}'].fixed = "
                f"{self.widget_fix.isChecked()} \n"
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].update()")
        )

        # Variable is bounded
        self.widget_bounds_on.toggled.connect(self._on_main_bounds_on_toggled)

        self.widget_lower_bound.editingFinished.connect(
            lambda: chisurf.run(
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['%s'].bounds = (%s, %s)" %
                (
                    fitting_parameter.name,
                    self.widget_lower_bound.value(),
                    self.widget_upper_bound.value()
                )
            )
        )

        self.widget_upper_bound.editingFinished.connect(
            lambda: chisurf.run(
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['%s'].bounds = (%s, %s)" %
                (
                    fitting_parameter.name,
                    self.widget_lower_bound.value(),
                    self.widget_upper_bound.value()
                )
            )
        )

        self.widget_link.clicked.connect(self.onLinkFitGroup)

        if isinstance(layout, QtWidgets.QLayout):
            layout.addWidget(self)
        try:
            self._update_role_visuals()
        except Exception:
            pass


    def _on_label_mouse_press(self, event: QtGui.QMouseEvent):
        try:
            if event.button() == QtCore.Qt.LeftButton:
                self._open_details_popup()
            else:
                # fall back to default behavior
                super().mousePressEvent(event)
        except Exception:
            pass

    def _open_details_popup(self):
        if getattr(self, "_is_output_param", False):
            return
        # Lazy-create popup
        if self._details_popup is None or not isinstance(self._details_popup, FittingParameterDetailPopup):
            self._details_popup = FittingParameterDetailPopup(self)
        # Position popup under the label
        try:
            global_pos = self.label.mapToGlobal(self.label.rect().bottomLeft())
        except Exception:
            global_pos = QtGui.QCursor.pos()
        self._details_popup.move(global_pos)
        self._details_popup.refresh_from_model()
        self._details_popup.show()
        # Ensure the popup gains focus and is on top
        try:
            self._details_popup.raise_()
            self._details_popup.activateWindow()
            self._details_popup.setFocus(QtCore.Qt.PopupFocusReason)
            # Some platforms need delayed activation
            QtCore.QTimer.singleShot(0, self._details_popup.activateWindow)
        except Exception:
            pass

    def _on_label_mouse_press(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton:
            self._open_details_popup()
        else:
            # For other buttons, fall back to default behavior (e.g., open context menu on right click)
            super().mousePressEvent(event)

    def _update_role_visuals(self):
        try:
            if getattr(self, "_is_output_param", False):
                role = "output"
            elif getattr(self.fitting_parameter, "is_linked", False):
                role = "linked"
            else:
                role = "input"
            bg_color = None
            if role == "output":
                bg_color = parameter_settings.get("role_color_output")
            elif role == "linked":
                bg_color = parameter_settings.get("role_color_linked")
            else:
                bg_color = parameter_settings.get("role_color_input")
            parts = []
            if bg_color:
                parts.append(f"background-color: {bg_color};")
            if role == "linked":
                parts.append("text-decoration: underline;")
            style = " ".join(parts)
            self.widget_value.setStyleSheet(style)
        except Exception:
            pass

    def set_linked(self, is_linked: bool):
        if getattr(self, "_is_output_param", False):
            self.widget_link.setCheckState(QtCore.Qt.Unchecked)
            return
        # Interpret linking state in terms of three visual roles:
        #   - Unchecked: not linked at all.
        #   - PartiallyChecked: this parameter follows another one (slave).
        #   - Checked: this parameter is the master within a fit group.
        is_master = bool(getattr(self.fitting_parameter, "is_link_master", False))

        if is_linked:
            # Follower: value is controlled by the master; disable editing
            # and show a partially-checked box.
            self.widget_link.setCheckState(QtCore.Qt.PartiallyChecked)
            self.widget_value.setEnabled(False)
        else:
            if is_master:
                # Master within the fit group: keep value editable but mark
                # the checkbox as fully checked so the user sees it as the
                # source of the group link.
                self.widget_link.setCheckState(QtCore.Qt.Checked)
                self.widget_value.setEnabled(True)
            else:
                # Not linked at all.
                self.widget_link.setCheckState(QtCore.Qt.Unchecked)
                self.widget_value.setEnabled(True)

        try:
            self._update_role_visuals()
        except Exception:
            pass

    def onLinkFitGroup(self):
        # Clicking the link checkbox should have intuitive semantics:
        #
        # - If this parameter is currently a *follower* (linked to some
        #   master), a click unlinks **only this parameter**.
        # - Otherwise (unlinked or acting as fit-group master), we delegate
        #   to the group-level macro so the user can establish or remove a
        #   fit-group link.
        fp = self.fitting_parameter
        if getattr(self, "_is_output_param", False):
            return

        is_linked = bool(getattr(fp, "is_linked", False))
        is_master = bool(getattr(fp, "is_link_master", False))

        self.blockSignals(True)
        try:
            if is_linked and not is_master:
                # Per-parameter unlink: this row was following another
                # parameter via ``fp.link``. Clear the link so only this
                # parameter becomes free again.
                try:
                    fp.link = None
                except Exception:
                    try:
                        chisurf.logging.warning(
                            f"FittingParameterWidget: failed to unlink parameter '{getattr(fp, 'name', '?')}'."
                        )
                    except Exception:
                        pass
            else:
                # Group-level behaviour: interpret the current checkbox
                # state as a request to link/unlink the whole fit group for
                # this parameter name.
                state = int(self.widget_link.checkState())
                chisurf.run(
                    f"chisurf.macros.link_fit_group('{fp.name}', {state})"
                )

            try:
                self.finalize()
            except Exception:
                pass
        finally:
            self.blockSignals(False)

    def setValue(self, v):
        self.widget_value.setValue(v)

    def _on_main_value_changed(self):
        if getattr(self, "_is_output_param", False):
            return
        fp = self.fitting_parameter
        try:
            fit_idx = fp.fit_idx
        except Exception:
            fit_idx = -1
        # Guard against invalid indices so we never accidentally target chisurf.fits[-1]
        try:
            n_fits = len(chisurf.fits)
        except Exception:
            n_fits = 0
        if not isinstance(fit_idx, int) or fit_idx < 0 or fit_idx >= n_fits:
            try:
                chisurf.logging.warning(
                    f"FittingParameterWidget: invalid fit_idx {fit_idx} for parameter '{getattr(fp, 'name', '?')}', "
                    f"skipping value change."
                )
            except Exception:
                pass
            return

        value = self.widget_value.value()
        chisurf.run(
            f"parameter = chisurf.fits[{fit_idx}].model.parameters_all_dict['{fp.name}']\n"
            f"fixed = parameter.fixed \n"
            f"parameter.fixed = False\n"
            f"parameter.value = {value} \n"
            f"parameter.fixed = fixed\n"
            f"chisurf.fits[{fit_idx}].finalize()"
        )

    def _on_main_bounds_on_toggled(self):
        if getattr(self, "_is_output_param", False):
            return
        fp = self.fitting_parameter
        checked = self.widget_bounds_on.isChecked()
        # Toggle bounds_on in the model
        chisurf.run(
            f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds_on = {checked}"
        )
        # If turning ON and current bounds are invalid/missing, initialize them from the UI spin boxes
        if checked:
            bounds_valid = False
            try:
                b = getattr(fp, 'bounds', None)
                bounds_valid = isinstance(b, (tuple, list)) and len(b) == 2
            except Exception:
                bounds_valid = False
            if not bounds_valid:
                chisurf.run(
                    f"chisurf.fits[{fp.fit_idx}].model.parameters_all_dict['{fp.name}'].bounds = ({self.widget_lower_bound.value()}, {self.widget_upper_bound.value()})"
                )
        # Refresh UI/model without risking unpack errors
        self.finalize()

    def finalize(self, *args):
        # Ensure execution on the widget's thread (GUI thread). If called from another thread,
        # reschedule finalize to run on the correct thread and return immediately.
        if QtCore.QThread.currentThread() is not self.thread():
            try:
                # Queue the call to this object's thread (GUI thread)
                QtCore.QMetaObject.invokeMethod(self, "finalize", QtCore.Qt.QueuedConnection)
            except Exception:
                # Fallback: schedule via QApplication event loop
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    QtCore.QTimer.singleShot(0, lambda: self.finalize())
            return
        #super().update(*args)
        self.blockSignals(True)

        # Sync link UI state first
        try:
            self.set_linked(self.fitting_parameter.is_linked)
        except Exception:
            pass

        # Update value of widget (guard against None)
        try:
            _v = float(self.fitting_parameter.value)
        except Exception:
            _v = self.widget_value.value()
        self.widget_value.setValue(_v)
        self.widget_fix.setCheckState(QtCore.Qt.Checked if self.fitting_parameter.fixed else QtCore.Qt.Unchecked)

        # Sync bounds UI safely (no unpack unless valid)
        try:
            self.widget_bounds_on.blockSignals(True)
            self.widget_lower_bound.blockSignals(True)
            self.widget_upper_bound.blockSignals(True)
            bounds_on = bool(getattr(self.fitting_parameter, 'bounds_on', False))
            self.widget_bounds_on.setCheckState(QtCore.Qt.Checked if bounds_on else QtCore.Qt.Unchecked)

            # Default to current UI values; replace with model values only if valid
            lb_val = self.widget_lower_bound.value()
            ub_val = self.widget_upper_bound.value()
            b = getattr(self.fitting_parameter, 'bounds', None)
            if isinstance(b, (tuple, list)) and len(b) == 2:
                try:
                    lb_val = float(b[0])
                    ub_val = float(b[1])
                except Exception:
                    pass
            self.widget_lower_bound.setValue(lb_val)
            self.widget_upper_bound.setValue(ub_val)
        except Exception:
            pass
        finally:
            try:
                self.widget_bounds_on.blockSignals(False)
                self.widget_lower_bound.blockSignals(False)
                self.widget_upper_bound.blockSignals(False)
            except Exception:
                pass

        # Tooltip (guard against invalid bounds)
        if getattr(self.fitting_parameter, 'bounds_on', False):
            b = getattr(self.fitting_parameter, 'bounds', None)
            if isinstance(b, (tuple, list)) and len(b) == 2:
                lower, upper = b
                tooltip_text = f"bound: ({lower}, {upper})\n"
            else:
                tooltip_text = "bounds: on (unset)\n"
        else:
            tooltip_text = "bounds: off\n"

        link_param = getattr(self.fitting_parameter, 'link', None)
        if self.fitting_parameter.is_linked and link_param is not None:
            target_param_name = getattr(link_param, 'name', "?")
            target_fit_label = "?"
            try:
                target_fit_idx = getattr(link_param, 'fit_idx', -1)
            except Exception:
                target_fit_idx = -1
            try:
                if isinstance(target_fit_idx, int) and target_fit_idx >= 0:
                    fits = getattr(chisurf, 'fits', None)
                    if fits is not None and 0 <= target_fit_idx < len(fits):
                        target_fit = fits[target_fit_idx]
                        target_fit_label = getattr(target_fit, 'name', str(target_fit_idx))
                    else:
                        target_fit_label = str(target_fit_idx)
            except Exception:
                pass
            tooltip_text += f"linked to fit '{target_fit_label}', \n parameter '{target_param_name}'"
        self.widget_value.setToolTip(tooltip_text)

        # Error-estimate
        value = float(self.fitting_parameter.value)
        if not np.isfinite(value):
            rel_error = "NA"
        else:
            error_estimate = self.fitting_parameter.error_estimate
            rel_error = abs(error_estimate / (value + 1e-12) * 100.0)

        if self.fitting_parameter.fixed or not isinstance(error_estimate, float):
            self.lineEdit.setText("NA")
            # Reset background color to default
            self.lineEdit.setStyleSheet("")
        else:
            self.lineEdit.setText("NA" if np.isnan(rel_error) else f"{rel_error:.0f}%")

            # Set background color based on relative error
            if not np.isnan(rel_error):
                # Create a colormap from error_color_small to error_color_large
                # Use default values if settings are not found
                error_color_small = parameter_settings.get('error_color_small', 'green')
                error_color_large = parameter_settings.get('error_color_large', 'magenta')
                error_threshold_small = parameter_settings.get('error_threshold_small', 20)
                error_threshold_large = parameter_settings.get('error_threshold_large', 100)

                cmap = mcolors.LinearSegmentedColormap.from_list(
                    'error_color_gradient',
                    [(0, error_color_small), (1, error_color_large)]
                )

                # Normalize error value: error_threshold_small -> error_color_small, error_threshold_large -> error_color_large
                error_range = error_threshold_large - error_threshold_small
                norm_error = min(1.0, max(0.0, (rel_error - error_threshold_small) / error_range))

                # Get RGB color from colormap
                rgb_color = cmap(norm_error)

                # Convert RGB to hex for stylesheet
                hex_color = mcolors.rgb2hex(rgb_color)

                # Set background color and ensure text is readable
                # Use white text for darker backgrounds, black for lighter ones
                r, g, b = rgb_color[:3]
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if brightness < 0.5 else "black"

                # Set background color and text color
                self.lineEdit.setStyleSheet(f"background-color: {hex_color}; color: {text_color};")

        # Link
        if link_param is not None:
            target_param_name = getattr(link_param, 'name', "?")
            target_fit_label = "?"
            try:
                target_fit_idx = getattr(link_param, 'fit_idx', -1)
            except Exception:
                target_fit_idx = -1
            try:
                if isinstance(target_fit_idx, int) and target_fit_idx >= 0:
                    fits = getattr(chisurf, 'fits', None)
                    if fits is not None and 0 <= target_fit_idx < len(fits):
                        target_fit = fits[target_fit_idx]
                        target_fit_label = getattr(target_fit, 'name', str(target_fit_idx))
                    else:
                        target_fit_label = str(target_fit_idx)
            except Exception:
                pass
            tooltip = f"linked to fit '{target_fit_label}', parameter '{target_param_name}'"
            self.widget_link.setToolTip(tooltip)
            self.widget_value.setEnabled(False)

        # If the details popup is open, refresh its contents to reflect latest model state
        try:
            if getattr(self, '_details_popup', None) is not None and self._details_popup.isVisible():
                self._details_popup.refresh_from_model()
        except Exception:
            pass

        # If the details popup is open, refresh its contents to reflect latest model state
        try:
            if getattr(self, '_details_popup', None) is not None and self._details_popup.isVisible():
                self._details_popup.refresh_from_model()
        except Exception:
            pass

        try:
            self._update_role_visuals()
        except Exception:
            pass

        self.blockSignals(False)


class FittingParameterGroupWidget(QtWidgets.QGroupBox):

    def __init__(
            self,
            parameter_group: chisurf.fitting.parameter.FittingParameterGroup,
            n_col: int = None,
            layout: QtWidgets.QVBoxLayout = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        if n_col is None:
            n_col = chisurf.settings.gui['fit_models']['n_columns']

        self.parameter_group = parameter_group
        self.n_col = n_col
        self.n_row = 0

        self.setTitle(parameter_group.name)
        if layout is None:
            layout = QtWidgets.QGridLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)
        for i, p in enumerate(parameter_group.parameters_all):
            label_text = p.__dict__.get('label_text', p.name)
            pw = chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
                fitting_parameter=p,
                label_text=label_text
            )
            col = i % self.n_col
            row = i // self.n_col
            layout.addWidget(pw, row, col)


def make_fitting_parameter_widget(
        fitting_parameter: chisurf.fitting.parameter.FittingParameter,
        label_text: str = None,
        layout: QtWidgets.QLayout = None,
        decimals: int = None,
        hide_label: bool = None,
        hide_error: bool = None,
        fixable: bool = None,
        hide_bounds: bool = None,
        name: str = None,
        hide_link: bool = None,
        suffix: str = "",
        callback: typing.Callable = None
) -> FittingParameterWidget:
    if label_text is None:
        # Safely get label_text from parameter's __dict__ or use name as fallback
        label_text = fitting_parameter.__dict__.get('label_text', fitting_parameter.name)
    # If no explicit suffix was provided, infer simple unit suffixes from the
    # parameter name (e.g. *_nm -> " nm", *_um -> " µm"). This keeps
    # backwards compatibility while improving readability for standard unit
    # conventions used throughout ChiSurf.
    auto_suffix = suffix
    if not auto_suffix:
        n = str(fitting_parameter.name)
        if n.endswith("_nm"):
            auto_suffix = " nm"
        elif n.endswith("_um"):
            auto_suffix = " µm"
        elif n.endswith("_ms"):
            auto_suffix = " ms"
        elif n.endswith("_us"):
            auto_suffix = " µs"
        elif n.endswith("_ns"):
            auto_suffix = " ns"
        elif n.endswith("_K"):
            auto_suffix = " K"

    widget = FittingParameterWidget(
        fitting_parameter,
        hide_label=hide_label,
        layout=layout,
        decimals=decimals,
        hide_error=hide_error,
        fixable=fixable,
        hide_bounds=hide_bounds,
        name=name,
        hide_link=hide_link,
        label_text=label_text,
        suffix=auto_suffix,
        callback=callback
    )
    fitting_parameter.controller = widget
    return widget


def make_fitting_parameter_group_widget(
        fitting_parameter_group: chisurf.fitting.parameter.FittingParameterGroup,
        *args,
        **kwargs
):
    return FittingParameterGroupWidget(
        fitting_parameter_group,
        *args,
        **kwargs
    )
