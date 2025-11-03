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

    def change_dataset(self) -> None:
        dataset = self.curve_select.selected_dataset
        self.fit.data = dataset
        self.fit.update()
        self.comboBox.setItemText(
            self.comboBox.currentIndex(),
            dataset.name
        )

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
        fit_names = [os.path.basename(f.data.name) for f in fit]
        self.comboBox.addItems(fit_names)

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

        if hide_fit_button:
            self.pushButton_fit.hide()
        if hide_range:
            self.toolButton_2.hide()
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
        chisurf.run(f"cs.current_fit.run(local_first={self.local_first})")
        self.fit.model.finalize()
        for pa in chisurf.fitting.parameter.FittingParameter.get_instances():
            try:
                pa.controller.finalize()
            except (AttributeError, RuntimeError, TypeError):
                chisurf.logging.warning(f"Fitting parameter {pa.name} does not have a controller to update.")
        chisurf.logging.info("Fitting finished!")
        # Update fit result selector
        self.spinBox_3.setMaximum(len(self.fit.results))
        self.spinBox_3.setMinimum(1)
        self.spinBox_3.setValue(1)

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
        self.fit.update()

    def onAutoFitRange(self):
        try:
            fit_range = self.fit.data.data_reader.autofitrange(self.fit.data)
            chisurf.logging.info(f'onAutoFitRange: {fit_range}')
            self.xmin, self.xmax = fit_range
            self.onFitRangeChanged(None, *fit_range)
        except AttributeError:
            s = (f"Fit {self.__class__.__name__} "
                 f"with model {self.fit.model.__class__.__name__} "
                 f"does not have an attribute data.data_reader")
            chisurf.logging.warning(s)


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
            self.plot_tab_widget.addTab(container, getattr(plot_class, 'name', plot_class.__name__))
        # Share created plot list with FitGroup and its member Fits
        fit.plots = self._created_plots
        for f in fit:
            f.plots = self._created_plots

        # Instantiate the initially visible plot after the event loop returns
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
        # Ensure the newly visible plot refreshes its content
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
        if chisurf.settings.gui['confirm_close_fit']:
            reply = chisurf.gui.widgets.MyMessageBox.question(
                self,
                'Message',
                "Are you sure to close this fit?:\n%s" % self.fit.name,
                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                chisurf.console.execute('chisurf.macros.close_fit()')
                chisurf.gui.widgets.hide_items_in_layout(chisurf.cs.modelLayout)
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
                    chisurf.run(s)
                    self.finalize()

                    # Adjust widget of parameter that is linker
                    self.widget_link.setToolTip(tooltip)
                    self.widget_link.setCheckState(QtCore.Qt.PartiallyChecked)
                    self.widget_value.setEnabled(False)
                    try:
                        param_other.controller.widget_link.setCheckState(QtCore.Qt.Checked)
                    except AttributeError:
                        chisurf.logging.warning("Could not set widget properties of controller")

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
        self.label.setText(label_text.ljust(5))

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
        self.widget_value.editingFinished.connect(
            lambda: chisurf.run(
                f"parameter = chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['{fitting_parameter.name}']\n"
                f"fixed = parameter.fixed \n"
                f"parameter.fixed = False\n"
                f"parameter.value = {self.widget_value.value()} \n"
                f"parameter.fixed = fixed\n"
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].finalize()"
            )
        )
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

    def set_linked(self, is_linked: bool):
        if is_linked:
            self.widget_value.setEnabled(False)
            self.widget_link.setCheckState(QtCore.Qt.PartiallyChecked)
        else:
            self.widget_link.setCheckState(QtCore.Qt.Unchecked)
            self.widget_value.setEnabled(True)

    def onLinkFitGroup(self):
        self.blockSignals(True)
        self.widget_value.setEnabled(True)
        chisurf.run(f"chisurf.macros.link_fit_group('{self.fitting_parameter.name}', {self.widget_link.checkState()})")
        self.blockSignals(False)

    def setValue(self, v):
        self.widget_value.setValue(v)

    def _on_main_bounds_on_toggled(self):
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

        if self.fitting_parameter.is_linked and getattr(self.fitting_parameter, 'link', None) is not None:
            tooltip_text += f"linked to: {self.fitting_parameter.link.name}"
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
        if self.fitting_parameter.link is not None:
            tooltip = "linked to " + self.fitting_parameter.link.name
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
        suffix=suffix,
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
