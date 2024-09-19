from __future__ import annotations

import os
import typing
import pathlib

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets, uic, QtCore, QtGui

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
        super().update()
        self.clear()

        for nbr, fit in enumerate(chisurf.fits):
            widget_name = pathlib.Path(fit.data.name).name
            model_name = fit.model.__class__.name
            item = QtWidgets.QTreeWidgetItem(self, [str(nbr), widget_name, model_name])
            item.setToolTip(1, fit.name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
            fit.update()

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

        self.toolButton_3.clicked.connect(
            lambda: chisurf.run(f"chisurf.fits[{self.fit.fit_idx}].next_result()")
        )
        self.toolButton_5.clicked.connect(
            lambda: chisurf.run(f"chisurf.fits[{self.fit.fit_idx}].previous_result()")
        )

        if hide_fit_button:
            self.pushButton_fit.hide()
        if hide_range:
            self.toolButton_2.hide()
            self.spinBox.hide()
            self.spinBox_2.hide()
        if hide_fitting:
            self.hide()

    def onDatasetChanged(self):
        chisurf.run(f"chisurf.macros.change_selected_fit_of_group({self.selected_fit})")

    def onErrorEstimate(self):
        chisurf.run(f"cs.status.showMessage('Sampling analysis: {self.fit.name}. Please wait...')")
        filename = chisurf.gui.widgets.save_file('Error estimate', '*.er4')
        kw = chisurf.settings.cs_settings['optimization']['sampling']
        chisurf.fitting.fit.sample_fit(self.fit, filename, **kw)
        chisurf.run("cs.status.showMessage('Sampling done!')")

    def onRunFit(self):
        chisurf.run(f"cs.status.showMessage('Please wait fitting: {self.fit.name}')")
        chisurf.run(f"cs.current_fit.run(local_first={self.local_first})")
        self.fit.model.finalize()
        for pa in chisurf.fitting.parameter.FittingParameter.get_instances():
            try:
                pa.controller.finalize()
            except (AttributeError, RuntimeError):
                chisurf.logging.warning(f"Fitting parameter {pa.name} does not have a controller to update.")
        chisurf.run("cs.status.showMessage('Fitting finished!')")

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
        chisurf.run(f"cs.current_fit.fit_range = {self.xmin}, {self.xmax}")
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

        plots = list()
        for plot_class, kwargs in fit.model.plot_classes:
            plot = plot_class(fit, **kwargs)
            plot.plot_controller.hide()
            plots.append(plot)
            self.plot_tab_widget.addTab(plot, plot_class.name)
            control_layout.addWidget(plot.plot_controller)

        fit.plots = plots
        for f in fit:
            f.plots = plots

        self.on_change_plot()
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

    def on_change_plot(self):
        idx = self.plot_tab_widget.currentIndex()
        self.current_plot_controller.hide()
        self.current_plot_controller = self.fit.plots[idx].plot_controller
        self.current_plot_controller.show()

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
            else:
                event.ignore()
        else:
            event.accept()


class FittingParameterWidget(Controller):

    def make_linkcall(self, fit_idx: int, parameter_name: str):

        def linkcall():
            self.blockSignals(True)
            tooltip = " linked to " + parameter_name
            s = (
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['{self.fitting_parameter.name}'].link = "
                f"chisurf.fits[{fit_idx}].model.parameters_all_dict['{parameter_name}']"
            )
            chisurf.run(s)
            # Adjust widget of parameter that is linker
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setCheckState(QtCore.Qt.PartiallyChecked)
            self.widget_value.setEnabled(False)

            try:
                # Adjust widget of parameter that is linked to
                p = chisurf.fits[fit_idx].model.parameters_all_dict[parameter_name]
                p.controller.widget_link.setCheckState(QtCore.Qt.Checked)
            except AttributeError:
                print("Could not set widget properties of controller")

            self.blockSignals(False)

        return linkcall

    def contextMenuEvent(self, event: QtGui.QCloseEvent):

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

        self.widget_value = pg.SpinBox(
            dec=True,
            decimals=decimals,
            suffix=suffix
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

        # Display of values
        self.widget_value.setValue(float(fitting_parameter.value))
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
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].update()"
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
        self.widget_bounds_on.toggled.connect(
            lambda: chisurf.run(
                f"chisurf.fits[{self.fitting_parameter.fit_idx}].model.parameters_all_dict['{fitting_parameter.name}'].bounds_on = "
                f"{self.widget_bounds_on.isChecked()}"
            )
        )

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

    def finalize(self, *args):
        super().update(*args)
        self.blockSignals(True)

        # Update value of widget
        self.widget_value.setValue(self.fitting_parameter.value)
        self.widget_fix.setCheckState(QtCore.Qt.Checked if self.fitting_parameter.fixed else QtCore.Qt.Unchecked)

        # Tooltip
        if self.fitting_parameter.bounds_on:
            lower, upper = self.fitting_parameter.bounds
            tooltip_text = f"bound: ({lower}, {upper})\n"
        else:
            tooltip_text = "bounds: off\n"

        if self.fitting_parameter.is_linked:
            tooltip_text += f"linked to: {self.fitting_parameter.link.name}"
        self.widget_value.setToolTip(tooltip_text)

        # Error-estimate
        value = self.fitting_parameter.value
        error_estimate = self.fitting_parameter.error_estimate

        if self.fitting_parameter.fixed or not isinstance(error_estimate, float):
            self.lineEdit.setText("NA")
        else:
            rel_error = abs(error_estimate / (value + 1e-12) * 100.0)
            self.lineEdit.setText("NA" if np.isnan(rel_error) else f"{rel_error:.0f}%")

        # Link
        if self.fitting_parameter.link is not None:
            tooltip = "linked to " + self.fitting_parameter.link.name
            self.widget_link.setToolTip(tooltip)
            self.widget_value.setEnabled(False)

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
        if fitting_parameter.label_text is None:
            label_text = fitting_parameter.name
        else:
            label_text = fitting_parameter.label_text
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

