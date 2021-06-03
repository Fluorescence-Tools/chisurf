"""

"""
from __future__ import annotations
import os

import pyqtgraph as pg
from qtpy import QtWidgets, uic, QtCore, QtGui

import chisurf.fitting
import chisurf.decorators
import chisurf.gui.decorators
import chisurf.settings
import chisurf.gui.widgets
import chisurf.gui.widgets.experiments.widgets
from chisurf.gui.widgets import Controller

parameter_settings = chisurf.settings.parameter


class FittingControllerWidget(
    Controller
):

    @property
    def selected_fit(
            self
    ) -> int:
        return int(self.comboBox.currentIndex())

    @selected_fit.setter
    def selected_fit(
            self,
            v: int
    ):
        self.comboBox.setCurrentIndex(int(v))

    @property
    def current_fit_type(
            self
    ) -> str:
        return str(self.comboBox.currentText())

    def change_dataset(
            self
    ) -> None:
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
        super().__init__(
            *args,
            **kwargs
        )

        self.fit = fit
        self.curve_select = chisurf.gui.widgets.experiments.widgets.ExperimentalDataSelector(
            parent=None,
            fit=fit,
            change_event=self.change_dataset,
            setup=fit.data.experiment.__class__
        )
        uic.loadUi(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "fittingWidget.ui"
            ),
            self
        )

        self.curve_select.hide()
        fit_names = [os.path.basename(f.data.name) for f in fit]
        self.comboBox.addItems(fit_names)

        # decorate the update method of the fit
        # after decoration it should also call the update of
        # the fitting widget
        def wrapper(f):

            def update_new(
                    *args,
                    **kwargs
            ):
                f(*args, **kwargs)
                self.update(*args)
            return update_new

        self.fit.run = wrapper(self.fit.run)

        self.actionFit.triggered.connect(self.onRunFit)
        self.actionFit_range_changed.triggered.connect(self.onAutoFitRange)
        self.actionChange_dataset.triggered.connect(self.show_selector)
        self.actionSelectionChanged.triggered.connect(self.onDatasetChanged)
        self.actionErrorEstimate.triggered.connect(self.onErrorEstimate)

        if hide_fit_button:
            self.pushButton_fit.hide()
        if hide_range:
            self.toolButton_2.hide()
            self.spinBox.hide()
            self.spinBox_2.hide()
        if hide_fitting:
            self.hide()

        self.onAutoFitRange()

    def onDatasetChanged(self):
        chisurf.run(
            "chisurf.macros.change_selected_fit_of_group(%s)" % self.selected_fit
        )

    def onErrorEstimate(self):
        filename = chisurf.gui.widgets.save_file('Error estimate', '*.er4')
        kw = chisurf.settings.cs_settings['optimization']['sampling']
        chisurf.fitting.fit.sample_fit(
            self.fit,
            filename,
            **kw
        )

    def onRunFit(self):
        chisurf.run("cs.current_fit.run()")
        for pa in chisurf.fitting.parameter.FittingParameter.get_instances():
            try:
                pa.controller.finalize()
            except (AttributeError, RuntimeError):
                chisurf.logging.warning(
                    "Fitting parameter %s does not have a controller to update." % pa.name
                )

    def onAutoFitRange(self):
        try:
            self.fit.fit_range = self.fit.data.data_reader.autofitrange(
                self.fit.data
            )
        except AttributeError:
            chisurf.logging.warning(
                "Fit %s with model %s does not have an attribute data.data_reader" %
                (self.__class__.__name__, self.fit.model.__class__.__name__)
            )
        self.fit.update()


class FitSubWindow(QtWidgets.QMdiSubWindow):

    def update(self, *args):
        self.setWindowTitle(self.fit.name)
        QtWidgets.QMdiSubWindow.update(self, *args)
        self.plot_tab_widget.update(*args)

    def __init__(
            self,
            fit: chisurf.fitting.fit.FitGroup,
            control_layout: QtWidgets.QLayout,
            close_confirm: bool = None,
            fit_widget: chisurf.gui.widgets.fitting.widgets.FittingControllerWidget = None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self.fit = fit
        self.fit_widget = fit_widget
        if close_confirm is None:
            close_confirm = chisurf.settings.gui['confirm_close_fit']
        self.close_confirm = close_confirm

        layout = self.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.plot_tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.plot_tab_widget)

        self.current_plot_controller = QtWidgets.QWidget(self)
        self.current_plot_controller.hide()

        plots = list()
        for plot_class, kwargs in fit.model.plot_classes:
            plot = plot_class(
                fit,
                **kwargs
            )
            plot.plot_controller.hide()
            plots.append(plot)
            self.plot_tab_widget.addTab(plot, plot_class.name)
            control_layout.addWidget(
                plot.plot_controller
            )

        fit.plots = plots
        for f in fit:
            f.plots = plots

        self.on_change_plot()
        self.plot_tab_widget.currentChanged.connect(self.on_change_plot)
        xs, ys = chisurf.settings.cs_settings['gui']['fit_windows_size']
        self.resize(xs, ys)

    def on_change_plot(self):
        idx = self.plot_tab_widget.currentIndex()
        self.current_plot_controller.hide()
        self.current_plot_controller = self.fit.plots[idx].plot_controller
        self.current_plot_controller.show()

    def updateStatusBar(
            self,
            msg: str
    ):
        self.statusBar().showMessage(msg)

    def closeEvent(
            self,
            event: QtCore.QEvent
    ):
        if self.close_confirm:
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


class FittingParameterWidget(
    Controller
):

    def make_linkcall(
            self,
            fit_idx: int,
            parameter_name: str
    ):

        def linkcall():
            self.blockSignals(True)
            tooltip = " linked to " + parameter_name
            chisurf.run(
                "cs.current_fit.model.parameters_all_dict['%s'].link = chisurf.fits[%s].model.parameters_all_dict['%s']" %
                (
                    self.fitting_parameter.name,
                    fit_idx,
                    parameter_name
                )
            )
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

    def contextMenuEvent(
            self,
            event: QtGui.QCloseEvent
    ):

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
                    ut = a.parameters
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
                for p in fs.model.parameters_all:
                    if p is not self:
                        Action = action_submenu.addAction(p.name)
                        Action.triggered.connect(
                            self.make_linkcall(fit_idx, p.name)
                        )
                submenu.addMenu(action_submenu)

                menu.addMenu(submenu)
        menu.exec_(event.globalPos())

    def __str__(self):
        return ""

    @chisurf.gui.decorators.init_with_ui(
        ui_filename="variable_widget.ui"
    )
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
            hide_link: bool = None
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

        self.name = fitting_parameter.name
        self.fitting_parameter = fitting_parameter

        self.widget_value = pg.SpinBox(
            dec=True,
            decimals=decimals
        )
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
        self.widget_value.editingFinished.connect(lambda: chisurf.run(
            "cs.current_fit.model.parameters_all_dict['%s'].value = %s\n"
            "cs.current_fit.update()" %
            (fitting_parameter.name, self.widget_value.value()))
        )

        self.widget_fix.toggled.connect(lambda: chisurf.run(
            "cs.current_fit.model.parameters_all_dict['%s'].fixed = %s" %
            (fitting_parameter.name, self.widget_fix.isChecked()))
        )

        # Variable is bounded
        self.widget_bounds_on.toggled.connect(lambda: chisurf.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds_on = %s" %
            (fitting_parameter.name, self.widget_bounds_on.isChecked()))
        )

        self.widget_lower_bound.editingFinished.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
                (
                    fitting_parameter.name,
                    self.widget_lower_bound.value(),
                    self.widget_upper_bound.value()
                )
            )
        )

        self.widget_upper_bound.editingFinished.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
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

    def set_linked(
            self,
            is_linked: bool
    ):
        if is_linked:
            self.widget_value.setEnabled(False)
            self.widget_link.setCheckState(QtCore.Qt.PartiallyChecked)
        else:
            self.widget_link.setCheckState(QtCore.Qt.Unchecked)
            self.widget_value.setEnabled(True)

    def onLinkFitGroup(self):
        self.blockSignals(True)
        self.widget_value.setEnabled(True)
        chisurf.run(
            "chisurf.macros.link_fit_group('%s', %s)" % (
                self.fitting_parameter.name,
                self.widget_link.checkState()
            )
        )
        self.blockSignals(False)

    def finalize(self, *args):
        super().update(*args)
        self.blockSignals(True)

        # Update value of widget
        self.widget_value.setValue(self.fitting_parameter.value)
        if self.fitting_parameter.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)

        # Tooltip
        if self.fitting_parameter.bounds_on:
            lower, upper = self.fitting_parameter.bounds
            s = "bound: (%s,%s)\n" % (lower, upper)
        else:
            s = "bounds: off\n"
        if self.fitting_parameter.is_linked:
            s += "linked to: %s" % self.fitting_parameter.link.name
        self.widget_value.setToolTip(s)

        # Error-estimate
        value = self.fitting_parameter.value
        if self.fitting_parameter.fixed or not isinstance(self.fitting_parameter.error_estimate, float):
            self.lineEdit.setText("NA")
        else:
            rel_error = abs(
                self.fitting_parameter.error_estimate / (value + 1e-12) * 100.0
            )
            self.lineEdit.setText("%.0f%%" % rel_error)

        # link
        if self.fitting_parameter.link is not None:
            tooltip = " linked to " + self.fitting_parameter.link.name
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setChecked(True)
            self.widget_value.setEnabled(False)

        self.blockSignals(False)


class FittingParameterGroupWidget(QtWidgets.QGroupBox):

    def __init__(
            self,
            parameter_group: chisurf.fitting.parameter.FittingParameterGroup,
            n_col: int = None,
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
        layout = QtWidgets.QGridLayout()

        self.setLayout(layout)
        for i, p in enumerate(parameter_group.parameters_all):
            pw = chisurf.gui.widgets.fitting.make_fitting_parameter_widget(
                fitting_parameter=p,
                label_text=p.name
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
        hide_link: bool = None
) -> FittingParameterWidget:

    if label_text is None:
        label_text = fitting_parameter.name
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
        label_text=label_text
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

