from __future__ import annotations
import os

import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, QtCore

import mfm
import mfm.widgets
from mfm.fitting.fit import sample_fit

parameter_settings = mfm.settings.cs_settings['parameter']


class FittingControllerWidget(QtWidgets.QWidget):

    @property
    def selected_fit(self) -> int:
        return int(self.comboBox.currentIndex())
        #return int(self.spinBox.value())

    @selected_fit.setter
    def selected_fit(
            self,
            v: int
    ):
        self.comboBox.setCurrentIndex(int(v))

    @property
    def current_fit_type(self) -> str:
        return str(self.comboBox.currentText())

    def change_dataset(self):
        dataset = self.curve_select.selected_dataset
        self.fit.data = dataset
        self.fit.update()
        self.lineEdit.clear()
        self.lineEdit.setText(dataset[0].name)

    def show_selector(self):
        self.curve_select.show()
        self.curve_select.update()

    def __init__(self, fit=None, **kwargs):
        super(FittingControllerWidget, self).__init__()
        self.curve_select = mfm.widgets.CurveSelector(
            parent=None, fit=self,
            change_event=self.change_dataset,
            setup=fit.data.setup.__class__
        )
        uic.loadUi("mfm/ui/fittingWidget.ui", self)

        self.curve_select.hide()
        hide_fit_button = kwargs.get('hide_fit_button', False)
        hide_range = kwargs.get('hide_range', False)
        hide_fitting = kwargs.get('hide_fitting', False)
        self.fit = fit
        fit_names = [os.path.basename(f.data.name) for f in fit]
        self.comboBox.addItems(fit_names)

        # decorate the update method of the fit
        # after decoration it should also call the update of
        # the fitting widget
        def wrapper(f):
            def update_new(*args, **kwargs):
                f(*args, **kwargs)
                self.update(*args)
                self.fit.model.update_plots(only_fit_range=True)
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

        #self.spinBox.setMaximum(len(fit) - 1)
        #self.lineEdit.setText(fit.data.name)
        self.onAutoFitRange()

    def onDatasetChanged(self):
        mfm.run("cs.current_fit.model.hide()")
        mfm.run("cs.current_fit.current_fit = %i" % self.selected_fit)
        mfm.run("cs.current_fit.update()")
        mfm.run("cs.current_fit.model.show()")
        #name = self.fit.data.name
        #self.lineEdit.setText(name)

    def onErrorEstimate(self):
        filename = mfm.widgets.save_file('Error estimate', '*.er4')
        kw = mfm.settings.cs_settings['fitting']['sampling']
        sample_fit(self.fit, filename, **kw)

    def onRunFit(self):
        mfm.run("cs.current_fit.run()")

    def onAutoFitRange(self):
        try:
            self.fit.fit_range = self.fit.data.setup.autofitrange(self.fit.data)
        except AttributeError:
            self.fit.fit_range = 0, len(self.fit.data.x)
        self.fit.update()


class FitSubWindow(QtWidgets.QMdiSubWindow):

    def update(self, *__args):
        self.setWindowTitle(self.fit.name)
        QtWidgets.QMdiSubWindow.update(self, *__args)
        self.tw.update(self, *__args)

    def __init__(self, fit, control_layout, **kwargs):
        QtWidgets.QMdiSubWindow.__init__(self, kwargs.get('parent', None))
        self.setWindowTitle(fit.name)
        l = self.layout()

        self.tw = QtWidgets.QTabWidget()
        self.tw.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tw.setTabPosition(QtWidgets.QTabWidget.South)
        self.tw.currentChanged.connect(self.on_change_plot)

        l.addWidget(self.tw)
        self.close_confirm = kwargs.get('close_confirm', mfm.settings.cs_settings['gui']['confirm_close_fit'])
        self.fit = fit
        self.fit_widget = kwargs.get('fit_widget')

        self.current_plt_ctrl = QtWidgets.QWidget(self)
        self.current_plt_ctrl.hide()

        plots = list()
        for plot_class, kwargs in fit.model.plot_classes:
            plot = plot_class(fit, **kwargs)
            plot.pltControl.hide()
            plots.append(plot)
            self.tw.addTab(plot, plot.name)
            control_layout.addWidget(plot.pltControl)

        fit.plots = plots
        for f in fit:
            f.plots = plots

        self.on_change_plot()

    def on_change_plot(self):
        idx = self.tw.currentIndex()
        self.current_plt_ctrl.hide()
        self.current_plt_ctrl = self.fit.plots[idx].pltControl
        self.current_plt_ctrl.show()

    def updateStatusBar(self, msg):
        self.statusBar().showMessage(msg)

    def closeEvent(self, event):
        if self.close_confirm:
            reply = QtWidgets.QMessageBox.question(self, 'Message',
                                                   "Are you sure to close this fit?:\n%s" % self.fit.name,
                                                   QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.Yes:
                mfm.console.execute('mfm.cmd.close_fit()')
            else:
                event.ignore()
        else:
            event.accept()


class FittingParameterWidget(QtWidgets.QWidget):

    def make_linkcall(self, fit_idx, parameter_name):

        def linkcall():
            tooltip = " linked to " + parameter_name
            mfm.run(
                "cs.current_fit.model.parameters_all_dict['%s'].link = mfm.fits[%s].model.parameters_all_dict['%s']" %
                (self.name, fit_idx, parameter_name)
            )
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setChecked(True)
            self.widget_value.setEnabled(False)

        self.update()
        return linkcall

    def contextMenuEvent(self, event):

        menu = QtWidgets.QMenu(self)
        menu.setTitle("Link " + self.fitting_parameter.name + " to:")

        for fit_idx, f in enumerate(mfm.fits):
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
                        if p is not self:
                            Action = action_submenu.addAction(p.name)
                            Action.triggered.connect(self.make_linkcall(fit_idx, p.name))
                    submenu.addMenu(action_submenu)
                action_submenu = QtWidgets.QMenu(submenu)

                # Simply all parameters
                action_submenu.setTitle("All parameters")
                for p in fs.model.parameters_all:
                    if p is not self:
                        Action = action_submenu.addAction(p.name)
                        Action.triggered.connect(self.make_linkcall(fit_idx, p.name))
                submenu.addMenu(action_submenu)

                menu.addMenu(submenu)
        menu.exec_(event.globalPos())

    def __str__(self):
        return ""

    def __init__(self, fitting_parameter, **kwargs):
        layout = kwargs.pop('layout', None)
        label_text = kwargs.pop('text', self.__class__.__name__)
        hide_bounds = kwargs.pop('hide_bounds', parameter_settings['hide_bounds'])
        hide_link = kwargs.pop('hide_link', parameter_settings['hide_link'])
        fixable = kwargs.pop('fixable', parameter_settings['fixable'])
        hide_fix_checkbox = kwargs.pop('hide_fix_checkbox', parameter_settings['fixable'])
        hide_error = kwargs.pop('hide_error', parameter_settings['hide_error'])
        hide_label = kwargs.pop('hide_label', parameter_settings['hide_label'])
        decimals = kwargs.pop('decimals', parameter_settings['decimals'])

        super(FittingParameterWidget, self).__init__(**kwargs)
        uic.loadUi('mfm/ui/variable_widget.ui', self)
        self.fitting_parameter = fitting_parameter
        self.widget_value = pg.SpinBox(dec=True, decimals=decimals)
        self.widget_value.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.widget_value.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addWidget(self.widget_value)

        self.widget_lower_bound = pg.SpinBox(dec=True, decimals=decimals)
        self.widget_lower_bound.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addWidget(self.widget_lower_bound)

        self.widget_upper_bound = pg.SpinBox(dec=True, decimals=decimals)
        self.widget_upper_bound.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addWidget(self.widget_upper_bound)

        # Hide and disable widgets
        self.label.setVisible(not hide_label)
        self.lineEdit.setVisible(not hide_error)
        self.widget_bounds_on.setDisabled(hide_bounds)
        self.widget_fix.setVisible(fixable or not hide_fix_checkbox)
        self.widget.setHidden(hide_bounds)
        self.widget_link.setDisabled(hide_link)

        # Display of values
        self.widget_value.setValue(float(self.fitting_parameter.value))

        self.label.setText(label_text.ljust(5))

        # variable bounds
        if not self.fitting_parameter.bounds_on:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Checked)

        # variable fixed
        if self.fitting_parameter.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        self.widget.hide()

        # The variable value
        self.widget_value.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].value = %s\n"
            "cs.current_fit.update()" %
            (self.fitting_parameter.name, self.widget_value.value()))
        )

        self.widget_fix.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].fixed = %s" %
            (self.fitting_parameter.name, self.widget_fix.isChecked()))
        )

        # Variable is bounded
        self.widget_bounds_on.toggled.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds_on = %s" %
            (self.fitting_parameter.name, self.widget_bounds_on.isChecked()))
        )

        self.widget_lower_bound.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
            (self.fitting_parameter.name, self.widget_lower_bound.value(), self.widget_upper_bound.value()))
        )

        self.widget_upper_bound.editingFinished.connect(lambda: mfm.run(
            "cs.current_fit.model.parameters_all_dict['%s'].bounds = (%s, %s)" %
            (self.fitting_parameter.name, self.widget_lower_bound.value(), self.widget_upper_bound.value()))
        )

        self.widget_link.clicked.connect(self.onLinkFitGroup)

        if isinstance(layout, QtWidgets.QLayout):
            layout.addWidget(self)

    def onLinkFitGroup(self):
        self.blockSignals(True)
        cs = self.widget_link.checkState()
        if cs == 2:
            t = """
s = cs.current_fit.model.parameters_all_dict['%s']
for f in cs.current_fit:
   try:
       p = f.model.parameters_all_dict['%s']
       if p is not s:
           p.link = s
   except KeyError:
       pass
            """ % (self.fitting_parameter.name, self.fitting_parameter.name)
            mfm.run(t)
            self.widget_link.setCheckState(QtCore.Qt.Checked)
            self.widget_value.setEnabled(False)
        elif cs == 0:
            t = """
s = cs.current_fit.model.parameters_all_dict['%s']
for f in cs.current_fit:
   try:
       p = f.model.parameters_all_dict['%s']
       p.link = None
   except KeyError:
       pass
            """ % (self.fitting_parameter.name, self.fitting_parameter.name)
            mfm.run(t)
        self.widget_value.setEnabled(True)
        self.widget_link.setCheckState(QtCore.Qt.Unchecked)
        self.blockSignals(False)

    def finalize(self, *args):
        QtWidgets.QWidget.update(self, *args)
        self.blockSignals(True)
        # Update value of widget
        self.widget_value.setValue(float(self.fitting_parameter.value))
        if self.fitting_parameter.fixed:
            self.widget_fix.setCheckState(QtCore.Qt.Checked)
        else:
            self.widget_fix.setCheckState(QtCore.Qt.Unchecked)
        # Tooltip
        s = "bound: (%s,%s)\n" % self.fitting_parameter.bounds if self.fitting_parameter.bounds_on else "bounds: off\n"
        if self.fitting_parameter.is_linked:
            s += "linked to: %s" % self.fitting_parameter.link.name
        self.widget_value.setToolTip(s)

        # Error-estimate
        value = self.fitting_parameter.value
        if self.fitting_parameter.fixed or not isinstance(self.fitting_parameter.error_estimate, float):
            self.lineEdit.setText("NA")
        else:
            rel_error = abs(self.fitting_parameter.error_estimate / (value + 1e-12) * 100.0)
            self.lineEdit.setText("%.0f%%" % rel_error)
        #link
        if self.fitting_parameter.link is not None:
            tooltip = " linked to " + self.fitting_parameter.link.name
            self.widget_link.setToolTip(tooltip)
            self.widget_link.setChecked(True)
            self.widget_value.setEnabled(False)

        self.blockSignals(False)


class FittingParameterGroupWidget(QtWidgets.QGroupBox):

    def __init__(
            self,
            parameter_group,
            n_col: int = None,
            *args, **kwargs):
        self.parameter_group = parameter_group
        super(FittingParameterGroupWidget, self).__init__(*args, **kwargs)
        self.setTitle(parameter_group.name)
        self.n_col = n_col if isinstance(n_col, int) else mfm.settings.cs_settings['gui']['fit_models']['n_columns']
        self.n_row = 0
        l = QtWidgets.QGridLayout()
        self.setLayout(l)

        for i, p in enumerate(parameter_group.parameters_all):
            pw = p.make_widget()
            col = i % self.n_col
            row = i // self.n_col
            l.addWidget(pw, row, col)


def make_fitting_parameter_widget(
        fitting_parameter,
        **kwargs
) -> FittingParameterWidget:
    text = kwargs.get('text', fitting_parameter.name)
    layout = kwargs.get('layout', None)
    update_widget = kwargs.get('update_widget', lambda x: x)
    decimals = kwargs.get('decimals', fitting_parameter.decimals)
    kw = {
        'text': text,
        'decimals': decimals,
        'layout': layout
    }
    widget = FittingParameterWidget(fitting_parameter, **kw)
    fitting_parameter.controller = widget
    return widget


def make_fitting_parameter_group_widget(
        fitting_parameter_group: mfm.fitting.parameter.FittingParameterGroup,
        *args,
        **kwargs
):
    return FittingParameterGroupWidget(fitting_parameter_group, *args, **kwargs)

