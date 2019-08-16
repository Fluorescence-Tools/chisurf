import os

from PyQt5 import QtWidgets, uic

import mfm
import mfm.widgets
from mfm.fitting.fit import sample_fit


class FittingControllerWidget(QtWidgets.QWidget):

    @property
    def selected_fit(self):
        return int(self.comboBox.currentIndex())
        #return int(self.spinBox.value())

    @selected_fit.setter
    def selected_fit(self, v):
        self.comboBox.setCurrentIndex(int(v))

    @property
    def current_fit_type(self):
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
        QtWidgets.QWidget.__init__(self)
        self.curve_select = widgets.CurveSelector(parent=None, fit=self,
                                                  change_event=self.change_dataset,
                                                  setup=fit.data.setup.__class__)
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
        kw = mfm.cs_settings['fitting']['sampling']
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
        self.close_confirm = kwargs.get('close_confirm', mfm.cs_settings['gui']['confirm_close_fit'])
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
                mfm.console.execute('cs.close_fit()')
            else:
                event.ignore()
        else:
            event.accept()
