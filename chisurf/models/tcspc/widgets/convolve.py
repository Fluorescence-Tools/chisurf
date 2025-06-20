from __future__ import annotations

import pathlib

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui, uic
import chisurf.gui.decorators
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments
import chisurf.fitting.fit

from chisurf.models.tcspc.nusiance import Convolve


class ConvolveWidget(Convolve, QtWidgets.QWidget):

    @property
    def fwhm(self) -> float:
        return self.irf.fwhm

    @fwhm.setter
    def fwhm(self, v: float):
        self.lineEdit_2.setText("%.3f" % v)

    @property
    def gui_mode(self):
        if self.radioButton_2.isChecked():
            return "exp"
        elif self.radioButton.isChecked():
            return "per"
        elif self.radioButton_3.isChecked():
            return "full"

    @chisurf.gui.decorators.init_with_ui("tcspc_convolve.ui")
    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            hide_curve_convolution: bool = True,
            *args,
            **kwargs
    ):
        if hide_curve_convolution:
            self.radioButton_3.setVisible(not hide_curve_convolution)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._dt, layout=layout)
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._n0, layout=layout)
        self.verticalLayout_2.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._start, layout=layout)
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._stop, layout=layout)
        self.verticalLayout_2.addLayout(layout)

        # Add IRF start and stop parameters
        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._irf_start, layout=layout)
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._irf_stop, layout=layout)
        self.verticalLayout_2.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._lb, layout=layout)
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._ts, layout=layout)
        self.verticalLayout_2.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._iw, layout=layout)
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(self._ik, layout=layout)
        self.verticalLayout_2.addLayout(layout)

        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            fitting_parameter=self._rep,
            layout=self.horizontalLayout_3,
            label_text='r[MHz]'
        )

        # Button to unload IRF
        self.unload_button.setToolTip("Unload IRF")
        self.unload_button.clicked.connect(self.onUnloadIRF)

        self.irf_select = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            parent=None,
            change_event=self.change_irf,
            fit=self.fit,
            experiment=self.fit.data.experiment.__class__
        )

        self.actionSelect_IRF.triggered.connect(self.irf_select.show)
        self.radioButton_3.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton_2.clicked.connect(self.onConvolutionModeChanged)
        self.radioButton.clicked.connect(self.onConvolutionModeChanged)
        self.checkBox.clicked.connect(self.onConvolutionModeChanged)

    def onConvolutionModeChanged(self):
        chisurf.run(
            "\n".join(
                [
                    f"for f in cs.current_fit:",
                    f"   f.model.convolve.mode = '{self.gui_mode}'",
                    f"cs.current_fit.model.convolve.do_convolution = {self.checkBox.isChecked()}",
                    f"cs.current_fit.update()"
                ]
            )
        )

    def change_irf(self):
        idx = self.irf_select.selected_curve_index
        name = self.irf_select.curve_name
        chisurf.run(f"chisurf.macros.model.change_irf({idx}, r'{name}')")
        self.fwhm = self.irf.fwhm

    def onUnloadIRF(self):
        """Unload the IRF and reset it to default (None)
        """
        chisurf.run("cs.current_fit.model.convolve.unload_irf()")
        self.lineEdit.setText("")
        chisurf.run("cs.current_fit.model.update()")