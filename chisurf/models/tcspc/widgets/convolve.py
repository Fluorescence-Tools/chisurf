from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui, uic
import chisurf.gui.decorators
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments

from chisurf.models.tcspc.nusiance import Convolve

if TYPE_CHECKING:
    from chisurf.fitting.fit import Fit


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
            fit: Fit,
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

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of this widget's state.

        Delegates to :class:`Convolve.get_state`, which already captures the
        IRF curve, convolution flags and the IRF label text shown in the
        line edit.
        """

        try:
            return Convolve.get_state(self)
        except Exception:
            return {}

    def set_state(self, state: dict) -> None:
        """Restore convolution/IRF state and synchronize the widget UI.

        This uses the model-level :class:`Convolve.set_state` implementation
        and then updates local controls (checkbox, radio buttons, FWHM
        display) without emitting their change signals, to avoid triggering
        global macros during project load.
        """

        if not isinstance(state, dict):
            return

        try:
            Convolve.set_state(self, state)
        except Exception:
            return

        # Sync "do_convolution" checkbox
        try:
            self.checkBox.blockSignals(True)
            self.checkBox.setChecked(bool(getattr(self, "do_convolution", self.checkBox.isChecked())))
        except Exception:
            pass
        finally:
            try:
                self.checkBox.blockSignals(False)
            except Exception:
                pass

        # Sync mode radio buttons from the restored ``mode`` attribute
        try:
            mode = getattr(self, "mode", None)
            self.radioButton.blockSignals(True)
            self.radioButton_2.blockSignals(True)
            self.radioButton_3.blockSignals(True)
            if isinstance(mode, str):
                if mode == "exp":
                    self.radioButton_2.setChecked(True)
                elif mode == "per":
                    self.radioButton.setChecked(True)
                elif mode == "full":
                    self.radioButton_3.setChecked(True)
        except Exception:
            pass
        finally:
            try:
                self.radioButton.blockSignals(False)
                self.radioButton_2.blockSignals(False)
                self.radioButton_3.blockSignals(False)
            except Exception:
                pass

        # Update FWHM line edit if an IRF is present
        try:
            irf = self.irf
            if irf is not None and hasattr(irf, "fwhm"):
                self.fwhm = irf.fwhm
        except Exception:
            pass

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