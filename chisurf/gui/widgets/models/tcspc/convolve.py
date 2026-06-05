from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import chisurf
from qtpy import QtWidgets, QtCore, QtGui, uic
import chisurf.gui.decorators
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments

from chisurf.core.models.tcspc.nusiance import Convolve

if TYPE_CHECKING:
    from chisurf.core.fitting.fit import Fit


class ConvolveWidget(Convolve, QtWidgets.QWidget):

    def _resolve_fit_group_index(self):
        """Resolve the fit group index for the current fit.

        Returns
        -------
        int or None
            The fit group index, or None if it could not be resolved.
        """
        try:
            target = getattr(self.fit, "fit_group", None) or self.fit
            for idx, fit_group in enumerate(list(getattr(chisurf, "fits", []) or [])):
                if fit_group is target:
                    return int(idx)
                try:
                    if target in fit_group:
                        return int(idx)
                except Exception:
                    pass
        except Exception:
            pass
        return None

    @property
    def fwhm(self) -> float:
        """Full width at half maximum of the IRF."""
        return self.irf.fwhm

    @fwhm.setter
    def fwhm(self, v: float):
        """Full width at half maximum of the IRF."""
        self.lineEdit_2.setText("%.3f" % v)

    def _refresh_fwhm_display(self):
        """Update the displayed FWHM value from the current IRF."""
        try:
            irf = self.irf
            if irf is not None and hasattr(irf, "fwhm"):
                self.fwhm = float(irf.fwhm)
        except Exception:
            pass

    @property
    def gui_mode(self):
        """Currently selected convolution mode in the GUI."""
        if self.radioButton_2.isChecked():
            return "exp"
        elif self.radioButton.isChecked():
            return "per"
        elif self.radioButton_3.isChecked():
            return "full"

    @chisurf.gui.decorators.init_with_ui("tcspc_convolve.ui")
    # TODO: needs docstring
    def __init__(
            self,
            fit: Fit,
            hide_curve_convolution: bool = True,
            *args,
            **kwargs
    ):
        """Initialize the instance."""
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

        try:
            self._install_code_badge()
        except Exception:
            pass

        self._refresh_fwhm_display()

    def _install_code_badge(self):
        """Install a code badge for dev mode source jumping."""
        try:
            import chisurf.core.settings
            if not chisurf.core.settings.is_dev_mode():
                return
            if hasattr(self, '_chisurf_code_badge_installed'):
                return
            from chisurf.gui.widgets.code_badge import install_code_badge
            from chisurf.gui.devtools.source_jump import resolve_object_source
            resolver = lambda: resolve_object_source(self)
            install_code_badge(self, resolver, corner='top-right', margin=4)
            self._chisurf_code_badge_installed = True
        except Exception:
            pass

    # TODO: needs docstring
    def onConvolutionModeChanged(self):
        """Handle convolution mode change."""
        # This complex operation sets multiple properties and updates the model
        # For now, we'll use the existing model update action and handle the
        # property changes through the model's own methods
        try:
            import chisurf
            for f in chisurf.fits:
                f.model.convolve.mode = self.gui_mode
            chisurf.fits[0].model.convolve.do_convolution = self.checkBox.isChecked()
            chisurf.core.actions.dispatch(
                name="model.update",
                payload={},
            )
        except Exception:
            pass

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

        self._refresh_fwhm_display()

    # TODO: needs docstring
    def change_irf(self):
        """Handle IRF selection change."""
        idx = self.irf_select.selected_curve_index
        name = self.irf_select.curve_name
        payload = {
            "irf_idx": int(idx),
            "irf_name": str(name),
        }
        fit_index = self._resolve_fit_group_index()
        if fit_index is not None:
            payload["fit_index"] = int(fit_index)
        chisurf.core.actions.dispatch(
            name="model.change_irf",
            payload=payload,
        )
        self._refresh_fwhm_display()

    # TODO: needs docstring
    def onUnloadIRF(self):
        """Handle IRF unload."""
        payload = {}
        fit_index = self._resolve_fit_group_index()
        if fit_index is not None:
            payload["fit_index"] = int(fit_index)
        chisurf.core.actions.dispatch(
            name="model.unload_irf",
            payload=payload,
        )
        chisurf.core.actions.dispatch(
            name="model.update",
            payload={},
        )
        self._refresh_fwhm_display()
