from __future__ import annotations

from typing import TYPE_CHECKING
import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui, uic
import chisurf.gui.decorators
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments
import chisurf.math.signal

from chisurf.models.tcspc.nusiance import Corrections

if TYPE_CHECKING:
    from chisurf.fitting.fit import Fit


class CorrectionsWidget(Corrections, QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("tcspcCorrections.ui")
    # TODO: needs docstring
    def __init__(
            self,
            fit: Fit | None = None,
            hide_corrections: bool = False,
            **kwargs
    ):
        """Initialize the instance."""
        self.groupBox.setChecked(False)
        self.comboBox.addItems(chisurf.math.signal.window_function_types)
        if hide_corrections:
            self.hide()

        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._dead_time,
            layout=self.horizontalLayout_2,
            label_text='t<sub>dead</sub>[ns]'
        )
        chisurf.gui.widgets.fitting.widgets.make_fitting_parameter_widget(
            self._window_length,
            layout=self.horizontalLayout_2,
            label_text='Lin.win.'
        )

        self.lin_select = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            parent=None,
            change_event=self.onChangeLin,
            fit=fit,
            experiment=fit.data.experiment.__class__
        )

        # Add toolbutton to unload linearization table
        self.unload_button = QtWidgets.QToolButton()
        self.unload_button.setText("x")
        self.unload_button.setToolTip("Unload linearization table")
        self.unload_button.clicked.connect(self.onUnloadLin)
        self.horizontalLayout.addWidget(self.unload_button)

        self.actionSelect_lintable.triggered.connect(self.lin_select.show)

        self.checkBox_3.toggled.connect(
            lambda: chisurf.actions.dispatch(
                name="model.set_correction",
                payload={
                    "correction_type": "correct_pile_up",
                    "value": bool(self.checkBox_3.isChecked()),
                },
            )
        )

        self.checkBox_2.toggled.connect(
            lambda: chisurf.actions.dispatch(
                name="model.set_correction",
                payload={
                    "correction_type": "reverse",
                    "value": bool(self.checkBox_2.isChecked()),
                },
            )
        )

        self.checkBox.toggled.connect(
            lambda: chisurf.actions.dispatch(
                name="model.set_correction",
                payload={
                    "correction_type": "correct_dnl",
                    "value": bool(self.checkBox.isChecked()),
                },
            )
        )

        self.comboBox.currentIndexChanged.connect(
            lambda: chisurf.actions.dispatch(
                name="model.set_correction",
                payload={
                    "correction_type": "window_function",
                    "value": str(self.comboBox.currentText()),
                },
            )
        )

        try:
            self._install_code_badge()
        except Exception:
            pass

    def _install_code_badge(self):
        """Install a code badge for dev mode source jumping."""
        try:
            import chisurf.settings
            if not chisurf.settings.is_dev_mode():
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

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of this widget's state.

        Delegates to :class:`Corrections.get_state` for the core linearization
        state and additionally records the text shown in the line edit (name
        of the lintable source) when available.
        """

        try:
            state = Corrections.get_state(self)
        except Exception:
            return {}

        try:
            le = getattr(self, "lineEdit", None)
            if le is not None:
                txt = str(le.text())
                if txt:
                    state["lin_name"] = txt
        except Exception:
            pass

        return state

    def set_state(self, state: dict) -> None:
        """Restore corrections state and synchronize the widget UI.

        The underlying model attributes are restored via
        :class:`Corrections.set_state`. Checkboxes and the window-function
        combobox are then updated without emitting their signals so that
        project load does not execute global macros.
        """

        if not isinstance(state, dict):
            return

        try:
            Corrections.set_state(self, state)
        except Exception:
            return

        # Sync DNL / reverse / pile-up checkboxes
        try:
            self.checkBox.blockSignals(True)
            self.checkBox.setChecked(bool(getattr(self, "correct_dnl", self.checkBox.isChecked())))
        except Exception:
            pass
        finally:
            try:
                self.checkBox.blockSignals(False)
            except Exception:
                pass

        try:
            self.checkBox_3.blockSignals(True)
            self.checkBox_3.setChecked(bool(getattr(self, "correct_pile_up", self.checkBox_3.isChecked())))
        except Exception:
            pass
        finally:
            try:
                self.checkBox_3.blockSignals(False)
            except Exception:
                pass

        try:
            self.checkBox_2.blockSignals(True)
            self.checkBox_2.setChecked(bool(getattr(self, "reverse", self.checkBox_2.isChecked())))
        except Exception:
            pass
        finally:
            try:
                self.checkBox_2.blockSignals(False)
            except Exception:
                pass

        # Sync window-function combobox
        try:
            wf = getattr(self, "window_function", None)
            self.comboBox.blockSignals(True)
            if isinstance(wf, str):
                idx = self.comboBox.findText(wf)
                if idx >= 0:
                    self.comboBox.setCurrentIndex(idx)
        except Exception:
            pass
        finally:
            try:
                self.comboBox.blockSignals(False)
            except Exception:
                pass

        # Restore lintable label text, if present
        try:
            lin_name = state.get("lin_name")
            if isinstance(lin_name, str) and lin_name:
                self.lineEdit.setText(lin_name)
        except Exception:
            pass

    # TODO: needs docstring
    def onChangeLin(self):
        """Handle linearization table selection."""
        idx = self.lin_select.selected_curve_index
        lin_name = self.lin_select.curve_name

        chisurf.actions.dispatch(
            name="model.set_linearization",
            payload={
                "idx": int(idx),
                "lin_name": str(lin_name),
            },
        )

    def onUnloadLin(self):
        """Unload the linearization table and reset it to default (array of ones)
        """
        chisurf.actions.dispatch(
            name="model.unload_lintable",
            payload={},
        )
        self.lineEdit.setText("")
        chisurf.actions.dispatch(
            name="model.update",
            payload={},
        )
