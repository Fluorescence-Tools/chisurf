from __future__ import annotations

import chisurf
from chisurf.gui import QtWidgets, QtCore, QtGui, uic
import chisurf.gui.decorators
import chisurf.gui.widgets.fitting
import chisurf.gui.widgets.experiments
import chisurf.fitting.fit
import chisurf.math.signal

from chisurf.models.tcspc.nusiance import Corrections


class CorrectionsWidget(Corrections, QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("tcspcCorrections.ui")
    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit = None,
            hide_corrections: bool = False,
            **kwargs
    ):
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
            lambda: chisurf.run(
                "cs.current_fit.model.corrections.correct_pile_up = %s\n" %
                self.checkBox_3.isChecked()
            )
        )

        self.checkBox_2.toggled.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.corrections.reverse = %s" %
                self.checkBox_2.isChecked()
            )
        )

        self.checkBox.toggled.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.corrections.correct_dnl = %s" %
                self.checkBox.isChecked()
            )
        )

        self.comboBox.currentIndexChanged.connect(
            lambda: chisurf.run(
                "cs.current_fit.model.corrections.window_function = '%s'" %
                self.comboBox.currentText()
            )
        )

    def onChangeLin(self):
        idx = self.lin_select.selected_curve_index
        lin_name = self.lin_select.curve_name
        chisurf.run(
            "chisurf.macros.model.set_linearization(%s, '%s')" %
            (idx, lin_name)
        )

    def onUnloadLin(self):
        """Unload the linearization table and reset it to default (array of ones)
        """
        chisurf.run("cs.current_fit.model.corrections.unload_lintable()")
        self.lineEdit.setText("")
        chisurf.run("cs.current_fit.model.update()")