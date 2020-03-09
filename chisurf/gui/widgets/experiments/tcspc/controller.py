from __future__ import annotations

from qtpy import QtWidgets

import pathlib

import chisurf.decorators
import chisurf.gui.widgets
from chisurf.experiments import reader
import chisurf.experiments.data
import chisurf.gui.widgets.fio
import chisurf.gui.widgets.experiments.widgets


class CsvTCSPCWidget(
    QtWidgets.QWidget,
):

    @chisurf.decorators.init_with_ui(
        ui_filename="csvTCSPCWidget.ui"
    )
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.actionDtChanged.triggered.connect(self.onParametersChanged)
        self.actionRebinChanged.triggered.connect(self.onParametersChanged)
        self.actionRepratechange.triggered.connect(self.onParametersChanged)
        self.actionPolarizationChange.triggered.connect(self.onParametersChanged)
        self.actionGfactorChanged.triggered.connect(self.onParametersChanged)
        self.actionIsjordiChanged.triggered.connect(self.onParametersChanged)
        self.actionMatrixColumnsChanged.triggered.connect(self.onParametersChanged)

    def onParametersChanged(self):
        is_jordi = bool(self.checkBox_3.isChecked())
        try:
            matrix_columns = list(
                map(
                    int,
                    str(
                        self.lineEdit.text()
                    ).strip().split(' ')
                )
            )
        except ValueError:
            matrix_columns = []
        gfactor = float(self.doubleSpinBox_3.value())
        pol = 'vm'
        if self.radioButton_4.isChecked():
            pol = 'vv/vh'
        if self.radioButton_3.isChecked():
            pol = 'vv'
        elif self.radioButton_2.isChecked():
            pol = 'vh'
        elif self.radioButton.isChecked():
            pol = 'vm'
        rep_rate = self.doubleSpinBox.value()
        rebin_y = int(self.comboBox.currentText())
        rebin_x = int(self.comboBox_2.currentText())
        rebin = int(self.comboBox.currentText())
        dt = float(
            self.doubleSpinBox_2.value()
        ) * rebin if self.checkBox_2.isChecked() else 1.0 * rebin
        chisurf.run(
            "\n".join(
                [
                    "cs.current_setup.is_jordi = %s" % is_jordi,
                    "cs.current_setup.use_header = %s" % (not is_jordi),
                    "cs.current_setup.matrix_columns = %s" % matrix_columns,
                    "cs.current_setup.g_factor = %f" % gfactor,
                    "cs.current_setup.polarization = '%s'" % pol,
                    "cs.current_setup.rep_rate = %s" % rep_rate,
                    "cs.current_setup.rebin = (%s, %s)" % (rebin_x, rebin_y),
                    "cs.current_setup.dt = %s" % dt
                ]
            )
        )


class TCSPCReaderControlWidget(
    reader.ExperimentReaderController,
    QtWidgets.QWidget
):

    def get_filename(
            self
    ) -> pathlib.Path:
        return chisurf.gui.widgets.get_filename(
            description='CSV-TCSPC file',
            file_type='All files (*.*)',
            working_path=None
        )

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layout = layout
        self.layout.addWidget(
            chisurf.gui.widgets.fio.CsvWidget()
        )
        self.layout.addWidget(
            CsvTCSPCWidget()
        )


class TCSPCSetupDummyWidget(
    QtWidgets.QWidget
):

    @chisurf.decorators.init_with_ui(ui_filename="tcspcDummy.ui")
    def __init__(
            self,
            *args,
            **kwargs
    ):
        self.selector = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            click_close=False,
            parent=self,
            context_menu_enabled=False
        )
        self.verticalLayout_2.addWidget(
            self.selector
        )
        self.actionParametersChanged.triggered.connect(self.onParametersChanged)

    def onParametersChanged(self):
        dt = self.doubleSpinBox.value()
        n_tac = self.spinBox.value()
        lifetime = self.doubleSpinBox_2.value()
        p0 = self.spinBox_2.value()
        sample_name = str(self.lineEdit.text())
        chisurf.run(
            "\n".join(
                [
                    "cs.current_setup.sample_name = '%s'" % sample_name,
                    "cs.current_setup.dt = %s" % dt,
                    "cs.current_setup.lifetime_spectrum = [1.0, %s]" % lifetime,
                    "cs.current_setup.n_tac = %s" % n_tac,
                    "cs.current_setup.p0 = %s" % p0
                ]
            )
        )