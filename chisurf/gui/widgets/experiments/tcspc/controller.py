from __future__ import annotations

from chisurf.gui import QtWidgets

import pathlib

import chisurf.decorators
import chisurf.gui.decorators
import chisurf.gui.widgets
from chisurf.experiments import reader
import chisurf.data
import chisurf.gui.widgets.fio
import chisurf.gui.widgets.experiments.widgets


class CsvTCSPCWidget(QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("tcspc_csv.ui")
    def __init__(self, *args, **kwargs):
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
                map(int, str(self.lineEdit.text()).strip().split(' '))
            )
        except ValueError:
            matrix_columns = []
        gfactor = float(self.doubleSpinBox_3.value())
        pol = 'vm'
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
                    f"cs.current_setup.is_jordi = {is_jordi}",
                    f"cs.current_setup.use_header = {(not is_jordi)}",
                    f"cs.current_setup.matrix_columns = {matrix_columns}",
                    f"cs.current_setup.g_factor = {gfactor:f}",
                    f"cs.current_setup.polarization = '{pol}'",
                    f"cs.current_setup.rep_rate = {rep_rate}",
                    f"cs.current_setup.rebin = ({rebin_x}, {rebin_y})",
                    f"cs.current_setup.dt = {dt}"
                ]
            )
        )


class TCSPCReaderControlWidget(
    reader.ExperimentReaderController,
    QtWidgets.QWidget
):
    def get_filename(self) -> pathlib.Path:
        return chisurf.gui.widgets.get_filename(
            description='CSV-TCSPC file',
            file_type='All files (*.*)',
            working_path=None
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.layout = layout
        csv_widget = chisurf.gui.widgets.fio.CsvWidget()
        self.layout.addWidget(csv_widget)
        csv_tcspc_widget = CsvTCSPCWidget()
        self.layout.addWidget(csv_tcspc_widget)


class TCSPCTTTRReaderControlWidget(
    QtWidgets.QWidget,
    reader.ExperimentReaderController
):
    def get_filename(self) -> pathlib.Path:
        self.onParametersChanged()
        return chisurf.gui.widgets.get_filename(
            description='TTTR files',
            file_type='All files (*.*)',
            working_path=None
        )

    @chisurf.gui.decorators.init_with_ui(ui_filename="tcspc_tttr.ui")
    def __init__(self, *args, **kwargs):
        self.actionParametersChanged.triggered.connect(self.onParametersChanged)
        self.onParametersChanged()

    def onParametersChanged(self):
        micro_time_coarsening = self.comboBox_2.currentText()
        channel_numbers = self.lineEdit.text()
        reading_routine = self.comboBox.currentText()
        chisurf.run(
            "\n".join(
                [
                    f"cs.current_setup.channel_numbers = np.array([{channel_numbers}], dtype=np.int8)",
                    f"cs.current_setup.reading_routine = '{reading_routine}'",
                    f"cs.current_setup.micro_time_coarsening = {micro_time_coarsening}"
                ]
            )
        )

class TCSPCSimulatorSetupWidget(QtWidgets.QWidget):

    @chisurf.gui.decorators.init_with_ui("tcspc_simulator.ui")
    def __init__(self, *args, **kwargs):
        self.selector = chisurf.gui.widgets.experiments.ExperimentalDataSelector(
            click_close=False,
            parent=self,
            context_menu_enabled=False,
            experiment=chisurf.experiments.types['tcspc']
        )
        self.verticalLayout_2.addWidget(self.selector)
        self.actionParametersChanged.triggered.connect(self.onParametersChanged)
        self.onParametersChanged()

    def get_filename(self) -> pathlib.Path:
        return pathlib.Path(self.lineEdit.text())

    def onParametersChanged(self):
        dt = self.doubleSpinBox.value()
        n_tac = self.spinBox.value()
        p0 = self.spinBox_2.value()
        sample_name = str(self.lineEdit.text())
        lt_text = self.lineEdit_2.text()
        chisurf.run(
            "\n".join(
                [
                    f"cs.current_setup.sample_name = '{sample_name}'",
                    f"cs.current_setup.dt = {dt}",
                    f"cs.current_setup.lifetime_spectrum = np.array([{lt_text}], dtype=np.float64)",
                    f"cs.current_setup.n_tac = {n_tac}",
                    f"cs.current_setup.p0 = {p0}"
                ]
            )
        )