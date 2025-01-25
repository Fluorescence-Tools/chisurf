import pathlib
import tttrlib

import chisurf.gui
from chisurf.experiments import reader

from chisurf.gui import QtWidgets, QtGui, QtCore


class PdaTTTRWidget(
    QtWidgets.QWidget,
    reader.ExperimentReaderController
):

    @chisurf.gui.decorators.init_with_ui("pda_tttr.ui")
    def __init__(self, *args, **kwargs):
        # super().__init__(parent=parent)
        self.actionParametersChanged.triggered.connect(self.onParametersChanged)
        self.onParametersChanged()
        self.comboBox.clear()
        self.comboBox.insertItems(0, tttrlib.TTTR.get_supported_container_names())

    def onParametersChanged(self):
        ch0 = [int(k) for k in self.lineEdit.text().split(',')]
        ch1 = [int(k) for k in self.lineEdit_4.text().split(',')]
        mt0 = [[int(j) for j in k.split('-')] for k in self.lineEdit_2.text().split(';')]
        mt1 = [[int(j) for j in k.split('-')] for k in self.lineEdit_3.text().split(';')]
        micro_time_ranges = [mt0, mt1]
        channels = [ch0, ch1]
        minimum_number_of_photons = self.spinBox.value()
        maximum_number_of_photons = self.spinBox_2.value()
        minimum_time_window_length = self.doubleSpinBox.value()

        reading_routine = self.comboBox.currentText()
        chisurf.run(
            "\n".join(
                [
                    f"cs.current_setup.reading_routine = '{reading_routine}'",
                    f"cs.current_setup.channels = {channels}",
                    f"cs.current_setup.micro_time_ranges = {micro_time_ranges}",
                    f"cs.current_setup.minimum_number_of_photons = {minimum_number_of_photons}",
                    f"cs.current_setup.maximum_number_of_photons = {maximum_number_of_photons}",
                    f"cs.current_setup.minimum_time_window_length = {minimum_time_window_length / 1000.0}"
                ]
            )
        )

    def get_filename(self) -> pathlib.Path:
        return chisurf.gui.widgets.open_files(
            description='HT3/PTU/SPC file',
            file_type='All files (*.*)',
            working_path=None
        )
