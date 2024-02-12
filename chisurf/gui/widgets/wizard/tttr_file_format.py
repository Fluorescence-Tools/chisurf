import pathlib
import tttrlib
import json

from chisurf.gui import QtWidgets
from typing import TypedDict

import chisurf.gui.decorators



class FileFormatSettings(TypedDict):
    header: dict
    macro_time_resolution: float
    micro_time_resolution: float
    micro_time_binning: int


class WizardTTTRFileFormat(QtWidgets.QWizardPage):

    @property
    def filename(self) -> pathlib.Path:
        return pathlib.Path(self.lineEdit_16.text())

    @property
    def filetype(self) -> str:
        return self.comboBox.currentText()

    def read_tttr(self):
        fn = self.filename.absolute().as_posix()
        tttr = tttrlib.TTTR(fn, self.filetype)
        header = tttr.get_header()
        s = header.json
        d = json.loads(s)
        self.settings['header'] = d
        self.settings['macro_time_resolution'] = header.macro_time_resolution
        self.settings['micro_time_resolution'] = header.micro_time_resolution
        self.settings['micro_time_binning'] = int(self.comboBox_2.currentText())
        self.updateUI()

    def update_parameter(self):
        self.settings['macro_time_resolution'] = float(self.lineEdit_14.text()) / 1e9
        self.settings['micro_time_resolution'] = float(self.lineEdit_15.text()) / 1e12
        self.settings['micro_time_binning'] = int(self.comboBox_2.currentText())

    def updateUI(self):
        self.lineEdit_14.setText("{:.2f}".format(self.settings['macro_time_resolution'] * 1e9))
        self.lineEdit_15.setText("{:.2f}".format(self.settings['micro_time_resolution'] * 1e12))
        self.plainTextEdit.setPlainText(
            json.dumps(self.settings['header'], indent=2)
        )

    @chisurf.gui.decorators.init_with_ui("tttr_file_format.ui")
    def __init__(self, *args, **kwargs):
        self.setTitle("File format and setup")
        self.settings: FileFormatSettings = dict()
        # self.splitter.setStretchFactor(0, 2)
        # self.splitter.setStretchFactor(1, 3)
        self.textEdit_3.setVisible(False)
        chisurf.gui.decorators.lineEdit_dragFile_injector(
            self.lineEdit_16,
            call=self.read_tttr
        )
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sizePolicy)

        self.actionUpdate_Values.triggered.connect(self.update_parameter)


