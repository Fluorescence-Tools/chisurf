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

class WizardTTTRChannelDefinition(QtWidgets.QWizardPage):

    @chisurf.gui.decorators.init_with_ui("tttr_channel_definition.ui")
    def __init__(self, *args, **kwargs):
        self.setTitle("Analysis channel definition")
        self.settings: FileFormatSettings = dict()
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)

        # chisurf.gui.decorators.lineEdit_dragFile_injector(
        #     self.lineEdit_16,
        #     call=self.read_tttr
        # )
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(sizePolicy)

        # self.actionUpdate_Values.triggered.connect(self.update_parameter)


