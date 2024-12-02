import json
import os.path
import pathlib
import typing
import numpy as np

from chisurf.gui import QtWidgets, QtGui, QtCore

import chisurf.gui
import chisurf.gui.widgets.wizard
import chisurf.gui.widgets
import chisurf.gui.decorators
import chisurf.gui.tools
import chisurf.gui.tools.parameter_editor

import chisurf.data
import chisurf.experiments
import chisurf.curve
import chisurf.fitting

import chisurf.macros


name = "Burst Selection"


class ChisurfWizard(QtWidgets.QWizard):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)

        # File format
        page = chisurf.gui.widgets.wizard.WizardTTTRFileFormat()
        self.addPage(page)

        # Analysis channel definition
        page = chisurf.gui.widgets.wizard.WizardTTTRChannelDefinition()
        self.addPage(page)


if __name__ == '__main__':
    wizard = ChisurfWizard()
    wizard.show()
