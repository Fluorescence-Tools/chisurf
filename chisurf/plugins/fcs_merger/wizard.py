import sys
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



class ChisurfWizard(QtWidgets.QWizard):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)

        # File format
        page = chisurf.gui.widgets.wizard.WizardFcsMerger()
        self.addPage(page)


if __name__ == "plugin":
    wizard = ChisurfWizard()
    wizard.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = ChisurfWizard()
    wizard.show()
    sys.exit(app.exec_())

