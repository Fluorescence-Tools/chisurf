import sys
from chisurf.gui import QtWidgets

import chisurf.gui
import chisurf.gui.widgets.wizard
import chisurf.gui.widgets
import chisurf.gui.decorators
import chisurf.gui.widgets.parameter_editor

import chisurf.core.data
import chisurf.core.experiments
import chisurf.core.curve
import chisurf.core.fitting

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

