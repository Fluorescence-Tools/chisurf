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

        self.channels = chisurf.gui.widgets.wizard.DetectorWizardPage()
        self.addPage(self.channels)

        self.filter = chisurf.gui.widgets.wizard.WizardTTTRBurstFinder(
            windows=self.channels.windows,
            detectors=self.channels.detectors
        )

        self.addPage(self.filter)
        self.button(QtWidgets.QWizard.FinishButton).clicked.connect(self.onFinish)

    def onFinish(self):
        print("Saving photon selection")
        self.filter.save_selection()


if __name__ == "plugin":
    wizard = ChisurfWizard()
    wizard.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = ChisurfWizard()
    wizard.show()
    sys.exit(app.exec_())

