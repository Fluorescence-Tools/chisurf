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

        self.channels = chisurf.gui.widgets.wizard.DetectorWizardPage()
        self.addPage(self.channels)

        self.filter = chisurf.gui.widgets.wizard.WizardTTTRPhotonFilter(
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

