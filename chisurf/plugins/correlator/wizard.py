import sys
import pathlib

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

    def page_actions(self):
        if self.currentPage().title() == "Correlator":
            self.photon_select.save_filter_data()
            p = self.photon_select.parent_directories[0]
            self.correlator_page.lineEdit_3.setText(p.as_posix())
            self.correlator_page.open_analysis_folder()
        elif self.currentPage().title() == "Correlation merging":
            correlation_folder = self.correlator_page.analysis_folder / self.correlator_page.output_path
            self.fcs_merger.lineEdit.setText(correlation_folder.as_posix())
            self.fcs_merger.open_correlation_folder()

    def onFinish(self):
        print("Correlation Wizard Finished")
        print("saving merged correlation")
        self.fcs_merger.save_mean_correlation()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)

        self.photon_select = chisurf.gui.widgets.wizard.WizardTTTRPhotonFilter()
        self.photon_select.toolButton_5.clicked.connect(self.photon_select.completeChanged.emit)

        self.correlator_page = chisurf.gui.widgets.wizard.WizardTTTRCorrelator()
        self.addPage(self.photon_select)
        self.addPage(self.correlator_page)

        self.fcs_merger = chisurf.gui.widgets.wizard.WizardFcsMerger()
        self.addPage(self.fcs_merger)

        self.button(QtWidgets.QWizard.NextButton).clicked.connect(self.page_actions)
        self.button(QtWidgets.QWizard.FinishButton).clicked.connect(self.onFinish)



if __name__ == "plugin":
    wizard = ChisurfWizard()
    wizard.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = ChisurfWizard()
    wizard.show()
    sys.exit(app.exec_())

