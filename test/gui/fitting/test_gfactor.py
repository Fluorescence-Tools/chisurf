from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage


def test_detector_wizard_page_creation(qtbot):
    wizard_page = DetectorWizardPage()
    qtbot.addWidget(wizard_page)


def test_add_detector_rows(qtbot):
    wizard_page = DetectorWizardPage()
    qtbot.addWidget(wizard_page)

    wizard_page._add_detector_row("Detector1", "0, 1", "0-2048", "1.00", "0.00", "0.00")
    wizard_page._add_detector_row("Detector2", "2, 3", "0-2048", "1.00", "0.00", "0.00")
