"""Setup Channel Definition GUI tool."""

from qtpy import QtWidgets
from chisurf.gui.widgets.wizard import DetectorWizardPage


class SetupChannelDefinitionWidget(QtWidgets.QWidget):
    """
    A settings widget wrapping the DetectorWizardPage.
    Provides detector channel and PIE time window definitions.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.page = DetectorWizardPage(None, self, show_save=False)
        layout.addWidget(self.page)
