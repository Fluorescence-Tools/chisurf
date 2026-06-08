from qtpy import QtCore, QtWidgets

import chisurf as cs
from .pages.welcome import WelcomePage
from .pages.settings_status import SettingsStatusPage
from .pages.repair import RepairPage
from .pages.dependencies import DependenciesPage
from .pages.detector_setups import DetectorSetupsPage
from .pages.fcs_channels import FCSChannelsPage
from .pages.finish import FinishPage


class WelcomeToChiSurfWizard(QtWidgets.QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Welcome to ChiSurf")
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)

        self.addPage(WelcomePage())
        self.addPage(SettingsStatusPage())
        self.addPage(RepairPage())
        self.addPage(DependenciesPage())
        self.addPage(DetectorSetupsPage())
        self.addPage(FCSChannelsPage())
        self.addPage(FinishPage())

        try:
            self.setOption(QtWidgets.QWizard.NoBackButtonOnStartPage, True)
        except Exception:
            pass


def show_onboarding(parent=None):
    wiz = WelcomeToChiSurfWizard(parent=parent)
    wiz.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
    try:
        wiz.setWindowModality(QtCore.Qt.ApplicationModal)
    except Exception:
        pass
    wiz.show()
    try:
        wiz.raise_()
        wiz.activateWindow()
    except Exception:
        pass
    cs.__init_chisurf_wizard__ = wiz
    return wiz


def _main():
    try:
        parent = getattr(cs, 'cs', None)
    except Exception:
        parent = None

    show_onboarding(parent=parent)


if __name__ in {"__main__", "plugin"}:
    _main()
