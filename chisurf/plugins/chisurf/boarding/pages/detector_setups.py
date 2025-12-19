from qtpy import QtCore, QtGui, QtWidgets

from ..utils import open_in_file_manager, settings_paths


class DetectorSetupsPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Detector setups")
        self.setSubTitle("Configure PIE / channel windows for TTTR workflows.")

        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "TTTR experiments (TCSPC/PIE) often need a detector/channel definition. "
            "ChiSurf stores these in <code>detector_setups.json</code> in your user settings folder.\n\n"
            "Typical workflow: open the Detector Wizard, define a setup once for your hardware, and save it."
        )
        info.setWordWrap(True)

        self.label = QtWidgets.QLabel("")
        self.label.setWordWrap(True)

        btns = QtWidgets.QHBoxLayout()
        self.open_wizard_btn = QtWidgets.QPushButton("Open Detector Wizard")
        self.open_wizard_btn.clicked.connect(self._open_detector_wizard)
        self.open_file_btn = QtWidgets.QPushButton("Open detector_setups.json")
        self.open_file_btn.clicked.connect(self._open_file)
        btns.addWidget(self.open_wizard_btn)
        btns.addWidget(self.open_file_btn)
        btns.addStretch(1)

        layout.addWidget(info)
        layout.addWidget(self.label)
        layout.addLayout(btns)
        layout.addStretch(1)

    def initializePage(self):
        p = settings_paths()['detector_setups_json']
        if p.exists():
            self.label.setText(f"Found: <code>{QtCore.QDir.toNativeSeparators(str(p))}</code>")
        else:
            self.label.setText(
                "Detector setups file is missing.\n\n"
                "You can create it by opening the Detector Wizard and saving a setup."
            )

    def _open_detector_wizard(self):
        try:
            from chisurf.gui.widgets.wizard.tttr_channel_definition import DetectorWizard
            wiz = DetectorWizard()
            wiz.exec_()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Detector Wizard", f"Could not open Detector Wizard: {e}")

    def _open_file(self):
        p = settings_paths()['detector_setups_json']
        if not p.exists():
            QtWidgets.QMessageBox.information(self, "Detector setups", "detector_setups.json does not exist yet.")
            return
        try:
            url = QtCore.QUrl.fromLocalFile(str(p))
            ok = QtGui.QDesktopServices.openUrl(url)
            if ok:
                return
        except Exception:
            pass
        open_in_file_manager(p.parent)
