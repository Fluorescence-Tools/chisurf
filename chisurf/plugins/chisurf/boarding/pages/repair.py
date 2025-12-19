from qtpy import QtWidgets

from ..utils import copy_defaults


class RepairPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Fix / Initialize")
        self.setSubTitle("Create missing settings files or restore packaged defaults.")

        layout = QtWidgets.QVBoxLayout(self)

        self.info = QtWidgets.QLabel(
            "Use this page if ChiSurf reports missing settings, or if you want to reset to a known-good baseline.\n\n"
            "Create missing files: only writes files that do not exist yet.\n"
            "Restore defaults (overwrite): replaces your current user settings files with packaged defaults."
        )
        self.info.setWordWrap(True)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)

        btns = QtWidgets.QHBoxLayout()
        self.create_missing_btn = QtWidgets.QPushButton("Create missing files")
        self.create_missing_btn.clicked.connect(self._create_missing)
        self.overwrite_btn = QtWidgets.QPushButton("Restore defaults (overwrite)")
        self.overwrite_btn.clicked.connect(self._overwrite)
        btns.addWidget(self.create_missing_btn)
        btns.addWidget(self.overwrite_btn)
        btns.addStretch(1)

        layout.addWidget(self.info)
        layout.addLayout(btns)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def _set_status(self, ok: bool, msg: str):
        color = "#2e7d32" if ok else "#c62828"
        self.status.setText(f"<span style='color:{color}; font-weight:600'>{msg}</span>")

    def _create_missing(self):
        ok, msg = copy_defaults(overwrite=False)
        self._set_status(ok, msg)

    def _overwrite(self):
        reply = QtWidgets.QMessageBox.question(
            self,
            "Restore defaults",
            "This will overwrite your current user settings files. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        ok, msg = copy_defaults(overwrite=True)
        self._set_status(ok, msg)
