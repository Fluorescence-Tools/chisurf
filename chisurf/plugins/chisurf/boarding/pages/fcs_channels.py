from qtpy import QtCore, QtGui, QtWidgets

from chisurf.fluorescence.fcs.channel_setups import FCS_CHANNEL_SETUPS_FILE

from ..utils import open_in_file_manager


class FCSChannelsPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("FCS channel setup")
        self.setSubTitle("Define which channels/pairs should be correlated for FCS workflows.")

        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "For burst-wise FCS and correlator tools, ChiSurf needs to know which detector channels (or channel pairs) "
            "belong to a given correlation setup. These presets are stored in <code>fcs_channel_setups.json</code>.\n\n"
            "Open the editor, create at least one setup, and save it."
        )
        info.setWordWrap(True)

        self.label = QtWidgets.QLabel("")
        self.label.setWordWrap(True)

        btns = QtWidgets.QHBoxLayout()
        self.open_editor_btn = QtWidgets.QPushButton("Open FCS Channel Definitions")
        self.open_editor_btn.clicked.connect(self._open_editor)
        self.open_file_btn = QtWidgets.QPushButton("Open fcs_channel_setups.json")
        self.open_file_btn.clicked.connect(self._open_file)
        btns.addWidget(self.open_editor_btn)
        btns.addWidget(self.open_file_btn)
        btns.addStretch(1)

        layout.addWidget(info)
        layout.addWidget(self.label)
        layout.addLayout(btns)
        layout.addStretch(1)

    def initializePage(self):
        p = FCS_CHANNEL_SETUPS_FILE
        if p.exists():
            self.label.setText(f"Found: <code>{QtCore.QDir.toNativeSeparators(str(p))}</code>")
        else:
            self.label.setText(
                "FCS channel presets are not configured yet.\n\n"
                "Open the FCS Channel Definitions dialog and save at least one setup."
            )

    def _open_editor(self):
        try:
            from chisurf.plugins.fcs.fcs_channel_preset import FCSChannelDialog
            dlg = FCSChannelDialog(self.wizard())
            dlg.setWindowModality(QtCore.Qt.NonModal)
            dlg.show()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "FCS channel setup", f"Could not open FCS channel editor: {e}")

    def _open_file(self):
        p = FCS_CHANNEL_SETUPS_FILE
        if not p.exists():
            QtWidgets.QMessageBox.information(self, "FCS channel setup", "fcs_channel_setups.json does not exist yet.")
            return
        try:
            url = QtCore.QUrl.fromLocalFile(str(p))
            ok = QtGui.QDesktopServices.openUrl(url)
            if ok:
                return
        except Exception:
            pass
        open_in_file_manager(p.parent)
