from qtpy import QtCore, QtWidgets

import chisurf as cs
from ..utils import build_status_html, settings_paths


class SettingsStatusPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Settings")
        self.setSubTitle("Review your user settings files and open the editor when needed.")

        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "ChiSurf stores most configuration in your user settings folder (<code>~/.cs</code>). "
            "If something looks wrong below, open the settings editor to adjust paths/options, then come back and click Refresh."
        )
        info.setWordWrap(True)

        self.browser = QtWidgets.QTextBrowser()
        self.browser.setOpenExternalLinks(False)

        btns = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)
        self.open_settings_editor_btn = QtWidgets.QPushButton("Open settings editor")
        self.open_settings_editor_btn.clicked.connect(self._open_settings_editor)
        btns.addWidget(self.refresh_btn)
        btns.addWidget(self.open_settings_editor_btn)
        btns.addStretch(1)

        layout.addWidget(info)
        layout.addWidget(self.browser)
        layout.addLayout(btns)

    def initializePage(self):
        self._refresh()

    def _refresh(self):
        self.browser.setHtml(build_status_html())

    def _open_settings_editor(self):
        try:
            from chisurf.gui.widgets.settings_editor import SettingsEditor
            parent = None
            try:
                parent = self.wizard()
            except Exception:
                parent = None

            dialog = QtWidgets.QDialog(parent)
            dialog.setWindowTitle("ChiSurf Settings")
            try:
                dialog.setWindowModality(QtCore.Qt.WindowModal)
            except Exception:
                pass
            try:
                dialog.setModal(True)
            except Exception:
                pass
            dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

            layout = QtWidgets.QVBoxLayout(dialog)
            editor = SettingsEditor(
                filename=str(settings_paths()['settings_chisurf_yaml']),
                window_title="ChiSurf Settings"
            )
            layout.addWidget(editor)

            try:
                dialog.resize(1000, 700)
            except Exception:
                pass
            try:
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
            except Exception:
                pass

            cs.__init_chisurf_settings_editor__ = dialog
        except Exception:
            pass
