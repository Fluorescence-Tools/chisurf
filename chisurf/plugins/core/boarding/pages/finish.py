from qtpy import QtWidgets

import chisurf as cs
class FinishPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Done")
        self.setSubTitle("You are ready to start working.")

        layout = QtWidgets.QVBoxLayout(self)

        text = QtWidgets.QLabel(
            "Recommended next steps:\n"
            "1) Open Settings and verify paths/options (working directory, plugins, GUI preferences).\n"
            "2) For TTTR data: define detector setups (PIE/windows) once per hardware configuration.\n"
            "3) For burst-wise FCS: define at least one FCS channel preset.\n"
            "4) Use Help to browse experiment-specific guides and built-in documentation.\n\n"
            "You can close this wizard now; it does not need to stay open."
        )
        text.setWordWrap(True)

        btns = QtWidgets.QHBoxLayout()
        self.open_help_btn = QtWidgets.QPushButton("Open Help")
        self.open_help_btn.clicked.connect(self._open_help)
        self.open_plugins_btn = QtWidgets.QPushButton("Open Plugin Manager")
        self.open_plugins_btn.clicked.connect(self._open_plugin_manager)
        btns.addWidget(self.open_help_btn)
        btns.addWidget(self.open_plugins_btn)
        btns.addStretch(1)

        layout.addWidget(text)
        layout.addLayout(btns)
        layout.addStretch(1)

    def _open_help(self):
        try:
            gui = getattr(cs, 'cs', None)
            if gui is not None and hasattr(gui, 'onOpenHelp'):
                gui.onOpenHelp()
        except Exception:
            pass

    def _open_plugin_manager(self):
        try:
            import importlib
            pm = importlib.import_module('chisurf.plugins.core.plugin_manager')
            w = pm.PluginManagerWidget()
            w.show()
            cs.__init_chisurf_plugin_manager__ = w
        except Exception:
            pass
