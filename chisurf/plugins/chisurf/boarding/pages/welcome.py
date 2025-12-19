from qtpy import QtWidgets

from ..utils import open_in_file_manager, settings_paths


class WelcomePage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Welcome")
        self.setSubTitle("A quick guided setup for a new ChiSurf installation.")

        layout = QtWidgets.QVBoxLayout(self)

        text = QtWidgets.QLabel(
            "This wizard helps you get ChiSurf ready for your data and workflows. "
            "It focuses on the user settings folder (<code>~/.chisurf</code>) and a few key configuration files.\n\n"
            "What you can do here:\n"
            "1) Check whether your settings files exist and look sane.\n"
            "2) Create or restore defaults if something is missing or broken.\n"
            "3) Configure TTTR detector setups and (optionally) FCS channel presets.\n\n"
            "You can run this wizard any time from <b>Plugins → Dev → Onboarding:Welcome to ChiSurf</b>."
        )
        text.setWordWrap(True)

        btns = QtWidgets.QHBoxLayout()
        self.open_settings_dir_btn = QtWidgets.QPushButton("Open settings folder")
        self.open_settings_dir_btn.clicked.connect(self._open_settings_dir)
        btns.addWidget(self.open_settings_dir_btn)
        btns.addStretch(1)

        layout.addWidget(text)
        layout.addLayout(btns)
        layout.addStretch(1)

    def _open_settings_dir(self):
        p = settings_paths()['user_settings_dir']
        open_in_file_manager(p)
