from qtpy import QtWidgets

from ..utils import build_deps_html


class DependenciesPage(QtWidgets.QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Dependencies")
        self.setSubTitle("Check optional Python packages used by specific features.")

        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "ChiSurf can use optional packages for certain workflows. Missing packages usually do not prevent startup, "
            "but specific tools/plugins may be unavailable."
        )
        info.setWordWrap(True)

        self.browser = QtWidgets.QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        layout.addWidget(info)
        layout.addWidget(self.browser)

        btns = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh)
        btns.addWidget(self.refresh_btn)
        btns.addStretch(1)
        layout.addLayout(btns)

    def initializePage(self):
        self._refresh()

    def _refresh(self):
        self.browser.setHtml(build_deps_html())
