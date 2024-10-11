import sys
import json
import os.path
import pathlib
import typing
import numpy as np
import chisurf

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *

name = "Notebooks"


class Browser(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.browser = QWebEngineView()

        # DROP SHADOW in PyDracula
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(17)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 150))

        self.browser.setGraphicsEffect(shadow)
        self.browser.graphicsEffect().setEnabled(False)

        # Original URL
        #adr = str(chisurf.__jupyter_address__ + '/notebooks') + fn
        #print(adr)
        self.original_url = QUrl(adr)

        self.browser_layout = QVBoxLayout()
        self.browser_layout.addWidget(self.browser)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.browser_layout)
        self.setCentralWidget(self.central_widget)

        self.browser.setUrl(self.original_url)

        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wizard = Browser()
    wizard.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    wizard = Browser()
    wizard.show()
