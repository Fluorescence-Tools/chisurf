import sys
import json
import os.path
import pathlib
import typing
import numpy as np

from chisurf.gui import QtWidgets, QtGui, QtCore

name = "Browser"



from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *

import sys

class Browser(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(QtWidgets.QWidget,self).__init__(*args, **kwargs)

        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl("https://www.peulen.xyz/software/chisurf/"))

        self.setCentralWidget(self.browser)

        self.show()


if __name__ == "plugin":
    wizard = Browser()
    wizard.show()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = Browser()
    wizard.show()
    sys.exit(app.exec_())

