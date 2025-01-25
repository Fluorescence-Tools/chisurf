import sys
import json
import os.path
import pathlib
import typing
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *

name = "Browser"

class Browser(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.browser = QWebEngineView()

        # Original URL
        # html_path = pathlib.Path(__file__).parent.absolute() / "p5js_example.html"
        html_path = pathlib.Path(__file__).parent.absolute() / "pr_example.html"
        self.original_url = QUrl().fromLocalFile(html_path.absolute().as_posix())

        self.url_bar = QLineEdit()
        self.url_bar.setText(self.original_url.toString())
        self.url_bar.returnPressed.connect(self.navigate_to_url)

        self.go_button = QPushButton('Go')
        self.go_button.clicked.connect(self.navigate_to_url)

        self.reload_button = QPushButton('Reload')
        self.reload_button.clicked.connect(self.reload_page)

        self.home_button = QPushButton('Home')
        self.home_button.clicked.connect(self.go_home)

        self.toolbar = QHBoxLayout()
        self.toolbar.addWidget(self.url_bar)
        self.toolbar.addWidget(self.go_button)
        self.toolbar.addWidget(self.reload_button)
        self.toolbar.addWidget(self.home_button)

        self.browser_layout = QVBoxLayout()
        self.browser_layout.addLayout(self.toolbar)
        self.browser_layout.addWidget(self.browser)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.browser_layout)
        self.setCentralWidget(self.central_widget)

        self.browser.setUrl(self.original_url)

        # Connect signal to slot
        self.browser.urlChanged.connect(self.update_url_bar)

        self.show()

    def navigate_to_url(self):
        url = self.url_bar.text()
        self.browser.setUrl(QUrl(url))

    def reload_page(self):
        self.browser.reload()

    def update_url_bar(self, url):
        self.url_bar.setText(url.toString())

    def go_home(self):
        self.browser.setUrl(self.original_url)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    wizard = Browser()
    wizard.show()
    sys.exit(app.exec_())

if __name__ == "plugin":
    wizard = Browser()
    wizard.show()
