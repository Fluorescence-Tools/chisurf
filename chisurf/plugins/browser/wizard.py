import sys

import chisurf
import urllib

name = "Miscellaneous:Browser"

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *

from PyQt5.QtCore import pyqtSlot, QSettings, QTimer, QUrl, Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDockWidget, QPlainTextEdit, QTabWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView, QWebEnginePage as QWebPage
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInterceptor
from PyQt5.QtWebEngineWidgets import QWebEngineProfile

log = chisurf.logging.info

SETTING_GEOMETRY = "net.fishandwhistle/JupyterQt/geometry"


class LoggerDock(QDockWidget):

    def __init__(self, *args):
        super(LoggerDock, self).__init__(*args)
        self.textview = QPlainTextEdit(self)
        self.textview.setReadOnly(True)
        self.setWidget(self.textview)

    @pyqtSlot(str)
    def log(self, message):
        self.textview.appendPlainText(message)


class CustomWebView(QWebView):

    def __init__(self, mainwindow, main=False):
        super(CustomWebView, self).__init__(None)
        self.parent = mainwindow
        self.tabIndex = -1
        self.main = main
        self.loadedPage = None
        self.loadFinished.connect(self.onpagechange)

    @pyqtSlot(bool)
    def onpagechange(self, ok):
        self.loadedPage = self.page()
        interceptor = MyUrlRequestInterceptor()
        profile = QWebEngineProfile.defaultProfile()
        profile.setUrlRequestInterceptor(interceptor)
        self.loadedPage.windowCloseRequested.connect(self.close)
        self.loadedPage.urlChanged.connect(self.handlelink)
        self.setWindowTitle(self.title())
        if not ok:
            QMessageBox.information(self, "Error", "Error loading page!", QMessageBox.Ok)

    @pyqtSlot(QUrl)
    def handlelink(self, url):
        urlstr = url.toString()
        log("handling link : %s" % urlstr)
        # check if url is for the current page
        if url.matches(self.url(), QUrl.RemoveFragment):
            # do nothing, probably a JS link
            return True

        if "/files/" in urlstr:
            # save, don't load new page
            self.parent.savefile(url)
        else:
            self.load(url)

        return True

    def createWindow(self, windowtype):
        return self

    def closeEvent(self, event):
        if self.loadedPage is not None:
            log("disconnecting on close and linkClicked signals")
            self.loadedPage.windowCloseRequested.disconnect(self.close)


class MyUrlRequestInterceptor(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, info):
        url = info.requestUrl()
        if url.scheme() != 'file':
            info.block(True)


class Browser(QMainWindow):

    def __init__(self, *args, **kwargs):
        # adr is in global

        url = kwargs.pop('url', globals().get('adr', "https://github.com/fluorescence-tools/chisurf"))  # Default fallback
        if isinstance(url, str):  # Ensure it's a string
            url = urllib.parse.quote(url, safe=":/?&=")
        else:
            raise ValueError("Invalid URL: Expected a string, got {}".format(type(url)))

        super().__init__(*args, **kwargs)

        self.browser = CustomWebView(self)

        # Original URL
        self.setWindowTitle(url)
        self.original_url = QUrl(url)

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
        self.browser_layout.setSpacing(0)  # Set spacing for main layout to 0
        self.browser_layout.setContentsMargins(0, 0, 0, 0)  # Set margins for the layout to 0

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.browser_layout)
        self.setCentralWidget(self.central_widget)

        self.browser.setUrl(self.original_url)

        # Connect signal to slot
        self.browser.urlChanged.connect(self.update_url_bar)

        self.show()

    def navigate_to_url(self):
        url = self.url_bar.text()
        self.browser.handlelink(QUrl(url))

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
