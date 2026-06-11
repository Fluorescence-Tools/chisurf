import sys

import chisurf as cs
import urllib

from qtpy import QtCore, QtWidgets, QtGui

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c
from qtpy.QtWebEngineCore import QWebEngineUrlRequestInterceptor
from qtpy.QtWebEngineWidgets import (
    QWebEnginePage as QWebPage,
    QWebEngineProfile,
    QWebEngineView as QWebView,
)

log = cs.logging.info

SETTING_GEOMETRY = "net.fishandwhistle/JupyterQt/geometry"


class LoggerDock(QtWidgets.QDockWidget):

    def __init__(self, *args):
        super(LoggerDock, self).__init__(*args)
        self.textview = QtWidgets.QPlainTextEdit(self)
        self.textview.setReadOnly(True)
        self.setWidget(self.textview)

    @QtCore.Slot(str)
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

    @QtCore.Slot(bool)
    def onpagechange(self, ok):
        self.loadedPage = self.page()
        interceptor = MyUrlRequestInterceptor()
        profile = QWebEngineProfile.defaultProfile()
        profile.setUrlRequestInterceptor(interceptor)
        self.loadedPage.windowCloseRequested.connect(self.close)
        self.loadedPage.urlChanged.connect(self.handlelink)
        self.setWindowTitle(self.title())
        if not ok:
            QtWidgets.QMessageBox.information(self, "Error", "Error loading page!", QtWidgets.QMessageBox.Ok)

    @QtCore.Slot(QtCore.QUrl)
    def handlelink(self, url):
        urlstr = url.toString()
        log("handling link : %s" % urlstr)
        # check if url is for the current page
        if url.matches(self.url(), QtCore.QUrl.RemoveFragment):
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


@persist_plugin_state("browser")
class Browser(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        # adr is in global

        url = kwargs.pop('url', globals().get('adr', "https://github.com/fluorescence-tools/cs"))  # Default fallback
        if isinstance(url, str):  # Ensure it's a string
            url = urllib.parse.quote(url, safe=":/?&=")
        else:
            raise ValueError("Invalid URL: Expected a string, got {}".format(type(url)))

        super().__init__(*args, **kwargs)

        self.browser = CustomWebView(self)

        # Original URL
        self.setWindowTitle(url)
        self.original_url = QtCore.QUrl(url)

        self.url_bar = QtWidgets.QLineEdit()
        self.url_bar.setText(self.original_url.toString())
        self.url_bar.returnPressed.connect(self.navigate_to_url)

        self.go_button = QtWidgets.QPushButton('Go')
        self.go_button.clicked.connect(self.navigate_to_url)

        self.reload_button = QtWidgets.QPushButton('Reload')
        self.reload_button.clicked.connect(self.reload_page)

        self.home_button = QtWidgets.QPushButton('Home')
        self.home_button.clicked.connect(self.go_home)

        self.toolbar = QtWidgets.QHBoxLayout()
        self.toolbar.addWidget(self.url_bar)
        self.toolbar.addWidget(self.go_button)
        self.toolbar.addWidget(self.reload_button)
        self.toolbar.addWidget(self.home_button)

        self.browser_layout = QtWidgets.QVBoxLayout()
        self.browser_layout.addLayout(self.toolbar)
        self.browser_layout.addWidget(self.browser)
        self.browser_layout.setSpacing(0)  # Set spacing for main layout to 0
        self.browser_layout.setContentsMargins(0, 0, 0, 0)  # Set margins for the layout to 0

        self.central_widget = QtWidgets.QWidget()
        self.central_widget.setLayout(self.browser_layout)
        self.setCentralWidget(self.central_widget)

        self.browser.setUrl(self.original_url)

        # Connect signal to slot
        self.browser.urlChanged.connect(self.update_url_bar)

        self.show()

    def navigate_to_url(self):
        url = self.url_bar.text()
        self.browser.handlelink(QtCore.QUrl(url))

    def reload_page(self):
        self.browser.reload()

    def update_url_bar(self, url):
        self.url_bar.setText(url.toString())

    def go_home(self):
        self.browser.setUrl(self.original_url)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    wizard = Browser()
    wizard.show()
    sys.exit(app.exec())

if __name__ == "plugin":
    wizard = Browser()
    wizard.show()
