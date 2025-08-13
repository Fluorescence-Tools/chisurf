"""
Entry point for running the TTTR Image Browser directly.

Usage:
  python -m chisurf.plugins.tttr_image_browser
"""
import sys
from PyQt5.QtWidgets import QApplication
from chisurf.plugins.tttr_image_browser.__init__ import TTTRImageBrowser


def main():
    app = QApplication(sys.argv)
    w = TTTRImageBrowser()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
