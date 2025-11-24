"""Entry point for running the Burst Browser directly.

Usage:
  python -m chisurf.plugins.burst_browser
"""
import sys
from PyQt5.QtWidgets import QApplication

from chisurf.plugins.burst_browser import BurstBrowserWidget


def main():
    app = QApplication(sys.argv)
    w = BurstBrowserWidget()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
