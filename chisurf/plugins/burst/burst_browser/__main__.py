"""Entry point for running the Burst Browser widget directly.

Usage:
  python -m chisurf.plugins.burst_browser
"""
import sys
from qtpy.QtWidgets import QApplication

from chisurf.plugins.burst_browser import BurstBrowserWidget


def main():
    app = QApplication(sys.argv)
    w = BurstBrowserWidget()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
