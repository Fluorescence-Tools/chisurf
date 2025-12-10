"""
Entry point for running the Trace Browser directly.

Usage:
  python -m chisurf.plugins.trace_browser
"""
import sys
from qtpy.QtWidgets import QApplication
from chisurf.plugins.trace_browser.__init__ import TraceBrowser


def main():
    app = QApplication(sys.argv)
    w = TraceBrowser()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
