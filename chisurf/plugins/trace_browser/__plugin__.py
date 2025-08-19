"""
Plugin entry point for Trace Browser.
This module allows loading via a plugin manager that imports ...trace_browser.__plugin__.
"""
from chisurf.plugins.trace_browser.__init__ import TraceBrowser

# When imported as a plugin module, create and show the window
try:
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    window = TraceBrowser()
    window.show()
except Exception:
    # Silently ignore import-time GUI failures in plugin discovery context
    pass
