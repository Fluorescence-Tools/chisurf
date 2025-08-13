"""
Plugin entry point for TTTR Image Browser.
This module allows loading via a plugin manager that imports ...tttr_image_browser.__plugin__.
"""
from chisurf.plugins.tttr_image_browser.__init__ import TTTRImageBrowser

# When imported as a plugin module, create and show the window
try:
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    window = TTTRImageBrowser()
    window.show()
except Exception:
    # Silently ignore import-time GUI failures in plugin discovery context
    pass
