"""
ChiSurf About Plugin

This plugin provides information about ChiSurf, including version, developer,
and contact information.

Features:
- Display ChiSurf logo
- Show version information
- Display developer contact details
"""

import sys
import pathlib
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5 import uic

from chisurf import info

# Define the plugin name - this will appear in the Plugins menu
name = "Help:About ChiSurf"

class AboutDialog(QDialog):
    """
    A dialog that displays information about ChiSurf.
    """

    def __init__(self, parent=None):
        """Initialize the about dialog."""
        super().__init__(parent)
        self.setWindowTitle("About ChiSurf")
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        # Set dialog size
        self.setMinimumSize(280, 370)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create text edit for content
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFrameShape(QTextEdit.NoFrame)
        self.text_edit.setLineWidth(0)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_edit.setTextInteractionFlags(Qt.NoTextInteraction)
        self.text_edit.setMinimumSize(280, 340)

        # Set HTML content
        html_content = f"""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html><head><meta name="qrichtext" content="1" /><style type="text/css">
p, li {{ white-space: pre-wrap; }}
</style></head><body style=" font-family:'.SF NS Text'; font-size:13pt; font-weight:400; font-style:normal;">
<p style=" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-family:'Arial'; font-size:14pt; font-weight:600; color:#ff5500;">ChiSurf </span></p>
<p align="center" style=" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><img src=":/icons/icons/cs_logo.png" /></p>
<p style=" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-family:'MS Shell Dlg 2'; font-size:8pt;">Version: {info.__version__}<br />Development: Thomas-Otavio Peulen <br />Email: thomas.otavio.peulen@gmail.com</span></p></body></html>"""
        self.text_edit.setHtml(html_content)

        # Add text edit to layout
        layout.addWidget(self.text_edit)

        # Create close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.hide)
        layout.addWidget(close_button)

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the AboutDialog class
    window = AboutDialog()
    # Show the window
    window.show()
