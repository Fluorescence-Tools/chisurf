"""
ChiSurf About Plugin

This plugin provides information about ChiSurf, including version, developer,
and contact information.

Features:
- Display ChiSurf logo
- Show version information
- Display developer contact details
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QGridLayout, QPushButton, QTextEdit, QSizePolicy, QLayout
from qtpy.QtWidgets import QFrame

# Define the plugin name - this will appear in the Plugins menu
name = "Help:About ChiSurf"


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        try:
            import chisurf.gui.resources  # noqa: F401
        except Exception:
            pass

        self.setWindowTitle("About")
        self.setMinimumSize(280, 370)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        layout = QGridLayout(self)
        layout.setSizeConstraint(QLayout.SetFixedSize)
        layout.setContentsMargins(0, 0, 0, 0)

        self.textEdit = QTextEdit(self)
        self.textEdit.setEnabled(True)
        self.textEdit.setMinimumSize(280, 340)
        self.textEdit.setFrameShape(QFrame.NoFrame)
        self.textEdit.setLineWidth(0)
        self.textEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textEdit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textEdit.setReadOnly(True)
        self.textEdit.setTextInteractionFlags(Qt.NoTextInteraction)

        self.textEdit.setHtml(
            """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html><head><meta name="qrichtext" content="1" /><style type="text/css">
p, li { white-space: pre-wrap; }
</style></head><body style=" font-family:'.SF NS Text'; font-size:13pt; font-weight:400; font-style:normal;">
<p style=" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-family:'Arial'; font-size:14pt; font-weight:600; color:#ff5500;">ChiSurf </span></p>
<p align="center" style=" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><img src=":/icons/icons/cs_logo.png" /></p>
<p style=" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style=" font-family:'MS Shell Dlg 2'; font-size:8pt;">Development: Thomas-Otavio Peulen <br />Email: thomas.otavio.peulen@gmail.com</span></p></body></html>"""
        )

        self.toolButton = QPushButton(self)
        self.toolButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toolButton.setText("close")
        self.toolButton.clicked.connect(self.hide)

        layout.addWidget(self.textEdit, 0, 0)
        layout.addWidget(self.toolButton, 1, 0)

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the AboutDialog class
    window = AboutDialog()
    # Show the window
    window.show()
