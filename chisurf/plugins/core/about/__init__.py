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

# Plugin icon - displayed in menus and ribbon
icon = "ℹ️"


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        try:
            import chisurf.gui.resources  # noqa: F401
        except Exception:
            pass

        self.setWindowTitle("About ChiSurf")
        self.setMinimumSize(320, 400)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)

        layout = QGridLayout(self)
        layout.setSizeConstraint(QLayout.SetFixedSize)
        layout.setContentsMargins(16, 16, 16, 16)

        self.textEdit = QTextEdit(self)
        self.textEdit.setEnabled(True)
        self.textEdit.setMinimumSize(280, 340)
        self.textEdit.setFrameShape(QFrame.NoFrame)
        self.textEdit.setLineWidth(0)
        self.textEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textEdit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textEdit.setReadOnly(True)
        self.textEdit.setTextInteractionFlags(Qt.NoTextInteraction)

        import pathlib
        import chisurf
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QPixmap

        logo = QPixmap(":/icons/icons/cs_logo.png")
        if logo.isNull():
            logo_path = pathlib.Path(chisurf.__file__).parent / "gui" / "resources" / "icons" / "cs_logo.png"
            logo = QPixmap(str(logo_path))

        if not logo.isNull():
            self.textEdit.document().addResource(
                1,  # QTextDocument.ImageResource
                QUrl(":/icons/icons/cs_logo.png"),
                logo
            )

        self.textEdit.setHtml(
            """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
<html><head><meta name="qrichtext" content="1" /><style type="text/css">
p, li { white-space: pre-wrap; }
</style></head><body style=" font-family:'.SF NS Text'; font-size:13pt; font-weight:400; font-style:normal; text-align:center;">
<p style=" margin-top:20px; margin-bottom:12px; text-align:center;"><img src=":/icons/icons/cs_logo.png" width="64" height="64" /></p>
<p style=" margin-top:12px; margin-bottom:4px;"><span style=" font-family:'Arial'; font-size:16pt; font-weight:600; color:#ff5500;">ChiSurf</span></p>
<p style=" margin-top:4px; margin-bottom:12px; font-size:11pt;">Fluorescence Data Analysis</p>
<p style=" margin-top:20px; margin-bottom:4px; font-size:10pt;"><strong>Version:</strong> """ + str(__import__('chisurf.core.info', fromlist=['__version__']).__version__) + """</p>
<p style=" margin-top:12px; margin-bottom:4px; font-size:10pt;">Development: Thomas-Otavio Peulen</p>
<p style=" margin-top:4px; margin-bottom:12px; font-size:9pt; color:#555555;">thomas.peulen@tu-dortmund.de</p>
<p style=" margin-top:20px; margin-bottom:4px; font-size:9pt; color:#888888;">Repository: github.com/fluorescence-tools/chisurf</p>
</body></html>"""
        )

        self.toolButton = QPushButton(self)
        self.toolButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toolButton.setText("Close")
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
