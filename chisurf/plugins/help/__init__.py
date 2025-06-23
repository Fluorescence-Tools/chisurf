"""
ChiSurf Help Plugin

This plugin provides access to the ChiSurf documentation and help resources.

Features:
- Open the ChiSurf documentation in a web browser
- Access to video tutorials
- Access to user guides and tutorials
- Quick reference for common tasks
"""

import webbrowser
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextBrowser
from PyQt5.QtCore import Qt, QUrl

from chisurf import info

# Define the plugin name - this will appear in the Plugins menu
name = "Help:Documentation"

class HelpWidget(QWidget):
    """
    A widget that provides access to ChiSurf documentation and help resources.
    """

    def __init__(self, parent=None):
        """Initialize the help widget."""
        super().__init__(parent)
        self.setWindowTitle("ChiSurf Help")
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Help description
        description = QLabel("ChiSurf Documentation and Help Resources")
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)

        # Help options
        help_layout = QVBoxLayout()

        # Documentation button
        doc_button = QPushButton("Open Documentation")
        doc_button.clicked.connect(self.open_documentation)
        help_layout.addWidget(doc_button)

        # Video tutorials button
        tutorials_button = QPushButton("Video Tutorials")
        tutorials_button.clicked.connect(self.open_video_tutorials)
        help_layout.addWidget(tutorials_button)

        layout.addLayout(help_layout)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.hide)
        layout.addWidget(close_button)

        self.setLayout(layout)

    def open_documentation(self):
        """Open the ChiSurf documentation in a web browser."""
        webbrowser.open_new(info.help_url)

    def open_video_tutorials(self):
        """Open the ChiSurf video tutorials in a web browser."""
        webbrowser.open_new("https://www.peulen.xyz/tutorial/")

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the HelpWidget class
    window = HelpWidget()
    # Show the window
    window.show()
