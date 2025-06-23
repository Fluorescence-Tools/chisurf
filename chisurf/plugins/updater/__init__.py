"""
ChiSurf Update Plugin

This plugin provides functionality to check for and install updates for ChiSurf.
It supports updating on Windows, macOS, and Linux, and handles elevated
privileges when needed.

Features:
- Check for available updates
- Download and install updates using conda
- Handle platform-specific update logic
- Restart the application after updating
"""

import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QProgressDialog, QApplication
from PyQt5.QtCore import Qt

from chisurf.settings.updater import ChiSurfUpdater, check_for_updates, update_chisurf
from chisurf import info

# Define the plugin name - this will appear in the Plugins menu
name = "Help:Check for Updates"

class UpdaterWidget(QWidget):
    """
    A widget that provides a UI for checking for and installing updates.
    """

    def __init__(self, parent=None):
        """Initialize the updater widget."""
        super().__init__(parent)
        self.setWindowTitle("ChiSurf Updater")
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Current version info
        version_layout = QHBoxLayout()
        version_layout.addWidget(QLabel("Current Version:"))
        version_layout.addWidget(QLabel(info.__version__))
        version_layout.addStretch()
        layout.addLayout(version_layout)

        # Status label
        self.status_label = QLabel("Click 'Check for Updates' to check for available updates.")
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        self.check_button = QPushButton("Check for Updates")
        self.check_button.clicked.connect(self.check_for_updates)
        button_layout.addWidget(self.check_button)

        self.update_button = QPushButton("Update Now")
        self.update_button.clicked.connect(self.update_chisurf)
        self.update_button.setEnabled(False)  # Disabled until updates are available
        button_layout.addWidget(self.update_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def check_for_updates(self):
        """Check for available updates."""
        self.status_label.setText("Checking for updates...")
        self.check_button.setEnabled(False)
        self.update_button.setEnabled(False)

        # Check for updates
        update_available, latest_version, error = check_for_updates()

        if error:
            self.status_label.setText(f"Error checking for updates: {error}")
        elif update_available:
            self.status_label.setText(f"Update available: version {latest_version}")
            self.update_button.setEnabled(True)
        else:
            self.status_label.setText(f"ChiSurf is already up to date (version {info.__version__}).")

        self.check_button.setEnabled(True)

    def update_chisurf(self):
        """Update ChiSurf to the latest version."""
        # Create progress dialog
        progress_dialog = QProgressDialog("Updating ChiSurf...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Updating")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.show()

        # Define callback to update progress dialog
        def update_callback(message):
            progress_dialog.setLabelText(message)
            QApplication.processEvents()

        # Perform update
        success, error = update_chisurf(callback=update_callback)

        # Close progress dialog
        progress_dialog.close()

        if not success:
            self.status_label.setText(f"Failed to update ChiSurf: {error}")
        # If successful, the application will restart automatically

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the UpdaterWidget class
    window = UpdaterWidget()
    # Show the window
    window.show()
