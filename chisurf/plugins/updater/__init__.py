"""
ChiSurf Update Plugin

This plugin provides functionality to check for and install updates for ChiSurf.
It supports updating on Windows, macOS, and Linux, and handles elevated
privileges when needed.

Features:
- Check for available updates
- Download and install updates using conda
- Handle platform-specific update logic
- Inform the user to restart the application manually after updating

Note:
The update process will close all ChiSurf windows and continue in a separate window.
After the update completes, the user will need to restart ChiSurf manually.

The update URL is configured in the settings or defaults to the one specified in info.py.
"""

import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QProgressDialog, QApplication, QComboBox, QTextEdit,
    QMessageBox, QRadioButton, QLineEdit, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .updater import ChiSurfUpdater, check_for_updates, update_chisurf
from chisurf import info

# Define the plugin name - this will appear in the Plugins menu
name = "Help:Check for Updates"

class UpdaterWidget(QWidget):
    """
    A widget that provides a UI for checking for and installing updates.

    The update URL is configured in the settings or defaults to the one specified in info.py.
    """

    def __init__(self, parent=None):
        """Initialize the updater widget."""
        super().__init__(parent)
        self.setWindowTitle("ChiSurf Updater")
        self.available_versions = []

        # Import settings
        from chisurf.settings import cs_settings
        self.cs_settings = cs_settings

        # Get update URL from settings or fall back to the one from info.py
        hardcoded_url = "https://www.peulen.xyz/downloads/chisurf/conda"
        update_url = cs_settings.get('update_url', hardcoded_url)

        # Initialize updater with the update URL
        self.updater = ChiSurfUpdater(update_url=update_url)

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

        # Version dropdown
        version_layout = QHBoxLayout()
        version_layout.addWidget(QLabel("Available Versions:"))
        self.version_dropdown = QComboBox()
        self.version_dropdown.setEnabled(False)  # Disabled until versions are available
        version_layout.addWidget(self.version_dropdown)
        layout.addLayout(version_layout)

        # Command display text widget
        layout.addWidget(QLabel("Conda Commands:"))
        self.command_display = QTextEdit()
        self.command_display.setReadOnly(True)
        self.command_display.setVisible(False)  # Hidden until commands are available

        # Set monospaced font
        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        self.command_display.setFont(font)

        # Set minimum height
        self.command_display.setMinimumHeight(100)
        layout.addWidget(self.command_display)

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
        self.version_dropdown.setEnabled(False)
        self.version_dropdown.clear()

        # Get update information directly from the updater
        update_info = self.updater._get_update_info()

        if not update_info:
            self.status_label.setText("Error checking for updates: No update information available")
            self.check_button.setEnabled(True)
            return

        # Check if there are available versions
        self.available_versions = update_info.get("available_versions", [])

        if not self.available_versions and self.updater._is_local_folder():
            # If the update URL is a local folder but no versions were found,
            # try to get them directly
            self.available_versions = self.updater._list_available_versions()

        # If we have available versions, populate the dropdown
        if self.available_versions:
            for version_info in self.available_versions:
                self.version_dropdown.addItem(
                    f"Version {version_info['version']}",
                    version_info
                )

            self.version_dropdown.setEnabled(True)
            self.update_button.setEnabled(True)
            self.status_label.setText(f"Found {len(self.available_versions)} available versions.")
        else:
            # Fall back to the standard update check
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
        """Update ChiSurf to the selected version."""
        # Get the selected version
        selected_index = self.version_dropdown.currentIndex()

        # Clear and show the command display
        self.command_display.clear()
        self.command_display.setVisible(True)

        # Create progress dialog
        progress_dialog = QProgressDialog("Updating ChiSurf...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Updating")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.show()

        # Define callback to update progress dialog and command display
        def update_callback(message):
            progress_dialog.setLabelText(message)

            # If the message is a command, display it in the command display
            if message.startswith("Command:"):
                self.command_display.append(message)
                self.command_display.append("")  # Add a blank line for readability
            elif message.startswith("WARNING:"):
                # For warnings, make them stand out
                self.command_display.append("")
                self.command_display.append(message)
                self.command_display.append("")
            else:
                # For other messages, add them to both the progress dialog and command display
                self.command_display.append(message)

            QApplication.processEvents()

        # Show a warning message before starting the update
        warning_result = QMessageBox.warning(
            self,
            "Update Warning",
            "The update process will close all ChiSurf windows and continue in a separate window.\n\n"
            "All unsaved work will be lost. After the update completes, you will need to restart ChiSurf manually.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if warning_result != QMessageBox.Yes:
            # User cancelled the update
            progress_dialog.close()
            self.status_label.setText("Update cancelled by user.")
            return

        # If we have available versions and one is selected, use it
        if self.available_versions and selected_index >= 0:
            selected_version = self.version_dropdown.itemData(selected_index)

            if selected_version:
                # Update the status
                self.status_label.setText(f"Updating to version {selected_version['version']}...")
                update_callback(f"Updating to version {selected_version['version']}...")

                # Perform the update using the selected version
                # Note: auto_restart is ignored as the application will be closed
                self.updater.update_to_version(
                    selected_version['file_path'], 
                    callback=update_callback,
                    auto_restart=False
                )

                # The application will exit during the update process, so this code won't be reached
                return

        # If no version is selected or available, fall back to the standard update
        # Note: auto_restart is ignored as the application will be closed
        update_chisurf(callback=update_callback, auto_restart=False)

        # The application will exit during the update process, so this code won't be reached

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the UpdaterWidget class
    window = UpdaterWidget()
    # Show the window
    window.show()
