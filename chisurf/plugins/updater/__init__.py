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
import logging
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
        logging.info("Checking for updates via UI")

        status_message = "Checking for updates..."
        self.status_label.setText(status_message)
        logging.info(status_message)

        self.check_button.setEnabled(False)
        self.update_button.setEnabled(False)
        self.version_dropdown.setEnabled(False)
        self.version_dropdown.clear()

        # Get update information directly from the updater
        logging.debug("Getting update information from updater")
        update_info = self.updater._get_update_info()

        if not update_info:
            error_message = "Error checking for updates: No update information available"
            logging.error(error_message)
            self.status_label.setText(error_message)
            self.check_button.setEnabled(True)
            return

        # Check if there are available versions
        self.available_versions = update_info.get("available_versions", [])
        logging.debug(f"Found {len(self.available_versions)} available versions in update info")

        if not self.available_versions and self.updater._is_local_folder():
            # If the update URL is a local folder but no versions were found,
            # try to get them directly
            logging.debug("No versions found in update info but using local folder, trying direct listing")
            self.available_versions = self.updater._list_available_versions()
            logging.debug(f"Found {len(self.available_versions)} available versions from direct listing")

        # If we have available versions, populate the dropdown
        if self.available_versions:
            logging.info(f"Found {len(self.available_versions)} available versions")
            for version_info in self.available_versions:
                version = version_info['version']
                logging.debug(f"Adding version {version} to dropdown")
                self.version_dropdown.addItem(
                    f"Version {version}",
                    version_info
                )

            self.version_dropdown.setEnabled(True)
            self.update_button.setEnabled(True)
            status_message = f"Found {len(self.available_versions)} available versions."
            self.status_label.setText(status_message)
            logging.info(status_message)
        else:
            # Fall back to the standard update check
            logging.info("No versions found, falling back to standard update check")
            update_available, latest_version, error = check_for_updates()

            if error:
                error_message = f"Error checking for updates: {error}"
                logging.error(error_message)
                self.status_label.setText(error_message)
            elif update_available:
                status_message = f"Update available: version {latest_version}"
                logging.info(status_message)
                self.status_label.setText(status_message)
                self.update_button.setEnabled(True)
            else:
                status_message = f"ChiSurf is already up to date (version {info.__version__})."
                logging.info(status_message)
                self.status_label.setText(status_message)

        self.check_button.setEnabled(True)
        logging.debug("Update check completed")

    def update_chisurf(self):
        """Update ChiSurf to the selected version."""
        logging.info("Starting ChiSurf update via UI")

        # Get the selected version
        selected_index = self.version_dropdown.currentIndex()
        logging.debug(f"Selected version index: {selected_index}")

        # Create progress dialog
        progress_dialog = QProgressDialog("Updating ChiSurf...", "Cancel", 0, 0, self)
        progress_dialog.setWindowTitle("Updating")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.show()
        logging.debug("Created and showed progress dialog")

        # Define callback to update progress dialog and command display
        def update_callback(message):
            # Update the progress dialog
            progress_dialog.setLabelText(message)

            # Process UI events to keep the interface responsive
            QApplication.processEvents()

        # Show a warning message before starting the update
        logging.info("Showing update warning dialog")
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
            logging.info("Update cancelled by user")
            progress_dialog.close()
            self.status_label.setText("Update cancelled by user.")
            return

        # If we have available versions and one is selected, use it
        if self.available_versions and selected_index >= 0:
            selected_version = self.version_dropdown.itemData(selected_index)
            logging.debug(f"Selected version data: {selected_version}")

            if selected_version:
                version = selected_version['version']
                file_path = selected_version['file_path']

                # Update the status
                status_message = f"Updating to version {version}..."
                logging.info(status_message)
                self.status_label.setText(status_message)
                update_callback(status_message)

                # Log the update file path
                logging.info(f"Update file path: {file_path}")

                # Perform the update using the selected version
                # Note: auto_restart is ignored as the application will be closed
                logging.info(f"Starting update to version {version}")
                self.updater.update_to_version(
                    file_path, 
                    callback=update_callback,
                    auto_restart=False
                )

                # The application will exit during the update process, so this code won't be reached
                return

        # If no version is selected or available, fall back to the standard update
        logging.info("No specific version selected, using standard update")
        # Note: auto_restart is ignored as the application will be closed
        update_chisurf(callback=update_callback, auto_restart=False)

        # The application will exit during the update process, so this code won't be reached
        logging.debug("This code should not be reached as the application will exit during update")

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the UpdaterWidget class
    window = UpdaterWidget()
    # Show the window
    window.show()
