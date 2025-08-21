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
    QMessageBox, QRadioButton, QLineEdit, QFileDialog, QGroupBox, QCheckBox
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

    def __init__(self, parent=None, suppress_initial_notification: bool = False):
        """Initialize the updater widget."""
        super().__init__(parent)
        self.setWindowTitle("ChiSurf Updater")
        self.available_versions = []
        # Whether to suppress the initial informational popup when the widget auto-checks on start
        self._suppress_initial_notification = bool(suppress_initial_notification)
        # Set default and minimum size to 600x600 as requested
        try:
            self.resize(600, 600)
            self.setMinimumSize(600, 600)
        except Exception:
            pass

        # Import settings
        from chisurf.settings import cs_settings
        self.cs_settings = cs_settings

        # Get update URL from settings or fall back to the one from info.py
        hardcoded_url = "https://www.peulen.xyz/downloads/chisurf/conda"
        update_url = cs_settings.get('update_url', hardcoded_url)

        # Initialize updater with the update URL
        self.updater = ChiSurfUpdater(update_url=update_url)

        self.setup_ui()

        # Ensure we use development channel when applicable (always checked and disabled for now)
        try:
            if getattr(self, 'dev_checkbox', None) is not None and self.dev_checkbox.isChecked():
                self.updater.channel = "development"
            else:
                self.updater.channel = "master"
            # Update the branch label after setting channel
            try:
                self._update_branch_label()
            except Exception:
                pass
        except Exception:
            # Default to dev if anything goes wrong
            self.updater.channel = "development"
            try:
                self._update_branch_label()
            except Exception:
                pass

        # React to version selection to update changelog
        try:
            self.version_dropdown.currentIndexChanged.connect(self._on_version_changed)
        except Exception:
            pass

        # Automatically check for updates shortly after the widget starts
        try:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(150, self._auto_check_on_start)
        except Exception:
            # Fallback: direct call if QTimer not available
            try:
                self._auto_check_on_start()
            except Exception:
                pass

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

        # Development branch checkbox (always on, disabled since no stable release exists)
        dev_layout = QHBoxLayout()
        self.dev_checkbox = QCheckBox("Development")
        try:
            self.dev_checkbox.setChecked(True)
            self.dev_checkbox.setEnabled(False)  # user cannot uncheck for now
            self.dev_checkbox.setToolTip("ChiSurf currently has no stable release; updates check the development branch.")
            # Even if disabled for now, wire stateChanged for future-proofing
            try:
                self.dev_checkbox.stateChanged.connect(self._on_branch_checkbox_changed)
            except Exception:
                pass
        except Exception:
            pass
        dev_layout.addWidget(self.dev_checkbox)
        # Label to display the selected branch
        self.branch_label = QLabel("")
        dev_layout.addWidget(self.branch_label)
        dev_layout.addStretch()
        layout.addLayout(dev_layout)
        # Initialize branch label text
        try:
            self._update_branch_label()
        except Exception:
            pass

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

        # Changelog area
        layout.addWidget(QLabel("Changes since your installed version:"))
        self.changelog_text = QTextEdit()
        try:
            self.changelog_text.setReadOnly(True)
            # Disable line wrapping in changelog for better readability of long entries
            self.changelog_text.setLineWrapMode(QTextEdit.NoWrap)
            font = QFont("Consolas")
            font.setPointSize(9)
            self.changelog_text.setFont(font)
        except Exception:
            pass
        self.changelog_text.setPlaceholderText("Changelog will appear here after checking for updates...")
        layout.addWidget(self.changelog_text)

        self.setLayout(layout)
        try:
            # Set default window size to 600x600
            self.resize(800, 600)
        except Exception:
            pass

    def _auto_check_on_start(self):
        """Perform an automatic update check and inform the user if an update is available.
        Also populate the version selection combobox on startup. Keeps UI responsive and robust."""
        # Prepare UI state
        try:
            self.check_button.setEnabled(False)
            self.update_button.setEnabled(False)
            self.version_dropdown.setEnabled(False)
            self.version_dropdown.clear()
        except Exception:
            pass

        # First, try to get full update info to populate the versions dropdown
        populated_versions = False
        try:
            update_info = self.updater._get_update_info()
            if update_info:
                self.available_versions = update_info.get("available_versions", [])
                if not self.available_versions and self.updater._is_local_folder():
                    # If using a local folder, attempt a direct listing
                    self.available_versions = self.updater._list_available_versions()
                if self.available_versions:
                    for version_info in self.available_versions:
                        version = version_info.get('version')
                        if version:
                            self.version_dropdown.addItem(f"Version {version}", version_info)
                    self.version_dropdown.setEnabled(True)
                    self.update_button.setEnabled(True)
                    self.status_label.setText(f"Found {len(self.available_versions)} available versions.")
                    populated_versions = True
                    # Populate changelog for latest version if provided
                    try:
                        changelog = update_info.get("changelog")
                        if changelog:
                            self.changelog_text.setPlainText(changelog)
                        else:
                            self._update_changelog_for_selected()
                    except Exception:
                        pass
        except Exception:
            populated_versions = False

        # Then, do a light availability check to inform the user
        try:
            update_available, latest_version, error = check_for_updates()
        except Exception as e:
            update_available, latest_version, error = False, None, str(e)

        try:
            if error:
                # Keep any versions we may have populated, but show the error
                self.status_label.setText(f"Update check failed or skipped: {error}")
                self.check_button.setEnabled(True)
                # If versions not populated, leave update button disabled
                if not populated_versions:
                    self.update_button.setEnabled(False)
                return
            if update_available and latest_version:
                self.status_label.setText(f"Update available: version {latest_version}")
                self.update_button.setEnabled(True)
                # Inform the user with a non-intrusive prompt unless suppressed
                if not getattr(self, "_suppress_initial_notification", False):
                    try:
                        QMessageBox.information(
                            self,
                            "Update Available",
                            f"A new version of ChiSurf ({latest_version}) is available.",
                            QMessageBox.Ok
                        )
                    except Exception:
                        pass
            else:
                from chisurf import info as _info
                if not populated_versions:
                    self.status_label.setText(f"ChiSurf is up to date (version {_info.__version__}).")
        except Exception:
            pass
        finally:
            try:
                self.check_button.setEnabled(True)
            except Exception:
                pass


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

        # If we have available versions, populate the dropdown first
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

            # Populate changelog for current selection
            try:
                changelog = update_info.get("changelog") if isinstance(update_info, dict) else None
                if changelog:
                    self.changelog_text.setPlainText(changelog)
                else:
                    self._update_changelog_for_selected()
            except Exception:
                pass

            # After population, always run a light availability check to set a correct label
            try:
                update_available, latest_version, error = check_for_updates()
            except Exception as e:
                update_available, latest_version, error = False, None, str(e)

            if error:
                error_message = f"Error checking for updates: {error}"
                logging.error(error_message)
                self.status_label.setText(error_message)
            elif update_available and latest_version:
                status_message = f"Update available: version {latest_version}"
                logging.info(status_message)
                self.status_label.setText(status_message)
            else:
                status_message = f"ChiSurf is already up to date (version {info.__version__})."
                logging.info(status_message)
                self.status_label.setText(status_message)
        else:
            # No versions found; still run availability check to inform the user
            logging.info("No versions found; performing availability check")
            try:
                update_available, latest_version, error = check_for_updates()
            except Exception as e:
                update_available, latest_version, error = False, None, str(e)

            if error:
                error_message = f"Error checking for updates: {error}"
                logging.error(error_message)
                self.status_label.setText(error_message)
            elif update_available and latest_version:
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
        # Update changelog after finishing
        try:
            self._update_changelog_for_selected()
        except Exception:
            pass

    def _on_version_changed(self, index):
        try:
            self._update_changelog_for_selected()
        except Exception:
            pass

    def _update_changelog_for_selected(self):
        try:
            idx = self.version_dropdown.currentIndex()
            if idx < 0 and self.available_versions:
                idx = 0
            if idx < 0:
                return
            data = self.version_dropdown.itemData(idx)
            if not isinstance(data, dict):
                return
            target_version = data.get('version')
            if not target_version:
                return
            from chisurf import info as _info

            # Default: show changes from currently installed to selected
            from_version = _info.__version__

            # If a non-latest version is selected and a previous version exists in the list,
            # show the changes between the previous version and the selected version.
            try:
                if isinstance(self.available_versions, list) and idx >= 0 and (idx + 1) < len(self.available_versions):
                    prev_info = self.available_versions[idx + 1]
                    if isinstance(prev_info, dict):
                        prev_ver = prev_info.get('version')
                        if isinstance(prev_ver, str) and prev_ver:
                            from_version = prev_ver
            except Exception:
                # Fall back to current installed version if anything goes wrong
                pass

            changelog = self.updater._build_changelog(from_version, target_version)
            self.changelog_text.setPlainText(changelog)
        except Exception as e:
            try:
                self.changelog_text.setPlainText(f"Could not load changelog: {e}")
            except Exception:
                pass

    def _on_branch_checkbox_changed(self, state):
        """Update channel and label if the branch checkbox changes."""
        try:
            if self.dev_checkbox.isChecked():
                self.updater.channel = "development"
            else:
                # If ever allowed to uncheck, fallback to main/master
                self.updater.channel = "master"
            self._update_branch_label()
        except Exception:
            pass

    def _update_branch_label(self):
        """Refresh the QLabel to show the currently selected branch."""
        try:
            # Prefer updater.channel if available
            branch_text = None
            try:
                ch = getattr(self.updater, 'channel', None)
                if isinstance(ch, str) and ch:
                    ch_lower = ch.lower()
                    if ch_lower.startswith('dev') or ch_lower == 'development':
                        branch_text = 'Development'
                    elif ch_lower in ('main', 'master'):
                        branch_text = 'Main'
                    else:
                        # Show raw channel name if it's custom
                        branch_text = ch
            except Exception:
                pass

            # Fallback to checkbox state if needed
            if not branch_text:
                branch_text = 'Development' if self.dev_checkbox.isChecked() else 'Main'

            self.branch_label.setText(f"Selected branch: {branch_text}")
        except Exception:
            pass

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
