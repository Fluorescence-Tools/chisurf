"""
ChiSurf Update & Conda Manager Plugin

This plugin provides:
- Update checker and installer for ChiSurf (via conda)
- A simple Conda Package Manager UI to search/install/update/remove packages,
  manage environments (list/create/remove/clone/export/import), and manage
  channels (list/add/remove)

It supports Windows, macOS, and Linux. On Windows, elevated privileges are
handled when required by the updater.

Notes:
- Updating ChiSurf may close all ChiSurf windows and continue in a separate window.
  After completion, restart ChiSurf manually.
- The update URL is configured in the settings or defaults to the one specified in info.py.
"""

import sys
import logging
import html
import yaml
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QProgressDialog, QApplication, QComboBox, QTextEdit,
    QMessageBox, QRadioButton, QLineEdit, QFileDialog, QGroupBox, QCheckBox,
    QListWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from .updater import ChiSurfUpdater, check_for_updates, update_chisurf
from .conda_widget import CondaManagerDialog
from chisurf import info
import chisurf.settings as _cs_settings_mod

# Define the plugin name - this will appear in the Plugins menu
name = "Help:Updates and Packages"

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

        # Load startup-related settings for the updater plugin
        try:
            self._load_startup_settings()
        except Exception:
            # Fallback defaults
            self._ignore_updates = False
            self._check_on_startup = True

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

        # Startup behavior group
        startup_group = QGroupBox("Startup behavior")
        sg_layout = QVBoxLayout()
        # Check on startup
        self.cb_check_on_start = QCheckBox("Check for updates on startup")
        try:
            self.cb_check_on_start.setToolTip("When enabled, ChiSurf will check for updates during startup.")
            self.cb_check_on_start.setChecked(bool(getattr(self, '_check_on_startup', True)))
            self.cb_check_on_start.stateChanged.connect(lambda s: self._on_toggle_check_on_startup(s == Qt.Checked))
        except Exception:
            pass
        sg_layout.addWidget(self.cb_check_on_start)
        # Ignore updates (suppress startup prompts)
        self.cb_ignore_updates = QCheckBox("Ignore updates (do not prompt on startup)")
        try:
            self.cb_ignore_updates.setToolTip("If enabled, ChiSurf will not prompt about updates during startup.")
            self.cb_ignore_updates.setChecked(bool(getattr(self, '_ignore_updates', False)))
            self.cb_ignore_updates.stateChanged.connect(lambda s: self._on_toggle_ignore_updates(s == Qt.Checked))
        except Exception:
            pass
        sg_layout.addWidget(self.cb_ignore_updates)
        startup_group.setLayout(sg_layout)
        layout.addWidget(startup_group)

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

        # Open Package Manager button
        self.conda_manager_button = QPushButton("Package Manager")
        try:
            self.conda_manager_button.setToolTip("Open the package manager to manage conda packages in your environment.")
            self.conda_manager_button.clicked.connect(self.open_conda_manager)
        except Exception:
            pass
        button_layout.addWidget(self.conda_manager_button)

        layout.addLayout(button_layout)

        # Changelog area
        layout.addWidget(QLabel("Changes since your installed version:"))
        self.changelog_text = QTextEdit()
        try:
            self.changelog_text.setReadOnly(True)
            # Enable line wrapping so long entries are easier to read
            self.changelog_text.setLineWrapMode(QTextEdit.WidgetWidth)
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

    def _format_changelog_html(self, text: str) -> str:
        """Return pretty HTML for the raw changelog string.
        - Wraps lines (handled by QTextEdit), adds indentation via list formatting
        - Converts lines starting with "- " into <li> items
        - Preserves non-list paragraphs and footer links
        """
        try:
            if not isinstance(text, str) or not text.strip():
                return "<i>No changelog available.</i>"

            lines = text.splitlines()
            header = None
            items = []
            others = []
            footer = []

            # Simple state machine: collect leading header, bullet items, other lines, and detect footer hint
            for i, ln in enumerate(lines):
                s = ln.strip("\r\n")
                if i == 0 and s.lower().startswith("changes "):
                    header = html.escape(s)
                    continue
                if s.startswith("- "):
                    # Keep the date/message nicely separated; escape HTML
                    items.append(html.escape(s[2:].strip()))
                elif s.lower().startswith("more details:") or s.lower().startswith("see commit history:"):
                    footer.append(s)
                elif s:
                    others.append(s)

            html_parts = []
            if header:
                html_parts.append(f"<b>{header}</b>")

            if items:
                html_parts.append("<ul>")
                for it in items:
                    html_parts.append(f"  <li>{it}</li>")
                html_parts.append("</ul>")

            # Any remaining paragraphs
            for para in others:
                html_parts.append(f"<p>{html.escape(para)}</p>")

            # Footer with links if any
            for ft in footer:
                # try to hyperlink if URL present
                parts = ft.split()  # naive
                url = None
                for p in parts:
                    if p.startswith("http://") or p.startswith("https://"):
                        url = p
                        break
                if url:
                    label = html.escape(ft.replace(url, "").strip(" :")) or "More details"
                    html_parts.append(f"<p>{label}: <a href=\"{html.escape(url)}\">{html.escape(url)}</a></p>")
                else:
                    html_parts.append(f"<p>{html.escape(ft)}</p>")

            return "\n".join(html_parts)
        except Exception:
            # Fallback: escaped preformatted
            return f"<pre>{html.escape(str(text))}</pre>"

    def open_conda_manager(self):
        """Open the Conda Package Manager dialog."""
        try:
            dlg = CondaManagerDialog(self)
            dlg.exec_()
        except Exception as e:
            try:
                QMessageBox.critical(self, "Conda Manager", f"Failed to open Conda Manager:\n{e}")
            except Exception:
                pass

    def _load_startup_settings(self) -> None:
        """Load updater startup settings from user chisurf settings.
        Defaults: ignore_updates_on_startup=False, check_on_startup=True.
        Stored under cs_settings['plugins']['updater']."""
        try:
            # Get plugin settings dict safely
            plugins = self.cs_settings.get('plugins') or {}
            updater_settings = plugins.get('updater') or {}
            self._ignore_updates = bool(updater_settings.get('ignore_updates_on_startup', False))
            self._check_on_startup = bool(updater_settings.get('check_on_startup', True))
        except Exception:
            self._ignore_updates = False
            self._check_on_startup = True

    def _save_startup_settings(self) -> bool:
        """Persist updater startup settings into settings_chisurf.yaml.
        Returns True on success, False otherwise."""
        try:
            # Ensure plugin settings path exists in cs_settings
            all_settings = _cs_settings_mod.cs_settings
            if 'plugins' not in all_settings or not isinstance(all_settings['plugins'], dict):
                all_settings['plugins'] = {}
            if 'updater' not in all_settings['plugins'] or not isinstance(all_settings['plugins']['updater'], dict):
                all_settings['plugins']['updater'] = {}
            all_settings['plugins']['updater']['ignore_updates_on_startup'] = bool(self._ignore_updates)
            all_settings['plugins']['updater']['check_on_startup'] = bool(self._check_on_startup)

            # Write back to yaml file
            settings_file = _cs_settings_mod.chisurf_settings_file
            with open(settings_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(all_settings, f, default_flow_style=False)
            return True
        except Exception:
            return False

    def _on_toggle_check_on_startup(self, enabled: bool) -> None:
        """Handle change of 'check on startup' checkbox."""
        try:
            self._check_on_startup = bool(enabled)
            self._save_startup_settings()
        except Exception:
            pass

    def _on_toggle_ignore_updates(self, enabled: bool) -> None:
        """Handle change of 'ignore updates on startup' checkbox."""
        try:
            self._ignore_updates = bool(enabled)
            self._save_startup_settings()
        except Exception:
            pass

    def _auto_check_on_start(self):
        """Perform an automatic update check on startup respecting user settings.
        If updates are ignored or startup checks are disabled, skip notifying on startup.
        Also populate the version selection combobox on startup when allowed. """
        # Respect user settings ONLY during application startup, not when user opens this widget
        try:
            if getattr(self, '_suppress_initial_notification', False) and (
                getattr(self, '_ignore_updates', False) or not getattr(self, '_check_on_startup', True)
            ):
                # Do not auto-check or prompt on startup
                self.status_label.setText("Startup update check is disabled by user settings.")
                # Ensure buttons are enabled for manual checks
                self.check_button.setEnabled(True)
                # Do not touch update button here; it will be enabled after manual checks
                return
        except Exception:
            pass

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
                            self.changelog_text.setHtml(self._format_changelog_html(changelog))
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
                    self.changelog_text.setHtml(self._format_changelog_html(changelog))
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
            self.changelog_text.setHtml(self._format_changelog_html(changelog))
        except Exception as e:
            try:
                self.changelog_text.setHtml(self._format_changelog_html(f"Could not load changelog: {e}"))
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


def build_installed_vs_latest_changelog(latest_version: str, max_chars: int = 1500, limit: int = 50):
    """
    Build changelog text comparing the installed version vs the provided latest version.

    This helper is used by the startup prompt. It should be quiet on errors
    (no raw HTTP errors in the popup) and mirror the UpdaterWidget defaults
    (development branch).

    Args:
        latest_version: The latest version string to compare against the installed version.
        max_chars: Optional maximum number of characters to return; truncates with hint if exceeded.
        limit: Maximum number of commit entries to include when querying GitHub.

    Returns:
        Tuple: (installed_version, changelog_text)
    """
    try:
        from chisurf import info as _info
        installed = getattr(_info, "__version__", "")
    except Exception:
        installed = ""

    try:
        up = ChiSurfUpdater()
        # Use the same default branch as the Updater UI (development)
        try:
            up.channel = "development"
        except Exception:
            pass
        changelog = up._build_changelog(installed, str(latest_version), limit=limit)
        # Suppress fallback error texts in startup popup
        if isinstance(changelog, str):
            cl_lower = changelog.lower()
            if ("could not be retrieved" in cl_lower) or ("could not be determined" in cl_lower):
                changelog = ""
            elif max_chars and len(changelog) > max_chars:
                changelog = changelog[:max_chars] + "\n...\n(Open Updater to see full changes)"
        return installed, changelog
    except Exception:
        return installed, ""
