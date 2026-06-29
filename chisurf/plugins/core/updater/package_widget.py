"""
Package Manager dialog for ChiSurf.

Provides a simple UI to manage packages, environments, and channels
using PackageManager.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
import logging

# Logger for the package manager dialog
logger = logging.getLogger("chisurf.packagemanager")

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QTabWidget, QWidget, QListWidget, QListWidgetItem, QTextEdit, QFileDialog,
    QMessageBox, QInputDialog, QTableWidget, QTableWidgetItem, QToolButton,
    QSpacerItem, QSizePolicy
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QTextCursor

from .updater import PackageManager

class PackageWorker(QThread):
    """
    Worker thread for running package manager commands asynchronously.
    """
    finished = Signal(bool, object, str)  # success, data, error_msg

    def __init__(self, func, *args, **kwargs):
        """Initialize the worker with a callable, positional and keyword arguments.

        Parameters
        ----------
        func : callable
            The function to execute in the worker thread.
        *args
            Positional arguments forwarded to ``func``.
        **kwargs
            Keyword arguments forwarded to ``func``.
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Execute ``self.func`` and emit ``finished`` with the normalised result tuple."""
        try:
            # Most PackageManager methods return (success, data/output, error_msg)
            # but some might return just (success, output)
            result = self.func(*self.args, **self.kwargs)
            
            if isinstance(result, tuple):
                if len(result) == 3:
                    self.finished.emit(result[0], result[1], result[2])
                elif len(result) == 2:
                    self.finished.emit(result[0], result[1], "")
                else:
                    self.finished.emit(True, result, "")
            else:
                self.finished.emit(True, result, "")
        except Exception as e:
            logger.error(f"Error in PackageWorker: {e}")
            self.finished.emit(False, None, str(e))

class PackageManagerWidget(QWidget):
    """
    An embeddable widget for managing packages, environments, and channels.

    This is the reusable panel used both as a standalone window and inside the
    unified Settings dialog. :class:`PackageManagerDialog` wraps it as a modal
    dialog for backwards compatibility.
    """
    def __init__(self, parent=None):
        """Initialize the widget, create the ``PackageManager`` and load initial data.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)

        # Initialize the package manager
        self.manager = PackageManager()

        self.setup_ui()

        # Load initial data
        self.refresh_all()

    def setup_ui(self):
        """Build the tabbed interface and operation log area."""
        layout = QVBoxLayout(self)

        # Tab widget for different management areas
        self.tabs = QTabWidget()

        # 1. Installed Packages Tab
        self.installed_tab = QWidget()
        self.setup_installed_tab()
        self.tabs.addTab(self.installed_tab, "Installed Packages")

        # 2. Search & Install Tab
        self.search_tab = QWidget()
        self.setup_search_tab()
        self.tabs.addTab(self.search_tab, "Search & Install")

        # 3. Environments Tab
        self.envs_tab = QWidget()
        self.setup_envs_tab()
        self.tabs.addTab(self.envs_tab, "Environments")

        # 4. Channels Tab
        self.channels_tab = QWidget()
        self.setup_channels_tab()
        self.tabs.addTab(self.channels_tab, "Channels")

        layout.addWidget(self.tabs)

        # Status log area (at the bottom)
        log_label = QLabel("Operation Log:")
        layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        # Close button
        button_box = QHBoxLayout()
        refresh_btn = QPushButton("Refresh All")
        refresh_btn.clicked.connect(self.refresh_all)
        button_box.addWidget(refresh_btn)

        button_box.addStretch()

        layout.addLayout(button_box)

    def setup_installed_tab(self):
        """Construct the 'Installed Packages' tab widgets."""
        layout = QVBoxLayout(self.installed_tab)

        # Environment selection
        env_layout = QHBoxLayout()
        env_layout.addWidget(QLabel("Current Environment:"))
        self.current_env_label = QLabel("Loading...")
        env_layout.addWidget(self.current_env_label)
        env_layout.addStretch()

        self.refresh_installed_btn = QPushButton("Refresh List")
        self.refresh_installed_btn.clicked.connect(self.refresh_installed)
        env_layout.addWidget(self.refresh_installed_btn)

        layout.addLayout(env_layout)

        # Search filter for installed packages
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Filter:"))
        self.installed_filter = QLineEdit()
        self.installed_filter.setPlaceholderText("Filter installed packages...")
        self.installed_filter.textChanged.connect(self.filter_installed)
        search_layout.addWidget(self.installed_filter)
        layout.addLayout(search_layout)

        # Table for installed packages
        self.installed_table = QTableWidget(0, 3)
        self.installed_table.setHorizontalHeaderLabels(["Name", "Version", "Channel"])
        self.installed_table.horizontalHeader().setStretchLastSection(True)
        self.installed_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.installed_table)

        # Actions for installed packages
        btn_layout = QHBoxLayout()

        self.update_btn = QPushButton("Update Selected")
        self.update_btn.clicked.connect(self.update_selected)
        btn_layout.addWidget(self.update_btn)

        self.update_all_btn = QPushButton("Update All")
        self.update_all_btn.clicked.connect(self.update_all)
        btn_layout.addWidget(self.update_all_btn)

        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_selected)
        btn_layout.addWidget(self.remove_btn)

        layout.addLayout(btn_layout)

    def setup_search_tab(self):
        """Construct the 'Search & Install' tab widgets."""
        layout = QVBoxLayout(self.search_tab)

        # Search input
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter package name to search...")
        self.search_input.returnPressed.connect(self.search_packages)
        search_layout.addWidget(self.search_input)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.search_packages)
        search_layout.addWidget(self.search_btn)

        layout.addLayout(search_layout)

        # Results list
        self.search_results = QTableWidget(0, 3)
        self.search_results.setHorizontalHeaderLabels(["Name", "Version", "Channel"])
        self.search_results.horizontalHeader().setStretchLastSection(True)
        self.search_results.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.search_results)

        # Install button
        self.install_btn = QPushButton("Install Selected")
        self.install_btn.clicked.connect(self.install_selected)
        layout.addWidget(self.install_btn)

    def setup_envs_tab(self):
        """Construct the 'Environments' tab widgets."""
        layout = QVBoxLayout(self.envs_tab)

        # List of environments
        self.envs_list = QListWidget()
        layout.addWidget(self.envs_list)

        # Environment actions
        btn_layout = QHBoxLayout()

        self.create_env_btn = QPushButton("Create New")
        self.create_env_btn.clicked.connect(self.create_env)
        btn_layout.addWidget(self.create_env_btn)

        self.clone_env_btn = QPushButton("Clone Selected")
        self.clone_env_btn.clicked.connect(self.clone_env)
        btn_layout.addWidget(self.clone_env_btn)

        self.remove_env_btn = QPushButton("Remove Selected")
        self.remove_env_btn.clicked.connect(self.remove_env)
        btn_layout.addWidget(self.remove_env_btn)

        layout.addLayout(btn_layout)

        env_io_layout = QHBoxLayout()
        self.export_env_btn = QPushButton("Export to File")
        self.export_env_btn.clicked.connect(self.export_env)
        env_io_layout.addWidget(self.export_env_btn)

        self.import_env_btn = QPushButton("Import from File")
        self.import_env_btn.clicked.connect(self.import_env)
        env_io_layout.addWidget(self.import_env_btn)

        layout.addLayout(env_io_layout)

    def setup_channels_tab(self):
        """Construct the 'Channels' tab widgets."""
        layout = QVBoxLayout(self.channels_tab)
        
        # List of channels
        self.channels_list = QListWidget()
        layout.addWidget(self.channels_list)
        
        # Channel actions
        btn_layout = QHBoxLayout()
        
        self.add_channel_btn = QPushButton("Add Channel")
        self.add_channel_btn.clicked.connect(self.add_channel)
        btn_layout.addWidget(self.add_channel_btn)
        
        self.remove_channel_btn = QPushButton("Remove Selected")
        self.remove_channel_btn.clicked.connect(self.remove_channel)
        btn_layout.addWidget(self.remove_channel_btn)
        
        layout.addLayout(btn_layout)

    # --- Operation Handlers ---

    def log(self, message: str):
        """Append a timestamped message to the operation log.

        Parameters
        ----------
        message : str
            Message to append to the log.
        """
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.log_text.moveCursor(QTextCursor.End)

    def refresh_all(self):
        """Refresh all tabs (installed packages, environments, channels) and the env label."""
        self.refresh_installed()
        self.refresh_envs()
        self.refresh_channels()
        self.current_env_label.setText(self.manager.current_prefix())

    def refresh_installed(self):
        """Start a worker that reloads the installed packages list."""
        self.log("Refreshing installed packages...")
        self.refresh_installed_btn.setEnabled(False)
        worker = PackageWorker(self.manager.list_installed)
        worker.finished.connect(self._on_installed_loaded)
        worker.finished.connect(lambda: self.refresh_installed_btn.setEnabled(True))
        # Keep reference to avoid GC
        self._installed_worker = worker
        worker.start()

    def _on_installed_loaded(self, success, data, error):
        """Handle the result of the installed packages worker.

        Parameters
        ----------
        success : bool
            Whether the worker call succeeded.
        data : list
            List of package dictionaries.
        error : str
            Error message when ``success`` is ``False``.
        """
        if not success:
            self.log(f"Error loading installed packages: {error}")
            return

        self.installed_packages_data = data  # Save for filtering
        self._populate_installed_table(data)
        self.log(f"Loaded {len(data)} packages.")

    def _populate_installed_table(self, data):
        """Replace the contents of the installed-packages table with ``data``.

        Parameters
        ----------
        data : list of dict
            Each dict must contain at least 'name', 'version', and 'channel'.
        """
        self.installed_table.setRowCount(0)
        for pkg in data:
            row = self.installed_table.rowCount()
            self.installed_table.insertRow(row)
            self.installed_table.setItem(row, 0, QTableWidgetItem(pkg.get('name', '')))
            self.installed_table.setItem(row, 1, QTableWidgetItem(pkg.get('version', '')))
            self.installed_table.setItem(row, 2, QTableWidgetItem(pkg.get('channel', '')))

    def filter_installed(self, text):
        """Show only the installed packages whose name contains ``text`` (case-insensitive).

        Parameters
        ----------
        text : str
            Substring filter applied to package names.
        """
        if not hasattr(self, 'installed_packages_data'):
            return

        filtered = [pkg for pkg in self.installed_packages_data if text.lower() in pkg.get('name', '').lower()]
        self._populate_installed_table(filtered)

    def search_packages(self):
        """Search configured channels for the query entered in the search input."""
        query = self.search_input.text().strip()
        if not query:
            return
        
        self.log(f"Searching for '{query}'...")
        self.search_btn.setEnabled(False)
        worker = PackageWorker(self.manager.search, query)
        worker.finished.connect(self._on_search_finished)
        worker.finished.connect(lambda: self.search_btn.setEnabled(True))
        self._search_worker = worker
        worker.start()

    def _on_search_finished(self, success, data, error):
        """Populate the search results table from the worker response.

        Parameters
        ----------
        success : bool
            Whether the search call succeeded.
        data : list
            List of package dictionaries on success.
        error : str
            Error message when ``success`` is ``False``.
        """
        if not success:
            self.log(f"Search failed: {error}")
            return

        self.search_results.setRowCount(0)
        count = 0
        # data format depends on solver,PackageManager tries to normalize
        if isinstance(data, list):
            for pkg in data:
                row = self.search_results.rowCount()
                self.search_results.insertRow(row)
                self.search_results.setItem(row, 0, QTableWidgetItem(pkg.get('name', '')))
                self.search_results.setItem(row, 1, QTableWidgetItem(pkg.get('version', '')))
                self.search_results.setItem(row, 2, QTableWidgetItem(pkg.get('channel', '')))
                count += 1

        self.log(f"Found {count} results.")

    def install_selected(self):
        """Install the currently selected search-result packages after confirmation."""
        selected = self.search_results.selectedItems()
        if not selected:
            return
        
        # Get names from column 0
        pkgs = list(set([self.search_results.item(item.row(), 0).text() for item in selected]))
        
        confirm = QMessageBox.question(self, "Confirm Installation", 
                                     f"Are you sure you want to install:\n{', '.join(pkgs)}?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if confirm == QMessageBox.Yes:
            self.log(f"Installing {', '.join(pkgs)}...")
            worker = PackageWorker(self.manager.install, pkgs)
            worker.finished.connect(self._on_operation_finished)
            self._op_worker = worker
            worker.start()

    def update_selected(self):
        """Update the selected installed packages."""
        selected = self.installed_table.selectedItems()
        if not selected:
            return
        
        pkgs = list(set([self.installed_table.item(item.row(), 0).text() for item in selected]))
        self.log(f"Updating {', '.join(pkgs)}...")
        worker = PackageWorker(self.manager.update, pkgs)
        worker.finished.connect(self._on_operation_finished)
        self._op_worker = worker
        worker.start()

    def update_all(self):
        """Ask for confirmation, then update every package in the current environment."""
        confirm = QMessageBox.question(self, "Update All", 
                                     "Update all packages in the current environment?",
                                     QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.log("Updating all packages...")
            worker = PackageWorker(self.manager.update)
            worker.finished.connect(self._on_operation_finished)
            self._op_worker = worker
            worker.start()

    def remove_selected(self):
        """Remove the selected installed packages after confirmation."""
        selected = self.installed_table.selectedItems()
        if not selected:
            return
        
        pkgs = list(set([self.installed_table.item(item.row(), 0).text() for item in selected]))
        confirm = QMessageBox.question(self, "Confirm Removal", 
                                     f"Are you sure you want to remove:\n{', '.join(pkgs)}?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if confirm == QMessageBox.Yes:
            self.log(f"Removing {', '.join(pkgs)}...")
            worker = PackageWorker(self.manager.remove, pkgs)
            worker.finished.connect(self._on_operation_finished)
            self._op_worker = worker
            worker.start()

    def _on_operation_finished(self, success, data, error):
        """Handle completion of an asynchronous package operation.

        Parameters
        ----------
        success : bool
            Whether the operation succeeded.
        data : object
            Optional payload returned by the operation.
        error : str
            Error message on failure.
        """
        if success:
            self.log("Operation completed successfully.")
            self.refresh_all()
        else:
            self.log(f"Operation failed: {error}")
            QMessageBox.critical(self, "Error", f"The operation failed:\n{error}")
        if success:
            self.log("Operation completed successfully.")
            self.refresh_all()
        else:
            self.log(f"Operation failed: {error}")
            QMessageBox.critical(self, "Error", f"The operation failed:\n{error}")

    # --- Env Handlers ---

    def refresh_envs(self):
        """Start a worker that reloads the list of available environments."""
        worker = PackageWorker(self.manager.list_envs)
        worker.finished.connect(self._on_envs_loaded)
        self._envs_worker = worker
        worker.start()

    def _on_envs_loaded(self, success, data, error):
        """Populate the environment list widget from the worker response.

        Parameters
        ----------
        success : bool
            Whether the call succeeded.
        data : list
            List of environment names/prefixes.
        error : str
            Error message on failure.
        """
        if success:
            self.envs_list.clear()
            for env in data:
                self.envs_list.addItem(env)
        else:
            self.log(f"Error loading environments: {error}")

    def create_env(self):
        """Prompt for a name and create a new conda environment."""
        name, ok = QInputDialog.getText(self, "New Environment", "Enter environment name:")
        if ok and name:
            self.log(f"Creating environment '{name}'...")
            worker = PackageWorker(self.manager.create_env, name=name)
            worker.finished.connect(self._on_operation_finished)
            self._op_worker = worker
            worker.start()

    def clone_env(self):
        """Clone the currently selected environment to a user-provided name."""
        selected = self.envs_list.currentItem()
        if not selected:
            return
        src = selected.text()
        dst, ok = QInputDialog.getText(self, "Clone Environment", f"Enter new name for clone of '{src}':")
        if ok and dst:
            self.log(f"Cloning environment '{src}' to '{dst}'...")
            # Detect if path or name
            if os.sep in src:
                worker = PackageWorker(self.manager.clone_env, prefix_src=src, name_dst=dst)
            else:
                worker = PackageWorker(self.manager.clone_env, name_src=src, name_dst=dst)
            worker.finished.connect(self._on_operation_finished)
            self._op_worker = worker
            worker.start()

    def remove_env(self):
        """Remove the currently selected environment after confirmation."""
        selected = self.envs_list.currentItem()
        if not selected:
            return
        env = selected.text()
        confirm = QMessageBox.question(self, "Confirm removal", f"Remove environment '{env}'?", 
                                     QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.log(f"Removing environment '{env}'...")
            if os.sep in env:
                worker = PackageWorker(self.manager.remove_env, prefix=env)
            else:
                worker = PackageWorker(self.manager.remove_env, name=env)
            worker.finished.connect(self._on_operation_finished)
            self._op_worker = worker
            worker.start()

    def export_env(self):
        """Export the selected (or current) environment to a YAML file."""
        selected = self.envs_list.currentItem()
        if not selected:
            prefix = self.manager.current_prefix()
        else:
            prefix = selected.text()
            if os.sep not in prefix:
                # Need to find prefix for name
                prefix = None # PackageManager export_env handles default
        
        path, _ = QFileDialog.getSaveFileName(self, "Export Environment", "", "YAML files (*.yaml *.yml)")
        if path:
            self.log(f"Exporting environment to {path}...")
            # This returns YAML text in data
            worker = PackageWorker(self.manager.export_env, prefix=prefix)
            def _on_exported(s, d, e):
                """Write the exported YAML to disk or log the failure.

                Parameters
                ----------
                s : bool
                    Whether the export call succeeded.
                d : str
                    YAML text returned by the export call.
                e : str
                    Error message when ``s`` is ``False``.
                """
                if s:
                    try:
                        with open(path, 'w') as f:
                            f.write(d)
                        self.log(f"Exported successfully to {path}")
                    except Exception as ex:
                        self.log(f"Failed to write file: {ex}")
                else:
                    self.log(f"Export failed: {e}")
            worker.finished.connect(_on_exported)
            self._op_worker = worker
            worker.start()

    def import_env(self):
        """Import an environment from a YAML file, optionally with a new name."""
        path, _ = QFileDialog.getOpenFileName(self, "Import Environment", "", "YAML files (*.yaml *.yml)")
        if not path:
            return
        name, ok = QInputDialog.getText(self, "Import Environment", "Enter name for new environment (optional):")
        self.log(f"Importing environment from {path}...")
        worker = PackageWorker(self.manager.import_env, path, name if ok and name else None)
        worker.finished.connect(self._on_operation_finished)
        self._op_worker = worker
        worker.start()

    # --- Channel Handlers ---

    def refresh_channels(self):
        """Start a worker that reloads the list of configured channels."""
        worker = PackageWorker(self.manager.get_channels)
        worker.finished.connect(self._on_channels_loaded)
        self._channels_worker = worker
        worker.start()

    def _on_channels_loaded(self, success, data, error):
        """Populate the channel list widget from the worker response.

        Parameters
        ----------
        success : bool
            Whether the call succeeded.
        data : list
            List of channel names.
        error : str
            Error message on failure.
        """
        if success:
            self.channels_list.clear()
            for ch in data:
                self.channels_list.addItem(ch)
        else:
            self.log(f"Error loading channels: {error}")

    def add_channel(self):
        """Prompt for a name/URL and add a new channel."""
        ch, ok = QInputDialog.getText(self, "Add Channel", "Enter channel name or URL:")
        if ok and ch:
            self.log(f"Adding channel '{ch}'...")
            worker = PackageWorker(self.manager.add_channel, ch)
            worker.finished.connect(self._on_operation_finished)
            self._op_worker = worker
            worker.start()

    def remove_channel(self):
        """Remove the currently selected channel."""
        selected = self.channels_list.currentItem()
        if not selected:
            return
        ch = selected.text()
        self.log(f"Removing channel '{ch}'...")
        worker = PackageWorker(self.manager.remove_channel, ch)
        worker.finished.connect(self._on_operation_finished)
        self._op_worker = worker
        worker.start()


class PackageManagerDialog(QDialog):
    """Modal dialog wrapper around :class:`PackageManagerWidget`.

    Retained for backwards compatibility (standalone "Package Manager" window).
    The actual UI lives in :class:`PackageManagerWidget`, which is also embedded
    as a panel in the unified Settings dialog.
    """

    def __init__(self, parent=None):
        """Build the dialog and embed a :class:`PackageManagerWidget`.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("ChiSurf Package Manager")
        self.resize(800, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.widget = PackageManagerWidget(self)
        layout.addWidget(self.widget)

        button_box = QHBoxLayout()
        button_box.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_box.addWidget(close_btn)
        layout.addLayout(button_box)
