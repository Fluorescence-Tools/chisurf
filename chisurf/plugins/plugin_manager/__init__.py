"""
Plugin Manager for ChiSurf

This plugin allows you to manage all installed plugins in ChiSurf. You can:
- View all available plugins
- Enable or disable plugins
- View plugin descriptions
- Import plugins from external directories

The plugin manager provides a convenient interface for configuring how plugins
appear in the ChiSurf menu system. The import feature allows you to add plugins
from external directories, with automatic handling of security elevation
when needed for protected system locations on Windows, macOS, and Linux.
"""

import sys
import os
import pathlib
import importlib
import pkgutil
import yaml
import ast
import shutil
import ctypes
import platform
import subprocess
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QCheckBox,
    QMessageBox, QGroupBox, QScrollArea, QSplitter, QTextEdit, QLineEdit,
    QFileDialog, QInputDialog
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon

import chisurf
import chisurf.plugins
import chisurf.settings

# Define the plugin name - this will appear in the Plugins menu
name = "Tools:Plugin Manager"


def read_module_docstring(package_path: pathlib.Path) -> Optional[str]:
    """
    Given a path to a package directory, reads its __init__.py
    and returns the module docstring (or None if there isn’t one).
    """
    init_py = package_path / "__init__.py"
    if not init_py.exists():
        return None

    # Read the source
    source = init_py.read_text(encoding="utf-8")

    # Parse into an AST and extract the docstring
    tree = ast.parse(source, filename=str(init_py))
    return ast.get_docstring(tree)



class PluginManagerWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plugin Manager")
        self.resize(800, 400)

        # Get plugin settings
        self.plugin_settings = chisurf.settings.cs_settings.get('plugins', {})
        self.disabled_plugins = self.plugin_settings.get('disabled_plugins', [])  # Keep the key for backward compatibility
        self.hide_disabled_plugins = self.plugin_settings.get('hide_disabled_plugins', True)  # Keep the key for backward compatibility
        self.plugin_order = self.plugin_settings.get('plugin_order', {})
        self.toolbar_plugins = self.plugin_settings.get('toolbar_plugins', [])

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create splitter for list and details
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Create list widget for plugins
        list_group = QGroupBox("Available Plugins")
        list_layout = QVBoxLayout(list_group)

        # Add filter line edit
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_line_edit = QLineEdit()
        self.filter_line_edit.setPlaceholderText("Enter plugin name to filter...")
        self.filter_line_edit.textChanged.connect(self.on_filter_text_changed)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_line_edit)
        list_layout.addLayout(filter_layout)

        self.plugin_list = QListWidget()
        self.plugin_list.setMinimumWidth(300)
        self.plugin_list.currentItemChanged.connect(self.on_plugin_selected)
        list_layout.addWidget(self.plugin_list)
        splitter.addWidget(list_group)

        # Create details widget
        details_group = QGroupBox("Plugin Details")
        details_layout = QVBoxLayout(details_group)

        # Plugin name and status
        name_layout = QHBoxLayout()
        self.plugin_name_label = QLabel("Select a plugin")
        name_layout.addWidget(self.plugin_name_label)

        # Add rename button
        self.rename_button = QPushButton("Rename")
        self.rename_button.clicked.connect(self.on_rename_plugin)
        self.rename_button.setEnabled(False)
        name_layout.addWidget(self.rename_button)

        name_layout.addStretch()
        details_layout.addLayout(name_layout)

        # Plugin status
        status_layout = QHBoxLayout()
        self.disabled_checkbox = QCheckBox("Disable plugin")
        self.disabled_checkbox.stateChanged.connect(self.on_disabled_changed)
        status_layout.addWidget(self.disabled_checkbox)

        # Toolbar placement
        self.toolbar_checkbox = QCheckBox("Show in toolbar")
        self.toolbar_checkbox.stateChanged.connect(self.on_toolbar_changed)
        status_layout.addWidget(self.toolbar_checkbox)

        status_layout.addStretch()
        details_layout.addLayout(status_layout)

        # Plugin ordering
        order_layout = QHBoxLayout()
        order_label = QLabel("Plugin Order:")
        order_layout.addWidget(order_label)

        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.clicked.connect(self.on_move_up)
        order_layout.addWidget(self.move_up_button)

        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.clicked.connect(self.on_move_down)
        order_layout.addWidget(self.move_down_button)

        order_layout.addStretch()
        details_layout.addLayout(order_layout)

        # Plugin path
        path_layout = QHBoxLayout()
        path_label = QLabel("Path:")
        path_layout.addWidget(path_label)
        self.plugin_path_label = QLabel("Not available")
        path_layout.addWidget(self.plugin_path_label)
        path_layout.addStretch()
        details_layout.addLayout(path_layout)

        # Plugin description
        self.description_edit = QTextEdit()
        self.description_edit.setReadOnly(True)
        details_layout.addWidget(self.description_edit)

        splitter.addWidget(details_group)

        # Create settings group
        settings_group = QGroupBox("Global Plugin Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Hide disabled plugins checkbox
        self.hide_disabled_checkbox = QCheckBox("Hide disabled plugins")
        self.hide_disabled_checkbox.setChecked(self.hide_disabled_plugins)
        self.hide_disabled_checkbox.stateChanged.connect(self.on_hide_disabled_changed)
        settings_layout.addWidget(self.hide_disabled_checkbox)


        main_layout.addWidget(settings_group)

        # Create buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)

        refresh_button = QPushButton("Refresh Plugin List")
        refresh_button.clicked.connect(self.load_plugins)
        button_layout.addWidget(refresh_button)

        import_button = QPushButton("Import Plugin")
        import_button.clicked.connect(self.import_plugin)
        button_layout.addWidget(import_button)

        main_layout.addLayout(button_layout)

        # Load plugins
        self.load_plugins()

        # Current selected plugin
        self.current_plugin = None

    def load_plugins(self):
        """Load all available plugins, sorted by custom order or module name, and display them in the list."""
        self.plugin_list.clear()
        self.plugins = {}

        # Store current filter text
        current_filter = self.filter_line_edit.text() if hasattr(self, 'filter_line_edit') else ""

        # Determine plugin directory
        plugin_root = pathlib.Path(chisurf.plugins.__file__).absolute().parent

        # Find all module names
        module_infos = list(pkgutil.iter_modules(chisurf.plugins.__path__))
        module_names = [name for _, name, _ in module_infos]

        # Create a list of (module_name, order, is_disabled) tuples
        module_order_pairs = []
        for module_name in module_names:
            # Try to load the module to get its name
            try:
                module_path = f"chisurf.plugins.{module_name}"
                module = importlib.import_module(module_path)

                # Import all submodules to ensure they're properly loaded
                package_path = module.__path__ if hasattr(module, '__path__') else None
                if package_path:
                    # Temporarily add the plugin directory to sys.path for relative imports
                    original_sys_path = sys.path.copy()
                    for path in package_path:
                        if path not in sys.path:
                            sys.path.insert(0, path)

                    try:
                        for _, submodule_name, is_pkg in pkgutil.walk_packages(package_path, f"{module_path}."):
                            try:
                                importlib.import_module(submodule_name)
                            except Exception as sub_e:
                                print(f"Error importing submodule {submodule_name}: {sub_e}")
                    finally:
                        # Restore the original sys.path
                        sys.path = original_sys_path

                name = getattr(module, 'name', module_name)
                # Get the order from plugin_order, default to 0 if not set
                order = self.plugin_order.get(name, 0)

                # Check if this plugin is marked as disabled
                clean_name = name.split(':')[-1].strip() if ':' in name else name
                is_disabled = (
                        name in self.disabled_plugins
                        or module_name in self.disabled_plugins
                        or clean_name in self.disabled_plugins
                )

                module_order_pairs.append((module_name, order, is_disabled))
            except Exception:
                # If module can't be loaded, use default order and mark as not disabled
                module_order_pairs.append((module_name, 0, False))

        # Sort by disabled status (enabled first), then by order (ascending), then by module_name (alphabetically)
        module_order_pairs.sort(key=lambda x: (x[2], x[1], x[0]))

        # Extract just the module names in the sorted order
        module_names = [pair[0] for pair in module_order_pairs]

        for module_name in module_names:
            module_path = f"chisurf.plugins.{module_name}"
            try:
                module = importlib.import_module(module_path)

                # Import all submodules to ensure they're properly loaded
                package_path = module.__path__ if hasattr(module, '__path__') else None
                if package_path:
                    # Temporarily add the plugin directory to sys.path for relative imports
                    original_sys_path = sys.path.copy()
                    for path in package_path:
                        if path not in sys.path:
                            sys.path.insert(0, path)

                    try:
                        for _, submodule_name, is_pkg in pkgutil.walk_packages(package_path, f"{module_path}."):
                            try:
                                importlib.import_module(submodule_name)
                            except Exception as sub_e:
                                print(f"Error importing submodule {submodule_name}: {sub_e}")
                    finally:
                        # Restore the original sys.path
                        sys.path = original_sys_path

                name = getattr(module, 'name', module_name)

                # Check if this plugin is marked as disabled
                clean_name = name.split(':')[-1].strip() if ':' in name else name
                is_disabled = (
                        name in self.disabled_plugins
                        or module_name in self.disabled_plugins
                        or clean_name in self.disabled_plugins
                )

                # Create list item
                item = QListWidgetItem(name)
                item.setData(Qt.UserRole, module_name)

                # Set icon if available
                icon_path = plugin_root / module_name / 'icon.png'
                if hasattr(module, 'icon'):
                    item.setIcon(module.icon)
                elif icon_path.exists():
                    item.setIcon(QIcon(str(icon_path)))

                # Mark plugins based on status
                if is_disabled:
                    item.setForeground(Qt.gray)
                    item.setText(f"{name} [DISABLED]")

                # Add to list widget
                self.plugin_list.addItem(item)

                # Store plugin metadata
                plugin_path = plugin_root / module_name
                doc = read_module_docstring(plugin_path)
                if doc is None:
                    doc = "No description available."
                d = {
                    'name': name,
                    'module': module,
                    'is_disabled': is_disabled,
                    'path': str(plugin_path),
                    'doc': doc
                }
                self.plugins[module_name] = d
            except Exception as e:
                print(f"Error loading plugin {module_name}: {e}")

        # Apply current filter if any
        if hasattr(self, 'filter_line_edit') and current_filter:
            self.on_filter_text_changed(current_filter)

    def on_plugin_selected(self, current, previous):
        """Handle plugin selection in the list."""
        if current is None:
            self.current_plugin = None
            self.plugin_name_label.setText("Select a plugin")
            self.plugin_path_label.setText("Not available")
            self.disabled_checkbox.setChecked(False)
            self.toolbar_checkbox.setChecked(False)
            self.description_edit.clear()
            self.rename_button.setEnabled(False)
            return

        module_name = current.data(Qt.UserRole)
        self.current_plugin = module_name
        plugin_info = self.plugins[module_name]

        self.plugin_name_label.setText(plugin_info['name'])
        self.disabled_checkbox.setChecked(plugin_info['is_disabled'])

        # Set toolbar checkbox state
        plugin_name = plugin_info['name']
        self.toolbar_checkbox.setChecked(plugin_name in self.toolbar_plugins)

        # Display the plugin path
        self.plugin_path_label.setText(plugin_info['path'])

        # Get plugin description if available
        description = plugin_info['doc']
        self.description_edit.setText(description)

        # Enable the rename button
        self.rename_button.setEnabled(True)

    def on_disabled_changed(self, state):
        """Handle disabled checkbox state change."""
        if self.current_plugin is None:
            return

        plugin_info = self.plugins[self.current_plugin]
        plugin_name = plugin_info['name']

        if state == Qt.Checked:
            if plugin_name not in self.disabled_plugins:
                self.disabled_plugins.append(plugin_name)
            plugin_info['is_disabled'] = True
        else:
            if plugin_name in self.disabled_plugins:
                self.disabled_plugins.remove(plugin_name)
            plugin_info['is_disabled'] = False

        # Update the list item
        for i in range(self.plugin_list.count()):
            item = self.plugin_list.item(i)
            if item.data(Qt.UserRole) == self.current_plugin:
                if plugin_info['is_disabled']:
                    item.setForeground(Qt.gray)
                    item.setText(f"{plugin_name} [DISABLED]")
                else:
                    item.setText(plugin_name)

                # Re-apply current filter
                current_filter = self.filter_line_edit.text()
                if current_filter:
                    self.on_filter_text_changed(current_filter)
                break

    def on_toolbar_changed(self, state):
        """Handle toolbar checkbox state change."""
        if self.current_plugin is None:
            return

        plugin_info = self.plugins[self.current_plugin]
        plugin_name = plugin_info['name']

        if state == Qt.Checked:
            if plugin_name not in self.toolbar_plugins:
                self.toolbar_plugins.append(plugin_name)
        else:
            if plugin_name in self.toolbar_plugins:
                self.toolbar_plugins.remove(plugin_name)

    def on_hide_disabled_changed(self, state):
        """Handle hide disabled plugins checkbox state change."""
        self.hide_disabled_plugins = (state == Qt.Checked)

        # Re-apply current filter
        if hasattr(self, 'filter_line_edit'):
            current_filter = self.filter_line_edit.text()
            if current_filter:
                self.on_filter_text_changed(current_filter)


    def on_filter_text_changed(self, text):
        """Filter plugins based on the entered text."""
        filter_text = text.lower()

        for i in range(self.plugin_list.count()):
            item = self.plugin_list.item(i)
            plugin_name = item.text()
            module_name = item.data(Qt.UserRole)
            plugin_info = self.plugins.get(module_name, {})

            # Remove [DISABLED] suffix for matching
            if "[DISABLED]" in plugin_name:
                clean_name = plugin_name.replace(" [DISABLED]", "")
            else:
                clean_name = plugin_name

            # Check if the plugin name contains the filter text
            if filter_text in clean_name.lower():
                # Plugin matches filter - show normally
                item.setHidden(False)
                if plugin_info.get('is_disabled', False):
                    # Keep disabled plugins gray
                    item.setForeground(Qt.gray)
                else:
                    # Reset color for enabled plugins
                    item.setForeground(Qt.black)
            else:
                # Plugin doesn't match filter - hide it
                item.setHidden(True)

    def on_move_up(self):
        """Move the selected plugin up in the order."""
        if self.current_plugin is None:
            return

        # Get the current item and its index
        current_row = self.plugin_list.currentRow()
        if current_row <= 0:
            return  # Already at the top

        # Get the plugin name
        plugin_info = self.plugins[self.current_plugin]
        plugin_name = plugin_info['name']

        # Update the order value
        current_order = self.plugin_order.get(plugin_name, 0)
        # Find the plugin above this one
        above_item = self.plugin_list.item(current_row - 1)
        above_module_name = above_item.data(Qt.UserRole)
        above_plugin_info = self.plugins[above_module_name]
        above_plugin_name = above_plugin_info['name']
        above_order = self.plugin_order.get(above_plugin_name, 0)

        # Swap the order values
        self.plugin_order[plugin_name] = above_order - 1

        # Reload the plugins to reflect the new order
        self.load_plugins()

        # Reselect the plugin
        for i in range(self.plugin_list.count()):
            item = self.plugin_list.item(i)
            if item.data(Qt.UserRole) == self.current_plugin:
                self.plugin_list.setCurrentItem(item)
                break

    def on_move_down(self):
        """Move the selected plugin down in the order."""
        if self.current_plugin is None:
            return

        # Get the current item and its index
        current_row = self.plugin_list.currentRow()
        if current_row >= self.plugin_list.count() - 1:
            return  # Already at the bottom

        # Get the plugin name
        plugin_info = self.plugins[self.current_plugin]
        plugin_name = plugin_info['name']

        # Update the order value
        current_order = self.plugin_order.get(plugin_name, 0)
        # Find the plugin below this one
        below_item = self.plugin_list.item(current_row + 1)
        below_module_name = below_item.data(Qt.UserRole)
        below_plugin_info = self.plugins[below_module_name]
        below_plugin_name = below_plugin_info['name']
        below_order = self.plugin_order.get(below_plugin_name, 0)

        # Swap the order values
        self.plugin_order[plugin_name] = below_order + 1

        # Reload the plugins to reflect the new order
        self.load_plugins()

        # Reselect the plugin
        for i in range(self.plugin_list.count()):
            item = self.plugin_list.item(i)
            if item.data(Qt.UserRole) == self.current_plugin:
                self.plugin_list.setCurrentItem(item)
                break

    def save_settings(self):
        """Save plugin settings to the settings file."""
        # Update plugin settings
        self.plugin_settings['disabled_plugins'] = self.disabled_plugins  # Keep the key for backward compatibility
        self.plugin_settings['hide_disabled_plugins'] = self.hide_disabled_plugins  # Keep the key for backward compatibility
        self.plugin_settings['plugin_order'] = self.plugin_order
        self.plugin_settings['toolbar_plugins'] = self.toolbar_plugins

        # Update settings in chisurf
        chisurf.settings.cs_settings['plugins'] = self.plugin_settings

        # Save settings to file
        try:
            with open(chisurf.settings.chisurf_settings_file, 'w') as f:
                yaml.dump(chisurf.settings.cs_settings, f, default_flow_style=False)

            # Update the toolbar in the main window
            app = QApplication.instance()
            for widget in app.topLevelWidgets():
                if widget.__class__.__name__ == 'Main':
                    # Found the main window, update the toolbar
                    if hasattr(widget, 'load_toolbar_plugins'):
                        # Clear existing toolbar first
                        if hasattr(widget, 'plugins_toolbar'):
                            widget.plugins_toolbar.clear()
                        # Reload toolbar plugins
                        widget.load_toolbar_plugins()
                        break

            QMessageBox.information(self, "Settings Saved", "Plugin settings have been saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save settings: {e}")

    def is_admin(self):
        """Check if the application is running with administrator privileges."""
        system = platform.system()

        if system == 'Windows':
            try:
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            except:
                return False
        elif system == 'Darwin':  # macOS
            try:
                return os.geteuid() == 0
            except:
                return False
        elif system == 'Linux':
            try:
                return os.geteuid() == 0
            except:
                return False
        else:
            return False

    def run_as_admin(self, cmd):
        """Run a command with administrator privileges."""
        system = platform.system()

        if system == 'Windows':
            try:
                ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, cmd, None, 1)
                return True
            except:
                return False
        elif system == 'Darwin':  # macOS
            try:
                # Use osascript to prompt for admin password with a graphical dialog
                sudo_cmd = f'''osascript -e 'do shell script "{sys.executable} {cmd}" with administrator privileges' '''
                subprocess.Popen(sudo_cmd, shell=True)
                return True
            except:
                return False
        elif system == 'Linux':
            try:
                # Use pkexec or gksudo if available, otherwise fall back to sudo
                if shutil.which('pkexec'):
                    sudo_cmd = f"pkexec {sys.executable} {cmd}"
                elif shutil.which('gksudo'):
                    sudo_cmd = f"gksudo {sys.executable} {cmd}"
                else:
                    sudo_cmd = f"sudo {sys.executable} {cmd}"
                subprocess.Popen(sudo_cmd, shell=True)
                return True
            except:
                return False
        else:
            return False

    def import_plugin(self):
        """Import a plugin from a directory."""
        # Open a file dialog to select a plugin directory
        plugin_dir = QFileDialog.getExistingDirectory(self, "Select Plugin Directory")

        if not plugin_dir:
            return

        plugin_dir_path = pathlib.Path(plugin_dir)
        plugin_name = plugin_dir_path.name

        # Check if the selected directory is a valid plugin
        init_file = plugin_dir_path / "__init__.py"
        if not init_file.exists():
            QMessageBox.critical(self, "Error", f"The selected directory is not a valid plugin. Missing __init__.py file.")
            return

        # Determine the destination directory
        plugin_root = pathlib.Path(chisurf.plugins.__file__).absolute().parent
        destination_dir = plugin_root / plugin_name

        # Check if plugin already exists
        if destination_dir.exists():
            reply = QMessageBox.question(self, "Plugin Exists", 
                                        f"A plugin named '{plugin_name}' already exists. Do you want to overwrite it?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        # Check if we need elevated privileges based on the operating system
        system = platform.system()
        needs_elevation = False
        elevation_message = ""

        if system == 'Windows':
            # Check for Windows Program Files
            program_files_paths = [
                os.environ.get("ProgramFiles", "C:\\Program Files"),
                os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
            ]
            needs_elevation = any(str(plugin_root).startswith(pf) for pf in program_files_paths)
            elevation_message = "The plugin directory is in Windows Program Files and requires administrator privileges to modify."

        elif system == 'Darwin':  # macOS
            # Check for macOS protected locations
            protected_paths = [
                '/Applications',
                '/System',
                '/Library',
                '/usr/local'
            ]
            needs_elevation = any(str(plugin_root).startswith(pf) for pf in protected_paths)
            elevation_message = "The plugin directory is in a macOS system location and requires administrator privileges to modify."

        elif system == 'Linux':
            # Check for Linux protected locations
            protected_paths = [
                '/usr',
                '/usr/share',
                '/usr/local',
                '/opt'
            ]
            needs_elevation = any(str(plugin_root).startswith(pf) for pf in protected_paths)
            elevation_message = "The plugin directory is in a Linux system location and requires administrator privileges to modify."

        if needs_elevation and not self.is_admin():
            reply = QMessageBox.question(self, "Elevation Required", 
                                        f"{elevation_message} "
                                        "Do you want to restart the application with administrator privileges?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Construct command to run on restart
                import_cmd = f"-c \"import chisurf; from chisurf.plugins.plugin_manager import PluginManagerWidget; " \
                            f"w = PluginManagerWidget(); w.import_plugin_elevated('{plugin_dir}', '{plugin_name}')\""
                self.run_as_admin(import_cmd)
                return
            else:
                return

        # Copy the plugin
        try:
            # Remove destination if it exists
            if destination_dir.exists():
                shutil.rmtree(destination_dir)

            # Copy the plugin directory
            shutil.copytree(plugin_dir_path, destination_dir)

            # Refresh the plugin list
            self.load_plugins()

            QMessageBox.information(self, "Plugin Imported", f"Plugin '{plugin_name}' has been imported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not import plugin: {e}")

    def import_plugin_elevated(self, plugin_dir, plugin_name):
        """Import a plugin with elevated privileges (called after privilege elevation prompt)."""
        plugin_dir_path = pathlib.Path(plugin_dir)
        plugin_root = pathlib.Path(chisurf.plugins.__file__).absolute().parent
        destination_dir = plugin_root / plugin_name

        try:
            # Remove destination if it exists
            if destination_dir.exists():
                shutil.rmtree(destination_dir)

            # Copy the plugin directory
            shutil.copytree(plugin_dir_path, destination_dir)

            QMessageBox.information(self, "Plugin Imported", f"Plugin '{plugin_name}' has been imported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not import plugin: {e}")

    def on_rename_plugin(self):
        """Handle renaming a plugin."""
        if self.current_plugin is None:
            return

        plugin_info = self.plugins[self.current_plugin]
        current_name = plugin_info['name']
        plugin_path = plugin_info['path']

        # Show input dialog to get new name
        new_name, ok = QInputDialog.getText(
            self, 
            "Rename Plugin", 
            "Enter new plugin name:",
            text=current_name
        )

        if not ok or not new_name or new_name == current_name:
            return

        # Check if we need elevated privileges
        plugin_dir_path = pathlib.Path(plugin_path)
        init_file = plugin_dir_path / "__init__.py"

        # Check if the file exists
        if not init_file.exists():
            QMessageBox.critical(self, "Error", f"Could not find __init__.py in {plugin_path}")
            return

        # Check if we need elevated privileges based on the operating system
        system = platform.system()
        needs_elevation = False
        elevation_message = ""

        if system == 'Windows':
            # Check for Windows Program Files
            program_files_paths = [
                os.environ.get("ProgramFiles", "C:\\Program Files"),
                os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
            ]
            needs_elevation = any(str(plugin_dir_path).startswith(pf) for pf in program_files_paths)
            elevation_message = "The plugin directory is in Windows Program Files and requires administrator privileges to modify."

        elif system == 'Darwin':  # macOS
            # Check for macOS protected locations
            protected_paths = [
                '/Applications',
                '/System',
                '/Library',
                '/usr/local'
            ]
            needs_elevation = any(str(plugin_dir_path).startswith(pf) for pf in protected_paths)
            elevation_message = "The plugin directory is in a macOS system location and requires administrator privileges to modify."

        elif system == 'Linux':
            # Check for Linux protected locations
            protected_paths = [
                '/usr',
                '/usr/share',
                '/usr/local',
                '/opt'
            ]
            needs_elevation = any(str(plugin_dir_path).startswith(pf) for pf in protected_paths)
            elevation_message = "The plugin directory is in a Linux system location and requires administrator privileges to modify."

        if needs_elevation and not self.is_admin():
            reply = QMessageBox.question(self, "Elevation Required", 
                                        f"{elevation_message} "
                                        "Do you want to restart the application with administrator privileges?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                # Construct command to run on restart
                rename_cmd = f"-c \"import chisurf; from chisurf.plugins.plugin_manager import PluginManagerWidget; " \
                            f"w = PluginManagerWidget(); w.rename_plugin_elevated('{plugin_path}', '{current_name}', '{new_name}')\""
                self.run_as_admin(rename_cmd)
                return
            else:
                return

        # If we don't need elevation or we already have admin rights, rename directly
        try:
            self.rename_plugin_file(init_file, current_name, new_name)

            # Update the UI
            self.load_plugins()

            # Find and select the renamed plugin
            for i in range(self.plugin_list.count()):
                item = self.plugin_list.item(i)
                if item.data(Qt.UserRole) == self.current_plugin:
                    self.plugin_list.setCurrentItem(item)
                    break

            QMessageBox.information(self, "Plugin Renamed", f"Plugin has been renamed from '{current_name}' to '{new_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not rename plugin: {e}")

    def rename_plugin_elevated(self, plugin_path, old_name, new_name):
        """Rename a plugin with elevated privileges (called after privilege elevation prompt)."""
        plugin_dir_path = pathlib.Path(plugin_path)
        init_file = plugin_dir_path / "__init__.py"

        try:
            self.rename_plugin_file(init_file, old_name, new_name)
            QMessageBox.information(self, "Plugin Renamed", f"Plugin has been renamed from '{old_name}' to '{new_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not rename plugin: {e}")

    def rename_plugin_file(self, init_file, old_name, new_name):
        """Modify the __init__.py file to change the plugin name."""
        # Read the file content
        content = init_file.read_text(encoding="utf-8")

        # Look for the name variable assignment
        import re
        # Match patterns like: name = "Tools:Plugin Manager" or name = 'Games:Pong'
        pattern = r'name\s*=\s*[\'"]([^\'"]*)[\'"]'

        # Replace the name
        new_content = re.sub(pattern, f'name = "{new_name}"', content)

        # Check if we actually made a replacement
        if new_content == content:
            raise ValueError("Could not find name variable in __init__.py")

        # Write the modified content back to the file
        init_file.write_text(new_content, encoding="utf-8")

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the PluginManagerWidget class
    window = PluginManagerWidget()
    # Show the window
    window.show()
