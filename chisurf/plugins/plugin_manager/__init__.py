"""
Plugin Manager for ChiSurf

This plugin allows you to manage all installed plugins in ChiSurf. You can:
- View all available plugins
- Enable or disable plugins
- Enable or disable plugin icons
- View plugin descriptions

The plugin manager provides a convenient interface for configuring how plugins
appear in the ChiSurf menu system.
"""

import sys
import pathlib
import importlib
import pkgutil
import yaml
import ast
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QCheckBox,
    QMessageBox, QGroupBox, QScrollArea, QSplitter, QTextEdit
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
        self.icons_enabled = self.plugin_settings.get('icons_enabled', True)
        self.plugin_order = self.plugin_settings.get('plugin_order', {})

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
        name_layout.addStretch()
        details_layout.addLayout(name_layout)

        # Plugin status
        status_layout = QHBoxLayout()
        self.disabled_checkbox = QCheckBox("Disable plugin")
        self.disabled_checkbox.stateChanged.connect(self.on_disabled_changed)
        status_layout.addWidget(self.disabled_checkbox)
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

        # Enable icons checkbox
        self.icons_checkbox = QCheckBox("Enable plugin icons")
        self.icons_checkbox.setChecked(self.icons_enabled)
        self.icons_checkbox.stateChanged.connect(self.on_icons_enabled_changed)
        settings_layout.addWidget(self.icons_checkbox)

        main_layout.addWidget(settings_group)

        # Create buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)

        refresh_button = QPushButton("Refresh Plugin List")
        refresh_button.clicked.connect(self.load_plugins)
        button_layout.addWidget(refresh_button)

        main_layout.addLayout(button_layout)

        # Load plugins
        self.load_plugins()

        # Current selected plugin
        self.current_plugin = None

    def load_plugins(self):
        """Load all available plugins, sorted by custom order or module name, and display them in the list."""
        self.plugin_list.clear()
        self.plugins = {}

        # Determine plugin directory
        plugin_root = pathlib.Path(chisurf.plugins.__file__).absolute().parent

        # Find all module names
        module_infos = list(pkgutil.iter_modules(chisurf.plugins.__path__))
        module_names = [name for _, name, _ in module_infos]

        # Create a list of (module_name, order) tuples
        module_order_pairs = []
        for module_name in module_names:
            # Try to load the module to get its name
            try:
                module_path = f"chisurf.plugins.{module_name}"
                module = importlib.import_module(module_path)
                name = getattr(module, 'name', module_name)
                # Get the order from plugin_order, default to 0 if not set
                order = self.plugin_order.get(name, 0)
                module_order_pairs.append((module_name, order))
            except Exception:
                # If module can't be loaded, use default order
                module_order_pairs.append((module_name, 0))

        # Sort by order (ascending) and then by module_name (alphabetically)
        module_order_pairs.sort(key=lambda x: (x[1], x[0]))

        # Extract just the module names in the sorted order
        module_names = [pair[0] for pair in module_order_pairs]

        for module_name in module_names:
            module_path = f"chisurf.plugins.{module_name}"
            try:
                module = importlib.import_module(module_path)
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
                if self.icons_enabled:
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

    def on_plugin_selected(self, current, previous):
        """Handle plugin selection in the list."""
        if current is None:
            self.current_plugin = None
            self.plugin_name_label.setText("Select a plugin")
            self.plugin_path_label.setText("Not available")
            self.disabled_checkbox.setChecked(False)
            self.description_edit.clear()
            return

        module_name = current.data(Qt.UserRole)
        self.current_plugin = module_name
        plugin_info = self.plugins[module_name]

        self.plugin_name_label.setText(plugin_info['name'])
        self.disabled_checkbox.setChecked(plugin_info['is_disabled'])

        # Display the plugin path
        self.plugin_path_label.setText(plugin_info['path'])

        # Get plugin description if available
        description = plugin_info['doc']
        self.description_edit.setText(description)

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
                break

    def on_hide_disabled_changed(self, state):
        """Handle hide disabled plugins checkbox state change."""
        self.hide_disabled_plugins = (state == Qt.Checked)

    def on_icons_enabled_changed(self, state):
        """Handle icons enabled checkbox state change."""
        self.icons_enabled = (state == Qt.Checked)
        self.load_plugins()  # Reload plugins to update icons

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
        self.plugin_settings['icons_enabled'] = self.icons_enabled
        self.plugin_settings['plugin_order'] = self.plugin_order

        # Update settings in chisurf
        chisurf.settings.cs_settings['plugins'] = self.plugin_settings

        # Save settings to file
        try:
            with open(chisurf.settings.chisurf_settings_file, 'w') as f:
                yaml.dump(chisurf.settings.cs_settings, f, default_flow_style=False)
            QMessageBox.information(self, "Settings Saved", "Plugin settings have been saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save settings: {e}")

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the PluginManagerWidget class
    window = PluginManagerWidget()
    # Show the window
    window.show()
