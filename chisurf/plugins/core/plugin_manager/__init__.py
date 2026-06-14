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

import ast
import base64
import ctypes
import hashlib
import importlib
import json
import logging
import os
import pathlib
import pkgutil
import platform
import shutil
import subprocess
import sys

import yaml
from qtpy import QtCore, QtGui
from qtpy.QtCore import QSize, Qt, QUrl
from qtpy.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import chisurf as cs
import chisurf.core.settings
import chisurf.plugins
from chisurf.core.plugin import load_manifest
from chisurf.core.settings import ai_settings

logger = logging.getLogger(__name__)


class AIIconRateLimitError(RuntimeError):
    """Raised when an AI icon provider remains rate-limited after retries."""


try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

# Import enhanced icon utilities
try:
    from ..icon_utils import (
        create_plugin_icon_with_fallback,
        plugin_icon_path,
    )
except ImportError:
    # Fallback if icon_utils is not available
    def create_plugin_icon_with_fallback(module, package_dir, size=64, manifest=None):
        """Fallback icon creation using existing system."""
        from pathlib import Path
        package_dir = Path(package_dir)
        if manifest is not None and getattr(manifest, "icon", None):
            icon_path = package_dir / manifest.icon
            if icon_path.exists():
                return QIcon(str(icon_path))
        if hasattr(module, 'icon') and isinstance(module.icon, QIcon):
            return module.icon
        icon_path = package_dir / 'icon.png'
        if icon_path.exists():
            return QIcon(str(icon_path))
        return QIcon()

    def plugin_icon_path(package_dir):
        """Return the default plugin icon path."""
        return pathlib.Path(package_dir) / "icon.png"

# Define the plugin name - this will appear in the Plugins menu
name = "Setup:Plugins"


def read_module_docstring(package_path: pathlib.Path) -> str | None:
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


class IconGenerationDialog(QDialog):
    """Dialog for customizing the prompt and previewing generated icon."""
    def __init__(self, parent, plugin_info, plugin_manager):
        super().__init__(parent)
        self.plugin_info = plugin_info
        self.plugin_manager = plugin_manager
        self.generated_pixmap = None
        self.generated_source = None

        self.setWindowTitle("Generate Plugin Icon")
        self.resize(550, 250)

        main_layout = QHBoxLayout(self)

        # Left side: Prompt and Buttons
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Prompt for the AI image generator:"))

        self.prompt_edit = QTextEdit()
        default_prompt = self.plugin_manager._ai_icon_prompt(plugin_info)
        self.prompt_edit.setPlainText(default_prompt)
        left_layout.addWidget(self.prompt_edit)

        # Action Buttons
        buttons_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.on_generate)

        self.accept_btn = QPushButton("Accept")
        self.accept_btn.setEnabled(False)
        self.accept_btn.clicked.connect(self.accept)

        self.discard_btn = QPushButton("Discard")
        self.discard_btn.clicked.connect(self.reject)

        buttons_layout.addWidget(self.generate_btn)
        buttons_layout.addWidget(self.accept_btn)
        buttons_layout.addWidget(self.discard_btn)
        left_layout.addLayout(buttons_layout)

        main_layout.addLayout(left_layout, stretch=2)

        # Right side: Preview area
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Preview (128x128):"))

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(128, 128)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.preview_label)
        right_layout.addStretch()

        main_layout.addLayout(right_layout, stretch=1)

    def on_generate(self):
        """Request the icon from provider, scale to 128x128, and show preview."""
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")
        QApplication.processEvents()

        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            prompt = self.plugin_manager._ai_icon_prompt(self.plugin_info)

        try:
            ai_error = None
            source_info = "local generator"
            pixmap = QPixmap()

            if self.plugin_manager._ai_icon_generation_available():
                try:
                    image_bytes = self.plugin_manager._request_ai_generated_icon_bytes(self.plugin_info, prompt=prompt)
                    if pixmap.loadFromData(image_bytes):
                        scaled = pixmap.scaled(
                            128,
                            128,
                            Qt.KeepAspectRatioByExpanding,
                            Qt.SmoothTransformation,
                        )
                        crop_x = max(0, (scaled.width() - 128) // 2)
                        crop_y = max(0, (scaled.height() - 128) // 2)
                        pixmap = scaled.copy(crop_x, crop_y, 128, 128)
                        source_info = "AI Settings"
                    else:
                        raise ValueError("AI provider returned unsupported image data")
                except Exception as e:
                    ai_error = e
                    if isinstance(e, AIIconRateLimitError):
                        logger.warning(f"AI Icon generation rate-limited: {e}")
                    else:
                        logger.error(f"AI Icon generation failed: {e}", exc_info=True)

            if ai_error is not None or not self.plugin_manager._ai_icon_generation_available():
                pixmap = self.plugin_manager._create_generated_icon(
                    self.plugin_info['name'],
                    self.plugin_info.get('doc', ''),
                    size=128,
                )
                if ai_error is not None:
                    source_info = "local fallback"
                else:
                    source_info = "local generator"

            self.generated_pixmap = pixmap
            self.generated_source = source_info

            self.preview_label.setPixmap(pixmap)
            self.accept_btn.setEnabled(True)

        except Exception as e:
            logger.error(f"Error during icon generation preview: {e}", exc_info=True)
        finally:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Regenerate")


@persist_plugin_state("plugin_manager")
class PluginManagerWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plugin Manager")
        self.resize(800, 400)

        # Get plugin settings
        self.plugin_settings = cs.core.settings.cs_settings.get('plugins', {})
        self.disabled_plugins = self.plugin_settings.get('disabled_plugins', [])  # Keep the key for backward compatibility
        self.hide_disabled_plugins = self.plugin_settings.get('hide_disabled_plugins', True)  # Keep the key for backward compatibility
        self.plugin_order = self.plugin_settings.get('plugin_order', {})
        self.toolbar_plugins = self.plugin_settings.get('toolbar_plugins', [])
        icon_generation_settings = self.plugin_settings.get('icon_generation', {})
        if not isinstance(icon_generation_settings, dict):
            icon_generation_settings = {}
        self.icon_generation_settings = icon_generation_settings
        statefulness_settings = self.plugin_settings.get('statefulness', {})
        if not isinstance(statefulness_settings, dict):
            statefulness_settings = {}
        self.statefulness_mode = statefulness_settings.get('mode', 'plugin_default')
        self.statefulness_overrides = statefulness_settings.get('per_plugin', {})
        if not isinstance(self.statefulness_overrides, dict):
            self.statefulness_overrides = {}

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

        # Statefulness override
        self.statefulness_checkbox = QCheckBox("Remember window state")
        self.statefulness_checkbox.setTristate(True)
        self.statefulness_checkbox.stateChanged.connect(self.on_statefulness_changed)
        status_layout.addWidget(self.statefulness_checkbox)

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

        icon_group = QGroupBox("Plugin Icon")
        icon_layout = QVBoxLayout(icon_group)

        icon_preview_layout = QHBoxLayout()
        self.icon_preview_label = QLabel()
        self.icon_preview_label.setFixedSize(72, 72)
        self.icon_preview_label.setAlignment(Qt.AlignCenter)
        self.icon_preview_label.setStyleSheet("QLabel { border: 1px solid #b8b8b8; background: #f6f6f6; }")
        icon_preview_layout.addWidget(self.icon_preview_label)

        icon_source_layout = QVBoxLayout()
        icon_path_layout = QHBoxLayout()
        icon_path_layout.addWidget(QLabel("Image:"))
        self.icon_path_edit = QLineEdit()
        self.icon_path_edit.setPlaceholderText("Select an image or use generated icon.png")
        icon_path_layout.addWidget(self.icon_path_edit)
        icon_source_layout.addLayout(icon_path_layout)

        icon_button_layout = QHBoxLayout()
        self.browse_icon_button = QPushButton("Choose Image")
        self.browse_icon_button.clicked.connect(self.on_choose_icon_image)
        self.browse_icon_button.setEnabled(False)
        icon_button_layout.addWidget(self.browse_icon_button)

        self.apply_icon_button = QPushButton("Use Image")
        self.apply_icon_button.clicked.connect(self.on_apply_icon_image)
        self.apply_icon_button.setEnabled(False)
        icon_button_layout.addWidget(self.apply_icon_button)

        self.generate_icon_button = QPushButton("Generate")
        self.generate_icon_button.clicked.connect(self.on_generate_icon)
        self.generate_icon_button.setEnabled(False)
        icon_button_layout.addWidget(self.generate_icon_button)

        self.edit_icon_button = QPushButton("Edit")
        self.edit_icon_button.clicked.connect(self.on_edit_icon)
        self.edit_icon_button.setEnabled(False)
        icon_button_layout.addWidget(self.edit_icon_button)

        self.clear_icon_button = QPushButton("Clear")
        self.clear_icon_button.clicked.connect(self.on_clear_icon)
        self.clear_icon_button.setEnabled(False)
        icon_button_layout.addWidget(self.clear_icon_button)
        icon_button_layout.addStretch()
        icon_source_layout.addLayout(icon_button_layout)
        icon_preview_layout.addLayout(icon_source_layout)
        icon_layout.addLayout(icon_preview_layout)

        generation_layout = QHBoxLayout()
        generation_layout.addWidget(QLabel("Provider:"))
        self.icon_provider_combo = QComboBox()
        for display_name, (key, _url, _api_url, _env) in ai_settings.PROVIDERS.items():
            self.icon_provider_combo.addItem(display_name, key)
        self.icon_provider_combo.currentIndexChanged.connect(self.on_icon_provider_changed)
        generation_layout.addWidget(self.icon_provider_combo)

        generation_layout.addWidget(QLabel("Endpoint:"))
        self.icon_endpoint_edit = QLineEdit()
        self.icon_endpoint_edit.setPlaceholderText("https://api.mistral.ai/v1")
        generation_layout.addWidget(self.icon_endpoint_edit)

        generation_layout.addWidget(QLabel("Image model:"))
        self.icon_model_edit = QLineEdit()
        self.icon_model_edit.setPlaceholderText("mistral-medium-latest")
        generation_layout.addWidget(self.icon_model_edit)

        icon_layout.addLayout(generation_layout)
        self._load_icon_generation_controls()

        details_layout.addWidget(icon_group)

        splitter.addWidget(details_group)

        # Create settings group
        settings_group = QGroupBox("Global Plugin Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Hide disabled plugins checkbox
        self.hide_disabled_checkbox = QCheckBox("Hide disabled plugins")
        self.hide_disabled_checkbox.setChecked(self.hide_disabled_plugins)
        self.hide_disabled_checkbox.stateChanged.connect(self.on_hide_disabled_changed)
        settings_layout.addWidget(self.hide_disabled_checkbox)

        mode_layout = QHBoxLayout()
        mode_label = QLabel("Plugin statefulness:")
        mode_layout.addWidget(mode_label)
        self.statefulness_mode_combo = QComboBox()
        self.statefulness_mode_combo.addItems([
            "Plugin default",
            "Enable all",
            "Disable all",
        ])
        mode_index = {
            "plugin_default": 0,
            "enabled": 1,
            "force_enabled": 1,
            "disabled": 2,
            "force_disabled": 2,
        }.get(str(self.statefulness_mode).lower(), 0)
        self.statefulness_mode_combo.setCurrentIndex(mode_index)
        self.statefulness_mode_combo.currentTextChanged.connect(self.on_statefulness_mode_changed)
        mode_layout.addWidget(self.statefulness_mode_combo)
        mode_layout.addStretch()
        settings_layout.addLayout(mode_layout)

        statefulness_hint = QLabel(
            "Per-plugin checkbox: checked = force remember, "
            "unchecked = force forget, partial = plugin default."
        )
        statefulness_hint.setWordWrap(True)
        settings_layout.addWidget(statefulness_hint)

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

    def _statefulness_key(
        self,
        name: str,
        module_name: str,
        manifest=None,
    ) -> str:
        """Return the settings key used for a plugin statefulness override."""
        if manifest is not None:
            return manifest.id
        clean_name = name.split(':')[-1].strip() if ':' in name else name
        return clean_name or module_name

    def _statefulness_override_state(self, key: str) -> Qt.CheckState:
        """Return the checkbox state for a plugin statefulness override."""
        if key not in self.statefulness_overrides:
            return Qt.PartiallyChecked
        return Qt.Checked if bool(self.statefulness_overrides[key]) else Qt.Unchecked

    def _statefulness_summary(self, key: str) -> str:
        """Return a short statefulness summary for the plugin manager."""
        state = self._statefulness_override_state(key)
        if state == Qt.Checked:
            return "stateful"
        if state == Qt.Unchecked:
            return "stateless"
        return "plugin default"

    def load_plugins(self):
        """Load all available plugins, sorted by custom order or module name, and display them in the list."""
        self.plugin_list.clear()
        self.plugins = {}

        # Store current filter text
        current_filter = self.filter_line_edit.text() if hasattr(self, 'filter_line_edit') else ""

        # Determine built-in plugin directory (used for backward-compatible paths)
        plugin_root = pathlib.Path(cs.plugins.__file__).absolute().parent

        # Discover plugins (built-in + user, including nested subpackages)
        try:
            plugin_infos = list(cs.plugins.iter_plugins())
        except Exception:
            plugin_infos = []

        # Create a list of (info, order, is_disabled) tuples
        module_order_pairs = []
        for info in plugin_infos:
            module_path = info.get('module_path')
            module_name = info.get('module_name') or ''
            plugin_name = info.get('plugin_name') or module_name
            source = info.get('source') or 'built-in'

            # Get the order from plugin_order, default to 0 if not set
            order = self.plugin_order.get(plugin_name, 0)

            # Check if this plugin is marked as disabled
            clean_name = plugin_name.split(':')[-1].strip() if ':' in plugin_name else plugin_name
            is_disabled = (
                plugin_name in self.disabled_plugins
                or module_name in self.disabled_plugins
                or clean_name in self.disabled_plugins
            )

            module_order_pairs.append((info, order, is_disabled, plugin_name, module_name, source))

        # Sort by disabled status (enabled first), then by order (ascending), then by plugin_name (alphabetically)
        module_order_pairs.sort(key=lambda x: (x[2], x[1], x[3]))

        for info, _order, is_disabled, plugin_name, module_name, source in module_order_pairs:
            module_path = info.get('module_path')
            package_dir = info.get('package_dir') or plugin_root / module_name
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

                name = getattr(module, 'name', plugin_name)
                manifest = load_manifest(pathlib.Path(package_dir) / "manifest.json")
                statefulness_key = self._statefulness_key(name, module_name, manifest)

                # Re-evaluate disabled status based on the resolved name
                clean_name = name.split(':')[-1].strip() if ':' in name else name
                is_disabled = (
                    name in self.disabled_plugins
                    or module_name in self.disabled_plugins
                    or clean_name in self.disabled_plugins
                )

                # Create list item
                display_name = f"{name} [{source}] ({self._statefulness_summary(statefulness_key)})"
                item = QListWidgetItem(display_name)
                # Track plugins by full module path so nested packages are unique
                item.setData(Qt.UserRole, module_path)
                item.setData(Qt.UserRole + 1, source)

                # Set icon using enhanced icon system
                try:
                    icon = create_plugin_icon_with_fallback(module, package_dir, size=32, manifest=manifest)
                    item.setIcon(icon)
                except Exception:
                    # Fallback to original system if enhanced system fails
                    try:
                        if hasattr(module, 'icon'):
                            if isinstance(module.icon, QIcon):
                                item.setIcon(module.icon)
                            elif isinstance(module.icon, str):
                                # Try to create a simple text icon as fallback
                                from qtpy.QtGui import QColor, QFont, QPainter, QPixmap

                                pm = QPixmap(32, 32)
                                pm.fill(Qt.transparent)
                                painter = QPainter(pm)
                                painter.setRenderHint(QPainter.Antialiasing, True)
                                painter.setRenderHint(QPainter.TextAntialiasing, True)

                                # Check if it's an emoji
                                if any(ord(char) > 0x1F000 for char in module.icon):
                                    font = QFont("Segoe UI Emoji", 16)
                                else:
                                    font = QFont("Arial", 12, QFont.Bold)

                                painter.setFont(font)
                                painter.setPen(QColor(0, 0, 0))
                                rect = pm.rect()
                                painter.drawText(rect, Qt.AlignCenter, module.icon)
                                painter.end()

                                item.setIcon(QIcon(pm))
                            else:
                                item.setIcon(QIcon())
                        else:
                            # Check for icon.png file
                            icon_path = pathlib.Path(package_dir) / 'icon.png'
                            if icon_path.exists():
                                item.setIcon(QIcon(str(icon_path)))
                    except Exception:
                        # Ultimate fallback - empty icon
                        item.setIcon(QIcon())

                # Mark plugins based on status
                if is_disabled:
                    item.setForeground(Qt.gray)
                    item.setText(
                        f"{name} [DISABLED] [{source}] "
                        f"({self._statefulness_summary(statefulness_key)})"
                    )

                # Add to list widget
                self.plugin_list.addItem(item)

                # Store plugin metadata
                plugin_path = pathlib.Path(package_dir)
                doc = read_module_docstring(plugin_path)
                if doc is None:
                    doc = "No description available."
                d = {
                    'name': name,
                    'module': module,
                    'is_disabled': is_disabled,
                    'path': str(plugin_path),
                    'doc': doc,
                    'manifest': manifest,
                    'icon_path': str(plugin_icon_path(plugin_path)),
                    'statefulness_key': statefulness_key,
                    'statefulness_default': bool(manifest.statefulness.enabled) if manifest is not None else False,
                    'statefulness_override': self._statefulness_override_state(statefulness_key),
                    'source': source,
                }
                self.plugins[module_path] = d
            except Exception as e:
                print(f"Error loading plugin {module_path}: {e}")

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
            self.statefulness_checkbox.setCheckState(Qt.PartiallyChecked)
            self.description_edit.clear()
            self.rename_button.setEnabled(False)
            self._set_icon_controls_enabled(False)
            self.icon_path_edit.clear()
            self.icon_preview_label.clear()
            return

        module_name = current.data(Qt.UserRole)
        self.current_plugin = module_name
        plugin_info = self.plugins[module_name]

        self.plugin_name_label.setText(plugin_info['name'])
        self.disabled_checkbox.setChecked(plugin_info['is_disabled'])

        # Set toolbar checkbox state
        plugin_name = plugin_info['name']
        self.toolbar_checkbox.setChecked(plugin_name in self.toolbar_plugins)

        # Set statefulness override state
        statefulness_key = plugin_info.get('statefulness_key', plugin_name)
        self.statefulness_checkbox.setCheckState(
            self._statefulness_override_state(statefulness_key)
        )

        # Display the plugin path
        self.plugin_path_label.setText(plugin_info['path'])

        # Get plugin description if available
        description = plugin_info['doc']
        statefulness_key = plugin_info.get('statefulness_key', plugin_name)
        description = (
            f"{description}\n\nStatefulness: "
            f"{self._statefulness_summary(statefulness_key)}"
        )
        self.description_edit.setText(description)

        # Enable the rename button
        self.rename_button.setEnabled(True)
        self._set_icon_controls_enabled(True)
        self._refresh_icon_controls(plugin_info)

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
                statefulness_key = plugin_info.get('statefulness_key', plugin_name)
                if plugin_info['is_disabled']:
                    item.setForeground(Qt.gray)
                    item.setText(
                        f"{plugin_name} [DISABLED] "
                        f"({self._statefulness_summary(statefulness_key)})"
                    )
                else:
                    item.setText(
                        f"{plugin_name} "
                        f"({self._statefulness_summary(statefulness_key)})"
                    )

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

    def on_statefulness_changed(self, state):
        """Handle per-plugin statefulness override changes."""
        if self.current_plugin is None:
            return

        plugin_info = self.plugins[self.current_plugin]
        key = plugin_info.get('statefulness_key') or plugin_info['name']
        if state == Qt.PartiallyChecked:
            self.statefulness_overrides.pop(key, None)
        else:
            self.statefulness_overrides[key] = state == Qt.Checked
        plugin_info['statefulness_override'] = state

        for i in range(self.plugin_list.count()):
            item = self.plugin_list.item(i)
            if item.data(Qt.UserRole) == self.current_plugin:
                info = self.plugins[self.current_plugin]
                item.setText(
                    f"{info['name']} [{info.get('source', '')}] "
                    f"({self._statefulness_summary(key)})"
                )
                if info.get('is_disabled', False):
                    item.setForeground(Qt.gray)
                break

    def on_hide_disabled_changed(self, state):
        """Handle hide disabled plugins checkbox state change."""
        self.hide_disabled_plugins = (state == Qt.Checked)

        # Re-apply current filter
        if hasattr(self, 'filter_line_edit'):
            current_filter = self.filter_line_edit.text()
            if current_filter:
                self.on_filter_text_changed(current_filter)

    def on_statefulness_mode_changed(self, text):
        """Handle global plugin statefulness mode changes."""
        mode_by_text = {
            "Plugin default": "plugin_default",
            "Enable all": "enabled",
            "Disable all": "disabled",
        }
        self.statefulness_mode = mode_by_text.get(text, "plugin_default")

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
        self.plugin_settings['icon_generation'] = {
            'provider': self.icon_provider_combo.currentData() if hasattr(self, 'icon_provider_combo') else 'mistral',
            'endpoint': self.icon_endpoint_edit.text().strip() if hasattr(self, 'icon_endpoint_edit') else '',
            'image_model': self.icon_model_edit.text().strip() if hasattr(self, 'icon_model_edit') else '',
        }
        self.plugin_settings['statefulness'] = {
            'mode': self.statefulness_mode,
            'per_plugin': self.statefulness_overrides,
        }

        # Update settings in cs
        cs.core.settings.cs_settings['plugins'] = self.plugin_settings

        # Save settings to file
        try:
            with open(cs.core.settings.chisurf_settings_file, 'w') as f:
                yaml.dump(cs.core.settings.cs_settings, f, default_flow_style=False)

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
            QMessageBox.critical(self, "Error", "The selected directory is not a valid plugin. Missing __init__.py file.")
            return

        # Determine the destination directory
        plugin_root = pathlib.Path(cs.plugins.__file__).absolute().parent
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
                import_cmd = f"-c \"import chisurf; from chisurf.plugins.core.plugin_manager import PluginManagerWidget; " \
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
        plugin_root = pathlib.Path(cs.plugins.__file__).absolute().parent
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
                rename_cmd = f"-c \"import chisurf; from chisurf.plugins.core.plugin_manager import PluginManagerWidget; " \
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

    def on_choose_icon_image(self):
        """Select an image file for the current plugin icon."""
        if self.current_plugin is None:
            return
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Plugin Icon Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.svg *.ico);;All Files (*)",
        )
        if image_path:
            self.icon_path_edit.setText(image_path)

    def on_apply_icon_image(self):
        """Copy the selected image into the plugin as its canonical icon."""
        if self.current_plugin is None:
            return
        source = pathlib.Path(self.icon_path_edit.text()).expanduser()
        if not source.exists():
            QMessageBox.critical(self, "Icon Error", f"Image file does not exist: {source}")
            return
        try:
            path = self._write_icon_from_image(source)
            self._set_manifest_icon("icon.png")
            self._reload_icon_for_current_plugin(path)
            QMessageBox.information(self, "Icon Updated", f"Plugin icon updated:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Icon Error", f"Could not update icon: {e}")

    def on_generate_icon(self):
        """Generate an icon for the current plugin."""
        if self.current_plugin is None:
            return
        plugin_info = self.plugins[self.current_plugin]
        
        dialog = IconGenerationDialog(self, plugin_info, self)
        if dialog.exec_() != QDialog.Accepted or dialog.generated_pixmap is None:
            logger.info("Icon generation discarded or cancelled.")
            return

        try:
            output_path = pathlib.Path(plugin_info['icon_path'])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not dialog.generated_pixmap.save(str(output_path), "PNG"):
                raise OSError(f"Could not write icon image to {output_path}")

            self._set_manifest_icon("icon.png")
            self._reload_icon_for_current_plugin(output_path)
            logger.info(f"Plugin icon generated with {dialog.generated_source} and saved to: {output_path}")
        except Exception as e:
            logger.error(f"Could not save generated icon: {e}", exc_info=True)

    def on_edit_icon(self):
        """Open the current plugin icon in the system image editor."""
        if self.current_plugin is None:
            return
        icon_path = pathlib.Path(self.plugins[self.current_plugin]['icon_path'])
        if not icon_path.exists():
            QMessageBox.information(
                self,
                "No Icon",
                "No editable icon.png exists for this plugin. Choose or generate an icon first.",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(icon_path)))

    def on_clear_icon(self):
        """Remove the generated icon image for the current plugin."""
        if self.current_plugin is None:
            return
        plugin_info = self.plugins[self.current_plugin]
        icon_path = pathlib.Path(plugin_info['icon_path'])
        if icon_path.exists():
            reply = QMessageBox.question(
                self,
                "Clear Icon",
                f"Remove {icon_path.name} from this plugin?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
            icon_path.unlink()
        self._set_manifest_icon(None)
        self._reload_icon_for_current_plugin(icon_path)

    def on_icon_provider_changed(self, index):
        """Update endpoint and model defaults when the icon provider changes."""
        provider = self.icon_provider_combo.itemData(index)
        endpoint, model = self._default_icon_generation_values(provider)
        self.icon_endpoint_edit.setText(endpoint)
        self.icon_model_edit.setText(model)

    def _load_icon_generation_controls(self):
        """Load saved icon generation provider, endpoint, and model controls."""
        provider = self.icon_generation_settings.get('provider') or 'mistral'
        idx = self.icon_provider_combo.findData(provider)
        if idx < 0:
            idx = self.icon_provider_combo.findData('mistral')
        if idx < 0:
            idx = 0
        self.icon_provider_combo.setCurrentIndex(idx)

        provider = self.icon_provider_combo.currentData()
        default_endpoint, default_model = self._default_icon_generation_values(provider)
        saved_model = self.icon_generation_settings.get('image_model') or self.icon_generation_settings.get('model') or ''
        if provider == 'openai' and saved_model and not ai_settings._looks_like_image_model(saved_model):
            saved_model = ''
        self.icon_endpoint_edit.setText(
            self.icon_generation_settings.get('endpoint') or default_endpoint
        )
        self.icon_model_edit.setText(
            saved_model or default_model
        )

    def _default_icon_generation_values(self, provider):
        """Return endpoint and model defaults for icon generation."""
        settings = ai_settings.get_api_settings(provider)
        endpoint = str(settings.get('base_url', '')).strip()
        model = str(settings.get('image_model', '')).strip()
        if provider == 'openai' and model and not ai_settings._looks_like_image_model(model):
            model = ''
        if provider == 'mistral':
            endpoint = endpoint or 'https://api.mistral.ai/v1'
            model = model or 'mistral-medium-latest'
        elif provider == 'openai':
            endpoint = endpoint or 'https://api.openai.com/v1'
            model = model or 'gpt-image-2'
        else:
            model = model or str(settings.get('image_model', '')).strip()
        return endpoint.rstrip('/'), model

    def _set_icon_controls_enabled(self, enabled):
        """Enable or disable icon editing controls."""
        self.browse_icon_button.setEnabled(enabled)
        self.apply_icon_button.setEnabled(enabled)
        self.generate_icon_button.setEnabled(enabled)
        self.edit_icon_button.setEnabled(enabled)
        self.clear_icon_button.setEnabled(enabled)

    def _refresh_icon_controls(self, plugin_info):
        """Refresh icon preview and editable path for a selected plugin."""
        icon_path = pathlib.Path(plugin_info['icon_path'])
        manifest = plugin_info.get('manifest')
        if manifest is not None and getattr(manifest, "icon", None):
            manifest_path = pathlib.Path(manifest.icon).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = pathlib.Path(plugin_info['path']) / manifest_path
            self.icon_path_edit.setText(str(manifest_path))
        elif icon_path.exists():
            self.icon_path_edit.setText(str(icon_path))
        else:
            self.icon_path_edit.clear()

        icon = create_plugin_icon_with_fallback(
            plugin_info['module'],
            plugin_info['path'],
            size=64,
            manifest=manifest,
        )
        pixmap = icon.pixmap(QSize(64, 64))
        self.icon_preview_label.setPixmap(pixmap)

    def _write_icon_from_image(self, source):
        """Copy or rasterize an image file into the selected plugin icon path."""
        plugin_info = self.plugins[self.current_plugin]
        destination = pathlib.Path(plugin_info['icon_path'])
        destination.parent.mkdir(parents=True, exist_ok=True)

        suffix = source.suffix.lower()
        if suffix == ".png":
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            return destination

        pixmap = QPixmap(str(source))
        if pixmap.isNull():
            raise ValueError(f"Unsupported or unreadable image: {source}")
        scaled = pixmap.scaled(
            256,
            256,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        canvas = QPixmap(256, 256)
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        x = (256 - scaled.width()) // 2
        y = (256 - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        painter.end()
        if not canvas.save(str(destination), "PNG"):
            raise OSError(f"Could not write icon image to {destination}")
        return destination

    def _save_generated_icon(self, plugin_info, output_path, size=128, prompt=None):
        """Generate and save a plugin icon inside the Plugin Manager."""
        ai_error = None
        if self._ai_icon_generation_available():
            try:
                self._save_ai_generated_icon(plugin_info, output_path, size=size, prompt=prompt)
                return "AI Settings"
            except Exception as e:
                ai_error = e

        pixmap = self._create_generated_icon(
            plugin_info['name'],
            plugin_info.get('doc', ''),
            size=size,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not pixmap.save(str(output_path), "PNG"):
            raise OSError(f"Could not write icon image to {output_path}")
        if ai_error is not None:
            return "local fallback"
        return "local generator"

    def _ai_icon_generation_available(self):
        """Return whether AI Settings can drive remote icon generation."""
        if not all(
            attr in self.__dict__
            for attr in ("icon_provider_combo", "icon_endpoint_edit", "icon_model_edit")
        ):
            return False
        provider = self._selected_icon_provider()
        base_url = self.icon_endpoint_edit.text().strip()
        image_model = self.icon_model_edit.text().strip()
        api_key = self._api_key_for_provider(provider)
        if not base_url or not image_model:
            return False
        if provider in {"openai", "mistral"} and not api_key:
            return False
        return provider != "local"

    def _save_ai_generated_icon(self, plugin_info, output_path, size=128, prompt=None):
        """Generate an icon through the OpenAI-compatible image endpoint."""
        image_bytes = self._request_ai_generated_icon_bytes(plugin_info, prompt=prompt)
        pixmap = QPixmap()
        if not pixmap.loadFromData(image_bytes):
            raise ValueError("AI provider returned unsupported image data")
        scaled = pixmap.scaled(
            size,
            size,
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )
        crop_x = max(0, (scaled.width() - size) // 2)
        crop_y = max(0, (scaled.height() - size) // 2)
        square = scaled.copy(crop_x, crop_y, size, size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not square.save(str(output_path), "PNG"):
            raise OSError(f"Could not write icon image to {output_path}")

    def _request_ai_generated_icon_bytes(self, plugin_info, prompt=None):
        """Request a generated icon image from the configured AI provider."""
        provider = self._selected_icon_provider()
        if provider == "mistral":
            return self._request_mistral_generated_icon_bytes(plugin_info, prompt=prompt)
        return self._request_openai_compatible_icon_bytes(plugin_info, prompt=prompt)

    def _request_openai_compatible_icon_bytes(self, plugin_info, prompt=None):
        """Request an icon through an OpenAI-compatible image endpoint."""
        import requests

        provider = self._selected_icon_provider()
        base_url = self.icon_endpoint_edit.text().strip().rstrip("/")
        image_model = self.icon_model_edit.text().strip()
        api_key = self._api_key_for_provider(provider)
        if not base_url:
            raise ValueError("Icon generation endpoint is empty")
        if not image_model:
            raise ValueError("Icon generation model is empty")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if not prompt:
            prompt = self._ai_icon_prompt(plugin_info)
        payload = {
            "model": image_model,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "output_format": "png",
        }
        url = f"{base_url}/images/generations"
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        if response.status_code >= 400:
            minimal_payload = dict(payload)
            minimal_payload.pop("output_format", None)
            response = requests.post(url, headers=headers, json=minimal_payload, timeout=90)
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code}: {response.text[:300]}")

        data = response.json()
        images = data.get("data") or []
        if not images:
            raise ValueError("AI provider returned no image data")
        first = images[0]
        if first.get("b64_json"):
            return base64.b64decode(first["b64_json"])
        if first.get("url"):
            image_response = requests.get(first["url"], timeout=60)
            image_response.raise_for_status()
            return image_response.content
        raise ValueError("AI provider returned neither b64_json nor url")

    def _request_mistral_generated_icon_bytes(self, plugin_info, prompt=None):
        """Request an icon through Mistral Agents image generation."""
        import requests

        model = self.icon_model_edit.text().strip()
        api_key = self._api_key_for_provider("mistral")
        if not self.icon_endpoint_edit.text().strip():
            raise ValueError("Icon generation endpoint is empty")
        if not model:
            raise ValueError("Icon generation model is empty")
        if not api_key:
            raise ValueError("Mistral API key is missing in AI Settings")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        agent_payload = {
            "model": model,
            "name": "ChiSurf Plugin Icon Generator",
            "description": "Agent used by ChiSurf Plugin Manager to generate plugin icons.",
            "instructions": (
                "Use the image generation tool to create a single square icon. "
                "Return the generated image file."
            ),
            "tools": [{"type": "image_generation"}],
            "completion_args": {"temperature": 0.3, "top_p": 0.95},
        }
        agent_response = self._post_mistral_json_with_retries(
            requests,
            "agents",
            headers=headers,
            payload=agent_payload,
            timeout=60,
        )
        agent_id = agent_response.json().get("id")
        if not agent_id:
            raise ValueError("Mistral did not return an agent id")

        if not prompt:
            prompt = self._ai_icon_prompt(plugin_info)
        conversation_response = self._post_mistral_json_with_retries(
            requests,
            "conversations",
            headers=headers,
            payload={
                "agent_id": agent_id,
                "inputs": prompt,
                "stream": False,
            },
            timeout=120,
        )
        file_id = self._extract_mistral_file_id(conversation_response.json())
        if not file_id:
            raise ValueError("Mistral response did not contain a generated image file id")

        file_response = requests.get(
            self._mistral_endpoint_url(f"files/{file_id}/content"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        if file_response.status_code == 404:
            file_response = requests.get(
                self._mistral_endpoint_url(f"files/{file_id}/download"),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60,
            )
        file_response.raise_for_status()
        return file_response.content

    def _post_mistral_json_with_retries(self, requests_module, path, headers, payload, timeout):
        """POST JSON to Mistral with bounded retry handling for HTTP 429."""
        import time

        max_attempts = 3
        last_response = None
        for attempt in range(max_attempts):
            response = requests_module.post(
                self._mistral_endpoint_url(path),
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code != 429:
                response.raise_for_status()
                return response

            last_response = response
            if attempt == max_attempts - 1:
                break
            retry_after = self._retry_after_seconds(response)
            delay = retry_after if retry_after is not None else 2 ** attempt
            logger.warning(
                "Mistral icon generation rate-limited on /%s; retrying in %.1f seconds",
                path,
                delay,
            )
            time.sleep(delay)

        detail = ""
        if last_response is not None and getattr(last_response, "text", ""):
            detail = f": {last_response.text[:300]}"
        raise AIIconRateLimitError(
            "Mistral returned 429 Too Many Requests after retrying. "
            "Wait before regenerating, or increase the Workspace rate limits in Mistral Studio"
            f"{detail}"
        )

    def _retry_after_seconds(self, response):
        """Return a Retry-After delay from a response when available."""
        headers = getattr(response, "headers", {}) or {}
        value = headers.get("Retry-After") or headers.get("retry-after")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    def _mistral_endpoint_url(self, path):
        """Return a normalized Mistral API URL for a relative endpoint path."""
        base_url = self.icon_endpoint_edit.text().strip().rstrip("/")
        if not base_url:
            raise ValueError("Icon generation endpoint is empty")
        if base_url.endswith("/beta"):
            base_url = base_url[:-5]
        return f"{base_url}/{path.lstrip('/')}"

    def _extract_mistral_file_id(self, data):
        """Extract the first generated image file id from a Mistral response."""
        if isinstance(data, dict):
            if data.get("type") == "tool_file" and data.get("file_id"):
                return data["file_id"]
            if data.get("file_id") and data.get("file_type") in (None, "png", "image/png"):
                return data["file_id"]
            for value in data.values():
                file_id = self._extract_mistral_file_id(value)
                if file_id:
                    return file_id
        elif isinstance(data, list):
            for value in data:
                file_id = self._extract_mistral_file_id(value)
                if file_id:
                    return file_id
        return None

    def _selected_icon_provider(self):
        """Return the selected icon generation provider key."""
        return str(self.icon_provider_combo.currentData() or "").strip()

    def _api_key_for_provider(self, provider):
        """Return API key for a provider using AI Settings and env fallback."""
        settings = ai_settings.get_api_settings(provider)
        api_key = str(settings.get("api_key", "")).strip()
        if api_key:
            return api_key
        for _display, (key, _url, _api_url, env_var) in ai_settings.PROVIDERS.items():
            if key == provider and env_var:
                return os.environ.get(env_var, "").strip()
        return ""

    def _ai_icon_prompt(self, plugin_info):
        """Build the remote image-generation prompt for a plugin icon."""
        plugin_name = plugin_info["name"]
        description = (plugin_info.get("doc") or "Scientific data analysis plugin.").strip()
        one_sentence = " ".join(description.splitlines()).strip()
        if "." in one_sentence:
            one_sentence = one_sentence.split(".", 1)[0].strip() + "."
        visual_elements = self._icon_visual_elements(plugin_name, description)
        primary_symbol, secondary_symbol = self._icon_symbol_pair(plugin_name, description)
        return f"""Scientific Software Icon Template

Create a single professional application icon for scientific software.

Output requirements

Square icon, 128x128 pixels
Toolbar/app icon style
Centered composition
One clear visual concept only
Clean silhouette recognizable at 16x16 and 32x32 pixels
High contrast
Minimal details
Modern scientific software aesthetic
Subtle depth and lighting
Transparent or simple neutral background
No text
No letters
No numbers
No UI screenshots
No watermark
No borders
No decorative clutter

Scientific context
Plugin: {plugin_name}

Purpose:
{one_sentence}

Visual metaphor
Represent:
{visual_elements}

Combine:
{primary_symbol} + {secondary_symbol}

Style:
scientific visualization, vector-like clarity, professional research software, publication-quality graphics, simple geometric forms, visually balanced, elegant and memorable.

Composition

Subject centered
Occupies ~70% of icon area
Strong foreground/background separation
Distinct shape visible at very small sizes
Limited color palette
Avoid thin lines
Avoid small labels
Avoid complex scenes

Negative prompts
text, letters, words, numbers, watermark, screenshot, interface, toolbar, menu, browser window, photorealistic scene, multiple unrelated objects, crowded composition, excessive detail, blurry image, low contrast"""

    def _icon_visual_elements(self, plugin_name, description):
        """Infer concise visual elements for an icon prompt from plugin metadata."""
        terms = f"{plugin_name} {description}".lower()
        if any(term in terms for term in ("fcs", "correlation")):
            return "fluorescence correlation data, smooth decay curve, focused detection volume"
        if any(term in terms for term in ("decay", "lifetime", "tcspc")):
            return "fluorescence lifetime decay, photon timing, clean exponential curve"
        if any(term in terms for term in ("molecule", "protein", "structure", "trajectory")):
            return "molecular structure, connected atoms, scientific 3D geometry"
        if any(term in terms for term in ("image", "microscopy", "camera")):
            return "scientific image analysis, microscope field, focused signal"
        if any(term in terms for term in ("database", "sample", "repository")):
            return "organized scientific records, structured data, sample archive"
        if any(term in terms for term in ("plot", "graph", "histogram")):
            return "scientific plotting, measured data trend, clean analytical chart"
        if any(term in terms for term in ("settings", "manager", "setup", "plugin")):
            return "software configuration, modular plugin component, scientific tool"
        return "scientific measurement, analytical data, precise research instrument"

    def _icon_symbol_pair(self, plugin_name, description):
        """Infer primary and secondary icon symbols from plugin metadata."""
        terms = f"{plugin_name} {description}".lower()
        if any(term in terms for term in ("fcs", "correlation")):
            return "correlation curve", "confocal detection spot"
        if any(term in terms for term in ("decay", "lifetime", "tcspc")):
            return "decay curve", "single photon pulse"
        if any(term in terms for term in ("molecule", "protein", "structure", "trajectory")):
            return "molecular node network", "subtle 3D depth cue"
        if any(term in terms for term in ("image", "microscopy", "camera")):
            return "microscope image frame", "bright analytical feature"
        if any(term in terms for term in ("database", "sample", "repository")):
            return "stacked data cylinder", "sample marker"
        if any(term in terms for term in ("plot", "graph", "histogram")):
            return "clean graph curve", "data point cluster"
        if any(term in terms for term in ("settings", "manager", "setup", "plugin")):
            return "modular hexagon", "calibration dot"
        return "scientific instrument glyph", "measured signal curve"

    def _create_generated_icon(self, plugin_name, description, size=256, prompt=None):
        """Create a deterministic icon image from plugin metadata."""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        seed_text = prompt if prompt else f"{plugin_name}\n{description}"
        digest = hashlib.sha256(
            seed_text.encode("utf-8", errors="ignore")
        ).hexdigest()
        hue = int(digest[:2], 16) % 360
        accent_hue = (hue + 48 + int(digest[2:4], 16) % 72) % 360
        base = QColor.fromHsv(hue, 135, 210)
        accent = QColor.fromHsv(accent_hue, 170, 230)
        dark = QColor.fromHsv(hue, 125, 75)
        light = QColor.fromHsv(hue, 35, 250)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)

        margin = max(4, size // 16)
        radius = max(6, size // 8)
        rect = pixmap.rect().adjusted(margin, margin, -margin, -margin)
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, light)
        gradient.setColorAt(0.38, base)
        gradient.setColorAt(0.72, accent)
        gradient.setColorAt(1.0, dark)
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, radius, radius)

        painter.setBrush(QColor(255, 255, 255, 44))
        painter.drawEllipse(QtCore.QPointF(size * 0.72, size * 0.25), size * 0.22, size * 0.22)
        painter.setBrush(QColor(0, 0, 0, 28))
        painter.drawEllipse(QtCore.QPointF(size * 0.28, size * 0.78), size * 0.18, size * 0.18)

        self._draw_generated_icon_symbol(painter, plugin_name, description, size, prompt=prompt)
        self._draw_generated_icon_label(painter, plugin_name, size)
        painter.end()
        return pixmap

    def _draw_generated_icon_symbol(self, painter, plugin_name, description, size, prompt=None):
        """Draw a compact metadata-derived symbol for the generated icon."""
        terms = f"{plugin_name} {description} {prompt or ''}".lower()
        painter.setPen(
            QPen(
                QColor(255, 255, 255, 215),
                max(2, size // 26),
                Qt.SolidLine,
                Qt.RoundCap,
                Qt.RoundJoin,
            )
        )
        symbol_rect = QtCore.QRectF(size * 0.24, size * 0.18, size * 0.52, size * 0.28)
        if any(term in terms for term in ("fcs", "correlation", "decay", "lifetime", "trace", "histogram")):
            points = []
            for i in range(9):
                x = symbol_rect.left() + symbol_rect.width() * i / 8
                y = symbol_rect.bottom() - symbol_rect.height() * (0.15 + 0.7 * (0.5 ** (i / 2)))
                points.append(QtCore.QPointF(x, y))
            for p1, p2 in zip(points, points[1:]):
                painter.drawLine(p1, p2)
        elif any(term in terms for term in ("image", "microscopy", "browser", "screenshot")):
            painter.drawRoundedRect(symbol_rect, size * 0.04, size * 0.04)
            painter.drawEllipse(QtCore.QPointF(symbol_rect.center()), size * 0.12, size * 0.12)
            painter.drawLine(
                QtCore.QPointF(symbol_rect.left(), symbol_rect.bottom()),
                QtCore.QPointF(symbol_rect.right(), symbol_rect.top()),
            )
        elif any(term in terms for term in ("model", "protein", "molecule", "fret", "trajectory", "structure")):
            centers = [
                QtCore.QPointF(size * 0.31, size * 0.38),
                QtCore.QPointF(size * 0.55, size * 0.31),
                QtCore.QPointF(size * 0.67, size * 0.54),
                QtCore.QPointF(size * 0.42, size * 0.61),
            ]
            for p1, p2 in zip(centers, centers[1:] + centers[:1]):
                painter.drawLine(p1, p2)
            painter.setBrush(QColor(255, 255, 255, 220))
            for center in centers:
                painter.drawEllipse(center, size * 0.055, size * 0.055)
            painter.setBrush(Qt.NoBrush)
        elif any(term in terms for term in ("settings", "manager", "setup", "tool", "plugin")):
            painter.drawEllipse(symbol_rect.center(), size * 0.19, size * 0.19)
            for i in range(8):
                line = QtCore.QLineF(symbol_rect.center(), QtCore.QPointF(symbol_rect.center().x(), symbol_rect.top()))
                line.setAngle(i * 45)
                painter.drawLine(line)
        else:
            painter.drawLine(
                QtCore.QPointF(symbol_rect.left(), symbol_rect.center().y()),
                QtCore.QPointF(symbol_rect.right(), symbol_rect.center().y()),
            )
            painter.drawLine(
                QtCore.QPointF(symbol_rect.center().x(), symbol_rect.top()),
                QtCore.QPointF(symbol_rect.center().x(), symbol_rect.bottom()),
            )
            painter.drawEllipse(symbol_rect.center(), size * 0.14, size * 0.14)

    def _draw_generated_icon_label(self, painter, plugin_name, size):
        """Draw the generated icon label."""
        label = self._generated_icon_label(plugin_name)

        label_rect = QtCore.QRectF(size * 0.12, size * 0.54, size * 0.76, size * 0.30)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(20, 24, 34, 150))
        painter.drawRoundedRect(label_rect, size * 0.07, size * 0.07)

        painter.setPen(QColor(255, 255, 255, 245))
        font = QFont("Arial")
        font.setBold(True)
        max_font_size = int(size * 0.19) if len(label) >= 3 else int(size * 0.23)
        min_font_size = max(10, int(size * 0.13))
        for font_size in range(max_font_size, min_font_size - 1, -1):
            font.setPointSize(font_size)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            if metrics.horizontalAdvance(label) <= label_rect.width() * 0.84:
                break
        painter.drawText(label_rect, Qt.AlignCenter, label)

    def _generated_icon_label(self, plugin_name):
        """Return a compact label for a generated plugin icon."""
        name_part = plugin_name.split(":")[-1].strip()
        words = [word for word in name_part.replace("-", " ").replace("_", " ").split() if word]
        if len(words) >= 2:
            return "".join(word[0] for word in words[:3]).upper()
        compact = "".join(ch for ch in name_part if ch.isalnum())
        if not compact:
            return "?"
        if len(compact) <= 4:
            return compact.upper()
        consonants = "".join(ch for ch in compact if ch.lower() not in "aeiou")
        return (consonants[:4] or compact[:4]).upper()

    def _set_manifest_icon(self, icon_value):
        """Update manifest icon metadata when a manifest is present."""
        if self.current_plugin is None:
            return
        plugin_info = self.plugins[self.current_plugin]
        manifest_path = pathlib.Path(plugin_info['path']) / "manifest.json"
        if not manifest_path.exists():
            return
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if icon_value is None:
            data.pop("icon", None)
        else:
            data["icon"] = icon_value
        manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        plugin_info['manifest'] = load_manifest(manifest_path)

    def _reload_icon_for_current_plugin(self, icon_path):
        """Refresh the selected plugin icon in the preview and list."""
        plugin_info = self.plugins[self.current_plugin]
        self._refresh_icon_controls(plugin_info)
        icon = create_plugin_icon_with_fallback(
            plugin_info['module'],
            plugin_info['path'],
            size=32,
            manifest=plugin_info.get('manifest'),
        )
        for i in range(self.plugin_list.count()):
            item = self.plugin_list.item(i)
            if item.data(Qt.UserRole) == self.current_plugin:
                item.setIcon(icon)
                break
        self.icon_path_edit.setText(str(icon_path) if pathlib.Path(icon_path).exists() else "")

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the PluginManagerWidget class
    window = PluginManagerWidget()
    # Show the window
    window.show()
