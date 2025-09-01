from __future__ import annotations

import pathlib
import yaml
import re

from qtpy import QtCore, QtGui, QtWidgets

import chisurf
import chisurf.fio as io
from chisurf import logging
import chisurf.settings
from chisurf.settings import cs_settings
from chisurf.gui.widgets.settings_editor import SettingsEditor, SettingsTreeModel, SettingsItemDelegate


class UCFRETSamplingItemDelegate(SettingsItemDelegate):
    """A delegate for editing UCFRET sampling settings with tooltips based on documentation."""

    def __init__(self, documentation_dict=None):
        super().__init__()
        self.documentation_dict = documentation_dict or {}

    def createEditor(self, parent, option, index):
        """Create an appropriate editor widget based on the data type."""
        # Get the documentation for this setting
        path = self._get_setting_path(index)
        tooltip = self.documentation_dict.get(path, "")

        # Create the editor using the parent class method
        editor = super().createEditor(parent, option, index)

        # Set the tooltip if available
        if tooltip and editor:
            editor.setToolTip(tooltip)

        return editor

    def _get_setting_path(self, index):
        """Get the full path of a setting in the tree."""
        if not index.isValid():
            return ""

        # Get the setting name
        setting_name = index.sibling(index.row(), 0).data(QtCore.Qt.DisplayRole)

        # Get the parent path recursively
        parent_index = index.parent()
        if parent_index.isValid():
            parent_path = self._get_setting_path(parent_index)
            return f"{parent_path}.{setting_name}" if parent_path else setting_name
        else:
            return setting_name


class UCFRETSamplingTreeModel(SettingsTreeModel):
    """A model for displaying and editing UCFRET sampling settings with documentation."""

    def __init__(self, parent=None, documentation_dict=None):
        super().__init__(parent)
        self.documentation_dict = documentation_dict or {}

    def _populate_model(self, settings_dict, parent=None, path=""):
        """Recursively populate the model with settings from the dictionary."""
        for key, value in settings_dict.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, dict):
                # Create a category item
                category_item = QtGui.QStandardItem(key)
                category_item.setEditable(False)

                # Set tooltip if documentation exists
                tooltip = self.documentation_dict.get(current_path, "")
                if tooltip:
                    category_item.setToolTip(tooltip)

                value_item = QtGui.QStandardItem("")
                value_item.setEditable(False)

                if parent is None:
                    self.appendRow([category_item, value_item])
                else:
                    parent.appendRow([category_item, value_item])

                # Recursively add child items
                self._populate_model(value, category_item, current_path)
            else:
                # Create a setting item
                setting_item = QtGui.QStandardItem(key)
                setting_item.setEditable(False)

                # Set tooltip if documentation exists
                tooltip = self.documentation_dict.get(current_path, "")
                if tooltip:
                    setting_item.setToolTip(tooltip)

                # Create a value item with appropriate editor
                value_item = QtGui.QStandardItem(str(value))

                # Also set tooltip on value item
                if tooltip:
                    value_item.setToolTip(tooltip)

                # Store the original data type with the item
                value_type = type(value)
                value_item.setData(value_type, QtCore.Qt.UserRole)

                # For lists, store the string representation in a way that can be parsed back
                if isinstance(value, list):
                    value_item.setText(str(value))

                if parent is None:
                    self.appendRow([setting_item, value_item])
                else:
                    parent.appendRow([setting_item, value_item])


class UCFRETSamplingSettingsEditor(SettingsEditor):
    """A specialized editor for UCFRET sampling settings with documentation tooltips."""

    def __init__(
        self,
        *args,
        filename: str = None,
        **kwargs
    ):
        # Parse documentation before initializing parent
        self.documentation_dict = self._parse_documentation(filename)

        # Initialize parent
        super().__init__(*args, filename=filename, **kwargs)

        # Set window title
        self.setWindowTitle("UCFRET Sampling Settings Editor")

    def setup_ui(self):
        """Set up the user interface."""
        # Call parent setup_ui
        super().setup_ui()

        # Replace the model and delegate with our custom versions
        self.model = UCFRETSamplingTreeModel(self, self.documentation_dict)
        self.tree_view.setModel(self.model)

        self.delegate = UCFRETSamplingItemDelegate(self.documentation_dict)
        self.tree_view.setItemDelegate(self.delegate)

        # Add a button to create a new settings file
        self.new_button = QtWidgets.QPushButton("New")
        self.new_button.clicked.connect(self.create_new_settings)

        # Insert the new button before the reload button
        button_layout = self.layout().itemAt(2).layout()
        button_layout.insertWidget(1, self.new_button)

    def load_file(self, filename: str = None):
        """Load settings from a file and update documentation."""
        if filename and filename != self.filename:
            # Update documentation when loading a new file
            self.documentation_dict = self._parse_documentation(filename)

            # Update delegate and model with new documentation
            self.delegate.documentation_dict = self.documentation_dict
            self.model.documentation_dict = self.documentation_dict

        # Call parent load_file
        super().load_file(filename)

    def create_new_settings(self):
        """Create a new settings file."""
        # Ask for a filename
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Create New Settings File",
            "",
            "YAML Files (*.yml *.yaml)"
        )

        if not filename:
            return

        # Create a new settings file with default values
        try:
            # Get default settings from the ucfret module
            default_settings_path = pathlib.Path(chisurf.__file__).parent / "plugins" / "ucfret" / "default_sampling_settings.yml"

            if default_settings_path.exists():
                # Copy default settings
                with open(default_settings_path, 'r') as f:
                    settings = yaml.safe_load(f)
            else:
                # Create minimal default settings
                settings = {
                    "init": {
                        "tauD0": 4.0,
                        "forster_radius": 52.0,
                        "n_axis_bins": 64,
                        "fret_efficiency_bounds": [0.01, 0.99999999],
                        "sigma_range": [3.0, 6.0],
                        "scatter_range": [0.0, 1.0],
                        "background_range": [0.0, None],
                        "time_shift_range": [-10.0, 10.0],
                        "lifetime_range": [0., 6.0],
                        "sample_d0": False,
                        "verbose": 1
                    },
                    "sample": {
                        "nsteps": 50,
                        "thin": 1,
                        "steps_per_write": 50,
                        "n_proc": None,
                        "walker_per_dim": 2,
                        "verbose": 1,
                        "new_run": True
                    },
                    "analyze": {
                        "burn_in": 30,
                        "n_best": 3,
                        "thin": 2,
                        "verbose": 1,
                        "n_amplitude_bins": 512,
                        "n_axis_bins": 512,
                        "plot_2d_vmax": 50,
                        "plot_2d_vmin": 10
                    }
                }

            # Write the settings to the new file
            with open(filename, 'w') as f:
                yaml.dump(settings, f, default_flow_style=False)

            # Load the new file
            self.load_file(filename)

        except Exception as e:
            logging.log(1, f"Error creating new settings file: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                f"Failed to create new settings file: {e}"
            )

    def _parse_documentation(self, filename: str = None) -> dict:
        """Parse documentation comments from a YAML file."""
        documentation_dict = {}

        if not filename:
            return documentation_dict

        try:
            # Read the file content
            with open(filename, 'r') as f:
                content = f.read()

            # Parse the YAML content
            settings = yaml.safe_load(content)

            # Find documentation comments in the content
            lines = content.split('\n')
            current_path = []

            for i, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    continue

                # Count leading spaces to determine indentation level
                indent = len(line) - len(line.lstrip())

                # Check if this is a setting line (contains a colon)
                if ':' in line and '#' not in line.split(':', 1)[0]:
                    # Extract the key
                    key = line.split(':', 1)[0].strip()

                    # Update the current path based on indentation
                    while len(current_path) > 0 and indent <= current_path[-1][1]:
                        current_path.pop()

                    current_path.append((key, indent))

                    # Check if there's a comment on this line
                    if '#' in line:
                        comment = line.split('#', 1)[1].strip()
                        if comment:
                            # Build the full path
                            full_path = '.'.join([p[0] for p in current_path])
                            documentation_dict[full_path] = comment

            # If we couldn't find documentation in the file, try to find it in the example files
            if not documentation_dict:
                example_files = [
                    pathlib.Path(chisurf.__file__).parent / "plugins" / "ucfret" / ".." / ".." / ".." / "modules" / "ucfret" / "example" / "ucfret_settings.yml",
                    pathlib.Path(chisurf.__file__).parent / "plugins" / "ucfret" / ".." / ".." / ".." / "modules" / "ucfret" / "ucfret" / "settings" / "ucfret_settings.yml"
                ]

                for example_file in example_files:
                    try:
                        example_file = example_file.resolve()
                        if example_file.exists():
                            with open(example_file, 'r') as f:
                                example_content = f.read()

                            # Parse the example YAML content
                            example_settings = yaml.safe_load(example_content)

                            # Find documentation comments in the example content
                            example_lines = example_content.split('\n')
                            current_path = []

                            for i, line in enumerate(example_lines):
                                # Skip empty lines
                                if not line.strip():
                                    continue

                                # Count leading spaces to determine indentation level
                                indent = len(line) - len(line.lstrip())

                                # Check if this is a setting line (contains a colon)
                                if ':' in line and '#' not in line.split(':', 1)[0]:
                                    # Extract the key
                                    key = line.split(':', 1)[0].strip()

                                    # Update the current path based on indentation
                                    while len(current_path) > 0 and indent <= current_path[-1][1]:
                                        current_path.pop()

                                    current_path.append((key, indent))

                                    # Check if there's a comment on this line
                                    if '#' in line:
                                        comment = line.split('#', 1)[1].strip()
                                        if comment:
                                            # Build the full path
                                            full_path = '.'.join([p[0] for p in current_path])
                                            documentation_dict[full_path] = comment

                            # If we found documentation, break
                            if documentation_dict:
                                break
                    except Exception as e:
                        logging.log(1, f"Error parsing example file {example_file}: {e}")

        except Exception as e:
            logging.log(1, f"Error parsing documentation from {filename}: {e}")

        return documentation_dict
