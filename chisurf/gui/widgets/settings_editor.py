from __future__ import annotations

import pathlib
import yaml
import re
import os
import numpy as np

from qtpy import QtCore, QtGui, QtWidgets

import chisurf
import chisurf.fio as io
from chisurf import logging
import chisurf.settings
from chisurf.settings import cs_settings


# Custom YAML representer for floats to preserve scientific notation
def float_representer(dumper, value):
    """
    Custom representer for float values to preserve scientific notation.

    Parameters
    ----------
    dumper : yaml.Dumper
        The YAML dumper instance.
    value : float
        The float value to represent.

    Returns
    -------
    yaml.ScalarNode
        The YAML scalar node with the appropriate representation.
    """
    # Use scientific notation for very small or very large numbers
    if abs(value) < 0.0001 or abs(value) > 1000000:
        # Format with scientific notation, preserving precision
        text = f"{value:.10e}"
        # Remove trailing zeros in the exponent part
        text = re.sub(r'e(\+|-)0*(\d+)', r'e\1\2', text)
        # Remove trailing zeros in the mantissa part
        text = re.sub(r'\.(\d*?)0+e', r'.\1e', text)
        # If mantissa ends with a decimal point, remove it
        text = re.sub(r'\.e', r'e', text)
        return dumper.represent_scalar('tag:yaml.org,2002:float', text)
    else:
        # Use default representation for regular floats
        return dumper.represent_scalar('tag:yaml.org,2002:float', str(value))

# Register the custom representer
yaml.add_representer(float, float_representer)


class SettingsItemDelegate(QtWidgets.QStyledItemDelegate):
    """A delegate for editing settings with appropriate widgets based on data type."""

    def __init__(self, documentation_dict=None):
        """
        Initialize the delegate.

        Parameters
        ----------
        documentation_dict : dict, optional
            A dictionary containing documentation for settings.
        """
        super().__init__()
        self.documentation_dict = documentation_dict or {}

    def is_hex_color(self, value):
        """Check if a value is a hex color code."""
        if not isinstance(value, str):
            return False
        # Match standard hex color format: #RRGGBB
        return bool(re.match(r'^#[0-9A-Fa-f]{6}$', value))

    def paint(self, painter, option, index):
        """Custom painting for color values."""
        if index.column() == 1:  # Value column
            value = index.data(QtCore.Qt.DisplayRole)
            if isinstance(value, str) and self.is_hex_color(value):
                # Draw the color swatch
                rect = option.rect
                color_rect = QtCore.QRect(rect.left() + 5, rect.top() + 5, 20, rect.height() - 10)

                # Draw selection background if selected
                if option.state & QtWidgets.QStyle.State_Selected:
                    painter.fillRect(rect, option.palette.highlight())
                    text_color = option.palette.highlightedText().color()
                else:
                    painter.fillRect(rect, option.palette.base())
                    text_color = option.palette.text().color()

                # Draw color swatch
                painter.setPen(QtGui.QPen(QtCore.Qt.black, 1))
                painter.setBrush(QtGui.QBrush(QtGui.QColor(value)))
                painter.drawRect(color_rect)

                # Draw text
                text_rect = QtCore.QRect(rect.left() + 30, rect.top(), rect.width() - 35, rect.height())
                painter.setPen(text_color)
                painter.drawText(text_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, value)
                return

        # For other items, use default painting
        super().paint(painter, option, index)

    def createEditor(self, parent, option, index):
        """Create an appropriate editor widget based on the data type."""
        if not index.isValid() or index.column() != 1:
            return super().createEditor(parent, option, index)

        # Get the setting path and check for documentation
        setting_path = self._get_setting_path(index)
        tooltip = self.documentation_dict.get(setting_path, "")

        value = index.data(QtCore.Qt.EditRole)
        data_type = type(value)

        # Create appropriate editor based on data type
        if data_type == bool:
            editor = QtWidgets.QComboBox(parent)
            editor.addItems(["True", "False"])
            editor.setCurrentIndex(0 if value else 1)
            if tooltip:
                editor.setToolTip(tooltip)
            return editor
        elif self.is_hex_color(value):
            # For hex color values, use a color dialog
            button = QtWidgets.QPushButton(parent)
            button.setText(value)
            button.setStyleSheet(f"background-color: {value}; color: {'black' if sum(QtGui.QColor(value).getRgb()[:3]) > 382 else 'white'};")
            button.clicked.connect(lambda: self._choose_color(button))
            if tooltip:
                button.setToolTip(tooltip)
            return button
        elif isinstance(value, (list, tuple)):
            # For lists, use a line edit with comma-separated values
            editor = QtWidgets.QLineEdit(parent)
            editor.setText(", ".join(str(item) for item in value))
            if tooltip:
                editor.setToolTip(tooltip)
            return editor
        elif data_type in (int, float):
            # For numbers, use a spin box or line edit depending on the value
            if data_type == int:
                editor = QtWidgets.QSpinBox(parent)
                editor.setRange(-1000000, 1000000)
                editor.setValue(value)
            else:
                # Check if the float is in scientific notation or has many decimal places
                str_value = str(value)
                if 'e' in str_value.lower() or abs(value) < 0.0001 or abs(value) > 1000000:
                    # For scientific notation or extreme values, use a line edit
                    editor = QtWidgets.QLineEdit(parent)
                    editor.setText(str_value)
                else:
                    # For regular floats, use a double spin box
                    editor = QtWidgets.QDoubleSpinBox(parent)
                    editor.setRange(-1000000.0, 1000000.0)
                    editor.setDecimals(6)
                    editor.setValue(value)
            if tooltip:
                editor.setToolTip(tooltip)
            return editor
        elif isinstance(value, str) and os.path.sep in value:
            # For file paths, use a line edit with a browse button
            widget = QtWidgets.QWidget(parent)
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            line_edit = QtWidgets.QLineEdit(widget)
            line_edit.setText(value)
            if tooltip:
                line_edit.setToolTip(tooltip)

            browse_button = QtWidgets.QPushButton("...", widget)
            browse_button.setMaximumWidth(30)
            browse_button.clicked.connect(lambda: self._browse_file(line_edit))

            layout.addWidget(line_edit)
            layout.addWidget(browse_button)

            # Store the line edit as a property of the widget for later access
            widget.setProperty("lineEdit", line_edit)
            return widget
        else:
            # For other types, use a line edit
            editor = QtWidgets.QLineEdit(parent)
            editor.setText(str(value))
            if tooltip:
                editor.setToolTip(tooltip)
            return editor

    def _get_setting_path(self, index):
        """Get the full path of a setting in the tree."""
        path_parts = []
        current = index

        # Traverse up the tree to build the path
        while current.isValid():
            if current.column() == 0:  # Only add key names
                path_parts.insert(0, current.data())
            current = current.parent()

        return ".".join(path_parts)

    def _choose_color(self, button):
        """Open a color dialog and set the selected color."""
        current_color = QtGui.QColor(button.text())
        color = QtWidgets.QColorDialog.getColor(current_color, button.parent())

        if color.isValid():
            hex_color = f"#{color.red():02x}{color.green():02x}{color.blue():02x}"
            button.setText(hex_color)
            button.setStyleSheet(f"background-color: {hex_color}; color: {'black' if sum(color.getRgb()[:3]) > 382 else 'white'};")

    def _browse_file(self, line_edit):
        """Open a file dialog and set the selected file path."""
        current_path = line_edit.text()
        start_dir = os.path.dirname(current_path) if current_path else ""

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            line_edit.parent(), "Select File", start_dir
        )

        if file_path:
            line_edit.setText(file_path)

    def setModelData(self, editor, model, index):
        """Set the model data from the editor."""
        if not index.isValid() or index.column() != 1:
            super().setModelData(editor, model, index)
            return

        value = index.data(QtCore.Qt.EditRole)
        data_type = type(value)

        if data_type == bool:
            # For boolean values, get from combo box
            combo_box = editor
            new_value = combo_box.currentText() == "True"
            model.setData(index, new_value, QtCore.Qt.EditRole)
        elif self.is_hex_color(value):
            # For hex color values, get from button text
            button = editor
            model.setData(index, button.text(), QtCore.Qt.EditRole)
        elif isinstance(value, (list, tuple)):
            # For lists, parse comma-separated values
            line_edit = editor
            text = line_edit.text()
            items = [item.strip() for item in text.split(",")]

            # Try to convert items to the same type as the original list items
            if value and all(isinstance(item, (int, float, str)) for item in value):
                item_type = type(value[0])
                try:
                    items = [item_type(item) for item in items]
                except ValueError:
                    pass

            model.setData(index, type(value)(items), QtCore.Qt.EditRole)
        elif data_type == int:
            # For integers, get from spin box
            spin_box = editor
            model.setData(index, spin_box.value(), QtCore.Qt.EditRole)
        elif data_type == float:
            # For floats, get from double spin box or line edit
            if isinstance(editor, QtWidgets.QDoubleSpinBox):
                model.setData(index, editor.value(), QtCore.Qt.EditRole)
            else:
                # For line edit (scientific notation)
                try:
                    text = editor.text()
                    # Convert to float, preserving scientific notation
                    value = float(text)
                    model.setData(index, value, QtCore.Qt.EditRole)
                except ValueError:
                    # If conversion fails, keep the original value
                    logging.log(1, f"Warning: Could not convert '{text}' to float. Using original value.")
                    model.setData(index, value, QtCore.Qt.EditRole)
        elif isinstance(value, str) and os.path.sep in value:
            # For file paths, get from line edit in the widget
            widget = editor
            line_edit = widget.property("lineEdit")
            model.setData(index, line_edit.text(), QtCore.Qt.EditRole)
        else:
            # For other types, get from line edit
            line_edit = editor
            text = line_edit.text()

            # Try to convert to the original data type
            try:
                if data_type != str:
                    converted_value = data_type(text)
                    model.setData(index, converted_value, QtCore.Qt.EditRole)
                else:
                    model.setData(index, text, QtCore.Qt.EditRole)
            except ValueError:
                # If conversion fails, use the string value
                model.setData(index, text, QtCore.Qt.EditRole)


class SettingsTreeModel(QtGui.QStandardItemModel):
    """A tree model for displaying and editing settings."""

    def __init__(self, parent=None, documentation_dict=None):
        """
        Initialize the model.

        Parameters
        ----------
        parent : QObject, optional
            The parent object.
        documentation_dict : dict, optional
            A dictionary containing documentation for settings.
        """
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Setting", "Value"])
        self.documentation_dict = documentation_dict or {}

    def load_settings(self, settings_dict):
        """
        Load settings into the model.

        Parameters
        ----------
        settings_dict : dict
            The settings dictionary to load.
        """
        self.clear()
        self.setHorizontalHeaderLabels(["Setting", "Value"])
        self._populate_model(settings_dict)

    def _populate_model(self, settings_dict, parent=None, path=""):
        """
        Recursively populate the model with settings.

        Parameters
        ----------
        settings_dict : dict
            The settings dictionary to populate from.
        parent : QStandardItem, optional
            The parent item to add children to.
        path : str, optional
            The current path in the settings hierarchy.
        """
        if parent is None:
            parent = self.invisibleRootItem()

        for key, value in sorted(settings_dict.items()):
            # Create key item
            key_item = QtGui.QStandardItem(key)
            key_item.setEditable(False)

            # Create value item
            value_item = QtGui.QStandardItem()
            value_item.setData(value, QtCore.Qt.EditRole)

            # Set display text based on data type
            if isinstance(value, dict):
                # For dictionaries, don't set display text (will be populated with children)
                pass
            elif isinstance(value, (list, tuple)):
                # For lists, show comma-separated values
                value_item.setText(", ".join(str(item) for item in value))
            elif isinstance(value, bool):
                # For booleans, show "True" or "False"
                value_item.setText(str(value))
            elif isinstance(value, float):
                # For floats, preserve scientific notation if present
                str_value = str(value)
                if 'e' in str_value.lower():
                    # Ensure scientific notation is preserved
                    value_item.setText(str_value)
                else:
                    value_item.setText(str_value)
            else:
                # For other types, show string representation
                value_item.setText(str(value))

            # Add tooltip if documentation exists
            current_path = f"{path}.{key}" if path else key
            if current_path in self.documentation_dict:
                tooltip = self.documentation_dict[current_path]
                key_item.setToolTip(tooltip)
                value_item.setToolTip(tooltip)

            # Add items to model
            row = [key_item, value_item]

            if isinstance(value, dict):
                # For dictionaries, add as parent and recurse
                parent.appendRow(row)
                parent_index = self.indexFromItem(key_item)
                parent_item = self.itemFromIndex(parent_index)
                self._populate_model(value, parent_item, current_path)
            else:
                # For other types, add as leaf
                parent.appendRow(row)

    def get_settings_dict(self):
        """
        Get the settings as a dictionary.

        Returns
        -------
        dict
            The settings dictionary.
        """
        root = self.invisibleRootItem()
        return self._get_dict_from_item(root)

    def _get_dict_from_item(self, item):
        """
        Recursively build a dictionary from a model item.

        Parameters
        ----------
        item : QStandardItem
            The item to build the dictionary from.

        Returns
        -------
        dict
            The dictionary built from the item.
        """
        result_dict = {}

        for row in range(item.rowCount()):
            key_item = item.child(row, 0)
            value_item = item.child(row, 1)

            key = key_item.text()

            if key_item.hasChildren():
                # If the key item has children, it's a dictionary
                value = self._get_dict_from_item(key_item)
            else:
                # Otherwise, get the value from the value item
                value = value_item.data(QtCore.Qt.EditRole)

                # Convert string values to appropriate types if possible
                if isinstance(value, str):
                    value_str = value

                    # Try to convert to appropriate type
                    try:
                        # Check for boolean values
                        if value_str.lower() == "true":
                            value = True
                        elif value_str.lower() == "false":
                            value = False
                        # Check for integer values
                        elif value_str.isdigit():
                            value = int(value_str)
                        # Check for float values
                        elif re.match(r'^-?\d+(\.\d+)?$', value_str):
                            value = float(value_str)
                        # For other types (like strings), keep as is
                    except ValueError:
                        # If conversion fails, show a warning and keep as string
                        logging.log(1, f"Warning: Could not convert '{value_str}' for setting '{key}'. Using string value.")

            result_dict[key] = value

        return result_dict


class SettingsEditor(QtWidgets.QWidget):
    """A tree-based editor for editing settings."""

    def __init__(
        self,
        *args,
        filename: str = None,
        documentation_dict: dict = None,
        window_title: str = "Settings Editor",
        **kwargs
    ):
        """
        Initialize the settings editor.

        Parameters
        ----------
        filename : str, optional
            The path to the settings file to edit.
        documentation_dict : dict, optional
            A dictionary containing documentation for settings.
        window_title : str, optional
            The window title to display.
        """
        super().__init__(*args, **kwargs)

        self.filename = filename
        self.settings_dict = {}
        self.documentation_dict = documentation_dict or {}
        self.window_title = window_title

        self.setup_ui()

        if filename is not None:
            if pathlib.Path(filename).is_file():
                self.load_file(filename)

    def setup_ui(self):
        """Set up the user interface."""
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Create search bar
        search_layout = QtWidgets.QHBoxLayout()
        search_label = QtWidgets.QLabel("Search:")
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("Filter settings...")
        self.search_bar.setClearButtonEnabled(True)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_bar)
        layout.addLayout(search_layout)

        # Create tree view
        self.tree_view = QtWidgets.QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setSortingEnabled(False)
        self.tree_view.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | 
                                       QtWidgets.QAbstractItemView.EditKeyPressed)

        # Create model
        self.model = self.create_model()
        self.tree_view.setModel(self.model)

        # Set custom delegate for editing
        self.delegate = self.create_delegate()
        self.tree_view.setItemDelegate(self.delegate)

        # Add tree view to layout
        layout.addWidget(self.tree_view)

        # Create buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.path_label = QtWidgets.QLabel()
        self.save_button = QtWidgets.QPushButton("Save")
        self.reload_button = QtWidgets.QPushButton("Reload")
        self.help_button = QtWidgets.QPushButton("Help")

        button_layout.addWidget(self.path_label, 1)
        button_layout.addWidget(self.help_button)
        button_layout.addWidget(self.reload_button)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Connect signals
        self.save_button.clicked.connect(self.save_settings)
        self.reload_button.clicked.connect(lambda: self.load_file(self.filename))
        self.search_bar.textChanged.connect(self.filter_settings)
        self.help_button.clicked.connect(self.show_help)

        # Set window properties
        self.setWindowTitle(self.window_title)
        self.resize(800, 600)

    def create_model(self):
        """
        Create the model for the tree view.

        This method can be overridden by subclasses to provide a custom model.

        Returns
        -------
        QAbstractItemModel
            The model for the tree view.
        """
        return SettingsTreeModel(self, self.documentation_dict)

    def create_delegate(self):
        """
        Create the delegate for the tree view.

        This method can be overridden by subclasses to provide a custom delegate.

        Returns
        -------
        QAbstractItemDelegate
            The delegate for the tree view.
        """
        return SettingsItemDelegate(self.documentation_dict)

    def load_file(self, filename: str = None):
        """
        Load settings from a file.

        Parameters
        ----------
        filename : str, optional
            The path to the settings file to load.
        """
        if not filename:
            return

        try:
            logging.log(0, f"Loading settings file: {filename}")
            with open(filename, encoding="utf-8") as file:
                self.settings_dict = yaml.safe_load(file)

            self.path_label.setText(str(filename))
            self.filename = filename

            # Load settings into model
            self.model.load_settings(self.settings_dict)

            # Expand all items
            self.tree_view.expandAll()

            # Resize columns to content
            self.tree_view.resizeColumnToContents(0)

            # Clear search bar to show all items
            if hasattr(self, 'search_bar'):
                self.search_bar.clear()

        except Exception as e:
            logging.log(1, f"Error loading settings file {filename}: {e}")

    def filter_settings(self, text):
        """
        Filter the settings tree to show only items matching the search text.

        Parameters
        ----------
        text : str
            The text to filter by.
        """
        if not text:
            # If search text is empty, show all items
            self._show_all_items()
            return

        # Convert to lowercase for case-insensitive search
        search_text = text.lower()

        # Start with all items hidden
        self._hide_all_items()

        # Show items that match the search text
        self._filter_items(self.tree_view.model().invisibleRootItem(), search_text)

        # Expand all visible items
        self.tree_view.expandAll()

    def _show_all_items(self):
        """Show all items in the tree."""
        self._set_item_hidden(self.tree_view.model().invisibleRootItem(), False)
        self.tree_view.expandAll()

    def _hide_all_items(self):
        """Hide all items in the tree."""
        self._set_item_hidden(self.tree_view.model().invisibleRootItem(), True)

    def _set_item_hidden(self, item, hidden):
        """
        Recursively set the hidden state of an item and its children.

        Parameters
        ----------
        item : QStandardItem
            The item to set the hidden state for.
        hidden : bool
            Whether to hide the item.
        """
        for row in range(item.rowCount()):
            index = self.tree_view.model().index(row, 0, self.tree_view.model().indexFromItem(item))
            self.tree_view.setRowHidden(row, index.parent(), hidden)

            child_item = item.child(row, 0)
            if child_item and child_item.hasChildren():
                self._set_item_hidden(child_item, hidden)

    def _filter_items(self, item, search_text):
        """
        Recursively filter items based on search text.

        Parameters
        ----------
        item : QStandardItem
            The item to filter.
        search_text : str
            The text to filter by.

        Returns
        -------
        bool
            True if the item or any of its children match the search text.
        """
        match_found = False

        for row in range(item.rowCount()):
            key_item = item.child(row, 0)
            value_item = item.child(row, 1)

            # Check if key or value contains the search text
            key_match = search_text in key_item.text().lower()
            value_match = value_item and search_text in value_item.text().lower()

            # Check children recursively
            child_match = False
            if key_item and key_item.hasChildren():
                child_match = self._filter_items(key_item, search_text)

            # Show this row if it matches or has matching children
            if key_match or value_match or child_match:
                index = self.tree_view.model().index(row, 0, self.tree_view.model().indexFromItem(item))
                self.tree_view.setRowHidden(row, index.parent(), False)
                match_found = True
            else:
                index = self.tree_view.model().index(row, 0, self.tree_view.model().indexFromItem(item))
                self.tree_view.setRowHidden(row, index.parent(), True)

        return match_found

    def save_settings(self):
        """Save the settings to the file."""
        if not self.filename:
            self.filename = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Settings", "", "YAML Files (*.yaml);;All Files (*.*)"
            )[0]

            if not self.filename:
                return

        try:
            # Get settings from model
            settings_dict = self.model.get_settings_dict()

            # Save to file
            with open(self.filename, 'w', encoding="utf-8") as file:
                yaml.dump(settings_dict, file, default_flow_style=False)

            self.path_label.setText(str(self.filename))
            logging.log(0, f"Settings saved to {self.filename}")

        except Exception as e:
            logging.log(1, f"Error saving settings to {self.filename}: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"Error saving settings: {str(e)}"
            )

    def show_help(self):
        """Show help information."""
        help_text = """
        <h2>Settings Editor Help</h2>
        <p>This editor allows you to view and modify settings in a tree structure.</p>

        <h3>Navigation</h3>
        <ul>
            <li>Use the search bar to filter settings</li>
            <li>Double-click on a value to edit it</li>
            <li>Click the Save button to save changes</li>
            <li>Click the Reload button to reload from the file</li>
        </ul>

        <h3>Editing Values</h3>
        <ul>
            <li>Boolean values: Select True or False from the dropdown</li>
            <li>Numbers: Use the spin box to set the value</li>
            <li>Colors: Click on the color to open a color picker</li>
            <li>Lists: Enter comma-separated values</li>
            <li>File paths: Enter the path or click "..." to browse</li>
            <li>Other values: Enter the text directly</li>
        </ul>
        """

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Settings Editor Help")
        msg_box.setTextFormat(QtCore.Qt.RichText)
        msg_box.setText(help_text)
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.exec_()


if __name__ == "__main__":
    import sys
    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)

    editor = SettingsEditor(filename=chisurf.settings.chisurf_settings_file)
    editor.show()

    sys.exit(app.exec_())
