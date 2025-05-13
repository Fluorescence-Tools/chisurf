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


class SettingsItemDelegate(QtWidgets.QStyledItemDelegate):
    """A delegate for editing settings with appropriate widgets based on data type."""

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

        # Get the data type from the model
        data_type = index.data(QtCore.Qt.UserRole)
        value = index.data(QtCore.Qt.DisplayRole)

        # Check if this is a hex color value
        if isinstance(value, str) and self.is_hex_color(value):
            # Create a color dialog for color values
            color = QtGui.QColor(value)
            initial_color = color if color.isValid() else QtGui.QColor("white")

            # Create a button that will open a color dialog
            editor = QtWidgets.QPushButton(parent)
            editor.setAutoFillBackground(True)
            editor.setText("Choose Color")

            # Set the button's background color
            palette = editor.palette()
            palette.setColor(QtGui.QPalette.Button, color)

            # Set contrasting text color for better readability
            brightness = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
            text_color = QtGui.QColor("black") if brightness > 128 else QtGui.QColor("white")
            palette.setColor(QtGui.QPalette.ButtonText, text_color)

            editor.setPalette(palette)

            # Store the index in the button's properties for later use
            editor.setProperty("model_index", index)

            # Connect button click to color dialog
            editor.clicked.connect(self._open_color_dialog)

            return editor
        elif data_type == bool:
            # Create a combobox for boolean values
            editor = QtWidgets.QComboBox(parent)
            editor.addItems(["true", "false"])
            return editor
        elif data_type == int:
            # Create a line edit with integer validator
            editor = QtWidgets.QLineEdit(parent)
            validator = QtGui.QIntValidator(editor)
            editor.setValidator(validator)
            return editor
        elif data_type == float:
            # Create a line edit with double validator
            editor = QtWidgets.QLineEdit(parent)
            validator = QtGui.QDoubleValidator(editor)
            editor.setValidator(validator)
            return editor
        else:
            # Default editor for other types
            return super().createEditor(parent, option, index)

    def _open_color_dialog(self):
        """Open a color dialog when the color button is clicked."""
        # Get the button that triggered this slot
        button = self.sender()
        if not button:
            return

        # Get all necessary data from the button BEFORE opening the color dialog
        current_color = button.palette().color(QtGui.QPalette.Button)
        parent_widget = button.parent()

        # Capture the model index before opening the dialog
        model_index = button.property("model_index")
        if not model_index or not model_index.isValid():
            return

        # Get the model reference before opening the dialog
        model = model_index.model()
        if not model:
            return

        # Open color dialog - after this point, the button might be deleted
        color = QtWidgets.QColorDialog.getColor(current_color, parent_widget)

        if color.isValid():
            # Update the model directly with the new color
            # We're using the captured model_index, not accessing the button anymore
            model.setData(model_index, color.name(), QtCore.Qt.EditRole)

    def setEditorData(self, editor, index):
        """Set the editor data based on the model data."""
        if not index.isValid() or index.column() != 1:
            super().setEditorData(editor, index)
            return

        data_type = index.data(QtCore.Qt.UserRole)
        value = index.data(QtCore.Qt.DisplayRole)

        if isinstance(value, str) and self.is_hex_color(value) and isinstance(editor, QtWidgets.QPushButton):
            # Set the button's color for hex color values
            color = QtGui.QColor(value)
            if color.isValid():
                palette = editor.palette()
                palette.setColor(QtGui.QPalette.Button, color)

                # Set contrasting text color for better readability
                brightness = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
                text_color = QtGui.QColor("black") if brightness > 128 else QtGui.QColor("white")
                palette.setColor(QtGui.QPalette.ButtonText, text_color)

                editor.setPalette(palette)
                # Store the original color value
                editor.setProperty("color_value", value)
        elif data_type == bool and isinstance(editor, QtWidgets.QComboBox):
            # Set the combobox to the current boolean value
            if value.lower() == "true":
                editor.setCurrentIndex(0)
            else:
                editor.setCurrentIndex(1)
        else:
            # For other types, use the default behavior
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        """Set the model data based on the editor data."""
        if not index.isValid() or index.column() != 1:
            super().setModelData(editor, model, index)
            return

        data_type = index.data(QtCore.Qt.UserRole)
        value = index.data(QtCore.Qt.DisplayRole)

        if isinstance(value, str) and self.is_hex_color(value) and isinstance(editor, QtWidgets.QPushButton):
            # Get the color value from the button's property
            color_value = editor.property("color_value")
            if color_value:
                model.setData(index, color_value, QtCore.Qt.EditRole)
        elif data_type == bool and isinstance(editor, QtWidgets.QComboBox):
            # Get the boolean value from the combobox
            value = editor.currentText()
            model.setData(index, value, QtCore.Qt.EditRole)
        elif data_type in (int, float) and isinstance(editor, QtWidgets.QLineEdit):
            # Validate and convert the value
            try:
                text = editor.text()
                if data_type == int:
                    value = int(text)
                else:
                    value = float(text)
                model.setData(index, str(value), QtCore.Qt.EditRole)
            except ValueError:
                # Show an error message if the conversion fails
                QtWidgets.QMessageBox.warning(
                    editor.parent(),
                    "Invalid Input",
                    f"The value must be a valid {data_type.__name__}."
                )
                # Don't update the model
                return
        else:
            # For other types, use the default behavior
            super().setModelData(editor, model, index)


class SettingsTreeModel(QtGui.QStandardItemModel):
    """A model for displaying and editing settings in a tree view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Setting", "Value"])
        self._settings_dict = {}

    def load_settings(self, settings_dict):
        """Load settings from a dictionary into the model."""
        self._settings_dict = settings_dict
        self.clear()
        self.setHorizontalHeaderLabels(["Setting", "Value"])
        self._populate_model(settings_dict)

    def _populate_model(self, settings_dict, parent=None):
        """Recursively populate the model with settings from the dictionary."""
        for key, value in settings_dict.items():
            if isinstance(value, dict):
                # Create a category item
                category_item = QtGui.QStandardItem(key)
                category_item.setEditable(False)
                value_item = QtGui.QStandardItem("")
                value_item.setEditable(False)

                if parent is None:
                    self.appendRow([category_item, value_item])
                else:
                    parent.appendRow([category_item, value_item])

                # Recursively add child items
                self._populate_model(value, category_item)
            else:
                # Create a setting item
                setting_item = QtGui.QStandardItem(key)
                setting_item.setEditable(False)

                # Create a value item with appropriate editor
                value_item = QtGui.QStandardItem(str(value))

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

    def get_settings_dict(self):
        """Convert the model back to a dictionary."""
        result = {}
        self._extract_settings(self.invisibleRootItem(), result)
        return result

    def _extract_settings(self, item, result_dict):
        """Recursively extract settings from the model into a dictionary."""
        for row in range(item.rowCount()):
            key_item = item.child(row, 0)
            value_item = item.child(row, 1)

            key = key_item.text()

            # If this is a category (has children)
            if key_item.hasChildren():
                result_dict[key] = {}
                self._extract_settings(key_item, result_dict[key])
            else:
                # Get the value string and original data type
                value_str = value_item.text()
                data_type = value_item.data(QtCore.Qt.UserRole)

                # Convert value based on the stored data type
                try:
                    if data_type == bool:
                        value = value_str.lower() == 'true'
                    elif data_type == int:
                        value = int(value_str)
                    elif data_type == float:
                        value = float(value_str)
                    elif data_type == type(None):
                        value = None
                    elif data_type == list:
                        # Handle arrays by parsing the string representation
                        try:
                            # Try to evaluate the string as a Python expression
                            # This handles cases like "[1, 2, 3]" or "['a', 'b', 'c']"
                            import ast
                            value = ast.literal_eval(value_str)
                            if not isinstance(value, list):
                                # If it's not a list after evaluation, convert it to a list
                                value = [value]
                        except (SyntaxError, ValueError):
                            # If evaluation fails, try to parse it as a comma-separated list
                            # This handles cases like "1, 2, 3" or "a, b, c"
                            value_str = value_str.strip('[]')
                            if value_str:
                                parts = [part.strip() for part in value_str.split(',')]
                                # Try to convert each part to a number if possible
                                value = []
                                for part in parts:
                                    try:
                                        # Try to convert to int or float
                                        if '.' in part:
                                            value.append(float(part))
                                        else:
                                            value.append(int(part))
                                    except ValueError:
                                        # If conversion fails, keep as string
                                        value.append(part.strip('"\''))
                            else:
                                value = []
                    else:
                        # For other types (like strings), keep as is
                        value = value_str
                except ValueError:
                    # If conversion fails, show a warning and keep as string
                    logging.log(1, f"Warning: Could not convert '{value_str}' to {data_type.__name__} for setting '{key}'. Using string value.")
                    value = value_str

                result_dict[key] = value


class SettingsEditor(QtWidgets.QWidget):
    """A tree-based editor for editing settings."""

    def __init__(
        self,
        *args,
        filename: str = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.filename = filename
        self.settings_dict = {}

        self.setup_ui()

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
        self.model = SettingsTreeModel(self)
        self.tree_view.setModel(self.model)

        # Set custom delegate for editing
        self.delegate = SettingsItemDelegate()
        self.tree_view.setItemDelegate(self.delegate)

        # Add tree view to layout
        layout.addWidget(self.tree_view)

        # Create buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.path_label = QtWidgets.QLabel()
        self.save_button = QtWidgets.QPushButton("Save")
        self.reload_button = QtWidgets.QPushButton("Reload")

        button_layout.addWidget(self.path_label, 1)
        button_layout.addWidget(self.reload_button)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Connect signals
        self.save_button.clicked.connect(self.save_settings)
        self.reload_button.clicked.connect(lambda: self.load_file(self.filename))
        self.search_bar.textChanged.connect(self.filter_settings)

        # Set window properties
        self.setWindowTitle("Settings Editor")
        self.resize(800, 600)

    def load_file(self, filename: str = None):
        """Load settings from a file."""
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
        """Filter the settings tree to show only items matching the search text."""
        if not text:
            # If search text is empty, show all items
            self._show_all_items()
            return

        # Convert search text to lowercase for case-insensitive search
        search_text = text.lower()

        # Start with all items hidden
        self._hide_all_items()

        # Find and show matching items
        self._filter_tree_items(self.model.invisibleRootItem(), search_text, set())

    def _hide_all_items(self):
        """Hide all items in the tree."""
        for row in range(self.model.rowCount()):
            index = self.model.index(row, 0)
            self.tree_view.setRowHidden(row, QtCore.QModelIndex(), True)
            self._hide_children(index)

    def _show_all_items(self):
        """Show all items in the tree."""
        for row in range(self.model.rowCount()):
            index = self.model.index(row, 0)
            self.tree_view.setRowHidden(row, QtCore.QModelIndex(), False)
            self._show_children(index)
        self.tree_view.expandAll()

    def _hide_children(self, parent_index):
        """Recursively hide all children of a parent item."""
        for row in range(self.model.rowCount(parent_index)):
            child_index = self.model.index(row, 0, parent_index)
            self.tree_view.setRowHidden(row, parent_index, True)
            if self.model.hasChildren(child_index):
                self._hide_children(child_index)

    def _show_children(self, parent_index):
        """Recursively show all children of a parent item."""
        for row in range(self.model.rowCount(parent_index)):
            child_index = self.model.index(row, 0, parent_index)
            self.tree_view.setRowHidden(row, parent_index, False)
            if self.model.hasChildren(child_index):
                self._show_children(child_index)

    def _filter_tree_items(self, parent_item, search_text, visible_paths):
        """
        Recursively filter tree items based on search text.
        Returns True if this item or any of its children match the search.
        """
        any_child_visible = False

        # Check all children of this item
        for row in range(parent_item.rowCount()):
            child_item = parent_item.child(row, 0)
            value_item = parent_item.child(row, 1) if parent_item.columnCount() > 1 else None

            # Skip if child_item is None
            if not child_item:
                continue

            # Get the model index for this item
            if parent_item == self.model.invisibleRootItem():
                parent_index = QtCore.QModelIndex()
            else:
                parent_index = self.model.indexFromItem(parent_item)

            child_index = self.model.indexFromItem(child_item)
            row_in_parent = child_index.row()

            # Check if this item matches the search text
            item_text = child_item.text().lower()
            value_text = value_item.text().lower() if value_item else ""

            item_matches = search_text in item_text or search_text in value_text

            # Check if any children match (recursive)
            children_match = False
            if child_item.hasChildren():
                children_match = self._filter_tree_items(child_item, search_text, visible_paths)

            # If this item or any of its children match, make it visible
            if item_matches or children_match:
                self.tree_view.setRowHidden(row_in_parent, parent_index, False)
                self.tree_view.expand(child_index)
                any_child_visible = True
            else:
                self.tree_view.setRowHidden(row_in_parent, parent_index, True)

        return any_child_visible

    def save_settings(self):
        """Save settings to the file."""
        if not self.filename:
            return

        try:
            # Extract settings from model
            settings_dict = self.model.get_settings_dict()

            # Save to file
            with io.zipped.open_maybe_zipped(self.filename, "w") as file:
                yaml.dump(settings_dict, file, default_flow_style=False)

            # Update cs_settings with the new settings
            cs_settings.clear()
            cs_settings.update(settings_dict)

            # Update local variables in chisurf.settings module
            import sys
            chisurf_settings_module = sys.modules['chisurf.settings']
            # Use the same approach as in __init__.py to update local variables
            chisurf_settings_module_dict = vars(chisurf_settings_module)
            chisurf_settings_module_dict.update(cs_settings)

            logging.log(0, f"Settings saved to {self.filename} and cs_settings updated")

            # Show info popup about potential restart requirement
            QtWidgets.QMessageBox.information(
                self,
                "Settings Updated",
                "Settings have been updated successfully.\n\n"
                "ChiSurf may need to be restarted to ensure all settings are properly applied throughout the software."
            )

        except Exception as e:
            logging.log(1, f"Error saving settings to {self.filename}: {e}")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    editor = SettingsEditor(filename=chisurf.settings.chisurf_settings_file)
    editor.show()
    sys.exit(app.exec_())
