from __future__ import annotations

import pathlib
import yaml
import re
import os

from qtpy import QtCore, QtGui, QtWidgets

import chisurf as cs
import chisurf.core.fio as io
from chisurf import logging
import chisurf.core.settings
from chisurf.core.settings import cs_settings

LIST_SEP = "|"


# Relative path (from project root) to the main settings documentation.
SETTINGS_DOC_REL_PATH = "docs/chisurf_settings.md"


# Cache for dynamically computed mapping from root setting keys to
# documentation headings, keyed by the YAML basename
_SETTINGS_HELP_SECTION_CACHE: dict[str, dict[str, str]] = {}


def _slugify_heading_for_docs(text: str) -> str:
    """Replicate the HelpWidget heading slug logic for anchor IDs.

    This matches cs.plugins.help.HelpWidget._slugify_heading so that
    anchors computed here correspond to those used when rendering the
    Markdown in the help browser.
    """
    slug = text.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug or "section"


def get_help_topic_for_setting(setting_path: str, source_filename: str | None) -> str | None:
    """Return a help topic string for a given settings path.

    The topic string is either of the form "docs/chisurf_settings.md#anchor"
    (for direct Markdown navigation) or None if no mapping exists.
    """
    if not source_filename:
        return None

    basename = os.path.basename(source_filename)

    # Resolve (and cache) the mapping from root keys to headings for this file
    section_map = _SETTINGS_HELP_SECTION_CACHE.get(basename)
    if section_map is None:
        section_map = _build_help_section_map_for_file(basename)
        _SETTINGS_HELP_SECTION_CACHE[basename] = section_map or {}

    if not section_map:
        return None

    # Only use the root key (top-level) to determine the section
    root_key = setting_path.split(".", 1)[0]
    heading = section_map.get(root_key)
    if not heading:
        return None

    anchor = _slugify_heading_for_docs(heading)
    return f"{SETTINGS_DOC_REL_PATH}#{anchor}"


def _build_help_section_map_for_file(basename: str) -> dict[str, str]:
    """Build a mapping from root setting keys to documentation headings.

    The mapping is derived dynamically from docs/chisurf_settings.md and the
    actual settings structures, so it does not hard-code section numbers or
    titles.
    """
    try:
        base = pathlib.Path(cs.__file__).resolve().parent
        root = base.parent
        md_path = root / SETTINGS_DOC_REL_PATH
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return {}

    # Collect all Markdown headings (without the leading # characters)
    headings: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if not m:
            continue
        heading = m.group(2).strip()
        if heading:
            headings.append(heading)

    # Normalized helper to locate "Top-level flags" irrespective of hyphen type
    def _find_top_level_flags_heading() -> str | None:
        for h in headings:
            norm = h.lower().replace("‑", "-")  # normalize non-breaking hyphen
            if "top-level flags" in norm:
                return h
        return None

    mapping: dict[str, str] = {}

    if basename == "settings_chisurf.yaml":
        # Use the loaded cs_settings keys as root keys
        try:
            cs_dict = getattr(cs.core.settings, "cs_settings", {}) or {}
            root_keys = [str(k) for k in cs_dict.keys()]
        except Exception:
            root_keys = []

        # First, try to find a heading that mentions each key in backticks
        for key in root_keys:
            needle = f"`{key}`"
            for h in headings:
                if needle in h:
                    mapping[key] = h
                    break

        # For any remaining top-level keys, fall back to the generic
        # "Top-level flags" heading if present.
        top_flags = _find_top_level_flags_heading()
        if top_flags:
            for key in root_keys:
                if key not in mapping:
                    mapping[key] = top_flags
        return mapping

    # Unknown YAML file: no mapping
    return {}

def _build_documentation_dict() -> dict:
    """Build a mapping from setting keys to their descriptions from the markdown docs."""
    try:
        base = pathlib.Path(cs.__file__).resolve().parent
        root = base.parent
        md_path = root / SETTINGS_DOC_REL_PATH
        if not md_path.exists():
            return {}
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return {}
    
    doc_dict = {}
    current_root = None
    current_keys = []
    current_desc = []
    
    def save_current():
        if not current_keys: return
        desc = " ".join(current_desc).strip()
        if not desc: return
        for k in current_keys:
            if current_root:
                doc_dict[f"{current_root}.{k}"] = desc
            else:
                doc_dict[k] = desc

    for line in text.splitlines():
        m_root = re.match(r'^###\s+[\d\.]+\s+`([^`]+)`(.*)', line)
        if m_root:
            save_current()
            current_root = m_root.group(1)
            cat_desc = m_root.group(2).strip()
            if cat_desc:
                cat_desc = cat_desc.strip("()")
                if cat_desc:
                    cat_desc = cat_desc[0].upper() + cat_desc[1:]
                doc_dict[current_root] = cat_desc
            current_keys = []
            current_desc = []
            continue
            
        if re.match(r'^###\s+[\d\.]+\s+Top', line, re.IGNORECASE):
            save_current()
            current_root = None
            current_keys = []
            current_desc = []
            continue
            
        if re.match(r'^- \*\*', line):
            save_current()
            keys_raw = re.findall(r'\*\*`?([^`\*]+)`?\*\*', line)
            if keys_raw:
                current_keys = keys_raw
            current_desc = []
            
            after_keys = line
            for kr in keys_raw:
                after_keys = after_keys.replace(f"**`{kr}`**", "").replace(f"**{kr}**", "")
            after_keys = after_keys.replace("-", "").replace(",", "").replace("/", "").strip()
            if after_keys:
                current_desc.append(after_keys)
            continue
            
        if current_keys:
            if line.startswith("###"):
                pass
            elif not line.strip() and not current_desc:
                continue
            else:
                current_desc.append(line.strip())
                
    save_current()
    return doc_dict

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
        
        # Ensure we preserve the original format for extreme values
        if abs(value) < 1e-10 or abs(value) > 1e10:
            # For extreme values, ensure we keep the decimal point and at least one digit
            if '.0e' in text:
                # Already has the format we want
                pass
            elif 'e' in text and '.' not in text:
                # Add .0 before the exponent
                text = text.replace('e', '.0e')
        
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

        # Prefer typed value from UserRole; fallback to EditRole
        value = index.data(QtCore.Qt.UserRole)
        if value is None and index.data(QtCore.Qt.EditRole) is not None:
            value = index.data(QtCore.Qt.EditRole)
        data_type = type(value)

        if self._is_theme_setting(setting_path):
            editor = self._create_theme_editor(parent, value, tooltip)
            return editor

        # Create appropriate editor based on data type
        if data_type == bool or (isinstance(value, str) and value.strip().lower() in ("true", "false")):
            editor = QtWidgets.QCheckBox(parent)
            # Determine checked state robustly for both bools and string booleans
            checked = value if data_type == bool else (str(value).strip().lower() == "true")
            editor.setChecked(bool(checked))
            if tooltip:
                editor.setToolTip(tooltip)
            # Commit data when state changes
            editor.stateChanged.connect(lambda _state: self.commitData.emit(editor))
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

    def _is_theme_setting(self, setting_path: str) -> bool:
        return setting_path == "gui.style_sheet"

    def _create_theme_editor(self, parent, value, tooltip):
        current_value = "" if value is None else str(value)
        combo = QtWidgets.QComboBox(parent)

        themes = []
        try:
            package_styles_dir = pathlib.Path(cs.__file__).parent / "gui" / "styles"
            if package_styles_dir.is_dir():
                for p in sorted(package_styles_dir.glob("*.qss")):
                    name = p.name
                    if name not in themes:
                        themes.append(name)
        except Exception as e:
            logging.log(1, f"Error while listing package styles: {e}")

        try:
            user_styles_dir = cs.core.settings.get_path('settings') / 'styles'
            if user_styles_dir.is_dir():
                for p in sorted(user_styles_dir.glob("*.qss")):
                    name = p.name
                    if name not in themes:
                        themes.append(name)
        except Exception as e:
            logging.log(1, f"Error while listing user styles: {e}")

        if not themes:
            combo.addItem("<no styles available>")
            combo.setProperty("hasThemes", False)
            combo.setEnabled(False)
            if tooltip:
                combo.setToolTip(tooltip)
            return combo

        for name in themes:
            combo.addItem(name)

        if current_value and current_value in themes:
            index = combo.findText(current_value)
            if index >= 0:
                combo.setCurrentIndex(index)
        else:
            combo.setCurrentIndex(0)

        if tooltip:
            combo.setToolTip(tooltip)

        combo.setProperty("hasThemes", True)

        return combo

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

        setting_path = self._get_setting_path(index)

        if self._is_theme_setting(setting_path) and isinstance(editor, QtWidgets.QComboBox):
            if editor.property("hasThemes") is False:
                return
            new_value = editor.currentText()
            model.setData(index, new_value, QtCore.Qt.EditRole)
            model.setData(index, new_value, QtCore.Qt.UserRole)
            return

        # Prefer typed value from UserRole; fallback to EditRole
        value = index.data(QtCore.Qt.UserRole)
        if value is None and index.data(QtCore.Qt.EditRole) is not None:
            value = index.data(QtCore.Qt.EditRole)
        data_type = type(value)

        # Handle checkbox editors explicitly regardless of original data type
        if isinstance(editor, QtWidgets.QCheckBox):
            new_value = editor.isChecked()
            model.setData(index, new_value, QtCore.Qt.EditRole)
            model.setData(index, new_value, QtCore.Qt.UserRole)
            return

        if data_type == bool:
            # For boolean values, get from checkbox
            if hasattr(editor, "isChecked"):
                try:
                    new_value = bool(editor.isChecked())
                except Exception:
                    new_value = bool(value)
            else:
                # Fallback in case of unexpected editor type
                try:
                    new_value = (str(editor.currentText()) == "True")
                except Exception:
                    new_value = bool(value)
            model.setData(index, new_value, QtCore.Qt.EditRole)
            model.setData(index, new_value, QtCore.Qt.UserRole)
        elif self.is_hex_color(value):
            # For hex color values, get from button text
            button = editor
            model.setData(index, button.text(), QtCore.Qt.EditRole)
            model.setData(index, button.text(), QtCore.Qt.UserRole)
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
            model.setData(index, type(value)(items), QtCore.Qt.UserRole)
        elif data_type == int:
            # For integers, get from spin box
            spin_box = editor
            model.setData(index, spin_box.value(), QtCore.Qt.EditRole)
            model.setData(index, spin_box.value(), QtCore.Qt.UserRole)
        elif data_type == float:
            # For floats, get from double spin box or line edit
            if isinstance(editor, QtWidgets.QDoubleSpinBox):
                new_value = editor.value()
                model.setData(index, new_value, QtCore.Qt.EditRole)
                model.setData(index, new_value, QtCore.Qt.UserRole)
            else:
                # For line edit (scientific notation)
                try:
                    text = editor.text()
                    # Convert to float, preserving scientific notation
                    new_value = float(text)
                    model.setData(index, new_value, QtCore.Qt.EditRole)
                    model.setData(index, new_value, QtCore.Qt.UserRole)
                except ValueError:
                    # If conversion fails, keep the original value
                    logging.log(1, f"Warning: Could not convert '{text}' to float. Using original value.")
                    model.setData(index, value, QtCore.Qt.EditRole)
                    model.setData(index, value, QtCore.Qt.UserRole)
        elif isinstance(value, str) and os.path.sep in value:
            # For file paths, get from line edit in the widget
            widget = editor
            line_edit = widget.property("lineEdit")
            model.setData(index, line_edit.text(), QtCore.Qt.EditRole)
            model.setData(index, line_edit.text(), QtCore.Qt.UserRole)
        else:
            # For other types, get from line edit
            line_edit = editor
            text = line_edit.text()

            # Try to convert to the original data type
            try:
                if data_type != str:
                    converted_value = data_type(text)
                    model.setData(index, converted_value, QtCore.Qt.EditRole)
                    model.setData(index, converted_value, QtCore.Qt.UserRole)
                else:
                    model.setData(index, text, QtCore.Qt.EditRole)
                    model.setData(index, text, QtCore.Qt.UserRole)
            except ValueError:
                # If conversion fails, use the string value
                model.setData(index, text, QtCore.Qt.EditRole)
                model.setData(index, text, QtCore.Qt.UserRole)


class SettingsTreeModel(QtGui.QStandardItemModel):
    """A tree model for displaying and editing settings."""

    def __init__(self, parent=None, documentation_dict=None, source_filename: str | None = None):
        """
        Initialize the model.

        Parameters
        ----------
        parent : QObject, optional
            The parent object.
        documentation_dict : dict, optional
            A dictionary containing documentation for settings.
        source_filename : str, optional
            Path to the YAML settings file being edited; used to resolve
            per-setting help topics into documentation anchors.
        """
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Setting", "Value", "Help"])
        self.documentation_dict = documentation_dict or {}
        self.source_filename: str | None = source_filename

    def load_settings(self, settings_dict):
        """
        Load settings into the model.

        Parameters
        ----------
        settings_dict : dict
            The settings dictionary to load.
        """
        self.clear()
        self.setHorizontalHeaderLabels(["Setting", "Value", "Help"])
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
            # Store the original typed value in a dedicated role
            value_item.setData(value, QtCore.Qt.UserRole)
            # Also set EditRole for convenience/editing
            value_item.setData(value, QtCore.Qt.EditRole)

            # Set display text based on data type
            if isinstance(value, dict):
                # For dictionaries, don't set display text (will be populated with children)
                pass
            elif isinstance(value, (list, tuple)):
                # For lists, show comma-separated values with special handling for complex items
                if not value:
                    # Empty list
                    value_item.setText("")
                elif any(isinstance(item, dict) for item in value):
                    # List contains dictionaries - show a placeholder
                    value_item.setText("[complex list - edit with caution]")
                else:
                    # Regular list - convert None to 'None' for display
                    items_str = []
                    for item in value:
                        if item is None:
                            items_str.append("None")
                        else:
                            items_str.append(str(item))
                    value_item.setText(LIST_SEP.join(items_str))
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
            elif value is None:
                # For None values, show "None"
                value_item.setText("None")
            else:
                # For other types, show string representation
                value_item.setText(str(value))

            # Add tooltip if documentation exists
            current_path = f"{path}.{key}" if path else key
            desc = self.documentation_dict.get(current_path)
            if desc:
                key_item.setToolTip(desc)
                value_item.setToolTip(desc)

            # Create help item with optional topic mapping
            help_item = QtGui.QStandardItem()
            help_item.setEditable(False)
            topic = get_help_topic_for_setting(current_path, getattr(self, "source_filename", None))
            
            if desc:
                import textwrap
                wrapped_desc = textwrap.fill(desc, width=60)
                help_item.setText(wrapped_desc)
                help_item.setToolTip(desc)
                # Ensure tooltip is on all columns
                key_item.setToolTip(desc)
                value_item.setToolTip(desc)
                if topic:
                    help_item.setData(topic, QtCore.Qt.UserRole)
            elif topic:
                help_item.setText("")
                help_item.setData(topic, QtCore.Qt.UserRole)
            else:
                help_item.setText("")

            # Add items to model
            row = [key_item, value_item, help_item]

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
                # Prefer the original typed value stored under UserRole
                value = value_item.data(QtCore.Qt.UserRole)
                # Fallback to EditRole if UserRole is not set
                if value is None and value_item.data(QtCore.Qt.EditRole) is not None:
                    value = value_item.data(QtCore.Qt.EditRole)

                # Convert string values to appropriate types if possible
                if isinstance(value, str):
                    value_str = value

                    # Helper to parse a single scalar token into the right type
                    def _parse_scalar(token: str):
                        t = token.strip()
                        if t == "":
                            return ""  # keep empty string
                        tl = t.lower()
                        if tl in ("none", "null"):
                            return None
                        if tl == "true":
                            return True
                        if tl == "false":
                            return False
                        # Integer (including leading sign)
                        if re.fullmatch(r"[+-]?\d+", t):
                            try:
                                return int(t)
                            except ValueError:
                                pass
                        # Float (including scientific notation)
                        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?", t):
                            try:
                                return float(t)
                            except ValueError:
                                pass
                        return t

                    # Detect list encoded as a string via LIST_SEP and convert items
                    is_list = LIST_SEP in value_str
                    if is_list:
                        parts = value_str.split(LIST_SEP)
                        value = [_parse_scalar(p) for p in parts]
                    else:
                        # Handle special cases and scalars
                        if value_str == "":
                            # Previously used to encode empty list; but keep empty string unless context requires list
                            value = ""
                        elif value_str == "[]":
                            value = []
                        else:
                            value = _parse_scalar(value_str)

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
        self.documentation_dict = documentation_dict or _build_documentation_dict()
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
        self.tree_view.setWordWrap(True)

        # Create model
        self.model = self.create_model()
        self.tree_view.setModel(self.model)

        # Set custom delegate for editing
        self.delegate = self.create_delegate()
        self.tree_view.setItemDelegate(self.delegate)

        # React to clicks in the Help column
        self.tree_view.clicked.connect(self.on_tree_view_clicked)

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
        return SettingsTreeModel(self, self.documentation_dict, source_filename=self.filename)

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

            # Ensure the model knows which file it represents (for help topics)
            try:
                if hasattr(self.model, "source_filename"):
                    self.model.source_filename = filename
            except Exception:
                pass

            # Load settings into model
            self.model.load_settings(self.settings_dict)

            # Expand all items
            self.tree_view.expandAll()

            # Resize columns and rows to content
            self.tree_view.resizeColumnToContents(0)
            self.tree_view.resizeRowsToContents()

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

    def on_tree_view_clicked(self, index: QtCore.QModelIndex):
        """Handle clicks in the settings tree.

        Clicking the Help column (column 2) opens the corresponding
        documentation section in the help browser when available.
        """
        try:
            if not index.isValid():
                return
            if index.column() != 2:
                return
            topic = index.data(QtCore.Qt.UserRole)
            if not topic:
                return
            self.open_help_topic(str(topic))
        except Exception as e:
            logging.log(1, f"Error handling help click: {e}")

    def open_help_topic(self, topic: str) -> None:
        """Open the help browser for a given topic string.

        The topic is typically of the form "docs/chisurf_settings.md#anchor".
        Preference is given to delegating to the main window's
        ``open_context_help_for_reader`` when available; otherwise we
        fall back to opening the help plugin directly.
        """
        if not topic:
            return

        txt = str(topic).strip()
        if not txt:
            return

        # 1) Prefer delegating to the main window if available
        try:
            main_win = getattr(cs, "cs", None)
            if main_win is not None and hasattr(main_win, "open_context_help_for_reader"):
                try:
                    main_win.open_context_help_for_reader(txt)
                    return
                except Exception:
                    pass
        except Exception:
            pass

        # 2) Fallback: directly use the help plugin
        try:
            import importlib
            import pathlib as _pl

            try:
                help_plugin = importlib.import_module("chisurf.plugins.core.help")
            except Exception:
                help_plugin = importlib.import_module("chisurf.plugins.core.help")

            # Reuse a singleton window attached to the cs module
            window = getattr(cs, "_settings_help_window", None)
            if window is None or not isinstance(window, help_plugin.HelpWidget):
                window = help_plugin.HelpWidget()
                try:
                    window.destroyed.connect(lambda _=None: setattr(cs, "_settings_help_window", None))
                except Exception:
                    pass
                cs._settings_help_window = window

            handled = False
            try:
                path_part = txt
                anchor = None
                if "#" in txt:
                    path_part, frag = txt.split("#", 1)
                    path_part = path_part.strip()
                    anchor = frag.strip() or None

                if path_part.lower().endswith(".md"):
                    raw_path = _pl.Path(path_part)

                    # Resolve relative paths against the project root (same
                    # convention as the help plugin itself).
                    if not raw_path.is_absolute():
                        try:
                            base = _pl.Path(cs.__file__).resolve().parent
                            root = base.parent
                            candidate = (root / raw_path).resolve()
                        except Exception:
                            candidate = raw_path
                    else:
                        candidate = raw_path

                    if candidate.exists():
                        try:
                            window.open_markdown_path(candidate, anchor)
                            handled = True
                        except Exception:
                            handled = False
            except Exception:
                handled = False

            # If we could not resolve a concrete Markdown path, fall back to
            # using the free-text filter.
            if not handled and txt:
                try:
                    window.filter_line_edit.setText(txt)
                except Exception:
                    pass

            window.show()
            try:
                window.raise_()
                window.activateWindow()
            except Exception:
                pass
        except Exception as e:
            try:
                QtWidgets.QMessageBox.critical(self, "Help Error", f"Could not open help: {e}")
            except Exception:
                logging.log(1, f"Could not open help: {e}")

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
            <li>Boolean values: Use the checkbox to toggle True/False</li>
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

    editor = SettingsEditor(filename=cs.core.settings.chisurf_settings_file)
    editor.show()

    sys.exit(app.exec_())
