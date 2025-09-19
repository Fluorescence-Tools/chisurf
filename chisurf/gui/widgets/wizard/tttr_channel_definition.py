# This file contains the DetectorWizardPage class which is used for detector and PIE-window definition.
# The UI for this class is now defined in a separate .ui file (detector_wizard_page.ui) instead of
# being created programmatically. This makes it easier to maintain and modify the UI.
# The UI file is loaded in the __init__ method of the DetectorWizardPage class.
#
# A helper method _hide_layout_widgets is used to hide/show all widgets in a layout.
# This method is used instead of trying to access widget containers directly,
# which can cause AttributeError if the widget names don't match between the
# code and the UI file.

import sys
import json
import pathlib
import tempfile
import numpy as np
from qtpy.QtWidgets import (
    QApplication, QWizard, QWizardPage, QVBoxLayout, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QTextEdit, QDialog,
    QMessageBox, QHBoxLayout, QGridLayout, QFileDialog, QToolButton, QWidget,
    QComboBox, QInputDialog, QDoubleSpinBox, QCheckBox
)

from qtpy.QtCore import Signal, Qt
from qtpy import QtWidgets as _QtWidgets
from qtpy import QtCore as _QtCore
from qtpy import uic as _uic

def qtpy_loadUi(path, baseinstance=None):
    return _uic.loadUi(path, baseinstance)

from chisurf.settings.path_utils import get_path
from chisurf.settings.file_utils import safe_open_file
import tttrlib
from chisurf.fio.fluorescence.bhfiles import BeckerHicklSetReader
from chisurf.fio import write_jordi
from chisurf.plugins.jordi_g_factor import JordiGFactorCalculator, DataCurve

# Path to the central detector setups file
DETECTOR_SETUPS_FILE = get_path('settings') / 'detector_setups.json'

help_text = """You can either load an existing detector Pulsed-Interleaved Excitation (PIE) 
window definition by clicking on the '...' button to define channels, or define your own PIE 
and detector settings by editing the tables below. New PIE windows and detector windows can 
be added by clicking the "Add" button next to the detector name field. The "Edit" button 
displays a JSON file representing the data, and the "Save" button allows you to save your 
channel configuration.

You can also select from predefined setups using the Setup dropdown, or save your current 
configuration as a new setup.

The TTTR reading routine section allows you to specify the file type and time resolution 
parameters used when reading TTTR files. These settings will be saved with your setup.
"""

def load_detector_setups(file_path=None):
    """Load detector setups from the central settings file or a custom file.

    If the central setups file does not exist, inform the user how to create it
    and allow suppressing this warning in the future.
    """
    path = pathlib.Path(file_path or DETECTOR_SETUPS_FILE)
    # Read preference for showing the warning
    try:
        import chisurf.settings  # local import to avoid circulars at import time
        show_warning = bool(chisurf.settings.cs_settings.get('warn_missing_detector_setups', True))
    except Exception:
        show_warning = True

    if not path.exists():
        # Show dialog only in GUI context and when using the default file
        is_default = (file_path is None) or (path == DETECTOR_SETUPS_FILE)
        app_running = False
        try:
            from qtpy.QtWidgets import QApplication  # local import
            app_running = QApplication.instance() is not None
        except Exception:
            app_running = False

        if show_warning and is_default and app_running:
            msg = QMessageBox()
            msg.setWindowTitle("Detector setups file not found")
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Detector setups file was not found:\n{str(path)}")
            msg.setInformativeText("You can create it by saving a setup from the Detector Wizard.\n"
                                   "Use the 'Save Settings' button to store your configuration.\n"
                                   "Alternatively, choose an existing JSON with the '...' button.")
            try:
                # Add 'Don't show again' checkbox
                cb = QCheckBox("Don't show this warning again")
                msg.setCheckBox(cb)
            except Exception:
                cb = None
            # Add action button to open the Detector Wizard
            open_btn = msg.addButton("Open Detector Wizard", QMessageBox.ActionRole)
            ok_btn = msg.addButton(QMessageBox.Ok)
            msg.exec_()
            # Persist suppression if chosen
            try:
                if cb is not None and cb.isChecked():
                    from chisurf.settings.settings_utils import set_warn_missing_detector_setups as _set_warn
                    _set_warn(False)
                    try:
                        chisurf.settings.cs_settings['warn_missing_detector_setups'] = False
                    except Exception:
                        pass
            except Exception:
                pass
            # If user chose to open the wizard, show it and then try to load again
            try:
                if msg.clickedButton() is open_btn:
                    # DetectorWizard is defined in this module
                    wiz = DetectorWizard()
                    wiz.exec_()
                    if path.exists():
                        return safe_open_file(
                            file_path=path,
                            processor=json.load,
                            default_value={"setups": {}}
                        )
            except Exception:
                pass
        # Return empty default
        return {"setups": {}}

    # File exists; load normally
    return safe_open_file(
        file_path=path,
        processor=json.load,
        default_value={"setups": {}}
    )

def save_detector_setups(setups_data, file_path=None, replace=False):
    """Save detector setups to the central settings file or a custom file.

    Args:
        setups_data: The data to save
        file_path: Optional custom path to save to. If None, uses DETECTOR_SETUPS_FILE.
    """
    try:
        save_path = file_path or DETECTOR_SETUPS_FILE

        # Try to load existing data to preserve unrelated keys when not replacing; with replace=True, write as-is
        try:
            if replace:
                updated_data = setups_data
            elif pathlib.Path(save_path).exists() and pathlib.Path(save_path).stat().st_size > 0:
                existing_data = load_detector_setups(save_path)
                # Start from existing data, then merge incoming setups (add/update only)
                updated_data = existing_data if isinstance(existing_data, dict) else {}
                if isinstance(setups_data, dict):
                    for k, v in setups_data.items():
                        if k == "setups":
                            # Merge setups: add or update keys, keep others intact
                            updated_data.setdefault("setups", {})
                            if isinstance(v, dict):
                                updated_data["setups"].update(v)
                        else:
                            updated_data[k] = v
                else:
                    updated_data = setups_data
            else:
                updated_data = setups_data
        except Exception:
            updated_data = setups_data

        # Create directory if it doesn't exist
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Write the updated data back to the file
        with open(save_path, 'w') as f:
            json.dump(updated_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving detector setups: {e}")
        return False

# Initial PIE-Windows and Detectors
_initial_windows = {
    "prompt": (0, 2048),
    "delayed": (2048, 4095)
}

_initial_detectors = {
    "green":  {"chs": [8, 0, 3], "micro_time_ranges": [(0, 4095)], "g_factor": 1, "l1": 0, "l2": 0},
    "red":    {"chs": [9, 1, 2], "micro_time_ranges": [(0, 2048)], "g_factor": 1, "l1": 0, "l2": 0},
    "yellow": {"chs": [9, 1, 2], "micro_time_ranges": [(2048, 4095)], "g_factor": 1, "l1": 0, "l2": 0},
}

# Initial TTTR reading routine settings
_initial_tttr_reading = {
    "file_type": "SPC-130",
    "macro_time_resolution": 50.0,  # in nanoseconds
    "micro_time_resolution": 50.0,  # in picoseconds
    "micro_time_binning": 1,
    "excitation_period": 13.6,  # in nanoseconds
    "g_factor": 1.08316,
    "l1": 0.03080,
    "l2": 0.03680
}


class JsonEditorDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit JSON Settings")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout(self)

        self.json_editor = QTextEdit(self)
        self.json_editor.setText(json.dumps(data, indent=4))
        layout.addWidget(self.json_editor)

        self.save_button = QPushButton("Save JSON", self)
        self.save_button.clicked.connect(self._on_save)
        layout.addWidget(self.save_button)

    def _on_save(self):
        try:
            self.edited_data = json.loads(self.json_editor.toPlainText())
            self.accept()
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", "Invalid JSON format.")

    def get_edited_data(self):
        return getattr(self, "edited_data", None)


class DetectorWizardPage(QWizardPage):
    detectorsChanged = Signal()

    def __init__(self, json_file=None, *args, show_edit_json=False, show_save=False,
                 show_setups_file=True, show_setup_selection=True, show_help=True,
                 show_tttr_reading=True, show_tables=True, show_add_inputs=True,
                 allow_finish=True, **kwargs):
        """Initialize the DetectorWizardPage.
        
        This class uses a UI file (detector_wizard_page.ui) for its layout and widgets.
        The UI file is loaded in the __init__ method and all signals are connected to their
        respective slots.

        Args:
            json_file (str, optional): Path to a JSON file to load. Defaults to None.
            show_edit_json (bool, optional): Whether to show the "Edit JSON" button. Defaults to False.
            show_save (bool, optional): Whether to show the "Save" button. Defaults to False.
            show_setups_file (bool, optional): Whether to show the setups file section. Defaults to True.
            show_setup_selection (bool, optional): Whether to show the setup selection section. Defaults to True.
            show_help (bool, optional): Whether to show the help button and text. Defaults to True.
            show_tttr_reading (bool, optional): Whether to show the TTTR reading routine section. Defaults to True.
            show_tables (bool, optional): Whether to show the PIE-Windows and Detectors tables. Defaults to True.
            show_add_inputs (bool, optional): Whether to show the controls for adding windows and detectors. Defaults to True.
            *args: Additional positional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.setTitle("Detectors and PIE-window definition")
        self.current_setup_name = None
        self.current_setups_file = str(DETECTOR_SETUPS_FILE)
        self._selected_detector_info = None
        self.show_edit_json = show_edit_json
        self.show_save = show_save
        self.show_setups_file = show_setups_file
        self.show_setup_selection = show_setup_selection
        self.show_help = show_help
        self.show_tttr_reading = show_tttr_reading
        self.show_tables = show_tables
        self.show_add_inputs = show_add_inputs

        # Protection flags/state for G-Factor edits
        # Only direct user edits or internal calculator/data loading may change g-factor fields
        self._allow_g_update = False  # internal whitelist for programmatic updates
        self._g_user_editing = {}     # row -> bool, True while the user is actively editing
        self._g_last_valid = {}       # row -> last accepted string value

        # Load the UI file
        ui_file_path = pathlib.Path(__file__).parent / "detector_wizard_page.ui"
        qtpy_loadUi(str(ui_file_path), self)

        # Set initial values
        self.setups_file_le.setText(str(DETECTOR_SETUPS_FILE))
        
        # Connect signals
        self.load_setups_file_button.clicked.connect(self._on_load_setups_file)
        self.setup_combo.currentIndexChanged.connect(self._on_setup_changed)
        self.save_setup_button.clicked.connect(self._on_save_setup)
        self.rename_setup_button.clicked.connect(self._on_rename_setup)
        self.delete_setup_button.clicked.connect(self._on_delete_setup)
        self.help_button.toggled.connect(self._toggle_help)
        self.read_tttr_button.clicked.connect(self._read_from_tttr_file)
        self.micro_time_le.textChanged.connect(self._update_effective_resolution)
        self.micro_binning_combo.currentTextChanged.connect(self._update_effective_resolution)
        self.windows_form.itemDoubleClicked.connect(self._remove_window)
        self.detectors_form.itemDoubleClicked.connect(self._remove_detector)
        self.add_window_button.clicked.connect(self._add_window)
        self.add_detector_button.clicked.connect(self._add_detector)
        self.edit_json_button.clicked.connect(self._edit_json)
        self.save_button.clicked.connect(self._on_save)
        self.toolButton_calc_g_factor.clicked.connect(self._on_calc_g_factor)
        
        # Set help text
        self.help_text.setText(help_text)
        self.help_text.setVisible(False)
        
        # Set visibility based on parameters
        # Use the helper method to hide/show widgets in layouts
        self._hide_layout_widgets(self.setups_file_layout, self.show_setups_file)
        self._hide_layout_widgets(self.setup_layout, self.show_setup_selection)
        self._hide_layout_widgets(self.tttr_layout, self.show_tttr_reading)
        self._hide_layout_widgets(self.gridLayout_3, self.show_tables)
        self._hide_layout_widgets(self.gridLayout_2, self.show_tables)
        self._hide_layout_widgets(self.controls, self.show_add_inputs)
        
        # For widgets that are directly accessible, we can use setVisible directly
        if not self.show_help:
            self.help_text.setVisible(False)
        self.edit_json_button.setVisible(self.show_edit_json)
        self.save_button.setVisible(self.show_save)
        
        # Initialize file type combo
        self.file_type_combo.addItem("Auto")
        self.file_type_combo.addItems(list(tttrlib.TTTR.get_supported_container_names()))
        
        # Initialize micro binning combo
        self.micro_binning_combo.addItems(["1", "2", "4", "8", "16", "32", "64", "128"])
        
        # Set initial values for TTTR reading
        self.macro_time_le.setText(str(_initial_tttr_reading["macro_time_resolution"]))
        self.micro_time_le.setText(str(_initial_tttr_reading["micro_time_resolution"]))
        self.micro_binning_combo.setCurrentText(str(_initial_tttr_reading["micro_time_binning"]))
        
        # Set table headers
        self.windows_form.setHorizontalHeaderLabels(["Window Name", "Start", "End"])
        try:
            self.detectors_form.setColumnCount(7)
        except Exception:
            pass
        self.detectors_form.setHorizontalHeaderLabels(["Detector Name", "Channels", "Micro Time Ranges", "G-Factor", "l1", "l2", "G-Factor Channels"])

        # Improve table space usage: adaptive column widths and stretch
        try:
            from qtpy.QtWidgets import QHeaderView, QSizePolicy
            # Make tables expand within layouts
            for table in (self.windows_form, self.detectors_form):
                table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                table.setWordWrap(False)
                table.horizontalHeader().setHighlightSections(False)
                table.horizontalHeader().setStretchLastSection(False)
                table.horizontalHeader().setMinimumSectionSize(60)
                table.verticalHeader().setVisible(False)
                table.setAlternatingRowColors(False)

            # Windows table: Name stretches, Start/End resize to contents but user-resizable
            wh = self.windows_form.horizontalHeader()
            wh.setSectionResizeMode(0, QHeaderView.Stretch)
            wh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            wh.setSectionResizeMode(2, QHeaderView.ResizeToContents)

            # Detectors table: allocate space sensibly across 7 columns
            dh = self.detectors_form.horizontalHeader()
            # Name, Channels, Micro Time Ranges should stretch
            dh.setSectionResizeMode(0, QHeaderView.Stretch)  # Detector Name
            dh.setSectionResizeMode(1, QHeaderView.Stretch)  # Channels
            dh.setSectionResizeMode(2, QHeaderView.Stretch)  # Micro Time Ranges
            # Numeric fields: size to contents but allow user to expand
            for col in (3, 4, 5):  # G-Factor, l1, l2
                dh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
            # G-Factor Channels: stretch (often a short range but can use leftover)
            dh.setSectionResizeMode(6, QHeaderView.Stretch)

            # Enable interactive resizing by the user
            for col in range(0, 7):
                # Start with Interactive so user can drag; the above modes define initial behavior
                dh.setSectionResizeMode(col, dh.sectionResizeMode(col))
            dh.setCascadingSectionResizes(True)
        except Exception:
            pass

        # Load available setups
        self._load_available_setups()

        # Load initial or file
        if json_file:
            with open(json_file, "r") as f:
                data = json.load(f)
            self._load_data(data)
        else:
            # If no file specified, try to load the last used setup or use defaults
            setups = load_detector_setups(self.current_setups_file)
            if setups.get("last_used") and setups["last_used"] in setups["setups"]:
                self.current_setup_name = setups["last_used"]
                self.setup_combo.setCurrentText(self.current_setup_name)
                data = setups["setups"][self.current_setup_name]
            else:
                data = {
                    "windows": _initial_windows, 
                    "detectors": _initial_detectors,
                    "tttr_reading": _initial_tttr_reading
                }
            self._load_data(data)
            
        # Initialize the effective micro time resolution
        self._update_effective_resolution()

        # Initialize finish state: disable Finish until user explicitly saves
        self._allow_finish = allow_finish

    def isComplete(self):
        """Only allow finishing the wizard after the user saved settings."""
        # QWizard queries this to enable/disable the Finish button
        return bool(getattr(self, "_allow_finish", False))

    # The _with_label method is no longer needed as the UI file already includes labels for widgets
    # This method is kept for backward compatibility but is not used in the new implementation
    def _with_label(self, text, widget):
        """Helper to wrap a widget with a label above (legacy method, not used with UI file)."""
        v = QVBoxLayout()
        v.addWidget(QLabel(text))
        v.addWidget(widget)
        w = QWidget()
        w.setLayout(v)
        return w
        
    def _hide_layout_widgets(self, layout, visible):
        """Helper method to hide/show all widgets in a layout.
        
        This method is used instead of trying to access widget containers directly,
        which can cause AttributeError if the widget names don't match between the
        code and the UI file. It iterates through all widgets in the given layout
        and sets their visibility based on the provided flag.
        
        Args:
            layout: The layout containing widgets to hide/show
            visible: Boolean indicating whether widgets should be visible
        """
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)

    def _toggle_help(self, on):
        self.help_text.setVisible(on)
        self.help_button.setText("Hide Help" if on else "Show Help")

    def _on_load_setups_file(self):
        """Open a file dialog to select a different detector setups file."""
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Detector Setups File", 
            "", 
            "JSON Files (*.json)"
        )
        if not path:
            return

        try:
            # Load setups from the selected file
            setups = load_detector_setups(path)

            # Update the UI to display the new file path
            self.setups_file_le.setText(path)

            # Store the current file path as an instance variable
            self.current_setups_file = path

            # Update the setup combo box with the setups from the new file
            self.setup_combo.blockSignals(True)
            self.setup_combo.clear()

            # Add a blank item for "custom" setup
            self.setup_combo.addItem("")

            # Add setups from the loaded file
            for setup_name in setups.get("setups", {}).keys():
                self.setup_combo.addItem(setup_name)

            # If there's a last used setup, select it
            if setups.get("last_used") and setups["last_used"] in setups["setups"]:
                self.current_setup_name = setups["last_used"]
                index = self.setup_combo.findText(self.current_setup_name)
                if index >= 0:
                    self.setup_combo.setCurrentIndex(index)

                    # Load the selected setup
                    data = setups["setups"][self.current_setup_name]
                    self._load_data(data)

            self.setup_combo.blockSignals(False)

            QMessageBox.information(
                self, 
                "Success", 
                f"Loaded detector setups from {path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to load detector setups file: {e}"
            )


    def _update_effective_resolution(self):
        """
        Calculate and update the effective micro time resolution based on the current
        micro time resolution and binning factor.
        """
        try:
            micro_time_res = float(self.micro_time_le.text())
            binning = int(self.micro_binning_combo.currentText())
            effective_res = micro_time_res * binning
            self.effective_micro_time_le.setText(f"{effective_res:.6f}")
        except (ValueError, TypeError):
            # Handle case where inputs are not valid numbers
            self.effective_micro_time_le.setText("N/A")

    def _load_data(self, data):
        # block updates/signals
        self.windows_form.setUpdatesEnabled(False)
        self.detectors_form.setUpdatesEnabled(False)
        self.windows_form.blockSignals(True)
        self.detectors_form.blockSignals(True)

        # reset g-factor protection state for fresh rows
        self._g_user_editing.clear()
        self._g_last_valid.clear()

        # clear
        self.windows_form.setRowCount(0)
        self.detectors_form.setRowCount(0)

        # During programmatic population, allow g-factor text changes
        prev_allow = self._allow_g_update
        self._allow_g_update = True
        try:
            # populate windows
            for name, (start, end) in data.get("windows", {}).items():
                self._add_window_row(name, str(start), str(end))

            # populate detectors
            for name, props in data.get("detectors", {}).items():
                chs = ", ".join(map(str, props["chs"]))
                mtr = ", ".join(f"{s}-{e}" for s, e in props["micro_time_ranges"]) 
                g_factor = str(props.get("g_factor", 1.00))
                l1 = str(props.get("l1", 0.00))
                l2 = str(props.get("l2", 0.00))
                # New: g_factor_channels supports [start, end] or "start-end"; anything else -> empty
                gfch = props.get("g_factor_channels")
                if isinstance(gfch, (list, tuple)) and len(gfch) == 2:
                    gf_channels_text = f"{int(gfch[0])}-{int(gfch[1])}"
                elif isinstance(gfch, str):
                    gf_channels_text = gfch
                else:
                    gf_channels_text = ""
                self._add_detector_row(name, chs, mtr, g_factor, l1, l2, gf_channels_text)
        finally:
            self._allow_g_update = prev_allow

        # populate TTTR reading routine settings
        tttr_reading = data.get("tttr_reading", _initial_tttr_reading)
        self.file_type_combo.setCurrentText(tttr_reading.get("file_type", "SPC-130"))
        self.macro_time_le.setText(str(tttr_reading.get("macro_time_resolution", 50.0)))
        self.micro_time_le.setText(str(tttr_reading.get("micro_time_resolution", 50.0)))
        self.micro_binning_combo.setCurrentText(str(tttr_reading.get("micro_time_binning", 1)))

        # Update the effective resolution
        self._update_effective_resolution()

        # re-enable
        self.windows_form.blockSignals(False)
        self.detectors_form.blockSignals(False)
        self.windows_form.setUpdatesEnabled(True)
        self.detectors_form.setUpdatesEnabled(True)
        self.detectorsChanged.emit()

    def _add_window_row(self, name, start, end):
        row = self.windows_form.rowCount()
        self.windows_form.insertRow(row)
        self.windows_form.setItem(row, 0, QTableWidgetItem(name))
        self.windows_form.setCellWidget(row, 1, QLineEdit(start))
        self.windows_form.setCellWidget(row, 2, QLineEdit(end))

    def _add_detector_row(self, name, ch_text, mtr_text, g_factor="1.00", l1="0.00", l2="0.00", gf_channels_text: str = ""):
        row = self.detectors_form.rowCount()
        self.detectors_form.insertRow(row)
        self.detectors_form.setItem(row, 0, QTableWidgetItem(name))
        self.detectors_form.setCellWidget(row, 1, QLineEdit(ch_text))
        self.detectors_form.setCellWidget(row, 2, QLineEdit(mtr_text))
        g_le = QLineEdit(g_factor)
        self.detectors_form.setCellWidget(row, 3, g_le)
        self._wire_g_factor_cell(row, g_le)
        self.detectors_form.setCellWidget(row, 4, QLineEdit(l1))
        self.detectors_form.setCellWidget(row, 5, QLineEdit(l2))
        # New column: G-Factor Channels (selection range in Jordi domain)
        try:
            self.detectors_form.setCellWidget(row, 6, QLineEdit(gf_channels_text))
        except Exception:
            pass

    def _add_window(self):
        name = self.new_window_le.text().strip() or f"PIE-Window {self.windows_form.rowCount()+1}"
        if any(self.windows_form.item(r,0).text()==name for r in range(self.windows_form.rowCount())):
            QMessageBox.warning(self, "Warning", "Window name exists.")
            return
        self._add_window_row(name, "0", "2048")
        self.new_window_le.clear()
        self.detectorsChanged.emit()

    def _add_detector(self):
        name = self.new_detector_le.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Enter a detector name.")
            return
        if any(self.detectors_form.item(r,0).text()==name for r in range(self.detectors_form.rowCount())):
            QMessageBox.warning(self, "Warning", "Detector name exists.")
            return
        self._add_detector_row(name, "0, 1", "0-2048", gf_channels_text="")
        self.new_detector_le.clear()
        self.detectorsChanged.emit()

    def _remove_window(self, item):
        self.windows_form.removeRow(item.row())
        self.detectorsChanged.emit()

    def _remove_detector(self, item):
        self.detectors_form.removeRow(item.row())
        self.detectorsChanged.emit()

    def _edit_json(self):
        data = self.get_settings()
        dlg = JsonEditorDialog(data, self)
        if dlg.exec_():
            edited = dlg.get_edited_data()
            if edited:
                self._load_data(edited)

    def get_settings(self):
        # windows
        wins = {}
        for r in range(self.windows_form.rowCount()):
            name = self.windows_form.item(r,0).text().strip()
            start = int(self.windows_form.cellWidget(r,1).text())
            end   = int(self.windows_form.cellWidget(r,2).text())
            wins[name] = (start, end)

        # detectors
        dets = {}
        for r in range(self.detectors_form.rowCount()):
            name = self.detectors_form.item(r,0).text().strip()
            chs = list(map(int, self.detectors_form.cellWidget(r,1).text().split(',')))
            mtr = [
                tuple(map(int, seg.split('-')))
                for seg in self.detectors_form.cellWidget(r,2).text().split(',')
            ]
            
            # Get the G-factor cell widget and its text
            g_factor_widget = self.detectors_form.cellWidget(r,3)
            g_factor_text = g_factor_widget.text() if g_factor_widget else "1.00"
            
            # Convert to float with fallback to default value
            try:
                g_factor = float(g_factor_text)
            except ValueError:
                g_factor = 1.00
                
            l1 = float(self.detectors_form.cellWidget(r,4).text())
            l2 = float(self.detectors_form.cellWidget(r,5).text())

            # Optional: G-Factor Channels from column 6 as "start-end"
            gf_channels = None
            try:
                gf_widget = self.detectors_form.cellWidget(r,6)
                if gf_widget:
                    txt = gf_widget.text().strip()
                    if txt:
                        parts = txt.replace(' ', '').split('-')
                        if len(parts) == 2:
                            gf_start = int(parts[0])
                            gf_end = int(parts[1])
                            gf_channels = [gf_start, gf_end]
            except Exception:
                gf_channels = None

            det_entry = {
                "chs": chs, 
                "micro_time_ranges": mtr,
                "g_factor": g_factor,
                "l1": l1,
                "l2": l2
            }
            if gf_channels is not None:
                det_entry["g_factor_channels"] = gf_channels
            dets[name] = det_entry

        # TTTR reading routine
        tttr_reading = {
            "file_type": self.file_type_combo.currentText(),
            "macro_time_resolution": float(self.macro_time_le.text()),
            "micro_time_resolution": float(self.micro_time_le.text()),
            "micro_time_binning": int(self.micro_binning_combo.currentText()),
            "effective_micro_time_resolution": self.effective_micro_time_resolution,
            "excitation_period": self.excitation_period
        }

        # Return the result
        return {"windows": wins, "detectors": dets, "tttr_reading": tttr_reading}

    def channels(self):
        chs = {}
        settings = self.get_settings()
        for wname, wrange in settings["windows"].items():
            for dname, dinfo in settings["detectors"].items():
                cname = f"{wname}_{dname}"
                chs[cname] = []
                for mtr in dinfo["micro_time_ranges"]:
                    chs[cname].append({
                        "window_range": wrange,
                        "detector_chs": dinfo["chs"],
                        "micro_time_range": mtr
                    })
        return chs

    def _on_save(self):
        data = self.get_settings()
        path, _ = QFileDialog.getSaveFileName(self, "Save Settings", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)

            # If we have a current setup, update it as well
            if self.current_setup_name:
                setups = load_detector_setups(self.current_setups_file)
                setups.setdefault("setups", {})
                
                # If the setup already exists, preserve any additional fields
                if self.current_setup_name in setups["setups"]:
                    existing_data = setups["setups"][self.current_setup_name]
                    # Update fields while preserving unknown nested data (e.g., per-detector mle_settings)
                    for key in data:
                        if key == 'detectors':
                            existing_data.setdefault('detectors', {})
                            # Merge per-detector entries
                            for det_name, det_info in data['detectors'].items():
                                if det_name in existing_data['detectors'] and isinstance(existing_data['detectors'][det_name], dict):
                                    # Update known fields only, preserve anything else
                                    existing_data['detectors'][det_name].update(det_info)
                                else:
                                    existing_data['detectors'][det_name] = det_info
                            # Keep detectors present in existing_data but not in new data as-is
                        else:
                            existing_data[key] = data[key]
                    # Use the updated existing data
                    setups["setups"][self.current_setup_name] = existing_data
                else:
                    # New setup, just use the data as is
                    setups["setups"][self.current_setup_name] = data
                    
                save_detector_setups(setups, self.current_setups_file)

            QMessageBox.information(self, "Success", f"Settings saved to {path}")

            # Mark page as complete and notify wizard so Finish becomes enabled
            self._allow_finish = True
            try:
                self.completeChanged.emit()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")


    @property
    def detectors(self):
        """
        Legacy accessor for external code:
        returns the same dict you’re saving as JSON.
        """
        return self.get_settings()['detectors']
        
    @detectors.setter
    def detectors(self, new_detectors):
        """
        Setter for detectors property. Updates the detectors in the UI.
        
        Args:
            new_detectors (dict): Dictionary of detector configurations
        """
        # Get current settings
        current_settings = self.get_settings()
        
        # Update detectors in settings
        current_settings['detectors'] = new_detectors
        
        # Load updated settings into UI
        self._load_data(current_settings)

    @property
    def windows(self):
        """
        Legacy accessor for external code: returns the same dict
        you're saving as JSON under "windows".
        """
        return self.get_settings()['windows']
        
    @windows.setter
    def windows(self, new_windows):
        """
        Setter for windows property. Updates the windows in the UI.
        
        Args:
            new_windows (dict): Dictionary of window name -> (start, end) tuples
        """
        # Get current settings
        current_settings = self.get_settings()
        
        # Update windows in settings
        current_settings['windows'] = new_windows
        
        # Load updated settings into UI
        self._load_data(current_settings)

    @property
    def filetype(self) -> str | None:
        """
        Returns the selected file type, handling the "Auto" option by trying to infer
        the file type from a file if available.

        Returns:
            str | None: The file type name, or None if "Auto" is selected and no file is available
                        to infer the type from.
        """
        txt = self.file_type_combo.currentText()
        if txt == 'Auto':
            # In this context, we don't have a specific file to infer from
            # External code should handle this by using tttrlib's auto-detection
            return None
        return txt

    @property
    def effective_micro_time_resolution(self):
        """
        Calculate and return the effective micro time resolution based on the current
        micro time resolution and binning factor.

        Returns:
            float: The effective micro time resolution in picoseconds.
        """
        try:
            micro_time_res = float(self.micro_time_le.text())
            binning = int(self.micro_binning_combo.currentText())
            return micro_time_res * binning
        except (ValueError, TypeError):
            # Return default value if inputs are not valid numbers
            return 50.0 * int(self.micro_binning_combo.currentText())

    @property
    def tttr_reading(self):
        """
        Accessor for external code: returns the TTTR reading routine settings
        as a dict with file_type, macro_time_resolution, micro_time_resolution,
        and micro_time_binning.
        """
        return self.get_settings()['tttr_reading']
        
    @property
    def excitation_period(self):
        """
        Returns the excitation period in nanoseconds.
        
        Returns:
            float: The excitation period in nanoseconds.
        """
        return float(self.macro_time_le.text()) #self.excitation_period_spin.value()
        
    @property
    def selected_detector(self):
        """
        Get the currently selected detector information.
        
        Returns:
            dict: A dictionary containing information about the selected detector,
                  or None if no detector is selected.
        """
        return self._selected_detector_info
        
    @selected_detector.setter
    def selected_detector(self, info):
        """
        Set the currently selected detector information.
        
        Args:
            info (dict): A dictionary containing information about the selected detector.
        """
        self._selected_detector_info = info

    def _load_available_setups(self):
        """Load available setups into the combobox."""
        self.setup_combo.blockSignals(True)
        self.setup_combo.clear()

        # Add a blank item for "custom" setup
        self.setup_combo.addItem("")

        # Load setups from the current setups file
        setups = load_detector_setups(self.current_setups_file)
        for setup_name in setups.get("setups", {}).keys():
            self.setup_combo.addItem(setup_name)

        # If we have a current setup, select it
        if self.current_setup_name:
            index = self.setup_combo.findText(self.current_setup_name)
            if index >= 0:
                self.setup_combo.setCurrentIndex(index)

        self.setup_combo.blockSignals(False)

    def _on_setup_changed(self, index):
        """Handle setup selection changes."""
        if index <= 0:  # Empty or custom setup
            self.current_setup_name = None
            return

        setup_name = self.setup_combo.currentText()
        if not setup_name:
            return

        # Load the selected setup from the current setups file
        setups = load_detector_setups(self.current_setups_file)
        if setup_name in setups.get("setups", {}):
            self.current_setup_name = setup_name
            data = setups["setups"][setup_name]
            self._load_data(data)

            # Update last used setup
            setups["last_used"] = setup_name
            save_detector_setups(setups, self.current_setups_file)

    def _on_save_setup(self):
        """Save the current settings as a setup."""
        # Get current settings
        data = self.get_settings()

        # Ask for a setup name
        setup_name, ok = QInputDialog.getText(
            self, "Save Setup", "Enter a name for this setup:",
            text=self.current_setup_name or ""
        )

        if not ok or not setup_name:
            return

        # Save to the current setups file
        setups = load_detector_setups(self.current_setups_file)
        setups.setdefault("setups", {})
        
        # If the setup already exists, preserve any additional fields that aren't in the current settings
        if setup_name in setups["setups"]:
            existing_data = setups["setups"][setup_name]
            # Update fields while preserving unknown nested data (e.g., per-detector mle_settings)
            for key in data:
                if key == 'detectors':
                    existing_data.setdefault('detectors', {})
                    # Merge per-detector entries
                    for det_name, det_info in data['detectors'].items():
                        if det_name in existing_data['detectors'] and isinstance(existing_data['detectors'][det_name], dict):
                            # Update known fields only, preserve anything else
                            existing_data['detectors'][det_name].update(det_info)
                        else:
                            existing_data['detectors'][det_name] = det_info
                    # Keep detectors present in existing_data but not in new data as-is
                else:
                    existing_data[key] = data[key]
            # Use the updated existing data
            setups["setups"][setup_name] = existing_data
        else:
            # New setup, just use the data as is
            setups["setups"][setup_name] = data
            
        setups["last_used"] = setup_name

        if save_detector_setups(setups, self.current_setups_file):
            self.current_setup_name = setup_name
            QMessageBox.information(self, "Success", f"Setup '{setup_name}' saved successfully.")

            # Refresh the combobox and select the new setup
            self._load_available_setups()
            index = self.setup_combo.findText(setup_name)
            if index >= 0:
                self.setup_combo.setCurrentIndex(index)
        else:
            QMessageBox.critical(self, "Error", f"Failed to save setup '{setup_name}'.")

    def _on_delete_setup(self):
        """Delete the current setup."""
        setup_name = self.setup_combo.currentText()
        if not setup_name:
            QMessageBox.warning(self, "Warning", "No setup selected.")
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete the setup '{setup_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Delete from the current setups file
        setups = load_detector_setups(self.current_setups_file)
        if setup_name in setups.get("setups", {}):
            del setups["setups"][setup_name]
            if setups.get("last_used") == setup_name:
                setups["last_used"] = ""

            if save_detector_setups(setups, self.current_setups_file, replace=True):
                QMessageBox.information(self, "Success", f"Setup '{setup_name}' deleted successfully.")

                # Refresh the combobox
                self.current_setup_name = None
                self._load_available_setups()
            else:
                QMessageBox.critical(self, "Error", f"Failed to delete setup '{setup_name}'.")

    def _on_rename_setup(self):
        """Rename the current setup."""
        old_name = self.setup_combo.currentText()
        if not old_name:
            QMessageBox.warning(self, "Warning", "No setup selected.")
            return

        # Ask for a new setup name
        new_name, ok = QInputDialog.getText(
            self, "Rename Setup", "Enter a new name for this setup:",
            text=old_name
        )

        if not ok or not new_name or new_name == old_name:
            return

        # Check if the new name already exists
        setups = load_detector_setups(self.current_setups_file)
        if new_name in setups.get("setups", {}):
            reply = QMessageBox.question(
                self, "Setup Exists", 
                f"A setup with the name '{new_name}' already exists. Do you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

        # Rename the setup in the current setups file
        if old_name in setups.get("setups", {}):
            # Get the current setup data
            setup_data = setups["setups"][old_name]

            # Remove the old setup and add with the new name
            del setups["setups"][old_name]
            setups["setups"][new_name] = setup_data

            # Update last_used if it was the renamed setup
            if setups.get("last_used") == old_name:
                setups["last_used"] = new_name

            if save_detector_setups(setups, self.current_setups_file, replace=True):
                self.current_setup_name = new_name
                QMessageBox.information(self, "Success", f"Setup renamed from '{old_name}' to '{new_name}' successfully.")

                # Refresh the combobox and select the renamed setup
                self._load_available_setups()
                index = self.setup_combo.findText(new_name)
                if index >= 0:
                    self.setup_combo.setCurrentIndex(index)
            else:
                QMessageBox.critical(self, "Error", f"Failed to rename setup from '{old_name}' to '{new_name}'.")

    def _read_from_tttr_file(self):
        """
        Open a TTTR or SPC file and read its settings.
        
        Behavior depends on file type:
        - .set files: Read only microtime calibration
        - .spc files: Read only macrotime calibration
        - Other TTTR files: Read both calibrations
        """
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open TTTR or SPC File", 
            "", 
            "All Files (*);;TTTR Files (*.ptu *.ht3 *.pt3);;SPC Files (*.spc *.set)"
        )
        if not path:
            return

        try:
            # Handle different file types
            if path.lower().endswith('.set'):
                # For .set files: Read only microtime calibration
                reader = BeckerHicklSetReader(path)
                
                # Get only the microtime information
                micro_time_res = reader.micro_time_resolution
                
                # Update only the microtime in the UI
                if micro_time_res is not None:
                    self.micro_time_le.setText(str(micro_time_res))  # Convert from ns to ps
                
                # Update the effective micro time resolution
                self._update_effective_resolution()
                
                # Show a success message
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Successfully read microtime calibration from SET file: {path}"
                )
            elif path.lower().endswith('.spc'):
                # For .spc files: Read only macrotime calibration
                tttr = tttrlib.TTTR(path)
                
                # Get the header information
                header = tttr.get_header()
                
                # Update only the macrotime in the UI
                self.macro_time_le.setText(str(header.macro_time_resolution * 1e9))  # Convert to ns
                
                # Update the effective micro time resolution
                self._update_effective_resolution()
                
                # Show a success message
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Successfully read macrotime calibration from SPC file: {path}"
                )
            else:
                # For other TTTR files: Read both calibrations
                tttr = tttrlib.TTTR(path)
                
                # Get the header information
                header = tttr.get_header()
                
                # Update both calibrations in the UI
                self.macro_time_le.setText(str(header.macro_time_resolution * 1e9))  # Convert to ns
                self.micro_time_le.setText(str(header.micro_time_resolution * 1e9))  # Convert to ns
                
                # Update the effective micro time resolution
                self._update_effective_resolution()
                
                # Show a success message
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Successfully read calibrations from TTTR file: {path}"
                )

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to read file: {e}"
            )

    def _on_calc_g_factor(self):
        """
        Handle the click of the Calculate G-Factor button.
        
        This method:
        1. Opens a file dialog for the user to select a TTTR file
        2. Identifies parallel and perpendicular channels from the detector settings
        3. Creates a Jordi file from the TTTR file using these channels
        4. Opens the g-factor calculator plugin with the Jordi file
        5. Updates the G-Factor value in the detectors table when the plugin closes
        """
        # Get the currently selected detector settings
        settings = self.get_settings()
        detectors = settings["detectors"]
        
        # We require at least two routing channels within the selected detector row.
        # Validation for channel count happens after a row is selected and channels are parsed.
        
        # Open a file dialog to select a TTTR file
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open TTTR File for G-Factor Calculation", 
            "", 
            "All Files (*)"
        )
        if not path:
            return
            
        try:
            # Create a TTTR object
            tttr = tttrlib.TTTR(path)
            
            # Get the micro time binning from the UI
            micro_time_binning = int(self.micro_binning_combo.currentText())
            
            # Get the currently selected row in detectors_form
            selected_rows = self.detectors_form.selectedIndexes()
            if not selected_rows:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please select a detector row first."
                )
                return
                
            # Get the row of the first selected cell
            selected_row = selected_rows[0].row()
            
            # Get the detector name from the selected row
            selected_detector = self.detectors_form.item(selected_row, 0).text().strip()
            
            # Get the routing channels from the selected row
            channels_text = self.detectors_form.cellWidget(selected_row, 1).text()
            all_channels = list(map(int, channels_text.split(',')))
            
            # Split channels into parallel and perpendicular (alternating pattern)
            parallel_channels = all_channels[::2]  # Even indices (0, 2, 4, ...)
            perpendicular_channels = all_channels[1::2]  # Odd indices (1, 3, 5, ...)
            
            # Validate that both parallel and perpendicular lists are non-empty
            if len(all_channels) < 2 or len(parallel_channels) == 0 or len(perpendicular_channels) == 0:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Selected detector must contain at least two routing channels (parallel and perpendicular) to calculate G-Factor."
                )
                return
            
            # Store the selected detector information for later use
            self.selected_detector = {
                'row': selected_row,
                'name': selected_detector,
                'parallel_channels': parallel_channels,
                'perpendicular_channels': perpendicular_channels
            }
            
            # Extract microtime histograms for parallel and perpendicular channels
            parallel_hist, _ = tttr.get_microtime_histogram(micro_time_binning, parallel_channels)
            perpendicular_hist, _ = tttr.get_microtime_histogram(micro_time_binning, perpendicular_channels)
            
            # Find non-zero bins in both histograms
            parallel_nonzero = np.where(parallel_hist > 0)[0]
            perpendicular_nonzero = np.where(perpendicular_hist > 0)[0]
            
            # Find the common range to ensure both histograms are aligned
            if len(parallel_nonzero) > 0 and len(perpendicular_nonzero) > 0:
                start_idx = min(parallel_nonzero[0], perpendicular_nonzero[0])
                end_idx = max(parallel_nonzero[-1], perpendicular_nonzero[-1]) + 1
                
                # Trim both histograms to the same range
                parallel_hist_trimmed = parallel_hist[start_idx:end_idx]
                perpendicular_hist_trimmed = perpendicular_hist[start_idx:end_idx]
            else:
                # If one or both histograms have no non-zero bins, use the original histograms
                parallel_hist_trimmed = parallel_hist
                perpendicular_hist_trimmed = perpendicular_hist
            
            # Create a temporary file for the Jordi data
            fd, jordi_file = tempfile.mkstemp(suffix='.dat')
            
            # Concatenate the trimmed histograms and save to the Jordi file
            jordi_data = np.concatenate([parallel_hist_trimmed, perpendicular_hist_trimmed])
            write_jordi(jordi_data, jordi_file)
            
            # Create and show the g-factor calculator plugin
            g_factor_calculator = JordiGFactorCalculator()
            g_factor_calculator.setWindowModality(Qt.ApplicationModal)  # Make it modal
            
            # Pass routing channel info and context to the calculator for reference
            try:
                setattr(g_factor_calculator, 'parallel_channels', parallel_channels)
                setattr(g_factor_calculator, 'perpendicular_channels', perpendicular_channels)
                setattr(g_factor_calculator, 'micro_time_binning', micro_time_binning)
                setattr(g_factor_calculator, 'detector_name', selected_detector)
            except Exception:
                pass
            
            # Store the calculator instance and file path for later use
            self.g_factor_calculator = g_factor_calculator
            self.jordi_file = jordi_file
            
            # Connect to the closeEvent to get the g-factor value when the calculator is closed
            original_close_event = g_factor_calculator.closeEvent
            
            def custom_close_event(event):
                # Call the original closeEvent first
                if original_close_event:
                    original_close_event(event)
                
                # Check if g_factor was calculated
                if hasattr(g_factor_calculator, 'g_factor') and g_factor_calculator.g_factor is not None:
                    # Get the selected detector information
                    selected_detector_info = self.selected_detector
                    print(f"Selected detector info: {selected_detector_info}")
                    if selected_detector_info:
                        # Update the G-Factor value in the selected row of the detectors table
                        row = selected_detector_info['row']
                        g_factor_value = f"{g_factor_calculator.g_factor:.3f}"
                        print(f"Updating detectors table row {row} with g-factor value {g_factor_value}")
                        
                        # Get the existing cell widget and update its text
                        existing_cell_widget = self.detectors_form.cellWidget(row, 3)
                        if existing_cell_widget:
                            # Use protected programmatic setter to update value
                            self._set_g_factor_programmatically(row, g_factor_value)
                        else:
                            # If no widget exists yet, create a new one and wire protection
                            new_cell_widget = QLineEdit(g_factor_value)
                            self.detectors_form.setCellWidget(row, 3, new_cell_widget)
                            self._wire_g_factor_cell(row, new_cell_widget)
                        
                        # Also capture the selection range (G-Factor Channels) from the calculator, if available
                        gf_range_text = None
                        try:
                            rng = None
                            if hasattr(g_factor_calculator, 'region') and g_factor_calculator.region is not None:
                                try:
                                    rng = g_factor_calculator.region.getRegion()
                                except Exception:
                                    rng = None
                            if rng is None and hasattr(g_factor_calculator, 'region_bounds'):
                                rng = getattr(g_factor_calculator, 'region_bounds', None)
                            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                                s = int(float(rng[0]))
                                e = int(float(rng[1]))
                                if e < s:
                                    s, e = e, s
                                gf_range_text = f"{s}-{e}"
                                # Update column 6 in the table
                                try:
                                    gf_widget = self.detectors_form.cellWidget(row, 6)
                                    if gf_widget is None:
                                        gf_widget = QLineEdit(gf_range_text)
                                        self.detectors_form.setCellWidget(row, 6, gf_widget)
                                    else:
                                        gf_widget.setText(gf_range_text)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                                                
                        # Save the updated setup automatically
                        if self.current_setup_name:
                            # Get current settings (now includes g_factor and g_factor_channels)
                            data = self.get_settings()
                            
                            # Save to the current setups file
                            setups = load_detector_setups(self.current_setups_file)
                            setups.setdefault("setups", {})
                            
                            # If the setup already exists, preserve any additional fields
                            if self.current_setup_name in setups["setups"]:
                                existing_data = setups["setups"][self.current_setup_name]
                                # Update only the fields we know about, preserving any other fields
                                for key in data:
                                    if key == 'detectors':
                                        existing_data.setdefault('detectors', {})
                                        # Merge per-detector entries
                                        for det_name, det_info in data['detectors'].items():
                                            if det_name in existing_data['detectors'] and isinstance(existing_data['detectors'][det_name], dict):
                                                # Update known fields only, preserve anything else
                                                existing_data['detectors'][det_name].update(det_info)
                                            else:
                                                existing_data['detectors'][det_name] = det_info
                                        # Keep detectors present in existing_data but not in new data as-is
                                    else:
                                        existing_data[key] = data[key]
                                # Use the updated existing data
                                setups["setups"][self.current_setup_name] = existing_data
                            else:
                                # New setup, just use the data as is
                                setups["setups"][self.current_setup_name] = data
                                
                            setups["last_used"] = self.current_setup_name
                            save_detector_setups(setups, self.current_setups_file)
                            
                            # Show a success message with save confirmation
                            msg = (
                                f"G-Factor calculated: {g_factor_calculator.g_factor:.4f}\n"
                                f"Updated G-Factor for detector: {selected_detector_info['name']}\n"
                            )
                            if gf_range_text:
                                msg += f"G-Factor Channels: {gf_range_text}\n"
                            msg += f"Setup '{self.current_setup_name}' saved automatically."
                            QMessageBox.information(self, "Success", msg)
                        else:
                            # Show a success message without save confirmation
                            msg = (
                                f"G-Factor calculated: {g_factor_calculator.g_factor:.4f}\n"
                                f"Updated G-Factor for detector: {selected_detector_info['name']}\n"
                            )
                            if gf_range_text:
                                msg += f"G-Factor Channels: {gf_range_text}\n"
                            msg += "Note: No setup was selected, so changes were not saved automatically."
                            QMessageBox.information(self, "Success", msg)
            
            # Override the closeEvent method
            g_factor_calculator.closeEvent = custom_close_event
            
            # Show the calculator
            g_factor_calculator.show()
            
            # Load the Jordi file using the calculator's load_jordi_file method
            try:
                # Set the effective micro time resolution for proper time axis scaling
                effective_dt = self.effective_micro_time_resolution
                
                # Load the Jordi file
                g_factor_calculator.load_jordi_file(jordi_file)

                # If user specified a G-Factor Channels range in the table, pass it to the calculator
                try:
                    gf_widget = self.detectors_form.cellWidget(selected_row, 6)
                    if gf_widget:
                        txt = gf_widget.text().strip()
                        if txt:
                            parts = txt.replace(' ', '').split('-')
                            if len(parts) == 2:
                                s = int(float(parts[0]))
                                e = int(float(parts[1]))
                                # Ensure order and bounds are sane
                                if e < s:
                                    s, e = e, s
                                # Apply to calculator
                                if hasattr(g_factor_calculator, 'region'):
                                    try:
                                        g_factor_calculator.region.setRegion([s, e])
                                    except Exception:
                                        pass
                                if hasattr(g_factor_calculator, 'region_bounds'):
                                    try:
                                        g_factor_calculator.region_bounds = [s, e]
                                    except Exception:
                                        pass
                                # Recompute with new region
                                try:
                                    g_factor_calculator.calculate_g_factor()
                                except Exception:
                                    pass
                except Exception:
                    pass
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load Jordi file: {str(e)}"
                )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to calculate G-Factor: {str(e)}"
            )
    
    def load_data_into_tables(self, data):
        """
        Legacy alias for external callers.
        """
        # reuse our internal loader
        self._load_data(data)


    # --- G-Factor protection helpers ---
    def _wire_g_factor_cell(self, row, line_edit: QLineEdit):
        """Protect a row's G-Factor QLineEdit so only user edits or internal allowed updates can change it."""
        # Initialize tracking for this row
        self._g_user_editing[row] = False
        self._g_last_valid[row] = line_edit.text()

        def on_text_edited(_):
            # Fired only by user typing
            self._g_user_editing[row] = True

        def on_editing_finished():
            try:
                txt = line_edit.text().strip()
                # Accept empty as default 1.0
                val = float(txt) if txt else 1.0
                # Normalize formatting
                new_txt = f"{val:.3f}"
                # Allow internal write for normalization
                prev = self._allow_g_update
                self._allow_g_update = True
                try:
                    if line_edit.text() != new_txt:
                        line_edit.setText(new_txt)
                finally:
                    self._allow_g_update = prev
                # Commit last valid
                self._g_last_valid[row] = new_txt
            except Exception:
                # Revert to last valid on invalid input
                prev = self._allow_g_update
                self._allow_g_update = True
                try:
                    line_edit.setText(self._g_last_valid.get(row, "1.000"))
                finally:
                    self._allow_g_update = prev
            finally:
                self._g_user_editing[row] = False

        def on_text_changed(_):
            # Reject programmatic changes unless explicitly allowed
            if self._allow_g_update:
                # Keep last_valid in sync during allowed writes
                self._g_last_valid[row] = line_edit.text()
                return
            if self._g_user_editing.get(row, False):
                # User typing: allow
                return
            # Unauthorised programmatic change: revert
            prev = self._allow_g_update
            self._allow_g_update = True
            try:
                line_edit.setText(self._g_last_valid.get(row, line_edit.text()))
            finally:
                self._allow_g_update = prev

        # Connect signals
        try:
            line_edit.textEdited.connect(on_text_edited)
        except Exception:
            pass
        line_edit.editingFinished.connect(on_editing_finished)
        line_edit.textChanged.connect(on_text_changed)

    def _set_g_factor_programmatically(self, row: int, value_text: str):
        """Safely set a row's G-Factor from internal code (calculator/data load)."""
        le = self.detectors_form.cellWidget(row, 3)
        if not isinstance(le, QLineEdit):
            return
        prev = self._allow_g_update
        self._allow_g_update = True
        try:
            le.setText(value_text)
            self._g_last_valid[row] = value_text
        finally:
            self._allow_g_update = prev

class DetectorWizard(QWizard):
    def __init__(self, json_file=None, show_edit_json=True, show_save=True, 
                 show_setups_file=True, show_setup_selection=True, show_help=True,
                 show_tttr_reading=True, show_tables=True, show_add_inputs=True, **kwargs):
        """Initialize the DetectorWizard.

        Args:
            json_file (str, optional): Path to a JSON file to load. Defaults to None.
            show_edit_json (bool, optional): Whether to show the "Edit JSON" button. Defaults to True.
            show_save (bool, optional): Whether to show the "Save" button. Defaults to True.
            show_setups_file (bool, optional): Whether to show the setups file section. Defaults to True.
            show_setup_selection (bool, optional): Whether to show the setup selection section. Defaults to True.
            show_help (bool, optional): Whether to show the help button and text. Defaults to True.
            show_tttr_reading (bool, optional): Whether to show the TTTR reading routine section. Defaults to True.
            show_tables (bool, optional): Whether to show the PIE-Windows and Detectors tables. Defaults to True.
            show_add_inputs (bool, optional): Whether to show the controls for adding windows and detectors. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the DetectorWizardPage.
        """
        super().__init__()
        self.addPage(DetectorWizardPage(
            json_file=json_file,
            show_edit_json=show_edit_json,
            show_save=show_save,
            show_setups_file=show_setups_file,
            show_setup_selection=show_setup_selection,
            show_help=show_help,
            show_tttr_reading=show_tttr_reading,
            show_tables=show_tables,
            show_add_inputs=show_add_inputs,
            **kwargs
        ))
        self.setWindowTitle("Detector Configuration Wizard")


if __name__ == "__main__":
    json_arg = sys.argv[1] if len(sys.argv) > 1 else None
    app = QApplication(sys.argv)
    wiz = DetectorWizard(json_file=json_arg)
    wiz.show()
    sys.exit(app.exec_())
