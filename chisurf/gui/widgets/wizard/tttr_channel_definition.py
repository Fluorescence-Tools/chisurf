import sys
import json
import pathlib
from PyQt5.QtWidgets import (
    QApplication, QWizard, QWizardPage, QVBoxLayout, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QTextEdit, QDialog,
    QMessageBox, QHBoxLayout, QGridLayout, QFileDialog, QToolButton, QWidget,
    QComboBox, QInputDialog
)

from PyQt5.QtCore import pyqtSignal

from chisurf.settings.path_utils import get_path
from chisurf.settings.file_utils import safe_open_file
import tttrlib

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

    Args:
        file_path: Optional custom path to load from. If None, uses DETECTOR_SETUPS_FILE.
    """
    return safe_open_file(
        file_path=file_path or DETECTOR_SETUPS_FILE,
        processor=json.load,
        default_value={"setups": {}}
    )

def save_detector_setups(setups_data, file_path=None):
    """Save detector setups to the central settings file or a custom file.

    Args:
        setups_data: The data to save
        file_path: Optional custom path to save to. If None, uses DETECTOR_SETUPS_FILE.
    """
    try:
        save_path = file_path or DETECTOR_SETUPS_FILE
        with open(save_path, 'w') as f:
            json.dump(setups_data, f, indent=4)
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
    "green":  {"chs": [8, 0, 3], "micro_time_ranges": [(0, 4095)]},
    "red":    {"chs": [9, 1, 2], "micro_time_ranges": [(0, 2048)]},
    "yellow": {"chs": [9, 1, 2], "micro_time_ranges": [(2048, 4095)]},
}

# Initial TTTR reading routine settings
_initial_tttr_reading = {
    "file_type": "SPC-130",
    "macro_time_resolution": 50.0,  # in nanoseconds
    "micro_time_resolution": 50.0,  # in picoseconds
    "micro_time_binning": 1
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
    detectorsChanged = pyqtSignal()

    def __init__(self, json_file=None, *args, show_edit_json=False, show_save=False,
                 show_setups_file=True, show_setup_selection=True, show_help=True,
                 show_tttr_reading=True, show_tables=True, show_add_inputs=True, **kwargs):
        """Initialize the DetectorWizardPage.

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
            *args: Additional positional arguments to pass to the parent class.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        super().__init__(*args, **kwargs)
        self.setTitle("Detectors and PIE-window definition")
        self.current_setup_name = None
        self.current_setups_file = str(DETECTOR_SETUPS_FILE)
        self.show_edit_json = show_edit_json
        self.show_save = show_save
        self.show_setups_file = show_setups_file
        self.show_setup_selection = show_setup_selection
        self.show_help = show_help
        self.show_tttr_reading = show_tttr_reading
        self.show_tables = show_tables
        self.show_add_inputs = show_add_inputs

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        # --- Setups File Path ---
        self.setups_file_widget = QWidget(self)
        setups_file_layout = QHBoxLayout(self.setups_file_widget)
        setups_file_layout.setContentsMargins(0, 0, 0, 0)
        setups_file_layout.addWidget(QLabel("Setups File:"))

        self.setups_file_le = QLineEdit(self)
        self.setups_file_le.setReadOnly(True)
        self.setups_file_le.setText(str(DETECTOR_SETUPS_FILE))
        setups_file_layout.addWidget(self.setups_file_le)

        self.load_setups_file_button = QToolButton(self)
        self.load_setups_file_button.setText("…")
        self.load_setups_file_button.setToolTip("Load a different detector setups file")
        self.load_setups_file_button.clicked.connect(self._on_load_setups_file)
        setups_file_layout.addWidget(self.load_setups_file_button)

        main_layout.addWidget(self.setups_file_widget)
        self.setups_file_widget.setVisible(self.show_setups_file)

        # --- Setup selection ---
        self.setup_selection_widget = QWidget(self)
        setup_layout = QHBoxLayout(self.setup_selection_widget)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.addWidget(QLabel("Setup:"))

        self.setup_combo = QComboBox(self)
        self.setup_combo.currentIndexChanged.connect(self._on_setup_changed)
        setup_layout.addWidget(self.setup_combo)

        self.save_setup_button = QToolButton(self)
        self.save_setup_button.setText("Save")
        self.save_setup_button.clicked.connect(self._on_save_setup)
        setup_layout.addWidget(self.save_setup_button)

        self.rename_setup_button = QToolButton(self)
        self.rename_setup_button.setText("Rename")
        self.rename_setup_button.clicked.connect(self._on_rename_setup)
        setup_layout.addWidget(self.rename_setup_button)

        self.delete_setup_button = QToolButton(self)
        self.delete_setup_button.setText("Delete")
        self.delete_setup_button.clicked.connect(self._on_delete_setup)
        setup_layout.addWidget(self.delete_setup_button)

        main_layout.addWidget(self.setup_selection_widget)
        self.setup_selection_widget.setVisible(self.show_setup_selection)

        # --- Help button ---
        self.help_widget = QWidget(self)
        help_layout = QHBoxLayout(self.help_widget)
        help_layout.setContentsMargins(0, 0, 0, 0)

        self.help_button = QToolButton(self)
        self.help_button.setText("Show Help")
        self.help_button.setCheckable(True)
        self.help_button.toggled.connect(self._toggle_help)
        help_layout.addWidget(self.help_button)

        main_layout.addWidget(self.help_widget)
        self.help_widget.setVisible(self.show_help)

        self.help_text = QTextEdit(help_text, self)
        self.help_text.setReadOnly(True)
        self.help_text.setVisible(False)
        main_layout.addWidget(self.help_text)
        # The help_text visibility is controlled by _toggle_help, but we'll hide it if show_help is False
        if not self.show_help:
            self.help_text.setVisible(False)

        # --- TTTR Reading Routine ---
        self.tttr_reading_widget = QWidget(self)
        tttr_layout = QGridLayout(self.tttr_reading_widget)
        tttr_layout.setContentsMargins(0, 0, 0, 0)
        tttr_layout.addWidget(QLabel("TTTR Reading Routine:"), 0, 0, 1, 4)

        # Row 1: label, filetype, read info button
        tttr_layout.addWidget(QLabel("File Type:"), 1, 0)
        self.file_type_combo = QComboBox(self)
        # Add Auto and all supported TTTR file types
        self.file_type_combo.addItem("Auto")
        self.file_type_combo.addItems(list(tttrlib.TTTR.get_supported_container_names()))
        tttr_layout.addWidget(self.file_type_combo, 1, 1)

        # Read from TTTR file button
        self.read_tttr_button = QPushButton("Info form File", self)
        self.read_tttr_button.clicked.connect(self._read_from_tttr_file)
        tttr_layout.addWidget(self.read_tttr_button, 1, 2, 1, 2)

        # Row 2: label, macro time res, label, micro time res
        tttr_layout.addWidget(QLabel("Macro Time Res. (ns):"), 2, 0)
        self.macro_time_le = QLineEdit(self)
        self.macro_time_le.setText(str(_initial_tttr_reading["macro_time_resolution"]))
        tttr_layout.addWidget(self.macro_time_le, 2, 1)

        tttr_layout.addWidget(QLabel("Micro Time Res. (ps):"), 2, 2)
        self.micro_time_le = QLineEdit(self)
        self.micro_time_le.setText(str(_initial_tttr_reading["micro_time_resolution"]))
        self.micro_time_le.textChanged.connect(self._update_effective_resolution)
        tttr_layout.addWidget(self.micro_time_le, 2, 3)

        # Row 3: label, micro bin, label, eff micro time
        tttr_layout.addWidget(QLabel("Micro Time Binning:"), 3, 0)
        self.micro_binning_combo = QComboBox(self)
        self.micro_binning_combo.addItems(["1", "2", "4", "8", "16"])
        self.micro_binning_combo.setCurrentText(str(_initial_tttr_reading["micro_time_binning"]))
        self.micro_binning_combo.currentTextChanged.connect(self._update_effective_resolution)
        tttr_layout.addWidget(self.micro_binning_combo, 3, 1)

        tttr_layout.addWidget(QLabel("Eff. Micro Time (ps):"), 3, 2)
        self.effective_micro_time_le = QLineEdit(self)
        self.effective_micro_time_le.setReadOnly(True)
        tttr_layout.addWidget(self.effective_micro_time_le, 3, 3)

        main_layout.addWidget(self.tttr_reading_widget)
        self.tttr_reading_widget.setVisible(self.show_tttr_reading)

        # --- Tables ---
        self.tables_widget = QWidget(self)
        tables_layout = QHBoxLayout(self.tables_widget)
        tables_layout.setContentsMargins(0, 0, 0, 0)
        tables_layout.setSpacing(0)
        main_layout.addWidget(self.tables_widget)
        self.tables_widget.setVisible(self.show_tables)

        # Windows table
        self.windows_form = QTableWidget(0, 3, self)
        self.windows_form.setHorizontalHeaderLabels(["Window Name", "Start", "End"])
        self.windows_form.itemDoubleClicked.connect(self._remove_window)
        tables_layout.addWidget(self._with_label("PIE-Windows", self.windows_form))

        # Detectors table
        self.detectors_form = QTableWidget(0, 3, self)
        self.detectors_form.setHorizontalHeaderLabels(["Detector Name", "Channels", "Micro Time Ranges"])
        self.detectors_form.itemDoubleClicked.connect(self._remove_detector)
        tables_layout.addWidget(self._with_label("Detectors", self.detectors_form))

        # --- Add inputs + Save/Edit JSON ---
        self.add_inputs_widget = QWidget(self)
        controls = QGridLayout(self.add_inputs_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)

        # Add window
        self.new_window_le = QLineEdit(self)
        self.new_window_le.setPlaceholderText("New PIE-Window name")
        controls.addWidget(self.new_window_le, 0, 0)
        btn = QPushButton("Add", self)
        btn.clicked.connect(self._add_window)
        controls.addWidget(btn, 0, 1)

        # Add detector
        self.new_detector_le = QLineEdit(self)
        self.new_detector_le.setPlaceholderText("New Detector name")
        controls.addWidget(self.new_detector_le, 1, 0)
        btn = QPushButton("Add", self)
        btn.clicked.connect(self._add_detector)
        controls.addWidget(btn, 1, 1)

        # JSON editor
        self.edit_json_button = QPushButton("Edit JSON", self)
        self.edit_json_button.clicked.connect(self._edit_json)
        self.edit_json_button.setVisible(self.show_edit_json)
        controls.addWidget(self.edit_json_button, 0, 2)

        # Save button
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self._on_save)
        self.save_button.setVisible(self.show_save)
        controls.addWidget(self.save_button, 1, 2)

        main_layout.addWidget(self.add_inputs_widget)
        self.add_inputs_widget.setVisible(self.show_add_inputs)

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

    def _with_label(self, text, widget):
        """Helper to wrap a widget with a label above."""
        v = QVBoxLayout()
        v.addWidget(QLabel(text))
        v.addWidget(widget)
        container = QVBoxLayout()  # dummy widget
        w = QWidget()
        w.setLayout(v)
        return w

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
            self.effective_micro_time_le.setText(f"{effective_res:.2f}")
        except (ValueError, TypeError):
            # Handle case where inputs are not valid numbers
            self.effective_micro_time_le.setText("N/A")

    def _load_data(self, data):
        # block updates/signals
        self.windows_form.setUpdatesEnabled(False)
        self.detectors_form.setUpdatesEnabled(False)
        self.windows_form.blockSignals(True)
        self.detectors_form.blockSignals(True)

        # clear
        self.windows_form.setRowCount(0)
        self.detectors_form.setRowCount(0)

        # populate windows
        for name, (start, end) in data.get("windows", {}).items():
            self._add_window_row(name, str(start), str(end))

        # populate detectors
        for name, props in data.get("detectors", {}).items():
            chs = ", ".join(map(str, props["chs"]))
            mtr = ", ".join(f"{s}-{e}" for s, e in props["micro_time_ranges"])
            self._add_detector_row(name, chs, mtr)

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

    def _add_detector_row(self, name, ch_text, mtr_text):
        row = self.detectors_form.rowCount()
        self.detectors_form.insertRow(row)
        self.detectors_form.setItem(row, 0, QTableWidgetItem(name))
        self.detectors_form.setCellWidget(row, 1, QLineEdit(ch_text))
        self.detectors_form.setCellWidget(row, 2, QLineEdit(mtr_text))

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
        self._add_detector_row(name, "0, 1", "0-2048")
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
            dets[name] = {"chs": chs, "micro_time_ranges": mtr}

        # TTTR reading routine
        tttr_reading = {
            "file_type": self.file_type_combo.currentText(),
            "macro_time_resolution": float(self.macro_time_le.text()),
            "micro_time_resolution": float(self.micro_time_le.text()),
            "micro_time_binning": int(self.micro_binning_combo.currentText()),
            "effective_micro_time_resolution": self.effective_micro_time_resolution
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
                setups["setups"][self.current_setup_name] = data
                save_detector_setups(setups, self.current_setups_file)

            QMessageBox.information(self, "Success", f"Settings saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")


    @property
    def detectors(self):
        """
        Legacy accessor for external code:
        returns the same dict you’re saving as JSON.
        """
        return self.get_settings()['detectors']

    @property
    def windows(self):
        """
        Legacy accessor for external code: returns the same dict
        you're saving as JSON under "windows".
        """
        return self.get_settings()['windows']

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

            if save_detector_setups(setups, self.current_setups_file):
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

            if save_detector_setups(setups, self.current_setups_file):
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
        """Open a TTTR file and read its settings."""
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open TTTR File", 
            "", 
            "All Files (*)"
        )
        if not path:
            return

        try:
            # Create a TTTR object
            tttr = tttrlib.TTTR(path)

            # Get the header information
            header = tttr.get_header()

            # Update the UI with the settings from the file
            self.macro_time_le.setText(str(header.macro_time_resolution * 1e9))  # Convert to ns
            self.micro_time_le.setText(str(header.micro_time_resolution * 1e12))  # Convert to ps

            # Update the effective micro time resolution
            self._update_effective_resolution()

            # Show a success message
            QMessageBox.information(
                self, 
                "Success", 
                f"Successfully read settings from {path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to read TTTR file: {e}"
            )

    def load_data_into_tables(self, data):
        """
        Legacy alias for external callers.
        """
        # reuse our internal loader
        self._load_data(data)


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
