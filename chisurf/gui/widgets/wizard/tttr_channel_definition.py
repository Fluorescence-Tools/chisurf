import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QWizard, QWizardPage, QVBoxLayout, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QTextEdit, QDialog,
    QMessageBox, QHBoxLayout, QGridLayout, QFileDialog, QToolButton, QWidget
)

from PyQt5.QtCore import pyqtSignal

help_text = """You can either load an existing detector Pulsed-Interleaved Excitation (PIE) 
window definition by clicking on the '...' button to define channels, or define your own PIE 
and detector settings by editing the tables below. New PIE windows and detector windows can 
be added by clicking the "Add" button next to the detector name field. The "Edit" button 
displays a JSON file representing the data, and the "Save" button allows you to save your 
channel configuration.
"""

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

    def __init__(self, json_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTitle("Detectors and PIE-window definition")

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        # --- File load + Help ---
        file_layout = QHBoxLayout()
        self.file_path_le = QLineEdit(self)
        self.file_path_le.setPlaceholderText("Select JSON file…")
        file_layout.addWidget(self.file_path_le)
        # **alias the old name for compatibility**
        self.file_path_line_edit = self.file_path_le

        self.load_button = QToolButton(self)
        self.load_button.setText("…")
        self.load_button.clicked.connect(self._on_load_file)
        file_layout.addWidget(self.load_button)

        self.help_button = QToolButton(self)
        self.help_button.setText("Show Help")
        self.help_button.setCheckable(True)
        self.help_button.toggled.connect(self._toggle_help)
        file_layout.addWidget(self.help_button)

        main_layout.addLayout(file_layout)

        self.help_text = QTextEdit(help_text, self)
        self.help_text.setReadOnly(True)
        self.help_text.setVisible(False)
        main_layout.addWidget(self.help_text)

        # --- Tables ---
        tables_layout = QHBoxLayout()
        tables_layout.setSpacing(0)
        main_layout.addLayout(tables_layout)

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
        controls = QGridLayout()
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
        btn = QPushButton("Edit JSON", self)
        btn.clicked.connect(self._edit_json)
        controls.addWidget(btn, 0, 2)

        # Save button
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self._on_save)
        controls.addWidget(self.save_button, 1, 2)

        main_layout.addLayout(controls)

        # Load initial or file
        if json_file:
            with open(json_file, "r") as f:
                data = json.load(f)
        else:
            data = {"windows": _initial_windows, "detectors": _initial_detectors}
        self._load_data(data)

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

    def _on_load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open JSON", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._load_data(data)
            self.file_path_le.setText(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load JSON: {e}")

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

        return {"windows": wins, "detectors": dets}

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

    def load_data_into_tables(self, data):
        """
        Legacy alias for external callers.
        """
        # reuse our internal loader
        self._load_data(data)


class DetectorWizard(QWizard):
    def __init__(self, json_file=None):
        super().__init__()
        self.addPage(DetectorWizardPage(json_file=json_file))
        self.setWindowTitle("Detector Configuration Wizard")


if __name__ == "__main__":
    json_arg = sys.argv[1] if len(sys.argv) > 1 else None
    app = QApplication(sys.argv)
    wiz = DetectorWizard(json_file=json_arg)
    wiz.show()
    sys.exit(app.exec_())
