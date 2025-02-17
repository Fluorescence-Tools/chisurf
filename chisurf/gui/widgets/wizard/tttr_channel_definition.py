import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QWizard, QWizardPage, QVBoxLayout, QLabel, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QTextEdit, QDialog,
    QMessageBox, QHBoxLayout, QGridLayout, QFileDialog, QToolButton
)


help_text = """You can either load an existing detector Pulsed-Interleaved Excitation (PIE) 
window definition by clicking on the '...' button to define channels, or define your own PIE 
and detector settings by editing the tables below. New PIE windows and detector windows can 
be added by clicking the "Add" button next to the detector name field. The "Edit" button 
displays a JSON file representing the data, and the "Save" button allows you to save your 
channel configuration.
"""

class JsonEditorDialog(QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit JSON Settings")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # JSON editor text area
        self.json_editor = QTextEdit(self)
        self.layout.addWidget(self.json_editor)

        # Populate text area with JSON string
        self.json_editor.setText(json.dumps(data, indent=4))

        # Save button
        self.save_button = QPushButton("Save JSON")
        self.save_button.clicked.connect(self.save_json)
        self.layout.addWidget(self.save_button)

    def save_json(self):
        try:
            # Load JSON from text area
            edited_data = json.loads(self.json_editor.toPlainText())
            self.done(1)  # Accept the dialog and set a return code
            self.edited_data = edited_data  # Store the edited data
        except json.JSONDecodeError:
            QMessageBox.critical(self, "Error", "Invalid JSON format.")

    def get_edited_data(self):
        return getattr(self, 'edited_data', None)  # Return edited data if available


class DetectorWizardPage(QWizardPage):

    def __init__(self, json_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setTitle("Detectors and PIE-window definition")

        # Create main vertical layout
        layout = QVBoxLayout()
        layout.setSpacing(0)  # Set spacing for main layout to 0
        layout.setContentsMargins(0, 0, 0, 0)  # Set margins for the layout to 0
        self.setLayout(layout)

        # JSON file loading section (LineEdit + ToolButton)
        file_help_layout = QHBoxLayout()
        self.file_path_line_edit = QLineEdit(self)
        self.file_path_line_edit.setPlaceholderText("Select JSON file...")
        file_help_layout.addWidget(self.file_path_line_edit)

        # Create a QTextEdit for displaying help text (hidden by default)
        self.help_text = QTextEdit()
        self.help_text.setText(help_text)
        self.help_text.setVisible(False)  # Hide by default
        layout.addWidget(self.help_text)

        # Create a QToolButton
        self.help_button = QToolButton()
        self.help_button.setText("Help")
        self.help_button.setCheckable(True)  # Make the tool button checkable
        self.help_button.toggled.connect(self.toggle_help)  # Connect toggle signal to function

        self.file_load_button = QToolButton(self)
        self.file_load_button.setText("...")
        self.file_load_button.clicked.connect(self.load_json_file)
        file_help_layout.addWidget(self.file_load_button)
        file_help_layout.addWidget(self.help_button)

        layout.addLayout(file_help_layout)

        # Create horizontal layout for PIE-Windows and detectors
        h_layout = QHBoxLayout()
        h_layout.setSpacing(0)  # Set spacing for horizontal layout to 0
        layout.addLayout(h_layout)

        # Windows Table
        self.windows_form = QTableWidget()
        self.windows_form.setColumnCount(3)  # Three columns: Name, Start, End
        self.windows_form.setHorizontalHeaderLabels(["Window Name", "Start", "End"])
        self.windows_form.itemDoubleClicked.connect(self.remove_pie_window)  # Connect double-click
        self.windows_dict = {}
        self.windows_widgets = {}
        self.create_windows_fields()

        # Detectors Table
        self.detectors_form = QTableWidget()
        self.detectors_form.setColumnCount(3)  # Three columns: Name, Channels, Micro Time Ranges
        self.detectors_form.setHorizontalHeaderLabels(["Detector Name", "Channels", "Micro Time Ranges"])
        self.detectors_form.itemDoubleClicked.connect(self.remove_detector)  # Connect double-click
        self.detectors_dict = {}
        self.detectors_widgets = {}
        self.create_detectors_fields()

        # Continue with rest of the layout setup...
        # Add PIE-Windows and detectors forms to horizontal layout
        windows_layout = QVBoxLayout()
        windows_layout.setSpacing(0)  # Set spacing for PIE-Windows layout to 0
        windows_layout.addWidget(QLabel("PIE-Windows:"))
        windows_layout.addWidget(self.windows_form)  # Add table
        h_layout.addLayout(windows_layout)

        detectors_layout = QVBoxLayout()
        detectors_layout.setSpacing(0)  # Set spacing for Detectors layout to 0
        detectors_layout.addWidget(QLabel("Detectors:"))
        detectors_layout.addWidget(self.detectors_form)  # Add table
        h_layout.addLayout(detectors_layout)

        # Create a single grid layout for new PIE-Window and Detector inputs
        input_grid = QGridLayout()
        input_grid.setSpacing(0)  # Set spacing for the grid layout to 0

        # New PIE-Window name input and button
        self.new_window_name_input = QLineEdit()
        self.new_window_name_input.setPlaceholderText("Enter new PIE-Window name")
        input_grid.addWidget(self.new_window_name_input, 0, 0)  # Row 0, Column 0

        self.add_window_button = QPushButton("Add")
        self.add_window_button.clicked.connect(self.add_pie_window)
        input_grid.addWidget(self.add_window_button, 0, 1)  # Row 0, Column 1

        # New Detector name input and button
        self.new_detector_name_input = QLineEdit()
        self.new_detector_name_input.setPlaceholderText("Enter new Detector name")
        input_grid.addWidget(self.new_detector_name_input, 1, 0)  # Row 1, Column 0

        self.add_detector_button = QPushButton("Add")
        self.add_detector_button.clicked.connect(self.add_detector)
        input_grid.addWidget(self.add_detector_button, 1, 1)  # Row 1, Column 1

        # JSON button
        self.json_editor_button = QPushButton("Edit")  # Relabeled to JSON
        self.json_editor_button.clicked.connect(self.open_json_editor)
        input_grid.addWidget(self.json_editor_button, 0, 2, 1, 1)  # Row 2, spans 2 columns

        # Add the grid layout to the main layout
        layout.addLayout(input_grid)  # Add grid layout to main layout

        # Button to save inputs
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        input_grid.addWidget(self.save_button, 1, 2, 1, 1)  # Row 2, spans 2 columns

        # Connect signals for editing name changes
        self.windows_form.itemChanged.connect(self.update_window_name)
        self.detectors_form.itemChanged.connect(self.update_detector_name)

        if json_file:
            with open(json_file, 'r') as f:
                data = json.load(f)
            self.load_data_into_tables(data)

    # Define the function to show/hide the help text
    def toggle_help(self, checked):
        if checked:
            self.help_text.setVisible(True)
            self.help_button.setText("Hide Help")
        else:
            self.help_text.setVisible(False)
            self.help_button.setText("Show Help")

    def dragEnterEvent(self, event):
        """Enable drag-enter event to allow file drop."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Handle the file drop event."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.endswith('.json'):
                self.filepath_input.setText(file_path)
                self.load_json_file()  # Automatically load the JSON after dropping the file

    def browse_file(self):
        """Open file dialog to browse for JSON files."""
        file_name, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "JSON Files (*.json);;All Files (*)")
        if file_name:
            self.filepath_input.setText(file_name)
            self.load_json_file()  # Load the JSON when a file is selected

    def load_json_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "JSON Files (*.json);;All Files (*)")
        if file_name:
            self.file_path_line_edit.setText(file_name)
            try:
                with open(file_name, 'r') as file:
                    data = json.load(file)
                    self.load_data_into_tables(data)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load JSON file: {e}")

    def load_data_into_tables(self, data):
        try:
            windows_data = data.get('windows', {})
            detectors_data = data.get('detectors', {})

            # Block signals for both tables before updating them
            self.windows_form.blockSignals(True)
            self.detectors_form.blockSignals(True)

            # Load PIE-Windows
            self.windows_dict.clear()
            self.windows_form.setRowCount(0)
            print(windows_data)
            for window_name, (start, end) in windows_data.items():
                self.add_pie_window_from_data(window_name, start, end)

            # Load Detectors
            self.detectors_dict.clear()
            self.detectors_form.setRowCount(0)
            for detector_name, properties in detectors_data.items():
                self.add_detector_from_data(detector_name, properties["chs"], properties["micro_time_ranges"])

        except KeyError as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")

        finally:
            # Re-enable signals after the update
            self.windows_form.blockSignals(False)
            self.detectors_form.blockSignals(False)

    def add_pie_window_from_data(self, window_name, start, end):
        # Add PIE window to the table based on data
        row = self.windows_form.rowCount()
        self.windows_form.insertRow(row)
        self.windows_form.setItem(row, 0, QTableWidgetItem(window_name))
        self.windows_form.setItem(row, 1, QTableWidgetItem(str(start)))
        self.windows_form.setItem(row, 2, QTableWidgetItem(str(end)))

        # Store the window data in the internal dictionary
        self.windows_dict[window_name] = (QLineEdit(str(start)), QLineEdit(str(end)))

    def add_detector_from_data(self, detector_name, channels, micro_time_ranges):
        # Add detector to the table based on data
        row = self.detectors_form.rowCount()
        self.detectors_form.insertRow(row)
        self.detectors_form.setItem(row, 0, QTableWidgetItem(detector_name))
        self.detectors_form.setItem(row, 1, QTableWidgetItem(', '.join(map(str, channels))))
        micro_time_ranges_str = ', '.join([f"{start}-{end}" for start, end in micro_time_ranges])
        self.detectors_form.setItem(row, 2, QTableWidgetItem(micro_time_ranges_str))

        # Store the detector data in the internal dictionary
        self.detectors_dict[detector_name] = {
            "chs": channels,
            "micro_time_ranges": micro_time_ranges
        }

    def add_detector(self):
        new_detector_name = self.new_detector_name_input.text().strip()
        if new_detector_name in self.detectors_dict:
            QMessageBox.warning(self, "Warning", "This detector name already exists.")
            return

        # Create new rows for channels and micro time ranges with default values
        channels = "0, 1"  # Default channels
        micro_time_ranges = "0-2048"  # Default micro time ranges

        # Add the new detector to the table
        row_position = self.detectors_form.rowCount()
        self.detectors_form.insertRow(row_position)

        # Set new detector name, channels, and micro time ranges
        self.detectors_form.setItem(row_position, 0, QTableWidgetItem(new_detector_name))
        self.detectors_form.setItem(row_position, 1, QTableWidgetItem(channels))
        self.detectors_form.setItem(row_position, 2, QTableWidgetItem(micro_time_ranges))

        # Store the new detector in the dictionary
        self.detectors_dict[new_detector_name] = {
            "chs": list(map(int, channels.split(','))),
            "micro_time_ranges": [tuple(map(int, r.split('-'))) for r in micro_time_ranges.split(',')]
        }

        # Clear the name input field
        self.new_detector_name_input.clear()

    def remove_pie_window(self, item):
        row = item.row()  # Get the row of the double-clicked item
        window_name = self.windows_form.item(row, 0).text()  # Get the window name
        if window_name in self.windows_dict:
            del self.windows_dict[window_name]  # Remove from the dictionary
            self.windows_form.removeRow(row)  # Remove from the table

    def remove_detector(self, item):
        row = item.row()  # Get the row of the double-clicked item
        detector_name = self.detectors_form.item(row, 0).text()  # Get the detector name
        if detector_name in self.detectors_dict:
            del self.detectors_dict[detector_name]  # Remove from the dictionary
            self.detectors_form.removeRow(row)  # Remove from the table

    def create_windows_fields(self):
        # Populate the table with existing windows
        self.windows_form.setRowCount(len(windows))  # Set number of rows based on existing windows
        for row, (window_name, (start, end)) in enumerate(windows.items()):
            start_input = QLineEdit(str(start))
            end_input = QLineEdit(str(end))
            self.windows_dict[window_name] = (start_input, end_input)
            self.windows_widgets[window_name] = (start_input, end_input)  # Store the widgets

            # Add data to table
            self.windows_form.setItem(row, 0, QTableWidgetItem(window_name))  # Window Name
            self.windows_form.setCellWidget(row, 1, start_input)  # Start Input
            self.windows_form.setCellWidget(row, 2, end_input)  # End Input

    def create_detectors_fields(self):
        # Populate the table with existing detectors
        self.detectors_form.setRowCount(len(detectors))  # Set number of rows based on existing detectors
        for row, (detector_name, properties) in enumerate(detectors.items()):
            chs_input = QLineEdit(', '.join(map(str, properties["chs"])))
            micro_time_input = QLineEdit(
                ', '.join([f"{start}-{end}" for start, end in properties["micro_time_ranges"]]))
            self.detectors_dict[detector_name] = (chs_input, micro_time_input)
            self.detectors_widgets[detector_name] = (chs_input, micro_time_input)  # Store the widgets

            # Add data to table
            self.detectors_form.setItem(row, 0, QTableWidgetItem(detector_name))  # Detector Name
            self.detectors_form.setCellWidget(row, 1, chs_input)  # Channels Input
            self.detectors_form.setCellWidget(row, 2, micro_time_input)  # Micro Time Ranges Input

    def add_pie_window(self):
        new_window_name = self.new_window_name_input.text().strip() or f"PIE-Window {len(self.windows_dict) + 1}"
        if new_window_name in self.windows_dict:
            QMessageBox.warning(self, "Warning", "This PIE-Window name already exists.")
            return

        self.windows_dict[new_window_name] = (QLineEdit(), QLineEdit())
        self.windows_widgets[new_window_name] = (
            self.windows_dict[new_window_name][0], self.windows_dict[new_window_name][1])  # Store the widgets

        # Initialize new fields
        self.windows_widgets[new_window_name][0].setText("0")  # Default Start
        self.windows_widgets[new_window_name][1].setText("2048")  # Default End

        # Add new fields to the table
        row = self.windows_form.rowCount()
        self.windows_form.insertRow(row)  # Insert new row for the new window
        self.windows_form.setItem(row, 0, QTableWidgetItem(new_window_name))  # Window Name
        self.windows_form.setCellWidget(row, 1, self.windows_widgets[new_window_name][0])  # Start Input
        self.windows_form.setCellWidget(row, 2, self.windows_widgets[new_window_name][1])  # End Input

    def update_window_name(self, item):
        if item.column() == 0:  # Only update for the Name column
            old_name = list(self.windows_dict.keys())[item.row()]
            new_name = item.text().strip()

            if new_name and new_name != old_name:
                self.windows_dict[new_name] = self.windows_dict.pop(old_name)
                self.windows_widgets[new_name] = self.windows_widgets.pop(old_name)  # Update widgets too
                self.windows_form.setItem(item.row(), 0, QTableWidgetItem(new_name))  # Update table entry

    def update_detector_name(self, item):
        if item.column() == 0:  # Only update for the Name column
            old_name = list(self.detectors_dict.keys())[item.row()]
            new_name = item.text().strip()

            if new_name and new_name != old_name:
                self.detectors_dict[new_name] = self.detectors_dict.pop(old_name)
                self.detectors_widgets[new_name] = self.detectors_widgets.pop(old_name)  # Update widgets too
                self.detectors_form.setItem(item.row(), 0, QTableWidgetItem(new_name))  # Update table entry


    @property
    def windows(self):
        # Gather the data for windows and detectors to be edited
        return {
            name: [(int(self.windows_widgets[name][0].text()), int(self.windows_widgets[name][1].text()))]
            for name in self.windows_dict
        }

    @property
    def detectors(self):
        print('self.detectors_dict:', self.detectors_dict)
        return {
            name: {
                "chs": list(map(int, self.detectors_dict[name][0].text().split(','))),
                "micro_time_ranges": [
                    tuple(map(int, r.split('-'))) for r in self.detectors_dict[name][1].text().split(',')
                ]
            }
            for name in self.detectors_dict
        }

    @property
    def channels(self):
        channels = {}
        w = self.windows
        d = self.detectors

        # Iterate through each window
        for window_name, window_ranges in w.items():
            # Iterate through each detector
            for detector_name, detector_info in d.items():
                for window_range in window_ranges:
                    for micro_time_range in detector_info["micro_time_ranges"]:
                        # Create a unique channel name
                        channel_name = f"{window_name}_{detector_name}"

                        # Store the channel info in the dictionary
                        if channel_name not in channels:
                            channels[channel_name] = []

                        channels[channel_name].append({
                            "window_range": window_range,
                            "detector_chs": detector_info["chs"],
                            "micro_time_range": micro_time_range
                        })

        return channels

    @property
    def settings(self):
        # Gather the data for windows and detectors to be edited
        return {
            "windows": self.windows,
            "detectors": self.detectors,
            "channels": self.channels
        }

    def open_json_editor(self):
        # Prepare data to edit
        try:
            # Open JSON editor dialog with data_to_edit property
            dialog = JsonEditorDialog(self.settings, self)
            if dialog.exec_() == 1:  # Check if the dialog was accepted
                # Update the internal data structure with edited JSON
                edited_data = dialog.get_edited_data()
                if edited_data:
                    self.windows_dict = edited_data['windows']
                    self.detectors_dict = edited_data['detectors']
                    self.update_tables()  # Update the tables with new data
        except KeyError as e:
            QMessageBox.critical(self, "Error", f"Missing data: {e}")

    def update_tables(self):
        # Update the windows table
        self.windows_form.setRowCount(len(self.windows_dict))  # Adjust row count
        for row, (name, (start, end)) in enumerate(self.windows_dict.items()):
            self.windows_form.setItem(row, 0, QTableWidgetItem(name))
            self.windows_form.setCellWidget(row, 1, QLineEdit(str(start)))
            self.windows_form.setCellWidget(row, 2, QLineEdit(str(end)))

        # Update the detectors table
        self.detectors_form.setRowCount(len(self.detectors_dict))  # Adjust row count
        for row, (name, properties) in enumerate(self.detectors_dict.items()):
            self.detectors_form.setItem(row, 0, QTableWidgetItem(name))
            self.detectors_form.setCellWidget(row, 1, QLineEdit(', '.join(map(str, properties["chs"]))))
            self.detectors_form.setCellWidget(row, 2, QLineEdit(
                ', '.join([f"{start}-{end}" for start, end in properties["micro_time_ranges"]])))

    def save_settings(self, filename=None):
        # Prepare window data
        window_data = {}
        for window_name, (start, end) in self.windows_dict.items():
            window_data[window_name] = (int(start.text()), int(end.text()))

        # Prepare detector data
        detector_data = {}
        for detector_name in self.detectors_dict.keys():
            chs, micro_time_ranges = self.detectors_dict[detector_name]
            chs = list(map(int, chs.text().split(',')))
            micro_time_ranges = [tuple(map(int, r.split('-'))) for r in micro_time_ranges.text().split(',')]
            detector_data[detector_name] = {
                "chs": chs,
                "micro_time_ranges": micro_time_ranges
            }

        # Consolidate data into one dictionary
        data = {
            "windows": window_data,
            "detectors": detector_data
        }

        # If no filename is provided, open a file dialog to get the filename
        if not filename:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Settings", "", "JSON Files (*.json);;All Files (*)")
            if not filename:  # If user cancels the file dialog
                return

        # Save the settings to the file
        try:
            with open(filename, 'w') as file:
                json.dump(data, file, indent=4)
            QMessageBox.information(self, "Success", f"Settings saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {e}")


class DetectorWizard(QWizard):
    def __init__(self, json_file=None):
        super().__init__()

        self.addPage(DetectorWizardPage(json_file=json_file))
        self.setWindowTitle("Detector Configuration Wizard")


# Initial PIE-Windows and Detectors
windows = {
    "prompt": (0, 2048),
    "delayed": (2048, 4095)
}

detectors = {
    "green": {
        "chs": [0, 8],
        "micro_time_ranges": [(0, 4095)]
    },
    "red": {
        "chs": [1, 9],
        "micro_time_ranges": [(0, 2048)]
    },
    "yellow": {
        "chs": [1, 9],
        "micro_time_ranges": [(2048, 4095)]
    },
}

if __name__ == "__main__":
    # Optionally pass a JSON file from the command line:
    json_file_arg = sys.argv[1] if len(sys.argv) > 1 else None

    app = QApplication(sys.argv)
    wizard = DetectorWizard(json_file=json_file_arg)
    wizard.show()
    sys.exit(app.exec_())
