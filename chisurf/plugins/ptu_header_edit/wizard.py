import sys
import json
import tttrlib  # Import the tttrlib module
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QPlainTextEdit,
    QTableWidgetItem,
    QPushButton,
    QMessageBox,
    QInputDialog,
    QHBoxLayout,
    QComboBox,
    QTextEdit,
    QFileDialog,
)


class TagsEditor(QMainWindow):
    TYPE_MAPPING = {
        0xFFFF0008: "Empty",
        0x00000008: "Bool",
        0x10000008: "Int8",
        0x11000008: "BitSet64",
        0x12000008: "Color8",
        0x20000008: "Float8",
        0x21000008: "DateTime",
        0x2001FFFF: "Float8Array",
        0x4001FFFF: "AnsiString",
        0x4002FFFF: "WideString",
        0xFFFFFFFF: "BinaryBlob"
    }

    REVERSE_TYPE_MAPPING = {v: k for k, v in TYPE_MAPPING.items()}

    def __init__(self, json_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Create and set up the table widget
        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(['Name', 'Type', 'Value', 'Idx'])

        # Set column widths: fixed for Name and Type, stretchable for Value
        self.table_widget.setColumnWidth(0, 150)  # Name column width
        self.table_widget.setColumnWidth(1, 120)  # Type column width
        self.table_widget.setColumnWidth(2, 200)  # Type column width
        self.table_widget.horizontalHeader().setStretchLastSection(True)  # Make the last section stretchable

        # Create JSON display area
        self.json_display = QTextEdit(self)
        self.json_display.setReadOnly(True)
        self.json_display.hide()  # Hide by default
        self.layout.addWidget(self.json_display)

        # Create button to show/hide JSON display
        self.toggle_json_button = QPushButton("Show JSON", self)
        self.toggle_json_button.clicked.connect(self.toggle_json_display)
        self.layout.addWidget(self.toggle_json_button)

        # Add the table to the layout
        self.layout.addWidget(self.table_widget)

        # Create buttons for adding, removing, and saving tags
        button_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Tag")
        self.remove_button = QPushButton("Remove Tag")
        self.open_button = QPushButton("Open PTU File")  # Button to open PTU files
        self.save_button = QPushButton("Save Modified PTU")  # Button to save modified PTU
        button_layout.addWidget(self.open_button)
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.save_button)  # Add save button
        self.layout.addLayout(button_layout)

        # Load the JSON data into the table widget
        self.load_tags(json_data)

        # Connect button actions
        self.add_button.clicked.connect(self.add_tag)
        self.remove_button.clicked.connect(self.remove_tag)
        self.open_button.clicked.connect(self.open_ptu_file)  # Connect open button
        self.save_button.clicked.connect(self.save_modified_ptu)  # Connect save button
        self.table_widget.itemChanged.connect(self.on_item_changed)

        # Update the JSON display on initialization
        self.update_json_display()

        self.opened_ptu_file_path = None
        self.parsed_json = dict()

    def convert_value_by_type(self, value_str, tag_type_str):
        """
        Convert the value string to the correct Python type based on the tag's type.
        :param value_str: The value as a string from the table.
        :param tag_type_str: The tag type as a string (e.g., "Int8", "Float8").
        :return: The converted value in its appropriate type.
        """
        try:
            if tag_type_str in ["Int8", "BitSet64"]:
                return int(value_str)
            elif tag_type_str in ["Float8", "Float8Array"]:
                return float(value_str)
            elif tag_type_str == "Bool":
                return value_str.lower() == "true"
            elif tag_type_str in ["AnsiString", "WideString", "BinaryBlob", "Empty"]:
                return value_str  # Keep these types as strings
            elif tag_type_str == "DateTime":
                return float(value_str)  # Assuming it's a timestamp stored as a float
            else:
                print(f"Unknown tag type: {tag_type_str}. Returning raw value.")
                return value_str  # Return as string if the type is unknown
        except ValueError as e:
            print(f"Error converting value: {value_str} for type: {tag_type_str}. Error: {str(e)}")
            return value_str  # Return raw value if conversion fails

    def load_tags(self, json_data):
        # Parse JSON data
        parsed_json = json.loads(json_data)
        self.parsed_json = parsed_json
        tags = parsed_json.get("tags", [])

        # Populate the table with tag data
        self.table_widget.setRowCount(len(tags))
        for row, tag in enumerate(tags):
            self.table_widget.setItem(row, 0, QTableWidgetItem(tag.get("name", "")))

            # Create a combobox for type selection
            combo_box = QComboBox()
            combo_box.addItems(self.TYPE_MAPPING.values())
            # Set the current type based on the tag's type
            tag_type_value = tag.get("type", "")
            combo_box.setCurrentText(self.TYPE_MAPPING.get(tag_type_value, ""))
            self.table_widget.setCellWidget(row, 1, combo_box)

            # Set the value
            self.table_widget.setItem(row, 2, QTableWidgetItem(str(tag.get("value", ""))))

            # Set the idx
            self.table_widget.setItem(row, 3, QTableWidgetItem(str(tag.get("idx", "-1"))))

    def add_tag(self):
        # Prompt for new tag details
        name, ok1 = QInputDialog.getText(self, 'Add Tag', 'Enter tag name:')
        if not ok1 or not name:
            return

        # Create a new row for the new tag
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        self.table_widget.setItem(row_position, 0, QTableWidgetItem(name))

        # Create a combobox for type selection
        combo_box = QComboBox()
        combo_box.addItems(self.TYPE_MAPPING.values())
        self.table_widget.setCellWidget(row_position, 1, combo_box)

        # Prompt for value
        value, ok3 = QInputDialog.getText(self, 'Add Tag', 'Enter tag value:')
        if not ok3 or not value:
            return
        self.table_widget.setItem(row_position, 2, QTableWidgetItem(value))

        # Prompt for idx (this returns an integer)
        idx, ok4 = QInputDialog.getInt(self, 'Add Tag', 'Enter tag idx:')
        if not ok4:
            return
        # Set the idx (convert the integer to a string)
        self.table_widget.setItem(row_position, 3, QTableWidgetItem(str(idx)))

        # Update the JSON display
        self.update_json_display()

    def remove_tag(self):
        current_row = self.table_widget.currentRow()
        if current_row >= 0:
            self.table_widget.removeRow(current_row)
            # Update the JSON display
            self.update_json_display()
        else:
            QMessageBox.warning(self, "Warning", "No tag selected to remove.")

    def open_ptu_file(self):
        # Open a file dialog to select a PTU file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open PTU File", "", "PTU Files (*.ptu);;All Files (*)",
                                                   options=options)
        if file_name:
            try:
                self.opened_ptu_file_path = file_name
                d = tttrlib.TTTR(file_name)  # Load the PTU file using tttrlib
                json_str = d.header.json  # Extract the JSON string from the header
                self.load_tags(json_str)  # Load the tags into the editor
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to open file:\n{str(e)}")

    def toggle_json_display(self):
        if self.json_display.isVisible():
            self.json_display.hide()
            self.toggle_json_button.setText("Show JSON")
        else:
            self.json_display.show()
            self.toggle_json_button.setText("Hide JSON")

    def on_item_changed(self, item):
        # Update the JSON display whenever an item is changed
        self.update_json_display()

    def save_modified_ptu(self):
        # Open a file dialog to save the modified PTU file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Modified PTU File", "",
                                                   "PTU Files (*.ptu);;All Files (*)", options=options)

        if file_name:
            # Ensure the filename ends with .ptu
            if not file_name.endswith('.ptu'):
                file_name += '.ptu'  # Append .ptu extension

            try:
                # Load the original PTU file
                original_ptu_path = self.opened_ptu_file_path  # Assume you have a way to store the path of the opened PTU
                tttr = tttrlib.TTTR(original_ptu_path)

                # Create a new empty TTTR container
                d2 = tttrlib.TTTR()

                # Load header from original TTTR
                header_dict = json.loads(tttr.header.json)
                header_dict['tags'] = self.parsed_json.get("tags", [])
                js = json.dumps(header_dict)
                print(js)

                # Set the header information of the empty TTTR container
                d2.header.set_json(js)

                # Assign events to the new TTTR container
                events = {
                    "macro_times": tttr.macro_times,
                    "micro_times": tttr.micro_times,
                    "routing_channels": tttr.routing_channels,
                    "event_types": tttr.event_types
                }
                d2.append_events(**events)

                # Write the new TTTR container to the selected file
                d2.write(file_name)

                QMessageBox.information(self, "Success", "Modified PTU file saved successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save modified PTU file:\n{str(e)}")


    def update_json_display(self):
        # Create JSON from the current table data
        tags = []
        for row in range(self.table_widget.rowCount()):
            name_item = self.table_widget.item(row, 0)
            tag_type_combo = self.table_widget.cellWidget(row, 1)
            value_item = self.table_widget.item(row, 2)
            idx_item = self.table_widget.item(row, 3)

            # Check if the name item exists and get its text
            name = name_item.text() if name_item is not None else ""

            # Check if the value item exists and get its text
            value_str = value_item.text() if value_item is not None else ""

            # Get the idx and convert it to an integer (handling any possible conversion errors)
            idx_str = idx_item.text() if idx_item is not None else "-1"
            try:
                idx = int(idx_str)
            except ValueError:
                idx = -1  # Default value if conversion fails

            # Ensure tag_type_combo exists before accessing it
            if tag_type_combo is not None:
                tag_type_str = tag_type_combo.currentText()
                tag_type = self.REVERSE_TYPE_MAPPING.get(tag_type_combo.currentText(), None)
            else:
                tag_type = None
                tag_type_str = None
                print(f"Warning: tag_type_combo is None for row {row}. Name: {name}")
                continue

            # Print type information for debugging
            if tag_type is None:
                print(
                    f"Warning: Unrecognized type for '{name}'. Current type: {tag_type_combo.currentText() if tag_type_combo else 'None'}")
                continue

            # Convert the value based on its type
            value = self.convert_value_by_type(value_str, tag_type_str)

            if tag_type is not None:
                tags.append({"name": name, "type": tag_type, "value": value, "idx": idx})

        self.parsed_json['tags'] = tags

        # Update the JSON display text
        self.json_display.setPlainText(json.dumps(self.parsed_json, indent=4))


# Enhanced JSON data for initialization
json_data = '''{
    "tags": [
        {"name": "File_GUID", "type": 1073872895, "value": "{D3D2D9C0-5B48-4D94-98CE-A5E2EA3558E6}"},
        {"name": "File_CreatingTime", "type": 553648136, "value": "1539705731.9470003"},
        {"name": "Measurement_SubMode", "type": 268435464, "value": "1"},
        {"name": "User_Author", "type": 2000000001, "value": "John Doe"},
        {"name": "File_Version", "type": 1000000002, "value": "1.0.0"},
        {"name": "File_Description", "type": 2000000003, "value": "This is a sample file description."}
    ]
}'''


if __name__ == "plugin":
    wizard = TagsEditor(json_data)
    wizard.resize(640, 480)
    wizard.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    editor = TagsEditor(json_data)
    editor.setWindowTitle("PTU tag Editor")
    editor.resize(640, 480)
    editor.show()
    sys.exit(app.exec_())
