import json

from qtpy.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QMessageBox


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
