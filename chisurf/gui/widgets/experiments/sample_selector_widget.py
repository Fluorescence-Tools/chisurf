from __future__ import annotations

from qtpy import QtWidgets

import chisurf as cs


class SampleLookupDialog(QtWidgets.QDialog):
    """Modal sample lookup dialog shown when a newly loaded file has no sample."""

    def __init__(self, controller, filename, content_md5, parent=None):
        """Initialize the dialog.

        Parameters
        ----------
        controller : ExperimentReaderController
            Reader controller that owns the current experiment reader.
        filename : str
            Path to the loaded file.
        content_md5 : str
            MD5 hex digest of the file content.
        parent : QWidget, optional
            Parent widget.
        """
        super().__init__(parent)
        self.controller = controller
        self.filename = filename
        self.content_md5 = content_md5
        self._updating = False
        self.setWindowTitle("Assign sample")
        self.resize(640, 360)

        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel(
            "This file is not linked to a sample yet.\n"
            "Select an existing sample, enter a new sample ID, or skip."
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        file_label = QtWidgets.QLabel(f"File: {filename}")
        file_label.setWordWrap(True)
        layout.addWidget(file_label)

        md5_label = QtWidgets.QLabel(f"MD5: {content_md5}")
        md5_label.setWordWrap(True)
        layout.addWidget(md5_label)

        sample_row = QtWidgets.QHBoxLayout()
        sample_row.addWidget(QtWidgets.QLabel("Sample ID:"))
        self.sample_combo = QtWidgets.QComboBox(self)
        self.sample_combo.setEditable(True)
        self.sample_combo.setPlaceholderText("select existing sample or type a new sample-id")
        self.sample_combo.addItem("Skip")
        self._populate_samples()
        self.sample_combo.editTextChanged.connect(self._on_edit_text_changed)
        self.sample_combo.currentTextChanged.connect(self._on_sample_changed)
        sample_row.addWidget(self.sample_combo, 1)
        layout.addLayout(sample_row)

        layout.addWidget(QtWidgets.QLabel("Details / description:"))
        self.details_edit = QtWidgets.QTextEdit(self)
        self.details_edit.setPlaceholderText(
            "Optional details for this sample. These are stored in MFDB."
        )
        self.details_edit.setMaximumHeight(120)
        layout.addWidget(self.details_edit, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel,
            self,
        )
        self.skip_button = QtWidgets.QPushButton("Skip", self)
        buttons.addButton(self.skip_button, QtWidgets.QDialogButtonBox.ActionRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.skip_button.clicked.connect(self.skip)
        layout.addWidget(buttons)

    def _reader(self):
        """Return the underlying reader instance."""
        return getattr(self.controller, "experiment_reader", None)

    def _db(self):
        """Return the MFDB connection, if available."""
        return getattr(self._reader(), "db", None) or getattr(self.controller, "db", None)

    def _populate_samples(self):
        """Populate the sample combo box from MFDB."""
        if self._updating:
            return
        self._updating = True
        try:
            db = self._db()
            if db is None:
                return
            for row in db.list_samples():
                self.sample_combo.addItem(row["sample_id"])
        except Exception:
            pass
        finally:
            self._updating = False

    def search_samples(self, query):
        """Update sample suggestions while the user types."""
        if self._updating:
            return
        self._updating = True
        self.sample_combo.clear()
        self.sample_combo.addItem("Skip")
        try:
            db = self._db()
            if db is None:
                return
            for row in db.search_samples(query):
                self.sample_combo.addItem(row["sample_id"])
        except Exception:
            pass
        if query and self.sample_combo.findText(query) < 0:
            self.sample_combo.addItem(query)
        finally:
            self._updating = False

    def _on_edit_text_changed(self, text):
        """Search MFDB dynamically as the user types."""
        if self._updating:
            return
        if len(text) >= 2:
            self.search_samples(text)
        elif text == "":
            self._populate_samples()

    def _on_sample_changed(self, sample_id):
        """Fill the details field when an existing sample is selected."""
        if self._updating or sample_id == "Skip" or sample_id == "":
            self.details_edit.clear()
            return
        try:
            db = self._db()
            if db is None:
                return
            row = db.get_sample(sample_id)
            if row is not None:
                self._updating = True
                self.details_edit.setPlainText(row.get("details") or "")
                self._updating = False
        except Exception:
            pass

    def _sync_reader_sample_id(self, sample_id):
        """Write the selected sample ID back to the reader."""
        reader = self._reader()
        if reader is not None:
            reader.sample_id = sample_id

    def _store_sample(self, sample_id, details):
        """Store the selected sample ID in MFDB and object metadata."""
        try:
            db = self._db()
            if db is None:
                return
            reader = self._reader()
            object_uuid = getattr(reader, "_source_object_uuids", [None])[-1]
            if sample_id:
                if db.get_sample(sample_id) is None:
                    db.add_sample(sample_id, details=details or "")
                elif details:
                    db.update_sample(sample_id, details=details)
                if object_uuid:
                    db.set_object_sample_id(object_uuid, sample_id)
                self._sync_reader_sample_id(sample_id)
            elif object_uuid:
                db.set_object_sample_id(object_uuid, None)
                self._sync_reader_sample_id(None)
        except Exception:
            pass

    def accept(self):
        """Store the selected sample and close the dialog."""
        text = self.sample_combo.currentText().strip()
        sample_id = None if text == "Skip" or text == "" else text
        details = self.details_edit.toPlainText().strip()
        self._store_sample(sample_id, details)
        super().accept()

    def skip(self):
        """Clear the sample assignment and close the dialog."""
        self._store_sample(None, "")
        self.reject()


def show_sample_lookup_dialog(controller, filename, content_md5, object_uuid=None):
    """Show the sample lookup dialog and return the selected sample ID.

    Parameters
    ----------
    controller : ExperimentReaderController
        Reader controller that owns the current experiment reader.
    filename : str
        Path to the loaded file.
    content_md5 : str
        MD5 hex digest of the file content.
    object_uuid : str, optional
        Object-store UUID of the loaded file.

    Returns
    -------
    str or None
        Selected sample ID, or None when the user skips.
    """
    dialog = SampleLookupDialog(controller, filename, content_md5)
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        return dialog.sample_combo.currentText().strip() or None
    return None
