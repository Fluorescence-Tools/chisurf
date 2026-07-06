"""Dynamic, schema-driven detail and edit form widget for MFDB tables."""

from __future__ import annotations
from typing import Any, Callable
from qtpy import QtCore, QtWidgets, QtGui
import chisurf.logging

# Legacy hardcoded schemas — kept for backward compatibility only.
# New code should use entity_schema.field_specs_for_category() instead.
SCHEMAS = {
    "user": [
        {"name": "user_id", "label": "User ID", "type": "str", "required": True},
        {"name": "user_uuid", "label": "UUID", "type": "str", "readonly": True, "placeholder": "Auto-generated if left empty"},
        {"name": "display_name", "label": "Display Name", "type": "str", "required": True},
        {"name": "email", "label": "Email", "type": "str"},
        {"name": "role", "label": "Role", "type": "choice", "choices": ["user", "admin", "pi", "postdoc", "phd", "student"]},
        {"name": "affiliation", "label": "Affiliation", "type": "str"},
        {"name": "department", "label": "Department", "type": "str"},
        {"name": "phone", "label": "Phone", "type": "str"},
        {"name": "website", "label": "Website", "type": "str"},
        {"name": "address", "label": "Address", "type": "text"},
        {"name": "is_admin", "label": "Is Admin", "type": "bool"},
        {"name": "allow_passwordless_login", "label": "Allow Passwordless Login", "type": "bool"},
        {"name": "active_branch_uuid", "label": "Active Branch UUID", "type": "str"},
        {"name": "created_at", "label": "Created At", "type": "str", "readonly": True},
        {"name": "updated_at", "label": "Updated At", "type": "str", "readonly": True},
        {"name": "details", "label": "Details", "type": "text"},
    ],
    "device": [
        {"name": "device_id", "label": "Device ID", "type": "str", "required": True},
        {"name": "name", "label": "Name", "type": "str", "required": True},
        {"name": "device_type", "label": "Device Type", "type": "str"},
        {"name": "model", "label": "Model", "type": "str"},
        {"name": "serial_number", "label": "Serial Number", "type": "str"},
        {"name": "location", "label": "Location", "type": "str"},
        {"name": "owner", "label": "Owner", "type": "str"},
        {"name": "details", "label": "Details", "type": "text"},
        {"name": "created_at", "label": "Created At", "type": "str", "readonly": True},
        {"name": "updated_at", "label": "Updated At", "type": "str", "readonly": True},
    ],
    "sample": [
        {"name": "sample_id", "label": "Sample ID", "type": "str", "required": True},
        {"name": "sample_uuid", "label": "UUID", "type": "str", "readonly": True, "placeholder": "Auto-generated if left empty"},
        {"name": "description", "label": "Description", "type": "str"},
        {"name": "num_of_probes", "label": "Number of Probes", "type": "int"},
        {"name": "solvent_phase", "label": "Solvent Phase", "type": "choice", "choices": ["liquid", "vitrified", "other"]},
        {"name": "sample_condition_id", "label": "Condition ID", "type": "str"},
        {"name": "entity_assembly_id", "label": "Entity Assembly ID", "type": "str"},
        {"name": "project_id", "label": "Project ID", "type": "str"},
        {"name": "measured_by_user_id", "label": "Measured By (User)", "type": "str"},
        {"name": "measured_by_device_id", "label": "Measured By (Device)", "type": "str"},
        {"name": "measured_at", "label": "Measured At", "type": "str"},
        {"name": "details", "label": "Details", "type": "text"},
        {"name": "created_at", "label": "Created At", "type": "str", "readonly": True},
        {"name": "updated_at", "label": "Updated At", "type": "str", "readonly": True},
    ],
    "experiment": [
        {"name": "experiment_id", "label": "Experiment ID", "type": "str", "required": True},
        {"name": "type_id", "label": "Type ID", "type": "int"},
        {"name": "sample_id", "label": "Sample ID", "type": "str"},
        {"name": "project_id", "label": "Project ID", "type": "str"},
        {"name": "measured_by_user_id", "label": "Measured By (User)", "type": "str"},
        {"name": "measured_by_device_id", "label": "Measured By (Device)", "type": "str"},
        {"name": "started_at", "label": "Started At", "type": "str"},
        {"name": "ended_at", "label": "Ended At", "type": "str"},
        {"name": "status", "label": "Status", "type": "str"},
        {"name": "setup_definition_id", "label": "Setup ID", "type": "str"},
        {"name": "details", "label": "Details", "type": "text"},
        {"name": "created_at", "label": "Created At", "type": "str", "readonly": True},
        {"name": "updated_at", "label": "Updated At", "type": "str", "readonly": True},
    ],
    "experiment_type": [
        {"name": "type_id", "label": "Type ID", "type": "int", "readonly": True},
        {"name": "name", "label": "Name", "type": "str", "required": True},
        {"name": "category", "label": "Category", "type": "str"},
        {"name": "description", "label": "Description", "type": "str"},
        {"name": "details", "label": "Details", "type": "text"},
        {"name": "created_at", "label": "Created At", "type": "str", "readonly": True},
        {"name": "updated_at", "label": "Updated At", "type": "str", "readonly": True},
    ],
    "branch": [
        {"name": "branch_uuid", "label": "Branch UUID", "type": "str", "readonly": True, "placeholder": "Auto-generated UUID"},
        {"name": "name", "label": "Branch Name", "type": "str", "required": True},
        {"name": "parent_branch_uuid", "label": "Parent Branch UUID", "type": "str"},
        {"name": "head_operation_id", "label": "Head Operation ID", "type": "str"},
        {"name": "description", "label": "Description", "type": "text"},
    ],
    "probe": [
        {"name": "probe_id", "label": "Probe ID", "type": "int", "readonly": True},
        {"name": "chromophore_name", "label": "Chromophore Name", "type": "str", "required": True},
        {"name": "reactive_probe_flag", "label": "Reactive Probe Flag", "type": "str"},
        {"name": "reactive_probe_name", "label": "Reactive Probe Name", "type": "str"},
        {"name": "probe_origin", "label": "Probe Origin", "type": "str"},
        {"name": "probe_link_type", "label": "Probe Link Type", "type": "str"},
        {"name": "fluorophore_type", "label": "Fluorophore Type", "type": "str"},
        {"name": "description", "label": "Description", "type": "text"},
        {"name": "category", "label": "Category", "type": "str"},
        {"name": "is_curated", "label": "Is Curated", "type": "bool"},
        {"name": "quality_flag", "label": "Quality Flag", "type": "int"},
        {"name": "abs_max", "label": "Absorbance Max (nm)", "type": "float"},
        {"name": "em_max", "label": "Emission Max (nm)", "type": "float"},
        {"name": "qy", "label": "Quantum Yield", "type": "float"},
        {"name": "ext_coeff", "label": "Extinction Coefficient", "type": "float"},
        {"name": "created_at", "label": "Created At", "type": "str", "readonly": True},
        {"name": "updated_at", "label": "Updated At", "type": "str", "readonly": True},
    ],
    "setup": [
        {"name": "setup_id", "label": "Setup ID", "type": "str", "required": True},
        {"name": "name", "label": "Name", "type": "str", "required": True},
        {"name": "instrument_type", "label": "Instrument Type", "type": "str"},
        {"name": "laser_wavelengths", "label": "Laser Wavelengths", "type": "str", "placeholder": "[485, 640]"},
        {"name": "detector_channels", "label": "Detector Channels", "type": "str", "placeholder": '{"ch1": ...}'},
        {"name": "details", "label": "Details", "type": "text"},
        {"name": "created_by_user_id", "label": "Owner", "type": "str", "readonly": True},
        {"name": "is_public", "label": "Public", "type": "bool"},
        {"name": "created_at", "label": "Created At", "type": "str", "readonly": True},
        {"name": "updated_at", "label": "Updated At", "type": "str", "readonly": True},
    ],
    "project": [
        {"name": "project_id", "label": "Version ID", "type": "str", "readonly": True},
        {"name": "name", "label": "Name", "type": "str", "readonly": True},
        {"name": "description", "label": "Notes", "type": "text", "readonly": True},
        {"name": "created_at", "label": "Created At", "type": "str", "readonly": True},
        {"name": "updated_at", "label": "Updated At", "type": "str", "readonly": True},
    ],
    "raw_data": [
        {"name": "raw_data_id", "label": "Raw Data ID", "type": "str", "readonly": True},
        {"name": "experiment_id", "label": "Experiment ID", "type": "str", "readonly": True},
        {"name": "data_type", "label": "Data Type", "type": "str", "readonly": True},
        {"name": "storage_mode", "label": "Storage Mode", "type": "str", "readonly": True},
        {"name": "location", "label": "File Path/URL/Folder", "type": "str", "readonly": True},
        {"name": "validation_status", "label": "Validation Status", "type": "str", "readonly": True},
        {"name": "checksum", "label": "Checksum (SHA-256)", "type": "str", "readonly": True},
        {"name": "details", "label": "Details/JSON", "type": "text", "readonly": True},
    ],
    "processing_run": [
        {"name": "processing_id", "label": "Processing ID", "type": "str", "readonly": True},
        {"name": "experiment_id", "label": "Experiment ID", "type": "str", "readonly": True},
        {"name": "type", "label": "Type", "type": "str", "readonly": True},
        {"name": "status", "label": "Status", "type": "str", "readonly": True},
        {"name": "settings", "label": "Settings/JSON", "type": "text", "readonly": True},
    ],
    "processed_product": [
        {"name": "product_id", "label": "Product ID", "type": "str", "readonly": True},
        {"name": "processing_id", "label": "Processing Run ID", "type": "str", "readonly": True},
        {"name": "product_type", "label": "Product Type", "type": "str", "readonly": True},
        {"name": "storage_mode", "label": "Storage Mode", "type": "str", "readonly": True},
        {"name": "location", "label": "File Path/URL/Folder", "type": "str", "readonly": True},
        {"name": "validation_status", "label": "Validation Status", "type": "str", "readonly": True},
        {"name": "checksum", "label": "Checksum (SHA-256)", "type": "str", "readonly": True},
        {"name": "experiment_id", "label": "Experiment ID", "type": "str", "readonly": True},
    ],
    "object": [
        {"name": "object_uuid", "label": "Object UUID", "type": "str", "readonly": True},
        {"name": "content_md5", "label": "MD5", "type": "str", "readonly": True},
        {"name": "original_filename", "label": "Original Filename", "type": "str", "readonly": True},
        {"name": "size_bytes", "label": "Size Bytes", "type": "int", "readonly": True},
        {"name": "mime_type", "label": "MIME Type", "type": "str", "readonly": True},
        {"name": "refcount", "label": "Refcount", "type": "int", "readonly": True},
        {"name": "storage_path", "label": "Storage Path", "type": "str", "readonly": True},
        {"name": "created_at", "label": "Created At", "type": "str", "readonly": True},
        {"name": "created_by", "label": "Created By", "type": "str", "readonly": True},
    ],
    "analysis": [
        {"name": "analysis_id", "label": "Analysis ID", "type": "str", "readonly": True},
        {"name": "experiment_id", "label": "Experiment ID", "type": "str", "readonly": True},
        {"name": "type", "label": "Type", "type": "str", "readonly": True},
        {"name": "model_name", "label": "Model Name", "type": "str", "readonly": True},
        {"name": "settings", "label": "Settings/JSON", "type": "text", "readonly": True},
    ],
    "condition": [
        {"name": "condition_id", "label": "Condition ID", "type": "str", "required": True},
        {"name": "ph", "label": "pH", "type": "float"},
        {"name": "temperature", "label": "Temperature [K]", "type": "float"},
        {"name": "ionic_strength", "label": "Ionic Strength [M]", "type": "float"},
        {"name": "buffer_composition", "label": "Buffer", "type": "str"},
        {"name": "details", "label": "Details", "type": "text"},
    ],
}

class MFDBDetailWidget(QtWidgets.QWidget):
    """A generic form layout widget driven by MFDB schemas.

    Accepts either a ``schema_type`` string (for backward compatibility
    with legacy code) or a ``field_specs`` list (preferred for new code).

    Signals
    -------
    dataChanged()
        Fires on every keystroke / live change (kept for backward compat).
    commitRequested()
        Fires only when the user explicitly commits an edit: Enter key on a
        line/spin field, focus-out from a text area, a dropdown selection, or
        a checkbox toggle.  Wire this to auto-save logic — it respects Qt's
        built-in Ctrl+Z undo stack because the user can still undo *within*
        a field before pressing Enter.
    """
    dataChanged = QtCore.Signal()
    commitRequested = QtCore.Signal()

    def __init__(
        self,
        schema_type: str | None = None,
        field_specs: list[dict[str, Any]] | None = None,
        dropdown_providers: dict[str, Callable[[], list[tuple[str, str]]]] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        from .entity_schema import FieldSpec

        super().__init__(parent)
        self.schema_type = schema_type or ""
        self.dropdown_providers = dropdown_providers or {}

        # Build schema from field_specs (preferred) or legacy SCHEMAS dict
        if field_specs:
            self.schema = [
                {
                    "name": fs.name if isinstance(fs, FieldSpec) else fs.get("name", ""),
                    "label": fs.label if isinstance(fs, FieldSpec) else fs.get("label", fs.get("name", "")),
                    "type": fs.widget if isinstance(fs, FieldSpec) else fs.get("type", "str"),
                    "required": fs.required if isinstance(fs, FieldSpec) else fs.get("required", False),
                    "readonly": fs.readonly if isinstance(fs, FieldSpec) else fs.get("readonly", False),
                    "choices": list(fs.choices) if isinstance(fs, FieldSpec) else fs.get("choices", []),
                    "placeholder": fs.placeholder if isinstance(fs, FieldSpec) else fs.get("placeholder", ""),
                    "tooltip": fs.tooltip if isinstance(fs, FieldSpec) else fs.get("tooltip", ""),
                }
                for fs in field_specs
            ]
        else:
            self.schema = SCHEMAS.get(self.schema_type, [])

        self.widgets: dict[str, QtWidgets.QWidget] = {}
        self._loading = False
        self._text_areas: list[QtWidgets.QPlainTextEdit] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        for field in self.schema:
            name = field["name"]
            label = field.get("label", name)
            ftype = field.get("type", "str")
            readonly = field.get("readonly", False)
            required = field.get("required", False)
            placeholder = field.get("placeholder", "")
            tooltip = field.get("tooltip", "")

            if required:
                label = f"* {label}"

            widget: QtWidgets.QWidget
            if ftype == "choice" or name in self.dropdown_providers:
                widget = QtWidgets.QComboBox()
                widget.setEditable(True)
                if ftype == "choice":
                    for choice in field.get("choices", []):
                        widget.addItem(choice, choice)
                # activated fires only on user selection (not programmatic),
                # preserving Ctrl+Z undo within the line edit before selecting.
                widget.activated.connect(self._on_commit)
                widget.currentIndexChanged.connect(self._on_changed)
                # Enter in the editable line of the combo
                if widget.lineEdit() is not None:
                    widget.lineEdit().editingFinished.connect(self._on_commit)
                    widget.lineEdit().textChanged.connect(self._on_changed)
            elif ftype == "bool":
                widget = QtWidgets.QCheckBox()
                widget.stateChanged.connect(self._on_changed)
                widget.toggled.connect(self._on_commit)
            elif ftype == "text":
                widget = QtWidgets.QPlainTextEdit()
                widget.setMinimumHeight(60)
                widget.textChanged.connect(self._on_changed)
                # Commit on focus-out (detected via eventFilter below)
                widget.installEventFilter(self)
                if not readonly:
                    self._text_areas.append(widget)
            elif ftype == "int":
                widget = QtWidgets.QSpinBox()
                widget.setRange(-2147483648, 2147483647)
                widget.valueChanged.connect(self._on_changed)
                widget.editingFinished.connect(self._on_commit)
            elif ftype == "float":
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.valueChanged.connect(self._on_changed)
                widget.editingFinished.connect(self._on_commit)
            else:
                widget = QtWidgets.QLineEdit()
                widget.textChanged.connect(self._on_changed)
                widget.editingFinished.connect(self._on_commit)

            if placeholder and hasattr(widget, "setPlaceholderText"):
                widget.setPlaceholderText(placeholder)

            if tooltip:
                widget.setToolTip(tooltip)

            if readonly:
                if hasattr(widget, "setReadOnly"):
                    widget.setReadOnly(True)
                elif hasattr(widget, "setEnabled"):
                    widget.setEnabled(False)

            self.widgets[name] = widget
            layout.addRow(label, widget)

    def refresh_dropdowns(self) -> None:
        """Call dropdown providers to populate choice fields asynchronously/on-demand."""
        old_loading = self._loading
        self._loading = True
        try:
            for name, provider in self.dropdown_providers.items():
                widget = self.widgets.get(name)
                if isinstance(widget, QtWidgets.QComboBox):
                    current_text = widget.currentText()
                    current_data = widget.currentData()
                    widget.clear()
                    try:
                        choices = provider()
                        for item in choices:
                            if isinstance(item, tuple):
                                val, lbl = item
                            else:
                                val = lbl = item
                            widget.addItem(lbl, val)
                    except Exception as _exc:
                        chisurf.logging.warning("Dropdown provider failed: %s", _exc)
                    # Restore selection
                    idx = -1
                    if current_data is not None:
                        idx = widget.findData(current_data)
                    if idx < 0 and current_text:
                        idx = widget.findText(current_text)
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                    else:
                        widget.setEditText(current_text)
        finally:
            self._loading = old_loading

    def _on_changed(self, *args: Any) -> None:
        if not self._loading:
            self.dataChanged.emit()

    def _on_commit(self, *args: Any) -> None:
        """Emit commitRequested when the user explicitly commits an edit."""
        if not self._loading:
            self.commitRequested.emit()

    def eventFilter(self, obj: Any, event: Any) -> bool:
        """Detect focus-out on QPlainTextEdit fields to trigger commit."""
        from qtpy.QtCore import QEvent
        if event.type() == QEvent.FocusOut and obj in self._text_areas:
            self._on_commit()
        return False

    def set_data(self, data: dict[str, Any]) -> None:
        """Populate form fields with database record data."""
        self._loading = True
        try:
            self.refresh_dropdowns()
            for field in self.schema:
                name = field["name"]
                val = data.get(name)
                widget = self.widgets.get(name)
                if not widget:
                    continue

                if isinstance(val, (list, dict)):
                    import json
                    val = json.dumps(val)

                if isinstance(widget, QtWidgets.QComboBox):
                    if val is not None:
                        idx = widget.findData(val)
                        if idx >= 0:
                            widget.setCurrentIndex(idx)
                        else:
                            idx_str = widget.findText(str(val))
                            if idx_str >= 0:
                                widget.setCurrentIndex(idx_str)
                            else:
                                widget.setEditText(str(val))
                    else:
                        widget.setCurrentIndex(-1)
                        widget.setEditText("")
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(bool(val))
                elif isinstance(widget, QtWidgets.QPlainTextEdit):
                    widget.setPlainText(str(val or ""))
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(int(val) if val is not None and val != "" else 0)
                elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.setValue(float(val) if val is not None and val != "" else 0.0)
                elif isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(str(val or ""))
        finally:
            self._loading = False

    def get_data(self) -> dict[str, Any]:
        """Collect form input data into a record dict."""
        data = {}
        for field in self.schema:
            name = field["name"]
            ftype = field.get("type", "str")
            widget = self.widgets.get(name)
            if not widget:
                continue

            val: Any = None
            if isinstance(widget, QtWidgets.QComboBox):
                val = widget.currentData()
                if val is None:
                    val = widget.currentText().strip() or None
            elif isinstance(widget, QtWidgets.QCheckBox):
                val = 1 if widget.isChecked() else 0
            elif isinstance(widget, QtWidgets.QPlainTextEdit):
                val = widget.toPlainText().strip() or None
            elif isinstance(widget, QtWidgets.QSpinBox):
                val = widget.value()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                val = widget.value()
            elif isinstance(widget, QtWidgets.QLineEdit):
                val = widget.text().strip() or None

            if name == "laser_wavelengths":
                import json
                try:
                    val = json.loads(val) if val else []
                except Exception:
                    val = []
            elif name == "detector_channels":
                import json
                try:
                    val = json.loads(val) if val else {}
                except Exception:
                    val = {}

            data[name] = val
        return data
