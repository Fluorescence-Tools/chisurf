"""GUI plugin for managing curated and user fluorescence sample databases."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QUrl

try:
    from qtpy import sip
except ImportError:
    try:
        import sip
    except ImportError:
        sip = None

from chisurf.core.fio.mmcif.db import FluorophoreDatabase, resolve_database_path
from chisurf.gui.misc_helpers import get_plugin_settings_path, persist_plugin_state
from chisurf.gui.widgets.dock_area import DockArea

from .client import SampleDatabaseClient


@persist_plugin_state("sample_database")
class SampleDatabaseWidget(QtWidgets.QMainWindow):
    """Window for browsing, editing, importing, and exporting samples."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.client = SampleDatabaseClient()
        self._loading = False
        self.setWindowTitle("Sample Database")
        self.resize(1100, 760)
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_status_bar()
        self.refresh()

    def _is_deleted(self) -> bool:
        """Return True when the underlying Qt/C++ object has been deleted."""
        return sip is not None and sip.isdeleted(self)

    @staticmethod
    def _is_widget_deleted(widget: QtWidgets.QWidget | None) -> bool:
        """Return True when a Qt widget's underlying C++ object is deleted."""
        return widget is not None and sip is not None and sip.isdeleted(widget)

    def _disconnect_signal(self, signal: Any) -> None:
        """Disconnect a Qt signal when present."""
        try:
            signal.disconnect()
        except Exception:
            pass

    def setup_ui(self) -> None:
        self._central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self._central_widget)
        layout = QtWidgets.QVBoxLayout(self._central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QtWidgets.QLabel("<h2>Fluorescence Sample Database</h2>")
        header.setContentsMargins(4, 4, 4, 0)
        layout.addWidget(header)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(4)
        layout.addWidget(self.splitter, stretch=1)

        self.left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        self.sample_table = QtWidgets.QTableWidget(0, 4)
        self.sample_table.setHorizontalHeaderLabels(["sample id", "uuid", "description", "probes"])
        self.sample_table.horizontalHeader().setStretchLastSection(True)
        self.sample_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sample_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.sample_table.itemSelectionChanged.connect(self.on_sample_selected)
        left_layout.addWidget(self.sample_table)
        self.splitter.addWidget(self.left_widget)

        self.tabs = DockArea(self)
        self.tabs.setNewTabButtonVisible(False)
        self.tabs.addTab(self.sample_tab(), "Sample")
        self.tabs.addTab(self.condition_tab(), "Condition")
        self.tabs.addTab(self.entities_tab(), "Entities")
        self.tabs.addTab(self.probes_tab(), "Probes")
        self.tabs.addTab(self.positions_tab(), "Label positions")
        self.tabs.addTab(self.metadata_tab(), "Metadata")
        self.tabs.addTab(self.users_tab(), "Users")
        self.tabs.addTab(self.devices_tab(), "Devices")
        self.tabs.addTab(self.experiment_types_tab(), "Experiment types")
        self.tabs.addTab(self.experiments_tab(), "Experiments")
        self.tabs.addTab(self.import_export_tab(), "Import/Export")
        self.splitter.addWidget(self.tabs)
        self.splitter.setSizes([360, 740])
        self.tabs.layoutChanged.connect(self._save_dock_layout)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)

    def setup_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("&Import...", self.import_file)
        file_menu.addAction("&Export selected sample...", self.export_selected_sample)
        file_menu.addAction("&Backup database...", self.backup_database)
        file_menu.addAction("Reset database from source...", self.reset_from_source)
        file_menu.addSeparator()
        file_menu.addAction("&Close", self.close)

        settings_menu = self.menuBar().addMenu("&Settings")
        settings_menu.addAction("&Reset window layout", self.reset_window_layout)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("&About Sample Database", self.show_about)

    def setup_toolbar(self) -> None:
        toolbar = self.addToolBar("Sample Database")
        toolbar.setObjectName("sampleDatabaseToolBar")
        toolbar.addAction("New", self.new_sample)
        toolbar.addAction("Import", self.import_file)
        toolbar.addAction("Delete", self.delete_sample)
        toolbar.addAction("Backup", self.backup_database)
        toolbar.addAction("Reset from source", self.reset_from_source)

    def setup_status_bar(self) -> None:
        self.statusBar().addPermanentWidget(self.status_label, stretch=1)
        self.statusBar().showMessage("Ready")

    def reset_window_layout(self) -> None:
        settings = self._dock_settings()
        settings.remove("dock_layout")
        settings.remove("geometry")
        settings.remove("state")
        self.resize(1100, 760)
        self.splitter.setSizes([360, 740])
        self._restore_dock_layout()

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About Sample Database",
            "Fluorescence Sample Database\n\nBrowse, edit, import, and export fluorescence sample metadata.",
        )

    def sample_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.sample_id_edit = QtWidgets.QLineEdit()
        self.uuid_edit = QtWidgets.QLineEdit()
        self.description_edit = QtWidgets.QLineEdit()
        self.details_edit = QtWidgets.QPlainTextEdit()
        self.details_edit.setMaximumHeight(90)
        self.num_probes_spin = QtWidgets.QSpinBox()
        self.num_probes_spin.setRange(0, 1000)
        self.solvent_edit = QtWidgets.QComboBox()
        self.solvent_edit.addItems(["liquid", "vitrified", "other"])
        self.condition_id_edit = QtWidgets.QLineEdit()
        self.assembly_id_edit = QtWidgets.QLineEdit()
        self.project_edit = QtWidgets.QLineEdit()
        self.measured_by_combo = QtWidgets.QComboBox()
        self.measured_device_combo = QtWidgets.QComboBox()
        self.measured_at_edit = QtWidgets.QLineEdit()
        save_button = QtWidgets.QPushButton("Save sample")
        save_button.clicked.connect(self.save_sample)
        layout.addRow("Sample id", self.sample_id_edit)
        layout.addRow("UUID", self.uuid_edit)
        layout.addRow("Description", self.description_edit)
        layout.addRow("Details", self.details_edit)
        layout.addRow("Number of probes", self.num_probes_spin)
        layout.addRow("Solvent phase", self.solvent_edit)
        layout.addRow("Condition id", self.condition_id_edit)
        layout.addRow("Entity assembly id", self.assembly_id_edit)
        layout.addRow("Project id", self.project_edit)
        layout.addRow("Measured by", self.measured_by_combo)
        layout.addRow("Device", self.measured_device_combo)
        layout.addRow("Measured at", self.measured_at_edit)
        layout.addRow("", save_button)
        return widget

    def condition_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.condition_id_field = QtWidgets.QLineEdit()
        self.ph_spin = QtWidgets.QDoubleSpinBox()
        self.ph_spin.setRange(-1.0, 14.0)
        self.ph_spin.setSpecialValueText("auto")
        self.temperature_spin = QtWidgets.QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 400.0)
        self.temperature_spin.setSpecialValueText("auto")
        self.ionic_spin = QtWidgets.QDoubleSpinBox()
        self.ionic_spin.setRange(0.0, 10.0)
        self.ionic_spin.setSpecialValueText("auto")
        self.buffer_edit = QtWidgets.QLineEdit()
        self.condition_details_edit = QtWidgets.QPlainTextEdit()
        self.condition_details_edit.setMaximumHeight(100)
        layout.addRow("Condition id", self.condition_id_field)
        layout.addRow("pH", self.ph_spin)
        layout.addRow("Temperature [K]", self.temperature_spin)
        layout.addRow("Ionic strength [M]", self.ionic_spin)
        layout.addRow("Buffer", self.buffer_edit)
        layout.addRow("Details", self.condition_details_edit)
        return widget

    def entities_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.entities_table = QtWidgets.QTableWidget(0, 5)
        self.entities_table.setHorizontalHeaderLabels(
            ["entity id", "type", "description", "common name", "sequence"]
        )
        self.entities_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.entities_table)
        return widget

    def probes_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.probes_table = QtWidgets.QTableWidget(0, 8)
        self.probes_table.setHorizontalHeaderLabels(
            ["id", "name", "category", "origin", "link", "abs", "em", "QY"]
        )
        self.probes_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.probes_table)
        return widget

    def positions_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.positions_table = QtWidgets.QTableWidget(0, 9)
        self.positions_table.setHorizontalHeaderLabels(
            [
                "sample_probe_id",
                "sample",
                "probe_id",
                "probe",
                "entity",
                "chain",
                "residue",
                "type",
                "description",
            ]
        )
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.positions_table)
        return widget

    def import_export_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        file_row = QtWidgets.QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(2)
        self.file_edit = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self.browse_import_file)
        file_row.addWidget(self.file_edit)
        file_row.addWidget(browse_button)
        layout.addLayout(file_row)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        import_button = QtWidgets.QPushButton("Import file")
        export_button = QtWidgets.QPushButton("Export selected sample")
        export_table_button = QtWidgets.QPushButton("Export table CSV/XLSX")
        import_button.clicked.connect(self.import_file)
        export_button.clicked.connect(self.export_selected_sample)
        export_table_button.clicked.connect(self.export_table)
        buttons.addWidget(import_button)
        buttons.addWidget(export_button)
        buttons.addWidget(export_table_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.preview_edit = QtWidgets.QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        layout.addWidget(self.preview_edit)
        return widget

    def metadata_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.metadata_table = QtWidgets.QTableWidget(0, 3)
        self.metadata_table.setHorizontalHeaderLabels(["key", "value", "details"])
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.metadata_table)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        add_button = QtWidgets.QPushButton("Add")
        delete_button = QtWidgets.QPushButton("Delete")
        add_button.clicked.connect(self.add_metadata_row)
        delete_button.clicked.connect(self.delete_metadata_row)
        buttons.addWidget(add_button)
        buttons.addWidget(delete_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def users_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.users_table = QtWidgets.QTableWidget(0, 5)
        self.users_table.setHorizontalHeaderLabels(
            ["id", "display name", "email", "affiliation", "details"]
        )
        self.users_table.horizontalHeader().setStretchLastSection(True)
        self.users_table.itemSelectionChanged.connect(self.load_user)
        layout.addWidget(self.users_table)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.user_id_edit = QtWidgets.QLineEdit()
        self.user_display_edit = QtWidgets.QLineEdit()
        self.user_email_edit = QtWidgets.QLineEdit()
        self.user_affiliation_edit = QtWidgets.QLineEdit()
        self.user_details_edit = QtWidgets.QPlainTextEdit()
        self.user_details_edit.setMaximumHeight(70)
        form.addRow("User id", self.user_id_edit)
        form.addRow("Display name", self.user_display_edit)
        form.addRow("Email", self.user_email_edit)
        form.addRow("Affiliation", self.user_affiliation_edit)
        form.addRow("Details", self.user_details_edit)
        layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_user_button = QtWidgets.QPushButton("Save user")
        delete_user_button = QtWidgets.QPushButton("Delete user")
        save_user_button.clicked.connect(self.save_user)
        delete_user_button.clicked.connect(self.delete_user)
        buttons.addWidget(save_user_button)
        buttons.addWidget(delete_user_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def devices_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.devices_table = QtWidgets.QTableWidget(0, 8)
        self.devices_table.setHorizontalHeaderLabels(
            ["id", "name", "type", "model", "serial", "location", "owner", "details"]
        )
        self.devices_table.horizontalHeader().setStretchLastSection(True)
        self.devices_table.itemSelectionChanged.connect(self.load_device)
        layout.addWidget(self.devices_table)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.device_id_edit = QtWidgets.QLineEdit()
        self.device_name_edit = QtWidgets.QLineEdit()
        self.device_type_edit = QtWidgets.QLineEdit()
        self.device_model_edit = QtWidgets.QLineEdit()
        self.device_serial_edit = QtWidgets.QLineEdit()
        self.device_location_edit = QtWidgets.QLineEdit()
        self.device_owner_edit = QtWidgets.QLineEdit()
        self.device_details_edit = QtWidgets.QPlainTextEdit()
        self.device_details_edit.setMaximumHeight(70)
        form.addRow("Device id", self.device_id_edit)
        form.addRow("Name", self.device_name_edit)
        form.addRow("Type", self.device_type_edit)
        form.addRow("Model", self.device_model_edit)
        form.addRow("Serial", self.device_serial_edit)
        form.addRow("Location", self.device_location_edit)
        form.addRow("Owner", self.device_owner_edit)
        form.addRow("Details", self.device_details_edit)
        layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_device_button = QtWidgets.QPushButton("Save device")
        delete_device_button = QtWidgets.QPushButton("Delete device")
        save_device_button.clicked.connect(self.save_device)
        delete_device_button.clicked.connect(self.delete_device)
        buttons.addWidget(save_device_button)
        buttons.addWidget(delete_device_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def experiment_types_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.experiment_types_table = QtWidgets.QTableWidget(0, 5)
        self.experiment_types_table.setHorizontalHeaderLabels(
            ["id", "name", "category", "description", "details"]
        )
        self.experiment_types_table.horizontalHeader().setStretchLastSection(True)
        self.experiment_types_table.itemSelectionChanged.connect(self.load_experiment_type)
        layout.addWidget(self.experiment_types_table)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.experiment_type_id_edit = QtWidgets.QLineEdit()
        self.experiment_type_name_edit = QtWidgets.QLineEdit()
        self.experiment_type_category_edit = QtWidgets.QLineEdit()
        self.experiment_type_description_edit = QtWidgets.QLineEdit()
        self.experiment_type_details_edit = QtWidgets.QPlainTextEdit()
        self.experiment_type_details_edit.setMaximumHeight(70)
        form.addRow("Type id", self.experiment_type_id_edit)
        form.addRow("Name", self.experiment_type_name_edit)
        form.addRow("Category", self.experiment_type_category_edit)
        form.addRow("Description", self.experiment_type_description_edit)
        form.addRow("Details", self.experiment_type_details_edit)
        layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_button = QtWidgets.QPushButton("Save experiment type")
        delete_button = QtWidgets.QPushButton("Delete experiment type")
        save_button.clicked.connect(self.save_experiment_type)
        delete_button.clicked.connect(self.delete_experiment_type)
        buttons.addWidget(save_button)
        buttons.addWidget(delete_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def experiments_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.experiments_table = QtWidgets.QTableWidget(0, 8)
        self.experiments_table.setHorizontalHeaderLabels(
            ["experiment id", "type", "sample", "project", "user", "device", "started", "status"]
        )
        self.experiments_table.horizontalHeader().setStretchLastSection(True)
        self.experiments_table.itemSelectionChanged.connect(self.load_experiment)
        layout.addWidget(self.experiments_table)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.experiment_id_edit = QtWidgets.QLineEdit()
        self.experiment_type_combo = QtWidgets.QComboBox()
        self.experiment_sample_combo = QtWidgets.QComboBox()
        self.experiment_project_edit = QtWidgets.QLineEdit()
        self.experiment_user_combo = QtWidgets.QComboBox()
        self.experiment_device_combo = QtWidgets.QComboBox()
        self.experiment_started_edit = QtWidgets.QLineEdit()
        self.experiment_ended_edit = QtWidgets.QLineEdit()
        self.experiment_status_edit = QtWidgets.QLineEdit()
        self.experiment_details_edit = QtWidgets.QPlainTextEdit()
        self.experiment_details_edit.setMaximumHeight(70)
        form.addRow("Experiment id", self.experiment_id_edit)
        form.addRow("Type", self.experiment_type_combo)
        form.addRow("Sample", self.experiment_sample_combo)
        form.addRow("Project", self.experiment_project_edit)
        form.addRow("User", self.experiment_user_combo)
        form.addRow("Device", self.experiment_device_combo)
        form.addRow("Started", self.experiment_started_edit)
        form.addRow("Ended", self.experiment_ended_edit)
        form.addRow("Status", self.experiment_status_edit)
        form.addRow("Details", self.experiment_details_edit)
        layout.addLayout(form)
        data_header = QtWidgets.QLabel("Experiment data / links")
        layout.addWidget(data_header)
        self.experiment_data_table = QtWidgets.QTableWidget(0, 8)
        self.experiment_data_table.setHorizontalHeaderLabels(
            ["id", "type", "mode", "path/url/folder", "mime", "checksum", "reading options", "details"]
        )
        self.experiment_data_table.horizontalHeader().setStretchLastSection(True)
        self.experiment_data_table.itemSelectionChanged.connect(self.load_experiment_data)
        layout.addWidget(self.experiment_data_table)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_experiment_button = QtWidgets.QPushButton("Save experiment")
        delete_experiment_button = QtWidgets.QPushButton("Delete experiment")
        add_data_button = QtWidgets.QPushButton("Add data")
        save_data_button = QtWidgets.QPushButton("Save data")
        delete_data_button = QtWidgets.QPushButton("Delete data")
        open_data_button = QtWidgets.QPushButton("Open linked data")
        save_experiment_button.clicked.connect(self.save_experiment)
        delete_experiment_button.clicked.connect(self.delete_experiment)
        add_data_button.clicked.connect(self.add_experiment_data_row)
        save_data_button.clicked.connect(self.save_experiment_data)
        delete_data_button.clicked.connect(self.delete_experiment_data)
        open_data_button.clicked.connect(self.open_experiment_data)
        buttons.addWidget(save_experiment_button)
        buttons.addWidget(delete_experiment_button)
        buttons.addWidget(add_data_button)
        buttons.addWidget(save_data_button)
        buttons.addWidget(delete_data_button)
        buttons.addWidget(open_data_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def refresh(self) -> None:
        self._loading = True
        try:
            status = self.client.status()
            status_text = (
                f"User DB: {status['user_database']} | schema {status['schema_version']} | "
                f"samples {status['sample_count']} | experiments {status.get('experiment_count', 0)}"
            )
            self.status_label.setText(status_text)
            self.sample_table.setRowCount(0)
            for row in self.client.list_samples():
                index = self.sample_table.rowCount()
                self.sample_table.insertRow(index)
                for column, key in enumerate(
                    [
                        "sample_id",
                        "sample_uuid",
                        "description",
                        "mapped_probe_count",
                    ]
                ):
                    self.sample_table.setItem(
                        index, column, QtWidgets.QTableWidgetItem(str(row.get(key, "")))
                    )
            self.fill_users()
            self.fill_devices()
            self.fill_user_table()
            self.fill_device_table()
            self.fill_experiment_types()
            self.fill_experiment_type_table()
            self.fill_experiment_table()
            self.fill_experiment_sample_combo()
            self.fill_experiment_user_combo()
            self.fill_experiment_device_combo()
            if self.sample_table.rowCount() > 0:
                self.sample_table.selectRow(0)
            else:
                self.clear_form()
        finally:
            self._loading = False

    def clear_form(self) -> None:
        for widget in (
            self.sample_id_edit,
            self.uuid_edit,
            self.description_edit,
            self.condition_id_edit,
            self.assembly_id_edit,
            self.project_edit,
            self.measured_at_edit,
            self.condition_id_field,
            self.buffer_edit,
            self.user_id_edit,
            self.user_display_edit,
            self.user_email_edit,
            self.user_affiliation_edit,
            self.device_id_edit,
            self.device_name_edit,
            self.device_type_edit,
            self.device_model_edit,
            self.device_serial_edit,
            self.device_location_edit,
            self.device_owner_edit,
            self.experiment_type_id_edit,
            self.experiment_type_name_edit,
            self.experiment_type_category_edit,
            self.experiment_type_description_edit,
            self.experiment_id_edit,
            self.experiment_project_edit,
            self.experiment_started_edit,
            self.experiment_ended_edit,
            self.experiment_status_edit,
        ):
            widget.clear()
        self.details_edit.clear()
        self.condition_details_edit.clear()
        self.user_details_edit.clear()
        self.device_details_edit.clear()
        self.experiment_type_details_edit.clear()
        self.experiment_details_edit.clear()
        self.measured_by_combo.setCurrentIndex(-1)
        self.measured_device_combo.setCurrentIndex(-1)
        self.num_probes_spin.setValue(0)
        self.solvent_edit.setCurrentText("liquid")
        self.entities_table.setRowCount(0)
        self.probes_table.setRowCount(0)
        self.positions_table.setRowCount(0)
        self.metadata_table.setRowCount(0)
        self.users_table.setRowCount(0)
        self.devices_table.setRowCount(0)
        self.experiment_types_table.setRowCount(0)
        self.experiments_table.setRowCount(0)
        self.experiment_data_table.setRowCount(0)
        self.preview_edit.clear()

    def on_sample_selected(self) -> None:
        if self._is_deleted():
            return
        if self._loading:
            return
        rows = self.sample_table.selectionModel().selectedRows()
        if not rows:
            return
        sample_id = self.sample_table.item(rows[0].row(), 0).text()
        self.load_sample(sample_id)

    def load_sample(self, sample_id: str) -> None:
        if self._is_deleted():
            return
        sample = self.client.get_sample(sample_id) or {}
        self._loading = True
        try:
            self.sample_id_edit.setText(sample.get("sample_id", ""))
            self.uuid_edit.setText(sample.get("sample_uuid", ""))
            self.description_edit.setText(sample.get("description", ""))
            self.details_edit.setPlainText(sample.get("details", ""))
            self.num_probes_spin.setValue(int(sample.get("num_of_probes") or 0))
            self.solvent_edit.setCurrentText(sample.get("solvent_phase") or "liquid")
            self.condition_id_edit.setText(sample.get("sample_condition_id", ""))
            self.assembly_id_edit.setText(sample.get("entity_assembly_id", ""))
            self.project_edit.setText(sample.get("project_id", ""))
            self.measured_at_edit.setText(sample.get("measured_at", ""))
            self.condition_id_field.setText((sample.get("condition") or {}).get("condition_id", ""))
            condition = sample.get("condition") or {}
            self.ph_spin.setValue(float(condition.get("ph") or 0))
            self.temperature_spin.setValue(float(condition.get("temperature") or 0))
            self.ionic_spin.setValue(float(condition.get("ionic_strength") or 0))
            self.buffer_edit.setText(condition.get("buffer_composition", ""))
            self.condition_details_edit.setPlainText(condition.get("details", ""))
            self.fill_entities(sample.get("entities", []))
            self.fill_probes()
            self.fill_positions(sample.get("sample_probes", []))
            self.fill_metadata(sample.get("key_values", []))
            self.fill_experiment_table(sample_id=sample_id)
            self.measured_by_combo.setCurrentText("")
            self.measured_by_combo.setCurrentIndex(
                self.measured_by_combo.findData(sample.get("measured_by_user_id") or "")
            )
            self.measured_device_combo.setCurrentText("")
            self.measured_device_combo.setCurrentIndex(
                self.measured_device_combo.findData(sample.get("measured_by_device_id") or "")
            )
        finally:
            self._loading = False

    def fill_users(self) -> None:
        self.measured_by_combo.blockSignals(True)
        self.measured_by_combo.clear()
        self.measured_by_combo.addItem("", "")
        for user in self.client.list_users():
            user_id = user.get("user_id", "")
            self.measured_by_combo.addItem(
                f"{user.get('display_name') or user_id} ({user_id})",
                user_id,
            )
        self.measured_by_combo.blockSignals(False)

    def fill_devices(self) -> None:
        self.measured_device_combo.blockSignals(True)
        self.measured_device_combo.clear()
        self.measured_device_combo.addItem("", "")
        for device in self.client.list_devices():
            device_id = device.get("device_id", "")
            self.measured_device_combo.addItem(
                f"{device.get('name') or device_id} ({device_id})",
                device_id,
            )
        self.measured_device_combo.blockSignals(False)

    def fill_entities(self, entities: list[dict[str, Any]]) -> None:
        self.entities_table.setRowCount(0)
        for entity in entities:
            row = self.entities_table.rowCount()
            self.entities_table.insertRow(row)
            sequence = (
                self.client.get_sample(self.sample_id_edit.text()).get("sequence", "")
                if False
                else ""
            )
            values = [
                entity.get("entity_id", ""),
                entity.get("type", ""),
                entity.get("description", ""),
                entity.get("common_name", ""),
                sequence,
            ]
            for column, value in enumerate(values):
                self.entities_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_probes(self) -> None:
        self.probes_table.setRowCount(0)
        with FluorophoreDatabase(resolve_database_path()) as db:
            for row in db.get_probes():
                index = self.probes_table.rowCount()
                self.probes_table.insertRow(index)
                props = db.get_standardized_optical_properties(int(row["probe_id"]))
                values = [
                    row["probe_id"],
                    row["chromophore_name"],
                    row["category"],
                    row["probe_origin"],
                    row["probe_link_type"],
                    props.get("abs_max", ""),
                    props.get("em_max", ""),
                    props.get("qy", ""),
                ]
                for column, value in enumerate(values):
                    self.probes_table.setItem(
                        index, column, QtWidgets.QTableWidgetItem(str(value or ""))
                    )

    def fill_positions(self, mappings: list[dict[str, Any]]) -> None:
        self.positions_table.setRowCount(0)
        for mapping in mappings:
            row = self.positions_table.rowCount()
            self.positions_table.insertRow(row)
            values = [
                mapping.get("sample_probe_id", ""),
                mapping.get("sample_id", ""),
                mapping.get("probe_id", ""),
                mapping.get("chromophore_name", ""),
                mapping.get("entity_id", ""),
                mapping.get("asym_id", ""),
                mapping.get("residue_number", ""),
                mapping.get("fluorophore_type", ""),
                mapping.get("description", "") or mapping.get("position_description", ""),
            ]
            for column, value in enumerate(values):
                self.positions_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_metadata(self, key_values: list[dict[str, Any]]) -> None:
        self.metadata_table.setRowCount(0)
        for item in key_values:
            self.metadata_table.insertRow(self.metadata_table.rowCount())
            for column, key in enumerate(["key", "value", "details"]):
                self.metadata_table.setItem(
                    self.metadata_table.rowCount() - 1,
                    column,
                    QtWidgets.QTableWidgetItem(str(item.get(key, ""))),
                )

    def collect_sample(self) -> dict[str, Any]:
        entities = []
        for row in range(self.entities_table.rowCount()):
            entities.append(
                {
                    "entity_id": self.entities_table.item(row, 0).text()
                    if self.entities_table.item(row, 0)
                    else "",
                    "type": self.entities_table.item(row, 1).text()
                    if self.entities_table.item(row, 1)
                    else "polymer",
                    "description": self.entities_table.item(row, 2).text()
                    if self.entities_table.item(row, 2)
                    else "",
                    "common_name": self.entities_table.item(row, 3).text()
                    if self.entities_table.item(row, 3)
                    else "",
                }
            )
        mappings = []
        for row in range(self.positions_table.rowCount()):
            mappings.append(
                {
                    "sample_probe_id": self.positions_table.item(row, 0).text()
                    if self.positions_table.item(row, 0)
                    else None,
                    "probe_id": self.positions_table.item(row, 2).text()
                    if self.positions_table.item(row, 2)
                    else None,
                    "fluorophore_type": self.positions_table.item(row, 7).text()
                    if self.positions_table.item(row, 7)
                    else "unspecified",
                    "description": self.positions_table.item(row, 8).text()
                    if self.positions_table.item(row, 8)
                    else "",
                }
            )
        key_values = []
        for row in range(self.metadata_table.rowCount()):
            key = (
                self.metadata_table.item(row, 0).text() if self.metadata_table.item(row, 0) else ""
            ).strip()
            if not key:
                continue
            key_values.append(
                {
                    "key": key,
                    "value": (
                        self.metadata_table.item(row, 1).text()
                        if self.metadata_table.item(row, 1)
                        else ""
                    ),
                    "details": (
                        self.metadata_table.item(row, 2).text()
                        if self.metadata_table.item(row, 2)
                        else ""
                    ),
                }
            )
        return {
            "sample_id": self.sample_id_edit.text().strip(),
            "sample_uuid": self.uuid_edit.text().strip(),
            "description": self.description_edit.text().strip(),
            "details": self.details_edit.toPlainText().strip(),
            "num_of_probes": self.num_probes_spin.value(),
            "solvent_phase": self.solvent_edit.currentText(),
            "sample_condition_id": self.condition_id_edit.text().strip(),
            "entity_assembly_id": self.assembly_id_edit.text().strip(),
            "project_id": self.project_edit.text().strip(),
            "measured_by_user_id": self.measured_by_combo.currentData() or None,
            "measured_by_device_id": self.measured_device_combo.currentData() or None,
            "measured_at": self.measured_at_edit.text().strip(),
            "condition": {
                "condition_id": self.condition_id_field.text().strip(),
                "ph": None if self.ph_spin.value() == 0 else self.ph_spin.value(),
                "temperature": None
                if self.temperature_spin.value() == 0
                else self.temperature_spin.value(),
                "ionic_strength": None if self.ionic_spin.value() == 0 else self.ionic_spin.value(),
                "buffer_composition": self.buffer_edit.text().strip(),
                "details": self.condition_details_edit.toPlainText().strip(),
            },
            "entities": entities,
            "sample_probes": mappings,
            "key_values": key_values,
        }

    def save_sample(self) -> None:
        sample = self.collect_sample()
        if not sample["sample_id"]:
            self.status_label.setText("Sample id is required")
            return
        saved = self.client.save_sample(sample)
        self.status_label.setText(f"Saved {saved['sample_id']}")
        self.refresh()

    def add_metadata_row(self) -> None:
        row = self.metadata_table.rowCount()
        self.metadata_table.insertRow(row)
        self.metadata_table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self.metadata_table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))
        self.metadata_table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))

    def delete_metadata_row(self) -> None:
        row = self.metadata_table.currentRow()
        if row >= 0:
            self.metadata_table.removeRow(row)

    def collect_user(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id_edit.text().strip(),
            "display_name": self.user_display_edit.text().strip(),
            "email": self.user_email_edit.text().strip() or None,
            "affiliation": self.user_affiliation_edit.text().strip() or None,
            "details": self.user_details_edit.toPlainText().strip() or None,
        }

    def save_user(self) -> None:
        users = self.client.save_user(self.collect_user())
        self.fill_user_table(users)
        self.refresh()

    def delete_user(self) -> None:
        user_id = self.user_id_edit.text().strip()
        if not user_id:
            return
        users = self.client.delete_user(user_id)
        self.fill_user_table(users)
        self.refresh()

    def load_user(self) -> None:
        rows = self.users_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        self.user_id_edit.setText(self.users_table.item(row, 0).text() or "")
        self.user_display_edit.setText(self.users_table.item(row, 1).text() or "")
        self.user_email_edit.setText(self.users_table.item(row, 2).text() or "")
        self.user_affiliation_edit.setText(self.users_table.item(row, 3).text() or "")
        self.user_details_edit.setPlainText(self.users_table.item(row, 4).text() or "")

    def fill_user_table(self, users: list[dict[str, Any]] | None = None) -> None:
        users = users if users is not None else self.client.list_users()
        self.users_table.setRowCount(0)
        for user in users:
            row = self.users_table.rowCount()
            self.users_table.insertRow(row)
            values = [
                user.get("user_id", ""),
                user.get("display_name", ""),
                user.get("email", ""),
                user.get("affiliation", ""),
                user.get("details", ""),
            ]
            for column, value in enumerate(values):
                self.users_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))

    def collect_device(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id_edit.text().strip(),
            "name": self.device_name_edit.text().strip(),
            "device_type": self.device_type_edit.text().strip() or None,
            "model": self.device_model_edit.text().strip() or None,
            "serial_number": self.device_serial_edit.text().strip() or None,
            "location": self.device_location_edit.text().strip() or None,
            "owner": self.device_owner_edit.text().strip() or None,
            "details": self.device_details_edit.toPlainText().strip() or None,
        }

    def save_device(self) -> None:
        devices = self.client.save_device(self.collect_device())
        self.fill_device_table(devices)
        self.refresh()

    def delete_device(self) -> None:
        device_id = self.device_id_edit.text().strip()
        if not device_id:
            return
        devices = self.client.delete_device(device_id)
        self.fill_device_table(devices)
        self.refresh()

    def load_device(self) -> None:
        rows = self.devices_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        self.device_id_edit.setText(self.devices_table.item(row, 0).text() or "")
        self.device_name_edit.setText(self.devices_table.item(row, 1).text() or "")
        self.device_type_edit.setText(self.devices_table.item(row, 2).text() or "")
        self.device_model_edit.setText(self.devices_table.item(row, 3).text() or "")
        self.device_serial_edit.setText(self.devices_table.item(row, 4).text() or "")
        self.device_location_edit.setText(self.devices_table.item(row, 5).text() or "")
        self.device_owner_edit.setText(self.devices_table.item(row, 6).text() or "")
        self.device_details_edit.setPlainText(self.devices_table.item(row, 7).text() or "")

    def collect_experiment_type(self) -> dict[str, Any]:
        return {
            "type_id": self.experiment_type_id_edit.text().strip() or None,
            "name": self.experiment_type_name_edit.text().strip(),
            "category": self.experiment_type_category_edit.text().strip() or None,
            "description": self.experiment_type_description_edit.text().strip() or None,
            "details": self.experiment_type_details_edit.toPlainText().strip() or None,
        }

    def save_experiment_type(self) -> None:
        types = self.client.save_experiment_type(self.collect_experiment_type())
        self.fill_experiment_type_table(types)
        self.refresh()

    def delete_experiment_type(self) -> None:
        type_id = self.experiment_type_id_edit.text().strip()
        if not type_id:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete experiment type", f"Delete experiment type {type_id}?"
        )
        if answer == QtWidgets.QMessageBox.Yes:
            types = self.client.delete_experiment_type(int(type_id))
            self.fill_experiment_type_table(types)
            self.refresh()

    def collect_experiment(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id_edit.text().strip(),
            "type_id": int(self.experiment_type_combo.currentData() or -1)
            if self.experiment_type_combo.currentData() not in (None, -1, "")
            else None,
            "sample_id": self.experiment_sample_combo.currentData() or None,
            "project_id": self.experiment_project_edit.text().strip() or None,
            "measured_by_user_id": self.experiment_user_combo.currentData() or None,
            "measured_by_device_id": self.experiment_device_combo.currentData() or None,
            "started_at": self.experiment_started_edit.text().strip() or None,
            "ended_at": self.experiment_ended_edit.text().strip() or None,
            "status": self.experiment_status_edit.text().strip() or None,
            "details": self.experiment_details_edit.toPlainText().strip() or None,
        }

    def save_experiment(self) -> None:
        experiment = self.collect_experiment()
        if not experiment["experiment_id"]:
            self.status_label.setText("Experiment id is required")
            return
        saved = self.client.save_experiment(experiment)
        self.status_label.setText(f"Saved {saved['experiment_id']}")
        self.refresh()

    def delete_experiment(self) -> None:
        experiment_id = self.experiment_id_edit.text().strip()
        if not experiment_id:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete experiment", f"Delete experiment {experiment_id}?"
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.client.delete_experiment(experiment_id)
            self.refresh()

    def add_experiment_data_row(self) -> None:
        row = self.experiment_data_table.rowCount()
        self.experiment_data_table.insertRow(row)
        for column in range(self.experiment_data_table.columnCount()):
            self.experiment_data_table.setItem(row, column, QtWidgets.QTableWidgetItem(""))
        self.experiment_data_table.setCurrentCell(row, 0)

    def collect_experiment_data(self) -> dict[str, Any]:
        row = self.experiment_data_table.currentRow()
        data_id = self.experiment_data_table.item(row, 0).text() if row >= 0 else ""
        location = self.experiment_data_table.item(row, 3).text() if row >= 0 else ""
        storage_mode = (
            self.experiment_data_table.item(row, 2).text() if row >= 0 else "link"
        ) or "link"
        file_path = location if storage_mode == "link" and not location.startswith(("http://", "https://", "file://")) else None
        url = location if location.startswith(("http://", "https://", "file://")) else None
        folder_path = location if storage_mode == "folder" else None
        return {
            "data_id": int(data_id) if data_id else None,
            "experiment_id": self.experiment_id_edit.text().strip(),
            "data_type": self.experiment_data_table.item(row, 1).text() if row >= 0 else "",
            "storage_mode": storage_mode,
            "file_path": file_path,
            "url": url,
            "folder_path": folder_path,
            "mime_type": self.experiment_data_table.item(row, 4).text() if row >= 0 else "",
            "checksum": self.experiment_data_table.item(row, 5).text() if row >= 0 else "",
            "reading_options_json": (
                self.experiment_data_table.item(row, 6).text() if row >= 0 else ""
            ),
            "details": self.experiment_data_table.item(row, 7).text() if row >= 0 else "",
        }

    def save_experiment_data(self) -> None:
        data = self.collect_experiment_data()
        if not data["experiment_id"]:
            self.status_label.setText("Select or create an experiment first")
            return
        if not data["data_type"]:
            self.status_label.setText("Data type is required")
            return
        try:
            json.loads(data["reading_options_json"] or "{}")
        except json.JSONDecodeError as exc:
            self.status_label.setText(f"Invalid reading options JSON: {exc}")
            return
        saved = self.client.save_experiment_data(data)
        self.fill_experiment_data_table(saved.get("data", []))
        self.fill_experiment_table()
        self.refresh()

    def delete_experiment_data(self) -> None:
        row = self.experiment_data_table.currentRow()
        if row < 0:
            return
        data_id = self.experiment_data_table.item(row, 0).text()
        if not data_id:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete experiment data", f"Delete data record {data_id}?"
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.client.delete_experiment_data(int(data_id))
            self.refresh()

    def open_experiment_data(self) -> None:
        row = self.experiment_data_table.currentRow()
        if row < 0:
            return
        location = self.experiment_data_table.item(row, 3).text() or ""
        if not location:
            QtWidgets.QMessageBox.information(self, "Experiment data", "No data location is set")
            return
        if location.startswith(("http://", "https://", "file://")):
            QtGui.QDesktopServices.openUrl(QUrl(location))
            return
        path = QtCore.QFileInfo(location).absoluteFilePath()
        if not QtCore.QFile(path).exists():
            QtWidgets.QMessageBox.warning(self, "Data path not found", f"Missing: {path}")
            return
        QtGui.QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def fill_device_table(self, devices: list[dict[str, Any]] | None = None) -> None:
        devices = devices if devices is not None else self.client.list_devices()
        self.devices_table.setRowCount(0)
        for device in devices:
            row = self.devices_table.rowCount()
            self.devices_table.insertRow(row)
            values = [
                device.get("device_id", ""),
                device.get("name", ""),
                device.get("device_type", ""),
                device.get("model", ""),
                device.get("serial_number", ""),
                device.get("location", ""),
                device.get("owner", ""),
                device.get("details", ""),
            ]
            for column, value in enumerate(values):
                self.devices_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_experiment_types(self) -> None:
        self.experiment_type_combo.blockSignals(True)
        self.experiment_type_combo.clear()
        self.experiment_type_combo.addItem("", -1)
        for item in self.client.list_experiment_types():
            type_id = int(item.get("type_id") or -1)
            name = item.get("name") or item.get("type_id") or ""
            self.experiment_type_combo.addItem(str(name), type_id)
        self.experiment_type_combo.blockSignals(False)

    def fill_experiment_type_table(self) -> None:
        self.experiment_types_table.setRowCount(0)
        for item in self.client.list_experiment_types():
            row = self.experiment_types_table.rowCount()
            self.experiment_types_table.insertRow(row)
            values = [
                item.get("type_id", ""),
                item.get("name", ""),
                item.get("category", ""),
                item.get("description", ""),
                item.get("details", ""),
            ]
            for column, value in enumerate(values):
                self.experiment_types_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_experiment_table(self, sample_id: str | None = None) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.experiments_table):
            return
        self.experiments_table.setRowCount(0)
        for item in self.client.list_experiments(sample_id=sample_id):
            row = self.experiments_table.rowCount()
            self.experiments_table.insertRow(row)
            values = [
                item.get("experiment_id", ""),
                item.get("experiment_type", ""),
                item.get("sample_id", ""),
                item.get("project_id", ""),
                item.get("measured_by_user", ""),
                item.get("measured_by_device", ""),
                item.get("started_at", ""),
                item.get("status", ""),
            ]
            for column, value in enumerate(values):
                self.experiments_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_experiment_sample_combo(self) -> None:
        self.experiment_sample_combo.blockSignals(True)
        self.experiment_sample_combo.clear()
        self.experiment_sample_combo.addItem("", "")
        for sample in self.client.list_samples():
            sample_id = sample.get("sample_id", "")
            description = sample.get("description") or sample_id
            self.experiment_sample_combo.addItem(f"{description} ({sample_id})", sample_id)
        self.experiment_sample_combo.blockSignals(False)

    def fill_experiment_user_combo(self) -> None:
        self.experiment_user_combo.blockSignals(True)
        self.experiment_user_combo.clear()
        self.experiment_user_combo.addItem("", "")
        for user in self.client.list_users():
            user_id = user.get("user_id", "")
            self.experiment_user_combo.addItem(
                f"{user.get('display_name') or user_id} ({user_id})",
                user_id,
            )
        self.experiment_user_combo.blockSignals(False)

    def fill_experiment_device_combo(self) -> None:
        self.experiment_device_combo.blockSignals(True)
        self.experiment_device_combo.clear()
        self.experiment_device_combo.addItem("", "")
        for device in self.client.list_devices():
            device_id = device.get("device_id", "")
            self.experiment_device_combo.addItem(
                f"{device.get('name') or device_id} ({device_id})",
                device_id,
            )
        self.experiment_device_combo.blockSignals(False)

    def load_experiment_type(self) -> None:
        rows = self.experiment_types_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        self.experiment_type_id_edit.setText(self.experiment_types_table.item(row, 0).text() or "")
        self.experiment_type_name_edit.setText(self.experiment_types_table.item(row, 1).text() or "")
        self.experiment_type_category_edit.setText(
            self.experiment_types_table.item(row, 2).text() or ""
        )
        self.experiment_type_description_edit.setText(
            self.experiment_types_table.item(row, 3).text() or ""
        )
        self.experiment_type_details_edit.setPlainText(
            self.experiment_types_table.item(row, 4).text() or ""
        )

    def load_experiment(self) -> None:
        rows = self.experiments_table.selectionModel().selectedRows()
        if not rows:
            return
        experiment_id = self.experiments_table.item(rows[0].row(), 0).text()
        experiment = self.client.get_experiment(experiment_id) or {}
        self._loading = True
        try:
            self.experiment_id_edit.setText(experiment.get("experiment_id", ""))
            self._set_combo_data(self.experiment_type_combo, experiment.get("type_id"))
            self._set_combo_data(self.experiment_sample_combo, experiment.get("sample_id"))
            self.experiment_project_edit.setText(experiment.get("project_id", ""))
            self._set_combo_data(self.experiment_user_combo, experiment.get("measured_by_user_id"))
            self._set_combo_data(self.experiment_device_combo, experiment.get("measured_by_device_id"))
            self.experiment_started_edit.setText(experiment.get("started_at", ""))
            self.experiment_ended_edit.setText(experiment.get("ended_at", ""))
            self.experiment_status_edit.setText(experiment.get("status", ""))
            self.experiment_details_edit.setPlainText(experiment.get("details", ""))
            self.fill_experiment_data_table(experiment.get("data", []))
        finally:
            self._loading = False

    def fill_experiment_data_table(self, data_rows: list[dict[str, Any]]) -> None:
        self.experiment_data_table.setRowCount(0)
        for item in data_rows:
            row = self.experiment_data_table.rowCount()
            self.experiment_data_table.insertRow(row)
            location = item.get("file_path") or item.get("url") or item.get("folder_path") or ""
            values = [
                item.get("data_id", ""),
                item.get("data_type", ""),
                item.get("storage_mode", ""),
                location,
                item.get("mime_type", ""),
                item.get("checksum", ""),
                item.get("reading_options_json", ""),
                item.get("details", ""),
            ]
            for column, value in enumerate(values):
                self.experiment_data_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def load_experiment_data(self) -> None:
        row = self.experiment_data_table.currentRow()
        if row < 0:
            return
        self.status_label.setText(
            self.experiment_data_table.item(row, 6).text()
            or self.experiment_data_table.item(row, 7).text()
            or ""
        )

    def _set_combo_data(self, combo: QtWidgets.QComboBox, data: Any) -> None:
        combo.setCurrentText("")
        combo.setCurrentIndex(combo.findData(data))

    def export_table(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export sample table",
            "samples.csv",
            "CSV/Excel files (*.csv *.tsv *.xlsx);;All files (*)",
        )
        if not path:
            return
        result = self.client.export_table(path)
        self.status_label.setText(f"Exported {result.get('output_path')}")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._restore_dock_layout()

    def closeEvent(self, event) -> None:
        self._save_dock_layout()
        super().closeEvent(event)

    def _dock_settings(self):
        return QtCore.QSettings(
            str(get_plugin_settings_path("sample_database")),
            QtCore.QSettings.IniFormat,
        )

    def _save_dock_layout(self) -> None:
        if not hasattr(self, "tabs"):
            return
        settings = self._dock_settings()
        settings.setValue("dock_layout", json.dumps(self.tabs.get_layout_state()))

    def _restore_dock_layout(self) -> None:
        if not hasattr(self, "tabs"):
            return
        settings = self._dock_settings()
        state = json.loads(settings.value("dock_layout", "") or "{}")
        if state:
            self.tabs.set_layout_state(state)

    def new_sample(self) -> None:
        sample_id = f"sample_{datetime.now():%Y%m%d_%H%M%S}"
        self.sample_id_edit.setText(sample_id)
        self.uuid_edit.setText("")
        self.description_edit.setText("")
        self.details_edit.clear()
        self.project_edit.clear()
        self.measured_by_combo.setCurrentIndex(-1)
        self.measured_device_combo.setCurrentIndex(-1)
        self.measured_at_edit.setText(datetime.now().isoformat(timespec="seconds"))
        self.condition_id_field.setText(f"condition_{sample_id}")
        self.entities_table.setRowCount(0)
        self.positions_table.setRowCount(0)
        self.metadata_table.setRowCount(0)

    def delete_sample(self) -> None:
        sample_id = self.sample_id_edit.text().strip()
        if not sample_id:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete sample", f"Delete {sample_id}?")
        if answer == QtWidgets.QMessageBox.Yes:
            self.client.delete_sample(sample_id)
            self.refresh()

    def browse_import_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import PDBx/PDB-IHM/FLR CIF",
            "",
            "CIF/mmCIF files (*.cif *.mmcif);;All files (*)",
        )
        if path:
            self.file_edit.setText(path)

    def import_file(self) -> None:
        path = self.file_edit.text().strip()
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Import PDBx/PDB-IHM/FLR CIF",
                "",
                "CIF/mmCIF files (*.cif *.mmcif);;All files (*)",
            )
        if not path:
            return
        summary = self.client.import_file(path)
        self.preview_edit.setPlainText(str(summary))
        self.refresh()

    def export_selected_sample(self) -> None:
        sample_id = self.sample_id_edit.text().strip()
        if not sample_id:
            self.status_label.setText("Select a sample to export")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export FLR CIF", f"{sample_id}.cif", "CIF files (*.cif *.mmcif);;All files (*)"
        )
        if not path:
            return
        result = self.client.export_sample(sample_id, output_path=path)
        self.status_label.setText(f"Exported {result.get('output_path')}")

    def backup_database(self) -> None:
        path = self.client.backup()
        self.status_label.setText(f"Backup: {path}")

    def reset_from_source(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self, "Reset database", "Replace the user database with the curated source database?"
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        result = self.client.reset_from_source()
        self.status_label.setText(f"Reset complete. Backup: {result.get('backup_path')}")
        self.refresh()
