"""Migrated PyQt Burst Selection GUI backed by the new API."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from chisurf.core.fio.mmcif.db.pdbx_metadata import get_pdbx_metadata_keys
from chisurf.gui.widgets.dock_area.dock_area import DockArea, DockTabWidget
from chisurf.gui.widgets.progress import EnhancedProgressDialog
from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_channel_definition import (
    DetectorWizardPage,
)
from chisurf.gui.widgets.wizard.tttr_channeldefinition.tttr_detector_setups import (
    load_detector_setups,
)
from chisurf.gui.widgets.wizard.tttr_photonfilter.tttr_photon_filter import WizardTTTRPhotonFilter
from chisurf.server.rpc_logging import RpcLogWriter

from ..api.models import (
    AnalysisSettings,
    BurstDetectionSettings,
    BurstFilterMode,
    CountRateFilterSettings,
    DeltaMacroTimeFilterSettings,
    PhotonFilterSettings,
)
from .adapter import (
    UI_COLUMNS,
    burst_rows_for_display,
    make_ui_dataframe,
)
from .client import BurstSelectionClient

# Curated common keys shown first; then all PDBx keys are appended.
COMMON_METADATA_KEYS = [
    "pH", "temperature", "ionic_strength", "buffer_composition",
    "solvent_phase", "labeling_efficiency", "donor_only_fraction",
    "acceptor_only_fraction", "dye_ratio", "quencher_concentration",
    "time_resolution", "excitation_wavelength", "emission_wavelength",
    "power", "temperature_control", "data_notes",
]

# Build the full key list once
try:
    _PDBX_KEYS = get_pdbx_metadata_keys()
except Exception:
    _PDBX_KEYS = []
ALL_METADATA_KEYS = COMMON_METADATA_KEYS + [k for k in _PDBX_KEYS if k not in COMMON_METADATA_KEYS]

_LOG = RpcLogWriter("chisurf.plugins.burst.burst_selection")

DEFAULT_CHANNELS = [0, 1, 8, 9]
DEFAULT_MIN_PHOTONS = 60
DEFAULT_PHOTON_WINDOW = 5
DEFAULT_TIME_WINDOW_MS = 1.0
DEFAULT_D_T_MIN = 0.0001
DEFAULT_D_T_MAX = 0.15
DEFAULT_MAX_GAP = 3
DEFAULT_HISTOGRAM_BINS = 61
DEFAULT_TRACE_BIN_WIDTH_MS = 0.25
DEFAULT_DECAY_BINS = 8
DEFAULT_BURST_BINS = 51
DEFAULT_PLOT_MAX = 100000
HISTOGRAM_FEATURES = [*UI_COLUMNS, "Proximity Ratio"]


def _normalize_filetype(filetype: str | None) -> str | None:
    """Normalize the detector setup file type for ``tttrlib``."""
    if not filetype or str(filetype).strip().lower() == "auto":
        return None
    return str(filetype).strip()


def _setup_summary(setup_name: str | None, filetype: str | None) -> str:
    """Return a compact detector setup summary for the status text."""
    if not setup_name:
        return "Detector setup: custom/default"
    if filetype:
        return f"Detector setup: {setup_name} (file type: {filetype})"
    return f"Detector setup: {setup_name} (file type: auto)"


def default_analysis_settings() -> AnalysisSettings:
    """Return default analysis settings matching the legacy Burst Selection GUI."""
    return AnalysisSettings(
        photon_filter=PhotonFilterSettings(
            channels=DEFAULT_CHANNELS,
            filter_active=True,
            used_filter=BurstFilterMode.BURST,
            count_rate_filter=CountRateFilterSettings(
                n_ph_max=DEFAULT_MIN_PHOTONS,
                time_window=DEFAULT_TIME_WINDOW_MS / 1000.0,
                invert=True,
            ),
            delta_macro_time_filter=DeltaMacroTimeFilterSettings(
                dT_min=DEFAULT_D_T_MIN,
                dT_max=DEFAULT_D_T_MAX,
                dT_min_active=False,
                dT_max_active=True,
            ),
            invert_filter=True,
            max_gap=DEFAULT_MAX_GAP,
            use_gap_fill=False,
        ),
        burst_detection=BurstDetectionSettings(
            min_photons=DEFAULT_MIN_PHOTONS,
            photon_window=DEFAULT_PHOTON_WINDOW,
            time_window=DEFAULT_TIME_WINDOW_MS / 1000.0,
        ),
    )


def histogram_data_from_frame(frame: pd.DataFrame, feature: str) -> np.ndarray:
    """Return numeric histogram data excluding Margarita zero separator rows."""
    data = pd.to_numeric(burst_rows_for_display(frame)[feature], errors="coerce").dropna()
    return data.to_numpy(dtype=float)


class MetadataDialog(QtWidgets.QDialog):
    """Dialog for adding/editing metadata for burst analysis."""

    def __init__(self, metadata: dict[str, str] | None = None, parent: QtWidgets.QWidget = None) -> None:
        """Initialize the metadata dialog."""
        super().__init__(parent)
        self.setWindowTitle("Burst Analysis Metadata")
        self.resize(600, 400)
        layout = QtWidgets.QVBoxLayout(self)

        self.metadata_table = QtWidgets.QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.metadata_table)

        buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add metadata")
        add_btn.clicked.connect(self._add_metadata_row)
        delete_btn = QtWidgets.QPushButton("Delete selected")
        delete_btn.clicked.connect(self._delete_metadata_row)
        buttons.addWidget(add_btn)
        buttons.addWidget(delete_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        dialog_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        dialog_buttons.accepted.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        layout.addWidget(dialog_buttons)

        if metadata:
            for key, value in sorted(metadata.items()):
                self._add_metadata_row(str(key), str(value))

    def _add_metadata_row(self, key: str = "", value: str = "") -> None:
        """Add a metadata row to the table."""
        row = self.metadata_table.rowCount()
        self.metadata_table.insertRow(row)
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.addItems(ALL_METADATA_KEYS)
        if key:
            combo.setCurrentText(key)
        else:
            combo.setCurrentIndex(-1)
        comp = combo.completer()
        if comp is not None:
            comp.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
            comp.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.metadata_table.setCellWidget(row, 0, combo)
        self.metadata_table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))

    def _delete_metadata_row(self) -> None:
        """Delete the selected metadata row."""
        row = self.metadata_table.currentRow()
        if row >= 0:
            self.metadata_table.removeRow(row)

    def get_metadata(self) -> dict[str, str]:
        """Return the metadata from the table."""
        metadata = {}
        for row in range(self.metadata_table.rowCount()):
            widget = self.metadata_table.cellWidget(row, 0)
            if isinstance(widget, QtWidgets.QComboBox):
                key = widget.currentText().strip()
            else:
                key_item = self.metadata_table.item(row, 0)
                key = key_item.text().strip() if key_item is not None else ""
            value_item = self.metadata_table.item(row, 1)
            value = value_item.text().strip() if value_item is not None else ""
            if key:
                metadata[key] = value
        return metadata


class DropListWidget(QtWidgets.QListWidget):
    """List widget that accepts dropped file and folder paths."""

    pathsDropped = QtCore.Signal(list)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept URL drops."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        """Accept URL moves."""
        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Emit local paths from dropped URLs."""
        paths: list[Path] = []
        for url in event.mimeData().urls():
            local_path = url.toLocalFile()
            if local_path:
                path = Path(local_path)
                if path.exists():
                    paths.append(path)
        if paths:
            self.pathsDropped.emit(paths)
        event.acceptProposedAction()

    def supportedDropActions(self) -> QtCore.Qt.DropAction:
        """Return supported drop actions."""
        return QtCore.Qt.DropAction.CopyAction


class BatchProcessingDialog(QtWidgets.QDialog):
    """Dialog for adding folders containing TTTR files."""

    allowed_extensions = {".ht3", ".ptu", ".spc", ".hdf", ".h5"}

    def __init__(self, parent: BurstSelectionTool) -> None:
        """Initialize the batch folder dialog."""
        super().__init__(parent)
        self.setWindowTitle("Batch Burst Analysis")
        self.resize(700, 500)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Drop folders here. Folders containing TTTR files will be added.\n"
                "Folders without TTTR files will be scanned recursively.",
                self,
            )
        )
        self.list_widget = DropListWidget(self)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.pathsDropped.connect(self._add_folders_from_paths)
        layout.addWidget(self.list_widget, 1)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.delete_button = QtWidgets.QPushButton("Delete Selected", self)
        self.clear_button = QtWidgets.QPushButton("Clear All", self)
        self.add_button = QtWidgets.QPushButton("Add", self)
        buttons.addWidget(self.delete_button)
        buttons.addWidget(self.clear_button)
        buttons.addWidget(self.add_button)
        layout.addLayout(buttons)
        self.delete_button.clicked.connect(self._delete_selected)
        self.clear_button.clicked.connect(self.list_widget.clear)
        self.add_button.clicked.connect(self.accept)

    def _folder_has_tttr_files(self, folder: Path) -> list[str]:
        """Return TTTR files directly contained in a folder."""
        files: list[str] = []
        try:
            for child in folder.iterdir():
                if child.is_file() and child.suffix.lower() in self.allowed_extensions:
                    files.append(str(child.resolve()))
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Folder scan failed", f"{folder}\n{exc}")
        return files

    def _add_folder_unique(self, folder: Path) -> None:
        """Add a folder path once."""
        folder_text = str(folder.resolve())
        for index in range(self.list_widget.count()):
            if self.list_widget.item(index).text() == folder_text:
                return
        self.list_widget.addItem(folder_text)

    def _add_folders_from_paths(self, paths: list[Path]) -> None:
        """Add folders containing TTTR files."""
        for path in paths:
            if not path.is_dir():
                continue
            direct_files = self._folder_has_tttr_files(path)
            if direct_files:
                self._add_folder_unique(path)
                continue
            for subfolder in path.rglob("*"):
                if subfolder.is_dir() and self._folder_has_tttr_files(subfolder):
                    self._add_folder_unique(subfolder)

    def _delete_selected(self) -> None:
        """Delete selected folder entries."""
        for item in list(self.list_widget.selectedItems()):
            self.list_widget.takeItem(self.list_widget.row(item))

    def folders(self) -> list[Path]:
        """Return selected folders."""
        return [Path(item.text()) for item_index in range(self.list_widget.count()) for item in [self.list_widget.item(item_index)]]


class BurstSelectionTool(QtWidgets.QMainWindow):
    """Migrated Burst Selection GUI with legacy-style controls and plots."""

    def __init__(
        self,
        *args: object,
        show_channel_selection: bool = True,
        show_clear_button: bool = False,
        show_decay_button: bool = False,
        show_filter_button: bool = False,
        show_mcs_plot: bool = True,
        show_decay_plot: bool = True,
        show_filter_plot: bool = True,
        show_burst_plot: bool = True,
        **kwargs: object,
    ) -> None:
        """Initialize the migrated GUI and its API-backed controls."""
        assert isinstance(self, QtWidgets.QMainWindow), "BurstSelectionTool must be a QMainWindow"
        self.show_channel_selection = show_channel_selection
        self.show_clear_button = show_clear_button
        self.show_decay_button = show_decay_button
        self.show_filter_button = show_filter_button
        self.show_mcs_plot = show_mcs_plot
        self.show_decay_plot = show_decay_plot
        self.show_filter_plot = show_filter_plot
        self.show_burst_plot = show_burst_plot
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Burst Selection")
        self._client = BurstSelectionClient()
        self._file_paths: list[Path] = []
        self._last_result: dict[str, Any] | None = None
        self._last_bur_frames: list[pd.DataFrame] = []
        self._last_frame: pd.DataFrame | None = None
        self._last_settings: AnalysisSettings | None = None
        self._last_tttr: Any | None = None
        self._last_selected: np.ndarray | None = None
        self._last_start_stop: np.ndarray | None = None
        self._last_diagnostic_path: Path | None = None
        self._diagnostic_plot_features: dict[str, dict[str, Any]] = {}
        self._closed_diagnostic_plots: set[str] = set()
        self._selected_setup_name: str | None = None
        self._selected_filetype: str | None = None
        self._fit_gmm_on_update = False
        self._building_ui = True
        self._metadata: dict[str, str] = {}
        self._has_processed: bool = False
        self._create_plot_widgets()
        self._setup_statusbar()
        self._build_ui()
        self._building_ui = False
        self._setup_menu()
        self._sync_output_format_controls()
        self._apply_visibility_toggles()
        self._connect_action_bar()
        self._configure_dock_context_menu()
        self._on_setup_changed(self.wizard.comboBox.currentText())
        self.dock_area.layoutChanged.connect(self._save_dock_layout)
        self._restore_dock_layout()
        self.setAcceptDrops(True)

    def _build_ui(self) -> None:
        """Build the migrated GUI layout."""
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(self._build_action_bar(central))

        self.dock_area = DockArea(central)
        self._build_docks()
        layout.addWidget(self.dock_area, 1)

    def _build_action_bar(self, parent: QtWidgets.QWidget) -> QtWidgets.QLayout:
        """Create the top-level action bar (empty - controls moved to toolbar)."""
        action_bar = QtWidgets.QHBoxLayout()
        action_bar.addStretch(1)
        return action_bar

    def _build_control_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Create the left-side control panel (deprecated - controls now in separate docks)."""
        # This method is no longer used; controls are now in separate dock panels
        panel = QtWidgets.QWidget(parent)
        return panel

    def _build_wizard_embed(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        """Embed WizardTTTRPhotonFilter for setup, channel, and filter controls."""
        group = QtWidgets.QGroupBox("Filter settings", parent)
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        default_windows = {"prompt": (0, 2048), "delayed": (2048, 4095)}
        default_detectors = {
            "green": {"chs": [8, 0, 3], "micro_time_ranges": [(0, 4095)], "g_factor": 1, "l1": 0, "l2": 0},
            "red": {"chs": [9, 1, 2], "micro_time_ranges": [(0, 2048)], "g_factor": 1, "l1": 0, "l2": 0},
            "yellow": {"chs": [9, 1, 2], "micro_time_ranges": [(2048, 4095)], "g_factor": 1, "l1": 0, "l2": 0},
        }
        self.wizard = WizardTTTRPhotonFilter(
            windows=default_windows,
            detectors=default_detectors,
            show_dT=True,
            show_burst=False,
            show_mcs=False,
            show_decay=False,
            show_filter=False,
        )
        # Connect wizard status messages to the main window's statusbar
        self.wizard.status_message.connect(self._status_bar.showMessage)
        # Connect wizard filter parameter changes to plot updates
        self.wizard.actionUpdate_Values.triggered.connect(self._on_filter_settings_changed)
        setup_layout = QtWidgets.QHBoxLayout()
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(4)
        setup_layout.addWidget(QtWidgets.QLabel("Detector setup", group))
        self.wizard.comboBox.setMaximumWidth(340)
        setup_layout.addWidget(self.wizard.comboBox)
        self.wizard.comboBox.setToolTip(
            "Select a detector setup. The setup defines detectors, PIE windows, "
            "microtime binning, burst settings, and the TTTR file type."
        )
        self.wizard.comboBox.currentTextChanged.connect(self._on_setup_changed)
        # Add save button to the right of the combobox
        self.wizard.toolButton_5.setText("💾")
        self.wizard.toolButton_5.setFixedSize(28, 22)
        self.wizard.toolButton_5.setToolTip("Save burst parameters as default to setup.")
        self.wizard.toolButton_5.show()
        # Disconnect the original save_selection slot and reconnect it
        try:
            self.wizard.toolButton_5.clicked.disconnect()
        except Exception:
            pass
        # Reconnect to the save_burst_selection_parameters method (same as Macro Time interval groupbox)
        self.wizard.toolButton_5.clicked.connect(self.wizard.save_burst_selection_parameters)
        setup_layout.addWidget(self.wizard.toolButton_5)
        setup_layout.addStretch(1)
        layout.addLayout(setup_layout)
        # Hide redundant UI elements — we use our own action bar, file list, and dock plots
        self.wizard.textEdit.hide()  # left help panel
        self.wizard.lineEdit.hide()  # file drop area
        self.wizard.lineEdit_2.hide()  # output path
        self.wizard.spinBox_4.hide()  # file index
        self.wizard.toolButton.hide()  # help toggle
        self.wizard.toolButton_2.hide()  # MCS toggle
        self.wizard.toolButton_3.hide()  # decay toggle
        self.wizard.toolButton_4.hide()  # filter toggle
        self.wizard.toolButton_6.hide()  # clear button
        self.wizard.toolButton_7.hide()  # burst toggle
        self.wizard.checkBox_6.hide()  # sl5 output
        self.wizard.checkBox_7.hide()  # bur output
        self.wizard.groupBox_4.hide()  # plot settings
        # Keep: groupBox_3 (channel selection), groupBox_2 (macro time), groupBox (filter)
        self._compact_filter_settings_widgets()
        layout.addWidget(self.wizard)
        return group

    def _compact_filter_settings_widgets(self) -> None:
        """Reduce unused space in the embedded filter-settings controls."""
        self.wizard.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        for layout_name in ("gridLayout_4", "gridLayout_8", "gridLayout_2", "gridLayout_3", "gridLayout"):
            widget_layout = getattr(self.wizard, layout_name, None)
            if widget_layout is not None:
                widget_layout.setContentsMargins(2, 2, 2, 2)
                widget_layout.setSpacing(2)

        compact_groups = (
            (self.wizard.groupBox_3, 168),
            (self.wizard.groupBox_2, 182),
            (self.wizard.groupBox, 220),
        )
        for box, width in compact_groups:
            box.setMaximumWidth(width)
            box.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )

        width_by_widget = {
            self.wizard.comboBox_2: 150,
            self.wizard.comboBox_3: 150,
            self.wizard.comboBox_burst_filter: 132,
            self.wizard.lineEdit_4: 150,
            self.wizard.lineEdit_5: 150,
            self.wizard.doubleSpinBox_2: 86,
            self.wizard.doubleSpinBox_3: 86,
            self.wizard.spinBox: 70,
            self.wizard.spinBox_7: 70,
            self.wizard.spinBox_8: 70,
            self.wizard.checkBox: 64,
            self.wizard.checkBox_4: 72,
        }
        for widget, width in width_by_widget.items():
            widget.setMaximumWidth(width)

        for checkbox in (self.wizard.checkBox_2, self.wizard.checkBox_3, self.wizard.checkBox_5):
            checkbox.setMaximumWidth(18)

        controls_container = getattr(self.wizard, "widget", None)
        if controls_container is not None:
            controls_container.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Minimum,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )

    def _build_histogram_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        """Create histogram and optional GMM controls."""
        group = QtWidgets.QGroupBox("Histogram", parent)
        layout = QtWidgets.QFormLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.feature_combo = QtWidgets.QComboBox(group)
        self.feature_combo.addItems(HISTOGRAM_FEATURES)
        self.feature_combo.setCurrentText("Proximity Ratio")
        layout.addRow("Feature", self.feature_combo)
        self.hist_bins_spin = QtWidgets.QSpinBox(group)
        self.hist_bins_spin.setRange(1, 999)
        self.hist_bins_spin.setValue(DEFAULT_HISTOGRAM_BINS)
        layout.addRow("# Bins", self.hist_bins_spin)
        self.hist_min_spin = QtWidgets.QDoubleSpinBox(group)
        self.hist_min_spin.setRange(-9999.0, 9999.0)
        self.hist_min_spin.setDecimals(6)
        self.hist_min_spin.setValue(0.0)
        self.hist_max_spin = QtWidgets.QDoubleSpinBox(group)
        self.hist_max_spin.setRange(-9999.0, 9999.0)
        self.hist_max_spin.setDecimals(6)
        self.hist_max_spin.setValue(1.0)
        range_layout = QtWidgets.QHBoxLayout()
        range_layout.setSpacing(2)
        range_layout.addWidget(self.hist_min_spin)
        range_layout.addWidget(QtWidgets.QLabel("to", group))
        range_layout.addWidget(self.hist_max_spin)
        self.auto_range_button = QtWidgets.QPushButton("Auto", group)
        range_layout.addWidget(self.auto_range_button)
        layout.addRow("Range", range_layout)
        self.gmm_components_spin = QtWidgets.QSpinBox(group)
        self.gmm_components_spin.setRange(0, 10)
        self.gmm_auto_components_check = QtWidgets.QCheckBox("Auto components", group)
        self.fit_gmm_button = QtWidgets.QPushButton("Fit GMM", group)
        gmm_layout = QtWidgets.QHBoxLayout()
        gmm_layout.setSpacing(2)
        gmm_layout.addWidget(self.gmm_components_spin)
        gmm_layout.addWidget(self.gmm_auto_components_check)
        gmm_layout.addWidget(self.fit_gmm_button)
        layout.addRow("GMM", gmm_layout)
        self.gmm_summary = QtWidgets.QTextEdit(group)
        self.gmm_summary.setReadOnly(True)
        self.gmm_summary.setMaximumHeight(40)
        layout.addRow(self.gmm_summary)
        self._connect_histogram_controls(self._update_histogram_if_available)
        self.auto_range_button.clicked.connect(self._set_histogram_range_to_data)
        self.fit_gmm_button.clicked.connect(self._fit_gmm)
        return group

    def _build_plot_group(self, parent: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        """Create burst diagnostic plot controls (deprecated - kept for compatibility)."""
        # This method is no longer used; controls are now in separate panels
        # and connected directly in _build_plot_settings_panel, _build_mcs_controls_panel, etc.
        group = QtWidgets.QGroupBox("Plot settings", parent)
        return group

    def _build_filter_settings_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Create filter settings panel for separate dock."""
        panel = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_wizard_embed(panel))
        return panel

    def _build_histogram_controls_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Create histogram controls panel for separate dock."""
        panel = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_histogram_group(panel))
        layout.addStretch(1)
        return panel

    def _build_files_controls_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Create files output format controls panel for separate dock."""
        panel = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        group = QtWidgets.QGroupBox("Output Format", panel)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(4, 4, 4, 4)
        group_layout.setSpacing(2)
        format_layout = QtWidgets.QHBoxLayout()
        format_layout.setSpacing(2)
        self.csv_output_check = QtWidgets.QCheckBox("CSV", group)
        self.csv_output_check.setChecked(True)
        self.hdf_output_check = QtWidgets.QCheckBox("MFD-HDF", group)
        self.hdf_output_check.setEnabled(False)
        self.zip_output_check = QtWidgets.QCheckBox("Zip Output", group)
        self.remove_folder_check = QtWidgets.QCheckBox("Remove Folder", group)
        format_layout.addWidget(self.csv_output_check)
        format_layout.addWidget(self.hdf_output_check)
        format_layout.addWidget(self.zip_output_check)
        format_layout.addWidget(self.remove_folder_check)
        group_layout.addLayout(format_layout)
        self.csv_output_check.stateChanged.connect(self._sync_output_format_controls)
        self.hdf_output_check.stateChanged.connect(self._sync_output_format_controls)
        self.zip_output_check.stateChanged.connect(self._sync_output_format_controls)
        layout.addWidget(group)
        return panel

    def _build_mcs_controls_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Create MCS controls panel for separate dock."""
        panel = QtWidgets.QWidget(parent)
        layout = QtWidgets.QFormLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.mcs_bin_spin = QtWidgets.QDoubleSpinBox(panel)
        self.mcs_bin_spin.setRange(0.05, 9999.0)
        self.mcs_bin_spin.setSuffix(" ms")
        self.mcs_bin_spin.setDecimals(3)
        self.mcs_bin_spin.setSingleStep(0.05)
        self.mcs_bin_spin.setValue(DEFAULT_TRACE_BIN_WIDTH_MS)
        layout.addRow("MCS bin-width", self.mcs_bin_spin)
        self.mcs_bin_spin.valueChanged.connect(self.update_burst_plots)
        return panel

    def _build_decay_controls_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Create Decay controls panel for separate dock."""
        panel = QtWidgets.QWidget(parent)
        layout = QtWidgets.QFormLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.decay_bins_spin = QtWidgets.QSpinBox(panel)
        self.decay_bins_spin.setRange(1, 9999)
        self.decay_bins_spin.setValue(DEFAULT_DECAY_BINS)
        layout.addRow("Decay bin", self.decay_bins_spin)
        self.decay_bins_spin.valueChanged.connect(self.update_burst_plots)
        return panel

    def _build_burst_controls_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Create Burst controls panel for separate dock."""
        panel = QtWidgets.QWidget(parent)
        layout = QtWidgets.QFormLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.burst_bins_spin = QtWidgets.QSpinBox(panel)
        self.burst_bins_spin.setRange(3, 999)
        self.burst_bins_spin.setValue(DEFAULT_BURST_BINS)
        layout.addRow("#Burst bins", self.burst_bins_spin)
        self.burst_bins_spin.valueChanged.connect(self.update_burst_plots)
        return panel

    def _plot_widget_is_docked(self, attr: str) -> bool:
        """Return whether a diagnostic plot widget is currently present in the dock area."""
        try:
            features = getattr(self, "_diagnostic_plot_features", {})
        except RuntimeError:
            features = {}
        feature = features.get(attr)
        if feature is None or not feature["initial_enabled"]:
            return False
        try:
            closed_diagnostic_plots = self._closed_diagnostic_plots
        except RuntimeError:
            closed_diagnostic_plots = set()
        if attr in closed_diagnostic_plots or not bool(feature["check"].isChecked()):
            return False
        dock_widget = feature.get("dock_widget", feature["widget"])
        return self._dock_widget_is_present(dock_widget)

    def _dock_widget_is_present(self, widget: QtWidgets.QWidget | None) -> bool:
        """Return whether a dock widget is currently shown in the dock area."""
        if widget is None:
            return False
        try:
            dock_area = getattr(self, "dock_area", None)
        except RuntimeError:
            dock_area = None
        if dock_area is None:
            return True
        return dock_area.indexOf(widget) >= 0

    def _create_plot_widgets(self) -> None:
        """Create all plot widgets early to avoid hot-reload deletion issues."""
        self.summary = QtWidgets.QTextEdit(self)
        self.summary.setReadOnly(True)
        
        self.histogram_plot = pg.PlotWidget(self)
        self.histogram_plot.setLabel("bottom", "Value")
        self.histogram_plot.setLabel("left", "Frequency")
        self.histogram_plot.setTitle("Histogram")
        
        self.filter_plot = pg.PlotWidget(self)
        self.filter_plot.setLabel("bottom", "Photon Index")
        self.filter_plot.setLabel("left", "Selected")
        self.filter_plot.setTitle("Filter/selection")
        
        self.mcs_plot = pg.PlotWidget(self)
        self.mcs_plot.setLabel("bottom", "Time (s)")
        self.mcs_plot.setLabel("left", "Intensity")
        self.mcs_plot.setTitle("Count rate display")
        
        self.decay_plot = pg.PlotWidget(self)
        self.decay_plot.setLabel("bottom", "Microtime (ns)")
        self.decay_plot.setLabel("left", "Counts")
        self.decay_plot.setTitle("Microtime histogram")
        self.decay_plot.getPlotItem().setLogMode(False, True)
        
        self.burst_plot = pg.PlotWidget(self)
        self.burst_plot.setLabel("bottom", "Burst size")
        self.burst_plot.setLabel("left", "Counts")
        self.burst_plot.setTitle("Burst histogram")
        
        self.dt_plot = pg.PlotWidget(self)
        self.dt_plot.setLabel("bottom", "Photon Index")
        self.dt_plot.setLabel("left", "dT (ms)")
        self.dt_plot.setTitle("Delta macro-time")
        self.dt_plot.getPlotItem().setLogMode(False, True)
        
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(len(UI_COLUMNS))
        self.table.setHorizontalHeaderLabels(UI_COLUMNS)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.table.horizontalHeader().setStretchLastSection(True)

    def _build_docks(self) -> None:
        """Create draggable result and diagnostic docks using pre-created widgets."""

        # Create file_list widget
        self.file_list = DropListWidget(self.dock_area)
        self.file_list.pathsDropped.connect(self._add_paths)
        self.file_list.itemSelectionChanged.connect(self._on_file_selected)
        self.file_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._show_file_list_context_menu)

        # Build separate control panels for each dock
        filter_settings_panel = self._build_filter_settings_panel(self.dock_area)
        self.filter_settings_panel = filter_settings_panel
        histogram_controls_panel = self._build_histogram_controls_panel(self.dock_area)
        files_controls_panel = self._build_files_controls_panel(self.dock_area)
        mcs_controls_panel = self._build_mcs_controls_panel(self.dock_area)
        decay_controls_panel = self._build_decay_controls_panel(self.dock_area)
        burst_controls_panel = self._build_burst_controls_panel(self.dock_area)

        # Add Filter Settings dock
        self.dock_area.addTab(filter_settings_panel, "Filter Settings")

        # Add Files dock with controls
        files_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self.dock_area)
        files_splitter.addWidget(files_controls_panel)

        # Add drop hint label
        drop_hint = QtWidgets.QLabel("Drop files or folders here", self.dock_area)
        drop_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        drop_hint.setStyleSheet("color: gray; font-style: italic;")
        files_splitter.addWidget(drop_hint)

        files_splitter.addWidget(self.file_list)
        files_splitter.setStretchFactor(0, 0)
        files_splitter.setStretchFactor(1, 0)
        files_splitter.setStretchFactor(2, 1)
        self.dock_area.addTab(files_splitter, "Files")

        self.dock_area.addTab(self.table, "Bursts")

        # Add dT plot dock
        self.dock_area.addTab(self.dt_plot, "dT")

        # Add Histogram dock with controls
        histogram_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self.dock_area)
        histogram_splitter.addWidget(histogram_controls_panel)
        histogram_splitter.addWidget(self.histogram_plot)
        histogram_splitter.setStretchFactor(0, 0)
        histogram_splitter.setStretchFactor(1, 1)
        self.dock_area.addTab(histogram_splitter, "Histogram")

        if self.show_filter_plot:
            self.dock_area.addTab(self.filter_plot, "Filter")

        # Add MCS dock with controls
        if self.show_mcs_plot:
            mcs_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self.dock_area)
            mcs_splitter.addWidget(mcs_controls_panel)
            mcs_splitter.addWidget(self.mcs_plot)
            mcs_splitter.setStretchFactor(0, 0)
            mcs_splitter.setStretchFactor(1, 1)
            self.dock_area.addTab(mcs_splitter, "MCS")

        # Add Decay dock with controls
        if self.show_decay_plot:
            decay_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self.dock_area)
            decay_splitter.addWidget(decay_controls_panel)
            decay_splitter.addWidget(self.decay_plot)
            decay_splitter.setStretchFactor(0, 0)
            decay_splitter.setStretchFactor(1, 1)
            self.dock_area.addTab(decay_splitter, "Decay")

        # Add Burst dock with controls
        if self.show_burst_plot:
            burst_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self.dock_area)
            burst_splitter.addWidget(burst_controls_panel)
            burst_splitter.addWidget(self.burst_plot)
            burst_splitter.setStretchFactor(0, 0)
            burst_splitter.setStretchFactor(1, 1)
            self.dock_area.addTab(burst_splitter, "Burst length")

        self.dock_area.addTab(self.summary, "Summary")

        self.filter_dock_widget = self.filter_plot
        self.mcs_dock_widget = locals().get("mcs_splitter")
        self.decay_dock_widget = locals().get("decay_splitter")
        self.burst_dock_widget = locals().get("burst_splitter")

        # Initialize plot checkboxes (used for context menu, not displayed)
        self.plot_mcs_check = QtWidgets.QCheckBox(self)
        self.plot_mcs_check.setChecked(self.show_mcs_plot)
        self.plot_decay_check = QtWidgets.QCheckBox(self)
        self.plot_decay_check.setChecked(self.show_decay_plot)
        self.plot_filter_check = QtWidgets.QCheckBox(self)
        self.plot_filter_check.setChecked(self.show_filter_plot)
        self.plot_burst_check = QtWidgets.QCheckBox(self)
        self.plot_burst_check.setChecked(self.show_burst_plot)

    def _setup_menu(self) -> None:
        """Create menu actions."""
        file_menu = self.menuBar().addMenu("File")
        open_files_action = QtWidgets.QAction("Add TTTR files", self)
        open_files_action.triggered.connect(self.add_files)
        open_folder_action = QtWidgets.QAction("Open folder", self)
        open_folder_action.triggered.connect(self.open_batch_dialog)
        file_menu.addSeparator()
        export_submenu = file_menu.addMenu("Export")
        export_bur_action = QtWidgets.QAction("Export as .bur", self)
        export_bur_action.triggered.connect(self.export_bur)
        export_submenu.addAction(export_bur_action)
        export_cif_action = QtWidgets.QAction("Export as flrCIF", self)
        export_cif_action.triggered.connect(self.export_flr_cif)
        export_submenu.addAction(export_cif_action)
        file_menu.addSeparator()
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_files_action)
        file_menu.addAction(open_folder_action)
        file_menu.addAction(exit_action)

        settings_menu = self.menuBar().addMenu("Settings")
        channels_action = QtWidgets.QAction("Channels", self)
        channels_action.triggered.connect(self._show_channel_settings)
        settings_menu.addAction(channels_action)
        gmm_action = QtWidgets.QAction("GMM", self)
        gmm_action.triggered.connect(self._focus_gmm_controls)
        settings_menu.addAction(gmm_action)
        metadata_action = QtWidgets.QAction("Metadata", self)
        metadata_action.triggered.connect(self._show_metadata_dialog)
        settings_menu.addAction(metadata_action)

        help_menu = self.menuBar().addMenu("Help")
        about_action = QtWidgets.QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        self._setup_toolbar()

    def _connect_histogram_controls(self, slot: Any) -> None:
        """Connect histogram controls to a common update slot."""
        controls = [self.feature_combo, self.hist_bins_spin, self.hist_min_spin, self.hist_max_spin]
        for control in controls:
            if isinstance(control, QtWidgets.QComboBox):
                control.currentTextChanged.connect(slot)
            else:
                control.valueChanged.connect(slot)

    def _connect_plot_controls(self, slot: Any) -> None:
        """Connect diagnostic plot controls to a common update slot (deprecated)."""
        # This method is no longer used; controls are now connected directly
        # in their respective panel building methods (_build_plot_settings_panel, etc.)
        pass

    def _apply_visibility_toggles(self) -> None:
        """Apply visibility toggles passed as constructor kwargs."""
        if not self.show_channel_selection:
            self.wizard.groupBox_3.hide()

    def _connect_action_bar(self) -> None:
        """Connect action bar control signals."""
        self.plot_min_spin.valueChanged.connect(self.update_burst_plots)
        self.plot_max_spin.valueChanged.connect(self.update_burst_plots)
        self.show_all_photons_check.stateChanged.connect(self.update_burst_plots)
        self.show_selected_photons_check.stateChanged.connect(self.update_burst_plots)

    def _configure_dock_context_menu(self) -> None:
        """Configure tab context menus for closing and re-enabling diagnostic plots."""
        self._diagnostic_plot_features = {
            "Filter": {
                "widget": self.filter_plot,
                "dock_widget": self.filter_dock_widget,
                "check": self.plot_filter_check,
                "initial_enabled": self.show_filter_plot,
            },
            "MCS": {
                "widget": self.mcs_plot,
                "dock_widget": self.mcs_dock_widget,
                "check": self.plot_mcs_check,
                "initial_enabled": self.show_mcs_plot,
            },
            "Decay": {
                "widget": self.decay_plot,
                "dock_widget": self.decay_dock_widget,
                "check": self.plot_decay_check,
                "initial_enabled": self.show_decay_plot,
            },
            "Burst length": {
                "widget": self.burst_plot,
                "dock_widget": self.burst_dock_widget,
                "check": self.plot_burst_check,
                "initial_enabled": self.show_burst_plot,
            },
        }
        self.dock_area.setContextMenuEnabled(True)
        self.dock_area.setContextMenuMode("basic")
        self.dock_area.setTabsClosable(True)
        self.dock_area.setCloseTabCallback(self._on_dock_tab_close_requested)
        self.dock_area.setContextMenuCallback(self._add_dock_context_menu_actions)

    def _on_dock_tab_close_requested(self, index: int) -> None:
        """Handle diagnostic plot close requests without computing hidden plots."""
        widget = self.dock_area.widget(index)
        # Find which diagnostic plot this widget corresponds to
        name = None
        for plot_name, feature in self._diagnostic_plot_features.items():
            if feature.get("dock_widget", feature["widget"]) is widget:
                name = plot_name
                break
        if name is not None and self._diagnostic_plot_features[name]["initial_enabled"]:
            self._closed_diagnostic_plots.add(name)
            self._diagnostic_plot_features[name]["check"].setChecked(False)
            self.dock_area.removeTab(index)
            self.update_burst_plots()
            return
        self.dock_area.removeTab(index)

    def _add_dock_context_menu_actions(self, menu: QtWidgets.QMenu, index: int) -> None:
        """Add actions to show closed or disabled diagnostic plots."""
        del index
        unchecked = self._disabled_diagnostic_plot_names()
        if unchecked:
            menu.addSeparator()
            enable_menu = menu.addMenu("Enable plots")
            for name in unchecked:
                action = enable_menu.addAction(name)
                action.triggered.connect(
                    lambda _checked=False, plot_name=name: self._enable_diagnostic_plot(plot_name)
                )

        closed = self._closed_diagnostic_plot_names()
        if closed:
            menu.addSeparator()
            show_menu = menu.addMenu("Reopen closed docks")
            for name in closed:
                action = show_menu.addAction(name)
                action.triggered.connect(
                    lambda _checked=False, plot_name=name: self._show_diagnostic_plot(plot_name)
                )

    def _disabled_diagnostic_plot_names(self) -> list[str]:
        """Return disabled diagnostic plots that still have open docks."""
        return [
            name
            for name, feature in self._diagnostic_plot_features.items()
            if feature["initial_enabled"]
            and name not in self._closed_diagnostic_plots
            and not bool(feature["check"].isChecked())
        ]

    def _closed_diagnostic_plot_names(self) -> list[str]:
        """Return diagnostic plots closed from the dock area."""
        return [
            name
            for name, feature in self._diagnostic_plot_features.items()
            if feature["initial_enabled"] and name in self._closed_diagnostic_plots
        ]

    def _enable_diagnostic_plot(self, name: str) -> None:
        """Enable a diagnostic plot that is still present as a tab."""
        if name in self._closed_diagnostic_plots:
            self._show_diagnostic_plot(name)
            return
        feature = self._diagnostic_plot_features.get(name)
        if feature is None:
            return
        feature["check"].setChecked(True)
        self.update_burst_plots()

    def _show_diagnostic_plot(self, name: str) -> None:
        """Re-add a diagnostic plot tab that was closed from the context menu."""
        feature = self._diagnostic_plot_features.get(name)
        if feature is None or name not in self._closed_diagnostic_plots:
            return
        self._closed_diagnostic_plots.remove(name)
        self.dock_area.addTab(feature.get("dock_widget", feature["widget"]), name)
        feature["check"].setChecked(True)
        self.update_burst_plots()

    def _show_channel_settings(self) -> None:
        """Open a DetectorWizardPage dialog to manage setups, then sync to wizard."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Channel Settings")
        layout = QtWidgets.QVBoxLayout(dialog)
        channel_definer = DetectorWizardPage(parent=dialog, json_file=None)
        layout.addWidget(channel_definer)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        if dialog.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            setup_name = getattr(channel_definer, "current_setup_name", None)
            if setup_name:
                self._apply_detector_setup(setup_name)
                return
            self._apply_custom_detector_settings(
                windows=channel_definer.windows,
                detectors=channel_definer.detectors,
                filetype=channel_definer.tttr_reading.get("file_type"),
            )

    def _on_setup_changed(self, setup_name: str) -> None:
        """Apply detector setup selected in the embedded photon-filter wizard."""
        if self._building_ui or not setup_name or setup_name == "No setups available":
            self._selected_setup_name = None
            self._selected_filetype = None
            self.summary.setPlainText(_setup_summary(None, None))
            return
        self._apply_detector_setup(setup_name)

    def _apply_detector_setup(self, setup_name: str) -> None:
        """Apply a named detector setup to the embedded wizard."""
        setups = load_detector_setups()
        setup_data = setups.get("setups", {}).get(setup_name)
        if not setup_data:
            self._selected_setup_name = None
            self._selected_filetype = None
            self.summary.setPlainText(_setup_summary(None, None))
            return
        reading = setup_data.get("tttr_reading", {}) or {}
        filetype = _normalize_filetype(reading.get("file_type"))
        self._selected_setup_name = setup_name
        self._selected_filetype = filetype
        self.wizard.comboBox.blockSignals(True)
        try:
            if self.wizard.comboBox.findText(setup_name) < 0:
                self.wizard.comboBox.addItem(setup_name)
            self.wizard.comboBox.setCurrentText(setup_name)
        finally:
            self.wizard.comboBox.blockSignals(False)
        self.wizard.update_channel_routing(setup_name)
        self.wizard.update_pie_windows_from_setup(setup_name)
        self.wizard.update_micro_time_binning(setup_name)
        self._apply_burst_selection_parameters(setup_data.get("burst_selection", {}) or {})
        self.summary.setPlainText(_setup_summary(setup_name, filetype))

    def _apply_burst_selection_parameters(self, burst_params: dict[str, Any]) -> None:
        """Apply burst-selection parameters from a detector setup when present."""
        if "dT_min" in burst_params and "dT_max" in burst_params:
            self.wizard._dT_min = float(burst_params["dT_min"])
            self.wizard._dT_max = float(burst_params["dT_max"])
            self.wizard.doubleSpinBox_2.setValue(float(burst_params["dT_min"]))
            self.wizard.doubleSpinBox_3.setValue(float(burst_params["dT_max"]))
        if "use_dT_min" in burst_params:
            self.wizard.checkBox_2.setChecked(bool(burst_params["use_dT_min"]))
        if "use_dT_max" in burst_params:
            self.wizard.checkBox_3.setChecked(bool(burst_params["use_dT_max"]))
        if "photon_threshold" in burst_params:
            self.wizard.spinBox.setValue(int(burst_params["photon_threshold"]))
        if "ph_window" in burst_params:
            self.wizard.ph_window = int(burst_params["ph_window"])
        if "count_rate_window_ms" in burst_params:
            self.wizard.doubleSpinBox.setValue(float(burst_params["count_rate_window_ms"]))
        if "invert_filter" in burst_params:
            self.wizard.checkBox.setChecked(bool(burst_params["invert_filter"]))
        if "filter_active" in burst_params:
            self.wizard.checkBox_4.setChecked(bool(burst_params["filter_active"]))
        if "filter_mode" in burst_params:
            mode = str(burst_params["filter_mode"]).lower()
            if mode == "count_rate":
                self.wizard.comboBox_burst_filter.setCurrentText("Count rate")
            elif mode in {"burst", "bocpd", "kalman"}:
                label = {"burst": "Burst", "bocpd": "BOCPD Burst", "kalman": "Kalman Burst"}[mode]
                self.wizard.comboBox_burst_filter.setCurrentText(label)
        if "use_gap_fill" in burst_params:
            self.wizard.checkBox_5.setChecked(bool(burst_params["use_gap_fill"]))
        if "max_gap" in burst_params:
            self.wizard.spinBox_7.setValue(int(burst_params["max_gap"]))
        if "trace_bin_width" in burst_params:
            self.wizard.doubleSpinBox_4.setValue(float(burst_params["trace_bin_width"]))
        if "number_of_burst_bins" in burst_params:
            self.wizard.spinBox_6.setValue(int(burst_params["number_of_burst_bins"]))
        if "decay_coarse" in burst_params:
            self.wizard.spinBox_5.setValue(int(burst_params["decay_coarse"]))
        self.wizard.update_parameter()

    def _sync_detector_controls(self) -> None:
        """Refresh detector and PIE-window combo boxes after custom edits."""
        self.wizard.comboBox_2.blockSignals(True)
        self.wizard.comboBox_2.clear()
        self.wizard.comboBox_2.addItem("All")
        self.wizard.comboBox_2.addItems(self.wizard.detectors.keys())
        if self.wizard.comboBox_2.count():
            self.wizard.comboBox_2.setCurrentIndex(0)
        self.wizard.comboBox_2.blockSignals(False)

        self.wizard.comboBox_3.blockSignals(True)
        self.wizard.comboBox_3.clear()
        self.wizard.comboBox_3.addItems(self.wizard.windows.keys())
        if self.wizard.comboBox_3.count():
            self.wizard.comboBox_3.setCurrentIndex(0)
        self.wizard.comboBox_3.blockSignals(False)

        if self.wizard.detectors:
            self.wizard.update_detectors()
        if self.wizard.windows:
            self.wizard.update_pie_windows()

    def _apply_custom_detector_settings(
        self,
        windows: dict[str, Any],
        detectors: dict[str, Any],
        filetype: str | None = None,
    ) -> None:
        """Apply detector settings that were not loaded from a named setup."""
        self._selected_setup_name = None
        self._selected_filetype = _normalize_filetype(filetype)
        self.wizard.detectors = detectors
        self.wizard.windows = windows
        self._sync_detector_controls()
        self.summary.setPlainText(_setup_summary(None, self._selected_filetype))

    def _settings_from_controls(self) -> AnalysisSettings:
        """Build API settings from the embedded wizard's current state."""
        mode_text = self.wizard.comboBox_burst_filter.currentText()
        used_filter = BurstFilterMode.BURST if "Burst" in mode_text else BurstFilterMode.COUNT_RATE
        threshold = int(self.wizard.min_ph)
        time_window = float(self.wizard.spinBox.value() if hasattr(self.wizard, 'spinBox') else DEFAULT_TIME_WINDOW_MS) / 1000.0
        if used_filter == BurstFilterMode.BURST:
            burst_detection = BurstDetectionSettings(
                min_photons=threshold,
                photon_window=self.wizard.ph_window,
                time_window=time_window,
            )
            count_rate = CountRateFilterSettings(
                n_ph_max=DEFAULT_MIN_PHOTONS,
                time_window=time_window,
                invert=self.wizard.checkBox.isChecked(),
            )
        else:
            burst_detection = BurstDetectionSettings(
                min_photons=DEFAULT_MIN_PHOTONS,
                photon_window=self.wizard.ph_window,
                time_window=time_window,
            )
            count_rate = CountRateFilterSettings(
                n_ph_max=threshold,
                time_window=time_window,
                invert=self.wizard.checkBox.isChecked(),
            )
        channels = list(self.wizard.channels) if self.wizard.channels else []
        micro_ranges = self.wizard.microtime_ranges or []
        output_formats = []
        if self.csv_output_check.isChecked():
            output_formats.append("bur")
        if self.hdf_output_check.isChecked():
            output_formats.append("hdf5")
        return AnalysisSettings(
            photon_filter=PhotonFilterSettings(
                channels=channels,
                microtime_ranges=micro_ranges,
                filter_active=self.wizard.checkBox_4.isChecked(),
                used_filter=used_filter,
                count_rate_filter=count_rate,
                delta_macro_time_filter=DeltaMacroTimeFilterSettings(
                    dT_min=float(self.wizard.dT_min),
                    dT_max=float(self.wizard.dT_max),
                    dT_min_active=bool(self.wizard.use_lower),
                    dT_max_active=bool(self.wizard.use_upper),
                ),
                invert_filter=self.wizard.checkBox.isChecked(),
                max_gap=int(self.wizard.max_gap),
                use_gap_fill=bool(self.wizard.use_gap_fill),
            ),
            burst_detection=burst_detection,
            output_formats=output_formats,
            zip_output=self.zip_output_check.isChecked(),
            remove_folder=self.remove_folder_check.isChecked(),
        )

    def add_files(self) -> None:
        """Open a file dialog and add TTTR files to the analysis queue."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select TTTR files",
            "",
            "TTTR files (*.spc *.ht3 *.ptu *.hdf *.h5);;All files (*)",
        )
        self._add_paths([Path(path) for path in paths])

    def _add_paths(self, paths: list[Path]) -> None:
        """Add files and TTTR-containing folders to the file list."""
        for path in paths:
            if path.is_dir():
                self._file_paths.extend(
                    sorted(
                        child.resolve()
                        for child in path.iterdir()
                        if child.is_file() and child.suffix.lower() in {".spc", ".ht3", ".ptu", ".hdf", ".h5"}
                    )
                )
            else:
                self._file_paths.append(path.resolve())
        self._refresh_file_list()

    def _on_filter_settings_changed(self) -> None:
        """Handle filter parameter changes by reloading diagnostics and updating plots."""
        try:
            # Get the currently selected file
            if not self.file_list or self.file_list is None:
                return
            selected_items = self.file_list.selectedItems()
            if not selected_items:
                return
            selected_item = selected_items[0]
            if selected_item is None:
                return
            selected_text = selected_item.text()
            if not selected_text:
                return
            selected_path = Path(selected_text)
            if selected_path not in self._file_paths:
                return
            
            # Reload diagnostics with new filter settings
            settings = self._settings_from_controls()
            self._status_bar.showMessage(f"Reloading diagnostics with new filter settings...")
            self._load_tttr_for_plots(selected_path, settings)
        except Exception as exc:
            self._status_bar.showMessage(f"Error updating plots after filter change: {exc}")
            _LOG.error("error updating plots after filter change", error=str(exc))

    def analyze_files(self) -> None:
        """Analyze queued files through the Burst Selection API."""
        if not self._file_paths:
            self.summary.setPlainText("No TTTR files selected.")
            return
        settings = self._settings_from_controls()
        if not settings.output_formats:
            self.summary.setPlainText("No output format selected.")
            return
        frames: list[pd.DataFrame] = []
        metadata: dict[str, Any] = {"n_files": len(self._file_paths), "n_bursts": 0, "n_photons": 0, "n_selected": 0}
        dialog = EnhancedProgressDialog(
            title="Burst Selection",
            label_text="Processing files...",
            min_value=0,
            max_value=100,
            parent=self,
        )
        dialog.show()
        dialog.update_progress(0, "Processing files...")
        cancelled = False
        try:
            settings_dict = asdict(settings) if settings else {}
            windows = getattr(self.wizard, "windows", None)
            detectors = getattr(self.wizard, "detectors", None)
            legacy_parameters: dict[str, Any] = {}
            try:
                if hasattr(self.wizard, "get_burst_selection_parameters"):
                    legacy_parameters.update(self.wizard.get_burst_selection_parameters())
            except Exception as exc:
                _LOG.warning("failed to collect legacy burst-selection parameters", error=str(exc))
            if hasattr(self.wizard, "decay_coarse"):
                legacy_parameters.setdefault("decay_coarse", self.wizard.decay_coarse)
            try:
                result = self._client.analyze_files(
                    self._file_paths,
                    settings=settings_dict,
                    windows=windows,
                    detectors=detectors,
                    filetype=self._selected_filetype,
                    legacy_output=True,
                    selected_setup=self.wizard.comboBox.currentText(),
                    legacy_parameters=legacy_parameters,
                )
            except RuntimeError as rpc_err:
                dialog.finish(
                    final_text=f"RPC error: {rpc_err}",
                    auto_close=False,
                    close_delay_ms=5000,
                )
                self.summary.setPlainText(f"RPC error: {rpc_err}")
                cancelled = True
                result = {}
            if not cancelled:
                dataframes = result.get("dataframes", {})
                for path in self._file_paths:
                    raw_frames = dataframes.get(str(path), [])
                    if not raw_frames:
                        raw_frames = dataframes.get(str(path.resolve()), [])
                    frame = pd.DataFrame(raw_frames)
                    frames.append(frame)
                metadata.update(result.get("metadata", {}))
                dialog.update_progress(100, "Processing files...")
        finally:
            final_text = "Burst selection cancelled." if cancelled else "Burst selection finished."
            dialog.finish(final_text=final_text, auto_close=True, close_delay_ms=0)

        metadata["n_files"] = len(frames)
        if cancelled:
            self.summary.setPlainText("Burst selection cancelled.")
            return

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            self._last_frame = combined
            self._last_settings = settings
            self._last_bur_frames = frames
            self._fill_table(make_ui_dataframe(combined))
            self._populate_feature_combo(combined)
            self._load_first_tttr_for_plots(settings)
            self.update_histogram()
            self.update_burst_plots()
            self._has_processed = True
        else:
            self.table.setRowCount(0)
            self._last_frame = None
            self._last_settings = settings
            self._last_bur_frames.clear()
            self._last_tttr = None
            self._last_selected = None
            self._last_start_stop = None
            self._clear_plots()
            self._has_processed = False

        self._last_result = {"metadata": metadata, "settings": settings}
        self.summary.setPlainText(json.dumps(metadata | {"settings": asdict(settings)}, indent=2, default=str))

    def _load_first_tttr_for_plots(self, settings: AnalysisSettings) -> None:
        """Load the first TTTR file and selected mask for diagnostic plots."""
        if not self._file_paths:
            return
        path = self._file_paths[0]
        try:
            settings_dict = asdict(settings) if settings else {}
            diag = self._client.load_diagnostics(path, settings_dict)
            if diag and "tttr" in diag:
                self._last_tttr = diag["tttr"]
                self._last_selected = diag["selected"]
                self._last_start_stop = diag["start_stop"]
                self._last_diagnostic_path = path
            else:
                raise ValueError("diagnostics returned no data")
        except Exception as exc:
            self._last_tttr = None
            self._last_selected = None
            self._last_start_stop = None
            self._last_diagnostic_path = None
            self.summary.append(f"Diagnostic plots unavailable for {path}: {exc}")

    def save_current_bur(self) -> None:
        """Save the last API result as a ChiSurf-compatible .bur file."""
        if self._last_result is None:
            self.summary.setPlainText("No analysis result to save.")
            return
        if not self._file_paths:
            self.summary.setPlainText("No TTTR files selected.")
            return
        if not self._last_bur_frames:
            self.summary.setPlainText("No burst table available to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save burst table",
            str(self._file_paths[0].with_suffix(".bur")),
            "Burst files (*.bur)",
        )
        if not path:
            return
        combined = pd.concat(self._last_bur_frames, ignore_index=True)
        self._client.save_bur(combined, Path(path))
        self.summary.setPlainText(f"Saved {path}")

    def clear(self) -> None:
        """Clear selected files and result views without resetting analysis controls."""
        self._file_paths.clear()
        self._last_result = None
        self._last_bur_frames.clear()
        self._last_frame = None
        self._last_settings = None
        self._last_tttr = None
        self._last_selected = None
        self._last_start_stop = None
        self._last_diagnostic_path = None
        self._has_processed = False
        self._refresh_file_list()
        self.table.setRowCount(0)
        self._clear_plots()
        self.summary.clear()
        self.gmm_summary.clear()

    def update_histogram(self) -> None:
        """Update the histogram plot without fitting GMMs by default."""
        if self._last_frame is None:
            self.histogram_plot.clear()
            self.gmm_summary.clear()
            return
        feature = self.feature_combo.currentText()
        if feature not in self._last_frame.columns:
            self.histogram_plot.clear()
            return
        data = histogram_data_from_frame(self._last_frame, feature)
        if data.size == 0:
            self.histogram_plot.clear()
            self.gmm_summary.clear()
            return
        min_value, max_value = self._valid_histogram_range(data)
        counts, edges = np.histogram(data, bins=int(self.hist_bins_spin.value()), range=(min_value, max_value))
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = edges[1] - edges[0]
        self.histogram_plot.clear()
        self.histogram_plot.setLabel("bottom", feature)
        self.histogram_plot.setLabel("left", "Frequency")
        self.histogram_plot.addItem(pg.BarGraphItem(x=centers, height=counts, width=width, brush="b", pen="k", alpha=0.7))
        if self._fit_gmm_on_update:
            self._plot_gmm(data, min_value, max_value, counts)
        else:
            self.gmm_summary.clear()

    def update_burst_plots(self) -> None:
        """Update diagnostic plots for the first analyzed TTTR file."""
        try:
            _LOG.debug("update_burst_plots called")
            if self._last_tttr is None or self._last_selected is None:
                _LOG.debug("diagnostic plot update skipped: TTTR or selection is missing")
                self._clear_burst_plots()
                return
            selected = self._last_selected.astype(bool)
            if selected.size == 0:
                _LOG.debug("diagnostic plot update skipped: selection is empty")
                self._clear_burst_plots()
                return
            n_photons = len(selected)
            self._sync_plot_range_controls(n_photons)
            start = max(0, int(self.plot_min_spin.value()))
            stop = min(n_photons, int(self.plot_max_spin.value()) + 1)
            if stop <= start:
                stop = min(n_photons, start + 1)
            indices = np.arange(start, stop)
            selected_slice = selected[start:stop]
            show_all_photons = self._show_all_photons()
            show_selected_photons = self._show_selected_photons()
            _LOG.debug(
                "updating diagnostic plots",
                selected_photons=int(np.count_nonzero(selected_slice)),
                visible_photons=int(len(selected_slice)),
                total_photons=int(n_photons),
                show_all_photons=show_all_photons,
                show_selected_photons=show_selected_photons,
            )
            show_dt = self._dock_widget_is_present(getattr(self, "dt_plot", None))
            show_filter = self._plot_widget_is_docked("Filter")
            show_filter_settings = self._dock_widget_is_present(getattr(self, "filter_settings_panel", None))
            d_t = None
            if show_dt or show_filter_settings:
                d_t = self._delta_macro_time_ms(self._last_tttr)[start:stop]
            
            if show_dt and d_t is not None:
                self.dt_plot.clear()
                if show_all_photons:
                    self.dt_plot.plot(indices, d_t, pen=pg.mkPen((255, 255, 0, 80), width=1))
                if show_selected_photons:
                    self.dt_plot.plot(indices[selected_slice], d_t[selected_slice], pen=pg.mkPen("c", width=1))
                self._set_log_dt_range(self.dt_plot, d_t)
                _LOG.debug("dT plot updated")
            
            if show_filter:
                self.filter_plot.clear()
                if show_all_photons:
                    self.filter_plot.plot(
                        indices,
                        np.zeros_like(indices, dtype=float),
                        pen=pg.mkPen((255, 255, 0, 80), width=1),
                        stepMode=False,
                    )
                if show_selected_photons:
                    selected_indices = indices[selected_slice]
                    self.filter_plot.plot(
                        selected_indices,
                        np.ones_like(selected_indices, dtype=float),
                        pen=pg.mkPen("c", width=1),
                        stepMode=False,
                    )
                self.filter_plot.setYRange(-0.1, 1.1)
                _LOG.debug("filter plot updated")

            if show_filter_settings and d_t is not None:
                self._update_filter_settings_diagnostics(start, stop, indices, selected_slice, d_t)
            
            if self._plot_widget_is_docked("MCS"):
                self._update_mcs_plot(start, stop)
                _LOG.debug("MCS plot updated")
            
            if self._plot_widget_is_docked("Decay"):
                self._update_decay_plot()
                _LOG.debug("decay plot updated")
            
            if self._plot_widget_is_docked("Burst length"):
                self._update_burst_length_plot()
                _LOG.debug("burst-length plot updated")
        except RuntimeError as exc:
            self._status_bar.showMessage(f"Error updating plots: {exc}")
            _LOG.error("runtime error while updating burst plots", error=str(exc))
        except Exception as exc:
            self._status_bar.showMessage(f"Error updating plots: {exc}")
            _LOG.error("error while updating burst plots", error=str(exc))

    def _show_all_photons(self) -> bool:
        """Return whether all-photon diagnostic layers should be shown."""
        try:
            check = getattr(self, "show_all_photons_check", None)
        except RuntimeError:
            check = None
        if check is None:
            try:
                check = getattr(self, "mcs_show_all_check", None)
            except RuntimeError:
                check = None
        return bool(check is None or check.isChecked())

    def _show_selected_photons(self) -> bool:
        """Return whether selected-photon diagnostic layers should be shown."""
        try:
            check = getattr(self, "show_selected_photons_check", None)
        except RuntimeError:
            check = None
        if check is None:
            try:
                check = getattr(self, "mcs_show_selected_check", None)
            except RuntimeError:
                check = None
        return bool(check is None or check.isChecked())

    def _update_mcs_plot(self, start: int, stop: int) -> None:
        """Update the MCS intensity trace plot."""
        if self._last_tttr is None or self._last_selected is None:
            self.mcs_plot.clear()
            return
        bin_width = float(self.mcs_bin_spin.value()) / 1000.0
        show_all = self._show_all_photons()
        show_selected = self._show_selected_photons()
        self.mcs_plot.clear()
        if not show_all and not show_selected:
            return
        if show_all:
            try:
                range_indices = np.arange(start, stop)
                trace_all = self._last_tttr[range_indices].get_intensity_trace(time_window_length=bin_width)
                self.mcs_plot.plot(
                    np.arange(len(trace_all)) * bin_width,
                    trace_all,
                    pen=pg.mkPen((255, 255, 0, 80), width=1),
                )
            except Exception:
                self.mcs_plot.plot([], [], pen=pg.mkPen((255, 255, 0, 80), width=1))
        if show_selected:
            selected_indices = np.where(self._last_selected[start:stop])[0] + start
            try:
                if selected_indices.size:
                    trace_selected = self._last_tttr[selected_indices].get_intensity_trace(time_window_length=bin_width)
                    self.mcs_plot.plot(
                        np.arange(len(trace_selected)) * bin_width,
                        trace_selected,
                        pen=pg.mkPen("c", width=2),
                    )
                else:
                    self.mcs_plot.plot([], [], pen=pg.mkPen("c", width=2))
            except Exception:
                self.mcs_plot.plot([], [], pen=pg.mkPen("c", width=2))
        self.mcs_plot.setLabel("bottom", "Time (s)")
        self.mcs_plot.setLabel("left", "Intensity")

    def _update_decay_plot(self) -> None:
        """Update the microtime decay plot."""
        if self._last_tttr is None or self._last_selected is None:
            self.decay_plot.clear()
            return
        coarse = int(self.decay_bins_spin.value())
        selected_indices = np.where(self._last_selected)[0]
        show_all = self._show_all_photons()
        show_selected = self._show_selected_photons()
        self.decay_plot.clear()
        if not show_all and not show_selected:
            return
        if show_all:
            try:
                y_all, x_all = self._last_tttr.get_microtime_histogram(coarse)
                self._plot_decay(y_all, x_all, (255, 255, 0, 80))
            except Exception:
                self.decay_plot.plot([], [], pen=pg.mkPen((255, 255, 0, 80), width=1))
        if show_selected:
            try:
                if selected_indices.size:
                    y_selected, x_selected = self._last_tttr[selected_indices].get_microtime_histogram(coarse)
                    self._plot_decay(y_selected, x_selected, "c")
                else:
                    self.decay_plot.plot([], [], pen=pg.mkPen("c", width=1))
            except Exception:
                self.decay_plot.plot([], [], pen=pg.mkPen("c", width=1))

    def _plot_decay(self, y: np.ndarray, x: np.ndarray, pen: Any) -> None:
        """Plot decay histogram data after trimming trailing zeros."""
        positive = np.where(y > 0)[0]
        if positive.size:
            last = int(positive[-1]) + 1
            x = x[:last] * 1e9
            y = y[:last]
            self.decay_plot.plot(x, y, pen=pg.mkPen(pen, width=1))

    def _update_burst_length_plot(self) -> None:
        """Update the burst duration histogram in milliseconds."""
        self.burst_plot.clear()
        if not self._show_selected_photons():
            self.burst_plot.addItem(pg.TextItem("Selected photons hidden", anchor=(0.5, 0.5), color="w"))
            return
        if self._last_start_stop is None or len(self._last_start_stop) == 0:
            self.burst_plot.addItem(pg.TextItem("No burst data available", anchor=(0.5, 0.5), color="w"))
            return
        durations = self._burst_durations_ms()
        if durations.size == 0:
            self.burst_plot.addItem(pg.TextItem("No burst data available", anchor=(0.5, 0.5), color="w"))
            return
        counts, edges = np.histogram(durations, bins=int(self.burst_bins_spin.value()))
        self.burst_plot.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], y0=0, y1=counts, brush="b", pen="w"))
        self.burst_plot.setLabel("bottom", "Burst duration", units="ms")
        self.burst_plot.setLabel("left", "Frequency")
        total = int(np.sum(counts))
        self.burst_plot.addItem(pg.TextItem(f"Total bursts: {total}", anchor=(0, 0), color="w"))

    def _burst_durations_ms(self) -> np.ndarray:
        """Return current burst durations in milliseconds."""
        if self._last_tttr is None or self._last_start_stop is None:
            return np.array([], dtype=float)
        macro_times = np.asarray(self._last_tttr.macro_times)
        if macro_times.size == 0:
            return np.array([], dtype=float)
        resolution_ms = float(self._last_tttr.header.macro_time_resolution) * 1000.0
        durations: list[float] = []
        for start, stop in np.asarray(self._last_start_stop, dtype=int):
            if 0 <= start < stop < macro_times.size:
                durations.append(float(macro_times[stop] - macro_times[start]) * resolution_ms)
        return np.asarray(durations, dtype=float)

    def _valid_histogram_range(self, data: np.ndarray) -> tuple[float, float]:
        """Return valid histogram bounds, falling back to data range."""
        min_value = float(self.hist_min_spin.value())
        max_value = float(self.hist_max_spin.value())
        if min_value >= max_value:
            min_value = float(np.min(data))
            max_value = float(np.max(data))
            if min_value == max_value:
                min_value -= 0.5
                max_value += 0.5
        return min_value, max_value

    def _set_histogram_range_to_data(self) -> None:
        """Set histogram min/max controls to the current feature range."""
        if self._last_frame is None:
            return
        feature = self.feature_combo.currentText()
        if feature not in self._last_frame.columns:
            return
        data = histogram_data_from_frame(self._last_frame, feature)
        if data.size == 0:
            return
        min_value = float(np.min(data))
        max_value = float(np.max(data))
        if min_value == max_value:
            min_value -= 0.5
            max_value += 0.5
        self.hist_min_spin.blockSignals(True)
        self.hist_max_spin.blockSignals(True)
        self.hist_min_spin.setValue(min_value)
        self.hist_max_spin.setValue(max_value)
        self.hist_min_spin.blockSignals(False)
        self.hist_max_spin.blockSignals(False)
        self.update_histogram()

    def _fit_gmm(self) -> None:
        """Fit and plot a GMM once, then return to normal histogram updates."""
        self._fit_gmm_on_update = True
        try:
            self.update_histogram()
        finally:
            self._fit_gmm_on_update = False

    def _plot_gmm(self, data: np.ndarray, min_value: float, max_value: float, counts: np.ndarray) -> None:
        """Fit and plot an optional Gaussian mixture model."""
        try:
            from sklearn.mixture import GaussianMixture
        except Exception as exc:
            self.gmm_summary.setPlainText(f"GMM fitting failed: {exc}")
            return
        n_components = int(self.gmm_components_spin.value())
        if self.gmm_auto_components_check.isChecked() and n_components == 0 and data.size > 1:
            max_components = min(10, data.size)
            bic_scores = []
            for components in range(1, max_components + 1):
                model = GaussianMixture(n_components=components, random_state=42)
                model.fit(data.reshape(-1, 1))
                bic_scores.append(model.bic(data.reshape(-1, 1)))
            n_components = int(np.argmin(bic_scores) + 1)
        if n_components <= 0 or data.size < n_components:
            self.gmm_summary.setPlainText("GMM fitting skipped: not enough data points or zero components.")
            return
        x_fit = np.linspace(min_value, max_value, 200).reshape(-1, 1)
        try:
            model = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42)
            model.fit(data.reshape(-1, 1))
            y_fit = np.exp(model.score_samples(x_fit))
            if np.max(y_fit) > 0:
                scale = float(np.max(counts)) / float(np.max(y_fit))
            else:
                scale = 1.0
            self.histogram_plot.plot(x_fit.ravel(), y_fit * scale, pen=pg.mkPen("r", width=2), name="GMM Fit")
            for index in range(n_components):
                mean = float(model.means_[index, 0])
                variance = float(model.covariances_[index, 0, 0])
                weight = float(model.weights_[index])
                component = weight * np.exp(-0.5 * ((x_fit.ravel() - mean) ** 2) / variance) / np.sqrt(2 * np.pi * variance) * scale
                self.histogram_plot.plot(x_fit.ravel(), component, pen=pg.mkPen(pg.intColor(index, hues=n_components), width=1, style=QtCore.Qt.PenStyle.DashLine), name=f"Gaussian {index + 1}")
            self.histogram_plot.addLegend()
            self.gmm_summary.setPlainText(self._gmm_table(model, n_components))
        except Exception as exc:
            self.gmm_summary.setPlainText(f"GMM fitting failed: {exc}")

    def _gmm_table(self, model: Any, n_components: int) -> str:
        """Return a text table for GMM parameters."""
        lines = ["Gaussian #  Weight  Mean  Std. Dev.", "-" * 36]
        for index in range(n_components):
            std = float(np.sqrt(model.covariances_[index, 0, 0]))
            lines.append(
                f"{index + 1:<10}  {float(model.weights_[index]):>6.3f}  "
                f"{float(model.means_[index, 0]):>8.3f}  {std:>9.3f}"
            )
        return "\n".join(lines)


    def _update_histogram_if_available(self) -> None:
        """Update the histogram only when analysis data is available."""
        if self._last_frame is not None:
            self.update_histogram()

    def _populate_feature_combo(self, frame: pd.DataFrame) -> None:
        """Populate the feature combo from DataFrame columns."""
        current = self.feature_combo.currentText()
        columns = list(frame.columns)
        if columns != [self.feature_combo.itemText(index) for index in range(self.feature_combo.count())]:
            self.feature_combo.blockSignals(True)
            self.feature_combo.clear()
            self.feature_combo.addItems(columns)
            index = self.feature_combo.findText(current)
            self.feature_combo.setCurrentIndex(index if index >= 0 else self.feature_combo.findText("Proximity Ratio"))
            self.feature_combo.blockSignals(False)

    def _refresh_file_list(self) -> None:
        """Refresh the visible file list."""
        self.file_list.blockSignals(True)
        try:
            self.file_list.clear()
            for path in self._file_paths:
                self.file_list.addItem(str(path))
        finally:
            self.file_list.blockSignals(False)
        # Auto-select first file if available (after unblocking signals)
        if self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)

    def _on_file_selected(self) -> None:
        """Handle file selection in the file list and generate plots for the selected file."""
        try:
            _LOG.debug("file selection changed")
            # Use QTimer.singleShot to defer processing to avoid Qt object deletion issues
            QtCore.QTimer.singleShot(0, self._process_file_selection)
        except Exception as exc:
            self._status_bar.showMessage(f"Error scheduling file selection: {exc}")
            _LOG.error("error scheduling file selection", error=str(exc))

    def _process_file_selection(self) -> None:
        """Process the file selection after a small delay."""
        try:
            _LOG.debug("processing file selection")
            if not self.file_list or self.file_list is None:
                _LOG.debug("file selection skipped: file list is missing")
                return
            selected_items = self.file_list.selectedItems()
            if not selected_items:
                _LOG.debug("file selection skipped: no selected items")
                return
            selected_item = selected_items[0]
            if selected_item is None:
                _LOG.debug("file selection skipped: selected item is missing")
                return
            selected_text = selected_item.text()
            if not selected_text:
                _LOG.debug("file selection skipped: selected text is empty")
                return
            selected_path = Path(selected_text)
            _LOG.debug("selected file path resolved", path=str(selected_path))
            if selected_path not in self._file_paths:
                _LOG.debug("file selection skipped: path is not queued", path=str(selected_path))
                return
            # Get current settings from wizard if _last_settings is not set
            settings = self._last_settings
            if settings is None:
                _LOG.debug("using current controls for diagnostic settings")
                settings = self._settings_from_controls()
            _LOG.debug("loading diagnostics for selected file", path=str(selected_path))
            self._status_bar.showMessage(f"Loading diagnostics for {selected_path.name}...")
            self._load_tttr_for_plots(selected_path, settings)
        except Exception as exc:
            self._status_bar.showMessage(f"Error selecting file: {exc}")
            _LOG.error("error selecting file", error=str(exc))

    def _show_file_list_context_menu(self, pos: QtCore.QPoint) -> None:
        """Show context menu for file list."""
        menu = QtWidgets.QMenu(self)

        remove_action = QtWidgets.QAction("Remove selected", self)
        remove_action.triggered.connect(self._remove_selected_files)
        menu.addAction(remove_action)

        clear_action = QtWidgets.QAction("Clear all", self)
        clear_action.triggered.connect(self._clear_file_list)
        menu.addAction(clear_action)

        menu.exec_(self.file_list.mapToGlobal(pos))

    def _remove_selected_files(self) -> None:
        """Remove selected files from the file list."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            path = Path(item.text())
            if path in self._file_paths:
                self._file_paths.remove(path)
        self._refresh_file_list()

    def _clear_file_list(self) -> None:
        """Clear all files from the file list."""
        self._file_paths.clear()
        self._refresh_file_list()

    def _load_tttr_for_plots(self, path: Path, settings: AnalysisSettings) -> None:
        """Load TTTR file and selected mask for diagnostic plots for a specific file."""
        try:
            _LOG.debug("loading TTTR diagnostics", path=str(path))
            settings_dict = asdict(settings) if settings else {}
            _LOG.debug("TTTR diagnostic settings prepared", settings=settings_dict)
            self._status_bar.showMessage(f"Loading diagnostics for {path.name}...")
            diag = self._client.load_diagnostics(path, settings_dict)
            _LOG.debug("TTTR diagnostics loaded", keys=list(diag.keys()) if diag else [])
            if diag and "tttr" in diag:
                self._last_tttr = diag["tttr"]
                self._last_selected = diag["selected"]
                self._last_start_stop = diag["start_stop"]
                self._last_diagnostic_path = path
                self._sync_plot_range_controls(len(self._last_selected), reset=True)
                _LOG.debug("TTTR diagnostics assigned; updating plots", path=str(path))
                self._status_bar.showMessage(f"Updating plots for {path.name}...")
                self.update_burst_plots()
                self._status_bar.showMessage(f"Ready: {path.name}")
            else:
                _LOG.warning("TTTR diagnostics returned no data", path=str(path))
                raise ValueError("diagnostics returned no data")
        except Exception as exc:
            _LOG.error("error loading TTTR diagnostics", path=str(path), error=str(exc))
            self._last_tttr = None
            self._last_selected = None
            self._last_start_stop = None
            self._last_diagnostic_path = None
            self.summary.append(f"Diagnostic plots unavailable for {path}: {exc}")
            self._status_bar.showMessage(f"Error loading {path.name}: {exc}")

    def _fill_table(self, frame: pd.DataFrame) -> None:
        """Fill the table widget from a GUI DataFrame."""
        self.table.setRowCount(len(frame))
        for row_index, row in enumerate(frame.to_numpy()):
            for column_index, value in enumerate(row):
                item = QtWidgets.QTableWidgetItem()
                if isinstance(value, (float, np.floating)):
                    item.setData(QtCore.Qt.ItemDataRole.DisplayRole, float(value))
                else:
                    item.setText(str(value))
                self.table.setItem(row_index, column_index, item)

    def _clear_plots(self) -> None:
        """Clear all result plots."""
        self._clear_burst_plots()
        self.histogram_plot.clear()
        self.gmm_summary.clear()

    def _clear_burst_plots(self) -> None:
        """Clear diagnostic burst plots."""
        try:
            self.dt_plot.clear()
        except RuntimeError:
            pass
        try:
            self.filter_plot.clear()
        except RuntimeError:
            pass
        try:
            self.mcs_plot.clear()
        except RuntimeError:
            pass
        try:
            self.decay_plot.clear()
        except RuntimeError:
            pass
        try:
            self.burst_plot.clear()
        except RuntimeError:
            pass
        self._clear_filter_settings_diagnostics()

    def _update_filter_settings_diagnostics(
        self,
        start: int,
        stop: int,
        indices: np.ndarray,
        selected_slice: np.ndarray,
        d_t: np.ndarray,
    ) -> None:
        """Mirror current file diagnostics into the embedded filter-settings plots.

        Parameters
        ----------
        start : int
            First photon index shown in the diagnostic range.
        stop : int
            One-past-last photon index shown in the diagnostic range.
        indices : numpy.ndarray
            Photon indices for the visible range.
        selected_slice : numpy.ndarray
            Boolean selection mask for the visible range.
        d_t : numpy.ndarray
            Delta macro-time values for the visible range, in milliseconds.

        """
        try:
            wizard = self.wizard
            path = self._last_diagnostic_path
            if path is not None:
                resolved_path = str(path.resolve())
                wizard.tttr_objects[resolved_path] = self._last_tttr
                wizard.settings["tttr_filenames"] = [resolved_path]
                wizard.lineEdit.setText(resolved_path)
                wizard.spinBox_4.blockSignals(True)
                wizard.spinBox_4.setMaximum(0)
                wizard.spinBox_4.setValue(0)
                wizard.spinBox_4.blockSignals(False)

            wizard.tttr = self._last_tttr
            wizard.spinBox_2.blockSignals(True)
            wizard.spinBox_2.setValue(start)
            wizard.spinBox_2.blockSignals(False)
            wizard.spinBox_3.blockSignals(True)
            wizard.spinBox_3.setValue(max(start, stop - 1))
            wizard.spinBox_3.blockSignals(False)

            selected_bool = selected_slice.astype(bool)
            if self._show_all_photons():
                wizard.plot_unselected.setData(x=indices, y=d_t)
            else:
                wizard.plot_unselected.setData([], [])
            if self._show_selected_photons():
                wizard.plot_selected.setData(x=indices[selected_bool], y=d_t[selected_bool])
                wizard.plot_select.setData(x=indices, y=selected_bool.astype(np.uint8))
            else:
                wizard.plot_selected.setData([], [])
                wizard.plot_select.setData([], [])
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return

    def _clear_filter_settings_diagnostics(self) -> None:
        """Clear the embedded filter-settings diagnostic plots."""
        try:
            self.wizard.plot_selected.setData([], [])
            self.wizard.plot_unselected.setData([], [])
            self.wizard.plot_select.setData([], [])
            self.wizard.tttr = None
        except (AttributeError, RuntimeError):
            pass

    def _delta_macro_time_ms(self, tttr: Any) -> np.ndarray:
        """Return delta macro times in milliseconds."""
        macro_times = tttr.macro_times
        d_t = np.diff(macro_times, prepend=macro_times[0])
        return d_t * tttr.header.macro_time_resolution * 1000.0

    def _sync_plot_range_controls(self, n_photons: int, reset: bool = False) -> None:
        """Clamp toolbar photon range controls to the current diagnostic file.

        Parameters
        ----------
        n_photons : int
            Number of photons in the currently selected TTTR file.
        reset : bool
            If ``True``, reset the visible range to the full file.

        """
        if not hasattr(self, "plot_min_spin") or not hasattr(self, "plot_max_spin"):
            return
        last_index = max(0, int(n_photons) - 1)
        min_value = 0 if reset else min(max(0, int(self.plot_min_spin.value())), last_index)
        max_value = last_index if reset else min(max(0, int(self.plot_max_spin.value())), last_index)
        if max_value < min_value:
            max_value = min_value

        for spin, value in ((self.plot_min_spin, min_value), (self.plot_max_spin, max_value)):
            block_signals = getattr(spin, "blockSignals", None)
            if block_signals is not None:
                block_signals(True)
            if hasattr(spin, "setRange"):
                spin.setRange(0, last_index)
            elif hasattr(spin, "setMaximum"):
                spin.setMaximum(last_index)
            if hasattr(spin, "setValue"):
                spin.setValue(value)
            if block_signals is not None:
                block_signals(False)

    def _set_log_dt_range(self, plot: pg.PlotWidget, d_t: np.ndarray) -> None:
        """Set a safe visible range for a log-scale dT plot."""
        positive = d_t[np.isfinite(d_t) & (d_t > 0)]
        if positive.size == 0:
            return
        lower = max(float(np.nanmin(positive)), 1e-12)
        upper = max(float(np.nanmax(positive)), lower * 10.0)
        try:
            plot.setYRange(np.log10(lower), np.log10(upper), padding=0.02)
        except Exception:
            plot.setYRange(lower, upper, padding=0.02)

    def _sync_output_format_controls(self) -> None:
        """Synchronize output-format checkboxes."""
        has_output_format = self.csv_output_check.isChecked() or self.hdf_output_check.isChecked()
        self.zip_output_check.setEnabled(has_output_format)
        if not has_output_format:
            self.zip_output_check.setChecked(False)
        self.remove_folder_check.setEnabled(self.zip_output_check.isChecked())
        if not self.zip_output_check.isChecked():
            self.remove_folder_check.setChecked(False)

    def _focus_gmm_controls(self) -> None:
        """Focus the GMM controls."""
        self.fit_gmm_button.setFocus()

    def _setup_toolbar(self) -> None:
        """Create the main toolbar."""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)

        add_files_action = QtWidgets.QAction("Add", self)
        add_files_action.triggered.connect(self.add_files)
        toolbar.addAction(add_files_action)

        batch_action = QtWidgets.QAction("Batch", self)
        batch_action.triggered.connect(self.open_batch_dialog)
        toolbar.addAction(batch_action)

        toolbar.addSeparator()

        analyze_action = QtWidgets.QAction("Process", self)
        analyze_action.triggered.connect(self.analyze_files)
        toolbar.addAction(analyze_action)

        toolbar.addSeparator()

        clear_action = QtWidgets.QAction("Clear", self)
        clear_action.triggered.connect(self.clear)
        toolbar.addAction(clear_action)

        toolbar.addSeparator()

        refresh_action = QtWidgets.QAction("Refresh", self)
        refresh_action.triggered.connect(self.update_burst_plots)
        toolbar.addAction(refresh_action)

        toolbar.addSeparator()

        # Add spacer to push range controls to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        show_label = QtWidgets.QLabel("Show:")
        self.show_all_photons_check = QtWidgets.QCheckBox("All photons", self)
        self.show_all_photons_check.setChecked(True)
        self.show_all_photons_check.setToolTip("Show diagnostic layers computed from all photons.")
        self.show_selected_photons_check = QtWidgets.QCheckBox("Selected photons", self)
        self.show_selected_photons_check.setChecked(True)
        self.show_selected_photons_check.setToolTip("Show diagnostic layers computed from selected burst photons.")
        # Backwards-compatible aliases for older code paths/tests that used
        # the original MCS-local controls.
        self.mcs_show_all_check = self.show_all_photons_check
        self.mcs_show_selected_check = self.show_selected_photons_check
        toolbar.addWidget(show_label)
        toolbar.addWidget(self.show_all_photons_check)
        toolbar.addWidget(self.show_selected_photons_check)
        toolbar.addSeparator()

        # Add plot range controls to toolbar
        plot_range_label = QtWidgets.QLabel("Photon Range:")
        self.plot_min_spin = QtWidgets.QSpinBox()
        self.plot_min_spin.setRange(0, 99_999_999)
        self.plot_min_spin.setValue(0)
        self.plot_min_spin.setToolTip("Minimum photon index to process (0 = start of file)")
        self.plot_max_spin = QtWidgets.QSpinBox()
        self.plot_max_spin.setRange(0, 99_999_999)
        self.plot_max_spin.setValue(DEFAULT_PLOT_MAX)
        self.plot_max_spin.setToolTip("Maximum photon index to process (default = end of file)")
        toolbar.addWidget(plot_range_label)
        toolbar.addWidget(self.plot_min_spin)
        toolbar.addWidget(QtWidgets.QLabel("to"))
        toolbar.addWidget(self.plot_max_spin)

        # Apply colors to toolbar buttons
        for widget in toolbar.children():
            if isinstance(widget, QtWidgets.QToolButton):
                action = widget.defaultAction()
                if action is None:
                    continue
                text = action.text()
                if text == "Add":
                    widget.setStyleSheet("color: cyan;")
                elif text == "Batch":
                    widget.setStyleSheet("color: orange;")
                elif text == "Process":
                    widget.setStyleSheet("color: blue;")
                elif text == "Clear":
                    widget.setStyleSheet("color: red;")
                elif text == "Refresh":
                    widget.setStyleSheet("color: gray;")

    def _setup_statusbar(self) -> None:
        """Create the status bar."""
        self._status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

    def _show_about(self) -> None:
        """Show about dialog."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("About Burst Selection")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Burst Selection Tool", dialog))
        layout.addWidget(QtWidgets.QLabel("ChiSurf Plugin for burst analysis of TTTR data.", dialog))
        layout.addStretch()
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec_()

    def _show_metadata_dialog(self) -> None:
        """Show metadata dialog for editing analysis metadata."""
        dialog = MetadataDialog(self._metadata, parent=self)
        if dialog.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            self._metadata = dialog.get_metadata()
            self.summary.setPlainText(f"Metadata updated. {len(self._metadata)} entries.")

    def export_bur(self) -> None:
        """Export burst data as .bur file."""
        if not self._last_frame:
            self.summary.setPlainText("No burst data to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export as .bur", "", "BUR files (*.bur);;All files (*)"
        )
        if not path:
            return
        try:
            self._last_frame.to_csv(path, sep="\t", index=False)
            self.summary.setPlainText(f"Exported to {path}")
        except Exception as exc:
            self.summary.setPlainText(f"Export failed: {exc}")

    def export_flr_cif(self) -> None:
        """Export burst data as flrCIF format."""
        if not self._last_frame:
            self.summary.setPlainText("No burst data to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export as flrCIF", "", "CIF files (*.cif *.mmcif);;All files (*)"
        )
        if not path:
            return
        try:
            self._write_flr_cif(path)
            self.summary.setPlainText(f"Exported to {path}")
        except Exception as exc:
            self.summary.setPlainText(f"Export failed: {exc}")

    def _write_flr_cif(self, path: Path) -> None:
        """Write burst data in flrCIF format."""
        lines = [
            "# flrCIF export from Burst Selection Tool",
            "#",
            "data_",
            "",
            "# Analysis metadata",
        ]
        for key, value in sorted(self._metadata.items()):
            lines.append(f"_{key} {value}")
        lines.append("")
        lines.append("# Burst data")
        lines.append("loop_")
        for col in self._last_frame.columns:
            lines.append(f"_{col}")
        for _, row in self._last_frame.iterrows():
            lines.append("\t".join(str(v) for v in row.values))
        Path(path).write_text("\n".join(lines))

    def open_batch_dialog(self) -> None:
        """Open the folder batch dialog."""
        dialog = BatchProcessingDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            self._add_paths(dialog.folders())

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept file URL drops on the main window."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Add dropped files to the analysis queue."""
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.toLocalFile()]
        self._add_paths(paths)
        event.acceptProposedAction()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Save dock layout before closing."""
        self._save_dock_layout()
        super().closeEvent(event)

    def _save_dock_layout(self) -> None:
        """Save the current dock layout to QSettings."""
        try:
            import json
            settings = QtCore.QSettings("chisurf", "BurstSelectionTool")
            layout_state = self.dock_area.get_layout_state()
            settings.setValue("dock_layout", json.dumps(layout_state, sort_keys=True))
            settings.sync()
        except Exception as exc:
            self._status_bar.showMessage(f"Failed to save dock layout: {exc}")

    def _restore_dock_layout(self) -> None:
        """Restore the dock layout from QSettings."""
        try:
            import json
            settings = QtCore.QSettings("chisurf", "BurstSelectionTool")
            value = settings.value("dock_layout")
            if isinstance(value, str):
                layout_state = json.loads(value)
            elif isinstance(value, dict):
                layout_state = value
            else:
                return
            self.dock_area.set_layout_state(layout_state, emit_change=False)
        except Exception as exc:
            self._status_bar.showMessage(f"Failed to restore dock layout: {exc}")
