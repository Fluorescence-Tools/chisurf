"""Panel for configuring labeling positions and viewing Accessible Volumes in 3D using a table interface."""

from __future__ import annotations

from chisurf import logging
import re
import os
from pathlib import Path
from typing import Any

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

import chisurf.core.structure
import chisurf.gui.widgets
from chisurf.plugins.chimol.chimol.renderer.view import MolView

from ..core.colors import DEFAULT_AV_COLOR, normalize_rgba, rgba_to_json
from ..core.mrc import save_av_mrc
from ..core.naming import default_label_name, unique_label_name
from .av_worker import AVWorker

logger = logging.getLogger("chisurf.plugins.modelling.fret")

# Updated default alpha to 0.35 for beautiful default transparency
DISTINGUISHABLE_COLORS = [
    (0.89, 0.10, 0.11, 0.35), # Red
    (0.12, 0.47, 0.71, 0.35), # Blue
    (0.20, 0.63, 0.17, 0.35), # Green
    (1.00, 0.50, 0.00, 0.35), # Orange
    (0.42, 0.24, 0.60, 0.35), # Purple
    (0.69, 0.35, 0.16, 0.35), # Brown
    (0.97, 0.51, 0.75, 0.35), # Pink
    (0.00, 0.75, 0.75, 0.35), # Cyan
    (0.87, 0.87, 0.00, 0.35), # Yellow
    (0.50, 0.50, 0.50, 0.35), # Gray
]


class CenteredCheckBox(QtWidgets.QWidget):
    """A wrapper widget that centers a QCheckBox inside a table cell."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        self.checkbox = QtWidgets.QCheckBox()
        layout.addWidget(self.checkbox)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)


class PdbSelectWidget(QtWidgets.QWidget):
    """A widget for entering PDB ID or selecting a file path."""

    pdbChanged = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.line_edit = QtWidgets.QLineEdit()
        self.btn = QtWidgets.QToolButton()
        self.btn.setText("...")
        self.btn.clicked.connect(self.on_browse)

        layout.addWidget(self.line_edit, stretch=1)
        layout.addWidget(self.btn)

        self.line_edit.editingFinished.connect(self.on_editing_finished)

    def on_browse(self) -> None:
        filename = chisurf.gui.widgets.get_filename(
            'Open PDB-File',
            'PDB-Files (*.pdb);;PDB-GZ (*.pdb.gz)'
        )
        if filename:
            self.line_edit.setText(filename)
            self.pdbChanged.emit(filename)

    def on_editing_finished(self) -> None:
        self.pdbChanged.emit(self.line_edit.text().strip())


class DetailSettingsDialog(QtWidgets.QDialog):
    """Modal dialog for editing detailed AV/Dye parameters."""

    def __init__(self, params: dict, dye_model: str = "AV1", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Labeling Site Details")
        self.resize(400, 520)

        main_layout = QtWidgets.QVBoxLayout(self)

        # Dimensions Group
        dim_group = QtWidgets.QGroupBox("Dye Dimensions")
        dim_layout = QtWidgets.QFormLayout(dim_group)

        self.l_spin = QtWidgets.QDoubleSpinBox()
        self.l_spin.setRange(0.0, 200.0)
        self.l_spin.setValue(float(params.get("linker_length", 20.0)))
        dim_layout.addRow("Linker Length (L):", self.l_spin)

        self.w_spin = QtWidgets.QDoubleSpinBox()
        self.w_spin.setRange(0.0, 50.0)
        self.w_spin.setValue(float(params.get("linker_width", 4.5)))
        dim_layout.addRow("Linker Width (W):", self.w_spin)

        self.r1_spin = QtWidgets.QDoubleSpinBox()
        self.r1_spin.setRange(0.0, 50.0)
        self.r1_spin.setValue(float(params.get("radius1", 3.5)))
        dim_layout.addRow("Radius 1 (R1):", self.r1_spin)

        self.r2_spin = QtWidgets.QDoubleSpinBox()
        self.r2_spin.setRange(0.0, 50.0)
        self.r2_spin.setValue(float(params.get("radius2", 0.0)))
        dim_layout.addRow("Radius 2 (R2):", self.r2_spin)

        self.r3_spin = QtWidgets.QDoubleSpinBox()
        self.r3_spin.setRange(0.0, 50.0)
        self.r3_spin.setValue(float(params.get("radius3", 0.0)))
        dim_layout.addRow("Radius 3 (R3):", self.r3_spin)

        # Disable R2/R3 if not AV3
        is_av3 = (dye_model == "AV3")
        self.r2_spin.setEnabled(is_av3)
        self.r3_spin.setEnabled(is_av3)

        main_layout.addWidget(dim_group)

        # Simulation Parameters Group
        sim_group = QtWidgets.QGroupBox("Simulation Settings")
        sim_layout = QtWidgets.QFormLayout(sim_group)

        self.body_id_spin = QtWidgets.QSpinBox()
        self.body_id_spin.setRange(0, 10000)
        self.body_id_spin.setValue(int(params.get("body_id", 0)))
        sim_layout.addRow("Body ID:", self.body_id_spin)

        self.allowed_sphere_spin = QtWidgets.QDoubleSpinBox()
        self.allowed_sphere_spin.setRange(0.0, 50.0)
        self.allowed_sphere_spin.setValue(float(params.get("allowed_sphere_radius", 1.5)))
        sim_layout.addRow("Allowed Sphere Radius:", self.allowed_sphere_spin)

        self.res_spin = QtWidgets.QDoubleSpinBox()
        self.res_spin.setRange(0.1, 10.0)
        self.res_spin.setSingleStep(0.1)
        self.res_spin.setValue(float(params.get("simulation_grid_resolution", 1.5)))
        sim_layout.addRow("Grid Resolution:", self.res_spin)

        self.anchor_edit = QtWidgets.QLineEdit()
        self.anchor_edit.setText(str(params.get("anchor_atoms", "")))
        sim_layout.addRow("Anchor Atoms:", self.anchor_edit)

        self.strip_edit = QtWidgets.QLineEdit()
        self.strip_edit.setText(str(params.get("strip_mask", "")))
        sim_layout.addRow("Strip Mask:", self.strip_edit)

        main_layout.addWidget(sim_group)

        # Advanced/Contact Volume Group
        adv_group = QtWidgets.QGroupBox("Advanced Settings")
        adv_layout = QtWidgets.QFormLayout(adv_group)

        self.chain_weighting_chk = QtWidgets.QCheckBox()
        self.chain_weighting_chk.setChecked(bool(params.get("chain_weighting", False)))
        adv_layout.addRow("Chain Weighting:", self.chain_weighting_chk)

        self.contact_thick_spin = QtWidgets.QDoubleSpinBox()
        self.contact_thick_spin.setRange(0.0, 50.0)
        self.contact_thick_spin.setValue(float(params.get("contact_volume_thickness", 0.0)))
        adv_layout.addRow("Contact Vol Thickness:", self.contact_thick_spin)

        self.contact_trap_spin = QtWidgets.QDoubleSpinBox()
        self.contact_trap_spin.setRange(-1.0, 1.0)
        self.contact_trap_spin.setSingleStep(0.05)
        self.contact_trap_spin.setValue(float(params.get("contact_volume_trapped_fraction", -1.0)))
        adv_layout.addRow("Contact Vol Trapped Frac:", self.contact_trap_spin)

        self.min_sphere_spin = QtWidgets.QDoubleSpinBox()
        self.min_sphere_spin.setRange(0.0, 1.0)
        self.min_sphere_spin.setSingleStep(0.05)
        self.min_sphere_spin.setValue(float(params.get("min_sphere_volume_fraction", 0.0)))
        adv_layout.addRow("Min Sphere Vol Frac:", self.min_sphere_spin)

        main_layout.addWidget(adv_group)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def get_settings(self) -> dict:
        return {
            "linker_length": self.l_spin.value(),
            "linker_width": self.w_spin.value(),
            "radius1": self.r1_spin.value(),
            "radius2": self.r2_spin.value(),
            "radius3": self.r3_spin.value(),
            "body_id": self.body_id_spin.value(),
            "allowed_sphere_radius": self.allowed_sphere_spin.value(),
            "simulation_grid_resolution": self.res_spin.value(),
            "anchor_atoms": self.anchor_edit.text().strip(),
            "strip_mask": self.strip_edit.text().strip(),
            "chain_weighting": self.chain_weighting_chk.isChecked(),
            "contact_volume_thickness": self.contact_thick_spin.value(),
            "contact_volume_trapped_fraction": self.contact_trap_spin.value(),
            "min_sphere_volume_fraction": self.min_sphere_spin.value(),
        }


class PositionPanel(QtWidgets.QWidget):
    """A widget for selecting labeling positions on PDB structures using a QTableWidget.

    Integrates a QTableWidget for listing and editing positions/dyes,
    with Chain/Residue/Atom dropdowns directly embedded in each row of the table.
    """

    position_added = QtCore.Signal(str, dict)
    position_removed = QtCore.Signal(str)

    def _show_status(self, msg: str, level: str = "info"):
        if level == "info":
            logger.info(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        win = self.window()
        if win and hasattr(win, "statusBar") and win.statusBar() is not None:
            win.statusBar().showMessage(msg, 5000)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        client: Any | None = None,
        mol_view_3d: MolView | None = None,
    ) -> None:
        super().__init__(parent)
        self._client = client or self._make_default_client()
        self.mol_view_3d = mol_view_3d or MolView()
        self._pdb_path: str | None = None
        self._positions: dict[str, dict[str, Any]] = {}
        self._av_cache: dict[
            str,
            tuple[np.ndarray, np.ndarray, float, tuple[float, float, float, float]],
        ] = {}

        self._block_table_signals = 0
        self._block_selector_sync = False
        self._loaded_structures: dict[str, chisurf.core.structure.Structure] = {}
        self._row_visibility: dict[str, bool] = {}
        self._row_colors: dict[str, tuple[float, float, float, float]] = {}
        self._av_signatures: dict[str, tuple[object, ...]] = {}
        self._active_workers: dict[str, AVWorker] = {}
        self._av_timers: dict[str, QtCore.QTimer] = {}

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)



        # Toolbar for panel actions
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setObjectName("fpsJsonEditorMainToolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        self.toolbar.setContentsMargins(4, 2, 4, 2)
        if self.toolbar.layout() is not None:
            self.toolbar.layout().setSpacing(6)
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolbar.setStyleSheet(
            """
            QToolBar#fpsJsonEditorMainToolbar {
                background-color: transparent;
                border: none;
                padding: 3px 4px;
                spacing: 6px;
            }
            QToolBar#fpsJsonEditorMainToolbar::separator {
                width: 8px;
            }
            QToolBar#fpsJsonEditorMainToolbar QToolButton {
                background-color: rgba(45, 45, 45, 210);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 4px;
                padding: 4px 8px;
                margin: 0px;
            }
            QToolBar#fpsJsonEditorMainToolbar QToolButton:hover {
                background-color: rgba(70, 70, 70, 230);
            }
            QToolBar#fpsJsonEditorMainToolbar QToolButton:pressed {
                background-color: rgba(90, 90, 90, 240);
            }
            QToolBar#fpsJsonEditorMainToolbar #toolbarAddRow { color: #7de3ff; }
            QToolBar#fpsJsonEditorMainToolbar #toolbarComputeAV { color: #ffb347; }
            QToolBar#fpsJsonEditorMainToolbar #toolbarSaveMRC { color: #a8ff9e; }
            """
        )

        name_map = {
            "➕ Add Row": "toolbarAddRow",
            "🚀 Compute AVs": "toolbarComputeAV",
            "💾 Save AV MRC": "toolbarSaveMRC",
        }
        
        self.add_row_action = self.toolbar.addAction("➕ Add Row")
        self.add_row_action.triggered.connect(self.onAddRowTriggered)

        self.compute_avs_action = self.toolbar.addAction("🚀 Compute AVs")
        self.compute_avs_action.triggered.connect(self.onComputeAVAll)

        self.save_av_mrc_action = self.toolbar.addAction("💾 Save AV MRC")
        self.save_av_mrc_action.setToolTip("Save selected computed AV(s) as MRC density map(s)")
        self.save_av_mrc_action.triggered.connect(self.onSaveSelectedAVsAsMRC)
        
        for widget in self.toolbar.children():
            if isinstance(widget, QtWidgets.QToolButton):
                action = widget.defaultAction()
                if action is None:
                    continue
                obj_name = name_map.get(action.text())
                if obj_name:
                    widget.setObjectName(obj_name)
                    widget.setAutoRaise(True)

        main_layout.addWidget(self.toolbar)

        # Table Widget
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "Show", "Name", "PDB (File/ID)", "Chain", "Res", "Atom", "Dye Preset", "Dye Model", "Settings", "Color", ""
        ])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(10, QtWidgets.QHeaderView.Fixed)
        self.table.setColumnWidth(10, 40)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        
        self.table.cellChanged.connect(self.onCellChanged)
        self.table.verticalHeader().sectionDoubleClicked.connect(self.onRowHeaderDoubleClicked)
        self.table.doubleClicked.connect(self.onTableDoubleClicked)
        self.table.itemSelectionChanged.connect(self.onTableSelectionChanged)

        # Context menu for deleting rows
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.onTableContextMenu)

        main_layout.addWidget(self.table, stretch=1)

        # Progress indicator and preview details
        self.av_preview_label = QtWidgets.QLabel("AV: Not computed")
        self.av_preview_label.setWordWrap(True)
        main_layout.addWidget(self.av_preview_label)

        self.av_progress_bar = QtWidgets.QProgressBar()
        self.av_progress_bar.setRange(0, 0)
        self.av_progress_bar.setVisible(False)
        main_layout.addWidget(self.av_progress_bar)

        self.mol_view_3d.atomSelectionChanged.connect(
            self.on_chimol_atom_selection_changed
        )

    @staticmethod
    def _make_default_client():
        """Create a default FpsJsonEditorClient with local in-process services."""
        from ..api.client import FpsJsonEditorClient
        return FpsJsonEditorClient()

    def _setup_row_widgets(self, row: int) -> None:
        self._block_table_signals += 1
        try:
            # Show Checkbox (col 0)
            cb_widget = CenteredCheckBox()
            cb_widget.checkbox.setChecked(True)
            cb_widget.checkbox.toggled.connect(self.onRowVisibilityToggled)
            self.table.setCellWidget(row, 0, cb_widget)

            # Name (col 1)
            name_item = QtWidgets.QTableWidgetItem("")
            default_params = {
                "linker_length": 20.0,
                "linker_width": 4.5,
                "radius1": 3.5,
                "radius2": 0.0,
                "radius3": 0.0,
                "body_id": 0,
                "allowed_sphere_radius": 1.5,
                "simulation_grid_resolution": 1.5,
                "anchor_atoms": "",
                "strip_mask": "",
                "contact_volume_thickness": 0.0,
                "contact_volume_trapped_fraction": -1.0,
                "min_sphere_volume_fraction": 0.0,
                "chain_weighting": False,
            }
            name_item.setData(QtCore.Qt.UserRole + 1, default_params)
            self.table.setItem(row, 1, name_item)

            # PDB path/ID (col 2)
            pdb_widget = PdbSelectWidget()
            pdb_widget.pdbChanged.connect(lambda val, r=row: self.onRowPdbChanged(r, val))
            self.table.setCellWidget(row, 2, pdb_widget)

            # Chain combobox (col 3) (editable dropdown)
            chain_cb = QtWidgets.QComboBox()
            chain_cb.setEditable(True)
            chain_cb.currentTextChanged.connect(self.onRowChainChanged)
            self.table.setCellWidget(row, 3, chain_cb)

            # Res combobox (col 4) (editable dropdown)
            res_cb = QtWidgets.QComboBox()
            res_cb.setEditable(True)
            res_cb.currentTextChanged.connect(self.onRowResidueChanged)
            self.table.setCellWidget(row, 4, res_cb)

            # Atom combobox (col 5) (editable dropdown)
            atom_cb = QtWidgets.QComboBox()
            atom_cb.setEditable(True)
            atom_cb.currentTextChanged.connect(self.onRowAtomChanged)
            self.table.setCellWidget(row, 5, atom_cb)

            # Dye Preset combobox (col 6)
            preset_cb = QtWidgets.QComboBox()
            preset_cb.addItems(list(chisurf.core.structure.av.dye_names))
            preset_cb.addItem("Custom")
            preset_cb.setCurrentText("Custom")
            preset_cb.currentTextChanged.connect(self.onDyePresetChanged)
            self.table.setCellWidget(row, 6, preset_cb)

            # Dye Model combobox (col 7)
            model_cb = QtWidgets.QComboBox()
            model_cb.addItems(["AV1", "AV0", "AV3"])
            model_cb.setCurrentText("AV1")
            model_cb.currentTextChanged.connect(self.onDyeModelChanged)
            self.table.setCellWidget(row, 7, model_cb)

            # Details Settings Button (col 8)
            settings_btn = QtWidgets.QPushButton("Details...")
            settings_btn.clicked.connect(self.onShowRowSettings)
            self.table.setCellWidget(row, 8, settings_btn)
            self._update_settings_tooltip(row, default_params)

            # Color Button (col 9)
            color_btn = QtWidgets.QPushButton()
            color_btn.setFixedWidth(40)
            color_btn.clicked.connect(self.onChooseRowColor)
            self.table.setCellWidget(row, 9, color_btn)

            # Delete Button (col 10)
            del_btn = QtWidgets.QPushButton("🗑️")
            del_btn.setFixedWidth(30)
            del_btn.clicked.connect(self.onDeleteRowClicked)
            self.table.setCellWidget(row, 10, del_btn)

        finally:
            self._block_table_signals -= 1

    def _update_settings_tooltip(self, row: int, params: dict) -> None:
        btn = self.table.cellWidget(row, 8)
        if isinstance(btn, QtWidgets.QPushButton):
            tooltip_lines = [
                "<b>AV/Dye Parameter Details:</b>",
                f"Linker Length (L): {params.get('linker_length', 20.0)} Å",
                f"Linker Width (W): {params.get('linker_width', 4.5)} Å",
                f"Radius 1 (R1): {params.get('radius1', 3.5)} Å",
                f"Radius 2 (R2): {params.get('radius2', 0.0)} Å",
                f"Radius 3 (R3): {params.get('radius3', 0.0)} Å",
                f"Body ID: {params.get('body_id', 0)}",
                f"Allowed Sphere Rad: {params.get('allowed_sphere_radius', 1.5)} Å",
                f"Grid Resolution: {params.get('simulation_grid_resolution', 1.5)} Å",
                f"Chain Weighting: {params.get('chain_weighting', False)}",
                f"Contact Vol Thickness: {params.get('contact_volume_thickness', 0.0)} Å",
                f"Contact Vol Trapped Frac: {params.get('contact_volume_trapped_fraction', -1.0)}",
                f"Min Sphere Vol Frac: {params.get('min_sphere_volume_fraction', 0.0)}",
            ]
            anchor = params.get('anchor_atoms', '')
            if anchor:
                tooltip_lines.append(f"Anchor Atoms: {anchor}")
            mask = params.get('strip_mask', '')
            if mask:
                tooltip_lines.append(f"Strip Mask: {mask}")

            # Include AV computation results in tooltip if available
            name_item = self.table.item(row, 1)
            if name_item:
                name = name_item.text().strip()
                if name:
                    if "av_volume" in params:
                        tooltip_lines.append("<hr><b>AV Computation Results:</b>")
                        tooltip_lines.append(f"Volume: {params['av_volume']:.1f} Å³")
                        tooltip_lines.append(f"Number of Points: {params.get('av_points', 0)}")
                        mean_xyz = params.get('av_mean', [0.0, 0.0, 0.0])
                        tooltip_lines.append(f"Mean Position (XYZ): ({mean_xyz[0]:.2f}, {mean_xyz[1]:.2f}, {mean_xyz[2]:.2f})")
                    elif name in self._av_cache:
                        coords, mean_xyz, grid_step, color = self._av_cache[name]
                        volume = len(coords) * (grid_step ** 3)
                        tooltip_lines.append("<hr><b>AV Computation Results:</b>")
                        tooltip_lines.append(f"Volume: {volume:.1f} Å³")
                        tooltip_lines.append(f"Number of Points: {len(coords)}")
                        tooltip_lines.append(f"Mean Position (XYZ): ({mean_xyz[0]:.2f}, {mean_xyz[1]:.2f}, {mean_xyz[2]:.2f})")

            btn.setToolTip("<br>".join(tooltip_lines))

    def _add_empty_row(self) -> None:
        self.table.blockSignals(True)
        try:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._setup_row_widgets(row)
        finally:
            self.table.blockSignals(False)

    def onAddRowTriggered(self) -> None:
        self._add_empty_row()
        new_row = self.table.rowCount() - 1
        self.table.setCurrentCell(new_row, 1)
        self._maybe_auto_fill_name(new_row)
        name_item = self.table.item(new_row, 1)
        if name_item:
            self.table.editItem(name_item)

    def _find_widget_row(self, widget: QtWidgets.QWidget, col: int) -> int:
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, col) is widget:
                return r
        return -1

    def _get_row_pdb_val(self, row: int) -> str:
        w = self.table.cellWidget(row, 2)
        if isinstance(w, PdbSelectWidget):
            return w.line_edit.text().strip()
        return ""

    def _get_row_structure(self, row: int) -> chisurf.core.structure.Structure | None:
        pdb_val = self._get_row_pdb_val(row)
        if pdb_val:
            return self._loaded_structures.get(pdb_val)
        return None

    def _existing_label_names(self, row: int | None = None) -> set[str]:
        """Return non-empty label names, optionally excluding one row."""
        existing_names = set()
        for r in range(self.table.rowCount()):
            if row is not None and r == row:
                continue
            item = self.table.item(r, 1)
            if item:
                name = item.text().strip()
                if name:
                    existing_names.add(name)
        return existing_names

    def _ensure_row_style_state(self, row: int, name: str) -> None:
        """Ensure generated and manually entered labels get the same row state."""
        if not name:
            return

        if name not in self._row_colors:
            self._row_colors[name] = DISTINGUISHABLE_COLORS[row % len(DISTINGUISHABLE_COLORS)]
        self._row_visibility.setdefault(name, True)

        color = self._row_colors[name]
        color_btn = self.table.cellWidget(row, 9)
        if isinstance(color_btn, QtWidgets.QPushButton):
            color_btn.setStyleSheet(
                "background-color: "
                f"rgba({int(color[0]*255)}, {int(color[1]*255)}, "
                f"{int(color[2]*255)}, {int(color[3]*255)});"
            )

        self._update_row_colors(row, self._row_visibility.get(name, True))

    def _ensure_trailing_empty_row(self) -> None:
        """Append a trailing empty row after the last row receives a label."""
        if self.table.rowCount() == 0:
            self._add_empty_row()
            return

        last_row = self.table.rowCount() - 1
        name_item = self.table.item(last_row, 1)
        if name_item and name_item.text().strip():
            self._add_empty_row()

    def _set_auto_name_for_row(self, row: int) -> str:
        """Fill an empty row label from chain/residue and return the label."""
        name_item = self.table.item(row, 1)
        if not name_item:
            return ""

        current_name = name_item.text().strip()
        if current_name:
            return current_name

        chain_cb = self.table.cellWidget(row, 3)
        res_cb = self.table.cellWidget(row, 4)
        chain = chain_cb.currentText() if isinstance(chain_cb, QtWidgets.QComboBox) else ""
        res = res_cb.currentText() if isinstance(res_cb, QtWidgets.QComboBox) else ""
        base_name = default_label_name(chain, res)
        if not base_name:
            return ""

        new_name = unique_label_name(base_name, self._existing_label_names(row))
        self._block_table_signals += 1
        self.table.blockSignals(True)
        try:
            name_item.setText(new_name)
            name_item.setData(QtCore.Qt.UserRole, new_name)
        finally:
            self.table.blockSignals(False)
            self._block_table_signals -= 1
        self._ensure_row_style_state(row, new_name)
        self._ensure_trailing_empty_row()
        return new_name

    def _maybe_auto_fill_name(self, row: int) -> None:
        self._set_auto_name_for_row(row)

    def onRowChainChanged(self, chain: str) -> None:
        sender = self.sender()
        if not sender or self._block_table_signals > 0:
            return
        row = self._find_widget_row(sender, 3)
        if row < 0 or self._block_selector_sync:
            return
            
        struct = self._get_row_structure(row)
        if struct:
            self._update_row_residues(row, struct, chain)
            
        self._maybe_auto_fill_name(row)
        self._trigger_row_av(row)
        self._trigger_row_change(row)

    def onRowResidueChanged(self, res_text: str) -> None:
        sender = self.sender()
        if not sender or self._block_table_signals > 0:
            return
        row = self._find_widget_row(sender, 4)
        if row < 0 or self._block_selector_sync:
            return
            
        struct = self._get_row_structure(row)
        chain_cb = self.table.cellWidget(row, 3)
        chain = chain_cb.currentText() if isinstance(chain_cb, QtWidgets.QComboBox) else ""
        
        if struct and chain and res_text.isdigit():
            self._update_row_atoms(row, struct, chain, int(res_text))
            
        self._maybe_auto_fill_name(row)
        self._trigger_row_av(row)
        self._trigger_row_change(row)

    def onRowAtomChanged(self, atom: str) -> None:
        sender = self.sender()
        if not sender or self._block_table_signals > 0:
            return
        row = self._find_widget_row(sender, 5)
        if row < 0 or self._block_selector_sync:
            return
            
        self._maybe_auto_fill_name(row)
        self._trigger_row_av(row)
        self._trigger_row_change(row)

    def _update_row_chains(self, row: int, struct: chisurf.core.structure.Structure) -> None:
        chain_cb = self.table.cellWidget(row, 3)
        if not isinstance(chain_cb, QtWidgets.QComboBox):
            return
            
        curr_text = chain_cb.currentText()
        chain_cb.blockSignals(True)
        chain_cb.clear()
        if struct and struct.atoms is not None:
            chains = sorted(list(set(struct.atoms['chain'])))
            chain_cb.addItems([str(c) for c in chains])
        chain_cb.blockSignals(False)
        
        if curr_text:
            chain_cb.setCurrentText(curr_text)
        else:
            chain_cb.setCurrentIndex(0)

    def _update_row_residues(self, row: int, struct: chisurf.core.structure.Structure, chain: str) -> None:
        res_cb = self.table.cellWidget(row, 4)
        if not isinstance(res_cb, QtWidgets.QComboBox):
            return
            
        curr_text = res_cb.currentText()
        res_cb.blockSignals(True)
        res_cb.clear()
        if struct and struct.atoms is not None and chain:
            atom_ids = np.where(struct.atoms['chain'] == chain)[0]
            res_ids = sorted(list(set(struct.atoms['res_id'][atom_ids])))
            res_cb.addItems([str(r) for r in res_ids])
        res_cb.blockSignals(False)
        
        if curr_text:
            res_cb.setCurrentText(curr_text)
        else:
            res_cb.setCurrentIndex(0)

    def _update_row_atoms(self, row: int, struct: chisurf.core.structure.Structure, chain: str, res: int) -> None:
        atom_cb = self.table.cellWidget(row, 5)
        if not isinstance(atom_cb, QtWidgets.QComboBox):
            return
            
        curr_text = atom_cb.currentText()
        atom_cb.blockSignals(True)
        atom_cb.clear()
        if struct and struct.atoms is not None and chain and res:
            atom_ids = np.where((struct.atoms['res_id'] == res) & (struct.atoms['chain'] == chain))[0]
            atom_names = sorted(list(set(struct.atoms['atom_name'][atom_ids])))
            atom_cb.addItems([str(a) for a in atom_names])
        atom_cb.blockSignals(False)
        
        if curr_text:
            atom_cb.setCurrentText(curr_text)
        else:
            idx = atom_cb.findText("CB")
            if idx >= 0:
                atom_cb.setCurrentIndex(idx)
            else:
                idx = atom_cb.findText("CA")
                if idx >= 0:
                    atom_cb.setCurrentIndex(idx)
                else:
                    atom_cb.setCurrentIndex(0)

    def onRowPdbChanged(self, row: int, val: str) -> None:
        if self._block_table_signals > 0:
            return
        struct = self._load_pdb_for_val(val)
        if struct:
            self._update_row_chains(row, struct)
            chain_cb = self.table.cellWidget(row, 3)
            chain = chain_cb.currentText() if isinstance(chain_cb, QtWidgets.QComboBox) else ""
            self._update_row_residues(row, struct, chain)
            res_cb = self.table.cellWidget(row, 4)
            res_text = res_cb.currentText() if isinstance(res_cb, QtWidgets.QComboBox) else ""
            self._update_row_atoms(row, struct, chain, int(res_text) if res_text.isdigit() else 0)
            self._maybe_auto_fill_name(row)
        self._trigger_row_av(row)
        self._trigger_row_change(row)

    def clear_all(self) -> None:
        """Clear all internal caches, overlays, and structures from the 3D viewer."""
        if self.mol_view_3d is None:
            return

        if hasattr(self.mol_view_3d, "clear_point_overlays"):
            self.mol_view_3d.clear_point_overlays()
            
        if hasattr(self.mol_view_3d, "list_objects") and hasattr(self.mol_view_3d, "remove_object"):
            for obj in self.mol_view_3d.list_objects():
                self.mol_view_3d.remove_object(obj["id"])
                
        self._av_cache.clear()
        self._av_signatures.clear()
        self._loaded_structures.clear()
        self._row_colors.clear()
        self._row_visibility.clear()
        self._pdb_path = ""

    def update_positions(self, positions: dict[str, dict[str, Any]]) -> None:
        """Update the QTableWidget with current positions."""
        if self._block_table_signals > 0:
            return

        self._block_table_signals += 1
        try:
            self._positions = dict(positions)
            self.table.setRowCount(0)

            for name, params in self._positions.items():
                row = self.table.rowCount()
                self.table.insertRow(row)
                self._setup_row_widgets(row)

                # Name
                name_item = self.table.item(row, 1)
                if name_item:
                    name_item.setText(name)
                    name_item.setData(QtCore.Qt.UserRole, name)
                    # Merge with default parameters to preserve everything
                    full_params = {
                        "linker_length": float(params.get("linker_length", 20.0)),
                        "linker_width": float(params.get("linker_width", 4.5)),
                        "radius1": float(params.get("radius1", 3.5)),
                        "radius2": float(params.get("radius2", 0.0)),
                        "radius3": float(params.get("radius3", 0.0)),
                        "body_id": int(params.get("body_id", 0)),
                        "allowed_sphere_radius": float(params.get("allowed_sphere_radius", 1.5)),
                        "simulation_grid_resolution": float(params.get("simulation_grid_resolution", 1.5)),
                        "anchor_atoms": str(params.get("anchor_atoms", "")),
                        "strip_mask": str(params.get("strip_mask", "")),
                        "chain_weighting": bool(params.get("chain_weighting", False)),
                        "contact_volume_thickness": float(params.get("contact_volume_thickness", 0.0)),
                        "contact_volume_trapped_fraction": float(params.get("contact_volume_trapped_fraction", -1.0)),
                        "min_sphere_volume_fraction": float(params.get("min_sphere_volume_fraction", 0.0)),
                    }
                    name_item.setData(QtCore.Qt.UserRole + 1, full_params)
                    self._update_settings_tooltip(row, full_params)

                # PDB path/ID
                pdb_val = params.get("pdb_path") or params.get("pdb_id") or ""
                if not pdb_val and self._pdb_path:
                    pdb_val = self._pdb_path
                pdb_widget = self.table.cellWidget(row, 2)
                if isinstance(pdb_widget, PdbSelectWidget):
                    pdb_widget.line_edit.blockSignals(True)
                    pdb_widget.line_edit.setText(pdb_val)
                    pdb_widget.line_edit.blockSignals(False)

                # Load structure to populate Chain, Res, Atom dropdowns
                struct = self._load_pdb_for_val(pdb_val)

                chain = str(params.get("chain_identifier", ""))
                res = str(params.get("residue_seq_number", ""))
                atom = str(params.get("atom_name", ""))

                chain_cb = self.table.cellWidget(row, 3)
                res_cb = self.table.cellWidget(row, 4)
                atom_cb = self.table.cellWidget(row, 5)

                if struct:
                    self._update_row_chains(row, struct)
                    if isinstance(chain_cb, QtWidgets.QComboBox):
                        chain_cb.setCurrentText(chain)
                        
                    self._update_row_residues(row, struct, chain)
                    if isinstance(res_cb, QtWidgets.QComboBox):
                        res_cb.setCurrentText(res)
                        
                    self._update_row_atoms(row, struct, chain, int(res) if res.isdigit() else 0)
                    if isinstance(atom_cb, QtWidgets.QComboBox):
                        atom_cb.setCurrentText(atom)
                else:
                    if isinstance(chain_cb, QtWidgets.QComboBox):
                        chain_cb.setCurrentText(chain)
                    if isinstance(res_cb, QtWidgets.QComboBox):
                        res_cb.setCurrentText(res)
                    if isinstance(atom_cb, QtWidgets.QComboBox):
                        atom_cb.setCurrentText(atom)

                # Dye Preset
                preset_name = params.get("dye_preset", "Custom")
                preset_cb = self.table.cellWidget(row, 6)
                if isinstance(preset_cb, QtWidgets.QComboBox):
                    idx = preset_cb.findText(preset_name)
                    if idx >= 0:
                        preset_cb.setCurrentIndex(idx)

                # Dye Model
                model_name = params.get("simulation_type", "AV1")
                model_cb = self.table.cellWidget(row, 7)
                if isinstance(model_cb, QtWidgets.QComboBox):
                    idx = model_cb.findText(model_name)
                    if idx >= 0:
                        model_cb.setCurrentIndex(idx)

                # Visibility
                visible = params.get("visible", True)
                self._row_visibility[name] = visible
                cb_widget = self.table.cellWidget(row, 0)
                if isinstance(cb_widget, CenteredCheckBox):
                    cb_widget.checkbox.blockSignals(True)
                    cb_widget.checkbox.setChecked(visible)
                    cb_widget.checkbox.blockSignals(False)

                # Color
                color_val = params.get("av_color", DEFAULT_AV_COLOR)
                color_rgba = normalize_rgba(color_val)
                self._row_colors[name] = color_rgba

                # Color button (col 9)
                color_btn = self.table.cellWidget(row, 9)
                if isinstance(color_btn, QtWidgets.QPushButton):
                    color_btn.setStyleSheet(f"background-color: rgba({int(color_rgba[0]*255)}, {int(color_rgba[1]*255)}, {int(color_rgba[2]*255)}, {int(color_rgba[3]*255)});")

                self._update_row_colors(row, visible)

            self._add_empty_row()

        finally:
            self._block_table_signals -= 1

        # Compute AVs for all populated rows
        for r in range(self.table.rowCount() - 1):
            self._trigger_row_av(r)

    def _trigger_row_change(self, row: int) -> None:
        name_item = self.table.item(row, 1)
        if not name_item:
            return
        name = name_item.text().strip()
        if not name:
            return

        params = name_item.data(QtCore.Qt.UserRole + 1)
        if not isinstance(params, dict):
            params = {}

        try:
            pdb_val = self._get_row_pdb_val(row)

            chain_cb = self.table.cellWidget(row, 3)
            chain = chain_cb.currentText().strip() if isinstance(chain_cb, QtWidgets.QComboBox) else ""

            res_cb = self.table.cellWidget(row, 4)
            res_text = res_cb.currentText().strip() if isinstance(res_cb, QtWidgets.QComboBox) else ""
            res = int(res_text) if res_text.isdigit() else 0

            atom_cb = self.table.cellWidget(row, 5)
            atom = atom_cb.currentText().strip() if isinstance(atom_cb, QtWidgets.QComboBox) else ""

            preset_cb = self.table.cellWidget(row, 6)
            preset = preset_cb.currentText() if isinstance(preset_cb, QtWidgets.QComboBox) else "Custom"

            model_cb = self.table.cellWidget(row, 7)
            model = model_cb.currentText() if isinstance(model_cb, QtWidgets.QComboBox) else "AV1"
        except (ValueError, AttributeError):
            return

        visible = self._row_visibility.get(name, True)
        color = self._row_colors.get(name, DEFAULT_AV_COLOR)

        out_params = {
            "pdb_path": pdb_val,
            "chain_identifier": chain,
            "residue_seq_number": res,
            "atom_name": atom,
            "simulation_type": model,
            "dye_preset": preset,
            "linker_length": float(params.get("linker_length", 20.0)),
            "linker_width": float(params.get("linker_width", 4.5)),
            "radius1": float(params.get("radius1", 3.5)),
            "radius2": float(params.get("radius2", 0.0)),
            "radius3": float(params.get("radius3", 0.0)),
            "body_id": int(params.get("body_id", 0)),
            "visible": visible,
            "av_color": rgba_to_json(color),
            "allowed_sphere_radius": float(params.get("allowed_sphere_radius", 1.5)),
            "anchor_atoms": str(params.get("anchor_atoms", "")),
            "chain_weighting": bool(params.get("chain_weighting", False)),
            "contact_volume_thickness": float(params.get("contact_volume_thickness", 0.0)),
            "contact_volume_trapped_fraction": float(params.get("contact_volume_trapped_fraction", -1.0)),
            "min_sphere_volume_fraction": float(params.get("min_sphere_volume_fraction", 0.0)),
            "simulation_grid_resolution": float(params.get("simulation_grid_resolution", 1.5)),
            "strip_mask": str(params.get("strip_mask", "")),
        }

        if "av_volume" in params:
            out_params["av_volume"] = float(params["av_volume"])
        if "av_points" in params:
            out_params["av_points"] = int(params["av_points"])
        if "av_mean" in params:
            out_params["av_mean"] = [float(x) for x in params["av_mean"]]

        self._block_table_signals += 1
        try:
            self.position_added.emit(name, out_params)
        finally:
            self._block_table_signals -= 1

    def _trigger_row_av(self, row: int, force: bool = False) -> None:
        if row < 0 or row >= self.table.rowCount():
            return

        name_item = self.table.item(row, 1)
        name = self._set_auto_name_for_row(row)
        if not name and name_item:
            name = name_item.text().strip()

        if row >= self.table.rowCount() - 1:
            return

        chain_cb = self.table.cellWidget(row, 3)
        chain = chain_cb.currentText().strip() if isinstance(chain_cb, QtWidgets.QComboBox) else ""

        res_cb = self.table.cellWidget(row, 4)
        res_text = res_cb.currentText().strip() if isinstance(res_cb, QtWidgets.QComboBox) else ""

        atom_cb = self.table.cellWidget(row, 5)
        atom = atom_cb.currentText().strip() if isinstance(atom_cb, QtWidgets.QComboBox) else ""

        if not name:
            msg = "AV: Not computed (labeling site Name is empty)"
            logger.warning(msg)
            self.av_preview_label.setText(msg)
            return

        pdb_val = self._get_row_pdb_val(row)
        if not pdb_val:
            msg = f"AV: Not computed (no PDB path/ID for '{name}')"
            logger.warning(msg)
            self.av_preview_label.setText(msg)
            return

        params = name_item.data(QtCore.Qt.UserRole + 1) if name_item else None
        if not isinstance(params, dict):
            params = {}

        if not (chain and res_text and atom):
            msg = f"AV: Not computed ('{name}' needs Chain, Residue, and Atom)"
            logger.warning(msg)
            self.av_preview_label.setText(msg)
            return

        try:
            res_id = int(res_text)
        except ValueError:
            msg = f"AV: Not computed (invalid residue seq number '{res_text}' for '{name}')"
            logger.warning(msg)
            self.av_preview_label.setText(msg)
            return

        try:
            model_cb = self.table.cellWidget(row, 7)
            model = model_cb.currentText() if isinstance(model_cb, QtWidgets.QComboBox) else "AV1"

            l = float(params.get("linker_length", 20.0))
            w = float(params.get("linker_width", 4.5))
            r1 = float(params.get("radius1", 3.5))
            r2 = float(params.get("radius2", 0.0))
            r3 = float(params.get("radius3", 0.0))
            resolution = float(params.get("simulation_grid_resolution", 1.5))
        except (ValueError, AttributeError) as e:
            msg = f"AV: Not computed (invalid parameter values for '{name}': {e})"
            logger.error(msg)
            self.av_preview_label.setText(msg)
            self._show_status(msg, "error")
            return

        signature = (
            pdb_val,
            chain,
            res_id,
            atom,
            model,
            round(l, 6),
            round(w, 6),
            round(r1, 6),
            round(r2, 6),
            round(r3, 6),
            round(resolution, 6),
        )
        if not force and self._av_signatures.get(name) == signature and name in self._av_cache:
            self._update_settings_tooltip(row, params)
            self._update_3d_overlays()
            return

        struct = self._load_pdb_for_val(pdb_val)
        if not struct:
            msg = f"AV: Not computed (failed to load structure for '{name}' with PDB '{pdb_val}')"
            logger.error(msg)
            self.av_preview_label.setText(msg)
            self._show_status(msg, "error")
            return
            
        pdb_path = struct.filename

        # Cancel any existing debounce timer
        if name in self._av_timers:
            self._av_timers[name].stop()
            self._av_timers[name].deleteLater()

        # Create a new timer to debounce the computation by 400ms
        timer = QtCore.QTimer()
        timer.setSingleShot(True)
        timer.setInterval(400)
        
        # Capture variables for lambda
        source_info = {
            "chain_identifier": chain,
            "residue_seq_number": res_id,
            "atom_name": atom,
        }
        
        timer.timeout.connect(lambda: self._do_trigger_row_av(
            name, row, pdb_path, chain, res_id, atom, l, w, (r1, r2, r3), resolution, source_info
        ))
        
        self._av_timers[name] = timer
        timer.start()
        self._av_signatures[name] = signature

    def _do_trigger_row_av(
        self, name: str, row: int, pdb_path: str, chain: str, res_id: int, atom: str,
        l: float, w: float, radii: tuple[float, float, float], resolution: float, source_info: dict
    ) -> None:
        try:
            # Clear running worker for this position if any
            if name in self._active_workers:
                old_worker = self._active_workers[name]
                if old_worker.isRunning():
                    old_worker.terminate()
                    old_worker.wait()

            self.av_preview_label.setText(f"AV: Computing {name}...")
            self.av_progress_bar.setVisible(True)

            worker = AVWorker(
                chain=chain,
                res_id=res_id,
                atom=atom,
                linker_length=l,
                linker_width=w,
                radii=radii,
                disc_step=resolution,
                pdb_path=pdb_path,
                source_info=source_info,
            )
            worker.position_name = name
            worker.row_index = row

            worker.result_ready.connect(lambda n, v, x, y, z, c, grid_step, w=worker: self.onAVComputationFinished(w.position_name, n, v, x, y, z, c, grid_step))
            worker.error.connect(lambda msg, w=worker: self.onAVComputationError(w.position_name, msg))
            self._active_workers[name] = worker
            worker.start()

        except Exception as e:
            msg = f"AV: Not computed (error preparing AV calculation for '{name}': {e})"
            logger.error(msg)
            self.av_preview_label.setText(msg)
            self._show_status(msg, "error")

    def onAVComputationFinished(
        self,
        name: str,
        n_points: int,
        volume: float,
        mx: float,
        my: float,
        mz: float,
        coords: np.ndarray,
        grid_step: float,
    ) -> None:
        if name not in self._row_colors:
            return  # Row was deleted or renamed in the meantime
            
        self.av_progress_bar.setVisible(False)
        self.av_preview_label.setText(f"AV: Calculated {name} (Vol: {volume:.1f} Å³, Points: {n_points})")

        color = self._row_colors.get(name, DEFAULT_AV_COLOR)

        self._av_cache[name] = (coords, np.array([mx, my, mz]), grid_step, color)
        self._active_workers.pop(name, None)

        self._update_3d_overlays()

        # ONLY update the tooltip (no longer saving outputs back into the inputs parameters dict)
        for row in range(self.table.rowCount() - 1):
            name_item = self.table.item(row, 1)
            if name_item and name_item.text().strip() == name:
                params = name_item.data(QtCore.Qt.UserRole + 1)
                if not isinstance(params, dict):
                    params = {}
                # Create dynamic mock params for the tooltip that won't get saved to json
                tooltip_params = dict(params)
                tooltip_params["av_volume"] = volume
                tooltip_params["av_points"] = n_points
                tooltip_params["av_mean"] = [mx, my, mz]
                self._update_settings_tooltip(row, tooltip_params)
                break

    def onAVComputationError(self, name: str, err_msg: str) -> None:
        self.av_progress_bar.setVisible(False)
        msg = f"AV calculation failed for {name}: {err_msg}"
        logger.error(msg)
        self.av_preview_label.setText(msg)
        self._show_status(msg, "error")
        self._active_workers.pop(name, None)

    def _update_3d_overlays(self) -> None:
        if self.mol_view_3d.isHidden():
            return

        self.mol_view_3d.clear_point_overlays()

        for row in range(self.table.rowCount() - 1):
            name_item = self.table.item(row, 1)
            if not name_item:
                continue
            name = name_item.text().strip()
            if not name:
                continue

            visible = self._row_visibility.get(name, True)
            if not visible:
                continue

            if name in self._av_cache:
                points, mean_xyz, grid_step, color = self._av_cache[name]

                # Transparent AV surface overlay (uses color[3] alpha for custom transparency)
                self.mol_view_3d.add_surface_overlay(
                    f"av_{name}",
                    points[:, :3],
                    color=color,
                    alpha=color[3],
                    grid_spacing=max(grid_step, 0.1),
                    padding=max(grid_step * 2.0, 1.0),
                    smoothing_sigma=0.75,
                    dilation_iterations=1,
                    max_dim=112,
                    fallback_size_scale=0.02,
                    fallback_min_size=1.5,
                )

                # Sphere overlay at mean position
                marker_color = (color[0], color[1], color[2], max(color[3], 0.9))
                self.mol_view_3d.add_sphere(
                    mean_xyz,
                    radius=1.5,
                    color=marker_color,
                    label=name,
                    key=f"mean_{name}"
                )

    def _load_pdb_for_val(self, pdb_val: str) -> chisurf.core.structure.Structure | None:
        if not pdb_val:
            return None
        if pdb_val in self._loaded_structures:
            return self._loaded_structures[pdb_val]

        if re.match(r"^[A-Za-z0-9]{4}$", pdb_val):
            self._show_status(f"Downloading PDB ID '{pdb_val}'...", "info")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                result = self._client.fetch_pdb(pdb_val) if self._client is not None else None
                if result is None:
                    raise RuntimeError("PDB fetch client is not configured")
                path = result["path"]
                struct = chisurf.core.structure.Structure(path)
                self._loaded_structures[pdb_val] = struct
                self._loaded_structures[path] = struct
                self.mol_view_3d.set_structure(struct)
                self.mol_view_3d.show()
                self._show_status(f"Downloaded and loaded PDB ID '{pdb_val}'.", "info")
                
                # Dynamic update of chains
                for r in range(self.table.rowCount() - 1):
                    if self._get_row_pdb_val(r) == pdb_val:
                        self._update_row_chains(r, struct)
                        chain_cb = self.table.cellWidget(r, 3)
                        chain = chain_cb.currentText() if isinstance(chain_cb, QtWidgets.QComboBox) else ""
                        self._update_row_residues(r, struct, chain)
                        res_cb = self.table.cellWidget(r, 4)
                        res_text = res_cb.currentText() if isinstance(res_cb, QtWidgets.QComboBox) else ""
                        self._update_row_atoms(r, struct, chain, int(res_text) if res_text.isdigit() else 0)
                        self._maybe_auto_fill_name(r)
                        
                return struct
            except Exception as e:
                self._show_status(f"Failed to fetch/load PDB ID '{pdb_val}': {e}", "error")
                return None
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        else:
            if os.path.exists(pdb_val):
                try:
                    struct = chisurf.core.structure.Structure(pdb_val)
                    self._loaded_structures[pdb_val] = struct
                    self.mol_view_3d.set_structure(struct)
                    self.mol_view_3d.show()
                    self._show_status(f"Loaded structure from path: {pdb_val}", "info")
                    
                    for r in range(self.table.rowCount() - 1):
                        if self._get_row_pdb_val(r) == pdb_val:
                            self._update_row_chains(r, struct)
                            
                    return struct
                except Exception as e:
                    self._show_status(f"Failed to load structure from path '{pdb_val}': {e}", "error")
                    return None
        return None

    def _update_row_colors(self, row: int, visible: bool) -> None:
        name_item = self.table.item(row, 1)
        if not name_item:
            return
        name = name_item.text().strip()
        if not name:
            return

        color = self._row_colors.get(name, DEFAULT_AV_COLOR)
        if not visible:
            qcolor = QtGui.QColor("gray")
        else:
            qcolor = QtGui.QColor.fromRgbF(color[0], color[1], color[2], 1.0)

        # Style standard text item columns (only name_item col 1)
        name_item.setForeground(QtGui.QBrush(qcolor))
        font = name_item.font()
        font.setItalic(not visible)
        name_item.setFont(font)

        # Style PdbSelectWidget line_edit (col 2)
        pdb_w = self.table.cellWidget(row, 2)
        if isinstance(pdb_w, PdbSelectWidget):
            if not visible:
                pdb_w.line_edit.setStyleSheet("color: gray; font-style: italic;")
            else:
                pdb_w.line_edit.setStyleSheet("color: normal; font-style: normal;")

        # Style Combobox and Button widgets
        for col in [3, 4, 5, 6, 7, 8]:
            widget = self.table.cellWidget(row, col)
            if isinstance(widget, QtWidgets.QComboBox):
                if not visible:
                    widget.setStyleSheet("color: gray; font-style: italic;")
                else:
                    r, g, b, _ = color
                    qr, qg, qb = int(r*255), int(g*255), int(b*255)
                    widget.setStyleSheet(f"color: rgb({qr}, {qg}, {qb}); font-style: normal;")
            elif isinstance(widget, QtWidgets.QPushButton):
                if not visible:
                    widget.setStyleSheet("color: gray; font-style: italic;")
                else:
                    widget.setStyleSheet("color: normal; font-style: normal;")

    def toggle_row_visibility(self, row: int) -> None:
        name_item = self.table.item(row, 1)
        if not name_item:
            return
        name = name_item.text().strip()
        if not name:
            return

        current_visible = self._row_visibility.get(name, True)
        new_visible = not current_visible
        self._row_visibility[name] = new_visible

        # Sync checkbox state
        cb_widget = self.table.cellWidget(row, 0)
        if isinstance(cb_widget, CenteredCheckBox):
            cb_widget.checkbox.blockSignals(True)
            cb_widget.checkbox.setChecked(new_visible)
            cb_widget.checkbox.blockSignals(False)

        self._update_row_colors(row, new_visible)
        self._update_3d_overlays()
        self._trigger_row_change(row)

    def onRowHeaderDoubleClicked(self, row: int) -> None:
        if row >= 0 and row < self.table.rowCount() - 1:
            self.toggle_row_visibility(row)

    def onTableDoubleClicked(self, index: QtCore.QModelIndex) -> None:
        row = index.row()
        col = index.column()
        if col in (0, 1, 2):
            if row >= 0 and row < self.table.rowCount() - 1:
                self.toggle_row_visibility(row)

    def onTableSelectionChanged(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= self.table.rowCount() - 1:
            return

        pdb_val = self._get_row_pdb_val(row)
        if not pdb_val:
            return

        struct = self._load_pdb_for_val(pdb_val)
        if struct:
            self._block_selector_sync = True
            try:
                pass
            finally:
                self._block_selector_sync = False

    def onChooseRowColor(self) -> None:
        button = self.sender()
        if not button:
            return
        row = -1
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 9) is button:
                row = r
                break
        if row < 0:
            return

        name_item = self.table.item(row, 1)
        if not name_item:
            return
        name = name_item.text().strip()
        if not name:
            return

        current_color = self._row_colors.get(name, DEFAULT_AV_COLOR)
        r, g, b, a = normalize_rgba(current_color)
        initial = QtGui.QColor.fromRgbF(r, g, b, a)

        color = QtWidgets.QColorDialog.getColor(
            initial,
            self,
            "Select AV Color",
            QtWidgets.QColorDialog.ShowAlphaChannel,
        )
        if not color.isValid():
            return

        new_color = (
            float(color.redF()),
            float(color.greenF()),
            float(color.blueF()),
            float(color.alphaF()),
        )
        self._row_colors[name] = new_color

        if isinstance(button, QtWidgets.QPushButton):
            button.setStyleSheet(f"background-color: rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()});")

        visible = self._row_visibility.get(name, True)
        self._update_row_colors(row, visible)

        if name in self._av_cache:
            coords, mean_xyz, grid_step, _ = self._av_cache[name]
            self._av_cache[name] = (coords, mean_xyz, grid_step, new_color)

        self._update_3d_overlays()
        self._trigger_row_change(row)

    def onShowRowSettings(self) -> None:
        button = self.sender()
        if not button:
            return
        row = -1
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 8) is button:
                row = r
                break
        if row < 0:
            return

        name_item = self.table.item(row, 1)
        if not name_item:
            return

        params = name_item.data(QtCore.Qt.UserRole + 1)
        if not isinstance(params, dict):
            params = {}

        model_cb = self.table.cellWidget(row, 7)
        model = model_cb.currentText() if isinstance(model_cb, QtWidgets.QComboBox) else "AV1"

        dialog = DetailSettingsDialog(params, dye_model=model, parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_params = dialog.get_settings()
            params.update(new_params)
            name_item.setData(QtCore.Qt.UserRole + 1, params)
            self._update_settings_tooltip(row, params)

            self._trigger_row_av(row)
            self._trigger_row_change(row)

    def onComputeAVAll(self) -> None:
        """Force recalculation of AV for all populated rows."""
        for r in range(self.table.rowCount() - 1):
            self._trigger_row_av(r, force=True)

    @staticmethod
    def _safe_mrc_stem(name: str) -> str:
        """Return a filesystem-safe stem for an AV label."""
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
        return stem or "av"

    def _computed_av_names_for_rows(self, rows: list[int]) -> list[str]:
        """Return computed AV names for the given table rows."""
        names = []
        for row in rows:
            name_item = self.table.item(row, 1)
            name = name_item.text().strip() if name_item else ""
            if name and name in self._av_cache:
                names.append(name)
        return names

    def _selected_or_all_computed_av_names(self) -> list[str]:
        """Return selected computed AV names, or all computed AV names."""
        selected_names = self._computed_av_names_for_rows(self._get_selected_rows())
        if selected_names:
            return selected_names

        names = []
        for row in range(self.table.rowCount() - 1):
            name_item = self.table.item(row, 1)
            name = name_item.text().strip() if name_item else ""
            if name and name in self._av_cache:
                names.append(name)
        return names

    def _save_av_mrc(self, name: str, path: str | Path) -> Path:
        """Save one cached AV as an MRC file."""
        coords, _mean_xyz, grid_step, _color = self._av_cache[name]
        return save_av_mrc(path, coords, grid_step)

    def onSaveSelectedAVsAsMRC(self) -> None:
        """Save selected computed AVs as MRC density maps."""
        names = self._selected_or_all_computed_av_names()
        if not names:
            msg = "No computed AVs available to save as MRC."
            self.av_preview_label.setText(msg)
            self._show_status(msg, "warning")
            return

        try:
            if len(names) == 1:
                default_name = f"{self._safe_mrc_stem(names[0])}.mrc"
                filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save AV as MRC",
                    default_name,
                    "MRC map (*.mrc *.map *.ccp4);;All files (*)",
                )
                if not filename:
                    return
                written = [self._save_av_mrc(names[0], filename)]
            else:
                directory = QtWidgets.QFileDialog.getExistingDirectory(
                    self,
                    f"Save {len(names)} AV MRC maps",
                )
                if not directory:
                    return
                written = [
                    self._save_av_mrc(name, Path(directory) / f"{self._safe_mrc_stem(name)}.mrc")
                    for name in names
                ]
        except Exception as exc:
            msg = f"Failed to save AV MRC: {exc}"
            logger.error(msg)
            self.av_preview_label.setText(msg)
            self._show_status(msg, "error")
            return

        msg = f"Saved {len(written)} AV MRC map(s)."
        self.av_preview_label.setText(msg)
        self._show_status(msg, "info")

    def onDyePresetChanged(self, text: str) -> None:
        sender = self.sender()
        if not sender or self._block_table_signals > 0:
            return
        row = -1
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 6) is sender:
                row = r
                break
        if row < 0:
            return

        if text == "Custom":
            return

        dye_def = chisurf.core.structure.av.dye_definition.get(text)
        if not dye_def:
            return

        name_item = self.table.item(row, 1)
        if not name_item:
            return
        params = name_item.data(QtCore.Qt.UserRole + 1)
        if not isinstance(params, dict):
            params = {}

        params["linker_length"] = float(dye_def.get("linker_length", 20.0))
        params["linker_width"] = float(dye_def.get("linker_width", 4.5))
        params["radius1"] = float(dye_def.get("radius1", 3.5))
        params["radius2"] = float(dye_def.get("radius2", 0.0))
        params["radius3"] = float(dye_def.get("radius3", 0.0))

        model_name = dye_def.get("simulation_type", "AV1")
        params["simulation_type"] = model_name

        name_item.setData(QtCore.Qt.UserRole + 1, params)
        self._update_settings_tooltip(row, params)

        model_cb = self.table.cellWidget(row, 7)
        if isinstance(model_cb, QtWidgets.QComboBox):
            model_cb.blockSignals(True)
            idx = model_cb.findText(model_name)
            if idx >= 0:
                model_cb.setCurrentIndex(idx)
            model_cb.blockSignals(False)

        self._trigger_row_av(row)
        self._trigger_row_change(row)

    def onDyeModelChanged(self, text: str) -> None:
        sender = self.sender()
        if not sender or self._block_table_signals > 0:
            return
        row = -1
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 7) is sender:
                row = r
                break
        if row < 0:
            return

        name_item = self.table.item(row, 1)
        if not name_item:
            return
        params = name_item.data(QtCore.Qt.UserRole + 1)
        if not isinstance(params, dict):
            params = {}

        params["simulation_type"] = text
        name_item.setData(QtCore.Qt.UserRole + 1, params)

        self._trigger_row_av(row)
        self._trigger_row_change(row)

    def _find_checkbox_row(self, checkbox: QtWidgets.QCheckBox) -> int:
        for r in range(self.table.rowCount()):
            w = self.table.cellWidget(r, 0)
            if isinstance(w, CenteredCheckBox) and w.checkbox is checkbox:
                return r
        return -1

    def onRowVisibilityToggled(self, checked: bool) -> None:
        sender = self.sender()
        if not isinstance(sender, QtWidgets.QCheckBox):
            return
        row = self._find_checkbox_row(sender)
        if row < 0:
            return
        name_item = self.table.item(row, 1)
        if not name_item:
            return
        name = name_item.text().strip()
        if not name:
            return

        self._row_visibility[name] = checked
        self._update_row_colors(row, checked)
        self._update_3d_overlays()
        self._trigger_row_change(row)

    def onCellChanged(self, row: int, column: int) -> None:
        if self._block_table_signals > 0:
            return

        if column == 1:
            name_item = self.table.item(row, 1)
            if name_item:
                new_name = name_item.text().strip()
                old_name = name_item.data(QtCore.Qt.UserRole)
                
                if old_name and old_name != new_name:
                    self._block_table_signals += 1
                    try:
                        self.position_removed.emit(old_name)
                    finally:
                        self._block_table_signals -= 1

                    if new_name:
                        if old_name in self._row_visibility:
                            self._row_visibility[new_name] = self._row_visibility.pop(old_name)
                        if old_name in self._row_colors:
                            self._row_colors[new_name] = self._row_colors.pop(old_name)
                        if old_name in self._av_cache:
                            self._av_cache[new_name] = self._av_cache.pop(old_name)
                        if old_name in self._av_signatures:
                            self._av_signatures[new_name] = self._av_signatures.pop(old_name)

                name_item.setData(QtCore.Qt.UserRole, new_name)

                if new_name:
                    self._ensure_row_style_state(row, new_name)
                    self._ensure_trailing_empty_row()

                if not new_name and old_name:
                    self._block_table_signals += 1
                    try:
                        self.position_removed.emit(old_name)
                    finally:
                        self._block_table_signals -= 1
                    self._row_colors.pop(old_name, None)
                    self._row_visibility.pop(old_name, None)
                    self._av_cache.pop(old_name, None)
                    self._av_signatures.pop(old_name, None)

                visible = self._row_visibility.get(new_name, True)
                self._update_row_colors(row, visible)

            self._trigger_row_av(row)
            self._trigger_row_change(row)

        if row == self.table.rowCount() - 1:
            name_item = self.table.item(row, 1)
            if name_item and name_item.text().strip():
                self._add_empty_row()

    def on_chimol_atom_selection_changed(self, selected_atom_indices):
        if not selected_atom_indices or self._block_selector_sync:
            return
        atom_index = selected_atom_indices[0]
        
        row = self.table.currentRow()
        if row < 0 or row >= self.table.rowCount():
            return

        struct = self._get_row_structure(row)
        if not struct or struct.atoms is None:
            return

        if atom_index < 0 or atom_index >= len(struct.atoms):
            return

        atom = struct.atoms[atom_index]
        chain = str(atom['chain'])
        res_id = str(atom['res_id'])
        atom_name = str(atom['atom_name'])

        chain_cb = self.table.cellWidget(row, 3)
        res_cb = self.table.cellWidget(row, 4)
        atom_cb = self.table.cellWidget(row, 5)

        self._block_selector_sync = True
        try:
            if isinstance(chain_cb, QtWidgets.QComboBox):
                chain_cb.setCurrentText(chain)
                self._update_row_residues(row, struct, chain)
            if isinstance(res_cb, QtWidgets.QComboBox):
                res_cb.setCurrentText(res_id)
                self._update_row_atoms(row, struct, chain, int(res_id))
            if isinstance(atom_cb, QtWidgets.QComboBox):
                atom_cb.setCurrentText(atom_name)
        finally:
            self._block_selector_sync = False

        self._maybe_auto_fill_name(row)
        self._trigger_row_av(row)
        self._trigger_row_change(row)

    def load_structure(self, path: str) -> None:
        try:
            self._pdb_path = path

            import chisurf as cs
            if hasattr(cs.core.settings, "structure_data"):
                cs.core.settings.structure_data.setdefault("IMP", {})["filter_non_standard_residues"] = False

            paths = [p.strip() for p in path.split(",") if p.strip()]
            if len(paths) == 1:
                structure = chisurf.core.structure.Structure(paths[0])
            else:
                atoms_list = []
                for p in paths:
                    s = chisurf.core.structure.Structure(p)
                    if s.atoms is not None and len(s.atoms) > 0:
                        atoms_list.append(s.atoms)
                if atoms_list:
                    combined_atoms = np.concatenate(atoms_list)
                    combined_atoms['atom_id'] = np.arange(1, len(combined_atoms) + 1)

                    structure = chisurf.core.structure.Structure()
                    structure.atoms = combined_atoms
                    structure.filename = path
                else:
                    structure = chisurf.core.structure.Structure()

            self._loaded_structures[path] = structure
            self.mol_view_3d.set_structure(structure)
            self.mol_view_3d.show()

            row = self.table.currentRow()
            if row >= 0 and row < self.table.rowCount() - 1:
                pdb_widget = self.table.cellWidget(row, 2)
                if isinstance(pdb_widget, PdbSelectWidget) and not pdb_widget.line_edit.text().strip():
                    pdb_widget.line_edit.setText(path)
                    self._update_row_chains(row, structure)
        except Exception as e:
            self._show_status(f"Error loading reference PDB structure: {str(e)}", "error")

    def _get_selected_rows(self) -> list[int]:
        selected_ranges = self.table.selectedRanges()
        rows = set()
        for r in selected_ranges:
            for row in range(r.topRow(), r.bottomRow() + 1):
                if row < self.table.rowCount() - 1:
                    rows.add(row)
        return sorted(list(rows))

    def onDeleteRowClicked(self) -> None:
        button = self.sender()
        if not button or self._block_table_signals > 0:
            return
        row = -1
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 10) is button:
                row = r
                break
        if row < 0 or row >= self.table.rowCount() - 1:
            return

        self.table.clearSelection()
        self.table.selectRow(row)
        self.onDeleteSelectedRows()

    def onDeleteSelectedRows(self) -> None:
        rows_to_delete = self._get_selected_rows()
        if not rows_to_delete:
            row = self.table.currentRow()
            if row >= 0 and row < self.table.rowCount() - 1:
                rows_to_delete = [row]
        
        if not rows_to_delete:
            return
            
        self._block_table_signals += 1
        try:
            for row in reversed(rows_to_delete):
                name_item = self.table.item(row, 1)
                if name_item:
                    name = name_item.text().strip()
                    if name:
                        self.position_removed.emit(name)
                        self._row_colors.pop(name, None)
                        self._row_visibility.pop(name, None)
                        self._av_cache.pop(name, None)
                        self._av_signatures.pop(name, None)
                        if name in self._active_workers:
                            worker = self._active_workers[name]
                            if worker.isRunning():
                                worker.terminate()
                                worker.wait()
                            self._active_workers.pop(name, None)
                self.table.removeRow(row)
        finally:
            self._block_table_signals -= 1
            
        self._update_3d_overlays()

    def onDeleteSelectedAVs(self) -> None:
        rows = self._get_selected_rows()
        if not rows:
            row = self.table.currentRow()
            if row >= 0 and row < self.table.rowCount() - 1:
                rows = [row]
                
        if not rows:
            return
            
        for row in rows:
            name_item = self.table.item(row, 1)
            if name_item:
                name = name_item.text().strip()
                if name:
                    self._av_cache.pop(name, None)
                    self._av_signatures.pop(name, None)
                    if name in self._active_workers:
                        worker = self._active_workers[name]
                        if worker.isRunning():
                            worker.terminate()
                            worker.wait()
                        self._active_workers.pop(name, None)
                    params = name_item.data(QtCore.Qt.UserRole + 1)
                    if isinstance(params, dict):
                        params.pop("av_volume", None)
                        params.pop("av_points", None)
                        params.pop("av_mean", None)
                        name_item.setData(QtCore.Qt.UserRole + 1, params)
                        self._update_settings_tooltip(row, params)
                        self._trigger_row_change(row)
                        
        self.av_preview_label.setText("AV: Not computed")
        self._update_3d_overlays()

    def onTableContextMenu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        add_action = menu.addAction("Add Row")
        delete_action = menu.addAction("Delete Selected Row(s)")
        clear_av_action = menu.addAction("Delete Selected AV(s)")
        save_av_mrc_action = menu.addAction("Save Selected AV(s) as MRC")
        menu.addSeparator()
        select_all_action = menu.addAction("Select All")
        clear_select_action = menu.addAction("Clear Selection")

        selected_rows = self._get_selected_rows()
        if not selected_rows:
            delete_action.setEnabled(False)
            clear_av_action.setEnabled(False)
            save_av_mrc_action.setText("Save All Computed AV(s) as MRC")
        elif not self._computed_av_names_for_rows(selected_rows):
            save_av_mrc_action.setEnabled(False)

        action = menu.exec_(self.table.mapToGlobal(pos))
        if action == add_action:
            self.onAddRowTriggered()
        elif action == delete_action:
            self.onDeleteSelectedRows()
        elif action == clear_av_action:
            self.onDeleteSelectedAVs()
        elif action == save_av_mrc_action:
            self.onSaveSelectedAVsAsMRC()
        elif action == select_all_action:
            self.table.selectAll()
        elif action == clear_select_action:
            self.table.clearSelection()
