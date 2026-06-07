"""ProteinMC model widget with Chimol visual feedback."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy import QtCore, QtWidgets

import chisurf as cs
import chisurf.core.fitting

from chisurf.gui.widgets.models.model_widget import ModelWidget
from chisurf.plugins.modelling.proteinmc.model import (
    ProteinMCProgress,
    ProteinMCRunner,
    build_move_map_from_flexfit,
    list_flexfit_sets,
)
from chisurf.gui.plots.proteinMC import ProteinMCDistanceNetworkPlot, ProteinMCPlot, ProteinMCStructurePlot


try:  # Chimol is preferred over PyMOL for ProteinMC visual feedback.
    from chisurf.plugins.chimol.chimol.renderer.view import MolView as ChimolView
except Exception:  # pragma: no cover - optional GUI backend
    ChimolView = None


class _ProteinMCWorker(QtCore.QObject):
    """Qt worker that runs the headless ProteinMC model."""

    progress = QtCore.Signal(object)
    prepared = QtCore.Signal(object)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, **runner_kwargs) -> None:
        """Initialize the worker with ProteinMCRunner keyword arguments."""
        super().__init__()
        self.runner_kwargs = runner_kwargs
        self.runner = None

    @QtCore.Slot()
    def run(self) -> None:
        """Run ProteinMC and emit Qt signals."""
        try:
            self.runner = ProteinMCRunner(
                progress_callback=self.progress.emit,
                **self.runner_kwargs,
            )
            self.prepared.emit(self.runner.structure)
            result = self.runner.run()
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)

    @QtCore.Slot()
    def stop(self) -> None:
        """Request the running ProteinMC job to stop."""
        if self.runner is not None:
            self.runner.stop()


class ProteinMCModelWidget(ModelWidget):
    """Model widget that runs ProteinMC and displays live Chimol feedback."""

    name = "ProteinMC"

    plot_classes = [(ProteinMCStructurePlot, {}), (ProteinMCDistanceNetworkPlot, {}), (ProteinMCPlot, {})]
    _POTENTIAL_SPECS = {
        "default": {
            "label": "Default clash",
            "name": "default",
            "weight": 1.0,
            "enabled": False,
            "params": {"clash_tolerance": 2.0, "covalent_radius": 1.5},
        },
        "hbond": {
            "label": "H-bond",
            "name": "hbond",
            "weight": 2.0,
            "enabled": True,
            "params": {
                "cutoff_ca": 8.0,
                "cutoff_hbond": 3.0,
                "oh": 1.0,
                "on": 1.0,
                "cn": 1.0,
                "ch": 1.0,
            },
        },
        "mj": {
            "label": "MJ",
            "name": "mj",
            "weight": 1.0,
            "enabled": False,
            "params": {"ca_cutcoff": 6.5},
        },
        "unres": {
            "label": "UNRES",
            "name": "unres",
            "weight": 1.0,
            "enabled": True,
            "params": {
                "ca_cutoff": 15.0,
                "repulsion": 100.0,
                "min_dist": 3.5,
                "max_dist": 19.0,
                "bin_width": 0.05,
                "centroid_number": 4.0,
            },
        },
    }

    def __init__(
        self,
        fit: "cs.core.fitting.fit.Fit",
        *args,
        **kwargs,
    ):
        """Initialize the ProteinMC MDL widget."""
        super().__init__(fit=fit, *args, **kwargs)
        self.structure = getattr(self.fit, "data", None)
        self.rmsd: list[float] = []
        self.drmsd: list[float] = []
        self.energy: list[float] = []
        self.chi2r: list[float] = []
        self.proteinmc_structure = None
        self.trajectory_frames: list[np.ndarray] = []
        self.current_frame_index: int = 0
        self._thread = None
        self._worker = None
        self._chimol_object_id = None
        self._live_frames: list[np.ndarray] = []
        self._chimol_update_pending: bool = False
        self._plot_update_pending: bool = False
        self._pending_xyz: Optional[np.ndarray] = None
        self._last_chimol_update_ms: int = 0
        self._last_plot_update_ms: int = 0
        self._chimol_frame_count: int = 0
        self._sampling_directory: Optional[Path] = None
        self._resume_rmsd: list[float] = []
        self._resume_drmsd: list[float] = []
        self._resume_energy: list[float] = []
        self._resume_chi2r: list[float] = []
        self._continuing_run: bool = False
        self._requested_run_count: int = 1
        # Throttle expensive UI updates (Chimol rebuild + plot setData) to
        # at most one per ``update_interval_ms`` so the main thread stays
        # responsive while ProteinMC is sampling.
        self.update_interval_ms: int = 500
        self._build_ui()

    @property
    def frame_count(self) -> int:
        """Number of trajectory frames currently available."""

        return len(self.trajectory_frames)

    def set_current_frame(self, index: int, *, update_plots: bool = True) -> None:
        """Set the globally displayed ProteinMC frame across all plots."""

        if self.frame_count <= 0:
            self.current_frame_index = 0
            return
        try:
            idx = int(index)
        except Exception:
            idx = 0
        idx = max(0, min(idx, self.frame_count - 1))
        self.current_frame_index = idx
        if self.viewer is not None and self._chimol_object_id is not None:
            try:
                self.viewer.set_active_frame(idx, object_id=self._chimol_object_id)
            except Exception:
                try:
                    self.viewer.set_current_frame(idx)
                except Exception:
                    pass
        if update_plots:
            self._schedule_plot_update()

    def _build_ui(self) -> None:
        """Create the ProteinMC controls and Chimol view."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        form = QtWidgets.QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(3)
        form.setVerticalSpacing(2)
        layout.addLayout(form)

        self.structure_edit = QtWidgets.QLineEdit(self)
        self.structure_edit.setPlaceholderText("PDB/PQR path or PDB ID, e.g. 148l")
        if self.structure is not None:
            self.structure_edit.setText(str(getattr(self.structure, "filename", "")))
        self.structure_button = QtWidgets.QPushButton("...", self)
        self.structure_button.clicked.connect(self.on_browse_structure)

        self.labeling_edit = QtWidgets.QLineEdit(self)
        self.labeling_edit.setPlaceholderText("FPS JSON labeling file")
        self.labeling_button = QtWidgets.QPushButton("...", self)
        self.labeling_button.clicked.connect(self.on_browse_labeling)
        self.labeling_edit_button = QtWidgets.QPushButton("E", self)
        self.labeling_edit_button.setToolTip("edit")
        self.labeling_edit_button.clicked.connect(self.on_edit_labeling_json)

        self.score_set_combo = QtWidgets.QComboBox(self)
        self.score_set_combo.setToolTip(
            "Select a χ² score set (scoring group) from the FPS JSON file. "
            "Leave empty to use all distances."
        )
        self.score_set_combo.addItem("")  # empty = all distances

        self.n_iter_spin = QtWidgets.QSpinBox(self)
        self.n_iter_spin.setRange(1, 100000000)
        self.n_iter_spin.setValue(10000)
        self.n_out_spin = QtWidgets.QSpinBox(self)
        self.n_out_spin.setRange(1, 1000000)
        self.n_out_spin.setValue(100)
        self.n_written_spin = QtWidgets.QSpinBox(self)
        self.n_written_spin.setRange(1, 1000000)
        self.n_written_spin.setValue(100)
        self.scale_spin = QtWidgets.QDoubleSpinBox(self)
        self.scale_spin.setDecimals(6)
        self.scale_spin.setRange(0.0, 10.0)
        self.scale_spin.setValue(0.0025)
        self.kt_spin = QtWidgets.QDoubleSpinBox(self)
        self.kt_spin.setDecimals(4)
        self.kt_spin.setRange(0.0, 100000.0)
        self.kt_spin.setValue(1.5)
        self.labeling_weight_spin = QtWidgets.QDoubleSpinBox(self)
        self.labeling_weight_spin.setDecimals(4)
        self.labeling_weight_spin.setRange(0.0, 100000.0)
        self.labeling_weight_spin.setValue(1.0)

        tooltips = {
            self.n_iter_spin: "Maximum Monte Carlo trial moves attempted by the ProteinMC run.",
            self.n_out_spin: "Accepted-step interval between saved trajectory frames.",
            self.n_written_spin: "Maximum number of saved trajectory frames; ProteinMC stops after this many frames or n_iter trials.",
            self.scale_spin: "Size of each torsion-angle Monte Carlo move; larger values explore faster but reject more moves.",
            self.kt_spin: "Metropolis temperature kT; higher values accept more uphill energy moves.",
            self.labeling_weight_spin: "Weight of the FPS/AV labeling restraint energy term.",
        }
        for widget, tooltip in tooltips.items():
            widget.setToolTip(tooltip)

        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        form.addWidget(QtWidgets.QLabel("Structure"), 0, 0)
        form.addWidget(self.structure_edit, 0, 1, 1, 3)
        form.addWidget(self.structure_button, 0, 4)
        form.addWidget(QtWidgets.QLabel("Labeling"), 1, 0)
        form.addWidget(self.labeling_edit, 1, 1, 1, 3)
        labeling_buttons = QtWidgets.QHBoxLayout()
        labeling_buttons.setContentsMargins(0, 0, 0, 0)
        labeling_buttons.setSpacing(2)
        labeling_buttons.addWidget(self.labeling_button)
        labeling_buttons.addWidget(self.labeling_edit_button)
        form.addLayout(labeling_buttons, 1, 4)

        self.score_set_combo.setMinimumWidth(120)
        form.addWidget(QtWidgets.QLabel("Score set"), 2, 0)
        form.addWidget(self.score_set_combo, 2, 1, 1, 3)
        self.score_set_combo.currentTextChanged.connect(self._on_score_set_changed)
        self.labeling_edit.textChanged.connect(self._populate_score_sets)

        # --- FlexFit controls ---
        self.flexfit_use_check = QtWidgets.QCheckBox("FlexFit", self)
        self.flexfit_use_check.setToolTip(
            "When checked, only residues listed in the FlexFit section of the "
            "FPS JSON file are allowed to move during sampling.  When unchecked, "
            "all residues are mobile (default behaviour)."
        )
        self.flexfit_use_check.setChecked(False)
        self.flexfit_set_combo = QtWidgets.QComboBox(self)
        self.flexfit_set_combo.setToolTip(
            "Select which FlexFit set defines the flexible residues."
        )
        self.flexfit_set_combo.setMinimumWidth(120)
        self.flexfit_n_residues_label = QtWidgets.QLabel("", self)
        self.flexfit_n_residues_label.setToolTip(
            "Number of flexible residues in the selected FlexFit set."
        )
        flexfit_row = QtWidgets.QHBoxLayout()
        flexfit_row.setContentsMargins(0, 0, 0, 0)
        flexfit_row.setSpacing(3)
        flexfit_row.addWidget(self.flexfit_use_check)
        flexfit_row.addWidget(self.flexfit_set_combo, stretch=1)
        flexfit_row.addWidget(self.flexfit_n_residues_label)
        form.addLayout(flexfit_row, 3, 1, 1, 3)
        self.flexfit_use_check.stateChanged.connect(self._on_flexfit_use_changed)
        self.flexfit_set_combo.currentTextChanged.connect(self._on_flexfit_set_changed)
        self.labeling_edit.textChanged.connect(self._populate_flexfit_sets)

        label_tooltips = {
            "MC trials": tooltips[self.n_iter_spin],
            "Save every": tooltips[self.n_out_spin],
            "Max frames": tooltips[self.n_written_spin],
            "Move scale": tooltips[self.scale_spin],
            "kT": tooltips[self.kt_spin],
            "FPS weight": tooltips[self.labeling_weight_spin],
        }
        parameter_labels = {
            name: QtWidgets.QLabel(name)
            for name in label_tooltips
        }
        for name, label in parameter_labels.items():
            label.setToolTip(label_tooltips[name])

        form.addWidget(parameter_labels["MC trials"], 4, 0)
        form.addWidget(self.n_iter_spin, 4, 1)
        form.addWidget(parameter_labels["Save every"], 4, 2)
        form.addWidget(self.n_out_spin, 4, 3)
        form.addWidget(parameter_labels["Max frames"], 5, 0)
        form.addWidget(self.n_written_spin, 5, 1)
        form.addWidget(parameter_labels["Move scale"], 5, 2)
        form.addWidget(self.scale_spin, 5, 3)
        form.addWidget(parameter_labels["kT"], 6, 0)
        form.addWidget(self.kt_spin, 6, 1)
        form.addWidget(parameter_labels["FPS weight"], 6, 2)
        form.addWidget(self.labeling_weight_spin, 6, 3)

        self._build_potential_ui(layout)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start ProteinMC", self)
        self.start_button.setToolTip("Starts the same ProteinMC run as the Sampling panel's Sample button.")
        self.start_button.hide()
        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_proteinmc)
        self.stop_button.clicked.connect(self.stop_proteinmc)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.status_label = QtWidgets.QLabel("ProteinMC idle", self)
        self.status_label.setMaximumHeight(22)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        if ChimolView is not None:
            self.viewer = ChimolView(parent=self, representation_mode="atoms")
            layout.addWidget(self.viewer, stretch=1)
        else:
            self.viewer = None
            layout.addWidget(QtWidgets.QLabel("Chimol viewer unavailable", self))

        self.setLayout(layout)
        self.layout = layout

    def _build_potential_ui(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create controls for ProteinMC structure energy terms."""

        group = QtWidgets.QGroupBox("Energy terms / force field", self)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(3, 3, 3, 3)
        group_layout.setSpacing(2)
        self.potential_table = QtWidgets.QTableWidget(0, 3, group)
        self.potential_table.setHorizontalHeaderLabels(["Term", "Weight", ""])
        self.potential_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.potential_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.potential_table.setToolTip("Use the row-local r button to remove an energy term.")
        self.potential_table.itemChanged.connect(self.on_potential_item_changed)
        self.potential_table.horizontalHeader().setStretchLastSection(False)
        self.potential_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.potential_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.potential_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.potential_table.verticalHeader().hide()
        self.potential_table.setMinimumHeight(82)
        self.potential_table.setMaximumHeight(130)
        group_layout.addWidget(self.potential_table)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(3)
        self.potential_combo = QtWidgets.QComboBox(group)
        for key, spec in self._POTENTIAL_SPECS.items():
            self.potential_combo.addItem(str(spec["label"]), key)
        self.potential_add_button = QtWidgets.QPushButton("Add", group)
        self.potential_add_button.clicked.connect(self.on_add_potential)
        self.potential_details_button = QtWidgets.QPushButton("Details...", group)
        self.potential_details_button.setToolTip("Edit detailed parameters for the selected energy term")
        self.potential_details_button.clicked.connect(self.on_edit_potential_details)
        controls.addWidget(self.potential_combo, stretch=1)
        controls.addWidget(self.potential_add_button)
        controls.addWidget(self.potential_details_button)
        group_layout.addLayout(controls)

        for key, spec in self._POTENTIAL_SPECS.items():
            if spec["enabled"]:
                self._add_potential_row(key, weight=float(spec["weight"]), settings=spec["params"])

        layout.addWidget(group)
        self._resize_potential_table()

    def _resize_potential_table(self) -> None:
        """Fit the energy table height to its rows without wasting vertical space."""

        header = self.potential_table.horizontalHeader().height()
        rows = max(self.potential_table.rowCount(), 2)
        row_height = self.potential_table.verticalHeader().defaultSectionSize()
        height = min(max(header + rows * row_height + 8, 82), 130)
        self.potential_table.setFixedHeight(height)

    def _add_potential_row(self, key: str, weight: float = None, settings: dict = None) -> None:
        """Add or update one energy-term row."""

        spec = self._POTENTIAL_SPECS.get(key)
        if spec is None:
            return
        for row in range(self.potential_table.rowCount()):
            item = self.potential_table.item(row, 0)
            if item is not None and item.data(QtCore.Qt.UserRole) == key:
                if weight is not None:
                    self.potential_table.item(row, 1).setText(f"{float(weight):.2f}")
                if settings is not None:
                    item.setData(QtCore.Qt.UserRole + 1, dict(settings))
                self._resize_potential_table()
                return

        row = self.potential_table.rowCount()
        self.potential_table.insertRow(row)
        term_item = QtWidgets.QTableWidgetItem(str(spec["label"]))
        term_item.setFlags(term_item.flags() & ~QtCore.Qt.ItemIsEditable)
        term_item.setData(QtCore.Qt.UserRole, key)
        term_item.setData(QtCore.Qt.UserRole + 1, dict(settings or spec["params"]))
        weight_item = QtWidgets.QTableWidgetItem(f"{float(weight if weight is not None else spec['weight']):.2f}")
        remove_button = QtWidgets.QPushButton("r", self.potential_table)
        remove_button.setMaximumWidth(24)
        remove_button.setToolTip("remove")
        remove_button.clicked.connect(lambda _checked=False, button=remove_button: self._remove_potential_button_row(button))
        self.potential_table.setItem(row, 0, term_item)
        self.potential_table.setItem(row, 1, weight_item)
        self.potential_table.setCellWidget(row, 2, remove_button)
        self._resize_potential_table()

    def _remove_potential_button_row(self, button: QtWidgets.QPushButton) -> None:
        """Remove the energy-term row that owns a row-local remove button."""

        for row in range(self.potential_table.rowCount()):
            if self.potential_table.cellWidget(row, 2) is button:
                self.potential_table.removeRow(row)
                self._resize_potential_table()
                return

    def on_potential_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Normalize edited energy-term weights as fixed-point floats."""

        if item.column() != 1:
            return
        try:
            value = float(item.text())
        except ValueError:
            value = 1.0
        text = f"{value:.2f}"
        if item.text() == text:
            return
        table = item.tableWidget()
        table.blockSignals(True)
        item.setText(text)
        table.blockSignals(False)

    @QtCore.Slot()
    def on_add_potential(self) -> None:
        """Add the energy term selected in the combobox."""

        key = self.potential_combo.currentData()
        if key:
            spec = self._POTENTIAL_SPECS[key]
            self._add_potential_row(key, weight=float(spec["weight"]), settings=deepcopy(spec["params"]))

    def on_remove_selected_potential(self, *args) -> None:
        """Remove the currently selected energy-term row."""

        row = self.potential_table.currentRow()
        if row >= 0:
            self.potential_table.removeRow(row)
            self._resize_potential_table()

    @QtCore.Slot()
    def on_edit_potential_details(self) -> None:
        """Open a popup editor for selected energy-term parameters."""

        row = self.potential_table.currentRow()
        if row < 0:
            return
        item = self.potential_table.item(row, 0)
        if item is None:
            return
        key = item.data(QtCore.Qt.UserRole)
        spec = self._POTENTIAL_SPECS.get(key)
        if spec is None:
            return
        settings = dict(item.data(QtCore.Qt.UserRole + 1) or spec["params"])
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"{spec['label']} parameters")
        form = QtWidgets.QFormLayout(dialog)
        controls = {}
        for param_name, default in spec["params"].items():
            spin = QtWidgets.QDoubleSpinBox(dialog)
            spin.setDecimals(6)
            spin.setRange(-100000.0, 100000.0)
            spin.setValue(float(settings.get(param_name, default)))
            form.addRow(param_name, spin)
            controls[param_name] = spin
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        updated = {name: spin.value() for name, spin in controls.items()}
        item.setData(QtCore.Qt.UserRole + 1, updated)

    def _potential_settings(self) -> list[dict]:
        """Return enabled ProteinMC potential settings from the UI."""

        out = []
        table = getattr(self, "potential_table", None)
        if table is None:
            return out
        for row in range(table.rowCount()):
            term_item = table.item(row, 0)
            weight_item = table.item(row, 1)
            if term_item is None or weight_item is None:
                continue
            key = term_item.data(QtCore.Qt.UserRole)
            spec = self._POTENTIAL_SPECS.get(key)
            if spec is None:
                continue
            settings = dict(term_item.data(QtCore.Qt.UserRole + 1) or spec["params"])
            for int_key in ("centroid_number",):
                if int_key in settings:
                    settings[int_key] = int(settings[int_key])
            try:
                weight = float(weight_item.text())
            except ValueError:
                weight = float(spec["weight"])
                weight_item.setText(f"{weight:.2f}")
            out.append(
                {
                    "name": spec["name"],
                    "weight": weight,
                    "settings": settings,
                }
            )
        return out

    def _populate_score_sets(self) -> None:
        """Read score sets from the current FPS JSON and fill the combo box."""
        filename = self.labeling_edit.text().strip()
        old = self.score_set_combo.currentText()
        self.score_set_combo.blockSignals(True)
        self.score_set_combo.clear()
        self.score_set_combo.addItem("")
        group_names: list[str] = []
        if filename:
            try:
                with open(filename) as fp:
                    payload = json.load(fp)
                group_names = list((payload.get("χ²", {}) or {}).keys())
                for name in group_names:
                    self.score_set_combo.addItem(name)
            except Exception:
                pass
        # Restore previous selection if still available;
        # otherwise default to the first score group.
        idx = self.score_set_combo.findText(old)
        if idx >= 0:
            self.score_set_combo.setCurrentIndex(idx)
        elif group_names:
            self.score_set_combo.setCurrentIndex(1)
        self.score_set_combo.blockSignals(False)

    def _on_score_set_changed(self, text: str) -> None:
        """React to score-set combo changes (persisted via get_state/set_state)."""

    # --- FlexFit methods ---

    def _populate_flexfit_sets(self) -> None:
        """Read FlexFit sets from the current FPS JSON and fill the combo box."""
        filename = self.labeling_edit.text().strip()
        old = self.flexfit_set_combo.currentText()
        self.flexfit_set_combo.blockSignals(True)
        self.flexfit_set_combo.clear()
        set_names: list[str] = []
        if filename:
            try:
                set_names = list_flexfit_sets(filename)
                for name in set_names:
                    self.flexfit_set_combo.addItem(name)
            except Exception:
                pass
        has_data = bool(set_names)
        self.flexfit_use_check.setEnabled(has_data)
        if not has_data:
            self.flexfit_use_check.setChecked(False)
        # Restore previous selection or default to first.
        idx = self.flexfit_set_combo.findText(old)
        if idx >= 0:
            self.flexfit_set_combo.setCurrentIndex(idx)
        elif set_names:
            self.flexfit_set_combo.setCurrentIndex(0)
        self.flexfit_set_combo.blockSignals(False)
        self._on_flexfit_set_changed(self.flexfit_set_combo.currentText())

    def _on_flexfit_use_changed(self, state: int) -> None:
        """Enable/disable the FlexFit set combo when the checkbox changes."""
        self.flexfit_set_combo.setEnabled(self.flexfit_use_check.isChecked())
        self._on_flexfit_set_changed(self.flexfit_set_combo.currentText())

    def _on_flexfit_set_changed(self, name: str) -> None:
        """Update the residue count indicator when the FlexFit set changes."""
        count, total = self._flexfit_residue_counts()
        if name and self.flexfit_use_check.isChecked():
            self.flexfit_n_residues_label.setText(f"{count}/{total} residues flexible")
        else:
            self.flexfit_n_residues_label.setText("")

    def _flexfit_residue_counts(self) -> tuple[int, int]:
        """Return (flexible_count, total_residues) from the active FlexFit set."""
        if not self.flexfit_use_check.isChecked():
            return 0, 0
        filename = self.labeling_edit.text().strip()
        set_name = self.flexfit_set_combo.currentText()
        if not filename or not set_name:
            return 0, 0
        try:
            from chisurf.plugins.modelling.proteinmc.model import load_json
            payload = load_json(filename)
            flexfit = payload.get("FlexFit", {}) or {}
            entry = flexfit.get(set_name, {})
            if not isinstance(entry, dict):
                return 0, 0
            residues = entry.get("Flexible residues", []) or []
            total = self._estimate_total_residues(filename)
            return len(residues), total
        except Exception:
            return 0, 0

    def _estimate_total_residues(self, labeling_file: str) -> int:
        """Estimate total residues from Positions section (upper bound)."""
        try:
            from chisurf.plugins.modelling.proteinmc.model import load_json
            payload = load_json(labeling_file)
            positions = payload.get("Positions", {}) or {}
            # Count unique residue numbers (upper bound on structure residues)
            seen: set[tuple[str, int]] = set()
            for pos in positions.values():
                if isinstance(pos, dict):
                    chain = str(pos.get("chain_identifier", "")).strip()
                    res_num = int(pos.get("residue_seq_number", 0))
                    seen.add((chain, res_num))
            return len(seen) if seen else 0
        except Exception:
            return 0

    @QtCore.Slot()
    def on_edit_labeling_json(self) -> None:
        """Open the selected FPS JSON file in an existing ChiSurf editor."""

        filename = self.labeling_edit.text().strip()
        if not filename:
            self.on_browse_labeling()
            filename = self.labeling_edit.text().strip()
        if not filename:
            return
        try:
            from chisurf.plugins.modelling.fps_json_editor.label_structure import LabelStructure

            editor = LabelStructure()
            editor.onLoadJSON(filename)
        except Exception:
            from chisurf.plugins.misc.code_editor import CodeEditor

            editor = CodeEditor(language="JSON")
            editor.open_file(filename)
        editor.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        editor.show()
        self._labeling_json_editor = editor

    @QtCore.Slot()
    def on_browse_structure(self) -> None:
        """Choose a structure file."""
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open ProteinMC structure",
            "",
            "Structures (*.pdb *.pqr *.cif);;All files (*)",
        )[0]
        if filename:
            self.structure_edit.setText(filename)

    @QtCore.Slot()
    def on_browse_labeling(self) -> None:
        """Choose an FPS JSON labeling file."""
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open FPS JSON labeling file",
            "",
            "FPS JSON (*.json *.fps.json);;All files (*)",
        )[0]
        if filename:
            self.labeling_edit.setText(filename)

    @QtCore.Slot()
    def start_proteinmc(self, output_directory: str | Path | None = None) -> None:
        """Start a ProteinMC run in a worker thread."""
        if self._thread is not None:
            return
        if output_directory is not None:
            self._sampling_directory = Path(output_directory)
        if self._sampling_directory is None:
            target_dir, _ = cs.gui.widgets.get_directory(caption="Select ProteinMC Sampling Output Folder")
            if target_dir is None:
                self.status_label.setText("ProteinMC sampling canceled")
                return
            self._sampling_directory = Path(target_dir)
        self._sampling_directory.mkdir(parents=True, exist_ok=True)
        output = self._sampling_directory / "proteinmc.rmf3"

        continuing = self.proteinmc_structure is not None and bool(self.trajectory_frames)
        self._continuing_run = continuing
        source = self.proteinmc_structure if continuing else (self.structure_edit.text().strip() or self.structure)
        if not source:
            self.status_label.setText("Select a structure or enter a PDB ID first")
            return
        settings = {
            "n_iter": self.n_iter_spin.value(),
            "n_out": self.n_out_spin.value(),
            "pdbOut": self.n_written_spin.value(),
            "scale": self.scale_spin.value(),
            "kt": self.kt_spin.value(),
            "labeling_weight": self.labeling_weight_spin.value(),
            "potentials": self._potential_settings(),
        }
        labeling = self.labeling_edit.text().strip() or None
        initial_frames = []
        if continuing:
            initial_frames = [np.asarray(frame, dtype=float) for frame in self.trajectory_frames[:-1]]
            self._resume_rmsd = list(self.rmsd)
            self._resume_drmsd = list(self.drmsd)
            self._resume_energy = list(self.energy)
            self._resume_chi2r = list(self.chi2r)
        else:
            if self.viewer is not None and self._chimol_object_id is not None:
                try:
                    self.viewer.remove_object(self._chimol_object_id)
                except Exception:
                    pass
            self._live_frames = []
            self.trajectory_frames = []
            self.current_frame_index = 0
            self.proteinmc_structure = None
            self._chimol_object_id = None
            self._resume_rmsd = []
            self._resume_drmsd = []
            self._resume_energy = []
            self._resume_chi2r = []
        self._chimol_update_pending = False
        self._plot_update_pending = False
        self._pending_xyz = None
        self._last_chimol_update_ms = 0
        self._chimol_frame_count = 0
        if not continuing and isinstance(source, (str, Path)):
            try:
                self._load_starting_structure(str(source))
            except Exception:
                pass
        if not continuing:
            self.rmsd = []
            self.drmsd = []
            self.energy = []
            self.chi2r = []
        start_frame_count = len(self.trajectory_frames)
        self.progress_bar.setRange(0, start_frame_count + self.n_written_spin.value() + 1)
        self.progress_bar.setValue(start_frame_count)
        verb = "continuing" if continuing else "running"
        run_text = f", runs={self._requested_run_count}" if self._requested_run_count > 1 else ""
        self.status_label.setText(f"ProteinMC {verb}{run_text}: {self._sampling_directory}")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self._thread = QtCore.QThread(self)
        flexfit_set = (
            self.flexfit_set_combo.currentText()
            if self.flexfit_use_check.isChecked()
            else ""
        )
        self._worker = _ProteinMCWorker(
            structure_source=source,
            labeling_file=labeling,
            score_set=self.score_set_combo.currentText(),
            flexfit_set=flexfit_set,
            settings=settings,
            output_file=output,
            initial_frames=initial_frames,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.prepared.connect(self.on_prepared)
        self._worker.progress.connect(self.on_progress)
        self._worker.finished.connect(self.on_finished)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._clear_thread)
        self._thread.start()

    def run_sampling(
        self,
        output_directory: str | Path | None = None,
        run_count: int = 1,
        n_iter: int | None = None,
    ) -> None:
        """Run ProteinMC from ChiSurf's generic Sampling button."""

        try:
            self._requested_run_count = max(1, int(run_count))
        except Exception:
            self._requested_run_count = 1
        if n_iter is not None:
            try:
                self.n_iter_spin.setValue(max(1, int(n_iter)))
            except Exception:
                pass
        self.start_proteinmc(output_directory=output_directory)

    @QtCore.Slot()
    def stop_proteinmc(self) -> None:
        """Stop the active ProteinMC run."""
        if self._worker is not None:
            self._worker.stop()
        self.status_label.setText("Stopping ProteinMC ...")

    @QtCore.Slot(object)
    def on_prepared(self, structure) -> None:
        """Initialize structure metadata before live coordinate frames arrive."""
        self.proteinmc_structure = structure
        if self.viewer is not None and self._chimol_object_id is None:
            self._chimol_object_id = self.viewer.add_structure(structure, name="ProteinMC")
            try:
                self.viewer.set_representation("atoms", object_id=self._chimol_object_id)
            except Exception:
                pass
        self.status_label.setText("ProteinMC prepared starting structure")

    @QtCore.Slot(object)
    def on_progress(self, progress: ProteinMCProgress) -> None:
        """Update arrays, plots, and Chimol from a progress payload.

        Heavy work (Chimol OpenGL upload and pyqtgraph setData calls) is
        throttled to ``update_interval_ms`` so the main UI thread stays
        responsive while the worker keeps emitting.
        """
        self.rmsd = self._resume_rmsd + progress.rmsd
        self.drmsd = self._resume_drmsd + progress.drmsd
        self.energy = self._resume_energy + progress.energies
        self.chi2r = self._resume_chi2r + progress.labeling_energies
        if progress.xyz is not None:
            self.trajectory_frames.append(np.asarray(progress.xyz, dtype=float))
            self.current_frame_index = len(self.trajectory_frames) - 1
        frame_index = len(self.trajectory_frames)
        self.progress_bar.setValue(frame_index)
        self.status_label.setText(
            f"Frame {frame_index} energy={progress.energy:.4g} "
            f"labeling={progress.labeling_energy:.4g}"
        )
        if self.viewer is not None and progress.xyz is not None:
            self._schedule_chimol_update(progress.xyz)
        self._schedule_plot_update()

    def _schedule_chimol_update(self, xyz: np.ndarray) -> None:
        """Coalesce Chimol updates to keep the UI thread responsive.

        The Chimol OpenGL rebuild is O(N) per call, so we throttle updates
        to at most one per ``update_interval_ms`` milliseconds. The latest
        xyz always wins: subsequent calls overwrite the pending payload.
        """
        now = QtCore.QDateTime.currentMSecsSinceEpoch()
        elapsed = now - self._last_chimol_update_ms
        self._pending_xyz = np.asarray(xyz, dtype=float)
        if self._chimol_update_pending:
            return
        if elapsed >= self.update_interval_ms:
            self._flush_chimol_update()
            return
        # Schedule a deferred flush so the latest xyz is rendered.
        self._chimol_update_pending = True
        delay = max(0, int(self.update_interval_ms - elapsed))
        QtCore.QTimer.singleShot(delay, self._flush_chimol_update)

    def _flush_chimol_update(self) -> None:
        self._chimol_update_pending = False
        pending = getattr(self, "_pending_xyz", None)
        self._pending_xyz = None
        if pending is None or self.viewer is None:
            return
        self._last_chimol_update_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
        try:
            self._update_chimol(pending)
        except Exception:
            pass

    def _schedule_plot_update(self) -> None:
        """Throttle the trajectory-plot updates to avoid main-thread stalls."""
        now = QtCore.QDateTime.currentMSecsSinceEpoch()
        elapsed = now - self._last_plot_update_ms
        if self._plot_update_pending:
            return
        if elapsed >= self.update_interval_ms:
            self._plot_update_pending = True
            QtCore.QTimer.singleShot(0, self._flush_plot_update)
            return
        self._plot_update_pending = True
        delay = max(0, int(self.update_interval_ms - elapsed))
        QtCore.QTimer.singleShot(delay, self._flush_plot_update)

    def _flush_plot_update(self) -> None:
        self._plot_update_pending = False
        self._last_plot_update_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
        for plot in list(getattr(self.fit, "plots", []) or []):
            try:
                plot.update()
            except Exception:
                pass

    def _update_chimol(self, xyz: np.ndarray) -> None:
        """Push one frame to the embedded Chimol viewer by appending it
        to the trajectory, so the frame slider and play/pause buttons

        can navigate all frames accumulated so far.
        """
        if self.viewer is None:
            return
        if self._chimol_object_id is None:
            if self.proteinmc_structure is not None:
                self._chimol_object_id = self.viewer.add_structure(self.proteinmc_structure, name="ProteinMC")
            else:
                self._chimol_object_id = self.viewer.add_coordinates(
                    xyz,
                    name="ProteinMC",
                )
        try:
            append_frame = getattr(self.viewer, "append_frame", None)
            if callable(append_frame):
                self._chimol_frame_count = append_frame(
                    np.asarray(xyz, dtype=float),
                    object_id=self._chimol_object_id,
                )
            else:
                self.viewer.set_frames(
                    np.asarray(xyz, dtype=float)[np.newaxis, :, :],
                    object_id=self._chimol_object_id,
                    active_frame=0,
                )
                self._chimol_frame_count = 1
        except TypeError:
            self.viewer.set_frames(np.asarray(xyz, dtype=float)[np.newaxis, :, :], object_id=self._chimol_object_id)
            set_active_frame = getattr(self.viewer, "set_active_frame", None)
            if set_active_frame is not None:
                set_active_frame(0, object_id=self._chimol_object_id)
            self._chimol_frame_count = 1

    @QtCore.Slot(object)
    def on_finished(self, result) -> None:
        """Handle successful ProteinMC completion."""
        self.proteinmc_structure = getattr(result, "structure", self.proteinmc_structure)
        self.status_label.setText(f"ProteinMC finished: {result.output_file}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._continuing_run = False
        # Force a final flush of any deferred Chimol/plot updates.
        self._pending_xyz = None
        self._last_chimol_update_ms = 0
        if self._chimol_update_pending:
            self._flush_chimol_update()
        if self._plot_update_pending:
            self._flush_plot_update()
        else:
            self._flush_plot_update()
        if self.viewer is not None and self._chimol_object_id is not None and self.trajectory_frames:
            try:
                frames = np.asarray(self.trajectory_frames, dtype=float)
                active_frame = min(max(0, self.current_frame_index), len(frames) - 1)
                self.viewer.set_frames(
                    frames,
                    object_id=self._chimol_object_id,
                    active_frame=active_frame,
                )
                self._chimol_frame_count = int(len(frames))
            except Exception:
                pass
        for plot in list(getattr(self.fit, "plots", []) or []):
            if getattr(plot, "name", "") != "Structure":
                continue
            try:
                controller = getattr(plot, "plot_controller", None)
                if controller is not None:
                    controller.refresh_from_viewer()
            except Exception:
                pass
        self._autosave_sampling_project()

    @QtCore.Slot(str)
    def on_failed(self, message: str) -> None:
        """Handle ProteinMC worker failures."""
        self.status_label.setText(f"ProteinMC failed: {message}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    @QtCore.Slot()
    def _clear_thread(self) -> None:
        """Clear completed worker references."""
        self._thread = None
        self._worker = None

    def update_model(self, **kwargs):
        """ProteinMC state is updated by worker progress callbacks."""
        return None

    def get_state(self) -> dict:
        """Return project-serializable ProteinMC widget state."""

        return {
            "proteinmc": {
                "structure_source": self.structure_edit.text().strip(),
                "labeling_file": self.labeling_edit.text().strip(),
                "score_set": self.score_set_combo.currentText(),
                "use_flexfit": self.flexfit_use_check.isChecked(),
                "flexfit_set": self.flexfit_set_combo.currentText(),
                "output_directory": str(self._sampling_directory) if self._sampling_directory else "",
                "settings": {
                    "n_iter": self.n_iter_spin.value(),
                    "n_out": self.n_out_spin.value(),
                    "n_written": self.n_written_spin.value(),
                    "pdbOut": self.n_written_spin.value(),
                    "scale": self.scale_spin.value(),
                    "kt": self.kt_spin.value(),
                    "labeling_weight": self.labeling_weight_spin.value(),
                    "potentials": self._potential_settings(),
                },
            }
        }

    def set_state(self, state: dict) -> None:
        """Restore project-serialized ProteinMC widget state."""

        payload = state.get("proteinmc", state) if isinstance(state, dict) else {}
        if not isinstance(payload, dict):
            return

        structure_source = payload.get("structure_source")
        labeling_file = payload.get("labeling_file")
        output_file = payload.get("output_file")
        output_directory = payload.get("output_directory")
        if structure_source:
            self.structure_edit.setText(str(structure_source))
            self._load_starting_structure(str(structure_source))
        if labeling_file:
            self.labeling_edit.setText(str(labeling_file))
        score_set = payload.get("score_set", "")
        if score_set:
            idx = self.score_set_combo.findText(score_set)
            if idx >= 0:
                self.score_set_combo.setCurrentIndex(idx)
        use_flexfit = payload.get("use_flexfit", False)
        flexfit_set = payload.get("flexfit_set", "")
        self.flexfit_use_check.setChecked(bool(use_flexfit))
        if flexfit_set:
            idx = self.flexfit_set_combo.findText(flexfit_set)
            if idx >= 0:
                self.flexfit_set_combo.setCurrentIndex(idx)
        if output_directory:
            self._sampling_directory = Path(output_directory)
        elif output_file:
            self._sampling_directory = Path(output_file).parent

        settings = payload.get("settings") or {}
        if not isinstance(settings, dict):
            settings = {}
        for spin, key in (
            (self.n_iter_spin, "n_iter"),
            (self.n_out_spin, "n_out"),
            (self.n_written_spin, "n_written"),
            (self.scale_spin, "scale"),
            (self.kt_spin, "kt"),
            (self.labeling_weight_spin, "labeling_weight"),
        ):
            if key not in settings and key == "n_written":
                value = settings.get("pdbOut")
            else:
                value = settings.get(key)
            if value is None:
                continue
            try:
                spin.setValue(value)
            except Exception:
                pass

        self._restore_potential_settings(settings.get("potentials", []))

        self._refresh_structure_plots()

    def _restore_potential_settings(self, potentials: object) -> None:
        """Restore potential UI controls from serialized settings."""

        aliases = {
            "default": "default",
            "clash potential": "default",
            "clash-potential": "default",
            "hbond": "hbond",
            "h-bond": "hbond",
            "h-potential": "hbond",
            "mj": "mj",
            "miyazawa-jernigan": "mj",
            "unres": "unres",
            "iso-unres": "unres",
        }
        table = getattr(self, "potential_table", None)
        if table is not None:
            table.setRowCount(0)
        if not isinstance(potentials, list):
            self._resize_potential_table()
            return
        for item in potentials:
            if not isinstance(item, dict):
                continue
            raw_name = str(item.get("name", "")).lower()
            key = aliases.get(raw_name, raw_name)
            if key not in self._POTENTIAL_SPECS:
                continue
            spec = self._POTENTIAL_SPECS[key]
            try:
                weight = float(item.get("weight", spec["weight"]))
            except Exception:
                weight = float(spec["weight"])
            settings = item.get("settings") or {}
            if not isinstance(settings, dict):
                settings = {}
            restored_settings = dict(spec["params"])
            for param_name, value in settings.items():
                if param_name in restored_settings:
                    restored_settings[param_name] = value
            self._add_potential_row(key, weight=weight, settings=restored_settings)
        self._resize_potential_table()

    def _load_starting_structure(self, filename: str) -> None:
        """Load and display the starting structure without running ProteinMC."""

        try:
            from chisurf.core.structure import Structure

            structure = Structure(filename)
        except Exception as exc:
            try:
                cs.logging.warning(f"ProteinMC: could not load starting structure '{filename}': {exc}")
            except Exception:
                pass
            return

        self.proteinmc_structure = structure
        self.trajectory_frames = []
        if self.viewer is None:
            return

        try:
            if self._chimol_object_id is not None:
                self.viewer.remove_object(self._chimol_object_id)
        except Exception:
            pass
        try:
            self._chimol_object_id = self.viewer.add_structure(structure, name="ProteinMC")
            self.viewer.set_representation("atoms", object_id=self._chimol_object_id)
        except Exception:
            self._chimol_object_id = None

    def _refresh_structure_plots(self) -> None:
        """Refresh any external ProteinMC structure plot tabs."""

        for plot in list(getattr(self.fit, "plots", []) or []):
            if getattr(plot, "name", "") != "Structure":
                continue
            try:
                plot.object_id = None
                plot.update_all()
            except Exception:
                pass

    def _autosave_sampling_project(self) -> None:
        """Autosave the current ChiSurf project into the ProteinMC sampling folder."""

        if self._sampling_directory is None:
            return
        try:
            from chisurf.macros.core_fit import save_project

            save_project(target_path=str(self._sampling_directory), project_name="project")
        except Exception as exc:
            try:
                cs.logging.warning(f"ProteinMC: could not autosave project: {exc}")
            except Exception:
                pass

    def update_widgets(self) -> None:
        """Update parameter widgets from the base model."""
        super().update_widgets()

    def update(self) -> None:
        """Refresh widgets and plots."""
        self.update_widgets()
        self.update_plots()
