"""Panel for managing experimental FRET distances and score sets."""

from __future__ import annotations

import logging
from typing import Any

from qtpy import QtCore, QtWidgets

import chisurf.core.fio
import chisurf.gui.widgets
import chisurf.gui.widgets.general

logger = logging.getLogger("chisurf.plugins.modelling.fret")


class DistancePanel(QtWidgets.QWidget):
    """A widget for managing distance restraints and score sets (χ² groups).

    Features a detailed table widget showing all restraints, supporting inline
    editing for distances, errors, and Forster radii.

    Attributes
    ----------
    distance_added : QtCore.Signal
        Emitted when a new distance is added. Passes (name, params, score_set).
    distance_removed : QtCore.Signal
        Emitted when a distance is removed. Passes (name).
    score_set_added : QtCore.Signal
        Emitted when a new scoring group/set is added. Passes (name).
    score_set_removed : QtCore.Signal
        Emitted when a scoring group/set is removed. Passes (name).
    distance_modified : QtCore.Signal
        Emitted when a distance value is modified inline in the table. Passes (name, field_key, value).
    """

    distance_added = QtCore.Signal(str, dict, str)
    distance_removed = QtCore.Signal(str)
    score_set_added = QtCore.Signal(str)
    score_set_removed = QtCore.Signal(str)
    distance_modified = QtCore.Signal(str, str, float)

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

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the DistancePanel layout and widgets."""
        super().__init__(parent)
        self._populating = False
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Input parameters GroupBox
        param_group = QtWidgets.QGroupBox("Distance Restraint Parameters", self)
        grid = QtWidgets.QGridLayout(param_group)
        grid.setSpacing(6)

        # Type ComboBox
        grid.addWidget(QtWidgets.QLabel("Type:"), 0, 0)
        self.distance_type_combo = QtWidgets.QComboBox()
        self.distance_type_combo.addItems(["dRDA", "dRDAE", "dRMP", "pRDA"])
        grid.addWidget(self.distance_type_combo, 0, 1, 1, 3)

        # Label 1 and Label 2 selectors
        grid.addWidget(QtWidgets.QLabel("Label 1:"), 1, 0)
        self.label1_combo = QtWidgets.QComboBox()
        grid.addWidget(self.label1_combo, 1, 1, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Label 2:"), 2, 0)
        self.label2_combo = QtWidgets.QComboBox()
        grid.addWidget(self.label2_combo, 2, 1, 1, 3)

        # Spinboxes for distance parameters
        grid.addWidget(QtWidgets.QLabel("Distance (Å):"), 3, 0)
        self.distance_spin = QtWidgets.QDoubleSpinBox()
        self.distance_spin.setRange(0, 999.0)
        self.distance_spin.setValue(50.0)
        self.distance_spin.setDecimals(1)
        grid.addWidget(self.distance_spin, 3, 1)

        grid.addWidget(QtWidgets.QLabel("err⁻:"), 3, 2)
        self.error_neg_spin = QtWidgets.QDoubleSpinBox()
        self.error_neg_spin.setRange(0, 99999.0)
        self.error_neg_spin.setValue(5.0)
        self.error_neg_spin.setDecimals(1)
        grid.addWidget(self.error_neg_spin, 3, 3)

        grid.addWidget(QtWidgets.QLabel("err⁺:"), 3, 4)
        self.error_pos_spin = QtWidgets.QDoubleSpinBox()
        self.error_pos_spin.setRange(0, 9999.0)
        self.error_pos_spin.setValue(5.0)
        self.error_pos_spin.setDecimals(1)
        grid.addWidget(self.error_pos_spin, 3, 5)

        # Forster radius (R0)
        grid.addWidget(QtWidgets.QLabel("Forster radius (Å):"), 4, 0)
        self.forster_radius_spin = QtWidgets.QDoubleSpinBox()
        self.forster_radius_spin.setRange(0, 999.0)
        self.forster_radius_spin.setValue(52.0)
        self.forster_radius_spin.setSingleStep(0.5)
        self.forster_radius_spin.setDecimals(1)
        grid.addWidget(self.forster_radius_spin, 4, 1, 1, 2)

        layout.addWidget(param_group)

        # Score sets and filtering row
        score_layout = QtWidgets.QHBoxLayout()
        score_layout.addWidget(QtWidgets.QLabel("Scoring group / set:"))
        self.score_set_combo = QtWidgets.QComboBox()
        self.score_set_combo.currentTextChanged.connect(self.onScoreSetSelectionChanged)
        score_layout.addWidget(self.score_set_combo, stretch=1)

        self.add_score_set_btn = QtWidgets.QPushButton("+")
        self.add_score_set_btn.setToolTip("Add a new score set")
        self.add_score_set_btn.setFixedWidth(30)
        self.add_score_set_btn.clicked.connect(self.onAddScoreSet)
        score_layout.addWidget(self.add_score_set_btn)

        self.remove_score_set_btn = QtWidgets.QPushButton("-")
        self.remove_score_set_btn.setToolTip("Remove current score set")
        self.remove_score_set_btn.setFixedWidth(30)
        self.remove_score_set_btn.clicked.connect(self.onRemoveScoreSet)
        score_layout.addWidget(self.remove_score_set_btn)

        layout.addLayout(score_layout)

        # Add distance button
        self.add_distance_btn = QtWidgets.QPushButton("Add distance")
        self.add_distance_btn.clicked.connect(self.onAddDistance)
        layout.addWidget(self.add_distance_btn)

        # Distances table
        layout.addWidget(QtWidgets.QLabel("Distances (double-click fields to edit, right-click to delete):"))
        self.distances_table = QtWidgets.QTableWidget(0, 9)
        self.distances_table.setHorizontalHeaderLabels([
            "Name", "Label 1", "Label 2", "Type", "d (Å)", "err⁻", "err⁺", "R₀ (Å)", "Score set"
        ])
        self.distances_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.distances_table.itemChanged.connect(self.onTableItemChanged)
        self.distances_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.distances_table.customContextMenuRequested.connect(self.onTableContextMenu)
        layout.addWidget(self.distances_table)

    def onScoreSetSelectionChanged(self, text: str) -> None:
        """Handle scoring group filter combobox change."""
        self.distance_modified.emit("", "", 0.0)  # Simply trigger parent refresh

    def onAddScoreSet(self) -> None:
        """Prompt user to add a new score set group."""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Scoring Group", "Scoring group name:"
        )
        if ok and name.strip():
            self.score_set_added.emit(name.strip())

    def onRemoveScoreSet(self) -> None:
        """Remove currently selected score set group."""
        name = self.score_set_combo.currentText()
        if not name or name == "All distances":
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Remove Scoring Group?",
            f"Are you sure you want to remove scoring group '{name}'?",
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.score_set_removed.emit(name)

    def onAddDistance(self) -> None:
        """Extract inputs and emit distance_added signal."""
        l1 = self.label1_combo.currentText()
        l2 = self.label2_combo.currentText()
        if not l1 or not l2:
            self._show_status("Please select two labels.", "warning")
            return
        if l1 == l2:
            self._show_status("Label 1 and Label 2 must be different.", "warning")
            return

        name = f"{l1}_{l2}"
        dtype = self.distance_type_combo.currentText()
        # Map visual type to backend key
        mapped_type = "RDAMean"
        if dtype == "dRDA":
            mapped_type = "RDAMean"
        elif dtype == "dRDAE":
            mapped_type = "RDAMeanE"
        elif dtype == "dRMP":
            mapped_type = "Rmp"
        elif dtype == "pRDA":
            mapped_type = "pRDA"

        params = {
            "Forster_radius": float(self.forster_radius_spin.value()),
            "distance_type": mapped_type,
            "position1_name": l1,
            "position2_name": l2,
        }

        if mapped_type == "pRDA":
            fn = chisurf.gui.widgets.get_filename(
                "DA-Distance distribution (1st column RDA, 2nd pRDA)"
            )
            if not fn:
                return
            try:
                csv = chisurf.core.fio.ascii.Csv(filename=fn, skiprows=1)
                params['rda'] = list(csv.data[0])
                params['prda'] = list(csv.data[1])
            except Exception as e:
                self._show_status(f"CSV Read Error: Could not load distribution: {str(e)}", "error")
                return
        else:
            params['distance'] = float(self.distance_spin.value())
            params['error_neg'] = float(self.error_neg_spin.value())
            params['error_pos'] = float(self.error_pos_spin.value())

        score_group = self.score_set_combo.currentText()
        if score_group == "All distances":
            score_group = ""

        self.distance_added.emit(name, params, score_group)

    def onTableItemChanged(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Handle inline editing on cells of columns 4-7."""
        if self._populating:
            return

        row = item.row()
        col = item.column()
        # Editable columns: d (4), err- (5), err+ (6), R0 (7)
        if col not in (4, 5, 6, 7):
            return

        name_item = self.distances_table.item(row, 0)
        if name_item is None:
            return
        dist_name = name_item.text()

        try:
            val = float(item.text())
        except ValueError:
            self._show_status("Please enter a valid float number.", "warning")
            return

        fields = {
            4: "distance",
            5: "error_neg",
            6: "error_pos",
            7: "Forster_radius"
        }
        self.distance_modified.emit(dist_name, fields[col], val)

    def onTableContextMenu(self, pos: QtCore.QPoint) -> None:
        """Provide a right click menu on distance rows to delete them."""
        item = self.distances_table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        name_item = self.distances_table.item(row, 0)
        if name_item is None:
            return
        dist_name = name_item.text()

        menu = QtWidgets.QMenu(self)
        delete_action = menu.addAction(f"Delete distance restraint '{dist_name}'")
        action = menu.exec_(self.distances_table.mapToGlobal(pos))
        if action == delete_action:
            reply = QtWidgets.QMessageBox.question(
                self, "Remove Restraint?",
                f"Are you sure you want to remove restraint '{dist_name}'?",
                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.distance_removed.emit(dist_name)

    def update_labels(self, label_names: list[str]) -> None:
        """Update label selection combo boxes with active label list."""
        self.label1_combo.clear()
        self.label2_combo.clear()
        self.label1_combo.addItems(label_names)
        self.label2_combo.addItems(label_names)

    def update_score_sets(self, score_set_names: list[str]) -> None:
        """Update scoring group filter combo box."""
        self.score_set_combo.blockSignals(True)
        old_selection = self.score_set_combo.currentText()
        self.score_set_combo.clear()
        self.score_set_combo.addItem("All distances")
        self.score_set_combo.addItems(score_set_names)
        idx = self.score_set_combo.findText(old_selection)
        if idx >= 0:
            self.score_set_combo.setCurrentIndex(idx)
        else:
            self.score_set_combo.setCurrentIndex(0)
        self.score_set_combo.blockSignals(False)

    def update_distances_table(
        self,
        distances: dict[str, dict[str, Any]],
        score_sets: dict[str, dict[str, Any]]
    ) -> None:
        """Repopulate the table with filtered restraints based on active score set.

        Parameters
        ----------
        distances : dict
            Distances dictionary from FpsJsonModel.
        score_sets : dict
            Score sets dictionary from FpsJsonModel.
        """
        self._populating = True
        self.distances_table.setRowCount(0)

        # Apply score set filter
        active_set = self.score_set_combo.currentText()
        if active_set and active_set != "All distances" and active_set in score_sets:
            group = score_sets[active_set]
            keys_to_show = [k for k in group.get("distances", []) if k in distances]
        else:
            keys_to_show = list(distances.keys())

        for key in keys_to_show:
            d = distances[key]
            row = self.distances_table.rowCount()
            self.distances_table.insertRow(row)

            # Name, Label 1, Label 2, Type (Non-editable)
            for c, val in enumerate([
                key,
                d.get("position1_name", ""),
                d.get("position2_name", ""),
                d.get("distance_type", "")
            ]):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.distances_table.setItem(row, c, item)

            # d, err-, err+, R0 (Editable for non-pRDA)
            is_prda = d.get("distance_type") == "pRDA"

            d_val = "N/A" if is_prda else f"{d.get('distance', 0.0):.1f}"
            err_neg_val = "N/A" if is_prda else f"{d.get('error_neg', 0.0):.1f}"
            err_pos_val = "N/A" if is_prda else f"{d.get('error_pos', 0.0):.1f}"
            r0_val = f"{d.get('Forster_radius', 0.0):.1f}"

            for c, val in enumerate([d_val, err_neg_val, err_pos_val, r0_val], start=4):
                item = QtWidgets.QTableWidgetItem(str(val))
                if is_prda or c == 8:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.distances_table.setItem(row, c, item)

            # Score set membership
            belonging_sets = []
            for name, gs in score_sets.items():
                if key in gs.get("distances", []):
                    belonging_sets.append(name)
            set_val = ", ".join(belonging_sets)
            set_item = QtWidgets.QTableWidgetItem(set_val)
            set_item.setFlags(set_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.distances_table.setItem(row, 8, set_item)

        self._populating = False
