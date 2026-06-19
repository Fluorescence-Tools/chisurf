"""Panel for managing experimental FRET distances and score sets."""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from qtpy import QtCore, QtWidgets

import chisurf.core.fio
import chisurf.gui.widgets

logger = logging.getLogger("chisurf.plugins.modelling.fret")


class CenteredCheckBox(QtWidgets.QWidget):
    """A wrapper widget that centers a QCheckBox inside a table cell."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        self.checkbox = QtWidgets.QCheckBox()
        layout.addWidget(self.checkbox)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)


class DistanceDetailSettingsDialog(QtWidgets.QDialog):
    """Modal dialog for editing detailed FRET distance restraint parameters."""

    def __init__(
        self,
        params: dict,
        distance_type: str = "dRDA",
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Distance Restraint Details")
        self.resize(380, 260)

        self.params = params.copy()
        self.distance_type = distance_type

        main_layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        # Forster radius (R0) is always editable
        self.r0_spin = QtWidgets.QDoubleSpinBox()
        self.r0_spin.setRange(0.0, 999.0)
        self.r0_spin.setValue(float(self.params.get("Forster_radius", 52.0)))
        self.r0_spin.setDecimals(1)
        form_layout.addRow("Forster Radius (R₀) (Å):", self.r0_spin)

        is_prda = (distance_type == "pRDA")

        if not is_prda:
            self.d_spin = QtWidgets.QDoubleSpinBox()
            self.d_spin.setRange(0.0, 999.0)
            self.d_spin.setValue(float(self.params.get("distance", 50.0)))
            self.d_spin.setDecimals(1)
            form_layout.addRow("Distance (d) (Å):", self.d_spin)

            self.err_neg_spin = QtWidgets.QDoubleSpinBox()
            self.err_neg_spin.setRange(0.0, 99999.0)
            self.err_neg_spin.setValue(float(self.params.get("error_neg", 5.0)))
            self.err_neg_spin.setDecimals(1)
            form_layout.addRow("Error Neg (err⁻) (Å):", self.err_neg_spin)

            self.err_pos_spin = QtWidgets.QDoubleSpinBox()
            self.err_pos_spin.setRange(0.0, 9999.0)
            self.err_pos_spin.setValue(float(self.params.get("error_pos", 5.0)))
            self.err_pos_spin.setDecimals(1)
            form_layout.addRow("Error Pos (err⁺) (Å):", self.err_pos_spin)
        else:
            # For pRDA, show a load button
            self.load_dist_btn = QtWidgets.QPushButton("Load DA Distribution...")
            self.load_dist_btn.clicked.connect(self.onLoadDistribution)
            form_layout.addRow("Distribution:", self.load_dist_btn)

            self.status_label = QtWidgets.QLabel()
            if "rda" in self.params and "prda" in self.params:
                self.status_label.setText(f"Loaded: {len(self.params['rda'])} points")
            else:
                self.status_label.setText("No distribution loaded")
            form_layout.addRow("", self.status_label)

        main_layout.addLayout(form_layout)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def onLoadDistribution(self) -> None:
        fn = chisurf.gui.widgets.get_filename(
            "DA-Distance distribution (1st column RDA, 2nd pRDA)",
            "CSV/Text Files (*.csv *.txt)"
        )
        if fn:
            try:
                csv = chisurf.core.fio.ascii.Csv(filename=fn, skiprows=1)
                self.params['rda'] = list(csv.data[0])
                self.params['prda'] = list(csv.data[1])
                self.status_label.setText(f"Loaded: {len(self.params['rda'])} points")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load distribution: {str(e)}")

    def get_settings(self) -> dict:
        res = {
            "Forster_radius": self.r0_spin.value()
        }
        if self.distance_type != "pRDA":
            res["distance"] = self.d_spin.value()
            res["error_neg"] = self.err_neg_spin.value()
            res["error_pos"] = self.err_pos_spin.value()
        else:
            if "rda" in self.params:
                res["rda"] = self.params["rda"]
            if "prda" in self.params:
                res["prda"] = self.params["prda"]
        return res


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

    def __init__(
        self,
        position_panel: QtWidgets.QWidget | None = None,
        mol_view_3d: Any | None = None,
        parent: QtWidgets.QWidget | None = None
    ) -> None:
        """Initialize the DistancePanel layout and widgets."""
        super().__init__(parent)
        self._position_panel = position_panel
        self._mol_view_3d = mol_view_3d
        self._populating = False
        self._block_table_signals = 0
        self.label_names: list[str] = []
        self.score_set_names: list[str] = []
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Toolbar
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
            QPushButton#scoreSetTinyButton {
                background-color: rgba(45, 45, 45, 210);
                border: 1px solid rgba(255, 255, 255, 45);
                border-radius: 4px;
                min-width: 24px;
                max-width: 24px;
                padding: 2px;
            }
            """
        )

        self.add_row_action = self.toolbar.addAction("➕ Add Row")
        self.add_row_action.triggered.connect(self.onAddRowTriggered)
        self.toolbar.addSeparator()

        # Score set controls in toolbar
        self.toolbar.addWidget(QtWidgets.QLabel(" Scoring group / set: "))
        self.score_set_combo = QtWidgets.QComboBox()
        self.score_set_combo.currentTextChanged.connect(self.onScoreSetSelectionChanged)
        self.toolbar.addWidget(self.score_set_combo)

        self.add_score_set_btn = QtWidgets.QPushButton("+")
        self.add_score_set_btn.setObjectName("scoreSetTinyButton")
        self.add_score_set_btn.setFixedWidth(24)
        self.add_score_set_btn.clicked.connect(self.onAddScoreSet)
        self.toolbar.addWidget(self.add_score_set_btn)

        self.remove_score_set_btn = QtWidgets.QPushButton("-")
        self.remove_score_set_btn.setObjectName("scoreSetTinyButton")
        self.remove_score_set_btn.setFixedWidth(24)
        self.remove_score_set_btn.clicked.connect(self.onRemoveScoreSet)
        self.toolbar.addWidget(self.remove_score_set_btn)

        for widget in self.toolbar.children():
            if isinstance(widget, QtWidgets.QToolButton):
                action = widget.defaultAction()
                if action is self.add_row_action:
                    widget.setObjectName("toolbarAddRow")
                    widget.setAutoRaise(True)

        layout.addWidget(self.toolbar)

        # Distances table
        self.distances_table = QtWidgets.QTableWidget(0, 8)
        self.distances_table.setHorizontalHeaderLabels([
            "Show", "Name", "Label 1", "Label 2", "Type", "Details", "Score set", ""
        ])
        self.distances_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.distances_table.horizontalHeader().setSectionResizeMode(7, QtWidgets.QHeaderView.Fixed)
        self.distances_table.setColumnWidth(7, 40)
        self.distances_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.distances_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.distances_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.distances_table.customContextMenuRequested.connect(self.onTableContextMenu)
        layout.addWidget(self.distances_table)

    def _setup_row_widgets(self, row: int) -> None:
        self._block_table_signals += 1
        try:
            # Show Checkbox (col 0)
            cb_widget = CenteredCheckBox()
            cb_widget.checkbox.setChecked(True)
            cb_widget.checkbox.toggled.connect(self.onRowVisibilityToggled)
            self.distances_table.setCellWidget(row, 0, cb_widget)

            # Name (col 1)
            name_item = QtWidgets.QTableWidgetItem("")
            default_params = {
                "Forster_radius": 52.0,
                "distance": 50.0,
                "error_neg": 5.0,
                "error_pos": 5.0,
                "distance_type": "RDAMean"
            }
            name_item.setData(QtCore.Qt.UserRole + 1, default_params)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.distances_table.setItem(row, 1, name_item)

            # Label 1 (col 2)
            l1_cb = QtWidgets.QComboBox()
            l1_cb.addItems(self.label_names)
            l1_cb.currentTextChanged.connect(self.onLabelComboChanged)
            self.distances_table.setCellWidget(row, 2, l1_cb)

            # Label 2 (col 3)
            l2_cb = QtWidgets.QComboBox()
            l2_cb.addItems(self.label_names)
            l2_cb.currentTextChanged.connect(self.onLabelComboChanged)
            self.distances_table.setCellWidget(row, 3, l2_cb)

            # Type (col 4)
            type_cb = QtWidgets.QComboBox()
            type_cb.addItems(["dRDA", "dRDAE", "dRMP", "pRDA"])
            type_cb.currentTextChanged.connect(self.onTypeComboChanged)
            self.distances_table.setCellWidget(row, 4, type_cb)

            # Details (col 5)
            details_btn = QtWidgets.QPushButton("Details...")
            details_btn.clicked.connect(self.onShowRowSettings)
            self.distances_table.setCellWidget(row, 5, details_btn)
            self._update_settings_tooltip(row, default_params, "dRDA")

            # Score set (col 6)
            set_cb = QtWidgets.QComboBox()
            set_cb.addItem("")
            set_cb.addItems(self.score_set_names)
            set_cb.currentTextChanged.connect(self.onScoreSetComboChanged)
            self.distances_table.setCellWidget(row, 6, set_cb)

            # Delete button (col 7)
            del_btn = QtWidgets.QPushButton("🗑️")
            del_btn.setFixedWidth(30)
            del_btn.clicked.connect(self.onDeleteRowClicked)
            self.distances_table.setCellWidget(row, 7, del_btn)

        finally:
            self._block_table_signals -= 1

    def _update_settings_tooltip(self, row: int, params: dict, distance_type: str) -> None:
        btn = self.distances_table.cellWidget(row, 5)
        if isinstance(btn, QtWidgets.QPushButton):
            tooltip_lines = [
                "<b>Distance restraint details:</b>",
                f"Type: {distance_type}",
                f"Forster radius (R₀): {params.get('Forster_radius', 52.0)} Å",
            ]
            if distance_type != "pRDA":
                tooltip_lines.extend([
                    f"Distance (d): {params.get('distance', 50.0)} Å",
                    f"Error Neg (err⁻): {params.get('error_neg', 5.0)} Å",
                    f"Error Pos (err⁺): {params.get('error_pos', 5.0)} Å",
                ])
            else:
                if "rda" in params:
                    tooltip_lines.append(f"Distribution: {len(params['rda'])} points")
                else:
                    tooltip_lines.append("Distribution: None loaded")
            btn.setToolTip("<br>".join(tooltip_lines))

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

    def onAddRowTriggered(self) -> None:
        self._add_empty_row()

    def _add_empty_row(self) -> None:
        self.distances_table.blockSignals(True)
        try:
            row = self.distances_table.rowCount()
            self.distances_table.insertRow(row)
            self._setup_row_widgets(row)
        finally:
            self.distances_table.blockSignals(False)

    def onRowVisibilityToggled(self, checked: bool) -> None:
        sender = self.sender()
        if not sender or self._block_table_signals > 0:
            return
        row = -1
        for r in range(self.distances_table.rowCount()):
            widget = self.distances_table.cellWidget(r, 0)
            if isinstance(widget, CenteredCheckBox) and widget.checkbox is sender:
                row = r
                break
        if row >= 0:
            self._trigger_row_change(row)

    def onLabelComboChanged(self, text: str) -> None:
        sender = self.sender()
        if not sender or self._block_table_signals > 0:
            return
        row = -1
        for r in range(self.distances_table.rowCount()):
            if (
                self.distances_table.cellWidget(r, 2) is sender
                or self.distances_table.cellWidget(r, 3) is sender
            ):
                row = r
                break
        if row >= 0:
            self._trigger_row_change(row)

    def onTypeComboChanged(self, text: str) -> None:
        sender = self.sender()
        if not sender or self._block_table_signals > 0:
            return
        row = -1
        for r in range(self.distances_table.rowCount()):
            if self.distances_table.cellWidget(r, 4) is sender:
                row = r
                break
        if row >= 0:
            self._trigger_row_change(row)

    def onScoreSetComboChanged(self, text: str) -> None:
        sender = self.sender()
        if not sender or self._block_table_signals > 0:
            return
        row = -1
        for r in range(self.distances_table.rowCount()):
            if self.distances_table.cellWidget(r, 6) is sender:
                row = r
                break
        if row >= 0:
            self._trigger_row_change(row)

    def _trigger_row_change(self, row: int) -> None:
        if row < 0:
            return

        # If they edit the placeholder row, insert a new placeholder!
        if row == self.distances_table.rowCount() - 1:
            self._add_empty_row()

        name_item = self.distances_table.item(row, 1)
        if not name_item:
            return
        old_name = name_item.text().strip()

        l1_cb = self.distances_table.cellWidget(row, 2)
        l2_cb = self.distances_table.cellWidget(row, 3)
        type_cb = self.distances_table.cellWidget(row, 4)
        set_cb = self.distances_table.cellWidget(row, 6)

        l1 = l1_cb.currentText().strip() if isinstance(l1_cb, QtWidgets.QComboBox) else ""
        l2 = l2_cb.currentText().strip() if isinstance(l2_cb, QtWidgets.QComboBox) else ""
        dtype = type_cb.currentText().strip() if isinstance(type_cb, QtWidgets.QComboBox) else "dRDA"
        score_group = set_cb.currentText().strip() if isinstance(set_cb, QtWidgets.QComboBox) else ""

        if not l1 or not l2:
            return

        if l1 == l2:
            self._show_status("Label 1 and Label 2 must be different.", "warning")
            return

        new_name = f"{l1}_{l2}"

        # Map display type to backend key
        mapped_type = "RDAMean"
        if dtype == "dRDA":
            mapped_type = "RDAMean"
        elif dtype == "dRDAE":
            mapped_type = "RDAMeanE"
        elif dtype == "dRMP":
            mapped_type = "Rmp"
        elif dtype == "pRDA":
            mapped_type = "pRDA"

        params = name_item.data(QtCore.Qt.UserRole + 1)
        if not isinstance(params, dict):
            params = {
                "Forster_radius": 52.0,
                "distance": 50.0,
                "error_neg": 5.0,
                "error_pos": 5.0,
            }

        params["position1_name"] = l1
        params["position2_name"] = l2
        params["distance_type"] = mapped_type

        # Check visibility state
        show_widget = self.distances_table.cellWidget(row, 0)
        visible = True
        if isinstance(show_widget, CenteredCheckBox):
            visible = show_widget.checkbox.isChecked()
        params["visible"] = visible

        self._block_table_signals += 1
        try:
            name_item.setText(new_name)
        finally:
            self._block_table_signals -= 1

        if old_name and old_name != new_name:
            self.distance_removed.emit(old_name)

        self.distance_added.emit(new_name, params, score_group)
        self._update_settings_tooltip(row, params, dtype)
        self.update_distance_lines()

    def onShowRowSettings(self) -> None:
        button = self.sender()
        if not button or self._block_table_signals > 0:
            return
        row = -1
        for r in range(self.distances_table.rowCount()):
            if self.distances_table.cellWidget(r, 5) is button:
                row = r
                break
        if row < 0:
            return

        name_item = self.distances_table.item(row, 1)
        if not name_item:
            return

        params = name_item.data(QtCore.Qt.UserRole + 1)
        if not isinstance(params, dict):
            params = {}

        type_cb = self.distances_table.cellWidget(row, 4)
        dtype = type_cb.currentText() if isinstance(type_cb, QtWidgets.QComboBox) else "dRDA"

        dialog = DistanceDetailSettingsDialog(params, distance_type=dtype, parent=self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_params = dialog.get_settings()
            params.update(new_params)
            name_item.setData(QtCore.Qt.UserRole + 1, params)
            self._update_settings_tooltip(row, params, dtype)
            self._trigger_row_change(row)

    def onTableContextMenu(self, pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        add_action = menu.addAction("Add Row")
        delete_action = menu.addAction("Delete Selected Row(s)")
        menu.addSeparator()
        select_all_action = menu.addAction("Select All")
        clear_select_action = menu.addAction("Clear Selection")

        action = menu.exec_(self.distances_table.mapToGlobal(pos))
        if action == add_action:
            self.onAddRowTriggered()
        elif action == delete_action:
            selected_ranges = self.distances_table.selectedRanges()
            rows_to_delete = set()
            for r in selected_ranges:
                for row in range(r.topRow(), r.bottomRow() + 1):
                    if row < self.distances_table.rowCount() - 1:
                        rows_to_delete.add(row)

            if not rows_to_delete:
                return

            reply = QtWidgets.QMessageBox.question(
                self, "Remove Restraints?",
                f"Are you sure you want to remove the {len(rows_to_delete)} selected restraint(s)?",
                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.distances_table.blockSignals(True)
                try:
                    for row in sorted(rows_to_delete, reverse=True):
                        name_item = self.distances_table.item(row, 1)
                        if name_item:
                            dist_name = name_item.text().strip()
                            if dist_name:
                                self.distance_removed.emit(dist_name)
                        self.distances_table.removeRow(row)
                finally:
                    self.distances_table.blockSignals(False)
                    self.update_distance_lines()
        elif action == select_all_action:
            self.distances_table.selectAll()
        elif action == clear_select_action:
            self.distances_table.clearSelection()

    def onDeleteRowClicked(self) -> None:
        button = self.sender()
        if not button or self._block_table_signals > 0:
            return
        row = -1
        for r in range(self.distances_table.rowCount()):
            if self.distances_table.cellWidget(r, 7) is button:
                row = r
                break
        if row < 0 or row >= self.distances_table.rowCount() - 1:
            return
            
        reply = QtWidgets.QMessageBox.question(
            self, "Remove Restraint?",
            "Are you sure you want to remove this restraint?",
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.distances_table.blockSignals(True)
            try:
                name_item = self.distances_table.item(row, 1)
                if name_item:
                    dist_name = name_item.text().strip()
                    if dist_name:
                        self.distance_removed.emit(dist_name)
                self.distances_table.removeRow(row)
            finally:
                self.distances_table.blockSignals(False)
                self.update_distance_lines()

    def update_labels(self, label_names: list[str]) -> None:
        """Update label selection combo boxes with active label list."""
        self.label_names = label_names
        for row in range(self.distances_table.rowCount()):
            l1_cb = self.distances_table.cellWidget(row, 2)
            l2_cb = self.distances_table.cellWidget(row, 3)
            if isinstance(l1_cb, QtWidgets.QComboBox):
                l1_cb.blockSignals(True)
                curr = l1_cb.currentText()
                l1_cb.clear()
                l1_cb.addItems(label_names)
                l1_cb.setCurrentText(curr)
                l1_cb.blockSignals(False)
            if isinstance(l2_cb, QtWidgets.QComboBox):
                l2_cb.blockSignals(True)
                curr = l2_cb.currentText()
                l2_cb.clear()
                l2_cb.addItems(label_names)
                l2_cb.setCurrentText(curr)
                l2_cb.blockSignals(False)

    def update_score_sets(self, score_set_names: list[str]) -> None:
        """Update scoring group filter combo box."""
        self.score_set_names = score_set_names

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

        for row in range(self.distances_table.rowCount()):
            set_cb = self.distances_table.cellWidget(row, 6)
            if isinstance(set_cb, QtWidgets.QComboBox):
                set_cb.blockSignals(True)
                curr = set_cb.currentText()
                set_cb.clear()
                set_cb.addItem("")
                set_cb.addItems(score_set_names)
                set_cb.setCurrentText(curr)
                set_cb.blockSignals(False)

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
        self._block_table_signals += 1
        try:
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
                self._setup_row_widgets(row)

                # Set Show Checkbox checked state
                show_widget = self.distances_table.cellWidget(row, 0)
                if isinstance(show_widget, CenteredCheckBox):
                    show_widget.checkbox.setChecked(bool(d.get("visible", True)))

                # Set Name item text & parameters
                name_item = self.distances_table.item(row, 1)
                if name_item:
                    name_item.setText(key)
                    name_item.setData(QtCore.Qt.UserRole + 1, d)

                # Select Label 1 and Label 2
                l1 = d.get("position1_name", "")
                l2 = d.get("position2_name", "")
                l1_cb = self.distances_table.cellWidget(row, 2)
                l2_cb = self.distances_table.cellWidget(row, 3)
                if isinstance(l1_cb, QtWidgets.QComboBox):
                    l1_cb.setCurrentText(l1)
                if isinstance(l2_cb, QtWidgets.QComboBox):
                    l2_cb.setCurrentText(l2)

                # Select Type
                mapped_type = d.get("distance_type", "RDAMean")
                dtype = "dRDA"
                if mapped_type == "RDAMean":
                    dtype = "dRDA"
                elif mapped_type == "RDAMeanE":
                    dtype = "dRDAE"
                elif mapped_type == "Rmp":
                    dtype = "dRMP"
                elif mapped_type == "pRDA":
                    dtype = "pRDA"

                type_cb = self.distances_table.cellWidget(row, 4)
                if isinstance(type_cb, QtWidgets.QComboBox):
                    type_cb.setCurrentText(dtype)

                self._update_settings_tooltip(row, d, dtype)

                # Select Score Set
                belonging_sets = []
                for name, gs in score_sets.items():
                    if key in gs.get("distances", []):
                        belonging_sets.append(name)
                set_val = belonging_sets[0] if belonging_sets else ""
                set_cb = self.distances_table.cellWidget(row, 6)
                if isinstance(set_cb, QtWidgets.QComboBox):
                    set_cb.setCurrentText(set_val)

            # Add the empty row at the bottom for new distance entries
            row = self.distances_table.rowCount()
            self.distances_table.insertRow(row)
            self._setup_row_widgets(row)

        finally:
            self._populating = False
            self._block_table_signals -= 1

        # Redraw 3D line connections
        self.update_distance_lines()

    def update_distance_lines(self) -> None:
        """Update/redraw distance connection lines between mean positions in the 3D View."""
        if self._mol_view_3d is None or self._mol_view_3d.isHidden():
            return

        measurements = getattr(self._mol_view_3d, "_measurements", None) or {}
        new_measurements = dict(measurements)

        # Clear previous distance lines
        keys_to_remove = [k for k in new_measurements if k.startswith("dist_line_")]
        for k in keys_to_remove:
            new_measurements.pop(k, None)

        for row in range(self.distances_table.rowCount() - 1):
            show_widget = self.distances_table.cellWidget(row, 0)
            if not isinstance(show_widget, CenteredCheckBox) or not show_widget.checkbox.isChecked():
                continue

            l1_cb = self.distances_table.cellWidget(row, 2)
            l2_cb = self.distances_table.cellWidget(row, 3)
            if not isinstance(l1_cb, QtWidgets.QComboBox) or not isinstance(l2_cb, QtWidgets.QComboBox):
                continue

            l1 = l1_cb.currentText().strip()
            l2 = l2_cb.currentText().strip()
            if not l1 or not l2 or l1 == l2:
                continue

            cache = getattr(self._position_panel, "_av_cache", {})
            if l1 in cache and l2 in cache:
                _, xyz1, _, _ = cache[l1]
                _, xyz2, _, _ = cache[l2]

                name_item = self.distances_table.item(row, 1)
                name = name_item.text().strip() if name_item else f"{l1}_{l2}"

                # Calculate current spatial distance in Å
                val = float(np.linalg.norm(xyz1 - xyz2))

                # Use Label 1's color for the connection line
                _, _, _, col1 = cache[l1]
                line_color = [col1[0], col1[1], col1[2], 0.8]

                new_measurements[f"dist_line_{name}"] = {
                    "kind": "distance",
                    "positions": np.array([xyz1, xyz2]),
                    "color": line_color,
                    "label": f"{val:.1f} Å"
                }

        self._mol_view_3d._measurements = new_measurements
        self._mol_view_3d._update_view(fit_camera=False)
