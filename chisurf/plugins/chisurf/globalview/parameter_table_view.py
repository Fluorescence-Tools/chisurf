from __future__ import annotations

from typing import List, Optional

from qtpy import QtWidgets, QtCore, QtGui

import chisurf
from chisurf import logging
from chisurf.core.fitting.fit import Fit, FitGroup
from chisurf.core.fitting.parameter import FittingParameter
from chisurf.gui.plots.table_plot import BooleanToggleDelegate
from chisurf.core.base import Base, find_by_uuid

from chisurf.plugins.chisurf.globalview.parameter_table_model import (
    ParameterTableModel,
    COL_FIT,
    COL_LOCAL_FIT,
    COL_PARAM,
    COL_VALUE,
    COL_FIXED,
    COL_BOUNDS_LO,
    COL_BOUNDS_HI,
    COL_BOUNDS_ON,
    COL_ERROR,
    COL_LINKED,
    COL_DESCRIPTION,
)


NUMERIC_COLS = {COL_VALUE, COL_BOUNDS_LO, COL_BOUNDS_HI, COL_ERROR}
FROZEN_COLS = 3  # Fit, Local fit, Parameter


class ParameterFilterProxy(QtCore.QSortFilterProxyModel):
    """Allows filtering by fit name (fit:…) and parameter name (name:…)."""

    def __init__(self, parent=None):
        """Initialize the widget."""
        super().__init__(parent)
        self._filter_fit = ""
        self._filter_param = ""
        self._filter_mode = "all"  # all | free | fixed | linked
# TODO: needs docstring

    def setFilterString(self, text: str):
        text = text.strip()
        self._filter_fit = ""
        self._filter_param = ""
        rest = text
        if text.startswith("fit:"):
            parts = text.split(None, 1)
            if len(parts) == 2:
                first, rest = parts
                self._filter_fit = first[4:].strip().lower()
            else:
                self._filter_fit = text[4:].strip().lower()
                rest = ""
        self._filter_param = rest.lower()
# TODO: needs docstring
        self.invalidateFilter()

    def setFilterMode(self, mode: str):
# TODO: needs docstring
        self._filter_mode = mode
        self.invalidateFilter()

    def filterAcceptsRow(
        self, source_row: int, source_parent: QtCore.QModelIndex
    ) -> bool:
        src = self.sourceModel()
        if not isinstance(src, ParameterTableModel):
            return True

        idx_fit = src.index(source_row, COL_FIT, source_parent)
        idx_param = src.index(source_row, COL_PARAM, source_parent)
        idx_fixed = src.index(source_row, COL_FIXED, source_parent)
        idx_linked = src.index(source_row, COL_LINKED, source_parent)

        fit_name = (src.data(idx_fit) or "").lower()
        param_name = (src.data(idx_param) or "").lower()

        if self._filter_fit and self._filter_fit not in fit_name:
            return False
        if self._filter_param and self._filter_param not in param_name:
            return False

        mode = self._filter_mode
        if mode == "free":
            fixed_val = src.data(idx_fixed)
            is_fixed = str(fixed_val).lower() in ("true", "1", "yes")
            if is_fixed:
                return False
            linked_val = src.data(idx_linked) or ""
            if linked_val:
                return False
        elif mode == "fixed":
            fixed_val = src.data(idx_fixed)
            is_fixed = str(fixed_val).lower() in ("true", "1", "yes")
            if not is_fixed:
                return False
        elif mode == "linked":
            linked_val = src.data(idx_linked) or ""
            if not linked_val:
                return False

        return True

    def lessThan(self, left: QtCore.QModelIndex, right: QtCore.QModelIndex) -> bool:
        """Numeric sort for value, bounds, and error columns."""
        if left.column() in NUMERIC_COLS and left.column() == right.column():
            try:
                lv = float(self.sourceModel().data(left))
                rv = float(self.sourceModel().data(right))
                return lv < rv
            except (ValueError, TypeError):
                pass
        return super().lessThan(left, right)

# TODO: needs docstring

class ParameterTableWidget(QtWidgets.QTableView):
    """QTableView subclass with Excel-like keyboard navigation and copy."""

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        key = event.key()

        if key == QtCore.Qt.Key_Tab:
            self._move_current(forward=True, down=False)
            return
        if key == QtCore.Qt.Key_Backtab:
            self._move_current(forward=False, down=False)
            return
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            modifiers = event.modifiers()
            if modifiers & QtCore.Qt.ShiftModifier:
                self._move_current(forward=False, down=True)
            else:
                self._move_current(forward=True, down=True)
            return

        if event.matches(QtGui.QKeySequence.Copy):
            self._copy_selection()
            return
        if event.matches(QtGui.QKeySequence.Paste):
            self._paste_to_selection()
            return

        super().keyPressEvent(event)

    # ── navigation ─────────────────────────────────────────────

    def _move_current(self, forward: bool = True, down: bool = True):
        """Move the current index like Excel: Enter=down, Tab=right."""
        idx = self.currentIndex()
        if not idx.isValid():
            if forward:
                idx = self.model().index(0, 0)
            else:
                rows = self.model().rowCount()
                cols = self.model().columnCount()
                idx = self.model().index(rows - 1, cols - 1)
            self.setCurrentIndex(idx)
            return

        model = self.model()
        rows = model.rowCount()
        cols = model.columnCount()

        if down:
            # Enter moves down; wraps to next column at bottom
            next_row = idx.row() + 1
            next_col = idx.column()
            if next_row >= rows:
                if forward:
                    next_row = 0
                    next_col = idx.column() + 1
                else:
                    next_row = rows - 1
                    next_col = idx.column() - 1
        else:
            # Tab moves horizontally; wraps to next row at end
            if forward:
                next_col = idx.column() + 1
                next_row = idx.row()
                if next_col >= cols:
                    next_col = 0
                    next_row = idx.row() + 1
            else:
                next_col = idx.column() - 1
                next_row = idx.row()
                if next_col < 0:
                    next_col = cols - 1
                    next_row = idx.row() - 1

        if 0 <= next_row < rows and 0 <= next_col < cols:
            next_idx = model.index(next_row, next_col)
            self.setCurrentIndex(next_idx)
            # Enter editing mode for editable cells
# TODO: needs docstring
            if next_idx.flags() & QtCore.Qt.ItemIsEditable:
                self.edit(next_idx)

    # ── copy / paste ───────────────────────────────────────────

    def _copy_selection(self):
        sel = self.selectionModel()
        indexes = sorted(sel.selectedIndexes())
        if not indexes:
            return

        rows: dict[int, dict[int, str]] = {}
        for idx in indexes:
            r = idx.row()
            c = idx.column()
            if r not in rows:
                rows[r] = {}
            rows[r][c] = self.model().data(idx) or ""

        lines = []
        for r in sorted(rows):
# TODO: needs docstring
            row_data = rows[r]
            line = "\t".join(str(row_data.get(c, "")) for c in sorted(row_data))
            lines.append(line)

        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def _paste_to_selection(self):
        text = QtWidgets.QApplication.clipboard().text()
        if not text:
            return
        lines = [line for line in text.split("\n") if line]
        if not lines:
            return

        idx = self.currentIndex()
        if not idx.isValid():
            return

        model = self.model()
        for ri, line in enumerate(lines):
            parts = line.split("\t")
            for ci, val in enumerate(parts):
                target_row = idx.row() + ri
                target_col = idx.column() + ci
                if target_row >= model.rowCount() or target_col >= model.columnCount():
                    continue
                target = model.index(target_row, target_col)
                if target.flags() & QtCore.Qt.ItemIsEditable:
                    model.setData(target, val, QtCore.Qt.EditRole)


class ParameterTableView(QtWidgets.QWidget):
    """Widget combining a toolbar and a table view of all fit parameters."""

    paramChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._parent_wizard = parent

        self._model = ParameterTableModel(self)
        self._filter_proxy = ParameterFilterProxy(self)
# TODO: needs docstring
        self._filter_proxy.setSourceModel(self._model)
        self._filter_proxy.setDynamicSortFilter(True)

        self._build_ui()
        self._connect_signals()

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # toolbar
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(4)

        self._filter_edit = QtWidgets.QLineEdit()
        self._filter_edit.setPlaceholderText("filter: fit:name param:name…")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.setMaximumWidth(250)
        toolbar.addWidget(self._filter_edit)

        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItems(["all", "free", "fixed", "linked"])
        self._mode_combo.setToolTip("Show only parameters matching status")
        toolbar.addWidget(self._mode_combo)

        toolbar.addStretch()

        self._btn_link = QtWidgets.QToolButton()
        self._btn_link.setText("Link")
        self._btn_link.setToolTip("Link selected parameters (first = master)")
        toolbar.addWidget(self._btn_link)

        self._btn_unlink = QtWidgets.QToolButton()
        self._btn_unlink.setText("Unlink")
        self._btn_unlink.setToolTip("Unlink selected parameters")
        toolbar.addWidget(self._btn_unlink)

        self._btn_fix = QtWidgets.QToolButton()
        self._btn_fix.setText("Fix")
        self._btn_fix.setToolTip("Fix selected parameters")
        toolbar.addWidget(self._btn_fix)

        self._btn_unfix = QtWidgets.QToolButton()
        self._btn_unfix.setText("Unfix")
        self._btn_unfix.setToolTip("Unfix selected parameters")
        toolbar.addWidget(self._btn_unfix)

        self._btn_set_value = QtWidgets.QToolButton()
        self._btn_set_value.setText("Set value…")
        self._btn_set_value.setToolTip("Set value for all selected parameters")
        toolbar.addWidget(self._btn_set_value)

        self._btn_link_by_name = QtWidgets.QToolButton()
        self._btn_link_by_name.setText("Link by name…")
        self._btn_link_by_name.setToolTip("Link parameters by name across fits")
        toolbar.addWidget(self._btn_link_by_name)

        self._btn_refresh = QtWidgets.QToolButton()
        self._btn_refresh.setText("Refresh")
        self._btn_refresh.setToolTip("Reload all parameters from fits")
        toolbar.addWidget(self._btn_refresh)

        self._btn_find_uid = QtWidgets.QToolButton()
        self._btn_find_uid.setText("Find by UUID…")
        self._btn_find_uid.setToolTip("Look up a parameter or fit by its unique identifier")
        toolbar.addWidget(self._btn_find_uid)

        layout.addLayout(toolbar)

        # table
        self._table = ParameterTableWidget()
        self._table.setModel(self._filter_proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(True)
        self._table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        # Edit triggers: double-click or F2 or keyboard entry
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.AnyKeyPressed
        )

        # Horizontal header
        hdr = self._table.horizontalHeader()
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        hdr.setHighlightSections(False)
        hdr.setSectionsClickable(True)

        # Vertical header: show row numbers
        vhdr = self._table.verticalHeader()
        vhdr.setDefaultSectionSize(24)
        vhdr.setHighlightSections(False)

        # Font: monospace for data cells
        font = QtGui.QFont("SF Mono", 11)
        font.setStyleHint(QtGui.QFont.Monospace)
        self._table.setFont(font)
        hdr.setFont(QtGui.QFont("SF Mono", 10))
        hdr.setDefaultSectionSize(24)

        # Frozen columns: first 3 columns always visible
        if hasattr(self._table, "setColumnHidden"):
            pass
        self._table.setColumnWidth(COL_FIT, 160)
        self._table.setColumnWidth(COL_LOCAL_FIT, 120)
        self._table.setColumnWidth(COL_PARAM, 140)
        self._table.setColumnWidth(COL_VALUE, 100)
        self._table.setColumnWidth(COL_FIXED, 60)
        self._table.setColumnWidth(COL_BOUNDS_LO, 80)
        self._table.setColumnWidth(COL_BOUNDS_HI, 80)
        self._table.setColumnWidth(COL_BOUNDS_ON, 70)
        self._table.setColumnWidth(COL_ERROR, 80)
        self._table.setColumnWidth(COL_LINKED, 160)
        self._table.setColumnWidth(COL_DESCRIPTION, 200)

        # delegates
        self._table.setItemDelegateForColumn(
            COL_FIXED, BooleanToggleDelegate(self._table)
# TODO: needs docstring
        )
        self._table.setItemDelegateForColumn(
            COL_BOUNDS_ON, BooleanToggleDelegate(self._table)
        )

        layout.addWidget(self._table)

    # ── signal wiring ─────────────────────────────────────────────────

    def _connect_signals(self):
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._table.customContextMenuRequested.connect(self._show_context_menu)

        self._btn_link.clicked.connect(self._on_link)
        self._btn_unlink.clicked.connect(self._on_unlink)
# TODO: needs docstring
        self._btn_fix.clicked.connect(self._on_fix)
        self._btn_unfix.clicked.connect(self._on_unfix)
# TODO: needs docstring
        self._btn_set_value.clicked.connect(self._on_set_value)
        self._btn_link_by_name.clicked.connect(self._on_link_by_name)
        self._btn_refresh.clicked.connect(self._on_refresh)
        self._btn_find_uid.clicked.connect(self._on_find_by_uuid)
# TODO: needs docstring

    # ── public API ────────────────────────────────────────────────────

    @property
    def model(self) -> ParameterTableModel:
        return self._model

    def refresh(self):
        # TODO: needs docstring
        self._model.refresh()
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)

    def refresh_from_fits(self, fit_list: List[Fit]):
        """Handle the mode_changed event (internal)."""
        self._model.refresh(fit_list=fit_list)
        self._table.resizeColumnsToContents()

    # ── filter ────────────────────────────────────────────────────────

    def _on_filter_changed(self, text: str):
        self._filter_proxy.setFilterString(text)
# TODO: needs docstring

    def _on_mode_changed(self, mode: str):
        self._filter_proxy.setFilterMode(mode)
# TODO: needs docstring

    # ── selection helpers ─────────────────────────────────────────────

    def _selected_source_rows(self) -> List[int]:
        """Return list of source-model row indices for selected rows."""
        rows = set()
        for idx in self._table.selectionModel().selectedRows():
            src = self._filter_proxy.mapToSource(idx)
            if src.isValid():
                rows.add(src.row())
# TODO: needs docstring
        return sorted(rows)

    def _selected_params(self) -> List[FittingParameter]:
        rows = self._selected_source_rows()
        return [self._model.param_at_row(r) for r in rows if self._model.param_at_row(r) is not None]

    def _selected_fits(self) -> List[Fit]:
        rows = self._selected_source_rows()
        result = []
        for r in rows:
            f = self._model.fit_at_row(r)
            if f is not None:
                result.append(f)
        return result

    # ── context menu ──────────────────────────────────────────────────

    def _show_context_menu(self, pos: QtCore.QPoint):
        menu = QtWidgets.QMenu(self)

        copy_action = menu.addAction("Copy", self._table._copy_selection)
        copy_action.setShortcut(QtGui.QKeySequence.Copy)
        paste_action = menu.addAction("Paste", self._table._paste_to_selection)
        paste_action.setShortcut(QtGui.QKeySequence.Paste)

        menu.addSeparator()
        menu.addAction("Link selected", self._on_link)
        menu.addAction("Unlink selected", self._on_unlink)
        menu.addSeparator()
        menu.addAction("Fix selected", self._on_fix)
        menu.addAction("Unfix selected", self._on_unfix)
        menu.addSeparator()
        menu.addAction("Set value…", self._on_set_value)
        menu.addSeparator()
        menu.addAction("Link by name across fits…", self._on_link_by_name)
        menu.addSeparator()
        menu.addAction("Refresh", self._on_refresh)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    # ── bulk actions ──────────────────────────────────────────────────

    def _on_link(self):
        # TODO: needs docstring
        params = self._selected_params()
        if len(params) < 2:
            logging.log(0, "Select at least two parameters to link")
            return
        master = params[0]
        for follower in params[1:]:
            """Handle the fix event (internal)."""
            if follower is master:
                continue
            try:
                follower.link = master
            except Exception as e:
                """Handle the unfix event (internal)."""
                logging.log(0, f"Link failed: {e}")
        self._notify_changed()

    def _on_unlink(self):
        for p in self._selected_params():
            try:
                p.link = None
            except Exception:
                """Handle the set_value event (internal)."""
                pass
        self._notify_changed()

    def _on_fix(self):
        for p in self._selected_params():
            try:
                p.fixed = True
            except Exception:
                pass
        self._finalize_all()
        self._notify_changed()

    def _on_unfix(self):
        for p in self._selected_params():
            try:
                p.fixed = False
            except Exception:
                pass
        self._finalize_all()
        self._notify_changed()

    def _on_set_value(self):
        params = self._selected_params()
        if not params:
            return
        value, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Set value",
            f"New value for {len(params)} parameter(s):",
            params[0].value,
            -1e12,
            1e12,
            6,
        )
        if not ok:
            return
        for p in params:
            try:
                p.value = value
            except Exception:
                pass
        self._finalize_all()
        self._notify_changed()

    def _on_link_by_name(self):
        """Link all parameters with a given name across fits."""
        # Collect available parameter names from the model
        names: set = set()
        for i in range(self._model.rowCount()):
            idx = self._model.index(i, COL_PARAM)
            n = self._model.data(idx) or ""
            if n:
                names.add(n)
        if not names:
            return

        name, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Link by name",
            "Parameter name to link across fits:",
            sorted(names),
            editable=False,
        )
        if not ok or not name:
            return

        # Find all params with this name
        matches: list = []
        for i in range(self._model.rowCount()):
            idx = self._model.index(i, COL_PARAM)
            if (self._model.data(idx) or "") == name:
                params_idx = self._model.index(i, COL_VALUE)
                matches.append(
                    (self._model.fit_at_row(i), self._model.param_at_row(i))
                )

        if len(matches) < 2:
            logging.log(0, "Need at least two parameters with that name to link")
            return

        # Use the first as master
        master_fit, master_param = matches[0]
        for fit_obj, param in matches[1:]:
            if param is master_param:
                continue
            try:
                target_name = getattr(param, "name", "")
                master_name = getattr(master_param, "name", "")
                if target_name and master_name:
                    if isinstance(fit_obj, FitGroup):
                        pass
                    param.link = master_param
            except Exception as e:
                logging.log(0, f"Link by name failed: {e}")
        self._notify_changed()

    # ── internal helpers ──────────────────────────────────────────────

    def _on_refresh(self):
        """Refresh table data and notify wizard to redraw graph."""
        self._model.refresh()
        self._table.resizeColumnsToContents()
        self.paramChanged.emit()

    def _on_find_by_uuid(self):
        """Open a dialog to look up an object by UUID and select its row."""
        uid, ok = QtWidgets.QInputDialog.getText(
            self,
            "Find by UUID",
            "Enter a unique identifier (UUID):",
        )
        if not ok or not uid:
            return
        uid = uid.strip()
        if not uid:
            return

        obj = find_by_uuid(uid)
        if obj is None:
            QtWidgets.QMessageBox.information(
                self,
                "Not found",
                f"No live object found with UID:\n{uid}",
            )
            return

        # If it's a Parameter, try to select its row in the table
        from chisurf.core.parameter import Parameter
        if isinstance(obj, Parameter):
            uids = {str(obj.unique_identifier)}
            for src_row in range(self._model.rowCount()):
                p = self._model.param_at_row(src_row)
                if p is not None and str(p.unique_identifier) in uids:
                    proxy_idx = self._filter_proxy.mapFromSource(
                        self._model.index(src_row, 0)
                    )
                    if proxy_idx.isValid():
                        self._table.selectRow(proxy_idx.row())
                        self._table.scrollTo(proxy_idx)
                        break
            else:
                r = self._model.rowCount()
# TODO: needs docstring
                QtWidgets.QMessageBox.information(
                    self,
                    "Found but not in table",
                    f"Parameter '{obj.name}' was found but is not in the current table view.\n"
                    f"Try changing the filter or refreshing.",
                )

        elif isinstance(obj, (Fit, FitGroup)):
            QtWidgets.QMessageBox.information(
# TODO: needs docstring
                self,
                    "Found",
                f"Found: {type(obj).__name__} '{getattr(obj, 'name', '')}'\n"
                f"UID: {uid}",
            )

        else:
            QtWidgets.QMessageBox.information(
                self,
                "Found",
                f"Found: {type(obj).__name__} '{getattr(obj, 'name', '')}'\n"
                f"UID: {uid}",
            )

    def _finalize_all(self):
        fits = self._model.get_all_fits()
        for f in fits:
            try:
                m = getattr(f, "model", None)
                if m is not None and hasattr(m, "finalize"):
                    m.finalize()
            except Exception:
                pass

    def _notify_changed(self):
        self._model.refresh()
        self.paramChanged.emit()
