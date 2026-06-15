"""RPC Monitoring widget — shows live RPC calls in a table."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets

import chisurf
from chisurf.gui.widgets.general import apply_compact_table_style

logger = logging.getLogger("chisurf.rpc")


class RPCMonitorWidget(QtWidgets.QWidget):
    """Displays all RPC calls dispatched through ``ServiceDispatcher`` instances.

    Connects to ``ServiceDispatcher.add_monitor()`` to receive live
    call data.  Each row shows method name, duration, status, params
    summary, and timestamp.
    """

    # ── signals ─────────────────────────────────────────────────────

    call_recorded = QtCore.Signal(str, dict, dict, float)  # method, params, result, elapsed

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._suspend_updates = False
        self._callback: Any = None
        self._max_rows = 5000

        self._build_ui()
        self._connect_signals()
        self._register_monitor()

    # ── public API ──────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all entries from the table."""
        self._table.setRowCount(0)

    @property
    def max_rows(self) -> int:
        return self._max_rows

    @max_rows.setter
    def max_rows(self, value: int) -> None:
        self._max_rows = max(1, value)

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()
        self._clear_btn = QtWidgets.QPushButton("Clear", self)
        self._clear_btn.setToolTip("Clear all entries")
        self._pause_btn = QtWidgets.QPushButton("Pause", self)
        self._pause_btn.setCheckable(True)
        self._pause_btn.setToolTip("Pause/resume updates")
        toolbar.addWidget(self._clear_btn)
        toolbar.addStretch()
        toolbar.addWidget(self._pause_btn)
        layout.addLayout(toolbar)

        # Table
        self._table = QtWidgets.QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(["Time", "Origin", "Target", "What", "Duration", "Status"])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        apply_compact_table_style(self._table)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.setColumnWidth(0, 75)
        self._table.setColumnWidth(1, 90)
        self._table.setColumnWidth(2, 90)
        self._table.setColumnWidth(3, 110)
        self._table.setColumnWidth(4, 65)
        self._table.setColumnWidth(5, 65)
        layout.addWidget(self._table)

        # Filter row below the table
        filter_row = QtWidgets.QHBoxLayout()
        self._filter_edit = QtWidgets.QLineEdit(self)
        self._filter_edit.setPlaceholderText("Filter method name…")
        filter_row.addWidget(self._filter_edit)
        layout.addLayout(filter_row)

    def _connect_signals(self) -> None:
        self._clear_btn.clicked.connect(self.clear)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        self._filter_edit.textChanged.connect(self._apply_filter)
        self.call_recorded.connect(self._on_call_recorded, QtCore.Qt.QueuedConnection)

    # ── monitor registration ────────────────────────────────────────

    def _register_monitor(self) -> None:
        from chisurf.server.dispatcher import ServiceDispatcher

        self._callback = self._monitor_callback
        ServiceDispatcher.add_monitor(self._callback)

    def _monitor_callback(self, method: str, params: dict, result: dict, elapsed: float) -> None:
        """Called from the dispatcher — emit Qt signal for thread safety."""
        self.call_recorded.emit(method, params, result, elapsed)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Unregister the monitor on widget close."""
        from chisurf.server.dispatcher import ServiceDispatcher

        if self._callback is not None:
            ServiceDispatcher.remove_monitor(self._callback)
        super().closeEvent(event)

    # ── slots ───────────────────────────────────────────────────────

    def _on_call_recorded(self, method: str, params: dict, result: dict, elapsed: float) -> None:
        if self._suspend_updates:
            return

        ok = result.get("ok", True)
        status = "OK" if ok else f"ERR({result.get('error_code', '?')})"

        # Format timestamp
        t_full = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        t_str = time.strftime("%H:%M:%S", time.localtime())

        origin, target, action = _split_method(method)
        params_summary = _params_summary(params)
        duration_str = f"{elapsed * 1000:.1f}ms"

        row = self._table.rowCount()
        self._table.insertRow(row)

        time_item = QtWidgets.QTableWidgetItem(t_str)
        time_item.setData(QtCore.Qt.UserRole, t_full)
        method_item = QtWidgets.QTableWidgetItem(method)
        method_item.setData(QtCore.Qt.UserRole, method)
        params_item = QtWidgets.QTableWidgetItem(params_summary)
        params_item.setData(QtCore.Qt.UserRole, params)
        params_item.setData(QtCore.Qt.UserRole + 1, result)
        target_item = QtWidgets.QTableWidgetItem(target)
        target_item.setData(QtCore.Qt.UserRole, method)
        action_item = QtWidgets.QTableWidgetItem(action)
        action_item.setData(QtCore.Qt.UserRole, params)
        action_item.setData(QtCore.Qt.UserRole + 1, result)

        self._table.setItem(row, 0, time_item)
        self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(origin))
        self._table.setItem(row, 2, target_item)
        self._table.setItem(row, 3, action_item)
        self._table.setItem(row, 4, QtWidgets.QTableWidgetItem(duration_str))
        self._table.setItem(row, 5, QtWidgets.QTableWidgetItem(status))

        # Color-code status column
        color = QtGui.QColor(50, 180, 50) if ok else QtGui.QColor(220, 50, 50)
        self._table.item(row, 5).setForeground(color)

        # Enforce max rows
        while self._table.rowCount() > self._max_rows:
            self._table.removeRow(0)

        self._table.scrollToBottom()

    def keyPressEvent(self, event):
        """Copy selected RPC rows with Ctrl+C."""
        if event.key() == QtCore.Qt.Key_C and event.modifiers() & QtCore.Qt.ControlModifier:
            self.copy_selected_items()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        """Show RPC-table context actions."""
        menu = QtWidgets.QMenu(self)
        select_all = menu.addAction("Select all")
        copy = menu.addAction("Copy")
        invert = menu.addAction("Invert selection")
        clear = menu.addAction("Clear")
        action = menu.exec(event.globalPos())
        if action == select_all:
            self._table.selectAll()
        elif action == copy:
            self.copy_selected_items()
        elif action == invert:
            self._invert_selection()
        elif action == clear:
            self.clear()

    def _invert_selection(self) -> None:
        selected_rows = {item.row() for item in self._table.selectedItems()}
        for row in range(self._table.rowCount()):
            self._table.selectRow(row) if row not in selected_rows else None
        for row in selected_rows:
            if 0 <= row < self._table.rowCount():
                self._table.selectionModel().select(
                    self._table.model().index(row, 0),
                    QtCore.QItemSelectionModel.Deselect | QtCore.QItemSelectionModel.Rows,
                )

    def copy_selected_items(self) -> None:
        """Copy selected RPC rows with full debugging payloads."""
        selected_items = self._table.selectedItems()
        if not selected_items:
            return

        lines = []
        for row in sorted({item.row() for item in selected_items}):
            values = []
            for column in range(self._table.columnCount()):
                item = self._table.item(row, column)
                if column == 0 and item is not None:
                    values.append(str(item.data(QtCore.Qt.UserRole) or item.text()))
                elif column == 2 and item is not None:
                    values.append(str(item.data(QtCore.Qt.UserRole) or item.text()))
                elif item is not None:
                    values.append(item.text())
                else:
                    values.append("")
            params_item = self._table.item(row, 3)
            params = params_item.data(QtCore.Qt.UserRole) if params_item is not None else {}
            result = params_item.data(QtCore.Qt.UserRole + 1) if params_item is not None else {}
            values.extend(
                [
                    json.dumps(params, indent=2, sort_keys=True, default=str),
                    json.dumps(result, indent=2, sort_keys=True, default=str),
                ]
            )
            lines.append("\t".join(values))

        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        logger.info(f"Copied {len(lines)} RPC entries to clipboard")

    def _on_pause_toggled(self, paused: bool) -> None:
        self._suspend_updates = paused
        self._pause_btn.setText("Resume" if paused else "Pause")

    def _apply_filter(self) -> None:
        filter_text = self._filter_edit.text().strip().lower()
        for row in range(self._table.rowCount()):
            row_text = " ".join(
                self._table.item(row, column).text()
                for column in range(self._table.columnCount())
                if self._table.item(row, column) is not None
            ).lower()
            match = not filter_text or filter_text in row_text
            self._table.setRowHidden(row, not match)


def _split_method(method: str) -> tuple[str, str, str]:
    parts = method.split(".")
    if len(parts) == 2:
        return parts[0], parts[1], parts[1]
    if len(parts) <= 1:
        return "", method, method
    origin_parts = parts[:-2]
    origin = ".".join(origin_parts[-2:])
    target = ".".join(parts[-2:])
    return origin, target, parts[-1]


def _params_summary(params: dict, max_len: int = 80) -> str:
    if not params:
        return ""
    keys = list(params.keys())
    if len(keys) <= 2:
        summary = ", ".join(f"{key}={_shorten(str(params[key]))}" for key in keys)
    else:
        summary = f"{len(keys)} keys: {', '.join(keys[:3])}…"
    return _shorten(summary, max_len)


def _shorten(val: str, max_len: int = 40) -> str:
    if len(val) > max_len:
        return val[: max_len - 3] + "..."
    return val
