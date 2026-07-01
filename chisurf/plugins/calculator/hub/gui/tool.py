"""Calculators hub: a two-panel launcher that embeds the selected calculator.

Left panel — a list of the registered calculators
(:func:`...core.registry.default_calculators`).
Right panel — the calculator chosen on the left, embedded inline. Each widget is
resolved from its dotted path and constructed lazily on first selection, so
opening the hub is cheap and heavy calculators are only built when picked.

This mirrors the Wizards hub (``chisurf.plugins.core.wizards``); the two share a
layout idiom but keep separate registries and state namespaces so each can evolve
independently.
"""

from __future__ import annotations

import importlib
import logging

from qtpy import QtCore, QtWidgets

from ..core.registry import CalculatorEntry, default_calculators

logger = logging.getLogger(__name__)

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:  # pragma: no cover

    def persist_plugin_state(_n):
        """No-op fallback when the state-persistence helper is unavailable."""
        return lambda c: c


def _resolve(path: str):
    """Import and return the object named by ``"pkg.module:Attr"`` / ``"pkg.module.Attr"``."""
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


@persist_plugin_state("calculators")
class CalculatorHub(QtWidgets.QWidget):
    """Two-panel calculator launcher (selector on the left, embedded tool on the right)."""

    name = "Calculators"

    def __init__(self, parent=None, entries: list[CalculatorEntry] | None = None):
        super().__init__(parent)
        self.setWindowTitle("ChiSurf Calculators")
        self.setMinimumSize(960, 640)
        self._entries = entries if entries is not None else default_calculators()
        self._built: dict[str, int] = {}  # entry id -> stack page index

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)

        # ── left: selector list (styled to match the boarding/wizards nav) ─
        self._list = QtWidgets.QListWidget()
        self._list.setMinimumWidth(210)
        self._list.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self._list.setSpacing(4)
        self._list.setStyleSheet(
            """
            QListWidget {
                border: none;
                border-right: 1px solid rgba(128, 128, 128, 0.3);
                padding-top: 5px;
            }
            QListWidget::item {
                height: 32px;
                padding-left: 10px;
                border-radius: 8px;
                margin: 2px 10px;
                font-weight: bold;
                font-size: 14px;
            }
            """
        )
        for entry in self._entries:
            text = f"{entry.icon}  {entry.label}" if entry.icon else entry.label
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, entry.id)
            item.setToolTip(entry.description)
            self._list.addItem(item)
        splitter.addWidget(self._list)

        # ── right: description + stacked embed ───────────────────────────
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(10, 8, 10, 8)
        right_layout.setSpacing(6)
        self._title = QtWidgets.QLabel("Calculators")
        self._title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._subtitle = QtWidgets.QLabel("Select a calculator on the left to get started.")
        self._subtitle.setWordWrap(True)
        self._subtitle.setStyleSheet("color: rgba(128,128,128,1);")
        right_layout.addWidget(self._title)
        right_layout.addWidget(self._subtitle)
        self._stack = QtWidgets.QStackedWidget()
        right_layout.addWidget(self._stack, 1)
        splitter.addWidget(right)
        splitter.setCollapsible(0, False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 760])

        # placeholder page (index 0)
        placeholder = QtWidgets.QLabel("No calculator selected.")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        placeholder.setStyleSheet("color: palette(mid);")
        self._stack.addWidget(placeholder)

        self._list.currentItemChanged.connect(self._on_select)
        if self._entries:
            self._list.setCurrentRow(0)

    def _entry_by_id(self, entry_id: str) -> CalculatorEntry | None:
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def _on_select(self, current, _previous=None) -> None:
        if current is None:
            self._stack.setCurrentIndex(0)
            return
        entry = self._entry_by_id(current.data(QtCore.Qt.UserRole))
        if entry is None:
            self._stack.setCurrentIndex(0)
            return
        self._title.setText(f"{entry.icon}  {entry.label}".strip() if entry.icon else entry.label)
        self._subtitle.setText(entry.description)
        index = self._built.get(entry.id)
        if index is None:
            index = self._build(entry)
            self._built[entry.id] = index
        self._stack.setCurrentIndex(index)

    def _build(self, entry: CalculatorEntry) -> int:
        """Construct *entry*'s widget, add it to the stack and return its index."""
        try:
            widget = _resolve(entry.widget)()
        except Exception as exc:
            logger.warning("calculators: could not build %r", entry.id, exc_info=True)
            widget = QtWidgets.QLabel(f"Could not load '{entry.label}':\n{exc}")
            widget.setAlignment(QtCore.Qt.AlignCenter)
            widget.setWordWrap(True)
        return self._stack.addWidget(widget)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = CalculatorHub()
    win.show()
    app.exec_()
