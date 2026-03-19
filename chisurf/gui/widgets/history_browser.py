from __future__ import annotations

import json

from chisurf import typing
from qtpy import QtCore, QtGui, QtWidgets
import chisurf


class HistoryBrowserWidget(QtWidgets.QWidget):
    cursorChanged = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self._history = None
        self._events: typing.List[typing.Dict[str, typing.Any]] = []
        self._cursor_event_id: typing.Optional[str] = None
        self._suspend_selection_signal = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        toolbar = QtWidgets.QHBoxLayout()
        self.filter_edit = QtWidgets.QLineEdit(self)
        self.filter_edit.setPlaceholderText("Filter history")
        self.clear_button = QtWidgets.QPushButton("Clear", self)
        self.clear_button.setToolTip("Clear in-memory history list")
        toolbar.addWidget(self.filter_edit)
        toolbar.addWidget(self.clear_button)
        layout.addLayout(toolbar)

        self.table = QtWidgets.QTreeWidget(self)
        self.table.setColumnCount(3)
        self.table.setHeaderLabels(["Time", "Action", "Summary"])
        self.table.setUniformRowHeights(True)
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(False)
        self.table.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.table.header().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

        self.details = QtWidgets.QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.details.setPlaceholderText("History event details")

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(self.details)
        self.splitter.setSizes([380, 140])
        layout.addWidget(self.splitter)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.table.itemSelectionChanged.connect(self._show_selected_details)
        self.table.itemClicked.connect(self._on_item_clicked)
        self.clear_button.clicked.connect(self._on_clear_clicked)

    @staticmethod
    def _is_incomplete_start_event(
            events: typing.List[typing.Dict[str, typing.Any]],
            index: int,
    ) -> bool:
        event = events[index]
        action = str(event.get("action_type", ""))
        source_uid = str(event.get("source_uid") or "")

        completion_map = {
            "fit.add.start": {"fit.add"},
            "fit.run.start": {"fit.run.finish", "fit.run.abort"},
            "app.reinitialize.start": {"app.reinitialize.finish"},
        }
        completions = completion_map.get(action)
        if not completions:
            return False

        for later in events[index + 1:]:
            later_action = str(later.get("action_type", ""))
            if later_action not in completions:
                continue
            later_source_uid = str(later.get("source_uid") or "")
            if source_uid:
                if later_source_uid == source_uid:
                    return False
            else:
                # No source uid: accept first completion of same action family.
                return False
        return True

    @staticmethod
    def _is_state_action(action_type: str) -> bool:
        return action_type in {
            "dataset.add",
            "dataset.remove",
            "dataset.group",
            "fit.add.start",
            "fit.add",
            "fit.close",
            "fit.run.start",
            "fit.run.finish",
            "fit.run.abort",
            "parameter.value",
            "parameter.fixed",
            "parameter.bounds.set",
            "parameter.bounds.on",
            "parameter.link",
            "parameter.unlink",
            "fit.range.set",
            "project.save",
            "project.load",
            "project.close",
            "app.reinitialize.start",
            "app.reinitialize.finish",
            # Model actions
            "model.add_component",
            "model.remove_component",
            "model.normalize_amplitudes",
            "model.absolute_amplitudes",
            "model.change_irf",
            "model.unload_irf",
            "model.update",
            "model.set_correction",
            "model.set_linearization",
            "model.unload_lintable",
            "model.unload_background_curve",
            "model.remove_local_fit",
            "model.clear_local_fits",
            "model.append_global_parameter",
            "model.append_fit",
        }

    @staticmethod
    def _log_info(message: str) -> None:
        try:
            chisurf.logging.info(f"HISTNAV: {message}")
        except Exception:
            pass

    def set_history(self, history_obj) -> None:
        if self._history is history_obj:
            return
        if self._history is not None:
            try:
                self._history.unsubscribe(self._on_event_recorded)
            except Exception:
                pass
        self._history = history_obj
        if self._history is not None:
            try:
                self._history.subscribe(self._on_event_recorded)
            except Exception:
                pass
        self.reload()

    def reload(self) -> None:
        self._events = []
        if self._history is not None:
            try:
                self._events = list(self._history.list_events())
            except Exception:
                self._events = []
        if self._events:
            if not self._cursor_event_id:
                self._cursor_event_id = str(self._events[-1].get("event_id") or "") or None
        else:
            self._cursor_event_id = None
        self._render()

    def _on_event_recorded(self, event: typing.Dict[str, typing.Any]) -> None:
        self._events.append(event)
        self._render()

    def _render(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        self._suspend_selection_signal = True
        self.table.blockSignals(True)
        self.table.clear()
        cursor_item = None
        for idx, event in enumerate(self._events):
            timestamp = str(event.get("timestamp", ""))
            action = str(event.get("action_type", ""))
            summary = str(event.get("summary", ""))
            row_text = f"{timestamp} {action} {summary}".lower()
            if needle and needle not in row_text:
                continue
            item = QtWidgets.QTreeWidgetItem([timestamp, action, summary])
            item.setData(0, QtCore.Qt.UserRole, event)
            if self._is_incomplete_start_event(self._events, idx):
                item.setForeground(2, QtGui.QBrush(QtGui.QColor("#d35400")))
                item.setToolTip(2, "Potential crash boundary: start event without matching finish")
            self.table.addTopLevelItem(item)
            event_id = str(event.get("event_id") or "")
            if self._cursor_event_id and event_id == self._cursor_event_id:
                cursor_item = item
                for col in range(3):
                    f = item.font(col)
                    f.setBold(True)
                    item.setFont(col, f)
        self.table.blockSignals(False)
        if cursor_item is not None:
            self.table.setCurrentItem(cursor_item)
            self.table.scrollToItem(cursor_item)
        elif self.table.topLevelItemCount() > 0:
            fallback = self.table.topLevelItem(self.table.topLevelItemCount() - 1)
            self.table.setCurrentItem(fallback)
            try:
                event = fallback.data(0, QtCore.Qt.UserRole)
                if isinstance(event, dict):
                    event_id = str(event.get("event_id") or "")
                    self._cursor_event_id = event_id or None
            except Exception:
                pass
        else:
            self.details.clear()
            self._cursor_event_id = None
        self._suspend_selection_signal = False

    def _apply_filter(self) -> None:
        self._render()

    def _show_selected_details(self) -> None:
        if self._suspend_selection_signal:
            return
        item = self.table.currentItem()
        if item is None:
            self.details.clear()
            return
        event = item.data(0, QtCore.Qt.UserRole)
        if not isinstance(event, dict):
            self.details.clear()
            return
        event_id = str(event.get("event_id") or "")
        self._cursor_event_id = event_id or None
        try:
            txt = json.dumps(event, indent=2, sort_keys=True)
        except Exception:
            txt = str(event)
        self.details.setPlainText(txt)
        self.cursorChanged.emit(event)

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        if item is None:
            return
        event = item.data(0, QtCore.Qt.UserRole)
        if not isinstance(event, dict):
            return
        event_id = str(event.get("event_id") or "")
        self._cursor_event_id = event_id or None
        self.cursorChanged.emit(event)

    def _on_clear_clicked(self) -> None:
        if self._history is not None:
            try:
                self._history.clear()
            except Exception:
                pass
        self._events = []
        self._cursor_event_id = None
        self._render()

    def current_event(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        if not self._events or not self._cursor_event_id:
            return None
        for event in self._events:
            if str(event.get("event_id") or "") == self._cursor_event_id:
                return event
        return None

    def _cursor_index(self) -> int:
        if not self._events or not self._cursor_event_id:
            return -1
        for i, event in enumerate(self._events):
            if str(event.get("event_id") or "") == self._cursor_event_id:
                return i
        return -1

    def cursor_index(self) -> int:
        return self._cursor_index()

    def events_upto_cursor(self) -> typing.List[typing.Dict[str, typing.Any]]:
        idx = self._cursor_index()
        if idx < 0:
            return []
        return list(self._events[:idx + 1])

    def all_events(self) -> typing.List[typing.Dict[str, typing.Any]]:
        return list(self._events)

    def can_undo(self) -> bool:
        idx = self._cursor_index()
        return idx > 0

    def can_redo(self) -> bool:
        idx = self._cursor_index()
        return 0 <= idx < (len(self._events) - 1)

    def move_cursor(self, delta: int, state_only: bool = False) -> typing.Optional[typing.Dict[str, typing.Any]]:
        if not self._events:
            return None
        idx = self._cursor_index()
        if idx < 0:
            idx = len(self._events) - 1
        new_idx = idx + int(delta)
        if new_idx < 0:
            new_idx = 0
        if new_idx >= len(self._events):
            new_idx = len(self._events) - 1

        skipped = 0
        if state_only:
            step = -1 if int(delta) < 0 else 1
            i = new_idx
            while 0 <= i < len(self._events):
                action = str(self._events[i].get("action_type", ""))
                if self._is_state_action(action):
                    new_idx = i
                    break
                i += step
                skipped += 1

        event = self._events[new_idx]
        self._cursor_event_id = str(event.get("event_id") or "") or None
        self._render()
        self._log_info(
            f"cursor {idx} -> {new_idx}; action={event.get('action_type','?')}; state_only={state_only}; skipped={skipped}"
        )
        self.cursorChanged.emit(event)
        return event

    def undo_step(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        return self.move_cursor(-1, state_only=True)

    def redo_step(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        return self.move_cursor(+1, state_only=True)

    def get_ui_state(self) -> typing.Dict[str, typing.Any]:
        state: typing.Dict[str, typing.Any] = {
            "filter_text": self.filter_edit.text(),
            "splitter_sizes": [int(v) for v in self.splitter.sizes()],
        }
        item = self.table.currentItem()
        if item is not None:
            event = item.data(0, QtCore.Qt.UserRole)
            if isinstance(event, dict):
                event_id = event.get("event_id")
                if event_id:
                    state["selected_event_id"] = str(event_id)
        return state

    def set_ui_state(self, state: typing.Dict[str, typing.Any]) -> None:
        if not isinstance(state, dict):
            return
        try:
            self.filter_edit.setText(str(state.get("filter_text", "")))
        except Exception:
            pass
        sizes = state.get("splitter_sizes")
        if isinstance(sizes, (list, tuple)) and len(sizes) == 2:
            try:
                self.splitter.setSizes([int(sizes[0]), int(sizes[1])])
            except Exception:
                pass
        selected_event_id = state.get("selected_event_id")
        if selected_event_id:
            for i in range(self.table.topLevelItemCount()):
                item = self.table.topLevelItem(i)
                event = item.data(0, QtCore.Qt.UserRole)
                if isinstance(event, dict) and str(event.get("event_id", "")) == str(selected_event_id):
                    self.table.setCurrentItem(item)
                    break
