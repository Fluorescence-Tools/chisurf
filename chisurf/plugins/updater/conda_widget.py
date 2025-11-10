"""
Conda Manager dialog for ChiSurf.

Provides a simple UI to manage packages, environments, and channels
using CondaManager.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Set
import logging

# Logger for the package manager dialog
logger = logging.getLogger("chisurf.packagemanager")

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QTabWidget, QWidget, QListWidget, QListWidgetItem, QTextEdit, QFileDialog,
    QMessageBox, QInputDialog, QTableWidget, QTableWidgetItem, QToolButton,
    QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .conda_manager import CondaManager


class CondaWorker(QThread):
    finished = pyqtSignal(bool, object, str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            ok = False
            payload: object = None
            msg = ""
            if isinstance(res, tuple):
                # Normalize results from CondaManager methods
                if len(res) == 3:
                    ok, payload, err = res
                    msg = err or ""
                elif len(res) == 2:
                    ok, text = res
                    payload = text
                else:
                    ok = False
                    msg = "Unexpected result tuple length"
            else:
                payload = res
                ok = True
            self.finished.emit(bool(ok), payload, msg)
        except Exception as e:
            self.finished.emit(False, None, str(e))


class CondaManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ChiSurf Package Manager")
        try:
            self.resize(640, 480)
        except Exception:
            pass
        self.manager = CondaManager()
        # Keep worker references alive
        self._workers: List[CondaWorker] = []
        # Dataset for current packages (search or installed)
        self._all_packages: List[Dict[str, Any]] = []
        self._filtered_packages: List[Dict[str, Any]] = []
        # Track names of currently installed packages to mark checkboxes
        self._installed_names: Set[str] = set()
        # Persist user selection independent of visibility/filtering
        self._checked_names: Set[str] = set()
        # Guard to suppress itemChanged recursion during programmatic updates
        self._suppress_item_changed: bool = False
        # Avoid starting background work before the dialog is shown
        self._did_initial_refresh = False
        # Guard to avoid handling worker results after the dialog starts closing
        self._closing = False
        # Pending summary for apply sequence (shown after refresh)
        self._pending_apply_summary_lines: List[str] = []

        self._build_ui()
        self._connect()
        # Defer initial refreshes to showEvent to avoid races on construction
        try:
            self.refresh_repo_info()
        except Exception:
            pass

    def _build_ui(self):
        layout = QVBoxLayout()
        try:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        except Exception:
            pass
        self.tabs = QTabWidget()
        # Make log about 1/4 of height -> tabs:log = 3:1
        layout.addWidget(self.tabs, 3)

        # Packages tab
        self.pkgs_tab = QWidget()
        pk_lay = QVBoxLayout()
        try:
            pk_lay.setContentsMargins(0, 0, 0, 0)
            pk_lay.setSpacing(0)
        except Exception:
            pass
        srch_lay = QHBoxLayout()
        try:
            srch_lay.setContentsMargins(0, 0, 0, 0)
            srch_lay.setSpacing(0)
        except Exception:
            pass
        srch_lay.addWidget(QLabel("Search:"))
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("Type to filter current list; press Search to query remote")
        srch_lay.addWidget(self.ed_search)
        # Use toolbuttons for actions
        self.btn_search = QToolButton()
        self.btn_search.setText("Search")
        self.btn_list_installed = QToolButton()
        self.btn_list_installed.setText("List Installed")
        srch_lay.addWidget(self.btn_search)
        srch_lay.addWidget(self.btn_list_installed)
        pk_lay.addLayout(srch_lay)

        # Packages table: [check, name, version, build, source]
        from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
        self.tbl_pkgs = QTableWidget(0, 5)
        self.tbl_pkgs.setHorizontalHeaderLabels(["✔", "Name", "Version", "Build", "Source"])
        try:
            self.tbl_pkgs.verticalHeader().setVisible(False)
            # Use default background with NO alternating colors
            self.tbl_pkgs.setAlternatingRowColors(False)
            self.tbl_pkgs.setStyleSheet("")
            self.tbl_pkgs.setSelectionBehavior(self.tbl_pkgs.SelectRows)
            self.tbl_pkgs.setEditTriggers(self.tbl_pkgs.NoEditTriggers)
            self.tbl_pkgs.horizontalHeader().setStretchLastSection(True)
            self.tbl_pkgs.horizontalHeader().setDefaultSectionSize(140)
            self.tbl_pkgs.setColumnWidth(0, 32)
        except Exception:
            pass
        pk_lay.addWidget(self.tbl_pkgs)

        # Single row for selection and actions (all toolbuttons)
        actions_lay = QHBoxLayout()
        try:
            actions_lay.setContentsMargins(0, 0, 0, 0)
            actions_lay.setSpacing(0)
        except Exception:
            pass
        # Toggle Select/Deselect affects only visible rows
        self.btn_toggle_select = QToolButton()
        self.btn_toggle_select.setText("Select Visible")
        actions_lay.addWidget(self.btn_toggle_select)
        # Separator (expanding spacer) between select toggle and actions
        actions_lay.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        # Unified apply button to install/remove according to checkbox states
        self.btn_apply = QToolButton()
        self.btn_apply.setText("Apply")
        self.btn_update = QToolButton()
        self.btn_update.setText("Update")
        self.btn_update_all = QToolButton()
        self.btn_update_all.setText("Update All")
        actions_lay.addWidget(self.btn_apply)
        actions_lay.addWidget(self.btn_update)
        actions_lay.addWidget(self.btn_update_all)
        pk_lay.addLayout(actions_lay)
        self.pkgs_tab.setLayout(pk_lay)
        self.tabs.addTab(self.pkgs_tab, "Packages")

        # Environments tab
        self.env_tab = QWidget()
        env_lay = QVBoxLayout()
        try:
            env_lay.setContentsMargins(0, 0, 0, 0)
            env_lay.setSpacing(0)
        except Exception:
            pass
        self.list_envs = QListWidget()
        env_lay.addWidget(self.list_envs)
        env_buttons = QHBoxLayout()
        try:
            env_buttons.setContentsMargins(0, 0, 0, 0)
            env_buttons.setSpacing(0)
        except Exception:
            pass
        self.btn_env_refresh = QToolButton()
        self.btn_env_refresh.setText("Refresh")
        self.ed_env_name = QLineEdit()
        self.ed_env_name.setPlaceholderText("New environment name (or leave empty and choose folder)")
        self.btn_env_create = QToolButton()
        self.btn_env_create.setText("Create")
        self.btn_env_remove = QToolButton()
        self.btn_env_remove.setText("Remove")
        self.btn_env_clone = QToolButton()
        self.btn_env_clone.setText("Clone")
        self.btn_env_export = QToolButton()
        self.btn_env_export.setText("Export YAML")
        self.btn_env_import = QToolButton()
        self.btn_env_import.setText("Import YAML…")
        env_buttons.addWidget(self.btn_env_refresh)
        env_buttons.addStretch()
        env_buttons.addWidget(self.ed_env_name)
        env_buttons.addWidget(self.btn_env_create)
        env_buttons.addWidget(self.btn_env_remove)
        env_buttons.addWidget(self.btn_env_clone)
        env_buttons.addWidget(self.btn_env_export)
        env_buttons.addWidget(self.btn_env_import)
        env_lay.addLayout(env_buttons)
        self.env_tab.setLayout(env_lay)
        self.tabs.addTab(self.env_tab, "Environments")

        # Channels tab
        self.chan_tab = QWidget()
        ch_lay = QVBoxLayout()
        try:
            ch_lay.setContentsMargins(0, 0, 0, 0)
            ch_lay.setSpacing(0)
        except Exception:
            pass
        self.list_channels = QListWidget()
        ch_lay.addWidget(self.list_channels)
        ch_btns = QHBoxLayout()
        try:
            ch_btns.setContentsMargins(0, 0, 0, 0)
            ch_btns.setSpacing(0)
        except Exception:
            pass
        self.ed_channel = QLineEdit()
        self.ed_channel.setPlaceholderText("Add channel, e.g. conda-forge")
        self.btn_channel_add = QToolButton()
        self.btn_channel_add.setText("Add")
        self.btn_channel_remove = QToolButton()
        self.btn_channel_remove.setText("Remove Selected")
        ch_btns.addWidget(self.ed_channel)
        ch_btns.addWidget(self.btn_channel_add)
        ch_btns.addWidget(self.btn_channel_remove)
        ch_lay.addLayout(ch_btns)
        self.chan_tab.setLayout(ch_lay)
        self.tabs.addTab(self.chan_tab, "Channels")

        # Hide/disable Environments and Channels tabs for now to avoid refresh crashes
        try:
            # Remove Environments (index 1) and Channels (which becomes index 1 after removal)
            self.tabs.removeTab(2)  # Channels at index 2 initially
            self.tabs.removeTab(1)  # Environments at index 1
        except Exception:
            try:
                # Fallback: disable if removeTab not applicable
                self.tabs.setTabEnabled(1, False)
                self.tabs.setTabEnabled(2, False)
            except Exception:
                pass

        # Log output
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.txt_log, 1)
        # Repo info line below output
        self.lbl_repo = QLabel("")
        try:
            self.lbl_repo.setTextInteractionFlags(Qt.TextSelectableByMouse)
        except Exception:
            pass
        layout.addWidget(self.lbl_repo)

        self.setLayout(layout)

    def _connect(self):
        self.btn_search.clicked.connect(self.run_search)
        self.btn_list_installed.clicked.connect(self.run_list_installed)
        # Unified apply and update actions
        self.btn_apply.clicked.connect(self.run_apply_changes)
        self.btn_update.clicked.connect(self.run_update)
        self.btn_update_all.clicked.connect(self.run_update_all)
        # Selection toggle
        self.btn_toggle_select.clicked.connect(self.toggle_select_visible)

        self.btn_env_refresh.clicked.connect(self.refresh_envs)
        self.btn_env_create.clicked.connect(self.create_env)
        self.btn_env_remove.clicked.connect(self.remove_env)
        self.btn_env_export.clicked.connect(self.export_env)
        self.btn_env_import.clicked.connect(self.import_env)
        self.btn_env_clone.clicked.connect(self.clone_env)

        self.btn_channel_add.clicked.connect(self.add_channel)
        self.btn_channel_remove.clicked.connect(self.remove_channel)

        # Live filter: typing filters current table, also update toggle label
        try:
            self.ed_search.textChanged.connect(self._apply_pkg_filter)
            # Pressing Enter in the search bar triggers an online search
            self.ed_search.returnPressed.connect(self.run_search)
            self.tbl_pkgs.itemChanged.connect(self._on_table_item_changed)
        except Exception:
            pass

    def showEvent(self, event):
        """On first show, automatically list installed packages."""
        try:
            super().showEvent(event)
        except Exception:
            pass
        if not getattr(self, "_did_initial_refresh", False):
            self._did_initial_refresh = True
            # Populate installed packages by default when dialog opens
            try:
                self.run_list_installed()
            except Exception:
                pass

    def closeEvent(self, event):
        """Mark closing to guard late signals from background workers."""
        try:
            self._closing = True
        except Exception:
            pass
        try:
            super().closeEvent(event)
        except Exception:
            pass

    # ---------- Helpers ----------
    def _append_log(self, text: str):
        try:
            logger.info(text)
        except Exception:
            pass
        try:
            self.txt_log.append(text)
        except Exception:
            pass

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable main interactive controls to avoid accidental actions while busy."""
        try:
            names = [
                'ed_search', 'btn_search', 'btn_list_installed',
                'tbl_pkgs', 'btn_toggle_select', 'btn_apply', 'btn_update', 'btn_update_all',
                # Hidden tabs, keep safe guards
                'btn_env_refresh', 'btn_env_create', 'btn_env_remove', 'btn_env_clone',
                'btn_env_export', 'btn_env_import', 'list_envs',
                'btn_channel_add', 'btn_channel_remove', 'list_channels', 'ed_channel'
            ]
            for n in names:
                try:
                    w = getattr(self, n, None)
                    if w is not None:
                        w.setEnabled(bool(enabled))
                except Exception:
                    continue
        except Exception:
            pass

    def _is_busy(self) -> bool:
        try:
            return getattr(self, '_busy_count', 0) > 0
        except Exception:
            return False

    def _show_busy(self, message: str) -> None:
        """Show an application-modal, auto-closing info box to block the UI during long ops."""
        try:
            cnt = getattr(self, '_busy_count', 0)
            if cnt <= 0 or not hasattr(self, '_busy_box') or self._busy_box is None:
                from PyQt5.QtWidgets import QMessageBox
                self._busy_box = QMessageBox(self)
                try:
                    self._busy_box.setIcon(QMessageBox.Information)
                except Exception:
                    pass
                try:
                    self._busy_box.setWindowTitle("Please wait")
                except Exception:
                    pass
                try:
                    self._busy_box.setStandardButtons(QMessageBox.NoButton)
                except Exception:
                    pass
                try:
                    self._busy_box.setWindowModality(Qt.ApplicationModal)
                except Exception:
                    pass
            # Update text and show
            try:
                base = "This may take a while — please be patient."
                msg = message.strip()
                text = msg if msg else base
                if msg:
                    text = f"{msg}\n\n{base}"
                self._busy_box.setText(text)
            except Exception:
                pass
            try:
                self._busy_box.show()
            except Exception:
                pass
            self._busy_count = cnt + 1
            self._set_controls_enabled(False)
        except Exception:
            pass

    def _hide_busy(self) -> None:
        """Decrease busy counter and close the busy box when it reaches zero."""
        try:
            cnt = getattr(self, '_busy_count', 0)
            cnt -= 1
            if cnt <= 0:
                self._busy_count = 0
                try:
                    if hasattr(self, '_busy_box') and self._busy_box is not None:
                        try:
                            self._busy_box.close()
                        except Exception:
                            pass
                        self._busy_box = None
                except Exception:
                    pass
                self._set_controls_enabled(True)
            else:
                self._busy_count = cnt
        except Exception:
            pass

    def _reset_busy(self) -> None:
        """Forcefully close any busy dialog and reset internal busy state.
        Use this as a safety net when an operation finished but the modal remained.
        """
        try:
            self._busy_count = 0
            try:
                if hasattr(self, '_busy_box') and self._busy_box is not None:
                    try:
                        self._busy_box.close()
                    except Exception:
                        pass
                    self._busy_box = None
            except Exception:
                pass
            # Reset flow flags so future operations behave predictably
            try:
                self._busy_until_refresh = False
            except Exception:
                pass
            try:
                self._list_busy_own = False
            except Exception:
                pass
            self._set_controls_enabled(True)
        except Exception:
            pass

    def refresh_repo_info(self):
        """Query conda for repo/cache dirs and display them under the output box."""
        try:
            worker = CondaWorker(self.manager.info)
            self._track_worker(worker)
            worker.finished.connect(self._on_repo_info)
            worker.start()
        except Exception:
            # Best-effort only
            pass

    def _on_repo_info(self, ok: bool, payload: object, msg: str):
        try:
            if not ok or not isinstance(payload, dict):
                # Show at least the current env
                prefix = self._current_env_prefix()
                self.lbl_repo.setText(f"Env: {prefix}")
                return
            info = payload
            pkgs_dirs = []
            try:
                if isinstance(info.get('pkgs_dirs'), list):
                    pkgs_dirs = [str(p) for p in info.get('pkgs_dirs') if p]
            except Exception:
                pkgs_dirs = []
            prefix = self._current_env_prefix()
            repo_dir = pkgs_dirs[0] if pkgs_dirs else "(unknown)"
            self.lbl_repo.setText(f"Repo: {repo_dir}    |    Env: {prefix}")
        except Exception:
            pass

    def _on_table_item_changed(self, item: 'QTableWidgetItem') -> None:
        """Keep persistent selection set in sync with user checkbox changes."""
        try:
            if getattr(self, '_suppress_item_changed', False):
                return
            if item is None:
                return
            row = item.row()
            col = item.column()
            if col != 0:
                return
            name_item = self.tbl_pkgs.item(row, 1)
            if not name_item:
                return
            nm = name_item.text().strip()
            if not nm:
                return
            if item.checkState() == Qt.Checked:
                self._checked_names.add(nm)
            else:
                if nm in self._checked_names:
                    self._checked_names.remove(nm)
            # Update label to reflect new state
            self._update_select_toggle_label()
        except Exception:
            pass

    def _set_packages(self, records: List[Dict[str, Any]]):
        """Set current dataset and populate the table."""
        self._all_packages = records or []
        self._apply_pkg_filter()

    def _populate_pkg_table(self, rows: List[Dict[str, Any]]):
        try:
            self._suppress_item_changed = True
            self.tbl_pkgs.setRowCount(0)
            for rec in rows:
                row = self.tbl_pkgs.rowCount()
                self.tbl_pkgs.insertRow(row)
                # Checkbox item
                chk_item = QTableWidgetItem("")
                try:
                    chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    # Decide checked state from persistent selection set
                    name_for_check = str(rec.get('name', ''))
                    is_checked = name_for_check in getattr(self, '_checked_names', set())
                    chk_item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
                except Exception:
                    pass
                self.tbl_pkgs.setItem(row, 0, chk_item)
                # Name, Version, Build, Source
                name = str(rec.get('name', ''))
                ver = str(rec.get('version', ''))
                bld = str(rec.get('build', '') or rec.get('build_string', ''))
                src = str(rec.get('channel', '') or rec.get('subdir', '') or rec.get('source', ''))
                self.tbl_pkgs.setItem(row, 1, QTableWidgetItem(name))
                self.tbl_pkgs.setItem(row, 2, QTableWidgetItem(ver))
                self.tbl_pkgs.setItem(row, 3, QTableWidgetItem(bld))
                self.tbl_pkgs.setItem(row, 4, QTableWidgetItem(src))
            # Update toggle button label after repopulating
            if hasattr(self, '_update_select_toggle_label'):
                self._update_select_toggle_label()
        except Exception:
            pass
        finally:
            try:
                self._suppress_item_changed = False
            except Exception:
                pass

    def _apply_pkg_filter(self):
        """Filter current dataset by search text and update table."""
        query = (self.ed_search.text() or "").strip().lower()
        if not self._all_packages:
            # Nothing loaded yet
            self.tbl_pkgs.setRowCount(0)
            return
        if not query:
            self._filtered_packages = list(self._all_packages)
        else:
            q = query
            out: List[Dict[str, Any]] = []
            for rec in self._all_packages:
                try:
                    hay = " ".join([
                        str(rec.get('name', '')),
                        str(rec.get('version', '')),
                        str(rec.get('build', '') or rec.get('build_string', '')),
                        str(rec.get('channel', '') or rec.get('subdir', '') or rec.get('source', ''))
                    ]).lower()
                    if q in hay:
                        out.append(rec)
                except Exception:
                    continue
            self._filtered_packages = out
        self._populate_pkg_table(self._filtered_packages)

    def _checked_pkg_names(self) -> List[str]:
        """Return persistent set of checked package names as a list."""
        try:
            return sorted(list(getattr(self, '_checked_names', set())))
        except Exception:
            return []

    def _visible_rows_checked_state(self) -> Tuple[int, int]:
        """Return (checked_count, total_visible_rows)."""
        try:
            rows = self.tbl_pkgs.rowCount()
            checked = 0
            for r in range(rows):
                it = self.tbl_pkgs.item(r, 0)
                if it is not None and it.checkState() == Qt.Checked:
                    checked += 1
            return checked, rows
        except Exception:
            return 0, 0

    def _update_select_toggle_label(self) -> None:
        try:
            checked, total = self._visible_rows_checked_state()
            if total == 0:
                self.btn_toggle_select.setText("Select Visible")
            elif checked < total:
                self.btn_toggle_select.setText("Select Visible")
            else:
                self.btn_toggle_select.setText("Deselect Visible")
        except Exception:
            pass

    def toggle_select_visible(self) -> None:
        """Toggle check state for currently visible rows only, and persist selection set."""
        try:
            checked, total = self._visible_rows_checked_state()
            target_state = Qt.Checked if checked < total else Qt.Unchecked
            rows = self.tbl_pkgs.rowCount()
            # Collect visible names
            visible_names: List[str] = []
            for r in range(rows):
                name_item = self.tbl_pkgs.item(r, 1)
                if name_item:
                    nm = name_item.text().strip()
                    if nm:
                        visible_names.append(nm)
            # Update persistent set first
            if target_state == Qt.Checked:
                for nm in visible_names:
                    self._checked_names.add(nm)
            else:
                for nm in visible_names:
                    if nm in self._checked_names:
                        self._checked_names.remove(nm)
            # Now reflect in table without triggering itemChanged storm
            self._suppress_item_changed = True
            try:
                for r in range(rows):
                    it = self.tbl_pkgs.item(r, 0)
                    if it is not None:
                        it.setCheckState(target_state)
            finally:
                self._suppress_item_changed = False
        except Exception:
            pass
        # Update label after toggling
        self._update_select_toggle_label()

    def _cleanup_worker(self, worker: "CondaWorker") -> None:
        try:
            if worker in self._workers:
                self._workers.remove(worker)
        except Exception:
            pass

    def _track_worker(self, worker: "CondaWorker") -> None:
        """Ensure worker lives as long as needed to avoid SIP/Qt crashes."""
        try:
            worker.setParent(self)
        except Exception:
            pass
        self._workers.append(worker)
        try:
            worker.finished.connect(lambda *_: self._cleanup_worker(worker))
            worker.finished.connect(worker.deleteLater)
        except Exception:
            pass

    def _has_active_workers(self) -> bool:
        try:
            return any(w.isRunning() for w in self._workers)
        except Exception:
            return False

    def _selected_pkg_names(self) -> List[str]:
        """Backward-compatible alias to checked names in the table."""
        return self._checked_pkg_names()

    def _current_env_prefix(self) -> str:
        # Use manager's current prefix
        return self.manager.current_prefix()

    # ---------- Packages ----------
    def run_search(self):
        query = self.ed_search.text().strip()
        if not query:
            return
        self._append_log(f"Searching for '{query}'...")
        worker = CondaWorker(self.manager.search, query)
        self._track_worker(worker)
        worker.finished.connect(self._on_search_finished)
        worker.start()

    def run_apply_changes(self):
        """Apply install/remove based on checkboxes vs installed state."""
        # Build sets
        try:
            installed = set(self._installed_names or set())
            checked = set(self._selected_pkg_names())
        except Exception:
            installed = set()
            checked = set()
        to_install = sorted(list(checked - installed))
        to_remove = sorted(list(installed - checked))
        if not to_install and not to_remove:
            QMessageBox.information(self, "Apply Changes", "No changes to apply.")
            return
        summary_lines = []
        if to_install:
            summary_lines.append(f"Install: {', '.join(to_install)}")
        if to_remove:
            summary_lines.append(f"Remove: {', '.join(to_remove)}")
        # Store plan for final summary
        self._apply_plan_install = to_install
        self._apply_plan_remove = to_remove
        if QMessageBox.question(
            self,
            "Confirm changes",
            "\n".join(summary_lines),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return
        # Queue operations: remove first, then install
        self._apply_queue: List[Tuple[str, List[str]]] = []
        if to_remove:
            self._apply_queue.append(("remove", to_remove))
        if to_install:
            self._apply_queue.append(("install", to_install))
        self._apply_running = False
        # No blocking modal; log status and start
        self._append_log("Applying changes…")
        self._run_next_apply_op()

    def _run_next_apply_op(self):
        if getattr(self, "_apply_running", False):
            return
        if not getattr(self, "_apply_queue", []):
            # Finished sequence
            self._append_log("Apply changes complete.")
            # Defer final summary dialog until after the installed list refresh completes
            try:
                plan_install = getattr(self, "_apply_plan_install", []) or []
                plan_remove = getattr(self, "_apply_plan_remove", []) or []
                lines = []
                if plan_install:
                    lines.append(f"Installed: {', '.join(plan_install)}")
                if plan_remove:
                    lines.append(f"Removed: {', '.join(plan_remove)}")
                self._pending_apply_summary_lines = lines
            except Exception:
                self._pending_apply_summary_lines = []
            # Trigger a refresh of installed packages; keep busy box up until this finishes
            try:
                self.run_list_installed()
            except Exception:
                pass
            return
        op, pkgs = self._apply_queue.pop(0)
        self._apply_running = True
        prefix = self._current_env_prefix()
        if op == "remove":
            self._append_log(f"Removing: {' '.join(pkgs)}")
            worker = CondaWorker(self.manager.remove, pkgs, prefix)
        else:
            self._append_log(f"Installing: {' '.join(pkgs)}")
            worker = CondaWorker(self.manager.install, pkgs, prefix)
        self._track_worker(worker)
        worker.finished.connect(self._on_apply_step_done)
        worker.start()

    def _on_apply_step_done(self, ok: bool, payload: object, msg: str):
        # Log result of the step
        if not ok:
            self._append_log(f"Step failed: {msg}")
            QMessageBox.warning(self, "Apply Changes", msg or "Step failed")
            # Continue to next to allow partial application or stop? We'll continue.
        else:
            # Stream payload to log similar to _on_text_result
            text = ""
            if isinstance(payload, str):
                text = payload
            elif isinstance(payload, (dict, list)):
                try:
                    text = json.dumps(payload, indent=2)
                except Exception:
                    text = str(payload)
            else:
                text = str(payload)
            if text:
                self._append_log(text)
        self._apply_running = False
        self._run_next_apply_op()

    def _on_search_finished(self, ok: bool, payload: object, msg: str):
        if not ok:
            QMessageBox.warning(self, "Search failed", msg or "Unknown error")
            self._append_log(f"Search failed: {msg}")
            self._hide_busy()
            return
        try:
            records: List[Dict[str, Any]] = []
            installed = getattr(self, '_installed_names', set())
            if isinstance(payload, dict):
                for name, recs in payload.items():
                    if isinstance(recs, list) and recs:
                        # Use the last record (often newest)
                        rec = recs[-1]
                        nm = rec.get('name', name)
                        records.append({
                            'name': nm,
                            'version': rec.get('version', ''),
                            'build': rec.get('build', '') or rec.get('build_string', ''),
                            'channel': rec.get('channel', '') or rec.get('subdir', ''),
                            'installed': nm in installed
                        })
                    else:
                        nm = str(name)
                        records.append({'name': nm, 'version': '', 'build': '', 'channel': '', 'installed': nm in installed})
            elif isinstance(payload, list):
                for rec in payload:
                    try:
                        nm = rec.get('name', '')
                        records.append({
                            'name': nm,
                            'version': rec.get('version', ''),
                            'build': rec.get('build', '') or rec.get('build_string', ''),
                            'channel': rec.get('channel', '') or rec.get('subdir', ''),
                            'installed': nm in installed
                        })
                    except Exception:
                        s = str(rec)
                        records.append({'name': s, 'version': '', 'build': '', 'channel': '', 'installed': s in installed})
            # Ensure installed packages appear checked in the table by default
            try:
                self._checked_names |= set(installed)
            except Exception:
                pass
            self._set_packages(records)
            self._append_log(f"Search returned {len(records)} items.")
        except Exception as e:
            self._append_log(f"Error parsing search results: {e}")
        finally:
            pass

    def run_list_installed(self):
        logger.info("Listing installed packages...")
        self._append_log("Listing installed packages...")
        worker = CondaWorker(self.manager.list_installed, self._current_env_prefix())
        self._track_worker(worker)
        worker.finished.connect(self._on_list_installed_finished)
        worker.start()

    def _on_list_installed_finished(self, ok: bool, payload: object, msg: str):
        if not ok:
            QMessageBox.warning(self, "List failed", msg or "Unknown error")
            self._append_log(f"List failed: {msg}")
            return
        try:
            records: List[Dict[str, Any]] = []
            installed_names: Set[str] = set()
            if isinstance(payload, list):
                for rec in payload:
                    try:
                        nm = rec.get('name', '')
                        installed_names.add(nm)
                        records.append({
                            'name': nm,
                            'version': rec.get('version', ''),
                            'build': rec.get('build_string', '') or rec.get('build', ''),
                            'channel': rec.get('channel', '') or rec.get('subdir', ''),
                            'installed': True
                        })
                    except Exception:
                        s = str(rec)
                        if s:
                            installed_names.add(s)
                        records.append({'name': s, 'version': '', 'build': '', 'channel': '', 'installed': True})
            # Save installed names and initialize persistent checked set to installed
            self._installed_names = installed_names
            self._checked_names = set(installed_names)
            self._set_packages(records)
            self._append_log(f"Installed packages: {len(records)}")
            try:
                self.refresh_repo_info()
            except Exception:
                pass
            # If this list was part of a larger busy operation, hide busy now and show final summary
            try:
                if self._list_busy_own or self._busy_until_refresh:
                    self._hide_busy()
                    self._list_busy_own = False
                    if self._busy_until_refresh:
                        self._busy_until_refresh = False
                        # Show pending apply summary if present
                        try:
                            lines = getattr(self, '_pending_apply_summary_lines', []) or []
                            if lines:
                                QMessageBox.information(self, "Apply Changes", "\n".join(lines))
                        except Exception:
                            pass
                        self._pending_apply_summary_lines = []
            except Exception:
                pass
        except Exception as e:
            self._append_log(f"Error parsing list results: {e}")
        # Safety: ensure the busy dialog is closed even if exceptions occurred above
        try:
            if self._list_busy_own or self._busy_until_refresh or not self._has_active_workers():
                # If this list was part of an apply/change flow, show the pending summary now
                try:
                    lines = getattr(self, '_pending_apply_summary_lines', []) or []
                    if lines and self._busy_until_refresh:
                        QMessageBox.information(self, "Apply Changes", "\n".join(lines))
                except Exception:
                    pass
                try:
                    self._pending_apply_summary_lines = []
                except Exception:
                    pass
                self._reset_busy()
        except Exception:
            pass

    def run_install(self):
        pkgs = self._selected_pkg_names()
        if not pkgs:
            QMessageBox.information(self, "Install", "Select one or more packages from the list.")
            return
        logger.info("Installing selected packages: %s", " ".join(pkgs))
        self._append_log(f"Installing: {' '.join(pkgs)}")
        worker = CondaWorker(self.manager.install, pkgs, self._current_env_prefix())
        self._track_worker(worker)
        worker.finished.connect(self._on_text_result)
        worker.start()

    def run_remove(self):
        pkgs = self._selected_pkg_names()
        if not pkgs:
            QMessageBox.information(self, "Remove", "Select one or more packages from the list.")
            return
        self._append_log(f"Removing: {' '.join(pkgs)}")
        self._busy_until_refresh = True
        self._show_busy("Removing selected packages…")
        worker = CondaWorker(self.manager.remove, pkgs, self._current_env_prefix())
        self._track_worker(worker)
        worker.finished.connect(self._on_text_result)
        worker.start()

    def run_update(self):
        pkgs = self._selected_pkg_names()
        if not pkgs:
            QMessageBox.information(self, "Update", "Select one or more packages, or click 'Update All'.")
            return
        logger.info("Updating selected packages: %s", " ".join(pkgs))
        self._append_log(f"Updating: {' '.join(pkgs)}")
        worker = CondaWorker(self.manager.update, pkgs, self._current_env_prefix())
        self._track_worker(worker)
        worker.finished.connect(self._on_text_result)
        worker.start()

    def run_update_all(self):
        logger.info("Updating all packages ...")
        self._append_log("Updating all packages (this may take a while)...")
        worker = CondaWorker(self.manager.update, None, self._current_env_prefix())
        self._track_worker(worker)
        worker.finished.connect(self._on_text_result)
        worker.start()

    def _on_text_result(self, ok: bool, payload: object, msg: str):
        text = ""
        if isinstance(payload, str):
            text = payload
        elif isinstance(payload, (dict, list)):
            try:
                text = json.dumps(payload, indent=2)
            except Exception:
                text = str(payload)
        else:
            text = str(payload)
        if not ok:
            # Show error and ensure busy UI is dismissed
            QMessageBox.warning(self, "Operation failed", msg or text or "Unknown error")
            try:
                if self._is_busy():
                    self._busy_until_refresh = False
                    self._list_busy_own = False
                    self._hide_busy()
            except Exception:
                pass
            self._append_log(text or (msg or "Failed"))
            return
        # OK path: log and refresh info; if the operation changes packages, run_list_installed will close busy later
        self._append_log(text or (msg or "Success"))
        try:
            self.refresh_repo_info()
        except Exception:
            pass
        try:
            if ok:
                self.run_list_installed()
            else:
                # If somehow not ok falls through, hide busy safely
                if self._is_busy():
                    self._hide_busy()
        except Exception:
            # If unable to refresh list, hide busy now to avoid locking the UI
            try:
                if self._is_busy():
                    self._hide_busy()
            except Exception:
                pass

    # ---------- Environments ----------
    def refresh_envs(self):
        if getattr(self, "_closing", False):
            return
        worker = CondaWorker(self.manager.list_envs)
        self._track_worker(worker)
        worker.finished.connect(self._on_envs)
        worker.start()

    def _on_envs(self, ok: bool, payload: object, msg: str):
        # Ignore late signals if dialog is closing or widget already deleted
        try:
            if getattr(self, '_closing', False):
                return
        except Exception:
            pass
        try:
            self.list_envs.blockSignals(True)
        except Exception:
            pass
        try:
            self.list_envs.clear()
            if not ok:
                self._append_log(f"Env list failed: {msg}")
                return
            envs: List[str] = []
            if isinstance(payload, list):
                # normalize all entries to strings
                envs = [str(p) for p in payload if p is not None]
            elif isinstance(payload, dict):
                # Some conda variants might return a mapping; show its values/keys
                # Prefer 'envs' key if present
                if 'envs' in payload and isinstance(payload['envs'], list):
                    envs = [str(p) for p in payload['envs'] if p is not None]
                else:
                    envs = [str(k) for k in payload.keys()]
            else:
                envs = []
            for p in envs:
                try:
                    self.list_envs.addItem(QListWidgetItem(p))
                except Exception:
                    # Best-effort add
                    self.list_envs.addItem(p)
            self._append_log(f"Environments: {len(envs)}")
        finally:
            try:
                self.list_envs.blockSignals(False)
            except Exception:
                pass

    def create_env(self):
        name = self.ed_env_name.text().strip()
        if not name:
            # Ask for folder path for prefix
            prefix = QFileDialog.getExistingDirectory(self, "Select environment folder")
            if not prefix:
                return
            worker = CondaWorker(self.manager.create_env, None, prefix, None, None)
        else:
            worker = CondaWorker(self.manager.create_env, name, None, None, None)
        self._append_log("Creating environment... this may take a while")
        self._track_worker(worker)
        worker.finished.connect(self._on_text_result)
        worker.finished.connect(lambda *_: self.refresh_envs())
        worker.start()

    def _selected_env_prefix(self) -> Optional[str]:
        it = self.list_envs.currentItem()
        return str(it.text()) if it else None

    def remove_env(self):
        prefix = self._selected_env_prefix()
        if not prefix:
            QMessageBox.information(self, "Remove env", "Select an environment from the list.")
            return
        if QMessageBox.question(self, "Confirm", f"Remove environment at\n{prefix}?") != QMessageBox.Yes:
            return
        worker = CondaWorker(self.manager.remove_env, None, prefix)
        self._append_log(f"Removing env: {prefix}")
        self._track_worker(worker)
        worker.finished.connect(self._on_text_result)
        worker.finished.connect(lambda *_: self.refresh_envs())
        worker.start()

    def export_env(self):
        prefix = self._selected_env_prefix() or self._current_env_prefix()
        worker = CondaWorker(self.manager.export_env, prefix)
        self._append_log("Exporting environment to YAML text...")
        def on_done(ok: bool, payload: object, msg: str):
            if getattr(self, '_closing', False):
                return
            if ok and isinstance(payload, str):
                path, _ = QFileDialog.getSaveFileName(self, "Save environment.yml", "environment.yml", "YAML Files (*.yml *.yaml)")
                if path:
                    try:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(payload)
                        self._append_log(f"Saved: {path}")
                    except Exception as e:
                        QMessageBox.warning(self, "Save failed", str(e))
            else:
                QMessageBox.warning(self, "Export failed", msg or "Unknown error")
                self._append_log(msg or "Export failed")
        self._track_worker(worker)
        worker.finished.connect(on_done)
        worker.start()

    def import_env(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select environment.yml", "", "YAML Files (*.yml *.yaml)")
        if not path:
            return
        # Optional environment name input
        name, ok = QInputDialog.getText(self, "Environment name (optional)", "Name:")
        if not ok:
            name = None
        name = str(name).strip() if name else None
        if name == "":
            name = None
        worker = CondaWorker(self.manager.import_env, path, name)
        self._append_log(f"Creating environment from {path}...")
        self._track_worker(worker)
        worker.finished.connect(self._on_text_result)
        worker.finished.connect(lambda *_: self.refresh_envs())
        worker.start()

    def clone_env(self):
        src = self._selected_env_prefix()
        if not src:
            QMessageBox.information(self, "Clone env", "Select a source environment from the list.")
            return
        dst = QFileDialog.getExistingDirectory(self, "Select destination folder for cloned environment")
        if not dst:
            return
        worker = CondaWorker(self.manager.clone_env, None, src, None, dst)
        self._append_log(f"Cloning env from {src} to {dst}...")
        self._track_worker(worker)
        worker.finished.connect(self._on_text_result)
        worker.finished.connect(lambda *_: self.refresh_envs())
        worker.start()

    # ---------- Channels ----------
    def refresh_channels(self):
        if getattr(self, "_closing", False):
            return
        worker = CondaWorker(self.manager.get_channels)
        self._track_worker(worker)
        worker.finished.connect(self._on_channels)
        worker.start()

    def _on_channels(self, ok: bool, payload: object, msg: str):
        # Ignore late signals if dialog is closing
        try:
            if getattr(self, '_closing', False):
                return
        except Exception:
            pass
        try:
            self.list_channels.blockSignals(True)
        except Exception:
            pass
        try:
            self.list_channels.clear()
            if not ok:
                self._append_log(f"Channels read failed: {msg}")
                return
            channels: List[str] = []
            if isinstance(payload, list):
                channels = [str(ch) for ch in payload if ch is not None]
            elif isinstance(payload, dict):
                # try common key
                if 'channels' in payload and isinstance(payload['channels'], list):
                    channels = [str(ch) for ch in payload['channels'] if ch is not None]
                else:
                    channels = [str(k) for k in payload.keys()]
            for ch in channels:
                try:
                    self.list_channels.addItem(QListWidgetItem(ch))
                except Exception:
                    self.list_channels.addItem(ch)
            self._append_log(f"Channels: {', '.join(channels) if channels else '(none)'}")
        finally:
            try:
                self.list_channels.blockSignals(False)
            except Exception:
                pass

    def add_channel(self):
        ch = self.ed_channel.text().strip()
        if not ch:
            return
        worker = CondaWorker(self.manager.add_channel, ch)
        self._append_log(f"Adding channel: {ch}")
        worker.finished.connect(self._on_text_result)
        worker.finished.connect(lambda *_: self.refresh_channels())
        worker.start()

    def remove_channel(self):
        it = self.list_channels.currentItem()
        if not it:
            return
        ch = it.text().strip()
        worker = CondaWorker(self.manager.remove_channel, ch)
        self._append_log(f"Removing channel: {ch}")
        worker.finished.connect(self._on_text_result)
        worker.finished.connect(lambda *_: self.refresh_channels())
        worker.start()
