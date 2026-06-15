"""GUI plugin for managing the Multiparametric Fluorescence Database."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QUrl

try:
    from qtpy import sip
except ImportError:
    try:
        import sip
    except ImportError:
        sip = None

from chisurf.gui.misc_helpers import get_plugin_settings_path
from chisurf.gui.widgets.dock_area import DockArea
from chisurf.gui.widgets.metadata_editor import MetadataEditor

from .client import MFDBClient


def _qurl_for_location(location: str) -> QtCore.QUrl:
    """Return a QUrl for either a local filesystem path or an existing URL."""
    url = QUrl(location)
    if url.scheme():
        return url
    return QUrl.fromLocalFile(location)


def _processed_location(item: dict[str, Any]) -> str:
    """Return the stored raw or processed data location."""
    return item.get("file_path") or item.get("url") or item.get("folder_path") or ""


def _processed_row_count(item: dict[str, Any]) -> Any:
    """Return the stored processed-data row count."""
    return item.get("row_count", "")


def _experiment_processing_ids(client: Any, experiment_id: str | None) -> set[str]:
    """Return processing run IDs scoped to an experiment."""
    if not experiment_id:
        return set()
    try:
        runs = client.list_processing_runs(experiment_id=experiment_id)
    except Exception:
        return set()
    return {str(run.get("processing_id", "")) for run in runs if run.get("processing_id")}


def _scope_processed_products_by_experiment(
    products: list[dict[str, Any]],
    client: Any,
    experiment_id: str | None,
) -> list[dict[str, Any]]:
    """Return processed products scoped to an experiment through processing runs."""
    if not experiment_id:
        return products
    processing_ids = _experiment_processing_ids(client, experiment_id)
    return [product for product in products if product.get("processing_id") in processing_ids]


def _experiment_id_for_processing_id(client: Any, processing_id: str | None) -> str:
    """Return the experiment ID for a processing run when available."""
    if not processing_id:
        return ""
    try:
        return client.get_processing_run(processing_id).get("experiment_id", "")
    except Exception:
        return ""


def _unwrap_provenance_graph_response(response: dict[str, Any]) -> dict[str, Any]:
    """Return the graph payload from export responses or the response itself."""
    if isinstance(response, dict) and "graph" in response:
        graph = response.get("graph")
        return graph if isinstance(graph, dict) else {}
    return response


class PasswordChangeDialog(QtWidgets.QDialog):
    """Dialog for changing an MFDB password with strength feedback."""

    def __init__(
        self,
        user_id: str,
        is_admin: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Create the password change dialog."""
        super().__init__(parent)
        self.user_id = user_id
        self.is_admin = is_admin
        self.password = ""
        self.cleared = False
        self.score = 0
        self.setWindowTitle(f"Change Password for {user_id}")
        self.resize(380, 240)
        self._setup_ui()
        self._update_strength("")

    def _setup_ui(self) -> None:
        """Build dialog widgets."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        self.password_edit = QtWidgets.QLineEdit()
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_edit.textChanged.connect(self._update_strength)
        self.confirm_edit = QtWidgets.QLineEdit()
        self.confirm_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        form.addRow("New password", self.password_edit)
        form.addRow("Confirm password", self.confirm_edit)
        layout.addLayout(form)

        self.strength_label = QtWidgets.QLabel()
        layout.addWidget(self.strength_label)

        self.strength_bar = QtWidgets.QProgressBar()
        self.strength_bar.setRange(0, 100)
        self.strength_bar.setTextVisible(False)
        self.strength_bar.setMaximumHeight(8)
        layout.addWidget(self.strength_bar)

        self.feedback_label = QtWidgets.QLabel()
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setStyleSheet("color: #555555;")
        layout.addWidget(self.feedback_label)

        buttons = QtWidgets.QHBoxLayout()
        save_button = QtWidgets.QToolButton()
        save_button.setText("💾 Save")
        save_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        save_button.setAutoRaise(True)
        save_button.setToolTip("Save password")
        clear_button = QtWidgets.QToolButton()
        clear_button.setText("🚫 Clear")
        clear_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        clear_button.setAutoRaise(True)
        clear_button.setToolTip(
            "Clear the password — the user will be able to log in without one"
            + (" (disabled for administrators)" if self.is_admin else "")
        )
        clear_button.setEnabled(not self.is_admin)
        cancel_button = QtWidgets.QToolButton()
        cancel_button.setText("❌ Cancel")
        cancel_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        cancel_button.setAutoRaise(True)
        cancel_button.setToolTip("Cancel password change")
        save_button.clicked.connect(self._accept_password)
        clear_button.clicked.connect(self._clear_password)
        cancel_button.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(save_button)
        buttons.addWidget(clear_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)

    def _update_strength(self, password: str) -> None:
        """Update the color-coded strength indicator."""
        self.score, feedback = self._score_password(password)
        self.strength_bar.setValue(self.score * 20)

        if not password:
            self._set_strength_style("#999999", "Enter a password", 0)
            self.feedback_label.clear()
        elif self.score <= 2:
            self._set_strength_style("#c62828", "Strength: Weak", self.score * 20)
            self.feedback_label.setText("Requirements: " + ", ".join(feedback))
        elif self.score <= 4:
            self._set_strength_style("#f9a825", "Strength: Medium", self.score * 20)
            self.feedback_label.setText("Requirements: " + ", ".join(feedback))
        else:
            self._set_strength_style("#2e7d32", "Strength: Strong", 100)
            self.feedback_label.clear()

    def _set_strength_style(self, color: str, label: str, value: int) -> None:
        """Apply the current strength color and label."""
        self.strength_label.setText(label)
        self.strength_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.strength_bar.setValue(value)
        self.strength_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

    def _accept_password(self) -> None:
        """Validate matching passwords and close the dialog."""
        password = self.password_edit.text()
        if password != self.confirm_edit.text():
            QtWidgets.QMessageBox.warning(self, "Validation Error", "Passwords do not match.")
            return
        if self.is_admin and not password:
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Error",
                "Administrator passwords cannot be empty.",
            )
            return
        if self.is_admin and password and self.score < 4:
            QtWidgets.QMessageBox.warning(
                self,
                "Validation Error",
                "Administrator passwords must be at least medium strength.",
            )
            return
        self.password = password
        self.cleared = False
        self.accept()

    def _clear_password(self) -> None:
        """Confirm and accept the dialog with an empty (cleared) password."""
        if self.is_admin:
            QtWidgets.QMessageBox.warning(
                self,
                "Not allowed",
                "Administrator passwords cannot be cleared.",
            )
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Clear password",
            (
                f"Remove the password for '{self.user_id}'?\n\n"
                "The user will be able to log in without a password. "
                "Enable 'Allow passwordless login' on the user record if "
                "you want passwordless login to actually work."
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        self.password = ""
        self.cleared = True
        self.accept()

    @staticmethod
    def _score_password(password: str) -> tuple[int, list[str]]:
        """Return strength score and missing requirements."""
        score = 0
        feedback: list[str] = []
        if len(password) >= 8:
            score += 1
        else:
            feedback.append("at least 8 characters")
        if any(char.islower() for char in password):
            score += 1
        else:
            feedback.append("one lowercase letter")
        if any(char.isupper() for char in password):
            score += 1
        else:
            feedback.append("one uppercase letter")
        if any(char.isdigit() for char in password):
            score += 1
        else:
            feedback.append("one number")
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        if any(char in special_chars for char in password):
            score += 1
        else:
            feedback.append("one special character")
        return score, feedback


class MFDBWidget(QtWidgets.QMainWindow):
    """Window for browsing, editing, importing, and exporting samples."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, client: Any | None = None):
        super().__init__(parent)
        self.client = client or MFDBClient()
        self._loading = False
        self._auth_login_user: str | None = None
        self._checkable_tables: list[QtWidgets.QTableWidget] = []

        self._verify_admin_access()
        self._ensure_authenticated()

        # Selection state
        self.current_sample_id = None
        self.current_experiment_id = None
        self.current_provenance_seed_type = "processed_data"
        self.current_provenance_seed_id = None

        self.setWindowTitle("mfdb-admin — Multiparametric Fluorescence Database")
        self.resize(1100, 760)
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_status_bar()
        self._update_login_actions(logged_in=bool(getattr(self.client, "token", None)))
        self.refresh()

    def _verify_admin_access(self) -> None:
        """Raise ``PermissionError`` when the active user is not an admin.

        Access is allowed for administrators and during bootstrap (when no
        admin user exists yet). When the MFDB transport is unreachable the
        check is skipped so the widget can still surface the transport error
        through its normal status path.
        """
        try:
            users = self.client.list_users()
        except Exception:
            return
        if not users:
            return
        has_any_admin = any(u.get("is_admin") for u in users)
        if not has_any_admin:
            return
        active_id = self._active_mfdb_user_id()
        active = next((u for u in users if u.get("user_id") == active_id), None)
        if active and active.get("is_admin"):
            return
        raise PermissionError(
            f"User '{active_id}' is not an administrator. "
            "mfdb-admin is restricted to MFDB administrators."
        )

    def _ensure_authenticated(self, username: str | None = None, password: str | None = None) -> None:
        """Acquire an MFDB session token for the active user.

        Most ``mfdb.*`` endpoints require authentication. We first attempt a
        passwordless login (works for bootstrap users and accounts that have
        not set a password yet); if that fails we prompt the operator for the
        password, retrying up to three times. Cancelling the prompt is allowed
        — the widget still opens but most tables will surface auth errors.
        
        Parameters
        ----------
        username : str, optional
            User ID to authenticate as. Falls back to active MFDB user if not provided.
        password : str, optional
            Password to use for authentication. If not provided, tries passwordless login first.
        """
        if getattr(self.client, "token", None):
            self._auth_login_user = getattr(self.client, "_auth_user_id", None)
            return
            
        user_id = username or self._active_mfdb_user_id()
        if not user_id:
            return
        
        client_metadata = {"name": "mfdb-admin", "host": "local"}
        
        # Try with provided password first, or passwordless if no password provided
        try:
            result = self.client.login(
                user_id=user_id, password=password or "", client_metadata=client_metadata
            )
        except Exception:
            result = {}
            
        if isinstance(result, dict) and result.get("ok"):
            self._auth_login_user = user_id
            return
            
        # Only show error if password was explicitly provided and non-empty
        # If password is empty or None, prompt for it
        if password is not None and password != "":
            QtWidgets.QMessageBox.warning(
                self,
                "MFDB login failed",
                "Invalid credentials. Please try again.",
            )
            return
            
        # No password provided or it failed, prompt for password
        for _ in range(3):
            password, ok = QtWidgets.QInputDialog.getText(
                self,
                f"MFDB login required ({user_id})",
                f"Password for '{user_id}':",
                QtWidgets.QLineEdit.Password,
            )
            if not ok:
                return
            try:
                result = self.client.login(
                    user_id=user_id,
                    password=password,
                    client_metadata=client_metadata,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "MFDB login failed", str(exc)
                )
                continue
            if isinstance(result, dict) and result.get("ok"):
                self._auth_login_user = user_id
                return
            QtWidgets.QMessageBox.warning(
                self,
                "MFDB login failed",
                "Invalid credentials. Please try again.",
            )

    def _is_deleted(self) -> bool:
        return sip is not None and sip.isdeleted(self)

    @staticmethod
    def _is_widget_deleted(widget: QtWidgets.QWidget | None) -> bool:
        return widget is not None and sip is not None and sip.isdeleted(widget)

    def _disconnect_signal(self, signal: Any) -> None:
        try:
            signal.disconnect()
        except Exception:
            pass

    @staticmethod
    def _icon_button(icon: QtWidgets.QStyle.StandardPixmap, tooltip: str, slot: Any) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        btn.setIcon(btn.style().standardIcon(icon))
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        btn.setAutoRaise(True)
        btn.setToolTip(tooltip)
        btn.setFixedWidth(32)
        btn.setFixedHeight(28)
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _text_icon_button(text: str, icon: QtWidgets.QStyle.StandardPixmap, tooltip: str, slot: Any) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        btn.setAutoRaise(True)
        btn.setToolTip(tooltip)
        btn.clicked.connect(slot)
        return btn

    @staticmethod
    def _text_button(text: str, tooltip: str, slot: Any) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        btn.setAutoRaise(True)
        btn.setToolTip(tooltip)
        btn.clicked.connect(slot)
        return btn

    # ------------------------------------------------------------------ #
    # Table context-menu + checkbox helpers
    # ------------------------------------------------------------------ #

    def _install_table_context_menu(
        self,
        table: QtWidgets.QTableWidget,
        *,
        item_kind: str = "item",
        id_col: int = 0,
        delete_one_fn: Any = None,
        usage_fn: Any = None,
        extra_actions: list[tuple[str, Any]] | None = None,
    ) -> None:
        """Install a right-click context menu on *table*.

        The first column is marked checkable so the menu can act on
        every checked row. ``delete_one_fn(item_id) -> None`` performs
        the actual delete (we loop over checked ids). ``usage_fn(ids)``
        returns ``{id: human_description}`` for items that are
        referenced elsewhere; when non-empty, a second confirmation
        dialog with a mandatory checkbox is shown.

        Parameters
        ----------
        item_kind : str
            Human label for the items (e.g. ``"user"``).
        id_col : int
            Column index that holds the unique id.
        extra_actions : list of (label, callable)
            Optional additional menu actions appended at the bottom.
        """
        table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(
            lambda pos, t=table: self._show_table_context_menu(
                t,
                pos,
                item_kind=item_kind,
                id_col=id_col,
                delete_one_fn=delete_one_fn,
                usage_fn=usage_fn,
                extra_actions=extra_actions or [],
            )
        )
        if table not in self._checkable_tables:
            self._checkable_tables.append(table)
        self._apply_checkable_first_column(table)

    def _show_table_context_menu(
        self,
        table: QtWidgets.QTableWidget,
        pos: QtCore.QPoint,
        *,
        item_kind: str,
        id_col: int,
        delete_one_fn: Any,
        usage_fn: Any,
        extra_actions: list[tuple[str, Any]],
    ) -> None:
        menu = QtWidgets.QMenu(table)
        menu.addAction("☑ Check all", lambda: self._set_all_checks(table, True))
        menu.addAction("☐ Uncheck all", lambda: self._set_all_checks(table, False))
        menu.addAction("🔁 Invert checks", lambda: self._invert_checks(table))
        menu.addSeparator()
        menu.addAction("⬛ Select all rows", table.selectAll)
        menu.addAction("🔳 Clear selection", table.clearSelection)
        if delete_one_fn is not None:
            menu.addSeparator()
            menu.addAction(
                f"🗑 Delete checked {item_kind}s…",
                lambda: self._confirm_and_delete_checked(
                    table,
                    item_kind=item_kind,
                    id_col=id_col,
                    delete_one_fn=delete_one_fn,
                    usage_fn=usage_fn,
                ),
            )
        for label, slot in extra_actions:
            menu.addAction(label, slot)
        menu.exec(table.viewport().mapToGlobal(pos))

    @staticmethod
    def _apply_checkable_first_column(table: QtWidgets.QTableWidget) -> None:
        """Make the first column of every row in *table* user-checkable."""
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is None:
                item = QtWidgets.QTableWidgetItem("")
                table.setItem(row, 0, item)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            if item.checkState() not in (QtCore.Qt.Checked, QtCore.Qt.PartiallyChecked):
                item.setCheckState(QtCore.Qt.Unchecked)

    @staticmethod
    def _set_all_checks(table: QtWidgets.QTableWidget, checked: bool) -> None:
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None and item.flags() & QtCore.Qt.ItemIsUserCheckable:
                item.setCheckState(state)

    @staticmethod
    def _invert_checks(table: QtWidgets.QTableWidget) -> None:
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is None or not (item.flags() & QtCore.Qt.ItemIsUserCheckable):
                continue
            state = QtCore.Qt.Unchecked if item.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked
            item.setCheckState(state)

    @staticmethod
    def _checked_row_ids(table: QtWidgets.QTableWidget, id_col: int = 0) -> list[str]:
        result: list[str] = []
        for row in range(table.rowCount()):
            mark = table.item(row, 0)
            if mark is None or mark.checkState() != QtCore.Qt.Checked:
                continue
            id_item = table.item(row, id_col)
            if id_item is not None and id_item.text():
                result.append(id_item.text())
        return result

    def _confirm_and_delete_checked(
        self,
        table: QtWidgets.QTableWidget,
        *,
        item_kind: str,
        id_col: int,
        delete_one_fn: Any,
        usage_fn: Any,
    ) -> None:
        ids = self._checked_row_ids(table, id_col=id_col)
        if not ids:
            QtWidgets.QMessageBox.information(
                self,
                "Nothing checked",
                f"Check at least one {item_kind} first (use the checkboxes in the first column).",
            )
            return

        preview = "\n  • ".join(ids[:10])
        if len(ids) > 10:
            preview += f"\n  …(+{len(ids) - 10} more)"
        first = QtWidgets.QMessageBox.question(
            self,
            f"Delete {item_kind}s",
            f"You are about to delete {len(ids)} {item_kind}(s):\n\n  • {preview}\n\nProceed?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if first != QtWidgets.QMessageBox.Yes:
            return

        usages: dict[str, str] = {}
        if usage_fn is not None:
            try:
                usages = usage_fn(ids) or {}
            except Exception:
                usages = {}

        if usages:
            details = "\n".join(f"  • {k}: {v}" for k, v in usages.items())
            warn = QtWidgets.QMessageBox(self)
            warn.setIcon(QtWidgets.QMessageBox.Warning)
            warn.setWindowTitle(f"⚠️ {item_kind.capitalize()}s in use")
            warn.setText(
                f"The following {item_kind}(s) are referenced elsewhere in MFDB.\n"
                "Deleting them may cascade or fail:\n\n" + details
            )
            warn.setInformativeText(
                "You must tick the box below to confirm you understand the consequences."
            )
            confirm_cb = QtWidgets.QCheckBox(
                f"I understand — delete these {item_kind}(s) anyway"
            )
            warn.setCheckBox(confirm_cb)
            proceed_btn = warn.addButton("Delete anyway", QtWidgets.QMessageBox.DestructiveRole)
            warn.addButton(QtWidgets.QMessageBox.Cancel)
            warn.exec()
            if warn.clickedButton() is not proceed_btn or not confirm_cb.isChecked():
                self.status_label.setText("Deletion cancelled.")
                return

        failures: list[str] = []
        for item_id in ids:
            try:
                delete_one_fn(item_id)
            except Exception as exc:
                failures.append(f"{item_id}: {exc}")

        if failures:
            QtWidgets.QMessageBox.warning(
                self,
                "Some deletes failed",
                "\n".join(failures[:10])
                + ("" if len(failures) <= 10 else f"\n…(+{len(failures) - 10} more)"),
            )
        else:
            self.status_label.setText(
                f"Deleted {len(ids)} {item_kind}(s) successfully."
            )
        self.refresh()

    # ------------------------------------------------------------------ #
    # Usage-check helpers (count references across MFDB)
    # ------------------------------------------------------------------ #

    def _safe_list(self, fn) -> list[dict[str, Any]]:
        try:
            return list(fn() or [])
        except Exception:
            return []

    def _usage_check_users(self, ids: list[str]) -> dict[str, str]:
        samples = self._safe_list(self.client.list_samples)
        experiments = self._safe_list(self.client.list_experiments)
        out: dict[str, str] = {}
        for uid in ids:
            n_s = sum(
                1 for s in samples
                if s.get("measured_by_user_id") == uid or s.get("measured_by_user") == uid
            )
            n_e = sum(
                1 for e in experiments
                if e.get("measured_by_user_id") == uid or e.get("measured_by_user") == uid
            )
            parts = []
            if n_s:
                parts.append(f"{n_s} sample(s)")
            if n_e:
                parts.append(f"{n_e} experiment(s)")
            if parts:
                out[uid] = ", ".join(parts)
        return out

    def _usage_check_devices(self, ids: list[str]) -> dict[str, str]:
        samples = self._safe_list(self.client.list_samples)
        experiments = self._safe_list(self.client.list_experiments)
        out: dict[str, str] = {}
        for did in ids:
            n_s = sum(1 for s in samples if s.get("measured_by_device_id") == did)
            n_e = sum(1 for e in experiments if e.get("measured_by_device_id") == did)
            parts = []
            if n_s:
                parts.append(f"{n_s} sample(s)")
            if n_e:
                parts.append(f"{n_e} experiment(s)")
            if parts:
                out[did] = ", ".join(parts)
        return out

    def _usage_check_samples(self, ids: list[str]) -> dict[str, str]:
        experiments = self._safe_list(self.client.list_experiments)
        out: dict[str, str] = {}
        for sid in ids:
            n_e = sum(1 for e in experiments if e.get("sample_id") == sid)
            if n_e:
                out[sid] = f"{n_e} experiment(s)"
        return out

    def _usage_check_experiments(self, ids: list[str]) -> dict[str, str]:
        raw = self._safe_list(self.client.list_raw_data)
        runs = self._safe_list(self.client.list_processing_runs)
        analyses = self._safe_list(self.client.list_analysis_runs)
        out: dict[str, str] = {}
        for eid in ids:
            parts = []
            n_r = sum(1 for r in raw if r.get("experiment_id") == eid)
            n_p = sum(1 for r in runs if r.get("experiment_id") == eid)
            n_a = sum(1 for a in analyses if a.get("experiment_id") == eid)
            if n_r:
                parts.append(f"{n_r} raw")
            if n_p:
                parts.append(f"{n_p} processing run(s)")
            if n_a:
                parts.append(f"{n_a} analysis run(s)")
            if parts:
                out[eid] = ", ".join(parts)
        return out

    def _usage_check_experiment_types(self, ids: list[str]) -> dict[str, str]:
        experiments = self._safe_list(self.client.list_experiments)
        out: dict[str, str] = {}
        for tid in ids:
            n = sum(
                1 for e in experiments
                if str(e.get("type_id")) == tid or str(e.get("experiment_type_id")) == tid
            )
            if n:
                out[tid] = f"{n} experiment(s)"
        return out

    def setup_ui(self) -> None:
        self._central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self._central_widget)
        layout = QtWidgets.QVBoxLayout(self._central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QtWidgets.QLabel("<h2>mfdb-admin</h2>")
        header.setContentsMargins(4, 4, 4, 0)
        layout.addWidget(header)

        self.tabs = DockArea(self, stacked_tabs=True)
        self.tabs.setNewTabButtonVisible(False)
        self.tabs.setContextMenuEnabled(True)
        self.tabs.setContextMenuMode("basic")
        self._tabs_by_name: dict[str, QtWidgets.QWidget] = {}
        layout.addWidget(self.tabs, stretch=1)
        tab_widgets = (
            ("All items", self.all_items_tab()),
            ("Sample", self.sample_tab()),
            ("Condition", self.condition_tab()),
            ("Entities", self.entities_tab()),
            ("Probes", self.probes_tab()),
            ("Label positions", self.positions_tab()),
            ("Metadata", self.metadata_tab()),
            ("Users", self.users_tab()),
            ("Branches", self.branches_tab()),
            ("Devices", self.devices_tab()),
            ("Setups", self.setups_tab()),
            ("Raw data", self.raw_data_tab()),
            ("Processing runs", self.processing_runs_tab()),
            ("Processed products", self.processed_products_tab()),
            ("Analyses", self.analyses_tab()),
            ("Provenance", self.provenance_tab()),
            ("Provenance graph", self.provenance_graph_dock()),
            ("Experiment types", self.experiment_types_tab()),
            ("Experiments", self.experiments_tab()),
            ("Projects", self.projects_tab()),
            ("Import/Export", self.import_export_tab()),
        )
        for label, widget in tab_widgets:
            self.tabs.addTab(widget, label)
            self._tabs_by_name[label] = widget
        self.tabs.layoutChanged.connect(self._save_dock_layout)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def setup_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("⬇️ &Import...", self.import_file)
        file_menu.addAction("⬆️ &Export selected sample...", self.export_selected_sample)
        file_menu.addAction("💾 &Backup database...", self.backup_database)
        file_menu.addAction("♻️ Reset database from source...", self.reset_from_source)
        file_menu.addSeparator()
        file_menu.addAction("❌ &Close", self.close)

        settings_menu = self.menuBar().addMenu("&Settings")
        settings_menu.addAction("🗔 &Reset window layout", self.reset_window_layout)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("ℹ️ &About mfdb-admin", self.show_about)

    DEFAULT_URL = "tcp://127.0.0.1:8765"

    def setup_toolbar(self) -> None:
        toolbar = self.addToolBar("mfdb-admin")
        toolbar.setObjectName("mfdbPluginToolBar")

        toolbar.addWidget(QtWidgets.QLabel(" 🌐 "))
        # Server and port fields
        server_layout = QtWidgets.QHBoxLayout()
        server_layout.setContentsMargins(0, 0, 0, 0)
        server_layout.setSpacing(2)
        
        # Load server and port from settings
        import chisurf.core.settings as cs_settings
        mfdb_settings = cs_settings.cs_settings.get("mfdb", {})
        last_server = mfdb_settings.get("last_server", "127.0.0.1")
        last_port = mfdb_settings.get("last_port", 8765)
        
        self.server_edit = QtWidgets.QLineEdit()
        self.server_edit.setText(last_server)
        self.server_edit.setPlaceholderText("127.0.0.1")
        self.server_edit.setFixedWidth(140)
        self.server_edit.setToolTip("MFDB server host")
        self.server_edit.returnPressed.connect(self._on_login_clicked)
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(int(last_port))
        self.port_spin.setFixedWidth(60)
        self.port_spin.setToolTip("MFDB server port")
        server_layout.addWidget(self.server_edit)
        server_layout.addWidget(self.port_spin)
        
        # Container widget for the layout
        server_container = QtWidgets.QWidget()
        server_container.setLayout(server_layout)
        toolbar.addWidget(server_container)
        
        # Keep url_edit for backwards compatibility with existing code
        self.url_edit = QtWidgets.QLineEdit()
        self.url_edit.setText(self.DEFAULT_URL)

        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel(" 👤 "))
        self.username_edit = QtWidgets.QLineEdit()
        self.username_edit.setFixedWidth(120)
        self.username_edit.setToolTip("MFDB username")
        self.username_edit.returnPressed.connect(self._on_login_clicked)
        toolbar.addWidget(self.username_edit)

        toolbar.addWidget(QtWidgets.QLabel(" 🔑 "))
        self.password_edit = QtWidgets.QLineEdit()
        self.password_edit.setFixedWidth(120)
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_edit.setToolTip("MFDB password (not required if already logged in on remote)")
        self.password_edit.returnPressed.connect(self._on_login_clicked)
        toolbar.addWidget(self.password_edit)

        self.login_action = toolbar.addAction("🔐 Login", self._on_login_clicked)
        self.logout_action = toolbar.addAction("🚪 Logout", self._on_logout_clicked)
        self.login_action.setToolTip("Connect and authenticate at the URL above")
        self.logout_action.setToolTip("Drop the current MFDB session")

        toolbar.addSeparator()
        self.user_label = QtWidgets.QLabel(" (not connected) ")
        self.user_label.setStyleSheet("color: #888; padding: 0 6px;")
        toolbar.addWidget(self.user_label)

        toolbar.addSeparator()
        # Add expanding spacer to push transport actions to the right
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addSeparator()
        
        self.refresh_action = toolbar.addAction("🔄 Refresh", self.refresh)
        self._transport_actions = [self.refresh_action]
        for text, slot in (
            ("⬇️ Import", self.import_file),
            ("💾 Backup", self.backup_database),
            ("♻️ Reset from source", self.reset_from_source),
        ):
            self._transport_actions.append(toolbar.addAction(text, slot))
        self._set_transport_connected(False)
        self._update_login_actions(logged_in=False)
        
        # Prefill username with current user
        self._update_username_prefill()

    # ------------------------------------------------------------------ #
    # Login / logout via toolbar URL
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_url(text: str) -> tuple[str, str, int]:
        """Return ``(transport, host, port)`` parsed from a URL string.

        ``transport`` is ``"inprocess"`` or ``"tcp"``. Anything that isn't
        ``inprocess`` is interpreted as a ``host:port`` (``tcp://`` prefix
        optional). Falls back to ``127.0.0.1:8765`` on parse failure.
        """
        raw = (text or "").strip()
        if not raw or raw.lower() in {"inprocess", "in-process", "local"}:
            return ("inprocess", "127.0.0.1", 8765)
        stripped = raw.split("://", 1)[1] if "://" in raw else raw
        host, _, port_s = stripped.partition(":")
        host = host or "127.0.0.1"
        try:
            port = int(port_s) if port_s else 8765
        except ValueError:
            port = 8765
        return ("tcp", host, port)

    def _update_login_actions(self, *, logged_in: bool = False, connecting: bool = False) -> None:
        self.login_action.setVisible(not logged_in and not connecting)
        self.logout_action.setVisible(logged_in)
        
        # Set connection status indicator with colored dot
        if connecting:
            self.user_label.setText(" ● ")
            self.user_label.setStyleSheet("color: #ffc107; padding: 0 6px; font-weight: bold;")
            self.user_label.setToolTip("Connecting...")
        elif logged_in:
            self.user_label.setText(" ● ")
            self.user_label.setStyleSheet("color: #2e7d32; padding: 0 6px; font-weight: bold;")
            self.user_label.setToolTip("Connected")
            # Update username field with the logged-in user
            if self._auth_login_user:
                self.username_edit.setText(self._auth_login_user)
        else:
            self.user_label.setText(" ● ")
            self.user_label.setStyleSheet("color: #f44336; padding: 0 6px; font-weight: bold;")
            self.user_label.setToolTip("Not connected")
        
        # Always ensure username is prefilled
        self._update_username_prefill()

    def _update_username_prefill(self) -> None:
        """Prefill username field with current user, if available."""
        # Try to get current user from active login
        if self._auth_login_user:
            self.username_edit.setText(self._auth_login_user)
        else:
            # Fall back to default user from settings
            default_user = self._active_mfdb_user_id()
            self.username_edit.setText(default_user)

    def _on_login_clicked(self) -> None:
        """Reconnect the client at the URL in the toolbar and authenticate."""
        # Show connecting state
        self._update_login_actions(logged_in=False, connecting=True)
        QtWidgets.QApplication.processEvents()
        
        # Use server and port fields if available, otherwise fall back to URL parsing
        if hasattr(self, 'server_edit') and hasattr(self, 'port_spin'):
            host = self.server_edit.text() or "127.0.0.1"
            port = self.port_spin.value()
            transport = "tcp"
        else:
            transport, host, port = self._parse_url(self.url_edit.text())
        
        try:
            if transport == "inprocess":
                self.client = MFDBClient(inprocess=True)
            else:
                self.client = MFDBClient(host=host, cmd_port=port, pub_port=port + 1)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Connection failed", str(exc))
            self._update_login_actions(logged_in=False, connecting=False)
            return
        
        try:
            self._verify_admin_access()
        except PermissionError as exc:
            QtWidgets.QMessageBox.critical(self, "Access denied", str(exc))
            self._update_login_actions(logged_in=False, connecting=False)
            return
        
        # Use username from toolbar field, or fall back to active user
        username = self.username_edit.text() or self._active_mfdb_user_id()
        password = self.password_edit.text()
        
        self._auth_login_user = None
        self._ensure_authenticated(username=username, password=password)
        self._update_login_actions(logged_in=bool(getattr(self.client, "token", None)), connecting=False)
        self.refresh()

    def _on_logout_clicked(self) -> None:
        """Log out of MFDB and clear the session token."""
        try:
            self.client.logout()
        except Exception:
            pass
        try:
            self.client.token = None
        except Exception:
            pass
        self._auth_login_user = None
        self._update_login_actions(logged_in=False)
        self.status_label.setText("Logged out.")

    # Backwards-compat alias used in earlier revisions
    def _relogin(self) -> None:
        self._on_logout_clicked()
        self._on_login_clicked()

    def setup_status_bar(self) -> None:
        self.statusBar().addPermanentWidget(self.status_label, stretch=1)
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------ #
    # All items dock
    # ------------------------------------------------------------------ #

    @property
    def _all_items_sources(self) -> list[dict[str, Any]]:
        """Definitions describing every MFDB type surfaced in the unified dock.

        Each entry maps a logical type to the client listing call, the dict
        keys used to render id/label, the target tab the user is jumped to,
        and the table attribute + id column used for the selection.
        """
        return [
            {"type": "sample", "list": lambda: self.client.list_samples(),
             "id": "sample_id", "label": "description",
             "tab": "Sample", "table": None, "id_col": 1},
            {"type": "experiment", "list": lambda: self.client.list_experiments(),
             "id": "experiment_id", "label": "sample_id",
             "tab": "Experiments", "table": "experiments_table", "id_col": 0},
            {"type": "user", "list": lambda: self.client.list_users(),
             "id": "user_id", "label": "display_name",
             "tab": "Users", "table": "users_table", "id_col": 0},
            {"type": "device", "list": lambda: self.client.list_devices(),
             "id": "device_id", "label": "name",
             "tab": "Devices", "table": "devices_table", "id_col": 0},
            {"type": "probe", "list": lambda: self.client.list_probes(),
             "id": "probe_id", "label": "chromophore_name",
             "tab": "Probes", "table": "probes_table", "id_col": 0},
            {"type": "branch", "list": lambda: self.client.list_branches(),
             "id": "branch_uuid", "label": "name",
             "tab": "Branches", "table": "branches_table", "id_col": 1},
            {"type": "experiment_type",
             "list": lambda: self.client.list_experiment_types(),
             "id": "type_id", "label": "name",
             "tab": "Experiment types", "table": "experiment_types_table", "id_col": 0},
            {"type": "setup",
             "list": lambda: self.client._call("mfdb.setups.list").get("setups", []),
             "id": "setup_id", "label": "name",
             "tab": "Setups", "table": "setups_table", "id_col": 0},
            {"type": "raw_data", "list": lambda: self.client.list_raw_data(),
             "id": "raw_data_id", "label": "data_type",
             "tab": "Raw data", "table": "raw_data_table", "id_col": 0},
            {"type": "processing_run",
             "list": lambda: self.client.list_processing_runs(),
             "id": "processing_id", "label": "processing_type",
             "tab": "Processing runs", "table": "processing_runs_table",
             "id_col": 0},
            {"type": "processed_data",
             "list": lambda: self.client.list_processed_data(),
             "id": "processed_data_id", "label": "product_type",
             "tab": "Processed products", "table": "processed_products_table",
             "id_col": 0},
            {"type": "analysis",
             "list": lambda: self.client.list_analysis_runs(),
             "id": "analysis_id", "label": "model_name",
             "tab": "Analyses", "table": "analyses_table", "id_col": 0},
            {"type": "project", "list": lambda: self.client.list_projects(),
             "id": "analysis_id", "label": "model_name",
             "tab": "Projects", "table": "projects_table", "id_col": 0},
        ]

    def all_items_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(4)
        controls.addWidget(QtWidgets.QLabel("Type:"))
        self.all_items_type_combo = QtWidgets.QComboBox()
        self.all_items_type_combo.addItem("all", "all")
        for source in self._all_items_sources:
            self.all_items_type_combo.addItem(source["type"], source["type"])
        self.all_items_type_combo.currentIndexChanged.connect(
            lambda _idx: self._filter_all_items_table()
        )
        controls.addWidget(self.all_items_type_combo)

        self.all_items_search_edit = QtWidgets.QLineEdit()
        self.all_items_search_edit.setPlaceholderText("Filter by id, label, or substring...")
        self.all_items_search_edit.textChanged.connect(self._filter_all_items_table)
        controls.addWidget(self.all_items_search_edit, stretch=1)

        self.all_items_count_label = QtWidgets.QLabel("0 / 0")
        controls.addWidget(self.all_items_count_label)
        refresh_btn = self._text_icon_button(
            "🔄 Refresh",
            QtWidgets.QStyle.SP_BrowserReload,
            "Reload all MFDB items",
            self._populate_all_items,
        )
        controls.addWidget(refresh_btn)
        layout.addLayout(controls)

        self.all_items_table = QtWidgets.QTableWidget(0, 4)
        self.all_items_table.setHorizontalHeaderLabels(["type", "id", "label", "details"])
        self.all_items_table.horizontalHeader().setStretchLastSection(True)
        self.all_items_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.all_items_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.all_items_table.setAlternatingRowColors(True)
        self.all_items_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.all_items_table.itemDoubleClicked.connect(self._on_all_item_activated)
        self.all_items_table.itemActivated.connect(self._on_all_item_activated)
        layout.addWidget(self.all_items_table, stretch=1)

        hint = QtWidgets.QLabel(
            "Double-click any row to jump to the matching dock and select the item."
        )
        hint.setStyleSheet("color: #777777;")
        layout.addWidget(hint)
        return widget

    def _populate_all_items(self) -> None:
        if not hasattr(self, "all_items_table"):
            return
        if self._is_widget_deleted(self.all_items_table):
            return
        self.all_items_table.setRowCount(0)
        for source in self._all_items_sources:
            try:
                items = source["list"]() or []
            except Exception:
                items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                row = self.all_items_table.rowCount()
                self.all_items_table.insertRow(row)
                payload = {
                    "type": source["type"],
                    "tab": source["tab"],
                    "table": source["table"],
                    "id_col": source["id_col"],
                    "id": str(item.get(source["id"], "") or ""),
                    "data": item,
                }
                type_item = QtWidgets.QTableWidgetItem(source["type"])
                type_item.setData(QtCore.Qt.UserRole, payload)
                self.all_items_table.setItem(row, 0, type_item)
                self.all_items_table.setItem(row, 1, QtWidgets.QTableWidgetItem(payload["id"]))
                label_value = str(item.get(source["label"], "") or "")
                self.all_items_table.setItem(row, 2, QtWidgets.QTableWidgetItem(label_value))
                details = self._summarise_item(source["type"], item)
                self.all_items_table.setItem(row, 3, QtWidgets.QTableWidgetItem(details))
        self._filter_all_items_table()

    @staticmethod
    def _summarise_item(item_type: str, item: dict[str, Any]) -> str:
        """Return a short, single-line description for the all-items table."""
        keys: tuple[str, ...]
        if item_type == "sample":
            keys = ("project_id", "sample_uuid")
        elif item_type == "experiment":
            keys = ("experiment_type", "project_id", "status")
        elif item_type == "user":
            keys = ("email", "affiliation", "role")
        elif item_type == "device":
            keys = ("device_type", "model", "serial_number")
        elif item_type == "probe":
            keys = ("category", "probe_origin")
        elif item_type == "branch":
            keys = ("parent_branch_uuid", "head_operation_id")
        elif item_type == "setup":
            keys = ("instrument_type",)
        elif item_type == "experiment_type":
            keys = ("category",)
        elif item_type == "raw_data":
            keys = ("storage_mode", "experiment_id")
        elif item_type == "processing_run":
            keys = ("status", "experiment_id")
        elif item_type == "processed_data":
            keys = ("product_type", "processing_id")
        elif item_type == "analysis":
            keys = ("analysis_type", "experiment_id")
        elif item_type == "project":
            keys = ("experiment_id", "created_at")
        else:
            keys = ()
        parts = [f"{k}={item.get(k)}" for k in keys if item.get(k)]
        return ", ".join(parts)

    def _filter_all_items_table(self) -> None:
        if not hasattr(self, "all_items_table"):
            return
        query = self.all_items_search_edit.text().strip().lower()
        type_filter = self.all_items_type_combo.currentData() or "all"
        visible = 0
        for row in range(self.all_items_table.rowCount()):
            type_item = self.all_items_table.item(row, 0)
            row_type = type_item.text() if type_item else ""
            if type_filter != "all" and row_type != type_filter:
                self.all_items_table.setRowHidden(row, True)
                continue
            if query:
                hit = False
                for col in range(self.all_items_table.columnCount()):
                    cell = self.all_items_table.item(row, col)
                    if cell and query in cell.text().lower():
                        hit = True
                        break
                self.all_items_table.setRowHidden(row, not hit)
                if hit:
                    visible += 1
            else:
                self.all_items_table.setRowHidden(row, False)
                visible += 1
        self.all_items_count_label.setText(
            f"{visible} / {self.all_items_table.rowCount()}"
        )

    def _on_all_item_activated(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item is None:
            return
        type_item = self.all_items_table.item(item.row(), 0)
        if type_item is None:
            return
        payload = type_item.data(QtCore.Qt.UserRole)
        if not isinstance(payload, dict):
            return
        self._jump_to_item(payload)

    def _jump_to_item(self, payload: dict[str, Any]) -> None:
        tab_name = payload.get("tab", "")
        tab_widget = self._tabs_by_name.get(tab_name)
        if tab_widget is not None:
            try:
                index = self.tabs.indexOf(tab_widget)
            except Exception:
                index = -1
            if index >= 0:
                if hasattr(self.tabs, "isTabVisible") and hasattr(self.tabs, "showTab"):
                    try:
                        if not self.tabs.isTabVisible(index):
                            self.tabs.showTab(index)
                    except Exception:
                        pass
                try:
                    self.tabs.setCurrentWidget(tab_widget)
                except Exception:
                    pass
        # Try to select in table first
        table_attr = payload.get("table")
        table = getattr(self, table_attr, None) if table_attr else None
        if table is not None and not self._is_widget_deleted(table):
            self._select_row_by_id(table, int(payload.get("id_col", 0)), payload.get("id", ""))
        # If no table or table not found, try to call a load method
        elif tab_name:
            item_type = payload.get("type", "")
            item_id = payload.get("id", "")
            load_method = getattr(self, f"load_{item_type}", None)
            if load_method and item_id:
                load_method(item_id)

    @staticmethod
    def _select_row_by_id(table: QtWidgets.QTableWidget, id_col: int, value: str) -> None:
        if table is None or not value:
            return
        for row in range(table.rowCount()):
            cell = table.item(row, id_col)
            if cell and cell.text() == value:
                table.selectRow(row)
                table.scrollToItem(cell, QtWidgets.QAbstractItemView.PositionAtCenter)
                return

    def reset_window_layout(self) -> None:
        settings = self._dock_settings()
        settings.remove("dock_layout")
        settings.remove("geometry")
        settings.remove("state")
        self.resize(1100, 760)
        self._restore_dock_layout()

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About mfdb-admin",
            "mfdb-admin\n\nMultiparametric Fluorescence Database\n\nBrowse, edit, import, and export fluorescence measurements, samples, setups, and analysis runs.",
        )

    def sample_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.sample_id_edit = QtWidgets.QLineEdit()
        self.sample_id_edit.setPlaceholderText("Type sample id (autocomplete searches existing)")
        self.uuid_edit = QtWidgets.QLineEdit()
        self.uuid_edit.setPlaceholderText("Auto-generated if left empty")
        self.description_edit = QtWidgets.QLineEdit()
        self.details_edit = QtWidgets.QPlainTextEdit()
        self.details_edit.setMinimumHeight(60)
        self.num_probes_spin = QtWidgets.QSpinBox()
        self.num_probes_spin.setRange(0, 1000)
        self.solvent_edit = QtWidgets.QComboBox()
        self.solvent_edit.addItems(["liquid", "vitrified", "other"])
        self.condition_id_edit = QtWidgets.QLineEdit()
        self.assembly_id_edit = QtWidgets.QLineEdit()
        self.project_edit = QtWidgets.QLineEdit()
        self.measured_by_combo = QtWidgets.QComboBox()
        self.measured_device_combo = QtWidgets.QComboBox()
        self.measured_at_edit = QtWidgets.QLineEdit()

        btn_row = QtWidgets.QHBoxLayout()
        new_btn = self._text_icon_button("🧪 New", QtWidgets.QStyle.SP_FileDialogNewFolder, "Create new sample", self.new_sample)
        save_btn = self._text_icon_button("💾 Save", QtWidgets.QStyle.SP_DialogSaveButton, "Save sample", self.save_sample)
        delete_btn = self._text_icon_button("🗑 Delete", QtWidgets.QStyle.SP_TrashIcon, "Delete sample", self.delete_sample)
        clear_btn = self._text_icon_button("🧹 Clear", QtWidgets.QStyle.SP_DialogResetButton, "Clear form", self.clear_form)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()

        layout.addRow("Sample id", self.sample_id_edit)
        layout.addRow("UUID", self.uuid_edit)
        layout.addRow("Description", self.description_edit)
        layout.addRow("Details", self.details_edit)
        layout.addRow("Number of probes", self.num_probes_spin)
        layout.addRow("Solvent phase", self.solvent_edit)
        layout.addRow("Condition id", self.condition_id_edit)
        layout.addRow("Entity assembly id", self.assembly_id_edit)
        layout.addRow("Project id", self.project_edit)
        layout.addRow("Measured by", self.measured_by_combo)
        layout.addRow("Device", self.measured_device_combo)
        layout.addRow("Measured at", self.measured_at_edit)
        layout.addRow("", btn_row)
        self._setup_sample_id_completer()
        self.sample_id_edit.textChanged.connect(self._auto_generate_uuid)
        self.sample_id_edit.editingFinished.connect(self._on_sample_id_edited)
        self.condition_id_edit.editingFinished.connect(self._auto_fill_condition)
        return widget

    def _auto_generate_uuid(self) -> None:
        if self._loading:
            return
        if self.sample_id_edit.text().strip() and not self.uuid_edit.text().strip():
            self.uuid_edit.setText(str(uuid.uuid4()))

    def _auto_fill_condition(self) -> None:
        cid = self.condition_id_edit.text().strip()
        if not cid or self._loading:
            return
        try:
            sample = self.client._call("sample_database.samples.get", {"sample_id": cid})
            cond = (sample.get("sample") or {}).get("condition")
            if cond:
                self.condition_id_field.setText(cond.get("condition_id", ""))
                self.ph_spin.setValue(float(cond.get("ph") or 0))
                self.temperature_spin.setValue(float(cond.get("temperature") or 0))
                self.ionic_spin.setValue(float(cond.get("ionic_strength") or 0))
                self.buffer_edit.setText(cond.get("buffer_composition", ""))
                self.condition_details_edit.setPlainText(cond.get("details", ""))
        except Exception:
            pass

    def _setup_sample_id_completer(self) -> None:
        try:
            samples = self.client.list_samples()
            ids = [s.get("sample_id", "") for s in samples if s.get("sample_id")]
            descs = [s.get("description", "") for s in samples if s.get("description")]
            items = list(dict.fromkeys(ids + descs))
            completer = QtWidgets.QCompleter(items, self)
            completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            completer.setFilterMode(QtCore.Qt.MatchContains)
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            self.sample_id_edit.setCompleter(completer)
        except Exception:
            pass

    def _on_sample_id_edited(self) -> None:
        sample_id = self.sample_id_edit.text().strip()
        if not sample_id:
            return
        try:
            sample = self.client.get_sample(sample_id)
            if sample:
                self.load_sample(sample_id)
        except Exception:
            pass

    def condition_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.condition_id_field = QtWidgets.QLineEdit()
        self.condition_id_field.setPlaceholderText("Type condition id to auto-fill from DB")
        self.ph_spin = QtWidgets.QDoubleSpinBox()
        self.ph_spin.setRange(-1.0, 14.0)
        self.ph_spin.setSpecialValueText("auto")
        self.temperature_spin = QtWidgets.QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 400.0)
        self.temperature_spin.setSpecialValueText("auto")
        self.ionic_spin = QtWidgets.QDoubleSpinBox()
        self.ionic_spin.setRange(0.0, 10.0)
        self.ionic_spin.setSpecialValueText("auto")
        self.buffer_edit = QtWidgets.QLineEdit()
        self.condition_details_edit = QtWidgets.QPlainTextEdit()
        self.condition_details_edit.setMinimumHeight(60)

        cond_btn_row = QtWidgets.QHBoxLayout()
        cond_save_btn = self._text_icon_button("💾 Save", QtWidgets.QStyle.SP_DialogSaveButton, "Save condition", self.save_condition)
        cond_clear_btn = self._text_icon_button("🧹 Clear", QtWidgets.QStyle.SP_DialogResetButton, "Clear condition", self.clear_condition_form)
        cond_btn_row.addWidget(cond_save_btn)
        cond_btn_row.addWidget(cond_clear_btn)
        cond_btn_row.addStretch()

        layout.addRow("Condition id", self.condition_id_field)
        layout.addRow("pH", self.ph_spin)
        layout.addRow("Temperature [K]", self.temperature_spin)
        layout.addRow("Ionic strength [M]", self.ionic_spin)
        layout.addRow("Buffer", self.buffer_edit)
        layout.addRow("Details", self.condition_details_edit)
        layout.addRow("", cond_btn_row)

        self.condition_id_field.editingFinished.connect(self._auto_fill_condition_details)
        return widget

    def save_condition(self) -> None:
        cid = self.condition_id_field.text().strip()
        if not cid:
            self.status_label.setText("Condition id is required")
            return
        try:
            condition = {
                "condition_id": cid,
                "ph": None if self.ph_spin.value() == 0 else self.ph_spin.value(),
                "temperature": None if self.temperature_spin.value() == 0 else self.temperature_spin.value(),
                "ionic_strength": None if self.ionic_spin.value() == 0 else self.ionic_spin.value(),
                "buffer_composition": self.buffer_edit.text().strip() or None,
                "details": self.condition_details_edit.toPlainText().strip() or None,
            }
            self.client.save_sample_condition(condition)
            self.status_label.setText(f"Condition '{cid}' saved")
        except Exception as exc:
            self.status_label.setText(f"Failed to save condition: {exc}")

    def clear_condition_form(self) -> None:
        self.condition_id_field.clear()
        self.ph_spin.setValue(0)
        self.temperature_spin.setValue(0)
        self.ionic_spin.setValue(0)
        self.buffer_edit.clear()
        self.condition_details_edit.clear()

    def _auto_fill_condition_details(self) -> None:
        cid = self.condition_id_field.text().strip()
        if not cid or self._loading:
            return
        try:
            row = self.client.get_sample_condition(cid)
            if row:
                self.ph_spin.setValue(float(row.get("ph") or 0))
                self.temperature_spin.setValue(float(row.get("temperature") or 0))
                self.ionic_spin.setValue(float(row.get("ionic_strength") or 0))
                self.buffer_edit.setText(row.get("buffer_composition", ""))
                self.condition_details_edit.setPlainText(row.get("details", ""))
                self.status_label.setText(f"Auto-filled condition '{cid}'")
        except Exception:
            pass

    def entities_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.entities_table = QtWidgets.QTableWidget(0, 5)
        self.entities_table.setHorizontalHeaderLabels(
            ["entity id", "type", "description", "common name", "sequence"]
        )
        self.entities_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.entities_table, stretch=1)
        return widget

    def probes_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.probes_table = QtWidgets.QTableWidget(0, 8)
        self.probes_table.setHorizontalHeaderLabels(
            ["id", "name", "category", "origin", "link", "abs", "em", "QY"]
        )
        self.probes_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.probes_table, stretch=1)
        return widget

    def positions_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.positions_table = QtWidgets.QTableWidget(0, 9)
        self.positions_table.setHorizontalHeaderLabels(
            [
                "sample_probe_id",
                "sample",
                "probe_id",
                "probe",
                "entity",
                "chain",
                "residue",
                "type",
                "description",
            ]
        )
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.positions_table, stretch=1)
        return widget

    def import_export_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        file_row = QtWidgets.QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(2)
        self.file_edit = QtWidgets.QLineEdit()
        browse_button = self._text_icon_button("📁 Browse", QtWidgets.QStyle.SP_DirOpenIcon, "Browse for file", self.browse_import_file)
        file_row.addWidget(self.file_edit)
        file_row.addWidget(browse_button)
        layout.addLayout(file_row)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        import_button = self._text_icon_button("⬇️ Import file", QtWidgets.QStyle.SP_ArrowDown, "Import file into database", self.import_file)
        export_button = self._text_icon_button("⬆️ Export selected sample", QtWidgets.QStyle.SP_ArrowUp, "Export selected sample to FLR CIF", self.export_selected_sample)
        export_table_button = self._text_icon_button("📊 Export table CSV/XLSX", QtWidgets.QStyle.SP_FileIcon, "Export sample table", self.export_table)
        buttons.addWidget(import_button)
        buttons.addWidget(export_button)
        buttons.addWidget(export_table_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.preview_edit = QtWidgets.QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        layout.addWidget(self.preview_edit)
        return widget

    def metadata_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.metadata_sample_label = QtWidgets.QLabel("No sample selected")
        self.metadata_sample_label.setStyleSheet("color: #555555;")
        self.metadata_sample_label.setContentsMargins(4, 4, 4, 0)
        layout.addWidget(self.metadata_sample_label)

        self.metadata_editor = MetadataEditor(columns=3)
        layout.addWidget(self.metadata_editor, stretch=1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_metadata_btn = self._text_icon_button(
            "💾 Save metadata",
            QtWidgets.QStyle.SP_DialogSaveButton,
            "Persist metadata key/value rows for the selected sample",
            self.save_sample_metadata,
        )
        add_metadata_btn = self._text_icon_button(
            "➕ Add row",
            QtWidgets.QStyle.SP_FileDialogNewFolder,
            "Add an empty metadata row",
            self.add_metadata_row,
        )
        delete_metadata_btn = self._text_icon_button(
            "🗑 Delete row",
            QtWidgets.QStyle.SP_TrashIcon,
            "Delete the selected metadata row",
            self.delete_metadata_row,
        )
        buttons.addWidget(save_metadata_btn)
        buttons.addWidget(add_metadata_btn)
        buttons.addWidget(delete_metadata_btn)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def save_sample_metadata(self) -> None:
        """Persist the current metadata editor rows for the loaded sample."""
        sample_id = (self.current_sample_id or self.sample_id_edit.text()).strip()
        if not sample_id:
            self.status_label.setText("Select or save a sample before saving metadata")
            QtWidgets.QMessageBox.warning(
                self,
                "No sample selected",
                "Select or create a sample first; metadata is stored per sample.",
            )
            return
        try:
            self.client.save_sample_key_values(sample_id, self.metadata_editor.get_data())
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save metadata failed", str(exc))
            return
        self.status_label.setText(f"Metadata saved for sample '{sample_id}'")

    USER_ROLE_OPTIONS = [
        "",
        "Generic",
        "Principal Investigator",
        "Postdoc",
        "PhD Student",
        "Master Student",
        "Bachelor Student",
        "Technician",
        "Industry Professional",
        "Scientist",
        "Manager",
        "Other",
    ]

    USERS_TABLE_COLUMNS = [
        "id",
        "display name",
        "email",
        "role",
        "department",
        "affiliation",
        "admin",
        "password",
    ]

    def users_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.users_table = QtWidgets.QTableWidget(0, len(self.USERS_TABLE_COLUMNS))
        self.users_table.setHorizontalHeaderLabels(list(self.USERS_TABLE_COLUMNS))
        self.users_table.horizontalHeader().setStretchLastSection(True)
        self.users_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.users_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.users_table.setAlternatingRowColors(True)
        self.users_table.itemSelectionChanged.connect(self.load_user)
        self._install_table_context_menu(
            self.users_table,
            item_kind="user",
            id_col=0,
            delete_one_fn=lambda uid: self.client.delete_user(uid),
            usage_fn=self._usage_check_users,
        )
        layout.addWidget(self.users_table, stretch=1)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)

        self.user_uuid_edit = QtWidgets.QLineEdit()
        self.user_uuid_edit.setPlaceholderText("Auto-generated if left empty")
        self.user_id_edit = QtWidgets.QLineEdit()
        self.user_display_edit = QtWidgets.QLineEdit()
        self.user_email_edit = QtWidgets.QLineEdit()
        self.user_role_combo = QtWidgets.QComboBox()
        self.user_role_combo.setEditable(True)
        self.user_role_combo.addItems(self.USER_ROLE_OPTIONS)
        self.user_affiliation_edit = QtWidgets.QLineEdit()
        self.user_department_edit = QtWidgets.QLineEdit()
        self.user_phone_edit = QtWidgets.QLineEdit()
        self.user_website_edit = QtWidgets.QLineEdit()
        self.user_address_edit = QtWidgets.QPlainTextEdit()
        self.user_address_edit.setMinimumHeight(60)
        self.user_is_admin_check = QtWidgets.QCheckBox("Administrator")
        self.user_passwordless_check = QtWidgets.QCheckBox("Allow passwordless login")
        self.user_active_branch_edit = QtWidgets.QLineEdit()
        self.user_active_branch_edit.setReadOnly(True)
        self.user_has_password_label = QtWidgets.QLabel("—")
        self.user_created_label = QtWidgets.QLabel("—")
        self.user_updated_label = QtWidgets.QLabel("—")
        self.user_details_edit = QtWidgets.QPlainTextEdit()
        self.user_details_edit.setMinimumHeight(60)

        form.addRow("User UUID", self.user_uuid_edit)
        form.addRow("User id", self.user_id_edit)
        form.addRow("Display name", self.user_display_edit)
        form.addRow("Email", self.user_email_edit)
        form.addRow("Role", self.user_role_combo)
        form.addRow("Affiliation", self.user_affiliation_edit)
        form.addRow("Department", self.user_department_edit)
        form.addRow("Phone", self.user_phone_edit)
        form.addRow("Website", self.user_website_edit)
        form.addRow("Address", self.user_address_edit)
        form.addRow("Flags", self.user_is_admin_check)
        form.addRow("", self.user_passwordless_check)
        form.addRow("Active branch", self.user_active_branch_edit)
        form.addRow("Password set", self.user_has_password_label)
        form.addRow("Created at", self.user_created_label)
        form.addRow("Updated at", self.user_updated_label)
        form.addRow("Details", self.user_details_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        new_user_button = self._text_icon_button(
            "👤 New user",
            QtWidgets.QStyle.SP_FileDialogNewFolder,
            "Prepare the form for a new user (auto-generates UUID)",
            self.new_user,
        )
        save_user_button = self._text_icon_button(
            "💾 Save user", QtWidgets.QStyle.SP_DialogSaveButton, "Save user", self.save_user
        )
        change_password_button = self._text_icon_button(
            "🔑 Password", QtWidgets.QStyle.SP_DialogApplyButton, "Change password", self.change_user_password
        )
        delete_user_button = self._text_icon_button(
            "🗑 Delete", QtWidgets.QStyle.SP_TrashIcon, "Delete user", self.delete_user
        )
        buttons.addWidget(new_user_button)
        buttons.addWidget(save_user_button)
        buttons.addWidget(change_password_button)
        buttons.addWidget(delete_user_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def devices_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.devices_table = QtWidgets.QTableWidget(0, 8)
        self.devices_table.setHorizontalHeaderLabels(
            ["id", "name", "type", "model", "serial", "location", "owner", "details"]
        )
        self.devices_table.horizontalHeader().setStretchLastSection(True)
        self.devices_table.itemSelectionChanged.connect(self.load_device)
        self._install_table_context_menu(
            self.devices_table,
            item_kind="device",
            id_col=0,
            delete_one_fn=lambda did: self.client.delete_device(did),
            usage_fn=self._usage_check_devices,
        )
        layout.addWidget(self.devices_table, stretch=1)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.device_id_edit = QtWidgets.QLineEdit()
        self.device_name_edit = QtWidgets.QLineEdit()
        self.device_type_edit = QtWidgets.QLineEdit()
        self.device_model_edit = QtWidgets.QLineEdit()
        self.device_serial_edit = QtWidgets.QLineEdit()
        self.device_location_edit = QtWidgets.QLineEdit()
        self.device_owner_edit = QtWidgets.QLineEdit()
        self.device_details_edit = QtWidgets.QPlainTextEdit()
        self.device_details_edit.setMinimumHeight(60)
        form.addRow("Device id", self.device_id_edit)
        form.addRow("Name", self.device_name_edit)
        form.addRow("Type", self.device_type_edit)
        form.addRow("Model", self.device_model_edit)
        form.addRow("Serial", self.device_serial_edit)
        form.addRow("Location", self.device_location_edit)
        form.addRow("Owner", self.device_owner_edit)
        form.addRow("Details", self.device_details_edit)
        layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_device_button = self._text_icon_button("💾 Save", QtWidgets.QStyle.SP_DialogSaveButton, "Save device", self.save_device)
        delete_device_button = self._text_icon_button("🗑 Delete", QtWidgets.QStyle.SP_TrashIcon, "Delete device", self.delete_device)
        buttons.addWidget(save_device_button)
        buttons.addWidget(delete_device_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def branches_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.branches_table = QtWidgets.QTableWidget(0, 5)
        self.branches_table.setHorizontalHeaderLabels(
            ["name", "uuid", "parent uuid", "head op", "description"]
        )
        self.branches_table.horizontalHeader().setStretchLastSection(True)
        self.branches_table.itemSelectionChanged.connect(self.load_branch)
        self._install_table_context_menu(
            self.branches_table,
            item_kind="branch",
            id_col=1,  # branch_uuid lives in col 1
            delete_one_fn=lambda uuid: self.client.delete_branch(uuid),
        )
        layout.addWidget(self.branches_table, stretch=1)

        user_group = QtWidgets.QGroupBox("User's Active Branch")
        user_group_layout = QtWidgets.QFormLayout(user_group)
        user_group_layout.setContentsMargins(4, 4, 4, 4)
        user_group_layout.setSpacing(2)
        self.branch_user_combo = QtWidgets.QComboBox()
        self.branch_user_combo.currentIndexChanged.connect(self.on_branch_user_changed)
        self.active_branch_label = QtWidgets.QLabel("Unknown")
        user_group_layout.addRow("Select User", self.branch_user_combo)
        user_group_layout.addRow("Active Branch", self.active_branch_label)

        user_buttons = QtWidgets.QHBoxLayout()
        self.set_active_branch_button = self._text_icon_button(
            "🔀 Switch Active Branch",
            QtWidgets.QStyle.SP_BrowserReload,
            "Switch the selected user to the chosen branch",
            self.switch_active_branch,
        )
        user_buttons.addWidget(self.set_active_branch_button)
        self.jump_branch_button = self._text_icon_button(
            "⏱ Create Time Branch",
            QtWidgets.QStyle.SP_FileDialogNewFolder,
            "Create and activate a branch at the requested operation",
            self.create_time_branch,
        )
        user_buttons.addWidget(self.jump_branch_button)
        user_buttons.addStretch()
        user_group_layout.addRow("", user_buttons)
        layout.addWidget(user_group)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.branch_uuid_edit = QtWidgets.QLineEdit()
        self.branch_uuid_edit.setPlaceholderText("Auto-generated UUID")
        self.branch_name_edit = QtWidgets.QLineEdit()
        self.branch_parent_uuid_edit = QtWidgets.QLineEdit()
        self.branch_head_op_edit = QtWidgets.QLineEdit()
        self.branch_description_edit = QtWidgets.QPlainTextEdit()
        self.branch_description_edit.setMinimumHeight(60)
        form.addRow("Branch UUID", self.branch_uuid_edit)
        form.addRow("Branch Name", self.branch_name_edit)
        form.addRow("Parent Branch UUID", self.branch_parent_uuid_edit)
        form.addRow("Head Operation ID", self.branch_head_op_edit)
        form.addRow("Description", self.branch_description_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_branch_button = self._text_icon_button("💾 Save", QtWidgets.QStyle.SP_DialogSaveButton, "Save/Create branch", self.save_branch)
        fork_branch_button = self._text_icon_button("🍴 Fork", QtWidgets.QStyle.SP_FileDialogNewFolder, "Prefill branch from selected head", self.prefill_branch_fork)
        delete_branch_button = self._text_icon_button("🗑 Delete", QtWidgets.QStyle.SP_TrashIcon, "Delete branch", self.delete_branch)
        buttons.addWidget(save_branch_button)
        buttons.addWidget(fork_branch_button)
        buttons.addWidget(delete_branch_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def experiment_types_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.experiment_types_table = QtWidgets.QTableWidget(0, 5)
        self.experiment_types_table.setHorizontalHeaderLabels(
            ["id", "name", "category", "description", "details"]
        )
        self.experiment_types_table.horizontalHeader().setStretchLastSection(True)
        self.experiment_types_table.itemSelectionChanged.connect(self.load_experiment_type)
        self._install_table_context_menu(
            self.experiment_types_table,
            item_kind="experiment type",
            id_col=0,
            delete_one_fn=lambda tid: self.client.delete_experiment_type(int(tid)),
            usage_fn=self._usage_check_experiment_types,
        )
        layout.addWidget(self.experiment_types_table, stretch=1)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.experiment_type_id_edit = QtWidgets.QLineEdit()
        self.experiment_type_name_edit = QtWidgets.QLineEdit()
        self.experiment_type_category_edit = QtWidgets.QLineEdit()
        self.experiment_type_description_edit = QtWidgets.QLineEdit()
        self.experiment_type_details_edit = QtWidgets.QPlainTextEdit()
        self.experiment_type_details_edit.setMinimumHeight(60)
        form.addRow("Type id", self.experiment_type_id_edit)
        form.addRow("Name", self.experiment_type_name_edit)
        form.addRow("Category", self.experiment_type_category_edit)
        form.addRow("Description", self.experiment_type_description_edit)
        form.addRow("Details", self.experiment_type_details_edit)
        layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_button = self._text_icon_button("💾 Save", QtWidgets.QStyle.SP_DialogSaveButton, "Save experiment type", self.save_experiment_type)
        delete_button = self._text_icon_button("🗑 Delete", QtWidgets.QStyle.SP_TrashIcon, "Delete experiment type", self.delete_experiment_type)
        buttons.addWidget(save_button)
        buttons.addWidget(delete_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def experiments_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.experiments_table = QtWidgets.QTableWidget(0, 8)
        self.experiments_table.setHorizontalHeaderLabels(
            ["experiment id", "type", "sample", "project", "user", "device", "started", "status"]
        )
        self.experiments_table.horizontalHeader().setStretchLastSection(True)
        self.experiments_table.itemSelectionChanged.connect(self.load_experiment)
        self._install_table_context_menu(
            self.experiments_table,
            item_kind="experiment",
            id_col=0,
            delete_one_fn=lambda eid: self.client.delete_experiment(eid),
            usage_fn=self._usage_check_experiments,
        )
        layout.addWidget(self.experiments_table, stretch=2)
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.experiment_id_edit = QtWidgets.QLineEdit()
        self.experiment_type_combo = QtWidgets.QComboBox()
        self.experiment_sample_combo = QtWidgets.QComboBox()
        self.experiment_project_edit = QtWidgets.QLineEdit()
        self.experiment_user_combo = QtWidgets.QComboBox()
        self.experiment_device_combo = QtWidgets.QComboBox()
        self.experiment_started_edit = QtWidgets.QLineEdit()
        self.experiment_ended_edit = QtWidgets.QLineEdit()
        self.experiment_status_edit = QtWidgets.QLineEdit()
        self.experiment_details_edit = QtWidgets.QPlainTextEdit()
        self.experiment_details_edit.setMinimumHeight(60)
        form.addRow("Experiment id", self.experiment_id_edit)
        form.addRow("Type", self.experiment_type_combo)
        form.addRow("Sample", self.experiment_sample_combo)
        form.addRow("Project", self.experiment_project_edit)
        form.addRow("User", self.experiment_user_combo)
        form.addRow("Device", self.experiment_device_combo)
        form.addRow("Started", self.experiment_started_edit)
        form.addRow("Ended", self.experiment_ended_edit)
        form.addRow("Status", self.experiment_status_edit)
        form.addRow("Details", self.experiment_details_edit)
        layout.addLayout(form)
        data_header = QtWidgets.QLabel("Experiment data / links")
        layout.addWidget(data_header)
        self.experiment_data_table = QtWidgets.QTableWidget(0, 8)
        self.experiment_data_table.setHorizontalHeaderLabels(
            ["id", "type", "mode", "path/url/folder", "mime", "checksum", "reading options", "details"]
        )
        self.experiment_data_table.horizontalHeader().setStretchLastSection(True)
        self.experiment_data_table.itemSelectionChanged.connect(self.load_experiment_data)
        layout.addWidget(self.experiment_data_table, stretch=1)
        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_experiment_button = self._text_icon_button("💾 Save", QtWidgets.QStyle.SP_DialogSaveButton, "Save experiment", self.save_experiment)
        delete_experiment_button = self._text_icon_button("🗑 Delete", QtWidgets.QStyle.SP_TrashIcon, "Delete experiment", self.delete_experiment)
        add_data_button = self._text_icon_button("➕ Add", QtWidgets.QStyle.SP_FileDialogNewFolder, "Add data row", self.add_experiment_data_row)
        save_data_button = self._text_icon_button("💾 Save data", QtWidgets.QStyle.SP_DialogSaveButton, "Save data", self.save_experiment_data)
        delete_data_button = self._text_icon_button("🗑 Del data", QtWidgets.QStyle.SP_TrashIcon, "Delete data", self.delete_experiment_data)
        open_data_button = self._text_icon_button("📂 Open", QtWidgets.QStyle.SP_DialogOpenButton, "Open linked data", self.open_experiment_data)
        buttons.addWidget(save_experiment_button)
        buttons.addWidget(delete_experiment_button)
        buttons.addWidget(add_data_button)
        buttons.addWidget(save_data_button)
        buttons.addWidget(delete_data_button)
        buttons.addWidget(open_data_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def _refresh_sample_id_completer(self) -> None:
        try:
            samples = self.client.list_samples()
            ids = [s.get("sample_id", "") for s in samples if s.get("sample_id")]
            descs = [s.get("description", "") for s in samples if s.get("description")]
            items = list(dict.fromkeys(ids + descs))
            completer = QtWidgets.QCompleter(items, self)
            completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            completer.setFilterMode(QtCore.Qt.MatchContains)
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            self.sample_id_edit.setCompleter(completer)
        except Exception:
            pass

    def _set_transport_connected(self, connected: bool) -> None:
        self.transport_connected = bool(connected)
        for action in getattr(self, "_transport_actions", []):
            if action is getattr(self, "refresh_action", None):
                action.setEnabled(True)
            else:
                action.setEnabled(bool(connected))

    def refresh(self) -> None:
        self._loading = True
        failures: list[str] = []
        transport_ok = False
        try:
            status = self.client.status() or {}
            transport_ok = True
            user_db = status.get("user_database", "—")
            schema = status.get("schema_version", "?")
            sample_count = status.get("sample_count", "?")
            experiment_count = status.get("experiment_count", "?")
            status_text = (
                f"User DB: {user_db} | schema {schema} | "
                f"samples {sample_count} | experiments {experiment_count}"
            )
            self.status_label.setText(status_text)
        except Exception as exc:
            self.status_label.setText(f"MFDB transport unavailable: {exc}")
            self._set_transport_connected(False)
            self._loading = False
            return

        def _safe(label: str, fn) -> None:
            try:
                fn()
            except Exception as exc:
                failures.append(f"{label}: {exc}")


        _safe("users (combo)", self.fill_users)
        _safe("devices (combo)", self.fill_devices)
        _safe("users", self.fill_user_table)
        _safe("devices", self.fill_device_table)
        _safe("branches", self.fill_branch_table)
        _safe("branch user combo", self.fill_branch_user_combo)
        _safe("experiment types (combo)", self.fill_experiment_types)
        _safe("experiment types", self.fill_experiment_type_table)
        _safe("experiments", self.fill_experiment_table)
        _safe("projects", self.fill_project_table)
        _safe("experiment sample combo", self.fill_experiment_sample_combo)
        _safe("experiment user combo", self.fill_experiment_user_combo)
        _safe("experiment device combo", self.fill_experiment_device_combo)
        _safe("setups", self.fill_setup_table)
        _safe("raw data", self.fill_raw_data_table)
        _safe("processing runs", self.fill_processing_runs_table)
        _safe("processed products", self.fill_processed_products_table)
        _safe("analyses", self.fill_analyses_table)
        _safe("all items", self._populate_all_items)
        _safe("sample completer", self._refresh_sample_id_completer)
        for _table in self._checkable_tables:
            try:
                self._apply_checkable_first_column(_table)
            except Exception:
                pass

        self._set_transport_connected(transport_ok)
        if failures:
            short = failures[0]
            if len(failures) > 1:
                short = f"{short} (+{len(failures) - 1} more)"
            self.status_label.setText(
                f"{self.status_label.text()}  ⚠ partial refresh — {short}"
            )
        self._loading = False

    def clear_form(self) -> None:
        for widget in (
            self.sample_id_edit,
            self.uuid_edit,
            self.description_edit,
            self.condition_id_edit,
            self.assembly_id_edit,
            self.project_edit,
            self.measured_at_edit,
            self.condition_id_field,
            self.buffer_edit,
            self.user_id_edit,
            self.user_uuid_edit,
            self.user_display_edit,
            self.user_email_edit,
            self.user_affiliation_edit,
            self.user_department_edit,
            self.user_phone_edit,
            self.user_website_edit,
            self.user_active_branch_edit,
            self.device_id_edit,
            self.branch_uuid_edit,
            self.branch_name_edit,
            self.branch_parent_uuid_edit,
            self.branch_head_op_edit,
            self.device_name_edit,
            self.device_type_edit,
            self.device_model_edit,
            self.device_serial_edit,
            self.device_location_edit,
            self.device_owner_edit,
            self.experiment_type_id_edit,
            self.experiment_type_name_edit,
            self.experiment_type_category_edit,
            self.experiment_type_description_edit,
            self.experiment_id_edit,
            self.experiment_project_edit,
            self.experiment_started_edit,
            self.experiment_ended_edit,
            self.experiment_status_edit,
        ):
            widget.clear()
        self.details_edit.clear()
        self.condition_details_edit.clear()
        self.user_details_edit.clear()
        self.user_address_edit.clear()
        self.user_is_admin_check.setChecked(False)
        self.user_passwordless_check.setChecked(False)
        self.user_has_password_label.setText("—")
        self.user_created_label.setText("—")
        self.user_updated_label.setText("—")
        self.user_role_combo.setCurrentIndex(0)
        self.device_details_edit.clear()
        self.branch_description_edit.clear()
        self.branches_table.setRowCount(0)
        self.experiment_type_details_edit.clear()
        self.experiment_details_edit.clear()
        self.measured_by_combo.setCurrentIndex(-1)
        self.measured_device_combo.setCurrentIndex(-1)
        self.num_probes_spin.setValue(0)
        self.solvent_edit.setCurrentText("liquid")
        self.entities_table.setRowCount(0)
        self.probes_table.setRowCount(0)
        self.positions_table.setRowCount(0)
        self.metadata_editor.clear()
        if hasattr(self, "metadata_sample_label"):
            self.metadata_sample_label.setText("No sample selected")
        self.users_table.setRowCount(0)
        self.devices_table.setRowCount(0)
        self.experiment_types_table.setRowCount(0)
        self.experiments_table.setRowCount(0)
        self.experiment_data_table.setRowCount(0)
        self.preview_edit.clear()

    def load_sample(self, sample_id: str) -> None:
        if self._is_deleted():
            return
        self.current_sample_id = sample_id
        self.current_experiment_id = None
        sample = self.client.get_sample(sample_id) or {}
        self._loading = True
        try:
            self.sample_id_edit.setText(sample.get("sample_id", ""))
            self.uuid_edit.setText(sample.get("sample_uuid", ""))
            self.description_edit.setText(sample.get("description", ""))
            self.details_edit.setPlainText(sample.get("details", ""))
            self.num_probes_spin.setValue(int(sample.get("num_of_probes") or 0))
            self.solvent_edit.setCurrentText(sample.get("solvent_phase") or "liquid")
            self.condition_id_edit.setText(sample.get("sample_condition_id", ""))
            self.assembly_id_edit.setText(sample.get("entity_assembly_id", ""))
            self.project_edit.setText(sample.get("project_id", ""))
            self.measured_at_edit.setText(sample.get("measured_at", ""))
            self.condition_id_field.setText((sample.get("condition") or {}).get("condition_id", ""))
            condition = sample.get("condition") or {}
            self.ph_spin.setValue(float(condition.get("ph") or 0))
            self.temperature_spin.setValue(float(condition.get("temperature") or 0))
            self.ionic_spin.setValue(float(condition.get("ionic_strength") or 0))
            self.buffer_edit.setText(condition.get("buffer_composition", ""))
            self.condition_details_edit.setPlainText(condition.get("details", ""))
            self.fill_entities(sample.get("entities", []))
            self.fill_probes()
            self.fill_positions(sample.get("sample_probes", []))
            self.fill_metadata(sample.get("key_values", []))
            self.fill_experiment_table(sample_id=sample_id)
            self.measured_by_combo.setCurrentText("")
            self.measured_by_combo.setCurrentIndex(
                self.measured_by_combo.findData(sample.get("measured_by_user_id") or "")
            )
            self.measured_device_combo.setCurrentText("")
            self.measured_device_combo.setCurrentIndex(
                self.measured_device_combo.findData(sample.get("measured_by_device_id") or "")
            )
        finally:
            self._loading = False

    def fill_users(self) -> None:
        self.measured_by_combo.blockSignals(True)
        self.measured_by_combo.clear()
        self.measured_by_combo.addItem("", "")
        for user in self.client.list_users():
            user_id = user.get("user_id", "")
            self.measured_by_combo.addItem(
                f"{user.get('display_name') or user_id} ({user_id})",
                user_id,
            )
        self.measured_by_combo.blockSignals(False)

    def fill_devices(self) -> None:
        self.measured_device_combo.blockSignals(True)
        self.measured_device_combo.clear()
        self.measured_device_combo.addItem("", "")
        for device in self.client.list_devices():
            device_id = device.get("device_id", "")
            self.measured_device_combo.addItem(
                f"{device.get('name') or device_id} ({device_id})",
                device_id,
            )
        self.measured_device_combo.blockSignals(False)

    def fill_entities(self, entities: list[dict[str, Any]]) -> None:
        self.entities_table.setRowCount(0)
        for entity in entities:
            row = self.entities_table.rowCount()
            self.entities_table.insertRow(row)
            sequence = (
                self.client.get_sample(self.sample_id_edit.text()).get("sequence", "")
                if False
                else ""
            )
            values = [
                entity.get("entity_id", ""),
                entity.get("type", ""),
                entity.get("description", ""),
                entity.get("common_name", ""),
                sequence,
            ]
            for column, value in enumerate(values):
                self.entities_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_probes(self) -> None:
        self.probes_table.setRowCount(0)
        for row in self.client.list_probes():
            index = self.probes_table.rowCount()
            self.probes_table.insertRow(index)
            props = {
                prop.get("property_name", ""): prop
                for prop in row.get("optical_properties", [])
            }
            values = [
                row.get("probe_id", ""),
                row.get("chromophore_name", ""),
                row.get("category", ""),
                row.get("probe_origin", ""),
                row.get("probe_link_type", ""),
                props.get("abs_max", {}).get("property_value", ""),
                props.get("em_max", {}).get("property_value", ""),
                props.get("qy", {}).get("property_value", ""),
            ]
            for column, value in enumerate(values):
                self.probes_table.setItem(
                    index, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_positions(self, mappings: list[dict[str, Any]]) -> None:
        self.positions_table.setRowCount(0)
        for mapping in mappings:
            row = self.positions_table.rowCount()
            self.positions_table.insertRow(row)
            values = [
                mapping.get("sample_probe_id", ""),
                mapping.get("sample_id", ""),
                mapping.get("probe_id", ""),
                mapping.get("chromophore_name", ""),
                mapping.get("entity_id", ""),
                mapping.get("asym_id", ""),
                mapping.get("residue_number", ""),
                mapping.get("fluorophore_type", ""),
                mapping.get("description", "") or mapping.get("position_description", ""),
            ]
            for column, value in enumerate(values):
                self.positions_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_metadata(self, key_values: list[dict[str, Any]]) -> None:
        self.metadata_editor.set_data(key_values)
        sample_id = (self.current_sample_id or self.sample_id_edit.text()).strip()
        if sample_id:
            self.metadata_sample_label.setText(
                f"Editing metadata for sample: <b>{sample_id}</b>"
            )
        else:
            self.metadata_sample_label.setText("No sample selected")

    def collect_sample(self) -> dict[str, Any]:
        entities = []
        for row in range(self.entities_table.rowCount()):
            entities.append(
                {
                    "entity_id": self.entities_table.item(row, 0).text()
                    if self.entities_table.item(row, 0)
                    else "",
                    "type": self.entities_table.item(row, 1).text()
                    if self.entities_table.item(row, 1)
                    else "polymer",
                    "description": self.entities_table.item(row, 2).text()
                    if self.entities_table.item(row, 2)
                    else "",
                    "common_name": self.entities_table.item(row, 3).text()
                    if self.entities_table.item(row, 3)
                    else "",
                }
            )
        mappings = []
        for row in range(self.positions_table.rowCount()):
            mappings.append(
                {
                    "sample_probe_id": self.positions_table.item(row, 0).text()
                    if self.positions_table.item(row, 0)
                    else None,
                    "probe_id": self.positions_table.item(row, 2).text()
                    if self.positions_table.item(row, 2)
                    else None,
                    "fluorophore_type": self.positions_table.item(row, 7).text()
                    if self.positions_table.item(row, 7)
                    else "unspecified",
                    "description": self.positions_table.item(row, 8).text()
                    if self.positions_table.item(row, 8)
                    else "",
                }
            )
        key_values = self.metadata_editor.get_data()
        return {
            "sample_id": self.sample_id_edit.text().strip(),
            "sample_uuid": self.uuid_edit.text().strip(),
            "description": self.description_edit.text().strip(),
            "details": self.details_edit.toPlainText().strip(),
            "num_of_probes": self.num_probes_spin.value(),
            "solvent_phase": self.solvent_edit.currentText(),
            "sample_condition_id": self.condition_id_edit.text().strip(),
            "entity_assembly_id": self.assembly_id_edit.text().strip(),
            "project_id": self.project_edit.text().strip(),
            "measured_by_user_id": self.measured_by_combo.currentData() or None,
            "measured_by_device_id": self.measured_device_combo.currentData() or None,
            "measured_at": self.measured_at_edit.text().strip(),
            "condition": {
                "condition_id": self.condition_id_field.text().strip(),
                "ph": None if self.ph_spin.value() == 0 else self.ph_spin.value(),
                "temperature": None
                if self.temperature_spin.value() == 0
                else self.temperature_spin.value(),
                "ionic_strength": None if self.ionic_spin.value() == 0 else self.ionic_spin.value(),
                "buffer_composition": self.buffer_edit.text().strip(),
                "details": self.condition_details_edit.toPlainText().strip(),
            },
            "entities": entities,
            "sample_probes": mappings,
            "key_values": key_values,
        }

    def save_sample(self) -> None:
        sample = self.collect_sample()
        if not sample["sample_id"]:
            self.status_label.setText("Sample id is required")
            return
        saved = self.client.save_sample(sample)
        self.status_label.setText(f"Saved {saved['sample_id']}")
        self.refresh()

    def add_metadata_row(self) -> None:
        self.metadata_editor._on_add_empty_row()

    def delete_metadata_row(self) -> None:
        self.metadata_editor._on_delete_row()

    def collect_user(self) -> dict[str, Any]:
        role = self.user_role_combo.currentText().strip() or None
        payload: dict[str, Any] = {
            "user_id": self.user_id_edit.text().strip(),
            "user_uuid": self.user_uuid_edit.text().strip() or None,
            "display_name": self.user_display_edit.text().strip(),
            "email": self.user_email_edit.text().strip() or None,
            "role": role,
            "affiliation": self.user_affiliation_edit.text().strip() or None,
            "department": self.user_department_edit.text().strip() or None,
            "phone": self.user_phone_edit.text().strip() or None,
            "website": self.user_website_edit.text().strip() or None,
            "address": self.user_address_edit.toPlainText().strip() or None,
            "details": self.user_details_edit.toPlainText().strip() or None,
            "is_admin": 1 if self.user_is_admin_check.isChecked() else 0,
            "allow_passwordless_login": 1 if self.user_passwordless_check.isChecked() else 0,
            "requester_id": self._active_mfdb_user_id(),
        }
        return payload

    def new_user(self) -> None:
        """Prepare the user form for a new entry with a generated UUID."""
        self.users_table.clearSelection()
        self._loading = True
        try:
            self.user_uuid_edit.setText(str(uuid.uuid4()))
            self.user_id_edit.clear()
            self.user_display_edit.clear()
            self.user_email_edit.clear()
            self.user_role_combo.setCurrentIndex(0)
            self.user_affiliation_edit.clear()
            self.user_department_edit.clear()
            self.user_phone_edit.clear()
            self.user_website_edit.clear()
            self.user_address_edit.clear()
            self.user_is_admin_check.setChecked(False)
            self.user_passwordless_check.setChecked(False)
            self.user_active_branch_edit.clear()
            self.user_has_password_label.setText("not set")
            self.user_created_label.setText("—")
            self.user_updated_label.setText("—")
            self.user_details_edit.clear()
        finally:
            self._loading = False
        self.user_id_edit.setFocus()
        self.status_label.setText("New user — fill fields and Save")

    def save_user(self) -> None:
        users = self.client.save_user(self.collect_user())
        self.fill_user_table(users)
        self.refresh()

    def delete_user(self) -> None:
        user_id = self.user_id_edit.text().strip()
        if not user_id:
            return
        users = self.client.delete_user(user_id)
        self.fill_user_table(users)
        self.refresh()

    def change_user_password(self) -> None:
        """Set or clear the selected user's MFDB password.

        Goes through ``save_user`` so the active admin can change the
        password of any other user. ``change_password`` only updates the
        password of the currently authenticated principal and is therefore
        unsuitable for this admin tool.
        """
        user_id = self.user_id_edit.text().strip()
        if not user_id:
            QtWidgets.QMessageBox.warning(
                self, "Selection Required", "Please select or save a user first."
            )
            return
        user = self._selected_user_payload()
        is_admin = bool(user.get("is_admin")) if user else False
        dialog = PasswordChangeDialog(user_id=user_id, is_admin=is_admin, parent=self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        payload = {
            "user_id": user_id,
            "password": "" if dialog.cleared else dialog.password,
            "requester_id": self._active_mfdb_user_id(),
        }
        try:
            self.client.save_user(payload)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not change password: {exc}")
            return
        self.refresh()
        if dialog.cleared:
            QtWidgets.QMessageBox.information(self, "Success", f"Password cleared for '{user_id}'.")
        else:
            QtWidgets.QMessageBox.information(self, "Success", f"Password updated for '{user_id}'.")

    def load_user(self) -> None:
        rows = self.users_table.selectionModel().selectedRows()
        if not rows:
            return
        user = self._selected_user_payload()
        if not user:
            return
        self._loading = True
        try:
            self.user_uuid_edit.setText(str(user.get("user_uuid") or ""))
            self.user_id_edit.setText(str(user.get("user_id") or ""))
            self.user_display_edit.setText(str(user.get("display_name") or ""))
            self.user_email_edit.setText(str(user.get("email") or ""))
            role = str(user.get("role") or "")
            idx = self.user_role_combo.findText(role)
            if idx >= 0:
                self.user_role_combo.setCurrentIndex(idx)
            else:
                self.user_role_combo.setEditText(role)
            self.user_affiliation_edit.setText(str(user.get("affiliation") or ""))
            self.user_department_edit.setText(str(user.get("department") or ""))
            self.user_phone_edit.setText(str(user.get("phone") or ""))
            self.user_website_edit.setText(str(user.get("website") or ""))
            self.user_address_edit.setPlainText(str(user.get("address") or ""))
            self.user_is_admin_check.setChecked(bool(user.get("is_admin")))
            self.user_passwordless_check.setChecked(bool(user.get("allow_passwordless_login")))
            self.user_active_branch_edit.setText(str(user.get("active_branch_uuid") or ""))
            self.user_has_password_label.setText("set" if user.get("has_password") else "not set")
            self.user_created_label.setText(str(user.get("created_at") or "—"))
            self.user_updated_label.setText(str(user.get("updated_at") or "—"))
            self.user_details_edit.setPlainText(str(user.get("details") or ""))
        finally:
            self._loading = False

    def fill_user_table(self, users: list[dict[str, Any]] | None = None) -> None:
        users = users if users is not None else self.client.list_users()
        self.users_table.setRowCount(0)
        for user in users:
            row = self.users_table.rowCount()
            self.users_table.insertRow(row)
            values = [
                user.get("user_id", ""),
                user.get("display_name", ""),
                user.get("email", ""),
                user.get("role", ""),
                user.get("department", ""),
                user.get("affiliation", ""),
                "yes" if user.get("is_admin") else "no",
                "set" if user.get("has_password") else "none",
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value or ""))
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, user)
                self.users_table.setItem(row, column, item)

    def _selected_user_payload(self) -> dict[str, Any]:
        """Return the selected user's full table payload."""
        rows = self.users_table.selectionModel().selectedRows()
        if not rows:
            return {}
        item = self.users_table.item(rows[0].row(), 0)
        data = item.data(QtCore.Qt.UserRole) if item is not None else None
        return data if isinstance(data, dict) else {}

    def _active_mfdb_user_id(self) -> str:
        """Return the active MFDB user ID from settings."""
        try:
            import chisurf.core.settings as cs_settings

            return cs_settings.cs_settings.get("mfdb", {}).get("default_user_id", "user_default")
        except Exception:
            return "user_default"

    def load_branch(self) -> None:
        rows = self.branches_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        self.branch_name_edit.setText(self.branches_table.item(row, 0).text() or "")
        self.branch_uuid_edit.setText(self.branches_table.item(row, 1).text() or "")
        self.branch_parent_uuid_edit.setText(self.branches_table.item(row, 2).text() or "")
        self.branch_head_op_edit.setText(self.branches_table.item(row, 3).text() or "")
        self.branch_description_edit.setPlainText(self.branches_table.item(row, 4).text() or "")

    def fill_branch_table(self, branches: list[dict[str, Any]] | None = None) -> None:
        if self._loading:
            return
        try:
            branches = branches if branches is not None else self.client.list_branches()
            self.branches_table.setRowCount(0)
            for b in branches:
                row = self.branches_table.rowCount()
                self.branches_table.insertRow(row)
                values = [
                    b.get("name", ""),
                    b.get("branch_uuid", ""),
                    b.get("parent_branch_uuid", ""),
                    b.get("head_operation_id", ""),
                    b.get("description", ""),
                ]
                for column, value in enumerate(values):
                    self.branches_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))
        except Exception:
            pass

    def save_branch(self) -> None:
        try:
            name = self.branch_name_edit.text().strip()
            uuid_val = self.branch_uuid_edit.text().strip() or None
            parent_uuid = self.branch_parent_uuid_edit.text().strip() or None
            head_op = self.branch_head_op_edit.text().strip() or None
            desc = self.branch_description_edit.toPlainText().strip() or None
            creator_user = self.branch_user_combo.currentData() or "user_default"

            self.client.create_branch(
                branch_uuid=uuid_val,
                name=name,
                parent_branch_uuid=parent_uuid,
                head_operation_id=head_op,
                created_by_user_id=creator_user,
                description=desc,
            )
            self.refresh()
            QtWidgets.QMessageBox.information(self, "Success", f"Branch '{name}' saved successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not save branch: {e}")

    def delete_branch(self) -> None:
        uuid_val = self.branch_uuid_edit.text().strip()
        if not uuid_val:
            return
        if QtWidgets.QMessageBox.question(
            self, "Confirm Delete", f"Are you sure you want to delete branch with UUID {uuid_val}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        ) != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.client.delete_branch(uuid_val)
            self.refresh()
            QtWidgets.QMessageBox.information(self, "Success", "Branch deleted.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not delete branch: {e}")

    def fill_branch_user_combo(self) -> None:
        current = self.branch_user_combo.currentData()
        self.branch_user_combo.blockSignals(True)
        self.branch_user_combo.clear()
        try:
            users = self.client.list_users()
            for user in users:
                user_id = user.get("user_id", "")
                self.branch_user_combo.addItem(
                    f"{user.get('display_name') or user_id} ({user_id})",
                    user_id,
                )
            if current:
                self._set_combo_data(self.branch_user_combo, current)
            else:
                self._set_combo_data(self.branch_user_combo, "user_default")
        except Exception:
            pass
        finally:
            self.branch_user_combo.blockSignals(False)
            self.on_branch_user_changed(0)

    def on_branch_user_changed(self, index: int) -> None:
        user_id = self.branch_user_combo.currentData()
        if not user_id:
            self.active_branch_label.setText("Unknown")
            return
        try:
            branch = self.client.get_user_active_branch(user_id)
            if branch:
                self.active_branch_label.setText(f"{branch.get('name')} ({branch.get('branch_uuid')})")
            else:
                self.active_branch_label.setText("None (Default main)")
        except Exception:
            self.active_branch_label.setText("Error fetching branch")

    def switch_active_branch(self) -> None:
        user_id = self.branch_user_combo.currentData()
        rows = self.branches_table.selectionModel().selectedRows()
        if not user_id:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a user first.")
            return
        if not rows:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a branch from the table first.")
            return
        branch_uuid = self.branches_table.item(rows[0].row(), 1).text()
        branch_name = self.branches_table.item(rows[0].row(), 0).text()
        try:
            self.client.set_user_active_branch(user_id, branch_uuid)
            self.on_branch_user_changed(0)
            QtWidgets.QMessageBox.information(
                self, "Success", f"Active branch for user '{user_id}' switched to '{branch_name}'."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not switch active branch: {e}")

    def prefill_branch_fork(self) -> None:
        """Populate branch fields from the selected branch head."""
        rows = self.branches_table.selectionModel().selectedRows()
        if not rows:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a branch from the table first.")
            return
        row = rows[0].row()
        name = self.branches_table.item(row, 0).text()
        branch_uuid = self.branches_table.item(row, 1).text()
        head_operation_id = self.branches_table.item(row, 3).text()
        self.branch_uuid_edit.clear()
        self.branch_name_edit.setText(f"{name}-branch")
        self.branch_parent_uuid_edit.setText(branch_uuid)
        self.branch_head_op_edit.setText(head_operation_id)
        self.branch_description_edit.setPlainText(f"Parallel branch from {name}")

    def create_time_branch(self) -> None:
        """Create and activate a branch at the requested operation."""
        user_id = self.branch_user_combo.currentData()
        operation_id = self.branch_head_op_edit.text().strip()
        branch_name = self.branch_name_edit.text().strip() or None
        branch_uuid = self.branch_uuid_edit.text().strip() or None
        parent_uuid = self.branch_parent_uuid_edit.text().strip() or None
        description = self.branch_description_edit.toPlainText().strip() or None
        if not user_id:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a user first.")
            return
        if not operation_id:
            QtWidgets.QMessageBox.warning(self, "Warning", "Head Operation ID is required.")
            return
        try:
            branch = self.client.jump_user_to_operation(
                user_id=user_id,
                operation_id=operation_id,
                branch_name=branch_name,
                branch_uuid=branch_uuid,
                parent_branch_uuid=parent_uuid,
                description=description,
            )
            self.refresh()
            self.on_branch_user_changed(0)
            QtWidgets.QMessageBox.information(
                self,
                "Success",
                f"User '{user_id}' jumped to branch '{branch.get('name')}'.",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not create time branch: {e}")

    def collect_device(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id_edit.text().strip(),
            "name": self.device_name_edit.text().strip(),
            "device_type": self.device_type_edit.text().strip() or None,
            "model": self.device_model_edit.text().strip() or None,
            "serial_number": self.device_serial_edit.text().strip() or None,
            "location": self.device_location_edit.text().strip() or None,
            "owner": self.device_owner_edit.text().strip() or None,
            "details": self.device_details_edit.toPlainText().strip() or None,
        }

    def save_device(self) -> None:
        devices = self.client.save_device(self.collect_device())
        self.fill_device_table(devices)
        self.refresh()

    def delete_device(self) -> None:
        device_id = self.device_id_edit.text().strip()
        if not device_id:
            return
        devices = self.client.delete_device(device_id)
        self.fill_device_table(devices)
        self.refresh()

    def load_device(self) -> None:
        rows = self.devices_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        self.device_id_edit.setText(self.devices_table.item(row, 0).text() or "")
        self.device_name_edit.setText(self.devices_table.item(row, 1).text() or "")
        self.device_type_edit.setText(self.devices_table.item(row, 2).text() or "")
        self.device_model_edit.setText(self.devices_table.item(row, 3).text() or "")
        self.device_serial_edit.setText(self.devices_table.item(row, 4).text() or "")
        self.device_location_edit.setText(self.devices_table.item(row, 5).text() or "")
        self.device_owner_edit.setText(self.devices_table.item(row, 6).text() or "")
        self.device_details_edit.setPlainText(self.devices_table.item(row, 7).text() or "")

    def collect_experiment_type(self) -> dict[str, Any]:
        return {
            "type_id": self.experiment_type_id_edit.text().strip() or None,
            "name": self.experiment_type_name_edit.text().strip(),
            "category": self.experiment_type_category_edit.text().strip() or None,
            "description": self.experiment_type_description_edit.text().strip() or None,
            "details": self.experiment_type_details_edit.toPlainText().strip() or None,
        }

    def save_experiment_type(self) -> None:
        types = self.client.save_experiment_type(self.collect_experiment_type())
        self.fill_experiment_type_table(types)
        self.refresh()

    def delete_experiment_type(self) -> None:
        type_id = self.experiment_type_id_edit.text().strip()
        if not type_id:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete experiment type", f"Delete experiment type {type_id}?"
        )
        if answer == QtWidgets.QMessageBox.Yes:
            types = self.client.delete_experiment_type(int(type_id))
            self.fill_experiment_type_table(types)
            self.refresh()

    def collect_experiment(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id_edit.text().strip(),
            "type_id": int(self.experiment_type_combo.currentData() or -1)
            if self.experiment_type_combo.currentData() not in (None, -1, "")
            else None,
            "sample_id": self.experiment_sample_combo.currentData() or None,
            "project_id": self.experiment_project_edit.text().strip() or None,
            "measured_by_user_id": self.experiment_user_combo.currentData() or None,
            "measured_by_device_id": self.experiment_device_combo.currentData() or None,
            "started_at": self.experiment_started_edit.text().strip() or None,
            "ended_at": self.experiment_ended_edit.text().strip() or None,
            "status": self.experiment_status_edit.text().strip() or None,
            "details": self.experiment_details_edit.toPlainText().strip() or None,
        }

    def save_experiment(self) -> None:
        experiment = self.collect_experiment()
        if not experiment["experiment_id"]:
            self.status_label.setText("Experiment id is required")
            return
        saved = self.client.save_experiment(experiment)
        self.status_label.setText(f"Saved {saved['experiment_id']}")
        self.refresh()

    def delete_experiment(self) -> None:
        experiment_id = self.experiment_id_edit.text().strip()
        if not experiment_id:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete experiment", f"Delete experiment {experiment_id}?"
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.client.delete_experiment(experiment_id)
            self.refresh()

    def add_experiment_data_row(self) -> None:
        row = self.experiment_data_table.rowCount()
        self.experiment_data_table.insertRow(row)
        for column in range(self.experiment_data_table.columnCount()):
            self.experiment_data_table.setItem(row, column, QtWidgets.QTableWidgetItem(""))
        self.experiment_data_table.setCurrentCell(row, 0)

    def collect_experiment_data(self) -> dict[str, Any]:
        row = self.experiment_data_table.currentRow()
        data_id = self.experiment_data_table.item(row, 0).text() if row >= 0 else ""
        location = self.experiment_data_table.item(row, 3).text() if row >= 0 else ""
        storage_mode = (
            self.experiment_data_table.item(row, 2).text() if row >= 0 else "link"
        ) or "link"
        file_path = location if storage_mode == "link" and not location.startswith(("http://", "https://", "file://")) else None
        url = location if location.startswith(("http://", "https://", "file://")) else None
        folder_path = location if storage_mode == "folder" else None
        return {
            "data_id": int(data_id) if data_id else None,
            "experiment_id": self.experiment_id_edit.text().strip(),
            "data_type": self.experiment_data_table.item(row, 1).text() if row >= 0 else "",
            "storage_mode": storage_mode,
            "file_path": file_path,
            "url": url,
            "folder_path": folder_path,
            "mime_type": self.experiment_data_table.item(row, 4).text() if row >= 0 else "",
            "checksum": self.experiment_data_table.item(row, 5).text() if row >= 0 else "",
            "reading_options_json": (
                self.experiment_data_table.item(row, 6).text() if row >= 0 else ""
            ),
            "details": self.experiment_data_table.item(row, 7).text() if row >= 0 else "",
        }

    def save_experiment_data(self) -> None:
        data = self.collect_experiment_data()
        if not data["experiment_id"]:
            self.status_label.setText("Select or create an experiment first")
            return
        if not data["data_type"]:
            self.status_label.setText("Data type is required")
            return
        try:
            json.loads(data["reading_options_json"] or "{}")
        except json.JSONDecodeError as exc:
            self.status_label.setText(f"Invalid reading options JSON: {exc}")
            return
        saved = self.client.save_experiment_data(data)
        self.fill_experiment_data_table(saved.get("data", []))
        self.fill_experiment_table()
        self.refresh()

    def delete_experiment_data(self) -> None:
        row = self.experiment_data_table.currentRow()
        if row < 0:
            return
        data_id = self.experiment_data_table.item(row, 0).text()
        if not data_id:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete experiment data", f"Delete data record {data_id}?"
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.client.delete_experiment_data(int(data_id))
            self.refresh()

    def open_experiment_data(self) -> None:
        row = self.experiment_data_table.currentRow()
        if row < 0:
            return
        location = self.experiment_data_table.item(row, 3).text() or ""
        if not location:
            QtWidgets.QMessageBox.information(self, "Experiment data", "No data location is set")
            return
        if location.startswith(("http://", "https://", "file://")):
            QtGui.QDesktopServices.openUrl(QUrl(location))
            return
        path = QtCore.QFileInfo(location).absoluteFilePath()
        if not QtCore.QFile(path).exists():
            QtWidgets.QMessageBox.warning(self, "Data path not found", f"Missing: {path}")
            return
        QtGui.QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def fill_device_table(self, devices: list[dict[str, Any]] | None = None) -> None:
        devices = devices if devices is not None else self.client.list_devices()
        self.devices_table.setRowCount(0)
        for device in devices:
            row = self.devices_table.rowCount()
            self.devices_table.insertRow(row)
            values = [
                device.get("device_id", ""),
                device.get("name", ""),
                device.get("device_type", ""),
                device.get("model", ""),
                device.get("serial_number", ""),
                device.get("location", ""),
                device.get("owner", ""),
                device.get("details", ""),
            ]
            for column, value in enumerate(values):
                self.devices_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_experiment_types(self) -> None:
        self.experiment_type_combo.blockSignals(True)
        self.experiment_type_combo.clear()
        self.experiment_type_combo.addItem("", -1)
        for item in self.client.list_experiment_types():
            type_id = int(item.get("type_id") or -1)
            name = item.get("name") or item.get("type_id") or ""
            self.experiment_type_combo.addItem(str(name), type_id)
        self.experiment_type_combo.blockSignals(False)

    def fill_experiment_type_table(self) -> None:
        self.experiment_types_table.setRowCount(0)
        for item in self.client.list_experiment_types():
            row = self.experiment_types_table.rowCount()
            self.experiment_types_table.insertRow(row)
            values = [
                item.get("type_id", ""),
                item.get("name", ""),
                item.get("category", ""),
                item.get("description", ""),
                item.get("details", ""),
            ]
            for column, value in enumerate(values):
                self.experiment_types_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_experiment_table(self, sample_id: str | None = None) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.experiments_table):
            return
        self.experiments_table.setRowCount(0)
        for item in self.client.list_experiments(sample_id=sample_id):
            row = self.experiments_table.rowCount()
            self.experiments_table.insertRow(row)
            values = [
                item.get("experiment_id", ""),
                item.get("experiment_type", ""),
                item.get("sample_id", ""),
                item.get("project_id", ""),
                item.get("measured_by_user", ""),
                item.get("measured_by_device", ""),
                item.get("started_at", ""),
                item.get("status", ""),
            ]
            for column, value in enumerate(values):
                self.experiments_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_experiment_sample_combo(self) -> None:
        self.experiment_sample_combo.blockSignals(True)
        self.experiment_sample_combo.clear()
        self.experiment_sample_combo.addItem("", "")
        for sample in self.client.list_samples():
            sample_id = sample.get("sample_id", "")
            description = sample.get("description") or sample_id
            self.experiment_sample_combo.addItem(f"{description} ({sample_id})", sample_id)
        self.experiment_sample_combo.blockSignals(False)

    def fill_experiment_user_combo(self) -> None:
        self.experiment_user_combo.blockSignals(True)
        self.experiment_user_combo.clear()
        self.experiment_user_combo.addItem("", "")
        for user in self.client.list_users():
            user_id = user.get("user_id", "")
            self.experiment_user_combo.addItem(
                f"{user.get('display_name') or user_id} ({user_id})",
                user_id,
            )
        self.experiment_user_combo.blockSignals(False)

    def fill_experiment_device_combo(self) -> None:
        self.experiment_device_combo.blockSignals(True)
        self.experiment_device_combo.clear()
        self.experiment_device_combo.addItem("", "")
        for device in self.client.list_devices():
            device_id = device.get("device_id", "")
            self.experiment_device_combo.addItem(
                f"{device.get('name') or device_id} ({device_id})",
                device_id,
            )
        self.experiment_device_combo.blockSignals(False)

    def load_experiment_type(self) -> None:
        rows = self.experiment_types_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        self.experiment_type_id_edit.setText(self.experiment_types_table.item(row, 0).text() or "")
        self.experiment_type_name_edit.setText(self.experiment_types_table.item(row, 1).text() or "")
        self.experiment_type_category_edit.setText(
            self.experiment_types_table.item(row, 2).text() or ""
        )
        self.experiment_type_description_edit.setText(
            self.experiment_types_table.item(row, 3).text() or ""
        )
        self.experiment_type_details_edit.setPlainText(
            self.experiment_types_table.item(row, 4).text() or ""
        )

    def load_experiment(self) -> None:
        rows = self.experiments_table.selectionModel().selectedRows()
        if not rows:
            return
        experiment_id = self.experiments_table.item(rows[0].row(), 0).text()
        self.current_experiment_id = experiment_id
        experiment = self.client.get_experiment(experiment_id) or {}
        self._loading = True
        try:
            self.experiment_id_edit.setText(experiment.get("experiment_id", ""))
            self._set_combo_data(self.experiment_type_combo, experiment.get("type_id"))
            self._set_combo_data(self.experiment_sample_combo, experiment.get("sample_id"))
            self.experiment_project_edit.setText(experiment.get("project_id", ""))
            self._set_combo_data(self.experiment_user_combo, experiment.get("measured_by_user_id"))
            self._set_combo_data(self.experiment_device_combo, experiment.get("measured_by_device_id"))
            self.experiment_started_edit.setText(experiment.get("started_at", ""))
            self.experiment_ended_edit.setText(experiment.get("ended_at", ""))
            self.experiment_status_edit.setText(experiment.get("status", ""))
            self.experiment_details_edit.setPlainText(experiment.get("details", ""))
            self.fill_experiment_data_table(experiment.get("data", []))

            # Refresh dependent views
            self.fill_raw_data_table()
            self.fill_processing_runs_table()
            self.fill_processed_products_table()
            self.fill_analyses_table()
        finally:
            self._loading = False

    def fill_experiment_data_table(self, data_rows: list[dict[str, Any]]) -> None:
        self.experiment_data_table.setRowCount(0)
        for item in data_rows:
            row = self.experiment_data_table.rowCount()
            self.experiment_data_table.insertRow(row)
            location = item.get("file_path") or item.get("url") or item.get("folder_path") or ""
            values = [
                item.get("data_id", ""),
                item.get("data_type", ""),
                item.get("storage_mode", ""),
                location,
                item.get("mime_type", ""),
                item.get("checksum", ""),
                item.get("reading_options_json", ""),
                item.get("details", ""),
            ]
            for column, value in enumerate(values):
                self.experiment_data_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def load_experiment_data(self) -> None:
        row = self.experiment_data_table.currentRow()
        if row < 0:
            return
        self.status_label.setText(
            self.experiment_data_table.item(row, 6).text()
            or self.experiment_data_table.item(row, 7).text()
            or ""
        )

    def _set_combo_data(self, combo: QtWidgets.QComboBox, data: Any) -> None:
        combo.setCurrentText("")
        combo.setCurrentIndex(combo.findData(data))

    def export_table(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export sample table",
            "samples.csv",
            "CSV/Excel files (*.csv *.tsv *.xlsx);;All files (*)",
        )
        if not path:
            return
        result = self.client.export_table(path)
        self.status_label.setText(f"Exported {result.get('output_path')}")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._restore_dock_layout()

    def closeEvent(self, event) -> None:
        self._save_dock_layout()
        super().closeEvent(event)

    def _dock_settings(self):
        settings = QtCore.QSettings(
            str(get_plugin_settings_path("mfdb_admin")),
            QtCore.QSettings.IniFormat,
        )
        if not settings.value("dock_layout"):
            legacy = QtCore.QSettings(
                str(get_plugin_settings_path("sample_database")),
                QtCore.QSettings.IniFormat,
            )
            legacy_state = legacy.value("dock_layout")
            if legacy_state:
                settings.setValue("dock_layout", legacy_state)
        return settings

    def _save_dock_layout(self) -> None:
        if not hasattr(self, "tabs"):
            return
        settings = self._dock_settings()
        settings.setValue("dock_layout", json.dumps(self.tabs.get_layout_state()))

    def _restore_dock_layout(self) -> None:
        if not hasattr(self, "tabs"):
            return
        settings = self._dock_settings()
        state = json.loads(settings.value("dock_layout", "") or "{}")
        if state:
            self.tabs.set_layout_state(state)

    def new_sample(self) -> None:
        sample_id = f"sample_{datetime.now():%Y%m%d_%H%M%S}"
        self.sample_id_edit.setText(sample_id)
        self.uuid_edit.setText(str(uuid.uuid4()))
        self.description_edit.setText("")
        self.details_edit.clear()
        self.project_edit.clear()
        self.measured_by_combo.setCurrentIndex(-1)
        self.measured_device_combo.setCurrentIndex(-1)
        self.measured_at_edit.setText(datetime.now().isoformat(timespec="seconds"))
        self.condition_id_field.setText(f"condition_{sample_id}")
        self.entities_table.setRowCount(0)
        self.positions_table.setRowCount(0)
        self.metadata_editor.clear()
        if hasattr(self, "metadata_sample_label"):
            self.metadata_sample_label.setText(
                f"Editing metadata for sample: <b>{sample_id}</b>"
            )

    def delete_sample(self) -> None:
        sample_id = self.sample_id_edit.text().strip()
        if not sample_id:
            return
        answer = QtWidgets.QMessageBox.question(self, "Delete sample", f"Delete {sample_id}?")
        if answer == QtWidgets.QMessageBox.Yes:
            self.client.delete_sample(sample_id)
            self.refresh()

    def browse_import_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import PDBx/PDB-IHM/FLR CIF",
            "",
            "CIF/mmCIF files (*.cif *.mmcif);;All files (*)",
        )
        if path:
            self.file_edit.setText(path)

    def import_file(self) -> None:
        path = self.file_edit.text().strip()
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Import PDBx/PDB-IHM/FLR CIF",
                "",
                "CIF/mmCIF files (*.cif *.mmcif);;All files (*)",
            )
        if not path:
            return
        summary = self.client.import_file(path)
        self.preview_edit.setPlainText(str(summary))
        self.refresh()

    def export_selected_sample(self) -> None:
        sample_id = self.sample_id_edit.text().strip()
        if not sample_id:
            self.status_label.setText("Select a sample to export")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export FLR CIF", f"{sample_id}.cif", "CIF files (*.cif *.mmcif);;All files (*)"
        )
        if not path:
            return
        result = self.client.export_sample(sample_id, output_path=path)
        self.status_label.setText(f"Exported {result.get('output_path')}")

    def backup_database(self) -> None:
        path = self.client.backup()
        self.status_label.setText(f"Backup: {path}")

    def reset_from_source(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self, "Reset database", "Replace the user database with the curated source database?"
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        result = self.client.reset_from_source()
        self.status_label.setText(f"Reset complete. Backup: {result.get('backup_path')}")
        self.refresh()

    def projects_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.projects_table = QtWidgets.QTableWidget(0, 5)
        self.projects_table.setHorizontalHeaderLabels(
            ["project id", "name", "experiment id", "created at", "notes"]
        )
        self.projects_table.horizontalHeader().setStretchLastSection(True)
        self.projects_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.projects_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.projects_table.itemSelectionChanged.connect(self.load_project_details)
        self._install_table_context_menu(
            self.projects_table,
            item_kind="project",
            id_col=0,
            delete_one_fn=lambda pid: self.client.delete_project(pid),
        )
        layout.addWidget(self.projects_table, stretch=1)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.project_id_edit = QtWidgets.QLineEdit()
        self.project_id_edit.setReadOnly(True)
        self.project_name_edit = QtWidgets.QLineEdit()
        self.project_name_edit.setReadOnly(True)
        self.project_notes_edit = QtWidgets.QPlainTextEdit()
        self.project_notes_edit.setReadOnly(True)
        self.project_notes_edit.setMinimumHeight(60)

        form.addRow("Project ID", self.project_id_edit)
        form.addRow("Name", self.project_name_edit)
        form.addRow("Notes", self.project_notes_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        restore_button = self._text_icon_button(
            "📥 Restore project to ChiSurf",
            QtWidgets.QStyle.SP_DialogOpenButton,
            "Restore project state from database",
            self.restore_selected_project,
        )
        delete_button = self._text_icon_button(
            "🗑 Delete archived project",
            QtWidgets.QStyle.SP_TrashIcon,
            "Delete the archived project",
            self.delete_selected_project,
        )
        buttons.addWidget(restore_button)
        buttons.addWidget(delete_button)
        buttons.addStretch()
        layout.addLayout(buttons)

        return widget

    def fill_project_table(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.projects_table):
            return
        self.projects_table.setRowCount(0)
        try:
            projects = self.client.list_projects()
        except Exception:
            projects = []
        for item in projects:
            row = self.projects_table.rowCount()
            self.projects_table.insertRow(row)
            values = [
                item.get("analysis_id", ""),
                item.get("model_name", ""),
                item.get("experiment_id", ""),
                item.get("created_at", ""),
                item.get("notes", ""),
            ]
            for column, value in enumerate(values):
                self.projects_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def load_project_details(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.projects_table):
            return
        selected = self.projects_table.selectedItems()
        if not selected:
            self.project_id_edit.clear()
            self.project_name_edit.clear()
            self.project_notes_edit.clear()
            return

        row = selected[0].row()
        project_id = self.projects_table.item(row, 0).text()
        project_name = self.projects_table.item(row, 1).text()
        project_notes = self.projects_table.item(row, 4).text()

        self.project_id_edit.setText(project_id)
        self.project_name_edit.setText(project_name)
        self.project_notes_edit.setPlainText(project_notes)

    def restore_selected_project(self) -> None:
        project_id = self.project_id_edit.text()
        if not project_id:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a project to restore.")
            return

        try:
            from chisurf.core.actions import dispatch
            dispatch("project.restore", {"project_id": project_id})
            QtWidgets.QMessageBox.information(
                self,
                "Project Restored",
                "Successfully restored project state from database."
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Restore Failed",
                f"Failed to restore project: {exc}"
            )

    def delete_selected_project(self) -> None:
        project_id = self.project_id_edit.text()
        if not project_id:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a project to delete.")
            return

        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete Project",
            f"Are you sure you want to delete the archived project '{project_id}' from the database?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        try:
            self.client.delete_project(project_id)
            QtWidgets.QMessageBox.information(
                self,
                "Project Deleted",
                "Project successfully deleted from the database."
            )
            self.refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Delete Failed",
                f"Failed to delete project: {exc}"
            )

    def setups_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setups_table = QtWidgets.QTableWidget(0, 5)
        self.setups_table.setHorizontalHeaderLabels(
            ["setup id", "name", "instrument type", "details", "lasers/detectors"]
        )
        self.setups_table.horizontalHeader().setStretchLastSection(True)
        self.setups_table.itemSelectionChanged.connect(self.load_setup)
        self._install_table_context_menu(
            self.setups_table,
            item_kind="setup",
            id_col=0,
            delete_one_fn=lambda sid: self.client._call("mfdb.setups.delete", {"setup_id": sid}),
        )
        layout.addWidget(self.setups_table, stretch=1)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.setup_id_edit = QtWidgets.QLineEdit()
        self.setup_name_edit = QtWidgets.QLineEdit()
        self.setup_instrument_edit = QtWidgets.QLineEdit()
        self.setup_lasers_edit = QtWidgets.QLineEdit()
        self.setup_detectors_edit = QtWidgets.QLineEdit()
        self.setup_details_edit = QtWidgets.QPlainTextEdit()
        self.setup_details_edit.setMinimumHeight(60)

        form.addRow("Setup id", self.setup_id_edit)
        form.addRow("Name", self.setup_name_edit)
        form.addRow("Instrument type", self.setup_instrument_edit)
        form.addRow("Laser wavelengths (JSON)", self.setup_lasers_edit)
        form.addRow("Detector channels (JSON)", self.setup_detectors_edit)
        form.addRow("Details/JSON", self.setup_details_edit)
        layout.addLayout(form)

        self.setup_validation_label = QtWidgets.QLabel("")
        self.setup_validation_label.setWordWrap(True)
        layout.addWidget(self.setup_validation_label)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        save_setup_button = self._text_icon_button(
            "💾 Save setup", QtWidgets.QStyle.SP_DialogSaveButton, "Save setup", self.save_setup
        )
        delete_setup_button = self._text_icon_button(
            "🗑 Delete setup", QtWidgets.QStyle.SP_TrashIcon, "Delete setup", self.delete_setup
        )
        validate_setup_button = self._text_icon_button(
            "✅ Validate setup", QtWidgets.QStyle.SP_DialogApplyButton, "Validate setup", self.validate_setup
        )
        buttons.addWidget(save_setup_button)
        buttons.addWidget(delete_setup_button)
        buttons.addWidget(validate_setup_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def fill_setup_table(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.setups_table):
            return
        self.setups_table.setRowCount(0)
        try:
            setups = self.client._call("mfdb.setups.list").get("setups", [])
        except Exception:
            setups = []
        for item in setups:
            row = self.setups_table.rowCount()
            self.setups_table.insertRow(row)
            lasers = item.get("laser_wavelengths", [])
            detectors = item.get("detector_channels", {})
            ld_str = f"Lasers: {lasers} | Detectors: {detectors}"
            values = [
                item.get("setup_id", ""),
                item.get("name", ""),
                item.get("instrument_type", ""),
                item.get("details", ""),
                ld_str
            ]
            for column, value in enumerate(values):
                self.setups_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))

    def load_setup(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.setups_table):
            return
        selected = self.setups_table.selectedItems()
        if not selected:
            self.setup_id_edit.clear()
            self.setup_name_edit.clear()
            self.setup_instrument_edit.clear()
            self.setup_lasers_edit.clear()
            self.setup_detectors_edit.clear()
            self.setup_details_edit.clear()
            self.setup_validation_label.clear()
            return
        row = selected[0].row()
        setup_id = self.setups_table.item(row, 0).text()
        try:
            setup = self.client._call("mfdb.setups.get", {"setup_id": setup_id}).get("setup", {})
        except Exception:
            setup = {}
        self.setup_id_edit.setText(setup.get("setup_id", ""))
        self.setup_name_edit.setText(setup.get("name", ""))
        self.setup_instrument_edit.setText(setup.get("instrument_type", ""))
        import json
        self.setup_lasers_edit.setText(json.dumps(setup.get("laser_wavelengths", [])))
        self.setup_detectors_edit.setText(json.dumps(setup.get("detector_channels", {})))
        self.setup_details_edit.setPlainText(setup.get("details", "") or "")
        self.setup_validation_label.clear()

    def collect_setup(self) -> dict[str, Any]:
        import json
        try:
            lasers = json.loads(self.setup_lasers_edit.text() or "[]")
        except Exception:
            lasers = []
        try:
            detectors = json.loads(self.setup_detectors_edit.text() or "{}")
        except Exception:
            detectors = {}
        return {
            "setup_id": self.setup_id_edit.text().strip(),
            "name": self.setup_name_edit.text().strip(),
            "instrument_type": self.setup_instrument_edit.text().strip() or None,
            "laser_wavelengths": lasers,
            "detector_channels": detectors,
            "details": self.setup_details_edit.toPlainText().strip() or None,
        }

    def save_setup(self) -> None:
        try:
            setup = self.collect_setup()
            self.client._call("mfdb.setups.save", {"setup": setup})
            QtWidgets.QMessageBox.information(self, "Setup Saved", "Setup definition successfully saved.")
            self.fill_setup_table()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Failed", f"Failed to save setup:\n{e}")

    def delete_setup(self) -> None:
        setup_id = self.setup_id_edit.text().strip()
        if not setup_id:
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete Setup",
            f"Are you sure you want to delete the setup definition '{setup_id}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.client._call("mfdb.setups.delete", {"setup_id": setup_id})
            QtWidgets.QMessageBox.information(self, "Setup Deleted", "Setup definition successfully deleted.")
            self.fill_setup_table()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Delete Failed", f"Failed to delete setup:\n{e}")

    def validate_setup(self) -> None:
        setup_id = self.setup_id_edit.text().strip()
        if not setup_id:
            return
        try:
            res = self.client._call("mfdb.setups.validate", {"setup_id": setup_id})
            valid = res.get("valid", False)
            errors = res.get("errors", [])
            if valid:
                self.setup_validation_label.setText("<font color='green'><b>Validation PASS</b>: Setup is valid and fully specified.</font>")
            else:
                err_str = "<br>".join(errors)
                self.setup_validation_label.setText(f"<font color='red'><b>Validation FAIL</b>:<br>{err_str}</font>")
        except Exception as e:
            self.setup_validation_label.setText(f"<font color='red'><b>Error validating setup</b>: {e}</font>")

    def raw_data_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.raw_data_table = QtWidgets.QTableWidget(0, 8)
        self.raw_data_table.setHorizontalHeaderLabels(
            ["raw data id", "experiment id", "data type", "storage mode", "path/url/folder", "validation", "checksum", "acquired at"]
        )
        self.raw_data_table.horizontalHeader().setStretchLastSection(True)
        self.raw_data_table.itemSelectionChanged.connect(self.load_raw_data)
        layout.addWidget(self.raw_data_table, stretch=1)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.raw_id_edit = QtWidgets.QLineEdit()
        self.raw_exp_edit = QtWidgets.QLineEdit()
        self.raw_type_edit = QtWidgets.QLineEdit()
        self.raw_storage_edit = QtWidgets.QLineEdit()
        self.raw_path_edit = QtWidgets.QLineEdit()
        self.raw_checksum_edit = QtWidgets.QLineEdit()
        self.raw_validation_edit = QtWidgets.QLineEdit()
        self.raw_details_edit = QtWidgets.QPlainTextEdit()
        self.raw_details_edit.setMinimumHeight(60)

        for w in (self.raw_id_edit, self.raw_exp_edit, self.raw_type_edit, self.raw_storage_edit, self.raw_path_edit, self.raw_checksum_edit, self.raw_validation_edit, self.raw_details_edit):
            w.setReadOnly(True)

        form.addRow("Raw Data ID", self.raw_id_edit)
        form.addRow("Experiment ID", self.raw_exp_edit)
        form.addRow("Data Type", self.raw_type_edit)
        form.addRow("Storage Mode", self.raw_storage_edit)
        form.addRow("File Path/URL/Folder", self.raw_path_edit)
        form.addRow("Validation Status", self.raw_validation_edit)
        form.addRow("Checksum (SHA-256)", self.raw_checksum_edit)
        form.addRow("Details/JSON", self.raw_details_edit)
        layout.addLayout(form)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_open = self._text_icon_button(
            "📂 Open", QtWidgets.QStyle.SP_DialogOpenButton, "Open raw data file/URL", self._on_raw_open_clicked
        )
        btn_copy = self._text_icon_button(
            "📋 Copy ID", QtWidgets.QStyle.SP_FileIcon, "Copy raw data ID to clipboard", self._on_raw_copy_clicked
        )
        btn_seed = self._text_icon_button(
            "🌱 Use as provenance seed",
            QtWidgets.QStyle.SP_ArrowRight,
            "Load this record into the provenance graph",
            self._on_raw_seed_clicked,
        )
        btn_layout.addWidget(btn_open)
        btn_layout.addWidget(btn_copy)
        btn_layout.addWidget(btn_seed)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        return widget

    def fill_raw_data_table(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.raw_data_table):
            return
        self.raw_data_table.setRowCount(0)
        try:
            raw_list = self.client.list_raw_data(experiment_id=self.current_experiment_id)
        except Exception:
            raw_list = []
        for item in raw_list:
            row = self.raw_data_table.rowCount()
            self.raw_data_table.insertRow(row)
            values = [
                item.get("raw_data_id", ""),
                item.get("experiment_id", ""),
                item.get("data_type", ""),
                item.get("storage_mode", ""),
                item.get("file_path") or item.get("url") or item.get("folder_path") or "",
                item.get("validation_status", ""),
                item.get("checksum", ""),
                item.get("acquired_at", "")
            ]
            for column, value in enumerate(values):
                self.raw_data_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))

    def load_raw_data(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.raw_data_table):
            return
        selected = self.raw_data_table.selectedItems()
        if not selected:
            self.raw_id_edit.clear()
            self.raw_exp_edit.clear()
            self.raw_type_edit.clear()
            self.raw_storage_edit.clear()
            self.raw_path_edit.clear()
            self.raw_checksum_edit.clear()
            self.raw_validation_edit.clear()
            self.raw_details_edit.clear()
            return
        row = selected[0].row()
        raw_id = self.raw_data_table.item(row, 0).text()
        try:
            item = self.client.get_raw_data(raw_id)
        except Exception:
            item = {}
        self.raw_id_edit.setText(item.get("raw_data_id", ""))
        self.raw_exp_edit.setText(item.get("experiment_id", ""))
        self.raw_type_edit.setText(item.get("data_type", ""))
        self.raw_storage_edit.setText(item.get("storage_mode", ""))
        self.raw_path_edit.setText(_processed_location(item))
        self.raw_checksum_edit.setText(item.get("checksum", ""))
        self.raw_validation_edit.setText(item.get("validation_status", ""))
        self.raw_details_edit.setPlainText(json.dumps(item, indent=2))

    def _on_raw_open_clicked(self) -> None:
        path = self.raw_path_edit.text().strip()
        if path:
            QtGui.QDesktopServices.openUrl(_qurl_for_location(path))

    def _on_raw_copy_clicked(self) -> None:
        raw_id = self.raw_id_edit.text().strip()
        if raw_id:
            QtWidgets.QApplication.clipboard().setText(raw_id)

    def _on_raw_seed_clicked(self) -> None:
        raw_id = self.raw_id_edit.text().strip()
        if raw_id:
            self._set_provenance_seed("raw_data", raw_id)

    def _show_provenance_graph_dock(self) -> None:
        """Show and activate the provenance graph dock."""
        if not hasattr(self, "_provenance_graph_widget"):
            return
        index = self.tabs.indexOf(self._provenance_graph_widget)
        if index >= 0 and not self.tabs.isTabVisible(index):
            self.tabs.showTab(index)
        self.tabs.setCurrentWidget(self._provenance_graph_widget)

    def _set_provenance_seed(self, seed_type: str, seed_id: str) -> None:
        self.current_provenance_seed_type = seed_type
        self.current_provenance_seed_id = seed_id
        self.prov_seed_id_edit.setText(seed_id)
        self.prov_seed_type_combo.setCurrentText(seed_type)
        self._show_provenance_graph_dock()

    def processing_runs_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.processing_runs_table = QtWidgets.QTableWidget(0, 8)
        self.processing_runs_table.setHorizontalHeaderLabels(
            ["processing id", "experiment id", "type", "started at", "status", "raw count", "product count", "operator"]
        )
        self.processing_runs_table.horizontalHeader().setStretchLastSection(True)
        self.processing_runs_table.itemSelectionChanged.connect(self.load_processing_run)
        layout.addWidget(self.processing_runs_table, stretch=1)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.proc_id_edit = QtWidgets.QLineEdit()
        self.proc_exp_edit = QtWidgets.QLineEdit()
        self.proc_type_edit = QtWidgets.QLineEdit()
        self.proc_status_edit = QtWidgets.QLineEdit()
        self.proc_settings_edit = QtWidgets.QPlainTextEdit()
        self.proc_settings_edit.setMinimumHeight(60)

        for w in (self.proc_id_edit, self.proc_exp_edit, self.proc_type_edit, self.proc_status_edit, self.proc_settings_edit):
            w.setReadOnly(True)

        form.addRow("Processing ID", self.proc_id_edit)
        form.addRow("Experiment ID", self.proc_exp_edit)
        form.addRow("Type", self.proc_type_edit)
        form.addRow("Status", self.proc_status_edit)
        form.addRow("Settings/JSON", self.proc_settings_edit)
        layout.addLayout(form)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_copy = self._text_icon_button(
            "📋 Copy ID", QtWidgets.QStyle.SP_FileIcon, "Copy processing ID to clipboard", self._on_proc_copy_clicked
        )
        btn_seed = self._text_icon_button(
            "🌱 Use as provenance seed",
            QtWidgets.QStyle.SP_ArrowRight,
            "Load this run into the provenance graph",
            self._on_proc_seed_clicked,
        )
        btn_layout.addWidget(btn_copy)
        btn_layout.addWidget(btn_seed)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        return widget

    def fill_processing_runs_table(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.processing_runs_table):
            return
        self.processing_runs_table.setRowCount(0)
        try:
            proc_list = self.client.list_processing_runs(experiment_id=self.current_experiment_id)
        except Exception:
            proc_list = []
        for item in proc_list:
            row = self.processing_runs_table.rowCount()
            self.processing_runs_table.insertRow(row)
            values = [
                item.get("processing_id", ""),
                item.get("experiment_id", ""),
                item.get("processing_type", ""),
                item.get("started_at", ""),
                item.get("status", ""),
                item.get("input_raw_count", ""),
                item.get("output_product_count", ""),
                item.get("operator_user_id", "")
            ]
            for column, value in enumerate(values):
                self.processing_runs_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))

    def load_processing_run(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.processing_runs_table):
            return
        selected = self.processing_runs_table.selectedItems()
        if not selected:
            self.proc_id_edit.clear()
            self.proc_exp_edit.clear()
            self.proc_type_edit.clear()
            self.proc_status_edit.clear()
            self.proc_settings_edit.clear()
            return
        row = selected[0].row()
        proc_id = self.processing_runs_table.item(row, 0).text()
        try:
            item = self.client.get_processing_run(proc_id)
        except Exception:
            item = {}
        self.proc_id_edit.setText(item.get("processing_id", ""))
        self.proc_exp_edit.setText(item.get("experiment_id", ""))
        self.proc_type_edit.setText(item.get("processing_type", ""))
        self.proc_status_edit.setText(item.get("status", ""))
        self.proc_settings_edit.setPlainText(json.dumps(item, indent=2))

    def _on_proc_copy_clicked(self) -> None:
        proc_id = self.proc_id_edit.text().strip()
        if proc_id:
            QtWidgets.QApplication.clipboard().setText(proc_id)

    def _on_proc_seed_clicked(self) -> None:
        proc_id = self.proc_id_edit.text().strip()
        if proc_id:
            self._set_provenance_seed("processing_run", proc_id)

    def processed_products_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Product type filter:"))
        self.prod_filter_combo = QtWidgets.QComboBox()
        self.prod_filter_combo.addItems([
            "all", "bur", "tcspc_decay", "fcs_correlation", "pda_histogram",
            "irf_curve", "hdf5", "zip", "gmm_summary", "spectra", "fit_results"
        ])
        self.prod_filter_combo.currentTextChanged.connect(self.fill_processed_products_table)
        filter_layout.addWidget(self.prod_filter_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        self.processed_products_table = QtWidgets.QTableWidget(0, 8)
        self.processed_products_table.setHorizontalHeaderLabels(
            ["product id", "processing id", "product type", "storage mode", "path/url/folder", "validation", "checksum", "row count"]
        )
        self.processed_products_table.horizontalHeader().setStretchLastSection(True)
        self.processed_products_table.itemSelectionChanged.connect(self.load_processed_product)
        layout.addWidget(self.processed_products_table, stretch=1)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.prod_id_edit = QtWidgets.QLineEdit()
        self.prod_proc_edit = QtWidgets.QLineEdit()
        self.prod_type_edit = QtWidgets.QLineEdit()
        self.prod_storage_edit = QtWidgets.QLineEdit()
        self.prod_path_edit = QtWidgets.QLineEdit()
        self.prod_validation_edit = QtWidgets.QLineEdit()
        self.prod_checksum_edit = QtWidgets.QLineEdit()
        self.prod_exp_edit = QtWidgets.QLineEdit()

        for w in (self.prod_id_edit, self.prod_proc_edit, self.prod_type_edit, self.prod_storage_edit, self.prod_path_edit, self.prod_validation_edit, self.prod_checksum_edit, self.prod_exp_edit):
            w.setReadOnly(True)

        form.addRow("Product ID", self.prod_id_edit)
        form.addRow("Processing Run ID", self.prod_proc_edit)
        form.addRow("Product Type", self.prod_type_edit)
        form.addRow("Storage Mode", self.prod_storage_edit)
        form.addRow("File Path/URL/Folder", self.prod_path_edit)
        form.addRow("Validation Status", self.prod_validation_edit)
        form.addRow("Checksum (SHA-256)", self.prod_checksum_edit)
        form.addRow("Experiment ID", self.prod_exp_edit)
        layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)

        btn_open = self._text_icon_button(
            "📂 Open", QtWidgets.QStyle.SP_DialogOpenButton, "Open processed product", self._on_prod_open_clicked
        )
        buttons.addWidget(btn_open)

        ndx_button = self._text_icon_button(
            "🔬 Open in NDXplorer",
            QtWidgets.QStyle.SP_FileDialogContentsView,
            "Open the selected product in NDXplorer",
            self.open_in_ndxplorer,
        )
        buttons.addWidget(ndx_button)

        btn_copy = self._text_icon_button(
            "📋 Copy ID", QtWidgets.QStyle.SP_FileIcon, "Copy product ID to clipboard", self._on_prod_copy_clicked
        )
        buttons.addWidget(btn_copy)

        btn_seed = self._text_icon_button(
            "🌱 Use as provenance seed",
            QtWidgets.QStyle.SP_ArrowRight,
            "Load this product into the provenance graph",
            self._on_prod_seed_clicked,
        )
        buttons.addWidget(btn_seed)

        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def fill_processed_products_table(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.processed_products_table):
            return
        self.processed_products_table.setRowCount(0)
        try:
            prod_list = self.client.list_processed_data()
            prod_list = _scope_processed_products_by_experiment(
                prod_list,
                self.client,
                self.current_experiment_id,
            )
            filt = self.prod_filter_combo.currentText()
            if filt != "all":
                prod_list = [p for p in prod_list if p.get("product_type") == filt]
        except Exception:
            prod_list = []
        for item in prod_list:
            row = self.processed_products_table.rowCount()
            self.processed_products_table.insertRow(row)
            values = [
                item.get("processed_data_id", ""),
                item.get("processing_id", ""),
                item.get("product_type", ""),
                item.get("storage_mode", ""),
                _processed_location(item),
                item.get("validation_status", ""),
                item.get("checksum", ""),
                _processed_row_count(item)
            ]
            for column, value in enumerate(values):
                self.processed_products_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))

    def load_processed_product(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.processed_products_table):
            return
        selected = self.processed_products_table.selectedItems()
        if not selected:
            self.prod_id_edit.clear()
            self.prod_proc_edit.clear()
            self.prod_type_edit.clear()
            self.prod_storage_edit.clear()
            self.prod_path_edit.clear()
            self.prod_validation_edit.clear()
            self.prod_checksum_edit.clear()
            self.prod_exp_edit.clear()
            return
        row = selected[0].row()
        prod_id = self.processed_products_table.item(row, 0).text()
        try:
            item = self.client.get_processed_data(prod_id)
        except Exception:
            item = {}
        self.prod_id_edit.setText(item.get("processed_data_id", ""))
        processing_id = item.get("processing_id", "")
        self.prod_proc_edit.setText(processing_id)
        self.prod_type_edit.setText(item.get("product_type", ""))
        self.prod_storage_edit.setText(item.get("storage_mode", ""))
        self.prod_path_edit.setText(_processed_location(item))
        self.prod_validation_edit.setText(item.get("validation_status", ""))
        self.prod_checksum_edit.setText(item.get("checksum", ""))
        self.prod_exp_edit.setText(_experiment_id_for_processing_id(self.client, processing_id))

    def _on_prod_open_clicked(self) -> None:
        path = self.prod_path_edit.text().strip()
        if path:
            QtGui.QDesktopServices.openUrl(_qurl_for_location(path))

    def _on_prod_copy_clicked(self) -> None:
        prod_id = self.prod_id_edit.text().strip()
        if prod_id:
            QtWidgets.QApplication.clipboard().setText(prod_id)

    def _on_prod_seed_clicked(self) -> None:
        prod_id = self.prod_id_edit.text().strip()
        if prod_id:
            self._set_provenance_seed("processed_data", prod_id)

    def open_in_ndxplorer(self) -> None:
        selected = self.processed_products_table.selectedItems()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a processed product.")
            return
        row = selected[0].row()
        prod_id = self.processed_products_table.item(row, 0).text()
        path_str = self.processed_products_table.item(row, 4).text()
        exp_id = self.prod_exp_edit.text().strip()
        if not path_str:
            QtWidgets.QMessageBox.warning(self, "No Path", "Selected product has no associated file path.")
            return

        from pathlib import Path
        path = Path(path_str)
        if not path.exists():
            QtWidgets.QMessageBox.critical(self, "File Not Found", f"The file or directory does not exist:\n{path_str}")
            return

        import sys
        root = Path(__file__).resolve().parents[4]
        ndx_path = root / "modules" / "ndxplorer"
        if ndx_path.is_dir() and str(ndx_path) not in sys.path:
            sys.path.insert(0, str(ndx_path))

        try:
            import ndxplorer.io.reader as ndx_reader
            from ndxplorer.plot_main import NDXplorer

            if path.is_dir():
                ds = ndx_reader.read_burst_analysis(str(path))
                ndx = NDXplorer(
                    data_source=ds,
                    zmq_cmd_port=8765,
                    processed_data_id=prod_id,
                    experiment_id=exp_id,
                )
                ndx.working_path = str(path)
                ndx.setWindowTitle(f"NDXplorer - {path.name}")
                ndx.show()
                ndx.raise_()
                ndx.activateWindow()
                try:
                    ndx.open_files(file_handles=str(path), file_type="burst_dir", append=False)
                except Exception:
                    pass
            else:
                ndx = NDXplorer(
                    zmq_cmd_port=8765,
                    processed_data_id=prod_id,
                    experiment_id=exp_id,
                )
                ndx.setWindowTitle(f"NDXplorer - {path.name}")
                ndx.show()
                ndx.raise_()
                ndx.activateWindow()
                file_type = "h5" if path.suffix == ".h5" else "zip"
                try:
                    ndx.open_files(file_handles=str(path), file_type=file_type, append=False)
                except Exception:
                    pass

            if not hasattr(self, "_ndxplorer_windows"):
                self._ndxplorer_windows = []
            self._ndxplorer_windows.append(ndx)
            self.statusBar().showMessage(f"Opened {path.name} in NDXplorer")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open in NDXplorer:\n{e}")

    def analyses_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.analyses_table = QtWidgets.QTableWidget(0, 5)
        self.analyses_table.setHorizontalHeaderLabels(
            ["analysis id", "experiment id", "type", "model name", "created at"]
        )
        self.analyses_table.horizontalHeader().setStretchLastSection(True)
        self.analyses_table.itemSelectionChanged.connect(self.load_analysis)
        layout.addWidget(self.analyses_table, stretch=1)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self.analysis_id_field = QtWidgets.QLineEdit()
        self.analysis_exp_field = QtWidgets.QLineEdit()
        self.analysis_type_field = QtWidgets.QLineEdit()
        self.analysis_model_field = QtWidgets.QLineEdit()
        self.analysis_settings_field = QtWidgets.QPlainTextEdit()
        self.analysis_settings_field.setMinimumHeight(60)

        for w in (self.analysis_id_field, self.analysis_exp_field, self.analysis_type_field, self.analysis_model_field, self.analysis_settings_field):
            w.setReadOnly(True)

        form.addRow("Analysis ID", self.analysis_id_field)
        form.addRow("Experiment ID", self.analysis_exp_field)
        form.addRow("Type", self.analysis_type_field)
        form.addRow("Model Name", self.analysis_model_field)
        form.addRow("Settings/JSON", self.analysis_settings_field)
        layout.addLayout(form)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_copy = self._text_icon_button(
            "📋 Copy ID", QtWidgets.QStyle.SP_FileIcon, "Copy analysis ID to clipboard", self._on_analysis_copy_clicked
        )
        btn_seed = self._text_icon_button(
            "🌱 Use as provenance seed",
            QtWidgets.QStyle.SP_ArrowRight,
            "Load this analysis into the provenance graph",
            self._on_analysis_seed_clicked,
        )
        btn_layout.addWidget(btn_copy)
        btn_layout.addWidget(btn_seed)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        return widget

    def fill_analyses_table(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.analyses_table):
            return
        self.analyses_table.setRowCount(0)
        try:
            analysis_list = self.client.list_analysis_runs(experiment_id=self.current_experiment_id)
        except Exception:
            analysis_list = []
        for item in analysis_list:
            if item.get("analysis_type") == "project":
                continue
            row = self.analyses_table.rowCount()
            self.analyses_table.insertRow(row)
            values = [
                item.get("analysis_id", ""),
                item.get("experiment_id", ""),
                item.get("analysis_type", ""),
                item.get("model_name", ""),
                item.get("created_at", "")
            ]
            for column, value in enumerate(values):
                self.analyses_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))

    def load_analysis(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.analyses_table):
            return
        selected = self.analyses_table.selectedItems()
        if not selected:
            self.analysis_id_field.clear()
            self.analysis_exp_field.clear()
            self.analysis_type_field.clear()
            self.analysis_model_field.clear()
            self.analysis_settings_field.clear()
            return
        row = selected[0].row()
        analysis_id = self.analyses_table.item(row, 0).text()
        try:
            item = self.client.get_analysis_run(analysis_id)
        except Exception:
            item = {}
        self.analysis_id_field.setText(item.get("analysis_id", ""))
        self.analysis_exp_field.setText(item.get("experiment_id", ""))
        self.analysis_type_field.setText(item.get("analysis_type", ""))
        self.analysis_model_field.setText(item.get("model_name", ""))
        self.analysis_settings_field.setPlainText(json.dumps(item, indent=2))

    def _on_analysis_copy_clicked(self) -> None:
        analysis_id = self.analysis_id_field.text().strip()
        if analysis_id:
            QtWidgets.QApplication.clipboard().setText(analysis_id)

    def _on_analysis_seed_clicked(self) -> None:
        analysis_id = self.analysis_id_field.text().strip()
        if analysis_id:
            self._set_provenance_seed("analysis_run", analysis_id)

    def provenance_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        self._provenance_widget = widget
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(QtWidgets.QLabel("Select a raw data, processing run, processed product, or analysis record, then use \"Use as provenance seed\" to load its graph."))

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.addWidget(QtWidgets.QLabel("Seed Type:"))
        self.prov_seed_type_combo = QtWidgets.QComboBox()
        self.prov_seed_type_combo.addItems(["raw_data", "processing_run", "processed_data", "analysis_run", "analysis_parameter"])
        toolbar.addWidget(self.prov_seed_type_combo)

        toolbar.addWidget(QtWidgets.QLabel("Seed ID:"))
        self.prov_seed_id_edit = QtWidgets.QLineEdit()
        toolbar.addWidget(self.prov_seed_id_edit)

        self.btn_load_upstream = self._text_icon_button(
            "⬆️ Load upstream",
            QtWidgets.QStyle.SP_ArrowUp,
            "Trace ancestors of the seed",
            self.load_provenance_upstream,
        )
        toolbar.addWidget(self.btn_load_upstream)

        self.btn_load_downstream = self._text_icon_button(
            "⬇️ Load downstream",
            QtWidgets.QStyle.SP_ArrowDown,
            "Trace descendants of the seed",
            self.load_provenance_downstream,
        )
        toolbar.addWidget(self.btn_load_downstream)

        self.btn_load_full = self._text_icon_button(
            "🕸 Load full graph",
            QtWidgets.QStyle.SP_FileDialogContentsView,
            "Load the full provenance graph for the seed",
            self.load_provenance_full_graph,
        )
        toolbar.addWidget(self.btn_load_full)

        self.btn_export_json = self._text_icon_button(
            "📄 Export JSON",
            QtWidgets.QStyle.SP_DialogSaveButton,
            "Export provenance graph as JSON",
            self.export_provenance_json_action,
        )
        toolbar.addWidget(self.btn_export_json)

        self.btn_export_zip = self._text_icon_button(
            "📦 Export ZIP",
            QtWidgets.QStyle.SP_DriveHDIcon,
            "Export provenance graph as ZIP archive",
            self.export_provenance_zip_action,
        )
        toolbar.addWidget(self.btn_export_zip)

        layout.addLayout(toolbar)
        layout.addStretch()
        return widget

    def provenance_graph_dock(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        self._provenance_graph_widget = widget
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.prov_edge_table = QtWidgets.QTableWidget(0, 7)
        self.prov_edge_table.setHorizontalHeaderLabels([
            "edge id", "source type", "source id", "relationship", "target type", "target id", "processing id"
        ])
        self.prov_edge_table.horizontalHeader().setStretchLastSection(True)
        self.prov_edge_table.itemSelectionChanged.connect(self.on_prov_edge_selected)
        left_splitter.addWidget(self.prov_edge_table)

        from chisurf.gui.widgets.node_editor.editor import NodeEditorWidget
        self.prov_node_editor = NodeEditorWidget(
            build_example=False, show_side_panel=False, show_timeline=False, read_only=True, graph_purpose="provenance_view"
        )
        self.prov_node_editor.nodeSelected.connect(self.on_prov_node_selected_in_editor)
        self.prov_node_editor.edgeSelected.connect(self.on_prov_edge_selected_in_editor)
        left_splitter.addWidget(self.prov_node_editor)

        main_splitter.addWidget(left_splitter)

        self.prov_details_text = QtWidgets.QPlainTextEdit()
        self.prov_details_text.setReadOnly(True)
        main_splitter.addWidget(self.prov_details_text)

        layout.addWidget(main_splitter, stretch=1)
        return widget

    def on_prov_node_selected_in_editor(self, node_dict: dict) -> None:
        self.prov_details_text.setPlainText(json.dumps(node_dict, indent=2))

    def on_prov_edge_selected_in_editor(self, edge_dict: dict) -> None:
        self.prov_details_text.setPlainText(json.dumps(edge_dict, indent=2))

    def on_prov_edge_selected(self) -> None:
        selected = self.prov_edge_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        item = self.prov_edge_table.item(row, 0)
        if item is not None:
            edge_data = item.data(QtCore.Qt.UserRole)
            self.prov_details_text.setPlainText(json.dumps(edge_data, indent=2))

    def load_provenance_upstream(self) -> None:
        seed_id = self.prov_seed_id_edit.text().strip()
        seed_type = self.prov_seed_type_combo.currentText()
        if not seed_id:
            QtWidgets.QMessageBox.warning(self, "No Seed ID", "Please enter a seed node ID.")
            return
        try:
            res = self.client.dependencies_upstream(node_type=seed_type, node_id=seed_id)
            self._display_provenance_graph(res)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Trace Failed", f"Failed to trace upstream:\n{e}")

    def load_provenance_downstream(self) -> None:
        seed_id = self.prov_seed_id_edit.text().strip()
        seed_type = self.prov_seed_type_combo.currentText()
        if not seed_id:
            QtWidgets.QMessageBox.warning(self, "No Seed ID", "Please enter a seed node ID.")
            return
        try:
            res = self.client.dependencies_downstream(node_type=seed_type, node_id=seed_id)
            self._display_provenance_graph(res)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Trace Failed", f"Failed to trace downstream:\n{e}")

    def load_provenance_full_graph(self) -> None:
        seed_id = self.prov_seed_id_edit.text().strip()
        seed_type = self.prov_seed_type_combo.currentText()
        if not seed_id:
            QtWidgets.QMessageBox.warning(self, "No Seed ID", "Please enter a seed node ID.")
            return
        try:
            res = self.client.export_provenance_graph(seed_node_type=seed_type, seed_node_id=seed_id)
            self._display_provenance_graph(_unwrap_provenance_graph_response(res))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Failed", f"Failed to load full graph:\n{e}")

    def _display_provenance_graph(self, graph: dict) -> None:
        from chisurf.plugins.core.mfdb_admin.gui.provenance_graph import mfdb_graph_to_node_editor_graph
        ne_graph = mfdb_graph_to_node_editor_graph(graph)
        self.prov_node_editor.load_graph_dict(ne_graph)
        self._show_provenance_graph_dock()

        # Populate edge table
        self.prov_edge_table.setRowCount(0)
        edges = graph.get("edges", [])
        for e in edges:
            row = self.prov_edge_table.rowCount()
            self.prov_edge_table.insertRow(row)
            self.prov_edge_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(e.get("edge_id") or "")))
            self.prov_edge_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(e.get("source_node_type") or "")))
            self.prov_edge_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(e.get("source_node_id") or "")))
            self.prov_edge_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(e.get("relationship_type") or "")))
            self.prov_edge_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(e.get("target_node_type") or "")))
            self.prov_edge_table.setItem(row, 5, QtWidgets.QTableWidgetItem(str(e.get("target_node_id") or "")))
            self.prov_edge_table.setItem(row, 6, QtWidgets.QTableWidgetItem(str(e.get("processing_id") or "")))
            self.prov_edge_table.item(row, 0).setData(QtCore.Qt.UserRole, e)

    def export_provenance_json_action(self) -> None:
        seed_id = self.prov_seed_id_edit.text().strip()
        seed_type = self.prov_seed_type_combo.currentText()
        if not seed_id:
            QtWidgets.QMessageBox.warning(self, "No Seed ID", "Please enter a seed node ID.")
            return
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Provenance JSON", "", "JSON Files (*.json)"
        )
        if not path_str:
            return
        try:
            res = self.client.export_provenance_graph(seed_node_type=seed_type, seed_node_id=seed_id)
            with open(path_str, "w") as f:
                json.dump(res, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Exported successfully to:\n{path_str}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Failed", f"Failed to export provenance JSON:\n{e}")

    def export_provenance_zip_action(self) -> None:
        seed_id = self.prov_seed_id_edit.text().strip()
        seed_type = self.prov_seed_type_combo.currentText()
        if not seed_id:
            QtWidgets.QMessageBox.warning(self, "No Seed ID", "Please enter a seed node ID.")
            return
        path_str, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Provenance ZIP Archive", "", "ZIP Archives (*.zip)"
        )
        if not path_str:
            return
        try:
            self.client.export_zip_archive(
                target_zip_path=path_str,
                seed_node_type=seed_type,
                seed_node_id=seed_id,
                include_external_data=False
            )
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Exported successfully to:\n{path_str}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Failed", f"Failed to export zip archive:\n{e}")
