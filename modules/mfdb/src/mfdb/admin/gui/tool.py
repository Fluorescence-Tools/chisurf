"""GUI plugin for managing the Multiparametric Fluorescence Database."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Callable

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
from chisurf.gui.widgets.general import apply_compact_table_style
from chisurf.gui.widgets.metadata_editor import MetadataEditor
from chisurf.gui.widgets.navigation import NavigationPanelTool
from mfdb.models import (
    COMMON_PROBE_NAMES,
    DEFAULT_FLUOROPHORE_SPECTRA,
    ENTITY_TYPES,
    VALIDATION_STATUS_VALUES,
)

import chisurf.logging

from .client import MFDBClient
from .generic_form import MFDBDetailWidget
from .entity_registry import ENTITY_REGISTRY, EntitySpec, build_registry_dict
from .entity_dock import EntityDock
from .metadata_dock import MetadataDock
from .studies_view import StudiesView
from .protocols_view import ProtocolsView
from .lifecycle_view import LifecycleView
from .calibrations_view import CalibrationsView
from .reagents_view import ReagentLotsView
from .pipelines_view import PipelinesView
from mfdb.schema.pdbx_metadata import MmcifDictionary
from mfdb.schema.dictionary_schema_map import DictionarySchemaMap, build_dictionary_schema_map


class _MFDBBackgroundTask(QtCore.QObject):
    """Run one blocking MFDB callable in a QThread."""

    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self._fn = fn

    @QtCore.Slot()
    def run(self) -> None:
        """Execute the task and emit its result."""
        try:
            self.finished.emit(self._fn())
        except Exception as exc:
            self.failed.emit(str(exc))


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


def _sample_quality_table_item(item: dict[str, Any]) -> QtWidgets.QTableWidgetItem:
    """Return a formatted table item for measurement sample metadata quality."""
    status = str(item.get("sample_quality_status") or "red").lower()
    score = item.get("sample_quality_score", 0)
    label = str(item.get("sample_quality_summary") or f"{status.upper()} {score}")
    table_item = QtWidgets.QTableWidgetItem(label)
    colors = {
        "green": QtGui.QColor(214, 245, 220),
        "yellow": QtGui.QColor(255, 244, 204),
        "red": QtGui.QColor(255, 220, 220),
    }
    table_item.setBackground(colors.get(status, colors["red"]))
    flags = item.get("sample_quality_flags") or []
    sample_id = item.get("sample_id") or "none"
    sample_name = item.get("sample_name") or ""
    tooltip_lines = [
        f"Sample: {sample_name or sample_id}",
        f"Status: {status.upper()}",
        f"Score: {score}",
    ]
    if flags:
        tooltip_lines.append("Flags:")
        tooltip_lines.extend(f"- {flag}" for flag in flags)
    table_item.setToolTip("\n".join(tooltip_lines))
    return table_item


def _property_value(properties: dict[str, Any], *names: str) -> Any:
    """Return the first matching optical-property scalar value."""
    for name in names:
        value = properties.get(name)
        if isinstance(value, dict):
            return value.get("value", "")
        if value not in (None, ""):
            return value
    return ""


def _table_text(table: QtWidgets.QTableWidget, row: int, column: int) -> str:
    """Return stripped table item text for a row/column."""
    item = table.item(row, column)
    return item.text().strip() if item else ""


def _int_or_none(value: Any) -> int | None:
    """Return an integer for non-empty values, otherwise None."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    """Return a float for non-empty values, otherwise None."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


class MFDBWidget(NavigationPanelTool):
    """Window for browsing, editing, importing, and exporting samples.

    Uses the shared left-nav / right-panel :class:`NavigationPanelTool` shell so
    the tool matches plugin:setup and the FCS/burst tools. Plain-Python state is
    prepared *before* ``super().__init__`` because the shell auto-loads panel 0
    during construction.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None, client: Any | None = None):
        # -- prep that does not need a live QWidget (must precede super()) --
        client_was_provided = client is not None
        self.client = client or MFDBClient()
        self._loading = False
        self._auth_login_user: str | None = None
        self._checkable_tables: list[QtWidgets.QTableWidget] = []
        self._mock_data_click_count = 0
        self._background_tasks: list[tuple[QtCore.QThread, _MFDBBackgroundTask]] = []
        self._mock_data_task_running = False
        self._loaded_tabs: set[str] = set()
        # Panel-0 (Overview) is loaded during super().__init__, before refresh()
        # initialises these, so seed them here.
        self._failures: list[str] = []
        self._refresh_in_progress = False

        # Selection state
        self.current_sample_id = None
        self.current_experiment_id = None
        self.current_provenance_seed_type = "processed_data"
        self.current_provenance_seed_id = None

        # Dictionary and schema map for dictionary-driven docks
        self._dictionary: MmcifDictionary | None = None
        self._schema_map: DictionarySchemaMap | None = None
        self._registry_dict: dict[str, Any] = build_registry_dict()
        self._entity_docks: dict[str, Any] = {}
        try:
            self._dictionary = MmcifDictionary.load_bundled()
            self._schema_map = build_dictionary_schema_map()
        except Exception:
            chisurf.logging.warning("Failed to load mmCIF dictionary for admin GUI", exc_info=True)

        # Status label is wired to entity-dock signals built lazily by factories.
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)

        # Build the nav shell (this auto-loads panel 0 — see Overview factory).
        super().__init__(
            title="mfdb-admin — Multiparametric Fluorescence Database",
            panels=self._build_panels(),
            parent=parent,
            minimum_size=(900, 560),
            initial_size=(1180, 780),
            navigation_width=230,
        )

        # -- everything below needs a live QWidget (post-super) --
        self._verify_admin_access()
        self._ensure_authenticated()
        self._row_by_entity = {
            p["entity_key"]: i for i, p in enumerate(self.panels) if p.get("entity_key")
        }
        self._row_by_name = {
            p.get("name"): i for i, p in enumerate(self.panels) if not p.get("separator")
        }

        # Initialize generic detail widgets
        self.user_detail_widget = MFDBDetailWidget("user", parent=self)
        self.device_detail_widget = MFDBDetailWidget("device", parent=self)
        
        sample_providers = {
            "measured_by_user_id": lambda: [(u["user_id"], u.get("display_name", u["user_id"])) for u in self.client.list_users()],
            "measured_by_device_id": lambda: [(d["device_id"], d["name"]) for d in self.client.list_devices()],
        }
        self.sample_detail_widget = MFDBDetailWidget("sample", dropdown_providers=sample_providers, parent=self)

        experiment_providers = {
            "type_id": lambda: [(t["type_id"], t["name"]) for t in self.client.list_experiment_types()],
            "sample_id": lambda: [(s["sample_id"], s.get("description") or s["sample_id"]) for s in self.client.list_samples()],
            "measured_by_user_id": lambda: [(u["user_id"], u.get("display_name", u["user_id"])) for u in self.client.list_users()],
            "measured_by_device_id": lambda: [(d["device_id"], d["name"]) for d in self.client.list_devices()],
            "setup_definition_id": lambda: [(s["setup_id"], s["name"]) for s in self.client._call("mfdb.setups.list").get("setups", [])],
        }
        self.experiment_detail_widget = MFDBDetailWidget("experiment", dropdown_providers=experiment_providers, parent=self)
        self.experiment_type_detail_widget = MFDBDetailWidget("experiment_type", parent=self)
        self.branch_detail_widget = MFDBDetailWidget("branch", parent=self)
        self.probe_detail_widget = MFDBDetailWidget("probe", parent=self)
        self.setup_detail_widget = MFDBDetailWidget("setup", parent=self)
        self.project_detail_widget = MFDBDetailWidget("project", parent=self)
        self.raw_data_detail_widget = MFDBDetailWidget("raw_data", parent=self)
        self.processing_run_detail_widget = MFDBDetailWidget("processing_run", parent=self)
        self.processed_product_detail_widget = MFDBDetailWidget("processed_product", parent=self)
        self.object_detail_widget = MFDBDetailWidget("object", parent=self)
        self.analysis_detail_widget = MFDBDetailWidget("analysis", parent=self)
        self.condition_detail_widget = MFDBDetailWidget("condition", parent=self)

        self.setup_ui()
        self.setup_menu_bar()
        self.setup_toolbar()
        self.setup_status_bar()
        self._update_login_actions(logged_in=bool(getattr(self.client, "token", None)))
        if client_was_provided:
            self.refresh()
        else:
            self._set_status_message("Loading MFDB overview...")
            QtCore.QTimer.singleShot(0, self.refresh)

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
        from mfdb.admin.gui.session import (
            cache_session,
            cached_token,
            cached_user,
        )

        if getattr(self.client, "token", None):
            self._auth_login_user = getattr(self.client, "_auth_user_id", None)
            return

        # SSO: reuse a session token cached earlier in this ChiSurf process
        # (e.g. a previous mfdb-admin login). No password prompt if already
        # authenticated this session.
        _host = getattr(self.client, "host", "127.0.0.1")
        _cmd = getattr(self.client, "cmd_port", 8765)
        _pub = getattr(self.client, "pub_port", 8766)
        token = cached_token(_host, _cmd, _pub)
        if token:
            self.client.token = token
            self._auth_login_user = cached_user()
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
            cache_session(user_id, getattr(self.client, "token", None), _host, _cmd, _pub)
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
            
        # No password provided or it failed — prompt with the AutoForm login
        # dialog (single-line password; user/host/ports under "Advanced").
        from mfdb.admin.gui.connection_dialog import ConnectionAuthDialog

        for _ in range(3):
            dialog = ConnectionAuthDialog(
                user=user_id,
                host=getattr(self.client, "host", "127.0.0.1"),
                cmd_port=getattr(self.client, "cmd_port", 8765),
                pub_port=getattr(self.client, "pub_port", 8766),
                parent=self,
            )
            if dialog.exec_() != QtWidgets.QDialog.Accepted:
                return
            values = dialog.values()
            user_id = values["user"] or user_id

            # Reconnect to a different server if the endpoint was changed.
            if not getattr(self.client, "inprocess", False) and (
                values["host"] != getattr(self.client, "host", values["host"])
                or values["cmd_port"] != getattr(self.client, "cmd_port", values["cmd_port"])
                or values["pub_port"] != getattr(self.client, "pub_port", values["pub_port"])
            ):
                self.client = MFDBClient(
                    host=values["host"],
                    cmd_port=values["cmd_port"],
                    pub_port=values["pub_port"],
                )

            try:
                result = self.client.login(
                    user_id=user_id,
                    password=values["password"],
                    client_metadata=client_metadata,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "MFDB login failed", str(exc)
                )
                continue
            if isinstance(result, dict) and result.get("ok"):
                self._auth_login_user = user_id
                cache_session(
                    user_id, getattr(self.client, "token", None),
                    getattr(self.client, "host", "127.0.0.1"),
                    getattr(self.client, "cmd_port", 8765),
                    getattr(self.client, "pub_port", 8766),
                )
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
        except Exception as exc:
            chisurf.logging.warning("_disconnect_signal(self, signal: Any) -> None: %s", exc)

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

        # Open details
        current_row = table.rowAt(pos.y())
        has_row = current_row >= 0 and current_row < table.rowCount()
        if has_row and delete_one_fn is not None:
            menu.addAction(
                "🔍 Open details",
                lambda: self._open_table_row(table, current_row, item_kind, id_col),
            )

        # Copy actions
        menu.addAction(
            "📋 Copy checked IDs",
            lambda: self._copy_checked_ids_to_clipboard(table, id_col),
        )
        menu.addAction(
            "📋 Copy selected row",
            lambda: self._copy_selected_row(table),
        )
        menu.addAction(
            "📋 Copy selected cell",
            lambda: self._copy_selected_cell(table),
        )

        menu.addSeparator()

        # Check/uncheck actions
        menu.addAction(
            "☑ Check selected rows",
            lambda: self._set_selected_checks(table, True),
        )
        menu.addAction(
            "☐ Uncheck selected rows",
            lambda: self._set_selected_checks(table, False),
        )
        menu.addAction("☑ Check all visible", lambda: self._set_all_checks(table, True))
        menu.addAction("☐ Uncheck all", lambda: self._set_all_checks(table, False))
        menu.addAction("🔁 Invert visible checks", lambda: self._invert_checks(table))

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
    def _open_table_row(
        table: QtWidgets.QTableWidget,
        row: int,
        item_kind: str,
        id_col: int,
    ) -> None:
        """Select the row under the cursor and trigger default action."""
        id_item = table.item(row, id_col)
        if id_item is not None:
            logging.info("Open details: %s %s", item_kind, id_item.text())

    def _copy_checked_ids_to_clipboard(self, table: QtWidgets.QTableWidget, id_col: int) -> None:
        """Copy all checked row IDs to clipboard, newline-separated."""
        ids = self._checked_row_ids(table, id_col)
        if ids:
            QtWidgets.QApplication.clipboard().setText("\n".join(ids))
            self.status_label.setText(f"Copied {len(ids)} IDs")

    @staticmethod
    def _copy_selected_row(table: QtWidgets.QTableWidget) -> None:
        """Copy the selected row as tab-separated text."""
        selected = table.selectedItems()
        if not selected:
            return
        rows: dict[int, list[str]] = {}
        for item in selected:
            row = item.row()
            if row not in rows:
                rows[row] = []
            rows[row].append(item.text())
        lines = ["\t".join(cols) for cols in rows.values()]
        if lines:
            QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    @staticmethod
    def _copy_selected_cell(table: QtWidgets.QTableWidget) -> None:
        """Copy the text of the selected (active) cell."""
        item = table.currentItem()
        if item is not None:
            QtWidgets.QApplication.clipboard().setText(item.text())

    @staticmethod
    def _set_selected_checks(table: QtWidgets.QTableWidget, checked: bool) -> None:
        """Check or uncheck only the selected rows."""
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        selected = table.selectedItems() if hasattr(table, 'selectedItems') else []
        rows = set()
        for item in selected:
            rows.add(item.row())
        for row in rows:
            cb = table.item(row, 0)
            if cb is not None and cb.flags() & QtCore.Qt.ItemIsUserCheckable:
                cb.setCheckState(state)

    def _setup_standard_table(
        self,
        table: QtWidgets.QTableWidget,
        *,
        headers: list[str],
        item_kind: str = "item",
        id_col: int = 1,
        delete_one_fn: Any = None,
        usage_fn: Any = None,
        open_details_fn: Any = None,
        extra_actions: list[tuple[str, Any]] | None = None,
    ) -> None:
        """Configure a QTableWidget with a dedicated checkbox column and
        standard context menu.

        Prepends a checkbox column (✓) before the supplied *headers*.
        The id_col parameter refers to the column index **after** the
        checkbox column is prepended; e.g. if you want the ID in the
        first data column, pass id_col=1.

        Parameters
        ----------
        table : QTableWidget
            The table to configure.
        headers : list of str
            Data column headers (checkbox column is prepended).
        item_kind : str
            Human-readable label for context menus and dialogs.
        id_col : int
            Column index where the item ID lives (after checkbox prepend).
        delete_one_fn : callable or None
            ``delete_one_fn(item_id) -> None``.
        usage_fn : callable or None
            ``usage_fn([ids]) -> {id: description}`` for dependency checks.
        open_details_fn : callable or None
            Called with the row ID when "Open details" is selected.
        extra_actions : list of (label, callable) or None
            Additional context-menu entries.
        """
        all_headers = ["✓"] + list(headers)
        table.setColumnCount(len(all_headers))
        table.setHorizontalHeaderLabels(all_headers)
        apply_compact_table_style(table)
        # Re-enable sorting disabled by apply_compact_table_style
        table.setSortingEnabled(True)

        self._install_table_context_menu(
            table,
            item_kind=item_kind,
            id_col=id_col,
            delete_one_fn=delete_one_fn,
            usage_fn=usage_fn,
            extra_actions=extra_actions,
        )

        if open_details_fn is not None:
            table.cellDoubleClicked.connect(
                lambda row, _col: (
                    open_details_fn(str(table.item(row, id_col).text()))
                    if table.item(row, id_col)
                    else None
                )
            )

    def _build_filter_bar(
        self,
        filters: list[dict[str, Any]],
    ) -> QtWidgets.QWidget:
        """Build a generic filter bar with labelled combo boxes.

        Each entry in *filters* supports:
            label      – display text for the QLabel
            attr       – stored as ``self.{attr}`` (QComboBox)
            items      – static list of items to add (optional)
            signal     – ``"currentIndexChanged"`` (default) or ``"currentTextChanged"``
            cb         – callable connected to *signal* (optional)
            default_data – userData for the empty first item (default ``""``)
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for f in filters:
            layout.addWidget(QtWidgets.QLabel(f["label"]))
            combo = QtWidgets.QComboBox()
            items = f.get("items")
            if items:
                for item in items:
                    if isinstance(item, tuple):
                        combo.addItem(item[0], item[1])
                    else:
                        combo.addItem(item, item)
            else:
                combo.addItem("", f.get("default_data", ""))
            setattr(self, f["attr"], combo)
            if f.get("cb"):
                sig = f.get("signal", "currentIndexChanged")
                getattr(combo, sig).connect(f["cb"])
            layout.addWidget(combo)
        layout.addStretch()
        return widget

    def _create_standard_dock_tab(
        self,
        table: QtWidgets.QTableWidget,
        detail_widget: MFDBDetailWidget,
        save_slot: Callable[[], None] | None = None,
        delete_slot: Callable[[], None] | None = None,
        extra_widgets_top: list[QtWidgets.QWidget] | None = None,
        extra_buttons: list[QtWidgets.QWidget] | None = None,
        extra_widgets_bottom: list[QtWidgets.QWidget] | None = None,
    ) -> QtWidgets.QWidget:
        """Create a standardized dock tab layout with a table at the top and a form at the bottom."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if extra_widgets_top:
            for w in extra_widgets_top:
                layout.addWidget(w)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(splitter, stretch=1)

        splitter.addWidget(table)

        bottom_widget = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(4, 4, 4, 4)
        bottom_layout.setSpacing(4)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(detail_widget)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        bottom_layout.addWidget(scroll, stretch=1)

        if extra_widgets_bottom:
            for w in extra_widgets_bottom:
                bottom_layout.addWidget(w)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(2)

        if save_slot:
            save_btn = self._text_icon_button(
                "💾 Save", QtWidgets.QStyle.SP_DialogSaveButton, "Save changes", save_slot
            )
            buttons_layout.addWidget(save_btn)

        if delete_slot:
            delete_btn = self._text_icon_button(
                "🗑 Delete", QtWidgets.QStyle.SP_TrashIcon, "Delete selected", delete_slot
            )
            buttons_layout.addWidget(delete_btn)

        if extra_buttons:
            for btn in extra_buttons:
                buttons_layout.addWidget(btn)

        buttons_layout.addStretch()
        bottom_layout.addLayout(buttons_layout)

        splitter.addWidget(bottom_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        return widget

    def _checkbox_item(self, checked: bool = False) -> QtWidgets.QTableWidgetItem:
        """Return a centered, checkable table item for column 0."""
        item = QtWidgets.QTableWidgetItem("")
        item.setFlags(
            QtCore.Qt.ItemIsUserCheckable
            | QtCore.Qt.ItemIsEnabled
            | QtCore.Qt.ItemIsSelectable
        )
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        return item

    def _checked_row_ids(self, table: QtWidgets.QTableWidget, id_col: int = 1) -> list[str]:
        """Return IDs of all checked rows in *table*."""
        ids: list[str] = []
        for row in range(table.rowCount()):
            cb = table.item(row, 0)
            if cb is not None and cb.checkState() == QtCore.Qt.Checked:
                id_item = table.item(row, id_col)
                if id_item is not None:
                    ids.append(id_item.text().strip())
        return ids

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

    def _extra_buttons_for_spec(self, spec: "EntitySpec") -> list[QtWidgets.QWidget]:
        """Return ChiSurf-specific extra toolbar buttons for a given EntitySpec.

        These are per-entity actions that are NOT part of generic CRUD but are
        specific to ChiSurf administration (branch management, password change,
        etc.).  Every button must be a QWidget (e.g. QToolButton).
        """
        btns: list[QtWidgets.QWidget] = []

        if spec.key == "user":
            pw_btn = QtWidgets.QToolButton()
            pw_btn.setText("🔑 Change password")
            pw_btn.setToolTip("Change the selected user's password")
            pw_btn.clicked.connect(self._change_password_for_selected_user)
            btns.append(pw_btn)

            branch_btn = QtWidgets.QToolButton()
            branch_btn.setText("⎇ Jump to branch")
            branch_btn.setToolTip("Move the selected user to a branch")
            branch_btn.clicked.connect(self.create_time_branch)
            btns.append(branch_btn)

        elif spec.key == "branch":
            head_btn = QtWidgets.QToolButton()
            head_btn.setText("🕐 Set head")
            head_btn.setToolTip("Update the head operation of this branch")
            head_btn.clicked.connect(self._set_branch_head_for_selected)
            btns.append(head_btn)

        elif spec.key == "object":
            copy_btn = QtWidgets.QToolButton()
            copy_btn.setText("📋 Copy UUID")
            copy_btn.setToolTip("Copy the selected object UUID")
            copy_btn.clicked.connect(self._copy_selected_object_entity_uuid)
            btns.append(copy_btn)

            reveal_btn = QtWidgets.QToolButton()
            reveal_btn.setText("📂 Reveal")
            reveal_btn.setToolTip("Reveal the selected object in the file manager")
            reveal_btn.clicked.connect(self._reveal_selected_object_entity)
            btns.append(reveal_btn)

            delete_btn = QtWidgets.QToolButton()
            delete_btn.setText("🗑 Delete object")
            delete_btn.setToolTip("Delete the selected object or decrement its refcount")
            delete_btn.clicked.connect(self._delete_selected_object_entity)
            btns.append(delete_btn)

        elif spec.key == "raw_data":
            copy_btn = QtWidgets.QToolButton()
            copy_btn.setText("📋 Copy ID")
            copy_btn.setToolTip("Copy the selected raw-data artifact ID")
            copy_btn.clicked.connect(self._copy_selected_raw_data_id)
            btns.append(copy_btn)

            reveal_btn = QtWidgets.QToolButton()
            reveal_btn.setText("📂 Reveal")
            reveal_btn.setToolTip("Open the selected raw-data file or URL")
            reveal_btn.clicked.connect(self._reveal_selected_raw_data)
            btns.append(reveal_btn)

            seed_btn = QtWidgets.QToolButton()
            seed_btn.setText("🌱 Use as provenance seed")
            seed_btn.setToolTip("Load the selected raw-data artifact in the provenance graph")
            seed_btn.clicked.connect(self._seed_selected_raw_data)
            btns.append(seed_btn)

            validate_btn = QtWidgets.QToolButton()
            validate_btn.setText("✓ Validate")
            validate_btn.setToolTip("Set validation status for the selected raw-data artifact")
            validate_btn.clicked.connect(self._validate_selected_raw_data)
            btns.append(validate_btn)

            delete_btn = QtWidgets.QToolButton()
            delete_btn.setText("🗑 Delete")
            delete_btn.setToolTip("Soft-delete the selected raw-data artifact")
            delete_btn.clicked.connect(self._delete_selected_raw_data)
            btns.append(delete_btn)

        elif spec.key == "processed_product":
            copy_btn = QtWidgets.QToolButton()
            copy_btn.setText("📋 Copy ID")
            copy_btn.setToolTip("Copy the selected processed-data artifact ID")
            copy_btn.clicked.connect(self._copy_selected_processed_product_id)
            btns.append(copy_btn)

            reveal_btn = QtWidgets.QToolButton()
            reveal_btn.setText("📂 Reveal")
            reveal_btn.setToolTip("Open the selected processed-data file or URL")
            reveal_btn.clicked.connect(self._reveal_selected_processed_product)
            btns.append(reveal_btn)

            seed_btn = QtWidgets.QToolButton()
            seed_btn.setText("🌱 Use as provenance seed")
            seed_btn.setToolTip("Load the selected processed-data artifact in the provenance graph")
            seed_btn.clicked.connect(self._seed_selected_processed_product)
            btns.append(seed_btn)

            validate_btn = QtWidgets.QToolButton()
            validate_btn.setText("✓ Validate")
            validate_btn.setToolTip("Set validation status for the selected processed-data artifact")
            validate_btn.clicked.connect(self._validate_selected_processed_product)
            btns.append(validate_btn)

            delete_btn = QtWidgets.QToolButton()
            delete_btn.setText("🗑 Delete")
            delete_btn.setToolTip("Soft-delete the selected processed-data artifact")
            delete_btn.clicked.connect(self._delete_selected_processed_product)
            btns.append(delete_btn)

        elif spec.key == "analysis":
            copy_btn = QtWidgets.QToolButton()
            copy_btn.setText("📋 Copy ID")
            copy_btn.setToolTip("Copy the selected analysis run ID")
            copy_btn.clicked.connect(self._copy_selected_analysis_id)
            btns.append(copy_btn)

            detail_btn = QtWidgets.QToolButton()
            detail_btn.setText("🔎 Details")
            detail_btn.setToolTip("Show parameters and linked products for the selected analysis run")
            detail_btn.clicked.connect(self._show_selected_analysis_details)
            btns.append(detail_btn)

            seed_btn = QtWidgets.QToolButton()
            seed_btn.setText("🌱 Use as provenance seed")
            seed_btn.setToolTip("Load the selected analysis run in the provenance graph")
            seed_btn.clicked.connect(self._seed_selected_analysis)
            btns.append(seed_btn)

        return btns

    def _change_password_for_selected_user(self) -> None:
        """Open password-change dialog for the user selected in the Users dock."""
        dock = self._entity_docks.get("user")
        if dock is None:
            return
        user_id = dock.selected_row_id() if isinstance(dock, EntityDock) else None
        if not user_id:
            QtWidgets.QMessageBox.information(self, "No user selected", "Select a user row first.")
            return
        dlg = PasswordChangeDialog(user_id=user_id, client=self.client, parent=self)
        dlg.exec()

    def _set_branch_head_for_selected(self) -> None:
        """Prompt for a head operation ID and update the selected branch."""
        dock = self._entity_docks.get("branch")
        if dock is None:
            return
        branch_uuid = dock.selected_row_id() if isinstance(dock, EntityDock) else None
        if not branch_uuid:
            QtWidgets.QMessageBox.information(self, "No branch selected", "Select a branch row first.")
            return
        op_id, ok = QtWidgets.QInputDialog.getText(
            self, "Set branch head", "Operation ID (leave empty to reset to None):"
        )
        if not ok:
            return
        try:
            self.client.update_branch_head(branch_uuid, op_id.strip() or None)
            self.status_label.setText(f"Branch {branch_uuid} head updated.")
            if isinstance(dock, EntityDock):
                dock.refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Update failed", str(exc))

    def _selected_entity_dock_data(self, entity_key: str) -> dict[str, Any]:
        """Return form or cached row data for the selected visible EntityDock row."""
        dock = self._entity_docks.get(entity_key)
        if not isinstance(dock, EntityDock):
            return {}
        data = dock.form.get_data() if getattr(dock, "form", None) is not None else {}
        if data:
            return data
        row_id = dock.selected_row_id()
        return dock._row_cache.get(row_id, {}) if row_id else {}

    def _selected_object_entity_data(self) -> dict[str, Any]:
        return self._selected_entity_dock_data("object")

    def _copy_selected_object_entity_uuid(self) -> None:
        """Copy the selected object UUID from the visible Objects entity dock."""
        object_uuid = str(self._selected_object_entity_data().get("object_uuid") or "")
        if object_uuid:
            QtWidgets.QApplication.clipboard().setText(object_uuid)

    def _reveal_selected_object_entity(self) -> None:
        """Reveal the selected object from the visible Objects entity dock."""
        data = self._selected_object_entity_data()
        storage_path = str(data.get("storage_path") or "").strip()
        if not storage_path:
            return
        try:
            from mfdb.store.database_resolver import object_store_root

            path = object_store_root() / storage_path
            QtGui.QDesktopServices.openUrl(_qurl_for_location(str(path)))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Reveal failed", str(exc))

    def _delete_selected_object_entity(self) -> None:
        """Delete or dereference the selected object from the visible Objects entity dock."""
        dock = self._entity_docks.get("object")
        if not isinstance(dock, EntityDock):
            return
        data = self._selected_object_entity_data()
        object_uuid = str(data.get("object_uuid") or dock.selected_row_id() or "")
        if not object_uuid:
            QtWidgets.QMessageBox.warning(self, "No object selected", "Select an object to delete.")
            return
        filename = data.get("original_filename") or object_uuid
        refcount = int(data.get("refcount") or 0)
        msg = f"Delete object '{filename}'?\n\nRefcount: {refcount}"
        if refcount > 1:
            msg += "\n\nThis will only decrement the refcount; the blob will remain."
        else:
            msg += "\n\nThis will permanently delete the blob from disk."
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete object",
            msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            result = self.client.delete_object(object_uuid)
            dock.refresh()
            self.status_label.setText(f"Object delete result: {result}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete failed", str(exc))

    def _copy_selected_raw_data_id(self) -> None:
        """Copy the selected raw-data artifact ID from the visible EntityDock."""
        raw_id = str(self._selected_entity_dock_data("raw_data").get("raw_data_id") or "")
        if raw_id:
            QtWidgets.QApplication.clipboard().setText(raw_id)

    def _reveal_selected_raw_data(self) -> None:
        """Open the selected raw-data artifact path or URL from the visible EntityDock."""
        data = self._selected_entity_dock_data("raw_data")
        path = str(data.get("location") or _processed_location(data)).strip()
        if path:
            QtGui.QDesktopServices.openUrl(_qurl_for_location(path))

    def _seed_selected_raw_data(self) -> None:
        """Use the selected raw-data artifact as the provenance graph seed."""
        raw_id = str(self._selected_entity_dock_data("raw_data").get("raw_data_id") or "")
        if raw_id:
            self._set_provenance_seed("raw_data", raw_id)

    def _validate_selected_raw_data(self) -> None:
        """Set validation status for the selected raw-data artifact."""
        self._set_selected_artifact_validation("raw_data", "raw_data_id")

    def _delete_selected_raw_data(self) -> None:
        """Soft-delete the selected raw-data artifact."""
        self._delete_selected_artifact("raw_data", "raw_data_id", "raw-data artifact")

    def _copy_selected_processed_product_id(self) -> None:
        """Copy the selected processed-data artifact ID from the visible EntityDock."""
        data = self._selected_entity_dock_data("processed_product")
        product_id = str(data.get("product_id") or data.get("processed_data_id") or "")
        if product_id:
            QtWidgets.QApplication.clipboard().setText(product_id)

    def _reveal_selected_processed_product(self) -> None:
        """Open the selected processed-data artifact path or URL from the visible EntityDock."""
        data = self._selected_entity_dock_data("processed_product")
        path = str(data.get("location") or _processed_location(data)).strip()
        if path:
            QtGui.QDesktopServices.openUrl(_qurl_for_location(path))

    def _seed_selected_processed_product(self) -> None:
        """Use the selected processed-data artifact as the provenance graph seed."""
        data = self._selected_entity_dock_data("processed_product")
        product_id = str(data.get("product_id") or data.get("processed_data_id") or "")
        if product_id:
            self._set_provenance_seed("processed_data", product_id)

    def _validate_selected_processed_product(self) -> None:
        """Set validation status for the selected processed-data artifact."""
        self._set_selected_artifact_validation(
            "processed_product",
            "product_id",
            fallback_id_key="processed_data_id",
        )

    def _delete_selected_processed_product(self) -> None:
        """Soft-delete the selected processed-data artifact."""
        self._delete_selected_artifact(
            "processed_product",
            "product_id",
            "processed-data artifact",
            fallback_id_key="processed_data_id",
        )

    def _selected_artifact_id(
        self,
        entity_key: str,
        id_key: str,
        *,
        fallback_id_key: str | None = None,
    ) -> str:
        """Return the selected artifact id from a visible EntityDock."""
        data = self._selected_entity_dock_data(entity_key)
        return str(data.get(id_key) or (data.get(fallback_id_key) if fallback_id_key else "") or "")

    def _set_selected_artifact_validation(
        self,
        entity_key: str,
        id_key: str,
        *,
        fallback_id_key: str | None = None,
    ) -> None:
        """Prompt for and persist validation status on the selected artifact."""
        artifact_id = self._selected_artifact_id(
            entity_key,
            id_key,
            fallback_id_key=fallback_id_key,
        )
        if not artifact_id:
            QtWidgets.QMessageBox.information(self, "No artifact selected", "Select an artifact row first.")
            return
        current = str(self._selected_entity_dock_data(entity_key).get("validation_status") or "unvalidated")
        choices = list(VALIDATION_STATUS_VALUES)
        status, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Set validation status",
            f"Validation status for {artifact_id}:",
            choices,
            choices.index(current) if current in choices else 0,
            False,
        )
        if not ok:
            return
        try:
            self.client.set_artifact_validation(artifact_id, str(status))
            dock = self._entity_docks.get(entity_key)
            if isinstance(dock, EntityDock):
                dock.refresh()
            self.status_label.setText(f"Artifact {artifact_id} validation set to {status}.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Validation update failed", str(exc))

    def _delete_selected_artifact(
        self,
        entity_key: str,
        id_key: str,
        label: str,
        *,
        fallback_id_key: str | None = None,
    ) -> None:
        """Confirm and soft-delete the selected artifact."""
        artifact_id = self._selected_artifact_id(
            entity_key,
            id_key,
            fallback_id_key=fallback_id_key,
        )
        if not artifact_id:
            QtWidgets.QMessageBox.information(self, "No artifact selected", "Select an artifact row first.")
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete artifact",
            f"Delete {label} '{artifact_id}'?\n\nThis soft-deletes the artifact and direct provenance links.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.client.delete_artifact(artifact_id)
            dock = self._entity_docks.get(entity_key)
            if isinstance(dock, EntityDock):
                dock.refresh()
            self.status_label.setText(f"Artifact {artifact_id} deleted.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Artifact delete failed", str(exc))

    def _copy_selected_analysis_id(self) -> None:
        """Copy the selected analysis run ID from the visible EntityDock."""
        analysis_id = str(self._selected_entity_dock_data("analysis").get("analysis_id") or "")
        if analysis_id:
            QtWidgets.QApplication.clipboard().setText(analysis_id)

    def _seed_selected_analysis(self) -> None:
        """Use the selected analysis run as the provenance graph seed."""
        analysis_id = str(self._selected_entity_dock_data("analysis").get("analysis_id") or "")
        if analysis_id:
            self._set_provenance_seed("analysis_run", analysis_id)

    def _fill_analysis_drilldown_table(
        self,
        table: QtWidgets.QTableWidget,
        rows: list[dict[str, Any]],
        columns: list[tuple[str, str]],
    ) -> None:
        """Populate a read-only analysis drilldown table."""
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels([label for label, _key in columns])
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, (_label, key) in enumerate(columns):
                value = row.get(key, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, sort_keys=True)
                item = QtWidgets.QTableWidgetItem("" if value is None else str(value))
                table.setItem(row_index, col_index, item)
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

    def _show_selected_analysis_details(self) -> None:
        """Show parameters and linked products for the selected analysis run."""
        analysis_id = str(self._selected_entity_dock_data("analysis").get("analysis_id") or "")
        if not analysis_id:
            QtWidgets.QMessageBox.information(self, "No analysis selected", "Select an analysis row first.")
            return
        try:
            detail = self.client.get_analysis_run_full(analysis_id)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Analysis details failed", str(exc))
            return
        if not detail:
            QtWidgets.QMessageBox.information(
                self,
                "Analysis not found",
                f"No analysis details were found for {analysis_id}.",
            )
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Analysis details: {analysis_id}")
        dialog.resize(760, 440)
        layout = QtWidgets.QVBoxLayout(dialog)
        summary = QtWidgets.QLabel(
            f"{analysis_id} · {detail.get('analysis_type') or detail.get('type') or ''} · "
            f"{detail.get('convergence_status') or detail.get('status') or ''}"
        )
        layout.addWidget(summary)

        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("analysis_drilldown_tabs")
        layout.addWidget(tabs)

        parameter_table = QtWidgets.QTableWidget()
        parameter_table.setObjectName("analysis_parameters_table")
        self._fill_analysis_drilldown_table(
            parameter_table,
            list(detail.get("parameters") or []),
            [
                ("Parameter", "name"),
                ("Value", "value"),
                ("Std. error", "standard_error"),
                ("Units", "units"),
                ("Type", "parameter_type"),
            ],
        )
        tabs.addTab(parameter_table, "Parameters")

        input_table = QtWidgets.QTableWidget()
        input_table.setObjectName("analysis_input_products_table")
        product_columns = [
            ("Product ID", "processed_data_id"),
            ("Type", "product_type"),
            ("Storage", "storage_mode"),
            ("Status", "validation_status"),
            ("Location", "file_path"),
        ]
        self._fill_analysis_drilldown_table(
            input_table,
            list(detail.get("input_processed_data") or []),
            product_columns,
        )
        tabs.addTab(input_table, "Input products")

        output_table = QtWidgets.QTableWidget()
        output_table.setObjectName("analysis_output_products_table")
        self._fill_analysis_drilldown_table(
            output_table,
            list(detail.get("processed_data") or []),
            product_columns,
        )
        tabs.addTab(output_table, "Output products")

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)

        self._analysis_drilldown_detail = detail
        self._analysis_drilldown_dialog = dialog
        dialog.show()

    # ------------------------------------------------------------------
    # Navigation panels (NavigationPanelTool)
    # ------------------------------------------------------------------

    _ENTITY_GROUPS = (
        "Samples & chemistry",
        "Experiments & data",
        "Provenance",
        "Administration",
    )

    #: Per-entity nav emoji, keyed by entity-registry key.
    _ENTITY_ICONS = {
        "sample": "🧪", "condition": "🌡️", "entity": "🧬", "probe": "💡",
        "position": "📍", "fret_pair": "🔗", "experiment": "🔬",
        "experiment_type": "🧾", "setup": "⚙️", "detector_channel": "📡",
        "pie_window": "🪟", "fcs_pair": "🔀", "device": "🖥️", "raw_data": "📂",
        "processing_run": "🏭", "processed_product": "📦", "analysis": "📈",
        "object": "🧱", "project": "📁", "branch": "🌿", "user": "👤",
    }

    def _build_panels(self) -> list[dict[str, Any]]:
        """Return the ordered left-nav panel list for the shell.

        The old nested DockArea/QTabWidget grouping is flattened into one nav
        list: emoji-tagged separator headers group the per-entity panels, the
        aggregate Overview/All-items/Measurements/Graph/Import-Export tabs become
        panels, the previously-orphaned workflow views are surfaced, and the
        fluorophore curation view is integrated from the fluorophore_db plugin.
        """
        panels: list[dict[str, Any]] = [
            {"name": "Overview", "icon": "📊", "factory": lambda p: self.overview_tab()},
            {"name": "All items", "icon": "🗂", "factory": lambda p: self.all_items_tab()},
            {"name": "Measurements", "icon": "📈", "factory": lambda p: self.measurements_tab()},
        ]

        def _add_group_entities(group: str) -> None:
            if self._dictionary is None:
                return
            for spec in [s for s in ENTITY_REGISTRY if s.group == group]:
                panels.append({
                    "name": spec.title,
                    "icon": self._ENTITY_ICONS.get(spec.key, "•"),
                    "entity_key": spec.key,
                    "factory": self._entity_factory(spec),
                })

        panels.append({"name": "Samples & chemistry", "icon": "🧫", "separator": True})
        _add_group_entities("Samples & chemistry")
        panels.append({
            "name": "Sample Metadata", "icon": "🏷️", "entity_key": "metadata",
            "factory": self._metadata_factory,
        })
        panels.append({
            "name": "Spectra", "icon": "🌈", "factory": self._fluorophore_factory,
        })

        panels.append({"name": "Experiments & data", "icon": "🔬", "separator": True})
        _add_group_entities("Experiments & data")

        panels.append({"name": "Provenance", "icon": "🕸️", "separator": True})
        _add_group_entities("Provenance")
        panels.append({"name": "Provenance Graph", "icon": "🕸️", "factory": lambda p: self.provenance_graph_dock()})

        panels.append({"name": "Administration", "icon": "🛡️", "separator": True})
        _add_group_entities("Administration")
        panels.append({"name": "Import / Export", "icon": "🔄", "factory": lambda p: self.import_export_tab()})

        panels.append({"name": "Workflows & QC", "icon": "🧰", "separator": True})
        panels += [
            {"name": "Studies", "icon": "📚", "factory": lambda p: StudiesView(self.client, p)},
            {"name": "Protocols", "icon": "📋", "factory": lambda p: ProtocolsView(self.client, p)},
            {"name": "Lifecycle", "icon": "♻️", "factory": lambda p: LifecycleView(self.client, p)},
            {"name": "Calibrations", "icon": "🎯", "factory": lambda p: CalibrationsView(self.client, p)},
            {"name": "Reagent Lots", "icon": "🧴", "factory": lambda p: ReagentLotsView(self.client, p)},
            {"name": "Pipelines", "icon": "🛠️", "factory": lambda p: PipelinesView(self.client, p)},
        ]
        return panels

    def _fluorophore_factory(self, parent):
        """Build the integrated optical component curation dock (lazy import)."""
        from .optical_components import OpticalComponentDock

        return OpticalComponentDock(self.client, parent)

    def _entity_factory(self, spec: EntitySpec):
        """Build a lazy factory creating + registering an EntityDock for ``spec``."""
        def factory(parent):
            dock = EntityDock(
                spec=spec,
                client=self.client,
                dictionary=self._dictionary,
                schema_map=self._schema_map,
                registry_dict=self._registry_dict,
                extra_buttons=self._extra_buttons_for_spec(spec),
            )
            self._entity_docks[spec.key] = dock
            dock.jumpRequested.connect(
                lambda ek, rid, _k=spec.key: self._jump_to_entity(ek, rid)
            )
            dock.statusMessage.connect(self.status_label.setText)
            return dock
        return factory

    def _metadata_factory(self, parent):
        dock = MetadataDock(client=self.client)
        self._entity_docks["metadata"] = dock
        return dock

    def _refresh_active_panel(self) -> None:
        """Refresh the currently visible panel if it is an entity dock."""
        idx = self.nav_list.currentRow()
        if 0 <= idx < len(self.panels):
            inst = self.panels[idx].get("instance")
            widget = self._unwrap(inst)
            if isinstance(widget, EntityDock):
                widget.refresh()

    @staticmethod
    def _unwrap(wrapper):
        """Return the panel content widget from a NavigationPanelTool wrapper."""
        if wrapper is None:
            return None
        layout = wrapper.layout()
        if layout is not None and layout.count():
            return layout.itemAt(layout.count() - 1).widget()
        return wrapper

    def _jump_to_entity(self, entity_key: str, record_id: str) -> None:
        """Select the entity's nav panel (lazy-building its dock), then the record."""
        row = self._row_by_entity.get(entity_key)
        if row is None:
            return
        self.nav_list.setCurrentRow(row)  # builds + shows the panel
        dock = self._entity_docks.get(entity_key)
        if dock is None:
            return
        if entity_key == "metadata":
            dock.load_sample(record_id)
        else:
            dock.jump_to(record_id)

    def setup_ui(self) -> None:
        # The NavigationPanelTool shell owns the central widget (nav list + stack).
        # This method only builds the hidden legacy tab widgets for their
        # attribute side-effects (self.*_table / *_detail_widget / *_edit) that
        # clear_form()/refresh() and other utilities still reference.

        # --- Initialise bespoke-tab attributes without showing the tabs ---
        # These calls create self.*_table / self.*_detail_widget / self.*_edit
        # attributes that clear_form(), refresh(), and other utility methods
        # still reference. The widgets are created but never added to self.tabs.
        # Build legacy tab widgets off-screen.  Store in an instance attr so
        # the underlying C++ QWidget objects are never garbage-collected while
        # this window exists (parentless, hidden widgets are freed by Qt if no
        # Python strong reference holds them).
        self._hidden_legacy_tabs = (
            self.sample_tab(),
            self.condition_tab(),
            self.entities_tab(),
            self.probes_tab(),
            self.positions_tab(),
            self.fret_pairs_tab(),
            self.metadata_tab(),
            self.experiments_tab(),
            self.processing_runs_tab(),
            self.raw_data_tab(),
            self.processed_products_tab(),
            self.objects_tab(),
            self.analyses_tab(),
            self.provenance_tab(),
            self.users_tab(),
            self.branches_tab(),
            self.devices_tab(),
            self.setups_tab(),
            self.experiment_types_tab(),
            self.projects_tab(),
        )
        # status_label is created in __init__ (before super); entity panels are
        # built lazily by the nav shell via the factories in _build_panels().

    def setup_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("⬇️ &Import...", self.import_file)
        file_menu.addAction("⬆️ &Export selected sample...", self.export_selected_sample)
        file_menu.addAction("💾 &Backup database...", self.backup_database)
        file_menu.addAction("♻️ Reset", self.reset_from_source)
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

        import chisurf.core.settings as cs_settings
        mfdb_settings = cs_settings.cs_settings.get("mfdb", {})
        last_server = mfdb_settings.get("last_server", "127.0.0.1")
        last_port = mfdb_settings.get("last_port", 8765)

        # Endpoint — the host/IP and port are inline so the user can point at a
        # different MFDB server directly.
        toolbar.addWidget(QtWidgets.QLabel(" 🌐 "))
        self.server_edit = QtWidgets.QLineEdit(last_server)
        self.server_edit.setPlaceholderText("127.0.0.1")
        self.server_edit.setFixedWidth(130)
        self.server_edit.setToolTip("MFDB server host / IP address")
        self.server_edit.returnPressed.connect(self._on_login_clicked)
        toolbar.addWidget(self.server_edit)
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(int(last_port))
        self.port_spin.setFixedWidth(64)
        self.port_spin.setToolTip("MFDB server port")
        toolbar.addWidget(self.port_spin)
        self.url_edit = QtWidgets.QLineEdit(self.DEFAULT_URL)  # back-compat

        # Auth — a single line: user + (optional) password. With an active
        # session the password is not needed; "More…" opens the AutoForm dialog.
        toolbar.addWidget(QtWidgets.QLabel(" 👤 "))
        self.username_edit = QtWidgets.QLineEdit()
        self.username_edit.setFixedWidth(110)
        self.username_edit.setToolTip("MFDB username")
        self.username_edit.returnPressed.connect(self._on_login_clicked)
        toolbar.addWidget(self.username_edit)
        toolbar.addWidget(QtWidgets.QLabel(" 🔑 "))
        self.password_edit = QtWidgets.QLineEdit()
        self.password_edit.setFixedWidth(110)
        self.password_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_edit.setPlaceholderText("(session)")
        self.password_edit.setToolTip("MFDB password — not needed if the session is already authorized")
        self.password_edit.returnPressed.connect(self._on_login_clicked)
        toolbar.addWidget(self.password_edit)

        self.login_action = toolbar.addAction("🔐 Login", self._on_login_clicked)
        self.more_action = toolbar.addAction("⋯", self._open_connection_dialog)
        self.more_action.setToolTip("More connection / authentication options")
        self.logout_action = toolbar.addAction("🚪 Logout", self._on_logout_clicked)
        self.login_action.setToolTip("Connect and authenticate at the host/port above")
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
        self.refresh_action.setToolTip("Reload MFDB status and visible tables")
        self._transport_actions = [self.refresh_action]
        for text, slot, tip in (
            ("⬇️ Import", self.import_file, "Import fluorescence/sample/project data into MFDB"),
            ("💾 Backup", self.backup_database, "Create a backup copy of the active MFDB database"),
            ("♻️ Reset", self.reset_from_source, "Reset the active MFDB database. A backup is created first."),
        ):
            action = toolbar.addAction(text, slot)
            action.setToolTip(tip)
            self._transport_actions.append(action)
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
            who = self._auth_login_user or ""
            self.user_label.setText(f" ● {who} ".rstrip() + " ")
            self.user_label.setStyleSheet("color: #2e7d32; padding: 0 6px; font-weight: bold;")
            self.user_label.setToolTip(f"Connected as {who}" if who else "Connected")
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

    def _open_connection_dialog(self) -> None:
        """Open the AutoForm login dialog, then connect with its values."""
        from mfdb.admin.gui.connection_dialog import ConnectionAuthDialog

        user = self.username_edit.text() or self._active_mfdb_user_id()
        dialog = ConnectionAuthDialog(
            user=user,
            host=self.server_edit.text() or "127.0.0.1",
            cmd_port=self.port_spin.value(),
            pub_port=self.port_spin.value() + 1,
            parent=self,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        v = dialog.values()
        self.server_edit.setText(v["host"])
        self.port_spin.setValue(int(v["cmd_port"]))
        self.username_edit.setText(v["user"])
        self.password_edit.setText(v["password"])
        self._on_login_clicked()

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
        except Exception as exc:
            chisurf.logging.warning("_on_logout_clicked(self) -> None: %s", exc)
        try:
            self.client.token = None
        except Exception as exc:
            chisurf.logging.warning("_on_logout_clicked(self) -> None: %s", exc)
        from mfdb.admin.gui.session import clear_cached_session

        clear_cached_session()
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

    def _set_status_message(self, message: str, timeout_ms: int = 0) -> None:
        """Show a status message in the label and status bar."""
        self.status_label.setText(message)
        self.statusBar().showMessage(message, timeout_ms)

    def _run_background_task(
        self,
        *,
        label: str,
        fn: Callable[[], Any],
        on_success: Callable[[Any], None],
        on_failure: Callable[[str], None] | None = None,
    ) -> None:
        """Run a blocking callable in a QThread and marshal results to the GUI."""
        thread = QtCore.QThread(self)
        worker = _MFDBBackgroundTask(fn)
        worker.moveToThread(thread)
        self._background_tasks.append((thread, worker))

        def _cleanup() -> None:
            try:
                self._background_tasks.remove((thread, worker))
            except ValueError:
                pass
            thread.deleteLater()

        def _failed(error: str) -> None:
            if on_failure is not None:
                on_failure(error)
            else:
                self._set_status_message(f"{label} failed: {error}")

        thread.started.connect(worker.run)
        worker.finished.connect(on_success)
        worker.failed.connect(_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(_cleanup)
        thread.start()

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

    def overview_tab(self) -> QtWidgets.QWidget:
        """Database overview with counts and health summary (Task 11.5)."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        refresh_btn = self._text_icon_button(
            "🔄 Refresh overview", QtWidgets.QStyle.SP_BrowserReload,
            "Reload database overview", self._refresh_overview
        )
        layout.addWidget(refresh_btn)

        self.overview_text = QtWidgets.QPlainTextEdit()
        self.overview_text.setReadOnly(True)
        self.overview_text.setStyleSheet("font-family: monospace; font-size: 11pt;")
        self.overview_text.setMinimumHeight(300)
        layout.addWidget(self.overview_text, stretch=1)

        self._refresh_overview()
        return widget

    def _refresh_overview(self) -> None:
        """Update the overview panel with current database stats."""
        try:
            status = self.client.status() or {}
            parts = [
                "=== MFDB Database Overview ===",
                "",
                f"  User DB:       {status.get('user_database', '—')}",
                f"  Source DB:     {status.get('source_database', '—')}",
                f"  Schema:        {status.get('schema_version', '?')}",
                f"  Samples:       {status.get('sample_count', '?')}",
                f"  Experiments:   {status.get('experiment_count', '?')}",
                f"  Raw data:      {status.get('raw_data_count', '?')}",
                f"  Processed:     {status.get('processed_run_count', '?')}",
                f"  Users:         {status.get('user_count', '?')}",
                f"  Devices:       {status.get('device_count', '?')}",
                f"  Prov. edges:   {status.get('provenance_edge_count', '?')}",
                "",
            ]

            # Quality warnings
            warnings_list = []
            try:
                samples = self.client.list_samples()
                for s in samples:
                    sid = s.get("sample_id", "")
                    if not s.get("description"):
                        warnings_list.append(f"  ⚠ Sample '{sid}' has no description")
            except Exception as exc:
                chisurf.logging.warning("Operation failed: %s", exc)

            if warnings_list:
                parts.append("=== Quality Warnings ===")
                parts.extend(warnings_list)
                parts.append("")
            else:
                parts.append("No quality warnings detected.")
                parts.append("")

            self.overview_text.setPlainText("\n".join(parts))
        except Exception as exc:
            self.overview_text.setPlainText(f"Failed to load overview: {exc}")

    def measurements_tab(self) -> QtWidgets.QWidget:
        """Unified Measurements dock — aggregates raw data, processing runs,
        and processed products (Task 11.1)."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        filter_bar = QtWidgets.QHBoxLayout()
        self.meas_kind_combo = QtWidgets.QComboBox()
        self.meas_kind_combo.addItems(["All", "Raw data", "Processing runs", "Processed products"])
        filter_bar.addWidget(QtWidgets.QLabel("Kind:"))
        filter_bar.addWidget(self.meas_kind_combo)
        self.meas_search_edit = QtWidgets.QLineEdit()
        self.meas_search_edit.setPlaceholderText("Search...")
        filter_bar.addWidget(self.meas_search_edit)
        filter_bar.addWidget(self._text_icon_button(
            "🔄 Refresh", QtWidgets.QStyle.SP_BrowserReload,
            "Refresh measurements list", self._refresh_measurements
        ))
        filter_bar.addStretch()
        layout.addLayout(filter_bar)

        self.measurements_table = QtWidgets.QTableWidget(0, 9)
        apply_compact_table_style(self.measurements_table)
        self.measurements_table.setHorizontalHeaderLabels(
            ["kind", "id", "sample", "experiment", "status", "created", "location", "project", "sample QA"]
        )
        self.measurements_table.horizontalHeader().setStretchLastSection(True)
        self.measurements_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.measurements_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.measurements_table, stretch=1)
        return widget

    def _refresh_measurements(self) -> None:
        """Reload measurements from all sources into the unified table."""
        self.measurements_table.setRowCount(0)
        kind_filter = self.meas_kind_combo.currentText()
        search_text = self.meas_search_edit.text().strip().lower()

        sources: list[tuple[str, str, list[dict[str, Any]]]] = []

        if kind_filter in ("All", "Raw data"):
            try:
                raw = self.client.list_raw_data()
                sources.append(("Raw", "raw_data_id", raw))
            except Exception as exc:
                chisurf.logging.warning("_refresh_measurements(self) -> None: %s", exc)

        if kind_filter in ("All", "Processing runs"):
            try:
                procs = self.client.list_processing_runs()
                sources.append(("Processing", "processing_id", procs))
            except Exception as exc:
                chisurf.logging.warning("_refresh_measurements(self) -> None: %s", exc)

        if kind_filter in ("All", "Processed products"):
            try:
                products = self.client.list_processed_data()
                sources.append(("Processed", "processed_data_id", products))
            except Exception as exc:
                chisurf.logging.warning("Operation failed: %s", exc)

        for kind, id_key, items in sources:
            for item in items:
                item_id = str(item.get(id_key, ""))
                sample = str(item.get("sample_id", ""))
                experiment = str(item.get("experiment_id", ""))
                status = str(item.get("status", ""))
                created = str(item.get("created_at", "") or item.get("acquired_at", ""))
                location = str(
                    item.get("file_path", "") or item.get("url", "") or item.get("folder_path", "")
                )
                project = str(item.get("project_id", ""))
                sample_qa = str(item.get("sample_quality_status", ""))

                if search_text and search_text not in (
                    kind.lower() + item_id.lower() + sample.lower() + project.lower()
                ):
                    continue

                row = self.measurements_table.rowCount()
                self.measurements_table.insertRow(row)
                values = [kind, item_id, sample, experiment, status, created, location, project, sample_qa]
                for col, val in enumerate(values):
                    self.measurements_table.setItem(
                        row, col, QtWidgets.QTableWidgetItem(val)
                    )

        self.status_label.setText(
            f"Measurements: {self.measurements_table.rowCount()} rows"
        )

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
        apply_compact_table_style(self.all_items_table)
        self.all_items_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
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
        self.all_items_table.resizeColumnsToContents()
        # Last column stretches; cap the others so the table doesn't sprawl
        for col in range(self.all_items_table.columnCount() - 1):
            w = self.all_items_table.columnWidth(col)
            self.all_items_table.setColumnWidth(col, min(w, 220))
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

    #: All-items "type" values that differ from the entity-registry key.
    _ALL_ITEMS_ENTITY_ALIAS = {"processed_data": "processed_product"}

    def _jump_to_item(self, payload: dict[str, Any]) -> None:
        item_type = payload.get("type", "")
        item_id = payload.get("id", "")
        entity_key = self._ALL_ITEMS_ENTITY_ALIAS.get(item_type, item_type)
        # Jump to the entity's nav panel and select the record there.
        if entity_key in self._row_by_entity and item_id:
            self._jump_to_entity(entity_key, item_id)
            return
        # Fallback: a bespoke loader for types without a registered entity panel.
        load_method = getattr(self, f"load_{item_type}", None)
        if load_method and item_id:
            try:
                load_method(item_id)
            except Exception as exc:
                chisurf.logging.warning("_jump_to_item: load_%s(%r): %s", item_type, item_id, exc)

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
        settings.remove("group_tab")
        settings.remove("entity_splitter")
        self.resize(1100, 760)
        self._restore_dock_layout()

    def show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About mfdb-admin",
            "mfdb-admin\n\nMultiparametric Fluorescence Database\n\nBrowse, edit, import, and export fluorescence measurements, samples, setups, and analysis runs.",
        )

    def _on_header_label_clicked(self, event) -> None:
        """Populate bundled mock data after ten clicks on the header label."""
        if event.button() != QtCore.Qt.LeftButton:
            return
        self._mock_data_click_count += 1
        if self._mock_data_click_count < 10:
            return
        self._mock_data_click_count = 0
        answer = QtWidgets.QMessageBox.question(
            self,
            "Populate mock MFDB data",
            "Populate the MFDB with bundled smFRET mock data from test fixtures?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.populate_mock_data()

    def populate_mock_data(self) -> None:
        """Populate the active MFDB with bundled mock/demo data."""
        if self._mock_data_task_running:
            self._set_status_message("Mock data population is already running.")
            return

        self._mock_data_task_running = True
        previous_transport_state = bool(getattr(self, "transport_connected", False))
        self._set_transport_connected(False)
        self._set_status_message(
            "Populating MFDB demo data in background. The GUI remains usable."
        )
        if hasattr(self, "preview_edit"):
            self.preview_edit.setPlainText(
                "Populating bundled smFRET demo data from test fixtures...\n"
                "This may take a moment for a fresh database."
            )

        def _succeeded(summary: Any) -> None:
            self._mock_data_task_running = False
            if isinstance(summary, dict) and hasattr(self, "preview_edit"):
                self.preview_edit.setPlainText(
                    json.dumps(summary, indent=2, default=str)
                )
            raw_count = (
                len(summary.get("raw_data_ids", []))
                if isinstance(summary, dict)
                else 0
            )
            sample_id = summary.get("sample_id", "?") if isinstance(summary, dict) else "?"
            self._set_status_message(
                "Mock data populated: "
                f"{raw_count} raw files, sample {sample_id}. Refreshing tables..."
            )
            QtCore.QTimer.singleShot(0, self.refresh)

        def _failed(error: str) -> None:
            self._mock_data_task_running = False
            self._set_transport_connected(previous_transport_state)
            self._set_status_message(f"Mock data population failed: {error}")
            if hasattr(self, "preview_edit"):
                self.preview_edit.setPlainText(
                    "Mock data population failed:\n\n"
                    f"{error}"
                )

        self._run_background_task(
            label="Mock data population",
            fn=self.client.populate_mock_data,
            on_success=_succeeded,
            on_failure=_failed,
        )

    def sample_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(splitter, stretch=1)

        header_widget = QtWidgets.QWidget()
        header_layout = QtWidgets.QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(2)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.sample_detail_widget)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        header_layout.addWidget(scroll, stretch=1)

        # Reassign widget references for compatibility with the rest of the class
        self.sample_id_edit = self.sample_detail_widget.widgets["sample_id"]
        self.uuid_edit = self.sample_detail_widget.widgets["sample_uuid"]
        self.description_edit = self.sample_detail_widget.widgets["description"]
        self.details_edit = self.sample_detail_widget.widgets["details"]
        self.num_probes_spin = self.sample_detail_widget.widgets["num_of_probes"]
        self.solvent_edit = self.sample_detail_widget.widgets["solvent_phase"]
        self.condition_id_edit = self.sample_detail_widget.widgets["sample_condition_id"]
        self.assembly_id_edit = self.sample_detail_widget.widgets["entity_assembly_id"]
        self.project_edit = self.sample_detail_widget.widgets["project_id"]
        self.measured_by_combo = self.sample_detail_widget.widgets["measured_by_user_id"]
        self.measured_device_combo = self.sample_detail_widget.widgets["measured_by_device_id"]
        self.measured_at_edit = self.sample_detail_widget.widgets["measured_at"]

        # Customize placeholder for sample_id_edit
        self.sample_id_edit.setPlaceholderText("Type sample id (autocomplete searches existing)")

        btn_row = QtWidgets.QHBoxLayout()
        new_btn = self._text_icon_button("🧪 New", QtWidgets.QStyle.SP_FileDialogNewFolder, "Create new sample", self.new_sample)
        save_btn = self._text_icon_button("💾 Save", QtWidgets.QStyle.SP_DialogSaveButton, "Save sample", self.save_sample)
        delete_btn = self._text_icon_button("🗑 Delete", QtWidgets.QStyle.SP_TrashIcon, "Delete sample", self.delete_sample)
        clear_btn = self._text_icon_button("🧹 Clear", QtWidgets.QStyle.SP_DialogResetButton, "Clear form", self.clear_form)
        full_btn = self._text_icon_button(
            "Full description",
            QtWidgets.QStyle.SP_FileDialogDetailedView,
            "Show PRD-02 nested sample description",
            self.show_sample_full_description,
        )
        validate_btn = self._text_icon_button(
            "Validate export",
            QtWidgets.QStyle.SP_DialogApplyButton,
            "Validate selected sample for FLR CIF export",
            self.validate_selected_sample_export,
        )
        btn_row.addWidget(new_btn)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(delete_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(full_btn)
        btn_row.addWidget(validate_btn)
        btn_row.addStretch()
        header_layout.addLayout(btn_row)
        splitter.addWidget(header_widget)

        self.sample_subtabs = QtWidgets.QTabWidget()
        self.sample_subtabs.addTab(self._entities_sub_panel(), "Entities")
        self.sample_subtabs.addTab(self._probes_sub_panel(), "Probes & Positions")
        self.sample_subtabs.addTab(self._fret_pairs_sub_panel(), "FRET Pairs")
        self.sample_subtabs.addTab(self._condition_sub_panel(), "Condition")
        self.sample_subtabs.addTab(self._full_description_panel(), "Full Description")
        splitter.addWidget(self.sample_subtabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._setup_sample_id_completer()
        self.sample_id_edit.textChanged.connect(self._auto_generate_uuid)
        self.sample_id_edit.editingFinished.connect(self._on_sample_id_edited)
        self.condition_id_edit.editingFinished.connect(self._auto_fill_condition)
        return widget

    def _entities_sub_panel(self) -> QtWidgets.QWidget:
        """Return the structured sample entities editor."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._text_icon_button("Add entity", QtWidgets.QStyle.SP_FileDialogNewFolder, "Add entity row", self.add_entity_row))
        buttons.addWidget(self._text_icon_button("Remove", QtWidgets.QStyle.SP_TrashIcon, "Remove selected entity row", self.remove_entity_row))
        buttons.addStretch()
        layout.addLayout(buttons)
        self.sample_entities_table = QtWidgets.QTableWidget(0, 5)
        self.sample_entities_table.setHorizontalHeaderLabels(
            ["entity_id", "name", "type", "sequence", "details"]
        )
        self.sample_entities_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.sample_entities_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.sample_entities_table, stretch=1)
        return panel

    def _probes_sub_panel(self) -> QtWidgets.QWidget:
        """Return the structured sample probe-position editor."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._text_icon_button("Add probe", QtWidgets.QStyle.SP_FileDialogNewFolder, "Add probe row", self.add_probe_position_row))
        buttons.addWidget(self._text_icon_button("Remove", QtWidgets.QStyle.SP_TrashIcon, "Remove selected probe row", self.remove_probe_position_row))
        buttons.addStretch()
        layout.addLayout(buttons)
        self.sample_probes_table = QtWidgets.QTableWidget(0, 12)
        self.sample_probes_table.setHorizontalHeaderLabels(
            [
                "probe_name", "entity", "seq_id", "comp_id", "asym_id", "atom_id",
                "mutation", "modification", "abs_nm", "em_nm", "QY", "sample_probe_id",
            ]
        )
        self.sample_probes_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.sample_probes_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.sample_probes_table, stretch=1)
        return panel

    def _fret_pairs_sub_panel(self) -> QtWidgets.QWidget:
        """Return the structured sample FRET-pair editor."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._text_icon_button("Add pair", QtWidgets.QStyle.SP_FileDialogNewFolder, "Add FRET pair row", self.add_fret_pair_row))
        buttons.addWidget(self._text_icon_button("Remove", QtWidgets.QStyle.SP_TrashIcon, "Remove selected FRET pair row", self.remove_fret_pair_row))
        buttons.addStretch()
        layout.addLayout(buttons)
        self.fret_pairs_table = QtWidgets.QTableWidget(0, 7)
        self.fret_pairs_table.setHorizontalHeaderLabels(
            ["donor", "acceptor", "R0 nm", "kappa^2", "n_refr", "overlap", "id"]
        )
        self.fret_pairs_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.fret_pairs_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.fret_pairs_table, stretch=1)
        return panel

    def _condition_sub_panel(self) -> QtWidgets.QWidget:
        """Return the sample-scoped condition editor used by structured saves."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.sample_condition_id_field = QtWidgets.QLineEdit()
        self.sample_ph_spin = QtWidgets.QDoubleSpinBox()
        self.sample_ph_spin.setRange(-1.0, 14.0)
        self.sample_ph_spin.setSpecialValueText("auto")
        self.sample_temperature_spin = QtWidgets.QDoubleSpinBox()
        self.sample_temperature_spin.setRange(0.0, 400.0)
        self.sample_temperature_spin.setSpecialValueText("auto")
        self.sample_ionic_spin = QtWidgets.QDoubleSpinBox()
        self.sample_ionic_spin.setRange(0.0, 10.0)
        self.sample_ionic_spin.setSpecialValueText("auto")
        self.sample_buffer_edit = QtWidgets.QLineEdit()
        self.sample_condition_details_edit = QtWidgets.QPlainTextEdit()
        self.sample_condition_details_edit.setMinimumHeight(60)
        layout.addRow("Condition id", self.sample_condition_id_field)
        layout.addRow("pH", self.sample_ph_spin)
        layout.addRow("Temperature [K]", self.sample_temperature_spin)
        layout.addRow("Ionic strength [M]", self.sample_ionic_spin)
        layout.addRow("Buffer", self.sample_buffer_edit)
        layout.addRow("Details", self.sample_condition_details_edit)
        return panel

    def _full_description_panel(self) -> QtWidgets.QWidget:
        """Return the JSON full-description and validation preview panel."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._text_icon_button("Refresh", QtWidgets.QStyle.SP_BrowserReload, "Refresh full description", self.refresh_full_description_panel))
        buttons.addWidget(self._text_icon_button("Copy", QtWidgets.QStyle.SP_DialogSaveButton, "Copy JSON", self.copy_full_description_json))
        buttons.addWidget(self._text_icon_button("Validate", QtWidgets.QStyle.SP_DialogApplyButton, "Validate export", self.validate_selected_sample_export))
        buttons.addStretch()
        layout.addLayout(buttons)
        self.full_description_edit = QtWidgets.QPlainTextEdit()
        self.full_description_edit.setReadOnly(True)
        layout.addWidget(self.full_description_edit, stretch=2)
        self.validation_list = QtWidgets.QListWidget()
        layout.addWidget(self.validation_list, stretch=1)
        return panel

    def add_entity_row(self) -> None:
        """Append an editable entity row to the active entity tables."""
        table = self._active_entities_table()
        row = table.rowCount()
        table.insertRow(row)
        values = ["", "", ENTITY_TYPES[0] if ENTITY_TYPES else "polymer", "", ""]
        for column, value in enumerate(values):
            table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def remove_entity_row(self) -> None:
        """Remove selected entity rows from the active entity table."""
        table = self._active_entities_table()
        for row in sorted({item.row() for item in table.selectedItems()}, reverse=True):
            table.removeRow(row)

    def add_probe_position_row(self) -> None:
        """Append an editable probe-position row to the sample probe table."""
        table = self.sample_probes_table
        row = table.rowCount()
        table.insertRow(row)
        probe_name = COMMON_PROBE_NAMES[0] if COMMON_PROBE_NAMES else ""
        defaults = DEFAULT_FLUOROPHORE_SPECTRA.get(probe_name, {})
        values = [
            probe_name, "", "", "", "A", "", "no", "no",
            defaults.get("absorption_wavelength_nm", ""),
            defaults.get("emission_wavelength_nm", ""),
            defaults.get("quantum_yield", ""),
            "",
        ]
        for column, value in enumerate(values):
            table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def remove_probe_position_row(self) -> None:
        """Remove selected probe-position rows."""
        table = self.sample_probes_table
        for row in sorted({item.row() for item in table.selectedItems()}, reverse=True):
            table.removeRow(row)

    def add_fret_pair_row(self) -> None:
        """Append an editable FRET pair row."""
        row = self.fret_pairs_table.rowCount()
        self.fret_pairs_table.insertRow(row)
        values = ["", "", "", "0.6666667", "1.4", "", ""]
        for column, value in enumerate(values):
            self.fret_pairs_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def remove_fret_pair_row(self) -> None:
        """Remove selected FRET pair rows."""
        for row in sorted({item.row() for item in self.fret_pairs_table.selectedItems()}, reverse=True):
            self.fret_pairs_table.removeRow(row)

    def _active_entities_table(self) -> QtWidgets.QTableWidget:
        """Return the entity table currently used for sample editing."""
        return getattr(self, "sample_entities_table", self.entities_table)

    def refresh_full_description_panel(self) -> None:
        """Refresh the embedded full-description JSON panel."""
        sample_id = self._active_sample_id()
        if not sample_id:
            self.status_label.setText("Select or enter a sample id")
            return
        description = self.client.get_sample_full_description(sample_id)
        self.full_description_edit.setPlainText(
            json.dumps(description, indent=2, default=str)
        )
        self.status_label.setText(f"Loaded full description for {sample_id}")

    def copy_full_description_json(self) -> None:
        """Copy the embedded full-description JSON to the clipboard."""
        QtWidgets.QApplication.clipboard().setText(
            self.full_description_edit.toPlainText()
        )

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
            sample = self.client._call("mfdb.samples.get", {"sample_id": cid})
            cond = (sample.get("sample") or {}).get("condition")
            if cond:
                self.condition_detail_widget.set_data(cond)
        except Exception as exc:
            chisurf.logging.warning("_auto_fill_condition(self) -> None: %s", exc)

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
        except Exception as exc:
            chisurf.logging.warning("_setup_sample_id_completer(self) -> None: %s", exc)

    def _on_sample_id_edited(self) -> None:
        sample_id = self.sample_id_edit.text().strip()
        if not sample_id:
            return
        try:
            sample = self.client.get_sample(sample_id)
            if sample:
                self.load_sample(sample_id)
        except Exception as exc:
            chisurf.logging.warning("_on_sample_id_edited(self) -> None: %s", exc)

    def condition_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.condition_id_field = self.condition_detail_widget.widgets["condition_id"]
        self.condition_id_field.setPlaceholderText("Type condition id to auto-fill from DB")
        self.condition_id_field.editingFinished.connect(self._auto_fill_condition_details)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.condition_detail_widget)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        layout.addWidget(scroll, stretch=1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._text_icon_button(
            "💾 Save", QtWidgets.QStyle.SP_DialogSaveButton, "Save condition", self.save_condition
        ))
        buttons.addWidget(self._text_icon_button(
            "🧹 Clear", QtWidgets.QStyle.SP_DialogResetButton, "Clear form", self.clear_condition_form
        ))
        buttons.addStretch()
        layout.addLayout(buttons)
        return widget

    def save_condition(self) -> None:
        data = self.condition_detail_widget.get_data()
        cid = (data.get("condition_id") or "").strip()
        if not cid:
            self.status_label.setText("Condition id is required")
            return
        try:
            condition = {
                "condition_id": cid,
                "ph": None if data.get("ph") in (None, 0, 0.0) else float(data["ph"]),
                "temperature": None if data.get("temperature") in (None, 0, 0.0) else float(data["temperature"]),
                "ionic_strength": None if data.get("ionic_strength") in (None, 0, 0.0) else float(data["ionic_strength"]),
                "buffer_composition": data.get("buffer_composition") or None,
                "details": data.get("details") or None,
            }
            self.client.save_sample_condition(condition)
            self.status_label.setText(f"Condition '{cid}' saved")
        except Exception as exc:
            self.status_label.setText(f"Failed to save condition: {exc}")

    def clear_condition_form(self) -> None:
        self.condition_detail_widget.set_data({})

    def _auto_fill_condition_details(self) -> None:
        cid = self.condition_detail_widget.widgets["condition_id"].text().strip()
        if not cid or self._loading:
            return
        try:
            row = self.client.get_sample_condition(cid)
            if row:
                self.condition_detail_widget.set_data(row)
                self.status_label.setText(f"Auto-filled condition '{cid}'")
        except Exception as exc:
            chisurf.logging.warning("_auto_fill_condition_details(self) -> None: %s", exc)

    def entities_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._text_icon_button("Refresh", QtWidgets.QStyle.SP_BrowserReload, "Refresh entities", self.refresh_entities_table))
        buttons.addWidget(self._text_icon_button("New entity", QtWidgets.QStyle.SP_FileDialogNewFolder, "Add entity row", self.add_entity_row))
        buttons.addWidget(self._text_icon_button("Save row", QtWidgets.QStyle.SP_DialogSaveButton, "Save selected entity", self.save_selected_entity))
        buttons.addWidget(self._text_icon_button("Delete", QtWidgets.QStyle.SP_TrashIcon, "Delete selected entity", self.delete_selected_entity))
        buttons.addStretch()
        layout.addLayout(buttons)
        self.entities_table = QtWidgets.QTableWidget(0, 5)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(splitter, stretch=1)
        self.entities_table.setHorizontalHeaderLabels(
            ["entity_id", "name", "type", "sequence", "details"]
        )
        self.entities_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.entities_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.entities_table)
        return widget

    def probes_tab(self) -> QtWidgets.QWidget:
        self.probes_table = QtWidgets.QTableWidget(0, 11)
        self.probes_table.setHorizontalHeaderLabels(
            [
                "id", "name", "category", "origin", "link_type", "reactive",
                "center_atom", "abs_nm", "em_nm", "QY", "ext_coeff",
            ]
        )
        self.probes_table.horizontalHeader().setStretchLastSection(True)
        self.probes_table.itemSelectionChanged.connect(self.load_probe)
        self.probes_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.probes_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._install_table_context_menu(
            self.probes_table,
            item_kind="probe",
            id_col=0,
            delete_one_fn=lambda pid: self.client.delete_probe(int(pid)),
        )
        
        new_probe_btn = self._text_icon_button("New probe", QtWidgets.QStyle.SP_FileDialogNewFolder, "Add probe", self.new_probe)
        refresh_btn = self._text_icon_button("Refresh", QtWidgets.QStyle.SP_BrowserReload, "Refresh probes", self.fill_probes)
        
        return self._create_standard_dock_tab(
            table=self.probes_table,
            detail_widget=self.probe_detail_widget,
            save_slot=self.save_selected_probe,
            delete_slot=self.delete_probe,
            extra_buttons=[new_probe_btn, refresh_btn],
        )

    def positions_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._text_icon_button(
            "🔄 Refresh", QtWidgets.QStyle.SP_BrowserReload,
            "Reload probe positions", self.fill_positions
        ))
        buttons.addStretch()
        layout.addLayout(buttons)
        self.positions_table = QtWidgets.QTableWidget(0, 13)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        layout.addWidget(splitter, stretch=1)
        self.positions_table.setHorizontalHeaderLabels(
            [
                "sample_probe_id",
                "sample",
                "probe_id",
                "probe",
                "entity",
                "chain",
                "residue",
                "atom_id",
                "mutation",
                "modification",
                "auth_name",
                "type",
                "description",
            ]
        )
        self.positions_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.positions_table)
        return widget

    def fret_pairs_tab(self) -> QtWidgets.QWidget:
        """Standalone FRET Pairs tab — view, add, edit, delete Forster radius records."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        filter_bar = QtWidgets.QHBoxLayout()
        self.fp_sample_combo = QtWidgets.QComboBox()
        self.fp_sample_combo.setEditable(True)
        self.fp_sample_combo.setPlaceholderText("Select sample...")
        filter_bar.addWidget(QtWidgets.QLabel("Sample:"))
        filter_bar.addWidget(self.fp_sample_combo)
        filter_bar.addWidget(self._text_icon_button(
            "🔄 Refresh", QtWidgets.QStyle.SP_BrowserReload,
            "Refresh FRET pairs for selected sample", self._refresh_fret_pairs_tab
        ))
        filter_bar.addStretch()
        layout.addLayout(filter_bar)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._text_icon_button(
            "Add pair", QtWidgets.QStyle.SP_FileDialogNewFolder,
            "Add FRET pair", self._add_fret_pair_standalone
        ))
        buttons.addWidget(self._text_icon_button(
            "Delete", QtWidgets.QStyle.SP_TrashIcon,
            "Delete selected FRET pair", self._delete_fret_pair_standalone
        ))
        buttons.addStretch()
        layout.addLayout(buttons)
        self.standalone_fret_pairs_table = QtWidgets.QTableWidget(0, 9)
        self.standalone_fret_pairs_table.setHorizontalHeaderLabels(
            ["id", "sample", "donor", "acceptor", "R₀ (nm)", "κ²", "n", "overlap_integral", "details"]
        )
        self.standalone_fret_pairs_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.standalone_fret_pairs_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.standalone_fret_pairs_table, stretch=1)
        return widget

    def _refresh_fret_pairs_tab(self) -> None:
        """Reload FRET pairs for the selected sample in the standalone tab."""
        sample_id = self.fp_sample_combo.currentText().strip()
        if not sample_id:
            self.status_label.setText("Select a sample to load FRET pairs")
            return
        pairs = self.client.list_fret_pairs(sample_id)
        self.standalone_fret_pairs_table.setRowCount(0)
        for pair in pairs:
            row = self.standalone_fret_pairs_table.rowCount()
            self.standalone_fret_pairs_table.insertRow(row)
            values = [
                pair.get("forster_radius_id", ""),
                pair.get("sample_id", ""),
                pair.get("donor_probe", ""),
                pair.get("acceptor_probe", ""),
                pair.get("forster_radius_nm") or pair.get("forster_radius", ""),
                pair.get("kappa_squared", ""),
                pair.get("refractive_index") or pair.get("index_of_refraction", ""),
                pair.get("overlap_integral", ""),
                pair.get("details", ""),
            ]
            for column, value in enumerate(values):
                self.standalone_fret_pairs_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )
        self.status_label.setText(f"Loaded {len(pairs)} FRET pairs for {sample_id}")

    def _add_fret_pair_standalone(self) -> None:
        """Open a dialog to create a new FRET pair."""
        sample_id = self.fp_sample_combo.currentText().strip()
        if not sample_id:
            self.status_label.setText("Select a sample first")
            return
        donor, ok1 = QtWidgets.QInputDialog.getText(self, "Donor probe", "Donor probe name:")
        if not ok1 or not donor.strip():
            return
        acceptor, ok2 = QtWidgets.QInputDialog.getText(self, "Acceptor probe", "Acceptor probe name:")
        if not ok2 or not acceptor.strip():
            return
        r0, ok3 = QtWidgets.QInputDialog.getDouble(self, "Förster radius", "R₀ (nm):", 5.0, 0.0, 20.0, 2)
        if not ok3:
            return
        try:
            probes = self.client.list_probes()
            donor_id = next((p["probe_id"] for p in probes if p.get("chromophore_name", "").lower() == donor.strip().lower()), None)
            acceptor_id = next((p["probe_id"] for p in probes if p.get("chromophore_name", "").lower() == acceptor.strip().lower()), None)
            if not donor_id or not acceptor_id:
                self.status_label.setText("Could not resolve probe names to IDs")
                return
            result = self.client.save_fret_pair({
                "sample_id": sample_id,
                "donor_probe_id": donor_id,
                "acceptor_probe_id": acceptor_id,
                "forster_radius": r0,
                "kappa_squared": 0.6666667,
                "refractive_index": 1.4,
            })
            self.status_label.setText(f"Created FRET pair: {result.get('fret_pair', {}).get('forster_radius_id', '')}")
            self._refresh_fret_pairs_tab()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Failed to create FRET pair", str(exc))

    def _delete_fret_pair_standalone(self) -> None:
        """Delete the selected FRET pair."""
        selected = self.standalone_fret_pairs_table.selectedItems()
        if not selected:
            self.status_label.setText("Select a FRET pair row to delete")
            return
        row = selected[0].row()
        pair_id = self.standalone_fret_pairs_table.item(row, 0).text()
        reply = QtWidgets.QMessageBox.question(
            self, "Delete FRET pair",
            f"Delete FRET pair '{pair_id}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.client.delete_fret_pair(pair_id)
            self.status_label.setText(f"Deleted FRET pair {pair_id}")
            self._refresh_fret_pairs_tab()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete failed", str(exc))

    def _fill_fp_sample_combo(self) -> None:
        """Populate the FRET Pairs tab sample combo."""
        if not hasattr(self, "fp_sample_combo"):
            return
        current = self.fp_sample_combo.currentText()
        self.fp_sample_combo.clear()
        try:
            samples = self.client.list_samples()
            for s in samples:
                sid = str(s.get("sample_id", ""))
                desc = str(s.get("description", ""))
                label = f"{sid} — {desc}" if desc else sid
                self.fp_sample_combo.addItem(label, sid)
        except Exception as exc:
            chisurf.logging.warning("_fill_fp_sample_combo(self) -> None: %s", exc)
        idx = self.fp_sample_combo.findText(current)
        if idx >= 0:
            self.fp_sample_combo.setCurrentIndex(idx)

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
        preview_button = self._text_icon_button("👁 Preview CIF", QtWidgets.QStyle.SP_FileDialogDetailedView, "Preview flrCIF output in the text area below", self.preview_cif)
        export_table_button = self._text_icon_button("📊 Export table CSV/XLSX", QtWidgets.QStyle.SP_FileIcon, "Export sample table", self.export_table)
        buttons.addWidget(import_button)
        buttons.addWidget(export_button)
        buttons.addWidget(preview_button)
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

        new_user_button = self._text_icon_button(
            "👤 New user",
            QtWidgets.QStyle.SP_FileDialogNewFolder,
            "Prepare the form for a new user (auto-generates UUID)",
            self.new_user,
        )
        change_password_button = self._text_icon_button(
            "🔑 Password", QtWidgets.QStyle.SP_DialogApplyButton, "Change password", self.change_user_password
        )

        return self._create_standard_dock_tab(
            table=self.users_table,
            detail_widget=self.user_detail_widget,
            save_slot=self.save_user,
            delete_slot=self.delete_user,
            extra_buttons=[new_user_button, change_password_button],
        )

    def devices_tab(self) -> QtWidgets.QWidget:
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
        return self._create_standard_dock_tab(
            table=self.devices_table,
            detail_widget=self.device_detail_widget,
            save_slot=self.save_device,
            delete_slot=self.delete_device,
        )

    def branches_tab(self) -> QtWidgets.QWidget:
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

        fork_branch_button = self._text_icon_button(
            "🍴 Fork",
            QtWidgets.QStyle.SP_FileDialogNewFolder,
            "Prefill branch from selected head",
            self.prefill_branch_fork,
        )

        return self._create_standard_dock_tab(
            table=self.branches_table,
            detail_widget=self.branch_detail_widget,
            save_slot=self.save_branch,
            delete_slot=self.delete_branch,
            extra_widgets_top=[user_group],
            extra_buttons=[fork_branch_button],
        )

    def experiment_types_tab(self) -> QtWidgets.QWidget:
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
        return self._create_standard_dock_tab(
            table=self.experiment_types_table,
            detail_widget=self.experiment_type_detail_widget,
            save_slot=self.save_experiment_type,
            delete_slot=self.delete_experiment_type,
        )

    def experiments_tab(self) -> QtWidgets.QWidget:
        filter_widget = self._build_filter_bar([
            {"label": "Type:", "attr": "experiment_type_combo",
             "default_data": -1, "cb": self._filter_experiments},
            {"label": "Sample:", "attr": "experiment_sample_combo",
             "cb": self._filter_experiments},
        ])

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

        data_container = QtWidgets.QWidget()
        data_layout = QtWidgets.QVBoxLayout(data_container)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(2)
        data_header = QtWidgets.QLabel("Experiment data / links")
        data_header.setStyleSheet("font-weight: bold; margin-top: 4px;")
        self.experiment_data_table = QtWidgets.QTableWidget(0, 8)
        self.experiment_data_table.setHorizontalHeaderLabels(
            ["id", "type", "mode", "path/url/folder", "mime", "checksum", "reading options", "details"]
        )
        self.experiment_data_table.horizontalHeader().setStretchLastSection(True)
        self.experiment_data_table.itemSelectionChanged.connect(self.load_experiment_data)
        data_layout.addWidget(data_header)
        data_layout.addWidget(self.experiment_data_table, stretch=1)

        add_data_button = self._text_icon_button("➕ Add", QtWidgets.QStyle.SP_FileDialogNewFolder, "Add data row", self.add_experiment_data_row)
        save_data_button = self._text_icon_button("💾 Save data", QtWidgets.QStyle.SP_DialogSaveButton, "Save data", self.save_experiment_data)
        delete_data_button = self._text_icon_button("🗑 Del data", QtWidgets.QStyle.SP_TrashIcon, "Delete data", self.delete_experiment_data)
        open_data_button = self._text_icon_button("📂 Open", QtWidgets.QStyle.SP_DialogOpenButton, "Open linked data", self.open_experiment_data)

        return self._create_standard_dock_tab(
            table=self.experiments_table,
            detail_widget=self.experiment_detail_widget,
            save_slot=self.save_experiment,
            delete_slot=self.delete_experiment,
            extra_widgets_top=[filter_widget],
            extra_buttons=[add_data_button, save_data_button, delete_data_button, open_data_button],
            extra_widgets_bottom=[data_container],
        )

    def _filter_experiments(self) -> None:
        type_id = self.experiment_type_combo.currentData()
        sample_id = self.experiment_sample_combo.currentData() or None
        if type_id == -1:
            type_id = None
        self.fill_experiment_table(sample_id=sample_id, type_id=type_id)

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
        except Exception as exc:
            chisurf.logging.warning("_refresh_sample_id_completer(self) -> None: %s", exc)

    def _set_transport_connected(self, connected: bool) -> None:
        self.transport_connected = bool(connected)
        for action in getattr(self, "_transport_actions", []):
            if action is getattr(self, "refresh_action", None):
                action.setEnabled(True)
            else:
                action.setEnabled(bool(connected))


    def save_raw_data(self) -> None:
        raw_id = self.raw_id_edit.text()
        if not raw_id: return
        import json
        details = self.raw_details_edit.toPlainText()
        try:
            details_dict = json.loads(details) if details else {}
        except json.JSONDecodeError:
            QtWidgets.QMessageBox.warning(self, "Invalid JSON", "Details field must be valid JSON.")
            return
        
        payload = {
            "raw_data_id": raw_id,
            "experiment_id": self.raw_exp_edit.text(),
            "data_type": self.raw_type_edit.text(),
            "storage_mode": self.raw_storage_edit.text(),
            "file_path": self.raw_path_edit.text(),
            "details": details_dict
        }
        try:
            self.client._call("mfdb.raw_data.save", {"raw_data": payload})
            QtWidgets.QMessageBox.information(self, "Success", "Saved successfully.")
            self.fill_raw_data_table()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save: {e}")

    def save_processed_product(self) -> None:
        data = self.processed_product_detail_widget.get_data()
        prod_id = data.get("product_id") or data.get("processed_data_id")
        if not prod_id:
            return
        import json
        details_raw = data.get("details") or data.get("metadata_json") or ""
        try:
            details_dict = json.loads(details_raw) if details_raw else {}
        except (json.JSONDecodeError, TypeError):
            details_dict = {}
        payload = {
            "product_id": prod_id,
            "processing_run_id": data.get("processing_id") or "",
            "product_type": data.get("product_type") or "",
            "storage_mode": data.get("storage_mode") or "",
            "file_path": data.get("location") or "",
            "details": details_dict,
        }
        try:
            self.client._call("mfdb.processed_data.save", {"product": payload})
            QtWidgets.QMessageBox.information(self, "Success", "Saved successfully.")
            self.fill_processed_products_table()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save: {e}")

    def save_processing_run(self) -> None:
        pass # To be implemented via client._call

    def save_analysis(self) -> None:
        pass

    def save_object(self) -> None:
        pass

    def save_experiment_tab(self) -> None:
        pass

    def save_project_tab(self) -> None:
        pass

    def save_experiment_type(self) -> None:
        pass

    def refresh(self) -> None:
        if getattr(self, "_refresh_in_progress", False):
            return
        self._refresh_in_progress = True
        self._loading = True
        self._failures: list[str] = []
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
            self._set_transport_connected(transport_ok)
        except Exception as exc:
            self.status_label.setText(f"MFDB transport unavailable: {exc}")
            self._set_transport_connected(False)
            self._loading = False
            self._refresh_in_progress = False
            return

        self._loaded_tabs.clear()
        self._load_active_tab()
        # Also refresh the currently visible entity dock (if any).
        self._refresh_active_panel()
        self._loading = False
        self._refresh_in_progress = False

    def _on_nav_changed(self, index: int) -> None:
        """Load the selected panel (shell) then populate its aggregate data."""
        super()._on_nav_changed(index)
        if self._is_deleted() or getattr(self, "_loading", False):
            return
        if 0 <= index < len(self.panels):
            name = self.panels[index].get("name")
            if name:
                self._load_tab_data(name)

    def _load_active_tab(self) -> None:
        if self._is_deleted():
            return
        idx = self.nav_list.currentRow()
        if 0 <= idx < len(self.panels):
            name = self.panels[idx].get("name")
            if name and not self.panels[idx].get("separator"):
                self._load_tab_data(name)

    def _load_tab_data(self, tab_name: str) -> None:
        if tab_name in self._loaded_tabs:
            return

        populators = {
            "Overview": [self._refresh_overview],
            "All items": [self._populate_all_items],
            "Measurements": [self._refresh_measurements],
        }

        fns = populators.get(tab_name, [])
        if not fns:
            self._loaded_tabs.add(tab_name)
            return

        self._loading = True
        try:
            for fn in fns:
                try:
                    fn()
                except Exception as exc:
                    self._failures.append(f"{tab_name} populator: {exc}")
            self._loaded_tabs.add(tab_name)
        finally:
            self._loading = False

        if self._failures:
            short = self._failures[0]
            if len(self._failures) > 1:
                short = f"{short} (+{len(self._failures) - 1} more)"
            self.status_label.setText(f"{self.status_label.text()}  ⚠ partial refresh — {short}")

    def clear_form(self) -> None:
        for widget in (
            self.user_detail_widget,
            self.device_detail_widget,
            self.sample_detail_widget,
            self.experiment_detail_widget,
            self.experiment_type_detail_widget,
            self.branch_detail_widget,
            self.probe_detail_widget,
            self.setup_detail_widget,
            self.project_detail_widget,
            self.raw_data_detail_widget,
            self.processing_run_detail_widget,
            self.processed_product_detail_widget,
            self.object_detail_widget,
            self.analysis_detail_widget,
            self.condition_detail_widget,
        ):
            if hasattr(widget, "set_data"):
                widget.set_data({})

        self.branches_table.setRowCount(0)
        self.entities_table.setRowCount(0)
        if hasattr(self, "sample_entities_table"):
            self.sample_entities_table.setRowCount(0)
        if hasattr(self, "sample_probes_table"):
            self.sample_probes_table.setRowCount(0)
        if hasattr(self, "fret_pairs_table"):
            self.fret_pairs_table.setRowCount(0)
        if hasattr(self, "full_description_edit"):
            self.full_description_edit.clear()
        if hasattr(self, "validation_list"):
            self.validation_list.clear()
        if hasattr(self, "sample_condition_id_field"):
            self.sample_condition_id_field.clear()
            self.sample_ph_spin.setValue(0)
            self.sample_temperature_spin.setValue(0)
            self.sample_ionic_spin.setValue(0)
            self.sample_buffer_edit.clear()
            self.sample_condition_details_edit.clear()
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
        sample = self.client.get_sample_full_description(sample_id) or {}
        if not sample:
            sample = self.client.get_sample(sample_id) or {}
        self._loading = True
        try:
            self.sample_id_edit.setText(sample.get("sample_id", sample_id))
            self.uuid_edit.setText(sample.get("sample_uuid", ""))
            self.description_edit.setText(sample.get("description") or sample.get("display_name", ""))
            self.details_edit.setPlainText(sample.get("details", ""))
            self.num_probes_spin.setValue(int(sample.get("num_of_probes") or len(sample.get("probes", [])) or 0))
            self.solvent_edit.setCurrentText(sample.get("solvent_phase") or "liquid")
            self.condition_id_edit.setText(sample.get("sample_condition_id", ""))
            self.assembly_id_edit.setText(sample.get("entity_assembly_id", ""))
            self.project_edit.setText(sample.get("project_id", ""))
            self.measured_at_edit.setText(sample.get("measured_at", ""))
            condition = sample.get("condition") or {}
            temp_val = (
                condition.get("temperature") or condition.get("temperature_k")
            )
            cond_data = {
                "condition_id": condition.get("condition_id", ""),
                "ph": condition.get("ph"),
                "temperature": temp_val,
                "ionic_strength": (
                    condition.get("ionic_strength") or condition.get("salt_concentration_m")
                ),
                "buffer_composition": condition.get("buffer_composition", ""),
                "details": condition.get("details", ""),
            }
            self.condition_detail_widget.set_data(cond_data)
            if hasattr(self, "sample_condition_id_field"):
                self.sample_condition_id_field.setText(condition.get("condition_id", ""))
                self.sample_ph_spin.setValue(float(condition.get("ph") or 0))
                self.sample_temperature_spin.setValue(float(temperature or 0))
                self.sample_ionic_spin.setValue(float(condition.get("ionic_strength") or condition.get("salt_concentration_m") or 0))
                self.sample_buffer_edit.setText(condition.get("buffer_composition", ""))
                self.sample_condition_details_edit.setPlainText(condition.get("details", ""))
            self.fill_entities(sample.get("entities", []))
            self.fill_probes_positions(sample.get("probes", []))
            self.fill_probes()
            self.fill_positions(sample.get("sample_probes", []))
            self.fill_fret_pairs(sample.get("fret_pairs", []))
            if hasattr(self, "full_description_edit"):
                self.full_description_edit.setPlainText(
                    json.dumps(sample, indent=2, default=str)
                )
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
        except Exception as exc:
            chisurf.logging.warning("load_sample(%r): %s", sample_id, exc)
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
        try:
            scoped_entities = self.client.list_entities(self.current_sample_id) if self.current_sample_id else []
            if scoped_entities:
                entities = scoped_entities
        except Exception as exc:
            chisurf.logging.warning("fill_entities(self, entities: list[dict[str, Any]]) -> None: %s", exc)
        for table in self._entity_tables():
            table.setRowCount(0)
        count = 0
        for entity in entities:
            sequence = entity.get("sequence", "")
            if isinstance(sequence, list):
                sequence = "".join(str(item) for item in sequence)
            values = [
                entity.get("entity_id", ""),
                entity.get("common_name") or entity.get("name", ""),
                entity.get("type") or entity.get("entity_type", ""),
                sequence,
                entity.get("details") or entity.get("description", ""),
            ]
            for table in self._entity_tables():
                row = table.rowCount()
                table.insertRow(row)
                for column, value in enumerate(values):
                    table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))
            count += 1
        self._set_entities_tab_label(count)

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
                row.get("reactive_probe_flag", ""),
                row.get("chromophore_center_atom", ""),
                props.get("abs_max", {}).get("property_value", ""),
                props.get("em_max", {}).get("property_value", ""),
                props.get("qy", {}).get("property_value", ""),
                props.get("ext_coeff", {}).get("property_value", ""),
            ]
            flat_row = dict(row)
            flat_row["abs_max"] = props.get("abs_max", {}).get("property_value")
            flat_row["em_max"] = props.get("em_max", {}).get("property_value")
            flat_row["qy"] = props.get("qy", {}).get("property_value")
            flat_row["ext_coeff"] = props.get("ext_coeff", {}).get("property_value")

            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value or ""))
                if column == 0:
                    item.setData(QtCore.Qt.UserRole, flat_row)
                self.probes_table.setItem(index, column, item)

    def load_probe(self) -> None:
        rows = self.probes_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.probes_table.item(rows[0].row(), 0)
        data = item.data(QtCore.Qt.UserRole) if item else None
        if data:
            self.probe_detail_widget.set_data(data)

    def new_probe(self) -> None:
        """Clear form to create a new probe."""
        self.probe_detail_widget.set_data({})
        widget = self.probe_detail_widget.widgets.get("chromophore_name")
        if widget:
            widget.setFocus()

    def delete_probe(self) -> None:
        """Delete the selected probe."""
        data = self.probe_detail_widget.get_data()
        probe_id = data.get("probe_id")
        if not probe_id:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete probe", f"Delete probe {probe_id}?"
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.client.delete_probe(int(probe_id))
            self.fill_probes()

    def fill_probes_positions(self, probes: list[dict[str, Any]]) -> None:
        """Populate the structured probe-position editor from full description."""
        if not hasattr(self, "sample_probes_table"):
            return
        self.sample_probes_table.setRowCount(0)
        if not probes and self.current_sample_id:
            try:
                probes = self.client.list_probe_positions(sample_id=self.current_sample_id)
            except Exception:
                probes = []
        for probe in probes:
            row = self.sample_probes_table.rowCount()
            self.sample_probes_table.insertRow(row)
            position = probe.get("position") or probe
            properties = probe.get("properties") or {}
            values = [
                probe.get("probe_name") or probe.get("chromophore_name") or probe.get("name", ""),
                position.get("entity_id", ""),
                position.get("seq_id") or position.get("residue_number", ""),
                position.get("comp_id") or position.get("residue_name", ""),
                position.get("asym_id") or position.get("chain_id", ""),
                position.get("atom_id", ""),
                position.get("mutation_flag", "no"),
                position.get("modification_flag", "no"),
                _property_value(properties, "abs_max", "absorption_wavelength"),
                _property_value(properties, "em_max", "emission_wavelength"),
                _property_value(properties, "qy", "quantum_yield"),
                probe.get("sample_probe_id", ""),
            ]
            for column, value in enumerate(values):
                self.sample_probes_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))

    def fill_positions(self, mappings: list[dict[str, Any]] | None = None) -> None:
        if mappings is None or isinstance(mappings, bool):
            mappings = []
        if self.current_sample_id:
            try:
                mappings = self.client.list_probe_positions(sample_id=self.current_sample_id)
            except Exception as exc:
                chisurf.logging.warning("fill_positions(self, mappings: list[dict[str, Any]] | None = None) -> %s", exc)
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
                mapping.get("seq_id") or mapping.get("residue_number", ""),
                mapping.get("atom_id", ""),
                mapping.get("mutation_flag", ""),
                mapping.get("modification_flag", ""),
                mapping.get("auth_name", ""),
                mapping.get("fluorophore_type", ""),
                mapping.get("description", "") or mapping.get("position_description", ""),
            ]
            for column, value in enumerate(values):
                self.positions_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(str(value or ""))
                )

    def fill_fret_pairs(self, pairs: list[dict[str, Any]]) -> None:
        """Populate the structured FRET-pair editor."""
        if not hasattr(self, "fret_pairs_table"):
            return
        if self.current_sample_id:
            try:
                pairs = self.client.list_fret_pairs(self.current_sample_id)
            except Exception as exc:
                chisurf.logging.warning("fill_fret_pairs(self, pairs: list[dict[str, Any]]) -> None: %s", exc)
        self.fret_pairs_table.setRowCount(0)
        for pair in pairs:
            row = self.fret_pairs_table.rowCount()
            self.fret_pairs_table.insertRow(row)
            values = [
                pair.get("donor_probe", ""),
                pair.get("acceptor_probe", ""),
                pair.get("forster_radius_nm") or pair.get("forster_radius", ""),
                pair.get("kappa_squared", ""),
                pair.get("refractive_index") or pair.get("index_of_refraction", ""),
                pair.get("overlap_integral", ""),
                pair.get("forster_radius_id", ""),
            ]
            for column, value in enumerate(values):
                self.fret_pairs_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))

    def refresh_entities_table(self) -> None:
        """Refresh standalone entities from the new PRD-02 entity service."""
        self.fill_entities(self.client.list_entities())

    def save_selected_entity(self) -> None:
        """Save the selected standalone entity row."""
        row = self.entities_table.currentRow()
        if row < 0:
            self.status_label.setText("Select an entity row to save")
            return
        entity = {
            "entity_id": _table_text(self.entities_table, row, 0),
            "name": _table_text(self.entities_table, row, 1),
            "common_name": _table_text(self.entities_table, row, 1),
            "type": _table_text(self.entities_table, row, 2) or "polymer",
            "sequence": _table_text(self.entities_table, row, 3),
            "details": _table_text(self.entities_table, row, 4),
        }
        saved = self.client.save_entity(entity)
        self.status_label.setText(f"Saved entity {saved.get('entity_id', entity['entity_id'])}")
        self.refresh_entities_table()

    def delete_selected_entity(self) -> None:
        """Soft-delete the selected standalone entity row."""
        row = self.entities_table.currentRow()
        if row < 0:
            self.status_label.setText("Select an entity row to delete")
            return
        entity_id = _table_text(self.entities_table, row, 0)
        if not entity_id:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete entity",
            f"Delete entity {entity_id}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self.client.delete_entity(entity_id)
        self.status_label.setText(f"Deleted entity {entity_id}")
        self.refresh_entities_table()

    def add_standalone_probe_row(self) -> None:
        """Append an editable standalone probe row."""
        row = self.probes_table.rowCount()
        self.probes_table.insertRow(row)
        probe_name = COMMON_PROBE_NAMES[0] if COMMON_PROBE_NAMES else ""
        defaults = DEFAULT_FLUOROPHORE_SPECTRA.get(probe_name, {})
        values = [
            "", probe_name, "dye", "extrinsic", "covalent", "no", "",
            defaults.get("absorption_wavelength_nm", ""),
            defaults.get("emission_wavelength_nm", ""),
            defaults.get("quantum_yield", ""),
            defaults.get("extinction_coefficient", ""),
        ]
        for column, value in enumerate(values):
            self.probes_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def save_selected_probe(self) -> None:
        """Save the selected standalone probe row and its optical properties."""
        data = self.probe_detail_widget.get_data()
        probe = {
            "chromophore_name": data.get("chromophore_name"),
            "category": data.get("category") or "dye",
            "probe_origin": data.get("probe_origin") or "extrinsic",
            "probe_link_type": data.get("probe_link_type") or "covalent",
            "reactive_probe_flag": data.get("reactive_probe_flag") or "no",
            "reactive_probe_name": data.get("reactive_probe_name"),
            "fluorophore_type": data.get("fluorophore_type"),
            "description": data.get("description"),
            "is_curated": bool(data.get("is_curated")),
            "quality_flag": int(data.get("quality_flag") or 0) if data.get("quality_flag") is not None else None,
        }
        
        probe["name"] = data.get("chromophore_name")
        probe_id_val = data.get("probe_id")
        if probe_id_val not in (None, "", 0):
            probe["probe_id"] = int(probe_id_val)

        saved = self.client.save_probe(probe)
        probe_id = saved.get("probe_id")
        if probe_id:
            properties = [
                {"property_name": "abs_max", "property_value": _float_or_none(data.get("abs_max")), "unit": "nm"},
                {"property_name": "em_max", "property_value": _float_or_none(data.get("em_max")), "unit": "nm"},
                {"property_name": "qy", "property_value": _float_or_none(data.get("qy")), "unit": ""},
                {"property_name": "ext_coeff", "property_value": _float_or_none(data.get("ext_coeff")), "unit": "M^-1 cm^-1"},
            ]
            self.client.save_probe_optical_properties(
                int(probe_id),
                [prop for prop in properties if prop["property_value"] is not None],
            )
        self.status_label.setText(f"Saved probe {saved.get('chromophore_name', probe.get('chromophore_name', ''))}")
        self.fill_probes()

    def _entity_tables(self) -> list[QtWidgets.QTableWidget]:
        """Return all initialized entity tables that should stay in sync."""
        tables = []
        for name in ("sample_entities_table", "entities_table"):
            table = getattr(self, name, None)
            if table is not None and table not in tables:
                tables.append(table)
        return tables

    def _set_entities_tab_label(self, count: int) -> None:
        """Update the entities nav item with a row count."""
        row = getattr(self, "_row_by_entity", {}).get("entity")
        if row is None:
            return
        item = self.nav_list.item(row)
        if item is not None:
            item.setText(f"{self.panels[row].get('name', 'Entities')} ({count})")

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
        entity_table = getattr(self, "sample_entities_table", self.entities_table)
        for row in range(entity_table.rowCount()):
            entity_id = _table_text(entity_table, row, 0)
            name = _table_text(entity_table, row, 1)
            entity_type = _table_text(entity_table, row, 2) or "polymer"
            sequence = _table_text(entity_table, row, 3)
            details = _table_text(entity_table, row, 4)
            if not any((entity_id, name, sequence, details)):
                continue
            entities.append(
                {
                    "entity_id": entity_id or name,
                    "name": name or entity_id,
                    "common_name": name or entity_id,
                    "type": entity_type,
                    "entity_type": entity_type,
                    "sequence": sequence,
                    "details": details,
                    "description": details,
                }
            )
        probes = []
        if hasattr(self, "sample_probes_table"):
            for row in range(self.sample_probes_table.rowCount()):
                probe_name = _table_text(self.sample_probes_table, row, 0)
                if not probe_name:
                    continue
                entity_id = _table_text(self.sample_probes_table, row, 1)
                entity_index = next(
                    (
                        index
                        for index, entity in enumerate(entities)
                        if entity_id
                        and entity_id in {entity.get("entity_id"), entity.get("name"), entity.get("common_name")}
                    ),
                    0,
                )
                probes.append(
                    {
                        "name": probe_name,
                        "chromophore_name": probe_name,
                        "entity_index": entity_index,
                        "seq_id": _int_or_none(_table_text(self.sample_probes_table, row, 2)),
                        "comp_id": _table_text(self.sample_probes_table, row, 3),
                        "asym_id": _table_text(self.sample_probes_table, row, 4) or "A",
                        "atom_id": _table_text(self.sample_probes_table, row, 5),
                        "mutation_flag": _table_text(self.sample_probes_table, row, 6) or "no",
                        "modification_flag": _table_text(self.sample_probes_table, row, 7) or "no",
                        "absorption_wavelength_nm": _float_or_none(_table_text(self.sample_probes_table, row, 8)),
                        "emission_wavelength_nm": _float_or_none(_table_text(self.sample_probes_table, row, 9)),
                        "quantum_yield": _float_or_none(_table_text(self.sample_probes_table, row, 10)),
                    }
                )
        fret_pairs = []
        if hasattr(self, "fret_pairs_table"):
            probe_names = [probe["name"] for probe in probes]
            for row in range(self.fret_pairs_table.rowCount()):
                donor = _table_text(self.fret_pairs_table, row, 0)
                acceptor = _table_text(self.fret_pairs_table, row, 1)
                if not donor or not acceptor:
                    continue
                if donor not in probe_names or acceptor not in probe_names:
                    continue
                fret_pairs.append(
                    {
                        "donor_probe": donor,
                        "acceptor_probe": acceptor,
                        "probe_1_index": probe_names.index(donor),
                        "probe_2_index": probe_names.index(acceptor),
                        "forster_radius_nm": _float_or_none(_table_text(self.fret_pairs_table, row, 2)),
                        "kappa_squared": _float_or_none(_table_text(self.fret_pairs_table, row, 3)),
                        "refractive_index": _float_or_none(_table_text(self.fret_pairs_table, row, 4)),
                        "overlap_integral": _float_or_none(_table_text(self.fret_pairs_table, row, 5)),
                        "forster_radius_id": _table_text(self.fret_pairs_table, row, 6),
                    }
                )
        mappings = []
        for row in range(self.positions_table.rowCount()):
            mappings.append(
                {
                    "sample_probe_id": _table_text(self.positions_table, row, 0) or None,
                    "sample_id": _table_text(self.positions_table, row, 1) or None,
                    "probe_id": _table_text(self.positions_table, row, 2) or None,
                    "entity_id": _table_text(self.positions_table, row, 4) or None,
                    "asym_id": _table_text(self.positions_table, row, 5) or None,
                    "seq_id": _int_or_none(_table_text(self.positions_table, row, 6)),
                    "fluorophore_type": _table_text(self.positions_table, row, 7) or "unspecified",
                    "description": _table_text(self.positions_table, row, 8),
                }
            )
        key_values = self.metadata_editor.get_data()
        condition_id = (
            self.sample_condition_id_field.text().strip()
            if hasattr(self, "sample_condition_id_field")
            else ""
        )
        ph_value = self.sample_ph_spin.value() if hasattr(self, "sample_ph_spin") else 0
        temperature_value = (
            self.sample_temperature_spin.value()
            if hasattr(self, "sample_temperature_spin")
            else 0
        )
        ionic_value = self.sample_ionic_spin.value() if hasattr(self, "sample_ionic_spin") else 0
        buffer_text = (
            self.sample_buffer_edit.text().strip()
            if hasattr(self, "sample_buffer_edit")
            else ""
        )
        condition_details = (
            self.sample_condition_details_edit.toPlainText().strip()
            if hasattr(self, "sample_condition_details_edit")
            else ""
        )
        return {
            "sample_id": self.sample_id_edit.text().strip(),
            "name": self.sample_id_edit.text().strip(),
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
                "condition_id": condition_id,
                "ph": None if ph_value == 0 else ph_value,
                "temperature": None if temperature_value == 0 else temperature_value,
                "temperature_k": None if temperature_value == 0 else temperature_value,
                "ionic_strength": None if ionic_value == 0 else ionic_value,
                "salt_concentration_m": None if ionic_value == 0 else ionic_value,
                "buffer_composition": buffer_text,
                "details": condition_details,
            },
            "entities": entities,
            "probes": probes,
            "fret_pairs": fret_pairs,
            "sample_probes": mappings,
            "key_values": key_values,
        }

    def save_sample(self) -> None:
        sample = self.collect_sample()
        if not sample["sample_id"]:
            self.status_label.setText("Sample id is required")
            return
        saved = self.client.save_sample(sample)
        if saved is not None:
            self.status_label.setText(f"Saved {saved['sample_id']}")
        else:
            self.status_label.setText("Save failed: no response from server")
        self.refresh()

    def add_metadata_row(self) -> None:
        self.metadata_editor._on_add_empty_row()

    def delete_metadata_row(self) -> None:
        self.metadata_editor._on_delete_row()

    def collect_user(self) -> dict[str, Any]:
        data = self.user_detail_widget.get_data()
        data["requester_id"] = self._active_mfdb_user_id()
        return data

    def new_user(self) -> None:
        """Prepare the user form for a new entry with a generated UUID."""
        self.users_table.clearSelection()
        new_data = {
            "user_uuid": str(uuid.uuid4()),
            "user_id": "",
            "display_name": "",
            "email": "",
            "role": "user",
            "affiliation": "",
            "department": "",
            "phone": "",
            "website": "",
            "address": "",
            "is_admin": 0,
            "allow_passwordless_login": 0,
            "active_branch_uuid": "",
            "details": "",
        }
        self.user_detail_widget.set_data(new_data)
        self.user_detail_widget.widgets["user_id"].setFocus()
        self.status_label.setText("New user — fill fields and Save")

    def save_user(self) -> None:
        users = self.client.save_user(self.collect_user())
        self.fill_user_table(users)
        self.refresh()

    def delete_user(self) -> None:
        user_id = self.user_detail_widget.get_data().get("user_id")
        if not user_id:
            return
        users = self.client.delete_user(user_id)
        self.fill_user_table(users)
        self.refresh()

    def change_user_password(self) -> None:
        """Set or clear the selected user's MFDB password."""
        user_id = self.user_detail_widget.get_data().get("user_id")
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
        self.user_detail_widget.set_data(user)

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
        item = self.branches_table.item(rows[0].row(), 0)
        data = item.data(QtCore.Qt.UserRole) if item else None
        if data:
            self.branch_detail_widget.set_data(data)

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
                    item = QtWidgets.QTableWidgetItem(str(value or ""))
                    if column == 0:
                        item.setData(QtCore.Qt.UserRole, b)
                    self.branches_table.setItem(row, column, item)
        except Exception as exc:
            chisurf.logging.warning("Operation failed: %s", exc)

    def save_branch(self) -> None:
        try:
            data = self.branch_detail_widget.get_data()
            creator_user = self.branch_user_combo.currentData() or "user_default"
            self.client.create_branch(
                branch_uuid=data.get("branch_uuid"),
                name=data.get("name"),
                parent_branch_uuid=data.get("parent_branch_uuid"),
                head_operation_id=data.get("head_operation_id"),
                created_by_user_id=creator_user,
                description=data.get("description"),
            )
            self.refresh()
            QtWidgets.QMessageBox.information(self, "Success", f"Branch '{data.get('name')}' saved successfully.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not save branch: {e}")

    def delete_branch(self) -> None:
        uuid_val = self.branch_detail_widget.get_data().get("branch_uuid")
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
        except Exception as exc:
            chisurf.logging.warning("fill_branch_user_combo(self) -> None: %s", exc)
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
        self.branch_detail_widget.set_data({
            "branch_uuid": "",
            "name": f"{name}-branch",
            "parent_branch_uuid": branch_uuid,
            "head_operation_id": head_operation_id,
            "description": f"Parallel branch from {name}",
        })

    def create_time_branch(self) -> None:
        """Create and activate a branch at the requested operation."""
        user_id = self.branch_user_combo.currentData()
        data = self.branch_detail_widget.get_data()
        operation_id = data.get("head_operation_id")
        branch_name = data.get("name")
        branch_uuid = data.get("branch_uuid")
        parent_uuid = data.get("parent_branch_uuid")
        description = data.get("description")
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
        return self.device_detail_widget.get_data()

    def save_device(self) -> None:
        devices = self.client.save_device(self.collect_device())
        self.fill_device_table(devices)
        self.refresh()

    def delete_device(self) -> None:
        device_id = self.device_detail_widget.get_data().get("device_id")
        if not device_id:
            return
        devices = self.client.delete_device(device_id)
        self.fill_device_table(devices)
        self.refresh()

    def load_device(self) -> None:
        rows = self.devices_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.devices_table.item(rows[0].row(), 0)
        data = item.data(QtCore.Qt.UserRole) if item else None
        if data:
            self.device_detail_widget.set_data(data)

    def save_experiment_type(self) -> None:
        data = self.experiment_type_detail_widget.get_data()
        types = self.client.save_experiment_type(data)
        self.fill_experiment_type_table(types)
        self.refresh()

    def delete_experiment_type(self) -> None:
        data = self.experiment_type_detail_widget.get_data()
        type_id = data.get("type_id")
        if type_id is None:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Delete experiment type", f"Delete experiment type {type_id}?"
        )
        if answer == QtWidgets.QMessageBox.Yes:
            types = self.client.delete_experiment_type(int(type_id))
            self.fill_experiment_type_table(types)
            self.refresh()

    def collect_experiment(self) -> dict[str, Any]:
        data = self.experiment_detail_widget.get_data()
        # Ensure type_id is an integer if present
        if data.get("type_id") is not None:
            try:
                data["type_id"] = int(data["type_id"])
            except (ValueError, TypeError):
                pass
        return data

    def save_experiment(self) -> None:
        experiment = self.collect_experiment()
        if not experiment.get("experiment_id"):
            self.status_label.setText("Experiment id is required")
            return
        saved = self.client.save_experiment(experiment)
        self.status_label.setText(f"Saved {saved['experiment_id']}")
        self.refresh()

    def delete_experiment(self) -> None:
        data = self.experiment_detail_widget.get_data()
        experiment_id = data.get("experiment_id")
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
                cell_item = QtWidgets.QTableWidgetItem(str(value or ""))
                if column == 0:
                    cell_item.setData(QtCore.Qt.UserRole, item)
                self.experiment_types_table.setItem(row, column, cell_item)

    def fill_experiment_table(self, sample_id: str | None = None, type_id: int | None = None) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.experiments_table):
            return
        self.experiments_table.setRowCount(0)
        for item in self.client.list_experiments(sample_id=sample_id, type_id=type_id):
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
                cell_item = QtWidgets.QTableWidgetItem(str(value or ""))
                if column == 0:
                    cell_item.setData(QtCore.Qt.UserRole, item)
                self.experiments_table.setItem(row, column, cell_item)

    def fill_experiment_sample_combo(self) -> None:
        self.experiment_sample_combo.blockSignals(True)
        self.experiment_sample_combo.clear()
        self.experiment_sample_combo.addItem("", "")
        for sample in self.client.list_samples():
            sample_id = sample.get("sample_id", "")
            description = sample.get("description") or sample_id
            self.experiment_sample_combo.addItem(f"{description} ({sample_id})", sample_id)
        self.experiment_sample_combo.blockSignals(False)

    def load_experiment_type(self) -> None:
        rows = self.experiment_types_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.experiment_types_table.item(rows[0].row(), 0)
        data = item.data(QtCore.Qt.UserRole) if item else None
        if data:
            self.experiment_type_detail_widget.set_data(data)

    def load_experiment(self) -> None:
        rows = self.experiments_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.experiments_table.item(rows[0].row(), 0)
        experiment_id = item.text()
        self.current_experiment_id = experiment_id
        experiment = self.client.get_experiment(experiment_id) or {}
        self._loading = True
        try:
            self.experiment_detail_widget.set_data(experiment)
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

    def show_sample_full_description(self) -> None:
        """Show the nested PRD-02 sample description in the preview pane."""
        sample_id = self._active_sample_id()
        if not sample_id:
            self.status_label.setText("Select or enter a sample id")
            return
        description = self.client.get_sample_full_description(sample_id)
        if hasattr(self, "sample_subtabs") and hasattr(self, "full_description_edit"):
            self.sample_subtabs.setCurrentWidget(self.full_description_edit.parentWidget())
            self.full_description_edit.setPlainText(
                json.dumps(description, indent=2, default=str)
            )
        if hasattr(self, "preview_edit"):
            self.preview_edit.setPlainText(json.dumps(description, indent=2, default=str))
        self.status_label.setText(f"Loaded full description for {sample_id}")

    def validate_selected_sample_export(self) -> None:
        """Run FLR CIF export validation for the selected sample."""
        sample_id = self._active_sample_id()
        if not sample_id:
            self.status_label.setText("Select or enter a sample id")
            return
        result = self.client.validate_sample_export(sample_id)
        if hasattr(self, "validation_list"):
            self.validation_list.clear()
            for warning in result.get("warnings", []):
                self.validation_list.addItem(str(warning))
            if not result.get("warnings"):
                self.validation_list.addItem("No export validation warnings.")
        if hasattr(self, "preview_edit"):
            self.preview_edit.setPlainText(json.dumps(result, indent=2, default=str))
        status = "valid" if result.get("valid") else "has warnings"
        self.status_label.setText(f"Export validation for {sample_id}: {status}")

    def _active_sample_id(self) -> str:
        """Return the current sample id from selection or the sample form."""
        return (self.current_sample_id or self.sample_id_edit.text()).strip()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._restore_dock_layout()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._save_dock_layout()

    def closeEvent(self, event) -> None:
        if self._background_tasks:
            self._set_status_message(
                "MFDB task still running. Wait for it to finish before closing."
            )
            event.ignore()
            return
        self._save_dock_layout()
        super().closeEvent(event)

    def _dock_settings(self):
        settings = QtCore.QSettings(
            str(get_plugin_settings_path("mfdb_admin")),
            QtCore.QSettings.IniFormat,
        )
        return settings

    def _save_dock_layout(self) -> None:
        if not hasattr(self, "nav_list"):
            return
        settings = self._dock_settings()

        # Selected nav panel + window geometry
        settings.setValue("nav_row", int(self.nav_list.currentRow()))
        settings.setValue("geometry", self.saveGeometry())

        # Per-entity-dock splitter position (table/form ratio); only built docks.
        for key, dock in getattr(self, "_entity_docks", {}).items():
            sp = getattr(dock, "_splitter", None)
            if sp is not None:
                settings.setValue(f"entity_splitter/{key}", sp.saveState())

    def _restore_dock_layout(self) -> None:
        if not hasattr(self, "nav_list"):
            return
        settings = self._dock_settings()

        # Window geometry
        geo = settings.value("geometry")
        if geo is not None:
            try:
                self.restoreGeometry(geo)
            except Exception as exc:
                chisurf.logging.warning("_restore_dock_layout: geometry: %s", exc)

        # Selected nav panel
        row = settings.value("nav_row")
        if row is not None:
            try:
                self.nav_list.setCurrentRow(int(row))
            except Exception:
                pass

        # Per-entity-dock splitter position
        for key, dock in getattr(self, "_entity_docks", {}).items():
            sp = getattr(dock, "_splitter", None)
            if sp is not None:
                saved = settings.value(f"entity_splitter/{key}")
                if saved is not None:
                    try:
                        sp.restoreState(saved)
                    except Exception:
                        pass

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
        # Pre-export validation (Task 9.2)
        try:
            validation = self.client.validate_sample_export(sample_id)
            if not validation.get("valid", True):
                warnings = validation.get("warnings", [])
                msg = "The following issues were found:\n"
                for w in warnings:
                    msg += f"\n• {w}"
                msg += "\n\nExport anyway?"
                reply = QtWidgets.QMessageBox.question(
                    self, "Export validation warnings", msg,
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                if reply != QtWidgets.QMessageBox.Yes:
                    return
        except Exception as exc:
            chisurf.logging.warning("export_selected_sample(self) -> None: %s", exc)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export FLR CIF", f"{sample_id}.cif", "CIF files (*.cif *.mmcif);;All files (*)"
        )
        if not path:
            return
        result = self.client.export_sample(sample_id, output_path=path)
        self.status_label.setText(f"Exported {result.get('output_path')}")

    def preview_cif(self) -> None:
        """Preview the flrCIF output for the selected sample (Task 9.3)."""
        sample_id = self.sample_id_edit.text().strip()
        if not sample_id:
            self.status_label.setText("Select a sample to preview")
            return
        try:
            result = self.client.export_sample(sample_id)
            text = result.get("text", "")
            if not text:
                text = result.get("output_path", "")
                if text:
                    text = Path(text).read_text(encoding="utf-8")
            self.preview_edit.setPlainText(text)
            self.status_label.setText(f"CIF preview for {sample_id}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Preview failed", str(exc))

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
        self.projects_table = QtWidgets.QTableWidget(0, 5)
        self.projects_table.setHorizontalHeaderLabels(
            ["version id", "name", "project id", "created at", "notes"]
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

        restore_button = self._text_icon_button(
            "📥 Restore project to ChiSurf",
            QtWidgets.QStyle.SP_DialogOpenButton,
            "Restore project state from database",
            self.restore_selected_project,
        )
        delete_button = self._text_icon_button(
            "🗑 Delete archived version",
            QtWidgets.QStyle.SP_TrashIcon,
            "Delete the archived project version",
            self.delete_selected_project,
        )

        return self._create_standard_dock_tab(
            table=self.projects_table,
            detail_widget=self.project_detail_widget,
            save_slot=None,
            delete_slot=None,
            extra_buttons=[restore_button, delete_button],
        )

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
            flat_item = dict(item)
            flat_item["project_id"] = item.get("analysis_id") or item.get("version_id") or item.get("project_id", "")
            flat_item["name"] = item.get("model_name") or item.get("project_name") or ""
            flat_item["description"] = item.get("notes", "")

            values = [
                flat_item["project_id"],
                flat_item["name"],
                item.get("project_id", ""),
                (item.get("created_at") or "")[:19].replace("T", " "),
                flat_item["description"],
            ]
            for column, value in enumerate(values):
                cell_item = QtWidgets.QTableWidgetItem(str(value or ""))
                if column == 0:
                    cell_item.setData(QtCore.Qt.UserRole, flat_item)
                self.projects_table.setItem(row, column, cell_item)

    def load_project_details(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.projects_table):
            return
        selected = self.projects_table.selectedItems()
        if not selected:
            self.project_detail_widget.set_data({})
            return

        row = selected[0].row()
        item = self.projects_table.item(row, 0)
        data = item.data(QtCore.Qt.UserRole) if item else None
        if data:
            self.project_detail_widget.set_data(data)

    def restore_selected_project(self) -> None:
        data = self.project_detail_widget.get_data()
        project_id = data.get("project_id")
        if not project_id:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a project to restore.")
            return

        try:
            import chisurf as cs
            from chisurf.core.actions import dispatch
            result = dispatch("project.restore", {"project_id": project_id})
            if result.get("ok") and hasattr(cs, "cs") and cs.cs is not None:
                cs.cs._current_project_id = result.get("project_id")
                cs.cs._current_project_version_id = result.get("version_id")
                cs.cs._current_project_name = result.get("project_name")
                cs.cs._current_project_visibility = result.get("visibility", "private")
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
        data = self.project_detail_widget.get_data()
        project_id = data.get("project_id")
        if not project_id:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a project version to delete.")
            return

        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete Project Version",
            f"Are you sure you want to delete the archived project version '{project_id}' from the database?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        try:
            self.client.delete_project(project_id)
            QtWidgets.QMessageBox.information(
                self,
                "Project Version Deleted",
                "Project version successfully deleted from the database."
            )
            self.refresh()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Delete Failed",
                f"Failed to delete project version: {exc}"
            )

    def setups_tab(self) -> QtWidgets.QWidget:
        self.setups_table = QtWidgets.QTableWidget(0, 7)
        self.setups_table.setHorizontalHeaderLabels(
            ["setup id", "name", "instrument type", "details", "lasers/detectors", "owner", "public"]
        )
        self.setups_table.horizontalHeader().setStretchLastSection(True)
        self.setups_table.itemSelectionChanged.connect(self.load_setup)
        self._install_table_context_menu(
            self.setups_table,
            item_kind="setup",
            id_col=0,
            delete_one_fn=lambda sid: self.client._call("mfdb.setups.delete", {"setup_id": sid}),
        )

        self.setup_validation_label = QtWidgets.QLabel("")
        self.setup_validation_label.setWordWrap(True)

        validate_setup_button = self._text_icon_button(
            "✅ Validate setup", QtWidgets.QStyle.SP_DialogApplyButton, "Validate setup", self.validate_setup
        )

        return self._create_standard_dock_tab(
            table=self.setups_table,
            detail_widget=self.setup_detail_widget,
            save_slot=self.save_setup,
            delete_slot=self.delete_setup,
            extra_buttons=[validate_setup_button],
            extra_widgets_bottom=[self.setup_validation_label],
        )

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
            owner = item.get("created_by_user_id") or ""
            is_public = item.get("is_public", 0)
            values = [
                item.get("setup_id", ""),
                item.get("name", ""),
                item.get("instrument_type", ""),
                item.get("details", ""),
                ld_str,
                owner,
                "Yes" if is_public else "No",
            ]
            for column, value in enumerate(values):
                cell_item = QtWidgets.QTableWidgetItem(str(value or ""))
                if column == 0:
                    cell_item.setData(QtCore.Qt.UserRole, item)
                self.setups_table.setItem(row, column, cell_item)

    def load_setup(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.setups_table):
            return
        rows = self.setups_table.selectionModel().selectedRows()
        if not rows:
            self.setup_detail_widget.set_data({})
            self.setup_validation_label.clear()
            return
        item = self.setups_table.item(rows[0].row(), 0)
        data = item.data(QtCore.Qt.UserRole) if item else None
        if data:
            self.setup_detail_widget.set_data(data)
        self.setup_validation_label.clear()

    def collect_setup(self) -> dict[str, Any]:
        return self.setup_detail_widget.get_data()

    def save_setup(self) -> None:
        try:
            setup = self.collect_setup()
            self.client._call("mfdb.setups.save", {"setup": setup})
            QtWidgets.QMessageBox.information(self, "Setup Saved", "Setup definition successfully saved.")
            self.fill_setup_table()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Failed", f"Failed to save setup:\n{e}")

    def delete_setup(self) -> None:
        data = self.setup_detail_widget.get_data()
        setup_id = data.get("setup_id")
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
        data = self.setup_detail_widget.get_data()
        setup_id = data.get("setup_id")
        if not setup_id:
            return
        try:
            res = self.client._call("mfdb.setups.validate", {"setup_id": setup_id})
            valid = res.get("valid", False)
            if valid:
                self.setup_validation_label.setText(
                    "<font color='green'><b>Validation OK</b></font>"
                )
            else:
                err_str = res.get("message", "Unknown validation error")
                self.setup_validation_label.setText(
                    f"<font color='red'><b>Validation FAIL</b>:<br>{err_str}</font>"
                )
        except Exception as e:
            self.setup_validation_label.setText(
                f"<font color='red'><b>Error validating setup</b>: {e}</font>"
            )

    def raw_data_tab(self) -> QtWidgets.QWidget:
        self.raw_data_table = QtWidgets.QTableWidget(0, 10)
        self.raw_data_table.setHorizontalHeaderLabels(
            [
                "raw data id", "experiment id", "data type", "storage mode",
                "path/url/folder", "validation", "checksum", "acquired at",
                "sample", "sample QA",
            ]
        )
        self.raw_data_table.horizontalHeader().setStretchLastSection(True)
        self.raw_data_table.itemSelectionChanged.connect(self.load_raw_data)
        self.raw_data_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.raw_data_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

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

        return self._create_standard_dock_tab(
            table=self.raw_data_table,
            detail_widget=self.raw_data_detail_widget,
            save_slot=None,
            delete_slot=None,
            extra_buttons=[btn_open, btn_copy, btn_seed],
        )

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
                item.get("acquired_at", ""),
                item.get("sample_name", ""),
            ]
            for column, value in enumerate(values):
                cell_item = QtWidgets.QTableWidgetItem(str(value or ""))
                if column == 0:
                    cell_item.setData(QtCore.Qt.UserRole, item)
                self.raw_data_table.setItem(row, column, cell_item)
            self.raw_data_table.setItem(row, len(values), _sample_quality_table_item(item))

    def load_raw_data(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.raw_data_table):
            return
        rows = self.raw_data_table.selectionModel().selectedRows()
        if not rows:
            self.raw_data_detail_widget.set_data({})
            return
        raw_id = self.raw_data_table.item(rows[0].row(), 0).text()
        try:
            item = self.client.get_raw_data(raw_id) or {}
        except Exception:
            item = {}
        
        flat_item = dict(item)
        flat_item["location"] = _processed_location(item)
        import json
        flat_item["details"] = json.dumps(item, indent=2)
        
        self.raw_data_detail_widget.set_data(flat_item)

    def _on_raw_open_clicked(self) -> None:
        path = self.raw_data_detail_widget.get_data().get("location")
        if path:
            QtGui.QDesktopServices.openUrl(_qurl_for_location(path))

    def _on_raw_copy_clicked(self) -> None:
        raw_id = self.raw_data_detail_widget.get_data().get("raw_data_id")
        if raw_id is not None:
            QtWidgets.QApplication.clipboard().setText(str(raw_id))

    def _on_raw_seed_clicked(self) -> None:
        raw_id = self.raw_data_detail_widget.get_data().get("raw_data_id")
        if raw_id is not None:
            self._set_provenance_seed("raw_data", str(raw_id))

    def _show_provenance_graph_dock(self) -> None:
        """Select the Provenance Graph nav panel (building it if needed)."""
        row = getattr(self, "_row_by_name", {}).get("Provenance Graph")
        if row is not None:
            self.nav_list.setCurrentRow(row)

    def _set_provenance_seed(self, seed_type: str, seed_id: str) -> None:
        self.current_provenance_seed_type = seed_type
        self.current_provenance_seed_id = seed_id
        self.prov_seed_id_edit.setText(seed_id)
        self.prov_seed_type_combo.setCurrentText(seed_type)
        self._show_provenance_graph_dock()

    def processing_runs_tab(self) -> QtWidgets.QWidget:
        self.processing_runs_table = QtWidgets.QTableWidget(0, 8)
        self.processing_runs_table.setHorizontalHeaderLabels(
            ["processing id", "experiment id", "type", "started at", "status", "raw count", "product count", "operator"]
        )
        self.processing_runs_table.horizontalHeader().setStretchLastSection(True)
        self.processing_runs_table.itemSelectionChanged.connect(self.load_processing_run)
        self.processing_runs_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.processing_runs_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        btn_copy = self._text_icon_button(
            "📋 Copy ID", QtWidgets.QStyle.SP_FileIcon, "Copy processing ID to clipboard", self._on_proc_copy_clicked
        )
        btn_seed = self._text_icon_button(
            "🌱 Use as provenance seed",
            QtWidgets.QStyle.SP_ArrowRight,
            "Load this run into the provenance graph",
            self._on_proc_seed_clicked,
        )

        return self._create_standard_dock_tab(
            table=self.processing_runs_table,
            detail_widget=self.processing_run_detail_widget,
            save_slot=None,
            delete_slot=None,
            extra_buttons=[btn_copy, btn_seed],
        )

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
                cell_item = QtWidgets.QTableWidgetItem(str(value or ""))
                if column == 0:
                    cell_item.setData(QtCore.Qt.UserRole, item)
                self.processing_runs_table.setItem(row, column, cell_item)

    def load_processing_run(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.processing_runs_table):
            return
        rows = self.processing_runs_table.selectionModel().selectedRows()
        if not rows:
            self.processing_run_detail_widget.set_data({})
            return
        proc_id = self.processing_runs_table.item(rows[0].row(), 0).text()
        try:
            item = self.client.get_processing_run(proc_id) or {}
        except Exception:
            item = {}
        
        flat_item = dict(item)
        flat_item["type"] = item.get("processing_type")
        import json
        flat_item["settings"] = json.dumps(item, indent=2)
        
        self.processing_run_detail_widget.set_data(flat_item)

    def _on_proc_copy_clicked(self) -> None:
        proc_id = self.processing_run_detail_widget.get_data().get("processing_id")
        if proc_id is not None:
            QtWidgets.QApplication.clipboard().setText(str(proc_id))

    def _on_proc_seed_clicked(self) -> None:
        proc_id = self.processing_run_detail_widget.get_data().get("processing_id")
        if proc_id is not None:
            self._set_provenance_seed("processing_run", str(proc_id))

    def processed_products_tab(self) -> QtWidgets.QWidget:
        PROD_TYPES = ["all", "bur", "tcspc_decay", "fcs_correlation", "pda_histogram",
                      "irf_curve", "hdf5", "zip", "gmm_summary", "spectra", "fit_results"]
        filter_widget = self._build_filter_bar([
            {"label": "Product type:", "attr": "prod_filter_combo",
             "items": PROD_TYPES, "signal": "currentTextChanged",
             "cb": self.fill_processed_products_table},
        ])

        self.processed_products_table = QtWidgets.QTableWidget(0, 10)
        self.processed_products_table.setHorizontalHeaderLabels(
            [
                "product id", "processing id", "product type", "storage mode",
                "path/url/folder", "validation", "checksum", "row count",
                "sample", "sample QA",
            ]
        )
        self.processed_products_table.horizontalHeader().setStretchLastSection(True)
        self.processed_products_table.itemSelectionChanged.connect(self.load_processed_product)
        self._install_table_context_menu(
            self.processed_products_table,
            item_kind="processed_product",
            id_col=0,
        )

        btn_open = self._text_icon_button(
            "📂 Open", QtWidgets.QStyle.SP_DialogOpenButton, "Open processed product", self._on_prod_open_clicked
        )
        ndx_button = self._text_icon_button(
            "🔬 Open in NDXplorer",
            QtWidgets.QStyle.SP_FileDialogContentsView,
            "Open the selected product in NDXplorer",
            self.open_in_ndxplorer,
        )
        btn_copy = self._text_icon_button(
            "📋 Copy ID", QtWidgets.QStyle.SP_FileIcon, "Copy product ID to clipboard", self._on_prod_copy_clicked
        )
        btn_seed = self._text_icon_button(
            "🌱 Use as provenance seed",
            QtWidgets.QStyle.SP_ArrowRight,
            "Load this product into the provenance graph",
            self._on_prod_seed_clicked,
        )

        return self._create_standard_dock_tab(
            table=self.processed_products_table,
            detail_widget=self.processed_product_detail_widget,
            save_slot=None,
            delete_slot=None,
            extra_widgets_top=[filter_widget],
            extra_buttons=[btn_open, ndx_button, btn_copy, btn_seed],
        )

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
                _processed_row_count(item),
                item.get("sample_name", ""),
            ]
            for column, value in enumerate(values):
                self.processed_products_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))
            self.processed_products_table.setItem(row, len(values), _sample_quality_table_item(item))

    def load_processed_product(self) -> None:
        if self._is_deleted() or self._is_widget_deleted(self.processed_products_table):
            return
        selected = self.processed_products_table.selectedItems()
        if not selected:
            self.processed_product_detail_widget.set_data({})
            return
        row = selected[0].row()
        prod_id = self.processed_products_table.item(row, 0).text()
        try:
            item = self.client.get_processed_data(prod_id)
        except Exception:
            item = {}
        flat_item = dict(item)
        flat_item["location"] = _processed_location(item)
        processing_id = item.get("processing_id", "")
        flat_item["experiment_id"] = _experiment_id_for_processing_id(self.client, processing_id)
        self.processed_product_detail_widget.set_data(flat_item)

    def _on_prod_open_clicked(self) -> None:
        data = self.processed_product_detail_widget.get_data()
        path = (data.get("location") or "").strip()
        if path:
            QtGui.QDesktopServices.openUrl(_qurl_for_location(path))

    def _on_prod_copy_clicked(self) -> None:
        data = self.processed_product_detail_widget.get_data()
        prod_id = data.get("product_id") or data.get("processed_data_id") or ""
        if prod_id:
            QtWidgets.QApplication.clipboard().setText(str(prod_id))

    def _on_prod_seed_clicked(self) -> None:
        data = self.processed_product_detail_widget.get_data()
        prod_id = data.get("product_id") or data.get("processed_data_id") or ""
        if prod_id:
            self._set_provenance_seed("processed_data", str(prod_id))

    def open_in_ndxplorer(self) -> None:
        selected = self.processed_products_table.selectedItems()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a processed product.")
            return
        row = selected[0].row()
        prod_id = self.processed_products_table.item(row, 0).text()
        path_str = self.processed_products_table.item(row, 4).text()
        prod_data = self.processed_product_detail_widget.get_data()
        exp_id = prod_data.get("experiment_id") or ""
        if not path_str:
            QtWidgets.QMessageBox.warning(self, "No Path", "Selected product has no associated file path.")
            return

        from pathlib import Path
        path = Path(path_str)
        if not path.exists():
            QtWidgets.QMessageBox.critical(self, "File Not Found", f"The file or directory does not exist:\n{path_str}")
            return

        import sys
        root = Path(__file__).resolve().parents[5]
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
                except Exception as exc:
                    chisurf.logging.warning("Operation failed: %s", exc)
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
                except Exception as exc:
                    chisurf.logging.warning("Operation failed: %s", exc)

            if not hasattr(self, "_ndxplorer_windows"):
                self._ndxplorer_windows = []
            self._ndxplorer_windows.append(ndx)
            self.statusBar().showMessage(f"Opened {path.name} in NDXplorer")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open in NDXplorer:\n{e}")

    def objects_tab(self) -> QtWidgets.QWidget:
        """Create the object store management tab."""
        filter_widget = QtWidgets.QWidget()
        filter_layout = QtWidgets.QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(QtWidgets.QLabel("Filename filter:"))
        self.object_filter_edit = QtWidgets.QLineEdit()
        self.object_filter_edit.setPlaceholderText("Filter by original filename")
        self.object_filter_edit.returnPressed.connect(self.fill_object_table)
        filter_layout.addWidget(self.object_filter_edit)
        self.object_limit_spin = QtWidgets.QSpinBox()
        self.object_limit_spin.setRange(1, 1000)
        self.object_limit_spin.setValue(100)
        filter_layout.addWidget(QtWidgets.QLabel("Limit:"))
        filter_layout.addWidget(self.object_limit_spin)
        refresh_button = self._text_icon_button(
            "🔄 Refresh",
            QtWidgets.QStyle.SP_BrowserReload,
            "Refresh object list",
            self.fill_object_table,
        )
        filter_layout.addWidget(refresh_button)
        filter_layout.addStretch()

        self.objects_table = QtWidgets.QTableWidget(0, 8)
        self.objects_table.setHorizontalHeaderLabels(
            [
                "object uuid", "md5", "original filename", "size bytes",
                "mime type", "refcount", "created at", "created by",
            ]
        )
        self.objects_table.horizontalHeader().setStretchLastSection(True)
        self.objects_table.itemSelectionChanged.connect(self.load_object)

        btn_delete = self._text_icon_button(
            "🗑 Delete object",
            QtWidgets.QStyle.SP_TrashIcon,
            "Delete the selected object (or decrement refcount)",
            self.delete_selected_object,
        )
        btn_copy = self._text_icon_button(
            "📋 Copy UUID",
            QtWidgets.QStyle.SP_FileIcon,
            "Copy object UUID to clipboard",
            self._on_object_copy_clicked,
        )
        btn_reveal = self._text_icon_button(
            "📂 Reveal",
            QtWidgets.QStyle.SP_DirOpenIcon,
            "Reveal the object in the file manager",
            self._on_object_reveal_clicked,
        )

        return self._create_standard_dock_tab(
            table=self.objects_table,
            detail_widget=self.object_detail_widget,
            save_slot=None,
            delete_slot=None,
            extra_widgets_top=[filter_widget],
            extra_buttons=[btn_delete, btn_copy, btn_reveal],
        )

    def fill_object_table(self) -> None:
        """Populate the object store table."""
        if self._is_deleted() or self._is_widget_deleted(self.objects_table):
            return
        self.objects_table.setRowCount(0)
        try:
            filename = self.object_filter_edit.text().strip() or None
            limit = int(self.object_limit_spin.value())
            objects = self.client.list_objects(filename=filename, limit=limit).get("objects", [])
        except Exception:
            objects = []
        for item in objects:
            row = self.objects_table.rowCount()
            self.objects_table.insertRow(row)
            values = [
                item.get("object_uuid", ""),
                item.get("content_md5", ""),
                item.get("original_filename", ""),
                item.get("size_bytes", ""),
                item.get("mime_type", ""),
                item.get("refcount", ""),
                item.get("created_at", ""),
                item.get("created_by_user_uuid", ""),
            ]
            for column, value in enumerate(values):
                self.objects_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value or "")))

    def load_object(self) -> None:
        """Load selected object details into the form."""
        if self._is_deleted() or self._is_widget_deleted(self.objects_table):
            return
        selected = self.objects_table.selectedItems()
        if not selected:
            self.object_detail_widget.set_data({})
            return
        row = selected[0].row()
        object_uuid = self.objects_table.item(row, 0).text()
        try:
            item = self.client.get_object_info(object_uuid).get("object", {})
        except Exception:
            item = {}
        self.object_detail_widget.set_data(item)

    def delete_selected_object(self) -> None:
        """Delete the selected object or decrement its refcount."""
        selected = self.objects_table.selectedItems()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "No object selected", "Select an object to delete.")
            return
        row = selected[0].row()
        object_uuid = self.objects_table.item(row, 0).text()
        info = self.client.get_object_info(object_uuid).get("object", {})
        filename = info.get("original_filename", object_uuid)
        refcount = info.get("refcount", 0)
        msg = f"Delete object '{filename}'?\n\nRefcount: {refcount}"
        if refcount > 1:
            msg += "\n\nThis will only decrement the refcount; the blob will remain."
        else:
            msg += "\n\nThis will permanently delete the blob from disk."
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete object",
            msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            result = self.client.delete_object(object_uuid)
            self.status_label.setText(f"Object delete result: {result}")
            self.fill_object_table()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Delete failed", str(exc))

    def _on_object_copy_clicked(self) -> None:
        """Copy selected object UUID to clipboard."""
        data = self.object_detail_widget.get_data()
        object_uuid = (data.get("object_uuid") or "").strip()
        if object_uuid:
            QtWidgets.QApplication.clipboard().setText(object_uuid)

    def _on_object_reveal_clicked(self) -> None:
        """Reveal the selected object in the file manager."""
        data = self.object_detail_widget.get_data()
        storage_path = (data.get("storage_path") or "").strip()
        if not storage_path:
            return
        try:
            from mfdb.store.database_resolver import object_store_root
            path = object_store_root() / storage_path
            QtGui.QDesktopServices.openUrl(_qurl_for_location(str(path)))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Reveal failed", str(exc))

    def analyses_tab(self) -> QtWidgets.QWidget:
        self.analyses_table = QtWidgets.QTableWidget(0, 5)
        self.analyses_table.setHorizontalHeaderLabels(
            ["analysis id", "experiment id", "type", "model name", "created at"]
        )
        self.analyses_table.horizontalHeader().setStretchLastSection(True)
        self.analyses_table.itemSelectionChanged.connect(self.load_analysis)

        btn_copy = self._text_icon_button(
            "📋 Copy ID", QtWidgets.QStyle.SP_FileIcon, "Copy analysis ID to clipboard", self._on_analysis_copy_clicked
        )
        btn_seed = self._text_icon_button(
            "🌱 Use as provenance seed",
            QtWidgets.QStyle.SP_ArrowRight,
            "Load this analysis into the provenance graph",
            self._on_analysis_seed_clicked,
        )

        return self._create_standard_dock_tab(
            table=self.analyses_table,
            detail_widget=self.analysis_detail_widget,
            save_slot=None,
            delete_slot=None,
            extra_buttons=[btn_copy, btn_seed],
        )

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
            self.analysis_detail_widget.set_data({})
            return
        row = selected[0].row()
        analysis_id = self.analyses_table.item(row, 0).text()
        try:
            item = self.client.get_analysis_run(analysis_id)
        except Exception:
            item = {}
        flat_item = dict(item)
        flat_item["type"] = item.get("analysis_type")
        import json
        flat_item["settings"] = json.dumps(item, indent=2)
        self.analysis_detail_widget.set_data(flat_item)

    def _on_analysis_copy_clicked(self) -> None:
        data = self.analysis_detail_widget.get_data()
        analysis_id = (data.get("analysis_id") or "").strip()
        if analysis_id:
            QtWidgets.QApplication.clipboard().setText(analysis_id)

    def _on_analysis_seed_clicked(self) -> None:
        data = self.analysis_detail_widget.get_data()
        analysis_id = (data.get("analysis_id") or "").strip()
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
        from mfdb.admin.gui.provenance_graph import (
            mfdb_graph_to_node_editor_graph,
        )
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
