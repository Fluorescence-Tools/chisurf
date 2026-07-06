"""Dedicated "Add to MFDB" panel — session-aware, endpoint + auth.

Adding scraped components to the MFDB is its own step, configured by an AutoForm
(``endpoint_auth.view.json``). It reuses the **current user session**: if the
session user is an MFDB administrator, the components are added with no login
prompt. The connection + authentication fields live in a collapsed "Advanced"
panel and are only needed when the session is not already authorized.

- **Local file** — write directly into a local MFDB SQLite file (the resolved
  live MFDB by default). The active user must be an administrator (checked
  against the file's user table); bootstrap DBs with no admin yet are allowed.
- **Server (ZMQ)** — reuse/establish a session against the server; an admin
  (token or passwordless) adds over RPC, otherwise the Advanced password is used.
"""

from __future__ import annotations

from pathlib import Path

from qtpy import QtGui, QtWidgets

from chisurf.core.dataspec import load_view_spec
from chisurf.gui.autoform import AutoForm
from mfdb.admin.gui.session import (
    active_user_id,
    client_is_admin,
    local_admin_status,
)

_VIEW = Path(__file__).with_name("endpoint_auth.view.json")


class _EndpointAuthModel:
    """Editable bag bound to the AutoForm endpoint/auth scheme."""

    def __init__(self) -> None:
        from chisurf.core.mfdb.database_resolver import resolve_database_path

        self.mode = "local"
        try:
            self.db_path = str(resolve_database_path())
        except Exception:
            self.db_path = ""
        self.host = "127.0.0.1"
        self.cmd_port = 8765
        self.pub_port = 8766
        self.user = active_user_id()
        self.password = ""
        self.replace = False
        self.mark_verified = False

    def view_spec(self):
        return load_view_spec(_VIEW)


class AddToMfdbPanel(QtWidgets.QWidget):
    """Add the staging components to the MFDB, reusing the current user session."""

    def __init__(self, db, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db  # staging FluorophoreDatabase
        self._model = _EndpointAuthModel()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Add staging components to the MFDB</b>"))
        self._session_label = QtWidgets.QLabel("")
        self._session_label.setWordWrap(True)
        layout.addWidget(self._session_label)

        self._form = AutoForm(self._model, parent=self)
        layout.addWidget(self._form)

        row = QtWidgets.QHBoxLayout()
        self._check_btn = QtWidgets.QPushButton("↻ Check session")
        self._check_btn.setToolTip("Re-check whether the current session may add to the MFDB.")
        self._check_btn.clicked.connect(self._refresh_session)
        row.addWidget(self._check_btn)
        row.addStretch()
        self._add_btn = QtWidgets.QPushButton("⬆ Add all to MFDB")
        self._add_btn.clicked.connect(self._add_all)
        row.addWidget(self._add_btn)
        layout.addLayout(row)

        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        _mono = QtGui.QFont("Monospace")
        _mono.setStyleHint(QtGui.QFont.Monospace)
        self._log.setFont(_mono)
        layout.addWidget(self._log, 1)

        self._refresh_session()

    # -- helpers -------------------------------------------------------------
    def _echo(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def _user(self) -> str:
        return self._model.user or active_user_id()

    def _authorized(self) -> tuple[bool, str]:
        """Return ``(is_admin, note)`` for the configured endpoint + session."""
        user = self._user()
        if self._model.mode == "server":
            try:
                client = self._server_client()
                ok = client_is_admin(client, user)
                return ok, (f"server {self._model.host}:{self._model.cmd_port}")
            except Exception as e:
                return False, f"not authenticated ({e})"
        target = self._model.db_path or self._resolved()
        is_admin, any_admin = local_admin_status(target, user)
        note = "bootstrap (no admin yet)" if not any_admin else f"local {Path(target).name}"
        return is_admin, note

    @staticmethod
    def _resolved() -> str:
        from chisurf.core.mfdb.database_resolver import resolve_database_path

        return str(resolve_database_path())

    def _server_client(self):
        """Authenticated MFDB client for the configured server (session-first)."""
        from mfdb.admin.gui.client import MFDBClient
        from mfdb.admin.gui.session import cache_session, cached_token

        m = self._model
        host, cmd, pub = m.host, int(m.cmd_port), int(m.pub_port)
        client = MFDBClient(host=host, cmd_port=cmd, pub_port=pub)

        # SSO: reuse a session token cached earlier this ChiSurf session (e.g.
        # from a previous mfdb-admin login) — no password needed.
        token = cached_token(host, cmd, pub)
        if token:
            client.token = token
            return client

        # Otherwise try a passwordless login; fall back to the Advanced password.
        try:
            result = client.login(self._user(), m.password or "")
        except Exception:
            result = client.login(self._user(), m.password) if m.password else {}
        if isinstance(result, dict) and result.get("ok"):
            cache_session(self._user(), client.token, host, cmd, pub)
        return client

    # -- actions -------------------------------------------------------------
    def _refresh_session(self) -> None:
        user = self._user()
        ok, note = self._authorized()
        if ok:
            self._session_label.setText(
                f"✅ Session user <b>{user}</b> is an administrator ({note}) — "
                f"no login needed."
            )
            self._add_btn.setEnabled(True)
        else:
            self._session_label.setText(
                f"🔒 Session user <b>{user}</b> may not add to the MFDB ({note}). "
                f"Set credentials under <i>Advanced</i> or use an admin account."
            )
            self._add_btn.setEnabled(True)  # still allow trying with Advanced creds

    def _add_all(self) -> None:
        m = self._model
        staging = str(self._db.db_path)
        user = self._user()
        self._add_btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor())
        try:
            if m.mode == "server":
                client = self._server_client()
                if not client_is_admin(client, user):
                    self._echo(f"User '{user}' is not an MFDB administrator — cannot add.")
                    return
                self._echo(f"Adding to server {m.host}:{m.cmd_port} as '{user}' "
                           f"(replace={m.replace}) …")
                res = client._call("fluorophores.import_reference_set", {
                    "source_path": staging,
                    "replace": bool(m.replace),
                    "mark_verified": bool(m.mark_verified),
                })
                self._echo(f"Done: {res}")
            else:
                from chisurf.core.mfdb.repository import MFDatabase

                target = m.db_path or self._resolved()
                is_admin, _any = local_admin_status(target, user)
                if not is_admin:
                    self._echo(f"User '{user}' is not an administrator of "
                               f"{target} — cannot add.")
                    return
                if m.replace:
                    self._backup(target)
                self._echo(f"Adding to local MFDB {target} as '{user}' "
                           f"(replace={m.replace}) …")
                with MFDatabase(target) as db:
                    counts = db.import_reference_set(
                        source_path=staging,
                        replace=bool(m.replace),
                        mark_verified=bool(m.mark_verified),
                    )
                self._echo(
                    f"Done: probes={counts['probes']} spectra={counts['spectra']} "
                    f"props={counts['optical_properties']} "
                    f"consolidated={counts.get('consolidated')}"
                    + (f" purged={counts['purged']}" if counts.get("purged") else "")
                )
        except Exception as e:
            self._echo(f"Add failed: {e}")
        finally:
            self._add_btn.setEnabled(True)
            QtWidgets.QApplication.restoreOverrideCursor()

    @staticmethod
    def _backup(target: str) -> None:
        import shutil

        if Path(target).exists():
            shutil.copy2(target, f"{target}.bak")
