"""Dedicated "Add to MFDB" panel — endpoint selection + authentication.

Integrating scraped components into the MFDB is its own step here, configured by
an AutoForm built from ``endpoint_auth.view.json``:

- **Local file** — write directly into a local MFDB SQLite file (the resolved
  live MFDB by default, or a custom path). No authentication is required for a
  local file.
- **Server (ZMQ)** — authenticate against a running MFDB server (host/ports +
  user/password → token) and import over the RPC transport, mirroring how
  mfdb-admin connects.
"""

from __future__ import annotations

from pathlib import Path

from chisurf.core.dataspec import load_view_spec
from chisurf.gui.autoform import AutoForm
from qtpy import QtGui, QtWidgets

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
        self.user = ""
        self.password = ""
        self.replace = False
        self.mark_verified = False

    def view_spec(self):
        return load_view_spec(_VIEW)


class AddToMfdbPanel(QtWidgets.QWidget):
    """Configure an endpoint + auth and add the staging components to the MFDB."""

    def __init__(self, db, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db  # staging FluorophoreDatabase
        self._model = _EndpointAuthModel()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Add staging components to the MFDB</b>"))
        self._form = AutoForm(self._model, parent=self)
        layout.addWidget(self._form)

        row = QtWidgets.QHBoxLayout()
        self._auth_btn = QtWidgets.QPushButton("🔑 Test / authenticate")
        self._auth_btn.setToolTip("For a server endpoint, log in and verify the connection.")
        self._auth_btn.clicked.connect(self._authenticate)
        row.addWidget(self._auth_btn)
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

    # -- helpers -------------------------------------------------------------
    def _echo(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def _server_client(self):
        """Create + authenticate an MFDB client for the configured server."""
        from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

        m = self._model
        client = MFDBClient(host=m.host, cmd_port=int(m.cmd_port), pub_port=int(m.pub_port))
        client.login(m.user, m.password or "")
        return client

    # -- actions -------------------------------------------------------------
    def _authenticate(self) -> None:
        m = self._model
        if m.mode != "server":
            self._echo("Local file endpoint — no authentication required.")
            return
        try:
            client = self._server_client()
            status = client.status()
            self._echo(f"Authenticated to {m.host}:{m.cmd_port} as '{m.user}'. "
                       f"Server status: {status}")
        except Exception as e:
            self._echo(f"Authentication failed: {e}")

    def _add_all(self) -> None:
        m = self._model
        staging = str(self._db.db_path)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor())
        self._add_btn.setEnabled(False)
        try:
            if m.mode == "server":
                self._echo(f"Adding to server {m.host}:{m.cmd_port} (replace={m.replace}) …")
                client = self._server_client()
                res = client._call("fluorophores.import_reference_set", {
                    "source_path": staging,
                    "replace": bool(m.replace),
                    "mark_verified": bool(m.mark_verified),
                })
                self._echo(f"Done: {res}")
            else:
                from chisurf.core.mfdb.database_resolver import resolve_database_path
                from chisurf.core.mfdb.repository import MFDatabase

                target = m.db_path or str(resolve_database_path())
                if m.replace:
                    self._backup(target)
                self._echo(f"Adding to local MFDB {target} (replace={m.replace}) …")
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
