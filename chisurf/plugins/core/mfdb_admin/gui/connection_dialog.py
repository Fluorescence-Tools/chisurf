"""AutoForm-driven MFDB login dialog.

The common case — type a password and hit Enter — is a single line. The less
common bits (user override, server host/ports) live in a collapsed "Advanced"
panel, so reconnecting to a different MFDB server is possible without cluttering
the everyday login.
"""

from __future__ import annotations

from pathlib import Path

from qtpy import QtWidgets

from chisurf.core.dataspec import load_view_spec
from chisurf.gui.autoform import AutoForm

_VIEW = Path(__file__).with_name("connection_auth.view.json")


class _LoginModel:
    """Editable bag bound to the AutoForm connection/auth scheme."""

    def __init__(self, user: str, host: str, cmd_port: int, pub_port: int) -> None:
        self.password = ""
        self.user = user
        self.host = host
        self.cmd_port = int(cmd_port)
        self.pub_port = int(pub_port)

    def view_spec(self):
        return load_view_spec(_VIEW)


class ConnectionAuthDialog(QtWidgets.QDialog):
    """Single-line password login with a collapsible advanced connection panel."""

    def __init__(
        self,
        user: str,
        host: str = "127.0.0.1",
        cmd_port: int = 8765,
        pub_port: int = 8766,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"MFDB login — {user}")
        self._model = _LoginModel(user, host, cmd_port, pub_port)

        layout = QtWidgets.QVBoxLayout(self)
        self._form = AutoForm(self._model, parent=self)
        layout.addWidget(self._form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        """Return the entered ``{user, password, host, cmd_port, pub_port}``."""
        m = self._model
        return {
            "user": m.user,
            "password": m.password,
            "host": m.host,
            "cmd_port": int(m.cmd_port),
            "pub_port": int(m.pub_port),
        }
