"""Main GUI widget for the Fluorophore Database plugin."""

from __future__ import annotations

from typing import Any

from qtpy import QtCore, QtWidgets

from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient
from .fluorophore_dock import FluorophoreDock


class FluorophoreTool(QtWidgets.QMainWindow):
    """Main window for browsing, curating, and managing fluorophores.

    Features
    --------
    - Browse/search fluorophore probes with auto-generated forms
    - View absorption/emission spectra inline
    - Import reference data from the bundled spectra.db
    - Approve/reject/curate probes with one click
    - Run deterministic AI triage checks
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        client: Any | None = None,
    ):
        super().__init__(parent)
        self._client = client or MFDBClient()

        self.setWindowTitle("Fluorophore Database")
        self.resize(1100, 760)

        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._dock = FluorophoreDock(client=self._client, parent=self)
        self._dock.statusMessage.connect(self._update_status)
        layout.addWidget(self._dock)

        self._status_bar = self.statusBar()
        self._status_label = QtWidgets.QLabel("Ready")
        self._status_bar.addPermanentWidget(self._status_label)

    def _update_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def refresh(self) -> None:
        """Reload the probe table from the backend."""
        if hasattr(self, "_dock"):
            self._dock.refresh()
