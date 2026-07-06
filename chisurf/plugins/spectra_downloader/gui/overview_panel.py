"""Start-page overview of the staging spectra database.

A read-only summary rendered with AutoForm (from ``overview.view.json``) plus a
JSON breakdown by category and source — so the first thing the user sees when
opening the Spectra tool is what has already been scraped.
"""

from __future__ import annotations

import json
from pathlib import Path

from qtpy import QtGui, QtWidgets

from mfdb.admin.gui.optical_components.component_detail_form import (
    ComponentDetailForm,
)

_VIEW = Path(__file__).with_name("overview.view.json")

# Categories grouped under the "Fluorophores" headline count.
_FLUO_CATS = {"fluorophore", "organic_dye", "protein", "quantum_dot", "nanoparticle"}


class OverviewPanel(QtWidgets.QWidget):
    """AutoForm summary + JSON breakdown of the staging database."""

    def __init__(self, db, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("<b>Staging database overview</b>"))
        self._form = ComponentDetailForm(_VIEW)
        layout.addWidget(self._form)

        refresh = QtWidgets.QPushButton("↻ Refresh")
        refresh.clicked.connect(self.refresh)
        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        row.addWidget(refresh)
        layout.addLayout(row)

        layout.addWidget(QtWidgets.QLabel("By category / source (JSON):"))
        self._json = QtWidgets.QPlainTextEdit()
        self._json.setReadOnly(True)
        _mono = QtGui.QFont("Monospace")
        _mono.setStyleHint(QtGui.QFont.Monospace)
        self._json.setFont(_mono)
        layout.addWidget(self._json, 1)

        self.refresh()

    def refresh(self) -> None:
        """Recompute the summary from the staging database."""
        conn = self._db.conn
        by_cat = {
            r[0] or "": r[1]
            for r in conn.execute(
                "SELECT category, COUNT(*) FROM probes WHERE deleted_at IS NULL GROUP BY category"
            )
        }
        by_source = {
            r[0] or "": r[1]
            for r in conn.execute(
                "SELECT source, COUNT(*) FROM probes WHERE deleted_at IS NULL GROUP BY source "
                "ORDER BY COUNT(*) DESC"
            )
        }
        total = sum(by_cat.values())
        with_spectra = conn.execute(
            "SELECT COUNT(DISTINCT probe_id) FROM spectra WHERE deleted_at IS NULL"
        ).fetchone()[0]

        summary = {
            "db_path": str(getattr(self._db, "db_path", "")),
            "total": total,
            "with_spectra": with_spectra,
            "fluorophores": sum(n for c, n in by_cat.items() if c in _FLUO_CATS),
            "filters": by_cat.get("filter", 0),
            "dichroics": by_cat.get("dichroic", 0),
            "detectors": by_cat.get("detector", 0),
            "light_sources": by_cat.get("light_source", 0),
        }
        self._form.set_data(summary)
        self._json.setPlainText(
            json.dumps({"by_category": by_cat, "by_source": by_source}, indent=2)
        )
