"""A basic data browser for the spectra downloader's staging ``spectra.db``.

Lets you inspect what was scraped *before* pushing it into the MFDB: a filterable
table of components on the left; an AutoForm-driven detail panel, the raw optical
properties, and the spectrum plot on the right. It reads the staging database
directly (no MFDB RPC), reusing the shared ``AutoForm`` detail form and
``SpectrumView`` from the mfdb-admin optical-components package.

Drafted from the old ``_dev/fluorophore_db/db_manager_widget.py`` browser, but
modernised onto AutoForm + the canonical view schemes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

import chisurf.plugins.core.mfdb_admin.gui.optical_components as _optical_components
from chisurf.gui.widgets.spectrum_view import SpectrumView
from chisurf.plugins.core.mfdb_admin.gui.optical_components.component_detail_form import (
    ComponentDetailForm,
)

_VIEW_DIR = Path(_optical_components.__file__).parent
_DEFAULT_VIEW = _VIEW_DIR / "fluorophore.view.json"

_COLUMNS = [
    ("ID", "probe_id"),
    ("Name", "chromophore_name"),
    ("Category", "category"),
    ("Source", "source"),
    ("Status", "verification_status"),
]


class SpectraBrowserWidget(QtWidgets.QWidget):
    """Left filter+table / right detail+properties+spectrum browser."""

    def __init__(self, db, parent: QtWidgets.QWidget | None = None,
                 initial_source: str | None = None) -> None:
        super().__init__(parent)
        self._db = db  # an open FluorophoreDatabase on the staging spectra.db
        self._rows: list[dict] = []
        self._initial_source = initial_source
        self._build_ui()
        self.refresh()

    # -- UI ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Filter:"))
        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText("name contains…")
        self._search.textChanged.connect(self._apply_filter)
        bar.addWidget(self._search, 1)
        bar.addWidget(QtWidgets.QLabel("Source:"))
        self._source = QtWidgets.QComboBox()
        self._source.currentIndexChanged.connect(self._apply_filter)
        bar.addWidget(self._source)
        bar.addWidget(QtWidgets.QLabel("Category:"))
        self._category = QtWidgets.QComboBox()
        self._category.currentIndexChanged.connect(self._apply_filter)
        bar.addWidget(self._category)
        self._count = QtWidgets.QLabel("")
        bar.addWidget(self._count)
        layout.addLayout(bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self._table = QtWidgets.QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # Extended selection so "Push selected" can act on several components.
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_select)
        splitter.addWidget(self._table)

        right = QtWidgets.QWidget()
        rlayout = QtWidgets.QVBoxLayout(right)
        rlayout.setContentsMargins(0, 0, 0, 0)
        self._detail = ComponentDetailForm(_DEFAULT_VIEW)
        rlayout.addWidget(self._detail)

        # Properties table + a raw JSON metadata view, in tabs.
        self._tabs = QtWidgets.QTabWidget()
        self._props = QtWidgets.QTableWidget(0, 2)
        self._props.setHorizontalHeaderLabels(["Property", "Value"])
        self._props.horizontalHeader().setStretchLastSection(True)
        self._props.verticalHeader().setVisible(False)
        self._props.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._tabs.addTab(self._props, "Properties")
        self._metadata = QtWidgets.QPlainTextEdit()
        self._metadata.setReadOnly(True)
        _mono = QtGui.QFont("Monospace")
        _mono.setStyleHint(QtGui.QFont.Monospace)
        self._metadata.setFont(_mono)
        self._tabs.addTab(self._metadata, "Metadata (JSON)")
        rlayout.addWidget(self._tabs, 1)

        self._spectrum = SpectrumView()
        rlayout.addWidget(self._spectrum, 2)

        push_bar = QtWidgets.QHBoxLayout()
        push_bar.addStretch()
        self._push_sel_btn = QtWidgets.QPushButton("⬆ Push selected")
        self._push_sel_btn.setToolTip("Push the selected component(s) into the connected MFDB.")
        self._push_sel_btn.clicked.connect(self._push_selected)
        push_bar.addWidget(self._push_sel_btn)
        self._push_all_btn = QtWidgets.QPushButton("⬆ Push all")
        self._push_all_btn.setToolTip("Push every component in this staging DB into the connected MFDB.")
        self._push_all_btn.clicked.connect(self._push_all)
        push_bar.addWidget(self._push_all_btn)
        rlayout.addLayout(push_bar)

        splitter.addWidget(right)
        splitter.setSizes([360, 520])
        layout.addWidget(splitter, 1)

    # -- data ----------------------------------------------------------------
    def refresh(self) -> None:
        """Reload the probe list and category filter from the staging DB."""
        op = (
            "(SELECT property_value FROM optical_properties o "
            "WHERE o.probe_id = p.probe_id AND o.property_name = ? AND o.deleted_at IS NULL LIMIT 1)"
        )
        rows = self._db.conn.execute(
            f"SELECT p.*, {op} AS abs_max, {op} AS em_max, {op} AS qy, {op} AS ext_coeff, "
            f"{op} AS lifetime, t.display_name AS type_name "
            f"FROM probes p LEFT JOIN probe_types t ON t.type_id = p.type_id "
            f"WHERE p.deleted_at IS NULL ORDER BY p.chromophore_name",
            ["abs_max", "em_max", "qy", "ext_coeff", "lifetime"],
        ).fetchall()
        self._rows = [dict(r) for r in rows]

        cats = sorted({(r.get("category") or "") for r in self._rows if r.get("category")})
        current = self._category.currentText()
        self._category.blockSignals(True)
        self._category.clear()
        self._category.addItem("All")
        self._category.addItems(cats)
        idx = self._category.findText(current)
        if idx >= 0:
            self._category.setCurrentIndex(idx)
        self._category.blockSignals(False)

        # Source filter — split comma-joined provenance into individual tokens.
        sources = set()
        for r in self._rows:
            for tok in str(r.get("source") or "").split(","):
                if tok.strip():
                    sources.add(tok.strip())
        current_src = self._initial_source or self._source.currentText()
        self._initial_source = None
        self._source.blockSignals(True)
        self._source.clear()
        self._source.addItem("All")
        self._source.addItems(sorted(sources))
        sidx = self._source.findText(current_src)
        if sidx >= 0:
            self._source.setCurrentIndex(sidx)
        self._source.blockSignals(False)

        self._apply_filter()

    def _filtered(self) -> list[dict]:
        text = self._search.text().strip().lower()
        cat = self._category.currentText()
        src = self._source.currentText()
        out = []
        for r in self._rows:
            if cat and cat != "All" and (r.get("category") or "") != cat:
                continue
            if src and src != "All":
                tokens = {t.strip() for t in str(r.get("source") or "").split(",")}
                if src not in tokens:
                    continue
            if text and text not in str(r.get("chromophore_name") or "").lower():
                continue
            out.append(r)
        return out

    def _apply_filter(self) -> None:
        rows = self._filtered()
        self._table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, (_label, key) in enumerate(_COLUMNS):
                item = QtWidgets.QTableWidgetItem(str(r.get(key) if r.get(key) is not None else ""))
                item.setData(QtCore.Qt.UserRole, r.get("probe_id"))
                self._table.setItem(i, j, item)
        self._table.resizeColumnsToContents()
        self._count.setText(f"{len(rows)} / {len(self._rows)}")

    # -- selection -----------------------------------------------------------
    def _on_select(self) -> None:
        items = self._table.selectedItems()
        if not items:
            return
        probe_id = items[0].data(QtCore.Qt.UserRole)
        data = self._load_probe(int(probe_id))
        # detail form: probe fields + surfaced numeric props
        form_data = dict(data["probe"])
        for p in data["optical_properties"]:
            form_data.setdefault(p["property_name"], p["property_value"])
        self._detail.set_data(form_data)
        # raw properties table
        props = data["optical_properties"]
        self._props.setRowCount(len(props))
        for i, p in enumerate(props):
            self._props.setItem(i, 0, QtWidgets.QTableWidgetItem(str(p["property_name"])))
            self._props.setItem(i, 1, QtWidgets.QTableWidgetItem(str(p["property_value"])))
        # JSON metadata blob (probe fields + properties + spectrum summary)
        meta = {
            "probe": data["probe"],
            "optical_properties": {
                p["property_name"]: p["property_value"] for p in props
            },
            "spectra": [
                {"type": s["spectrum_type"], "points": len(s["wavelengths"])}
                for s in data["spectra"]
            ],
        }
        self._metadata.setPlainText(json.dumps(meta, indent=2, default=str))
        # spectrum
        self._spectrum.display(data)

    # -- push to MFDB --------------------------------------------------------
    def _selected_probe_ids(self) -> list[int]:
        ids = []
        for idx in self._table.selectionModel().selectedRows():
            item = self._table.item(idx.row(), 0)
            if item is not None:
                ids.append(int(item.data(QtCore.Qt.UserRole)))
        return ids

    def _push_selected(self) -> None:
        ids = self._selected_probe_ids()
        if not ids:
            QtWidgets.QMessageBox.information(self, "Push selected", "No components selected.")
            return
        self._push(ids, f"{len(ids)} selected component(s)")

    def _push_all(self) -> None:
        self._push(None, f"all {len(self._rows)} component(s)")

    def _push(self, probe_ids, label: str) -> None:
        from chisurf.plugins.spectra_downloader.download.merge import push_staging_to_mfdb

        if QtWidgets.QMessageBox.question(
            self, "Push to MFDB",
            f"Push {label} from this staging database into the connected MFDB?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No,
        ) != QtWidgets.QMessageBox.Yes:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            summary = push_staging_to_mfdb(str(self._db.db_path), probe_ids=probe_ids)
        except Exception as e:  # pragma: no cover - surfaced to the user
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Push failed", str(e))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.information(
            self, "Push complete",
            f"Pushed {summary.get('merged', 0)} component(s) into the MFDB.\n"
            f"Consolidated: {summary.get('consolidated')}",
        )

    def _load_probe(self, probe_id: int) -> dict:
        probe = self._db.conn.execute(
            "SELECT * FROM probes WHERE probe_id = ? AND deleted_at IS NULL", (probe_id,)
        ).fetchone()
        props = self._db.conn.execute(
            "SELECT property_name, property_value FROM optical_properties "
            "WHERE probe_id = ? AND deleted_at IS NULL ORDER BY property_name",
            (probe_id,),
        ).fetchall()
        spectra = self._db.conn.execute(
            "SELECT * FROM spectra WHERE probe_id = ? AND deleted_at IS NULL", (probe_id,)
        ).fetchall()
        return {
            "probe": dict(probe) if probe else {},
            "optical_properties": [dict(p) for p in props],
            "spectra": [
                {
                    "spectrum_type": s["spectrum_type"],
                    "wavelengths": list(np.frombuffer(s["wavelengths"], dtype=np.float64)),
                    "intensity": list(np.frombuffer(s["intensity_values"], dtype=np.float64)),
                }
                for s in spectra
            ],
        }


class SpectraBrowserDialog(QtWidgets.QDialog):
    """Standalone dialog wrapper around :class:`SpectraBrowserWidget`."""

    def __init__(self, db, parent: QtWidgets.QWidget | None = None,
                 initial_source: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spectra Browser — staging database")
        self.resize(960, 600)
        layout = QtWidgets.QVBoxLayout(self)
        self.browser = SpectraBrowserWidget(db, self, initial_source=initial_source)
        layout.addWidget(self.browser)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def main() -> None:
    """Launch the browser on the bundled staging spectra.db."""
    import sys

    from chisurf.plugins._dev.fluorophore_db.mfdb_adapter import (
        DEFAULT_DATABASE_PATH,
        FluorophoreDatabase,
    )

    db_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DATABASE_PATH)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    db = FluorophoreDatabase(db_path)
    db.connect()
    dlg = SpectraBrowserDialog(db)
    dlg.show()
    app.exec_()


if __name__ == "__main__":
    main()
