from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

import chisurf.core.fitting
from chisurf.gui.plots import plotbase
from chisurf.gui.widgets.metadata_editor import MetadataEditor

def _configure_fill_table(table: QtWidgets.QTableWidget) -> None:
    """Configure a table to fill the available tab space."""
    table.setSizePolicy(
        QtWidgets.QSizePolicy.Expanding,
        QtWidgets.QSizePolicy.Expanding,
    )
    table.setMinimumSize(0, 0)
    table.setMaximumSize(16777215, 16777215)
    table.setWordWrap(False)
    table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)


class DropTable(QtWidgets.QTableWidget):
    """QTableWidget that accepts file/URL drops as new rows."""

    def __init__(self, columns: int, parent=None):
        super().__init__(0, columns, parent)
        self.setAcceptDrops(True)

    def _format_for_path(self, path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in {".ptu", ".bin", ".t3r", ".t3z", ".t3x"}:
            return "ptu"
        if suffix in {".tttr", ".hdf5", ".h5"}:
            return "tttr"
        if suffix in {".csv", ".txt"}:
            return suffix.lstrip(".")
        return ""

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        mime = event.mimeData()
        added = 0
        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile() or url.toString()
                if path:
                    row = self.rowCount()
                    self.insertRow(row)
                    self.setItem(row, 0, QtWidgets.QTableWidgetItem(path))
                    self.setItem(row, 1, QtWidgets.QTableWidgetItem(self._format_for_path(path)))
                    added += 1
        if added == 0 and mime.hasText():
            text = mime.text().strip()
            if text:
                row = self.rowCount()
                self.insertRow(row)
                self.setItem(row, 0, QtWidgets.QTableWidgetItem(text))
                self.setItem(row, 1, QtWidgets.QTableWidgetItem(self._format_for_path(text)))
                added += 1
        if added:
            event.acceptProposedAction()



class FitInfo(plotbase.Plot):

    name = "Info"

    def __init__(
            self,
            fit: chisurf.core.fitting.fit.FitGroup,
            parent: QtWidgets.QWidget = None,
            **kwargs
    ):
        super().__init__(
            fit,
            parent=parent,
            **kwargs
        )
        self.analysis_id = getattr(fit, "name", None) or str(getattr(fit, "fit_idx", "analysis_1"))
        self.db = self._find_flr_database()
        self._memory_metadata = getattr(fit, "flr_metadata", {})
        self._memory_streams = getattr(fit, "flr_photon_streams", [])

        self.textedit = QtWidgets.QPlainTextEdit()
        self.textedit.setReadOnly(True)
        font = self.textedit.font()
        font.setFamily("Menlo, Courier, monospace")
        font.setPointSize(font.pointSize() - 1)
        self.textedit.setFont(font)
        self.layout.addWidget(self.textedit)

        self.plot_controller = QtWidgets.QTabWidget(self)
        self.plot_controller.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.plot_controller.setMinimumSize(0, 0)
        self.plot_controller.setDocumentMode(True)
        self.plot_controller.setToolTip("FLR metadata editor and flrCIF export")
        self.plot_controller.currentChanged.connect(self._on_tab_changed)
        self._suppress_change = False
        self._build_analysis_tab()
        self._build_metadata_tab()
        self._build_external_tab()
        self._build_export_tab()
        self.layout.addWidget(self.plot_controller, stretch=3)
        self._refresh()
        self._update_cif_preview(full=False)

    def _find_flr_database(self):
        for obj in (self.fit, getattr(self.fit, "model", None)):
            for attr in ("fluorophore_database", "db", "mmcif_db"):
                value = getattr(obj, attr, None)
                if value is not None:
                    return value
        return None

    def _get_metadata(self):
        if self.db is not None:
            return self.db.get_analysis_metadata(self.analysis_id)
        return dict(getattr(self.fit, "flr_metadata", self._memory_metadata))

    def _set_metadata(self, metadata):
        if self.db is not None:
            self.db.set_analysis_metadata(self.analysis_id, metadata)
        else:
            self._memory_metadata = dict(metadata)
            self.fit.flr_metadata = self._memory_metadata

    # ── Analysis tab ───────────────────────────────────────────────

    def _build_analysis_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        self.analysis_id_edit = QtWidgets.QLineEdit()
        self.analysis_id_edit.setReadOnly(True)

        self.method_edit = QtWidgets.QLineEdit()
        self.method_edit.setPlaceholderText("auto-detected or type here")
        # Prefill from model / method name
        model = getattr(self.fit, "model", None)
        if model is not None:
            hint = type(model).__name__
            if hint and hint != "object":
                self.method_edit.setText(hint)
        self.method_edit.textChanged.connect(self._on_changed)

        # Sample id: editable combo with autocomplete from DB
        self.sample_combo = QtWidgets.QComboBox()
        self.sample_combo.setEditable(True)
        self.sample_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.sample_combo.setPlaceholderText("type sample id or select existing")
        self.sample_combo.currentTextChanged.connect(self._update_sample_uuid_display)
        self.sample_combo.currentTextChanged.connect(self._on_changed)

        completer = self.sample_combo.completer()
        if completer is not None:
            completer.setFilterMode(Qt.MatchContains)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._populate_sample_combo()

        # UUID button next to sample combo — small toolbutton
        uuid_row = QtWidgets.QHBoxLayout()
        uuid_row.setSpacing(2)
        gen_uuid_btn = QtWidgets.QToolButton()
        gen_uuid_btn.setText("🔄")
        gen_uuid_btn.setToolTip("Generate new UUID")
        gen_uuid_btn.setFixedSize(24, 24)
        gen_uuid_btn.clicked.connect(self._generate_sample_uuid)
        uuid_row.addWidget(self.sample_combo, stretch=1)
        uuid_row.addWidget(gen_uuid_btn, stretch=0)

        self.sample_uuid_label = QtWidgets.QLabel("")
        self.sample_uuid_label.setStyleSheet("color: gray; font-size: 10px;")

        self.sample_details_edit = QtWidgets.QPlainTextEdit()
        self.sample_details_edit.setPlaceholderText("sample description and free-form details")
        self.sample_details_edit.setMaximumHeight(60)
        self.sample_details_edit.textChanged.connect(self._on_changed)

        self.condition_details_edit = QtWidgets.QPlainTextEdit()
        self.condition_details_edit.setPlaceholderText("pH=7.4; temperature=293.15 K; buffer=...")
        self.condition_details_edit.setMaximumHeight(60)
        self.condition_details_edit.textChanged.connect(self._on_changed)

        layout.addRow("Analysis id", self.analysis_id_edit)
        layout.addRow("Analysis type", self.method_edit)
        layout.addRow("Sample", uuid_row)
        layout.addRow("Sample UUID", self.sample_uuid_label)
        layout.addRow("Sample details", self.sample_details_edit)
        layout.addRow("Condition details", self.condition_details_edit)
        self.plot_controller.addTab(tab, "Analysis")

    def _populate_sample_combo(self):
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        if self.db is not None:
            samples = self.db.list_samples()
            for s in samples:
                sid = s["sample_id"]
                suuid = s.get("sample_uuid") or ""
                display = f"{sid}  [{suuid[:8]}...]" if suuid else sid
                self.sample_combo.addItem(display, (sid, suuid))
        self.sample_combo.blockSignals(False)

    def _generate_sample_uuid(self):
        new_uuid = str(uuid.uuid4())
        self.sample_uuid_label.setText(new_uuid)
        self._on_changed()

    def _on_tab_changed(self, index):
        if self.plot_controller.tabText(index) == "Export":
            self._update_cif_preview(full=False)

    def _update_sample_uuid_display(self):
        sid = self.sample_combo.currentText().strip()
        if not sid:
            self.sample_uuid_label.setText("")
            return
        if self.db is not None:
            row = self.db.get_sample(sid)
            if row and row.get("sample_uuid"):
                self.sample_uuid_label.setText(row["sample_uuid"])
                return
        # New sample: auto-generate UUID
        self.sample_uuid_label.setText(str(uuid.uuid4()))

    def _iter_fit_members(self) -> List[Any]:
        """Return grouped fits or the current fit as a one-item list."""
        grouped_fits = getattr(self.fit, "grouped_fits", None)
        if isinstance(grouped_fits, (list, tuple)) and grouped_fits:
            return list(grouped_fits)
        return [self.fit]

    def _analysis_data_type_for_curve(self, fit: Any, curve_key: str) -> str:
        """Return the small-data type for a named fit curve."""
        if curve_key == "data":
            model_name = str(type(getattr(fit, "model", None)).__name__).lower()
            fit_name = str(type(fit).__name__).lower()
            if "spectrum" in model_name or "spectrum" in fit_name:
                return "spectrum"
            if "fcs" in model_name or "fcs" in fit_name or "correlation" in model_name:
                return "correlation"
            return "decay"
        if curve_key == "model":
            return "model"
        if curve_key == "weighted residuals":
            return "residual"
        if curve_key == "autocorrelation":
            return "correlation"
        return "fit_curve"

    def _analysis_data_name(self, fit: Any, curve_key: str) -> str:
        """Return a stable data name for a named fit curve."""
        fit_name = str(getattr(fit, "name", "") or getattr(fit, "fit_idx", "") or "fit").strip()
        fit_name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in fit_name).strip("._")
        curve_name = str(curve_key).strip()
        curve_name = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in curve_name).strip("._")
        if not fit_name:
            return curve_name or "data"
        if not curve_name:
            return fit_name
        return f"{fit_name}.{curve_name}"

    def _collect_embedded_analysis_data(self) -> List[Dict[str, Any]]:
        """Collect small x/y curves from the current fit for mmCIF embedding."""
        records: List[Dict[str, Any]] = []
        for fit in self._iter_fit_members():
            try:
                curves = fit.get_curves(copy_curves=False, full_length=True)
            except Exception:
                curves = {}
            if not curves:
                data = getattr(fit, "data", None)
                if data is not None:
                    curves = {"data": data}
            for curve_key, curve in getattr(curves, "items", lambda: [])():
                try:
                    x = np.asarray(getattr(curve, "x", []), dtype=np.float64).ravel()
                    y = np.asarray(getattr(curve, "y", []), dtype=np.float64).ravel()
                except Exception:
                    continue
                if x.size == 0 or y.size == 0:
                    continue
                n = min(x.size, y.size)
                x = x[:n]
                y = y[:n]
                finite = np.isfinite(x) & np.isfinite(y)
                if not np.all(finite):
                    x = x[finite]
                    y = y[finite]
                if x.size == 0:
                    continue
                data_type = self._analysis_data_type_for_curve(fit, str(curve_key))
                records.append(
                    {
                        "data_type": data_type,
                        "data_name": self._analysis_data_name(fit, str(curve_key)),
                        "x_values": x,
                        "y_values": y,
                        "x_unit": None,
                        "y_unit": None,
                        "details": "Embedded from FitInfo current fit curves",
                    }
                )
        return records

    def _sync_embedded_analysis_data(self) -> None:
        """Store current fit curves in the FLR database analysis_data table."""
        if self.db is None:
            return
        for record in self._collect_embedded_analysis_data():
            self.db.add_analysis_data(
                self.analysis_id,
                record["data_type"],
                record["x_values"],
                record["y_values"],
                data_name=record["data_name"],
                x_unit=record["x_unit"],
                y_unit=record["y_unit"],
                details=record["details"],
            )

    @staticmethod
    def _array_to_text(values: Any) -> str:
        """Convert numeric values to a CIF-compatible space-separated string."""
        arr = np.asarray(values, dtype=np.float64).ravel()
        return " ".join(f"{float(v):.8g}" for v in arr)

    # ── Metadata tab ──────────────────────────────────────────────

    def _build_metadata_tab(self):
        self.metadata_editor = MetadataEditor(columns=2)
        self.metadata_editor.changed.connect(self._on_changed)
        self.plot_controller.addTab(self.metadata_editor, "Metadata")

    # ── External data tab (simplified: path + format only) ────────

    def _build_external_tab(self):
        tab = QtWidgets.QWidget()
        tab.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        tab.setMinimumSize(0, 0)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)
        self.external_table = DropTable(2)
        self.external_table.setHorizontalHeaderLabels(["file path / URL", "format"])
        self.external_table.horizontalHeader().setStretchLastSection(True)
        self.external_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.external_table.setColumnWidth(1, 90)
        self.external_table.setToolTip("Drag & drop files or URLs here")
        _configure_fill_table(self.external_table)
        self.external_table.itemChanged.connect(self._on_external_table_changed)
        layout.addWidget(self.external_table, stretch=1)

        note = QtWidgets.QLabel(
            "Drag & drop files or URLs to add external data references (PTU/TTTR/CSV).\n"
            "Detector/ channel info goes in the Metadata tab."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.plot_controller.addTab(tab, "External data")

    def _add_photon_stream(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select photon stream")
        if not path:
            return
        stream_id = f"stream_{self.external_table.rowCount() + 1}"
        if self.db is not None:
            self.db.add_photon_stream(self.analysis_id, path, stream_id=stream_id)
        else:
            self._memory_streams.append({"stream_id": stream_id, "file_path": path, "file_format": ""})
            self.fit.flr_photon_streams = self._memory_streams
        self._refresh()

    def _on_external_table_changed(self):
        if not hasattr(self, "_suppress_change") or self._suppress_change:
            return
        if self.db is not None:
            # Clear existing photon streams for this analysis, then re-add from table.
            self.db.conn.execute("DELETE FROM flr_photon_stream WHERE analysis_id = ?", (self.analysis_id,))
            for row in range(self.external_table.rowCount()):
                path_item = self.external_table.item(row, 0)
                format_item = self.external_table.item(row, 1)
                if path_item is None or not path_item.text().strip():
                    continue
                path = path_item.text().strip()
                file_format = format_item.text().strip() if format_item is not None else ""
                self.db.add_photon_stream(self.analysis_id, path, stream_id=f"stream_{row + 1}", file_format=file_format or None)
        else:
            self._memory_streams = []
            for row in range(self.external_table.rowCount()):
                path_item = self.external_table.item(row, 0)
                format_item = self.external_table.item(row, 1)
                if path_item is None or not path_item.text().strip():
                    continue
                path = path_item.text().strip()
                file_format = format_item.text().strip() if format_item is not None else ""
                self._memory_streams.append({"stream_id": f"stream_{row + 1}", "file_path": path, "file_format": file_format})
            self.fit.flr_photon_streams = self._memory_streams
        self._update_cif_preview(full=False)

    def add_external_data(self, path: str, file_format: Optional[str] = None) -> None:
        """Add an external data reference to the external data table."""
        self._suppress_change = True
        row = self.external_table.rowCount()
        self.external_table.insertRow(row)
        self.external_table.setItem(row, 0, QtWidgets.QTableWidgetItem(path))
        self.external_table.setItem(row, 1, QtWidgets.QTableWidgetItem(file_format or ""))
        self._suppress_change = False
        self._on_external_table_changed()

    # ── Export tab: live preview + copy / save tool btns ──────────

    def _build_export_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        toolbar = QtWidgets.QToolBar()
        refresh_btn = QtWidgets.QToolButton()
        refresh_btn.setText("🔄")
        refresh_btn.setToolTip("Compute full mmCIF preview")
        refresh_btn.setFixedSize(24, 24)
        refresh_btn.clicked.connect(lambda: self._update_cif_preview(full=True))
        toolbar.addWidget(refresh_btn)
        copy_btn = QtWidgets.QAction("Copy to clipboard", self)
        copy_btn.triggered.connect(self._copy_cif)
        toolbar.addAction(copy_btn)
        save_btn = QtWidgets.QAction("Save to file...", self)
        save_btn.triggered.connect(self._save_cif)
        toolbar.addAction(save_btn)
        layout.addWidget(toolbar)

        self.cif_preview = QtWidgets.QPlainTextEdit()
        self.cif_preview.setReadOnly(True)
        font = self.cif_preview.font()
        font.setFamily("Menlo, Courier, monospace")
        font.setPointSize(font.pointSize() - 2)
        self.cif_preview.setFont(font)
        layout.addWidget(self.cif_preview, 1)

        self.plot_controller.addTab(tab, "Export")

    def _get_tttr_entries_from_data_curve(self) -> typing.Dict[str, str]:
        """Parse TTTR header JSON from the first TTTR-sourced data curve.

        Returns a flat dict of ``{name: value}`` pairs from the PTU/TTTR
        header tags, or an empty dict if no TTTR data is available.
        """
        if not hasattr(self.fit, 'grouped_fits'):
            return {}
        for grouped in getattr(self.fit, 'grouped_fits', []):
            dc = getattr(grouped, 'data', None)
            if dc is None:
                continue
            meta = getattr(dc, 'meta_data', None) or {}
            hdr = meta.get('tttr_header_json', '')
            if not hdr:
                continue
            try:
                raw = json.loads(hdr) if isinstance(hdr, str) else hdr
            except Exception:
                continue
            tags = raw.get('tags', [])
            result: typing.Dict[str, str] = {}
            for tag in tags:
                name = tag.get('name', '')
                value = tag.get('value', '')
                idx = tag.get('idx', 0)
                if value is None or value == '':
                    continue
                key = name
                if idx and idx > 0:
                    key = f'{name}[{idx}]'
                result[str(key)] = str(value)
            if result:
                return result
        return {}

    def _update_cif_preview(self, full: bool = False):
        tttr_data = self._get_tttr_entries_from_data_curve()
        has_analysis_meta = bool(self._memory_metadata or self._get_metadata())
        if self.db is None and not has_analysis_meta and not tttr_data:
            self.cif_preview.setPlainText("(no data to export)")
            return
        if not full:
            lines = ["mmCIF preview (metadata only)", ""]
            if self.db is not None:
                meta = self._get_metadata()
                if meta:
                    lines.append("--- Analysis metadata ---")
                    lines.extend(f"{k}: {v}" for k, v in sorted(meta.items()))
            else:
                if self._memory_metadata:
                    lines.append("--- Analysis metadata ---")
                    lines.extend(f"{k}: {v}" for k, v in sorted(self._memory_metadata.items()))
            if tttr_data:
                lines.append("")
                lines.append("--- Instrument info (TTTR header) ---")
                lines.extend(f"{k}: {v}" for k, v in sorted(tttr_data.items()))
            if not has_analysis_meta and not tttr_data:
                lines.append("(no metadata)")
            lines.append("")
            lines.append("Click 🔄 to compute full mmCIF with embedded small data.")
            self.cif_preview.setPlainText("\n".join(lines))
            return
        try:
            buf = io.StringIO()
            if self.db is not None:
                self._sync_embedded_analysis_data()
                self.db.export_flr_cif(buf, analysis_id=self.analysis_id)
            else:
                # Minimal in-memory CIF with separate TTTR header section
                import ihm.format
                writer = ihm.format.CifWriter(buf)
                writer.start_block("chisurf_flr_export")
                # General analysis metadata
                if self._memory_metadata:
                    with writer.loop("_chisurf_analysis_metadata", ["analysis_id", "key", "value"]) as loop:
                        for key, value in sorted(self._memory_metadata.items()):
                            loop.write(analysis_id=self.analysis_id or "analysis_1", key=key, value=value)
                # TTTR instrument header in its own category
                if tttr_data:
                    with writer.loop("_chisurf_tttr_header", ["analysis_id", "name", "value"]) as loop:
                        for key, value in sorted(tttr_data.items()):
                            loop.write(analysis_id=self.analysis_id or "analysis_1", name=key, value=value)
                analysis_data = self._collect_embedded_analysis_data()
                if analysis_data:
                    with writer.loop(
                        "_chisurf_analysis_data",
                        ["analysis_id", "data_type", "data_name", "x_values", "y_values", "x_unit", "y_unit", "details"],
                    ) as loop:
                        for data in analysis_data:
                            loop.write(
                                analysis_id=self.analysis_id or "analysis_1",
                                data_type=data.get("data_type"),
                                data_name=data.get("data_name"),
                                x_values=self._array_to_text(data.get("x_values")),
                                y_values=self._array_to_text(data.get("y_values")),
                                x_unit=data.get("x_unit"),
                                y_unit=data.get("y_unit"),
                                details=data.get("details"),
                            )
            text = buf.getvalue()
            if not text.strip():
                self.cif_preview.setPlainText("(empty — no data to export)")
                return
            # Validate with ihm CifTokenReader
            try:
                import ihm.format
                reader = ihm.format.CifTokenReader(io.StringIO(text))
                tokens = list(reader.read_file())
                from ihm.format import CifParserError
                errors = [t for t in tokens if isinstance(t, CifParserError)]
                if errors:
                    valid = False
                    val_msg = "; ".join(str(e) for e in errors)
                else:
                    valid = True
            except Exception as val_err:
                valid = False
                val_msg = str(val_err)
            if valid:
                self.cif_preview.setPlainText(text + "\n# ✅ mmCIF validates OK")
            else:
                self.cif_preview.setPlainText(text + f"\n# ⚠️ Validation: {val_msg}")
        except Exception as exc:
            self.cif_preview.setPlainText(f"(preview failed: {exc})")

    def _copy_cif(self):
        QtWidgets.QApplication.clipboard().setText(self.cif_preview.toPlainText())

    def _save_cif(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save mmCIF", f"{self.analysis_id}.cif",
            "mmCIF files (*.cif *.mmcif);;All files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self.cif_preview.toPlainText())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))

    # ── Refresh ──────────────────────────────────────────────────

    def _refresh(self):
        self._suppress_change = True
        self.analysis_id_edit.setText(self.analysis_id)
        if self.db is not None:
            row = self.db.conn.execute("SELECT * FROM flr_fret_analysis WHERE analysis_id = ?", (self.analysis_id,)).fetchone()
            if row:
                self.method_edit.setText(row["type"] or row["method"] or self.method_edit.text())
                self.sample_combo.setEditText(row["sample_id"] or self.analysis_id)
                self.sample_details_edit.setPlainText(row["details"] or "")
            condition = self.db.conn.execute("SELECT * FROM flr_sample_condition WHERE condition_id = ?", (f"condition_{self.analysis_id}",)).fetchone()
            if condition:
                self.condition_details_edit.setPlainText(condition["details"] or "")
        self._update_sample_uuid_display()
        # Metadata table
        metadata = self._get_metadata()
        self.metadata_editor.set_data([{"key": str(k), "value": str(v)} for k, v in sorted(metadata.items())])
        # External table
        self.external_table.setRowCount(0)
        if self.db is not None:
            for row in self.db.get_photon_streams(self.analysis_id):
                idx = self.external_table.rowCount()
                self.external_table.insertRow(idx)
                self.external_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(str(row.get("file_path") or "")))
                self.external_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(str(row.get("file_format") or "")))
        else:
            for row in getattr(self.fit, "flr_photon_streams", self._memory_streams):
                idx = self.external_table.rowCount()
                self.external_table.insertRow(idx)
                self.external_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(str(row.get("file_path", ""))))
                self.external_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(str(row.get("file_format", ""))))
        self._suppress_change = False

    def _on_changed(self):
        if not hasattr(self, "_suppress_change") or self._suppress_change:
            return
        if self.db is not None:
            sample_id = self.sample_combo.currentText().strip() or None
            # Ensure sample exists in DB
            if sample_id and self.db.get_sample(sample_id) is None:
                suuid = self.sample_uuid_label.text().strip() or None
                self.db.add_sample(sample_id, uuid=suuid)
                self._populate_sample_combo()
            self.db.update_analysis_record(
                self.analysis_id,
                type=self.method_edit.text().strip() or None,
                method=self.method_edit.text().strip() or None,
                sample_id=sample_id,
                details=self.sample_details_edit.toPlainText().strip() or None,
            )
            self.db.conn.execute(
                "INSERT OR REPLACE INTO flr_sample_condition (condition_id, details) VALUES (?, ?)",
                (f"condition_{self.analysis_id}", self.condition_details_edit.toPlainText().strip()),
            )
            self.db.set_analysis_metadata(self.analysis_id, self.metadata_editor.as_dict())
        else:
            self._memory_metadata = self.metadata_editor.as_dict()
            self.fit.flr_metadata = self._memory_metadata
        self.update()

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        fit = self.fit
        meta = self._get_metadata()
        lines = [str(fit), f"\n--- Analysis: {self.analysis_id} ---"]
        if self.db is not None:
            lines.append("DB-backed metadata")
        else:
            lines.append("In-memory metadata")
        lines.append(f"  type: {self.method_edit.text().strip() or '?'}")
        sample_id = self.sample_combo.currentText().strip() or self.analysis_id
        lines.append(f"  sample: {sample_id}")
        suuid = self.sample_uuid_label.text().strip()
        if suuid:
            lines.append(f"  uuid: {suuid}")
        sd = self.sample_details_edit.toPlainText().strip()
        if sd:
            lines.append(f"  details: {sd}")
        cd = self.condition_details_edit.toPlainText().strip()
        if cd:
            lines.append(f"  condition: {cd}")
        if meta:
            lines.append("")
            for k, v in sorted(meta.items()):
                lines.append(f"  {k}: {v}")
        else:
            lines.append("  (no metadata)")
        streams = self.db.get_photon_streams(self.analysis_id) if self.db is not None else getattr(self.fit, "flr_photon_streams", self._memory_streams)
        if streams:
            lines.append(f"\n--- Photon streams ({len(streams)}) ---")
            for s in streams:
                fp = s.get("file_path") or ""
                lines.append(f"  path={fp}")
        self.textedit.setPlainText("\n".join(lines))
        self._refresh()
        self._update_cif_preview(full=False)
