"""
Trace Browser Plugin

A simple two-page plugin to browse intensity traces from PTU/TTTR files in a folder.
Page 0: Setup definition using DetectorWizardPage (TTTR channel setup).
Page 1: Trace browser with folder selection, file list, star quality rating (0–3),
        annotation text, filter by rating, preview plot, and export selected files.

Metadata (ratings and annotations) are stored in a JSON file in the same folder
as the traces: .trace_browser_meta.json
"""
import os
import json
import shutil
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QListWidget, QListWidgetItem, QSplitter, QTextEdit, QComboBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QEvent

# Reuse existing widgets/utilities
from chisurf.plugins.intensity_trace.__init__ import IntensityPlotWidget, IntensityTrace
from chisurf.gui.widgets.wizard.tttr_channel_definition import DetectorWizardPage

try:
    import tttrlib
except Exception:
    tttrlib = None

# Optional docx dependency (python-docx)
try:
    from docx import Document
    from docx.shared import Inches
except Exception:
    Document = None
    Inches = None

name = "TTTR:Trace Browser"

META_FILENAME = ".trace_browser_meta.json"
# Determine supported extensions strictly via tttrlib.get_supported_filetypes()
# as requested. No other probing or fallbacks.

def get_tttr_supported_exts() -> List[str]:
    exts: List[str] = []
    try:
        if tttrlib is not None and hasattr(tttrlib, "get_supported_filetypes"):
            exts = list(tttrlib.get_supported_filetypes())
    except Exception:
        exts = []

    # Normalize to dotted lowercase extensions
    norm: List[str] = []
    for e in exts:
        s = str(e).strip().lower()
        if not s:
            continue
        if not s.startswith('.'):
            s = '.' + s
        if s not in norm:
            norm.append(s)
    return norm


def _meta_path(folder: pathlib.Path) -> pathlib.Path:
    return folder / META_FILENAME


def _load_meta(folder: pathlib.Path) -> Dict[str, Dict]:
    p = _meta_path(folder)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_meta(folder: pathlib.Path, data: Dict[str, Dict]):
    p = _meta_path(folder)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if p.exists():
        p.unlink()
    tmp.replace(p)


class StarCombo(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Ratings 0..3
        self.addItem("☆☆☆", 0)
        self.addItem("★☆☆", 1)
        self.addItem("★★☆", 2)
        self.addItem("★★★", 3)

    def set_rating(self, r: int):
        idx = max(0, min(3, int(r)))
        self.setCurrentIndex(idx)

    def rating(self) -> int:
        return int(self.currentData())

    def keyPressEvent(self, event):
        # Make Tab/Shift+Tab jump directly between rating comboboxes (same column, next/prev row)
        try:
            key = event.key()
            if key in (Qt.Key_Tab, Qt.Key_Backtab):
                table = self.parent()
                # Ensure parent is a QTableWidget
                if isinstance(table, QTableWidget):
                    # Find my row by matching this widget in column 1
                    my_row = -1
                    col = 1
                    for r in range(table.rowCount()):
                        if table.cellWidget(r, col) is self:
                            my_row = r
                            break
                    if my_row != -1:
                        forward = (key == Qt.Key_Tab) and not (event.modifiers() & Qt.ShiftModifier)
                        step = 1 if forward else -1
                        next_row = my_row + step
                        # If next row is within table, focus its StarCombo; else fall through to default
                        if 0 <= next_row < table.rowCount():
                            nxt = table.cellWidget(next_row, col)
                            if isinstance(nxt, StarCombo):
                                # Select the row for visual feedback
                                table.selectRow(next_row)
                                # Move focus to next combo
                                nxt.setFocus(Qt.TabFocusReason)
                                event.accept()
                                return
        except Exception:
            # Fall back to default behavior on any error
            pass
        # Default behavior for other keys or edge cases
        super().keyPressEvent(event)


class TraceBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trace Browser")

        # State
        self.current_folder: Optional[pathlib.Path] = None
        self.meta: Dict[str, Dict] = {}
        self.setup_settings: Optional[Dict] = None
        self.selected_channels: Optional[List[int]] = None

        # Two-page layout using a simple stacked layout approach
        self.root_layout = QVBoxLayout(self)

        # Page 0: Detector setup
        self.page0 = QWidget(self)
        p0_layout = QVBoxLayout(self.page0)
        p0_layout.addWidget(QLabel("Setup definition (DetectorWizard)", self.page0))
        self.detector_page = DetectorWizardPage(show_help=False, show_setups_file=True,
                                                show_setup_selection=True, show_tttr_reading=True,
                                                show_tables=True, show_add_inputs=True)
        p0_layout.addWidget(self.detector_page)
        self.btn_continue = QPushButton("Use setup and continue →", self.page0)
        self.btn_continue.clicked.connect(self._on_continue)
        p0_layout.addWidget(self.btn_continue)

        # Page 1: Trace browser
        self.page1 = QWidget(self)
        p1_layout = QVBoxLayout(self.page1)

        # Controls row
        ctrl_row = QHBoxLayout()
        self.folder_label = QLabel("No folder selected", self.page1)
        self.btn_pick_folder = QPushButton("Pick folder", self.page1)
        self.btn_pick_folder.clicked.connect(self._on_pick_folder)

        self.filter_combo = QComboBox(self.page1)
        self.filter_combo.addItems([
            "All",
            "≥ 1★",
            "≥ 2★★",
            "≥ 3★★★",
            "Only 0★"
        ])
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)

        self.sort_combo = QComboBox(self.page1)
        self.sort_combo.addItems(["Sort: Name", "Sort: Rating desc"])
        self.sort_combo.currentIndexChanged.connect(self._refresh_list)

        ctrl_row.addWidget(self.folder_label)
        ctrl_row.addWidget(self.btn_pick_folder)
        ctrl_row.addWidget(QLabel("Filter:"))
        ctrl_row.addWidget(self.filter_combo)
        ctrl_row.addWidget(self.sort_combo)

        self.window_ms_spin = QSpinBox(self.page1)
        self.window_ms_spin.setRange(1, 10000)
        self.window_ms_spin.setValue(10)
        self.window_ms_spin.setSuffix(" ms bin")
        self.window_ms_spin.valueChanged.connect(self._on_window_changed)
        ctrl_row.addWidget(self.window_ms_spin)

        self.btn_export = QPushButton("Export selected…", self.page1)
        self.btn_export.clicked.connect(self._on_export)
        ctrl_row.addWidget(self.btn_export)

        self.btn_export_docx = QPushButton("Export DOCX…", self.page1)
        self.btn_export_docx.clicked.connect(self._on_export_docx)
        ctrl_row.addWidget(self.btn_export_docx)

        p1_layout.addLayout(ctrl_row)

        # Splitter: left list, right details
        splitter = QSplitter(self.page1)
        splitter.setOrientation(Qt.Horizontal)

        # Left: table of files with rating
        self.table = QTableWidget(self.page1)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["File", "Rating"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Disable mouse hover effects for the table
        try:
            self.table.setAttribute(Qt.WA_Hover, False)
        except Exception:
            pass
        try:
            self.table.setMouseTracking(False)
        except Exception:
            pass
        # Neutralize hover highlight via stylesheet (keeps normal selection highlight)
        self.table.setStyleSheet("QTableView::item:hover { background: transparent; }")
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        splitter.addWidget(self.table)

        # Right: plot and annotation
        right = QWidget(self.page1)
        right_layout = QVBoxLayout(right)
        self.plot = IntensityPlotWidget(self.page1)
        right_layout.addWidget(self.plot)
        right_layout.addWidget(QLabel("Annotation:"))
        self.annotation = QTextEdit(self.page1)
        self.annotation.textChanged.connect(self._on_annotation_changed)
        right_layout.addWidget(self.annotation)
        splitter.addWidget(right)

        p1_layout.addWidget(splitter)

        # Add pages to root
        self.root_layout.addWidget(self.page0)
        self.root_layout.addWidget(self.page1)
        self.page1.hide()

        # cache of last plotted file
        self._current_file: Optional[pathlib.Path] = None
        self._annotation_changing: bool = False

        # Accept drops on the whole widget and the file table
        self.setAcceptDrops(True)
        self.table.setAcceptDrops(True)
        # Forward drag-and-drop events from table to this widget via event filter
        self.table.installEventFilter(self)

    def _on_continue(self):
        # Store setup settings and selected channels
        self.setup_settings = self.detector_page.get_settings()
        # Union of all detector channels
        chs: List[int] = []
        for det in self.setup_settings.get("detectors", {}).values():
            for c in det.get("chs", []):
                if c not in chs:
                    chs.append(c)
        self.selected_channels = sorted(chs) if chs else None
        self.page0.hide()
        self.page1.show()

    def _on_pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder with PTU/TTTR files")
        if not path:
            return
        self._open_folder(pathlib.Path(path))

    def _open_folder(self, folder: pathlib.Path):
        try:
            if not folder.exists() or not folder.is_dir():
                return
            self.current_folder = folder
            self.folder_label.setText(str(folder))
            self.meta = _load_meta(folder)
            self._scan_and_fill()
        except Exception:
            pass

    def _scan_and_fill(self):
        if not self.current_folder:
            return
        # Find files
        files = []
        supported_exts = set(get_tttr_supported_exts())
        for p in sorted(self.current_folder.iterdir()):
            if p.is_file() and p.suffix.lower() in supported_exts:
                files.append(p)
        # Build rows according to filter/sort
        rows: List[Tuple[pathlib.Path, int]] = []
        for p in files:
            rec = self.meta.get(p.name, {})
            rating = int(rec.get("rating", 0))
            if self._filter_accept(rating):
                rows.append((p, rating))
        # Sort
        if self.sort_combo.currentIndex() == 1:  # rating desc
            rows.sort(key=lambda t: (t[1], t[0].name.lower()), reverse=True)
        else:
            rows.sort(key=lambda t: t[0].name.lower())

        # Fill table
        self.table.setRowCount(len(rows))
        for r, (p, rating) in enumerate(rows):
            item_name = QTableWidgetItem(p.name)
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            item_name.setData(Qt.UserRole, str(p))
            self.table.setItem(r, 0, item_name)

            combo = StarCombo(self.table)
            combo.set_rating(rating)
            def _on_combo_changed(idx, path=p, c=combo):
                self._update_rating(path, c.rating())
                # Refresh the list to respect active filters/sorting
                self._scan_and_fill()
            combo.currentIndexChanged.connect(_on_combo_changed)
            self.table.setCellWidget(r, 1, combo)

        if rows:
            self.table.selectRow(0)
        else:
            self._clear_plot_and_annotation()

    def _update_rating(self, path: pathlib.Path, rating: int):
        if not self.current_folder:
            return
        rec = self.meta.get(path.name) or {}
        rec["rating"] = int(rating)
        self.meta[path.name] = rec
        _save_meta(self.current_folder, self.meta)

    def _apply_filter(self):
        self._scan_and_fill()

    def _refresh_list(self):
        self._scan_and_fill()

    def _filter_accept(self, rating: int) -> bool:
        idx = self.filter_combo.currentIndex()
        if idx == 0:
            return True
        elif idx == 1:
            return rating >= 1
        elif idx == 2:
            return rating >= 2
        elif idx == 3:
            return rating >= 3
        elif idx == 4:
            return rating == 0
        return True

    def _on_selection_changed(self):
        paths = self._selected_paths()
        if not paths:
            self._clear_plot_and_annotation()
            return
        # Plot only first selected for preview
        self._plot_file(paths[0])
        # Load annotation of the first selected
        self._annotation_changing = True
        rec = self.meta.get(paths[0].name, {})
        self.annotation.setPlainText(rec.get("annotation", ""))
        self._annotation_changing = False

    def _on_annotation_changed(self):
        if self._annotation_changing or not self.current_folder:
            return
        paths = self._selected_paths()
        if not paths:
            return
        # Update annotation of the first selected
        p = paths[0]
        rec = self.meta.get(p.name) or {}
        rec["annotation"] = self.annotation.toPlainText()
        self.meta[p.name] = rec
        _save_meta(self.current_folder, self.meta)

    def _on_window_changed(self, _):
        # Re-plot with new binning if a file is selected
        paths = self._selected_paths()
        if paths:
            self._plot_file(paths[0])

    def _plot_file(self, path: pathlib.Path):
        if not tttrlib:
            self.folder_label.setText("tttrlib not available - cannot plot")
            return
        try:
            window_ms = self.window_ms_spin.value()
            time_window_s = float(window_ms) / 1000.0
            # Determine channels
            sel_chs = self.selected_channels
            if sel_chs is None:
                # Fall back to all channels in file
                tttr_obj = tttrlib.TTTR(str(path))
                sel_chs = sorted(tttr_obj.get_used_routing_channels())
            # Use IntensityTrace.process_ptu; when a setup is available, aggregate per detector with microtime gating
            if isinstance(self.setup_settings, dict) and 'detectors' in self.setup_settings:
                dets = self.setup_settings.get('detectors') or {}
                time_axis, padded, labels = IntensityTrace().process_ptu(path, time_window_s, sel_chs, detectors=dets)
            else:
                time_axis, padded, chs = IntensityTrace().process_ptu(path, time_window_s, sel_chs)
                labels = [str(c) for c in chs]
            # Plot
            self.plot.plot_trace_and_histogram(time_axis, padded, labels,
                                                bin_count=100,
                                                time_window_ms=window_ms,
                                                hist_min=None, hist_max=None,
                                                hmm_states=None)
            self._current_file = path
        except Exception as e:
            self.folder_label.setText(f"Failed to plot {path.name}: {e}")

    def _clear_plot_and_annotation(self):
        self.plot.plot_widget.clear()
        self.annotation.clear()
        self._current_file = None

    def _selected_paths(self) -> List[pathlib.Path]:
        sel = []
        for idx in self.table.selectionModel().selectedRows():
            item = self.table.item(idx.row(), 0)
            if item is not None:
                p = pathlib.Path(item.data(Qt.UserRole))
                sel.append(p)
        return sel

    def _build_channel_labels(self, chs: List[int]) -> List[str]:
        # Build labels like "<detector_name>, start-end[;start2-end2]" for each routing channel
        try:
            settings = self.setup_settings or {}
            dets = settings.get("detectors", {}) if isinstance(settings, dict) else {}
            # Map routing channel -> list of label parts (in case multiple detectors include same channel)
            label_map: dict[int, List[str]] = {}
            for det_name, dinfo in dets.items():
                try:
                    det_chs = list(dinfo.get("chs", []))
                    mtrs = dinfo.get("micro_time_ranges", []) or []
                    # Build range text
                    rng_txt = ";".join(f"{int(a)}-{int(b)}" for (a, b) in mtrs if isinstance(a, (int, float)) and isinstance(b, (int, float)))
                    base = det_name if det_name is not None else ""
                    lbl = f"{base}, {rng_txt}" if rng_txt else base
                    for ch in det_chs:
                        label_map.setdefault(int(ch), []).append(lbl)
                except Exception:
                    continue
            labels: List[str] = []
            for ch in chs:
                parts = label_map.get(int(ch))
                if parts:
                    # Deduplicate identical parts while preserving order
                    seen = set()
                    uniq = []
                    for p in parts:
                        if p not in seen:
                            seen.add(p)
                            uniq.append(p)
                    labels.append(" + ".join(uniq))
                else:
                    labels.append(str(ch))
            return labels
        except Exception:
            # Fallback: just stringify channels
            return [str(c) for c in chs]

    # Drag-and-drop support
    def eventFilter(self, obj, event):
        try:
            if obj is self.table and event is not None:
                et = event.type()
                if et in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop):
                    # Forward to the widget handlers
                    if et == QEvent.DragEnter:
                        self.dragEnterEvent(event)
                        return True
                    elif et == QEvent.DragMove:
                        self.dragMoveEvent(event)
                        return True
                    elif et == QEvent.Drop:
                        self.dropEvent(event)
                        return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _first_dropped_directory(self, event) -> Optional[pathlib.Path]:
        try:
            md = event.mimeData()
            if md and md.hasUrls():
                for url in md.urls():
                    local = url.toLocalFile()
                    if local:
                        p = pathlib.Path(local)
                        if p.exists() and p.is_dir():
                            return p
        except Exception:
            return None
        return None

    def dragEnterEvent(self, event):
        folder = self._first_dropped_directory(event)
        if folder is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        folder = self._first_dropped_directory(event)
        if folder is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        folder = self._first_dropped_directory(event)
        if folder is not None:
            event.acceptProposedAction()
            self._open_folder(folder)
        else:
            event.ignore()

    def _on_export(self):
        paths = self._selected_paths()
        if not paths:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select destination folder")
        if not out_dir:
            return
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for p in paths:
            try:
                shutil.copy2(str(p), str(out / p.name))
            except Exception:
                pass

    def _on_export_docx(self):
        # Collect all paths that are currently displayed in the table (respecting filter/sort)
        paths: List[pathlib.Path] = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None:
                p_str = item.data(Qt.UserRole)
                if p_str:
                    paths.append(pathlib.Path(p_str))
        if not paths:
            return
        # Check for python-docx availability
        if Document is None:
            try:
                QMessageBox.warning(self, "DOCX Export", "python-docx is not installed. Please install 'python-docx' to enable DOCX export.")
            except Exception:
                pass
            return
        # Determine output path automatically: save as '<foldername>.docx' in the parent of that folder
        folder = self.current_folder
        if folder is None and paths:
            folder = paths[0].parent
        if folder is None:
            try:
                QMessageBox.critical(self, "DOCX Export", "No folder context available to determine DOCX save location.")
            except Exception:
                pass
            return
        docx_name = (folder.name or "traces") + ".docx"
        parent_dir = folder.parent if folder.parent != folder else pathlib.Path(".")
        save_path = parent_dir / docx_name
        try:
            import tempfile
            tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="trace_export_"))
        except Exception:
            tmpdir = self.current_folder or pathlib.Path(".")
        doc = Document()
        # Add a title
        try:
            doc.add_heading("Trace Browser Export", level=1)
        except Exception:
            pass
        # Remember current selection to restore later
        current_selected = self._selected_paths()
        # Iterate displayed traces
        for p in paths:
            # Plot the file in the existing widget to reuse rendering
            try:
                self._plot_file(p)
                QApplication.processEvents()
            except Exception:
                pass
            # Save snapshot of the plot area
            img_path = tmpdir / f"{p.stem}.png"
            try:
                pix = self.plot.plot_widget.grab()
                pix.save(str(img_path), "PNG")
            except Exception:
                img_path = None
            # Read metadata
            rec = self.meta.get(p.name, {})
            rating = int(rec.get("rating", 0))
            annotation = rec.get("annotation", "")
            folder_text = str(p.parent)
            # Write docx content
            try:
                doc.add_heading(p.name, level=2)
            except Exception:
                pass
            try:
                doc.add_paragraph(f"Folder: {folder_text}")
                doc.add_paragraph(f"Rating: {rating}")
                if annotation:
                    doc.add_paragraph(f"Annotation: {annotation}")
            except Exception:
                pass
            # Insert image if available
            if img_path and img_path.exists():
                try:
                    if Inches is not None:
                        doc.add_picture(str(img_path), width=Inches(6))
                    else:
                        doc.add_picture(str(img_path))
                except Exception:
                    pass
            try:
                doc.add_paragraph("")  # spacing
            except Exception:
                pass
        # Save the document
        try:
            doc.save(str(save_path))
            try:
                QMessageBox.information(self, "DOCX Export", f"Saved: {save_path}")
            except Exception:
                pass
        except Exception as e:
            try:
                QMessageBox.critical(self, "DOCX Export", f"Failed to save DOCX: {e}")
            except Exception:
                pass
        # Cleanup temp images
        try:
            if tmpdir and tmpdir.exists() and tmpdir.name.startswith("trace_export_"):
                import shutil as _sh
                _sh.rmtree(str(tmpdir), ignore_errors=True)
        except Exception:
            pass
        # Restore the first selected plot if any, otherwise the first row
        try:
            if current_selected:
                self._plot_file(current_selected[0])
            elif paths:
                self._plot_file(paths[0])
        except Exception:
            pass


if __name__ == "__main__":
    # Basic manual run to show the widget standalone
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = TraceBrowser()
    w.show()
    sys.exit(app.exec_())

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed by the Plugin Manager
if __name__ == "plugin":
    window = TraceBrowser()
    window.show()
