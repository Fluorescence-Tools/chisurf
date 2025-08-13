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
from PyQt5.QtCore import Qt, QEvent, pyqtSignal, QSize

# Logging
from chisurf import logging

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
        # Make Tab/Shift+Tab jump directly between rating controls (same column, next/prev row)
        try:
            key = event.key()
            if key in (Qt.Key_Tab, Qt.Key_Backtab):
                table = self.parent()
                if isinstance(table, QTableWidget):
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
                        if 0 <= next_row < table.rowCount():
                            nxt = table.cellWidget(next_row, col)
                            # Accept either StarCombo or StarRatingWidget
                            if isinstance(nxt, (StarCombo, StarRatingWidget)):
                                table.selectRow(next_row)
                                nxt.setFocus(Qt.TabFocusReason)
                                event.accept()
                                return
        except Exception:
            pass
        super().keyPressEvent(event)


class StarRatingWidget(QWidget):
    ratingChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rating = 0
        self._stars = 3
        self._padding = 4
        self.setFocusPolicy(Qt.StrongFocus)
        try:
            self.setCursor(Qt.PointingHandCursor)
        except Exception:
            pass

    def sizeHint(self):
        try:
            # Approximate width for 3 stars at a decent font size
            return QSize(60, 22)
        except Exception:
            return super().sizeHint()

    def set_rating(self, r: int):
        r = max(0, min(self._stars, int(r)))
        if r != self._rating:
            self._rating = r
            self.update()

    def rating(self) -> int:
        return int(self._rating)

    def _rating_from_pos(self, x: int) -> int:
        # Map click x-position to 1..3; clicking same rating toggles to 0
        w = max(1, self.width() - 2 * self._padding)
        rel = max(0.0, min(1.0, (x - self._padding) / float(w)))
        new_r = int(rel * self._stars) + 1
        new_r = max(1, min(self._stars, new_r))
        # toggle off if clicking same value
        if new_r == self._rating:
            return 0
        return new_r

    def mousePressEvent(self, event):
        try:
            if event.button() == Qt.LeftButton:
                new_r = self._rating_from_pos(event.x())
                if new_r != self._rating:
                    self._rating = new_r
                    self.update()
                    self.ratingChanged.emit(int(self._rating))
                    event.accept()
                    return
            elif event.button() == Qt.RightButton:
                # Right-click clears rating
                if self._rating != 0:
                    self._rating = 0
                    self.update()
                    self.ratingChanged.emit(0)
                    event.accept()
                    return
        except Exception:
            pass
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        try:
            key = event.key()
            if key in (Qt.Key_Left, Qt.Key_Down):
                self.set_rating(self._rating - 1)
                self.ratingChanged.emit(int(self._rating))
                event.accept()
                return
            elif key in (Qt.Key_Right, Qt.Key_Up):
                self.set_rating(self._rating + 1)
                self.ratingChanged.emit(int(self._rating))
                event.accept()
                return
            elif key in (Qt.Key_Tab, Qt.Key_Backtab):
                table = self.parent()
                if isinstance(table, QTableWidget):
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
                        if 0 <= next_row < table.rowCount():
                            nxt = table.cellWidget(next_row, col)
                            if isinstance(nxt, (StarCombo, StarRatingWidget)):
                                table.selectRow(next_row)
                                nxt.setFocus(Qt.TabFocusReason)
                                event.accept()
                                return
        except Exception:
            pass
        super().keyPressEvent(event)

    def paintEvent(self, event):
        try:
            from PyQt5.QtGui import QPainter, QColor, QFont
        except Exception:
            return super().paintEvent(event)
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            rect = self.rect().adjusted(self._padding, 0, -self._padding, 0)
            # Choose font size to fit height
            font = painter.font()
            font.setPointSize(max(9, int(rect.height() * 0.6)))
            painter.setFont(font)
            stars_str = ("★" * int(self._rating)) + ("☆" * int(self._stars - self._rating))
            # Center text
            painter.setPen(QColor(240, 180, 0))
            painter.drawText(rect, Qt.AlignCenter, stars_str)
        finally:
            painter.end()


class NoHoverSelectTable(QTableWidget):
    """A table view that never changes selection on mere mouse hover.
    Selection only changes on clicks/keyboard. Mouse move without button is ignored.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            self.setMouseTracking(False)
            self.setAttribute(Qt.WA_Hover, False)
        except Exception:
            pass

    def mouseMoveEvent(self, event):
        try:
            if getattr(event, 'buttons', lambda: Qt.NoButton)() == Qt.NoButton:
                # Ignore pure hover moves to avoid hover-driven selection changes
                event.ignore()
                return
        except Exception:
            pass
        try:
            super().mouseMoveEvent(event)
        except Exception:
            pass

    def event(self, event):
        try:
            et = event.type() if event is not None else None
            if et in (QEvent.HoverEnter, QEvent.HoverMove, QEvent.HoverLeave):
                return False  # do not handle hover events at all
        except Exception:
            pass
        return super().event(event)


class TraceBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trace Browser")

        # State
        self.current_folder: Optional[pathlib.Path] = None
        self.meta: Dict[str, Dict] = {}
        self.setup_settings: Optional[Dict] = None
        self.selected_channels: Optional[List[int]] = None
        self._is_loading: bool = False

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
        
        # Back to setup button
        self.btn_back = QPushButton("\u2190 Back to setup", self.page1)
        self.btn_back.clicked.connect(self._on_back_to_setup)
        ctrl_row.addWidget(self.btn_back)
        
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


        # Clear button to clear the file list
        self.btn_clear = QPushButton("Clear", self.page1)
        self.btn_clear.setToolTip("Clear file list")
        self.btn_clear.clicked.connect(self._on_clear)

        ctrl_row.addWidget(self.folder_label)
        ctrl_row.addWidget(self.btn_pick_folder)
        ctrl_row.addWidget(QLabel("Filter:"))
        ctrl_row.addWidget(self.filter_combo)
        ctrl_row.addWidget(self.btn_clear)

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

        # Left: table of files with rating and size
        self.table = NoHoverSelectTable(self.page1)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File", "Rating", "Size (MB)"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        # Make Size column a bit narrower
        try:
            self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        except Exception:
            pass
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # Enable header-based sorting
        try:
            self.table.setSortingEnabled(True)
            self.table.horizontalHeader().setSortIndicatorShown(True)
        except Exception:
            pass
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
        # In-memory cache: path -> (sig, (time_axis, padded, labels))
        self._trace_mem_cache: Dict[Tuple[pathlib.Path, str], Tuple[np.ndarray, np.ndarray, List[str]]] = {}
        # Cache for quick image-file detection (path -> bool)
        self._is_image_cache: Dict[pathlib.Path, bool] = {}

        # Accept drops on the whole widget and the file table
        self.setAcceptDrops(True)
        self.table.setAcceptDrops(True)
        # Forward drag-and-drop events from table to this widget via event filter
        self.table.installEventFilter(self)

    def _on_continue(self):
        # Store setup settings and selected channels
        self.setup_settings = self.detector_page.get_settings()
        logging.info("TraceBrowser: Setup accepted from DetectorWizard")
        # Union of all detector channels
        chs: List[int] = []
        for det in self.setup_settings.get("detectors", {}).values():
            for c in det.get("chs", []):
                if c not in chs:
                    chs.append(c)
        self.selected_channels = sorted(chs) if chs else None
        logging.debug(f"TraceBrowser: Selected channels = {self.selected_channels}")
        self.page0.hide()
        self.page1.show()

    def _on_back_to_setup(self):
        logging.info("TraceBrowser: Back to setup")
        try:
            self.page1.hide()
            self.page0.show()
        except Exception:
            pass
        
    def _on_pick_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder with PTU/TTTR files")
        if not path:
            return
        self._open_folder(pathlib.Path(path))

    def _open_folder(self, folder: pathlib.Path):
        try:
            if not folder.exists() or not folder.is_dir():
                logging.warning(f"TraceBrowser: Selected path is not a folder: {folder}")
                return
            self._is_loading = True
            self.current_folder = folder
            self.folder_label.setText(str(folder))
            logging.info(f"TraceBrowser: Opened folder {folder}")
            self.meta = _load_meta(folder)
            self._scan_and_fill()
        except Exception as e:
            logging.exception(f"TraceBrowser: Failed to open folder {folder}: {e}")
        finally:
            self._is_loading = False

    def _rel_key(self, path: pathlib.Path) -> str:
        try:
            if self.current_folder is not None:
                return str(pathlib.Path(path).resolve().relative_to(self.current_folder.resolve()))
        except Exception:
            pass
        try:
            return pathlib.Path(path).name
        except Exception:
            return str(path)

    def _meta_get(self, path: pathlib.Path) -> Dict:
        key = self._rel_key(path)
        rec = self.meta.get(key)
        if rec is not None:
            return rec
        # Backward compatibility: older meta by filename only
        try:
            return self.meta.get(pathlib.Path(path).name, {})
        except Exception:
            return {}

    def _meta_set(self, path: pathlib.Path, rec: Dict):
        key = self._rel_key(path)
        self.meta[key] = rec
        # Optionally remove old name-only key to avoid duplicates
        try:
            name_key = pathlib.Path(path).name
            if name_key != key and name_key in self.meta:
                del self.meta[name_key]
        except Exception:
            pass

    def _scan_and_fill(self):
        if not self.current_folder:
            return
        # Find files recursively
        files: List[pathlib.Path] = []
        # Determine allowed extensions based on selected setup file type
        allowed_exts = self._allowed_exts_for_setup()
        try:
            it = self.current_folder.rglob('*')
        except Exception:
            it = self.current_folder.iterdir()
        for p in sorted(it):
            try:
                if p.is_file() and (not allowed_exts or p.suffix.lower() in allowed_exts):
                    # Skip CLSM/image TTTR files; Trace Browser should only list non-image TTTRs
                    try:
                        if self._is_image_tttr(p):
                            continue
                    except Exception:
                        pass
                    files.append(p)
            except Exception:
                continue
        # Build rows according to filter/sort
        rows: List[Tuple[pathlib.Path, int]] = []
        for p in files:
            rec = self._meta_get(p)
            rating = int(rec.get("rating", 0))
            if self._filter_accept(rating):
                rows.append((p, rating))
        # Do not sort here; allow user to click header to sort

        logging.debug(f"TraceBrowser: Found {len(files)} files (recursive), displaying {len(rows)} after filter")
        # Fill table
        self.table.setRowCount(len(rows))
        for r, (p, rating) in enumerate(rows):
            # Display relative path for clarity when using subfolders
            try:
                rel_txt = str(p.relative_to(self.current_folder))
            except Exception:
                rel_txt = p.name
            item_name = QTableWidgetItem(rel_txt)
            item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
            item_name.setData(Qt.UserRole, str(p))
            self.table.setItem(r, 0, item_name)

            # Provide an item in the Rating column for proper sorting
            rating_item = QTableWidgetItem()
            rating_item.setFlags(rating_item.flags() & ~Qt.ItemIsEditable)
            rating_item.setData(Qt.EditRole, int(rating))  # numeric sort key
            self.table.setItem(r, 1, rating_item)

            # Size column (bytes with human-readable display)
            try:
                sz = p.stat().st_size
            except Exception:
                sz = 0
            size_item = QTableWidgetItem()
            size_item.setFlags(size_item.flags() & ~Qt.ItemIsEditable)
            size_item.setText(self._human_size(sz))
            size_item.setToolTip(f"{self._human_size(sz)} ({sz} bytes)")
            size_item.setData(Qt.EditRole, int(sz))  # numeric sort key
            self.table.setItem(r, 2, size_item)

            stars = StarRatingWidget(self.table)
            stars.set_rating(rating)
            def _on_rating_changed(val, row=r, path=p, w=stars):
                # Update meta
                self._update_rating(path, int(val))
                # Update sort key for the rating column item
                it = self.table.item(row, 1)
                if it is not None:
                    it.setData(Qt.EditRole, int(val))
                # Re-apply current filter and maintain current sorting (lightweight)
                self._refresh_list()
                try:
                    header = self.table.horizontalHeader()
                    self.table.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())
                except Exception:
                    pass
            stars.ratingChanged.connect(_on_rating_changed)
            self.table.setCellWidget(r, 1, stars)

        # Initial sort by File ascending for convenience
        try:
            self.table.sortItems(0, Qt.AscendingOrder)
        except Exception:
            pass
        if rows:
            # Keep selection on the first visible row
            self.table.selectRow(0)
        else:
            self._clear_plot_and_annotation()
        # After populating, precompute traces for all listed files for snappy browsing
        try:
            self._precompute_all_traces()
        except Exception as _e:
            logging.debug(f"TraceBrowser: Precompute skipped or failed: {_e}")

    def _update_rating(self, path: pathlib.Path, rating: int):
        if not self.current_folder:
            return
        rec = self._meta_get(path)
        rec["rating"] = int(rating)
        self._meta_set(path, rec)
        _save_meta(self.current_folder, self.meta)

    def _apply_filter(self):
        # Lightweight refresh: only update row visibility based on current filter and allowed extensions
        self._refresh_list()

    def _refresh_list(self):
        # Update row visibility without rescanning files or recomputing traces
        try:
            rows = []
            for r in range(self.table.rowCount()):
                item0 = self.table.item(r, 0)
                if item0 is None:
                    continue
                p_str = item0.data(Qt.UserRole)
                if not p_str:
                    continue
                p = pathlib.Path(p_str)
                rec = self._meta_get(p)
                rating = int(rec.get("rating", 0))
                rows.append((r, p, rating))
            # Filter rows by rating and by allowed extensions (in case setup filetype changed)
            idx = self.filter_combo.currentIndex()
            allowed_exts = self._allowed_exts_for_setup()
            def accept(rt: int, path: pathlib.Path) -> bool:
                rating_ok = (
                    idx == 0 or
                    (idx == 1 and rt >= 1) or
                    (idx == 2 and rt >= 2) or
                    (idx == 3 and rt >= 3) or
                    (idx == 4 and rt == 0)
                )
                ext_ok = (not allowed_exts) or (path.suffix.lower() in allowed_exts)
                return rating_ok and ext_ok
            visible = [t[0] for t in rows if accept(t[2], t[1])]
            # Hide all, then show accepted
            for r in range(self.table.rowCount()):
                self.table.setRowHidden(r, True)
            for r in visible:
                self.table.setRowHidden(r, False)
            # Keep a valid selection
            try:
                sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
                sel_rows = [s.row() for s in sel]
                sel_rows = [r for r in sel_rows if r in visible]
                if not sel_rows and visible:
                    self.table.selectRow(visible[0])
            except Exception:
                if visible:
                    self.table.selectRow(visible[0])
            # Maintain existing sort order
            try:
                header = self.table.horizontalHeader()
                self.table.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())
            except Exception:
                pass
        except Exception:
            # Fallback silently on any error
            pass

    def _allowed_exts_for_setup(self) -> set:
        """Return a set of allowed file extensions (lowercase, with dot) based on the selected setup's file type.
        If Auto or unavailable, return all tttr-supported extensions. """
        try:
            # Ask DetectorWizardPage for selected file type
            filetype = None
            try:
                filetype = self.detector_page.filetype  # returns None for Auto
            except Exception:
                filetype = None
            # Base set: all supported exts (normalized)
            all_exts = set(get_tttr_supported_exts())
            if not filetype or str(filetype).strip().lower() == 'auto':
                logging.debug(f"TraceBrowser: Using all supported extensions (Auto): {sorted(all_exts)}")
                return all_exts
            # Map container/format names to typical extensions
            ft = str(filetype).strip().upper()
            mapping = {
                'PTU': {'.ptu'},
                'PT3': {'.pt3'},
                'HT3': {'.ht3'},
                'PT2': {'.pt2'},
                'PT5': {'.pt5'},
                'SPC-130': {'.spc'},
                'SPC-600': {'.spc'},
                'SPC-830': {'.spc'},
                'PHU': {'.phu'},
                'PHOTON_HDF5': {'.h5', '.hdf5', '.photon.hdf5'},
                'HDF5': {'.h5', '.hdf5'},
            }
            exts = mapping.get(ft)
            if exts:
                # Intersect with actually supported to be safe
                result = {e for e in exts if (not all_exts or e in all_exts)} or exts
                logging.debug(f"TraceBrowser: Using extensions for filetype '{filetype}': {sorted(result)}")
                return result
            # Fallback: if unknown filetype name, return all
            logging.debug(f"TraceBrowser: Unknown filetype '{filetype}', falling back to all supported extensions")
            return all_exts
        except Exception:
            return set(get_tttr_supported_exts())

    def _on_clear(self):
        # Clear the file list (non-destructive; does not modify files or metadata)
        try:
            self.table.setRowCount(0)
            self._clear_plot_and_annotation()
            logging.info("TraceBrowser: Cleared file list")
        except Exception:
            pass

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
        # Commit current annotation for the currently shown file before switching
        try:
            self._commit_current_annotation()
        except Exception:
            pass
        paths = self._selected_paths()
        if not paths:
            self._clear_plot_and_annotation()
            return
        # Plot only first selected for preview
        self._plot_file(paths[0])
        # Load annotation of the first selected
        self._annotation_changing = True
        try:
            rec = self._meta_get(paths[0])
            # Block signals while setting text to avoid spurious textChanged
            try:
                self.annotation.blockSignals(True)
            except Exception:
                pass
            self.annotation.setPlainText(rec.get("annotation", ""))
        finally:
            try:
                self.annotation.blockSignals(False)
            except Exception:
                pass
            self._annotation_changing = False

    def _on_annotation_changed(self):
        if getattr(self, '_is_loading', False) or self._annotation_changing or not self.current_folder:
            return
        paths = self._selected_paths()
        if not paths:
            return
        # Update annotation of the first selected
        p = paths[0]
        rec = self._meta_get(p)
        rec["annotation"] = self.annotation.toPlainText()
        self._meta_set(p, rec)
        _save_meta(self.current_folder, self.meta)

    def _commit_current_annotation(self):
        """Persist current annotation text for the currently displayed file, if any."""
        try:
            if not self.current_folder:
                return
            p = getattr(self, '_current_file', None)
            if p is None:
                return
            rec = self._meta_get(p)
            rec["annotation"] = self.annotation.toPlainText()
            self._meta_set(p, rec)
            _save_meta(self.current_folder, self.meta)
        except Exception:
            pass

    def _on_window_changed(self, _):
        # Re-plot with new binning if a file is selected
        paths = self._selected_paths()
        if paths:
            self._plot_file(paths[0])

    # --- Image detection helpers ---
    def _is_clsm_compatible(self, tttr_obj) -> bool:
        """Probe whether this TTTR object supports CLSM imaging (image-like dataset).
        We attempt to construct a minimal CLSMImage and access a lightweight attribute.
        """
        try:
            # If tttrlib or CLSMImage is unavailable, this will raise and we return False
            clsm = tttrlib.CLSMImage(tttr_data=tttr_obj)
            _ = getattr(clsm, 'intensity', None)
            return _ is not None
        except Exception:
            return False

    def _is_image_tttr(self, path: pathlib.Path) -> bool:
        # Cached decision first
        v = self._is_image_cache.get(path)
        if v is not None:
            return v
        if tttrlib is None:
            self._is_image_cache[path] = False
            return False
        try:
            tt = tttrlib.TTTR(str(path))
            v = self._is_clsm_compatible(tt)
            self._is_image_cache[path] = bool(v)
            return bool(v)
        except Exception:
            # On failure to open, treat as non-image to let other filters decide
            self._is_image_cache[path] = False
            return False

    # --- Caching helpers ---
    def _human_size(self, n: int) -> str:
        """Return size in megabytes with one decimal place."""
        try:
            mb = float(n) / (1024.0 * 1024.0)
            return f"{mb:.1f} MB"
        except Exception:
            return "? MB"

    def _cache_dir_for(self, file_path: pathlib.Path) -> pathlib.Path:
        base = self.current_folder or file_path.parent
        d = base / ".tttr_trace_cache"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return d

    def _trace_signature(self, file_path: pathlib.Path, window_ms: int) -> str:
        try:
            st = file_path.stat()
            size = int(getattr(st, 'st_size', 0))
            mtime = int(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))) if hasattr(st, 'st_mtime_ns') else int(st.st_mtime * 1e9)
        except Exception:
            size = 0
            mtime = 0
        # Build mode info (detectors vs selected channels)
        mode = {}
        if isinstance(self.setup_settings, dict) and 'detectors' in self.setup_settings:
            # Reduce detectors info to stable signature
            dets = self.setup_settings.get('detectors') or {}
            mode = {k: {
                'chs': list(v.get('chs', [])),
                'micro_time_ranges': list(v.get('micro_time_ranges', []))
            } for k, v in dets.items()}
        else:
            mode = {'chs': list(self.selected_channels) if self.selected_channels else None}
        try:
            key = {
                'v': 1,
                'path': str(file_path.resolve()),
                'size': size,
                'mtime': mtime,
                'win_ms': int(window_ms),
                'mode': mode,
            }
            import json as _json
            s = _json.dumps(key, sort_keys=True, separators=(',', ':'))
            import hashlib
            return hashlib.sha256(s.encode('utf-8')).hexdigest()
        except Exception:
            return f"fallback_{file_path.name}_{window_ms}_{size}_{mtime}"

    def _trace_cache_file(self, file_path: pathlib.Path, window_ms: int) -> pathlib.Path:
        sig = self._trace_signature(file_path, window_ms)
        return self._cache_dir_for(file_path) / f"{file_path.stem}_{sig}.npz"

    def _load_trace_cache(self, file_path: pathlib.Path, window_ms: int):
        try:
            p = self._trace_cache_file(file_path, window_ms)
            if p.exists():
                with np.load(str(p), allow_pickle=True) as z:
                    time_axis = z['time_axis']
                    padded = z['padded']
                    labels = list(z['labels']) if 'labels' in z else []
                    return time_axis, padded, labels
        except Exception:
            pass
        return None

    def _save_trace_cache(self, file_path: pathlib.Path, window_ms: int, time_axis, padded, labels):
        try:
            p = self._trace_cache_file(file_path, window_ms)
            np.savez_compressed(str(p), time_axis=time_axis, padded=padded, labels=np.array(labels, dtype=object))
        except Exception:
            pass

    def _compute_trace_cached(self, file_path: pathlib.Path, window_ms: int):
        # Memory cache first
        sig = self._trace_signature(file_path, window_ms)
        key = (file_path, sig)
        m = self._trace_mem_cache.get(key)
        if m is not None:
            return m
        # Disk cache
        loaded = self._load_trace_cache(file_path, window_ms)
        if loaded is not None:
            self._trace_mem_cache[key] = loaded
            return loaded
        # Compute
        time_window_s = float(window_ms) / 1000.0
        sel_chs = self.selected_channels
        if sel_chs is None:
            try:
                tttr_obj = tttrlib.TTTR(str(file_path))
                sel_chs = sorted(tttr_obj.get_used_routing_channels())
            except Exception:
                sel_chs = []
        if isinstance(self.setup_settings, dict) and 'detectors' in self.setup_settings:
            dets = self.setup_settings.get('detectors') or {}
            time_axis, padded, labels = IntensityTrace().process_ptu(file_path, time_window_s, sel_chs, detectors=dets)
        else:
            time_axis, padded, chs = IntensityTrace().process_ptu(file_path, time_window_s, sel_chs)
            labels = [str(c) for c in chs]
        # Save caches
        self._trace_mem_cache[key] = (time_axis, padded, labels)
        self._save_trace_cache(file_path, window_ms, time_axis, padded, labels)
        return time_axis, padded, labels

    def _precompute_all_traces(self):
        if self.current_folder is None or tttrlib is None:
            return
        # Collect paths from table
        paths: List[pathlib.Path] = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is None:
                continue
            p_str = item.data(Qt.UserRole)
            if p_str:
                pth = pathlib.Path(p_str)
                # Defensive: skip image files even if present in table for any reason
                try:
                    if self._is_image_tttr(pth):
                        continue
                except Exception:
                    pass
                paths.append(pth)
        if not paths:
            return
        window_ms = self.window_ms_spin.value()
        # Determine which files actually need computation; warm memory cache for those already on disk
        files_to_process: List[pathlib.Path] = []
        for p in paths:
            try:
                loaded = self._load_trace_cache(p, window_ms)
                if loaded is not None:
                    # Warm in-memory cache for instant browsing
                    sig = self._trace_signature(p, window_ms)
                    self._trace_mem_cache[(p, sig)] = loaded
                else:
                    files_to_process.append(p)
            except Exception:
                files_to_process.append(p)
        # If everything is already cached, skip recomputation and avoid showing any dialog
        if not files_to_process:
            return
        # Show progress dialog only for files we are going to process
        try:
            from PyQt5.QtWidgets import QProgressDialog
            dlg = QProgressDialog("Precomputing traces...", "Cancel", 0, len(files_to_process), self)
            dlg.setWindowTitle("Precomputing traces")
            dlg.setAutoClose(True)
            dlg.setAutoReset(False)
            dlg.setMinimumDuration(0)
        except Exception:
            dlg = None
        try:
            for i, p in enumerate(files_to_process):
                if dlg is not None:
                    try:
                        dlg.setValue(i)
                        dlg.setLabelText(f"Processing {p.name} ({i+1}/{len(files_to_process)})")
                    except Exception:
                        pass
                QApplication.processEvents()
                if dlg is not None and dlg.wasCanceled():
                    break
                try:
                    _ = self._compute_trace_cached(p, window_ms)
                except Exception:
                    pass
        finally:
            if dlg is not None:
                try:
                    dlg.setValue(len(files_to_process))
                    dlg.close()
                except Exception:
                    pass

    def _plot_file(self, path: pathlib.Path):
        if not tttrlib:
            logging.warning("TraceBrowser: tttrlib not available - cannot plot")
            self.folder_label.setText("tttrlib not available - cannot plot")
            return
        try:
            window_ms = self.window_ms_spin.value()
            # Use cached precomputed results if available
            time_axis, padded, labels = self._compute_trace_cached(path, window_ms)
            # Plot
            self.plot.plot_trace_and_histogram(time_axis, padded, labels,
                                                bin_count=100,
                                                time_window_ms=window_ms,
                                                hist_min=None, hist_max=None,
                                                hmm_states=None)
            self._current_file = path
        except Exception as e:
            logging.exception(f"TraceBrowser: Failed to plot {path}: {e}")
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
        # Export all files currently listed (respecting active filter/sort)
        paths: List[pathlib.Path] = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None:
                p_str = item.data(Qt.UserRole)
                if p_str:
                    paths.append(pathlib.Path(p_str))
        if not paths:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select destination folder")
        if not out_dir:
            return
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        copied = 0
        for p in paths:
            try:
                shutil.copy2(str(p), str(out / p.name))
                copied += 1
            except Exception as e:
                logging.warning(f"TraceBrowser: Failed to copy {p} to {out}: {e}")
        logging.info(f"TraceBrowser: Exported {copied}/{len(paths)} files to {out}")

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
            rec = self._meta_get(p)
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
            logging.info(f"TraceBrowser: DOCX exported to {save_path}")
            try:
                QMessageBox.information(self, "DOCX Export", f"Saved: {save_path}")
            except Exception:
                pass
        except Exception as e:
            logging.exception(f"TraceBrowser: Failed to save DOCX {save_path}: {e}")
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
