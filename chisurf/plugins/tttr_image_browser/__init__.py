"""
TTTR Image Browser Plugin

Browse TTTR files in a folder and preview intensity images for all
DetectorWizard-defined detector windows (window × detector combinations).
Includes: star rating (0–3) per file, annotations, filtering/sorting,
export selected files, and DOCX export identical to TraceBrowser.

Metadata is stored per folder in: .image_browser_meta.json
"""
from __future__ import annotations
import os
import json
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import hashlib

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView, QTextEdit, QComboBox,
    QSpinBox, QSplitter, QMessageBox, QProgressDialog
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QPen
import pyqtgraph as pg

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

# Reuse widgets/utilities from existing plugins
from chisurf.gui.widgets.wizard.tttr_channel_definition import DetectorWizardPage
from chisurf.plugins.trace_browser.__init__ import get_tttr_supported_exts, StarCombo

# Logging
from chisurf import logging

name = "TTTR:Image Browser"

META_FILENAME = ".image_browser_meta.json"
CACHE_DIR_NAME = ".tttr_image_cache"
CACHE_VERSION = "1"


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


class TTTRImageBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTTR Image Browser")

        # State
        self.current_folder: Optional[pathlib.Path] = None
        self.meta: Dict[str, Dict] = {}
        self.setup_settings: Optional[Dict] = None
        self._current_file: Optional[pathlib.Path] = None
        # Keep the last full-resolution pixmap to scale smoothly on resize
        self._fullres_pixmap: Optional[QPixmap] = None
        self._mosaic_cache: Dict[pathlib.Path, Tuple[np.ndarray, List[str], int, int]] = {}
        self._is_loading: bool = False

        # Root layout
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

        # Page 1: Image browser
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
        self.filter_combo.addItems(["All", "≥ 1★", "≥ 2★★", "≥ 3★★★", "Only 0★"])
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)


        # Clear button to clear the file list
        self.btn_clear = QPushButton("Clear", self.page1)
        self.btn_clear.setToolTip("Clear file list")
        self.btn_clear.clicked.connect(self._on_clear)

        self.btn_export = QPushButton("Export selected…", self.page1)
        self.btn_export.clicked.connect(self._on_export)

        self.btn_export_docx = QPushButton("Export DOCX…", self.page1)
        self.btn_export_docx.clicked.connect(self._on_export_docx)

        # Save intensity stacks as TIFFs
        self.btn_save_tiff = QPushButton("Save TIFF…", self.page1)
        self.btn_save_tiff.setToolTip("Save intensity images (per combo) as TIFF stacks (pre-sum over frames)")
        self.btn_save_tiff.clicked.connect(self._on_save_tiff)

        ctrl_row.addWidget(self.folder_label)
        ctrl_row.addWidget(self.btn_pick_folder)
        ctrl_row.addWidget(QLabel("Filter:"))
        ctrl_row.addWidget(self.filter_combo)
        ctrl_row.addWidget(self.btn_clear)
        ctrl_row.addWidget(self.btn_export)
        ctrl_row.addWidget(self.btn_save_tiff)
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
        # Enable header-based sorting
        try:
            self.table.setSortingEnabled(True)
            self.table.horizontalHeader().setSortIndicatorShown(True)
        except Exception:
            pass
        try:
            self.table.setAttribute(Qt.WA_Hover, False)
            self.table.setMouseTracking(False)
        except Exception:
            pass
        self.table.setStyleSheet("QTableView::item:hover { background: transparent; }")
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.table)

        # Right: image canvas (pyqtgraph) and annotation
        right = QWidget(self.page1)
        right_layout = QVBoxLayout(right)
        # PyQtGraph viewer
        self.pg_canvas = pg.GraphicsLayoutWidget(right)
        self.viewbox = self.pg_canvas.addViewBox(lockAspect=True)
        # Display with y going down to match numpy array orientation
        self.viewbox.invertY(False)
        self.image_item = pg.ImageItem()
        self.viewbox.addItem(self.image_item)
        # Apply preferred colormap (magma)
        try:
            # Prefer modern API if available
            if hasattr(self.image_item, 'setColorMap'):
                try:
                    cmap = pg.colormap.get('magma')
                except Exception:
                    cmap = None
                if cmap is not None:
                    self.image_item.setColorMap(cmap)
                else:
                    lut = self._get_magma_lut(256)
                    if lut is not None:
                        self.image_item.setLookupTable(lut)
            else:
                lut = self._get_magma_lut(256)
                if lut is not None:
                    self.image_item.setLookupTable(lut)
        except Exception:
            pass
        right_layout.addWidget(self.pg_canvas)
        # Overlay label items
        self._overlay_items: List[pg.TextItem] = []
        right_layout.addWidget(QLabel("Annotation:"))
        self.annotation = QTextEdit(self.page1)
        # Limit annotation editor height to a maximum of 100 px
        try:
            self.annotation.setMaximumHeight(100)
        except Exception:
            pass
        self.annotation.textChanged.connect(self._on_annotation_changed)
        right_layout.addWidget(self.annotation)
        splitter.addWidget(right)

        p1_layout.addWidget(splitter)

        # Add pages
        self.root_layout.addWidget(self.page0)
        self.root_layout.addWidget(self.page1)
        self.page1.hide()

        # Drag-and-drop like TraceBrowser
        self.setAcceptDrops(True)
        self.table.setAcceptDrops(True)
        self.table.installEventFilter(self)


    def _allowed_exts_for_setup(self) -> set:
        """Return a set of allowed file extensions (lowercase, with dot) based on the selected setup's file type.
        If Auto or unavailable, return all tttr-supported extensions (via get_tttr_supported_exts from TraceBrowser). """
        try:
            # filetype: None if Auto
            try:
                filetype = self.detector_page.filetype
            except Exception:
                filetype = None
            all_exts = set(get_tttr_supported_exts())
            if not filetype or str(filetype).strip().lower() == 'auto':
                logging.debug(f"TTTRImageBrowser: Using all supported extensions (Auto): {sorted(all_exts)}")
                return all_exts
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
                result = {e for e in exts if (not all_exts or e in all_exts)} or exts
                logging.debug(f"TTTRImageBrowser: Using extensions for filetype '{filetype}': {sorted(result)}")
                return result
            logging.debug(f"TTTRImageBrowser: Unknown filetype '{filetype}', falling back to all supported extensions")
            return all_exts
        except Exception:
            return set(get_tttr_supported_exts())

    # --- Page switching ---
    def _on_continue(self):
        # Store settings from detector page
        self.setup_settings = self.detector_page.get_settings()
        self.page0.hide()
        self.page1.show()

    def _on_back_to_setup(self):
        logging.info("TTTRImageBrowser: Back to setup")
        try:
            self.page1.hide()
            self.page0.show()
        except Exception:
            pass

    def _on_pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with TTTR files")
        if not folder:
            return
        self._open_folder(pathlib.Path(folder))

    def _open_folder(self, folder: pathlib.Path):
        self._is_loading = True
        self.current_folder = folder
        self.folder_label.setText(str(folder))
        logging.info(f"TTTRImageBrowser: Opened folder {folder}")
        self.meta = _load_meta(folder)
        try:
            self._scan_and_fill()
        finally:
            self._is_loading = False

    # --- Table population & meta ---
    def _scan_and_fill(self):
        self.table.setRowCount(0)
        if self.current_folder is None:
            return
        # Determine allowed extensions based on selected setup file type
        allowed_exts = self._allowed_exts_for_setup()
        files = []
        for p in self.current_folder.iterdir():
            if not p.is_file():
                continue
            if (not allowed_exts) or (p.suffix.lower() in allowed_exts):
                files.append(p)
        # Sort by name initially
        files.sort(key=lambda p: p.name.lower())

        logging.debug(f"TTTRImageBrowser: Scanning folder {self.current_folder}, found {len(files)} supported files")
        for p in files:
            r = self.table.rowCount()
            self.table.insertRow(r)
            item = QTableWidgetItem(p.name)
            item.setData(Qt.UserRole, str(p))
            self.table.setItem(r, 0, item)

            # Create sortable rating item and StarCombo widget
            rating = int(self.meta.get(p.name, {}).get("rating", 0))
            rating_item = QTableWidgetItem()
            rating_item.setFlags(rating_item.flags() & ~Qt.ItemIsEditable)
            rating_item.setData(Qt.EditRole, int(rating))
            self.table.setItem(r, 1, rating_item)

            combo = StarCombo(self.table)
            combo.set_rating(rating)
            # connect inline handler to update meta and sort key
            def _on_combo_changed(idx, row=r, path=p, c=combo):
                self._update_rating(path, int(c.currentData()))
                it = self.table.item(row, 1)
                if it is not None:
                    it.setData(Qt.EditRole, int(c.currentData()))
                # Re-apply filter and current sorting
                self._refresh_list()
                try:
                    header = self.table.horizontalHeader()
                    self.table.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())
                except Exception:
                    pass
            combo.currentIndexChanged.connect(_on_combo_changed)
            self.table.setCellWidget(r, 1, combo)

        self._apply_filter()
        # Initial sort by File ascending for convenience
        try:
            self.table.sortItems(0, Qt.AscendingOrder)
        except Exception:
            pass
        # After populating the table, precompute images (or load from mosaic cache) with a progress dialog when needed
        try:
            self._precompute_all_images()
        except Exception as _e:
            # Non-fatal: precomputation is best-effort
            logging.debug(f"TTTRImageBrowser: Precompute skipped or failed: {_e}")

    def _update_rating(self, path: pathlib.Path, rating: int):
        rec = self.meta.get(path.name) or {}
        rec["rating"] = int(rating)
        self.meta[path.name] = rec
        if self.current_folder:
            _save_meta(self.current_folder, self.meta)
        # Refresh order if sorting by rating
        self._refresh_list()

    def _apply_filter(self):
        self._refresh_list()

    def _refresh_list(self):
        # Apply filter by rating and re-sort rows
        # Collect rows
        rows = []
        for r in range(self.table.rowCount()):
            item0 = self.table.item(r, 0)
            if item0 is None:
                continue
            p_str = item0.data(Qt.UserRole)
            if not p_str:
                continue
            p = pathlib.Path(p_str)
            rating = int(self.meta.get(p.name, {}).get("rating", 0))
            rows.append((r, p, rating))
        # Filter rows by rating and by allowed extensions (in case filetype changed)
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
        rows = [t for t in rows if accept(t[2], t[1])]
        # Rebuild table visibility based on filter only (no manual sorting)
        for r in range(self.table.rowCount()):
            self.table.setRowHidden(r, True)
        for i, (r, _, _) in enumerate(rows):
            self.table.setRowHidden(r, False)
        # Select first visible row to update preview
        vis = [r for (r,_,_) in rows]
        if vis:
            # Keep current selection if any visible
            if not self.table.selectionModel().selectedRows():
                self.table.selectRow(vis[0])

    def _on_clear(self):
        # Clear the file list (non-destructive; does not modify files or metadata)
        try:
            self.table.setRowCount(0)
            self._clear_preview_and_annotation()
            logging.info("TTTRImageBrowser: Cleared file list")
        except Exception:
            pass

    def _on_selection_changed(self):
        # Commit current annotation for the currently shown file before switching
        try:
            self._commit_current_annotation()
        except Exception:
            pass
        paths = self._selected_paths()
        if not paths:
            self._clear_preview_and_annotation()
            return
        self._plot_file(paths[0])
        # Load annotation for the first selected
        p = paths[0]
        rec = self.meta.get(p.name) or {}
        self._annotation_changing = True
        try:
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
        if getattr(self, '_is_loading', False) or getattr(self, '_annotation_changing', False) or not self.current_folder:
            return
        paths = self._selected_paths()
        if not paths:
            return
        p = paths[0]
        rec = self.meta.get(p.name) or {}
        rec["annotation"] = self.annotation.toPlainText()
        self.meta[p.name] = rec
        _save_meta(self.current_folder, self.meta)

    def _commit_current_annotation(self):
        """Persist current annotation text for the currently displayed file, if any."""
        try:
            if not self.current_folder:
                return
            p = getattr(self, '_current_file', None)
            if p is None:
                return
            rec = self.meta.get(p.name) or {}
            rec["annotation"] = self.annotation.toPlainText()
            self.meta[p.name] = rec
            _save_meta(self.current_folder, self.meta)
        except Exception:
            pass

    def _on_scale_changed(self, _):
        paths = self._selected_paths()
        if paths:
            self._plot_file(paths[0])

    def _plot_file(self, path: pathlib.Path):
        # If not in-memory, try to load mosaic from disk cache
        if path not in self._mosaic_cache:
            try:
                settings = self.setup_settings or {}
                reading = settings.get('tttr_reading', {}) if isinstance(settings, dict) else {}
                reading_routine = reading.get('file_type') or None
                try:
                    channels_map = self.detector_page.channels()
                    channels_map = self._group_channels_by_detector(channels_map)
                except Exception:
                    channels_map = {"Image": [{'window_range': (None, None), 'detector_chs': [], 'micro_time_range': (None, None)}]}
                loaded = self._load_mosaic_cache(path, channels_map, reading_routine)
                if loaded is not None:
                    mosaic, labels, cols, rows, tile_w, tile_h = loaded
                    self._mosaic_cache[path] = (mosaic, labels, cols, rows)
                else:
                    self._clear_preview_and_annotation()
                    return
            except Exception:
                self._clear_preview_and_annotation()
                return
        mosaic, labels, cols, rows = self._mosaic_cache[path]
        self.image_item.setImage(mosaic, autoLevels=True)

        for it in self._overlay_items:
            self.viewbox.removeItem(it)
        self._overlay_items = []
        tile_h = mosaic.shape[0] // rows
        tile_w = mosaic.shape[1] // cols
        for idx, text in enumerate(labels):
            r, c = divmod(idx, cols)
            txt = pg.TextItem(text=text, color=(255, 255, 255))
            txt.setAnchor((0, 0))
            txt.setPos(c * tile_w + 3, r * tile_h + 3)
            self.viewbox.addItem(txt)
            self._overlay_items.append(txt)

        self.viewbox.autoRange()
        self._current_file = path

    def _clear_preview_and_annotation(self):
        # Clear image and overlays
        try:
            self.image_item.setImage(np.zeros((1,1), dtype=np.uint8))
        except Exception:
            pass
        for it in getattr(self, '_overlay_items', []) or []:
            try:
                self.viewbox.removeItem(it)
            except Exception:
                pass
        self._overlay_items = []
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

    # --- Cache utilities ---
    def _cache_dir_for(self, file_path: pathlib.Path) -> pathlib.Path:
        d = file_path.parent / CACHE_DIR_NAME
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return d

    # --- Mosaic disk cache helpers ---
    def _channels_signature(self, channels_map: Dict[str, List[Dict]]) -> List[Dict]:
        """Build a compact signature of channels_map for hashing (order-independent per detector).
        Only includes detector_chs and micro_time_range per entry under each detector name.
        """
        sig: List[Dict] = []
        try:
            if not isinstance(channels_map, dict):
                return sig
            for det_name, entries in sorted(channels_map.items(), key=lambda kv: str(kv[0])):
                part = {
                    'name': str(det_name),
                    'entries': []
                }
                for e in entries or []:
                    chs = e.get('detector_chs') or []
                    try:
                        chs = sorted([int(c) for c in chs])
                    except Exception:
                        pass
                    mtr = e.get('micro_time_range') or None
                    if isinstance(mtr, (list, tuple)) and len(mtr) == 2:
                        try:
                            mtr = [int(mtr[0]), int(mtr[1])]
                        except Exception:
                            mtr = list(mtr)
                    else:
                        mtr = None
                    part['entries'].append({'chs': chs, 'mtr': mtr})
                # Normalize entry order for hashing
                part['entries'].sort(key=lambda d: (tuple(d.get('chs') or []), tuple(d.get('mtr') or [])))
                sig.append(part)
        except Exception:
            return sig
        return sig

    def _mosaic_hash(self, file_path: pathlib.Path, channels_map: Dict[str, List[Dict]], reading_routine) -> str:
        try:
            st = file_path.stat()
            payload = {
                'v': CACHE_VERSION,
                'file': str(file_path.name),
                'mtime_ns': getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9)),
                'reading': str(reading_routine) if reading_routine is not None else 'Auto',
                'channels': self._channels_signature(channels_map),
            }
            s = json.dumps(payload, sort_keys=True)
            return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]
        except Exception:
            return hashlib.sha1((str(file_path)+str(reading_routine)+json.dumps(self._channels_signature(channels_map))).encode('utf-8')).hexdigest()[:16]

    def _mosaic_cache_file(self, file_path: pathlib.Path, channels_map: Dict[str, List[Dict]], reading_routine) -> pathlib.Path:
        key = self._mosaic_hash(file_path, channels_map, reading_routine)
        return self._cache_dir_for(file_path) / f"{file_path.stem}_mosaic_{key}.npz"

    def _save_mosaic_cache(self, file_path: pathlib.Path, channels_map: Dict[str, List[Dict]], reading_routine, mosaic: np.ndarray, labels: List[str], cols: int, rows: int, tile_w: int, tile_h: int):
        try:
            cache_path = self._mosaic_cache_file(file_path, channels_map, reading_routine)
            lab_arr = np.array(labels, dtype=object)
            np.savez_compressed(str(cache_path), mosaic=mosaic, labels=lab_arr, cols=int(cols), rows=int(rows), tile_w=int(tile_w), tile_h=int(tile_h))
        except Exception:
            pass

    def _load_mosaic_cache(self, file_path: pathlib.Path, channels_map: Dict[str, List[Dict]], reading_routine):
        try:
            cache_path = self._mosaic_cache_file(file_path, channels_map, reading_routine)
            if not cache_path.exists():
                return None
            with np.load(str(cache_path), allow_pickle=True) as z:
                mosaic = z['mosaic']
                labels = [str(x) for x in z['labels'].tolist()]
                cols = int(z['cols'])
                rows = int(z['rows'])
                tile_w = int(z['tile_w'])
                tile_h = int(z['tile_h'])
                return mosaic, labels, cols, rows, tile_w, tile_h
        except Exception:
            return None

    def _entry_hash(self, file_path: pathlib.Path, det_chs, mtr, reading_routine) -> str:
        try:
            st = file_path.stat()
            payload = {
                'v': CACHE_VERSION,
                'file': str(file_path.name),
                'mtime_ns': getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9)),
                'reading': str(reading_routine) if reading_routine is not None else 'Auto',
                'chs': list(map(int, det_chs)) if det_chs else [],
                'mtr': [int(mtr[0]), int(mtr[1])] if (isinstance(mtr, (list, tuple)) and len(mtr) == 2) else None,
            }
            s = json.dumps(payload, sort_keys=True)
            return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]
        except Exception:
            # fallback weak hash
            return hashlib.sha1((str(file_path)+str(det_chs)+str(mtr)+str(reading_routine)).encode('utf-8')).hexdigest()[:16]

    def _entry_cache_file(self, file_path: pathlib.Path, det_chs, mtr, reading_routine) -> pathlib.Path:
        key = self._entry_hash(file_path, det_chs, mtr, reading_routine)
        return self._cache_dir_for(file_path) / f"{file_path.stem}_{key}.npz"

    def _compute_entry_stack_cached(self, tttr_obj, file_path: pathlib.Path, det_chs, mtr, reading_routine):
        """Return a 3D stack (n_frames, nx, ny) for a single entry, using cache when available."""
        cache_path = self._entry_cache_file(file_path, det_chs, mtr, reading_routine)
        # Load cache
        try:
            if cache_path.exists():
                with np.load(str(cache_path)) as z:
                    arr = z['stack']
                    return arr
        except Exception:
            pass
        # Compute
        params = {'tttr_data': tttr_obj, 'fill': True}
        if det_chs:
            try:
                params['channels'] = list(det_chs)
            except Exception:
                params['channels'] = det_chs
        if mtr is not None and isinstance(mtr, (list, tuple)) and len(mtr) == 2:
            try:
                a, b = int(mtr[0]), int(mtr[1])
                params['micro_time_ranges'] = [(a, b)]
            except Exception:
                pass
        clsm = tttrlib.CLSMImage(**params)
        img = np.array(clsm.intensity)
        # Ensure 3D (frames, x, y)
        if img.ndim == 2:
            img3 = img[None, ...]
        elif img.ndim == 3:
            img3 = img
        else:
            img3 = np.zeros((1, 1, 1), dtype=img.dtype)
        # Save cache
        try:
            np.savez_compressed(str(cache_path), stack=img3)
        except Exception:
            pass
        return img3

    def _sum_stacks_with_padding(self, stacks: List[np.ndarray]) -> Optional[np.ndarray]:
        if not stacks:
            return None
        # Determine max shape along spatial dims
        max_x = 0
        max_y = 0
        max_f = 0
        for s in stacks:
            if s.ndim != 3:
                continue
            f, x, y = s.shape
            max_f = max(max_f, f)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        if max_x == 0 or max_y == 0:
            return None
        acc = np.zeros((max_f, max_x, max_y), dtype=np.float32)
        for s in stacks:
            if s.ndim != 3:
                continue
            f, x, y = s.shape
            acc[:f, :x, :y] += s.astype(np.float32, copy=False)
        return acc

    def _get_magma_lut(self, n: int = 256):
        try:
            # Try pyqtgraph colormap first
            try:
                cmap = pg.colormap.get('magma')
            except Exception:
                cmap = None
            if cmap is not None:
                lut = cmap.getLookupTable(nPts=max(2, int(n)))
                # Ensure uint8 Nx3
                if lut is not None and lut.ndim == 2 and lut.shape[1] >= 3:
                    if lut.dtype != np.uint8:
                        lut = lut.astype(np.uint8, copy=False)
                    return lut[:, :3]
            # Fallback to matplotlib
            try:
                import matplotlib.cm as cm
                import numpy as _np
                m = cm.get_cmap('magma')
                arr = (m(_np.linspace(0,1,max(2,int(n))))[:, :3] * 255).astype(_np.uint8)
                return arr
            except Exception:
                return None
        except Exception:
            return None

    def _get_combo_stack(self, tttr_obj, file_path: pathlib.Path, entries: List[Dict], reading_routine) -> Optional[np.ndarray]:
        stacks = []
        for e in entries:
            det_chs = e.get('detector_chs') or None
            mtr = e.get('micro_time_range') or None
            try:
                st = self._compute_entry_stack_cached(tttr_obj, file_path, det_chs, mtr, reading_routine)
                stacks.append(st)
            except Exception:
                continue
        return self._sum_stacks_with_padding(stacks)

    def _is_clsm_compatible(self, tttr_obj) -> bool:
        """Quickly probe whether CLSM imaging is supported for this TTTR object.
        We try to construct a minimal CLSMImage. If this raises (e.g., missing CLSM header),
        return False. We keep the probe very light.
        """
        try:
            clsm = tttrlib.CLSMImage(tttr_data=tttr_obj)
            # Access a lightweight attribute to force header read
            _ = getattr(clsm, 'intensity', None)
            return _ is not None
        except Exception:
            return False

    # --- Grouping helpers ---
    def _group_channels_by_detector(self, channels_map: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Collapse window_detector combos into detector-only combos.
        DetectorWizardPage.channels() returns keys like "<window>_<detector>".
        We want only one image per detector, so we merge entries across windows
        by detector name. If a key has no underscore, it's used as-is.
        """
        grouped: Dict[str, List[Dict]] = {}
        try:
            for key, entries in (channels_map or {}).items():
                if not isinstance(key, str):
                    det_name = str(key)
                else:
                    # Split at first underscore to remove window prefix
                    parts = key.split('_', 1)
                    det_name = parts[1] if len(parts) > 1 else parts[0]
                lst = grouped.setdefault(det_name, [])
                # Extend with original entries (window_range is ignored by our imaging logic)
                if isinstance(entries, list):
                    lst.extend(entries)
        except Exception:
            # Fallback: return original map
            return channels_map
        return grouped

    # --- Image generation ---
    def _compute_combo_image(self, tttr_obj, entries: List[Dict]) -> Optional[np.ndarray]:
        """
        Build a 2D intensity image for a given combo defined by DetectorWizardPage.channels().
        For each entry, construct a CLSMImage using the specified routing channels and
        micro time range, then sum the resulting intensity images across all entries.
        """
        if tttr_obj is None or not entries:
            return None
        try:
            acc: Optional[np.ndarray] = None
            max_shape: Optional[Tuple[int, int]] = None

            for e in entries:
                det_chs = e.get('detector_chs') or None
                mtr = e.get('micro_time_range') or None

                params = {
                    'tttr_data': tttr_obj,
                    'fill': True
                }
                # Channels for this entry
                if det_chs:
                    try:
                        params['channels'] = list(det_chs)
                    except Exception:
                        params['channels'] = det_chs
                # Micro time gating for this entry
                if mtr is not None and isinstance(mtr, (list, tuple)) and len(mtr) == 2:
                    try:
                        a, b = int(mtr[0]), int(mtr[1])
                        params['micro_time_ranges'] = [(a, b)]
                    except Exception:
                        pass

                # Create CLSM image for this gated subset
                clsm = tttrlib.CLSMImage(**params)
                img = np.array(clsm.intensity)

                # Reduce to 2D if needed (sum across frames)
                if img.ndim == 3:
                    tile = img.sum(axis=0)
                elif img.ndim == 2:
                    tile = img
                else:
                    continue

                tile = tile.astype(float, copy=False)

                if acc is None:
                    acc = tile.copy()
                    max_shape = acc.shape
                else:
                    # Ensure shapes match by zero-padding to the max dimensions
                    sx, sy = acc.shape
                    tx, ty = tile.shape
                    new_sx = max(sx, tx)
                    new_sy = max(sy, ty)
                    if (new_sx, new_sy) != (sx, sy):
                        tmp = np.zeros((new_sx, new_sy), dtype=float)
                        tmp[:sx, :sy] = acc
                        acc = tmp
                    if (new_sx, new_sy) != (tx, ty):
                        tmp2 = np.zeros((new_sx, new_sy), dtype=float)
                        tmp2[:tx, :ty] = tile
                        tile = tmp2
                    acc += tile

            return acc if acc is not None else None
        except Exception as ex:
            logging.exception(f"TTTRImageBrowser: Failed to compute combo image: {ex}")
            return None

    def _render_mosaic_array(self, path: pathlib.Path):
        """Render mosaic into a numpy uint8 image and return with label data.
        Returns: (mosaic_uint8, labels, cols, rows, tile_w_scaled, tile_h_scaled) or None.
        """
        # Acquire channels definition from DetectorWizard
        settings = self.setup_settings or {}
        reading = settings.get('tttr_reading', {}) if isinstance(settings, dict) else {}
        reading_routine = reading.get('file_type') or None
        try:
            channels_map = self.detector_page.channels()
            channels_map = self._group_channels_by_detector(channels_map)
            logging.debug(f"TTTRImageBrowser: Grouped channels by detector: {list(channels_map.keys())}")
        except Exception as e:
            logging.exception(f"TTTRImageBrowser: Failed to get channels from DetectorWizard, fallback used: {e}")
            channels_map = {"Image": [{'window_range': (None, None), 'detector_chs': [], 'micro_time_range': (None, None)}]}
        tttr_obj = tttrlib.TTTR(str(path), reading_routine)
        # Early bail-out if this file has no CLSM header/support
        try:
            if not self._is_clsm_compatible(tttr_obj):
                return None
        except Exception:
            return None
        names = list(channels_map.keys())
        entries_list = [channels_map[n] for n in names]
        images: List[Tuple[str, Optional[np.ndarray]]] = []
        labels: List[str] = []
        for name, entries in zip(names, entries_list):
            stack = self._get_combo_stack(tttr_obj, path, entries, reading_routine)
            if stack is None:
                img2d = None
            else:
                try:
                    img2d = stack.sum(axis=0)
                except Exception:
                    img2d = None
            images.append((name, img2d))
            # Label construction (detector name + channels + microtime ranges)
            try:
                det_chs = None
                for e in entries:
                    chs = e.get('detector_chs') if isinstance(e, dict) else None
                    if chs:
                        det_chs = list({int(c) for c in chs}); det_chs.sort(); break
                ch_txt = ",".join(map(str, det_chs)) if det_chs else ""
                mtr_list = []
                for e in entries:
                    mtr = e.get('micro_time_range') if isinstance(e, dict) else None
                    if isinstance(mtr, (list, tuple)) and len(mtr)==2 and mtr[0] is not None and mtr[1] is not None:
                        try:
                            a,b = int(mtr[0]), int(mtr[1]); mtr_list.append((a,b))
                        except Exception:
                            pass
                seen=set(); uniq_mtrs=[]
                for ab in mtr_list:
                    if ab not in seen:
                        seen.add(ab); uniq_mtrs.append(ab)
                mtr_txt = ";".join(f"{a}-{b}" for (a,b) in uniq_mtrs)
                parts = [str(name)]
                if ch_txt: parts.append(f"ch: {ch_txt}")
                if mtr_txt: parts.append(f"mt: {mtr_txt}")
                label_str = "  |  ".join(parts)
            except Exception:
                label_str = str(name)
            labels.append(label_str)
        non_none = [im for _, im in images if im is not None]
        if not non_none:
            return None
        k = len(images)
        cols = int(np.ceil(np.sqrt(k)))
        rows = int(np.ceil(k / cols))
        shapes = [(im.shape if im is not None else (1,1)) for _, im in images]
        max_nx = max(s[0] for s in shapes)
        max_ny = max(s[1] for s in shapes)
        max_side = 512
        def _to_uint8(arr: np.ndarray) -> np.ndarray:
            a = arr.astype(float)
            if not np.isfinite(a).any():
                a = np.zeros_like(a)
            mn, mx = np.nanmin(a), np.nanmax(a)
            if mx <= mn:
                return np.zeros(a.shape, dtype=np.uint8)
            a = (a - mn) / (mx - mn)
            a = (a * 255.0).clip(0, 255).astype(np.uint8)
            return a
        def _resize_nn(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
            if a.shape == shape:
                return a
            sx = shape[0] / a.shape[0]
            sy = shape[1] / a.shape[1]
            xi = (np.floor(np.arange(shape[0]) / sx)).astype(int)
            yi = (np.floor(np.arange(shape[1]) / sy)).astype(int)
            xi = np.clip(xi, 0, a.shape[0]-1)
            yi = np.clip(yi, 0, a.shape[1]-1)
            return a[np.ix_(xi, yi)]
        tile_h, tile_w = max_nx, max_ny
        mosaic_h = rows * tile_h
        mosaic_w = cols * tile_w
        scale = 1.0
        if max(mosaic_h, mosaic_w) > max_side:
            scale = max_side / float(max(mosaic_h, mosaic_w))
        tile_h_scaled = max(1, int(round(tile_h * scale)))
        tile_w_scaled = max(1, int(round(tile_w * scale)))
        mosaic = np.zeros((rows * tile_h_scaled, cols * tile_w_scaled), dtype=np.uint8)
        for idx, (_, img) in enumerate(images):
            r = idx // cols
            c = idx % cols
            y0 = r * tile_h_scaled
            x0 = c * tile_w_scaled
            if img is None:
                tile = np.zeros((tile_h, tile_w), dtype=np.uint8)
            else:
                tile = _to_uint8(img)
                tile = _resize_nn(tile, (tile_h, tile_w))
                if scale != 1.0:
                    tile = _resize_nn(tile, (tile_h_scaled, tile_w_scaled))
            if tile.shape != (tile_h_scaled, tile_w_scaled):
                tile = _resize_nn(tile, (tile_h_scaled, tile_w_scaled))
            mosaic[y0:y0+tile_h_scaled, x0:x0+tile_w_scaled] = tile
        return mosaic, labels, cols, rows, tile_w_scaled, tile_h_scaled

    def _render_mosaic_pixmap(self, path: pathlib.Path) -> Optional[QPixmap]:
        # Acquire channels definition from DetectorWizard
        settings = self.setup_settings or {}
        reading = settings.get('tttr_reading', {}) if isinstance(settings, dict) else {}
        reading_routine = reading.get('file_type') or None
        # channels_map: name -> list of entries
        try:
            channels_map = self.detector_page.channels()
            channels_map = self._group_channels_by_detector(channels_map)
            logging.debug(f"TTTRImageBrowser: Grouped channels by detector: {list(channels_map.keys())}")
        except Exception as e:
            logging.exception(f"TTTRImageBrowser: Failed to get channels from DetectorWizard, fallback used: {e}")
            # Fallback: build trivial single image without gating
            channels_map = {"Image": [{'window_range': (None, None), 'detector_chs': [], 'micro_time_range': (None, None)}]}
        # Read TTTR file
        tttr_obj = tttrlib.TTTR(str(path), reading_routine)
        # Early bail-out if this file has no CLSM header/support
        try:
            if not self._is_clsm_compatible(tttr_obj):
                return None
        except Exception:
            return None

        # Compute each image
        names = list(channels_map.keys())
        entries_list = [channels_map[n] for n in names]
        images: List[Tuple[str, Optional[np.ndarray]]] = []
        labels: List[str] = []
        for name, entries in zip(names, entries_list):
            # Use cached per-entry stacks and then reduce to 2D by summation over frames
            stack = self._get_combo_stack(tttr_obj, path, entries, reading_routine)
            if stack is None:
                img2d = None
            else:
                try:
                    img2d = stack.sum(axis=0)
                except Exception:
                    img2d = None
            images.append((name, img2d))
            # Build label: detector name (via combo name), channels, microtime ranges
            try:
                # Channels: collect from first non-empty entry
                det_chs = None
                for e in entries:
                    chs = e.get('detector_chs') if isinstance(e, dict) else None
                    if chs:
                        det_chs = list({int(c) for c in chs})
                        det_chs.sort()
                        break
                ch_txt = ",".join(map(str, det_chs)) if det_chs else ""
                # Microtime ranges: aggregate unique ranges from entries
                mtr_list = []
                for e in entries:
                    mtr = e.get('micro_time_range') if isinstance(e, dict) else None
                    if isinstance(mtr, (list, tuple)) and len(mtr) == 2 and mtr[0] is not None and mtr[1] is not None:
                        try:
                            a, b = int(mtr[0]), int(mtr[1])
                            mtr_list.append((a, b))
                        except Exception:
                            pass
                # Deduplicate while preserving order
                seen = set()
                uniq_mtrs = []
                for ab in mtr_list:
                    if ab not in seen:
                        seen.add(ab)
                        uniq_mtrs.append(ab)
                mtr_txt = ";".join(f"{a}-{b}" for (a, b) in uniq_mtrs)
                # Combine into label string
                base = str(name)
                parts = [base]
                if ch_txt:
                    parts.append(f"ch: {ch_txt}")
                if mtr_txt:
                    parts.append(f"mt: {mtr_txt}")
                label_str = "  |  ".join(parts)
            except Exception:
                label_str = str(name)
            labels.append(label_str)

        # Determine grid size (roughly square)
        non_none = [im for _, im in images if im is not None]
        if not non_none:
            return None
        k = len(images)
        cols = int(np.ceil(np.sqrt(k)))
        rows = int(np.ceil(k / cols))

        # Normalize and assemble mosaic
        # First, determine target size (max x,y among all)
        shapes = [(im.shape if im is not None else (1,1)) for _, im in images]
        max_nx = max(s[0] for s in shapes)
        max_ny = max(s[1] for s in shapes)
        # Scale factor to limit overall size
        max_side = int(self.max_side_spin.value())
        # We'll resize each tile to the same size via simple numpy zoom (nearest)
        def _to_uint8(arr: np.ndarray) -> np.ndarray:
            a = arr.astype(float)
            if not np.isfinite(a).any():
                a = np.zeros_like(a)
            mn, mx = np.nanmin(a), np.nanmax(a)
            if mx <= mn:
                return np.zeros(a.shape, dtype=np.uint8)
            a = (a - mn) / (mx - mn)
            a = (a * 255.0).clip(0, 255).astype(np.uint8)
            return a
        
        # Simple nearest-neighbor resize utility
        def _resize_nn(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
            if a.shape == shape:
                return a
            sx = shape[0] / a.shape[0]
            sy = shape[1] / a.shape[1]
            xi = (np.floor(np.arange(shape[0]) / sx)).astype(int)
            yi = (np.floor(np.arange(shape[1]) / sy)).astype(int)
            xi = np.clip(xi, 0, a.shape[0]-1)
            yi = np.clip(yi, 0, a.shape[1]-1)
            return a[np.ix_(xi, yi)]

        tile_h, tile_w = max_nx, max_ny
        # Adjust tile size to respect max_side
        # Compute mosaic size in pixels then scale factor
        mosaic_h = rows * tile_h
        mosaic_w = cols * tile_w
        scale = 1.0
        if max(mosaic_h, mosaic_w) > max_side:
            scale = max_side / float(max(mosaic_h, mosaic_w))
        tile_h_scaled = max(1, int(round(tile_h * scale)))
        tile_w_scaled = max(1, int(round(tile_w * scale)))

        # Create grayscale mosaic first
        mosaic = np.zeros((rows * tile_h_scaled, cols * tile_w_scaled), dtype=np.uint8)
        for idx, (_, img) in enumerate(images):
            r = idx // cols
            c = idx % cols
            y0 = r * tile_h_scaled
            x0 = c * tile_w_scaled
            if img is None:
                tile = np.zeros((tile_h, tile_w), dtype=np.uint8)
            else:
                tile = _to_uint8(img)
                tile = _resize_nn(tile, (tile_h, tile_w))
                if scale != 1.0:
                    tile = _resize_nn(tile, (tile_h_scaled, tile_w_scaled))
            if tile.shape != (tile_h_scaled, tile_w_scaled):
                tile = _resize_nn(tile, (tile_h_scaled, tile_w_scaled))
            mosaic[y0:y0+tile_h_scaled, x0:x0+tile_w_scaled] = tile

        # Convert to colored QPixmap using magma LUT
        try:
            from PyQt5.QtGui import QImage
            h, w = mosaic.shape
            lut = self._get_magma_lut(256)
            if lut is not None:
                rgb = lut[mosaic]
                # Ensure contiguous memory and create QImage in RGB888
                if not rgb.flags['C_CONTIGUOUS']:
                    rgb = np.ascontiguousarray(rgb)
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg.copy())
            else:
                # Fallback to grayscale if LUT unavailable
                qimg = QImage(mosaic.data, w, h, w, QImage.Format_Grayscale8)
                pix = QPixmap.fromImage(qimg.copy())
        except Exception:
            pix = None
        # Overlay labels on each tile
        try:
            if pix is not None and not pix.isNull():
                painter = QPainter(pix)
                try:
                    painter.setRenderHint(QPainter.Antialiasing, True)
                    # Choose font size relative to tile height
                    font = painter.font()
                    font.setPointSize(max(7, int(tile_h_scaled * 0.08)))
                    painter.setFont(font)
                    fm = painter.fontMetrics()
                    text_h = fm.height() + 4
                    for idx, label in enumerate(labels):
                        r = idx // cols
                        c = idx % cols
                        y0 = r * tile_h_scaled
                        x0 = c * tile_w_scaled
                        # Background rectangle for readability
                        painter.fillRect(x0, y0, tile_w_scaled, text_h, QColor(0, 0, 0, 160))
                        painter.setPen(QColor(255, 255, 255))
                        # Clip text within tile width
                        elided = fm.elidedText(label, Qt.ElideRight, tile_w_scaled - 6)
                        painter.drawText(x0 + 3, y0 + fm.ascent() + 2, elided)
                finally:
                    painter.end()
        except Exception:
            pass
        return pix

    # --- Drag-and-drop support (same pattern as TraceBrowser) ---
    def eventFilter(self, obj, event):
        try:
            if obj is self.table and event is not None:
                et = event.type()
                if et in (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop):
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

    def _precompute_all_images(self):
        """
        Precompute and cache images for all files in the current table, with a progress bar.
        Now also builds the final mosaics + labels for each file and stores them in memory
        for instant display when switching files.
        """
        if self.current_folder is None or tttrlib is None:
            return

        # Ensure mosaic cache exists
        self._mosaic_cache: Dict[pathlib.Path, Tuple[np.ndarray, List[str], int, int]] = {}

        # Collect all paths from the table (all rows, regardless of filter visibility)
        paths: List[pathlib.Path] = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is None:
                continue
            p_str = item.data(Qt.UserRole)
            if p_str:
                paths.append(pathlib.Path(p_str))
        if not paths:
            return

        # Prepare reading routine and channel definitions once
        settings = self.setup_settings or {}
        reading = settings.get('tttr_reading', {}) if isinstance(settings, dict) else {}
        reading_routine = reading.get('file_type') or None
        try:
            base_channels_map = self.detector_page.channels()
            base_channels_map = self._group_channels_by_detector(base_channels_map)
        except Exception:
            base_channels_map = {
                "Image": [{
                    'window_range': (None, None),
                    'detector_chs': [],
                    'micro_time_range': (None, None)
                }]
            }

        # Determine files that need processing: load mosaics from cache when present
        files_to_process: List[pathlib.Path] = []
        for p in paths:
            loaded = self._load_mosaic_cache(p, base_channels_map, reading_routine)
            if loaded is not None:
                mosaic, labels, cols, rows, tile_w, tile_h = loaded
                self._mosaic_cache[p] = (mosaic, labels, cols, rows)
            else:
                files_to_process.append(p)
        # If nothing to do, skip dialog entirely
        if not files_to_process:
            return

        # Show progress dialog only for files we are going to process
        dlg = QProgressDialog("Precomputing images...", "Cancel", 0, len(files_to_process), self)
        dlg.setWindowTitle("Precomputing images")
        dlg.setAutoClose(True)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)

        try:
            for i, p in enumerate(files_to_process):
                if dlg is not None:
                    try:
                        dlg.setValue(i)
                        dlg.setLabelText(f"Processing {p.name} ({i + 1}/{len(files_to_process)})")
                    except Exception:
                        pass
                QApplication.processEvents()
                if dlg is not None and dlg.wasCanceled():
                    break

                try:
                    # Read TTTR file once
                    tttr_obj = tttrlib.TTTR(str(p), reading_routine)

                    # Skip if not CLSM compatible
                    if not self._is_clsm_compatible(tttr_obj):
                        self._remove_path_from_table(p)
                        continue

                    # Ensure per-entry stack cache exists
                    for _, entries in (base_channels_map or {}).items():
                        for e in entries or []:
                            det_chs = e.get('detector_chs') or None
                            mtr = e.get('micro_time_range') or None
                            cache_path = self._entry_cache_file(p, det_chs, mtr, reading_routine)
                            if not cache_path.exists():
                                try:
                                    _ = self._compute_entry_stack_cached(
                                        tttr_obj, p, det_chs, mtr, reading_routine
                                    )
                                except Exception:
                                    pass

                    # Build mosaic + labels from cached stacks
                    images = []
                    labels = []
                    for combo_name, entries in base_channels_map.items():
                        combo_stacks = []
                        for e in entries:
                            det_chs = e.get('detector_chs') or None
                            mtr = e.get('micro_time_range') or None
                            cache_path = self._entry_cache_file(p, det_chs, mtr, reading_routine)
                            if cache_path.exists():
                                try:
                                    with np.load(str(cache_path)) as z:
                                        combo_stacks.append(z['stack'])
                                except Exception:
                                    pass
                        arr = self._sum_stacks_with_padding(combo_stacks)
                        img2d = arr.sum(axis=0) if arr is not None else None
                        images.append(img2d)

                        # Build label
                        det_chs_txt = ""
                        if entries and entries[0].get('detector_chs'):
                            chs = sorted(set(map(int, entries[0]['detector_chs'])))
                            det_chs_txt = f"ch: {','.join(map(str, chs))}"
                        labels.append("  |  ".join(filter(None, [combo_name, det_chs_txt])))

                    non_none = [im for im in images if im is not None]
                    if not non_none:
                        self._remove_path_from_table(p)
                        continue

                    # Build mosaic array
                    cols = int(np.ceil(np.sqrt(len(images))))
                    rows = int(np.ceil(len(images) / cols))
                    max_x = max(s.shape[0] for s in non_none)
                    max_y = max(s.shape[1] for s in non_none)
                    scale = min(1.0, 512 / max(max_x * rows, max_y * cols))
                    tile_h = max(1, int(round(max_x * scale)))
                    tile_w = max(1, int(round(max_y * scale)))
                    mosaic = np.zeros((rows * tile_h, cols * tile_w), np.uint8)

                    def to_u8(a):
                        a = a.astype(float)
                        mn, mx = np.nanmin(a), np.nanmax(a)
                        if mx <= mn:
                            return np.zeros_like(a, dtype=np.uint8)
                        return np.clip((a - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)

                    def resize_nn(a, shape):
                        if a.shape == shape:
                            return a
                        sx = shape[0] / a.shape[0]
                        sy = shape[1] / a.shape[1]
                        xi = np.clip((np.floor(np.arange(shape[0]) / sx)).astype(int), 0, a.shape[0] - 1)
                        yi = np.clip((np.floor(np.arange(shape[1]) / sy)).astype(int), 0, a.shape[1] - 1)
                        return a[np.ix_(xi, yi)]

                    for idx, img in enumerate(images):
                        r, c = divmod(idx, cols)
                        y0, x0 = r * tile_h, c * tile_w
                        tile = np.zeros((tile_h, tile_w), np.uint8) if img is None else resize_nn(to_u8(img),
                                                                                                  (tile_h, tile_w))
                        mosaic[y0:y0 + tile_h, x0:x0 + tile_w] = tile

                    # Store in memory and save to disk cache
                    self._mosaic_cache[p] = (mosaic, labels, cols, rows)
                    try:
                        self._save_mosaic_cache(p, base_channels_map, reading_routine, mosaic, labels, cols, rows, tile_w, tile_h)
                    except Exception:
                        pass

                except Exception as e:
                    logging.debug(f"TTTRImageBrowser: Precompute failed for {p}: {e}")

        finally:
            if dlg is not None:
                dlg.setValue(len(files_to_process))
                dlg.close()

    def _remove_path_from_table(self, path: pathlib.Path):
        """Remove the row corresponding to path from the table and update selection.
        If no rows remain, clear preview and annotation."""
        try:
            target_row = -1
            for r in range(self.table.rowCount()):
                item = self.table.item(r, 0)
                if item is None:
                    continue
                p_str = item.data(Qt.UserRole)
                if not p_str:
                    continue
                if pathlib.Path(p_str) == path:
                    target_row = r
                    break
            if target_row >= 0:
                self.table.removeRow(target_row)
                # Select next visible row if any
                for r in range(self.table.rowCount()):
                    if not self.table.isRowHidden(r):
                        try:
                            self.table.selectRow(r)
                        except Exception:
                            pass
                        return
                # If nothing left, clear
                self._clear_preview_and_annotation()
        except Exception:
            # Best-effort removal; ignore errors
            pass

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

    # --- Export ---
    def _on_export(self):
        # Export all files currently visible in the list (respecting active filter/sort)
        paths: List[pathlib.Path] = []
        for r in range(self.table.rowCount()):
            if self.table.isRowHidden(r):
                continue
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
                from shutil import copy2
                copy2(str(p), str(out / p.name))
                copied += 1
            except Exception as e:
                logging.warning(f"TTTRImageBrowser: Failed to copy {p} to {out}: {e}")
        logging.info(f"TTTRImageBrowser: Exported {copied}/{len(paths)} files to {out}")

    def _on_save_tiff(self):
        paths = self._selected_paths()
        if not paths:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select destination folder for TIFF stacks")
        if not out_dir:
            return
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = 0
        for p in paths:
            try:
                # Prepare channels map and reading routine
                settings = self.setup_settings or {}
                reading = settings.get('tttr_reading', {}) if isinstance(settings, dict) else {}
                reading_routine = reading.get('file_type') or None
                try:
                    channels_map = self.detector_page.channels()
                    channels_map = self._group_channels_by_detector(channels_map)
                except Exception:
                    channels_map = {"Image": [{'window_range': (None, None), 'detector_chs': [], 'micro_time_range': (None, None)}]}
                # Read TTTR file once
                tttr_obj = tttrlib.TTTR(str(p), reading_routine)
                for combo_name, entries in channels_map.items():
                    stack = self._get_combo_stack(tttr_obj, p, entries, reading_routine)
                    if stack is None:
                        continue
                    # Save as TIFF stack: prefer tifffile, fall back to imageio
                    safe_name = ''.join(ch if ch.isalnum() or ch in ('-','_') else '_' for ch in str(combo_name))
                    fname = f"{p.stem}_{safe_name}.tiff"
                    dest = out / fname
                    ok = False
                    try:
                        import tifffile as tiff
                        tiff.imwrite(str(dest), stack.astype(np.uint32, copy=False))
                        ok = True
                    except Exception:
                        try:
                            import imageio
                            imageio.mimwrite(str(dest), [frame for frame in stack.astype(np.uint16, copy=False)], format='TIFF')
                            ok = True
                        except Exception:
                            pass
                    if ok:
                        saved += 1
            except Exception as e:
                logging.warning(f"TTTRImageBrowser: Failed to save TIFFs for {p}: {e}")
        logging.info(f"TTTRImageBrowser: Saved {saved} TIFF file(s) to {out}")

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
        if Document is None:
            try:
                QMessageBox.warning(self, "DOCX Export", "python-docx is not installed. Please install 'python-docx' to enable DOCX export.")
            except Exception:
                pass
            return
        folder = self.current_folder
        if folder is None and paths:
            folder = paths[0].parent
        if folder is None:
            try:
                QMessageBox.critical(self, "DOCX Export", "No folder context available to determine DOCX save location.")
            except Exception:
                pass
            return
        docx_name = (folder.name or "images") + ".docx"
        parent_dir = folder.parent if folder.parent != folder else pathlib.Path(".")
        save_path = parent_dir / docx_name
        try:
            import tempfile
            tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="image_export_"))
        except Exception:
            tmpdir = self.current_folder or pathlib.Path(".")
        doc = Document()
        try:
            doc.add_heading("TTTR Image Browser Export", level=1)
        except Exception:
            pass
        current_selected = self._selected_paths()
        for p in paths:
            # Try to render as QPixmap first (GUI path)
            pix = None
            try:
                pix = self._render_mosaic_pixmap(p)
            except Exception:
                pix = None
            img_path = None
            # Attempt to save QPixmap
            if pix is not None and not pix.isNull():
                try:
                    img_path = tmpdir / f"{p.stem}.png"
                    if not pix.save(str(img_path), "PNG"):
                        img_path = None
                except Exception:
                    img_path = None
            # Fallback: render via numpy array and save using Pillow (or imageio)
            if img_path is None:
                try:
                    mosaic_pack = self._render_mosaic_array(p)
                except Exception:
                    mosaic_pack = None
                if mosaic_pack is not None:
                    try:
                        mosaic_u8, labels, cols, rows, tile_w_scaled, tile_h_scaled = mosaic_pack
                        lut = self._get_magma_lut(256)
                        if lut is not None:
                            rgb = lut[mosaic_u8]
                        else:
                            # grayscale to RGB
                            rgb = np.stack([mosaic_u8]*3, axis=-1)
                        # Optionally overlay labels with Pillow
                        try:
                            from PIL import Image, ImageDraw, ImageFont
                            img = Image.fromarray(rgb, mode='RGB')
                            draw = ImageDraw.Draw(img)
                            try:
                                font = ImageFont.load_default()
                            except Exception:
                                font = None
                            text_h = (font.getsize("Ag")[1] if font else 10) + 4
                            for idx, label in enumerate(labels):
                                r = idx // cols
                                c = idx % cols
                                y0 = r * tile_h_scaled
                                x0 = c * tile_w_scaled
                                # Background rectangle
                                draw.rectangle([x0, y0, x0 + tile_w_scaled, y0 + text_h], fill=(0,0,0,160))
                                # Draw text
                                draw.text((x0 + 3, y0 + 2), str(label), fill=(255,255,255), font=font)
                        except Exception:
                            # If PIL not available for drawing, still save the RGB image
                            from PIL import Image  # may still exist even if ImageDraw fails
                            img = Image.fromarray(rgb, mode='RGB')
                        # Save to temp path
                        img_path = tmpdir / f"{p.stem}.png"
                        try:
                            img.save(str(img_path), format='PNG')
                        except Exception:
                            # Final fallback using imageio
                            try:
                                import imageio
                                imageio.imwrite(str(img_path), rgb)
                            except Exception:
                                img_path = None
                    except Exception:
                        img_path = None
            # Compose doc content
            rec = self.meta.get(p.name, {})
            rating = int(rec.get("rating", 0))
            annotation = rec.get("annotation", "")
            folder_text = str(p.parent)
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
            if img_path and img_path.exists():
                try:
                    if Inches is not None:
                        doc.add_picture(str(img_path), width=Inches(6))
                    else:
                        doc.add_picture(str(img_path))
                except Exception as ex:
                    logging.warning(f"TTTRImageBrowser: Failed to add picture for {p}: {ex}")
            else:
                logging.warning(f"TTTRImageBrowser: No image generated for DOCX for {p}")
            try:
                doc.add_paragraph("")  # spacing
            except Exception:
                pass
        try:
            doc.save(str(save_path))
            logging.info(f"TTTRImageBrowser: DOCX exported to {save_path}")
            try:
                QMessageBox.information(self, "DOCX Export", f"Saved: {save_path}")
            except Exception:
                pass
        except Exception as e:
            logging.exception(f"TTTRImageBrowser: Failed to save DOCX {save_path}: {e}")
            try:
                QMessageBox.critical(self, "DOCX Export", f"Failed to save DOCX: {e}")
            except Exception:
                pass
        # Cleanup temp images
        try:
            if tmpdir and tmpdir.exists() and tmpdir.name.startswith("image_export_"):
                import shutil as _sh
                _sh.rmtree(str(tmpdir), ignore_errors=True)
        except Exception:
            pass
        # Restore selection preview
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
    w = TTTRImageBrowser()
    w.show()
    sys.exit(app.exec_())

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed by the Plugin Manager
if __name__ == "plugin":
    window = TTTRImageBrowser()
    window.show()
