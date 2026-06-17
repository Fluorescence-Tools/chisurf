"""
TTTR Settings Generator

This plugin provides a graphical interface for creating TTTR settings JSON files
containing LUTs and time shifts per channel for TTTR data correction.

## Usage

1. Launch the plugin from ChiSurf's Plugins menu under "TTTR:Settings Generator"
2. Select TTTR reading routine if needed (default is Auto)
3. Drag and drop TTTR files or use "Add files" button
4. Load LUT files (drag and drop onto LUT list or use "Load LUT" button)
5. Check/uncheck channels in the channel list to show/hide them in the plot
6. Select a channel to adjust its time shift
7. Assign LUTs to selected channel or all channels
8. Preview the JSON settings
9. Save the settings.tttr.json file

The generated settings files can be used for correcting TTTR data in other
analysis tools.
"""

import sys
import os
import json
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import tttrlib

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

VALID_EXTS = {".spc", ".ht3", ".ptu", ".phu", ".photonhdf5"}
EPS = 1e-12  # for log10 plots

# Define the plugin name - this will appear in the Plugins menu
name = "TTTR:Create LUT Settings"
menu_hidden = True
deprecated = True
deprecation_message = "Use TTTR:LUT Tools instead. This legacy plugin is hidden from the menu."

# Helper functions from original script
def load_lut_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".txt":
        arr = np.loadtxt(path)
    elif ext == ".npz":
        data = np.load(path)
        keys = list(data.keys())
        if not keys:
            raise ValueError(".npz file contains no arrays.")
        arr = data[keys[0]]
    else:
        raise ValueError(f"Unsupported LUT file type: {ext}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected 1D LUT array.")
    return arr

def nice_pen(color, width=2):
    return pg.mkPen(color=color, width=width)

COLOR_CYCLE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]

def json_safe(obj):
    if isinstance(obj, (np.generic,)):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    return obj

def f32(a):
    return np.asarray(a, dtype=np.float32)

def logify(y):
    return f32(np.log10(np.maximum(y, 0) + EPS))

# File list widget with drag and drop
class FileListWidget(QtWidgets.QListWidget):
    filesChanged = QtCore.Signal()  # Signal emitted when files change

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)
        self.setMinimumHeight(60)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e: QtGui.QDragMoveEvent):
        e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        if e.mimeData().hasUrls():
            paths = []
            for url in e.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isfile(path):
                    ext = os.path.splitext(path)[1].lower()
                    if ext in VALID_EXTS:
                        paths.append(path)
            if paths:
                self.add_files(paths)
        e.acceptProposedAction()

    def add_files(self, paths):
        """Add files to the list, avoiding duplicates."""
        existing = {self.item(i).text() for i in range(self.count())}
        for path in paths:
            if path not in existing:
                self.addItem(path)
        self.filesChanged.emit()

    def current_paths(self):
        """Get list of all file paths."""
        return [self.item(i).text() for i in range(self.count())]

    def clear_files(self):
        """Clear all files."""
        self.clear()
        self.filesChanged.emit()

# LUT list widget with drag and drop
class LUTListWidget(QtWidgets.QListWidget):
    lutDropped = QtCore.Signal(str)  # Signal emitted with the dropped file path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setMaximumHeight(80)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            # Check if any URL is a valid LUT file
            for url in e.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in ['.npy', '.npz', '.txt']:
                        e.acceptProposedAction()
                        return
        e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent):
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isfile(path):
                    ext = os.path.splitext(path)[1].lower()
                    if ext in ['.npy', '.npz', '.txt']:
                        # Emit signal with the path
                        self.lutDropped.emit(path)
                        break  # Only handle the first valid LUT file
        e.acceptProposedAction()

def _hist_time_safe(tttr_obj, ch):
    """
    Robustly obtain (x=time, y=hist) from tttrlib regardless of return order.
    Some builds return (hist, time), others (time, hist). We normalize here.
    """
    a, b = tttr_obj.get_microtime_histogram(channels=[int(ch)])
    a = np.asarray(a)
    b = np.asarray(b)
    # Heuristic: time axis is usually strictly increasing small ints/floats; hist is nonnegative counts
    def looks_like_time(v):
        if v.ndim != 1 or v.size == 0:
            return False
        if not np.all(np.isfinite(v)):
            return False
        dif = np.diff(v)
        return np.all(dif >= 0) and (np.count_nonzero(dif) > 0)
    def looks_like_hist(v):
        return v.ndim == 1 and v.size > 0 and np.all(np.isfinite(v)) and np.all(v >= 0)
    if looks_like_time(a) and looks_like_hist(b):
        x, y = a, b
    elif looks_like_time(b) and looks_like_hist(a):
        x, y = b, a
    else:
        # fallback: assume (hist, time) like in most examples
        y, x = a, b
    # Ensure 1D and same length
    x = np.ravel(x)
    y = np.ravel(y)
    n = min(len(x), len(y))
    return f32(x[:n]), f32(y[:n])

# Data class from original script
class TTTRBundle:
    """Load & combine files; precompute/caches (linear + log10)."""
    def __init__(self, paths, container_type=None):
        if not paths:
            raise FileNotFoundError("No TTTR files.")
        self.paths = list(paths)
        self.container_type = container_type
        self.tt = self._load_combined(self.paths, self.container_type)
        self._unc_lin = {}   # ch -> (x_f32, y_f32)
        self._unc_log = {}   # ch -> (x_f32, log10_y_f32)
        self._cor_lin = {}   # ch -> (x_f32, y_f32)
        self._cor_log = {}   # ch -> (x_f32, log10_y_f32)
        self._lut_sig = None

    @classmethod
    def _load_combined(cls, paths, container_type=None):
        it = iter(paths)
        tt = tttrlib.TTTR(next(it), container_type=container_type)
        for p in it:
            tt += tttrlib.TTTR(p, container_type=container_type)
        return tt

    def used_channels(self):
        # force Python ints to avoid numpy types in JSON later
        return [int(ch) for ch in self.tt.get_used_routing_channels()]

    def precompute_uncorrected(self):
        """Compute ALL uncorrected histograms ONCE after load (both linear & log10)."""
        self._unc_lin.clear()
        self._unc_log.clear()
        chans = self.used_channels()
        for ch in chans:
            x, y = _hist_time_safe(self.tt, ch)
            self._unc_lin[int(ch)] = (x, y)
            self._unc_log[int(ch)] = (x, logify(y))

    def get_unc(self, ch, want_log):
        return (self._unc_log if want_log else self._unc_lin)[int(ch)]

    def _lut_signature(self, channel_luts):
        keys = tuple(sorted(int(k) for k in (channel_luts or {}).keys()))
        lens = tuple((int(len(channel_luts[k])) for k in keys))
        return ("lut", keys, lens)

    def ensure_corrected_cache(self, channel_luts):
        """Rebuild corrected caches only if LUT signature changed (both linear & log10)."""
        sig = self._lut_signature(channel_luts)
        if sig == self._lut_sig:
            return
        self._cor_lin.clear()
        self._cor_log.clear()
        tt_corr = self._load_combined(self.paths)
        if channel_luts:
            tt_corr.apply_channel_luts(channel_luts, {})   # shifts handled in view
            tt_corr.apply_luts_and_shifts(-1, True)
        for ch in self.used_channels():
            x, y = _hist_time_safe(tt_corr, ch)
            self._cor_lin[int(ch)] = (x, y)
            self._cor_log[int(ch)] = (x, logify(y))
        self._lut_sig = sig

    def get_cor(self, ch, want_log, channel_luts):
        self.ensure_corrected_cache(channel_luts)
        return (self._cor_log if want_log else self._cor_lin)[int(ch)]

# LUT list widget with drag and drop
class LUTListWidget(QtWidgets.QListWidget):
    lutDropped = QtCore.Signal(str)  # Signal emitted with the dropped file path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setMaximumHeight(80)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            # Check if any URL is a valid LUT file
            for url in e.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in ['.npy', '.npz', '.txt']:
                        e.acceptProposedAction()
                        return
        e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent):
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isfile(path):
                    ext = os.path.splitext(path)[1].lower()
                    if ext in ['.npy', '.npz', '.txt']:
                        # Emit signal with the path
                        self.lutDropped.emit(path)
                        break  # Only handle the first valid LUT file
        e.acceptProposedAction()

# Data class from original script
class JsonPreviewDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings JSON Preview")
        self.resize(700, 600)
        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close, alignment=QtCore.Qt.AlignRight)

    def set_json(self, txt: str):
        self.text.setPlainText(txt)

# README dialog class
class ReadmeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TTTR Settings Generator — Help")
        self.resize(800, 600)
        layout = QtWidgets.QVBoxLayout(self)
        
        # Read README.md file
        readme_path = os.path.join(os.path.dirname(__file__), "README.md")
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_text = f.read()
        except Exception as e:
            readme_text = f"Error loading README.md: {e}"
        
        self.text = QtWidgets.QTextEdit()
        self.text.setPlainText(readme_text)
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close, alignment=QtCore.Qt.AlignRight)

# Channel Check List from original script
class ChannelCheckList(QtWidgets.QWidget):
    toggled = QtCore.Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self._checks = {}
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0,0,0,0)
        v.addStretch(1)

    def clear(self):
        for cb in self._checks.values():
            cb.setParent(None)
            cb.deleteLater()
        self._checks.clear()

    def set_channels(self, channels):
        self.clear()
        lay = self.layout()
        for ch in sorted(channels):
            cb = QtWidgets.QCheckBox(f"Show Channel {ch}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.toggled.emit)
            lay.insertWidget(lay.count()-1, cb)
            self._checks[int(ch)] = cb

    def selected_channels(self):
        return [ch for ch, cb in self._checks.items() if cb.isChecked()]

@persist_plugin_state("tttr_settings_generator")
class TTTRSettingsGenerator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TTTR Settings Generator — LUTs & Shifts")
        self.resize(800, 600)

        # State
        self.bundle = None
        self.channel_luts = {}    # ch -> np.ndarray
        self.channel_shifts = {}  # ch -> int
        self.loaded_luts = {}     # filename -> np.ndarray
        self.reading_routine = None  # container type for TTTR reading

        # Display mode
        self.use_native_log_axis = True   # default: use native log axis

        # Central splitter: hist + (optional) LUT
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setLabel('bottom', 'Microtime (bins)')
        self._set_y_label()
        self.hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self._apply_log_mode()
        self.hist_plot.setDownsampling(auto=True)
        self.hist_plot.setClipToView(True)
        self.splitter.addWidget(self.hist_plot)

        self.lut_container = QtWidgets.QWidget()
        lut_layout = QtWidgets.QVBoxLayout(self.lut_container)
        lut_layout.setContentsMargins(0, 0, 0, 0)
        lut_layout.setSpacing(0)
        self.lut_plot_cum = pg.PlotWidget()
        self.lut_plot_delta = pg.PlotWidget()
        for p in (self.lut_plot_cum, self.lut_plot_delta):
            p.showGrid(x=True, y=True, alpha=0.3)
        self.lut_plot_cum.setLabel('bottom', 'Bin index'); self.lut_plot_cum.setLabel('left', 'Cumulative NTAC')
        self.lut_plot_delta.setLabel('bottom', 'Bin index'); self.lut_plot_delta.setLabel('left', 'ΔNTAC / bin')
        lut_layout.addWidget(QtWidgets.QLabel("Selected channel LUT"))
        lut_layout.addWidget(self.lut_plot_cum, 2)
        lut_layout.addWidget(self.lut_plot_delta, 2)
        self.splitter.addWidget(self.lut_container)
        self.lut_container.setVisible(False)
        self.setCentralWidget(self.splitter)

        # Curves
        self.curves = {}       # ch -> PlotDataItem
        self.curve_colors = {} # ch -> color

        # JSON preview dialog (hidden by default)
        self.json_dialog = JsonPreviewDialog(self)

        # README dialog (hidden by default)
        self.readme_dialog = ReadmeDialog(self)

        # Controls
        self._build_controls()

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Add or drop TTTR files to begin.")

    # -------------------- Controls --------------------

    def _build_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        panel.setAcceptDrops(True)  # Enable drops on the panel
        dock.setAcceptDrops(True)  # Enable drops on the dock
        dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        self.file_list = FileListWidget()
        self.file_list.setMinimumHeight(60)
        self.file_list.filesChanged.connect(self._files_changed)

        self.btn_add_files = QtWidgets.QToolButton()
        self.btn_add_files.setText("Add")
        self.btn_clear_files = QtWidgets.QToolButton()
        self.btn_clear_files.setText("Clear")
        self.btn_add_files.clicked.connect(self._add_files_dialog)
        self.btn_clear_files.clicked.connect(self._clear_files)

        # Reading routine selection
        self.reading_combo = QtWidgets.QComboBox()
        self.reading_combo.addItem("Auto", None)
        self.reading_combo.addItem("PTU", 0)
        self.reading_combo.addItem("HT3", 1)
        self.reading_combo.addItem("SPC-130", 2)
        self.reading_combo.addItem("SPC-600_256", 3)
        self.reading_combo.addItem("SPC-600_4096", 4)
        self.reading_combo.addItem("PHOTON-HDF5", 5)
        self.reading_combo.currentIndexChanged.connect(self._on_reading_routine_changed)

        # LUT buttons
        self.btn_load_lut  = QtWidgets.QToolButton()
        self.btn_load_lut.setText("Load LUT")
        self.btn_clear_luts = QtWidgets.QToolButton()
        self.btn_clear_luts.setText("Clear LUTs")
        self.btn_assign_lut_selected = QtWidgets.QToolButton()
        self.btn_assign_lut_selected.setText("Assign to Selected")
        self.btn_assign_lut_all = QtWidgets.QToolButton()
        self.btn_assign_lut_all.setText("Assign to All")
        self.btn_load_lut.clicked.connect(self._load_lut)
        self.btn_clear_luts.clicked.connect(self._clear_luts)
        self.btn_assign_lut_selected.clicked.connect(self._assign_lut_selected)
        self.btn_assign_lut_all.clicked.connect(self._assign_lut_all)
        self.btn_assign_lut_selected.setEnabled(False)
        self.btn_assign_lut_all.setEnabled(False)

        # LUT list
        self.lut_list = LUTListWidget()
        self.lut_list.customContextMenuRequested.connect(self._show_lut_context_menu)
        self.lut_list.lutDropped.connect(self._handle_lut_drop)

        # Display toggles
        self.chk_show_lut = QtWidgets.QCheckBox("Show LUT panel (selected channel)")
        self.chk_show_lut.toggled.connect(self._toggle_lut_panel)

        # Channel selection + shift (now combined with visibility)
        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.channel_list.currentRowChanged.connect(self._on_active_channel_changed)
        # Make items checkable for visibility
        self.channel_list.itemChanged.connect(self._on_channel_visibility_changed)

        self.shift_spin = QtWidgets.QSpinBox()
        self.shift_spin.setRange(-1_000_000, 1_000_000)
        self.shift_spin.setSingleStep(1)
        self.shift_spin.setEnabled(False)
        self.shift_spin.valueChanged.connect(self._on_shift_changed_live)

        self.btn_show_json = QtWidgets.QToolButton()
        self.btn_show_json.setText("Show JSON")
        self.btn_show_json.setCheckable(True)
        self.btn_show_json.toggled.connect(self._toggle_json_preview)

        self.btn_info = QtWidgets.QToolButton()
        self.btn_info.setText("Info")
        self.btn_info.setCheckable(True)
        self.btn_info.toggled.connect(self._toggle_readme)

        self.btn_load_settings = QtWidgets.QToolButton()
        self.btn_load_settings.setText("Load JSON")
        self.btn_load_settings.clicked.connect(self._load_settings)

        self.btn_save_settings = QtWidgets.QToolButton()
        self.btn_save_settings.setText("Save JSON")
        self.btn_save_settings.setEnabled(False)
        self.btn_save_settings.clicked.connect(self._save_settings)

        # Layout
        form = QtWidgets.QFormLayout(panel)
        form.setContentsMargins(2, 2, 2, 2)
        form.setSpacing(2)
        form.addRow(QtWidgets.QLabel("Files (drop here):"))
        form.addRow(self.file_list)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.btn_add_files)
        h.addWidget(self.btn_clear_files)
        form.addRow(h)

        form.addRow(QtWidgets.QLabel("Reading routine:"))
        form.addRow(self.reading_combo)

        h_lut = QtWidgets.QHBoxLayout()
        h_lut.addWidget(self.btn_load_lut)
        h_lut.addWidget(self.btn_clear_luts)
        form.addRow(h_lut)
        form.addRow(QtWidgets.QLabel("Loaded LUTs:"))
        form.addRow(self.lut_list)
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(self.btn_assign_lut_selected)
        h2.addWidget(self.btn_assign_lut_all)
        form.addRow(h2)

        form.addRow(self.chk_show_lut)

        form.addRow(QtWidgets.QLabel("Channels (check to show):"))
        form.addRow(self.channel_list)

        shift_row = QtWidgets.QHBoxLayout()
        shift_row.addWidget(QtWidgets.QLabel("Shift (bins):"))
        shift_row.addWidget(self.shift_spin)
        form.addRow(shift_row)

        h_json = QtWidgets.QHBoxLayout()
        h_json.addWidget(self.btn_show_json)
        h_json.addWidget(self.btn_info)
        h_json.addWidget(self.btn_load_settings)
        h_json.addWidget(self.btn_save_settings)
        form.addRow(h_json)

    def _handle_lut_drop(self, path: str):
        """Handle a LUT file dropped onto the LUT list."""
        try:
            lut_array = load_lut_file(path)
            filename = os.path.basename(path)
            self.loaded_luts[filename] = lut_array
            # Add to list if not already there
            items = [self.lut_list.item(i).text() for i in range(self.lut_list.count())]
            if filename not in items:
                self.lut_list.addItem(filename)
            # Select the newly loaded one
            for i in range(self.lut_list.count()):
                if self.lut_list.item(i).text() == filename:
                    self.lut_list.setCurrentRow(i)
                    break
            self.btn_assign_lut_selected.setEnabled(True)
            self.btn_assign_lut_all.setEnabled(True)
            QtWidgets.QMessageBox.information(
                self, "LUT loaded", f"Loaded LUT '{filename}' with {len(lut_array)} entries."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error loading LUT", str(e))

    # -------------------- Log axis mode --------------------

    def _set_y_label(self):
        self.hist_plot.setLabel('left', 'Counts (log)')

    def _apply_log_mode(self):
        self.hist_plot.setLogMode(x=False, y=self.use_native_log_axis)

    # -------------------- File handling --------------------

    def _add_files_dialog(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select TTTR files", "", "TTTR files (*.spc *.ht3 *.ptu *.phu *.photonhdf5);;All files (*)"
        )
        if files:
            self.file_list.add_files(files)

    def _clear_files(self):
        self.file_list.clear_files()
        self._unload_all()

    def _files_changed(self):
        paths = self.file_list.current_paths()
        if not paths:
            self._unload_all()
            return
        try:
            self._load_tttr_paths(paths)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            self._unload_all()

    def _unload_all(self):
        self.bundle = None
        self.channel_luts.clear()
        self.channel_shifts.clear()
        self.loaded_luts.clear()
        self._remove_all_curves()
        self.curve_colors.clear()
        self.lut_plot_cum.clear(); self.lut_plot_delta.clear()
        self.channel_list.clear()
        self.lut_list.clear()
        self.shift_spin.setEnabled(False)
        self.btn_assign_lut_selected.setEnabled(False)
        self.btn_assign_lut_all.setEnabled(False)
        self.btn_save_settings.setEnabled(False)
        self.status.showMessage("No files loaded.")
        if self.json_dialog.isVisible():
            self.json_dialog.set_json("")

    def _load_tttr_paths(self, paths):
        self.bundle = TTTRBundle(paths, self.reading_routine)
        used = sorted(self.bundle.used_channels())
        if not used:
            self.status.showMessage("Loaded files but found no used channels.")
        self.bundle.precompute_uncorrected()

        self.curve_colors = {ch: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, ch in enumerate(used)}
        for ch in used:
            self.channel_shifts.setdefault(int(ch), 0)

        self.channel_list.clear()
        for ch in used:
            it = QtWidgets.QListWidgetItem(f"Channel {ch}")
            it.setData(QtCore.Qt.UserRole, int(ch))
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)  # Default to checked
            self.channel_list.addItem(it)
        if self.channel_list.count() > 0:
            self.channel_list.setCurrentRow(0)

        self.btn_assign_lut_selected.setEnabled(bool(self.loaded_luts))
        self.btn_assign_lut_all.setEnabled(bool(self.loaded_luts))
        self.btn_save_settings.setEnabled(True)

        self._update_all_curves_full()
        self.status.showMessage(f"Loaded {len(paths)} files. Used channels: {used}")
        if self.json_dialog.isVisible():
            self._refresh_json_preview()

    # -------------------- LUT handling --------------------

    def _load_lut(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select LUT file", "", "LUT files (*.npy *.npz *.txt);;All files (*)"
        )
        if not path:
            return
        try:
            lut_array = load_lut_file(path)
            filename = os.path.basename(path)
            self.loaded_luts[filename] = lut_array
            # Add to list if not already there
            items = [self.lut_list.item(i).text() for i in range(self.lut_list.count())]
            if filename not in items:
                self.lut_list.addItem(filename)
            # Select the newly loaded one
            for i in range(self.lut_list.count()):
                if self.lut_list.item(i).text() == filename:
                    self.lut_list.setCurrentRow(i)
                    break
            self.btn_assign_lut_selected.setEnabled(True)
            self.btn_assign_lut_all.setEnabled(True)
            QtWidgets.QMessageBox.information(
                self, "LUT loaded", f"Loaded LUT '{filename}' with {len(lut_array)} entries."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error loading LUT", str(e))

    def _clear_luts(self):
        self.loaded_luts.clear()
        self.lut_list.clear()
        self.btn_assign_lut_selected.setEnabled(False)
        self.btn_assign_lut_all.setEnabled(False)

    def _show_lut_context_menu(self, position):
        menu = QtWidgets.QMenu()
        remove_action = menu.addAction("Remove Selected")
        clear_action = menu.addAction("Clear All")
        
        remove_action.triggered.connect(self._remove_selected_lut)
        clear_action.triggered.connect(self._clear_luts)
        
        menu.exec_(self.lut_list.mapToGlobal(position))

    def _remove_selected_lut(self):
        item = self.lut_list.currentItem()
        if item:
            filename = item.text()
            self.loaded_luts.pop(filename, None)
            self.lut_list.takeItem(self.lut_list.row(item))
            if self.lut_list.count() == 0:
                self.btn_assign_lut_selected.setEnabled(False)
                self.btn_assign_lut_all.setEnabled(False)

    def _get_selected_lut(self):
        """Get the currently selected LUT array."""
        item = self.lut_list.currentItem()
        if not item:
            return None
        filename = item.text()
        return self.loaded_luts.get(filename)

    def _assign_lut_selected(self):
        lut_array = self._get_selected_lut()
        if lut_array is None:
            QtWidgets.QMessageBox.warning(self, "No LUT selected", "Select a LUT from the list.")
            return
        if not self.bundle:
            QtWidgets.QMessageBox.warning(self, "No data", "Load TTTR files first.")
            return
        ch = self._active_channel()
        if ch is None:
            QtWidgets.QMessageBox.warning(self, "No channel selected", "Select a channel from the list.")
            return
        arr = np.array(lut_array, dtype=np.float64)
        self.channel_luts[int(ch)] = arr
        self._update_all_curves_full()
        if self.chk_show_lut.isChecked():
            self._update_lut_plot_for_channel(self._active_channel())
        if self.json_dialog.isVisible():
            self._refresh_json_preview()

    def _assign_lut_all(self):
        lut_array = self._get_selected_lut()
        if lut_array is None:
            QtWidgets.QMessageBox.warning(self, "No LUT selected", "Select a LUT from the list.")
            return
        if not self.bundle:
            QtWidgets.QMessageBox.warning(self, "No data", "Load TTTR files first.")
            return
        arr = np.array(lut_array, dtype=np.float64)
        for ch in self.bundle.used_channels():
            self.channel_luts[int(ch)] = arr
        self._update_all_curves_full()
        if self.chk_show_lut.isChecked():
            self._update_lut_plot_for_channel(self._active_channel())
        if self.json_dialog.isVisible():
            self._refresh_json_preview()

    # -------------------- Plotting --------------------

    def _toggle_lut_panel(self, checked):
        self.lut_container.setVisible(bool(checked))
        if checked:
            self._update_lut_plot_for_channel(self._active_channel())

    def _get_base_xy(self, ch):
        want_log = False  # always use linear data since native log is on
        if self.channel_luts:
            return self.bundle.get_cor(ch, want_log, self.channel_luts)
        return self.bundle.get_unc(ch, want_log)

    def _remove_all_curves(self):
        for item in self.curves.values():
            try:
                self.hist_plot.removeItem(item)
            except Exception:
                pass
        self.curves.clear()

    def _update_all_curves_full(self):
        """Create/update curves for visible channels. No global clear()."""
        if not self.bundle:
            self._remove_all_curves()
            return

        draw = set(self._selected_channels())
        if not draw:
            self._remove_all_curves()
            self.status.showMessage("No channels selected for display.")
            return

        # Ensure corrected cache (depends only on LUTs)
        if self.channel_luts:
            self.bundle.ensure_corrected_cache(self.channel_luts)

        # Remove curves that are no longer visible
        for ch in list(self.curves.keys()):
            if ch not in draw:
                try:
                    self.hist_plot.removeItem(self.curves[ch])
                except Exception:
                    pass
                self.curves.pop(ch, None)

        any_points = False
        # Create or update visible curves
        for idx, ch in enumerate(sorted(draw)):
            color = self.curve_colors.get(ch, COLOR_CYCLE[idx % len(COLOR_CYCLE)])
            x, y = self._get_base_xy(ch)
            if x.size == 0 or y.size == 0:
                self.status.showMessage(f"Channel {ch}: empty histogram.")
                continue
            # shift
            shift = int(self.channel_shifts.get(ch, 0))
            y_disp = np.roll(y, shift) if (shift and len(y) > 0) else y
            any_points = any_points or np.any(np.isfinite(y_disp))

            if ch in self.curves:
                self.curves[ch].setData(x, y_disp)
                self.curves[ch].setPen(nice_pen(color, width=2))
            else:
                item = self.hist_plot.plot(x, y_disp, pen=nice_pen(color, width=2))
                item.setZValue(0)
                item.setDownsampling(auto=True)
                item.setClipToView(True)
                self.curves[ch] = item

        self._emphasize_active_curve()
        self._force_range_from_visible()
        if not any_points:
            self.status.showMessage("No finite data points to display (all zeros or NaNs?).")

    def _force_range_from_visible(self):
        """Compute a robust axis range from visible curves and set it explicitly."""
        xs = []
        ys = []
        for ch, item in self.curves.items():
            data = item.getData()
            if not data:
                continue
            x, y = data
            if x is None or y is None:
                continue
            x = np.asarray(x)
            y = np.asarray(y)
            m = np.isfinite(x) & np.isfinite(y)
            if not np.any(m):
                continue
            xs.append(x[m])
            ys.append(y[m])
        if not xs or not ys:
            return
        xmin = min(float(a.min()) for a in xs)
        xmax = max(float(a.max()) for a in xs)
        ymin = min(float(a.min()) for a in ys)
        ymax = max(float(a.max()) for a in ys)
        if not np.isfinite([xmin, xmax, ymin, ymax]).all():
            return
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        # generous padding
        xr = xmax - xmin
        yr = ymax - ymin
        self.hist_plot.setXRange(xmin - 0.02*xr, xmax + 0.02*xr, padding=0)
        self.hist_plot.setYRange(ymin - 0.05*yr, ymax + 0.05*yr, padding=0)

    def _emphasize_active_curve(self):
        ch = self._active_channel()
        for c, item in self.curves.items():
            col = self.curve_colors.get(c, (100, 100, 100))
            item.setPen(nice_pen(col, width=2))
            item.setZValue(0)
        if ch in self.curves:
            col = self.curve_colors.get(ch, (0, 0, 0))
            self.curves[ch].setPen(nice_pen(col, width=4))
            self.curves[ch].setZValue(10)

    # -------------------- Interaction: channel & shift --------------------

    def _active_channel(self):
        it = self.channel_list.currentItem()
        if not it:
            return None
        return int(it.data(QtCore.Qt.UserRole))

    def _on_active_channel_changed(self, _row):
        ch = self._active_channel()
        if ch is None:
            self.shift_spin.setEnabled(False)
            return
        self.shift_spin.setEnabled(True)
        self.shift_spin.blockSignals(True)
        self.shift_spin.setValue(int(self.channel_shifts.get(ch, 0)))
        self.shift_spin.blockSignals(False)
        self._emphasize_active_curve()
        if self.lut_container.isVisible():
            self._update_lut_plot_for_channel(ch)

    def _on_reading_routine_changed(self, index):
        self.reading_routine = self.reading_combo.itemData(index)
        # If files are already loaded, reload them with new reading routine
        paths = self.file_list.current_paths()
        if paths:
            self._files_changed()

    def _selected_channels(self):
        """Get list of channels that are checked in the channel list."""
        selected = []
        for i in range(self.channel_list.count()):
            item = self.channel_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                ch = int(item.data(QtCore.Qt.UserRole))
                selected.append(ch)
        return selected

    def _on_shift_changed_live(self, value):
        """Roll only the active curve; no recompute, no global redraw."""
        ch = self._active_channel()
        if ch is None or not self.bundle:
            return
        self.channel_shifts[int(ch)] = int(value)

        item = self.curves.get(ch)
        if item is not None:
            x, y = self._get_base_xy(ch)  # already precomputed (log or lin)
            shift = int(value)
            y_disp = np.roll(y, shift) if (shift and len(y) > 0) else y
            item.setData(x, y_disp)
            self._emphasize_active_curve()
            self._force_range_from_visible()

    def _on_channel_visibility_changed(self, item):
        """Called when a channel's check state changes."""
        self._update_all_curves_full()

    # -------------------- LUT plotting --------------------

    def _update_lut_plot_for_channel(self, ch):
        self.lut_plot_cum.clear()
        self.lut_plot_delta.clear()
        if ch is None:
            return
        lut = self.channel_luts.get(int(ch))
        if lut is None or len(lut) == 0:
            self.lut_plot_cum.addItem(pg.TextItem("No LUT for this channel", anchor=(0,0)))
            return
        n = len(lut)
        x = np.arange(n, dtype=np.float32)
        self.lut_plot_cum.plot(x, f32(lut), pen=nice_pen((70, 70, 200), 2))
        delta = np.diff(np.concatenate(([0.0], lut)))
        self.lut_plot_delta.plot(x, f32(delta), pen=nice_pen((200, 70, 70), 2))

    # -------------------- JSON preview & save --------------------

    def _current_settings_dict(self):
        used = self.bundle.used_channels() if self.bundle else []
        return {
            "description": "TTTR microtime correction settings",
            "version": "1.0",
            "reading_routine": self.reading_routine,
            "channel_luts": {int(ch): np.asarray(arr).tolist() for ch, arr in self.channel_luts.items()},
            "channel_shifts": {int(ch): int(shift) for ch, shift in self.channel_shifts.items()},
            "metadata": {
                "created": str(np.datetime64('now')),
                "used_channels": [int(ch) for ch in used],
                "notes": "Generated by TTTR Settings Generator plugin",
            },
        }

    def _toggle_json_preview(self, checked):
        if checked:
            self._refresh_json_preview()
            self.json_dialog.show()
            self.json_dialog.raise_()
            self.json_dialog.activateWindow()
        else:
            self.json_dialog.hide()

    def _toggle_readme(self, checked):
        if checked:
            self.readme_dialog.show()
            self.readme_dialog.raise_()
            self.readme_dialog.activateWindow()
        else:
            self.readme_dialog.hide()

    def _refresh_json_preview(self):
        try:
            txt = json.dumps(json_safe(self._current_settings_dict()), indent=2)
        except Exception as e:
            txt = f"<!> Error rendering JSON:\n{e}"
        self.json_dialog.set_json(txt)

    def _load_settings(self):
        """Load settings from a JSON file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load settings.tttr.json", "", "JSON (*.json);;All files (*)"
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate basic structure
            if not isinstance(data, dict):
                raise ValueError("Invalid JSON structure: expected object")

            # Check version compatibility
            version = data.get("version", "1.0")
            if version != "1.0":
                QtWidgets.QMessageBox.warning(
                    self, "Version Warning",
                    f"Settings file version {version} may not be fully compatible with this version."
                )

            # Apply reading routine if present
            reading_routine = data.get("reading_routine")
            if reading_routine is not None:
                # Find the index for this reading routine
                for i in range(self.reading_combo.count()):
                    if self.reading_combo.itemData(i) == reading_routine:
                        self.reading_combo.setCurrentIndex(i)
                        break

            # Load channel LUTs
            channel_luts = data.get("channel_luts", {})
            if channel_luts:
                # Clear existing LUTs first
                self.channel_luts.clear()
                for ch_str, lut_list in channel_luts.items():
                    ch = int(ch_str)
                    lut_array = np.array(lut_list, dtype=np.float64)
                    self.channel_luts[ch] = lut_array

            # Load channel shifts
            channel_shifts = data.get("channel_shifts", {})
            if channel_shifts:
                # Clear existing shifts first
                self.channel_shifts.clear()
                for ch_str, shift_val in channel_shifts.items():
                    ch = int(ch_str)
                    self.channel_shifts[ch] = int(shift_val)

            # Update UI and plots
            self._update_all_curves_full()
            if self.chk_show_lut.isChecked():
                self._update_lut_plot_for_channel(self._active_channel())
            if self.json_dialog.isVisible():
                self._refresh_json_preview()

            QtWidgets.QMessageBox.information(
                self, "Settings Loaded",
                f"Successfully loaded settings from:\n{path}\n\n"
                f"LUTs: {len(channel_luts)} channels\n"
                f"Shifts: {len(channel_shifts)} channels"
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to load settings:\n{str(e)}")

    def _save_settings(self):
        if not self.bundle:
            QtWidgets.QMessageBox.warning(self, "No data", "Load TTTR files first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save settings.tttr.json", "settings.tttr.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(json_safe(self._current_settings_dict()), f, indent=2)
            QtWidgets.QMessageBox.information(self, "Saved", f"Settings saved to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the TTTRSettingsGenerator class
    window = TTTRSettingsGenerator()
    # Show the window
    window.show()
