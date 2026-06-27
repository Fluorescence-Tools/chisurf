"""Dockable panel for creating TTTR LUT settings JSON files."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import tttrlib
from qtpy import QtCore, QtGui, QtWidgets

VALID_EXTS = {".spc", ".ht3", ".ptu", ".phu", ".photonhdf5"}
EPS = 1e-12
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


def load_lut_file(path: str) -> np.ndarray:
    """Load a 1D LUT array from disk.

    Parameters
    ----------
    path : str
        LUT file path. Supported extensions are ``.npy``, ``.txt``, and ``.npz``.

    Returns
    -------
    numpy.ndarray
        One-dimensional float64 LUT array.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        array = np.load(path)
    elif ext == ".txt":
        array = np.loadtxt(path)
    elif ext == ".npz":
        data = np.load(path)
        keys = list(data.keys())
        if not keys:
            raise ValueError(".npz file contains no arrays.")
        array = data[keys[0]]
    else:
        raise ValueError(f"Unsupported LUT file type: {ext}")

    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Expected 1D LUT array.")
    return array


def nice_pen(color, width: int = 2):
    """Create a pyqtgraph pen."""
    return pg.mkPen(color=color, width=width)


def json_safe(obj: object) -> object:
    """Convert numpy values to JSON-serializable Python objects."""
    if isinstance(obj, np.generic):
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
        return {str(json_safe(key)): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    return obj


def f32(array: np.ndarray) -> np.ndarray:
    """Return an array as float32."""
    return np.asarray(array, dtype=np.float32)


def logify(y_values: np.ndarray) -> np.ndarray:
    """Return a log10-transformed histogram with a small offset."""
    return f32(np.log10(np.maximum(y_values, 0) + EPS))


class FileListWidget(QtWidgets.QListWidget):
    """List widget that accepts dropped TTTR file paths."""

    filesChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the file list widget."""
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)
        self.setMinimumHeight(60)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept dropped URL lists."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        """Accept drag moves."""
        event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Add valid TTTR files from a drop event."""
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isfile(path):
                    ext = os.path.splitext(path)[1].lower()
                    if ext in VALID_EXTS:
                        paths.append(path)
            if paths:
                self.add_files(paths)
        event.acceptProposedAction()

    def add_files(self, paths: list[str]) -> None:
        """Add files to the list without duplicates."""
        existing = {self.item(index).text() for index in range(self.count())}
        for path in paths:
            if path not in existing:
                self.addItem(path)
        self.filesChanged.emit()

    def current_paths(self) -> list[str]:
        """Return all paths currently listed."""
        return [self.item(index).text() for index in range(self.count())]

    def clear_files(self) -> None:
        """Clear all file entries."""
        self.clear()
        self.filesChanged.emit()


class LUTListWidget(QtWidgets.QListWidget):
    """List widget that accepts dropped LUT files."""

    lutDropped = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the LUT list widget."""
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setMaximumHeight(80)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DropOnly)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Accept dropped LUT URLs."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    ext = os.path.splitext(path)[1].lower()
                    if ext in {".npy", ".npz", ".txt"}:
                        event.acceptProposedAction()
                        return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Emit the first valid dropped LUT path."""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isfile(path):
                    ext = os.path.splitext(path)[1].lower()
                    if ext in {".npy", ".npz", ".txt"}:
                        self.lutDropped.emit(path)
                        break
        event.acceptProposedAction()


def _hist_time_safe(tttr_obj: object, channel: int) -> tuple[np.ndarray, np.ndarray]:
    """Return microtime histogram coordinates from tttrlib robustly."""
    first, second = tttr_obj.get_microtime_histogram(channels=[int(channel)])
    first = np.asarray(first)
    second = np.asarray(second)

    def looks_like_time(values: np.ndarray) -> bool:
        if values.ndim != 1 or values.size == 0:
            return False
        if not np.all(np.isfinite(values)):
            return False
        diff = np.diff(values)
        return np.all(diff >= 0) and np.count_nonzero(diff) > 0

    def looks_like_hist(values: np.ndarray) -> bool:
        return values.ndim == 1 and values.size > 0 and np.all(np.isfinite(values)) and np.all(values >= 0)

    if looks_like_time(first) and looks_like_hist(second):
        x_values, y_values = first, second
    elif looks_like_time(second) and looks_like_hist(first):
        x_values, y_values = second, first
    else:
        y_values, x_values = first, second

    x_values = np.ravel(x_values)
    y_values = np.ravel(y_values)
    size = min(len(x_values), len(y_values))
    return f32(x_values[:size]), f32(y_values[:size])


class TTTRBundle:
    """Load TTTR files and cache corrected and uncorrected histograms."""

    def __init__(self, paths: list[str], container_type: int | None = None) -> None:
        """Load and combine TTTR files."""
        if not paths:
            raise FileNotFoundError("No TTTR files.")
        self.paths = list(paths)
        self.container_type = container_type
        self.tt = self._load_combined(self.paths, self.container_type)
        self._unc_lin: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._unc_log: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._cor_lin: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._cor_log: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._lut_sig: tuple[object, ...] | None = None

    @classmethod
    def _load_combined(cls, paths: list[str], container_type: int | None = None) -> object:
        """Combine multiple TTTR containers."""
        iterator = iter(paths)
        tttr = tttrlib.TTTR(next(iterator), container_type=container_type)
        for path in iterator:
            tttr += tttrlib.TTTR(path, container_type=container_type)
        return tttr

    def used_channels(self) -> list[int]:
        """Return used routing channels as Python integers."""
        return [int(channel) for channel in self.tt.get_used_routing_channels()]

    def precompute_uncorrected(self) -> None:
        """Precompute uncorrected linear and log histograms."""
        self._unc_lin.clear()
        self._unc_log.clear()
        for channel in self.used_channels():
            x_values, y_values = _hist_time_safe(self.tt, channel)
            self._unc_lin[int(channel)] = (x_values, y_values)
            self._unc_log[int(channel)] = (x_values, logify(y_values))

    def get_unc(self, channel: int, want_log: bool) -> tuple[np.ndarray, np.ndarray]:
        """Return uncorrected histogram data."""
        cache = self._unc_log if want_log else self._unc_lin
        return cache[int(channel)]

    def _lut_signature(self, channel_luts: dict[int, np.ndarray]) -> tuple[object, ...]:
        """Build a cache signature from assigned LUTs."""
        keys = tuple(sorted(int(key) for key in (channel_luts or {}).keys()))
        lengths = tuple(int(len(channel_luts[key])) for key in keys)
        return "lut", keys, lengths

    def ensure_corrected_cache(self, channel_luts: dict[int, np.ndarray]) -> None:
        """Rebuild corrected caches when assigned LUTs change."""
        signature = self._lut_signature(channel_luts)
        if signature == self._lut_sig:
            return

        self._cor_lin.clear()
        self._cor_log.clear()
        tttr_corrected = self._load_combined(self.paths)
        if channel_luts:
            tttr_corrected.apply_channel_luts(channel_luts, {})
            tttr_corrected.apply_luts_and_shifts(-1, True)
        for channel in self.used_channels():
            x_values, y_values = _hist_time_safe(tttr_corrected, channel)
            self._cor_lin[int(channel)] = (x_values, y_values)
            self._cor_log[int(channel)] = (x_values, logify(y_values))
        self._lut_sig = signature

    def get_cor(self, channel: int, want_log: bool, channel_luts: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Return corrected histogram data."""
        self.ensure_corrected_cache(channel_luts)
        cache = self._cor_log if want_log else self._cor_lin
        return cache[int(channel)]


class JsonPreviewDialog(QtWidgets.QDialog):
    """Dialog showing generated settings JSON."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the JSON preview dialog."""
        super().__init__(parent)
        self.setWindowTitle("Settings JSON Preview")
        self.resize(700, 600)
        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignRight)

    def set_json(self, text: str) -> None:
        """Set preview text."""
        self.text.setPlainText(text)


class ReadmeDialog(QtWidgets.QDialog):
    """Dialog showing plugin help text."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the help dialog."""
        super().__init__(parent)
        self.setWindowTitle("TTTR LUT Tools - Help")
        self.resize(800, 600)
        layout = QtWidgets.QVBoxLayout(self)

        readme_path = Path(__file__).parents[2] / "tttr_settings_generator" / "README.md"
        try:
            readme_text = readme_path.read_text(encoding="utf-8")
        except Exception:
            readme_text = (
                "TTTR LUT Tools combines microtime LUT computation and channel LUT "
                "settings creation in one dockable plugin."
            )

        self.text = QtWidgets.QTextEdit()
        self.text.setPlainText(readme_text)
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=QtCore.Qt.AlignRight)


class ChannelCheckList(QtWidgets.QWidget):
    """Checkable channel list widget."""

    toggled = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the channel list."""
        super().__init__(parent)
        self._checks: dict[int, QtWidgets.QCheckBox] = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)

    def clear(self) -> None:
        """Remove all checkboxes."""
        for checkbox in self._checks.values():
            checkbox.setParent(None)
            checkbox.deleteLater()
        self._checks.clear()

    def set_channels(self, channels: list[int]) -> None:
        """Populate the list with channels."""
        self.clear()
        layout = self.layout()
        for channel in sorted(channels):
            checkbox = QtWidgets.QCheckBox(f"Show Channel {channel}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.toggled.emit)
            layout.insertWidget(layout.count() - 1, checkbox)
            self._checks[int(channel)] = checkbox

    def selected_channels(self) -> list[int]:
        """Return checked channels."""
        return [channel for channel, checkbox in self._checks.items() if checkbox.isChecked()]


class StatusLabel(QtWidgets.QLabel):
    """Label with a minimal status-bar-like API."""

    def showMessage(self, text: str, timeout: int = 0) -> None:
        """Show a message."""
        self.setText(text)

    def clear(self) -> None:
        """Clear the status text."""
        self.setText("")


class TTTRSettingsPanel(QtWidgets.QWidget):
    """Panel for assigning LUTs, channel shifts, and saving settings JSON."""

    def __init__(self) -> None:
        """Create the settings panel."""
        super().__init__()

        self.bundle = None
        self.channel_luts: dict[int, np.ndarray] = {}
        self.channel_shifts: dict[int, int] = {}
        self.loaded_luts: dict[str, np.ndarray] = {}
        self.reading_routine = None
        self.use_native_log_axis = True

        plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.setLabel("bottom", "Microtime (bins)")
        self._set_y_label()
        self.hist_plot.showGrid(x=True, y=True, alpha=0.3)
        self._apply_log_mode()
        self.hist_plot.setDownsampling(auto=True)
        self.hist_plot.setClipToView(True)
        plot_splitter.addWidget(self.hist_plot)

        self.lut_container = QtWidgets.QWidget()
        lut_layout = QtWidgets.QVBoxLayout(self.lut_container)
        lut_layout.setContentsMargins(0, 0, 0, 0)
        lut_layout.setSpacing(0)
        self.lut_plot_cum = pg.PlotWidget()
        self.lut_plot_delta = pg.PlotWidget()
        for plot in (self.lut_plot_cum, self.lut_plot_delta):
            plot.showGrid(x=True, y=True, alpha=0.3)
        self.lut_plot_cum.setLabel("bottom", "Bin index")
        self.lut_plot_cum.setLabel("left", "Cumulative NTAC")
        self.lut_plot_delta.setLabel("bottom", "Bin index")
        self.lut_plot_delta.setLabel("left", "ΔNTAC / bin")
        lut_layout.addWidget(QtWidgets.QLabel("Selected channel LUT"))
        lut_layout.addWidget(self.lut_plot_cum, 2)
        lut_layout.addWidget(self.lut_plot_delta, 2)
        plot_splitter.addWidget(self.lut_container)
        self.lut_container.setVisible(False)

        self.curves: dict[int, object] = {}
        self.curve_colors: dict[int, tuple[int, int, int]] = {}
        self.json_dialog = JsonPreviewDialog(self)
        self.readme_dialog = ReadmeDialog(self)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.addWidget(plot_splitter)

        controls_panel = QtWidgets.QWidget()
        controls_panel.setAcceptDrops(True)
        controls_panel.setMinimumWidth(270)
        controls_panel.setMaximumWidth(360)
        self._build_controls(controls_panel)
        self.splitter.addWidget(controls_panel)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setCollapsible(1, False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.splitter, 1)
        self.status = StatusLabel("Add or drop TTTR files to begin.")
        layout.addWidget(self.status)

    def _build_controls(self, panel: QtWidgets.QWidget) -> None:
        """Build the settings control panel."""
        self.file_list = FileListWidget()
        self.file_list.setMinimumHeight(45)
        self.file_list.setMaximumHeight(70)
        self.file_list.filesChanged.connect(self._files_changed)

        self.btn_add_files = QtWidgets.QToolButton()
        self.btn_add_files.setText("📂 Add")
        self.btn_clear_files = QtWidgets.QToolButton()
        self.btn_clear_files.setText("🧹 Clear")
        self.btn_add_files.clicked.connect(self._add_files_dialog)
        self.btn_clear_files.clicked.connect(self._clear_files)

        self.reading_combo = QtWidgets.QComboBox()
        self.reading_combo.addItem("Auto", None)
        self.reading_combo.addItem("PTU", 0)
        self.reading_combo.addItem("HT3", 1)
        self.reading_combo.addItem("SPC-130", 2)
        self.reading_combo.addItem("SPC-600_256", 3)
        self.reading_combo.addItem("SPC-600_4096", 4)
        self.reading_combo.addItem("PHOTON-HDF5", 5)
        self.reading_combo.currentIndexChanged.connect(self._on_reading_routine_changed)

        self.btn_load_lut = QtWidgets.QToolButton()
        self.btn_load_lut.setText("📥 Load LUT")
        self.btn_clear_luts = QtWidgets.QToolButton()
        self.btn_clear_luts.setText("🧹 Clear LUTs")
        self.btn_assign_lut_selected = QtWidgets.QToolButton()
        self.btn_assign_lut_selected.setText("🎯 Selected")
        self.btn_assign_lut_all = QtWidgets.QToolButton()
        self.btn_assign_lut_all.setText("🌐 All")
        self.btn_load_lut.clicked.connect(self._load_lut)
        self.btn_clear_luts.clicked.connect(self._clear_luts)
        self.btn_assign_lut_selected.clicked.connect(self._assign_lut_selected)
        self.btn_assign_lut_all.clicked.connect(self._assign_lut_all)
        self.btn_assign_lut_selected.setEnabled(False)
        self.btn_assign_lut_all.setEnabled(False)

        self.lut_list = LUTListWidget()
        self.lut_list.setMinimumHeight(40)
        self.lut_list.setMaximumHeight(65)
        self.lut_list.customContextMenuRequested.connect(self._show_lut_context_menu)
        self.lut_list.lutDropped.connect(self._handle_lut_drop)

        self.chk_show_lut = QtWidgets.QCheckBox("Show LUT panel")
        self.chk_show_lut.toggled.connect(self._toggle_lut_panel)

        self.channel_list = QtWidgets.QListWidget()
        self.channel_list.setMinimumHeight(75)
        self.channel_list.setMaximumHeight(130)
        self.channel_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.channel_list.currentRowChanged.connect(self._on_active_channel_changed)
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

        main_layout = QtWidgets.QVBoxLayout(panel)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        files_group = QtWidgets.QGroupBox("Files")
        files_layout = QtWidgets.QVBoxLayout(files_group)
        files_layout.addWidget(self.file_list)
        file_buttons = QtWidgets.QHBoxLayout()
        file_buttons.addWidget(self.btn_add_files)
        file_buttons.addWidget(self.btn_clear_files)
        file_buttons.addStretch(1)
        files_layout.addLayout(file_buttons)
        main_layout.addWidget(files_group)

        io_group = QtWidgets.QGroupBox("Reading / LUTs")
        io_layout = QtWidgets.QGridLayout(io_group)
        io_layout.addWidget(QtWidgets.QLabel("Reading"), 0, 0)
        io_layout.addWidget(self.reading_combo, 0, 1, 1, 2)

        lut_buttons = QtWidgets.QHBoxLayout()
        lut_buttons.addWidget(self.btn_load_lut)
        lut_buttons.addWidget(self.btn_clear_luts)
        io_layout.addLayout(lut_buttons, 1, 0, 1, 3)

        io_layout.addWidget(QtWidgets.QLabel("Loaded LUTs"), 2, 0, 1, 3)
        io_layout.addWidget(self.lut_list, 3, 0, 1, 3)

        assign_buttons = QtWidgets.QHBoxLayout()
        assign_buttons.addWidget(self.btn_assign_lut_selected)
        assign_buttons.addWidget(self.btn_assign_lut_all)
        assign_buttons.addStretch(1)
        io_layout.addLayout(assign_buttons, 4, 0, 1, 3)
        main_layout.addWidget(io_group)

        channels_group = QtWidgets.QGroupBox("Channels")
        channels_layout = QtWidgets.QVBoxLayout(channels_group)
        channels_layout.addWidget(self.chk_show_lut)
        channels_layout.addWidget(self.channel_list)
        shift_row = QtWidgets.QHBoxLayout()
        shift_row.addWidget(QtWidgets.QLabel("Shift"))
        shift_row.addWidget(self.shift_spin)
        shift_row.addStretch(1)
        channels_layout.addLayout(shift_row)
        main_layout.addWidget(channels_group)

        json_group = QtWidgets.QGroupBox("JSON")
        json_layout = QtWidgets.QHBoxLayout(json_group)
        json_layout.addWidget(self.btn_show_json)
        json_layout.addWidget(self.btn_info)
        json_layout.addWidget(self.btn_load_settings)
        json_layout.addWidget(self.btn_save_settings)
        json_layout.addStretch(1)
        main_layout.addWidget(json_group)
        main_layout.addStretch(1)

    def _handle_lut_drop(self, path: str) -> None:
        """Handle a dropped LUT file."""
        try:
            lut_array = load_lut_file(path)
            filename = os.path.basename(path)
            self.loaded_luts[filename] = lut_array
            items = [self.lut_list.item(index).text() for index in range(self.lut_list.count())]
            if filename not in items:
                self.lut_list.addItem(filename)
            for index in range(self.lut_list.count()):
                if self.lut_list.item(index).text() == filename:
                    self.lut_list.setCurrentRow(index)
                    break
            self.btn_assign_lut_selected.setEnabled(True)
            self.btn_assign_lut_all.setEnabled(True)
            QtWidgets.QMessageBox.information(
                self,
                "LUT loaded",
                f"Loaded LUT '{filename}' with {len(lut_array)} entries.",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error loading LUT", str(exc))

    def _set_y_label(self) -> None:
        """Set the histogram y-axis label."""
        self.hist_plot.setLabel("left", "Counts (log)")

    def _apply_log_mode(self) -> None:
        """Apply native log mode to the histogram."""
        self.hist_plot.setLogMode(x=False, y=self.use_native_log_axis)

    def _add_files_dialog(self) -> None:
        """Open a TTTR file dialog."""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select TTTR files",
            "",
            "TTTR files (*.spc *.ht3 *.ptu *.phu *.photonhdf5);;All files (*)",
        )
        if files:
            self.file_list.add_files(files)

    def _clear_files(self) -> None:
        """Clear TTTR files."""
        self.file_list.clear_files()
        self._unload_all()

    def _files_changed(self) -> None:
        """Reload files when the file list changes."""
        paths = self.file_list.current_paths()
        if not paths:
            self._unload_all()
            return
        try:
            self._load_tttr_paths(paths)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))
            self._unload_all()

    def _unload_all(self) -> None:
        """Clear all panel state."""
        self.bundle = None
        self.channel_luts.clear()
        self.channel_shifts.clear()
        self.loaded_luts.clear()
        self._remove_all_curves()
        self.curve_colors.clear()
        self.lut_plot_cum.clear()
        self.lut_plot_delta.clear()
        self.channel_list.clear()
        self.lut_list.clear()
        self.shift_spin.setEnabled(False)
        self.btn_assign_lut_selected.setEnabled(False)
        self.btn_assign_lut_all.setEnabled(False)
        self.btn_save_settings.setEnabled(False)
        self.status.showMessage("No files loaded.")
        if self.json_dialog.isVisible():
            self.json_dialog.set_json("")

    def _load_tttr_paths(self, paths: list[str]) -> None:
        """Load TTTR paths and populate channel controls."""
        self.bundle = TTTRBundle(paths, self.reading_routine)
        used = sorted(self.bundle.used_channels())
        if not used:
            self.status.showMessage("Loaded files but found no used channels.")
        self.bundle.precompute_uncorrected()

        self.curve_colors = {
            channel: COLOR_CYCLE[index % len(COLOR_CYCLE)] for index, channel in enumerate(used)
        }
        for channel in used:
            self.channel_shifts.setdefault(int(channel), 0)

        self.channel_list.clear()
        for channel in used:
            item = QtWidgets.QListWidgetItem(f"Channel {channel}")
            item.setData(QtCore.Qt.UserRole, int(channel))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.channel_list.addItem(item)
        if self.channel_list.count() > 0:
            self.channel_list.setCurrentRow(0)

        self.btn_assign_lut_selected.setEnabled(bool(self.loaded_luts))
        self.btn_assign_lut_all.setEnabled(bool(self.loaded_luts))
        self.btn_save_settings.setEnabled(True)

        self._update_all_curves_full()
        self.status.showMessage(f"Loaded {len(paths)} files. Used channels: {used}")
        if self.json_dialog.isVisible():
            self._refresh_json_preview()

    def _load_lut(self) -> None:
        """Open a LUT file dialog."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select LUT file",
            "",
            "LUT files (*.npy *.npz *.txt);;All files (*)",
        )
        if not path:
            return
        try:
            lut_array = load_lut_file(path)
            filename = os.path.basename(path)
            self.loaded_luts[filename] = lut_array
            items = [self.lut_list.item(index).text() for index in range(self.lut_list.count())]
            if filename not in items:
                self.lut_list.addItem(filename)
            for index in range(self.lut_list.count()):
                if self.lut_list.item(index).text() == filename:
                    self.lut_list.setCurrentRow(index)
                    break
            self.btn_assign_lut_selected.setEnabled(True)
            self.btn_assign_lut_all.setEnabled(True)
            QtWidgets.QMessageBox.information(
                self,
                "LUT loaded",
                f"Loaded LUT '{filename}' with {len(lut_array)} entries.",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error loading LUT", str(exc))

    def _clear_luts(self) -> None:
        """Clear loaded LUTs."""
        self.loaded_luts.clear()
        self.lut_list.clear()
        self.btn_assign_lut_selected.setEnabled(False)
        self.btn_assign_lut_all.setEnabled(False)

    def _show_lut_context_menu(self, position: QtCore.QPoint) -> None:
        """Show the LUT context menu."""
        menu = QtWidgets.QMenu()
        remove_action = menu.addAction("Remove Selected")
        clear_action = menu.addAction("Clear All")
        remove_action.triggered.connect(self._remove_selected_lut)
        clear_action.triggered.connect(self._clear_luts)
        menu.exec_(self.lut_list.mapToGlobal(position))

    def _remove_selected_lut(self) -> None:
        """Remove the selected LUT."""
        item = self.lut_list.currentItem()
        if item:
            filename = item.text()
            self.loaded_luts.pop(filename, None)
            self.lut_list.takeItem(self.lut_list.row(item))
            if self.lut_list.count() == 0:
                self.btn_assign_lut_selected.setEnabled(False)
                self.btn_assign_lut_all.setEnabled(False)

    def _get_selected_lut(self) -> np.ndarray | None:
        """Return the selected LUT array."""
        item = self.lut_list.currentItem()
        if not item:
            return None
        return self.loaded_luts.get(item.text())

    def _assign_lut_selected(self) -> None:
        """Assign the selected LUT to the active channel."""
        lut_array = self._get_selected_lut()
        if lut_array is None:
            QtWidgets.QMessageBox.warning(self, "No LUT selected", "Select a LUT from the list.")
            return
        if not self.bundle:
            QtWidgets.QMessageBox.warning(self, "No data", "Load TTTR files first.")
            return
        channel = self._active_channel()
        if channel is None:
            QtWidgets.QMessageBox.warning(self, "No channel selected", "Select a channel from the list.")
            return
        self.channel_luts[int(channel)] = np.array(lut_array, dtype=np.float64)
        self._update_all_curves_full()
        if self.chk_show_lut.isChecked():
            self._update_lut_plot_for_channel(self._active_channel())
        if self.json_dialog.isVisible():
            self._refresh_json_preview()

    def _assign_lut_all(self) -> None:
        """Assign the selected LUT to all used channels."""
        lut_array = self._get_selected_lut()
        if lut_array is None:
            QtWidgets.QMessageBox.warning(self, "No LUT selected", "Select a LUT from the list.")
            return
        if not self.bundle:
            QtWidgets.QMessageBox.warning(self, "No data", "Load TTTR files first.")
            return
        array = np.array(lut_array, dtype=np.float64)
        for channel in self.bundle.used_channels():
            self.channel_luts[int(channel)] = array
        self._update_all_curves_full()
        if self.chk_show_lut.isChecked():
            self._update_lut_plot_for_channel(self._active_channel())
        if self.json_dialog.isVisible():
            self._refresh_json_preview()

    def _toggle_lut_panel(self, checked: bool) -> None:
        """Show or hide the LUT preview panel."""
        self.lut_container.setVisible(bool(checked))
        if checked:
            self._update_lut_plot_for_channel(self._active_channel())

    def _get_base_xy(self, channel: int) -> tuple[np.ndarray, np.ndarray]:
        """Return histogram data for a channel."""
        if self.channel_luts:
            return self.bundle.get_cor(channel, False, self.channel_luts)
        return self.bundle.get_unc(channel, False)

    def _remove_all_curves(self) -> None:
        """Remove all histogram curves."""
        for item in self.curves.values():
            try:
                self.hist_plot.removeItem(item)
            except Exception:
                pass
        self.curves.clear()

    def _update_all_curves_full(self) -> None:
        """Create or update visible channel curves."""
        if not self.bundle:
            self._remove_all_curves()
            return

        draw = set(self._selected_channels())
        if not draw:
            self._remove_all_curves()
            self.status.showMessage("No channels selected for display.")
            return

        if self.channel_luts:
            self.bundle.ensure_corrected_cache(self.channel_luts)

        for channel in list(self.curves.keys()):
            if channel not in draw:
                try:
                    self.hist_plot.removeItem(self.curves[channel])
                except Exception:
                    pass
                self.curves.pop(channel, None)

        any_points = False
        for index, channel in enumerate(sorted(draw)):
            color = self.curve_colors.get(channel, COLOR_CYCLE[index % len(COLOR_CYCLE)])
            x_values, y_values = self._get_base_xy(channel)
            if x_values.size == 0 or y_values.size == 0:
                self.status.showMessage(f"Channel {channel}: empty histogram.")
                continue

            shift = int(self.channel_shifts.get(channel, 0))
            y_display = np.roll(y_values, shift) if shift and len(y_values) > 0 else y_values
            any_points = any_points or np.any(np.isfinite(y_display))

            if channel in self.curves:
                self.curves[channel].setData(x_values, y_display)
                self.curves[channel].setPen(nice_pen(color, width=2))
            else:
                item = self.hist_plot.plot(x_values, y_display, pen=nice_pen(color, width=2))
                item.setZValue(0)
                item.setDownsampling(auto=True)
                item.setClipToView(True)
                self.curves[channel] = item

        self._emphasize_active_curve()
        self._force_range_from_visible()
        if not any_points:
            self.status.showMessage("No finite data points to display (all zeros or NaNs?).")

    def _force_range_from_visible(self) -> None:
        """Set the histogram range from visible curves."""
        x_values_list = []
        y_values_list = []
        for item in self.curves.values():
            data = item.getData()
            if not data:
                continue
            x_values, y_values = data
            if x_values is None or y_values is None:
                continue
            x_values = np.asarray(x_values)
            y_values = np.asarray(y_values)
            mask = np.isfinite(x_values) & np.isfinite(y_values)
            if not np.any(mask):
                continue
            x_values_list.append(x_values[mask])
            y_values_list.append(y_values[mask])

        if not x_values_list or not y_values_list:
            return

        x_min = min(float(values.min()) for values in x_values_list)
        x_max = max(float(values.max()) for values in x_values_list)
        y_min = min(float(values.min()) for values in y_values_list)
        y_max = max(float(values.max()) for values in y_values_list)
        if not np.isfinite([x_min, x_max, y_min, y_max]).all():
            return
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            y_max = y_min + 1.0

        x_range = x_max - x_min
        y_range = y_max - y_min
        self.hist_plot.setXRange(x_min - 0.02 * x_range, x_max + 0.02 * x_range, padding=0)
        self.hist_plot.setYRange(y_min - 0.05 * y_range, y_max + 0.05 * y_range, padding=0)

    def _emphasize_active_curve(self) -> None:
        """Emphasize the active channel curve."""
        channel = self._active_channel()
        for curve_channel, item in self.curves.items():
            color = self.curve_colors.get(curve_channel, (100, 100, 100))
            item.setPen(nice_pen(color, width=2))
            item.setZValue(0)
        if channel in self.curves:
            color = self.curve_colors.get(channel, (0, 0, 0))
            self.curves[channel].setPen(nice_pen(color, width=4))
            self.curves[channel].setZValue(10)

    def _active_channel(self) -> int | None:
        """Return the active channel."""
        item = self.channel_list.currentItem()
        if not item:
            return None
        return int(item.data(QtCore.Qt.UserRole))

    def _on_active_channel_changed(self, _row: int) -> None:
        """Update controls when the active channel changes."""
        channel = self._active_channel()
        if channel is None:
            self.shift_spin.setEnabled(False)
            return
        self.shift_spin.setEnabled(True)
        self.shift_spin.blockSignals(True)
        self.shift_spin.setValue(int(self.channel_shifts.get(channel, 0)))
        self.shift_spin.blockSignals(False)
        self._emphasize_active_curve()
        if self.lut_container.isVisible():
            self._update_lut_plot_for_channel(channel)

    def _on_reading_routine_changed(self, index: int) -> None:
        """Reload files when the reading routine changes."""
        self.reading_routine = self.reading_combo.itemData(index)
        paths = self.file_list.current_paths()
        if paths:
            self._files_changed()

    def _selected_channels(self) -> list[int]:
        """Return checked channels."""
        selected = []
        for index in range(self.channel_list.count()):
            item = self.channel_list.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                selected.append(int(item.data(QtCore.Qt.UserRole)))
        return selected

    def _on_shift_changed_live(self, value: int) -> None:
        """Roll only the active curve when the shift changes."""
        channel = self._active_channel()
        if channel is None or not self.bundle:
            return
        self.channel_shifts[int(channel)] = int(value)

        item = self.curves.get(channel)
        if item is not None:
            x_values, y_values = self._get_base_xy(channel)
            shift = int(value)
            y_display = np.roll(y_values, shift) if shift and len(y_values) > 0 else y_values
            item.setData(x_values, y_display)
            self._emphasize_active_curve()
            self._force_range_from_visible()

    def _on_channel_visibility_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        """Update curves when channel visibility changes."""
        self._update_all_curves_full()

    def _update_lut_plot_for_channel(self, channel: int | None) -> None:
        """Update the selected-channel LUT preview."""
        self.lut_plot_cum.clear()
        self.lut_plot_delta.clear()
        if channel is None:
            return
        lut = self.channel_luts.get(int(channel))
        if lut is None or len(lut) == 0:
            self.lut_plot_cum.addItem(pg.TextItem("No LUT for this channel", anchor=(0, 0)))
            return
        x_values = np.arange(len(lut), dtype=np.float32)
        self.lut_plot_cum.plot(x_values, f32(lut), pen=nice_pen((70, 70, 200), 2))
        delta = np.diff(np.concatenate(([0.0], lut)))
        self.lut_plot_delta.plot(x_values, f32(delta), pen=nice_pen((200, 70, 70), 2))

    def _current_settings_dict(self) -> dict[str, object]:
        """Return the current settings dictionary."""
        used = self.bundle.used_channels() if self.bundle else []
        return {
            "description": "TTTR microtime correction settings",
            "version": "1.0",
            "reading_routine": self.reading_routine,
            "channel_luts": {
                int(channel): np.asarray(array).tolist()
                for channel, array in self.channel_luts.items()
            },
            "channel_shifts": {int(channel): int(shift) for channel, shift in self.channel_shifts.items()},
            "metadata": {
                "created": str(np.datetime64("now")),
                "used_channels": [int(channel) for channel in used],
                "notes": "Generated by TTTR LUT Tools plugin",
            },
        }

    def _toggle_json_preview(self, checked: bool) -> None:
        """Show or hide the JSON preview dialog."""
        if checked:
            self._refresh_json_preview()
            self.json_dialog.show()
            self.json_dialog.raise_()
            self.json_dialog.activateWindow()
        else:
            self.json_dialog.hide()

    def _toggle_readme(self, checked: bool) -> None:
        """Show or hide the help dialog."""
        if checked:
            self.readme_dialog.show()
            self.readme_dialog.raise_()
            self.readme_dialog.activateWindow()
        else:
            self.readme_dialog.hide()

    def _refresh_json_preview(self) -> None:
        """Refresh JSON preview text."""
        try:
            text = json.dumps(json_safe(self._current_settings_dict()), indent=2)
        except Exception as exc:
            text = f"<!> Error rendering JSON:\n{exc}"
        self.json_dialog.set_json(text)

    def _load_settings(self) -> None:
        """Load settings from JSON."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load settings.tttr.json",
            "",
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return

        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("Invalid JSON structure: expected object")

            version = data.get("version", "1.0")
            if version != "1.0":
                QtWidgets.QMessageBox.warning(
                    self,
                    "Version Warning",
                    f"Settings file version {version} may not be fully compatible with this version.",
                )

            reading_routine = data.get("reading_routine")
            if reading_routine is not None:
                for index in range(self.reading_combo.count()):
                    if self.reading_combo.itemData(index) == reading_routine:
                        self.reading_combo.setCurrentIndex(index)
                        break

            channel_luts = data.get("channel_luts", {})
            if channel_luts:
                self.channel_luts.clear()
                for channel_text, lut_list in channel_luts.items():
                    channel = int(channel_text)
                    self.channel_luts[channel] = np.array(lut_list, dtype=np.float64)

            channel_shifts = data.get("channel_shifts", {})
            if channel_shifts:
                self.channel_shifts.clear()
                for channel_text, shift_value in channel_shifts.items():
                    self.channel_shifts[int(channel_text)] = int(shift_value)

            self._update_all_curves_full()
            if self.chk_show_lut.isChecked():
                self._update_lut_plot_for_channel(self._active_channel())
            if self.json_dialog.isVisible():
                self._refresh_json_preview()

            QtWidgets.QMessageBox.information(
                self,
                "Settings Loaded",
                "Successfully loaded settings from:\n"
                f"{path}\n\n"
                f"LUTs: {len(channel_luts)} channels\n"
                f"Shifts: {len(channel_shifts)} channels",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load error", f"Failed to load settings:\n{exc}")

    def _save_settings(self) -> None:
        """Save settings JSON."""
        if not self.bundle:
            QtWidgets.QMessageBox.warning(self, "No data", "Load TTTR files first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save settings.tttr.json",
            "settings.tttr.json",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(json_safe(self._current_settings_dict()), handle, indent=2)
            QtWidgets.QMessageBox.information(self, "Saved", f"Settings saved to:\n{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save error", str(exc))
