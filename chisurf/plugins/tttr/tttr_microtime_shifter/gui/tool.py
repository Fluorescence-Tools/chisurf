"""Micro-time Shifter GUI using DockArea (burst_selection pattern)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.dock_area.dock_area import DockArea
from chisurf.gui.widgets.tools import ChisurfDockTool
from chisurf.gui.widgets.tools import PathDropListWidget as DropListWidget

from .client import MicrotimeShifterClient


class MicrotimeShifterTool(ChisurfDockTool):
    """Micro-time Shifter with DockArea tabs and toolbar."""

    tool_settings_name = "MicrotimeShifterTool"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Micro-time Shifter")
        self.resize(1000, 500)
        self._client = MicrotimeShifterClient()

        # state
        self._file_paths: list[Path] = []
        self._current_path: str | None = None
        self.setAcceptDrops(True)
        self._routing_channels: list[int] = []
        self._n_mt: int = 0
        self._global_shift: int = 0
        self._channel_shifts: dict[int, int] = {}
        self._orig_mt: np.ndarray | None = None
        self._routing: np.ndarray | None = None
        self._mfdb_status: str = ""

        # Trigger / Auto-align state
        self._trigger_level: int = 0
        self._trigger_pos: int = 0

        self._create_widgets()
        self._build_docks()
        self._setup_toolbar()
        self._setup_statusbar()
        self._restore_window_geometry()

    def _create_widgets(self) -> None:
        """Create plot, control, and files list widgets."""
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Micro-time bin")
        self.plot.setLabel("left", "Counts")
        self.plot.setTitle("Micro-time Histograms")

        # Create trigger lines (initially visible, movable)
        self.trigger_level_line = pg.InfiniteLine(
            angle=0,
            movable=True,
            pos=0.0,
            pen=pg.mkPen('y', style=QtCore.Qt.PenStyle.DashLine, width=2)
        )
        self.trigger_pos_line = pg.InfiniteLine(
            angle=90,
            movable=True,
            pos=0.0,
            pen=pg.mkPen('g', style=QtCore.Qt.PenStyle.DashLine, width=2)
        )
        self.plot.addItem(self.trigger_level_line)
        self.plot.addItem(self.trigger_pos_line)

        # Connect line signals
        self.trigger_level_line.sigPositionChanged.connect(self._on_trigger_level_line_changed)
        self.trigger_pos_line.sigPositionChanged.connect(self._on_trigger_pos_line_changed)
        self.trigger_level_line.sigPositionChangeFinished.connect(self._on_trigger_level_line_finished)
        self.trigger_pos_line.sigPositionChangeFinished.connect(self._on_trigger_pos_line_finished)

        self.controls_panel = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(self.controls_panel)
        self.controls_layout.setContentsMargins(2, 2, 2, 2)
        self.controls_layout.setSpacing(2)

        controls_header = QtWidgets.QLabel("Micro-time Shift")
        controls_header.setObjectName("microtimeShiftHeader")
        controls_header.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px;"
        )
        self.controls_layout.addWidget(controls_header)

        # Trigger / Auto-align controls (instantiated here, added to toolbar)
        self.trigger_level_spin = pg.SpinBox(value=0, int=True, step=10, bounds=[0, 1000000])
        self.trigger_level_spin.setMinimumWidth(80)
        self.trigger_level_spin.setMaximumWidth(120)

        self.trigger_pos_spin = pg.SpinBox(value=0, int=True, step=1, bounds=[0, 100000])
        self.trigger_pos_spin.setMinimumWidth(60)
        self.trigger_pos_spin.setMaximumWidth(100)

        # Connect spin box signals
        self.trigger_level_spin.editingFinished.connect(self._on_trigger_level_spin_changed)
        self.trigger_pos_spin.editingFinished.connect(self._on_trigger_pos_spin_changed)

        # Container layout for channel shifts
        self.shifts_container = QtWidgets.QWidget()
        self.shifts_layout = QtWidgets.QVBoxLayout(self.shifts_container)
        self.shifts_layout.setContentsMargins(0, 0, 0, 0)
        self.shifts_layout.setSpacing(2)
        self.controls_layout.addWidget(self.shifts_container)

        self.plot_panel = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_panel)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_header = QtWidgets.QLabel("Histogram")
        plot_header.setObjectName("histogramHeader")
        plot_header.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px;"
        )
        self.plot_layout.addWidget(plot_header)
        self.plot_layout.addWidget(self.plot, 1)

        self.status_panel = QtWidgets.QWidget()
        self.status_layout = QtWidgets.QVBoxLayout(self.status_panel)
        self.status_layout.setContentsMargins(2, 2, 2, 2)
        self.status_label = QtWidgets.QLabel("No file loaded.")
        self.status_label.setWordWrap(True)
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addStretch()

        self.files_panel = QtWidgets.QWidget()
        self.files_layout = QtWidgets.QVBoxLayout(self.files_panel)
        self.files_layout.setContentsMargins(2, 2, 2, 2)
        self.files_layout.setSpacing(4)

        files_header = QtWidgets.QLabel("Files")
        files_header.setObjectName("filesHeader")
        files_header.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px;"
        )
        self.files_layout.addWidget(files_header)

        drop_hint = QtWidgets.QLabel("Drop files or folders here")
        drop_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        drop_hint.setStyleSheet("color: gray; font-style: italic; padding: 4px;")
        self.files_layout.addWidget(drop_hint)

        self.file_list = DropListWidget(self.files_panel)
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.file_list.pathsDropped.connect(self._add_paths)
        self.file_list.itemSelectionChanged.connect(self._on_file_selected)
        self.file_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._show_file_list_context_menu)
        self.files_layout.addWidget(self.file_list, 1)

    def _build_docks(self) -> None:
        """Build DockArea with tabs for files, controls, histogram, and status."""
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.dock_area = DockArea(central)
        self.dock_area.setContextMenuEnabled(True)
        self.dock_area.setContextMenuMode("basic")
        self.dock_area.setTabsClosable(True)
        self.dock_area.setCloseTabCallback(self._on_dock_tab_close_requested)
        self.dock_area.setContextMenuCallback(self._add_dock_context_menu_actions)

        self.dock_area.addTab(self.files_panel, "Files")
        self.dock_area.addTab(self.controls_panel, "Micro-time Shift")
        self.dock_area.addTab(self.plot_panel, "Histogram")
        self.dock_area.addTab(self.status_panel, "Status")
        layout.addWidget(self.dock_area, 1)

        self.dock_area.layoutChanged.connect(self._save_dock_layout)
        self._restore_dock_layout()

    def _setup_toolbar(self) -> None:
        """Create toolbar with Load/Save actions."""
        tb = QtWidgets.QToolBar("Main")
        tb.setObjectName("microtimeShifterMainToolbar")
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QtCore.QSize(16, 16))
        tb.setContentsMargins(4, 2, 4, 2)
        if tb.layout() is not None:
            tb.layout().setSpacing(6)
        tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.setStyleSheet("""
            QToolBar#microtimeShifterMainToolbar {
                background-color: transparent;
                border: none;
                padding: 3px 4px;
                spacing: 6px;
            }
            QToolBar#microtimeShifterMainToolbar QToolButton {
                background-color: #2a2a4a;
                border: 1px solid #5a5a8a;
                border-radius: 5px;
                padding: 5px 10px;
                margin: 0px;
                font-weight: bold;
                font-size: 12px;
                color: #ffffff;
            }
            QToolBar#microtimeShifterMainToolbar QToolButton:hover {
                background-color: #3a3a6a;
                border-color: #8a8aba;
            }
            QToolBar#microtimeShifterMainToolbar QToolButton:pressed {
                background-color: #4a4a8a;
            }
            QToolBar#microtimeShifterMainToolbar QToolButton:disabled {
                color: #666;
                background-color: #1a1a2e;
                border-color: #3a3a5e;
            }
            QToolBar#microtimeShifterMainToolbar QToolButton:checked {
                background-color: #4a4a8a;
                border-color: #a0a0ff;
            }
            QToolBar#microtimeShifterMainToolbar QLabel {
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
                margin-left: 4px;
                margin-right: 2px;
            }
        """)

        load_action = QtWidgets.QAction("📂 Load...", self)
        load_action.setObjectName("microtimeShifterLoad")
        load_action.triggered.connect(self._on_load)
        tb.addAction(load_action)

        tb.addSeparator()

        self.save_action = QtWidgets.QAction("💾 Save...", self)
        self.save_action.setObjectName("microtimeShifterSave")
        self.save_action.setEnabled(False)
        self.save_action.triggered.connect(self._open_save_dialog)
        tb.addAction(self.save_action)

        tb.addSeparator()

        self.show_trigger_action = QtWidgets.QAction("🎯 Show Trigger", self, checkable=True)
        self.show_trigger_action.setObjectName("microtimeShifterShowTrigger")
        self.show_trigger_action.setChecked(True)
        self.show_trigger_action.toggled.connect(self._on_toggle_trigger_lines)
        tb.addAction(self.show_trigger_action)

        self.logy_action = QtWidgets.QAction("📈 Log Y", self, checkable=True)
        self.logy_action.setObjectName("microtimeShifterLogY")
        self.logy_action.setChecked(False)
        self.logy_action.toggled.connect(self._on_toggle_logy)
        tb.addAction(self.logy_action)

        tb.addSeparator()

        lbl_level = QtWidgets.QLabel("Level:")
        tb.addWidget(lbl_level)
        tb.addWidget(self.trigger_level_spin)

        lbl_pos = QtWidgets.QLabel("Pos:")
        tb.addWidget(lbl_pos)
        tb.addWidget(self.trigger_pos_spin)

        tb.addSeparator()

        self.auto_align_action = QtWidgets.QAction("⚡ Auto Align", self)
        self.auto_align_action.setObjectName("microtimeShifterAutoAlign")
        self.auto_align_action.triggered.connect(self.auto_align)
        tb.addAction(self.auto_align_action)

    def _setup_statusbar(self) -> None:
        self.statusBar().showMessage("Ready")

    # ── file loading ───────────────────────────────────────────────

    def _on_load(self) -> None:
        """Load TTTR files — from MFDB if connected, otherwise file dialog."""
        db = self._db()
        if db is not None:
            try:
                from chisurf.gui.widgets.mfdb.dataset_browser import (
                    MfdbDatasetPickerDialog,
                )
                from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

                client = MFDBClient(inprocess=True)
                sel = MfdbDatasetPickerDialog.pick_dataset(
                    parent=self,
                    # Both raw measurements and shifted (processed) TTTR outputs
                    # are loadable; the format filter keeps it to TTTR files.
                    kinds=["raw_measurement", "processed_data"],
                    formats=["spc", "ptu", "ht3", "hdf", "h5"],
                    scope="mine",
                    client=client,
                )
            except Exception as exc:
                self.statusBar().showMessage(f"MFDB picker error: {exc}")
                return

            if sel is None:
                return

            try:
                result = client.call("mfdb.datasets.open", {"artifact_id": sel.artifact_id})
                local_path = (result or {}).get("local_path")
                if not local_path:
                    self.statusBar().showMessage(
                        f"Cannot open dataset {sel.artifact_id}: no local path"
                    )
                    return
                self._add_paths([Path(local_path)])
                self.statusBar().showMessage(
                    f"Loaded from MFDB: {sel.artifact_id[:16]}..."
                )
                return
            except Exception as exc:
                self.statusBar().showMessage(f"MFDB open error: {exc}")
                return

        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open TTTR files", "", "TTTR Files (*.*)"
        )
        if paths:
            self._add_paths([Path(p) for p in paths])

    def _on_file_path(self, path: str) -> None:
        if not path:
            return
        self._current_path = path
        self._load_metadata()
        self._identify_file()
        self._build_shift_controls()
        self._update_plot()

    def _add_paths(self, paths: list[Path]) -> None:
        """Add files and folders to the file list."""
        for path in paths:
            if path.is_dir():
                self._file_paths.extend(
                    sorted(
                        child.resolve()
                        for child in path.iterdir()
                        if child.is_file() and child.suffix.lower() in {".spc", ".ht3", ".ptu", ".hdf", ".h5"}
                    )
                )
            else:
                self._file_paths.append(path.resolve())
        # Remove duplicates while preserving order
        seen = set()
        self._file_paths = [p for p in self._file_paths if not (p in seen or seen.add(p))]
        self._refresh_file_list()

    def _refresh_file_list(self) -> None:
        """Refresh the files in the QListWidget."""
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for path in self._file_paths:
            self.file_list.addItem(str(path))
        self.file_list.blockSignals(False)

        if self._file_paths:
            # If no active file, or active file not in list, select first
            if not self._current_path or Path(self._current_path) not in self._file_paths:
                self.file_list.setCurrentRow(0)
            else:
                # Sync row selection to match self._current_path
                for i in range(self.file_list.count()):
                    item = self.file_list.item(i)
                    if item and item.text() == self._current_path:
                        self.file_list.setCurrentRow(i)
                        break

    def _on_file_selected(self) -> None:
        """Handle selection change in the file list."""
        selected = self.file_list.selectedItems()
        if not selected:
            return
        path = selected[0].text()
        if path != self._current_path:
            self._on_file_path(path)

    def _show_file_list_context_menu(self, pos: QtCore.QPoint) -> None:
        """Show context menu for file list."""
        menu = QtWidgets.QMenu(self)

        remove_action = QtWidgets.QAction("Remove selected", self)
        remove_action.triggered.connect(self._remove_selected_files)
        menu.addAction(remove_action)

        clear_action = QtWidgets.QAction("Clear all", self)
        clear_action.triggered.connect(self._clear_file_list)
        menu.addAction(clear_action)

        menu.exec_(self.file_list.mapToGlobal(pos))

    def _remove_selected_files(self) -> None:
        """Remove selected files from the file list."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            path = Path(item.text())
            if path in self._file_paths:
                self._file_paths.remove(path)
        self._refresh_file_list()
        if not self._file_paths:
            self._current_path = None
            self.plot.clear()
            self.save_action.setEnabled(False)
            self._update_status()

    def _clear_file_list(self) -> None:
        """Clear all files from the file list."""
        self._file_paths.clear()
        self._refresh_file_list()
        self._current_path = None
        self.plot.clear()
        self.save_action.setEnabled(False)
        self._update_status()

    def _load_metadata(self) -> None:
        if not self._current_path:
            return
        try:
            meta = self._client.load_metadata(Path(self._current_path))
            self._routing_channels = meta.get("routing_channels", [])
            self._n_mt = int(meta.get("n_mt", 0))
            self._channel_shifts = {ch: 0 for ch in self._routing_channels}
            self._global_shift = 0

            # Calculate sensible defaults for trigger levels and positions
            histogram = self._client.histogram(
                self._file_paths,
                global_shift=self._global_shift,
                channel_shifts=self._channel_shifts,
            )
            max_peak = 0
            for hist_values in histogram.get("histograms", {}).values():
                if hist_values:
                    max_peak = max(max_peak, int(max(hist_values)))

            self._trigger_level = int(max_peak * 0.2) if max_peak > 0 else 100
            self._trigger_pos = int(self._n_mt * 0.1) if self._n_mt > 0 else 50

            # Update controls and lines
            self.trigger_level_spin.blockSignals(True)
            self.trigger_level_spin.setOpts(bounds=[0, max(1000000, max_peak)])
            self.trigger_level_spin.setValue(self._trigger_level)
            self.trigger_level_spin.blockSignals(False)

            self.trigger_pos_spin.blockSignals(True)
            self.trigger_pos_spin.setOpts(bounds=[0, self._n_mt - 1])
            self.trigger_pos_spin.setValue(self._trigger_pos)
            self.trigger_pos_spin.blockSignals(False)

            self.trigger_level_line.blockSignals(True)
            self.trigger_level_line.setValue(self._trigger_level)
            self.trigger_level_line.blockSignals(False)

            self.trigger_pos_line.blockSignals(True)
            self.trigger_pos_line.setValue(self._trigger_pos)
            self.trigger_pos_line.blockSignals(False)

            self.save_action.setEnabled(True)
            self.statusBar().showMessage(f"Loaded: {self._current_path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Cannot load file:\n{exc}"
            )

    def _identify_file(self) -> None:
        if not self._current_path:
            return
        try:
            info = self._client.identify(Path(self._current_path))
            if info.get("found"):
                self._mfdb_status = (
                    f"Identified (artifact: {info.get('artifact_id', '?')[:8]})"
                )
            else:
                self._mfdb_status = "New file (not in MFDB)"
        except Exception:
            self._mfdb_status = "MFDB check unavailable"
        self._update_status()

    def _update_status(self) -> None:
        parts = []
        if self._current_path:
            parts.append(f"File: {self._current_path}")
        if self._mfdb_status:
            parts.append(f"MFDB: {self._mfdb_status}")
        if not self._current_path:
            parts.append("No file loaded.")
        self.status_label.setText("\n".join(parts))

    # ── shift controls ─────────────────────────────────────────────

    def _build_shift_controls(self) -> None:
        while self.shifts_layout.count():
            w = self.shifts_layout.takeAt(0).widget()
            if w:
                w.setParent(None)

        for ch in sorted(self._channel_shifts):
            self._add_shift_row(f"Ch{ch}", ch)
        self.shifts_layout.addStretch()

    def _add_shift_row(self, label: str, channel: int | None) -> None:
        row = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        lbl = QtWidgets.QLabel(label)
        lbl.setFixedWidth(30)
        rl.addWidget(lbl)

        value = (
            self._global_shift
            if channel is None
            else self._channel_shifts.get(channel, 0)
        )
        spin = pg.SpinBox(
            value=value,
            int=True,
            step=1,
            bounds=[-(self._n_mt - 1), self._n_mt - 1],
        )
        spin.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        rl.addWidget(spin)

        if channel is None:
            spin.editingFinished.connect(self._make_global_fn(spin))
        else:
            spin.editingFinished.connect(self._make_chan_fn(channel, spin))

        btn = QtWidgets.QPushButton("↻")
        btn.setFixedWidth(30)
        if channel is None:
            btn.clicked.connect(self._make_global_reset_fn(spin))
        else:
            btn.clicked.connect(self._make_chan_reset_fn(channel, spin))
        rl.addWidget(btn)

        self.shifts_layout.addWidget(row)

    def _make_global_fn(self, spin: pg.SpinBox):
        def fn() -> None:
            try:
                self._global_shift = int(spin.value())
            except Exception:
                return
            self._update_plot()
        return fn

    def _make_chan_fn(self, ch: int, spin: pg.SpinBox):
        def fn() -> None:
            try:
                self._channel_shifts[ch] = int(spin.value())
            except Exception:
                return
            self._update_plot()
        return fn

    def _make_global_reset_fn(self, spin: pg.SpinBox):
        def fn() -> None:
            self._global_shift = 0
            spin.blockSignals(True)
            spin.setValue(0)
            spin.blockSignals(False)
            self._update_plot()
        return fn

    def _make_chan_reset_fn(self, ch: int, spin: pg.SpinBox):
        def fn() -> None:
            self._channel_shifts[ch] = 0
            spin.blockSignals(True)
            spin.setValue(0)
            spin.blockSignals(False)
            self._update_plot()
        return fn

    # ── plot ───────────────────────────────────────────────────────

    def _update_plot(self) -> None:
        """Update the micro-time histogram plot via RPC."""
        if not self._file_paths or self._n_mt < 1:
            return

        try:
            histogram = self._client.histogram(
                self._file_paths,
                global_shift=self._global_shift,
                channel_shifts=self._channel_shifts,
            )
        except Exception:
            self.plot.clear()
            self.plot.setTitle("Cannot load files for preview")
            return

        self.plot.clear()
        # Add back trigger lines
        self.plot.addItem(self.trigger_level_line)
        self.plot.addItem(self.trigger_pos_line)

        edges = np.arange(self._n_mt + 1)

        is_logy = self.logy_action.isChecked()
        for i, ch in enumerate(sorted(self._channel_shifts)):
            hist_values = histogram.get("histograms", {}).get(str(ch), [])
            if not hist_values:
                continue
            hist = np.array(hist_values, dtype=int)
            c = pg.intColor(i, len(self._channel_shifts))
            if is_logy:
                self.plot.plot(
                    edges, hist, pen=c, stepMode=True, name=str(ch)
                )
            else:
                c.setAlpha(100)
                self.plot.plot(
                    edges, hist, pen=c, stepMode=True,
                    fillLevel=0, brush=c, name=str(ch)
                )

        try:
            self.plot.addLegend()
        except Exception:
            pass
        self.plot.setTitle("Micro-time Histograms (preview)")

    def auto_align(self) -> None:
        if not self._file_paths or self._n_mt < 1:
            return

        target_bin = int(self._trigger_pos)
        trigger_level = int(self._trigger_level)

        try:
            histogram = self._client.histogram(
                self._file_paths,
                global_shift=self._global_shift,
                channel_shifts=self._channel_shifts,
            )
        except Exception:
            return

        for ch in self._channel_shifts:
            hist_values = histogram.get("histograms", {}).get(str(ch), [])
            if not hist_values:
                continue
            hist = np.array(hist_values, dtype=int)
            peak_bin = int(np.argmax(hist))
            peak_val = hist[peak_bin]

            # Find rising edge crossing the trigger level
            below_peak = hist[:peak_bin + 1]
            crossings = np.where(below_peak >= trigger_level)[0]
            if len(crossings) > 0:
                ch_trigger_bin = crossings[0]
            else:
                ch_trigger_bin = peak_bin

            # Calculate required shift
            current_shift = self._channel_shifts.get(ch, 0)
            shift = (current_shift + target_bin - ch_trigger_bin) % self._n_mt
            self._channel_shifts[ch] = int(shift)

        # Update the UI spinboxes for the channel shifts
        self._build_shift_controls()
        self._update_plot()

    def _on_trigger_level_spin_changed(self, *args: object) -> None:
        val = int(self.trigger_level_spin.value())
        val = max(1, val)
        self._trigger_level = val
        self.trigger_level_line.blockSignals(True)
        if self.logy_action.isChecked():
            self.trigger_level_line.setValue(np.log10(val))
        else:
            self.trigger_level_line.setValue(val)
        self.trigger_level_line.blockSignals(False)
        self.auto_align()

    def _on_trigger_pos_spin_changed(self, *args: object) -> None:
        val = int(self.trigger_pos_spin.value())
        self._trigger_pos = val
        self.trigger_pos_line.blockSignals(True)
        self.trigger_pos_line.setValue(val)
        self.trigger_pos_line.blockSignals(False)
        self.auto_align()

    def _on_trigger_level_line_changed(self, *args: object) -> None:
        line_val = self.trigger_level_line.value()
        if self.logy_action.isChecked():
            val = int(round(10**line_val))
        else:
            val = int(round(line_val))
        val = max(1, val)
        self._trigger_level = val
        self.trigger_level_spin.blockSignals(True)
        self.trigger_level_spin.setValue(val)
        self.trigger_level_spin.blockSignals(False)

    def _on_trigger_pos_line_changed(self, *args: object) -> None:
        val = int(self.trigger_pos_line.value())
        self._trigger_pos = val
        self.trigger_pos_spin.blockSignals(True)
        self.trigger_pos_spin.setValue(val)
        self.trigger_pos_spin.blockSignals(False)

    def _on_trigger_level_line_finished(self, *args: object) -> None:
        self.auto_align()

    def _on_trigger_pos_line_finished(self, *args: object) -> None:
        self.auto_align()

    def _on_toggle_trigger_lines(self, checked: bool) -> None:
        self.trigger_level_line.setVisible(checked)
        self.trigger_pos_line.setVisible(checked)

    def _on_toggle_logy(self, checked: bool) -> None:
        self.plot.getPlotItem().setLogMode(False, checked)
        self.trigger_level_line.blockSignals(True)
        if checked:
            self.trigger_level_line.setValue(np.log10(max(1, self._trigger_level)))
        else:
            self.trigger_level_line.setValue(self._trigger_level)
        self.trigger_level_line.blockSignals(False)
        self._update_plot()
        self.plot.getPlotItem().vb.autoRange()

    def acquire_mfdb_connection(self) -> Any:
        """Return the active MFDB connection (PRD-23 base hook)."""
        from ..api.mfdb import active_mfdb_connection

        return active_mfdb_connection()

    def _db(self) -> Any:
        """Return the active MFDB connection if available."""
        return self.acquire_mfdb_connection()

    # ── save ───────────────────────────────────────────────────────

    def _open_save_dialog(self) -> None:
        if not self._file_paths:
            return
        paths_to_shift = self._file_paths

        db = self._db()
        has_mfdb = db is not None

        mode = "file"
        if has_mfdb:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setWindowTitle("Save Shifted Files")
            msg_box.setText("An active MFDB database connection was found.")
            msg_box.setInformativeText("Would you like to register the shifted files in the database, or save them to a local file/folder?")
            
            btn_register = msg_box.addButton("Register in DB", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            btn_save = msg_box.addButton("Save to File/Folder...", QtWidgets.QMessageBox.ButtonRole.ApplyRole)
            btn_cancel = msg_box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            
            msg_box.exec_()
            clicked = msg_box.clickedButton()
            if clicked == btn_cancel:
                return
            elif clicked == btn_register:
                mode = "db"
            else:
                mode = "file"

        if mode == "db":
            from chisurf.gui.widgets.sample_picker import show_sample_picker_dialog
            sample_id = show_sample_picker_dialog(db=db, parent=self)
            if not sample_id:
                return

            # Prepare the MFDB Context
            mfdb_context = {
                "enabled": True,
                "sample_id": sample_id,
                "register_missing_inputs": True,
            }

            try:
                result = self._client.apply(
                    file_paths=paths_to_shift,
                    global_shift=self._global_shift,
                    channel_shifts=self._channel_shifts,
                    mfdb=mfdb_context,
                )
                warnings = result.get("warnings", [])
                warn_str = "\nWarnings:\n" + "\n".join(warnings) if warnings else ""
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Successfully registered {len(paths_to_shift)} file(s) in MFDB.{warn_str}"
                )
                self.statusBar().showMessage(f"Registered in MFDB: {len(paths_to_shift)} file(s)")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to register in MFDB:\n{exc}"
                )

        else:  # mode == "file"
            if len(paths_to_shift) == 1:
                path = paths_to_shift[0]
                d = path.parent
                f = path.name
                sp, _ = QtWidgets.QFileDialog.getSaveFileName(
                    self, "Save shifted TTTR file", str(d / f), "TTTR Files (*.*)"
                )
                if not sp:
                    return
                try:
                    result = self._client.apply(
                        file_paths=paths_to_shift,
                        global_shift=self._global_shift,
                        channel_shifts=self._channel_shifts,
                        output_dir=Path(sp).parent,
                        mfdb={"enabled": False},
                    )
                    shifted_generated = result.get("output_paths_by_file", {}).get(str(path))
                    saved_path = sp
                    if shifted_generated and shifted_generated != sp:
                        try:
                            import shutil
                            import os
                            if os.path.exists(sp):
                                os.remove(sp)
                            shutil.move(shifted_generated, sp)
                        except Exception:
                            saved_path = shifted_generated
                    self.statusBar().showMessage(f"Saved to: {saved_path}")
                    QtWidgets.QMessageBox.information(
                        self, "Saved", f"Saved to:\n{saved_path}"
                    )
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(
                        self, "Error", f"Cannot save:\n{exc}"
                    )
            else:
                output_dir = QtWidgets.QFileDialog.getExistingDirectory(
                    self, "Select Output Directory", ""
                )
                if not output_dir:
                    return
                try:
                    result = self._client.apply(
                        file_paths=paths_to_shift,
                        global_shift=self._global_shift,
                        channel_shifts=self._channel_shifts,
                        output_dir=Path(output_dir),
                        mfdb={"enabled": False},
                    )
                    self.statusBar().showMessage(f"Saved {len(paths_to_shift)} file(s) to: {output_dir}")
                    QtWidgets.QMessageBox.information(
                        self, "Saved", f"Saved {len(paths_to_shift)} file(s) to:\n{output_dir}"
                    )
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(
                        self, "Error", f"Cannot save:\n{exc}"
                    )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Save window geometry and dock layout before closing."""
        self._save_window_geometry()
        self._save_dock_layout()
        super().closeEvent(event)

    def _save_window_geometry(self) -> None:
        """Save the main window geometry to QSettings."""
        try:
            settings = QtCore.QSettings("chisurf", "MicrotimeShifterTool")
            settings.setValue("geometry", self.saveGeometry())
            settings.sync()
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to save window geometry: {exc}")

    def _restore_window_geometry(self) -> None:
        """Restore the main window geometry from QSettings."""
        try:
            settings = QtCore.QSettings("chisurf", "MicrotimeShifterTool")
            geometry = settings.value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to restore window geometry: {exc}")

    def _save_dock_layout(self) -> None:
        """Save the current dock layout to QSettings."""
        try:
            import json
            settings = QtCore.QSettings("chisurf", "MicrotimeShifterTool")
            layout_state = self.dock_area.get_layout_state()
            settings.setValue("dock_layout", json.dumps(layout_state, sort_keys=True))
            settings.sync()
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to save dock layout: {exc}")

    def _restore_dock_layout(self) -> None:
        """Restore the dock layout from QSettings, or load default."""
        try:
            import json
            settings = QtCore.QSettings("chisurf", "MicrotimeShifterTool")
            value = settings.value("dock_layout")
            if isinstance(value, str):
                layout_state = json.loads(value)
                if self.dock_area.set_layout_state(layout_state, emit_change=False):
                    return
            elif isinstance(value, dict):
                if self.dock_area.set_layout_state(value, emit_change=False):
                    return
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to restore dock layout: {exc}")

        # Default layout fallback: split controls and status (left) from plot (right)
        try:
            default_layout = {
                "version": 1,
                "root": {
                    "type": "splitter",
                    "orientation": "horizontal",
                    "sizes": [300, 700],
                    "children": [
                        {
                            "type": "tab",
                            "current_index": 0,
                            "tabs": [
                                {
                                    "widget_key": "Micro-time Shift",
                                    "tab_name": "Micro-time Shift",
                                    "tab_text": "Micro-time Shift"
                                },
                                {
                                    "widget_key": "Status",
                                    "tab_name": "Status",
                                    "tab_text": "Status"
                                }
                            ]
                        },
                        {
                            "type": "tab",
                            "current_index": 0,
                            "tabs": [
                                {
                                    "widget_key": "Histogram",
                                    "tab_name": "Histogram",
                                    "tab_text": "Histogram"
                                }
                            ]
                        }
                    ]
                },
                "active_tab_widget": None,
                "current_index": 0
            }
            self.dock_area.set_layout_state(default_layout, emit_change=False)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to set default dock layout: {exc}")

    def _on_dock_tab_close_requested(self, index: int) -> None:
        """Hide closed dock tabs instead of deleting them."""
        self.dock_area.hideTab(index)

    def _add_dock_context_menu_actions(self, menu: QtWidgets.QMenu, index: int) -> None:
        """Add context menu actions to restore hidden tabs."""
        hidden_names = []
        for widget in [self.controls_panel, self.plot_panel, self.status_panel]:
            idx = self.dock_area.indexOf(widget)
            if idx != -1 and not self.dock_area.isTabVisible(idx):
                hidden_names.append((self.dock_area.tabText(idx), widget))

        if hidden_names:
            menu.addSeparator()
            show_menu = menu.addMenu("Reopen closed docks")
            for name, widget in hidden_names:
                action = show_menu.addAction(name)
                action.triggered.connect(
                    lambda _checked=False, w=widget: self._show_dock(w)
                )

    def _show_dock(self, widget: QtWidgets.QWidget) -> None:
        """Show a hidden dock widget."""
        idx = self.dock_area.indexOf(widget)
        if idx != -1:
            self.dock_area.showTab(idx)
