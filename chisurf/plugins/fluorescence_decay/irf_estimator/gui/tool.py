from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from chisurf.gui.widgets.dock_area.dock_area import DockArea, DockSplitter

from ..api.models import IRFEstimationSettings
from ..core.estimation import estimate_irf as _estimate_irf


class IRFEstimatorTool(QtWidgets.QMainWindow):
    """Main window for IRF estimation with dock-based layout."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowTitle("IRF Estimator - Blind IRF Estimation")

        # Data storage
        self.decay_data: np.ndarray | None = None
        self.decay_data_original: np.ndarray | None = None
        self.channel_axis: np.ndarray | None = None
        self.dt: float = 1.0
        self.irf_data: np.ndarray | None = None
        self.irf_params: dict[str, float] | None = None
        self.current_file_path: str | None = None
        self.current_dataset: Any = None
        self.all_decays: list[np.ndarray] | None = None
        self.manual_background: float = 0.0
        self.use_range_selection: bool = False
        self.range_bounds: list[float] = [0.0, 100.0]
        self.auto_update_enabled: bool = False
        self.is_estimating: bool = False
        self._building_ui: bool = True

        self._create_plot_widgets()
        self._setup_statusbar()
        self._build_ui()
        self._building_ui = False
        self._setup_menu()
        self._setup_toolbar()
        self._update_control_states()
        self._configure_dock_context_menu()
        self.dock_area.layoutChanged.connect(self._save_dock_layout)
        self._restore_dock_layout()
        self._restore_window_geometry()

    # ------------------------------------------------------------------
    # Plot widgets
    # ------------------------------------------------------------------

    def _create_plot_widgets(self) -> None:
        """Create plot widgets early to avoid hot-reload deletion issues."""
        self.main_plot = pg.PlotWidget()
        self.main_plot.setLabel("left", "Intensity (counts/channel)")
        self.main_plot.setLabel("bottom", "Time (ns)")
        self.main_plot.setTitle("IRF Estimation Results")
        self.main_plot.addLegend()
        self.main_plot.setLogMode(x=False, y=True)
        self.main_plot.setMenuEnabled(True)
        self.main_plot.setMouseEnabled(x=True, y=True)
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)

        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        self.main_plot.addItem(self.crosshair_v, ignoreBounds=True)
        self.main_plot.addItem(self.crosshair_h, ignoreBounds=True)
        self.main_plot.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self.range_selector = pg.LinearRegionItem(
            values=self.range_bounds,
            brush=pg.mkBrush(color=(50, 200, 50, 50)),
            movable=True,
        )
        self.range_selector.sigRegionChanged.connect(self._on_range_changed)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _setup_statusbar(self) -> None:
        """Create the status bar with time axis info."""
        self._status_bar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self.status_time_label = QtWidgets.QLabel("Time axis: Not available")
        self.status_time_label.setStyleSheet(
            "color: #888; font-style: italic; padding: 0 8px;"
        )
        self._status_bar.addPermanentWidget(self.status_time_label)
        self._status_bar.showMessage("Ready")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Build the main window layout with dock area."""
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        self.dock_area = DockArea(central)
        self._build_docks()
        layout.addWidget(self.dock_area, 1)

    def _build_params_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Build the IRF estimation parameters panel (left dock)."""
        panel = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        group = QtWidgets.QGroupBox("IRF Estimation Parameters", panel)
        grid = QtWidgets.QGridLayout(group)
        grid.setSpacing(4)

        grid.addWidget(QtWidgets.QLabel("Time/Channel (ns):"), 0, 0)
        self.dt_spinbox = QtWidgets.QDoubleSpinBox()
        self.dt_spinbox.setRange(0.001, 10.0)
        self.dt_spinbox.setValue(1.0)
        self.dt_spinbox.setDecimals(4)
        self.dt_spinbox.setSingleStep(0.01)
        self.dt_spinbox.setToolTip(
            "Time per channel in nanoseconds (automatically set from data)"
        )
        self.dt_spinbox.setEnabled(False)
        self.dt_spinbox.valueChanged.connect(lambda v: setattr(self, "dt", v))
        grid.addWidget(self.dt_spinbox, 0, 1)

        grid.addWidget(QtWidgets.QLabel("SG Window Length:"), 1, 0)
        self.window_length_spinbox = QtWidgets.QSpinBox()
        self.window_length_spinbox.setRange(5, 500)
        self.window_length_spinbox.setValue(11)
        self.window_length_spinbox.setSingleStep(2)
        self.window_length_spinbox.setToolTip(
            "Savitzky-Golay filter window length (must be odd)"
        )
        self.window_length_spinbox.valueChanged.connect(self._on_parameter_changed)
        grid.addWidget(self.window_length_spinbox, 1, 1)

        grid.addWidget(QtWidgets.QLabel("SG Poly Order:"), 2, 0)
        self.polyorder_spinbox = QtWidgets.QSpinBox()
        self.polyorder_spinbox.setRange(1, 10)
        self.polyorder_spinbox.setValue(3)
        self.polyorder_spinbox.setToolTip("Polynomial order for Savitzky-Golay filter")
        self.polyorder_spinbox.valueChanged.connect(self._on_parameter_changed)
        grid.addWidget(self.polyorder_spinbox, 2, 1)

        grid.addWidget(QtWidgets.QLabel("RL Iterations:"), 3, 0)
        self.rl_iterations_spinbox = QtWidgets.QSpinBox()
        self.rl_iterations_spinbox.setRange(5, 2000)
        self.rl_iterations_spinbox.setValue(500)
        self.rl_iterations_spinbox.setSingleStep(10)
        self.rl_iterations_spinbox.setToolTip(
            "Richardson-Lucy deconvolution iterations"
        )
        self.rl_iterations_spinbox.valueChanged.connect(self._on_parameter_changed)
        grid.addWidget(self.rl_iterations_spinbox, 3, 1)

        grid.addWidget(QtWidgets.QLabel("Regularization:"), 4, 0)
        self.regularization_spinbox = QtWidgets.QSpinBox()
        self.regularization_spinbox.setRange(1, 51)
        self.regularization_spinbox.setValue(3)
        self.regularization_spinbox.setSingleStep(2)
        self.regularization_spinbox.setToolTip(
            "Median filter size for regularization (1 = no regularization)"
        )
        self.regularization_spinbox.valueChanged.connect(self._on_parameter_changed)
        grid.addWidget(self.regularization_spinbox, 4, 1)

        grid.addWidget(QtWidgets.QLabel("Manual Background:"), 5, 0)
        self.background_spinbox = QtWidgets.QDoubleSpinBox()
        self.background_spinbox.setRange(0.0, 100000.0)
        self.background_spinbox.setValue(0.0)
        self.background_spinbox.setDecimals(2)
        self.background_spinbox.setSingleStep(1.0)
        self.background_spinbox.setToolTip(
            "Manual background offset (0 = auto-estimate)"
        )
        self.background_spinbox.valueChanged.connect(self._on_background_changed)
        grid.addWidget(self.background_spinbox, 5, 1)

        self.range_selection_checkbox = QtWidgets.QCheckBox("\U0001f4cd Use Range Selection")
        self.range_selection_checkbox.setToolTip(
            "Select a range in the decay plot to use for IRF estimation"
        )
        self.range_selection_checkbox.stateChanged.connect(
            self._on_range_selection_changed
        )
        grid.addWidget(self.range_selection_checkbox, 6, 0)

        self.auto_update_checkbox = QtWidgets.QCheckBox("\U0001f504 Auto-Update IRF")
        self.auto_update_checkbox.setToolTip(
            "Automatically re-estimate IRF when parameters change (50 RL iterations)"
        )
        self.auto_update_checkbox.stateChanged.connect(self._on_auto_update_changed)
        grid.addWidget(self.auto_update_checkbox, 6, 1)

        self.estimate_button = QtWidgets.QPushButton("\U0001f52e Estimate IRF")
        self.estimate_button.clicked.connect(self.estimate_irf)
        self.estimate_button.setEnabled(False)
        self.estimate_button.setStyleSheet(
            "QPushButton {"
            "  font-weight: bold; padding: 10px;"
            "  background-color: #d4a017; color: #1e1e1e;"
            "  border: 2px solid #b8860b; border-radius: 6px;"
            "  font-size: 13px;"
            "}"
            "QPushButton:hover { background-color: #e8b830; }"
            "QPushButton:disabled { background-color: #555; color: #888; border-color: #666; }"
        )
        grid.addWidget(self.estimate_button, 7, 0, 1, 2)

        layout.addWidget(group)
        layout.addStretch(1)
        return panel

    def _build_results_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Build the estimation results panel (bottom dock)."""
        panel = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        group = QtWidgets.QGroupBox("Estimation Results", panel)
        grid = QtWidgets.QGridLayout(group)
        grid.setSpacing(4)

        grid.addWidget(QtWidgets.QLabel("Lifetime (τ):"), 0, 0)
        self.lifetime_value = QtWidgets.QLineEdit("N/A")
        self.lifetime_value.setReadOnly(True)
        grid.addWidget(self.lifetime_value, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Decay Rate (k):"), 1, 0)
        self.decay_rate_value = QtWidgets.QLineEdit("N/A")
        self.decay_rate_value.setReadOnly(True)
        grid.addWidget(self.decay_rate_value, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Amplitude (A):"), 2, 0)
        self.amplitude_value = QtWidgets.QLineEdit("N/A")
        self.amplitude_value.setReadOnly(True)
        grid.addWidget(self.amplitude_value, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Offset (C):"), 3, 0)
        self.offset_value = QtWidgets.QLineEdit("N/A")
        self.offset_value.setReadOnly(True)
        grid.addWidget(self.offset_value, 3, 1)

        layout.addWidget(group)
        layout.addStretch(1)
        return panel

    def _build_plot_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Build the main plot panel (right dock)."""
        panel = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.main_plot, 1)
        return panel

    def _build_docks(self) -> None:
        """Create and arrange all dock panels.

        Layout: [Left: Params] | [Right-top: Plots, Right-bottom: Results]
        """
        params_panel = self._build_params_panel(self.dock_area)
        results_panel = self._build_results_panel(self.dock_area)
        plot_panel = self._build_plot_panel(self.dock_area)

        self.params_panel = params_panel
        self.results_panel = results_panel
        self.plot_panel = plot_panel

        self.dock_area._all_widgets.extend(
            [params_panel, results_panel, plot_panel]
        )
        self.dock_area._tab_names[params_panel] = "IRF Est Parameters"
        self.dock_area._tab_names[results_panel] = "Est Results"
        self.dock_area._tab_names[plot_panel] = "Plots"

        right_splitter = DockSplitter(
            QtCore.Qt.Orientation.Vertical, self.dock_area
        )
        right_splitter.addWidget(plot_panel)
        right_splitter.addWidget(results_panel)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        main_splitter = DockSplitter(
            QtCore.Qt.Orientation.Horizontal, self.dock_area
        )
        main_splitter.addWidget(params_panel)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)

        self.dock_area.set_root_widget(main_splitter)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _setup_menu(self) -> None:
        """Create menu bar actions."""
        pass

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _setup_toolbar(self) -> None:
        """Create the main toolbar with file operations."""
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("irfEstimatorMainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setContentsMargins(4, 2, 4, 2)
        if toolbar.layout() is not None:
            toolbar.layout().setSpacing(6)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toolbar.setStyleSheet("""
            QToolBar#irfEstimatorMainToolbar {
                background-color: transparent;
                border: none;
                padding: 3px 4px;
                spacing: 6px;
            }
            QToolBar#irfEstimatorMainToolbar QToolButton {
                background-color: #2a2a4a;
                border: 1px solid #5a5a8a;
                border-radius: 5px;
                padding: 5px 10px;
                margin: 0px;
                font-weight: bold;
                font-size: 12px;
            }
            QToolBar#irfEstimatorMainToolbar QToolButton:hover {
                background-color: #3a3a6a;
                border-color: #8a8aba;
            }
            QToolBar#irfEstimatorMainToolbar QToolButton:pressed {
                background-color: #4a4a8a;
            }
            QToolBar#irfEstimatorMainToolbar QToolButton:disabled {
                color: #666;
            }
            QToolBar#irfEstimatorMainToolbar QLabel {
                margin: 0px 3px;
            }
        """)

        load_action = QtWidgets.QAction("\U0001f4c2 Load Decay", self)
        load_action.triggered.connect(self.load_decay_file)
        toolbar.addAction(load_action)

        load_ds_action = QtWidgets.QAction("\U0001f4ca Load from Dataset", self)
        load_ds_action.triggered.connect(self.load_from_dataset)
        toolbar.addAction(load_ds_action)

        toolbar.addSeparator()

        self.save_action = QtWidgets.QAction("\U0001f4be Save IRF", self)
        self.save_action.triggered.connect(self.save_irf)
        self.save_action.setEnabled(False)
        toolbar.addAction(self.save_action)

        self.transfer_action = QtWidgets.QAction("\U0001f680 Transfer to ChiSurf", self)
        self.transfer_action.triggered.connect(self.add_to_chisurf)
        self.transfer_action.setEnabled(False)
        toolbar.addAction(self.transfer_action)

        toolbar.addSeparator()

        self.data_info_label = QtWidgets.QLabel("No data loaded")
        self.data_info_label.setStyleSheet("color: #aaa; padding: 0 4px;")
        toolbar.addWidget(self.data_info_label)

    # ------------------------------------------------------------------
    # Dock context menu
    # ------------------------------------------------------------------

    def _configure_dock_context_menu(self) -> None:
        """Configure context menu for dock panels."""
        self.dock_area.setContextMenuEnabled(True)
        self.dock_area.setContextMenuMode("basic")
        self.dock_area.setTabsClosable(False)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Save window geometry and dock layout before closing."""
        self._save_window_geometry()
        self._save_dock_layout()
        super().closeEvent(event)

    def _save_window_geometry(self) -> None:
        """Save the main window geometry to QSettings."""
        try:
            settings = QtCore.QSettings("chisurf", "IRFEstimatorTool")
            settings.setValue("geometry", self.saveGeometry())
            settings.sync()
        except Exception:
            pass

    def _restore_window_geometry(self) -> None:
        """Restore the main window geometry from QSettings."""
        try:
            settings = QtCore.QSettings("chisurf", "IRFEstimatorTool")
            geometry = settings.value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
        except Exception:
            self.resize(1000, 600)

    def _save_dock_layout(self) -> None:
        """Save the current dock layout to QSettings."""
        try:
            settings = QtCore.QSettings("chisurf", "IRFEstimatorTool")
            layout_state = self.dock_area.get_layout_state()
            settings.setValue(
                "dock_layout", json.dumps(layout_state, sort_keys=True)
            )
            settings.sync()
        except Exception:
            pass

    def _restore_dock_layout(self) -> None:
        """Restore the dock layout from QSettings."""
        try:
            settings = QtCore.QSettings("chisurf", "IRFEstimatorTool")
            value = settings.value("dock_layout")
            if isinstance(value, str):
                layout_state = json.loads(value)
            elif isinstance(value, dict):
                layout_state = value
            else:
                return
            self.dock_area.set_layout_state(layout_state, emit_change=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Control states
    # ------------------------------------------------------------------

    def _update_control_states(self) -> None:
        """Enable or disable controls based on data availability."""
        has_data = self.decay_data is not None
        has_irf = self.irf_data is not None
        self.estimate_button.setEnabled(has_data)
        self.save_action.setEnabled(has_irf)
        self.transfer_action.setEnabled(has_irf)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_decay_file(self, file_path: str | None = None) -> None:
        """Load a Jordi format decay file."""
        if file_path is None or isinstance(file_path, bool):
            try:
                import chisurf as cs
                start_dir = str(getattr(cs, "working_path", "") or "")
            except Exception:
                start_dir = ""
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load Decay File", start_dir,
                "Jordi Files (*.dat);;All Files (*)",
            )
            if not file_path:
                return

        self.current_file_path = file_path
        self.data_info_label.setText(str(file_path))

        try:
            from chisurf.core.fio import read_jordi
            data, metadata = read_jordi(file_path, return_metadata=True)
            data = np.asarray(data, dtype=np.float32)

            if len(data) % 2 == 0:
                half = len(data) // 2
                vv = data[:half]
            else:
                vv = data

            dt = float(metadata.get("dt", 1.0))
            time_axis = np.arange(len(vv), dtype=np.float32) * dt
            decay_data = np.column_stack((time_axis, vv))
            self._process_decay_data(decay_data, dt)

            bg_start = int(0.9 * len(decay_data))
            bg_estimate = float(np.median(decay_data[bg_start:, 1]))
            self.background_spinbox.setValue(bg_estimate)

            self.range_bounds = [0.0, float(len(decay_data) - 1)]
            self.range_selector.setRegion(self.range_bounds)

            self._update_all_plots()
            self._update_control_states()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error Loading File",
                f"Failed to load decay file:\n{str(e)}",
            )
            self.data_info_label.setText(f"Error: {str(e)}")

    def load_from_dataset(self) -> None:
        """Open a dataset selector dialog."""
        try:
            from chisurf.gui.widgets.experiments import ExperimentalDataSelector

            self.dataset_selector = ExperimentalDataSelector(
                parent=None,
                change_event=self._on_dataset_selected,
                fit=None,
                experiment=None,
            )
            self.dataset_selector.show()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Failed to open dataset selector:\n{str(e)}",
            )

    def _on_dataset_selected(self) -> None:
        """Handle dataset selection from the selector dialog."""
        try:
            selected = self.dataset_selector.selected_dataset
            if selected is not None:
                self._load_dataset(selected)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Failed to load selected dataset:\n{str(e)}",
            )

    def _load_dataset(self, dataset: Any) -> None:
        """Load data from a ChiSurf dataset."""
        try:
            if hasattr(dataset, "y") and hasattr(dataset, "x"):
                y_data = np.asarray(dataset.y, dtype=np.float32)
                x_data = np.asarray(dataset.x, dtype=np.float32)
                dt = (
                    float(np.mean(np.diff(x_data)))
                    if len(x_data) > 1
                    else float(getattr(dataset, "dt", 1.0))
                )
                self.current_dataset = dataset
                self.current_file_path = getattr(dataset, "filename", None)
                name = getattr(dataset, "name", "Unnamed")
                exp = getattr(dataset, "experiment", None)
                exp_name = (
                    getattr(exp, "name", "Uncategorized") if exp else "Uncategorized"
                )
                self.data_info_label.setText(f"Dataset: {exp_name} - {name}")
                self._process_decay_data(
                    np.column_stack((x_data, y_data)), dt
                )
            elif hasattr(dataset, "data"):
                data = np.asarray(dataset.data, dtype=np.float32)
                dt = float(getattr(dataset, "dt", 1.0))
                if data.ndim == 1:
                    time_axis = np.arange(len(data)) * dt
                    decay_data = np.column_stack((time_axis, data))
                elif data.ndim == 2 and data.shape[1] >= 2:
                    decay_data = data[:, :2]
                    if len(decay_data) > 1:
                        dt = float(np.mean(np.diff(decay_data[:, 0])))
                else:
                    raise ValueError(
                        f"Unsupported data shape: {data.shape}"
                    )
                self.current_dataset = dataset
                self.current_file_path = getattr(dataset, "filename", None)
                name = getattr(dataset, "name", "Unnamed")
                exp = getattr(dataset, "experiment", None)
                exp_name = (
                    getattr(exp, "name", "Uncategorized") if exp else "Uncategorized"
                )
                self.data_info_label.setText(f"Dataset: {exp_name} - {name}")
                self._process_decay_data(decay_data, dt)
            else:
                raise ValueError(
                    "Unsupported dataset format. "
                    "Expected 'x' and 'y' or 'data' attributes."
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Failed to load dataset:\n{str(e)}",
            )

    def _process_decay_data(
        self, decay_data: np.ndarray, dt: float
    ) -> None:
        """Process loaded decay data (common path for file and dataset)."""
        decay_data = np.asarray(decay_data, dtype=np.float32)
        if decay_data.ndim == 1:
            time_axis = np.arange(len(decay_data)) * dt
            decay_data = np.column_stack((time_axis, decay_data))
        elif decay_data.ndim == 2 and decay_data.shape[1] >= 2:
            decay_data = decay_data[:, :2]
            if len(decay_data) > 1:
                dt = float(np.mean(np.diff(decay_data[:, 0])))

        if np.any(np.isnan(decay_data)) or np.any(np.isinf(decay_data)):
            raise ValueError("Data contains NaN or Inf values")

        self.channel_axis = decay_data[:, 0]
        self.decay_data = decay_data[:, 1]
        self.decay_data_original = self.decay_data.copy()
        self.dt = dt
        self.dt_spinbox.setValue(dt)

        if len(self.channel_axis) > 1:
            t_min = float(np.min(self.channel_axis))
            t_max = float(np.max(self.channel_axis))
            t_range = t_max - t_min
            self.status_time_label.setText(
                f"Time: {t_min:.2f} to {t_max:.2f} ns "
                f"(\u0394 = {t_range:.2f} ns, "
                f"dt = {dt:.4f} ns, {len(self.channel_axis)} pts)"
            )

        self._on_background_changed(self.background_spinbox.value())
        self._update_all_plots()
        self._update_control_states()

    # ------------------------------------------------------------------
    # IRF Estimation
    # ------------------------------------------------------------------

    def estimate_irf(self) -> None:
        """Estimate IRF from loaded decay data with full iterations."""
        if self.decay_data is None:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Please load a decay file first."
            )
            return

        if self.is_estimating:
            return
        self.is_estimating = True

        try:
            self._status_bar.showMessage("Estimating IRF...")
            QtWidgets.QApplication.processEvents()

            settings = IRFEstimationSettings(
                window_length=self.window_length_spinbox.value(),
                polyorder=self.polyorder_spinbox.value(),
                rl_iterations=self.rl_iterations_spinbox.value(),
                regularization=self.regularization_spinbox.value(),
                manual_background=self.manual_background,
                use_range_selection=self.use_range_selection,
                range_bounds=self.range_bounds,
            )

            result = _estimate_irf(
                intensity=self.decay_data,
                dt=self.dt,
                settings=settings,
                channel_axis=self.channel_axis,
            )

            self.irf_data = np.array(result.irf)
            self.irf_params = result.params
            self._update_results(result)
            self._update_all_plots()
            self._update_control_states()

            self._status_bar.showMessage(
                "IRF estimation completed", timeout=5000
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Estimation Error", str(e)
            )
            self._status_bar.showMessage(
                "IRF estimation failed", timeout=5000
            )
            import traceback
            traceback.print_exc()
        finally:
            self.is_estimating = False

    def _estimate_irf_quick(self) -> None:
        """Quick IRF estimation with reduced iterations for auto-update."""
        if self.is_estimating or self.decay_data is None:
            return
        self.is_estimating = True
        try:
            settings = IRFEstimationSettings(
                window_length=self.window_length_spinbox.value(),
                polyorder=self.polyorder_spinbox.value(),
                rl_iterations=50,
                regularization=self.regularization_spinbox.value(),
                manual_background=self.manual_background,
                use_range_selection=self.use_range_selection,
                range_bounds=self.range_bounds,
            )
            result = _estimate_irf(
                intensity=self.decay_data,
                dt=self.dt,
                settings=settings,
                channel_axis=self.channel_axis,
            )
            self.irf_data = np.array(result.irf)
            self.irf_params = result.params
            self._update_results(result)
            self._update_all_plots()
            self._update_control_states()
        except Exception:
            pass
        finally:
            self.is_estimating = False

    def _update_results(self, result: Any) -> None:
        """Update the results display with estimated parameters."""
        self.lifetime_value.setText(f"{result.lifetime_ns:.4f} ns")
        self.decay_rate_value.setText(f"{result.decay_rate_ns:.6f} ns\u207b\u00b9")
        self.amplitude_value.setText(f"{result.amplitude:.2f}")
        self.offset_value.setText(f"{result.offset:.2f}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _update_all_plots(self) -> None:
        """Update the main plot with decay, IRF, and forward model."""
        if self.decay_data is None or self.channel_axis is None:
            return

        y_data = self.decay_data_original
        if y_data is not None and y_data.ndim > 1:
            y_data = y_data.flatten()

        had_range = self.range_selector in self.main_plot.items()
        self.main_plot.clear()

        self.main_plot.plot(
            self.channel_axis,
            y_data,
            pen=pg.mkPen("b", width=2),
            name="Measured Decay",
        )

        if self.manual_background > 0:
            decay_corrected = np.maximum(
                self.decay_data_original - self.manual_background, 0.1
            )
            self.main_plot.plot(
                self.channel_axis,
                decay_corrected,
                pen=pg.mkPen(
                    "cyan", width=2, style=QtCore.Qt.PenStyle.DashLine
                ),
                name=f"BG Corrected (BG={self.manual_background:.1f})",
            )

        if self.irf_data is not None:
            irf_scaled = self.irf_data * (
                self.decay_data.max() / self.irf_data.max()
            )
            irf_thresholded = np.where(
                irf_scaled >= 1.0, irf_scaled, np.nan
            )
            self.main_plot.plot(
                self.channel_axis,
                irf_thresholded,
                pen=pg.mkPen("g", width=2),
                name="Estimated IRF (scaled)",
            )

            if self.irf_params is not None:
                from chisurf.core.fluorescence.tcspc.irf_estimation import (
                    generate_truncated_exponential,
                    partial_convolution_fft,
                )

                kernel = generate_truncated_exponential(
                    np.arange(len(self.irf_data)) * self.dt,
                    {
                        "A": 1.0,
                        "C": 0.0,
                        "k": self.irf_params["k"],
                        "t0": 0.0,
                    },
                )
                kernel = np.maximum(kernel, 0)
                kernel = kernel / kernel.sum()

                forward = partial_convolution_fft(
                    self.irf_data.reshape(-1, 1), kernel, axis=0
                )
                forward += self.irf_params["C"]

                self.main_plot.plot(
                    self.channel_axis,
                    forward[:, 0],
                    pen=pg.mkPen(
                        "orange", width=2, style=QtCore.Qt.PenStyle.DashLine
                    ),
                    name="IRF \u2297 Exp (Forward Model)",
                )

        if had_range and self.use_range_selection:
            self.main_plot.addItem(self.range_selector)

    def _on_mouse_moved(self, pos: QtCore.QPointF) -> None:
        """Handle mouse movement for crosshair display."""
        if self.main_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            self.crosshair_v.setPos(x)
            self.crosshair_h.setPos(y)
            if self.channel_axis is not None and self.decay_data is not None:
                idx = int(np.argmin(np.abs(self.channel_axis - x)))
                if 0 <= idx < len(self.channel_axis):
                    x_val = float(self.channel_axis[idx])
                    y_val = (
                        float(self.decay_data[idx])
                        if idx < len(self.decay_data)
                        else 0.0
                    )
                    self.main_plot.setToolTip(
                        f"Time: {x_val:.2f} ns, Intensity: {y_val:.1f}"
                    )

    # ------------------------------------------------------------------
    # Parameter change handlers
    # ------------------------------------------------------------------

    def _on_parameter_changed(self) -> None:
        """Handle estimation parameter changes with optional auto-update."""
        if (
            self.auto_update_enabled
            and self.irf_data is not None
            and not self.is_estimating
        ):
            self._estimate_irf_quick()

    def _on_background_changed(self, value: float) -> None:
        """Handle manual background value changes."""
        self.manual_background = value
        if self.decay_data_original is not None:
            self.decay_data = np.maximum(
                self.decay_data_original - self.manual_background, 0.0
            )
            self._update_all_plots()
            if (
                self.auto_update_enabled
                and self.irf_data is not None
                and not self.is_estimating
            ):
                self._estimate_irf_quick()

    def _on_range_selection_changed(self) -> None:
        """Handle range selection checkbox changes."""
        self.use_range_selection = self.range_selection_checkbox.isChecked()
        if self.use_range_selection and self.decay_data is not None:
            self.main_plot.addItem(self.range_selector)
        elif (
            hasattr(self, "range_selector")
            and self.range_selector in self.main_plot.items()
        ):
            self.main_plot.removeItem(self.range_selector)

    def _on_range_changed(self) -> None:
        """Handle range selector region changes."""
        self.range_bounds = list(self.range_selector.getRegion())

    def _on_auto_update_changed(self) -> None:
        """Handle auto-update checkbox changes."""
        self.auto_update_enabled = self.auto_update_checkbox.isChecked()

    # ------------------------------------------------------------------
    # Save and Transfer
    # ------------------------------------------------------------------

    def save_irf(self) -> None:
        """Save the estimated IRF in Jordi format."""
        if self.irf_data is None or len(self.irf_data) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No IRF", "Please estimate an IRF first."
            )
            return

        try:
            import chisurf as cs
            start_dir = str(getattr(cs, "working_path", "") or "")
        except Exception:
            start_dir = ""

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save IRF", start_dir,
            "Jordi Files (*.dat);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".dat"):
            file_path += ".dat"

        try:
            from chisurf.core.fio import write_jordi
            irf_data = np.asarray(self.irf_data).flatten()
            write_jordi(file_path, irf_data, irf_data)
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise RuntimeError(
                    "Failed to save IRF file or file is empty"
                )
            QtWidgets.QMessageBox.information(
                self, "Success",
                f"IRF successfully saved to:\n{file_path}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Save Error", str(e)
            )

    def add_to_chisurf(self) -> None:
        """Transfer the estimated IRF to ChiSurf as a dataset."""
        if self.irf_data is None or len(self.irf_data) == 0:
            QtWidgets.QMessageBox.warning(
                self, "No IRF", "Please estimate an IRF first."
            )
            return

        irf_data = np.asarray(self.irf_data, dtype=float).flatten()
        if len(irf_data) == 0:
            return
        if not np.isfinite(irf_data).all():
            QtWidgets.QMessageBox.critical(
                self, "Error",
                "IRF data contains NaN or infinite values."
            )
            return

        try:
            from chisurf.core.fio import write_jordi

            fd, tmp_path = tempfile.mkstemp(suffix=".dat")
            os.close(fd)

            temp_dir = os.path.dirname(tmp_path)
            temp_file = os.path.join(
                temp_dir, f"temp_{os.urandom(8).hex()}.dat"
            )
            write_jordi(temp_file, irf_data, irf_data)

            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            os.rename(temp_file, tmp_path)

            import chisurf as cs
            filename = Path(tmp_path).name

            if hasattr(cs, "core") and hasattr(cs.core, "actions"):
                cs.core.actions.dispatch(
                    name="experiment.set",
                    payload={"name": "TCSPC"},
                )
                cs.core.actions.dispatch(
                    name="setup.params.set",
                    payload={
                        "params": {
                            "is_jordi": True,
                            "use_header": False,
                            "matrix_columns": [],
                            "g_factor": 1.0,
                            "polarization": "V",
                            "rep_rate": 10.0,
                            "rebin": (1, 1),
                            "dt": float(self.dt),
                        }
                    },
                )
                cs.core.actions.dispatch(
                    name="dataset.add",
                    payload={
                        "filename": tmp_path,
                        "experiment_reader": None,
                    },
                )
                QtWidgets.QMessageBox.information(
                    self, "Success",
                    f"IRF '{filename}' has been transferred to ChiSurf.",
                )
            else:
                QtWidgets.QMessageBox.information(
                    self, "IRF Ready",
                    f"IRF saved to:\n{tmp_path}\n\n"
                    "You can now load this file as an IRF in your analysis.",
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                f"Failed to transfer IRF:\n{str(e)}",
            )
