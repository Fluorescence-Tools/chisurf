"""
import chisurf as cs
IRF Estimator Plugin

This plugin provides blind instrument response function (IRF) estimation from
fluorescence decay data without requiring separate IRF measurements.

Features:
- Load Jordi format decay files
- Automatically estimate IRF using blind deconvolution
- Display decay and computed IRF side-by-side
- Save computed IRF in Jordi format
- Load IRF directly into ChiSurf for further analysis
- Interactive parameter adjustment for IRF estimation

The plugin implements the algorithm described in:
Gómez-Sánchez et al., "Blind instrument response function identification from 
fluorescence decays", Biophysical Reports, 2024.
https://doi.org/10.1016/j.bpr.2024.100155

Reference:
    Gómez-Sánchez, A., Fersini, F., Zappone, S., Slenders, E., Donato, M., 
    Pelicci, S., Tortarolo, G., Bega, G., Bouzin, M., Cardarelli, F., Lanzanò, L., 
    Koho, S. V., & Vicidomini, G. (2024). "Blind instrument response function 
    identification from fluorescence decays." Biophysical Reports, 4(2), 100155.
"""

name = "Spectroscopy:Fluorescence decay:IRF Extraction"

import chisurf as cs
import sys
import os
import numpy as np
import warnings
from pathlib import Path

from qtpy.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFileDialog, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QGroupBox, QMessageBox, QProgressDialog, QCheckBox,
    QDialog, QListWidget, QDialogButtonBox, QVBoxLayout as QVBoxLayout_,
    QHBoxLayout as QHBoxLayout_, QTreeWidget, QTreeWidgetItem, QAbstractItemView,
    QLineEdit, QHeaderView, QSizePolicy
)
from qtpy.QtCore import Qt
import pyqtgraph as pg

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c



# ChiSurf imports
try:
    from chisurf.core.fio import read_jordi as _read_jordi
    from chisurf.core.fio import write_jordi as _write_jordi
    from chisurf.core.fluorescence.tcspc import IRFEstimator
    CHISURF_AVAILABLE = True
except Exception:
    _read_jordi = None
    _write_jordi = None
    IRFEstimator = None
    CHISURF_AVAILABLE = False


@persist_plugin_state("irf_estimator")
class IRFEstimatorPlugin(QWidget):
    """Main widget for the IRF Estimator plugin."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IRF Estimator - Blind IRF Estimation")
        
        # Data storage
        self.decay_data = None
        self.decay_data_original = None  # Store original data before BG correction
        self.channel_axis = None
        self.dt = 1.0  # Time per channel in nanoseconds (for lifetime calculation only)
        self.estimator = None
        self.irf_data = None
        self.current_file_path = None
        self.current_dataset = None  # Store reference to current dataset
        self.use_range_selection = False
        self.range_bounds = [0, 100]  # Default range bounds in channels
        self.manual_background = 0.0  # Manual background value
        self.auto_update_enabled = True  # Auto-update flag
        self.is_estimating = False  # Prevent recursive updates
        
        # Create UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # File controls
        file_group = QGroupBox("File Operations")
        file_layout = QGridLayout()
        
        # First row: Load buttons
        self.load_button = QPushButton("Load Decay (Jordi)")
        self.load_button.clicked.connect(self.load_decay_file)
        
        self.load_dataset_button = QPushButton("Load from Dataset")
        self.load_dataset_button.clicked.connect(self.load_from_dataset)
        
        # Second row: Save/Transfer buttons
        self.save_irf_button = QPushButton("Save IRF (Jordi)")
        self.save_irf_button.clicked.connect(self.save_irf)
        self.save_irf_button.setEnabled(False)
        
        self.transfer_button = QPushButton("Transfer to ChiSurf")
        self.transfer_button.clicked.connect(self.on_transfer_clicked)
        self.transfer_button.setEnabled(False)
        
        # File/dataset info
        self.file_label = QLineEdit("No data loaded")
        self.file_label.setReadOnly(True)
        self.file_label.setToolTip("Loaded dataset information")
        
        # Time axis info display
        self.time_axis_label = QLabel("Time axis: Not available")
        self.time_axis_label.setToolTip("Information about the time axis of the loaded data")
        self.time_axis_label.setStyleSheet("color: #666666; font-style: italic;")
        
        # Data stats display
        self.data_stats_label = QLabel("Data points: 0 | Duration: 0.00 ns")
        self.data_stats_label.setToolTip("Statistics about the loaded data")
        self.data_stats_label.setStyleSheet("color: #666666;")
        
        # Add to layout
        # First row: Load buttons
        file_layout.addWidget(self.load_button, 0, 0)
        file_layout.addWidget(self.load_dataset_button, 0, 1)
        
        # Second row: Save/Transfer buttons
        file_layout.addWidget(self.save_irf_button, 1, 0)
        file_layout.addWidget(self.transfer_button, 1, 1)
        
        # Third row: File info (spans 2 columns)
        file_layout.addWidget(self.file_label, 2, 0, 1, 2)
        
        # Fourth row: Time axis info
        file_layout.addWidget(self.time_axis_label, 3, 0, 1, 2)
        
        # Fifth row: Data stats
        file_layout.addWidget(self.data_stats_label, 4, 0, 1, 2)
        
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # Estimation parameters
        params_group = QGroupBox("IRF Estimation Parameters")
        params_layout = QGridLayout()
        
        # Time per channel (for lifetime calculation)
        params_layout.addWidget(QLabel("Time/Channel (ns):"), 0, 0)
        self.dt_spinbox = QDoubleSpinBox()
        self.dt_spinbox.setRange(0.001, 10.0)
        self.dt_spinbox.setValue(1.0)
        self.dt_spinbox.setDecimals(4)
        self.dt_spinbox.setSingleStep(0.01)
        self.dt_spinbox.setToolTip("Time per channel in nanoseconds (automatically set from data when available)")
        self.dt_spinbox.setEnabled(False)  # Will be enabled only for manual entry
        self.dt_spinbox.valueChanged.connect(lambda v: setattr(self, 'dt', v))
        params_layout.addWidget(self.dt_spinbox, 0, 1)

        # Window length for SG filter
        params_layout.addWidget(QLabel("SG Window Length:"), 1, 0)
        self.window_length_spinbox = QSpinBox()
        self.window_length_spinbox.setRange(5, 500)
        self.window_length_spinbox.setValue(11)
        self.window_length_spinbox.setSingleStep(2)  # Keep it odd
        self.window_length_spinbox.setToolTip("Savitzky-Golay filter window length (must be odd)")
        self.window_length_spinbox.valueChanged.connect(self.on_parameter_changed)
        params_layout.addWidget(self.window_length_spinbox, 1, 1)
        
        # Polynomial order
        params_layout.addWidget(QLabel("SG Poly Order:"), 2, 0)
        self.polyorder_spinbox = QSpinBox()
        self.polyorder_spinbox.setRange(1, 10)
        self.polyorder_spinbox.setValue(3)
        self.polyorder_spinbox.setToolTip("Polynomial order for Savitzky-Golay filter")
        self.polyorder_spinbox.valueChanged.connect(self.on_parameter_changed)
        params_layout.addWidget(self.polyorder_spinbox, 2, 1)
        
        # RL iterations
        params_layout.addWidget(QLabel("RL Iterations:"), 3, 0)
        self.rl_iterations_spinbox = QSpinBox()
        self.rl_iterations_spinbox.setRange(5, 2000)
        self.rl_iterations_spinbox.setValue(500)
        self.rl_iterations_spinbox.setSingleStep(10)
        self.rl_iterations_spinbox.setToolTip("Richardson-Lucy deconvolution iterations (default: 500)")
        params_layout.addWidget(self.rl_iterations_spinbox, 3, 1)
        
        # Regularization
        params_layout.addWidget(QLabel("Regularization:"), 4, 0)
        self.regularization_spinbox = QSpinBox()
        self.regularization_spinbox.setRange(1, 51)
        self.regularization_spinbox.setValue(3)
        self.regularization_spinbox.setSingleStep(2)  # Keep it odd
        self.regularization_spinbox.setToolTip("Median filter size for regularization (1 = no regularization)")
        self.regularization_spinbox.valueChanged.connect(self.on_parameter_changed)
        params_layout.addWidget(self.regularization_spinbox, 4, 1)
        
        # Manual background correction
        params_layout.addWidget(QLabel("Manual Background:"), 5, 0)
        self.background_spinbox = QDoubleSpinBox()
        self.background_spinbox.setRange(0.0, 100000.0)
        self.background_spinbox.setValue(0.0)
        self.background_spinbox.setDecimals(2)
        self.background_spinbox.setSingleStep(1.0)
        self.background_spinbox.setToolTip("Manual background offset to subtract from data (0 = auto-estimate)")
        self.background_spinbox.valueChanged.connect(self.on_background_changed)
        params_layout.addWidget(self.background_spinbox, 5, 1)
        
        # Range selection checkbox
        self.range_selection_checkbox = QCheckBox("Use Range Selection")
        self.range_selection_checkbox.setChecked(False)
        self.range_selection_checkbox.stateChanged.connect(self.on_range_selection_changed)
        self.range_selection_checkbox.setToolTip("Select a range in the decay plot to use for IRF estimation")
        params_layout.addWidget(self.range_selection_checkbox, 6, 0)
        
        # Auto-update checkbox
        self.auto_update_checkbox = QCheckBox("Auto-Update IRF")
        self.auto_update_checkbox.setChecked(False)
        self.auto_update_checkbox.stateChanged.connect(self.on_auto_update_changed)
        self.auto_update_checkbox.setToolTip("Automatically re-estimate IRF when parameters change (uses 50 RL iterations)")
        params_layout.addWidget(self.auto_update_checkbox, 6, 1)
        
        # Estimate button
        self.estimate_button = QPushButton("Estimate IRF")
        self.estimate_button.clicked.connect(self.estimate_irf)
        self.estimate_button.setEnabled(False)
        self.estimate_button.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        params_layout.addWidget(self.estimate_button, 7, 0, 1, 2)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Results display
        results_group = QGroupBox("Estimation Results")
        results_layout = QGridLayout()
        
        results_layout.addWidget(QLabel("Estimated Lifetime (τ):"), 0, 0)
        self.lifetime_value = QLineEdit("N/A")
        self.lifetime_value.setReadOnly(True)
        results_layout.addWidget(self.lifetime_value, 0, 1)
        
        results_layout.addWidget(QLabel("Decay Rate (k):"), 1, 0)
        self.decay_rate_value = QLineEdit("N/A")
        self.decay_rate_value.setReadOnly(True)
        results_layout.addWidget(self.decay_rate_value, 1, 1)
        
        results_layout.addWidget(QLabel("Amplitude (A):"), 2, 0)
        self.amplitude_value = QLineEdit("N/A")
        self.amplitude_value.setReadOnly(True)
        results_layout.addWidget(self.amplitude_value, 2, 1)
        
        results_layout.addWidget(QLabel("Offset (C):"), 3, 0)
        self.offset_value = QLineEdit("N/A")
        self.offset_value.setReadOnly(True)
        results_layout.addWidget(self.offset_value, 3, 1)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # Create a horizontal layout for the main window
        hbox = QHBoxLayout()
        
        # Create a container widget for the controls
        controls_container = QWidget()
        controls_container.setLayout(main_layout)
        
        # Add the controls container to the left side
        hbox.addWidget(controls_container, stretch=1)
        
        # Create a vertical layout for the plot
        plot_container = QVBoxLayout()
        
        # Main plot widget
        self.main_plot = pg.PlotWidget()
        self.main_plot.setLabel('left', 'Intensity (counts/channel)')
        self.main_plot.setLabel('bottom', 'Time (ns)')
        self.main_plot.setTitle('IRF Estimation Results')
        self.main_plot.addLegend()
        self.main_plot.setLogMode(x=False, y=True)
        self.main_plot.setMenuEnabled(True)
        self.main_plot.setMouseEnabled(x=True, y=True)
        self.main_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Add crosshair for better data inspection
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        self.main_plot.addItem(self.crosshair_v, ignoreBounds=True)
        self.main_plot.addItem(self.crosshair_h, ignoreBounds=True)
        
        # Add mouse tracking for crosshair
        self.main_plot.scene().sigMouseMoved.connect(self.on_mouse_moved)
        
        # Create range selector for decay region (initially hidden)
        self.range_selector = pg.LinearRegionItem(
            values=self.range_bounds,
            brush=pg.mkBrush(color=(50, 200, 50, 50)),
            movable=True
        )
        self.range_selector.sigRegionChanged.connect(self.on_range_changed)
        
        # Add the plot to the plot container
        plot_container.addWidget(self.main_plot)
        
        # Create a widget to hold the plot container
        plot_widget = QWidget()
        plot_widget.setLayout(plot_container)
        
        # Add the plot widget to the right side with stretch factor 2 (wider than controls)
        hbox.addWidget(plot_widget, stretch=2)
        
        # Set the main layout
        self.setLayout(hbox)
        self.resize(800, 400)
        
    def load_decay_file(self, file_path=None):
        """Load a Jordi format decay file.
        
        Parameters
        ----------
        file_path : str, optional
            Path to the Jordi file. If None, a file dialog will be opened.
        """
        # Handle boolean from clicked signal
        if isinstance(file_path, bool):
            file_path = None
            
        if file_path is None:
            # Start in ChiSurf working directory if available
            try:
                start_dir = str(getattr(cs, 'working_path', '') or '')
            except Exception:
                start_dir = ""
                
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Decay File", start_dir, 
                "Jordi Files (*.dat);;All Files (*)"
            )
            
            if not file_path:
                return
                
        self.current_file_path = file_path
        self.file_label.setText(str(file_path))
        
        try:
            # Load Jordi file
            if _read_jordi is not None:
                # Read all decays from the file
                decay_data_list = []
                with open(file_path, 'r') as f:
                    # Read the first line to check the format
                    first_line = f.readline().strip()
                    f.seek(0)  # Reset file pointer
                    
                    if first_line.startswith('#'):
                        # Multi-decay format
                        current_decay = []
                        for line in f:
                            if line.startswith('#'):
                                if current_decay:  # Save previous decay if exists
                                    decay_data_list.append(np.array(current_decay))
                                    current_decay = []
                                continue
                            try:
                                # Try to parse the line as a number
                                value = float(line.strip().split()[0])  # Take first column
                                current_decay.append(value)
                            except (ValueError, IndexError):
                                continue
                        # Add the last decay
                        if current_decay:
                            decay_data_list.append(np.array(current_decay))
                    else:
                        # Single decay format
                        decay_data = np.loadtxt(file_path)
                        if len(decay_data.shape) == 1:
                            decay_data_list = [decay_data]
                        else:
                            # Each column is a separate decay
                            decay_data_list = [decay_data[:, i] for i in range(decay_data.shape[1])]
                
                if not decay_data_list:
                    raise ValueError("No valid decay data found in the file")
                    
                # For multi-decay format, we assume time is the first column and intensity is the second
                if len(decay_data_list) >= 2:
                    # If we have at least two columns, use first as time and second as intensity
                    time_axis = decay_data_list[0]
                    intensity_data = decay_data_list[1]
                    decay_data = np.column_stack((time_axis, intensity_data))
                    
                    # Calculate time step from the time axis if possible
                    if len(time_axis) > 1:
                        self.dt = float(np.mean(np.diff(time_axis)))
                    else:
                        self.dt = 1.0  # Fallback if we can't calculate dt
                else:
                    # Single column format, create time axis using default dt
                    intensity_data = decay_data_list[0]
                    self.dt = 1.0  # Default time step
                    time_axis = np.arange(len(intensity_data)) * self.dt
                    decay_data = np.column_stack((time_axis, intensity_data))
                
                # Process the decay data with the calculated time step
                self.process_decay_data(decay_data, self.dt)
                
                # Store all decays for separate processing
                self.all_decays = decay_data_list
                
            else:
                # Fallback to numpy
                warnings.warn(
                    "Using numpy.loadtxt fallback. Install ChiSurf for better Jordi support.",
                    UserWarning
                )
                jordi_data = np.loadtxt(file_path)
                if len(jordi_data.shape) == 1:
                    decay_data = jordi_data
                else:
                    # Use first column by default
                    decay_data = jordi_data[:, 0]
                    
                # For numpy fallback, check if we have time and intensity columns
                if len(jordi_data.shape) == 2 and jordi_data.shape[1] >= 2:
                    # First column is time, second is intensity
                    time_axis = jordi_data[:, 0]
                    intensity_data = jordi_data[:, 1]
                    decay_data = np.column_stack((time_axis, intensity_data))
                    
                    # Calculate time step from the time axis if possible
                    if len(time_axis) > 1:
                        self.dt = float(np.mean(np.diff(time_axis)))
                    else:
                        self.dt = 1.0  # Fallback if we can't calculate dt
                else:
                    # Single column format, create time axis using default dt
                    intensity_data = jordi_data.flatten()
                    self.dt = 1.0  # Default time step
                    time_axis = np.arange(len(intensity_data)) * self.dt
                    decay_data = np.column_stack((time_axis, intensity_data))
                
                # Process the decay data with the calculated time step
                self.process_decay_data(decay_data, self.dt)
            
            # Auto-estimate background from last 10% of data
            bg_start = int(0.9 * len(decay_data))
            bg_estimate = np.median(decay_data[bg_start:])
            self.background_spinbox.setValue(float(bg_estimate))
            
            # Set initial range to full data (in channels)
            self.range_bounds = [0, len(decay_data) - 1]
            self.range_selector.setRegion(self.range_bounds)
            
            # Update plot with the first decay
            self.update_all_plots()
            
            # Enable estimate button
            self.estimate_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error Loading File",
                f"Failed to load decay file:\n{str(e)}"
            )
            self.file_label.setText(f"Error: {str(e)}")
            
    def process_all_decays(self, rl_iterations=250):
        """Process all loaded decays separately and return the average IRF."""
        if not hasattr(self, 'all_decays') or not self.all_decays:
            return None
            
        all_irfs = []
        
        # Get current parameters
        window_length = self.window_length_spinbox.value()
        if window_length % 2 == 0:
            window_length += 1
            
        polyorder = self.polyorder_spinbox.value()
        regularization = self.regularization_spinbox.value()
        if regularization > 1 and regularization % 2 == 0:
            regularization += 1
            
        # Process each decay
        for i, decay in enumerate(self.all_decays):
            try:
                # Create a copy for this decay
                decay_for_estimation = decay.copy()
                
                # Apply range selection if enabled
                if self.use_range_selection:
                    min_ch, max_ch = self.range_bounds
                    mask = (np.arange(len(decay_for_estimation)) >= min_ch) & \
                           (np.arange(len(decay_for_estimation)) <= max_ch)
                    decay_for_estimation[~mask] = 0.0
                
                # Create and configure estimator
                estimator = IRFEstimator(decay_for_estimation.reshape(-1, 1), dt=1.0)
                
                # Process this decay
                estimator.find_t0_t1(window_length=window_length, polyorder=polyorder)
                estimator.fit_exponential()
                estimator.generate_data_fit()
                estimator.generate_kernel()
                estimator.richardson_lucy_deconvolution(
                    iterations=rl_iterations,
                    regularization=regularization
                )
                
                # Store the IRF
                irf = estimator.irf[:, 0]
                all_irfs.append(irf)
                
            except Exception as e:
                print(f"Error processing decay {i+1}: {str(e)}")
                continue
        
        if not all_irfs:
            return None
            
        # Average all IRFs
        avg_irf = np.mean(all_irfs, axis=0)
        
        # Normalize the average IRF
        avg_irf = avg_irf / np.max(avg_irf) * np.max(self.decay_data_original)
        
        return avg_irf
            
    def update_all_plots(self):
        """Update the main plot with all data: measured decay, fit, IRF, and forward model."""
        if self.decay_data is None or self.channel_axis is None:
            return
        
        # Ensure data is 1D
        y_data = self.decay_data_original
        if y_data.ndim > 1:
            if y_data.shape[1] == 1:
                y_data = y_data.flatten()
            else:
                y_data = y_data[:, 0]  # Take first column if multiple
        
        # Remember if range selector was there
        had_range_selector = self.range_selector in self.main_plot.items()
            
        self.main_plot.clear()
        
        # Plot original measured decay
        self.main_plot.plot(
            self.channel_axis,
            y_data,
            pen=pg.mkPen('b', width=2),
            name='Measured Decay'
        )
        
        # Plot background-corrected decay if background is set
        if self.manual_background > 0:
            decay_corrected = np.maximum(self.decay_data_original - self.manual_background, 0.1)  # 0.1 for log plot
            self.main_plot.plot(
                self.channel_axis,
                decay_corrected,
                pen=pg.mkPen('cyan', width=2, style=Qt.DashLine),
                name=f'BG Corrected (BG={self.manual_background:.1f})'
            )
        
        # If we have estimation results, plot IRF and forward model
        if self.estimator is not None and self.estimator.params is not None:
            from chisurf.core.fluorescence.tcspc.irf_estimation import (
                partial_convolution_fft
            )
            
            # Plot estimated IRF (scaled to decay height and thresholded)
            if self.irf_data is not None:
                # Scale IRF to match the height of the measured decay
                irf_scaled = self.irf_data * (self.decay_data.max() / self.irf_data.max())
                # Threshold at 1 count to remove noise floor
                irf_scaled_thresholded = np.where(irf_scaled >= 1.0, irf_scaled, np.nan)
                self.main_plot.plot(
                    self.channel_axis,
                    irf_scaled_thresholded,
                    pen=pg.mkPen('g', width=2),
                    name='Estimated IRF (scaled, threshold=1)'
                )
                
                # Plot forward model (IRF ⊗ Exponential)
                forward_model = partial_convolution_fft(
                    self.estimator.irf, 
                    self.estimator.kernel, 
                    axis=0
                )
                forward_model += self.estimator.params['C'].reshape(1, -1)
                
                self.main_plot.plot(
                    self.channel_axis,
                    forward_model[:, 0],
                    pen=pg.mkPen('orange', width=2, style=Qt.DashLine),
                    name='IRF ⊗ Exp (Forward Model)'
                )
        
        # Re-add range selector if it was there and checkbox is checked
        if had_range_selector and self.use_range_selection:
            self.main_plot.addItem(self.range_selector)
    
    def on_range_selection_changed(self):
        """Handle changes to the range selection checkbox."""
        self.use_range_selection = self.range_selection_checkbox.isChecked()
        
        if self.use_range_selection and self.decay_data is not None:
            # Add range selector to plot
            self.main_plot.addItem(self.range_selector)
        elif hasattr(self, 'range_selector') and self.range_selector in self.main_plot.items():
            # Remove range selector from plot
            self.main_plot.removeItem(self.range_selector)
    
    def on_range_changed(self):
        """Handle changes to the selected range."""
        self.range_bounds = self.range_selector.getRegion()
    
    def on_background_changed(self, value):
        """Handle changes to the manual background value."""
        self.manual_background = value
        
        # Apply background correction to working data
        if self.decay_data_original is not None:
            self.decay_data = np.maximum(self.decay_data_original - self.manual_background, 0.0)
            self.update_all_plots()
            
            # Trigger auto-update if enabled and we have previous estimation
            if self.auto_update_enabled and self.estimator is not None:
                self.estimate_irf_quick()
    
    def on_auto_update_changed(self):
        """Handle changes to the auto-update checkbox."""
        self.auto_update_enabled = self.auto_update_checkbox.isChecked()
    
    def on_parameter_changed(self):
        """Handle changes to estimation parameters."""
        # Trigger auto-update if enabled and we have previous estimation
        if self.auto_update_enabled and self.estimator is not None and not self.is_estimating:
            self.estimate_irf_quick()
    
    def estimate_irf_quick(self):
        """Quick IRF estimation with reduced iterations for auto-update."""
        if self.is_estimating:
            return
        
        self.is_estimating = True
        try:
            # Use reduced iterations for quick update (50 instead of user setting)
            self._estimate_irf_internal(rl_iterations=50, show_progress=False, show_message=False)
        finally:
            self.is_estimating = False
            
    def estimate_irf(self):
        """Estimate the IRF from the loaded decay data with full iterations."""
        if self.decay_data is None:
            QMessageBox.warning(
                self, "No Data",
                "Please load a decay file first."
            )
            return
            
        if IRFEstimator is None:
            QMessageBox.critical(
                self, "Missing Module",
                "IRFEstimator not available. Please check ChiSurf installation."
            )
            return
        
        # Use user-specified RL iterations
        rl_iterations = self.rl_iterations_spinbox.value()
        
        # Check if we have multiple decays to process
        if hasattr(self, 'all_decays') and len(self.all_decays) > 1:
            # Show progress dialog
            progress = QProgressDialog(
                "Processing multiple decays...", "Cancel", 0, 1, self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            try:
                # Process all decays
                avg_irf = self.process_all_decays(rl_iterations=rl_iterations)
                
                if avg_irf is not None:
                    # Update the estimator with the average IRF
                    self.irf_data = avg_irf
                    
                    # Create a dummy estimator for results display
                    if self.estimator is None:
                        self.estimator = IRFEstimator(
                            self.decay_data.reshape(-1, 1), dt=1.0
                        )
                    
                    # Update results and plots
                    self.update_results()
                    self.update_all_plots()
                    
                    # Enable save buttons
                    self.save_irf_button.setEnabled(True)
                    self.transfer_button.setEnabled(True)
                    
                    QMessageBox.information(
                        self, "Success",
                        f"Successfully processed {len(self.all_decays)} decays.\n"
                        "The average IRF has been computed and displayed."
                    )
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Estimation Error",
                    f"Failed to process decays:\n{str(e)}"
                )
                import traceback
                traceback.print_exc()
                
            finally:
                progress.close()
        else:
            # Single decay processing
            self._estimate_irf_internal(
                rl_iterations=rl_iterations, 
                show_progress=True, 
                show_message=True
            )
    
    def _estimate_irf_internal(self, rl_iterations=250, show_progress=True, show_message=True):
        """Internal method to estimate IRF with configurable iterations and UI feedback.
        
        Parameters
        ----------
        rl_iterations : int
            Number of Richardson-Lucy iterations to use
        show_progress : bool
            Whether to show progress dialog
        show_message : bool
            Whether to show completion message
        """
        if self.decay_data is None or IRFEstimator is None:
            return
        
        # Create progress dialog if requested
        progress = None
        if show_progress:
            progress = QProgressDialog("Estimating IRF...", "Cancel", 0, 5, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
        
        try:
            # Get parameters
            window_length = self.window_length_spinbox.value()
            # Ensure window length is odd
            if window_length % 2 == 0:
                window_length += 1
                self.window_length_spinbox.setValue(window_length)
                
            polyorder = self.polyorder_spinbox.value()
            regularization = self.regularization_spinbox.value()
            # Ensure regularization is odd
            if regularization > 1 and regularization % 2 == 0:
                regularization += 1
                self.regularization_spinbox.setValue(regularization)
            
            # Prepare data for estimation
            if progress:
                progress.setLabelText("Preparing data...")
                progress.setValue(1)
                QApplication.processEvents()
            
            # Start with background-corrected data
            decay_for_estimation = self.decay_data.copy()
            
            # Apply range selection if enabled
            if self.use_range_selection:
                # Zero out data outside the selected range (in channels)
                min_ch, max_ch = self.range_bounds
                mask = (self.channel_axis >= min_ch) & (self.channel_axis <= max_ch)
                decay_for_estimation[~mask] = 0.0
            
            # Create estimator
            if progress:
                progress.setLabelText("Initializing estimator...")
                QApplication.processEvents()
            
            # Create estimator with dt=1.0 (working in channels, not time)
            self.estimator = IRFEstimator(decay_for_estimation.reshape(-1, 1), dt=1.0)
            
            # Step 1: Find boundaries
            if progress:
                progress.setLabelText("Finding decay boundaries...")
                progress.setValue(2)
                QApplication.processEvents()
            
            self.estimator.find_t0_t1(
                window_length=window_length,
                polyorder=polyorder
            )
            
            # Step 2: Fit exponential
            if progress:
                progress.setLabelText("Fitting exponential decay...")
                progress.setValue(3)
                QApplication.processEvents()
            
            self.estimator.fit_exponential()
            self.estimator.generate_data_fit()
            self.estimator.generate_kernel()
            
            # Step 3: Richardson-Lucy deconvolution
            if progress:
                progress.setLabelText("Performing Richardson-Lucy deconvolution...")
                progress.setValue(4)
                QApplication.processEvents()
            
            self.estimator.richardson_lucy_deconvolution(
                iterations=rl_iterations,
                regularization=regularization
            )
            
            if progress is not None:
                progress.setValue(5)
            
            # Store IRF
            self.irf_data = self.estimator.irf[:, 0]
            
            # Update results
            self.update_results()
            
            # Update all plots
            self.update_all_plots()
            
            # Enable save buttons
            self.save_irf_button.setEnabled(True)
            self.transfer_button.setEnabled(True)
            
            # Only show success message if requested
            if show_message:
                QMessageBox.information(
                    self, "Success",
                    "IRF estimation completed successfully!"
                )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Estimation Error",
                f"Failed to estimate IRF:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
            
        finally:
            if progress is not None:
                progress.close()
            
    def update_results(self):
        """Update the results display with estimated parameters."""
        if self.estimator is None or self.estimator.params is None:
            return
            
        params = self.estimator.params
        
        # Calculate lifetime from decay rate
        # k is in units of 1/channel, convert to 1/ns using dt
        k_per_channel = params['k']
        k_per_ns = k_per_channel / self.dt  # Convert to 1/ns
        tau_channels = 1.0 / k_per_channel if k_per_channel > 0 else float('inf')
        tau_ns = tau_channels * self.dt  # Convert to ns
        
        self.lifetime_value.setText(f"{tau_ns:.4f} ns ({tau_channels:.2f} ch)")
        self.decay_rate_value.setText(f"{k_per_ns:.6f} ns⁻¹ ({k_per_channel:.6f} ch⁻¹)")
        self.amplitude_value.setText(f"{params['A'][0]:.2f}")
        self.offset_value.setText(f"{params['C'][0]:.2f}")
        
    def save_irf(self):
        """Save the estimated IRF in Jordi format with robust error handling."""
        if self.irf_data is None or len(self.irf_data) == 0:
            QMessageBox.warning(
                self, "No IRF",
                "Please estimate an IRF first."
            )
            return
            
        # Get save file path
        try:
            start_dir = str(getattr(cs, 'working_path', '') or '')
        except Exception:
            start_dir = ""
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save IRF", start_dir,
            "Jordi Files (*.dat);;All Files (*)"
        )
        
        if not file_path:
            return  # User cancelled
            
        # Ensure the file has the correct extension
        if not file_path.lower().endswith('.dat'):
            file_path += '.dat'
            
        # Ensure the data is in the correct format (1D array)
        irf_data = np.asarray(self.irf_data).flatten()

        # Debug logging
        if CHISURF_AVAILABLE:
            cs.logging.debug(f"Saving IRF to {file_path}")
            cs.logging.debug(f"IRF data shape: {irf_data.shape}")
            cs.logging.debug(f"IRF data sample: {irf_data[:5]}")

        # Save as Jordi format (two columns with same data)
        if _write_jordi is not None:
            _write_jordi(file_path, irf_data, irf_data)
        else:
            # Fallback: save as two-column text file with tab delimiter
            jordi_data = np.column_stack((irf_data, irf_data))
            np.savetxt(file_path, jordi_data, fmt='%.6f', delimiter='\t')

        # Verify the file was created and has content
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise RuntimeError("Failed to save IRF file or file is empty")

        if CHISURF_AVAILABLE:
            cs.logging.info(f"Successfully saved IRF to {file_path}")

        QMessageBox.information(
            self, "Success",
            f"IRF successfully saved to:\n{file_path}"
        )

            
    def on_transfer_clicked(self):
        """
        Handle the "Transfer to ChiSurf" button click event.
        Transfers the computed IRF to ChiSurf when clicked.
        """
        self.add_to_chisurf()

    def add_to_chisurf(self):
        """
        Add the computed IRF to cs as a dataset.
        This method saves the IRF to a temporary file and loads it into ChiSurf.
        """
        if self.irf_data is None or len(self.irf_data) == 0:
            QMessageBox.warning(
                self, "No IRF",
                "Please estimate an IRF first."
            )
            return
            
        irf_data = np.asarray(self.irf_data, dtype=float).flatten()
        if len(irf_data) == 0:
            raise ValueError("IRF data is empty")
        if not np.isfinite(irf_data).all():
            raise ValueError("IRF data contains NaN or infinite values")

        if CHISURF_AVAILABLE:
            cs.logging.debug(f"IRF data validated - shape: {irf_data.shape}, dtype: {irf_data.dtype}")

        import tempfile
        import os
        from pathlib import Path

        # Create a temporary file with a proper extension and explicit file handling
        if CHISURF_AVAILABLE:
            cs.logging.debug("Creating temporary file...")
            cs.logging.debug(f"IRF data type: {type(irf_data)}, shape: {irf_data.shape}, dtype: {irf_data.dtype}")
            cs.logging.debug(f"Sample data: {irf_data[:5]}")

        fd, tmp_path = tempfile.mkstemp(suffix='.dat')
        if CHISURF_AVAILABLE:
            cs.logging.debug(f"Temporary file created at: {tmp_path}")

        # Close the file descriptor before writing to avoid locking issues on Windows
        os.close(fd)

        # Create a temporary file with a random name in the same directory
        temp_dir = os.path.dirname(tmp_path)
        temp_file = os.path.join(temp_dir, f'temp_{os.urandom(8).hex()}.dat')

        # Use _write_jordi to write the data
        _write_jordi(temp_file, irf_data, irf_data)

        # Verify the temporary file was written correctly
        if not os.path.exists(temp_file):
            raise RuntimeError(f"Temporary file was not created at: {temp_file}")

        file_size = os.path.getsize(temp_file)
        if file_size == 0:
            raise RuntimeError(f"Temporary file is empty: {temp_file}")

        # Remove destination file if it exists
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        # Move the temporary file to the final location
        os.rename(temp_file, tmp_path)

        if CHISURF_AVAILABLE:
            cs.logging.debug(f"File saved to {tmp_path}")
            cs.logging.debug(f"File size: {os.path.getsize(tmp_path)} bytes")

            # Debug logging
            if CHISURF_AVAILABLE:
                cs.logging.debug(f"IRF data shape: {irf_data.shape}")
                cs.logging.debug(f"IRF data sample: {irf_data[:5]}")

                # Verify the final file exists and has content
                if os.path.exists(tmp_path):
                    file_size = os.path.getsize(tmp_path)
                    cs.logging.debug(f"Final file size: {file_size} bytes")
                    if file_size == 0:
                        cs.logging.warning("Warning: Final file is empty")


        # Verify the file was created and has content
        if not os.path.exists(tmp_path):
            raise RuntimeError(f"Temporary file was not created at: {tmp_path}")

        file_size = os.path.getsize(tmp_path)

        if CHISURF_AVAILABLE:
            cs.logging.debug(f"Temporary file created successfully: {tmp_path} ({file_size} bytes)")

        # Get the base filename for display
        filename = Path(tmp_path).name

        # Set up ChiSurf experiment settings for TCSPC data
        if CHISURF_AVAILABLE and hasattr(cs, 'cs'):
            # Configure the current setup for IRF data
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

            # Add the IRF dataset to ChiSurf
            cs.core.actions.dispatch(
                name="dataset.add",
                payload={"filename": tmp_path, "experiment_reader": None},
            )

            # Show success message
            QMessageBox.information(
                self, "Success",
                f"IRF '{filename}' has been transferred to ChiSurf."
            )

            cs.logging.info(f"Transferred IRF to ChiSurf: {filename}")

        else:
            # Fallback if ChiSurf is not available
            QMessageBox.information(
                self, "IRF Ready",
                f"IRF saved to:\n{tmp_path}\n\n"
                "You can now load this file as an IRF in your analysis."
            )



    def get_available_datasets(self):
        """Get a list of available datasets from ChiSurf."""
        if not CHISURF_AVAILABLE:
            return []
            
        try:
            # Use the same approach as in ConvolveWidget
            from chisurf.core.data import get_data, ExperimentalData
            
            # Get all datasets from ChiSurf's imported datasets
            all_curves = get_data(
                data_set=getattr(cs, "imported_datasets", []),
                curve_type='experiment'
            )
            
            # Filter for datasets with data and dt attributes
            datasets = [
                ds for ds in all_curves 
                if hasattr(ds, 'data') and hasattr(ds, 'dt')
            ]
            
            return datasets
            
        except Exception as e:
            cs.logging.error(f"Error getting available datasets: {str(e)}")
            return []
    
    def load_from_dataset(self):
        """Open a dialog to select a dataset from loaded experiments."""
        if not CHISURF_AVAILABLE:
            QMessageBox.warning(
                self, "Error",
                "ChiSurf integration is not available. Cannot load from datasets."
            )
            return
            
        try:
            # Create the dataset selector dialog
            from chisurf.gui.widgets.experiments import ExperimentalDataSelector
            
            # Create the selector
            self.dataset_selector = ExperimentalDataSelector(
                parent=None,
                change_event=self.on_dataset_selected,
                fit=None,  # Pass self as fit object to get experiment info
                experiment=None  # Will be set from fit if available
            )
            
            # Show the dialog
            self.dataset_selector.show()
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to open dataset selector: {str(e)}\n\n"
                f"{traceback.format_exc()}"
            )
    
    def on_dataset_selected(self):
        """Handle dataset selection from the ExperimentalDataSelector."""
        try:
            selected = self.dataset_selector.selected_dataset
            if selected is not None:
                self.load_dataset(selected)
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load selected dataset: {str(e)}"
            )
    
    def load_dataset(self, dataset):
        """Load data from a ChiSurf dataset."""
        try:
            # For DataCurve objects, use x and y attributes
            if hasattr(dataset, 'y') and hasattr(dataset, 'x'):
                # Get the decay data (counts) and time axis
                y_data = np.asarray(dataset.y, dtype=np.float32)
                x_data = np.asarray(dataset.x, dtype=np.float32)
                
                # Calculate dt from x_data if possible
                if len(x_data) > 1:
                    dt = float(np.mean(np.diff(x_data)))
                else:
                    dt = float(getattr(dataset, 'dt', 1.0))  # Fallback to dataset.dt or 1.0
                
                # Store reference to the dataset
                self.current_dataset = dataset
                self.current_file_path = getattr(dataset, 'filename', None)
                
                # Update UI with dataset information
                name = getattr(dataset, 'name', 'Unnamed')
                exp = getattr(dataset, 'experiment', None)
                exp_name = getattr(exp, 'name', 'Uncategorized') if exp else 'Uncategorized'
                self.file_label.setText(f"Dataset: {exp_name} - {name}")
                
                # Process the loaded data with time axis from dataset
                self.process_decay_data(
                    np.column_stack((x_data, y_data)),
                    dt,
                    time_source='dataset'  # Indicate time axis comes from dataset
                )
                
            # Fallback for older format with 'data' attribute
            elif hasattr(dataset, 'data'):
                data = np.asarray(dataset.data, dtype=np.float32)
                
                # Handle different data shapes
                if data.ndim == 1:
                    # If only y values are provided, create time axis using dt
                    dt = float(getattr(dataset, 'dt', 1.0))
                    time_axis = np.arange(len(data)) * dt
                    decay_data = np.column_stack((time_axis, data))
                elif data.ndim == 2 and data.shape[1] == 1:
                    # If data is (N,1), create time axis using dt
                    dt = float(getattr(dataset, 'dt', 1.0))
                    time_axis = np.arange(len(data)) * dt
                    decay_data = np.column_stack((time_axis, data.flatten()))
                elif data.ndim == 2 and data.shape[1] == 2:
                    # If data is (N,2) with time in first column and values in second
                    time_axis = data[:, 0]
                    decay_data = data
                    dt = float(np.mean(np.diff(time_axis))) if len(time_axis) > 1 else 1.0
                else:
                    raise ValueError(f"Unsupported data dimensions: {data.shape}")
                
                # Get time step from dataset if not already determined
                if 'dt' not in locals() and hasattr(dataset, 'dt'):
                    dt = float(dataset.dt)
                
                self.current_dataset = dataset
                self.current_file_path = getattr(dataset, 'filename', None)
                
                # Update UI
                name = getattr(dataset, 'name', 'Unnamed')
                exp = getattr(dataset, 'experiment', None)
                exp_name = getattr(exp, 'name', 'Uncategorized') if exp else 'Uncategorized'
                self.file_label.setText(f"Dataset: {exp_name} - {name}")
                
                self.process_decay_data(
                    decay_data,
                    dt,
                    time_source='calculated' if 'time_axis' in locals() else 'auto'
                )
                
            else:
                raise ValueError("Unsupported dataset format. Expected 'x' and 'y' or 'data' attributes.")
                
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load dataset: {str(e)}\n\n"
                f"Dataset type: {type(dataset).__name__}\n"
                f"Data shape: {getattr(dataset, 'y', getattr(dataset, 'data', None)) is not None and getattr(dataset, 'y', getattr(dataset, 'data', None)).shape}\n"
                f"Available attributes: {', '.join([a for a in dir(dataset) if not a.startswith('_')])}"
            )
            raise
    
    def on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair and tooltip display."""
        if self.main_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.main_plot.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            
            # Update crosshair position
            self.crosshair_v.setPos(x)
            self.crosshair_h.setPos(y)
            
            # Show tooltip with coordinates
            if hasattr(self, 'channel_axis') and self.channel_axis is not None:
                # Find nearest data point
                idx = np.abs(self.channel_axis - x).argmin()
                if 0 <= idx < len(self.channel_axis) and hasattr(self, 'decay_data'):
                    x_val = self.channel_axis[idx]
                    y_val = self.decay_data[idx] if idx < len(self.decay_data) else 0
                    self.main_plot.setToolTip(f"Time: {x_val:.2f} ns, Intensity: {y_val:.1f}")
    
    def update_plot_axes(self):
        """Update plot axes based on the current time axis settings.
        """
        if hasattr(self, 'channel_axis') and self.channel_axis is not None:
            # Update x-axis label to show time units
            self.main_plot.setLabel('bottom', 'Time (ns)')
            
            # Update data stats
            if len(self.channel_axis) > 1:
                duration = self.channel_axis[-1] - self.channel_axis[0]
                self.data_stats_label.setText(
                    f"Data points: {len(self.channel_axis):,} | "
                    f"Duration: {duration:.2f} ns | "
                    f"dt: {self.dt:.4f} ns"
                )
    
    def process_decay_data(self, decay_data, dt, time_source='auto'):
        """Process loaded decay data (common for both file and dataset loading).
        
        Args:
            decay_data: numpy array or DataCurve object
                If array: shape (N,) or (N,2) where first column is time, second is intensity
                If DataCurve: uses x and y attributes
            dt: float, time step in nanoseconds (used as fallback if time axis not provided)
            time_source: str, source of the time axis ('auto', 'dataset', 'calculated')
        """
        try:
            # Handle DataCurve objects
            if hasattr(decay_data, 'y') and hasattr(decay_data, 'x'):
                # Extract data from DataCurve
                y_data = np.asarray(decay_data.y, dtype=np.float32)
                x_data = np.asarray(decay_data.x, dtype=np.float32)
                
                # Ensure data is 1D
                if y_data.ndim > 1:
                    y_data = y_data.flatten()
                if x_data.ndim > 1:
                    x_data = x_data.flatten()
                
                # Create time-intensity pairs
                decay_data = np.column_stack((x_data, y_data))
                
                # Calculate dt from x_data if possible
                if len(x_data) > 1:
                    dt = float(np.mean(np.diff(x_data)))
            else:
                # Handle numpy arrays
                decay_data = np.asarray(decay_data, dtype=np.float32)
                
                # Handle different input shapes
                if decay_data.ndim == 1:  # (N,) format - use provided dt to create time axis
                    time_axis = np.arange(len(decay_data)) * dt
                    decay_data = np.column_stack((time_axis, decay_data))
                elif decay_data.ndim == 2:
                    if decay_data.shape[1] == 1:  # (N,1) -> (N,2)
                        time_axis = np.arange(len(decay_data)) * dt
                        decay_data = np.column_stack((time_axis, decay_data.flatten()))
                    elif decay_data.shape[1] >= 2:  # (N,2+) -> take first two columns
                        decay_data = decay_data[:, :2]
                        # Update dt based on actual time axis if possible
                        if len(decay_data) > 1:
                            dt = float(np.mean(np.diff(decay_data[:, 0])))
                else:
                    raise ValueError(f"Unsupported data shape: {decay_data.shape}")
            
            # Ensure we have valid data
            if np.any(np.isnan(decay_data)) or np.any(np.isinf(decay_data)):
                raise ValueError("Data contains NaN or Inf values")
                
            # Store the time axis and intensity data separately
            self.channel_axis = decay_data[:, 0]  # Time axis (x)
            self.decay_data = decay_data[:, 1]    # Intensity data (y)
            self.decay_data_original = self.decay_data.copy()
            
            # Update dt and UI
            self.dt = dt
            self.dt_spinbox.setValue(dt)
            
            # Update time axis info
            if len(self.channel_axis) > 1:
                t_min = np.min(self.channel_axis)
                t_max = np.max(self.channel_axis)
                t_range = t_max - t_min
                
                # Update time source info
                if time_source == 'dataset':
                    source_info = "Using time axis from dataset"
                    self.dt_spinbox.setEnabled(False)
                elif time_source == 'calculated':
                    source_info = f"Calculated time axis (dt = {dt:.4f} ns)"
                    self.dt_spinbox.setEnabled(True)
                else:
                    source_info = f"Using default time axis (dt = {dt:.4f} ns)"
                    self.dt_spinbox.setEnabled(True)
                
                self.time_axis_label.setText(
                    f"Time range: {t_min:.2f} to {t_max:.2f} ns "
                    f"(Δ = {t_range:.2f} ns, {len(self.channel_axis)} points)"
                )

                # Update plot axes and stats
                self.update_plot_axes()
            
            # Apply any background correction
            self.on_background_changed(self.background_spinbox.value())
            
            # Update plots
            self.update_all_plots()
            
            # Enable save button
            self.estimate_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self, "Data Processing Error",
                f"Failed to process data: {str(e)}\n"
                f"Data type: {type(decay_data).__name__}\n"
                f"Data shape: {getattr(decay_data, 'shape', 'N/A') if hasattr(decay_data, 'shape') else 'N/A'}"
            )
            raise
        self.save_irf_button.setEnabled(True)
        self.transfer_button.setEnabled(True)


# Standalone execution
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IRFEstimatorPlugin()
    window.show()
    sys.exit(app.exec_())

# Plugin execution
elif __name__ == "plugin":
    window = IRFEstimatorPlugin()
    window.show()
