"""
Jordi G-Factor Calculator

This standalone plugin calculates the g-factor based on tail matching for Jordi files.
It allows users to:
- Load and display Jordi files
- Select a tail matching region using interactive controls
- Calculate the g-factor based on the selected region
- Display the g-factor and its standard deviation in copyable text fields
- Visualize the tail-matched decays in a separate plot
- Display decay curves in semilog plots (logarithmic y-axis) for better visualization of exponential decays
- Update calculations when the region selection changes
- Apply background correction with a separate background region selector
- Display both corrected and uncorrected g-factors

The g-factor is an important correction factor in fluorescence anisotropy measurements,
accounting for the different detection efficiencies of the parallel and perpendicular
emission components.

Background correction is useful for removing constant offsets in the data, which can
improve the accuracy of the g-factor calculation, especially for data with significant
background signal.

This plugin can run independently of ChiSurf or as a ChiSurf plugin.
"""

name = "Fluorescence decay:Jordi G-Factor Calculator"

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QGridLayout,
    QDoubleSpinBox, QLineEdit, QCheckBox
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg

# No ChiSurf dependencies

# Simple data class to replace chisurf.data.DataCurve
class DataCurve:
    def __init__(self, x=None, y=None, name=None):
        self.x = x
        self.y = y
        self.name = name


class JordiGFactorCalculator(QWidget):
    """Main widget for the Jordi G-Factor Calculator plugin."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jordi G-Factor Calculator")
        
        # Data storage
        self.jordi_data = None
        self.time_axis = None
        self.parallel_data = None
        self.perpendicular_data = None
        self.g_factor = None
        self.g_factor_stddev = None
        self.region_bounds = [0, 100]  # Default region bounds
        self.bg_region_bounds = [0, 100]  # Default background region bounds
        self.use_background_correction = False
        
        # Create UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Controls layout
        controls_layout = QGridLayout()
        
        # File loading
        self.load_button = QPushButton("Load Jordi File")
        self.load_button.clicked.connect(self.load_jordi_file)
        self.file_label = QLabel("No file loaded")
        controls_layout.addWidget(self.load_button, 0, 0)
        controls_layout.addWidget(self.file_label, 0, 1, 1, 3)
        
        # G-factor display
        self.g_factor_label = QLabel("G-Factor: ")
        self.g_factor_value = QLineEdit("N/A")
        self.g_factor_value.setReadOnly(True)  # Make it non-editable
        self.g_factor_stddev_label = QLabel("StdDev: ")
        self.g_factor_stddev_value = QLineEdit("N/A")
        self.g_factor_stddev_value.setReadOnly(True)  # Make it non-editable
        controls_layout.addWidget(self.g_factor_label, 1, 0)
        controls_layout.addWidget(self.g_factor_value, 1, 1)
        controls_layout.addWidget(self.g_factor_stddev_label, 1, 2)
        controls_layout.addWidget(self.g_factor_stddev_value, 1, 3)
        
        # Corrected G-factor display (initially hidden)
        self.corrected_g_factor_label = QLabel("Corrected G-Factor: ")
        self.corrected_g_factor_value = QLineEdit("N/A")
        self.corrected_g_factor_value.setReadOnly(True)
        self.corrected_g_factor_stddev_label = QLabel("Corrected StdDev: ")
        self.corrected_g_factor_stddev_value = QLineEdit("N/A")
        self.corrected_g_factor_stddev_value.setReadOnly(True)
        controls_layout.addWidget(self.corrected_g_factor_label, 3, 0)
        controls_layout.addWidget(self.corrected_g_factor_value, 3, 1)
        controls_layout.addWidget(self.corrected_g_factor_stddev_label, 3, 2)
        controls_layout.addWidget(self.corrected_g_factor_stddev_value, 3, 3)
        
        # Background values display (initially hidden)
        self.bg_parallel_label = QLabel("BG Parallel: ")
        self.bg_parallel_value = QLineEdit("N/A")
        self.bg_parallel_value.setReadOnly(True)
        self.bg_perpendicular_label = QLabel("BG Perpendicular: ")
        self.bg_perpendicular_value = QLineEdit("N/A")
        self.bg_perpendicular_value.setReadOnly(True)
        controls_layout.addWidget(self.bg_parallel_label, 4, 0)
        controls_layout.addWidget(self.bg_parallel_value, 4, 1)
        controls_layout.addWidget(self.bg_perpendicular_label, 4, 2)
        controls_layout.addWidget(self.bg_perpendicular_value, 4, 3)
        
        # Initially hide corrected g-factor and background displays
        self.corrected_g_factor_label.setVisible(False)
        self.corrected_g_factor_value.setVisible(False)
        self.corrected_g_factor_stddev_label.setVisible(False)
        self.corrected_g_factor_stddev_value.setVisible(False)
        self.bg_parallel_label.setVisible(False)
        self.bg_parallel_value.setVisible(False)
        self.bg_perpendicular_label.setVisible(False)
        self.bg_perpendicular_value.setVisible(False)
        
        # Background correction checkbox
        self.bg_correction_checkbox = QCheckBox("Background Correction")
        self.bg_correction_checkbox.setChecked(False)
        self.bg_correction_checkbox.stateChanged.connect(self.on_bg_correction_changed)
        controls_layout.addWidget(self.bg_correction_checkbox, 2, 0, 1, 4)
        
        main_layout.addLayout(controls_layout)
        
        # Main plot widget for full decay curves
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Intensity')
        self.plot_widget.setLabel('bottom', 'Time (ns)')
        self.plot_widget.setTitle('Full Decay Curves')
        self.plot_widget.addLegend()
        # Set y-axis to logarithmic scale
        self.plot_widget.setLogMode(x=False, y=True)
        
        # Create region selector for tail matching
        self.region = pg.LinearRegionItem(
            values=self.region_bounds,
            brush=pg.mkBrush(color=(50, 50, 200, 50)),
            movable=True
        )
        self.region.sigRegionChanged.connect(self.on_region_changed)
        
        # Create region selector for background (initially hidden)
        self.bg_region = pg.LinearRegionItem(
            values=self.bg_region_bounds,
            brush=pg.mkBrush(color=(200, 50, 50, 50)),  # Red tint for background
            movable=True
        )
        self.bg_region.sigRegionChanged.connect(self.on_bg_region_changed)
        
        # Second plot widget for tail-matched decays
        self.tail_plot_widget = pg.PlotWidget()
        self.tail_plot_widget.setLabel('left', 'Intensity')
        self.tail_plot_widget.setLabel('bottom', 'Time (ns)')
        self.tail_plot_widget.setTitle('Tail-Matched Decays')
        self.tail_plot_widget.addLegend()
        # Set y-axis to logarithmic scale
        self.tail_plot_widget.setLogMode(x=False, y=True)
        
        # Create a layout for plots side by side
        plots_layout = QHBoxLayout()
        plots_layout.addWidget(self.plot_widget)
        plots_layout.addWidget(self.tail_plot_widget)
        
        # Add plots layout to main layout
        main_layout.addLayout(plots_layout)
        
        self.setLayout(main_layout)
        self.resize(1000, 600)
    
    def load_jordi_file(self):
        """Load a Jordi file and display it.
        
        Jordi files are loaded directly using numpy.loadtxt and split into two equal chunks:
        - First half: VV channel (parallel)
        - Second half: VH channel (perpendicular)
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Jordi File", "", "Data Files (*.dat);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self.file_label.setText(file_path)
        
        # Load the Jordi file using numpy.loadtxt
        try:
            jordi_data = np.loadtxt(file_path)
            
            # Split the data into two equal chunks for VV and VH channels
            half_length = len(jordi_data) // 2
            vv_data = jordi_data[:half_length]  # Parallel (VV)
            vh_data = jordi_data[half_length:]  # Perpendicular (VH)
            
            # Create time axis (assuming equal time steps)
            # For a more realistic time axis, use a range from 0 to 10 ns (matching test data)
            self.time_axis = np.linspace(0, 10, len(vv_data))
            
            # Create DataCurve objects manually
            self.parallel_data = DataCurve(
                x=self.time_axis,
                y=vv_data,
                name="Parallel"
            )
            
            self.perpendicular_data = DataCurve(
                x=self.time_axis,
                y=vh_data,
                name="Perpendicular"
            )
        except Exception as e:
            self.file_label.setText(f"Error loading file: {str(e)}")
            return
        
        # Update plot
        self.update_plot()
        
        # Add region selector to plot
        self.plot_widget.addItem(self.region)
        
        # Set initial region to the last 20% of the data
        data_length = len(self.time_axis)
        self.region_bounds = [
            self.time_axis[int(data_length * 0.7)],
            self.time_axis[int(data_length * 0.9)]
        ]
        self.region.setRegion(self.region_bounds)
        
        # Set initial background region to the first 10% of the data
        self.bg_region_bounds = [
            self.time_axis[int(data_length * 0.05)],
            self.time_axis[int(data_length * 0.15)]
        ]
        self.bg_region.setRegion(self.bg_region_bounds)
        
        # Add background region selector if checkbox is checked
        if self.use_background_correction:
            self.plot_widget.addItem(self.bg_region)
        
        # Calculate initial g-factor
        self.calculate_g_factor()
    
    def update_plot(self):
        """Update the plot with current data."""
        if self.parallel_data is None or self.perpendicular_data is None:
            return
        
        # Clear the plot but remember if regions were there
        had_region = self.region in self.plot_widget.items()
        had_bg_region = hasattr(self, 'bg_region') and self.bg_region in self.plot_widget.items()
        self.plot_widget.clear()
        
        # Plot parallel data
        self.plot_widget.plot(
            self.time_axis, 
            self.parallel_data.y, 
            pen=pg.mkPen('b', width=2),
            name='Parallel'
        )
        
        # Plot perpendicular data
        self.plot_widget.plot(
            self.time_axis, 
            self.perpendicular_data.y, 
            pen=pg.mkPen('r', width=2),
            name='Perpendicular'
        )
        
        # Re-add region selectors if they were there before
        if had_region:
            self.plot_widget.addItem(self.region)
            
        if had_bg_region:
            self.plot_widget.addItem(self.bg_region)
    
    def on_bg_correction_changed(self):
        """Handle changes to the background correction checkbox."""
        self.use_background_correction = self.bg_correction_checkbox.isChecked()
        
        # Show or hide the background region selector
        if self.use_background_correction and hasattr(self, 'bg_region'):
            self.plot_widget.addItem(self.bg_region)
        elif hasattr(self, 'bg_region') and self.bg_region in self.plot_widget.items():
            self.plot_widget.removeItem(self.bg_region)
        
        # Show or hide the corrected g-factor display
        self.corrected_g_factor_label.setVisible(self.use_background_correction)
        self.corrected_g_factor_value.setVisible(self.use_background_correction)
        self.corrected_g_factor_stddev_label.setVisible(self.use_background_correction)
        self.corrected_g_factor_stddev_value.setVisible(self.use_background_correction)
        
        # Show or hide the background values display
        self.bg_parallel_label.setVisible(self.use_background_correction)
        self.bg_parallel_value.setVisible(self.use_background_correction)
        self.bg_perpendicular_label.setVisible(self.use_background_correction)
        self.bg_perpendicular_value.setVisible(self.use_background_correction)
            
        # Recalculate g-factor with the new settings
        self.calculate_g_factor()
    
    def on_region_changed(self):
        """Handle changes to the selected region."""
        self.region_bounds = self.region.getRegion()
        self.calculate_g_factor()
        
    def on_bg_region_changed(self):
        """Handle changes to the background region."""
        self.bg_region_bounds = self.bg_region.getRegion()
        self.calculate_g_factor()
    
    def calculate_g_factor(self):
        """Calculate the g-factor based on the selected region."""
        if self.parallel_data is None or self.perpendicular_data is None:
            return
        
        # Get region bounds for tail matching
        min_time, max_time = self.region_bounds
        
        # Find indices corresponding to the region
        min_idx = np.argmin(np.abs(self.time_axis - min_time))
        max_idx = np.argmin(np.abs(self.time_axis - max_time))
        
        # Extract data in the region
        parallel_region = self.parallel_data.y[min_idx:max_idx]
        perpendicular_region = self.perpendicular_data.y[min_idx:max_idx]
        time_region = self.time_axis[min_idx:max_idx]
        
        # Calculate uncorrected g-factor
        g_factors_uncorrected = parallel_region / perpendicular_region
        valid_indices_uncorrected = ~np.isnan(g_factors_uncorrected) & ~np.isinf(g_factors_uncorrected) & (g_factors_uncorrected > 0)
        valid_g_factors_uncorrected = g_factors_uncorrected[valid_indices_uncorrected]
        
        if len(valid_g_factors_uncorrected) > 0:
            # Calculate mean and standard deviation for uncorrected g-factor
            g_factor_uncorrected = np.mean(valid_g_factors_uncorrected)
            g_factor_stddev_uncorrected = np.std(valid_g_factors_uncorrected)
            
            # Update uncorrected g-factor display
            self.g_factor_value.setText(f"{g_factor_uncorrected:.4f}")
            self.g_factor_stddev_value.setText(f"{g_factor_stddev_uncorrected:.4f}")
            
            # Select all text to make it easy to copy
            self.g_factor_value.selectAll()
            self.g_factor_value.clearFocus()
        else:
            self.g_factor_value.setText("N/A")
            self.g_factor_stddev_value.setText("N/A")
        
        # Apply background correction if enabled
        if self.use_background_correction:
            # Get background region bounds
            bg_min_time, bg_max_time = self.bg_region_bounds
            
            # Find indices corresponding to the background region
            bg_min_idx = np.argmin(np.abs(self.time_axis - bg_min_time))
            bg_max_idx = np.argmin(np.abs(self.time_axis - bg_max_time))
            
            # Extract background data
            bg_parallel = self.parallel_data.y[bg_min_idx:bg_max_idx]
            bg_perpendicular = self.perpendicular_data.y[bg_min_idx:bg_max_idx]
            
            # Calculate average background levels
            bg_parallel_avg = np.mean(bg_parallel)
            bg_perpendicular_avg = np.mean(bg_perpendicular)
            
            # Update background value display
            self.bg_parallel_value.setText(f"{bg_parallel_avg:.4f}")
            self.bg_perpendicular_value.setText(f"{bg_perpendicular_avg:.4f}")
            
            # Subtract background from the data
            parallel_region_corrected = parallel_region - bg_parallel_avg
            perpendicular_region_corrected = perpendicular_region - bg_perpendicular_avg
            
            # Ensure no negative values after background subtraction
            parallel_region_corrected = np.maximum(parallel_region_corrected, 0)
            perpendicular_region_corrected = np.maximum(perpendicular_region_corrected, 0)
            
            # Calculate g-factor as the ratio of background-corrected parallel to perpendicular
            g_factors_corrected = parallel_region_corrected / perpendicular_region_corrected
            
            # Filter out invalid values
            valid_indices_corrected = ~np.isnan(g_factors_corrected) & ~np.isinf(g_factors_corrected) & (g_factors_corrected > 0)
            valid_g_factors_corrected = g_factors_corrected[valid_indices_corrected]
            
            if len(valid_g_factors_corrected) > 0:
                # Calculate mean and standard deviation for corrected g-factor
                g_factor_corrected = np.mean(valid_g_factors_corrected)
                g_factor_stddev_corrected = np.std(valid_g_factors_corrected)
                
                # Update corrected g-factor display
                self.corrected_g_factor_value.setText(f"{g_factor_corrected:.4f}")
                self.corrected_g_factor_stddev_value.setText(f"{g_factor_stddev_corrected:.4f}")
                
                # Store the corrected g-factor for the tail plot
                self.g_factor = g_factor_corrected
                
                # Update tail-matched decay plot with corrected data
                self.update_tail_plot(time_region, parallel_region_corrected, perpendicular_region_corrected)
            else:
                self.corrected_g_factor_value.setText("N/A")
                self.corrected_g_factor_stddev_value.setText("N/A")
                
                # Use uncorrected g-factor for the tail plot
                self.g_factor = g_factor_uncorrected
                
                # Update tail-matched decay plot with uncorrected data
                self.update_tail_plot(time_region, parallel_region, perpendicular_region)
        else:
            # Use uncorrected g-factor for the tail plot
            self.g_factor = g_factor_uncorrected
            
            # Update tail-matched decay plot with uncorrected data
            self.update_tail_plot(time_region, parallel_region, perpendicular_region)
            
    def update_tail_plot(self, time_region, parallel_region, perpendicular_region):
        """Update the tail-matched decay plot with the entire range data."""
        self.tail_plot_widget.clear()
        
        # Apply background correction to the entire dataset if enabled
        if self.use_background_correction:
            # Get background region bounds
            bg_min_time, bg_max_time = self.bg_region_bounds
            
            # Find indices corresponding to the background region
            bg_min_idx = np.argmin(np.abs(self.time_axis - bg_min_time))
            bg_max_idx = np.argmin(np.abs(self.time_axis - bg_max_time))
            
            # Extract background data
            bg_parallel = self.parallel_data.y[bg_min_idx:bg_max_idx]
            bg_perpendicular = self.perpendicular_data.y[bg_min_idx:bg_max_idx]
            
            # Calculate average background levels
            bg_parallel_avg = np.mean(bg_parallel)
            bg_perpendicular_avg = np.mean(bg_perpendicular)
            
            # Update background value display
            self.bg_parallel_value.setText(f"{bg_parallel_avg:.4f}")
            self.bg_perpendicular_value.setText(f"{bg_perpendicular_avg:.4f}")
            
            # Subtract background from the entire dataset
            parallel_corrected = np.maximum(self.parallel_data.y - bg_parallel_avg, 0)
            perpendicular_corrected = np.maximum(self.perpendicular_data.y - bg_perpendicular_avg, 0)
            
            # Scale background-corrected perpendicular data by g-factor
            scaled_perpendicular = perpendicular_corrected * self.g_factor
            
            # Plot background-corrected parallel data
            self.tail_plot_widget.plot(
                self.time_axis, 
                parallel_corrected, 
                pen=pg.mkPen('b', width=2),
                name='Parallel (BG corrected)'
            )
            
            # Plot scaled background-corrected perpendicular data
            self.tail_plot_widget.plot(
                self.time_axis, 
                scaled_perpendicular, 
                pen=pg.mkPen('r', width=2, style=Qt.DashLine),
                name=f'Perpendicular × G ({self.g_factor:.4f}) (BG corrected)'
            )
            
            # Update the plot title to indicate background correction
            self.tail_plot_widget.setTitle('Background-Corrected Tail-Matched Decays')
        else:
            # Scale perpendicular data by g-factor without background correction
            scaled_perpendicular = self.perpendicular_data.y * self.g_factor
            
            # Plot parallel data for the entire range
            self.tail_plot_widget.plot(
                self.time_axis, 
                self.parallel_data.y, 
                pen=pg.mkPen('b', width=2),
                name='Parallel'
            )
            
            # Plot scaled perpendicular data for the entire range
            self.tail_plot_widget.plot(
                self.time_axis, 
                scaled_perpendicular, 
                pen=pg.mkPen('r', width=2, style=Qt.DashLine),
                name=f'Perpendicular × G ({self.g_factor:.4f})'
            )
            
            # Update the plot title to indicate the entire range is being displayed
            self.tail_plot_widget.setTitle('Tail-Matched Decays (Entire Range)')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JordiGFactorCalculator()
    window.show()
    sys.exit(app.exec_())

elif __name__ == "plugin":
    window = JordiGFactorCalculator()
    window.show()