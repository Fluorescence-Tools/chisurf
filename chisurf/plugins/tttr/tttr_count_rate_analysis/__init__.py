"""
Count Rate Analysis Plugin

This plugin allows file drops of TTTR files and computes the count rate in each file.
It plots the count rate as a function of the file and computes mean count rate and 
standard deviation of count rate over all dropped files.

Features:
- Drag and drop TTTR files for analysis
- Compute count rates for all combinations of windows and detectors
- Display count rates for each file
- Calculate and display mean and standard deviation of count rates
- Report count rates for individual channels
- Use the DetectorWizardPage for channel definition
- Export results table to a text file (via UI button or programmatically)
- Command-line interface (CLI) for batch processing

CLI Usage:
The plugin provides a command-line interface for batch processing of TTTR files:

    csc_count_rate --help                    # Show help
    csc_count_rate list-setups SETUP_FILE    # List available detector setups
    csc_count_rate analyze FILE1 FILE2...    # Analyze TTTR files
        --setup-file SETUP_FILE              # Specify detector setups file
        --setup-name NAME                    # Specify setup name
        --output OUTPUT_FILE                 # Save results to file
        --verbose                            # Enable verbose output

Example:
    csc_count_rate analyze data/*.ptu --setup-file detector_setups.json --output results.txt
"""

name = "TTTR:Analysis:Count Rate Analysis"

# Expose the plugin CLI through chisurf.cli
cli_entrypoint = "count-rate=chisurf.plugins.tttr.tttr_count_rate_analysis.cli:cli"

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from qtpy.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QMessageBox, QLineEdit, QSplitter, QToolButton
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QDragEnterEvent, QDropEvent

import pyqtgraph as pg
import tttrlib

from chisurf.gui.widgets.wizard.tttr_channeldefinition import DetectorWizardPage

class CountRateAnalyzer(QWidget):
    """Main widget for the Count Rate Analysis plugin."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Count Rate Analysis")
        
        # Data storage
        self.tttr_files = []  # List of loaded TTTR files
        self.count_rates = {}  # Dictionary to store count rates for each file and channel
        self.n_photons = {}  # Dictionary to store number of photons for each file and channel
        self.measurement_times = {}  # Dictionary to store measurement times for each file
        self.detector_wizard_page = None  # Will hold the DetectorWizardPage instance
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Create UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Create a tab widget for organizing sections
        tab_widget = QTabWidget()
        
        # First tab: Channel definition (DetectorWizardPage)
        self.detector_wizard_page = DetectorWizardPage(
            show_edit_json=False,
            show_save=False,
            show_setups_file=True,
            show_setup_selection=True,
            show_help=True,
            show_tttr_reading=True,
            show_tables=True,
            show_add_inputs=True
        )
        tab_widget.addTab(self.detector_wizard_page, "Channel Definition")
        
        # Second tab: File controls and list
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        
        # Controls layout
        controls_layout = QHBoxLayout()
        
        # File loading button
        self.load_button = QPushButton("Load TTTR Files")
        self.load_button.clicked.connect(self.load_tttr_files)
        controls_layout.addWidget(self.load_button)
        
        # Clear files button
        self.clear_button = QPushButton("Clear Files")
        self.clear_button.clicked.connect(self.clear_files)
        controls_layout.addWidget(self.clear_button)
        
        # Calculate button
        self.calculate_button = QPushButton("Calculate Count Rates")
        self.calculate_button.clicked.connect(self.calculate_count_rates)
        controls_layout.addWidget(self.calculate_button)
        
        # Add controls to file layout
        file_layout.addLayout(controls_layout)
        
        # File list table
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(2)
        self.file_table.setHorizontalHeaderLabels(["File", "Status"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        file_layout.addWidget(self.file_table)
        
        # Add file tab to tab widget
        tab_widget.addTab(file_tab, "Files")
        
        # Third tab: Results
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Create a splitter to separate table and plot
        results_splitter = QSplitter(Qt.Vertical)
        
        # Add a button to save the table as a text file
        save_table_button = QToolButton()
        save_table_button.setText("Save Table")
        save_table_button.clicked.connect(self.save_table_as_txt)
        results_layout.addWidget(save_table_button)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Channel", 
            "Mean Count Rate (kHz)", 
            "Std Count Rate (kHz)",
            "#Photons",
            "Measurement Time (s)"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_splitter.addWidget(self.results_table)
        
        # Plot widget for count rates
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Count Rate (kHz)')
        self.plot_widget.setLabel('bottom', 'File Index')
        self.plot_widget.setTitle('Count Rates by File')
        self.plot_widget.addLegend()
        results_splitter.addWidget(self.plot_widget)
        
        # Set initial sizes for the splitter (equal sizes)
        results_splitter.setSizes([200, 200])
        
        # Add the splitter to the results layout
        results_layout.addWidget(results_splitter)
        
        # Add results tab to tab widget
        tab_widget.addTab(results_tab, "Results")
        
        # Add tab widget to main layout
        main_layout.addWidget(tab_widget)
        
        self.setLayout(main_layout)
        self.resize(640, 480)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for file drops."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events for file drops."""
        urls = event.mimeData().urls()
        file_paths = [url.toLocalFile() for url in urls]
        self.add_tttr_files(file_paths)
        event.acceptProposedAction()
    
    def load_tttr_files(self):
        """Open a file dialog to load TTTR files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load TTTR Files", "", "All Files (*)"
        )
        
        if file_paths:
            self.add_tttr_files(file_paths)
    
    def add_tttr_files(self, file_paths: List[str]):
        """Add TTTR files to the list and update the UI.
        
        Files are sorted lexically by their basename in the table.
        """
        # Add new files to the list
        new_files_added = False
        for file_path in file_paths:
            if file_path not in self.tttr_files:
                self.tttr_files.append(file_path)
                new_files_added = True
        
        # If new files were added, sort the list and rebuild the table
        if new_files_added:
            # Sort files lexically by basename
            self.tttr_files.sort(key=lambda path: os.path.basename(path).lower())
            
            # Rebuild the table
            self.file_table.setRowCount(0)
            for file_path in self.tttr_files:
                row = self.file_table.rowCount()
                self.file_table.insertRow(row)
                self.file_table.setItem(row, 0, QTableWidgetItem(os.path.basename(file_path)))
                self.file_table.setItem(row, 1, QTableWidgetItem("Loaded"))
    
    def clear_files(self):
        """Clear all loaded files."""
        self.tttr_files = []
        self.count_rates = {}
        self.n_photons = {}
        self.measurement_times = {}
        self.file_table.setRowCount(0)
        self.results_table.setRowCount(0)
        self.plot_widget.clear()
    
    def calculate_count_rates(self):
        """Calculate count rates for all files and channels."""
        if not self.tttr_files:
            QMessageBox.warning(self, "No Files", "Please load TTTR files first.")
            return
        
        # Get channels from DetectorWizardPage
        channels = self.detector_wizard_page.channels()
        if not channels:
            QMessageBox.warning(self, "No Channels", "Please define channels in the detector wizard.")
            return
        
        # Clear previous results
        self.count_rates = {}
        self.n_photons = {}
        self.measurement_times = {}
        self.plot_widget.clear()
        
        # Process each file
        for file_idx, file_path in enumerate(self.tttr_files):
            # Update status in table
            self.file_table.setItem(file_idx, 1, QTableWidgetItem("Processing..."))
            QApplication.processEvents()  # Update UI

            # Load TTTR file
            tttr = tttrlib.TTTR(file_path)

            # Get macro time resolution (in seconds)
            header = tttr.header
            macro_time_resolution = header.macro_time_resolution

            # Calculate count rates for each channel
            file_count_rates = {}
            file_n_photons = {}
            measurement_time = tttr.macro_times[-1] * macro_time_resolution  # in seconds
            
            # Store measurement time for this file
            self.measurement_times[file_path] = measurement_time

            for channel_name, channel_info_list in channels.items():
                channel_count_rates = []
                channel_n_photons = []

                for channel_info in channel_info_list:
                    # Extract channel parameters
                    window_range = channel_info["window_range"]
                    detector_chs = channel_info["detector_chs"]
                    micro_time_range = channel_info["micro_time_range"]

                    # Filter TTTR data by detector channels
                    tttr_filtered = tttr.get_tttr_by_channel(detector_chs)

                    # Filter by micro time range if specified
                    print(micro_time_range)
                    if micro_time_range:
                        micro_times = tttr_filtered.micro_times
                        micro_mask = (micro_times >= micro_time_range[0]) & (micro_times <= micro_time_range[1])
                        # Convert int64 indices to int32 to avoid TypeError
                        selection_indices = np.where(micro_mask)[0].astype(np.int32)
                        tttr_filtered = tttr_filtered.get_tttr_by_selection(selection_indices)

                    # Filter by macro time window if specified
                    if window_range:
                        micro_times = tttr_filtered.micro_times
                        macro_mask = (micro_times >= window_range[0]) & (micro_times <= window_range[1])
                        # Convert int64 indices to int32 to avoid TypeError
                        selection_indices = np.where(macro_mask)[0].astype(np.int32)
                        tttr_filtered = tttr_filtered.get_tttr_by_selection(selection_indices)

                    # Calculate count rate (photons per second)
                    n_photons = len(tttr_filtered.macro_times)
                    channel_n_photons.append(n_photons)

                    if measurement_time > 0:
                        count_rate = n_photons / measurement_time  # in Hz
                        channel_count_rates.append(count_rate)

                # Store average count rate and total photons for this channel
                if channel_count_rates:
                    file_count_rates[channel_name] = np.mean(channel_count_rates)
                    file_n_photons[channel_name] = sum(channel_n_photons)
                else:
                    file_count_rates[channel_name] = 0.0
                    file_n_photons[channel_name] = 0

            # Store count rates and photon counts for this file
            self.count_rates[file_path] = file_count_rates
            self.n_photons[file_path] = file_n_photons

            # Update status in table
            self.file_table.setItem(file_idx, 1, QTableWidgetItem("Processed"))

        # Update results table and plot
        self.update_results()
    
    def update_results(self):
        """Update the results table and plot with calculated count rates."""
        if not self.count_rates:
            return
        
        # Get all channel names
        all_channels = set()
        for file_rates in self.count_rates.values():
            all_channels.update(file_rates.keys())
        
        # Prepare data for table
        channel_stats = {}
        for channel in all_channels:
            # Get count rates for this channel across all files
            rates = [file_rates.get(channel, 0.0) for file_rates in self.count_rates.values()]
            mean_rate = np.mean(rates) / 1000.0  # Convert to kHz
            std_rate = np.std(rates) / 1000.0  # Convert to kHz
            
            # Get photon counts for this channel across all files
            photons = [file_photons.get(channel, 0) for file_photons in self.n_photons.values()]
            total_photons = np.sum(photons)
            
            # Get measurement times for all files
            times = list(self.measurement_times.values())
            total_time = np.sum(times)
            
            # Store statistics
            channel_stats[channel] = (mean_rate, std_rate, total_photons, total_time)
        
        # Update results table
        self.results_table.setRowCount(len(channel_stats))
        for row, (channel, stats) in enumerate(channel_stats.items()):
            mean_rate, std_rate, total_photons, total_time = stats
            self.results_table.setItem(row, 0, QTableWidgetItem(channel))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{mean_rate:.2f}"))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{std_rate:.2f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{total_photons:.0f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{total_time:.3f}"))
        
        # Update plot
        self.plot_widget.clear()
        
        # Create x-axis (file indices)
        x = np.arange(len(self.tttr_files))
        
        # Plot count rates for each channel
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']  # Cycle through these colors
        for i, channel in enumerate(all_channels):
            color = colors[i % len(colors)]
            
            # Get count rates for this channel across all files
            y = [file_rates.get(channel, 0.0) / 1000.0 for file_rates in self.count_rates.values()]  # Convert to kHz
            
            # Plot the data
            self.plot_widget.plot(
                x, y, 
                pen=pg.mkPen(color, width=2),
                symbol='o',
                symbolPen=color,
                symbolBrush=color,
                name=channel
            )
        
        # Set plot title with summary
        self.plot_widget.setTitle(f'Count Rates by File ({len(self.tttr_files)} files, {len(all_channels)} channels)')
    
    def save_table_as_txt(self, file_path=None):
        """Save the results table as a text file.
        
        Parameters
        ----------
        file_path : str, optional
            Path where the file should be saved. If None, a file dialog will be opened.
        """
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "No Data", "There is no data to save.")
            return
        
        # If no file_path provided, open file dialog to select save location
        if file_path is None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Table as Text File", "", "Text Files (*.txt);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
            
        try:
            with open(file_path, 'w') as f:
                # Write header
                headers = []
                for col in range(self.results_table.columnCount()):
                    headers.append(self.results_table.horizontalHeaderItem(col).text())
                f.write('\t'.join(headers) + '\n')
                
                # Write data rows
                for row in range(self.results_table.rowCount()):
                    row_data = []
                    for col in range(self.results_table.columnCount()):
                        item = self.results_table.item(row, col)
                        if item is not None:
                            row_data.append(item.text())
                        else:
                            row_data.append("")
                    f.write('\t'.join(row_data) + '\n')
                    
            QMessageBox.information(self, "Success", f"Table saved to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CountRateAnalyzer()
    window.show()
    sys.exit(app.exec())

elif __name__ == "plugin":
    window = CountRateAnalyzer()
    window.show()