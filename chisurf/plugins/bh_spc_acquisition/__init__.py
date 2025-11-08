"""
BH SPC Acquisition

This plugin provides tools for acquiring single molecule data using a Becker & Hickl SPC 830 TCSPC board.
It displays fluorescence decays of user-defined channels (up to 4) and correlation curves.
Data is acquired into RAM and saved at the end of data acquisition.

Features:
- Acquisition of time-tagged time-resolved (TTTR) data from BH SPC 830
- Display of fluorescence decays for up to 4 user-defined channels
- Display of correlation curves
- Configurable acquisition time
- RAM usage monitoring
- Data saving at the end of acquisition
- Manufacturer-agnostic wrapper for future extension to other hardware (e.g., Picoquant)

The plugin is designed for single-molecule experiments where real-time monitoring of
fluorescence decays and correlation curves is essential for data quality assessment
and experimental optimization.
"""

name = "Single-Molecule:BH SPC Acquisition"

import os
import time
import psutil
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QProgressBar, QCheckBox, QGroupBox, QTabWidget,
    QMessageBox, QMenuBar, QAction, QTextEdit, QDialog
)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread
import pyqtgraph as pg

# Import the BH SPC wrapper
from .bh_spc_wrapper import (
    DLLOperationMode,
    InitStatus,
    ParID,
    SPCMError,
    BHSPC,
    minimal_spcm_ini,
    ini_file,
    TCSPCDevice,
    BHSPCCardSetupDialog
)

# Import tttrlib for correlation
import tttrlib

# Import for saving data
from chisurf.fio.ascii import save_xy
from chisurf.fio.fluorescence.fcs.kristine import write_kristine

# Define a worker thread for data acquisition
class AcquisitionThread(QThread):
    """Thread for acquiring data from the TCSPC device."""

    data_ready = pyqtSignal(object)
    acquisition_complete = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, device, duration, parent=None):
        """Initialize the acquisition thread.

        Args:
            device (TCSPCDevice): The TCSPC device to acquire data from.
            duration (float): The duration of the acquisition in seconds.
            parent (QObject): The parent object.
        """
        super().__init__(parent)
        self.device = device
        self.duration = duration
        self.running = False
        self.data = []

    def run(self):
        """Run the acquisition thread."""
        self.running = True
        self.data = []

        try:
            if not self.device.start_measurement():
                self.error.emit("Failed to start measurement")
                return

            start_time = time.monotonic()
            buf_size = 32768  # Max number of 16-bit words in a single read

            while self.running:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.duration:
                    self.device.stop_measurement()
                    break

                buf = self.device.read_fifo(buf_size)
                if buf is not None and len(buf):
                    self.data.append(buf)
                    self.data_ready.emit(buf)

                if buf is None or len(buf) < buf_size:
                    # We've read all there is to read, wait a bit
                    time.sleep(0.001)

            # Make sure to read the data that arrived after stopping
            while True:
                buf = self.device.read_fifo(buf_size)
                if buf is None or not len(buf):
                    break
                self.data.append(buf)
                self.data_ready.emit(buf)

            self.acquisition_complete.emit()

        except Exception as e:
            self.error.emit(f"Error during acquisition: {e}")
            try:
                self.device.stop_measurement()
            except:
                pass

        self.running = False

    def stop(self):
        """Stop the acquisition thread."""
        self.running = False
        self.wait()

    def get_data(self):
        """Get the acquired data.

        Returns:
            numpy.ndarray: Array of 32-bit records.
        """
        if not self.data:
            return np.array([], dtype=np.uint32)
        return np.concatenate(self.data)

# Define the main plugin widget
class BHSPCAcquisitionWidget(QWidget):
    """Main widget for the BH SPC Acquisition plugin."""

    def __init__(self, parent=None):
        """Initialize the widget."""
        super().__init__(parent)
        self.setWindowTitle("BH SPC Acquisition")
        self.resize(1000, 800)

        self.device = TCSPCDevice()
        self.acquisition_thread = None
        self.data = None
        self.decay_data = [np.zeros(4096) for _ in range(4)]  # Up to 4 channels
        self.correlation_data = None
        self.correlation_times = None
        self.correlation_amplitudes = None
        self.mean_countrate = 0.0
        self.start_time = 0.0

        self.setup_ui()
        self.setup_connections()

        # Start RAM usage timer
        self.ram_timer = QTimer(self)
        self.ram_timer.timeout.connect(self.update_ram_usage)
        self.ram_timer.start(1000)  # Update every second

        # Initialize device in simulation mode
        self.initialize_device()

    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)

        # Create menubar
        self.menubar = QMenuBar()
        main_layout.addWidget(self.menubar)

        # Create Settings menu
        settings_menu = self.menubar.addMenu("Settings")

        # Add Card Setup action
        card_setup_action = QAction("Card Setup", self)
        card_setup_action.triggered.connect(self.open_card_setup)
        settings_menu.addAction(card_setup_action)

        # Control panel
        control_panel = QGroupBox("Control Panel")
        control_layout = QGridLayout(control_panel)

        # Device initialization
        self.device_type_combo = QComboBox()
        self.device_type_combo.addItem("BH SPC 830")
        self.device_type_combo.setEnabled(False)  # Only BH SPC 830 is supported for now

        self.simulation_checkbox = QCheckBox("Simulation Mode")
        self.simulation_checkbox.setChecked(True)

        self.initialize_button = QPushButton("Initialize Device")
        self.card_setup_button = QPushButton("Hardware Setup")
        self.card_setup_button.setEnabled(False)  # Disabled until device is initialized

        control_layout.addWidget(QLabel("Device Type:"), 0, 0)
        control_layout.addWidget(self.device_type_combo, 0, 1)
        control_layout.addWidget(self.simulation_checkbox, 0, 2)
        control_layout.addWidget(self.initialize_button, 0, 3)
        control_layout.addWidget(self.card_setup_button, 0, 4)

        # Acquisition parameters
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setRange(0.1, 3600.0)
        self.duration_spinbox.setValue(10.0)
        self.duration_spinbox.setSuffix(" s")

        self.start_button = QPushButton("Start Acquisition")
        self.start_button.setEnabled(False)

        self.stop_button = QPushButton("Stop Acquisition")
        self.stop_button.setEnabled(False)

        self.save_button = QPushButton("Save Data")
        self.save_button.setEnabled(False)

        control_layout.addWidget(QLabel("Acquisition Time:"), 1, 0)
        control_layout.addWidget(self.duration_spinbox, 1, 1)
        control_layout.addWidget(self.start_button, 1, 2)
        control_layout.addWidget(self.stop_button, 1, 3)
        control_layout.addWidget(self.save_button, 1, 4)

        # Channel selection
        self.channel_spinboxes = []
        for i in range(4):
            spinbox = QSpinBox()
            spinbox.setRange(0, 15)
            spinbox.setValue(i)
            self.channel_spinboxes.append(spinbox)
            control_layout.addWidget(QLabel(f"Channel {i+1}:"), 2, i)
            control_layout.addWidget(spinbox, 3, i)

        # Status indicators
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Status: Not initialized")
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        self.ram_label = QLabel("RAM Usage: 0%")
        status_layout.addWidget(self.ram_label)

        self.fifo_label = QLabel("FIFO Usage: 0%")
        status_layout.addWidget(self.fifo_label)

        control_layout.addLayout(status_layout, 4, 0, 1, 5)

        main_layout.addWidget(control_panel)

        # Tab widget for plots
        self.tab_widget = QTabWidget()

        # Decay plots
        self.decay_tab = QWidget()
        decay_layout = QVBoxLayout(self.decay_tab)

        self.decay_plot_widget = pg.GraphicsLayoutWidget()
        self.decay_plots = []

        for i in range(4):
            plot = self.decay_plot_widget.addPlot(row=i, col=0)
            plot.setLogMode(y=True)
            plot.setLabel('left', f'Channel {i+1}')
            if i < 3:
                plot.hideAxis('bottom')
            else:
                plot.setLabel('bottom', 'Time (ns)')

            curve = plot.plot(pen=pg.mkPen(color=(i*60, 100, 255-i*60), width=2))
            self.decay_plots.append(curve)

        decay_layout.addWidget(self.decay_plot_widget)
        self.tab_widget.addTab(self.decay_tab, "Fluorescence Decays")

        # Correlation plot
        self.correlation_tab = QWidget()
        correlation_layout = QVBoxLayout(self.correlation_tab)

        self.correlation_plot_widget = pg.PlotWidget()
        self.correlation_plot_widget.setLogMode(x=True)
        self.correlation_plot_widget.setLabel('left', 'G(τ)')
        self.correlation_plot_widget.setLabel('bottom', 'τ (s)')

        self.correlation_curve = self.correlation_plot_widget.plot(pen=pg.mkPen(color=(0, 150, 150), width=2))

        correlation_layout.addWidget(self.correlation_plot_widget)
        self.tab_widget.addTab(self.correlation_tab, "Correlation Curve")

        # Logs tab
        self.logs_tab = QWidget()
        logs_layout = QVBoxLayout(self.logs_tab)

        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        logs_layout.addWidget(self.logs_text)

        self.tab_widget.addTab(self.logs_tab, "Logs")

        main_layout.addWidget(self.tab_widget)

    def setup_connections(self):
        """Set up signal-slot connections."""
        self.initialize_button.clicked.connect(self.initialize_device)
        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.save_button.clicked.connect(self.save_data)
        self.card_setup_button.clicked.connect(self.open_card_setup)

        # Connect device log messages to the log display
        self.device.message_logged.connect(self.log_message)

    def initialize_device(self):
        """Initialize the TCSPC device."""
        simulation = self.simulation_checkbox.isChecked()

        if self.device.initialized:
            self.device.close()

        # Initialize the device
        if self.device.initialize(simulation):
            # Get the active cards
            active_cards = self.device.get_active_cards()

            if active_cards:
                self.status_label.setText(f"Status: Initialized ({len(active_cards)} cards)")
                self.log_message(f"Active cards: {active_cards}")
                self.start_button.setEnabled(True)
                self.initialize_button.setText("Re-Initialize Device")
                self.card_setup_button.setEnabled(True)
            else:
                self.status_label.setText("Status: Initialized (no active cards)")
                self.log_message("No active cards detected")
                self.start_button.setEnabled(False)
                self.card_setup_button.setEnabled(True)
        else:
            self.status_label.setText("Status: Initialization failed")
            self.start_button.setEnabled(False)
            self.card_setup_button.setEnabled(False)

    def start_acquisition(self):
        """Start data acquisition."""
        if not self.device.initialized:
            QMessageBox.warning(self, "Warning", "Device not initialized")
            return

        duration = self.duration_spinbox.value()

        self.data = None
        self.decay_data = [np.zeros(4096) for _ in range(4)]
        self.correlation_data = None

        for curve in self.decay_plots:
            curve.setData([])
        self.correlation_curve.setData([])

        self.acquisition_thread = AcquisitionThread(self.device, duration, self)
        self.acquisition_thread.data_ready.connect(self.process_data)
        self.acquisition_thread.acquisition_complete.connect(self.acquisition_completed)
        self.acquisition_thread.error.connect(self.acquisition_error)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.status_label.setText("Status: Acquiring data")

        # Start progress timer
        self.start_time = time.monotonic()
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(100)  # Update every 100 ms

        # Start FIFO usage timer
        self.fifo_timer = QTimer(self)
        self.fifo_timer.timeout.connect(self.update_fifo_usage)
        self.fifo_timer.start(500)  # Update every 500 ms

        self.acquisition_thread.start()

    def stop_acquisition(self):
        """Stop data acquisition."""
        if self.acquisition_thread and self.acquisition_thread.isRunning():
            self.acquisition_thread.stop()
            self.status_label.setText("Status: Acquisition stopped")

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)

        if self.progress_timer:
            self.progress_timer.stop()

        if self.fifo_timer:
            self.fifo_timer.stop()

    def process_data(self, data):
        """Process acquired data.

        Args:
            data (numpy.ndarray): Array of 32-bit records.
        """
        if data is None or len(data) == 0:
            return

        # Extract photon records (bits 31 and 28 cleared)
        photons = data[np.bitwise_and(data, 0b1001 << 28) == 0]

        if len(photons) == 0:
            return

        # Extract microtimes (bits 16-27)
        max_12bit = (1 << 12) - 1  # 4095
        microtimes = np.bitwise_and(np.right_shift(photons, 16), max_12bit)

        # Extract routing channels (bits 8-15)
        channels = np.bitwise_and(np.right_shift(photons, 8), 0xFF)

        # Update decay histograms for each channel
        for i, spinbox in enumerate(self.channel_spinboxes):
            channel = spinbox.value()
            mask = channels == channel
            if np.any(mask):
                channel_microtimes = microtimes[mask]
                hist, _ = np.histogram(channel_microtimes, bins=4096, range=(0, 4096))
                self.decay_data[i] += hist

                # Update plot
                x = np.linspace(0, 100, 4096)  # Assuming 100 ns time range
                self.decay_plots[i].setData(x, self.decay_data[i])

        # Update correlation data (using first channel)
        if len(self.channel_spinboxes) > 0:
            channel = self.channel_spinboxes[0].value()
            mask = channels == channel
            if np.any(mask):
                # Extract macrotimes (bits 0-15)
                macrotimes = np.bitwise_and(photons[mask], 0xFFFF)

                # Compute correlation
                if len(macrotimes) > 1:
                    try:
                        # Create a correlator with appropriate settings
                        correlator = tttrlib.Correlator(
                            n_bins=20,  # Number of bins per cascade
                            n_casc=25,  # Number of cascades
                            make_fine=False  # Don't use microtime information
                        )

                        # Set the events (macrotimes and weights)
                        weights = np.ones_like(macrotimes, dtype=np.float64)
                        correlator.set_events(macrotimes, weights, macrotimes, weights)

                        # Get the correlation data
                        x = correlator.x
                        y = correlator.y

                        # Store the correlation data
                        self.correlation_times = x
                        self.correlation_amplitudes = y

                        # Calculate mean countrate (in kHz)
                        # Convert from counts per macrotime unit to kHz
                        # Assuming macrotime unit is 50 ns (20 MHz)
                        self.mean_countrate = len(macrotimes) / (macrotimes[-1] * 50e-9) / 1000

                        # Update the plot
                        self.correlation_curve.setData(x, y)
                    except Exception as e:
                        print(f"Error computing correlation: {e}")

    def acquisition_completed(self):
        """Handle acquisition completion."""
        self.data = self.acquisition_thread.get_data()
        self.status_label.setText("Status: Acquisition completed")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)

        if self.progress_timer:
            self.progress_timer.stop()

        if self.fifo_timer:
            self.fifo_timer.stop()

        self.progress_bar.setValue(100)

        # Calculate actual acquisition time
        self.acquisition_time = time.monotonic() - self.start_time

    def acquisition_error(self, error_message):
        """Handle acquisition error.

        Args:
            error_message (str): Error message.
        """
        self.status_label.setText(f"Status: Error - {error_message}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        if self.progress_timer:
            self.progress_timer.stop()

        if self.fifo_timer:
            self.fifo_timer.stop()

    def update_progress(self):
        """Update the progress bar."""
        if not self.acquisition_thread or not self.acquisition_thread.isRunning():
            return

        elapsed = time.monotonic() - self.start_time
        duration = self.duration_spinbox.value()
        progress = min(100, int(elapsed / duration * 100))
        self.progress_bar.setValue(progress)

    def update_fifo_usage(self):
        """Update the FIFO usage label."""
        if not self.device.initialized:
            return

        usage_dict = self.device.get_fifo_usage()
        if not usage_dict:
            return

        # If there's only one active card, show its usage
        if len(usage_dict) == 1:
            mod_no, usage = next(iter(usage_dict.items()))
            if usage >= 0:
                self.fifo_label.setText(f"FIFO Usage (Module {mod_no}): {usage:.1f}%")
        # If there are multiple active cards, show the average usage
        else:
            valid_usages = [u for u in usage_dict.values() if u >= 0]
            if valid_usages:
                avg_usage = sum(valid_usages) / len(valid_usages)
                self.fifo_label.setText(f"Avg FIFO Usage ({len(valid_usages)} cards): {avg_usage:.1f}%")

    def update_ram_usage(self):
        """Update the RAM usage label."""
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
        total_ram = psutil.virtual_memory().total / (1024 * 1024)  # MB
        percent = ram_usage / total_ram * 100
        self.ram_label.setText(f"RAM Usage: {ram_usage:.1f} MB ({percent:.1f}%)")

    def log_message(self, message):
        """Log a message to the logs tab.

        Args:
            message (str): The message to log.
        """
        # Add timestamp to the message
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"

        # Append the message to the logs text widget
        self.logs_text.append(formatted_message)

        # Scroll to the bottom to show the latest message
        self.logs_text.verticalScrollBar().setValue(self.logs_text.verticalScrollBar().maximum())

    def save_data(self):
        """Save the acquired data."""
        if self.data is None or len(self.data) == 0:
            QMessageBox.warning(self, "Warning", "No data to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Data", "", "Binary Files (*.bin);;All Files (*)"
        )

        if not file_path:
            return

        try:
            # Save raw data
            self.data.tofile(file_path)

            # Save decay data as NPZ
            decay_file = Path(file_path).with_suffix('.decay.npz')
            np.savez(
                decay_file,
                ch0=self.decay_data[0],
                ch1=self.decay_data[1],
                ch2=self.decay_data[2],
                ch3=self.decay_data[3],
                channels=[spinbox.value() for spinbox in self.channel_spinboxes]
            )

            # Save decay data as CSV
            for i, decay_data in enumerate(self.decay_data):
                if np.any(decay_data):  # Only save non-empty decay data
                    channel = self.channel_spinboxes[i].value()
                    csv_file = Path(file_path).with_suffix(f'.ch{channel}.csv')

                    # Create time axis (0-100 ns with 4096 points)
                    time_axis = np.linspace(0, 100, 4096)

                    # Save as CSV
                    header = f"Time (ns),Counts (Channel {channel})"
                    data_to_save = np.column_stack((time_axis, decay_data))
                    np.savetxt(csv_file, data_to_save, delimiter=',', header=header, comments='')

            # Save FCS curve as Kristine file (.cor)
            if self.correlation_times is not None and self.correlation_amplitudes is not None:
                cor_file = Path(file_path).with_suffix('.cor')

                # Use actual acquisition time if available, otherwise use the duration set in the spinbox
                acquisition_time = getattr(self, 'acquisition_time', self.duration_spinbox.value())

                # Save as Kristine file
                write_kristine(
                    filename=str(cor_file),
                    correlation_amplitude=self.correlation_amplitudes,
                    correlation_time=self.correlation_times,
                    mean_countrate=self.mean_countrate,
                    acquisition_time=acquisition_time,
                    verbose=True
                )

            success_message = f"Data saved to:\n{file_path} (raw data)\n{decay_file} (decay data)"

            # Add CSV files to success message
            for i in range(len(self.decay_data)):
                if np.any(self.decay_data[i]):
                    channel = self.channel_spinboxes[i].value()
                    csv_file = Path(file_path).with_suffix(f'.ch{channel}.csv')
                    success_message += f"\n{csv_file} (decay CSV)"

            # Add Kristine file to success message
            if self.correlation_times is not None:
                cor_file = Path(file_path).with_suffix('.cor')
                success_message += f"\n{cor_file} (FCS curve)"

            QMessageBox.information(
                self, "Success", success_message
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving data: {e}")

    def open_card_setup(self):
        """Open the card setup dialog."""
        if not self.device.initialized:
            QMessageBox.warning(self, "Warning", "Device not initialized")
            return

        # Create the card setup dialog
        dialog = BHSPCCardSetupDialog(self.device.device, self)
        result = dialog.exec_()

        # If the dialog was accepted, update the active cards
        if result == QDialog.Accepted:
            self.log_message(f"Active cards: {self.device.get_active_cards()}")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.acquisition_thread and self.acquisition_thread.isRunning():
            self.acquisition_thread.stop()

        if self.device.initialized:
            self.device.close()

        event.accept()

# Initialize the plugin when loaded
if __name__ == "plugin":
    window = BHSPCAcquisitionWidget()
    window.show()
