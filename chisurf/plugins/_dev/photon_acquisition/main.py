# Main acquisition classes for SM Acquisition plugin

import json
import copy

import logging
import time
import numpy as np

# Try to import numba for performance
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: njit does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLCDNumber,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSpinBox,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)
from qtpy.QtCore import QThread, Qt, QTimer, Signal

from chisurf import settings
import tttrlib

from .windows import DecayWindow, CorrelationWindow, CountRateWindow, MCSWindow, MacrotimeWindow
from .tcspc_devices import TCSPCDevice, BHSPCCardSetupDialog

logger = logging.getLogger(__name__)


# Numba-accelerated photon processing (10-50x faster)
@njit(cache=True)
def _process_bh_spc_records_numba(data, initial_overflow):
    """Fast numba-compiled BH SPC-130 record processing.

    This implementation mirrors tttrlib's
    RecordProcessor<BH_RECORD_TYPE_SPC130>:

    - Regular photon records (invalid == 0):
        overflow_counter += mtov
        true_nsync = mt + overflow_counter * 4096
        micro_time = 4095 - adc
        channel = rout

    - Overflow records (invalid == 1 and mtov == 1):
        cnt = low 28 bits
        overflow_counter += cnt

    Args:
        data: numpy array of uint32 BH SPC-130 records
        initial_overflow: overflow counter from previous chunk (in units of
                          4096-tick wraps)

    Returns:
        tuple: (photons, microtimes, channels, total_overflows, final_overflow)
            photons: absolute macro times (true_nsync) as uint64
            microtimes: TAC bins (uint16)
            channels: routing channels (uint8)
            total_overflows: number of macrotime wraps seen in this chunk
            final_overflow: updated overflow counter for next chunk
    """
    n_records = len(data)

    photons = np.zeros(n_records, dtype=np.uint64)
    microtimes = np.zeros(n_records, dtype=np.uint16)
    channels = np.zeros(n_records, dtype=np.uint8)

    overflow_counter = np.uint64(initial_overflow)
    total_overflows = np.uint64(0)
    photon_idx = 0

    for i in range(n_records):
        record = data[i]

        # Decode fields according to bh_spc130_record_t
        mt = record & 0xFFF                 # 12-bit macrotime within wrap
        rout = (record >> 12) & 0xF         # 4-bit routing channel
        adc = (record >> 16) & 0xFFF        # 12-bit ADC
        mark = (record >> 28) & 0x1
        gap = (record >> 29) & 0x1
        mtov = (record >> 30) & 0x1
        invalid = (record >> 31) & 0x1

        # Regular photon record: invalid == 0
        if invalid == 0:
            # Single wrap encoded in mtov bit
            overflow_counter += np.uint64(mtov)
            total_overflows += np.uint64(mtov)

            # Absolute macrotime in units of base clock ticks
            true_nsync = np.uint64(mt) + overflow_counter * np.uint64(4096)

            photons[photon_idx] = true_nsync
            # tttrlib uses 4095 - adc
            microtimes[photon_idx] = np.uint16(4095 - adc)
            channels[photon_idx] = np.uint8(rout)
            photon_idx += 1
            continue

        # Overflow record: invalid == 1 and mtov == 1
        if invalid == 1 and mtov == 1:
            # bh_overflow_t: cnt is low 28 bits
            cnt = record & 0x0FFFFFFF
            overflow_counter += np.uint64(cnt)
            total_overflows += np.uint64(cnt)
            continue

        # All other records (gaps, markers, etc.) are ignored
        # to match RecordProcessor behaviour.

    return (
        photons[:photon_idx].copy(),
        microtimes[:photon_idx].copy(),
        channels[:photon_idx].copy(),
        int(total_overflows),
        int(overflow_counter),  # overflow counter for next chunk
    )


class AcquisitionThread(QThread):
    """Thread for acquiring data from the TCSPC device."""

    data_ready = Signal(object)
    acquisition_complete = Signal()
    error = Signal(str)

    def __init__(self, device, duration, parent=None, chunk_size=32768):
        """Initialize the acquisition thread.

        Args:
            device (TCSPCDevice): The TCSPC device to acquire data from.
            duration (float): The duration of the acquisition in seconds.
            parent (QObject): The parent object.
            chunk_size (int): Number of 16-bit words to read per chunk.
        """
        super().__init__(parent)
        self.device = device
        self.duration = duration
        self.chunk_size = chunk_size
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
            buf_size = self.chunk_size  # Use configurable chunk size

            while self.running:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.duration:
                    logger.info(f"Time limit reached ({elapsed:.1f} s / {self.duration:.1f} s), stopping measurement")
                    self.device.stop_measurement()
                    break

                # Read data from device
                buf = self.device.read_fifo(buf_size)
                if buf is not None and len(buf):
                    logger.debug(f"Read {len(buf)} words from device")
                    self.data.append(buf)
                    self.data_ready.emit(buf)
                else:
                    logger.debug("No data read from device")

                # Check if we should stop before sleeping
                if not self.running:
                    break

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
        try:
            # First stop the device measurement to interrupt any blocking read operations
            if hasattr(self, 'device') and self.device is not None:
                try:
                    self.device.stop_measurement()
                except Exception as e:
                    print(f"Error stopping device measurement: {e}")

            # Then set the running flag to False
            self.running = False

            # Don't call wait() here - let the caller handle thread cleanup
            # This prevents issues when stop() is called from signal handlers
        except Exception as e:
            print(f"Error in AcquisitionThread.stop(): {e}")

    def get_data(self):
        """Get the acquired data.

        Returns:
            numpy.ndarray: Array of 32-bit records.
        """
        if not self.data:
            return np.array([], dtype=np.uint32)
        return np.concatenate(self.data)


class AcquisitionDockWidget(QDockWidget):
    """Dock widget for acquisition controls and settings."""

    def __init__(self, parent=None):
        """Initialize the acquisition dock widget."""
        super().__init__("Acquisition", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._create_contents()

    def _create_contents(self):
        """Create the dock widget contents."""
        # Control panel
        control_panel = QGroupBox("Control Panel")
        control_layout = QGridLayout(control_panel)
        control_layout.setSpacing(0)
        control_layout.setContentsMargins(0, 0, 0, 0)

        # Device initialization
        self.device_type_combo = QComboBox()
        # Prefer Simulation first for quick testing, then hardware devices
        self.device_type_combo.addItem("Simulation")
        self.device_type_combo.addItem("BH SPC 830")
        self.device_type_combo.addItem("PicoQuant")
        self.device_type_combo.addItem("BrickMic")
        self.device_type_combo.setEnabled(True)

        self.initialize_button = QToolButton()
        self.initialize_button.setText("Init Device")
        self.initialize_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.card_setup_button = QToolButton()
        self.card_setup_button.setText("Setup")
        self.card_setup_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.card_setup_button.setEnabled(False)

        # Acquisition parameters (global stop conditions)
        # Time limit (seconds). 0 disables time-based stopping.
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setRange(0.0, 36000.0)
        self.duration_spinbox.setValue(900.0)
        self.duration_spinbox.setSuffix(" s")

        # Photon-count limit in kilo-photons (kPh). 0 disables photon-based stopping.
        self.photon_limit_spinbox = QDoubleSpinBox()
        # 0 = disabled, otherwise value is in kPh (1 kPh = 1000 photons)
        self.photon_limit_spinbox.setRange(0.0, 1e6)  # Up to 1e9 photons
        self.photon_limit_spinbox.setDecimals(0)
        self.photon_limit_spinbox.setSingleStep(5.0)
        self.photon_limit_spinbox.setValue(20.0)  # Default 20 kPh
        self.photon_limit_spinbox.setSuffix(" k photons")
        self.photon_limit_spinbox.setToolTip("Photon stop limit in kilo-photons (kPh). 0 disables photon-based stopping.")

        # Internal chunk-size parameter (advanced; configured via setup dialogs)
        self.chunk_size_spinbox = QSpinBox()
        self.chunk_size_spinbox.setRange(1000, 65536)  # 1000 to 65536 photons (each photon = 32 bits = 2 words)
        self.chunk_size_spinbox.setValue(16384)  # Default 16384 photons = 32768 words (32KB)
        self.chunk_size_spinbox.setSuffix(" photons")
        self.chunk_size_spinbox.setToolTip("Number of photons to read per chunk (each photon = 32 bits)")
        # Hide from main UI; can be adjusted in device-specific setup dialogs if needed
        self.chunk_size_spinbox.setVisible(False)

        self.start_button = QToolButton()
        self.start_button.setText("Start")
        self.start_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.start_button.setEnabled(False)

        self.stop_button = QToolButton()
        self.stop_button.setText("Stop")
        self.stop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_button.setEnabled(False)

        # Channel selection
        self.channel_spinboxes = []
        for i in range(4):
            spinbox = QSpinBox()
            spinbox.setRange(0, 15)
            spinbox.setValue(i)
            self.channel_spinboxes.append(spinbox)

        # Top row: device selection and Init/Setup (same row, no empty space)
        control_layout.addWidget(QLabel("Device Type:"), 0, 0)
        control_layout.addWidget(self.device_type_combo, 0, 1, 1, 3)
        control_layout.addWidget(self.initialize_button, 0, 4)
        control_layout.addWidget(self.card_setup_button, 0, 5)

        # Next row: stop conditions (time and/or number of photons) and Start/Stop
        control_layout.addWidget(QLabel("Time [s]:"), 2, 0)
        control_layout.addWidget(self.duration_spinbox, 2, 1)
        control_layout.addWidget(QLabel("Nbr Ph [k]:"), 2, 2)
        control_layout.addWidget(self.photon_limit_spinbox, 2, 3)
        control_layout.addWidget(self.start_button, 2, 4)
        control_layout.addWidget(self.stop_button, 2, 5)

        # Status row directly below the controls
        status_layout = QHBoxLayout()
        status_layout.setSpacing(0)
        status_layout.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        # Set initial status text in progress bar
        self.progress_bar.setFormat("Status: Not initialized")
        status_layout.addWidget(self.progress_bar)

        self.count_rate_lcd = QLCDNumber()
        self.count_rate_lcd.setDigitCount(8)
        self.count_rate_lcd.setSegmentStyle(QLCDNumber.Flat)
        self.count_rate_lcd.display(0.0)
        status_layout.addWidget(self.count_rate_lcd)

        control_layout.addLayout(status_layout, 3, 0, 1, 6)

        # Output folder (shown below status)
        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(4)

        output_label = QLabel("Output folder:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Optional folder for SPC/TTTR output")
        self.output_browse_button = QToolButton()
        self.output_browse_button.setText("...")
        self.output_browse_button.clicked.connect(self._browse_output_path)

        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_browse_button)

        control_layout.addLayout(output_layout, 4, 0, 1, 4)

        # Add JSON save/load buttons
        json_layout = QHBoxLayout()
        json_layout.setContentsMargins(0, 0, 0, 0)
        json_layout.setSpacing(4)

        self.save_json_button = QToolButton()
        self.save_json_button.setText("Save Settings")
        self.save_json_button.setToolTip("Save acquisition settings to JSON file")
        self.save_json_button.clicked.connect(lambda: self.parent().save_settings_json() if hasattr(self.parent(), 'save_settings_json') else None)

        self.load_json_button = QToolButton()
        self.load_json_button.setText("Load Settings")
        self.load_json_button.setToolTip("Load acquisition settings from JSON file")
        self.load_json_button.clicked.connect(lambda: self.parent().load_settings_json() if hasattr(self.parent(), 'load_settings_json') else None)

        json_layout.addWidget(self.save_json_button)
        json_layout.addWidget(self.load_json_button)

        control_layout.addLayout(json_layout, 4, 4, 1, 2)

        # Show windows controls (compact grid layout)
        windows_group = QGroupBox("Show")
        windows_layout = QGridLayout(windows_group)
        windows_layout.setSpacing(2)
        windows_layout.setContentsMargins(2, 2, 2, 2)

        self.show_decay_checkbox = QCheckBox("Fluorescence Decays")
        self.show_decay_checkbox.setChecked(True)
        self.show_correlation_checkbox = QCheckBox("Correlation Curve")
        self.show_correlation_checkbox.setChecked(True)
        self.show_count_rate_checkbox = QCheckBox("Count Rate")
        self.show_count_rate_checkbox.setChecked(True)
        self.show_macrotime_checkbox = QCheckBox("Macrotime Plot")
        self.show_macrotime_checkbox.setChecked(False)  # Default off since it's new
        self.show_mcs_checkbox = QCheckBox("MCS Trace")
        self.show_mcs_checkbox.setChecked(False)  # Default off since it's new

        windows_layout.addWidget(self.show_decay_checkbox, 0, 0)
        windows_layout.addWidget(self.show_correlation_checkbox, 0, 1)
        windows_layout.addWidget(self.show_count_rate_checkbox, 1, 0)
        windows_layout.addWidget(self.show_macrotime_checkbox, 1, 1)
        windows_layout.addWidget(self.show_mcs_checkbox, 2, 0, 1, 2)

        control_layout.addWidget(windows_group, 4, 0, 1, 6)

        self.setWidget(control_panel)

    def _recreate_contents(self):
        """Recreate the dock widget contents if they were deleted."""
        logger.info("Recreating dock widget contents...")
        self._create_contents()

    def _browse_output_path(self):
        """Open a dialog to select an output folder for saving data files."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            "",
        )
        if path:
            self.output_path_edit.setText(path)

    # Convenience properties for accessing core UI values

    @property
    def duration(self) -> float:
        """Acquisition duration in seconds."""
        return float(self.duration_spinbox.value())

    @duration.setter
    def duration(self, value: float) -> None:
        self.duration_spinbox.setValue(float(value))

    @property
    def photon_limit(self) -> int:
        """Photon-count limit in photons (0 disables).

        The UI spinbox is expressed in kilo-photons (kPh). 0 means disabled.
        Any non-zero value less than 5 kPh is clamped to 5 kPh.
        """
        kph = float(self.photon_limit_spinbox.value())
        if 0.0 < kph < 5:
            kph = 5
        return kph * 1000

    @photon_limit.setter
    def photon_limit(self, value: int) -> None:
        # Value is given in photons; convert to kPh for the UI spinbox.
        if value <= 0:
            self.photon_limit_spinbox.setValue(0.0)
        else:
            kph = float(value) / 1000
            if 0.0 < kph < 5.0:
                kph = 5.0
            self.photon_limit_spinbox.setValue(kph)

    @property
    def chunk_size(self) -> int:
        """Chunk size in photons."""
        return int(self.chunk_size_spinbox.value())

    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        self.chunk_size_spinbox.setValue(int(value))

    @property
    def output_path(self) -> str:
        """Output folder path for SPC/TTTR data."""
        return self.output_path_edit.text()

    @output_path.setter
    def output_path(self, value: str) -> None:
        self.output_path_edit.setText(value)

    @property
    def show_decay(self) -> bool:
        return self.show_decay_checkbox.isChecked()

    @show_decay.setter
    def show_decay(self, value: bool) -> None:
        self.show_decay_checkbox.setChecked(bool(value))

    @property
    def show_correlation(self) -> bool:
        return self.show_correlation_checkbox.isChecked()

    @show_correlation.setter
    def show_correlation(self, value: bool) -> None:
        self.show_correlation_checkbox.setChecked(bool(value))

    @property
    def show_count_rate(self) -> bool:
        return self.show_count_rate_checkbox.isChecked()

    @show_count_rate.setter
    def show_count_rate(self, value: bool) -> None:
        self.show_count_rate_checkbox.setChecked(bool(value))

    @property
    def show_macrotime(self) -> bool:
        return self.show_macrotime_checkbox.isChecked()

    @show_macrotime.setter
    def show_macrotime(self, value: bool) -> None:
        self.show_macrotime_checkbox.setChecked(bool(value))

    @property
    def show_mcs(self) -> bool:
        return self.show_mcs_checkbox.isChecked()

    @show_mcs.setter
    def show_mcs(self, value: bool) -> None:
        self.show_mcs_checkbox.setChecked(bool(value))

    def closeEvent(self, event):
        """When the dock is closed, close acquisition mode."""
        # Close acquisition mode without additional confirmation
        try:
            manager = None
            parent = self.parent()
            if parent is not None and hasattr(parent, '_acquisition_manager'):
                manager = getattr(parent, '_acquisition_manager', None)
            else:
                try:
                    import chisurf
                    if hasattr(chisurf.cs, '_acquisition_manager'):
                        manager = chisurf.cs._acquisition_manager
                except Exception:
                    manager = None

            if manager is not None and hasattr(manager, 'close_acquisition_mode'):
                manager.close_acquisition_mode()
        except Exception as e:
            logger.warning(f"Error closing acquisition mode from dock closeEvent: {e}")

        event.accept()


class SMAcquisitionManager:
    """Manager for Single-Molecule Acquisition components."""

    def __init__(self, standalone_main_window=None):
        """Initialize the acquisition manager.

        Args:
            standalone_main_window: If provided, use this as main window instead of chisurf
        """
        # Check if chisurf is available
        try:
            import chisurf
            self.chisurf_available = True
            self.main_window = chisurf.cs
            logger.info("Running in chisurf mode")
        except ImportError:
            self.chisurf_available = False
            if standalone_main_window is None:
                raise RuntimeError("chisurf not available and no standalone main window provided")
            self.main_window = standalone_main_window
            logger.info("Running in standalone mode")

        # Check if acquisition manager already exists
        if hasattr(self.main_window, '_acquisition_manager'):
            logger.warning("SM Acquisition manager already exists, not creating another instance")
            # Maybe bring the existing windows to front or show a message
            if hasattr(self.main_window, '_acquisition_manager') and self.main_window._acquisition_manager:
                existing_manager = self.main_window._acquisition_manager
                # Bring existing windows to front
                try:
                    existing_manager.decay_window.raise_()
                    existing_manager.decay_window.activateWindow()
                except:
                    pass
                try:
                    existing_manager.correlation_window.raise_()
                    existing_manager.correlation_window.activateWindow()
                except:
                    pass
                try:
                    existing_manager.count_rate_window.raise_()
                    existing_manager.count_rate_window.activateWindow()
                except:
                    pass
            return

        # Initialize device
        # Start with a Simulation device so it matches the default selection
        # in the GUI's device_type_combo ("Simulation"). This will be
        # immediately updated by on_device_type_changed below if the combo
        # is changed or settings are loaded.
        self.device = TCSPCDevice("SIMULATION")
        self.data = None
        self.decay_data = [np.zeros(4096) for _ in range(4)]
        self.correlation_data = None
        self.correlation_times = None
        self.correlation_amplitudes = None
        self.mean_countrate = 0.0
        self.start_time = 0.0

        # Simulation mode (default to True)
        self.simulation_mode = True

        # Global flat photon arrays
        # Routing channel for each photon (as stored in device records)
        self.routing_channels = np.array([], dtype=np.int16)
        # Absolute macrotime (with overflow handled) for each photon
        self.absolute_macrotimes = np.array([], dtype=np.uint64)
        # Microtime (TAC bin) for each photon
        self.microtimes_all = np.array([], dtype=np.uint16)

        # Index range [start:end) of the most recently appended chunk
        self._last_chunk_start = 0
        self._last_chunk_end = 0

        # Count rate data
        self.count_rate_times = []
        self.count_rate_data = [[] for _ in range(5)]
        self.count_rate_update_counter = 0
        self.last_count_rate_time = 0.0
        # Per-plot bookkeeping of how many photons per logical channel have been seen
        self.last_macrotime_counts = [0] * 4

        # Device timing parameters
        self.macrotime_clock = 50e-9  # Default 50 ns (20 MHz), will be updated from device

        # Macrotime data for plotting
        self.macrotime_data = []  # List of macrotime values (dt differences)
        self.macrotime_times = []  # Corresponding time points for plotting

        # MCS trace data
        self.mcs_trace = None  # MCS trace histogram (bins, counts)
        self.mcs_bin_width_ms = 1.0  # Bin width in milliseconds (default 1ms)
        self.mcs_rollaround_time_ms = 1000.0  # Rollaround time in milliseconds (default 1s)
        self.mcs_intensity_accumulator = None  # Accumulated intensity data

        # Update frequency counters
        self.decay_update_counter = 0
        self.correlation_update_counter = 0
        self.macrotime_update_counter = 0

        # Create UI components
        self.decay_window = DecayWindow()
        self.correlation_window = CorrelationWindow()
        self.count_rate_window = CountRateWindow()
        self.macrotime_window = MacrotimeWindow()
        self.mcs_window = MCSWindow()
        self.acquisition_dock = AcquisitionDockWidget(self.main_window)

        # Store reference in main window
        self.main_window._acquisition_manager = self

        # Connect signals
        self.setup_connections()

        # Ensure internal device type matches GUI selection
        # (e.g. default "Simulation" or value restored from settings).
        try:
            current_type = self.acquisition_dock.device_type_combo.currentText()
            self.on_device_type_changed(current_type)
        except Exception:
            # If anything goes wrong here, keep the existing device rather
            # than failing construction.
            logger.exception("Failed to synchronize device type with GUI selection during init")

        # Read device timing parameters
        self._read_device_timing_parameters()

        # Add to main window
        if self.chisurf_available:
            # chisurf mode - use MDI area
            self.main_window.mdiarea.addSubWindow(self.decay_window)
            self.main_window.mdiarea.addSubWindow(self.correlation_window)
            self.main_window.mdiarea.addSubWindow(self.count_rate_window)
            self.main_window.mdiarea.addSubWindow(self.macrotime_window)
            self.main_window.mdiarea.addSubWindow(self.mcs_window)
            self.main_window.addDockWidget(Qt.LeftDockWidgetArea, self.acquisition_dock)
        else:
            # Standalone mode - use MDI area from standalone window
            if hasattr(self.main_window, 'mdiarea'):
                self.main_window.mdiarea.addSubWindow(self.decay_window)
                self.main_window.mdiarea.addSubWindow(self.correlation_window)
                self.main_window.mdiarea.addSubWindow(self.count_rate_window)
                self.main_window.mdiarea.addSubWindow(self.macrotime_window)
                self.main_window.mdiarea.addSubWindow(self.mcs_window)
                # For standalone, add dock as regular widget
                if hasattr(self.main_window, 'addDockWidget'):
                    self.main_window.addDockWidget(Qt.LeftDockWidgetArea, self.acquisition_dock)
                else:
                    # Fallback: add to central widget
                    self.main_window.centralWidget().layout().addWidget(self.acquisition_dock)

        # Show windows
        self.decay_window.show()
        self.correlation_window.show()
        self.count_rate_window.show()
        self.acquisition_dock.show()  # Make sure dock is visible

    def _process_photons(self, data):
        """Extract photon records, microtimes, and channels from raw data.

        Handles different data formats depending on the acquisition device type.
        BH_SPC: bits 31,28 markers; 16-27 microtimes; 8-15 channels; 0-15 macrotimes
        PicoQuant: uses different bit layout with special records and channel extraction
        """
        if len(data) == 0:
            return None, None, None

        logger.debug(f"_process_photons: processing {len(data)} words")
        if len(data) > 0:
            logger.debug(f"First 5 words: {data[:min(5, len(data))]}")

        if self.device.device_type in ("BH_SPC", "BRICKMIC"):
            # BH_SPC format: Separate photon records from overflow records
            # Overflow records have byte3 >= 0xC0 (bit 31 effectively set)
            overflow_mask = np.bitwise_and(np.right_shift(data, 24), 0xFF) >= 0xC0
            photon_mask = ~overflow_mask
            
            photons = data[photon_mask]
            overflow_records = data[overflow_mask]
            
            logger.debug(f"BH_SPC: found {len(photons)} photons, {len(overflow_records)} overflow records from {len(data)} words")

            if len(photons) == 0:
                logger.debug("No photons after BH_SPC filtering")
                return None, None, None

            # Extract microtimes (bits 16-27)
            max_12bit = (1 << 12) - 1  # 4095
            microtimes = np.bitwise_and(np.right_shift(photons, 16), max_12bit)

            # Extract routing channels (bits 8-15)
            channels = np.bitwise_and(np.right_shift(photons, 8), 0xFF)

            # Process overflow records to accumulate macrotime overflows
            overflow_count = 0
            for overflow_record in overflow_records:
                # Extract overflow count from overflow record
                # byte0: low 8 bits, byte1: mid-low 8 bits, byte2: mid-high 8 bits, byte3: marker + high 4 bits
                byte0 = overflow_record & 0xFF
                byte1 = (overflow_record >> 8) & 0xFF
                byte2 = (overflow_record >> 16) & 0xFF
                byte3 = (overflow_record >> 24) & 0xFF
                
                # Remove marker from byte3 and combine
                ov_count_high = (byte3 - 0xC0) & 0x0F
                ov_count = (ov_count_high << 24) | (byte2 << 16) | (byte1 << 8) | byte0
                overflow_count += ov_count
                
            logger.debug(f"BH_SPC: accumulated {overflow_count} macrotime overflows")

            # Store overflow count for later use in macrotime accumulation
            # TODO: Pass this to the macrotime processing code

        elif self.device.device_type == "SIMULATION":
            # Simulation format: Separate photon records from overflow records
            # Same format as BH_SPC
            overflow_mask = np.bitwise_and(np.right_shift(data, 24), 0xFF) >= 0xC0
            photon_mask = ~overflow_mask
            
            photons = data[photon_mask]
            overflow_records = data[overflow_mask]
            
            logger.debug(f"SIMULATION: found {len(photons)} photons, {len(overflow_records)} overflow records from {len(data)} words")

            if len(photons) == 0:
                logger.debug("No photons after SIMULATION filtering")
                return None, None, None

            # Extract microtimes (bits 16-27)
            max_12bit = (1 << 12) - 1  # 4095
            microtimes = np.bitwise_and(np.right_shift(photons, 16), max_12bit)

            # Extract routing channels (bits 8-15)
            channels = np.bitwise_and(np.right_shift(photons, 8), 0xFF)

            # Process overflow records to accumulate macrotime overflows
            overflow_count = 0
            for overflow_record in overflow_records:
                # Extract overflow count from overflow record
                # byte0: low 8 bits, byte1: mid-low 8 bits, byte2: mid-high 8 bits, byte3: marker + high 4 bits
                byte0 = overflow_record & 0xFF
                byte1 = (overflow_record >> 8) & 0xFF
                byte2 = (overflow_record >> 16) & 0xFF
                byte3 = (overflow_record >> 24) & 0xFF
                
                # Remove marker from byte3 and combine
                ov_count_high = (byte3 - 0xC0) & 0x0F
                ov_count = (ov_count_high << 24) | (byte2 << 16) | (byte1 << 8) | byte0
                overflow_count += ov_count
                
            logger.debug(f"SIMULATION: accumulated {overflow_count} macrotime overflows")

            # Store overflow count for later use in macrotime accumulation
            # TODO: Pass this to the macrotime processing code

        elif self.device.device_type == "PICOQUANT":
            # PicoQuant format: Filter out special records and extract timing info
            # Special records have bit 31 set
            photons = data[np.bitwise_and(data, 0x80000000) == 0]

            if len(photons) == 0:
                return None, None, None

            # For PicoQuant, we need to extract microtimes and channels differently
            # PicoQuant uses T3 mode with different bit layout
            # Extract microtimes from lower bits (implementation may need adjustment based on exact format)
            microtimes = np.bitwise_and(photons, 0xFFFF)  # Lower 16 bits for microtimes

            # Extract channels: PicoQuant uses bits 25-30 for channel info
            channels = np.bitwise_and(np.right_shift(photons, 25), 0x3F) + 1

            overflow_count = 0  # PicoQuant doesn't use BH_SPC style overflows

        else:
            logger.warning(f"Unknown device type: {self.device.device_type}, assuming BH_SPC format")
            # Fallback to BH_SPC format
            photons = data[np.bitwise_and(data, 0b1001 << 28) == 0]

            if len(photons) == 0:
                return None, None, None

            max_12bit = (1 << 12) - 1  # 4095
            microtimes = np.bitwise_and(np.right_shift(photons, 16), max_12bit)
            channels = np.bitwise_and(np.right_shift(photons, 8), 0xFF)

            overflow_count = 0  # Assume no overflows for unknown formats

        return photons, microtimes, channels, overflow_count

    def _accumulate_flat_photons(self, photons, microtimes, channels):
        """Append routing channel, absolute macrotime, and microtime to flat arrays.

        This keeps a single global time-ordered list of photons. When
        per-channel data are needed (decays, correlations, count rates), we
        filter these flat arrays by routing channel.
        """
        if photons is None or len(photons) == 0:
            return

        # Ensure numpy arrays with stable dtypes
        photons_u64 = np.asarray(photons, dtype=np.uint64)
        micro_u16 = np.asarray(microtimes, dtype=np.uint16)
        chan_i16 = np.asarray(channels, dtype=np.int16)

        # Compute insertion indices for this chunk
        start = int(self.absolute_macrotimes.size)

        if self.absolute_macrotimes.size == 0:
            # First chunk: just assign
            self.absolute_macrotimes = photons_u64
            self.microtimes_all = micro_u16
            self.routing_channels = chan_i16
        else:
            # Subsequent chunks: concatenate to preserve temporal order
            self.absolute_macrotimes = np.concatenate([self.absolute_macrotimes, photons_u64])
            self.microtimes_all = np.concatenate([self.microtimes_all, micro_u16])
            self.routing_channels = np.concatenate([self.routing_channels, chan_i16])

        end = int(self.absolute_macrotimes.size)
        self._last_chunk_start = start
        self._last_chunk_end = end

    def _update_decay_data(self, photons, microtimes, channels):
        """Update decay histograms using the current chunk.

        Photon storage is handled separately via :meth:`_accumulate_flat_photons`.
        """
        logger.debug(f"_update_decay_data: {len(photons)} photons, unique channels: {np.unique(channels)}")
        # Update decay histograms for each channel
        for i, spinbox in enumerate(self.acquisition_dock.channel_spinboxes):
            channel = spinbox.value()
            logger.debug(f"Processing channel {i}, spinbox set to {channel}")
            mask = channels == channel
            logger.debug(f"Channel {channel} has {np.sum(mask)} photons")
            if np.any(mask):
                channel_microtimes = microtimes[mask]

                # Extract macrotimes based on device type
                # photons are now absolute macrotimes from _process_photons
                channel_macrotimes = photons[mask]

                hist, _ = np.histogram(channel_microtimes, bins=4096, range=(0, 4096))
                self.decay_data[i] += hist

                # Collect macrotime differences for time series plotting
                # Store differences in **seconds** using macrotime_clock
                if len(channel_macrotimes) > 1:
                    # Compute differences between consecutive macrotimes (clock ticks)
                    diffs = np.diff(channel_macrotimes).astype(np.float64) * self.macrotime_clock
                    # All diffs in this chunk share the same wall-clock time stamp
                    current_time = time.monotonic() - self.start_time
                    self.macrotime_data.extend(diffs.tolist())
                    self.macrotime_times.extend([current_time] * len(diffs))
                # If only one photon, can't compute difference, skip

    def _compute_correlation(self, photons, channels, microtimes):
        """Compute correlation function for ALL accumulated photons of a channel.

        Uses the flat arrays `absolute_macrotimes` and `routing_channels` and
        selects the channel via the first channel spinbox.
        """
        # Guard: need global photon arrays
        if self.absolute_macrotimes is None or self.absolute_macrotimes.size == 0:
            return
        if self.routing_channels is None or self.routing_channels.size == 0:
            return

        # Determine which routing channel to correlate (first spinbox)
        try:
            if len(self.acquisition_dock.channel_spinboxes) > 0:
                routing_ch = self.acquisition_dock.channel_spinboxes[0].value()
            else:
                routing_ch = 0
        except Exception:
            routing_ch = 0

        # Select photons for this routing channel
        chan_mask = (self.routing_channels == routing_ch)
        if not np.any(chan_mask):
            return

        channel_macrotimes = self.absolute_macrotimes[chan_mask]

        # Guard: need enough photons to get a meaningful correlation
        if channel_macrotimes.size <= 10:
            return

        try:
            # Ensure numpy uint64 array (tttrlib expects this)
            macrotimes = np.asarray(channel_macrotimes, dtype=np.uint64)

            # Create correlator with standard settings
            correlator = tttrlib.Correlator(
                n_bins=9,   # Number of bins per cascade
                n_casc=15,  # Number of cascades
                make_fine=False  # Don't use microtime information
            )

            weights = np.ones_like(macrotimes, dtype=np.float64)
            correlator.set_events(macrotimes, weights, macrotimes, weights)

            # Get the correlation data (x in macrotime units)
            # Convert to milliseconds for display
            x = correlator.x * self.macrotime_clock * 1e3
            y = correlator.y

            # Calculate mean countrate (in kHz) using macrotime-based timing
            if len(macrotimes) > 1:
                total_time_span = macrotimes[-1] - macrotimes[0]
                if total_time_span > 0:
                    time_span_seconds = total_time_span * self.macrotime_clock
                    self.mean_countrate = len(macrotimes) / time_span_seconds / 1000
                else:
                    self.mean_countrate = 0.0
            else:
                self.mean_countrate = 0.0

            # Update correlation data and plot
            self.correlation_times = x
            self.correlation_amplitudes = y
            self.update_correlation_plot()
        except Exception as e:
            logger.error(f"Error computing correlation: {e}")

    def _update_count_rates(self):
        """Calculate and update count rates for plotting and display."""
        # Use macrotime for X-axis (last photon timestamp in seconds)
        if self.absolute_macrotimes is not None and self.absolute_macrotimes.size > 0:
            current_macrotime_seconds = float(self.absolute_macrotimes[-1]) * self.macrotime_clock
            self.count_rate_times.append(current_macrotime_seconds)
        else:
            self.count_rate_times.append(0.0)
            # No data yet
            for i in range(5):
                self.count_rate_data[i].append(0.0)
            return

        # Determine slice corresponding to the last processed chunk
        start = getattr(self, "_last_chunk_start", 0)
        end = getattr(self, "_last_chunk_end", 0)
        if end <= start or self.routing_channels is None or self.routing_channels.size == 0:
            # No new photons in this chunk
            for i in range(5):
                self.count_rate_data[i].append(0.0)
            self.update_count_rate_plot()
            self.update_macrotime_plot()
            return

        chunk_channels = self.routing_channels[start:end]

        # Calculate instantaneous count rates for THIS CHUNK ONLY by routing channel
        # Compare number of photons in this chunk to the time span between chunk endpoints
        for i in range(4):
            # Logical channel defined by spinboxes (fall back to 0..3)
            try:
                if len(self.acquisition_dock.channel_spinboxes) > i:
                    routing_ch = self.acquisition_dock.channel_spinboxes[i].value()
                else:
                    routing_ch = i
            except Exception:
                routing_ch = i

            photons_in_chunk = int(np.count_nonzero(chunk_channels == routing_ch))

            # Calculate time span for this chunk (macrotime axis)
            if len(self.count_rate_times) > 1 and photons_in_chunk > 0:
                time_span = self.count_rate_times[-1] - self.count_rate_times[-2]
                if time_span > 0:
                    count_rate = photons_in_chunk / time_span
                else:
                    count_rate = 0.0
            else:
                # First chunk or no new photons
                count_rate = 0.0

            self.count_rate_data[i].append(count_rate)

        # Calculate "All" count rate (sum of all channels)
        if len(self.count_rate_data[0]) > 0:
            all_count_rate = sum(
                self.count_rate_data[i][-1]
                for i in range(4)
                if len(self.count_rate_data[i]) > 0
            )
            self.count_rate_data[4].append(all_count_rate)

            # Update count rate display immediately after each chunk
            count_rate_khz = all_count_rate / 1000
            self._safe_set_count_rate_text(count_rate_khz)

        # Update count rate plot
        self.update_count_rate_plot()

        # Update macrotime plot
        self.update_macrotime_plot()

    def _read_device_timing_parameters(self):
        """Read timing parameters from the device and update instance variables."""
        try:
            if self.device.device_type == "BH_SPC" and self.device.initialized:
                # For BH SPC devices, read the macrotime clock parameter
                # The macrotime clock is typically derived from the master clock and divider
                # For now, we'll try to read MACRO_TIME_CLK parameter
                try:
                    from .photon_sources.bh_spc.wrapper import ParID
                    if hasattr(self.device.device, 'get_parameter') and self.device.active_cards:
                        mod_no = self.device.active_cards[0]  # Use first active card
                        macrotime_clock_param = self.device.device.get_parameter(mod_no, ParID.MACRO_TIME_CLK)
                        if macrotime_clock_param > 0:
                            # Convert from parameter value to actual time in seconds
                            # BH SPC macrotime clock parameter is typically in units of 50 ps (0.05 ns)
                            self.macrotime_clock = macrotime_clock_param * 50e-12  # Convert to seconds
                            logger.info(f"Read macrotime clock from BH SPC: {self.macrotime_clock*1e9:.1f} ns")
                        else:
                            logger.warning("Invalid macrotime clock parameter, using default")
                    else:
                        logger.info("Cannot read macrotime clock from BH SPC device, using default: 50 ns")
                except Exception as e:
                    logger.error(f"Error reading macrotime clock from BH SPC device: {e}")
                    logger.info("Using default macrotime clock: 50 ns")
                logger.info(f"PicoQuant macrotime clock: {self.macrotime_clock*1e12:.0f} ps ({1/(self.macrotime_clock*1e-6):.0f} MHz)")
            elif self.device.device_type == "SIMULATION":
                # For simulation, use the same timing as BH_SPC devices since the DLL
                # produces BH_SPC compatible data with realistic timing
                self.macrotime_clock = 50e-9  # 50 ns (same as BH_SPC default)
                logger.info(f"Simulation macrotime clock: {self.macrotime_clock*1e9:.1f} ns")
            else:
                logger.info(f"Unknown device type {self.device.device_type}, using default macrotime clock: {self.macrotime_clock*1e9:.1f} ns")

            # Update GUI display if available
            if self.macrotime_clock < 1e-9:  # Less than 1 ns
                display_text = f"{self.macrotime_clock*1e12:.1f} ps"
            else:  # 1 ns or more
                display_text = f"{self.macrotime_clock*1e9:.1f} ns"

            self.acquisition_dock.macrotime_clock_label.setText(f"Macrotime Clock: {display_text}")
        except Exception as e:
            logger.debug(f"Could not update macrotime clock display: {e}")

    def _calculate_mcs_trace(self):
        """Calculate MCS trace using tttrlib.

        - Stacks all photons so far into a single MCS (raw counts per bin)
        - Displays only a limited window (last rollaround_time_ms worth of bins)
        """

        # Only calculate if we have macrotime data
        if len(self.absolute_macrotimes) == 0:
            return

        try:
            # --- Get current settings (GUI overrides defaults if present) ---
            bin_width_ms = float(self.mcs_bin_width_ms)
            rollaround_time_ms = float(self.mcs_rollaround_time_ms)

            if hasattr(self, 'mcs_window') and self.mcs_window is not None \
            and hasattr(self.mcs_window, 'plot_controller'):
                pc = self.mcs_window.plot_controller
                try:
                    if hasattr(pc, 'bin_width_spinbox'):
                        bin_width_ms = float(pc.bin_width_spinbox.value())
                    if hasattr(pc, 'rollaround_spinbox'):
                        rollaround_time_ms = float(pc.rollaround_spinbox.value())
                except RuntimeError:
                    # Widgets may already be deleted – fall back to defaults
                    pass

            # --- Sanity checks ---
            if bin_width_ms <= 0:
                bin_width_ms = 1.0
            if rollaround_time_ms <= 0:
                rollaround_time_ms = bin_width_ms

            # Convert bin width from milliseconds to seconds for tttrlib
            time_window_length = bin_width_ms / 1000.0  # s

            # Convert absolute macrotimes for tttrlib
            macrotimes_uint64 = np.asarray(self.absolute_macrotimes, dtype=np.uint64)
            if macrotimes_uint64.size == 0:
                return

            # --- Compute full MCS for ALL photons so far (stacked) ---
            current_intensity_trace = tttrlib.compute_intensity_trace(
                macrotimes_uint64,
                time_window_length=time_window_length,
                macro_time_resolution=self.macrotime_clock
            )

            current_intensity_trace = np.asarray(current_intensity_trace, dtype=float)

            # Store full stacked MCS (for potential later use)
            self.mcs_intensity_accumulator = current_intensity_trace

            # --- Display only a limited time window of the MCS ---
            n_bins_total = len(current_intensity_trace)
            n_bins_view = max(1, int(rollaround_time_ms / bin_width_ms))

            if n_bins_total > n_bins_view:
                start_idx = n_bins_total - n_bins_view
                display_data = current_intensity_trace[start_idx:]
            else:
                start_idx = 0
                display_data = current_intensity_trace

            # Time axis: bin centers in ms (absolute from start of acquisition)
            bin_indices = start_idx + np.arange(len(display_data))
            bin_centers = bin_indices * bin_width_ms

            # Final trace: raw counts
            self.mcs_trace = (bin_centers, display_data)

        except Exception as e:
            logger.error(f"Error calculating MCS trace with tttrlib: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.mcs_trace = None
    def setup_connections(self):
        """Set up signal-slot connections."""
        self.acquisition_dock.device_type_combo.currentTextChanged.connect(self.on_device_type_changed)
        self.acquisition_dock.initialize_button.clicked.connect(self.initialize_device)
        self.acquisition_dock.start_button.clicked.connect(self.start_acquisition)
        self.acquisition_dock.stop_button.clicked.connect(self.stop_acquisition)
        self.acquisition_dock.card_setup_button.clicked.connect(self.open_card_setup)
        self.acquisition_dock.show_decay_checkbox.toggled.connect(self.toggle_decay_window)
        self.acquisition_dock.show_correlation_checkbox.toggled.connect(self.toggle_correlation_window)
        self.acquisition_dock.show_count_rate_checkbox.toggled.connect(self.toggle_count_rate_window)
        self.acquisition_dock.show_macrotime_checkbox.toggled.connect(self.toggle_macrotime_window)
        self.acquisition_dock.show_mcs_checkbox.toggled.connect(self.toggle_mcs_window)

        # Connect device log messages
        self.device.message_logged.connect(self._device_log_handler)

    @staticmethod
    def _safe_spinbox_value(spinbox, default_value):
        """Return spinbox value or default if widget was deleted."""
        if spinbox is None:
            return default_value

        try:
            return spinbox.value()
        except RuntimeError:
            return default_value

    def on_device_type_changed(self, device_type_text):
        """Handle device type change."""
        # Close current device if initialized
        if self.device.initialized:
            self.device.close()

        # Create new device with selected type
        if device_type_text == "BH SPC 830":
            self.device = TCSPCDevice("BH_SPC")
        elif device_type_text == "PicoQuant":
            self.device = TCSPCDevice("PICOQUANT")
        elif device_type_text == "Simulation":
            self.device = TCSPCDevice("SIMULATION")
            # Auto-initialize simulation device (instant, no hardware)
            try:
                # Use QTimer to defer initialization after UI is ready
                def do_init():
                    try:
                        if self.device.initialize(simulation=self.simulation_mode):
                            # Read timing parameters
                            self._read_device_timing_parameters()
                            # Update status
                            self._safe_set_status_text("Status: Simulation device initialized")
                            self.acquisition_dock.start_button.setEnabled(True)
                            self.acquisition_dock.card_setup_button.setEnabled(False)
                            logger.info("Simulation device auto-initialized")
                        else:
                            logger.error("Failed to auto-initialize simulation device")
                    except Exception as e:
                        logger.error(f"Failed to auto-initialize simulation device: {e}")
                        import traceback
                        traceback.print_exc()
                # Defer by 100ms to let UI settle
                QTimer.singleShot(100, do_init)
            except Exception as e:
                logger.error(f"Failed to set up auto-init: {e}")
        elif device_type_text == "BrickMic":
            self.device = TCSPCDevice("BRICKMIC")
        else:
            self.device = TCSPCDevice("BH_SPC")

        # Reconnect device log messages
        self.device.message_logged.connect(self._device_log_handler)

        # Update UI based on device type
        self.update_ui_for_device_type()

        # Reset status (only if not simulation which was already set above)
        if device_type_text != "Simulation":
            self._safe_set_status_text("Status: Device type changed")
            self.acquisition_dock.start_button.setEnabled(False)
            self.acquisition_dock.card_setup_button.setEnabled(False)

    def update_ui_for_device_type(self):
        """Update UI elements based on selected device type."""
        device_type = self.acquisition_dock.device_type_combo.currentText()

        # Card setup button is only relevant for hardware/simulation devices
        if device_type == "BH SPC 830":
            self.acquisition_dock.card_setup_button.setText("Setup")
            self.acquisition_dock.card_setup_button.setToolTip("Configure BH SPC hardware settings")
        elif device_type == "PicoQuant":
            self.acquisition_dock.card_setup_button.setText("Setup")
            self.acquisition_dock.card_setup_button.setToolTip("Configure PicoQuant device settings")
        elif device_type == "Simulation":
            self.acquisition_dock.card_setup_button.setText("Setup")
            self.acquisition_dock.card_setup_button.setToolTip("Configure simulation parameters")
        elif device_type == "BrickMic":
            self.acquisition_dock.card_setup_button.setText("Setup")
            self.acquisition_dock.card_setup_button.setToolTip("No additional BrickMic setup available")
        else:
            self.acquisition_dock.card_setup_button.setText("Setup")
            self.acquisition_dock.card_setup_button.setToolTip("Configure hardware settings")

        # Configure stop-condition semantics per device type
        time_sb = self.acquisition_dock.duration_spinbox
        ph_sb = self.acquisition_dock.photon_limit_spinbox
        csb = self.acquisition_dock.chunk_size_spinbox

        # Time control is always in seconds for all devices
        time_sb.setDecimals(1)
        time_sb.setRange(0.0, 3600.0)
        time_sb.setSingleStep(0.1)
        time_sb.setSuffix(" s")

        # Photon limit spinbox works in kilo-photons (kPh); 0 disables.
        ph_sb.setDecimals(0)
        ph_sb.setRange(0.0, 1e6)
        ph_sb.setSingleStep(5.0)
        ph_sb.setSuffix(" kPh")

        if device_type == "Simulation":
            # Use larger defaults for simulation if not already set
            if ph_sb.value() <= 0:
                ph_sb.setValue(20.0)  # 20 kPh default for simulation
            csb.setRange(1000, 1000000)
            csb.setValue(100000)
        else:
            # Hardware devices: keep more conservative chunk size
            csb.setRange(1000, 65536)
            if csb.value() <= 0:
                csb.setValue(16384)

    def initialize_device(self):
        """Initialize the TCSPC device."""
        simulation = self.simulation_mode

        if self.device.initialized:
            self.device.close()

        # Initialize the device
        if self.device.initialize(simulation):
            # Get the active cards/devices
            active_cards = self.device.get_active_cards()

            if active_cards:
                device_type = self.acquisition_dock.device_type_combo.currentText()
                if device_type == "PicoQuant":
                    self._safe_set_status_text(f"Status: Initialized ({len(active_cards)} devices)")
                else:
                    self._safe_set_status_text(f"Status: Initialized ({len(active_cards)} cards)")
                logger.info(f"Active devices: {active_cards}")
                self._safe_set_button_enabled("start_button", True)
                self.acquisition_dock.initialize_button.setText("Init Device")
                self._safe_set_button_enabled("card_setup_button", True)
            else:
                self._safe_set_status_text("Status: Initialized (no active devices)")
                logger.info("No active devices detected")
                self._safe_set_button_enabled("start_button", False)
                self._safe_set_button_enabled("card_setup_button", True)

            # Read timing parameters from the device
            self._read_device_timing_parameters()

            return True
        else:
            self._safe_set_status_text("Status: Initialization failed")
            self._safe_set_button_enabled("start_button", False)
            self._safe_set_button_enabled("card_setup_button", False)

    def start_acquisition(self):
        """Start data acquisition."""
        if not self.device.initialized:
            QMessageBox.warning(self.main_window, "Warning", "Device not initialized")
            return

        # Check if acquisition is already running
        if getattr(self, '_acquisition_in_progress', False):
            QMessageBox.information(self.main_window, "Information", "Acquisition is already running")
            return

        # Read stop conditions
        time_limit = float(self.acquisition_dock.duration)
        photon_limit = float(self.acquisition_dock.photon_limit)

        # Require at least one active stop condition
        if time_limit <= 0.0 and photon_limit <= 0.0:
            QMessageBox.warning(self.main_window, "Warning", "Set a time and/or photon stop condition before starting acquisition")
            return

        # Persist limits for progress and photon-based stopping
        self._time_limit = max(0.0, time_limit)
        self._photon_limit = max(0, int(photon_limit))

        # Thread still uses a time duration; if disabled, use a very large sentinel
        thread_duration = self._time_limit if self._time_limit > 0.0 else 1e12

        # Set flag to prevent multiple starts
        self._acquisition_in_progress = True

        chunk_size_photons = self.acquisition_dock.chunk_size_spinbox.value()
        chunk_size_words = chunk_size_photons * 2  # Convert photons to 16-bit words (each photon = 32 bits = 2 words)

        self.data = None
        self.decay_data = [np.zeros(4096) for _ in range(4)]
        self.correlation_data = None
        self.correlation_times = [None] * 4
        self.correlation_amplitudes = [None] * 4

        # Reset flat photon storage
        self.routing_channels = np.array([], dtype=np.int16)
        self.absolute_macrotimes = np.array([], dtype=np.uint64)
        self.microtimes_all = np.array([], dtype=np.uint16)

        # Reset count rate data
        self.count_rate_times = []  # Time points for count rate plot
        self.count_rate_data = [[] for _ in range(5)]  # 4 channels + 1 for "All"
        self.count_rate_update_counter = 0
        self.last_count_rate_time = time.monotonic()  # Initialize for rate calculation
        self.last_channel_counts = [0, 0, 0, 0]  # Track photons per channel for rate calculation

        # Reset macrotime data
        self.macrotime_data = []
        self.macrotime_times = []
        self.mcs_trace = None
        self.mcs_intensity_accumulator = None
        self.macrotime_update_counter = 0

        # Reset update counters
        self.decay_update_counter = 0
        self.correlation_update_counter = 0
        self.mcs_update_counter = 0

        # Reset photon counter used for photon-based stop condition
        self.total_photons = 0
        
        # Reset overflow accumulator for continuous macrotimes across chunks
        self.overflow_accumulator = 0

        # Clear plots
        for i in range(4):
            self.decay_window.decay_curves[i].setData([])
        for i in range(4):
            self.correlation_window.correlation_curves[i].setData([])

        # Clear count rate plot and mean lines
        for i in range(5):
            self.count_rate_window.count_rate_curves[i].setData([])
        for line in self.count_rate_window.mean_lines:
            self.count_rate_window.count_rate_plot_widget.removeItem(line)
        self.count_rate_window.mean_lines.clear()

        # Pass output folder and photon target to device/simulation where applicable
        output_path = (self.acquisition_dock.output_path or "").strip()
        device_type = self.acquisition_dock.device_type_combo.currentText()
        if output_path:
            # For simulation device, store as SPC output path in simulation parameters
            if device_type == "Simulation" and hasattr(self.device, "simulation_params"):
                self.device.simulation_params["spc_output_path"] = output_path
            elif device_type == "BrickMic" and hasattr(self.device, "device") and hasattr(self.device.device, "spc_output_path"):
                self.device.device.spc_output_path = output_path

        # For Simulation device, use chunk size (photons) as the per-file photon target
        if device_type == "Simulation" and hasattr(self.device, "simulation_params"):
            chunk_size_photons = int(self.acquisition_dock.chunk_size)
            target_photons = max(1, chunk_size_photons)
            self.device.simulation_params["N_ph_per_file"] = target_photons
        elif device_type == "BrickMic" and hasattr(self.device, "device") and hasattr(self.device.device, "N_ph_per_file"):
            chunk_size_photons = int(self.acquisition_dock.chunk_size)
            target_photons = max(1, chunk_size_photons)
            self.device.device.N_ph_per_file = target_photons

        self.acquisition_thread = AcquisitionThread(self.device, thread_duration, self.main_window, chunk_size_words)
        self.acquisition_thread.data_ready.connect(self.process_data)
        self.acquisition_thread.acquisition_complete.connect(self.acquisition_completed)
        self.acquisition_thread.error.connect(self.acquisition_error)

        self._safe_set_button_enabled("start_button", False)
        self._safe_set_button_enabled("stop_button", True)
        self._safe_set_status_text("Status: Acquiring data")

        # Start progress timer
        self.start_time = time.monotonic()
        self.progress_timer = QTimer(self.main_window)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(100)  # Update every 100 ms

        # Start FIFO usage timer
        self.fifo_timer = QTimer(self.main_window)
        self.fifo_timer.timeout.connect(self.update_fifo_usage)
        self.fifo_timer.start(2000)  # Update every 2 seconds instead of 500ms

        self.acquisition_thread.start()

    def process_data(self, data):
        """Slot for incoming raw data from AcquisitionThread.

        Decodes device records to photons, appends to flat arrays, updates
        decay, correlation (every N chunks), count rates, and checks
        macrotime/photon stop conditions, then triggers plot updates.
        """
        try:
            # Decode raw records to absolute macrotimes, microtimes, channels
            result = self._process_photons(data)
            if result is None:
                return

            photons, microtimes, channels, overflow_count = result
            if photons is None or len(photons) == 0:
                return

            # Append this chunk into the flat photon arrays used for
            # correlation, count rates, and macrotime-based limits.
            self._accumulate_flat_photons(photons, microtimes, channels)

            # Check which windows are enabled via dock properties
            decay_enabled = self.acquisition_dock.show_decay
            corr_enabled = self.acquisition_dock.show_correlation
            macro_enabled = self.acquisition_dock.show_macrotime
            mcs_enabled = self.acquisition_dock.show_mcs
            count_enabled = self.acquisition_dock.show_count_rate

            # Update decay histograms for this chunk (only if decay window enabled)
            if decay_enabled:
                self._update_decay_data(photons, microtimes, channels)

            # Compute correlation function (every N chunks) only if correlation window enabled
            if corr_enabled:
                self.correlation_update_counter += 1
                update_frequency = 5  # default
                if hasattr(self, "correlation_window") and hasattr(self.correlation_window, "plot_controller") and hasattr(self.correlation_window.plot_controller, "update_frequency_spinbox"):
                    update_frequency = self._safe_spinbox_value(
                        self.correlation_window.plot_controller.update_frequency_spinbox,
                        update_frequency,
                    )

                if self.correlation_update_counter >= update_frequency:
                    self._compute_correlation(photons, channels, microtimes)
                    self.correlation_update_counter = 0  # Reset counter

            # Compute MCS trace (every N chunks) only if macrotime or MCS windows are enabled
            if macro_enabled or mcs_enabled:
                self.mcs_update_counter += 1
                update_frequency = 5  # default
                if hasattr(self, "mcs_window") and hasattr(self.mcs_window, "plot_controller") and hasattr(self.mcs_window.plot_controller, "update_frequency_spinbox"):
                    update_frequency = self._safe_spinbox_value(
                        self.mcs_window.plot_controller.update_frequency_spinbox,
                        update_frequency,
                    )

                if self.mcs_update_counter >= update_frequency:
                    self._calculate_mcs_trace()
                    self.mcs_update_counter = 0  # Reset counter

            # Update count rates for this chunk only if count rate window enabled
            if count_enabled:
                self._update_count_rates()

            # Check macrotime-based time limit (for simulation)
            time_limit = getattr(self, "_time_limit", 0)
            if time_limit and time_limit > 0 and len(self.absolute_macrotimes) > 0:
                current_macrotime_seconds = self.absolute_macrotimes[-1] * self.macrotime_clock
                if current_macrotime_seconds >= time_limit:
                    # Only stop if not already stopping
                    if getattr(self, "_acquisition_in_progress", False):
                        logger.info(
                            f"Time limit reached (macrotime: {current_macrotime_seconds:.1f} s / {time_limit:.1f} s), stopping acquisition"
                        )
                        self._safe_set_status_text(
                            f"Status: Time limit reached ({current_macrotime_seconds:.1f} s)"
                        )
                        QTimer.singleShot(0, self.stop_acquisition)

            # Check photon-based stop condition
            photon_limit = getattr(self, "_photon_limit", 0)
            if photon_limit and photon_limit > 0:
                if self.total_photons >= photon_limit:
                    if getattr(self, "_acquisition_in_progress", False):
                        logger.info(
                            f"Photon limit reached ({self.total_photons} / {photon_limit}), stopping acquisition"
                        )
                        self._safe_set_status_text(
                            f"Status: Photon limit reached ({self.total_photons:,} photons)"
                        )
                        QTimer.singleShot(0, self.stop_acquisition)

            # Update plots periodically (already throttled internally where needed)
            if decay_enabled:
                self.update_decay_plot()
            if corr_enabled:
                self.update_correlation_plot()
            if count_enabled:
                self.update_count_rate_plot()
            if macro_enabled:
                self.update_macrotime_plot()
            if mcs_enabled:
                self.update_mcs_plot()

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _safe_set_button_enabled(self, button_name, enabled):
        """Safely set button enabled state, handling widget deletion."""
        try:
            button = getattr(self.acquisition_dock, button_name)
            button.setEnabled(enabled)
        except (RuntimeError, AttributeError) as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning(f"{button_name} was deleted, skipping button state update")
            else:
                raise

    def _safe_set_status_text(self, text):
        """Safely set status text in progress bar, handling widget deletion."""
        try:
            # For progress bar format, we want to show the status text
            # The default format will show percentage, but we want custom status
            self.acquisition_dock.progress_bar.setFormat(text)
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Progress bar was deleted, skipping status update")
            else:
                raise

    def stop_acquisition(self):
        """Stop data acquisition."""
        try:
            if hasattr(self, 'acquisition_thread') and self.acquisition_thread is not None:
                logger.debug(f"Stopping acquisition thread (running: {self.acquisition_thread.isRunning()})")
                if self.acquisition_thread.isRunning():
                    self.acquisition_thread.stop()
                    # Give the thread a moment to stop
                    import time
                    timeout = 0
                    while self.acquisition_thread.isRunning() and timeout < 50:  # 5 second timeout
                        time.sleep(0.1)
                        timeout += 1

                    if self.acquisition_thread.isRunning():
                        logger.warning("Acquisition thread did not stop gracefully")
                    else:
                        logger.debug("Acquisition thread stopped successfully")
                else:
                    logger.debug("Acquisition thread was not running")

            # Update UI regardless of thread state
            self._safe_set_status_text("Status: Acquisition stopped")
            self._safe_set_button_enabled("start_button", True)
            self._safe_set_button_enabled("stop_button", False)

            # Clear acquisition flag
            self._acquisition_in_progress = False

        except Exception as e:
            # If something goes wrong, at least update the UI
            logger.error(f"Error stopping acquisition: {e}")
            try:
                self._safe_set_status_text("Status: Stopped (with errors)")
                self._safe_set_button_enabled("start_button", True)
                self._safe_set_button_enabled("stop_button", False)
                # Clear acquisition flag even on error
                self._acquisition_in_progress = False
            except:
                pass

        # Stop timers safely
        try:
            if hasattr(self, 'progress_timer') and self.progress_timer is not None:
                self.progress_timer.stop()
        except Exception as e:
            logger.error(f"Error stopping progress timer: {e}")

        try:
            if hasattr(self, 'fifo_timer') and self.fifo_timer is not None:
                self.fifo_timer.stop()
        except Exception as e:
            logger.error(f"Error stopping FIFO timer: {e}")

    # Add remaining methods here...

    def update_progress(self):
        """Update the progress bar."""
        if not self.acquisition_thread or not self.acquisition_thread.isRunning():
            return

        progress = 0
        have_progress = False

        # Time-based progress (if enabled)
        time_limit = getattr(self, "_time_limit", 0.0)
        if time_limit and time_limit > 0.0:
            elapsed = time.monotonic() - self.start_time
            if time_limit > 0.0:
                time_progress = int(min(100, elapsed / time_limit * 100))
                progress = max(progress, time_progress)
                have_progress = True

        # Photon-based progress (if enabled)
        photon_limit = getattr(self, "_photon_limit", 0)
        if photon_limit and photon_limit > 0 and hasattr(self, "total_photons"):
            ph_progress = int(min(100, self.total_photons / photon_limit * 100))
            progress = max(progress, ph_progress)
            have_progress = True

        if not have_progress:
            return

        # Safe progress bar update
        try:
            self.acquisition_dock.progress_bar.setValue(progress)
            # During progress updates, show percentage
            self.acquisition_dock.progress_bar.setFormat("%p%")
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Progress bar was deleted, skipping progress update")
            else:
                raise

    def update_fifo_usage(self):
        """Log the FIFO usage."""
        # Only update FIFO usage if measurement is running
        if not hasattr(self, 'device') or not self.device.initialized:
            return

        # Check if measurement is running (for simulation and other devices)
        try:
            if hasattr(self.device, 'measurement_running'):
                if not self.device.measurement_running:
                    return
            elif hasattr(self.device, 'device') and hasattr(self.device.device, 'measurement_running'):
                # For factory devices
                if not self.device.device.measurement_running:
                    return
        except:
            pass

        usage_dict = self.device.get_fifo_usage()
        if not usage_dict:
            return

        try:
            device_type = self.acquisition_dock.device_type_combo.currentText()
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Device type combo was deleted, skipping FIFO usage update")
                return
            else:
                raise

        # If there's only one active device, log its usage
        if len(usage_dict) == 1:
            device_index, usage = next(iter(usage_dict.items()))
            if usage >= 0:
                if device_type == "PicoQuant":
                    logger.debug(f"Buffer Usage (Device {device_index}): {usage:.1f}%")
                else:
                    logger.debug(f"FIFO Usage (Module {device_index}): {usage:.1f}%")
        # If there are multiple active devices, log the average usage
        else:
            valid_usages = [u for u in usage_dict.values() if u >= 0]
            if valid_usages:
                avg_usage = sum(valid_usages) / len(valid_usages)
                if device_type == "PicoQuant":
                    logger.debug(f"Avg Buffer Usage ({len(valid_usages)} devices): {avg_usage:.1f}%")
                else:
                    logger.debug(f"Avg FIFO Usage ({len(valid_usages)} cards): {avg_usage:.1f}%")

    def _safe_set_count_rate_text(self, count_rate_khz):
        """Safely set count rate in kHz on the LCD display, handling widget deletion."""
        try:
            # Format to one decimal place
            self.acquisition_dock.count_rate_lcd.display(f"{count_rate_khz:.1f}")
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Count rate LCD was deleted, skipping count rate update")
            else:
                raise

    def update_ram_usage(self):
        """Log the RAM usage."""
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
        total_ram = psutil.virtual_memory().total / (1024 * 1024)  # MB
        percent = ram_usage / total_ram * 100
        logger.debug(f"RAM usage: {ram_usage:.1f} MB ({percent:.1f}%)")

    def open_card_setup(self):
        """Open the card setup dialog."""
        if not self.device.initialized:
            QMessageBox.warning(self.main_window, "Warning", "Device not initialized")
            return

        # Create the card setup dialog
        device_type = self.acquisition_dock.device_type_combo.currentText()
        if device_type == "Simulation":
            try:
                logger.info("Attempting to import EnhancedSimulationSetupDialog")
                from .tcspc_devices.simulation.setup_dialog import EnhancedSimulationSetupDialog
                logger.info("Successfully imported EnhancedSimulationSetupDialog")
                dialog = EnhancedSimulationSetupDialog(self.device, self.main_window)
                logger.info("Successfully created EnhancedSimulationSetupDialog instance")
            except Exception as e:
                logger.error(f"Failed to create enhanced setup dialog: {e}")
                QMessageBox.warning(self.main_window, "Setup Error", 
                                  f"Could not create simulation setup dialog: {e}")
                return
        elif device_type == "PicoQuant":
            from .photon_sources import PicoQuantSetupDialog
            dialog = PicoQuantSetupDialog(self.device, self.main_window)
        else:
            dialog = BHSPCCardSetupDialog(self.device.device, self.main_window)
            # Set the current simulation mode in the dialog (only for BH cards)
            dialog.set_simulation_mode(self.simulation_mode)
        
        result = dialog.exec_()
        logger.info(f"Dialog exec_() returned: {result}")
        if hasattr(dialog, 'windowTitle'):
            logger.info(f"Dialog title: {dialog.windowTitle()}")
        else:
            logger.info("Dialog has no windowTitle method")

        # If the dialog was accepted, update the simulation mode and active cards
        if result == QDialog.Accepted:
            if hasattr(dialog, 'get_simulation_mode'):
                self.simulation_mode = dialog.get_simulation_mode()
            elif hasattr(dialog, 'get_parameters'):
                # Simulation or PicoQuant dialog returns parameters
                new_params = dialog.get_parameters()
                if hasattr(self.device, 'simulation_params'):
                    self.device.simulation_params.update(new_params)
                    logger.info("Updated simulation parameters")
                # For PicoQuant, we could store parameters if needed
                logger.info("Updated device parameters")
            active_cards = self.device.get_active_cards()
            logger.info(f"Active devices: {active_cards}")

    def _device_log_handler(self, message):
        """Handle device log messages and forward to chisurf logging."""
        logger.info(message)

    def toggle_decay_window(self, checked):
        """Toggle the fluorescence decays window visibility."""
        if checked:
            self.decay_window.show()
            self.decay_window.raise_()  # Bring to front
        else:
            self.decay_window.hide()

    def toggle_correlation_window(self, checked):
        """Toggle the correlation curve window visibility."""
        if checked:
            self.correlation_window.show()
            self.correlation_window.raise_()  # Bring to front
        else:
            self.correlation_window.hide()

    def toggle_count_rate_window(self, checked):
        """Toggle the count rate window visibility."""
        if checked:
            self.count_rate_window.show()
            self.count_rate_window.raise_()  # Bring to front
        else:
            self.count_rate_window.hide()

    def toggle_macrotime_window(self, checked):
        """Toggle the macrotime window visibility."""
        if checked:
            self.macrotime_window.show()
            self.macrotime_window.raise_()  # Bring to front
        else:
            self.macrotime_window.hide()

    def toggle_mcs_window(self, checked):
        """Toggle the MCS window visibility."""
        if checked:
            self.mcs_window.show()
            self.mcs_window.raise_()  # Bring to front
        else:
            self.mcs_window.hide()

    def show_help(self):
        """Show the help documentation window."""
        try:
            if not hasattr(self, 'help_window') or self.help_window is None:
                self.help_window = HelpViewerWindow(self.main_window)
                if self.chisurf_available:
                    self.main_window.mdiarea.addSubWindow(self.help_window)
                else:
                    # In standalone mode, just show the window
                    pass

            self.help_window.show()
            self.help_window.raise_()  # Bring to front
            self.help_window.activateWindow()
        except Exception as e:
            logger.error(f"Error opening help window: {e}")
            QMessageBox.warning(self.main_window, "Help Error", f"Could not open help documentation:\n{e}")

    def close_acquisition_mode(self):
        """Close the acquisition mode and clean up all components."""

        # Ask for confirmation
        reply = QMessageBox.question(
            self.main_window,
            "Close Acquisition Mode",
            "Are you sure you want to close the acquisition mode?\n\nThis will stop any ongoing acquisition and close all acquisition windows.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Stop any ongoing acquisition
        if hasattr(self, 'acquisition_thread') and self.acquisition_thread and self.acquisition_thread.isRunning():
            self.acquisition_thread.stop()

        # Close device
        if self.device.initialized:
            self.device.close()

        # Stop timers
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()
        if hasattr(self, 'fifo_timer'):
            self.fifo_timer.stop()

        # Remove plot controllers from chisurf's plot options layout
        if hasattr(self.main_window, 'plotOptionsLayout'):
            # Remove decay plot controller
            if hasattr(self.decay_window, 'plot_controller') and self.decay_window.plot_controller is not None:
                try:
                    self.main_window.plotOptionsLayout.removeWidget(self.decay_window.plot_controller)
                    self.decay_window.plot_controller.hide()
                except RuntimeError as e:
                    if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                        logger.debug("Decay plot controller was already removed from layout")
                    else:
                        raise

            # Remove correlation plot controller
            if hasattr(self.correlation_window, 'plot_controller') and self.correlation_window.plot_controller is not None:
                try:
                    self.main_window.plotOptionsLayout.removeWidget(self.correlation_window.plot_controller)
                    self.correlation_window.plot_controller.hide()
                except RuntimeError as e:
                    if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                        logger.debug("Correlation plot controller was already removed from layout")
                    else:
                        raise

            # Remove count rate plot controller
            if hasattr(self.count_rate_window, 'plot_controller') and self.count_rate_window.plot_controller is not None:
                try:
                    self.main_window.plotOptionsLayout.removeWidget(self.count_rate_window.plot_controller)
                    self.count_rate_window.plot_controller.hide()
                except RuntimeError as e:
                    if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                        logger.debug("Count rate plot controller was already removed from layout")
                    else:
                        raise

        # Remove windows from MDI area
        self.main_window.mdiarea.removeSubWindow(self.decay_window)
        self.main_window.mdiarea.removeSubWindow(self.correlation_window)
        self.main_window.mdiarea.removeSubWindow(self.count_rate_window)
        self.main_window.mdiarea.removeSubWindow(self.macrotime_window)
        self.main_window.mdiarea.removeSubWindow(self.mcs_window)

        # Close and remove dock widgets
        self.main_window.removeDockWidget(self.acquisition_dock)

        # Close windows
        self.decay_window.close()
        self.correlation_window.close()
        self.count_rate_window.close()
        self.macrotime_window.close()
        self.mcs_window.close()
        self.acquisition_dock.close()

        # Clear plot controllers to default state (after windows are closed)
        self._clear_plot_controllers()

        # Clean up widget references
        try:
            if hasattr(self, 'acquisition_dock'):
                self.acquisition_dock = None
            if hasattr(self, 'decay_window'):
                self.decay_window = None
            if hasattr(self, 'correlation_window'):
                self.correlation_window = None
            if hasattr(self, 'count_rate_window'):
                self.count_rate_window = None
            if hasattr(self, 'macrotime_window'):
                self.macrotime_window = None
            if hasattr(self, 'mcs_window'):
                self.mcs_window = None
        except Exception as e:
            logger.error(f"Error cleaning up widget references: {e}")

        # Remove reference from main window
        if hasattr(self.main_window, '_acquisition_manager'):
            delattr(self.main_window, '_acquisition_manager')

        # Log the closure
        logger.info("Single-Molecule Acquisition mode closed.")

    def update_decay_plot(self):
        """Update the decay plot based on current plot controller settings."""
        logger.debug("update_decay_plot called")
        if len(self.decay_data) == 0:
            logger.debug("No decay data, returning")
            return

        # Early return if window or plot controller is not available (e.g., during cleanup)
        if not hasattr(self, 'decay_window') or self.decay_window is None:
            return
        if not hasattr(self.decay_window, 'plot_controller') or self.decay_window.plot_controller is None:
            return

        # Check update frequency
        if hasattr(self.decay_window, 'plot_controller') and hasattr(self.decay_window.plot_controller, 'update_frequency_spinbox'):
            try:
                update_frequency = self.decay_window.plot_controller.update_frequency_spinbox.value()
                self.decay_update_counter += 1
                if self.decay_update_counter < update_frequency:
                    return  # Skip update this time
                self.decay_update_counter = 0  # Reset counter
            except RuntimeError as e:
                if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                    logger.warning("Decay plot controller update_frequency_spinbox was deleted, skipping frequency check")
                else:
                    raise

        # Get log Y setting from controller
        use_log_y = False  # Default to linear scale for decays to show zeros
        try:
            if hasattr(self.decay_window, 'plot_controller') and hasattr(self.decay_window.plot_controller, 'log_y_checkbox'):
                use_log_y = self.decay_window.plot_controller.log_y_checkbox.isChecked()
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Decay plot controller log_y_checkbox was deleted, using default log_y=False")
                use_log_y = False
            else:
                raise

        # Set log mode for Y axis
        try:
            self.decay_window.decay_plot_widget.setLogMode(y=use_log_y)
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Decay plot widget was deleted, skipping log mode update")
                return
            else:
                raise

        # Get visible channels from plot controller
        visible_channels = []
        try:
            if hasattr(self.decay_window, 'plot_controller'):
                if hasattr(self.decay_window.plot_controller, 'channel_widgets'):
                    for i, (spinbox, checkbox) in enumerate(self.decay_window.plot_controller.channel_widgets):
                        try:
                            if checkbox.isChecked():
                                channel = spinbox.value()
                                if channel == -1:
                                    # Sum of all channels
                                    if len(self.decay_data) >= 3:
                                        combined = np.sum(self.decay_data[:3], axis=0)
                                        visible_channels.append((i, -1))  # Special marker for combined
                                elif channel in {8, 9, 10}:
                                    data_idx = {8: 0, 9: 1, 10: 2}[channel]
                                    visible_channels.append((i, data_idx))
                                elif 0 <= channel <= 2:
                                    data_idx = channel
                                    visible_channels.append((i, data_idx))
                        except RuntimeError as e:
                            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                                logger.warning(f"Decay curve {i} widget was deleted, skipping")
                                continue
                            else:
                                raise
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Decay plot controller was deleted, using defaults")
                visible_channels = [(0, 0), (1, 1), (2, 2)]  # Show routing 8,9,10 on curves 0,1,2
            else:
                raise

        # Update each channel curve in the single plot
        try:
            for curve_idx, data_idx in visible_channels:
                if data_idx == -1:
                    # Combined data
                    x = np.linspace(0, 100, len(combined))  # Assuming 100 ns time range
                    self.decay_window.decay_curves[curve_idx].setData(x, combined)
                    self.decay_window.decay_curves[curve_idx].show()
                elif data_idx < len(self.decay_data) and len(self.decay_data[data_idx]) > 0:
                    logger.debug(f"Setting decay data for curve {curve_idx} (channel data {data_idx}), length {len(self.decay_data[data_idx])}, total counts {np.sum(self.decay_data[data_idx])}")
                    x = np.linspace(0, 100, len(self.decay_data[data_idx]))  # Assuming 100 ns time range
                    self.decay_window.decay_curves[curve_idx].setData(x, self.decay_data[data_idx])
                    self.decay_window.decay_curves[curve_idx].show()
                else:
                    logger.debug(f"Hiding decay curve {curve_idx}, no data for channel data {data_idx}")
                    self.decay_window.decay_curves[curve_idx].hide()
            # Hide unused curves
            for i in range(4):
                if i not in [c for c, d in visible_channels]:
                    self.decay_window.decay_curves[i].hide()
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Decay plot curves were deleted, skipping plot update")
            else:
                raise

    def _clear_plot_controllers(self):
        """Reset plot controllers to their default states."""
        try:
            # Clear decay plot controller
            if hasattr(self, 'decay_window') and self.decay_window is not None and hasattr(self.decay_window, 'plot_controller'):
                try:
                    # Reset channel spinboxes and checkboxes
                    for i, (spinbox, checkbox) in enumerate(self.decay_window.plot_controller.channel_widgets):
                        spinbox.setValue([8, 9, 10, -1][i])
                        checkbox.setChecked(i < 3)
                    # Reset log Y checkbox to unchecked (linear scale for decays)
                    if hasattr(self.decay_window.plot_controller, 'log_y_checkbox'):
                        self.decay_window.plot_controller.log_y_checkbox.setChecked(False)
                    # Reset update frequency to default
                    self.decay_window.plot_controller.update_frequency_spinbox.setValue(1)
                except RuntimeError as e:
                    if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                        logger.debug("Decay plot controller widgets were deleted during cleanup")
                    else:
                        raise

            # Clear correlation plot controller
            if hasattr(self, 'correlation_window') and self.correlation_window is not None and hasattr(self.correlation_window, 'plot_controller'):
                try:
                    # Reset correlation spinboxes and checkboxes
                    for i, (spinbox_a, spinbox_b, checkbox) in enumerate(self.correlation_window.plot_controller.correlation_widgets):
                        spinbox_a.setValue(-1)
                        spinbox_b.setValue(-1)
                        checkbox.setChecked(i == 0)  # Enable first by default
                    # Reset update frequency to default
                    self.correlation_window.plot_controller.update_frequency_spinbox.setValue(5)
                except RuntimeError as e:
                    if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                        logger.debug("Correlation plot controller widgets were deleted during cleanup")
                    else:
                        raise

            # Clear count rate plot controller
            if hasattr(self.count_rate_window, 'plot_controller') and self.count_rate_window.plot_controller is not None:
                try:
                    # Reset channel spinboxes, checkboxes and "All"
                    for i, widget in enumerate(self.count_rate_window.plot_controller.channel_widgets):
                        if i < 4:
                            spinbox, checkbox, _ = widget
                            # Default: first curve plots sum (-1), others keep routing defaults and are disabled
                            default_channels = [-1, 9, 10, 8]
                            spinbox.setValue(default_channels[i])
                            checkbox.setChecked(i == 0)
                        # "All" curve is always shown, no checkbox to set
                    # Reset log Y checkbox to unchecked (linear scale)
                    if hasattr(self.count_rate_window.plot_controller, 'log_y_checkbox'):
                        self.count_rate_window.plot_controller.log_y_checkbox.setChecked(False)
                    # Reset rolling window settings
                    if hasattr(self.count_rate_window.plot_controller, 'rolling_window_checkbox'):
                        self.count_rate_window.plot_controller.rolling_window_checkbox.setChecked(True)  # Default enabled
                    if hasattr(self.count_rate_window.plot_controller, 'window_size_spinbox'):
                        self.count_rate_window.plot_controller.window_size_spinbox.setValue(500)
                    if hasattr(self.count_rate_window.plot_controller, 'binning_spinbox'):
                        self.count_rate_window.plot_controller.binning_spinbox.setValue(1)
                    # Reset update frequency to default (every chunk)
                    self.count_rate_window.plot_controller.update_frequency_spinbox.setValue(1)

                    # Clear any mean lines
                    for line in self.count_rate_window.mean_lines:
                        self.count_rate_window.count_rate_plot_widget.removeItem(line)
                    self.count_rate_window.mean_lines.clear()
                except RuntimeError as e:
                    if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                        logger.debug("Count rate plot controller widgets were deleted during cleanup")
                    else:
                        raise

            # Clear macrotime plot controller
            if hasattr(self, 'macrotime_window') and self.macrotime_window is not None and hasattr(self.macrotime_window, 'plot_controller'):
                try:
                    # Reset checkbox to checked
                    if hasattr(self.macrotime_window.plot_controller, 'show_macrotimes_checkbox'):
                        self.macrotime_window.plot_controller.show_macrotimes_checkbox.setChecked(True)
                    # Reset update frequency to default
                    self.macrotime_window.plot_controller.update_frequency_spinbox.setValue(1)
                except RuntimeError as e:
                    if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                        logger.debug("Macrotime plot controller widgets were deleted during cleanup")
                    else:
                        raise

            # Clear MCS plot controller
            if hasattr(self, 'mcs_window') and self.mcs_window is not None and hasattr(self.mcs_window, 'plot_controller'):
                try:
                    # Reset MCS controls to defaults
                    if hasattr(self.mcs_window.plot_controller, 'bin_width_spinbox'):
                        self.mcs_window.plot_controller.bin_width_spinbox.setValue(1.0)
                    if hasattr(self.mcs_window.plot_controller, 'rollaround_spinbox'):
                        self.mcs_window.plot_controller.rollaround_spinbox.setValue(1.0)
                    if hasattr(self.mcs_window.plot_controller, 'manual_y_range_checkbox'):
                        self.mcs_window.plot_controller.manual_y_range_checkbox.setChecked(False)
                    if hasattr(self.mcs_window.plot_controller, 'y_min_spinbox'):
                        self.mcs_window.plot_controller.y_min_spinbox.setValue(0.0)
                    if hasattr(self.mcs_window.plot_controller, 'y_max_spinbox'):
                        self.mcs_window.plot_controller.y_max_spinbox.setValue(1000.0)
                    # Reset update frequency to default
                    self.mcs_window.plot_controller.update_frequency_spinbox.setValue(5)
                except RuntimeError as e:
                    if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                        logger.debug("MCS plot controller widgets were deleted during cleanup")
                    else:
                        raise

            logger.debug("Plot controllers cleared to default state")

        except Exception as e:
            logger.warning(f"Error clearing plot controllers: {e}")

    def save_settings_json(self):
        """Save current acquisition settings to JSON file."""
        filename, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Acquisition Settings", "", "JSON files (*.json);;All files (*)"
        )
        if filename:
            try:
                settings = self._get_current_settings()
                json_str = json.dumps(settings, indent=2)
                with open(filename, 'w') as f:
                    f.write(json_str)
                QMessageBox.information(self.main_window, "Save Successful", "Settings saved to JSON file.")
            except Exception as e:
                QMessageBox.warning(self.main_window, "Save Error", f"Failed to save JSON file: {e}")

    def load_settings_json(self):
        """Load acquisition settings from JSON file."""
        filename, _ = QFileDialog.getOpenFileName(
            self.main_window, "Load Acquisition Settings", "", "JSON files (*.json);;All files (*)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                self._apply_settings(settings)
                QMessageBox.information(self.main_window, "Load Successful", "Settings loaded from JSON file.")
            except Exception as e:
                QMessageBox.warning(self.main_window, "Load Error", f"Failed to load JSON file: {e}")

    def _get_current_settings(self):
        """Get current acquisition settings as dictionary."""
        settings = {
            'device_type': self.acquisition_dock.device_type_combo.currentText(),
            'simulation_mode': getattr(self, 'simulation_mode', True),
            'duration': self.acquisition_dock.duration,
            'photon_limit': self.acquisition_dock.photon_limit,
            'chunk_size': self.acquisition_dock.chunk_size,
            'output_path': self.acquisition_dock.output_path,
            'channel_spinboxes': [spin.value() for spin in self.acquisition_dock.channel_spinboxes],
            'show_windows': {
                'decay': self.acquisition_dock.show_decay,
                'correlation': self.acquisition_dock.show_correlation,
                'count_rate': self.acquisition_dock.show_count_rate,
                'macrotime': self.acquisition_dock.show_macrotime,
                'mcs': self.acquisition_dock.show_mcs,
            }
        }

        # Add device-specific settings
        try:
            if hasattr(self.device, 'simulation_params') and self.device.simulation_params:
                settings['simulation_params'] = copy.deepcopy(self.device.simulation_params)
        except AttributeError:
            pass

        # Add plot controller settings
        plot_controllers = {}
        if hasattr(self, 'decay_window') and self.decay_window and hasattr(self.decay_window, 'plot_controller'):
            plot_controllers['decay'] = {
                'channels': [w[0].value() for w in self.decay_window.plot_controller.channel_widgets],
                'enabled': [w[1].isChecked() for w in self.decay_window.plot_controller.channel_widgets],
                'log_y': self.decay_window.plot_controller.log_y_checkbox.isChecked(),
                'update_frequency': self.decay_window.plot_controller.update_frequency_spinbox.value(),
            }
        if hasattr(self, 'count_rate_window') and self.count_rate_window and hasattr(self.count_rate_window, 'plot_controller'):
            plot_controllers['count_rate'] = {
                'channels': [w[0].value() if w[0] else None for w in self.count_rate_window.plot_controller.channel_widgets[:4]],
                'enabled': [w[1].isChecked() if w[1] else None for w in self.count_rate_window.plot_controller.channel_widgets[:4]],
                'log_y': self.count_rate_window.plot_controller.log_y_checkbox.isChecked(),
                'rolling_window': self.count_rate_window.plot_controller.rolling_window_checkbox.isChecked(),
                'window_size': self.count_rate_window.plot_controller.window_size_spinbox.value(),
                'binning': self.count_rate_window.plot_controller.binning_spinbox.value(),
                'update_frequency': self.count_rate_window.plot_controller.update_frequency_spinbox.value(),
            }
        if hasattr(self, 'correlation_window') and self.correlation_window and hasattr(self.correlation_window, 'plot_controller'):
            plot_controllers['correlation'] = {
                'curves': [{'ch_a': w[0].value(), 'ch_b': w[1].value(), 'enabled': w[2].isChecked()} for w in self.correlation_window.plot_controller.correlation_widgets],
                'update_frequency': self.correlation_window.plot_controller.update_frequency_spinbox.value(),
            }
        if hasattr(self, 'mcs_window') and self.mcs_window and hasattr(self.mcs_window, 'plot_controller'):
            plot_controllers['mcs'] = {
                'bin_width': self.mcs_window.plot_controller.bin_width_spinbox.value(),
                'rollaround': self.mcs_window.plot_controller.rollaround_spinbox.value(),
                'manual_y_range': self.mcs_window.plot_controller.manual_y_range_checkbox.isChecked(),
                'y_min': self.mcs_window.plot_controller.y_min_spinbox.value(),
                'y_max': self.mcs_window.plot_controller.y_max_spinbox.value(),
                'update_frequency': self.mcs_window.plot_controller.update_frequency_spinbox.value(),
            }
        if hasattr(self, 'macrotime_window') and self.macrotime_window and hasattr(self.macrotime_window, 'plot_controller'):
            plot_controllers['macrotime'] = {
                'show_macrotimes': self.macrotime_window.plot_controller.show_macrotimes_checkbox.isChecked(),
                'plot_type': self.macrotime_window.plot_controller.plot_type_combo.currentText(),
                'update_frequency': self.macrotime_window.plot_controller.update_frequency_spinbox.value(),
            }
        settings['plot_controllers'] = plot_controllers

        return settings

    def _apply_settings(self, settings):
        """Apply loaded settings to the UI and device."""
        # Update device type
        device_type = settings.get('device_type', 'Simulation')
        device_index = self.acquisition_dock.device_type_combo.findText(device_type)
        if device_index >= 0:
            self.acquisition_dock.device_type_combo.setCurrentIndex(device_index)
            self.on_device_type_changed(device_type)

        # Update simulation mode
        if 'simulation_mode' in settings:
            self.simulation_mode = settings['simulation_mode']

        # Update acquisition parameters
        if 'duration' in settings:
            self.acquisition_dock.duration = settings['duration']
        if 'photon_limit' in settings:
            self.acquisition_dock.photon_limit = settings['photon_limit']
        if 'chunk_size' in settings:
            self.acquisition_dock.chunk_size = settings['chunk_size']
        if 'output_path' in settings:
            self.acquisition_dock.output_path = settings['output_path']

        # Update channel spinboxes
        if 'channel_spinboxes' in settings:
            for i, value in enumerate(settings['channel_spinboxes']):
                if i < len(self.acquisition_dock.channel_spinboxes):
                    self.acquisition_dock.channel_spinboxes[i].setValue(value)

        # Update window visibility checkboxes
        if 'show_windows' in settings:
            show_windows = settings['show_windows']
            self.acquisition_dock.show_decay = show_windows.get('decay', True)
            self.acquisition_dock.show_correlation = show_windows.get('correlation', True)
            self.acquisition_dock.show_count_rate = show_windows.get('count_rate', True)
            self.acquisition_dock.show_macrotime = show_windows.get('macrotime', False)
            self.acquisition_dock.show_mcs = show_windows.get('mcs', False)

        # Apply simulation parameters
        if 'simulation_params' in settings:
            try:
                if hasattr(self.device, 'simulation_params'):
                    self.device.simulation_params = copy.deepcopy(settings['simulation_params'])
            except AttributeError:
                pass

        # Apply plot controller settings
        if 'plot_controllers' in settings:
            pc = settings['plot_controllers']
            if 'decay' in pc and hasattr(self, 'decay_window') and self.decay_window and hasattr(self.decay_window, 'plot_controller'):
                d = pc['decay']
                for i, w in enumerate(self.decay_window.plot_controller.channel_widgets):
                    if i < len(d['channels']):
                        w[0].setValue(d['channels'][i])
                    if i < len(d['enabled']):
                        w[1].setChecked(d['enabled'][i])
                self.decay_window.plot_controller.log_y_checkbox.setChecked(d.get('log_y', False))
                self.decay_window.plot_controller.update_frequency_spinbox.setValue(d.get('update_frequency', 1))
            if 'count_rate' in pc and hasattr(self, 'count_rate_window') and self.count_rate_window and hasattr(self.count_rate_window, 'plot_controller'):
                cr = pc['count_rate']
                for i, w in enumerate(self.count_rate_window.plot_controller.channel_widgets[:4]):
                    if w[0] and i < len(cr['channels']) and cr['channels'][i] is not None:
                        w[0].setValue(cr['channels'][i])
                    if w[1] and i < len(cr['enabled']) and cr['enabled'][i] is not None:
                        w[1].setChecked(cr['enabled'][i])
                self.count_rate_window.plot_controller.log_y_checkbox.setChecked(cr.get('log_y', False))
                self.count_rate_window.plot_controller.rolling_window_checkbox.setChecked(cr.get('rolling_window', True))
                self.count_rate_window.plot_controller.window_size_spinbox.setValue(cr.get('window_size', 500))
                self.count_rate_window.plot_controller.binning_spinbox.setValue(cr.get('binning', 1))
                self.count_rate_window.plot_controller.update_frequency_spinbox.setValue(cr.get('update_frequency', 1))
            if 'correlation' in pc and hasattr(self, 'correlation_window') and self.correlation_window and hasattr(self.correlation_window, 'plot_controller'):
                corr = pc['correlation']
                for i, w in enumerate(self.correlation_window.plot_controller.correlation_widgets):
                    if i < len(corr['curves']):
                        w[0].setValue(corr['curves'][i]['ch_a'])
                        w[1].setValue(corr['curves'][i]['ch_b'])
                        w[2].setChecked(corr['curves'][i]['enabled'])
                self.correlation_window.plot_controller.update_frequency_spinbox.setValue(corr.get('update_frequency', 5))
            if 'mcs' in pc and hasattr(self, 'mcs_window') and self.mcs_window and hasattr(self.mcs_window, 'plot_controller'):
                mcs = pc['mcs']
                self.mcs_window.plot_controller.bin_width_spinbox.setValue(mcs.get('bin_width', 1.0))
                self.mcs_window.plot_controller.rollaround_spinbox.setValue(mcs.get('rollaround', 1.0))
                self.mcs_window.plot_controller.manual_y_range_checkbox.setChecked(mcs.get('manual_y_range', False))
                self.mcs_window.plot_controller.y_min_spinbox.setValue(mcs.get('y_min', 0.0))
                self.mcs_window.plot_controller.y_max_spinbox.setValue(mcs.get('y_max', 1000.0))
                self.mcs_window.plot_controller.update_frequency_spinbox.setValue(mcs.get('update_frequency', 5))
            if 'macrotime' in pc and hasattr(self, 'macrotime_window') and self.macrotime_window and hasattr(self.macrotime_window, 'plot_controller'):
                mt = pc['macrotime']
                self.macrotime_window.plot_controller.show_macrotimes_checkbox.setChecked(mt.get('show_macrotimes', True))
                self.macrotime_window.plot_controller.plot_type_combo.setCurrentText(mt.get('plot_type', 'Time Differences (dt)'))
                self.macrotime_window.plot_controller.update_frequency_spinbox.setValue(mt.get('update_frequency', 1))

        # Update UI for device type
        self.update_ui_for_device_type()

    def update_correlation_plot(self):
        """Update the correlation plot based on current plot controller settings."""
        # Early return if window or plot controller is not available (e.g., during cleanup)
        if not hasattr(self, 'correlation_window') or self.correlation_window is None:
            return
        if not hasattr(self.correlation_window, 'plot_controller') or self.correlation_window.plot_controller is None:
            return

        # Note: Update frequency is now checked at computation stage (in process_data)
        # This just updates the plot with already-computed correlation data

        try:
            for i in range(4):
                if self.correlation_times[i] is not None and self.correlation_amplitudes[i] is not None:
                    self.correlation_window.correlation_curves[i].setData(self.correlation_times[i], self.correlation_amplitudes[i])
                    self.correlation_window.correlation_curves[i].show()
                else:
                    self.correlation_window.correlation_curves[i].hide()
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Correlation plot curves were deleted, skipping plot update")
            else:
                raise

    def update_count_rate_plot(self):
        """Update the count rate plot based on current plot controller settings."""
        if len(self.count_rate_times) == 0:
            return

        # Early return if window or plot controller is not available (e.g., during cleanup)
        if not hasattr(self, 'count_rate_window') or self.count_rate_window is None:
            return
        if not hasattr(self.count_rate_window, 'plot_controller') or self.count_rate_window.plot_controller is None:
            return

        # Get visible channels from plot controller
        visible_channels = []
        show_all = False
        try:
            if hasattr(self.count_rate_window, 'plot_controller'):
                for i, widget in enumerate(self.count_rate_window.plot_controller.channel_widgets):
                    try:
                        if i < 4:
                            spinbox, checkbox, _ = widget
                            if checkbox.isChecked():
                                channel = spinbox.value()
                                if channel == -1:
                                    data_idx = 4  # All channels
                                elif 0 <= channel <= 2:
                                    data_idx = channel
                                elif channel in {8, 9, 10}:
                                    data_idx = {8: 0, 9: 1, 10: 2}[channel]
                                else:
                                    data_idx = None
                                if data_idx is not None:
                                    visible_channels.append((i, data_idx))
                        elif i == 4:
                            # "All" curve is not always shown - shown when a spinbox is set to -1
                            pass
                    except RuntimeError as e:
                        if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                            logger.warning(f"Count rate curve {i} widget was deleted, skipping")
                            continue
                        else:
                            raise
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Count rate plot controller was deleted, using defaults")
                visible_channels = [(0, 0), (1, 1), (2, 2)]
            else:
                raise

        # Get log Y setting from controller
        use_log_y = False  # Default to linear scale
        try:
            if hasattr(self.count_rate_window, 'plot_controller') and hasattr(self.count_rate_window.plot_controller, 'log_y_checkbox'):
                use_log_y = self.count_rate_window.plot_controller.log_y_checkbox.isChecked()
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Count rate plot controller log_y_checkbox was deleted, using default log_y=False")
                use_log_y = False
            else:
                raise

        # Set log mode for Y axis
        try:
            self.count_rate_window.count_rate_plot_widget.setLogMode(y=use_log_y)
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Count rate plot widget was deleted, skipping log mode update")
                return
            else:
                raise

        # Update each visible curve
        try:
            for curve_idx, data_idx in visible_channels:
                if data_idx < len(self.count_rate_data) and len(self.count_rate_data[data_idx]) > 0:
                    # Pad time array to match data length
                    time_array = np.array(self.count_rate_times[:len(self.count_rate_data[data_idx])])
                    data_array = np.array(self.count_rate_data[data_idx])
                    self.count_rate_window.count_rate_curves[curve_idx].setData(time_array, data_array)
                    self.count_rate_window.count_rate_curves[curve_idx].show()
                else:
                    self.count_rate_window.count_rate_curves[curve_idx].hide()

            # Hide curves that are not visible
            for i in range(5):
                if i not in [c for c, d in visible_channels]:
                    self.count_rate_window.count_rate_curves[i].hide()
            
            # Force plot refresh and auto-range
            try:
                self.count_rate_window.count_rate_plot_widget.enableAutoRange()
                self.count_rate_window.count_rate_plot_widget.update()
            except Exception:
                # Swallow plot refresh errors to avoid spurious console output
                pass
                
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Count rate plot curves were deleted, skipping plot update")
            else:
                raise

    def _process_photons(self, data):
        """Process raw BH SPC records using the optimized decoder.

        Handles overflow accumulation to produce absolute macrotimes.
        Returns photons, microtimes, channels, overflow_count.

        This wraps the low-level BH-SPC decoding (numba or pure Python)
        and is called from process_data.

        Args:
            data: numpy array of uint32 BH SPC-130 records

        Returns:
            tuple: (photons, microtimes, channels, overflow_count)
                  photons: array of absolute macrotimes (uint64)
                  microtimes: array of microtimes (uint16)
                  channels: array of channel numbers (uint8)
                  overflow_count: total overflow count in this chunk
        """
        if data is None or len(data) == 0:
            return None, None, None, 0

        # Always use the numba-optimized BH SPC decoder for BH_SPC and SIMULATION.
        # If numba is not installed, njit is a no-op but the same implementation
        # is still used (just without JIT acceleration).
        if self.device.device_type in ("BH_SPC", "SIMULATION"):
            photons, microtimes, channels, overflows, new_overflow = _process_bh_spc_records_numba(
                data,
                self.overflow_accumulator,
            )
            # Update persistent overflow state for next chunk
            self.overflow_accumulator = new_overflow
            return photons, microtimes, channels, overflows

        # Non-BH devices use their existing decoding logic (PICOQUANT, etc.)
        # implemented in the base class.
        return super()._process_photons(data)

    def acquisition_completed(self):
        """Handle acquisition completion."""
        logger.info("Acquisition completed")
        self._safe_set_status_text("Status: Acquisition completed")
        self._safe_set_button_enabled("start_button", True)
        self._safe_set_button_enabled("stop_button", False)

        # Clear acquisition flag
        self._acquisition_in_progress = False

        # Save data if enabled
        self._save_data()

    def acquisition_error(self, error_msg):
        """Handle acquisition error."""
        logger.error(f"Acquisition error: {error_msg}")
        self._safe_set_status_text(f"Status: Error - {error_msg}")
        self._safe_set_button_enabled("start_button", True)
        self._safe_set_button_enabled("stop_button", False)

        # Clear acquisition flag
        self._acquisition_in_progress = False

    def _save_data(self):
        """Save acquired data."""
        # No automatic saving - user can specify output folder for SPC files
        pass

    def update_macrotime_plot(self):
        """Update the macrotime plot."""
        if not hasattr(self, 'macrotime_window') or self.macrotime_window is None:
            return

        # Check update frequency
        if hasattr(self.macrotime_window, 'plot_controller') and hasattr(self.macrotime_window.plot_controller, 'update_frequency_spinbox'):
            try:
                update_frequency = self.macrotime_window.plot_controller.update_frequency_spinbox.value()
                self.macrotime_update_counter += 1
                if self.macrotime_update_counter < update_frequency:
                    return
                self.macrotime_update_counter = 0
            except RuntimeError:
                pass

        # Update MCS trace
        self._calculate_mcs_trace()

        # Update MCS trace display (in separate window)
        self.update_mcs_plot()

        # Check if macrotime plot should be shown
        show_macrotimes = True
        try:
            if hasattr(self.macrotime_window, 'plot_controller') and hasattr(self.macrotime_window.plot_controller, 'show_macrotimes_checkbox'):
                show_macrotimes = self.macrotime_window.plot_controller.show_macrotimes_checkbox.isChecked()
        except RuntimeError:
            pass

        try:
            if show_macrotimes and len(self.macrotime_data) > 0:
                # Plot macrotime differences vs time (dt already in seconds)
                times = np.array(self.macrotime_times[-len(self.macrotime_data):])
                dts = np.array(self.macrotime_data)
                self.macrotime_window.macrotime_curve.setData(times, dts)
                self.macrotime_window.macrotime_plot_widget.setLabel('left', 'Macrotime Difference (s)')
                self.macrotime_window.macrotime_plot_widget.setLabel('bottom', 'Time (s)')

            self.macrotime_window.macrotime_curve.show()
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("Macrotime plot curve was deleted, skipping plot update")
            else:
                raise

    def update_mcs_plot(self):
        """Update the MCS trace plot."""
        if self.mcs_trace is None:
            return

        # Early return if window is not available (e.g., during cleanup)
        if not hasattr(self, 'mcs_window') or self.mcs_window is None:
            return

        # Update MCS trace display if available
        try:
            if self.mcs_trace is not None:
                bin_centers, hist_counts = self.mcs_trace
                self.mcs_window.mcs_curve.setData(bin_centers, hist_counts)
                self.mcs_window.mcs_curve.show()

                # Apply Y range if manual range is enabled
                if hasattr(self.mcs_window, 'plot_controller') and hasattr(self.mcs_window.plot_controller, 'manual_y_range_checkbox'):
                    if self.mcs_window.plot_controller.manual_y_range_checkbox.isChecked():
                        y_min = self.mcs_window.plot_controller.y_min_spinbox.value()
                        y_max = self.mcs_window.plot_controller.y_max_spinbox.value()
                        self.mcs_window.mcs_plot_widget.setYRange(y_min, y_max)
                    else:
                        self.mcs_window.mcs_plot_widget.enableAutoRange(axis=self.mcs_window.mcs_plot_widget.getViewBox().YAxis)
            else:
                self.mcs_window.mcs_curve.hide()
        except RuntimeError as e:
            if "wrapped C/C++ object" in str(e) and "has been deleted" in str(e):
                logger.warning("MCS plot curve was deleted, skipping MCS plot update")
            else:
                raise

    def _update_count_rates(self):
        """Update count rates for the current update interval."""
        current_time = time.monotonic()
        time_interval = current_time - self.last_count_rate_time

        if time_interval > 0 and len(self.routing_channels) > 0:
            # Calculate count rates for each channel
            for i in range(3):
                current_total = np.count_nonzero(self.routing_channels == [8, 9, 10][i])
                photons_in_interval = current_total - self.last_channel_counts[i]
                count_rate = photons_in_interval / time_interval
                self.count_rate_data[i].append(count_rate)
                self.last_channel_counts[i] = current_total

            # 4th channel (no data, set to 0)
            self.count_rate_data[3].append(0.0)

            # All channels
            current_all_total = len(self.routing_channels)
            photons_all_in_interval = current_all_total - self.last_channel_counts[3]
            all_count_rate = photons_all_in_interval / time_interval
            self.count_rate_data[4].append(all_count_rate)
            self.last_channel_counts[3] = current_all_total

            # Update time
            self.count_rate_times.append(current_time - self.start_time)
            self.last_count_rate_time = current_time

    def _compute_correlation(self, photons, channels, microtimes):
        """Compute correlation functions for enabled curves."""
        try:
            import tttrlib
        except ImportError:
            logger.warning("tttrlib not available, skipping correlation computation")
            return

        # Get correlation settings from controller
        enabled_correlations = []
        try:
            if hasattr(self.correlation_window, 'plot_controller') and hasattr(self.correlation_window.plot_controller, 'correlation_widgets'):
                for i, (spinbox_a, spinbox_b, checkbox) in enumerate(self.correlation_window.plot_controller.correlation_widgets):
                    if checkbox.isChecked():
                        ch_a = spinbox_a.value()
                        ch_b = spinbox_b.value()
                        enabled_correlations.append((i, ch_a, ch_b))
        except RuntimeError:
            pass

        for curve_idx, ch_a, ch_b in enabled_correlations:
            try:
                # Filter photons for channel A
                if ch_a == -1:
                    photons_a = self.absolute_macrotimes
                else:
                    mask_a = self.routing_channels == ch_a
                    if not np.any(mask_a):
                        continue
                    photons_a = self.absolute_macrotimes[mask_a]

                # Filter photons for channel B
                if ch_b == -1:
                    photons_b = self.absolute_macrotimes
                else:
                    mask_b = self.routing_channels == ch_b
                    if not np.any(mask_b):
                        continue
                    photons_b = self.absolute_macrotimes[mask_b]

                # Compute correlation
                correlator = tttrlib.Correlator(n_bins=9, n_casc=15, make_fine=False)
                weights_a = np.ones_like(photons_a, dtype=np.float64)
                weights_b = np.ones_like(photons_b, dtype=np.float64)
                correlator.set_events(photons_a, weights_a, photons_b, weights_b)

                # Get correlation data
                x = correlator.x * self.macrotime_clock * 1e3  # Convert to milliseconds
                y = correlator.y

                # Store results
                self.correlation_times[curve_idx] = x
                self.correlation_amplitudes[curve_idx] = y

            except Exception as e:
                logger.warning(f"Error computing correlation for curve {curve_idx}: {e}")

    def add_count_rate_mean_line(self, curve_index, mean_value):
        """Add a horizontal mean line to the count rate plot."""
        if not hasattr(self, 'count_rate_window') or self.count_rate_window is None:
            return

        try:
            # Create horizontal line at mean value
            import pyqtgraph as pg
            mean_line = pg.InfiniteLine(pos=mean_value, angle=0, pen=pg.mkPen('r', width=2, style=Qt.DashLine))

            # Add label
            curve_name = f"Curve{curve_index}" if curve_index < 4 else "All"
            mean_line.label = pg.InfLineLabel(mean_line, f"{curve_name} Mean: {mean_value:.1f}", position=0.1, anchor=(1, 1))

            # Add to plot and store reference
            self.count_rate_window.count_rate_plot_widget.addItem(mean_line)
            self.count_rate_window.mean_lines.append(mean_line)

        except Exception as e:
            logger.warning(f"Could not add mean line for curve {curve_index}: {e}")

    # More methods would be added here...


