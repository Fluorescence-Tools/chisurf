"""TCSPC Device Factory

Manufacturer-agnostic factory for creating TCSPC devices.
"""

import numpy as np
from qtpy.QtCore import QObject, Signal


class TCSPCDevice(QObject):
    """A manufacturer-agnostic wrapper for TCSPC devices."""

    # Signal emitted when a message is logged
    message_logged = Signal(str)

    def __init__(self, device_type="BH_SPC"):
        """Initialize the TCSPC device.

        Args:
            device_type (str): The type of TCSPC device. Supported: "BH_SPC", "PICOQUANT".
        """
        super().__init__()
        self.device_type = device_type

        # Import the appropriate device wrapper
        if self.device_type == "BH_SPC":
            from .bh_spc import BHSPCDevice
            self.device = BHSPCDevice()
        elif self.device_type == "PICOQUANT":
            try:
                from .picoquant import PicoQuantDevice as PQDevice
                self.device = PQDevice()
            except ImportError:
                raise RuntimeError("PicoQuant wrapper not available")
        elif self.device_type == "SIMULATION":
            try:
                from .simulation import SimulationDevice
                self.device = SimulationDevice()
            except ImportError:
                raise RuntimeError("Simulation wrapper not available")
        elif self.device_type == "BRICKMIC":
            try:
                from .brickmic import BrickMicDevice
                self.device = BrickMicDevice()
            except ImportError:
                raise RuntimeError("BrickMic wrapper not available")
        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

        # Connect device signals
        if self.device:
            self.device.message_logged.connect(self._forward_message)

    def _forward_message(self, message):
        """Forward messages from the device to our signal."""
        self.message_logged.emit(message)

    def log_message(self, message):
        """Log a message and emit the message_logged signal."""
        print(message)
        self.message_logged.emit(message)

    def detect_cards(self, simulation=False):
        """Detect available TCSPC cards/devices."""
        if self.device:
            return self.device.detect_cards(simulation)
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return []

    def set_active_cards(self, card_numbers):
        """Set the active cards/devices."""
        if self.device:
            return self.device.set_active_cards(card_numbers)
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return False

    def get_active_cards(self):
        """Get the active cards/devices."""
        if self.device:
            return self.device.get_active_cards()
        else:
            return []

    def initialize(self, simulation=True):
        """Initialize the TCSPC device."""
        if self.device:
            return self.device.initialize(simulation)
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return False

    def start_measurement(self):
        """Start measurement."""
        if self.device:
            return self.device.start_measurement()
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return False

    def stop_measurement(self):
        """Stop measurement."""
        if self.device:
            return self.device.stop_measurement()
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return False

    def read_fifo(self, max_words=32768):
        """Read data from FIFO."""
        if self.device:
            return self.device.read_fifo(max_words)
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return np.array([], dtype=np.uint32)

    def get_fifo_usage(self):
        """Get FIFO usage."""
        if self.device:
            return self.device.get_fifo_usage()
        else:
            self.log_message(f"Unsupported device type: {self.device_type}")
            return {}

    def close(self):
        """Close the TCSPC device."""
        if self.device:
            self.device.close()

    @property
    def initialized(self):
        """Get the initialization status from the underlying device."""
        if self.device:
            return self.device.initialized
        return False

    @property
    def simulation_params(self):
        """Proxy simulation parameters to the underlying simulation device if available."""
        if hasattr(self.device, "simulation_params"):
            return self.device.simulation_params
        raise AttributeError("Underlying device does not support simulation parameters")

    @simulation_params.setter
    def simulation_params(self, value):
        if hasattr(self.device, "simulation_params"):
            self.device.simulation_params = value
        else:
            raise AttributeError("Underlying device does not support simulation parameters")
